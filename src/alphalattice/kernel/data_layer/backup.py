"""Online machine-root backups and verified restoration through the data-layer owners.

Generations hold index, log and catalog snapshots in that order. A catalog can be newer than
its log: restore verifies the saved log head, then uses StorePort's ordinary startup repair.
The repository's shared files/ pool keeps admitted immutable data and blobs once. Transport,
staging, process locks, keychains and installed extensions are not backup inputs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Final

from alphalattice.kernel import storage
from alphalattice.kernel.clock import utc_now
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import CHAIN_VERSION, LINKS, fold_segments
from alphalattice.kernel.data_layer.governance import COPYING
from alphalattice.kernel.data_layer.history_runtime import host_duckdb
from alphalattice.kernel.data_layer.log import HostLog, verify_whole
from alphalattice.kernel.data_layer.root import (
    FOLDERS,
    bound_reads,
    catalog,
    load_extensions,
    sql_path,
    sqlite_read_only_uri,
)
from alphalattice.kernel.data_layer.store import StorePort
from alphalattice.kernel.identity import integrity_blake3, integrity_sha256
from alphalattice.kernel.shared_kernel.environment import duckdb_thread_config

DATABASES: Final = ("log.sqlite", "catalog.sqlite")
DIGESTS: Final = {
    "blake3:v1": integrity_blake3,
    "blake3": integrity_blake3,
    "sha256": integrity_sha256,
}


def _valid_digest(value: Any) -> bool:
    """A versioned backup seal, or a recognized tagged digest retained from its owner."""
    if not isinstance(value, str):
        return False
    prefix, _, value = value.rpartition(":")
    return prefix in DIGESTS and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _file_digest(path: Path, prefix: str) -> str:
    """Stream one immutable file once at this boundary; never load a large body into memory."""
    state = DIGESTS[prefix]()
    with _real(path).open("rb") as source:
        while block := source.read(1024 * 1024):
            state.update(block)
    return prefix + ":" + state.hexdigest()


def _refuse(code: str, message: str) -> DataError:
    """A named data-layer refusal, without any copied content in its diagnostic."""
    return DataError(message, code=f"data_layer.{code}")


def _real(path: Path) -> Path:
    """Refuse a file or directory redirected away from its admitted path."""
    if path.resolve() != path.absolute():
        raise _refuse("backup_path_aliased", f"{path} is aliased; use its admitted real path")
    return path


def _handle(value: Any) -> bool:
    """Whether a portable manifest handle names only admitted data or blob storage."""
    if not isinstance(value, str) or "\\" in value or ":" in value or "\0" in value:
        return False
    parts = PurePosixPath(value).parts
    return bool(
        len(parts) > 1
        and parts[0] in {"data", "blobs"}
        and ".." not in parts
        and PurePosixPath(value).as_posix() == value
    )


def read_manifest(generation: Path) -> dict[str, Any]:
    """Read a portable generation; malformed paths or a missing saved head refuse by name."""
    path = _real(generation.resolve() / "manifest.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        head, files = value["head"], value["files"]
        valid_head = (
            type(head["seq"]) is int
            and head["seq"] >= 0
            and type(head["chain_version"]) is int
            and head["chain_version"] in LINKS
            and isinstance(head["head"], str)
            and (
                head["head"] == ""
                if head["seq"] == 0
                else bool(re.fullmatch(r"blake3:[0-9a-f]{64}", head["head"]))
            )
        )
        paths = [entry["path"] for entry in files]
        valid_files = all(
            _handle(entry["path"])
            and type(entry["size"]) is int
            and entry["size"] >= 0
            and (
                (entry.get("kind") is None and _valid_digest(entry["digest"]))
                or (
                    entry.get("kind") == "array"
                    and entry["path"].startswith("blobs/")
                    and isinstance(entry["digest"], str)
                    and bool(re.fullmatch(r"[0-9a-f]{64}", entry["digest"]))
                )
            )
            for entry in files
        ) and len(set(paths)) == len(paths)
        if (
            type(value["version"]) is not int
            or value["version"] != 2
            or value["databases"] != list(DATABASES)
            or not valid_head
            or not valid_files
        ):
            raise ValueError("format, saved head or file set")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise _refuse(
            "backup_manifest_invalid",
            f"machine backup manifest {path} is invalid; use a complete generation",
        ) from error
    return value


def _verify_file(path: Path, entry: dict[str, Any], read: Any = None) -> None:
    """Verify an immutable body by the recorded owner's rule."""
    if _real(path).stat().st_size != entry["size"]:
        raise _refuse(
            "backup_file_changed", f"{path} differs from its sealed bytes; use an intact generation"
        )
    if entry.get("kind") == "array":
        if read is None:
            raise _refuse(
                "backup_manifest_invalid",
                "array backup lacks its saved owner; use a complete generation",
            )
        import alphalattice.kernel.data_layer.arrays as arrays  # lazy: array bodies only

        arrays.verify(path.parent.parent, read, entry["digest"], path.name)
    elif _file_digest(path, entry["digest"].rpartition(":")[0]) != entry["digest"]:
        raise _refuse(
            "backup_file_changed", f"{path} differs from its sealed bytes; use an intact generation"
        )


def copy_files(
    root: Path, pool: Path, files: list[dict[str, Any]], *, read: Any = None
) -> tuple[str, ...]:
    """Copy only new immutable files and verify every reused or copied body against its owner."""
    copied = []
    for entry in files:
        handle, kind = entry["path"], entry.get("kind")
        valid = (
            _valid_digest(entry["digest"])
            if kind is None
            else (
                kind == "array"
                and isinstance(entry["digest"], str)
                and bool(re.fullmatch(r"[0-9a-f]{64}", entry["digest"]))
                and handle.startswith("blobs/")
            )
        )
        if not _handle(handle) or not valid:
            raise _refuse(
                "backup_manifest_invalid", f"machine backup manifest handle {handle!r} is invalid"
            )
        source, target = _real(root / handle), _real(pool / handle)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(handle)
        _verify_file(target, entry, read)
    return tuple(copied)


def _snapshot(source: Path, target: Path) -> None:
    """Copy one committed SQLite state. A published catalog is one self-contained file."""
    with (
        closing(sqlite3.connect(sqlite_read_only_uri(_real(source)), uri=True)) as src,
        closing(sqlite3.connect(_real(target))) as dst,
    ):
        src.backup(dst, pages=128)
        if target.name == "catalog.sqlite":
            mode = dst.execute("PRAGMA journal_mode = DELETE").fetchone()
            if mode != ("delete",):
                raise _refuse("backup_unreadable", f"could not finalize catalog snapshot {target}")


def _head(path: Path) -> dict[str, Any]:
    """Verify the entire copied chain and return its actual head and final sequence."""
    with closing(sqlite3.connect(sqlite_read_only_uri(_real(path)), uri=True)) as con:
        con.row_factory = sqlite3.Row
        stored = con.execute(
            "SELECT seq, head, chain_version FROM chain_head WHERE id = 1"
        ).fetchone()
        last = con.execute("SELECT coalesce(max(seq), 0) FROM event").fetchone()[0]
        try:
            valid = verify_whole(con) and (stored[0] if stored else 0) == last
        except KeyError:
            valid = False
        if not valid:
            raise _refuse(
                "backup_chain_broken",
                f"{path} does not verify its whole chain; use an untampered backup",
            )
        return dict(
            zip(
                ("seq", "head", "chain_version"),
                tuple(stored) if stored else (0, "", CHAIN_VERSION),
                strict=True,
            )
        )


def _consistent(machine: Path) -> None:
    """A copied log must never name a snapshot newer than its copied catalog's newest: the
    catalog would be behind the log. A snapshot older than those the catalog keeps, expired by
    StorePort.maintain under the snapshot rule, is not missing."""
    with closing(sqlite3.connect(sqlite_read_only_uri(machine / "log.sqlite"), uri=True)) as con:
        con.execute(
            "ATTACH DATABASE ? AS catalog", (sqlite_read_only_uri(machine / "catalog.sqlite"),)
        )
        absent = con.execute(
            "SELECT a.catalog_snapshot FROM admission a WHERE a.catalog_snapshot >"
            " (SELECT max(snapshot_id) FROM catalog.ducklake_snapshot) LIMIT 1"
        ).fetchone()
        if absent:
            raise _refuse(
                "backup_catalog_behind_log",
                f"{machine} lacks catalog snapshot {absent[0]} named by its log; "
                "take a new ordered backup",
            )


def _walk(folder: Path):
    """Read admitted files; refuse an unreadable folder or aliased child."""
    for parent, directories, files in folder.walk(on_error=_unreadable):
        _real(parent)
        for name in directories:
            _real(parent / name)
        for name in files:
            yield _real(parent / name)


def _unreadable(error: OSError) -> None:
    """Name a source folder which could not be enumerated."""
    raise _refuse(
        "backup_unreadable",
        f"{error.filename} cannot be read; restore access and take a new backup",
    ) from error


_REFERENCED: Final = (
    "SELECT s.path,s.path_is_relative,t.path,t.path_is_relative,"
    "f.path,f.path_is_relative FROM ducklake_{table} f "
    "LEFT JOIN ducklake_table t ON t.table_id=f.table_id "
    "AND t.begin_snapshot<=f.begin_snapshot "
    "AND (t.end_snapshot IS NULL OR t.end_snapshot>f.begin_snapshot) "
    "LEFT JOIN ducklake_schema s ON s.schema_id=t.schema_id "
    "AND s.begin_snapshot<=f.begin_snapshot "
    "AND (s.end_snapshot IS NULL OR s.end_snapshot>f.begin_snapshot)"
)


def _catalog_paths(catalog: Path) -> set[str]:
    """Resolve the copied catalog's paths at each file's creation snapshot, including history.

    A refusal is raised only once the connection and its cursors are closed, so it leaves no
    open handle on the copied catalog (Windows cannot remove an open file). A file scheduled for
    deletion outside the data path belongs to the folder the catalog was moved from; it is not
    one of this root's files and is never followed (``_relocate_catalog`` re-roots them)."""
    found: set[str] = set()
    with closing(sqlite3.connect(sqlite_read_only_uri(_real(catalog)), uri=True)) as con:
        bases = con.execute(
            "SELECT value FROM ducklake_metadata WHERE key='data_path' AND scope IS NULL"
        ).fetchall()
        if len(bases) != 1 or not isinstance(bases[0][0], str):
            raise _refuse(
                "backup_catalog_invalid", f"{catalog} lacks one data path; use an intact backup"
            )
        base = PurePosixPath(bases[0][0].replace("\\", "/"))
        refused = catalog_files(
            con.execute(_REFERENCED.format(table="data_file")), base, catalog, found
        ) or catalog_files(
            con.execute(_REFERENCED.format(table="delete_file")), base, catalog, found
        )
        scheduled = con.execute(  # the same parts as a file's, each empty one relative
            "SELECT '',true,'',true,path,path_is_relative"
            " FROM ducklake_files_scheduled_for_deletion"
        )
        refused = refused or catalog_files(scheduled, base, catalog, found, outside=False)
        del scheduled  # no cursor outlives the connection, so a refusal holds no handle
    if refused is not None:
        raise _refuse(*refused)
    return found


def catalog_files(
    rows: Iterable[tuple[Any, ...]],
    base: PurePosixPath,
    catalog: Path,
    found: set[str],
    *,
    outside: bool = True,
) -> tuple[str, str] | None:
    """Add each catalog row's file, resolved from its (path, is relative) parts under the data
    path `base`, to `found` as its backup handle; return the first refusal's code and text
    instead. A row outside `base` is refused, or passed over when `outside` is False."""
    for row in rows:
        if any(value is None for value in row):
            return (
                "backup_catalog_invalid",
                f"{catalog} cannot resolve every referenced file; use an intact backup",
            )
        path = base
        for value, relative in zip(row[::2], row[1::2], strict=True):
            part = PurePosixPath(value.replace("\\", "/"))
            path = path / part if relative else part
        if not path.is_relative_to(base):
            if outside:
                return ("backup_path_aliased", f"{catalog} references {path} outside its data root")
            continue
        named = "data/" + path.relative_to(base).as_posix()
        if not _handle(named):
            return ("backup_manifest_invalid", f"{catalog} references unsafe file {named}")
        found.add(named)
    return None


def _owners(machine: Path, data: Path, owner: Any = None) -> dict[str, str | dict[str, str] | None]:
    """The copied catalog's file/body references and copied log's retained blob proofs."""
    import duckdb  # lazy: the optional data extra reads catalog Evidence body references

    found: dict[str, str | dict[str, str] | None] = dict.fromkeys(
        _catalog_paths(machine / "catalog.sqlite")
    )
    with closing(
        sqlite3.connect(sqlite_read_only_uri(_real(machine / "log.sqlite")), uri=True)
    ) as con:
        profiles = con.execute(
            "SELECT a.artifact_id,a.content_digest,a.body_state,d.name FROM artifact a "
            "LEFT JOIN definition_name d ON d.definition_id=a.type_id AND d.family='artifact_type' "
            "WHERE a.storage='blob'"
        ).fetchall()
    for name, digest, state, kind in profiles:
        if state == "present":
            found["blobs/" + name] = (
                {"kind": "array", "digest": digest} if kind == "array" else digest
            )
    collected = {name for name, _, state, _ in profiles if state == "collected"}
    catalog = sql_path(_real(machine / "catalog.sqlite"))
    location = sql_path(_real(data)) + "/"
    alias = f"backup_{uuid.uuid4().hex}"
    try:
        with closing(
            owner.cursor()
            if owner is not None
            else storage.connect(
                storage.MEMORY,
                config={
                    "autoinstall_known_extensions": False,
                    "autoload_known_extensions": False,
                    **duckdb_thread_config(),
                },
            )
        ) as lake:
            if owner is None:
                load_extensions(lake)
                bound_reads(lake)
            lake.execute(
                f"ATTACH 'ducklake:sqlite:{catalog}' AS {alias} "
                f"(READ_ONLY, DATA_PATH '{location}', OVERRIDE_DATA_PATH true)"
            ).complete()
            try:
                for (name,) in lake.execute(
                    f"SELECT DISTINCT doc_digest FROM {alias}.world.evidence_doc"
                ).fetchall():
                    if name not in collected:
                        found.setdefault("blobs/" + name, None)
            finally:
                if owner is not None:
                    lake.execute(f"DETACH {alias}").complete()
    except duckdb.Error as error:
        raise _refuse(
            "backup_unreadable", f"{machine} cannot read its body owners; use an intact backup"
        ) from error
    if any(
        not _handle(path)
        or (
            not (
                isinstance(digest.get("digest"), str)
                and re.fullmatch(r"[0-9a-f]{64}", digest["digest"])
            )
            if isinstance(digest, dict)
            else digest is not None and not _valid_digest(digest)
        )
        for path, digest in found.items()
    ):
        raise _refuse(
            "backup_manifest_invalid",
            f"{machine} has an unsafe path or unreadable owner digest; repair its owner",
        )
    return found


def _files(root: Path, machine: Path, owner: Any) -> list[dict[str, Any]]:
    """Seal the owners' required files and the admitted blob store without changing identities."""
    owners = _owners(machine, root / "data", owner)
    handles = set(owners) | {path.relative_to(root).as_posix() for path in _walk(root / "blobs")}
    files = []
    for handle in sorted(handles):
        owner = owners.get(handle)
        proof = (
            owner
            if isinstance(owner, dict)
            else {"digest": owner or _file_digest(root / handle, "blake3:v1")}
        )
        files.append({"path": handle, "size": _real(root / handle).stat().st_size, **proof})
    return files


def backup_data_root(root: Path, repository: Path) -> Path:
    """Snapshot index, log, catalog and admitted files; return the published generation. No
    collection removes a body the snapshot names while it copies (governance.COPYING)."""
    with COPYING:
        root, repository = root.resolve(), repository.resolve()
        if repository.is_relative_to(root):
            raise _refuse(
                "backup_overlaps_root",
                f"{repository} is inside {root}; choose an external backup repository",
            )
        owner = host_duckdb(root)
        catalog(root, owner=owner)  # a backup is a use of the catalog
        generations, pool = repository / "generations", repository / "files"
        name = uuid.uuid4().hex
        partial, generation = generations / ("." + name + ".partial"), generations / name
        try:
            _real(generations).mkdir(parents=True, exist_ok=True)
            for folder in ("data", "blobs"):
                _real(pool / folder).mkdir(parents=True, exist_ok=True)
            (partial / "machine").mkdir(parents=True)
            for database in DATABASES:
                _snapshot(root / "machine" / database, partial / "machine" / database)
            head = _head(partial / "machine/log.sqlite")
            _consistent(partial / "machine")
            files = _files(root, partial / "machine", owner)
            with closing(
                sqlite3.connect(sqlite_read_only_uri(partial / "machine/log.sqlite"), uri=True)
            ) as saved_log:
                copied = copy_files(
                    root,
                    pool,
                    files,
                    read=lambda sql, args: saved_log.execute(sql, args).fetchall(),
                )
            manifest = dict(
                version=2,
                databases=list(DATABASES),
                head=head,
                created_at=utc_now().isoformat(),
                copied_files=list(copied),
                files=files,
            )
            (partial / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            os.rename(partial, generation)
            return generation
        except (OSError, sqlite3.Error) as error:
            raise _refuse(
                "backup_unreadable",
                f"machine backup of {root} could not copy {error}; restore access and retry",
            ) from error
        finally:
            if partial.exists():
                shutil.rmtree(partial)


def _relocate_catalog(catalog: Path, data: Path) -> None:
    """Relocate the restored catalog's global data path, and its files scheduled for deletion
    with it: DuckLake records those absolute, so one under the old data path is re-rooted as a
    path relative to the data path, and one outside it, a folder the catalog was moved from
    before, is dropped. A relocated root's cleanup then never deletes outside its own folder."""
    with closing(sqlite3.connect(catalog)) as con:
        old = con.execute(
            "SELECT max(value), count(*) FROM ducklake_metadata"
            " WHERE key = 'data_path' AND scope IS NULL"
        ).fetchone()
        if old is None or old[1] != 1 or not isinstance(old[0], str):
            raise _refuse(
                "backup_catalog_invalid",
                f"{catalog} lacks one global data path; use an intact backup",
            )
        prefix = old[0].replace("\\", "/").rstrip("/") + "/"
        posix = r"replace(path, '\', '/')"  # SQLite reads a backslash in a string as itself
        under = f"substr({posix}, 1, length(?1)) = ?1"
        con.execute(
            "UPDATE ducklake_files_scheduled_for_deletion"
            f" SET path = substr({posix}, length(?1) + 1), path_is_relative = true"
            f" WHERE NOT path_is_relative AND {under}",
            (prefix,),
        )
        con.execute(
            f"DELETE FROM ducklake_files_scheduled_for_deletion WHERE NOT path_is_relative"
            f" AND NOT ({under})",
            (prefix,),
        )
        con.execute(
            "UPDATE ducklake_metadata SET value = ? WHERE key = 'data_path' AND scope IS NULL",
            (data.as_posix() + "/",),
        )
        con.commit()


def _verified_manifest(generation: Path, owner: Any = None) -> dict[str, Any]:
    """The saved chain and owner-derived file set, shared by verification and restoration."""
    generation = generation.resolve()
    manifest = read_manifest(generation)
    try:
        if _head(generation / "machine/log.sqlite") != manifest["head"]:
            raise _refuse(
                "backup_chain_broken",
                f"{generation} does not match its saved chain head; use an untampered backup",
            )
        _consistent(generation / "machine")
        pool = generation.parent.parent / "files"
        owners = _owners(generation / "machine", pool / "data", owner)
        listed = {entry["path"]: entry for entry in manifest["files"]}
        if missing := owners.keys() - listed.keys():
            raise _refuse(
                "backup_manifest_invalid",
                f"{generation} omits required file {min(missing)}; use a complete generation",
            )
        if any(
            (listed[path].get("kind") != "array" or listed[path]["digest"] != digest["digest"])
            if isinstance(digest, dict)
            else (
                listed[path].get("kind") is not None
                or (digest is not None and digest != listed[path]["digest"])
            )
            for path, digest in owners.items()
        ):
            raise _refuse(
                "backup_file_changed",
                f"{generation} differs from its owner's digest; use an intact generation",
            )
        if any(
            entry.get("kind") == "array" and not isinstance(owners.get(path), dict)
            for path, entry in listed.items()
        ):
            raise _refuse(
                "backup_manifest_invalid",
                f"{generation} names an unowned array; use a complete generation",
            )
    except (OSError, sqlite3.Error) as error:
        raise _refuse(
            "backup_unreadable",
            f"{generation / 'machine'} cannot be verified; use an intact backup",
        ) from error
    return manifest


def verify_data_root(
    generation: Path,
    *,
    source: Path | None = None,
    pre_upgrade_head: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the chain, complete file set and sealed bytes without restoring or writing.

    First admission requires the source's current head. Resume may retain a prefix only when
    it is the exact pre-upgrade head already recorded by the upgrade's progress owner.
    """
    generation = generation.resolve()
    manifest = _verified_manifest(generation, None if source is None else host_duckdb(source))
    try:
        pool = generation.parent.parent / "files"
        with closing(
            sqlite3.connect(
                sqlite_read_only_uri(_real(generation / "machine/log.sqlite")), uri=True
            )
        ) as saved_log:
            for entry in manifest["files"]:
                _verify_file(
                    pool / entry["path"],
                    entry,
                    lambda sql, args: saved_log.execute(sql, args).fetchall(),
                )
        if source is not None:
            saved = manifest["head"]
            if pre_upgrade_head is not None and saved != pre_upgrade_head:
                raise _refuse(
                    "backup_chain_broken",
                    f"{generation} differs from the recorded pre-upgrade head; "
                    "supply the recorded backup generation before resuming",
                )
            path = _real(source.resolve() / "machine/log.sqlite")
            with closing(sqlite3.connect(sqlite_read_only_uri(path), uri=True)) as con:
                con.row_factory = sqlite3.Row
                events = con.execute(
                    "SELECT * FROM event WHERE seq <= ? ORDER BY seq", (saved["seq"],)
                )
                bounds = con.execute(
                    "SELECT seq, chain_version FROM chain_checkpoint WHERE seq <= ? ORDER BY seq",
                    (saved["seq"],),
                ).fetchall()
                head = fold_segments(
                    events, [(row[0], row[1]) for row in bounds], saved["chain_version"]
                )
                last = con.execute(
                    "SELECT max(seq) FROM event WHERE seq <= ?", (saved["seq"],)
                ).fetchone()[0]
                if not saved["seq"] or last != saved["seq"] or head != saved["head"]:
                    raise _refuse(
                        "backup_chain_broken",
                        f"{generation} is not a backup of {source}; "
                        "choose that root's verified backup",
                    )
                if (
                    pre_upgrade_head is None
                    and con.execute("SELECT max(seq) FROM event").fetchone()[0] != saved["seq"]
                ):
                    raise _refuse(
                        "backup_chain_broken",
                        f"{generation} is not the current pre-upgrade head of {source}; "
                        "create a fresh backup before starting the upgrade",
                    )
    except (OSError, sqlite3.Error) as error:
        raise _refuse(
            "backup_unreadable",
            f"{generation / 'machine'} cannot be verified; use an intact backup",
        ) from error
    return manifest


@contextmanager
def generation_reads(generation: Path) -> Iterator[tuple[Path, Any]]:
    """A generation as a restore reads it, in place: its pooled files, laid out as a data root's
    `data/` and `blobs/`, and a read of its saved log. Verify the generation first."""
    saved = sqlite_read_only_uri(_real(generation.resolve() / "machine/log.sqlite"))
    with closing(sqlite3.connect(saved, uri=True)) as log:

        def read(sql: str, args: tuple[Any, ...] = ()) -> list[Any]:
            return log.execute(sql, args).fetchall()

        yield generation.resolve().parent.parent / "files", read


def restore_data_root(generation: Path, target: Path) -> Path:
    """Verify a generation, restore into a fresh root and run the existing owed-event repair."""
    import duckdb  # lazy: only restore opens the optional data extra's DuckDB reader

    generation, target = generation.resolve(), Path(os.path.abspath(target))
    if os.path.lexists(target):
        raise _refuse("restore_target_exists", f"{target} exists; restore into a fresh root")
    manifest = _verified_manifest(generation)
    pool = generation.parent.parent / "files"
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.restore")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.mkdir()
        for folder in FOLDERS:
            (partial / folder).mkdir()
        for database in DATABASES:
            _snapshot(generation / "machine" / database, partial / "machine" / database)
        with closing(
            sqlite3.connect(
                sqlite_read_only_uri(_real(generation / "machine/log.sqlite")), uri=True
            )
        ) as saved_log:
            copy_files(
                pool,
                partial,
                manifest["files"],
                read=lambda sql, args: saved_log.execute(sql, args).fetchall(),
            )
        _relocate_catalog(partial / "machine/catalog.sqlite", partial / "data")
        with HostLog.open(partial) as log:
            port = StorePort.open(partial, log)
            port.close()
        _relocate_catalog(partial / "machine/catalog.sqlite", target / "data")
        if os.path.lexists(target):
            raise _refuse("restore_target_exists", f"{target} exists; restore into a fresh root")
        os.rename(partial, target)
        return target
    except (OSError, sqlite3.Error, duckdb.Error) as error:
        raise _refuse(
            "backup_unreadable",
            f"machine restore from {generation} could not copy {error}; use an intact backup",
        ) from error
    finally:
        if partial.exists():
            shutil.rmtree(partial)
