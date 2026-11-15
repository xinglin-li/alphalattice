"""StorePort: the one door to the catalog and its files, the write order, and its one repair.

The data root's files, by who names them:
    staging/<admission_id>/  one admission's Parquet before its catalog commit
    staging/<admission_id>.lock  held by its producer from before the folder exists until its
                             commit returns; staging whose lock no one holds belongs to a write
                             that will never finish, and start collects it
    data/                    DuckLake's files, named by its catalog; any other file is collected
    blobs/<artifact_id>      an artifact's body, kept while a live_ref holds it

The write order (catalog.sql's header). `stage` mints the admission's id, digests its rows and
writes their Parquet under staging. It needs no log, lock or catalog, so it runs beside the
writer, in the process that fetched the rows, and hands the writer only a commit request
(`Staged`): the digests are Python work, and in the Host's process they would hold the
interpreter while an append waits. `StorePort.commit` is the writer: one catalog snapshot inserts
the rows with one shared known_at, writes the supersession of every version they replace and
carries in its commit info the whole envelope of the `admission.committed` it owes; then `repair`
appends that event, durably, and the staging folder is deleted. `start` runs the same `repair`
when the Host opens, so the normal path and crash recovery are one path, and collects what no
snapshot or write owns. Every file leaves through `delete`: one another process holds open waits
in deletion_queue and is retried at each start, and nothing a live_ref holds is ever deleted.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, NamedTuple

from alphalattice.control.workspace_runtime.storage.capacity import (
    FREE_FLOOR_BYTES,
    free_floor_shortfall,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import ENVELOPE, canonical, digest
from alphalattice.kernel.data_layer.log import HostLog, drop, envelope, hold, uuid7
from alphalattice.kernel.data_layer.root import load_extensions, schema_text, statements

KINDS: Final = {"staging": "staging", "blobs": "blob", "data": "file"}
"""The deletion queue's kind for each top folder a deleted file may live in."""
WORLD_KEYS: Final = {"raw_bar": ("provider", "listing_id", "session_date")}
"""Each world table StorePort admits, with its key less value_digest. The bar goes first: it is
the largest table, and Step M measured its as-of read."""
METADATA: Final = "__ducklake_metadata_lake"
"""The DuckLake catalog's own tables, as the Host's session names them."""


class Staged(NamedTuple):
    """One admission's rows, digested and written under staging, waiting for its commit."""

    admission_id: str
    table: str
    provider: str
    row_count: int
    parts: tuple[str, ...]
    """The years its rows' sessions fall in: the partitions a cache reads them from."""


class StorePort:
    """The Host's one writer of the catalog; open it with `StorePort.open`."""

    def __init__(self, root: Path, log: HostLog, lake: Any):
        """Hold the data root, its log and the Host's DuckDB session; `open` makes one."""
        self.root, self.log, self.lake = root, log, lake
        self._writer = threading.Lock()
        self.started: dict[str, list[str]] = {}

    @classmethod
    def open(cls, root: Path, log: HostLog) -> StorePort:
        """Attach the catalog, the log read-only and the reading surface, then run `start`."""
        import duckdb  # lazy: DuckDB is the optional data extra; only a Host with data opens this

        lake = duckdb.connect(  # in memory: the catalog and the log are attached to it
            ":memory:",
            config={"autoinstall_known_extensions": False, "autoload_known_extensions": False},
        )
        load_extensions(lake)
        machine, data = (root / "machine").as_posix(), (root / "data").as_posix()
        for sql in (
            f"ATTACH 'ducklake:sqlite:{machine}/catalog.sqlite' AS lake"
            f" (DATA_PATH '{data}/', DATA_INLINING_ROW_LIMIT 1000)",
            f"ATTACH '{machine}/log.sqlite' AS log (TYPE SQLITE, READ_ONLY)",
            "SET TimeZone = 'UTC'",
            *statements(schema_text("views.sql")),
        ):
            lake.execute(sql).complete()
        port = cls(root, log, lake)
        port.started = start(root, log, lake)
        return port

    def commit(self, staged: Staged) -> int:
        """Commit one staged admission as one catalog snapshot, then its owed event; returns the
        snapshot. The event is durable, so it is on disk when this returns."""
        with self._writer:
            known = self.log.clock()
            parts = [
                {"part": part, "min_known_asof_us": known, "max_known_asof_us": known}
                for part in staged.parts
            ]
            attrs = {
                "provider": staged.provider,
                "dataset": staged.table,
                "row_count": staged.row_count,
                "known_us": known,
                "parts": parts,
            }
            owed = envelope(
                "admission.committed",
                ("admission", staged.admission_id),
                attrs,
                actor=("host", "store"),
                node="store",
                occurred_us=known,
            )
            self._catalog_commit(staged, known, owed)
            repair(self.log, self.lake)
            delete(self.root, self.log, f"staging/{staged.admission_id}")
            found = self.log.read(
                "SELECT catalog_snapshot FROM admission WHERE admission_id = ?",
                (staged.admission_id,),
            )
            return int(found[0][0])

    def close(self) -> None:
        """Close the Host's DuckDB session; the log is the caller's."""
        self.lake.close()

    def _catalog_commit(self, staged: Staged, known: int, owed: Mapping[str, Any]) -> None:
        """One snapshot: the supersession of each version the rows replace, then the rows."""
        table, key = staged.table, WORLD_KEYS[staged.table]
        parquet = self.root / "staging" / staged.admission_id / f"{table}.parquet"
        source = f"read_parquet('{parquet.as_posix()}')"
        same = " AND ".join(f"t.{k} = n.{k}" for k in key)
        at = "make_timestamptz(?::BIGINT)"
        run = self.lake.execute
        run("BEGIN").complete()
        try:
            run(
                f"INSERT INTO lake.world.superseded SELECT ?, t.value_digest, t.admission_id, {at},"
                f" ? FROM lake.world.{table} t WHERE EXISTS (SELECT 1 FROM {source} n WHERE {same})"
                " AND NOT EXISTS (SELECT 1 FROM lake.world.superseded s WHERE s.table_name = ?"
                " AND s.value_digest = t.value_digest AND s.admission_id = t.admission_id)",
                (table, known, staged.admission_id, table),
            ).complete()
            run(
                f"INSERT INTO lake.world.{table} BY NAME SELECT *, {at} AS known_at,"
                f" {at} AS known_asof, false AS retracted, ? AS admission_id FROM {source}",
                (known, known, staged.admission_id),
            ).complete()
            run(
                "CALL lake.set_commit_message('host', 'admission', extra_info => ?)",
                (canonical(dict(owed)).decode(),),
            ).complete()
            run("COMMIT").complete()
        except BaseException:
            run("ROLLBACK").complete()
            raise


@contextmanager
def stage(
    root: Path,
    table: str,
    provider: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    floor_bytes: int = FREE_FLOOR_BYTES,
) -> Iterator[Staged]:
    """Digest one admission's rows and write them under staging; the commit request it yields is
    all the writer needs. Its producer holds the admission's lock until the block ends, so a Host
    that starts meanwhile leaves it: commit it inside the block. Refused by name below the
    free-space floor, and as `digested` refuses."""
    short = free_floor_shortfall(root, floor_bytes)
    if short:
        raise DataError(
            f"the data root's disk is {short} bytes below its {floor_bytes}-byte free-space floor,"
            " kept for the log, its repair and cleanup; free space on that disk first",
            code="data_layer.below_free_floor",
        )
    admission = uuid7()  # first: the rows and the staging folder carry it
    staged = digested(table, provider, rows)
    import pyarrow as pa  # lazy: the optional data extra, as DuckDB
    import pyarrow.parquet as pq  # lazy: as pyarrow

    lock = root / "staging" / f"{admission}.lock"
    held = hold(lock)  # before the folder, so no start sees the folder unheld
    if held is None:
        raise DataError(
            "a starting Host collected this admission's staging before its producer held it;"
            " stage the rows again",
            code="data_layer.staging_taken",
        )
    try:
        (root / "staging" / admission).mkdir()
        pq.write_table(
            pa.Table.from_pylist(staged), root / "staging" / admission / f"{table}.parquet"
        )
        years = sorted({str(row["session_date"])[:4] for row in staged})
        yield Staged(admission, table, provider, len(staged), tuple(years))
    finally:
        drop(held, lock)


def digested(table: str, provider: str, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each row with its provider and its value_digest, the digest of the two together.

    Refused by name: a table StorePort does not admit, and two rows of one key, since DuckLake
    keeps no keys.
    """
    key = WORLD_KEYS.get(table)
    if key is None:
        raise DataError(
            f"{table} is not a world table StorePort admits; it admits {', '.join(WORLD_KEYS)}",
            code="data_layer.world_table_not_admitted",
        )
    staged: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for given in rows:
        row = {**given, "provider": provider}
        values = tuple(row[k] for k in key)
        if values in seen:
            raise DataError(
                f"{table} {dict(zip(key, values, strict=True))} appears twice in one admission;"
                " send one version of a key",
                code="data_layer.admission_key_repeated",
            )
        seen.add(values)
        staged.append(row | {"value_digest": digest(canonical(row))})
    return staged


def repair(log: HostLog, lake: Any) -> list[str]:
    """Append each event a catalog snapshot carried and the log lacks; returns their ids.

    An owed event keeps every field its writer made, its id, version, workspace, actor, trace and
    time included; only its recording time is new, after the moment it carries, and its
    attributes name the snapshot. Running it again adds none.
    """
    carried = lake.execute(
        "SELECT snapshot_id, commit_extra_info FROM lake.snapshots()"
        " WHERE commit_extra_info IS NOT NULL ORDER BY snapshot_id"
    ).fetchall()
    added = []
    for snapshot, info in carried:
        owed = json.loads(info)
        if log.read("SELECT 1 FROM event WHERE event_id = ?", (owed["event_id"],)):
            continue
        attrs = json.loads(owed["attrs"])
        row = {column: owed[column] for column in ENVELOPE if column in owed and column != "seq"}
        row["attrs"] = canonical(attrs | {"catalog_snapshot": snapshot}).decode()
        log.clock(at_least=attrs.get("known_us", 0))
        log.append(row)
        added.append(owed["event_id"])
    return added


def delete(root: Path, log: HostLog, handle: str) -> bool:
    """Delete the file or folder at `handle` (a path under the data root); True once it is gone.

    A body a live_ref holds is refused by name. One another process holds open goes to
    deletion_queue with its error, to be retried; False then.
    """
    path, kind = _resolve(root, handle)
    if kind == "blob" and log.read("SELECT 1 FROM live_ref WHERE artifact_id = ?", (path.name,)):
        raise DataError(
            f"{handle} is held by a live reference and is never collected",
            code="data_layer.live_ref_held",
        )
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as error:
        log.queue_deletion(handle, kind, f"{type(error).__name__}: {error.strerror or error}")
        return False
    log.dequeue_deletion(handle)
    return True


def start(root: Path, log: HostLog, lake: Any) -> dict[str, list[str]]:
    """The Host's start: repair the log from the catalog, then collect. Returns the events it
    repaired, by id, and what it collected and what still waits, by handle.

    It retries the deletion queue, collects the staging of every admission no producer holds,
    holding that admission's lock itself until its staging and lock are gone, so no producer can
    take it meanwhile, and collects every file under data/ the catalog does not name.
    """
    report: dict[str, list[str]] = {"repaired": repair(log, lake), "collected": [], "waiting": []}
    queued = [row[0] for row in log.read("SELECT handle FROM deletion_queue ORDER BY handle")]
    staging, data = root / "staging", root / "data"
    admissions = {Path(handle).name for handle in queued if handle.startswith("staging/")}
    if staging.is_dir():
        admissions |= {child.name.removesuffix(".lock") for child in staging.iterdir()}
    for admission in sorted(admissions):
        lock, handle = staging / f"{admission}.lock", f"staging/{admission}"
        probe = hold(lock)
        if probe is None:
            continue  # its producer is staging it or waiting for its commit
        try:
            if (staging / admission).exists() or handle in queued:
                report["collected" if delete(root, log, handle) else "waiting"].append(handle)
        finally:
            drop(probe, lock)
    named = _named_files(lake)
    handles = [handle for handle in queued if not handle.startswith("staging/")]
    handles += sorted(
        f"data/{(folder / name).relative_to(data).as_posix()}"
        for folder, _, names in data.walk(on_error=_unreadable)
        for name in names
        if name not in named
    )
    for handle in dict.fromkeys(handles):
        report["collected" if delete(root, log, handle) else "waiting"].append(handle)
    log.flush()
    return report


def _named_files(lake: Any) -> frozenset[str]:
    """The name of every file the catalog names: data, deletes, and those DuckLake will clean up.
    DuckLake names each file by a fresh UUIDv7, so a name is unique under data/."""
    rows = lake.execute(
        " UNION ALL ".join(
            f"SELECT path FROM {METADATA}.ducklake_{table}"
            for table in ("data_file", "delete_file", "files_scheduled_for_deletion")
        )
    ).fetchall()
    return frozenset(Path(row[0]).name for row in rows)


def _unreadable(error: OSError) -> None:
    """A folder under data/ that cannot be listed stops the start: read as empty, its unnamed
    files would stay unseen."""
    raise DataError(
        f"{error.filename} under the data root's data/ cannot be listed ({error.strerror});"
        " give the Host read access to it and start again",
        code="data_layer.data_folder_unreadable",
    ) from error


def _resolve(root: Path, handle: str) -> tuple[Path, str]:
    """The path a handle names and its deletion kind; refused unless it lies in a known folder."""
    parts = Path(handle).parts
    kind = KINDS.get(parts[0]) if len(parts) > 1 else None
    if kind is None or ".." in parts or Path(handle).is_absolute():
        raise DataError(
            f"{handle} names no file under staging/, data/ or blobs/",
            code="data_layer.handle_outside_root",
        )
    return root.joinpath(*parts), kind
