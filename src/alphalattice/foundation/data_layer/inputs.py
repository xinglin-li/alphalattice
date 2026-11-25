"""The research-input door: a bound Factor input's files, kept once per machine under blobs/.

A research-input file (a binding's market database, its Parquet panels and manifests) is one
artifact of type `research_input`: its bytes under blobs/<artifact_id>, a minted id, with the BLAKE3
of those bytes as `content_digest`, the seal a backup proves and never a key; equal bytes reuse the
one artifact. A binding's manifest, the canonical JSON list of [relative path, member digest] in
path order, is one more such file. The binding itself is a relation with no body, of type
`research_input_binding`, whose `content_digest` is its 0.2.7 `binding_hash` and whose shape names
its manifest's digest, as an Alpha part's relation is (history.keep_alpha_part). Each
workspace that binds it holds the relation, the manifest and each member (`artifact.held`, as
arrays are held), and a deleted workspace's holds go with it (project.py), so StorePort.collect
takes a body once no workspace binds it, past the age rule (data_layer.governance), and the
views' links to it (`drop_views`).

Write order is the array door's: a body is linked or copied beside its place
(blobs/.<artifact_id>.<pid>.staging), its profile recorded, and on commit renamed into place; a
crash's staged body is settled by StorePort's start (`arrays.settle`), as an array's is.

A binding is read as a folder (`view`): each member hard-linked at its relative path under
views/<workspace_id>/research-inputs/, on the data root's own volume, made on a miss and never
backed up; a copy's binding and workspace are read from that place as they were from the old.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Final

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.waits import wait_until
from alphalattice.foundation.data_layer.arrays import STAGED, _settled, _unplaced, _writing
from alphalattice.foundation.data_layer.chain import canonical
from alphalattice.foundation.data_layer.log import drop as drop_lock
from alphalattice.foundation.data_layer.log import hold as hold_lock
from alphalattice.foundation.data_layer.project import UNIVERSE_TEXT, uuid7
from alphalattice.kernel import storage
from alphalattice.kernel.identity import integrity_blake3

if TYPE_CHECKING:
    from alphalattice.foundation.data_layer.log import HostLog

INPUT: Final = "research_input"
BINDING: Final = "research_input_binding"
VIEWS: Final = Path("views") / "research-inputs"
"""Under the data root: views/<workspace_id>/research-inputs/, each workspace's kept bindings."""
CARRIED: Final = Path("research-inputs") / "carried"
"""In a workspace copy: the bindings its workspace holds, each manifest as <binding_hash>.json and
each body by its digest, which the copy's Host keeps on its own root at its open."""
_ANCHOR: Final = "market_catalog_anchor"
"""The table of a binding's sealed index naming the catalog snapshot and cutoff of its selection
(market_snapshots.ANCHOR, written in the transaction that read them)."""
_PANEL_MANIFESTS: Final = "source/artifacts/feature-panel/manifests"
"""Where a binding's folder holds its Feature panel's manifest, <panel_snapshot_hash>.json."""
_MEANING: Final = {
    INPUT: "One file of a bound Factor input, kept once by its bytes.",
    BINDING: "A bound Factor input: its 0.2.7 binding hash and its manifest's digest.",
}


def file_digest(path: Path) -> str:
    """The BLAKE3 seal of a file's bytes, read once without loading it whole."""
    return "blake3:" + integrity_blake3(threaded=True).update_mmap(path).hexdigest()


_DIGEST: Final = re.compile(r"blake3:[0-9a-f]{64}")
_BINDING_HASH: Final = re.compile(r"[0-9a-f]{64}")


def _confined(relative: str) -> str:
    """A member's path under `research-inputs/`, refused by name when it could leave its binding
    on any host: absolute, rooted or drive-bound, or stepping up, in POSIX or Windows syntax (a
    backslash is a separator on Windows), before anything is looked up, linked or logged."""
    for form in (PurePosixPath(relative), PureWindowsPath(relative)):
        if not relative or form.is_absolute() or form.anchor or ".." in form.parts:
            raise DataError(
                f"a research input path leaves its binding: {relative}",
                code="data_layer.input_invalid",
            )
    return relative


def kept(root: Path, read: Any, binding_hash: str) -> bool:
    """Whether this data root keeps the binding: its relation, its manifest and every member's
    body present (none collected)."""
    try:
        _listed(root, read, binding_hash)
    except DataError:
        return False
    return True


def keep_binding(
    root: Path,
    log: HostLog,
    binding_hash: str,
    files: Iterable[tuple[str, Path]],
    *,
    workspace: str | None = None,
    durable: bool = True,
) -> None:
    """Keep each file at its relative path with the binding's manifest and relation; the
    workspace named (its id on the log) holds them all (`hold`), so a copy of it carries the
    binding. Kept with no workspace, nothing holds them: they go by the age rule unless a
    workspace that binds them holds them first.

    A binding kept again keeps what it kept and records nothing twice, so a repeated call (a
    resumed upgrade step) finishes its own incomplete writes and changes nothing it finished."""
    listed = []
    for relative, path in sorted((_confined(relative), path) for relative, path in files):
        digest = _keep(root, log, INPUT, path.stat().st_size, file_digest(path), path)
        listed.append([relative, digest])
    manifest = canonical(listed)
    # Written beside the views, never under blobs/, whose every file a backup copies.
    staged = root / VIEWS.parent / f".{binding_hash}.{os.getpid()}.manifest"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(manifest)
    try:
        sealed = _keep(root, log, INPUT, len(manifest), file_digest(staged), staged)
    finally:
        staged.unlink(missing_ok=True)
    _relate(log, binding_hash, sealed)
    if workspace is not None:
        hold(root, log, binding_hash, workspace)
    if durable:
        log.flush()


def carry(read: Any, root: Path, workspace: str, copy: Path) -> int:
    """Write each binding `workspace` holds into its copy `copy` under CARRIED: its manifest as
    <binding_hash>.json and each body by its digest, linked where the volume allows. Returns the
    bindings the copy now carries. An equal manifest is kept and a different one is refused."""
    rows = read(
        "SELECT a.content_digest FROM live_ref l JOIN artifact a ON a.artifact_id = l.artifact_id"
        " WHERE l.root_kind = 'workspace' AND l.root_id = ? AND a.body_state <> 'collected'"
        " AND a.type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = ?) ORDER BY a.content_digest",
        (workspace, BINDING),
    )
    folder = copy / CARRIED
    for (binding_hash,) in rows:
        listed = _listed(root, read, binding_hash)
        folder.mkdir(parents=True, exist_ok=True)
        for _relative, digest, body in listed:
            _place(body, folder / digest.partition(":")[2])
        manifest = canonical([[relative, digest] for relative, digest, _ in listed])
        _carry_manifest(folder / f"{binding_hash}.json", manifest, binding_hash)
    return len(rows)


def market_selection_window(
    *, cutoff_us: int, universe: str, first: str, last: str, snapshot: int, recipes: dict[str, str]
) -> dict[str, Any]:
    """The window a version 3 binding's market selection records: the cutoff and catalog snapshot
    of its sealed index, its panel's universe, its first and last session and its recipes. A bind
    records it (`factor_inputs`) and a copy's open derives it again from the binding's own files
    (`import_carried`), so the two record one window."""
    spec = {"market_catalog_snapshot": snapshot, "recipes": recipes}
    return {
        "cutoff_us": cutoff_us,
        "universe": universe,
        "valid_from": first,
        "valid_to": last,
        "spec": spec,
    }


def _sealed_selection(binding_hash: str, files: list[tuple[str, Path]]) -> dict[str, Any] | None:
    """The market selection a carried binding names, its id and window derived from the binding's
    own verified files and never from anything carried beside them: its document names the id,
    snapshot, sessions and recipes; its sealed index's anchor the cutoff, under the same snapshot;
    its panel's manifest the universe. None for a binding naming none; a document of another
    binding, an index of another snapshot, or a selection its files do not hold is refused by
    name."""
    found = dict(files)
    document = found.get(f"{binding_hash}/manifest.json")
    bundle = json.loads(document.read_bytes()) if document is not None else {}
    if not isinstance(bundle, dict) or bundle.get("market_selection_id") is None:
        return None

    def refuse(what: str) -> DataError:
        return DataError(
            f"the research input {binding_hash} {what}", code="data_layer.input_invalid"
        )

    if bundle.get("binding_hash") != binding_hash:
        raise refuse("holds the document of another binding")
    unheld = "names a selection its files do not hold"
    try:
        index = found[f"{binding_hash}/source/market-data.duckdb"]
        panel = found[f"{binding_hash}/{_PANEL_MANIFESTS}/{bundle['panel_snapshot_hash']}.json"]
        universe = str(
            json.loads(panel.read_bytes())["safe_summary"]["lineage"]["manifest_revision"]
        )
        sessions = [str(session) for session in bundle["sessions"]]
        recipes = {
            "feature_panel": str(bundle["panel_snapshot_hash"]),
            "execution_outcome": str(bundle["outcome_snapshot_hash"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise refuse(unheld) from error
    import duckdb  # lazy: a copy's open reads an index only for a binding naming a selection

    try:
        with closing(storage.connect(str(index), read_only=True)) as connection:
            anchors = connection.execute(f"SELECT snapshot_id, cutoff_us FROM {_ANCHOR}").fetchall()
    except duckdb.Error as error:
        raise refuse(unheld) from error
    if len(anchors) != 1 or anchors[0][0] != bundle.get("market_catalog_snapshot"):
        raise refuse("seals the index of another selection")
    window = market_selection_window(
        cutoff_us=int(anchors[0][1]),
        universe=universe,
        first=min(sessions),
        last=max(sessions),
        snapshot=int(anchors[0][0]),
        recipes=recipes,
    )
    return {"selection_id": str(bundle["market_selection_id"])} | window


def _carry_manifest(target: Path, manifest: bytes, binding_hash: str) -> None:
    """Rename a complete sibling under a per-manifest hold, even on a linkless volume."""
    if not target.exists(follow_symlinks=False):
        staged = target.with_name(f".{uuid7()}{STAGED}")
        lock = target.with_name(f".{target.name}.lock")
        try:
            staged.write_bytes(manifest)
            held = wait_until(
                lambda _: hold_lock(lock), timeout=5.0, what=f"carry publisher for {target}"
            )
            try:
                if not target.exists(follow_symlinks=False):
                    os.replace(staged, target)
                    return
            finally:
                drop_lock(held, lock)
        finally:
            staged.unlink(missing_ok=True)
    if target.read_bytes() != manifest:
        raise DataError(
            f"a carried research input manifest differs: {binding_hash}",
            code="data_layer.input_invalid",
        )


def import_carried(
    root: Path,
    log: HostLog,
    folder: Path,
    *,
    workspace: str,
    check: Callable[[str, Mapping[str, Path]], object] | None = None,
) -> int:
    """Keep the bindings a copy carries (`folder / CARRIED`) on `log`, held by `workspace`, then
    remove what it carried: the copy's Host does at its open. A manifest not named by a binding
    hash, a member whose digest is not a BLAKE3 seal or whose path leaves its binding, and a body
    whose bytes are not its digest are each refused by name before anything is kept. So is a
    binding naming a market selection that its owner's `check` (factor_inputs'
    `carried_binding_check`, given each path under the binding and its body) refuses, or that no
    check verifies, or whose own files do not hold it (`_sealed_selection`); the selection derived
    from them is then recorded under its id. Returns the bindings kept."""
    carried = folder / CARRIED
    kept = []
    for name in sorted(n for n in os.listdir(carried) if n.endswith(".json")):
        if not _BINDING_HASH.fullmatch(name[:-5]):
            raise DataError(
                f"a carried research input is not named by its binding: {name}",
                code="data_layer.input_invalid",
            )
        files = []
        for relative, digest in json.loads((carried / name).read_bytes()):
            if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
                raise DataError(
                    f"a carried file of the research input {name[:-5]} has no seal: {digest}",
                    code="data_layer.input_invalid",
                )
            body = carried / digest.partition(":")[2]
            if file_digest(body) != digest:
                raise DataError(
                    f"a carried file of the research input {name[:-5]} is not its bytes",
                    code="data_layer.input_invalid",
                )
            files.append((_confined(str(relative)), body))
        kept.append((name[:-5], files))
    selections = []
    for binding_hash, files in kept:
        window = _sealed_selection(binding_hash, files)
        if window is None:
            continue
        if check is None:
            raise DataError(
                f"the research input {binding_hash} names a market selection nothing verifies",
                code="data_layer.input_invalid",
            )
        folder_of = f"{binding_hash}/"
        check(
            binding_hash,
            {
                relative.removeprefix(folder_of): body
                for relative, body in files
                if relative.startswith(folder_of)
            },
        )
        selections.append(window)
    for binding_hash, files in kept:
        keep_binding(root, log, binding_hash, files, workspace=workspace, durable=False)
    for window in selections:
        _adopt_selection(log, workspace, window)
    log.flush()
    shutil.rmtree(carried, ignore_errors=True)
    return len(kept)


def _adopt_selection(log: HostLog, workspace: str, window: dict[str, Any]) -> None:
    """Record a carried binding's market selection on `log`, made by `workspace`, under the id
    the binding names, so its verified read finds it (`world_reads.market_selection_held`); its
    universe registered first, as a selection's is. A log that holds the id already keeps it."""
    selection = window["selection_id"]
    if log.read("SELECT 1 FROM selection WHERE selection_id = ?", (selection,)):
        return
    universe = window.get("universe")
    if universe is not None and not log.read(
        "SELECT 1 FROM definition WHERE family = 'universe' AND name = ? AND version = 1",
        (universe,),
    ):
        key = {"family": "universe", "name": universe, "version": 1}
        body = key | {"spec": {}, "status": "standard", "owner": "host", "text": UNIVERSE_TEXT}
        log.record(
            "definition.registered",
            ("definition", f"universe/{universe}"),
            key,
            text=json.dumps(body, sort_keys=True),
            node="host",
        )
    log.record(
        "selection.made",
        ("selection", selection),
        {name: window.get(name) for name in ("cutoff_us", "universe", "valid_from", "valid_to")}
        | {"spec": window.get("spec", {})},
        workspace=workspace,
        node="host",
    )


def members(root: Path, read: Any, binding_hash: str) -> list[tuple[str, Path]]:
    """A kept binding's files: each relative path and the blob holding its bytes, in path order."""
    return [(relative, body) for relative, _digest, body in _listed(root, read, binding_hash)]


def manifest_held(root: Path, read: Any, binding_hash: str, payload: bytes) -> bool:
    """Prove a binding document against the immutable member seal in the log's relation."""
    digest = _manifest_digest(read, binding_hash)
    body = _body(root, read, INPUT, digest) if digest else None
    if body is None:
        return False
    listed = body.read_bytes()
    if "blake3:" + integrity_blake3(listed).hexdigest() != digest:
        return False
    wanted = f"{binding_hash}/manifest.json"
    actual = "blake3:" + integrity_blake3(payload).hexdigest()
    return [seal for name, seal in json.loads(listed) if name == wanted] == [actual]


def _manifest_digest(read: Any, binding_hash: str) -> str | None:
    """The digest of the manifest the binding's relation names, if the relation is known."""
    relation = read(
        "SELECT json_extract(shape, '$.manifest') FROM artifact WHERE content_digest = ?"
        " AND body_state <> 'collected' AND type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = ?) ORDER BY produced_seq DESC LIMIT 1",
        (binding_hash, BINDING),
    )
    return str(relation[0][0]) if relation else None


def _listed(root: Path, read: Any, binding_hash: str) -> list[tuple[str, str, Path]]:
    """A kept binding's manifest: each relative path, its digest and its body, in path order."""
    digest = _manifest_digest(read, binding_hash)
    manifest = _body(root, read, INPUT, digest) if digest else None
    if manifest is None:
        raise DataError(
            f"the research input {binding_hash} is not kept", code="data_layer.input_missing"
        )
    found = []
    for relative, digest in json.loads(manifest.read_bytes()):
        body = _body(root, read, INPUT, digest)
        if body is None:
            raise DataError(
                f"a file of the research input {binding_hash} is gone",
                code="data_layer.input_missing",
            )
        found.append((str(relative), str(digest), body))
    return found


def view(root: Path, read: Any, binding_hash: str, holder: str) -> Path:
    """The `research-inputs/` folder of the workspace `holder` names (its id on the log), where
    the binding reads as it was bound: each member linked at its relative path, its binding's
    folder made whole beside its place and renamed in, so a reader never sees a part of one. A
    copy's identities (its binding, its workspace) are read from this place, as from the old."""
    folder = root / VIEWS.parent / holder / VIEWS.name
    if (folder / binding_hash).is_dir():
        return folder
    found = members(root, read, binding_hash)
    staged = folder / f".{binding_hash}.{os.getpid()}{STAGED}"
    shutil.rmtree(staged, ignore_errors=True)
    for relative, body in found:
        target = staged / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(body, target)
    for shared in (p for p in staged.iterdir() if p.name != binding_hash):  # pooled databases
        for relative, path in folder_files(shared):
            place = folder / shared.name / relative
            if not place.exists():
                place.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.replace(path, place)
                except OSError:  # another reader placed the same body first (Windows refuses)
                    if not place.is_file():
                        raise
    (staged / binding_hash).mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staged / binding_hash, folder / binding_hash)
    except OSError:
        if not (folder / binding_hash).is_dir():
            raise
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return folder


def folder_files(source: Path) -> Iterable[tuple[str, Path]]:
    """Every file under `source` by its relative POSIX path, refusing a linked folder or file."""

    def refuse(error: OSError) -> None:
        raise error

    for parent, folders, names in os.walk(source, onerror=refuse):
        for name in folders:
            if Path(parent, name).is_symlink():
                raise DataError(
                    "a research input folder is a link", code="data_layer.input_invalid"
                )
        for name in names:
            path = Path(parent, name)
            if path.is_symlink():
                raise DataError("a research input file is a link", code="data_layer.input_invalid")
            yield path.relative_to(source).as_posix(), path


def _type_version(log: HostLog, name: str, storage: str) -> int:
    versions = log.read(
        "SELECT version FROM definition_current WHERE family = 'artifact_type' AND name = ?",
        (name,),
    )
    if versions:
        return int(versions[0][0])
    key = {"family": "artifact_type", "name": name, "version": 1}
    definition = key | {
        "spec": {"storage": storage, "table": None, "measures": [], "keep": "while_active"},
        "status": "standard",
        "owner": "host",
        "text": _MEANING[name],
    }
    log.record(
        "definition.registered",
        ("definition", "artifact_type/" + name),
        key,
        text=json.dumps(definition, sort_keys=True),
        node="host",
    )
    return 1


def _known(read: Any, kind: str, digest: str) -> tuple[str, str] | None:
    rows = read(
        "SELECT artifact_id, body_state FROM artifact WHERE content_digest = ?"
        " AND type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = ?) ORDER BY produced_seq DESC LIMIT 1",
        (digest, kind),
    )
    return (str(rows[0][0]), str(rows[0][1])) if rows else None


def _body(root: Path, read: Any, kind: str, digest: str) -> Path | None:
    known = _known(read, kind, digest)
    if known is None or known[1] != "present":
        return None
    target = root / "blobs" / known[0]
    staged = _unplaced.get(target)
    return staged if staged is not None and staged.is_file() else target


def _keep(root: Path, log: HostLog, kind: str, size: int, digest: str, source: Path) -> str:
    """One body under blobs/: the artifact its bytes already are, or a new one placed on commit."""
    with _writing:
        known = _known(log.read, kind, digest)
        if known is not None and known[1] == "present":
            return digest
        version = _type_version(log, kind, "blob")
        artifact = uuid7()
        target = root / "blobs" / artifact
        staged = target.with_name(f".{artifact}.{os.getpid()}{STAGED}")
        try:
            os.link(source, staged)
        except OSError:  # another volume, or a file system without links: a copy
            shutil.copyfile(source, staged)
        _unplaced[target] = staged
        attrs = {
            "type": kind,
            "type_version": version,
            "identity_kind": "content",
            "storage": "blob",
            "meaning": _MEANING[kind],
            "size_bytes": size,
            "content_digest": digest,
            "held": False,
        }
        try:
            log.record(
                "artifact.produced",
                ("artifact", artifact),
                attrs,
                node="host",
                committed=lambda done: _settled(target, staged, committed=done),
            )
        except BaseException:
            del _unplaced[target]
            raise
        return digest


def _relate(log: HostLog, binding_hash: str, manifest: str) -> None:
    """The binding's relation, once: its 0.2.7 identity and its manifest's digest, with no body."""
    if _known(log.read, BINDING, binding_hash) is not None:
        return
    used = _known(log.read, INPUT, manifest)
    log.record(
        "artifact.produced",
        ("artifact", uuid7()),
        {
            "type": BINDING,
            "type_version": _type_version(log, BINDING, "none"),
            "identity_kind": "content",
            "storage": "none",
            "meaning": _MEANING[BINDING],
            "content_digest": binding_hash,
            "shape": {"manifest": manifest},
            "used": [used[0]] if used else [],
            "held": False,
        },
        node="host",
    )


def _place(source: Path, target: Path) -> None:
    """A body at `target`, linked where the volume allows; one already there is kept."""
    if target.is_file():
        return
    try:
        os.link(source, target)
    except OSError:  # another volume, or no links: a copy
        shutil.copyfile(source, target)


def hold(root: Path, log: HostLog, binding_hash: str, workspace: str) -> None:
    """The workspace's hold on a kept binding, each once (`artifact.held`): its relation, its
    manifest and every member, so no body of a binding a workspace names is collected."""
    relation = _known(log.read, BINDING, binding_hash)
    if relation is None:
        raise DataError("a kept binding has no relation", code="data_layer.input_missing")
    listed = _listed(root, log.read, binding_hash)
    digests = [_manifest_digest(log.read, binding_hash), *(digest for _r, digest, _b in listed)]
    artifacts = [relation[0]]
    artifacts += [found[0] for d in digests if d and (found := _known(log.read, INPUT, d))]
    for artifact in dict.fromkeys(artifacts):
        held = log.read(
            "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace'"
            " AND root_id = ?",
            (artifact, workspace),
        )
        if not held:
            log.record("artifact.held", ("artifact", artifact), {}, workspace=workspace)


def drop_views(root: Path, read: Any) -> int:
    """Remove each view of a binding this root no longer keeps (its bodies collected), and each
    pooled database no binding left in that view lists: a view's links would keep the collected
    bytes. A view of a kept binding stays. Returns the folders removed."""
    removed, views = 0, root / VIEWS.parent
    holders = sorted(views.iterdir()) if views.is_dir() else []
    for holder in (p / VIEWS.name for p in holders if (p / VIEWS.name).is_dir()):
        listed: set[str] = set()
        for folder in sorted(p for p in holder.iterdir() if _BINDING_HASH.fullmatch(p.name)):
            if kept(root, read, folder.name):
                listed |= {rel for rel, _digest, _body in _listed(root, read, folder.name)}
            else:
                shutil.rmtree(folder)
                removed += 1
        pooled = holder / "databases"
        for database in sorted(pooled.iterdir()) if pooled.is_dir() else []:
            names = {f"databases/{database.name}/{body.name}" for body in database.iterdir()}
            if not names & listed:
                shutil.rmtree(database)
                removed += 1
    return removed
