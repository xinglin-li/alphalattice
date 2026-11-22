"""Carry a workspace's sealed Evidence snapshots with exactly their admitted revisions."""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path
from sqlite3 import Connection
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import canonical, digest
from alphalattice.kernel.data_layer.history_runtime import _open, holding
from alphalattice.kernel.data_layer.knowledge import KnowledgeStore, SourceOrigin
from alphalattice.kernel.data_layer.log import attach
from alphalattice.kernel.data_layer.root import data_root

if TYPE_CHECKING:
    from alphalattice.kernel.data_layer.log import HostLog


CARRIED_KNOWLEDGE: Final = Path("runtime") / "carried-knowledge-snapshots.json"
_SCHEMA: Final = "alphalattice.carried-knowledge-snapshots.v1"


def _invalid(detail: str) -> DataError:
    return DataError(detail, code="data_layer.knowledge_publication_conflict")


def _carried(folder: Path) -> Path:
    path = folder / CARRIED_KNOWLEDGE
    if path.is_symlink() or not path.parent.resolve().is_relative_to(folder.resolve()):
        raise _invalid("carried knowledge leaves its workspace")
    return path


def _snapshot_set(con: Connection, workspace: Path) -> bytes | None:
    """Read one workspace and its snapshot closure in one consistent log read."""
    row = con.execute(
        "SELECT workspace_id FROM workspace WHERE folder = ? AND status = 'active'",
        (str(workspace.resolve()),),
    ).fetchone()
    if row is None:
        return None
    named = row[0]
    snapshots = [
        json.loads(text)["metadata"]
        for (text,) in con.execute(
            "SELECT t.text FROM knowledge_snapshot s"
            " JOIN event_text t ON t.seq = s.metadata_seq WHERE s.workspace_id = ?"
            " ORDER BY s.reference_id",
            (named,),
        )
    ]
    if not snapshots:
        return None
    needed = {
        (item["document"]["document_id"], item["revision"])
        for snapshot in snapshots
        for item in snapshot["revisions"]
    }
    revisions = []
    for document, revision, metadata, body, source, version, kind, known, seal in con.execute(
        "SELECT w.document_id, w.revision, w.metadata, t.text, v.source_id, v.version,"
        " s.kind, v.known_us, v.body_digest FROM knowledge_workspace_source w"
        " JOIN knowledge_source_version v USING (version_id)"
        " JOIN knowledge_source s USING (source_id)"
        " LEFT JOIN event_text t ON t.seq = v.body_seq WHERE w.workspace_id = ?"
        " ORDER BY w.document_id, w.revision",
        (named,),
    ):
        if (document, revision) not in needed:
            continue
        if kind != "reference" or body is None:
            raise _invalid(f"an Evidence snapshot names no kept reference body: {document}")
        revisions.append(
            [
                json.loads(metadata),
                json.loads(body)["content"],
                [source, version, kind, known],
                seal,
            ]
        )
        needed.remove((document, revision))
    if needed:
        raise _invalid("an Evidence snapshot names a revision absent from its workspace")
    return canonical({"schema": _SCHEMA, "snapshots": snapshots, "revisions": revisions})


def carry_knowledge_snapshots(
    workspace: Path, copy: Path, *, root: Path | None = None
) -> tuple[Path, ...]:
    """Place a complete carry file in a copy or backup staging area, if it has Evidence."""
    where = data_root() if root is None else root
    log = holding()
    if log is not None and Path(_open["root"]).resolve() == where.resolve():
        with log.reading() as con:
            payload = _snapshot_set(con, workspace)
    elif (where / "machine" / "log.sqlite").is_file():
        with closing(attach(where)) as con:
            con.execute("BEGIN")
            payload = _snapshot_set(con, workspace)
    else:
        payload = None
    if payload is None:
        return ()
    target = _carried(copy)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=".knowledge-", delete=False) as stream:
            pending = Path(stream.name)
            stream.write(payload)
        os.replace(pending, target)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return (target,)


def import_carried_knowledge(log: HostLog, folder: Path, workspace: str) -> int:
    """Admit the sealed revisions and snapshots together, then remove the carry."""
    target = _carried(folder)
    try:
        raw = target.read_bytes()
    except FileNotFoundError:
        return 0
    store = KnowledgeStore(log)
    try:
        payload: dict[str, Any] = json.loads(raw)
        if payload["schema"] != _SCHEMA or canonical(payload) != raw:
            raise _invalid("carried knowledge does not have its sealed form")
        snapshots = payload["snapshots"]
        with log.unit():
            for metadata, content, origin, seal in payload["revisions"]:
                body = SourceOrigin(*origin)
                if body.kind != "reference" or digest(content.encode()) != seal:
                    raise _invalid("a carried Evidence revision differs from its sealed body")
                store.publish_revision(
                    workspace_id=workspace, metadata=metadata, content=content, origin=body
                )
            for snapshot in snapshots:
                store.publish_snapshot(workspace_id=workspace, metadata=snapshot)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise _invalid("carried knowledge is unreadable") from error
    log.flush()
    target.unlink()
    return len(snapshots)
