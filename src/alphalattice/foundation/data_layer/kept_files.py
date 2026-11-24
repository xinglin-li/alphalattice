"""A workspace's named files kept on the log: each file's bytes in the array door, its name in a
`file.kept` event.

A store of files that moves here keeps every file byte for byte, under its name relative to the
workspace's folder: the body is the array door's packed array of those bytes (`arrays.keep_packed`,
held by the workspace, so storage governance keeps it as it keeps any array), and one `file.kept`
event per change names it. The event's subject is `<workspace_id>:<name>`; its newest event is the
file, and one whose digest is null is a removal. A store reads and writes its files through here
once its upgrade family has moved them; the files it wrote before stay where they are.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import alphalattice.foundation.data_layer.arrays as arrays
from alphalattice.foundation.data_layer.chain import canonical
from alphalattice.foundation.data_layer.log import HostLog

EVENT: Final = "file.kept"
SPEC: Final = {
    "work": False,
    "subject": "kept_file",
    "required": ["name", "digest"],
    "text": "none",
    "durable": True,
}
"""A kept file's change: its name and the digest of its new bytes, null when it is removed."""

type Read = Callable[[str, tuple[Any, ...]], list[Any]]


def subject(workspace_id: str, name: str) -> str:
    """The event subject of `workspace_id`'s file `name`."""
    return f"{workspace_id}:{name}"


def digest_of(read: Read, workspace_id: str, name: str) -> str | None:
    """The digest of the file's newest bytes, or None when it is not kept or was removed."""
    rows = read(
        "SELECT attrs FROM event WHERE type = ? AND subject_id = ? ORDER BY seq DESC LIMIT 1",
        (EVENT, subject(workspace_id, name)),
    )
    return json.loads(rows[0][0])["digest"] if rows else None


def read(root: Path, query: Read, workspace_id: str, name: str) -> bytes | None:
    """The file's bytes, proved against their digest by the door, or None when it is not kept."""
    digest = digest_of(query, workspace_id, name)
    return None if digest is None else arrays.read_packed(root, query, digest, packed_only=True)


def names(query: Read, workspace_id: str, folder: str, *, by_write: bool = False) -> list[str]:
    """The names of the kept files directly in `folder`: sorted, or `by_write`, the one written
    last at the end."""
    inside = f"{folder.rstrip('/')}/"
    found = [
        name.removeprefix(inside)
        for name in _newest(query, workspace_id, inside)
        if "/" not in name.removeprefix(inside)
    ]
    return found if by_write else sorted(found)


def held(query: Read, workspace_id: str) -> dict[str, str]:
    """Every kept file of the workspace, by name: the digest of its newest bytes, in the order of
    their newest writes."""
    return _newest(query, workspace_id, "")


def known(query: Read, workspace_id: str) -> set[str]:
    """The names the log has any event for under the workspace, removed ones included."""
    prefix = subject(workspace_id, "")
    rows = query(
        "SELECT DISTINCT subject_id FROM event WHERE type = ? AND subject_id >= ?"
        " AND subject_id < ?",
        (EVENT, prefix, prefix + "￿"),
    )
    return {str(row[0]).removeprefix(prefix) for row in rows}


def _newest(query: Read, workspace_id: str, inside: str) -> dict[str, str]:
    """The workspace's kept files whose names start with `inside`: each one's newest digest, in
    the order of their newest writes."""
    prefix = subject(workspace_id, "")
    newest: dict[str, str | None] = {}
    for found, attrs in query(
        "SELECT subject_id, attrs FROM event WHERE type = ? AND subject_id >= ?"
        " AND subject_id < ? ORDER BY seq",
        (EVENT, prefix + inside, prefix + inside + "\uffff"),
    ):
        name = str(found).removeprefix(prefix)
        newest.pop(name, None)  # a later write moves it to the end
        newest[name] = json.loads(attrs)["digest"]
    return {name: digest for name, digest in newest.items() if digest is not None}


def keep(
    root: Path,
    log: HostLog,
    workspace_id: str,
    name: str,
    data: bytes,
    *,
    durable: bool = True,
    rewrite: bool = False,
) -> str:
    """Keep `data` as the file `name`: its body once per digest, then its event, unless its newest
    event already names these bytes. Returns the digest. Not `durable`, the caller flushes;
    `rewrite` records the event even then, so this write is the file's newest (a move that
    keeps a store's write order)."""
    digest: str = arrays.keep_packed(root, log, data, workspace=workspace_id, durable=durable)
    index(log, workspace_id, name, digest, rewrite=rewrite)
    return digest


def index(
    log: HostLog, workspace_id: str, name: str, digest: str, *, rewrite: bool = False
) -> None:
    """Name the kept body `digest` as the file `name`, unless its newest event already does and
    this is no `rewrite`."""
    if rewrite or digest_of(log.read, workspace_id, name) != digest:
        _register(log)
        log.record(
            EVENT,
            ("kept_file", subject(workspace_id, name)),
            {"name": name, "digest": digest},
            workspace=workspace_id,
            node="host",
        )


CARRIED: Final = "kept-files.json"
"""A copy's carried index of its kept files, beside the arrays it carries (`arrays.CARRIED`): a
list of `[name, digest]` in the order of their newest writes."""


def remove(log: HostLog, workspace_id: str, name: str) -> None:
    """Remove the file `name`; its earlier bodies stay the array door's, as every array does."""
    if digest_of(log.read, workspace_id, name) is not None:
        _register(log)
        log.record(
            EVENT,
            ("kept_file", subject(workspace_id, name)),
            {"name": name, "digest": None},
            workspace=workspace_id,
            node="host",
        )


def _register(log: HostLog) -> None:
    """Admit the event type through definition.registered, as the upgrade's progress is: the
    upgrade keeps files before the Host records."""
    if log.read(
        "SELECT 1 FROM definition WHERE family = 'event_type' AND name = ? AND version = 1",
        (EVENT,),
    ):
        return
    definition = {"family": "event_type", "name": EVENT, "version": 1}
    text = {**definition, "status": "standard", "spec": SPEC, "text": "A workspace's file changed."}
    log.record(
        "definition.registered",
        ("definition", f"event_type/{EVENT}"),
        definition,
        text=canonical(text).decode(),
        node="host",
    )
