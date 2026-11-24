"""Keep and restore Alpha's digest-bound array parts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.history_runtime import (
    _array_reader,
    _held_log,
    _once,
    flush,
    held,
)
from alphalattice.foundation.data_layer.log import HostLog, uuid7
from alphalattice.foundation.data_layer.root import (
    data_root,
)


def _alpha_part_link(read: Callable[..., Any], content_hash: str, layout: str) -> Any:
    """The checked relation and array profile, read together from the log."""
    rows = read(
        "SELECT a.artifact_id, u.artifact_id, b.content_digest, b.body_state,"
        " b.type_id = (SELECT definition_id FROM definition_name"
        " WHERE family = 'artifact_type' AND name = 'array'), b.shape"
        " FROM artifact a JOIN definition_name d ON d.definition_id = a.type_id"
        " LEFT JOIN event_artifact u ON u.seq = a.produced_seq AND u.role = 'used'"
        " LEFT JOIN artifact b ON b.artifact_id = u.artifact_id"
        " WHERE d.family = 'artifact_type' AND d.name = 'alpha_part'"
        " AND a.content_digest = ? AND json_extract(a.shape, '$.layout') = ?",
        (content_hash, layout),
    )
    if not rows:
        return None
    if (
        len(rows) != 1
        or rows[0][1] is None
        or rows[0][2] is None
        or rows[0][3] != "present"
        or rows[0][4] != 1
        or rows[0][5] is None
    ):
        raise DataError(
            "the Alpha part's array relation is invalid", code="data_layer.part_invalid"
        )
    return rows[0]


def _alpha_part_body(read: Callable[..., Any], content_hash: str, layout: str) -> str | None:
    link = _alpha_part_link(read, content_hash, layout)
    return None if link is None else str(link[2])


type AlphaParts = dict[tuple[str, str], tuple[str, Path]]

_ALPHA_PARTS: ContextVar[tuple[Path, AlphaParts] | None] = ContextVar("alpha_parts", default=None)


@contextmanager
def collect_alpha_parts(folder: Path) -> Iterator[AlphaParts]:
    """Stage one child's parts in its call folder, with no writable log."""
    parts: AlphaParts = {}
    token = _ALPHA_PARTS.set((folder, parts))
    try:
        yield parts
    finally:
        _ALPHA_PARTS.reset(token)


def _collected_alpha_part(content_hash: str, layout: str) -> tuple[str, Path] | None:
    collected = _ALPHA_PARTS.get()
    return None if collected is None else collected[1].get((content_hash, layout))


def keep_collected_alpha_parts(parts: AlphaParts) -> None:
    """The Host keeps a successful child's parts through the ordinary array door."""
    for (content_hash, layout), (digest, path) in parts.items():
        kept = keep_alpha_part(
            content_hash, layout, _read_collected_alpha_part(digest, path), durable=False
        )
        if kept != digest:
            raise DataError(
                "the Alpha part changed during the child call", code="data_layer.part_invalid"
            )
    if parts:
        flush()


def _read_collected_alpha_part(digest: str, path: Path) -> bytes:
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    payload = path.read_bytes()
    if arrays.packed_digest(payload) != digest:
        raise DataError("the staged Alpha part body changed", code="data_layer.part_invalid")
    return payload


def _collect_alpha_part(
    folder: Path,
    parts: AlphaParts,
    content_hash: str,
    layout: str,
    payload: bytes,
    capacity: Callable[[int], None] | None,
) -> str:
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    key = (content_hash, layout)
    if key in parts:
        return parts[key][0]
    digest = alpha_part_body_digest(content_hash, layout)
    if digest is not None:
        # Preserve the first body even when a later fit serialized it differently.
        payload = read_alpha_part(content_hash, layout)
    else:
        digest = arrays.packed_digest(payload)
        if capacity is not None:
            capacity(len(payload))
    path = folder / f"alpha-part-{len(parts)}"
    path.write_bytes(payload)
    parts[key] = (digest, path)
    return digest


def keep_alpha_part(
    content_hash: str,
    layout: str,
    payload: bytes,
    *,
    capacity: Callable[[int], None] | None = None,
    durable: bool = True,
    upgrade_owner: tuple[HostLog, Path, str | None] | None = None,
) -> str:
    """Keep a legacy-format body and relate it to its unchanged content and layout identities."""
    collected = _ALPHA_PARTS.get()
    if collected is not None and upgrade_owner is None:
        return _collect_alpha_part(*collected, content_hash, layout, payload, capacity)

    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _once:
        if upgrade_owner is not None:
            log, root, workspace = upgrade_owner
        else:
            with _held_log() as found:
                current = found
            log, workspace, root = current if current is not None else (held(), None, data_root())
        link = _alpha_part_link(log.read, content_hash, layout)
        if link is not None:
            existing = str(link[2])
            # The first body for this semantic identity remains authoritative and is held here.
            body = arrays.read_packed(root, log.read, existing)
            if (
                arrays.keep_packed(
                    root, log, body, workspace=workspace, durable=False, capacity=capacity
                )
                != existing
            ):
                raise DataError("the Alpha part body changed", code="data_layer.part_invalid")
            arrays._hold(log, str(link[0]), workspace)
            if durable:
                log.flush()
            return existing
        described = arrays.describe_packed(payload)
        digest = arrays._keep(root, log, described, workspace, False, capacity)
        name = "alpha_part"
        versions = log.read(
            "SELECT version FROM definition_current WHERE family = 'artifact_type' AND name = ?",
            (name,),
        )
        if not versions:
            key = {"family": "artifact_type", "name": name, "version": 1}
            definition = key | {
                "spec": {"storage": "none", "table": None, "measures": [], "keep": "while_active"},
                "status": "standard",
                "owner": "host",
                "text": "An Alpha part's content and layout identities and the array it uses.",
            }
            log.record(
                "definition.registered",
                ("definition", "artifact_type/" + name),
                key,
                text=json.dumps(definition, sort_keys=True),
                node="host",
            )
            version = 1
        else:
            version = int(versions[0][0])
        log.record(
            "artifact.produced",
            ("artifact", uuid7()),
            {
                "type": name,
                "type_version": version,
                "identity_kind": "content",
                "storage": "none",
                "meaning": "An Alpha part's content and layout identities.",
                "content_digest": content_hash,
                "shape": {"layout": layout},
                "used": [described.artifact_id],
                "held": workspace is not None,
            },
            workspace=workspace,
            node="host",
        )
        if durable:
            log.flush()
        return digest


def read_alpha_part(content_hash: str, layout: str) -> bytes:
    """Resolve the typed Alpha part relation and read its verified packed body."""
    staged = _collected_alpha_part(content_hash, layout)
    if staged is not None:
        return _read_collected_alpha_part(*staged)

    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (root, read):
        link = _alpha_part_link(read, content_hash, layout)
        if link is None:
            raise DataError("the Alpha part is missing", code="data_layer.part_missing")
        digest, artifact, shape = str(link[2]), str(link[1]), link[5]

        def profile(_sql: str, args: tuple[Any, ...]) -> list[Any]:
            if args != (digest, arrays.TYPE):
                raise DataError("the Alpha array profile changed", code="data_layer.part_invalid")
            return [(artifact, "present", shape)]

        return arrays.read_packed(root, profile, digest, packed_only=True)


def alpha_part_body_digest(content_hash: str, layout: str) -> str | None:
    """The packed-body digest linked to a content and layout pair, if upgraded."""
    staged = _collected_alpha_part(content_hash, layout)
    if staged is not None:
        return staged[0]
    with _array_reader() as (_root, read):
        return _alpha_part_body(read, content_hash, layout)


def alpha_part_body_path(content_hash: str, layout: str) -> Path:
    """The linked array's source path for the existing sealed-source verifier."""
    staged = _collected_alpha_part(content_hash, layout)
    if staged is not None:
        return staged[1]
    with _array_reader() as (root, read):
        link = _alpha_part_link(read, content_hash, layout)
        if link is None:
            raise DataError("the Alpha part is missing", code="data_layer.part_missing")
        return root / "blobs" / str(link[1])


def _held_alpha_parts(read: Callable[..., Any], workspace: str) -> dict[tuple[str, str], str]:
    """The Alpha part relations `workspace` holds with their present, held array bodies, read in
    one query: each content and layout pair to its body's digest."""
    rows = read(
        "SELECT a.content_digest, json_extract(a.shape, '$.layout'), b.content_digest"
        " FROM live_ref l JOIN artifact a ON a.artifact_id = l.artifact_id"
        " JOIN definition_name d ON d.definition_id = a.type_id"
        " JOIN event_artifact u ON u.seq = a.produced_seq AND u.role = 'used'"
        " JOIN artifact b ON b.artifact_id = u.artifact_id"
        " JOIN live_ref body ON body.artifact_id = b.artifact_id"
        " AND body.root_kind = 'workspace' AND body.root_id = l.root_id"
        " WHERE l.root_kind = 'workspace' AND l.root_id = ?"
        " AND d.family = 'artifact_type' AND d.name = 'alpha_part' AND b.body_state = 'present'",
        (workspace,),
    )
    return {(str(content), str(layout)): str(digest) for content, layout, digest in rows}


def _import_alpha_parts(root: Path, log: HostLog, folder: Path, workspace: str) -> None:
    """Restore carried Alpha content/layout relations after the carried arrays are admitted. The
    relations stay in the copy and are read at every start; one `workspace` already holds with
    the same body is not read again."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    source = folder / arrays.CARRIED / "alpha-parts.json"
    if source.is_symlink():
        raise DataError("carried Alpha part relations are a link", code="data_layer.part_invalid")
    try:
        payload = source.read_bytes()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DataError(
            "carried Alpha part relations are unreadable", code="data_layer.part_invalid"
        ) from error
    try:
        rows = json.loads(payload)
        if not isinstance(rows, list):
            raise ValueError("not a list")
        held = _held_alpha_parts(log.read, workspace)
        seen: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError("not a part relation")
            content, layout, digest = row
            if (
                not isinstance(content, str)
                or not isinstance(digest, str)
                or any(
                    len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                    for v in (content, digest)
                )
                or not isinstance(layout, str)
                or not layout
                or (content, layout) in seen
            ):
                raise ValueError("invalid or duplicate part relation")
            seen.add((content, layout))
            if held.get((content, layout)) == digest:
                continue  # admitted at an earlier start: this workspace holds it and its body
            body = arrays.read_packed(root, log.read, digest, packed_only=True)
            if (
                keep_alpha_part(content, layout, body, upgrade_owner=(log, root, workspace))
                != digest
            ):
                raise ValueError("part relation differs from the held one")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise DataError(
            "carried Alpha part relations are invalid", code="data_layer.part_invalid"
        ) from error
