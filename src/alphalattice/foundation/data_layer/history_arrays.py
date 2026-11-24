"""Keep and read arrays through the Host's array door."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.history_runtime import (
    _array_reader,
    _held_log,
    held,
)
from alphalattice.foundation.data_layer.root import (
    data_root,
)


def keep_array(
    columns: Mapping[str, Any] | None = None,
    *,
    packed: bytes | None = None,
    durable: bool = True,
) -> str:
    """Keep one array through the door (`arrays`) and return the digest a document cites: in a
    Host, on its log and held by its workspace; in another process, on the log it holds (`held`),
    held by none. `durable=False` leaves its profile in the log's batch for the caller to commit
    (`flush`) before it writes anything citing the digest."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _held_log() as found:
        current = found
    log, workspace, root = current if current is not None else (held(), None, data_root())
    if packed is not None:
        return arrays.keep_packed(root, log, packed, workspace=workspace, durable=durable)
    assert columns is not None
    return arrays.keep(root, log, columns, workspace=workspace, durable=durable)


def read_array(content_digest: str, *, packed: bool = False) -> Any:
    """The array a digest names, checked against its profile and proved: its flat columns, or its
    C-order bytes with `packed`; from any process."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (root, read):
        return (arrays.read_packed if packed else arrays.read_columns)(root, read, content_digest)


def array_profile(content_digest: str) -> dict[str, Any] | None:
    """The profile of the array a digest names (its kind, shape and dtypes, its artifact_id and
    body state); None when the door holds none."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (_root, read):
        return arrays.profile(read, content_digest)


def array_body_path(content_digest: str) -> Path:
    """The array owner's body path for a sealed-source cache; read through the door."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (root, read):
        known = arrays.profile(read, content_digest)
        if known is None or known["body_state"] != "present":
            raise DataError(f"no array is named {content_digest}", code="data_layer.array_missing")
        return root / "blobs" / known["artifact_id"]


def array_held_by(content_hash: str, workspace: str | None) -> bool:
    """Whether an array is present and, when named, held by this workspace."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (_root, read):
        profile = arrays.profile(read, content_hash)
        if profile is None or profile["body_state"] != "present":
            return False
        return workspace is None or bool(
            read(
                "SELECT 1 FROM live_ref WHERE artifact_id = ? AND root_kind = 'workspace'"
                " AND root_id = ?",
                (profile["artifact_id"], workspace),
            )
        )


def arrays_lacking(digests: Iterable[str]) -> list[str]:
    """The digests the door holds no array for, in one read of the log."""
    import alphalattice.foundation.data_layer.arrays as arrays  # lazy: numpy, BLAKE3 and Parquet

    with _array_reader() as (_root, read):
        return [digest for digest in digests if arrays.profile(read, digest) is None]
