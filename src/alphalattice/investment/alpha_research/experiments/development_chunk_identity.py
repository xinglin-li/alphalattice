"""Alpha development chunk identities, consumed by artifact publication and readback."""

from hashlib import sha256
from threading import Lock

import pyarrow as pa

from alphalattice.kernel.shared_kernel.arrow_identity import canonical_hash_with_rows
from alphalattice.kernel.shared_kernel.bounded_memo import BoundedMemo
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _chunk_content_hash(table: pa.Table, identity: dict[str, object]) -> str:
    """The content identity of one development chunk: its identity, schema and rows.

    The same canonical document the chunk was first sealed with -- identity,
    the schema text and every row as a sorted-key object -- hashed from the
    Arrow columns (``canonical_hash_with_rows``, bit for bit the encoder's
    output) rather than from a Python copy of every row.
    """

    return canonical_hash_with_rows(
        {"identity": identity, "schema": str(table.schema)}, key="rows", table=table
    )


_VERIFIED_CHUNKS: BoundedMemo[tuple[str, str], str] = BoundedMemo()
"""Each development chunk this process derived in full: its file's SHA-256 and its identity's
hash, to the content hash derived from those bytes. The derivation is a function of the
bytes alone, so a chunk whose bytes hash the same was derived before; its read still hashes the
file and runs every other check. Kept in memory only: nothing on disk vouches for a file.
Capacity bounds retained proofs; a forgotten entry derives and verifies again."""


def configure_chunk_proof_memo(*, capacity: int | None = None) -> None:
    """Set this owner's derived-chunk memo capacity; a miss derives and verifies in full."""
    global _VERIFIED_CHUNKS
    with _VERIFIED_CHUNK_LOCK:
        _VERIFIED_CHUNKS = BoundedMemo(capacity=capacity)


_VERIFIED_CHUNK_LOCK = Lock()


def _verified_content_hash(raw: bytes, table: pa.Table, identity: dict[str, object]) -> str:
    """The chunk's content hash, derived in full the first time this process reads its bytes."""

    key = (sha256(raw).hexdigest(), str(canonical_hash(identity)))
    with _VERIFIED_CHUNK_LOCK:
        derived = _VERIFIED_CHUNKS.get(key)
    if derived is None:
        derived = _chunk_content_hash(table, identity)
        with _VERIFIED_CHUNK_LOCK:
            _VERIFIED_CHUNKS[key] = derived
    return derived
