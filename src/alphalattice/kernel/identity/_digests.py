"""Digest algorithms and the questions their callers ask."""

from __future__ import annotations

import json
from collections.abc import Buffer
from hashlib import file_digest as _file_digest
from hashlib import sha256 as _sha256
from io import BufferedReader, BytesIO
from os import PathLike
from typing import Protocol, Self

# Bind the existing algorithm directly, including its streaming interface, for inherited digests.
identity_sha256 = _sha256
integrity_sha256 = _sha256
change_sha256 = _sha256


class Sha256Digest(Protocol):
    """A SHA-256 stream that can be copied at a known prefix."""

    def update(self, data: Buffer, /) -> None: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...
    def copy(self) -> Self: ...


def integrity_sha256_stream(stream: BufferedReader | BytesIO) -> Sha256Digest:
    """Digest the bytes of an already-open binary stream."""
    return _file_digest(stream, "sha256")


def integrity_hmac_sha256(key: bytes, body: bytes) -> str:
    """Authenticate a message with the existing SHA-256 HMAC."""
    import hmac  # lazy: only committee authentication needs HMAC.

    return hmac.new(key, body, _sha256).hexdigest()


class _Blake3Digest(Protocol):
    """The streaming operations used by BLAKE3 seals."""

    def update(self, data: Buffer, /) -> Self: ...
    def update_mmap(self, path: str | PathLike[str]) -> Self: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...


def _blake3(data: Buffer = b"", *, threaded: bool = False) -> _Blake3Digest:
    """Open a BLAKE3 stream with the caller's existing threading choice."""
    from blake3 import blake3  # lazy: only a BLAKE3 digest needs this backend.

    return blake3(data, max_threads=blake3.AUTO if threaded else 1)


identity_blake3 = _blake3
integrity_blake3 = _blake3
change_blake3 = _blake3


def change_xxh3_128_hex(data: Buffer) -> str:
    """Index an in-process change key with the existing 128-bit xxHash."""
    from xxhash import xxh3_128_hexdigest  # lazy: only a change key needs xxHash.

    return xxh3_128_hexdigest(data)


# ``json.dumps`` with these options builds a new encoder on every call; the
# encoder is stateless, so one instance encodes every value to the same bytes.
_CANONICAL_ENCODER = json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=str)


def canonical_json(value: object) -> str:
    """The canonical text whose UTF-8 SHA-256 is ``canonical_hash``.

    Exposed for an owner that streams a large payload through the digest in
    pieces; the pieces must concatenate to exactly this text.
    """
    return _CANONICAL_ENCODER.encode(value)


def canonical_hash(value: object) -> str:
    """Hash canonical JSON-safe contract content, never a payload or local path."""
    return identity_sha256(canonical_json(value).encode("utf-8")).hexdigest()
