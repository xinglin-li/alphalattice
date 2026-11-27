"""Digest algorithms and the questions their callers ask."""

from __future__ import annotations

import json
import re
from collections.abc import Buffer
from hashlib import sha256 as _sha256
from os import PathLike
from typing import BinaryIO, Final, Protocol, Self


def integrity_sha256_lock(data: Buffer) -> str:
    """Emit the SHA-256 checksum recorded for the external uv lock file."""
    return _sha256(data).hexdigest()


def integrity_sha256_browser_style(data: Buffer) -> bytes:
    """Bind exact inline style bytes in the browser's SHA-256 CSP format."""
    return _sha256(data).digest()


def integrity_sha256_model_release(stream: BinaryIO) -> str:
    """Check upstream model release files against their published SHA-256 pins."""
    digest = _sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def integrity_hmac_sha256(key: bytes, body: bytes) -> str:
    """Authenticate a message with the existing SHA-256 HMAC."""
    import hmac  # lazy: only committee authentication needs HMAC.

    return hmac.new(key, body, _sha256).hexdigest()


class Blake3Digest(Protocol):
    """The streaming operations used by BLAKE3 seals."""

    def update(self, data: Buffer, /) -> Self: ...
    def update_mmap(self, path: str | PathLike[str]) -> Self: ...
    def copy(self) -> Self: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...


def _blake3(data: Buffer = b"", *, threaded: bool = False) -> Blake3Digest:
    """Open a BLAKE3 stream with the caller's existing threading choice."""
    from blake3 import blake3  # lazy: only a BLAKE3 digest needs this backend.

    return blake3(data, max_threads=blake3.AUTO if threaded else 1)


identity_blake3 = _blake3
integrity_blake3 = _blake3
change_blake3 = _blake3

BLAKE3_PATTERN: Final = r"^blake3:[0-9a-f]{64}$"
_BLAKE3 = re.compile(BLAKE3_PATTERN)


def is_blake3(value: object) -> bool:
    """Whether the value is one complete tagged BLAKE3 digest."""
    return isinstance(value, str) and _BLAKE3.fullmatch(value) is not None


def blake3_hex(value: str) -> str:
    """The validated digest's hex for an on-disk filename or binary encoding."""
    if not is_blake3(value):
        raise ValueError("identity.blake3_digest_invalid")
    return value.removeprefix("blake3:")


def blake3_filename(value: str) -> str:
    """Use a tagged digest as a filename stem on all supported systems."""
    return "blake3-" + blake3_hex(value)


def blake3_from_filename(stem: str) -> str | None:
    """Recover a tagged digest from a recognized filename stem."""
    if not stem.startswith("blake3-"):
        return None
    value = "blake3:" + stem.removeprefix("blake3-")
    return value if is_blake3(value) else None


def tagged_blake3(digest: Blake3Digest) -> str:
    """Write a complete BLAKE3 digest in the product's persisted form."""
    return "blake3:" + digest.hexdigest()


def blake3_digest(data: Buffer) -> str:
    """Return the tagged digest of bytes already in memory."""
    return tagged_blake3(integrity_blake3(data))


def integrity_blake3_stream(stream: BinaryIO) -> Blake3Digest:
    """Hash the remaining bytes of an open file with bounded memory."""
    digest = integrity_blake3()
    while block := stream.read(1 << 20):
        digest.update(block)
    return digest


def change_xxh3_128_hex(data: Buffer) -> str:
    """Index an in-process change key with the existing 128-bit xxHash."""
    from xxhash import xxh3_128_hexdigest  # lazy: only a change key needs xxHash.

    return xxh3_128_hexdigest(data)


# ``json.dumps`` with these options builds a new encoder on every call; the
# encoder is stateless, so one instance encodes every value to the same bytes.
_CANONICAL_ENCODER = json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=str)


def canonical_json(value: object) -> str:
    """The canonical text whose UTF-8 BLAKE3 digest is ``canonical_hash``.

    Exposed for an owner that streams a large payload through the digest in
    pieces; the pieces must concatenate to exactly this text.
    """
    return _CANONICAL_ENCODER.encode(value)


def canonical_hash(value: object) -> str:
    """Hash canonical JSON-safe contract content, never a payload or local path."""
    return tagged_blake3(identity_blake3(canonical_json(value).encode("utf-8")))
