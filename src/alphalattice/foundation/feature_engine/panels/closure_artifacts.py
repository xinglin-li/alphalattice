"""Content-addressed artifact storage for Panel rematerialization closures."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from pydantic import BaseModel

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_contracts import ClosureArtifactRef
from alphalattice.kernel.shared_kernel.persistence import (
    DURABLE_REPLACE_DELAYS,
    replace_with_retry,
)

_Model = TypeVar("_Model", bound=BaseModel)


def ordered_key_hash(session_dates: Iterable[object], listing_ids: Iterable[str]) -> str:
    """Hash an exact ordered Panel key axis without JSON expansion.

    Each key is its session's text, a zero byte, its listing's length and its listing; each
    session's and listing's bytes are encoded once and every key's hashed in one update, the
    same bytes in the same order as one update per field (V92).
    """
    sessions: dict[tuple[type, object], bytes] = {}
    listings: dict[tuple[type, object], bytes] = {}
    parts: list[bytes] = []
    count = 0
    for session, listing_id in zip(session_dates, listing_ids, strict=True):
        session_key = (type(session), session)
        session_bytes = sessions.get(session_key)
        if session_bytes is None:
            session_bytes = sessions[session_key] = str(session).encode("ascii") + b"\0"
        listing_key = (type(listing_id), listing_id)
        listing_bytes = listings.get(listing_key)
        if listing_bytes is None:
            listing = str(listing_id).encode("utf-8")
            listing_bytes = listings[listing_key] = struct.pack(">I", len(listing)) + listing
        parts.append(session_bytes)
        parts.append(listing_bytes)
        count += 1
    digest = hashlib.sha256(b"PanelBaseKeyAxis\0")
    digest.update(b"".join(parts))
    digest.update(struct.pack(">Q", count))
    return digest.hexdigest()


def ordered_value_hash(
    *, factor_id: str, key_hash: str, values: np.ndarray, valid: np.ndarray
) -> str:
    """Hash exact IEEE-754 bits plus the explicit validity mask."""
    floats = np.asarray(values, dtype="<f8")
    mask = np.asarray(valid, dtype=np.bool_)
    if floats.ndim != 1 or mask.ndim != 1 or len(floats) != len(mask):
        raise ValueError("Panel value closure arrays must be aligned one-dimensional arrays")
    digest = hashlib.sha256(b"PanelBaseValueColumn\0")
    digest.update(factor_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(key_hash.encode("ascii"))
    digest.update(struct.pack(">Q", len(floats)))
    digest.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(floats.view("<u8")).tobytes())
    return digest.hexdigest()


def arrow_table_logical_hash(table: pa.Table) -> str:
    """Hash the exact Arrow schema and buffers, independent of Parquet layout."""
    sink = pa.BufferOutputStream()
    normalized = table.replace_schema_metadata(None).combine_chunks()
    with ipc.new_stream(sink, normalized.schema) as writer:
        writer.write_table(normalized)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


class PanelClosureArtifactStore:
    """Publish immutable closure children under the existing artifact root."""

    _URI_PREFIX = "playpen://feature-panel/closure"

    def __init__(
        self, resolver: ArtifactResolver, *, capacity: Callable[[int], None] | None = None
    ) -> None:
        """Locate closure artifacts and bind the optional capacity admission."""
        self._root = Path(resolver.root) / "feature-panel" / "closure"
        self.capacity = capacity

    @property
    def root(self) -> Path:
        """Return the root of the immutable closure artifact store."""
        return self._root

    def publish_json(
        self, *, category: str, content_hash: str, payload: Mapping[str, object]
    ) -> str:
        """Publish a content-addressed JSON closure artifact."""
        target = self._path(category, content_hash, ".json")
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        if self.capacity is not None and not target.exists():
            self.capacity(len(serialized))
        self._atomic_publish(target, serialized)
        if target.read_bytes() != serialized:
            raise ValueError("closure JSON authoritative readback failed")
        return self._uri(category, content_hash)

    def load_json(self, *, category: str, content_hash: str) -> dict[str, object]:
        """Load a JSON closure artifact by its category and content hash."""
        target = self._path(category, content_hash, ".json")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("closure JSON payload is not an object")
        return payload

    def load_model(self, *, category: str, content_hash: str, model: type[_Model]) -> _Model:
        """Validate a JSON closure artifact as the declared model."""
        return cast(
            _Model,
            model.model_validate(self.load_json(category=category, content_hash=content_hash)),
        )

    def find_models(
        self, *, category: str, model: type[_Model], matches: Callable[[_Model], bool]
    ) -> tuple[_Model, ...]:
        """Every published document of one category the predicate admits, by content hash."""
        self._validate_category(category)
        found: list[_Model] = []
        for path in sorted((self._root / category).glob("*.json")):
            if len(path.stem) != 64:
                continue
            candidate = self.load_model(category=category, content_hash=path.stem, model=model)
            if matches(candidate):
                found.append(candidate)
        return tuple(found)

    def publish_parquet(
        self,
        *,
        category: str,
        content_hash: str,
        kind: str,
        table: pa.Table,
    ) -> ClosureArtifactRef:
        """Publish a Parquet closure child with logical and physical hashes."""
        target = self._path(category, content_hash, ".parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(table.schema.metadata or {})
        metadata.update(
            {
                b"alphalattice.closure_kind": kind.encode("utf-8"),
                b"alphalattice.content_hash": content_hash.encode("ascii"),
            }
        )
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        if not target.exists():
            try:
                if self.capacity is not None:
                    sink = pa.BufferOutputStream()
                    pq.write_table(
                        table.replace_schema_metadata(metadata), sink, compression="zstd"
                    )
                    encoded = sink.getvalue().to_pybytes()
                    self.capacity(len(encoded))
                    staged.write_bytes(encoded)
                else:
                    pq.write_table(
                        table.replace_schema_metadata(metadata), staged, compression="zstd"
                    )
                replace_with_retry(staged, target, delays=DURABLE_REPLACE_DELAYS)
            finally:
                staged.unlink(missing_ok=True)
        self._validate_parquet(target, content_hash=content_hash, kind=kind)
        physical = hashlib.sha256(target.read_bytes()).hexdigest()
        return ClosureArtifactRef(
            kind=kind,
            content_hash=content_hash,
            physical_sha256=physical,
            byte_count=target.stat().st_size,
            row_count=table.num_rows,
            uri=self._uri(category, content_hash),
        )

    def load_parquet(self, *, category: str, reference: ClosureArtifactRef) -> pa.Table:
        """Verify a Parquet reference and load its table."""
        expected_uri = self._uri(category, reference.content_hash)
        if reference.uri != expected_uri:
            raise ValueError("closure Parquet URI does not match its content hash")
        target = self._path(category, reference.content_hash, ".parquet")
        self._validate_parquet(target, content_hash=reference.content_hash, kind=reference.kind)
        if hashlib.sha256(target.read_bytes()).hexdigest() != reference.physical_sha256:
            raise ValueError("closure Parquet physical hash does not match its reference")
        if target.stat().st_size != reference.byte_count:
            raise ValueError("closure Parquet byte count does not match its reference")
        table = pq.read_table(target)
        if table.num_rows != reference.row_count:
            raise ValueError("closure Parquet row count does not match its reference")
        return table

    def physical_path(self, *, category: str, reference: ClosureArtifactRef) -> Path:
        """Return a verified Parquet artifact's local path."""
        self.load_parquet(category=category, reference=reference)
        return self._path(category, reference.content_hash, ".parquet")

    def _path(self, category: str, content_hash: str, suffix: str) -> Path:
        self._validate_category(category)
        self._validate_hash(content_hash)
        return self._root / category / f"{content_hash}{suffix}"

    def _uri(self, category: str, content_hash: str) -> str:
        self._validate_category(category)
        self._validate_hash(content_hash)
        return f"{self._URI_PREFIX}/{category}/{content_hash}"

    @staticmethod
    def _validate_category(category: str) -> None:
        if not category or any(part in category for part in ("..", "\\", ":")):
            raise ValueError("invalid closure artifact category")
        if any(not part.replace("-", "").isalnum() for part in category.split("/")):
            raise ValueError("invalid closure artifact category")

    @staticmethod
    def _validate_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("invalid closure artifact hash")

    @staticmethod
    def _atomic_publish(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError("closure artifact identity was reused with different content")
            return
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        try:
            replace_with_retry(staged, target, delays=DURABLE_REPLACE_DELAYS)
        finally:
            staged.unlink(missing_ok=True)

    @staticmethod
    def _validate_parquet(path: Path, *, content_hash: str, kind: str) -> None:
        if not path.is_file():
            raise FileNotFoundError("closure Parquet artifact is missing")
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        if metadata.get(b"alphalattice.content_hash", b"").decode("ascii") != content_hash:
            raise ValueError("closure Parquet content hash metadata is invalid")
        if metadata.get(b"alphalattice.closure_kind", b"").decode("utf-8") != kind:
            raise ValueError("closure Parquet kind metadata is invalid")


__all__ = [
    "PanelClosureArtifactStore",
    "arrow_table_logical_hash",
    "ordered_key_hash",
    "ordered_value_hash",
]
