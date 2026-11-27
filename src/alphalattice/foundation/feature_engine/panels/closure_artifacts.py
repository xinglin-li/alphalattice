"""Content-addressed artifact storage for Panel rematerialization closures."""

from __future__ import annotations

import json
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
from alphalattice.control.workspace_runtime.verified_facts import (
    record_sealed_file,
    sealed_file_holds,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import ClosureArtifactRef
from alphalattice.foundation.feature_engine.panels.closure_files import (
    ClosureFiles,
    FolderFiles,
    closure_files,
)
from alphalattice.kernel.identity import (
    blake3_filename,
    blake3_from_filename,
    identity_blake3,
    integrity_blake3,
    is_blake3,
    tagged_blake3,
)

_Model = TypeVar("_Model", bound=BaseModel)


def ordered_key_hash(session_dates: Iterable[object], listing_ids: Iterable[str]) -> str:
    """Hash an exact ordered Panel key axis without JSON expansion.

    Each key is its session's text, a zero byte, its listing's length and its listing; each
    session's and listing's bytes are encoded once and every key's hashed in one update, the
    same bytes in the same order as one update per field.
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
    digest = identity_blake3(b"PanelBaseKeyAxis\0")
    digest.update(b"".join(parts))
    digest.update(struct.pack(">Q", count))
    return tagged_blake3(digest)


def ordered_value_hash(
    *, factor_id: str, key_hash: str, values: np.ndarray, valid: np.ndarray
) -> str:
    """Hash exact IEEE-754 bits plus the explicit validity mask."""
    floats = np.asarray(values, dtype="<f8")
    mask = np.asarray(valid, dtype=np.bool_)
    if floats.ndim != 1 or mask.ndim != 1 or len(floats) != len(mask):
        raise ValueError("Panel value closure arrays must be aligned one-dimensional arrays")
    digest = identity_blake3(b"PanelBaseValueColumn\0")
    digest.update(factor_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(key_hash.encode("ascii"))
    digest.update(struct.pack(">Q", len(floats)))
    digest.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(floats.view("<u8")).tobytes())
    return tagged_blake3(digest)


def arrow_table_logical_hash(table: pa.Table) -> str:
    """Hash the exact Arrow schema and buffers, independent of Parquet layout."""
    sink = pa.BufferOutputStream()
    normalized = table.replace_schema_metadata(None).combine_chunks()
    with ipc.new_stream(sink, normalized.schema) as writer:
        writer.write_table(normalized)
    return tagged_blake3(identity_blake3(sink.getvalue()))


def _physical(data: bytes) -> str:
    """A closure Parquet file's physical digest, as its reference records it: BLAKE3 of its
    bytes, taken by the writer and checked again by a reader of the log's copy."""
    return tagged_blake3(integrity_blake3(data))


class PanelClosureArtifactStore:
    """Publish immutable closure children under the existing artifact root: in its folder, or a
    workspace's own root on the log once the upgrade moved it (`closure_files`)."""

    _URI_PREFIX = "playpen://feature-panel/closure"

    def __init__(
        self, resolver: ArtifactResolver, *, capacity: Callable[[int], None] | None = None
    ) -> None:
        """Locate closure artifacts and bind the optional capacity admission."""
        self._root = Path(resolver.root) / "feature-panel" / "closure"
        self.capacity = capacity
        self._files: ClosureFiles | None = None

    @property
    def root(self) -> Path:
        """Return the root of the immutable closure artifact store."""
        return self._root

    @property
    def files(self) -> ClosureFiles:
        """The root's files, decided at the first use: the Host composes its stores after the
        upgrade, so one store reads one place."""
        if self._files is None:
            self._files = closure_files(self._root)
        return self._files

    def holds(self, category: str, content_hash: str, suffix: str = ".json") -> bool:
        """Whether the child is published, its bytes unread."""
        return self.files.exists(self._name(category, content_hash, suffix))

    def publish_json(
        self, *, category: str, content_hash: str, payload: Mapping[str, object]
    ) -> str:
        """Publish a content-addressed JSON closure artifact."""
        name = self._name(category, content_hash, ".json")
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        self._publish(name, serialized)
        if self.files.read(name) != serialized:
            raise ValueError("closure JSON authoritative readback failed")
        return self._uri(category, content_hash)

    def load_json(self, *, category: str, content_hash: str) -> dict[str, object]:
        """Load a JSON closure artifact by its category and content hash."""
        name = self._name(category, content_hash, ".json")
        data = self.files.read(name)
        if data is None:
            raise FileNotFoundError(f"closure JSON artifact is missing: {name}")
        payload = json.loads(data.decode("utf-8"))
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
        for name in self.files.names(category):
            stem, _, suffix = name.rpartition(".")
            if suffix != "json" or (digest := blake3_from_filename(stem)) is None:
                continue
            candidate = self.load_model(category=category, content_hash=digest, model=model)
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
        """Publish a Parquet closure child with logical and physical hashes; one already
        published keeps its own bytes."""
        name = self._name(category, content_hash, ".parquet")
        data = self.files.read(name)
        if data is None:
            metadata = dict(table.schema.metadata or {})
            metadata.update(
                {
                    b"alphalattice.closure_kind": kind.encode("utf-8"),
                    b"alphalattice.content_hash": content_hash.encode("ascii"),
                }
            )
            sink = pa.BufferOutputStream()
            pq.write_table(table.replace_schema_metadata(metadata), sink, compression="zstd")
            data = sink.getvalue().to_pybytes()
            self._publish(name, data)
        self._validate_parquet(data, content_hash=content_hash, kind=kind)
        physical = _physical(data)
        if isinstance(self.files, FolderFiles):  # checked here once; reads take it by its stat
            record_sealed_file(self._root / name, physical)
        return ClosureArtifactRef(
            kind=kind,
            content_hash=content_hash,
            physical_digest=physical,
            byte_count=len(data),
            row_count=table.num_rows,
            uri=self._uri(category, content_hash),
        )

    def load_parquet(self, *, category: str, reference: ClosureArtifactRef) -> pa.Table:
        """Verify a Parquet reference and load its table: its bytes are the digest its writer
        recorded, while a folder file's stat holds or as the log proves them."""
        expected_uri = self._uri(category, reference.content_hash)
        if reference.uri != expected_uri:
            raise ValueError("closure Parquet URI does not match its content hash")
        name = self._name(category, reference.content_hash, ".parquet")
        if isinstance(self.files, FolderFiles):
            target = self._root / name
            if not target.is_file():
                raise FileNotFoundError("closure Parquet artifact is missing")
            self._validate_parquet(target, content_hash=reference.content_hash, kind=reference.kind)
            if not sealed_file_holds(target, reference.physical_digest):
                raise ValueError("closure Parquet physical hash does not match its reference")
            size = target.stat().st_size
            source: Path | pa.BufferReader = target
        else:
            data = self.files.read(name)
            if data is None:
                raise FileNotFoundError("closure Parquet artifact is missing")
            if _physical(data) != reference.physical_digest:
                raise ValueError("closure Parquet physical hash does not match its reference")
            self._validate_parquet(data, content_hash=reference.content_hash, kind=reference.kind)
            size, source = len(data), pa.BufferReader(data)
        if size != reference.byte_count:
            raise ValueError("closure Parquet byte count does not match its reference")
        table = pq.read_table(source)
        if table.num_rows != reference.row_count:
            raise ValueError("closure Parquet row count does not match its reference")
        return table

    def physical_path(self, *, category: str, reference: ClosureArtifactRef) -> Path:
        """Return a verified Parquet artifact's local path: a folder's alone, since the log
        keeps no file to name."""
        self.load_parquet(category=category, reference=reference)
        if not isinstance(self.files, FolderFiles):
            raise ValueError("closure Parquet artifact is kept on the log, not in a folder")
        return self._root / self._name(category, reference.content_hash, ".parquet")

    def _publish(self, name: str, content: bytes) -> None:
        """Write an immutable child once: the same bytes again are left as they are, other bytes
        under its identity are refused, and only a new child asks the capacity admission."""
        held = self.files.read(name)
        if held is not None:
            if held != content:
                raise ValueError("closure artifact identity was reused with different content")
            return
        if self.capacity is not None:
            self.capacity(len(content))
        self.files.write(name, content)

    def _name(self, category: str, content_hash: str, suffix: str) -> str:
        self._validate_category(category)
        return f"{category}/{blake3_filename(content_hash)}{suffix}"

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
        if not is_blake3(value):
            raise ValueError("invalid closure artifact hash")

    @staticmethod
    def _validate_parquet(source: Path | bytes, *, content_hash: str, kind: str) -> None:
        held = pa.BufferReader(source) if isinstance(source, bytes) else source
        metadata = pq.ParquetFile(held).schema_arrow.metadata or {}
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
