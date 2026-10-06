"""Content-addressed storage for causal execution outcomes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .compile import _SCHEMA_ID
from .contracts import (
    CausalExecutionOutcomeChunk,
    CausalExecutionOutcomeManifest,
    CausalExecutionOutcomeReceipt,
    DevelopmentOnlyExecutionOutcomeManifest,
    DevelopmentOnlyExecutionOutcomeMarker,
)
from .methods import ExecutionOutcomeMethodBinding, ExecutionOutcomeMethodSealMarker

#: Storage category of the artifact method seal. It lives beside the manifest
#: rather than inside it because the manifest's bytes are frozen, and a seal is
#: a separate assertion about a snapshot rather than a field of one.
METHOD_BINDING_CATEGORY = "method-bindings"

#: Terminal authority for a method-bound snapshot. Separate from the binding
#: because a binding on disk proves only that a well-formed file exists; the
#: marker is what the publisher writes last, so its presence is the claim.
METHOD_SEAL_MARKER_CATEGORY = "method-seal-markers"

#: The development-only successor snapshot and its publication marker. Separate
#: categories rather than shared directories, so a reader of the frozen contract
#: never has to decide which of two schemas a file in "manifests" is -- and a
#: scope that carries no sealed holdout cannot be reached by a consumer looking
#: for one.
DEVELOPMENT_ONLY_MANIFEST_CATEGORY = "development-only-manifests"
DEVELOPMENT_ONLY_MARKER_CATEGORY = "development-only-markers"


class _ExecutionOutcomeArtifactStore:
    _PREFIX = "playpen://data-operations/execution-outcomes/"

    def __init__(self, artifact_root: Path) -> None:
        self.root = artifact_root.resolve() / "data-operations" / "execution-outcomes"

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, path)

    def _uri(self, category: str, value: str) -> str:
        return f"{self._PREFIX}{category}/{value}"

    def _json_path(self, category: str, value: str) -> Path:
        return self.root / category / f"{value}.json"

    def _parquet_path(self, split: str, value: str) -> Path:
        return self.root / split.casefold().replace("_", "-") / "chunks" / f"{value}.parquet"

    def publish_json(
        self, *, category: str, payload: Mapping[str, object], identity_field: str
    ) -> ArtifactDescriptor:
        value = str(payload[identity_field])
        content = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        path = self._json_path(category, value)
        if path.exists() and path.read_bytes() != content:
            raise ValueError("causal execution identity was reused with different bytes")
        if not path.exists():
            self._atomic_write(path, content)
        return ArtifactDescriptor(
            kind=f"causal-execution-{category}",
            content_hash=value,
            metadata_hash=hashlib.sha256(content).hexdigest(),
            uri=self._uri(category, value),
        )

    def load_json(self, *, category: str, uri: str) -> dict[str, object]:
        prefix = f"{self._PREFIX}{category}/"
        if not uri.startswith(prefix):
            raise ValueError("causal execution URI is invalid")
        value = uri[len(prefix) :]
        payload = json.loads(self._json_path(category, value).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("causal execution JSON is invalid")
        return payload

    def publish_chunk(self, *, split: str, table: pa.Table) -> CausalExecutionOutcomeChunk:
        ordered = table.combine_chunks().sort_by(
            [("formation_session", "ascending"), ("listing_id", "ascending")]
        )
        # The identity is the canonical JSON of (listing, formation, row hash)
        # per row; a formation session is encoded as ``str(value)`` by the
        # canonical encoder, so the columns are read once each and the
        # session is given as that string -- the same bytes, without a dict
        # per row.
        content_hash = canonical_hash(
            {
                "schema": _SCHEMA_ID,
                "rows": tuple(
                    zip(
                        ordered.column("listing_id").to_pylist(),
                        (str(value) for value in ordered.column("formation_session").to_pylist()),
                        ordered.column("row_hash").to_pylist(),
                        strict=True,
                    )
                ),
            }
        )
        path = self._parquet_path(split, content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            b"alphalattice.snapshot_kind": b"CausalExecutionOutcomeChunk",
            b"alphalattice.chunk_hash": content_hash.encode(),
            b"alphalattice.split": split.encode(),
        }
        if not path.exists():
            staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            pq.write_table(ordered.replace_schema_metadata(metadata), staged, compression="zstd")
            os.replace(staged, path)
        parquet = pq.ParquetFile(path)
        actual_metadata = parquet.schema_arrow.metadata or {}
        if actual_metadata.get(b"alphalattice.chunk_hash") != content_hash.encode():
            raise ValueError("causal execution chunk metadata is invalid")
        metadata_hash = hashlib.sha256(
            json.dumps(
                {key.decode(): value.decode() for key, value in sorted(actual_metadata.items())},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        dates = ordered.column("formation_session").to_pylist()
        year = dates[0].year
        return CausalExecutionOutcomeChunk(
            split=split,
            year=year,
            row_count=ordered.num_rows,
            first_formation_session=min(dates),
            last_formation_session=max(dates),
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            uri=self._uri(f"{split.casefold().replace('_', '-')}/chunks", content_hash),
        )

    def resolve_chunk(self, chunk: CausalExecutionOutcomeChunk) -> Path:
        path = self._parquet_path(chunk.split, chunk.content_hash)
        metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
        if metadata.get(b"alphalattice.chunk_hash") != chunk.content_hash.encode():
            raise ValueError("causal execution chunk readback is invalid")
        return path

    def manifests(self) -> tuple[CausalExecutionOutcomeManifest, ...]:
        root = self.root / "manifests"
        return (
            tuple(
                CausalExecutionOutcomeManifest.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def receipts(self) -> tuple[CausalExecutionOutcomeReceipt, ...]:
        root = self.root / "receipts"
        return (
            tuple(
                CausalExecutionOutcomeReceipt.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def development_only_manifests(self) -> tuple[DevelopmentOnlyExecutionOutcomeManifest, ...]:
        """Every development-only snapshot on disk, validated as it is read."""
        root = self.root / DEVELOPMENT_ONLY_MANIFEST_CATEGORY
        return (
            tuple(
                DevelopmentOnlyExecutionOutcomeManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def development_only_markers(self) -> tuple[DevelopmentOnlyExecutionOutcomeMarker, ...]:
        """Every development-only publication marker on disk, validated as it is read."""
        root = self.root / DEVELOPMENT_ONLY_MARKER_CATEGORY
        return (
            tuple(
                DevelopmentOnlyExecutionOutcomeMarker.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def method_bindings(self) -> tuple[ExecutionOutcomeMethodBinding, ...]:
        """Every artifact method seal on disk, validated as it is read.

        Absence is normal and means legacy: a snapshot published before the
        method seam existed has no seal, and this returning nothing for it is
        what keeps it legacy rather than letting a reader mint one.
        """
        root = self.root / METHOD_BINDING_CATEGORY
        return (
            tuple(
                ExecutionOutcomeMethodBinding.model_validate_json(path.read_text(encoding="utf-8"))
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )

    def method_seal_markers(self) -> tuple[ExecutionOutcomeMethodSealMarker, ...]:
        """Every terminal method seal on disk, validated as it is read."""
        root = self.root / METHOD_SEAL_MARKER_CATEGORY
        return (
            tuple(
                ExecutionOutcomeMethodSealMarker.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(root.glob("*.json"))
            )
            if root.is_dir()
            else ()
        )


__all__ = [
    "DEVELOPMENT_ONLY_MANIFEST_CATEGORY",
    "DEVELOPMENT_ONLY_MARKER_CATEGORY",
    "METHOD_BINDING_CATEGORY",
    "METHOD_SEAL_MARKER_CATEGORY",
    "_ExecutionOutcomeArtifactStore",
]
