"""Packed covariance artifacts and lease-scoped readback."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor
from alphalattice.investment.risk_research.contracts import (
    HistoricalCovarianceBuildCheckpoint,
    HistoricalCovarianceChunk,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = NDArray[np.float64]
_PREFIX = "playpen://risk-research/"


class RiskArtifactError(ValueError):
    """Stable artifact, tamper, and lease boundary failure."""


class RiskArtifactStore:
    """Own immutable Risk JSON and packed lower-triangle matrix chunks."""

    def __init__(self, artifact_root: Path) -> None:
        """Bind Risk artifact publication/readback to a resolved caller-owned root.

        Args:
            artifact_root: Root whose risk-research subdirectory holds JSON and packed covariance
                payloads.
        """
        self.root = artifact_root.resolve() / "risk-research"

    @staticmethod
    def _require_hash(value: str) -> None:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise RiskArtifactError("risk_research.artifact_identity_invalid")

    @staticmethod
    def _atomic_write(target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        staged.write_bytes(content)
        os.replace(staged, target)
        staged.unlink(missing_ok=True)

    @staticmethod
    def uri(category: str, content_hash: str) -> str:
        """Construct a path-free Risk category/content address.

        Args:
            category: Risk artifact category retained in the address.
            content_hash: Declared content identity; construction does not validate stored bytes.

        Returns:
            Risk artifact URI for the supplied category and identity.
        """
        return f"{_PREFIX}{category}/{content_hash}"

    def _path(self, category: str, content_hash: str, suffix: str) -> Path:
        self._require_hash(content_hash)
        return self.root / category / f"{content_hash}.{suffix}"

    def publish_json(
        self,
        *,
        category: str,
        payload: Mapping[str, object],
        identity_field: str,
    ) -> ArtifactDescriptor:
        """Publish canonical self-identified Risk JSON without reusing an identity.

        Args:
            category: Caller-selected Risk artifact category.
            payload: Mapping containing its declared self-identity field.
            identity_field: Field naming the canonical hash of all other mapping fields.

        Returns:
            Path-free descriptor with content identity and serialized metadata digest.

        Raises:
            RiskArtifactError: Identity syntax/hash differs or an existing identity has different
                bytes.
        """
        content_hash = str(payload[identity_field])
        self._require_hash(content_hash)
        identity = dict(payload)
        identity.pop(identity_field)
        if canonical_hash(identity) != content_hash:
            raise RiskArtifactError("risk_research.json_identity_invalid")
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        target = self._path(category, content_hash, "json")
        if target.exists() and target.read_bytes() != content:
            raise RiskArtifactError("risk_research.json_identity_reused")
        if not target.exists():
            self._atomic_write(target, content)
        return ArtifactDescriptor(
            kind=f"risk-research-{category.replace('/', '-')}",
            content_hash=content_hash,
            metadata_hash=hashlib.sha256(content).hexdigest(),
            uri=self.uri(category, content_hash),
        )

    def load_json(self, *, category: str, uri: str, identity_field: str) -> dict[str, object]:
        """Read Risk JSON and verify its requested URI and canonical self identity.

        Args:
            category: Exact requested Risk artifact category.
            uri: Declared artifact URI selecting its identity.
            identity_field: Self-identity field to reconcile with the requested hash.

        Returns:
            Verified JSON mapping.

        Raises:
            RiskArtifactError: URI, requested self identity or canonical content differs.
            FileNotFoundError: The requested JSON file is absent.
        """
        prefix = f"{_PREFIX}{category}/"
        if not uri.startswith(prefix):
            raise RiskArtifactError("risk_research.artifact_uri_invalid")
        content_hash = uri[len(prefix) :]
        target = self._path(category, content_hash, "json")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get(identity_field) != content_hash:
            raise RiskArtifactError("risk_research.json_requested_identity_invalid")
        identity = dict(payload)
        identity.pop(identity_field)
        if canonical_hash(identity) != content_hash:
            raise RiskArtifactError("risk_research.json_tampered")
        return payload

    def iter_json(self, *, category: str, identity_field: str) -> Iterator[dict[str, object]]:
        """Yield verified category records in sorted stored-path order.

        Args:
            category: Risk JSON category to enumerate.
            identity_field: Self-identity field required by every record.

        Returns:
            Iterator of verified mappings; an absent category yields no records.

        Raises:
            RiskArtifactError: An enumerated record fails URI/content verification.
        """
        root = self.root / category
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            yield self.load_json(
                category=category,
                uri=self.uri(category, path.stem),
                identity_field=identity_field,
            )

    def publish_covariance_chunk(
        self,
        *,
        formation_sessions: tuple[date, ...],
        matrices: Sequence[FloatArray],
        matrix_hashes: tuple[str, ...],
    ) -> HistoricalCovarianceChunk:
        """Pack finite square covariance lower triangles and publish immutable chunk bytes.

        One to 21 matrices share the first matrix asset count. The content identity binds dates,
        declared matrix identities and the little-endian float64 packed-byte digest; this method
        does not recompute each declared matrix identity.

        Args:
            formation_sessions: One formation date per matrix in declared order.
            matrices: Finite square covariance matrices on a shared asset axis.
            matrix_hashes: Exact declared lowercase hexadecimal matrix identities.

        Returns:
            Typed chunk metadata selecting the packed lower triangles.

        Raises:
            RiskArtifactError: Input axes/counts/hash syntax, shared matrix shape/finiteness or
                existing content identity is invalid.
        """
        if (
            not matrices
            or len(matrices) != len(formation_sessions)
            or len(matrices) != len(matrix_hashes)
            or len(matrices) > 21
        ):
            raise RiskArtifactError("risk_research.covariance_chunk_input_invalid")
        for matrix_hash in matrix_hashes:
            self._require_hash(matrix_hash)
        asset_count = int(matrices[0].shape[0])
        if asset_count == 0:
            raise RiskArtifactError("risk_research.covariance_chunk_axis_invalid")
        lower = np.tril_indices(asset_count)
        packed: FloatArray = np.empty((len(matrices), len(lower[0])), dtype="<f8")
        for index, matrix in enumerate(matrices):
            values = np.asarray(matrix, dtype=np.float64)
            if values.shape != (asset_count, asset_count) or not np.isfinite(values).all():
                raise RiskArtifactError("risk_research.covariance_chunk_matrix_invalid")
            packed[index] = values[lower]
        content = packed.tobytes(order="C")
        packed_bytes_sha256 = hashlib.sha256(content).hexdigest()
        content_hash = canonical_hash(
            {
                "dtype": "little-endian-float64",
                "asset_count": asset_count,
                "formation_sessions": formation_sessions,
                "matrix_hashes": matrix_hashes,
                "packed_bytes_sha256": packed_bytes_sha256,
            }
        )
        target = self._path("covariance/chunks", content_hash, "bin")
        if target.exists() and target.read_bytes() != content:
            raise RiskArtifactError("risk_research.covariance_chunk_identity_reused")
        if not target.exists():
            self._atomic_write(target, content)
        return HistoricalCovarianceChunk(
            formation_sessions=formation_sessions,
            matrix_count=len(matrices),
            asset_count=asset_count,
            packed_value_count=packed.size,
            matrix_hashes=matrix_hashes,
            packed_bytes_sha256=packed_bytes_sha256,
            content_hash=content_hash,
            uri=self.uri("covariance/chunks", content_hash),
        )

    def chunk_path(self, chunk: HistoricalCovarianceChunk) -> Path:
        """Verify packed chunk URI, declared byte length and payload digest.

        Args:
            chunk: Declared historical covariance chunk.

        Returns:
            Verified local packed-file path.

        Raises:
            RiskArtifactError: URI, file existence/length or packed-byte digest differs.
        """
        if chunk.uri != self.uri("covariance/chunks", chunk.content_hash):
            raise RiskArtifactError("risk_research.covariance_chunk_uri_invalid")
        target = self._path("covariance/chunks", chunk.content_hash, "bin")
        expected_bytes = chunk.packed_value_count * 8
        if not target.is_file() or target.stat().st_size != expected_bytes:
            raise RiskArtifactError("risk_research.covariance_chunk_incomplete")
        if hashlib.sha256(target.read_bytes()).hexdigest() != chunk.packed_bytes_sha256:
            raise RiskArtifactError("risk_research.covariance_chunk_tampered")
        return target

    def publish_build_checkpoint(self, checkpoint: HistoricalCovarianceBuildCheckpoint) -> None:
        """Atomically replace a covariance build checkpoint at its program address.

        Checkpoints record resumable state and can be replaced; this operation does not
        independently validate chunk payloads.

        Args:
            checkpoint: Typed checkpoint serialized under its program_hash.
        """
        target = self.root / "covariance" / "builds" / f"{checkpoint.program_hash}.json"
        content = json.dumps(
            checkpoint.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self._atomic_write(target, content)

    def load_build_checkpoint(
        self, program_hash: str
    ) -> HistoricalCovarianceBuildCheckpoint | None:
        """Read a typed build checkpoint and verify every retained covariance chunk.

        Args:
            program_hash: Exact lowercase hexadecimal program identity selecting the checkpoint.

        Returns:
            Verified checkpoint, or None when the checkpoint file is absent.

        Raises:
            RiskArtifactError: Identity syntax, checkpoint contract/program binding or any chunk
                readback fails.
        """
        self._require_hash(program_hash)
        target = self.root / "covariance" / "builds" / f"{program_hash}.json"
        if not target.is_file():
            return None
        try:
            checkpoint = HistoricalCovarianceBuildCheckpoint.model_validate_json(
                target.read_bytes()
            )
            if checkpoint.program_hash != program_hash:
                raise RiskArtifactError("risk_research.covariance_checkpoint_invalid")
            for chunk in checkpoint.chunks:
                self.chunk_path(chunk)
            return cast(HistoricalCovarianceBuildCheckpoint, checkpoint)
        except RiskArtifactError:
            raise
        except Exception as error:
            raise RiskArtifactError("risk_research.covariance_checkpoint_invalid") from error


class CovarianceChunkLease:
    """Own one memmap and return independent read-only matrices."""

    def __init__(self, *, path: Path, chunk: HistoricalCovarianceChunk) -> None:
        """Open one read-only packed covariance mapping with a reusable reconstruction buffer.

        Args:
            path: Verified packed-file path supplied by the reader.
            chunk: Declared matrix count, asset axis and formation dates.
        """
        self.chunk = chunk
        packed_width = chunk.asset_count * (chunk.asset_count + 1) // 2
        self._mapping: FloatArray = np.memmap(
            path,
            dtype="<f8",
            mode="r",
            shape=(chunk.matrix_count, packed_width),
        )
        self._buffer: FloatArray = np.empty(
            (chunk.asset_count, chunk.asset_count), dtype=np.float64
        )
        self._lower = np.tril_indices(chunk.asset_count)
        self._closed = False

    def matrix(self, formation_session: date) -> FloatArray:
        """Reconstruct an independent read-only symmetric matrix for one declared formation.

        Args:
            formation_session: Exact formation date within this chunk.

        Returns:
            Independent non-writeable float64 matrix reconstructed from the stored lower triangle.

        Raises:
            RiskArtifactError: The lease is closed or the formation is absent.
        """
        if self._closed:
            raise RiskArtifactError("risk_research.covariance_lease_closed")
        try:
            index = self.chunk.formation_sessions.index(formation_session)
        except ValueError as error:
            raise RiskArtifactError("risk_research.covariance_formation_unavailable") from error
        self._buffer.setflags(write=True)
        self._buffer[self._lower] = self._mapping[index]
        self._buffer[(self._lower[1], self._lower[0])] = self._mapping[index]
        result = np.array(self._buffer, copy=True)
        result.setflags(write=False)
        return result

    def close(self) -> None:
        """Close this mapping once and prevent further matrix reads."""
        if self._closed:
            return
        self._closed = True
        mapping = self._mapping
        mmap = getattr(mapping, "_mmap", None)
        if mmap is not None:
            mmap.close()


class HistoricalCovarianceReader:
    """Open at most one immutable covariance chunk at a time."""

    def __init__(self, artifact_root: Path) -> None:
        """Bind covariance readback to one artifact store and a single active lease.

        Args:
            artifact_root: Caller-owned root containing retained Risk artifacts.
        """
        self.store = RiskArtifactStore(artifact_root)
        self._active = False

    @contextmanager
    def lease(self, chunk: HistoricalCovarianceChunk) -> Iterator[CovarianceChunkLease]:
        """Yield one verified covariance chunk lease and close it on scope exit.

        Args:
            chunk: Exact chunk declaration whose URI/length/digest is verified before mapping.

        Returns:
            Context manager yielding the active covariance lease.

        Raises:
            RiskArtifactError: Another lease is active or chunk verification fails.
        """
        if self._active:
            raise RiskArtifactError("risk_research.covariance_lease_already_active")
        self._active = True
        lease = CovarianceChunkLease(path=self.store.chunk_path(chunk), chunk=chunk)
        try:
            yield lease
        finally:
            lease.close()
            self._active = False


__all__ = [
    "CovarianceChunkLease",
    "HistoricalCovarianceReader",
    "RiskArtifactError",
    "RiskArtifactStore",
]
