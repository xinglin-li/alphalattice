"""Packed covariance artifacts and lease-scoped readback."""

from __future__ import annotations

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

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.bounded_memo import stage_digest
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.domain_contracts.sealing import scoped_payload_digest
from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor
from alphalattice.foundation.data_layer import history, risk_history
from alphalattice.foundation.market_data_ops.storage.world import world_listing
from alphalattice.investment.risk_research.contracts import (
    HistoricalCovarianceBuildCheckpoint,
    HistoricalCovarianceChunk,
)
from alphalattice.kernel.identity import integrity_sha256

type FloatArray = NDArray[np.float64]
_PREFIX = "playpen://risk-research/"


def _packed_digest(content: bytes | memoryview) -> str:
    return integrity_sha256(content).hexdigest()


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
        # Once per stage for these exact bytes; the record its reader builds answers the same slot.
        if scoped_payload_digest(identity, identity_field, content_hash, digest=canonical_hash) != (
            content_hash
        ):
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
            metadata_hash=integrity_sha256(content).hexdigest(),
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
        # Once per stage for these exact bytes; the record its reader builds answers the same slot.
        if scoped_payload_digest(identity, identity_field, content_hash, digest=canonical_hash) != (
            content_hash
        ):
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
        ordered_listing_ids: tuple[str, ...] | None = None,
    ) -> HistoricalCovarianceChunk:
        """Pack finite square covariance lower triangles and publish immutable chunk bytes.

        One to 21 matrices share the first matrix asset count. The content identity binds dates,
        declared matrix identities and the little-endian float64 packed-byte digest; this method
        does not recompute each declared matrix identity.

        Args:
            formation_sessions: One formation date per matrix in declared order.
            matrices: Finite square covariance matrices on a shared asset axis.
            matrix_hashes: Exact declared lowercase hexadecimal matrix identities.
            ordered_listing_ids: The market listing axis for catalog admission.

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
        packed_bytes_sha256 = _packed_digest(content)
        content_hash = canonical_hash(
            {
                "dtype": "little-endian-float64",
                "asset_count": asset_count,
                "formation_sessions": formation_sessions,
                "matrix_hashes": matrix_hashes,
                "packed_bytes_sha256": packed_bytes_sha256,
            }
        )
        chunk = HistoricalCovarianceChunk(
            formation_sessions=formation_sessions,
            matrix_count=len(matrices),
            asset_count=asset_count,
            packed_value_count=packed.size,
            matrix_hashes=matrix_hashes,
            packed_bytes_sha256=packed_bytes_sha256,
            content_hash=content_hash,
            uri=self.uri("covariance/chunks", content_hash),
        )
        if history.recording():
            self.admit_covariance(chunk, ordered_listing_ids, packed)
        else:  # a writer outside the Host keeps the packed file, admitted when it resumes there
            target = self._path("covariance/chunks", content_hash, "bin")
            if target.exists() and target.read_bytes() != content:
                raise RiskArtifactError("risk_research.covariance_chunk_identity_reused")
            if not target.exists():
                self._atomic_write(target, content)
        return chunk

    @staticmethod
    def coordinate_axis(
        chunk: HistoricalCovarianceChunk, listing_ids: tuple[str, ...] | None
    ) -> tuple[str, ...]:
        """Resolve the sealed Risk axis into its registered world listing identities."""
        if (
            listing_ids is None
            or len(listing_ids) != chunk.asset_count
            or len(set(listing_ids)) != chunk.asset_count
            or len(set(chunk.formation_sessions)) != chunk.matrix_count
        ):
            raise RiskArtifactError("risk_research.covariance_chunk_axis_invalid")
        return tuple(world_listing(item) for item in listing_ids)

    @staticmethod
    def coordinate_key(listings: tuple[str, ...]) -> str:
        """Name the exact ordered listing axis used to resolve covariance profiles: once per
        stage for one axis, which every chunk of a covariance history shares."""
        return stage_digest(("risk-coordinate-axis", listings), listings, canonical_hash)

    def admit_covariance(
        self,
        chunk: HistoricalCovarianceChunk,
        ordered_listing_ids: tuple[str, ...] | None,
        packed: FloatArray,
        *,
        workspace_id: str | None = None,
    ) -> None:
        """Admit verified packed values with the original formation and listing axes."""
        listings = self.coordinate_axis(chunk, ordered_listing_ids)
        try:
            risk_history.write_risk_covariance(
                chunk.content_hash,
                self.coordinate_key(listings),
                tuple(day.isoformat() for day in chunk.formation_sessions),
                chunk.matrix_hashes,
                listings,
                packed,
                workspace_id=workspace_id,
            )
        except DataError as error:
            raise RiskArtifactError(str(error.failure.code)) from error

    def continue_covariance(
        self, lease: CovarianceChunkLease, ordered_listing_ids: tuple[str, ...]
    ) -> None:
        """Move a validated legacy checkpoint chunk when its writer resumes on the Host."""
        if isinstance(lease._mapping, np.memmap) and history.recording():
            self.admit_covariance(lease.chunk, ordered_listing_ids, lease._mapping)

    def catalog_covariance(
        self,
        chunk: HistoricalCovarianceChunk,
        ordered_listing_ids: tuple[str, ...] | None,
        *,
        root: Path | None = None,
        retry_unadmitted: bool = False,
    ) -> FloatArray | None:
        """Read the catalog coordinates and prove the original sealed packed bytes."""
        if chunk.uri != self.uri("covariance/chunks", chunk.content_hash):
            raise RiskArtifactError("risk_research.covariance_chunk_uri_invalid")
        listings = self.coordinate_axis(chunk, ordered_listing_ids)
        try:
            packed = risk_history.read_risk_covariance(
                chunk.content_hash,
                self.coordinate_key(listings),
                tuple(day.isoformat() for day in chunk.formation_sessions),
                chunk.matrix_hashes,
                listings,
                root=root,
                retry_unadmitted=retry_unadmitted,
            )
        except DataError as error:
            raise RiskArtifactError(str(error.failure.code)) from error
        if packed is None:
            return None
        if (
            _packed_digest(memoryview(packed.astype("<f8", copy=False)))
            != chunk.packed_bytes_sha256
        ):
            raise RiskArtifactError("risk_research.covariance_chunk_tampered")
        return cast(FloatArray, packed)

    def packed_covariance(
        self,
        chunk: HistoricalCovarianceChunk,
        ordered_listing_ids: tuple[str, ...] | None,
        *,
        resume: bool = False,
    ) -> FloatArray:
        """Read the sealed floats from the catalog, or from the packed file a writer outside the
        Host kept, or a restored workspace keeps until its listings arrive, before admission."""
        if ordered_listing_ids is not None:
            packed = self.catalog_covariance(chunk, ordered_listing_ids, retry_unadmitted=resume)
            if packed is not None:
                return packed
        return self.legacy_covariance(chunk)

    def legacy_covariance(self, chunk: HistoricalCovarianceChunk) -> FloatArray:
        """Map the packed file of a chunk not yet admitted: the upgrade's input, and the read
        before admission."""
        return np.memmap(
            self.chunk_path(chunk),
            dtype="<f8",
            mode="r",
            shape=(chunk.matrix_count, chunk.packed_value_count // chunk.matrix_count),
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
        if _packed_digest(target.read_bytes()) != chunk.packed_bytes_sha256:
            raise RiskArtifactError("risk_research.covariance_chunk_tampered")
        return target

    def load_build_checkpoint(
        self, program_hash: str, *, ordered_listing_ids: tuple[str, ...] | None = None
    ) -> HistoricalCovarianceBuildCheckpoint | None:
        """Read a typed build checkpoint and verify every retained covariance chunk.

        Args:
            program_hash: Exact lowercase hexadecimal program identity selecting the checkpoint.
            ordered_listing_ids: Listing axis shared by the checkpoint's chunks.

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
                with HistoricalCovarianceReader(self.root.parent).lease(
                    chunk, ordered_listing_ids=ordered_listing_ids
                ):
                    pass
            return cast(HistoricalCovarianceBuildCheckpoint, checkpoint)
        except RiskArtifactError:
            raise
        except Exception as error:
            raise RiskArtifactError("risk_research.covariance_checkpoint_invalid") from error


class CovarianceChunkLease:
    """Own one verified packed projection and return independent read-only matrices."""

    def __init__(self, *, packed: FloatArray, chunk: HistoricalCovarianceChunk) -> None:
        """Open one read-only packed covariance mapping with a reusable reconstruction buffer.

        Args:
            packed: Verified packed entries supplied by the reader.
            chunk: Declared matrix count, asset axis and formation dates.
        """
        self.chunk = chunk
        self._mapping = packed
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
    def lease(
        self,
        chunk: HistoricalCovarianceChunk,
        *,
        ordered_listing_ids: tuple[str, ...] | None = None,
        resume: bool = False,
    ) -> Iterator[CovarianceChunkLease]:
        """Yield one verified covariance chunk lease and close it on scope exit.

        Args:
            chunk: Exact chunk declaration whose URI, length and digest are verified on opening.
            ordered_listing_ids: The owning surface's listing axis.
            resume: Allow a verified packed file to complete a profile-only checkpoint.

        Returns:
            Context manager yielding the active covariance lease.

        Raises:
            RiskArtifactError: Another lease is active or chunk verification fails.
        """
        if self._active:
            raise RiskArtifactError("risk_research.covariance_lease_already_active")
        packed = self.store.packed_covariance(chunk, ordered_listing_ids, resume=resume)
        lease = CovarianceChunkLease(packed=packed, chunk=chunk)
        self._active = True
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
