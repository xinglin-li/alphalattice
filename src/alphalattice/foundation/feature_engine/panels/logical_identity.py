"""Logical/physical Feature Panel identity publication without data duplication."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
    ordered_key_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    PanelDerivationRecipe,
    PanelRetentionAssessment,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
)
from alphalattice.foundation.feature_engine.panels.logical_contracts import (
    LegacyPanelLogicalEquivalenceAttestation,
    PanelDerivationBinding,
    PanelLogicalAnnualContent,
    PanelLogicalContentManifest,
    PanelLogicalFactorDigest,
    PanelLogicalMembershipEpoch,
    PanelLogicalRevision,
    PanelPhysicalChunk,
    PanelPhysicalMaterializationManifest,
    PanelPublicationMarker,
    identity_payload,
)
from alphalattice.foundation.feature_engine.panels.logical_semantic_index import (
    FeaturePanelLogicalSemanticIndex,
    FeaturePanelLogicalSessionSemantic,
    build_logical_semantic_index,
)
from alphalattice.kernel.shared_kernel.environment import package_versions
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_Model = TypeVar("_Model", bound=BaseModel)
_RESERVED_COLUMNS = frozenset(
    {
        "manifest_revision",
        "sector_revision",
        "catalog_hash",
        "policy_hash",
        "cross_section_identity",
        "session_date",
        "listing_id",
        "row_hash",
        "materialization_receipt_hash",
    }
)


class PanelLogicalArtifactStore:
    """Own logical identity artifacts and tiny operational mapping pointers."""

    def __init__(self, closure_store: PanelClosureArtifactStore) -> None:
        """Place logical artifacts beside the immutable Panel closure store."""
        self._store = closure_store
        self._operational = closure_store.root / "logical" / "operational"

    @property
    def closure_store(self) -> PanelClosureArtifactStore:
        """Return the underlying immutable closure artifact store."""
        return self._store

    def publish_model(self, category: str, identity: str, model: BaseModel) -> None:
        """Publish a logical model and verify its content by reading it back."""
        self._store.publish_json(
            category=f"logical/{category}",
            content_hash=identity,
            payload=model.model_dump(mode="json"),
        )
        loaded = self.load_model(category, identity, type(model))
        if loaded != model:
            raise ValueError("feature_panel.logical_content_mismatch")

    def load_model(self, category: str, identity: str, model: type[_Model]) -> _Model:
        """Load a named logical artifact as its declared model."""
        return self._store.load_model(
            category=f"logical/{category}", content_hash=identity, model=model
        )

    def marker_for_snapshot(self, snapshot_hash: str) -> PanelPublicationMarker | None:
        """Resolve a legacy snapshot to its sealed logical publication marker."""
        pointer = self._mapping_path(snapshot_hash)
        if not pointer.exists():
            return None
        payload = self._read_pointer(pointer)
        marker = self.load_model(
            "publication-markers",
            str(payload["marker_hash"]),
            PanelPublicationMarker,
        )
        if marker.legacy_snapshot_hash != snapshot_hash:
            raise ValueError("feature_panel.logical_content_mismatch")
        return marker

    def publish_snapshot_mapping(self, marker: PanelPublicationMarker) -> None:
        """Publish an immutable snapshot-to-marker mapping pointer."""
        pointer = self._mapping_path(marker.legacy_snapshot_hash)
        if pointer.exists():
            current = self._read_pointer(pointer)
            if current.get("marker_hash") != marker.marker_hash:
                raise ValueError("feature_panel.logical_content_mismatch")
            return
        self._write_pointer(pointer, {"marker_hash": marker.marker_hash})

    def current_marker(self) -> PanelPublicationMarker | None:
        """Read the current operational logical publication marker."""
        pointer = self._operational / "current.json"
        if not pointer.exists():
            return None
        payload = self._read_pointer(pointer)
        return self.load_model(
            "publication-markers", str(payload["marker_hash"]), PanelPublicationMarker
        )

    def compare_and_swap_current(
        self, *, expected_marker_hash: str | None, next_marker: PanelPublicationMarker
    ) -> None:
        """Advance the current marker only from the expected identity."""
        pointer = self._operational / "current.json"
        current = self._read_pointer(pointer).get("marker_hash") if pointer.exists() else None
        if current != expected_marker_hash:
            if current == next_marker.marker_hash:
                return
            raise ValueError("feature_panel.identity_migration_cas_failed")
        self._write_pointer(pointer, {"marker_hash": next_marker.marker_hash})

    def restore_current(
        self,
        *,
        expected_marker_hash: str,
        replacement_marker_hash: str | None,
    ) -> None:
        """Restore the captured operational pointer after a failed HEAD cutover."""
        pointer = self._operational / "current.json"
        current = self._read_pointer(pointer).get("marker_hash") if pointer.exists() else None
        if current != expected_marker_hash:
            raise ValueError("feature_panel.identity_migration_cas_failed")
        if replacement_marker_hash is None:
            pointer.unlink(missing_ok=True)
            return
        marker = self.load_model(
            "publication-markers", replacement_marker_hash, PanelPublicationMarker
        )
        self._write_pointer(pointer, {"marker_hash": marker.marker_hash})

    def _mapping_path(self, snapshot_hash: str) -> Path:
        _validate_hash(snapshot_hash)
        return self._operational / "snapshot-mappings" / f"{snapshot_hash}.json"

    @staticmethod
    def _read_pointer(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("feature Panel logical identity pointer is invalid")
        return payload

    @staticmethod
    def _write_pointer(path: Path, payload: Mapping[str, object]) -> None:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_bytes(serialized)
        os.replace(staged, path)


class PanelLogicalIdentityPublisher:
    """Map one immutable annual-wide Panel into logical and physical authorities."""

    def __init__(
        self,
        *,
        resolver: ArtifactResolver,
        store: PanelLogicalArtifactStore,
        closure_ledger: FeatureClosureLedger | None = None,
    ) -> None:
        """Bind logical artifacts, resolver, and optional recovery ledger."""
        self._resolver = resolver
        self._store = store
        self._closure_ledger = closure_ledger

    def publish_snapshot(
        self, manifest_ref: str, *, published_at: datetime | None = None
    ) -> PanelPublicationMarker:
        """Publish logical and physical authorities for a sealed Panel snapshot."""
        manifest = self._resolver.load_feature_panel_manifest(manifest_ref)
        snapshot_hash = str(manifest["snapshot_hash"])
        existing = self._store.marker_for_snapshot(snapshot_hash)
        if existing is not None:
            return existing
        observed = published_at or datetime.now(UTC)
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError("Panel logical publication clock must be timezone-aware")
        built = self._build_logical_authorities(manifest)
        logical_manifest, revision, semantic_index, physical = built
        derivation = self._build_derivation(manifest=manifest, revision=revision)
        self._store.publish_model(
            "content-manifests", logical_manifest.logical_manifest_hash, logical_manifest
        )
        self._store.publish_model("revisions", revision.logical_panel_hash, revision)
        self._store.publish_model("semantic-indexes", semantic_index.index_hash, semantic_index)
        self._store.publish_model(
            "derivation-bindings", derivation.derivation_binding_hash, derivation
        )
        self._store.publish_model(
            "physical-materializations",
            physical.physical_materialization_hash,
            physical,
        )
        retention = (
            "REMATERIALIZATION_VERIFIED_CURRENT_ENVIRONMENT"
            if derivation.recovery_disposition == "RECOVERY_CLOSURE_VERIFIED"
            else "PHYSICAL_RETENTION_REQUIRED"
        )
        marker = _identified(
            PanelPublicationMarker,
            {
                "legacy_snapshot_hash": snapshot_hash,
                "logical_panel_hash": revision.logical_panel_hash,
                "derivation_binding_hash": derivation.derivation_binding_hash,
                "physical_materialization_hash": physical.physical_materialization_hash,
                "logical_semantic_index_hash": semantic_index.index_hash,
                "retention_disposition": retention,
                "published_at": observed.astimezone(UTC),
            },
            "marker_hash",
        )
        self._store.publish_model("publication-markers", marker.marker_hash, marker)
        self._validate_publication(marker)
        self._store.publish_snapshot_mapping(marker)
        return marker

    def semantic_index(self, marker: PanelPublicationMarker) -> FeaturePanelLogicalSemanticIndex:
        """Load the logical semantic index named by a publication marker."""
        return self._store.load_model(
            "semantic-indexes",
            marker.logical_semantic_index_hash,
            FeaturePanelLogicalSemanticIndex,
        )

    def verify_snapshot(self, snapshot_hash: str) -> PanelPublicationMarker:
        """Read and validate the named publication without creating a mapping."""
        marker = self._store.marker_for_snapshot(snapshot_hash)
        if marker is None:
            raise ValueError("research_authoring.panel_logical_identity_unavailable")
        self._validate_publication(marker)
        self.semantic_index(marker)
        return marker

    def publish_equivalence_attestation(
        self, marker: PanelPublicationMarker
    ) -> LegacyPanelLogicalEquivalenceAttestation:
        """Attest the exact physical rows used to derive a native logical authority."""
        derivation = self._store.load_model(
            "derivation-bindings",
            marker.derivation_binding_hash,
            PanelDerivationBinding,
        )
        revision = self._store.load_model(
            "revisions", marker.logical_panel_hash, PanelLogicalRevision
        )
        manifest = self._store.load_model(
            "content-manifests",
            revision.logical_manifest_hash,
            PanelLogicalContentManifest,
        )
        annual_hashes = tuple(
            canonical_hash(item.model_dump(mode="json")) for item in manifest.annual_content
        )
        attestation = _identified(
            LegacyPanelLogicalEquivalenceAttestation,
            {
                "legacy_snapshot_hash": marker.legacy_snapshot_hash,
                "legacy_panel_content_hash": derivation.legacy_panel_content_hash,
                "logical_panel_hash": marker.logical_panel_hash,
                "axes_equal": True,
                "ieee_values_and_null_masks_equal": True,
                "availability_semantics_equal": True,
                "verified_annual_hashes": annual_hashes,
            },
            "attestation_hash",
        )
        self._store.publish_model(
            "equivalence-attestations", attestation.attestation_hash, attestation
        )
        return attestation

    def _build_logical_authorities(
        self, manifest: Mapping[str, object]
    ) -> tuple[
        PanelLogicalContentManifest,
        PanelLogicalRevision,
        FeaturePanelLogicalSemanticIndex,
        PanelPhysicalMaterializationManifest,
    ]:
        chunks = manifest.get("chunks")
        summary = manifest.get("safe_summary")
        if not isinstance(chunks, list) or not chunks or not isinstance(summary, dict):
            raise ValueError("feature_panel.logical_content_mismatch")
        lineage = summary.get("lineage")
        factor_summary = summary.get("factor_catalog_summary")
        if not isinstance(lineage, dict) or not isinstance(factor_summary, dict):
            raise ValueError("feature_panel.logical_content_mismatch")
        annual: list[PanelLogicalAnnualContent] = []
        physical_chunks: list[PanelPhysicalChunk] = []
        session_semantics: list[FeaturePanelLogicalSessionSemantic] = []
        all_sessions: list[date] = []
        factor_ids: tuple[str, ...] | None = None
        listing_ids: tuple[str, ...] | None = None
        # A Panel whose sessions hold their own members records its epochs in
        # the manifest; its rows are then judged session by session against
        # them, and its listing axis is the union the rows hold.
        membership_epochs = _membership_epochs(summary)
        axis: set[str] = set()
        ordered_chunks = sorted(
            chunks,
            key=lambda item: int(str(cast(dict[str, object], item)["year"])),
        )
        for raw_chunk in ordered_chunks:
            if not isinstance(raw_chunk, dict):
                raise ValueError("feature_panel.logical_content_mismatch")
            path = self._resolver.resolve_feature_panel_chunk_ref(
                uri=str(raw_chunk["uri"]),
                content_hash=str(raw_chunk["chunk_hash"]),
                metadata_hash=str(raw_chunk["metadata_hash"]),
            )
            table = pq.read_table(path).combine_chunks()
            observed_factors = tuple(
                name for name in table.column_names if name not in _RESERVED_COLUMNS
            )
            # ``factor_catalog_summary`` is a user-safe projection.  Some legacy
            # quarantined snapshots contain additional physical factor columns
            # that were no longer active when the summary was rendered.  The
            # Parquet schema is the exact machine axis authority; the summary
            # may only prove that every factor it names is present.
            if not {str(value) for value in factor_summary}.issubset(observed_factors):
                raise ValueError("feature_panel.logical_content_mismatch")
            if factor_ids is None:
                factor_ids = observed_factors
            elif observed_factors != factor_ids:
                raise ValueError("feature_panel.logical_content_mismatch")
            built = _build_annual_logical_content(
                table=table,
                year=int(raw_chunk["year"]),
                factor_ids=factor_ids,
                expected_listing_ids=listing_ids if membership_epochs is None else None,
                membership_epochs=membership_epochs,
            )
            annual_item, observed_listings, observed_sessions = built
            if listing_ids is None:
                listing_ids = observed_listings
            axis.update(observed_listings)
            annual.append(annual_item)
            all_sessions.extend(item.session_date for item in observed_sessions)
            session_semantics.extend(observed_sessions)
            physical_chunks.append(
                _physical_chunk(
                    path=path,
                    raw_chunk=raw_chunk,
                    annual=annual_item,
                    schema_hash=str(manifest["schema_hash"]),
                )
            )
        if factor_ids is None or listing_ids is None:
            raise ValueError("feature_panel.logical_content_mismatch")
        if membership_epochs is not None:
            listing_ids = tuple(sorted(axis))
        sessions = tuple(all_sessions)
        availability_semantics_hash = canonical_hash(
            {
                "active_listing_count": int(str(manifest["active_listing_count"])),
                "availability_count": int(summary.get("availability_count", 0)),
                "factor_catalog_summary": factor_summary,
            }
        )
        logical_values: dict[str, object] = {
            "sessions": sessions,
            "listing_ids": listing_ids,
            "factor_ids": factor_ids,
            "annual_content": tuple(annual),
            "availability_semantics_hash": availability_semantics_hash,
        }
        if membership_epochs is not None:
            logical_values["membership_epochs"] = membership_epochs
        logical_manifest = _identified(
            PanelLogicalContentManifest, logical_values, "logical_manifest_hash"
        )
        listing_set_hash = canonical_hash(listing_ids)
        if listing_set_hash != str(manifest["listing_set_hash"]):
            raise ValueError("feature_panel.logical_content_mismatch")
        revision = _identified(
            PanelLogicalRevision,
            {
                "logical_manifest_hash": logical_manifest.logical_manifest_hash,
                "catalog_semantics_hash": str(lineage["catalog_hash"]),
                "cross_section_policy_hash": str(lineage["policy_hash"]),
                "listing_set_hash": listing_set_hash,
                "calendar_hash": canonical_hash(sessions),
            },
            "logical_panel_hash",
        )
        semantic_index = build_logical_semantic_index(
            logical_panel_hash=revision.logical_panel_hash,
            catalog_semantics_hash=revision.catalog_semantics_hash,
            listing_set_hash=listing_set_hash,
            factor_ids=factor_ids,
            active_listing_count=len(listing_ids),
            sessions=tuple(session_semantics),
            membership_epochs=membership_epochs,
        )
        physical = _identified(
            PanelPhysicalMaterializationManifest,
            {
                "logical_panel_hash": revision.logical_panel_hash,
                "physical_chunks": tuple(physical_chunks),
                # How these bytes were encoded: a fact of the record this hash
                # addresses, compared nowhere (LAWS.md ID6).
                "encoder_provenance": package_versions(("pyarrow",)),
            },
            "physical_materialization_hash",
        )
        return logical_manifest, revision, semantic_index, physical

    def _build_derivation(
        self, *, manifest: Mapping[str, object], revision: PanelLogicalRevision
    ) -> PanelDerivationBinding:
        summary = cast(dict[str, object], manifest["safe_summary"])
        lineage = cast(dict[str, object], summary["lineage"])
        snapshot_hash = str(manifest["snapshot_hash"])
        recipe = self._recipe_for_snapshot(snapshot_hash)
        recovery_binding = (
            self._closure_ledger.panel_binding(snapshot_hash)
            if self._closure_ledger is not None
            else None
        )
        if recovery_binding is not None and (
            recovery_binding.catalog_hash != str(lineage["catalog_hash"])
            or recovery_binding.sector_map_hash
            != (recipe.sector_map_hash if recipe is not None else recovery_binding.sector_map_hash)
        ):
            raise ValueError("feature_panel.logical_content_mismatch")
        return _identified(
            PanelDerivationBinding,
            {
                "legacy_snapshot_hash": snapshot_hash,
                "legacy_panel_content_hash": str(manifest["panel_content_hash"]),
                "legacy_panel_binding_hash": str(manifest["panel_binding_hash"]),
                "logical_panel_hash": revision.logical_panel_hash,
                "manifest_revision": str(lineage["manifest_revision"]),
                "sector_revision": str(lineage["sector_revision"]),
                "sector_map_hash": recipe.sector_map_hash if recipe is not None else None,
                "catalog_hash": str(lineage["catalog_hash"]),
                "policy_hash": str(lineage["policy_hash"]),
                "spy_revision": str(lineage["spy_revision"]),
                "temporal_identity_hash": str(manifest["temporal_identity_hash"]),
                # The closure ledger's head is decided by how many listings each transition
                # writes, an execution parameter, so it stays out of the Panel's identity; the
                # ledger keeps it by snapshot (V330). A binding published before keeps its own.
                "closure_head_hash": None,
                "recovery_recipe_hash": recipe.recipe_hash if recipe is not None else None,
                "recovery_disposition": (
                    "RECOVERY_CLOSURE_VERIFIED"
                    if recipe is not None
                    else "PHYSICAL_RETENTION_REQUIRED"
                ),
            },
            "derivation_binding_hash",
        )

    def _recipe_for_snapshot(self, snapshot_hash: str) -> PanelDerivationRecipe | None:
        assessments = self._store.closure_store.root / "retention-assessments"
        matches: set[str] = set()
        if assessments.is_dir():
            for path in assessments.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("snapshot_hash") != snapshot_hash:
                    continue
                assessment = PanelRetentionAssessment.model_validate(payload)
                if (
                    assessment.disposition == "REMATERIALIZATION_VERIFIED_CURRENT_ENVIRONMENT"
                    and assessment.recipe_hash is not None
                ):
                    matches.add(assessment.recipe_hash)
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("feature_panel.logical_content_mismatch")
        return self._store.closure_store.load_model(
            category="recipes", content_hash=matches.pop(), model=PanelDerivationRecipe
        )

    def _validate_publication(self, marker: PanelPublicationMarker) -> None:
        revision = self._store.load_model(
            "revisions", marker.logical_panel_hash, PanelLogicalRevision
        )
        manifest = self._store.load_model(
            "content-manifests",
            revision.logical_manifest_hash,
            PanelLogicalContentManifest,
        )
        derivation = self._store.load_model(
            "derivation-bindings",
            marker.derivation_binding_hash,
            PanelDerivationBinding,
        )
        physical = self._store.load_model(
            "physical-materializations",
            marker.physical_materialization_hash,
            PanelPhysicalMaterializationManifest,
        )
        semantic = self.semantic_index(marker)
        if (
            derivation.logical_panel_hash != revision.logical_panel_hash
            or physical.logical_panel_hash != revision.logical_panel_hash
            or semantic.logical_panel_hash != revision.logical_panel_hash
            or manifest.logical_manifest_hash != revision.logical_manifest_hash
        ):
            raise ValueError("feature_panel.logical_content_mismatch")


def _membership_epochs(
    summary: Mapping[str, object],
) -> tuple[PanelLogicalMembershipEpoch, ...] | None:
    """The membership epochs a manifest records, or None for a dense Panel."""
    lineage = summary.get("lineage")
    basis = lineage.get("row_identity_basis") if isinstance(lineage, Mapping) else None
    membership = summary.get("membership")
    if basis is None:
        return None
    if not isinstance(membership, Mapping) or not isinstance(membership.get("epochs"), list):
        raise ValueError("feature_panel.logical_content_mismatch")
    epochs = tuple(
        PanelLogicalMembershipEpoch(
            first_session=date.fromisoformat(str(entry["first_session"])),
            last_session=date.fromisoformat(str(entry["last_session"])),
            member_count=int(str(entry["member_count"])),
            membership_hash=str(entry["membership_hash"]),
        )
        for entry in (
            cast(Mapping[str, object], item) for item in cast(list[object], membership["epochs"])
        )
    )
    if not epochs:
        raise ValueError("feature_panel.logical_content_mismatch")
    return epochs


def _build_annual_logical_content(
    *,
    table: pa.Table,
    year: int,
    factor_ids: tuple[str, ...],
    expected_listing_ids: tuple[str, ...] | None,
    membership_epochs: tuple[PanelLogicalMembershipEpoch, ...] | None = None,
) -> tuple[
    PanelLogicalAnnualContent,
    tuple[str, ...],
    tuple[FeaturePanelLogicalSessionSemantic, ...],
]:
    session_values = tuple(table["session_date"].to_pylist())
    listing_values = tuple(str(value) for value in table["listing_id"].to_pylist())
    if not session_values or any(value.year != year for value in session_values):
        raise ValueError("feature_panel.logical_content_mismatch")
    sessions = tuple(dict.fromkeys(session_values))
    if sessions != tuple(sorted(set(sessions))):
        raise ValueError("feature_panel.logical_content_mismatch")
    # Each session's rows, in physical order. A dense Panel holds the same
    # listing block on every session; one with per-session membership holds
    # each session's members, as many as its epoch records, sorted.
    blocks: list[tuple[int, int]] = []
    start = 0
    counts = Counter(session_values)
    for session in sessions:
        stop = start + counts[session]
        blocks.append((start, stop))
        start = stop
    if membership_epochs is None:
        listing_ids = listing_values[blocks[0][0] : blocks[0][1]]
        if expected_listing_ids is not None and listing_ids != expected_listing_ids:
            raise ValueError("feature_panel.logical_content_mismatch")
        for block_start, block_stop in blocks:
            if listing_values[block_start:block_stop] != listing_ids:
                raise ValueError("feature_panel.logical_content_mismatch")
        if len(session_values) != len(sessions) * len(listing_ids):
            raise ValueError("feature_panel.logical_content_mismatch")
    else:
        for session, (block_start, block_stop) in zip(sessions, blocks, strict=True):
            block = listing_values[block_start:block_stop]
            expected = next(
                (
                    epoch.member_count
                    for epoch in membership_epochs
                    if epoch.first_session <= session <= epoch.last_session
                ),
                None,
            )
            if expected is None or len(block) != expected or block != tuple(sorted(set(block))):
                raise ValueError("feature_panel.logical_content_mismatch")
        listing_ids = tuple(sorted(set(listing_values)))
    key_hash = ordered_key_hash(session_values, listing_values)
    masks: list[np.ndarray] = []
    bits: list[np.ndarray] = []
    factor_digests: list[PanelLogicalFactorDigest] = []
    for factor_id in factor_ids:
        mask, column_bits = _column_mask_and_bits(table[factor_id])
        masks.append(mask)
        bits.append(column_bits)
        factor_digests.append(
            PanelLogicalFactorDigest(
                factor_id=factor_id,
                value_null_digest=_logical_column_digest(
                    factor_id=factor_id,
                    key_hash=key_hash,
                    valid=mask,
                    bits=column_bits,
                ),
            )
        )
    mask_matrix = np.column_stack(masks).astype(np.uint8, copy=False)
    bits_matrix = np.column_stack(bits).astype("<u8", copy=False)
    factor_axis_hash = canonical_hash(factor_ids)
    session_semantics: list[FeaturePanelLogicalSessionSemantic] = []
    for session, (block_start, block_stop) in zip(sessions, blocks, strict=True):
        session_semantics.append(
            FeaturePanelLogicalSessionSemantic(
                session_date=session,
                row_count=block_stop - block_start,
                ordered_logical_row_digest=logical_session_digest(
                    session=session,
                    listing_ids=listing_values[block_start:block_stop],
                    factor_ids=factor_ids,
                    valid_matrix=mask_matrix[block_start:block_stop],
                    bits_matrix=bits_matrix[block_start:block_stop],
                    factor_axis_hash=factor_axis_hash,
                ),
            )
        )
    annual = PanelLogicalAnnualContent(
        year=year,
        first_session=sessions[0],
        last_session=sessions[-1],
        session_count=len(sessions),
        row_count=table.num_rows,
        ordered_key_hash=key_hash,
        factor_digests=tuple(factor_digests),
    )
    return annual, listing_ids, tuple(session_semantics)


def _column_mask_and_bits(column: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    array = column.combine_chunks()
    if not pa.types.is_floating(array.type):
        raise ValueError("logical Panel factor column is not floating point")
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype="<f8")
    valid = np.asarray(array.is_valid().to_numpy(zero_copy_only=False), dtype=np.bool_)
    bits = np.ascontiguousarray(values).view("<u8").copy()
    bits[~valid] = 0
    return valid, bits


def _logical_column_digest(
    *, factor_id: str, key_hash: str, valid: np.ndarray, bits: np.ndarray
) -> str:
    digest = hashlib.sha256(b"PanelLogicalValueColumn\0")
    digest.update(factor_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(key_hash.encode("ascii"))
    digest.update(struct.pack(">Q", len(bits)))
    digest.update(np.ascontiguousarray(valid.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(bits.astype("<u8", copy=False)).tobytes())
    return digest.hexdigest()


def logical_session_digest(
    *,
    session: date,
    listing_ids: tuple[str, ...],
    factor_ids: tuple[str, ...],
    valid_matrix: np.ndarray,
    bits_matrix: np.ndarray,
    factor_axis_hash: str | None = None,
) -> str:
    """Hash one ordered logical session independently of physical row encoding."""
    valid = np.asarray(valid_matrix, dtype=np.uint8)
    bits = np.asarray(bits_matrix, dtype="<u8")
    expected_shape = (len(listing_ids), len(factor_ids))
    if valid.shape != expected_shape or bits.shape != expected_shape:
        raise ValueError("logical Panel session matrix differs from its axes")
    axis = (factor_axis_hash or canonical_hash(factor_ids)).encode("ascii")
    session_text = session.isoformat().encode("ascii")
    digest = hashlib.sha256(b"FeaturePanelLogicalSession\0")
    digest.update(session_text)
    digest.update(struct.pack(">I", len(listing_ids)))
    # Each row's hash is the same bytes in the same order as one update per field: its
    # session's prefix hashed once and copied, and its validity and bits sliced from the
    # session's matrices, row by row as they lie (V92).
    prefix = hashlib.sha256(b"FeaturePanelLogicalRow\0" + session_text)
    valid_cells = np.ascontiguousarray(valid)
    bits_cells = np.ascontiguousarray(bits)
    valid_width = valid_cells.itemsize * len(factor_ids)
    bits_width = bits_cells.itemsize * len(factor_ids)
    valid_rows = valid_cells.tobytes()
    bits_rows = bits_cells.tobytes()
    for row_index, listing_id in enumerate(listing_ids):
        listing = listing_id.encode("utf-8")
        row = prefix.copy()
        row.update(
            struct.pack(">I", len(listing))
            + listing
            + axis
            + valid_rows[row_index * valid_width : (row_index + 1) * valid_width]
            + bits_rows[row_index * bits_width : (row_index + 1) * bits_width]
        )
        digest.update(row.digest())
    return digest.hexdigest()


def _physical_chunk(
    *,
    path: Path,
    raw_chunk: Mapping[str, object],
    annual: PanelLogicalAnnualContent,
    schema_hash: str,
) -> PanelPhysicalChunk:
    metadata = pq.ParquetFile(path).metadata
    codecs = {
        str(metadata.row_group(group).column(column).compression)
        for group in range(metadata.num_row_groups)
        for column in range(metadata.row_group(group).num_columns)
    }
    return PanelPhysicalChunk(
        year=annual.year,
        logical_annual_hash=canonical_hash(annual.model_dump(mode="json")),
        legacy_chunk_hash=str(raw_chunk["chunk_hash"]),
        uri=str(raw_chunk["uri"]),
        raw_sha256=_file_sha256(path),
        byte_count=path.stat().st_size,
        row_count=annual.row_count,
        schema_hash=schema_hash,
        metadata_hash=str(raw_chunk["metadata_hash"]),
        compression_codecs=tuple(sorted(codecs)),
        parquet_created_by=metadata.created_by,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identified[Model: BaseModel](
    model: type[Model], values: Mapping[str, object], field: str
) -> Model:
    provisional = model.model_construct(**values, **{field: "0" * 64})
    payload = identity_payload(provisional, field)
    return cast(Model, model.model_validate({**values, field: canonical_hash(payload)}))


def _validate_hash(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("invalid Panel identity hash")


__all__ = [
    "PanelLogicalArtifactStore",
    "PanelLogicalIdentityPublisher",
    "logical_session_digest",
]
