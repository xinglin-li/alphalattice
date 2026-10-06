"""Marker-last current publication for Data-owned tradability authorities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor

from .contracts import (
    CurrentTradabilityDataMarker,
    CurrentTradabilityDataPointer,
    CurrentTradabilityDataProjection,
    CurrentTradabilityDataReceipt,
    HistoricalDecisionTradabilitySurface,
    HistoricalExecutionAvailabilitySurface,
    HistoricalTradabilityBundle,
    TradabilityArtifactReference,
    seal_tradability_contract,
)
from .surface import PublishedTradabilitySurfaces, TradabilityArtifactStore


class CurrentTradabilityBoundaryError(ValueError):
    """Stable fail-closed current Data publication error."""


@dataclass(frozen=True, slots=True)
class CurrentTradabilityPublication:
    """Return the action and complete current publication chain.

    Attributes:
        action: Publish, replace, exact reuse or current-read disposition.
        projection: Typed coverage and readiness projection.
        receipt: Ordered immutable artifact commitments.
        marker: Terminal lineage seal published before pointer replacement.
        pointer: Active marker/receipt/projection handles.
    """

    action: str
    projection: CurrentTradabilityDataProjection
    receipt: CurrentTradabilityDataReceipt
    marker: CurrentTradabilityDataMarker
    pointer: CurrentTradabilityDataPointer


@dataclass(frozen=True, slots=True)
class CurrentTradabilityLineage:
    """Return the owner-verified decision, execution and bundle authorities.

    Attributes:
        decision: Formation-time planned-order eligibility surface.
        execution: Post-session observed execution-availability surface.
        bundle: Typed commitment joining the two surfaces.
    """

    decision: HistoricalDecisionTradabilitySurface
    execution: HistoricalExecutionAvailabilitySurface
    bundle: HistoricalTradabilityBundle


def _reference(artifact_kind: str, descriptor: ArtifactDescriptor) -> TradabilityArtifactReference:
    return cast(
        TradabilityArtifactReference,
        TradabilityArtifactReference.model_validate(
            {
                "artifact_kind": artifact_kind,
                "content_hash": descriptor.content_hash,
                "metadata_hash": descriptor.metadata_hash,
                "uri": descriptor.uri,
            }
        ),
    )


class CurrentTradabilityDataService:
    """Publish and exactly replay one Data tradability pointer."""

    def __init__(self, artifact_root: Path) -> None:
        """Choose the tradability store and active-pointer path below a workspace root.

        Args:
            artifact_root: Workspace artifact directory used for immutable surfaces and current
                publication.
        """
        self.store = TradabilityArtifactStore(artifact_root)
        self.pointer_path = self.store.root / "current" / "active.json"

    def publish(
        self,
        *,
        surfaces: PublishedTradabilitySurfaces,
        published_at: datetime,
    ) -> CurrentTradabilityPublication:
        """Publish a verified surface chain before atomically replacing the active pointer.

        An identical existing projection and lineage return REUSED_EXACT. New projection,
        receipt and terminal marker artifacts are immutable; the active pointer is written
        last, then reopened and checked with the durable surface and chunk lineage.

        Args:
            surfaces: Published decision/execution surfaces, their descriptors and matching bundle.
            published_at: Timezone-aware publication instant.

        Returns:
            The published, replaced or exactly reused current publication chain.

        Raises:
            CurrentTradabilityBoundaryError: The clock, surface lineage or durable pointer readback
                is inconsistent.
        """
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise CurrentTradabilityBoundaryError("data_tradability.current_clock_invalid")
        decision = surfaces.decision
        execution = surfaces.execution
        bundle = surfaces.bundle
        if (
            decision.surface_hash != bundle.decision_surface_hash
            or execution.surface_hash != bundle.execution_surface_hash
            or decision.universe_epoch_hash != execution.universe_epoch_hash
            or decision.schedule_hash != execution.schedule_hash
            or decision.formation_sessions != execution.formation_sessions
            or decision.intended_execution_sessions != execution.intended_execution_sessions
        ):
            raise CurrentTradabilityBoundaryError("data_tradability.current_lineage_invalid")
        projection = seal_tradability_contract(
            CurrentTradabilityDataProjection,
            "projection_hash",
            status="DATA_TRADABILITY_READY",
            universe_epoch_label="CURRENT_UNIVERSE_RESEARCH_ONLY",
            coverage_start=bundle.coverage_start,
            coverage_end=bundle.coverage_end,
            formation_count=bundle.formation_count,
            asset_count=bundle.asset_count,
            latest_decision_formation=decision.formation_sessions[-1],
            latest_intended_execution=decision.intended_execution_sessions[-1],
            latest_observed_execution=execution.intended_execution_sessions[-1],
            eligible_decision_count=decision.eligible_row_count,
            limitations=decision.limitations,
        )
        existing = self.read_current()
        if existing is not None:
            lineage = self.read_lineage(existing)
            if (
                existing.projection == projection
                and lineage.decision.surface_hash == decision.surface_hash
                and lineage.execution.surface_hash == execution.surface_hash
                and lineage.bundle.bundle_hash == bundle.bundle_hash
            ):
                return CurrentTradabilityPublication(
                    action="REUSED_EXACT",
                    projection=existing.projection,
                    receipt=existing.receipt,
                    marker=existing.marker,
                    pointer=existing.pointer,
                )
        projection_artifact = self.store.publish_json(
            category="current/projections",
            payload=projection.model_dump(mode="json"),
            identity_field="projection_hash",
        )
        artifacts = (
            _reference("decision-surface", surfaces.decision_artifact),
            _reference("execution-surface", surfaces.execution_artifact),
            _reference("tradability-bundle", surfaces.bundle_artifact),
            _reference("current-projection", projection_artifact),
        )
        receipt = seal_tradability_contract(
            CurrentTradabilityDataReceipt,
            "receipt_hash",
            artifacts=artifacts,
            published_at=published_at,
        )
        receipt_artifact = self.store.publish_json(
            category="current/receipts",
            payload=receipt.model_dump(mode="json"),
            identity_field="receipt_hash",
        )
        marker = seal_tradability_contract(
            CurrentTradabilityDataMarker,
            "marker_hash",
            receipt_hash=receipt.receipt_hash,
            receipt_ref=receipt_artifact.uri,
            projection_hash=projection.projection_hash,
            projection_ref=projection_artifact.uri,
            bundle_hash=bundle.bundle_hash,
            bundle_ref=surfaces.bundle_artifact.uri,
            terminal_state=projection.status,
            completed_at=published_at,
        )
        marker_artifact = self.store.publish_json(
            category="current/markers",
            payload=marker.model_dump(mode="json"),
            identity_field="marker_hash",
        )
        pointer = seal_tradability_contract(
            CurrentTradabilityDataPointer,
            "pointer_hash",
            marker_hash=marker.marker_hash,
            marker_ref=marker_artifact.uri,
            receipt_hash=receipt.receipt_hash,
            receipt_ref=receipt_artifact.uri,
            projection_hash=projection.projection_hash,
            projection_ref=projection_artifact.uri,
        )
        action = "PUBLISHED" if existing is None else "REPLACED_CURRENT"
        self._write_pointer(pointer)
        durable = self.read_current()
        if durable is None or durable.pointer != pointer:
            raise CurrentTradabilityBoundaryError("data_tradability.current_readback_failed")
        self.read_lineage(durable)
        return CurrentTradabilityPublication(
            action=action,
            projection=projection,
            receipt=receipt,
            marker=marker,
            pointer=pointer,
        )

    def read_sealed_marker(self, marker_hash: str) -> CurrentTradabilityDataProjection:
        """Read one sealed marker by exact handle and verify its whole lineage.

        The current pointer is never opened.  A frozen consumer that recorded
        ``marker_hash`` keeps reading after Data publishes a newer current
        marker; being *current* is asked separately, only at admission.
        """
        marker, receipt, projection = self._read_marker_lineage(marker_hash)
        self._verify_lineage(receipt=receipt, marker=marker)
        if marker.terminal_state != projection.status:
            raise CurrentTradabilityBoundaryError("data_tradability.sealed_marker_tampered")
        return projection

    def _read_marker_lineage(
        self, marker_hash: str
    ) -> tuple[
        CurrentTradabilityDataMarker,
        CurrentTradabilityDataReceipt,
        CurrentTradabilityDataProjection,
    ]:
        """Load one marker and both of its content-addressed children."""
        try:
            marker = CurrentTradabilityDataMarker.model_validate(
                self.store.load_json(
                    category="current/markers",
                    uri=self.store.uri("current/markers", marker_hash),
                    identity_field="marker_hash",
                )
            )
            receipt = CurrentTradabilityDataReceipt.model_validate(
                self.store.load_json(
                    category="current/receipts",
                    uri=marker.receipt_ref,
                    identity_field="receipt_hash",
                )
            )
            projection = CurrentTradabilityDataProjection.model_validate(
                self.store.load_json(
                    category="current/projections",
                    uri=marker.projection_ref,
                    identity_field="projection_hash",
                )
            )
        except FileNotFoundError as error:
            raise FileNotFoundError("data_tradability.sealed_marker_missing") from error
        except Exception as error:
            raise CurrentTradabilityBoundaryError(
                "data_tradability.current_pointer_tampered"
            ) from error
        if (
            marker.marker_hash != marker_hash
            or receipt.receipt_hash != marker.receipt_hash
            or projection.projection_hash != marker.projection_hash
        ):
            raise CurrentTradabilityBoundaryError("data_tradability.current_lineage_invalid")
        return marker, receipt, projection

    def read_current(self) -> CurrentTradabilityPublication | None:
        """Read the active pointer and cross-check its marker, receipt and projection.

        Returns:
            The current typed publication, or None when no active pointer exists.

        Raises:
            CurrentTradabilityBoundaryError: Pointer bytes, linked identities or terminal projection
                state are inconsistent.
        """
        if not self.pointer_path.is_file():
            return None
        try:
            pointer = CurrentTradabilityDataPointer.model_validate_json(
                self.pointer_path.read_bytes()
            )
        except Exception as error:
            raise CurrentTradabilityBoundaryError(
                "data_tradability.current_pointer_tampered"
            ) from error
        marker, receipt, projection = self._read_marker_lineage(pointer.marker_hash)
        if (
            marker.receipt_hash != pointer.receipt_hash
            or marker.projection_hash != pointer.projection_hash
            or receipt.receipt_hash != pointer.receipt_hash
            or projection.projection_hash != pointer.projection_hash
            or marker.terminal_state != projection.status
        ):
            raise CurrentTradabilityBoundaryError("data_tradability.current_lineage_invalid")
        return CurrentTradabilityPublication(
            action="READ_CURRENT",
            projection=projection,
            receipt=receipt,
            marker=marker,
            pointer=pointer,
        )

    def read_lineage(self, publication: CurrentTradabilityPublication) -> CurrentTradabilityLineage:
        """Reopen the publication's surfaces and verify their bundle and immutable chunks.

        Args:
            publication: Typed publication whose receipt and terminal marker select the lineage.

        Returns:
            The verified decision, execution and bundle contracts.

        Raises:
            CurrentTradabilityBoundaryError: Referenced surfaces or bundle disagree with the
                marker/receipt.
        """
        return self._verify_lineage(receipt=publication.receipt, marker=publication.marker)

    def _verify_lineage(
        self,
        *,
        receipt: CurrentTradabilityDataReceipt,
        marker: CurrentTradabilityDataMarker,
    ) -> CurrentTradabilityLineage:
        """Read and cross-check every surface one receipt commits to."""
        references = {item.artifact_kind: item for item in receipt.artifacts}
        decision = HistoricalDecisionTradabilitySurface.model_validate(
            self.store.load_json(
                category="decision/manifests",
                uri=references["decision-surface"].uri,
                identity_field="surface_hash",
            )
        )
        execution = HistoricalExecutionAvailabilitySurface.model_validate(
            self.store.load_json(
                category="execution/manifests",
                uri=references["execution-surface"].uri,
                identity_field="surface_hash",
            )
        )
        bundle = HistoricalTradabilityBundle.model_validate(
            self.store.load_json(
                category="bundles",
                uri=references["tradability-bundle"].uri,
                identity_field="bundle_hash",
            )
        )
        if (
            decision.surface_hash != bundle.decision_surface_hash
            or execution.surface_hash != bundle.execution_surface_hash
            or bundle.bundle_hash != marker.bundle_hash
            or bundle.bundle_hash != references["tradability-bundle"].content_hash
        ):
            raise CurrentTradabilityBoundaryError("data_tradability.current_lineage_invalid")
        for chunk in (*decision.chunks, *execution.chunks):
            self.store.resolve_chunk(chunk)
        return CurrentTradabilityLineage(
            decision=decision,
            execution=execution,
            bundle=bundle,
        )

    def _write_pointer(self, pointer: CurrentTradabilityDataPointer) -> None:
        content = json.dumps(
            pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.pointer_path.with_name(f".{self.pointer_path.name}.{os.getpid()}.tmp")
        staged.write_bytes(content)
        os.replace(staged, self.pointer_path)
        staged.unlink(missing_ok=True)


__all__ = [
    "CurrentTradabilityBoundaryError",
    "CurrentTradabilityDataService",
    "CurrentTradabilityLineage",
    "CurrentTradabilityPublication",
]
