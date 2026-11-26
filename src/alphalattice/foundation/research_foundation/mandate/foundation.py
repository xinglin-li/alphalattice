"""Deterministic assembly of the frozen Research Desk foundation binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
)
from alphalattice.foundation.factor_research.inputs.research_input import FactorHorizonResearchInput
from alphalattice.foundation.factor_research.programs.program import FactorResearchProgramSpec
from alphalattice.foundation.factor_research.publication.artifacts import (
    FactorResearchArtifactStore,
)
from alphalattice.foundation.research_foundation.contracts import (
    PreResearchDeskSafeProjection,
    ResearchDeskExecutionOutcomeRef,
    ResearchFoundationBinding,
    ResearchFoundationMarker,
)
from alphalattice.kernel.shared_kernel.domain.enums import DataValidityClass
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    sector_treatment,
)
from alphalattice.kernel.validation.enums import RebalanceFrequency

_CURRENT_FACTOR_LIMITATIONS = (
    "CURRENT_ACTIVE_SURVIVORS",
    SECTOR_HISTORY_BACKFILLED,
    "CURRENT_UNIVERSE_RESEARCH_ONLY",
    "NON_POINT_IN_TIME_RESEARCH",
    "SEALED_HOLDOUT_LOCKED",
)


def build_current_research_foundation_binding(
    *,
    panel_manifest: Mapping[str, object],
    factor_program: FactorResearchProgramSpec,
    factor_input: FactorHorizonResearchInput,
    evidence_report_hash: str,
    redundancy_structure_hash: str,
    execution_outcome: CausalExecutionOutcomeManifest,
    execution_marker_hash: str,
    logical_panel_hash: str,
    logical_semantic_index_hash: str,
) -> ResearchFoundationBinding:
    """Bind the verified one-session Factor publication without rerunning research.

    Args:
        panel_manifest: Admitted Feature Panel manifest and qualified lineage.
        factor_program: Validated one-session program naming those Panel and outcome inputs.
        factor_input: Host-curated nonempty core/conditional slate with complete evidence
            references.
        evidence_report_hash: Verified complete-report identity named by the Factor input.
        redundancy_structure_hash: Verified cluster identity named by the Factor input.
        execution_outcome: Admitted execution-outcome manifest with development and reserved
            commitments.
        execution_marker_hash: Publication marker for those outcomes.
        logical_panel_hash: Native logical Panel identity to retain with the Foundation.
        logical_semantic_index_hash: Paired semantic-index identity for that logical Panel.

    Returns:
        Validated Foundation binding preserving source lineage and current-universe
        limitations without rerunning Factor statistics or reading reserved values.

    Raises:
        ValueError: Input identities differ, Panel lineage is absent, logical/sector
            references are incomplete, or the selected Factor slate is empty.
    """
    program = FactorResearchProgramSpec.model_validate(factor_program)
    factor_input = FactorHorizonResearchInput.model_validate(factor_input)
    if program.feature_panel_snapshot_hash != str(panel_manifest.get("snapshot_hash")):
        raise ValueError("current Factor foundation panel identity differs")
    if program.causal_outcome_snapshot_hash != execution_outcome.snapshot_hash:
        raise ValueError("current Factor foundation outcome identity differs")
    if factor_input.evidence_report_hash != evidence_report_hash:
        raise ValueError("current Factor foundation evidence identity differs")
    if factor_input.redundancy_structure_hash != redundancy_structure_hash:
        raise ValueError("current Factor foundation redundancy identity differs")
    safe = panel_manifest.get("safe_summary")
    lineage = safe.get("lineage") if isinstance(safe, Mapping) else None
    if not isinstance(lineage, Mapping):
        raise ValueError("current Factor foundation panel lineage is unavailable")
    sector_revision = lineage.get("sector_revision")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in (logical_panel_hash, logical_semantic_index_hash, sector_revision)
    ):
        raise ValueError("current Factor foundation logical lineage is incomplete")
    selected_factor_ids = (
        *factor_input.core_factor_ids,
        *factor_input.conditional_factor_ids,
    )
    if not selected_factor_ids:
        raise ValueError("current Factor foundation has no selected factors")
    execution_ref = ResearchDeskExecutionOutcomeRef(
        research_cadence=RebalanceFrequency.DAILY,
        snapshot_hash=execution_outcome.snapshot_hash,
        schedule_hash=execution_outcome.schedule_hash,
        development_content_hash=canonical_hash(
            tuple(value.content_hash for value in execution_outcome.development_chunks)
        ),
        sealed_holdout_content_hash=canonical_hash(
            tuple(value.content_hash for value in execution_outcome.sealed_holdout_chunks)
        ),
        marker_hash=execution_marker_hash,
        market_as_of=execution_outcome.market_as_of.isoformat(),
        data_validity_class=DataValidityClass.CURRENT_UNIVERSE_RESEARCH_ONLY,
    )
    values = {
        "research_cadence": RebalanceFrequency.DAILY,
        "feature_panel_snapshot_hash": program.feature_panel_snapshot_hash,
        "logical_panel_hash": logical_panel_hash,
        "logical_semantic_index_hash": logical_semantic_index_hash,
        "factor_training_outcome_snapshot_hash": program.causal_outcome_snapshot_hash,
        "factor_screening_result_hash": evidence_report_hash,
        "factor_candidate_slate_hash": redundancy_structure_hash,
        "research_desk_factor_input_hash": factor_input.input_hash,
        "factor_research_program_hash": program.program_hash,
        "factor_research_evidence_hash": evidence_report_hash,
        "factor_research_redundancy_hash": redundancy_structure_hash,
        "prediction_horizon_sessions": 1,
        "execution_timing": "NEXT_COMMON_SESSION_OPEN",
        "target_semantics": "LOG_EXECUTION_RETURN",
        "sector_revision": sector_revision,
        "sector_point_in_time_qualified": False,
        # The Sector treatment the Panel's sessions read.
        "limitations": tuple(
            sector_treatment(reclassified=bool(lineage.get("sector_reclassifications")))
            if value == SECTOR_HISTORY_BACKFILLED
            else value
            for value in _CURRENT_FACTOR_LIMITATIONS
        ),
        "execution_outcome": execution_ref,
        "ordered_factor_ids": selected_factor_ids,
        "downstream_factor_research_forbidden": True,
        "secondary_feature_preprocessing_forbidden": True,
    }
    identity = ResearchFoundationBinding.model_construct(**values, foundation_hash="").model_dump(
        mode="json", exclude={"foundation_hash"}, exclude_none=True
    )
    return ResearchFoundationBinding(**values, foundation_hash=canonical_hash(identity))


@dataclass(frozen=True)
class PublishedResearchFoundation:
    """Carry the published Foundation binding and marker with their artifact descriptors.

    Attributes:
        binding: Qualified research lineage admitted for downstream use.
        binding_artifact: Descriptor of its durable content-addressed publication.
        marker: Marker sealed after the binding was published.
        marker_artifact: Descriptor of the durable authoritative marker.
    """

    binding: ResearchFoundationBinding
    binding_artifact: ArtifactDescriptor
    marker: ResearchFoundationMarker
    marker_artifact: ArtifactDescriptor


class ResearchFoundationService:
    """Publish the verified pre-Desk boundary marker-last and replay it exactly."""

    def __init__(
        self,
        *,
        artifact_store: FactorResearchArtifactStore,
        mutation_gate: WorkspaceMutationGate,
    ) -> None:
        """Bind Foundation artifact publication and the workspace mutation gate.

        Args:
            artifact_store: Owner of durable Foundation records and safe projections.
            mutation_gate: Gate covering marker publication and authoritative readback.
        """
        self.artifact_store = artifact_store
        self.mutation_gate = mutation_gate

    def publish(
        self,
        *,
        binding: ResearchFoundationBinding,
        execution_outcome_marker_hash: str,
        updated_at: datetime,
        publish_projection: bool = True,
    ) -> PublishedResearchFoundation:
        """Publish a Foundation binding, then seal and verify its authoritative marker.

        Args:
            binding: Validated research lineage to publish.
            execution_outcome_marker_hash: Execution marker to bind into this publication.
            updated_at: Timezone-aware publication/projection clock.
            publish_projection: Publish the safe pre-Desk projection after authoritative readback.

        Returns:
            Durable binding and marker descriptors. An existing projection for the same
            Foundation/marker pair is reused; disabling the projection stops after marker readback.

        Raises:
            ValueError: The clock is naive, a publication contract is invalid, or binding,
                marker, or safe-projection readback differs from the published contents.
        """
        if updated_at.tzinfo is None or updated_at.utcoffset() is None:
            raise ValueError("Research Foundation publication clock must be timezone-aware")
        binding_payload = binding.model_dump(mode="json", exclude_none=True)
        if (
            canonical_hash(
                {key: value for key, value in binding_payload.items() if key != "foundation_hash"}
            )
            != binding.foundation_hash
        ):
            binding_payload = binding.model_dump(mode="json")
        binding_artifact = self.artifact_store.publish_research_foundation(
            payload=binding_payload,
            foundation_hash=binding.foundation_hash,
        )
        marker_values = {
            "kind": "ResearchFoundationMarker",
            "foundation_hash": binding.foundation_hash,
            "foundation_ref": binding_artifact.uri,
            "research_cadence": binding.research_cadence,
            "factor_screening_result_hash": binding.factor_screening_result_hash,
            "factor_research_program_hash": binding.factor_research_program_hash,
            "factor_research_input_hash": (
                binding.research_desk_factor_input_hash
                if binding.factor_research_program_hash is not None
                else None
            ),
            "execution_outcome_marker_hash": execution_outcome_marker_hash,
        }
        marker_identity = {key: value for key, value in marker_values.items() if value is not None}
        marker = ResearchFoundationMarker(
            **marker_values, marker_hash=canonical_hash(marker_identity)
        )
        with self.mutation_gate.try_hold(timeout_seconds=30.0):
            marker_artifact = self.artifact_store.publish_research_foundation_marker(
                payload=marker.model_dump(mode="json", exclude_none=True),
                marker_hash=marker.marker_hash,
            )
            durable_marker = ResearchFoundationMarker.model_validate(
                self.artifact_store.load_research_foundation_marker(marker_artifact.uri)
            )
            durable_binding = ResearchFoundationBinding.model_validate(
                self.artifact_store.load_research_foundation(durable_marker.foundation_ref)
            )
            if durable_marker != marker or durable_binding != binding:
                raise ValueError("Research Foundation authoritative readback differs")
            if not publish_projection:
                return PublishedResearchFoundation(
                    binding=binding,
                    binding_artifact=binding_artifact,
                    marker=marker,
                    marker_artifact=marker_artifact,
                )
            try:
                existing_projection = PreResearchDeskSafeProjection.model_validate(
                    self.artifact_store.pre_research_desk_projection()
                )
            except (FileNotFoundError, ValueError):
                existing_projection = None
            if existing_projection is not None and (
                existing_projection.foundation_hash == binding.foundation_hash
                and existing_projection.foundation_marker_hash == marker.marker_hash
            ):
                return PublishedResearchFoundation(
                    binding=binding,
                    binding_artifact=binding_artifact,
                    marker=marker,
                    marker_artifact=marker_artifact,
                )
            projection_values = {
                "kind": "PreResearchDeskSafeProjection",
                "status": "PRE_RESEARCH_DESK_CLOSED",
                "next_action": "READY_TO_START_FIRST_HIGH_RESEARCH_DESK_SLICE",
                "research_cadence": binding.research_cadence,
                "foundation_hash": binding.foundation_hash,
                "foundation_marker_hash": marker.marker_hash,
                "logical_panel_hash": binding.logical_panel_hash,
                "logical_semantic_index_hash": binding.logical_semantic_index_hash,
                "factor_screening_result_hash": binding.factor_screening_result_hash,
                "factor_candidate_slate_hash": binding.factor_candidate_slate_hash,
                "research_desk_factor_input_hash": binding.research_desk_factor_input_hash,
                "factor_research_program_hash": binding.factor_research_program_hash,
                "prediction_horizon_sessions": binding.prediction_horizon_sessions,
                "execution_timing": binding.execution_timing,
                "target_semantics": binding.target_semantics,
                "sector_point_in_time_qualified": binding.sector_point_in_time_qualified,
                "execution_outcome_snapshot_hash": binding.execution_outcome.snapshot_hash,
                "candidate_count": len(binding.ordered_factor_ids),
                "market_as_of": binding.execution_outcome.market_as_of,
                "updated_at": updated_at.astimezone(UTC),
            }
            projection_identity = PreResearchDeskSafeProjection.model_construct(
                **projection_values, projection_hash=""
            ).model_dump(mode="json", exclude={"projection_hash"}, exclude_none=True)
            projection = PreResearchDeskSafeProjection(
                **projection_values, projection_hash=canonical_hash(projection_identity)
            )
            self.artifact_store.publish_pre_research_desk_projection(
                projection.model_dump(mode="json", exclude_none=True)
            )
            durable_projection = PreResearchDeskSafeProjection.model_validate(
                self.artifact_store.pre_research_desk_projection()
            )
            if durable_projection != projection:
                raise ValueError("pre-Research Desk safe projection readback differs")
        return PublishedResearchFoundation(
            binding=binding,
            binding_artifact=binding_artifact,
            marker=marker,
            marker_artifact=marker_artifact,
        )


__all__ = [
    "PublishedResearchFoundation",
    "ResearchFoundationService",
    "build_current_research_foundation_binding",
]
