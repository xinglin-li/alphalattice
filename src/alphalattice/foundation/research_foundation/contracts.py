"""Code-owned constitution for the post-Factor Research desktop boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.domain.enums import DataValidityClass
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SECTOR_HISTORY_FORWARD,
)
from alphalattice.kernel.validation.enums import RebalanceFrequency


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchDeskExecutionOutcomeRef(_ContractModel):
    """Reference the execution outcomes admitted by a research cadence.

    The snapshot, schedule, development and reserved-content commitments are retained
    with the publication marker and market clock. They describe current-universe
    research lineage without granting point-in-time historical validity.
    """

    research_cadence: RebalanceFrequency
    snapshot_hash: str
    schedule_hash: str
    development_content_hash: str
    sealed_holdout_content_hash: str
    marker_hash: str
    market_as_of: str
    data_validity_class: Literal[DataValidityClass.CURRENT_UNIVERSE_RESEARCH_ONLY]


class ResearchFoundationBinding(_ContractModel):
    """Bind admitted Feature, Factor and execution evidence for downstream research.

    The ordered factor axis and execution cadence are explicit. Current Factor
    lineage is complete or absent as a unit and retains its research limitations.
    Downstream Factor research and secondary Feature preprocessing are forbidden;
    optional logical Panel identities must occur together.
    """

    research_cadence: RebalanceFrequency
    feature_panel_snapshot_hash: str
    logical_panel_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    logical_semantic_index_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    factor_training_outcome_snapshot_hash: str
    factor_screening_result_hash: str
    factor_candidate_slate_hash: str
    research_desk_factor_input_hash: str
    factor_research_program_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    factor_research_evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    factor_research_redundancy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_horizon_sessions: Literal[1] | None = None
    execution_timing: Literal["NEXT_COMMON_SESSION_OPEN"] | None = None
    target_semantics: Literal["LOG_EXECUTION_RETURN"] | None = None
    sector_revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sector_point_in_time_qualified: Literal[False] | None = None
    limitations: tuple[str, ...] | None = None
    execution_outcome: ResearchDeskExecutionOutcomeRef
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1, max_length=55)
    downstream_factor_research_forbidden: Literal[True] = True
    secondary_feature_preprocessing_forbidden: Literal[True] = True
    foundation_hash: str

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchFoundationBinding:
        """Verify cadence, unique factors, complete lineage and the Foundation identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Logical identities are incomplete, cadences differ, factors repeat, current
                lineage/limitations are incomplete, or neither supported canonical hash matches.
        """
        if (self.logical_panel_hash is None) != (self.logical_semantic_index_hash is None):
            raise ValueError("research foundation native logical identity is incomplete")
        if self.research_cadence is not self.execution_outcome.research_cadence:
            raise ValueError("research foundation cadence differs from execution outcome")
        if len(self.ordered_factor_ids) != len(set(self.ordered_factor_ids)):
            raise ValueError("research foundation factor IDs must be unique")
        current_fields = (
            self.factor_research_program_hash,
            self.factor_research_evidence_hash,
            self.factor_research_redundancy_hash,
            self.prediction_horizon_sessions,
            self.execution_timing,
            self.target_semantics,
            self.sector_revision,
            self.sector_point_in_time_qualified,
            self.limitations,
        )
        if any(value is not None for value in current_fields):
            if any(value is None for value in current_fields):
                raise ValueError("current Factor Research foundation lineage is incomplete")
            required = {
                "CURRENT_ACTIVE_SURVIVORS",
                "CURRENT_UNIVERSE_RESEARCH_ONLY",
                "NON_POINT_IN_TIME_RESEARCH",
                "SEALED_HOLDOUT_LOCKED",
            }
            # One Sector treatment, the one its Panel's sessions read.
            if (
                self.limitations is None
                or not required.issubset(self.limitations)
                or len({SECTOR_HISTORY_BACKFILLED, SECTOR_HISTORY_FORWARD} & set(self.limitations))
                != 1
            ):
                raise ValueError("current Factor Research foundation omits limitations")
        identities = (
            self.model_dump(mode="json", exclude={"foundation_hash"}, exclude_none=True),
            self.model_dump(mode="json", exclude={"foundation_hash"}),
        )
        if self.foundation_hash not in {canonical_hash(identity) for identity in identities}:
            raise ValueError("research foundation hash is invalid")
        return self


class ResearchFoundationMarker(_ContractModel):
    """Seal the published Foundation reference and its execution/Factor lineage.

    The marker binds the Foundation identity, publication reference, research cadence
    and execution-outcome marker. Optional current Factor program/input references
    are retained under the supported canonical identity forms.
    """

    kind: Literal["ResearchFoundationMarker"] = "ResearchFoundationMarker"
    foundation_hash: str
    foundation_ref: str
    research_cadence: RebalanceFrequency
    factor_screening_result_hash: str
    factor_research_program_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    factor_research_input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_outcome_marker_hash: str
    marker_hash: str

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchFoundationMarker:
        """Verify the publication marker using either supported canonical null representation.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The recorded marker hash matches neither supported canonical representation.
        """
        identities = (
            self.model_dump(mode="json", exclude={"marker_hash"}, exclude_none=True),
            self.model_dump(mode="json", exclude={"marker_hash"}),
        )
        if self.marker_hash not in {canonical_hash(identity) for identity in identities}:
            raise ValueError("Research Foundation marker hash is invalid")
        return self


class ResearchFoundationAdmission(_ContractModel):
    """An explicitly selected research snapshot, never a current pointer."""

    kind: Literal["ResearchFoundationAdmission"] = "ResearchFoundationAdmission"
    foundation: ResearchFoundationBinding
    factor_task_id: str = Field(pattern=r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")
    factor_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    curation_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_id: str = Field(min_length=1)
    input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_input_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    admission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchFoundationAdmission:
        """Verify the explicit snapshot admission and its receipt/input provenance.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The admission hash differs from the canonical recorded contents.
        """
        if self.admission_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"admission_hash"})
        ):
            raise ValueError("research_foundation.admission_identity_invalid")
        return self


class PreResearchDeskSafeProjection(_ContractModel):
    """Expose the verified pre-Desk boundary as a sealed safe status projection.

    The projection retains Foundation, Factor, execution and optional logical Panel
    lineage, candidate count, market clock and the next admitted action. Its update
    clock is timezone-aware; it carries no authority to recompute the evidence.
    """

    kind: Literal["PreResearchDeskSafeProjection"] = "PreResearchDeskSafeProjection"
    status: Literal["PRE_RESEARCH_DESK_CLOSED"] = "PRE_RESEARCH_DESK_CLOSED"
    next_action: Literal["READY_TO_START_FIRST_HIGH_RESEARCH_DESK_SLICE"] = (
        "READY_TO_START_FIRST_HIGH_RESEARCH_DESK_SLICE"
    )
    research_cadence: Literal[RebalanceFrequency.DAILY]
    foundation_hash: str
    foundation_marker_hash: str
    logical_panel_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    logical_semantic_index_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    factor_screening_result_hash: str
    factor_candidate_slate_hash: str
    research_desk_factor_input_hash: str
    factor_research_program_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prediction_horizon_sessions: Literal[1] | None = None
    execution_timing: Literal["NEXT_COMMON_SESSION_OPEN"] | None = None
    target_semantics: Literal["LOG_EXECUTION_RETURN"] | None = None
    sector_point_in_time_qualified: Literal[False] | None = None
    execution_outcome_snapshot_hash: str
    candidate_count: int = Field(ge=1, le=55)
    market_as_of: str
    updated_at: datetime
    projection_hash: str

    @model_validator(mode="after")
    def validate_identity(self) -> PreResearchDeskSafeProjection:
        """Verify the projection clock, paired logical identities and canonical content identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The clock is naive, logical lineage is incomplete, or neither supported hash
                matches.
        """
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("pre-Research Desk projection clock is invalid")
        if (self.logical_panel_hash is None) != (self.logical_semantic_index_hash is None):
            raise ValueError("pre-Research Desk projection logical identity is incomplete")
        identities = (
            self.model_dump(mode="json", exclude={"projection_hash"}, exclude_none=True),
            self.model_dump(mode="json", exclude={"projection_hash"}),
        )
        if self.projection_hash not in {canonical_hash(identity) for identity in identities}:
            raise ValueError("pre-Research Desk projection hash is invalid")
        return self


__all__ = [
    "PreResearchDeskSafeProjection",
    "ResearchDeskExecutionOutcomeRef",
    "ResearchFoundationAdmission",
    "ResearchFoundationBinding",
    "ResearchFoundationMarker",
]
