"""Current Alpha publication contracts and authoritative readback identities."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.sector_history import SectorHistory, SectorReclassification
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..experiments.development_contracts import seal_current_contract


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class WorkspaceObservationHistoryPart(_Contract):
    """One complete session segment of exact float64 observation columns."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["WorkspaceObservationHistoryPart"] = "WorkspaceObservationHistoryPart"
    start: int = Field(ge=0)
    stop: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    part_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> WorkspaceObservationHistoryPart:
        """Verify the nonempty session range and canonical part commitment."""
        if self.stop <= self.start or self.part_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"part_hash"})
        ):
            raise ValueError("alpha_research.workspace_observation_history_invalid")
        return self


class WorkspaceObservationHistoryHead(_Contract):
    """Select verified immutable prefix segments and one replaceable unmatured tail.

    The dependency proof belongs to the caller's actual source-prefix owner. The head binds
    exact ordered axes, column names and float64 parts without changing scientific snapshots.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["WorkspaceObservationHistoryHead"] = "WorkspaceObservationHistoryHead"
    scope_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_prefix_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    column_names: tuple[str, ...] = Field(min_length=1)
    dtype: Literal["float64"] = "float64"
    stable_session_count: int = Field(ge=0)
    predecessor_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parts: tuple[WorkspaceObservationHistoryPart, ...] = Field(min_length=1)
    head_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> WorkspaceObservationHistoryHead:
        """Verify contiguous immutable segments, ordered axes and the head seal."""
        stop = 0
        for part in self.parts:
            if part.start != stop or part.start < self.stable_session_count < part.stop:
                raise ValueError("alpha_research.workspace_observation_history_invalid")
            stop = part.stop
        if (
            self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(set(self.ordered_listing_ids)) != len(self.ordered_listing_ids)
            or any(not name for name in self.ordered_listing_ids)
            or self.column_names != tuple(sorted(set(self.column_names)))
            or any(not name for name in self.column_names)
            or self.stable_session_count > len(self.formation_sessions)
            or stop != len(self.formation_sessions)
            or sum(part.start >= self.stable_session_count for part in self.parts) > 1
            or self.predecessor_head_hash == self.head_hash
            or self.head_hash != canonical_hash(self.model_dump(mode="json", exclude={"head_hash"}))
        ):
            raise ValueError("alpha_research.workspace_observation_history_invalid")
        return self


class WorkspaceObservationHistoryMarker(_Contract):
    """Marker-last selection of the current head and its independently retained predecessor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    kind: Literal["WorkspaceObservationHistoryMarker"] = "WorkspaceObservationHistoryMarker"
    scope_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_head_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> WorkspaceObservationHistoryMarker:
        """Verify distinct current and previous heads and the marker seal."""
        if self.current_head_hash == self.previous_head_hash or self.marker_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"marker_hash"})
        ):
            raise ValueError("alpha_research.workspace_observation_history_invalid")
        return self


class SectorReclassificationRecord(_Contract):
    """One listing's move to another Sector, read from its effective session on (V346)."""

    listing_id: str = Field(min_length=1)
    effective_session: date
    prior_sector: str = Field(min_length=1)
    sector: str = Field(min_length=1)


class SectorHistoryRecord(_Contract):
    """A Sector history's revision and reclassifications; its current map is recorded beside."""

    current_revision: str = Field(min_length=1)
    reclassifications: tuple[SectorReclassificationRecord, ...] = Field(min_length=1)

    @classmethod
    def of(cls, sectors: Mapping[str, str]) -> SectorHistoryRecord | None:
        """The record of a history holding reclassifications; None for any other map."""
        if not isinstance(sectors, SectorHistory) or not sectors.reclassifications:
            return None
        return cls(
            current_revision=sectors.current_revision,
            reclassifications=tuple(
                SectorReclassificationRecord(
                    listing_id=item.listing_id,
                    effective_session=item.effective_session,
                    prior_sector=item.prior_sector,
                    sector=item.sector,
                )
                for item in sectors.reclassifications
            ),
        )

    def history(self, current: Mapping[str, str]) -> SectorHistory:
        """The history this record and a current map make."""
        return SectorHistory(
            current_revision=self.current_revision,
            current=dict(current),
            reclassifications=tuple(
                SectorReclassification(
                    listing_id=item.listing_id,
                    effective_session=item.effective_session,
                    prior_sector=item.prior_sector,
                    sector=item.sector,
                )
                for item in self.reclassifications
            ),
        )


class FrozenScoreObservationSnapshot(_Contract):
    """Input-only snapshot: no training labels, selected model or research marker."""

    kind: Literal["FrozenScoreObservationSnapshot"] = "FrozenScoreObservationSnapshot"
    disposition: Literal["RECORDED_INPUT_QA", "SYNTHETIC_INPUT_QA"]
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    sector_by_listing_id: dict[str, str]
    sector_history: SectorHistoryRecord | None = Field(default=None, exclude_if=lambda v: v is None)
    """The reclassifications its sessions read beside the current map (V346); absent while
    none is in force, so such a snapshot is what it was."""
    array_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_formula_ids: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    reference_eligibility_recorded: bool = Field(default=False, exclude_if=lambda v: not v)
    nominal_population_recorded: bool = Field(default=False, exclude_if=lambda v: not v)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> FrozenScoreObservationSnapshot:
        if (
            not self.formation_sessions
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(self.ordered_listing_ids) != len(set(self.ordered_listing_ids))
            or set(self.sector_by_listing_id) != set(self.ordered_listing_ids)
            or self.ordered_formula_ids != tuple(sorted(set(self.ordered_formula_ids)))
            or self.snapshot_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"snapshot_hash"}))
        ):
            raise ValueError("alpha_research.frozen_observation_identity_invalid")
        return self


class FrozenFeaturePreparation(_Contract):
    kind: Literal["FrozenFeaturePreparation"] = "FrozenFeaturePreparation"
    observation_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    ordered_listing_ids: tuple[str, ...]
    ordered_feature_ids: tuple[str, ...]
    vintages: tuple[str, ...]
    source_binding_hashes: tuple[str, ...]
    feature_values_hashes: tuple[str, ...]
    array_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> FrozenFeaturePreparation:
        if (
            not self.vintages
            or len(set(self.vintages)) != len(self.vintages)
            or len(self.source_binding_hashes) != len(self.vintages)
            or len(self.feature_values_hashes) != len(self.vintages)
            or self.preparation_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"preparation_hash"}))
        ):
            raise ValueError("alpha_research.frozen_feature_preparation_invalid")
        return self


class FrozenComponentScoreReadout(_Contract):
    """Display facts derived by Alpha; browsers and Agents only render them."""

    eligible_listing_count: int = Field(ge=0)
    score_minimum: float | None
    score_maximum: float | None
    score_range_display: str


class FrozenComponentScoreSnapshot(_Contract):
    """A computed component score, not a Foundation or Portfolio recommendation."""

    kind: Literal["FrozenComponentScoreSnapshot"] = "FrozenComponentScoreSnapshot"
    disposition: Literal["INPUT_TO_SCORE_QA_NOT_PORTFOLIO_RECOMMENDATION"] = (
        "INPUT_TO_SCORE_QA_NOT_PORTFOLIO_RECOMMENDATION"
    )
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    ordered_listing_ids: tuple[str, ...]
    feature_values_hashes: tuple[str, ...]
    model_identity_hashes: tuple[str, ...]
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    scores: tuple[float | None, ...]
    live: tuple[bool, ...]
    prediction_calls: int = Field(ge=0)
    fit_calls: int = Field(default=0, ge=0)
    model_set_publication_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> FrozenComponentScoreSnapshot:
        if (
            not self.ordered_listing_ids
            or len(self.ordered_listing_ids) != len(set(self.ordered_listing_ids))
            or len(self.scores) != len(self.ordered_listing_ids)
            or len(self.live) != len(self.scores)
            or any(value is not None and not math.isfinite(value) for value in self.scores)
            or any(
                active != (value is not None)
                for active, value in zip(self.live, self.scores, strict=True)
            )
            or self.prediction_calls != len(self.model_identity_hashes)
            or (self.fit_calls != 0 and self.model_set_publication_hash is None)
            or self.snapshot_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"snapshot_hash"}))
        ):
            raise ValueError("alpha_research.frozen_score_identity_invalid")
        return self

    def readout(self) -> FrozenComponentScoreReadout:
        values = tuple(value for value in self.scores if value is not None)
        low, high = (min(values), max(values)) if values else (None, None)
        return FrozenComponentScoreReadout(
            eligible_listing_count=sum(self.live),
            score_minimum=low,
            score_maximum=high,
            score_range_display="Unavailable"
            if low is None or high is None
            else f"{low:.6f} to {high:.6f}",
        )


class AlphaStabilityCheck(_Contract):
    """Record one refit stability check, observed value and optional admitted bounds."""

    check_id: str
    passed: bool
    observed: float | str
    lower: float | None = Field(default=None, allow_inf_nan=False)
    upper: float | None = Field(default=None, allow_inf_nan=False)


class CurrentRefitStabilityAssessment(_Contract):
    """Seal current versus development refit checks and their failure disposition.

    Aggregate passage equals every retained check. A failed assessment carries the registered
    insufficiency code; a passed assessment carries no failure code.
    """

    kind: Literal["CurrentRefitStabilityAssessment"] = "CurrentRefitStabilityAssessment"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str
    development_state_hashes: tuple[str, ...] = Field(min_length=1)
    current_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: tuple[AlphaStabilityCheck, ...] = Field(min_length=1)
    passed: bool
    failure_code: Literal["alpha_research.current_refit_insufficient"] | None = None
    assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> CurrentRefitStabilityAssessment:
        """Require check-derived passage, matching failure presence and exact assessment identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Aggregate passage differs from checks, failure presence disagrees or
                assessment_hash is inconsistent.
        """
        if self.passed != all(value.passed for value in self.checks):
            raise ValueError("Alpha refit stability result does not match its checks")
        if self.passed == (self.failure_code is not None):
            raise ValueError("Alpha refit stability failure code is inconsistent")
        if self.assessment_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"assessment_hash"})
        ):
            raise ValueError("Alpha refit stability assessment hash is invalid")
        return self


class AlphaCurrentCandidateScoreChunkRef(_Contract):
    """Bind one candidate formation-score chunk to exact model and stability lineage.

    The reference retains source/request identities, ordered listing-axis hash, row count,
    content/metadata commitments and URI.
    """

    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    estimator_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str
    row_count: int = Field(ge=1)


class AlphaCurrentCandidateScoreChild(_Contract):
    """Seal one candidate current-score child against its complete chunk lineage.

    The child retains the exact listing axis and a training cutoff no later than formation.
    """

    kind: Literal["AlphaCurrentCandidateScoreChild"] = "AlphaCurrentCandidateScoreChild"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str
    estimator_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    training_cutoff: date
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    chunk: AlphaCurrentCandidateScoreChunkRef
    child_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCurrentCandidateScoreChild:
        """Require exact chunk lineage, listing coverage and causal current training cutoff.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Chunk request/source/model/stability/formation/listing lineage, row count,
                training clock or child_hash differs.
        """
        if (
            self.chunk.request_hash != self.request_hash
            or self.chunk.foundation_hash != self.foundation_hash
            or self.chunk.logical_panel_hash != self.logical_panel_hash
            or self.chunk.candidate_id != self.candidate_id
            or self.chunk.estimator_state_hash != self.estimator_state_hash
            or self.chunk.stability_assessment_hash != self.stability_assessment_hash
            or self.chunk.formation_session != self.formation_session
            or self.chunk.ordered_listing_ids_hash != canonical_hash(self.ordered_listing_ids)
            or self.chunk.row_count != len(self.ordered_listing_ids)
            or self.training_cutoff > self.formation_session
        ):
            raise ValueError("Alpha current candidate score lineage differs")
        if self.child_hash != canonical_hash(self.model_dump(mode="json", exclude={"child_hash"})):
            raise ValueError("Alpha current candidate score hash is invalid")
        return self


class AlphaCurrentCandidateQualification(_Contract):
    """Seal current candidate qualification and its stability/score evidence.

    Stable means a current-score child is present and failure codes are empty.
    """

    candidate_id: str
    development_candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_score_child_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stable: bool
    failure_codes: tuple[str, ...]
    qualification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCurrentCandidateQualification:
        """Require score/failure-derived stability and exact qualification identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Stability disagrees with score-child/failure presence or qualification_hash
                is inconsistent.
        """
        if self.stable != (self.current_score_child_hash is not None and not self.failure_codes):
            raise ValueError("Alpha current candidate qualification is inconsistent")
        if self.qualification_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"qualification_hash"})
        ):
            raise ValueError("Alpha current candidate qualification hash is invalid")
        return self


class AlphaCurrentStabilitySurface(_Contract):
    """Seal ordered current qualification and the reconciled dual-qualified peer axis."""

    kind: Literal["AlphaCurrentStabilitySurface"] = "AlphaCurrentStabilitySurface"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    viability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    training_cutoff: date
    candidates: tuple[AlphaCurrentCandidateQualification, ...] = Field(min_length=1)
    dual_qualified_candidate_ids: tuple[str, ...]
    disposition: Literal[
        "DUAL_QUALIFIED_MODELS_AVAILABLE",
        "NO_STABLE_CURRENT_ALPHA_MODEL",
    ]
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def assessment_hash(self) -> str:
        """Compatibility-shaped access for bounded selector factories."""
        return self.viability_assessment_hash

    @property
    def admissible_candidate_ids(self) -> tuple[str, ...]:
        """Read the reconciled dual-qualified candidate axis for downstream admission.

        Returns:
            The declared dual_qualified_candidate_ids tuple in retained candidate order.
        """
        return self.dual_qualified_candidate_ids

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCurrentStabilitySurface:
        """Require unique current peers and reconciled dual qualification/disposition.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Candidate identifiers repeat, stable peers differ from the dual-qualified
                axis, disposition disagrees or surface_hash is inconsistent.
        """
        candidate_ids = tuple(value.candidate_id for value in self.candidates)
        if candidate_ids != tuple(dict.fromkeys(candidate_ids)):
            raise ValueError("Alpha current stability candidate axis is not unique")
        stable = tuple(value.candidate_id for value in self.candidates if value.stable)
        if stable != self.dual_qualified_candidate_ids:
            raise ValueError("Alpha current stability surface does not reconcile")
        expected = "DUAL_QUALIFIED_MODELS_AVAILABLE" if stable else "NO_STABLE_CURRENT_ALPHA_MODEL"
        if self.disposition != expected:
            raise ValueError("Alpha current stability disposition is inconsistent")
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise ValueError("Alpha current stability surface hash is invalid")
        return self


class CurrentFormationScoreChunkRef(_Contract):
    """Bind selected current formation scores to the decision, estimator and stability."""

    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str
    decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str
    row_count: int = Field(ge=1)


class CurrentFormationScoreSnapshot(_Contract):
    """Seal selected current scores, exact listing coverage and availability counts.

    The chunk must match all declared decision/source/model lineage. Training cannot extend beyond
    the published formation.
    """

    kind: Literal["CurrentFormationScoreSnapshot"] = "CurrentFormationScoreSnapshot"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str
    decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    training_cutoff: date
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    availability_counts: dict[Literal["SCORED", "FEATURE_INCOMPLETE"], int]
    chunk: CurrentFormationScoreChunkRef
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> CurrentFormationScoreSnapshot:
        """Require exhaustive score coverage, exact chunk lineage and causal training.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Availability/row counts, request/source/decision/model/stability/listing
                lineage, training clock or snapshot_hash disagrees.
        """
        if sum(self.availability_counts.values()) != len(self.ordered_listing_ids):
            raise ValueError("current Alpha score availability does not reconcile")
        if self.chunk.row_count != len(self.ordered_listing_ids):
            raise ValueError("current Alpha score chunk does not cover the Foundation axis")
        if (
            self.chunk.request_hash != self.request_hash
            or self.chunk.foundation_hash != self.foundation_hash
            or self.chunk.logical_panel_hash != self.logical_panel_hash
            or self.chunk.selected_candidate_id != self.selected_candidate_id
            or self.chunk.decision_hash != self.decision_hash
            or self.chunk.estimator_state_hash != self.estimator_state_hash
            or self.chunk.stability_assessment_hash != self.stability_assessment_hash
            or self.chunk.formation_session != self.formation_session
            or self.chunk.ordered_listing_ids_hash != canonical_hash(self.ordered_listing_ids)
        ):
            raise ValueError("current Alpha score chunk lineage differs")
        if self.training_cutoff > self.formation_session:
            raise ValueError("current Alpha training cutoff exceeds formation")
        if self.snapshot_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"snapshot_hash"})
        ):
            raise ValueError("current Alpha score snapshot hash is invalid")
        return self


class AlphaCurrentRuntimeMarker(_Contract):
    """Seal current runtime publication lineage and disposition-specific children.

    The marker retains development evidence, optional current stability/decision/score children and
    blocked failure state. Validation admits the current payload and its declared historical
    serialization.
    """

    kind: Literal["AlphaCurrentRuntimeMarker"] = "AlphaCurrentRuntimeMarker"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_fold_surface_hashes: tuple[str, ...] = Field(min_length=1)
    candidate_report_hashes: tuple[str, ...] = Field(min_length=1)
    candidate_inference_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    viability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_stability_surface_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_candidate_score_child_hashes: tuple[str, ...] = ()
    decision_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reproducibility_report_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_score_snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_estimator_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_refit_assessment_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    disposition: Literal[
        "CURRENT_SCORE_PUBLISHED",
        "NO_ADMISSIBLE_ALPHA_MODEL",
        "NO_STABLE_CURRENT_ALPHA_MODEL",
        "BLOCKED",
    ]
    failure_code: str | None = Field(default=None, max_length=120)
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCurrentRuntimeMarker:
        """Require disposition-complete child evidence and a compatible runtime marker identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Report/inference axes differ, terminal children are incomplete/inconsistent,
                blocked failure presence disagrees or neither admitted serialization matches
                marker_hash.
        """
        if len(self.candidate_report_hashes) != len(self.candidate_inference_evidence_hashes):
            raise ValueError("Alpha current marker candidate evidence axes differ")
        if self.disposition == "CURRENT_SCORE_PUBLISHED" and (
            self.decision_hash is None
            or self.current_stability_surface_hash is None
            or self.current_score_snapshot_hash is None
            or self.current_estimator_state_hash is None
            or self.current_refit_assessment_hash is None
        ):
            raise ValueError("Alpha current-score marker lacks decision or score")
        if self.disposition == "NO_STABLE_CURRENT_ALPHA_MODEL" and (
            self.current_stability_surface_hash is None or self.decision_hash is not None
        ):
            raise ValueError("Alpha no-stable-model marker is inconsistent")
        if (self.disposition == "BLOCKED") != (self.failure_code is not None):
            raise ValueError("Alpha runtime blocked failure identity is inconsistent")
        identity = self.model_dump(mode="json", exclude={"marker_hash"})
        accepted = {canonical_hash(identity)}
        legacy = dict(identity)
        legacy.pop("current_stability_surface_hash", None)
        legacy.pop("current_candidate_score_child_hashes", None)
        accepted.add(canonical_hash(legacy))
        if self.marker_hash not in accepted:
            raise ValueError("Alpha current runtime marker hash is invalid")
        return self


class AlphaCurrentRuntimeReceipt(_Contract):
    """Seal admitted/completed work, resource counts and the aware completion clock.

    Exact reuse records zero Panel/outcome reads and zero fit/predict/metric/agent/provider calls;
    completed candidate-fold work cannot exceed admission.
    """

    kind: Literal["AlphaCurrentRuntimeReceipt"] = "AlphaCurrentRuntimeReceipt"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: Literal["PUBLISHED", "REUSED_EXACT"]
    admitted_candidate_count: int = Field(ge=1)
    admitted_fold_count: int = Field(ge=1)
    completed_candidate_fold_count: int = Field(ge=0)
    panel_read_count: int = Field(ge=0)
    outcome_read_count: int = Field(ge=0)
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    metric_call_count: int = Field(ge=0)
    agent_model_call_count: int = Field(ge=0)
    provider_call_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0, allow_inf_nan=False)
    peak_rss_bytes: int = Field(ge=0)
    completed_at: datetime
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCurrentRuntimeReceipt:
        """Require an aware completion clock, bounded work and zero-work exact reuse.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The clock is naive, candidate-fold work exceeds admission, exact reuse
                records numerical/provider work or receipt_hash is inconsistent.
        """
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("Alpha current runtime receipt clock is invalid")
        if self.completed_candidate_fold_count > (
            self.admitted_candidate_count * self.admitted_fold_count
        ):
            raise ValueError("Alpha candidate-fold work exceeds admission")
        if self.action == "REUSED_EXACT" and any(
            (
                self.panel_read_count,
                self.outcome_read_count,
                self.fit_call_count,
                self.predict_call_count,
                self.metric_call_count,
                self.agent_model_call_count,
                self.provider_call_count,
            )
        ):
            raise ValueError("Alpha exact replay performed work")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("Alpha current runtime receipt hash is invalid")
        return self


class AlphaCurrentSafeProjection(_Contract):
    """Seal public current-score availability, failure state, limits and aware update clock."""

    kind: Literal["AlphaCurrentSafeProjection"] = "AlphaCurrentSafeProjection"
    status: Literal[
        "CURRENT_FORMATION_SCORE_READY",
        "NO_ADMISSIBLE_ALPHA_MODEL",
        "NO_STABLE_CURRENT_ALPHA_MODEL",
        "BLOCKED",
    ]
    next_action: Literal[
        "READY_TO_START_RISK_RESEARCH_DESK",
        "STOP_WITH_EVIDENCE",
        "PM_OR_HUMAN_REVIEW_REQUIRED",
    ]
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    viability_assessment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str | None = None
    current_score_snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidate_count: int = Field(ge=1)
    fold_count: int = Field(ge=1)
    hypothesis_count: int = Field(ge=0)
    shadow_divergence_observed: bool
    failure_code: str | None = Field(default=None, max_length=120)
    limitations: tuple[str, ...]
    updated_at: datetime
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCurrentSafeProjection:
        """Require an aware update clock and status-complete score/failure evidence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The clock is naive, a ready score/selection is absent, blocked failure
                presence disagrees or projection_hash is inconsistent.
        """
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("Alpha safe projection clock is invalid")
        if self.status == "CURRENT_FORMATION_SCORE_READY" and (
            self.selected_candidate_id is None or self.current_score_snapshot_hash is None
        ):
            raise ValueError("Alpha ready projection lacks current score")
        if (self.status == "BLOCKED") != (self.failure_code is not None):
            raise ValueError("Alpha safe projection failure identity is inconsistent")
        if self.projection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"projection_hash"})
        ):
            raise ValueError("Alpha current projection hash is invalid")
        return self


__all__ = [
    "AlphaCurrentCandidateQualification",
    "AlphaCurrentCandidateScoreChild",
    "AlphaCurrentCandidateScoreChunkRef",
    "AlphaCurrentRuntimeMarker",
    "AlphaCurrentRuntimeReceipt",
    "AlphaCurrentSafeProjection",
    "AlphaCurrentStabilitySurface",
    "AlphaStabilityCheck",
    "CurrentFormationScoreChunkRef",
    "CurrentFormationScoreSnapshot",
    "CurrentRefitStabilityAssessment",
    "WorkspaceObservationHistoryHead",
    "WorkspaceObservationHistoryMarker",
    "WorkspaceObservationHistoryPart",
    "seal_current_contract",
]
