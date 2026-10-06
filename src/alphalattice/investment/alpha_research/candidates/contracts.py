"""Typed contracts of Alpha's candidate registry: candidates, evidence, the set and the stop."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..experiments.contracts import (
    MODEL_SPEC_ADAPTER,
    AlphaExperimentBatch,
    AlphaExperimentBatchResult,
    AlphaExperimentCandidateResult,
    AlphaResearchProgram,
    ElasticNetModelSpec,
    LassoModelSpec,
    LegacyModelSpec,
    ResolvedRegularizedLinearSpec,
    RidgeModelSpec,
    StoredModelSpec,
    candidate_id_for_spec,
    resolve_model_spec,
)
from ..experiments.contracts import (
    _validate_hash as _validate_hash,
)
from ..experiments.contracts import (
    _validate_hash_compatible as _validate_hash_compatible,
)
from ..experiments.contracts import (
    seal_contract as seal_contract,
)

Hash = str


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaModelResearchGoalCriteria(_Contract):
    """Seal bounded Alpha research targets, batch limits and model-family authority.

    Legacy criteria carry the fixed three linear families. Current criteria bind the model mandate
    and leave family authority to it.
    """

    kind: Literal["AlphaModelResearchGoalCriteria"] = "AlphaModelResearchGoalCriteria"
    target_current_qualified_candidates: int = Field(default=3, ge=1)
    initial_batch_size: int = Field(default=6, ge=1)
    refinement_batch_max_size: int = Field(default=3, ge=1)
    max_batch_count: int = Field(default=2, ge=1)
    max_unique_new_specs: int = Field(default=9, ge=1)
    allowed_families: tuple[str, ...] | None = (
        "ridge",
        "lasso",
        "elastic_net",
    )
    model_mandate_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    terminal_outputs: tuple[
        Literal["CURRENT_ALPHA_CANDIDATE_SET", "EVIDENCE_COMPLETE_SCIENTIFIC_STOP"], ...
    ] = ("CURRENT_ALPHA_CANDIDATE_SET", "EVIDENCE_COMPLETE_SCIENTIFIC_STOP")
    criteria_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_criteria(self) -> Self:
        """Require exclusive legacy/current authority and compatible criteria identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Legacy families differ, current criteria duplicate family authority or
                criteria_hash is inconsistent.
        """
        if self.model_mandate_hash is None:
            if self.allowed_families != ("ridge", "lasso", "elastic_net"):
                raise ValueError("legacy Alpha research family authority changed")
        elif self.allowed_families is not None:
            raise ValueError("current Alpha criteria duplicates model Mandate authority")
        _validate_hash_compatible(
            self,
            "criteria_hash",
            optional_fields=("model_mandate_hash",),
        )
        return self


class AlphaCandidateStatus(StrEnum):
    """Name candidate development, current qualification and final selection states."""

    PROPOSED = "PROPOSED"
    DEVELOPMENT_RUNNING = "DEVELOPMENT_RUNNING"
    DEVELOPMENT_EVALUATED = "DEVELOPMENT_EVALUATED"
    DEVELOPMENT_ADMISSIBLE = "DEVELOPMENT_ADMISSIBLE"
    DEVELOPMENT_REJECTED = "DEVELOPMENT_REJECTED"
    CURRENT_QUALIFICATION_RUNNING = "CURRENT_QUALIFICATION_RUNNING"
    CURRENT_QUALIFIED = "CURRENT_QUALIFIED"
    CURRENT_STABILITY_REJECTED = "CURRENT_STABILITY_REJECTED"
    SELECTED_FOR_CANDIDATE_SET = "SELECTED_FOR_CANDIDATE_SET"
    NOT_SELECTED = "NOT_SELECTED"


class AlphaOosEvidenceClassification(StrEnum):
    """Name positive, mixed, undetected, negative or unavailable out-of-sample evidence."""

    POSITIVE_OOS_EVIDENCE = "POSITIVE_OOS_EVIDENCE"
    MIXED_OOS_EVIDENCE = "MIXED_OOS_EVIDENCE"
    NO_DETECTABLE_EFFECT = "NO_DETECTABLE_EFFECT"
    NEGATIVE_OOS_EVIDENCE = "NEGATIVE_OOS_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class AlphaOosEvidenceRecord(_Contract):
    """Seal one candidate out-of-sample metrics, adjusted significance and disposition.

    Unavailable evidence carries failure reasons; available evidence retains classification reason
    codes without failure codes.
    """

    kind: Literal["AlphaOosEvidenceRecord"] = "AlphaOosEvidenceRecord"
    candidate_id: str
    mean_rank_ic: float | None = Field(default=None, allow_inf_nan=False)
    mean_gross_decile_spread: float | None = Field(default=None, allow_inf_nan=False)
    rank_ic_raw_p_value: float | None = Field(default=None, ge=0, le=1)
    spread_raw_p_value: float | None = Field(default=None, ge=0, le=1)
    holm_adjusted_p_value: float | None = Field(default=None, ge=0, le=1)
    classification: AlphaOosEvidenceClassification
    reason_codes: tuple[str, ...] = Field(min_length=1)
    failure_codes: tuple[str, ...] = ()
    evidence_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_evidence(self) -> Self:
        """Require failure evidence exactly for unavailable OOS results and exact record identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Unavailable evidence lacks failures, available evidence carries failures or
                evidence_hash is inconsistent.
        """
        if self.classification is AlphaOosEvidenceClassification.INSUFFICIENT_EVIDENCE:
            if not self.failure_codes:
                raise ValueError("Insufficient Alpha evidence lacks a reason")
        elif self.failure_codes:
            raise ValueError("Available Alpha evidence carries failure codes")
        _validate_hash(self, "evidence_hash")
        return self


class AlphaOosEvidenceAssessment(_Contract):
    """Seal a unique tested candidate family and reconcile positive/mixed dispositions."""

    kind: Literal["AlphaOosEvidenceAssessment"] = "AlphaOosEvidenceAssessment"
    program_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[AlphaOosEvidenceRecord, ...] = Field(min_length=1)
    positive_candidate_ids: tuple[str, ...]
    mixed_candidate_ids: tuple[str, ...]
    hypothesis_count: int = Field(ge=1)
    assessment_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_assessment(self) -> Self:
        """Require one unique hypothesis per candidate and reconciled OOS disposition axes.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Candidates repeat, hypothesis_count differs, positive/mixed axes disagree or
                assessment_hash is inconsistent.
        """
        ids = tuple(value.candidate_id for value in self.records)
        if len(set(ids)) != len(ids) or self.hypothesis_count != len(ids):
            raise ValueError("Alpha OOS evidence family is not canonical")
        expected_positive = tuple(
            value.candidate_id
            for value in self.records
            if value.classification is AlphaOosEvidenceClassification.POSITIVE_OOS_EVIDENCE
        )
        expected_mixed = tuple(
            value.candidate_id
            for value in self.records
            if value.classification is AlphaOosEvidenceClassification.MIXED_OOS_EVIDENCE
        )
        if (
            self.positive_candidate_ids != expected_positive
            or self.mixed_candidate_ids != expected_mixed
        ):
            raise ValueError("Alpha OOS evidence dispositions do not reconcile")
        _validate_hash(self, "assessment_hash")
        return self


class AlphaHistoricalCandidateEvidence(_Contract):
    """Retain historical current-stability rejection evidence for a stored model recipe."""

    kind: Literal["AlphaHistoricalCandidateEvidence"] = "AlphaHistoricalCandidateEvidence"
    candidate_id: str
    spec: StoredModelSpec
    failure_codes: tuple[str, ...] = Field(min_length=1)
    stability_assessment_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_evidence(self) -> Self:
        """Require the exact historical candidate rejection evidence identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: evidence_hash differs from the declared rejection payload.
        """
        _validate_hash(self, "evidence_hash")
        return self


class AlphaCandidateRecord(_Contract):
    """Seal one recipe candidate lifecycle and its development/current evidence lineage.

    Qualified and terminal selection states require model state, current score and stability
    identities with no failure codes. Rejection states retain failure evidence.
    """

    kind: Literal["AlphaCandidateRecord"] = "AlphaCandidateRecord"
    spec: StoredModelSpec
    candidate_id: str = Field(pattern=r"^(?:agent-linear|alpha-candidate)-[0-9a-f]{16}$")
    batch_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    status: AlphaCandidateStatus
    development_result_ref: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    development_candidate_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_state_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stability_assessment_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_score_child_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    holm_adjusted_p_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    oos_evidence_classification: AlphaOosEvidenceClassification | None = None
    mean_rank_ic: float | None = Field(default=None, allow_inf_nan=False)
    mean_gross_decile_spread: float | None = Field(default=None, allow_inf_nan=False)
    stability_failed_check_ids: tuple[str, ...] = ()
    current_score_mean: float | None = Field(default=None, allow_inf_nan=False)
    current_score_std: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    current_score_coverage: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    failure_codes: tuple[str, ...] = ()
    record_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_record(self) -> Self:
        """Require recipe-derived candidate identity and status-complete evidence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Candidate identity differs from its recipe, qualified evidence is
                incomplete, rejection reasons are absent or compatible record_hash is inconsistent.
        """
        if self.candidate_id != candidate_id_for_spec(self.spec):
            raise ValueError("Alpha candidate identity differs from its model spec")
        if self.status in {
            AlphaCandidateStatus.CURRENT_QUALIFIED,
            AlphaCandidateStatus.SELECTED_FOR_CANDIDATE_SET,
            AlphaCandidateStatus.NOT_SELECTED,
        } and (
            self.development_candidate_hash is None
            or self.current_state_hash is None
            or self.stability_assessment_hash is None
            or self.current_score_child_hash is None
            or self.failure_codes
        ):
            raise ValueError("qualified or terminal Alpha candidate lacks current evidence")
        if (
            self.status
            in {
                AlphaCandidateStatus.DEVELOPMENT_REJECTED,
                AlphaCandidateStatus.CURRENT_STABILITY_REJECTED,
            }
            and not self.failure_codes
        ):
            raise ValueError("rejected Alpha candidate lacks failure evidence")
        _validate_hash_compatible(
            self,
            "record_hash",
            optional_fields=(
                "oos_evidence_classification",
                "mean_rank_ic",
                "mean_gross_decile_spread",
            ),
        )
        return self


class AlphaCandidateRegistrySnapshot(_Contract):
    """Seal a unique candidate registry revision with explicit predecessor and historical evidence.

    Recipe and candidate identities are unique; historical exhausted recipes/rejections are retained
    separately from the current candidate axis.
    """

    kind: Literal["AlphaCandidateRegistrySnapshot"] = "AlphaCandidateRegistrySnapshot"
    program_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=0, le=32)
    predecessor_registry_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[AlphaCandidateRecord, ...]
    exhausted_legacy_specs: tuple[ResolvedRegularizedLinearSpec, ...] = ()
    historical_current_stability_rejections: tuple[AlphaHistoricalCandidateEvidence, ...] = ()
    latest_hypothesis_family_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    registry_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_registry(self) -> Self:
        """Require unique recipe/candidate/history axes and revision-consistent predecessor lineage.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: An axis repeats, predecessor presence disagrees with revision or
                registry_hash is inconsistent.
        """
        spec_hashes = tuple(item.spec.spec_hash for item in self.candidates)
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if len(set(spec_hashes)) != len(spec_hashes) or len(set(candidate_ids)) != len(
            candidate_ids
        ):
            raise ValueError("Alpha Candidate Registry axis is not unique")
        exhausted_hashes = tuple(value.spec_hash for value in self.exhausted_legacy_specs)
        if len(set(exhausted_hashes)) != len(exhausted_hashes) or len(
            {value.candidate_id for value in self.historical_current_stability_rejections}
        ) != len(self.historical_current_stability_rejections):
            raise ValueError("Alpha Candidate Registry repeats historical evidence")
        if self.revision == 0 and self.predecessor_registry_hash is not None:
            raise ValueError("initial Alpha Candidate Registry has a predecessor")
        if self.revision > 0 and self.predecessor_registry_hash is None:
            raise ValueError("revised Alpha Candidate Registry lacks predecessor")
        _validate_hash(self, "registry_hash")
        return self


class AlphaCandidateScoreCorrelation(_Contract):
    """Seal a canonical candidate pair correlation and common scored-row count."""

    kind: Literal["AlphaCandidateScoreCorrelation"] = "AlphaCandidateScoreCorrelation"
    left_candidate_id: str
    right_candidate_id: str
    common_scored_count: int = Field(ge=0)
    correlation: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    correlation_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_correlation(self) -> Self:
        """Require an ordered peer pair and count-consistent correlation availability.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The left identifier is not below the right, correlation exists with fewer
                than two rows or correlation_hash is inconsistent.
        """
        if self.left_candidate_id >= self.right_candidate_id:
            raise ValueError("Alpha score-correlation pair is not canonical")
        if self.common_scored_count < 2 and self.correlation is not None:
            raise ValueError("Alpha score-correlation availability is inconsistent")
        _validate_hash(self, "correlation_hash")
        return self


class AlphaQualificationSnapshot(_Contract):
    """Seal attempted/nominated/current candidate axes and one qualification evidence owner.

    Exactly one viability or out-of-sample assessment supplies evidence authority. Correlation pairs
    refer only to current-qualified candidates.
    """

    kind: Literal["AlphaQualificationSnapshot"] = "AlphaQualificationSnapshot"
    program_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    hypothesis_family_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    attempted_candidate_ids: tuple[str, ...] = Field(min_length=1)
    nominated_candidate_ids: tuple[str, ...]
    development_admissible_ids: tuple[str, ...]
    current_qualified_ids: tuple[str, ...]
    current_stability_rejected_ids: tuple[str, ...]
    viability_assessment_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    oos_evidence_assessment_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    score_correlations: tuple[AlphaCandidateScoreCorrelation, ...] = ()
    qualification_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_qualification(self) -> Self:
        """Require one evidence owner and valid nomination, disposition and correlation axes.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Assessment owner is ambiguous, nomination/qualified/rejected axes disagree,
                correlation pairs repeat or leave the qualified set, or compatible
                qualification_hash is inconsistent.
        """
        if (self.viability_assessment_hash is None) == (self.oos_evidence_assessment_hash is None):
            raise ValueError("Alpha qualification evidence owner is ambiguous")
        attempted = set(self.attempted_candidate_ids)
        nominated = set(self.nominated_candidate_ids)
        if len(attempted) != len(self.attempted_candidate_ids) or not nominated <= attempted:
            raise ValueError("Alpha qualification candidate axis is invalid")
        if set(self.current_qualified_ids) & set(self.current_stability_rejected_ids):
            raise ValueError("Alpha current qualification dispositions overlap")
        if not set(self.current_qualified_ids) <= nominated:
            raise ValueError("Alpha current qualification includes an unnominated candidate")
        pairs = tuple(
            (value.left_candidate_id, value.right_candidate_id) for value in self.score_correlations
        )
        if len(set(pairs)) != len(pairs) or any(
            not set(pair) <= set(self.current_qualified_ids) for pair in pairs
        ):
            raise ValueError("Alpha qualification score correlations are not current-qualified")
        _validate_hash_compatible(
            self,
            "qualification_hash",
            optional_fields=("oos_evidence_assessment_hash",),
        )
        return self


class AlphaGoalDisposition(StrEnum):
    """Name satisfied, continuing, exhausted or blocked bounded research progress."""

    SATISFIED = "SATISFIED"
    CONTINUE = "CONTINUE"
    EXHAUSTED_STOP = "EXHAUSTED_STOP"
    BLOCKED = "BLOCKED"


class AlphaGoalProgress(_Contract):
    """Seal candidate counts, remaining research bounds and the resulting goal disposition.

    The declared target and batch bound take precedence over historical defaults. Satisfied,
    continuing and exhausted states reconcile counts and remaining work.
    """

    kind: Literal["AlphaGoalProgress"] = "AlphaGoalProgress"
    criteria_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    proposed_count: int = Field(ge=0)
    development_admissible_count: int = Field(ge=0)
    development_rejected_count: int = Field(ge=0)
    current_qualified_count: int = Field(ge=0)
    current_stability_rejected_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    completed_batch_count: int = Field(ge=0)
    remaining_batch_count: int = Field(ge=0)
    remaining_spec_count: int = Field(ge=0)
    target_candidate_count: int | None = Field(default=None, ge=1)
    max_batch_count: int | None = Field(default=None, ge=1)
    unresolved_failure_codes: tuple[str, ...]
    disposition: AlphaGoalDisposition
    progress_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_progress(self) -> Self:
        """Require bounded counts and a disposition matching target and remaining work.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Counts exceed authority, satisfaction/continuation/exhaustion disagrees with
                target or budget, or compatible progress_hash is inconsistent.
        """
        target = self.target_candidate_count or 3
        max_batches = self.max_batch_count or 2
        if (
            self.selected_count > target
            or self.completed_batch_count > max_batches
            or self.remaining_batch_count > max_batches
        ):
            raise ValueError("Alpha Goal progress exceeds its bound authority")
        if (
            self.disposition is AlphaGoalDisposition.SATISFIED
            and self.current_qualified_count < target
        ):
            raise ValueError("Alpha Goal cannot be satisfied without its target candidates")
        if self.disposition is AlphaGoalDisposition.CONTINUE and (
            self.current_qualified_count >= target
            or self.remaining_batch_count == 0
            or self.remaining_spec_count == 0
        ):
            raise ValueError("Alpha Goal continuation does not match remaining work")
        if self.disposition is AlphaGoalDisposition.EXHAUSTED_STOP and (
            self.current_qualified_count >= target
            or (self.remaining_batch_count > 0 and self.remaining_spec_count > 0)
        ):
            raise ValueError("Alpha Goal exhaustion does not match its budget")
        _validate_hash_compatible(
            self,
            "progress_hash",
            optional_fields=("target_candidate_count", "max_batch_count"),
        )
        return self


class CurrentAlphaCandidateSetSnapshot(_Contract):
    """Seal the mandate-sized ordered current candidate set and complete peer evidence axes.

    Optional Sector component lineage admits exactly stock-only and stock-plus-Sector modes
    together. Each selected peer retains score, estimator-state and stability identities.
    """

    kind: Literal["CurrentAlphaCandidateSetSnapshot"] = "CurrentAlphaCandidateSetSnapshot"
    program_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_candidate_ids: tuple[str, ...] = Field(min_length=1)
    current_score_child_hashes: tuple[Hash, ...] = Field(min_length=1)
    estimator_state_hashes: tuple[Hash, ...] = Field(min_length=1)
    stability_assessment_hashes: tuple[Hash, ...] = Field(min_length=1)
    model_mandate_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_candidate_count: int | None = Field(default=None, ge=1)
    sector_ema_manifest_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    downstream_score_modes: tuple[Literal["STOCK_ONLY", "STOCK_PLUS_SECTOR_COMPONENT"], ...] = ()
    limitations: tuple[str, ...]
    snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_snapshot(self) -> Self:
        """Require complete unique mandate-sized peer axes and paired Sector mode authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: A selected/evidence axis has wrong size/duplicates, Sector/mode authority is
                partial or undeclared, or compatible snapshot_hash is inconsistent.
        """
        expected = self.target_candidate_count or 3
        if (
            len(self.ordered_candidate_ids) != expected
            or len(set(self.ordered_candidate_ids)) != expected
        ):
            raise ValueError("current Alpha candidate-set peer axis is invalid")
        for values in (
            self.current_score_child_hashes,
            self.estimator_state_hashes,
            self.stability_assessment_hashes,
        ):
            if (
                len(values) != expected
                or len(set(values)) != expected
                or any(len(value) != 64 for value in values)
            ):
                raise ValueError("current Alpha candidate-set evidence axis is invalid")
        if (self.sector_ema_manifest_hash is None) != (not self.downstream_score_modes):
            raise ValueError("current Alpha sector component authority is incomplete")
        if self.downstream_score_modes and self.downstream_score_modes != (
            "STOCK_ONLY",
            "STOCK_PLUS_SECTOR_COMPONENT",
        ):
            raise ValueError("current Alpha downstream score modes are invalid")
        _validate_hash_compatible(
            self,
            "snapshot_hash",
            optional_fields=(
                "sector_ema_manifest_hash",
                "downstream_score_modes",
                "model_mandate_hash",
                "target_candidate_count",
            ),
        )
        return self


class AlphaResearchScientificStop(_Contract):
    """Seal exhausted research evidence below the candidate target and its next authority steps.

    Exactly one legacy-spec or current-recipe attempt axis is retained, with unique identities. A
    stop cannot represent a target already satisfied.
    """

    kind: Literal["AlphaResearchScientificStop"] = "AlphaResearchScientificStop"
    program_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    goal_progress_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    attempted_spec_hashes: tuple[Hash, ...] = ()
    attempted_recipe_hashes: tuple[Hash, ...] | None = None
    current_qualified_count: int = Field(ge=0)
    model_mandate_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    target_candidate_count: int | None = Field(default=None, ge=1)
    failure_codes: tuple[str, ...] = Field(min_length=1)
    next_research_actions: tuple[str, ...] = Field(min_length=1)
    science_policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    stop_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_stop(self) -> Self:
        """Require one unique attempt axis and an unsatisfied target before scientific stop.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Attempt authority is ambiguous/duplicated, the target is already met or
                compatible stop_hash is inconsistent.
        """
        recipe_hashes = self.attempted_recipe_hashes or ()
        if bool(self.attempted_spec_hashes) == bool(recipe_hashes):
            raise ValueError("Alpha scientific stop attempted-recipe authority is ambiguous")
        attempted = recipe_hashes or self.attempted_spec_hashes
        if len(set(attempted)) != len(attempted):
            raise ValueError("Alpha scientific stop repeats an attempted recipe")
        expected = self.target_candidate_count or 3
        if self.current_qualified_count >= expected:
            raise ValueError("Alpha scientific stop already satisfies its candidate target")
        _validate_hash_compatible(
            self,
            "stop_hash",
            optional_fields=(
                "attempted_recipe_hashes",
                "model_mandate_hash",
                "target_candidate_count",
            ),
        )
        return self


class AlphaGoalResearchMarker(_Contract):
    """Seal the candidate-set or scientific-stop terminal lineage without admitting Risk.

    The terminal disposition determines which child identity is present; risk_admitted remains
    False.
    """

    kind: Literal["AlphaGoalResearchMarker"] = "AlphaGoalResearchMarker"
    program_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    goal_progress_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    board_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The retired goal loop's board a legacy marker concluded; a qualification's names none,
    its family bound through its Program (GR3)."""
    candidate_set_snapshot_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scientific_stop_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["CURRENT_ALPHA_CANDIDATE_SET_READY", "NO_STABLE_CURRENT_ALPHA_MODEL"]
    risk_admitted: Literal[False] = False
    marker_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> Self:
        """Require the disposition-selected terminal child and exact marker identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Candidate-set/stop presence disagrees with disposition or marker_hash is
                inconsistent.
        """
        if self.disposition == "CURRENT_ALPHA_CANDIDATE_SET_READY":
            valid = (
                self.candidate_set_snapshot_hash is not None and self.scientific_stop_hash is None
            )
        else:
            valid = (
                self.candidate_set_snapshot_hash is None and self.scientific_stop_hash is not None
            )
        if not valid:
            raise ValueError("Alpha goal-research terminal marker is inconsistent")
        _validate_hash(self, "marker_hash")
        return self


class AlphaGoalResearchSafeProjection(_Contract):
    """Seal terminal Alpha counts, selected identities, next action and a timezone-aware clock."""

    kind: Literal["AlphaGoalResearchSafeProjection"] = "AlphaGoalResearchSafeProjection"
    program_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    marker_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    goal_target: int = Field(ge=1)
    current_qualified_count: int = Field(ge=0)
    attempted_spec_count: int = Field(ge=0)
    completed_batch_count: int = Field(ge=0)
    status: Literal["CURRENT_ALPHA_CANDIDATE_SET_READY", "NO_STABLE_CURRENT_ALPHA_MODEL"]
    next_action: Literal[
        "READY_TO_DESIGN_MULTI_MODEL_DOWNSTREAM_CONSUMPTION",
        "STOP_WITH_EVIDENCE",
    ]
    selected_candidate_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...]
    updated_at: datetime
    projection_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_projection(self) -> Self:
        """Require a timezone-aware terminal projection with matching selection and next action.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The clock lacks an offset, selection/next action disagrees with terminal
                status or projection_hash is inconsistent.
        """
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("Alpha goal-research projection clock is invalid")
        if self.status == "CURRENT_ALPHA_CANDIDATE_SET_READY":
            valid = (
                len(self.selected_candidate_ids) == self.goal_target
                and self.next_action == "READY_TO_DESIGN_MULTI_MODEL_DOWNSTREAM_CONSUMPTION"
            )
        else:
            valid = not self.selected_candidate_ids and self.next_action == "STOP_WITH_EVIDENCE"
        if not valid:
            raise ValueError("Alpha goal-research safe projection is inconsistent")
        _validate_hash(self, "projection_hash")
        return self


__all__ = [
    "MODEL_SPEC_ADAPTER",
    "AlphaCandidateRecord",
    "AlphaCandidateRegistrySnapshot",
    "AlphaCandidateScoreCorrelation",
    "AlphaCandidateStatus",
    "AlphaExperimentBatch",
    "AlphaExperimentBatchResult",
    "AlphaExperimentCandidateResult",
    "AlphaGoalDisposition",
    "AlphaGoalProgress",
    "AlphaGoalResearchMarker",
    "AlphaGoalResearchSafeProjection",
    "AlphaHistoricalCandidateEvidence",
    "AlphaModelResearchGoalCriteria",
    "AlphaOosEvidenceAssessment",
    "AlphaOosEvidenceClassification",
    "AlphaOosEvidenceRecord",
    "AlphaQualificationSnapshot",
    "AlphaResearchProgram",
    "AlphaResearchScientificStop",
    "CurrentAlphaCandidateSetSnapshot",
    "ElasticNetModelSpec",
    "LassoModelSpec",
    "LegacyModelSpec",
    "ResolvedRegularizedLinearSpec",
    "RidgeModelSpec",
    "StoredModelSpec",
    "candidate_id_for_spec",
    "resolve_model_spec",
    "seal_contract",
]
