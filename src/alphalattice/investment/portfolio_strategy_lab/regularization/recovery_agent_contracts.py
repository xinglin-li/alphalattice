"""Review, decision and publication contracts for Portfolio evidence recovery."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import PortfolioLabContract, SectorDeviationPenaltyPolicy
from .contracts import PortfolioResearchAlphaSource
from .recovery_contracts import (
    PortfolioCrossFoldEvidenceClassification,
    PortfolioCrossFoldGateAudit,
    PortfolioDevelopmentPlausibilitySummary,
)

_HASH = r"^[0-9a-f]{64}$"


def _identity(value: PortfolioLabContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("portfolio_strategy_lab.recovery_review_identity_invalid")


class PortfolioRecoveryDecisionRoute(StrEnum):
    """Name bounded recovery routes including conditional freeze and evidence-backed exhaustion."""

    FREEZE_CONDITIONAL_CANDIDATES = "FREEZE_CONDITIONAL_CANDIDATES"
    ROUTE_TO_ALPHA = "ROUTE_TO_ALPHA"
    ROUTE_TO_RISK = "ROUTE_TO_RISK"
    REQUEST_BENCHMARK_RELATIVE_OBJECTIVE_STUDY = "REQUEST_BENCHMARK_RELATIVE_OBJECTIVE_STUDY"
    REQUEST_CAUSAL_ALPHA_BUDGET_STUDY = "REQUEST_CAUSAL_ALPHA_BUDGET_STUDY"
    NO_REMAINING_PLAUSIBLE_HYPOTHESIS = "NO_REMAINING_PLAUSIBLE_HYPOTHESIS"


class PortfolioRecoveryBoardTask(StrEnum):
    """Declare open analysis, constrained decision, completion or deterministic fallback."""

    OPEN_ANALYSIS = "OPEN_ANALYSIS"
    CONSTRAINED_DECISION = "CONSTRAINED_DECISION"
    COMPLETE = "COMPLETE"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


class PortfolioRecoveryCandidateEvidence(PortfolioLabContract):
    """Retain semantic candidate controls and conservative cross-fold recovery diagnostics."""

    semantic_handle: str = Field(min_length=1, max_length=120)
    source: PortfolioResearchAlphaSource
    classification: PortfolioCrossFoldEvidenceClassification
    top_k: int = Field(ge=1)
    maximum_weight: float = Field(gt=0.0, le=1.0)
    risk_aversion: float = Field(gt=0.0)
    turnover_regularization: float = Field(ge=0.0)
    sector_deviation_penalty: float = Field(ge=0.0)
    fold_active_net_log_wealth: tuple[float, float, float]
    fold_active_block_standard_error: tuple[float, float, float]
    fold_sortino: tuple[float, float, float]
    fold_drawdown_excess: tuple[float, float, float]
    positive_fold_count: int = Field(ge=2, le=3)
    median_active_net_log_wealth: float
    worst_fold_active_net_log_wealth: float
    median_turnover: float = Field(ge=0.0)
    median_hhi: float = Field(ge=0.0, le=1.0)
    neighbor_support_count: int = Field(ge=2)


class PortfolioRecoveryDecisionDossier(PortfolioLabContract):
    """Bind ordered source audits, candidate evidence, frozen facts and bounded research routes."""

    kind: Literal["PortfolioRecoveryDecisionDossier"] = "PortfolioRecoveryDecisionDossier"
    evidence_audit_hash: str = Field(pattern=_HASH)
    shared_config_dossier_hash: str = Field(pattern=_HASH)
    historical_gate_findings: tuple[PortfolioCrossFoldGateAudit, PortfolioCrossFoldGateAudit]
    development_plausibility: tuple[
        PortfolioDevelopmentPlausibilitySummary,
        PortfolioDevelopmentPlausibilitySummary,
    ]
    shared_config_classification_counts: dict[str, dict[str, int]]
    shared_config_failure_counts: dict[str, dict[str, int]]
    candidate_evidence: tuple[PortfolioRecoveryCandidateEvidence, ...] = Field(max_length=2)
    frozen_facts: tuple[str, ...] = Field(min_length=1)
    unresolved_questions: tuple[str, ...] = Field(min_length=1)
    bounded_routes: tuple[PortfolioRecoveryDecisionRoute, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    dossier_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_dossier(self) -> Self:
        """Require the declared source pair and unique semantic candidate evidence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Audit/plausibility source order, candidate handle uniqueness or dossier_hash
                differs.
        """
        expected_sources = (
            PortfolioResearchAlphaSource.CURRENT_ALPHA,
            PortfolioResearchAlphaSource.SECTOR_AWARE_ALPHA,
        )
        if (
            tuple(value.source for value in self.historical_gate_findings) != expected_sources
            or tuple(value.source for value in self.development_plausibility) != expected_sources
            or len({value.semantic_handle for value in self.candidate_evidence})
            != len(self.candidate_evidence)
        ):
            raise ValueError("portfolio_strategy_lab.recovery_decision_dossier_invalid")
        _identity(self, "dossier_hash")
        return self


class PortfolioRecoveryBoundedDecision(PortfolioLabContract):
    """Declare a constrained recovery route, evidence rationale and candidate selection."""

    route: PortfolioRecoveryDecisionRoute
    selected_candidate_handles: tuple[str, ...] = Field(max_length=2)
    competing_hypotheses: tuple[str, ...] = Field(min_length=2, max_length=8)
    evidence_rationale: str = Field(min_length=1, max_length=2400)
    extreme_winner_skepticism: str = Field(min_length=1, max_length=1000)
    falsified_or_exhausted_hypotheses: tuple[str, ...] = Field(max_length=8)
    relationship_to_open_analysis: str = Field(min_length=1, max_length=1600)
    constraint_impact: str = Field(min_length=1, max_length=1600)
    recommended_next_evidence: str = Field(min_length=1, max_length=1600)
    limitations_acknowledged: Literal[True] = True

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_decision(self) -> Self:
        """Require candidate selection exactly for freeze and evidence for exhaustion.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Freeze route disagrees with selected handles or exhausted-hypothesis route
                lacks falsified/exhausted evidence.
        """
        freezes = self.route is PortfolioRecoveryDecisionRoute.FREEZE_CONDITIONAL_CANDIDATES
        if freezes != bool(self.selected_candidate_handles):
            raise ValueError("portfolio_strategy_lab.recovery_candidate_selection_invalid")
        if (
            self.route is PortfolioRecoveryDecisionRoute.NO_REMAINING_PLAUSIBLE_HYPOTHESIS
            and not self.falsified_or_exhausted_hypotheses
        ):
            raise ValueError("portfolio_strategy_lab.recovery_stop_evidence_missing")
        return self


class PortfolioOpenResearchAnalysis(PortfolioLabContract):
    """Record competing explanations and a minimal discriminating research proposal."""

    title: str = Field(min_length=1, max_length=160)
    observed_anomaly: str = Field(min_length=1, max_length=1600)
    competing_hypotheses: tuple[str, ...] = Field(min_length=2, max_length=8)
    inference: str = Field(min_length=1, max_length=2400)
    recommended_research_direction: str = Field(min_length=1, max_length=2400)
    recommended_candidate_handles: tuple[str, ...] = Field(max_length=2)
    minimal_discriminative_experiment: str = Field(min_length=1, max_length=2400)
    expected_uncertainty_reduction: str = Field(min_length=1, max_length=1000)
    estimated_compute_class: Literal["LOW", "MEDIUM", "HIGH"]
    falsification_condition: str = Field(min_length=1, max_length=1600)
    posterior_branching_rule: str = Field(min_length=1, max_length=2000)
    limitations_acknowledged: Literal[True] = True


class PortfolioRecoveryResearchBoard(PortfolioLabContract):
    """Bind open/constrained work, ownership, bounded call failures and host corrections."""

    kind: Literal["PortfolioRecoveryResearchBoard"] = "PortfolioRecoveryResearchBoard"
    dossier_hash: str = Field(pattern=_HASH)
    review_binding_hash: str = Field(pattern=_HASH)
    current_task: PortfolioRecoveryBoardTask
    open_analysis: PortfolioOpenResearchAnalysis | None = None
    bounded_decision: PortfolioRecoveryBoundedDecision | None = None
    open_task_failure_count: int = Field(ge=0, le=5)
    constrained_task_failure_count: int = Field(ge=0, le=5)
    total_model_calls: int = Field(ge=0, le=12)
    typed_repair_count: int = Field(ge=0)
    host_feedback_codes: tuple[str, ...]
    decision_owner: Literal["PENDING", "AGENT", "DETERMINISTIC_FALLBACK"]
    board_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_board(self) -> Self:
        """Require task state, analysis/decision presence and ownership to agree.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Open, constrained, complete or fallback state contradicts record
                presence/owner, or board_hash differs.
        """
        opened = self.open_analysis is not None
        bounded = self.bounded_decision is not None
        valid = {
            PortfolioRecoveryBoardTask.OPEN_ANALYSIS: (
                not opened and not bounded and self.decision_owner == "PENDING"
            ),
            PortfolioRecoveryBoardTask.CONSTRAINED_DECISION: (
                opened and not bounded and self.decision_owner == "PENDING"
            ),
            PortfolioRecoveryBoardTask.COMPLETE: (
                opened and bounded and self.decision_owner == "AGENT"
            ),
            PortfolioRecoveryBoardTask.DETERMINISTIC_FALLBACK: (
                bounded and self.decision_owner == "DETERMINISTIC_FALLBACK"
            ),
        }[self.current_task]
        if not valid:
            raise ValueError("portfolio_strategy_lab.recovery_board_state_invalid")
        _identity(self, "board_hash")
        return self


class HistoricalPortfolioRecoveryResearchBoard(PortfolioLabContract):
    """Exact reader for pre-review-binding boards; never used for new work."""

    kind: Literal["PortfolioRecoveryResearchBoard"] = "PortfolioRecoveryResearchBoard"
    dossier_hash: str = Field(pattern=_HASH)
    current_task: PortfolioRecoveryBoardTask
    open_analysis: PortfolioOpenResearchAnalysis | None = None
    bounded_decision: PortfolioRecoveryBoundedDecision | None = None
    open_task_failure_count: int = Field(ge=0, le=5)
    constrained_task_failure_count: int = Field(ge=0, le=5)
    total_model_calls: int = Field(ge=0, le=12)
    typed_repair_count: int = Field(ge=0)
    host_feedback_codes: tuple[str, ...]
    decision_owner: Literal["PENDING", "AGENT", "DETERMINISTIC_FALLBACK"]
    board_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_board(self) -> Self:
        """Require exact historical recovery board identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: board_hash differs.
        """
        _identity(self, "board_hash")
        return self


class PortfolioRecoveryAgentReview(PortfolioLabContract):
    """Bind completed or fallback recovery decisions to exact review/board authority."""

    kind: Literal["PortfolioRecoveryAgentReview"] = "PortfolioRecoveryAgentReview"
    dossier_hash: str = Field(pattern=_HASH)
    board_hash: str = Field(pattern=_HASH)
    completion_status: Literal["COMPLETE", "DETERMINISTIC_FALLBACK"]
    decision_owner: Literal["AGENT", "DETERMINISTIC_FALLBACK"]
    open_analysis: PortfolioOpenResearchAnalysis | None = None
    bounded_decision: PortfolioRecoveryBoundedDecision
    model_calls: int = Field(ge=0, le=12)
    typed_repair_count: int = Field(ge=0)
    nonprogress_turn_count: int = Field(ge=0, le=10)
    agent_failure_count: int = Field(ge=0, le=5)
    fallback_reason: str | None = Field(default=None, max_length=600)
    board_binding_policy: Literal["REVIEW_BOUND"] = "REVIEW_BOUND"
    review_binding_hash: str = Field(pattern=_HASH)
    review_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_review(self) -> Self:
        """Require complete analysis or coherent deterministic fallback evidence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Fallback status/owner/reason disagree, complete review lacks open analysis
                or review_hash differs.
        """
        fallback = self.completion_status == "DETERMINISTIC_FALLBACK"
        if (
            fallback != (self.decision_owner == "DETERMINISTIC_FALLBACK")
            or fallback != (self.fallback_reason is not None)
            or (not fallback and self.open_analysis is None)
        ):
            raise ValueError("portfolio_strategy_lab.recovery_agent_review_invalid")
        _identity(self, "review_hash")
        return self


class HistoricalPortfolioRecoveryAgentReview(PortfolioLabContract):
    """Exact reader for the first recovery review schema."""

    kind: Literal["PortfolioRecoveryAgentReview"] = "PortfolioRecoveryAgentReview"
    dossier_hash: str = Field(pattern=_HASH)
    board_hash: str = Field(pattern=_HASH)
    completion_status: Literal["COMPLETE", "DETERMINISTIC_FALLBACK"]
    decision_owner: Literal["AGENT", "DETERMINISTIC_FALLBACK"]
    open_analysis: PortfolioOpenResearchAnalysis | None = None
    bounded_decision: PortfolioRecoveryBoundedDecision
    model_calls: int = Field(ge=0, le=12)
    typed_repair_count: int = Field(ge=0)
    nonprogress_turn_count: int = Field(ge=0, le=10)
    agent_failure_count: int = Field(ge=0, le=5)
    fallback_reason: str | None = Field(default=None, max_length=600)
    review_binding_hash: str = Field(pattern=_HASH)
    review_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_review(self) -> Self:
        """Require exact historical recovery review identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: review_hash differs.
        """
        _identity(self, "review_hash")
        return self


class PortfolioConditionalPolicyCandidate(PortfolioLabContract):
    """Identify a selected conditional policy and its evidence classification/owner."""

    source: PortfolioResearchAlphaSource
    candidate_id: str = Field(min_length=1)
    semantic_handle: str = Field(min_length=1, max_length=120)
    policy: SectorDeviationPenaltyPolicy
    classification: PortfolioCrossFoldEvidenceClassification
    selected_trial_evidence_hash: str = Field(pattern=_HASH)
    selection_owner: Literal["AGENT", "DETERMINISTIC_FALLBACK"]


class PortfolioConditionalPolicyCandidateSet(PortfolioLabContract):
    """Bind unique source/semantic candidates without claiming independent validation."""

    kind: Literal["PortfolioConditionalPolicyCandidateSet"] = (
        "PortfolioConditionalPolicyCandidateSet"
    )
    recovery_mandate_hash: str = Field(pattern=_HASH)
    decision_dossier_hash: str = Field(pattern=_HASH)
    agent_review_hash: str = Field(pattern=_HASH)
    candidates: tuple[PortfolioConditionalPolicyCandidate, ...] = Field(min_length=1, max_length=2)
    selection_owner: Literal["AGENT", "DETERMINISTIC_FALLBACK"]
    evidence_status: Literal["CONDITIONAL_RESEARCH_CANDIDATES"] = "CONDITIONAL_RESEARCH_CANDIDATES"
    independent_validation_status: Literal["NOT_RUN"] = "NOT_RUN"
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    candidate_set_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_candidate_set(self) -> Self:
        """Require unique source/semantic candidates and uniform selection ownership.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Source/handle repeats, child ownership differs or candidate_set_hash
                differs.
        """
        if (
            len({value.source for value in self.candidates}) != len(self.candidates)
            or len({value.semantic_handle for value in self.candidates}) != len(self.candidates)
            or any(value.selection_owner != self.selection_owner for value in self.candidates)
        ):
            raise ValueError("portfolio_strategy_lab.conditional_candidate_set_invalid")
        _identity(self, "candidate_set_hash")
        return self


class PortfolioStrategyRecoveryPublicationBundle(PortfolioLabContract):
    """Bind recovery artifacts, decision ownership, conditional status and explicit limits."""

    kind: Literal["PortfolioStrategyRecoveryPublicationBundle"] = (
        "PortfolioStrategyRecoveryPublicationBundle"
    )
    recovery_mandate_hash: str = Field(pattern=_HASH)
    evidence_audit_hash: str = Field(pattern=_HASH)
    shared_config_dossier_hash: str = Field(pattern=_HASH)
    decision_dossier_hash: str = Field(pattern=_HASH)
    agent_review_hash: str = Field(pattern=_HASH)
    candidate_set_hash: str | None = Field(default=None, pattern=_HASH)
    status: Literal[
        "PORTFOLIO_CONDITIONAL_POLICY_CANDIDATES_FROZEN",
        "PORTFOLIO_RECOVERY_RESEARCH_REQUIRED",
    ]
    conditional_candidate_count: int = Field(ge=0, le=2)
    decision_owner: Literal["AGENT", "DETERMINISTIC_FALLBACK"]
    open_analysis: PortfolioOpenResearchAnalysis | None = None
    constrained_decision: PortfolioRecoveryBoundedDecision
    model_call_count: int = Field(ge=0, le=12)
    agent_failure_count: int = Field(ge=0, le=5)
    typed_repair_count: int = Field(ge=0)
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    limitations: tuple[str, ...] = Field(min_length=1)
    bundle_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_bundle(self) -> Self:
        """Require frozen conditional status to match positive count and candidate-set presence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Frozen status differs from count/set presence or bundle_hash differs.
        """
        frozen = self.status == "PORTFOLIO_CONDITIONAL_POLICY_CANDIDATES_FROZEN"
        if frozen != (self.conditional_candidate_count > 0) or frozen != (
            self.candidate_set_hash is not None
        ):
            raise ValueError("portfolio_strategy_lab.recovery_publication_status_invalid")
        _identity(self, "bundle_hash")
        return self

    def model_view(self) -> dict[str, object]:
        """Project recovery decisions, call/repair counts, conditional status and limits.

        Returns:
            Declared interpretation mapping without internal artifact identity fields or publication
            authority.
        """
        return {
            "kind": self.kind,
            "status": self.status,
            "conditional_candidate_count": self.conditional_candidate_count,
            "decision_owner": self.decision_owner,
            "open_analysis": (
                self.open_analysis.model_dump(mode="json")
                if self.open_analysis is not None
                else None
            ),
            "constrained_decision": self.constrained_decision.model_dump(mode="json"),
            "model_call_count": self.model_call_count,
            "agent_failure_count": self.agent_failure_count,
            "typed_repair_count": self.typed_repair_count,
            "policy_holdout_state": self.policy_holdout_state,
            "system_holdout_state": self.system_holdout_state,
            "limitations": self.limitations,
        }


class CurrentPortfolioStrategyRecoveryMarker(PortfolioLabContract):
    """Bind current recovery status and bundle lineage to an aware publication clock."""

    kind: Literal["CurrentPortfolioStrategyRecoveryMarker"] = (
        "CurrentPortfolioStrategyRecoveryMarker"
    )
    recovery_mandate_hash: str = Field(pattern=_HASH)
    decision_dossier_hash: str = Field(pattern=_HASH)
    bundle_hash: str = Field(pattern=_HASH)
    status: Literal[
        "PORTFOLIO_CONDITIONAL_POLICY_CANDIDATES_FROZEN",
        "PORTFOLIO_RECOVERY_RESEARCH_REQUIRED",
    ]
    published_at: datetime
    marker_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> Self:
        """Require an aware recovery publication clock and exact marker identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Clock is naive or marker_hash differs.
        """
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("portfolio_strategy_lab.recovery_publication_clock_invalid")
        _identity(self, "marker_hash")
        return self


__all__ = [
    "CurrentPortfolioStrategyRecoveryMarker",
    "HistoricalPortfolioRecoveryAgentReview",
    "HistoricalPortfolioRecoveryResearchBoard",
    "PortfolioConditionalPolicyCandidate",
    "PortfolioConditionalPolicyCandidateSet",
    "PortfolioOpenResearchAnalysis",
    "PortfolioRecoveryAgentReview",
    "PortfolioRecoveryBoardTask",
    "PortfolioRecoveryBoundedDecision",
    "PortfolioRecoveryCandidateEvidence",
    "PortfolioRecoveryDecisionDossier",
    "PortfolioRecoveryDecisionRoute",
    "PortfolioRecoveryResearchBoard",
    "PortfolioStrategyRecoveryPublicationBundle",
]
