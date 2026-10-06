"""Typed Fold-3 attribution evidence and bounded research decisions."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import PortfolioLabContract
from .contracts import PortfolioResearchAlphaSource

_HASH = r"^[0-9a-f]{64}$"


def _identity(value: PortfolioLabContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("portfolio_strategy_lab.attribution_identity_invalid")


class PortfolioAttributionHypothesisStatus(StrEnum):
    """Declare supported, weakened, falsified or unresolved attribution evidence."""

    SUPPORTED = "SUPPORTED"
    WEAKENED = "WEAKENED"
    FALSIFIED = "FALSIFIED"
    UNRESOLVED = "UNRESOLVED"


class PortfolioAttributionDecisionRoute(StrEnum):
    """Declare bounded follow-up studies, control repair or retained conditional candidacy."""

    REQUEST_REGIME_SENSITIVITY_STUDY = "REQUEST_REGIME_SENSITIVITY_STUDY"
    REQUEST_ALPHA_BUDGET_DIAGNOSTIC = "REQUEST_ALPHA_BUDGET_DIAGNOSTIC"
    REQUEST_RISK_CALIBRATION_DIAGNOSTIC = "REQUEST_RISK_CALIBRATION_DIAGNOSTIC"
    REQUEST_PORTFOLIO_CONTROL_REPAIR = "REQUEST_PORTFOLIO_CONTROL_REPAIR"
    REQUEST_ADDITIONAL_NON_HOLDOUT_EVIDENCE = "REQUEST_ADDITIONAL_NON_HOLDOUT_EVIDENCE"
    MAINTAIN_CONDITIONAL_CANDIDATE_WITHOUT_VALIDATION = (
        "MAINTAIN_CONDITIONAL_CANDIDATE_WITHOUT_VALIDATION"
    )


class PortfolioFoldAttributionSnapshot(PortfolioLabContract):
    """Retain fold support, return concentration, turnover and operational/Risk diagnostics."""

    fold_index: Literal[1, 2, 3]
    formation_start: date
    formation_end: date
    formation_count: int = Field(ge=125, le=126)
    strategy_net_log_wealth_10bps: float
    benchmark_log_wealth: float
    active_net_log_wealth: float
    negative_active_day_fraction: float = Field(ge=0.0, le=1.0)
    negative_chronological_block_count: int = Field(ge=0, le=5)
    worst_5_session_start: date
    worst_5_session_end: date
    worst_5_session_active_log_wealth: float
    worst_21_session_start: date
    worst_21_session_end: date
    worst_21_session_active_log_wealth: float
    worst_21_negative_active_mass_fraction: float = Field(ge=0.0, le=1.0)
    mean_one_way_turnover: float = Field(ge=0.0)
    mean_hhi: float = Field(ge=0.0, le=1.0)
    mean_holding_count: float = Field(ge=0.0)
    solver_failure_count: int = Field(ge=0)
    missing_execution_count: int = Field(ge=0)
    realized_to_predicted_variance_ratio: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_snapshot(self) -> Self:
        """Require declared fold counts and ordered diagnostic date ranges.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Fold count or formation/five-session/21-session range order differs.
        """
        expected = 126 if self.fold_index == 2 else 125
        if (
            self.formation_count != expected
            or self.formation_start > self.formation_end
            or self.worst_5_session_start > self.worst_5_session_end
            or self.worst_21_session_start > self.worst_21_session_end
        ):
            raise ValueError("portfolio_strategy_lab.attribution_fold_invalid")
        return self


class PortfolioFold3AttributionDiagnostic(PortfolioLabContract):
    """Bind ordered fold diagnostics from frozen evidence without numerical recomputation."""

    kind: Literal["PortfolioFold3AttributionDiagnostic"] = "PortfolioFold3AttributionDiagnostic"
    recovery_bundle_hash: str = Field(pattern=_HASH)
    candidate_set_hash: str = Field(pattern=_HASH)
    selected_trial_evidence_hash: str = Field(pattern=_HASH)
    diagnostic_binding_hash: str = Field(pattern=_HASH)
    source: PortfolioResearchAlphaSource
    semantic_candidate_handle: str = Field(min_length=1, max_length=120)
    folds: tuple[
        PortfolioFoldAttributionSnapshot,
        PortfolioFoldAttributionSnapshot,
        PortfolioFoldAttributionSnapshot,
    ]
    evidence_status: Literal["COMPLETE_FROM_FROZEN_EVIDENCE", "INSUFFICIENT_EXISTING_EVIDENCE"]
    available_diagnostics: tuple[str, ...] = Field(min_length=1)
    unavailable_diagnostics: tuple[str, ...]
    findings: tuple[str, ...] = Field(min_length=1)
    optimizer_or_model_recomputations: Literal[0] = 0
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    diagnostic_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_diagnostic(self) -> Self:
        """Require the ordered three-fold axis and exact diagnostic identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Fold order differs from 1/2/3 or diagnostic_hash differs.
        """
        if tuple(value.fold_index for value in self.folds) != (1, 2, 3):
            raise ValueError("portfolio_strategy_lab.attribution_fold_axis_invalid")
        _identity(self, "diagnostic_hash")
        return self


class PortfolioAttributionHypothesisUpdate(PortfolioLabContract):
    """Record one attribution hypothesis status, supporting evidence and remaining uncertainty."""

    handle: Literal["H1", "H2", "H3"]
    status: PortfolioAttributionHypothesisStatus
    evidence: str = Field(min_length=1, max_length=1600)
    remaining_uncertainty: str = Field(min_length=1, max_length=1200)


class PortfolioAttributionEvidenceDelta(PortfolioLabContract):
    """Bind the ordered attribution hypothesis updates and next discriminating question."""

    kind: Literal["PortfolioAttributionEvidenceDelta"] = "PortfolioAttributionEvidenceDelta"
    prior_recovery_review_hash: str = Field(pattern=_HASH)
    diagnostic_hash: str = Field(pattern=_HASH)
    hypothesis_updates: tuple[
        PortfolioAttributionHypothesisUpdate,
        PortfolioAttributionHypothesisUpdate,
        PortfolioAttributionHypothesisUpdate,
    ]
    discriminating_findings: tuple[str, ...] = Field(min_length=1)
    next_evidence_question: str = Field(min_length=1, max_length=1800)
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    delta_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_delta(self) -> Self:
        """Require the complete ordered hypothesis axis and exact evidence delta identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Hypothesis handles differ from H1/H2/H3 or delta_hash differs.
        """
        if tuple(value.handle for value in self.hypothesis_updates) != ("H1", "H2", "H3"):
            raise ValueError("portfolio_strategy_lab.attribution_hypothesis_axis_invalid")
        _identity(self, "delta_hash")
        return self


class PortfolioAttributionDecisionSubmission(PortfolioLabContract):
    """Declare primary attribution, bounded route and all three hypothesis updates."""

    primary_hypothesis: Literal["H1", "H2", "H3", "UNRESOLVED"]
    route: PortfolioAttributionDecisionRoute
    hypothesis_updates: tuple[
        PortfolioAttributionHypothesisUpdate,
        PortfolioAttributionHypothesisUpdate,
        PortfolioAttributionHypothesisUpdate,
    ]
    attribution_conclusion: str = Field(min_length=1, max_length=2600)
    recommended_next_mandate: str = Field(min_length=1, max_length=2200)
    evidence_gaps_acknowledged: Literal[True] = True

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_submission(self) -> Self:
        """Require the complete ordered attribution hypothesis axis.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Hypothesis handles differ from H1/H2/H3.
        """
        if tuple(value.handle for value in self.hypothesis_updates) != ("H1", "H2", "H3"):
            raise ValueError("portfolio_strategy_lab.attribution_hypothesis_axis_invalid")
        return self


class PortfolioAttributionAgentReview(PortfolioLabContract):
    """Bind attribution decision ownership, call/repair counts and explicit fallback reason."""

    kind: Literal["PortfolioAttributionAgentReview"] = "PortfolioAttributionAgentReview"
    evidence_delta_hash: str = Field(pattern=_HASH)
    board_hash: str = Field(pattern=_HASH)
    decision: PortfolioAttributionDecisionSubmission
    decision_owner: Literal["AGENT", "DETERMINISTIC_FALLBACK"]
    model_call_count: int = Field(ge=0, le=6)
    failure_count: int = Field(ge=0, le=5)
    typed_repair_count: int = Field(ge=0)
    fallback_reason: str | None = Field(default=None, max_length=800)
    review_binding_hash: str = Field(pattern=_HASH)
    review_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_review(self) -> Self:
        """Require fallback reason exactly for deterministic fallback and exact review identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Fallback ownership differs from reason presence or review_hash differs.
        """
        fallback = self.decision_owner == "DETERMINISTIC_FALLBACK"
        if fallback != (self.fallback_reason is not None):
            raise ValueError("portfolio_strategy_lab.attribution_review_invalid")
        _identity(self, "review_hash")
        return self


class PortfolioFold3AttributionPublicationBundle(PortfolioLabContract):
    """Bind attribution artifacts, route, evidence status and explicit claim limitations."""

    kind: Literal["PortfolioFold3AttributionPublicationBundle"] = (
        "PortfolioFold3AttributionPublicationBundle"
    )
    recovery_bundle_hash: str = Field(pattern=_HASH)
    candidate_set_hash: str = Field(pattern=_HASH)
    diagnostic_hash: str = Field(pattern=_HASH)
    evidence_delta_hash: str = Field(pattern=_HASH)
    agent_review_hash: str = Field(pattern=_HASH)
    status: Literal[
        "PORTFOLIO_FOLD3_ATTRIBUTION_EVIDENCE_UPDATED",
        "PORTFOLIO_FOLD3_ATTRIBUTION_INSUFFICIENT",
    ]
    conditional_candidate_count: Literal[1] = 1
    decision_owner: Literal["AGENT", "DETERMINISTIC_FALLBACK"]
    primary_hypothesis: Literal["H1", "H2", "H3", "UNRESOLVED"]
    decision_route: PortfolioAttributionDecisionRoute
    attribution_conclusion: str = Field(min_length=1, max_length=2600)
    recommended_next_mandate: str = Field(min_length=1, max_length=2200)
    model_call_count: int = Field(ge=0, le=6)
    failure_count: int = Field(ge=0, le=5)
    typed_repair_count: int = Field(ge=0)
    limitations: tuple[str, ...] = Field(min_length=1)
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    bundle_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_bundle(self) -> Self:
        """Require exact attribution publication bundle identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: bundle_hash differs.
        """
        _identity(self, "bundle_hash")
        return self

    def model_view(self) -> dict[str, object]:
        """Project attribution conclusions, route, call counts and limits for model interpretation.

        Returns:
            Declared interpretation view without artifact identity internals or publication
            authority.
        """
        return {
            "kind": self.kind,
            "status": self.status,
            "conditional_candidate_count": self.conditional_candidate_count,
            "decision_owner": self.decision_owner,
            "primary_hypothesis": self.primary_hypothesis,
            "decision_route": self.decision_route,
            "attribution_conclusion": self.attribution_conclusion,
            "recommended_next_mandate": self.recommended_next_mandate,
            "model_call_count": self.model_call_count,
            "failure_count": self.failure_count,
            "typed_repair_count": self.typed_repair_count,
            "limitations": self.limitations,
            "policy_holdout_state": self.policy_holdout_state,
            "system_holdout_state": self.system_holdout_state,
        }


class CurrentPortfolioFold3AttributionMarker(PortfolioLabContract):
    """Bind the current attribution bundle and status to an aware publication clock."""

    kind: Literal["CurrentPortfolioFold3AttributionMarker"] = (
        "CurrentPortfolioFold3AttributionMarker"
    )
    diagnostic_hash: str = Field(pattern=_HASH)
    evidence_delta_hash: str = Field(pattern=_HASH)
    bundle_hash: str = Field(pattern=_HASH)
    status: Literal[
        "PORTFOLIO_FOLD3_ATTRIBUTION_EVIDENCE_UPDATED",
        "PORTFOLIO_FOLD3_ATTRIBUTION_INSUFFICIENT",
    ]
    published_at: datetime
    marker_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> Self:
        """Require an aware attribution publication clock and exact marker identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Clock is naive or marker_hash differs.
        """
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("portfolio_strategy_lab.attribution_publication_clock_invalid")
        _identity(self, "marker_hash")
        return self


__all__ = [
    "CurrentPortfolioFold3AttributionMarker",
    "PortfolioAttributionAgentReview",
    "PortfolioAttributionDecisionRoute",
    "PortfolioAttributionDecisionSubmission",
    "PortfolioAttributionEvidenceDelta",
    "PortfolioAttributionHypothesisStatus",
    "PortfolioAttributionHypothesisUpdate",
    "PortfolioFold3AttributionDiagnostic",
    "PortfolioFold3AttributionPublicationBundle",
    "PortfolioFoldAttributionSnapshot",
]
