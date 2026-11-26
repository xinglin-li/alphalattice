"""Typed classification and recovery evidence for Portfolio Strategy research."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import (
    PortfolioLabContract,
    PortfolioTrialMetrics,
    SectorDeviationPenaltyPolicy,
)
from .contracts import PortfolioResearchAlphaSource

_HASH = r"^[0-9a-f]{64}$"


def _identity(value: PortfolioLabContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("portfolio_strategy_lab.recovery_identity_invalid")


class PortfolioCrossFoldEvidenceClassification(StrEnum):
    """Classify cross-fold evidence direction, support or operational failure.

    Classify robust, mixed, undetectable, negative or operationally invalid cross-fold evidence.
    """

    ROBUST_POSITIVE_CV = "ROBUST_POSITIVE_CV"
    MIXED_CV_EVIDENCE = "MIXED_CV_EVIDENCE"
    NO_DETECTABLE_POLICY_EDGE = "NO_DETECTABLE_POLICY_EDGE"
    NEGATIVE_CV_EVIDENCE = "NEGATIVE_CV_EVIDENCE"
    OPERATIONALLY_INVALID = "OPERATIONALLY_INVALID"


class PortfolioCrossFoldGateAudit(PortfolioLabContract):
    """Retain cross-fold direction, drawdown, region and operational gates.

    Retain fold direction, uncertainty, drawdown, connected-region and operational gate findings.
    """

    source: PortfolioResearchAlphaSource
    classification: PortfolioCrossFoldEvidenceClassification
    active_net_log_wealth: tuple[float, float, float]
    active_block_standard_errors: tuple[float, float, float]
    positive_fold_count: int = Field(ge=0, le=3)
    materially_positive_fold_count: int = Field(ge=0, le=3)
    materially_negative_fold_count: int = Field(ge=0, le=3)
    median_active_net_log_wealth: float
    direction_gate_passed: bool
    drawdown_guard_passed_by_fold: tuple[bool, bool, bool]
    connected_region_passed: bool
    all_operational_gates_passed: bool
    rejection_reasons: tuple[str, ...]


class PortfolioDevelopmentPlausibilitySummary(PortfolioLabContract):
    """Retain candidate distribution and conservative one-standard-error region evidence."""

    source: PortfolioResearchAlphaSource
    evaluated_candidate_count: int = Field(ge=1)
    benchmark_cumulative_return: float
    candidate_return_p25: float
    candidate_return_median: float
    candidate_return_p75: float
    candidate_return_p90: float
    best_candidate_return: float
    best_active_log_wealth: float
    best_gross_wealth_to_benchmark_ratio: float = Field(gt=0.0)
    best_21_session_active_contribution_fraction: float
    one_standard_error_region_count: int = Field(ge=1)
    conservative_region_representative_return: float
    conservative_region_representative_top_k: int = Field(ge=1)
    conservative_region_representative_turnover: float = Field(ge=0.0)
    conservative_region_representative_hhi: float = Field(ge=0.0, le=1.0)


class PortfolioEvidenceClassificationAudit(PortfolioLabContract):
    """Bind published-evidence classification without numerical recomputation.

    Bind published-evidence classification without solver, model or input-surface recomputation.
    """

    kind: Literal["PortfolioEvidenceClassificationAudit"] = "PortfolioEvidenceClassificationAudit"
    mandate_hash: str = Field(pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    classification_policy_hash: str = Field(pattern=_HASH)
    source_audits: tuple[PortfolioCrossFoldGateAudit, PortfolioCrossFoldGateAudit]
    development_plausibility: tuple[
        PortfolioDevelopmentPlausibilitySummary,
        PortfolioDevelopmentPlausibilitySummary,
    ]
    solver_calls: Literal[0] = 0
    optuna_attempts: Literal[0] = 0
    alpha_risk_data_surface_reads: Literal[0] = 0
    published_evidence_reads: int = Field(ge=1)
    agent_calls: Literal[0] = 0
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    audit_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_audit(self) -> Self:
        """Require exact ordered Alpha source audits/plausibility and canonical audit identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Declared current/sector-aware source order or audit_hash differs.
        """
        expected_sources = (
            PortfolioResearchAlphaSource.CURRENT_ALPHA,
            PortfolioResearchAlphaSource.SECTOR_AWARE_ALPHA,
        )
        if (
            tuple(value.source for value in self.source_audits) != expected_sources
            or tuple(value.source for value in self.development_plausibility) != expected_sources
        ):
            raise ValueError("portfolio_strategy_lab.evidence_audit_source_invalid")
        _identity(self, "audit_hash")
        return self


class PortfolioStrategyRecoveryMandate(PortfolioLabContract):
    """Bind authorized recovery and exact source/numerical authority.

    Bind authorized shared-config recovery, source pair and exact numerical execution authority.
    """

    kind: Literal["PortfolioStrategyRecoveryMandate"] = "PortfolioStrategyRecoveryMandate"
    prior_mandate_hash: str = Field(pattern=_HASH)
    prior_dossier_hash: str = Field(pattern=_HASH)
    evidence_audit_hash: str = Field(pattern=_HASH)
    workspace_mandate_hash: str = Field(pattern=_HASH)
    sources: tuple[PortfolioResearchAlphaSource, PortfolioResearchAlphaSource] = (
        PortfolioResearchAlphaSource.CURRENT_ALPHA,
        PortfolioResearchAlphaSource.SECTOR_AWARE_ALPHA,
    )
    shared_config_trials_per_source: Literal[24] = 24
    fold_count: Literal[3] = 3
    policy_family: Literal["SECTOR_DEVIATION_PENALTY"] = "SECTOR_DEVIATION_PENALTY"
    selection_rule: Literal["CROSS_FOLD_ONE_STANDARD_ERROR_SIMPLE_REGION"] = (
        "CROSS_FOLD_ONE_STANDARD_ERROR_SIMPLE_REGION"
    )
    typed_authority: Literal["USER_AUTHORIZED_PORTFOLIO_EVIDENCE_RECOVERY"] = (
        "USER_AUTHORIZED_PORTFOLIO_EVIDENCE_RECOVERY"
    )
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    numerical_execution_binding_hash: str = Field(pattern=_HASH)
    numerical_environment_hash: str = Field(pattern=_HASH)
    mandate_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_mandate(self) -> Self:
        """Require the declared current/sector-aware Alpha source pair and exact mandate.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Source pair or mandate_hash differs.
        """
        if self.sources != (
            PortfolioResearchAlphaSource.CURRENT_ALPHA,
            PortfolioResearchAlphaSource.SECTOR_AWARE_ALPHA,
        ):
            raise ValueError("portfolio_strategy_lab.recovery_source_invalid")
        _identity(self, "mandate_hash")
        return self


class PortfolioSharedConfigFoldEvidence(PortfolioLabContract):
    """Retain shared-config fold wealth, uncertainty and drawdown facts.

    Retain one shared-config validation fold's active wealth, uncertainty and drawdown evidence.
    """

    fold_index: Literal[1, 2, 3]
    validation_metrics: PortfolioTrialMetrics
    validation_net_log_returns_10bps: tuple[float, ...] = Field(min_length=125, max_length=126)
    benchmark_log_returns: tuple[float, ...] = Field(min_length=125, max_length=126)
    active_net_log_wealth: float
    active_block_standard_error: float = Field(ge=0.0)
    sortino: float
    maximum_drawdown: float = Field(ge=0.0, le=1.0)
    benchmark_maximum_drawdown: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_fold(self) -> Self:
        """Require strategy and benchmark return lengths for the declared fold.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Return lengths differ from the fold's 125/126-session geometry.
        """
        expected = 126 if self.fold_index == 2 else 125
        if (
            len(self.validation_net_log_returns_10bps) != expected
            or len(self.benchmark_log_returns) != expected
        ):
            raise ValueError("portfolio_strategy_lab.recovery_fold_axis_invalid")
        return self


class PortfolioSharedConfigTrialEvidence(PortfolioLabContract):
    """Bind completed shared-config three-fold evidence or explicit failed-trial disposition."""

    kind: Literal["PortfolioSharedConfigTrialEvidence"] = "PortfolioSharedConfigTrialEvidence"
    mandate_hash: str = Field(pattern=_HASH)
    source: PortfolioResearchAlphaSource
    candidate_id: str = Field(min_length=1)
    trial_number: int = Field(ge=0, lt=24)
    policy: SectorDeviationPenaltyPolicy
    search_parameters: dict[str, float | int]
    status: Literal["COMPLETED", "FAILED"]
    failure_code: str | None = None
    fold_evidence: (
        tuple[
            PortfolioSharedConfigFoldEvidence,
            PortfolioSharedConfigFoldEvidence,
            PortfolioSharedConfigFoldEvidence,
        ]
        | tuple[()]
    ) = ()
    median_active_net_log_wealth: float | None = None
    median_annualized_volatility: float | None = None
    evidence_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_evidence(self) -> Self:
        """Require coherent completed metrics/folds or failed-trial evidence.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Completion differs from failure absence, three folds or median metrics,
                completed fold order differs, or evidence_hash differs.
        """
        completed = self.status == "COMPLETED"
        if (
            completed != (self.failure_code is None)
            or completed != (len(self.fold_evidence) == 3)
            or completed != (self.median_active_net_log_wealth is not None)
            or completed != (self.median_annualized_volatility is not None)
        ):
            raise ValueError("portfolio_strategy_lab.recovery_trial_invalid")
        if completed and tuple(value.fold_index for value in self.fold_evidence) != (1, 2, 3):
            raise ValueError("portfolio_strategy_lab.recovery_trial_fold_invalid")
        _identity(self, "evidence_hash")
        return self


class PortfolioSharedConfigCandidate(PortfolioLabContract):
    """Retain a cross-fold supported conservative policy candidate and neighborhood evidence."""

    source: PortfolioResearchAlphaSource
    candidate_id: str = Field(min_length=1)
    policy: SectorDeviationPenaltyPolicy
    selected_trial_evidence_hash: str = Field(pattern=_HASH)
    supporting_trial_evidence_hashes: tuple[str, ...] = Field(min_length=3)
    classification: PortfolioCrossFoldEvidenceClassification
    positive_fold_count: int = Field(ge=2, le=3)
    median_active_net_log_wealth: float = Field(gt=0.0)
    worst_fold_active_net_log_wealth: float
    worst_fold_drawdown_excess: float
    median_sortino: float
    median_turnover: float = Field(ge=0.0)
    median_hhi: float = Field(ge=0.0, le=1.0)
    neighbor_support_count: int = Field(ge=2)
    one_standard_error_supported: Literal[True] = True
    semantic_region_handle: str = Field(min_length=1, max_length=120)


class PortfolioSharedConfigEvidenceDossier(PortfolioLabContract):
    """Bind two distinct source ledgers, classified candidates, observations and limitations."""

    kind: Literal["PortfolioSharedConfigEvidenceDossier"] = "PortfolioSharedConfigEvidenceDossier"
    mandate_hash: str = Field(pattern=_HASH)
    evidence_audit_hash: str = Field(pattern=_HASH)
    ledger_hashes: tuple[str, str]
    candidates: tuple[PortfolioSharedConfigCandidate, ...] = Field(max_length=2)
    classification_counts: dict[str, dict[str, int]]
    observations: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    dossier_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_dossier(self) -> Self:
        """Require two distinct shared-config source ledgers and exact dossier identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Ledger identities are not distinct or dossier_hash differs.
        """
        if len(set(self.ledger_hashes)) != 2:
            raise ValueError("portfolio_strategy_lab.recovery_dossier_invalid")
        _identity(self, "dossier_hash")
        return self


__all__ = [
    "PortfolioCrossFoldEvidenceClassification",
    "PortfolioCrossFoldGateAudit",
    "PortfolioDevelopmentPlausibilitySummary",
    "PortfolioEvidenceClassificationAudit",
    "PortfolioSharedConfigCandidate",
    "PortfolioSharedConfigEvidenceDossier",
    "PortfolioSharedConfigFoldEvidence",
    "PortfolioSharedConfigTrialEvidence",
    "PortfolioStrategyRecoveryMandate",
]
