"""Typed evidence for the bounded Portfolio Risk calibration diagnostic."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.investment.risk_research.contracts import RiskContract
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"


def _identity(value: RiskContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("portfolio_strategy_lab.risk_calibration_identity_invalid")


class PortfolioRiskCalibrationClassification(StrEnum):
    """Name the diagnostic conclusions about predicted versus realized portfolio variance.

    The values distinguish systematic covariance, broad variance and portfolio-specific
    underprediction signals from an undetected signal.
    """

    SYSTEMATIC_COVARIANCE_UNDERPREDICTION_SIGNAL = "SYSTEMATIC_COVARIANCE_UNDERPREDICTION_SIGNAL"
    BROAD_VARIANCE_UNDERPREDICTION_SIGNAL = "BROAD_VARIANCE_UNDERPREDICTION_SIGNAL"
    PORTFOLIO_SPECIFIC_UNDERPREDICTION_SIGNAL = "PORTFOLIO_SPECIFIC_UNDERPREDICTION_SIGNAL"
    RISK_UNDERPREDICTION_NOT_DETECTED = "RISK_UNDERPREDICTION_NOT_DETECTED"


class PortfolioRiskCalibrationAlphaSource(StrEnum):
    """Name the current Alpha or Sector-aware Alpha source of calibration evidence."""

    CURRENT_ALPHA = "CURRENT_ALPHA"
    SECTOR_AWARE_ALPHA = "SECTOR_AWARE_ALPHA"


class PortfolioRiskCalibrationFoldEvidence(RiskContract):
    """Record one fixed fold of portfolio and control variance-calibration evidence.

    The three folds contain 125, 126 and 125 formations. Ratios compare realized squared returns
    with predicted variance for the selected policy, equal weights and individual names. Half-fold,
    six block and active-day summaries retain the declared diagnostic decomposition.
    """

    fold_index: Literal[1, 2, 3]
    formation_start: date
    formation_end: date
    formation_count: int = Field(ge=125, le=126)
    selected_policy_realized_to_predicted_ratio: float = Field(gt=0.0)
    equal_weight_realized_to_predicted_ratio: float = Field(gt=0.0)
    single_name_realized_to_predicted_ratio: float = Field(gt=0.0)
    equal_weight_first_half_ratio: float = Field(gt=0.0)
    equal_weight_second_half_ratio: float = Field(gt=0.0)
    equal_weight_21_session_block_ratios: tuple[float, float, float, float, float, float]
    peak_21_session_block_index: int = Field(ge=0, le=5)
    peak_21_session_start: date
    peak_21_session_end: date
    peak_21_session_ratio: float = Field(gt=0.0)
    negative_active_day_equal_weight_ratio: float = Field(gt=0.0)
    nonnegative_active_day_equal_weight_ratio: float = Field(gt=0.0)
    positive_calibration_gap_share_on_negative_active_days: float = Field(ge=0.0, le=1.0)
    selected_policy_minus_equal_weight_ratio: float
    equal_weight_mean_predicted_variance: float = Field(gt=0.0)
    equal_weight_mean_realized_squared_return: float = Field(ge=0.0)
    single_name_mean_predicted_variance: float = Field(gt=0.0)
    single_name_mean_realized_squared_return: float = Field(ge=0.0)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_fold(self) -> Self:
        """Require the fixed fold count, ordered intervals and matching peak block ratio.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The count differs from the fold, either interval is reversed or the peak
                ratio differs from its indexed block.
        """
        expected = 126 if self.fold_index == 2 else 125
        block_index = self.peak_21_session_block_index
        if (
            self.formation_count != expected
            or self.formation_start > self.formation_end
            or self.peak_21_session_start > self.peak_21_session_end
            or self.peak_21_session_ratio != self.equal_weight_21_session_block_ratios[block_index]
        ):
            raise ValueError("portfolio_strategy_lab.risk_calibration_fold_invalid")
        return self


class PortfolioRiskCalibrationDiagnostic(RiskContract):
    """Seal a three-fold calibration diagnostic against prior attribution and source evidence.

    The diagnostic binds candidate, trial, mandate, Risk and return identities. It declares 376
    covariance reads and return sessions and zero optimization, Alpha fitting, covariance rebuilding
    and protected decision reads. These fields describe the record; they do not execute or refresh
    the evidence.
    """

    kind: Literal["PortfolioRiskCalibrationDiagnostic"] = "PortfolioRiskCalibrationDiagnostic"
    prior_attribution_marker_hash: str = Field(pattern=_HASH)
    prior_attribution_bundle_hash: str = Field(pattern=_HASH)
    prior_agent_review_hash: str = Field(pattern=_HASH)
    candidate_set_hash: str = Field(pattern=_HASH)
    selected_trial_evidence_hash: str = Field(pattern=_HASH)
    research_mandate_hash: str = Field(pattern=_HASH)
    risk_surface_hash: str = Field(pattern=_HASH)
    return_surface_hash: str = Field(pattern=_HASH)
    diagnostic_binding_hash: str = Field(pattern=_HASH)
    source: PortfolioRiskCalibrationAlphaSource
    semantic_candidate_handle: str = Field(min_length=1, max_length=120)
    classification: PortfolioRiskCalibrationClassification
    folds: tuple[
        PortfolioRiskCalibrationFoldEvidence,
        PortfolioRiskCalibrationFoldEvidence,
        PortfolioRiskCalibrationFoldEvidence,
    ]
    findings: tuple[str, ...] = Field(min_length=1)
    unavailable_diagnostics: tuple[str, ...] = Field(min_length=1)
    conclusion: str = Field(min_length=1, max_length=3000)
    recommended_next_mandate: str = Field(min_length=1, max_length=2200)
    covariance_matrix_read_count: Literal[376] = 376
    return_value_session_count: Literal[376] = 376
    optimizer_calls: Literal[0] = 0
    optuna_attempts: Literal[0] = 0
    alpha_fits: Literal[0] = 0
    covariance_rebuilds: Literal[0] = 0
    policy_holdout_decision_reads: Literal[0] = 0
    boundary_holding_end_session: date
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    diagnostic_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_diagnostic(self) -> Self:
        """Require the ordered three-fold axis and exact diagnostic identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The fold axis is not (1, 2, 3) or diagnostic_hash differs from the complete
                payload.
        """
        if tuple(value.fold_index for value in self.folds) != (1, 2, 3):
            raise ValueError("portfolio_strategy_lab.risk_calibration_fold_axis_invalid")
        _identity(self, "diagnostic_hash")
        return self


class PortfolioRiskCalibrationPublicationBundle(RiskContract):
    """Seal the diagnostic conclusion and next mandate against its prior attribution lineage.

    Publication retains the classification, one conditional candidate and declared protected-data
    states. The bundle hash binds the complete declared payload.
    """

    kind: Literal["PortfolioRiskCalibrationPublicationBundle"] = (
        "PortfolioRiskCalibrationPublicationBundle"
    )
    prior_attribution_marker_hash: str = Field(pattern=_HASH)
    prior_attribution_bundle_hash: str = Field(pattern=_HASH)
    diagnostic_hash: str = Field(pattern=_HASH)
    status: Literal["PORTFOLIO_RISK_CALIBRATION_DIAGNOSTIC_COMPLETE"] = (
        "PORTFOLIO_RISK_CALIBRATION_DIAGNOSTIC_COMPLETE"
    )
    classification: PortfolioRiskCalibrationClassification
    conditional_candidate_count: Literal[1] = 1
    conclusion: str = Field(min_length=1, max_length=3000)
    recommended_next_mandate: str = Field(min_length=1, max_length=2200)
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    bundle_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_bundle(self) -> Self:
        """Require the exact canonical calibration publication identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: bundle_hash differs from the complete payload excluding that hash.
        """
        _identity(self, "bundle_hash")
        return self

    def model_view(self) -> dict[str, object]:
        """Return the compact calibration decision view without lineage identities.

        Returns:
            Kind, status, classification, candidate count, conclusion, next mandate and declared
            protected-data states.
        """
        return {
            "kind": self.kind,
            "status": self.status,
            "classification": self.classification,
            "conditional_candidate_count": self.conditional_candidate_count,
            "conclusion": self.conclusion,
            "recommended_next_mandate": self.recommended_next_mandate,
            "policy_holdout_state": self.policy_holdout_state,
            "system_holdout_state": self.system_holdout_state,
        }


class CurrentPortfolioRiskCalibrationMarker(RiskContract):
    """Bind a completed calibration publication to its exact diagnostic and bundle.

    The marker retains the prior attribution identity and a timezone-aware publication clock.
    """

    kind: Literal["CurrentPortfolioRiskCalibrationMarker"] = "CurrentPortfolioRiskCalibrationMarker"
    prior_attribution_marker_hash: str = Field(pattern=_HASH)
    diagnostic_hash: str = Field(pattern=_HASH)
    bundle_hash: str = Field(pattern=_HASH)
    status: Literal["PORTFOLIO_RISK_CALIBRATION_DIAGNOSTIC_COMPLETE"] = (
        "PORTFOLIO_RISK_CALIBRATION_DIAGNOSTIC_COMPLETE"
    )
    published_at: datetime
    marker_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> Self:
        """Require a timezone-aware publication clock and exact marker identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: published_at has no timezone offset or marker_hash differs from the complete
                payload.
        """
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("portfolio_strategy_lab.risk_calibration_publication_clock_invalid")
        _identity(self, "marker_hash")
        return self


__all__ = [
    "CurrentPortfolioRiskCalibrationMarker",
    "PortfolioRiskCalibrationAlphaSource",
    "PortfolioRiskCalibrationClassification",
    "PortfolioRiskCalibrationDiagnostic",
    "PortfolioRiskCalibrationFoldEvidence",
    "PortfolioRiskCalibrationPublicationBundle",
]
