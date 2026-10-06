"""Typed metric policy and evidence owned by Alpha scientific evaluation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def seal_evaluation_contract[ContractT: _Contract](
    model: type[ContractT],
    values: dict[str, Any],
    identity_field: str,
) -> ContractT:
    """Seal an evaluation-owned contract without crossing owner boundaries."""
    return seal_model(model, values, field=identity_field)


class AlphaMetricAvailability(StrEnum):
    """Name available, constant-score or insufficient-data metric results."""

    AVAILABLE = "AVAILABLE"
    CONSTANT_SCORE = "CONSTANT_SCORE"
    INSUFFICIENT_CROSS_SECTION = "INSUFFICIENT_CROSS_SECTION"
    INSUFFICIENT_FOLDS = "INSUFFICIENT_FOLDS"


class AlphaMetricPolicy(_Contract):
    """Seal the fixed nine-metric inventory and common causal comparison policy.

    The policy fixes 100-row cross sections, ten deciles, zero-forecast R-squared baseline and
    one-way formation turnover. Constant scores make rank/spread unavailable without declaring
    failure.
    """

    kind: Literal["AlphaMetricPolicy"] = "AlphaMetricPolicy"
    metric_ids: tuple[str, ...]
    minimum_cross_section: Literal[100] = 100
    decile_count: Literal[10] = 10
    r_squared_baseline: Literal["ZERO_FORECAST"] = "ZERO_FORECAST"
    comparison_surface: Literal["COMMON_FACTOR_AND_CAUSAL_OUTCOME_COMPLETE"] = (
        "COMMON_FACTOR_AND_CAUSAL_OUTCOME_COMPLETE"
    )
    constant_score_policy: Literal["RANK_AND_SPREAD_UNAVAILABLE_NOT_FAILURE"] = (
        "RANK_AND_SPREAD_UNAVAILABLE_NOT_FAILURE"
    )
    turnover_policy: Literal["FORMATION_DECILE_ONE_WAY"] = "FORMATION_DECILE_ONE_WAY"
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaMetricPolicy:
        """Require the exact admitted metric inventory and canonical metric policy.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Metric identifiers differ from the fixed inventory or policy_hash is
                inconsistent.
        """
        expected = (
            "MAE",
            "MSE",
            "ZERO_RELATIVE_OOS_R2",
            "RANK_IC",
            "ICIR",
            "GROSS_DECILE_SPREAD",
            "FOLD_COVERAGE",
            "FOLD_STABILITY",
            "FORMATION_DECILE_TURNOVER",
        )
        if self.metric_ids != expected:
            raise ValueError("Alpha metric inventory differs from the admitted policy")
        if self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise ValueError("Alpha metric policy hash is invalid")
        return self


class AlphaMetricValue(_Contract):
    """Pair one finite available metric with its explicit availability state."""

    availability: AlphaMetricAvailability
    value: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_value(self) -> AlphaMetricValue:
        """Require numerical value presence exactly when the metric is available.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Availability and numerical value presence disagree.
        """
        if (self.availability is AlphaMetricAvailability.AVAILABLE) != (self.value is not None):
            raise ValueError("Alpha metric value and availability disagree")
        return self


class AlphaFoldMetrics(_Contract):
    """Seal one fold comparison/scored counts, coverage and availability-aware metrics."""

    fold_index: int = Field(ge=0)
    comparison_row_count: int = Field(ge=0)
    scored_comparison_row_count: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    mae: AlphaMetricValue
    mse: AlphaMetricValue
    zero_relative_oos_r2: AlphaMetricValue
    rank_ic: AlphaMetricValue
    gross_decile_spread: AlphaMetricValue
    formation_decile_turnover: AlphaMetricValue
    metrics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaFoldMetrics:
        """Require bounded scored counts, reconciled coverage and exact fold metrics identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Scored rows exceed comparison rows, coverage differs by more than 1e-15 or
                metrics_hash is inconsistent.
        """
        if self.scored_comparison_row_count > self.comparison_row_count:
            raise ValueError("Alpha fold scored rows exceed comparison surface")
        expected_coverage = (
            self.scored_comparison_row_count / self.comparison_row_count
            if self.comparison_row_count
            else 0.0
        )
        if abs(self.coverage - expected_coverage) > 1e-15:
            raise ValueError("Alpha fold coverage is inconsistent")
        if self.metrics_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"metrics_hash"})
        ):
            raise ValueError("Alpha fold metrics hash is invalid")
        return self


class AlphaCandidateMetrics(_Contract):
    """Seal contiguous fold metrics and candidate-level common-surface evidence."""

    candidate_id: str
    fold_metrics: tuple[AlphaFoldMetrics, ...] = Field(min_length=1)
    common_surface_row_count: int = Field(ge=1)
    scored_row_count: int = Field(ge=0)
    mae: AlphaMetricValue
    mse: AlphaMetricValue
    zero_relative_oos_r2: AlphaMetricValue
    rank_ic_mean: AlphaMetricValue
    icir: AlphaMetricValue
    gross_decile_spread_mean: AlphaMetricValue
    formation_decile_turnover_mean: AlphaMetricValue
    fold_coverage_mean: float = Field(ge=0, le=1, allow_inf_nan=False)
    fold_mse_std: AlphaMetricValue
    metrics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaCandidateMetrics:
        """Require a contiguous fold axis, bounded scored rows and exact candidate metrics identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Fold indices are not zero-based contiguous, scored count exceeds common
                surface or metrics_hash is inconsistent.
        """
        if tuple(value.fold_index for value in self.fold_metrics) != tuple(
            range(len(self.fold_metrics))
        ):
            raise ValueError("Alpha candidate fold metrics are not contiguous")
        if self.scored_row_count > self.common_surface_row_count:
            raise ValueError("Alpha candidate scored rows exceed common surface")
        if self.metrics_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"metrics_hash"})
        ):
            raise ValueError("Alpha candidate metrics hash is invalid")
        return self


__all__ = [
    "AlphaCandidateMetrics",
    "AlphaFoldMetrics",
    "AlphaMetricAvailability",
    "AlphaMetricPolicy",
    "AlphaMetricValue",
    "seal_evaluation_contract",
]
