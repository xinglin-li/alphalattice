"""Lightweight contracts for preregistered multi-frequency research schedules."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
)
from alphalattice.kernel.validation.enums import (
    RebalanceFrequency,
)

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class CadenceThresholdProfile(DomainModel):
    """Frozen observation and bootstrap thresholds for one cadence."""

    frequency: RebalanceFrequency
    train_observations: PositiveInt
    validation_observations: PositiveInt
    step_observations: PositiveInt
    holdout_observations: PositiveInt
    purge_observations: Literal[1] = 1
    embargo_observations: Literal[0] = 0
    minimum_folds: Literal[3] = 3
    minimum_calendar_years: PositiveInt
    minimum_bootstrap_observations: PositiveInt
    minimum_bootstrap_calendar_years: Literal[3] = 3
    bootstrap_seed: Literal[1729] = 1729
    bootstrap_resamples: Literal[2000] = 2000
    annualization_periods: PositiveInt

    @model_validator(mode="after")
    def validate_frozen_profile(self) -> CadenceThresholdProfile:
        """Require the threshold tuple registered for this cadence."""
        expected = {
            RebalanceFrequency.DAILY: (756, 252, 252, 252, 7, 756, 252),
            RebalanceFrequency.WEEKLY: (156, 52, 52, 52, 7, 156, 52),
            RebalanceFrequency.MONTHLY: (60, 12, 12, 12, 9, 36, 12),
        }[self.frequency]
        actual = (
            self.train_observations,
            self.validation_observations,
            self.step_observations,
            self.holdout_observations,
            self.minimum_calendar_years,
            self.minimum_bootstrap_observations,
            self.annualization_periods,
        )
        if actual != expected:
            raise ValueError("cadence threshold profile differs from version 1")
        return self


def cadence_threshold_profile(frequency: RebalanceFrequency) -> CadenceThresholdProfile:
    """Build the frozen threshold profile for a rebalance cadence.

    Args:
        frequency: Daily, weekly or monthly rebalance cadence.

    Returns:
        The validated threshold profile for that cadence.

    """
    values = {
        RebalanceFrequency.DAILY: (756, 252, 252, 252, 7, 756, 252),
        RebalanceFrequency.WEEKLY: (156, 52, 52, 52, 7, 156, 52),
        RebalanceFrequency.MONTHLY: (60, 12, 12, 12, 9, 36, 12),
    }[frequency]
    return CadenceThresholdProfile(
        frequency=frequency,
        train_observations=values[0],
        validation_observations=values[1],
        step_observations=values[2],
        holdout_observations=values[3],
        minimum_calendar_years=values[4],
        minimum_bootstrap_observations=values[5],
        annualization_periods=values[6],
    )


__all__ = ["CadenceThresholdProfile", "cadence_threshold_profile"]
