"""``EWMA_SECTOR_MEAN``: an exponentially weighted mean over matured history.

The estimator is a *normalized weighted mean*, not a seeded recursion. The two
agree once history is long, but they differ exactly where it matters: a
recursion needs an initial value, and every choice of seed -- first
observation, zero, a long-run mean -- is information the method was never
given. The explicit-warmup decision this Desk confirmed is implemented by
refusing instead: below ``minimum_history_sessions`` matured observations a
sector emits no forecast and a typed reason, and above it the forecast is the
exponentially weighted mean of exactly the observations that exist. No zero
fill, no backfill, no seeded fallback.

Weights decay per ordered training observation. The training axis the Host
binds is the matured formation-session axis, so one step is one exchange
session of matured history; the half-life is stated in those sessions, and the
recipe carries it as identity rather than as a default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts import SectorResearchError
from .contracts import (
    INSUFFICIENT_MATURED_HISTORY,
    BoundSectorForecastInput,
    SectorForecastRecipe,
    SectorForecastValues,
    SectorNumericalBinding,
)
from .zero import SECTOR_FORECAST_CONTENT_FORMAT_ID

EWMA_SECTOR_MEAN_METHOD_ID = "EWMA_SECTOR_MEAN"
EWMA_SECTOR_MEAN_SCHEMA_ID = "sector-forecast/ewma-sector-mean@1"

_PARAMETER_FIELDS = (
    "forecast_horizon_sessions",
    "half_life_sessions",
    "minimum_history_sessions",
    "refit_every_sessions",
)


class EwmaSectorMeanAdapter:
    """Deterministic single-pass float64 EWMA of the clean sector-mean target."""

    method_id = EWMA_SECTOR_MEAN_METHOD_ID
    recipe_schema_id = EWMA_SECTOR_MEAN_SCHEMA_ID

    def describe_numerical_binding(self) -> SectorNumericalBinding:
        """Seal this installed adapter source-rule identity and deterministic numerical policy.

        Returns:
            Exact method/content-format, implementation, stack and arithmetic/warmup binding.
        """
        return SectorNumericalBinding.create(
            method_id=self.method_id,
            forecast_content_format_id=SECTOR_FORECAST_CONTENT_FORMAT_ID,
            implementation_sources={
                "sector_research.models.ewma": Path(__file__),
                "sector_research.models.contracts": Path(__file__).with_name("contracts.py"),
            },
            deterministic_policy={
                "arithmetic": "SINGLE_PASS_FLOAT64_NORMALIZED_EXPONENTIAL_MEAN",
                "warmup": "EXPLICIT_REFUSAL_BELOW_MINIMUM_HISTORY",
            },
        )

    def validate_recipe(self, recipe: SectorForecastRecipe) -> None:
        """Require this installed method/schema route and its declared parameter shape.

        The declared half-life and minimum-history parameters must be positive integers.

        Args:
            recipe: Sealed recipe to validate before forecasting.

        Raises:
            SectorResearchError: Method/schema routing or the declared parameter rules fail.
        """
        if recipe.method_id != self.method_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise SectorResearchError("sector_research.recipe_route_invalid")
        if tuple(sorted(recipe.parameters)) != _PARAMETER_FIELDS or any(
            recipe.parameters[field] < 1 for field in _PARAMETER_FIELDS
        ):
            raise SectorResearchError("sector_research.recipe_parameters_invalid")

    def forecast(
        self,
        *,
        bound_input: BoundSectorForecastInput,
        recipe: SectorForecastRecipe,
    ) -> SectorForecastValues:
        """Forecast each sector with a normalized exponential mean of matured observations.

        Finite history is weighted by session age; a sector below minimum_history_sessions returns
        INSUFFICIENT_MATURED_HISTORY.

        Args:
            bound_input: Admitted matured observations and formation/sector axes.
            recipe: Exact method recipe checked before numerical work.

        Returns:
            Sector-axis-aligned values and per-cell unavailability reasons.

        Raises:
            SectorResearchError: Recipe routing/parameters or resulting forecast cells are invalid.
        """
        self.validate_recipe(recipe)
        half_life = float(recipe.parameters["half_life_sessions"])
        minimum_history = int(recipe.parameters["minimum_history_sessions"])
        observations = bound_input.training_values
        count = observations.shape[0]
        # Age counted back from the most recent matured observation, so the
        # newest label always carries weight one before normalization.
        ages = np.arange(count - 1, -1, -1, dtype=np.float64)
        decay = np.power(0.5, ages / half_life)
        values: list[float | None] = []
        reasons: list[str | None] = []
        for column in range(len(bound_input.ordered_sectors)):
            history = observations[:, column]
            finite = np.isfinite(history)
            if int(finite.sum()) < minimum_history:
                values.append(None)
                reasons.append(INSUFFICIENT_MATURED_HISTORY)
                continue
            weights = decay[finite]
            values.append(float(np.sum(weights * history[finite]) / np.sum(weights)))
            reasons.append(None)
        return SectorForecastValues(
            method_id=self.method_id,
            forecast_formation_at=bound_input.forecast_formation_at,
            ordered_sectors=bound_input.ordered_sectors,
            values=tuple(values),
            unavailable_reasons=tuple(reasons),
        )


__all__ = [
    "EWMA_SECTOR_MEAN_METHOD_ID",
    "EWMA_SECTOR_MEAN_SCHEMA_ID",
    "EwmaSectorMeanAdapter",
]
