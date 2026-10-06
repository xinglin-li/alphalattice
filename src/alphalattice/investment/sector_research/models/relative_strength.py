"""``SECTOR_RELATIVE_STRENGTH_126``: slow cross-sector strength, per session.

A slow *input* method, not a slow target. The declared outcome is unchanged --
the next-session Sector mean constituent log return -- and this method forecasts
exactly that; the 126 sessions describe how far back the signal looks, not what
it predicts. If a half-year of relative strength carries no daily predictive
scale, the Sector shrink-to-zero calibration is where that is discovered, and a
slope at zero is an admissible result rather than a failure of this method.

Formula, over the matured training matrix the Host binds (rows are formation
sessions ascending, columns are the ordered sectors, values are realized
one-session Sector mean constituent log returns)::

    window_s   = the last `lookback_sessions` rows, finite cells only
    strength_s = sum(window_s)
    breadth    = mean over sectors that produced a strength
    forecast_s = (strength_s - breadth) / lookback_sessions

Three decisions, each forced rather than chosen. The sum of consecutive
one-session log returns *is* the cumulative log return over the window, so no
separate price path is needed and no new causal input is introduced.
Subtracting the cross-sector mean is what makes the strength relative -- the
common market move is not sector information. Dividing by the window converts a
126-session cumulative log return into the per-session units of the declared
target, without which the raw forecast would sit roughly two orders of
magnitude above the value it is compared against and its evaluation lane would
measure scale rather than skill.

Units: natural log return per session, same as the target.

Worked example, three sectors and a four-session window, cumulative window sums
`(0.08, 0.02, -0.04)`. The breadth is `0.02`, the relative strengths are
`(0.06, 0.00, -0.06)`, and the forecasts are `(0.015, 0.0, -0.015)` log return
per session. They sum to zero across sectors by construction, which is the
method saying it takes no view on the market level.

Warmup is a refusal, not a fill: below `minimum_history_sessions` finite
observations inside the window a sector emits no forecast and a typed reason. A
sector that refuses is also excluded from the breadth, so the cross-sector mean
is taken over the sectors that actually produced a strength rather than over a
padded axis.
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

SECTOR_RELATIVE_STRENGTH_METHOD_ID = "SECTOR_RELATIVE_STRENGTH_126"
SECTOR_RELATIVE_STRENGTH_SCHEMA_ID = "sector-forecast/sector-relative-strength-126@1"

_PARAMETER_FIELDS = (
    "forecast_horizon_sessions",
    "lookback_sessions",
    "minimum_history_sessions",
    "refit_every_sessions",
)


class SectorRelativeStrengthAdapter:
    """Deterministic float64 cross-sector relative strength on the clean target."""

    method_id = SECTOR_RELATIVE_STRENGTH_METHOD_ID
    recipe_schema_id = SECTOR_RELATIVE_STRENGTH_SCHEMA_ID

    def describe_numerical_binding(self) -> SectorNumericalBinding:
        """Seal this installed adapter source-rule identity and deterministic numerical policy.

        Returns:
            Exact method/content-format, implementation, stack and arithmetic/warmup binding.
        """
        return SectorNumericalBinding.create(
            method_id=self.method_id,
            forecast_content_format_id=SECTOR_FORECAST_CONTENT_FORMAT_ID,
            implementation_sources={
                "sector_research.models.relative_strength": Path(__file__),
                "sector_research.models.contracts": Path(__file__).with_name("contracts.py"),
            },
            deterministic_policy={
                "arithmetic": "SINGLE_PASS_FLOAT64_WINDOW_SUM_MINUS_CROSS_SECTOR_MEAN",
                "scale": "DIVIDED_BY_LOOKBACK_TO_TARGET_SESSION_UNITS",
                "warmup": "EXPLICIT_REFUSAL_BELOW_MINIMUM_HISTORY",
            },
        )

    def validate_recipe(self, recipe: SectorForecastRecipe) -> None:
        """Require this installed method/schema route and its declared parameter shape.

        Positive declared lookback/minimum-history parameters are required; minimum history cannot
        exceed lookback.

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
        if recipe.parameters["minimum_history_sessions"] > recipe.parameters["lookback_sessions"]:
            # A minimum the window can never supply would refuse every sector
            # forever while still looking like an installed method.
            raise SectorResearchError("sector_research.recipe_parameters_invalid")

    def forecast(
        self,
        *,
        bound_input: BoundSectorForecastInput,
        recipe: SectorForecastRecipe,
    ) -> SectorForecastValues:
        """Forecast recent sector strength relative to the available cross-sector mean.

        Finite lookback sums are centered across produced sectors and divided by the declared
        lookback. A sector below minimum history remains unavailable.

        Args:
            bound_input: Admitted matured observations and formation/sector axes.
            recipe: Exact method recipe checked before numerical work.

        Returns:
            Sector-axis-aligned values and per-cell unavailability reasons.

        Raises:
            SectorResearchError: Recipe routing/parameters or resulting forecast cells are invalid.
        """
        self.validate_recipe(recipe)
        lookback = int(recipe.parameters["lookback_sessions"])
        minimum_history = int(recipe.parameters["minimum_history_sessions"])
        window = bound_input.training_values[-lookback:, :]

        strengths: list[float | None] = []
        for column in range(len(bound_input.ordered_sectors)):
            history = window[:, column]
            finite = np.isfinite(history)
            if int(finite.sum()) < minimum_history:
                strengths.append(None)
                continue
            strengths.append(float(np.sum(history[finite])))

        produced = [value for value in strengths if value is not None]
        # Every sector refused: there is no cross-sector mean to subtract, and
        # inventing one from an empty axis would be a forecast about nothing.
        breadth = float(np.mean(np.asarray(produced, dtype=np.float64))) if produced else 0.0

        values: list[float | None] = []
        reasons: list[str | None] = []
        for strength in strengths:
            if strength is None:
                values.append(None)
                reasons.append(INSUFFICIENT_MATURED_HISTORY)
                continue
            values.append((strength - breadth) / float(lookback))
            reasons.append(None)
        return SectorForecastValues(
            method_id=self.method_id,
            forecast_formation_at=bound_input.forecast_formation_at,
            ordered_sectors=bound_input.ordered_sectors,
            values=tuple(values),
            unavailable_reasons=tuple(reasons),
        )


__all__ = [
    "SECTOR_RELATIVE_STRENGTH_METHOD_ID",
    "SECTOR_RELATIVE_STRENGTH_SCHEMA_ID",
    "SectorRelativeStrengthAdapter",
]
