"""The ZERO control: no information, stated as a method rather than assumed.

Forecasting zero for every sector is the null hypothesis every real Sector
method must beat, and it is installed as a first-class capability -- with its
own recipe schema, numerical binding and content identity -- so that "the
control ran" is a verifiable claim rather than an array someone synthesized
from a method id. The Portfolio side resolves the control through this owner
for exactly that reason.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts import SectorResearchError
from .contracts import (
    BoundSectorForecastInput,
    SectorForecastRecipe,
    SectorForecastValues,
    SectorNumericalBinding,
)

ZERO_SECTOR_FORECAST_METHOD_ID = "ZERO_SECTOR_FORECAST"
ZERO_SECTOR_FORECAST_SCHEMA_ID = "sector-forecast/zero-control@1"
SECTOR_FORECAST_CONTENT_FORMAT_ID = "sector-forecast/per-sector-float64@1"


class ZeroSectorForecastAdapter:
    """Every sector, every formation: exactly ``0.0``. Always available.

    No warmup and no history requirement -- a control that could be unavailable
    would make "the method beat the control" conditional on when it was asked.
    """

    method_id = ZERO_SECTOR_FORECAST_METHOD_ID
    recipe_schema_id = ZERO_SECTOR_FORECAST_SCHEMA_ID

    def describe_numerical_binding(self) -> SectorNumericalBinding:
        """Seal this installed adapter source-rule identity and deterministic numerical policy.

        Returns:
            Exact method/content-format, implementation, stack and arithmetic/warmup binding.
        """
        return SectorNumericalBinding.create(
            method_id=self.method_id,
            forecast_content_format_id=SECTOR_FORECAST_CONTENT_FORMAT_ID,
            implementation_sources={
                "sector_research.models.zero": Path(__file__),
                "sector_research.models.contracts": Path(__file__).with_name("contracts.py"),
            },
            deterministic_policy={"arithmetic": "CONSTANT_ZERO"},
        )

    def validate_recipe(self, recipe: SectorForecastRecipe) -> None:
        """Require this installed method/schema route and its declared parameter shape.

        The control requires an empty parameter mapping.

        Args:
            recipe: Sealed recipe to validate before forecasting.

        Raises:
            SectorResearchError: Method/schema routing or the declared parameter rules fail.
        """
        if recipe.method_id != self.method_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise SectorResearchError("sector_research.recipe_route_invalid")
        if recipe.parameters != {}:
            # The control has no knobs. A parameter here would be a second
            # method wearing the control's identity.
            raise SectorResearchError("sector_research.recipe_parameters_invalid")

    def forecast(
        self,
        *,
        bound_input: BoundSectorForecastInput,
        recipe: SectorForecastRecipe,
    ) -> SectorForecastValues:
        """Return the constant-zero forecast control on the exact admitted sector axis.

        The control needs no fitted history and exposes no parameter choices; every sector receives
        an available zero.

        Args:
            bound_input: Admitted matured observations and formation/sector axes.
            recipe: Exact method recipe checked before numerical work.

        Returns:
            Sector-axis-aligned values and per-cell unavailability reasons.

        Raises:
            SectorResearchError: Recipe routing/parameters or resulting forecast cells are invalid.
        """
        self.validate_recipe(recipe)
        return SectorForecastValues(
            method_id=self.method_id,
            forecast_formation_at=bound_input.forecast_formation_at,
            ordered_sectors=bound_input.ordered_sectors,
            values=tuple(0.0 for _ in bound_input.ordered_sectors),
            unavailable_reasons=tuple(None for _ in bound_input.ordered_sectors),
        )


__all__ = [
    "SECTOR_FORECAST_CONTENT_FORMAT_ID",
    "ZERO_SECTOR_FORECAST_METHOD_ID",
    "ZERO_SECTOR_FORECAST_SCHEMA_ID",
    "ZeroSectorForecastAdapter",
]
