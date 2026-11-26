"""Explicit immutable catalog of Sector forecast methods installed by the Host.

The clean-target catalog installs the first coherent method batch:
``ZERO_SECTOR_FORECAST``, ``EWMA_SECTOR_MEAN``, ``SECTOR_RELATIVE_STRENGTH_126``
and ``DISTRIBUTED_LAG_ELASTIC_NET``. ``SECTOR_EXCESS_EMA_LEGACY`` is
deliberately absent: it modelled market-excess returns against a different
target, and only its surfaces' readback remains, so putting it here would
present a different question as a control for this one.

The batch is a null control, a persistence control and two candidates, which is
what makes the comparison answerable: without ``ZERO`` present a candidate can
only be compared against another candidate, and "better than EWMA" is not the
question a Sector expected-return branch has to answer.

There is no parameter domain and no borrowed search grid. Each method has one
admissible recipe -- a frozen singleton, every field inside recipe identity --
and ``seal_recipe`` refuses anything else. A Sector search domain is declared if
and when a Campaign asks for one; installing a grid today would smuggle another
Desk's plan into one it was never written for.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import SectorResearchError
from .contracts import (
    SectorForecastAdapter,
    SectorForecastCapabilityIdentity,
    SectorForecastCatalogBinding,
    SectorForecastRecipe,
)
from .distributed_lag_elastic_net import (
    DISTRIBUTED_LAG_ELASTIC_NET_METHOD_ID,
    DistributedLagElasticNetAdapter,
)
from .ewma import EWMA_SECTOR_MEAN_METHOD_ID, EwmaSectorMeanAdapter
from .relative_strength import (
    SECTOR_RELATIVE_STRENGTH_METHOD_ID,
    SectorRelativeStrengthAdapter,
)
from .zero import ZERO_SECTOR_FORECAST_METHOD_ID, ZeroSectorForecastAdapter

FROZEN_SINGLETON_PARAMETERS: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        ZERO_SECTOR_FORECAST_METHOD_ID: MappingProxyType({}),
        EWMA_SECTOR_MEAN_METHOD_ID: MappingProxyType(
            {
                "half_life_sessions": 21,
                "minimum_history_sessions": 21,
                "refit_every_sessions": 21,
                "forecast_horizon_sessions": 1,
            }
        ),
        SECTOR_RELATIVE_STRENGTH_METHOD_ID: MappingProxyType(
            {
                "lookback_sessions": 126,
                "minimum_history_sessions": 126,
                "refit_every_sessions": 21,
                "forecast_horizon_sessions": 1,
            }
        ),
        DISTRIBUTED_LAG_ELASTIC_NET_METHOD_ID: MappingProxyType(
            {
                "lag_count": 5,
                "minimum_history_sessions": 252,
                "refit_every_sessions": 21,
                "forecast_horizon_sessions": 1,
                # 0.10 of alpha_max, an even L1/L2 mix. A single declared point:
                # a Sector search domain does not exist, so this is the method,
                # not a default somebody may tune at the request boundary.
                "alpha_max_multiplier_basis_points": 1000,
                "l1_ratio_percent": 50,
            }
        ),
    }
)
"""The whole admissible parameter space, one point per method."""


class SectorForecastCatalog:
    """Immutable installed-capability index resolved by method and recipe schema."""

    def __init__(self, adapters: tuple[SectorForecastAdapter, ...]) -> None:
        """Hold a nonempty immutable catalog of unique method and recipe-schema owners.

        Args:
            adapters: Installed forecast adapters in registration order.

        Raises:
            SectorResearchError: The adapter list is empty or method/schema identifiers repeat.
        """
        indexed = {value.method_id: value for value in adapters}
        by_schema = {value.recipe_schema_id: value for value in adapters}
        if not adapters or len(indexed) != len(adapters) or len(by_schema) != len(adapters):
            raise SectorResearchError("sector_research.catalog_invalid")
        self._adapters = MappingProxyType(indexed)

    @property
    def method_ids(self) -> tuple[str, ...]:
        """Return installed forecast method identifiers in registration order.

        Returns:
            Ordered method identifiers from the immutable catalog.
        """
        return tuple(self._adapters)

    @property
    def adapters(self) -> tuple[SectorForecastAdapter, ...]:
        """The installed adapters in registration order.

        Exposed so a conformance guard can read each installed method's declared
        numerical binding without reaching into private state. The underlying
        mapping stays immutable; the Risk catalog already provides the same
        accessor for the same reason.
        """
        return tuple(self._adapters.values())

    @property
    def binding(self) -> SectorForecastCatalogBinding:
        """Seal installed methods, numerical bindings and admitted singleton recipes.

        Returns:
            Exact ordered capability catalog and canonical catalog_hash.

        Raises:
            SectorResearchError: An installed adapter cannot describe or admit its frozen recipe.
        """
        values = tuple(
            SectorForecastCapabilityIdentity(
                method_id=adapter.method_id,
                recipe_schema_id=adapter.recipe_schema_id,
                numerical_binding_hash=adapter.describe_numerical_binding().numerical_binding_hash,
                # The frozen singleton is part of the installed identity: sealed
                # through the same admission path a caller uses, so the binding
                # cannot drift from what seal_recipe would actually admit.
                admitted_recipe_hash=self.seal_recipe(method_id=adapter.method_id).recipe_hash,
            )
            for adapter in self._adapters.values()
        )
        identity = {"ordered_capabilities": [value.model_dump(mode="json") for value in values]}
        return SectorForecastCatalogBinding(
            ordered_capabilities=values,
            catalog_hash=canonical_hash(identity),
        )

    def frozen_singleton_parameters(self, method_id: str) -> dict[str, int]:
        """Return a copy of the only admitted parameters for an installed method.

        Args:
            method_id: Exact installed forecast method identifier.

        Returns:
            Declared singleton parameter mapping; changing the copy grants no tuning authority.

        Raises:
            SectorResearchError: The method is absent or its singleton is undeclared.
        """
        if method_id not in self._adapters:
            raise SectorResearchError("sector_research.method_not_installed")
        admitted = FROZEN_SINGLETON_PARAMETERS.get(method_id)
        if admitted is None:
            raise SectorResearchError("sector_research.method_singleton_undeclared")
        return dict(admitted)

    def seal_recipe(
        self,
        *,
        method_id: str,
        parameters: Mapping[str, int] | None = None,
    ) -> SectorForecastRecipe:
        """Seal the one admissible recipe for an installed method.

        A caller naming parameters is stating a claim, not making a choice: the
        claim must equal the frozen singleton exactly, or the recipe is refused.
        This is the boundary that keeps "no search domain" true in code rather
        than in a comment.

        Args:
            method_id: Exact installed method identifier.
            parameters: Optional claim that must equal the frozen singleton exactly.

        Returns:
            Sealed singleton recipe after adapter validation.

        Raises:
            SectorResearchError: The method is absent or the parameter claim differs from its
                singleton.
        """
        admitted = self.frozen_singleton_parameters(method_id)
        if parameters is not None and dict(parameters) != admitted:
            raise SectorResearchError("sector_research.recipe_outside_frozen_singleton")
        adapter = self._adapters[method_id]
        recipe = SectorForecastRecipe.create(
            method_id=method_id,
            recipe_schema_id=adapter.recipe_schema_id,
            parameters=admitted,
        )
        adapter.validate_recipe(recipe)
        return recipe

    def resolve(self, recipe: SectorForecastRecipe) -> SectorForecastAdapter:
        """Resolve and validate a recipe against its installed method and numerical owner.

        Args:
            recipe: Sealed forecast recipe whose method/schema route is being resolved.

        Returns:
            Installed adapter after method, schema, numerical route and parameter checks.

        Raises:
            SectorResearchError: Method/schema installation, numerical route or recipe validation
                fails.
        """
        adapter = self._adapters.get(recipe.method_id)
        if adapter is None:
            raise SectorResearchError("sector_research.method_not_installed")
        if adapter.recipe_schema_id != recipe.recipe_schema_id:
            raise SectorResearchError("sector_research.recipe_schema_not_installed")
        binding = adapter.describe_numerical_binding()
        if binding.method_id != adapter.method_id:
            raise SectorResearchError("sector_research.numerical_binding_route_invalid")
        adapter.validate_recipe(recipe)
        return adapter


SECTOR_CATALOG_ROLE = "sector_research.forecast_catalog"
"""The identity role of the installed Sector catalog's hash (`config/identity-roles.json`)."""


def numerical_binding_role(method_id: str) -> str:
    """The identity role of one installed Sector method's numerical binding."""
    return f"sector_research.method.{method_id}"


def installed_numerical_binding_hash(method_id: str) -> str:
    """The numerical binding this build installs for one Sector method: its role's value."""
    for adapter in build_installed_sector_forecast_catalog().adapters:
        if adapter.method_id == method_id:
            return adapter.describe_numerical_binding().numerical_binding_hash
    raise SectorResearchError("sector_research.method_not_installed")


def build_installed_sector_forecast_catalog() -> SectorForecastCatalog:
    """Construct the explicit installed catalog; no discovery, no runtime mutation."""
    return SectorForecastCatalog(
        (
            ZeroSectorForecastAdapter(),
            EwmaSectorMeanAdapter(),
            SectorRelativeStrengthAdapter(),
            DistributedLagElasticNetAdapter(),
        )
    )


__all__ = [
    "FROZEN_SINGLETON_PARAMETERS",
    "SectorForecastCatalog",
    "build_installed_sector_forecast_catalog",
]
