"""The explicitly installed cross-sectional scale methods.

Campaign methods state initialization, missing-update and causal maturity
semantics explicitly. HAR remains trigger-gated by the Program even though its
adapter and search-free recipe are installed.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, cast

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current

from .adapters import (
    AsymmetricEwmaXsDispersionAdapter,
    CrossSectionalScaleAdapter,
    EwmaXsDispersionAdapter,
    HarXsDispersionAdapter,
    LaggedXsDispersionAdapter,
)
from .contracts import (
    ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID,
    EWMA_XS_DISPERSION_RECIPE_ID,
    HAR_XS_DISPERSION_RECIPE_ID,
    LAGGED_XS_DISPERSION_RECIPE_ID,
    CrossSectionalScalingError,
    LaggedXsDispersionRecipe,
    RecursiveXsDispersionRecipe,
    XsDispersionRecipe,
)

CAMPAIGN_SCALE_CATALOG_ROLE = "alpha_research.campaign_scale_catalog"
"""The identity role of the campaign scale catalog's hash (`config/identity-roles.json`)."""


def scale_implementation_role(recipe_id: str) -> str:
    """The identity role of one scale method's implementation binding."""
    return f"alpha_research.scale_implementation.{recipe_id}"


class CrossSectionalScaleCatalog:
    """Immutable index of installed scale methods and the code that runs them."""

    def __init__(self, adapters: tuple[CrossSectionalScaleAdapter, ...]) -> None:
        """Bind a nonempty immutable catalog of unique dispersion recipe owners.

        Args:
            adapters: Installed adapters in declared registration order.

        Raises:
            CrossSectionalScalingError: The catalog is empty or recipe identifiers repeat.
        """
        indexed = {adapter.recipe_id: adapter for adapter in adapters}
        if not adapters or len(indexed) != len(adapters):
            raise CrossSectionalScalingError("SCALING_CATALOG_INVALID")
        self._adapters = MappingProxyType(indexed)

    @property
    def recipe_ids(self) -> tuple[str, ...]:
        """Read installed dispersion recipe identifiers in registration order.

        Returns:
            Ordered recipe identifiers from the immutable catalog.
        """
        return tuple(self._adapters)

    @property
    def catalog_hash(self) -> str:
        """Identity of the installed set, including each implementation's content."""
        return str(
            canonical_hash(
                {
                    "kind": "CrossSectionalScaleCatalog",
                    "ordered_methods": [
                        {
                            "recipe_id": key,
                            "implementation_binding_hash": (
                                self._adapters[key]
                                .describe_implementation_binding()
                                .implementation_binding_hash
                            ),
                        }
                        for key in sorted(self._adapters)
                    ],
                }
            )
        )

    def implementation_current(self, recipe_id: str, recorded: str) -> bool:
        """Whether a recorded implementation binding names this recipe's installed one."""
        installed = self.resolve(recipe_id).describe_implementation_binding()
        return is_current(
            scale_implementation_role(recipe_id), recorded, installed.implementation_binding_hash
        )

    def resolve(self, recipe_id: str) -> CrossSectionalScaleAdapter:
        """Resolve one exact installed dispersion recipe adapter.

        Args:
            recipe_id: Installed recipe route to resolve.

        Returns:
            Adapter owning the requested recipe.

        Raises:
            CrossSectionalScalingError: The requested recipe is not installed.
        """
        adapter = self._adapters.get(recipe_id)
        if adapter is None:
            raise CrossSectionalScalingError("SCALING_RECIPE_NOT_INSTALLED")
        return adapter

    def seal_recipe(
        self,
        *,
        recipe_id: str,
        maturity_lag_sessions: int,
        symmetric_half_life_sessions: int = 21,
    ) -> XsDispersionRecipe:
        """Seal a recipe against the outcome method's own lag.

        The lag is a parameter here rather than a constant because the method is
        the same for every horizon; only the offset differs, and it is dictated
        by the sealed outcome recipe rather than chosen by this owner.
        """
        self.resolve(recipe_id)
        if recipe_id == LAGGED_XS_DISPERSION_RECIPE_ID:
            return LaggedXsDispersionRecipe.create(source_offset_sessions=maturity_lag_sessions)
        if recipe_id == EWMA_XS_DISPERSION_RECIPE_ID:
            if symmetric_half_life_sessions not in {10, 21, 42}:
                raise CrossSectionalScalingError("SCALING_EWMA_HALF_LIFE_INVALID")
            return RecursiveXsDispersionRecipe.create(
                recipe_id=cast(
                    Literal[
                        "EWMA_XS_DISPERSION",
                        "ASYMMETRIC_EWMA_XS_DISPERSION",
                        "HAR_XS_DISPERSION",
                    ],
                    EWMA_XS_DISPERSION_RECIPE_ID,
                ),
                source_offset_sessions=maturity_lag_sessions,
                symmetric_half_life_sessions=cast(
                    Literal[10, 21, 42], symmetric_half_life_sessions
                ),
            )
        if recipe_id in {
            ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID,
            HAR_XS_DISPERSION_RECIPE_ID,
        }:
            return RecursiveXsDispersionRecipe.create(
                recipe_id=cast(
                    Literal[
                        "EWMA_XS_DISPERSION",
                        "ASYMMETRIC_EWMA_XS_DISPERSION",
                        "HAR_XS_DISPERSION",
                    ],
                    recipe_id,
                ),
                source_offset_sessions=maturity_lag_sessions,
            )
        raise CrossSectionalScalingError("SCALING_RECIPE_NOT_INSTALLED")


def installed_scale_implementation_hash(recipe_id: str) -> str:
    """The implementation binding this build installs for one scale method: its role's value."""
    adapter = build_alpha_campaign_cross_sectional_scale_catalog().resolve(recipe_id)
    return adapter.describe_implementation_binding().implementation_binding_hash


def build_installed_cross_sectional_scale_catalog() -> CrossSectionalScaleCatalog:
    """Keep the product/default mandate on the already-admitted control."""
    return CrossSectionalScaleCatalog((LaggedXsDispersionAdapter(),))


def build_alpha_campaign_cross_sectional_scale_catalog() -> CrossSectionalScaleCatalog:
    """Install development-only Campaign competitors without production admission."""
    return CrossSectionalScaleCatalog(
        (
            LaggedXsDispersionAdapter(),
            EwmaXsDispersionAdapter(),
            AsymmetricEwmaXsDispersionAdapter(),
            HarXsDispersionAdapter(),
        )
    )


__all__ = [
    "CrossSectionalScaleCatalog",
    "build_alpha_campaign_cross_sectional_scale_catalog",
    "build_installed_cross_sectional_scale_catalog",
]
