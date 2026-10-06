"""Explicit immutable catalog of Alpha model adapters installed by the Host."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    AlphaModelAdapter,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaModelCapabilityIdentity(_Contract):
    """Identify one installed adapter's qualified recipe, domain and numerical capability.

    Attributes:
        adapter_id: Stable installed adapter handle.
        recipe_schema_id: Schema used to interpret numerical recipe fields.
        search_domain_schema_id: Schema used to interpret admitted search constraints.
        numerical_binding_hash: Declared numerical capability identity, absent only in legacy
            records.
    """

    adapter_id: str = Field(min_length=1, max_length=96)
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    search_domain_schema_id: str = Field(min_length=1, max_length=128)
    numerical_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class AlphaModelCatalogBinding(_Contract):
    """Content identity of the explicitly installed Host model capabilities."""

    ordered_capabilities: tuple[AlphaModelCapabilityIdentity, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> AlphaModelCatalogBinding:
        """Verify unique capabilities and the supported catalog identity representation.

        Returns:
            This validated immutable catalog binding.

        Raises:
            ValueError: Adapter handles repeat or neither current nor eligible legacy content hash
                matches.
        """
        keys = tuple(value.adapter_id for value in self.ordered_capabilities)
        if len(set(keys)) != len(keys):
            raise ValueError("ALPHA_MODEL_CATALOG_CAPABILITY_DUPLICATED")
        identity = self.model_dump(mode="json", exclude={"catalog_hash"})
        accepted_hashes = {canonical_hash(identity)}
        if all(value.numerical_binding_hash is None for value in self.ordered_capabilities):
            legacy_identity = {
                "ordered_capabilities": [
                    {
                        key: value
                        for key, value in capability.items()
                        if key != "numerical_binding_hash"
                    }
                    for capability in identity["ordered_capabilities"]
                ]
            }
            accepted_hashes.add(canonical_hash(legacy_identity))
        if self.catalog_hash not in accepted_hashes:
            raise ValueError("ALPHA_MODEL_CATALOG_IDENTITY_INVALID")
        return self

    def restricted_to(self, adapter_ids: Iterable[str]) -> AlphaModelCatalogBinding:
        """The capabilities named, in this binding's order, and nothing installed beside them.

        What a mandate admits of a catalog (V118): the models a study or a goal may run,
        so a model installed beside them moves no identity that binds them. Refuses an
        adapter this binding does not hold.

        Args:
            adapter_ids: Nonempty set of handles which this binding already contains.

        Returns:
            Sealed subset preserving this binding's capability order.

        Raises:
            ValueError: The subset is empty or requests an uninstalled handle.
        """
        wanted = set(adapter_ids)
        values = tuple(value for value in self.ordered_capabilities if value.adapter_id in wanted)
        if not values or len(values) != len(wanted):
            raise ValueError("ALPHA_MODEL_ADAPTER_NOT_INSTALLED")
        identity = {"ordered_capabilities": [value.model_dump(mode="json") for value in values]}
        return AlphaModelCatalogBinding(
            ordered_capabilities=values, catalog_hash=canonical_hash(identity)
        )


class AlphaModelCatalog:
    """Resolve an explicit immutable set of Host-installed model adapters.

    Registration order is retained in capability readouts. Recipes and domains must
    name installed schemas, and each adapter validates its own numerical authority.
    """

    def __init__(self, adapters: tuple[AlphaModelAdapter, ...]) -> None:
        """Install a nonempty explicit set of uniquely named model adapters.

        Args:
            adapters: Qualified adapter implementations in registration order.

        Raises:
            ValueError: The installed set is empty or repeats an adapter handle.
        """
        indexed = {value.adapter_id: value for value in adapters}
        if not adapters or len(indexed) != len(adapters):
            raise ValueError("ALPHA_MODEL_CATALOG_INVALID")
        self._adapters = MappingProxyType(indexed)

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        """Read installed adapter handles in registration order.

        Returns:
            The immutable catalog's ordered adapter handles.
        """
        return tuple(self._adapters)

    @property
    def adapters(self) -> tuple[AlphaModelAdapter, ...]:
        """The installed adapters in registration order.

        Exposed so a conformance guard can read each installed capability's
        declared numerical binding without reaching into private state or
        inventing a recipe to resolve one. The underlying mapping stays
        immutable; this is the same accessor the Risk catalog already provides.
        """
        return tuple(self._adapters.values())

    @property
    def binding(self) -> AlphaModelCatalogBinding:
        """Seal the installed schema and declared numerical capabilities in catalog order.

        Returns:
            Validated catalog binding containing each installed adapter's numerical-policy identity.
        """
        values = tuple(
            AlphaModelCapabilityIdentity(
                adapter_id=adapter.adapter_id,
                recipe_schema_id=adapter.recipe_schema_id,
                search_domain_schema_id=adapter.search_domain_schema_id,
                numerical_binding_hash=adapter.describe_numerical_binding().numerical_binding_hash,
            )
            for adapter in self._adapters.values()
        )
        identity = {"ordered_capabilities": [value.model_dump(mode="json") for value in values]}
        return AlphaModelCatalogBinding(
            ordered_capabilities=values,
            catalog_hash=canonical_hash(identity),
        )

    def adapter(self, adapter_id: str) -> AlphaModelAdapter:
        """One installed adapter, by its id.

        Args:
            adapter_id: The adapter.

        Returns:
            It.

        Raises:
            ValueError: `ALPHA_MODEL_ADAPTER_NOT_INSTALLED`.
        """
        try:
            return self._adapters[adapter_id]
        except KeyError as error:
            raise ValueError("ALPHA_MODEL_ADAPTER_NOT_INSTALLED") from error

    def resolve(self, recipe: AlphaModelRecipeEnvelope) -> AlphaModelAdapter:
        """Resolve an installed adapter and validate its recipe route and numerical declaration.

        Args:
            recipe: Host-sealed recipe naming an installed adapter and qualified schema.

        Returns:
            The installed adapter after recipe validation.

        Raises:
            ValueError: Adapter/schema is not installed, numerical binding names another adapter, or
                the recipe is invalid.
        """
        try:
            adapter = self._adapters[recipe.adapter_id]
        except KeyError as error:
            raise ValueError("ALPHA_MODEL_ADAPTER_NOT_INSTALLED") from error
        if adapter.recipe_schema_id != recipe.recipe_schema_id:
            raise ValueError("ALPHA_MODEL_RECIPE_SCHEMA_NOT_INSTALLED")
        binding = adapter.describe_numerical_binding()
        if binding.adapter_id != adapter.adapter_id:
            raise ValueError("ALPHA_MODEL_NUMERICAL_BINDING_ROUTE_INVALID")
        adapter.validate_recipe(recipe)
        return adapter

    def resolve_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> AlphaModelAdapter:
        """Resolve an installed adapter and validate its domain route and constraints.

        Args:
            domain: Host-sealed recipe/domain schema and numerical constraints.

        Returns:
            The installed adapter after domain validation.

        Raises:
            ValueError: Adapter/schemas are not installed, numerical route differs, or domain
                constraints are invalid.
        """
        try:
            adapter = self._adapters[domain.adapter_id]
        except KeyError as error:
            raise ValueError("ALPHA_MODEL_ADAPTER_NOT_INSTALLED") from error
        if adapter.recipe_schema_id != domain.recipe_schema_id:
            raise ValueError("ALPHA_MODEL_RECIPE_SCHEMA_NOT_INSTALLED")
        if adapter.search_domain_schema_id != domain.search_domain_schema_id:
            raise ValueError("ALPHA_MODEL_SEARCH_DOMAIN_SCHEMA_NOT_INSTALLED")
        binding = adapter.describe_numerical_binding()
        if binding.adapter_id != adapter.adapter_id:
            raise ValueError("ALPHA_MODEL_NUMERICAL_BINDING_ROUTE_INVALID")
        adapter.validate_search_domain(domain)
        return adapter

    def admit_recipe(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> AlphaModelAdapter:
        """Require a recipe and domain to share the installed route and admit their parameters.

        Args:
            recipe: Qualified numerical configuration proposed for execution.
            domain: Host-admitted constraints for the same adapter and recipe schema.

        Returns:
            The installed adapter after recipe-within-domain validation.

        Raises:
            ValueError: Routes disagree, capability is not installed, or the recipe lies outside the
                admitted domain.
        """
        if (
            recipe.adapter_id != domain.adapter_id
            or recipe.recipe_schema_id != domain.recipe_schema_id
        ):
            raise ValueError("ALPHA_MODEL_RECIPE_DOMAIN_ROUTE_INVALID")
        adapter = self.resolve_search_domain(domain)
        adapter.validate_recipe_for_domain(recipe=recipe, domain=domain)
        return adapter


def build_installed_alpha_model_catalog(extensions: Iterable[str] = ()) -> AlphaModelCatalog:
    """Construct the explicit installed catalog; no discovery or runtime mutation.

    The regularized linear family stays first, so the capability handle every
    existing declaration names keeps resolving to it; the Dynamic Panel
    LightGBM adapter -- the one the installed G2/G6 components train and score
    with -- is the second installed research capability. Installation opens the
    research inventory; the frozen strategies' recipes, current pointers and
    capital are untouched by it.

    Args:
        extensions: Explicit names of activated workspace model declarations.

    Returns:
        Immutable installed catalog with the maintained linear and Dynamic Panel
        LightGBM adapters followed by the explicitly named extensions.

    Raises:
        ValueError: An extension declaration or resulting catalog cannot be admitted.
    """
    from .adapters.lightgbm_dynamic_panel import DynamicPanelLightGBMAdapter
    from .adapters.regularized_linear import RegularizedLinearAdapter
    from .extension import extension_adapter

    # A workspace's activated models follow the installed ones, loaded by name (EX): a
    # Program binds only the models its mandate admits (V118), so they move no result.
    return AlphaModelCatalog(
        (
            RegularizedLinearAdapter(),
            DynamicPanelLightGBMAdapter(),
            *(extension_adapter(value) for value in extensions),
        )
    )


def build_installed_alpha_model_search_domains(
    catalog: AlphaModelCatalog,
) -> tuple[AlphaModelSearchDomainEnvelope, ...]:
    """The domain each of a catalog's adapters admits, in its order.

    An installed model's is its builder's; an extension's is its declaration's axes (EX).

    Args:
        catalog: The catalog.

    Returns:
        The domains.

    Raises:
        ValueError: `ALPHA_MODEL_SEARCH_DOMAIN_NOT_INSTALLED` for an adapter with none.
    """
    from .adapters.lightgbm_dynamic_panel import (
        DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID,
        build_dynamic_panel_lightgbm_search_domain,
    )
    from .adapters.regularized_linear import (
        REGULARIZED_LINEAR_ADAPTER_ID,
        build_regularized_linear_search_domain,
    )
    from .extension import DeclaredModelAdapter

    builders = {
        REGULARIZED_LINEAR_ADAPTER_ID: build_regularized_linear_search_domain,
        DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID: build_dynamic_panel_lightgbm_search_domain,
    }
    domains = []
    for adapter in catalog.adapters:
        if isinstance(adapter, DeclaredModelAdapter):
            domains.append(adapter.declared_search_domain())
        elif adapter.adapter_id in builders:
            domains.append(builders[adapter.adapter_id]())
        else:
            raise ValueError("ALPHA_MODEL_SEARCH_DOMAIN_NOT_INSTALLED")
    return tuple(domains)


def build_alpha_campaign_development_catalog() -> AlphaModelCatalog:
    """Install Campaign-only competitors without widening production mandate."""
    from .adapters.hist_gradient_boosting import HistGradientBoostingAdapter
    from .adapters.huber import HuberLinearAdapter
    from .adapters.lightgbm_chronological import (
        ChronologicalLightGBMAdapter,
        RegularizedChronologicalLightGBMAdapter,
    )
    from .adapters.rank_composite import RankCompositeAdapter
    from .adapters.regularized_linear import RegularizedLinearAdapter

    return AlphaModelCatalog(
        (
            RegularizedLinearAdapter(),
            RankCompositeAdapter(),
            ChronologicalLightGBMAdapter(),
            RegularizedChronologicalLightGBMAdapter(),
            HistGradientBoostingAdapter(),
            HuberLinearAdapter(),
        )
    )


__all__ = [
    "AlphaModelCapabilityIdentity",
    "AlphaModelCatalog",
    "AlphaModelCatalogBinding",
    "build_alpha_campaign_development_catalog",
    "build_installed_alpha_model_catalog",
]
