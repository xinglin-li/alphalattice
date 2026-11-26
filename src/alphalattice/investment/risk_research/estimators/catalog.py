"""Explicit immutable catalog of Risk estimator adapters installed by the Host.

The catalog describes what the Host has explicitly installed. It never expresses
current admission, and it performs no filesystem or entry-point discovery.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .capability import (
    RiskCapability,
    RiskCapabilityError,
    RiskRecipeAdmission,
    admit_determinism,
    admit_recipe,
)
from .contracts import RiskEstimatorAdapter, RiskEstimatorRecipeEnvelope
from .domains import RiskParameterDomain, installed_parameter_domain


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class RiskEstimatorCapabilityIdentity(_Contract):
    """Bind an installed Risk adapter to its recipe schema and numerical identity."""

    adapter_id: str = Field(min_length=1, max_length=96)
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class RiskEstimatorCatalogBinding(_Contract):
    """Content identity of the explicitly installed Host estimator capabilities."""

    ordered_capabilities: tuple[RiskEstimatorCapabilityIdentity, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> RiskEstimatorCatalogBinding:
        """Require unique installed adapters and an exact ordered catalog identity.

        Returns:
            This binding after adapter-axis and catalog_hash validation.

        Raises:
            ValueError: Adapter identifiers repeat or catalog_hash differs from the declared ordered
                capabilities.
        """
        keys = tuple(value.adapter_id for value in self.ordered_capabilities)
        if len(set(keys)) != len(keys):
            raise ValueError("RISK_ESTIMATOR_CATALOG_CAPABILITY_DUPLICATED")
        identity = self.model_dump(mode="json", exclude={"catalog_hash"})
        if self.catalog_hash != canonical_hash(identity):
            raise ValueError("RISK_ESTIMATOR_CATALOG_IDENTITY_INVALID")
        return self


class RiskEstimatorCatalog:
    """Immutable installed-capability index resolved by adapter and recipe schema."""

    def __init__(
        self,
        adapters: tuple[RiskEstimatorAdapter, ...],
        *,
        capabilities: tuple[RiskCapability, ...] = (),
    ) -> None:
        """Construct a nonempty immutable catalog of unique Risk routes and capabilities.

        Args:
            adapters: Numerical owners in registration order; adapter/schema identifiers must be
                unique.
            capabilities: Optional declared capabilities with unique handles and matching installed
                routes.

        Raises:
            ValueError: The catalog is empty, a route/handle repeats or a capability cannot be
                admitted against its adapter.
        """
        indexed = {value.adapter_id: value for value in adapters}
        by_schema = {value.recipe_schema_id: value for value in adapters}
        if not adapters or len(indexed) != len(adapters) or len(by_schema) != len(adapters):
            raise ValueError("RISK_ESTIMATOR_CATALOG_INVALID")
        self._adapters = MappingProxyType(indexed)
        self._by_schema = MappingProxyType(by_schema)
        by_handle = {value.capability_handle: value for value in capabilities}
        if len(by_handle) != len(capabilities):
            raise ValueError("RISK_ESTIMATOR_CAPABILITY_DUPLICATED")
        for capability in capabilities:
            self._admit_capability_route(capability)
        self._capabilities = MappingProxyType(by_handle)

    def _admit_capability_route(self, capability: RiskCapability) -> None:
        """Prove the adapter that seals a Program is the adapter that will run.

        Admission reads identity from ``capability.adapter``; execution resolves
        an adapter out of ``self._adapters`` by id. Only "an adapter with this id
        is installed" was checked, so a capability could seal a Program under
        implementation A's numerical binding while implementation B -- same id,
        different code -- computed the numbers. Nothing downstream could notice:
        every hash on the Program would be internally consistent and describe a
        computation that never happened.

        The comparison is by declared identity, not by object identity. A
        capability that constructs its own instance of the very same
        implementation is the normal case and stays legal; what is refused is two
        different implementations wearing one id.
        """

        adapter = capability.adapter
        if capability.capability_handle != adapter.recipe_schema_id:
            # The handle an author writes must be the schema the adapter decodes,
            # or the domain admitted under one schema seals a recipe for another.
            raise ValueError("RISK_ESTIMATOR_CAPABILITY_HANDLE_NOT_ITS_SCHEMA")
        installed = self._adapters.get(adapter.adapter_id)
        if installed is None:
            raise ValueError("RISK_ESTIMATOR_CAPABILITY_ADAPTER_NOT_INSTALLED")
        if installed.recipe_schema_id != adapter.recipe_schema_id:
            raise ValueError("RISK_ESTIMATOR_CAPABILITY_SCHEMA_NOT_INSTALLED")
        if (
            installed.describe_numerical_binding().numerical_binding_hash
            != adapter.describe_numerical_binding().numerical_binding_hash
        ):
            raise ValueError("RISK_ESTIMATOR_CAPABILITY_BINDING_NOT_INSTALLED")

    def admit_recipe(
        self,
        *,
        capability_handle: str,
        parameters: Mapping[str, object],
        seed: int,
    ) -> RiskRecipeAdmission:
        """The compiler's only route from authored parameters to a sealed recipe.

        Typed, and schema-neutral by construction: the caller supplies a handle
        and a mapping and receives identities. Nothing here lets a caller name a
        recipe contract, an adapter, or a family -- which is what let the
        compiler acquire knowledge of the covariance method in the first place.
        """
        capability = self._capabilities.get(capability_handle)
        if capability is None:
            raise RiskCapabilityError("risk_research.authoring_capability_not_installed")
        # Determinism first: a seed this capability cannot consume is refused
        # before any parameter is admitted, so the failure names the real cause.
        admit_determinism(capability, seed=seed)
        return admit_recipe(capability, parameters=parameters)

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        """Return installed adapter identifiers in registration order.

        Returns:
            Ordered adapter identifiers from the immutable catalog.
        """
        return tuple(self._adapters)

    @property
    def capability_handles(self) -> tuple[str, ...]:
        """The handles an author may write, in installation order."""
        return tuple(self._capabilities)

    @property
    def adapters(self) -> tuple[RiskEstimatorAdapter, ...]:
        """The installed adapters in registration order.

        Exposed so a wrapper can rebuild an equivalent catalog without reaching
        into private state; the underlying mapping stays immutable.
        """
        return tuple(self._adapters.values())

    @property
    def binding(self) -> RiskEstimatorCatalogBinding:
        """Seal installed adapter, recipe-schema and numerical identities in registration order.

        Returns:
            Validated ordered capability catalog and its canonical catalog_hash.
        """
        values = tuple(
            RiskEstimatorCapabilityIdentity(
                adapter_id=adapter.adapter_id,
                recipe_schema_id=adapter.recipe_schema_id,
                numerical_binding_hash=adapter.describe_numerical_binding().numerical_binding_hash,
            )
            for adapter in self._adapters.values()
        )
        identity = {"ordered_capabilities": [value.model_dump(mode="json") for value in values]}
        return RiskEstimatorCatalogBinding(
            ordered_capabilities=values,
            catalog_hash=canonical_hash(identity),
        )

    def parameter_domain(self, recipe_schema_id: str) -> RiskParameterDomain:
        """Resolve the declared admissible parameter space of an installed schema.

        The domain lives beside the estimators rather than on the adapter,
        because the adapter modules are inside the Risk source closure and every
        published covariance identity moves when one of them is edited.
        """
        if recipe_schema_id not in self._by_schema:
            raise ValueError("RISK_ESTIMATOR_RECIPE_SCHEMA_NOT_INSTALLED")
        return installed_parameter_domain(recipe_schema_id)

    def seal_recipe(
        self,
        *,
        recipe_schema_id: str,
        parameters: Mapping[str, object],
    ) -> RiskEstimatorRecipeEnvelope:
        """Route one Desk-owned frozen recipe to its installed adapter and seal it."""
        adapter = self._by_schema.get(recipe_schema_id)
        if adapter is None:
            raise ValueError("RISK_ESTIMATOR_RECIPE_SCHEMA_NOT_INSTALLED")
        return RiskEstimatorRecipeEnvelope.create(
            adapter_id=adapter.adapter_id,
            recipe_schema_id=recipe_schema_id,
            parameters=parameters,
        )

    def resolve(self, recipe: RiskEstimatorRecipeEnvelope) -> RiskEstimatorAdapter:
        """Resolve a sealed recipe against its installed adapter and numerical route.

        Args:
            recipe: Sealed envelope selecting an adapter, recipe schema and parameters.

        Returns:
            Installed adapter after route, numerical-owner and recipe checks.

        Raises:
            ValueError: The adapter/schema is absent, its numerical route differs or recipe
                validation fails.
        """
        try:
            adapter = self._adapters[recipe.adapter_id]
        except KeyError as error:
            raise ValueError("RISK_ESTIMATOR_ADAPTER_NOT_INSTALLED") from error
        if adapter.recipe_schema_id != recipe.recipe_schema_id:
            raise ValueError("RISK_ESTIMATOR_RECIPE_SCHEMA_NOT_INSTALLED")
        binding = adapter.describe_numerical_binding()
        if binding.adapter_id != adapter.adapter_id:
            raise ValueError("RISK_ESTIMATOR_NUMERICAL_BINDING_ROUTE_INVALID")
        adapter.validate_recipe(recipe)
        return adapter


ESTIMATOR_CATALOG_ROLE = "risk_research.estimator_catalog"
"""The identity role of the installed catalog's hash (`config/identity-roles.json`)."""


def numerical_binding_role(adapter_id: str) -> str:
    """The identity role of one installed adapter's numerical binding."""
    return f"risk_research.estimator.{adapter_id}"


def installed_numerical_binding_hash(adapter_id: str) -> str:
    """The numerical binding this build installs for one adapter: its role's value."""
    for adapter in build_installed_risk_estimator_catalog().adapters:
        if adapter.adapter_id == adapter_id:
            return adapter.describe_numerical_binding().numerical_binding_hash
    raise ValueError("RISK_ESTIMATOR_ADAPTER_NOT_INSTALLED")


def build_installed_risk_estimator_catalog() -> RiskEstimatorCatalog:
    """Construct the explicit installed catalog; no discovery or runtime mutation.

    Two real capabilities: the production covariance control and the blended
    fast/slow correlation challenger. Both are named here, at one readable call
    site. Any further capability -- including every case-study one -- is injected
    explicitly by whoever wants it, so a test method can never become reachable
    from a production composition.

    Installing the challenger moves ``catalog_hash`` and therefore every
    development Program identity, which is correct: the Host installed something
    new. It does not move ``selected_method_binding_hash`` for the control, and
    it does not reach a published production surface at all -- those are bound by
    the frozen execution closure, which this file is outside of.
    """
    from .covariance import CovarianceCapability, CovarianceEstimatorAdapter
    from .diagonal import DiagonalShrunkCovarianceAdapter, DiagonalShrunkCovarianceCapability
    from .fast_slow import FastSlowCovarianceAdapter, FastSlowCovarianceCapability

    # Order is declaration order and it is inside ``catalog_hash``, so installing
    # R2 is an append. The two existing entries keep their positions: reordering
    # them would move a catalog identity that published development evidence
    # already names, for no reason anyone could point at.
    return RiskEstimatorCatalog(
        (
            CovarianceEstimatorAdapter(),
            FastSlowCovarianceAdapter(),
            DiagonalShrunkCovarianceAdapter(),
        ),
        capabilities=(
            CovarianceCapability(),
            FastSlowCovarianceCapability(),
            DiagonalShrunkCovarianceCapability(),
        ),
    )


__all__ = [
    "RiskEstimatorCapabilityIdentity",
    "RiskEstimatorCatalog",
    "RiskEstimatorCatalogBinding",
    "build_installed_risk_estimator_catalog",
]
