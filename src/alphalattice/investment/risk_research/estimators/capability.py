"""One installed Risk capability: adapter, domain, sealing, and identity.

The compiler used to know the covariance method by name. It imported
``CovarianceRecipe`` to seal, ``COVARIANCE_RECIPE_SCHEMA_ID`` to route, and
reached through the catalog into a central ``_INSTALLED_DOMAINS`` table to find
a parameter domain. So "add a method" was never a registration -- it was an edit
to the compiler, and a second recipe schema could not enter the chain at all.

A capability owns those five things together, because they are five statements
about one method and keeping them apart is what let them disagree:

``adapter``                the implementation
``recipe_schema_id``       what the author names
``parameter_domain``       which values are admissible
``seal``                   how admitted values become a sealed, typed recipe
``numerical binding``      the identity of the code that will run

Installation stays explicit. There is no filesystem discovery, no entry points,
no dynamic import, and no plugin manager: the Host names what it installs, and
the set is readable at the call site.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .contracts import RiskEstimatorAdapter, RiskEstimatorRecipeEnvelope
from .domains import ParameterDomainError, RiskParameterDomain

CANONICAL_NO_RANDOMNESS_SEED = 0
"""The only seed a ``randomness_policy = NONE`` capability will accept.

The generic envelope carries a seed because some Desk somewhere will need one.
Neither Risk capability does: both are deterministic, and neither reads it. But
``seed`` was folded into ``envelope_hash`` and therefore into ``program_hash``,
so two documents differing only in seed produced two Programs, two identities and
two evidence records over byte-identical numbers.

That is worse than untidy. It makes "different Program" stop meaning "different
computation", which is the property every reuse and admission decision here
depends on. Rather than invent a seed parameter no Risk estimator consumes --
which would make the difference real by making the mathematics worse -- the Host
admits exactly one canonical representation and refuses the rest.

A capability that genuinely consumes randomness declares
``randomness_policy = "SEEDED"`` and takes the seed through its own recipe, where
it belongs. The generic executor never guesses.
"""

RANDOMNESS_NONE = "NONE"
RANDOMNESS_SEEDED = "SEEDED"


class RiskCapabilityError(ValueError):
    """A capability handle, parameter, or sealed recipe was inadmissible."""


@dataclass(frozen=True, slots=True)
class RiskRecipeAdmission:
    """One admitted recipe, sealed, with every identity the Program consumes.

    ``sealed_recipe`` is the schema's own typed object. It is deliberately typed
    as ``object`` here: the compiler must not know which schema it received, and
    the executor that will run it is the only component entitled to narrow it.
    """

    capability_handle: str
    adapter_id: str
    recipe: RiskEstimatorRecipeEnvelope
    sealed_recipe: object
    recipe_hash: str
    parameter_domain_hash: str
    selected_numerical_binding_hash: str


class RiskCapability(Protocol):
    """What a Risk method must provide to be installable."""

    capability_handle: str
    """The handle an author writes. Equal to the adapter's recipe schema id."""

    randomness_policy: str
    """``NONE`` or ``SEEDED``. Declared by the method, never inferred.

    A capability that declares ``NONE`` is stating that its output depends on no
    random draw, which is what lets the Host refuse a seed that would otherwise
    change identity without changing a single number.
    """

    @property
    def adapter(self) -> RiskEstimatorAdapter:
        """Return the numerical adapter owned by this admitted Risk capability.

        Returns:
            Adapter exposing this capability method and recipe-schema route.
        """
        ...

    @property
    def parameter_domain(self) -> RiskParameterDomain:
        """Return the complete declared parameter domain for this capability.

        Returns:
            Finite named axes and their default values; this declaration grants no undeclared
            tuning.
        """
        ...

    def seal(self, admitted: Mapping[str, object]) -> tuple[object, str]:
        """Turn admitted parameters into this schema's typed recipe and its hash.

        The capability owns this because the recipe contract is the schema's,
        not the Host's. Returning the hash alongside the object keeps the Host
        from having to know which field carries identity.
        """


def admit_determinism(capability: RiskCapability, *, seed: int) -> None:
    """Refuse a seed the selected capability cannot possibly consume.

    Fails closed rather than ignoring the value: silently accepting a seed that
    nothing reads is exactly how two Programs came to describe one computation.
    """
    if capability.randomness_policy == RANDOMNESS_NONE:
        if seed != CANONICAL_NO_RANDOMNESS_SEED:
            raise RiskCapabilityError("risk_research.authoring_seed_not_applicable")
        return
    if capability.randomness_policy != RANDOMNESS_SEEDED:
        raise RiskCapabilityError("risk_research.capability_randomness_policy_invalid")


def admit_recipe(
    capability: RiskCapability,
    *,
    parameters: Mapping[str, object],
) -> RiskRecipeAdmission:
    """Admit authored parameters through one capability, or fail closed.

    The whole route in one place: declared domain, then the schema's own seal,
    then the envelope the adapter will actually receive, then the adapter's own
    validation. Every rejection happens here, before any numerical call.
    """
    try:
        admitted = capability.parameter_domain.admit(parameters)
    except ParameterDomainError as error:
        raise RiskCapabilityError(str(error)) from error
    try:
        sealed_recipe, recipe_hash = capability.seal(admitted)
    except (ValueError, TypeError) as error:
        raise RiskCapabilityError("risk_research.authoring_parameter_not_sealable") from error

    adapter = capability.adapter
    envelope = RiskEstimatorRecipeEnvelope.create(
        adapter_id=adapter.adapter_id,
        recipe_schema_id=capability.capability_handle,
        parameters=_envelope_parameters(sealed_recipe, admitted),
    )
    # The adapter validates its own recipe before anything is sealed into a
    # Program, so an envelope this adapter would reject at estimate time is
    # rejected at admission time instead.
    try:
        adapter.validate_recipe(envelope)
    except ValueError as error:
        raise RiskCapabilityError("risk_research.authoring_recipe_rejected_by_adapter") from error

    binding = adapter.describe_numerical_binding()
    if binding.adapter_id != adapter.adapter_id:
        raise RiskCapabilityError("risk_research.numerical_binding_route_invalid")
    return RiskRecipeAdmission(
        capability_handle=capability.capability_handle,
        adapter_id=adapter.adapter_id,
        recipe=envelope,
        sealed_recipe=sealed_recipe,
        recipe_hash=recipe_hash,
        parameter_domain_hash=capability.parameter_domain.domain_hash,
        selected_numerical_binding_hash=binding.numerical_binding_hash,
    )


def _envelope_parameters(
    sealed_recipe: object, admitted: Mapping[str, object]
) -> Mapping[str, object]:
    """The adapter-visible parameter payload.

    A schema whose sealed object is a pydantic model carries its own canonical
    projection, which is what the adapter re-validates. Anything else falls back
    to the admitted values, so a capability is not forced to adopt pydantic to
    be installable.
    """

    dump = getattr(sealed_recipe, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json")
        if isinstance(payload, dict):
            return payload
    return dict(admitted)


__all__ = [
    "CANONICAL_NO_RANDOMNESS_SEED",
    "RANDOMNESS_NONE",
    "RANDOMNESS_SEEDED",
    "RiskCapability",
    "RiskCapabilityError",
    "RiskRecipeAdmission",
    "admit_determinism",
    "admit_recipe",
]
