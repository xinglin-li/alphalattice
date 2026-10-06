"""Declared parameter domains for the installed Risk estimator recipes.

The catalog says which capability exists. A domain says which parameter values
that capability will accept, and carries its own identity so a development
Program can record the domain it was admitted against.

The axes below are **declared**, not read back from the default recipe.
Rejecting anything that differs from the current default is not a domain: it
cannot say what would be admissible, only what happens to be configured, and it
silently widens the moment a default moves. Declaring the axes means the
admissible set is a reviewable statement, and
``test_declared_domain_matches_the_recipe_contract`` is what stops the
declaration from drifting away from what ``CovarianceRecipe`` will actually
validate.

``ewma_decay`` now admits two values; every other axis is still a single point.
That is enough to make the authoring chain observable -- an authored recipe can
now differ from the default, so a builder that ignored it would produce visibly
different numbers -- but it is not a claim that arbitrary tuning works. The
admissible set is two named RiskMetrics constants, and widening any other axis
still requires an edit to ``risk_research/contracts.py``, which is inside the
Risk source closure and therefore explicitly identity-moving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from alphalattice.investment.risk_research.contracts import (
    ADMISSIBLE_EWMA_DECAY,
    DEFAULT_EWMA_DECAY,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class ParameterDomainError(ValueError):
    """A parameter was outside the declared admissible set."""


@dataclass(frozen=True, slots=True)
class ParameterAxis:
    """One named axis and the complete set of values admitted along it."""

    name: str
    admissible: tuple[object, ...]
    default: object

    def __post_init__(self) -> None:
        """Require a named nonempty parameter axis whose default is an admitted value.

        Raises:
            ValueError: The name/values are empty or the default is not a declared member.
        """
        if not self.name:
            raise ValueError("risk_research.parameter_axis_unnamed")
        if not self.admissible:
            raise ValueError("risk_research.parameter_axis_empty")
        if self.default not in self.admissible:
            raise ValueError("risk_research.parameter_axis_default_not_admissible")

    def admit(self, value: object) -> object:
        # Membership by equality, so a bool never satisfies an int axis by
        # arithmetic accident.
        """Admit one declared axis value by exact Python type and value.

        Args:
            value: Requested parameter value; boolean and integer values are distinct.

        Returns:
            The matching declared member, preserving its exact type.

        Raises:
            ParameterDomainError: No declared member has both the requested type and value.
        """
        for candidate in self.admissible:
            if type(candidate) is type(value) and candidate == value:
                return candidate
        raise ParameterDomainError("risk_research.parameter_value_not_admitted")


@dataclass(frozen=True, slots=True)
class RiskParameterDomain:
    """The admissible parameter space of one installed recipe schema."""

    recipe_schema_id: str
    axes: tuple[ParameterAxis, ...]

    def __post_init__(self) -> None:
        """Require nonempty unique named axes for the declared Risk parameter domain.

        Raises:
            ValueError: The domain has no axes or an axis name repeats.
        """
        names = tuple(axis.name for axis in self.axes)
        if not names or len(set(names)) != len(names):
            raise ValueError("risk_research.parameter_domain_invalid")

    @property
    def domain_hash(self) -> str:
        """Hash the recipe schema and ordered parameter axes with all values and defaults.

        Returns:
            Canonical domain identity binding the complete declared search boundary.
        """
        return str(
            canonical_hash(
                {
                    "kind": "RiskParameterDomain",
                    "recipe_schema_id": self.recipe_schema_id,
                    "axes": [
                        {
                            "name": axis.name,
                            "admissible": list(axis.admissible),
                            "default": axis.default,
                        }
                        for axis in self.axes
                    ],
                }
            )
        )

    def admit(self, parameters: Mapping[str, object]) -> dict[str, object]:
        """Return the complete admitted parameter set, or fail closed.

        An axis the author did not mention takes its declared default, so an
        author states only what they are choosing rather than restating the
        whole recipe.
        """
        by_name = {axis.name: axis for axis in self.axes}
        for name in parameters:
            if name not in by_name:
                raise ParameterDomainError("risk_research.parameter_axis_not_declared")
        return {
            axis.name: (
                axis.admit(parameters[axis.name]) if axis.name in parameters else axis.default
            )
            for axis in self.axes
        }


COVARIANCE_RECIPE_SCHEMA_ID = "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION"

COVARIANCE_PARAMETER_DOMAIN = RiskParameterDomain(
    recipe_schema_id=COVARIANCE_RECIPE_SCHEMA_ID,
    axes=(
        ParameterAxis("recipe_name", (COVARIANCE_RECIPE_SCHEMA_ID,), COVARIANCE_RECIPE_SCHEMA_ID),
        ParameterAxis("initialization_sessions", (63,), 63),
        ParameterAxis("standardized_residual_sessions", (252,), 252),
        # The one axis with a real choice, and the only reason the authoring
        # chain is observable: with every axis a single point, an authored recipe
        # was byte-identical to the default, so a builder that ignored it
        # produced identical numbers. Declared from the contract's own admissible
        # set rather than restated, so the two cannot drift apart.
        ParameterAxis("ewma_decay", ADMISSIBLE_EWMA_DECAY, DEFAULT_EWMA_DECAY),
        ParameterAxis("ledoit_wolf_store_precision", (False,), False),
        ParameterAxis("ledoit_wolf_assume_centered", (False,), False),
        ParameterAxis(
            "output_unit",
            ("one-session-open-to-open-log-return-covariance",),
            "one-session-open-to-open-log-return-covariance",
        ),
    ),
)

FAST_SLOW_RECIPE_SCHEMA_ID = "FAST_SLOW_LEDOIT_WOLF_CORRELATION"

ADMISSIBLE_BLEND_WEIGHT: tuple[float, ...] = (0.25, 0.50, 0.75)
"""The blend axis, declared here beside the domain that admits it.

Stated as a constant in this module rather than imported from the estimator:
``fast_slow.py`` imports this domain, so reading the constant back out of the
estimator would close the loop and make the declared admissible set depend on
the implementation it is supposed to bound.
"""

FAST_SLOW_PARAMETER_DOMAIN = RiskParameterDomain(
    recipe_schema_id=FAST_SLOW_RECIPE_SCHEMA_ID,
    axes=(
        ParameterAxis("recipe_name", (FAST_SLOW_RECIPE_SCHEMA_ID,), FAST_SLOW_RECIPE_SCHEMA_ID),
        ParameterAxis("initialization_sessions", (63,), 63),
        ParameterAxis("standardized_residual_sessions", (252,), 252),
        # Held to the control's admissible set and its default, so the two
        # methods differ in correlation lookback and in nothing else. A decay
        # this method admitted and the control did not would confound the very
        # comparison it exists for.
        ParameterAxis("ewma_decay", ADMISSIBLE_EWMA_DECAY, DEFAULT_EWMA_DECAY),
        ParameterAxis("slow_correlation_sessions", (252,), 252),
        ParameterAxis("fast_correlation_sessions", (63,), 63),
        # The one axis with a real choice: three declared blend weights.
        ParameterAxis("blend_weight", ADMISSIBLE_BLEND_WEIGHT, ADMISSIBLE_BLEND_WEIGHT[1]),
        ParameterAxis("ledoit_wolf_store_precision", (False,), False),
        ParameterAxis("ledoit_wolf_assume_centered", (False,), False),
        ParameterAxis(
            "output_unit",
            ("one-session-open-to-open-log-return-covariance",),
            "one-session-open-to-open-log-return-covariance",
        ),
    ),
)

DIAGONAL_RECIPE_SCHEMA_ID = "DIAGONAL_SHRUNK_CORRELATION"

ADMISSIBLE_CORRELATION_SHRINKAGE: tuple[float, ...] = (0.0, 1.0)
"""The identity-shrink axis: the two endpoints, declared beside the domain.

Stated here rather than imported from the estimator, for the same reason the
blend axis is: ``diagonal.py`` imports this domain, so reading the constant back
out of the estimator would close the loop and make the declared admissible set
depend on the implementation it bounds.

The endpoints are the whole method. R1 excludes its endpoints because each of
them would be a different estimator under one schema; these two are the same
formula at the boundary of its own axis, and the contrast between them is the
experiment -- ``0.0`` reproduces the control bit for bit, ``1.0`` deletes every
off-diagonal, and a Portfolio that cannot tell them apart has answered whether
correlation structure reaches the optimizer at all.
"""

DIAGONAL_PARAMETER_DOMAIN = RiskParameterDomain(
    recipe_schema_id=DIAGONAL_RECIPE_SCHEMA_ID,
    axes=(
        ParameterAxis("recipe_name", (DIAGONAL_RECIPE_SCHEMA_ID,), DIAGONAL_RECIPE_SCHEMA_ID),
        ParameterAxis("initialization_sessions", (63,), 63),
        ParameterAxis("standardized_residual_sessions", (252,), 252),
        # Held to the control's admissible set and its default, so the three
        # installed methods differ in correlation treatment and in nothing else.
        ParameterAxis("ewma_decay", ADMISSIBLE_EWMA_DECAY, DEFAULT_EWMA_DECAY),
        ParameterAxis(
            "correlation_shrinkage",
            ADMISSIBLE_CORRELATION_SHRINKAGE,
            ADMISSIBLE_CORRELATION_SHRINKAGE[0],
        ),
        ParameterAxis("ledoit_wolf_store_precision", (False,), False),
        ParameterAxis("ledoit_wolf_assume_centered", (False,), False),
        ParameterAxis(
            "output_unit",
            ("one-session-open-to-open-log-return-covariance",),
            "one-session-open-to-open-log-return-covariance",
        ),
    ),
)

_INSTALLED_DOMAINS: Mapping[str, RiskParameterDomain] = MappingProxyType(
    {
        COVARIANCE_RECIPE_SCHEMA_ID: COVARIANCE_PARAMETER_DOMAIN,
        FAST_SLOW_RECIPE_SCHEMA_ID: FAST_SLOW_PARAMETER_DOMAIN,
        DIAGONAL_RECIPE_SCHEMA_ID: DIAGONAL_PARAMETER_DOMAIN,
    }
)


def installed_parameter_domain(recipe_schema_id: str) -> RiskParameterDomain:
    """Resolve the declared domain of an installed recipe schema."""
    domain = _INSTALLED_DOMAINS.get(recipe_schema_id)
    if domain is None:
        raise ParameterDomainError("risk_research.parameter_domain_not_installed")
    return domain


__all__ = [
    "ADMISSIBLE_BLEND_WEIGHT",
    "ADMISSIBLE_CORRELATION_SHRINKAGE",
    "COVARIANCE_PARAMETER_DOMAIN",
    "COVARIANCE_RECIPE_SCHEMA_ID",
    "DIAGONAL_PARAMETER_DOMAIN",
    "DIAGONAL_RECIPE_SCHEMA_ID",
    "FAST_SLOW_PARAMETER_DOMAIN",
    "FAST_SLOW_RECIPE_SCHEMA_ID",
    "ParameterAxis",
    "ParameterDomainError",
    "RiskParameterDomain",
    "installed_parameter_domain",
]
