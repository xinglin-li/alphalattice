"""The declared recipe domain and its contract.

The declared domain matches the recipe contract, is narrow beyond the one
widened axis, admits a second value on that axis without touching the
mechanism, and never lets a bool satisfy a numeric axis by arithmetic accident.
"""

from __future__ import annotations

import pytest

from alphalattice.investment.risk_research.contracts import (
    CovarianceRecipe,
    default_covariance_recipe,
    seal_contract,
)
from alphalattice.investment.risk_research.estimators.domains import (
    COVARIANCE_PARAMETER_DOMAIN,
    ParameterAxis,
    ParameterDomainError,
    RiskParameterDomain,
)


def test_declared_domain_matches_the_recipe_contract() -> None:
    """Declared domain matches the recipe contract."""

    admitted = COVARIANCE_PARAMETER_DOMAIN.admit({})
    sealed = seal_contract(CovarianceRecipe, "recipe_hash", **admitted)

    assert sealed == default_covariance_recipe()
    declared = {axis.name for axis in COVARIANCE_PARAMETER_DOMAIN.axes}
    contract = set(default_covariance_recipe().model_dump(mode="json")) - {"kind", "recipe_hash"}
    assert declared == contract


def test_the_domain_is_declared_and_narrow_beyond_the_one_widened_axis() -> None:
    """The domain is declared and narrow beyond the one widened axis."""

    widened = {axis.name for axis in COVARIANCE_PARAMETER_DOMAIN.axes if len(axis.admissible) > 1}
    assert widened == {"ewma_decay"}
    # Admissible is a declared set, not "anything but the default": a plausible
    # third decay is still refused.
    with pytest.raises(ParameterDomainError, match="parameter_value_not_admitted"):
        COVARIANCE_PARAMETER_DOMAIN.admit({"ewma_decay": 0.95})


def test_a_wider_axis_admits_a_second_value_without_touching_the_mechanism() -> None:
    """The domain mechanism is general and independent of the installed axes."""

    widened = RiskParameterDomain(
        recipe_schema_id="fixture.schema",
        axes=(ParameterAxis("ewma_decay", (0.94, 0.97), 0.94),),
    )

    assert widened.admit({})["ewma_decay"] == 0.94
    assert widened.admit({"ewma_decay": 0.97})["ewma_decay"] == 0.97
    assert widened.domain_hash != COVARIANCE_PARAMETER_DOMAIN.domain_hash


def test_a_bool_never_satisfies_a_numeric_axis_by_arithmetic_accident() -> None:
    axis = ParameterAxis("initialization_sessions", (63, 1), 63)

    assert axis.admit(1) == 1
    with pytest.raises(ParameterDomainError):
        axis.admit(True)
