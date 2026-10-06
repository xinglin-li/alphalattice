"""Setup shared by the Risk development program suites that were one module.

The helpers two or more of those suites reach, moved here once when the
2,900-line module was split by lifecycle; nothing here is product authority
or evidence.
"""

from __future__ import annotations

from alphalattice.investment.risk_research.estimators.capability import (
    CANONICAL_NO_RANDOMNESS_SEED,
    RiskRecipeAdmission,
)
from alphalattice.investment.risk_research.estimators.catalog import (
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.covariance import (
    COVARIANCE_RECIPE_SCHEMA_ID,
)


def _admission(**parameters: object) -> RiskRecipeAdmission:
    """Admit through the real installed capability, exactly as the compiler does."""

    return build_installed_risk_estimator_catalog().admit_recipe(
        capability_handle=COVARIANCE_RECIPE_SCHEMA_ID,
        parameters=parameters,
        seed=CANONICAL_NO_RANDOMNESS_SEED,
    )
