"""Compile a dispersion forecast from sealed canonical target evidence.

The executor's real job is refusal. Producing the numbers is one indexing step;
what earns the module is everything it checks before doing it, because the
failure this owner exists to prevent is a *same-shaped* one. A scale surface
built from the wrong horizon, shifted by a session, or taken from a different
residual lane has the identical dtype, length and plausible magnitude as the
correct one, and silently produces return signals that are wrong by a factor
nobody can see.

So every check happens before the arithmetic, against identities carried in the
evidence rather than against anything the caller asserts.
"""

from __future__ import annotations

from datetime import date

from alphalattice.investment.alpha_research.targets.canonical import (
    CanonicalAlphaTargetEvidence,
    CanonicalAlphaTargetRecipeBinding,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .catalog import CrossSectionalScaleCatalog, build_installed_cross_sectional_scale_catalog
from .contracts import (
    LAGGED_XS_DISPERSION_RECIPE_ID,
    CrossSectionalDispersionForecast,
    CrossSectionalScalingError,
)


def compile_cross_sectional_dispersion_forecast(
    *,
    target_evidence: CanonicalAlphaTargetEvidence,
    target_recipe_binding: CanonicalAlphaTargetRecipeBinding,
    outcome_method_binding_hash: str,
    maturity_lag_sessions: int,
    recipe_id: str = LAGGED_XS_DISPERSION_RECIPE_ID,
    symmetric_half_life_sessions: int = 21,
    catalog: CrossSectionalScaleCatalog | None = None,
) -> CrossSectionalDispersionForecast:
    """Forecast the common cross-sectional scale for every formation on the axis.

    ``maturity_lag_sessions`` is passed in but not trusted: it must equal the lag
    the sealed target binding already carries, which in turn came from the
    outcome recipe. The argument exists so a caller states its intent and gets
    contradicted, rather than having the value silently taken from elsewhere.
    """
    installed = catalog or build_installed_cross_sectional_scale_catalog()
    adapter = installed.resolve(recipe_id)

    # --- authority, before any arithmetic ---------------------------------------------
    if target_evidence.recipe_binding_hash != target_recipe_binding.binding_hash:
        raise CrossSectionalScalingError("SCALING_TARGET_EVIDENCE_BINDING_MISMATCH")
    if target_recipe_binding.outcome_method_binding_hash != outcome_method_binding_hash:
        raise CrossSectionalScalingError("SCALING_OUTCOME_METHOD_MISMATCH")
    if target_recipe_binding.maturity_lag_sessions != maturity_lag_sessions:
        raise CrossSectionalScalingError("SCALING_MATURITY_LAG_MISMATCH")
    if target_recipe_binding.standardization_id != "CROSS_SECTIONAL_STD_Z":
        # A robust-Z or rank-Gauss surface has a scale, but not one that inverts
        # its own standardization. Refusing here is what stops a legacy lane
        # being rescaled and presented as a return.
        raise CrossSectionalScalingError("SCALING_TARGET_LANE_NOT_CANONICAL")

    recipe = installed.seal_recipe(
        recipe_id=recipe_id,
        maturity_lag_sessions=target_recipe_binding.maturity_lag_sessions,
        symmetric_half_life_sessions=symmetric_half_life_sessions,
    )
    if recipe.source_offset_sessions != maturity_lag_sessions:
        raise CrossSectionalScalingError("SCALING_MATURITY_LAG_MISMATCH")

    sessions = target_evidence.formation_sessions
    realized = target_evidence.cross_sectional_dispersion
    if len(sessions) != len(realized):
        raise CrossSectionalScalingError("SCALING_SOURCE_AXIS_MISMATCH")

    # --- the arithmetic ----------------------------------------------------------------
    values: list[float | None] = []
    source_sessions: list[date | None] = []
    for index in range(len(sessions)):
        value, source_index = adapter.select(recipe=recipe, realized=realized, index=index)
        values.append(value)
        source_sessions.append(None if source_index is None else sessions[source_index])

    implementation = adapter.describe_implementation_binding()
    payload: dict[str, object] = {
        "kind": "CrossSectionalDispersionForecast",
        "recipe_id": recipe.recipe_id,
        "recipe_hash": recipe.recipe_hash,
        "implementation_binding_hash": implementation.implementation_binding_hash,
        "implementation": implementation.model_dump(mode="json"),
        "target_evidence_hash": target_evidence.evidence_hash,
        "target_recipe_binding_hash": target_recipe_binding.binding_hash,
        "outcome_method_binding_hash": outcome_method_binding_hash,
        "maturity_lag_sessions": int(maturity_lag_sessions),
        "ordered_listing_ids_hash": target_evidence.ordered_listing_ids_hash,
        "source_dispersion_identity": target_evidence.cross_sectional_dispersion_identity,
        "formation_sessions": [value.isoformat() for value in sessions],
        "forecast_values": list(values),
        "source_sessions": [None if v is None else v.isoformat() for v in source_sessions],
    }
    return CrossSectionalDispersionForecast(
        recipe_id=recipe.recipe_id,
        recipe_hash=recipe.recipe_hash,
        implementation_binding_hash=implementation.implementation_binding_hash,
        implementation=implementation,
        target_evidence_hash=target_evidence.evidence_hash,
        target_recipe_binding_hash=target_recipe_binding.binding_hash,
        outcome_method_binding_hash=outcome_method_binding_hash,
        maturity_lag_sessions=int(maturity_lag_sessions),
        ordered_listing_ids_hash=target_evidence.ordered_listing_ids_hash,
        source_dispersion_identity=target_evidence.cross_sectional_dispersion_identity,
        formation_sessions=sessions,
        forecast_values=tuple(values),
        source_sessions=tuple(source_sessions),
        forecast_hash=canonical_hash(payload),
    )


__all__ = ["compile_cross_sectional_dispersion_forecast"]
