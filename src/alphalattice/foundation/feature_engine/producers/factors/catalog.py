"""The explicit installed Factor composition, and nothing that shapes a value.

This module states which kernels this build installs and which typed recipes it
can offer. It computes nothing and it validates nothing: the protocol, the
registry mechanics and the identity formulas live in ``registry``, which every
extension family's measured closure contains.

The separation is load-bearing rather than tidy. While composition and mechanics
shared one module, that module was inside every family's source closure, so
registering a *second* family rotated the implementation identity of an
unchanged *first* one. "Add a method without invalidating unrelated methods" is
the promise the per-family closure exists to keep, and a closure containing the
list of installed families cannot keep it. Installing something new now moves
``FeatureKernelRegistry.installed_capability_hash`` -- which is the honest place
for it -- and reaches no other factor's identity.

Installation is a list, not a search. There is no filesystem discovery, no entry
point group, no plugin manager and no dynamic import: a kernel exists because
this module names it, so the set of Factors a build can compute is readable in
one place and reviewable in one diff. A scan would make a Factor's availability a
property of what happened to be on the path.
"""

from __future__ import annotations

from alphalattice.foundation.feature_engine.catalog.contracts import (
    DesktopCoreFeatureBundle,
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    installed_method_family_owners,
)
from alphalattice.foundation.feature_engine.producers.factors.absolute_state import (
    ABSOLUTE_MOMENTUM_20_ID,
    ABSOLUTE_STATE_DECLARATIONS,
    ABSOLUTE_STATE_METHOD_FAMILY,
    DRAWDOWN_FROM_63D_HIGH_ID,
    PRICE_VS_SMA20_ATR_ID,
    PRICE_VS_SMA60_ATR_ID,
    absolute_momentum_20,
    absolute_state_factor_specs,
    drawdown_from_63d_high,
    price_vs_sma20_atr,
    price_vs_sma60_atr,
)
from alphalattice.foundation.feature_engine.producers.factors.downside_tail import (
    BETA_ASYMMETRY_ID,
    DOWN_DAY_ABSORPTION_ID,
    DOWNSIDE_TAIL_DECLARATIONS,
    DOWNSIDE_TAIL_METHOD_FAMILY,
    RECOVERY_RATIO_ID,
    TAIL_RESILIENCE_ID,
    beta_asymmetry_252,
    down_day_absorption_63,
    downside_tail_factor_specs,
    recovery_ratio_252,
    tail_resilience_252,
)
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    FORMULA_DECLARATION,
    FORMULA_ID,
    FORMULA_METHOD_FAMILY,
    FORMULA_POINT_IN_TIME_DECLARATION,
    FORMULA_POINT_IN_TIME_ID,
    FORMULA_POINT_IN_TIME_REQUIRED_FIELDS,
    FORMULA_REQUIRED_FIELDS,
    FORMULA_SECTOR_DECLARATION,
    FORMULA_SECTOR_ID,
    FORMULA_SECTOR_POINT_IN_TIME_DECLARATION,
    FORMULA_SECTOR_POINT_IN_TIME_ID,
    FORMULA_SECTOR_POINT_IN_TIME_REQUIRED_FIELDS,
    FORMULA_SECTOR_REQUIRED_FIELDS,
    compute_formula,
)
from alphalattice.foundation.feature_engine.producers.factors.interactions import (
    INTERACTION_DECLARATIONS,
    MARKET_DRAWDOWN_X_MOMENTUM_ID,
    MARKET_VOL_RATIO_X_REVERSAL_ID,
    STATE_INTERACTION_METHOD_FAMILY,
    interaction_factor_specs,
    market_drawdown_x_momentum_raw,
    market_vol_ratio_x_reversal_raw,
)
from alphalattice.foundation.feature_engine.producers.factors.leader_lag import (
    SECTOR_LEADER_LAG_5_ID,
    SECTOR_LEADER_LAG_DECLARATION,
    SECTOR_LEADER_LAG_METHOD_FAMILY,
    SECTOR_LEADER_LAG_REQUIRED_FIELDS,
    leader_lag_factor_specs,
    sector_leader_lag_5,
)
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    GAP_ABSORPTION_DECLARATION,
    GAP_ABSORPTION_ID,
    INTRADAY_REVERSAL_5_DECLARATION,
    INTRADAY_REVERSAL_5_ID,
    MEAN_ADJUSTED_RETURN_DECLARATION,
    MEAN_ADJUSTED_RETURN_ID,
    MEAN_ADJUSTED_RETURN_REQUIRED_FIELDS,
    OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS,
    OPEN_INTRADAY_METHOD_FAMILY,
    OVERNIGHT_INTRADAY_TUG_OF_WAR_DECLARATION,
    OVERNIGHT_INTRADAY_TUG_OF_WAR_ID,
    OVERNIGHT_RETURN_DECLARATION,
    OVERNIGHT_RETURN_ID,
    OVERNIGHT_RETURN_REQUIRED_FIELDS,
    OVERNIGHT_REVERSAL_5_DECLARATION,
    OVERNIGHT_REVERSAL_5_ID,
    gap_absorption,
    intraday_reversal_5,
    mean_adjusted_return,
    open_intraday_factor_specs,
    overnight_intraday_tug_of_war,
    overnight_return,
    overnight_reversal_5,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    FeatureKernelRegistry,
    RegisteredFeatureKernel,
    factor_methodology_hash,
)
from alphalattice.foundation.feature_engine.producers.factors.residual_reversal import (
    RESIDUAL_REVERSAL_DECLARATION,
    RESIDUAL_REVERSAL_METHOD_FAMILY,
    RESIDUAL_REVERSAL_REQUIRED_FIELDS,
    RESIDUAL_REVERSAL_VOL_SCALED_5_ID,
    residual_reversal_factor_specs,
    residual_reversal_vol_scaled_5,
)
from alphalattice.foundation.feature_engine.producers.factors.session_liquidity import (
    SESSION_DOLLAR_VOLUME_DECLARATION,
    SESSION_DOLLAR_VOLUME_ID,
    SESSION_DOLLAR_VOLUME_REQUIRED_FIELDS,
    SESSION_LIQUIDITY_METHOD_FAMILY,
    session_dollar_volume,
    session_liquidity_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.session_observation import (
    SESSION_OBSERVATION_DECLARATIONS,
    SESSION_OBSERVATION_KERNELS,
    SESSION_OBSERVATION_METHOD_FAMILY,
    session_observation_factor_specs,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def catalog_implementation_hashes(
    catalog: FeatureCatalog,
    *,
    registry: FeatureKernelRegistry,
    core_bundle: DesktopCoreFeatureBundle | None = None,
) -> dict[str, str]:
    """Resolve every catalog factor to the identity of the code that computes it.

    A catalog that binds only factor IDs lets an implementation change under a
    stable ID, so evidence produced by the old code still looks reusable. This
    is the one place that answers "which code computed this factor", and it is
    the value a published panel quotes.
    """
    bundle = core_bundle if core_bundle is not None else desktop_core_feature_bundle()
    return {
        spec.factor_id: registry.implementation_hash(spec, core_bundle=bundle)
        for spec in catalog.factors
    }


def catalog_methodology_hashes(
    catalog: FeatureCatalog,
    *,
    registry: FeatureKernelRegistry,
    core_bundle: DesktopCoreFeatureBundle | None = None,
) -> dict[str, str]:
    """Resolve every catalog factor to the identity of the *method* it installs.

    The companion to ``catalog_implementation_hashes``, and the stronger of the
    two: a factor rewritten to a different window under an unchanged id and an
    unchanged kernel moves this and not that one. Published per factor so a
    downstream Desk can bind the method it actually consumed.
    """
    bundle = core_bundle if core_bundle is not None else desktop_core_feature_bundle()
    return {
        spec.factor_id: factor_methodology_hash(
            spec,
            implementation_hash=registry.implementation_hash(spec, core_bundle=bundle),
        )
        for spec in catalog.factors
    }


def extension_factor_specs() -> tuple[FactorSpec, ...]:
    """Every typed recipe this build can install, in factor-id order.

    Assembled from the installed families rather than listed again here, so a new
    family is one module and one entry below rather than a second list somebody
    has to remember to update. Installation stays explicit and stays someone
    else's decision: nothing here reaches a panel until a catalog revision names
    its ``formula_ref``. The shipped desktop catalog names none of them.
    """
    return tuple(
        sorted(
            (
                *open_intraday_factor_specs(),
                *residual_reversal_factor_specs(),
                *absolute_state_factor_specs(),
                *interaction_factor_specs(),
                *leader_lag_factor_specs(),
                *downside_tail_factor_specs(),
                *session_observation_factor_specs(),
                *session_liquidity_factor_specs(),
            ),
            key=lambda specification: specification.factor_id,
        )
    )


def installed_formula_specs(
    base_catalog: FeatureCatalog | None = None,
) -> tuple[FactorSpec, ...]:
    """Logical installed Formula Registry: activated base plus inert extensions.

    Registration and base-Panel activation are independent.  This composition
    therefore validates identity coverage without adding extension columns to
    the base DuckDB or changing the base catalog identity.
    """
    base = base_catalog if base_catalog is not None else FeatureCatalog.load()
    extensions = extension_factor_specs()
    base_ids = set(base.factor_ids)
    extension_ids = {item.factor_id for item in extensions}
    if overlap := base_ids & extension_ids:
        raise ValueError(
            "feature_engine.installed_formula_registry_identity_overlap:"
            + ",".join(sorted(overlap))
        )
    installed = tuple(sorted((*base.factors, *extensions), key=lambda item: item.factor_id))
    if len(installed) != len(base_ids | extension_ids):
        raise ValueError("feature_engine.installed_formula_registry_identity_duplicate")
    return installed


def default_extension_kernel_registry() -> FeatureKernelRegistry:
    """Every extension kernel this build owns, keyed by the ref a catalog names.

    Installation is registration, not activation: a kernel here computes nothing
    until some catalog revision names its ``formula_ref``. The shipped desktop
    catalog names none of them, so they are inert in production and reachable
    only by a catalog that asks for them.

    Each declared summary is imported from the family that owns the formula
    rather than restated here. A summary written at the composition site is one
    that keeps describing the formula it described when it was written.
    """
    return FeatureKernelRegistry(installed_feature_kernels())


def installed_feature_kernels() -> tuple[RegisteredFeatureKernel, ...]:
    """The same code-owned installation for execution and research discovery."""
    return (
        RegisteredFeatureKernel(
            implementation_id=MEAN_ADJUSTED_RETURN_ID,
            method_family=OPEN_INTRADAY_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(OPEN_INTRADAY_METHOD_FAMILY),
            required_fields=MEAN_ADJUSTED_RETURN_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(MEAN_ADJUSTED_RETURN_DECLARATION)),
            compute=mean_adjusted_return,
        ),
        RegisteredFeatureKernel(
            implementation_id=OVERNIGHT_RETURN_ID,
            method_family=OPEN_INTRADAY_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(OPEN_INTRADAY_METHOD_FAMILY),
            required_fields=OVERNIGHT_RETURN_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(OVERNIGHT_RETURN_DECLARATION)),
            compute=overnight_return,
        ),
        RegisteredFeatureKernel(
            implementation_id=OVERNIGHT_INTRADAY_TUG_OF_WAR_ID,
            method_family=OPEN_INTRADAY_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(OPEN_INTRADAY_METHOD_FAMILY),
            required_fields=OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(OVERNIGHT_INTRADAY_TUG_OF_WAR_DECLARATION)),
            compute=overnight_intraday_tug_of_war,
        ),
        RegisteredFeatureKernel(
            implementation_id=OVERNIGHT_REVERSAL_5_ID,
            method_family=OPEN_INTRADAY_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(OPEN_INTRADAY_METHOD_FAMILY),
            required_fields=OVERNIGHT_RETURN_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(OVERNIGHT_REVERSAL_5_DECLARATION)),
            compute=overnight_reversal_5,
        ),
        RegisteredFeatureKernel(
            implementation_id=INTRADAY_REVERSAL_5_ID,
            method_family=OPEN_INTRADAY_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(OPEN_INTRADAY_METHOD_FAMILY),
            required_fields=OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(INTRADAY_REVERSAL_5_DECLARATION)),
            compute=intraday_reversal_5,
        ),
        RegisteredFeatureKernel(
            implementation_id=GAP_ABSORPTION_ID,
            method_family=OPEN_INTRADAY_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(OPEN_INTRADAY_METHOD_FAMILY),
            required_fields=OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(GAP_ABSORPTION_DECLARATION)),
            compute=gap_absorption,
        ),
        *(
            RegisteredFeatureKernel(
                implementation_id=implementation_id,
                method_family=ABSOLUTE_STATE_METHOD_FAMILY,
                method_family_owners=installed_method_family_owners(ABSOLUTE_STATE_METHOD_FAMILY),
                required_fields=next(
                    item.required_fields
                    for item in absolute_state_factor_specs()
                    if item.formula_ref == implementation_id
                ),
                implementation_hash=canonical_hash(
                    {
                        "implementation_id": implementation_id,
                        **ABSOLUTE_STATE_DECLARATIONS[implementation_id],
                    }
                ),
                compute=compute,
            )
            for implementation_id, compute in (
                (ABSOLUTE_MOMENTUM_20_ID, absolute_momentum_20),
                (DRAWDOWN_FROM_63D_HIGH_ID, drawdown_from_63d_high),
                (PRICE_VS_SMA20_ATR_ID, price_vs_sma20_atr),
                (PRICE_VS_SMA60_ATR_ID, price_vs_sma60_atr),
            )
        ),
        RegisteredFeatureKernel(
            implementation_id=RESIDUAL_REVERSAL_VOL_SCALED_5_ID,
            method_family=RESIDUAL_REVERSAL_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(RESIDUAL_REVERSAL_METHOD_FAMILY),
            required_fields=RESIDUAL_REVERSAL_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(RESIDUAL_REVERSAL_DECLARATION)),
            compute=residual_reversal_vol_scaled_5,
        ),
        *(
            RegisteredFeatureKernel(
                implementation_id=implementation_id,
                method_family=STATE_INTERACTION_METHOD_FAMILY,
                method_family_owners=installed_method_family_owners(
                    STATE_INTERACTION_METHOD_FAMILY
                ),
                required_fields=next(
                    item.required_fields
                    for item in interaction_factor_specs()
                    if item.formula_ref == implementation_id
                ),
                implementation_hash=canonical_hash(
                    {
                        "implementation_id": implementation_id,
                        **INTERACTION_DECLARATIONS[implementation_id],
                    }
                ),
                compute=compute,
            )
            for implementation_id, compute in (
                (MARKET_DRAWDOWN_X_MOMENTUM_ID, market_drawdown_x_momentum_raw),
                (MARKET_VOL_RATIO_X_REVERSAL_ID, market_vol_ratio_x_reversal_raw),
            )
        ),
        RegisteredFeatureKernel(
            implementation_id=SECTOR_LEADER_LAG_5_ID,
            method_family=SECTOR_LEADER_LAG_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(SECTOR_LEADER_LAG_METHOD_FAMILY),
            required_fields=SECTOR_LEADER_LAG_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(SECTOR_LEADER_LAG_DECLARATION)),
            compute=sector_leader_lag_5,
        ),
        *(
            RegisteredFeatureKernel(
                implementation_id=implementation_id,
                method_family=DOWNSIDE_TAIL_METHOD_FAMILY,
                method_family_owners=installed_method_family_owners(DOWNSIDE_TAIL_METHOD_FAMILY),
                required_fields=next(
                    item.required_fields
                    for item in downside_tail_factor_specs()
                    if item.formula_ref == implementation_id
                ),
                implementation_hash=canonical_hash(
                    {
                        "implementation_id": implementation_id,
                        **DOWNSIDE_TAIL_DECLARATIONS[implementation_id],
                    }
                ),
                compute=compute,
            )
            for implementation_id, compute in (
                (BETA_ASYMMETRY_ID, beta_asymmetry_252),
                (TAIL_RESILIENCE_ID, tail_resilience_252),
                (RECOVERY_RATIO_ID, recovery_ratio_252),
                (DOWN_DAY_ABSORPTION_ID, down_day_absorption_63),
            )
        ),
        *(
            RegisteredFeatureKernel(
                implementation_id=implementation_id,
                method_family=SESSION_OBSERVATION_METHOD_FAMILY,
                method_family_owners=installed_method_family_owners(
                    SESSION_OBSERVATION_METHOD_FAMILY
                ),
                required_fields=next(
                    item.required_fields
                    for item in session_observation_factor_specs()
                    if item.formula_ref == implementation_id
                ),
                implementation_hash=canonical_hash(
                    {
                        "implementation_id": implementation_id,
                        **SESSION_OBSERVATION_DECLARATIONS[implementation_id],
                    }
                ),
                compute=compute,
            )
            for implementation_id, compute in SESSION_OBSERVATION_KERNELS.items()
        ),
        RegisteredFeatureKernel(
            implementation_id=SESSION_DOLLAR_VOLUME_ID,
            method_family=SESSION_LIQUIDITY_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(SESSION_LIQUIDITY_METHOD_FAMILY),
            required_fields=SESSION_DOLLAR_VOLUME_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(SESSION_DOLLAR_VOLUME_DECLARATION)),
            compute=session_dollar_volume,
        ),
        RegisteredFeatureKernel(
            implementation_id=FORMULA_ID,
            method_family=FORMULA_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(FORMULA_METHOD_FAMILY),
            required_fields=FORMULA_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(FORMULA_DECLARATION)),
            compute=compute_formula,
        ),
        RegisteredFeatureKernel(
            implementation_id=FORMULA_SECTOR_ID,
            method_family=FORMULA_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(FORMULA_METHOD_FAMILY),
            required_fields=FORMULA_SECTOR_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(FORMULA_SECTOR_DECLARATION)),
            compute=compute_formula,
        ),
        RegisteredFeatureKernel(
            implementation_id=FORMULA_POINT_IN_TIME_ID,
            method_family=FORMULA_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(FORMULA_METHOD_FAMILY),
            required_fields=FORMULA_POINT_IN_TIME_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(FORMULA_POINT_IN_TIME_DECLARATION)),
            compute=compute_formula,
        ),
        RegisteredFeatureKernel(
            implementation_id=FORMULA_SECTOR_POINT_IN_TIME_ID,
            method_family=FORMULA_METHOD_FAMILY,
            method_family_owners=installed_method_family_owners(FORMULA_METHOD_FAMILY),
            required_fields=FORMULA_SECTOR_POINT_IN_TIME_REQUIRED_FIELDS,
            implementation_hash=canonical_hash(dict(FORMULA_SECTOR_POINT_IN_TIME_DECLARATION)),
            compute=compute_formula,
        ),
    )


__all__ = [
    "catalog_implementation_hashes",
    "catalog_methodology_hashes",
    "default_extension_kernel_registry",
    "extension_factor_specs",
    "installed_feature_kernels",
    "installed_formula_specs",
]
