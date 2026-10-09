"""Prove a new Alpha target standardization needs no compiler edit.

`AlphaTargetPolicy` already bound the raw causal outcome, ordered transform
sequence, winsorization, neutralization, and output semantics. What it lacked was
an installation seam: the lane standardization was chosen by a hard-coded branch
inside `compile_alpha_target_surface`. That branch is now a catalog lookup.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pyarrow as pa
import pytest
from numpy.typing import NDArray

from alphalattice.investment.alpha_research.targets.catalog import (
    AlphaTargetCatalog,
    build_installed_alpha_target_catalog,
)
from alphalattice.investment.alpha_research.targets.contracts import StandardizedTargetLane
from alphalattice.investment.alpha_research.targets.development import (
    AlphaDevelopmentTargetRecipe,
    compile_alpha_development_target_surface,
)
from alphalattice.investment.alpha_research.targets.execution_outcome import (
    AlphaTargetLane,
    build_alpha_target_policy,
)
from alphalattice.investment.alpha_research.targets.standardization import (
    RANK_GAUSS_STANDARDIZATION_ID,
    ROBUST_Z_STANDARDIZATION_ID,
)

type FloatArray = NDArray[np.float64]

_SESSIONS = (date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3))
# MIN_SECTOR_SAMPLE is 5 and MIN_COVERAGE is 0.98, so the fixture needs a real
# cross-section per sector or the Host masks every session before the lane runs.
_LISTINGS = tuple(f"listing-{index:02d}" for index in range(12))
_SECTORS = {
    listing: ("sector-1" if index < 6 else "sector-2") for index, listing in enumerate(_LISTINGS)
}
_SECTOR_REVISION = "a" * 64


def _source_table() -> pa.Table:
    rng = np.random.default_rng(5)
    rows = len(_SESSIONS) * len(_LISTINGS)
    returns = rng.normal(0.0, 0.02, size=rows)
    return pa.table(
        {
            "formation_session": [session for session in _SESSIONS for _ in _LISTINGS],
            "listing_id": [listing for _ in _SESSIONS for listing in _LISTINGS],
            "fit_target": pa.array(returns, type=pa.float64()),
            "simple_economic_return": pa.array(np.expm1(returns), type=pa.float64()),
        }
    )


class _SignStandardization:
    """A third standardization owned entirely by this case study."""

    standardization_id = "CASE_STUDY_SIGN"

    def standardize(self, values: FloatArray, *, mad_scale: float) -> StandardizedTargetLane:
        del mad_scale
        lane_values = np.sign(values)
        return StandardizedTargetLane(
            values=lane_values,
            dispersion=np.nanstd(lane_values, axis=1),
        )


def test_a_third_standardization_runs_under_its_own_name() -> None:
    """A third standardization runs under its own name."""

    policy = build_alpha_target_policy(
        lane=AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
        sector_revision=_SECTOR_REVISION,
    )
    recipe = AlphaDevelopmentTargetRecipe.create(
        policy=policy,
        standardization_id=_SignStandardization.standardization_id,
    )
    installed = build_installed_alpha_target_catalog()

    surface = compile_alpha_development_target_surface(
        source_table=_source_table(),
        recipe=recipe,
        sector_by_listing_id=_SECTORS,
        standardizations=AlphaTargetCatalog((*installed.adapters, _SignStandardization())),
    )

    # The recipe names the method that ran, while the lane's derived key still
    # says rank-gauss: the two are genuinely separable now.
    assert recipe.standardization_id == "CASE_STUDY_SIGN"
    assert policy.standardization_id == RANK_GAUSS_STANDARDIZATION_ID
    # Two recipes over one lane with different standardizations are two recipes,
    # and the lane identity they share is unchanged.
    alternate = AlphaDevelopmentTargetRecipe.create(
        policy=policy, standardization_id=RANK_GAUSS_STANDARDIZATION_ID
    )
    assert alternate.recipe_hash != recipe.recipe_hash
    assert alternate.policy.policy_hash == recipe.policy.policy_hash

    values = np.asarray(
        surface["fit_target"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    # The case-study method produced the lane, not rank-gauss.
    assert set(np.unique(finite)).issubset({-1.0, 1.0})
    # Raw economic returns are retained regardless of the lane method.
    assert "simple_economic_return" in surface.schema.names
    assert "raw_log_execution_return" in surface.schema.names


def test_installed_catalog_exposes_exactly_the_real_standardizations() -> None:
    """Installed catalog exposes exactly the real standardizations."""

    from alphalattice.investment.alpha_research.targets.standardization import (
        CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID,
    )

    catalog = build_installed_alpha_target_catalog()
    assert catalog.standardization_ids == (
        RANK_GAUSS_STANDARDIZATION_ID,
        ROBUST_Z_STANDARDIZATION_ID,
        CROSS_SECTIONAL_STD_Z_STANDARDIZATION_ID,
    )
    assert "CASE_STUDY_SIGN" not in catalog.standardization_ids
    extended = AlphaTargetCatalog((*catalog.adapters, _SignStandardization()))
    assert extended.binding.catalog_hash != catalog.binding.catalog_hash


@pytest.mark.parametrize(
    ("lane", "expected"),
    [
        (AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS, RANK_GAUSS_STANDARDIZATION_ID),
        (AlphaTargetLane.SECTOR_RESIDUAL_ROBUST_Z, ROBUST_Z_STANDARDIZATION_ID),
        (AlphaTargetLane.LOG_RETURN_RANK_GAUSS, RANK_GAUSS_STANDARDIZATION_ID),
        (AlphaTargetLane.LOG_RETURN_ROBUST_Z, ROBUST_Z_STANDARDIZATION_ID),
    ],
)
def test_every_frozen_lane_routes_without_changing_its_policy_hash(
    lane: AlphaTargetLane, expected: str
) -> None:
    policy = build_alpha_target_policy(lane=lane, sector_revision=_SECTOR_REVISION)
    assert policy.standardization_id == expected
    # A property, never a field: frozen target artifact identity is untouched.
    assert "standardization_id" not in policy.model_dump(mode="json")
    build_installed_alpha_target_catalog().resolve(policy.standardization_id)


def test_uninstalled_standardization_fails_closed() -> None:
    with pytest.raises(ValueError, match="ALPHA_TARGET_STANDARDIZATION_NOT_INSTALLED"):
        build_installed_alpha_target_catalog().resolve("CASE_STUDY_SIGN")

    # Naming is not installing: a recipe may state any routing key, and the
    # catalog still refuses one nothing installed. Checked before any numbers
    # move, so a rejected recipe computes nothing.
    recipe = AlphaDevelopmentTargetRecipe.create(
        policy=build_alpha_target_policy(
            lane=AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
            sector_revision=_SECTOR_REVISION,
        ),
        standardization_id="CASE_STUDY_SIGN",
    )
    with pytest.raises(ValueError, match="ALPHA_TARGET_STANDARDIZATION_NOT_INSTALLED"):
        compile_alpha_development_target_surface(
            source_table=_source_table(),
            recipe=recipe,
            sector_by_listing_id=_SECTORS,
        )
