"""Factor computation: the scalar as-of oracle, the formula language and the vectorized
core materializer that fills every listing's daily feature block."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.producers.base_materializer import (
    BaseFeatureMaterializer,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    FORMULA_ID,
    formula_specification,
    formula_tree,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack
from alphalattice.kernel.quant.factor_formulas import compute_factor

FORMULAS = {
    "momentum": "close / lag(close, 20) - 1",
    "price_volume_corr": "ts_corr(close / lag(close, 1), volume / lag(volume, 1), 20)",
    "ts_rank": "ts_rank(adjusted_close / lag(adjusted_close, 5), 60)",
    "ewm_over_std": "ewm(close / lag(close, 1) - 1, 3.5) / ts_std(close / lag(close, 1), 30)",
    "beta": "ts_beta(close / lag(close, 1), open / lag(open, 1), 40)",
    "range": "log(ts_max(high, 10) / ts_min(low, 10))",
}


def _scalar_spec(factor_id: str, window: int, lag: int, minimum: int) -> FactorSpec:
    return FactorSpec(
        factor_id=factor_id,
        family=FactorFamily.PRICE_LEVEL_TREND,
        formula_ref=f"factor.{factor_id}",
        formula=factor_id,
        window_sessions=window,
        lag_sessions=lag,
        return_convention="simple",
        required_fields=("close_total_return_adjusted",),
        literature_sources=("benchmark",),
        minimum_observations=minimum,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
    )


def _formula_spec(formula: str) -> FactorSpec:
    return formula_specification(
        FactorSpec(
            factor_id="formula_probe",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref=FORMULA_ID,
            formula=formula,
            window_sessions=1,
            lag_sessions=0,
            return_convention="as declared",
            required_fields=("close_split_adjusted",),
            literature_sources=("benchmark",),
            minimum_observations=1,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
        )
    )


SCALAR_SPECS = {
    "mom_252_21": _scalar_spec("mom_252_21", 252, 21, 253),
    "vol_252": _scalar_spec("vol_252", 252, 0, 253),
    "beta_252": _scalar_spec("beta_252", 252, 0, 253),
    "idio_vol_252": _scalar_spec("idio_vol_252", 252, 0, 253),
}


@pytest.mark.benchmark
@pytest.mark.parametrize("factor_id", sorted(SCALAR_SPECS))
def test_compute_factor(
    asset_bars: pd.DataFrame, market_bars: pd.DataFrame, factor_id: str
) -> None:
    result = compute_factor(SCALAR_SPECS[factor_id], asset_bars, market_bars)
    assert result.value is not None, result.reason


@pytest.mark.benchmark
def test_formula_parse() -> None:
    for formula in FORMULAS.values():
        formula_tree(formula)


@pytest.mark.parametrize("name", sorted(FORMULAS))
def test_formula_compute(formula_panel: pd.DataFrame, name: str, benchmark: Any) -> None:
    registry = default_extension_kernel_registry()
    spec = _formula_spec(FORMULAS[name])
    values = benchmark(registry.compute, formula_panel, spec)
    assert values.notna().any()


def test_materialize_listing(
    asset_bars: pd.DataFrame, market_bars: pd.DataFrame, benchmark: Any
) -> None:
    materializer = BaseFeatureMaterializer(FeatureCatalog.load())
    block = benchmark(
        materializer.materialize_listing,
        listing_id="BENCH",
        projected_bars=asset_bars,
        market_bars=market_bars,
    )
    assert not block.values.empty
