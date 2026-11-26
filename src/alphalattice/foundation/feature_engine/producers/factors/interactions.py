"""Raw children for explicit Market-state interaction preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    NO_ECONOMIC_SKIP,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

from .series_math import ordered_frame

STATE_INTERACTION_METHOD_FAMILY: Final = "STATE_INTERACTION"
MARKET_DRAWDOWN_X_MOMENTUM_ID: Final = "factor.desktop.experimental.market_drawdown_x_momentum.v1"
MARKET_VOL_RATIO_X_REVERSAL_ID: Final = "factor.desktop.experimental.market_vol_ratio_x_reversal.v1"
INTERACTION_DECLARATIONS: Final[Mapping[str, Mapping[str, str]]] = {
    MARKET_DRAWDOWN_X_MOMENTUM_ID: {
        "stock_child": "absolute_momentum_20 raw formula ln(A[t]/A[t-20])",
        "market_child": "clip(max(-market_drawdown_63[t],0)/0.2,0,1)",
        "composition": "STATE_INTERACTION_BLOCK",
    },
    MARKET_VOL_RATIO_X_REVERSAL_ID: {
        "stock_child": "verified base Panel rev_5 cross-sectional child",
        "market_child": "clip(log(std21[t]/std252[t])/log(2),-1,1)",
        "composition": "STATE_INTERACTION_BLOCK",
    },
}


def interaction_factor_specs() -> tuple[FactorSpec, ...]:
    """Describe the stock-child and market-state interaction recipes.

    Returns:
        Momentum/drawdown and reversal/volatility-state specifications. Recipes retain their
        declared source fields, economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    return (
        FactorSpec(
            factor_id="market_drawdown_x_momentum",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref=MARKET_DRAWDOWN_X_MOMENTUM_ID,
            formula=(
                "STATE_INTERACTION_BLOCK(XS_CHILD(absolute_momentum_20), market_drawdown_state)"
            ),
            window_sessions=63,
            # Both children are observed at session t: the stock leg is the
            # absolute-momentum Formula and the Market leg is a state read at the
            # same close. Neither declares an economic skip.
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="provider_adjusted_stock_and_market_state_children",
            required_fields=("market_provider_adjusted_close", "provider_adjusted_close"),
            literature_sources=("https://doi.org/10.1111/jofi.12021",),
            minimum_observations=63,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
        FactorSpec(
            factor_id="market_vol_ratio_x_reversal",
            family=FactorFamily.REVERSAL,
            formula_ref=MARKET_VOL_RATIO_X_REVERSAL_ID,
            formula="STATE_INTERACTION_BLOCK(base_panel_rev_5, market_vol_ratio_state)",
            window_sessions=252,
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="verified_panel_child_and_market_log_return_state",
            required_fields=("market_provider_adjusted_close", "rev_5"),
            literature_sources=("https://doi.org/10.1093/rfs/hhm014",),
            minimum_observations=253,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
    )


def _market_states(source: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    ordered = ordered_frame(source)
    market_rows = ordered.loc[:, ["session_date", "market_provider_adjusted_close"]].copy()
    consistency = market_rows.groupby("session_date", sort=True)[
        "market_provider_adjusted_close"
    ].nunique(dropna=False)
    if bool((consistency > 1).any()):
        raise ValueError("STATE_INTERACTION_MARKET_CHILD_INCONSISTENT")
    market = (
        market_rows.drop_duplicates("session_date")
        .sort_values("session_date", kind="mergesort")
        .set_index("session_date")["market_provider_adjusted_close"]
        .pipe(pd.to_numeric, errors="coerce")
        .astype(float)
    )
    # Both Market states are read at the observation session's own close: the
    # 63-session high and the volatility windows end at t rather than t-1.
    rolling_high = market.rolling(63, min_periods=63).max()
    drawdown_by_session = (-(market / rolling_high - 1.0) / 0.2).clip(0.0, 1.0)
    returns = np.log(market / market.shift(1))
    slow = returns.rolling(252, min_periods=252).std(ddof=1)
    fast = returns.rolling(21, min_periods=21).std(ddof=1)
    vol_by_session = (np.log(fast / slow) / np.log(2.0)).clip(-1.0, 1.0)
    drawdown = ordered["session_date"].map(drawdown_by_session).to_numpy(float)
    vol_ratio = ordered["session_date"].map(vol_by_session).to_numpy(float)
    return ordered, drawdown, vol_ratio


def market_drawdown_x_momentum_raw(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Produce the stock momentum child before applying the market-drawdown state.

    Args:
        source: Listing/session rows with provider-adjusted close prices.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Twenty-session absolute log momentum; the state multiplication belongs to preprocessing.
        Results follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """
    ordered = ordered_frame(source)
    prices = pd.to_numeric(ordered["provider_adjusted_close"], errors="coerce")
    groups = ordered["listing_id"]
    start = prices.groupby(groups, sort=False).shift(20)
    end = prices
    source_row = ordered.groupby("listing_id", sort=False).cumcount()
    valid = (start > 0.0) & (end > 0.0) & (source_row >= specification.minimum_observations - 1)
    output = np.log((end / start).where(valid)).to_numpy(float)
    ordered["_value"] = output
    restored = ordered.sort_values("_source_position", kind="mergesort")
    return pd.Series(restored["_value"].to_numpy(dtype=float), index=source.index)


def market_vol_ratio_x_reversal_raw(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Project the verified reversal child before applying the market-volatility state.

    Args:
        source: Listing/session rows with the admitted rev_5 child values.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        The reversal child after the declared history minimum; state multiplication is separate.
        Results follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """
    ordered = ordered_frame(source)
    child = pd.to_numeric(ordered["rev_5"], errors="coerce").astype(float)
    source_row = ordered.groupby("listing_id", sort=False).cumcount()
    child = child.where(source_row >= specification.minimum_observations - 1)
    restored = ordered.assign(_value=child).sort_values("_source_position", kind="mergesort")
    return pd.Series(restored["_value"].to_numpy(dtype=float), index=source.index)


def interaction_market_state_children(source: pd.DataFrame) -> pd.DataFrame:
    """Return the two separately identified state-child value columns."""
    ordered, drawdown, vol = _market_states(source)
    ordered["market_drawdown_x_momentum__state"] = drawdown
    ordered["market_vol_ratio_x_reversal__state"] = vol
    return ordered.sort_values("_source_position", kind="mergesort").loc[
        :, ["market_drawdown_x_momentum__state", "market_vol_ratio_x_reversal__state"]
    ]


__all__ = [
    "INTERACTION_DECLARATIONS",
    "MARKET_DRAWDOWN_X_MOMENTUM_ID",
    "MARKET_VOL_RATIO_X_REVERSAL_ID",
    "STATE_INTERACTION_METHOD_FAMILY",
    "interaction_factor_specs",
    "interaction_market_state_children",
    "market_drawdown_x_momentum_raw",
    "market_vol_ratio_x_reversal_raw",
]
