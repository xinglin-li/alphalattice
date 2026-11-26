"""Absolute price-state Factors; preprocessing remains a separate capability."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    NO_ECONOMIC_SKIP,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

ABSOLUTE_STATE_METHOD_FAMILY: Final = "ABSOLUTE_STATE"
ABSOLUTE_STATE_REQUIRED_OHLC: Final = (
    "close_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
)

PRICE_VS_SMA20_ATR_ID: Final = "factor.desktop.experimental.price_vs_sma20_atr.v1"
PRICE_VS_SMA60_ATR_ID: Final = "factor.desktop.experimental.price_vs_sma60_atr.v1"
ABSOLUTE_MOMENTUM_20_ID: Final = "factor.desktop.experimental.absolute_momentum_20.v1"
DRAWDOWN_FROM_63D_HIGH_ID: Final = "factor.desktop.experimental.drawdown_from_63d_high.v1"

ABSOLUTE_STATE_DECLARATIONS: Final[Mapping[str, Mapping[str, str]]] = {
    PRICE_VS_SMA20_ATR_ID: {"algorithm": "(C[t]-SMA20[t-19:t])/ATR20[t-19:t]"},
    PRICE_VS_SMA60_ATR_ID: {"algorithm": "(C[t]-SMA60[t-59:t])/ATR20[t-19:t]"},
    ABSOLUTE_MOMENTUM_20_ID: {"algorithm": "ln(A[t]/A[t-20])"},
    DRAWDOWN_FROM_63D_HIGH_ID: {"algorithm": "C[t]/max(C[t-62:t])-1"},
}


def _spec(
    *,
    factor_id: str,
    formula_ref: str,
    formula: str,
    window: int,
    rows: int,
    fields: tuple[str, ...],
) -> FactorSpec:
    return FactorSpec(
        factor_id=factor_id,
        family=FactorFamily.PRICE_LEVEL_TREND,
        formula_ref=formula_ref,
        formula=formula,
        window_sessions=window,
        # Absolute price state is observed at the session it names; these
        # Formulas declare no economic skip.
        lag_sessions=NO_ECONOMIC_SKIP,
        return_convention="split_adjusted_absolute_price_state",
        required_fields=tuple(sorted(fields)),
        literature_sources=("https://doi.org/10.1111/j.1540-6261.1993.tb04702.x",),
        minimum_observations=rows,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        core_anchor=False,
    )


def absolute_state_factor_specs() -> tuple[FactorSpec, ...]:
    """Describe the installed absolute momentum, drawdown, and price-to-ATR state formulas.

    Returns:
        Four code-owned absolute-state specifications. Recipes retain their declared source fields,
        economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    return (
        _spec(
            factor_id="absolute_momentum_20",
            formula_ref=ABSOLUTE_MOMENTUM_20_ID,
            formula="ln(provider_adjusted_close[t]/provider_adjusted_close[t-20])",
            window=20,
            rows=21,
            fields=("provider_adjusted_close",),
        ),
        _spec(
            factor_id="drawdown_from_63d_high",
            formula_ref=DRAWDOWN_FROM_63D_HIGH_ID,
            formula="close[t]/max(close[t-62:t])-1",
            window=63,
            rows=63,
            fields=("close_split_adjusted",),
        ),
        _spec(
            factor_id="price_vs_sma20_atr",
            formula_ref=PRICE_VS_SMA20_ATR_ID,
            formula="(close[t]-SMA20(close[t-19:t]))/ATR20[t-19:t]",
            window=20,
            rows=21,
            fields=ABSOLUTE_STATE_REQUIRED_OHLC,
        ),
        _spec(
            factor_id="price_vs_sma60_atr",
            formula_ref=PRICE_VS_SMA60_ATR_ID,
            formula="(close[t]-SMA60(close[t-59:t]))/ATR20[t-19:t]",
            window=60,
            rows=60,
            fields=ABSOLUTE_STATE_REQUIRED_OHLC,
        ),
    )


def _ordered(source: pd.DataFrame) -> pd.DataFrame:
    return source.assign(_position=np.arange(len(source), dtype=int)).sort_values(
        ["listing_id", "session_date"], kind="mergesort"
    )


def _restore(source: pd.DataFrame, ordered: pd.DataFrame, values: np.ndarray) -> pd.Series:
    ordered["_value"] = values
    return pd.Series(
        ordered.sort_values("_position", kind="mergesort")["_value"].to_numpy(dtype=float),
        index=source.index,
    )


def _per_listing_arrays(
    source: pd.DataFrame,
    evaluator: Callable[[dict[str, np.ndarray], int], float],
    *,
    minimum_rows: int,
    fields: tuple[str, ...],
    prepare: Callable[[dict[str, np.ndarray]], None] | None = None,
) -> pd.Series:
    """Extract each listing once, then evaluate fixed causal windows.

    The former implementation passed a growing pandas prefix to every row's
    evaluator.  Each evaluator then converted the same columns to numpy again,
    making a 2520-session listing perform roughly 2500 DataFrame slices and
    conversions per Factor.  These methods only consume fixed trailing windows,
    so the prefix carried no additional authority or mathematical information.
    """
    ordered = _ordered(source)
    output: np.ndarray = np.full(len(ordered), np.nan, dtype=float)
    for _listing, positions in ordered.groupby("listing_id", sort=False).indices.items():
        indexes: np.ndarray = np.asarray(positions, dtype=np.intp)
        block = ordered.iloc[indexes]
        arrays = {
            field: pd.to_numeric(block[field], errors="coerce").to_numpy(dtype=float)
            for field in fields
        }
        if prepare is not None:
            prepare(arrays)
        for end in range(minimum_rows - 1, len(indexes)):
            value = float(evaluator(arrays, end))
            if np.isfinite(value):
                output[int(indexes[end])] = value
    return _restore(source, ordered, output)


def _true_ranges(*, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prior = np.roll(close, 1)
    prior[0] = np.nan
    result: np.ndarray = np.asarray(
        np.maximum.reduce((high - low, np.abs(high - prior), np.abs(low - prior))),
        dtype=np.float64,
    )
    return result


def _prepare_true_range(arrays: dict[str, np.ndarray]) -> None:
    arrays["_true_range"] = _true_ranges(
        high=arrays["high_split_adjusted"],
        low=arrays["low_split_adjusted"],
        close=arrays["close_split_adjusted"],
    )


def price_vs_sma20_atr(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Measure distance from the 20-session mean in units of 20-session ATR.

    Args:
        source: Listing/session rows with split-adjusted high, low, and close prices.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Current close minus its 20-session mean, divided by positive finite ATR. Results follow the
        input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(arrays: dict[str, np.ndarray], end: int) -> float:
        close = arrays["close_split_adjusted"]
        tr = arrays["_true_range"]
        level = close[end]
        sma = float(np.mean(close[end - 19 : end + 1]))
        atr = float(np.mean(tr[end - 19 : end + 1]))
        return (level - sma) / atr if np.isfinite([level, sma, atr]).all() and atr > 0 else np.nan

    return _per_listing_arrays(
        source,
        evaluate,
        minimum_rows=specification.minimum_observations,
        fields=ABSOLUTE_STATE_REQUIRED_OHLC,
        prepare=_prepare_true_range,
    )


def price_vs_sma60_atr(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Measure distance from the 60-session mean in units of 20-session ATR.

    Args:
        source: Listing/session rows with split-adjusted high, low, and close prices.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Current close minus its 60-session mean, divided by positive finite ATR. Results follow the
        input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(arrays: dict[str, np.ndarray], end: int) -> float:
        close = arrays["close_split_adjusted"]
        tr = arrays["_true_range"]
        level = close[end]
        sma = float(np.mean(close[end - 59 : end + 1]))
        atr = float(np.mean(tr[end - 19 : end + 1]))
        return (level - sma) / atr if np.isfinite([level, sma, atr]).all() and atr > 0 else np.nan

    return _per_listing_arrays(
        source,
        evaluate,
        minimum_rows=specification.minimum_observations,
        fields=ABSOLUTE_STATE_REQUIRED_OHLC,
        prepare=_prepare_true_range,
    )


def absolute_momentum_20(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Measure each listing's 20-session provider-adjusted log momentum.

    Args:
        source: Listing/session rows with provider-adjusted close prices.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Log of current close over the close 20 rows earlier, for positive endpoints. Results follow
        the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(arrays: dict[str, np.ndarray], end: int) -> float:
        close = arrays["provider_adjusted_close"]
        start, stop = close[end - 20], close[end]
        return np.log(stop / start) if start > 0 and stop > 0 else np.nan

    return _per_listing_arrays(
        source,
        evaluate,
        minimum_rows=specification.minimum_observations,
        fields=("provider_adjusted_close",),
    )


def drawdown_from_63d_high(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Measure the current split-adjusted close relative to its trailing 63-session high.

    Args:
        source: Listing/session rows with split-adjusted close prices.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Current close divided by the finite positive window maximum, minus one. Results follow the
        input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(arrays: dict[str, np.ndarray], end: int) -> float:
        close = arrays["close_split_adjusted"]
        window = close[end - 62 : end + 1]
        high = float(np.max(window)) if np.isfinite(window).all() else np.nan
        level = close[end]
        return level / high - 1.0 if high > 0 and level > 0 else np.nan

    return _per_listing_arrays(
        source,
        evaluate,
        minimum_rows=specification.minimum_observations,
        fields=("close_split_adjusted",),
    )


__all__ = [
    "ABSOLUTE_MOMENTUM_20_ID",
    "ABSOLUTE_STATE_DECLARATIONS",
    "ABSOLUTE_STATE_METHOD_FAMILY",
    "DRAWDOWN_FROM_63D_HIGH_ID",
    "PRICE_VS_SMA20_ATR_ID",
    "PRICE_VS_SMA60_ATR_ID",
    "absolute_momentum_20",
    "absolute_state_factor_specs",
    "drawdown_from_63d_high",
    "price_vs_sma20_atr",
    "price_vs_sma60_atr",
]
