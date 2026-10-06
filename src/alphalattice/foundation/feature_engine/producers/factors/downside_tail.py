"""Market-relative downside, tail and recovery Factor families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    NO_ECONOMIC_SKIP,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

from .series_math import centered_beta, ordered_frame

DOWNSIDE_TAIL_METHOD_FAMILY: Final = "DOWNSIDE_TAIL"
BETA_ASYMMETRY_ID: Final = "factor.desktop.experimental.beta_asymmetry_252.v1"
TAIL_RESILIENCE_ID: Final = "factor.desktop.experimental.tail_resilience_252.v1"
RECOVERY_RATIO_ID: Final = "factor.desktop.experimental.recovery_ratio_252.v1"
DOWN_DAY_ABSORPTION_ID: Final = "factor.desktop.experimental.down_day_absorption_63.v1"
DOWNSIDE_TAIL_DECLARATIONS: Final[Mapping[str, Mapping[str, str]]] = {
    BETA_ASYMMETRY_ID: {"algorithm": "beta_down_252-beta_up_252; each side >=20"},
    TAIL_RESILIENCE_ID: {
        "algorithm": (
            "mean six-session residual episode over worst 25 market events, "
            "normalized by idio_std*sqrt(6)"
        ),
        "ols_interval": "[t-251,t]",
        "event_interval": "[t-251,t-5]",
        "recovery_interval": "[s,s+5]",
        "normalization": "idio_std*sqrt(6); sample std ddof=1; non-positive scale is missing",
    },
    RECOVERY_RATIO_ID: {"algorithm": "ln(A[t]/trough)/ln(peak/trough), earliest ties"},
    DOWN_DAY_ABSORPTION_ID: {
        "algorithm": "mean(market-model residual | market<0)/sample_std(residual), down>=10"
    },
}


def downside_tail_factor_specs() -> tuple[FactorSpec, ...]:
    """Describe the installed asymmetric beta, downside absorption, recovery, and tail formulas.

    Returns:
        The ordered downside and recovery specifications. Recipes retain their declared source
        fields, economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    common = ("market_return_log", "provider_adjusted_close")
    return (
        FactorSpec(
            factor_id="beta_asymmetry_252",
            family=FactorFamily.SYSTEMATIC,
            formula_ref=BETA_ASYMMETRY_ID,
            formula="beta_down_252-beta_up_252 with intercept and >=20 observations per side",
            window_sessions=252,
            # Market-relative downside state observed at session t; no skip.
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="provider_adjusted_stock_log_return_vs_market_log_return",
            required_fields=common,
            literature_sources=("https://doi.org/10.1111/0022-1082.00324",),
            minimum_observations=253,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
        FactorSpec(
            factor_id="down_day_absorption_63",
            family=FactorFamily.SYSTEMATIC,
            formula_ref=DOWN_DAY_ABSORPTION_ID,
            formula="mean(OLS63 residual on market-down sessions)/sample_std(all residuals)",
            window_sessions=63,
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="provider_adjusted_stock_log_return_vs_market_log_return",
            required_fields=common,
            literature_sources=("https://doi.org/10.1111/0022-1082.00324",),
            minimum_observations=64,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
        FactorSpec(
            factor_id="recovery_ratio_252",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref=RECOVERY_RATIO_ID,
            formula="ln(A[t]/trough)/ln(peak/trough) over max drawdown in A[t-251:t]",
            window_sessions=252,
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="provider_adjusted_absolute_price_state",
            required_fields=("provider_adjusted_close",),
            literature_sources=("https://doi.org/10.3905/jpm.2013.39.4.065",),
            minimum_observations=252,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
        FactorSpec(
            factor_id="tail_resilience_252",
            family=FactorFamily.SYSTEMATIC,
            formula_ref=TAIL_RESILIENCE_ID,
            formula=(
                "mean_s(sum_{j=s}^{s+5} residual_j)/(idio_std*sqrt(6)), "
                "worst 25 market s in [t-251,t-5]); OLS residuals fitted on [t-251,t]; "
                "idio_std = sample std(residual, ddof=1)"
            ),
            window_sessions=252,
            lag_sessions=NO_ECONOMIC_SKIP,
            return_convention="provider_adjusted_stock_log_return_vs_market_log_return",
            required_fields=common,
            literature_sources=("https://doi.org/10.1093/rfs/hhm014",),
            minimum_observations=253,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
            core_anchor=False,
        ),
    )


def _returns(prices: np.ndarray) -> np.ndarray:
    stock: np.ndarray = np.full(len(prices), np.nan, dtype=float)
    valid = (prices[1:] > 0.0) & (prices[:-1] > 0.0)
    valid_positions: np.ndarray = np.flatnonzero(valid) + 1
    stock[valid_positions] = np.log(prices[valid_positions] / prices[valid_positions - 1])
    return stock


def _apply(
    source: pd.DataFrame,
    rows: int,
    evaluator: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], float],
) -> pd.Series:
    """Extract each listing's arrays once, then evaluate fixed causal windows.

    The former implementation sliced a growing pandas prefix and rebuilt stock
    returns for every output row.  For a 2520-session listing that converted the
    same price history roughly 2200 times per Factor.  All four methods consume
    fixed trailing windows, so one immutable listing array is equivalent and
    keeps their original numpy reduction order inside each window.
    """
    ordered = ordered_frame(source)
    output: np.ndarray = np.full(len(ordered), np.nan, dtype=float)
    for _listing, positions in ordered.groupby("listing_id", sort=False).indices.items():
        indexes = np.asarray(positions, dtype=np.intp)
        block = ordered.iloc[indexes]
        prices = pd.to_numeric(block["provider_adjusted_close"], errors="coerce").to_numpy(
            dtype=float
        )
        stock = _returns(prices)
        # Three of this family's four methods are market-relative and one --
        # recovery_ratio_252 -- is a pure price-path statistic that declares only
        # the price column. Demanding the market column regardless made the
        # helper require more than any specification promises, which the boundary
        # golden refuses to supply and correctly failed on.
        market = (
            pd.to_numeric(block["market_return_log"], errors="coerce").to_numpy(dtype=float)
            if "market_return_log" in block.columns
            else np.full(len(indexes), np.nan, dtype=float)
        )
        dates = (
            pd.to_datetime(block["session_date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        )
        for end in range(rows - 1, len(indexes)):
            value = float(evaluator(prices, stock, market, dates, end))
            if np.isfinite(value):
                output[int(indexes[end])] = value
    ordered["_value"] = output
    restored = ordered.sort_values("_source_position", kind="mergesort")
    return pd.Series(restored["_value"].to_numpy(dtype=float), index=source.index)


def beta_asymmetry_252(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Compare downside and non-downside market betas over 252 returns.

    Args:
        source: Listing/session rows with provider-adjusted closes and market log returns.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Downside beta minus non-downside beta, requiring at least 20 observations per side. Results
        follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(
        _prices: np.ndarray,
        stock: np.ndarray,
        market: np.ndarray,
        _dates: np.ndarray,
        end: int,
    ) -> float:
        x, y = stock[end - 251 : end + 1], market[end - 251 : end + 1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return np.nan
        down, up = y < 0.0, y >= 0.0
        if int(down.sum()) < 20 or int(up.sum()) < 20:
            return np.nan
        return float(centered_beta(x[down], y[down]) - centered_beta(x[up], y[up]))

    return _apply(source, specification.minimum_observations, evaluate)


def down_day_absorption_63(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Scale average stock residuals on market-down days by full-window residual volatility.

    Args:
        source: Listing/session rows with provider-adjusted closes and market log returns.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Mean down-day OLS residual over sample residual deviation, with at least ten down days.
        Results follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(
        _prices: np.ndarray,
        stock: np.ndarray,
        market: np.ndarray,
        _dates: np.ndarray,
        end: int,
    ) -> float:
        x, y = stock[end - 62 : end + 1], market[end - 62 : end + 1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return np.nan
        down = y < 0.0
        if int(down.sum()) < 10:
            return np.nan
        beta = centered_beta(x, y)
        if not np.isfinite(beta):
            return np.nan
        residual = x - float(np.mean(x)) - beta * (y - float(np.mean(y)))
        scale = float(np.std(residual, ddof=1))
        return float(np.mean(residual[down]) / scale) if scale > 0.0 else np.nan

    return _apply(source, specification.minimum_observations, evaluate)


def tail_resilience_252(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Measure six-return residual episodes starting at the worst 25 market events.

    Args:
        source: Listing/session rows with provider-adjusted closes and market log returns.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Mean six-return residual sum divided by sample residual deviation times sqrt(6).
        Event ties use dates; each episode ends within the 252-return observation window. Results
        follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(
        _prices: np.ndarray,
        stock: np.ndarray,
        market: np.ndarray,
        dates: np.ndarray,
        end: int,
    ) -> float:
        x, y = stock[end - 251 : end + 1], market[end - 251 : end + 1]
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            return np.nan
        beta = centered_beta(x, y)
        if not np.isfinite(beta):
            return np.nan
        residual = x - float(np.mean(x)) - beta * (y - float(np.mean(y)))
        idio_scale = float(np.std(residual, ddof=1))
        if not np.isfinite(idio_scale) or idio_scale <= 0.0:
            return np.nan
        candidate = y[:247]
        event_dates = dates[end - 251 : end - 4]
        order = np.lexsort((event_dates, candidate))
        events = order[:25]
        episodes = np.asarray([np.sum(residual[index : index + 6]) for index in events])
        if not np.isfinite(episodes).all():
            return np.nan
        return float(np.mean(episodes) / (idio_scale * np.sqrt(6.0)))

    return _apply(source, specification.minimum_observations, evaluate)


def recovery_ratio_252(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Measure recovery from the largest log drawdown in 252 provider-adjusted prices.

    Args:
        source: Listing/session rows with provider-adjusted close prices.
        specification: Qualified catalog recipe whose declared source-row minimum governs
            eligibility.

    Returns:
        Log recovery from the selected trough divided by its peak-to-trough log drawdown.
        Nonpositive, nonfinite, or no-drawdown windows are missing. Results follow the input row
        axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(
        prices: np.ndarray,
        _stock: np.ndarray,
        _market: np.ndarray,
        _dates: np.ndarray,
        end: int,
    ) -> float:
        values = prices[end - 251 : end + 1]
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            return np.nan
        best = 0.0
        peak_index = trough_index = 0
        running_peak = values[0]
        running_peak_index = 0
        for index, value in enumerate(values):
            if value > running_peak:
                running_peak = value
                running_peak_index = index
            drawdown = np.log(running_peak / value)
            if drawdown > best:
                best = float(drawdown)
                peak_index, trough_index = running_peak_index, index
        if best <= 0.0:
            return np.nan
        denominator = np.log(values[peak_index] / values[trough_index])
        return float(np.log(values[-1] / values[trough_index]) / denominator)

    return _apply(source, specification.minimum_observations, evaluate)


__all__ = [
    "BETA_ASYMMETRY_ID",
    "DOWNSIDE_TAIL_DECLARATIONS",
    "DOWNSIDE_TAIL_METHOD_FAMILY",
    "DOWN_DAY_ABSORPTION_ID",
    "RECOVERY_RATIO_ID",
    "TAIL_RESILIENCE_ID",
    "beta_asymmetry_252",
    "down_day_absorption_63",
    "downside_tail_factor_specs",
    "recovery_ratio_252",
    "tail_resilience_252",
]
