"""Vectorized daily materialization for the immutable core-equity registry.

The formal ``compute_factor`` function is intentionally a scalar, as-of oracle.
This module calculates one listing/time block at a time: it never builds a
Python ``listing x session x factor`` frame-slicing loop.  A small oracle probe
is kept separate and compares selected rows to the Quant implementation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.contracts import (
    MARKET_DEPENDENT_FACTOR_IDS,
    FeatureCatalog,
)
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    formula_skip_sessions,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.kernel.quant.factor_formulas import (
    ANNUALIZATION,
    REASON_INSUFFICIENT_HISTORY,
    REASON_MARKET_ALIGNMENT,
    REASON_ZERO_DENOMINATOR,
    FormulaResult,
)

_REQUIRED_COLUMNS: Final = (
    "session_date",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
    "open_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
    "close_split_adjusted",
    "provider_adjusted_close",
)


@dataclass(frozen=True)
class MaterializedFeatureBlock:
    """Contain one listing's daily feature values and sparse ineligibility facts.

    Attributes:
        values: Listing-session rows, factor values, and serialized source cutoffs.
        ineligibility: Missing factor observations with reasons and source-history counts.
    """

    values: pd.DataFrame
    ineligibility: pd.DataFrame


def _rolling_linear_slope(values: pd.Series, window: int) -> pd.Series:
    """OLS slope for a normalized, fixed 0..1 session index, fully vectorized."""
    x = np.linspace(0.0, 1.0, window, dtype=np.float64)
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    raw = values.to_numpy(dtype=np.float64)
    result = np.full(len(raw), np.nan, dtype=np.float64)
    if len(raw) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(raw, window)
        valid = np.isfinite(windows).all(axis=1)
        slopes = np.full(len(windows), np.nan, dtype=np.float64)
        slopes[valid] = np.sum(windows[valid] * centered_x, axis=1) / denominator
        result[window - 1 :] = slopes
    return pd.Series(result, index=values.index, dtype=float)


def _rolling_trend_r2(values: pd.Series, window: int) -> pd.Series:
    x = np.linspace(0.0, 1.0, window, dtype=np.float64)
    centered_x = x - x.mean()
    denominator = float(np.dot(centered_x, centered_x))
    raw = values.to_numpy(dtype=np.float64)
    result = np.full(len(raw), np.nan, dtype=np.float64)
    if len(raw) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(raw, window)
        valid = np.isfinite(windows).all(axis=1)
        centered_y = windows - windows.mean(axis=1, keepdims=True)
        slope = np.sum(centered_y * centered_x, axis=1) / denominator
        fitted = slope[:, None] * centered_x
        total = np.sum(centered_y**2, axis=1)
        residual = np.sum((centered_y - fitted) ** 2, axis=1)
        output = np.full(len(windows), np.nan, dtype=np.float64)
        output[valid & (total > 0.0)] = (
            1.0 - residual[valid & (total > 0.0)] / total[valid & (total > 0.0)]
        )
        result[window - 1 :] = output
    return pd.Series(result, index=values.index, dtype=float)


def _window_matrix(values: pd.Series, window: int, skip: int) -> tuple[np.ndarray, int]:
    """Return direct source windows and their first aligned output position.

    ``skip`` is the Formula's own economic skip: the window that lands on
    observation session ``t`` ends at source row ``t - skip``. It is not an
    availability margin and not an execution offset -- see
    ``feature_engine.catalog.observation_clock`` for the three owners this used to
    conflate.
    """
    if window < 1 or skip < 0:
        raise ValueError("rolling window and formula skip must be non-negative")
    raw = values.to_numpy(dtype=np.float64)
    first_output = window + skip - 1
    if len(raw) <= first_output:
        return np.empty((0, window), dtype=np.float64), first_output
    windows = np.lib.stride_tricks.sliding_window_view(raw, window)
    return windows[: len(raw) - first_output], first_output


def _rolling_stat(
    values: pd.Series,
    window: int,
    *,
    skip: int = 0,
    statistic: str,
) -> pd.Series:
    """Evaluate one finite rolling statistic from each window independently.

    Pandas' online add/remove accumulators retain prefix-dependent floating
    state.  Direct window evaluation makes bounded correction rebuilds exactly
    partition-independent while retaining the same finite-window formulas.
    """
    windows, first_output = _window_matrix(values, window, skip)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if not len(windows):
        return pd.Series(result, index=values.index, dtype=float)
    valid = np.isfinite(windows).all(axis=1)
    usable = windows[valid]
    computed = np.full(len(windows), np.nan, dtype=np.float64)
    if len(usable):
        if statistic == "sum":
            selected = np.sum(usable, axis=1)
        elif statistic == "mean":
            selected = np.mean(usable, axis=1)
        elif statistic == "std":
            selected = np.std(usable, axis=1, ddof=1)
        elif statistic == "var":
            selected = np.var(usable, axis=1, ddof=1)
        elif statistic == "min":
            selected = np.min(usable, axis=1)
        elif statistic == "max":
            selected = np.max(usable, axis=1)
        elif statistic in {"skew", "kurt"}:
            centered = usable - np.mean(usable, axis=1, keepdims=True)
            sample_std = np.std(usable, axis=1, ddof=1)
            positive = sample_std > 0.0
            selected = np.full(len(usable), np.nan, dtype=np.float64)
            if statistic == "skew" and window >= 3:
                standardized_cube = np.sum(
                    (centered[positive] / sample_std[positive, None]) ** 3, axis=1
                )
                selected[positive] = window * standardized_cube / ((window - 1) * (window - 2))
            elif statistic == "kurt" and window >= 4:
                standardized_fourth = np.sum(
                    (centered[positive] / sample_std[positive, None]) ** 4, axis=1
                )
                selected[positive] = window * (window + 1) * standardized_fourth / (
                    (window - 1) * (window - 2) * (window - 3)
                ) - 3.0 * (window - 1) ** 2 / ((window - 2) * (window - 3))
        else:
            raise ValueError(f"unknown finite rolling statistic: {statistic}")
        computed[valid] = selected
    result[first_output:] = computed
    return pd.Series(result, index=values.index, dtype=float)


AMIHUD_MINIMUM_SHARE: Final = 0.8
"""The share of an Amihud window's sessions that must carry the ratio (17 of 21, 202 of 252)."""


def _rolling_observed_mean(
    values: pd.Series, window: int, *, minimum: int, defined_from: int
) -> pd.Series:
    """The mean of each window's finite values, given at least ``minimum`` of them.

    Evaluated window by window, as ``_rolling_stat`` is, so a bounded correction rebuild
    reproduces it exactly; a complete window's mean is ``_rolling_stat``'s to the bit, and a
    window with fewer finite values than ``minimum`` is missing. A window reaching back before
    ``defined_from``, where the series begins (a return's first session), is missing too: its
    first value is the history's start, not a session without one, so the first value lands
    where the Formula's clock declares.
    """
    windows, first_output = _window_matrix(values, window, 0)
    result = np.full(len(values), np.nan, dtype=np.float64)
    if len(windows):
        finite = np.isfinite(windows)
        counts = finite.sum(axis=1)
        sums = np.where(finite, windows, 0.0).sum(axis=1)
        observed = counts >= minimum
        computed = np.full(len(windows), np.nan, dtype=np.float64)
        computed[observed] = sums[observed] / counts[observed]
        result[first_output:] = computed
        result[: first_output + defined_from] = np.nan
    return pd.Series(result, index=values.index, dtype=float)


def _rolling_autocorrelation(returns: pd.Series, window: int, skip: int = 0) -> pd.Series:
    source_windows, first_output = _window_matrix(returns, window + 1, skip)
    result = np.full(len(returns), np.nan, dtype=np.float64)
    if not len(source_windows):
        return pd.Series(result, index=returns.index, dtype=float)
    valid = np.isfinite(source_windows).all(axis=1)
    left = source_windows[:, 1:]
    right = source_windows[:, :-1]
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
    correlation = np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=np.full(len(source_windows), np.nan),
        where=valid & (denominator > 0.0),
    )
    result[first_output:] = correlation
    return pd.Series(result, index=returns.index, dtype=float)


def _paired_metrics(
    asset_returns: pd.Series, market_returns: pd.Series, window: int
) -> dict[str, pd.Series]:
    """Compute costly systematic-window statistics in vectorized numpy blocks."""
    asset = asset_returns.to_numpy(dtype=np.float64)
    market = market_returns.to_numpy(dtype=np.float64)
    length = len(asset)
    output = {
        key: np.full(length, np.nan, dtype=np.float64)
        for key in (
            "beta",
            "corr",
            "coskew",
            "idio_vol",
            "idio_skew",
            "downside_beta",
        )
    }
    if length < window:
        return {key: pd.Series(value, index=asset_returns.index) for key, value in output.items()}
    asset_windows = np.lib.stride_tricks.sliding_window_view(asset, window)
    market_windows = np.lib.stride_tricks.sliding_window_view(market, window)
    valid = np.isfinite(asset_windows).all(axis=1) & np.isfinite(market_windows).all(axis=1)
    x = asset_windows
    y = market_windows
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    market_sum_sq = np.sum(yc**2, axis=1)
    asset_sum_sq = np.sum(xc**2, axis=1)
    beta = np.divide(
        np.sum(xc * yc, axis=1),
        market_sum_sq,
        out=np.full(len(x), np.nan),
        where=market_sum_sq > 0.0,
    )
    corr_denom = np.sqrt(asset_sum_sq * market_sum_sq)
    corr = np.divide(
        np.sum(xc * yc, axis=1), corr_denom, out=np.full(len(x), np.nan), where=corr_denom > 0.0
    )
    # Intercept regression residuals can be expressed from the one window's
    # centered data; no rolling Python regression is needed.
    residual = xc - beta[:, None] * yc
    residual_std = np.sqrt(np.sum(residual**2, axis=1) / max(window - 1, 1))
    residual_centered = residual - residual.mean(axis=1, keepdims=True)
    idio_skew = np.divide(
        window * np.sum((residual_centered / residual_std[:, None]) ** 3, axis=1),
        (window - 1) * (window - 2),
        out=np.full(len(x), np.nan),
        where=(residual_std > 0.0) & (window >= 3),
    )
    asset_std = np.sqrt(asset_sum_sq / max(window - 1, 1))
    market_var = market_sum_sq / max(window - 1, 1)
    coskew = np.divide(
        np.mean(xc * yc**2, axis=1),
        asset_std * market_var,
        out=np.full(len(x), np.nan),
        where=(asset_std > 0.0) & (market_var > 0.0),
    )
    negative = y < 0.0
    negative_count = negative.sum(axis=1)
    masked_x = np.where(negative, x, np.nan)
    masked_y = np.where(negative, y, np.nan)
    x_mean = np.divide(
        np.nansum(masked_x, axis=1),
        negative_count,
        out=np.full(len(x), np.nan),
        where=negative_count > 0,
    )
    y_mean = np.divide(
        np.nansum(masked_y, axis=1),
        negative_count,
        out=np.full(len(y), np.nan),
        where=negative_count > 0,
    )
    x_delta = masked_x - x_mean[:, None]
    y_delta = masked_y - y_mean[:, None]
    downside_denominator = np.nansum(y_delta**2, axis=1)
    downside_beta = np.divide(
        np.nansum(x_delta * y_delta, axis=1),
        downside_denominator,
        out=np.full(len(x), np.nan),
        where=(negative_count >= 2) & (downside_denominator > 0.0),
    )
    place = slice(window - 1, None)
    usable = valid & (market_sum_sq > 0.0)
    output["beta"][place] = np.where(usable, beta, np.nan)
    output["corr"][place] = np.where(usable, corr, np.nan)
    output["coskew"][place] = np.where(usable, coskew, np.nan)
    output["idio_vol"][place] = np.where(usable, residual_std * ANNUALIZATION, np.nan)
    output["idio_skew"][place] = np.where(usable, idio_skew, np.nan)
    output["downside_beta"][place] = np.where(usable, downside_beta, np.nan)
    return {
        key: pd.Series(value, index=asset_returns.index, dtype=float)
        for key, value in output.items()
    }


def _residual_momentum(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    *,
    estimation_window: int = 252,
    formation_window: int = 231,
    skip: int = 21,
) -> pd.Series:
    """Sum out-of-subwindow residual log returns under one fixed beta fit.

    At feature session ``t``, beta and intercept are estimated over 252 return
    observations ending at ``t-21``.  Residual momentum then sums only the
    terminal 231 residuals of that fit.  The formation window is therefore a
    true subset of the estimation window instead of the in-sample residual set
    whose sum is identically zero.
    """
    if formation_window >= estimation_window:
        raise ValueError("residual formation window must be smaller than estimation window")
    asset_windows, first_output = _window_matrix(asset_returns, estimation_window, skip)
    market_windows, market_first_output = _window_matrix(market_returns, estimation_window, skip)
    if market_first_output != first_output or len(asset_windows) != len(market_windows):
        raise ValueError("residual momentum windows are not aligned")
    result = np.full(len(asset_returns), np.nan, dtype=np.float64)
    if not len(asset_windows):
        return pd.Series(result, index=asset_returns.index, dtype=float)
    valid = np.isfinite(asset_windows).all(axis=1) & np.isfinite(market_windows).all(axis=1)
    asset_centered = asset_windows - asset_windows.mean(axis=1, keepdims=True)
    market_centered = market_windows - market_windows.mean(axis=1, keepdims=True)
    market_sum_sq = np.sum(market_centered**2, axis=1)
    beta = np.divide(
        np.sum(asset_centered * market_centered, axis=1),
        market_sum_sq,
        out=np.full(len(asset_windows), np.nan),
        where=valid & (market_sum_sq > 0.0),
    )
    residuals = asset_centered - beta[:, None] * market_centered
    residual_sum = np.sum(residuals[:, -formation_window:], axis=1)
    result[first_output:] = np.where(valid & (market_sum_sq > 0.0), residual_sum, np.nan)
    return pd.Series(result, index=asset_returns.index, dtype=float)


def _safe(series: pd.Series) -> pd.Series:
    return series.where(np.isfinite(series), np.nan)


def _feature_series(
    frame: pd.DataFrame, market: pd.DataFrame | None
) -> tuple[dict[str, pd.Series], dict[str, np.ndarray]]:
    """Materialize every formula in the selected catalog for one sorted listing block.

    Each rolling expression ends at the observation session its own catalog entry
    declares: ``t`` for a Formula with no economic skip, ``t - skip`` for one that
    declares a skip.  A row dated ``t`` therefore holds that Formula's value *on*
    session ``t``, computed from information complete at ``close(t)``.

    The previous build shifted every expression by one further session, so a row
    dated ``t`` consumed nothing later than ``t-1`` while presenting itself as
    ``t``.  Trading safety belongs to the execution recipe -- formation at
    ``close(T)``, entry at ``open(T+1)`` -- and a Formula paying for it again made
    two lags where the strategy asked for one.
    """
    close = frame["provider_adjusted_close"].astype(float)
    split_close = frame["close_split_adjusted"].astype(float)
    high = frame["high_split_adjusted"].astype(float)
    low = frame["low_split_adjusted"].astype(float)
    open_ = frame["open_split_adjusted"].astype(float)
    raw_close = frame["close_raw"].astype(float)
    volume = frame["volume_raw"].astype(float)
    returns = np.log(close / close.shift(1))
    # Yahoo can report zero volume for a valid session.  It is missing evidence
    # for log-volume statistics, not a value that can be silently shifted with
    # ``log1p``.  Positive volume keeps the desired additive-constant behavior
    # under a split share-basis rebase; a window containing zero fails closed.
    log_volume = np.log(volume.where(volume > 0.0))

    values: dict[str, pd.Series] = {}
    # Momentum and reversal.
    # Short, unskipped cumulative returns are reversal signals.  Keep them
    # only under the reversal identity; publishing the same window as
    # "momentum" with the opposite sign makes sparse-model coefficients
    # non-identifiable.  Momentum starts at the skip-month horizons below.
    for factor_id, window, skip in (
        ("mom_126_21", 105, 21),
        ("mom_252_21", 231, 21),
    ):
        values[factor_id] = _rolling_stat(returns, window, skip=skip, statistic="sum")
    for factor_id, window in (("rev_5", 5), ("rev_21", 21), ("rev_63", 63)):
        values[factor_id] = -_rolling_stat(returns, window, statistic="sum")
    formation_windows, formation_first = _window_matrix(returns, 231, 21)
    discreteness = np.full(len(frame), np.nan, dtype=np.float64)
    consistency = np.full(len(frame), np.nan, dtype=np.float64)
    if len(formation_windows):
        valid_formation = np.isfinite(formation_windows).all(axis=1)
        cumulative = np.sum(formation_windows, axis=1)
        negative_share = np.mean(formation_windows < 0.0, axis=1)
        positive_share = np.mean(formation_windows > 0.0, axis=1)
        formation_sign = np.where(cumulative >= 0.0, 1.0, -1.0)
        formation_discreteness = np.sign(cumulative) * (negative_share - positive_share)
        formation_consistency = np.mean(
            np.sign(formation_windows) == formation_sign[:, None], axis=1
        )
        discreteness[formation_first:] = np.where(valid_formation, formation_discreteness, np.nan)
        consistency[formation_first:] = np.where(valid_formation, formation_consistency, np.nan)
    values["information_discreteness_252"] = pd.Series(discreteness, index=frame.index)
    values["momentum_consistency_252"] = pd.Series(consistency, index=frame.index)

    # Return distribution and lottery features.
    for factor_id, window in (("vol_21", 21), ("vol_63", 63), ("vol_252", 252)):
        values[factor_id] = _rolling_stat(returns, window, statistic="std") * ANNUALIZATION
    values["downside_vol_63"] = (
        np.sqrt(_rolling_stat(np.minimum(returns, 0.0) ** 2, 63, statistic="mean")) * ANNUALIZATION
    )
    values["skew_63"] = _rolling_stat(returns, 63, statistic="skew")
    values["skew_252"] = _rolling_stat(returns, 252, statistic="skew")
    values["kurt_63"] = _rolling_stat(returns, 63, statistic="kurt")
    values["max_21"] = _rolling_stat(returns, 21, statistic="max")
    values["min_21"] = _rolling_stat(returns, 21, statistic="min")
    top_windows, top_first = _window_matrix(returns, 21, 0)
    top5 = np.full(len(frame), np.nan, dtype=np.float64)
    if len(top_windows):
        valid_top = np.isfinite(top_windows).all(axis=1)
        selected_top = np.full(len(top_windows), np.nan, dtype=np.float64)
        selected_top[valid_top] = np.mean(
            np.partition(top_windows[valid_top], -5, axis=1)[:, -5:], axis=1
        )
        top5[top_first:] = selected_top
    values["top5_return_mean_21"] = pd.Series(top5, index=frame.index)
    # Raw-dollar-volume liquidity plus split-adjusted technical liquidity.
    dollar_volume = raw_close * volume
    # Amihud (2002) illiquidity: the mean |return| per dollar traded over the window's sessions
    # with positive volume, given that at least 80% of the window has one (Amihud admits a
    # stock with 200 of a year's sessions). A zero-volume bar is a session without the ratio,
    # never an infinite one that blanks the name for its whole window (V517).
    ratio = (returns.abs() / dollar_volume).where(dollar_volume > 0.0)
    for factor_id, window in (("amihud_21", 21), ("amihud_252", 252)):
        values[factor_id] = _rolling_observed_mean(
            ratio, window, minimum=math.ceil(AMIHUD_MINIMUM_SHARE * window), defined_from=1
        )
    for factor_id, window in (("dollar_volume_21", 21), ("dollar_volume_252", 252)):
        values[factor_id] = _rolling_stat(dollar_volume, window, statistic="mean")
    values["volume_vol_63"] = _rolling_stat(log_volume, 63, statistic="std")
    high_low_sq = np.log(high / low) ** 2
    beta = high_low_sq + high_low_sq.shift(1)
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    denominator = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (
        (math.sqrt(2.0) - 1.0) * np.sqrt(beta) / denominator - np.sqrt(gamma / denominator)
    ).clip(lower=0.0)
    corwin = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    values["corwin_schultz_spread_21"] = _rolling_stat(corwin, 21, statistic="mean")

    # Price level/trend.
    values["dist_52w_high"] = split_close / _rolling_stat(high, 252, statistic="max") - 1.0
    values["dist_52w_low"] = split_close / _rolling_stat(low, 252, statistic="min") - 1.0
    values["price_to_ma_200"] = (
        split_close / _rolling_stat(split_close, 200, statistic="mean") - 1.0
    )
    values["ma_gap_50_200"] = (
        _rolling_stat(split_close, 50, statistic="mean")
        / _rolling_stat(split_close, 200, statistic="mean")
        - 1.0
    )
    values["drawdown_252"] = split_close / _rolling_stat(split_close, 252, statistic="max") - 1.0
    values["trend_r2_252"] = _rolling_trend_r2(np.log(split_close), 252)

    # OHLC volatility family.
    #
    # A true range is defined against the *prior* close, so the first ordered row
    # has none. ``DataFrame.max`` skips NaN, so the first row used to fall back to
    # its own high-low spread -- a finite number that is not a true range -- and
    # every ATR consumer started one row early on a fabricated value. Screened
    # explicitly rather than left to a skip-NaN default.
    prior_close = split_close.shift()
    true_range = pd.concat(
        [high - low, (high - prior_close).abs(), (low - prior_close).abs()], axis=1
    ).max(axis=1)
    true_range = true_range.where(np.isfinite(prior_close))
    values["parkinson_vol_21"] = (
        np.sqrt(
            _rolling_stat(np.log(high / low) ** 2, 21, statistic="mean") / (4.0 * math.log(2.0))
        )
        * ANNUALIZATION
    )
    gk = (
        0.5 * np.log(high / low) ** 2
        - (2.0 * math.log(2.0) - 1.0) * np.log(split_close / open_) ** 2
    )
    values["garman_klass_vol_21"] = (
        np.sqrt(_rolling_stat(gk, 21, statistic="mean").clip(lower=0.0)) * ANNUALIZATION
    )
    rs = np.log(high / open_) * np.log(high / split_close) + np.log(low / open_) * np.log(
        low / split_close
    )
    values["rogers_satchell_vol_21"] = (
        np.sqrt(_rolling_stat(rs, 21, statistic="mean").clip(lower=0.0)) * ANNUALIZATION
    )
    overnight = np.log(open_ / split_close.shift())
    open_close = np.log(split_close / open_)
    yz_rs = rs
    n = 21
    yz_k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    yz_variance = (
        _rolling_stat(overnight, n, statistic="var")
        + yz_k * _rolling_stat(open_close, n, statistic="var")
        + (1.0 - yz_k) * _rolling_stat(yz_rs, n, statistic="mean")
    )
    values["yang_zhang_vol_21"] = np.sqrt(yz_variance.clip(lower=0.0)) * ANNUALIZATION

    # Price/volume family.
    values["return_autocorr_21"] = _rolling_autocorrelation(returns, 21)
    volume_change = log_volume.diff()
    pair_price = returns.to_numpy(dtype=float)
    pair_volume = volume_change.to_numpy(dtype=float)
    output = np.full(len(frame), np.nan, dtype=float)
    if len(frame) >= 64:
        px = np.lib.stride_tricks.sliding_window_view(pair_price, 63)
        vx = np.lib.stride_tricks.sliding_window_view(pair_volume, 63)
        valid = np.isfinite(px).all(axis=1) & np.isfinite(vx).all(axis=1)
        px_centered = px - px.mean(axis=1, keepdims=True)
        vx_centered = vx - vx.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(px_centered**2, axis=1) * np.sum(vx_centered**2, axis=1))
        corr = np.divide(
            np.sum(px_centered * vx_centered, axis=1),
            denom,
            out=np.full(len(px), np.nan),
            where=denom > 0,
        )
        output[62:] = np.where(valid, corr, np.nan)  # 63 returns ending at the observation
    values["price_volume_corr_63"] = pd.Series(output, index=frame.index)
    values["volume_zscore_21"] = (
        log_volume - _rolling_stat(log_volume, 21, statistic="mean")
    ) / _rolling_stat(log_volume, 21, statistic="std")
    values["volume_trend_slope_63"] = _rolling_linear_slope(log_volume, 63)

    # Technical series.
    gain = returns.clip(lower=0.0)
    loss = -returns.clip(upper=0.0)
    average_gain = _rolling_stat(gain, 14, statistic="mean")
    average_loss = _rolling_stat(loss, 14, statistic="mean")
    values["cutler_rsi_14"] = (100.0 - 100.0 / (1.0 + average_gain / average_loss)).where(
        average_loss != 0.0, 100.0
    )
    mean_20 = _rolling_stat(split_close, 20, statistic="mean")
    std_20 = _rolling_stat(split_close, 20, statistic="std")
    values["bollinger_pctb_20_2"] = (split_close - (mean_20 - 2.0 * std_20)) / (4.0 * std_20)
    low_14 = _rolling_stat(low, 14, statistic="min")
    high_14 = _rolling_stat(high, 14, statistic="max")
    values["stoch_k_14"] = 100.0 * (split_close - low_14) / (high_14 - low_14)
    direction = np.sign(split_close.diff().fillna(0.0))
    signed_volume = direction * volume
    obv_windows, obv_first = _window_matrix(signed_volume, 63, 0)
    obv_slope = np.full(len(frame), np.nan, dtype=np.float64)
    if len(obv_windows):
        valid_obv = np.isfinite(obv_windows).all(axis=1)
        local_obv = np.cumsum(obv_windows, axis=1)
        x = np.linspace(0.0, 1.0, 63, dtype=np.float64)
        centered_x = x - x.mean()
        denominator_x = float(np.dot(centered_x, centered_x))
        slope = np.sum(local_obv * centered_x, axis=1) / denominator_x
        obv_slope[obv_first:] = np.where(valid_obv, slope, np.nan)
    values["obv_slope_63"] = pd.Series(obv_slope, index=frame.index) / _rolling_stat(
        volume.abs(), 63, statistic="sum"
    )
    spread = high - low
    cmf = ((2.0 * split_close - high - low) / spread) * volume
    values["cmf_21"] = _rolling_stat(cmf, 21, statistic="sum") / _rolling_stat(
        volume, 21, statistic="sum"
    )
    # Directional movement is a difference against the prior session, so the
    # first ordered row has neither an up nor a down move. ``Series.where`` sends
    # its NaN comparison down the false branch, which used to publish a directional
    # movement of exactly zero there -- a value, not a missing one. The mature
    # Formula needs 28 ordered rows and now produces its first finite value at
    # exactly that boundary instead of two rows early.
    up = high.diff()
    down = -low.diff()
    observed_move = np.isfinite(up) & np.isfinite(down)
    plus_dm = up.where((up > down) & (up > 0.0), 0.0).where(observed_move)
    minus_dm = down.where((down > up) & (down > 0.0), 0.0).where(observed_move)
    atr = _rolling_stat(true_range, 14, statistic="mean")
    plus_di = 100.0 * _rolling_stat(plus_dm, 14, statistic="mean") / atr
    minus_di = 100.0 * _rolling_stat(minus_dm, 14, statistic="mean") / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    values["directional_strength_14"] = _rolling_stat(dx, 14, statistic="mean")

    # Current-month values are constructed once per time block.  The formula
    # is intentionally evaluated with the final observed close for each month
    # at each daily cutoff; no global look-ahead is used.
    periods = frame["session_date"].dt.to_period("M")
    monthly_last = (
        pd.DataFrame({"period": periods, "close": close}).groupby("period")["close"].last()
    )
    # This is exactly the scalar helper's `close at month end / prior month
    # close - 1`; selecting an already finished historical month cannot look
    # through the current feature-session cutoff.
    month_lookup = np.log(monthly_last / monthly_last.shift(1))
    period_index = pd.PeriodIndex(periods)
    season_12 = pd.Series(
        month_lookup.reindex(period_index - 12).to_numpy(dtype=float),
        index=frame.index,
        dtype=float,
    )
    # The evaluation month belongs to the observation session and the selected
    # month is already twelve months historical, so this Formula carries no
    # session skip at all -- the offset it declares is a calendar one.  Shifting
    # the mapped result would use the prior calendar month on every month's
    # first trading session.
    values["seasonality_12m"] = season_12
    # The newest source event this value actually reads is the final close of the
    # month twelve back, not the observation session's own close.  Recorded here
    # so the persisted input cutoff can state the real event instead of the
    # session offset every other Formula happens to use.
    monthly_last_position = (
        pd.DataFrame({"period": periods, "position": np.arange(len(frame), dtype=np.int64)})
        .groupby("period")["position"]
        .last()
    )
    calendar_source_positions = pd.Series(
        monthly_last_position.reindex(period_index - 12).to_numpy(dtype=float),
        index=frame.index,
        dtype=float,
    )

    # Systematic series, aligned by calendar session.  This is a vectorized
    # per-listing time block and fails closed where a market observation is
    # missing; it never substitutes a local/partial market history.
    if market is not None:
        market_series = market.set_index("session_date")["provider_adjusted_close"].astype(float)
        aligned_market_close = market_series.reindex(frame["session_date"])
        market_returns = np.log(aligned_market_close / aligned_market_close.shift(1))
        for window in (63, 252):
            # ``_paired_metrics`` already lands each statistic on the last session
            # of its own window, which is the observation session.  These Formulas
            # declare no economic skip, so nothing is shifted on top of it.
            metrics = _paired_metrics(returns, market_returns, window)
            if window == 63:
                values["beta_63"] = metrics["beta"]
                values["market_corr_63"] = metrics["corr"]
            else:
                values["beta_252"] = metrics["beta"]
                values["downside_beta_252"] = metrics["downside_beta"]
                values["market_corr_252"] = metrics["corr"]
                values["coskew_252"] = metrics["coskew"]
                values["idio_vol_252"] = metrics["idio_vol"]
                values["idio_skew_252"] = metrics["idio_skew"]
        values["residual_mom_252_21"] = _residual_momentum(returns, market_returns)

    # Calendar-selected Formulas report the ordered position of the source row
    # they actually read, because no session offset expresses it.
    calendar_positions = {"seasonality_12m": calendar_source_positions.to_numpy(dtype=float)}
    return (
        {factor_id: _safe(series) for factor_id, series in values.items()},
        calendar_positions,
    )


class BaseFeatureMaterializer:
    """Listing/time-block materializer with a deliberately separate oracle probe."""

    def __init__(
        self,
        catalog: FeatureCatalog,
        *,
        kernel_registry: FeatureKernelRegistry | None = None,
    ) -> None:
        """Use the qualified catalog and an explicit or installed extension registry.

        Args:
            catalog: Ordered recipes, observation clocks, and maintenance contracts.
            kernel_registry: Extension execution authority; the installed registry when omitted.
        """
        self.catalog = catalog
        self.kernel_registry = (
            kernel_registry if kernel_registry is not None else default_extension_kernel_registry()
        )

    def materialize_listing(
        self,
        *,
        listing_id: str,
        projected_bars: pd.DataFrame,
        market_bars: pd.DataFrame | None,
    ) -> MaterializedFeatureBlock:
        """Compute catalog features and source cutoffs over one listing's ordered history.

        Args:
            listing_id: Listing whose source history is projected.
            projected_bars: Required projected source columns with unique session dates.
                Rows are sorted stably by session before calculation.
            market_bars: Market-reference history for market-dependent formulas, if available.

        Returns:
            Feature rows with per-formula source cutoffs and separate ineligibility rows.
            Nonfinite results become missing values; missing market input and insufficient
            history retain distinct reasons.

        Raises:
            ValueError: Required source columns are missing, session dates are duplicated,
                an extension is unregistered, or its result violates the registry's
                row-count contract.
        """
        missing = set(_REQUIRED_COLUMNS) - set(projected_bars.columns)
        if missing:
            raise ValueError(f"projected bars are missing required columns: {sorted(missing)}")
        # A field only an extension's formula requires (the as-traded ones, V395) rides along
        # when the source carries it; the core series read the required columns alone.
        carried = sorted(
            {field for spec in self.catalog.factors for field in spec.required_fields}.difference(
                _REQUIRED_COLUMNS
            ).intersection(projected_bars.columns)
        )
        frame = projected_bars.loc[:, [*_REQUIRED_COLUMNS, *carried]].copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"], utc=False)
        frame = frame.sort_values("session_date", kind="mergesort").reset_index(drop=True)
        if frame["session_date"].duplicated().any():
            raise ValueError("projected bars contain duplicate sessions")
        series, calendar_source_positions = _feature_series(frame, market_bars)
        # The core bundle is materialized as one vectorized block because its
        # factors share intermediates.  Anything else the catalog asks for is
        # resolved by the ref it declares, through the registry -- so a catalog
        # can name a factor this module has never heard of, and this module
        # still names no factor.  An unregistered ref fails closed in `resolve`.
        extensions = tuple(spec for spec in self.catalog.factors if spec.factor_id not in series)
        if extensions:
            kernel_source = frame.assign(listing_id=listing_id)
            for spec in extensions:
                series[spec.factor_id] = _safe(self.kernel_registry.compute(kernel_source, spec))

        output = pd.DataFrame(
            {
                "listing_id": listing_id,
                "session_date": frame["session_date"].dt.date,
            }
        )
        session_labels = frame["session_date"].dt.strftime("%Y-%m-%d").to_numpy()
        # The latest source session each Formula's value on this row consumed.
        # One token per declared skip class rather than the two the previous
        # build hard-coded: the qualified classes are a property of the installed
        # catalog, and a guard naming them by value refuses a lawful catalog for
        # the wrong reason.
        calendar_ids = self.catalog.calendar_factor_ids
        skip_tokens = {
            value: f"__AL_SKIP_{value}__"
            for value in sorted(
                {
                    formula_skip_sessions(spec)
                    for spec in self.catalog.factors
                    if spec.factor_id not in calendar_ids
                }
            )
        }
        # A calendar-selected Formula gets its own token: its newest source event
        # is the final close of a month twelve back, and writing the observation
        # session there would claim a source row it never read.
        calendar_tokens = {
            factor_id: f"__AL_CAL_{position}__"
            for position, factor_id in enumerate(sorted(calendar_ids))
        }
        cutoff_template = (
            "{"
            + ",".join(
                json.dumps(spec.factor_id, ensure_ascii=False, separators=(",", ":"))
                + ":"
                + (
                    calendar_tokens[spec.factor_id]
                    if spec.factor_id in calendar_ids
                    else skip_tokens[formula_skip_sessions(spec)]
                )
                for spec in self.catalog.factors
            )
            + "}"
        )
        cutoffs: list[str] = []
        for position in range(len(frame)):
            rendered = cutoff_template
            for skip, token in skip_tokens.items():
                consumed = position - skip
                rendered = rendered.replace(
                    token, f'"{session_labels[consumed]}"' if consumed >= 0 else "null"
                )
            for factor_id, token in calendar_tokens.items():
                source = calendar_source_positions.get(factor_id)
                value = None if source is None else source[position]
                rendered = rendered.replace(
                    token,
                    "null"
                    if value is None or not np.isfinite(value)
                    else f'"{session_labels[int(value)]}"',
                )
            cutoffs.append(rendered)
        output["input_cutoffs_json"] = cutoffs
        ineligibility_frames: list[pd.DataFrame] = []
        session_dates = frame["session_date"].dt.date.to_numpy()
        for spec in self.catalog.factors:
            factor = spec.factor_id
            current = series[factor]
            output[factor] = current
            observations = min(len(frame), spec.minimum_observations)
            market_missing = factor in MARKET_DEPENDENT_FACTOR_IDS and market_bars is None
            positions = np.flatnonzero(current.isna().to_numpy())
            if not len(positions):
                continue
            if market_missing:
                reasons = np.full(len(positions), REASON_MARKET_ALIGNMENT, dtype=object)
            else:
                reasons = np.where(
                    positions + 1 < spec.minimum_observations,
                    REASON_INSUFFICIENT_HISTORY,
                    REASON_ZERO_DENOMINATOR,
                )
            ineligibility_frames.append(
                pd.DataFrame(
                    {
                        "listing_id": listing_id,
                        "session_date": session_dates[positions],
                        "factor_id": factor,
                        "reason": reasons,
                        "observation_count": np.minimum(positions + 1, observations),
                    }
                )
            )
        ineligibility = (
            pd.concat(ineligibility_frames, ignore_index=True)
            if ineligibility_frames
            else pd.DataFrame(
                columns=("listing_id", "session_date", "factor_id", "reason", "observation_count")
            )
        )
        return MaterializedFeatureBlock(output, ineligibility)

    def oracle_at(
        self,
        *,
        factor_id: str,
        feature_session: date,
        projected_bars: pd.DataFrame,
        market_bars: pd.DataFrame,
    ) -> FormulaResult:
        """Compatibility probe for callers migrating from the formal oracle.

        The desktop formulas deliberately differ from the tracked 60-item
        registry.  Independent scalar/NumPy references live in tests; this
        method only projects the local contract into the old ``FormulaResult``
        shape for existing diagnostic callers.

        Args:
            factor_id: Catalog factor to project through the compatibility probe.
            feature_session: Feature observation session to inspect.
            projected_bars: Listing history supplied to the local materializer.
            market_bars: Market-reference history supplied to the local materializer.

        Returns:
            Legacy result projection of the local feature value or its ineligibility reason.

        Raises:
            KeyError: A required history column or requested catalog factor is absent.
            ValueError: Source history or an extension fails materialization qualification.
        """
        asset = projected_bars.loc[
            pd.to_datetime(projected_bars["session_date"]).dt.date <= feature_session
        ]
        market = market_bars.loc[
            pd.to_datetime(market_bars["session_date"]).dt.date <= feature_session
        ]
        block = self.materialize_listing(
            listing_id="oracle",
            projected_bars=asset,
            market_bars=market,
        )
        row = block.values.loc[block.values["session_date"] == feature_session]
        if row.empty:
            return FormulaResult(
                value=None, reason=REASON_INSUFFICIENT_HISTORY, observation_count=0
            )
        value = row.iloc[-1][factor_id]
        if pd.notna(value):
            return FormulaResult(
                value=float(value),
                reason=None,
                observation_count=len(asset),
            )
        fact = block.ineligibility.loc[
            (block.ineligibility["session_date"] == feature_session)
            & (block.ineligibility["factor_id"] == factor_id)
        ]
        if fact.empty:
            return FormulaResult(
                value=None,
                reason=REASON_ZERO_DENOMINATOR,
                observation_count=len(asset),
            )
        latest = fact.iloc[-1]
        return FormulaResult(
            value=None,
            reason=str(latest["reason"]),
            observation_count=int(latest["observation_count"]),
        )
