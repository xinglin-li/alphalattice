"""Closed deterministic implementations for the WP60B factor registry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.kernel.quant.factor_contracts import FactorSpec, finite_factor_value

ANNUALIZATION: Final = math.sqrt(252.0)
REASON_INSUFFICIENT_HISTORY: Final = "insufficient_history"
REASON_MISSING_FIELD: Final = "missing_required_field"
REASON_MARKET_ALIGNMENT: Final = "incomplete_market_alignment"
REASON_ZERO_DENOMINATOR: Final = "zero_denominator"
REASON_NON_FINITE: Final = "non_finite_intermediate"


@dataclass(frozen=True, slots=True)
class FormulaResult:
    """Computed factor value or typed ineligibility with observation count."""

    value: float | None
    reason: str | None
    observation_count: int


def _ineligible(reason: str, observations: int) -> FormulaResult:
    return FormulaResult(value=None, reason=reason, observation_count=max(observations, 0))


def _computed(value: float, observations: int) -> FormulaResult:
    try:
        normalized = finite_factor_value(float(value))
    except (TypeError, ValueError):
        return _ineligible(REASON_NON_FINITE, observations)
    return FormulaResult(value=normalized, reason=None, observation_count=observations)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("session_date", kind="mergesort").reset_index(drop=True)
    return result


def _required(frame: pd.DataFrame, spec: FactorSpec) -> bool:
    return all(
        field in frame.columns and not frame[field].isna().any() for field in spec.required_fields
    )


def _rolling_rows(
    frame: pd.DataFrame,
    *,
    window: int,
    lag: int,
) -> pd.DataFrame | None:
    end = len(frame) - lag
    start = end - window
    if start < 0 or end <= start:
        return None
    return frame.iloc[start:end]


def _rolling_returns(
    frame: pd.DataFrame,
    *,
    field: str,
    window: int,
    lag: int,
) -> pd.Series | None:
    values = frame[field].astype(float)
    returns = values.pct_change(fill_method=None)
    end = len(returns) - lag
    start = end - window
    if start < 1 or end <= start:
        return None
    result = returns.iloc[start:end]
    return None if result.isna().any() else result


def _aligned_returns(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    *,
    window: int,
    lag: int,
) -> pd.DataFrame | None:
    asset_close = asset.set_index("session_date")["close_total_return_adjusted"].astype(float)
    market_close = market.set_index("session_date")["close_total_return_adjusted"].astype(float)
    if asset_close.index.has_duplicates or market_close.index.has_duplicates:
        return None

    end = len(asset_close) - lag
    start = end - window
    if start < 1 or end <= start:
        return None

    required_sessions = asset_close.index[start - 1 : end]
    if len(required_sessions) != window + 1 or not required_sessions.isin(market_close.index).all():
        return None

    asset_window = asset_close.reindex(required_sessions)
    market_window = market_close.reindex(required_sessions)
    if asset_window.isna().any() or market_window.isna().any():
        return None

    result = pd.DataFrame(
        {
            "asset": asset_window.pct_change(fill_method=None).iloc[1:],
            "market": market_window.pct_change(fill_method=None).iloc[1:],
        },
        index=required_sessions[1:],
    )
    return None if len(result) != window or result.isna().any().any() else result


def _sample_beta(asset: np.ndarray, market: np.ndarray) -> float | None:
    market_variance = float(np.var(market, ddof=1))
    if not math.isfinite(market_variance) or market_variance <= 0.0:
        return None
    return float(np.cov(asset, market, ddof=1)[0, 1] / market_variance)


def _residuals(aligned: pd.DataFrame) -> np.ndarray | None:
    x = aligned["market"].to_numpy(dtype=float)
    y = aligned["asset"].to_numpy(dtype=float)
    if float(np.var(x, ddof=1)) <= 0.0:
        return None
    design = np.column_stack((np.ones(len(x), dtype=float), x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        return None
    return np.asarray(y - design @ coefficients, dtype=np.float64)


def _sample_skew(values: np.ndarray) -> float | None:
    count = len(values)
    if count < 3:
        return None
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= 0.0:
        return None
    centered = (values - values.mean()) / standard_deviation
    return float(count / ((count - 1) * (count - 2)) * np.sum(centered**3))


def _sample_excess_kurtosis(values: np.ndarray) -> float | None:
    count = len(values)
    if count < 4:
        return None
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= 0.0:
        return None
    centered = (values - values.mean()) / standard_deviation
    first = count * (count + 1) / ((count - 1) * (count - 2) * (count - 3))
    second = 3.0 * (count - 1) ** 2 / ((count - 2) * (count - 3))
    return float(first * np.sum(centered**4) - second)


def _monthly_returns(frame: pd.DataFrame) -> tuple[tuple[tuple[int, int], float], ...]:
    closes: dict[tuple[int, int], float] = {}
    for session_date, close in zip(
        frame["session_date"].to_list(),
        frame["close_total_return_adjusted"].to_list(),
        strict=True,
    ):
        timestamp = pd.Timestamp(str(session_date))
        closes[(timestamp.year, timestamp.month)] = float(close)
    ordered = tuple(sorted(closes.items()))
    result = []
    for (key, close), (_, previous_close) in zip(
        ordered[1:],
        ordered[:-1],
        strict=True,
    ):
        if previous_close <= 0.0:
            continue
        result.append((key, close / previous_close - 1.0))
    return tuple(result)


def _linear_slope(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    x = np.linspace(0.0, 1.0, len(values), dtype=float)
    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator <= 0.0:
        return None
    return float(np.sum((x - x.mean()) * (values - values.mean())) / denominator)


def _trend_r2(values: np.ndarray) -> float | None:
    slope = _linear_slope(values)
    if slope is None:
        return None
    x = np.linspace(0.0, 1.0, len(values), dtype=float)
    fitted = values.mean() + slope * (x - x.mean())
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 0.0:
        return None
    residual = float(np.sum((values - fitted) ** 2))
    return 1.0 - residual / total


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    change = close.astype(float).diff()
    gain = change.clip(lower=0.0)
    loss = -change.clip(upper=0.0)
    average_gain = _wilder_average(gain, window)
    average_loss = _wilder_average(loss, window)
    strength = average_gain / average_loss
    result = 100.0 - 100.0 / (1.0 + strength)
    return result.where(average_loss != 0.0, 100.0)


def _wilder_average(values: pd.Series, window: int) -> pd.Series:
    source = values.astype(float).to_numpy()
    result = np.full(len(source), np.nan, dtype=np.float64)
    valid_indices = np.flatnonzero(np.isfinite(source))
    if len(valid_indices) < window:
        return pd.Series(result, index=values.index, dtype=float)
    seed_indices = valid_indices[:window]
    seed_position = int(seed_indices[-1])
    previous = float(np.mean(source[seed_indices]))
    result[seed_position] = previous
    for position in valid_indices[window:]:
        previous = (previous * (window - 1) + float(source[position])) / window
        result[position] = previous
    return pd.Series(result, index=values.index, dtype=float)


def _macd_histogram(close: pd.Series) -> pd.Series:
    values = close.astype(float)
    fast = values.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = values.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return (macd - signal) / values


def _bollinger_pctb(close: pd.Series) -> pd.Series:
    values = close.astype(float)
    mean = values.rolling(20, min_periods=20).mean()
    standard_deviation = values.rolling(20, min_periods=20).std(ddof=1)
    denominator = 4.0 * standard_deviation
    return (values - (mean - 2.0 * standard_deviation)) / denominator


def compute_factor(
    spec: FactorSpec,
    asset_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
) -> FormulaResult:
    """Compute one registry-selected scalar without changing either input frame."""
    asset = _clean_frame(asset_frame)
    market = _clean_frame(market_frame)
    if len(asset) < spec.minimum_observations:
        return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
    relevant_asset = asset.iloc[-spec.minimum_observations :]
    if not _required(relevant_asset, spec):
        return _ineligible(REASON_MISSING_FIELD, len(asset))

    factor_id = spec.factor_id
    window = spec.window_sessions
    lag = spec.lag_sessions

    if factor_id.startswith("mom_"):
        close = asset["close_total_return_adjusted"].astype(float)
        start = len(close) - window - 1
        end = len(close) - lag - 1
        if start < 0 or end <= start or close.iloc[start] == 0.0:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(close))
        return _computed(close.iloc[end] / close.iloc[start] - 1.0, window - lag)

    if factor_id.startswith("rev_"):
        returns = _rolling_returns(
            asset,
            field="close_total_return_adjusted",
            window=window,
            lag=lag,
        )
        if returns is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        return _computed(-float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0), len(returns))

    if factor_id in {"vol_21", "vol_63", "vol_252"}:
        returns = _rolling_returns(
            asset,
            field="close_total_return_adjusted",
            window=window,
            lag=lag,
        )
        if returns is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        return _computed(float(returns.std(ddof=1)) * ANNUALIZATION, len(returns))

    if factor_id == "downside_vol_63":
        returns = _rolling_returns(
            asset,
            field="close_total_return_adjusted",
            window=window,
            lag=lag,
        )
        if returns is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
        return _computed(float(np.sqrt(np.mean(downside**2))) * ANNUALIZATION, len(returns))

    if factor_id in {
        "beta_252",
        "beta_63",
        "downside_beta_252",
        "market_corr_252",
        "market_corr_63",
        "coskew_252",
        "idio_skew_252",
        "idio_vol_252",
        "residual_mom_252_21",
    }:
        if (
            "close_total_return_adjusted" not in market.columns
            or market["close_total_return_adjusted"].iloc[-spec.minimum_observations :].isna().any()
        ):
            return _ineligible(REASON_MISSING_FIELD, 0)
        aligned_window = window - lag if factor_id == "residual_mom_252_21" else window
        aligned = _aligned_returns(asset, market, window=aligned_window, lag=lag)
        if aligned is None:
            return _ineligible(REASON_MARKET_ALIGNMENT, 0)
        x = aligned["asset"].to_numpy(dtype=float)
        y = aligned["market"].to_numpy(dtype=float)
        if factor_id.startswith("beta_"):
            value = _sample_beta(x, y)
        elif factor_id == "downside_beta_252":
            mask = y < 0.0
            value = _sample_beta(x[mask], y[mask]) if int(mask.sum()) >= 2 else None
        elif factor_id.startswith("market_corr_"):
            value = (
                None
                if np.std(x, ddof=1) <= 0.0 or np.std(y, ddof=1) <= 0.0
                else float(np.corrcoef(x, y)[0, 1])
            )
        elif factor_id == "coskew_252":
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            denominator = float(np.std(x, ddof=1) * np.var(y, ddof=1))
            value = (
                None
                if denominator <= 0.0
                else float(np.mean(x_centered * y_centered**2) / denominator)
            )
        else:
            residuals = _residuals(aligned)
            if residuals is None:
                value = None
            elif factor_id == "idio_vol_252":
                value = float(np.std(residuals, ddof=1)) * ANNUALIZATION
            elif factor_id == "idio_skew_252":
                value = _sample_skew(residuals)
            else:
                value = float(np.prod(1.0 + residuals) - 1.0)
        if value is None:
            return _ineligible(REASON_ZERO_DENOMINATOR, len(aligned))
        return _computed(value, len(aligned))

    if factor_id in {"skew_63", "skew_252", "kurt_63"}:
        returns = _rolling_returns(
            asset,
            field="close_total_return_adjusted",
            window=window,
            lag=lag,
        )
        if returns is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        return_values = returns.to_numpy(dtype=float)
        value = (
            _sample_excess_kurtosis(return_values)
            if factor_id == "kurt_63"
            else _sample_skew(return_values)
        )
        if value is None:
            return _ineligible(REASON_ZERO_DENOMINATOR, len(returns))
        return _computed(value, len(returns))

    if factor_id in {"max_21", "min_21", "top5_return_mean_21"}:
        returns = _rolling_returns(
            asset,
            field="close_total_return_adjusted",
            window=window,
            lag=lag,
        )
        if returns is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        lottery_values = returns.to_numpy(dtype=float)
        if factor_id == "max_21":
            value = float(np.max(lottery_values))
        elif factor_id == "min_21":
            value = float(np.min(lottery_values))
        else:
            value = float(np.mean(np.sort(lottery_values)[-5:]))
        return _computed(value, len(lottery_values))

    if factor_id in {"amihud_21", "amihud_252"}:
        rows = _rolling_rows(asset, window=window + 1, lag=lag)
        if rows is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        returns = rows["close_total_return_adjusted"].astype(float).pct_change().iloc[1:]
        dollar_volume = (
            rows["close_raw"].astype(float).iloc[1:] * rows["volume_raw"].astype(float).iloc[1:]
        )
        if (dollar_volume <= 0.0).any():
            return _ineligible(REASON_ZERO_DENOMINATOR, len(returns))
        return _computed(float((returns.abs() / dollar_volume).mean()), len(returns))

    if factor_id in {"dollar_volume_21", "dollar_volume_252"}:
        rows = _rolling_rows(asset, window=window, lag=lag)
        if rows is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        dollar_values = (
            rows["close_raw"].astype(float) * rows["volume_raw"].astype(float)
        ).to_numpy(dtype=float)
        return _computed(float(np.mean(dollar_values)), len(dollar_values))

    if factor_id == "zero_return_days_21":
        returns = _rolling_returns(
            asset,
            field="close_total_return_adjusted",
            window=window,
            lag=lag,
        )
        if returns is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        return _computed(float((returns == 0.0).sum()), len(returns))

    if factor_id == "roll_spread_63":
        rows = _rolling_rows(asset, window=window + 1, lag=lag)
        if rows is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        price_changes = rows["close_split_adjusted"].astype(float).diff().dropna().to_numpy()
        covariance = float(np.cov(price_changes[1:], price_changes[:-1], ddof=1)[0, 1])
        mean_price = float(rows["close_split_adjusted"].mean())
        if mean_price <= 0.0:
            return _ineligible(REASON_ZERO_DENOMINATOR, len(price_changes))
        return _computed(
            2.0 * math.sqrt(max(-covariance, 0.0)) / mean_price,
            len(price_changes),
        )

    if factor_id == "corwin_schultz_spread_21":
        rows = _rolling_rows(asset, window=window + 1, lag=lag)
        if rows is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        spread_high = rows["high_split_adjusted"].to_numpy(dtype=float)
        spread_low = rows["low_split_adjusted"].to_numpy(dtype=float)
        if np.any(spread_low <= 0.0):
            return _ineligible(REASON_ZERO_DENOMINATOR, len(rows))
        squared = np.log(spread_high / spread_low) ** 2
        beta = squared[1:] + squared[:-1]
        gamma = (
            np.log(
                np.maximum(spread_high[1:], spread_high[:-1])
                / np.minimum(spread_low[1:], spread_low[:-1])
            )
            ** 2
        )
        denominator = 3.0 - 2.0 * math.sqrt(2.0)
        alpha = (math.sqrt(2.0) - 1.0) * np.sqrt(beta) / denominator - np.sqrt(gamma / denominator)
        alpha = np.maximum(alpha, 0.0)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
        return _computed(float(np.mean(spread)), len(spread))

    if factor_id == "volume_vol_63":
        rows = _rolling_rows(asset, window=window, lag=lag)
        if rows is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        log_volume = np.log1p(rows["volume_raw"].to_numpy(dtype=float))
        return _computed(float(np.std(log_volume, ddof=1)), len(log_volume))

    if factor_id in {
        "dist_52w_high",
        "dist_52w_low",
        "price_to_ma_200",
        "ma_gap_50_200",
        "trend_r2_252",
        "drawdown_252",
    }:
        rows = _rolling_rows(asset, window=window, lag=lag)
        if rows is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        trend_close = rows["close_split_adjusted"].to_numpy(dtype=float)
        latest = float(trend_close[-1])
        if factor_id == "dist_52w_high":
            denominator = float(rows["high_split_adjusted"].max())
            value = None if denominator <= 0.0 else latest / denominator - 1.0
        elif factor_id == "dist_52w_low":
            denominator = float(rows["low_split_adjusted"].min())
            value = None if denominator <= 0.0 else latest / denominator - 1.0
        elif factor_id == "price_to_ma_200":
            denominator = float(np.mean(trend_close))
            value = None if denominator <= 0.0 else latest / denominator - 1.0
        elif factor_id == "ma_gap_50_200":
            slow = float(np.mean(trend_close))
            value = None if slow <= 0.0 else float(np.mean(trend_close[-50:])) / slow - 1.0
        elif factor_id == "trend_r2_252":
            value = _trend_r2(np.log(trend_close)) if np.all(trend_close > 0.0) else None
        else:
            peak = float(np.max(trend_close))
            value = None if peak <= 0.0 else latest / peak - 1.0
        if value is None:
            return _ineligible(REASON_ZERO_DENOMINATOR, len(rows))
        return _computed(value, len(rows))

    if factor_id in {
        "atr_norm_14",
        "parkinson_vol_21",
        "garman_klass_vol_21",
        "rogers_satchell_vol_21",
        "yang_zhang_vol_21",
    }:
        extra = 1 if factor_id in {"atr_norm_14", "yang_zhang_vol_21"} else 0
        rows = _rolling_rows(asset, window=window + extra, lag=lag)
        if rows is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        range_high = rows["high_split_adjusted"].to_numpy(dtype=float)
        range_low = rows["low_split_adjusted"].to_numpy(dtype=float)
        if np.any(range_low <= 0.0):
            return _ineligible(REASON_ZERO_DENOMINATOR, len(rows))
        if factor_id == "parkinson_vol_21":
            value = math.sqrt(
                float(np.mean(np.log(range_high / range_low) ** 2)) / (4.0 * math.log(2.0))
            )
        elif factor_id == "atr_norm_14":
            visible = asset.iloc[: len(asset) - lag]
            atr_high = visible["high_split_adjusted"].astype(float)
            atr_low = visible["low_split_adjusted"].astype(float)
            atr_close_series = visible["close_split_adjusted"].astype(float)
            true_range = pd.concat(
                [
                    atr_high - atr_low,
                    (atr_high - atr_close_series.shift()).abs(),
                    (atr_low - atr_close_series.shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_value = _wilder_average(true_range, 14).iloc[-1]
            denominator = float(atr_close_series.iloc[-1])
            value = (
                None if denominator <= 0.0 or pd.isna(atr_value) else float(atr_value) / denominator
            )
            if value is None:
                return _ineligible(REASON_ZERO_DENOMINATOR, len(visible))
            return _computed(value, 14)
        else:
            open_ = rows["open_split_adjusted"].to_numpy(dtype=float)
            ohlc_close = rows["close_split_adjusted"].to_numpy(dtype=float)
            if np.any(open_ <= 0.0) or np.any(ohlc_close <= 0.0):
                return _ineligible(REASON_ZERO_DENOMINATOR, len(rows))
            if factor_id == "garman_klass_vol_21":
                variance = (
                    0.5 * np.log(range_high / range_low) ** 2
                    - (2.0 * math.log(2.0) - 1.0) * np.log(ohlc_close / open_) ** 2
                )
                value = math.sqrt(max(float(np.mean(variance)), 0.0))
            elif factor_id == "rogers_satchell_vol_21":
                variance = np.log(range_high / open_) * np.log(range_high / ohlc_close) + np.log(
                    range_low / open_
                ) * np.log(range_low / ohlc_close)
                value = math.sqrt(max(float(np.mean(variance)), 0.0))
            else:
                overnight = np.log(open_[1:] / ohlc_close[:-1])
                open_close = np.log(ohlc_close[1:] / open_[1:])
                rs = np.log(range_high[1:] / open_[1:]) * np.log(
                    range_high[1:] / ohlc_close[1:]
                ) + np.log(range_low[1:] / open_[1:]) * np.log(range_low[1:] / ohlc_close[1:])
                n = len(overnight)
                k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
                variance = (
                    float(np.var(overnight, ddof=1))
                    + k * float(np.var(open_close, ddof=1))
                    + (1.0 - k) * float(np.mean(rs))
                )
                value = math.sqrt(max(variance, 0.0))
        return _computed(value * ANNUALIZATION, window)

    if factor_id in {
        "price_volume_corr_63",
        "return_autocorr_21",
        "volume_zscore_21",
        "volume_trend_slope_63",
    }:
        if factor_id == "return_autocorr_21":
            autocorr_returns = _rolling_returns(
                asset,
                field="close_total_return_adjusted",
                window=window + 1,
                lag=lag,
            )
            if autocorr_returns is None:
                return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
            left = autocorr_returns.iloc[1:].to_numpy(dtype=float)
            right = autocorr_returns.iloc[:-1].to_numpy(dtype=float)
            value = (
                None
                if np.std(left, ddof=1) <= 0.0 or np.std(right, ddof=1) <= 0.0
                else float(np.corrcoef(left, right)[0, 1])
            )
            observations = len(left)
        elif factor_id == "price_volume_corr_63":
            rows = _rolling_rows(asset, window=window + 1, lag=lag)
            if rows is None:
                return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
            price_volume_returns = (
                rows["close_total_return_adjusted"]
                .astype(float)
                .pct_change()
                .iloc[1:]
                .to_numpy(dtype=float)
            )
            price_volume_changes = np.diff(np.log(rows["volume_raw"].to_numpy(dtype=float)))
            value = (
                None
                if np.std(price_volume_returns, ddof=1) <= 0.0
                or np.std(price_volume_changes, ddof=1) <= 0.0
                else float(np.corrcoef(price_volume_returns, price_volume_changes)[0, 1])
            )
            observations = len(price_volume_returns)
        else:
            rows = _rolling_rows(asset, window=window, lag=lag)
            if rows is None:
                return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
            price_volume_values = np.log1p(rows["volume_raw"].to_numpy(dtype=float))
            observations = len(price_volume_values)
            if factor_id == "volume_zscore_21":
                standard_deviation = float(np.std(price_volume_values, ddof=1))
                value = (
                    None
                    if standard_deviation <= 0.0
                    else float(
                        (price_volume_values[-1] - price_volume_values.mean()) / standard_deviation
                    )
                )
            else:
                value = _linear_slope(price_volume_values)
        if value is None:
            return _ineligible(REASON_ZERO_DENOMINATOR, observations)
        return _computed(value, observations)

    if factor_id in {"seasonality_12m", "seasonality_same_month_5y"}:
        visible = asset.iloc[: len(asset) - lag]
        monthly = _monthly_returns(visible)
        evaluation_date = pd.Timestamp(str(asset["session_date"].iloc[-1]))
        if factor_id == "seasonality_12m":
            target = (evaluation_date.year - 1, evaluation_date.month)
            selected_monthly = tuple(item for item in monthly if item[0] == target)
        else:
            selected_monthly = tuple(
                item
                for item in monthly
                if item[0][1] == evaluation_date.month and item[0][0] < evaluation_date.year
            )[-5:]
        if len(selected_monthly) < (1 if factor_id == "seasonality_12m" else 5):
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(selected_monthly))
        return _computed(
            float(np.mean([item[1] for item in selected_monthly])),
            len(selected_monthly),
        )

    if factor_id in {
        "information_discreteness_252",
        "momentum_consistency_252",
    }:
        returns = _rolling_returns(
            asset,
            field="close_total_return_adjusted",
            window=window - lag,
            lag=lag,
        )
        if returns is None:
            return _ineligible(REASON_INSUFFICIENT_HISTORY, len(asset))
        discreteness_values = returns.to_numpy(dtype=float)
        cumulative = float(np.prod(1.0 + discreteness_values) - 1.0)
        if factor_id == "momentum_consistency_252":
            sign = 1.0 if cumulative >= 0.0 else -1.0
            value = float(np.mean(np.sign(discreteness_values) == sign))
        else:
            positive = float(np.mean(discreteness_values > 0.0))
            negative = float(np.mean(discreteness_values < 0.0))
            value = math.copysign(1.0, cumulative) * (negative - positive)
        return _computed(value, len(discreteness_values))

    if factor_id in {
        "rsi_14",
        "macd_hist_norm_12_26_9",
        "bollinger_pctb_20_2",
        "stoch_k_14",
        "adx_14",
        "obv_slope_63",
        "cmf_21",
    }:
        visible = asset.iloc[: len(asset) - lag]
        if factor_id == "rsi_14":
            series = _wilder_rsi(visible["close_split_adjusted"], 14)
            value = series.iloc[-1]
            observations = 14
        elif factor_id == "macd_hist_norm_12_26_9":
            series = _macd_histogram(visible["close_split_adjusted"])
            value = series.iloc[-1]
            observations = 35
        elif factor_id == "bollinger_pctb_20_2":
            series = _bollinger_pctb(visible["close_split_adjusted"])
            value = series.iloc[-1]
            observations = 20
        elif factor_id == "stoch_k_14":
            rows = visible.iloc[-14:]
            stoch_low = float(rows["low_split_adjusted"].min())
            stoch_high = float(rows["high_split_adjusted"].max())
            value = (
                None
                if stoch_high <= stoch_low
                else 100.0
                * (float(rows["close_split_adjusted"].iloc[-1]) - stoch_low)
                / (stoch_high - stoch_low)
            )
            observations = len(rows)
        elif factor_id == "obv_slope_63":
            rows = visible.iloc[-63:]
            obv_directions = np.sign(
                rows["close_split_adjusted"].astype(float).diff().fillna(0.0).to_numpy(dtype=float)
            )
            obv = np.cumsum(obv_directions * rows["volume_raw"].to_numpy(dtype=float))
            scale = float(np.sum(np.abs(rows["volume_raw"].to_numpy(dtype=float))))
            value = None if scale <= 0.0 else _linear_slope(obv / scale)
            observations = len(rows)
        elif factor_id == "cmf_21":
            rows = visible.iloc[-21:]
            cmf_high = rows["high_split_adjusted"].to_numpy(dtype=float)
            cmf_low = rows["low_split_adjusted"].to_numpy(dtype=float)
            cmf_close = rows["close_split_adjusted"].to_numpy(dtype=float)
            volume = rows["volume_raw"].to_numpy(dtype=float)
            spread = cmf_high - cmf_low
            denominator = float(np.sum(volume))
            value = (
                None
                if np.any(spread <= 0.0) or denominator <= 0.0
                else float(
                    np.sum(((2.0 * cmf_close - cmf_high - cmf_low) / spread) * volume) / denominator
                )
            )
            observations = len(rows)
        else:
            rows = visible
            adx_high = rows["high_split_adjusted"].astype(float)
            adx_low = rows["low_split_adjusted"].astype(float)
            adx_close = rows["close_split_adjusted"].astype(float)
            up = adx_high.diff()
            down = -adx_low.diff()
            plus_dm = up.where((up > down) & (up > 0.0), 0.0)
            minus_dm = down.where((down > up) & (down > 0.0), 0.0)
            true_range = pd.concat(
                [
                    adx_high - adx_low,
                    (adx_high - adx_close.shift()).abs(),
                    (adx_low - adx_close.shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = _wilder_average(true_range, 14)
            plus_di = 100.0 * _wilder_average(plus_dm, 14) / atr
            minus_di = 100.0 * _wilder_average(minus_dm, 14) / atr
            adx_denominator = plus_di + minus_di
            dx = 100.0 * (plus_di - minus_di).abs() / adx_denominator
            value = _wilder_average(dx, 14).iloc[-1]
            observations = min(len(rows), 28)
        if value is None or pd.isna(value):
            return _ineligible(REASON_ZERO_DENOMINATOR, observations)
        return _computed(float(value), observations)

    raise RuntimeError(f"closed factor dispatch is missing {factor_id}")


__all__ = [
    "REASON_INSUFFICIENT_HISTORY",
    "REASON_MARKET_ALIGNMENT",
    "REASON_MISSING_FIELD",
    "REASON_NON_FINITE",
    "REASON_ZERO_DENOMINATOR",
    "FormulaResult",
    "compute_factor",
]
