"""Inert single-session observation kernels used by development overlays.

These formulas describe facts observable through the named session close. They
are installed extension methods, not Base Panel activations: adding or changing
one therefore rotates its family identity without rotating the current base
materializer closure. Temporal lags and rolling transforms remain consumers'
work and are deliberately absent here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import NO_ECONOMIC_SKIP
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

SESSION_OBSERVATION_METHOD_FAMILY: Final = "SESSION_OBSERVATION"
_INTERNAL_METHOD_REFERENCE: Final = "urn:alphalattice:research:session-observation-formulas"
_EXTREME_BAND: Final = 0.999
_RUN_WINDOW: Final = 21
_HIGH_WINDOW: Final = 252

SESSION_OBSERVATION_FACTOR_IDS: Final = (
    "at_own_high_share_63",
    "close_to_close",
    "gap",
    "gap_amplitude",
    "high_extension",
    "intraday",
    "intraday_amplitude",
    "low_extension",
    "previous_close_excursion_intraday_adjusted_square",
    "previous_close_excursion_intraday_cross",
    "previous_close_excursion_scaled_span_square",
    "range_position",
    "return_run_length_21",
    "session_span",
    "sessions_since_252_high",
)

SESSION_OBSERVATION_IMPLEMENTATION_IDS: Final = {
    factor_id: f"factor.desktop.experimental.{factor_id}.v1"
    for factor_id in SESSION_OBSERVATION_FACTOR_IDS
}

SESSION_OBSERVATION_DECLARATIONS: Final[Mapping[str, Mapping[str, str]]] = {
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["gap"]: {
        "algorithm": "log(open[t]/close[t-1])",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["intraday"]: {
        "algorithm": "log(close[t]/open[t])",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["close_to_close"]: {
        "algorithm": "log(close[t]/close[t-1])",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["high_extension"]: {
        "algorithm": "log(high[t]/close[t-1])",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["low_extension"]: {
        "algorithm": "log(low[t]/close[t-1])",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["session_span"]: {
        "algorithm": "abs(log(high[t]/close[t-1]))+abs(log(low[t]/close[t-1]))",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["previous_close_excursion_scaled_span_square"]: {
        "algorithm": "session_span^2/(4*log(2)); not the Parkinson high-low estimator",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["previous_close_excursion_intraday_cross"]: {
        "algorithm": "h*(h+intraday)+l*(l+intraday), h/l measured from prior close",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["previous_close_excursion_intraday_adjusted_square"]: {
        "algorithm": "session_span^2/2-(2*log(2)-1)*intraday^2; prior-close excursion",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["gap_amplitude"]: {
        "algorithm": "abs(log(open[t]/close[t-1]))",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["intraday_amplitude"]: {
        "algorithm": "abs(log(close[t]/open[t]))",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["range_position"]: {
        "algorithm": "abs(low_extension)/session_span",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["return_run_length_21"]: {
        "algorithm": "signed same-sign close-return run length, bounded to 21",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["at_own_high_share_63"]: {
        "algorithm": "mean(close >= 0.999*trailing_high_21, trailing 63)",
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["sessions_since_252_high"]: {
        "algorithm": "sessions since first occurrence of maximum close in trailing 252",
    },
}


def _spec(
    factor_id: str,
    *,
    family: FactorFamily,
    formula: str,
    window: int,
    rows: int,
    fields: tuple[str, ...],
    convention: str,
) -> FactorSpec:
    return FactorSpec(
        factor_id=factor_id,
        family=family,
        formula_ref=SESSION_OBSERVATION_IMPLEMENTATION_IDS[factor_id],
        formula=formula,
        window_sessions=window,
        lag_sessions=NO_ECONOMIC_SKIP,
        return_convention=convention,
        required_fields=tuple(sorted(fields)),
        literature_sources=(_INTERNAL_METHOD_REFERENCE,),
        minimum_observations=rows,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        core_anchor=False,
    )


def session_observation_factor_specs() -> tuple[FactorSpec, ...]:
    """Describe session returns, excursions, and bounded own-history state recipes.

    Returns:
        Specifications sorted by factor ID, preserving each price basis and source-row minimum.
        Recipes retain their declared source fields, economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    specs = (
        _spec(
            "gap",
            family=FactorFamily.REVERSAL,
            formula="log(open[t]/close[t-1])",
            window=1,
            rows=2,
            fields=("close_split_adjusted", "open_split_adjusted"),
            convention="split-adjusted-open-close",
        ),
        _spec(
            "intraday",
            family=FactorFamily.REVERSAL,
            formula="log(close[t]/open[t])",
            window=1,
            rows=1,
            fields=("close_split_adjusted", "open_split_adjusted"),
            convention="split-adjusted-open-close",
        ),
        _spec(
            "close_to_close",
            family=FactorFamily.REVERSAL,
            formula="log(close[t]/close[t-1])",
            window=1,
            rows=2,
            fields=("close_split_adjusted",),
            convention="split-adjusted-close",
        ),
        *(
            _spec(
                factor_id,
                family=FactorFamily.VOLATILITY,
                formula=formula,
                window=1,
                rows=minimum_rows,
                fields=fields,
                convention="split-adjusted-previous-close-session-excursion",
            )
            for factor_id, formula, fields, minimum_rows in (
                (
                    "high_extension",
                    "log(high[t]/close[t-1])",
                    ("close_split_adjusted", "high_split_adjusted"),
                    2,
                ),
                (
                    "low_extension",
                    "log(low[t]/close[t-1])",
                    ("close_split_adjusted", "low_split_adjusted"),
                    2,
                ),
                (
                    "session_span",
                    "abs(log(high[t]/close[t-1]))+abs(log(low[t]/close[t-1]))",
                    ("close_split_adjusted", "high_split_adjusted", "low_split_adjusted"),
                    2,
                ),
                (
                    "previous_close_excursion_scaled_span_square",
                    "session_span^2/(4*log(2))",
                    ("close_split_adjusted", "high_split_adjusted", "low_split_adjusted"),
                    2,
                ),
                (
                    "previous_close_excursion_intraday_cross",
                    "h*(h+intraday)+l*(l+intraday), h/l measured from close[t-1]",
                    (
                        "close_split_adjusted",
                        "high_split_adjusted",
                        "low_split_adjusted",
                        "open_split_adjusted",
                    ),
                    2,
                ),
                (
                    "previous_close_excursion_intraday_adjusted_square",
                    "session_span^2/2-(2*log(2)-1)*intraday^2",
                    (
                        "close_split_adjusted",
                        "high_split_adjusted",
                        "low_split_adjusted",
                        "open_split_adjusted",
                    ),
                    2,
                ),
                (
                    "gap_amplitude",
                    "abs(log(open[t]/close[t-1]))",
                    ("close_split_adjusted", "open_split_adjusted"),
                    2,
                ),
                (
                    "intraday_amplitude",
                    "abs(log(close[t]/open[t]))",
                    ("close_split_adjusted", "open_split_adjusted"),
                    1,
                ),
            )
        ),
        _spec(
            "range_position",
            family=FactorFamily.TECHNICAL,
            formula="abs(low_extension)/session_span",
            window=1,
            rows=2,
            fields=("close_split_adjusted", "high_split_adjusted", "low_split_adjusted"),
            convention="split-adjusted-previous-close-session-excursion",
        ),
        _spec(
            "return_run_length_21",
            family=FactorFamily.MOMENTUM,
            formula="signed count of consecutive same-sign close returns, bounded at 21",
            window=21,
            rows=22,
            fields=("close_split_adjusted",),
            convention="split-adjusted-close",
        ),
        _spec(
            "at_own_high_share_63",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula="share of 63 sessions within 0.1% of own trailing 21-session high",
            window=83,
            rows=83,
            fields=("close_split_adjusted",),
            convention="split-adjusted-close",
        ),
        _spec(
            "sessions_since_252_high",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula="sessions since the first maximum close in the trailing 252 sessions",
            window=252,
            rows=252,
            fields=("close_split_adjusted",),
            convention="split-adjusted-close",
        ),
    )
    return tuple(sorted(specs, key=lambda value: value.factor_id))


def append_session_observation_values(
    source: pd.DataFrame,
    specification: FactorSpec,
    *,
    previous: pd.Series | None,
    compute: Callable[[pd.DataFrame, FactorSpec], pd.Series],
) -> pd.Series:
    """Append an owned recipe's finite tail to a source-proved observation prefix.

    Args:
        source: Complete listing/session source on the required output row axis.
        specification: Exact recipe requested from the installed registry.
        previous: Immutable, independently source-proved values for the leading
            source rows, or no prefix for an ordinary full computation.
        compute: The installed registry's ordinary compute boundary, retaining
            its ownership, source-field, output-shape and numeric checks.

    Returns:
        Values on the original source index. Only these exact owned recipes
        admit finite reuse; any other recipe or non-prefix axis uses the full
        compute boundary. Every new row receives the original kernel's value
        after its declared finite source history.

    Raises:
        ValueError: The ordinary registry refuses the source or recipe.
    """
    owned = next(
        (value for value in session_observation_factor_specs() if value == specification), None
    )
    if (
        owned is None
        or previous is None
        or previous.empty
        or len(previous) > len(source)
        or previous.dtype != np.dtype(np.float64)
        or not previous.index.equals(source.index[: len(previous)])
        or not {"listing_id", "session_date", *owned.required_fields} <= set(source.columns)
    ):
        return compute(source, specification)
    prefix_size = len(previous)
    if prefix_size == len(source):
        compute(source.iloc[:0], specification)
        return previous.copy()
    ordered_positions = (
        source[["listing_id", "session_date"]]
        .reset_index(drop=True)
        .sort_values(["listing_id", "session_date"], kind="mergesort")
        .index.to_numpy(dtype=np.intp)
    )
    ordered = source.iloc[ordered_positions]
    tails = []
    for positions in ordered.groupby("listing_id", sort=False).indices.values():
        original_positions = ordered_positions[np.asarray(positions, dtype=np.intp)]
        new = np.flatnonzero(original_positions >= prefix_size)
        if not len(new):
            continue
        first = int(new[0])
        if np.any(original_positions[first:] < prefix_size):
            return compute(source, specification)
        tails.append(original_positions[max(0, first - owned.minimum_observations + 1) :])
    tail_positions = np.sort(np.concatenate(tails)) if tails else np.empty(0, dtype=np.intp)
    new_positions = tail_positions[tail_positions >= prefix_size]
    if not np.array_equal(new_positions, np.arange(prefix_size, len(source))):
        return compute(source, specification)
    tail_values = compute(source.iloc[tail_positions], specification).to_numpy(dtype=np.float64)
    values = np.empty(len(source), dtype=np.float64)
    values[:prefix_size] = previous.to_numpy(dtype=np.float64)
    values[new_positions] = tail_values[tail_positions >= prefix_size]
    return pd.Series(values, index=source.index)


def _ordered(source: pd.DataFrame) -> pd.DataFrame:
    return source.assign(_source_position=np.arange(len(source), dtype=np.int64)).sort_values(
        ["listing_id", "session_date"], kind="mergesort"
    )


def _restore(source: pd.DataFrame, ordered: pd.DataFrame, values: pd.Series) -> pd.Series:
    frame = ordered.assign(_value=pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    return pd.Series(
        frame.sort_values("_source_position", kind="mergesort")["_value"].to_numpy(float),
        index=source.index,
    )


def _prices(ordered: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    def numeric(name: str) -> pd.Series:
        if name not in ordered:
            return pd.Series(np.nan, index=ordered.index, dtype=float)
        return pd.to_numeric(ordered[name], errors="coerce").astype(float)

    groups = ordered["listing_id"]
    close = numeric("close_split_adjusted")
    prior_close = close.groupby(groups, sort=False).shift(1)
    open_ = numeric("open_split_adjusted")
    high = numeric("high_split_adjusted")
    low = numeric("low_split_adjusted")
    return close, prior_close, open_, high, low


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = (
        np.isfinite(numerator) & np.isfinite(denominator) & (numerator > 0.0) & (denominator > 0.0)
    )
    return np.log((numerator / denominator).where(valid))


def _components(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    ordered = _ordered(source)
    close, prior, open_, high, low = _prices(ordered)
    high_extension = _safe_log_ratio(high, prior)
    low_extension = _safe_log_ratio(low, prior)
    gap = _safe_log_ratio(open_, prior)
    intraday = _safe_log_ratio(close, open_)
    span = high_extension.abs() + low_extension.abs()
    return ordered, {
        "gap": gap,
        "intraday": intraday,
        "close_to_close": _safe_log_ratio(close, prior),
        "high_extension": high_extension,
        "low_extension": low_extension,
        "session_span": span,
        "previous_close_excursion_scaled_span_square": span**2 / (4.0 * math.log(2.0)),
        "previous_close_excursion_intraday_cross": high_extension * (high_extension + intraday)
        + low_extension * (low_extension + intraday),
        "previous_close_excursion_intraday_adjusted_square": span**2 / 2.0
        - (2.0 * math.log(2.0) - 1.0) * intraday**2,
        "gap_amplitude": gap.abs(),
        "intraday_amplitude": intraday.abs(),
        "range_position": low_extension.abs() / span.where(span > 0.0),
    }


def _component(factor_id: str) -> Callable[[pd.DataFrame, FactorSpec], pd.Series]:
    def compute(source: pd.DataFrame, _specification: FactorSpec) -> pd.Series:
        ordered, values = _components(source)
        return _restore(source, ordered, values[factor_id])

    return compute


def return_run_length_21(source: pd.DataFrame, _specification: FactorSpec) -> pd.Series:
    """Count the signed trailing same-sign return run within 21 sessions.

    Args:
        source: Listing/session rows with split-adjusted close prices.
        _specification: Kernel-protocol recipe argument; this implementation uses its fixed history
            rule.

    Returns:
        The signed run length, capped by the fixed finite return window; zero returns yield zero.
        Results follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """
    ordered = _ordered(source)
    close = pd.to_numeric(ordered["close_split_adjusted"], errors="coerce").astype(float)
    returns = np.log(close / close.groupby(ordered["listing_id"], sort=False).shift(1))
    result = np.full(len(ordered), np.nan, dtype=np.float64)
    for positions in ordered.groupby("listing_id", sort=False).indices.values():
        indexes = np.asarray(positions, dtype=np.intp)
        values = returns.iloc[indexes].to_numpy(dtype=np.float64)
        for position in range(_RUN_WINDOW, len(indexes)):
            window = values[position - _RUN_WINDOW + 1 : position + 1]
            if not np.isfinite(window).all():
                continue
            sign = np.sign(window[-1])
            length = 0.0
            for value in window[::-1]:
                if sign == 0.0 or np.sign(value) != sign:
                    break
                length += 1.0
            result[indexes[position]] = sign * length
    return _restore(source, ordered, pd.Series(result, index=ordered.index))


def at_own_high_share_63(source: pd.DataFrame, _specification: FactorSpec) -> pd.Series:
    """Measure the 63-session share of closes within 0.1% of their trailing 21-session high.

    Args:
        source: Listing/session rows with split-adjusted close prices.
        _specification: Kernel-protocol recipe argument; this implementation uses its fixed history
            rule.

    Returns:
        Mean own-high indicator after complete finite rolling history. Results follow the input row
        axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """
    ordered = _ordered(source)
    groups = ordered["listing_id"]
    close = pd.to_numeric(ordered["close_split_adjusted"], errors="coerce").astype(float)
    high = (
        close.groupby(groups, sort=False)
        .rolling(21, min_periods=21)
        .max()
        .reset_index(level=0, drop=True)
    )
    at_high = (close >= high * _EXTREME_BAND).astype(float).where(close.notna() & high.notna())
    values = (
        at_high.groupby(groups, sort=False)
        .rolling(63, min_periods=63)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return _restore(source, ordered, values)


def sessions_since_252_high(source: pd.DataFrame, _specification: FactorSpec) -> pd.Series:
    """Count sessions since the first maximum in a finite 252-session close window.

    Args:
        source: Listing/session rows with split-adjusted close prices.
        _specification: Kernel-protocol recipe argument; this implementation uses its fixed history
            rule.

    Returns:
        Distance from the earliest window maximum; nonfinite windows are missing. Results follow the
        input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """
    ordered = _ordered(source)
    close = pd.to_numeric(ordered["close_split_adjusted"], errors="coerce").astype(float)
    result = np.full(len(ordered), np.nan, dtype=np.float64)
    for positions in ordered.groupby("listing_id", sort=False).indices.values():
        indexes = np.asarray(positions, dtype=np.intp)
        values = close.iloc[indexes].to_numpy(dtype=np.float64)
        for position in range(_HIGH_WINDOW - 1, len(indexes)):
            window = values[position - _HIGH_WINDOW + 1 : position + 1]
            if np.isfinite(window).all():
                result[indexes[position]] = float(_HIGH_WINDOW - 1 - int(np.argmax(window)))
    return _restore(source, ordered, pd.Series(result, index=ordered.index))


SESSION_OBSERVATION_KERNELS: Final[
    Mapping[str, Callable[[pd.DataFrame, FactorSpec], pd.Series]]
] = {
    **{
        SESSION_OBSERVATION_IMPLEMENTATION_IDS[factor_id]: _component(factor_id)
        for factor_id in SESSION_OBSERVATION_FACTOR_IDS
        if factor_id
        not in {"at_own_high_share_63", "return_run_length_21", "sessions_since_252_high"}
    },
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["at_own_high_share_63"]: at_own_high_share_63,
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["return_run_length_21"]: return_run_length_21,
    SESSION_OBSERVATION_IMPLEMENTATION_IDS["sessions_since_252_high"]: sessions_since_252_high,
}

__all__ = [
    "SESSION_OBSERVATION_DECLARATIONS",
    "SESSION_OBSERVATION_FACTOR_IDS",
    "SESSION_OBSERVATION_IMPLEMENTATION_IDS",
    "SESSION_OBSERVATION_KERNELS",
    "SESSION_OBSERVATION_METHOD_FAMILY",
    "append_session_observation_values",
    "session_observation_factor_specs",
]
