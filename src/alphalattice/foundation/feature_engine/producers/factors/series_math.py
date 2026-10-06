"""Shared causal series mechanics for development Factor method families."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def ordered_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Return a stable listing/session ordering with a reversible row position.

    Args:
        source: Frame with listing and session identities.

    Returns:
        Detached stable listing/session ordering with each row's original position recorded.

    Raises:
        KeyError: A listing or session column is absent.
    """
    return source.assign(_source_position=np.arange(len(source), dtype=int)).sort_values(
        ["listing_id", "session_date"], kind="mergesort"
    )


def finite_positive(values: pd.Series) -> pd.Series:
    """Keep finite positive numeric observations and mark other values missing.

    Args:
        values: Observations to coerce to the numeric domain.

    Returns:
        Values on the same index, with nonnumeric, nonpositive, and nonfinite entries missing.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric) & (numeric > 0.0))


def log_return(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Compute a causal one-row log price return within each listing group.

    Args:
        values: Price observations in each listing's chronological order.
        groups: Listing labels aligned with those observations.

    Returns:
        Log price ratios on the input index; first rows and invalid price pairs are missing.
    """
    prices = finite_positive(values)
    prior = prices.groupby(groups, sort=False).shift(1)
    ratio = prices / prior
    return pd.Series(
        np.log(ratio.where(np.isfinite(ratio) & (ratio > 0.0))).to_numpy(dtype=float),
        index=values.index,
    )


def grouped_pair_rolling_apply(
    left: pd.Series,
    right: pd.Series,
    groups: pd.Series,
    *,
    window: int,
    skip: int,
    function: Callable[[np.ndarray, np.ndarray], float],
) -> pd.Series:
    """Apply a strict finite two-series rolling function per listing.

    ``skip`` is the Formula's declared economic skip, so the window landing on
    observation session ``t`` ends at source row ``t - skip``.

    Args:
        left: First series in each group's chronological order.
        right: Paired series on the same index.
        groups: Listing labels aligned with both series.
        window: Positive number of source observations in each complete window.
        skip: Nonnegative economic skip before the observation row.
        function: Scalar calculation over the two finite source windows.

    Returns:
        Per-row values on the aligned series index. Incomplete or nonfinite windows and
        nonfinite scalar results remain missing; callback failures propagate.
    """
    frame = pd.DataFrame({"left": left, "right": right, "group": groups})
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    for _group, positions in frame.groupby("group", sort=False).groups.items():
        indexes = list(positions)
        x = frame.loc[indexes, "left"].to_numpy(dtype=float)
        y = frame.loc[indexes, "right"].to_numpy(dtype=float)
        result: np.ndarray = np.full(len(indexes), np.nan, dtype=float)
        first = window + skip - 1
        for end in range(first, len(indexes)):
            stop = end - skip + 1
            xs = x[stop - window : stop]
            ys = y[stop - window : stop]
            if np.isfinite(xs).all() and np.isfinite(ys).all():
                value = float(function(xs, ys))
                if np.isfinite(value):
                    result[end] = value
        output.loc[indexes] = result
    return output


def centered_beta(asset: np.ndarray, market: np.ndarray) -> float:
    """Fit the market slope of an asset-return window with an intercept.

    Args:
        asset: Asset observations in the paired window.
        market: Market observations of equal length and order.

    Returns:
        Centered cross-product over centered market sum of squares; flat market windows are missing.

    Raises:
        ValueError: Paired arrays have incompatible lengths or shapes.
    """
    x = asset - float(np.mean(asset))
    y = market - float(np.mean(market))
    denominator = float(np.dot(y, y))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denominator)


__all__ = [
    "centered_beta",
    "finite_positive",
    "grouped_pair_rolling_apply",
    "log_return",
    "ordered_frame",
]
