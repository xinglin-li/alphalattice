"""NumPy-only deterministic statistical procedures for WP60C."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
MINIMUM_CORRELATION_OBSERVATIONS: Final = 2


@dataclass(frozen=True, slots=True)
class NeweyWestResult:
    """Mean test statistics with a Newey-West variance estimate."""

    mean: float
    icir: float | None
    lag: int
    t_stat: float | None
    p_value: float


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def average_ranks(values: FloatArray) -> FloatArray:
    """Assign average ranks to ties in a finite vector.

    Args:
        values: One-dimensional values to rank.

    Returns:
        A read-only array of one-based average ranks.

    Raises:
        ValueError: If the vector is empty, nonfinite or not one-dimensional.

    """
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("average ranks require a non-empty finite vector")
    order = np.argsort(source, kind="stable")
    sorted_values = source[order]
    ranks = np.empty(source.size, dtype=np.float64)
    start = 0
    while start < source.size:
        stop = start + 1
        while stop < source.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average
        start = stop
    ranks.setflags(write=False)
    return ranks


def spearman_rank_correlation(left: FloatArray, right: FloatArray) -> float | None:
    """Compute pairwise finite Spearman correlation with average tie ranks.

    Args:
        left: First observation vector.
        right: Second aligned observation vector.

    Returns:
        Correlation, or ``None`` when too few finite pairs or no dispersion exist.

    Raises:
        ValueError: If the input vectors do not have equal one-dimensional shape.

    """
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Spearman inputs must be equal one-dimensional vectors")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < MINIMUM_CORRELATION_OBSERVATIONS:
        return None
    xr = average_ranks(x[mask])
    yr = average_ranks(y[mask])
    xc = xr - float(np.mean(xr))
    yc = yr - float(np.mean(yr))
    denominator = math.sqrt(float(np.dot(xc, xc)) * float(np.dot(yc, yc)))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    result = float(np.dot(xc, yc) / denominator)
    if not math.isfinite(result):
        return None
    return _clean_zero(max(-1.0, min(1.0, result)))


def gross_decile_spread(scores: FloatArray, targets: FloatArray) -> float | None:
    """Compute the top-minus-bottom gross target spread by score rank.

    Args:
        scores: Cross-sectional ranking scores.
        targets: Aligned realized targets.

    Returns:
        Gross spread, or ``None`` when fewer than 100 finite pairs qualify.

    Raises:
        ValueError: If the inputs do not have equal one-dimensional shape.

    """
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("decile spread inputs must have equal one-dimensional shape")
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 100:
        return None
    ranks = average_ranks(x[mask])
    percentiles = (ranks - 0.5) / ranks.size
    bottom = y[mask][percentiles <= 0.1]
    top = y[mask][percentiles >= 0.9]
    if bottom.size == 0 or top.size == 0:
        return None
    result = float(np.mean(top) - np.mean(bottom))
    return _clean_zero(result) if math.isfinite(result) else None


def newey_west_mean_test(values: FloatArray) -> NeweyWestResult:
    """Test a series mean using a deterministic Newey-West lag.

    Args:
        values: Finite observations in time order.

    Returns:
        Mean, lag, information ratio and test results.

    Raises:
        ValueError: If the series is empty, nonfinite or not one-dimensional.

    """
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("Newey-West input must be a non-empty finite vector")
    count = source.size
    mean = float(np.mean(source))
    sample_std = float(np.std(source, ddof=1)) if count > 1 else 0.0
    icir = None
    if sample_std > 0.0 and math.isfinite(sample_std):
        icir = _clean_zero(mean / sample_std)
    lag = min(count - 1, math.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    centered = source - mean
    long_run_variance = float(np.dot(centered, centered) / count)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / count)
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if not math.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return NeweyWestResult(
            mean=_clean_zero(mean),
            icir=icir,
            lag=lag,
            t_stat=None,
            p_value=1.0,
        )
    standard_error = math.sqrt(long_run_variance / count)
    if not math.isfinite(standard_error) or standard_error <= 0.0:
        return NeweyWestResult(
            mean=_clean_zero(mean),
            icir=icir,
            lag=lag,
            t_stat=None,
            p_value=1.0,
        )
    t_stat = mean / standard_error
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return NeweyWestResult(
        mean=_clean_zero(mean),
        icir=icir,
        lag=lag,
        t_stat=_clean_zero(t_stat),
        p_value=max(0.0, min(1.0, p_value)),
    )


def benjamini_yekutieli(
    p_values: tuple[tuple[str, float], ...],
) -> dict[str, float]:
    """Adjust ordered hypothesis probabilities for dependent tests.

    Args:
        p_values: Sorted, unique identifiers paired with finite probabilities.

    Returns:
        Adjusted probability by hypothesis identifier.

    Raises:
        ValueError: If identifiers or probabilities violate the input contract.

    """
    identifiers = tuple(item[0] for item in p_values)
    if identifiers != tuple(sorted(set(identifiers))):
        raise ValueError("BY hypotheses must be sorted and unique")
    if not p_values:
        raise ValueError("BY correction requires at least one hypothesis")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for _, value in p_values):
        raise ValueError("BY p-values must be finite probabilities")
    count = len(p_values)
    harmonic = sum(1.0 / index for index in range(1, count + 1))
    ordered = sorted(p_values, key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * count
    running = 1.0
    for position in range(count - 1, -1, -1):
        rank = position + 1
        candidate = ordered[position][1] * count * harmonic / rank
        running = min(running, candidate)
        adjusted[position] = max(0.0, min(1.0, running))
    return {identifier: adjusted[position] for position, (identifier, _) in enumerate(ordered)}


__all__ = [
    "NeweyWestResult",
    "average_ranks",
    "benjamini_yekutieli",
    "gross_decile_spread",
    "newey_west_mean_test",
    "spearman_rank_correlation",
]
