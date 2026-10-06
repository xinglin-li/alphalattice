"""Benchmark-relative path statistics, computed in one place.

Beta, tracking error, information ratio, Sharpe and the zero-cash-rate Jensen
alpha were written three times -- in ``evaluation/benchmark.py``,
``evaluation/oos.py`` and ``regularization/evaluation.py`` -- to the same
formulas, from the same two arrays, with the same ``math.inf`` guard on a
degenerate tracking error. Three copies of arithmetic nobody disagrees about is
not redundancy that protects anything: it is three chances for one of them to be
corrected and the other two left behind, and a reader who finds a discrepancy
has no way to tell which is the intended one.

They are also the statistics a risk study needs *inside* its own comparison
rows. "Did the risk estimator lower volatility, or did it give up too much
return" is answered by decomposing a Sharpe difference into its two legs, and
that is not answerable from a campaign whose measurement contract carries
neither.

Simple returns throughout, not log. The active return of a portfolio against a
benchmark is the difference of their simple returns; differencing log returns
gives the log of the wealth *ratio*, which is a different quantity that agrees
only in the limit of small moves and is not what a tracking error means.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .metrics import annualized_volatility

type FloatArray = npt.NDArray[np.float64]

TRADING_SESSIONS_PER_YEAR = 252.0


class ActiveMetricsError(ValueError):
    """Stable fail-closed boundary for an unusable benchmark pairing."""


@dataclass(frozen=True, slots=True)
class ActivePathMetrics:
    """One portfolio path measured against one benchmark path."""

    beta: float
    tracking_error: float
    information_ratio: float
    sharpe: float
    zero_cash_rate_jensen_alpha: float


def active_path_metrics(
    *, portfolio_simple: FloatArray, benchmark_simple: FloatArray
) -> ActivePathMetrics:
    """Beta, tracking error, information ratio, Sharpe and Jensen alpha.

    ``information_ratio`` is ``math.inf`` when the tracking error is exactly
    zero, which is the convention all three previous copies already used: a
    portfolio that reproduces its benchmark session for session has no active
    risk, and dividing its zero active return by that is not a finite number
    anyone should invent a value for.
    """
    portfolio = np.asarray(portfolio_simple, dtype=np.float64)
    benchmark = np.asarray(benchmark_simple, dtype=np.float64)
    if portfolio.shape != benchmark.shape or portfolio.ndim != 1 or portfolio.size < 2:
        raise ActiveMetricsError("portfolio_backtesting.active_axis_invalid")
    benchmark_variance = float(np.var(benchmark, ddof=1))
    # Two checks, because the obvious one is not enough. All three previous
    # copies guarded ``variance <= 0`` and a *constant* benchmark path does not
    # trip it: summing two hundred copies of one float and dividing does not
    # recover that float exactly, so the deviations are around 1e-20, the
    # variance is around 1e-40, and beta comes out at 1e20 without a word.
    # Peak-to-peak is exact and needs no invented tolerance.
    if not math.isfinite(benchmark_variance) or benchmark_variance <= 0.0:
        raise ActiveMetricsError("portfolio_backtesting.active_benchmark_degenerate")
    if float(np.ptp(benchmark)) == 0.0:
        raise ActiveMetricsError("portfolio_backtesting.active_benchmark_constant")
    active = portfolio - benchmark
    tracking_error = annualized_volatility(active)
    beta = float(np.cov(portfolio, benchmark, ddof=1)[0, 1] / benchmark_variance)
    deviation = float(np.std(portfolio, ddof=1))
    return ActivePathMetrics(
        beta=beta,
        tracking_error=tracking_error,
        information_ratio=(
            float(np.mean(active) * TRADING_SESSIONS_PER_YEAR / tracking_error)
            if tracking_error
            else math.inf
        ),
        sharpe=(
            float(np.mean(portfolio) / deviation * math.sqrt(TRADING_SESSIONS_PER_YEAR))
            if deviation
            else math.inf
        ),
        zero_cash_rate_jensen_alpha=float(
            (np.mean(portfolio) - beta * np.mean(benchmark)) * TRADING_SESSIONS_PER_YEAR
        ),
    )


__all__ = [
    "TRADING_SESSIONS_PER_YEAR",
    "ActiveMetricsError",
    "ActivePathMetrics",
    "active_path_metrics",
]
