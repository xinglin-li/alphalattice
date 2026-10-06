"""Report-unit projection: the same numbers, bucketed three admitted ways.

No metric is defined here. Every row is ``prod(1 + r) - 1`` over a bucket of the
already-published net series -- the identical arithmetic the economic ledger uses
for ``cumulative_net_wealth`` -- so changing the report unit changes how a
result is displayed and never what it is.

The beta-stripped unit subtracts `beta * anchor`, not the anchor. Subtracting the
anchor outright is an *excess* return and calling it beta-stripped would be a
false label: a book with beta 1.19 that is up exactly as much as its benchmark
has negative beta-stripped return, and excess return zero.

`beta` is not computed here. It comes from `active_path_metrics`, the Backtesting
owner, and arrives on the economic ledger already estimated over the whole
materialized path -- never inside the selected window, because a descriptive
window may slice what is displayed and may not move a coefficient.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import numpy.typing as npt

from alphalattice.investment.portfolio_strategy_lab.application.controls import (
    PortfolioReportUnit,
)

type FloatArray = npt.NDArray[np.float64]


class PortfolioReportUnitError(ValueError):
    """Stable refusal for a report-unit projection axis failure."""


def _compound(values: FloatArray) -> float:
    return float(np.prod(1.0 + values) - 1.0)


def project_report_unit_rows(
    *,
    report_unit: PortfolioReportUnit,
    sessions: tuple[date, ...],
    net_simple_returns: FloatArray,
    anchor_simple_returns: FloatArray,
    benchmark_beta: float | None,
) -> tuple[tuple[str, float], ...]:
    """Bucket one window's already-computed returns into the selected unit.

    Returns ``()`` for an empty window rather than a zero row. A report that
    shows ``0.00%`` for a period it has no observations in is worse than one that
    shows nothing, because only the second is obviously empty.
    """
    if len(sessions) != net_simple_returns.size or len(sessions) != anchor_simple_returns.size:
        raise PortfolioReportUnitError("portfolio_reporting.report_unit_axis_invalid")
    if not sessions:
        return ()
    if not np.isfinite(net_simple_returns).all() or not np.isfinite(anchor_simple_returns).all():
        raise PortfolioReportUnitError("portfolio_reporting.report_unit_values_invalid")

    if report_unit == "MONTHLY_BETA_STRIPPED_LEDGER":
        if benchmark_beta is None or not np.isfinite(benchmark_beta):
            # Only this unit needs the coefficient. Refusing here rather than at
            # the ledger keeps a degenerate anchor from disabling the two views
            # that never asked for a beta.
            raise PortfolioReportUnitError("portfolio_reporting.report_unit_requires_beta")
        stripped = np.asarray(
            net_simple_returns - benchmark_beta * anchor_simple_returns, dtype=np.float64
        )
        return _bucketed(
            sessions, stripped, key=lambda value: f"{value.year:04d}-{value.month:02d}"
        )
    if report_unit == "CALENDAR_YEAR_TABLE":
        return _bucketed(sessions, net_simple_returns, key=lambda value: f"{value.year:04d}")

    # SIMPLE_CUMULATIVE: month-end cumulative net wealth relative to the window
    # open, so the last row is exactly the window's total net return.
    rows: list[tuple[str, float]] = []
    monthly = _buckets(sessions, key=lambda value: f"{value.year:04d}-{value.month:02d}")
    for label, _first, stop in monthly:
        rows.append((label, _compound(net_simple_returns[:stop])))
    return tuple(rows)


def _buckets(
    sessions: tuple[date, ...],
    *,
    key: Callable[[date], str],
) -> tuple[tuple[str, int, int], ...]:
    out: list[tuple[str, int, int]] = []
    start = 0
    for index in range(1, len(sessions) + 1):
        if index == len(sessions) or key(sessions[index]) != key(sessions[start]):
            out.append((key(sessions[start]), start, index))
            start = index
    return tuple(out)


def _bucketed(
    sessions: tuple[date, ...],
    values: FloatArray,
    *,
    key: Callable[[date], str],
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (label, _compound(values[first:stop])) for label, first, stop in _buckets(sessions, key=key)
    )


__all__ = ["PortfolioReportUnitError", "project_report_unit_rows"]
