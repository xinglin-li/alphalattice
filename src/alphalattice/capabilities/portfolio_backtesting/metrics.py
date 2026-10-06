"""Continuous Portfolio path metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import NamedTuple, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    DEFAULT_REPORTING_COST_BPS,
    FloatArray,
    PortfolioBootstrapSettings,
    PortfolioCostPolicy,
    PortfolioTrialMetrics,
    PortfolioWalkForwardError,
    PortfolioWalkForwardResult,
    PortfolioWalkForwardSegmentResult,
)


class PortfolioMetricsWorkspace(Protocol):
    """The minimal workspace surface required for path-level metrics."""

    @property
    def mandate(self) -> PortfolioBootstrapSettings: ...

    @property
    def formation_sessions(self) -> tuple[date, ...]: ...


class PortfolioEconomicMetricSet(BaseModel):  # type: ignore[misc]
    """Raw-simple-return economic metrics for one explicit cost arm.

    This is the shared numerical owner used by development and OOS callers.  A
    Target-Z lane cannot satisfy this contract because the caller must provide
    the gross execution-return path and the realized turnover path separately.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cost_bps: float = Field(ge=0.0)
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe: float
    sortino: float
    maximum_drawdown: float = Field(ge=0.0, le=1.0)
    beta: float
    zero_cash_jensen_alpha: float
    tracking_error: float = Field(ge=0.0)
    information_ratio: float
    benchmark_relative_return: float
    anchor_relative_return: float | None = None


class PortfolioNetReturnMetricSet(BaseModel):  # type: ignore[misc]
    """Return-only metrics on a supplied, already cost-adjusted daily path.

    Sharpe uses zero cash return and Sortino uses zero downside threshold. The
    owner retains its nonfinite Sortino convention; presentation must name that
    absence rather than encode infinity as an observed metric.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cumulative_return: float
    annualized_return: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe: float
    sortino: float
    maximum_drawdown: float = Field(ge=0.0, le=1.0)


VARIANCE_CALIBRATION_METHOD_ID = "REALIZED_TO_PREDICTED_VARIANCE_RATIO"
"""One name for one formula, because there were three of each.

``metrics.py`` computed ``mean(realized**2) / mean(predicted)`` and stored it in a
field called ``predicted_realized_variance_ratio``. ``campaign/measurement.py``
and ``campaign/reporting.py`` computed the *reciprocal* and stored it under the
same name. Downstream, ``regularization`` read one of them into a field called
``realized_to_predicted_variance_ratio`` and compared folds with it. Two of the
three paths therefore reported a number that means the opposite of what the
consumer thought, and nothing in the types could say so.
"""


class VarianceCalibration(NamedTuple):
    """The ratio and the support it was computed on.

    ``support_count`` is not decoration. A formation with no decision-time
    predicted variance -- a hold, which makes no forecast -- contributes to
    neither the numerator nor the denominator, so a ratio without its support
    cannot be compared against another ratio taken over a different number of
    sessions.
    """

    ratio: float | None
    support_count: int
    excluded_count: int
    disposition: str


def realized_to_predicted_variance_ratio(
    *,
    predicted: Sequence[float | None],
    realized_simple: Sequence[float],
    risk_forecast_required: Sequence[bool] | None = None,
) -> VarianceCalibration:
    """``mean(realized**2) / mean(predicted)`` on the common decision-time support.

    Greater than one is Risk *under*-prediction: the book moved more than the
    covariance said it would. Less than one is over-prediction. One is
    calibrated. The direction is stated here and nowhere else, so a consumer
    cannot pick it up from a field name that disagrees with its own arithmetic.

    ``predicted`` accepts ``None`` for a formation that made no forecast. Those
    positions are dropped from *both* sides rather than zero-filled: a zero
    predicted variance would divide, and carrying the realised square of a
    session whose forecast is absent would inflate the numerator against a
    denominator that never saw it.
    """
    if len(predicted) != len(realized_simple) or (
        risk_forecast_required is not None and len(risk_forecast_required) != len(predicted)
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.variance_calibration_axis_invalid")
    if risk_forecast_required is not None and not any(risk_forecast_required):
        if any(value is not None for value in predicted):
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.variance_calibration_closed_form_forecast_present"
            )
        return VarianceCalibration(
            ratio=None,
            support_count=0,
            excluded_count=len(predicted),
            disposition="NOT_APPLICABLE_POLICY_DOES_NOT_CONSUME_RISK_FORECAST",
        )
    support = [
        (float(forecast), float(outcome))
        for forecast, outcome in zip(predicted, realized_simple, strict=True)
        if forecast is not None
    ]
    excluded = len(predicted) - len(support)
    if not support:
        return VarianceCalibration(
            ratio=None,
            support_count=0,
            excluded_count=excluded,
            disposition="MEASURED",
        )
    mean_predicted = float(np.mean([value for value, _ in support]))
    mean_realized_square = float(np.mean([outcome**2 for _, outcome in support]))
    if not np.isfinite(mean_predicted) or mean_predicted <= 0.0:
        return VarianceCalibration(
            ratio=None,
            support_count=len(support),
            excluded_count=excluded,
            disposition="MEASURED",
        )
    return VarianceCalibration(
        ratio=mean_realized_square / mean_predicted,
        support_count=len(support),
        excluded_count=excluded,
        disposition="MEASURED",
    )


def break_even_cost_bps(
    *, gross_simple_returns: FloatArray, one_way_turnovers: FloatArray
) -> float | None:
    """Linear-cost rate that exhausts the path's mean raw-simple return."""
    gross = np.asarray(gross_simple_returns, dtype=np.float64)
    turnover = np.asarray(one_way_turnovers, dtype=np.float64)
    if (
        gross.ndim != 1
        or gross.size < 1
        or turnover.shape != gross.shape
        or not np.isfinite(gross).all()
        or not np.isfinite(turnover).all()
        or np.any(turnover < 0.0)
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.break_even_input_invalid")
    mean_turnover = float(np.mean(turnover))
    return float(np.mean(gross) / mean_turnover * 10_000.0) if mean_turnover > 0.0 else None


def evaluate_raw_simple_return_path(
    *,
    gross_simple_returns: FloatArray,
    one_way_turnovers: FloatArray,
    benchmark_simple_returns: FloatArray,
    anchor_simple_returns: FloatArray | None = None,
    cost_bps: float,
    cost_policy: PortfolioCostPolicy | None = None,
) -> PortfolioEconomicMetricSet:
    """Evaluate one Portfolio cost arm against a same-clock raw SPY path."""
    gross = np.asarray(gross_simple_returns, dtype=np.float64)
    turnover = np.asarray(one_way_turnovers, dtype=np.float64)
    benchmark = np.asarray(benchmark_simple_returns, dtype=np.float64)
    anchor = (
        None
        if anchor_simple_returns is None
        else np.asarray(anchor_simple_returns, dtype=np.float64)
    )
    if (
        gross.ndim != 1
        or gross.size < 2
        or turnover.shape != gross.shape
        or benchmark.shape != gross.shape
        or not np.isfinite(gross).all()
        or not np.isfinite(turnover).all()
        or not np.isfinite(benchmark).all()
        or (anchor is not None and anchor.shape != gross.shape)
        or (anchor is not None and not np.isfinite(anchor).all())
        or np.any(gross <= -1.0)
        or np.any(benchmark <= -1.0)
        or (anchor is not None and np.any(anchor <= -1.0))
        or np.any(turnover < 0.0)
        or not np.isfinite(cost_bps)
        or cost_bps < 0.0
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.economic_metric_axis_invalid")
    policy = cost_policy or PortfolioCostPolicy()
    net = policy.net_simple_returns(
        gross_simple_returns=gross,
        one_way_turnovers=turnover,
        cost_bps=cost_bps,
    )
    if bool(np.any(net <= -1.0)) or not np.isfinite(net).all():
        raise PortfolioWalkForwardError("portfolio_strategy_lab.net_return_invalid")
    net_metrics = evaluate_net_simple_return_path(net_simple_returns=net)
    log_returns = np.log1p(net)
    benchmark_log_returns = np.log1p(benchmark)
    anchor_log_returns = np.log1p(anchor) if anchor is not None else None
    active = net - benchmark
    benchmark_variance = float(np.var(benchmark, ddof=1))
    beta = (
        float(np.cov(net, benchmark, ddof=1)[0, 1] / benchmark_variance)
        if benchmark_variance > 0.0
        else 0.0
    )
    tracking = annualized_volatility(active)
    return PortfolioEconomicMetricSet(
        cost_bps=float(cost_bps),
        **net_metrics.model_dump(),
        beta=beta,
        zero_cash_jensen_alpha=float((np.mean(net) - beta * np.mean(benchmark)) * 252.0),
        tracking_error=tracking,
        information_ratio=(float(np.mean(active) * 252.0 / tracking) if tracking > 0.0 else 0.0),
        benchmark_relative_return=float(np.expm1(log_returns.sum() - benchmark_log_returns.sum())),
        anchor_relative_return=(
            float(np.expm1(log_returns.sum() - anchor_log_returns.sum()))
            if anchor_log_returns is not None
            else None
        ),
    )


def evaluate_net_simple_return_path(
    *, net_simple_returns: FloatArray
) -> PortfolioNetReturnMetricSet:
    """Reuse the economic owner's six return metrics without inventing a benchmark.

    The supplied net path must contain at least two finite daily observations
    greater than -1. Annual return compounds over 252 sessions; volatility and
    Sharpe use sample deviation (ddof=1). Costs have already been charged by the
    sealed path's owner and are never applied a second time here.
    """
    net = np.asarray(net_simple_returns, dtype=np.float64)
    if net.ndim != 1 or net.size < 2 or not np.isfinite(net).all() or np.any(net <= -1.0):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.net_return_invalid")
    log_returns = np.log1p(net)
    standard_deviation = float(np.std(net, ddof=1))
    return PortfolioNetReturnMetricSet(
        cumulative_return=float(np.expm1(log_returns.sum())),
        annualized_return=float(np.expm1(log_returns.sum() * 252.0 / int(net.size))),
        annualized_volatility=annualized_volatility(net),
        sharpe=(
            float(np.mean(net) / standard_deviation * math.sqrt(252.0))
            if standard_deviation > 0.0
            else 0.0
        ),
        sortino=sortino_ratio(net),
        maximum_drawdown=maximum_drawdown(log_returns),
    )


def maximum_drawdown(log_returns: FloatArray) -> float:
    """Measure the largest wealth loss relative to the prior running peak.

    Args:
        log_returns: Chronological log-return path whose cumulative sum defines wealth.

    Returns:
        The maximum fractional drawdown, including the initial unit-wealth anchor.
    """
    wealth = np.exp(np.cumsum(log_returns))
    peaks = np.maximum.accumulate(np.concatenate((np.asarray([1.0]), wealth)))[:-1]
    return float(np.max(1.0 - wealth / peaks, initial=0.0))


def annualized_volatility(simple_returns: FloatArray) -> float:
    """Scale sample simple-return deviation by the square root of 252 sessions.

    Args:
        simple_returns: Simple-return observations; sample deviation uses ddof=1.

    Returns:
        The annualized sample volatility.
    """
    return float(np.std(simple_returns, ddof=1) * math.sqrt(252.0))


def sortino_ratio(simple_returns: FloatArray) -> float:
    """Scale mean simple return by downside RMS and the annualization factor.

    Args:
        simple_returns: Simple-return observations measured against zero downside threshold.

    Returns:
        The annualized Sortino ratio, or infinity when downside RMS is zero.
    """
    downside = np.minimum(simple_returns, 0.0)
    deviation = float(np.sqrt(np.mean(np.square(downside))))
    return float(np.mean(simple_returns) / deviation * math.sqrt(252.0)) if deviation else math.inf


def moving_block_probability_positive(
    log_returns: FloatArray,
    *,
    block_size: int,
    resamples: int,
    seed: int,
) -> float:
    """Estimate positive log-wealth frequency under a seeded circular-block resample.

    Each replication sums ceil(path length / block size) complete circular blocks;
    the final block is not truncated. This is a resampling statistic on the supplied
    path, not a claim of predictive probability for a future strategy.

    Args:
        log_returns: Nonempty finite one-dimensional chronological log-return path.
        block_size: Positive circular block length in observations.
        resamples: Positive number of seeded bootstrap replications.
        seed: Deterministic NumPy generator seed.

    Returns:
        The fraction of resampled complete-block log sums strictly above zero.

    Raises:
        ValueError: The path dimension, block/resample count or return values are invalid.
    """
    if log_returns.ndim != 1 or block_size < 1 or resamples < 1:
        raise ValueError("portfolio_strategy_lab.bootstrap_input_invalid")
    if not np.isfinite(log_returns).all():
        raise ValueError("portfolio_strategy_lab.bootstrap_return_invalid")
    block_count = int(np.ceil(log_returns.size / block_size))
    # Wrap by tiling rather than by one slice. ``log_returns[: block_size - 1]``
    # silently returns a short array whenever the block is longer than the path,
    # and the block sums then index past the cumulative axis -- so a path shorter
    # than the 21-session block crashed instead of wrapping. For every path at
    # least ``block_size - 1`` long the tiled slice is the identical array, so no
    # already-published bootstrap value moves.
    repeats = -(-(block_size - 1) // log_returns.size) if block_size > 1 else 0
    circular = np.concatenate(
        (log_returns, np.tile(log_returns, max(repeats, 1))[: block_size - 1])
    )
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(circular)))
    starts = np.arange(log_returns.size)
    block_sums = cumulative[starts + block_size] - cumulative[starts]
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, log_returns.size, size=(resamples, block_count))
    totals = block_sums[sampled].sum(axis=1)
    return float(np.mean(totals > 0.0))


def _year_summary(sessions: tuple[date, ...], values: FloatArray) -> tuple[tuple[int, float], ...]:
    years = tuple(sorted(set(value.year for value in sessions)))
    return tuple(
        (year, float(values[[value.year == year for value in sessions]].sum())) for year in years
    )


def _chronological_blocks(values: FloatArray, block_size: int = 63) -> tuple[float, ...]:
    return tuple(
        float(values[start : start + block_size].sum())
        for start in range(0, values.size, block_size)
    )


def evaluate_portfolio_walk_forward_segments(
    *,
    workspace: PortfolioMetricsWorkspace,
    segments: tuple[PortfolioWalkForwardSegmentResult, ...],
    bootstrap_seed: int,
    bootstrap_block_size: int = 21,
    cost_policy: PortfolioCostPolicy | None = None,
) -> PortfolioWalkForwardResult:
    """Evaluate ordered nonoverlapping execution segments on explicit cost lanes.

    The owner retains the historical 5/10/20 bps reporting lanes while honoring the
    declared selection lane. Forecast calibration uses only decision-time forecast
    support and retains an explicit not-applicable disposition for non-Risk policies.
    The returned path joins segment evidence; passive sessions are recorded by the
    sequence owner separately.

    Args:
        workspace: Formation sessions and declared bootstrap resampling settings.
        segments: Nonempty execution segments whose combined indices are sorted and unique.
        bootstrap_seed: Deterministic circular-block resampling seed.
        bootstrap_block_size: Positive bootstrap block length in sessions.
        cost_policy: Declared reporting and selection cost policy, or the installed default.

    Returns:
        Path metrics, reporting returns, selection summaries and bootstrap evidence.

    Raises:
        PortfolioWalkForwardError: Segment evidence/order, net returns or forecast calibration is
            invalid.
    """
    if not segments:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.segment_evidence_missing")
    indices = tuple(
        index for segment in segments for index in range(segment.start_index, segment.stop_index)
    )
    if indices != tuple(sorted(set(indices))):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.segment_axis_invalid")
    sessions = tuple(workspace.formation_sessions[index] for index in indices)
    gross: FloatArray = np.asarray(
        [value for segment in segments for value in segment.gross_simple_returns],
        dtype=np.float64,
    )
    turnovers: FloatArray = np.asarray(
        [value for segment in segments for value in segment.one_way_turnovers],
        dtype=np.float64,
    )
    predicted_variances = [value for segment in segments for value in segment.predicted_variances]
    forecast_required = [
        required
        for segment in segments
        for required in (
            segment.risk_forecast_required or (True,) * len(segment.predicted_variances)
        )
    ]
    hhi: FloatArray = np.asarray(
        [value for segment in segments for value in segment.hhi], dtype=np.float64
    )
    holding_counts: FloatArray = np.asarray(
        [value for segment in segments for value in segment.holding_counts], dtype=np.float64
    )
    weighted_adv20: FloatArray = np.asarray(
        [value for segment in segments for value in segment.weighted_adv20], dtype=np.float64
    )
    sector_deviations: FloatArray = np.asarray(
        [value for segment in segments for value in segment.maximum_absolute_sector_deviations],
        dtype=np.float64,
    )
    gross_log = np.log1p(gross)
    policy = cost_policy or PortfolioCostPolicy()
    # The union of the declared axis and the three lanes published evidence
    # names by field. Reporting fewer than those three would make an existing
    # contract unfillable; the declaration decides which lane *selects*.
    lanes = tuple(sorted(set(policy.reporting_bps) | set(DEFAULT_REPORTING_COST_BPS)))
    net_paths = {
        bps: np.log1p(
            policy.net_simple_returns(
                gross_simple_returns=gross,
                one_way_turnovers=turnovers,
                cost_bps=bps,
            )
        )
        for bps in lanes
    }
    if gross.size < 2 or any(not np.isfinite(values).all() for values in net_paths.values()):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.net_return_invalid")
    net = net_paths[policy.selection_bps]
    forecasts: list[float | None] = [
        None if value is None else float(value) for value in predicted_variances
    ]
    calibration = realized_to_predicted_variance_ratio(
        predicted=forecasts,
        realized_simple=list(gross),
        risk_forecast_required=forecast_required,
    )
    selection_simple = np.expm1(net)
    selection_deviation = float(np.std(selection_simple, ddof=1))
    metrics = PortfolioTrialMetrics(
        cumulative_gross_log_wealth=float(gross_log.sum()),
        cumulative_net_log_wealth_5bps=float(net_paths[5].sum()),
        cumulative_net_log_wealth_10bps=float(net_paths[10].sum()),
        cumulative_net_log_wealth_20bps=float(net_paths[20].sum()),
        realized_annualized_volatility=float(np.std(selection_simple, ddof=1) * np.sqrt(252.0)),
        maximum_drawdown=maximum_drawdown(net),
        mean_one_way_turnover=float(turnovers.mean()),
        mean_hhi=float(hhi.mean()),
        mean_holding_count=float(holding_counts.mean()),
        mean_weighted_adv20=float(weighted_adv20.mean()),
        minimum_weighted_adv20=float(weighted_adv20.min()),
        solver_failure_count=0,
        missing_execution_count=sum(value.missed_execution_count for value in segments),
        realized_to_predicted_variance_ratio=calibration.ratio,
        variance_calibration_support_count=calibration.support_count,
        variance_calibration_disposition=calibration.disposition,
    )
    return PortfolioWalkForwardResult(
        metrics=metrics,
        gross_log_returns=tuple(float(value) for value in gross_log),
        net_log_returns_5bps=tuple(float(value) for value in net_paths[5]),
        net_log_returns_10bps=tuple(float(value) for value in net_paths[10]),
        net_log_returns_20bps=tuple(float(value) for value in net_paths[20]),
        net_log_returns_by_bps=tuple(
            (int(bps), tuple(float(value) for value in net_paths[bps]))
            for bps in policy.reporting_bps
        ),
        net_log_wealth_by_bps=tuple(
            (int(bps), float(net_paths[bps].sum())) for bps in policy.reporting_bps
        ),
        selection_cost_bps=int(policy.selection_bps),
        sharpe=(
            float(np.mean(selection_simple) / selection_deviation * np.sqrt(252.0))
            if selection_deviation > 0.0
            else None
        ),
        annual_net_log_returns=_year_summary(sessions, net),
        chronological_block_net_log_returns=_chronological_blocks(net),
        bootstrap_probability_net_positive=moving_block_probability_positive(
            net,
            block_size=bootstrap_block_size,
            resamples=workspace.mandate.bootstrap_resamples,
            seed=bootstrap_seed,
        ),
        sector_exposure_summary={
            "mean_maximum_absolute_deviation": float(sector_deviations.mean()),
            "maximum_absolute_deviation": float(sector_deviations.max()),
        },
    )


__all__ = [
    "PortfolioEconomicMetricSet",
    "PortfolioNetReturnMetricSet",
    "annualized_volatility",
    "break_even_cost_bps",
    "evaluate_net_simple_return_path",
    "evaluate_portfolio_walk_forward_segments",
    "evaluate_raw_simple_return_path",
    "maximum_drawdown",
    "moving_block_probability_positive",
    "sortino_ratio",
]
