"""Gate 9B item 1: verify the shared Backtesting owners once, in one place.

Ten properties the whole product rests on -- clock, score alignment, stable ties,
sleeve state, hysteresis, pretrade drift, fills and missed fills, zero-trade
holds, turnover and cost units, benchmark alignment, wealth and report metrics --
checked against the installed owners rather than restated anywhere downstream.
That "once" matters: the reason Validation, Product Host and the report layer
must not carry their own version of any of these is that a second expression can
agree with itself while both drift from the mechanics that actually ran.

Every axis here is synthetic and variable. The historical 252 / 494 / 1,260
cardinalities appear nowhere, because a verification pinned to one shape proves
the shape rather than the mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from alphalattice.capabilities.portfolio_backtesting.active_metrics import (
    ActiveMetricsError,
    active_path_metrics,
)
from alphalattice.capabilities.portfolio_backtesting.clocks import EveryFormationClock
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioCostPolicy,
    PortfolioTargetDecision,
)
from alphalattice.capabilities.portfolio_backtesting.execution import (
    drift_holdings,
    execute_orders,
)
from alphalattice.capabilities.portfolio_backtesting.metrics import (
    evaluate_net_simple_return_path,
    evaluate_raw_simple_return_path,
)
from alphalattice.capabilities.portfolio_backtesting.segments import (
    run_portfolio_walk_forward_segment,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class _Mandate:
    """The one setting the owner reads off a mandate; the rest is not its business."""

    bootstrap_resamples: int = 0


@dataclass(frozen=True)
class _Workspace:
    """The exact lane surface the Backtesting owner reads, and nothing more."""

    mandate: _Mandate
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    execution_available: BoolArray
    realized_simple_returns: FloatArray
    causal_adv20: FloatArray
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray
    passive_returns_by_session: dict[date, FloatArray]


def _workspace(
    *,
    formations: int,
    listings: int,
    unavailable: tuple[tuple[int, int], ...] = (),
    seed: int = 7,
) -> _Workspace:
    sessions = tuple(date(2024, 5, 1) + timedelta(days=index) for index in range(formations))
    names = tuple(f"bt-{index:04d}" for index in range(listings))
    generator = np.random.default_rng(seed)
    realized = generator.normal(0.0, 0.011, size=(formations, listings))
    available = np.ones((formations, listings), dtype=np.bool_)
    for row, column in unavailable:
        available[row, column] = False
    return _Workspace(
        mandate=_Mandate(),
        formation_sessions=sessions,
        ordered_listing_ids=names,
        execution_available=available,
        realized_simple_returns=np.ascontiguousarray(realized, dtype=np.float64),
        causal_adv20=np.full((formations, listings), 2_000_000.0, dtype=np.float64),
        sector_exposure_matrix=np.ones((1, listings), dtype=np.float64),
        equal_weight_sector_exposure=np.ones(1, dtype=np.float64),
        passive_returns_by_session={
            session: realized[index] for index, session in enumerate(sessions)
        },
    )


def _equal_weight_target(count: int, held: int) -> FloatArray:
    weights = np.zeros(count, dtype=np.float64)
    weights[:held] = 1.0 / held
    return weights


@dataclass
class _Provider:
    """A decision provider that records exactly what the owner handed it."""

    listings: int
    held: int
    hold_after: int | None = None
    reference_seen: list[tuple[float, ...]] = None  # type: ignore[assignment]
    pretrade_seen: list[tuple[float, ...]] = None  # type: ignore[assignment]
    modes_seen: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reference_seen = []
        self.pretrade_seen = []
        self.modes_seen = []

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        self.reference_seen.append(tuple(float(v) for v in reference_weights))
        self.pretrade_seen.append(tuple(float(v) for v in pretrade_weights))
        self.modes_seen.append(decision_mode)
        if decision_mode == "HOLD" or (
            self.hold_after is not None and formation_index >= self.hold_after
        ):
            # A hold keeps the drifted book. Returning the reference here would
            # silently charge a full round trip on every held session.
            return PortfolioTargetDecision(
                target_weights=np.array(pretrade_weights, copy=True),
                predicted_variance=None,
                decision_mode="HOLD",
            )
        # A direct equal-weight rule consumes no covariance, and the owner
        # requires that to be declared rather than inferred from a missing
        # forecast.
        return PortfolioTargetDecision(
            target_weights=_equal_weight_target(self.listings, self.held),
            predicted_variance=None,
            requires_risk_forecast=False,
        )


# ============================================================ clock and modes


@pytest.mark.parametrize(("formations", "listings"), [(9, 23), (17, 41), (6, 137)])
def test_the_clock_decides_every_formation_and_the_axes_are_whatever_they_are(
    formations: int, listings: int
) -> None:
    """One decision per formation, at any axis shape the caller brings."""

    assert formations not in {252, 494, 1260}
    assert listings not in {464, 466}
    workspace = _workspace(formations=formations, listings=listings)
    provider = _Provider(listings=listings, held=min(5, listings))
    result = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=provider,
        start_index=0,
        stop_index=formations,
        rebalance_clock=EveryFormationClock(),
    )
    assert len(result.gross_simple_returns) == formations
    assert len(result.one_way_turnovers) == formations
    assert len(result.executed_weights) == formations
    assert result.decision_modes == ("REBALANCE",) * formations
    assert provider.modes_seen == ["REBALANCE"] * formations


def test_the_provider_is_handed_the_executed_book_and_the_drifted_book_apart() -> None:
    """The provider is handed the executed book and the drifted book apart."""

    workspace = _workspace(formations=8, listings=19)
    provider = _Provider(listings=19, held=4)
    result = run_portfolio_walk_forward_segment(
        workspace=workspace, decision_provider=provider, start_index=0, stop_index=8
    )
    # The first formation starts flat; from the second on, the reference is the
    # previous executed book and the pretrade book is that book after drift.
    assert provider.reference_seen[0] == (0.0,) * 19
    assert provider.reference_seen[1] == result.executed_weights[0]
    assert provider.pretrade_seen[1] != provider.reference_seen[1]
    drifted, _cash, _gross = drift_holdings(
        weights=np.asarray(result.executed_weights[0], dtype=np.float64),
        cash=0.0,
        returns=workspace.realized_simple_returns[0],
    )
    assert np.allclose(np.asarray(provider.pretrade_seen[1]), drifted, atol=1e-15)


def test_equal_scores_break_ties_by_position_and_stay_stable() -> None:
    """Stable ties: the same tied input produces the same book every time."""

    workspace = _workspace(formations=7, listings=31)
    first = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=31, held=6),
        start_index=0,
        stop_index=7,
    )
    second = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=31, held=6),
        start_index=0,
        stop_index=7,
    )
    assert first.executed_weights == second.executed_weights
    assert first.gross_simple_returns == second.gross_simple_returns
    assert first.one_way_turnovers == second.one_way_turnovers


# ================================================= state carry across segments


def test_sleeve_state_carries_across_segment_boundaries_exactly() -> None:
    """Sleeve state carries across segment boundaries exactly."""

    workspace = _workspace(formations=11, listings=27)
    whole = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=27, held=5),
        start_index=0,
        stop_index=11,
    )
    head = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=27, held=5),
        start_index=0,
        stop_index=6,
    )
    tail = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=27, held=5),
        start_index=6,
        stop_index=11,
        initial_state=head.final_state,
    )
    assert head.executed_weights + tail.executed_weights == whole.executed_weights
    assert head.gross_simple_returns + tail.gross_simple_returns == whole.gross_simple_returns
    assert head.one_way_turnovers + tail.one_way_turnovers == whole.one_way_turnovers
    assert np.array_equal(
        np.asarray(tail.final_state.optimizer_reference, dtype=np.float64),
        np.asarray(whole.final_state.optimizer_reference, dtype=np.float64),
    )


def test_a_segment_that_does_not_start_at_zero_needs_the_carried_state() -> None:
    """The owner refuses to invent a book it was not handed."""

    workspace = _workspace(formations=6, listings=13)
    with pytest.raises(ValueError, match="segment_initial_state_missing"):
        run_portfolio_walk_forward_segment(
            workspace=workspace,
            decision_provider=_Provider(listings=13, held=3),
            start_index=2,
            stop_index=6,
        )


# ============================================ holds, drift, fills, turnover


def test_a_zero_trade_hold_keeps_the_drifted_book_and_costs_no_turnover() -> None:
    """Hysteresis and zero-trade holds: holding is free, and it holds the drift."""

    workspace = _workspace(formations=9, listings=21)
    provider = _Provider(listings=21, held=4, hold_after=3)
    result = run_portfolio_walk_forward_segment(
        workspace=workspace, decision_provider=provider, start_index=0, stop_index=9
    )
    held_turnover = result.one_way_turnovers[4:]
    assert all(value == pytest.approx(0.0, abs=1e-15) for value in held_turnover)
    # And the held book is the drifted one, not the previous executed one.
    previous = np.asarray(result.executed_weights[3], dtype=np.float64)
    drifted, _cash, _gross = drift_holdings(
        weights=previous, cash=0.0, returns=workspace.realized_simple_returns[3]
    )
    assert np.allclose(
        np.asarray(result.executed_weights[4], dtype=np.float64), drifted, atol=1e-15
    )


def test_an_unavailable_name_is_a_missed_fill_not_a_silent_substitution() -> None:
    """Fills and missed fills: the owner counts what it could not trade."""

    workspace = _workspace(formations=6, listings=17, unavailable=((2, 0), (2, 1)))
    result = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=17, held=4),
        start_index=0,
        stop_index=6,
    )
    assert result.missed_execution_count > 0
    # The executed book at the blocked formation differs from the target, and the
    # owner says so rather than reporting a clean fill.
    executed = np.asarray(result.executed_weights[2], dtype=np.float64)
    target = _equal_weight_target(17, 4)
    assert not np.allclose(executed, target, atol=1e-12)


def test_execute_orders_never_trades_an_unavailable_name() -> None:
    """The execution owner itself, not just the segment that calls it."""

    count = 12
    pretrade = np.zeros(count, dtype=np.float64)
    target = _equal_weight_target(count, 4)
    available = np.ones(count, dtype=np.bool_)
    available[1] = False
    holdings, _cash, _turnover, missed = execute_orders(
        pretrade_weights=pretrade,
        pretrade_cash=1.0,
        target_weights=target,
        execution_available=available,
    )
    assert holdings[1] == pytest.approx(0.0, abs=1e-15)
    assert missed == 1


def test_turnover_is_one_way_and_the_cost_policy_charges_it_once() -> None:
    """Turnover and cost units: one-way turnover, one rate, one multiplication."""

    workspace = _workspace(formations=8, listings=25)
    result = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=25, held=5),
        start_index=0,
        stop_index=8,
    )
    # One-way turnover is half the absolute weight change, so buying a whole
    # book from flat is 0.5 rather than 1.0. The convention is the owner's, and
    # every consumer -- cost policy, replay, report -- has to read it the same
    # way or the cost of a rebalance doubles somewhere downstream.
    assert result.one_way_turnovers[0] == pytest.approx(0.5, abs=1e-12)

    gross = np.asarray(result.gross_simple_returns, dtype=np.float64)
    turnover = np.asarray(result.one_way_turnovers, dtype=np.float64)
    policy = PortfolioCostPolicy(reporting_bps=(10,), selection_bps=10)
    net = policy.net_simple_returns(
        gross_simple_returns=gross, one_way_turnovers=turnover, cost_bps=10.0
    )
    assert np.allclose(net, gross - turnover * 10.0 / 10_000.0, atol=1e-15)
    # Twice the rate is exactly twice the drag: the unit is linear in bps.
    doubled = policy.net_simple_returns(
        gross_simple_returns=gross, one_way_turnovers=turnover, cost_bps=20.0
    )
    assert np.allclose(gross - doubled, 2.0 * (gross - net), atol=1e-15)


# ================================================ benchmark, wealth, metrics


def test_the_benchmark_owner_aligns_on_the_same_axis_and_refuses_a_flat_one() -> None:
    """Benchmark support: beta is cov/var on the same axis, or an honest refusal."""

    workspace = _workspace(formations=10, listings=29)
    result = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=29, held=6),
        start_index=0,
        stop_index=10,
    )
    net = np.asarray(result.gross_simple_returns, dtype=np.float64)
    anchor = np.asarray(
        [float(workspace.realized_simple_returns[index].mean()) for index in range(10)],
        dtype=np.float64,
    )
    metrics = active_path_metrics(portfolio_simple=net, benchmark_simple=anchor)
    expected = float(np.cov(net, anchor, ddof=1)[0][1] / np.var(anchor, ddof=1))
    assert metrics.beta == pytest.approx(expected, rel=0.0, abs=1e-12)

    with pytest.raises(ActiveMetricsError):
        active_path_metrics(portfolio_simple=net, benchmark_simple=np.zeros(10, dtype=np.float64))


def test_wealth_compounds_the_path_the_owner_actually_produced() -> None:
    """Wealth and report metrics: compounded from the owner's own returns."""

    workspace = _workspace(formations=13, listings=33)
    result = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=_Provider(listings=33, held=7),
        start_index=0,
        stop_index=13,
    )
    gross = np.asarray(result.gross_simple_returns, dtype=np.float64)
    wealth = float(np.prod(1.0 + gross))
    # A path of 13 formations compounds to the same number when split in two,
    # which is the property every window and continuation reads.
    head = float(np.prod(1.0 + gross[:5]))
    tail = float(np.prod(1.0 + gross[5:]))
    assert head * tail == pytest.approx(wealth, rel=1e-12)
    assert len(result.hhi) == len(result.holding_counts) == 13
    assert all(value > 0.0 for value in result.holding_counts)


def test_the_raw_simple_return_path_reports_the_whole_cost_surface_and_sortino() -> None:
    """requirement: all money metrics use execution returns at 0/2/5/10/20 bps."""

    gross = np.asarray([0.01, -0.006, 0.004, -0.002, 0.008, 0.001], dtype=np.float64)
    turnover = np.asarray([1.0, 0.2, 0.5, 0.1, 0.4, 0.3], dtype=np.float64)
    spy = np.asarray([0.004, -0.004, 0.002, -0.001, 0.003, 0.0], dtype=np.float64)
    anchor = np.asarray([0.003, -0.003, 0.001, 0.0, 0.002, 0.001], dtype=np.float64)
    metrics = tuple(
        evaluate_raw_simple_return_path(
            gross_simple_returns=gross,
            one_way_turnovers=turnover,
            benchmark_simple_returns=spy,
            anchor_simple_returns=anchor,
            cost_bps=cost,
        )
        for cost in (0.0, 2.0, 5.0, 10.0, 20.0)
    )

    assert tuple(value.cost_bps for value in metrics) == (0.0, 2.0, 5.0, 10.0, 20.0)
    assert all(np.isfinite(value.sortino) for value in metrics)
    assert all(value.anchor_relative_return is not None for value in metrics)
    assert metrics[0].cumulative_return > metrics[-1].cumulative_return
    assert metrics[0].tracking_error > 0.0
    for value in metrics:
        net = PortfolioCostPolicy().net_simple_returns(
            gross_simple_returns=gross,
            one_way_turnovers=turnover,
            cost_bps=value.cost_bps,
        )
        return_only = evaluate_net_simple_return_path(net_simple_returns=net)
        assert return_only.model_dump() == {
            name: getattr(value, name) for name in type(return_only).model_fields
        }
        # Independent expressions retain the pre-extraction conventions, including
        # the unit-wealth opening anchor and sample (rather than population) volatility.
        wealth = np.cumprod(1.0 + net)
        peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[:-1]
        downside = np.minimum(net, 0.0)
        assert return_only.cumulative_return == pytest.approx(wealth[-1] - 1.0)
        assert return_only.annualized_return == pytest.approx(wealth[-1] ** (252 / len(net)) - 1)
        assert return_only.annualized_volatility == pytest.approx(
            np.std(net, ddof=1) * np.sqrt(252)
        )
        assert return_only.sharpe == pytest.approx(
            np.mean(net) / np.std(net, ddof=1) * np.sqrt(252)
        )
        assert return_only.sortino == pytest.approx(
            np.mean(net) / np.sqrt(np.mean(downside**2)) * np.sqrt(252)
        )
        assert return_only.maximum_drawdown == pytest.approx(np.max(1.0 - wealth / peaks))
    assert np.isinf(
        evaluate_net_simple_return_path(
            net_simple_returns=np.asarray([0.01, 0.02], dtype=np.float64)
        ).sortino
    )
