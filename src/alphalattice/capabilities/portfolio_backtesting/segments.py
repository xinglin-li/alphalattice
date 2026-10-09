"""Causal segment execution and state carry."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from itertools import pairwise

import numpy as np

from .clocks import EveryFormationClock
from .contracts import (
    FloatArray,
    PortfolioBacktestWorkspace,
    PortfolioDecisionProvider,
    PortfolioPassiveHoldResult,
    PortfolioWalkForwardError,
    PortfolioWalkForwardSegmentResult,
    PortfolioWalkForwardSequenceResult,
    PortfolioWalkForwardState,
    RebalanceClock,
)
from .execution import TOLERANCE, drift_holdings, execute_portfolio_entry
from .reference_marks import OPEN_PROXY_REFERENCE_MARK_LANE, ReferenceMarkLane
from .state import (
    advance_portfolio_state_without_decision,
    initial_portfolio_walk_forward_state,
    validate_portfolio_state,
)


def run_portfolio_walk_forward_segment(
    *,
    workspace: PortfolioBacktestWorkspace,
    decision_provider: PortfolioDecisionProvider,
    start_index: int,
    stop_index: int,
    initial_state: PortfolioWalkForwardState | None = None,
    heartbeat: Callable[[int, int], None] | None = None,
    rebalance_clock: RebalanceClock | None = None,
    reference_mark: ReferenceMarkLane | None = None,
    entry_observer: Callable[[int, FloatArray, float], None] | None = None,
) -> PortfolioWalkForwardSegmentResult:
    """Execute a causal start-inclusive, stop-exclusive formation segment.

    Each formation resolves its clock mode and values the optimizer reference through
    the installed mark lane. The provider receives REBALANCE or HOLD explicitly; a
    hold preserves the drifted pretrade book. Fills precede outcome drift, and the
    exact executed weights are retained rather than reconstructed from targets.

    Args:
        workspace: Aligned causal execution and realized-outcome surfaces.
        decision_provider: Target/hold owner receiving separately valued reference and pretrade
            books.
        start_index: Inclusive start on the formation axis.
        stop_index: Exclusive stop on the formation axis.
        initial_state: Both carried books; required when the segment starts after index zero.
        heartbeat: Optional completed/total callback every 21 formations and at completion.
        rebalance_clock: Installed cadence; omitted uses every formation.
        reference_mark: Installed reference valuation lane; omitted uses the open proxy.
        entry_observer: Optional callback receiving index, exact executed weights and cash before
            drift.

    Returns:
        Exact fill/outcome metrics and the separately carried final book/reference state.

    Raises:
        PortfolioWalkForwardError: Axis, initial state, carry clock, forecast, fill, return or held
            liquidity evidence is invalid.
    """
    available_formations = len(workspace.formation_sessions)
    asset_count = len(workspace.ordered_listing_ids)
    if (
        workspace.execution_available.shape != (available_formations, asset_count)
        or workspace.realized_simple_returns.shape != (available_formations, asset_count)
        or start_index < 0
        or stop_index <= start_index
        or stop_index > available_formations
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.workspace_axis_invalid")
    state = initial_state or initial_portfolio_walk_forward_state(asset_count)
    if initial_state is None and start_index != 0:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.segment_initial_state_missing")
    validate_portfolio_state(state, asset_count)
    lane = reference_mark or OPEN_PROXY_REFERENCE_MARK_LANE
    lane.require_carry_segment(workspace.formation_sessions[start_index:stop_index])
    # The book at the open of the next decision's session, still unmarked. The
    # lane values it at the point of use, keyed on that session's own label, so
    # the state carries an instant rather than a valuation.
    carried_reference = np.array(state.optimizer_reference, copy=True)
    carried_reference_cash = state.optimizer_reference_cash
    pretrade_weights = np.array(state.pretrade_weights, copy=True)
    pretrade_cash = state.pretrade_cash
    formation_count = stop_index - start_index
    gross: FloatArray = np.empty(formation_count, dtype=np.float64)
    turnovers: FloatArray = np.empty(formation_count, dtype=np.float64)
    predicted_variances: list[float | None] = [None] * formation_count
    hhi: FloatArray = np.empty(formation_count, dtype=np.float64)
    holding_counts: FloatArray = np.empty(formation_count, dtype=np.float64)
    weighted_adv20: FloatArray = np.empty(formation_count, dtype=np.float64)
    sector_deviations: FloatArray = np.empty(formation_count, dtype=np.float64)
    missed_execution_count = 0
    decision_modes: list[str] = []
    risk_forecast_required: list[bool] = []
    executed_path: list[tuple[float, ...]] = []
    clock = rebalance_clock or EveryFormationClock()
    for local_index, index in enumerate(range(start_index, stop_index)):
        # A hold keeps the *drifted* book. Handing the provider the executed
        # reference and letting it return that would trade the drift back out and
        # charge a round trip of turnover for a session on which nothing was
        # decided -- which is why the clock is asked here rather than wrapped
        # around the provider.
        rebalances = clock.rebalances(formation_index=index, segment_start_index=start_index)
        # Valued here, at the decision, and not when it was produced. The book
        # carried in sits at this session's open; what the optimizer must price
        # turnover against is that book as of this session's *close*, which is
        # the instant the decision is taken. Marking it where it was produced
        # would mean naming the session from a formation index, which is the
        # off-by-one this lane exists to make unstateable.
        # The cash beside the valued reference is deliberately dropped: an
        # optimizer measures turnover against weights, and the pair only has to
        # exist so the valuation is a portfolio rather than a rescaled vector.
        # The *state* carries the executed pair at its own instant, below.
        optimizer_reference, _marked_reference_cash = lane.value_at_decision(
            weights=carried_reference,
            cash=carried_reference_cash,
            decision_session=workspace.formation_sessions[index],
        )
        optimizer_reference.setflags(write=False)
        decision = decision_provider(
            formation_index=index,
            reference_weights=optimizer_reference,
            pretrade_weights=pretrade_weights,
            decision_mode="REBALANCE" if rebalances else "HOLD",
        )
        executed, cash, turnover, missed = execute_portfolio_entry(
            decision=decision,
            pretrade_weights=pretrade_weights,
            pretrade_cash=pretrade_cash,
            execution_available=workspace.execution_available[index],
        )
        executed_path.append(tuple(float(value) for value in executed))
        if entry_observer is not None:
            entry_observer(index, executed, cash)
        pretrade_weights, pretrade_cash, gross[local_index] = drift_holdings(
            weights=executed,
            cash=cash,
            returns=workspace.realized_simple_returns[index],
        )
        # What the *next* formation will price turnover against: the executed
        # book, at the entry open it was filled at, which
        # ``require_carry_segment`` has already established is the next decision
        # session's own open. It is carried unvalued -- the lane applies the
        # method at the top of the next iteration, against that session's label.
        carried_reference = np.array(executed, copy=True)
        carried_reference_cash = cash
        carried_reference.setflags(write=False)
        turnovers[local_index] = turnover
        decision_modes.append(decision.decision_mode)
        risk_forecast_required.append(decision.requires_risk_forecast)
        predicted_variances[local_index] = decision.predicted_variance
        hhi[local_index] = float(np.square(executed).sum())
        holding_counts[local_index] = float(np.count_nonzero(executed > TOLERANCE))
        held = executed > TOLERANCE
        if getattr(workspace, "capacity_evidence_available", True):
            if bool(np.any(held & ~np.isfinite(workspace.causal_adv20[index]))):
                raise PortfolioWalkForwardError("portfolio_strategy_lab.held_adv20_missing")
            weighted_adv20[local_index] = float(
                executed[held] @ workspace.causal_adv20[index, held]
            )
        else:
            weighted_adv20[local_index] = np.nan
        anchor_sector = workspace.equal_weight_sector_exposure
        if anchor_sector.ndim == 2:
            anchor_sector = anchor_sector[index]
        # A Sector history's per-session exposure reads the formation's.
        exposure = workspace.sector_exposure_matrix
        if exposure.ndim == 3:
            exposure = exposure[index]
        sector_deviations[local_index] = float(np.max(np.abs(exposure @ executed - anchor_sector)))
        missed_execution_count += missed
        completed = local_index + 1
        if heartbeat is not None and (completed % 21 == 0 or completed == formation_count):
            heartbeat(completed, formation_count)
    final_weights = np.array(pretrade_weights, copy=True)
    final_reference = np.array(carried_reference, copy=True)
    final_weights.setflags(write=False)
    final_reference.setflags(write=False)
    return PortfolioWalkForwardSegmentResult(
        start_index=start_index,
        stop_index=stop_index,
        gross_simple_returns=tuple(float(value) for value in gross),
        one_way_turnovers=tuple(float(value) for value in turnovers),
        predicted_variances=tuple(
            None if value is None else float(value) for value in predicted_variances
        ),
        decision_modes=tuple(decision_modes),
        risk_forecast_required=tuple(risk_forecast_required),
        hhi=tuple(float(value) for value in hhi),
        holding_counts=tuple(float(value) for value in holding_counts),
        weighted_adv20=tuple(float(value) for value in weighted_adv20),
        maximum_absolute_sector_deviations=tuple(float(value) for value in sector_deviations),
        missed_execution_count=missed_execution_count,
        final_state=PortfolioWalkForwardState(
            pretrade_weights=final_weights,
            pretrade_cash=pretrade_cash,
            optimizer_reference=final_reference,
            optimizer_reference_cash=carried_reference_cash,
        ),
        executed_weights=tuple(executed_path),
    )


def run_portfolio_walk_forward_segment_sequence_with_passive_holds(
    *,
    workspace: PortfolioBacktestWorkspace,
    decision_provider: PortfolioDecisionProvider,
    ranges: tuple[tuple[int, int], ...],
    initial_state: PortfolioWalkForwardState | None = None,
    passive_sessions: tuple[date, ...] = (),
    rebalance_clock: RebalanceClock | None = None,
    reference_mark: ReferenceMarkLane | None = None,
) -> PortfolioWalkForwardSequenceResult:
    """Carry state through ordered segments and explicit no-decision embargo sessions.

    Passive sessions, when supplied, name one bridge between each pair of segments.
    The bridge must not be the next decision session and must satisfy the installed
    reference lane's carry transition. Passive results retain zero decisions,
    optimizer calls and turnover rather than pretending they are target decisions.

    Args:
        workspace: Aligned causal surfaces and session-keyed passive outcomes.
        decision_provider: Target/hold owner used within each decision segment.
        ranges: Nonempty ordered nonoverlapping start/stop formation ranges.
        initial_state: Optional carried state for the first segment.
        passive_sessions: Empty, or one explicit embargo session per segment transition.
        rebalance_clock: Installed cadence used by segment execution.
        reference_mark: Installed carry/valuation lane, or the open proxy.

    Returns:
        Executed decision segments and the separate passive-hold bridge results.

    Raises:
        PortfolioWalkForwardError: Ranges, embargo labels, carry transitions or segment execution
            evidence are invalid.
    """
    if (
        not ranges
        or any(stop <= start for start, stop in ranges)
        or len(passive_sessions) not in {0, len(ranges) - 1}
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.segment_axis_invalid")
    if any(left_stop > right_start for (_, left_stop), (right_start, _) in pairwise(ranges)):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.segment_axis_invalid")
    lane = reference_mark or OPEN_PROXY_REFERENCE_MARK_LANE
    state = initial_state
    results: list[PortfolioWalkForwardSegmentResult] = []
    passive_holds: list[PortfolioPassiveHoldResult] = []
    for position, (start, stop) in enumerate(ranges):
        if position and passive_sessions:
            assert state is not None
            last_decided = workspace.formation_sessions[ranges[position - 1][1] - 1]
            embargo_session = passive_sessions[position - 1]
            next_decision = workspace.formation_sessions[start]
            if embargo_session == next_decision:
                # Not an embargo. A split policy whose embargo index is excluded
                # from the local axis makes the two regions adjacent, and the
                # session offered here is then a formation that *is* decided on
                # -- so the carry below would drift the book through that
                # formation's own outcome window before its decision was taken.
                # Refused under its own name, because the previous behaviour was
                # to raise a message about a gap in the axis and send a reader
                # looking for a purge that is not there.
                raise PortfolioWalkForwardError(
                    "portfolio_strategy_lab.embargo_session_is_a_decision"
                )
            # The one edge that crosses the embargo, checked explicitly because
            # its three sessions are not adjacent on the decision axis: the
            # embargo formation is not decided on and is therefore not in it.
            lane.require_carry_transition(
                last_decided=last_decided,
                embargo_session=embargo_session,
                next_decision=next_decision,
            )
            passive_hold = advance_portfolio_state_without_decision(
                workspace=workspace,
                state=state,
                formation_session=passive_sessions[position - 1],
            )
            passive_holds.append(passive_hold)
            state = passive_hold.final_state
        result = run_portfolio_walk_forward_segment(
            workspace=workspace,
            decision_provider=decision_provider,
            start_index=start,
            stop_index=stop,
            initial_state=state,
            rebalance_clock=rebalance_clock,
            reference_mark=reference_mark,
        )
        results.append(result)
        state = result.final_state
    return PortfolioWalkForwardSequenceResult(
        segments=tuple(results), passive_holds=tuple(passive_holds)
    )
