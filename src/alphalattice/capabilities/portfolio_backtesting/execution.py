"""Holdings, cash, order-execution, and failed-fill mechanics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from .contracts import BoolArray, FloatArray, PortfolioTargetDecision, PortfolioWalkForwardError

TOLERANCE = 1e-8


def execute_portfolio_entry(
    *,
    decision: PortfolioTargetDecision,
    pretrade_weights: FloatArray,
    pretrade_cash: float,
    execution_available: BoolArray,
) -> tuple[FloatArray, float, float, int]:
    """Execute a resolved conditional target without requiring its future outcome.

    Both a historical segment and an observed-entry continuation use this exact
    boundary. Returns remain a separate call to ``drift_holdings``.
    """
    if decision.target_weights.shape != pretrade_weights.shape:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.workspace_axis_invalid")
    if decision.decision_mode == "HOLD":
        if decision.predicted_variance is not None:
            raise PortfolioWalkForwardError("portfolio_strategy_lab.hold_carries_a_forecast")
    elif decision.predicted_variance is None:
        if decision.requires_risk_forecast:
            raise PortfolioWalkForwardError("portfolio_strategy_lab.workspace_axis_invalid")
    elif not decision.requires_risk_forecast or not np.isfinite(decision.predicted_variance):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.workspace_axis_invalid")
    return execute_orders(
        pretrade_weights=pretrade_weights,
        pretrade_cash=pretrade_cash,
        target_weights=decision.target_weights,
        execution_available=execution_available,
    )


def execute_orders(
    *,
    pretrade_weights: FloatArray,
    pretrade_cash: float,
    target_weights: FloatArray,
    execution_available: BoolArray,
) -> tuple[FloatArray, float, float, int]:
    """Fill available sells, scale available buys to cash and retain missed orders.

    Unavailable assets keep their pretrade weight. Tiny negative cash residues within
    the owner's tolerance are absorbed by the largest executed position before state
    carry. Asset turnover is half the L1 executed weight delta and excludes cash.

    Args:
        pretrade_weights: Carried asset weights entering the fill.
        pretrade_cash: Cash belonging to that same carried book.
        target_weights: Conditional desired asset weights on the shared listing axis.
        execution_available: Mask admitting sells and buys on each listing.

    Returns:
        Read-only executed weights, nonnegative cash, one-way asset turnover and missed-change
        count.

    Raises:
        PortfolioWalkForwardError: The executed state is nonfinite, negative or outside the unit
            budget tolerance.
    """
    delta = target_weights - pretrade_weights
    successful_sell = np.where(execution_available & (delta < 0.0), delta, 0.0)
    successful_buy = np.where(execution_available & (delta > 0.0), delta, 0.0)
    after_sells = pretrade_weights + successful_sell
    cash_after_sells = pretrade_cash - float(successful_sell.sum())
    desired_buys = float(successful_buy.sum())
    buy_scale = (
        1.0
        if desired_buys <= cash_after_sells + TOLERANCE
        else (max(cash_after_sells, 0.0) / desired_buys)
    )
    executed = after_sells + successful_buy * buy_scale
    cash = 1.0 - float(executed.sum())
    if -TOLERANCE <= cash < 0.0:
        # OSQP and proportional buy arithmetic may exceed the unit budget by a
        # few ulps. Seal that numerical residue here, before state carry, at the
        # execution owner rather than letting drift reinterpret leverage.
        position = int(np.argmax(executed))
        executed = np.array(executed, copy=True)
        executed[position] += cash
        cash = 1.0 - float(executed.sum())
    if (
        not np.isfinite(executed).all()
        or bool(np.any(executed < -TOLERANCE))
        or cash < -TOLERANCE
        or abs(float(executed.sum()) + cash - 1.0) > TOLERANCE
    ):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.execution_state_invalid")
    turnover = float(np.abs(executed - pretrade_weights).sum() / 2.0)
    missed = int(np.count_nonzero((np.abs(delta) > TOLERANCE) & ~execution_available))
    executed.setflags(write=False)
    return executed, max(cash, 0.0), turnover, missed


def drift_holdings(
    *, weights: FloatArray, cash: float, returns: FloatArray
) -> tuple[FloatArray, float, float]:
    """Advance an executed book through its simple-return outcome window.

    Args:
        weights: Executed asset weights, with held positions above the owner's tolerance.
        cash: Cash paired with these weights.
        returns: Asset simple returns; held assets must have finite outcomes.

    Returns:
        Read-only drifted weights, drifted cash and the gross portfolio simple return.

    Raises:
        PortfolioWalkForwardError: A held return is missing, wealth is nonpositive/nonfinite, or
            carried weights and cash violate budget.
    """
    held = weights > TOLERANCE
    if bool(np.any(held & ~np.isfinite(returns))):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.held_return_missing")
    safe_returns = np.where(held, returns, 0.0)
    gross_simple = float(weights @ safe_returns)
    gross_factor = 1.0 + gross_simple
    if not np.isfinite(gross_factor) or gross_factor <= 0.0:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.portfolio_return_invalid")
    drifted = weights * (1.0 + safe_returns) / gross_factor
    drifted_cash = cash / gross_factor
    # ``TOLERANCE``, not a private 1e-9. Drifting is scale-invariant -- it divides
    # the whole state by one factor -- so this can only fail on a state that
    # already did not sum to one, and the producer of that state is
    # ``execute_orders`` above, which admits cash down to ``-TOLERANCE`` and then
    # clamps it to zero. Checking ten times tighter than the only producer
    # guarantees made this a false alarm rather than a check: it removed 32 of
    # 156 walk-forward segments from a study, all of them the score-weighted
    # policy, because a book that puts nearly all its capital to work is exactly
    # the one whose residual cash lands within 1e-8 of zero.
    if abs(float(drifted.sum()) + drifted_cash - 1.0) > TOLERANCE:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.holdings_carry_invalid")
    drifted.setflags(write=False)
    return drifted, drifted_cash, gross_simple


def mark_book_to_session_close(
    *,
    weights: FloatArray,
    cash: float,
    marks_by_session: Mapping[date, Sequence[float]],
    session: date,
) -> tuple[FloatArray, float]:
    """Re-value a book held at ``open(session)`` for that session's close.

    Keyed on the **session label**, never on a position. A caller holding a
    formation index cannot reach this function without first resolving which
    session it means, which is the whole point: an array indexed beside the
    formation axis silently applies the wrong session's move, and no check inside
    a loop that knows only indices could ever notice.

    The arithmetic is ``drift_holdings``. Re-valuing a book by a return is one
    operation and only the window differs -- here the intraday move of the
    session the book is being valued at, which is complete at its close and
    therefore knowable by a decision taken there.
    """
    marks = marks_by_session.get(session)
    if marks is None:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.session_mark_absent_for_session")
    row: FloatArray = np.asarray(marks, dtype=np.float64)
    if row.shape != weights.shape:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.session_mark_axis_mismatch")
    if not np.isfinite(row).all() or bool(np.any(row <= -1.0)):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.session_mark_value_invalid")
    marked, marked_cash, _gross = drift_holdings(weights=weights, cash=cash, returns=row)
    return marked, marked_cash
