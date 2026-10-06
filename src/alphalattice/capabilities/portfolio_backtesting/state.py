"""Continuous Portfolio state and embargo carry."""

from __future__ import annotations

from datetime import date

import numpy as np

from .contracts import (
    FloatArray,
    PortfolioBacktestWorkspace,
    PortfolioPassiveHoldResult,
    PortfolioWalkForwardError,
    PortfolioWalkForwardState,
)
from .execution import TOLERANCE, drift_holdings


def initial_portfolio_walk_forward_state(asset_count: int) -> PortfolioWalkForwardState:
    """Start both carried books flat with the entire unit budget in cash.

    Args:
        asset_count: Positive length of the shared listing axis.

    Returns:
        Read-only zero weights and matching full-cash state for both book and reference.

    Raises:
        PortfolioWalkForwardError: The asset count is below one.
    """
    if asset_count < 1:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.workspace_axis_invalid")
    weights: FloatArray = np.zeros(asset_count, dtype=np.float64)
    reference: FloatArray = np.zeros(asset_count, dtype=np.float64)
    weights.setflags(write=False)
    reference.setflags(write=False)
    return PortfolioWalkForwardState(
        pretrade_weights=weights,
        pretrade_cash=1.0,
        optimizer_reference=reference,
        optimizer_reference_cash=1.0,
    )


def validate_portfolio_state(state: PortfolioWalkForwardState, asset_count: int) -> None:
    """Both books, each against its own cash.

    The reference used to be checked for shape and finiteness only, while the
    budget identity was checked for the pre-trade book alone. That is exactly the
    hole the reference fell through: it was advanced with the *other* book's
    cash, so the pair stopped summing to one and nothing here said so -- the
    failure surfaced several calls later, inside ``drift_holdings``, as a carry
    error on a book that was fine.
    """
    for weights, cash in (
        (state.pretrade_weights, state.pretrade_cash),
        (state.optimizer_reference, state.optimizer_reference_cash),
    ):
        if (
            weights.shape != (asset_count,)
            or not np.isfinite(weights).all()
            or not np.isfinite(cash)
            or bool(np.any(weights < -TOLERANCE))
            or cash < -TOLERANCE
            or abs(float(weights.sum()) + cash - 1.0) > TOLERANCE
        ):
            raise PortfolioWalkForwardError("portfolio_strategy_lab.segment_state_invalid")


def advance_portfolio_state_without_decision(
    *,
    workspace: PortfolioBacktestWorkspace,
    state: PortfolioWalkForwardState,
    formation_session: date,
) -> PortfolioPassiveHoldResult:
    """Carry both books across one formation nobody was allowed to trade on.

    The two move differently, and that is not an oversight. The *book* is held
    through the embargo formation's own outcome window, so it drifts. The
    *reference* is defined as the book at the open of the next decision's
    session -- and the book entering this embargo already is that, because the
    last decided formation filled one session earlier and drifted into it. So the
    reference is taken, not drifted.

    Both of the previous shapes were wrong and wrong differently. Copying the
    reference unchanged left it at a session the decision had long passed;
    drifting it by this formation's outcome pushed it one session *past* the
    decision that reads it, and paid for the error with a budget identity that
    mixed one book's weights with the other's cash.
    """
    asset_count = len(workspace.ordered_listing_ids)
    validate_portfolio_state(state, asset_count)
    try:
        returns = workspace.passive_returns_by_session[formation_session]
    except KeyError as error:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.embargo_return_missing") from error
    if returns.shape != (asset_count,):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.embargo_return_axis_invalid")
    held = state.pretrade_weights > TOLERANCE
    if bool(np.any(held & ~np.isfinite(returns))):
        raise PortfolioWalkForwardError("portfolio_strategy_lab.held_return_missing")
    weights, cash, gross = drift_holdings(
        weights=state.pretrade_weights,
        cash=state.pretrade_cash,
        returns=returns,
    )
    result_weights = np.array(weights, copy=True)
    reference = np.array(state.pretrade_weights, copy=True)
    result_weights.setflags(write=False)
    reference.setflags(write=False)
    return PortfolioPassiveHoldResult(
        formation_session=formation_session,
        state_mode="NO_SCORE_PASSIVE_HOLD",
        gross_simple_return=float(gross),
        one_way_turnover=0,
        optimizer_call_count=0,
        target_decision_count=0,
        final_state=PortfolioWalkForwardState(
            pretrade_weights=result_weights,
            pretrade_cash=cash,
            optimizer_reference=reference,
            optimizer_reference_cash=state.pretrade_cash,
        ),
    )
