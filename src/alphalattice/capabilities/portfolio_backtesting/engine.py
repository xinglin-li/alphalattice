"""High-level deterministic Portfolio path engine."""

from __future__ import annotations

from collections.abc import Callable

from .contracts import (
    PortfolioBacktestWorkspace,
    PortfolioCostPolicy,
    PortfolioDecisionProvider,
    PortfolioWalkForwardError,
    PortfolioWalkForwardResult,
    RebalanceClock,
)
from .metrics import evaluate_portfolio_walk_forward_segments
from .segments import run_portfolio_walk_forward_segment


def run_portfolio_walk_forward(
    *,
    workspace: PortfolioBacktestWorkspace,
    decision_provider: PortfolioDecisionProvider,
    bootstrap_seed: int,
    formation_limit: int | None = None,
    heartbeat: Callable[[int, int], None] | None = None,
    bootstrap_block_size: int = 21,
    cost_policy: PortfolioCostPolicy | None = None,
    rebalance_clock: RebalanceClock | None = None,
) -> PortfolioWalkForwardResult:
    """Execute the causal formation prefix and evaluate its declared cost lanes.

    Args:
        workspace: Aligned decision, fill, outcome, liquidity and sector surfaces.
        decision_provider: Owner returning an explicit target or hold for each formation.
        bootstrap_seed: Deterministic circular-block bootstrap seed.
        formation_limit: Optional formation prefix count; omitted uses the complete axis.
        heartbeat: Optional completed/total callback from segment execution.
        bootstrap_block_size: Positive block length used by path resampling.
        cost_policy: Reporting/selection cost policy, or the installed default.
        rebalance_clock: Installed cadence, or every formation when omitted.

    Returns:
        The evaluated gross/net path, metrics and bootstrap summaries.

    Raises:
        PortfolioWalkForwardError: The formation limit, aligned workspace, execution state or
            evaluated path is invalid.
    """
    available = len(workspace.formation_sessions)
    formation_count = formation_limit or available
    if formation_count < 1 or formation_count > available:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.workspace_axis_invalid")
    segment = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=decision_provider,
        start_index=0,
        stop_index=formation_count,
        heartbeat=heartbeat,
        rebalance_clock=rebalance_clock,
    )
    return evaluate_portfolio_walk_forward_segments(
        workspace=workspace,
        segments=(segment,),
        bootstrap_seed=bootstrap_seed,
        bootstrap_block_size=bootstrap_block_size,
        cost_policy=cost_policy,
    )
