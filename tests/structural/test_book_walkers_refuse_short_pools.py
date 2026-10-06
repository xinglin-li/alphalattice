"""Every walker of the book selection rules refuses a short pool before its walk (V500, V519).

A rebalance selects its names from those both tradable and scored, so a formation with fewer
cannot be decided. Each module that drives book decisions -- a research book, a strategy book,
a research update -- refuses such a formation by its session before it decides anything, worded
with its way on, never in the middle of the walk; the modules that hold the walk itself or a
selection rule name why they are not walkers. A new walker fails here until it names its guard.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "src/alphalattice/investment/portfolio_strategy_lab"
DRIVES = re.compile(
    r"\b(?:run_portfolio_walk_forward_segment|run_portfolio_walk_forward|decide_tranche_book"
    r"|TrancheBookDecisionProvider|CappedSleeveBookProvider|PortfolioPolicyDecisionProvider)\("
)
WALKERS = {
    # The research book: its plan holds an under-scored formation and its run refuses a
    # tradability-short one before the first segment (V500).
    f"{PACKAGE}/application/research_experiment.py": "portfolio_research.eligible_pool_short",
    # The component owner refuses the strategy book's walk before the book opens (V519).
    f"{PACKAGE}/application/tranche_book_execution.py": (
        "portfolio_strategy_lab.eligible_pool_short"
    ),
    # The research update refuses a short session before it proposes (V519).
    f"{PACKAGE}/application/decision_updates.py": "portfolio_update.eligible_pool_short",
    # The strategy coordinator opens its book through the component owner's check.
    f"{PACKAGE}/application/executor.py": "open_component_book(",
}
NOT_WALKERS = {
    "src/alphalattice/capabilities/portfolio_backtesting/segments.py": "the walk itself",
    "src/alphalattice/capabilities/portfolio_backtesting/engine.py": "the walk itself",
    f"{PACKAGE}/policies/tranche_book.py": "the tranche selection rule's owner",
    f"{PACKAGE}/research_loop/paired_alpha_portfolio.py": (
        "the panel-methodology route no Host operation reaches (V1); it retires with the "
        "research CLI"
    ),
}


def test_every_walker_of_the_book_selection_refuses_a_short_pool_before_its_walk() -> None:
    """requirement (V519): the modules driving book decisions are exactly the walkers and the
    named non-walkers, and each walker carries its refusal of a short pool before the walk."""

    found = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src" / "alphalattice").rglob("*.py")
        if DRIVES.search(path.read_text(encoding="utf-8"))
    }
    assert found == set(WALKERS) | set(NOT_WALKERS)
    for path, guard in WALKERS.items():
        assert guard in (ROOT / path).read_text(encoding="utf-8"), path
    # The coordinator constructs no provider of its own: every book it walks opens through
    # the component owner, whose check stands before the first decision.
    coordinator = (ROOT / f"{PACKAGE}/application/executor.py").read_text(encoding="utf-8")
    assert not re.search(r"\b\w+(?:Decision|Book)Provider\(", coordinator)
