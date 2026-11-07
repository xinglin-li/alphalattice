"""Frozen heterogeneous Alpha recipe and its one-economy Portfolio consumer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pytest

from alphalattice.capabilities.portfolio_backtesting.segments import (
    run_portfolio_walk_forward_segment,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    COMPONENT_IDS,
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    SUCCESSOR_PACKAGE_HASH,
)
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    CappedSleeveBookProvider,
    CappedSleeveComponent,
    TrancheFormationInputs,
    open_component_book,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    INSTALLED_HETEROGENEOUS_BOOK_RECIPE,
)

LISTING_COUNT = 100
FORMATION_COUNT = 9
SESSIONS = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(FORMATION_COUNT))
LISTINGS = tuple(f"L{index:03d}" for index in range(LISTING_COUNT))


@dataclass(frozen=True)
class _Mandate:
    bootstrap_resamples: int = 0


@dataclass(frozen=True)
class _Workspace:
    mandate: _Mandate
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    execution_available: np.ndarray
    realized_simple_returns: np.ndarray
    causal_adv20: np.ndarray
    sector_exposure_matrix: np.ndarray
    equal_weight_sector_exposure: np.ndarray
    passive_returns_by_session: dict[date, np.ndarray]


def _workspace() -> _Workspace:
    returns = np.vstack(
        [
            np.linspace(-0.004, 0.006, LISTING_COUNT) * (1.0 if index % 2 == 0 else -0.7)
            for index in range(FORMATION_COUNT)
        ]
    )
    return _Workspace(
        mandate=_Mandate(),
        formation_sessions=SESSIONS,
        ordered_listing_ids=LISTINGS,
        execution_available=np.ones((FORMATION_COUNT, LISTING_COUNT), dtype=np.bool_),
        realized_simple_returns=returns,
        causal_adv20=np.full((FORMATION_COUNT, LISTING_COUNT), 10_000_000.0),
        sector_exposure_matrix=np.zeros((1, LISTING_COUNT)),
        equal_weight_sector_exposure=np.zeros(1),
        passive_returns_by_session={
            session: returns[index] for index, session in enumerate(SESSIONS)
        },
    )


def _component_formations(offset: int) -> tuple[TrancheFormationInputs, ...]:
    base = np.arange(LISTING_COUNT, dtype=np.float64)
    return tuple(
        TrancheFormationInputs(
            formation_session=session,
            scores=np.roll(base, offset + 7 * index),
            decision_eligible=np.ones(LISTING_COUNT, dtype=np.bool_),
            risk_allocation=None,
            risk_attribution=None,
            causal_rank_return_curve=None,
        )
        for index, session in enumerate(SESSIONS)
    )


def _formations() -> dict[str, tuple[TrancheFormationInputs, ...]]:
    return {
        component: _component_formations(13 * index)
        for index, component in enumerate(COMPONENT_IDS)
    }


def test_the_successor_restates_all_four_complete_recipes_without_rotating_g0() -> None:
    strategy = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY

    assert tuple(value.component_id for value in strategy.components) == COMPONENT_IDS
    assert strategy.successor_package_hash == SUCCESSOR_PACKAGE_HASH
    assert strategy.predecessor_recipe_hash == INSTALLED_ALPHA_PRODUCT_RECIPE.recipe_hash
    assert strategy.component("G2_R0_TREND").target_recipe == "T1_H3_PURE_TOTAL_RETURN_Z"
    assert strategy.component("G6_R0_FAST_REBOUND").objective == "ROW_BALANCED_L2"
    assert strategy.component("G7_R1_CONTEXTUAL_MOMENTUM").feature_count == 44
    assert strategy.allocation_basis_points == (2500, 2500, 2500, 2500)
    assert INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe.weight_rule == "ew"
    assert not INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe.true_up
    assert (
        INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe.sleeve_share_policy
        == "PRESERVE_DRIFTED_SLEEVE_NOTIONAL"
    )


def _components(
    formations: dict[str, tuple[TrancheFormationInputs, ...]],
) -> tuple[CappedSleeveComponent, ...]:
    book = INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe
    return tuple(
        CappedSleeveComponent(
            component_id=component,
            allocation_basis_points=2_500,
            top_k=book.top_k,
            exit_rank=book.exit_rank,
            tranches=book.tranches,
            aggregate_name_cap=book.aggregate_name_cap,
            aggregate_cap_start_formation=book.aggregate_cap_start_formation,
            formations=formations[component],
            ordered_listing_ids=LISTINGS,
        )
        for component in COMPONENT_IDS
    )


def test_four_independent_books_merge_before_the_shared_transition() -> None:
    workspace = _workspace()
    formations = _formations()
    book = INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe
    standalone_weights: list[np.ndarray] = []
    standalone_sleeve_masses: list[np.ndarray] = []
    for component in COMPONENT_IDS:
        provider = CappedSleeveBookProvider(
            top_k=book.top_k,
            exit_rank=book.exit_rank,
            tranches=book.tranches,
            aggregate_name_cap=book.aggregate_name_cap,
            aggregate_cap_start_formation=book.aggregate_cap_start_formation,
            formations=formations[component],
            ordered_listing_ids=LISTINGS,
        )
        result = run_portfolio_walk_forward_segment(
            workspace=workspace,
            decision_provider=provider,
            start_index=0,
            stop_index=FORMATION_COUNT,
        )
        standalone_weights.append(np.asarray(result.executed_weights, dtype=np.float64))
        sleeves = provider.sleeve_state
        assert sleeves is not None
        standalone_sleeve_masses.append(sleeves.sum(axis=1))

    merged = open_component_book(
        _components(formations), initial_sleeve_weights=None, schedule_offset=0
    )
    combined = run_portfolio_walk_forward_segment(
        workspace=workspace,
        decision_provider=merged,
        start_index=0,
        stop_index=FORMATION_COUNT,
    )
    expected = np.mean(np.stack(standalone_weights, axis=0), axis=0)

    np.testing.assert_allclose(combined.executed_weights, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        merged.target_weights_by_formation,
        expected,
        rtol=0.0,
        atol=1e-15,
    )
    assert merged.consumed_risk_projection_hashes == ()
    sleeves = merged.sleeve_state
    assert sleeves is not None
    assert sleeves.shape == (12, LISTING_COUNT)
    assert float(sleeves.sum()) == pytest.approx(4.0)
    assert any(
        not np.allclose(masses, np.full(3, 1.0 / 3.0), rtol=0.0, atol=1e-6)
        for masses in standalone_sleeve_masses
    )


def test_one_component_carrying_everything_is_the_book_itself() -> None:
    """One component carrying everything is the book itself."""

    formations = _formations()
    book = INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe
    single = CappedSleeveComponent(
        component_id=COMPONENT_IDS[0],
        allocation_basis_points=10_000,
        top_k=book.top_k,
        exit_rank=book.exit_rank,
        tranches=book.tranches,
        aggregate_name_cap=book.aggregate_name_cap,
        aggregate_cap_start_formation=book.aggregate_cap_start_formation,
        formations=formations[COMPONENT_IDS[0]],
        ordered_listing_ids=LISTINGS,
    )
    provider = open_component_book((single,), initial_sleeve_weights=None, schedule_offset=0)
    assert isinstance(provider, CappedSleeveBookProvider)

    direct = CappedSleeveBookProvider(
        top_k=book.top_k,
        exit_rank=book.exit_rank,
        tranches=book.tranches,
        aggregate_name_cap=book.aggregate_name_cap,
        aggregate_cap_start_formation=book.aggregate_cap_start_formation,
        formations=formations[COMPONENT_IDS[0]],
        ordered_listing_ids=LISTINGS,
    )
    workspace = _workspace()
    through_plan = run_portfolio_walk_forward_segment(
        workspace=workspace, decision_provider=provider, start_index=0, stop_index=FORMATION_COUNT
    )
    alone = run_portfolio_walk_forward_segment(
        workspace=workspace, decision_provider=direct, start_index=0, stop_index=FORMATION_COUNT
    )
    assert (
        np.asarray(through_plan.executed_weights).tobytes()
        == np.asarray(alone.executed_weights).tobytes()
    )
