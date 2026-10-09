"""The tranche executor: carried sleeve state, and the arithmetic it refuses to own.

Two claims carry this slice. The executor implements no drift, and the installed
product recipe explicitly resets sleeve notionals to 1/T at every formation.
Both are asserted directly because both can stay plausible while being wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pytest

from alphalattice.capabilities.portfolio_backtesting.engine import (
    run_portfolio_walk_forward,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    CappedSleeveComponent,
    TrancheBookComponent,
    TrancheBookDecisionProvider,
    TrancheExecutionError,
    TrancheFormationInputs,
    open_component_book,
    project_sleeves_onto_book,
    rebuild_sleeve_state,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import TrancheBookRecipe
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
    FactorIdiosyncraticRiskSurface,
    RiskAllocationProjection,
    RiskAttributionProjection,
    RiskDecompositionRecipe,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

LISTINGS = 120
RECIPE = TrancheBookRecipe.create(top_k=35, tranches=3, exit_rank=70)
SESSIONS = tuple(date(2023, 1, 3) + timedelta(days=index) for index in range(24))


def _curve(formation_index: int, session: date) -> CausalRankReturnCurveSlice:
    return CausalRankReturnCurveSlice(
        formation_index=formation_index,
        formation_session=session,
        bucket_means=np.array([0.05, 0.02, -0.01, -0.04], dtype=np.float64),
        bucket_support_counts=(252,) * 4,
        admitted_formation_count=252,
        disposition="AVAILABLE",
        curve_hash="0" * 64,
    )


def _risk_surface(
    session: date,
    *,
    factor_scale: float = 1.0,
    recipe: RiskDecompositionRecipe = INSTALLED_RISK_DECOMPOSITION_RECIPE,
) -> FactorIdiosyncraticRiskSurface:
    rng = np.random.default_rng(11)
    exposures = np.zeros((LISTINGS, 4), dtype=np.float64)
    exposures[np.arange(LISTINGS), rng.integers(0, 4, LISTINGS)] = 1.0
    root = rng.normal(size=(4, 4))
    return FactorIdiosyncraticRiskSurface.create(
        recipe=recipe,
        formation_session=session,
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        ordered_factor_ids=tuple(f"S{j}" for j in range(4)),
        exposures=exposures,
        factor_covariance=root @ root.T * 0.0004 * factor_scale,
        idiosyncratic_variance=rng.uniform(1e-4, 9e-4, LISTINGS),
        conditional_volatility=np.linspace(0.10, 0.30, LISTINGS),
        producer_identity="TEST_PRODUCER",
    )


def _risk(
    session: date,
    *,
    factor_scale: float = 1.0,
    recipe: RiskDecompositionRecipe = INSTALLED_RISK_DECOMPOSITION_RECIPE,
) -> RiskAllocationProjection:
    return RiskAllocationProjection.of(
        _risk_surface(session, factor_scale=factor_scale, recipe=recipe)
    )


def _attribution(
    session: date,
    *,
    factor_scale: float = 1.0,
    recipe: RiskDecompositionRecipe = INSTALLED_RISK_DECOMPOSITION_RECIPE,
) -> RiskAttributionProjection:
    return RiskAttributionProjection.of(
        _risk_surface(session, factor_scale=factor_scale, recipe=recipe),
        classification_authority="TEST_FOUNDATION_CLASSIFICATION",
    )


def _formations(count: int) -> tuple[TrancheFormationInputs, ...]:
    return tuple(
        TrancheFormationInputs(
            formation_session=SESSIONS[index],
            scores=np.random.default_rng(100 + index).normal(size=LISTINGS),
            decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
            risk_allocation=_risk(SESSIONS[index]),
            risk_attribution=_attribution(SESSIONS[index]),
            causal_rank_return_curve=_curve(index, SESSIONS[index]),
        )
        for index in range(count)
    )


def _provider(count: int = 6) -> TrancheBookDecisionProvider:
    return TrancheBookDecisionProvider(
        recipe=RECIPE,
        formations=_formations(count),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
    )


def _drive(
    provider: TrancheBookDecisionProvider, *, formations: int, growth: np.ndarray | None = None
):
    """Drive the provider the way the engine does, applying growth between formations."""

    book = np.zeros(LISTINGS, dtype=np.float64)
    decisions = []
    for index in range(formations):
        pretrade = book if index == 0 else book * (1.0 + growth if growth is not None else 1.0)
        total = float(pretrade.sum())
        pretrade = pretrade / total if total > 0 else pretrade
        decision = provider(
            formation_index=index,
            reference_weights=book,
            pretrade_weights=pretrade,
            decision_mode="REBALANCE",
        )
        decisions.append(decision)
        book = decision.target_weights
    return decisions


def test_the_first_formation_stages_every_sleeve_and_later_ones_review_one() -> None:
    provider = _provider()
    _drive(provider, formations=4, growth=np.zeros(LISTINGS))
    reviewed = provider.reviewed_sleeves_by_formation
    assert reviewed[0][1] == (0, 1, 2)
    assert [entry[1] for entry in reviewed[1:]] == [(1,), (2,), (0,)]


def test_the_carried_state_is_book_scale_and_sums_to_the_target() -> None:
    """Sleeve state that did not sum to the book would drift apart silently."""

    provider = _provider()
    decisions = _drive(provider, formations=3, growth=np.zeros(LISTINGS))
    sleeves = provider.sleeve_state
    assert sleeves is not None
    assert sleeves.shape == (RECIPE.tranches, LISTINGS)
    assert sleeves.sum(axis=0) == pytest.approx(decisions[-1].target_weights)


def test_sleeve_shares_reset_to_equal_notional_at_each_formation() -> None:
    """The frozen product construction states the true-up rather than defaulting it."""

    provider = _provider()
    growth = np.zeros(LISTINGS)
    growth[: LISTINGS // 2] = 0.5

    # Enough formations for the sleeves to hold different names. At the staging
    # formation they all see the same scores and therefore select the same book,
    # so they cannot diverge until each has been reviewed against its own
    # formation -- which is the schedule working, not a true-up.
    _drive(provider, formations=6, growth=growth)
    assert provider.sleeve_shares_by_formation
    for shares in provider.sleeve_shares_by_formation:
        assert shares == pytest.approx((1.0 / RECIPE.tranches,) * RECIPE.tranches)


def test_the_projection_never_computes_a_return() -> None:
    """Drift enters only as the engine's own drifted-over-executed ratio.

    Scaling the drifted book by any positive constant must leave the projection
    unchanged, because the sleeves are renormalised. A projection that had
    recomputed returns from prices would not have that property, so this is a
    behavioural check on where the arithmetic lives rather than a comment.
    """

    sleeves = np.zeros((2, 6), dtype=np.float64)
    sleeves[0, :3] = 0.25
    sleeves[1, 3:] = 0.25
    executed = sleeves.sum(axis=0)
    drifted = executed * np.array([1.4, 1.0, 0.8, 1.1, 1.0, 0.9])

    base = project_sleeves_onto_book(
        sleeves=sleeves, reference_weights=executed, pretrade_weights=drifted / drifted.sum()
    )
    scaled = project_sleeves_onto_book(
        sleeves=sleeves, reference_weights=executed, pretrade_weights=drifted * 7.0
    )
    assert base == pytest.approx(scaled)
    assert base.sum() == pytest.approx(1.0)


def test_the_projection_absorbs_a_short_fill_across_the_sleeves_holding_that_name() -> None:
    """The executed book is what happened; the sleeves must agree with it."""

    sleeves = np.zeros((2, 4), dtype=np.float64)
    sleeves[0, :2] = 0.25
    sleeves[1, 2:] = 0.25
    executed = np.array([0.10, 0.25, 0.25, 0.25])  # first name filled short
    projected = project_sleeves_onto_book(
        sleeves=sleeves, reference_weights=executed, pretrade_weights=executed
    )
    assert projected.sum() == pytest.approx(1.0)
    assert projected.sum(axis=0) == pytest.approx(executed / executed.sum())


def test_a_name_no_sleeve_holds_is_not_attributed_to_one() -> None:
    sleeves = np.zeros((2, 4), dtype=np.float64)
    sleeves[0, 0] = 0.5
    sleeves[1, 1] = 0.5
    executed = np.array([0.4, 0.4, 0.2, 0.0])  # a third name appeared from elsewhere
    projected = project_sleeves_onto_book(
        sleeves=sleeves, reference_weights=executed, pretrade_weights=executed
    )
    assert projected[:, 2] == pytest.approx(np.zeros(2))


def test_an_empty_book_is_refused_rather_than_silently_restaged() -> None:
    sleeves = np.zeros((2, 4), dtype=np.float64)
    sleeves[0, 0] = 0.5
    with pytest.raises(TrancheExecutionError, match="tranche_execution_book_empty"):
        project_sleeves_onto_book(
            sleeves=sleeves,
            reference_weights=np.zeros(4),
            pretrade_weights=np.zeros(4),
        )


def test_rebuilt_sleeve_state_carries_the_aggregate_cap_feedback() -> None:
    """When the cap binds, the sleeves must be scaled by the same per-name ratio."""

    books = (
        np.array([0.6, 0.4, 0.0, 0.0]),
        np.array([0.6, 0.0, 0.4, 0.0]),
    )
    shares = (0.5, 0.5)
    uncapped = 0.5 * books[0] + 0.5 * books[1]
    capped = np.array([0.30, 0.35, 0.35, 0.0])  # what an aggregate cap would return

    rebuilt = rebuild_sleeve_state(
        sleeve_weights=books, sleeve_shares=shares, target_weights=capped
    )
    assert rebuilt.sum(axis=0) == pytest.approx(capped)
    assert uncapped[0] > capped[0]


def test_a_hold_carries_the_drifted_book_and_takes_no_decision() -> None:
    provider = _provider()
    _drive(provider, formations=1, growth=np.zeros(LISTINGS))
    before = provider.sleeve_state
    assert before is not None
    book = before.sum(axis=0)

    held = provider(
        formation_index=1,
        reference_weights=book,
        pretrade_weights=book,
        decision_mode="HOLD",
    )
    assert held.decision_mode == "HOLD"
    assert held.predicted_variance is None
    assert held.requires_risk_forecast is False
    assert held.target_weights == pytest.approx(book)
    # A hold reviews nothing, so no formation is recorded against it.
    assert [entry[0] for entry in provider.reviewed_sleeves_by_formation] == [0]


def test_the_provider_refuses_a_formation_outside_its_resolved_inputs() -> None:
    provider = _provider(count=2)
    with pytest.raises(TrancheExecutionError, match="tranche_execution_formation_invalid"):
        provider(
            formation_index=5,
            reference_weights=np.zeros(LISTINGS),
            pretrade_weights=np.zeros(LISTINGS),
            decision_mode="REBALANCE",
        )


def test_the_provider_refuses_inputs_off_the_listing_axis() -> None:
    with pytest.raises(TrancheExecutionError, match="tranche_execution_axis_invalid"):
        TrancheBookDecisionProvider(
            recipe=RECIPE,
            formations=(
                TrancheFormationInputs(
                    formation_session=SESSIONS[0],
                    scores=np.zeros(7),
                    decision_eligible=np.ones(7, dtype=np.bool_),
                    risk_allocation=None,
                    causal_rank_return_curve=_curve(0, SESSIONS[0]),
                ),
            ),
            ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
            sector_exposure_matrix=np.zeros((1, LISTINGS)),
            equal_weight_sector_exposure=np.zeros(1),
        )


def test_sleeve_state_is_exposed_as_a_copy_not_a_handle() -> None:
    """A caller that could mutate the carried state could rewrite a walk."""

    provider = _provider()
    _drive(provider, formations=1, growth=np.zeros(LISTINGS))
    first = provider.sleeve_state
    assert first is not None
    first[:] = 0.0
    second = provider.sleeve_state
    assert second is not None
    assert second.sum() == pytest.approx(1.0)


@dataclass(frozen=True)
class _Mandate:
    bootstrap_resamples: int = 64


@dataclass(frozen=True)
class _Workspace:
    """The smallest thing satisfying ``PortfolioBacktestWorkspace``.

    Written against the Protocol rather than borrowed from Strategy Lab's own
    development workspace, so this test exercises the capability engine's
    contract and nothing larger.
    """

    mandate: _Mandate
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    execution_available: np.ndarray
    realized_simple_returns: np.ndarray
    causal_adv20: np.ndarray
    sector_exposure_matrix: np.ndarray
    equal_weight_sector_exposure: np.ndarray
    passive_returns_by_session: dict[date, np.ndarray]


def _engine_workspace(formations: int) -> _Workspace:
    sessions = SESSIONS[:formations]
    returns = np.tile(np.linspace(0.004, -0.002, LISTINGS), (formations, 1))
    return _Workspace(
        mandate=_Mandate(),
        formation_sessions=sessions,
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        execution_available=np.ones((formations, LISTINGS), dtype=np.bool_),
        realized_simple_returns=returns,
        causal_adv20=np.full((formations, LISTINGS), 5_000_000.0),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
        passive_returns_by_session={
            session: returns[index] for index, session in enumerate(sessions)
        },
    )


def test_the_provider_drives_the_installed_engine_end_to_end() -> None:
    """The composition claim, exercised rather than asserted in prose.

    The executor contributes no loop and no mechanics: turnover, costs and the
    path all come back from the shared engine. If the provider did not satisfy
    the engine's decision contract, this is where it would fail rather than in
    a hand-rolled driver that happens to agree with it.
    """

    formations = 9
    provider = _provider(count=formations)
    result = run_portfolio_walk_forward(
        workspace=_engine_workspace(formations),
        decision_provider=provider,
        bootstrap_seed=11,
    )

    assert len(result.gross_log_returns) == formations
    assert len(result.net_log_returns_5bps) == formations
    # Costs come from the engine's own convention, so a higher rate must keep
    # less. Asserting the ordering rather than a level keeps this a statement
    # about who owns the arithmetic.
    assert sum(result.net_log_returns_20bps) < sum(result.net_log_returns_5bps)
    assert sum(result.net_log_returns_5bps) <= sum(result.gross_log_returns)

    # And the executor really did schedule: every sleeve reviewed at least once.
    reviewed = {
        sleeve for _, sleeves in provider.reviewed_sleeves_by_formation for sleeve in sleeves
    }
    assert reviewed == set(range(RECIPE.tranches))


def test_the_engine_driven_walk_leaves_consistent_sleeve_state() -> None:
    formations = 7
    provider = _provider(count=formations)
    run_portfolio_walk_forward(
        workspace=_engine_workspace(formations),
        decision_provider=provider,
        bootstrap_seed=5,
    )
    sleeves = provider.sleeve_state
    assert sleeves is not None
    assert sleeves.shape == (RECIPE.tranches, LISTINGS)
    assert float(sleeves.sum()) == pytest.approx(1.0)
    assert (sleeves >= 0.0).all()


def test_a_risk_projection_for_another_session_is_refused() -> None:
    """Only the executor knows the formation session, so it makes this check."""

    stale = TrancheFormationInputs(
        formation_session=SESSIONS[0],
        scores=np.random.default_rng(1).normal(size=LISTINGS),
        decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
        risk_allocation=_risk(SESSIONS[5]),
        risk_attribution=_attribution(SESSIONS[5]),
        causal_rank_return_curve=_curve(0, SESSIONS[0]),
    )
    provider = TrancheBookDecisionProvider(
        recipe=RECIPE,
        formations=(stale,),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
    )
    with pytest.raises(TrancheExecutionError, match="risk_session_mismatch"):
        provider(
            formation_index=0,
            reference_weights=np.zeros(LISTINGS),
            pretrade_weights=np.zeros(LISTINGS),
            decision_mode="REBALANCE",
        )


def test_a_missing_risk_projection_is_refused_for_a_risk_consuming_rule() -> None:
    absent = TrancheFormationInputs(
        formation_session=SESSIONS[0],
        scores=np.random.default_rng(1).normal(size=LISTINGS),
        decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
        risk_allocation=None,
        causal_rank_return_curve=_curve(0, SESSIONS[0]),
    )
    provider = TrancheBookDecisionProvider(
        recipe=RECIPE,
        formations=(absent,),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
    )
    with pytest.raises(TrancheExecutionError, match="risk_projection_absent"):
        provider(
            formation_index=0,
            reference_weights=np.zeros(LISTINGS),
            pretrade_weights=np.zeros(LISTINGS),
            decision_mode="REBALANCE",
        )


def test_the_executor_refuses_a_risk_projection_tampered_after_sealing() -> None:
    projection = _risk(SESSIONS[0])
    forged = replace(
        projection,
        per_name_volatility=np.asarray(projection.per_name_volatility) * 1.01,
    )
    formations = (
        TrancheFormationInputs(
            formation_session=SESSIONS[0],
            scores=np.random.default_rng(2).normal(size=LISTINGS),
            decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
            risk_allocation=forged,
            risk_attribution=_attribution(SESSIONS[0]),
            causal_rank_return_curve=_curve(0, SESSIONS[0]),
        ),
    )
    provider = TrancheBookDecisionProvider(
        recipe=RECIPE,
        formations=formations,
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
    )
    with pytest.raises(TrancheExecutionError, match="risk_projection_invalid"):
        provider(
            formation_index=0,
            reference_weights=np.zeros(LISTINGS),
            pretrade_weights=np.zeros(LISTINGS),
            decision_mode="REBALANCE",
        )


def test_the_risk_recipe_may_not_change_inside_one_walk() -> None:
    """Two valid Risk representations spliced into one ledger would be invisible."""

    payload = INSTALLED_RISK_DECOMPOSITION_RECIPE.model_dump(mode="json", exclude={"recipe_hash"})
    payload["conditional_volatility_decay"] = 0.90
    other = RiskDecompositionRecipe(**payload, recipe_hash=canonical_hash(payload))
    formations = (
        _formations(2)[0],
        replace(
            _formations(2)[1],
            risk_allocation=_risk(SESSIONS[1], recipe=other),
            risk_attribution=_attribution(SESSIONS[1], recipe=other),
        ),
    )
    provider = TrancheBookDecisionProvider(
        recipe=RECIPE,
        formations=formations,
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
    )
    _drive(provider, formations=1, growth=np.zeros(LISTINGS))
    sleeves = provider.sleeve_state
    assert sleeves is not None
    book = sleeves.sum(axis=0)
    with pytest.raises(TrancheExecutionError, match="risk_recipe_changed"):
        provider(
            formation_index=1,
            reference_weights=book,
            pretrade_weights=book,
            decision_mode="REBALANCE",
        )


def test_the_curve_must_belong_to_the_exact_formation() -> None:
    resolved = replace(
        _formations(1)[0],
        causal_rank_return_curve=_curve(1, SESSIONS[0]),
    )
    provider = TrancheBookDecisionProvider(
        recipe=RECIPE,
        formations=(resolved,),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
    )
    with pytest.raises(TrancheExecutionError, match="curve_formation_mismatch"):
        provider(
            formation_index=0,
            reference_weights=np.zeros(LISTINGS),
            pretrade_weights=np.zeros(LISTINGS),
            decision_mode="REBALANCE",
        )


def test_the_resolved_inputs_must_match_the_backtesting_session_axis() -> None:
    shifted_session = SESSIONS[1]
    original = _formations(1)[0]
    shifted = replace(
        original,
        formation_session=shifted_session,
        risk_allocation=_risk(shifted_session),
        risk_attribution=_attribution(shifted_session),
        causal_rank_return_curve=_curve(0, shifted_session),
    )
    # The axis check moved to where every component is opened, so it now covers
    # a merged book too rather than only a single tranche provider.
    with pytest.raises(TrancheExecutionError, match="workspace_session_axis_mismatch"):
        open_component_book(
            (
                TrancheBookComponent(
                    component_id="single",
                    allocation_basis_points=10_000,
                    recipe=RECIPE,
                    formations=(shifted,),
                    ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
                    sector_exposure_matrix=np.zeros((1, LISTINGS)),
                    equal_weight_sector_exposure=np.zeros(1),
                ),
            ),
            initial_sleeve_weights=None,
            schedule_offset=0,
            formation_sessions=_engine_workspace(1).formation_sessions,
        )


def test_the_walk_records_every_risk_lane_it_consumed() -> None:
    provider = _provider(count=4)
    _drive(provider, formations=4, growth=np.zeros(LISTINGS))
    consumed = provider.consumed_risk_projection_hashes
    assert len(consumed) == 4
    assert all(len(value) == 64 for value in consumed)


def test_the_executor_holds_no_covariance_matrix() -> None:
    """``TrancheFormationInputs`` carries a lane, not an ``N x N``."""

    fields = {field for field in TrancheFormationInputs.__dataclass_fields__}
    assert "covariance" not in fields
    assert "risk_allocation" in fields
    projection = _formations(1)[0].risk_allocation
    assert projection is not None
    assert projection.per_name_volatility.ndim == 1


def test_a_walk_too_short_for_a_rebalance_is_refused_before_the_book_opens() -> None:
    """requirement (class): every formation a strategy book or a research update
    walks selects each component's names from those both tradable and scored, so a formation
    with fewer is refused by its session before the book opens -- the tranche book and the
    capped sleeve book alike, a flat start and a continuation, for names unscored or
    untradable -- never in the middle of the walk; a short formation past the walk's end is
    not the walk's."""

    listings = tuple(f"L{i:03d}" for i in range(LISTINGS))
    flat = _formations(6)
    unscored = replace(flat[3], scores=np.where(np.arange(LISTINGS) < 20, flat[3].scores, np.nan))
    untradable = replace(flat[3], decision_eligible=np.arange(LISTINGS) < 34)

    def components(formations: tuple[TrancheFormationInputs, ...]) -> tuple[object, ...]:
        return (
            TrancheBookComponent(
                component_id="tranche",
                allocation_basis_points=10_000,
                recipe=RECIPE,
                formations=formations,
                ordered_listing_ids=listings,
                sector_exposure_matrix=np.zeros((1, LISTINGS)),
                equal_weight_sector_exposure=np.zeros(1),
            ),
            CappedSleeveComponent(
                component_id="capped",
                allocation_basis_points=10_000,
                top_k=35,
                exit_rank=70,
                tranches=3,
                aggregate_name_cap=0.06,
                aggregate_cap_start_formation=0,
                # The capped book's equal weight reads no Risk lane, so it is given none.
                formations=tuple(
                    replace(
                        value,
                        risk_allocation=None,
                        risk_attribution=None,
                        causal_rank_return_curve=None,
                    )
                    for value in formations
                ),
                ordered_listing_ids=listings,
            ),
        )

    for short in (unscored, untradable):
        formations = (*flat[:3], short, *flat[4:])
        for component in components(formations):
            with pytest.raises(TrancheExecutionError) as refused:
                open_component_book(
                    (component,),  # type: ignore[arg-type]
                    initial_sleeve_weights=None,
                    schedule_offset=0,
                    formation_sessions=SESSIONS[:6],
                )
            assert str(refused.value) == f"portfolio_strategy_lab.eligible_pool_short:{SESSIONS[3]}"
            # A walk that ends before the short formation is not refused for it.
            open_component_book(
                (component,),  # type: ignore[arg-type]
                initial_sleeve_weights=None,
                schedule_offset=0,
                formation_sessions=SESSIONS[:3],
            )
        # A continuation resuming on the short formation, as a research update does.
        for component in components(formations[3:]):
            with pytest.raises(TrancheExecutionError, match="eligible_pool_short"):
                open_component_book(
                    (component,),  # type: ignore[arg-type]
                    initial_sleeve_weights=np.full((3, LISTINGS), 1.0 / (3 * LISTINGS)),
                    schedule_offset=3,
                    formation_sessions=SESSIONS[3:6],
                )
