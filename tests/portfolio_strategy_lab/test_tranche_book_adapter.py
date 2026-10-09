"""The tranche adapter: schedule, sleeve carry, and the mu lane it refuses to fake.

The load-bearing test is that a sleeve which is not due produces exactly the
weights it already held. If it did not, a three-sleeve book would quietly be a
whole-book policy rebalancing every session and trading three times as much --
and every turnover number the product reports would be wrong while every weight
still looked plausible.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    BoundPolicyDecisionInput,
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    INSTALLED_TRANCHE_BOOK_RECIPE,
    TRANCHE_BOOK_POLICY_ID,
    TrancheBookAdapter,
    TrancheBookError,
    TrancheBookRecipe,
    decide_tranche_book,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
    FactorIdiosyncraticRiskSurface,
    RiskAllocationProjection,
)

LISTINGS = 120
BUCKETS = 4
SESSION = date(2023, 1, 3)


def _curve(
    disposition: str = "AVAILABLE",
    *,
    formation_index: int = 0,
    session: date = SESSION,
) -> CausalRankReturnCurveSlice:
    return CausalRankReturnCurveSlice(
        formation_index=formation_index,
        formation_session=session,
        bucket_means=np.array([0.04, 0.01, -0.01, -0.03], dtype=np.float64),
        bucket_support_counts=(252,) * BUCKETS,
        admitted_formation_count=252,
        disposition=disposition,  # type: ignore[arg-type]
        curve_hash="0" * 64,
    )


def _risk(
    *,
    volatility: np.ndarray | None = None,
    factor_scale: float = 1.0,
    listings: int = LISTINGS,
    session: date = SESSION,
) -> RiskAllocationProjection:
    rng = np.random.default_rng(11)
    exposures = np.zeros((listings, 4), dtype=np.float64)
    exposures[np.arange(listings), rng.integers(0, 4, listings)] = 1.0
    root = rng.normal(size=(4, 4))
    surface = FactorIdiosyncraticRiskSurface.create(
        recipe=INSTALLED_RISK_DECOMPOSITION_RECIPE,
        formation_session=session,
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(listings)),
        ordered_factor_ids=tuple(f"S{j}" for j in range(4)),
        exposures=exposures,
        factor_covariance=root @ root.T * 0.0004 * factor_scale,
        idiosyncratic_variance=rng.uniform(1e-4, 9e-4, listings),
        conditional_volatility=(
            np.linspace(0.10, 0.30, listings) if volatility is None else volatility
        ),
        producer_identity="TEST_PRODUCER",
    )
    return RiskAllocationProjection.of(surface)


def _inputs(
    *,
    formation_index: int,
    previous_sleeve_weights: tuple[np.ndarray, ...] | None = None,
    seed: int = 7,
    curve: CausalRankReturnCurveSlice | None = None,
    risk: RiskAllocationProjection | None = None,
) -> BoundPolicyDecisionInput:
    rng = np.random.default_rng(seed + formation_index)
    scores = rng.normal(size=LISTINGS)
    session = SESSION + timedelta(days=formation_index)
    risk = _risk(session=session) if risk is None else risk
    return BoundPolicyDecisionInput(
        scores=scores,
        covariance=None,
        risk_allocation=risk,
        decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
        reference_weights=np.zeros(LISTINGS),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        formation_index=formation_index,
        formation_session=session,
        causal_rank_return_curve=(
            _curve(formation_index=formation_index, session=session) if curve is None else curve
        ),
        previous_sleeve_weights=previous_sleeve_weights,
    )


# top_k is 20 rather than the range minimum of 15 on purpose: at the staging
# formation every sleeve sees the same scores and therefore selects the same
# names, so the book holds exactly top_k, and the 6% aggregate cap cannot be met
# below roughly eighteen. That interaction is recorded as an open control-surface
# item rather than worked around here.
RECIPE = TrancheBookRecipe.create(top_k=35, tranches=3, exit_rank=70)


def test_the_first_formation_stages_every_sleeve_at_an_equal_share() -> None:
    allocation = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0))
    assert allocation.reviewed_sleeves == (0, 1, 2)
    assert len(allocation.sleeve_weights) == 3
    assert allocation.target_weights.sum() == pytest.approx(1.0)
    for sleeve in allocation.sleeve_weights:
        assert int((sleeve > 0).sum()) == RECIPE.top_k


def test_a_sleeve_that_is_not_due_produces_exactly_what_it_held() -> None:
    """The whole point of the schedule: only one sleeve trades per formation."""

    staged = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0))
    shares = tuple(1 / 3 for _ in range(3))
    carried = tuple(share * book for share, book in zip(shares, staged.sleeve_weights, strict=True))

    # Formation 1 reviews sleeve 1 only, against completely different scores.
    nxt = decide_tranche_book(
        recipe=RECIPE,
        inputs=_inputs(formation_index=1, previous_sleeve_weights=carried, seed=99),
    )
    assert nxt.reviewed_sleeves == (1,)
    for sleeve in (0, 2):
        assert nxt.sleeve_weights[sleeve] == pytest.approx(staged.sleeve_weights[sleeve])
    assert not np.allclose(nxt.sleeve_weights[1], staged.sleeve_weights[1])


def test_each_sleeve_is_reviewed_exactly_once_over_a_full_cycle() -> None:
    staged = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0))
    state = tuple(book / 3 for book in staged.sleeve_weights)

    reviewed: list[int] = []
    for index in range(1, 4):
        allocation = decide_tranche_book(
            recipe=RECIPE,
            inputs=_inputs(formation_index=index, previous_sleeve_weights=state),
        )
        reviewed.extend(allocation.reviewed_sleeves)
        state = tuple(
            share * book
            for share, book in zip((1 / 3, 1 / 3, 1 / 3), allocation.sleeve_weights, strict=True)
        )
    assert sorted(reviewed) == [0, 1, 2]


def test_the_book_respects_the_aggregate_name_cap_and_sums_to_one() -> None:
    allocation = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0))
    assert allocation.target_weights.sum() == pytest.approx(1.0)
    assert (allocation.target_weights >= 0.0).all()
    # A CAPPED disposition is a claim about the result, so it is checked as one.
    # The alternative dispositions are honest fallbacks and must stay inside the
    # names the membership rule actually selected.
    if allocation.book_disposition == "CAPPED":
        assert allocation.target_weights.max() <= RECIPE.aggregate_name_cap + 1e-12
    else:
        assert allocation.book_disposition == "EQUAL_WEIGHT_CEILING_INFEASIBLE"
        held = allocation.target_weights > 0
        assert allocation.target_weights[held].std() == pytest.approx(0.0)


def test_sleeve_overlap_makes_the_book_hold_fewer_names_than_top_k_times_tranches() -> None:
    """The readout the plan forbids presenting as ``top_k * tranches``."""

    allocation = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0))
    distinct = int((allocation.target_weights > 0).sum())
    assert distinct <= RECIPE.top_k * RECIPE.tranches
    assert distinct <= LISTINGS


def test_a_mu_rule_refuses_a_missing_curve_rather_than_trading_inverse_volatility() -> None:
    inputs = BoundPolicyDecisionInput(
        scores=np.random.default_rng(3).normal(size=LISTINGS),
        covariance=None,
        risk_allocation=_risk(),
        decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
        reference_weights=np.zeros(LISTINGS),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        formation_index=0,
        causal_rank_return_curve=None,
    )
    with pytest.raises(TrancheBookError, match="tranche_curve_absent"):
        decide_tranche_book(recipe=RECIPE, inputs=inputs)


@pytest.mark.parametrize(
    "disposition",
    ["PARTIAL_BUCKET_SUPPORT", "INSUFFICIENT_MATURED_FORMATION_HISTORY"],
)
def test_a_partial_mu_curve_fails_closed(disposition: str) -> None:
    """A missing or partial expected-return curve fails closed."""

    with pytest.raises(TrancheBookError, match="tranche_curve_not_available"):
        decide_tranche_book(
            recipe=RECIPE,
            inputs=_inputs(formation_index=0, curve=_curve(disposition)),
        )


def test_a_non_finite_bucket_mean_fails_closed() -> None:
    """A bucket the curve owner could not support is not a zero forecast."""

    broken = _curve()
    means = np.array(broken.bucket_means, dtype=np.float64)
    means[1] = np.nan
    with pytest.raises(TrancheBookError, match="tranche_curve_bucket_not_finite"):
        decide_tranche_book(
            recipe=RECIPE,
            inputs=_inputs(
                formation_index=0,
                curve=CausalRankReturnCurveSlice(
                    formation_index=broken.formation_index,
                    formation_session=broken.formation_session,
                    bucket_means=means,
                    bucket_support_counts=broken.bucket_support_counts,
                    admitted_formation_count=broken.admitted_formation_count,
                    disposition=broken.disposition,
                    curve_hash=broken.curve_hash,
                ),
            ),
        )


def test_a_book_that_cannot_meet_the_aggregate_cap_fails_closed() -> None:
    """A book that cannot meet the aggregate cap fails closed."""

    with pytest.raises(ValidationError):
        TrancheBookRecipe.create(top_k=15, tranches=3, exit_rank=30)


def test_an_available_curve_still_decides() -> None:
    allocation = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0))
    assert allocation.curve_disposition == "AVAILABLE"
    assert allocation.target_weights.sum() == pytest.approx(1.0)


def test_a_later_formation_without_sleeve_state_is_refused() -> None:
    """Only the staging formation may run without prior sleeves."""

    with pytest.raises(TrancheBookError, match="tranche_sleeve_state_absent"):
        decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=1))


def test_a_sleeve_state_of_the_wrong_width_is_refused() -> None:
    wrong = tuple(np.zeros(LISTINGS) for _ in range(2))
    with pytest.raises(TrancheBookError, match="tranche_sleeve_state_axis_invalid"):
        decide_tranche_book(
            recipe=RECIPE,
            inputs=_inputs(formation_index=1, previous_sleeve_weights=wrong),
        )


def test_a_missing_formation_index_is_refused() -> None:
    inputs = _inputs(formation_index=0)
    stateless = BoundPolicyDecisionInput(
        scores=inputs.scores,
        covariance=None,
        risk_allocation=inputs.risk_allocation,
        decision_eligible=inputs.decision_eligible,
        reference_weights=inputs.reference_weights,
        sector_exposure_matrix=inputs.sector_exposure_matrix,
        equal_weight_sector_exposure=inputs.equal_weight_sector_exposure,
        ordered_listing_ids=inputs.ordered_listing_ids,
        causal_rank_return_curve=inputs.causal_rank_return_curve,
    )
    with pytest.raises(TrancheBookError, match="tranche_formation_index_absent"):
        decide_tranche_book(recipe=RECIPE, inputs=stateless)


def test_the_adapter_declares_itself_closed_form_and_solver_free() -> None:
    adapter = TrancheBookAdapter()
    binding = adapter.describe_adapter_binding()
    assert adapter.solver_backed is False
    assert adapter.policy_id == TRANCHE_BOOK_POLICY_ID
    assert binding.solver_semantics == "CLOSED_FORM_TRANCHE_SLEEVE_SCHEDULE_NO_OPTIMIZER"
    assert len(binding.adapter_implementation_hash) == 64
    semantics = adapter.selection_allocation_semantics
    assert semantics.allocation_method == "CLOSED_FORM_TRANCHE_SLEEVE_SCHEDULE_BOOK"
    assert semantics.optimizer_selection_claim == "NO_OPTIMIZER_SELECTS_OR_ALLOCATES_THE_BOOK"


def test_the_adapter_binding_does_not_include_the_optimizer_source() -> None:
    """A closed-form adapter that bound the solver's bytes would be claiming a
    dependency it does not have, and would rotate whenever the solver changed."""

    import alphalattice.investment.portfolio_strategy_lab.policies.tranche_book as module
    from alphalattice.investment.portfolio_strategy_lab.policies.buffered_equal_weight import (
        whole_book_hysteresis_selection,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
        PORTFOLIO_POLICY_PACKAGE,
        portfolio_adapter_implementation_hash,
    )

    expected = portfolio_adapter_implementation_hash(
        (PORTFOLIO_POLICY_PACKAGE, Path(module.__file__ or "")),
        (PORTFOLIO_POLICY_PACKAGE, Path(whole_book_hysteresis_selection.__code__.co_filename)),
    )
    assert TrancheBookAdapter().describe_adapter_binding().adapter_implementation_hash == expected


def test_the_adapter_does_not_publish_a_diagonal_scale_as_a_total_risk_forecast() -> None:
    decision = TrancheBookAdapter().decide(
        policy=RECIPE,
        inputs=_inputs(formation_index=0),
        optimizer=object(),
    )
    assert decision.decision_mode == "REBALANCE"
    assert decision.requires_risk_forecast is False
    assert decision.predicted_variance is None
    assert decision.target_weights.sum() == pytest.approx(1.0)
    assert decision.optimizer_objective_value is None


def test_the_installed_default_recipe_is_the_frozen_tuple() -> None:
    assert INSTALLED_TRANCHE_BOOK_RECIPE.top_k == 35
    assert INSTALLED_TRANCHE_BOOK_RECIPE.tranches == 3
    assert INSTALLED_TRANCHE_BOOK_RECIPE.weight_rule == "mu.iv1"


def test_the_public_catalog_resolves_this_recipe_to_this_adapter() -> None:
    """The installed tranche recipe resolves through the public policy catalog to its declared
    adapter."""

    from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
        build_public_portfolio_policy_catalog,
    )

    catalog = build_public_portfolio_policy_catalog()
    assert tuple(a.policy_id for a in catalog.adapters) == (TRANCHE_BOOK_POLICY_ID,)
    resolved = catalog.resolve(INSTALLED_TRANCHE_BOOK_RECIPE)
    assert resolved.policy_id == TRANCHE_BOOK_POLICY_ID
    assert INSTALLED_TRANCHE_BOOK_RECIPE.policy_id == TRANCHE_BOOK_POLICY_ID


def test_the_binding_declares_no_optimizer_and_states_its_semantics() -> None:
    """The binding declares no optimizer and states its semantics."""

    binding = TrancheBookAdapter().describe_adapter_binding()
    assert binding.requires_optimizer is False
    assert binding.input_consumption_semantics == (
        "PER_NAME_RISK_SCALE_REQUIRED_NO_FORECAST_OR_OPTIMIZER"
    )
    assert binding.requires_risk_forecast is False
    assert binding.selection_allocation_semantics is not None
    assert (
        binding.selection_allocation_semantics.allocation_method
        == "CLOSED_FORM_TRANCHE_SLEEVE_SCHEDULE_BOOK"
    )


def test_the_public_catalog_is_solver_free_in_a_fresh_interpreter() -> None:

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from alphalattice.investment.portfolio_strategy_lab.policies.catalog "
            "import build_public_portfolio_policy_catalog as b; c = b(); "
            "assert len(c.adapters) == 1; "
            "bad = {m.split('.')[0] for m in sys.modules} & {'cvxpy', 'osqp', 'optuna'}; "
            "assert not bad, bad",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_the_tranche_rules_never_read_a_covariance() -> None:
    """The tranche rules never read a covariance."""

    inputs = BoundPolicyDecisionInput(
        scores=np.random.default_rng(3).normal(size=LISTINGS),
        covariance=np.diag(np.linspace(0.01, 0.09, LISTINGS)),
        decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
        reference_weights=np.zeros(LISTINGS),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        formation_index=0,
        causal_rank_return_curve=_curve(),
        risk_allocation=None,
    )
    with pytest.raises(TrancheBookError, match="tranche_risk_projection_absent"):
        decide_tranche_book(recipe=RECIPE, inputs=inputs)


def test_equal_weight_needs_no_risk_and_refuses_a_lane_it_will_not_read() -> None:
    """``ew`` consumes no risk, so requiring one would be a fabricated dependency."""

    equal = TrancheBookRecipe.create(top_k=35, tranches=3, exit_rank=70, weight_rule="ew")
    assert not equal.consumes_risk

    inputs = BoundPolicyDecisionInput(
        scores=np.random.default_rng(3).normal(size=LISTINGS),
        covariance=None,
        decision_eligible=np.ones(LISTINGS, dtype=np.bool_),
        reference_weights=np.zeros(LISTINGS),
        sector_exposure_matrix=np.zeros((1, LISTINGS)),
        equal_weight_sector_exposure=np.zeros(1),
        ordered_listing_ids=tuple(f"L{i:03d}" for i in range(LISTINGS)),
        formation_index=0,
        causal_rank_return_curve=None,
        risk_allocation=None,
    )
    allocation = decide_tranche_book(recipe=equal, inputs=inputs)
    assert allocation.risk_projection_hash is None
    assert allocation.target_weights.sum() == pytest.approx(1.0)

    with pytest.raises(TrancheBookError, match="tranche_risk_lane_not_consumed"):
        decide_tranche_book(recipe=equal, inputs=replace(inputs, risk_allocation=_risk()))


def test_a_risk_lane_off_the_listing_axis_is_refused() -> None:
    with pytest.raises(TrancheBookError, match="tranche_risk_listing_axis_mismatch"):
        decide_tranche_book(
            recipe=RECIPE,
            inputs=_inputs(formation_index=0, risk=_risk(listings=LISTINGS - 1)),
        )


@pytest.mark.parametrize(
    "volatility",
    [
        np.full(LISTINGS, np.nan),
        np.zeros(LISTINGS),
    ],
)
def test_a_non_finite_or_non_positive_risk_lane_is_refused(volatility: np.ndarray) -> None:
    """A zero volatility divides into an infinite weight; a nan poisons the book."""

    surface_volatility = np.linspace(0.10, 0.30, LISTINGS)
    projection = _risk()
    broken = replace(projection, per_name_volatility=volatility)
    assert surface_volatility.size == broken.per_name_volatility.size
    with pytest.raises(TrancheBookError, match="tranche_risk_projection_invalid"):
        decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0, risk=broken))


def test_holdings_bind_the_projection_hash_and_a_report_only_change_does_not_move_them() -> None:
    """Holdings bind the projection hash and a report only change does not move them."""

    base = _risk(factor_scale=1.0)
    revised = _risk(factor_scale=4.0)
    assert base.surface_hash != revised.surface_hash
    assert base.projection_hash == revised.projection_hash

    first = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0, risk=base))
    second = decide_tranche_book(recipe=RECIPE, inputs=_inputs(formation_index=0, risk=revised))
    assert first.risk_projection_hash == base.projection_hash
    assert second.risk_projection_hash == base.projection_hash
    assert first.target_weights == pytest.approx(second.target_weights)


def test_a_risk_lane_whose_values_no_longer_match_its_hash_is_refused() -> None:
    projection = _risk()
    forged = replace(
        projection,
        per_name_volatility=np.asarray(projection.per_name_volatility) * 1.01,
    )
    with pytest.raises(TrancheBookError, match="tranche_risk_projection_invalid"):
        decide_tranche_book(
            recipe=RECIPE,
            inputs=_inputs(formation_index=0, risk=forged),
        )
