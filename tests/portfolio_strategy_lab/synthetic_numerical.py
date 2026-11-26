"""The synthetic numerical authority of the decision update suites.

A frozen strategy package, a planned QA schedule, deterministic prices and
scores, one causal rank curve per component and a bound checkpoint, built in
memory. Synthetic authority has its own axis and is labelled as such in its
hash; nothing here is strategy evidence or a product artifact.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from alphalattice.capabilities.portfolio_backtesting.segments import (
    run_portfolio_walk_forward_segment,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import planned_local_qa_schedule
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    FrozenRankCalibrationRule,
    PreparedPortfolioBookInput,
    PreparedPortfolioComponentInput,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
    PortfolioEntryBook,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
    PackagePolicyIdentity,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    CappedSleeveBookProvider,
    ComponentBookPlan,
    TrancheFormationInputs,
    build_component_book,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (
    FrozenHistoricalBookRecipe,
    FrozenHistoricalComponentRecipe,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.local_web_support import TEST_PACKAGE

HASH = canonical_hash("synthetic QA authority; not strategy evidence")


def build_numerical(
    *,
    installed_package=None,
    listing_ids=None,
    first_day=date(2026, 7, 23),
    model_hash=HASH,
    model_recipe_hash=None,
    listing_labels=None,
    epoch_start=date(2026, 7, 1),
    epoch_end=date(2026, 9, 30),
    book_recipe=None,
    model_hashes=(),
    model_recipe_hashes=(),
):
    # Synthetic authority has its own axis. Align both frozen activation anchors;
    # production materialization retains the research axis rather than recreating it.
    schedule = planned_local_qa_schedule(date(2019, 8, 13), date(2026, 9, 30))
    sessions = tuple(v.formation_session for v in schedule)
    first = sessions.index(first_day)
    ids = listing_ids or tuple(f"qa-{i:03d}" for i in range(80))
    base = installed_package or TEST_PACKAGE
    recipe = book_recipe or FrozenHistoricalBookRecipe(
        strategy_id=base.strategy_id,
        components=(FrozenHistoricalComponentRecipe(base.component_ids[0], 10_000, "mu.iv0"),),
    )
    if installed_package is None:
        values = {
            name: getattr(base, name) for name in type(base).model_fields if name != "package_hash"
        }
        values.update(
            policy_binding="PACKAGE_FROZEN",
            frozen_policy=PackagePolicyIdentity(
                policy_recipe_hash=recipe.recipe_hash,
                policy_catalog_hash=HASH,
                policy_adapter_binding_hash=HASH,
            ),
            component_plan=tuple(
                base.component_plan[0].model_copy(
                    update={
                        "component_id": component.component_id,
                        "allocation_basis_points": component.allocation_basis_points,
                        "recipe_hash": base.component_plan[0].recipe_hash
                        if len(recipe.components) == 1
                        else canonical_hash(component.component_id),
                    }
                )
                for component in recipe.components
            ),
            merge_semantics="SINGLE_COMPONENT_BOOK"
            if len(recipe.components) == 1
            else "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS",
        )
        package = FrozenStrategyPackage.create(**values)
    else:
        package = installed_package
    width = len(ids)
    j = np.arange(width)[None, :]
    t = np.arange(len(sessions) + 2)[:, None]
    prices = 100 * np.exp(0.0001 * t + 0.06 * np.sin(t / 11 + j / 13))
    scores = np.sin(t[: len(sessions)] / 7 + j / 5)
    curves = np.tile(np.linspace(0.002, 0.0001, 20), (len(sessions), 1))
    component_scores = [
        scores if i == 0 else np.cos(t[: len(sessions)] / 9 + j / 11)
        for i in range(len(recipe.components))
    ]
    for component, values in zip(recipe.components, component_scores, strict=True):
        if component.outsider_sentinel is not None:
            for row in values:
                order = np.argsort(-row, kind="stable")
                row[order[100:]] = component.outsider_sentinel
    formation_sets = tuple(
        tuple(
            TrancheFormationInputs(
                formation_session=day,
                scores=values[i],
                decision_eligible=values[i] != component.outsider_sentinel
                if component.outsider_sentinel is not None
                else np.ones(width, dtype=bool),
                risk_allocation=None,
                causal_rank_return_curve=CausalRankReturnCurveSlice(
                    formation_index=i,
                    formation_session=day,
                    bucket_means=curves[i],
                    bucket_support_counts=(252,) * 20,
                    admitted_formation_count=252,
                    disposition="AVAILABLE",
                    curve_hash=HASH,
                )
                if component.weight_rule == "mu.iv0"
                else None,
            )
            for i, day in enumerate(sessions)
        )
        for component, values in zip(recipe.components, component_scores, strict=True)
    )

    def provider():
        return build_component_book(
            tuple(
                ComponentBookPlan(
                    component.component_id,
                    component.allocation_basis_points,
                    recipe.tranches,
                    CappedSleeveBookProvider(
                        top_k=recipe.top_k,
                        exit_rank=recipe.exit_rank,
                        tranches=recipe.tranches,
                        aggregate_name_cap=recipe.aggregate_name_cap,
                        aggregate_cap_start_formation=recipe.aggregate_cap_start_formation,
                        formations=formations,
                        ordered_listing_ids=ids,
                        weight_rule=component.weight_rule,
                        review_phase=recipe.review_phase,
                        sizing_activation_formation=recipe.sizing_activation_formation,
                    ),
                )
                for component, formations in zip(recipe.components, formation_sets, strict=True)
            )
        )

    returns = prices[2:] / prices[1:-1] - 1.0
    workspace = SimpleNamespace(
        formation_sessions=sessions,
        ordered_listing_ids=ids,
        execution_available=np.ones(scores.shape, dtype=bool),
        realized_simple_returns=returns,
        causal_adv20=np.zeros(scores.shape),
        capacity_evidence_available=False,
        sector_exposure_matrix=np.zeros((1, width)),
        equal_weight_sector_exposure=np.zeros(1),
    )
    owner = provider()
    captured = []

    class EntryReached(Exception):
        pass

    def capture(index, weights, cash):
        if index == first - 1:
            captured.append(
                PortfolioEntryBook.create(
                    schedule=schedule[index],
                    next_position=index + 1,
                    weights=tuple(weights),
                    cash=cash,
                    sleeves=tuple(tuple(row) for row in owner.sleeve_state),
                    component_sleeve_counts=(recipe.tranches,) * len(recipe.components)
                    if len(recipe.components) > 1
                    else (),
                )
            )
            raise EntryReached

    with pytest.raises(EntryReached):
        run_portfolio_walk_forward_segment(
            workspace=workspace,
            decision_provider=owner,
            start_index=0,
            stop_index=first,
            entry_observer=capture,
        )
    hashes = model_hashes or (model_hash,) * len(recipe.components)
    recipes = model_recipe_hashes or tuple(plan.recipe_hash for plan in package.component_plan)
    checkpoint = PortfolioDecisionCheckpoint.create(
        package=package,
        recipe=recipe,
        ordered_listing_ids=ids,
        listing_labels=listing_labels or ids,
        model_authority_hash=model_hash
        if len(recipe.components) == 1
        else canonical_hash(tuple(zip(package.component_ids, hashes, strict=True))),
        inference_recipe_hash=(model_recipe_hash or package.component_plan[0].recipe_hash)
        if len(recipe.components) == 1
        else canonical_hash(tuple(zip(package.component_ids, recipes, strict=True))),
        component_authority_hashes=hashes if len(recipe.components) > 1 else (),
        component_recipe_hashes=recipes if len(recipe.components) > 1 else (),
        epoch_start=epoch_start,
        epoch_end=epoch_end,
        formation_sessions=sessions,
        initial_book=captured[0],
        source_hashes=(HASH,),
    )
    reference = run_portfolio_walk_forward_segment(
        workspace=workspace, decision_provider=provider(), start_index=0, stop_index=first + 4
    )
    return SimpleNamespace(
        checkpoint=checkpoint,
        first=first,
        schedule=schedule,
        sessions=sessions,
        prices=prices,
        scores=scores,
        component_scores=component_scores,
        curves=curves,
        reference=reference,
    )


def prepared_for(n, index):
    c = n.checkpoint
    if len(c.recipe.components) > 1:
        return PreparedPortfolioBookInput.create(
            component_ids=c.package.component_ids,
            components=tuple(
                PreparedPortfolioComponentInput.create(
                    request_hash=HASH,
                    score_snapshot_hash=HASH,
                    observation_hash=None,
                    strategy_package_hash=c.package.package_hash,
                    component_recipe_hash=c.model_recipe_hashes[i],
                    formation_session=n.sessions[index],
                    rule=None,
                    ordered_listing_ids=c.ordered_listing_ids,
                    scores=tuple(n.component_scores[i][index]),
                    decision_eligible=tuple(
                        float(value) != part.outsider_sentinel
                        for value in n.component_scores[i][index]
                    ),
                    bucket_means=(),
                    selected_observation_sessions=(),
                    latest_holding_end_session=None,
                    sizing_rule="ew",
                )
                for i, part in enumerate(c.recipe.components)
            ),
        )
    return PreparedPortfolioComponentInput.create(
        request_hash=HASH,
        score_snapshot_hash=HASH,
        observation_hash=HASH,
        strategy_package_hash=c.package.package_hash,
        component_recipe_hash=c.inference_recipe_hash,
        formation_session=n.sessions[index],
        rule=FrozenRankCalibrationRule.create(activation_session=date(2021, 9, 7)),
        ordered_listing_ids=c.ordered_listing_ids,
        scores=tuple(n.scores[index]),
        decision_eligible=(True,) * len(c.ordered_listing_ids),
        bucket_means=tuple(n.curves[index]),
        selected_observation_sessions=n.sessions[index - 253 : index - 1],
        latest_holding_end_session=n.sessions[index],
        sizing_rule="mu.iv0",
    )


def snapshot_for(n, index, *, prices=None):
    values = n.prices if prices is None else prices
    bars = tuple(
        RawDailyBar(
            listing_id=listing,
            provider="SYNTHETIC_QA",
            session_date=n.sessions[i],
            open=float(values[i, j]),
            high=float(values[i, j] * 1.02),
            low=float(values[i, j] * 0.98),
            close=float(values[i, j] * (1 + 0.002 * np.sin(j))),
            volume=100_000,
        )
        for i in range(n.first - 1, index + 1)
        for j, listing in enumerate(n.checkpoint.ordered_listing_ids)
    )
    return LocalQAMarketSnapshot.create(
        source_hash=HASH,
        through=n.sessions[index],
        ordered_listing_ids=n.checkpoint.ordered_listing_ids,
        schedule=n.schedule[n.first - 1 : index + 3],
        bars=bars,
        actions=(),
    )


__all__ = ["HASH", "build_numerical", "prepared_for", "snapshot_for"]
