"""Exact live-score aggregation and existing Portfolio-consumer boundaries."""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt
import pytest

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    build_dynamic_panel_lightgbm_recipe,
)
from alphalattice.capabilities.alpha_modeling.contracts import AlphaEstimatorContent
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    COMPONENT_IDS,
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    ComponentId,
    HeterogeneousAlphaComponentRecipe,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    HeterogeneousFormationScoreInput,
    HeterogeneousLiveModel,
    HeterogeneousScoringError,
    HeterogeneousVintageFeatureSurface,
    live_vintages,
    percentile_rank,
    score_heterogeneous_component,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    CappedSleeveComponent,
    TrancheFormationInputs,
    open_component_book,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    INSTALLED_HETEROGENEOUS_BOOK_RECIPE,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

FORMATION = date(2026, 7, 29)
LISTINGS = tuple(f"L{index:03d}" for index in range(105))
FEATURE_IDS = ("synthetic_feature",)


def _component(component_id: ComponentId) -> HeterogeneousAlphaComponentRecipe:
    values = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id).model_dump(
        mode="python", exclude={"recipe_hash"}
    )
    values.update(
        feature_axis_id=f"{component_id}_SYNTHETIC_AXIS",
        feature_axis_hash=canonical_hash(list(FEATURE_IDS)),
        feature_count=1,
        ordered_feature_ids=FEATURE_IDS,
    )
    return HeterogeneousAlphaComponentRecipe.create(**values)


def _models(component: HeterogeneousAlphaComponentRecipe) -> tuple[HeterogeneousLiveModel, ...]:
    values = []
    for vintage in live_vintages(FORMATION, count=4):
        for seed in component.seeds:
            estimator = AlphaEstimatorContent.create(
                adapter_id="synthetic_live_boundary",
                content_format_id="synthetic.estimator",
                ordered_feature_ids=FEATURE_IDS,
                payload={"vintage": vintage, "seed": seed},
            )
            values.append(
                HeterogeneousLiveModel(
                    vintage=vintage,
                    seed=seed,
                    recipe_hash=build_dynamic_panel_lightgbm_recipe(
                        component.estimator_point.resolve(seed=seed)
                    ).recipe_hash,
                    training_binding_hash=canonical_hash(
                        {"kind": "SyntheticTrainingBinding", "vintage": vintage, "seed": seed}
                    ),
                    lineage_hash=canonical_hash(
                        {"kind": "SyntheticModelLineage", "vintage": vintage, "seed": seed}
                    ),
                    estimator=estimator,
                )
            )
    return tuple(values)


def _inputs(
    component: HeterogeneousAlphaComponentRecipe,
) -> HeterogeneousFormationScoreInput:
    base = np.linspace(-1.0, 1.0, len(LISTINGS), dtype=np.float64)
    vintages = live_vintages(FORMATION, count=4)
    surfaces = tuple(
        HeterogeneousVintageFeatureSurface.create(
            vintage=vintage,
            ordered_listing_ids=LISTINGS,
            ordered_feature_ids=FEATURE_IDS,
            features=(base + 0.03 * index)[:, None],
            source_binding_hash=canonical_hash(
                {"kind": "SyntheticFeatureSource", "vintage": vintage}
            ),
        )
        for index, vintage in enumerate(vintages)
    )
    return HeterogeneousFormationScoreInput.create(
        formation_session=FORMATION,
        ordered_listing_ids=LISTINGS,
        decision_eligible=np.ones(len(LISTINGS), dtype=np.bool_),
        raw_12_1_momentum=np.arange(len(LISTINGS), dtype=np.float64),
        feature_surfaces=surfaces,
        models=_models(component),
        model_set_manifest_hash=canonical_hash(
            {"kind": "SyntheticLiveModelSet", "component": component.component_id}
        ),
    )


def _predict(
    *,
    component: HeterogeneousAlphaComponentRecipe,
    model: HeterogeneousLiveModel,
    features: FloatArray,
) -> FloatArray:
    del component
    x = np.asarray(features[:, 0], dtype=np.float64)
    if model.seed == 1729:
        return x
    if model.seed == 2718:
        return -0.6 * x
    return np.square(x) + 0.1 * x


def test_g0_ranks_each_child_before_seed_and_vintage_aggregation() -> None:
    component = _component("G0_IW184")
    inputs = _inputs(component)
    projection = score_heterogeneous_component(
        component=component,
        inputs=inputs,
        prediction_owner=_predict,
    )

    expected = np.zeros(len(LISTINGS), dtype=np.float64)
    for surface, weight in zip(inputs.feature_surfaces, component.vintage_weights, strict=True):
        seed_ranks = [
            percentile_rank(
                _predict(
                    component=component,
                    model=next(
                        value
                        for value in inputs.models
                        if value.vintage == surface.vintage and value.seed == seed
                    ),
                    features=surface.features,
                )
            )
            for seed in component.seeds
        ]
        expected += float(weight) * np.mean(np.vstack(seed_ranks), axis=0) / 10.0

    np.testing.assert_allclose(projection.scores, expected, rtol=0.0, atol=2e-16)
    assert projection.live_name_count == len(LISTINGS)
    assert projection.aggregation_semantics.startswith("WITHIN_SESSION_PERCENTILE")


def test_specialist_averages_raw_seeds_before_ranking_and_keeps_outsiders() -> None:
    component = _component("G2_R0_TREND")
    inputs = _inputs(component)
    projection = score_heterogeneous_component(
        component=component,
        inputs=inputs,
        prediction_owner=_predict,
    )

    outsiders = np.arange(5)
    np.testing.assert_array_equal(projection.scores[outsiders], np.full(5, -1.0))
    assert np.isfinite(projection.scores).all()
    assert projection.live_name_count == len(LISTINGS)
    assert projection.scores[5:].min() >= 0.01
    assert projection.scores[5:].max() <= 1.0
    assert projection.aggregation_semantics.startswith("EQUAL_RAW_SEED_MEAN")


def test_incomplete_model_set_refuses_before_prediction() -> None:
    component = _component("G7_R1_CONTEXTUAL_MOMENTUM")
    inputs = _inputs(component)
    incomplete = HeterogeneousFormationScoreInput.create(
        formation_session=inputs.formation_session,
        ordered_listing_ids=inputs.ordered_listing_ids,
        decision_eligible=inputs.decision_eligible,
        raw_12_1_momentum=inputs.raw_12_1_momentum,
        feature_surfaces=inputs.feature_surfaces,
        models=inputs.models[:-1],
        model_set_manifest_hash=inputs.model_set_manifest_hash,
    )

    with pytest.raises(
        HeterogeneousScoringError,
        match=r"alpha_research\.heterogeneous_live_model_set_incomplete",
    ):
        score_heterogeneous_component(
            component=component,
            inputs=incomplete,
            prediction_owner=_predict,
        )


def test_four_live_projections_enter_the_existing_heterogeneous_book() -> None:
    projections = {
        component_id: score_heterogeneous_component(
            component=(component := _component(component_id)),
            inputs=_inputs(component),
            prediction_owner=_predict,
        )
        for component_id in COMPONENT_IDS
    }
    formations = {
        component_id: (
            TrancheFormationInputs(
                formation_session=FORMATION,
                scores=projection.scores,
                decision_eligible=projection.live,
                risk_allocation=None,
                risk_attribution=None,
                causal_rank_return_curve=None,
                score_projection=projection,
            ),
        )
        for component_id, projection in projections.items()
    }
    book = INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe
    provider = open_component_book(
        tuple(
            CappedSleeveComponent(
                component_id=component_id,
                allocation_basis_points=2_500,
                top_k=book.top_k,
                exit_rank=book.exit_rank,
                tranches=book.tranches,
                aggregate_name_cap=book.aggregate_name_cap,
                aggregate_cap_start_formation=book.aggregate_cap_start_formation,
                formations=formations[component_id],
                ordered_listing_ids=LISTINGS,
            )
            for component_id in COMPONENT_IDS
        ),
        initial_sleeve_weights=None,
        schedule_offset=0,
    )
    decision = provider(
        formation_index=0,
        reference_weights=np.zeros(len(LISTINGS), dtype=np.float64),
        pretrade_weights=np.zeros(len(LISTINGS), dtype=np.float64),
        decision_mode="REBALANCE",
    )

    assert decision.requires_risk_forecast is False
    assert float(decision.target_weights.sum()) == pytest.approx(1.0)
    # Each component admits its own live projection, so the merged book records
    # all four rather than one digest over them.
    assert provider.consumed_score_projection_hashes == tuple(
        projections[value].projection_hash for value in COMPONENT_IDS
    )
    assert all(np.isfinite(decision.target_weights))
