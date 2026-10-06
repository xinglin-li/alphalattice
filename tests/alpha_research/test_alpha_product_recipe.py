"""The frozen IW184 product recipe, and the claim Gate 8A actually has to establish.

The load-bearing test is not that the recipe holds the right numbers -- that is
transcription. It is that the point it pins is admissible by the *installed*
adapter, and that the recipe resolves into that adapter's own type. If it did
not, Gate 8B would discover at fit time that the product recipe describes a model
the product cannot build.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID,
    DYNAMIC_PANEL_LIGHTGBM_SEEDS,
    DynamicPanelLightGBMParameters,
    build_dynamic_panel_lightgbm_recipe,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (
    build_panel_model_catalog,
)
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
    AlphaProductRecipe,
    AlphaProductRecipeError,
)


def test_the_pinned_point_is_admissible_by_the_installed_adapter() -> None:
    """Every seed resolves into the adapter's own bounded parameter type.

    ``DynamicPanelLightGBMParameters`` validates against declared bounds on
    construction, so this failing would mean the frozen product point sits
    outside the installed space.
    """

    resolved = INSTALLED_ALPHA_PRODUCT_RECIPE.resolve_estimator_parameters()
    assert len(resolved) == 3
    assert all(isinstance(value, DynamicPanelLightGBMParameters) for value in resolved)
    assert tuple(value.seed for value in resolved) == INSTALLED_ALPHA_PRODUCT_RECIPE.seeds


def test_the_resolved_point_builds_the_installed_recipe_envelope() -> None:
    """The recipe reaches the real adapter route, not a parallel one."""

    for parameters in INSTALLED_ALPHA_PRODUCT_RECIPE.resolve_estimator_parameters():
        envelope = build_dynamic_panel_lightgbm_recipe(parameters)
        assert envelope.adapter_id == DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID


def test_the_adapter_this_recipe_targets_is_already_installed() -> None:
    """Gate 8A reuses an installed adapter; it does not add a second one.

    The entry-gate record originally claimed this adapter was installed in no
    catalog. It is installed here, and that correction is what stopped a
    duplicate adapter being written.
    """

    catalog = build_panel_model_catalog()
    assert DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID in catalog.adapter_ids

    # And the installed adapter resolves this recipe's own envelope, so the
    # membership above is a live route rather than a name in a tuple.
    parameters = INSTALLED_ALPHA_PRODUCT_RECIPE.resolve_estimator_parameters()[0]
    resolved = catalog.resolve(build_dynamic_panel_lightgbm_recipe(parameters))
    assert resolved.adapter_id == DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID


def test_the_product_seeds_are_the_installed_seed_set() -> None:
    assert INSTALLED_ALPHA_PRODUCT_RECIPE.seeds == DYNAMIC_PANEL_LIGHTGBM_SEEDS


def test_the_frozen_scientific_identity() -> None:
    """The four choices that make this recipe what it is, plus the axis it was fitted on."""

    recipe = INSTALLED_ALPHA_PRODUCT_RECIPE
    assert recipe.training_window_sessions == 1_260
    assert recipe.purge_sessions == 1
    assert recipe.refit_quarter_start_months == (1, 4, 7, 10)
    assert recipe.vintage_weights == (4, 3, 2, 1)
    assert recipe.feature_count == 184
    assert recipe.feature_axis_hash == (
        "7d307c7637b046da4fca9eb05e5e19946635ab58e4a8904c115d6b7b7091d976"
    )


def test_vintage_shares_are_the_declared_ratio() -> None:
    assert INSTALLED_ALPHA_PRODUCT_RECIPE.vintage_weight_shares() == (0.4, 0.3, 0.2, 0.1)


def test_the_refit_calendar_admits_only_its_own_quarters() -> None:
    recipe = INSTALLED_ALPHA_PRODUCT_RECIPE
    assert [month for month in range(1, 13) if recipe.is_refit_month(month)] == [1, 4, 7, 10]


def test_live_model_count_cannot_disagree_with_seeds_times_vintages() -> None:
    """Twelve is derived, not asserted.

    A recipe that states twelve while carrying two seeds and four vintages is the
    failure this check exists for, and it is the kind of drift a Literal alone
    would not catch.
    """

    recipe = INSTALLED_ALPHA_PRODUCT_RECIPE
    assert recipe.live_model_count == len(recipe.seeds) * recipe.vintage_count

    payload = recipe.model_dump(mode="json")
    payload["seeds"] = [1729, 2718]
    with pytest.raises(ValidationError, match="product_recipe_model_count_invalid"):
        AlphaProductRecipe.model_validate(payload)


def test_vintage_weights_must_decay_from_the_newest_vintage() -> None:
    """4:3:2:1 is an ordering claim, not just four numbers."""

    payload = INSTALLED_ALPHA_PRODUCT_RECIPE.model_dump(mode="json")
    payload["vintage_weights"] = [1, 2, 3, 4]
    with pytest.raises(ValidationError, match="product_recipe_vintage_order_invalid"):
        AlphaProductRecipe.model_validate(payload)


def test_identity_is_tamper_evident() -> None:
    payload = INSTALLED_ALPHA_PRODUCT_RECIPE.model_dump(mode="json")
    payload["feature_axis_hash"] = "0" * 64
    with pytest.raises(ValidationError, match="product_recipe_identity_invalid"):
        AlphaProductRecipe.model_validate(payload)


def test_the_evidence_disposition_is_carried_not_upgraded() -> None:
    """Citing a diagnostic package must not read as promoting it.

    The support window also ends exactly at the evidence firewall date, which is
    the property that keeps a recipe from quietly widening the admitted region.
    """

    recipe = INSTALLED_ALPHA_PRODUCT_RECIPE
    assert recipe.evidence_disposition == "DEVELOPMENT_DIAGNOSTIC_CANDIDATE_NOT_INSTALLED"
    assert recipe.evidence_support_last == date(2024, 8, 12)
    assert recipe.researcher_payload()["evidence_disposition"] == (
        "DEVELOPMENT_DIAGNOSTIC_CANDIDATE_NOT_INSTALLED"
    )


def test_the_recipe_round_trips_and_is_stable() -> None:
    recipe = INSTALLED_ALPHA_PRODUCT_RECIPE
    assert AlphaProductRecipe.model_validate_json(recipe.model_dump_json()) == recipe
    assert AlphaProductRecipe.installed().recipe_hash == recipe.recipe_hash


def test_a_recipe_error_is_a_stable_typed_refusal() -> None:
    assert issubclass(AlphaProductRecipeError, ValueError)
