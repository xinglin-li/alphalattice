"""Numerical goldens for the active regularized-linear model owner."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    RegularizedLinearAdapter,
    RegularizedLinearParameters,
    build_regularized_linear_recipe,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from tests.alpha_research.fixtures import registered_test_cards, synthetic_prepared_arrays


def test_registered_ridge_weights_its_penalty_and_never_rewrites_the_panel() -> None:
    """Registered ridge weights its penalty and never rewrites the panel."""
    prepared = synthetic_prepared_arrays()
    adapter = RegularizedLinearAdapter()
    for card in registered_test_cards()[2:]:
        assert card.alpha is not None
        recipe = build_regularized_linear_recipe(
            RegularizedLinearParameters(family="ridge", alpha=card.alpha)
        )
        for fold in prepared.folds:
            fitted = adapter.fit(
                recipe=recipe,
                inputs=BoundAlphaTrainingInput(
                    training_binding_hash=fold.commitment.commitment_hash,
                    ordered_feature_ids=fold.ordered_factor_ids,
                    features=fold.training_features,
                    targets=fold.training_targets,
                ),
            )
            actual = adapter.predict(
                estimator=fitted.estimator_content,
                inputs=BoundAlphaPredictionInput(
                    training_binding_hash=fold.commitment.commitment_hash,
                    ordered_feature_ids=fold.ordered_factor_ids,
                    features=fold.validation_features,
                ),
            ).predictions
            training_before = fold.training_features.copy()
            validation_before = fold.validation_features.copy()
            penalty_scale = fold.training_features.std(axis=0)
            penalty_scale = np.where(penalty_scale > 0.0, penalty_scale, 1.0)
            expected = (
                Ridge(
                    alpha=card.alpha,
                    fit_intercept=True,
                    solver="svd",
                    tol=1e-8,
                )
                .fit(fold.training_features / penalty_scale, fold.training_targets)
                .predict(fold.validation_features / penalty_scale)
            )
            superseded = (
                Ridge(
                    alpha=card.alpha,
                    fit_intercept=True,
                    solver="svd",
                    tol=1e-8,
                )
                .fit(fold.training_features, fold.training_targets)
                .predict(fold.validation_features)
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            # The successor objective is a real change, not a respelling.
            assert not np.allclose(actual, superseded, rtol=1e-8, atol=1e-8)
            # The Panel keeps sole ownership of feature content.
            np.testing.assert_array_equal(fold.training_features, training_before)
            np.testing.assert_array_equal(fold.validation_features, validation_before)
