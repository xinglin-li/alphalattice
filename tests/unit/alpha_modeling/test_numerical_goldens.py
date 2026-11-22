"""Numerical goldens for the active regularized-linear model owner."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    RegularizedLinearAdapter,
    RegularizedLinearParameters,
    build_regularized_linear_recipe,
    fit_regularized_linear,
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


@pytest.mark.proves("alpha_modeling.sparse_multiplier_counts_from_alpha_max")
@pytest.mark.parametrize("ratio", [None, 0.5], ids=["lasso", "elastic_net"])
def test_a_sparse_multiplier_counts_from_the_alpha_that_zeroes_every_coefficient(ratio) -> None:
    """A sparse multiplier counts from the data's own alpha-max: zero above it, a fit below it."""
    family = "lasso" if ratio is None else "elastic_net"
    rng = np.random.default_rng(31)
    features = 3.0 + rng.normal(size=(40, 3)) * np.array([1.0, 0.5, 2.0])
    # The strongest link is negative and the mean is not zero: alpha-max needs |.| and centering.
    targets = 0.5 - 0.05 * features[:, 0] + 0.01 * rng.normal(size=40)

    def fit(multiplier: float, values: np.ndarray = targets) -> np.ndarray:
        parameters = RegularizedLinearParameters(
            family, alpha_max_multiplier=multiplier, l1_ratio=ratio
        )
        return fit_regularized_linear(
            parameters=parameters,
            training_features=features,
            training_targets=values,
            prediction_features=features[:1],
        ).coefficients

    assert not fit(1.001).any() and fit(0.99).any()
    with pytest.raises(ValueError, match=r"^ALPHA_REGULARIZED_LINEAR_ALPHA_MAX_INVALID$"):
        fit(0.5, np.full(40, 0.25))
