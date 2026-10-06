"""Conditional development HistGradientBoosting Alpha capability."""

from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ..contracts import (
    AlphaEstimatorContent,
    AlphaModelFitProtocol,
    AlphaModelFitResult,
    AlphaModelNumericalBinding,
    AlphaModelPredictionResult,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    AlphaModelStateProjection,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from ..runtime.numerical_environment import SINGLE_THREAD_RUNTIME_CAPABILITY

HIST_GRADIENT_BOOSTING_ADAPTER_ID = "hist_gradient_boosting"
HIST_GRADIENT_BOOSTING_RECIPE_SCHEMA_ID = "alpha-model.hist-gradient-boosting"
HIST_GRADIENT_BOOSTING_SEARCH_DOMAIN_SCHEMA_ID = "alpha-model.hist-gradient-boosting.search-domain"
HIST_GRADIENT_BOOSTING_CONTENT_FORMAT_ID = "alpha-model.hist-gradient-boosting.sklearn-pickle"


@dataclass(frozen=True, slots=True)
class HistGradientBoostingParameters:
    """The admitted rates, tree widths and regularization strengths for this adapter."""

    learning_rate: float
    max_leaf_nodes: int
    l2_regularization: float

    def __post_init__(self) -> None:
        """Reject a parameter tuple outside the fixed development grid.

        Raises:
            ValueError: Any parameter is not one of the admitted values.

        """
        if (
            self.learning_rate not in {0.03, 0.05}
            or self.max_leaf_nodes not in {15, 31}
            or self.l2_regularization not in {0.1, 1.0}
        ):
            raise ValueError("ALPHA_HIST_GRADIENT_BOOSTING_PARAMETER_INVALID")


def build_hist_gradient_boosting_recipe(
    parameters: HistGradientBoostingParameters,
) -> AlphaModelRecipeEnvelope:
    """Build a recipe envelope from admitted HistGradientBoosting parameters.

    Args:
        parameters: A point on the fixed development grid.

    Returns:
        The versioned recipe envelope consumed by the adapter.

    """
    return AlphaModelRecipeEnvelope.create(
        adapter_id=HIST_GRADIENT_BOOSTING_ADAPTER_ID,
        recipe_schema_id=HIST_GRADIENT_BOOSTING_RECIPE_SCHEMA_ID,
        parameters={
            "learning_rate": parameters.learning_rate,
            "max_leaf_nodes": parameters.max_leaf_nodes,
            "l2_regularization": parameters.l2_regularization,
        },
    )


def build_hist_gradient_boosting_search_domain() -> AlphaModelSearchDomainEnvelope:
    """Build the fixed search domain and training limits for this adapter.

    Returns:
        The installed parameter grid and deterministic training policy.

    """
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=HIST_GRADIENT_BOOSTING_ADAPTER_ID,
        recipe_schema_id=HIST_GRADIENT_BOOSTING_RECIPE_SCHEMA_ID,
        search_domain_schema_id=HIST_GRADIENT_BOOSTING_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={
            "allowed_learning_rate": [0.03, 0.05],
            "allowed_max_leaf_nodes": [15, 31],
            "allowed_l2_regularization": [0.1, 1.0],
            "max_iter": 300,
            "max_depth": 5,
            "min_samples_leaf": 20,
            "random_state": 1729,
        },
    )


class HistGradientBoostingAdapter:
    """Fit and predict with the admitted scikit-learn histogram boosting recipe."""

    adapter_id = HIST_GRADIENT_BOOSTING_ADAPTER_ID
    recipe_schema_id = HIST_GRADIENT_BOOSTING_RECIPE_SCHEMA_ID
    search_domain_schema_id = HIST_GRADIENT_BOOSTING_SEARCH_DOMAIN_SCHEMA_ID

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Declare the implementation owner and single-thread runtime requirements.

        Returns:
            The numerical binding for this adapter.

        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=HIST_GRADIENT_BOOSTING_CONTENT_FORMAT_ID,
            implementation_owners=("sklearn.ensemble.HistGradientBoostingRegressor",),
            deterministic_policy={
                "max_iter": 300,
                "max_depth": 5,
                "min_samples_leaf": 20,
                "early_stopping": False,
                "random_state": 1729,
            },
            required_runtime_capabilities=(
                "numpy",
                "scikit-learn",
                SINGLE_THREAD_RUNTIME_CAPABILITY,
            ),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        """Admit direct fits for this recipe family.

        Args:
            recipe: Unused; every admitted recipe has the same fit protocol.

        Returns:
            The sole supported direct-fit protocol.

        """
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> HistGradientBoostingParameters:
        """Check recipe routing and parse its fixed-grid parameters.

        Args:
            recipe: Envelope to validate against this adapter's route.

        Returns:
            Parsed, admitted model parameters.

        Raises:
            ValueError: The route or parameter tuple is not admitted.

        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_HIST_GRADIENT_BOOSTING_RECIPE_ROUTE_INVALID")
        return HistGradientBoostingParameters(**recipe.parameters)

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> dict[str, object]:
        """Require the exact installed search domain and return its constraints.

        Args:
            domain: Search domain to compare with the installed policy.

        Returns:
            The admitted constraints.

        Raises:
            ValueError: The search domain differs from the installed one.

        """
        expected = build_hist_gradient_boosting_search_domain()
        if domain != expected:
            raise ValueError("ALPHA_HIST_GRADIENT_BOOSTING_SEARCH_DOMAIN_INVALID")
        return dict(domain.constraints)

    def validate_recipe_for_domain(
        self, *, recipe: AlphaModelRecipeEnvelope, domain: AlphaModelSearchDomainEnvelope
    ) -> HistGradientBoostingParameters:
        """Check that the routed recipe lies inside the installed search domain.

        Args:
            recipe: Recipe to admit.
            domain: Search domain that defines allowed values.

        Returns:
            The recipe's admitted parameters.

        Raises:
            ValueError: Routing, domain or parameter membership is invalid.

        """
        parameters = self.validate_recipe(recipe)
        constraints = self.validate_search_domain(domain)
        if (
            parameters.learning_rate not in constraints["allowed_learning_rate"]  # type: ignore[operator]
            or parameters.max_leaf_nodes not in constraints["allowed_max_leaf_nodes"]  # type: ignore[operator]
            or parameters.l2_regularization not in constraints["allowed_l2_regularization"]  # type: ignore[operator]
        ):
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return parameters

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Fit a deterministic tree ensemble and seal its estimator content.

        Args:
            recipe: Admitted recipe for the estimator.
            inputs: Bound training matrix and targets.
            fit_plan: Optional direct-fit plan bound to the training input.

        Returns:
            Serialized estimator, state summary and training error.

        Raises:
            ValueError: The fit plan or recipe is outside the admitted binding.

        """
        if fit_plan is not None and (
            fit_plan.protocol_id != "DIRECT_FIT"
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_HIST_GRADIENT_BOOSTING_FIT_PLAN_INVALID")
        parameters = self.validate_recipe(recipe)
        estimator = HistGradientBoostingRegressor(
            learning_rate=parameters.learning_rate,
            max_leaf_nodes=parameters.max_leaf_nodes,
            l2_regularization=parameters.l2_regularization,
            max_iter=300,
            max_depth=5,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=1729,
        ).fit(inputs.features, inputs.targets)
        predictions = np.asarray(estimator.predict(inputs.features), dtype=np.float64)
        serialized = base64.b64encode(pickle.dumps(estimator, protocol=5)).decode("ascii")
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=HIST_GRADIENT_BOOSTING_CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={"estimator_base64": serialized},
        )
        projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="hist_gradient_boosting",
            state_kind="TREE_ENSEMBLE_SUMMARY",
            state_schema_id="alpha-model.hist-gradient-boosting.development-state",
            payload={"iteration_count": int(estimator.n_iter_)},
        )
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=projection,
            training_mse=float(np.mean(np.square(inputs.targets - predictions))),
            iteration_count=int(estimator.n_iter_),
        )

    def predict(
        self, *, estimator: AlphaEstimatorContent, inputs: BoundAlphaPredictionInput
    ) -> AlphaModelPredictionResult:
        """Predict from matching sealed estimator content and a bound feature matrix.

        Args:
            estimator: Serialized histogram boosting estimator content.
            inputs: Bound prediction matrix with the estimator's feature order.

        Returns:
            A read-only prediction vector.

        Raises:
            ValueError: The content format, feature order or estimator type is invalid.

        """
        if (
            estimator.adapter_id != self.adapter_id
            or estimator.content_format_id != HIST_GRADIENT_BOOSTING_CONTENT_FORMAT_ID
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_HIST_GRADIENT_BOOSTING_ESTIMATOR_BINDING_INVALID")
        fitted = pickle.loads(base64.b64decode(estimator.payload["estimator_base64"]))
        if not isinstance(fitted, HistGradientBoostingRegressor):
            raise ValueError("ALPHA_HIST_GRADIENT_BOOSTING_CONTENT_INVALID")
        predictions = np.asarray(fitted.predict(inputs.features), dtype=np.float64)
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


__all__ = [
    "HIST_GRADIENT_BOOSTING_ADAPTER_ID",
    "HistGradientBoostingAdapter",
    "HistGradientBoostingParameters",
    "build_hist_gradient_boosting_recipe",
    "build_hist_gradient_boosting_search_domain",
]
