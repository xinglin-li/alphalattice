"""Development-only robust linear Alpha model capability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from sklearn.linear_model import HuberRegressor

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

HUBER_ADAPTER_ID = "huber_linear"
HUBER_RECIPE_SCHEMA_ID = "alpha-model.huber-linear"
HUBER_SEARCH_DOMAIN_SCHEMA_ID = "alpha-model.huber-linear.search-domain"
HUBER_CONTENT_FORMAT_ID = "alpha-model.huber-linear.coefficients"


@dataclass(frozen=True, slots=True)
class HuberParameters:
    """An admitted robust-loss threshold and regularization strength."""

    epsilon: float
    alpha: float

    def __post_init__(self) -> None:
        """Reject parameters outside the fixed Huber development grid.

        Raises:
            ValueError: The threshold or strength is not admitted.

        """
        if self.epsilon not in {1.1, 1.35, 1.7} or self.alpha not in {1e-5, 1e-4, 1e-3}:
            raise ValueError("ALPHA_HUBER_PARAMETER_INVALID")


def build_huber_recipe(parameters: HuberParameters) -> AlphaModelRecipeEnvelope:
    """Build the routed recipe envelope for admitted Huber parameters.

    Args:
        parameters: A point on the fixed development grid.

    Returns:
        The recipe consumed by the Huber adapter.

    """
    return AlphaModelRecipeEnvelope.create(
        adapter_id=HUBER_ADAPTER_ID,
        recipe_schema_id=HUBER_RECIPE_SCHEMA_ID,
        parameters={"epsilon": parameters.epsilon, "alpha": parameters.alpha},
    )


def build_huber_search_domain() -> AlphaModelSearchDomainEnvelope:
    """Build the fixed Huber parameter search domain.

    Returns:
        The installed threshold and regularization grid.

    """
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=HUBER_ADAPTER_ID,
        recipe_schema_id=HUBER_RECIPE_SCHEMA_ID,
        search_domain_schema_id=HUBER_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={"allowed_epsilon": [1.1, 1.35, 1.7], "allowed_alpha": [1e-5, 1e-4, 1e-3]},
    )


class HuberLinearAdapter:
    """Fit and predict with the admitted robust linear estimator."""

    adapter_id = HUBER_ADAPTER_ID
    recipe_schema_id = HUBER_RECIPE_SCHEMA_ID
    search_domain_schema_id = HUBER_SEARCH_DOMAIN_SCHEMA_ID

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Declare the Huber implementation and single-thread runtime requirements.

        Returns:
            The numerical binding for this adapter.

        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=HUBER_CONTENT_FORMAT_ID,
            implementation_owners=("sklearn.linear_model.HuberRegressor",),
            deterministic_policy={"fit_intercept": True, "max_iter": 1000, "tol": 1e-8},
            required_runtime_capabilities=(
                "numpy",
                "scikit-learn",
                SINGLE_THREAD_RUNTIME_CAPABILITY,
            ),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        """Admit direct fits for the Huber recipe family.

        Args:
            recipe: Unused; every admitted recipe has the same protocol.

        Returns:
            The sole supported direct-fit protocol.

        """
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> HuberParameters:
        """Check recipe routing and parse its admitted parameter tuple.

        Args:
            recipe: Envelope to validate against this adapter's route.

        Returns:
            Parsed, admitted Huber parameters.

        Raises:
            ValueError: The route or parameter tuple is invalid.

        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_HUBER_RECIPE_ROUTE_INVALID")
        return HuberParameters(**recipe.parameters)

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> dict[str, object]:
        """Require the exact installed Huber search domain.

        Args:
            domain: Search domain to compare with the installed grid.

        Returns:
            The admitted constraints.

        Raises:
            ValueError: The search domain differs from the installed one.

        """
        expected = build_huber_search_domain()
        if domain != expected:
            raise ValueError("ALPHA_HUBER_SEARCH_DOMAIN_INVALID")
        return dict(domain.constraints)

    def validate_recipe_for_domain(
        self, *, recipe: AlphaModelRecipeEnvelope, domain: AlphaModelSearchDomainEnvelope
    ) -> HuberParameters:
        """Check that the recipe's parameters lie inside the installed domain.

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
        allowed_epsilon = cast(Sequence[float], constraints["allowed_epsilon"])
        allowed_alpha = cast(Sequence[float], constraints["allowed_alpha"])
        if parameters.epsilon not in allowed_epsilon or parameters.alpha not in allowed_alpha:
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return parameters

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Fit Huber regression and seal its coefficients as hexadecimal values.

        Args:
            recipe: Admitted Huber recipe.
            inputs: Bound training matrix and targets.
            fit_plan: Optional direct-fit plan bound to the training input.

        Returns:
            Coefficient content, state projection and training error.

        Raises:
            ValueError: The fit plan or recipe is outside the admitted binding.

        """
        if fit_plan is not None and (
            fit_plan.protocol_id != "DIRECT_FIT"
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_HUBER_FIT_PLAN_INVALID")
        parameters = self.validate_recipe(recipe)
        estimator = HuberRegressor(
            epsilon=parameters.epsilon,
            alpha=parameters.alpha,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-8,
        ).fit(inputs.features, inputs.targets)
        coefficients: np.ndarray = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = float(estimator.intercept_)
        predictions = np.asarray(estimator.predict(inputs.features), dtype=np.float64)
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=HUBER_CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={
                "coefficient_hex": tuple(float(value).hex() for value in coefficients),
                "intercept_hex": intercept.hex(),
            },
        )
        projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="huber_linear",
            state_kind="ROBUST_LINEAR_COEFFICIENTS",
            state_schema_id="alpha-model.huber-linear.development-state",
            payload={"coefficient_hex": content.payload["coefficient_hex"]},
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
        """Predict from bound coefficient content and a matching feature matrix.

        Args:
            estimator: Sealed hexadecimal coefficient content.
            inputs: Bound prediction matrix with the estimator's feature order.

        Returns:
            A read-only prediction vector.

        Raises:
            ValueError: The estimator or feature binding is invalid.

        """
        if (
            estimator.adapter_id != self.adapter_id
            or estimator.content_format_id != HUBER_CONTENT_FORMAT_ID
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_HUBER_ESTIMATOR_BINDING_INVALID")
        coefficients: np.ndarray = np.asarray(
            [float.fromhex(value) for value in estimator.payload["coefficient_hex"]],
            dtype=np.float64,
        )
        intercept = float.fromhex(str(estimator.payload["intercept_hex"]))
        predictions = np.asarray(inputs.features @ coefficients + intercept, dtype=np.float64)
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


__all__ = [
    "HUBER_ADAPTER_ID",
    "HuberLinearAdapter",
    "HuberParameters",
    "build_huber_recipe",
    "build_huber_search_domain",
]
