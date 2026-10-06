"""Pure Ridge, Lasso, and Elastic Net fitting for frozen Alpha surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import ElasticNet, Lasso, Ridge

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

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RegularizedLinearParameters:
    """Host-resolved numerical parameters; public Tool inputs are validated elsewhere."""

    family: Literal["ridge", "lasso", "elastic_net"]
    alpha: float | None = None
    alpha_max_multiplier: float | None = None
    l1_ratio: float | None = None

    def __post_init__(self) -> None:
        """Validate finite family-specific linear penalty fields.

        Raises:
            ValueError: Penalty fields are nonfinite, positive penalties are absent, or
                Ridge/Lasso/Elastic-Net field shapes disagree.
        """
        values = tuple(
            value
            for value in (self.alpha, self.alpha_max_multiplier, self.l1_ratio)
            if value is not None
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("ALPHA_REGULARIZED_LINEAR_PARAMETER_NOT_FINITE")
        if self.family == "ridge":
            if self.alpha is None or self.alpha <= 0.0:
                raise ValueError("ALPHA_RIDGE_ALPHA_INVALID")
            if self.alpha_max_multiplier is not None or self.l1_ratio is not None:
                raise ValueError("ALPHA_RIDGE_PARAMETER_SHAPE_INVALID")
            return
        if self.alpha is not None or self.alpha_max_multiplier is None:
            raise ValueError("ALPHA_SPARSE_PARAMETER_SHAPE_INVALID")
        if self.alpha_max_multiplier <= 0.0:
            raise ValueError("ALPHA_SPARSE_ALPHA_MULTIPLIER_INVALID")
        if self.family == "lasso" and self.l1_ratio is not None:
            raise ValueError("ALPHA_LASSO_RATIO_NOT_ALLOWED")
        if self.family == "elastic_net" and (
            self.l1_ratio is None or not 0.0 < self.l1_ratio < 1.0
        ):
            raise ValueError("ALPHA_ELASTIC_NET_RATIO_INVALID")


@dataclass(frozen=True, slots=True)
class RegularizedLinearFit:
    """Carry immutable fitted linear coefficients, intercept, predictions and training diagnostics.

    Coefficients are expressed on the original admitted Feature axis after the
    estimator's training-fold variance penalty weighting. Iterations are absent
    for the closed-form Ridge route.
    """

    coefficients: FloatArray
    intercept: float
    training_mse: float
    predictions: FloatArray
    iteration_count: int | None


REGULARIZED_LINEAR_ADAPTER_ID = "regularized_linear"
REGULARIZED_LINEAR_RECIPE_SCHEMA_ID = "alpha-model.regularized-linear"
REGULARIZED_LINEAR_SEARCH_DOMAIN_SCHEMA_ID = "alpha-model.regularized-linear.search-domain"
REGULARIZED_LINEAR_CONTENT_FORMAT_ID = "alpha-model.regularized-linear.coefficients"
REGULARIZED_LINEAR_STATE_SCHEMA_ID = "alpha-model.regularized-linear.development-state"


@dataclass(frozen=True, slots=True)
class RegularizedLinearSearchDomain:
    """Declare admitted linear families and their numerical penalty bounds.

    Ridge alphas and sparse alpha-max multipliers have positive ordered bounds.
    Elastic-Net mixing bounds lie strictly between zero and one; family handles
    form a nonempty unique declared axis.
    """

    allowed_families: tuple[Literal["ridge", "lasso", "elastic_net"], ...]
    ridge_alpha_min: float
    ridge_alpha_max: float
    sparse_alpha_max_multiplier_min: float
    sparse_alpha_max_multiplier_max: float
    elastic_net_l1_ratio_min: float
    elastic_net_l1_ratio_max: float

    def __post_init__(self) -> None:
        """Validate unique admitted families and ordered positive penalty/mixing bounds.

        Raises:
            ValueError: Families are empty/repeated or a declared penalty/mixing interval is
                invalid.
        """
        if (
            not self.allowed_families
            or len(set(self.allowed_families)) != len(self.allowed_families)
            or self.ridge_alpha_min <= 0.0
            or self.ridge_alpha_min > self.ridge_alpha_max
            or self.sparse_alpha_max_multiplier_min <= 0.0
            or self.sparse_alpha_max_multiplier_min > self.sparse_alpha_max_multiplier_max
            or not 0.0 < self.elastic_net_l1_ratio_min <= self.elastic_net_l1_ratio_max < 1.0
        ):
            raise ValueError("ALPHA_REGULARIZED_LINEAR_SEARCH_DOMAIN_INVALID")


def _alpha_max(features: FloatArray, targets: FloatArray) -> float:
    centered = targets - float(np.mean(targets))
    value = float(np.max(np.abs(features.T @ centered)) / features.shape[0])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("ALPHA_REGULARIZED_LINEAR_ALPHA_MAX_INVALID")
    return value


def fit_regularized_linear(
    *,
    parameters: RegularizedLinearParameters,
    training_features: FloatArray,
    training_targets: FloatArray,
    prediction_features: FloatArray,
) -> RegularizedLinearFit:
    """Fit one deterministic model without preprocessing or runtime side effects.

    Args:
        parameters: Validated Ridge, Lasso or Elastic-Net numerical recipe.
        training_features: Finite admitted training matrix on the Feature axis.
        training_targets: Finite target vector aligned with training rows.
        prediction_features: Finite matrix on the same feature axis; an empty row axis is allowed.

    Returns:
        Frozen coefficients and predictions, intercept, training MSE and optional solver
        iteration count. Fold-local variance weights the estimator penalty; learned
        coefficients return to the original admitted Feature space.

    Raises:
        ValueError: Numerical shapes/values are invalid, sparse fitting reaches its
            iteration limit, or learned values cannot satisfy the result boundary.
    """
    if (
        training_features.ndim != 2
        or prediction_features.ndim != 2
        or training_targets.ndim != 1
        or len(training_features) != len(training_targets)
        or training_features.shape[1] != prediction_features.shape[1]
        or not np.isfinite(training_features).all()
        or not np.isfinite(training_targets).all()
        or not np.isfinite(prediction_features).all()
    ):
        raise ValueError("ALPHA_REGULARIZED_LINEAR_SURFACE_INVALID")
    # One penalty over columns of different scale is not one penalty. The effective
    # ridge shrinkage on a column of standard deviation c is alpha / c^2, so a
    # preprocessing lane that delivers 0.25 is regularized fifteen times harder than
    # one that delivers 1.0 -- a property of the lane, not of the Factor. The
    # registered objective therefore weights the penalty per column,
    # ``alpha_j = alpha * var(x_j)``, which is a property of the estimator rather
    # than a preprocessing stage: no value is centered, imputed or otherwise
    # rewritten, and the Panel remains the sole owner of feature content.
    penalty_scale = training_features.std(axis=0)
    penalty_scale = np.where(penalty_scale > 0.0, penalty_scale, 1.0)
    weighted_training = training_features / penalty_scale
    if parameters.family == "ridge":
        assert parameters.alpha is not None
        estimator: Ridge | Lasso | ElasticNet = Ridge(
            alpha=parameters.alpha,
            fit_intercept=True,
            solver="svd",
            tol=1e-8,
        )
    else:
        assert parameters.alpha_max_multiplier is not None
        penalty = _alpha_max(weighted_training, training_targets)
        penalty *= parameters.alpha_max_multiplier
        if parameters.family == "lasso":
            estimator = Lasso(
                alpha=penalty,
                fit_intercept=True,
                precompute=True,
                max_iter=10000,
                tol=1e-8,
                selection="cyclic",
                warm_start=False,
            )
        else:
            assert parameters.l1_ratio is not None
            estimator = ElasticNet(
                alpha=penalty / parameters.l1_ratio,
                l1_ratio=parameters.l1_ratio,
                fit_intercept=True,
                precompute=True,
                max_iter=10000,
                tol=1e-8,
                selection="cyclic",
                warm_start=False,
            )
    estimator.fit(weighted_training, training_targets)
    iteration = getattr(estimator, "n_iter_", None)
    iteration_count = int(iteration) if iteration is not None else None
    if iteration_count is not None and iteration_count >= 10000:
        raise ValueError("ALPHA_REGULARIZED_LINEAR_CONVERGENCE_FAILURE")
    # Coefficients return to the Panel's own feature space, so a sealed estimator
    # stays comparable with the Factor it belongs to rather than with a fold-local
    # weighting. Both prediction lanes then reconstruct from those coefficients, so
    # this fit and a later replay through ``predict`` agree bitwise by construction.
    coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1) / penalty_scale
    intercept = float(estimator.intercept_)
    training_predictions = np.asarray(
        training_features @ coefficients + intercept, dtype=np.float64
    )
    predictions = (
        np.asarray(prediction_features @ coefficients + intercept, dtype=np.float64)
        if len(prediction_features)
        else np.empty(0, dtype=np.float64)
    )
    training_mse = float(np.mean(np.square(training_targets - training_predictions)))
    if (
        coefficients.shape != (training_features.shape[1],)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept)
        or not np.isfinite(training_mse)
        or not np.isfinite(predictions).all()
    ):
        raise ValueError("ALPHA_REGULARIZED_LINEAR_RESULT_INVALID")
    coefficients.setflags(write=False)
    predictions.setflags(write=False)
    return RegularizedLinearFit(
        coefficients=coefficients,
        intercept=intercept,
        training_mse=training_mse,
        predictions=predictions,
        iteration_count=iteration_count,
    )


def build_regularized_linear_recipe(
    parameters: RegularizedLinearParameters,
) -> AlphaModelRecipeEnvelope:
    """Seal one numerical recipe without target or research-program authority."""
    return AlphaModelRecipeEnvelope.create(
        adapter_id=REGULARIZED_LINEAR_ADAPTER_ID,
        recipe_schema_id=REGULARIZED_LINEAR_RECIPE_SCHEMA_ID,
        parameters={
            "family": parameters.family,
            "alpha": parameters.alpha,
            "alpha_max_multiplier": parameters.alpha_max_multiplier,
            "l1_ratio": parameters.l1_ratio,
        },
    )


def build_regularized_linear_search_domain() -> AlphaModelSearchDomainEnvelope:
    """Build the current bounded Alpha Research domain for the installed adapter."""
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=REGULARIZED_LINEAR_ADAPTER_ID,
        recipe_schema_id=REGULARIZED_LINEAR_RECIPE_SCHEMA_ID,
        search_domain_schema_id=REGULARIZED_LINEAR_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={
            "allowed_families": ["ridge", "lasso", "elastic_net"],
            "ridge_alpha_min": 0.1,
            "ridge_alpha_max": 100.0,
            "sparse_alpha_max_multiplier_min": 0.01,
            "sparse_alpha_max_multiplier_max": 1.0,
            "elastic_net_l1_ratio_min": 0.25,
            "elastic_net_l1_ratio_max": 0.75,
        },
    )


def decode_regularized_linear_content(
    estimator: AlphaEstimatorContent,
) -> tuple[FloatArray, float]:
    """Decode and validate the stable coefficient representation."""
    if (
        estimator.adapter_id != REGULARIZED_LINEAR_ADAPTER_ID
        or estimator.content_format_id != REGULARIZED_LINEAR_CONTENT_FORMAT_ID
    ):
        raise ValueError("ALPHA_REGULARIZED_LINEAR_ESTIMATOR_BINDING_INVALID")
    coefficients: FloatArray = np.asarray(
        [float.fromhex(value) for value in estimator.payload["coefficient_hex"]],
        dtype=np.float64,
    )
    intercept = float.fromhex(str(estimator.payload["intercept_hex"]))
    if (
        coefficients.shape != (len(estimator.ordered_feature_ids),)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(intercept)
    ):
        raise ValueError("ALPHA_REGULARIZED_LINEAR_ESTIMATOR_CONTENT_INVALID")
    coefficients.setflags(write=False)
    return coefficients, intercept


class RegularizedLinearAdapter:
    """Existing deterministic Ridge/Lasso/Elastic-Net owner behind the Desk seam."""

    adapter_id = REGULARIZED_LINEAR_ADAPTER_ID
    recipe_schema_id = REGULARIZED_LINEAR_RECIPE_SCHEMA_ID
    search_domain_schema_id = REGULARIZED_LINEAR_SEARCH_DOMAIN_SCHEMA_ID

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Seal declared training-fold variance-weighted linear penalties and runtime requirements.

        Returns:
            Adapter/content-format handles, implementation owners, deterministic policy and required
            capabilities.
        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=REGULARIZED_LINEAR_CONTENT_FORMAT_ID,
            implementation_owners=(
                "sklearn.linear_model.Ridge",
                "sklearn.linear_model.Lasso",
                "sklearn.linear_model.ElasticNet",
            ),
            deterministic_policy={
                "fit_intercept": True,
                "penalty_weighting": "TRAINING_FOLD_PER_COLUMN_VARIANCE",
                "ridge_solver": "svd",
                "sparse_selection": "cyclic",
                "sparse_max_iter": 10000,
                "tolerance": 1e-8,
            },
            required_runtime_capabilities=(
                "numpy",
                "scikit-learn",
                SINGLE_THREAD_RUNTIME_CAPABILITY,
            ),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        """Declare the fit protocol consumed by this installed adapter.

        Args:
            recipe: Protocol argument; this adapter has no per-recipe protocol variation.

        Returns:
            The direct-fit protocol.
        """
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> RegularizedLinearParameters:
        """Validate this adapter's route and registered numerical recipe fields.

        Args:
            recipe: Qualified envelope naming this adapter and recipe schema.

        Returns:
            Parsed family-specific numerical parameters.

        Raises:
            ValueError: Recipe route or registered numerical fields are invalid.
        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_REGULARIZED_LINEAR_RECIPE_ROUTE_INVALID")
        try:
            return RegularizedLinearParameters(**recipe.parameters)
        except TypeError as error:
            # A parameter the schema has no field for, or a value of a wrong
            # kind, is a recipe outside the schema: refused as one rather than
            # surfacing as a construction error nobody can act on.
            raise ValueError("ALPHA_REGULARIZED_LINEAR_RECIPE_PARAMETERS_INVALID") from error

    def validate_search_domain(
        self, domain: AlphaModelSearchDomainEnvelope
    ) -> RegularizedLinearSearchDomain:
        """Validate the qualified search-domain route and constraints.

        Args:
            domain: Sealed constraints proposed for this adapter.

        Returns:
            Validated family and penalty bounds.

        Raises:
            ValueError: Domain route or numerical bounds are invalid.
        """
        if (
            domain.adapter_id != self.adapter_id
            or domain.recipe_schema_id != self.recipe_schema_id
            or domain.search_domain_schema_id != self.search_domain_schema_id
        ):
            raise ValueError("ALPHA_REGULARIZED_LINEAR_SEARCH_DOMAIN_ROUTE_INVALID")
        return RegularizedLinearSearchDomain(**domain.constraints)

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> RegularizedLinearParameters:
        """Require the numerical recipe to belong to its admitted bounds or profiles.

        Args:
            recipe: Qualified numerical configuration.
            domain: Host-admitted constraints for this adapter.

        Returns:
            Parsed parameters within every admitted bound/profile.

        Raises:
            ValueError: Recipe/domain routes, registered fields, bounds or paired profiles are
                invalid.
        """
        parameters = self.validate_recipe(recipe)
        admitted = self.validate_search_domain(domain)
        if parameters.family not in admitted.allowed_families:
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        if parameters.family == "ridge":
            assert parameters.alpha is not None
            valid = admitted.ridge_alpha_min <= parameters.alpha <= admitted.ridge_alpha_max
        else:
            assert parameters.alpha_max_multiplier is not None
            valid = (
                admitted.sparse_alpha_max_multiplier_min
                <= parameters.alpha_max_multiplier
                <= admitted.sparse_alpha_max_multiplier_max
            )
            if parameters.family == "elastic_net":
                assert parameters.l1_ratio is not None
                valid = valid and (
                    admitted.elastic_net_l1_ratio_min
                    <= parameters.l1_ratio
                    <= admitted.elastic_net_l1_ratio_max
                )
        if not valid:
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return parameters

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Fit one qualified linear recipe and seal learned content on the admitted Feature axis.

        Args:
            recipe: Qualified adapter recipe whose numerical fields are validated.
            inputs: Finite read-only parent training matrix and target values.
            fit_plan: Optional direct plan bound to the same training identity and feature axis.

        Returns:
            Learned coefficients/intercept, model-neutral state and training diagnostics; no outer
            prediction surface is consumed.

        Raises:
            ValueError: Recipe/plan authority, convergence or resulting numerical content is
                invalid.
        """
        if fit_plan is not None and (
            fit_plan.protocol_id != "DIRECT_FIT"
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_REGULARIZED_LINEAR_FIT_PLAN_INVALID")
        parameters = self.validate_recipe(recipe)
        empty: FloatArray = np.empty((0, len(inputs.ordered_feature_ids)), dtype=np.float64)
        empty.setflags(write=False)
        fit = fit_regularized_linear(
            parameters=parameters,
            training_features=inputs.features,
            training_targets=inputs.targets,
            prediction_features=empty,
        )
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=REGULARIZED_LINEAR_CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={
                "family": parameters.family,
                "coefficient_hex": tuple(float(value).hex() for value in fit.coefficients),
                "intercept_hex": float(fit.intercept).hex(),
            },
        )
        state_projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id=parameters.family,
            state_kind="LINEAR",
            state_schema_id=REGULARIZED_LINEAR_STATE_SCHEMA_ID,
            payload={
                "coefficient_hex": tuple(float(value).hex() for value in fit.coefficients),
                "intercept_hex": float(fit.intercept).hex(),
                "coefficient_l2_norm": float(np.linalg.norm(fit.coefficients)),
                "coefficient_max_abs": float(np.max(np.abs(fit.coefficients))),
                "nonzero_support_count": int(np.count_nonzero(fit.coefficients)),
                "model_text_hash": None,
                "best_iteration": None,
                "feature_gain_hex": (),
                "top_feature_gain_share": None,
            },
        )
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=state_projection,
            training_mse=fit.training_mse,
            iteration_count=fit.iteration_count,
        )

    def predict(
        self,
        *,
        estimator: AlphaEstimatorContent,
        inputs: BoundAlphaPredictionInput,
    ) -> AlphaModelPredictionResult:
        """Reconstruct admitted learned content and predict on its exact Feature axis.

        Args:
            estimator: Qualified immutable learned content for this adapter and format.
            inputs: Finite read-only prediction matrix on the fitted feature axis.

        Returns:
            Finite frozen predictions aligned with the input rows, from one prediction call.

        Raises:
            ValueError: Estimator route, content/feature binding or prediction output is invalid.
        """
        if (
            estimator.adapter_id != self.adapter_id
            or estimator.content_format_id != REGULARIZED_LINEAR_CONTENT_FORMAT_ID
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_REGULARIZED_LINEAR_ESTIMATOR_BINDING_INVALID")
        coefficients, intercept = decode_regularized_linear_content(estimator)
        predictions = np.asarray(inputs.features @ coefficients + intercept, dtype=np.float64)
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


__all__ = [
    "REGULARIZED_LINEAR_ADAPTER_ID",
    "REGULARIZED_LINEAR_CONTENT_FORMAT_ID",
    "REGULARIZED_LINEAR_RECIPE_SCHEMA_ID",
    "REGULARIZED_LINEAR_STATE_SCHEMA_ID",
    "RegularizedLinearAdapter",
    "RegularizedLinearFit",
    "RegularizedLinearParameters",
    "RegularizedLinearSearchDomain",
    "build_regularized_linear_recipe",
    "build_regularized_linear_search_domain",
    "decode_regularized_linear_content",
    "fit_regularized_linear",
]
