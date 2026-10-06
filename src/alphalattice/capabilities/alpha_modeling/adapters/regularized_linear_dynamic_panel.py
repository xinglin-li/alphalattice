"""High-dimensional Ridge and Elastic Net successor for Dynamic Panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, cast

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import ElasticNet

from alphalattice.capabilities.alpha_modeling.contracts import (
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
from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
    SINGLE_THREAD_RUNTIME_CAPABILITY,
)

type FloatArray = npt.NDArray[np.float64]

DYNAMIC_PANEL_LINEAR_ADAPTER_ID = "dynamic_panel_regularized_linear"
DYNAMIC_PANEL_LINEAR_RECIPE_SCHEMA_ID = "alpha-model.dynamic-panel-regularized-linear"
DYNAMIC_PANEL_LINEAR_SEARCH_DOMAIN_SCHEMA_ID = (
    "alpha-model.dynamic-panel-regularized-linear.search-domain"
)
DYNAMIC_PANEL_LINEAR_CONTENT_FORMAT_ID = "alpha-model.dynamic-panel-regularized-linear.coefficients"
DYNAMIC_PANEL_LINEAR_STATE_SCHEMA_ID = (
    "alpha-model.dynamic-panel-regularized-linear.development-state"
)
DYNAMIC_PANEL_LINEAR_MAX_ITERATIONS = 50_000
DYNAMIC_PANEL_LINEAR_TOLERANCE = 1e-6
_RIDGE_CROSS_PRODUCT_BLOCK_ROWS: Final = 32_768


@dataclass(frozen=True, slots=True)
class DynamicPanelRegularizedLinearParameters:
    """Declare Ridge or Elastic-Net penalties on the admitted Dynamic Panel matrix.

    Ridge uses a positive fixed alpha; Elastic-Net uses a positive training alpha-max
    multiplier and an interior L1 ratio. Private per-column scaling is not part of
    this adapter's declared numerical method.
    """

    family: Literal["ridge", "elastic_net"]
    alpha: float | None = None
    alpha_max_multiplier: float | None = None
    l1_ratio: float | None = None

    def __post_init__(self) -> None:
        """Validate finite Ridge or Elastic-Net fields without private matrix scaling.

        Raises:
            ValueError: Numerical fields are nonfinite or the family-specific positive penalty and
                mixing fields disagree.
        """
        values = tuple(
            value
            for value in (self.alpha, self.alpha_max_multiplier, self.l1_ratio)
            if value is not None
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_PARAMETER_NOT_FINITE")
        if self.family == "ridge":
            if (
                self.alpha is None
                or self.alpha <= 0.0
                or self.alpha_max_multiplier is not None
                or self.l1_ratio is not None
            ):
                raise ValueError("ALPHA_DYNAMIC_PANEL_RIDGE_PARAMETER_INVALID")
            return
        if (
            self.alpha is not None
            or self.alpha_max_multiplier is None
            or self.alpha_max_multiplier <= 0.0
            or self.l1_ratio is None
            or not 0.0 < self.l1_ratio < 1.0
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_ELASTIC_PARAMETER_INVALID")


def build_dynamic_panel_regularized_linear_recipe(
    parameters: DynamicPanelRegularizedLinearParameters,
) -> AlphaModelRecipeEnvelope:
    """Seal one Dynamic Panel linear recipe without research or target authority.

    Args:
        parameters: Validated registered numerical fields for this adapter.

    Returns:
        Host recipe envelope bound to this adapter, its schema and the supplied parameters.
    """
    return AlphaModelRecipeEnvelope.create(
        adapter_id=DYNAMIC_PANEL_LINEAR_ADAPTER_ID,
        recipe_schema_id=DYNAMIC_PANEL_LINEAR_RECIPE_SCHEMA_ID,
        parameters={
            "family": parameters.family,
            "alpha": parameters.alpha,
            "alpha_max_multiplier": parameters.alpha_max_multiplier,
            "l1_ratio": parameters.l1_ratio,
        },
    )


def build_dynamic_panel_regularized_linear_search_domain() -> AlphaModelSearchDomainEnvelope:
    """Seal the installed Dynamic Panel linear search constraints.

    Returns:
        Qualified domain envelope preserving registered parameter bounds/profiles and numerical
        policy.
    """
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=DYNAMIC_PANEL_LINEAR_ADAPTER_ID,
        recipe_schema_id=DYNAMIC_PANEL_LINEAR_RECIPE_SCHEMA_ID,
        search_domain_schema_id=DYNAMIC_PANEL_LINEAR_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={
            "allowed_families": ["ridge", "elastic_net"],
            "ridge_alpha_min": 0.1,
            "ridge_alpha_max": 100.0,
            "sparse_alpha_max_multiplier_min": 0.01,
            "sparse_alpha_max_multiplier_max": 1.0,
            "elastic_net_l1_ratio_min": 0.25,
            "elastic_net_l1_ratio_max": 0.75,
            "maximum_iterations": DYNAMIC_PANEL_LINEAR_MAX_ITERATIONS,
            "tolerance": DYNAMIC_PANEL_LINEAR_TOLERANCE,
        },
    )


def _ridge_gram_spectral(
    features: FloatArray, targets: FloatArray, alphas: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Ridge numerical successor for every alpha from one Gram eigendecomposition.

    The solution uses NumPy centered cross-products and ``np.linalg.eigh`` on
    ``X'X``. It is mathematically equivalent to the predecessor SVD route in
    exact arithmetic but has a distinct floating-point reduction order and is
    therefore identity-bound as a numerical successor, never as bitwise reuse.

    The decomposition is required rather than optional: this design is exactly
    rank deficient by construction, because ``delta1`` is ``current - lag1``, so
    ``X'X`` is singular and a Cholesky factorization relies entirely on the
    penalty to condition it. The spectral form handles the deficiency the way
    an SVD does. Taking it on the Gram is what makes it affordable when
    ``n >> p``, and it costs nothing per additional alpha, which is what the
    batch route needs.

    The Gram and cross-product are formed in bounded row blocks.  This avoids a
    panel-sized centred copy without using ``X'X - n*mean(X)'mean(X)``, whose
    cancellation is unacceptable for large-offset or ill-conditioned inputs.
    """
    rows = features.shape[0]
    feature_mean = features.mean(axis=0)
    target_mean = float(targets.mean())
    gram = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    cross = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, rows, _RIDGE_CROSS_PRODUCT_BLOCK_ROWS):
        stop = min(rows, start + _RIDGE_CROSS_PRODUCT_BLOCK_ROWS)
        centered_features = features[start:stop] - feature_mean
        centered_targets = targets[start:stop] - target_mean
        gram += centered_features.T @ centered_features
        cross += centered_features.T @ centered_targets
    eigenvalues, vectors = np.linalg.eigh(gram)
    # Roundoff can place a mathematically non-negative Gram eigenvalue below
    # zero. The installed successor clips only those negative numerical values.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = vectors.T @ cross
    coefficients = np.empty((alphas.shape[0], features.shape[1]), dtype=np.float64)
    intercepts = np.empty(alphas.shape[0], dtype=np.float64)
    for position in range(alphas.shape[0]):
        weights = vectors @ (projected / (eigenvalues + float(alphas[position])))
        coefficients[position] = weights
        intercepts[position] = target_mean - float(feature_mean @ weights)
    return coefficients, intercepts


def _alpha_max(features: FloatArray, targets: FloatArray) -> float:
    centered = targets - float(np.mean(targets))
    value = float(np.max(np.abs(features.T @ centered)) / features.shape[0])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_ALPHA_MAX_INVALID")
    return value


def _decode(estimator: AlphaEstimatorContent) -> tuple[FloatArray, float]:
    if (
        estimator.adapter_id != DYNAMIC_PANEL_LINEAR_ADAPTER_ID
        or estimator.content_format_id != DYNAMIC_PANEL_LINEAR_CONTENT_FORMAT_ID
    ):
        raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_CONTENT_ROUTE_INVALID")
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
        raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_CONTENT_INVALID")
    coefficients.setflags(write=False)
    return coefficients, intercept


class DynamicPanelRegularizedLinearAdapter:
    """Fit qualified Ridge or Elastic-Net recipes directly on the admitted Dynamic Panel matrix.

    Ridge uses the declared centered-Gram spectral solver; Elastic-Net uses its
    registered cyclic solver. Learned linear content preserves the admitted Feature axis.
    """

    adapter_id = DYNAMIC_PANEL_LINEAR_ADAPTER_ID
    recipe_schema_id = DYNAMIC_PANEL_LINEAR_RECIPE_SCHEMA_ID
    search_domain_schema_id = DYNAMIC_PANEL_LINEAR_SEARCH_DOMAIN_SCHEMA_ID

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Seal declared direct admitted-matrix linear penalties and runtime requirements.

        Returns:
            Adapter/content-format handles, implementation owners, deterministic policy and required
            capabilities.
        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=DYNAMIC_PANEL_LINEAR_CONTENT_FORMAT_ID,
            implementation_owners=(
                "numpy.matmul",
                "numpy.linalg.eigh",
                "sklearn.linear_model.ElasticNet",
            ),
            deterministic_policy={
                "fit_intercept": True,
                "matrix_preprocessing": "NONE",
                "penalty_metric": "DECLARED_FOLD_PREPROCESSED_MATRIX",
                "private_column_scaling": "FORBIDDEN",
                "ridge_numerical_disposition": "NUMERICAL_SUCCESSOR",
                "ridge_solver": "NUMPY_CENTERED_GRAM_EIGH",
                "ridge_sample_weight": "NONE",
                "ridge_cross_product": "BOUNDED_BLOCK_CENTERED_XTX",
                "ridge_target_cross_product": "BOUNDED_BLOCK_CENTERED_XTY",
                "ridge_cross_product_block_rows": _RIDGE_CROSS_PRODUCT_BLOCK_ROWS,
                "ridge_negative_eigenvalue_policy": "CLIP_TO_ZERO",
                "ridge_intercept": "MEAN_Y_MINUS_MEAN_X_DOT_COEFFICIENT",
                "ridge_multi_alpha_execution": "ONE_EIGH_REUSED_BY_NAMED_ALPHA_POINTS",
                "elastic_selection": "cyclic",
                "elastic_max_iterations": DYNAMIC_PANEL_LINEAR_MAX_ITERATIONS,
                "tolerance": DYNAMIC_PANEL_LINEAR_TOLERANCE,
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

    def validate_recipe(
        self, recipe: AlphaModelRecipeEnvelope
    ) -> DynamicPanelRegularizedLinearParameters:
        """Validate this adapter's route and registered numerical recipe fields.

        Args:
            recipe: Qualified envelope naming this adapter and recipe schema.

        Returns:
            Parsed family-specific numerical parameters.

        Raises:
            ValueError: Recipe route or registered numerical fields are invalid.
        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_RECIPE_ROUTE_INVALID")
        try:
            return DynamicPanelRegularizedLinearParameters(**recipe.parameters)
        except TypeError as error:
            raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_RECIPE_PARAMETERS_INVALID") from error

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> dict[str, object]:
        """Validate the qualified search-domain route and constraints.

        Args:
            domain: Sealed constraints proposed for this adapter.

        Returns:
            Copied constraints equal to the installed domain.

        Raises:
            ValueError: The domain differs from the installed declaration.
        """
        if domain != build_dynamic_panel_regularized_linear_search_domain():
            raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_DOMAIN_INVALID")
        return dict(domain.constraints)

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> DynamicPanelRegularizedLinearParameters:
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
        if parameters.family not in admitted["allowed_families"]:  # type: ignore[operator]
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        ridge_alpha_min = float(cast(int | float, admitted["ridge_alpha_min"]))
        ridge_alpha_max = float(cast(int | float, admitted["ridge_alpha_max"]))
        sparse_multiplier_min = float(
            cast(int | float, admitted["sparse_alpha_max_multiplier_min"])
        )
        sparse_multiplier_max = float(
            cast(int | float, admitted["sparse_alpha_max_multiplier_max"])
        )
        elastic_l1_min = float(cast(int | float, admitted["elastic_net_l1_ratio_min"]))
        elastic_l1_max = float(cast(int | float, admitted["elastic_net_l1_ratio_max"]))
        if parameters.family == "ridge":
            assert parameters.alpha is not None
            valid = ridge_alpha_min <= parameters.alpha <= ridge_alpha_max
        else:
            assert parameters.alpha_max_multiplier is not None
            assert parameters.l1_ratio is not None
            valid = (
                sparse_multiplier_min <= parameters.alpha_max_multiplier <= sparse_multiplier_max
                and elastic_l1_min <= parameters.l1_ratio <= elastic_l1_max
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
            raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_FIT_PLAN_INVALID")
        parameters = self.validate_recipe(recipe)
        if parameters.family == "ridge":
            assert parameters.alpha is not None
            fitted, offsets = _ridge_gram_spectral(
                inputs.features,
                inputs.targets,
                np.asarray((parameters.alpha,), dtype=np.float64),
            )
            coefficients = fitted[0]
            intercept = float(offsets[0])
            iteration_count: int | None = None
        else:
            assert parameters.alpha_max_multiplier is not None
            assert parameters.l1_ratio is not None
            penalty = _alpha_max(inputs.features, inputs.targets)
            penalty *= parameters.alpha_max_multiplier
            estimator = ElasticNet(
                alpha=penalty / parameters.l1_ratio,
                l1_ratio=parameters.l1_ratio,
                fit_intercept=True,
                precompute=True,
                max_iter=DYNAMIC_PANEL_LINEAR_MAX_ITERATIONS,
                tol=DYNAMIC_PANEL_LINEAR_TOLERANCE,
                selection="cyclic",
                warm_start=False,
            )
            estimator.fit(inputs.features, inputs.targets)
            iteration = getattr(estimator, "n_iter_", None)
            iteration_count = int(iteration) if iteration is not None else None
            if (
                iteration_count is not None
                and iteration_count >= DYNAMIC_PANEL_LINEAR_MAX_ITERATIONS
            ):
                raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_CONVERGENCE_FAILURE")
            coefficients = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
            intercept = float(estimator.intercept_)
        return self._seal_fit_result(
            parameters=parameters,
            inputs=inputs,
            coefficients=coefficients,
            intercept=intercept,
            iteration_count=iteration_count,
        )

    def fit_ridge_batch(
        self,
        *,
        recipes: tuple[AlphaModelRecipeEnvelope, ...],
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput,
    ) -> tuple[AlphaModelFitResult, ...]:
        """Fit identical targets for one or more Ridge alphas from one decomposition.

        The Gram eigendecomposition is formed once and each alpha is a division
        by ``eigenvalues + alpha``, so one alpha costs the same decomposition as
        four. Requiring two encoded an assumption about the caller's grid rather
        than anything this arithmetic needs.
        """
        if (
            not recipes
            or fit_plan.protocol_id != "DIRECT_FIT"
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_RIDGE_BATCH_FIT_PLAN_INVALID")
        parameters = tuple(self.validate_recipe(recipe) for recipe in recipes)
        alphas = tuple(value.alpha for value in parameters)
        if any(value.family != "ridge" or value.alpha is None for value in parameters) or len(
            set(alphas)
        ) != len(alphas):
            raise ValueError("ALPHA_DYNAMIC_PANEL_RIDGE_BATCH_RECIPE_INVALID")
        alpha_array = np.asarray(alphas, dtype=np.float64)
        coefficients, intercepts = _ridge_gram_spectral(
            inputs.features, inputs.targets, alpha_array
        )
        if coefficients.shape != (len(recipes), len(inputs.ordered_feature_ids)) or (
            intercepts.shape != (len(recipes),)
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_RIDGE_BATCH_RESULT_INVALID")
        return tuple(
            self._seal_fit_result(
                parameters=parameters[position],
                inputs=inputs,
                coefficients=coefficients[position],
                intercept=float(intercepts[position]),
                iteration_count=None,
            )
            for position in range(len(recipes))
        )

    def _seal_fit_result(
        self,
        *,
        parameters: DynamicPanelRegularizedLinearParameters,
        inputs: BoundAlphaTrainingInput,
        coefficients: FloatArray,
        intercept: float,
        iteration_count: int | None,
    ) -> AlphaModelFitResult:
        predictions = inputs.features @ coefficients + intercept
        training_mse = float(np.mean(np.square(inputs.targets - predictions)))
        if (
            coefficients.shape != (len(inputs.ordered_feature_ids),)
            or not np.isfinite(coefficients).all()
            or not np.isfinite(intercept)
            or not np.isfinite(training_mse)
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_RESULT_INVALID")
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=DYNAMIC_PANEL_LINEAR_CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={
                "family": parameters.family,
                "coefficient_hex": tuple(float(value).hex() for value in coefficients),
                "intercept_hex": intercept.hex(),
            },
        )
        state = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id=parameters.family,
            state_kind="LINEAR",
            state_schema_id=DYNAMIC_PANEL_LINEAR_STATE_SCHEMA_ID,
            payload={
                "coefficient_hex": tuple(float(value).hex() for value in coefficients),
                "intercept_hex": intercept.hex(),
                "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
                "coefficient_max_abs": float(np.max(np.abs(coefficients))),
                "nonzero_support_count": int(np.count_nonzero(coefficients)),
                "model_text_hash": None,
                "best_iteration": None,
                "feature_gain_hex": (),
                "top_feature_gain_share": None,
            },
        )
        return AlphaModelFitResult(
            estimator_content=content,
            state_projection=state,
            training_mse=training_mse,
            iteration_count=iteration_count,
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
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_DYNAMIC_PANEL_LINEAR_PREDICTION_INVALID")
        coefficients, intercept = _decode(estimator)
        predictions = np.asarray(inputs.features @ coefficients + intercept, dtype=np.float64)
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


__all__ = [
    "DYNAMIC_PANEL_LINEAR_ADAPTER_ID",
    "DYNAMIC_PANEL_LINEAR_MAX_ITERATIONS",
    "DYNAMIC_PANEL_LINEAR_TOLERANCE",
    "DynamicPanelRegularizedLinearAdapter",
    "DynamicPanelRegularizedLinearParameters",
    "build_dynamic_panel_regularized_linear_recipe",
    "build_dynamic_panel_regularized_linear_search_domain",
]
