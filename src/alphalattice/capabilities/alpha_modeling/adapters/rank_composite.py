"""The null control: pick a few Factors by arithmetic and average their ranks.

Every competitor installed so far estimates coefficients, and every comparison
so far has been between two things that estimate coefficients. That leaves the
prior question unasked -- whether estimating them beats not estimating them at
all -- and on the evidence it does not: raw unfitted Factors reach a positive
turnover-net rank IC out of sample where every fitted arm is negative, because
fitting a near-noise target produces a score that churns.

This adapter exists to make that question unavoidable. It selects the strongest
`selected_factor_count` columns by their pooled correlation with the training
target, gives each a weight of plus or minus one over that count, and stops.
Nothing is optimised, nothing is regularised, and no search touches the weights.

The estimator it emits is nonetheless a coefficient vector, so calibration,
prediction, replay and every downstream consumer read it exactly as they read a
Ridge fit. The null control is not a special case in the pipeline; it is a
linear model whose coefficients were decided by counting rather than by fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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

RANK_COMPOSITE_ADAPTER_ID = "rank_composite"
RANK_COMPOSITE_RECIPE_SCHEMA_ID = "alpha-model.rank-composite"
RANK_COMPOSITE_SEARCH_DOMAIN_SCHEMA_ID = "alpha-model.rank-composite.search-domain"
RANK_COMPOSITE_CONTENT_FORMAT_ID = "alpha-model.rank-composite.coefficients"
RANK_COMPOSITE_STATE_SCHEMA_ID = "alpha-model.rank-composite.development-state"


@dataclass(frozen=True, slots=True)
class RankCompositeParameters:
    """How many Factors the composite averages, and nothing else."""

    selected_factor_count: int

    def __post_init__(self) -> None:
        """Require at least one selected Factor.

        Raises:
            ValueError: The requested count is not positive.

        """
        if self.selected_factor_count < 1:
            raise ValueError("ALPHA_RANK_COMPOSITE_SELECTION_COUNT_INVALID")


@dataclass(frozen=True, slots=True)
class RankCompositeSearchDomain:
    """The sorted, unique counts admitted for the null-control selection."""

    allowed_selected_factor_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject empty, repeated, unsorted or nonpositive selection counts.

        Raises:
            ValueError: The count grid violates any of those constraints.

        """
        counts = self.allowed_selected_factor_counts
        if (
            not counts
            or len(set(counts)) != len(counts)
            or counts != tuple(sorted(counts))
            or any(value < 1 for value in counts)
        ):
            raise ValueError("ALPHA_RANK_COMPOSITE_SEARCH_DOMAIN_INVALID")


@dataclass(frozen=True, slots=True)
class RankCompositeFit:
    """Selected feature IDs, signed equal weights and training error."""

    coefficients: FloatArray
    selected_feature_ids: tuple[str, ...]
    training_mse: float


def _pooled_correlations(features: FloatArray, targets: FloatArray) -> FloatArray:
    """Each column's correlation with the target over the whole training fold.

    The Feature surface is standardized within each session and so is the
    Target, so a pooled correlation is the session-size weighted mean of the
    per-session correlations -- the same quantity a daily rank IC reports,
    without the adapter needing a session axis it is not given.
    """
    centered_targets = targets - float(np.mean(targets))
    target_norm = float(np.linalg.norm(centered_targets))
    if target_norm <= 0.0:
        return np.zeros(features.shape[1], dtype=np.float64)
    centered = features - features.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    numerator = centered.T @ centered_targets
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = numerator / (norms * target_norm)
    # A column with no dispersion in this fold carries no ordering, and must not
    # win a place through a division that produced a non-finite value.
    return np.where(np.isfinite(correlations), correlations, 0.0)


def fit_rank_composite(
    *,
    parameters: RankCompositeParameters,
    ordered_feature_ids: tuple[str, ...],
    training_features: FloatArray,
    training_targets: FloatArray,
) -> RankCompositeFit:
    """Select, sign-align and equal-weight Factors without optimization.

    Ties in absolute pooled correlation break by Factor ID, not matrix order.

    Args:
        parameters: Number of Factors to select.
        ordered_feature_ids: Feature IDs aligned with matrix columns.
        training_features: Bound training feature matrix.
        training_targets: Bound training target vector.

    Returns:
        The sealed coefficient vector, selected IDs and training error.

    Raises:
        ValueError: The requested count exceeds the feature axis.

    """
    if parameters.selected_factor_count > len(ordered_feature_ids):
        raise ValueError("ALPHA_RANK_COMPOSITE_SELECTION_EXCEEDS_AXIS")
    correlations = _pooled_correlations(training_features, training_targets)
    # Ties break on the Factor id so the same fold always yields the same
    # composite; ordering by magnitude alone would let column order decide.
    order = sorted(
        range(len(ordered_feature_ids)),
        key=lambda index: (-abs(float(correlations[index])), ordered_feature_ids[index]),
    )
    chosen = tuple(order[: parameters.selected_factor_count])
    coefficients: FloatArray = np.zeros(len(ordered_feature_ids), dtype=np.float64)
    weight = 1.0 / float(parameters.selected_factor_count)
    for index in chosen:
        # A Factor's published sign is a convention, not a claim: a column that
        # correlates negatively is held short rather than discarded.
        coefficients[index] = weight * (-1.0 if correlations[index] < 0.0 else 1.0)
    # No intercept is fitted. The score is a cross-sectional ordering and a
    # location shift cannot move one, so estimating it would add a parameter
    # that provably changes nothing.
    predictions = training_features @ coefficients
    residual = training_targets - predictions
    coefficients.setflags(write=False)
    return RankCompositeFit(
        coefficients=coefficients,
        selected_feature_ids=tuple(ordered_feature_ids[index] for index in chosen),
        training_mse=float(np.mean(np.square(residual))),
    )


def build_rank_composite_recipe(
    parameters: RankCompositeParameters,
) -> AlphaModelRecipeEnvelope:
    """Build a recipe envelope for an admitted Factor selection count.

    Args:
        parameters: Number of Factors the null control selects.

    Returns:
        The routed recipe envelope.

    """
    return AlphaModelRecipeEnvelope.create(
        adapter_id=RANK_COMPOSITE_ADAPTER_ID,
        recipe_schema_id=RANK_COMPOSITE_RECIPE_SCHEMA_ID,
        parameters={"selected_factor_count": parameters.selected_factor_count},
    )


def build_rank_composite_search_domain() -> AlphaModelSearchDomainEnvelope:
    """Build the fixed selection-count search domain.

    Returns:
        The installed grid of allowed Factor counts.

    """
    return AlphaModelSearchDomainEnvelope.create(
        adapter_id=RANK_COMPOSITE_ADAPTER_ID,
        recipe_schema_id=RANK_COMPOSITE_RECIPE_SCHEMA_ID,
        search_domain_schema_id=RANK_COMPOSITE_SEARCH_DOMAIN_SCHEMA_ID,
        constraints={"allowed_selected_factor_counts": [3, 5, 10, 20]},
    )


def decode_rank_composite_content(estimator: AlphaEstimatorContent) -> FloatArray:
    """Validate coefficient content and return its read-only vector.

    Args:
        estimator: Sealed rank-composite coefficient content.

    Returns:
        Coefficients aligned to the estimator's feature IDs.

    Raises:
        ValueError: The route or coefficient vector is invalid.

    """
    if (
        estimator.adapter_id != RANK_COMPOSITE_ADAPTER_ID
        or estimator.content_format_id != RANK_COMPOSITE_CONTENT_FORMAT_ID
    ):
        raise ValueError("ALPHA_RANK_COMPOSITE_ESTIMATOR_BINDING_INVALID")
    coefficients: FloatArray = np.asarray(
        [float.fromhex(value) for value in estimator.payload["coefficient_hex"]],
        dtype=np.float64,
    )
    if (
        coefficients.shape != (len(estimator.ordered_feature_ids),)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError("ALPHA_RANK_COMPOSITE_ESTIMATOR_CONTENT_INVALID")
    coefficients.setflags(write=False)
    return coefficients


class RankCompositeAdapter:
    """The installed null control every fitted competitor has to beat."""

    adapter_id = RANK_COMPOSITE_ADAPTER_ID
    recipe_schema_id = RANK_COMPOSITE_RECIPE_SCHEMA_ID
    search_domain_schema_id = RANK_COMPOSITE_SEARCH_DOMAIN_SCHEMA_ID

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """Declare the single-thread arithmetic behind Factor selection.

        Returns:
            The numerical binding for the null control.

        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=RANK_COMPOSITE_CONTENT_FORMAT_ID,
            implementation_owners=("numpy.linalg",),
            deterministic_policy={
                "fit_intercept": False,
                "selection_rule": "MAXIMUM_ABSOLUTE_POOLED_CORRELATION_THEN_FEATURE_ID",
                "weighting": "EQUAL_SIGN_ALIGNED",
                "estimation": "NONE",
            },
            # The selection reduces a matrix against a vector, which BLAS may
            # split across threads; an adapter that does not pin the count seals
            # a null environment while the ambient one decides the bits.
            required_runtime_capabilities=("numpy", SINGLE_THREAD_RUNTIME_CAPABILITY),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        """Admit direct fits without coefficient estimation.

        Args:
            recipe: Unused; every admitted recipe has the same protocol.

        Returns:
            The sole supported direct-fit protocol.

        """
        del recipe
        return ("DIRECT_FIT",)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> RankCompositeParameters:
        """Check recipe routing and parse the selected Factor count.

        Args:
            recipe: Envelope to validate against this adapter's route.

        Returns:
            The parsed selection count.

        Raises:
            ValueError: The route or selection count is invalid.

        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("ALPHA_RANK_COMPOSITE_RECIPE_ROUTE_INVALID")
        return RankCompositeParameters(**recipe.parameters)

    def validate_search_domain(
        self, domain: AlphaModelSearchDomainEnvelope
    ) -> RankCompositeSearchDomain:
        """Check domain routing and parse its admitted selection counts.

        Args:
            domain: Search domain to validate.

        Returns:
            The sorted, unique count grid.

        Raises:
            ValueError: The route or count grid is invalid.

        """
        if (
            domain.adapter_id != self.adapter_id
            or domain.recipe_schema_id != self.recipe_schema_id
            or domain.search_domain_schema_id != self.search_domain_schema_id
        ):
            raise ValueError("ALPHA_RANK_COMPOSITE_SEARCH_DOMAIN_ROUTE_INVALID")
        constraints = dict(domain.constraints)
        counts = constraints.get("allowed_selected_factor_counts")
        if isinstance(counts, list):
            constraints["allowed_selected_factor_counts"] = tuple(int(value) for value in counts)
        return RankCompositeSearchDomain(**constraints)

    def validate_recipe_for_domain(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        domain: AlphaModelSearchDomainEnvelope,
    ) -> RankCompositeParameters:
        """Require the recipe's Factor count to occur in the admitted domain.

        Args:
            recipe: Recipe to admit.
            domain: Search domain that defines allowed counts.

        Returns:
            The admitted selection count.

        Raises:
            ValueError: Routing, domain or count membership is invalid.

        """
        parameters = self.validate_recipe(recipe)
        admitted = self.validate_search_domain(domain)
        if parameters.selected_factor_count not in admitted.allowed_selected_factor_counts:
            raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return parameters

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Select signed equal weights and seal them as estimator content.

        Args:
            recipe: Admitted selection-count recipe.
            inputs: Bound training matrix and targets.
            fit_plan: Optional direct-fit plan bound to the training input.

        Returns:
            Coefficient content, state projection and training error.

        Raises:
            ValueError: The fit plan or recipe violates its binding.

        """
        if fit_plan is not None and (
            fit_plan.protocol_id != "DIRECT_FIT"
            or fit_plan.parent_training_binding_hash != inputs.training_binding_hash
            or fit_plan.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_RANK_COMPOSITE_FIT_PLAN_INVALID")
        parameters = self.validate_recipe(recipe)
        fit = fit_rank_composite(
            parameters=parameters,
            ordered_feature_ids=inputs.ordered_feature_ids,
            training_features=inputs.features,
            training_targets=inputs.targets,
        )
        content = AlphaEstimatorContent.create(
            adapter_id=self.adapter_id,
            content_format_id=RANK_COMPOSITE_CONTENT_FORMAT_ID,
            ordered_feature_ids=inputs.ordered_feature_ids,
            payload={
                "selected_factor_count": parameters.selected_factor_count,
                "selected_feature_ids": fit.selected_feature_ids,
                "coefficient_hex": tuple(float(value).hex() for value in fit.coefficients),
            },
        )
        state_projection = AlphaModelStateProjection.create(
            adapter_id=self.adapter_id,
            model_family_id="rank_composite",
            state_kind="LINEAR",
            state_schema_id=RANK_COMPOSITE_STATE_SCHEMA_ID,
            payload={
                "coefficient_hex": tuple(float(value).hex() for value in fit.coefficients),
                "intercept_hex": (0.0).hex(),
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
            iteration_count=None,
        )

    def predict(
        self,
        *,
        estimator: AlphaEstimatorContent,
        inputs: BoundAlphaPredictionInput,
    ) -> AlphaModelPredictionResult:
        """Score a bound feature matrix using the sealed coefficient vector.

        Args:
            estimator: Sealed rank-composite coefficients.
            inputs: Bound prediction matrix with the estimator's feature order.

        Returns:
            A read-only prediction vector.

        Raises:
            ValueError: The estimator or feature binding is invalid.

        """
        if (
            estimator.adapter_id != self.adapter_id
            or estimator.content_format_id != RANK_COMPOSITE_CONTENT_FORMAT_ID
            or estimator.ordered_feature_ids != inputs.ordered_feature_ids
        ):
            raise ValueError("ALPHA_RANK_COMPOSITE_ESTIMATOR_BINDING_INVALID")
        coefficients = decode_rank_composite_content(estimator)
        predictions = np.asarray(inputs.features @ coefficients, dtype=np.float64)
        predictions.setflags(write=False)
        return AlphaModelPredictionResult(predictions=predictions)


__all__ = [
    "RANK_COMPOSITE_ADAPTER_ID",
    "RANK_COMPOSITE_CONTENT_FORMAT_ID",
    "RANK_COMPOSITE_RECIPE_SCHEMA_ID",
    "RANK_COMPOSITE_STATE_SCHEMA_ID",
    "RankCompositeAdapter",
    "RankCompositeFit",
    "RankCompositeParameters",
    "RankCompositeSearchDomain",
    "build_rank_composite_recipe",
    "build_rank_composite_search_domain",
    "decode_rank_composite_content",
    "fit_rank_composite",
]
