"""Family-aware guarded current refit and current-formation score construction."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from lightgbm import LGBMRegressor

from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    RegularizedLinearParameters,
    build_regularized_linear_recipe,
    decode_regularized_linear_content,
    fit_regularized_linear,
)
from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalog,
    build_installed_alpha_model_catalog,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaEstimatorContent,
    AlphaFitProvenanceReceipt,
    AlphaModelFitResult,
    AlphaModelRecipeEnvelope,
)
from alphalattice.capabilities.alpha_modeling.runtime.numerical_environment import (
    alpha_model_numerical_scope,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..evaluation.stability import AlphaDevelopmentState, build_estimator_stability_checks
from ..experiments.contracts import AlphaExperimentCard
from ..experiments.development_contracts import (
    AlphaEstimatorState,
    AlphaSessionScoreStatistics,
    seal_current_contract,
)
from ..inputs.folds import AlphaCurrentRefitArrays
from ..inputs.training import bind_alpha_model_inputs
from ..publication.contracts import (
    CurrentRefitStabilityAssessment,
)
from ..publication.contracts import (
    seal_current_contract as seal_publication_contract,
)


@dataclass(frozen=True, slots=True)
class CurrentAlphaRefitResult:
    """Retain current refit state, assessment and optional prediction/provenance outputs.

    Fit/predict counts and the operation identity describe work already performed. Optional
    estimator content and fit provenance retain the corresponding sealed evidence; their absence is
    explicit.
    """

    estimator_state: AlphaEstimatorState
    assessment: CurrentRefitStabilityAssessment
    score_table: pa.Table | None
    model_text: str | None
    fit_calls: int
    predict_calls: int
    fit_operation_hash: str | None = None
    estimator_content: AlphaEstimatorContent | None = None
    fit_provenance: AlphaFitProvenanceReceipt | None = None

    @property
    def estimator_content_hash(self) -> str | None:
        """Read the retained estimator-content identity when content was recorded.

        Returns:
            Estimator content_hash, or None when no content record is present.
        """
        return self.estimator_content.content_hash if self.estimator_content is not None else None

    @property
    def fit_provenance_hash(self) -> str | None:
        """Read the retained fit-provenance identity when provenance was recorded.

        Returns:
            Fit provenance_hash, or None when no provenance record is present.
        """
        return self.fit_provenance.provenance_hash if self.fit_provenance is not None else None


def _score_statistics(
    arrays: AlphaCurrentRefitArrays, scores: npt.NDArray[np.float64]
) -> tuple[AlphaSessionScoreStatistics, ...]:
    finite = scores[np.isfinite(scores)]
    return (
        AlphaSessionScoreStatistics(
            formation_session=arrays.formation_session,
            listing_count=len(scores),
            scored_count=len(finite),
            score_mean=float(np.mean(finite)) if finite.size else 0.0,
            score_std=float(np.std(finite, ddof=0)) if finite.size else 0.0,
            score_coverage=finite.size / len(scores) if len(scores) else 0.0,
        ),
    )


def _linear_state(
    *,
    request_hash: str,
    candidate_id: str,
    family_id: str,
    arrays: AlphaCurrentRefitArrays,
    coefficients: npt.NDArray[np.float64],
    intercept: float,
    training_mse: float,
    scores: npt.NDArray[np.float64],
) -> AlphaEstimatorState:
    statistics = _score_statistics(arrays, scores)
    return seal_current_contract(
        AlphaEstimatorState,
        {
            "kind": "AlphaEstimatorState",
            "request_hash": request_hash,
            "candidate_id": candidate_id,
            "scope": "CURRENT_REFIT",
            "family_id": family_id,
            "state_kind": "LINEAR",
            "fold_index": None,
            "ordered_factor_ids": arrays.ordered_factor_ids,
            "coefficient_hex": tuple(float(value).hex() for value in coefficients),
            "intercept_hex": float(intercept).hex(),
            "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            "coefficient_max_abs": float(np.max(np.abs(coefficients))),
            "nonzero_support_count": int(np.count_nonzero(coefficients)),
            "model_text_hash": None,
            "best_iteration": None,
            "feature_gain_hex": (),
            "top_feature_gain_share": None,
            "training_mse": training_mse,
            "validation_score_mean": statistics[0].score_mean,
            "validation_score_std": statistics[0].score_std,
            "validation_score_coverage": statistics[0].score_coverage,
            "validation_session_statistics": statistics,
        },
        "state_hash",
    )


def _tree_state(
    *,
    request_hash: str,
    card: AlphaExperimentCard,
    arrays: AlphaCurrentRefitArrays,
    model_text: str,
    best_iteration: int,
    gains: npt.NDArray[np.float64],
    training_mse: float,
    scores: npt.NDArray[np.float64],
) -> AlphaEstimatorState:
    total = float(gains.sum())
    top_share = float(gains.max() / total) if total > 0.0 else 0.0
    statistics = _score_statistics(arrays, scores)
    return seal_current_contract(
        AlphaEstimatorState,
        {
            "kind": "AlphaEstimatorState",
            "request_hash": request_hash,
            "candidate_id": card.candidate_id,
            "scope": "CURRENT_REFIT",
            "family_id": "lightgbm",
            "state_kind": "TREE",
            "fold_index": None,
            "ordered_factor_ids": arrays.ordered_factor_ids,
            "coefficient_hex": (),
            "intercept_hex": "0x0.0p+0",
            "coefficient_l2_norm": 0.0,
            "coefficient_max_abs": 0.0,
            "nonzero_support_count": None,
            "model_text_hash": canonical_hash(model_text),
            "best_iteration": best_iteration,
            "feature_gain_hex": tuple(float(value).hex() for value in gains),
            "top_feature_gain_share": top_share,
            "training_mse": training_mse,
            "validation_score_mean": statistics[0].score_mean,
            "validation_score_std": statistics[0].score_std,
            "validation_score_coverage": statistics[0].score_coverage,
            "validation_session_statistics": statistics,
        },
        "state_hash",
    )


def _fit_linear(
    *,
    request_hash: str,
    card: AlphaExperimentCard,
    arrays: AlphaCurrentRefitArrays,
) -> tuple[AlphaEstimatorState, npt.NDArray[np.float64], int, int]:
    mask = arrays.training_mask
    x = arrays.training_features[mask]
    y = arrays.training_targets[mask]
    if card.family_id == "ridge" and card.alpha is not None:
        parameters = RegularizedLinearParameters(
            family="ridge",
            alpha=card.alpha,
        )
    elif card.family_id in {"lasso", "elastic_net"} and card.alpha_multiplier is not None:
        if card.family_id == "lasso":
            parameters = RegularizedLinearParameters(
                family="lasso",
                alpha_max_multiplier=card.alpha_multiplier,
            )
        else:
            if card.l1_ratio is None:
                raise ValueError("ALPHA_CURRENT_REFIT_ELASTIC_RATIO_MISSING")
            parameters = RegularizedLinearParameters(
                family="elastic_net",
                alpha_max_multiplier=card.alpha_multiplier,
                l1_ratio=card.l1_ratio,
            )
    else:
        raise ValueError("ALPHA_CURRENT_REFIT_FAMILY_UNSUPPORTED")
    fit = fit_regularized_linear(
        parameters=parameters,
        training_features=x,
        training_targets=y,
        prediction_features=arrays.current_features[arrays.current_feature_complete],
    )
    scores: npt.NDArray[np.float64] = np.full(
        len(arrays.ordered_listing_ids), np.nan, dtype=np.float64
    )
    if bool(arrays.current_feature_complete.any()):
        scores[arrays.current_feature_complete] = fit.predictions
    state = _linear_state(
        request_hash=request_hash,
        candidate_id=card.candidate_id,
        family_id=card.family_id,
        arrays=arrays,
        coefficients=fit.coefficients,
        intercept=fit.intercept,
        training_mse=fit.training_mse,
        scores=scores,
    )
    return state, scores, 1, 2


def _fit_resolved_regularized_linear(
    *,
    request_hash: str,
    candidate_id: str,
    parameters: RegularizedLinearParameters,
    arrays: AlphaCurrentRefitArrays,
    package_identity_hash: str,
    model_catalog: AlphaModelCatalog,
) -> tuple[
    AlphaEstimatorState,
    npt.NDArray[np.float64],
    int,
    int,
    str,
    AlphaEstimatorContent,
    AlphaFitProvenanceReceipt,
]:
    training_input, prediction_input = bind_alpha_model_inputs(
        binding=arrays.training_input_binding,
        ordered_feature_ids=arrays.ordered_factor_ids,
        training_features=arrays.training_features,
        training_targets=arrays.training_targets,
        training_mask=arrays.training_mask,
        prediction_features=arrays.current_features,
        prediction_mask=arrays.current_feature_complete,
    )
    recipe = build_regularized_linear_recipe(parameters)
    adapter = model_catalog.resolve(recipe)
    fit = adapter.fit(recipe=recipe, inputs=training_input)
    prediction = adapter.predict(estimator=fit.estimator_content, inputs=prediction_input)
    provenance = AlphaFitProvenanceReceipt.create(
        recipe_hash=recipe.recipe_hash,
        training_binding_hash=training_input.training_binding_hash,
        estimator_content_hash=fit.estimator_content.content_hash,
        package_identity_hash=package_identity_hash,
        fit_call_count=fit.fit_call_count,
        predict_call_count=fit.predict_call_count + prediction.predict_call_count,
    )
    fit_operation_hash = canonical_hash(
        {
            "kind": "AlphaCurrentModelFitOperation",
            "request_hash": request_hash,
            "candidate_id": candidate_id,
            "training_binding_hash": training_input.training_binding_hash,
        }
    )
    scores: npt.NDArray[np.float64] = np.full(
        len(arrays.ordered_listing_ids), np.nan, dtype=np.float64
    )
    if bool(arrays.current_feature_complete.any()):
        scores[arrays.current_feature_complete] = prediction.predictions
    coefficients, intercept = decode_regularized_linear_content(fit.estimator_content)
    state = _linear_state(
        request_hash=request_hash,
        candidate_id=candidate_id,
        family_id=parameters.family,
        arrays=arrays,
        coefficients=coefficients,
        intercept=intercept,
        training_mse=fit.training_mse,
        scores=scores,
    )
    return (
        state,
        scores,
        fit.fit_call_count,
        fit.predict_call_count + prediction.predict_call_count,
        fit_operation_hash,
        fit.estimator_content,
        provenance,
    )


def _fit_tree(
    *,
    request_hash: str,
    card: AlphaExperimentCard,
    arrays: AlphaCurrentRefitArrays,
    development_states: tuple[AlphaDevelopmentState, ...],
) -> tuple[AlphaEstimatorState, npt.NDArray[np.float64], str, int, int]:
    if card.num_leaves is None or card.learning_rate is None:
        raise ValueError("ALPHA_CURRENT_REFIT_LIGHTGBM_PARAMETER_MISSING")
    iterations = tuple(
        value.best_iteration for value in development_states if value.best_iteration is not None
    )
    if len(iterations) != len(development_states):
        raise ValueError("ALPHA_CURRENT_REFIT_LIGHTGBM_STATE_MISMATCH")
    best_iteration = int(median(iterations))
    common = dict(
        objective="l2",
        metric="l2",
        num_leaves=card.num_leaves,
        learning_rate=card.learning_rate,
        max_depth=5,
        n_estimators=best_iteration,
        min_child_samples=20,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=1729,
        bagging_seed=1729,
        feature_fraction_seed=1729,
        data_random_seed=1729,
        drop_seed=1729,
        extra_seed=1729,
        verbosity=-1,
    )
    mask = arrays.training_mask
    estimator = LGBMRegressor(**common)
    estimator.fit(arrays.training_features[mask], arrays.training_targets[mask])
    training_prediction = np.asarray(
        estimator.predict(arrays.training_features[mask]), dtype=np.float64
    )
    scores: npt.NDArray[np.float64] = np.full(
        len(arrays.ordered_listing_ids), np.nan, dtype=np.float64
    )
    if bool(arrays.current_feature_complete.any()):
        scores[arrays.current_feature_complete] = estimator.predict(
            arrays.current_features[arrays.current_feature_complete]
        )
    gains = np.asarray(
        estimator.booster_.feature_importance(importance_type="gain"), dtype=np.float64
    )
    model_text = str(estimator.booster_.model_to_string(num_iteration=best_iteration))
    state = _tree_state(
        request_hash=request_hash,
        card=card,
        arrays=arrays,
        model_text=model_text,
        best_iteration=best_iteration,
        gains=gains,
        training_mse=float(np.mean(np.square(arrays.training_targets[mask] - training_prediction))),
        scores=scores,
    )
    return state, scores, model_text, 1, 2


def fit_and_assess_current_alpha(
    *,
    request_hash: str,
    card: AlphaExperimentCard,
    arrays: AlphaCurrentRefitArrays,
    development_states: tuple[AlphaDevelopmentState, ...],
) -> CurrentAlphaRefitResult:
    """Refit one development-admissible registered model and enforce its fold envelope."""
    if not development_states or any(
        (isinstance(value, AlphaEstimatorState) and value.scope != "DEVELOPMENT_FOLD")
        or value.candidate_id != card.candidate_id
        or value.ordered_factor_ids != arrays.ordered_factor_ids
        for value in development_states
    ):
        raise ValueError("ALPHA_CURRENT_REFIT_DEVELOPMENT_STATE_MISMATCH")
    if int(arrays.training_mask.sum()) <= len(arrays.ordered_factor_ids):
        raise ValueError("ALPHA_CURRENT_REFIT_TRAINING_SURFACE_INSUFFICIENT")
    if card.family_id == "lightgbm":
        current, scores, model_text, fit_calls, predict_calls = _fit_tree(
            request_hash=request_hash,
            card=card,
            arrays=arrays,
            development_states=development_states,
        )
    else:
        current, scores, fit_calls, predict_calls = _fit_linear(
            request_hash=request_hash,
            card=card,
            arrays=arrays,
        )
        model_text = None
    checks = list(build_estimator_stability_checks(current, development_states))
    passed = all(value.passed for value in checks)
    assessment = seal_publication_contract(
        CurrentRefitStabilityAssessment,
        {
            "kind": "CurrentRefitStabilityAssessment",
            "request_hash": request_hash,
            "selected_candidate_id": card.candidate_id,
            "development_state_hashes": tuple(value.state_hash for value in development_states),
            "current_state_hash": current.state_hash,
            "checks": tuple(checks),
            "passed": passed,
            "failure_code": None if passed else "alpha_research.current_refit_insufficient",
        },
        "assessment_hash",
    )
    table = None
    if passed:
        table = _current_score_table(arrays=arrays, scores=scores)
    return CurrentAlphaRefitResult(
        estimator_state=current,
        assessment=assessment,
        score_table=table,
        model_text=model_text,
        fit_calls=fit_calls,
        predict_calls=predict_calls,
    )


def fit_and_assess_current_regularized_linear(
    *,
    request_hash: str,
    candidate_id: str,
    parameters: RegularizedLinearParameters,
    arrays: AlphaCurrentRefitArrays,
    development_states: tuple[AlphaDevelopmentState, ...],
    package_identity_hash: str,
    model_catalog: AlphaModelCatalog | None = None,
    retain_diagnostic_score: bool = False,
) -> CurrentAlphaRefitResult:
    """Refit one governed Agent spec under the unchanged current-stability policy."""
    if not development_states or any(
        (isinstance(value, AlphaEstimatorState) and value.scope != "DEVELOPMENT_FOLD")
        or value.candidate_id != candidate_id
        or value.ordered_factor_ids != arrays.ordered_factor_ids
        or value.family_id != parameters.family
        for value in development_states
    ):
        raise ValueError("ALPHA_CURRENT_REFIT_DEVELOPMENT_STATE_MISMATCH")
    if int(arrays.training_mask.sum()) <= len(arrays.ordered_factor_ids):
        raise ValueError("ALPHA_CURRENT_REFIT_TRAINING_SURFACE_INSUFFICIENT")
    current, scores, fit_calls, predict_calls, operation_hash, content, provenance = (
        _fit_resolved_regularized_linear(
            request_hash=request_hash,
            candidate_id=candidate_id,
            parameters=parameters,
            arrays=arrays,
            package_identity_hash=package_identity_hash,
            model_catalog=model_catalog or build_installed_alpha_model_catalog(),
        )
    )
    checks = tuple(build_estimator_stability_checks(current, development_states))
    passed = all(value.passed for value in checks)
    assessment = seal_publication_contract(
        CurrentRefitStabilityAssessment,
        {
            "kind": "CurrentRefitStabilityAssessment",
            "request_hash": request_hash,
            "selected_candidate_id": candidate_id,
            "development_state_hashes": tuple(value.state_hash for value in development_states),
            "current_state_hash": current.state_hash,
            "checks": checks,
            "passed": passed,
            "failure_code": None if passed else "alpha_research.current_refit_insufficient",
        },
        "assessment_hash",
    )
    table = None
    if passed or retain_diagnostic_score:
        table = _current_score_table(arrays=arrays, scores=scores)
    return CurrentAlphaRefitResult(
        estimator_state=current,
        assessment=assessment,
        score_table=table,
        model_text=None,
        fit_calls=fit_calls,
        predict_calls=predict_calls,
        fit_operation_hash=operation_hash,
        estimator_content=content,
        fit_provenance=provenance,
    )


def _projected_state(
    *,
    request_hash: str,
    candidate_id: str,
    arrays: AlphaCurrentRefitArrays,
    fit: AlphaModelFitResult,
    numerical_binding_hash: str,
    fit_evidence_hash: str,
    scores: npt.NDArray[np.float64],
) -> AlphaEstimatorState:
    """The current state of a model bound to its adapter's projection, of the kind it projects."""

    projection = fit.state_projection
    payload = projection.payload
    linear = projection.state_kind == "LINEAR"
    statistics = _score_statistics(arrays, scores)
    return seal_current_contract(
        AlphaEstimatorState,
        {
            "kind": "AlphaEstimatorState",
            "request_hash": request_hash,
            "candidate_id": candidate_id,
            "scope": "CURRENT_REFIT",
            "family_id": projection.model_family_id,
            "state_kind": projection.state_kind,
            "adapter_id": projection.adapter_id,
            "numerical_binding_hash": numerical_binding_hash,
            "fit_evidence_hash": fit_evidence_hash,
            "state_projection_hash": projection.projection_hash,
            "state_schema_id": projection.state_schema_id,
            "projection_payload": payload,
            "fold_index": None,
            "ordered_factor_ids": arrays.ordered_factor_ids,
            "coefficient_hex": tuple(payload["coefficient_hex"]) if linear else (),
            "intercept_hex": payload["intercept_hex"] if linear else "0x0.0p+0",
            "coefficient_l2_norm": payload["coefficient_l2_norm"] if linear else 0.0,
            "coefficient_max_abs": payload["coefficient_max_abs"] if linear else 0.0,
            "nonzero_support_count": payload["nonzero_support_count"] if linear else None,
            "model_text_hash": None,
            "best_iteration": None,
            "feature_gain_hex": (),
            "top_feature_gain_share": None,
            "training_mse": fit.training_mse,
            "validation_score_mean": statistics[0].score_mean,
            "validation_score_std": statistics[0].score_std,
            "validation_score_coverage": statistics[0].score_coverage,
            "validation_session_statistics": statistics,
        },
        "state_hash",
    )


def fit_and_assess_current_model(
    *,
    request_hash: str,
    candidate_id: str,
    recipe: AlphaModelRecipeEnvelope,
    arrays: AlphaCurrentRefitArrays,
    development_states: tuple[AlphaDevelopmentState, ...],
    package_identity_hash: str,
    model_catalog: AlphaModelCatalog,
    retain_diagnostic_score: bool = False,
) -> CurrentAlphaRefitResult:
    """Refit a model bound to its adapter's projection under the unchanged stability policy.

    The current refit of an agent's model, of a kind the installed families do not seal (V342).
    It fits and predicts in its adapter's numerical scope (an extension at one thread); its state
    is the kind its adapter projects, checked against its development folds' states by the
    policy's statistics for that kind. A tree an agent projects is refused by name, since the
    policy's tree statistics read the installed family's diagnostics.

    Args:
        request_hash: The qualification Program's identity.
        candidate_id: The candidate refitted.
        recipe: The candidate's admitted model recipe.
        arrays: The current window's training and prediction arrays.
        development_states: The candidate's development fold states.
        package_identity_hash: The owner package identity its provenance records.
        model_catalog: The catalog holding the model, a workspace's activations included.
        retain_diagnostic_score: Whether a failed assessment keeps its score table.

    Returns:
        The refit's state, assessment, scores and provenance.

    Raises:
        ValueError: The development states do not match the candidate, the training surface is
            too small, or the model projects a tree.
    """
    if not development_states or any(
        (isinstance(value, AlphaEstimatorState) and value.scope != "DEVELOPMENT_FOLD")
        or value.candidate_id != candidate_id
        or value.ordered_factor_ids != arrays.ordered_factor_ids
        for value in development_states
    ):
        raise ValueError("ALPHA_CURRENT_REFIT_DEVELOPMENT_STATE_MISMATCH")
    if int(arrays.training_mask.sum()) <= len(arrays.ordered_factor_ids):
        raise ValueError("ALPHA_CURRENT_REFIT_TRAINING_SURFACE_INSUFFICIENT")
    training_input, prediction_input = bind_alpha_model_inputs(
        binding=arrays.training_input_binding,
        ordered_feature_ids=arrays.ordered_factor_ids,
        training_features=arrays.training_features,
        training_targets=arrays.training_targets,
        training_mask=arrays.training_mask,
        prediction_features=arrays.current_features,
        prediction_mask=arrays.current_feature_complete,
    )
    adapter = model_catalog.resolve(recipe)
    numerical_binding = adapter.describe_numerical_binding()
    with alpha_model_numerical_scope(numerical_binding):
        fit = adapter.fit(recipe=recipe, inputs=training_input)
        prediction = adapter.predict(estimator=fit.estimator_content, inputs=prediction_input)
    projection = fit.state_projection
    if projection.state_kind == "TREE":
        raise ValueError("ALPHA_CURRENT_REFIT_AGENT_TREE_UNSUPPORTED")
    if any(
        value.family_id != projection.model_family_id or value.state_kind != projection.state_kind
        for value in development_states
    ):
        raise ValueError("ALPHA_CURRENT_REFIT_DEVELOPMENT_STATE_MISMATCH")
    provenance = AlphaFitProvenanceReceipt.create(
        recipe_hash=recipe.recipe_hash,
        training_binding_hash=training_input.training_binding_hash,
        estimator_content_hash=fit.estimator_content.content_hash,
        package_identity_hash=package_identity_hash,
        fit_call_count=fit.fit_call_count,
        predict_call_count=fit.predict_call_count + prediction.predict_call_count,
    )
    fit_operation_hash = canonical_hash(
        {
            "kind": "AlphaCurrentModelFitOperation",
            "request_hash": request_hash,
            "candidate_id": candidate_id,
            "training_binding_hash": training_input.training_binding_hash,
        }
    )
    scores: npt.NDArray[np.float64] = np.full(
        len(arrays.ordered_listing_ids), np.nan, dtype=np.float64
    )
    if bool(arrays.current_feature_complete.any()):
        scores[arrays.current_feature_complete] = prediction.predictions
    current = _projected_state(
        request_hash=request_hash,
        candidate_id=candidate_id,
        arrays=arrays,
        fit=fit,
        numerical_binding_hash=numerical_binding.numerical_binding_hash,
        fit_evidence_hash=provenance.provenance_hash,
        scores=scores,
    )
    checks = tuple(build_estimator_stability_checks(current, development_states))
    passed = all(value.passed for value in checks)
    assessment = seal_publication_contract(
        CurrentRefitStabilityAssessment,
        {
            "kind": "CurrentRefitStabilityAssessment",
            "request_hash": request_hash,
            "selected_candidate_id": candidate_id,
            "development_state_hashes": tuple(value.state_hash for value in development_states),
            "current_state_hash": current.state_hash,
            "checks": checks,
            "passed": passed,
            "failure_code": None if passed else "alpha_research.current_refit_insufficient",
        },
        "assessment_hash",
    )
    table = None
    if passed or retain_diagnostic_score:
        table = _current_score_table(arrays=arrays, scores=scores)
    return CurrentAlphaRefitResult(
        estimator_state=current,
        assessment=assessment,
        score_table=table,
        model_text=None,
        fit_calls=fit.fit_call_count,
        predict_calls=fit.predict_call_count + prediction.predict_call_count,
        fit_operation_hash=fit_operation_hash,
        estimator_content=fit.estimator_content,
        fit_provenance=provenance,
    )


def _current_score_table(
    *, arrays: AlphaCurrentRefitArrays, scores: npt.NDArray[np.float64]
) -> pa.Table:
    return pa.table(
        {
            "formation_session": pa.array(
                [arrays.formation_session] * len(arrays.ordered_listing_ids),
                type=pa.date32(),
            ),
            "listing_id": pa.array(arrays.ordered_listing_ids, type=pa.string()),
            "score": pa.array(
                [float(value) if np.isfinite(value) else None for value in scores],
                type=pa.float64(),
            ),
            "availability": pa.array(
                [
                    "SCORED" if bool(value) else "FEATURE_INCOMPLETE"
                    for value in arrays.current_feature_complete
                ],
                type=pa.string(),
            ),
        }
    )


__all__ = [
    "CurrentAlphaRefitResult",
    "fit_and_assess_current_alpha",
    "fit_and_assess_current_model",
    "fit_and_assess_current_regularized_linear",
]
