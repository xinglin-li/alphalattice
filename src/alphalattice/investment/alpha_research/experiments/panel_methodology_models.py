"""Installed model execution for the paired Panel research methodology."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_chronological import (
    chronological_lightgbm_dataset_cache,
)
from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DynamicPanelLightGBMAdapter,
    DynamicPanelLightGBMParameters,
    build_dynamic_panel_lightgbm_recipe,
    build_dynamic_panel_lightgbm_search_domain,
    dynamic_panel_lightgbm_parameters,
)
from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear_dynamic_panel import (
    DynamicPanelRegularizedLinearAdapter,
    DynamicPanelRegularizedLinearParameters,
    build_dynamic_panel_regularized_linear_recipe,
    build_dynamic_panel_regularized_linear_search_domain,
)
from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.capabilities.alpha_modeling.contracts import (
    NESTED_FIT_RUNTIME_CAPABILITY,
    AlphaEstimatorContent,
    AlphaFitProvenanceReceipt,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
    alpha_model_array_content_hash,
)
from alphalattice.capabilities.alpha_modeling.runtime.service import (
    AlphaModelExecutionResult,
    AlphaModelRuntimeService,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_authoring import (
    PanelResearchMethodologyRequest,
    panel_model_recipe_points,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_statistics import (
    CandidateComplexity,
    CausalScoreAggregationSelection,
    LongOnlySessionEvidence,
    PanelModelScientificSelection,
    PanelScientificSessionEvidence,
    panel_scientific_session_evidence,
    select_panel_model_scientifically,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    PanelFeatureProjection,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    INSTALLED_PANEL_VIEW_IDS,
    HistoricalPanelEvidenceViewId,
    InstalledPanelViewId,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]
type ModelFamilyId = Literal["RIDGE", "ELASTIC_NET", "LIGHTGBM"]
_HASH = r"^[0-9a-f]{64}$"
_CONTROL_VIEW_ID: Final[InstalledPanelViewId] = "RELATIVE_CONTROL"
_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
_ELASTIC_MULTIPLIERS = (0.02, 0.05, 0.2, 0.5)
_ELASTIC_L1_RATIOS = (0.35, 0.5, 0.65)


class PanelModelExecutionError(ValueError):
    """Stable refusal for a model boundary or scientific-axis mismatch."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PanelModelTrialEvidence(_Contract):
    kind: str = "PanelModelTrialEvidence"
    program_hash: str = Field(pattern=_HASH)
    fold_index: int = Field(ge=0)
    phase: Literal["INNER_SELECTION", "OUTER_VALIDATION"]
    view_id: InstalledPanelViewId | HistoricalPanelEvidenceViewId
    model_family_id: ModelFamilyId
    recipe_id: str | None = None
    recipe_hash: str = Field(pattern=_HASH)
    span_sessions: int = Field(ge=1)
    feature_axis_hash: str = Field(pattern=_HASH)
    training_binding_hash: str = Field(pattern=_HASH)
    fit_plan_hash: str = Field(pattern=_HASH)
    estimator_content_hash: str = Field(pattern=_HASH)
    state_projection_hash: str = Field(pattern=_HASH)
    provenance_hash: str = Field(pattern=_HASH)
    numerical_environment_hash: str = Field(pattern=_HASH)
    training_mse: float = Field(ge=0.0)
    validation_mse: float = Field(ge=0.0)
    best_iteration: int | None = Field(default=None, ge=1)
    early_stopping_metric_id: str | None
    early_stopping_metric_value: float | None
    feature_gain_concentration: float | None = Field(default=None, ge=0.0, le=1.0)
    scientific_evidence: PanelScientificSessionEvidence | None = None
    long_only_evidence: LongOnlySessionEvidence | None = None
    scale_receipt_hashes: tuple[str, ...] = Field(min_length=1)
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        if values.get("view_id") not in INSTALLED_PANEL_VIEW_IDS:
            raise PanelModelExecutionError("alpha_research.historical_panel_view_not_writable")
        provisional = cls.model_construct(**values, evidence_hash="0" * 64)
        return cls(
            **values,
            evidence_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"evidence_hash"}))
            ),
        )


class PanelModelFoldEvidence(_Contract):
    kind: str = "PanelModelFoldEvidence"
    program_hash: str = Field(pattern=_HASH)
    fold_index: int = Field(ge=0)
    inner_trial_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    group_selection_hashes: tuple[str, ...] = Field(min_length=1)
    deployable_selection_hash: str = Field(pattern=_HASH)
    dynamic_challenger_selection_hash: str = Field(pattern=_HASH)
    score_aggregation_selection_hash: str | None = Field(default=None, pattern=_HASH)
    selection_contract_kind: Literal[
        "CausalModelSpecSelection", "PanelModelScientificSelection"
    ] = "CausalModelSpecSelection"
    outer_trial_evidence_hashes: tuple[str, ...] = Field(min_length=1)
    raw_dynamic_surface_hash: str | None = Field(default=None, pattern=_HASH)
    selected_dynamic_surface_hash: str = Field(pattern=_HASH)
    common_span_control_surface_hash: str = Field(pattern=_HASH)
    fixed_candidate_surface_hashes: tuple[str, ...] = ()
    fit_call_count: int = Field(ge=1)
    predict_call_count: int = Field(ge=1)
    metric_call_count: int = Field(ge=1)
    ridge_factorization_call_count: int | None = Field(default=None, ge=1)
    dataset_reuse_scope_count: int = Field(ge=1)
    fold_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, fold_hash="0" * 64)
        return cls(
            **values,
            fold_hash=str(
                canonical_hash(
                    provisional.model_dump(mode="json", exclude={"fold_hash"}, exclude_none=True)
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelTrialSpec:
    recipe_id: str
    view_id: InstalledPanelViewId
    family_id: ModelFamilyId
    recipe: AlphaModelRecipeEnvelope
    domain: AlphaModelSearchDomainEnvelope

    @property
    def spec_id(self) -> str:
        return f"{self.recipe_id}::{self.view_id}::{self.family_id}::{self.recipe.recipe_hash}"


@dataclass(frozen=True, slots=True)
class PanelScoreSurface:
    surface_id: str
    surface_hash: str
    fold_index: int
    span_sessions: int
    row_sessions: tuple[date, ...]
    row_listing_ids: tuple[str, ...]
    scores: FloatArray
    target_z: FloatArray
    raw_simple_returns: FloatArray
    model_recipe_id: str | None = None

    def __post_init__(self) -> None:
        rows = len(self.row_sessions)
        if (
            rows != len(self.row_listing_ids)
            or any(
                value.shape != (rows,)
                for value in (self.scores, self.target_z, self.raw_simple_returns)
            )
            or any(
                value.flags.writeable
                for value in (self.scores, self.target_z, self.raw_simple_returns)
            )
            or not all(
                np.isfinite(value).all()
                for value in (self.scores, self.target_z, self.raw_simple_returns)
            )
        ):
            raise PanelModelExecutionError("alpha_research.panel_score_surface_invalid")


@dataclass(frozen=True, slots=True)
class ModelContentBundle:
    evidence_hash: str
    estimator: AlphaEstimatorContent
    execution: AlphaModelExecutionResult


@dataclass(frozen=True, slots=True)
class PanelModelFoldResult:
    inner_evidence: tuple[PanelModelTrialEvidence, ...]
    group_selections: tuple[PanelModelScientificSelection, ...]
    deployable_selection: PanelModelScientificSelection
    dynamic_challenger_selection: PanelModelScientificSelection
    score_aggregation_selection: CausalScoreAggregationSelection | None
    outer_evidence: tuple[PanelModelTrialEvidence, ...]
    model_contents: tuple[ModelContentBundle, ...]
    raw_dynamic_surface: PanelScoreSurface
    selected_dynamic_surface: PanelScoreSurface
    common_span_control_surface: PanelScoreSurface
    fixed_candidate_surfaces: tuple[PanelScoreSurface, ...]
    fold_evidence: PanelModelFoldEvidence


def build_panel_model_catalog() -> AlphaModelCatalog:
    return AlphaModelCatalog(
        (DynamicPanelRegularizedLinearAdapter(), DynamicPanelLightGBMAdapter())
    )


def _linear_parameters(
    family: str, point: Mapping[str, Any]
) -> DynamicPanelRegularizedLinearParameters:
    """Build one linear parameter object from a declared or enumerated point."""

    return DynamicPanelRegularizedLinearParameters(
        family="ridge" if family == "RIDGE" else "elastic_net", **point
    )


def installed_model_trials(
    request: PanelResearchMethodologyRequest,
) -> tuple[ModelTrialSpec, ...]:
    """Every trial this request fits, per estimator family and Feature view.

    Two declarations reach here. `model_family_ids` names a family and asks for
    its installed grid; `model_recipes` names configurations and asks for exactly
    those scalar points. The
    second form is what makes a stated configuration a document change rather
    than a source change -- the parameters are passed straight to the family's
    adapter, which validates them against the bounds its domain declares.

    The wider estimators used to be pinned to `DYNAMIC_JOINT_PRIMARY` no matter
    what the request declared, which was invisible while a scope decided the
    admissible families. Once scope and family were decoupled, a `LEAN` request
    naming `LIGHTGBM` would have built trials against a view the request never
    declared and the Feature preflight never materialized. Ridge is the control
    estimator and runs on every declared view; the wider estimators are
    treatments and run on the declared treatment views.
    """

    linear = build_dynamic_panel_regularized_linear_search_domain()
    lightgbm = build_dynamic_panel_lightgbm_search_domain()
    if request.model_recipes:
        exact: list[ModelTrialSpec] = []
        for declaration in request.model_recipes:
            for view_id in declaration.view_ids:
                for point in panel_model_recipe_points(declaration):
                    if declaration.family in {"RIDGE", "ELASTIC_NET"}:
                        recipe = build_dynamic_panel_regularized_linear_recipe(
                            _linear_parameters(declaration.family, point)
                        )
                        domain = linear
                    else:
                        recipe = build_dynamic_panel_lightgbm_recipe(
                            DynamicPanelLightGBMParameters(**cast(Any, point))
                        )
                        domain = lightgbm
                    exact.append(
                        ModelTrialSpec(
                            recipe_id=declaration.recipe_id,
                            view_id=view_id,
                            family_id=declaration.family,
                            recipe=recipe,
                            domain=domain,
                        )
                    )
        return tuple(exact)

    declared = request.model_families
    treatment_view_ids = tuple(
        value for value in request.feature_view_ids if value != _CONTROL_VIEW_ID
    )
    trials: list[ModelTrialSpec] = []
    if "RIDGE" in declared:
        ridge_points = tuple({"alpha": value} for value in _RIDGE_ALPHAS)
        trials.extend(
            ModelTrialSpec(
                recipe_id="HYPERPARAMETER_DISCOVERY",
                view_id=view_id,
                family_id="RIDGE",
                recipe=build_dynamic_panel_regularized_linear_recipe(
                    _linear_parameters("RIDGE", point)
                ),
                domain=linear,
            )
            for view_id in request.feature_view_ids
            for point in ridge_points
        )
    if "ELASTIC_NET" in declared:
        elastic_points = tuple(
            {"alpha_max_multiplier": multiplier, "l1_ratio": ratio}
            for multiplier in _ELASTIC_MULTIPLIERS
            for ratio in _ELASTIC_L1_RATIOS
        )
        trials.extend(
            ModelTrialSpec(
                recipe_id="HYPERPARAMETER_DISCOVERY",
                view_id=view_id,
                family_id="ELASTIC_NET",
                recipe=build_dynamic_panel_regularized_linear_recipe(
                    _linear_parameters("ELASTIC_NET", point)
                ),
                domain=linear,
            )
            for view_id in treatment_view_ids
            for point in elastic_points
        )
    if "LIGHTGBM" in declared:
        lightgbm_points = dynamic_panel_lightgbm_parameters()
        trials.extend(
            ModelTrialSpec(
                recipe_id="HYPERPARAMETER_DISCOVERY",
                view_id=view_id,
                family_id="LIGHTGBM",
                recipe=build_dynamic_panel_lightgbm_recipe(parameters),
                domain=lightgbm,
            )
            for view_id in treatment_view_ids
            for parameters in lightgbm_points
        )
    return tuple(trials)


def _readonly(values: npt.NDArray[Any]) -> FloatArray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _session_codes(sessions: tuple[date, ...]) -> npt.NDArray[np.int64]:
    indexed = {value: position for position, value in enumerate(sorted(set(sessions)))}
    result: IntArray = np.asarray([indexed[value] for value in sessions], dtype=np.int64)
    result.setflags(write=False)
    return result


def _fit_plan(
    *,
    spec: ModelTrialSpec,
    binding_hash: str,
    projection: PanelFeatureProjection,
) -> BoundAlphaModelFitInput:
    binding = build_panel_model_catalog().resolve(spec.recipe).describe_numerical_binding()
    if NESTED_FIT_RUNTIME_CAPABILITY not in binding.required_runtime_capabilities:
        identity: dict[str, object] = {
            "protocol_id": "DIRECT_FIT",
            "parent_training_binding_hash": binding_hash,
            "ordered_feature_ids": list(projection.ordered_feature_ids),
        }
        return BoundAlphaModelFitInput(
            fit_plan_hash=str(canonical_hash(identity)),
            protocol_id="DIRECT_FIT",
            parent_training_binding_hash=binding_hash,
            ordered_feature_ids=projection.ordered_feature_ids,
        )
    policy = str(spec.recipe.parameters["training_policy"])
    selection_metric = {
        "L2_EARLY_STOPPING": "l2",
        "SESSION_RANK_IC_EARLY_STOPPING": "session_rank_ic",
        "FIXED_ITERATION": "fixed_iteration",
    }[policy]
    codes = _session_codes(projection.row_sessions)
    identity = {
        "protocol_id": "NESTED_EARLY_STOPPING_REFIT",
        "parent_training_binding_hash": binding_hash,
        "ordered_feature_ids": list(projection.ordered_feature_ids),
        "tuning_training_row_axis_hash": projection.common_training_row_axis_hash,
        "tuning_validation_row_axis_hash": projection.common_transform_row_axis_hash,
        "tuning_validation_session_axis_hash": canonical_hash([int(value) for value in codes]),
        "selection_metric_id": selection_metric,
        "maximum_iterations": 500,
        "early_stopping_rounds": 50,
    }
    return BoundAlphaModelFitInput(
        fit_plan_hash=str(canonical_hash(identity)),
        protocol_id="NESTED_EARLY_STOPPING_REFIT",
        parent_training_binding_hash=binding_hash,
        ordered_feature_ids=projection.ordered_feature_ids,
        tuning_training_row_axis_hash=projection.common_training_row_axis_hash,
        purge_row_axis_hash=str(canonical_hash([])),
        tuning_validation_row_axis_hash=projection.common_transform_row_axis_hash,
        tuning_training_feature_values_hash=alpha_model_array_content_hash(
            projection.training_features
        ),
        tuning_training_target_values_hash=alpha_model_array_content_hash(
            projection.training_targets
        ),
        tuning_validation_feature_values_hash=alpha_model_array_content_hash(
            projection.transformed_features
        ),
        tuning_validation_target_values_hash=alpha_model_array_content_hash(
            projection.transformed_targets
        ),
        tuning_training_features=projection.training_features,
        tuning_training_targets=projection.training_targets,
        tuning_validation_features=projection.transformed_features,
        tuning_validation_targets=projection.transformed_targets,
        tuning_validation_session_codes=(codes if selection_metric == "session_rank_ic" else None),
        tuning_validation_session_axis_hash=(
            str(canonical_hash([int(value) for value in codes]))
            if selection_metric == "session_rank_ic"
            else None
        ),
        selection_metric_id=selection_metric,
        maximum_iterations=500,
        early_stopping_rounds=50,
    )


def _fit(
    *,
    runtime: AlphaModelRuntimeService,
    spec: ModelTrialSpec,
    program_hash: str,
    fold_index: int,
    phase: str,
    training_projection: PanelFeatureProjection,
    prediction_projection: PanelFeatureProjection,
    tuning_projection: PanelFeatureProjection,
    package_identity_hash: str,
) -> tuple[AlphaModelExecutionResult, str]:
    if (
        training_projection.ordered_feature_ids != prediction_projection.ordered_feature_ids
        or training_projection.ordered_feature_ids != tuning_projection.ordered_feature_ids
    ):
        raise PanelModelExecutionError("alpha_research.panel_model_feature_axis_mismatch")
    binding_hash = _training_binding_hash(
        program_hash=program_hash,
        fold_index=fold_index,
        phase=phase,
        view_id=spec.view_id,
        training_projection=training_projection,
    )
    training = BoundAlphaTrainingInput(
        training_binding_hash=binding_hash,
        ordered_feature_ids=training_projection.ordered_feature_ids,
        features=training_projection.training_features,
        targets=training_projection.training_targets,
    )
    prediction = BoundAlphaPredictionInput(
        training_binding_hash=binding_hash,
        ordered_feature_ids=prediction_projection.ordered_feature_ids,
        features=prediction_projection.transformed_features,
    )
    return (
        runtime.execute(
            recipe=spec.recipe,
            domain=spec.domain,
            fit_plan=_fit_plan(
                spec=spec,
                binding_hash=binding_hash,
                projection=tuning_projection,
            ),
            training_input=training,
            prediction_input=prediction,
            package_identity_hash=package_identity_hash,
        ),
        binding_hash,
    )


def _training_binding_hash(
    *,
    program_hash: str,
    fold_index: int,
    phase: str,
    view_id: InstalledPanelViewId,
    training_projection: PanelFeatureProjection,
) -> str:
    """Identify immutable training values independently of candidate recipes."""

    return str(
        canonical_hash(
            {
                "program_hash": program_hash,
                "fold_index": fold_index,
                "phase": phase,
                "view_id": view_id,
                "training_row_axis_hash": (training_projection.common_training_row_axis_hash),
                "ordered_feature_ids": list(training_projection.ordered_feature_ids),
                "training_feature_hash": alpha_model_array_content_hash(
                    training_projection.training_features
                ),
                "training_target_hash": alpha_model_array_content_hash(
                    training_projection.training_targets
                ),
            }
        )
    )


def _fit_ridge_batch(
    *,
    runtime: AlphaModelRuntimeService,
    specs: tuple[ModelTrialSpec, ...],
    program_hash: str,
    fold_index: int,
    phase: str,
    projection: PanelFeatureProjection,
    package_identity_hash: str,
) -> tuple[tuple[ModelTrialSpec, AlphaModelExecutionResult, str], ...]:
    """Run one or more Ridge recipes over one matrix through one decomposition.

    A single recipe is admitted. The lower bound of two encoded "batching one is
    pointless", which held while the alpha grid was always four; a request that
    states one configuration rather than searching for one now supplies exactly
    one spec per view, and refusing it sent a legitimate run into a refusal about
    batch shape. The decomposition is formed once either way, so the single-spec
    path is the same arithmetic with one alpha rather than a different path.
    """

    if not specs or any(
        spec.family_id != "RIDGE" or spec.view_id != specs[0].view_id for spec in specs
    ):
        raise PanelModelExecutionError("alpha_research.ridge_batch_spec_invalid")
    binding_hash = _training_binding_hash(
        program_hash=program_hash,
        fold_index=fold_index,
        phase=phase,
        view_id=specs[0].view_id,
        training_projection=projection,
    )
    training = BoundAlphaTrainingInput(
        training_binding_hash=binding_hash,
        ordered_feature_ids=projection.ordered_feature_ids,
        features=projection.training_features,
        targets=projection.training_targets,
    )
    prediction_input = BoundAlphaPredictionInput(
        training_binding_hash=binding_hash,
        ordered_feature_ids=projection.ordered_feature_ids,
        features=projection.transformed_features,
    )
    fit_plan = _fit_plan(spec=specs[0], binding_hash=binding_hash, projection=projection)
    fitted = runtime.fit_ridge_batch(
        recipes=tuple(spec.recipe for spec in specs),
        domain=specs[0].domain,
        fit_plan=fit_plan,
        training_input=training,
        package_identity_hash=package_identity_hash,
    )
    results: list[tuple[ModelTrialSpec, AlphaModelExecutionResult, str]] = []
    for spec, fit in zip(specs, fitted, strict=True):
        predicted = runtime.predict(
            recipe=spec.recipe,
            domain=spec.domain,
            estimator=fit.fit.estimator_content,
            prediction_input=prediction_input,
        )
        if (
            predicted.numerical_environment.environment_hash
            != fit.numerical_environment.environment_hash
        ):
            raise PanelModelExecutionError("alpha_research.ridge_batch_environment_changed")
        provenance = AlphaFitProvenanceReceipt.create(
            recipe_hash=spec.recipe.recipe_hash,
            training_binding_hash=binding_hash,
            estimator_content_hash=fit.fit.estimator_content.content_hash,
            package_identity_hash=package_identity_hash,
            fit_plan_hash=fit_plan.fit_plan_hash,
            numerical_environment_hash=fit.numerical_environment.environment_hash,
            fit_call_count=fit.fit.fit_call_count,
            predict_call_count=(
                fit.fit.predict_call_count + predicted.prediction.predict_call_count
            ),
        )
        results.append(
            (
                spec,
                AlphaModelExecutionResult(
                    fit=fit.fit,
                    prediction=predicted.prediction,
                    provenance=provenance,
                    numerical_binding=fit.numerical_binding,
                    numerical_environment=fit.numerical_environment,
                    fit_plan_hash=fit.fit_plan_hash,
                ),
                binding_hash,
            )
        )
    return tuple(results)


def _gain_concentration(estimator: AlphaEstimatorContent) -> float | None:
    raw = estimator.payload.get("feature_gain_hex")
    if not isinstance(raw, (tuple, list)) or not raw:
        return None
    values: FloatArray = np.asarray([float.fromhex(str(value)) for value in raw], dtype=np.float64)
    total = float(np.sum(values))
    return float(np.max(values) / total) if total > 0.0 else 0.0


def _trial_evidence(
    *,
    program_hash: str,
    fold_index: int,
    phase: Literal["INNER_SELECTION", "OUTER_VALIDATION"],
    spec: ModelTrialSpec,
    projection: PanelFeatureProjection,
    result: AlphaModelExecutionResult,
    binding_hash: str,
    scientific: PanelScientificSessionEvidence,
) -> PanelModelTrialEvidence:
    predictions = result.prediction.predictions
    diagnostic = result.fit.selection_diagnostic
    return PanelModelTrialEvidence.create(
        program_hash=program_hash,
        fold_index=fold_index,
        phase=phase,
        view_id=spec.view_id,
        model_family_id=spec.family_id,
        recipe_id=spec.recipe_id,
        recipe_hash=spec.recipe.recipe_hash,
        span_sessions=1,
        feature_axis_hash=str(canonical_hash(list(projection.ordered_feature_ids))),
        training_binding_hash=binding_hash,
        fit_plan_hash=result.fit_plan_hash,
        estimator_content_hash=result.fit.estimator_content.content_hash,
        state_projection_hash=result.fit.state_projection.projection_hash,
        provenance_hash=result.provenance.provenance_hash,
        numerical_environment_hash=result.numerical_environment.environment_hash,
        training_mse=result.fit.measured_training_mse(),
        validation_mse=float(np.mean(np.square(projection.transformed_targets - predictions))),
        best_iteration=result.fit.iteration_count,
        early_stopping_metric_id=diagnostic.metric_id if diagnostic is not None else None,
        early_stopping_metric_value=diagnostic.value if diagnostic is not None else None,
        feature_gain_concentration=_gain_concentration(result.fit.estimator_content),
        scientific_evidence=scientific,
        long_only_evidence=None,
        scale_receipt_hashes=tuple(value.receipt_hash for value in projection.scale_receipts),
    )


def _surface(
    *,
    surface_id: str,
    model_recipe_id: str,
    fold_index: int,
    projection: PanelFeatureProjection,
    predictions: FloatArray,
    common_evaluation_sessions: tuple[date, ...],
) -> PanelScoreSurface:
    admitted_sessions = set(common_evaluation_sessions)
    mask: BoolArray = np.fromiter(
        (value in admitted_sessions for value in projection.row_sessions),
        dtype=np.bool_,
        count=len(projection.row_sessions),
    )
    values = _readonly(predictions[mask])
    if not np.isfinite(values).all():
        raise PanelModelExecutionError("alpha_research.panel_score_surface_warmup_invalid")
    sessions = tuple(
        value for value, include in zip(projection.row_sessions, mask, strict=True) if include
    )
    listings = tuple(
        value for value, include in zip(projection.row_listing_ids, mask, strict=True) if include
    )
    target = _readonly(projection.transformed_targets[mask])
    raw = _readonly(projection.transformed_raw_simple_returns[mask])
    identity = str(
        canonical_hash(
            {
                "surface_id": surface_id,
                "model_recipe_id": model_recipe_id,
                "fold_index": fold_index,
                "span": 1,
                "row_axis": [
                    (session.isoformat(), listing)
                    for session, listing in zip(sessions, listings, strict=True)
                ],
                "score_hash": alpha_model_array_content_hash(values),
                "target_hash": alpha_model_array_content_hash(target),
                "raw_simple_return_hash": alpha_model_array_content_hash(raw),
            }
        )
    )
    return PanelScoreSurface(
        surface_id=surface_id,
        surface_hash=identity,
        fold_index=fold_index,
        span_sessions=1,
        row_sessions=sessions,
        row_listing_ids=listings,
        scores=values,
        target_z=target,
        raw_simple_returns=raw,
        model_recipe_id=model_recipe_id,
    )


def execute_panel_model_fold(
    *,
    program_hash: str,
    fold_index: int,
    request: PanelResearchMethodologyRequest,
    inner_projection: Callable[[InstalledPanelViewId], PanelFeatureProjection],
    outer_projection: Callable[[InstalledPanelViewId], PanelFeatureProjection],
    holding_horizon_sessions: int,
    release_inner_projection: Callable[[], None] | None = None,
) -> PanelModelFoldResult:
    """Select models on raw span-one Alpha evidence, then refit outer folds."""

    trials = installed_model_trials(request)
    catalog = build_panel_model_catalog()
    runtime = AlphaModelRuntimeService(catalog)
    package_identity_hash = str(
        canonical_hash(
            {
                "catalog_hash": catalog.binding.catalog_hash,
                "program_hash": program_hash,
                "request_hash": request.request_hash,
            }
        )
    )
    inner_evidence: list[PanelModelTrialEvidence] = []
    contents: list[ModelContentBundle] = []
    evidence_to_spec: dict[str, ModelTrialSpec] = {}
    complexities: dict[str, CandidateComplexity] = {}
    fit_calls = predict_calls = metric_calls = 0
    ridge_factorization_calls = 0

    def record_inner(
        *,
        spec: ModelTrialSpec,
        projection: PanelFeatureProjection,
        result: AlphaModelExecutionResult,
        binding_hash: str,
    ) -> None:
        nonlocal fit_calls, predict_calls, metric_calls
        fit_calls += result.provenance.fit_call_count
        predict_calls += result.provenance.predict_call_count
        scientific = panel_scientific_session_evidence(
            fold_index=fold_index,
            spec_id=spec.spec_id,
            view_id=spec.view_id,
            model_family_id=spec.family_id,
            recipe_hash=spec.recipe.recipe_hash,
            sessions=projection.row_sessions,
            listings=projection.row_listing_ids,
            scores=result.prediction.predictions,
            raw_simple_returns=projection.transformed_raw_simple_returns,
            evaluation_sessions=tuple(sorted(set(projection.row_sessions))),
        )
        evidence = _trial_evidence(
            program_hash=program_hash,
            fold_index=fold_index,
            phase="INNER_SELECTION",
            spec=spec,
            projection=projection,
            result=result,
            binding_hash=binding_hash,
            scientific=scientific,
        )
        inner_evidence.append(evidence)
        contents.append(
            ModelContentBundle(
                evidence_hash=evidence.evidence_hash,
                estimator=result.fit.estimator_content,
                execution=result,
            )
        )
        evidence_to_spec[scientific.evidence_hash] = spec
        complexities[spec.spec_id] = CandidateComplexity(
            transform_count=len(projection.ordered_feature_ids),
            model_order={"RIDGE": 0, "ELASTIC_NET": 1, "LIGHTGBM": 2}[spec.family_id],
        )
        metric_calls += 1

    with chronological_lightgbm_dataset_cache():
        for view_id in sorted({value.view_id for value in trials if value.family_id == "RIDGE"}):
            ridge_specs = tuple(
                sorted(
                    (
                        value
                        for value in trials
                        if value.family_id == "RIDGE" and value.view_id == view_id
                    ),
                    key=lambda value: value.spec_id,
                )
            )
            projection = inner_projection(view_id)
            for spec, result, binding_hash in _fit_ridge_batch(
                runtime=runtime,
                specs=ridge_specs,
                program_hash=program_hash,
                fold_index=fold_index,
                phase="INNER_SELECTION",
                projection=projection,
                package_identity_hash=package_identity_hash,
            ):
                record_inner(
                    spec=spec,
                    projection=projection,
                    result=result,
                    binding_hash=binding_hash,
                )
            ridge_factorization_calls += 1

        for spec in sorted(
            (value for value in trials if value.family_id != "RIDGE"),
            key=lambda value: (value.view_id, value.spec_id),
        ):
            projection = inner_projection(spec.view_id)
            result, binding_hash = _fit(
                runtime=runtime,
                spec=spec,
                program_hash=program_hash,
                fold_index=fold_index,
                phase="INNER_SELECTION",
                training_projection=projection,
                prediction_projection=projection,
                tuning_projection=projection,
                package_identity_hash=package_identity_hash,
            )
            record_inner(
                spec=spec,
                projection=projection,
                result=result,
                binding_hash=binding_hash,
            )

        grouped: dict[
            tuple[InstalledPanelViewId, ModelFamilyId], list[PanelModelTrialEvidence]
        ] = {}
        for value in inner_evidence:
            grouped.setdefault((value.view_id, value.model_family_id), []).append(value)
        group_selection_by_key = {
            group: select_panel_model_scientifically(
                fold_index=fold_index,
                evidence=tuple(
                    item.scientific_evidence
                    for item in values
                    if item.scientific_evidence is not None
                ),
                complexities=complexities,
                holding_horizon_sessions=holding_horizon_sessions,
            )
            for group, values in sorted(grouped.items())
        }
        group_selections = tuple(group_selection_by_key.values())
        scientific_evidence = tuple(
            value.scientific_evidence
            for value in inner_evidence
            if value.scientific_evidence is not None
        )
        deployable = select_panel_model_scientifically(
            fold_index=fold_index,
            evidence=scientific_evidence,
            complexities=complexities,
            holding_horizon_sessions=holding_horizon_sessions,
        )
        dynamic_evidence = tuple(
            value.scientific_evidence
            for value in inner_evidence
            if value.view_id != "RELATIVE_CONTROL" and value.scientific_evidence is not None
        )
        if not dynamic_evidence:
            raise PanelModelExecutionError("alpha_research.dynamic_challenger_candidate_absent")
        dynamic_challenger = select_panel_model_scientifically(
            fold_index=fold_index,
            evidence=dynamic_evidence,
            complexities=complexities,
            holding_horizon_sessions=holding_horizon_sessions,
        )
        selected_hashes = {
            value.operational_fallback_evidence_hash
            for value in (*group_selections, deployable, dynamic_challenger)
        }
        selected_specs = (
            {spec.spec_id: spec for spec in trials}
            if request.model_recipes
            else {
                evidence_to_spec[evidence_hash].spec_id: evidence_to_spec[evidence_hash]
                for evidence_hash in selected_hashes
            }
        )
        dynamic_spec = evidence_to_spec[dynamic_challenger.operational_fallback_evidence_hash]
        control_selection = group_selection_by_key[("RELATIVE_CONTROL", "RIDGE")]
        control_spec = evidence_to_spec[control_selection.operational_fallback_evidence_hash]
        nested_outer_tuning_required = any(
            spec.family_id == "LIGHTGBM"
            and spec.recipe.parameters.get("training_policy") != "FIXED_ITERATION"
            for spec in selected_specs.values()
        )
        if not nested_outer_tuning_required and release_inner_projection is not None:
            # Direct-fit candidates do not consume the inner matrix during the
            # outer refit.  Release it before constructing any outer boundary.
            release_inner_projection()

        outer_evidence: list[PanelModelTrialEvidence] = []
        outer_surfaces: dict[str, PanelScoreSurface] = {}
        for spec in sorted(
            selected_specs.values(), key=lambda value: (value.view_id, value.spec_id)
        ):
            outer = outer_projection(spec.view_id)
            tuning = inner_projection(spec.view_id) if nested_outer_tuning_required else outer
            result, binding_hash = _fit(
                runtime=runtime,
                spec=spec,
                program_hash=program_hash,
                fold_index=fold_index,
                phase="OUTER_VALIDATION",
                training_projection=outer,
                prediction_projection=outer,
                tuning_projection=tuning,
                package_identity_hash=package_identity_hash,
            )
            fit_calls += result.provenance.fit_call_count
            predict_calls += result.provenance.predict_call_count
            if spec.family_id == "RIDGE":
                ridge_factorization_calls += 1
            scientific = panel_scientific_session_evidence(
                fold_index=fold_index,
                spec_id=spec.spec_id,
                view_id=spec.view_id,
                model_family_id=spec.family_id,
                recipe_hash=spec.recipe.recipe_hash,
                sessions=outer.row_sessions,
                listings=outer.row_listing_ids,
                scores=result.prediction.predictions,
                raw_simple_returns=outer.transformed_raw_simple_returns,
                evaluation_sessions=tuple(sorted(set(outer.row_sessions))),
            )
            evidence = _trial_evidence(
                program_hash=program_hash,
                fold_index=fold_index,
                phase="OUTER_VALIDATION",
                spec=spec,
                projection=outer,
                result=result,
                binding_hash=binding_hash,
                scientific=scientific,
            )
            outer_evidence.append(evidence)
            contents.append(
                ModelContentBundle(
                    evidence_hash=evidence.evidence_hash,
                    estimator=result.fit.estimator_content,
                    execution=result,
                )
            )
            metric_calls += 1
            outer_surfaces[spec.spec_id] = _surface(
                surface_id=f"FIXED_CANDIDATE::{spec.recipe_id}::{spec.spec_id}",
                model_recipe_id=spec.recipe_id,
                fold_index=fold_index,
                projection=outer,
                predictions=result.prediction.predictions,
                common_evaluation_sessions=tuple(sorted(set(outer.row_sessions))),
            )

        if nested_outer_tuning_required and release_inner_projection is not None:
            release_inner_projection()

    raw_dynamic_surface = outer_surfaces[dynamic_spec.spec_id]
    selected_surface = raw_dynamic_surface
    control_surface = outer_surfaces[control_spec.spec_id]
    fixed_candidate_surfaces = tuple(
        outer_surfaces[spec.spec_id] for spec in trials if spec.spec_id in outer_surfaces
    )
    if any(
        value.row_sessions != control_surface.row_sessions
        or value.row_listing_ids != control_surface.row_listing_ids
        for value in fixed_candidate_surfaces
    ):
        raise PanelModelExecutionError("alpha_research.panel_score_common_axis_mismatch")
    fold_evidence = PanelModelFoldEvidence.create(
        program_hash=program_hash,
        fold_index=fold_index,
        inner_trial_evidence_hashes=tuple(value.evidence_hash for value in inner_evidence),
        group_selection_hashes=tuple(value.selection_hash for value in group_selections),
        deployable_selection_hash=deployable.selection_hash,
        dynamic_challenger_selection_hash=dynamic_challenger.selection_hash,
        score_aggregation_selection_hash=None,
        selection_contract_kind="PanelModelScientificSelection",
        outer_trial_evidence_hashes=tuple(value.evidence_hash for value in outer_evidence),
        raw_dynamic_surface_hash=raw_dynamic_surface.surface_hash,
        selected_dynamic_surface_hash=selected_surface.surface_hash,
        common_span_control_surface_hash=control_surface.surface_hash,
        fixed_candidate_surface_hashes=tuple(
            value.surface_hash for value in fixed_candidate_surfaces
        ),
        fit_call_count=fit_calls,
        predict_call_count=predict_calls,
        metric_call_count=metric_calls,
        ridge_factorization_call_count=ridge_factorization_calls,
        dataset_reuse_scope_count=1,
    )
    expected_content_evidence_hashes = {
        value.evidence_hash for value in (*inner_evidence, *outer_evidence)
    }
    if (
        len(contents) != len(expected_content_evidence_hashes)
        or {value.evidence_hash for value in contents} != expected_content_evidence_hashes
    ):
        raise PanelModelExecutionError("alpha_research.panel_model_content_set_incomplete")
    return PanelModelFoldResult(
        inner_evidence=tuple(inner_evidence),
        group_selections=group_selections,
        deployable_selection=deployable,
        dynamic_challenger_selection=dynamic_challenger,
        score_aggregation_selection=None,
        outer_evidence=tuple(outer_evidence),
        model_contents=tuple(contents),
        raw_dynamic_surface=raw_dynamic_surface,
        selected_dynamic_surface=selected_surface,
        common_span_control_surface=control_surface,
        fixed_candidate_surfaces=fixed_candidate_surfaces,
        fold_evidence=fold_evidence,
    )


__all__ = [
    "ModelContentBundle",
    "ModelFamilyId",
    "ModelTrialSpec",
    "PanelModelExecutionError",
    "PanelModelFoldEvidence",
    "PanelModelFoldResult",
    "PanelModelTrialEvidence",
    "PanelScoreSurface",
    "build_panel_model_catalog",
    "execute_panel_model_fold",
    "installed_model_trials",
]
