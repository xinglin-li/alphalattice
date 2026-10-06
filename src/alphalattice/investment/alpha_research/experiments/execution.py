"""Fold-at-a-time execution for Host-admitted Alpha model recipes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from alphalattice.capabilities.alpha_modeling.catalog import (
    AlphaModelCatalog,
    build_installed_alpha_model_catalog,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelFitProtocol,
    AlphaModelNumericalBinding,
    AlphaModelSearchDomainEnvelope,
    AlphaModelStateProjection,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from alphalattice.capabilities.alpha_modeling.runtime import AlphaModelRuntimeService
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..evaluation.contracts import AlphaMetricPolicy
from ..evaluation.metrics import (
    AlphaFoldMetricEvidence,
    FoldScoreVector,
    aggregate_candidate_metrics,
    compute_fold_metric_evidence,
)
from ..inputs.folds import (
    AlphaArrayWorkspaceLike,
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
    build_alpha_fold_commitment,
)
from ..inputs.loading import load_alpha_fold_arrays
from ..inputs.training import (
    AlphaTrainingInputAuthorityError,
    assert_alpha_training_authority,
    bind_alpha_model_inputs,
)
from ..targets.execution_outcome import AlphaTargetLane, AlphaTargetPolicy
from .bindings import (
    AlphaDevelopmentProgramAuthorityError,
    assert_alpha_development_fold_materialization,
    assert_alpha_development_program_authority,
)
from .contracts import (
    AlphaCandidateRole,
    AlphaDevelopmentFitEvidence,
    AlphaDevelopmentProgram,
    AlphaDevelopmentProgramAuthority,
    AlphaExperimentBatch,
    AlphaExperimentBatchResult,
    AlphaExperimentCandidateResult,
    AlphaFitLedgerEntry,
    candidate_id_for_spec,
    seal_contract,
)
from .contracts import (
    AlphaCandidateStatus as NumericalCandidateStatus,
)
from .development_artifacts import AlphaDevelopmentArtifactStore
from .development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaCandidateFoldEvidence,
    AlphaCandidateInferenceEvidence,
    AlphaCandidateNumericalFoldResult,
    AlphaDevelopmentEstimatorState,
    AlphaDevelopmentFoldSurface,
    AlphaDevelopmentSurfaceManifest,
    AlphaSessionScoreStatistics,
    seal_current_contract,
)
from .fit_plan import (
    AlphaModelFitPlan,
    build_alpha_model_fit_plan,
    resolve_alpha_model_fit_protocol,
)
from .mandate import (
    AlphaModelCapabilityAuthority,
    AlphaResearchModelRecipe,
    admitted_capabilities_installed,
    assert_research_recipe_target_authority,
)

type FloatArray = npt.NDArray[np.float64]


class AlphaExperimentProgress(Protocol):
    """Narrow observation sink implemented by Goal task-scoped progress."""

    def record_candidate_fold(
        self,
        *,
        candidate_id: str,
        fold_index: int,
        fit_calls: int,
        predict_calls: int,
        metric_calls: int,
        chunk_published: bool,
    ) -> None: ...


@dataclass(slots=True)
class _CandidateAccumulator:
    numerical_results: list[AlphaCandidateNumericalFoldResult] = field(default_factory=list)
    fold_evidence: list[AlphaCandidateFoldEvidence] = field(default_factory=list)
    metric_evidence: list[AlphaFoldMetricEvidence] = field(default_factory=list)
    estimator_states: list[AlphaDevelopmentEstimatorState] = field(default_factory=list)


def _state_projection_fields(projection: AlphaModelStateProjection) -> dict[str, object]:
    """Translate known current evaluators without constraining development adapters."""

    common: dict[str, object] = {
        "state_schema_id": projection.state_schema_id,
        "projection_payload": projection.payload,
        "coefficient_hex": (),
        "intercept_hex": "0x0.0p+0",
        "coefficient_l2_norm": 0.0,
        "coefficient_max_abs": 0.0,
        "nonzero_support_count": None,
        "model_text_hash": None,
        "best_iteration": None,
        "feature_gain_hex": (),
        "top_feature_gain_share": None,
    }
    if projection.state_kind == "LINEAR":
        common.update(
            {
                "coefficient_hex": projection.payload["coefficient_hex"],
                "intercept_hex": projection.payload["intercept_hex"],
                "coefficient_l2_norm": projection.payload["coefficient_l2_norm"],
                "coefficient_max_abs": projection.payload["coefficient_max_abs"],
                "nonzero_support_count": projection.payload["nonzero_support_count"],
            }
        )
    return common


def _validation_table(fold: AlphaFoldArrays) -> pa.Table:
    return pa.table(
        {
            "row_index": pa.array(range(len(fold.validation_listing_ids)), type=pa.int32()),
            "formation_session": pa.array(fold.validation_row_sessions, type=pa.date32()),
            "listing_id": pa.array(fold.validation_listing_ids, type=pa.string()),
            "target": pa.array(
                [
                    float(value) if bool(available) and np.isfinite(value) else None
                    for value, available in zip(
                        fold.validation_targets,
                        fold.validation_outcome_complete,
                        strict=True,
                    )
                ],
                type=pa.float64(),
            ),
            "economic_return": pa.array(
                [
                    float(value) if bool(available) and np.isfinite(value) else None
                    for value, available in zip(
                        fold.economic_validation_targets,
                        fold.validation_outcome_complete,
                        strict=True,
                    )
                ],
                type=pa.float64(),
            ),
            "feature_available": pa.array(
                [bool(value) for value in fold.validation_feature_complete], type=pa.bool_()
            ),
            "outcome_available": pa.array(
                [bool(value) for value in fold.validation_outcome_complete], type=pa.bool_()
            ),
            "feature_row_hash": pa.array(fold.validation_feature_row_hashes, type=pa.string()),
            "causal_outcome_row_hash": pa.array(
                fold.validation_outcome_row_hashes, type=pa.string()
            ),
        }
    )


def _score_table(scores: FloatArray, feature_complete: npt.NDArray[np.bool_]) -> pa.Table:
    return pa.table(
        {
            "row_index": pa.array(range(len(scores)), type=pa.int32()),
            "score": pa.array(
                [float(value) if np.isfinite(value) else None for value in scores],
                type=pa.float64(),
            ),
            "availability": pa.array(
                [
                    "SCORED" if bool(available) else "FEATURE_INCOMPLETE"
                    for available in feature_complete
                ],
                type=pa.string(),
            ),
        }
    )


def _session_statistics(
    sessions: tuple[date, ...], scores: FloatArray
) -> tuple[AlphaSessionScoreStatistics, ...]:
    # Sessions compared as ordinals: the same rows each session selects, without
    # an object comparison over every row for every session.
    ordinals = np.fromiter(
        (value.toordinal() for value in sessions), dtype=np.int64, count=len(sessions)
    )
    result: list[AlphaSessionScoreStatistics] = []
    for session in sorted(set(sessions)):
        values = scores[ordinals == session.toordinal()]
        finite = values[np.isfinite(values)]
        result.append(
            AlphaSessionScoreStatistics(
                formation_session=session,
                listing_count=len(values),
                scored_count=len(finite),
                score_mean=float(np.mean(finite)) if finite.size else 0.0,
                score_std=float(np.std(finite, ddof=0)) if finite.size else 0.0,
                score_coverage=len(finite) / len(values),
            )
        )
    return tuple(result)


def _execute_fold(
    *,
    program: AlphaDevelopmentProgramAuthority,
    batch: AlphaExperimentBatch,
    spec: AlphaResearchModelRecipe,
    surface_binding_hash: str,
    surface_hash: str,
    fold_surface: AlphaDevelopmentFoldSurface,
    fold: AlphaFoldArrays,
    store: AlphaDevelopmentArtifactStore,
    runtime_service: AlphaModelRuntimeService,
    domain: AlphaModelSearchDomainEnvelope,
    numerical_binding: AlphaModelNumericalBinding,
    fit_plan: AlphaModelFitPlan,
    bound_fit_plan: BoundAlphaModelFitInput,
    training_input: BoundAlphaTrainingInput,
    prediction_input: BoundAlphaPredictionInput,
) -> tuple[
    AlphaCandidateNumericalFoldResult,
    AlphaCandidateFoldEvidence,
    AlphaFoldMetricEvidence,
    AlphaDevelopmentEstimatorState,
    int,
    int,
]:
    candidate_id = candidate_id_for_spec(spec)
    execution_binding_hash = canonical_hash(
        {
            "program_hash": program.program_hash,
            "batch_hash": batch.batch_hash,
            "surface_binding_hash": surface_binding_hash,
            "spec_hash": spec.spec_hash,
            "fold_commitment_hash": fold.commitment.commitment_hash,
            "adapter_id": spec.recipe.adapter_id,
            "numerical_binding_hash": numerical_binding.numerical_binding_hash,
            "fit_plan_hash": fit_plan.fit_plan_hash,
        }
    )
    existing = store.find_candidate_numerical_fold_result(execution_binding_hash)
    if existing is not None:
        if existing.estimator_state_hash is None:
            raise ValueError("alpha_research.dynamic_fold_state_missing")
        state = store.load_development_estimator_state(existing.estimator_state_hash)
        if existing.score_chunk is None or existing.metrics is None:
            raise ValueError("alpha_research.dynamic_fold_result_incomplete")
        table = store.resolve_candidate_score_chunk(existing.score_chunk)
        scores: FloatArray = np.asarray(
            [float(value) if value is not None else np.nan for value in table["score"].to_pylist()],
            dtype=np.float64,
        )
        scores.setflags(write=False)
        # The sealed fold is reused as sealed: its metrics are the ones its numerical result
        # carries. Recomputed here they could differ in the last bits (OpenBLAS splits a long
        # dot product by thread count), and a run resumed under another thread count or on
        # another host would publish a report its own receipt refuses.
        metric = replace(
            compute_fold_metric_evidence(
                FoldScoreVector(fold=fold, scores=scores), policy=_metric_policy()
            ),
            metrics=existing.metrics,
            session_rank_ics=existing.session_rank_ics,
            session_spreads=existing.session_spreads,
        )
        thin = seal_current_contract(
            AlphaCandidateFoldEvidence,
            {
                "kind": "AlphaCandidateFoldEvidence",
                "execution_binding_hash": execution_binding_hash,
                "numerical_result_hash": existing.numerical_result_hash,
                "request_hash": program.program_hash,
                "development_surface_hash": surface_hash,
                "surface_fold_hash": fold_surface.surface_fold_hash,
                "candidate_id": candidate_id,
                "candidate_card_hash": spec.spec_hash,
                "fold_index": fold.commitment.fold_index,
                "fold_commitment_hash": fold.commitment.commitment_hash,
            },
            "fold_evidence_hash",
        )
        # Content-addressed and idempotent: the same bytes when the earlier run
        # published it, the missing file when that run died between the
        # numerical child and its evidence.
        store.publish_candidate_fold_evidence(thin)
        return existing, thin, metric, state, 0, 0
    train_mask = fold.training_model_mask
    if int(train_mask.sum()) <= len(fold.ordered_factor_ids):
        raise ValueError("alpha_research.dynamic_training_surface_insufficient")
    prediction_mask = fold.validation_feature_complete
    execution = runtime_service.execute(
        recipe=spec.recipe,
        domain=domain,
        fit_plan=bound_fit_plan,
        training_input=training_input,
        prediction_input=prediction_input,
        package_identity_hash=program.package_identity_hash,
    )
    fit = execution.fit
    prediction = execution.prediction
    provenance = execution.provenance
    scores = np.full(len(fold.validation_listing_ids), np.nan, dtype=np.float64)
    scores[prediction_mask] = prediction.predictions
    scores.setflags(write=False)
    metric = compute_fold_metric_evidence(
        FoldScoreVector(fold=fold, scores=scores), policy=_metric_policy()
    )
    chunk = store.publish_numerical_score_chunk(
        _score_table(scores, prediction_mask),
        execution_binding_hash=execution_binding_hash,
        development_surface_binding_hash=surface_binding_hash,
        candidate_id=candidate_id,
        candidate_card_hash=spec.spec_hash,
        fold_index=fold.commitment.fold_index,
        fold_commitment_hash=fold.commitment.commitment_hash,
    )
    statistics = _session_statistics(fold.validation_row_sessions, scores)
    finite = scores[np.isfinite(scores)]
    fit_evidence = seal_contract(
        AlphaDevelopmentFitEvidence,
        {
            "kind": "AlphaDevelopmentFitEvidence",
            "execution_binding_hash": execution_binding_hash,
            "candidate_id": candidate_id,
            "fold_index": fold.commitment.fold_index,
            "adapter_id": spec.recipe.adapter_id,
            "recipe_hash": spec.recipe.recipe_hash,
            "training_binding_hash": training_input.training_binding_hash,
            "fit_plan_hash": execution.fit_plan_hash,
            "numerical_binding_hash": execution.numerical_binding.numerical_binding_hash,
            "numerical_environment_hash": (execution.numerical_environment.environment_hash),
            "estimator_content_hash": fit.estimator_content.content_hash,
            "fit_provenance_hash": provenance.provenance_hash,
            "state_projection_hash": fit.state_projection.projection_hash,
            "score_evidence_hash": canonical_hash(
                {
                    "score_chunk_content_hash": chunk.content_hash,
                    "fold_metrics_hash": metric.metrics.metrics_hash,
                }
            ),
            "fit_call_count": fit.fit_call_count,
            "predict_call_count": fit.predict_call_count + prediction.predict_call_count,
        },
        "evidence_hash",
    )
    store.publish_model_fit_plan(fit_plan)
    store.publish_model_numerical_environment(execution.numerical_environment)
    store.publish_development_fit_evidence(fit_evidence)
    projection = fit.state_projection
    state = seal_current_contract(
        AlphaDevelopmentEstimatorState,
        {
            "kind": "AlphaDevelopmentEstimatorState",
            "execution_binding_hash": execution_binding_hash,
            "development_surface_binding_hash": surface_binding_hash,
            "candidate_id": candidate_id,
            "candidate_card_hash": spec.spec_hash,
            "fold_index": fold.commitment.fold_index,
            "fold_commitment_hash": fold.commitment.commitment_hash,
            "adapter_id": projection.adapter_id,
            "numerical_binding_hash": execution.numerical_binding.numerical_binding_hash,
            "fit_evidence_hash": fit_evidence.evidence_hash,
            "state_projection_hash": projection.projection_hash,
            **_state_projection_fields(projection),
            "family_id": projection.model_family_id,
            "state_kind": projection.state_kind,
            "ordered_factor_ids": fold.ordered_factor_ids,
            "training_mse": fit.training_mse,
            "validation_score_mean": float(np.mean(finite)) if finite.size else 0.0,
            "validation_score_std": float(np.std(finite, ddof=0)) if finite.size else 0.0,
            "validation_score_coverage": finite.size / scores.size if scores.size else 0.0,
            "validation_session_statistics": statistics,
        },
        "state_hash",
    )
    store.publish_model_fit_sidecar(
        operation_binding_hash=execution_binding_hash,
        estimator_content=fit.estimator_content,
        fit_provenance=provenance,
    )
    store.publish_development_estimator_state(state)
    ledger = _fit_ledger(candidate_id, fold, train_mask, prediction_mask, state.state_hash)
    numerical = seal_current_contract(
        AlphaCandidateNumericalFoldResult,
        {
            "kind": "AlphaCandidateNumericalFoldResult",
            "execution_binding_hash": execution_binding_hash,
            "development_surface_binding_hash": surface_binding_hash,
            "candidate_id": candidate_id,
            "candidate_card_hash": spec.spec_hash,
            "role": AlphaCandidateRole.MODEL_ALPHA,
            "fold_index": fold.commitment.fold_index,
            "fold_commitment_hash": fold.commitment.commitment_hash,
            "status": NumericalCandidateStatus.SUCCEEDED,
            "score_chunk": chunk,
            "metrics": metric.metrics,
            "session_rank_ics": metric.session_rank_ics,
            "session_spreads": metric.session_spreads,
            "fit_ledger": ledger,
            "estimator_state_hash": state.state_hash,
            "failure": None,
        },
        "numerical_result_hash",
    )
    store.publish_candidate_numerical_fold_result(numerical)
    thin = seal_current_contract(
        AlphaCandidateFoldEvidence,
        {
            "kind": "AlphaCandidateFoldEvidence",
            "execution_binding_hash": execution_binding_hash,
            "numerical_result_hash": numerical.numerical_result_hash,
            "request_hash": program.program_hash,
            "development_surface_hash": surface_hash,
            "surface_fold_hash": fold_surface.surface_fold_hash,
            "candidate_id": candidate_id,
            "candidate_card_hash": spec.spec_hash,
            "fold_index": fold.commitment.fold_index,
            "fold_commitment_hash": fold.commitment.commitment_hash,
        },
        "fold_evidence_hash",
    )
    store.publish_candidate_fold_evidence(thin)
    return (
        numerical,
        thin,
        metric,
        state,
        fit.fit_call_count,
        fit.predict_call_count + prediction.predict_call_count,
    )


def execute_alpha_model_batch(
    *,
    program: AlphaDevelopmentProgramAuthority,
    batch: AlphaExperimentBatch,
    fold_plan: AlphaFoldArrayPlan,
    store: AlphaDevelopmentArtifactStore,
    array_workspace: AlphaArrayWorkspaceLike | None = None,
    fold_plans: Mapping[AlphaTargetLane, AlphaFoldArrayPlan] | None = None,
    array_workspaces: Mapping[AlphaTargetLane, AlphaArrayWorkspaceLike] | None = None,
    model_catalog: AlphaModelCatalog | None = None,
    model_mandate: AlphaModelCapabilityAuthority,
    operational_progress: AlphaExperimentProgress | None = None,
) -> AlphaExperimentBatchResult:
    """Run one admitted batch fold-at-a-time and publish reusable numerical children."""

    if isinstance(program, AlphaDevelopmentProgram) and (
        fold_plans is not None or array_workspaces is not None
    ):
        raise AlphaDevelopmentProgramAuthorityError("ALPHA_DEVELOPMENT_PROGRAM_ONE_TARGET_REQUIRED")
    assert_alpha_development_program_authority(
        program=program,
        fold_plan=fold_plan,
        model_mandate=model_mandate,
    )
    if batch.program_hash != program.program_hash:
        if isinstance(program, AlphaDevelopmentProgram):
            raise AlphaDevelopmentProgramAuthorityError("ALPHA_DEVELOPMENT_BATCH_PROGRAM_MISMATCH")
        raise ValueError("alpha_research.batch_program_mismatch")
    catalog = model_catalog or build_installed_alpha_model_catalog()
    if (
        program.model_mandate_hash != model_mandate.mandate_hash
        or program.model_catalog_hash != model_mandate.catalog_binding.catalog_hash
        or not admitted_capabilities_installed(model_mandate, catalog)
    ):
        if isinstance(program, AlphaDevelopmentProgram):
            raise AlphaDevelopmentProgramAuthorityError(
                "ALPHA_DEVELOPMENT_MODEL_AUTHORITY_MISMATCH"
            )
        raise ValueError("alpha_research.batch_model_authority_mismatch")
    if not all(isinstance(value, AlphaResearchModelRecipe) for value in batch.specs):
        raise ValueError("alpha_research.legacy_model_spec_not_active")
    admitted_specs = tuple(
        value for value in batch.specs if isinstance(value, AlphaResearchModelRecipe)
    )
    targeted = tuple(value.target_lane for value in admitted_specs)
    used_lanes = {value for value in targeted if value is not None}
    standalone_lane = (
        program.target_policy.lane
        if isinstance(program, AlphaDevelopmentProgram) and program.target_policy is not None
        else None
    )
    target_plans = (
        {standalone_lane: fold_plan} if standalone_lane is not None else (fold_plans or {})
    )
    assert_research_recipe_target_authority(
        program_target_policy_hashes=program.target_policy_hashes,
        recipe_target_lanes=targeted,
        default_target_policy=(None if standalone_lane is not None else fold_plan.target_policy),
        lane_target_policies={
            lane: target_plans[lane].target_policy for lane in used_lanes if lane in target_plans
        },
    )
    domains = {value.search_domain_hash: value for value in model_mandate.ordered_search_domains}
    numerical_bindings: dict[str, AlphaModelNumericalBinding] = {}
    fit_protocols: dict[str, AlphaModelFitProtocol] = {}
    for spec in admitted_specs:
        try:
            domain = domains[spec.search_domain_hash]
        except KeyError as error:
            raise ValueError("alpha_research.recipe_domain_not_mandated") from error
        adapter = catalog.admit_recipe(recipe=spec.recipe, domain=domain)
        numerical_bindings[spec.spec_hash] = adapter.describe_numerical_binding()
        fit_protocols[spec.spec_hash] = resolve_alpha_model_fit_protocol(
            adapter=adapter, recipe=spec.recipe, domain=domain
        )
    runtime_service = AlphaModelRuntimeService(catalog)
    if any(value is not None for value in targeted) and any(value is None for value in targeted):
        raise ValueError("alpha_research.target_lane_batch_incomplete")
    if targeted and targeted[0] is not None:
        if fold_plans is None and standalone_lane is None:
            raise ValueError("alpha_research.target_lane_plan_unavailable")
        grouped_specs: tuple[
            tuple[AlphaTargetLane | None, tuple[AlphaResearchModelRecipe, ...]], ...
        ] = tuple(
            (lane, tuple(value for value in admitted_specs if value.target_lane is lane))
            for lane in AlphaTargetLane
            if any(value.target_lane is lane for value in batch.specs)
        )
    else:
        grouped_specs = ((None, admitted_specs),)
    accumulators = {
        candidate_id_for_spec(value): _CandidateAccumulator() for value in admitted_specs
    }
    candidate_surfaces: dict[
        str, tuple[AlphaTargetLane | None, str, str, tuple[str, ...], int]
    ] = {}
    all_fold_surfaces: list[AlphaDevelopmentFoldSurface] = []
    fit_calls = predict_calls = metric_calls = 0
    for lane, specs in grouped_specs:
        active_plan = fold_plan if lane is None else target_plans[lane]
        active_workspace = (
            array_workspace
            if lane is None or lane is standalone_lane
            else (array_workspaces or {}).get(lane)
        )
        commitments = tuple(
            build_alpha_fold_commitment(window).commitment_hash
            for window in active_plan.split_plan.windows
        )
        surface_binding_hash = canonical_hash(
            {
                "program_hash": program.program_hash,
                "foundation_hash": program.foundation_hash,
                "logical_panel_hash": program.logical_panel_hash,
                "logical_semantic_index_hash": program.logical_semantic_index_hash,
                "causal_outcome_snapshot_hash": program.causal_outcome_snapshot_hash,
                "split_policy_hash": program.split_policy_hash,
                "target_lane": lane.value if lane is not None else None,
                "target_policy_hash": (
                    AlphaTargetPolicy.model_validate(active_plan.target_policy).policy_hash
                    if active_plan.target_policy is not None
                    else None
                ),
                "additional_factor_ids": active_plan.additional_factor_ids,
                "feature_context_hash": active_plan.feature_context_hash,
                "fold_window_hashes": commitments,
            }
        )
        candidate_ids = tuple(candidate_id_for_spec(spec) for spec in specs)
        surface = seal_current_contract(
            AlphaDevelopmentSurfaceManifest,
            {
                "kind": "AlphaDevelopmentSurfaceManifest",
                "request_hash": program.program_hash,
                "foundation_hash": program.foundation_hash,
                "logical_panel_hash": program.logical_panel_hash,
                "logical_semantic_index_hash": program.logical_semantic_index_hash,
                "causal_outcome_snapshot_hash": program.causal_outcome_snapshot_hash,
                "listing_set_hash": canonical_hash(active_plan.ordered_listing_ids),
                "ordered_listing_ids_hash": program.ordered_listing_ids_hash,
                "ordered_factor_ids_hash": canonical_hash(
                    (
                        *active_plan.base_feature_ids,
                        *active_plan.additional_factor_ids,
                    )
                ),
                "candidate_ids": candidate_ids,
                "candidate_card_hashes": tuple(value.spec_hash for value in specs),
                "fold_commitment_hashes": commitments,
                "split_hash": active_plan.split_plan.split_hash,
            },
            "surface_hash",
        )
        store.publish_surface(surface)
        lane_fold_surfaces: list[AlphaDevelopmentFoldSurface] = []
        for fold_index in range(active_plan.fold_count):
            fold_context = (
                active_workspace.fold_lease(fold_index)
                if active_workspace is not None
                else nullcontext(load_alpha_fold_arrays(active_plan, fold_index))
            )
            with fold_context as fold:
                assert_alpha_development_fold_materialization(
                    program=program,
                    fold_plan=active_plan,
                    fold=fold,
                )
                commitment_hash = fold.commitment.commitment_hash
                # The plan's base axis, which is the foundation's whole axis
                # unless a development run declared a subset. Taken from the plan
                # so this check compares the training binding against the axis the
                # arrays were built over rather than the one the parent evidence
                # answered for.
                expected_feature_ids = (
                    *active_plan.base_feature_ids,
                    *active_plan.additional_factor_ids,
                )
                try:
                    assert_alpha_training_authority(
                        fold.training_input_binding,
                        scope="DEVELOPMENT_FOLD",
                        foundation_hash=active_plan.foundation.foundation_hash,
                        feature_panel_snapshot_hash=(
                            active_plan.foundation.feature_panel_snapshot_hash
                        ),
                        causal_outcome_snapshot_hash=(
                            active_plan.foundation.execution_outcome.snapshot_hash
                        ),
                        target_policy=active_plan.target_policy,
                        ordered_feature_ids=expected_feature_ids,
                        feature_context_hash=active_plan.feature_context_hash,
                        training_cutoff=(
                            active_plan.split_plan.windows[fold_index].train_sessions[-1]
                        ),
                        outcome_maturity_session=(
                            active_plan.split_plan.windows[fold_index].train_sessions[-1]
                        ),
                        prediction_anchor=(
                            active_plan.split_plan.windows[fold_index].validation_sessions[0]
                        ),
                        fold_commitment_hash=commitment_hash,
                    )
                    training_input, prediction_input = bind_alpha_model_inputs(
                        binding=fold.training_input_binding,
                        ordered_feature_ids=fold.ordered_factor_ids,
                        training_features=fold.training_features,
                        training_targets=fold.training_targets,
                        training_mask=fold.training_model_mask,
                        prediction_features=fold.validation_features,
                        prediction_mask=fold.validation_feature_complete,
                    )
                except AlphaTrainingInputAuthorityError as error:
                    if isinstance(program, AlphaDevelopmentProgram):
                        raise AlphaDevelopmentProgramAuthorityError(
                            "ALPHA_DEVELOPMENT_TRAINING_AUTHORITY_MISMATCH"
                        ) from error
                    raise
                validation = store.publish_validation_chunk(
                    _validation_table(fold),
                    request_hash=program.program_hash,
                    development_surface_hash=surface.surface_hash,
                    fold_index=fold_index,
                    fold_commitment_hash=commitment_hash,
                )
                fold_surface = seal_current_contract(
                    AlphaDevelopmentFoldSurface,
                    {
                        "kind": "AlphaDevelopmentFoldSurface",
                        "request_hash": program.program_hash,
                        "development_surface_hash": surface.surface_hash,
                        "fold_index": fold_index,
                        "fold_commitment_hash": commitment_hash,
                        "validation_chunk": validation,
                    },
                    "surface_fold_hash",
                )
                store.publish_fold_surface(fold_surface)
                lane_fold_surfaces.append(fold_surface)
                all_fold_surfaces.append(fold_surface)
                for spec in specs:
                    candidate_id = candidate_id_for_spec(spec)
                    domain = domains[spec.search_domain_hash]
                    numerical_binding = numerical_bindings[spec.spec_hash]
                    fit_plan, bound_fit_plan = build_alpha_model_fit_plan(
                        fold=fold,
                        training_input=training_input,
                        domain=domain,
                        numerical_binding=numerical_binding,
                        protocol=fit_protocols[spec.spec_hash],
                    )
                    numerical, thin, metric, state, fit_count, predict_count = _execute_fold(
                        program=program,
                        batch=batch,
                        spec=spec,
                        surface_binding_hash=surface_binding_hash,
                        surface_hash=surface.surface_hash,
                        fold_surface=fold_surface,
                        fold=fold,
                        store=store,
                        runtime_service=runtime_service,
                        domain=domain,
                        numerical_binding=numerical_binding,
                        fit_plan=fit_plan,
                        bound_fit_plan=bound_fit_plan,
                        training_input=training_input,
                        prediction_input=prediction_input,
                    )
                    if operational_progress is not None:
                        operational_progress.record_candidate_fold(
                            candidate_id=candidate_id,
                            fold_index=fold.commitment.fold_index,
                            fit_calls=fit_count,
                            predict_calls=predict_count,
                            metric_calls=int(fit_count > 0),
                            chunk_published=fit_count > 0,
                        )
                    accumulator = accumulators[candidate_id]
                    accumulator.numerical_results.append(numerical)
                    accumulator.fold_evidence.append(thin)
                    accumulator.metric_evidence.append(metric)
                    accumulator.estimator_states.append(state)
                    fit_calls += fit_count
                    predict_calls += predict_count
                    metric_calls += int(fit_count > 0)
        fold_hashes = tuple(value.surface_fold_hash for value in lane_fold_surfaces)
        for candidate_id in candidate_ids:
            candidate_surfaces[candidate_id] = (
                lane,
                surface.surface_hash,
                surface_binding_hash,
                fold_hashes,
                active_plan.fold_count,
            )

    results: list[AlphaExperimentCandidateResult] = []
    for spec in batch.specs:
        candidate_id = candidate_id_for_spec(spec)
        accumulator = accumulators[candidate_id]
        lane, surface_hash, surface_binding_hash, fold_hashes, fold_count = candidate_surfaces[
            candidate_id
        ]
        metrics = aggregate_candidate_metrics(candidate_id, tuple(accumulator.metric_evidence))
        report = seal_current_contract(
            AlphaCandidateDevelopmentReport,
            {
                "kind": "AlphaCandidateDevelopmentReport",
                "request_hash": program.program_hash,
                "development_surface_hash": surface_hash,
                "foundation_hash": program.foundation_hash,
                "candidate_id": candidate_id,
                "candidate_card_hash": spec.spec_hash,
                "role": AlphaCandidateRole.MODEL_ALPHA,
                "status": NumericalCandidateStatus.SUCCEEDED,
                "admitted_fold_count": fold_count,
                "fold_evidence_hashes": tuple(
                    value.fold_evidence_hash for value in accumulator.fold_evidence
                ),
                "metrics": metrics,
                "estimator_state_hashes": tuple(
                    value.state_hash for value in accumulator.estimator_states
                ),
                "failure_codes": (),
            },
            "report_hash",
        )
        store.publish_candidate_report(report)
        inference = seal_current_contract(
            AlphaCandidateInferenceEvidence,
            {
                "kind": "AlphaCandidateInferenceEvidence",
                "request_hash": program.program_hash,
                "candidate_id": candidate_id,
                "report_hash": report.report_hash,
                "score_chunk_hashes": tuple(
                    value.score_chunk.content_hash
                    for value in accumulator.numerical_results
                    if value.score_chunk is not None
                ),
                "successful_fold_count": fold_count,
                "admitted_fold_count": fold_count,
                "scored_row_count": sum(
                    value.metrics.scored_comparison_row_count
                    for value in accumulator.numerical_results
                    if value.metrics is not None
                ),
                "common_surface_row_count": sum(
                    value.metrics.comparison_row_count
                    for value in accumulator.numerical_results
                    if value.metrics is not None
                ),
                "estimator_state_hashes": tuple(
                    value.state_hash for value in accumulator.estimator_states
                ),
            },
            "evidence_hash",
        )
        store.publish_candidate_inference_evidence(inference)
        results.append(
            seal_contract(
                AlphaExperimentCandidateResult,
                {
                    "candidate_id": candidate_id,
                    "spec_hash": spec.spec_hash,
                    "development_report_hash": report.report_hash,
                    "inference_evidence_hash": inference.evidence_hash,
                    "numerical_result_hashes": tuple(
                        value.numerical_result_hash for value in accumulator.numerical_results
                    ),
                    "estimator_state_hashes": tuple(
                        value.state_hash for value in accumulator.estimator_states
                    ),
                    "pooled_oos_r2": metrics.zero_relative_oos_r2.value,
                    "mean_rank_ic": metrics.rank_ic_mean.value,
                    "mean_gross_decile_spread": metrics.gross_decile_spread_mean.value,
                    "fold_coverage_mean": metrics.fold_coverage_mean,
                    "target_lane": lane,
                    "development_surface_hash": surface_hash,
                    "development_surface_binding_hash": surface_binding_hash,
                    "fold_surface_hashes": fold_hashes,
                    "status": "DEVELOPMENT_EVALUATED",
                    "failure_codes": (),
                },
                "result_hash",
            )
        )
    surface_hashes = tuple(dict.fromkeys(value[1] for value in candidate_surfaces.values()))
    surface_binding_hashes = tuple(dict.fromkeys(value[2] for value in candidate_surfaces.values()))
    return seal_contract(
        AlphaExperimentBatchResult,
        {
            "program_hash": program.program_hash,
            "batch_hash": batch.batch_hash,
            "development_surface_hash": (
                surface_hashes[0] if len(surface_hashes) == 1 else canonical_hash(surface_hashes)
            ),
            "development_surface_binding_hash": (
                surface_binding_hashes[0]
                if len(surface_binding_hashes) == 1
                else canonical_hash(surface_binding_hashes)
            ),
            "fold_surface_hashes": tuple(value.surface_fold_hash for value in all_fold_surfaces),
            "candidates": tuple(results),
            "fit_call_count": fit_calls,
            "predict_call_count": predict_calls,
            "metric_call_count": metric_calls,
        },
        "result_hash",
    )


def _fit_ledger(
    candidate_id: str,
    fold: AlphaFoldArrays,
    training_mask: npt.NDArray[np.bool_],
    prediction_mask: npt.NDArray[np.bool_],
    state_hash: str,
) -> AlphaFitLedgerEntry:
    from .contracts import seal_contract as seal_alpha_contract

    return seal_alpha_contract(
        AlphaFitLedgerEntry,
        {
            "candidate_id": candidate_id,
            "fold_index": fold.commitment.fold_index,
            "training_row_count": int(training_mask.sum()),
            "prediction_row_count": int(prediction_mask.sum()),
            "estimator_state_hash": state_hash,
            "status": "COMPLETED",
            "failure_code": None,
        },
        "ledger_hash",
    )


def _metric_policy() -> AlphaMetricPolicy:
    from .policies import load_alpha_metric_policy

    return load_alpha_metric_policy()


__all__ = ["execute_alpha_model_batch"]
