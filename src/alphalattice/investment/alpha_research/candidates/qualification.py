"""Viability and current-stability qualification of Alpha candidates (the owner's rule)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    RegularizedLinearParameters,
)
from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.capabilities.alpha_modeling.extension import DeclaredModelAdapter
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.source_identity import switched_source_identity
from alphalattice.kernel.validation.return_model_statistics import holm_adjust
from alphalattice.kernel.validation.screening_statistics import newey_west_mean_test

from ..evaluation.contracts import AlphaMetricPolicy
from ..evaluation.metrics import (
    AlphaFoldMetricEvidence,
    FoldScoreVector,
    aggregate_candidate_metrics,
    compute_fold_metric_evidence,
)
from ..evaluation.stability import AlphaDevelopmentState
from ..evaluation.viability import (
    AlphaValidationRobustnessEvidence,
    AlphaValidationRows,
    assess_alpha_model_viability,
)
from ..experiments.contracts import AlphaCandidateRole
from ..experiments.contracts import AlphaCandidateStatus as NumericalCandidateStatus
from ..experiments.development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaDevelopmentEstimatorState,
    seal_current_contract,
)
from ..experiments.mandate import (
    AlphaResearchModelMandate,
    AlphaResearchModelRecipe,
    admitted_capabilities_installed,
    assert_research_recipe_target_authority,
)
from ..inputs.folds import (
    AlphaArrayWorkspaceLike,
    AlphaCurrentRefitArrays,
    AlphaFoldArrayPlan,
)
from ..inputs.loading import load_alpha_fold_arrays, prepare_alpha_current_refit_arrays
from ..inputs.training import assert_alpha_training_authority
from ..publication.artifacts import (
    AlphaCurrentArtifactStore,
    build_current_candidate_score_child,
)
from ..publication.contracts import AlphaStabilityCheck, CurrentRefitStabilityAssessment
from ..scores.refit import (
    CurrentAlphaRefitResult,
    fit_and_assess_current_model,
    fit_and_assess_current_regularized_linear,
)
from ..targets.execution_outcome import AlphaTargetLane
from .contracts import (
    AlphaCandidateRecord,
    AlphaCandidateRegistrySnapshot,
    AlphaCandidateScoreCorrelation,
    AlphaCandidateStatus,
    AlphaExperimentBatchResult,
    AlphaExperimentCandidateResult,
    AlphaOosEvidenceAssessment,
    AlphaOosEvidenceClassification,
    AlphaOosEvidenceRecord,
    AlphaQualificationSnapshot,
    AlphaResearchProgram,
    StoredModelSpec,
    seal_contract,
)
from .control import update_registry_candidates

type FloatArray = npt.NDArray[np.float64]


class _OosRecordFields(TypedDict):
    development_result_ref: str
    development_candidate_hash: str
    holm_adjusted_p_value: float | None
    oos_evidence_classification: AlphaOosEvidenceClassification
    mean_rank_ic: float | None
    mean_gross_decile_spread: float | None


@dataclass(frozen=True, slots=True)
class AlphaQualificationResult:
    """Retain qualification, revised registry and deterministic numerical call counts."""

    registry: AlphaCandidateRegistrySnapshot
    qualification: AlphaQualificationSnapshot
    fit_call_count: int
    predict_call_count: int
    metric_call_count: int
    oos_evidence: AlphaOosEvidenceAssessment | None = None


def current_stability_policy_hash() -> str:
    """The installed current-refit stability rule's identity.

    Its owner's source, by its rule and kept at its value when the rule replaced the text
    (LAWS.md ID3), and the contracts it seals, bound into every qualification Program.
    """
    source_root = Path(__file__).resolve().parents[2]
    policy_owner = source_root / "alpha_research" / "evaluation" / "stability.py"
    return cast(
        str,
        canonical_hash(
            {
                "owner": "alpha_research.evaluation.stability.build_estimator_stability_checks",
                "policy": "CURRENT_REFIT_STABILITY_GATE",
                "source_hash": switched_source_identity(
                    {"alpha_research.evaluation.stability": policy_owner},
                    semantic_owner="alpha_research.evaluation",
                    numerical_role="CURRENT_REFIT_STABILITY_GATE",
                ),
                "contract_schema_hash": canonical_hash(
                    {
                        "check": schema_structure(AlphaStabilityCheck),
                        "assessment": schema_structure(CurrentRefitStabilityAssessment),
                    }
                ),
            }
        ),
    )


def _parameters(spec: StoredModelSpec) -> RegularizedLinearParameters:
    if isinstance(spec, AlphaResearchModelRecipe):
        if spec.recipe.adapter_id != "regularized_linear":
            raise ValueError("SCIENTIFIC_NON_ADMISSION")
        return RegularizedLinearParameters(**spec.recipe.parameters)
    return RegularizedLinearParameters(
        family=spec.family,
        alpha=spec.alpha,
        alpha_max_multiplier=spec.alpha_max_multiplier,
        l1_ratio=spec.l1_ratio,
    )


def _declared_model(spec: StoredModelSpec, catalog: AlphaModelCatalog | None) -> bool:
    """Whether the spec runs an agent's model a person activated (EX), whose contract proved it
    fits, predicts and projects its state."""

    if not isinstance(spec, AlphaResearchModelRecipe) or catalog is None:
        return False
    try:
        return isinstance(catalog.adapter(spec.recipe.adapter_id), DeclaredModelAdapter)
    except (KeyError, ValueError):
        return False


def _current_evaluator_available(
    spec: StoredModelSpec, catalog: AlphaModelCatalog | None = None
) -> bool:
    """Current stability is admitted for regularized linear and, through its adapter's
    projection, for an activated agent's model; an installed tree's is not."""

    return (
        not isinstance(spec, AlphaResearchModelRecipe)
        or spec.recipe.adapter_id == "regularized_linear"
        or _declared_model(spec, catalog)
    )


def _current_refit(
    spec: StoredModelSpec,
    *,
    catalog: AlphaModelCatalog | None,
    request_hash: str,
    candidate_id: str,
    arrays: AlphaCurrentRefitArrays,
    development_states: tuple[AlphaDevelopmentState, ...],
    package_identity_hash: str,
    retain_diagnostic_score: bool = False,
) -> CurrentAlphaRefitResult:
    """The current refit its evaluator admits: an agent's model through its adapter's
    projection, regularized linear as before."""

    if _declared_model(spec, catalog):
        assert isinstance(spec, AlphaResearchModelRecipe) and catalog is not None
        return fit_and_assess_current_model(
            request_hash=request_hash,
            candidate_id=candidate_id,
            recipe=spec.recipe,
            arrays=arrays,
            development_states=development_states,
            package_identity_hash=package_identity_hash,
            model_catalog=catalog,
            retain_diagnostic_score=retain_diagnostic_score,
        )
    return fit_and_assess_current_regularized_linear(
        request_hash=request_hash,
        candidate_id=candidate_id,
        parameters=_parameters(spec),
        arrays=arrays,
        development_states=development_states,
        package_identity_hash=package_identity_hash,
        retain_diagnostic_score=retain_diagnostic_score,
    )


def _fold_validation_rows(
    *,
    fold_index: int,
    validation_table: pa.Table,
    score_table: pa.Table,
) -> tuple[tuple[tuple[int, date, str], ...], FloatArray, FloatArray]:
    # Kept local to this owner so PyArrow tables never enter the durable Graph state.
    validation = validation_table
    scores_source = score_table
    sessions = tuple(validation["formation_session"].to_pylist())
    listings = tuple(str(value) for value in validation["listing_id"].to_pylist())
    target_column = "economic_return" if "economic_return" in validation.column_names else "target"
    targets: FloatArray = np.asarray(
        [
            float(value) if value is not None else np.nan
            for value in validation[target_column].to_pylist()
        ],
        dtype=np.float64,
    )
    scores: FloatArray = np.asarray(
        [
            float(value) if value is not None else np.nan
            for value in scores_source["score"].to_pylist()
        ],
        dtype=np.float64,
    )
    admitted = (
        np.asarray(validation["feature_available"].to_pylist(), dtype=np.bool_)
        & np.asarray(validation["outcome_available"].to_pylist(), dtype=np.bool_)
        & np.isfinite(targets)
        & np.isfinite(scores)
    )
    indices = np.flatnonzero(admitted)
    keys = tuple((fold_index, sessions[index], listings[index]) for index in indices)
    target_values = targets[indices]
    score_values = scores[indices]
    target_values.setflags(write=False)
    score_values.setflags(write=False)
    return keys, target_values, score_values


def _candidate_rows(
    *,
    candidate_id: str,
    result: AlphaExperimentBatchResult,
    candidate_result_hashes: tuple[str, ...],
    fold_surface_hashes: tuple[str, ...] | None,
    store: AlphaCurrentArtifactStore,
) -> tuple[AlphaValidationRows, AlphaValidationRobustnessEvidence]:
    keys: list[tuple[int, date, str]] = []
    targets: list[FloatArray] = []
    scores: list[FloatArray] = []
    rank_ics: list[float] = []
    spreads: list[float] = []
    surfaces = fold_surface_hashes or result.fold_surface_hashes
    if len(candidate_result_hashes) != len(surfaces):
        raise ValueError("alpha_research.development_surface_incomplete")
    for fold_surface_hash, numerical_hash in zip(surfaces, candidate_result_hashes, strict=True):
        surface, validation_table = store.read_fold_surface(fold_surface_hash)
        numerical, score_table = store.read_candidate_numerical_fold_result(numerical_hash)
        if (
            numerical.candidate_id != candidate_id
            or numerical.fold_index != surface.fold_index
            or numerical.score_chunk is None
            or score_table is None
        ):
            raise ValueError("alpha_research.development_surface_incomplete")
        fold_keys, fold_targets, fold_scores = _fold_validation_rows(
            fold_index=surface.fold_index,
            validation_table=validation_table,
            score_table=score_table,
        )
        keys.extend(fold_keys)
        targets.append(fold_targets)
        scores.append(fold_scores)
        rank_ics.extend(numerical.session_rank_ics)
        spreads.extend(numerical.session_spreads)
    return (
        AlphaValidationRows(
            candidate_id=candidate_id,
            keys=tuple(keys),
            targets=np.concatenate(targets),
            predictions=np.concatenate(scores),
        ),
        AlphaValidationRobustnessEvidence(
            candidate_id=candidate_id,
            session_rank_ics=tuple(rank_ics),
            session_spreads=tuple(spreads),
        ),
    )


def _historical_mean_benchmark(
    *,
    program: AlphaResearchProgram,
    fold_plan: AlphaFoldArrayPlan,
    array_workspace: AlphaArrayWorkspaceLike | None = None,
) -> tuple[AlphaCandidateDevelopmentReport, AlphaValidationRows]:
    metric_evidence: list[AlphaFoldMetricEvidence] = []
    keys: list[tuple[int, date, str]] = []
    targets: list[FloatArray] = []
    predictions: list[FloatArray] = []
    fold_hashes: list[str] = []
    for fold_index in range(fold_plan.fold_count):
        fold = (
            array_workspace.load_fold(fold_index)
            if array_workspace is not None
            else load_alpha_fold_arrays(fold_plan, fold_index)
        )
        admitted_training = fold.training_outcome_complete & np.isfinite(fold.training_targets)
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for listing_id, target, admitted in zip(
            fold.training_listing_ids,
            fold.training_targets,
            admitted_training,
            strict=True,
        ):
            if not bool(admitted):
                continue
            sums[listing_id] = sums.get(listing_id, 0.0) + float(target)
            counts[listing_id] = counts.get(listing_id, 0) + 1
        scores: FloatArray = np.asarray(
            [
                sums[listing_id] / counts[listing_id] if listing_id in counts else np.nan
                for listing_id in fold.validation_listing_ids
            ],
            dtype=np.float64,
        )
        scores.setflags(write=False)
        metric = compute_fold_metric_evidence(
            FoldScoreVector(fold=fold, scores=scores), policy=_metric_policy()
        )
        metric_evidence.append(metric)
        admitted = (
            fold.validation_feature_complete
            & fold.validation_outcome_complete
            & np.isfinite(fold.validation_targets)
            & np.isfinite(scores)
        )
        indices = np.flatnonzero(admitted)
        keys.extend(
            (fold_index, fold.validation_row_sessions[index], fold.validation_listing_ids[index])
            for index in indices
        )
        targets.append(fold.validation_targets[indices])
        predictions.append(scores[indices])
        fold_hashes.append(
            canonical_hash(
                {
                    "candidate_id": "benchmark.historical-mean",
                    "fold_commitment_hash": fold.commitment.commitment_hash,
                }
            )
        )
    metrics = aggregate_candidate_metrics("benchmark.historical-mean", tuple(metric_evidence))
    report = seal_current_contract(
        AlphaCandidateDevelopmentReport,
        {
            "kind": "AlphaCandidateDevelopmentReport",
            "request_hash": program.program_hash,
            "development_surface_hash": canonical_hash(
                {
                    "program_hash": program.program_hash,
                    "owner": "historical_mean_benchmark",
                    "fold_hashes": tuple(fold_hashes),
                }
            ),
            "foundation_hash": program.foundation_hash,
            "candidate_id": "benchmark.historical-mean",
            "candidate_card_hash": canonical_hash("benchmark.historical-mean"),
            "role": AlphaCandidateRole.BENCHMARK,
            "status": NumericalCandidateStatus.SUCCEEDED,
            "admitted_fold_count": fold_plan.fold_count,
            "fold_evidence_hashes": tuple(fold_hashes),
            "metrics": metrics,
            "estimator_state_hashes": (),
            "failure_codes": (),
        },
        "report_hash",
    )
    target_values = np.concatenate(targets)
    prediction_values = np.concatenate(predictions)
    target_values.setflags(write=False)
    prediction_values.setflags(write=False)
    return report, AlphaValidationRows(
        candidate_id="benchmark.historical-mean",
        keys=tuple(keys),
        targets=target_values,
        predictions=prediction_values,
    )


def _oos_evidence(
    *,
    program: AlphaResearchProgram,
    attempted_ids: tuple[str, ...],
    result_by_candidate: Mapping[
        str, tuple[AlphaExperimentBatchResult, AlphaExperimentCandidateResult]
    ],
    robustness: Mapping[str, AlphaValidationRobustnessEvidence],
) -> AlphaOosEvidenceAssessment:
    staged: list[dict[str, object]] = []
    family: list[tuple[str, float]] = []
    for candidate_id in attempted_ids:
        candidate = result_by_candidate[candidate_id][1]
        evidence = robustness[candidate_id]
        complete = (
            len(evidence.session_rank_ics) > 1
            and len(evidence.session_spreads) > 1
            and candidate.fold_coverage_mean is not None
            and candidate.fold_coverage_mean >= 0.95
        )
        ic_test = (
            newey_west_mean_test(np.asarray(evidence.session_rank_ics, dtype=np.float64))
            if evidence.session_rank_ics
            else None
        )
        spread_test = (
            newey_west_mean_test(np.asarray(evidence.session_spreads, dtype=np.float64))
            if evidence.session_spreads
            else None
        )
        raw_p_value = (
            max(ic_test.p_value, spread_test.p_value)
            if ic_test is not None and spread_test is not None
            else 1.0
        )
        family.append((candidate_id, raw_p_value))
        staged.append(
            {
                "candidate_id": candidate_id,
                "complete": complete,
                "mean_rank_ic": ic_test.mean if ic_test is not None else None,
                "mean_gross_decile_spread": (spread_test.mean if spread_test is not None else None),
                "rank_ic_raw_p_value": ic_test.p_value if ic_test is not None else None,
                "spread_raw_p_value": (spread_test.p_value if spread_test is not None else None),
            }
        )
    adjusted = holm_adjust(tuple(family))
    records: list[AlphaOosEvidenceRecord] = []
    for values in staged:
        candidate_id = str(values["candidate_id"])
        mean_ic = cast(float | None, values["mean_rank_ic"])
        mean_spread = cast(float | None, values["mean_gross_decile_spread"])
        adjusted_p = adjusted[candidate_id]
        failure_codes: tuple[str, ...]
        if not bool(values["complete"]) or mean_ic is None or mean_spread is None:
            classification = AlphaOosEvidenceClassification.INSUFFICIENT_EVIDENCE
            reason_codes = ("COMMON_OOS_EVIDENCE_INCOMPLETE",)
            failure_codes = ("alpha_research.oos_evidence_insufficient",)
        elif mean_ic > 0.0 and mean_spread > 0.0:
            classification = AlphaOosEvidenceClassification.POSITIVE_OOS_EVIDENCE
            reason_codes = (
                "DIRECTIONALLY_POSITIVE_HOLM_CONFIRMED"
                if adjusted_p <= 0.05
                else "DIRECTIONALLY_POSITIVE_NOT_HOLM_CONFIRMED",
            )
            failure_codes = ()
        elif mean_ic < 0.0 and mean_spread < 0.0 and adjusted_p <= 0.05:
            classification = AlphaOosEvidenceClassification.NEGATIVE_OOS_EVIDENCE
            reason_codes = ("DIRECTIONALLY_NEGATIVE_HOLM_CONFIRMED",)
            failure_codes = ()
        elif mean_ic <= 0.0 and mean_spread <= 0.0:
            classification = AlphaOosEvidenceClassification.NO_DETECTABLE_EFFECT
            reason_codes = ("NONPOSITIVE_DIRECTION_NOT_HOLM_CONFIRMED",)
            failure_codes = ()
        else:
            classification = AlphaOosEvidenceClassification.MIXED_OOS_EVIDENCE
            reason_codes = ("MIXED_OOS_DIRECTION",)
            failure_codes = ()
        records.append(
            seal_contract(
                AlphaOosEvidenceRecord,
                {
                    "candidate_id": candidate_id,
                    "mean_rank_ic": mean_ic,
                    "mean_gross_decile_spread": mean_spread,
                    "rank_ic_raw_p_value": values["rank_ic_raw_p_value"],
                    "spread_raw_p_value": values["spread_raw_p_value"],
                    "holm_adjusted_p_value": adjusted_p,
                    "classification": classification,
                    "reason_codes": reason_codes,
                    "failure_codes": failure_codes,
                },
                "evidence_hash",
            )
        )
    return seal_contract(
        AlphaOosEvidenceAssessment,
        {
            "program_hash": program.program_hash,
            "records": tuple(records),
            "positive_candidate_ids": tuple(
                value.candidate_id
                for value in records
                if value.classification is AlphaOosEvidenceClassification.POSITIVE_OOS_EVIDENCE
            ),
            "mixed_candidate_ids": tuple(
                value.candidate_id
                for value in records
                if value.classification is AlphaOosEvidenceClassification.MIXED_OOS_EVIDENCE
            ),
            "hypothesis_count": len(records),
        },
        "assessment_hash",
    )


def _qualify_target_lane_candidates(
    *,
    program: AlphaResearchProgram,
    registry: AlphaCandidateRegistrySnapshot,
    batch_results: tuple[AlphaExperimentBatchResult, ...],
    nominated_candidate_ids: tuple[str, ...],
    fold_plans: Mapping[AlphaTargetLane, AlphaFoldArrayPlan],
    train_session_count: int,
    store: AlphaCurrentArtifactStore,
    array_workspaces: Mapping[AlphaTargetLane, AlphaArrayWorkspaceLike],
    development_state_loader: Callable[[str], AlphaDevelopmentEstimatorState] | None,
    model_catalog: AlphaModelCatalog | None = None,
) -> AlphaQualificationResult:
    attempted_ids = tuple(value.candidate_id for value in registry.candidates)
    if len(set(nominated_candidate_ids)) != len(nominated_candidate_ids) or not set(
        nominated_candidate_ids
    ) <= set(attempted_ids):
        raise ValueError("alpha_research.qualification_nomination_invalid")
    result_by_candidate = _index_attempted_family(
        attempted_ids=attempted_ids,
        batch_results=batch_results,
    )
    robustness: dict[str, AlphaValidationRobustnessEvidence] = {}
    for candidate_id in attempted_ids:
        batch_result, candidate = result_by_candidate[candidate_id]
        _, candidate_robustness = _candidate_rows(
            candidate_id=candidate_id,
            result=batch_result,
            candidate_result_hashes=candidate.numerical_result_hashes,
            fold_surface_hashes=getattr(candidate, "fold_surface_hashes", None),
            store=store,
        )
        robustness[candidate_id] = candidate_robustness
    evidence = _oos_evidence(
        program=program,
        attempted_ids=attempted_ids,
        result_by_candidate=result_by_candidate,
        robustness=robustness,
    )
    evidence_by_id = {value.candidate_id: value for value in evidence.records}
    current_arrays: dict[AlphaTargetLane, AlphaCurrentRefitArrays] = {}
    fit_calls = predict_calls = 0
    updated: list[AlphaCandidateRecord] = []
    load_state = development_state_loader or store.load_development_estimator_state
    for record in registry.candidates:
        candidate = result_by_candidate[record.candidate_id][1]
        item = evidence_by_id[record.candidate_id]
        common: _OosRecordFields = {
            "development_result_ref": candidate.result_hash,
            "development_candidate_hash": item.evidence_hash,
            "holm_adjusted_p_value": item.holm_adjusted_p_value,
            "oos_evidence_classification": item.classification,
            "mean_rank_ic": item.mean_rank_ic,
            "mean_gross_decile_spread": item.mean_gross_decile_spread,
        }
        if item.classification is not AlphaOosEvidenceClassification.POSITIVE_OOS_EVIDENCE:
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.DEVELOPMENT_REJECTED,
                    failure_codes=("alpha_research.oos_evidence_not_positive",),
                    **common,
                )
            )
            continue
        if (
            record.status is AlphaCandidateStatus.CURRENT_QUALIFIED
            and record.current_state_hash is not None
            and record.stability_assessment_hash is not None
            and record.current_score_child_hash is not None
        ):
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.CURRENT_QUALIFIED,
                    current_state_hash=record.current_state_hash,
                    stability_assessment_hash=record.stability_assessment_hash,
                    current_score_child_hash=record.current_score_child_hash,
                    stability_failed_check_ids=record.stability_failed_check_ids,
                    current_score_mean=record.current_score_mean,
                    current_score_std=record.current_score_std,
                    current_score_coverage=record.current_score_coverage,
                    **common,
                )
            )
            continue
        if record.candidate_id not in nominated_candidate_ids:
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.DEVELOPMENT_ADMISSIBLE,
                    **common,
                )
            )
            continue
        if not _current_evaluator_available(record.spec, model_catalog):
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.CURRENT_STABILITY_REJECTED,
                    failure_codes=("SCIENTIFIC_NON_ADMISSION",),
                    **common,
                )
            )
            continue
        lane = record.spec.target_lane
        if lane is None or lane not in fold_plans:
            raise ValueError("alpha_research.target_lane_plan_unavailable")
        arrays = current_arrays.get(lane)
        if arrays is None:
            workspace = array_workspaces.get(lane)
            arrays = (
                workspace.prepare_current_refit()
                if workspace is not None
                else prepare_alpha_current_refit_arrays(
                    fold_plans[lane], train_session_count=train_session_count
                )
            )
            current_arrays[lane] = arrays
        development_states = tuple(load_state(value) for value in candidate.estimator_state_hashes)
        assert_alpha_training_authority(
            arrays.training_input_binding,
            scope="CURRENT_REFIT",
            foundation_hash=fold_plans[lane].foundation.foundation_hash,
            feature_panel_snapshot_hash=(fold_plans[lane].foundation.feature_panel_snapshot_hash),
            causal_outcome_snapshot_hash=(
                fold_plans[lane].foundation.execution_outcome.snapshot_hash
            ),
            target_policy=fold_plans[lane].target_policy,
            ordered_feature_ids=(
                *fold_plans[lane].base_feature_ids,
                *fold_plans[lane].additional_factor_ids,
            ),
            feature_context_hash=fold_plans[lane].feature_context_hash,
            training_cutoff=arrays.training_cutoff,
            outcome_maturity_session=arrays.formation_session,
            prediction_anchor=arrays.formation_session,
            fold_commitment_hash=None,
        )
        refit = _current_refit(
            record.spec,
            catalog=model_catalog,
            request_hash=program.program_hash,
            candidate_id=record.candidate_id,
            arrays=arrays,
            development_states=development_states,
            package_identity_hash=program.package_identity_hash,
            retain_diagnostic_score=True,
        )
        store.publish_estimator_state(refit.estimator_state)
        store.publish_refit_assessment(refit.assessment)
        if (
            refit.fit_operation_hash is not None
            and refit.estimator_content is not None
            and refit.fit_provenance is not None
        ):
            store.publish_model_fit_sidecar(
                operation_binding_hash=refit.fit_operation_hash,
                estimator_content=refit.estimator_content,
                fit_provenance=refit.fit_provenance,
            )
        fit_calls += refit.fit_calls
        predict_calls += refit.predict_calls
        hard_quality_passed = (
            refit.score_table is not None
            and refit.estimator_state.validation_score_coverage >= 0.95
            and refit.estimator_state.validation_score_std > 0.001
        )
        child_hash = None
        if hard_quality_passed:
            assert refit.score_table is not None
            child = build_current_candidate_score_child(
                request_hash=program.program_hash,
                foundation_hash=program.foundation_hash,
                logical_panel_hash=program.logical_panel_hash,
                candidate_id=record.candidate_id,
                estimator_state=refit.estimator_state,
                assessment=refit.assessment,
                formation_session=arrays.formation_session,
                training_cutoff=arrays.training_cutoff,
                ordered_listing_ids=fold_plans[lane].ordered_listing_ids,
                table=refit.score_table,
                store=store,
            )
            store.publish_current_candidate_score(child)
            store.load_current_candidate_score(child.child_hash)
            child_hash = child.child_hash
        updated.append(
            _updated_record(
                record,
                status=(
                    AlphaCandidateStatus.CURRENT_QUALIFIED
                    if hard_quality_passed
                    else AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
                ),
                current_state_hash=refit.estimator_state.state_hash,
                stability_assessment_hash=refit.assessment.assessment_hash,
                current_score_child_hash=child_hash,
                stability_failed_check_ids=tuple(
                    value.check_id for value in refit.assessment.checks if not value.passed
                ),
                current_score_mean=refit.estimator_state.validation_score_mean,
                current_score_std=refit.estimator_state.validation_score_std,
                current_score_coverage=refit.estimator_state.validation_score_coverage,
                failure_codes=(
                    ()
                    if hard_quality_passed
                    else ("alpha_research.current_score_quality_insufficient",)
                ),
                **common,
            )
        )
    hypothesis_hash = canonical_hash(
        {
            "attempted_candidate_ids": attempted_ids,
            "oos_evidence_assessment_hash": evidence.assessment_hash,
        }
    )
    revised_registry = update_registry_candidates(
        registry, tuple(updated), hypothesis_family_hash=hypothesis_hash
    )
    qualification = seal_contract(
        AlphaQualificationSnapshot,
        {
            "program_hash": program.program_hash,
            "registry_hash": revised_registry.registry_hash,
            "hypothesis_family_hash": hypothesis_hash,
            "attempted_candidate_ids": attempted_ids,
            "nominated_candidate_ids": nominated_candidate_ids,
            "development_admissible_ids": evidence.positive_candidate_ids,
            "current_qualified_ids": tuple(
                value.candidate_id
                for value in revised_registry.candidates
                if value.status is AlphaCandidateStatus.CURRENT_QUALIFIED
            ),
            "current_stability_rejected_ids": tuple(
                value.candidate_id
                for value in revised_registry.candidates
                if value.status is AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
            ),
            "viability_assessment_hash": None,
            "oos_evidence_assessment_hash": evidence.assessment_hash,
            "score_correlations": _score_correlations(revised_registry, store),
        },
        "qualification_hash",
    )
    return AlphaQualificationResult(
        registry=revised_registry,
        qualification=qualification,
        fit_call_count=fit_calls,
        predict_call_count=predict_calls,
        metric_call_count=len(evidence.records),
        oos_evidence=evidence,
    )


def qualify_alpha_model_candidates(
    *,
    program: AlphaResearchProgram,
    registry: AlphaCandidateRegistrySnapshot,
    batch_results: tuple[AlphaExperimentBatchResult, ...],
    nominated_candidate_ids: tuple[str, ...],
    fold_plan: AlphaFoldArrayPlan | None,
    train_session_count: int,
    store: AlphaCurrentArtifactStore,
    array_workspace: AlphaArrayWorkspaceLike | None = None,
    development_state_loader: Callable[[str], AlphaDevelopmentEstimatorState] | None = None,
    fold_plans: Mapping[AlphaTargetLane, AlphaFoldArrayPlan] | None = None,
    array_workspaces: Mapping[AlphaTargetLane, AlphaArrayWorkspaceLike] | None = None,
    model_catalog: AlphaModelCatalog,
    model_mandate: AlphaResearchModelMandate,
) -> AlphaQualificationResult:
    """Apply the complete attempted-recipe family and current stability without Agent math."""
    if (
        program.model_mandate_hash != model_mandate.mandate_hash
        or program.model_catalog_hash != model_mandate.catalog_binding.catalog_hash
        # The admitted models as installed; a model beside them refuses nothing.
        or not admitted_capabilities_installed(model_mandate, model_catalog)
    ):
        raise ValueError("alpha_research.qualification_model_authority_mismatch")
    domains = {value.search_domain_hash: value for value in model_mandate.ordered_search_domains}
    targeted = tuple(
        value.spec.target_lane
        for value in registry.candidates
        if isinstance(value.spec, AlphaResearchModelRecipe)
    )
    used_lanes = {value for value in targeted if value is not None}
    target_plans = fold_plans or {}
    assert_research_recipe_target_authority(
        program_target_policy_hashes=program.target_policy_hashes,
        recipe_target_lanes=targeted,
        default_target_policy=getattr(fold_plan, "target_policy", None),
        lane_target_policies={
            lane: getattr(target_plans[lane], "target_policy", None)
            for lane in used_lanes
            if lane in target_plans
        },
    )
    for record in registry.candidates:
        if not isinstance(record.spec, AlphaResearchModelRecipe):
            raise ValueError("alpha_research.legacy_model_spec_not_active")
        try:
            domain = domains[record.spec.search_domain_hash]
        except KeyError as error:
            raise ValueError("alpha_research.recipe_domain_not_mandated") from error
        model_catalog.admit_recipe(recipe=record.spec.recipe, domain=domain)

    if program.target_policy_hashes is not None:
        if fold_plans is None:
            raise ValueError("alpha_research.target_lane_plan_unavailable")
        return _qualify_target_lane_candidates(
            program=program,
            registry=registry,
            batch_results=batch_results,
            nominated_candidate_ids=nominated_candidate_ids,
            fold_plans=fold_plans,
            train_session_count=train_session_count,
            store=store,
            array_workspaces=array_workspaces or {},
            development_state_loader=development_state_loader,
            model_catalog=model_catalog,
        )
    # A lane-free Program's candidates train on its one fold plan; a lane Program's on each lane's
    # own, so a single-lane question has no lane-free plan to give (GR3).
    if fold_plan is None:
        raise ValueError("alpha_research.qualification_fold_plan_required")

    attempted_ids = tuple(value.candidate_id for value in registry.candidates)
    if len(set(nominated_candidate_ids)) != len(nominated_candidate_ids) or not set(
        nominated_candidate_ids
    ) <= set(attempted_ids):
        raise ValueError("alpha_research.qualification_nomination_invalid")
    result_by_candidate = _index_attempted_family(
        attempted_ids=attempted_ids,
        batch_results=batch_results,
    )
    reports: list[AlphaCandidateDevelopmentReport] = []
    rows: dict[str, AlphaValidationRows] = {}
    robustness: dict[str, AlphaValidationRobustnessEvidence] = {}
    state_counts: dict[str, int] = {}
    for candidate_id in attempted_ids:
        batch_result, candidate = result_by_candidate[candidate_id]
        report = store.load_candidate_report(candidate.development_report_hash)
        candidate_rows, candidate_robustness = _candidate_rows(
            candidate_id=candidate_id,
            result=batch_result,
            candidate_result_hashes=candidate.numerical_result_hashes,
            fold_surface_hashes=getattr(candidate, "fold_surface_hashes", None),
            store=store,
        )
        reports.append(report)
        rows[candidate_id] = candidate_rows
        robustness[candidate_id] = candidate_robustness
        state_counts[candidate_id] = len(candidate.estimator_state_hashes)
    benchmark_report, benchmark_rows = _historical_mean_benchmark(
        program=program,
        fold_plan=fold_plan,
        array_workspace=array_workspace,
    )
    viability = assess_alpha_model_viability(
        request_hash=program.program_hash,
        surface_hash=canonical_hash(
            {
                "program_hash": program.program_hash,
                "batch_result_hashes": tuple(value.result_hash for value in batch_results),
                "attempted_candidate_ids": attempted_ids,
            }
        ),
        candidate_results=(benchmark_report, *reports),
        validation_rows={"benchmark.historical-mean": benchmark_rows, **rows},
        robustness_evidence=robustness,
        admitted_fold_count=fold_plan.fold_count,
        estimator_state_counts=state_counts,
    )
    store.publish_viability(viability)
    viability_by_id = {value.candidate_id: value for value in viability.candidates}
    arrays = None
    fit_calls = predict_calls = 0
    updated: list[AlphaCandidateRecord] = []
    for record in registry.candidates:
        candidate_viability = viability_by_id[record.candidate_id]
        if not candidate_viability.admitted:
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.DEVELOPMENT_REJECTED,
                    development_result_ref=result_by_candidate[record.candidate_id][1].result_hash,
                    development_candidate_hash=candidate_viability.candidate_hash,
                    holm_adjusted_p_value=candidate_viability.holm_adjusted_p_value,
                    failure_codes=candidate_viability.failure_codes,
                )
            )
            continue
        if (
            record.status is AlphaCandidateStatus.CURRENT_QUALIFIED
            and record.current_state_hash is not None
            and record.stability_assessment_hash is not None
            and record.current_score_child_hash is not None
        ):
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.CURRENT_QUALIFIED,
                    development_result_ref=result_by_candidate[record.candidate_id][1].result_hash,
                    development_candidate_hash=candidate_viability.candidate_hash,
                    current_state_hash=record.current_state_hash,
                    stability_assessment_hash=record.stability_assessment_hash,
                    current_score_child_hash=record.current_score_child_hash,
                    holm_adjusted_p_value=candidate_viability.holm_adjusted_p_value,
                    stability_failed_check_ids=record.stability_failed_check_ids,
                    current_score_mean=record.current_score_mean,
                    current_score_std=record.current_score_std,
                    current_score_coverage=record.current_score_coverage,
                )
            )
            continue
        carried_rejection = _carry_forward_prior_current_rejection(
            record,
            development_result_ref=result_by_candidate[record.candidate_id][1].result_hash,
            development_candidate_hash=candidate_viability.candidate_hash,
            holm_adjusted_p_value=candidate_viability.holm_adjusted_p_value,
        )
        if carried_rejection is not None:
            updated.append(carried_rejection)
            continue
        if record.candidate_id not in nominated_candidate_ids:
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.DEVELOPMENT_ADMISSIBLE,
                    development_result_ref=result_by_candidate[record.candidate_id][1].result_hash,
                    development_candidate_hash=candidate_viability.candidate_hash,
                    holm_adjusted_p_value=candidate_viability.holm_adjusted_p_value,
                )
            )
            continue
        if not _current_evaluator_available(record.spec, model_catalog):
            updated.append(
                _updated_record(
                    record,
                    status=AlphaCandidateStatus.CURRENT_STABILITY_REJECTED,
                    development_result_ref=(
                        result_by_candidate[record.candidate_id][1].result_hash
                    ),
                    development_candidate_hash=candidate_viability.candidate_hash,
                    holm_adjusted_p_value=candidate_viability.holm_adjusted_p_value,
                    failure_codes=("SCIENTIFIC_NON_ADMISSION",),
                )
            )
            continue
        if arrays is None:
            arrays = (
                array_workspace.prepare_current_refit()
                if array_workspace is not None
                else prepare_alpha_current_refit_arrays(
                    fold_plan, train_session_count=train_session_count
                )
            )
        candidate = result_by_candidate[record.candidate_id][1]
        load_state = development_state_loader or store.load_development_estimator_state
        development_states = tuple(load_state(value) for value in candidate.estimator_state_hashes)
        assert_alpha_training_authority(
            arrays.training_input_binding,
            scope="CURRENT_REFIT",
            foundation_hash=fold_plan.foundation.foundation_hash,
            feature_panel_snapshot_hash=fold_plan.foundation.feature_panel_snapshot_hash,
            causal_outcome_snapshot_hash=fold_plan.foundation.execution_outcome.snapshot_hash,
            target_policy=fold_plan.target_policy,
            ordered_feature_ids=(
                *fold_plan.base_feature_ids,
                *fold_plan.additional_factor_ids,
            ),
            feature_context_hash=fold_plan.feature_context_hash,
            training_cutoff=arrays.training_cutoff,
            outcome_maturity_session=arrays.formation_session,
            prediction_anchor=arrays.formation_session,
            fold_commitment_hash=None,
        )
        refit = _current_refit(
            record.spec,
            catalog=model_catalog,
            request_hash=program.program_hash,
            candidate_id=record.candidate_id,
            arrays=arrays,
            development_states=development_states,
            package_identity_hash=program.package_identity_hash,
        )
        store.publish_estimator_state(refit.estimator_state)
        store.publish_refit_assessment(refit.assessment)
        fit_calls += refit.fit_calls
        predict_calls += refit.predict_calls
        child_hash = None
        if refit.assessment.passed:
            if refit.score_table is None:
                raise ValueError("alpha_research.current_score_readback_failed")
            child = build_current_candidate_score_child(
                request_hash=program.program_hash,
                foundation_hash=program.foundation_hash,
                logical_panel_hash=program.logical_panel_hash,
                candidate_id=record.candidate_id,
                estimator_state=refit.estimator_state,
                assessment=refit.assessment,
                formation_session=arrays.formation_session,
                training_cutoff=arrays.training_cutoff,
                ordered_listing_ids=fold_plan.ordered_listing_ids,
                table=refit.score_table,
                store=store,
            )
            store.publish_current_candidate_score(child)
            store.load_current_candidate_score(child.child_hash)
            child_hash = child.child_hash
        updated.append(
            _updated_record(
                record,
                status=(
                    AlphaCandidateStatus.CURRENT_QUALIFIED
                    if refit.assessment.passed
                    else AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
                ),
                development_result_ref=candidate.result_hash,
                development_candidate_hash=candidate_viability.candidate_hash,
                current_state_hash=refit.estimator_state.state_hash,
                stability_assessment_hash=refit.assessment.assessment_hash,
                current_score_child_hash=child_hash,
                holm_adjusted_p_value=candidate_viability.holm_adjusted_p_value,
                stability_failed_check_ids=tuple(
                    value.check_id for value in refit.assessment.checks if not value.passed
                ),
                current_score_mean=refit.estimator_state.validation_score_mean,
                current_score_std=refit.estimator_state.validation_score_std,
                current_score_coverage=refit.estimator_state.validation_score_coverage,
                failure_codes=(
                    ()
                    if refit.assessment.passed
                    else ("alpha_research.current_refit_insufficient",)
                ),
            )
        )
    hypothesis_hash = canonical_hash(
        {
            "attempted_candidate_ids": attempted_ids,
            "viability_assessment_hash": viability.assessment_hash,
        }
    )
    revised_registry = update_registry_candidates(
        registry, tuple(updated), hypothesis_family_hash=hypothesis_hash
    )
    score_correlations = _score_correlations(revised_registry, store)
    qualification = seal_contract(
        AlphaQualificationSnapshot,
        {
            "program_hash": program.program_hash,
            "registry_hash": revised_registry.registry_hash,
            "hypothesis_family_hash": hypothesis_hash,
            "attempted_candidate_ids": attempted_ids,
            "nominated_candidate_ids": nominated_candidate_ids,
            "development_admissible_ids": viability.admissible_candidate_ids,
            "current_qualified_ids": tuple(
                value.candidate_id
                for value in revised_registry.candidates
                if value.status is AlphaCandidateStatus.CURRENT_QUALIFIED
            ),
            "current_stability_rejected_ids": tuple(
                value.candidate_id
                for value in revised_registry.candidates
                if value.status is AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
            ),
            "viability_assessment_hash": viability.assessment_hash,
            "score_correlations": score_correlations,
        },
        "qualification_hash",
    )
    return AlphaQualificationResult(
        registry=revised_registry,
        qualification=qualification,
        fit_call_count=fit_calls,
        predict_call_count=predict_calls,
        metric_call_count=len(viability.candidates),
    )


def _index_attempted_family(
    *,
    attempted_ids: tuple[str, ...],
    batch_results: tuple[AlphaExperimentBatchResult, ...],
) -> dict[str, tuple[AlphaExperimentBatchResult, AlphaExperimentCandidateResult]]:
    """Validate a complete family without treating batch readback order as identity."""

    ordered_results = tuple(
        (candidate.candidate_id, batch, candidate)
        for batch in batch_results
        for candidate in batch.candidates
    )
    result_by_candidate = {
        candidate_id: (batch, candidate) for candidate_id, batch, candidate in ordered_results
    }
    if (
        len(ordered_results) != len(attempted_ids)
        or len(result_by_candidate) != len(ordered_results)
        or set(result_by_candidate) != set(attempted_ids)
    ):
        raise ValueError("alpha_research.qualification_attempted_family_incomplete")
    return result_by_candidate


def _carry_forward_prior_current_rejection(
    record: AlphaCandidateRecord,
    *,
    development_result_ref: str,
    development_candidate_hash: str,
    holm_adjusted_p_value: float | None,
) -> AlphaCandidateRecord | None:
    """Preserve a verified current rejection when a later Holm family still admits it."""

    if (
        record.status is not AlphaCandidateStatus.CURRENT_STABILITY_REJECTED
        or record.current_state_hash is None
        or record.stability_assessment_hash is None
        or "alpha_research.current_refit_insufficient" not in record.failure_codes
    ):
        return None
    return _updated_record(
        record,
        status=AlphaCandidateStatus.CURRENT_STABILITY_REJECTED,
        development_result_ref=development_result_ref,
        development_candidate_hash=development_candidate_hash,
        current_state_hash=record.current_state_hash,
        stability_assessment_hash=record.stability_assessment_hash,
        current_score_child_hash=record.current_score_child_hash,
        holm_adjusted_p_value=holm_adjusted_p_value,
        stability_failed_check_ids=record.stability_failed_check_ids,
        current_score_mean=record.current_score_mean,
        current_score_std=record.current_score_std,
        current_score_coverage=record.current_score_coverage,
        failure_codes=record.failure_codes,
    )


def _score_correlations(
    registry: AlphaCandidateRegistrySnapshot,
    store: AlphaCurrentArtifactStore,
) -> tuple[AlphaCandidateScoreCorrelation, ...]:
    qualified = tuple(
        value
        for value in registry.candidates
        if value.status is AlphaCandidateStatus.CURRENT_QUALIFIED
        and value.current_score_child_hash is not None
    )
    score_vectors: dict[str, FloatArray] = {}
    for record in qualified:
        score_child_hash = record.current_score_child_hash
        if score_child_hash is None:
            raise ValueError("alpha_research.qualified_candidate_score_missing")
        child = store.load_current_candidate_score(score_child_hash)
        table = store.resolve_current_candidate_score_chunk(child.chunk)
        scores: FloatArray = np.asarray(
            [float(value) if value is not None else np.nan for value in table["score"].to_pylist()],
            dtype=np.float64,
        )
        scores.setflags(write=False)
        score_vectors[record.candidate_id] = scores
    result: list[AlphaCandidateScoreCorrelation] = []
    for left, right in combinations(sorted(score_vectors), 2):
        left_scores = score_vectors[left]
        right_scores = score_vectors[right]
        admitted = np.isfinite(left_scores) & np.isfinite(right_scores)
        common = int(admitted.sum())
        correlation = None
        if (
            common >= 2
            and float(np.std(left_scores[admitted])) > 0.0
            and float(np.std(right_scores[admitted])) > 0.0
        ):
            correlation = float(np.corrcoef(left_scores[admitted], right_scores[admitted])[0, 1])
        result.append(
            seal_contract(
                AlphaCandidateScoreCorrelation,
                {
                    "left_candidate_id": left,
                    "right_candidate_id": right,
                    "common_scored_count": common,
                    "correlation": correlation,
                },
                "correlation_hash",
            )
        )
    return tuple(result)


def _updated_record(
    record: AlphaCandidateRecord,
    *,
    status: AlphaCandidateStatus,
    development_result_ref: str,
    development_candidate_hash: str,
    current_state_hash: str | None = None,
    stability_assessment_hash: str | None = None,
    current_score_child_hash: str | None = None,
    holm_adjusted_p_value: float | None = None,
    oos_evidence_classification: AlphaOosEvidenceClassification | None = None,
    mean_rank_ic: float | None = None,
    mean_gross_decile_spread: float | None = None,
    stability_failed_check_ids: tuple[str, ...] = (),
    current_score_mean: float | None = None,
    current_score_std: float | None = None,
    current_score_coverage: float | None = None,
    failure_codes: tuple[str, ...] = (),
) -> AlphaCandidateRecord:
    return seal_contract(
        AlphaCandidateRecord,
        {
            "spec": record.spec,
            "candidate_id": record.candidate_id,
            "batch_hash": record.batch_hash,
            "status": status,
            "development_result_ref": development_result_ref,
            "development_candidate_hash": development_candidate_hash,
            "current_state_hash": current_state_hash,
            "stability_assessment_hash": stability_assessment_hash,
            "current_score_child_hash": current_score_child_hash,
            "holm_adjusted_p_value": holm_adjusted_p_value,
            "oos_evidence_classification": (
                oos_evidence_classification or record.oos_evidence_classification
            ),
            "mean_rank_ic": mean_rank_ic if mean_rank_ic is not None else record.mean_rank_ic,
            "mean_gross_decile_spread": (
                mean_gross_decile_spread
                if mean_gross_decile_spread is not None
                else record.mean_gross_decile_spread
            ),
            "stability_failed_check_ids": stability_failed_check_ids,
            "current_score_mean": current_score_mean,
            "current_score_std": current_score_std,
            "current_score_coverage": current_score_coverage,
            "failure_codes": failure_codes,
        },
        "record_hash",
    )


def _metric_policy() -> AlphaMetricPolicy:
    from ..experiments.policies import load_alpha_metric_policy

    return load_alpha_metric_policy()


__all__ = ["AlphaQualificationResult", "qualify_alpha_model_candidates"]
