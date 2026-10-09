"""Current-runtime viability, multiplicity, and refit stability tests."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from alphalattice.investment.alpha_research.evaluation.metrics import (
    FoldScoreVector,
    compute_candidate_metrics,
)
from alphalattice.investment.alpha_research.evaluation.viability import (
    AlphaValidationRobustnessEvidence,
    AlphaValidationRows,
    assess_alpha_model_viability,
)
from alphalattice.investment.alpha_research.experiments.contracts import AlphaCandidateStatus
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaEstimatorState,
    AlphaSessionScoreStatistics,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.experiments.policies import load_alpha_metric_policy
from alphalattice.investment.alpha_research.inputs.folds import AlphaCurrentRefitArrays
from alphalattice.investment.alpha_research.scores.refit import fit_and_assess_current_alpha
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.alpha_research.fixtures import (
    registered_test_cards,
    synthetic_prepared_arrays,
    synthetic_request,
)


def _candidate_results():
    request, _inventory = synthetic_request()
    prepared = synthetic_prepared_arrays()
    results = []
    for card in registered_test_cards()[:4]:
        fold_scores = []
        for fold in prepared.folds:
            values = (
                np.zeros(len(fold.validation_targets), dtype=np.float64)
                if card.role.value == "BENCHMARK"
                else fold.validation_targets.copy()
            )
            values.setflags(write=False)
            fold_scores.append(FoldScoreVector(fold=fold, scores=values))
        folds = tuple(fold_scores)
        metrics = compute_candidate_metrics(
            card.candidate_id,
            folds,
            policy=load_alpha_metric_policy(),
        ).metrics
        results.append(
            seal_current_contract(
                AlphaCandidateDevelopmentReport,
                {
                    "kind": "AlphaCandidateDevelopmentReport",
                    "request_hash": request.request_hash,
                    "development_surface_hash": "a" * 64,
                    "foundation_hash": request.foundation_hash,
                    "candidate_id": card.candidate_id,
                    "candidate_card_hash": card.card_hash,
                    "role": card.role,
                    "status": AlphaCandidateStatus.SUCCEEDED,
                    "admitted_fold_count": len(prepared.folds),
                    "fold_evidence_hashes": tuple(
                        canonical_hash((card.candidate_id, index))
                        for index in range(len(prepared.folds))
                    ),
                    "metrics": metrics,
                    "estimator_state_hashes": tuple(
                        canonical_hash((card.candidate_id, "state", index))
                        for index in range(len(prepared.folds))
                    ),
                    "failure_codes": (),
                },
                "report_hash",
            )
        )
    return request, prepared, tuple(results)


def _rows_for_predictions(candidate_id: str, predictions: np.ndarray) -> AlphaValidationRows:
    sessions = tuple(date(2025, 1, 1) + timedelta(days=value) for value in range(6))
    keys = tuple(
        (session_index // 2, session, f"listing-{listing:03d}")
        for session_index, session in enumerate(sessions)
        for listing in range(100)
    )
    targets = np.asarray(
        [
            0.01 * session_index + 0.002 * listing
            for session_index in range(6)
            for listing in range(100)
        ],
        dtype=np.float64,
    )
    return AlphaValidationRows(
        candidate_id=candidate_id,
        keys=keys,
        targets=targets,
        predictions=predictions,
    )


def test_viability_uses_directional_dm_and_dynamic_holm_family() -> None:
    request, prepared, results = _candidate_results()
    target = _rows_for_predictions("target", np.zeros(600, dtype=np.float64)).targets
    rows = {
        "benchmark.historical-mean": _rows_for_predictions(
            "benchmark.historical-mean", np.zeros(600, dtype=np.float64)
        ),
        "model.ridge.alpha-1p0": _rows_for_predictions("model.ridge.alpha-1p0", target.copy()),
        "model.ridge.alpha-10p0": _rows_for_predictions("model.ridge.alpha-10p0", target * 0.999),
    }
    assessment = assess_alpha_model_viability(
        request_hash=request.request_hash,
        surface_hash="a" * 64,
        candidate_results=results,
        validation_rows=rows,
        robustness_evidence={
            candidate_id: AlphaValidationRobustnessEvidence(
                candidate_id=candidate_id,
                session_rank_ics=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                session_spreads=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
            )
            for candidate_id in (
                "model.ridge.alpha-1p0",
                "model.ridge.alpha-10p0",
            )
        },
        admitted_fold_count=len(prepared.folds),
        estimator_state_counts={
            "model.ridge.alpha-1p0": len(prepared.folds),
            "model.ridge.alpha-10p0": len(prepared.folds),
        },
    )
    assert assessment.hypothesis_count == 2
    assert assessment.admissible_candidate_ids == (
        "model.ridge.alpha-1p0",
        "model.ridge.alpha-10p0",
    )
    assert all(
        value.dm_statistic is not None and value.dm_statistic < 0 for value in assessment.candidates
    )
    assert all(value.holm_adjusted_p_value is not None for value in assessment.candidates)


def test_a_refused_candidate_names_the_checks_it_failed() -> None:
    """Regression: a significant candidate a positivity check refuses carries that
    check's code, so its qualification record holds the evidence and the qualification seals
    its end instead of stopping on a record without failure evidence."""
    request, prepared, results = _candidate_results()
    target = _rows_for_predictions("target", np.zeros(600, dtype=np.float64)).targets
    rows = {
        "benchmark.historical-mean": _rows_for_predictions(
            "benchmark.historical-mean", np.zeros(600, dtype=np.float64)
        ),
        "model.ridge.alpha-1p0": _rows_for_predictions("model.ridge.alpha-1p0", target.copy()),
        "model.ridge.alpha-10p0": _rows_for_predictions("model.ridge.alpha-10p0", target * 0.999),
    }
    spreads = {
        "model.ridge.alpha-1p0": (0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        "model.ridge.alpha-10p0": (-0.06, -0.05, -0.04, -0.03, -0.02, -0.01),
    }
    assessment = assess_alpha_model_viability(
        request_hash=request.request_hash,
        surface_hash="a" * 64,
        candidate_results=results,
        validation_rows=rows,
        robustness_evidence={
            candidate_id: AlphaValidationRobustnessEvidence(
                candidate_id=candidate_id,
                session_rank_ics=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                session_spreads=values,
            )
            for candidate_id, values in spreads.items()
        },
        admitted_fold_count=len(prepared.folds),
        estimator_state_counts={candidate_id: len(prepared.folds) for candidate_id in spreads},
    )
    refused = {value.candidate_id: value for value in assessment.candidates}
    assert assessment.admissible_candidate_ids == ("model.ridge.alpha-1p0",)
    assert not refused["model.ridge.alpha-10p0"].admitted
    assert refused["model.ridge.alpha-10p0"].failure_codes == (
        "GROSS_SPREAD_NOT_POSITIVE",
        "BOOTSTRAP_LOWER_NOT_POSITIVE",
    )


def _estimator_state(
    *,
    request_hash: str,
    candidate_id: str,
    fold_index: int,
    factor_ids: tuple[str, ...],
    coefficients: np.ndarray,
    intercept: float,
    training_mse: float,
    score_mean: float,
    score_std: float,
    session_score_mean: float | None = None,
    session_score_std: float | None = None,
) -> AlphaEstimatorState:
    return seal_current_contract(
        AlphaEstimatorState,
        {
            "kind": "AlphaEstimatorState",
            "request_hash": request_hash,
            "candidate_id": candidate_id,
            "scope": "DEVELOPMENT_FOLD",
            "fold_index": fold_index,
            "ordered_factor_ids": factor_ids,
            "coefficient_hex": tuple(float(value).hex() for value in coefficients),
            "intercept_hex": float(intercept).hex(),
            "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            "coefficient_max_abs": float(np.max(np.abs(coefficients))),
            "training_mse": training_mse,
            "validation_score_mean": score_mean,
            "validation_score_std": score_std,
            "validation_score_coverage": 1.0,
            "validation_session_statistics": (
                AlphaSessionScoreStatistics(
                    formation_session=date(2024, 1, fold_index + 1),
                    listing_count=100,
                    scored_count=100,
                    score_mean=score_mean if session_score_mean is None else session_score_mean,
                    score_std=score_std if session_score_std is None else session_score_std,
                    score_coverage=1.0,
                ),
            ),
        },
        "state_hash",
    )


def test_current_refit_requires_development_vector_and_distribution_envelope() -> None:
    rng = np.random.default_rng(1729)
    card = registered_test_cards()[2]
    factor_ids = tuple(f"factor-{value:02d}" for value in range(card.factor_count))
    coefficients = np.linspace(-0.4, 0.5, card.factor_count, dtype=np.float64)
    training = rng.normal(size=(400, card.factor_count))
    targets = training @ coefficients
    current = rng.normal(size=(100, card.factor_count))
    mask = np.ones(400, dtype=np.bool_)
    complete = np.ones(100, dtype=np.bool_)
    for value in (training, targets, current, mask, complete):
        value.setflags(write=False)
    arrays = AlphaCurrentRefitArrays(
        ordered_factor_ids=factor_ids,
        ordered_listing_ids=tuple(f"listing-{value:03d}" for value in range(100)),
        training_sessions=(date(2025, 1, 1), date(2025, 1, 2)),
        training_cutoff=date(2025, 1, 2),
        formation_session=date(2025, 1, 3),
        training_features=training,
        training_targets=targets,
        training_mask=mask,
        current_features=current,
        current_feature_complete=complete,
    )
    request_hash = "b" * 64
    perturbation = np.zeros(card.factor_count, dtype=np.float64)
    perturbation[0] = 0.2
    broad_states = tuple(
        _estimator_state(
            request_hash=request_hash,
            candidate_id=card.candidate_id,
            fold_index=index,
            factor_ids=factor_ids,
            coefficients=state_coefficients,
            intercept=intercept,
            training_mse=mse,
            score_mean=mean,
            score_std=std,
        )
        for index, state_coefficients, intercept, mse, mean, std in (
            (0, coefficients * 0.5 + perturbation, -1.0, 0.0, -100.0, 0.0),
            (1, coefficients.copy(), 0.0, 0.5, 0.0, 50.0),
            (2, coefficients * 2.0 - perturbation, 1.0, 1.0, 100.0, 100.0),
        )
    )
    passed = fit_and_assess_current_alpha(
        request_hash=request_hash,
        card=card,
        arrays=arrays,
        development_states=broad_states,
    )
    assert passed.assessment.passed
    assert passed.score_table is not None

    narrow_states = tuple(
        _estimator_state(
            request_hash=request_hash,
            candidate_id=card.candidate_id,
            fold_index=index,
            factor_ids=factor_ids,
            coefficients=coefficients * scale,
            intercept=0.0,
            training_mse=1.0,
            score_mean=0.0,
            score_std=1.0,
        )
        for index, scale in ((0, 0.1), (1, 0.2))
    )
    blocked = fit_and_assess_current_alpha(
        request_hash=request_hash,
        card=card,
        arrays=arrays,
        development_states=narrow_states,
    )
    assert not blocked.assessment.passed
    assert blocked.assessment.failure_code == "alpha_research.current_refit_insufficient"
    assert blocked.score_table is None

    session_narrow_states = tuple(
        _estimator_state(
            request_hash=request_hash,
            candidate_id=card.candidate_id,
            fold_index=index,
            factor_ids=factor_ids,
            coefficients=state_coefficients,
            intercept=intercept,
            training_mse=mse,
            score_mean=mean,
            score_std=std,
            session_score_mean=1_000.0 + index,
            session_score_std=1_000.0 + index,
        )
        for index, state_coefficients, intercept, mse, mean, std in (
            (0, coefficients * 0.5 + perturbation, -1.0, 0.0, -100.0, 0.0),
            (1, coefficients.copy(), 0.0, 0.5, 0.0, 50.0),
            (2, coefficients * 2.0 - perturbation, 1.0, 1.0, 100.0, 100.0),
        )
    )
    session_blocked = fit_and_assess_current_alpha(
        request_hash=request_hash,
        card=card,
        arrays=arrays,
        development_states=session_narrow_states,
    )
    checks = {value.check_id: value for value in session_blocked.assessment.checks}
    assert not checks["CURRENT_SCORE_MEAN"].passed
    assert not checks["CURRENT_SCORE_STD"].passed
