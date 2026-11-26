"""Multiplicity-controlled Host viability for registered Alpha candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import cast

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.knowledge.catalog import SystemResourceCatalog
from alphalattice.kernel.knowledge.contracts import PackageArgSpecRef
from alphalattice.kernel.knowledge.enums import ResourceKind, ResourceNamespace
from alphalattice.kernel.validation.return_model_statistics import (
    diebold_mariano_hac,
    holm_adjust,
)

from ..experiments.contracts import AlphaCandidateRole, AlphaCandidateStatus
from ..experiments.development_contracts import (
    AlphaCandidateDevelopmentReport,
    AlphaCandidateViability,
    AlphaModelViabilityAssessment,
    seal_current_contract,
)
from .metrics import circular_block_bootstrap_interval

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AlphaValidationRows:
    """One candidate's exact OOS rows, ordered by fold/session/listing."""

    candidate_id: str
    keys: tuple[tuple[int, date, str], ...]
    targets: FloatArray
    predictions: FloatArray

    def __post_init__(self) -> None:
        """Require strictly ordered unique row keys and complete finite paired targets/predictions.

        Raises:
            ValueError: Keys are empty/unordered/duplicated, numerical axes differ or either vector
                contains nonfinite values.
        """
        if (
            not self.keys
            or any(left >= right for left, right in pairwise(self.keys))
            or self.targets.shape != self.predictions.shape
            or self.targets.shape != (len(self.keys),)
            or not np.isfinite(self.targets).all()
            or not np.isfinite(self.predictions).all()
        ):
            raise ValueError("Alpha validation rows are incomplete or non-canonical")


@dataclass(frozen=True, slots=True)
class AlphaValidationRobustnessEvidence:
    """Already-reduced session evidence used by the conjunctive viability guards."""

    candidate_id: str
    session_rank_ics: tuple[float, ...]
    session_spreads: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require finite session rank correlations and spread evidence.

        Raises:
            ValueError: A retained session rank correlation or spread is nonfinite.
        """
        if (
            not np.isfinite(np.asarray(self.session_rank_ics, dtype=np.float64)).all()
            or not np.isfinite(np.asarray(self.session_spreads, dtype=np.float64)).all()
        ):
            raise ValueError("Alpha robustness evidence must be finite")


def _dm_package_context() -> tuple[SystemResourceCatalog, tuple[PackageArgSpecRef, ...]]:
    catalog = SystemResourceCatalog.load()
    entries = {value.stable_id: value for value in catalog.manifest.entries}
    ids = ("package.statsmodels.cov-hac", "package.statsmodels.ols")
    return catalog, tuple(
        PackageArgSpecRef(
            resource_kind=ResourceKind.PACKAGE_ARG_SPEC,
            namespace=ResourceNamespace.SYSTEM,
            stable_id=stable_id,
            revision=entries[stable_id].revision,
            logical_hash=entries[stable_id].logical_hash,
        )
        for stable_id in ids
    )


def _aligned_session_losses(
    candidate: AlphaValidationRows,
    benchmark: AlphaValidationRows,
) -> tuple[FloatArray, FloatArray]:
    if candidate.keys == benchmark.keys:
        left_axis: slice | npt.NDArray[np.int64] = slice(None)
        right_axis: slice | npt.NDArray[np.int64] = slice(None)
        aligned_keys = candidate.keys
    else:
        left_indices: list[int] = []
        right_indices: list[int] = []
        left = right = 0
        while left < len(candidate.keys) and right < len(benchmark.keys):
            candidate_key = candidate.keys[left]
            benchmark_key = benchmark.keys[right]
            if candidate_key == benchmark_key:
                left_indices.append(left)
                right_indices.append(right)
                left += 1
                right += 1
            elif candidate_key < benchmark_key:
                left += 1
            else:
                right += 1
        if not left_indices:
            raise ValueError("ALPHA_VIABILITY_COMMON_SURFACE_EMPTY")
        left_axis = np.asarray(left_indices, dtype=np.int64)
        right_axis = np.asarray(right_indices, dtype=np.int64)
        aligned_keys = tuple(candidate.keys[index] for index in left_indices)
    sessions: npt.NDArray[np.int64] = np.asarray(
        [key[1].toordinal() for key in aligned_keys], dtype=np.int64
    )
    candidate_losses = np.square(candidate.targets[left_axis] - candidate.predictions[left_axis])
    benchmark_losses = np.square(benchmark.targets[right_axis] - benchmark.predictions[right_axis])
    _unique, starts, counts = np.unique(sessions, return_index=True, return_counts=True)
    admitted = counts >= 100
    candidate_session_losses = (
        np.add.reduceat(candidate_losses, starts)[admitted] / counts[admitted]
    )
    benchmark_session_losses = (
        np.add.reduceat(benchmark_losses, starts)[admitted] / counts[admitted]
    )
    if len(candidate_session_losses) < 3:
        raise ValueError("ALPHA_VIABILITY_HAC_PERIODS_INSUFFICIENT")
    return candidate_session_losses, benchmark_session_losses


def _directional_candidate_better_p_value(statistic: float, two_sided: float) -> float:
    """Convert the audited two-sided DM result to the predeclared lower-loss direction."""

    return 0.5 * two_sided if statistic < 0.0 else 1.0 - 0.5 * two_sided


_ADMISSION_CHECKS = (
    ("complete_folds", "FOLDS_INCOMPLETE"),
    ("common_surface_complete", "COMMON_SURFACE_INCOMPLETE"),
    ("finite_metrics", "METRICS_NOT_FINITE"),
    ("positive_oos_r2", "OOS_R2_NOT_POSITIVE"),
    ("positive_rank_ic", "RANK_IC_NOT_POSITIVE"),
    ("positive_gross_spread", "GROSS_SPREAD_NOT_POSITIVE"),
    ("positive_bootstrap_lower", "BOOTSTRAP_LOWER_NOT_POSITIVE"),
)
"""The checks a significant candidate must also pass, each with the code that names its failure."""


def assess_alpha_model_viability(
    *,
    request_hash: str,
    surface_hash: str,
    candidate_results: tuple[AlphaCandidateDevelopmentReport, ...],
    validation_rows: dict[str, AlphaValidationRows],
    robustness_evidence: dict[str, AlphaValidationRobustnessEvidence],
    admitted_fold_count: int,
    estimator_state_counts: dict[str, int],
    progress: Callable[[int, int, str], None] | None = None,
) -> AlphaModelViabilityAssessment:
    """Apply one dynamic formal family plus conjunctive robustness exclusions."""
    by_id = {value.candidate_id: value for value in candidate_results}
    benchmark_id = "benchmark.historical-mean"
    benchmark = validation_rows.get(benchmark_id)
    if benchmark is None or benchmark_id not in by_id:
        raise ValueError("ALPHA_HISTORICAL_MEAN_BENCHMARK_MISSING")
    regularized = tuple(
        value
        for value in candidate_results
        if value.role in {AlphaCandidateRole.REGULARIZED_ALPHA, AlphaCandidateRole.MODEL_ALPHA}
    )
    catalog, package_refs = _dm_package_context()
    staged: dict[str, dict[str, object]] = {}
    raw_family: list[tuple[str, float]] = []
    for hypothesis_index, result in enumerate(regularized, start=1):
        candidate = validation_rows.get(result.candidate_id)
        failure_codes: list[str] = []
        dm_statistic: float | None = None
        raw_p: float | None = None
        lag: int | None = None
        robustness = robustness_evidence.get(result.candidate_id)
        derived_values: tuple[float, float, float, float] | None = None
        if result.status is not AlphaCandidateStatus.SUCCEEDED or candidate is None:
            failure_codes.append("CANDIDATE_EXECUTION_INCOMPLETE")
        else:
            try:
                candidate_losses, benchmark_losses = _aligned_session_losses(candidate, benchmark)
                dm = diebold_mariano_hac(
                    candidate_losses,
                    benchmark_losses,
                    package_catalog=catalog,
                    package_arg_refs=package_refs,
                )
                dm_statistic = dm.statistic
                raw_p = _directional_candidate_better_p_value(dm.statistic, dm.p_value)
                lag = dm.lag
                raw_family.append((result.candidate_id, raw_p))
                if robustness is None:
                    raise ValueError("ALPHA_ROBUSTNESS_EVIDENCE_MISSING")
                residual = candidate.targets - candidate.predictions
                denominator = float(np.dot(candidate.targets, candidate.targets))
                r2 = (
                    1.0 - float(np.dot(residual, residual)) / denominator
                    if denominator > 0.0
                    else 0.0
                )
                rank_ic = float(np.mean(robustness.session_rank_ics))
                spread = float(np.mean(robustness.session_spreads))
                bootstrap_lower, _bootstrap_upper = circular_block_bootstrap_interval(
                    robustness.session_spreads
                )
                derived_values = (r2, rank_ic, spread, bootstrap_lower)
            except Exception as error:
                failure_codes.append(type(error).__name__)
        complete_folds = estimator_state_counts.get(result.candidate_id, 0) == admitted_fold_count
        common_complete = bool(
            result.metrics is not None
            and result.metrics.scored_row_count == result.metrics.common_surface_row_count
            and result.metrics.fold_coverage_mean == 1.0
        )
        finite_metrics = bool(
            derived_values is not None
            and np.isfinite(np.asarray(derived_values, dtype=np.float64)).all()
        )
        staged[result.candidate_id] = {
            "candidate_id": result.candidate_id,
            "dm_statistic": dm_statistic,
            "directional_raw_p_value": raw_p,
            "holm_adjusted_p_value": None,
            "hac_lag": lag,
            "complete_folds": complete_folds,
            "common_surface_complete": common_complete,
            "finite_metrics": finite_metrics,
            "positive_oos_r2": bool(derived_values is not None and derived_values[0] > 0.0),
            "positive_rank_ic": bool(derived_values is not None and derived_values[1] > 0.0),
            "positive_gross_spread": bool(derived_values is not None and derived_values[2] > 0.0),
            "positive_bootstrap_lower": bool(
                derived_values is not None and derived_values[3] > 0.0
            ),
            "failure_codes": tuple(failure_codes),
        }
        if progress is not None:
            progress(hypothesis_index, len(regularized), result.candidate_id)
    adjusted = holm_adjust(tuple(raw_family))
    candidates: list[AlphaCandidateViability] = []
    for result in regularized:
        values = staged[result.candidate_id]
        holm = adjusted.get(result.candidate_id)
        values["holm_adjusted_p_value"] = holm
        passed = (
            holm is not None
            and holm <= 0.05
            and not values["failure_codes"]
            and all(bool(values[key]) for key, _ in _ADMISSION_CHECKS)
        )
        if holm is None or holm > 0.05:
            prior_failures = cast(tuple[str, ...], values["failure_codes"])
            values["failure_codes"] = (*prior_failures, "DM_HOLM_NOT_SIGNIFICANT")
        if not passed and not values["failure_codes"]:
            # A refused candidate names what refused it, so its record carries the evidence
            # and the qualification seals its end instead of stopping.
            values["failure_codes"] = tuple(
                code for key, code in _ADMISSION_CHECKS if not values[key]
            )
        values["admitted"] = passed
        candidates.append(seal_current_contract(AlphaCandidateViability, values, "candidate_hash"))
    admissible = tuple(value.candidate_id for value in candidates if value.admitted)
    return seal_current_contract(
        AlphaModelViabilityAssessment,
        {
            "kind": "AlphaModelViabilityAssessment",
            "request_hash": request_hash,
            "surface_hash": surface_hash,
            "benchmark_candidate_id": benchmark_id,
            "familywise_alpha": 0.05,
            "hypothesis_count": len(candidates),
            "candidates": tuple(candidates),
            "admissible_candidate_ids": admissible,
            "disposition": (
                "ADMISSIBLE_MODELS_AVAILABLE" if admissible else "NO_ADMISSIBLE_ALPHA_MODEL"
            ),
        },
        "assessment_hash",
    )


__all__ = [
    "AlphaValidationRobustnessEvidence",
    "AlphaValidationRows",
    "assess_alpha_model_viability",
]
