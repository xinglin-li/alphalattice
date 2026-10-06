"""Deterministic development metrics over the common Alpha comparison surface."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import numpy as np
import numpy.typing as npt

from alphalattice.kernel.validation.screening_statistics import (
    average_ranks,
    gross_decile_spread,
    spearman_rank_correlation,
)

from ..inputs.folds import AlphaFoldArrays
from .contracts import (
    AlphaCandidateMetrics,
    AlphaFoldMetrics,
    AlphaMetricAvailability,
    AlphaMetricPolicy,
    AlphaMetricValue,
    seal_evaluation_contract,
)

type FloatArray = npt.NDArray[np.float64]
type ConstantScorePolicy = Literal["ABSTAIN_AS_ZERO", "UNDEFINED"]

_CIRCULAR_BOOTSTRAP_SEED = 1729
_CIRCULAR_BOOTSTRAP_RESAMPLES = 2_000

ONE_WAY_COST: Final = 0.001
"""The registered one-way cost proxy, ten basis points per unit of turnover."""

DECILE_TAIL_MEAN: Final = 1.7550
"""Standard-normal mean above the 90th percentile, ``phi(1.2816) / 0.1``.

Recorded because it is the constant a structural spread-to-IC conversion would
need. That conversion was tested against the sealed evidence and refused: the
realized ``spread / (DECILE_TAIL_MEAN * sigma * IC)`` ratio is -1.369 and -0.216
at one session and +31.0 and +7.27 at five, where a sound conversion would give
1.0. The turnover coefficient below is therefore a declared preference, not an
estimate of that relation.
"""

SIGMA_XS_SECTOR_NEUTRAL: Final[Mapping[int, float]] = {1: 0.01468, 5: 0.03264}
"""Median per-session cross-sectional SD of the fixed Sector-neutral economic lane.

Measured over the 1,067 (one-session) and 1,259 (five-session) paired evaluation
sessions of the sealed development surface. A lane property, so it cannot favour
an arm.
"""

TURNOVER_IC_COST: Final[Mapping[int, float]] = {
    horizon: ONE_WAY_COST / dispersion for horizon, dispersion in SIGMA_XS_SECTOR_NEUTRAL.items()
}
"""How much rank IC one unit of one-way turnover is declared to be worth.

A risk preference pinned into Program identity, not a measured quantity -- see
``DECILE_TAIL_MEAN``. Every published contrast must also report the same
difference at a zero coefficient so the choice stays auditable.
"""


@dataclass(frozen=True, slots=True)
class FoldScoreVector:
    """Bind one fold to read-only float64 scores aligned with its validation targets."""

    fold: AlphaFoldArrays
    scores: FloatArray

    def __post_init__(self) -> None:
        """Require validation-aligned float64 scores with writeability disabled.

        Raises:
            ValueError: Shape differs from validation targets, dtype is not float64 or scores are
                writeable.
        """
        if self.scores.shape != self.fold.validation_targets.shape:
            raise ValueError("Alpha fold score shape differs from validation rows")
        if self.scores.dtype != np.dtype(np.float64):
            raise ValueError("Alpha fold scores must use float64")
        if self.scores.flags.writeable:
            raise ValueError("Alpha fold scores must be read-only")


@dataclass(frozen=True, slots=True)
class AlphaMetricComputation:
    """Retain candidate metrics and session rank correlations used by aggregate evidence."""

    metrics: AlphaCandidateMetrics
    session_rank_ics: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AlphaFoldMetricEvidence:
    """Compact validation-only evidence; training arrays can be released per fold."""

    metrics: AlphaFoldMetrics
    scored_sessions: tuple[date, ...]
    scored_targets: FloatArray
    scored_predictions: FloatArray
    session_rank_ics: tuple[float, ...]
    session_spreads: tuple[float, ...]

    def __post_init__(self) -> None:
        """Require equal compact score/target/session axes and non-writeable arrays.

        Raises:
            ValueError: Compact targets/predictions/session counts differ or either array is
                writeable.
        """
        if (
            self.scored_targets.shape != self.scored_predictions.shape
            or self.scored_targets.shape != (len(self.scored_sessions),)
            or self.scored_targets.flags.writeable
            or self.scored_predictions.flags.writeable
        ):
            raise ValueError("compact Alpha fold evidence is not aligned and immutable")


def _clean(value: float) -> float:
    return 0.0 if value == 0.0 else value


def circular_block_bootstrap_interval(
    observations: tuple[float, ...],
) -> tuple[float, float]:
    """Return the registered deterministic interval from compact period evidence.

    The policy intentionally matches the tracked return-model statistics owner while
    avoiding a second row-level session reduction during Alpha viability admission.
    """
    values: FloatArray = np.asarray(observations, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Alpha bootstrap observations must be finite and non-empty")
    periods = len(values)
    block = max(1, round(periods ** (1.0 / 3.0)))
    blocks = math.ceil(periods / block)
    rng = np.random.Generator(np.random.PCG64(_CIRCULAR_BOOTSTRAP_SEED))
    estimates: FloatArray = np.empty(_CIRCULAR_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    offsets = np.arange(block, dtype=np.int64)
    for index in range(_CIRCULAR_BOOTSTRAP_RESAMPLES):
        starts = rng.integers(0, periods, size=blocks, dtype=np.int64)
        sample_indices = ((starts[:, None] + offsets[None, :]) % periods).reshape(-1)[:periods]
        estimates[index] = float(np.mean(values[sample_indices]))
    lower, upper = np.percentile(estimates, (2.5, 97.5), method="linear")
    return float(lower), float(upper)


def _available(value: float) -> AlphaMetricValue:
    if not math.isfinite(value):
        raise ValueError("Alpha metric value must be finite")
    return AlphaMetricValue(availability=AlphaMetricAvailability.AVAILABLE, value=_clean(value))


def _unavailable(reason: AlphaMetricAvailability) -> AlphaMetricValue:
    if reason is AlphaMetricAvailability.AVAILABLE:
        raise ValueError("unavailable Alpha metric requires a reason")
    return AlphaMetricValue(availability=reason, value=None)


def _metric_or_unavailable(
    values: list[float], *, reason: AlphaMetricAvailability
) -> AlphaMetricValue:
    return _available(float(np.mean(values))) if values else _unavailable(reason)


def build_formation_decile_weights(
    listing_ids: tuple[str, ...], scores: FloatArray
) -> dict[str, float] | None:
    """Build the registered equal-weight long/short formation deciles."""
    if scores.size < 100 or not np.isfinite(scores).all() or float(np.ptp(scores)) == 0.0:
        return None
    ranks = average_ranks(scores)
    percentiles = (ranks - 0.5) / scores.size
    bottom = np.flatnonzero(percentiles <= 0.1)
    top = np.flatnonzero(percentiles >= 0.9)
    if bottom.size == 0 or top.size == 0:
        return None
    result = {listing_id: 0.0 for listing_id in listing_ids}
    for index in bottom:
        result[listing_ids[int(index)]] = -1.0 / bottom.size
    for index in top:
        result[listing_ids[int(index)]] = 1.0 / top.size
    return result


def _validation_session_rows(fold: AlphaFoldArrays) -> dict[date, npt.NDArray[np.int64]]:
    """The validation row positions of every session, in row order, found once.

    Every per-session statistic below selects a session's rows and then keeps
    the admitted ones. Finding them was one pass over every validation row per
    session -- sessions times rows per fold, twice -- for a result that is the
    same on every pass; one pass groups the positions, and each session takes
    its own, in the same ascending row order.
    """

    positions: dict[date, list[int]] = {}
    for index, session in enumerate(fold.validation_row_sessions):
        positions.setdefault(session, []).append(index)
    return {session: np.asarray(rows, dtype=np.int64) for session, rows in positions.items()}


def _formation_turnover(
    fold: AlphaFoldArrays,
    scores: FloatArray,
    *,
    session_rows: Mapping[date, npt.NDArray[np.int64]],
) -> tuple[float | None, AlphaMetricAvailability]:
    weights: dict[date, dict[str, float]] = {}
    common = fold.common_comparison_mask & np.isfinite(scores)
    for session in fold.validation_sessions:
        rows = session_rows.get(session)
        if rows is None:
            continue
        indices: npt.NDArray[np.int64] = rows[common[rows]]
        if indices.size < 100:
            continue
        session_scores = scores[indices]
        listing_ids = tuple(fold.validation_listing_ids[int(index)] for index in indices)
        value = build_formation_decile_weights(listing_ids, session_scores)
        if value is not None:
            weights[session] = value
    observations: list[float] = []
    for previous, current in zip(
        fold.validation_sessions[:-1], fold.validation_sessions[1:], strict=True
    ):
        if previous not in weights or current not in weights:
            continue
        listing_set = set(weights[previous]).union(weights[current])
        turnover = 0.5 * sum(
            abs(weights[current].get(value, 0.0) - weights[previous].get(value, 0.0))
            for value in listing_set
        )
        observations.append(_clean(float(turnover)))
    if observations:
        return _clean(float(np.mean(observations))), AlphaMetricAvailability.AVAILABLE
    if any(
        np.isfinite(scores[np.asarray(fold.validation_row_sessions) == session]).any()
        and float(np.ptp(scores[np.asarray(fold.validation_row_sessions) == session])) == 0.0
        for session in fold.validation_sessions
    ):
        return None, AlphaMetricAvailability.CONSTANT_SCORE
    return None, AlphaMetricAvailability.INSUFFICIENT_CROSS_SECTION


def _fold_metrics(
    value: FoldScoreVector,
    *,
    policy: AlphaMetricPolicy,
) -> tuple[AlphaFoldMetrics, tuple[float, ...], tuple[float, ...]]:
    fold = value.fold
    comparison = fold.common_comparison_mask
    scored = comparison & np.isfinite(value.scores)
    count = int(comparison.sum())
    scored_count = int(scored.sum())
    unavailable = AlphaMetricAvailability.INSUFFICIENT_CROSS_SECTION
    if scored_count:
        target = fold.validation_targets[scored]
        score = value.scores[scored]
        residual = target - score
        mae = _available(float(np.mean(np.abs(residual))))
        mse = _available(float(np.mean(np.square(residual))))
        denominator = float(np.dot(target, target))
        r2 = (
            _available(1.0 - float(np.dot(residual, residual)) / denominator)
            if denominator > 0.0 and math.isfinite(denominator)
            else _unavailable(unavailable)
        )
    else:
        mae = mse = r2 = _unavailable(unavailable)

    rank_ics: list[float] = []
    spreads: list[float] = []
    any_constant = False
    session_rows = _validation_session_rows(fold)
    for session in fold.validation_sessions:
        rows = session_rows.get(session)
        if rows is None:
            continue
        session_indices: npt.NDArray[np.int64] = rows[scored[rows]]
        if session_indices.size < policy.minimum_cross_section:
            continue
        session_scores = value.scores[session_indices]
        session_targets = fold.economic_validation_targets[session_indices]
        if float(np.ptp(session_scores)) == 0.0:
            any_constant = True
        rank_ic = spearman_rank_correlation(session_scores, session_targets)
        spread = gross_decile_spread(session_scores, session_targets)
        if rank_ic is not None:
            rank_ics.append(rank_ic)
        if spread is not None:
            spreads.append(spread)
    reason = (
        AlphaMetricAvailability.CONSTANT_SCORE
        if any_constant and not rank_ics
        else AlphaMetricAvailability.INSUFFICIENT_CROSS_SECTION
    )
    turnover, turnover_reason = _formation_turnover(fold, value.scores, session_rows=session_rows)
    values = {
        "fold_index": fold.commitment.fold_index,
        "comparison_row_count": count,
        "scored_comparison_row_count": scored_count,
        "coverage": scored_count / count if count else 0.0,
        "mae": mae,
        "mse": mse,
        "zero_relative_oos_r2": r2,
        "rank_ic": _metric_or_unavailable(rank_ics, reason=reason),
        "gross_decile_spread": _metric_or_unavailable(spreads, reason=reason),
        "formation_decile_turnover": (
            _available(turnover) if turnover is not None else _unavailable(turnover_reason)
        ),
    }
    return (
        seal_evaluation_contract(AlphaFoldMetrics, values, "metrics_hash"),
        tuple(rank_ics),
        tuple(spreads),
    )


def compute_fold_metric_evidence(
    value: FoldScoreVector,
    *,
    policy: AlphaMetricPolicy,
) -> AlphaFoldMetricEvidence:
    """Reduce one completed fold to bounded validation evidence."""
    metrics, rank_ics, spreads = _fold_metrics(value, policy=policy)
    mask = value.fold.common_comparison_mask & np.isfinite(value.scores)
    targets: FloatArray = np.asarray(value.fold.validation_targets[mask], dtype=np.float64)
    predictions: FloatArray = np.asarray(value.scores[mask], dtype=np.float64)
    targets.setflags(write=False)
    predictions.setflags(write=False)
    return AlphaFoldMetricEvidence(
        metrics=metrics,
        scored_sessions=tuple(
            session
            for session, admitted in zip(value.fold.validation_row_sessions, mask, strict=True)
            if bool(admitted)
        ),
        scored_targets=targets,
        scored_predictions=predictions,
        session_rank_ics=rank_ics,
        session_spreads=spreads,
    )


def aggregate_candidate_metrics(
    candidate_id: str,
    folds: tuple[AlphaFoldMetricEvidence, ...],
) -> AlphaCandidateMetrics:
    """Aggregate compact fold evidence without retaining training feature arrays."""
    if not folds:
        raise ValueError("Alpha metrics require at least one fold")
    fold_metrics = tuple(value.metrics for value in folds)
    common_count = sum(value.comparison_row_count for value in fold_metrics)
    scored_count = sum(value.scored_comparison_row_count for value in fold_metrics)
    pooled_scores = tuple(
        value.scored_predictions for value in folds if value.scored_predictions.size
    )
    pooled_targets = tuple(value.scored_targets for value in folds if value.scored_targets.size)
    reason = AlphaMetricAvailability.INSUFFICIENT_CROSS_SECTION
    if pooled_scores:
        scores = np.concatenate(pooled_scores).astype(np.float64, copy=False)
        targets = np.concatenate(pooled_targets).astype(np.float64, copy=False)
        residual = targets - scores
        mae = _available(float(np.mean(np.abs(residual))))
        mse = _available(float(np.mean(np.square(residual))))
        denominator = float(np.dot(targets, targets))
        r2 = (
            _available(1.0 - float(np.dot(residual, residual)) / denominator)
            if denominator > 0.0 and math.isfinite(denominator)
            else _unavailable(reason)
        )
    else:
        mae = mse = r2 = _unavailable(reason)
    rank_ics = [item for value in folds for item in value.session_rank_ics]
    spreads = [item for value in folds for item in value.session_spreads]
    rank_reason = (
        AlphaMetricAvailability.CONSTANT_SCORE
        if all(
            value.rank_ic.availability is AlphaMetricAvailability.CONSTANT_SCORE
            for value in fold_metrics
        )
        else AlphaMetricAvailability.INSUFFICIENT_CROSS_SECTION
    )
    rank_mean = _metric_or_unavailable(rank_ics, reason=rank_reason)
    if len(rank_ics) >= 2:
        dispersion = float(np.std(np.asarray(rank_ics, dtype=np.float64), ddof=1))
        icir = (
            _available(float(np.mean(rank_ics)) / dispersion)
            if dispersion > 0.0 and math.isfinite(dispersion)
            else _unavailable(AlphaMetricAvailability.INSUFFICIENT_FOLDS)
        )
    else:
        icir = _unavailable(AlphaMetricAvailability.INSUFFICIENT_FOLDS)
    turnover_values = [
        value.formation_decile_turnover.value
        for value in fold_metrics
        if value.formation_decile_turnover.value is not None
    ]
    fold_mses = [value.mse.value for value in fold_metrics if value.mse.value is not None]
    values = {
        "candidate_id": candidate_id,
        "fold_metrics": fold_metrics,
        "common_surface_row_count": common_count,
        "scored_row_count": scored_count,
        "mae": mae,
        "mse": mse,
        "zero_relative_oos_r2": r2,
        "rank_ic_mean": rank_mean,
        "icir": icir,
        "gross_decile_spread_mean": _metric_or_unavailable(spreads, reason=rank_reason),
        "formation_decile_turnover_mean": _metric_or_unavailable(
            [float(value) for value in turnover_values], reason=rank_reason
        ),
        "fold_coverage_mean": float(np.mean([value.coverage for value in fold_metrics])),
        "fold_mse_std": (
            _available(float(np.std(fold_mses, ddof=1)))
            if len(fold_mses) >= 2
            else _unavailable(AlphaMetricAvailability.INSUFFICIENT_FOLDS)
        ),
    }
    return seal_evaluation_contract(AlphaCandidateMetrics, values, "metrics_hash")


def compute_candidate_metrics(
    candidate_id: str,
    folds: tuple[FoldScoreVector, ...],
    *,
    policy: AlphaMetricPolicy,
) -> AlphaMetricComputation:
    """Compute all admitted metrics; no candidate comparison or selection occurs here."""
    if not folds:
        raise ValueError("Alpha metrics require at least one fold")
    evidence = tuple(compute_fold_metric_evidence(value, policy=policy) for value in folds)
    return AlphaMetricComputation(
        metrics=aggregate_candidate_metrics(candidate_id, evidence),
        session_rank_ics=tuple(item for value in evidence for item in value.session_rank_ics),
    )


__all__ = [
    "DECILE_TAIL_MEAN",
    "ONE_WAY_COST",
    "SIGMA_XS_SECTOR_NEUTRAL",
    "TURNOVER_IC_COST",
    "AlphaFoldMetricEvidence",
    "AlphaMetricComputation",
    "ConstantScorePolicy",
    "FoldScoreVector",
    "aggregate_candidate_metrics",
    "build_formation_decile_weights",
    "circular_block_bootstrap_interval",
    "compute_candidate_metrics",
    "compute_fold_metric_evidence",
]
