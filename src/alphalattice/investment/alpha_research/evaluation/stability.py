"""Identity-bound current-refit stability checks shared by Alpha owners.

The gate answers one question: is the current refit's estimator state consistent
with the distribution of the development-fold states, or has it drifted?

It used to answer that with a min-max envelope over the ``K`` fold values. That
is a nonparametric prediction interval with coverage ``(K-1)/(K+1)``, so at
``K=5`` a zero-drift model failed any single check one time in three, and the
conjunction of four or five such checks rejected a stable model most of the
time. The envelope also punished monotone-good metrics for being good: a
current refit that scored every listing was rejected for exceeding the fold
maximum.

This module replaces the envelope with the statistic the envelope was reaching
for -- a Student prediction interval for one new observation drawn from the
same population as the folds --

    t = (x - mean(v)) / (sd(v) * sqrt(1 + 1/K))    ~  t(K-1)

applied one-sided where the metric has a direction, and with the family
controlled by Holm so that adding a check no longer silently raises the
rejection rate.

Assumption worth naming: the Student pivot treats the fold values as normal.
At ``K=5`` that is a modelling choice, not an established fact, and it is a
much weaker assumption than the envelope's implicit one. Bounded metrics such
as cosine and Jaccard are the least normal of the set; they are tested
one-sided, where the approximation matters least.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import stats  # type: ignore[import-untyped]

from ..experiments.development_contracts import (
    AlphaDevelopmentEstimatorState,
    AlphaEstimatorState,
)
from ..publication.contracts import AlphaStabilityCheck

type AlphaDevelopmentState = AlphaDevelopmentEstimatorState | AlphaEstimatorState
type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type Direction = Literal["two_sided", "lower", "upper"]

STABILITY_FAMILY_ALPHA = 0.01
# Two folds already admit a sample standard deviation. The resulting t(1) pivot
# is close to Cauchy and so almost never rejects, which is the honest reading of
# two observations rather than a reason to invent a narrower rule.
MINIMUM_REFERENCE_SAMPLE = 2
# A reference with no dispersion carries no scale. Compare on the value itself,
# with a tolerance that absorbs float noise but not a real change.
DEGENERATE_REL_TOL = 1e-9
DEGENERATE_ABS_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class _Statistic:
    """One drift hypothesis awaiting family correction."""

    check_id: str
    observed: float
    reference: tuple[float, ...]
    direction: Direction


def _degenerate_p_value(statistic: _Statistic, center: float) -> float:
    """Verdict when the fold reference is a point mass.

    Every fold agreeing exactly leaves no scale to standardize by, so the only
    defensible question is whether the observation is that same value. The
    tolerance keeps float noise from reading as drift; a genuine change (a
    support set that shrank, a coverage that collapsed) still fails.
    """

    tolerance = max(DEGENERATE_ABS_TOL, DEGENERATE_REL_TOL * abs(center))
    if statistic.direction == "lower":
        return 1.0 if statistic.observed >= center - tolerance else 0.0
    if statistic.direction == "upper":
        return 1.0 if statistic.observed <= center + tolerance else 0.0
    return 1.0 if abs(statistic.observed - center) <= tolerance else 0.0


def _prediction_p_value(statistic: _Statistic) -> float:
    """Two- or one-sided p-value for one new draw from the fold population."""

    reference: FloatArray = np.asarray(statistic.reference, dtype=np.float64)
    count = reference.size
    center = float(np.mean(reference))
    spread = float(np.std(reference, ddof=1)) if count >= MINIMUM_REFERENCE_SAMPLE else 0.0
    if spread <= 0.0 or not math.isfinite(spread):
        return _degenerate_p_value(statistic, center)
    scale = spread * math.sqrt(1.0 + 1.0 / count)
    t_statistic = (statistic.observed - center) / scale
    degrees = count - 1
    if statistic.direction == "two_sided":
        return float(2.0 * stats.t.sf(abs(t_statistic), degrees))
    if statistic.direction == "lower":
        return float(stats.t.cdf(t_statistic, degrees))
    return float(stats.t.sf(t_statistic, degrees))


def _prediction_bounds(statistic: _Statistic, alpha: float) -> tuple[float | None, float | None]:
    """Interval implied by the level this hypothesis was actually judged at."""

    reference: FloatArray = np.asarray(statistic.reference, dtype=np.float64)
    count = reference.size
    center = float(np.mean(reference))
    spread = float(np.std(reference, ddof=1)) if count >= MINIMUM_REFERENCE_SAMPLE else 0.0
    if spread <= 0.0 or not math.isfinite(spread):
        # Degenerate reference: the bound collapses onto the centre, but a
        # one-sided hypothesis must still publish only the side it tested.
        if statistic.direction == "lower":
            return (center, None)
        if statistic.direction == "upper":
            return (None, center)
        return (center, center)
    scale = spread * math.sqrt(1.0 + 1.0 / count)
    degrees = count - 1
    if statistic.direction == "two_sided":
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees))
        return (center - critical * scale, center + critical * scale)
    critical = float(stats.t.ppf(1.0 - alpha, degrees))
    if statistic.direction == "lower":
        return (center - critical * scale, None)
    return (None, center + critical * scale)


def _holm(statistics: tuple[_Statistic, ...], alpha: float) -> tuple[AlphaStabilityCheck, ...]:
    """Judge every drift hypothesis at its Holm level and report the bound used."""

    p_values = tuple(_prediction_p_value(item) for item in statistics)
    total = len(statistics)
    order = sorted(range(total), key=lambda index: p_values[index])
    levels = [alpha] * total
    rejected = [False] * total
    still_rejecting = True
    for rank, index in enumerate(order):
        level = alpha / float(total - rank)
        levels[index] = level
        if still_rejecting and p_values[index] <= level:
            rejected[index] = True
        else:
            still_rejecting = False
    checks: list[AlphaStabilityCheck] = []
    for index, statistic in enumerate(statistics):
        lower, upper = _prediction_bounds(statistic, levels[index])
        checks.append(
            AlphaStabilityCheck(
                check_id=statistic.check_id,
                passed=not rejected[index],
                observed=statistic.observed,
                lower=lower,
                upper=upper,
            )
        )
    return tuple(checks)


def _cosine(left: FloatArray, right: FloatArray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 1.0 if np.array_equal(left, right) else -1.0
    return float(np.dot(left, right) / denominator)


def _jaccard(left: BoolArray, right: BoolArray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _leave_one_out_centers(vectors: tuple[FloatArray, ...]) -> tuple[FloatArray, ...]:
    """Median of every fold's peers, excluding the fold being scored.

    The previous baseline compared each fold against a median that contained the
    fold itself while the current refit was compared against a median it was
    absent from. In-sample similarity is structurally higher, so the comparison
    was biased toward rejection. Excluding the scored fold removes that bias;
    the residual difference (the current refit sees ``K`` peers, a fold sees
    ``K-1``) is second order and no longer favours either side systematically.
    """

    stacked = np.stack(vectors)
    return tuple(
        cast(FloatArray, np.median(np.delete(stacked, index, axis=0), axis=0))
        for index in range(len(vectors))
    )


def _linear_statistics(
    current: AlphaDevelopmentState,
    development: tuple[AlphaDevelopmentState, ...],
) -> tuple[tuple[AlphaStabilityCheck, ...], tuple[_Statistic, ...]]:
    fold_vectors: tuple[FloatArray, ...] = tuple(
        np.asarray([float.fromhex(value) for value in state.coefficient_hex], dtype=np.float64)
        for state in development
    )
    current_vector: FloatArray = np.asarray(
        [float.fromhex(value) for value in current.coefficient_hex], dtype=np.float64
    )
    center = np.median(np.stack(fold_vectors), axis=0)
    peers = _leave_one_out_centers(fold_vectors)
    structural = (
        AlphaStabilityCheck(
            check_id="COEFFICIENT_AXIS_AND_FINITE",
            passed=all(
                value.ordered_factor_ids == current.ordered_factor_ids for value in development
            ),
            observed="MATCH",
        ),
    )
    statistics = [
        _Statistic(
            check_id="COEFFICIENT_L2_NORM",
            observed=current.coefficient_l2_norm,
            reference=tuple(value.coefficient_l2_norm for value in development),
            direction="two_sided",
        ),
        _Statistic(
            check_id="COEFFICIENT_MAX_ABS",
            observed=current.coefficient_max_abs,
            reference=tuple(value.coefficient_max_abs for value in development),
            direction="two_sided",
        ),
        _Statistic(
            check_id="INTERCEPT",
            observed=float.fromhex(current.intercept_hex),
            reference=tuple(float.fromhex(value.intercept_hex) for value in development),
            direction="two_sided",
        ),
        _Statistic(
            check_id="COEFFICIENT_MEDIAN_COSINE",
            observed=_cosine(current_vector, cast(FloatArray, center)),
            reference=tuple(
                _cosine(vector, peer) for vector, peer in zip(fold_vectors, peers, strict=True)
            ),
            direction="lower",
        ),
    ]
    if current.family_id in {"lasso", "elastic_net"}:
        current_support = cast(BoolArray, current_vector != 0.0)
        median_support = cast(
            BoolArray,
            np.median(np.stack([vector != 0.0 for vector in fold_vectors]).astype(np.float64), 0)
            >= 0.5,
        )
        peer_supports = tuple(
            cast(BoolArray, peer >= 0.5)
            for peer in _leave_one_out_centers(
                tuple(
                    cast(FloatArray, (vector != 0.0).astype(np.float64)) for vector in fold_vectors
                )
            )
        )
        statistics.extend(
            (
                _Statistic(
                    check_id="NONZERO_SUPPORT_COUNT",
                    observed=float(current.nonzero_support_count or 0),
                    reference=tuple(
                        float(value.nonzero_support_count or 0) for value in development
                    ),
                    direction="two_sided",
                ),
                _Statistic(
                    check_id="SUPPORT_MEDIAN_JACCARD",
                    observed=_jaccard(current_support, median_support),
                    reference=tuple(
                        _jaccard(cast(BoolArray, vector != 0.0), peer)
                        for vector, peer in zip(fold_vectors, peer_supports, strict=True)
                    ),
                    direction="lower",
                ),
            )
        )
    return structural, tuple(statistics)


def _tree_statistics(
    current: AlphaDevelopmentState,
    development: tuple[AlphaDevelopmentState, ...],
) -> tuple[tuple[AlphaStabilityCheck, ...], tuple[_Statistic, ...]]:
    fold_gains: tuple[FloatArray, ...] = tuple(
        np.asarray([float.fromhex(value) for value in state.feature_gain_hex], dtype=np.float64)
        for state in development
    )
    current_gain: FloatArray = np.asarray(
        [float.fromhex(value) for value in current.feature_gain_hex], dtype=np.float64
    )
    center = np.median(np.stack(fold_gains), axis=0)
    peers = _leave_one_out_centers(fold_gains)
    structural = (
        AlphaStabilityCheck(
            check_id="FEATURE_GAIN_AXIS_AND_FINITE",
            passed=all(
                value.ordered_factor_ids == current.ordered_factor_ids for value in development
            ),
            observed="MATCH",
        ),
    )
    statistics = (
        _Statistic(
            check_id="FEATURE_GAIN_MEDIAN_COSINE",
            observed=_cosine(current_gain, cast(FloatArray, center)),
            reference=tuple(
                _cosine(gain, peer) for gain, peer in zip(fold_gains, peers, strict=True)
            ),
            direction="lower",
        ),
        _Statistic(
            check_id="TOP_FEATURE_GAIN_SHARE",
            observed=current.top_feature_gain_share or 0.0,
            reference=tuple(value.top_feature_gain_share or 0.0 for value in development),
            direction="two_sided",
        ),
    )
    return structural, statistics


def build_estimator_stability_checks(
    current: AlphaDevelopmentState,
    development: tuple[AlphaDevelopmentState, ...],
    *,
    family_alpha: float = STABILITY_FAMILY_ALPHA,
) -> tuple[AlphaStabilityCheck, ...]:
    """Apply the registered state prediction interval without fitting a model."""
    if not development or any(
        value.candidate_id != current.candidate_id
        or value.ordered_factor_ids != current.ordered_factor_ids
        or value.state_kind != current.state_kind
        for value in development
    ):
        raise ValueError("ALPHA_STABILITY_REFERENCE_STATE_MISMATCH")
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("ALPHA_STABILITY_FAMILY_ALPHA_INVALID")
    # A kind the installed families do not seal (an agent's model, V342) has no structural
    # diagnostics to compare; the policy's model-agnostic statistics below judge it alone.
    structural, family = (
        _tree_statistics(current, development)
        if current.state_kind == "TREE"
        else _linear_statistics(current, development)
        if current.state_kind == "LINEAR"
        else ((), ())
    )
    session_statistics = tuple(
        statistic for state in development for statistic in state.validation_session_statistics
    )
    family = (
        *family,
        _Statistic(
            check_id="TRAINING_MSE",
            observed=current.training_mse,
            reference=tuple(value.training_mse for value in development),
            # Only a materially worse fit is a stability failure. A current
            # window that fits better than every fold is not drift evidence.
            direction="upper",
        ),
        _Statistic(
            check_id="CURRENT_SCORE_MEAN",
            observed=current.validation_score_mean,
            reference=tuple(value.score_mean for value in session_statistics),
            direction="two_sided",
        ),
        _Statistic(
            check_id="CURRENT_SCORE_STD",
            observed=current.validation_score_std,
            reference=tuple(value.score_std for value in session_statistics),
            direction="two_sided",
        ),
        _Statistic(
            check_id="CURRENT_SCORE_COVERAGE",
            observed=current.validation_score_coverage,
            # Coverage is monotone good. Scoring every listing must never be a
            # failure for exceeding what the folds happened to reach.
            reference=tuple(value.score_coverage for value in session_statistics),
            direction="lower",
        ),
    )
    return (*structural, *_holm(family, family_alpha))


__all__ = [
    "MINIMUM_REFERENCE_SAMPLE",
    "STABILITY_FAMILY_ALPHA",
    "AlphaDevelopmentState",
    "build_estimator_stability_checks",
]
