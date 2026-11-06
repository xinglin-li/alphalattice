"""Metric edge cases for ties, constants, ICIR, spread, and turnover."""

from __future__ import annotations

import math

import numpy as np

from alphalattice.investment.alpha_research.evaluation.metrics import (
    FoldScoreVector,
    circular_block_bootstrap_interval,
    compute_candidate_metrics,
)
from alphalattice.investment.alpha_research.experiments.policies import load_alpha_metric_policy
from tests.alpha_research.fixtures import synthetic_prepared_arrays


def _readonly(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def test_constant_score_rank_and_spread_are_unavailable_not_failure() -> None:
    prepared = synthetic_prepared_arrays()
    folds = tuple(
        FoldScoreVector(
            fold=fold,
            scores=_readonly(np.zeros(len(fold.validation_listing_ids))),
        )
        for fold in prepared.folds
    )
    metrics = compute_candidate_metrics(
        "benchmark.zero-forecast", folds, policy=load_alpha_metric_policy()
    ).metrics
    assert metrics.mae.value is not None
    assert metrics.mse.value is not None
    assert metrics.rank_ic_mean.availability.value == "CONSTANT_SCORE"
    assert metrics.gross_decile_spread_mean.availability.value == "CONSTANT_SCORE"


def test_ties_and_partial_scores_preserve_common_surface_and_coverage() -> None:
    prepared = synthetic_prepared_arrays()
    values = []
    for fold in prepared.folds:
        scores = np.asarray(
            [float(index // 5) for index in range(len(fold.validation_listing_ids))],
            dtype=np.float64,
        )
        scores[::2] = np.nan
        values.append(FoldScoreVector(fold=fold, scores=_readonly(scores)))
    metrics = compute_candidate_metrics(
        "model.ridge.alpha-1p0", tuple(values), policy=load_alpha_metric_policy()
    ).metrics
    assert metrics.common_surface_row_count == prepared.common_surface_row_count
    assert metrics.scored_row_count == prepared.common_surface_row_count // 2
    assert metrics.fold_coverage_mean == 0.5
    assert metrics.zero_relative_oos_r2.value is not None


def _registered_policy_interval(spreads: np.ndarray) -> tuple[float, float]:
    """The registered policy, as the retired return-model statistics owner stated it.

    Seed 1729, 2,000 circular-block resamples, a block of round(n ** (1/3)) periods,
    the 2.5 and 97.5 linear percentiles of the resampled means.
    """

    periods = len(spreads)
    block = max(1, round(periods ** (1.0 / 3.0)))
    blocks = math.ceil(periods / block)
    rng = np.random.Generator(np.random.PCG64(1729))
    estimates = np.empty(2_000, dtype=np.float64)
    offsets = np.arange(block, dtype=np.int64)
    for index in range(2_000):
        starts = rng.integers(0, periods, size=blocks, dtype=np.int64)
        sample_indices = ((starts[:, None] + offsets[None, :]) % periods).reshape(-1)[:periods]
        estimates[index] = float(np.mean(spreads[sample_indices]))
    lower, upper = np.percentile(estimates, (2.5, 97.5), method="linear")
    return float(lower), float(upper)


def test_compact_bootstrap_matches_tracked_registered_policy() -> None:
    spreads = tuple(np.linspace(-0.02, 0.04, 37, dtype=np.float64))
    expected = _registered_policy_interval(np.asarray(spreads, dtype=np.float64))
    assert circular_block_bootstrap_interval(spreads) == expected
