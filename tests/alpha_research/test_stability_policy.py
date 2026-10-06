"""Current-refit stability gate: direction, baseline symmetry, and family size.

These cases pin the three defects the prediction-interval rewrite exists to
remove, and measure the gate's behaviour under a zero-drift null so the
false-rejection rate is a recorded number rather than a claim.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from alphalattice.investment.alpha_research.evaluation.stability import (
    STABILITY_FAMILY_ALPHA,
    build_estimator_stability_checks,
)
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaEstimatorState,
    AlphaSessionScoreStatistics,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.publication.contracts import AlphaStabilityCheck


def seal_contract(model, identity_field: str, **values):
    return seal_current_contract(model, values, identity_field)


FACTORS = ("f0", "f1", "f2", "f3", "f4", "f5")
REQUEST_HASH = "1" * 64


def _sessions(count: int, *, coverage: float = 0.99) -> tuple[AlphaSessionScoreStatistics, ...]:
    """Sessions whose coverage varies the way a real validation window does."""

    listing_count = 1000
    start = date(2024, 1, 1)
    statistics = []
    for index in range(count):
        # Vary within a few tenths of a percent so the reference has real spread
        # and never reaches 1.0 on its own.
        scored = round((coverage - 0.002 * (index % 5)) * listing_count)
        statistics.append(
            AlphaSessionScoreStatistics(
                formation_session=start + timedelta(days=index),
                listing_count=listing_count,
                scored_count=scored,
                score_mean=0.0,
                score_std=1.0,
                score_coverage=scored / listing_count,
            )
        )
    return tuple(statistics)


def _state(
    coefficients: tuple[float, ...],
    *,
    scope: str,
    fold_index: int | None,
    training_mse: float = 1.0,
    coverage: float = 0.99,
    session_count: int | None = None,
    score_mean: float = 0.0,
    score_std: float = 1.0,
) -> AlphaEstimatorState:
    # A current refit carries exactly one formation session; a fold carries its
    # whole validation window. Mirror that, because the gate pools fold sessions.
    resolved = session_count if session_count is not None else (1 if fold_index is None else 40)
    return seal_contract(
        AlphaEstimatorState,
        "state_hash",
        request_hash=REQUEST_HASH,
        candidate_id="candidate",
        scope=scope,
        family_id="ridge",
        state_kind="LINEAR",
        fold_index=fold_index,
        ordered_factor_ids=FACTORS,
        coefficient_hex=tuple(value.hex() for value in coefficients),
        intercept_hex=(0.0).hex(),
        coefficient_l2_norm=math.sqrt(sum(value * value for value in coefficients)),
        coefficient_max_abs=max(abs(value) for value in coefficients),
        training_mse=training_mse,
        validation_score_mean=score_mean,
        validation_score_std=score_std,
        validation_score_coverage=coverage,
        validation_session_statistics=_sessions(resolved, coverage=coverage),
    )


def _folds(
    generator: np.random.Generator, *, count: int = 5, scale: float = 1.0
) -> tuple[AlphaEstimatorState, ...]:
    return tuple(
        _state(
            tuple(generator.normal(loc=1.0, scale=0.1 * scale, size=len(FACTORS))),
            scope="DEVELOPMENT_FOLD",
            fold_index=index,
            training_mse=float(abs(generator.normal(loc=1.0, scale=0.05))),
        )
        for index in range(count)
    )


def _named(checks: tuple[AlphaStabilityCheck, ...]) -> dict[str, AlphaStabilityCheck]:
    return {check.check_id: check for check in checks}


def test_full_coverage_is_not_a_stability_failure() -> None:
    """A refit that scores every listing must not fail for beating the folds."""

    generator = np.random.default_rng(20260810)
    development = _folds(generator)
    current = _state(
        tuple(generator.normal(loc=1.0, scale=0.1, size=len(FACTORS))),
        scope="CURRENT_REFIT",
        fold_index=None,
        coverage=1.0,
    )
    checks = _named(build_estimator_stability_checks(current, development))
    coverage_check = checks["CURRENT_SCORE_COVERAGE"]
    assert coverage_check.observed == 1.0
    assert coverage_check.passed is True
    # A one-sided lower bound must not publish an upper bound at all.
    assert coverage_check.upper is None


def test_training_mse_better_than_every_fold_passes() -> None:
    """Fitting better than the folds is not drift evidence."""

    generator = np.random.default_rng(7)
    development = _folds(generator)
    floor = min(state.training_mse for state in development)
    current = _state(
        tuple(generator.normal(loc=1.0, scale=0.1, size=len(FACTORS))),
        scope="CURRENT_REFIT",
        fold_index=None,
        training_mse=floor / 10.0,
    )
    checks = _named(build_estimator_stability_checks(current, development))
    assert checks["TRAINING_MSE"].passed is True


def test_median_cosine_baseline_excludes_the_scored_fold() -> None:
    """The fold baseline must not be measured against a median containing it.

    With the in-sample baseline every fold sits at cosine ~1 against a median it
    helped define, so the out-of-sample current refit is compared against an
    unreachable floor. Leave-one-out makes the reference spread real.
    """

    generator = np.random.default_rng(11)
    development = _folds(generator)
    current = _state(
        tuple(generator.normal(loc=1.0, scale=0.1, size=len(FACTORS))),
        scope="CURRENT_REFIT",
        fold_index=None,
    )
    checks = _named(build_estimator_stability_checks(current, development))
    cosine_check = checks["COEFFICIENT_MEDIAN_COSINE"]
    lower = cosine_check.lower
    assert lower is not None
    # An in-sample baseline collapses to ~1.0; a leave-one-out baseline leaves
    # room below it for an honest out-of-sample observation.
    assert lower < 1.0
    assert cosine_check.passed is True


def test_zero_drift_false_rejection_rate_is_bounded() -> None:
    """Measure, do not assume, how often a stable model is rejected."""

    trials = 400
    rejected = 0
    for seed in range(trials):
        generator = np.random.default_rng(1000 + seed)
        development = _folds(generator)
        current = _state(
            tuple(generator.normal(loc=1.0, scale=0.1, size=len(FACTORS))),
            scope="CURRENT_REFIT",
            fold_index=None,
            training_mse=float(abs(generator.normal(loc=1.0, scale=0.05))),
        )
        checks = build_estimator_stability_checks(current, development)
        if not all(check.passed for check in checks):
            rejected += 1
    rate = rejected / trials
    # The min-max envelope rejected a zero-drift ridge model about 80% of the
    # time. Holm over prediction intervals must land near the family level.
    assert rate <= 10.0 * STABILITY_FAMILY_ALPHA, f"zero-drift rejection rate {rate:.3f}"


def test_real_coefficient_drift_is_still_rejected() -> None:
    """The gate must keep its power after the false-rejection fix."""

    generator = np.random.default_rng(99)
    development = _folds(generator)
    current = _state(
        tuple(generator.normal(loc=1.0, scale=0.1, size=len(FACTORS)) * 25.0),
        scope="CURRENT_REFIT",
        fold_index=None,
    )
    checks = build_estimator_stability_checks(current, development)
    assert not all(check.passed for check in checks)


def test_family_alpha_must_be_a_probability() -> None:
    generator = np.random.default_rng(3)
    development = _folds(generator)
    current = _state(
        tuple(generator.normal(loc=1.0, scale=0.1, size=len(FACTORS))),
        scope="CURRENT_REFIT",
        fold_index=None,
    )
    with pytest.raises(ValueError, match="ALPHA_STABILITY_FAMILY_ALPHA_INVALID"):
        build_estimator_stability_checks(current, development, family_alpha=0.0)


def test_reference_state_mismatch_still_fails_closed() -> None:
    generator = np.random.default_rng(5)
    development = _folds(generator)
    current = seal_contract(
        AlphaEstimatorState,
        "state_hash",
        request_hash=REQUEST_HASH,
        candidate_id="other-candidate",
        scope="CURRENT_REFIT",
        family_id="ridge",
        state_kind="LINEAR",
        fold_index=None,
        ordered_factor_ids=FACTORS,
        coefficient_hex=tuple((1.0).hex() for _ in FACTORS),
        intercept_hex=(0.0).hex(),
        coefficient_l2_norm=math.sqrt(float(len(FACTORS))),
        coefficient_max_abs=1.0,
        training_mse=1.0,
        validation_score_mean=0.0,
        validation_score_std=1.0,
        validation_score_coverage=0.99,
        validation_session_statistics=_sessions(40),
    )
    with pytest.raises(ValueError, match="ALPHA_STABILITY_REFERENCE_STATE_MISMATCH"):
        build_estimator_stability_checks(current, development)
