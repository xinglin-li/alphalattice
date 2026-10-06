"""Requirement and regression tests for the Strategy Lab numerical owner."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import alphalattice.investment.portfolio_strategy_lab.optimizer.service as optimizer_module
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
    AlphaUtilityUnitsAdmission,
    PortfolioOptimizationError,
    PortfolioOptimizer,
    solve_with_cvxpy_oracle,
    stable_rank_buffered_top_k,
    stable_sector_feasible_top_k,
    stable_top_k,
    validate_covariance_lane,
)

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"


def _covariance(size: int, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(size, size))
    return values @ values.T / size + np.eye(size) * 0.2


def _admission(*, risk: float, turnover: float, cost: float = 0.001) -> AlphaUtilityUnitsAdmission:
    return AlphaUtilityUnitsAdmission.create(
        score_scale=1.0,
        risk_aversion=risk,
        turnover_regularization=turnover,
        transaction_cost_rate=cost,
        normalization_reference_id="FOCUSED_TEST_SCORE_UTILITY",
    )


def test_the_solver_binding_holds_no_thread_variable_of_the_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Binding plan B1c: the same solve gives the same bytes at any BLAS thread count, and a
    thread variable read into the Program's binding made it a different identity per shell."""

    before = optimizer_module.solver_numerical_environment()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "8")

    assert optimizer_module.solver_numerical_environment() == before

    # Nor the installed solver's version (LAWS.md ID6): the binding declares the version its
    # settings were validated with, and the run records the installed one beside it.
    monkeypatch.setattr(optimizer_module.osqp, "__version__", "0.0.0")
    assert optimizer_module.solver_numerical_environment() == before


def test_top_k_ties_are_deterministic_and_universe_size_is_not_fixed() -> None:
    scores = np.asarray([1.0, 2.0, 2.0, 0.0, 2.0])
    eligible = np.asarray([True, True, True, True, True])
    np.testing.assert_array_equal(stable_top_k(scores, eligible, 3), [1, 2, 4])

    selected = stable_top_k(np.linspace(-1.0, 1.0, 317), np.ones(317, dtype=np.bool_), 20)
    assert selected.shape == (20,)

    buffered = stable_rank_buffered_top_k(
        scores=np.asarray([6.0, 5.0, 4.0, 3.0, 2.0]),
        eligible=np.ones(5, dtype=np.bool_),
        previous_target_weights=np.asarray([0.0, 0.0, 0.0, 0.5, 0.5]),
        top_k=2,
        exit_rank=4,
    )
    # Intended target, not the marked-reference book, decides rank retention.
    np.testing.assert_array_equal(buffered, [0, 1, 3])


def test_rank_buffered_recipe_is_fixed_and_installed() -> None:
    from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
        build_installed_portfolio_policy_catalog,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.score_risk_cost import (
        RankBufferedScoreRiskCostDevelopmentRecipe,
        ScoreRiskCostDevelopmentRecipe,
    )

    capped = RankBufferedScoreRiskCostDevelopmentRecipe.create(
        exit_rank=150, maximum_one_way_turnover=0.2
    )
    assert capped.policy_id == "RANK_BUFFERED_SCORE_RISK_COST"
    assert capped.initial_deployment_disposition == "INITIAL_DEPLOYMENT_EXEMPT"
    assert build_installed_portfolio_policy_catalog().resolve(capped).policy_id == capped.policy_id
    with pytest.raises(ValueError, match="rank_buffered_score_risk_cost_recipe_invalid"):
        RankBufferedScoreRiskCostDevelopmentRecipe.create(
            exit_rank=200, maximum_one_way_turnover=0.25
        )

    historical = ScoreRiskCostDevelopmentRecipe.create(
        risk_aversion=100.0,
        turnover_regularization=10.0,
        transaction_cost_rate=0.0005,
        sector_capacity=None,
        score_scale=0.01,
        normalization_reference_id="SESSION_ROLE_NORMALIZED_TARGET_Z_SCORE",
    )
    assert historical.recipe_hash == (
        "72a5f8978431f745e0209f811063374e87fb07d114fdd4cab4a3857f50e415e2"
    )


def test_rank_buffered_carry_and_complete_turnover_cap_match_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = np.asarray([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    eligible = np.ones(scores.size, dtype=np.bool_)
    selected = stable_rank_buffered_top_k(
        scores=scores,
        eligible=eligible,
        previous_target_weights=np.asarray([0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]),
        top_k=3,
        exit_rank=5,
    )
    reference = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
    arguments = {
        "scores": scores,
        "covariance": np.eye(scores.size, dtype=np.float64) * 0.01,
        "reference_weights": reference,
        "decision_eligible": eligible,
        "top_k": 3,
        "maximum_weight": 1.0 / 3.0,
        "risk_aversion": 3.0,
        "turnover_regularization": 0.1,
        "alpha_utility_admission": _admission(risk=3.0, turnover=0.1),
        "selected_indices": selected,
        "liquidation_only_carry": True,
        "maximum_one_way_turnover": 0.2,
    }
    calls: list[object] = []
    original_warm_start = optimizer_module.osqp.OSQP.warm_start

    def record_warm_start(instance: object, *args: object, **kwargs: object) -> object:
        calls.append(instance)
        return original_warm_start(instance, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(optimizer_module.osqp.OSQP, "warm_start", record_warm_start)
        direct = PortfolioOptimizer().solve_score_risk_cost(**arguments)
    assert calls == []
    oracle = solve_with_cvxpy_oracle(**arguments)
    np.testing.assert_allclose(direct.weights, oracle, rtol=0.0, atol=1e-8)
    assert direct.weights[5] <= reference[5] + 1e-9
    assert direct.weights[6] <= reference[6] + 1e-9
    assert direct.predicted_one_way_turnover <= 0.2 + 1e-8

    from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
        BoundPolicyDecisionInput,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.score_risk_cost import (
        RankBufferedScoreRiskCostAdapter,
        RankBufferedScoreRiskCostDevelopmentRecipe,
    )

    policy_scores = np.linspace(2.0, -2.0, 200, dtype=np.float64)

    def policy_inputs(previous: np.ndarray | None) -> BoundPolicyDecisionInput:
        return BoundPolicyDecisionInput(
            scores=policy_scores,
            covariance=np.eye(policy_scores.size, dtype=np.float64) * 0.01,
            decision_eligible=np.ones(policy_scores.size, dtype=np.bool_),
            reference_weights=np.zeros(policy_scores.size, dtype=np.float64),
            sector_exposure_matrix=np.ones((1, policy_scores.size), dtype=np.float64),
            equal_weight_sector_exposure=np.ones(1, dtype=np.float64),
            previous_target_weights=previous,
        )

    adapter = RankBufferedScoreRiskCostAdapter()
    recipe = RankBufferedScoreRiskCostDevelopmentRecipe.create(
        exit_rank=150, maximum_one_way_turnover=0.2
    )
    initial = adapter.decide(
        policy=recipe,
        inputs=policy_inputs(None),
        optimizer=PortfolioOptimizer(),
    )
    assert float(np.abs(initial.target_weights).sum() / 2.0) > 0.2
    uncapped = RankBufferedScoreRiskCostDevelopmentRecipe.create(
        exit_rank=150, maximum_one_way_turnover=None
    )
    assert np.isclose(
        adapter.decide(
            policy=uncapped,
            inputs=policy_inputs(None),
            optimizer=PortfolioOptimizer(),
        ).target_weights.sum(),
        1.0,
    )
    broad_reference = np.full(policy_scores.size, 1.0 / policy_scores.size)
    assert np.isclose(
        adapter.decide(
            policy=uncapped,
            inputs=dataclasses.replace(policy_inputs(None), reference_weights=broad_reference),
            optimizer=PortfolioOptimizer(),
        ).target_weights.sum(),
        1.0,
    )
    with pytest.raises(PortfolioOptimizationError, match="turnover_cap_infeasible"):
        adapter.decide(
            policy=recipe,
            inputs=policy_inputs(np.zeros(policy_scores.size, dtype=np.float64)),
            optimizer=PortfolioOptimizer(),
        )


def test_wide_complete_cap_retains_261_name_reference_axis() -> None:
    """regression: complete cap semantics retain the broad carry axis before solve."""

    size = 454
    held = np.arange(222, dtype=np.int64)
    selected = np.concatenate((np.arange(88, dtype=np.int64), np.arange(222, 261)))
    reference = np.zeros(size, dtype=np.float64)
    reference[held] = 1.0 / held.size
    axis, active_reference, lower, upper, turnover_cap = optimizer_module._score_risk_cost_axis(
        scores=np.linspace(0.02, -0.02, size, dtype=np.float64),
        decision_eligible=np.ones(size, dtype=np.bool_),
        reference_weights=reference,
        selected=selected,
        top_k=100,
        maximum_weight=0.02,
        liquidation_only_carry=True,
        maximum_one_way_turnover=0.2,
        initial_deployment_exempt=False,
    )
    assert (held.size, selected.size, axis.size, turnover_cap) == (222, 127, 261, 0.2)
    assert np.all(active_reference >= lower)
    assert np.all(active_reference <= upper)


def test_validated_covariance_lane_keeps_direct_boundary_fail_closed() -> None:
    """requirement: owner validation removes repeated scans without accepting foreign input."""

    scores = np.asarray([3.0, 2.0, 1.0, 0.0], dtype=np.float64)
    lane = np.stack((_covariance(scores.size), _covariance(scores.size, seed=8)))
    validation = validate_covariance_lane(values=lane)
    arguments = {
        "scores": scores,
        "covariance": lane[0],
        "covariance_validation": validation,
        "reference_weights": np.zeros(scores.size, dtype=np.float64),
        "decision_eligible": np.ones(scores.size, dtype=np.bool_),
        "top_k": 4,
        "maximum_weight": 0.25,
        "risk_aversion": 3.0,
        "turnover_regularization": 0.1,
        "alpha_utility_admission": _admission(risk=3.0, turnover=0.1),
    }
    direct = PortfolioOptimizer().solve_score_risk_cost(**arguments)
    oracle = solve_with_cvxpy_oracle(**arguments)
    np.testing.assert_allclose(direct.weights, oracle, rtol=0.0, atol=1e-8)

    foreign = np.array(lane[0], copy=True)
    with pytest.raises(PortfolioOptimizationError, match="covariance_owner_mismatch"):
        PortfolioOptimizer().solve_score_risk_cost(**{**arguments, "covariance": foreign})
    foreign[0, 1] += 0.1
    with pytest.raises(PortfolioOptimizationError, match="covariance_not_symmetric"):
        PortfolioOptimizer().solve_score_risk_cost(
            **{
                key: value
                for key, value in arguments.items()
                if key not in {"covariance", "covariance_validation"}
            },
            covariance=foreign,
        )


def test_hard_sector_capacity_admits_signed_score_names_before_optimization() -> None:
    """regression: hard Sector bands use a deterministic feasible Top-K axis."""

    scores = np.asarray(
        [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, -1.0, -2.0, -3.0],
        dtype=np.float64,
    )
    sector_by_asset = np.asarray([0, 0, 0, 0, 0, 0, 1, 2, 3, 4])
    membership = np.eye(5, dtype=np.float64)[:, sector_by_asset]
    reference = np.zeros(scores.size, dtype=np.float64)
    eligible = np.ones(scores.size, dtype=np.bool_)
    sector_reference = np.asarray([0.4, 0.15, 0.15, 0.15, 0.15], dtype=np.float64)
    selected = stable_sector_feasible_top_k(
        scores=scores,
        eligible=eligible,
        top_k=5,
        maximum_weight=0.2,
        reference_weights=reference,
        sector_membership=membership,
        sector_reference=sector_reference,
        sector_capacity=0.15,
    )
    np.testing.assert_array_equal(selected, [0, 1, 6, 7, 8])
    assert scores[selected[-1]] < 0.0

    admission = _admission(risk=3.0, turnover=0.1)
    optimizer = PortfolioOptimizer()
    direct = optimizer.solve_score_risk_cost(
        scores=scores,
        covariance=np.eye(scores.size, dtype=np.float64) * 0.01,
        reference_weights=reference,
        decision_eligible=eligible,
        top_k=5,
        maximum_weight=0.2,
        risk_aversion=3.0,
        turnover_regularization=0.1,
        sector_matrix=membership,
        sector_reference=sector_reference,
        sector_capacity=0.15,
        alpha_utility_admission=admission,
    )
    oracle = solve_with_cvxpy_oracle(
        scores=scores,
        covariance=np.eye(scores.size, dtype=np.float64) * 0.01,
        reference_weights=reference,
        decision_eligible=eligible,
        top_k=5,
        maximum_weight=0.2,
        risk_aversion=3.0,
        turnover_regularization=0.1,
        sector_matrix=membership,
        sector_reference=sector_reference,
        sector_capacity=0.15,
        alpha_utility_admission=admission,
    )
    realized = membership @ direct.weights
    assert bool(np.all(realized >= sector_reference - 0.15 - 1e-6))
    assert bool(np.all(realized <= sector_reference + 0.15 + 1e-6))
    np.testing.assert_allclose(direct.weights, oracle, rtol=0.0, atol=1e-8)


def test_direct_osqp_matches_independent_cvxpy_oracle() -> None:
    """requirement: installed Direct OSQP agrees with its independent oracle."""

    size = 18
    rng = np.random.default_rng(11)
    scores = rng.normal(size=size)
    covariance = _covariance(size)
    reference = np.zeros(size)
    eligible = np.ones(size, dtype=np.bool_)
    optimizer = PortfolioOptimizer()
    direct = optimizer.solve_score_risk_cost(
        scores=scores,
        covariance=covariance,
        reference_weights=reference,
        decision_eligible=eligible,
        top_k=8,
        maximum_weight=0.2,
        risk_aversion=3.0,
        turnover_regularization=0.1,
        alpha_utility_admission=_admission(risk=3.0, turnover=0.1),
    )
    oracle = solve_with_cvxpy_oracle(
        scores=scores,
        covariance=covariance,
        reference_weights=reference,
        decision_eligible=eligible,
        top_k=8,
        maximum_weight=0.2,
        risk_aversion=3.0,
        turnover_regularization=0.1,
        alpha_utility_admission=_admission(risk=3.0, turnover=0.1),
    )
    np.testing.assert_allclose(direct.weights, oracle, rtol=0.0, atol=1e-8)
    # Legacy TOP_K_SCORE_RISK_COST remains the exact five-block sparse problem:
    # 1 budget row plus four capacity-sized blocks, with no infinite cap row.
    assert optimizer._templates[8].constraints.shape == (1 + 4 * 16, 32)
    assert not direct.weights.flags.writeable


def test_score_utility_requires_units_admission_and_preserves_negative_scores() -> None:
    """requirement: dimensionless Alpha is identity-scaled and never clipped at zero."""

    scores = np.asarray([-10.0, -4.0, -3.0, -2.0, -1.0], dtype=np.float64)
    arguments = {
        "scores": scores,
        "covariance": np.eye(5) * 1e-4,
        "reference_weights": np.zeros(5),
        "decision_eligible": np.ones(5, dtype=np.bool_),
        "top_k": 4,
        "maximum_weight": 0.4,
        "risk_aversion": 3.0,
        "turnover_regularization": 0.1,
    }
    with pytest.raises(PortfolioOptimizationError, match="ALPHA_UTILITY_UNITS_NOT_ADMITTED"):
        PortfolioOptimizer().solve_score_risk_cost(**arguments)
    result = PortfolioOptimizer().solve_score_risk_cost(
        **arguments,
        alpha_utility_admission=_admission(risk=3.0, turnover=0.1),
    )
    assert result.weights[0] == 0.0
    assert np.count_nonzero(result.weights) >= 3
    assert result.objective_audit.alpha_utility_term < 0.0
    assert result.objective_audit.sector_penalty_term is None
    assert result.objective_audit.utility_admission_hash is not None


def test_covariance_changes_admitted_objective_and_solution() -> None:
    """regression: covariance is a construction input, not a diagnostic-only lane."""

    scores = np.linspace(-0.2, 0.3, 8)
    common = {
        "scores": scores,
        "reference_weights": np.zeros(8),
        "decision_eligible": np.ones(8, dtype=np.bool_),
        "top_k": 8,
        "maximum_weight": 0.3,
        "risk_aversion": 10.0,
        "turnover_regularization": 0.01,
        "alpha_utility_admission": _admission(risk=10.0, turnover=0.01),
    }
    first = PortfolioOptimizer().solve_score_risk_cost(covariance=np.eye(8) * 0.001, **common)
    alternate = np.eye(8) * 0.001
    alternate[-1, -1] = 0.2
    second = PortfolioOptimizer().solve_score_risk_cost(covariance=alternate, **common)
    assert not np.allclose(first.weights, second.weights, rtol=0.0, atol=1e-8)
    assert first.objective_value != pytest.approx(second.objective_value, abs=1e-8)
    assert first.solver_call_count > 0
    assert second.solver_call_count > 0


def test_direct_osqp_retry_is_counted_and_residual_gated() -> None:
    """regression: an accuracy retry is evidence, and inaccurate output is bounded."""

    def result(status: str, residual: float) -> SimpleNamespace:
        return SimpleNamespace(
            info=SimpleNamespace(
                status=status,
                iter=40_000,
                prim_res=residual,
                dual_res=residual,
            ),
            x=np.ones(2),
            y=np.ones(2),
        )

    class FakeSolver:
        def __init__(self, final_residual: float, final_status: str = "solved inaccurate") -> None:
            self.final_residual = final_residual
            self.final_status = final_status
            self.calls = 0
            self.maximum_iterations = 20_000

        def solve(self, *, raise_error: bool) -> SimpleNamespace:
            assert raise_error is False
            self.calls += 1
            return (
                result("maximum iterations reached", 1e-5)
                if self.calls == 1
                else result(self.final_status, self.final_residual)
            )

        def update_settings(self, *, max_iter: int) -> None:
            self.maximum_iterations = max_iter

    admitted = FakeSolver(4e-9)
    _result, calls = optimizer_module._solve_direct_osqp(admitted)
    assert (calls, admitted.calls, admitted.maximum_iterations) == (2, 2, 40_000)
    bounded = FakeSolver(4e-9, "maximum iterations reached")
    _result, calls = optimizer_module._solve_direct_osqp(bounded)
    assert (calls, bounded.calls, bounded.maximum_iterations) == (2, 2, 40_000)
    with pytest.raises(PortfolioOptimizationError, match="solver_accuracy_not_admitted"):
        optimizer_module._solve_direct_osqp(FakeSolver(2e-8))


def test_owner_verification_failure_triggers_one_strict_solve() -> None:
    """regression: a solved iterate outside owner tolerance gets one strict solve."""

    class VerificationRetryTemplate:
        calls: list[bool]

        def __init__(self) -> None:
            self.calls = []

        def solve(self, **arguments: object) -> tuple[np.ndarray, int, np.ndarray, int]:
            strict_accuracy = bool(arguments.get("strict_accuracy", False))
            self.calls.append(strict_accuracy)
            weights = np.zeros(8, dtype=np.float64)
            weights[:4] = 0.25 if strict_accuracy else 0.250000005
            return weights, 7, np.zeros(1, dtype=np.float64), 1

    optimizer = PortfolioOptimizer()
    template = VerificationRetryTemplate()
    optimizer._templates[4] = template  # type: ignore[assignment]
    result = optimizer.solve_score_risk_cost(
        scores=np.arange(5, dtype=np.float64),
        covariance=np.eye(5, dtype=np.float64) * 0.01,
        reference_weights=np.zeros(5, dtype=np.float64),
        decision_eligible=np.ones(5, dtype=np.bool_),
        top_k=4,
        maximum_weight=0.4,
        risk_aversion=3.0,
        turnover_regularization=0.1,
        alpha_utility_admission=_admission(risk=3.0, turnover=0.1),
    )
    assert template.calls == [False, True]
    assert result.solver_call_count == 2
    assert float(result.weights.sum()) == pytest.approx(1.0, abs=1e-15)


def test_direct_osqp_builds_the_structure_once_and_the_solver_per_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirement: the cached thing is the structure, not the solver instance.

    One instance driven by repeated ``update`` calls was the earlier design and
    it is what made a campaign's completed trial set depend on execution order:
    see the history test below. What is still cached is the 2K sparse pattern,
    which is what is expensive to assemble and carries no state.
    """

    setup_calls = 0
    original_setup = optimizer_module.osqp.OSQP.setup

    def counted_setup(self: object, *args: object, **kwargs: object) -> object:
        nonlocal setup_calls
        setup_calls += 1
        return original_setup(self, *args, **kwargs)

    monkeypatch.setattr(optimizer_module.osqp.OSQP, "setup", counted_setup)
    size = 18
    optimizer = PortfolioOptimizer()
    covariance = _covariance(size)
    reference = np.zeros(size)
    eligible = np.ones(size, dtype=np.bool_)
    for offset in (0.0, 0.1):
        optimizer.solve_score_risk_cost(
            scores=np.linspace(-1.0, 1.0, size) + offset,
            covariance=covariance,
            reference_weights=reference,
            decision_eligible=eligible,
            top_k=8,
            maximum_weight=0.2,
            risk_aversion=3.0,
            turnover_regularization=0.1,
            alpha_utility_admission=_admission(risk=3.0, turnover=0.1),
        )
    assert setup_calls == 2
    # One structure, shared. Two solves against the same top_k must not have
    # rebuilt the pattern.
    assert list(optimizer._templates) == [8]


def test_a_solve_does_not_depend_on_the_solves_before_it() -> None:
    """regression: the same problem after a different history gave a different answer.

    Two runs of one campaign over the same sealed Program published 16 and 15
    solver non-convergences. The cause was a reused OSQP instance: state carried
    across ``update`` calls moved the answer in the last bit, and for a problem
    that needs most of its 20,000 iterations that is enough to flip whether it
    certifies. This pins the property the fix buys -- an answer that is a
    function of the problem alone -- rather than the fix itself.
    """

    size = 22
    covariance = _covariance(size)
    eligible = np.ones(size, dtype=np.bool_)
    subject = dict(
        scores=np.linspace(-1.0, 1.0, size),
        covariance=covariance,
        reference_weights=np.zeros(size),
        decision_eligible=eligible,
        top_k=9,
        maximum_weight=0.25,
        risk_aversion=4.0,
        turnover_regularization=0.05,
        alpha_utility_admission=_admission(risk=4.0, turnover=0.05),
    )

    cold = PortfolioOptimizer().solve_score_risk_cost(**subject)
    warmed = PortfolioOptimizer()
    for seed in range(5):
        warmed.solve_score_risk_cost(
            **{
                **subject,
                "scores": np.linspace(-1.0, 1.0, size) + 0.31 * (seed + 1),
                "covariance": _covariance(size, seed=seed + 3),
                "risk_aversion": 1.0 + seed,
                "alpha_utility_admission": _admission(risk=1.0 + seed, turnover=0.05),
            }
        )
    after = warmed.solve_score_risk_cost(**subject)

    # Bitwise, not close. A tolerance here would admit exactly the last-bit
    # drift that caused the defect.
    assert np.array_equal(cold.weights, after.weights)
    assert cold.objective_value == after.objective_value
    assert cold.predicted_variance == after.predicted_variance


def test_untradable_existing_holding_is_frozen_and_not_repaired() -> None:
    """requirement: safety constraints preserve unavailable held positions."""

    size = 12
    scores = np.linspace(-1.0, 1.0, size)
    reference = np.zeros(size)
    reference[0] = 0.1
    reference[1:10] = 0.1
    eligible = np.ones(size, dtype=np.bool_)
    eligible[0] = False
    result = PortfolioOptimizer().solve_score_risk_cost(
        scores=scores,
        covariance=_covariance(size),
        reference_weights=reference,
        decision_eligible=eligible,
        top_k=10,
        maximum_weight=0.1,
        risk_aversion=2.0,
        turnover_regularization=0.2,
        alpha_utility_admission=_admission(risk=2.0, turnover=0.2),
    )
    assert result.weights[0] == pytest.approx(0.1, abs=1e-9)
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_nonfinite_score_is_ignored_only_for_a_frozen_holding() -> None:
    """regression: missing mutable utility fails while frozen utility is irrelevant."""

    size = 12
    scores = np.linspace(-1.0, 1.0, size)
    scores[0] = np.nan
    reference = np.full(size, 0.1)
    reference[-2:] = 0.0
    eligible = np.ones(size, dtype=np.bool_)
    eligible[0] = False
    optimizer = PortfolioOptimizer()
    result = optimizer.solve_score_risk_cost(
        scores=scores,
        covariance=_covariance(size),
        reference_weights=reference,
        decision_eligible=eligible,
        top_k=10,
        maximum_weight=0.1,
        risk_aversion=2.0,
        turnover_regularization=0.2,
        alpha_utility_admission=_admission(risk=2.0, turnover=0.2),
    )
    oracle = solve_with_cvxpy_oracle(
        scores=scores,
        covariance=_covariance(size),
        reference_weights=reference,
        decision_eligible=eligible,
        top_k=10,
        maximum_weight=0.1,
        risk_aversion=2.0,
        turnover_regularization=0.2,
        alpha_utility_admission=_admission(risk=2.0, turnover=0.2),
    )
    assert result.weights[0] == pytest.approx(0.1, abs=1e-9)
    assert np.isfinite(result.objective_value)
    np.testing.assert_allclose(result.weights, oracle, rtol=0.0, atol=1e-8)


def test_infeasible_cap_fails_before_solver() -> None:
    with pytest.raises(PortfolioOptimizationError, match="cap_infeasible"):
        PortfolioOptimizer().solve_score_risk_cost(
            scores=np.arange(10, dtype=np.float64),
            covariance=np.eye(10),
            reference_weights=np.zeros(10),
            decision_eligible=np.ones(10, dtype=np.bool_),
            top_k=5,
            maximum_weight=0.1,
            risk_aversion=1.0,
            turnover_regularization=0.0,
        )


def test_minimum_variance_selects_by_score_but_ignores_score_in_objective() -> None:
    scores = np.asarray([10.0, 9.0, 1.0, 0.0])
    covariance = np.diag([4.0, 1.0, 0.01, 0.01])
    result = PortfolioOptimizer().minimum_variance(
        scores=scores,
        covariance=covariance,
        reference_weights=np.zeros(4),
        decision_eligible=np.ones(4, dtype=np.bool_),
        top_k=2,
        maximum_weight=0.8,
    )
    assert set(np.flatnonzero(result.weights)) == {0, 1}
    assert result.weights[1] > result.weights[0]


def test_score_risk_cost_binding_separates_score_admission_from_allocation() -> None:
    """A covariance-aware allocator is not a claim to select global Top-K."""

    from alphalattice.investment.portfolio_strategy_lab.policies.score_risk_cost import (
        RankBufferedScoreRiskCostAdapter,
        TopKScoreRiskCostAdapter,
    )

    top_k = TopKScoreRiskCostAdapter().describe_adapter_binding()
    buffered = RankBufferedScoreRiskCostAdapter().describe_adapter_binding()
    assert top_k.selection_allocation_semantics is not None
    assert buffered.selection_allocation_semantics is not None
    assert top_k.selection_allocation_semantics.score_preselection_method == (
        "STABLE_FINITE_SCORE_TOP_K"
    )
    assert buffered.selection_allocation_semantics.score_preselection_method == (
        "STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_BUFFER"
    )
    for binding in (top_k, buffered):
        semantics = binding.selection_allocation_semantics
        assert semantics is not None
        assert semantics.score_preselection_axis == "DECISION_ELIGIBLE_FINITE_SCORE_LISTING_AXIS"
        assert semantics.selection_count_source == "RECIPE_TOP_K"
        assert semantics.allocation_method == "CONVEX_SCORE_RISK_COST"
        assert semantics.optimizer_selection_claim == "OPTIMIZER_DOES_NOT_SELECT_GLOBAL_TOP_K"
    assert top_k.binding_hash != buffered.binding_hash
