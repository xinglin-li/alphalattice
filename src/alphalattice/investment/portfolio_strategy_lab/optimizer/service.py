"""Direct OSQP Portfolio optimizer with a CVXPY golden oracle."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import osqp  # type: ignore[import-untyped]
import scipy.sparse as sp  # type: ignore[import-untyped]

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

# Existing published problem shapes retain their measured feasibility tolerance.
# Golden comparison tolerances are separate and reported by the oracle evidence;
# objective values and weights never share one dimensionless magic tolerance.
_FEASIBILITY_WEIGHT_TOLERANCE = 1e-8
_SECTOR_FEASIBILITY_TOLERANCE = 1e-6
_DIRECT_OSQP_INITIAL_MAX_ITERATIONS = 20_000
_DIRECT_OSQP_ACCURACY_RETRY_MAX_ITERATIONS = 40_000
_DIRECT_OSQP_VERIFICATION_RETRY_TOLERANCE = 1e-10
DIRECT_OSQP_MAXIMUM_INVOCATIONS_PER_OPTIMIZATION = 2

_SOLUTION_VERIFICATION_ERRORS = frozenset(
    {
        "portfolio_strategy_lab.solution_nonfinite",
        "portfolio_strategy_lab.solution_negative_weight",
        "portfolio_strategy_lab.solution_budget_invalid",
        "portfolio_strategy_lab.solution_lower_bound_invalid",
        "portfolio_strategy_lab.solution_upper_bound_invalid",
        "portfolio_strategy_lab.band_violated",
        "portfolio_strategy_lab.exposure_guardrail_violated",
    }
)


class PortfolioOptimizationError(ValueError):
    """Stable fail-closed numerical boundary."""


_COVARIANCE_LANE_SEAL = object()


class CovarianceValidationProof(Protocol):
    """Structural proof that a solver view belongs to an immutable owner lane."""

    @property
    def values(self) -> FloatArray: ...

    def owns(self, covariance: FloatArray) -> bool: ...


@dataclass(frozen=True, slots=True)
class ValidatedCovarianceLane:
    """One immutable 2-D or 3-D covariance owner lane admitted before solving."""

    values: FloatArray
    _seal: object = field(repr=False)

    def owns(self, covariance: FloatArray) -> bool:
        """Whether this exact solver view belongs to the owner-validated lane."""
        return bool(
            self._seal is _COVARIANCE_LANE_SEAL
            and covariance.shape == self.values.shape[-2:]
            and np.shares_memory(covariance, self.values)
        )


def validate_covariance_lane(*, values: FloatArray) -> ValidatedCovarianceLane:
    """Validate and freeze a covariance owner lane once, before solver use.

    The solver receives one 2-D session view at a time. A caller that owns a
    complete immutable lane pays the symmetry scan once here and then supplies
    this typed proof to each solve. Direct callers remain fail-closed through
    ``_validate_inputs`` below, which validates their individual matrix.
    """
    if (
        values.ndim not in {2, 3}
        or values.shape[-1] != values.shape[-2]
        or not np.isfinite(values).all()
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.optimizer_input_invalid")
    matrices = (values,) if values.ndim == 2 else values
    if any(not np.allclose(value, value.T, rtol=0.0, atol=1e-12) for value in matrices):
        raise PortfolioOptimizationError("portfolio_strategy_lab.covariance_not_symmetric")
    if values.flags.writeable:
        values.setflags(write=False)
    return ValidatedCovarianceLane(values=values, _seal=_COVARIANCE_LANE_SEAL)


def _solve_direct_osqp(solver: osqp.OSQP, *, allow_accuracy_retry: bool = True) -> tuple[Any, int]:
    """Solve once, then continue a bounded OSQP convergence result.

    The installed CVXPY/OSQP golden route already admits 40,000 iterations.
    Direct OSQP starts at the existing 20,000 limit and may use that same
    measured owner limit for one accuracy retry after either ``solved
    inaccurate`` or ``maximum iterations reached``.  A bounded terminal iterate
    is admitted only inside the installed residual envelope below; the caller
    then re-verifies weights and every constraint.  Both numerical invocations
    are returned for exact evidence.
    """

    try:
        result = solver.solve(raise_error=False)
        solver_call_count = 1
        if allow_accuracy_retry and result.info.status in {
            "solved inaccurate",
            "maximum iterations reached",
        }:
            solver.update_settings(max_iter=_DIRECT_OSQP_ACCURACY_RETRY_MAX_ITERATIONS)
            result = solver.solve(raise_error=False)
            solver_call_count += 1
    except Exception as error:
        raise PortfolioOptimizationError(
            "portfolio_strategy_lab.solver_execution_failed"
        ) from error
    residuals = np.asarray([result.info.prim_res, result.info.dual_res], dtype=np.float64)
    bounded_terminal_status = result.info.status in {
        "solved inaccurate",
        "maximum iterations reached",
    }
    bounded_within_installed_residual_envelope = (
        bounded_terminal_status
        and np.isfinite(residuals).all()
        and bool(np.all(np.abs(residuals) <= _FEASIBILITY_WEIGHT_TOLERANCE))
    )
    if (
        (result.info.status != "solved" and not bounded_within_installed_residual_envelope)
        or result.x is None
        or result.y is None
    ):
        diagnostic = RuntimeError(
            "status="
            f"{result.info.status};iterations={result.info.iter};"
            f"primal_residual={result.info.prim_res};dual_residual={result.info.dual_res}"
        )
        code = (
            "portfolio_strategy_lab.solver_accuracy_not_admitted"
            if bounded_terminal_status
            else "portfolio_strategy_lab.solver_not_solved"
        )
        raise PortfolioOptimizationError(code) from diagnostic
    return result, solver_call_count


@dataclass(frozen=True, slots=True)
class AlphaUtilityUnitsAdmission:
    """Bind dimensionless score scaling to exact Risk, turnover and cost parameters."""

    method_id: Literal["DIMENSIONLESS_SCORE_UTILITY_IDENTITY_BOUND"]
    alpha_unit: Literal["DIMENSIONLESS_SCORE"]
    risk_unit: Literal["RAW_SIMPLE_RETURN_SQUARED"]
    cost_unit: Literal["RAW_SIMPLE_RETURN"]
    score_scale: float
    risk_aversion: float
    turnover_regularization: float
    transaction_cost_rate: float
    normalization_reference_id: str
    admission_hash: str

    @classmethod
    def create(
        cls,
        *,
        score_scale: float,
        risk_aversion: float,
        turnover_regularization: float,
        transaction_cost_rate: float,
        normalization_reference_id: str,
    ) -> AlphaUtilityUnitsAdmission:
        """Admit finite utility parameters and seal their units/reference binding.

        Args:
            score_scale: Positive dimensionless score multiplier.
            risk_aversion: Positive covariance penalty multiplier.
            turnover_regularization: Nonnegative squared holdings-change penalty.
            transaction_cost_rate: Nonnegative raw-return cost rate.
            normalization_reference_id: Nonempty identity of the score normalization reference.

        Returns:
            Exact utility admission with canonical parameter/unit identity.

        Raises:
            PortfolioOptimizationError: A parameter is nonfinite/outside its range or the
                normalization reference is empty.
        """
        values = {
            "method_id": "DIMENSIONLESS_SCORE_UTILITY_IDENTITY_BOUND",
            "alpha_unit": "DIMENSIONLESS_SCORE",
            "risk_unit": "RAW_SIMPLE_RETURN_SQUARED",
            "cost_unit": "RAW_SIMPLE_RETURN",
            "score_scale": score_scale,
            "risk_aversion": risk_aversion,
            "turnover_regularization": turnover_regularization,
            "transaction_cost_rate": transaction_cost_rate,
            "normalization_reference_id": normalization_reference_id,
        }
        if (
            not normalization_reference_id
            or not np.isfinite(
                [score_scale, risk_aversion, turnover_regularization, transaction_cost_rate]
            ).all()
            or score_scale <= 0.0
            or risk_aversion <= 0.0
            or turnover_regularization < 0.0
            or transaction_cost_rate < 0.0
        ):
            raise PortfolioOptimizationError("ALPHA_UTILITY_UNITS_NOT_ADMITTED")
        return cls(
            method_id="DIMENSIONLESS_SCORE_UTILITY_IDENTITY_BOUND",
            alpha_unit="DIMENSIONLESS_SCORE",
            risk_unit="RAW_SIMPLE_RETURN_SQUARED",
            cost_unit="RAW_SIMPLE_RETURN",
            score_scale=score_scale,
            risk_aversion=risk_aversion,
            turnover_regularization=turnover_regularization,
            transaction_cost_rate=transaction_cost_rate,
            normalization_reference_id=normalization_reference_id,
            admission_hash=str(canonical_hash(values)),
        )


@dataclass(frozen=True, slots=True)
class PortfolioObjectiveAudit:
    """Retain objective terms and paired sector-constraint displacement diagnostics."""

    alpha_utility_term: float
    risk_penalty_term: float
    transaction_cost_term: float
    turnover_regularization_term: float
    sector_penalty_term: float | None
    reference_to_sector_unconstrained_l1_distance: float
    reference_to_final_l1_distance: float
    sector_unconstrained_to_final_l1_distance: float
    predicted_variance: float
    expected_one_way_turnover: float
    sector_capacity: float | None
    sector_lower_slack: tuple[float, ...]
    sector_upper_slack: tuple[float, ...]
    sector_dual: tuple[float, ...]
    binding_sector_count: int
    desired_sector_exposure: tuple[float, ...]
    final_sector_exposure: tuple[float, ...]
    displaced_name_count: int
    displaced_weight_mass: float
    pre_constraint_score_utility: float
    post_constraint_score_utility: float
    pre_constraint_predicted_variance: float
    post_constraint_predicted_variance: float
    utility_admission_hash: str | None


@dataclass(frozen=True, slots=True)
class PortfolioGoldenTolerances:
    """Dimension-specific tolerances measured on a calibration subset."""

    maximum_weight_error: float
    objective_error: float
    feasibility_residual: float
    calibration_observation_count: int


def measure_golden_tolerances(
    *,
    weight_errors: tuple[float, ...],
    objective_errors: tuple[float, ...],
    feasibility_residuals: tuple[float, ...],
) -> PortfolioGoldenTolerances:
    """Freeze held-out golden tolerances from measured Direct/CVXPY drift.

    Floors come from the installed feasibility owner and floating-point scale;
    weights, objective utility and feasibility therefore never share one
    dimensionless magic constant.
    """
    if (
        not weight_errors
        or len(objective_errors) != len(weight_errors)
        or len(feasibility_residuals) != len(weight_errors)
        or not np.isfinite((*weight_errors, *objective_errors, *feasibility_residuals)).all()
        or min((*weight_errors, *objective_errors, *feasibility_residuals)) < 0.0
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.golden_measurement_invalid")
    return PortfolioGoldenTolerances(
        maximum_weight_error=max(_FEASIBILITY_WEIGHT_TOLERANCE, 4.0 * max(weight_errors)),
        objective_error=max(float(np.finfo(np.float64).eps * 1_000.0), 4.0 * max(objective_errors)),
        feasibility_residual=max(_SECTOR_FEASIBILITY_TOLERANCE, 4.0 * max(feasibility_residuals)),
        calibration_observation_count=len(weight_errors),
    )


def score_risk_cost_objective(
    *,
    weights: FloatArray,
    scores: FloatArray,
    covariance: FloatArray,
    reference_weights: FloatArray,
    score_scale: float,
    risk_aversion: float,
    turnover_regularization: float,
    transaction_cost_rate: float,
) -> float:
    """One unit-admitted objective value, shared by Direct and golden evidence."""
    if any(value.shape != weights.shape for value in (scores, reference_weights)) or (
        covariance.shape != (weights.size, weights.size)
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.oracle_axis_invalid")
    turnover = float(np.abs(weights - reference_weights).sum() / 2.0)
    return float(
        score_scale * (scores @ weights)
        - risk_aversion * (weights @ covariance @ weights)
        - transaction_cost_rate * turnover
        - turnover_regularization * np.square(weights - reference_weights).sum()
    )


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationResult:
    """Retain verified weights, selected support, solver counts and objective diagnostics."""

    weights: FloatArray
    selected_indices: npt.NDArray[np.int64]
    objective_value: float
    predicted_variance: float
    predicted_one_way_turnover: float
    solver_iterations: int
    solver_call_count: int
    objective_audit: PortfolioObjectiveAudit
    sector_unconstrained_weights: FloatArray


_OBJECTIVE_SCALE_FLOOR = 1e-12


def _objective_scale(
    *, hessian: FloatArray, local_scores: FloatArray, transaction_cost_rate: float
) -> float:
    """A positive factor that puts this problem in the range the tolerance means.

    The nine parameter points Stage 6 lost to ``solver_execution_failed`` were
    not infeasible and were not badly posed. They were *small*: with
    ``risk_aversion`` at ten and covariance entries around ``1e-4``, the
    quadratic block is order ``1e-3`` while the linear block carries scores
    around ``1e-2``, and OSQP was being asked to certify that near-LP to
    ``eps_abs = eps_rel = 1e-9`` inside twenty thousand iterations. An absolute
    tolerance is a statement about the units of the objective, and nobody had
    told it what those units were.

    Scaling ``(P, q)`` by any positive constant leaves the argmin exactly where
    it was -- the feasible set is untouched and the objective is multiplied
    through -- so this changes which problems converge and not which answer they
    converge to. The reported objective is divided back out.
    """

    magnitudes = (
        float(np.max(np.abs(np.diagonal(hessian)))),
        float(np.max(np.abs(local_scores))),
        abs(float(transaction_cost_rate)),
    )
    largest = max(magnitudes)
    if not math.isfinite(largest) or largest <= _OBJECTIVE_SCALE_FLOOR:
        return 1.0
    return 1.0 / largest


SOLVER_SETTINGS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "verbose": False,
        "warm_starting": True,
        "polishing": True,
        "eps_abs": 1e-9,
        "eps_rel": 1e-9,
        "max_iter": _DIRECT_OSQP_INITIAL_MAX_ITERATIONS,
    }
)
"""Every setting the installed solver runs under, in one place.

Named rather than repeated at each ``setup`` call because it is part of the
numerical environment the capability declares: two runs are the same experiment
only if the solver saw the same tolerances and the same iteration budget, and a
tolerance that lived in two literals could differ in one of them.
"""


SOLVER_VALIDATED_VERSION: Final = "1.1.3"
"""The OSQP version these settings were validated with: a declaration, like a model
card's package, never the installed version read at run time. The installed version
is recorded beside each run (the Portfolio runtime's environment) and compared with
nothing, so an upgrade moves no identity (LAWS.md ID6); a re-validation that changes
this value is a recorded move."""


def solver_numerical_environment() -> dict[str, object]:
    """What decides a solve besides its own inputs, as a sealed payload.

    Solver, the version its settings were validated with, settings and the
    solver's instance lifetime. The process's thread environment is not here: it
    decides no number (the same solve gives the same bytes with OMP/OPENBLAS/MKL
    thread counts unset, 1 or 8, checked in fresh processes), and read into this
    payload it made the same solve a different identity in every shell (binding
    plan, B1c).
    """
    return {
        "solver": "osqp",
        "solver_version": SOLVER_VALIDATED_VERSION,
        "settings": dict(SOLVER_SETTINGS),
        "accuracy_retry": {
            "statuses": ("solved inaccurate", "maximum iterations reached"),
            "maximum_invocations": DIRECT_OSQP_MAXIMUM_INVOCATIONS_PER_OPTIMIZATION,
            "retry_max_iter": _DIRECT_OSQP_ACCURACY_RETRY_MAX_ITERATIONS,
        },
        "verification_retry": {
            "trigger": "OWNER_WEIGHT_OR_CONSTRAINT_VERIFICATION_FAILURE",
            "maximum_total_invocations": DIRECT_OSQP_MAXIMUM_INVOCATIONS_PER_OPTIMIZATION,
            "eps_abs": _DIRECT_OSQP_VERIFICATION_RETRY_TOLERANCE,
            "eps_rel": _DIRECT_OSQP_VERIFICATION_RETRY_TOLERANCE,
            "max_iter": _DIRECT_OSQP_ACCURACY_RETRY_MAX_ITERATIONS,
            "weight_repair": "NONE",
        },
        "instance_lifetime": "ONE_SOLVER_PER_SOLVE",
    }


@dataclass(slots=True)
class _OSQPTemplate:
    """One fixed 2K sparse structure reused across a trial's formations.

    The *structure* is cached and the *solver* is not, and that split is the
    reproducibility fix. One OSQP instance driven by repeated ``update`` calls
    carries state across problems: the same problem solved after a different
    sequence of predecessors came back differing in the last bit, which is
    enough to flip whether a marginal problem certifies inside ``max_iter``. A
    campaign then published 16 solver non-convergences on one execution and 15
    on another over the same sealed Program -- an experiment whose completed
    trial set depended on the order units happened to run in.

    Building the instance per solve costs roughly 2.2x on the solve itself and
    makes the answer a function of the problem alone. That is the trade this
    package takes, because a cheaper number that is not reproducible is not
    cheaper.
    """

    capacity: int
    problem: sp.csc_matrix
    constraints: sp.csc_matrix
    p_rows: npt.NDArray[np.int32]
    p_columns: npt.NDArray[np.int64]
    constraint_count: int
    includes_turnover_cap: bool = False

    def solve(
        self,
        *,
        hessian: FloatArray,
        local_scores: FloatArray,
        reference: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        transaction_cost_rate: float,
        maximum_one_way_turnover: float | None = None,
        strict_accuracy: bool = False,
    ) -> tuple[FloatArray, int, FloatArray, int]:
        capacity = self.capacity
        if any(value.shape != (capacity, capacity) for value in (hessian,)) or any(
            value.shape != (capacity,) for value in (local_scores, reference, lower, upper)
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.template_axis_invalid")
        scale = _objective_scale(
            hessian=hessian,
            local_scores=local_scores,
            transaction_cost_rate=transaction_cost_rate,
        )
        p_values = 2.0 * scale * hessian[self.p_rows, self.p_columns]
        q = scale * np.concatenate(
            (
                -local_scores,
                np.full(capacity, transaction_cost_rate / 2.0, dtype=np.float64),
            )
        )
        if self.includes_turnover_cap != (maximum_one_way_turnover is not None):
            raise PortfolioOptimizationError(
                "portfolio_strategy_lab.template_turnover_cap_mismatch"
            )
        lower_bounds = np.concatenate(
            (
                np.asarray([1.0]),
                lower,
                np.full(capacity, -np.inf),
                np.full(capacity, -np.inf),
                np.zeros(capacity),
            )
        )
        upper_bounds = np.concatenate(
            (
                np.asarray([1.0]),
                upper,
                reference,
                -reference,
                np.full(capacity, np.inf),
            )
        )
        if self.includes_turnover_cap:
            assert maximum_one_way_turnover is not None
            lower_bounds = np.concatenate((lower_bounds, np.asarray([-np.inf])))
            upper_bounds = np.concatenate(
                (upper_bounds, np.asarray([2.0 * maximum_one_way_turnover]))
            )
        try:
            solver = osqp.OSQP()
            problem = self.problem.copy()
            problem.data[:] = p_values
            settings = dict(SOLVER_SETTINGS)
            if strict_accuracy:
                settings.update(
                    {
                        "eps_abs": _DIRECT_OSQP_VERIFICATION_RETRY_TOLERANCE,
                        "eps_rel": _DIRECT_OSQP_VERIFICATION_RETRY_TOLERANCE,
                        "max_iter": _DIRECT_OSQP_ACCURACY_RETRY_MAX_ITERATIONS,
                    }
                )
            solver.setup(
                P=problem,
                q=q,
                A=self.constraints,
                l=lower_bounds,
                u=upper_bounds,
                **settings,
            )
            if not self.includes_turnover_cap:
                solver.warm_start(
                    x=np.concatenate((reference, np.zeros(capacity, dtype=np.float64))),
                    y=np.zeros(self.constraint_count, dtype=np.float64),
                )
            result, solver_call_count = _solve_direct_osqp(
                solver, allow_accuracy_retry=not strict_accuracy
            )
        except Exception as error:
            if isinstance(error, PortfolioOptimizationError):
                raise
            raise PortfolioOptimizationError(
                "portfolio_strategy_lab.solver_execution_failed"
            ) from error
        return (
            np.asarray(result.x[:capacity], dtype=np.float64),
            int(result.info.iter),
            np.asarray(result.y, dtype=np.float64),
            solver_call_count,
        )


def _build_osqp_template(capacity: int) -> _OSQPTemplate:
    variable_count = capacity * 2
    weight_pattern = sp.triu(
        sp.csc_matrix(np.ones((capacity, capacity), dtype=np.float64)),
        format="csc",
    )
    p = sp.block_diag(
        (weight_pattern, sp.csc_matrix((capacity, capacity))),
        format="csc",
    )
    p_columns: npt.NDArray[np.int64] = np.asarray(
        np.repeat(np.arange(variable_count), np.diff(p.indptr)), dtype=np.int64
    )
    identity = sp.eye(capacity, format="csc")
    zeros = sp.csc_matrix((capacity, capacity))
    ones = sp.csc_matrix(np.ones((1, capacity), dtype=np.float64))
    constraints = sp.vstack(
        (
            sp.hstack((ones, sp.csc_matrix((1, capacity))), format="csc"),
            sp.hstack((identity, zeros), format="csc"),
            sp.hstack((identity, -identity), format="csc"),
            sp.hstack((-identity, -identity), format="csc"),
            sp.hstack((zeros, identity), format="csc"),
        ),
        format="csc",
    )
    # No solver here. The structure is what is expensive to build and safe to
    # share; the instance is what carries state between problems and is built
    # per solve.
    return _OSQPTemplate(
        capacity=capacity,
        problem=p,
        constraints=constraints,
        p_rows=np.asarray(p.indices, dtype=np.int32),
        p_columns=np.asarray(p_columns, dtype=np.int64),
        constraint_count=constraints.shape[0],
    )


def _build_capped_osqp_template(capacity: int) -> _OSQPTemplate:
    """Build the successor-only cap row without changing the legacy problem."""

    unbanded = _build_osqp_template(capacity)
    constraints = sp.vstack(
        (
            unbanded.constraints,
            sp.hstack(
                (sp.csc_matrix((1, capacity)), np.ones((1, capacity), dtype=np.float64)),
                format="csc",
            ),
        ),
        format="csc",
    )
    return _OSQPTemplate(
        capacity=capacity,
        problem=unbanded.problem,
        constraints=constraints,
        p_rows=unbanded.p_rows,
        p_columns=unbanded.p_columns,
        constraint_count=constraints.shape[0],
        includes_turnover_cap=True,
    )


@dataclass(slots=True)
class _BandedTemplate:
    """One fixed sparse structure that also carries Sector bands and beta.

    Separate from ``_OSQPTemplate`` rather than an extension of it, because the
    unbanded problem is what already-published Portfolio evidence was solved
    against. Adding rows to that template -- even rows with infinite bounds --
    changes the problem OSQP is handed and can move the last bits of a solution
    nothing else in the graph would notice had moved.

    The Sector and beta blocks are stored **dense**. OSQP can update constraint
    values but not sparsity, and the active axis changes every formation, so a
    sparsity pattern derived from one formation's Sector membership would be
    wrong at the next one. A dense ``sector_count x capacity`` block is a few
    thousand nonzeros and is the honest way to keep one structure.
    """

    top_k: int
    sector_count: int
    problem: sp.csc_matrix
    constraints: sp.csc_matrix
    p_rows: npt.NDArray[np.int32]
    p_columns: npt.NDArray[np.int64]
    constraint_count: int
    a_rows: npt.NDArray[np.int32]
    a_columns: npt.NDArray[np.int64]

    @property
    def capacity(self) -> int:
        return self.top_k * 2

    def solve(
        self,
        *,
        hessian: FloatArray,
        local_scores: FloatArray,
        reference: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
        transaction_cost_rate: float,
        sector_membership: FloatArray,
        sector_lower: FloatArray,
        sector_upper: FloatArray,
        exposure: FloatArray,
        exposure_lower: float,
        exposure_upper: float,
        strict_accuracy: bool = False,
    ) -> tuple[FloatArray, int, FloatArray, int]:
        capacity = self.capacity
        if (
            hessian.shape != (capacity, capacity)
            or any(value.shape != (capacity,) for value in (local_scores, reference, lower, upper))
            or sector_membership.shape != (self.sector_count, capacity)
            or sector_lower.shape != (self.sector_count,)
            or sector_upper.shape != (self.sector_count,)
            or exposure.shape != (capacity,)
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.template_axis_invalid")
        scale = _objective_scale(
            hessian=hessian,
            local_scores=local_scores,
            transaction_cost_rate=transaction_cost_rate,
        )
        p_values = 2.0 * scale * hessian[self.p_rows, self.p_columns]
        q = scale * np.concatenate(
            (
                -local_scores,
                np.full(capacity, transaction_cost_rate / 2.0, dtype=np.float64),
            )
        )
        lower_bounds = np.concatenate(
            (
                np.asarray([1.0]),
                lower,
                np.full(capacity, -np.inf),
                np.full(capacity, -np.inf),
                np.zeros(capacity),
                sector_lower,
                np.asarray([exposure_lower]),
            )
        )
        upper_bounds = np.concatenate(
            (
                np.asarray([1.0]),
                upper,
                reference,
                -reference,
                np.full(capacity, np.inf),
                sector_upper,
                np.asarray([exposure_upper]),
            )
        )
        # The Sector and beta rows are the only ones whose *values* move, but
        # OSQP takes the whole data vector in the assembled matrix's own CSC
        # order -- which is column-major and interleaves every block. Building
        # the dense matrix and indexing it with the stored pattern is the same
        # trick ``P`` already uses, and it is the one that cannot silently
        # permute a constraint into a different row.
        dense: FloatArray = np.zeros((self.constraint_count, capacity * 2), dtype=np.float64)
        dense[0, :capacity] = 1.0
        rows = np.arange(capacity)
        dense[1 + rows, rows] = 1.0
        dense[1 + capacity + rows, rows] = 1.0
        dense[1 + capacity + rows, capacity + rows] = -1.0
        dense[1 + capacity * 2 + rows, rows] = -1.0
        dense[1 + capacity * 2 + rows, capacity + rows] = -1.0
        dense[1 + capacity * 3 + rows, capacity + rows] = 1.0
        sector_start = 1 + capacity * 4
        dense[sector_start : sector_start + self.sector_count, :capacity] = sector_membership
        dense[sector_start + self.sector_count, :capacity] = exposure
        constraint_values = dense[self.a_rows, self.a_columns]
        try:
            solver = osqp.OSQP()
            problem = self.problem.copy()
            problem.data[:] = p_values
            assembled = self.constraints.copy()
            assembled.data[:] = constraint_values
            settings = dict(SOLVER_SETTINGS)
            if strict_accuracy:
                settings.update(
                    {
                        "eps_abs": _DIRECT_OSQP_VERIFICATION_RETRY_TOLERANCE,
                        "eps_rel": _DIRECT_OSQP_VERIFICATION_RETRY_TOLERANCE,
                        "max_iter": _DIRECT_OSQP_ACCURACY_RETRY_MAX_ITERATIONS,
                    }
                )
            solver.setup(
                P=problem,
                q=q,
                A=assembled,
                l=lower_bounds,
                u=upper_bounds,
                **settings,
            )
            solver.warm_start(
                x=np.concatenate((reference, np.zeros(capacity, dtype=np.float64))),
                y=np.zeros(self.constraint_count, dtype=np.float64),
            )
            result, solver_call_count = _solve_direct_osqp(
                solver, allow_accuracy_retry=not strict_accuracy
            )
        except Exception as error:
            if isinstance(error, PortfolioOptimizationError):
                raise
            raise PortfolioOptimizationError(
                "portfolio_strategy_lab.solver_execution_failed"
            ) from error
        return (
            np.asarray(result.x[:capacity], dtype=np.float64),
            int(result.info.iter),
            np.asarray(result.y, dtype=np.float64),
            solver_call_count,
        )


def _build_banded_template(top_k: int, sector_count: int) -> _BandedTemplate:
    capacity = top_k * 2
    variable_count = capacity * 2
    weight_pattern = sp.triu(
        sp.csc_matrix(np.ones((capacity, capacity), dtype=np.float64)), format="csc"
    )
    p = sp.block_diag((weight_pattern, sp.csc_matrix((capacity, capacity))), format="csc")
    p_columns: npt.NDArray[np.int64] = np.asarray(
        np.repeat(np.arange(variable_count), np.diff(p.indptr)), dtype=np.int64
    )
    identity = sp.eye(capacity, format="csc")
    zeros = sp.csc_matrix((capacity, capacity))
    ones = sp.csc_matrix(np.ones((1, capacity), dtype=np.float64))
    dense_sector = sp.csc_matrix(np.ones((sector_count, capacity), dtype=np.float64))
    dense_beta = sp.csc_matrix(np.ones((1, capacity), dtype=np.float64))
    constraints = sp.vstack(
        (
            sp.hstack((ones, sp.csc_matrix((1, capacity))), format="csc"),
            sp.hstack((identity, zeros), format="csc"),
            sp.hstack((identity, -identity), format="csc"),
            sp.hstack((-identity, -identity), format="csc"),
            sp.hstack((zeros, identity), format="csc"),
            sp.hstack((dense_sector, sp.csc_matrix((sector_count, capacity))), format="csc"),
            sp.hstack((dense_beta, sp.csc_matrix((1, capacity))), format="csc"),
        ),
        format="csc",
    )
    row_count = int(constraints.shape[0])
    a_columns: npt.NDArray[np.int64] = np.asarray(
        np.repeat(np.arange(variable_count), np.diff(constraints.indptr)), dtype=np.int64
    )
    return _BandedTemplate(
        top_k=top_k,
        sector_count=sector_count,
        problem=p,
        constraints=constraints,
        p_rows=np.asarray(p.indices, dtype=np.int32),
        p_columns=np.asarray(p_columns, dtype=np.int64),
        constraint_count=row_count,
        a_rows=np.asarray(constraints.indices, dtype=np.int32),
        a_columns=a_columns,
    )


def stable_top_k(scores: FloatArray, eligible: BoolArray, top_k: int) -> npt.NDArray[np.int64]:
    """Select finite eligible scores with stable position-based tie breaking.

    Args:
        scores: One-dimensional scores on the listing axis.
        eligible: Matching decision eligibility mask.
        top_k: Positive requested selection size.

    Returns:
        Int64 positions ordered by descending score then original axis position.

    Raises:
        PortfolioOptimizationError: Axis shapes/count are invalid or finite eligible support is
            insufficient.
    """
    if scores.ndim != 1 or eligible.shape != scores.shape or top_k < 1:
        raise PortfolioOptimizationError("portfolio_strategy_lab.selection_input_invalid")
    positions = np.flatnonzero(eligible & np.isfinite(scores))
    if positions.size < top_k:
        raise PortfolioOptimizationError("portfolio_strategy_lab.selection_pool_insufficient")
    order = np.lexsort((positions, -scores[positions]))
    return np.asarray(positions[order[:top_k]], dtype=np.int64)


def stable_rank_buffered_top_k(
    *,
    scores: FloatArray,
    eligible: BoolArray,
    previous_target_weights: FloatArray | None,
    top_k: int,
    exit_rank: int,
) -> npt.NDArray[np.int64]:
    """Keep intended eligible holdings through ``exit_rank`` beside the entry pool."""
    entry = stable_top_k(scores, eligible, top_k)
    if exit_rank < top_k or exit_rank > scores.size:
        raise PortfolioOptimizationError("portfolio_strategy_lab.rank_buffer_invalid")
    if previous_target_weights is None:
        return entry
    if (
        previous_target_weights.shape != scores.shape
        or not np.isfinite(previous_target_weights).all()
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.rank_buffer_intent_invalid")
    positions = np.flatnonzero(eligible & np.isfinite(scores))
    order = positions[np.lexsort((positions, -scores[positions]))]
    held = previous_target_weights > _FEASIBILITY_WEIGHT_TOLERANCE
    retained = held[order[:exit_rank]]
    return np.asarray(
        order[:exit_rank][np.isin(order[:exit_rank], entry) | retained], dtype=np.int64
    )


def stable_sector_feasible_top_k(
    *,
    scores: FloatArray,
    eligible: BoolArray,
    top_k: int,
    maximum_weight: float,
    reference_weights: FloatArray,
    sector_membership: FloatArray,
    sector_reference: FloatArray,
    sector_capacity: float,
) -> npt.NDArray[np.int64]:
    """Admit the best signed-score Top-K that can satisfy hard Sector floors.

    Sector capacity is a portfolio constraint, not a post-optimization repair.
    A globally truncated Top-K may nevertheless omit the names needed to make a
    valid hard floor reachable.  This deterministic admission keeps exactly K
    decision-eligible names, adds the highest-scoring missing-sector name, and
    removes the lowest-scoring name whose Sector remains above its minimum
    count.  The optimizer still proves the complete band system afterwards.
    """
    selected = stable_top_k(scores, eligible, top_k)
    asset_count = scores.size
    if (
        maximum_weight <= 0.0
        or sector_capacity <= 0.0
        or sector_membership.ndim != 2
        or sector_membership.shape[1] != asset_count
        or sector_reference.shape != (sector_membership.shape[0],)
        or reference_weights.shape != (asset_count,)
        or not np.isfinite(sector_membership).all()
        or not np.isfinite(sector_reference).all()
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.sector_capacity_input_missing")
    relevant = eligible | ((~eligible) & (reference_weights > _FEASIBILITY_WEIGHT_TOLERANCE))
    relevant_membership = sector_membership[:, relevant]
    if bool(
        np.any(
            np.abs(relevant_membership - np.rint(relevant_membership))
            > _FEASIBILITY_WEIGHT_TOLERANCE
        )
    ) or not np.allclose(
        relevant_membership.sum(axis=0),
        1.0,
        rtol=0.0,
        atol=_FEASIBILITY_WEIGHT_TOLERANCE,
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.sector_membership_not_categorical")
    sector_by_asset = np.argmax(sector_membership, axis=0)
    frozen = (~eligible) & (reference_weights > _FEASIBILITY_WEIGHT_TOLERANCE)
    frozen_exposure = sector_membership @ np.where(frozen, reference_weights, 0.0)
    sector_lower = np.maximum(0.0, sector_reference - sector_capacity)
    sector_upper = np.minimum(1.0, sector_reference + sector_capacity)
    required_mutable_exposure = np.maximum(0.0, sector_lower - frozen_exposure)
    minimum_count = np.ceil(
        np.maximum(
            0.0,
            required_mutable_exposure - _FEASIBILITY_WEIGHT_TOLERANCE,
        )
        / maximum_weight
    ).astype(np.int64)
    if int(minimum_count.sum()) > top_k:
        raise PortfolioOptimizationError("portfolio_strategy_lab.band_capacity_insufficient")

    eligible_positions = np.flatnonzero(eligible & np.isfinite(scores))
    stable_order = eligible_positions[np.lexsort((eligible_positions, -scores[eligible_positions]))]
    selected_set = set(int(value) for value in selected)
    counts: npt.NDArray[np.int64] = np.bincount(
        sector_by_asset[selected], minlength=sector_membership.shape[0]
    ).astype(np.int64)
    for deficient_sector in range(sector_membership.shape[0]):
        while counts[deficient_sector] < minimum_count[deficient_sector]:
            added = next(
                (
                    int(index)
                    for index in stable_order
                    if int(index) not in selected_set
                    and sector_by_asset[int(index)] == deficient_sector
                ),
                None,
            )
            donors = [
                index
                for index in selected_set
                if counts[sector_by_asset[index]] > minimum_count[sector_by_asset[index]]
            ]
            if added is None or not donors:
                raise PortfolioOptimizationError(
                    "portfolio_strategy_lab.band_capacity_insufficient"
                )
            removed = min(donors, key=lambda index: (scores[index], -index))
            selected_set.remove(removed)
            selected_set.add(added)
            counts[sector_by_asset[removed]] -= 1
            counts[deficient_sector] += 1

    # Floors alone do not prove that a fully invested portfolio is reachable.
    # Names above a Sector ceiling add no usable capacity, so redistribute the
    # lowest-cost surplus names until the sum of capped reachable exposures can
    # fund the budget.  Reachability improvement is primary; signed-score loss
    # and stable positions are deterministic secondary choices.
    def reachable_exposure(current_counts: npt.NDArray[np.int64]) -> FloatArray:
        return np.minimum(
            sector_upper,
            frozen_exposure + current_counts.astype(np.float64) * maximum_weight,
        )

    reachable = reachable_exposure(counts)
    while float(reachable.sum()) < 1.0 - _FEASIBILITY_WEIGHT_TOLERANCE:
        add_by_sector: dict[int, int] = {}
        for index in stable_order:
            position = int(index)
            sector = int(sector_by_asset[position])
            if position not in selected_set and sector not in add_by_sector:
                add_by_sector[sector] = position
        candidates: list[tuple[tuple[float, float, int, int], int, int]] = []
        for added_sector, added in add_by_sector.items():
            added_reachable = min(
                float(sector_upper[added_sector]),
                float(frozen_exposure[added_sector])
                + float(counts[added_sector] + 1) * maximum_weight,
            )
            added_gain = added_reachable - float(reachable[added_sector])
            if added_gain <= _FEASIBILITY_WEIGHT_TOLERANCE:
                continue
            for removed in selected_set:
                removed_sector = int(sector_by_asset[removed])
                if counts[removed_sector] <= minimum_count[removed_sector]:
                    continue
                removed_reachable = min(
                    float(sector_upper[removed_sector]),
                    float(frozen_exposure[removed_sector])
                    + float(counts[removed_sector] - 1) * maximum_weight,
                )
                removed_loss = float(reachable[removed_sector]) - removed_reachable
                reach_gain = added_gain - removed_loss
                if reach_gain <= _FEASIBILITY_WEIGHT_TOLERANCE:
                    continue
                candidates.append(
                    (
                        (
                            reach_gain,
                            float(scores[added] - scores[removed]),
                            -added,
                            removed,
                        ),
                        added,
                        removed,
                    )
                )
        if not candidates:
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_total_capacity_short")
        _key, added, removed = max(candidates, key=lambda value: value[0])
        added_sector = int(sector_by_asset[added])
        removed_sector = int(sector_by_asset[removed])
        selected_set.remove(removed)
        selected_set.add(added)
        counts[removed_sector] -= 1
        counts[added_sector] += 1
        reachable = reachable_exposure(counts)
    return np.asarray(
        [int(index) for index in stable_order if int(index) in selected_set],
        dtype=np.int64,
    )


def _validate_inputs(
    *,
    scores: FloatArray,
    covariance: FloatArray,
    reference_weights: FloatArray,
    decision_eligible: BoolArray,
    top_k: int,
    maximum_weight: float,
    covariance_validation: CovarianceValidationProof | None,
) -> None:
    asset_count = scores.size
    if (
        scores.shape != (asset_count,)
        or covariance.shape != (asset_count, asset_count)
        or reference_weights.shape != (asset_count,)
        or decision_eligible.shape != (asset_count,)
        or not np.isfinite(covariance).all()
        or not np.isfinite(reference_weights).all()
        or bool(np.any(reference_weights < -_FEASIBILITY_WEIGHT_TOLERANCE))
        or float(reference_weights.sum()) > 1.0 + _FEASIBILITY_WEIGHT_TOLERANCE
        or top_k < 1
        or top_k > asset_count
        or maximum_weight <= 0.0
        or maximum_weight > 1.0
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.optimizer_input_invalid")
    if covariance_validation is None:
        # This is the direct/synthetic boundary. Owner-resolved high-volume
        # paths pass their immutable lane proof and do not repeat this O(N^2)
        # check for every session solve.
        validate_covariance_lane(values=covariance)
    elif not covariance_validation.owns(covariance):
        raise PortfolioOptimizationError(
            "portfolio_strategy_lab.optimizer_covariance_owner_mismatch"
        )
    if top_k * maximum_weight < 1.0 - _FEASIBILITY_WEIGHT_TOLERANCE:
        raise PortfolioOptimizationError("portfolio_strategy_lab.cap_infeasible")


def _active_axis(
    *,
    selected: npt.NDArray[np.int64],
    reference_weights: FloatArray,
) -> npt.NDArray[np.int64]:
    held = np.flatnonzero(reference_weights > _FEASIBILITY_WEIGHT_TOLERANCE)
    return np.asarray(tuple(dict.fromkeys((*selected.tolist(), *held.tolist()))), dtype=np.int64)


def _score_risk_cost_axis(
    *,
    scores: FloatArray,
    decision_eligible: BoolArray,
    reference_weights: FloatArray,
    selected: npt.NDArray[np.int64],
    top_k: int,
    maximum_weight: float,
    liquidation_only_carry: bool,
    maximum_one_way_turnover: float | None,
    initial_deployment_exempt: bool,
) -> tuple[npt.NDArray[np.int64], FloatArray, FloatArray, FloatArray, float | None]:
    if (
        selected.ndim != 1
        or not top_k <= selected.size <= top_k * 2
        or len(set(selected.tolist())) != selected.size
        or bool(np.any(selected < 0) | np.any(selected >= scores.size))
        or bool(np.any(~decision_eligible[selected]) | np.any(~np.isfinite(scores[selected])))
        or (maximum_one_way_turnover is not None and not 0.0 < maximum_one_way_turnover <= 1.0)
        or (initial_deployment_exempt and maximum_one_way_turnover is None)
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.rank_buffer_selection_invalid")
    axis = _active_axis(selected=selected, reference_weights=reference_weights)
    reference = reference_weights[axis]
    lower = np.zeros(axis.size, dtype=np.float64)
    frozen = (~decision_eligible[axis]) & (reference > _FEASIBILITY_WEIGHT_TOLERANCE)
    lower[frozen] = reference[frozen]
    selected_mask = np.isin(axis, selected)
    upper = np.where(selected_mask, maximum_weight, 0.0).astype(np.float64)
    upper[frozen] = reference[frozen]
    if liquidation_only_carry:
        carry = (reference > _FEASIBILITY_WEIGHT_TOLERANCE) & ~selected_mask & ~frozen
        upper[carry] = reference[carry]
    reference_mass = float(reference_weights.sum())
    if (
        maximum_one_way_turnover is not None
        and not initial_deployment_exempt
        and abs(1.0 - reference_mass) / 2.0
        > maximum_one_way_turnover + _FEASIBILITY_WEIGHT_TOLERANCE
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.turnover_cap_infeasible")
    turnover_cap = None if initial_deployment_exempt else maximum_one_way_turnover
    return axis, reference, lower, upper, turnover_cap


class PortfolioOptimizer:
    """Solve one fully invested long-only target without fallback or repair."""

    def __init__(self) -> None:
        """Initialize reusable ordinary, wide, capped and banded OSQP templates.

        Initialize reusable OSQP templates for ordinary, wide, turnover-capped and banded solves.
        """
        self._templates: dict[int, _OSQPTemplate] = {}
        self._wide_templates: dict[int, _OSQPTemplate] = {}
        self._capped_templates: dict[int, _OSQPTemplate] = {}
        self._banded: dict[tuple[int, int], _BandedTemplate] = {}

    def _banded_template(self, top_k: int, sector_count: int) -> _BandedTemplate:
        key = (top_k, sector_count)
        template = self._banded.get(key)
        if template is None:
            template = _build_banded_template(top_k, sector_count)
            self._banded[key] = template
        return template

    def _template(self, *, top_k: int, capacity: int, capped: bool) -> _OSQPTemplate:
        if capped:
            template = self._capped_templates.get(capacity)
            if template is None:
                template = _build_capped_osqp_template(capacity)
                self._capped_templates[capacity] = template
            return template
        if capacity == top_k * 2:
            template = self._templates.get(top_k)
            if template is None:
                template = _build_osqp_template(capacity)
                self._templates[top_k] = template
            return template
        template = self._wide_templates.get(capacity)
        if template is None:
            template = _build_osqp_template(capacity)
            self._wide_templates[capacity] = template
        return template

    def solve_score_risk_cost(
        self,
        *,
        scores: FloatArray,
        covariance: FloatArray,
        reference_weights: FloatArray,
        decision_eligible: BoolArray,
        top_k: int,
        maximum_weight: float,
        risk_aversion: float,
        turnover_regularization: float,
        covariance_validation: CovarianceValidationProof | None = None,
        transaction_cost_rate: float = 0.001,
        sector_matrix: FloatArray | None = None,
        sector_reference: FloatArray | None = None,
        sector_deviation_penalty: float = 0.0,
        sector_capacity: float | None = None,
        score_scale: float = 1.0,
        alpha_utility_admission: AlphaUtilityUnitsAdmission | None = None,
        selected_indices: npt.NDArray[np.int64] | None = None,
        liquidation_only_carry: bool = False,
        maximum_one_way_turnover: float | None = None,
        initial_deployment_exempt: bool = False,
    ) -> PortfolioOptimizationResult:
        """Solve admitted score/Risk/cost allocation and retain a verified safety-only comparison.

        Args:
            scores: Scores on the common listing axis.
            covariance: Owner-admitted covariance on that axis.
            reference_weights: Holdings reference including frozen carry.
            decision_eligible: Listings admitted for a mutable decision.
            top_k: Declared selection size.
            maximum_weight: Single-name upper bound.
            risk_aversion: Positive covariance penalty.
            turnover_regularization: Nonnegative squared holdings-change penalty.
            covariance_validation: Optional exact covariance validation proof.
            transaction_cost_rate: Nonnegative one-way turnover cost rate.
            sector_matrix: Optional sector membership/exposure on the listing axis.
            sector_reference: Sector reference exposures.
            sector_capacity: Optional hard absolute sector deviation; mutually exclusive with the
                soft penalty.
            sector_deviation_penalty: Optional nonnegative soft sector penalty.
            score_scale: Nonnegative score utility multiplier; zero disables utility admission.
            alpha_utility_admission: Exact identity-bound utility parameters required for positive
                score_scale.
            selected_indices: Optional explicit preselection; unavailable with hard sector capacity.
            liquidation_only_carry: Admit unselected mutable carry only for liquidation.
            maximum_one_way_turnover: Optional hard one-way turnover cap.
            initial_deployment_exempt: Exempt initial deployment from that turnover cap.

        Returns:
            Read-only verified weights/support and sector-unconstrained weights, with objective
            terms, solver counts and displacement diagnostics.

        Raises:
            PortfolioOptimizationError: Input/utility/selection admission, sector or turnover
                feasibility, solver convergence or final bounds verification fails.
        """
        _validate_inputs(
            scores=scores,
            covariance=covariance,
            reference_weights=reference_weights,
            decision_eligible=decision_eligible,
            top_k=top_k,
            maximum_weight=maximum_weight,
            covariance_validation=covariance_validation,
        )
        if (
            risk_aversion <= 0.0
            or turnover_regularization < 0.0
            or transaction_cost_rate < 0.0
            or score_scale < 0.0
            or sector_deviation_penalty < 0.0
            or (sector_capacity is not None and not 0.0 < sector_capacity <= 1.0)
            or (sector_capacity is not None and sector_deviation_penalty > 0.0)
            or (selected_indices is not None and sector_capacity is not None)
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.optimizer_parameter_invalid")
        if liquidation_only_carry and selected_indices is None:
            raise PortfolioOptimizationError("portfolio_strategy_lab.optimizer_parameter_invalid")
        if score_scale > 0.0:
            admission = alpha_utility_admission
            if (
                admission is None
                or admission.admission_hash
                != canonical_hash(
                    {
                        "method_id": admission.method_id,
                        "alpha_unit": admission.alpha_unit,
                        "risk_unit": admission.risk_unit,
                        "cost_unit": admission.cost_unit,
                        "score_scale": admission.score_scale,
                        "risk_aversion": admission.risk_aversion,
                        "turnover_regularization": admission.turnover_regularization,
                        "transaction_cost_rate": admission.transaction_cost_rate,
                        "normalization_reference_id": admission.normalization_reference_id,
                    }
                )
                or admission.score_scale != score_scale
                or admission.risk_aversion != risk_aversion
                or admission.turnover_regularization != turnover_regularization
                or admission.transaction_cost_rate != transaction_cost_rate
            ):
                raise PortfolioOptimizationError("ALPHA_UTILITY_UNITS_NOT_ADMITTED")
        elif alpha_utility_admission is not None:
            raise PortfolioOptimizationError("ALPHA_UTILITY_UNITS_NOT_ADMITTED")
        if selected_indices is not None:
            selected = np.asarray(selected_indices, dtype=np.int64)
        elif sector_capacity is None:
            selected = stable_top_k(scores, decision_eligible, top_k)
        else:
            if sector_matrix is None or sector_reference is None:
                raise PortfolioOptimizationError(
                    "portfolio_strategy_lab.sector_capacity_input_missing"
                )
            selected = stable_sector_feasible_top_k(
                scores=scores,
                eligible=decision_eligible,
                top_k=top_k,
                maximum_weight=maximum_weight,
                reference_weights=reference_weights,
                sector_membership=sector_matrix,
                sector_reference=sector_reference,
                sector_capacity=sector_capacity,
            )
        (
            axis,
            active_reference,
            active_lower,
            active_upper,
            turnover_cap,
        ) = _score_risk_cost_axis(
            scores=scores,
            decision_eligible=decision_eligible,
            reference_weights=reference_weights,
            selected=selected,
            top_k=top_k,
            maximum_weight=maximum_weight,
            liquidation_only_carry=liquidation_only_carry,
            maximum_one_way_turnover=maximum_one_way_turnover,
            initial_deployment_exempt=initial_deployment_exempt,
        )
        active_count = axis.size
        capacity = max(top_k * 2, active_count)
        ref: FloatArray = np.zeros(capacity, dtype=np.float64)
        ref[:active_count] = active_reference
        active_covariance = covariance[np.ix_(axis, axis)]
        upper: FloatArray = np.zeros(capacity, dtype=np.float64)
        upper[:active_count] = active_upper
        lower: FloatArray = np.zeros(capacity, dtype=np.float64)
        lower[:active_count] = active_lower
        axis_scores = np.asarray(scores[axis], dtype=np.float64)
        mutable = upper[:active_count] > lower[:active_count] + _FEASIBILITY_WEIGHT_TOLERANCE
        if bool(np.any(mutable & ~np.isfinite(axis_scores))):
            raise PortfolioOptimizationError("portfolio_strategy_lab.mutable_score_missing")
        local_scores: FloatArray = np.zeros(capacity, dtype=np.float64)
        finite_axis_scores = np.isfinite(axis_scores)
        local_scores[:active_count][finite_axis_scores] = (
            axis_scores[finite_axis_scores] * score_scale
        )
        hessian: FloatArray = np.zeros((capacity, capacity), dtype=np.float64)
        hessian[:active_count, :active_count] = risk_aversion * active_covariance
        diagonal = np.arange(capacity)
        hessian[diagonal, diagonal] += turnover_regularization
        local_scores = local_scores + 2.0 * turnover_regularization * ref
        if sector_deviation_penalty > 0.0:
            if sector_matrix is None or sector_reference is None:
                raise PortfolioOptimizationError(
                    "portfolio_strategy_lab.sector_penalty_input_missing"
                )
            local_sector = np.zeros((sector_matrix.shape[0], capacity), dtype=np.float64)
            local_sector[:, :active_count] = sector_matrix[:, axis]
            hessian = hessian + sector_deviation_penalty * (local_sector.T @ local_sector)
            local_scores = local_scores + 2.0 * sector_deviation_penalty * (
                local_sector.T @ np.asarray(sector_reference, dtype=np.float64)
            )

        # The safety-only comparison removes *only* Sector capacity.  Budget,
        # long-only, name cap, tradability/frozen holdings, reference and cost
        # terms remain exactly the same.  We retain this paired solution so a
        # hard Sector constraint can be audited without inventing a soft
        # ``Sector penalty`` objective term.
        def verified_full_weights(local_weights: FloatArray) -> FloatArray:
            full_weights = np.zeros(scores.size, dtype=np.float64)
            full_weights[axis] = local_weights[:active_count]
            self._verify(
                weights=full_weights,
                reference_weights=reference_weights,
                selected=axis[upper[:active_count] > _FEASIBILITY_WEIGHT_TOLERANCE],
                decision_eligible=decision_eligible,
                maximum_weight=maximum_weight,
                lower=lower[:active_count],
                upper=upper[:active_count],
                axis=axis,
                maximum_one_way_turnover=turnover_cap,
            )
            return full_weights

        unconstrained_template = self._template(
            top_k=top_k,
            capacity=capacity,
            capped=turnover_cap is not None,
        )
        (
            sector_unconstrained_weights,
            unconstrained_iterations,
            _unconstrained_dual,
            unconstrained_solver_calls,
        ) = unconstrained_template.solve(
            hessian=hessian,
            local_scores=local_scores,
            reference=ref,
            lower=lower,
            upper=upper,
            transaction_cost_rate=transaction_cost_rate,
            maximum_one_way_turnover=turnover_cap,
        )
        try:
            sector_unconstrained_full = verified_full_weights(sector_unconstrained_weights)
        except PortfolioOptimizationError as error:
            if str(error) not in _SOLUTION_VERIFICATION_ERRORS or unconstrained_solver_calls >= 2:
                raise
            (
                sector_unconstrained_weights,
                unconstrained_iterations,
                _unconstrained_dual,
                strict_solver_calls,
            ) = unconstrained_template.solve(
                hessian=hessian,
                local_scores=local_scores,
                reference=ref,
                lower=lower,
                upper=upper,
                transaction_cost_rate=transaction_cost_rate,
                maximum_one_way_turnover=turnover_cap,
                strict_accuracy=True,
            )
            unconstrained_solver_calls += strict_solver_calls
            sector_unconstrained_full = verified_full_weights(sector_unconstrained_weights)
        sector_lower: FloatArray | None = None
        sector_upper: FloatArray | None = None
        sector_dual: FloatArray = np.empty(0, dtype=np.float64)
        if sector_capacity is None:
            local_weights = sector_unconstrained_weights
            solver_iterations = unconstrained_iterations
            solver_call_count = unconstrained_solver_calls
        else:
            if (
                sector_matrix is None
                or sector_reference is None
                or sector_matrix.ndim != 2
                or sector_matrix.shape[1] != scores.size
                or sector_reference.shape != (sector_matrix.shape[0],)
                or not np.isfinite(sector_matrix).all()
                or not np.isfinite(sector_reference).all()
            ):
                raise PortfolioOptimizationError(
                    "portfolio_strategy_lab.sector_capacity_input_missing"
                )
            sector_lower = np.maximum(
                0.0, np.asarray(sector_reference, dtype=np.float64) - sector_capacity
            )
            sector_upper = np.minimum(
                1.0, np.asarray(sector_reference, dtype=np.float64) + sector_capacity
            )
            admitted = np.zeros(scores.size, dtype=np.bool_)
            admitted[selected] = decision_eligible[selected]
            frozen = (~decision_eligible) & (reference_weights > _FEASIBILITY_WEIGHT_TOLERANCE)
            self.prove_banded_feasibility(
                admitted=admitted,
                frozen_weights=np.where(frozen, reference_weights, 0.0),
                sector_membership=sector_matrix,
                sector_lower=sector_lower,
                sector_upper=sector_upper,
                maximum_weight=maximum_weight,
            )
            local_sector = np.zeros((sector_matrix.shape[0], capacity), dtype=np.float64)
            local_sector[:, :active_count] = sector_matrix[:, axis]
            template = self._banded_template(top_k, int(sector_matrix.shape[0]))
            (
                local_weights,
                solver_iterations,
                constraint_dual,
                constrained_solver_calls,
            ) = template.solve(
                hessian=hessian,
                local_scores=local_scores,
                reference=ref,
                lower=lower,
                upper=upper,
                transaction_cost_rate=transaction_cost_rate,
                sector_membership=local_sector,
                sector_lower=sector_lower,
                sector_upper=sector_upper,
                exposure=np.zeros(capacity, dtype=np.float64),
                exposure_lower=-np.inf,
                exposure_upper=np.inf,
            )
            try:
                weights = verified_full_weights(local_weights)
                realized_sector = sector_matrix @ weights
                if bool(
                    np.any(realized_sector < sector_lower - _SECTOR_FEASIBILITY_TOLERANCE)
                ) or bool(np.any(realized_sector > sector_upper + _SECTOR_FEASIBILITY_TOLERANCE)):
                    raise PortfolioOptimizationError("portfolio_strategy_lab.band_violated")
            except PortfolioOptimizationError as error:
                if str(error) not in _SOLUTION_VERIFICATION_ERRORS or constrained_solver_calls >= 2:
                    raise
                (
                    local_weights,
                    solver_iterations,
                    constraint_dual,
                    strict_solver_calls,
                ) = template.solve(
                    hessian=hessian,
                    local_scores=local_scores,
                    reference=ref,
                    lower=lower,
                    upper=upper,
                    transaction_cost_rate=transaction_cost_rate,
                    sector_membership=local_sector,
                    sector_lower=sector_lower,
                    sector_upper=sector_upper,
                    exposure=np.zeros(capacity, dtype=np.float64),
                    exposure_lower=-np.inf,
                    exposure_upper=np.inf,
                    strict_accuracy=True,
                )
                constrained_solver_calls += strict_solver_calls
                weights = verified_full_weights(local_weights)
                realized_sector = sector_matrix @ weights
                if bool(
                    np.any(realized_sector < sector_lower - _SECTOR_FEASIBILITY_TOLERANCE)
                ) or bool(np.any(realized_sector > sector_upper + _SECTOR_FEASIBILITY_TOLERANCE)):
                    raise PortfolioOptimizationError(
                        "portfolio_strategy_lab.band_violated"
                    ) from error
            sector_start = 1 + capacity * 4
            sector_dual = np.asarray(
                constraint_dual[sector_start : sector_start + sector_matrix.shape[0]],
                dtype=np.float64,
            )
            solver_call_count = unconstrained_solver_calls + constrained_solver_calls
        if sector_capacity is None:
            weights = sector_unconstrained_full
        active_weights = local_weights[:active_count]
        unconstrained_active_weights = sector_unconstrained_weights[:active_count]
        variance = float(active_weights @ active_covariance @ active_weights)
        unconstrained_variance = float(
            unconstrained_active_weights @ active_covariance @ unconstrained_active_weights
        )
        turnover = float(np.abs(weights - reference_weights).sum() / 2.0)
        score_utility = float(
            scores[axis][finite_axis_scores] @ local_weights[:active_count][finite_axis_scores]
        )
        alpha_term = score_scale * score_utility
        risk_term = risk_aversion * variance
        transaction_term = transaction_cost_rate * turnover
        turnover_term = turnover_regularization * np.square(weights - reference_weights).sum()
        sector_penalty_term: float | None = None
        if sector_deviation_penalty > 0.0:
            assert sector_matrix is not None
            assert sector_reference is not None
            deviation = sector_matrix @ weights - sector_reference
            sector_penalty_term = float(sector_deviation_penalty * np.dot(deviation, deviation))
        objective = float(
            alpha_term - risk_term - transaction_term - turnover_term - (sector_penalty_term or 0.0)
        )
        sector_lower_slack: tuple[float, ...] = ()
        sector_upper_slack: tuple[float, ...] = ()
        desired_sector_exposure: tuple[float, ...] = ()
        final_sector_exposure: tuple[float, ...] = ()
        binding_sector_count = 0
        if sector_matrix is not None:
            desired_sector_exposure = tuple(
                float(value) for value in sector_matrix @ sector_unconstrained_full
            )
            final_sector_exposure = tuple(float(value) for value in sector_matrix @ weights)
        if sector_capacity is not None:
            assert sector_matrix is not None
            assert sector_lower is not None
            assert sector_upper is not None
            realized_sector = sector_matrix @ weights
            lower_slack = realized_sector - sector_lower
            upper_slack = sector_upper - realized_sector
            if bool(np.any(lower_slack < -_SECTOR_FEASIBILITY_TOLERANCE)) or bool(
                np.any(upper_slack < -_SECTOR_FEASIBILITY_TOLERANCE)
            ):
                raise PortfolioOptimizationError("portfolio_strategy_lab.band_violated")
            sector_lower_slack = tuple(float(value) for value in lower_slack)
            sector_upper_slack = tuple(float(value) for value in upper_slack)
            binding_sector_count = int(
                np.sum(
                    (lower_slack <= _SECTOR_FEASIBILITY_TOLERANCE)
                    | (upper_slack <= _SECTOR_FEASIBILITY_TOLERANCE)
                )
            )
        weights.setflags(write=False)
        selected.setflags(write=False)
        sector_unconstrained_full.setflags(write=False)
        return PortfolioOptimizationResult(
            weights=weights,
            selected_indices=selected,
            objective_value=objective,
            predicted_variance=variance,
            predicted_one_way_turnover=turnover,
            solver_iterations=solver_iterations,
            solver_call_count=solver_call_count,
            objective_audit=PortfolioObjectiveAudit(
                alpha_utility_term=float(alpha_term),
                risk_penalty_term=float(risk_term),
                transaction_cost_term=float(transaction_term),
                turnover_regularization_term=float(turnover_term),
                sector_penalty_term=sector_penalty_term,
                reference_to_sector_unconstrained_l1_distance=float(
                    np.abs(sector_unconstrained_full - reference_weights).sum()
                ),
                reference_to_final_l1_distance=float(np.abs(weights - reference_weights).sum()),
                sector_unconstrained_to_final_l1_distance=float(
                    np.abs(weights - sector_unconstrained_full).sum()
                ),
                predicted_variance=variance,
                expected_one_way_turnover=turnover,
                sector_capacity=sector_capacity,
                sector_lower_slack=sector_lower_slack,
                sector_upper_slack=sector_upper_slack,
                sector_dual=tuple(float(value) for value in sector_dual),
                binding_sector_count=binding_sector_count,
                desired_sector_exposure=desired_sector_exposure,
                final_sector_exposure=final_sector_exposure,
                displaced_name_count=int(
                    np.count_nonzero(
                        np.abs(weights - sector_unconstrained_full) > _FEASIBILITY_WEIGHT_TOLERANCE
                    )
                ),
                displaced_weight_mass=float(
                    np.abs(weights - sector_unconstrained_full).sum() / 2.0
                ),
                pre_constraint_score_utility=float(
                    score_scale
                    * (
                        scores[axis][finite_axis_scores]
                        @ unconstrained_active_weights[finite_axis_scores]
                    )
                ),
                post_constraint_score_utility=float(alpha_term),
                pre_constraint_predicted_variance=unconstrained_variance,
                post_constraint_predicted_variance=variance,
                utility_admission_hash=(
                    alpha_utility_admission.admission_hash
                    if alpha_utility_admission is not None
                    else None
                ),
            ),
            sector_unconstrained_weights=sector_unconstrained_full,
        )

    def prove_banded_feasibility(
        self,
        *,
        admitted: BoolArray,
        frozen_weights: FloatArray,
        sector_membership: FloatArray,
        sector_lower: FloatArray,
        sector_upper: FloatArray,
        maximum_weight: float,
    ) -> None:
        """Refuse an infeasible band system before OSQP is asked to solve it.

        Solver infeasibility is a stable answer, but it is a late and uninformative
        one: "primal infeasible" does not say whether a Sector floor exceeded its
        own capacity, whether the floors summed past full investment, or whether a
        frozen holding had already breached a ceiling. Those are three different
        research errors and the researcher has to be told which one happened.

        Nothing here relaxes anything. This is the check that makes relaxation
        unnecessary to *contemplate*: an infeasible trial fails closed with a
        typed reason and stays failed.
        """
        if (
            sector_membership.ndim != 2
            or sector_membership.shape[1] != admitted.size
            or frozen_weights.shape != (admitted.size,)
            or sector_lower.shape[0] != sector_membership.shape[0]
            or sector_upper.shape[0] != sector_membership.shape[0]
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_axis_invalid")
        if bool(np.any(sector_lower < -_FEASIBILITY_WEIGHT_TOLERANCE)) or bool(
            np.any(sector_upper < sector_lower - _FEASIBILITY_WEIGHT_TOLERANCE)
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_bounds_invalid")
        if float(sector_lower.sum()) > 1.0 + _FEASIBILITY_WEIGHT_TOLERANCE:
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_floor_exceeds_budget")
        if float(sector_upper.sum()) < 1.0 - _FEASIBILITY_WEIGHT_TOLERANCE:
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_ceiling_below_budget")
        frozen_by_sector = sector_membership @ frozen_weights
        capacity_by_sector = sector_membership @ np.where(admitted, maximum_weight, 0.0)
        reachable = np.minimum(sector_upper, frozen_by_sector + capacity_by_sector)
        if bool(np.any(frozen_by_sector > sector_upper + _FEASIBILITY_WEIGHT_TOLERANCE)):
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_frozen_exceeds_ceiling")
        if bool(np.any(reachable < sector_lower - _FEASIBILITY_WEIGHT_TOLERANCE)):
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_capacity_insufficient")
        if float(reachable.sum()) < 1.0 - _FEASIBILITY_WEIGHT_TOLERANCE:
            raise PortfolioOptimizationError("portfolio_strategy_lab.band_total_capacity_short")

    def solve_banded_total_signal(
        self,
        *,
        scores: FloatArray,
        covariance: FloatArray,
        reference_weights: FloatArray,
        decision_eligible: BoolArray,
        admitted: BoolArray,
        sector_membership: FloatArray,
        sector_lower: FloatArray,
        sector_upper: FloatArray,
        exposure: FloatArray,
        exposure_lower: float,
        exposure_upper: float,
        maximum_weight: float,
        risk_aversion: float,
        transaction_cost_rate: float,
    ) -> PortfolioOptimizationResult:
        """One global QP under bands, beta, caps, freezes and an admitted pool.

        ``admitted`` replaces the global top-K selection: a Sector floor and a
        global rank are two different admission questions, and answering the
        first with the second is how a floor becomes infeasible for reasons no
        error message explains. The stratified pool is decided by the policy and
        arrives here already resolved.
        """
        asset_count = scores.size
        if (
            covariance.shape != (asset_count, asset_count)
            or reference_weights.shape != (asset_count,)
            or decision_eligible.shape != (asset_count,)
            or admitted.shape != (asset_count,)
            or exposure.shape != (asset_count,)
            or not np.isfinite(covariance).all()
            or not np.isfinite(reference_weights).all()
            or maximum_weight <= 0.0
            or maximum_weight > 1.0
            or risk_aversion <= 0.0
            or transaction_cost_rate < 0.0
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.optimizer_input_invalid")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise PortfolioOptimizationError("portfolio_strategy_lab.covariance_not_symmetric")
        frozen = (~decision_eligible) & (reference_weights > _FEASIBILITY_WEIGHT_TOLERANCE)
        selected = np.flatnonzero(admitted & decision_eligible)
        axis: npt.NDArray[np.int64] = np.asarray(
            tuple(dict.fromkeys((*selected.tolist(), *np.flatnonzero(frozen).tolist()))),
            dtype=np.int64,
        )
        if axis.size < 1:
            raise PortfolioOptimizationError("portfolio_strategy_lab.selection_pool_insufficient")
        top_k = max(1, (axis.size + 1) // 2)
        capacity = top_k * 2
        if axis.size > capacity:
            raise PortfolioOptimizationError("portfolio_strategy_lab.active_axis_exceeded")
        active_count = axis.size

        # A guardrail evaluated with an unresolved exposure is not a guardrail.
        axis_exposure = np.asarray(exposure[axis], dtype=np.float64)
        if not np.isfinite(axis_exposure).all():
            raise PortfolioOptimizationError("portfolio_strategy_lab.exposure_unresolved")
        axis_scores = np.asarray(scores[axis], dtype=np.float64)
        if not np.isfinite(axis_scores).all():
            raise PortfolioOptimizationError("portfolio_strategy_lab.mutable_score_missing")

        self.prove_banded_feasibility(
            admitted=admitted & decision_eligible,
            frozen_weights=np.where(frozen, reference_weights, 0.0),
            sector_membership=sector_membership,
            sector_lower=sector_lower,
            sector_upper=sector_upper,
            maximum_weight=maximum_weight,
        )

        ref: FloatArray = np.zeros(capacity, dtype=np.float64)
        ref[:active_count] = reference_weights[axis]
        lower: FloatArray = np.zeros(capacity, dtype=np.float64)
        upper: FloatArray = np.zeros(capacity, dtype=np.float64)
        upper[:active_count] = np.where(
            admitted[axis] & decision_eligible[axis], maximum_weight, 0.0
        )
        frozen_axis = frozen[axis]
        lower[:active_count][frozen_axis] = ref[:active_count][frozen_axis]
        upper[:active_count][frozen_axis] = ref[:active_count][frozen_axis]
        local_scores: FloatArray = np.zeros(capacity, dtype=np.float64)
        local_scores[:active_count] = axis_scores
        local_exposure: FloatArray = np.zeros(capacity, dtype=np.float64)
        local_exposure[:active_count] = axis_exposure
        active_covariance = covariance[np.ix_(axis, axis)]
        hessian: FloatArray = np.zeros((capacity, capacity), dtype=np.float64)
        hessian[:active_count, :active_count] = risk_aversion * active_covariance
        local_sector: FloatArray = np.zeros((sector_membership.shape[0], capacity), np.float64)
        local_sector[:, :active_count] = sector_membership[:, axis]

        template = self._banded_template(top_k, int(sector_membership.shape[0]))

        def verify_banded_solution(
            candidate_weights: FloatArray,
        ) -> tuple[FloatArray, FloatArray, float]:
            full_weights = np.zeros(asset_count, dtype=np.float64)
            full_weights[axis] = candidate_weights[:active_count]
            self._verify(
                weights=full_weights,
                reference_weights=reference_weights,
                selected=np.asarray(selected, dtype=np.int64),
                decision_eligible=decision_eligible,
                maximum_weight=maximum_weight,
                lower=lower[:active_count],
                upper=upper[:active_count],
                axis=axis,
            )
            realized_sector = sector_membership @ full_weights
            if bool(np.any(realized_sector < sector_lower - _SECTOR_FEASIBILITY_TOLERANCE)) or bool(
                np.any(realized_sector > sector_upper + _SECTOR_FEASIBILITY_TOLERANCE)
            ):
                raise PortfolioOptimizationError("portfolio_strategy_lab.band_violated")
            realized_exposure = float(np.where(np.isfinite(exposure), exposure, 0.0) @ full_weights)
            if not (
                exposure_lower - _SECTOR_FEASIBILITY_TOLERANCE
                <= realized_exposure
                <= exposure_upper + _SECTOR_FEASIBILITY_TOLERANCE
            ):
                raise PortfolioOptimizationError(
                    "portfolio_strategy_lab.exposure_guardrail_violated"
                )
            return full_weights, realized_sector, realized_exposure

        local_weights, solver_iterations, constraint_dual, solver_call_count = template.solve(
            hessian=hessian,
            local_scores=local_scores,
            reference=ref,
            lower=lower,
            upper=upper,
            transaction_cost_rate=transaction_cost_rate,
            sector_membership=local_sector,
            sector_lower=np.asarray(sector_lower, dtype=np.float64),
            sector_upper=np.asarray(sector_upper, dtype=np.float64),
            exposure=local_exposure,
            exposure_lower=exposure_lower,
            exposure_upper=exposure_upper,
        )
        try:
            weights, realized_sector, _realized_exposure = verify_banded_solution(local_weights)
        except PortfolioOptimizationError as error:
            if str(error) not in _SOLUTION_VERIFICATION_ERRORS or solver_call_count >= 2:
                raise
            local_weights, solver_iterations, constraint_dual, strict_solver_calls = template.solve(
                hessian=hessian,
                local_scores=local_scores,
                reference=ref,
                lower=lower,
                upper=upper,
                transaction_cost_rate=transaction_cost_rate,
                sector_membership=local_sector,
                sector_lower=np.asarray(sector_lower, dtype=np.float64),
                sector_upper=np.asarray(sector_upper, dtype=np.float64),
                exposure=local_exposure,
                exposure_lower=exposure_lower,
                exposure_upper=exposure_upper,
                strict_accuracy=True,
            )
            solver_call_count += strict_solver_calls
            weights, realized_sector, _realized_exposure = verify_banded_solution(local_weights)
        active_weights = local_weights[:active_count]
        variance = float(active_weights @ active_covariance @ active_weights)
        turnover = float(np.abs(weights - reference_weights).sum() / 2.0)
        objective = float(
            axis_scores @ active_weights
            - risk_aversion * variance
            - transaction_cost_rate * turnover
        )
        sector_start = 1 + capacity * 4
        sector_dual = constraint_dual[sector_start : sector_start + sector_membership.shape[0]]
        lower_slack = realized_sector - sector_lower
        upper_slack = sector_upper - realized_sector
        weights.setflags(write=False)
        pool = np.asarray(selected, dtype=np.int64)
        pool.setflags(write=False)
        return PortfolioOptimizationResult(
            weights=weights,
            selected_indices=pool,
            objective_value=objective,
            predicted_variance=variance,
            predicted_one_way_turnover=turnover,
            solver_iterations=solver_iterations,
            solver_call_count=solver_call_count,
            objective_audit=PortfolioObjectiveAudit(
                alpha_utility_term=float(axis_scores @ active_weights),
                risk_penalty_term=float(risk_aversion * variance),
                transaction_cost_term=float(transaction_cost_rate * turnover),
                turnover_regularization_term=0.0,
                sector_penalty_term=None,
                reference_to_sector_unconstrained_l1_distance=float(
                    np.abs(weights - reference_weights).sum()
                ),
                reference_to_final_l1_distance=float(np.abs(weights - reference_weights).sum()),
                sector_unconstrained_to_final_l1_distance=0.0,
                predicted_variance=variance,
                expected_one_way_turnover=turnover,
                sector_capacity=None,
                sector_lower_slack=tuple(float(value) for value in lower_slack),
                sector_upper_slack=tuple(float(value) for value in upper_slack),
                sector_dual=tuple(float(value) for value in sector_dual),
                binding_sector_count=int(
                    np.sum(
                        (lower_slack <= _SECTOR_FEASIBILITY_TOLERANCE)
                        | (upper_slack <= _SECTOR_FEASIBILITY_TOLERANCE)
                    )
                ),
                desired_sector_exposure=tuple(float(value) for value in realized_sector),
                final_sector_exposure=tuple(float(value) for value in realized_sector),
                displaced_name_count=0,
                displaced_weight_mass=0.0,
                pre_constraint_score_utility=float(axis_scores @ active_weights),
                post_constraint_score_utility=float(axis_scores @ active_weights),
                pre_constraint_predicted_variance=variance,
                post_constraint_predicted_variance=variance,
                utility_admission_hash=None,
            ),
            sector_unconstrained_weights=weights,
        )

    def minimum_variance(
        self,
        *,
        scores: FloatArray,
        covariance: FloatArray,
        reference_weights: FloatArray,
        decision_eligible: BoolArray,
        top_k: int,
        maximum_weight: float,
        sector_matrix: FloatArray | None = None,
        sector_reference: FloatArray | None = None,
        sector_capacity: float | None = None,
    ) -> PortfolioOptimizationResult:
        """Solve minimum variance with zero score utility, turnover penalty and transaction cost.

        Args:
            scores: Scores on the common listing axis.
            covariance: Owner-admitted covariance on that axis.
            reference_weights: Holdings reference including frozen carry.
            decision_eligible: Listings admitted for a mutable decision.
            top_k: Declared selection size.
            maximum_weight: Single-name upper bound.
            sector_matrix: Optional sector membership/exposure on the listing axis.
            sector_reference: Sector reference exposures.
            sector_capacity: Optional hard absolute sector deviation; mutually exclusive with the
                soft penalty.

        Returns:
            Verified minimum-variance result and retained solver/sector diagnostics.

        Raises:
            PortfolioOptimizationError: Common input, selection, feasibility or solution
                verification fails.
        """
        return self.solve_score_risk_cost(
            scores=scores,
            covariance=covariance,
            reference_weights=reference_weights,
            decision_eligible=decision_eligible,
            top_k=top_k,
            maximum_weight=maximum_weight,
            risk_aversion=1.0,
            turnover_regularization=0.0,
            transaction_cost_rate=0.0,
            sector_matrix=sector_matrix,
            sector_reference=sector_reference,
            sector_capacity=sector_capacity,
            score_scale=0.0,
        )

    @staticmethod
    def _verify(
        *,
        weights: FloatArray,
        reference_weights: FloatArray,
        selected: npt.NDArray[np.int64],
        decision_eligible: BoolArray,
        maximum_weight: float,
        lower: FloatArray,
        upper: FloatArray,
        axis: npt.NDArray[np.int64],
        maximum_one_way_turnover: float | None = None,
    ) -> None:
        if not np.isfinite(weights).all():
            raise PortfolioOptimizationError("portfolio_strategy_lab.solution_nonfinite")
        if bool(np.any(weights < -_FEASIBILITY_WEIGHT_TOLERANCE)):
            raise PortfolioOptimizationError("portfolio_strategy_lab.solution_negative_weight")
        if abs(float(weights.sum()) - 1.0) > _FEASIBILITY_WEIGHT_TOLERANCE:
            raise PortfolioOptimizationError("portfolio_strategy_lab.solution_budget_invalid")
        if bool(np.any(weights[axis] < lower - _FEASIBILITY_WEIGHT_TOLERANCE)):
            raise PortfolioOptimizationError("portfolio_strategy_lab.solution_lower_bound_invalid")
        if bool(np.any(weights[axis] > upper + _FEASIBILITY_WEIGHT_TOLERANCE)):
            raise PortfolioOptimizationError("portfolio_strategy_lab.solution_upper_bound_invalid")
        selected_set = set(selected.tolist())
        for index, value in enumerate(weights):
            if index in selected_set or value <= _FEASIBILITY_WEIGHT_TOLERANCE:
                continue
            if (
                decision_eligible[index]
                or abs(value - reference_weights[index]) > _FEASIBILITY_WEIGHT_TOLERANCE
            ):
                raise PortfolioOptimizationError(
                    "portfolio_strategy_lab.new_allocation_outside_selection"
                )
        cap_exceptions = (reference_weights > maximum_weight) & (
            weights <= reference_weights + _FEASIBILITY_WEIGHT_TOLERANCE
        )
        if bool(
            np.any((weights > maximum_weight + _FEASIBILITY_WEIGHT_TOLERANCE) & ~cap_exceptions)
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.maximum_weight_exceeded")
        if (
            maximum_one_way_turnover is not None
            and float(np.abs(weights - reference_weights).sum() / 2.0)
            > maximum_one_way_turnover + _FEASIBILITY_WEIGHT_TOLERANCE
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.turnover_cap_violated")


def solve_with_cvxpy_oracle(
    *,
    scores: FloatArray,
    covariance: FloatArray,
    reference_weights: FloatArray,
    decision_eligible: BoolArray,
    top_k: int,
    maximum_weight: float,
    risk_aversion: float,
    turnover_regularization: float,
    covariance_validation: CovarianceValidationProof | None = None,
    transaction_cost_rate: float = 0.001,
    sector_matrix: FloatArray | None = None,
    sector_reference: FloatArray | None = None,
    sector_capacity: float | None = None,
    score_scale: float = 1.0,
    alpha_utility_admission: AlphaUtilityUnitsAdmission | None = None,
    selected_indices: npt.NDArray[np.int64] | None = None,
    liquidation_only_carry: bool = False,
    maximum_one_way_turnover: float | None = None,
    initial_deployment_exempt: bool = False,
) -> FloatArray:
    """Independently express the admitted problem for focused numerical goldens."""
    _validate_inputs(
        scores=scores,
        covariance=covariance,
        reference_weights=reference_weights,
        decision_eligible=decision_eligible,
        top_k=top_k,
        maximum_weight=maximum_weight,
        covariance_validation=covariance_validation,
    )
    if (
        score_scale <= 0.0
        or risk_aversion <= 0.0
        or turnover_regularization < 0.0
        or transaction_cost_rate < 0.0
        or (sector_capacity is not None and not 0.0 < sector_capacity <= 1.0)
        or (selected_indices is not None and sector_capacity is not None)
    ):
        raise PortfolioOptimizationError("portfolio_strategy_lab.optimizer_parameter_invalid")
    if liquidation_only_carry and selected_indices is None:
        raise PortfolioOptimizationError("portfolio_strategy_lab.optimizer_parameter_invalid")
    admission = alpha_utility_admission
    if (
        admission is None
        or admission.admission_hash
        != canonical_hash(
            {
                "method_id": admission.method_id,
                "alpha_unit": admission.alpha_unit,
                "risk_unit": admission.risk_unit,
                "cost_unit": admission.cost_unit,
                "score_scale": admission.score_scale,
                "risk_aversion": admission.risk_aversion,
                "turnover_regularization": admission.turnover_regularization,
                "transaction_cost_rate": admission.transaction_cost_rate,
                "normalization_reference_id": admission.normalization_reference_id,
            }
        )
        or admission.score_scale != score_scale
        or admission.risk_aversion != risk_aversion
        or admission.turnover_regularization != turnover_regularization
        or admission.transaction_cost_rate != transaction_cost_rate
    ):
        raise PortfolioOptimizationError("ALPHA_UTILITY_UNITS_NOT_ADMITTED")
    if selected_indices is not None:
        selected = np.asarray(selected_indices, dtype=np.int64)
    elif sector_capacity is None:
        selected = stable_top_k(scores, decision_eligible, top_k)
    else:
        if sector_matrix is None or sector_reference is None:
            raise PortfolioOptimizationError("portfolio_strategy_lab.sector_capacity_input_missing")
        selected = stable_sector_feasible_top_k(
            scores=scores,
            eligible=decision_eligible,
            top_k=top_k,
            maximum_weight=maximum_weight,
            reference_weights=reference_weights,
            sector_membership=sector_matrix,
            sector_reference=sector_reference,
            sector_capacity=sector_capacity,
        )
    axis, ref, lower, upper, turnover_cap = _score_risk_cost_axis(
        scores=scores,
        decision_eligible=decision_eligible,
        reference_weights=reference_weights,
        selected=selected,
        top_k=top_k,
        maximum_weight=maximum_weight,
        liquidation_only_carry=liquidation_only_carry,
        maximum_one_way_turnover=maximum_one_way_turnover,
        initial_deployment_exempt=initial_deployment_exempt,
    )
    local_scores = np.asarray(scores[axis], dtype=np.float64)
    mutable = upper > lower + _FEASIBILITY_WEIGHT_TOLERANCE
    if bool(np.any(mutable & ~np.isfinite(local_scores))):
        raise PortfolioOptimizationError("portfolio_strategy_lab.mutable_score_missing")
    local_scores = np.where(np.isfinite(local_scores), local_scores * score_scale, 0.0)
    weights = cp.Variable(axis.size)
    delta = weights - ref
    objective = cp.Maximize(
        local_scores @ weights
        - risk_aversion * cp.quad_form(weights, cp.psd_wrap(covariance[np.ix_(axis, axis)]))
        - transaction_cost_rate * cp.norm1(delta) / 2.0
        - turnover_regularization * cp.sum_squares(delta)
    )
    constraints: list[cp.Constraint] = [
        cp.sum(weights) == 1.0,
        weights >= lower,
        weights <= upper,
    ]
    if turnover_cap is not None:
        constraints.append(cp.norm1(delta) / 2.0 <= turnover_cap)
    if sector_capacity is not None:
        if (
            sector_matrix is None
            or sector_reference is None
            or sector_matrix.ndim != 2
            or sector_matrix.shape[1] != scores.size
            or sector_reference.shape != (sector_matrix.shape[0],)
            or not np.isfinite(sector_matrix).all()
            or not np.isfinite(sector_reference).all()
        ):
            raise PortfolioOptimizationError("portfolio_strategy_lab.sector_capacity_input_missing")
        local_sector = sector_matrix[:, axis]
        constraints.extend(
            (
                local_sector @ weights >= np.maximum(0.0, sector_reference - sector_capacity),
                local_sector @ weights <= np.minimum(1.0, sector_reference + sector_capacity),
            )
        )
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.OSQP,
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=20_000,
        polishing=True,
        verbose=False,
    )
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise PortfolioOptimizationError("portfolio_strategy_lab.oracle_not_solved")
    result = np.zeros(scores.size, dtype=np.float64)
    result[axis] = np.asarray(weights.value, dtype=np.float64)
    result.setflags(write=False)
    return result


__all__ = [
    "AlphaUtilityUnitsAdmission",
    "PortfolioGoldenTolerances",
    "PortfolioObjectiveAudit",
    "PortfolioOptimizationError",
    "PortfolioOptimizationResult",
    "PortfolioOptimizer",
    "measure_golden_tolerances",
    "score_risk_cost_objective",
    "solve_with_cvxpy_oracle",
    "stable_rank_buffered_top_k",
    "stable_sector_feasible_top_k",
    "stable_top_k",
]
