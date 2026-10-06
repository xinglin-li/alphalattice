"""Installed Portfolio/OOS evaluation for a paired Alpha score experiment.

The seam is experiment-neutral: callers provide two immutable score handles and
Host-resolved market/Risk surfaces.  It uses installed policy adapters, Direct
OSQP, the CVXPY/OSQP oracle and the shared walk-forward/metric owners.  No caller
can supply matrices through an authoring request; this runtime value is built by
the Host after all handles and common axes have been resolved.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.portfolio_backtesting.clocks import (
    EveryFormationClock,
    EveryNFormationsClock,
    resolve_rebalance_clock,
)
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioStateTransitionBinding,
    PortfolioTargetDecision,
    RebalanceClock,
)
from alphalattice.capabilities.portfolio_backtesting.metrics import (
    PortfolioEconomicMetricSet,
    break_even_cost_bps,
    evaluate_raw_simple_return_path,
    realized_to_predicted_variance_ratio,
)
from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
from alphalattice.capabilities.portfolio_backtesting.segments import (
    run_portfolio_walk_forward_segment_sequence_with_passive_holds,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_statistics import (
    circular_block_interval,
    dependence_aware_block_length,
)
from alphalattice.investment.portfolio_management.mandates.paired_panel_grid import (
    PAIRED_PANEL_PORTFOLIO_GRID,
    RANK_BUFFERED_TURNOVER_SUCCESSOR_GRID,
)
from alphalattice.investment.portfolio_strategy_lab.contracts import (
    PortfolioScoreMode,
    TopKEqualWeightPolicy,
    TopKMinimumVariancePolicy,
)
from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
    PortfolioPolicyDecisionProvider,
)
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
    DIRECT_OSQP_MAXIMUM_INVOCATIONS_PER_OPTIMIZATION,
    AlphaUtilityUnitsAdmission,
    CovarianceValidationProof,
    PortfolioGoldenTolerances,
    PortfolioObjectiveAudit,
    PortfolioOptimizer,
    measure_golden_tolerances,
    score_risk_cost_objective,
    solve_with_cvxpy_oracle,
    stable_rank_buffered_top_k,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    PortfolioPolicyRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.policies.minimum_variance import (
    MinimumVarianceDevelopmentRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.policies.score_risk_cost import (
    RankBufferedScoreRiskCostDevelopmentRecipe,
    ScoreRiskCostDevelopmentRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.kernel.quant.sector_history import sector_exposure
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .paired_alpha_portfolio_evidence import (
    ControlledCovarianceRegressionEvidence,
    PairedAlphaPortfolioReplayReceipt,
    PairedAlphaPortfolioResearchRoot,
    PairedPortfolioError,
    PairedPortfolioTrialEvidence,
    PortfolioEffectDimension,
    PortfolioGoldenOracleEvidence,
    PortfolioObjectiveAuditSummary,
    PortfolioPairedDecisionEvidence,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]

_HASH = r"^[0-9a-f]{64}$"
_GRID = PAIRED_PANEL_PORTFOLIO_GRID
_COST_ARMS = tuple(float(value) for value in _GRID.cost_stress_bps)
_RISK_GRID = _GRID.risk_aversion_grid
_TURNOVER_GRID = _GRID.turnover_regularization_grid
_SCORE_SCALE = 0.01
_TRANSACTION_COST_RATE = 0.0005
_BOOTSTRAP_RESAMPLES = 2_000
_BOOTSTRAP_SEED = 20260822
_TRIAL_CATEGORY = "development/paired-alpha-portfolio/trials"
_TARGET_WEIGHT_CATEGORY = "development/paired-alpha-portfolio/target-weights"
_REFERENCE_WEIGHT_CATEGORY = "development/paired-alpha-portfolio/reference-weights"
_PATH_LANE_CATEGORY = "development/paired-alpha-portfolio/path-lanes"
_REPLAY_CATEGORY = "development/paired-alpha-portfolio/replays"
_RANK_BUFFERED_INNER_FORMATION_COUNT = 840
_RANK_BUFFERED_SELECTION_COST_BPS = 5.0


@dataclass(frozen=True, slots=True)
class RankBufferedR0InnerCandidate:
    """One fixed turnover-successor arm owned by the paired-study route."""

    arm_id: str
    policy: RankBufferedScoreRiskCostDevelopmentRecipe
    rebalance_clock: RebalanceClock


def rank_buffered_r0_inner_candidates() -> tuple[RankBufferedR0InnerCandidate, ...]:
    """Construct the six preregistered R0-only inner candidates in fixed order."""
    daily = EveryFormationClock()
    every_five = EveryNFormationsClock(interval=5)
    definitions = (
        ("RANK_BUFFERED::DAILY::ENTRY100_EXIT150::UNCAPPED", 150, None, daily),
        ("RANK_BUFFERED::DAILY::ENTRY100_EXIT150::CAP_10_PERCENT", 150, 0.1, daily),
        ("RANK_BUFFERED::DAILY::ENTRY100_EXIT150::CAP_20_PERCENT", 150, 0.2, daily),
        ("RANK_BUFFERED::DAILY::ENTRY100_EXIT150::CAP_25_PERCENT", 150, 0.25, daily),
        ("RANK_BUFFERED::EVERY_5::ENTRY100_EXIT150::CAP_20_PERCENT", 150, 0.2, every_five),
        ("RANK_BUFFERED::DAILY::ENTRY100_EXIT200::CAP_20_PERCENT", 200, 0.2, daily),
    )
    return tuple(
        RankBufferedR0InnerCandidate(
            arm_id=arm_id,
            policy=RankBufferedScoreRiskCostDevelopmentRecipe.create(
                exit_rank=exit_rank, maximum_one_way_turnover=turnover_cap
            ),
            rebalance_clock=clock,
        )
        for arm_id, exit_rank, turnover_cap, clock in definitions
    )


def rank_buffered_r0_inner_candidate_set_hash() -> str:
    """Bind the fixed study route, rather than only its first policy recipe."""
    return str(
        canonical_hash(
            {
                "risk_method_id": "R0",
                "formation_count": _RANK_BUFFERED_INNER_FORMATION_COUNT,
                "selection_cost_bps": _RANK_BUFFERED_SELECTION_COST_BPS,
                "candidates": [
                    {
                        "arm_id": value.arm_id,
                        "policy_recipe_hash": value.policy.recipe_hash,
                        "rebalance_clock_binding_hash": value.rebalance_clock.binding.binding_hash,
                    }
                    for value in rank_buffered_r0_inner_candidates()
                ],
            }
        )
    )


def maximum_rank_buffered_r0_inner_solver_calls(formation_count: int) -> int:
    """Return the exact six-arm inner-only Direct/CVXPY admission ceiling."""
    if formation_count != _RANK_BUFFERED_INNER_FORMATION_COUNT:
        raise PairedPortfolioError("portfolio_strategy_lab.rank_buffered_inner_axis_invalid")
    return int(
        len(rank_buffered_r0_inner_candidates())
        * formation_count
        * DIRECT_OSQP_MAXIMUM_INVOCATIONS_PER_OPTIMIZATION
        + 7
        + 5
    )


def portfolio_study_formation_limit(
    policy_ids: tuple[str, ...],
) -> int | None:
    """Return the fixed inner axis limit for the installed turnover successor."""
    return (
        _RANK_BUFFERED_INNER_FORMATION_COUNT
        if policy_ids == ("RANK_BUFFERED_SCORE_RISK_COST",)
        else None
    )


def portfolio_solver_call_upper_bound(*, policy_ids: tuple[str, ...], formation_count: int) -> int:
    """Select the installed Portfolio study's budget from its declared policy route."""
    if policy_ids == ("RANK_BUFFERED_SCORE_RISK_COST",):
        return maximum_rank_buffered_r0_inner_solver_calls(formation_count)
    return maximum_paired_alpha_portfolio_solver_calls(formation_count)


def maximum_paired_alpha_portfolio_solver_calls(formation_count: int) -> int:
    """Return the sealed worst-case numerical-call budget for this experiment.

    Inner selection owns 18 unbanded Direct optimizations per formation.  The
    outer evidence owns 14 Direct optimizations per formation after expanding
    each hard Sector arm into its safety-only unconstrained companion.  The
    declared five-filter frontier adds five score-risk-cost optimizations per
    formation; its Top-K diagnostics do not call a solver.  Golden and
    covariance evidence add seven Direct and five CVXPY/OSQP calls.  Every
    Direct optimization may consume one accuracy retry; CVXPY calls may not.
    """
    if formation_count < 1:
        raise PairedPortfolioError("portfolio_strategy_lab.formation_axis_empty")
    selection_stop = formation_count * 2 // 3
    direct_optimizations = selection_stop * 18 + formation_count * 19 + 7
    return int(direct_optimizations * DIRECT_OSQP_MAXIMUM_INVOCATIONS_PER_OPTIMIZATION + 5)


def _safety_constraints_force_unique_weights(inputs: PairedAlphaPortfolioInputs) -> bool:
    """Prove the narrow case where budget/name-cap leave exactly one solution."""

    required_names = round(1.0 / 0.02)
    return all(
        int(np.count_nonzero(inputs.market.decision_eligible[row])) == required_names
        for row in range(len(inputs.formation_sessions))
    )


@dataclass(frozen=True, slots=True)
class _BacktestMandate:
    bootstrap_resamples: int = _BOOTSTRAP_RESAMPLES


@dataclass(frozen=True, slots=True)
class _BacktestWorkspace:
    mandate: _BacktestMandate
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    scores: dict[tuple[str, PortfolioScoreMode], FloatArray]
    covariances: FloatArray
    covariance_validation: CovarianceValidationProof
    decision_eligible: BoolArray
    execution_available: BoolArray
    realized_simple_returns: FloatArray
    causal_adv20: FloatArray
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray
    passive_returns_by_session: dict[date, FloatArray]


def sector_exposures(
    ordered_listing_ids: tuple[str, ...],
    ordered_sector_ids: tuple[str, ...],
    sector_by_listing_id: Mapping[str, str],
    sessions: tuple[date, ...] = (),
) -> tuple[FloatArray, FloatArray]:
    """The sector exposure matrix and the equal-weight book's exposure to each sector (V152).

    One row a sector and one column a listing, in their orders, each listing exposed to its one
    sector; the equal-weight exposure is that matrix applied to equal weights. Both read-only.
    A Sector history whose reclassification falls inside `sessions` gives one of each per
    session (V346).
    """
    exposure = sector_exposure(
        sector_by_listing_id, sessions, ordered_listing_ids, ordered_sector_ids
    )
    equal_weight: FloatArray = np.ascontiguousarray(
        exposure
        @ np.full(len(ordered_listing_ids), 1.0 / len(ordered_listing_ids), dtype=np.float64),
        dtype=np.float64,
    )
    equal_weight.setflags(write=False)
    return exposure, equal_weight


@dataclass(frozen=True, slots=True)
class ImmutablePortfolioMarketInputs:
    """Previously published market lanes decoded without a current pointer."""

    formation_sessions: tuple[date, ...]
    economic_formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    sector_ids: tuple[str, ...]
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray
    decision_eligible: BoolArray
    execution_available: BoolArray
    realized_simple_returns: FloatArray
    passive_returns_by_session: dict[date, FloatArray]
    causal_adv20: FloatArray
    tradability_bundle_hash: str
    universe_epoch_hash: str
    sector_revision: str
    source_surface_hash: str

    def __post_init__(self) -> None:
        """Require complete immutable finite passive returns on the declared economic axis.

        Raises:
            PairedPortfolioError: Economic support/order, decision inclusion, passive membership or
                passive array shape/mutability/finiteness differs.
        """
        passive = tuple(
            value
            for value in self.economic_formation_sessions
            if value not in set(self.formation_sessions)
        )
        if (
            self.economic_formation_sessions != tuple(sorted(set(self.economic_formation_sessions)))
            or not set(self.formation_sessions) <= set(self.economic_formation_sessions)
            or set(self.passive_returns_by_session) != set(passive)
            or any(
                values.shape != (len(self.ordered_listing_ids),)
                or values.flags.writeable
                or not np.isfinite(values).all()
                for values in self.passive_returns_by_session.values()
            )
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.market_economic_axis_invalid")


@dataclass(frozen=True, slots=True)
class ImmutablePortfolioBenchmark:
    """Same-clock SPY lane previously produced by the benchmark owner."""

    formation_sessions: tuple[date, ...]
    economic_formation_sessions: tuple[date, ...]
    simple_returns: tuple[float, ...]
    log_returns: tuple[float, ...]
    economic_simple_returns: tuple[float, ...]
    economic_log_returns: tuple[float, ...]
    surface_hash: str

    def __post_init__(self) -> None:
        """Require aligned benchmark axes and exact simple/log return equivalence.

        Raises:
            PairedPortfolioError: Lane lengths, economic support/order, finiteness, exact expm1
                equivalence or surface hash length differs.
        """
        if (
            len(self.formation_sessions) != len(self.simple_returns)
            or len(self.simple_returns) != len(self.log_returns)
            or len(self.economic_formation_sessions) != len(self.economic_simple_returns)
            or len(self.economic_simple_returns) != len(self.economic_log_returns)
            or not set(self.formation_sessions) <= set(self.economic_formation_sessions)
            or self.economic_formation_sessions
            != tuple(sorted(set(self.economic_formation_sessions)))
            or not np.isfinite(np.asarray(self.log_returns, dtype=np.float64)).all()
            or not np.isfinite(np.asarray(self.economic_log_returns, dtype=np.float64)).all()
            or not np.array_equal(
                np.asarray(self.simple_returns, dtype=np.float64),
                np.expm1(np.asarray(self.log_returns, dtype=np.float64)),
            )
            or not np.array_equal(
                np.asarray(self.economic_simple_returns, dtype=np.float64),
                np.expm1(np.asarray(self.economic_log_returns, dtype=np.float64)),
            )
            or len(self.surface_hash) != 64
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.benchmark_lane_invalid")


@dataclass(frozen=True, slots=True)
class DeclaredFilteredScore:
    """One Host-resolved score-filter surface; never accepted from YAML as a matrix."""

    spec_id: str
    surface_hash: str
    values: FloatArray


@dataclass(frozen=True, slots=True)
class PairedAlphaPortfolioInputs:
    """Host-built matrices bound to durable score, Risk and market handles."""

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    fold_indices: npt.NDArray[np.int64]
    economic_fold_indices: npt.NDArray[np.int64]
    decision_ranges: tuple[tuple[int, int], ...]
    passive_sessions: tuple[date, ...]
    dynamic_scores: FloatArray
    control_scores: FloatArray
    raw_dynamic_scores: FloatArray
    fixed_simple_scores: FloatArray
    declared_filtered_scores: tuple[DeclaredFilteredScore, ...]
    dynamic_score_surface_hash: str
    control_score_surface_hash: str
    raw_dynamic_score_surface_hash: str
    fixed_simple_score_surface_hash: str
    fixed_simple_score_binding_hash: str
    selected_aggregation_span: int
    r0_covariances: FloatArray
    r0_covariance_validation: CovarianceValidationProof
    r1_covariances: FloatArray | None
    r1_covariance_validation: CovarianceValidationProof | None
    r0_surface_hash: str
    r1_surface_hash: str | None
    market: ImmutablePortfolioMarketInputs
    benchmark: ImmutablePortfolioBenchmark
    state_transition: PortfolioStateTransitionBinding
    reference_mark: ReferenceMarkLane
    execution_events_hash: str
    alpha_score_temporal_handoff_hash: str
    strategy_schedule_hash: str
    causal_admission_hash: str
    input_binding_hash: str

    def __post_init__(self) -> None:
        """Require exact immutable common score, Risk, market, benchmark and transition authority.

        Raises:
            PairedPortfolioError: Common axes, contiguous decision ranges/passive gaps,
                score/filter/covariance shapes or proof ownership, read-only/finiteness, optional
                Risk pairing, aggregation span, reference marks or identity syntax differs.
        """
        sessions = len(self.formation_sessions)
        listings = len(self.ordered_listing_ids)
        if (
            sessions < 100
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or self.market.formation_sessions != self.formation_sessions
            or self.market.economic_formation_sessions != self.benchmark.economic_formation_sessions
            or self.market.ordered_listing_ids != self.ordered_listing_ids
            or self.benchmark.formation_sessions != self.formation_sessions
            or self.fold_indices.shape != (sessions,)
            or self.economic_fold_indices.shape != (len(self.market.economic_formation_sessions),)
            or tuple(self.market.passive_returns_by_session) != self.passive_sessions
            or len(self.passive_sessions) != len(self.decision_ranges) - 1
            or not self.decision_ranges
            or self.decision_ranges[0][0] != 0
            or self.decision_ranges[-1][1] != sessions
            or any(stop <= start for start, stop in self.decision_ranges)
            or any(
                left_stop != right_start
                for (_, left_stop), (right_start, _) in zip(
                    self.decision_ranges, self.decision_ranges[1:], strict=False
                )
            )
            or self.dynamic_scores.shape != (sessions, listings)
            or self.control_scores.shape != (sessions, listings)
            or self.raw_dynamic_scores.shape != (sessions, listings)
            or self.fixed_simple_scores.shape != (sessions, listings)
            or any(
                value.values.shape != (sessions, listings)
                or value.values.flags.writeable
                or not np.isfinite(value.values).all()
                or len(value.surface_hash) != 64
                for value in self.declared_filtered_scores
            )
            or len({value.spec_id for value in self.declared_filtered_scores})
            != len(self.declared_filtered_scores)
            or self.r0_covariances.shape != (sessions, listings, listings)
            or self.r0_covariance_validation.values is not self.r0_covariances
            or (
                self.r1_covariances is not None
                and self.r1_covariances.shape != (sessions, listings, listings)
            )
            or (
                (self.r1_covariances is None) != (self.r1_covariance_validation is None)
                or (
                    self.r1_covariances is not None
                    and self.r1_covariance_validation is not None
                    and self.r1_covariance_validation.values is not self.r1_covariances
                )
            )
            or any(
                value.flags.writeable
                for value in (
                    self.fold_indices,
                    self.economic_fold_indices,
                    self.dynamic_scores,
                    self.control_scores,
                    self.raw_dynamic_scores,
                    self.fixed_simple_scores,
                    self.r0_covariances,
                    *(() if self.r1_covariances is None else (self.r1_covariances,)),
                )
            )
            or not np.isfinite(self.dynamic_scores).all()
            or not np.isfinite(self.control_scores).all()
            or not np.isfinite(self.raw_dynamic_scores).all()
            or not np.isfinite(self.fixed_simple_scores).all()
            or not np.isfinite(self.r0_covariances).all()
            or (self.r1_covariances is not None and not np.isfinite(self.r1_covariances).all())
            or ((self.r1_covariances is None) != (self.r1_surface_hash is None))
            or self.selected_aggregation_span not in {1, 21, 42, 63}
            or self.reference_mark.state_transition_binding_hash
            != self.state_transition.binding_hash
            or self.reference_mark.marks_by_session is None
            or tuple(self.reference_mark.marks_by_session) != self.formation_sessions
            or any(
                len(value) != listings or not np.isfinite(value).all()
                for value in (
                    np.asarray(self.reference_mark.marks_by_session[session])
                    for session in self.formation_sessions
                )
            )
            or any(
                len(value) != 64
                for value in (
                    self.dynamic_score_surface_hash,
                    self.control_score_surface_hash,
                    self.raw_dynamic_score_surface_hash,
                    self.fixed_simple_score_surface_hash,
                    self.fixed_simple_score_binding_hash,
                    self.r0_surface_hash,
                    *(() if self.r1_surface_hash is None else (self.r1_surface_hash,)),
                    self.execution_events_hash,
                    self.alpha_score_temporal_handoff_hash,
                    self.strategy_schedule_hash,
                    self.causal_admission_hash,
                    self.input_binding_hash,
                )
            )
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.paired_input_axis_invalid")


@dataclass(frozen=True, slots=True)
class PairedAlphaPortfolioResearchResult:
    """Retain paired trial paths, oracle/regression evidence and benchmark facts.

    Retain paired research root, trial paths, oracle checks, covariance regression and benchmark
    facts.
    """

    root: PairedAlphaPortfolioResearchRoot
    trials: tuple[PairedPortfolioTrialEvidence, ...]
    golden: PortfolioGoldenOracleEvidence
    covariance_regression: ControlledCovarianceRegressionEvidence
    paired_decisions: tuple[PortfolioPairedDecisionEvidence, ...]
    spy_metrics: PortfolioEconomicMetricSet


def _readonly(values: npt.NDArray[Any], *, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    result = np.ascontiguousarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _recipe_hash(policy: PortfolioPolicyRecipe) -> str:
    existing = getattr(policy, "recipe_hash", None)
    if isinstance(existing, str):
        return existing
    dump = cast(Any, policy).model_dump(mode="json")
    return str(canonical_hash(dump))


def _workspace(
    *,
    inputs: PairedAlphaPortfolioInputs,
    scores: FloatArray,
    covariances: FloatArray,
    covariance_validation: CovarianceValidationProof,
) -> _BacktestWorkspace:
    finite = np.isfinite(scores)
    eligible = np.asarray(inputs.market.decision_eligible & finite, dtype=np.bool_)
    if bool(np.any(np.sum(eligible, axis=1) < 100)):
        raise PairedPortfolioError("portfolio_strategy_lab.top100_common_axis_insufficient")
    eligible.setflags(write=False)
    score_values = np.array(scores, copy=True)
    score_values[~finite] = np.nan
    score_values.setflags(write=False)
    return _BacktestWorkspace(
        mandate=_BacktestMandate(),
        formation_sessions=inputs.formation_sessions,
        ordered_listing_ids=inputs.ordered_listing_ids,
        scores={("PAIRED_ALPHA", PortfolioScoreMode.STOCK_ONLY): score_values},
        covariances=covariances,
        covariance_validation=covariance_validation,
        decision_eligible=eligible,
        execution_available=inputs.market.execution_available,
        realized_simple_returns=inputs.market.realized_simple_returns,
        causal_adv20=inputs.market.causal_adv20,
        sector_exposure_matrix=inputs.market.sector_exposure_matrix,
        equal_weight_sector_exposure=inputs.market.equal_weight_sector_exposure,
        passive_returns_by_session=inputs.market.passive_returns_by_session,
    )


def _pack(store: PortfolioResearchArtifactStore, *, category: str, value: FloatArray) -> str:
    return store.publish_array(
        category=category, values=np.ascontiguousarray(value, dtype=np.float64)
    )


def _anchor_returns(workspace: _BacktestWorkspace) -> FloatArray:
    result: FloatArray = np.empty(len(workspace.formation_sessions), dtype=np.float64)
    for index in range(len(result)):
        admitted = workspace.decision_eligible[index] & np.isfinite(
            workspace.realized_simple_returns[index]
        )
        if not bool(np.any(admitted)):
            raise PairedPortfolioError("portfolio_strategy_lab.anchor_axis_empty")
        result[index] = float(np.mean(workspace.realized_simple_returns[index, admitted]))
    return cast(FloatArray, _readonly(result, dtype=np.dtype(np.float64)))


@dataclass(frozen=True, slots=True)
class _EconomicPath:
    sessions: tuple[date, ...]
    gross_simple_returns: FloatArray
    one_way_turnovers: FloatArray
    predicted_variances: FloatArray
    benchmark_simple_returns: FloatArray
    anchor_simple_returns: FloatArray
    fold_indices: npt.NDArray[np.int64]
    passive_sessions: tuple[date, ...]
    missing_execution_count: int


def _prefix_sequence_axis(
    inputs: PairedAlphaPortfolioInputs, execution_stop: int
) -> tuple[tuple[tuple[int, int], ...], tuple[date, ...]]:
    ranges = tuple(
        (start, min(stop, execution_stop))
        for start, stop in inputs.decision_ranges
        if start < execution_stop
    )
    return ranges, inputs.passive_sessions[: max(0, len(ranges) - 1)]


def _rebalanced_formation_count(
    *, ranges: tuple[tuple[int, int], ...], clock: RebalanceClock
) -> int:
    return sum(
        clock.rebalances(formation_index=index, segment_start_index=start)
        for start, stop in ranges
        for index in range(start, stop)
    )


def _economic_path(
    *,
    inputs: PairedAlphaPortfolioInputs,
    workspace: _BacktestWorkspace,
    sequence: object,
) -> _EconomicPath:
    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        PortfolioWalkForwardSequenceResult,
    )

    resolved = cast(PortfolioWalkForwardSequenceResult, sequence)
    decision_anchor = _anchor_returns(workspace)
    benchmark_by_session = dict(
        zip(
            inputs.benchmark.economic_formation_sessions,
            inputs.benchmark.economic_simple_returns,
            strict=True,
        )
    )
    fold_by_session = dict(
        zip(
            inputs.market.economic_formation_sessions,
            inputs.economic_fold_indices,
            strict=True,
        )
    )
    sessions: list[date] = []
    gross: list[float] = []
    turnover: list[float] = []
    predicted: list[float] = []
    benchmark: list[float] = []
    anchor: list[float] = []
    folds: list[int] = []
    passive_sessions: list[date] = []
    for position, segment in enumerate(resolved.segments):
        for offset, decision_index in enumerate(range(segment.start_index, segment.stop_index)):
            session = inputs.formation_sessions[decision_index]
            sessions.append(session)
            gross.append(segment.gross_simple_returns[offset])
            turnover.append(segment.one_way_turnovers[offset])
            forecast = segment.predicted_variances[offset]
            predicted.append(float("nan") if forecast is None else forecast)
            benchmark.append(benchmark_by_session[session])
            anchor.append(float(decision_anchor[decision_index]))
            folds.append(fold_by_session[session])
        if position < len(resolved.passive_holds):
            hold = resolved.passive_holds[position]
            session = hold.formation_session
            passive_returns = workspace.passive_returns_by_session[session]
            finite = np.isfinite(passive_returns)
            if not bool(np.any(finite)):
                raise PairedPortfolioError("portfolio_strategy_lab.passive_anchor_axis_empty")
            sessions.append(session)
            gross.append(hold.gross_simple_return)
            turnover.append(float(hold.one_way_turnover))
            predicted.append(float("nan"))
            benchmark.append(benchmark_by_session[session])
            anchor.append(float(np.mean(passive_returns[finite])))
            folds.append(fold_by_session[session])
            passive_sessions.append(session)
    expected = tuple(
        value for value in inputs.market.economic_formation_sessions if value <= sessions[-1]
    )
    if tuple(sessions) != expected:
        raise PairedPortfolioError("portfolio_strategy_lab.economic_path_axis_invalid")
    return _EconomicPath(
        sessions=tuple(sessions),
        gross_simple_returns=cast(
            FloatArray, _readonly(np.asarray(gross), dtype=np.dtype(np.float64))
        ),
        one_way_turnovers=cast(
            FloatArray, _readonly(np.asarray(turnover), dtype=np.dtype(np.float64))
        ),
        predicted_variances=cast(
            FloatArray, _readonly(np.asarray(predicted), dtype=np.dtype(np.float64))
        ),
        benchmark_simple_returns=cast(
            FloatArray, _readonly(np.asarray(benchmark), dtype=np.dtype(np.float64))
        ),
        anchor_simple_returns=cast(
            FloatArray, _readonly(np.asarray(anchor), dtype=np.dtype(np.float64))
        ),
        fold_indices=cast(
            npt.NDArray[np.int64],
            _readonly(np.asarray(folds), dtype=np.dtype(np.int64)),
        ),
        passive_sessions=tuple(passive_sessions),
        missing_execution_count=sum(value.missed_execution_count for value in resolved.segments),
    )


def _audit_summary(
    *,
    audits: tuple[PortfolioObjectiveAudit, ...],
    unconstrained_targets: tuple[FloatArray | None, ...],
    final_targets: FloatArray,
    realized_returns: FloatArray,
    store: PortfolioResearchArtifactStore,
) -> PortfolioObjectiveAuditSummary | None:
    if not audits:
        return None
    packed = json.dumps(
        [asdict(value) for value in audits], sort_keys=True, separators=(",", ":")
    ).encode()
    packed_hash = store.publish_document(
        category="development/paired-alpha-portfolio/objective-audits", payload=packed
    )
    lower_upper_slacks = tuple(
        value
        for audit in audits
        for value in (*audit.sector_lower_slack, *audit.sector_upper_slack)
    )
    duals = tuple(value for audit in audits for value in audit.sector_dual)
    binding = sum(value.binding_sector_count > 0 for value in audits)
    pre_variance = float(np.mean([value.pre_constraint_predicted_variance for value in audits]))
    post_variance = float(np.mean([value.post_constraint_predicted_variance for value in audits]))
    impacts: list[float] = []
    for row, unconstrained in enumerate(unconstrained_targets):
        if unconstrained is not None:
            impacts.append(
                float(
                    (final_targets[row] - unconstrained)
                    @ np.where(np.isfinite(realized_returns[row]), realized_returns[row], 0.0)
                )
            )
    soft = [value.sector_penalty_term for value in audits if value.sector_penalty_term is not None]
    return PortfolioObjectiveAuditSummary(
        alpha_utility_term_mean=float(np.mean([value.alpha_utility_term for value in audits])),
        risk_penalty_term_mean=float(np.mean([value.risk_penalty_term for value in audits])),
        transaction_cost_term_mean=float(
            np.mean([value.transaction_cost_term for value in audits])
        ),
        turnover_regularization_term_mean=float(
            np.mean([value.turnover_regularization_term for value in audits])
        ),
        sector_penalty_status=("SOFT_PENALTY_INSTALLED" if soft else "NOT_APPLICABLE"),
        sector_penalty_term_mean=float(np.mean(soft)) if soft else None,
        reference_to_sector_unconstrained_l1_mean=float(
            np.mean([value.reference_to_sector_unconstrained_l1_distance for value in audits])
        ),
        reference_to_final_l1_mean=float(
            np.mean([value.reference_to_final_l1_distance for value in audits])
        ),
        sector_unconstrained_to_final_l1_mean=float(
            np.mean([value.sector_unconstrained_to_final_l1_distance for value in audits])
        ),
        binding_session_count=binding,
        binding_session_frequency=float(binding / len(audits)),
        minimum_sector_slack=min(lower_upper_slacks) if lower_upper_slacks else None,
        mean_absolute_sector_dual=(float(np.mean(np.abs(duals))) if duals else None),
        mean_displaced_name_count=float(np.mean([value.displaced_name_count for value in audits])),
        mean_displaced_weight_mass=float(
            np.mean([value.displaced_weight_mass for value in audits])
        ),
        pre_constraint_score_utility_mean=float(
            np.mean([value.pre_constraint_score_utility for value in audits])
        ),
        post_constraint_score_utility_mean=float(
            np.mean([value.post_constraint_score_utility for value in audits])
        ),
        pre_constraint_predicted_risk=float(math.sqrt(max(pre_variance, 0.0) * 252.0)),
        post_constraint_predicted_risk=float(math.sqrt(max(post_variance, 0.0) * 252.0)),
        mean_realized_return_impact=float(np.mean(impacts)) if impacts else 0.0,
        packed_session_audit_hash=packed_hash,
    )


def _execute_trial(
    *,
    program_hash: str,
    phase: Literal["INNER_RISK_SELECTION", "OUTER_DEVELOPMENT_EVALUATION"],
    arm_id: str,
    effect_dimension: PortfolioEffectDimension,
    inputs: PairedAlphaPortfolioInputs,
    scores: FloatArray,
    score_surface_hash: str,
    risk_method_id: Literal["R0", "R1"],
    covariances: FloatArray,
    covariance_validation: CovarianceValidationProof,
    risk_surface_hash: str,
    policy: PortfolioPolicyRecipe,
    execution_stop: int,
    evaluation_start: int,
    rebalance_clock: RebalanceClock | None = None,
    store: PortfolioResearchArtifactStore,
) -> PairedPortfolioTrialEvidence:
    workspace = _workspace(
        inputs=inputs,
        scores=scores,
        covariances=covariances,
        covariance_validation=covariance_validation,
    )
    provider = PortfolioPolicyDecisionProvider(
        workspace=workspace,
        candidate_id="PAIRED_ALPHA",
        score_mode=PortfolioScoreMode.STOCK_ONLY,
        policy=policy,
        record_targets=True,
    )
    ranges, passive_sessions = _prefix_sequence_axis(inputs, execution_stop)
    clock = rebalance_clock or cast(
        RebalanceClock,
        resolve_rebalance_clock(
            inputs.state_transition.rebalance_clock.clock_id,
            **dict(inputs.state_transition.rebalance_clock.parameters),
        ),
    )
    sequence = run_portfolio_walk_forward_segment_sequence_with_passive_holds(
        workspace=workspace,
        decision_provider=provider,
        ranges=ranges,
        passive_sessions=passive_sessions,
        rebalance_clock=clock,
        reference_mark=inputs.reference_mark,
    )
    economic = _economic_path(inputs=inputs, workspace=workspace, sequence=sequence)
    if (
        len(provider.recorded_targets) != execution_stop
        or len(provider.recorded_sector_unconstrained_targets) != execution_stop
        or len(provider.recorded_optimization_audits) != execution_stop
        or provider.decided_count != _rebalanced_formation_count(ranges=ranges, clock=clock)
        or any(
            hold.optimizer_call_count != 0 or hold.target_decision_count != 0
            for hold in sequence.passive_holds
        )
    ):
        # All three, because each is read positionally beside formation-aligned
        # arrays below. A short list there does not raise; it silently
        # attributes one formation's impact to another.
        raise PairedPortfolioError("portfolio_strategy_lab.target_receipt_incomplete")
    start = evaluation_start
    economic_start = economic.sessions.index(inputs.formation_sessions[start])
    gross = economic.gross_simple_returns[economic_start:]
    turnover = economic.one_way_turnovers[economic_start:]
    predicted_lane = economic.predicted_variances[economic_start:]
    predicted_support = np.isfinite(predicted_lane)
    predicted_variances = predicted_lane[predicted_support]
    benchmark = economic.benchmark_simple_returns[economic_start:]
    sessions = economic.sessions[economic_start:]
    if gross.size != len(sessions) or gross.size < 2:
        raise PairedPortfolioError("portfolio_strategy_lab.evaluation_axis_insufficient")
    anchor = economic.anchor_simple_returns[economic_start:]
    metrics = tuple(
        evaluate_raw_simple_return_path(
            gross_simple_returns=gross,
            one_way_turnovers=turnover,
            benchmark_simple_returns=benchmark,
            anchor_simple_returns=anchor,
            cost_bps=value,
        )
        for value in _COST_ARMS
    )
    net5 = gross - turnover * 5.0 / 10_000.0
    fold_values: list[tuple[int, float]] = []
    evaluation_folds = economic.fold_indices[economic_start:]
    for fold in sorted(set(int(value) for value in evaluation_folds)):
        selected = evaluation_folds == fold
        fold_values.append((fold, float(np.mean(net5[selected]))))
    midpoint = len(net5) // 2
    targets = np.vstack(provider.recorded_targets)
    # Sliced on the formation axis *first*, then filtered. Filtering before the
    # slice applied a formation index to a list a hold had already shortened, so
    # the evaluation window moved onto the wrong formations the moment a hold
    # existed -- the recorder is formation-aligned precisely so this slice means
    # what it says.
    audits = tuple(
        value
        for value in provider.recorded_optimization_audits[start:]
        if isinstance(value, PortfolioObjectiveAudit)
    )
    audit = _audit_summary(
        audits=audits,
        unconstrained_targets=tuple(provider.recorded_sector_unconstrained_targets[start:]),
        final_targets=targets[start:],
        realized_returns=inputs.market.realized_simple_returns[start:execution_stop],
        store=store,
    )
    path_lane = np.column_stack((gross, turnover, predicted_lane, benchmark, anchor))
    target_hash = _pack(
        store=store,
        category=_TARGET_WEIGHT_CATEGORY,
        value=targets,
    )
    reference_hash = _pack(
        store=store,
        category=_REFERENCE_WEIGHT_CATEGORY,
        value=np.vstack(provider.recorded_reference_weights),
    )
    path_hash = _pack(
        store=store,
        category=_PATH_LANE_CATEGORY,
        value=path_lane,
    )
    calibration = realized_to_predicted_variance_ratio(
        predicted=tuple(
            float(value) if supported else None
            for value, supported in zip(predicted_lane, predicted_support, strict=True)
        ),
        realized_simple=tuple(float(value) for value in gross),
    )
    supported_gross = gross[predicted_support]
    predicted_risk = (
        float(math.sqrt(max(float(np.mean(predicted_variances)), 0.0) * 252.0))
        if predicted_variances.size
        else 0.0
    )
    realized_risk = (
        float(math.sqrt(float(np.mean(np.square(supported_gross))) * 252.0))
        if supported_gross.size
        else 0.0
    )
    recipe_hash = _recipe_hash(policy)
    admission_hash: str | None = None
    admission_kind: Literal["NOT_APPLICABLE", "DIMENSIONLESS_SCORE_UTILITY_IDENTITY_BOUND"] = (
        "NOT_APPLICABLE"
    )
    if isinstance(policy, ScoreRiskCostDevelopmentRecipe):
        admission = AlphaUtilityUnitsAdmission.create(
            score_scale=policy.score_scale,
            risk_aversion=policy.risk_aversion,
            turnover_regularization=policy.turnover_regularization,
            transaction_cost_rate=policy.transaction_cost_rate,
            normalization_reference_id=policy.normalization_reference_id,
        )
        admission_hash = admission.admission_hash
        admission_kind = "DIMENSIONLESS_SCORE_UTILITY_IDENTITY_BOUND"
    return PairedPortfolioTrialEvidence.create(
        program_hash=program_hash,
        phase=phase,
        arm_id=arm_id,
        effect_dimension=effect_dimension,
        score_surface_hash=score_surface_hash,
        risk_method_id=risk_method_id,
        risk_surface_hash=risk_surface_hash,
        policy_id=policy.policy_id,
        policy_recipe_hash=recipe_hash,
        rebalance_clock_binding=(clock.binding if rebalance_clock is not None else None),
        alpha_utility_units_admission=admission_kind,
        alpha_utility_admission_hash=admission_hash,
        sector_capacity=cast(float | None, getattr(policy, "sector_capacity", None)),
        executed_sessions=economic.sessions,
        evaluated_sessions=sessions,
        decision_session_count=execution_stop,
        economic_session_count=len(economic.sessions),
        passive_hold_sessions=economic.passive_sessions,
        passive_optimizer_call_count=0,
        passive_target_decision_count=0,
        cost_metrics=metrics,
        predicted_risk=predicted_risk,
        realized_risk=realized_risk,
        realized_to_predicted_variance_ratio=calibration.ratio,
        variance_calibration_support_count=calibration.support_count,
        variance_calibration_excluded_count=calibration.excluded_count,
        mean_one_way_turnover=float(np.mean(turnover)),
        break_even_cost_bps=break_even_cost_bps(
            gross_simple_returns=gross, one_way_turnovers=turnover
        ),
        anchor_active_log_wealth_5bps=float(np.log1p(net5).sum() - np.log1p(anchor).sum()),
        fold_stability_net_5bps=tuple(fold_values),
        half_stability_net_5bps=(
            float(np.mean(net5[:midpoint])),
            float(np.mean(net5[midpoint:])),
        ),
        missing_execution_count=economic.missing_execution_count,
        solver_call_count=provider.solver_call_count,
        objective_audit=audit,
        target_weight_lane_hash=target_hash,
        reference_weight_lane_hash=reference_hash,
        path_lane_hash=path_hash,
    )


@dataclass(slots=True)
class _TargetReadbackDecisionProvider:
    targets: FloatArray
    covariances: FloatArray
    references: list[FloatArray]
    decision_count: int = 0

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray,
        decision_mode: Literal["REBALANCE", "HOLD"],
    ) -> PortfolioTargetDecision:
        if formation_index != self.decision_count:
            raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_decision_axis_invalid")
        target = self.targets[formation_index]
        reference = np.asarray(reference_weights, dtype=np.float64)
        reference.setflags(write=False)
        self.references.append(reference)
        self.decision_count += 1
        if decision_mode == "HOLD":
            if pretrade_weights is None or not np.array_equal(target, pretrade_weights):
                raise PairedPortfolioError(
                    "portfolio_strategy_lab.paired_replay_hold_target_invalid"
                )
            return PortfolioTargetDecision(
                target_weights=target,
                predicted_variance=None,
                decision_mode="HOLD",
            )
        return PortfolioTargetDecision(
            target_weights=target,
            predicted_variance=float(target @ self.covariances[formation_index] @ target),
        )


def _load_trial_lane(
    *,
    store: PortfolioResearchArtifactStore,
    category: str,
    content_hash: str,
    shape: tuple[int, int],
) -> FloatArray:
    payload = store.load_packed_bytes(category=category, content_hash=content_hash)
    expected_bytes = math.prod(shape) * np.dtype(np.float64).itemsize
    if len(payload) != expected_bytes:
        raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_lane_shape_invalid")
    return cast(
        FloatArray,
        _readonly(
            np.frombuffer(payload, dtype=np.float64).reshape(shape), dtype=np.dtype(np.float64)
        ),
    )


def _replay_trial(
    *,
    inputs: PairedAlphaPortfolioInputs,
    trial: PairedPortfolioTrialEvidence,
    store: PortfolioResearchArtifactStore,
) -> tuple[int, int, int]:
    if trial.reference_weight_lane_hash is None:
        raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_reference_lane_missing")
    decision_count = trial.decision_session_count
    listing_count = len(inputs.ordered_listing_ids)
    targets = _load_trial_lane(
        store=store,
        category=_TARGET_WEIGHT_CATEGORY,
        content_hash=trial.target_weight_lane_hash,
        shape=(decision_count, listing_count),
    )
    references = _load_trial_lane(
        store=store,
        category=_REFERENCE_WEIGHT_CATEGORY,
        content_hash=trial.reference_weight_lane_hash,
        shape=(decision_count, listing_count),
    )
    stored_path = _load_trial_lane(
        store=store,
        category=_PATH_LANE_CATEGORY,
        content_hash=trial.path_lane_hash,
        shape=(len(trial.evaluated_sessions), 5),
    )
    scores_by_hash = {
        inputs.dynamic_score_surface_hash: inputs.dynamic_scores,
        inputs.control_score_surface_hash: inputs.control_scores,
        inputs.raw_dynamic_score_surface_hash: inputs.raw_dynamic_scores,
        inputs.fixed_simple_score_surface_hash: inputs.fixed_simple_scores,
        **{value.surface_hash: value.values for value in inputs.declared_filtered_scores},
    }
    try:
        scores = scores_by_hash[trial.score_surface_hash]
    except KeyError as error:
        raise PairedPortfolioError(
            "portfolio_strategy_lab.paired_replay_score_surface_not_this_graph"
        ) from error
    covariances = inputs.r0_covariances if trial.risk_method_id == "R0" else inputs.r1_covariances
    covariance_validation = (
        inputs.r0_covariance_validation
        if trial.risk_method_id == "R0"
        else inputs.r1_covariance_validation
    )
    expected_risk_hash = (
        inputs.r0_surface_hash if trial.risk_method_id == "R0" else inputs.r1_surface_hash
    )
    if covariances is None or trial.risk_surface_hash != expected_risk_hash:
        raise PairedPortfolioError(
            "portfolio_strategy_lab.paired_replay_risk_surface_not_this_graph"
        )
    if covariances is None or covariance_validation is None:
        raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_risk_not_admitted")
    workspace = _workspace(
        inputs=inputs,
        scores=scores,
        covariances=covariances,
        covariance_validation=covariance_validation,
    )
    provider = _TargetReadbackDecisionProvider(
        targets=targets,
        covariances=covariances,
        references=[],
    )
    ranges, passive_sessions = _prefix_sequence_axis(inputs, decision_count)
    clock = (
        cast(
            RebalanceClock,
            resolve_rebalance_clock(
                trial.rebalance_clock_binding.clock_id,
                **dict(trial.rebalance_clock_binding.parameters),
            ),
        )
        if trial.rebalance_clock_binding is not None
        else cast(
            RebalanceClock,
            resolve_rebalance_clock(
                inputs.state_transition.rebalance_clock.clock_id,
                **dict(inputs.state_transition.rebalance_clock.parameters),
            ),
        )
    )
    sequence = run_portfolio_walk_forward_segment_sequence_with_passive_holds(
        workspace=workspace,
        decision_provider=provider,
        ranges=ranges,
        passive_sessions=passive_sessions,
        rebalance_clock=clock,
        reference_mark=inputs.reference_mark,
    )
    economic = _economic_path(inputs=inputs, workspace=workspace, sequence=sequence)
    try:
        evaluation_start = inputs.formation_sessions.index(trial.evaluated_sessions[0])
        economic_start = economic.sessions.index(inputs.formation_sessions[evaluation_start])
    except ValueError as error:
        raise PairedPortfolioError(
            "portfolio_strategy_lab.paired_replay_evaluation_axis_invalid"
        ) from error
    replay_path = np.column_stack(
        (
            economic.gross_simple_returns[economic_start:],
            economic.one_way_turnovers[economic_start:],
            economic.predicted_variances[economic_start:],
            economic.benchmark_simple_returns[economic_start:],
            economic.anchor_simple_returns[economic_start:],
        )
    )
    captured_references = np.vstack(provider.references)
    if (
        provider.decision_count != decision_count
        or economic.sessions[economic_start:] != trial.evaluated_sessions
        or economic.sessions != trial.executed_sessions
        or economic.passive_sessions != trial.passive_hold_sessions
        or not np.array_equal(captured_references, references)
        or not np.array_equal(replay_path[:, (0, 1, 3, 4)], stored_path[:, (0, 1, 3, 4)])
        or not np.allclose(
            replay_path[:, 2],
            stored_path[:, 2],
            rtol=1e-12,
            atol=1e-15,
            equal_nan=True,
        )
        or any(
            hold.optimizer_call_count != 0 or hold.target_decision_count != 0
            for hold in sequence.passive_holds
        )
    ):
        raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_state_economic_mismatch")
    replay_metrics = tuple(
        evaluate_raw_simple_return_path(
            gross_simple_returns=replay_path[:, 0],
            one_way_turnovers=replay_path[:, 1],
            benchmark_simple_returns=replay_path[:, 3],
            anchor_simple_returns=replay_path[:, 4],
            cost_bps=value,
        )
        for value in _COST_ARMS
    )
    if replay_metrics != trial.cost_metrics:
        raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_metric_mismatch")
    predicted = replay_path[:, 2]
    support = np.isfinite(predicted)
    calibration = realized_to_predicted_variance_ratio(
        predicted=tuple(
            float(value) if supported else None
            for value, supported in zip(predicted, support, strict=True)
        ),
        realized_simple=tuple(float(value) for value in replay_path[:, 0]),
    )
    supported_gross = replay_path[support, 0]
    predicted_risk = (
        float(math.sqrt(max(float(np.mean(predicted[support])), 0.0) * 252.0))
        if bool(np.any(support))
        else 0.0
    )
    realized_risk = (
        float(math.sqrt(float(np.mean(np.square(supported_gross))) * 252.0))
        if supported_gross.size
        else 0.0
    )
    if (
        calibration.support_count != trial.variance_calibration_support_count
        or calibration.excluded_count != trial.variance_calibration_excluded_count
        or (
            calibration.ratio is None
            or trial.realized_to_predicted_variance_ratio is None
            or not math.isclose(
                calibration.ratio,
                trial.realized_to_predicted_variance_ratio,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
        or not math.isclose(predicted_risk, trial.predicted_risk, rel_tol=1e-12, abs_tol=1e-15)
        or not math.isclose(realized_risk, trial.realized_risk, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_risk_metric_mismatch")
    return decision_count, len(economic.sessions), len(replay_metrics)


def replay_paired_alpha_portfolio_research(
    *,
    program_hash: str,
    inputs: PairedAlphaPortfolioInputs,
    root: PairedAlphaPortfolioResearchRoot,
    store: PortfolioResearchArtifactStore,
) -> PairedAlphaPortfolioReplayReceipt:
    """Re-run the causal state/economic path from durable decisions, with no solve."""
    transition = root.state_transition
    if (
        root.program_hash != program_hash
        or root.input_binding_hash != inputs.input_binding_hash
        or transition is None
        or transition.binding_hash != inputs.state_transition.binding_hash
        or transition.mark_surface_hash != inputs.state_transition.mark_surface_hash
        or root.execution_events_hash != inputs.execution_events_hash
        or root.alpha_score_temporal_handoff_hash != inputs.alpha_score_temporal_handoff_hash
        or root.strategy_schedule_hash != inputs.strategy_schedule_hash
        or root.causal_admission_hash != inputs.causal_admission_hash
    ):
        raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_authority_not_this_graph")
    totals = [0, 0, 0]
    for trial_hash in root.ordered_trial_hashes:
        trial = store.load(
            category=_TRIAL_CATEGORY,
            content_hash=trial_hash,
            model=PairedPortfolioTrialEvidence,
            identity_field="evidence_hash",
        )
        if trial.program_hash != program_hash:
            raise PairedPortfolioError("portfolio_strategy_lab.paired_replay_trial_not_this_graph")
        values = _replay_trial(inputs=inputs, trial=trial, store=store)
        totals = [left + right for left, right in zip(totals, values, strict=True)]
    receipt = PairedAlphaPortfolioReplayReceipt.create(
        program_hash=program_hash,
        root_hash=root.root_hash,
        input_binding_hash=inputs.input_binding_hash,
        state_transition_binding_hash=inputs.state_transition.binding_hash,
        session_mark_surface_hash=cast(str, inputs.state_transition.mark_surface_hash),
        execution_events_hash=inputs.execution_events_hash,
        verified_trial_count=len(root.ordered_trial_hashes),
        rederived_decision_count=totals[0],
        rederived_economic_record_count=totals[1],
        rederived_metric_set_count=totals[2],
    )
    store.publish(category=_REPLAY_CATEGORY, value=receipt, identity_field="receipt_hash")
    return receipt


def verify_paired_alpha_portfolio_graph(
    *,
    output_workspace: Path,
    store: PortfolioResearchArtifactStore,
    root_hash: str,
    fixed_simple_score_artifact_hash: str | None,
    formation_sessions: tuple[date, ...],
    replay_receipt_hash: str | None,
) -> PairedAlphaPortfolioResearchRoot:
    """Open every paired child and its exact Alpha/Market owner handle."""
    from alphalattice.foundation.market_data_ops.publication.session_marks import (
        SessionMarkArtifactStore,
    )
    from alphalattice.investment.alpha_research.experiments.development_artifacts import (
        AlphaDevelopmentArtifactStore,
    )

    root = store.load(
        category="development/paired-alpha-portfolio/roots",
        content_hash=root_hash,
        model=PairedAlphaPortfolioResearchRoot,
        identity_field="root_hash",
    )
    transition = root.state_transition
    if transition is not None:
        mark_store = SessionMarkArtifactStore(output_workspace / "portfolio-development")
        surface = mark_store.load_manifest(cast(str, transition.mark_surface_hash))
        mark_store.read_marks(
            surface,
            sessions=formation_sessions,
            listing_ids=surface.epoch.ordered_listing_ids,
        )
        if (
            transition.mark_manifest_ref != surface.manifest_uri
            or transition.mark_epoch_hash != surface.epoch.epoch_hash
            or transition.mark_price_basis != surface.price_basis
            or transition.mark_source_watermark_hash != surface.source_watermark_hash
            or transition.mark_availability_policy_hash != surface.availability_policy_hash
        ):
            raise PairedPortfolioError(
                "portfolio_strategy_lab.paired_state_transition_not_this_graph"
            )
    if fixed_simple_score_artifact_hash is not None:
        binding_hash = root.fixed_simple_score_binding_hash
        if binding_hash is None:
            raise PairedPortfolioError("portfolio_strategy_lab.paired_simple_score_not_admitted")
        binding, _artifact, _values = AlphaDevelopmentArtifactStore(
            output_workspace / "portfolio-development"
        ).load_simple_signed_score_values(
            artifact_hash=fixed_simple_score_artifact_hash,
            binding_hash=binding_hash,
        )
        if (
            root.fixed_simple_score_binding_hash != binding.binding_hash
            or root.fixed_simple_score_surface_hash != binding.values_identity
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.paired_simple_score_not_this_graph")
    replay_decision_count = 0
    replay_economic_record_count = 0
    replay_metric_set_count = 0
    for trial_hash in root.ordered_trial_hashes:
        trial = store.load(
            category=_TRIAL_CATEGORY,
            content_hash=trial_hash,
            model=PairedPortfolioTrialEvidence,
            identity_field="evidence_hash",
        )
        replay_decision_count += trial.decision_session_count
        replay_economic_record_count += trial.economic_session_count
        replay_metric_set_count += len(trial.cost_metrics)
        store.load_packed_bytes(
            category=_TARGET_WEIGHT_CATEGORY,
            content_hash=trial.target_weight_lane_hash,
        )
        store.load_packed_bytes(category=_PATH_LANE_CATEGORY, content_hash=trial.path_lane_hash)
        if trial.reference_weight_lane_hash is not None:
            store.load_packed_bytes(
                category=_REFERENCE_WEIGHT_CATEGORY,
                content_hash=trial.reference_weight_lane_hash,
            )
        if trial.objective_audit is not None:
            store.load_document(
                category="development/paired-alpha-portfolio/objective-audits",
                content_hash=trial.objective_audit.packed_session_audit_hash,
            )
    store.load(
        category="development/paired-alpha-portfolio/golden-oracle",
        content_hash=root.golden_oracle_evidence_hash,
        model=PortfolioGoldenOracleEvidence,
        identity_field="evidence_hash",
    )
    store.load(
        category="development/paired-alpha-portfolio/covariance-regressions",
        content_hash=root.controlled_covariance_evidence_hash,
        model=ControlledCovarianceRegressionEvidence,
        identity_field="evidence_hash",
    )
    for decision_hash in root.ordered_paired_decision_hashes:
        store.load(
            category="development/paired-alpha-portfolio/paired-decisions",
            content_hash=decision_hash,
            model=PortfolioPairedDecisionEvidence,
            identity_field="decision_hash",
        )
    if transition is not None:
        if replay_receipt_hash is None:
            raise PairedPortfolioError(
                "portfolio_strategy_lab.paired_strong_replay_receipt_missing"
            )
        replay = store.load(
            category=_REPLAY_CATEGORY,
            content_hash=replay_receipt_hash,
            model=PairedAlphaPortfolioReplayReceipt,
            identity_field="receipt_hash",
        )
        if (
            replay.program_hash != root.program_hash
            or replay.root_hash != root.root_hash
            or replay.input_binding_hash != root.input_binding_hash
            or replay.state_transition_binding_hash != transition.binding_hash
            or replay.session_mark_surface_hash != transition.mark_surface_hash
            or replay.execution_events_hash != root.execution_events_hash
            or replay.verified_trial_count != len(root.ordered_trial_hashes)
            or replay.rederived_decision_count != replay_decision_count
            or replay.rederived_economic_record_count != replay_economic_record_count
            or replay.rederived_metric_set_count != replay_metric_set_count
        ):
            raise PairedPortfolioError("portfolio_strategy_lab.paired_strong_replay_not_this_graph")
    return root


def _score_policy(
    *, risk: float, turnover: float, sector: float | None
) -> ScoreRiskCostDevelopmentRecipe:
    return ScoreRiskCostDevelopmentRecipe.create(
        risk_aversion=risk,
        turnover_regularization=turnover,
        transaction_cost_rate=_TRANSACTION_COST_RATE,
        sector_capacity=sector,
        score_scale=_SCORE_SCALE,
        normalization_reference_id="SESSION_ROLE_NORMALIZED_TARGET_Z_SCORE",
    )


def _select_grid(trials: tuple[PairedPortfolioTrialEvidence, ...]) -> PairedPortfolioTrialEvidence:
    return max(
        trials,
        key=lambda value: (
            next(item for item in value.cost_metrics if item.cost_bps == 5.0).annualized_return,
            -value.mean_one_way_turnover,
            value.policy_recipe_hash,
        ),
    )


def _golden_errors(
    *,
    inputs: PairedAlphaPortfolioInputs,
    policy: ScoreRiskCostDevelopmentRecipe,
    indices: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]:
    admission = AlphaUtilityUnitsAdmission.create(
        score_scale=policy.score_scale,
        risk_aversion=policy.risk_aversion,
        turnover_regularization=policy.turnover_regularization,
        transaction_cost_rate=policy.transaction_cost_rate,
        normalization_reference_id=policy.normalization_reference_id,
    )
    optimizer = PortfolioOptimizer()
    weight_errors: list[float] = []
    objective_errors: list[float] = []
    feasibility: list[float] = []
    direct_osqp_call_count = 0
    for index in indices:
        reference: FloatArray = np.zeros(len(inputs.ordered_listing_ids), dtype=np.float64)
        selected_indices = (
            stable_rank_buffered_top_k(
                scores=inputs.dynamic_scores[index],
                eligible=inputs.market.decision_eligible[index],
                previous_target_weights=None,
                top_k=policy.top_k,
                exit_rank=policy.exit_rank,
            )
            if isinstance(policy, RankBufferedScoreRiskCostDevelopmentRecipe)
            else None
        )
        arguments = {
            "scores": inputs.dynamic_scores[index],
            "covariance": inputs.r0_covariances[index],
            "covariance_validation": inputs.r0_covariance_validation,
            "reference_weights": reference,
            "decision_eligible": inputs.market.decision_eligible[index],
            "top_k": 100,
            "maximum_weight": 0.02,
            "risk_aversion": policy.risk_aversion,
            "turnover_regularization": policy.turnover_regularization,
            "transaction_cost_rate": policy.transaction_cost_rate,
            "score_scale": policy.score_scale,
            "alpha_utility_admission": admission,
        }
        if isinstance(policy, RankBufferedScoreRiskCostDevelopmentRecipe):
            direct = optimizer.solve_score_risk_cost(
                **arguments,
                selected_indices=selected_indices,
                liquidation_only_carry=True,
                maximum_one_way_turnover=policy.maximum_one_way_turnover,
                initial_deployment_exempt=policy.maximum_one_way_turnover is not None,
            )
        else:
            direct = optimizer.solve_score_risk_cost(**arguments)
        direct_osqp_call_count += direct.solver_call_count
        if isinstance(policy, RankBufferedScoreRiskCostDevelopmentRecipe):
            oracle = solve_with_cvxpy_oracle(
                **arguments,
                selected_indices=selected_indices,
                liquidation_only_carry=True,
                maximum_one_way_turnover=policy.maximum_one_way_turnover,
                initial_deployment_exempt=policy.maximum_one_way_turnover is not None,
            )
        else:
            oracle = solve_with_cvxpy_oracle(**arguments)
        oracle_objective = score_risk_cost_objective(
            weights=oracle,
            scores=inputs.dynamic_scores[index],
            covariance=inputs.r0_covariances[index],
            reference_weights=reference,
            score_scale=policy.score_scale,
            risk_aversion=policy.risk_aversion,
            turnover_regularization=policy.turnover_regularization,
            transaction_cost_rate=policy.transaction_cost_rate,
        )
        weight_errors.append(float(np.max(np.abs(direct.weights - oracle))))
        objective_errors.append(abs(direct.objective_value - oracle_objective))
        feasibility.append(
            max(
                abs(float(np.sum(oracle)) - 1.0),
                max(0.0, -float(np.min(oracle))),
                max(0.0, float(np.max(oracle)) - 0.02),
            )
        )
    return (
        tuple(weight_errors),
        tuple(objective_errors),
        tuple(feasibility),
        direct_osqp_call_count,
    )


def _golden(
    *,
    program_hash: str,
    inputs: PairedAlphaPortfolioInputs,
    policy: ScoreRiskCostDevelopmentRecipe,
) -> PortfolioGoldenOracleEvidence:
    calibration_indices = (0, len(inputs.formation_sessions) // 3)
    validation_indices = (
        len(inputs.formation_sessions) // 2,
        len(inputs.formation_sessions) * 3 // 4,
        len(inputs.formation_sessions) - 1,
    )
    calibration = _golden_errors(inputs=inputs, policy=policy, indices=calibration_indices)
    tolerances: PortfolioGoldenTolerances = measure_golden_tolerances(
        weight_errors=calibration[0],
        objective_errors=calibration[1],
        feasibility_residuals=calibration[2],
    )
    validation = _golden_errors(inputs=inputs, policy=policy, indices=validation_indices)
    validation_maximum_weight_error = max(validation[0])
    validation_maximum_objective_error = max(validation[1])
    validation_maximum_feasibility_residual = max(validation[2])
    admitted = (
        validation_maximum_weight_error <= tolerances.maximum_weight_error
        and validation_maximum_objective_error <= tolerances.objective_error
        and validation_maximum_feasibility_residual <= tolerances.feasibility_residual
    )
    return PortfolioGoldenOracleEvidence.create(
        program_hash=program_hash,
        calibration_sessions=tuple(
            inputs.formation_sessions[index] for index in calibration_indices
        ),
        validation_sessions=tuple(inputs.formation_sessions[index] for index in validation_indices),
        maximum_weight_tolerance=tolerances.maximum_weight_error,
        objective_tolerance=tolerances.objective_error,
        feasibility_tolerance=tolerances.feasibility_residual,
        validation_maximum_weight_error=validation_maximum_weight_error,
        validation_maximum_objective_error=validation_maximum_objective_error,
        validation_maximum_feasibility_residual=validation_maximum_feasibility_residual,
        direct_osqp_call_count=calibration[3] + validation[3],
        admitted=admitted,
    )


def _controlled_covariance(
    *,
    program_hash: str,
    inputs: PairedAlphaPortfolioInputs,
    policy: ScoreRiskCostDevelopmentRecipe,
) -> ControlledCovarianceRegressionEvidence:
    index = len(inputs.formation_sessions) // 2
    covariance = inputs.r0_covariances[index]
    diagonal = np.maximum(np.diag(covariance), 1e-10)
    multipliers = np.linspace(1.0, 1_000.0, diagonal.size)
    alternate = np.diag(diagonal * multipliers)
    reference: FloatArray = np.zeros(len(inputs.ordered_listing_ids), dtype=np.float64)
    admission = AlphaUtilityUnitsAdmission.create(
        score_scale=policy.score_scale,
        risk_aversion=policy.risk_aversion,
        turnover_regularization=policy.turnover_regularization,
        transaction_cost_rate=policy.transaction_cost_rate,
        normalization_reference_id=policy.normalization_reference_id,
    )
    optimizer = PortfolioOptimizer()
    selected_indices = (
        stable_rank_buffered_top_k(
            scores=inputs.dynamic_scores[index],
            eligible=inputs.market.decision_eligible[index],
            previous_target_weights=None,
            top_k=policy.top_k,
            exit_rank=policy.exit_rank,
        )
        if isinstance(policy, RankBufferedScoreRiskCostDevelopmentRecipe)
        else None
    )
    rank_buffer_arguments = (
        {
            "selected_indices": selected_indices,
            "liquidation_only_carry": True,
            "maximum_one_way_turnover": policy.maximum_one_way_turnover,
            "initial_deployment_exempt": policy.maximum_one_way_turnover is not None,
        }
        if isinstance(policy, RankBufferedScoreRiskCostDevelopmentRecipe)
        else {}
    )
    common = dict(
        scores=inputs.dynamic_scores[index],
        reference_weights=reference,
        decision_eligible=inputs.market.decision_eligible[index],
        top_k=_GRID.top_k,
        maximum_weight=_GRID.name_cap,
        risk_aversion=policy.risk_aversion,
        turnover_regularization=policy.turnover_regularization,
        transaction_cost_rate=policy.transaction_cost_rate,
        score_scale=policy.score_scale,
        alpha_utility_admission=admission,
        **rank_buffer_arguments,
    )
    original = optimizer.solve_score_risk_cost(
        covariance=covariance,
        covariance_validation=inputs.r0_covariance_validation,
        **common,
    )
    changed = optimizer.solve_score_risk_cost(covariance=alternate, **common)
    weight_difference = float(np.max(np.abs(original.weights - changed.weights)))
    objective_difference = abs(original.objective_value - changed.objective_value)
    if weight_difference <= 0.0 or objective_difference <= 0.0:
        raise PairedPortfolioError("portfolio_strategy_lab.controlled_covariance_not_consumed")
    return ControlledCovarianceRegressionEvidence.create(
        program_hash=program_hash,
        session=inputs.formation_sessions[index],
        source_covariance_hash=str(canonical_hash(covariance.tobytes().hex())),
        alternate_covariance_hash=str(canonical_hash(alternate.tobytes().hex())),
        objective_difference=objective_difference,
        maximum_weight_difference=weight_difference,
        direct_osqp_call_count=(original.solver_call_count + changed.solver_call_count),
    )


def _paired_decision(
    *,
    comparison_id: str,
    inputs: PairedAlphaPortfolioInputs,
    left: PairedPortfolioTrialEvidence,
    right: PairedPortfolioTrialEvidence,
    store: PortfolioResearchArtifactStore,
) -> PortfolioPairedDecisionEvidence:
    def net5(trial: PairedPortfolioTrialEvidence) -> FloatArray:
        payload = store.load_packed_bytes(
            category="development/paired-alpha-portfolio/path-lanes",
            content_hash=trial.path_lane_hash,
        )
        matrix: FloatArray = np.frombuffer(payload, dtype=np.float64).reshape(
            len(trial.evaluated_sessions), 5
        )
        return matrix[:, 0] - matrix[:, 1] * 5.0 / 10_000.0

    left_values = net5(left)
    right_values = net5(right)
    if left.evaluated_sessions != right.evaluated_sessions:
        raise PairedPortfolioError("portfolio_strategy_lab.paired_trial_axis_mismatch")
    differences = left_values - right_values
    block = dependence_aware_block_length(
        spans=(inputs.selected_aggregation_span,), holding_horizon_sessions=1
    )
    interval = circular_block_interval(
        tuple(float(value) for value in differences), block_length=block
    )
    disposition: Literal[
        "SUPERIOR_PAIRED_EVIDENCE",
        "INFERIOR_PAIRED_EVIDENCE",
        "INCONCLUSIVE_PAIRED_EVIDENCE",
    ]
    if interval[0] > 0.0:
        disposition = "SUPERIOR_PAIRED_EVIDENCE"
    elif interval[1] < 0.0:
        disposition = "INFERIOR_PAIRED_EVIDENCE"
    else:
        disposition = "INCONCLUSIVE_PAIRED_EVIDENCE"
    fold_by_session = dict(
        zip(
            inputs.market.economic_formation_sessions,
            inputs.economic_fold_indices,
            strict=True,
        )
    )
    folds = np.asarray(
        [fold_by_session[value] for value in left.evaluated_sessions], dtype=np.int64
    )
    fold_stability = tuple(
        float(np.mean(differences[folds == value])) for value in sorted(set(folds.tolist()))
    )
    midpoint = len(differences) // 2
    return PortfolioPairedDecisionEvidence.create(
        comparison_id=comparison_id,
        left_trial_hash=left.evidence_hash,
        right_trial_hash=right.evidence_hash,
        common_sessions=left.evaluated_sessions,
        point_difference=float(np.mean(differences)),
        paired_interval=interval,
        block_length=block,
        fraction_sessions_left_won=float(np.mean(differences > 0.0)),
        fold_stability=fold_stability,
        half_stability=(
            float(np.mean(differences[:midpoint])),
            float(np.mean(differences[midpoint:])),
        ),
        disposition=disposition,
    )


def _run_rank_buffered_r0_inner_research(
    *,
    program_hash: str,
    inputs: PairedAlphaPortfolioInputs,
    store: PortfolioResearchArtifactStore,
) -> PairedAlphaPortfolioResearchResult:
    """Run the preregistered R0 turnover screen on its fixed inner prefix."""

    if len(inputs.formation_sessions) != _RANK_BUFFERED_INNER_FORMATION_COUNT:
        raise PairedPortfolioError("portfolio_strategy_lab.rank_buffered_inner_axis_invalid")
    trials: list[PairedPortfolioTrialEvidence] = []
    for candidate in rank_buffered_r0_inner_candidates():
        trial = _execute_trial(
            program_hash=program_hash,
            phase="INNER_RISK_SELECTION",
            arm_id=candidate.arm_id,
            effect_dimension="RANK_BUFFERED_TURNOVER_SELECTION",
            inputs=inputs,
            scores=inputs.dynamic_scores,
            score_surface_hash=inputs.dynamic_score_surface_hash,
            risk_method_id="R0",
            covariances=inputs.r0_covariances,
            covariance_validation=inputs.r0_covariance_validation,
            risk_surface_hash=inputs.r0_surface_hash,
            policy=candidate.policy,
            execution_stop=len(inputs.formation_sessions),
            evaluation_start=0,
            rebalance_clock=candidate.rebalance_clock,
            store=store,
        )
        store.publish(category=_TRIAL_CATEGORY, value=trial, identity_field="evidence_hash")
        trials.append(trial)
    selected = _select_grid(tuple(trials))
    selected_policy = next(
        value.policy
        for value in rank_buffered_r0_inner_candidates()
        if value.policy.recipe_hash == selected.policy_recipe_hash
    )
    golden = _golden(program_hash=program_hash, inputs=inputs, policy=selected_policy)
    if not golden.admitted:
        raise PairedPortfolioError("portfolio_strategy_lab.golden_oracle_not_admitted")
    store.publish(
        category="development/paired-alpha-portfolio/golden-oracle",
        value=golden,
        identity_field="evidence_hash",
    )
    covariance = _controlled_covariance(
        program_hash=program_hash, inputs=inputs, policy=selected_policy
    )
    store.publish(
        category="development/paired-alpha-portfolio/covariance-regressions",
        value=covariance,
        identity_field="evidence_hash",
    )
    paired = tuple(
        _paired_decision(
            comparison_id=f"RANK_BUFFERED::{selected.arm_id}::MINUS::{trial.arm_id}",
            inputs=inputs,
            left=selected,
            right=trial,
            store=store,
        )
        for trial in trials
        if trial.evidence_hash != selected.evidence_hash
    )
    for value in paired:
        store.publish(
            category="development/paired-alpha-portfolio/paired-decisions",
            value=value,
            identity_field="decision_hash",
        )
    spy: FloatArray = np.asarray(inputs.benchmark.economic_simple_returns, dtype=np.float64)
    zeros: FloatArray = np.zeros(len(spy), dtype=np.float64)
    spy_metrics = evaluate_raw_simple_return_path(
        gross_simple_returns=spy,
        one_way_turnovers=zeros,
        benchmark_simple_returns=spy,
        cost_bps=0.0,
    )
    distinct_behaviors = len({value.target_weight_lane_hash for value in trials})
    if distinct_behaviors < 2:
        raise PairedPortfolioError("portfolio_strategy_lab.numerically_degenerate_grid")
    trial_direct_osqp_call_count = sum(value.solver_call_count for value in trials)
    direct_osqp_call_count = (
        trial_direct_osqp_call_count
        + golden.direct_osqp_call_count
        + covariance.direct_osqp_call_count
    )
    root = PairedAlphaPortfolioResearchRoot.create(
        program_hash=program_hash,
        input_binding_hash=inputs.input_binding_hash,
        dynamic_score_surface_hash=inputs.dynamic_score_surface_hash,
        control_score_surface_hash=None,
        raw_dynamic_score_surface_hash=None,
        fixed_simple_score_surface_hash=None,
        fixed_simple_score_binding_hash=None,
        declared_filter_surface_hashes=(),
        r0_surface_hash=inputs.r0_surface_hash,
        r1_surface_hash=None,
        benchmark_surface_hash=inputs.benchmark.surface_hash,
        state_transition=inputs.state_transition,
        execution_events_hash=inputs.execution_events_hash,
        alpha_score_temporal_handoff_hash=inputs.alpha_score_temporal_handoff_hash,
        strategy_schedule_hash=inputs.strategy_schedule_hash,
        causal_admission_hash=inputs.causal_admission_hash,
        inner_selection_stop=len(inputs.formation_sessions),
        evaluation_start=0,
        selected_r0_policy_recipe_hash=selected.policy_recipe_hash,
        selected_r1_best_achievable_recipe_hash=None,
        selection_study_type="RANK_BUFFERED_R0_INNER",
        ordered_trial_hashes=tuple(value.evidence_hash for value in trials),
        golden_oracle_evidence_hash=golden.evidence_hash,
        controlled_covariance_evidence_hash=covariance.evidence_hash,
        ordered_paired_decision_hashes=tuple(value.decision_hash for value in paired),
        spy_metrics=spy_metrics,
        risk_grid_distinct_behavior_count=distinct_behaviors,
        risk_grid_behavior_disposition="BEHAVIORALLY_DISTINCT",
        trial_direct_osqp_call_count=trial_direct_osqp_call_count,
        direct_osqp_call_count=direct_osqp_call_count,
        solver_call_count=direct_osqp_call_count + golden.cvxpy_osqp_call_count,
        cvxpy_oracle_call_count=golden.cvxpy_osqp_call_count,
    )
    store.publish(
        category="development/paired-alpha-portfolio/roots",
        value=root,
        identity_field="root_hash",
    )
    return PairedAlphaPortfolioResearchResult(
        root=root,
        trials=tuple(trials),
        golden=golden,
        covariance_regression=covariance,
        paired_decisions=paired,
        spy_metrics=spy_metrics,
    )


def run_paired_alpha_portfolio_research(
    *,
    program_hash: str,
    inputs: PairedAlphaPortfolioInputs,
    store: PortfolioResearchArtifactStore,
    portfolio_policy_ids: tuple[str, ...] = _GRID.portfolio_policy_ids,
) -> PairedAlphaPortfolioResearchResult:
    """Run orthogonal Policy, Risk and Sector-capacity evidence end to end."""
    if portfolio_policy_ids == RANK_BUFFERED_TURNOVER_SUCCESSOR_GRID.portfolio_policy_ids:
        return _run_rank_buffered_r0_inner_research(
            program_hash=program_hash,
            inputs=inputs,
            store=store,
        )
    if (
        portfolio_policy_ids != _GRID.portfolio_policy_ids
        or inputs.r1_covariances is None
        or inputs.r1_surface_hash is None
    ):
        raise PairedPortfolioError("portfolio_strategy_lab.paired_research_route_invalid")

    count = len(inputs.formation_sessions)
    selection_stop = count * 2 // 3
    evaluation_start = selection_stop + 1
    if count - evaluation_start < 82:
        raise PairedPortfolioError("portfolio_strategy_lab.outer_evaluation_axis_insufficient")
    trials: list[PairedPortfolioTrialEvidence] = []

    def run(
        *,
        phase: Literal["INNER_RISK_SELECTION", "OUTER_DEVELOPMENT_EVALUATION"],
        arm: str,
        dimension: PortfolioEffectDimension,
        score_values: FloatArray,
        score_hash: str,
        risk_id: Literal["R0", "R1"],
        policy: PortfolioPolicyRecipe,
        stop: int,
        start: int,
    ) -> PairedPortfolioTrialEvidence:
        trial = _execute_trial(
            program_hash=program_hash,
            phase=phase,
            arm_id=arm,
            effect_dimension=dimension,
            inputs=inputs,
            scores=score_values,
            score_surface_hash=score_hash,
            risk_method_id=risk_id,
            covariances=(inputs.r0_covariances if risk_id == "R0" else inputs.r1_covariances),
            covariance_validation=(
                inputs.r0_covariance_validation
                if risk_id == "R0"
                else cast(CovarianceValidationProof, inputs.r1_covariance_validation)
            ),
            risk_surface_hash=cast(
                str,
                inputs.r0_surface_hash if risk_id == "R0" else inputs.r1_surface_hash,
            ),
            policy=policy,
            execution_stop=stop,
            evaluation_start=start,
            store=store,
        )
        store.publish(
            category="development/paired-alpha-portfolio/trials",
            value=trial,
            identity_field="evidence_hash",
        )
        trials.append(trial)
        return trial

    r0_grid = tuple(
        run(
            phase="INNER_RISK_SELECTION",
            arm=f"R0_GRID::{risk:g}::{turnover:g}",
            dimension="R0_HYPERPARAMETER_SELECTION",
            score_values=inputs.dynamic_scores,
            score_hash=inputs.dynamic_score_surface_hash,
            risk_id="R0",
            policy=_score_policy(risk=risk, turnover=turnover, sector=None),
            stop=selection_stop,
            start=0,
        )
        for risk in _RISK_GRID
        for turnover in _TURNOVER_GRID
    )
    selected_r0 = _select_grid(r0_grid)
    installed_points = {
        value.recipe_hash: value
        for value in (
            _score_policy(risk=risk, turnover=turnover, sector=None)
            for risk in _RISK_GRID
            for turnover in _TURNOVER_GRID
        )
    }
    selected_r0_policy = installed_points[selected_r0.policy_recipe_hash]
    r1_grid = tuple(
        run(
            phase="INNER_RISK_SELECTION",
            arm=f"R1_BEST_ACHIEVABLE::{risk:g}::{turnover:g}",
            dimension="R1_BEST_ACHIEVABLE_DIAGNOSTIC",
            score_values=inputs.dynamic_scores,
            score_hash=inputs.dynamic_score_surface_hash,
            risk_id="R1",
            policy=_score_policy(risk=risk, turnover=turnover, sector=None),
            stop=selection_stop,
            start=0,
        )
        for risk in _RISK_GRID
        for turnover in _TURNOVER_GRID
    )
    selected_r1 = _select_grid(r1_grid)
    selected_r1_policy = installed_points[selected_r1.policy_recipe_hash]

    topk = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="POLICY::TOP_K_EQUAL_WEIGHT::R0::NONE",
        dimension="POLICY_EFFECT",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R0",
        policy=TopKEqualWeightPolicy(top_k=_GRID.top_k, maximum_weight=_GRID.name_cap),
        stop=count,
        start=evaluation_start,
    )
    minimum = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="POLICY::MINIMUM_VARIANCE::R0::NONE",
        dimension="POLICY_EFFECT",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R0",
        policy=TopKMinimumVariancePolicy(top_k=_GRID.top_k, maximum_weight=_GRID.name_cap),
        stop=count,
        start=evaluation_start,
    )
    score_r0 = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="POLICY_RISK::SCORE_RISK_COST::R0::NONE",
        dimension="POLICY_EFFECT",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R0",
        policy=selected_r0_policy,
        stop=count,
        start=evaluation_start,
    )
    score_r1 = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="RISK::SCORE_RISK_COST::R1::NONE::SHARED_R0_PARAMETERS",
        dimension="RISK_EFFECT",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R1",
        policy=selected_r0_policy,
        stop=count,
        start=evaluation_start,
    )
    score_r1_best = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="CAPABILITY::SCORE_RISK_COST::R1::NONE::R1_SELECTED_PARAMETERS",
        dimension="R1_BEST_ACHIEVABLE_DIAGNOSTIC",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R1",
        policy=selected_r1_policy,
        stop=count,
        start=evaluation_start,
    )
    score_r0_20 = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="SECTOR_CAPACITY::R0::20_PERCENT",
        dimension="SECTOR_CAPACITY_EFFECT",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R0",
        policy=_score_policy(
            risk=selected_r0_policy.risk_aversion,
            turnover=selected_r0_policy.turnover_regularization,
            sector=0.2,
        ),
        stop=count,
        start=evaluation_start,
    )
    score_r0_08 = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="SECTOR_CAPACITY::R0::8_PERCENT",
        dimension="SECTOR_CAPACITY_EFFECT",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R0",
        policy=_score_policy(
            risk=selected_r0_policy.risk_aversion,
            turnover=selected_r0_policy.turnover_regularization,
            sector=0.08,
        ),
        stop=count,
        start=evaluation_start,
    )
    minimum_20 = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="INTERACTION::MINIMUM_VARIANCE::R0::20_PERCENT",
        dimension="SECONDARY_INTERACTION",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R0",
        policy=MinimumVarianceDevelopmentRecipe.create(sector_capacity=0.2),
        stop=count,
        start=evaluation_start,
    )
    score_r1_20 = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="INTERACTION::SCORE_RISK_COST::R1::20_PERCENT",
        dimension="SECONDARY_INTERACTION",
        score_values=inputs.dynamic_scores,
        score_hash=inputs.dynamic_score_surface_hash,
        risk_id="R1",
        policy=_score_policy(
            risk=selected_r0_policy.risk_aversion,
            turnover=selected_r0_policy.turnover_regularization,
            sector=0.2,
        ),
        stop=count,
        start=evaluation_start,
    )
    control = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="DYNAMIC_CONTROL::RELATIVE_CONTROL::SCORE_RISK_COST::R0::NONE",
        dimension="DYNAMIC_CONTROL",
        score_values=inputs.control_scores,
        score_hash=inputs.control_score_surface_hash,
        risk_id="R0",
        policy=selected_r0_policy,
        stop=count,
        start=evaluation_start,
    )
    raw_topk = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="SIGNAL::RAW_SPAN1::TOP_K_EQUAL_WEIGHT::R0::NONE",
        dimension="SIGNAL_EFFECT",
        score_values=inputs.raw_dynamic_scores,
        score_hash=inputs.raw_dynamic_score_surface_hash,
        risk_id="R0",
        policy=TopKEqualWeightPolicy(top_k=_GRID.top_k, maximum_weight=_GRID.name_cap),
        stop=count,
        start=evaluation_start,
    )
    raw_score_r0 = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="SIGNAL::RAW_SPAN1::SCORE_RISK_COST::R0::NONE",
        dimension="SIGNAL_EFFECT",
        score_values=inputs.raw_dynamic_scores,
        score_hash=inputs.raw_dynamic_score_surface_hash,
        risk_id="R0",
        policy=selected_r0_policy,
        stop=count,
        start=evaluation_start,
    )
    fixed_simple = run(
        phase="OUTER_DEVELOPMENT_EVALUATION",
        arm="SIGNAL::FIXED_SIMPLE::SCORE_RISK_COST::R0::NONE",
        dimension="SIGNAL_EFFECT",
        score_values=inputs.fixed_simple_scores,
        score_hash=inputs.fixed_simple_score_surface_hash,
        risk_id="R0",
        policy=selected_r0_policy,
        stop=count,
        start=evaluation_start,
    )
    filter_trials: list[
        tuple[DeclaredFilteredScore, PairedPortfolioTrialEvidence, PairedPortfolioTrialEvidence]
    ] = []
    for filtered in inputs.declared_filtered_scores:
        filtered_topk = run(
            phase="OUTER_DEVELOPMENT_EVALUATION",
            arm=f"FILTER::{filtered.spec_id}::TOP_K_EQUAL_WEIGHT::R0::NONE",
            dimension="SCORE_FILTER_EFFECT",
            score_values=filtered.values,
            score_hash=filtered.surface_hash,
            risk_id="R0",
            policy=TopKEqualWeightPolicy(top_k=_GRID.top_k, maximum_weight=_GRID.name_cap),
            stop=count,
            start=evaluation_start,
        )
        filtered_qp = run(
            phase="OUTER_DEVELOPMENT_EVALUATION",
            arm=f"FILTER::{filtered.spec_id}::SCORE_RISK_COST::R0::NONE",
            dimension="SCORE_FILTER_EFFECT",
            score_values=filtered.values,
            score_hash=filtered.surface_hash,
            risk_id="R0",
            policy=selected_r0_policy,
            stop=count,
            start=evaluation_start,
        )
        filter_trials.append((filtered, filtered_topk, filtered_qp))
    golden = _golden(program_hash=program_hash, inputs=inputs, policy=selected_r0_policy)
    if not golden.admitted:
        raise PairedPortfolioError("portfolio_strategy_lab.golden_oracle_not_admitted")
    store.publish(
        category="development/paired-alpha-portfolio/golden-oracle",
        value=golden,
        identity_field="evidence_hash",
    )
    covariance = _controlled_covariance(
        program_hash=program_hash, inputs=inputs, policy=selected_r0_policy
    )
    store.publish(
        category="development/paired-alpha-portfolio/covariance-regressions",
        value=covariance,
        identity_field="evidence_hash",
    )
    comparison_trials = [
        ("POLICY::MINIMUM_VARIANCE_MINUS_TOP_K", minimum, topk),
        ("POLICY::SCORE_RISK_COST_MINUS_TOP_K", score_r0, topk),
        ("POLICY::SCORE_RISK_COST_MINUS_MINIMUM_VARIANCE", score_r0, minimum),
        ("RISK::R1_MINUS_R0_SHARED_PARAMETERS", score_r1, score_r0),
        ("SECTOR::20_PERCENT_MINUS_NONE", score_r0_20, score_r0),
        ("SECTOR::8_PERCENT_MINUS_NONE", score_r0_08, score_r0),
        ("SECTOR::8_PERCENT_MINUS_20_PERCENT", score_r0_08, score_r0_20),
        ("INTERACTION::POLICY_AT_20_PERCENT", score_r0_20, minimum_20),
        ("INTERACTION::RISK_AT_20_PERCENT", score_r1_20, score_r0_20),
        ("CAPABILITY::R1_BEST_MINUS_R1_SHARED", score_r1_best, score_r1),
        ("DYNAMIC::DYNAMIC_MINUS_RELATIVE_CONTROL", score_r0, control),
        ("SIGNAL::RAW_OPTIMIZER_MINUS_RAW_TOP_K", raw_score_r0, raw_topk),
        ("SIGNAL::AGGREGATED_MINUS_RAW", score_r0, raw_score_r0),
        ("SIGNAL::RAW_DYNAMIC_MINUS_FIXED_SIMPLE", raw_score_r0, fixed_simple),
        ("SIGNAL::AGGREGATED_DYNAMIC_MINUS_FIXED_SIMPLE", score_r0, fixed_simple),
    ]
    comparison_trials.extend(
        (
            f"FILTER::{filtered.spec_id}::QP_MINUS_TOP_K",
            filtered_qp,
            filtered_topk,
        )
        for filtered, filtered_topk, filtered_qp in filter_trials
    )
    comparison_trials.extend(
        (
            f"FILTER::{filtered.spec_id}::QP_MINUS_RAW_QP",
            filtered_qp,
            raw_score_r0,
        )
        for filtered, _filtered_topk, filtered_qp in filter_trials
    )
    paired = tuple(
        _paired_decision(
            comparison_id=comparison_id,
            inputs=inputs,
            left=left,
            right=right,
            store=store,
        )
        for comparison_id, left, right in comparison_trials
    )
    for value in paired:
        store.publish(
            category="development/paired-alpha-portfolio/paired-decisions",
            value=value,
            identity_field="decision_hash",
        )
    economic_start = inputs.benchmark.economic_formation_sessions.index(
        inputs.formation_sessions[evaluation_start]
    )
    spy: FloatArray = np.asarray(
        inputs.benchmark.economic_simple_returns[economic_start:], dtype=np.float64
    )
    zeros: FloatArray = np.zeros(len(spy), dtype=np.float64)
    spy_metrics = evaluate_raw_simple_return_path(
        gross_simple_returns=spy,
        one_way_turnovers=zeros,
        benchmark_simple_returns=spy,
        cost_bps=0.0,
    )
    distinct_behaviors = len({value.target_weight_lane_hash for value in (*r0_grid, *r1_grid)})
    constraint_degenerate = distinct_behaviors < 2 and _safety_constraints_force_unique_weights(
        inputs
    )
    if distinct_behaviors < 2 and not constraint_degenerate:
        raise PairedPortfolioError("portfolio_strategy_lab.numerically_degenerate_grid")
    trial_direct_osqp_call_count = sum(value.solver_call_count for value in trials)
    direct_osqp_call_count = (
        trial_direct_osqp_call_count
        + golden.direct_osqp_call_count
        + covariance.direct_osqp_call_count
    )
    root = PairedAlphaPortfolioResearchRoot.create(
        program_hash=program_hash,
        input_binding_hash=inputs.input_binding_hash,
        dynamic_score_surface_hash=inputs.dynamic_score_surface_hash,
        control_score_surface_hash=inputs.control_score_surface_hash,
        raw_dynamic_score_surface_hash=inputs.raw_dynamic_score_surface_hash,
        fixed_simple_score_surface_hash=inputs.fixed_simple_score_surface_hash,
        fixed_simple_score_binding_hash=inputs.fixed_simple_score_binding_hash,
        declared_filter_surface_hashes=tuple(
            value.surface_hash for value in inputs.declared_filtered_scores
        ),
        r0_surface_hash=inputs.r0_surface_hash,
        r1_surface_hash=inputs.r1_surface_hash,
        benchmark_surface_hash=inputs.benchmark.surface_hash,
        state_transition=inputs.state_transition,
        execution_events_hash=inputs.execution_events_hash,
        alpha_score_temporal_handoff_hash=inputs.alpha_score_temporal_handoff_hash,
        strategy_schedule_hash=inputs.strategy_schedule_hash,
        causal_admission_hash=inputs.causal_admission_hash,
        inner_selection_stop=selection_stop,
        evaluation_start=evaluation_start,
        selected_r0_policy_recipe_hash=selected_r0.policy_recipe_hash,
        selected_r1_best_achievable_recipe_hash=selected_r1.policy_recipe_hash,
        ordered_trial_hashes=tuple(value.evidence_hash for value in trials),
        golden_oracle_evidence_hash=golden.evidence_hash,
        controlled_covariance_evidence_hash=covariance.evidence_hash,
        ordered_paired_decision_hashes=tuple(value.decision_hash for value in paired),
        spy_metrics=spy_metrics,
        risk_grid_distinct_behavior_count=distinct_behaviors,
        risk_grid_behavior_disposition=(
            "CONSTRAINT_DEGENERATE" if constraint_degenerate else "BEHAVIORALLY_DISTINCT"
        ),
        trial_direct_osqp_call_count=trial_direct_osqp_call_count,
        direct_osqp_call_count=direct_osqp_call_count,
        solver_call_count=direct_osqp_call_count + golden.cvxpy_osqp_call_count,
        cvxpy_oracle_call_count=golden.cvxpy_osqp_call_count,
    )
    store.publish(
        category="development/paired-alpha-portfolio/roots",
        value=root,
        identity_field="root_hash",
    )
    return PairedAlphaPortfolioResearchResult(
        root=root,
        trials=tuple(trials),
        golden=golden,
        covariance_regression=covariance,
        paired_decisions=paired,
        spy_metrics=spy_metrics,
    )


__all__ = [
    "ControlledCovarianceRegressionEvidence",
    "DeclaredFilteredScore",
    "ImmutablePortfolioBenchmark",
    "ImmutablePortfolioMarketInputs",
    "PairedAlphaPortfolioInputs",
    "PairedAlphaPortfolioReplayReceipt",
    "PairedAlphaPortfolioResearchResult",
    "PairedAlphaPortfolioResearchRoot",
    "PairedPortfolioError",
    "PairedPortfolioTrialEvidence",
    "PortfolioGoldenOracleEvidence",
    "PortfolioObjectiveAuditSummary",
    "PortfolioPairedDecisionEvidence",
    "RankBufferedR0InnerCandidate",
    "maximum_paired_alpha_portfolio_solver_calls",
    "maximum_rank_buffered_r0_inner_solver_calls",
    "portfolio_solver_call_upper_bound",
    "portfolio_study_formation_limit",
    "rank_buffered_r0_inner_candidate_set_hash",
    "rank_buffered_r0_inner_candidates",
    "replay_paired_alpha_portfolio_research",
    "run_paired_alpha_portfolio_research",
    "verify_paired_alpha_portfolio_graph",
]
