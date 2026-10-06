"""Strategy Lab policy adapter for the deterministic Portfolio path engine."""

from __future__ import annotations

from typing import Literal, Protocol, cast

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    BoolArray,
    FloatArray,
    PortfolioBacktestWorkspace,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
    PortfolioWalkForwardResult,
    PortfolioWalkForwardSegmentResult,
    PortfolioWalkForwardState,
)
from alphalattice.capabilities.portfolio_backtesting.metrics import (
    evaluate_portfolio_walk_forward_segments,
)
from alphalattice.capabilities.portfolio_backtesting.state import (
    advance_portfolio_state_without_decision,
    initial_portfolio_walk_forward_state,
)
from alphalattice.investment.portfolio_strategy_lab.contracts import PortfolioScoreMode
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
    PortfolioOptimizationError,
    PortfolioOptimizer,
)
from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import (
    CausalRankReturnCurve,
)
from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
    PortfolioPolicyCatalog,
    build_installed_portfolio_policy_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    BoundPolicyDecisionInput,
    PortfolioPolicyRecipe,
)


class PortfolioWalkForwardWorkspace(PortfolioBacktestWorkspace, Protocol):
    @property
    def scores(self) -> dict[tuple[str, PortfolioScoreMode], FloatArray]: ...

    @property
    def covariances(self) -> FloatArray | None: ...

    @property
    def decision_eligible(self) -> BoolArray: ...


class PortfolioPolicyDecisionProvider:
    """Translate one frozen Strategy Lab policy into backtest target decisions."""

    def __init__(
        self,
        *,
        workspace: PortfolioWalkForwardWorkspace,
        candidate_id: str,
        score_mode: PortfolioScoreMode,
        policy: PortfolioPolicyRecipe,
        optimizer: PortfolioOptimizer | None = None,
        optimizer_construction_count: int = 0,
        policies: PortfolioPolicyCatalog | None = None,
        market_exposure: FloatArray | None = None,
        record_targets: bool = False,
        covariance_offset: int = 0,
        previous_target_weights: FloatArray | None = None,
    ) -> None:
        self.workspace = workspace
        self.covariance_offset = covariance_offset
        """The formation index of ``workspace.covariances``' first row.

        Zero for a workspace that holds every formation's covariance; a caller that projects
        one segment's at a time names where they start, so no formation reads another's (V310).
        """
        self.policy = policy
        self.policies = policies or build_installed_portfolio_policy_catalog()
        self.adapter = self.policies.resolve(policy)
        self.adapter_binding = self.adapter.describe_adapter_binding()
        if optimizer is not None and not self.adapter_binding.requires_optimizer:
            raise PortfolioWalkForwardError("portfolio_strategy_lab.closed_form_optimizer_injected")
        if optimizer_construction_count not in (0, 1) or (
            optimizer is None and optimizer_construction_count != 0
        ):
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.optimizer_construction_count_invalid"
            )
        self.optimizer = optimizer
        """Constructed only when this adapter's sealed input semantics requires it."""
        self.optimizer_construction_count = optimizer_construction_count
        self._pending_optimizer_construction_count = optimizer_construction_count
        """A campaign-owned construction is recorded on its first formation."""
        self.market_exposure = market_exposure
        """Per-formation raw SPY beta, or absent for a policy that never reads it."""

        self.decided_count = 0
        """Adapter invocations, which for a solver-backed policy is its solve count.

        Counted here because this is the only place that calls ``decide``. A
        Campaign that reported an optimizer count it derived from the formation
        axis would be reporting the axis, not the solver.
        """

        # Every recorded list below is **formation-aligned**: one entry per
        # call, holds included. Consumers slice them positionally against a
        # region width that counts holds, so a list that skipped one would shift
        # every entry after it onto the wrong formation -- or short-slice into
        # ``campaign_region_axis_invalid``.
        self.recorded_targets: list[FloatArray] = []
        self.recorded_reference_weights: list[FloatArray] = []
        """The book each formation actually priced turnover against, as received.

        Captured at the point of use rather than reconstructed afterwards. What
        the campaign publishes is held against this bit for bit, which is the
        difference between a durable lane that *is* the runtime array and one
        that re-derives the same declaration and hopes.
        """

        self.recorded_objective_values: list[float | None] = []
        """The optimizer service's own post-solve total, or ``None`` if unstated.

        Absent on a hold and on any adapter that does not carry one. Taken from
        the owner rather than rebuilt here from the audit's terms, so a term the
        objective later gains cannot be dropped by a second copy that still
        agrees with itself.
        """

        self.recorded_sector_unconstrained_targets: list[FloatArray | None] = []
        self.recorded_optimization_audits: list[object | None] = []
        """``None`` on a hold, which called no adapter and produced neither.

        Formation-aligned like the rest. An earlier version appended these on the
        decision branch alone and argued it was safe because nothing sliced them
        against a formation width -- which was wrong: the research loop slices
        the unconstrained targets from an *evaluation start* index and then reads
        them positionally beside formation-aligned targets and returns, so one
        hold moved every impact after it onto the wrong session. Fixing one
        member of a misaligned set and leaving the others is the same defect
        renamed.
        """
        self.solver_call_count = 0
        """Actual solver calls reported by the installed numerical owner."""

        self.recorded_solver_call_counts: list[int] = []
        self.recorded_optimizer_construction_counts: list[int] = []
        self.recorded_risk_forecast_required: list[bool] = []
        self._record_targets = record_targets
        self._previous_target: FloatArray | None = previous_target_weights
        """The last *intended* target, which is not the executed book.

        A policy with hysteresis compares against what it meant to hold, and a
        fill that partly missed makes those two different arrays. The provider
        already recorded targets for evidence and then threw the last one away.
        """

        try:
            self.score_matrix = workspace.scores[(candidate_id, score_mode)]
        except KeyError as error:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.score_stratum_unavailable"
            ) from error
        if self.score_matrix.shape != (
            len(workspace.formation_sessions),
            len(workspace.ordered_listing_ids),
        ):
            raise PortfolioWalkForwardError("portfolio_strategy_lab.workspace_axis_invalid")

    @property
    def previous_target_weights(self) -> FloatArray | None:
        """The last intended target, which a later segment's provider starts from."""
        return self._previous_target

    def _consume_optimizer_construction_count(self) -> int:
        """Attach a preconstructed campaign resource to exactly one formation."""

        count = self._pending_optimizer_construction_count
        self._pending_optimizer_construction_count = 0
        return count

    def __call__(
        self,
        *,
        formation_index: int,
        reference_weights: FloatArray,
        pretrade_weights: FloatArray | None = None,
        decision_mode: Literal["REBALANCE", "HOLD"] = "REBALANCE",
    ) -> PortfolioTargetDecision:
        if decision_mode == "HOLD":
            # No adapter call, so ``decided_count`` stays a solver count rather
            # than a formation count, and the turnover the engine measures on
            # this session is zero because the target *is* the pretrade book.
            #
            # ``predicted_variance`` is ``None``: a hold makes no forecast. The
            # number that used to sit here was ``held @ covariance @ held`` over
            # the *drifted* book, an ex-post quantity that needs the entry open
            # to exist, and it was flowing into decision-time risk calibration as
            # though a forecast had been made.
            held = pretrade_weights if pretrade_weights is not None else reference_weights
            decision = PortfolioTargetDecision(
                target_weights=held,
                predicted_variance=None,
                decision_mode="HOLD",
                requires_risk_forecast=self.adapter_binding.requires_risk_forecast,
            )
            # A hold calls no solver, so its count is zero rather than absent.
            # Appending only on the decision branch left this list short by one
            # per hold while the campaign sliced it against a width that counted
            # them -- so an every-N clock either short-sliced into
            # ``campaign_region_axis_invalid`` or, at interval two, published a
            # training region carrying the validation region's counts.
            self.recorded_solver_call_counts.append(0)
            self.recorded_optimizer_construction_counts.append(
                self._consume_optimizer_construction_count()
            )
            self.recorded_risk_forecast_required.append(self.adapter_binding.requires_risk_forecast)
            if self._record_targets:
                self.recorded_targets.append(decision.target_weights)
                self.recorded_reference_weights.append(reference_weights)
                self.recorded_objective_values.append(None)
                self.recorded_sector_unconstrained_targets.append(None)
                self.recorded_optimization_audits.append(None)
            return decision
        try:
            anchor_sector = self.workspace.equal_weight_sector_exposure
            if anchor_sector.ndim == 2:
                anchor_sector = anchor_sector[formation_index]
            # A Sector history's per-session exposure reads the formation's (V346).
            exposure = self.workspace.sector_exposure_matrix
            if exposure.ndim == 3:
                exposure = exposure[formation_index]
            covariance = None
            if self.adapter_binding.requires_risk_forecast:
                covariances = self.workspace.covariances
                if covariances is None:
                    raise PortfolioWalkForwardError(
                        "portfolio_strategy_lab.risk_forecast_covariance_not_provided"
                    )
                row = formation_index - self.covariance_offset
                if not 0 <= row < covariances.shape[0]:
                    raise PortfolioWalkForwardError(
                        "portfolio_strategy_lab.risk_forecast_covariance_axis_invalid"
                    )
                covariance = covariances[row]
            optimizer = self.optimizer
            constructed = self._consume_optimizer_construction_count()
            if self.adapter_binding.requires_optimizer and optimizer is None:
                optimizer = PortfolioOptimizer()
                self.optimizer = optimizer
                self.optimizer_construction_count += 1
                constructed += 1
            causal_curve = cast(
                CausalRankReturnCurve | None,
                getattr(self.workspace, "causal_rank_return_curve", None),
            )
            decision = self.adapter.decide(
                policy=self.policy,
                inputs=BoundPolicyDecisionInput(
                    scores=self.score_matrix[formation_index],
                    covariance=covariance,
                    covariance_validation=getattr(self.workspace, "covariance_validation", None),
                    decision_eligible=self.workspace.decision_eligible[formation_index],
                    reference_weights=reference_weights,
                    sector_exposure_matrix=exposure,
                    equal_weight_sector_exposure=anchor_sector,
                    market_exposure=(
                        None
                        if self.market_exposure is None
                        else self.market_exposure[formation_index]
                    ),
                    previous_target_weights=self._previous_target,
                    ordered_listing_ids=tuple(self.workspace.ordered_listing_ids),
                    formation_index=formation_index,
                    causal_rank_return_curve=(
                        None if causal_curve is None else causal_curve.at(formation_index)
                    ),
                ),
                optimizer=cast(PortfolioOptimizer, optimizer),
            )
        except PortfolioOptimizationError as error:
            raise PortfolioWalkForwardError(f"{error};formation_index={formation_index}") from error
        self.decided_count += 1
        self.solver_call_count += decision.solver_call_count
        self.recorded_solver_call_counts.append(decision.solver_call_count)
        self.recorded_optimizer_construction_counts.append(constructed)
        self.recorded_risk_forecast_required.append(decision.requires_risk_forecast)
        self._previous_target = decision.target_weights
        if self._record_targets:
            self.recorded_targets.append(decision.target_weights)
            self.recorded_reference_weights.append(reference_weights)
            self.recorded_objective_values.append(decision.optimizer_objective_value)
            self.recorded_sector_unconstrained_targets.append(decision.sector_unconstrained_weights)
            self.recorded_optimization_audits.append(decision.optimization_audit)
        return decision


__all__ = [
    "PortfolioWalkForwardError",
    "PortfolioWalkForwardResult",
    "PortfolioWalkForwardSegmentResult",
    "PortfolioWalkForwardState",
    "advance_portfolio_state_without_decision",
    "evaluate_portfolio_walk_forward_segments",
    "initial_portfolio_walk_forward_state",
]
