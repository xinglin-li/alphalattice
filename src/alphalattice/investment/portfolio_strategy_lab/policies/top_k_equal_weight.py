"""Equal-weight top-K policy: the one installed family that needs no solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    BoolArray,
    FloatArray,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
)
from alphalattice.investment.portfolio_strategy_lab.optimizer import service as optimizer_service
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
    PortfolioOptimizer,
    stable_top_k,
)

from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    portfolio_adapter_implementation_hash,
)

_TOLERANCE = 1e-8


def equal_weight_target(
    *,
    scores: FloatArray,
    decision_eligible: BoolArray,
    reference_weights: FloatArray,
    top_k: int,
    maximum_weight: float,
) -> FloatArray:
    """Allocate free capital evenly across the eligible top-K selection."""
    selected = stable_top_k(scores, decision_eligible, top_k)
    frozen = (~decision_eligible) & (reference_weights > _TOLERANCE)
    target = np.zeros_like(reference_weights)
    target[frozen] = reference_weights[frozen]
    available_capital = 1.0 - float(target.sum())
    allocatable = np.asarray([value for value in selected if not frozen[value]], dtype=np.int64)
    if allocatable.size < 1 or available_capital < -_TOLERANCE:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.equal_weight_infeasible")
    allocation = available_capital / allocatable.size
    if allocation > maximum_weight + _TOLERANCE:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.cap_infeasible")
    target[allocatable] = allocation
    target.setflags(write=False)
    return target


class TopKEqualWeightAdapter:
    """Select stable top-k support and equally allocate free capital without solving a QP."""

    policy_id = "TOP_K_EQUAL_WEIGHT"
    solver_backed = False

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind equal allocation and stable selection code with explicit Risk consumption.

        Returns:
            Adapter binding including the module owning selection and requiring a forecast without
            optimization.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            # The optimizer module is part of this adapter's content even though
            # no solver runs: ``stable_top_k`` lives there and chooses which names
            # are held, so a change to it changes the holdings and the weights.
            # Hashing this module alone would have left that invisible -- the
            # same "same id, changed code" defect the binding exists to catch.
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
                (PORTFOLIO_POLICY_PACKAGE, Path(optimizer_service.__file__)),
            ),
            recipe_schema_id="TOP_K_EQUAL_WEIGHT",
            solver_semantics="CLOSED_FORM_EQUAL_ALLOCATION",
            deterministic_policy={"selection": "stable-top-k", "allocation": "equal-free-capital"},
            input_consumption_semantics="RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        """Resolve equal-weight targets and their owner-admitted covariance forecast.

        Args:
            policy: Concrete admitted installed recipe.
            inputs: Exact scores, eligibility, holdings reference and declared auxiliary lanes.
            optimizer: Deterministic solver owner; closed-form adapters do not use it.

        Returns:
            Target weights and predicted variance.

        Raises:
            PortfolioWalkForwardError: Selection, carry, cap or covariance inputs are not admitted.
        """
        del optimizer
        target = equal_weight_target(
            scores=inputs.scores,
            decision_eligible=inputs.decision_eligible,
            reference_weights=inputs.reference_weights,
            top_k=policy.top_k,
            maximum_weight=policy.maximum_weight,
        )
        return PortfolioTargetDecision(
            target_weights=target,
            predicted_variance=float(target @ inputs.require_covariance() @ target),
        )


__all__ = ["TopKEqualWeightAdapter", "equal_weight_target"]
