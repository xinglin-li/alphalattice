"""The explicit tradable anchor: equal weight over the eligible Universe, every formation.

The current control computes each formation's eligible equal-weight return with
no durable trading state and no cost path, and it filters on *realized* execution
availability -- so it knows at decision time which orders were going to fill.
That is a diagnostic, not a portfolio.

This recipe answers the tradable version of the same question. It targets equal
weights across the causally decision-eligible listings, and then its intended
orders pass through exactly the execution, failed-fill, frozen-holding, cash,
turnover, state-carry and cost authorities the compared policy uses. A failed
fill stays failed rather than being redistributed with hindsight.

Because the old control filtered on hindsight, this successor is not required to
reproduce its numbers and must not be asked to. The legacy value stays legacy
readback; the causal anchor gets its own method and its own evidence identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    FloatArray,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
)

from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    portfolio_adapter_implementation_hash,
)

CURRENT_UNIVERSE_EQUAL_WEIGHT_POLICY_ID = "CURRENT_UNIVERSE_EQUAL_WEIGHT_EVERY_FORMATION"
_TOLERANCE = 1e-8


def current_universe_equal_weight_target(
    *, decision_eligible: FloatArray, reference_weights: FloatArray
) -> FloatArray:
    """Equal weight across every eligible name, with frozen holdings retained."""
    eligible = np.asarray(decision_eligible, dtype=np.bool_)
    frozen = (~eligible) & (reference_weights > _TOLERANCE)
    target = np.zeros_like(reference_weights)
    target[frozen] = reference_weights[frozen]
    allocatable = np.flatnonzero(eligible)
    available = 1.0 - float(target.sum())
    if allocatable.size < 1 or available < -_TOLERANCE:
        raise PortfolioWalkForwardError("portfolio_strategy_lab.anchor_pool_empty")
    target[allocatable] = available / allocatable.size
    target.setflags(write=False)
    return target


class CurrentUniverseEqualWeightAdapter:
    """Equally allocate admitted universe free capital without score selection.

    Allocate equal free capital across the admitted universe without score selection or
    optimization.
    """

    policy_id = CURRENT_UNIVERSE_EQUAL_WEIGHT_POLICY_ID
    solver_backed = False

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind this adapter's implementation, declared input consumption and allocation semantics.

        Returns:
            Exact adapter binding for every decision-eligible listing with equal free capital and
            frozen carry, with admitted covariance for predicted variance.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            # No optimizer in the closure: this adapter never selects and never
            # solves, so the optimizer module cannot change its numbers. Listing
            # it anyway would make the binding move for reasons that are not
            # about this policy.
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
            ),
            recipe_schema_id="CURRENT_UNIVERSE_EQUAL_WEIGHT_EVERY_FORMATION",
            solver_semantics="CLOSED_FORM_UNIVERSE_EQUAL_ALLOCATION",
            deterministic_policy={
                "selection": "none-every-eligible-listing",
                "allocation": "equal-free-capital",
                "hindsight_execution_filter": "ABSENT",
            },
            input_consumption_semantics="RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: object,
    ) -> PortfolioTargetDecision:
        """Allocate equal free capital with frozen carry and admitted Risk forecast.

        Apply every decision-eligible listing with equal free capital and frozen carry, with
        admitted covariance for predicted variance.

        Args:
            policy: Concrete admitted recipe for this installed adapter.
            inputs: Bound scores, eligibility, reference and declared auxiliary lanes.
            optimizer: Deterministic numerical owner; closed-form adapters do not use it.

        Returns:
            Target decision and the Risk forecast disposition produced by this allocation owner.

        Raises:
            PortfolioWalkForwardError: Required listing/formation, curve or covariance inputs are
                absent or incompatible.
        """
        del policy, optimizer
        target = current_universe_equal_weight_target(
            decision_eligible=inputs.decision_eligible,
            reference_weights=inputs.reference_weights,
        )
        return PortfolioTargetDecision(
            target_weights=target,
            predicted_variance=float(target @ inputs.require_covariance() @ target),
        )


__all__ = [
    "CURRENT_UNIVERSE_EQUAL_WEIGHT_POLICY_ID",
    "CurrentUniverseEqualWeightAdapter",
    "current_universe_equal_weight_target",
]
