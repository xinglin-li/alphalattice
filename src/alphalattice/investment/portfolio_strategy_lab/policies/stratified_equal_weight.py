"""Stratified within-Sector Top-K, equally weighted: the control Policy A must beat.

The global Top-K equal-weight control answers "what do the best names do"; this
one answers "what does the same admission rule Policy A uses do, without the
optimizer". That is the comparison worth having. If Policy A cannot beat its own
candidate pool equally weighted, the QP is contributing nothing and the evidence
should say so rather than crediting the objective for the pool's work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
)
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import PortfolioOptimizer

from . import sector_admission
from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    portfolio_adapter_implementation_hash,
)
from .sector_admission import (
    derive_sector_safety_bands,
    sector_enter_exit_counts,
    stratified_admitted_pool,
)

STRATIFIED_TOP_K_EQUAL_WEIGHT_POLICY_ID = "STRATIFIED_TOP_K_EQUAL_WEIGHT"


class StratifiedPoolRecipe(Protocol):
    """The band geometry this control shares with Policy A."""

    top_k: int
    maximum_weight: float
    sector_absolute_deviation: float
    sector_relative_deviation: float


_TOLERANCE = 1e-8


class StratifiedTopKEqualWeightAdapter:
    """Apply sector-ranked hysteresis admission then equal allocation of free capital."""

    policy_id = STRATIFIED_TOP_K_EQUAL_WEIGHT_POLICY_ID
    solver_backed = False

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind sector admission and closed-form equal allocation implementations.

        Returns:
            Adapter declaring Risk forecast consumption and no optimizer dependency.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            # The admission module decides which names are held, so it is part of
            # this adapter's content in exactly the way the optimizer is part of a
            # solver-backed adapter's.
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
                (PORTFOLIO_POLICY_PACKAGE, Path(sector_admission.__file__)),
            ),
            recipe_schema_id="STRATIFIED_TOP_K_EQUAL_WEIGHT",
            solver_semantics="CLOSED_FORM_STRATIFIED_EQUAL_ALLOCATION",
            deterministic_policy={
                "selection": "within-sector-rank-with-hysteresis",
                "allocation": "equal-free-capital",
            },
            input_consumption_semantics="RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        """Preserve frozen carry and equally allocate admitted sector-ranked free capital.

        Args:
            policy: Concrete admitted installed recipe.
            inputs: Exact scores, eligibility, holdings reference and declared auxiliary lanes.
            optimizer: Deterministic solver owner; closed-form adapters do not use it.

        Returns:
            Read-only target and full admitted covariance predicted variance.

        Raises:
            PortfolioWalkForwardError: Admitted pool/free capital is empty or equal allocation
                exceeds the name cap.
        """
        del optimizer
        recipe = cast(StratifiedPoolRecipe, policy)
        bands = derive_sector_safety_bands(
            equal_weight_sector_exposure=inputs.equal_weight_sector_exposure,
            absolute_deviation=float(recipe.sector_absolute_deviation),
            relative_deviation=float(recipe.sector_relative_deviation),
        )
        enter, exit_counts = sector_enter_exit_counts(
            upper=bands.upper, maximum_weight=recipe.maximum_weight
        )
        sector_by_asset = np.argmax(inputs.sector_exposure_matrix, axis=0).astype(np.int64)
        admitted = stratified_admitted_pool(
            scores=inputs.scores,
            decision_eligible=inputs.decision_eligible,
            reference_weights=inputs.reference_weights,
            sector_by_asset=sector_by_asset,
            enter_counts=enter,
            exit_counts=exit_counts,
        )
        frozen = (~inputs.decision_eligible) & (inputs.reference_weights > _TOLERANCE)
        target = np.zeros_like(inputs.reference_weights)
        target[frozen] = inputs.reference_weights[frozen]
        allocatable = np.flatnonzero(admitted & inputs.decision_eligible)
        available = 1.0 - float(target.sum())
        if allocatable.size < 1 or available < -_TOLERANCE:
            raise PortfolioWalkForwardError("portfolio_strategy_lab.stratified_pool_empty")
        allocation = available / allocatable.size
        if allocation > recipe.maximum_weight + _TOLERANCE:
            raise PortfolioWalkForwardError("portfolio_strategy_lab.cap_infeasible")
        target[allocatable] = allocation
        target.setflags(write=False)
        return PortfolioTargetDecision(
            target_weights=target,
            predicted_variance=float(target @ inputs.require_covariance() @ target),
        )


__all__ = [
    "STRATIFIED_TOP_K_EQUAL_WEIGHT_POLICY_ID",
    "StratifiedPoolRecipe",
    "StratifiedTopKEqualWeightAdapter",
]
