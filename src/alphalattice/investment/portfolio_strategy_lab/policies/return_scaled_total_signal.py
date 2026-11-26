"""Policy A: the return-scaled total-signal global QP under bands and a beta guardrail.

    maximize   s @ w  -  lambda_r * w' Sigma w  -  c * ||w - w_ref||_1 / 2
    subject to w >= 0, sum(w) = 1, w_i <= w_max,
               L_g <= sum_{i in g} w_i <= U_g,
               untradable existing holdings frozen,
               new allocations only inside the admitted pool,
               |beta @ w - 1| <= guardrail.

``lambda_r`` is an empirical, identity-bound conversion between return-scaled
utility and variance, not a number with a known value. The stock component has
return units but is not calibrated, so no universal 2-5 is assumed; the bounded
domain is declared by the capability and every trial records the value it used.

A global covariance scale is irrelevant to minimum-variance weights and is
therefore *not* something ``lambda_r`` corrects: it is absorbed here only in the
sense that the search finds whichever trade-off this covariance's units imply.
Publishing that as a calibration fix would be a different claim entirely.

The Sector bounds are static safety limits. Sector expected return already enters
``s``; letting the bands encode it again would apply one view twice, with the
second application invisible in the objective.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioTargetDecision
from alphalattice.investment.portfolio_strategy_lab.optimizer import service as optimizer_service
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
    PortfolioOptimizationError,
    PortfolioOptimizer,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

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

RETURN_SCALED_TOTAL_SIGNAL_POLICY_ID = "RETURN_SCALED_TOTAL_SIGNAL_GLOBAL_QP"


class BandedTotalSignalRecipe(Protocol):
    """The structural view this adapter reads, beyond the shared policy seam."""

    top_k: int
    maximum_weight: float
    sector_absolute_deviation: float
    sector_relative_deviation: float
    risk_aversion: float
    transaction_cost_rate: float
    beta_guardrail: float


REFERENCE_BETA_GUARDRAIL = 0.10
"""``abs(beta @ w - 1) <= 0.10``. Frozen before any Portfolio outcome was seen."""

SENSITIVITY_BETA_GUARDRAIL = 0.20
"""Sensitivity only. Never the reference, and never a relaxation after a result."""


class ReturnScaledTotalSignalRecipe(BaseModel):  # type: ignore[misc]
    """One frozen Policy A trial: cap, bands, risk trade-off, cost and guardrail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ReturnScaledTotalSignalRecipe"] = "ReturnScaledTotalSignalRecipe"
    top_k: int = Field(ge=1)
    """Nominal, for the policy seam. Admission is per-Sector, so the realized
    pool size varies by formation and is recorded in the trial, not here."""

    maximum_weight: float = Field(gt=0.0, le=1.0)
    sector_absolute_deviation: float = Field(ge=0.0, le=0.5)
    sector_relative_deviation: float = Field(ge=0.0, le=2.0)
    risk_aversion: float = Field(gt=0.0)
    transaction_cost_rate: float = Field(ge=0.0, le=0.05)
    beta_guardrail: float = Field(gt=0.0, le=1.0)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def policy_id(self) -> str:
        """Read the installed return-scaled total-signal policy identity.

        Returns:
            Exact declared policy identity.
        """
        return RETURN_SCALED_TOTAL_SIGNAL_POLICY_ID

    @classmethod
    def create(
        cls,
        *,
        top_k: int,
        maximum_weight: float,
        sector_absolute_deviation: float,
        sector_relative_deviation: float,
        risk_aversion: float,
        transaction_cost_rate: float,
        beta_guardrail: float = REFERENCE_BETA_GUARDRAIL,
    ) -> Self:
        """Seal total-signal allocation, sector safety and market-exposure controls.

        Args:
            top_k: Declared selection size.
            maximum_weight: Single-name upper bound.
            sector_absolute_deviation: Absolute sector safety tolerance.
            sector_relative_deviation: Relative sector safety tolerance.
            risk_aversion: Covariance penalty multiplier.
            transaction_cost_rate: Turnover cost rate.
            beta_guardrail: Allowed market-exposure deviation from one.

        Returns:
            Validated concrete recipe with canonical recipe_hash.
        """
        values: dict[str, object] = {
            "kind": "ReturnScaledTotalSignalRecipe",
            "top_k": top_k,
            "maximum_weight": maximum_weight,
            "sector_absolute_deviation": sector_absolute_deviation,
            "sector_relative_deviation": sector_relative_deviation,
            "risk_aversion": risk_aversion,
            "transaction_cost_rate": transaction_cost_rate,
            "beta_guardrail": beta_guardrail,
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact return-scaled policy recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: recipe_hash differs.
        """
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ValueError("portfolio_strategy_lab.policy_a_recipe_identity_invalid")
        return self


class ReturnScaledTotalSignalAdapter:
    """Solve admitted total signal with hard sector bands and a market-exposure guardrail."""

    policy_id = RETURN_SCALED_TOTAL_SIGNAL_POLICY_ID
    solver_backed = True

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind total-signal, sector admission and optimizer implementation semantics.

        Returns:
            Exact solver-backed adapter binding with static safety bands and beta guardrail.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
                (PORTFOLIO_POLICY_PACKAGE, Path(sector_admission.__file__)),
                (PORTFOLIO_POLICY_PACKAGE, Path(optimizer_service.__file__)),
            ),
            recipe_schema_id="RETURN_SCALED_TOTAL_SIGNAL_GLOBAL_QP",
            solver_semantics="CONVEX_BANDED_TOTAL_SIGNAL_WITH_BETA_GUARDRAIL",
            deterministic_policy={
                "objective": "total-signal-risk-cost",
                "admission": "within-sector-rank-with-hysteresis",
                "sector_bounds": "STATIC_SAFETY_BANDS",
                "beta_guardrail": "ABSOLUTE_DEVIATION_FROM_ONE",
            },
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        """Admit sector-ranked hysteresis support and solve total signal inside safety bands.

        Args:
            policy: Concrete admitted installed recipe.
            inputs: Exact scores, eligibility, holdings reference and declared auxiliary lanes.
            optimizer: Deterministic solver owner; closed-form adapters do not use it.

        Returns:
            Verified target, predicted variance and retained actual solver-call count.

        Raises:
            PortfolioOptimizationError: Market exposure is absent or allocation/input/constraint
                admission fails.
        """
        recipe = cast(BandedTotalSignalRecipe, policy)
        if inputs.market_exposure is None:
            raise PortfolioOptimizationError("portfolio_strategy_lab.exposure_input_missing")
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
        guardrail = float(recipe.beta_guardrail)
        solution = optimizer.solve_banded_total_signal(
            scores=np.where(np.isfinite(inputs.scores), inputs.scores, 0.0),
            covariance=inputs.require_covariance(),
            reference_weights=inputs.reference_weights,
            decision_eligible=inputs.decision_eligible,
            admitted=admitted,
            sector_membership=inputs.sector_exposure_matrix,
            sector_lower=bands.lower,
            sector_upper=bands.upper,
            exposure=inputs.market_exposure,
            exposure_lower=1.0 - guardrail,
            exposure_upper=1.0 + guardrail,
            maximum_weight=recipe.maximum_weight,
            risk_aversion=float(recipe.risk_aversion),
            transaction_cost_rate=float(recipe.transaction_cost_rate),
        )
        return PortfolioTargetDecision(
            target_weights=solution.weights,
            predicted_variance=solution.predicted_variance,
            # Reported, not dropped. This adapter is solver-backed and returned a
            # count of zero, so every published row read ``ADAPTER_DECIDED``
            # beside ``solver_call_count = 0`` -- and the replay, which infers
            # solver backing from that count, rebuilt every row as
            # ``NOT_SOLVER_BACKED`` and refused a graph nobody forged.
            solver_call_count=solution.solver_call_count,
        )


__all__ = [
    "REFERENCE_BETA_GUARDRAIL",
    "RETURN_SCALED_TOTAL_SIGNAL_POLICY_ID",
    "SENSITIVITY_BETA_GUARDRAIL",
    "BandedTotalSignalRecipe",
    "ReturnScaledTotalSignalAdapter",
    "ReturnScaledTotalSignalRecipe",
]
