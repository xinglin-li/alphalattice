"""Top-K minimum-variance policy backed by the Desk-owned optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioTargetDecision
from alphalattice.investment.portfolio_strategy_lab.optimizer import service as optimizer_service
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
    PortfolioOptimizer,
    solver_numerical_environment,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    portfolio_adapter_implementation_hash,
)


class MinimumVarianceDevelopmentRecipe(BaseModel):  # type: ignore[misc]
    """Parameterize the installed policy for a hard Sector-capacity diagnostic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: Literal["TOP_K_MINIMUM_VARIANCE"] = "TOP_K_MINIMUM_VARIANCE"
    top_k: Literal[100] = 100
    maximum_weight: float = Field(default=0.02, gt=0.0, le=1.0)
    sector_capacity: float | None = Field(default=None, gt=0.0, le=1.0)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one minimum-variance development recipe.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical recipe_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = cls.model_construct(**values, recipe_hash="0" * 64)
        identity = draft.model_dump(mode="json", exclude={"recipe_hash"})
        return cls(**values, recipe_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the fixed name cap, admitted sector capacity and recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Maximum weight differs from 0.02, sector capacity is outside None/0.2 or
                recipe_hash differs.
        """
        if (
            self.maximum_weight != 0.02
            or self.sector_capacity not in {None, 0.2}
            or self.recipe_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        ):
            raise ValueError("portfolio_strategy_lab.minimum_variance_recipe_invalid")
        return self


class TopKMinimumVarianceAdapter:
    """Allocate an admitted top-k selection through the deterministic minimum-variance owner."""

    policy_id = "TOP_K_MINIMUM_VARIANCE"
    solver_backed = True

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind variance allocation code and exact solver numerical environment.

        Returns:
            Adapter binding including optimizer implementation, solver version/tolerances and
            iteration budget.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            # The optimizer is part of this adapter's content: it decides the
            # weights, so a change there that moved no adapter identity would be
            # the same "same id, changed code" defect one level down.
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
                (PORTFOLIO_POLICY_PACKAGE, Path(optimizer_service.__file__)),
            ),
            recipe_schema_id="TOP_K_MINIMUM_VARIANCE",
            solver_semantics="CONVEX_MINIMUM_VARIANCE",
            deterministic_policy={
                "objective": "minimum-variance",
                "selection": "stable-top-k",
                # The numerical environment, inside the binding hash. Two runs are
                # the same experiment only if the solver, its version, its
                # tolerances and its iteration budget were the same, so a change
                # to any of them moves the Program hash and reuse refuses rather
                # than silently mixing two numerical environments in one graph.
                "numerical_environment": solver_numerical_environment(),
            },
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        """Solve minimum variance with optional declared hard sector capacity.

        Args:
            policy: Concrete admitted recipe for this installed adapter.
            inputs: Bound scores, eligibility, reference and declared auxiliary lanes.
            optimizer: Deterministic numerical owner; closed-form adapters do not use it.

        Returns:
            Verified target, predicted variance, objective audit, solve count and safety-only sector
            comparison.

        Raises:
            PortfolioOptimizationError: Covariance, selection, sector feasibility or solution
                admission fails.
        """
        sector_capacity = cast(float | None, getattr(policy, "sector_capacity", None))
        solution = optimizer.minimum_variance(
            scores=inputs.scores,
            covariance=inputs.require_covariance(),
            reference_weights=inputs.reference_weights,
            decision_eligible=inputs.decision_eligible,
            top_k=policy.top_k,
            maximum_weight=policy.maximum_weight,
            sector_matrix=(inputs.sector_exposure_matrix if sector_capacity is not None else None),
            sector_reference=(
                inputs.equal_weight_sector_exposure if sector_capacity is not None else None
            ),
            sector_capacity=sector_capacity,
        )
        return PortfolioTargetDecision(
            target_weights=solution.weights,
            predicted_variance=solution.predicted_variance,
            optimization_audit=solution.objective_audit,
            solver_call_count=solution.solver_call_count,
            sector_unconstrained_weights=solution.sector_unconstrained_weights,
        )


__all__ = ["MinimumVarianceDevelopmentRecipe", "TopKMinimumVarianceAdapter"]
