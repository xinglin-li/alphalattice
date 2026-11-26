"""Score/risk/cost policies, with and without a sector-deviation penalty.

Both families reach the same optimizer entry point and differ only in the sector
penalty they bind, so they stay in one module rather than duplicating the solver
call. Each still installs as its own capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioTargetDecision
from alphalattice.investment.portfolio_strategy_lab.optimizer import service as optimizer_service
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import (
    AlphaUtilityUnitsAdmission,
    PortfolioOptimizer,
    solver_numerical_environment,
    stable_rank_buffered_top_k,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    PortfolioSelectionAllocationSemantics,
    portfolio_adapter_implementation_hash,
)


def _score_risk_cost_implementation_hash() -> str:
    """This module plus the optimizer, which is where both families' numbers come from."""

    return portfolio_adapter_implementation_hash(
        (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
        (PORTFOLIO_POLICY_PACKAGE, Path(optimizer_service.__file__)),
    )


class ScoreRiskCostRecipe(Protocol):
    """Expose the score/Risk/cost recipe parameters consumed by deterministic allocation."""

    top_k: int
    maximum_weight: float
    risk_aversion: float
    turnover_regularization: float


class ScoreRiskCostDevelopmentRecipe(BaseModel):  # type: ignore[misc]
    """Development extension of the installed policy, not a successor policy id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: Literal["TOP_K_SCORE_RISK_COST", "RANK_BUFFERED_SCORE_RISK_COST"] = (
        "TOP_K_SCORE_RISK_COST"
    )
    top_k: Literal[100] = 100
    maximum_weight: float = Field(default=0.02, gt=0.0, le=1.0)
    risk_aversion: float = Field(gt=0.0)
    turnover_regularization: float = Field(ge=0.0)
    transaction_cost_rate: float = Field(ge=0.0, le=0.01)
    sector_capacity: float | None = Field(default=None, gt=0.0, le=1.0)
    score_scale: float = Field(gt=0.0)
    normalization_reference_id: str = Field(min_length=1, max_length=128)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one admitted score/Risk/cost recipe.

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
        """Require the declared development parameter grid and exact recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Policy ID, fixed 0.02 name cap, admitted Risk/turnover grids, sector
                capacity or recipe_hash differs.
        """
        if (
            self.policy_id != "TOP_K_SCORE_RISK_COST"
            or self.maximum_weight != 0.02
            or self.risk_aversion not in {100.0, 1_000.0, 10_000.0}
            or self.turnover_regularization not in {0.001, 0.1, 10.0}
            or self.sector_capacity not in {None, 0.08, 0.2}
            or self.recipe_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        ):
            raise ValueError("portfolio_strategy_lab.score_risk_cost_recipe_invalid")
        return self


class RankBufferedScoreRiskCostDevelopmentRecipe(ScoreRiskCostDevelopmentRecipe):
    """The fixed R0 turnover successor, with explicit carry and cap semantics."""

    policy_id: Literal["RANK_BUFFERED_SCORE_RISK_COST"] = "RANK_BUFFERED_SCORE_RISK_COST"
    exit_rank: Literal[150, 200]
    maximum_one_way_turnover: float | None = Field(default=None, gt=0.0, le=1.0)
    outside_selection_policy: Literal["LIQUIDATION_ONLY_CARRY"] = "LIQUIDATION_ONLY_CARRY"
    initial_deployment_disposition: Literal["INITIAL_DEPLOYMENT_EXEMPT"] = (
        "INITIAL_DEPLOYMENT_EXEMPT"
    )
    risk_aversion: float = 100.0
    turnover_regularization: float = 10.0
    transaction_cost_rate: float = 0.0005
    sector_capacity: Literal[None] = None
    score_scale: float = 0.01
    normalization_reference_id: Literal["SESSION_ROLE_NORMALIZED_TARGET_Z_SCORE"] = (
        "SESSION_ROLE_NORMALIZED_TARGET_Z_SCORE"
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the measured exit/turnover ladder and fixed numerical utility parameters.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Exit/turnover pair, name cap, Risk/turnover/cost/score multipliers or
                recipe_hash differs.
        """
        if (
            (self.exit_rank, self.maximum_one_way_turnover)
            not in {(150, None), (150, 0.1), (150, 0.2), (150, 0.25), (200, 0.2)}
            or self.maximum_weight != 0.02
            or self.risk_aversion != 100.0
            or self.turnover_regularization != 10.0
            or self.transaction_cost_rate != 0.0005
            or self.score_scale != 0.01
            or self.recipe_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"recipe_hash"}))
        ):
            raise ValueError("portfolio_strategy_lab.rank_buffered_score_risk_cost_recipe_invalid")
        return self


class SectorDeviationRecipe(ScoreRiskCostRecipe, Protocol):
    """Extend score/Risk/cost controls with an explicit sector deviation penalty."""

    sector_deviation_penalty: float


def _score_utility_coefficient(policy: ScoreRiskCostRecipe) -> float:
    """What one unit of score is worth, or the identity of not saying.

    ``1.0`` is not a silent default: it is exactly the behaviour every published
    trial ran with while ``score_scale`` sat on the solver unused by this path,
    and stating it in one place is what keeps those trials reproducible.

    It matters most when the score is *dimensionless*. Against a return-unit
    score, ``risk_aversion`` alone is a meaningful trade-off. Against a
    cross-sectional z it is variance per unit of nothing, and the coefficient is
    the only parameter that reads as a quantity -- what one cross-sectional
    standard deviation of score is worth in the same units as the variance and
    cost terms it is traded against.
    """

    development_scale = getattr(policy, "score_scale", None)
    if development_scale is not None:
        return float(development_scale)
    return float(getattr(policy, "score_utility_coefficient", 1.0))


@dataclass(frozen=True, slots=True)
class ScoreRiskCostObjectiveCoefficients:
    """What this policy prices each term of its objective at.

    One resolution, used by the solver that maximizes the objective and by the
    evidence that reports it. ``transaction_cost_rate`` is the one that had a
    ``getattr`` default written out at both call sites, and the two disagreed:
    the adapter resolved ``0.001`` -- which is what every
    ``TOP_K_SCORE_RISK_COST`` trial actually ran at, since its recipe carries no
    such field -- while the formation record resolved ``0.0`` and published an
    objective short by ``0.001 x predicted turnover`` on every point.

    A default written twice is two defaults. This is the one.
    """

    score_scale: float
    risk_aversion: float
    turnover_regularization: float
    transaction_cost_rate: float


def resolve_objective_coefficients(
    policy: ScoreRiskCostRecipe,
) -> ScoreRiskCostObjectiveCoefficients:
    """Read every objective coefficient off one recipe, in one place."""
    return ScoreRiskCostObjectiveCoefficients(
        score_scale=_score_utility_coefficient(policy),
        risk_aversion=float(policy.risk_aversion),
        turnover_regularization=float(policy.turnover_regularization),
        transaction_cost_rate=float(getattr(policy, "transaction_cost_rate", 0.001)),
    )


def _solve(
    *,
    policy: ScoreRiskCostRecipe,
    inputs: BoundPolicyDecisionInput,
    optimizer: PortfolioOptimizer,
    sector_deviation_penalty: float,
) -> PortfolioTargetDecision:
    coefficients = resolve_objective_coefficients(policy)
    admission = (
        AlphaUtilityUnitsAdmission.create(
            score_scale=coefficients.score_scale,
            risk_aversion=coefficients.risk_aversion,
            turnover_regularization=coefficients.turnover_regularization,
            transaction_cost_rate=coefficients.transaction_cost_rate,
            # Read here rather than on the coefficients: it prices no term of
            # the objective, it identifies the admission, and it has exactly one
            # reader -- so the "a default written twice is two defaults"
            # argument does not reach it.
            normalization_reference_id=str(
                getattr(policy, "normalization_reference_id", "LEGACY_INSTALLED_SCORE_SCALE")
            ),
        )
        if coefficients.score_scale > 0.0
        else None
    )
    selected_indices = None
    liquidation_only_carry = False
    maximum_one_way_turnover = None
    initial_deployment_exempt = False
    if isinstance(policy, RankBufferedScoreRiskCostDevelopmentRecipe):
        selected_indices = stable_rank_buffered_top_k(
            scores=inputs.scores,
            eligible=inputs.decision_eligible,
            previous_target_weights=inputs.previous_target_weights,
            top_k=policy.top_k,
            exit_rank=policy.exit_rank,
        )
        liquidation_only_carry = True
        maximum_one_way_turnover = policy.maximum_one_way_turnover
        initial_deployment_exempt = (
            policy.maximum_one_way_turnover is not None and inputs.previous_target_weights is None
        )
    solution = optimizer.solve_score_risk_cost(
        scores=inputs.scores,
        covariance=inputs.require_covariance(),
        covariance_validation=inputs.covariance_validation,
        reference_weights=inputs.reference_weights,
        decision_eligible=inputs.decision_eligible,
        top_k=policy.top_k,
        maximum_weight=policy.maximum_weight,
        risk_aversion=coefficients.risk_aversion,
        turnover_regularization=coefficients.turnover_regularization,
        transaction_cost_rate=coefficients.transaction_cost_rate,
        sector_matrix=inputs.sector_exposure_matrix,
        sector_reference=inputs.equal_weight_sector_exposure,
        sector_deviation_penalty=sector_deviation_penalty,
        sector_capacity=cast(float | None, getattr(policy, "sector_capacity", None)),
        score_scale=coefficients.score_scale,
        alpha_utility_admission=admission,
        selected_indices=selected_indices,
        liquidation_only_carry=liquidation_only_carry,
        maximum_one_way_turnover=maximum_one_way_turnover,
        initial_deployment_exempt=initial_deployment_exempt,
    )
    return PortfolioTargetDecision(
        target_weights=solution.weights,
        predicted_variance=solution.predicted_variance,
        optimizer_objective_value=float(solution.objective_value),
        optimization_audit=solution.objective_audit,
        solver_call_count=solution.solver_call_count,
        sector_unconstrained_weights=solution.sector_unconstrained_weights,
    )


class ScoreRiskCostTrialRecipe(BaseModel):  # type: ignore[misc]
    """The Stage 6 trial recipe: the frozen family's parameters plus the coefficient.

    Seals its own ``recipe_hash`` the way ``ReturnScaledTotalSignalRecipe`` does,
    and reports ``policy_id`` so the installed catalog routes it to the same
    adapter. Owning its identity here is what lets the coefficient exist without
    touching the frozen ``PortfolioPolicySpec`` union that published
    regularization evidence decodes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ScoreRiskCostTrialRecipe"] = "ScoreRiskCostTrialRecipe"
    top_k: int = Field(ge=1, le=500)
    maximum_weight: float = Field(gt=0.0, le=1.0)
    risk_aversion: float = Field(gt=0.0)
    turnover_regularization: float = Field(ge=0.0)
    score_utility_coefficient: float = Field(ge=0.0)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one admitted score/Risk/cost recipe.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical recipe_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("recipe_hash", None)
        provisional = cls.model_construct(**draft, recipe_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"recipe_hash"})
        return cls(**draft, recipe_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require feasible top-k name capacity and exact trial recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: top_k times maximum_weight is below one or recipe_hash differs.
        """
        if self.top_k * self.maximum_weight < 1.0:
            raise ValueError("portfolio_strategy_lab.score_risk_cost_cap_infeasible")
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ValueError("portfolio_strategy_lab.score_risk_cost_recipe_invalid")
        return self

    @property
    def policy_id(self) -> str:
        """Read the score/Risk/cost trial policy identity.

        Returns:
            TOP_K_SCORE_RISK_COST.
        """
        return "TOP_K_SCORE_RISK_COST"


class TopKScoreRiskCostAdapter:
    """Allocate stable top-k scores through the admitted convex score/Risk/cost owner."""

    policy_id = "TOP_K_SCORE_RISK_COST"
    solver_backed = True
    recipe_schema_id = "TOP_K_SCORE_RISK_COST"
    solver_semantics = "CONVEX_SCORE_RISK_COST"
    deterministic_policy: ClassVar[dict[str, str]] = {
        "objective": "score-risk-cost",
        "sector_penalty": "DISABLED",
    }
    selection_allocation_semantics: ClassVar[PortfolioSelectionAllocationSemantics] = (
        PortfolioSelectionAllocationSemantics.create(
            score_preselection_method="STABLE_FINITE_SCORE_TOP_K"
        )
    )

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind score/Risk/cost allocation and the declared solver environment.

        Returns:
            Exact adapter identity with zero sector penalty and explicit preselection semantics.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=_score_risk_cost_implementation_hash(),
            recipe_schema_id=self.recipe_schema_id,
            solver_semantics=self.solver_semantics,
            # The two families share a module and a solver call and differ in the
            # sector penalty they bind, so the penalty is what separates their
            # bindings -- not the class each happens to be defined as.
            deterministic_policy={
                **self.deterministic_policy,
                "numerical_environment": solver_numerical_environment(),
            },
            selection_allocation_semantics=self.selection_allocation_semantics,
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        """Solve admitted score/Risk/cost allocation with zero sector penalty.

        Args:
            policy: Concrete admitted installed recipe.
            inputs: Exact scores, eligibility, holdings reference and declared auxiliary lanes.
            optimizer: Deterministic solver owner; closed-form adapters do not use it.

        Returns:
            Verified targets and retained Risk/solver/objective diagnostics.

        Raises:
            PortfolioOptimizationError: Input, utility, selection, feasibility or solution admission
                fails.
        """
        return _solve(
            policy=cast(ScoreRiskCostRecipe, policy),
            inputs=inputs,
            optimizer=optimizer,
            sector_deviation_penalty=0.0,
        )


class RankBufferedScoreRiskCostAdapter(TopKScoreRiskCostAdapter):
    """Declare prior-target exit-buffer selection and complete turnover-cap allocation."""

    policy_id = "RANK_BUFFERED_SCORE_RISK_COST"
    recipe_schema_id = "RANK_BUFFERED_SCORE_RISK_COST"
    solver_semantics = "CONVEX_SCORE_RISK_COST_WITH_COMPLETE_TURNOVER_CAP"
    deterministic_policy: ClassVar[dict[str, str]] = {"recipe": "RANK_BUFFERED_SCORE_RISK_COST"}
    selection_allocation_semantics: ClassVar[PortfolioSelectionAllocationSemantics] = (
        PortfolioSelectionAllocationSemantics.create(
            score_preselection_method=("STABLE_FINITE_SCORE_TOP_K_WITH_PREVIOUS_TARGET_EXIT_BUFFER")
        )
    )


class SectorDeviationPenaltyAdapter:
    """Allocate score/Risk/cost with the recipe's explicit soft sector deviation penalty."""

    policy_id = "SECTOR_DEVIATION_PENALTY"
    solver_backed = True

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind score/Risk/cost allocation and the declared solver environment.

        Returns:
            Exact adapter identity with recipe-selected soft sector penalty and explicit
            preselection semantics.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=_score_risk_cost_implementation_hash(),
            recipe_schema_id="SECTOR_DEVIATION_PENALTY",
            solver_semantics="CONVEX_SCORE_RISK_COST",
            deterministic_policy={
                "objective": "score-risk-cost",
                "sector_penalty": "FROM_RECIPE",
                # The numerical environment, inside the binding hash. Two runs are
                # the same experiment only if the solver, its version, its
                # tolerances and its iteration budget were the same, so a change
                # to any of them moves the Program hash and reuse refuses rather
                # than silently mixing two numerical environments in one graph.
                "numerical_environment": solver_numerical_environment(),
            },
            selection_allocation_semantics=PortfolioSelectionAllocationSemantics.create(
                score_preselection_method="STABLE_FINITE_SCORE_TOP_K"
            ),
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        """Solve admitted score/Risk/cost allocation with recipe-selected soft sector penalty.

        Args:
            policy: Concrete admitted installed recipe.
            inputs: Exact scores, eligibility, holdings reference and declared auxiliary lanes.
            optimizer: Deterministic solver owner; closed-form adapters do not use it.

        Returns:
            Verified targets and retained Risk/solver/objective diagnostics.

        Raises:
            PortfolioOptimizationError: Input, utility, selection, feasibility or solution admission
                fails.
        """
        sector_policy = cast(SectorDeviationRecipe, policy)
        return _solve(
            policy=sector_policy,
            inputs=inputs,
            optimizer=optimizer,
            sector_deviation_penalty=sector_policy.sector_deviation_penalty,
        )


__all__ = [
    "RankBufferedScoreRiskCostAdapter",
    "RankBufferedScoreRiskCostDevelopmentRecipe",
    "ScoreRiskCostDevelopmentRecipe",
    "ScoreRiskCostRecipe",
    "ScoreRiskCostTrialRecipe",
    "SectorDeviationPenaltyAdapter",
    "SectorDeviationRecipe",
    "TopKScoreRiskCostAdapter",
]
