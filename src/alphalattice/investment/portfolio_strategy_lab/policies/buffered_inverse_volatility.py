"""Buffered Inverse Volatility: whole-book hysteresis selection, inverse-volatility weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    FloatArray,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .buffered_equal_weight import whole_book_equal_weight_target, whole_book_hysteresis_selection
from .contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    PortfolioSelectionAllocationSemantics,
    portfolio_adapter_implementation_hash,
)

INVERSE_VOLATILITY_POLICY_ID = "WHOLE_BOOK_HYSTERESIS_INVERSE_VOLATILITY"


@dataclass(frozen=True, slots=True)
class InverseVolatilityAllocation:
    """One selected-book target and whether it produced a usable forecast."""

    target_weights: FloatArray
    produces_risk_forecast: bool


class WholeBookHysteresisInverseVolatilityRecipe(BaseModel):  # type: ignore[misc]
    """The one fixed recipe supplied by the immutable baseline package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["WholeBookHysteresisInverseVolatilityRecipe"] = (
        "WholeBookHysteresisInverseVolatilityRecipe"
    )
    top_k: Literal[50] = 50
    exit_rank: Literal[150] = 150
    cadence: Literal[3] = 3
    burn_in: Literal[126] = 126
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def policy_id(self) -> str:
        """Read this recipe's installed policy identity.

        Returns:
            The policy identity declared by this recipe owner.
        """
        return INVERSE_VOLATILITY_POLICY_ID

    @classmethod
    def create(cls) -> Self:
        """Seal the fixed 50-name/150-exit whole-book inverse-volatility recipe.

        Returns:
            Validated installed recipe with canonical recipe_hash.
        """
        values: dict[str, object] = {
            "kind": "WholeBookHysteresisInverseVolatilityRecipe",
            "top_k": 50,
            "exit_rank": 150,
            "cadence": 3,
            "burn_in": 126,
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the declared recipe parameters and canonical self identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The recipe hash differs.
        """
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ValueError(
                "portfolio_strategy_lab.whole_book_inverse_volatility_recipe_identity_invalid"
            )
        return self


def whole_book_inverse_volatility_target(
    *,
    selected: npt.NDArray[np.int64],
    covariance: FloatArray,
    decision_eligible: np.ndarray,
    reference_weights: FloatArray,
    top_k: int,
) -> InverseVolatilityAllocation:
    """Weight a whole selected book by inverse diagonal volatility.

    A malformed selected volatility cannot silently change the selected support:
    it degrades the entire selected book to equal-weight allocation, while
    retaining that owner's frozen/untradable carry semantics.
    """
    if covariance.ndim != 2 or covariance.shape != (reference_weights.size, reference_weights.size):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_inverse_volatility_covariance_axis_invalid"
        )
    fallback = whole_book_equal_weight_target(
        selected=selected,
        decision_eligible=decision_eligible,
        reference_weights=reference_weights,
        top_k=top_k,
    )
    diagonal = np.diag(covariance)[selected]
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        return InverseVolatilityAllocation(
            target_weights=fallback,
            produces_risk_forecast=False,
        )
    sigma = np.sqrt(diagonal)
    inverse = 1.0 / np.maximum(sigma, 1e-12)
    if np.any(~np.isfinite(inverse)) or float(inverse.sum()) <= 0.0:
        return InverseVolatilityAllocation(
            target_weights=fallback,
            produces_risk_forecast=False,
        )
    target = np.array(fallback, copy=True)
    target[selected] = float(fallback[selected].sum()) * inverse / float(inverse.sum())
    target.setflags(write=False)
    return InverseVolatilityAllocation(target_weights=target, produces_risk_forecast=True)


class WholeBookHysteresisInverseVolatilityAdapter:
    """Direct decision owner: Risk diagonal input, but no optimizer or solver."""

    policy_id = INVERSE_VOLATILITY_POLICY_ID
    solver_backed = False
    selection_allocation_semantics = (
        PortfolioSelectionAllocationSemantics.closed_form_hysteresis_inverse_volatility()
    )

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind this adapter's implementation, declared input consumption and allocation semantics.

        Returns:
            Exact adapter binding for previous-target hysteresis and admitted covariance-diagonal
            sizing; invalid selected volatility preserves support with equal-weight fallback.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
                (
                    PORTFOLIO_POLICY_PACKAGE,
                    Path(whole_book_hysteresis_selection.__code__.co_filename),
                ),
            ),
            recipe_schema_id="WHOLE_BOOK_HYSTERESIS_INVERSE_VOLATILITY",
            solver_semantics="CLOSED_FORM_INVERSE_VOLATILITY_NO_OPTIMIZER",
            deterministic_policy={
                "selection": "stable-score-listing-id-top-k-with-previous-target-exit-rank",
                "allocation": (
                    "inverse-covariance-diagonal-volatility-selected-book-plus-frozen-carry"
                ),
                "invalid_selected_sigma": "whole-book-equal-weight-fallback-no-support-change",
                "book_semantics": "whole-book-not-tranche-or-phase-composite",
                "covariance_input": "OWNER_VALIDATED_DIAGONAL_FOR_ALLOCATION",
                "cap_machinery": "NOT_USED",
            },
            selection_allocation_semantics=self.selection_allocation_semantics,
            input_consumption_semantics="RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: object,
    ) -> PortfolioTargetDecision:
        """Allocate the hysteresis-selected book using admitted inverse volatility.

        Apply previous-target hysteresis and admitted covariance-diagonal sizing; invalid selected
        volatility preserves support with equal-weight fallback.

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
        del optimizer
        recipe = WholeBookHysteresisInverseVolatilityRecipe.model_validate(policy)
        if inputs.ordered_listing_ids is None:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.whole_book_inverse_volatility_listing_axis_absent"
            )
        covariance = inputs.require_covariance()
        selected = whole_book_hysteresis_selection(
            scores=inputs.scores,
            decision_eligible=inputs.decision_eligible,
            ordered_listing_ids=inputs.ordered_listing_ids,
            previous_target_weights=inputs.previous_target_weights,
            top_k=recipe.top_k,
            exit_rank=recipe.exit_rank,
        )
        allocation = whole_book_inverse_volatility_target(
            selected=selected,
            covariance=covariance,
            decision_eligible=inputs.decision_eligible,
            reference_weights=inputs.reference_weights,
            top_k=recipe.top_k,
        )
        return PortfolioTargetDecision(
            target_weights=allocation.target_weights,
            predicted_variance=(
                float(allocation.target_weights @ covariance @ allocation.target_weights)
                if allocation.produces_risk_forecast
                else None
            ),
            requires_risk_forecast=allocation.produces_risk_forecast,
        )


__all__ = [
    "INVERSE_VOLATILITY_POLICY_ID",
    "InverseVolatilityAllocation",
    "WholeBookHysteresisInverseVolatilityAdapter",
    "WholeBookHysteresisInverseVolatilityRecipe",
    "whole_book_inverse_volatility_target",
]
