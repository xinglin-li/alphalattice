"""Buffered Equal Weight: whole-book hysteresis selection and equal-weight allocation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    FloatArray,
    PortfolioTargetDecision,
    PortfolioWalkForwardError,
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

WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT_POLICY_ID = "WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT"
_TOLERANCE = 1e-8


class WholeBookHysteresisEqualWeightRecipe(BaseModel):  # type: ignore[misc]
    """The one fixed recipe supplied by the immutable baseline package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["WholeBookHysteresisEqualWeightRecipe"] = "WholeBookHysteresisEqualWeightRecipe"
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
        return WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT_POLICY_ID

    @classmethod
    def create(cls) -> Self:
        """Seal the fixed 50-name/150-exit whole-book equal-weight baseline.

        Returns:
            Validated installed recipe with canonical recipe_hash.
        """
        values: dict[str, object] = {
            "kind": "WholeBookHysteresisEqualWeightRecipe",
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
                "portfolio_strategy_lab.whole_book_equal_weight_recipe_identity_invalid"
            )
        return self


def whole_book_hysteresis_selection(
    *,
    scores: FloatArray,
    decision_eligible: np.ndarray,
    ordered_listing_ids: tuple[str, ...],
    previous_target_weights: FloatArray | None,
    top_k: int,
    exit_rank: int,
) -> npt.NDArray[np.int64]:
    """Select exactly ``top_k`` names with stable score/listing-id hysteresis."""
    if (
        scores.ndim != 1
        or decision_eligible.shape != scores.shape
        or len(ordered_listing_ids) != scores.size
        or len(set(ordered_listing_ids)) != len(ordered_listing_ids)
    ):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_equal_weight_decision_axis_invalid"
        )
    eligible = np.flatnonzero(np.asarray(decision_eligible, dtype=np.bool_) & np.isfinite(scores))
    if eligible.size < top_k:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_equal_weight_eligible_pool_too_small"
        )
    ids = np.asarray(ordered_listing_ids, dtype=str)
    order = eligible[np.lexsort((ids[eligible], -scores[eligible]))]
    if previous_target_weights is None:
        return cast(npt.NDArray[np.int64], np.asarray(order[:top_k], dtype=np.int64))
    if previous_target_weights.shape != scores.shape:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_equal_weight_previous_target_axis_invalid"
        )
    ranks = np.full(scores.size, scores.size + 1, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    # An unavailable name is carried by ``whole_book_equal_weight_target`` as frozen
    # reference state.  It cannot also occupy one of the fresh, eligible book
    # slots; doing so would either make the selection fail or silently reduce
    # the intended 50-name book.
    held = np.flatnonzero(
        (previous_target_weights > _TOLERANCE) & np.asarray(decision_eligible, dtype=np.bool_)
    )
    survivors = held[ranks[held] < exit_rank]
    survivors = survivors[np.argsort(ranks[survivors], kind="stable")][:top_k]
    retained = np.zeros(scores.size, dtype=np.bool_)
    retained[survivors] = True
    fill = order[~retained[order]][: top_k - survivors.size]
    selected = np.concatenate((survivors, fill))
    if selected.size != top_k or len(set(int(value) for value in selected)) != top_k:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_equal_weight_selection_invalid"
        )
    return cast(npt.NDArray[np.int64], np.asarray(selected, dtype=np.int64))


def whole_book_equal_weight_target(
    *,
    selected: npt.NDArray[np.int64],
    decision_eligible: np.ndarray,
    reference_weights: FloatArray,
    top_k: int,
) -> FloatArray:
    """Equal-weight the selected book while preserving existing frozen carry."""
    if reference_weights.ndim != 1 or decision_eligible.shape != reference_weights.shape:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_equal_weight_decision_axis_invalid"
        )
    if selected.size != top_k or np.any(~np.asarray(decision_eligible, dtype=np.bool_)[selected]):
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_equal_weight_selection_invalid"
        )
    frozen = (~np.asarray(decision_eligible, dtype=np.bool_)) & (reference_weights > _TOLERANCE)
    target = np.zeros_like(reference_weights)
    target[frozen] = reference_weights[frozen]
    available = 1.0 - float(target.sum())
    if available < -_TOLERANCE:
        raise PortfolioWalkForwardError(
            "portfolio_strategy_lab.whole_book_equal_weight_frozen_carry_infeasible"
        )
    target[selected] = available / top_k
    target.setflags(write=False)
    return target


class WholeBookHysteresisEqualWeightAdapter:
    """Direct decision owner: no optimizer, solver, cap, or covariance read."""

    policy_id = WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT_POLICY_ID
    solver_backed = False
    selection_allocation_semantics = (
        PortfolioSelectionAllocationSemantics.closed_form_hysteresis_equal_weight()
    )

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """Bind this adapter's implementation, declared input consumption and allocation semantics.

        Returns:
            Exact adapter binding for previous-target hysteresis and equal weights with frozen
            carry; it reads no covariance and requests no Risk forecast.
        """
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (PORTFOLIO_POLICY_PACKAGE, Path(__file__)),
            ),
            recipe_schema_id="WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT",
            solver_semantics="CLOSED_FORM_WHOLE_BOOK_EQUAL_WEIGHT_NO_COVARIANCE",
            deterministic_policy={
                "selection": "stable-score-listing-id-top-k-with-previous-target-exit-rank",
                "allocation": "equal-selected-book-plus-frozen-untradable-carry",
                "book_semantics": "whole-book-not-tranche-or-phase-composite",
                "covariance_input": "NOT_READ",
                "cap_machinery": "NOT_USED",
            },
            selection_allocation_semantics=self.selection_allocation_semantics,
            input_consumption_semantics="NO_RISK_FORECAST_OR_OPTIMIZER",
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: object,
    ) -> PortfolioTargetDecision:
        """Allocate the hysteresis-selected equal-weight book with frozen carry.

        Apply previous-target hysteresis and equal weights with frozen carry; it reads no covariance
        and requests no Risk forecast.

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
        recipe = WholeBookHysteresisEqualWeightRecipe.model_validate(policy)
        if inputs.ordered_listing_ids is None:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.whole_book_equal_weight_listing_axis_absent"
            )
        selected = whole_book_hysteresis_selection(
            scores=inputs.scores,
            decision_eligible=inputs.decision_eligible,
            ordered_listing_ids=inputs.ordered_listing_ids,
            previous_target_weights=inputs.previous_target_weights,
            top_k=recipe.top_k,
            exit_rank=recipe.exit_rank,
        )
        return PortfolioTargetDecision(
            target_weights=whole_book_equal_weight_target(
                selected=selected,
                decision_eligible=inputs.decision_eligible,
                reference_weights=inputs.reference_weights,
                top_k=recipe.top_k,
            ),
            predicted_variance=None,
            requires_risk_forecast=False,
        )


__all__ = [
    "WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT_POLICY_ID",
    "WholeBookHysteresisEqualWeightAdapter",
    "WholeBookHysteresisEqualWeightRecipe",
    "whole_book_equal_weight_target",
    "whole_book_hysteresis_selection",
]
