"""The research desk's candidate inventory and its cards.

The Alpha model, Risk candidate and Portfolio scenario cards, in the order the inventory seals.
The PM's effort resolution and the inventories' loaders left with V327; the loaders are the
tests' fixtures.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash


class _ContractModel(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class DesktopAlphaModelCard(_ContractModel):
    """Constrain the factor axis and seed policy of an Alpha candidate."""

    candidate_id: str = Field(min_length=1, max_length=120)
    family_id: str = Field(min_length=1, max_length=120)
    feature_shape: Literal["NONE", "DYNAMIC_ORDERED"]
    minimum_factor_count: int = Field(ge=0, le=55)
    maximum_factor_count: int = Field(ge=0, le=55)
    secondary_feature_preprocessing: Literal[False] = False
    deterministic_seed_required: bool
    card_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_card(self) -> DesktopAlphaModelCard:
        """Verify the candidate's factor interval and card hash."""
        if self.minimum_factor_count > self.maximum_factor_count:
            raise ValueError("Alpha model factor interval is inverted")
        if self.feature_shape == "NONE" and (
            self.minimum_factor_count != 0 or self.maximum_factor_count != 0
        ):
            raise ValueError("featureless Alpha model card accepts no factors")
        if self.feature_shape == "DYNAMIC_ORDERED" and self.minimum_factor_count < 1:
            raise ValueError("dynamic Alpha model card requires at least one factor")
        if self.card_hash != canonical_hash(self.model_dump(mode="json", exclude={"card_hash"})):
            raise ValueError("Alpha model card hash is invalid")
        return self


class DesktopRiskCandidateCard(_ContractModel):
    """Name a governed Risk candidate and its symbol-order contract."""

    candidate_id: Literal["sample", "ledoit_wolf", "vol_scaled_ledoit_wolf"]
    exact_symbol_order_required: Literal[True] = True
    deterministic_seed_required: bool = False
    card_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_card(self) -> DesktopRiskCandidateCard:
        """Verify the Risk candidate card hash."""
        if self.card_hash != canonical_hash(self.model_dump(mode="json", exclude={"card_hash"})):
            raise ValueError("Risk candidate card hash is invalid")
        return self


class DesktopPortfolioScenarioCard(_ContractModel):
    """Name a governed Portfolio scenario and immutable constraints."""

    scenario_id: Literal[
        "risk_only_gmv",
        "forecast_mvo_conservative",
        "forecast_mvo_balanced",
        "forecast_mvo_alpha_tilted",
        "forecast_mvo_low_turnover",
        "black_litterman_mvo",
    ]
    host_qualified_weights_only: Literal[True] = True
    hard_constraints_immutable: Literal[True] = True
    card_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_card(self) -> DesktopPortfolioScenarioCard:
        """Verify the Portfolio scenario card hash."""
        if self.card_hash != canonical_hash(self.model_dump(mode="json", exclude={"card_hash"})):
            raise ValueError("Portfolio scenario card hash is invalid")
        return self


class ResearchDeskCandidateInventory(_ContractModel):
    """Bind the admitted Alpha, Risk, and Portfolio candidate axes."""

    alpha_models: tuple[DesktopAlphaModelCard, ...]
    risk_candidates: tuple[DesktopRiskCandidateCard, ...]
    portfolio_scenarios: tuple[DesktopPortfolioScenarioCard, ...]
    inventory_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_inventory(self) -> ResearchDeskCandidateInventory:
        """Verify candidate uniqueness, scenario breadth, and hash."""
        for label, values in (
            ("Alpha", tuple(item.candidate_id for item in self.alpha_models)),
            ("Risk", tuple(item.candidate_id for item in self.risk_candidates)),
            ("Portfolio", tuple(item.scenario_id for item in self.portfolio_scenarios)),
        ):
            if values != tuple(dict.fromkeys(values)):
                raise ValueError(f"{label} candidate inventory must be unique")
        if len(self.portfolio_scenarios) != 6:
            raise ValueError("HIGH inventory requires the six governed portfolio scenarios")
        if self.inventory_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"inventory_hash"})
        ):
            raise ValueError("Research Desk inventory hash is invalid")
        return self

    @property
    def dynamic_model_factor_floor(self) -> int:
        """Return the smallest required factor count among dynamic Alpha models."""
        values = tuple(
            item.minimum_factor_count
            for item in self.alpha_models
            if item.feature_shape == "DYNAMIC_ORDERED"
        )
        if not values:
            raise ValueError("Research Desk inventory lacks a dynamic Alpha model")
        return min(values)

    @property
    def dynamic_model_factor_ceiling(self) -> int:
        """Return the largest admitted factor count among dynamic Alpha models."""
        values = tuple(
            item.maximum_factor_count
            for item in self.alpha_models
            if item.feature_shape == "DYNAMIC_ORDERED"
        )
        if not values:
            raise ValueError("Research Desk inventory lacks a dynamic Alpha model")
        return max(values)


__all__ = [
    "DesktopAlphaModelCard",
    "DesktopPortfolioScenarioCard",
    "DesktopRiskCandidateCard",
    "ResearchDeskCandidateInventory",
]
