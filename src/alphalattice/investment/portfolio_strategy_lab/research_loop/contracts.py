"""Actor-neutral durable contracts for Portfolio scientific review."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.investment.portfolio_strategy_lab.contracts import PortfolioLabContract
from alphalattice.investment.portfolio_strategy_lab.regularization.contracts import (
    PortfolioResearchTerminalOutcome,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class PortfolioResearchScientistReview(PortfolioLabContract):
    """Durable result of one admitted scientific-review actor."""

    kind: Literal["PortfolioResearchScientistReview"] = "PortfolioResearchScientistReview"
    dossier_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    board_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal: PortfolioResearchTerminalOutcome
    value_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_at: datetime
    review_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_review(self) -> Self:
        """Require an aware scientist review clock and exact review identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Clock is naive or review_hash differs.
        """
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("portfolio_strategy_lab.research_review_clock_invalid")
        if self.review_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"review_hash"})
        ):
            raise ValueError("portfolio_strategy_lab.research_review_identity_invalid")
        return self


__all__ = ["PortfolioResearchScientistReview"]
