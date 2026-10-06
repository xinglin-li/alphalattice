"""Current semantic contracts for regularized Portfolio Strategy research."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.investment.portfolio_strategy_lab.contracts import PortfolioLabContract
from alphalattice.investment.portfolio_strategy_lab.regularization.contracts import (
    PortfolioAgentValueClassification,
    PortfolioResearchTerminalRoute,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"


def _identity(value: PortfolioLabContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("portfolio_strategy_lab.strategy_research_identity_invalid")


class PortfolioStrategyResearchPublicationBundle(PortfolioLabContract):
    """One current Strategy result; heavy research evidence remains content-addressed."""

    kind: Literal["PortfolioStrategyResearchPublicationBundle"] = (
        "PortfolioStrategyResearchPublicationBundle"
    )
    mandate_hash: str = Field(pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    review_hash: str = Field(pattern=_HASH)
    agent_value_report_hash: str = Field(pattern=_HASH)
    validation_slate_hash: str | None = Field(default=None, pattern=_HASH)
    status: Literal[
        "PORTFOLIO_POLICY_VALIDATION_SLATE_FROZEN",
        "NO_PORTFOLIO_POLICY_READY_FOR_VALIDATION",
    ]
    validation_candidate_count: int = Field(ge=0, le=3)
    fold_count: Literal[3] = 3
    search_attempt_count: Literal[144] = 144
    experiment_attempt_count: int = Field(ge=0, le=48)
    model_call_count: int = Field(ge=0, le=12)
    terminal_route: PortfolioResearchTerminalRoute
    agent_value_classification: PortfolioAgentValueClassification
    research_evidence_state: Literal["OOS_RECLASSIFIED_FOR_STRATEGY_RESEARCH"] = (
        "OOS_RECLASSIFIED_FOR_STRATEGY_RESEARCH"
    )
    policy_holdout_state: Literal["SEALED"] = "SEALED"
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    limitations: tuple[str, ...] = Field(min_length=1)
    bundle_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_bundle(self) -> Self:
        """Require frozen-slate readiness to match candidate count, acceptance route and slate.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Readiness differs from candidate count, terminal route, slate presence or
                bundle_hash differs.
        """
        ready = self.status == "PORTFOLIO_POLICY_VALIDATION_SLATE_FROZEN"
        if ready != (self.validation_candidate_count > 0):
            raise ValueError("portfolio_strategy_lab.strategy_research_status_invalid")
        if ready != (
            self.terminal_route is PortfolioResearchTerminalRoute.ACCEPT_POLICY_CANDIDATES
        ):
            raise ValueError("portfolio_strategy_lab.strategy_research_status_invalid")
        if ready != (self.validation_slate_hash is not None):
            raise ValueError("portfolio_strategy_lab.strategy_research_status_invalid")
        _identity(self, "bundle_hash")
        return self

    def model_view(self) -> dict[str, object]:
        """Return model-safe research semantics without hashes or storage references."""
        return {
            "kind": self.kind,
            "status": self.status,
            "validation_candidate_count": self.validation_candidate_count,
            "fold_count": self.fold_count,
            "search_attempt_count": self.search_attempt_count,
            "experiment_attempt_count": self.experiment_attempt_count,
            "model_call_count": self.model_call_count,
            "terminal_route": self.terminal_route,
            "agent_value_classification": self.agent_value_classification,
            "research_evidence_state": self.research_evidence_state,
            "policy_holdout_state": self.policy_holdout_state,
            "system_holdout_state": self.system_holdout_state,
            "limitations": self.limitations,
        }


class CurrentPortfolioStrategyResearchMarker(PortfolioLabContract):
    """Bind current strategy-research status and publication clock to exact bundle lineage."""

    kind: Literal["CurrentPortfolioStrategyResearchMarker"] = (
        "CurrentPortfolioStrategyResearchMarker"
    )
    mandate_hash: str = Field(pattern=_HASH)
    dossier_hash: str = Field(pattern=_HASH)
    bundle_hash: str = Field(pattern=_HASH)
    status: Literal[
        "PORTFOLIO_POLICY_VALIDATION_SLATE_FROZEN",
        "NO_PORTFOLIO_POLICY_READY_FOR_VALIDATION",
    ]
    published_at: datetime
    marker_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> Self:
        """Require an aware strategy publication clock and exact marker identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Clock is naive or marker_hash differs.
        """
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("portfolio_strategy_lab.strategy_research_clock_invalid")
        _identity(self, "marker_hash")
        return self


class CurrentPortfolioStrategyResearchPointer(PortfolioLabContract):
    """Bind the exact current strategy-research marker through canonical pointer identity."""

    kind: Literal["CurrentPortfolioStrategyResearchPointer"] = (
        "CurrentPortfolioStrategyResearchPointer"
    )
    marker_hash: str = Field(pattern=_HASH)
    pointer_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_pointer(self) -> Self:
        """Require exact current strategy-research pointer identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: pointer_hash differs.
        """
        _identity(self, "pointer_hash")
        return self


__all__ = [
    "CurrentPortfolioStrategyResearchMarker",
    "CurrentPortfolioStrategyResearchPointer",
    "PortfolioStrategyResearchPublicationBundle",
]
