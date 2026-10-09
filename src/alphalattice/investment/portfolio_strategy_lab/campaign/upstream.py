"""Reopen the Risk input binding a published covariance surface names, through its owner.

What is left of the Stage-6 campaign's upstream authority (the campaign retired with R01,
2026-09-29). A load goes through the owning Desk's reader and its self-validating contract,
never through a filename Check: a file sitting where its name says is not a contract satisfied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphalattice.investment.risk_research.experiments.window import (
        RiskDevelopmentInputBinding,
    )
    from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore


class PortfolioUpstreamAuthorityError(ValueError):
    """Stable fail-closed boundary for every upstream-authority failure."""


def load_exact_risk_input_binding(
    *, store: RiskArtifactStore, input_binding_hash: str
) -> RiskDevelopmentInputBinding:
    """Reopen one self-validating Risk input binding by its named identity."""
    from alphalattice.investment.risk_research.experiments.window import (
        RISK_INPUT_BINDING_CATEGORY,
        RiskDevelopmentInputBinding,
    )

    return RiskDevelopmentInputBinding(
        **store.load_json(
            category=RISK_INPUT_BINDING_CATEGORY,
            uri=store.uri(RISK_INPUT_BINDING_CATEGORY, input_binding_hash),
            identity_field="input_binding_hash",
        )
    )


__all__ = [
    "PortfolioUpstreamAuthorityError",
    "load_exact_risk_input_binding",
]
