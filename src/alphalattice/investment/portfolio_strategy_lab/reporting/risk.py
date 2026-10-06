"""Report-side consumption of the factor Risk projection.

Nothing here is a holdings input.  The policy consumes only
``RiskAllocationProjection``; this owner turns the separate attribution
projection and an executed target into immutable report facts.
"""

from __future__ import annotations

from datetime import date
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.risk_research.surfaces.decomposition import (
    RiskAttributionProjection,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class PortfolioRiskReportError(ValueError):
    """Stable refusal for report projection lineage or arithmetic."""


class PortfolioRiskAttributionFacts(BaseModel):  # type: ignore[misc]
    """One formation's declared-path ``D F D' + E`` report facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = "PortfolioRiskAttributionFacts"
    formation_session: date
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    risk_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_factor_ids: tuple[str, ...]
    industry_exposure: tuple[float, ...]
    factor_variance_contribution: tuple[float, ...]
    systematic_variance: float = Field(ge=0.0)
    idiosyncratic_variance: float = Field(ge=0.0)
    total_predicted_volatility: float = Field(ge=0.0)
    classification_authority: str = Field(min_length=1)
    facts_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        projection: RiskAttributionProjection,
        weights: np.ndarray,
    ) -> Self:
        """Seal owner-derived Risk attribution facts for exact portfolio weights.

        Seal owner-derived Risk exposure, variance split and contribution facts for exact weights.

        Args:
            projection: Admitted Risk attribution projection on its declared axes.
            weights: Finite portfolio weights on the exact listing axis.

        Returns:
            Validated canonical facts retaining source/projection identity and factor/classification
            authority.

        Raises:
            PortfolioRiskReportError: Weight axis/finiteness or summed systematic contribution
                verification fails.
        """
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != (len(projection.ordered_listing_ids),) or not np.isfinite(values).all():
            raise PortfolioRiskReportError("portfolio_strategy_lab.risk_report_weight_axis_invalid")
        systematic, specific = projection.variance_split(values)
        exposure = projection.industry_exposure(values)
        contribution = projection.factor_variance_contribution(values)
        if not np.isclose(float(contribution.sum()), systematic, rtol=1e-9, atol=1e-15):
            raise PortfolioRiskReportError(
                "portfolio_strategy_lab.risk_report_factor_contribution_invalid"
            )
        identity: dict[str, object] = {
            "kind": "PortfolioRiskAttributionFacts",
            "formation_session": projection.formation_session.isoformat(),
            "surface_hash": projection.surface_hash,
            "risk_projection_hash": projection.projection_hash,
            "ordered_factor_ids": list(projection.ordered_factor_ids),
            "industry_exposure": [float(value) for value in exposure],
            "factor_variance_contribution": [float(value) for value in contribution],
            "systematic_variance": systematic,
            "idiosyncratic_variance": specific,
            "total_predicted_volatility": projection.total_predicted_volatility(values),
            "classification_authority": projection.classification_authority,
        }
        return cls(**identity, facts_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require complete finite attribution axes/values and exact facts identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioRiskReportError: Factor IDs are empty/repeated, exposure/contribution lengths
                or finite values differ, or facts_hash differs.
        """
        if (
            not self.ordered_factor_ids
            or len(set(self.ordered_factor_ids)) != len(self.ordered_factor_ids)
            or len(self.industry_exposure) != len(self.ordered_factor_ids)
            or len(self.factor_variance_contribution) != len(self.ordered_factor_ids)
            or not all(
                np.isfinite(value)
                for value in (
                    *self.industry_exposure,
                    *self.factor_variance_contribution,
                    self.systematic_variance,
                    self.idiosyncratic_variance,
                    self.total_predicted_volatility,
                )
            )
        ):
            raise PortfolioRiskReportError("portfolio_strategy_lab.risk_report_values_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"facts_hash"}))
        if self.facts_hash != expected:
            raise PortfolioRiskReportError("portfolio_strategy_lab.risk_report_identity_invalid")
        return self


__all__ = ["PortfolioRiskAttributionFacts", "PortfolioRiskReportError"]
