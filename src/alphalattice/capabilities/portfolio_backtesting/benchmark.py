"""Causal same-clock benchmark contract for Portfolio path evaluation."""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"


class PortfolioBenchmarkBoundaryError(ValueError):
    """Stable fail-closed benchmark boundary."""


class PortfolioBenchmarkSurface(BaseModel):  # type: ignore[misc]
    """Seal a same-clock dividend-aware SPY benchmark return surface.

    The ordered unique formation sessions align simple and log returns. Each log
    value must match log1p(simple) bit for bit. The surface retains source and
    corporate-action commitments rather than inventing a benchmark from Target-Z.

    Attributes:
        reference_id: Installed SPY benchmark identifier.
        return_semantics: Dividend-aware open-to-open outcome semantics.
        formation_sessions: Nonempty sorted unique benchmark formation sessions.
        simple_returns: Finite simple returns strictly greater than minus one.
        log_returns: Corresponding finite log1p values with exact float equality.
        universe_manifest_revision: Source universe manifest identity.
        raw_input_hash: Raw benchmark input commitment.
        action_set_hash: Corporate-action set commitment.
        surface_hash: Canonical identity of all other surface fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioBenchmarkSurface"] = "PortfolioBenchmarkSurface"
    reference_id: Literal["SPY"] = "SPY"
    return_semantics: Literal["DIVIDEND_AWARE_OPEN_TO_OPEN"] = "DIVIDEND_AWARE_OPEN_TO_OPEN"
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    simple_returns: tuple[float, ...]
    log_returns: tuple[float, ...]
    universe_manifest_revision: str = Field(pattern=_HASH)
    raw_input_hash: str = Field(pattern=_HASH)
    action_set_hash: str = Field(pattern=_HASH)
    surface_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> Self:
        """Require aligned causal return pairs and the benchmark surface identity.

        Returns:
            This validated surface.

        Raises:
            ValueError: Session order, return values/pairs or the surface identity is invalid.
        """
        count = len(self.formation_sessions)
        if (
            self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(self.simple_returns) != count
            or len(self.log_returns) != count
            or not all(np.isfinite(self.simple_returns))
            or not all(np.isfinite(self.log_returns))
            or any(value <= -1.0 for value in self.simple_returns)
        ):
            raise ValueError("portfolio_benchmark.surface_invalid")
        for simple, logarithmic in zip(self.simple_returns, self.log_returns, strict=True):
            if float(np.log1p(simple)).hex() != float(logarithmic).hex():
                raise ValueError("portfolio_benchmark.return_pair_invalid")
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise ValueError("portfolio_benchmark.identity_invalid")
        return self


__all__ = [
    "PortfolioBenchmarkBoundaryError",
    "PortfolioBenchmarkSurface",
]
