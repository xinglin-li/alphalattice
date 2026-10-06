"""Safe Data Operations projection of provider-adjusted return semantics."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    MarketDataRepository,
    ProviderAdjustedSeriesRevision,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class AdjustedReturnSemanticDelta(BaseModel):
    """Describe one listing's adjusted-return semantic change."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    listing_id: str
    changed_return_sessions: tuple[date, ...]
    session_set_changed: bool
    uniform_rescale: bool
    semantic_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class AdjustedReturnSemanticRevision(BaseModel):
    """Bind ordered adjusted-return deltas to one chain identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cursor: int = Field(ge=0)
    chain_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    deltas: tuple[AdjustedReturnSemanticDelta, ...]


def build_adjusted_return_semantic_revision(
    ledger: tuple[ProviderAdjustedSeriesRevision, ...],
) -> AdjustedReturnSemanticRevision:
    """Build a content-addressed chain without exposing mutable rows downstream."""
    deltas: list[AdjustedReturnSemanticDelta] = []
    for value in ledger:
        identity = {
            "listing_id": value.listing_id,
            "changed_return_sessions": value.changed_return_sessions,
            "session_set_changed": value.session_set_changed,
            "uniform_rescale": value.uniform_rescale,
        }
        deltas.append(
            AdjustedReturnSemanticDelta(
                **identity,
                semantic_hash=canonical_hash(identity),
            )
        )
    frozen = tuple(deltas)
    return AdjustedReturnSemanticRevision(
        cursor=len(frozen),
        chain_hash=canonical_hash([value.semantic_hash for value in frozen]),
        deltas=frozen,
    )


class AdjustedReturnSemanticRevisionPublisher:
    """The only adapter that turns the mutable ledger into a safe projection."""

    def __init__(self, *, store: MarketDataRepository, resolver: ArtifactResolver) -> None:
        """Bind the market-data ledger and its publication resolver.

        Args:
            store: Owner of provider-adjusted semantic revisions.
            resolver: Owner of the current immutable projection.

        """
        self.store = store
        self.resolver = resolver

    def refresh(self) -> AdjustedReturnSemanticRevision:
        """Publish the current semantic ledger as an immutable revision."""
        revision = build_adjusted_return_semantic_revision(
            self.store.provider_adjusted_semantic_ledger()
        )
        self.resolver.publish_adjusted_return_semantic_revision(
            payload=revision.model_dump(mode="json"),
            chain_hash=revision.chain_hash,
        )
        return revision

    def load(self) -> AdjustedReturnSemanticRevision:
        """Load and validate the current published semantic revision."""
        return AdjustedReturnSemanticRevision.model_validate(
            self.resolver.load_current_adjusted_return_semantic_revision()
        )

    def load_or_backfill(self) -> AdjustedReturnSemanticRevision:
        """Load the current revision or publish one if none exists."""
        try:
            return self.load()
        except FileNotFoundError:
            return self.refresh()


__all__ = [
    "AdjustedReturnSemanticDelta",
    "AdjustedReturnSemanticRevision",
    "AdjustedReturnSemanticRevisionPublisher",
    "build_adjusted_return_semantic_revision",
]
