"""Shared Feature-input fixtures: a manifest, its temporal boundary, evidence and sectors."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from alphalattice.foundation.feature_engine.contracts import (
    TemporalKnowledgeBoundary,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    ListingQualityEvidence,
    PanelImpactProjection,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)

NOW = datetime(2026, 8, 3, 15, tzinfo=UTC)
AS_OF = date(2026, 7, 31)


def _manifest(count: int, *, revision: str = "a" * 64) -> UniverseManifest:
    profile = MarketProfile(
        market_profile_id=f"feature-input-{count}",
        display_name="Feature Input fixture",
        market="US",
        currency="USD",
        calendar_id="XNYS",
        provider="fixture",
        daily_price_basis="unadjusted",
        manifest_as_of=AS_OF,
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    listings = tuple(
        ManifestListing(
            listing_id=f"listing-{index:03d}",
            symbol=f"S{index:03d}",
            mic="XNYS",
            provider_symbol=f"S{index:03d}",
        )
        for index in range(count)
    )
    return UniverseManifest(
        manifest_id=f"feature-input-{count}:{revision[:8]}",
        profile=profile,
        listings=listings,
        revision_sha256=revision,
    )


def _temporal(*, cutoff: datetime = NOW) -> TemporalKnowledgeBoundary:
    return TemporalKnowledgeBoundary(
        market_as_of_session=AS_OF,
        knowledge_cutoff_at=cutoff,
        materialized_at=cutoff + timedelta(minutes=1),
        universe_source_observed_at=cutoff - timedelta(days=1),
        sector_source_observed_at=cutoff - timedelta(hours=1),
    )


def _evidence(
    manifest: UniverseManifest,
    *,
    failed: set[int] | None = None,
    failure_code: str = "data.provider_timeout",
    extreme: bool = False,
) -> tuple[ListingQualityEvidence, ...]:
    failed = failed or set()
    return tuple(
        ListingQualityEvidence(
            listing_id=listing.listing_id,
            provider="fixture",
            range_start=date(2016, 8, 1),
            range_end=AS_OF,
            evidence_hash=canonical_hash([listing.listing_id, index in failed, failure_code]),
            failure_code=failure_code if index in failed else None,
            reason_codes=(("UNEXPLAINED_RAW_MOVE",) if extreme else ("PROVIDER_TIMEOUT",))
            if index in failed
            else (),
            retry_exhausted=index in failed,
            extreme_move_unexplained=extreme and index in failed,
        )
        for index, listing in enumerate(manifest.listings)
    )


def _sectors(manifest: UniverseManifest, *, size: int = 10) -> dict[str, str]:
    return {
        listing.listing_id: f"sector-{index // size:02d}"
        for index, listing in enumerate(manifest.listings)
    }


def _panel_impact(manifest: UniverseManifest, sectors: dict[str, str]) -> PanelImpactProjection:
    distribution: dict[str, int] = {}
    for sector in sectors.values():
        distribution[sector] = distribution.get(sector, 0) + 1
    return PanelImpactProjection(distribution, 1.0, 60, True)
