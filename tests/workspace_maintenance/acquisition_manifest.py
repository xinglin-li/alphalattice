"""The two-name acquisition manifest the workspace-maintenance suites bind.

One fixture universe (AAPL, MSFT on XNAS) under the research-only current
universe profile, as of the suites' shared clock. Test support beside its
owner (the maintenance coordinator proof); nothing here is product authority.
"""

from __future__ import annotations

from datetime import UTC, datetime

from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)

NOW = datetime(2026, 8, 3, 22, tzinfo=UTC)


def acquisition_manifest() -> UniverseManifest:
    profile = MarketProfile(
        market_profile_id="us-current-index-research",
        display_name="US current index research",
        market="US",
        currency="USD",
        calendar_id="XNAS+XNYS",
        provider="fixture",
        daily_price_basis="split_adjusted",
        manifest_as_of=NOW.date(),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    listings = tuple(
        ManifestListing(
            listing_id=f"listing-{symbol.casefold()}",
            symbol=symbol,
            mic="XNAS",
            provider_symbol=symbol,
        )
        for symbol in ("AAPL", "MSFT")
    )
    return UniverseManifest(
        manifest_id="us-current-index-research:acquisition:fixture",
        profile=profile,
        listings=listings,
        revision_sha256="a" * 64,
        membership_fingerprint="b" * 64,
        qualification_policy_hash="c" * 64,
    )
