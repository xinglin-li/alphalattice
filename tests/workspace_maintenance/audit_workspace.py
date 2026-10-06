"""Small product-written raw evidence for receipt mechanics, without a Feature build.

These cases compare a complete range with 20- and 45-session rolling ranges,
correct bytes, and exercise receipt freshness. They do not judge universe or
Feature qualification. Two listings keep the missing-child-evidence case.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    OBSERVED_AT,
    SeededWalkProvider,
)
from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest


def build_audit_workspace(root: Path) -> tuple[Path, UniverseManifest]:
    start = AS_OF - timedelta(days=180)
    base = acquisition_manifest()
    manifest = replace(
        base,
        manifest_id="receipt-mechanics",
        revision_sha256="3" * 64,
        profile=replace(base.profile, provider="yfinance", manifest_as_of=AS_OF),
    )
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=start, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(row["session_date"] for row in schedule.to_pylist())
    # The 45-session scope must leave an older part of the full range.
    if len(sessions) < 90:
        raise AssertionError("receipt fixture requires an older range plus its rolling tail")
    symbols = tuple(listing.symbol for listing in manifest.listings)
    provider = SeededWalkProvider(symbols, sessions)
    market = MarketDataRepository(root)
    market.bootstrap(manifest)
    market.apply_validated_batch(
        manifest,
        sanitize_payload(
            manifest, provider.name, provider.fetch_daily(symbols, start=start, end=AS_OF), symbols
        ),
        ingestion_id="receipt-mechanics",
        observed_at=OBSERVED_AT,
    )
    for listing in manifest.listings:
        market.complete_action_audit(
            manifest,
            listing_id=listing.listing_id,
            provider=provider.name,
            observed_actions=(),
            observed_adjusted_closes=provider.fetch_adjusted_close_history(
                listing_id=listing.listing_id,
                provider_symbol=listing.provider_symbol,
                start=start,
                end=AS_OF,
            ),
            history_start=sessions[0],
            history_end=AS_OF,
            requested_as_of=AS_OF,
            observed_at=OBSERVED_AT,
        )
    return root, manifest
