"""Append one synthetic session to a QA copy without restating its held history."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime, time
from hashlib import sha256

from alphalattice.control.data_platform.preflight import resolve_trading_session_authority
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
)
from alphalattice.foundation.feature_engine.producers.reference_data import (
    MarketReference,
    SectorRefreshStager,
)
from alphalattice.foundation.feature_engine.publication.sector_map_activation import (
    SectorRevisionMapActivationCoordinator,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    CurrentUniverseMaintenanceStatus,
)
from alphalattice.foundation.market_data_ops.sources.contracts import ProviderAdjustedClosePoint
from alphalattice.foundation.market_data_ops.sources.providers import (
    HydrationEvidence,
    ProviderFetchError,
    SectorObservation,
)


def prepare_friday_parent(provider, market, manifest, target, now, gate):
    """Prepare every held source member, including ones outside the current child.

    Routine maintenance uses the product's bounded correction window. The copied
    QA history remains unchanged. No full-history permission, quarantine, retry
    time or qualified membership record is invented or edited.
    """
    history = market.manifest_raw_range(manifest)
    runner = CurrentUniverseMaintenance(
        store=market,
        manifest=manifest,
        provider=provider,
        as_of_session=target,
        trading_session_authority=resolve_trading_session_authority(
            start=history[0], end=target, as_of_timestamp=now
        ),
        mutation_gate=gate,
        max_workers=2,
    )
    while True:
        result = runner.run(observed_at=now)
        if result.status is not CurrentUniverseMaintenanceStatus.RUNNING:
            break
    assert result.status is CurrentUniverseMaintenanceStatus.COMPLETED and not result.failed, {
        "status": result.status.value,
        "updated": result.updated,
        "failed": result.failed,
        "failure_codes": dict(
            Counter(
                row.failure_code
                for row in market.current_universe_maintenance_listings(runner.maintenance_id)
                if row.state == "FAILED"
            )
        ),
    }
    assert market.manifest_raw_through(manifest) == target
    assert market.manifest_provider_adjusted_through(manifest) == target


def prepare_friday_sector(provider, market, manifest, now, gate):
    """Publish the complete parent reference through the real Sector owner."""
    store = FeatureStateRepository(market.database, market_data=market)
    coordinator = SectorRevisionMapActivationCoordinator(
        store=store,
        mutation_gate=gate,
        ledger=FeatureClosureLedger(
            PanelClosureArtifactStore(ArtifactResolver(market.workspace / "artifacts"))
        ),
    )
    stager = SectorRefreshStager(
        manifest=manifest,
        provider=provider,
        artifact_root=market.workspace / "staging" / "sector-reference",
        staging_id="fixture-friday-" + manifest.revision_sha256,
        activation_coordinator=coordinator,
    )
    assert stager.acquire(observed_at=now).status == "completed"
    assert stager.commit(store=store, mutation_gate=gate, observed_at=now).status == "completed"
    assert not store.sector_revision_refresh_due(manifest, observed_at=now)


def append_friday(provider, market, manifest, target):
    """Cache the copied store before writes; all overlapping bars/actions stay exact.

    The added day is synthetic. Prices, actions and adjusted closes in the QA
    prefix are served unchanged, so preparation proves no correction or new fit.
    """
    bars, listing_bars, actions, adjusted = {}, {}, {}, {}
    listings = (*manifest.listings, *MarketReference.spy(manifest).manifest.listings)
    features = FeatureStateRepository(market.database, market_data=market)
    sector_rows = features.reusable_current_sector_observations(
        manifest, observed_at=datetime.combine(target, time(23, 5), tzinfo=UTC)
    )
    assert sector_rows, "fixture needs its held Sector evidence"
    sectors = {
        str(row["provider_symbol"]): SectorObservation(
            **{
                name: row[name]
                for name in (
                    "provider",
                    "provider_symbol",
                    "sector_name",
                    "sector_key",
                    "payload_hash",
                )
            }
        )
        for row in sector_rows
    }
    # The new fixture supplies the one previously unclassified Data member too.
    # This is a synthetic current observation, not a claim about its real Sector.
    # Keep every held classification and the complete membership. Use a populated
    # held group so the product's unchanged Sector floor admits this scenario.
    example = next(iter(sectors.values()))
    for listing in manifest.listings:
        if listing.provider_symbol not in sectors:
            sectors[listing.provider_symbol] = SectorObservation(
                provider=provider.name,
                provider_symbol=listing.provider_symbol,
                sector_name=example.sector_name,
                sector_key=example.sector_key,
                payload_hash=sha256(("fixture-sector:" + listing.listing_id).encode()).hexdigest(),
            )

    def sector(*, provider_symbol):
        return sectors[provider_symbol]

    provider.fetch_current_sector = sector
    for index, listing in enumerate(listings):
        # Preserve the complete held name/key/payload and provider-symbol identity.
        held = market.raw_bars(listing.listing_id, through=target)
        assert held and held[-1].session_date < target
        values = [
            {
                key: value
                for key, value in asdict(bar).items()
                if key not in {"listing_id", "provider"}
            }
            for bar in held
        ]
        # An asymmetric, nonconstant new observation, with no action or restatement.
        ratio = 1.0005 + (index % 7) * 0.0001
        last = held[-1]
        values.append(
            {
                "session_date": target,
                "open": last.close * ratio * 0.999,
                "high": last.close * ratio * 1.002,
                "low": last.close * ratio * 0.998,
                "close": last.close * ratio,
                "volume": last.volume + index + 1,
            }
        )
        bars[listing.provider_symbol] = values
        listing_bars[listing.listing_id] = values
        actions[listing.listing_id] = market.actions(listing.listing_id)
        prior = market.provider_adjusted_closes(listing.listing_id, through=target)
        assert prior and prior[-1].session_date == last.session_date
        adjusted[listing.listing_id] = (
            *prior,
            ProviderAdjustedClosePoint(
                listing.listing_id, provider.name, target, prior[-1].adjusted_close * ratio
            ),
        )

    def daily(symbols, *, start, end):
        provider.calls.append((tuple(symbols), start, end))
        if set(symbols) - bars.keys():
            # The QA store has no admitted history for these failed candidates.
            # Their real retry records unavailability; invent no successful data.
            raise ProviderFetchError("data.empty_payload", "No fixture history", retryable=False)
        return {
            symbol: [
                {**row, "session_date": row["session_date"].isoformat()}
                for row in bars[symbol]
                if start <= row["session_date"] <= end
            ]
            for symbol in symbols
        }

    def action_history(*, listing_id, provider_symbol, start, end):
        return tuple(v for v in actions[listing_id] if start <= v.effective_date <= end)

    def adjusted_history(*, listing_id, provider_symbol, start, end):
        return tuple(v for v in adjusted[listing_id] if start <= v.session_date <= end)

    def hydration(*, listing_id, provider_symbol, start, end):
        provider.calls.append(((provider_symbol,), start, end))
        if listing_id not in listing_bars:
            raise ProviderFetchError("data.empty_payload", "No fixture history", retryable=False)
        return HydrationEvidence(
            daily_rows=tuple(
                {**row, "session_date": row["session_date"].isoformat()}
                for row in listing_bars[listing_id]
                if start <= row["session_date"] <= end
            ),
            actions=action_history(
                listing_id=listing_id, provider_symbol=provider_symbol, start=start, end=end
            ),
            adjusted_closes=adjusted_history(
                listing_id=listing_id, provider_symbol=provider_symbol, start=start, end=end
            ),
        )

    provider.fetch_daily = daily
    provider.fetch_hydration = hydration
    provider.fetch_action_history = action_history
    provider.fetch_adjusted_close_history = adjusted_history
    return provider
