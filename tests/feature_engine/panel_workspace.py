"""The offline Feature workspace the Panel suites build and publish against.

A fixture provider over a deterministic 1,600-session walk, the fifteen-name
fixture manifest, the seeded Market Data workspace with its genesis closure,
and the Feature service plus snapshot publisher composed the way the product
composes them. Test support beside its owner (the Feature Foundation proof);
nothing here is product authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import (
    FeatureBuildOutcome,
    FeatureBuildRequest,
    FeatureBuildStatus,
    FeatureInvalidation,
)
from alphalattice.foundation.feature_engine.inputs.closure_source import (
    FeatureClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_coordinator import (
    FeatureBaseClosureCoordinator,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
)
from alphalattice.foundation.feature_engine.panels.logical_identity import (
    PanelLogicalArtifactStore,
    PanelLogicalIdentityPublisher,
)
from alphalattice.foundation.feature_engine.panels.recovery_binding import (
    PanelRecoveryBindingPublisher,
)
from alphalattice.foundation.feature_engine.producers.reference_data import (
    MarketReference,
)
from alphalattice.foundation.feature_engine.publication.sector_map_activation import (
    SectorRevisionMapActivationCoordinator,
)
from alphalattice.foundation.feature_engine.publication.snapshots import (
    FeaturePanelSnapshotPublisher,
)
from alphalattice.foundation.feature_engine.runtime.closure_genesis import (
    FeatureClosureGenesisService,
)
from alphalattice.foundation.feature_engine.runtime.service import (
    FeatureFoundationService,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    ProviderAdjustedClosePoint,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    build_trading_session_authority,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    ProviderFetchError,
    SectorObservation,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    MarketDataRepository,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

NOW = datetime(2026, 8, 3, 22, tzinfo=UTC)


SESSIONS = tuple(pd.bdate_range("2018-01-02", periods=1_600).date)


def fixture_session_authority(*, calendar_ids, start, end, as_of_timestamp):
    """Explicit synthetic weekday calendar, not a claim about exchange holidays."""
    assert as_of_timestamp.tzinfo is not None
    return build_trading_session_authority(
        calendar_ids=calendar_ids, sessions=tuple(day for day in SESSIONS if start <= day <= end)
    )


def open_closure(
    *,
    market_data: MarketDataRepository,
    panel_state: PanelStateRepository,
    feature_state: FeatureStateRepository,
    resolver: ArtifactResolver,
    manifest: UniverseManifest,
) -> tuple[FeatureClosureLedger, FeatureBaseClosureCoordinator]:
    """Open a real genesis closure and return the product persistence owner.

    A workspace cannot persist Feature rows before a closure exists, which is
    what the genesis service establishes. Composing the real coordinator here
    rather than a bypass is what keeps this case honest about the ordering the
    product actually enforces.
    """

    catalog = FeatureCatalog.load()
    ledger = FeatureClosureLedger(PanelClosureArtifactStore(resolver))
    source = FeatureClosureSourceRepository(market_data.database.path)
    genesis = FeatureClosureGenesisService(
        panel_state=panel_state, source=source, ledger=ledger
    ).open_genesis(manifest=manifest, catalog=catalog)
    if genesis.disposition != "GENESIS_READY":
        raise AssertionError(f"unexpected genesis disposition: {genesis.disposition}")
    return ledger, FeatureBaseClosureCoordinator(
        store=feature_state,
        source=source,
        ledger=ledger,
        factor_ids=catalog.factor_ids,
    )


class FixtureProvider:
    name = "fixture"

    def __init__(self, symbols: tuple[str, ...]) -> None:
        self._symbols = symbols
        self.sectors = {symbol: f"Sector-{index // 5}" for index, symbol in enumerate(symbols)}
        self.sectors["SPY"] = "Reference"

    def fetch_daily(self, symbols, *, start, end):
        result = {}
        for symbol in symbols:
            seed = sum(ord(item) for item in symbol)
            rng = np.random.default_rng(seed)
            selected = tuple(session for session in SESSIONS if start <= session <= end)
            # Construct from its full history, so refresh windows do not change
            # a given bar's price when returned independently.
            history = tuple(session for session in SESSIONS if session <= end)
            prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(history)))
            by_date = dict(zip(history, prices, strict=True))
            position_by_date = {session: index for index, session in enumerate(history)}
            result[symbol] = tuple(
                {
                    "session_date": session.isoformat(),
                    "open": float(
                        by_date[session]
                        * (0.997 + 0.001 * np.sin((position_by_date[session] + seed) / 7))
                    ),
                    "high": float(
                        by_date[session]
                        * (1.01 + 0.001 * np.sin((position_by_date[session] + seed) / 11))
                    ),
                    "low": float(
                        by_date[session]
                        * (0.99 - 0.001 * np.cos((position_by_date[session] + seed) / 13))
                    ),
                    "close": float(by_date[session]),
                    "volume": 1_000_000
                    + seed
                    + ((position_by_date[session] * (seed % 97 + 3)) % 100_000),
                    "split_ratio": 0.0,
                    "cash_dividend": 0.0,
                    "capital_gain": 0.0,
                }
                for session in selected
            )
        return result

    def fetch_action_history(self, *, listing_id, provider_symbol, start, end):
        del listing_id, provider_symbol, start, end
        return ()

    def fetch_adjusted_close_history(self, *, listing_id, provider_symbol, start, end):
        payload = self.fetch_daily((provider_symbol,), start=start, end=end)[provider_symbol]
        return tuple(
            ProviderAdjustedClosePoint(
                listing_id=listing_id,
                provider=self.name,
                session_date=date.fromisoformat(str(item["session_date"])),
                adjusted_close=float(item["close"]),
            )
            for item in payload
        )

    def fetch_current_sector(self, *, provider_symbol):
        if provider_symbol not in self.sectors:
            raise ProviderFetchError(
                "sector.missing_current_sector", "fixture missing sector", retryable=False
            )
        sector = self.sectors[provider_symbol]
        return SectorObservation(
            provider=self.name,
            provider_symbol=provider_symbol,
            sector_name=sector,
            sector_key=sector.casefold(),
            payload_hash=(sector.encode().hex() * 16)[:64],
        )


def fixture_manifest() -> UniverseManifest:
    profile = MarketProfile(
        "fixture-feature-market",
        "Fixture feature market",
        "US",
        "USD",
        "XNYS",
        "fixture",
        "unadjusted",
        SESSIONS[-1],
        "CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    listings = tuple(
        ManifestListing(f"listing-{index:02d}", f"T{index:02d}", "XNYS", f"T{index:02d}")
        for index in range(15)
    )
    return UniverseManifest("fixture-feature-manifest", profile, listings, "f" * 64)


def seed_feature_input_fixture(
    market_data: MarketDataRepository,
    feature_state: FeatureStateRepository,
    manifest: UniverseManifest,
    provider: FixtureProvider,
    *,
    end: date = SESSIONS[-1],
    spy_revision: str = "a" * 64,
) -> str:
    """Bulk-load verified fixture bars; raw-pipeline behavior has its own case.

    This keeps the feature case focused on vectorized materialization rather
    than spending its entire budget in the already-tested per-listing raw
    ingestion exercise.
    """

    market_data.bootstrap(manifest)
    reference = MarketReference.spy(manifest)
    market_data.bootstrap(reference.manifest)
    insert_fixture_bars(market_data, manifest, provider, start=SESSIONS[0], end=end)
    feature_state.upsert_market_reference(
        reference_id="SPY",
        listing_id=reference.listing_id,
        provider=provider.name,
        symbol="SPY",
        revision_hash=spy_revision,
        action_audit_receipt_hash=None,
        latest_session=end,
        observed_at=NOW,
    )
    return spy_revision


def insert_fixture_bars(
    market_data: MarketDataRepository,
    manifest: UniverseManifest,
    provider: FixtureProvider,
    *,
    start: date,
    end: date,
) -> None:
    reference = MarketReference.spy(manifest)
    payload = provider.fetch_daily(
        tuple(item.symbol for item in manifest.listings), start=start, end=end
    )
    asset_bars = sanitize_payload(manifest, provider.name, payload, tuple(payload)).bars
    spy_payload = provider.fetch_daily(("SPY",), start=start, end=end)
    spy_bars = sanitize_payload(reference.manifest, provider.name, spy_payload, ("SPY",)).bars
    records = [
        {
            "listing_id": bar.listing_id,
            "provider": bar.provider,
            "session_date": bar.session_date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            # Fixture lineage is deliberately deterministic; action behavior is
            # covered by the market-data-operations case.
            "payload_hash": f"fixture-{bar.listing_id}-{bar.session_date.isoformat()}",
            "observed_at": NOW.replace(tzinfo=None),
        }
        for bar in (*asset_bars, *spy_bars)
    ]
    adjusted_records = [
        {
            "listing_id": bar.listing_id,
            "provider": bar.provider,
            "session_date": bar.session_date,
            "adjusted_close": bar.close,
            "payload_hash": f"fixture-adjusted-{bar.listing_id}-{bar.session_date.isoformat()}",
            "updated_at": NOW.replace(tzinfo=None),
        }
        for bar in (*asset_bars, *spy_bars)
    ]
    connection = market_data._connect()
    try:
        connection.register("fixture_raw", pa.Table.from_pylist(records))
        connection.execute("INSERT INTO raw_daily_bar_current SELECT * FROM fixture_raw")
        connection.unregister("fixture_raw")
        connection.register("fixture_adjusted", pa.Table.from_pylist(adjusted_records))
        connection.execute(
            "INSERT INTO provider_adjusted_close_current SELECT * FROM fixture_adjusted"
        )
        connection.unregister("fixture_adjusted")
    finally:
        connection.close()


@dataclass
class PanelWorkspace:
    market_data: MarketDataRepository
    feature_state: FeatureStateRepository
    panel_state: PanelStateRepository
    resolver: ArtifactResolver
    gate: WorkspaceMutationGate
    ledger: FeatureClosureLedger
    service: FeatureFoundationService
    publisher: FeaturePanelSnapshotPublisher
    provider: FixtureProvider


def panel_workspace(
    root: Path, manifest: UniverseManifest, *, end: date, spy: str
) -> PanelWorkspace:
    provider = FixtureProvider(tuple(item.symbol for item in manifest.listings))
    market_data = MarketDataRepository(root / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    seed_feature_input_fixture(
        market_data, feature_state, manifest, provider, end=end, spy_revision=spy
    )
    gate = WorkspaceMutationGate()
    # The product layout: artifacts beside the database, which the residue
    # owner resolves from the workspace root.
    resolver = ArtifactResolver(market_data.workspace / "artifacts")
    ledger, feature_persistence = open_closure(
        market_data=market_data,
        panel_state=panel_state,
        feature_state=feature_state,
        resolver=resolver,
        manifest=manifest,
    )
    service = FeatureFoundationService(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=provider,
        mutation_gate=gate,
        panel_artifacts=PanelArtifactCompositionOwner(resolver),
        feature_persistence=feature_persistence,
        sector_activation=SectorRevisionMapActivationCoordinator(
            store=feature_state, mutation_gate=gate, ledger=ledger
        ),
        session_authority_resolver=fixture_session_authority,
    )
    publisher = FeaturePanelSnapshotPublisher(
        feature_state=feature_state,
        panel_state=panel_state,
        resolver=resolver,
        mutation_gate=gate,
        recovery_binding=PanelRecoveryBindingPublisher(ledger=ledger, resolver=resolver),
        logical_identity=PanelLogicalIdentityPublisher(
            resolver=resolver,
            store=PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver)),
        ),
    )
    return PanelWorkspace(
        market_data,
        feature_state,
        panel_state,
        resolver,
        gate,
        ledger,
        service,
        publisher,
        provider,
    )


def build_and_publish(
    workspace: PanelWorkspace,
    manifest: UniverseManifest,
    *,
    spy: str,
    as_of: date,
    observed_at: datetime,
    invalidations: tuple[FeatureInvalidation, ...] = (),
    refresh_sector: bool,
) -> tuple[FeatureBuildOutcome, dict[str, object]]:
    request = FeatureBuildRequest.create(
        manifest_revision=manifest.revision_sha256,
        catalog=workspace.service.catalog.binding,
        spy_revision=spy,
        history_start=SESSIONS[0],
        as_of_session=as_of,
        invalidations=invalidations,
    )
    outcome = workspace.service.build(
        request, observed_at=observed_at, refresh_sector=refresh_sector
    )
    assert outcome.status is FeatureBuildStatus.COMPLETED, outcome
    published = workspace.publisher.publish(
        manifest=manifest, history_start=SESSIONS[0], as_of_session=as_of, observed_at=observed_at
    )
    return outcome, workspace.resolver.load_feature_panel_manifest(published.artifact.uri)


def new_session_invalidations(manifest: UniverseManifest, session: date):
    return tuple(
        FeatureInvalidation(
            "normal_new_session",
            listing_id=item.listing_id,
            earliest_session=session,
            affected_sessions=(session,),
            source_receipt_hash=canonical_hash([item.listing_id, session.isoformat()]),
        )
        for item in manifest.listings
    )
