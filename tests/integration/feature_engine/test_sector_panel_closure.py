from __future__ import annotations

import json
from dataclasses import replace
from dataclasses import replace as with_fields
from datetime import UTC, date, datetime, timedelta
from threading import Lock

import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.contracts import (
    canonical_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
)
from alphalattice.foundation.feature_engine.producers.cross_section import (
    SectorNeutralPanelMaterializer,
)
from alphalattice.foundation.feature_engine.producers.reference_data import (
    SectorRefreshStager,
    SectorTransportPolicy,
)
from alphalattice.foundation.feature_engine.publication.sector_map_activation import (
    SectorRevisionMapActivationCoordinator,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    ProviderFetchError,
    SectorObservation,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError

NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)


def _activation(feature_state, gate, artifact_root):
    return SectorRevisionMapActivationCoordinator(
        store=feature_state,
        mutation_gate=gate,
        ledger=FeatureClosureLedger(PanelClosureArtifactStore(ArtifactResolver(artifact_root))),
    )


def test_small_sector_is_warning_without_changing_equal_mean_policy() -> None:
    catalog = FeatureCatalog.load()
    rows = [
        {
            "listing_id": f"listing-{index:02d}",
            "session_date": "2026-08-03",
            **{
                factor_id: float(index + factor_position / 100)
                for factor_position, factor_id in enumerate(catalog.factor_ids)
            },
        }
        for index in range(18)
    ]
    sectors = {
        f"listing-{index:02d}": "sector-a" if index < 9 else "sector-b" for index in range(18)
    }
    panel = SectorNeutralPanelMaterializer(catalog).materialize(
        feature_rows=rows,
        active_listing_ids=tuple(sectors),
        manifest_revision="a" * 64,
        sector_revision="b" * 64,
        sector_by_listing_id=sectors,
        spy_revision="c" * 64,
    )
    assert panel.complete
    assert panel.admission.research_admissible
    assert len(panel.admission.small_sector_warning_factors) == len(catalog.factor_ids)
    assert all(item["small_sector_warning"] for item in panel.availability)


def test_sector_map_publication_failure_prevents_store_activation(tmp_path, monkeypatch) -> None:
    profile = MarketProfile(
        market_profile_id="sector-map-failure-fixture",
        display_name="Sector map failure fixture",
        market="US",
        currency="USD",
        calendar_id="XNYS",
        provider="fixture",
        daily_price_basis="unadjusted",
        manifest_as_of=NOW.date(),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    manifest = UniverseManifest(
        manifest_id="sector-map-failure-fixture",
        profile=profile,
        listings=(
            ManifestListing(
                listing_id="listing-a",
                symbol="AAA",
                mic="XNYS",
                provider_symbol="AAA",
            ),
        ),
        revision_sha256="d" * 64,
        universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
        is_point_in_time_historical=False,
    )
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    ledger = FeatureClosureLedger(PanelClosureArtifactStore(ArtifactResolver(tmp_path / "closure")))
    coordinator = SectorRevisionMapActivationCoordinator(
        store=feature_state,
        mutation_gate=WorkspaceMutationGate(),
        ledger=ledger,
    )

    observations = (
        {
            "listing_id": "listing-a",
            "provider": "fixture",
            "provider_symbol": "AAA",
            "sector_name": "Technology",
            "sector_key": "technology",
            "payload_hash": "a" * 64,
            "evidence_hash": "b" * 64,
        },
    )

    # The ledger could not place the map on disk: a publication I/O refusal,
    # named apart from bad evidence, with the typed cause chained.
    def blocked_map(_sector_map) -> None:
        raise WorkspaceConflictError("fixture: replacement blocked", code="catalog.replace_blocked")

    monkeypatch.setattr(ledger, "publish_sector_map", blocked_map)
    with pytest.raises(ValueError, match=r"feature_closure\.publication_blocked") as blocked:
        coordinator.activate(manifest=manifest, observations=observations, observed_at=NOW)
    assert isinstance(blocked.value.__cause__, WorkspaceConflictError)
    assert feature_state.current_sector_state(manifest) is None

    def failing_disk(_sector_map) -> None:
        raise OSError("fixture artifact failure")

    monkeypatch.setattr(ledger, "publish_sector_map", failing_disk)
    with pytest.raises(ValueError, match=r"feature_closure\.publication_blocked"):
        coordinator.activate(manifest=manifest, observations=observations, observed_at=NOW)
    assert feature_state.current_sector_state(manifest) is None

    # The ledger refusing the evidence itself keeps its own name.
    def refused_map(_sector_map) -> None:
        raise ValueError("fixture: map identity mismatch")

    monkeypatch.setattr(ledger, "publish_sector_map", refused_map)
    with pytest.raises(ValueError, match=r"feature_closure\.sector_map_capture_failed"):
        coordinator.activate(manifest=manifest, observations=observations, observed_at=NOW)
    assert feature_state.current_sector_state(manifest) is None


@pytest.mark.parametrize(
    "refresh_failure",
    ("sector.missing_current_sector", "sector.identity_mismatch", "data.rate_limited"),
)
def test_sector_staging_is_resumable_and_commits_only_at_fan_in(tmp_path, refresh_failure) -> None:
    """Requirement: the staging resumes by cursor and commits only at fan-in; regression
    it is JSON and binds no transport, so a staging deferred under two workers
    resumes under a one-worker policy, each chunk recording the workers that fetched it."""
    profile = MarketProfile(
        market_profile_id="yaml-sector-staging-fixture",
        display_name="YAML sector staging fixture",
        market="US",
        currency="USD",
        calendar_id="XNYS",
        provider="fixture",
        daily_price_basis="unadjusted",
        manifest_as_of=NOW.date(),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    listings = tuple(
        ManifestListing(
            listing_id=f"yaml-listing-{index:02d}",
            symbol=f"Y{index:02d}",
            mic="XNYS",
            provider_symbol=f"Y{index:02d}",
        )
        for index in range(6)
    )
    manifest = UniverseManifest(
        manifest_id="yaml-sector-staging-fixture",
        profile=profile,
        listings=listings,
        revision_sha256="e" * 64,
        universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
        is_point_in_time_historical=False,
    )

    class Provider:
        name = "fixture"

        def __init__(self) -> None:
            self.calls: dict[str, int] = {}
            self.lock = Lock()

        def fetch_current_sector(self, *, provider_symbol: str) -> SectorObservation:
            with self.lock:
                self.calls[provider_symbol] = self.calls.get(provider_symbol, 0) + 1
                attempt = self.calls[provider_symbol]
            if provider_symbol == "Y02" and attempt == 1:
                raise ProviderFetchError("data.rate_limited", "fixture 429", retryable=True)
            return SectorObservation(
                provider=self.name,
                provider_symbol=provider_symbol,
                sector_name=f"sector-{int(provider_symbol[1:]) % 2}",
                sector_key=None,
                payload_hash=canonical_hash([provider_symbol, "sector"]),
            )

    provider = Provider()
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    gate = WorkspaceMutationGate()
    stager = SectorRefreshStager(
        manifest=manifest,
        provider=provider,
        artifact_root=tmp_path / "sector-staging",
        staging_id=canonical_hash([manifest.revision_sha256, "initial-sector-refresh"]),
        activation_coordinator=_activation(feature_state, gate, tmp_path / "closure"),
        transport_policy=SectorTransportPolicy(max_workers=2),
    )
    first = stager.acquire(observed_at=NOW)
    assert first.status == "deferred"
    assert stager.path.suffix == ".json" and stager.path.exists()
    assert feature_state.staged_sector_observations(manifest.revision_sha256) == []
    assert feature_state.current_sector_state(manifest) is None
    stager.commit(store=feature_state, mutation_gate=gate, observed_at=NOW)
    calls_before_early_resume = dict(provider.calls)
    early = stager.acquire(observed_at=NOW)
    assert early.status == "deferred"
    assert provider.calls == calls_before_early_resume

    due = NOW.replace(minute=NOW.minute + 6)

    serial = with_fields(stager, transport_policy=SectorTransportPolicy(max_workers=1))
    acquired = serial.acquire(observed_at=due)
    assert acquired.status == "completed"
    staged = json.loads(stager.path.read_text(encoding="utf-8"))
    assert "transport_policy_hash" not in staged
    assert [(chunk["listings"], chunk["workers"]) for chunk in staged["chunks"]] == [(6, 2), (1, 1)]
    assert feature_state.staged_sector_observations(manifest.revision_sha256) == []
    assert feature_state.current_sector_state(manifest) is None
    committed = stager.commit(
        store=feature_state,
        mutation_gate=gate,
        observed_at=due,
    )
    assert committed.status == "completed"
    assert feature_state.current_sector_state(manifest) is not None
    assert all(count == 1 for symbol, count in provider.calls.items() if symbol != "Y02")
    assert provider.calls["Y02"] == 2

    prior = feature_state.current_sector_state(manifest)
    fetch = provider.fetch_current_sector

    def unavailable(*, provider_symbol):
        if provider_symbol == "Y05":
            raise ProviderFetchError(refresh_failure, "controlled Sector failure", retryable=False)
        return fetch(provider_symbol=provider_symbol)

    provider.fetch_current_sector = unavailable
    later = NOW + timedelta(days=31)
    refresh = replace(stager, staging_id=canonical_hash([stager.staging_id, "review"]))
    assert refresh.acquire(observed_at=later).status == (
        "deferred" if refresh_failure == "data.rate_limited" else "blocked"
    )
    result = refresh.commit(store=feature_state, mutation_gate=gate, observed_at=later)
    assert result.status == (
        "blocked" if refresh_failure == "sector.identity_mismatch" else "deferred"
    )
    assert feature_state.current_sector_state(manifest) == prior
    if refresh_failure == "sector.missing_current_sector":
        assert result.failure_code == "sector.refresh_unavailable_prior_reference_retained"
        assert feature_state.sector_progress(manifest.revision_sha256)["status"] == "DEFERRED"
    elif refresh_failure == "data.rate_limited":
        assert result.failure_code == "data.rate_limited" and result.retry_after_at is not None


def test_reachability_reports_orphans_without_deleting(tmp_path) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")

    def publish(seed: str):
        row_hash = canonical_hash(seed)
        table = pa.Table.from_pylist(
            [
                {
                    "session_date": date(2026, 8, 3),
                    "listing_id": f"listing-{seed}",
                    "row_hash": row_hash,
                }
            ]
        )
        chunk_hash = canonical_hash({"seed": seed, "row_hash": row_hash})
        chunk = resolver.publish_feature_panel_chunk(
            table=table,
            content_hash=chunk_hash,
            metadata={"panel_binding_hash": "a" * 64, "calendar_year": "2026"},
        )
        payload = {
            "kind": "FeaturePanelSnapshotManifest",
            "chunks": [{"chunk_hash": chunk_hash, "uri": chunk.uri}],
            "safe_summary": {"seed": seed},
        }
        snapshot_hash = canonical_hash(payload)
        artifact = resolver.publish_feature_panel_manifest(
            payload={**payload, "snapshot_hash": snapshot_hash},
            snapshot_hash=snapshot_hash,
        )
        return artifact, chunk

    rooted, rooted_chunk = publish("rooted")
    orphan, orphan_chunk = publish("orphan")
    report = resolver.feature_panel_reachability(root_manifest_uris=(rooted.uri,))
    assert rooted.uri in report.reachable
    assert rooted_chunk.uri in report.reachable
    assert orphan.uri in report.unreferenced
    assert orphan_chunk.uri in report.unreferenced
    assert resolver.inspect_feature_panel_snapshot(rooted.uri) == {"seed": "rooted"}
    # Classification is observational. No method in this case can remove the orphan.
    orphan_path = (
        tmp_path / "artifacts" / "feature-panel" / "chunks" / f"{orphan_chunk.content_hash}.parquet"
    )
    assert orphan_path.exists()
