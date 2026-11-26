"""First use continues after a recoverable sector exclusion, through real
numerical owners with recorded source adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import UUID

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.foundation.market_data_ops.sources.providers import ProviderFetchError
from alphalattice.foundation.market_data_ops.storage import market_snapshots
from tests._support import development_universe
from tests.portfolio_strategy_lab.host_requests import _json
from tests.researcher_methodology_surface.real_workspace import OBSERVED_AT


def test_first_use_continues_after_recoverable_sector_exclusion(tmp_path: Path, monkeypatch):
    """First use continues after recoverable sector exclusion."""

    from alphalattice.foundation.feature_engine.producers.reference_data import SectorRefreshStager
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
    from alphalattice.kernel.resources import current_resource_lease

    sector_workers = []
    acquire = SectorRefreshStager.acquire

    def observe_acquisition(stager, **kwargs):
        lease = current_resource_lease()
        sector_workers.append(None if lease is None else lease.amounts.workers)
        return acquire(stager, **kwargs)

    monkeypatch.setattr(SectorRefreshStager, "acquire", observe_acquisition)
    symbols = {item.symbol for item in development_universe.LISTINGS} - {"DVM", "DVQ", "DVR", "DVS"}
    provider = development_universe.development_provider()
    missing = "DVK"
    sector = provider.fetch_current_sector

    def missing_sector(*, provider_symbol):
        if provider_symbol == missing:
            raise ProviderFetchError(
                "sector.missing_current_sector", "not observed", retryable=False
            )
        return sector(provider_symbol=provider_symbol)

    monkeypatch.setattr(provider, "fetch_current_sector", missing_sector)
    live = LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT)
    live.data_provider, live.data_source_loader = (
        provider,
        development_universe.source_loader,
    )
    with live:
        budget = live.operations.set_cpu_budget("1", chosen_by="EXTERNAL_AUTOMATION")
        assert budget["status"] == "CPU_BUDGET" and budget["cpu_budget"] == 1
        plan = _json(live, "/api/workspace/preparation/plan", method="POST", payload={})
        admitted = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={"preparation_plan_hash": plan["plan_hash"]},
        )
        task_id = UUID(admitted["task_id"])
        live.dispatcher.drain_for_tests(timeout=900)
        record = live.session.task_control_registry.task(task_id)
        app = live.operations.preparation
        assert record.lifecycle is TaskLifecycle.SUCCEEDED, (
            record.failure_code,
            live.dispatcher.failure(task_id),
            app.readback(),
        )
        assert sector_workers and set(sector_workers) == {1}, (
            "the sector coordinator carries its grant"
        )
        parent_revision = app._load(task_id, "prepare_data")["manifest_revision"]
        progress = app._load(task_id, "progress", optional=True) or {}
        market = MarketDataRepository(tmp_path)
        readiness = market.readiness.load("us-current-index-research")
        assert readiness is not None and readiness.active_manifest_revision != parent_revision
        child = market.load_universe_manifest_revision(readiness.active_manifest_revision)
        parent = market.load_universe_manifest_revision(parent_revision)
        assert {v.symbol for v in parent.listings} == symbols
        assert {v.symbol for v in parent.listings} - {v.symbol for v in child.listings} == {
            missing,
            "DVD",
            "DVI",
        }
        assert progress["membership_revision"] == child.revision_sha256
        quarantines = PanelStateRepository(
            market.database, market_data=market
        ).active_listing_quarantines(parent_revision)
        assert [q.listing_id for q in quarantines] == [
            v.listing_id for v in parent.listings if v.symbol == missing
        ]
        snapshot = PanelStateRepository(
            market.database, market_data=market
        ).feature_panel_snapshot_for_active("us-current-index-research")
        assert snapshot is not None
        assert str(snapshot["manifest_revision"]) == child.revision_sha256
        assert (
            str(snapshot["snapshot_hash"])
            == app._load(task_id, "prepare_features")["snapshot_hash"]
        )
        assert _json(live, "/api/experiments/controls")["status"] == "READY"
        manifest = read_research_workspace_manifest(tmp_path)
        bundle = read_factor_bundle(tmp_path, manifest.experiment_inputs[0].binding_hash)
        assert bundle.panel_snapshot_hash == str(snapshot["snapshot_hash"])
        assert bundle.binding_version == 3 and bundle.market_selection is not None
        bound = MarketDataRepository(factor_input_paths(tmp_path, bundle.binding_hash)[0])
        selections = []
        for held in (market, bound):
            with held.database.read_transaction() as connection:
                selections.append(market_snapshots.closed_selection(connection, held.path))
        selection = selections[0]
        assert selection is not None and selection == selections[1]
        assert selection[0] == bundle.market_catalog_snapshot
        listing = child.listings[0].listing_id
        assert bound.raw_bars(listing) == market.raw_bars(listing) != ()
        # A sealed Parquet body or manifest document is stored once, the binding's.
        kept = factor_input_paths(tmp_path, bundle.binding_hash)[0]
        sealed = [n for n, _ in bundle.files if re.search(r"(\.parquet|/[0-9a-f]{64}\.json)$", n)]
        assert sealed and all((tmp_path / name).samefile(kept / name) for name in sealed)
        # One Panel is composed for the counterexample: the first pass
        # stops on the recoverable sector
        # exclusion over the parent, the baseline qualification applies from the rows it
        # left, and one pass completes over the membership every qualification derived and
        # step, so the passes are the cycle's records.
        connection = market.database.connect(read_only=True)
        try:
            cycles = connection.execute(
                "SELECT phase, status, failure_code, effect_receipts_json "
                "FROM workspace_maintenance_cycle ORDER BY created_at"
            ).fetchall()
        finally:
            connection.close()
        assert [row[:3] for row in cycles] == [
            ("feature", "running", "sector.recoverable_exclusion_applied"),
            ("quality", "running", "feature.baseline_qualification_applied"),
            ("completed", "completed", None),
        ], cycles
        published = [json.loads(row[3]) for row in cycles if json.loads(row[3])]
        assert len(published) == 1 and published[0][-1] == str(snapshot["snapshot_hash"])
