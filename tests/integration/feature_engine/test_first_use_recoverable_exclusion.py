"""First use continues after a recoverable sector exclusion, through real
numerical owners with recorded source adapters."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import read_factor_bundle
from alphalattice.control.task_control.contracts import TaskLifecycle
from tests.portfolio_strategy_lab.local_web_support import _json
from tests.researcher_methodology_surface.real_workspace import (
    OBSERVED_AT,
    _source_loader_for,
)


def test_first_use_continues_after_recoverable_sector_exclusion(tmp_path: Path):
    """First use continues after recoverable sector exclusion."""

    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
    from alphalattice.kernel.data.calendar import materialize_calendar_schedule
    from tests.researcher_methodology_surface.real_workspace import (
        AS_OF,
        HISTORY_START,
        SeededWalkProvider,
    )

    symbols = tuple(f"F{i:03d}" for i in range(104))
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    provider = SeededWalkProvider(symbols, tuple(v["session_date"] for v in schedule.to_pylist()))
    # Three wide sectors keep every sector above the gateway minimum after one
    # exclusion; the missing symbol makes the sector refresh partial.
    provider.sectors = {s: f"Sector-{i % 3}" for i, s in enumerate(symbols)}
    provider.sectors["SPY"] = "Reference"
    missing = symbols[7]
    del provider.sectors[missing]
    live = LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT)
    live.data_provider, live.data_source_loader = provider, _source_loader_for(symbols)
    with live:
        # Four pytest workers share the CPU; the full first-use scope is unchanged.
        budget = live.operations.set_cpu_budget("2", chosen_by="EXTERNAL_AUTOMATION")
        assert budget["status"] == "CPU_BUDGET" and budget["cpu_budget"] == 2
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
        parent_revision = app._load(task_id, "prepare_data")["manifest_revision"]
        progress = app._load(task_id, "progress", optional=True) or {}
        market = MarketDataRepository(tmp_path)
        readiness = market.readiness.load("us-current-index-research")
        assert readiness is not None and readiness.active_manifest_revision != parent_revision
        child = market.load_universe_manifest_revision(readiness.active_manifest_revision)
        parent = market.load_universe_manifest_revision(parent_revision)
        assert len(parent.listings) == len(symbols)
        assert {v.symbol for v in parent.listings} - {v.symbol for v in child.listings} == {missing}
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
