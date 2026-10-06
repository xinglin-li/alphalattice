"""First use judges U0 from the rows the current source still explains,
through real numerical owners with recorded source adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.task_control.contracts import TaskLifecycle
from tests.portfolio_strategy_lab.local_web_support import _json
from tests.researcher_methodology_surface.real_workspace import (
    OBSERVED_AT,
    _source_loader_for,
)


def test_first_use_judges_u0_from_rows_the_current_source_still_explains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """counterexample: a row that exists is not a row that still applies to the source.

    First use has materialized every candidate's as-of row (the build that
    the partial sector then refused) and U0 is not frozen. At the lawful
    pause between two cycles -- the coordinator answered RUNNING after the
    sector exclusion, the next cycle has not started -- two legitimate
    corrections land through the existing write and audit entries, each
    followed by the fresh full-history action audit the correction needs:

    * A's last 64 volumes were identical, so ``price_volume_corr_63`` (a
      shipped baseline factor over adjusted close and ``volume_raw``) has a
      zero denominator at the as-of session and its stored row excludes A;
      one corrected volume inside the window makes it computable.
    * B computed normally; the correction makes its last 64 volumes
      identical, so the recomputed row cannot qualify B.

    The early baseline judgement must not exclude A from its stale row: the
    row's source-verification receipt no longer verifies against today's
    inputs, so the judgement is left to the build, which recomputes both and
    is then judged by the owner it always was. U0 therefore holds A and not
    B -- the same membership the original order (judge after the build)
    produces -- and B's exclusion carries its recomputed reason; the
    qualification obligation written into the membership earlier skips no
    later check.
    """

    from datetime import timedelta
    from uuid import UUID

    from alphalattice.control.data_platform.maintenance.coordinator import (
        WorkspaceMaintenanceCoordinator,
    )
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.sources.sanitization import sanitize_payload
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
    from alphalattice.kernel.data.calendar import materialize_calendar_schedule
    from tests.researcher_methodology_surface.real_workspace import (
        AS_OF,
        HISTORY_START,
        SeededWalkProvider,
    )

    class VolumeShapedProvider(SeededWalkProvider):
        """The seeded walk, with per-symbol volume overrides by session."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.volume_overrides: dict[str, dict[str, int]] = {}

        def fetch_daily(self, symbols, *, start, end):  # type: ignore[no-untyped-def]
            result = super().fetch_daily(symbols, start=start, end=end)
            for symbol, rows in result.items():
                overrides = self.volume_overrides.get(symbol)
                if overrides:
                    result[symbol] = tuple(
                        {**row, "volume": overrides.get(str(row["session_date"]), row["volume"])}
                        for row in rows
                    )
            return result

    symbols = tuple(f"F{i:03d}" for i in range(104))
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(v["session_date"] for v in schedule.to_pylist())
    provider = VolumeShapedProvider(symbols, sessions)
    provider.sectors = {s: f"Sector-{i % 3}" for i, s in enumerate(symbols)}
    provider.sectors["SPY"] = "Reference"
    missing_sector = symbols[7]
    del provider.sectors[missing_sector]
    stale_unavailable, stale_available = symbols[20], symbols[41]
    tail = [s.isoformat() for s in sessions[-64:]]
    # A: the last 64 volumes identical at first use.
    provider.volume_overrides[stale_unavailable] = dict.fromkeys(tail, 2_000_000)
    live = LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT)
    live.data_provider, live.data_source_loader = provider, _source_loader_for(symbols)
    market = MarketDataRepository(tmp_path)
    corrected_at: list[str] = []
    original_run = WorkspaceMaintenanceCoordinator.run

    def correct_between_cycles(self, request, **kwargs):
        outcome = original_run(self, request, **kwargs)
        if outcome.failure_code == "sector.recoverable_exclusion_applied" and not corrected_at:
            # The pause: this cycle returned, the next has not begun; the
            # coordinator holds nothing across it.
            corrected_at.append(outcome.failure_code)
            readiness = market.readiness.load("us-current-index-research")
            assert readiness is not None and readiness.active_manifest_revision is not None
            manifest = market.load_universe_manifest_revision(readiness.active_manifest_revision)
            # Observed before this frozen clock's instant, as a correction that
            # landed before the next cycle is; the audits stay fresh.
            observed_at = OBSERVED_AT - timedelta(minutes=1)
            provider.volume_overrides[stale_unavailable][tail[-5]] = 2_600_000
            provider.volume_overrides[stale_available] = dict.fromkeys(tail, 1_500_000)
            for symbol in (stale_unavailable, stale_available):
                listing = next(item for item in manifest.listings if item.symbol == symbol)
                rows = provider.fetch_daily((symbol,), start=sessions[-64], end=AS_OF)[symbol]
                live.session.mutation_gate.run(
                    market.apply_validated_batch,
                    manifest,
                    sanitize_payload(manifest, provider.name, {symbol: rows}, (symbol,)),
                    ingestion_id=f"correction:{listing.listing_id}",
                    observed_at=observed_at,
                )
                bars = market.raw_bars(listing.listing_id, through=AS_OF)
                live.session.mutation_gate.run(
                    market.complete_action_audit,
                    manifest,
                    listing_id=listing.listing_id,
                    provider=provider.name,
                    observed_actions=(),
                    observed_adjusted_closes=provider.fetch_adjusted_close_history(
                        listing_id=listing.listing_id,
                        provider_symbol=symbol,
                        start=bars[0].session_date,
                        end=AS_OF,
                    ),
                    history_start=bars[0].session_date,
                    history_end=AS_OF,
                    requested_as_of=AS_OF,
                    observed_at=observed_at,
                )
        return outcome

    monkeypatch.setattr(WorkspaceMaintenanceCoordinator, "run", correct_between_cycles)
    with live:
        # Keep the full U0 counterexample, with two cores per parallel test worker.
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
        live.dispatcher.drain_for_tests(timeout=1200)
        record = live.session.task_control_registry.task(task_id)
        app = live.operations.preparation
        assert record.lifecycle is TaskLifecycle.SUCCEEDED, (
            record.failure_code,
            live.dispatcher.failure(task_id),
            app.readback(),
        )
        assert corrected_at == ["sector.recoverable_exclusion_applied"]
        readiness = market.readiness.load("us-current-index-research")
        assert readiness is not None and readiness.active_manifest_revision is not None
        final = market.load_universe_manifest_revision(readiness.active_manifest_revision)
        members = {item.symbol for item in final.listings}
        assert stale_unavailable in members, "the corrected candidate was judged from its stale row"
        assert stale_available not in members
        assert missing_sector not in members
        assert len(members) == len(symbols) - 2
        panel_state = PanelStateRepository(market.database, market_data=market)
        by_symbol = {item.symbol: item.listing_id for item in final.listings}
        parent_revision = app._load(task_id, "prepare_data")["manifest_revision"]
        quarantined = {
            q.listing_id: q.reason_codes
            for q in panel_state.active_listing_quarantines(
                parent_revision, include_profile_history=True
            )
        }
        excluded_id = next(
            item.listing_id
            for item in market.load_universe_manifest_revision(parent_revision).listings
            if item.symbol == stale_available
        )
        assert (
            "BASE_FEATURE_UNAVAILABLE:price_volume_corr_63:zero_denominator"
            in (quarantined[excluded_id])
        )
        assert by_symbol[stale_unavailable] not in quarantined
        snapshot = panel_state.feature_panel_snapshot_for_active("us-current-index-research")
        assert snapshot is not None
        assert str(snapshot["manifest_revision"]) == final.revision_sha256
        assert _json(live, "/api/experiments/controls")["status"] == "READY"
        # The stale exclusion deferred the judgement to the sector-exclusion child, a
        # membership of its own (both corrected rows recomputed there); the membership that
        # judgement derived was built once more and published as the one Panel. Since W10
        # (V102) the build is its maintenance cycle's step, so the passes are the cycle's
        # records.
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
        readiness_before_final(market, parent_revision, final)


def readiness_before_final(market, parent_revision: str, final):
    """The sector-exclusion child: the parent minus the sector-less listing, no obligation."""

    parent = market.load_universe_manifest_revision(parent_revision)
    connection = market.database.connect(read_only=True)
    try:
        revisions = [
            row[0]
            for row in connection.execute(
                "SELECT revision_sha256 FROM universe_manifest WHERE listing_count = ?",
                [len(parent.listings) - 1],
            ).fetchall()
        ]
    finally:
        connection.close()
    assert len(revisions) == 1, revisions
    assert revisions[0] != final.revision_sha256
    return revisions[0]
