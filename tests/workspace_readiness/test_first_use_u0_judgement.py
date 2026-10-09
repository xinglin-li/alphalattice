"""First use judges the counterexample from the rows the current source still explains,
through real numerical owners with recorded source adapters."""

from __future__ import annotations

import json
from contextvars import copy_context
from datetime import date, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.task_control.contracts import TaskLifecycle, WorkItemLifecycle
from alphalattice.interface.local_application.cli_contract import ANSWER_LANGUAGE, worded
from alphalattice.interface.local_application.client import LocalResearchClient
from tests.portfolio_strategy_lab.local_web_support import _json
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    OBSERVED_AT,
    _source_loader_for,
)
from tests.workspace_maintenance.local_data_provider import RecordingProvider, recording_provider


class LaggingProvider(RecordingProvider):
    def __init__(self, symbols, *, lag_sessions=1):
        super().__init__(symbols, recording_provider(now=OBSERVED_AT, symbols=symbols).sessions)
        self.covered = dict.fromkeys(symbols, self.sessions[-lag_sessions - 1])

    def fetch_daily(self, requested, *, start, end):
        return {
            symbol: super(LaggingProvider, self).fetch_daily(
                (symbol,), start=start, end=min(end, self.covered.get(symbol, end))
            )[symbol]
            for symbol in requested
        }


def test_first_use_judges_u0_from_rows_the_current_source_still_explains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """First use judges baseline eligibility from rows whose current source verification still
    applies."""

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
        # Keep the full counterexample, with two cores per parallel test worker.
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
        # the build is its maintenance cycle's step, so the passes are the cycle's
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


def test_a_baseline_short_of_qualified_names_says_how_many_why_and_the_way_on(tmp_path: Path):
    """Partial lag with five current names still reaches Feature's per-sector judgement.
    The real numerical qualification and Host need more than ten seconds."""

    symbols = tuple(f"F{i:03d}" for i in range(10))
    provider = LaggingProvider(symbols, lag_sessions=20)
    latest = provider.sessions[-21]
    provider.covered.update(dict.fromkeys((*symbols[:2], *symbols[5:8]), AS_OF))
    live = LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT)
    live.data_provider = provider
    live.data_source_loader = _source_loader_for(symbols)
    with live:
        plan = _json(live, "/api/workspace/preparation/plan", method="POST", payload={})
        admitted = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={"preparation_plan_hash": plan["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=900)
        task = live.session.task_control_registry.task(UUID(admitted["task_id"]))
        assert task.failure_code == "feature.baseline_qualified_population_insufficient", task
        shown = LocalResearchClient(live.workspace).request(
            {"operation": "STATUS", "task_id": admitted["task_id"]}
        )
    assert "five in each sector present" in shown["detail"], shown["detail"]
    cause = shown["failure_cause"]
    assert cause["exception_type"] == "FeatureBaselinePopulationError"
    assert cause["detail"].startswith(f"5 of 10 candidates qualify at {AS_OF}"), cause
    assert f"5 no market observation at the session (latest bar {latest})" in cause["detail"]


def test_a_lagging_provider_resumes_its_data_stage_before_any_feature_work(tmp_path, monkeypatch):
    """Real Host, source admission and publication's 100-listing floor take over ten seconds."""
    symbols = tuple(f"L{i:03d}" for i in range(100))
    provider = LaggingProvider(symbols)
    target, latest = provider.sessions[-1], provider.sessions[-2]
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.setenv("CODEX_THREAD_ID", "00000000-0000-4000-8000-0000000000d1")
    live = LocalPortfolioWebSession.from_workspace(tmp_path, clock=lambda: OBSERVED_AT)
    live.data_provider, live.data_source_loader = provider, _source_loader_for(symbols)
    with live:
        agent = LocalResearchClient(tmp_path)
        goal = dict(kind="FIRST_USE", title="First use", objective="Build a reviewed book.")
        goal["criteria"] = [dict(criterion_id="book", text="A reviewed book.")]
        opened = agent.request(
            dict(operation="GOAL_OPEN", goal_declaration=goal, change_reason="First")
        )
        assert opened["goal_id"]
        budget = agent.request(dict(operation="CPU_BUDGET_SET", cpu_budget="2"))
        assert budget["cpu_budget"] == 2
        plan = agent.request({"operation": "WORKSPACE_PREPARE_PLAN"})
        admitted = agent.request(plan["next_requests"]["confirm"])
        assert admitted["status"] == "ADMITTED", admitted
        task_id = UUID(admitted["task_id"])
        registry = live.session.task_control_registry
        read = dict(operation="WORKSPACE_PREPARE_READBACK", task_id=str(task_id))
        chinese = copy_context()
        chinese.run(ANSWER_LANGUAGE.set, "zh")
        for reaching in (0, 4):
            live.dispatcher.drain_for_tests(timeout=900)
            record, stages = registry.task_with_work_items(task_id)
            stages = {stage.stage_id: stage for stage in stages}
            assert record.lifecycle is TaskLifecycle.DEFERRED
            assert record.failure_code == "data.target_session_not_covered"
            assert stages["freeze_sources"].lifecycle is WorkItemLifecycle.VERIFIED
            status = agent.request({"operation": "STATUS", "task_id": str(task_id)})
            assert status["failure_cause"]["step"] == "prepare_data"
            detail = status["failure_cause"]["detail"]
            facts = (str(target), f"{reaching} of 100", "at least 5", str(latest))
            assert all(value in detail for value in facts)
            translated = chinese.run(worded, detail)
            assert translated != detail and any("\u4e00" <= c <= "\u9fff" for c in translated)
            assert all(str(v) in translated for v in (target, latest, reaching, 100, 5))
            assert stages["prepare_features"].attempt_count == 0
            assert date.fromisoformat(record.input.payload["plan"]["target_session"]) == target
            held = agent.request(read)
            assert held["failure_code"] == record.failure_code
            resume = held["next_requests"]["resume"]
            assert resume["preparation_plan_hash"] == plan["plan_hash"]
            covered = symbols[:4] if reaching == 0 else symbols
            provider.covered.update(dict.fromkeys(covered, target))
            provider.calls.clear()
            assert agent.request(resume)["task_id"] == str(task_id)
        live.dispatcher.drain_for_tests(timeout=900)
        assert registry.task(task_id).lifecycle is TaskLifecycle.SUCCEEDED
        finished = agent.request(read)
        assert finished["inputs"] and finished["published_binding_hash"]
        fetched = {symbol for names, _, _ in provider.calls for symbol in names}
        assert fetched.intersection(symbols) == set(symbols[4:])
