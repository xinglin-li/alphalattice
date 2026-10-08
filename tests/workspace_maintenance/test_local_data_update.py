"""Existing-workspace update through real Data/Feature owners, offline."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.maintenance.data_update import (
    WorkspaceDataUpdateApplication,
    bind_existing_data_workspace,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import (
    InstalledAgent,
    _json,
    _resolved,
    _Resolver,
)
from tests.researcher_methodology_surface.real_workspace import (
    HISTORY_START,
    SYMBOLS,
    _bootstrap,
)
from tests.workspace_maintenance.data_update_support import _provider
from tests.workspace_maintenance.local_data_provider import (
    NOW,
    recording_provider,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("target_current", [True, False])
def test_recovery_lookup_validates_current_binding_only_for_the_selected_plan(target_current):
    from types import SimpleNamespace

    from alphalattice.control.data_platform.maintenance.contracts import WorkspaceDataUpdatePlan
    from alphalattice.control.product_host.storage.plan_previews import PreviewRegistry

    owner = object.__new__(WorkspaceDataUpdateApplication)
    owner._previews = PreviewRegistry(
        model=WorkspaceDataUpdatePlan, clock=lambda: NOW, hash_field="content_hash"
    )
    prior = SimpleNamespace(content_hash="a" * 64, current=False)
    selected = SimpleNamespace(content_hash="b" * 64, current=target_current)
    tasks = [SimpleNamespace(task_kind=owner.task_kind, plan=p) for p in (prior, selected)]
    owner.session = SimpleNamespace(task_control_registry=SimpleNamespace(tasks=lambda: tasks))
    checked = []

    def require(plan):
        checked.append(plan.content_hash)
        if not plan.current:
            raise ValueError("workspace_data_update.task_binding_mismatch")

    def parse(task, *, require_current=True):
        # The real parser validates intrinsic identity even for historical Tasks;
        # its optional current binding check belongs only to the selected target.
        if require_current:
            require(task.plan)
        return task.plan

    owner._plan_of = parse
    owner._require_plan = require
    if target_current:
        assert owner.prepare(selected.content_hash) is selected
    else:
        with pytest.raises(ValueError, match="task_binding_mismatch"):
            owner.prepare(selected.content_hash)
    assert checked == [selected.content_hash]


def test_data_change_scope_never_turns_into_an_arbitrary_audit_grant():
    from alphalattice.control.data_platform.maintenance.contracts import WorkspaceDataChange

    change = WorkspaceDataChange.seal(
        action="FULL_HISTORY_AUDIT", full_history_listing_ids=("one",)
    )
    assert change.full_history_listing_ids == ("one",)
    for values in (
        {"action": "UNIVERSE"},
        {"action": "FULL_HISTORY_AUDIT", "full_history_listing_ids": ()},
        {
            "action": "FULL_HISTORY_AUDIT",
            "full_history_listing_ids": ("one",),
            "additions": ("OTHER",),
        },
        {
            "action": "UNIVERSE",
            "transition_id": "a",
            "candidate_document_hash": "b",
            "additions": ("X",),
            "removals": ("X",),
        },
    ):
        with pytest.raises(ValueError):
            WorkspaceDataChange.seal(**values)


def test_data_change_confirmation_is_human_only_before_any_io():
    owner = object.__new__(WorkspaceDataUpdateApplication)
    for caller in ("INSTALLED_AGENT", "SERVICE_AUTOMATION"):
        with pytest.raises(ValueError, match="human_confirmation_required"):
            owner.confirm("a" * 64, caller=caller)


def test_verified_universe_proposal_requires_human_and_binds_book_head(qualified, tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    provider = _provider()
    bootstrap = _bootstrap((*SYMBOLS[1:], "NEW"))
    change_time = NOW + timedelta(days=7)
    with LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=lambda **_: bootstrap,
        clock=lambda: change_time,
    ) as service:
        owner = service.operations.data_update
        market = MarketDataRepository(workspace)
        gate = owner._readiness(market)
        assert (
            gate.refresh_sources_if_due(observed_at=change_time).status.value
            == "MANIFEST_UPDATE_PENDING"
        )
        heads = {"a" * 64: ("b" * 64, ())}
        owner.portfolio_obligations = lambda: heads
        plan = _json(service, "/api/data-update/plan", method="POST", payload={})
        assert plan["status"] == "CONFIRMATION_REQUIRED"
        assert plan["change"]["additions"] == ["NEW"]
        assert plan["change"]["removals"] == [SYMBOLS[0]]
        assert provider.calls == []
        before = len(service.session.task_control_registry.tasks())
        agent = InstalledAgent(service.operations)
        refused = json.loads(
            agent.invoke(
                PortfolioResearchAgentRequest(
                    operation="DATA_CHANGE_CONFIRM", update_plan_hash=plan["plan_hash"]
                )
            )
        )
        assert "human_confirmation_required" in str(refused)
        assert len(service.session.task_control_registry.tasks()) == before
        heads["a" * 64] = ("c" * 64, ())
        with pytest.raises(ValueError, match="proposal_stale"):
            owner.confirm(plan["plan_hash"], caller="HUMAN")
        heads["a" * 64] = ("b" * 64, ())
        approved = _json(
            service,
            "/api/data-update/confirm",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        repeated = _json(
            service,
            "/api/data-update/confirm",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        assert approved == repeated and approved["lifecycle"] == "QUEUED"
        assert len(service.session.task_control_registry.tasks()) == before + 1
        assert provider.calls == []
        stored = owner.changes.content.root / "data-update-plans" / (plan["plan_hash"] + ".json")
        original = stored.read_bytes()
        stored.write_text(original.decode().replace('"UNIVERSE"', '"OTHER"'))
        with pytest.raises(ValueError):
            owner.confirm(plan["plan_hash"], caller="HUMAN")
        stored.write_bytes(original)


@pytest.mark.parametrize("repair_source", [True, False])
def test_a_partial_membership_update_reopens_its_exact_approved_task(
    qualified, tmp_path, repair_source
):
    """A terminal source stop preserves approval and verified work across a real restart."""
    from dataclasses import asdict
    from uuid import UUID

    from alphalattice.control.data_platform.readiness import build_workspace_readiness
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository

    workspace = tmp_path / "partial-membership"
    shutil.copytree(qualified, workspace)
    binding = bind_existing_data_workspace(workspace)
    market = MarketDataRepository(workspace)
    prior = market.load_universe_manifest(
        market.readiness.load("us-current-index-research").active_manifest_id
    )
    source = _bootstrap((*SYMBOLS[1:], "NEW"))
    changed_at = NOW + timedelta(days=1)
    provider = recording_provider(symbols=(*SYMBOLS, "NEW"), now=changed_at)
    provider.sectors["NEW"] = "Sector-0"
    fetch = provider.fetch_daily
    corrupt = True

    def source_response(symbols, **kwargs):
        rows = fetch(symbols, **kwargs)
        active = market.readiness.load("us-current-index-research")
        if corrupt and active.active_manifest_revision != prior.revision_sha256:
            for symbol in symbols:
                if symbol == SYMBOLS[0] and rows[symbol]:
                    rows[symbol][0]["volume"] = -1
        return rows

    provider.fetch_daily = source_response
    settings = dict(
        workspace=workspace,
        workspace_manifest=binding,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=lambda **_: source,
        clock=lambda: changed_at,
    )
    with LocalPortfolioWebSession(**settings) as live:
        build_workspace_readiness(
            market,
            profile_path=ROOT / "config/market-profiles/us-current-index-research.yaml",
            provider=provider,
            source_loader=lambda **_: source,
        ).refresh_sources_if_due(observed_at=changed_at)
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert plan["status"] == "CONFIRMATION_REQUIRED"
        approved = _json(
            live,
            "/api/data-update/confirm",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        task_id = UUID(approved["task_id"])
        _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=300)
        stopped = _json(live, "/api/status?task_id=" + str(task_id))
        assert stopped["lifecycle"] == "BLOCKED", stopped
        assert stopped["latest_failure_code"] == "data.sanitizer.corrupted_payload"
        assert stopped["failure_cause"]["unit"] == SYMBOLS[0]
        assert stopped["failure_cause"]["exception_type"] == "CorruptedPayload"
        registry = live.session.task_control_registry
        original = registry.task(task_id)
        verified = registry.work_items(task_id)[0]
        assert verified.lifecycle.value == "VERIFIED" and verified.attempt_count == 1
        journal = market.manifest_transition(plan["change"]["transition_id"])
        assert journal.lifecycle == "ACTIVATED" and journal.approved_at is not None
        panel = PanelStateRepository(
            market.database, market_data=market
        ).feature_panel_snapshot_for_active("us-current-index-research")
        assert panel["manifest_revision"] == journal.prior_manifest_revision
        assert journal.next_manifest_revision != journal.prior_manifest_revision
        assert live.operations.data_update.readback(task_id)["receipt"] is None
        events = market.membership_events("us-current-index-research")

    corrupt = not repair_source
    with LocalPortfolioWebSession(**settings) as live:
        agent = InstalledAgent(live.operations)
        recovery = json.loads(
            agent.invoke(PortfolioResearchAgentRequest(operation="TASK_RECOVERY", task_id=task_id))
        )
        assert recovery["health"]["status"] == "TERMINAL_BLOCKED"
        assert recovery["stop"]["recoverable"] is False
        replan = recovery["next_requests"]["replan"]
        assert replan["recovery_task_id"] == str(task_id)
        fetched = len(provider.calls)
        stale = json.loads(
            agent.invoke(
                PortfolioResearchAgentRequest(**{**replan, "recovery_task_hash": "0" * 64})
            )
        )
        assert stale["failure_code"] == "local_application.confirmation_stale"
        assert len(provider.calls) == fetched
        assert live.session.task_control_registry.task(task_id) == original
        # Hostile drift in synthetic source authority cannot become a continuation grant.
        # Use its public writer; the unchanged active Manifest and Panel do not excuse a
        # different candidate document from the one the person approved.
        readiness = market.readiness.load("us-current-index-research")
        saved_state = asdict(readiness)
        saved_state["observed_at"] = saved_state.pop("updated_at")
        drifted = {**saved_state["active_candidate_manifest_document"], "unapproved": True}
        market.readiness.save(**{**saved_state, "active_candidate_manifest_document": drifted})
        try:
            refused = json.loads(agent.invoke(PortfolioResearchAgentRequest(**replan)))
            assert refused["failure_code"] == "workspace_data_update.transition_not_verified"
            assert len(provider.calls) == fetched
            assert live.session.task_control_registry.task(task_id) == original
            assert live.session.task_control_registry.work_items(task_id)[0] == verified
            assert live.operations.data_update.readback(task_id)["receipt"] is None
        finally:
            market.readiness.save(**saved_state)
        assert market.readiness.load("us-current-index-research") == readiness
        preview = json.loads(agent.invoke(PortfolioResearchAgentRequest(**replan)))
        assert preview["status"] == "PLANNED"
        assert preview["plan_hash"] == plan["plan_hash"]
        assert preview["change"] == plan["change"]
        assert preview["target_session"] == plan["target_session"]
        assert preview["next_requests"]["run"]["operation"] == "DATA_UPDATE_RUN"
        run = json.loads(
            agent.invoke(PortfolioResearchAgentRequest(**preview["next_requests"]["run"]))
        )
        assert run["task_id"] == str(task_id)
        live.dispatcher.drain_for_tests(timeout=300)
        registry = live.session.task_control_registry
        final = registry.task(task_id)
        assert final.input == original.input and final.admitted_at == original.admitted_at
        assert registry.work_items(task_id)[0] == verified
        assert registry.work_items(task_id)[1].attempt_count == 2
        assert market.manifest_transition(journal.transition_id) == journal
        assert len([t for t in registry.tasks() if t.task_kind == final.task_kind]) == 1
        final_events = market.membership_events("us-current-index-research")
        assert all(event in final_events for event in events)
        assert len({(e.listing_id, e.kind) for e in final_events}) == len(final_events)
        if repair_source:
            assert final.lifecycle is TaskLifecycle.SUCCEEDED, final.failure_code
            assert live.operations.data_update.readback(task_id)["receipt"] is not None
        else:
            assert final.lifecycle is TaskLifecycle.BLOCKED
            assert final.failure_code == "data.sanitizer.corrupted_payload"
            assert final_events == events
            assert live.operations.data_update.readback(task_id)["receipt"] is None


@pytest.mark.parametrize(
    "candidate_features_available",
    [True, False, "recoverable_range", "recoverable_sector", "recoverable_history"],
)
def test_new_source_member_is_feature_qualified_before_entry_and_its_history_stays_out(
    qualified, tmp_path, monkeypatch, candidate_features_available
):
    from alphalattice.control.task_control.child import ChildCalls
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_current_index_acquisition_manifest,
    )

    workspace = tmp_path / "new-member"
    shutil.copytree(qualified, workspace)
    binding = bind_existing_data_workspace(workspace)
    symbols = (*SYMBOLS, "NEW")
    candidate = _bootstrap(symbols)
    tomorrow = NOW + timedelta(days=1)
    clock_now = NOW
    provider = recording_provider(symbols=symbols, now=tomorrow + timedelta(days=1))
    provider.sectors["NEW"] = "Sector-0"
    entrant = build_current_index_acquisition_manifest(
        ROOT / "config/market-profiles/us-current-index-research.yaml", candidate
    ).listing_for_symbol("NEW")
    assert entrant is not None
    initially_eligible = candidate_features_available is True
    if candidate_features_available == "recoverable_history":
        fetch = provider.fetch_daily

        def short_history(symbols, **kwargs):
            result = fetch(symbols, **kwargs)
            if "NEW" in result and clock_now == NOW:
                result["NEW"] = tuple(
                    row for row in result["NEW"] if row["session_date"] >= "2020-01-01"
                )
            return result

        monkeypatch.setattr(provider, "fetch_daily", short_history)
    elif candidate_features_available == "recoverable_sector":
        from alphalattice.foundation.market_data_ops.sources.providers import ProviderFetchError

        sector_fetch = provider.fetch_current_sector

        def missing_sector(*, provider_symbol):
            if provider_symbol == "NEW" and clock_now == NOW:
                raise ProviderFetchError(
                    "sector.missing_current_sector", "not observed yet", retryable=False
                )
            return sector_fetch(provider_symbol=provider_symbol)

        monkeypatch.setattr(provider, "fetch_current_sector", missing_sector)
    elif not initially_eligible:
        fetch = provider.fetch_daily
        flat_session = tuple(day for day in provider.sessions if day <= NOW.date())[-21]

        def with_zero_volume(symbols, **kwargs):
            result = fetch(symbols, **kwargs)
            if "NEW" in result:
                result["NEW"] = tuple(
                    {**row, "volume": 0}
                    if candidate_features_available is False
                    else {**row, "open": row["close"], "high": row["close"], "low": row["close"]}
                    if date.fromisoformat(row["session_date"]) == flat_session
                    else row
                    for row in result["NEW"]
                )
            return result

        monkeypatch.setattr(provider, "fetch_daily", with_zero_volume)
    market = MarketDataRepository(workspace)
    calls = []
    made = ChildCalls.make

    # A listing's Feature computation, as the build hands it to a Host worker (W10), with the
    # sessions of the input frame the worker projects from the stored inputs (V92).
    def counted(self, arguments):
        if "source" in arguments:
            calls.append(
                (
                    arguments["listing_id"],
                    market.membership_events("us-current-index-research"),
                    len(arguments["source_sessions"])
                    if arguments["source_sessions"] is not None
                    else arguments["source"].bars.num_rows,
                )
            )
        return made(self, arguments)

    monkeypatch.setattr(ChildCalls, "make", counted)
    with LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=binding,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=lambda **_: candidate,
        clock=lambda: clock_now,
    ) as live:
        live.operations.data_update._readiness(market).refresh_sources_if_due(observed_at=NOW)
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert plan["status"] == "CONFIRMATION_REQUIRED" and plan["change"]["additions"] == ["NEW"]
        queued = _json(
            live,
            "/api/data-update/confirm",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        started = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        assert started["task_id"] == queued["task_id"]
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + queued["task_id"])
        assert status["lifecycle"] == "SUCCEEDED", status.get("latest_failure_code", status)
        current = market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        assert (current.listing_for_symbol("NEW") is not None) == initially_eligible
        events = market.membership_events("us-current-index-research")
        assert [(event.listing_id, event.kind) for event in events] == (
            [(entrant.listing_id, "ENTRY")] if initially_eligible else []
        )
        if events:
            assert events[0].authority == "qualified_source_membership"
            assert events[0].effective_session > NOW.date()  # First observed after close.
        own_calls = [item for item in calls if item[0] == entrant.listing_id]
        if candidate_features_available == "recoverable_history":
            assert own_calls == []
        else:
            assert len(own_calls) == 1 and not own_calls[0][1] and own_calls[0][2] > 2000
        snapshot = PanelStateRepository(
            market.database, market_data=market
        ).feature_panel_snapshot_for_active("us-current-index-research")
        from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver

        payload = ArtifactResolver(workspace / "artifacts").load_feature_panel_manifest(
            str(snapshot["manifest_uri"])
        )
        assert payload["safe_summary"]["membership"]["as_of_member_count"] == len(SYMBOLS)
        assert payload["active_listing_count"] == len(SYMBOLS)
        repeated = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        assert repeated["status"] == "REUSED_EXACT" and repeated["task_id"] is None
        if candidate_features_available in {
            "recoverable_range",
            "recoverable_sector",
            "recoverable_history",
        }:
            clock_now = tomorrow
            prices_before = len(provider.calls)
            recheck = _json(live, "/api/data-update/plan", method="POST", payload={})
            assert recheck["status"] == "PLANNED", recheck
            scope_key = (
                "candidate_data_recheck"
                if candidate_features_available == "recoverable_history"
                else "candidate_recheck"
            )
            assert recheck[scope_key]["listing_ids"] == [entrant.listing_id]
            rerun = _json(
                live,
                "/api/data-update/run",
                method="POST",
                payload={"update_plan_hash": recheck["plan_hash"]},
            )
            live.dispatcher.drain_for_tests(timeout=300)
            result = _json(live, "/api/status?task_id=" + rerun["task_id"])
            assert result["lifecycle"] == "SUCCEEDED", result
            events = market.membership_events("us-current-index-research")
            assert [(event.listing_id, event.kind) for event in events] == [
                (entrant.listing_id, "ENTRY")
            ]
            assert events[0].effective_session > tomorrow.date()
            assert all(start > HISTORY_START for _, start, _ in provider.calls[prices_before:])
            if candidate_features_available == "recoverable_history":
                assert all(
                    symbols == ("NEW",)
                    for symbols, start, _ in provider.calls[prices_before:]
                    if start < date(2026, 1, 1)
                )
                raw_scope = market.latest_current_universe_onboarding_disclosure(
                    market_profile_id="us-current-index-research"
                )
                assert raw_scope["candidate_listing_count"] == 1
                assert raw_scope["state_counts"] == {"FEATURE_READY": 1}
            assert sum(item[0] == entrant.listing_id and item[2] > 2000 for item in calls) == 1
            assert not [
                q
                for q in PanelStateRepository(
                    market.database, market_data=market
                ).active_listing_quarantines(
                    current.revision_sha256,
                    include_profile_history=True,
                    qualification_domain=(
                        "SECTOR_REFERENCE"
                        if candidate_features_available == "recoverable_sector"
                        else "BASE_FEATURES"
                    ),
                )
                if q.listing_id == entrant.listing_id
            ]
        if initially_eligible or candidate_features_available in {
            "recoverable_range",
            "recoverable_sector",
            "recoverable_history",
        }:
            # The already-approved entrant joins on its effective day, without
            # another membership confirmation or another whole-history backfill.
            clock_now = tomorrow if initially_eligible else tomorrow + timedelta(days=1)
            next_plan = _json(live, "/api/data-update/plan", method="POST", payload={})
            assert next_plan["status"] != "CONFIRMATION_REQUIRED", next_plan
            next_run = _json(
                live,
                "/api/data-update/run",
                method="POST",
                payload={"update_plan_hash": next_plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests(timeout=300)
            next_status = _json(live, "/api/status?task_id=" + next_run["task_id"])
            assert next_status["lifecycle"] == "SUCCEEDED", next_status.get(
                "latest_failure_code", next_status
            )
            next_snapshot = PanelStateRepository(
                market.database, market_data=market
            ).feature_panel_snapshot_for_active("us-current-index-research")
            next_payload = ArtifactResolver(workspace / "artifacts").load_feature_panel_manifest(
                str(next_snapshot["manifest_uri"])
            )
            assert next_payload["safe_summary"]["membership"]["as_of_member_count"] == len(symbols)
            assert next_payload["chunks"][:-1] == payload["chunks"][:-1]
            assert market.membership_events("us-current-index-research") == events
            # Only one preparation consumed the entrant's full history.
            assert sum(item[0] == entrant.listing_id and item[2] > 2000 for item in calls) == 1


def test_a_stale_member_on_the_entrant_recheck_day_is_governed_before_any_feature_work(
    qualified, tmp_path, monkeypatch
):
    """requirement: bad data stops the day in governance, not after the Feature build.

    Day one admits an entrant whose Sector is not observed yet (a Sector
    quarantine, the other ten publish). Day two's plan rechecks the entrant
    against the parent, which has no Sector evidence of its own; one prior
    member's refresh comes back stale. The cycle used to build Features over
    every listing and only then stop by ``data.listing_updates_incomplete``.
    Now the existing governance runs first over the prior manifest's Sector
    evidence: the stale member gets its case before any Feature work, with
    the wait option and no exclusion whose Panel impact cannot be projected;
    the day-one Panel stays the sealed input; the confirmed wait defers the
    same Task with its retry time; after the Provider recovers the resumed
    cycle refreshes only the stale member, builds, and admits the entrant.
    """

    from alphalattice.control.task_control.child import ChildCalls
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_current_index_acquisition_manifest,
    )
    from alphalattice.foundation.market_data_ops.sources.providers import ProviderFetchError

    workspace = tmp_path / "entrant-stale-member"
    shutil.copytree(qualified, workspace)
    binding = bind_existing_data_workspace(workspace)
    symbols = (*SYMBOLS, "NEW")
    candidate = _bootstrap(symbols)
    tomorrow = NOW + timedelta(days=1)
    clock_now = NOW
    provider = recording_provider(symbols=symbols, now=tomorrow + timedelta(days=1))
    provider.sectors["NEW"] = "Sector-0"
    entrant = build_current_index_acquisition_manifest(
        ROOT / "config/market-profiles/us-current-index-research.yaml", candidate
    ).listing_for_symbol("NEW")
    assert entrant is not None
    sector_fetch = provider.fetch_current_sector

    def missing_sector(*, provider_symbol):
        if provider_symbol == "NEW" and clock_now == NOW:
            raise ProviderFetchError(
                "sector.missing_current_sector", "not observed yet", retryable=False
            )
        return sector_fetch(provider_symbol=provider_symbol)

    monkeypatch.setattr(provider, "fetch_current_sector", missing_sector)
    stale_symbol = SYMBOLS[1]
    stale = {"on": False}
    fetch = provider.fetch_daily

    def stale_member(symbols, **kwargs):
        result = fetch(symbols, **kwargs)
        if stale["on"] and stale_symbol in result:
            result[stale_symbol] = [
                row for row in result[stale_symbol] if row["session_date"] <= NOW.date().isoformat()
            ]
        return result

    monkeypatch.setattr(provider, "fetch_daily", stale_member)
    market = MarketDataRepository(workspace)
    panel_state = PanelStateRepository(market.database, market_data=market)
    built: list[str] = []
    made = ChildCalls.make

    # A listing's Feature computation, as the build hands it to a Host worker (W10).
    def counted(self, arguments):
        if "source" in arguments:
            built.append(arguments["listing_id"])
        return made(self, arguments)

    monkeypatch.setattr(ChildCalls, "make", counted)
    with LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=binding,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=lambda **_: candidate,
        clock=lambda: clock_now,
    ) as live:
        live.operations.data_update._readiness(market).refresh_sources_if_due(observed_at=NOW)
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert plan["status"] == "CONFIRMATION_REQUIRED" and plan["change"]["additions"] == ["NEW"]
        _json(
            live,
            "/api/data-update/confirm",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        started = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + started["task_id"])
        assert status["lifecycle"] == "SUCCEEDED", status.get("latest_failure_code", status)
        day_one = panel_state.feature_panel_snapshot_for_active("us-current-index-research")
        assert day_one is not None
        prior = market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        assert prior.listing_for_symbol("NEW") is None
        stale_listing = prior.listing_for_symbol(stale_symbol)
        assert stale_listing is not None
        built_day_one = len(built)

        # Day two: the entrant's recheck rides on the parent; one member is stale.
        clock_now = tomorrow
        stale["on"] = True
        recheck = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert recheck["status"] == "PLANNED", recheck
        assert recheck["candidate_recheck"]["listing_ids"] == [entrant.listing_id]
        request = {"update_plan_hash": recheck["plan_hash"]}
        rerun = _json(live, "/api/data-update/run", method="POST", payload=request)
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + rerun["task_id"])
        assert status["lifecycle"] == "BLOCKED", status.get("worker_failure") or status
        assert status["latest_failure_code"] == "data.truth_review_required", (
            status["latest_failure_code"],
            f"feature units built before the stop: {len(built) - built_day_one}",
        )
        assert len(built) == built_day_one, "Feature work started before the data decision"
        issues = _json(live, "/api/workspace/data-issues")
        (issue,) = [item for item in issues["issues"] if item["status"] == "AWAITING_CHOICE"]
        case = issue["case"]
        # Every case named in a line beside the whole page, and the pending view offered
        # first; one page holds them all, so none follows (V236, V237, V172).
        assert issues["case_count"] == len(issues["case_index"]) == len(issues["issues"])
        assert issues["case_index"][0]["case_token"] == case["case_token"]
        assert issues["next_requests"]["pending"] == {"operation": "PENDING_DECISIONS"}
        assert "next_page" not in issues["next_requests"]
        assert case["listing_ids"] == [stale_listing.listing_id]
        assert case["failure_code"] == "data.maintenance_stale_payload"
        assert issue["subjects"] == {stale_listing.listing_id: stale_symbol}
        offered = {option["option_id"] for option in case["options"]}
        assert {"bounded_full_history_retry", "wait_for_provider_recovery"} <= offered
        assert not offered & {"recoverable_quarantine", "exclude_from_next_manifest"}
        assert issue["confirmable_option_ids"] == ["wait_for_provider_recovery"]
        (continuation,) = issues["continuations"]
        assert continuation["task_id"] == rerun["task_id"]
        assert continuation["failure_reason"]["code"] == "data.truth_review_required"
        # The sealed input is still day one's Panel.
        readback = _json(live, "/api/data-update")
        assert readback["inputs"]["panel_through"] == NOW.date().isoformat()
        assert (
            panel_state.feature_panel_snapshot_for_active("us-current-index-research")[
                "manifest_uri"
            ]
            == day_one["manifest_uri"]
        )
        # The Human chooses to wait; the same Task continues and defers with
        # its retry time, still without Feature work or a new Panel.
        option = next(v for v in case["options"] if v["option_id"] == "wait_for_provider_recovery")
        decision = _json(
            live,
            "/api/workspace/data-issues/confirm",
            method="POST",
            payload={
                "data_issue_case_token": case["case_token"],
                "data_issue_evidence_hash": case["evidence_hash"],
                "data_issue_option_id": option["option_id"],
                "data_issue_option_hash": option["option_hash"],
            },
        )
        assert decision["status"] == "CONFIRMED_PENDING_REVALIDATION"
        resumed = _json(live, "/api/data-update/run", method="POST", payload=request)
        assert resumed["task_id"] == rerun["task_id"], resumed
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + rerun["task_id"])
        assert status["lifecycle"] == "DEFERRED", status
        assert status["latest_failure_code"] == "data.remediation_wait", status
        waiting = _json(live, "/api/data-update")
        assert waiting["retry_after_at"] is not None
        # V375: the deferred update says why, that the published inputs stand, and which
        # request resumes it: the same plan again once its time has passed.
        assert waiting["next_requests"]["resume"] == {
            "operation": "DATA_UPDATE_RUN",
            "update_plan_hash": recheck["plan_hash"],
        }
        assert "inputs are unchanged" in waiting["detail"], waiting["detail"]
        retry_after = datetime.fromisoformat(waiting["retry_after_at"])
        assert retry_after > tomorrow
        assert waiting["inputs"]["panel_through"] == NOW.date().isoformat()
        assert len(built) == built_day_one
        assert market.membership_events("us-current-index-research") == ()
        waiting_issue = _json(live, "/api/workspace/data-issues")["issues"][0]
        assert waiting_issue["status"] == "WAITING_FOR_RETRY"
        # Not due yet: the Task is not resumed early.
        early = _json(live, "/api/data-update/run", method="POST", payload=request)
        assert early["status"] == "REFUSED_INVALID_COMMAND", early
        assert early["failure_code"] == "workspace_data_update.retry_not_due"
        assert "`retry_after_at`" in early["detail"], early

        # The Provider recovers; at the retry time the same Task resumes,
        # refreshes only the stale member, builds, and admits the entrant.
        stale["on"] = False
        clock_now = retry_after + timedelta(minutes=1)
        fetches_before = len(provider.calls)
        again = _json(live, "/api/data-update/run", method="POST", payload=request)
        assert again["task_id"] == rerun["task_id"]
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + rerun["task_id"])
        assert status["lifecycle"] == "SUCCEEDED", (
            status.get("worker_failure"),
            status.get("lifecycle"),
            status.get("latest_failure_code"),
            _json(live, "/api/workspace/data-issues")["issues"],
        )
        member_fetches = [
            symbols_fetched
            for symbols_fetched, _start, _end in provider.calls[fetches_before:]
            if set(symbols_fetched) & set(SYMBOLS)
        ]
        # Persisted progress is reused: only the stale member is refreshed
        # (its bars and its adjusted-close audit), no other member.
        assert member_fetches and all(item == (stale_symbol,) for item in member_fetches), (
            member_fetches
        )
        assert len(built) > built_day_one
        events = market.membership_events("us-current-index-research")
        assert [(event.listing_id, event.kind) for event in events] == [
            (entrant.listing_id, "ENTRY")
        ]
        assert events[0].effective_session > tomorrow.date()
        published = panel_state.feature_panel_snapshot_for_active("us-current-index-research")
        assert published["manifest_uri"] != day_one["manifest_uri"]
        assert _json(live, "/api/data-update")["inputs"]["panel_through"] == (
            tomorrow.date().isoformat()
        )
        assert _json(live, "/api/workspace/data-issues")["issues"] == []


@pytest.mark.parametrize("raw_admits", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_due_raw_and_feature_rechecks_are_both_planned_and_executed(
    qualified, tmp_path, monkeypatch, raw_admits, interrupted
):
    """regression: a due raw retry hid a due Feature/Sector recheck.

    Two entrants: one whose history is too short (a failed raw candidate,
    retried a day later) and one whose provider reports no current Sector (a
    Feature/Sector quarantine, rechecked a day later). The day both fall due
    the plan carried only the raw retry and the run executed only that, so
    the quarantined name was never re-diagnosed while the raw retry stayed
    due -- every day, on a workspace with permanently short candidates. The
    plan now carries both scopes, the run executes the Feature/Sector recheck
    whatever the raw retry admitted (against the parent as it stands: the
    merged one when the retry admitted, the planned one when it did not), and
    a Task interrupted between the two still owes the recheck when it resumes.
    """

    from alphalattice.control.data_platform.maintenance.coordinator import (
        WorkspaceMaintenanceCoordinator,
    )
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_current_index_acquisition_manifest,
    )
    from alphalattice.foundation.market_data_ops.sources.providers import ProviderFetchError

    workspace = tmp_path / "two-rechecks"
    shutil.copytree(qualified, workspace)
    binding = bind_existing_data_workspace(workspace)
    symbols = (*SYMBOLS, "NEWH", "NEWS")
    candidate = _bootstrap(symbols)
    tomorrow = NOW + timedelta(days=1)
    clock_now = NOW
    provider = recording_provider(symbols=symbols, now=tomorrow + timedelta(days=1))
    provider.sectors["NEWH"] = "Sector-0"
    provider.sectors["NEWS"] = "Sector-1"
    acquisition = build_current_index_acquisition_manifest(
        ROOT / "config/market-profiles/us-current-index-research.yaml", candidate
    )
    short = acquisition.listing_for_symbol("NEWH")
    sectorless = acquisition.listing_for_symbol("NEWS")
    assert short is not None and sectorless is not None
    fetch = provider.fetch_daily

    def short_history(symbols, **kwargs):
        result = fetch(symbols, **kwargs)
        if "NEWH" in result and (clock_now == NOW or not raw_admits):
            result["NEWH"] = tuple(
                row for row in result["NEWH"] if row["session_date"] >= "2020-01-01"
            )
        return result

    monkeypatch.setattr(provider, "fetch_daily", short_history)
    sector_fetch = provider.fetch_current_sector

    def missing_sector(*, provider_symbol):
        if provider_symbol == "NEWS" and clock_now == NOW:
            raise ProviderFetchError(
                "sector.missing_current_sector", "not observed yet", retryable=False
            )
        return sector_fetch(provider_symbol=provider_symbol)

    monkeypatch.setattr(provider, "fetch_current_sector", missing_sector)
    market = MarketDataRepository(workspace)

    def session():
        return LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=binding,
            resolver=_Resolver(_resolved()),
            data_provider=provider,
            data_source_loader=lambda **_: candidate,
            clock=lambda: clock_now,
        )

    with session() as live:
        live.operations.data_update._readiness(market).refresh_sources_if_due(observed_at=NOW)
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert plan["status"] == "CONFIRMATION_REQUIRED"
        assert plan["change"]["additions"] == ["NEWH", "NEWS"]
        _json(
            live,
            "/api/data-update/confirm",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        started = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + started["task_id"])
        assert status["lifecycle"] == "SUCCEEDED", status.get("latest_failure_code", status)
        current = market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        assert current.listing_for_symbol("NEWH") is None
        assert current.listing_for_symbol("NEWS") is None
        assert market.membership_events("us-current-index-research") == ()

        # The next day both rechecks are due, and the plan says so.
        clock_now = tomorrow
        recheck = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert recheck["status"] == "PLANNED", recheck
        assert recheck["candidate_data_recheck"]["listing_ids"] == [short.listing_id]
        assert recheck["candidate_recheck"] is not None, recheck
        assert recheck["candidate_recheck"]["listing_ids"] == [sectorless.listing_id]
        planned_parent = recheck["candidate_recheck"]["parent_manifest_revision"]
        parent = market.source_admission_manifest(market_profile_id="us-current-index-research")
        assert planned_parent == parent.revision_sha256 != current.revision_sha256
        assert parent.listing_for_symbol("NEWS") is not None
        prices_before = len(provider.calls)
        bound: list[str] = []
        if interrupted:
            # The process dies once the rechecks are resolved and the working
            # manifest is bound, before the first cycle's own work.
            run_cycle = WorkspaceMaintenanceCoordinator._run_cycle

            def crash_once(self, cycle, request, now, **kwargs):
                if not bound:
                    bound.append(self.manifest.revision_sha256)
                    raise RuntimeError("simulated process death after the rechecks")
                return run_cycle(self, cycle, request, now, **kwargs)

            monkeypatch.setattr(WorkspaceMaintenanceCoordinator, "_run_cycle", crash_once)
        rerun = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": recheck["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=300)
        result = _json(live, "/api/status?task_id=" + rerun["task_id"])
        if interrupted:
            assert result["lifecycle"] == "RECOVERY_REQUIRED", result
            assert result["latest_failure_code"] == "TASK_EXECUTION_INTERRUPTED"
            # The parent was bound as the working manifest before the death:
            # the planned one when the raw retry admitted nothing, the merged
            # one when it admitted the short candidate.
            assert bound and bound[0] != current.revision_sha256
            assert (bound[0] == planned_parent) is not raw_admits
    if interrupted:
        # A restart resumes the same Task; the rechecks are re-resolved from
        # the stored request, not from any memory the dead process held.
        with session() as revived:
            assert [str(item) for item in revived.resumed_task_ids] == [rerun["task_id"]]
            revived.dispatcher.drain_for_tests(timeout=300)
            result = _json(revived, "/api/status?task_id=" + rerun["task_id"])
    assert result["lifecycle"] == "SUCCEEDED", result.get("latest_failure_code", result)
    current = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    events = market.membership_events("us-current-index-research")
    entered = sorted(event.listing_id for event in events)
    assert all(event.kind == "ENTRY" for event in events)
    assert all(event.effective_session > tomorrow.date() for event in events)
    # The quarantined name was re-diagnosed and admitted whatever the raw
    # retry did; the short candidate joins only when its history arrived.
    assert current.listing_for_symbol("NEWS") is not None
    assert (current.listing_for_symbol("NEWH") is not None) == raw_admits
    assert entered == sorted([sectorless.listing_id, *([short.listing_id] if raw_admits else [])])
    raw_scope = market.latest_current_universe_onboarding_disclosure(
        market_profile_id="us-current-index-research"
    )
    assert raw_scope["candidate_listing_count"] == 1
    assert raw_scope["state_counts"] == (
        {"FEATURE_READY": 1} if raw_admits else {"QUALITY_INELIGIBLE": 1}
    )
    # Nothing before the ten-year window was fetched for anyone.
    assert all(start > HISTORY_START for _, start, _ in provider.calls[prices_before:])
