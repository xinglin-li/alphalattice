"""A stock-list change that stops part-way resumes through its own cycle."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.data_platform.readiness import build_workspace_readiness
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.maintenance.data_update import bind_existing_data_workspace
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as Request,
)
from tests.portfolio_strategy_lab.local_web_support import (
    InstalledAgent,
    _json,
    _resolved,
    _Resolver,
)
from tests.researcher_methodology_surface.real_workspace import SYMBOLS
from tests.researcher_methodology_surface.session_workspace import copy_workspace
from tests.workspace_maintenance.data_update_support import one_sector_seed
from tests.workspace_maintenance.local_data_provider import (
    NOW,
    recording_provider,
    unchanged_membership_source,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "us-current-index-research"
MEMBERS = SYMBOLS[:7]
"""One Sector of seven: after a leaver and a sectorless joiner it keeps the five-member floor."""
CHANGED_AT = NOW + timedelta(days=1)


def _host(workspace: Path, binding, provider, source) -> LocalPortfolioWebSession:  # type: ignore[no-untyped-def]
    return LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=binding,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=source,
        clock=lambda: CHANGED_AT,
    )


def _agent(live: LocalPortfolioWebSession, document: dict) -> dict:  # type: ignore[type-arg]
    return json.loads(InstalledAgent(live.operations).invoke(Request(**document)))


def _registry(live: LocalPortfolioWebSession) -> DuckDbWorkspaceMaintenanceRegistry:
    return DuckDbWorkspaceMaintenanceRegistry(
        live.workspace / "market-data.duckdb", gate=live.session.mutation_gate
    )


@pytest.fixture(scope="module")
def stopped_change(tmp_path_factory) -> SimpleNamespace:  # type: ignore[no-untyped-def]
    """A stock-list change with two short-tailed leavers, a sectorless joiner and an owed audit,
    stopped on the Panel's coverage floor after its cycle made its own Sector-reduced membership
    active, with every step's answer kept."""
    seed = one_sector_seed(tmp_path_factory, "transition", MEMBERS)
    workspace = copy_workspace(seed, tmp_path_factory.mktemp("stopped") / "workspace")
    binding = bind_existing_data_workspace(workspace)
    market = MarketDataRepository(workspace)
    prior = market.current_quality_filtered_research_manifest(market_profile_id=PROFILE)
    leaver, audited = prior.listing_for_symbol(MEMBERS[0]), prior.listing_for_symbol(MEMBERS[1])
    short = prior.listing_for_symbol(MEMBERS[2])
    provider = recording_provider(symbols=(*MEMBERS, "NEW"), sector_size=7, now=CHANGED_AT)
    del provider.sectors["NEW"]
    raw = market.raw_bars(audited.listing_id)
    # A restated bar inside the rolling audit's reach owes a full-history audit.
    restated = next(
        bar.session_date
        for bar in raw
        if raw[-1].session_date - timedelta(days=45)
        <= bar.session_date
        < CHANGED_AT.date() - timedelta(days=45)
    )
    fetch = provider.fetch_daily

    def source_rows(asked, **kwargs):  # type: ignore[no-untyped-def]
        rows = fetch(asked, **kwargs)
        for row in rows.get(audited.symbol, ()):
            if date.fromisoformat(str(row["session_date"])) == restated:
                row["volume"] += 1
        moved = market.readiness.load(PROFILE).active_manifest_revision != prior.revision_sha256
        if moved and rows.get(leaver.symbol):
            last = rows[leaver.symbol][-1]
            last["open"] = last["low"] - 0.0288
        if moved and rows.get(short.symbol):
            rows[short.symbol] = rows[short.symbol][:-1]  # the source stops a session short
        return rows

    provider.fetch_daily = source_rows  # type: ignore[method-assign]
    source = unchanged_membership_source(tuple(sorted((MEMBERS[1], *MEMBERS[3:], "NEW"))))
    build_workspace_readiness(
        market,
        profile_path=ROOT / "config/market-profiles/us-current-index-research.yaml",
        provider=provider,
        source_loader=source,
    ).refresh_sources_if_due(observed_at=CHANGED_AT)
    seen: dict[str, object] = {}
    with _host(workspace, binding, provider, source) as live:
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        confirm = {"update_plan_hash": plan["plan_hash"]}
        task_id = _json(live, "/api/data-update/confirm", method="POST", payload=confirm)["task_id"]
        _json(live, "/api/data-update/run", method="POST", payload=confirm)
        live.dispatcher.drain_for_tests(timeout=300)
        seen["first_stop"] = _json(live, f"/api/status?task_id={task_id}")
        seen["first_active"] = market.readiness.load(PROFILE).active_manifest_revision
        seen["audit"] = audit = _agent(live, {"operation": "DATA_UPDATE_PLAN"})
        seen["admitted"] = admitted = _agent(live, audit["next_requests"]["confirm"])
        _agent(live, admitted["next_requests"]["run"])
        live.dispatcher.drain_for_tests(timeout=300)
        seen["second_stop"] = _json(live, f"/api/status?task_id={task_id}")
        seen["readback"] = _agent(live, {"operation": "DATA_UPDATE_READBACK", "task_id": task_id})
        cycle = _registry(live).latest_cycle(PROFILE).cycle_id  # type: ignore[union-attr]
        seen["working"] = _registry(live).working_manifest(cycle)
    return SimpleNamespace(
        workspace=workspace,
        binding=binding,
        provider=provider,
        source=source,
        task_id=task_id,
        cycle_id=cycle,
        journal=market.manifest_transition(plan["change"]["transition_id"]),
        prior=prior,
        audited=audited,
        active=market.readiness.load(PROFILE).active_manifest_revision,
        seen=seen,
    )


def test_a_stopped_membership_change_resumes_through_the_membership_its_cycle_made(
    stopped_change, tmp_path
):
    """regression (a stopped stock-list change): a change stopped after its cycle made a derived
    membership active resumes through its own cycle's record across a Host restart, as the same
    Task, with the audit it owed admitted onto that Task."""
    seen = stopped_change.seen
    task_id, journal = stopped_change.task_id, stopped_change.journal
    first = seen["first_stop"]
    assert (first["lifecycle"], first["latest_failure_code"]) == (
        "BLOCKED",
        "data.full_history_audit_approval_required",
    )
    assert seen["first_active"] == journal.next_manifest_revision
    audit = seen["audit"]
    assert audit["change"]["action"] == "FULL_HISTORY_AUDIT"
    assert audit["change"]["full_history_listing_ids"] == [stopped_change.audited.listing_id]
    admitted = seen["admitted"]
    assert (admitted["status"], admitted["task_id"]) == ("APPROVED", task_id)
    second = seen["second_stop"]
    assert second["latest_failure_code"] == "feature.panel_coverage_not_ready", second
    # The floor is missed by the leavers' disclosed tails: AAA's rejected last row and CCC's
    # source that stopped short, which admitted nothing and no longer stops the update.
    tails = seen["readback"]["removed_member_tails"]
    assert sorted((t["symbol"], t["sanitizer_code"]) for t in tails) == [
        (MEMBERS[0], "INVALID_OHLC"),
        (MEMBERS[2], "STALE_PAYLOAD"),
    ], tails
    assert all(t["missing_sessions"] for t in tails), tails
    # The half-switched state: the cycle's own Sector-reduced membership is active, the
    # journal still names its root, and the cycle's record names the active one.
    assert stopped_change.active != journal.next_manifest_revision
    assert seen["working"] == stopped_change.active

    workspace = copy_workspace(stopped_change.workspace, tmp_path / "workspace")
    with _host(
        workspace, stopped_change.binding, stopped_change.provider, stopped_change.source
    ) as live:
        again = _agent(live, {"operation": "DATA_UPDATE_PLAN"})
        assert again["status"] == "PLANNED" and again["change"]["action"] == "UNIVERSE", again
        assert _agent(live, again["next_requests"]["run"])["task_id"] == task_id
        live.dispatcher.drain_for_tests(timeout=300)
        resumed = _json(live, f"/api/status?task_id={task_id}")
        tasks = live.session.task_control_registry.tasks()
    # The same Task ran the same cycle to the same coverage floor; the audit was not owed again.
    assert resumed["latest_failure_code"] == "feature.panel_coverage_not_ready", resumed
    assert [t.task_id for t in tasks if t.task_kind == "workspace_data_update"] == [UUID(task_id)]


def test_a_cycle_record_naming_another_membership_is_refused_by_name(stopped_change, tmp_path):
    """tamper: recovery accepts exactly the membership the stopped change's own cycle made
    active. A cycle record naming any other is refused by name, and nothing runs."""
    workspace = copy_workspace(stopped_change.workspace, tmp_path / "workspace")
    with _host(
        workspace, stopped_change.binding, stopped_change.provider, stopped_change.source
    ) as live:
        _registry(live).record_working_manifest(
            stopped_change.cycle_id,
            manifest_revision=stopped_change.prior.revision_sha256,
            authority="tamper",
            observed_at=CHANGED_AT,
        )
        calls = len(stopped_change.provider.calls)
        refused = _agent(live, {"operation": "DATA_UPDATE_PLAN"})
        task = live.session.task_control_registry.task(UUID(stopped_change.task_id))
    assert refused["failure_code"] == "workspace_data_update.transition_not_verified", refused
    assert len(stopped_change.provider.calls) == calls
    assert task.lifecycle.value == "BLOCKED"


def test_a_change_stopped_before_the_cycle_record_is_adopted_through_its_admission(
    stopped_change, tmp_path
):
    """compatibility (retire in 0.2): a change stopped part-way by 0.1.3, which kept no cycle
    record, resumes as the same Task, its membership adopted only when the run has verified it
    through the gateway's admission."""
    import duckdb

    workspace = copy_workspace(stopped_change.workspace, tmp_path / "workspace")
    events = "SELECT kind, details_json FROM workspace_maintenance_event WHERE cycle_id = ?"
    connection = duckdb.connect(str(workspace / "market-data.duckdb"))
    try:
        # A 0.1.3 cycle wrote no working-membership record.
        connection.execute(
            "DELETE FROM workspace_maintenance_event WHERE kind = ?",
            ["workspace_maintenance.working_manifest"],
        )
    finally:
        connection.close()
    with _host(
        workspace, stopped_change.binding, stopped_change.provider, stopped_change.source
    ) as live:
        again = _agent(live, {"operation": "DATA_UPDATE_PLAN"})
        assert again["status"] == "PLANNED", again
        # Planning reads; only the run records the adoption, once it has verified.
        assert _registry(live).working_manifest(stopped_change.cycle_id) is None
        assert _agent(live, again["next_requests"]["run"])["task_id"] == stopped_change.task_id
        live.dispatcher.drain_for_tests(timeout=300)
        resumed = _json(live, f"/api/status?task_id={stopped_change.task_id}")
        assert _registry(live).working_manifest(stopped_change.cycle_id) == stopped_change.active
    assert resumed["latest_failure_code"] == "feature.panel_coverage_not_ready", resumed
    connection = duckdb.connect(str(workspace / "market-data.duckdb"), read_only=True)
    try:
        kept = connection.execute(events, [stopped_change.cycle_id]).fetchall()
    finally:
        connection.close()
    adopted = [
        json.loads(details)
        for kind, details in kept
        if kind == "workspace_maintenance.working_manifest"
    ]
    assert [(row["manifest_revision"], row["authority"][:8]) for row in adopted] == [
        (stopped_change.active, "adopted:")
    ]
