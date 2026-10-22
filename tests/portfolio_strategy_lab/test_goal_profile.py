"""The Host owns complete profile reads; the CLI bounds only their display."""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.publication.goals import session_work_declaration
from alphalattice.control.task_control.runner import TaskHeartbeatReader
from alphalattice.control.task_control.timing import record_stage_spans, stage_spans_path
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
)
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.kernel.shared_kernel.clocks import ReplayClock
from tests.portfolio_strategy_lab.cli_support import _cli
from tests.workspace_task_runner.task_control_support import task_contract

NOW = datetime.fromisoformat("2026-09-15T00:00:00+00:00")


def _profile_scene(live, monkeypatch, kind, namespace, stages=1, close=True, existing_goal=None):
    clock = live.clock
    correlation = clock.correlation if isinstance(clock.correlation, dict) else {}
    start = datetime.fromisoformat(correlation["replay_anchor"]) if correlation else NOW
    phase_start = datetime.fromisoformat(correlation["system_anchor"]) if correlation else start
    clock.return_value = start
    runtime = live.workspace / "runtime" / namespace / "checkpoints.sqlite"
    monkeypatch.setattr(live.operations, "heartbeats", TaskHeartbeatReader(runtime))
    client = LocalResearchClient(live.workspace)
    goal = (
        existing_goal
        or client.request(
            {
                "operation": "GOAL_OPEN",
                "goal_declaration": session_work_declaration("PLAN", "codex")
                | dict(kind=kind, objective=namespace),
                "change_reason": "Register fixture",
            }
        )["goal_id"]
    )
    registry = live.session.task_control_registry
    envelope, research, plan = task_contract(salt=namespace)
    task = registry.admit(
        input_envelope=envelope, goal=research, plan=plan, observed_at=start
    ).record
    token = REQUEST_PROVENANCE.set(RequestProvenance(goal_id=goal))
    try:
        span = live.activity.entered(
            PortfolioResearchOperationRequest(
                operation="EXPERIMENT_RUN", experiment_plan_hash="a" * 64
            ),
            caller="HUMAN",
        )
        clock.return_value = start + timedelta(seconds=5)
        live.activity.returned(span, {"status": "DONE", "task_id": str(task.task_id)})
    finally:
        REQUEST_PROVENANCE.reset(token)
    for stage in range(stages):
        execution = uuid4()
        for phase, t, wall in (("execute", 2, 5), ("verify", 8, 2)):
            record_stage_spans(
                runtime,
                task_id=task.task_id,
                execution_id=execution,
                stage_id=f"stage-{stage}",
                phase=phase,
                clock=clock,
                readout={
                    "started_at": (phase_start + timedelta(seconds=t)).isoformat(),
                    "wall_seconds": wall,
                },
            )
    clock.return_value = start + timedelta(seconds=12)
    if close:
        client.request(
            {"operation": "GOAL_ABANDON", "goal_id": goal, "change_reason": "End fixture"}
        )
    return client, goal, registry, runtime


def _correlated_clock(anchor, system):
    clock = Mock(return_value=anchor)

    def elapsed():
        return (clock.return_value - anchor).total_seconds()

    replay = ReplayClock(anchor, ticks=elapsed, wall=lambda: system + timedelta(seconds=elapsed()))
    clock.reference, clock.correlation = replay.reference, replay.correlation
    clock.phase_metadata = replay.phase_metadata
    return clock


@pytest.mark.parametrize("fault", [None, "missing", "drift"])
@pytest.mark.parametrize("live", [_correlated_clock(NOW, NOW + timedelta(days=30))], indirect=True)
def test_profile_aligns_two_hosts_without_charging_the_scripted_date_jump(live, monkeypatch, fault):
    """Real served reads align active and closed windows; missing or drifting origins stay gaps."""
    system = NOW + timedelta(days=30)
    client, goal, _, _ = _profile_scene(live, monkeypatch, "OPERATIONS", "portfolio", close=False)
    active = client.request({"operation": "GOAL_PROFILE", "goal_id": goal})
    assert active["window"]["end"] == (system + timedelta(seconds=12)).isoformat()
    live.stop()
    clock = _correlated_clock(NOW + timedelta(days=7), system + timedelta(seconds=60))
    if fault == "missing":
        clock.reference = "missing-clock"
        clock.phase_metadata = lambda: {"clock_ref": clock.reference, "clock_drift": False}
    if fault == "drift":
        clock.phase_metadata = lambda: {"clock_ref": clock.reference, "clock_drift": True}
    second = LocalPortfolioWebSession(
        workspace=live.workspace,
        workspace_manifest=live.workspace_manifest,
        resolver=live.resolver,
        clock=clock,
    )
    second.start()
    try:
        client, _, _, _ = _profile_scene(
            second, monkeypatch, "OPERATIONS", "portfolio", existing_goal=goal
        )
        result = client.request({"operation": "GOAL_PROFILE", "goal_id": goal})
        assert result == client.request({"operation": "GOAL_PROFILE", "goal_id": goal})
        (live.workspace.parent / "clock-profile.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        if fault:
            assert f"CLOCK_MAPPING_{fault.upper()}" in {g["code"] for g in result["gaps"]}
            assert result["coverage"]["status"] == "PARTIAL"
        else:
            assert result["window"]["start"] == system.isoformat()
            assert result["window"]["end"] == (system + timedelta(seconds=72)).isoformat()
            assert result["coverage"]["host_seconds"] == 18
            assert result["coverage"]["phase_seconds"] == 14
            assert result["clock_domains"] == ["UTC"]
    finally:
        second.stop()


@pytest.mark.parametrize("kind,namespace", [("FIRST_USE", "research"), ("RESEARCH", "portfolio")])
@pytest.mark.parametrize("live", [Mock(return_value=NOW)], indirect=True)
def test_host_profile_scopes_unions_reads_once_and_writes_nothing(
    live, monkeypatch, kind, namespace
):
    """A served read scopes the ledger and phase owner once and preserves every stored input."""
    client, goal, registry, runtime = _profile_scene(live, monkeypatch, kind, namespace)
    _profile_scene(live, monkeypatch, "OPERATIONS", "foreign")
    monkeypatch.setattr(live.operations, "heartbeats", TaskHeartbeatReader(runtime, runtime))
    ledger = live.activity._ledger
    before_bytes = {p: p.read_bytes() for p in live.workspace.rglob("*.json*")}
    before = ledger.head_ordinal(), registry.tasks()
    calls, original = [], ledger.observations_after

    def page(cursor, *, limit):
        calls.append(cursor.ordinal)
        return original(cursor, limit=1)

    monkeypatch.setattr(ledger, "observations_after", page)
    from alphalattice.control.product_host.composition import workspace_activity

    reads, reader = [], workspace_activity.read_stage_spans

    def phases(path, task, **kwargs):
        reads.append((path, task))
        return reader(path, task, **kwargs)

    monkeypatch.setattr(workspace_activity, "read_stage_spans", phases)

    def read():
        return client.request({"operation": "GOAL_PROFILE", "goal_id": goal})

    answer = read()
    assert answer["status"] == "GOAL_PROFILE", answer
    assert {
        k: answer["coverage"][k]
        for k in ("host_seconds", "rpc_seconds", "phase_seconds", "unattributed_seconds")
    } == {"host_seconds": 9, "rpc_seconds": 5, "phase_seconds": 7, "unattributed_seconds": 3}
    assert len(reads) == 1 and reads[0][0] == runtime
    assert calls[0] == 0 and calls == sorted(set(calls))
    assert answer == read()
    assert before == (ledger.head_ordinal(), registry.tasks())
    assert before_bytes == {p: p.read_bytes() for p in live.workspace.rglob("*.json*")}
    assert "UNPAIRED_RPC" not in {gap["code"] for gap in answer["gaps"]}
    assert answer["cache_counts"]["status"] == "UNOBSERVED" and "critical_path" not in answer
    assert all(v["en"] != v["zh-CN"] for v in answer["labels"].values())
    assert not answer.keys() & {"ask_now", "person_questions"}


@pytest.mark.parametrize("live", [Mock(return_value=NOW)], indirect=True)
def test_active_profile_window_keeps_background_work_after_the_last_rpc(live, monkeypatch):
    """The read snapshot includes kept background phases after the last synchronous return."""
    _, goal, _, _ = _profile_scene(live, monkeypatch, "OPERATIONS", "portfolio", close=False)
    held = live.operations.goals.store.head(UUID(goal))
    reads = live.clock.call_count
    answer = live.activity.goal_profile(held, live.operations.heartbeats.runtime_paths)
    assert live.clock.call_count - reads == 1
    assert answer["window"]["end"] == (NOW + timedelta(seconds=12)).isoformat()
    assert answer["window"]["source"] == "HOST_READ_SNAPSHOT"
    assert answer["coverage"]["host_seconds"] == 9


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("fault", ["stages", "unreadable"])
@pytest.mark.parametrize("live", [Mock(return_value=NOW)], indirect=True)
def test_cli_profile_bounds_stdout_but_keeps_complete_output(
    live, monkeypatch, tmp_path, language, fault
):
    """Many stages survive the complete answer file even when stdout becomes a bounded pointer."""
    client, goal, registry, runtime = _profile_scene(
        live, monkeypatch, "OPERATIONS", "portfolio", stages=150
    )
    if fault == "unreadable":
        path = stage_spans_path(runtime, registry.tasks()[0].task_id)
        path.write_text("{\n" * 1000, encoding="utf-8")
    complete = client.request({"operation": "GOAL_PROFILE", "goal_id": goal})
    saved = tmp_path / "profile.json"
    code, body, stdout = _cli(
        live.workspace, "--lang", language, "goal", "profile", goal, "--output", str(saved)
    )
    assert code == 0, body
    output = json.loads(saved.read_text(encoding="utf-8"))
    for field in ("stages", "gaps", "coverage", "labels", "stdout_hint"):
        assert output[field] == complete[field]
    assert len(output["stages"]) == (150 if fault == "stages" else 0)
    assert len(stdout.encode("utf-8")) <= 32768 < len(saved.read_bytes())
    locale = "zh-CN" if language == "zh" else "en"
    assert body["full_response"] == complete["stdout_hint"][locale]
    (tmp_path / "profile-receipt.json").write_text(
        json.dumps(
            {
                "language": language,
                "fault": fault,
                "stdout_bytes": len(stdout.encode("utf-8")),
                "saved_bytes": saved.stat().st_size,
                "stages": len(output["stages"]),
                "gaps": len(output["gaps"]),
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("live", [Mock(return_value=NOW)], indirect=True)
def test_terminal_profile_window_ignores_later_refused_traffic(live, monkeypatch):
    """A refused request after closure cannot extend the terminal revision's window."""
    client, goal, _, _ = _profile_scene(live, monkeypatch, "OPERATIONS", "portfolio")
    held = live.operations.goals.store.head(UUID(goal))
    before = live.activity.goal_profile(held, live.operations.heartbeats.runtime_paths)
    live.clock.return_value = NOW + timedelta(seconds=20)
    denied = client.request(
        {"operation": "GOAL_ABANDON", "goal_id": goal, "change_reason": "Later refused traffic"}
    )
    assert denied["status"] == "REFUSED"
    assert live.operations.goals.store.head(UUID(goal)).goal_hash == held.goal_hash
    after = live.activity.goal_profile(held, live.operations.heartbeats.runtime_paths)
    assert after["window"] == before["window"]
