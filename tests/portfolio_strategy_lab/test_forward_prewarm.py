"""Startup and update warming use the ordinary verified Forward read owner."""

from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
    LocalWebSessionError,
    PortfolioForwardPrewarm,
    read_workbench_portfolio,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    TASK_KIND as PORTFOLIO_UPDATE_TASK_KIND,
)
from alphalattice.control.product_host.maintenance.data_update import DATA_UPDATE_TASK_KIND
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.interface.local_application.web import LocalWebError
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)


def _operations(states):
    return SimpleNamespace(
        activations=SimpleNamespace(summary=lambda package: states[package]),
        manifests=SimpleNamespace(
            current=SimpleNamespace(
                decision_updates=tuple(
                    SimpleNamespace(strategy_package_id=package) for package in states
                )
            )
        ),
    )


def _active(task_id, when="2026-10-05T10:00:00+00:00"):
    return {"status": "ACTIVE", "book_task_id": str(task_id), "activated_at": when}


def test_startup_queues_only_the_latest_authoritative_active_book(monkeypatch):
    earlier, latest = uuid4(), uuid4()
    operations = _operations(
        {
            "earlier": _active(earlier, "2026-10-04T10:00:00+00:00"),
            "latest": _active(latest),
            "moved": {**_active(uuid4(), "2026-10-06T10:00:00+00:00"), "status": "MOVED"},
            "inactive": {"status": "INACTIVE"},
            "unbound": {**_active(uuid4()), "book_task_id": None},
        }
    )
    entered, release = Event(), Event()
    calls = []

    def read(owner, query, *, caller):
        calls.append((owner, query, caller))
        entered.set()
        assert release.wait(5)
        return {"status": "READ"}

    monkeypatch.setattr(
        "alphalattice.control.product_host.composition.local_web_session.read_workbench_portfolio",
        read,
    )
    prewarm = PortfolioForwardPrewarm(operations)
    prewarm.start()
    try:
        # start returned while the read is held, and the retained activation
        # owner selected the exact book rather than a newest Task or package.
        assert entered.wait(5)
        assert calls == [
            (
                operations,
                {"task_id": [str(latest)], "performance": ["latest"]},
                "SERVICE_AUTOMATION",
            )
        ]
    finally:
        release.set()
        assert prewarm.close()


def test_update_wake_preserves_activity_and_requires_terminal_success():
    task_id = uuid4()
    state = [TaskLifecycle.SUCCEEDED]
    operations = _operations({})
    operations.workspace_session = SimpleNamespace(
        task_control_registry=SimpleNamespace(
            task=lambda identity: SimpleNamespace(task_id=identity, lifecycle=state[0])
        )
    )
    observed, wakes = [], []
    dispatcher = SimpleNamespace(on_command_returned=lambda *args: observed.append(args))
    prewarm = PortfolioForwardPrewarm(operations)
    prewarm.wake = lambda: wakes.append("wake")
    prewarm.attach(dispatcher)
    for kind in (DATA_UPDATE_TASK_KIND, ADVANCEMENT_TASK_KIND, PORTFOLIO_UPDATE_TASK_KIND):
        dispatcher.on_command_returned(kind, task_id, None)
    dispatcher.on_command_returned("unrelated", task_id, None)
    dispatcher.on_command_returned(DATA_UPDATE_TASK_KIND, task_id, "worker_failed")
    for lifecycle in (TaskLifecycle.DEFERRED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED):
        state[0] = lifecycle
        dispatcher.on_command_returned(DATA_UPDATE_TASK_KIND, task_id, None)
    assert len(observed) == 8
    assert wakes == ["wake"] * 3
    assert prewarm.close()


def test_pending_wakes_coalesce_and_reads_remain_serial(monkeypatch):
    operations = _operations({"active": _active(uuid4())})
    entered = (Event(), Event())
    release = (Event(), Event())
    calls = []

    def read(_owner, _query, *, caller):
        index = len(calls)
        calls.append(caller)
        assert index < 2
        entered[index].set()
        assert release[index].wait(5)
        return {"status": "READ"}

    monkeypatch.setattr(
        "alphalattice.control.product_host.composition.local_web_session.read_workbench_portfolio",
        read,
    )
    prewarm = PortfolioForwardPrewarm(operations)
    prewarm.start()
    try:
        assert entered[0].wait(5)
        for _ in range(100):
            prewarm.wake()
        assert not entered[1].is_set()
        release[0].set()
        assert entered[1].wait(5)
        assert calls == ["SERVICE_AUTOMATION"] * 2
    finally:
        release[0].set()
        release[1].set()
        assert prewarm.close()


def test_bounded_stop_retains_session_until_the_read_has_ended(tmp_path: Path, monkeypatch):
    operations = _operations({"active": _active(uuid4())})
    entered, release = Event(), Event()
    exits = []

    def read(_owner, _query, *, caller):
        entered.set()
        assert release.wait(5)
        return {"status": "READ"}

    monkeypatch.setattr(
        "alphalattice.control.product_host.composition.local_web_session.read_workbench_portfolio",
        read,
    )
    retained = SimpleNamespace(__exit__=lambda *args: exits.append(args))
    session = LocalPortfolioWebSession(
        workspace=tmp_path,
        workspace_manifest=SimpleNamespace(workspace_id="prewarm-shutdown"),
        resolver=None,
    )
    session.session = retained
    session.forward_prewarm = PortfolioForwardPrewarm(operations)
    session.forward_prewarm.start()
    try:
        assert entered.wait(5)
        with pytest.raises(LocalWebSessionError, match=r"local_web_session\.writer_still_live"):
            session.stop(timeout=0)
        assert session.session is retained and exits == []
        session.forward_prewarm.wake()
    finally:
        release.set()
        session.stop()
    assert len(exits) == 1
    assert session.session is None and session.forward_prewarm is None


def test_shared_forward_read_preserves_exact_refusal_and_service_caller():
    task_id = uuid4()
    refusal = {"status": "REFUSED", "failure_code": "artifact_tampered"}
    calls = []

    def execute(request, **kwargs):
        calls.append((request, kwargs))
        return refusal

    operations = SimpleNamespace(application=None, execute=execute)
    query = {"task_id": [str(task_id)], "performance": ["latest"]}
    assert read_workbench_portfolio(operations, query) is refusal
    with pytest.raises(LocalWebError, match="artifact_tampered"):
        read_workbench_portfolio(operations, query, caller="SERVICE_AUTOMATION")
    assert [(request.operation, request.task_id) for request, _ in calls] == [
        ("EXPERIMENT_READBACK", task_id)
    ] * 2
    assert [kwargs for _, kwargs in calls] == [{}, {"caller": "SERVICE_AUTOMATION"}]
    with pytest.raises(LocalWebError, match=r"workbench\.query_field_unknown"):
        read_workbench_portfolio(operations, {**query, "unknown": ["1"]})
    assert len(calls) == 2


def test_failed_verification_leaves_foreground_readable_and_worker_wakeable(monkeypatch):
    task_id = uuid4()
    operations = _operations({"active": _active(task_id)})
    operations.application = None
    refusal = {"status": "REFUSED", "failure_code": "artifact_tampered"}
    operations.execute = lambda request: refusal
    entered, repaired = Event(), Event()
    calls = []

    def read(_owner, _query, *, caller):
        calls.append(caller)
        if len(calls) == 1:
            entered.set()
            raise ValueError("synthetic_invalid_archive")
        repaired.set()
        return {"status": "READ"}

    monkeypatch.setattr(
        "alphalattice.control.product_host.composition.local_web_session.read_workbench_portfolio",
        read,
    )
    prewarm = PortfolioForwardPrewarm(operations)
    prewarm.start()
    try:
        assert entered.wait(5)
        deadline = monotonic() + 5
        while prewarm.last_error_code is None and monotonic() < deadline:
            repaired.wait(0.01)
        assert prewarm.last_error_code == "workspace_maintenance.background_cycle_failed"
        assert read_workbench_portfolio(operations, {"task_id": [str(task_id)]}) is refusal
        prewarm.wake()
        assert repaired.wait(5)
    finally:
        assert prewarm.close()
    assert prewarm.last_error_code is None


def test_service_authority_admits_only_retained_forward_reads(live, monkeypatch):
    """Exercise the real operation authority before synthetic retained read ports."""
    operations = live.operations
    before = tuple(live.session.task_control_registry.tasks())
    calls = []

    def absent(code):
        return {
            "status": "REFUSED",
            "failure_code": code,
            "detail": "The retained artifact is absent in this synthetic workspace.",
            "next_requests": {"workspace": {"operation": "WORKSPACE_SHOW"}},
        }

    def report(_owner, result_hash, portfolio_session):
        calls.append(("REPORT", result_hash, portfolio_session))
        return absent("portfolio_research.result_not_found")

    def experiment(_owner, request, *, caller, agent_execution):
        calls.append((request.operation, request.task_id, caller))
        return absent("task_control.task_not_found")

    def update(_owner, task_id):
        calls.append(("PORTFOLIO_UPDATE_READBACK", task_id))
        return absent("portfolio_update.artifact_missing")

    monkeypatch.setattr(type(operations), "report", report)
    monkeypatch.setattr(type(operations.experiments), "operate", experiment)
    monkeypatch.setattr(type(operations.updates), "readback", update)
    task_id = uuid4()
    requests = (
        PortfolioResearchOperationRequest(operation="REPORT", result_hash="a" * 64),
        PortfolioResearchOperationRequest(operation="EXPERIMENT_READBACK", task_id=task_id),
        PortfolioResearchOperationRequest(operation="PORTFOLIO_UPDATE_READBACK", task_id=task_id),
    )
    for request, code in zip(
        requests,
        (
            "portfolio_research.result_not_found",
            "task_control.task_not_found",
            "portfolio_update.artifact_missing",
        ),
        strict=True,
    ):
        answer = operations.execute(request, caller="SERVICE_AUTOMATION")
        assert answer["status"] == "REFUSED" and answer["failure_code"] == code
    assert calls == [
        ("REPORT", "a" * 64, None),
        ("EXPERIMENT_READBACK", task_id, "SERVICE_AUTOMATION"),
        ("PORTFOLIO_UPDATE_READBACK", task_id),
    ]
    for request in (
        PortfolioResearchOperationRequest(operation="RUN", spec={}),
        PortfolioResearchOperationRequest(operation="STRATEGY_ACTIVATE", task_id=task_id),
        PortfolioResearchOperationRequest(
            operation="PORTFOLIO_UPDATE_RUN", update_plan_hash="b" * 64
        ),
        PortfolioResearchOperationRequest(operation="DATA_UPDATE_RUN", update_plan_hash="b" * 64),
    ):
        with pytest.raises(ValueError, match=r"research_update\.automation_operation_not_admitted"):
            operations.execute(request, caller="SERVICE_AUTOMATION")
    assert len(calls) == 3
    assert tuple(live.session.task_control_registry.tasks()) == before


def test_returned_refusal_is_a_failed_prewarm_cycle():
    task_id = uuid4()
    operations = _operations({"active": _active(task_id)})
    operations.application = None
    entered = Event()

    def execute(request, *, caller):
        assert request.operation == "EXPERIMENT_READBACK" and request.task_id == task_id
        assert caller == "SERVICE_AUTOMATION"
        entered.set()
        return {"status": "REFUSED", "failure_code": "portfolio_update.artifact_missing"}

    operations.execute = execute
    prewarm = PortfolioForwardPrewarm(operations)
    prewarm.start()
    try:
        assert entered.wait(5)
        deadline = monotonic() + 5
        while prewarm.last_error_code is None and monotonic() < deadline:
            Event().wait(0.01)
        assert prewarm.last_error_code == "workspace_maintenance.background_cycle_failed"
    finally:
        assert prewarm.close()
