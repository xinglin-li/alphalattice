"""Tasks that may run beside others run at once, within the running places and the CPU budget."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from threading import Barrier, Event, Thread
from types import SimpleNamespace
from typing import Any, Never
from uuid import uuid4

import pytest
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from alphalattice.capabilities.domain_contracts.bounded_memo import open_stage_digests
from alphalattice.capabilities.domain_contracts.environment import (
    numerical_thread_limit,
    numerical_thread_pools,
)
from alphalattice.capabilities.domain_contracts.source_identity import open_source_role_acquisition
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.local_web_session import runs_beside_others
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.queue import (
    read_queue_setting,
    running_places,
    write_queue_setting,
)
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    TaskTransitionRejected,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.evidence.alternative_evidence.runtime import execution
from alphalattice.evidence.alternative_evidence.runtime.execution import (
    TASK_LEASES,
    TaskCpuLeases,
)
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    LocalBackgroundDispatcher,
)
from alphalattice.kernel.resources import ResourceRefused
from tests.workspace_task_runner.task_control_support import (
    FixtureTaskAdapter,
    compatibility,
    task_contract,
)

NOW = datetime(2026, 10, 9, 15, tzinfo=UTC)


def _places(runtime: Path, running: str) -> None:
    for field, value in (("tasks_waiting", "8"), ("tasks_running", running)):
        write_queue_setting(runtime, value, chosen_by="HUMAN", chosen_at=NOW, field=field)


class _Study:
    """A Factor study's command over a fixture Task; with `waiting`, one whose turn has not come
    returns and sets it, as a refused start does."""

    command_kind = "factor_research"

    def __init__(self, session: Any, salt: str, adapter: Any, waiting: Event | None = None):
        self.session, self.salt, self.adapter, self.waiting = session, salt, adapter, waiting

    def admit(self) -> CommandAdmission:
        envelope, goal, plan = task_contract(salt=self.salt)
        task = self.session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=NOW
        ).record
        return CommandAdmission(task.task_id, task.lifecycle.value)

    def execute(self, task_id: Any, *, expected_task_hash: str | None = None) -> None:
        registry = self.session.task_control_registry
        if self.waiting is not None and not registry.may_start(task_id):
            self.waiting.set()
            return
        task = registry.task(task_id)
        self.session.execute_admitted(task, self.adapter, lambda: NOW, expected_task_hash)


def test_tasks_that_may_run_beside_others_share_the_places_in_queue_order(tmp_path: Path) -> None:
    """requirement: Tasks that may run beside others start together within the running places;
    one that runs alone waits for them to end and holds back every Task behind it."""
    _places(tmp_path / "runtime", "2")
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    registry.overlapping = lambda task: task.input.payload["salt"] != "alone"
    started: list[int] = []
    registry.started = lambda: started.append(1)
    ids = {}
    for i, salt in enumerate(("a", "b", "c", "alone", "d")):
        envelope, goal, plan = task_contract(salt=salt)
        at = NOW + timedelta(seconds=i)
        ids[salt] = registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=at
        ).record.task_id

    def start() -> str | None:
        begun = registry.start_next(
            compatibility=compatibility(plan), worker_instance_id=uuid4(), observed_at=NOW
        )
        return None if begun is None else begun[0].input.payload["salt"]

    def end(salt: str) -> None:
        for step in (registry.request_cancel, registry.finalize_cancel_after_writer_stopped):
            current = registry.task(ids[salt])
            step(task_id=current.task_id, expected_task_hash=current.record_hash, observed_at=NOW)

    assert (start(), start(), start()) == ("a", "b", None)
    assert not registry.may_start(ids["c"]) and not registry.may_start(ids["d"])
    end("a")
    assert registry.may_start(ids["c"]) and start() == "c"
    end("b")
    assert start() is None and not registry.may_start(ids["alone"])
    end("c")
    assert start() == "alone" and not registry.may_start(ids["d"]) and len(started) == 4
    assert running_places(read_queue_setting(tmp_path / "runtime")) == (2, "set to 2")
    # A recovery takes a place by the same rule: alone, not beside a running Task.
    registry.mark_recovery_required(task_id=ids["alone"], failure_code="RETRY_DUE", observed_at=NOW)
    assert start() == "d"
    recover = {"compatibility": compatibility(plan), "worker_instance_id": uuid4()}
    with pytest.raises(TaskTransitionRejected, match="recovery_waits_for_the_running_task"):
        registry.restart_recovery(task_id=ids["alone"], observed_at=NOW, **recover)
    registry.overlapping = lambda _task: True
    registry.restart_recovery(task_id=ids["alone"], observed_at=NOW, **recover)


def test_the_host_runs_two_studies_at_once_within_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: with two running places two Tasks that may run beside others run at once,
    their leases within the CPU budget and the process's priority kept in the execution log."""
    _places(tmp_path / "runtime", "2")
    execution.CpuBudgetStore(tmp_path / "runtime").write(4, chosen_by="HUMAN", chosen_at=NOW)
    both, leased = Barrier(2, timeout=60), []
    load_samples: list[object] = []

    def sample_load(**kwargs):
        load_samples.append(kwargs)
        return 0.0

    class Meeting(FixtureTaskAdapter):
        def execute_stage(self, *, task, execution, work_item):
            if work_item.stage_id == "resolve_inputs":
                leased.append(TASK_LEASES.current())
                both.wait()  # each stage waits for the other Task's: they run at once
            return super().execute_stage(task=task, execution=execution, work_item=work_item)

    import platform

    original_processor = platform.processor
    expected_processor = original_processor() or platform.machine()
    processor_calls = []

    def processor():
        processor_calls.append(True)
        return original_processor()

    monkeypatch.setattr(platform, "processor", processor)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        assert len(processor_calls) == 1  # the persisted value is acquired at Host start
        monkeypatch.setattr(execution, "busy_logical_processors", sample_load)
        events = []
        watching = [True]

        def audit(event, _args):
            if watching[0] and event == "_wmi.exec_query":
                events.append(event)

        sys.addaudithook(audit)
        registry = session.task_control_registry
        registry.overlapping = lambda _task: True
        dispatcher = LocalBackgroundDispatcher(
            registry, version_moved_error=TaskTransitionRejected, workers=lambda: 2
        )
        registry.started = dispatcher.drive_waiting
        try:
            sent = [
                dispatcher.submit(_Study(session, s, Meeting())).task_id for s in ("one", "two")
            ]
            dispatcher.drain_for_tests()
            for task_id in sent:
                assert registry.task(task_id).lifecycle is TaskLifecycle.SUCCEEDED
        finally:
            watching[0] = False
            dispatcher.close()
    assert not load_samples, "explicit Task CPU budgets do not sample live load"
    assert not (set(sent) & set(TASK_LEASES.holders()))
    assert events == []  # startup is excluded; served Task requests issue no WMI query
    log = (tmp_path / "runtime" / "execution" / "tasks.jsonl").read_text(encoding="utf-8")
    # Each Task's last line is the one it ran with; a command driven again wrote one before it.
    lines = list({row["task_id"]: row for row in map(json.loads, log.splitlines())}.values())
    assert len(processor_calls) == 1  # no processor/WMI lookup while requests run
    assert {line["platform"]["processor"] for line in lines} == {expected_processor}
    assert all(
        line["cpu_budget"] == 4 and line["machine"]["busy_processors"] is None for line in lines
    )
    # Either task can observe the other after both have started; the barrier above
    # proves overlap, while each snapshot must stay within the two-place budget.
    assert sorted(line["running_beside"] for line in lines) in ([0, 1], [1, 1])
    assert sorted(leased) == sorted(line["cores"] for line in lines)  # the fits' cores
    assert sum(line["cores"] for line in lines) <= lines[0]["cpu_ceiling"]
    lowered = [
        line["running_beside"] > 0 for line in lines if "thread below_normal" in line["priority"]
    ]
    assert all(lowered) and bool(lowered) == (sys.platform == "win32")


@pytest.mark.parametrize(
    "refusal", [ValueError, TaskTransitionRejected], ids=["value_error", "transition_rejected"]
)
def test_a_command_that_read_the_head_before_another_start_is_driven_again(
    tmp_path: Path, refusal: type[Exception]
) -> None:
    """regression: a Task's command read the queue's head before an earlier Task's start
    committed, was refused (the runner's or the registry's mismatch) and was dropped though it
    could then start; a run that stops stays the command's, with its failure."""
    _places(tmp_path / "runtime", "2")
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        registry.overlapping = lambda _task: True
        envelope, goal, plan = task_contract(salt="first")
        registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=NOW)
        calls: list[object] = []
        scoped: list[object] = []  # each run hashes and observes once outside its stages

        class Second:
            command_kind = "factor_research"

            def admit(self) -> CommandAdmission:
                scoped.append(open_stage_digests() is not None)
                envelope, goal, plan = task_contract(salt="second")
                task = registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=NOW
                ).record
                return CommandAdmission(task.task_id, task.lifecycle.value)

            def execute(self, task_id, *, expected_task_hash=None):
                calls.append(task_id)
                scoped.append(None not in (open_stage_digests(), open_source_role_acquisition()))
                if len(calls) == 1:  # the first Task's start lands after this read of the head
                    registry.start_next(
                        compatibility=compatibility(plan),
                        worker_instance_id=uuid4(),
                        observed_at=NOW,
                    )
                    raise refusal("task_control.queued_task_identity_mismatch")
                session.execute_admitted(
                    registry.task(task_id), FixtureTaskAdapter(), lambda: NOW, expected_task_hash
                )

        dispatcher = LocalBackgroundDispatcher(
            registry, version_moved_error=TaskTransitionRejected, workers=lambda: 2
        )
        registry.started = dispatcher.drive_waiting
        try:
            second = dispatcher.submit(Second()).task_id
            dispatcher.drain_for_tests()
            assert registry.task(second).lifecycle is TaskLifecycle.SUCCEEDED and len(calls) == 2
            assert all(scoped) and len(scoped) == 3
            stopping = FixtureTaskAdapter(fail_once=True)
            Second.execute = lambda self, task_id, **_: session.execute_admitted(  # type: ignore[method-assign]
                registry.task(task_id), stopping, lambda: NOW
            )
            stopped = dispatcher.submit(Second()).task_id
            dispatcher.drain_for_tests()
            assert registry.task(stopped).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
            failure = str(dispatcher.failure(stopped))
            assert "simulated process interruption" in failure and stopping.executed == [
                "resolve_inputs"
            ]
        finally:
            dispatcher.close()


def test_a_command_waiting_its_turn_at_a_close_runs_before_the_workers_stop(
    tmp_path: Path,
) -> None:
    """regression: a command deferred behind a running Task was dropped when the dispatcher
    closed, though a queued one runs before the workers stop; its Task stayed queued."""
    _places(tmp_path / "runtime", "2")
    release, tried = Event(), Event()

    class Held(FixtureTaskAdapter):
        def execute_stage(self, *, task, execution, work_item):
            release.wait(60)  # the first Task runs until the close has begun
            return super().execute_stage(task=task, execution=execution, work_item=work_item)

    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        dispatcher = LocalBackgroundDispatcher(
            registry, version_moved_error=TaskTransitionRejected, workers=lambda: 2
        )
        registry.started = dispatcher.drive_waiting
        dispatcher.submit(_Study(session, "one", Held(), tried))
        second = dispatcher.submit(_Study(session, "two", FixtureTaskAdapter(), tried)).task_id
        assert tried.wait(60)  # its turn had not come: the first Task holds the place
        closing = Thread(target=dispatcher.close)
        closing.start()
        while not dispatcher.closing:
            closing.join(0.01)
        release.set()
        closing.join(60)
        assert registry.task(second).lifecycle is TaskLifecycle.SUCCEEDED


def test_studies_other_than_alpha_and_the_install_may_run_beside_others() -> None:
    """requirement: Risk and Factor studies and the Evidence install may run beside others; an
    Alpha study, a qualification and any other Task run alone."""

    def task(kind: str, kept: dict | None = None) -> Any:
        return SimpleNamespace(task_kind=kind, input=SimpleNamespace(payload=kept or {}))

    def study(kind: str, family: object = None) -> Any:
        plan = {"program": {"kind": kind}, "qualification_family": family}
        return task("research_experiment", {"plan": plan})

    assert all(
        map(runs_beside_others, (study("risk.covariance-development"), task("evidence_install")))
    )
    assert runs_beside_others(study("factor.screening-development"))
    assert not any(
        map(
            runs_beside_others,
            (
                study("alpha.model-development"),
                study("risk.covariance-development", {"members": []}),
                task("workspace_preparation"),
            ),
        )
    )


@pytest.mark.parametrize("shrink", [False, True])
def test_a_lease_that_would_pass_the_budget_waits_for_cores(monkeypatch, shrink) -> None:
    """Wait for a grant while it fits the current budget, then admit or refuse by name."""
    leases, taken, waiting, settled = TaskCpuLeases(), Event(), Event(), Event()
    with (
        pytest.raises(ResourceRefused, match=r"^resources.capacity_exceeded:cpu$"),
        leases.lease(uuid4(), 5, 4),
    ):
        pass
    first, second, third = uuid4(), uuid4(), uuid4()
    refused = []

    def hold() -> None:
        try:
            with leases.lease(third, 2, 8):
                taken.set()
        except ResourceRefused as error:
            refused.append(str(error))
        finally:
            settled.set()

    wait_for = leases._turn.wait_for

    def mark_wait(predicate):
        waiting.set()
        return wait_for(predicate)

    with leases.lease(first, 4, 16), leases.lease(second, 4, 16) as beside:
        monkeypatch.setattr(leases._turn, "wait_for", mark_wait)
        waiter = Thread(target=hold)
        waiter.start()
        assert waiting.wait(5)
        assert beside == 1 and not taken.is_set() and leases.widest() == 4
        if shrink:
            with (
                pytest.raises(ResourceRefused, match=r"^resources.capacity_exceeded:cpu$"),
                leases.lease(uuid4(), 2, 1),
            ):
                pass
            settled_while_held = settled.wait(5)
        else:
            settled_while_held = settled.is_set()
    # Reset the budget after the scenario to let even a faulty waiter exit cleanly.
    with leases.lease(uuid4(), 1, 16):
        pass
    waiter.join(5)
    assert not waiter.is_alive() and leases.holders() == ()
    if shrink:
        assert settled_while_held, "a smaller policy must wake the old grant waiter"
        assert refused == ["resources.capacity_exceeded:cpu"] and not taken.is_set()
    else:
        assert not settled_while_held and taken.is_set() and not refused


def test_thread_limits_closing_out_of_order_never_leave_an_open_scope_unlimited() -> None:
    """regression: the limit is the process's, and a scope that closed first restored the
    counts from before it while another Task's scope was still open."""

    def counts() -> list[int]:
        return [pool["num_threads"] for pool in numerical_thread_pools()]

    with numerical_thread_limit(1):  # the first scope loads the libraries it holds; this does
        pass
    with threadpool_limits(limits=3):  # a known count above one for the scopes to narrow
        before = counts()
        assert before and max(before) > 1
        first, second = numerical_thread_limit(2), numerical_thread_limit(1)
        first.__enter__()
        second.__enter__()
        first.__exit__(None, None, None)
        assert set(counts()) == {1}
        with pytest.raises(RuntimeError), numerical_thread_limit(2):
            raise RuntimeError("a fit that failed inside its scope")
        assert set(counts()) == {1}
        second.__exit__(None, None, None)
        assert counts() == before


def _dispatch_case(tmp_path: Path, *, width, extra, expected) -> None:
    runtime = tmp_path / "runtime"
    _places(runtime, str(width))
    execution.CpuBudgetStore(runtime).write(width, chosen_by="HUMAN", chosen_at=NOW)
    begun = Barrier(width + 1, timeout=20)
    release, queued_started = Event(), Event()
    order: list[str] = []

    class Held(FixtureTaskAdapter):
        def execute_stage(self, *, task, execution, work_item):
            if work_item.stage_id == "resolve_inputs":
                begun.wait()
                assert release.wait(20)
            return super().execute_stage(task=task, execution=execution, work_item=work_item)

    class Remembered(FixtureTaskAdapter):
        def execute_stage(self, *, task, execution, work_item):
            if work_item.stage_id == "resolve_inputs":
                queued_started.set()
                order.append(task.input.payload["salt"])
            return super().execute_stage(task=task, execution=execution, work_item=work_item)

    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        registry.overlapping = lambda _task: True
        dispatcher = LocalBackgroundDispatcher(
            registry, version_moved_error=TaskTransitionRejected, workers=lambda: width
        )
        registry.started = dispatcher.drive_waiting
        try:
            active = [
                dispatcher.submit(_Study(session, f"held-{index}", Held())).task_id
                for index in range(width)
            ]
            try:
                begun.wait()
                queued = []
                for lane in extra:
                    command = _Study(session, lane, Remembered())
                    submission = (
                        dispatcher.admit_request(partial(dispatcher.submit, command))
                        if lane == "interactive"
                        else dispatcher.submit(command)
                    )
                    queued.append(submission.task_id)
                assert not queued_started.is_set()
                assert {
                    task.task_id
                    for task in registry.tasks()
                    if task.lifecycle is TaskLifecycle.RUNNING
                } == set(active)
                assert set(TASK_LEASES.holders()) == set(active)
            finally:
                release.set()
            dispatcher.drain_for_tests(timeout=20)
            remaining = {}
            for task_id in (*active, *queued):
                task = registry.task(task_id)
                if task.lifecycle is not TaskLifecycle.SUCCEEDED:
                    remaining[task.input.payload["salt"]] = (
                        task.lifecycle,
                        task.failure_code,
                        dispatcher.failure(task_id),
                    )
            assert remaining == {}, (remaining, order)
        finally:
            dispatcher.close(timeout=5)
    assert tuple(order) == expected


def _query_stop_case(tmp_path: Path, *, deadline: bool) -> None:
    from concurrent.futures import Future
    from contextlib import closing, suppress
    from threading import Timer

    import duckdb

    import alphalattice.kernel.execution.cancel as cancellation
    from alphalattice.control.workspace_runtime.database import open_workspace_database

    entered, interrupted = Event(), Event()
    raw = []
    finished = Future()

    def announce(_index: int) -> bool:
        entered.set()
        return True

    class Query(FixtureTaskAdapter):
        def execute_stage(self, **_arguments: Any) -> Never:
            with closing(open_workspace_database(tmp_path / "query.duckdb", read_only=False)) as db:
                db.execute("SET threads = 1")
                db.create_function("announce", announce, ["BIGINT"], "BOOLEAN", side_effects=True)
                raw.append(db._raw)
                try:
                    db.execute(
                        "SELECT sum(hash(i)) FROM range(1000000000) AS work(i)"
                        " WHERE i <> 0 OR announce(i)"
                    ).fetchall()
                except duckdb.InterruptException:
                    interrupted.set()
                    raise
            raise AssertionError("the live query completed before its execution was stopped")

    def short_timer(seconds, action, args):
        assert 0 < seconds < float("inf")
        return Timer(3, action, args)

    with (
        WorkspaceApplicationSession.acquire(tmp_path) as session,
        pytest.MonkeyPatch.context() as patch,
    ):
        if deadline:
            patch.setattr(cancellation, "Timer", short_timer)
        registry = session.task_control_registry
        envelope, goal, plan = task_contract(salt="execution-query-stop")
        task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=NOW).record
        adapter = Query()

        def clock():
            return NOW + timedelta(seconds=1)

        def drive() -> None:
            try:
                finished.set_result(session.execute_admitted(task, adapter, clock))
            except BaseException as error:
                finished.set_exception(error)

        work = Thread(target=drive)
        work.start()
        try:
            assert entered.wait(8), (
                finished.exception() if finished.done() else "query never entered"
            )
            if not deadline:
                current = registry.task(task.task_id)
                registry.request_cancel(
                    task_id=task.task_id,
                    expected_task_hash=current.record_hash,
                    observed_at=NOW + timedelta(seconds=1),
                )
            assert interrupted.wait(5 if deadline else 1), "DuckDB did not receive its interrupt"
            stopped = finished.result(timeout=5)
            assert stopped is not None
            if deadline:
                assert stopped.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
                assert stopped.failure_code == "execution.timeout"
            else:
                assert stopped.lifecycle is TaskLifecycle.CANCELLED
        finally:
            if work.is_alive() and raw and not interrupted.is_set():
                # A failing mutation must still release its test-owned raw query.
                with suppress(duckdb.Error):
                    raw[0].interrupt()
            work.join(5)
            assert not work.is_alive()


def _crash_case(tmp_path: Path) -> None:
    from alphalattice.control.task_control import child
    from alphalattice.control.task_control.runner import TaskControlRunner
    from tests.integration.task_control.test_task_child import HERE

    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="execution-crashed-worker")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=NOW).record

    class CrashingStage(FixtureTaskAdapter):
        def execute_stage(self, **_arguments: Any) -> Never:
            child.run_in_child(f"{HERE}:ended", {})
            raise AssertionError("the exited child returned to its stage")

    adapter = CrashingStage()
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: NOW + timedelta(seconds=1),
    )
    try:
        first = child.run_in_child(f"{HERE}:doubled", {"value": 1})
        (generation,) = child._WORKERS[0]._processes.values()
        with pytest.raises(child.ChildInterrupted, match=r"^task_control\.child_process_ended$"):
            runner.run_next()
        stopped = registry.task(task.task_id)
        assert stopped.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert stopped.failure_code == "TASK_EXECUTION_INTERRUPTED"
        interrupted = next(
            item for item in registry.work_items(task.task_id) if item.stage_id == "resolve_inputs"
        )
        assert interrupted.failure_cause is not None
        assert interrupted.failure_cause.detail == "task_control.child_process_ended"
        assert generation.exitcode == 7 and 0 not in child._WORKERS

        second = child.run_in_child(f"{HERE}:doubled", {"value": 2})
        assert second["value"] == 4 and second["pid"] != first["pid"]
    finally:
        runner.close()
        if (worker := child._WORKERS.get(0)) is not None:
            child._retire_worker(0, worker, wait_for_exit=True)


@pytest.mark.proves("execution.orders_tasks_within_the_grant")
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            partial(
                _dispatch_case,
                width=1,
                extra=("background", "interactive"),
                expected=("interactive", "background"),
            ),
            id="interactive ahead of queued background",
        ),
        pytest.param(
            partial(_dispatch_case, width=4, extra=("background",), expected=("background",)),
            id="five background Tasks on four cores",
        ),
        pytest.param(partial(_query_stop_case, deadline=False), id="cancel a running DuckDB query"),
        pytest.param(partial(_query_stop_case, deadline=True), id="Task execution timeout"),
        pytest.param(_crash_case, id="a crashed worker is replaced"),
    ],
)
def test_execution_admits_tasks_in_their_lane_within_the_grant(tmp_path: Path, case) -> None:
    case(tmp_path)
