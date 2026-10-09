"""Tasks that may run beside others run at once, within the running places and the CPU budget."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event, Thread
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

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
from alphalattice.evidence.alternative_evidence.runtime.execution import (
    TASK_LEASES,
    TaskCpuLeases,
)
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    LocalBackgroundDispatcher,
)
from alphalattice.kernel.shared_kernel.environment import (
    numerical_thread_limit,
    numerical_thread_pools,
)
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


def test_the_host_runs_two_studies_at_once_within_the_budget(tmp_path: Path) -> None:
    """requirement: with two running places two Tasks that may run beside others run at once,
    their leases within the CPU budget and the process's priority kept in the execution log."""
    _places(tmp_path / "runtime", "2")
    both, leased = Barrier(2, timeout=60), []

    class Meeting(FixtureTaskAdapter):
        def execute_stage(self, *, task, execution, work_item):
            if work_item.stage_id == "resolve_inputs":
                leased.append(TASK_LEASES.current())
                both.wait()  # each stage waits for the other Task's: they run at once
            return super().execute_stage(task=task, execution=execution, work_item=work_item)

    with WorkspaceApplicationSession.acquire(tmp_path) as session:
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
            dispatcher.close()
    log = (tmp_path / "runtime" / "execution" / "tasks.jsonl").read_text(encoding="utf-8")
    # Each Task's last line is the one it ran with; a command driven again wrote one before it.
    lines = list({row["task_id"]: row for row in map(json.loads, log.splitlines())}.values())
    assert sorted(line["running_beside"] for line in lines) == [0, 1]
    assert sorted(leased) == sorted(line["cores"] for line in lines)  # the fits' cores
    assert sum(line["cores"] for line in lines) <= lines[0]["cpu_ceiling"]
    lowered = [
        line["running_beside"] > 0 for line in lines if "thread below_normal" in line["priority"]
    ]
    assert all(lowered) and bool(lowered) == (sys.platform == "win32")


@pytest.mark.parametrize("refusal", [ValueError, TaskTransitionRejected])
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

        class Second:
            command_kind = "factor_research"

            def admit(self) -> CommandAdmission:
                envelope, goal, plan = task_contract(salt="second")
                task = registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=NOW
                ).record
                return CommandAdmission(task.task_id, task.lifecycle.value)

            def execute(self, task_id, *, expected_task_hash=None):
                calls.append(task_id)
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


def test_a_lease_that_would_pass_the_budget_waits_for_cores() -> None:
    """requirement: the leases of Tasks running at once stay within the budget; a Task that does
    not fit waits for cores rather than run narrower."""
    leases, taken = TaskCpuLeases(), Event()
    first, second, third = uuid4(), uuid4(), uuid4()

    def hold() -> None:
        with leases.lease(third, 2, 4):
            taken.set()

    with leases.lease(first, 2, 4), leases.lease(second, 2, 4) as beside:
        waiting = Thread(target=hold)
        waiting.start()
        assert beside == 1 and not taken.wait(0.2) and leases.widest() == 2
    waiting.join(10)
    assert taken.is_set() and leases.holders() == ()


def test_thread_limits_closing_out_of_order_never_leave_an_open_scope_unlimited() -> None:
    """regression: the limit is the process's, and a scope that closed first restored the
    counts from before it while another Task's scope was still open."""

    def counts() -> list[int]:
        return [pool["num_threads"] for pool in numerical_thread_pools()]

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
