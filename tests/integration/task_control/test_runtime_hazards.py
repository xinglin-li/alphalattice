"""The runtime hazards inventory ranked, each checked the way it would bite (RH2).

The private inventory names five two-path suspects. Each is checked here with a second party
arriving while the first holds what it holds. A check passes where the owner prevents the
hazard. Where it does not yet, the check is marked with the phase-4 fix that owns it, and it
fails once that fix lands, so the mark is removed with the fix. The recorder (`lock_order.py`)
names any pair of the product's locks a check took in both orders.

The market store's read units, suspect 2, are W0's checks
(`tests/integration/product_host/test_market_read_units.py`).
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Timer
from time import monotonic
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime import database
from alphalattice.control.workspace_runtime.database import retain_workspace_database
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _counted_runtime,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW
from tests.workspace_task_runner.lock_order import LockOrder, lock_order, record_lock_order
from tests.workspace_task_runner.task_control_support import compatibility, task_contract

__all__ = ["lock_order"]  # the fixture, used below by name


def test_the_recorder_names_a_pair_taken_in_both_orders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with record_lock_order(monkeypatch) as recorder:
        gate = WorkspaceMutationGate()
        registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
        with gate.hold(), registry._connection_lock:
            pass
        assert recorder.inversions() == []
        with registry._connection_lock, gate.hold():
            pass
    assert recorder.inversions() == [("gate", "registry")]


def test_the_recorder_names_the_order_that_stalled_the_workbench(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Before `31c49cf3` a status read retained the registry read-only and then asked for its
    connection lock, while a writer takes the lock and then the file read-write. The recorder
    names that pair even when the two never meet on two threads."""

    with record_lock_order(monkeypatch) as recorder:
        registry = DuckDbTaskControlRegistry(
            resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
        )
        first = task_contract()
        now = datetime.now(UTC)
        registry.admit(input_envelope=first[0], goal=first[1], plan=first[2], observed_at=now)
        assert recorder.inversions() == []
        with (
            retain_workspace_database(registry.database_path, read_only=True),
            registry._connection_lock,
        ):
            pass
    assert recorder.inversions() == [("instance:research-task-control.duckdb", "registry")]


def test_a_status_read_beside_a_starting_task_takes_no_pair_in_both_orders(
    tmp_path: Path, lock_order: LockOrder
) -> None:
    """Suspect 1: the Task registry's read-only unit against a starting Task (`31c49cf3`).
    The unit takes the connection lock before its instance, so the writer waits for it and
    both finish; the recorder sees one order."""

    gate = WorkspaceMutationGate()
    registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
    first = task_contract()
    now = datetime.now(UTC)
    registry.admit(input_envelope=first[0], goal=first[1], plan=first[2], observed_at=now)
    inside, go_on = Event(), Event()

    def status_read() -> int:
        with registry.retain(read_only=True):
            inside.set()
            assert go_on.wait(10)
            return len(registry.tasks())

    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(status_read)
        assert inside.wait(5)
        writer = pool.submit(
            registry.start_next,
            compatibility=compatibility(first[2]),
            worker_instance_id=uuid4(),
            observed_at=now,
        )
        with pytest.raises(FutureTimeout):
            writer.result(timeout=0.3)
        go_on.set()
        assert reader.result(timeout=10) == 1
        running, _ = writer.result(timeout=10)
    assert running.lifecycle is TaskLifecycle.RUNNING


def test_a_session_read_defers_while_a_writer_holds_the_gate(
    tmp_path: Path, lock_order: LockOrder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session's context read defers while a writer holds the gate instead of holding the page."""

    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        held, release = Event(), Event()

        def write() -> None:
            with session.mutation_gate.hold():
                held.set()
                assert release.wait(10)

        with ThreadPoolExecutor(max_workers=1) as pool:
            writer = pool.submit(write)
            assert held.wait(5)
            started = monotonic()
            with session.reads(timeout_seconds=0.2) as acquired:
                assert acquired is False
            assert monotonic() - started < 5
            from alphalattice.control.task_control.contracts import failure_code_from
            from alphalattice.control.workspace_runtime import mutation_gate
            from alphalattice.interface.local_application.cli_contract import refusal_words
            from alphalattice.interface.local_application.failure_codes import owner_failure_code

            monkeypatch.setattr(mutation_gate, "DEFAULT_WAIT_SECONDS", 0.05, raising=False)
            safety = Timer(5, release.set)
            safety.start()
            try:
                with pytest.raises(mutation_gate.GateBusy) as background:
                    session.mutation_gate.run(lambda: pytest.fail("gate was stolen"))
                assert (
                    owner_failure_code(background.value)
                    == failure_code_from(background.value)
                    == "workspace_context.writer_busy"
                )
                assert refusal_words(str(background.value))["next_requests"]["activity"] == {
                    "operation": "ACTIVITY_RECENT"
                }
                with (
                    pytest.raises(mutation_gate.GateBusy),
                    session.mutation_gate.try_hold(timeout_seconds=0.05),
                ):
                    pytest.fail("bounded gate was stolen")
                with (
                    mutation_gate.request_deadline(monotonic() + 0.01),
                    pytest.raises(mutation_gate.GateBusy),
                    session.mutation_gate.hold(),
                ):
                    pytest.fail("deadline gate was stolen")
                assert not release.is_set() and not writer.done()
            finally:
                safety.cancel()
                release.set()
            writer.result(timeout=10)
            with session.mutation_gate.hold(), session.mutation_gate.hold():
                assert session.mutation_gate.holder() is not None


def test_the_workbench_build_replaces_an_asset_a_reader_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Suspect 4: the build replaces the assets a starting Host reads; Windows
    refuses to replace a file a reader holds open, so the build retries the replace while the
    reader finishes."""

    from scripts import build_local_web_ui

    for name in build_local_web_ui.FONT_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build_local_web_ui.ASSETS / name, target)
    monkeypatch.setattr(build_local_web_ui, "ASSETS", tmp_path)
    (tmp_path / "workbench.html").write_bytes(b"<!doctype html>")
    reading = (tmp_path / "workbench.html").open("rb")  # a Host reading its page at start
    finished = Timer(0.3, reading.close)
    finished.start()
    try:
        build_local_web_ui.build()
    finally:
        finished.join()
        reading.close()
    page = (tmp_path / "workbench.html").read_bytes()
    assert page.startswith(b"<!DOCTYPE html>") and b'<script src="/workbench.' in page


def test_an_index_a_session_reads_is_not_evicted_from_under_it(tmp_path: Path) -> None:
    """Suspect 5: a storage clean-up evicts a hybrid index while a retriever holds it open.
    The retrieval service refuses while any session on the generation is open, under its
    lease lock, and evicts once the session ends."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    (database,) = sorted((tmp_path / "workspace" / ".system" / "knowledge-indexes").rglob("*.db"))
    relative = database.resolve().relative_to((tmp_path / "workspace").resolve()).as_posix()
    session = runtime.retrieval.open_session(
        document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
    )
    try:
        with pytest.raises(ValueError, match="retrieval_index_in_use"):
            runtime.retrieval.evict_generation(relative, plan_hash="a" * 64, evicted_at=_NOW)
        assert database.is_file()
    finally:
        session.close()
    runtime.retrieval.evict_generation(relative, plan_hash="a" * 64, evicted_at=_NOW)
    assert not database.is_file()
    runtime.close()


@pytest.mark.proves("workspace_runtime.background_task_gate_wait_and_cancel")
@pytest.mark.parametrize("cancel_while_returning", [False, True])
def test_dispatcher_finishes_deferred_cancel_only_after_command_returns(
    tmp_path: Path, cancel_while_returning: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Background Tasks wait through gate contention and cancellation remains once-only."""
    from alphalattice.control.task_control.contracts import WorkItemLifecycle
    from alphalattice.control.task_control.registry import TaskTransitionRejected
    from alphalattice.control.task_control.runner import TaskControlRunner
    from alphalattice.control.workspace_runtime import mutation_gate
    from alphalattice.interface.local_application.dispatcher import (
        CommandAdmission,
        LocalBackgroundDispatcher,
    )
    from tests.workspace_task_runner.task_control_support import FixtureTaskAdapter

    now = datetime.now(UTC)
    deferred, release = Event(), Event()
    entered, permit, stage_entered, stage_permit = Event(), Event(), Event(), Event()
    stage_holding, stage_release = Event(), Event()
    monkeypatch.setattr(mutation_gate, "DEFAULT_WAIT_SECONDS", 0.05)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        envelope, goal, plan = task_contract()

        class BackgroundAdapter(FixtureTaskAdapter):
            def concurrent_work_items(self, task):
                return 2

            def execute_stage(self, *, task, execution, work_item):
                stage_entered.set()
                assert stage_permit.wait(10)
                with session.mutation_gate.hold():
                    holder = session.mutation_gate.holder()
                    assert holder is not None and holder["operation"] == "factor_research"
                    assert holder["task_id"] == str(task.task_id)
                    stage_holding.set()
                    assert stage_release.wait(10)
                    return super().execute_stage(
                        task=task, execution=execution, work_item=work_item
                    )

        adapter = BackgroundAdapter(defer_once=True)

        class Command:
            command_kind = "factor_research"

            def admit(self):
                task = registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=now
                ).record
                return CommandAdmission(task.task_id, task.lifecycle)

            def execute(self, task_id):
                entered.set()
                assert permit.wait(10)
                runner = TaskControlRunner(
                    registry=registry,
                    adapters={adapter.task_kind: adapter},
                    runtime_path=str(tmp_path / "task-runtime.sqlite"),
                    clock=lambda: now,
                )
                try:
                    assert runner.run_next().lifecycle is TaskLifecycle.DEFERRED
                finally:
                    runner.close()
                deferred.set()
                assert release.wait(10)

        dispatcher = LocalBackgroundDispatcher(registry, version_moved_error=TaskTransitionRejected)
        try:
            task_id = dispatcher.submit(Command()).task_id
            assert entered.wait(10)
            with (
                mutation_gate.gate_subject("FIXTURE_WRITER", "holder"),
                session.mutation_gate.hold(),
            ):
                permit.set()
                assert not stage_entered.wait(0.2)
                assert dispatcher.failure(task_id) is None
                holder = session.mutation_gate.holder()
                assert holder is not None and holder["operation"] == "FIXTURE_WRITER"
            assert stage_entered.wait(10), dispatcher.failure(task_id)
            with (
                mutation_gate.gate_subject("FIXTURE_WRITER", "holder"),
                session.mutation_gate.hold(),
            ):
                stage_permit.set()
                assert not deferred.wait(0.2)
                assert dispatcher.failure(task_id) is None
            assert stage_holding.wait(10), dispatcher.failure(task_id)
            with pytest.raises(mutation_gate.GateBusy) as busy, session.mutation_gate.hold():
                pytest.fail("foreground caller stole the background writer")
            assert busy.value.holder is not None
            assert (busy.value.holder["operation"], busy.value.holder["task_id"]) == (
                "factor_research",
                str(task_id),
            )
            stage_release.set()
            assert deferred.wait(10), dispatcher.failure(task_id)
            if cancel_while_returning:
                assert dispatcher.request_cancel(task_id)
                assert registry.task(task_id).lifecycle is TaskLifecycle.CANCEL_REQUESTED
            release.set()
            dispatcher.drain_for_tests()
            if not cancel_while_returning:
                assert dispatcher.request_cancel(task_id)
            task = registry.task(task_id)
            assert task.lifecycle is TaskLifecycle.CANCELLED
            assert task.failure_code == "TASK_CANCELLED_AFTER_WRITER_STOPPED"
            assert task.input == envelope
            assert all(
                item.lifecycle is WorkItemLifecycle.CANCELLED
                for item in registry.work_items(task_id)
            )
            assert dispatcher.executions == 1 and dispatcher.failure(task_id) is None
            assert dispatcher.request_cancel(task_id)  # exact repeat, no second execution
        finally:
            permit.set()
            stage_permit.set()
            stage_release.set()
            release.set()
            dispatcher.close()


@pytest.mark.proves("workspace_runtime.host_stop_reaches_parallel_work_items")
def test_host_stop_reaches_a_concurrent_task_and_keeps_it_recoverable(tmp_path: Path) -> None:
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.control.task_control.runner import TaskControlRunner
    from alphalattice.interface.local_application.dispatcher import (
        CommandAdmission,
        LocalBackgroundDispatcher,
    )
    from alphalattice.kernel.shared_kernel.host_stop import checkpoint
    from tests.workspace_task_runner.task_control_support import FixtureTaskAdapter

    now = datetime.now(UTC)
    entered, release, stopping = Event(), Event(), Event()
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        envelope, goal, plan = task_contract(salt="parallel-host-stop")

        class Adapter(FixtureTaskAdapter):
            def concurrent_work_items(self, task):
                return 2

            def execute_stage(self, *, task, execution, work_item):
                entered.set()
                assert release.wait(10)
                checkpoint()
                return super().execute_stage(task=task, execution=execution, work_item=work_item)

        adapter = Adapter()
        runner = TaskControlRunner(
            registry=registry,
            adapters={adapter.task_kind: adapter},
            runtime_path=str(tmp_path / "task-runtime.sqlite"),
            clock=lambda: now,
        )

        class Command:
            command_kind = "factor_research"

            def admit(self):
                task = registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=now
                ).record
                return CommandAdmission(task.task_id, task.lifecycle)

            def execute(self, task_id):
                runner.run_next(expected_task_id=task_id)

        dispatcher = LocalBackgroundDispatcher(registry, stop_requested=stopping.is_set)
        try:
            task_id = dispatcher.submit(Command()).task_id
            assert entered.wait(10)
            stopping.set()
            dispatcher._begin_drain()
            release.set()
            assert dispatcher.close(timeout=10)
            assert registry.task(task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
            stopping.clear()
            assert runner.recover(task_id).lifecycle is TaskLifecycle.SUCCEEDED
        finally:
            release.set()
            dispatcher.close()
            runner.close()


@pytest.mark.proves("workspace_runtime.seal_record_survives_a_held_file")
def test_a_held_seal_record_stops_no_worker(tmp_path: Path, monkeypatch) -> None:
    """A reader holding the seal record is waited out; a close that outlasts it reads as a crash."""

    store = tmp_path / "research-task-control.duckdb"
    real, refusals = os.replace, [PermissionError(13, "Access is denied")] * 2

    def held(source: object, target: object) -> None:
        if refusals:
            raise refusals.pop()
        real(source, target)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", held)
    database._write_seal_record(store, epoch="first", fingerprint=None)
    assert database._seal_record(store)["epoch"] == "first"
    refusals.extend([PermissionError(13, "Access is denied")] * 99)
    with pytest.raises(WorkspaceConflictError):
        database._write_seal_record(store, epoch="second", fingerprint=None)
    assert database._seal_record(store)["epoch"] == "first"  # an open that can't clear it refuses
    database._EPOCHS[store], database._CLOSING[store] = ("second", False), 1
    database._closed(store)
    assert store not in database._EPOCHS and [p.name for p in tmp_path.iterdir()] == [
        f"{store.name}.seal.json"
    ]


@pytest.mark.proves("task_control.host_stop_preserves_recovery")
def test_host_stop_preserves_verified_work_for_resume(tmp_path: Path) -> None:
    from alphalattice.control.task_control.runner import TaskControlRunner
    from alphalattice.kernel.shared_kernel.host_stop import host_stop_scope
    from tests.workspace_task_runner.task_control_support import (
        FixtureTaskAdapter as _FixtureTaskAdapter,
    )

    now = datetime(2026, 8, 5, 17, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="host-stop")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    stopping = Event()

    class Adapter(_FixtureTaskAdapter):
        def execute_stage(self, *, task, execution, work_item):
            result = super().execute_stage(task=task, execution=execution, work_item=work_item)
            if work_item.stage_id == "resolve_inputs":
                stopping.set()
            return result

    adapter = Adapter()
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now,
        heartbeat_seconds=0.01,
    )
    try:
        with host_stop_scope(stopping.is_set):
            stopped = runner.run_next()
        assert stopped is not None and stopped.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert registry.safe_projection(task.task_id).verified_stage_count == 1
        assert adapter.executed == ["resolve_inputs"]
        with host_stop_scope(stopping.is_set):
            assert runner.recover(task.task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert registry.safe_projection(task.task_id).verified_stage_count == 1
        stopping.clear()
        with host_stop_scope(stopping.is_set):
            resumed = runner.recover(task.task_id)
        assert resumed.lifecycle is TaskLifecycle.SUCCEEDED
        assert adapter.executed == ["resolve_inputs", "publish_result"]

        from alphalattice.kernel.shared_kernel.host_stop import checkpoint

        class Interrupting(_FixtureTaskAdapter):
            interrupt = True

            def execute_stage(self, *, task, execution, work_item):
                if self.interrupt:
                    stopping.set()
                    checkpoint()
                return super().execute_stage(task=task, execution=execution, work_item=work_item)

        envelope, goal, plan = task_contract(salt="repeated-host-stop")
        other = registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=now
        ).record
        interrupting = Interrupting()
        runner.adapters[interrupting.task_kind] = interrupting
        for attempt in range(4):
            stopping.clear()
            with host_stop_scope(stopping.is_set):
                result = runner.run_next() if attempt == 0 else runner.recover(other.task_id)
            assert result.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
            assert registry.consecutive_interruptions(other.task_id, "resolve_inputs") == 0
        interrupting.interrupt = False
        stopping.clear()
        with host_stop_scope(stopping.is_set):
            assert runner.recover(other.task_id).lifecycle is TaskLifecycle.SUCCEEDED

        class Cancelling(_FixtureTaskAdapter):
            def execute_stage(self, *, task, execution, work_item):
                current = registry.task(task.task_id)
                registry.request_cancel(
                    task_id=task.task_id, expected_task_hash=current.record_hash, observed_at=now
                )
                stopping.set()
                checkpoint()
                raise AssertionError("stop was not observed")

        envelope, goal, plan = task_contract(salt="cancel-and-host-stop")
        registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
        cancelling = Cancelling()
        runner.adapters[cancelling.task_kind] = cancelling
        stopping.clear()
        with host_stop_scope(stopping.is_set):
            assert runner.run_next().lifecycle is TaskLifecycle.CANCELLED
    finally:
        runner.close()
