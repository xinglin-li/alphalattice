"""A 0.2.7 workspace's Task ledgers moved onto the log: its store, its requests, a store lost."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.task_control.contracts import (
    TaskHeartbeatSignal,
    TaskLifecycle,
    WorkItemLifecycle,
)
from alphalattice.control.task_control.ledger import SubmittingAgent, TaskAdmissionRequest
from alphalattice.control.task_control.queue import write_queue_setting
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    TaskTransitionRejected,
    TaskVersionStale,
    resolve_task_control_database,
)
from alphalattice.control.task_control.upgrade import RETIRED_DIRECTORY
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.log import uuid7
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from tests.workspace_task_runner.task_control_support import (
    compatibility,
    complete_stage,
    lose_the_log_keeping_frozen_requests,
    task_contract,
    write_0_2_7_store,
)

TEST_SOURCE = "src/alphalattice/control/task_control/upgrade.py"


def _runtime_ledgers_keep_their_readings_across_an_interrupted_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family_name: str,
) -> None:
    """Integration: real legacy files, committed log waits and SDK spans cross a failed rename
    and a reopened log; an in-memory unit cannot prove the reader/retirement boundary."""
    from alphalattice.control.task_control import upgrade as owner
    from alphalattice.control.task_control.runner import TaskControlRunner, TaskHeartbeatReader
    from alphalattice.control.task_control.timing import read_task_telemetry
    from alphalattice.foundation.data_layer.log import replay

    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt=family_name)
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    worker = uuid4()
    started = registry.start_next(
        compatibility=compatibility(plan), worker_instance_id=worker, observed_at=now
    )
    assert started is not None
    _, execution = started
    signal = TaskHeartbeatSignal.from_identity(
        task_id=task.task_id,
        execution_id=execution.execution_id,
        worker_instance_id=worker,
        sequence=1,
        observed_at=now,
    )
    runtime = tmp_path / "runtime" / "old.sqlite"
    path = runtime.with_name("old-heartbeats.sqlite")
    path.parent.mkdir(exist_ok=True)
    if family_name == "task-heartbeats":
        with sqlite3.connect(path) as con:
            con.execute("CREATE TABLE workspace_task_heartbeat (signal_json TEXT)")
            con.execute(
                "INSERT INTO workspace_task_heartbeat VALUES (?)", [signal.model_dump_json()]
            )
        con.close()
        family = owner.heartbeat_family(path)
        with pytest.raises(ValueError, match=f"live_lease_present:{execution.execution_id}"):
            family.step(None)
        assert path.is_file() and owner.read_heartbeat(path, execution.execution_id) is None
        expected = signal
    else:
        path = runtime.parent / "execution" / "spans" / f"{task.task_id}.jsonl"
        path.parent.mkdir(parents=True)
        expected = (
            {
                "task_id": str(task.task_id),
                "execution_id": str(execution.execution_id),
                "stage_id": "resolve_inputs",
                "phase": "execute",
                "started_at": now.isoformat(),
                "wall_seconds": 0.25,
                "spans": [],
            },
        )
        path.write_text("".join(json.dumps(row) + "\n" for row in expected), encoding="utf-8")
        from alphalattice.capabilities.domain_contracts.identity import canonical_hash
        from alphalattice.control.data_platform.task_telemetry import TaskTelemetry
        from alphalattice.control.observation_runtime.telemetry.progress import (
            WorkProgressUpdate,
            WorkspaceProgressPublisher,
        )

        progress = tmp_path / "runtime" / "preparation" / str(task.task_id) / "work-progress.json"
        projection = WorkspaceProgressPublisher(tmp_path / "artifacts").publish(
            WorkProgressUpdate(
                operation_id="fixture",
                stage_id="resolve_inputs",
                status="RUNNING",
                completed_units=3,
                total_units=10,
                unit_name="rows",
            )
        )
        snapshot = {
            "task_id": str(task.task_id),
            "execution_id": str(execution.execution_id),
            "input_hash": task.input.input_hash,
            "stage": "resolve_inputs",
            "projection": projection.model_dump(mode="json"),
        }
        snapshot["content_hash"] = canonical_hash(snapshot)
        TaskTelemetry.replace_json(progress, snapshot)
        family = owner.telemetry_family((path, progress))
    registry.mark_recovery_required(
        task_id=task.task_id, failure_code="fixture.stop", observed_at=now + timedelta(seconds=1)
    )
    with monkeypatch.context() as patch:  # killed after the import, before its set-aside

        def killed(*_paths: object) -> None:
            raise _Killed

        patch.setattr(owner.os, "replace", killed)
        with pytest.raises(_Killed):
            family.step(None)
    assert path.is_file()  # the old file is kept until its import is set aside
    history.close_history()
    family.step(None)
    assert family.verify(None) is True
    assert not path.exists() and (path.parent / RETIRED_DIRECTORY / path.name).is_file()
    if family_name == "task-heartbeats":
        assert TaskHeartbeatReader(runtime).read(execution.execution_id).signals == (expected,)
        clock = [now + timedelta(seconds=31)]
        runner = TaskControlRunner(
            registry=registry, adapters={}, runtime_path=str(runtime), clock=lambda: clock[0]
        )
        state_sql = "SELECT state FROM task WHERE task_id = ?"
        state_before = history.held().read(state_sql, (str(task.task_id),))[0][0]
        # Renewals under load keep the old strict expiry boundary and latest sealed reading.
        for sequence in range(2, 52):
            signal = TaskHeartbeatSignal.from_identity(
                task_id=task.task_id,
                execution_id=execution.execution_id,
                worker_instance_id=worker,
                sequence=sequence,
                observed_at=now + timedelta(seconds=1),
            )
            owner.write_heartbeat(path, signal)
        assert history.held().read(state_sql, (str(task.task_id),))[0][0] == state_before
        assert runner.stale_active_tasks() == ()
        clock[0] += timedelta(microseconds=1)
        assert tuple(row.task_id for row in runner.stale_active_tasks()) == (task.task_id,)
        assert runner.heartbeat_signal(execution.execution_id) == signal
    else:
        assert read_task_telemetry(path, task.task_id) == expected
        assert TaskTelemetry.read_json(progress, task.task_id) == snapshot
        assert not progress.exists()
        later = {key: value for key, value in snapshot.items() if key != "content_hash"}
        later["stage"] = "load"
        later["content_hash"] = canonical_hash(later)
        TaskTelemetry.record_json(progress, later)
        # A copy opens with its Tasks' telemetry where they kept it, the snapshot as its latest.
        copy, publish = tmp_path / "copy", owner.os.replace
        with monkeypatch.context() as patch:  # killed carrying its telemetry: no store yet

            def killed_at_the_telemetry(source: Path, target: Path) -> None:
                if Path(target) == copy / path.relative_to(tmp_path):  # its phase JSONL
                    raise _Killed
                publish(source, target)

            patch.setattr(owner.os, "replace", killed_at_the_telemetry)
            with pytest.raises(_Killed):
                owner.carry_tasks(tmp_path, copy)
        assert not owner._store(copy).exists()
        assert owner.carry_tasks(tmp_path, copy) == 1  # the next carry completes it
        kept = copy / path.relative_to(tmp_path)
        assert [json.loads(line) for line in kept.read_text("utf-8").splitlines()] == list(expected)
        carried = copy / progress.relative_to(tmp_path)
        assert TaskTelemetry.read_json(carried, task.task_id) == later
        # Carried back onto the log that holds its every snapshot, it imports as already held.
        progress.write_bytes(carried.read_bytes())
        owner.import_telemetry((progress,))
        assert not progress.exists() and TaskTelemetry.read_json(progress, task.task_id) == later
    log = history.held()
    log.flush()
    with log.reading() as source:
        rebuilt = replay(source, tmp_path / "replayed.sqlite")
        try:
            assert [
                tuple(row) for row in rebuilt.execute("SELECT * FROM wait ORDER BY wait_id")
            ] == [tuple(row) for row in source.execute("SELECT * FROM wait ORDER BY wait_id")]
        finally:
            rebuilt.close()


@pytest.mark.proves("task_control.a_lost_ledger_rebuilds_only_admitted_work")
def test_a_lost_ledger_lists_its_tasks_again_from_their_frozen_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (LAWS DA2): a 0.2.7 workspace whose store was lost lists each Task again,
    blocked, from its frozen request, which the log then holds byte for byte."""

    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    agent = SubmittingAgent(vendor="claude-code", session="session-1", goal_id=None)
    database = resolve_task_control_database(tmp_path)
    gate = WorkspaceMutationGate()
    registry = DuckDbTaskControlRegistry(database, gate=gate, submitted_by=lambda: agent)
    envelope, goal, plan = task_contract(salt="frozen")
    admitted = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    assert registry.submitted_by(admitted.record.task_id) == agent
    request = _frozen(admitted.record.task_id)
    assert (request.task_id, request.submitted_by, request.plan) == (
        admitted.record.task_id,
        agent,
        plan,
    )

    rebuilt_store = lose_the_log_keeping_frozen_requests(
        registry, gate, monkeypatch, tmp_path / "data"
    )
    (rebuilt,) = rebuilt_store.tasks()
    assert rebuilt.task_id == admitted.record.task_id
    assert rebuilt.lifecycle is TaskLifecycle.BLOCKED
    assert rebuilt.failure_code == "task_control.ledger_rebuilt"
    assert rebuilt_store.safe_projection(rebuilt.task_id).cancel_available is True
    assert rebuilt_store.submitted_by(rebuilt.task_id) == agent
    assert _frozen(rebuilt.task_id) == request  # its frozen bytes moved with it
    runtime = database.parent
    assert not (runtime / "task-requests").exists()  # set aside, never deleted
    assert len(list((runtime / RETIRED_DIRECTORY / "task-requests").glob("*.json"))) == 1
    assert DuckDbTaskControlRegistry(database, gate=gate).tasks() == (rebuilt,)  # once only
    again = rebuilt_store.admit(
        input_envelope=envelope, goal=goal, plan=plan, observed_at=now + timedelta(minutes=3)
    )
    assert not again.duplicate_active and again.record.task_id != rebuilt.task_id
    cancelled, _command = rebuilt_store.request_cancel(
        task_id=rebuilt.task_id,
        expected_task_hash=rebuilt.record_hash,
        observed_at=now + timedelta(minutes=3, seconds=1),
    )
    assert cancelled.lifecycle is TaskLifecycle.CANCELLED
    assert cancelled.failure_code == "task_control.ledger_rebuilt"
    assert rebuilt_store.safe_projection(rebuilt.task_id).cancel_available is False

    # A request that does not read as written is refused by name, never listed.
    retired = next((runtime / RETIRED_DIRECTORY / "task-requests").glob("*.json"))
    (runtime / "task-requests").mkdir()
    (runtime / "task-requests" / ("0" * 64 + ".json")).write_bytes(retired.read_bytes())
    with pytest.raises(ValueError, match="admission_request_unreadable"):
        DuckDbTaskControlRegistry(database, gate=gate)


def _frozen(task_id: UUID) -> TaskAdmissionRequest:
    """A Task's frozen request, as its admission's text on the log."""
    with history.held().reading() as connection:
        (text,) = connection.execute(
            "SELECT t.text FROM event e JOIN event_text t ON t.seq = e.seq"
            " WHERE e.type = 'task.requested' AND e.subject_id = ?",
            [str(task_id)],
        ).fetchone()
    return TaskAdmissionRequest.model_validate_json(text)


@pytest.mark.proves("task_control.a_lost_ledger_rebuilds_only_admitted_work")
@pytest.mark.parametrize("failure_code", ["task_control.ledger_rebuilt", "fixture.blocked"])
def test_only_a_ledger_rebuilt_block_can_close_without_discarding_its_history(
    tmp_path: Path, failure_code: str
) -> None:
    """BADGE regression: version-confirmed closing keeps the blocked reason and all
    stored work, evidence and events; an ordinary block still requires its recovery."""
    now = datetime(2026, 10, 8, 5, tzinfo=UTC)
    database = resolve_task_control_database(tmp_path)
    registry = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    envelope, goal, plan = task_contract(salt="close-blocked")
    admitted = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    started = registry.start_next(
        compatibility=compatibility(plan),
        worker_instance_id=uuid4(),
        observed_at=now + timedelta(seconds=1),
    )
    assert started is not None
    running, execution = started
    complete_stage(
        registry,
        task_id=running.task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        evidence_kind="input_binding",
        now=now + timedelta(seconds=2),
    )
    registry.begin_work_item(
        task_id=running.task_id,
        execution_id=execution.execution_id,
        stage_id="publish_result",
        observed_at=now + timedelta(seconds=5),
    )
    blocked = registry.block_work_item(
        task_id=running.task_id,
        execution_id=execution.execution_id,
        stage_id="publish_result",
        failure_code=failure_code,
        observed_at=now + timedelta(seconds=6),
    )
    board = registry.board(blocked.task_id)
    receipts = registry.stage_receipts(blocked.task_id)
    request = _frozen(blocked.task_id)
    with history.held().reading() as connection:
        journal = connection.execute(
            "SELECT * FROM workspace_task_event WHERE task_id = ? ORDER BY sequence",
            [str(blocked.task_id)],
        ).fetchall()
    can_close = failure_code == "task_control.ledger_rebuilt"
    assert board.projection.cancel_available is can_close
    assert board.projection.verified_stage_count == 1
    with pytest.raises(TaskVersionStale, match="stale task version"):
        registry.request_cancel(
            task_id=blocked.task_id,
            expected_task_hash=admitted.record_hash,
            observed_at=now + timedelta(seconds=7),
        )
    assert registry.task(blocked.task_id) == blocked
    if not can_close:
        with pytest.raises(TaskTransitionRejected, match="terminal task cannot be cancelled"):
            registry.request_cancel(
                task_id=blocked.task_id,
                expected_task_hash=blocked.record_hash,
                observed_at=now + timedelta(seconds=8),
            )
        assert registry.board(blocked.task_id) == board
        return

    closed_at = now + timedelta(seconds=8)
    cancelled, command = registry.request_cancel(
        task_id=blocked.task_id,
        expected_task_hash=blocked.record_hash,
        observed_at=closed_at,
    )
    assert cancelled.lifecycle is TaskLifecycle.CANCELLED
    assert cancelled.failure_code == failure_code
    assert cancelled.active_work_item_id is None
    assert command.expected_task_hash == blocked.record_hash
    repeated, repeated_command = registry.request_cancel(
        task_id=blocked.task_id,
        expected_task_hash=blocked.record_hash,
        observed_at=now + timedelta(seconds=9),
    )
    assert (repeated, repeated_command) == (cancelled, command)
    after = registry.board(blocked.task_id)
    assert after.task == cancelled
    assert after.work_items == board.work_items
    assert after.execution == board.execution
    assert registry.stage_receipts(blocked.task_id) == receipts
    assert _frozen(blocked.task_id) == request
    assert after.projection.cancel_available is False
    assert after.projection.cancel_pending is False
    assert after.projection.verified_stage_count == 1
    assert after.projection.artifact_refs == board.projection.artifact_refs
    with pytest.raises(TaskTransitionRejected, match="does not require recovery"):
        registry.restart_recovery(
            task_id=blocked.task_id,
            compatibility=compatibility(plan),
            worker_instance_id=uuid4(),
            observed_at=now + timedelta(seconds=10),
        )
    with history.held().reading() as connection:
        after_history = connection.execute(
            "SELECT * FROM workspace_task_event WHERE task_id = ? ORDER BY sequence",
            [str(blocked.task_id)],
        ).fetchall()
        commands = connection.execute(
            "SELECT command_hash, resolution, resolved_us FROM workspace_task_command "
            "WHERE task_id = ?",
            [str(blocked.task_id)],
        ).fetchall()
    assert after_history[:-1] == journal
    assert after_history[-1][4] == "task.cancel_requested"
    assert json.loads(after_history[-1][5]) == {
        "command_hash": command.command_hash,
    }
    closed = (closed_at - datetime(1970, 1, 1, tzinfo=UTC)) // timedelta(microseconds=1)
    assert [tuple(row) for row in commands] == [(command.command_hash, "CANCELLED", closed)]


class _Killed(BaseException):
    """The process ends here: nothing after it runs."""


@pytest.mark.proves("task_control.a_killed_import_resumes_with_nothing_lost")
@pytest.mark.parametrize("family_name", ["task-control", "task-heartbeats", "task-telemetry"])
def test_a_kill_between_the_import_and_the_retirement_resumes_with_nothing_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family_name: str
) -> None:
    """An interrupted ledger move resumes ALL EQUAL without lost or duplicate rows."""

    if family_name != "task-control":
        _runtime_ledgers_keep_their_readings_across_an_interrupted_switch(
            tmp_path, monkeypatch, family_name
        )
        return

    import alphalattice.control.task_control.upgrade as owner

    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    gate = WorkspaceMutationGate()
    registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
    # two wait before one starts, whatever the machine's processors give `auto`
    write_queue_setting(registry.database_path.parent, "2", chosen_by="HUMAN", chosen_at=now)
    for salt in ("killed-first", "killed-second"):
        envelope, goal, plan = task_contract(salt=salt)
        registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    started = registry.start_next(
        compatibility=compatibility(plan), worker_instance_id=uuid4(), observed_at=now
    )
    assert started is not None
    running, _ = started
    registry.request_cancel(  # a Task stored past a move its admission never made
        task_id=running.task_id, expected_task_hash=running.record_hash, observed_at=now
    )
    originals = registry.tasks()
    assert {task.lifecycle for task in originals} == {
        TaskLifecycle.QUEUED,
        TaskLifecycle.CANCEL_REQUESTED,
    }
    write_0_2_7_store(registry)

    def killed(*_paths: object) -> None:
        raise _Killed

    monkeypatch.setattr(owner.os, "replace", killed)
    with pytest.raises((_Killed, Exception)) as ended:
        lose_the_log_keeping_frozen_requests(registry, gate, monkeypatch, tmp_path / "data")
    assert ended.type is _Killed, f"the import refused before the kill: {ended.value!r}"
    history.close_history()  # the process is gone; what its unit committed stays
    monkeypatch.undo()
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(tmp_path / "data"))
    assert registry.database_path.is_file()  # not yet set aside

    resumed = DuckDbTaskControlRegistry(registry.database_path, gate=gate)
    assert resumed.tasks() == originals
    with history.held().reading() as connection:
        admitted = connection.execute(
            "SELECT subject_id, count(*) FROM event WHERE type = 'task.requested'"
            " AND workspace_id = ? GROUP BY subject_id",
            [resumed.workspace_id],
        ).fetchall()
    assert sorted(tuple(row) for row in admitted) == sorted(
        (str(task.task_id), 1) for task in originals
    )
    assert not registry.database_path.exists()
    runtime = registry.database_path.parent
    assert (runtime / RETIRED_DIRECTORY / registry.database_path.name).is_file()

    store = registry.database_path  # a later copy's store and WAL, set aside beside the first
    wal = store.with_name(f"{store.name}.wal")
    for path in (store, wal):
        path.write_bytes(path.name.encode())
    real, done = owner.os.replace, []

    def after_the_store(source: object, target: object) -> None:
        if len(done) == 2:  # the set's record, then the store: the WAL is left behind
            raise _Killed
        real(source, target)
        done.append(target)

    monkeypatch.setattr(owner.os, "replace", after_the_store)
    with pytest.raises(_Killed):
        owner.set_aside(runtime, (store, wal, runtime / owner.REQUESTS_DIRECTORY))
    monkeypatch.setattr(owner.os, "replace", real)
    DuckDbTaskControlRegistry(store, gate=gate)
    assert not store.exists() and not wal.exists()
    kept = (runtime / RETIRED_DIRECTORY).iterdir()  # the first set's requests are a directory
    (dated,) = [path for path in kept if path.is_dir() and path.name != owner.REQUESTS_DIRECTORY]
    assert sorted(path.name for path in dated.iterdir()) == sorted((store.name, wal.name))


@pytest.mark.proves("task_control.a_0_2_7_store_moves_in_admission_order")
def test_task_admission_order_survives_equal_clocks_and_reopen(tmp_path, monkeypatch):
    """requirement: Tasks admitted at one clock keep their admission order across a reopen, and a
    0.2.7 store from before admissions were ordered moves onto the log in that order."""

    import alphalattice.control.task_control.registry as owner

    counter = iter(range(1000, 0, -1))
    monkeypatch.setattr(owner, "uuid7", lambda: str(UUID(int=next(counter))))  # ids descend
    now = datetime(2026, 8, 5, 23, tzinfo=UTC)
    path = resolve_task_control_database(tmp_path)
    registry = DuckDbTaskControlRegistry(path, gate=WorkspaceMutationGate())
    expected = []
    for i in range(3):
        envelope, goal, plan = task_contract(salt="same-clock-" + str(i))
        record = registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=now
        ).record
        registry.request_cancel(
            task_id=record.task_id, expected_task_hash=record.record_hash, observed_at=now
        )
        expected.append(record.task_id)
    assert [t.task_id for t in registry.tasks()] == expected
    reopened = DuckDbTaskControlRegistry(path, gate=WorkspaceMutationGate())
    assert [t.task_id for t in reopened.read_existing_tasks(path)] == expected
    # A 0.2.7 store from before admissions were ordered moves onto the log in admission time
    # and id order, every record unchanged.
    originals = sorted(registry.tasks(), key=lambda task: str(task.task_id))
    write_0_2_7_store(registry, ordered=False)
    migrated = lose_the_log_keeping_frozen_requests(
        registry, WorkspaceMutationGate(), monkeypatch, tmp_path / "data"
    )
    assert list(migrated.tasks()) == originals
    monkeypatch.setattr(owner, "uuid7", uuid7)  # the Host's own ids again
    envelope, goal, plan = task_contract(salt="after-legacy-bootstrap")
    added = migrated.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    assert migrated.tasks()[-1] == added
    assert added.task_id.version == 7  # Host-minted, time-ordered


@pytest.mark.proves("task_control.a_lost_ledger_rebuilds_only_admitted_work")
@pytest.mark.parametrize(
    "state",
    (*TaskLifecycle, "RECOVERY_WITH_CANCEL"),
    ids=lambda state: getattr(state, "value", state),
)
def test_new_writer_reconciles_only_orphan_execution_and_accepted_cancel(
    tmp_path: Path, state
) -> None:
    now = datetime.now(UTC)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        envelope, goal, plan = task_contract()
        task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
        if state not in {TaskLifecycle.QUEUED, TaskLifecycle.CANCELLED}:
            task, execution = registry.start_next(
                compatibility=compatibility(plan), worker_instance_id=uuid4(), observed_at=now
            )
            task = complete_stage(
                registry,
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id="resolve_inputs",
                evidence_kind="input_binding",
                now=now,
            )
        if state in {
            TaskLifecycle.CANCELLED,
            TaskLifecycle.CANCEL_REQUESTED,
            "RECOVERY_WITH_CANCEL",
        }:
            task, _ = registry.request_cancel(
                task_id=task.task_id, expected_task_hash=task.record_hash, observed_at=now
            )
            if state == "RECOVERY_WITH_CANCEL":
                task = registry.mark_recovery_required(
                    task_id=task.task_id, failure_code="INTERRUPTED_AFTER_CANCEL", observed_at=now
                )
        elif state is TaskLifecycle.SUCCEEDED:
            task = complete_stage(
                registry,
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id="publish_result",
                evidence_kind="screening_report",
                now=now,
            )
        elif state in {TaskLifecycle.BLOCKED, TaskLifecycle.DEFERRED}:
            registry.begin_work_item(
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id="publish_result",
                observed_at=now,
            )
            task = registry.block_work_item(
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id="publish_result",
                failure_code="WAITING_FOR_AUTHORITY",
                observed_at=now,
                deferred=state is TaskLifecycle.DEFERRED,
            )
        elif state is TaskLifecycle.RECOVERY_REQUIRED:
            task = registry.mark_recovery_required(
                task_id=task.task_id, failure_code="PRIOR_FAILURE", observed_at=now
            )
        elif state is TaskLifecycle.REVIEW_PENDING:
            # Typed stored-state fixtures, including waiting states whose actor
            # protocol is deliberately outside this Task-registry test.
            task = registry._replace_task(task, lifecycle=state, observed_at=now)
            registry._write(lambda connection: registry._save_task(connection, task))
        receipts = registry.stage_receipts(task.task_id)
        items = registry.work_items(task.task_id)
        registry.safe_projection(task.task_id)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        actual = registry.task(task.task_id)
        if state is TaskLifecycle.RUNNING:
            assert actual.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
            with LocalBackgroundDispatcher(registry) as dispatcher:
                assert dispatcher.resume({}) == ()
                assert "recovery_command_not_installed" in dispatcher.failure(task.task_id)
        elif state in {TaskLifecycle.CANCEL_REQUESTED, "RECOVERY_WITH_CANCEL"}:
            assert actual.lifecycle is TaskLifecycle.CANCELLED
            assert actual.failure_code == "TASK_CANCELLED_AFTER_WRITER_STOPPED"
            assert all(
                i.lifecycle in {WorkItemLifecycle.CANCELLED, WorkItemLifecycle.VERIFIED}
                for i in registry.work_items(task.task_id)
            )
        else:
            assert actual == task
            assert registry.work_items(task.task_id) == items
        assert actual.input == task.input and actual.latest_execution_id == task.latest_execution_id
        assert registry.stage_receipts(task.task_id) == receipts
        assert registry.latest_safe_projections()[0].lifecycle is actual.lifecycle
        assert registry.reconcile_after_writer_acquisition(observed_at=now) == ()
        assert registry.task(task.task_id) == actual
