"""A 0.2.7 workspace's Task ledgers moved onto the log: its store, its requests, a store lost."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.task_control.contracts import (
    TaskLifecycle,
    WorkItemLifecycle,
)
from alphalattice.control.task_control.ledger import SubmittingAgent, TaskAdmissionRequest
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    TaskTransitionRejected,
    TaskVersionStale,
    resolve_task_control_database,
)
from alphalattice.control.task_control.upgrade import RETIRED_DIRECTORY
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.kernel.data_layer import history
from tests.workspace_task_runner.task_control_support import (
    compatibility,
    complete_stage,
    lose_the_log_keeping_frozen_requests,
    task_contract,
    write_0_2_7_store,
)

TEST_SOURCE = "src/alphalattice/control/task_control/upgrade.py"


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
            "SELECT command_hash, resolution, resolved_at FROM workspace_task_command "
            "WHERE task_id = ?",
            [str(blocked.task_id)],
        ).fetchall()
    assert after_history[:-1] == journal
    assert after_history[-1][4] == "task.cancel_requested"
    assert json.loads(after_history[-1][5]) == {
        "command_hash": command.command_hash,
    }
    closed = closed_at.replace(tzinfo=None).isoformat(sep=" ", timespec="microseconds")
    assert [tuple(row) for row in commands] == [(command.command_hash, "CANCELLED", closed)]


class _Killed(BaseException):
    """The process ends here: nothing after it runs."""


@pytest.mark.proves("task_control.a_killed_import_resumes_with_nothing_lost")
def test_a_kill_between_the_import_and_the_retirement_resumes_with_nothing_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: a 0.2.7 workspace killed after its Tasks reached the log and before its store
    was set aside opens again with every Task once, verified, and the store set aside."""

    import alphalattice.control.task_control.upgrade as owner

    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    gate = WorkspaceMutationGate()
    registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
    for salt in ("killed-first", "killed-second"):
        envelope, goal, plan = task_contract(salt=salt)
        registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    registry.start_next(
        compatibility=compatibility(plan), worker_instance_id=uuid4(), observed_at=now
    )
    originals = registry.tasks()
    write_0_2_7_store(registry)

    def killed(*_paths: object) -> None:
        raise _Killed

    monkeypatch.setattr(owner.os, "replace", killed)
    with pytest.raises(_Killed):
        lose_the_log_keeping_frozen_requests(registry, gate, monkeypatch, tmp_path / "data")
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


@pytest.mark.proves("task_control.a_0_2_7_store_moves_in_admission_order")
def test_task_admission_order_survives_equal_clocks_and_reopen(tmp_path, monkeypatch):
    """requirement: Tasks admitted at one clock keep their admission order across a reopen, and a
    0.2.7 store from before admissions were ordered moves onto the log in that order."""

    import alphalattice.control.task_control.registry as owner

    counter = iter(range(1000, 0, -1))
    monkeypatch.setattr(owner, "uuid4", lambda: UUID(int=next(counter)))
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
    envelope, goal, plan = task_contract(salt="after-legacy-bootstrap")
    added = migrated.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    assert migrated.tasks()[-1] == added


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
