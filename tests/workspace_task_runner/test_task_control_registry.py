from __future__ import annotations

import json
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock, get_ident
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.task_recovery import (
    build_task_recovery_view,
)
from alphalattice.control.task_control.contracts import (
    FAILURE_CODE_MAX_LENGTH,
    ResearchGoal,
    ResearchPlan,
    StageFailureCause,
    TaskEvidence,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskSafeProjection,
    TaskStageReceipt,
    WorkItemLifecycle,
    WorkItemState,
    failure_code_from,
)
from alphalattice.control.task_control.ledger import SubmittingAgent, read_requests
from alphalattice.control.task_control.queue import (
    read_queue_setting,
    waiting_places,
    write_queue_setting,
)
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    TaskBoardSnapshot,
    TaskQueueFull,
    TaskQueueHeadAuthorityError,
    TaskRecordAuthorityError,
    TaskTransitionRejected,
    TaskVersionStale,
    resolve_task_control_database,
)
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
    TaskControlRunner,
    TaskHeartbeatReadout,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application.dispatcher import (
    CANCEL_REAIM_LIMIT,
    CommandAdmission,
    LocalBackgroundDispatcher,
    TaskVersionMovedError,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.workspace_task_runner.task_control_support import (
    FixtureTaskAdapter as _FixtureTaskAdapter,
)
from tests.workspace_task_runner.task_control_support import (
    compatibility,
    digest,
    task_contract,
    work_item,
)

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def _complete_stage(
    registry: DuckDbTaskControlRegistry,
    *,
    task_id,
    execution_id,
    stage_id: str,
    evidence_kind: str,
    now: datetime,
):
    item = registry.begin_work_item(
        task_id=task_id,
        execution_id=execution_id,
        stage_id=stage_id,
        observed_at=now,
    )
    evidence = (
        TaskEvidence(
            evidence_kind=evidence_kind,
            reference=f"playpen://task-evidence/{stage_id}",
            content_hash=digest(stage_id),
        ),
    )
    registry.mark_ready(
        task_id=task_id,
        execution_id=execution_id,
        stage_id=stage_id,
        evidence=evidence,
        observed_at=now + timedelta(seconds=1),
    )
    definition = next(
        value for value in registry.task(task_id).plan.work_items if value.stage_id == stage_id
    )
    return registry.verify_work_item(
        TaskStageReceipt.from_identity(
            receipt_id=uuid4(),
            task_id=task_id,
            execution_id=execution_id,
            stage_id=stage_id,
            work_item_definition_hash=item.definition_hash,
            verifier_id=definition.verifier_id,
            evidence=evidence,
            status="VERIFIED",
            failure_code=None,
            observed_at=now + timedelta(seconds=2),
        )
    )


def test_registry_owns_queue_work_board_receipts_and_projection(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 14, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract()
    admission = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    duplicate = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    assert duplicate.duplicate_active is True
    assert duplicate.record.task_id == admission.record.task_id

    worker_id = uuid4()
    with pytest.raises(TaskTransitionRejected, match="queued_task_identity_mismatch"):
        registry.start_next(
            compatibility=compatibility(plan),
            worker_instance_id=worker_id,
            observed_at=now,
            expected_task_id=uuid4(),
        )
    assert registry.task(admission.record.task_id) == admission.record
    runner = TaskControlRunner(
        registry=registry, adapters={}, runtime_path=str(tmp_path / "checkpoints.sqlite")
    )
    try:
        with pytest.raises(ValueError, match="queued_task_identity_mismatch"):
            runner.run_next(expected_task_id=uuid4())
    finally:
        runner.close()

    assert registry.task(admission.record.task_id) == admission.record
    started = registry.start_next(
        compatibility=compatibility(plan),
        worker_instance_id=worker_id,
        observed_at=now + timedelta(seconds=1),
        expected_task_id=admission.record.task_id,
    )
    assert started is not None
    task, execution = started
    assert task.lifecycle is TaskLifecycle.RUNNING
    with pytest.raises(TaskTransitionRejected, match="dependency"):
        registry.begin_work_item(
            task_id=task.task_id,
            execution_id=execution.execution_id,
            stage_id="publish_result",
            observed_at=now + timedelta(seconds=2),
        )

    task = _complete_stage(
        registry,
        task_id=task.task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        evidence_kind="input_binding",
        now=now + timedelta(seconds=3),
    )
    assert task.lifecycle is TaskLifecycle.RUNNING
    task = _complete_stage(
        registry,
        task_id=task.task_id,
        execution_id=execution.execution_id,
        stage_id="publish_result",
        evidence_kind="screening_report",
        now=now + timedelta(seconds=6),
    )
    assert task.lifecycle is TaskLifecycle.SUCCEEDED
    assert all(
        item.lifecycle is WorkItemLifecycle.VERIFIED for item in registry.work_items(task.task_id)
    )
    projection = registry.safe_projection(task.task_id)
    assert projection.verified_stage_count == projection.total_stage_count == 2
    assert projection.cancel_available is False
    assert len(registry.stage_receipts(task.task_id)) == 2
    with pytest.raises(TaskTransitionRejected, match="terminal task cannot be cancelled"):
        registry.request_cancel(
            task_id=task.task_id,
            expected_task_hash=task.record_hash,
            observed_at=now + timedelta(seconds=10),
        )


def test_projection_collection_keeps_queue_head_failure_out_of_requested_task_refusal(
    tmp_path: Path,
) -> None:
    """A broken queued head is a queue authority dependency, not the requested cache miss."""
    import duckdb

    now = datetime(2026, 8, 5, 14, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    head_envelope, head_goal, head_plan = task_contract(salt="broken-queue-head")
    head = registry.admit(
        input_envelope=head_envelope,
        goal=head_goal,
        plan=head_plan,
        observed_at=now,
    ).record
    peer_envelope, peer_goal, peer_plan = task_contract(salt="healthy-cache-miss")
    peer = registry.admit(
        input_envelope=peer_envelope,
        goal=peer_goal,
        plan=peer_plan,
        observed_at=now + timedelta(seconds=1),
    ).record
    with duckdb.connect(str(registry.database_path)) as connection:
        connection.execute(
            "UPDATE workspace_task SET record_json = '{' WHERE task_id = ?",
            [str(head.task_id)],
        )
        connection.execute(
            "DELETE FROM workspace_task_projection WHERE task_id = ?", [str(peer.task_id)]
        )

    with pytest.raises(
        TaskQueueHeadAuthorityError, match="database_authority_unreadable"
    ) as raised:
        registry.projection_collection([peer.task_id])
    assert raised.value.task_id == str(head.task_id)


@pytest.mark.parametrize("damage", ["bad-json", "different-canonical-id"])
def test_canonical_task_collection_keeps_peers_but_complete_authority_refuses(
    tmp_path: Path, damage: str
) -> None:
    """V661: damaged canonical authority is named, never discarded as an absent Task."""
    import duckdb

    now = datetime(2026, 10, 4, 6, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    write_queue_setting(tmp_path / "runtime", "4", chosen_by="HUMAN", chosen_at=now)
    records = []
    for index in range(3):
        envelope, goal, plan = task_contract(salt=f"canonical-collection-{index}")
        records.append(
            registry.admit(
                input_envelope=envelope,
                goal=goal,
                plan=plan,
                observed_at=now + timedelta(seconds=index),
            ).record
        )
    first, damaged, last = records
    # Warm the complete answer too: an external record change must invalidate it.
    assert registry.tasks() == tuple(records)
    assert registry.task(damaged.task_id) == damaged
    assert registry.task_with_work_items(damaged.task_id)[0] == damaged
    alien = TaskRecord.from_identity(
        **{**damaged.model_dump(exclude={"record_hash"}), "task_id": uuid4()}
    )
    replacement = "{" if damage == "bad-json" else alien.model_dump_json()
    with duckdb.connect(str(registry.database_path)) as connection:
        connection.execute(
            "UPDATE workspace_task SET record_json = ? WHERE task_id = ?",
            [replacement, str(damaged.task_id)],
        )
        before = connection.execute(
            "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
        ).fetchall()

    batch = registry.record_collection()
    assert batch.records == (first, last)
    assert batch.refused_task_ids == (str(damaged.task_id),)
    assert str(alien.task_id) not in batch.refused_task_ids
    for read in (
        registry.tasks,
        lambda: registry.task(damaged.task_id),
        lambda: registry.task_with_work_items(damaged.task_id),
        lambda: registry.read_existing_tasks(registry.database_path),
        registry.active_task,
        lambda: registry.stale_active_tasks(observed_at=now + timedelta(hours=1)),
    ):
        with pytest.raises(
            TaskRecordAuthorityError, match="database_authority_unreadable"
        ) as raised:
            read()
        assert raised.value.refused_task_ids == (str(damaged.task_id),)
    # A confirmation for the stored Task never cancels an alien canonical identity.
    with pytest.raises(TaskRecordAuthorityError, match="database_authority_unreadable") as raised:
        registry.request_cancel(
            task_id=damaged.task_id,
            expected_task_hash=damaged.record_hash,
            observed_at=now + timedelta(minutes=1),
        )
    assert raised.value.refused_task_ids == (str(damaged.task_id),)
    with duckdb.connect(str(registry.database_path), read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
            ).fetchall()
            == before
        )


@pytest.mark.parametrize("damage", ["bad-json", "different-canonical-id", "terminal-missing-hash"])
def test_canonical_task_damage_refuses_admission_start_and_writer_reconciliation(
    tmp_path: Path, damage: str
) -> None:
    """V661: filtered SQL consumers keep the same stored-identity authority boundary."""
    import duckdb

    now = datetime(2026, 10, 4, 6, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    write_queue_setting(tmp_path / "runtime", "4", chosen_by="HUMAN", chosen_at=now)
    envelope, goal, plan = task_contract(salt="damaged-queued-authority")
    damaged = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    peer_envelope, peer_goal, peer_plan = task_contract(salt="readable-queued-authority")
    peer = registry.admit(
        input_envelope=peer_envelope,
        goal=peer_goal,
        plan=peer_plan,
        observed_at=now + timedelta(seconds=1),
    ).record
    if damage == "terminal-missing-hash":
        damaged, _command = registry.request_cancel(
            task_id=damaged.task_id,
            expected_task_hash=damaged.record_hash,
            observed_at=now + timedelta(seconds=2),
        )
        assert damaged.lifecycle is TaskLifecycle.CANCELLED
        assert registry.safe_projection(damaged.task_id).task_record_hash == damaged.record_hash
    alien = TaskRecord.from_identity(
        **{**damaged.model_dump(exclude={"record_hash"}), "task_id": uuid4()}
    )
    replacement = {
        "bad-json": "{",
        "different-canonical-id": alien.model_dump_json(),
        "terminal-missing-hash": "{}",
    }[damage]
    tables = (
        "workspace_task",
        "workspace_task_work_item",
        "workspace_task_execution",
        "workspace_task_command",
        "workspace_task_event",
        "workspace_task_projection",
        "workspace_task_stage_receipt",
    )
    with duckdb.connect(str(registry.database_path)) as connection:
        if damage == "terminal-missing-hash":
            # An existing current projection must not hide a missing canonical hash.
            assert connection.execute(
                "SELECT task_record_hash FROM workspace_task_projection WHERE task_id = ?",
                [str(damaged.task_id)],
            ).fetchone() == (damaged.record_hash,)
        connection.execute(
            "UPDATE workspace_task SET record_json = ? WHERE task_id = ?",
            [replacement, str(damaged.task_id)],
        )
        before = {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
            for table in tables
        }

    # Duplicate admission, version-confirmed start and writer reconciliation all read
    # filtered canonical rows directly. The legacy start has the same identity obligation.
    actions = (
        lambda: registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now),
        lambda: registry.start_next(
            compatibility=compatibility(plan),
            worker_instance_id=uuid4(),
            observed_at=now + timedelta(seconds=2),
            expected_task_id=damaged.task_id,
            expected_task_hash=damaged.record_hash,
        ),
        lambda: registry.start_next(
            compatibility=compatibility(plan),
            worker_instance_id=uuid4(),
            observed_at=now + timedelta(seconds=2),
        ),
        lambda: registry.reconcile_after_writer_acquisition(observed_at=now + timedelta(seconds=2)),
    )
    if damage == "terminal-missing-hash":
        # Terminal rows are outside duplicate admission/start; reconciliation owns their hash.
        actions = actions[-1:]
    for action in actions:
        with pytest.raises(
            TaskRecordAuthorityError, match="database_authority_unreadable"
        ) as raised:
            action()
        assert raised.value.refused_task_ids == (str(damaged.task_id),)
        assert str(alien.task_id) not in raised.value.refused_task_ids
    assert registry.record_collection().records == (peer,)
    with duckdb.connect(str(registry.database_path), read_only=True) as connection:
        assert {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
            for table in tables
        } == before


def test_canonical_task_collection_does_not_turn_a_store_failure_into_empty_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """V661: row isolation does not make an unreadable whole store look empty."""
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )

    def unavailable(_path, *, read_only):
        raise OSError("the Task store cannot be read")

    monkeypatch.setattr(
        "alphalattice.control.task_control.registry.open_workspace_database", unavailable
    )
    with pytest.raises(OSError, match="Task store cannot be read"):
        registry.record_collection()


def test_board_snapshot_keeps_many_references_and_a_confirmed_version_binds_at_the_start(
    tmp_path: Path,
) -> None:
    """Board snapshot keeps many references and a confirmed version binds at the start."""

    now = datetime(2026, 9, 15, 9, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="seventeen")
    task_id = registry.admit(
        input_envelope=envelope, goal=goal, plan=plan, observed_at=now
    ).record.task_id
    worker_id = uuid4()
    started = registry.start_next(
        compatibility=compatibility(plan),
        worker_instance_id=worker_id,
        observed_at=now,
        expected_task_id=task_id,
    )
    assert started is not None
    _task, execution = started
    item = registry.begin_work_item(
        task_id=task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        observed_at=now + timedelta(seconds=1),
    )
    evidence = tuple(
        TaskEvidence(
            evidence_kind="input_binding",
            reference=f"playpen://task-evidence/resolve_inputs/{index:02d}",
            content_hash=digest(f"resolve_inputs-{index}"),
        )
        for index in range(17)
    )
    registry.mark_ready(
        task_id=task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        evidence=evidence,
        observed_at=now + timedelta(seconds=2),
    )
    definition = next(v for v in plan.work_items if v.stage_id == "resolve_inputs")
    registry.verify_work_item(
        TaskStageReceipt.from_identity(
            receipt_id=uuid4(),
            task_id=task_id,
            execution_id=execution.execution_id,
            stage_id="resolve_inputs",
            work_item_definition_hash=item.definition_hash,
            verifier_id=definition.verifier_id,
            evidence=evidence,
            status="VERIFIED",
            failure_code=None,
            observed_at=now + timedelta(seconds=3),
        )
    )
    interrupted = registry.mark_recovery_required(
        task_id=task_id,
        failure_code="TASK_EXECUTION_INTERRUPTED",
        observed_at=now + timedelta(seconds=4),
    )
    version_a = interrupted.record_hash

    snapshot = registry.board(task_id)
    assert isinstance(snapshot, TaskBoardSnapshot)
    assert snapshot.task.record_hash == snapshot.projection.task_record_hash == version_a
    assert snapshot.execution is not None and snapshot.execution.task_id == task_id
    assert len(snapshot.projection.artifact_refs) == 16, "the projection's own contract"
    (verified,) = [v for v in snapshot.work_items if v.lifecycle is WorkItemLifecycle.VERIFIED]
    assert len(verified.evidence) == 17
    view = build_task_recovery_view(
        snapshot=snapshot,
        running=False,
        worker_failure=None,
        heartbeats=TaskHeartbeatReadout(signals=(), unreadable=()),
        recoverable_kinds=frozenset({"factor_research"}),
        observed_at=now + timedelta(seconds=5),
        replans={},
    )
    stage = next(s for s in view.stages if s.stage_id == "resolve_inputs")
    assert stage.evidence_count == len(stage.evidence) == 17
    assert set(view.artifact_refs) < set(stage.evidence) and len(view.artifact_refs) == 16
    assert view.verified_stage_count == 1 and view.lifecycle == "RECOVERY_REQUIRED"
    assert view.execution_binding_hash == execution.compatibility.compatibility_hash
    assert view.guardian is None and view.model_facts == "NOT_OBSERVED"
    assert view.task_record_hash == version_a
    # A projection carrying seventeen references, as another owner may publish it, builds too.
    wide = snapshot.projection.model_copy(update={"artifact_refs": tuple(stage.evidence)})
    wide = TaskSafeProjection.from_identity(
        **wide.model_dump(exclude={"projection_hash"}),
    )
    assert len(wide.artifact_refs) == 17
    widened = build_task_recovery_view(
        snapshot=TaskBoardSnapshot(
            task=snapshot.task,
            work_items=snapshot.work_items,
            execution=snapshot.execution,
            projection=wide,
        ),
        running=False,
        worker_failure=None,
        heartbeats=TaskHeartbeatReadout(signals=(), unreadable=()),
        recoverable_kinds=frozenset(),
        observed_at=now + timedelta(seconds=5),
        replans={},
    )
    assert len(widened.artifact_refs) == 17

    # The Task moves (another actor's versionless recovery starts it); a recovery confirmed
    # against A is refused at the transaction, and so is a cancel confirmed against A.
    moved, _execution = registry.restart_recovery(
        task_id=task_id,
        compatibility=compatibility(plan),
        worker_instance_id=worker_id,
        observed_at=now + timedelta(seconds=6),
    )
    assert moved.lifecycle is TaskLifecycle.RUNNING and moved.record_hash != version_a
    registry.mark_recovery_required(
        task_id=task_id,
        failure_code="TASK_EXECUTION_INTERRUPTED",
        observed_at=now + timedelta(seconds=7),
    )
    version_b = registry.task(task_id).record_hash
    with pytest.raises(TaskVersionStale, match=r"task_control.recovery_version_stale"):
        registry.restart_recovery(
            task_id=task_id,
            compatibility=compatibility(plan),
            worker_instance_id=worker_id,
            observed_at=now + timedelta(seconds=8),
            expected_task_hash=version_a,
        )
    assert registry.task(task_id).record_hash == version_b, "nothing was started"
    with pytest.raises(TaskVersionStale, match="stale task version"):
        registry.request_cancel(
            task_id=task_id, expected_task_hash=version_a, observed_at=now + timedelta(seconds=8)
        )
    assert registry.task(task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
    resumed, _execution = registry.restart_recovery(
        task_id=task_id,
        compatibility=compatibility(plan),
        worker_instance_id=worker_id,
        observed_at=now + timedelta(seconds=9),
        expected_task_hash=version_b,
    )
    assert resumed.lifecycle is TaskLifecycle.RUNNING
    kept = next(v for v in registry.work_items(task_id) if v.stage_id == "resolve_inputs")
    assert kept.lifecycle is WorkItemLifecycle.VERIFIED and len(kept.evidence) == 17

    # A queued Task: a start confirmed against a version it has left is refused the same way.
    envelope2, goal2, plan2 = task_contract(salt="queued-behind")
    queued = registry.admit(
        input_envelope=envelope2, goal=goal2, plan=plan2, observed_at=now + timedelta(seconds=10)
    ).record
    cancelled = registry.request_cancel(
        task_id=queued.task_id,
        expected_task_hash=queued.record_hash,
        observed_at=now + timedelta(seconds=11),
    )[0]
    assert cancelled.lifecycle is TaskLifecycle.CANCELLED
    envelope3, goal3, plan3 = task_contract(salt="queued-next")
    third = registry.admit(
        input_envelope=envelope3, goal=goal3, plan=plan3, observed_at=now + timedelta(seconds=12)
    ).record
    registry.mark_recovery_required(
        task_id=task_id,
        failure_code="TASK_EXECUTION_INTERRUPTED",
        observed_at=now + timedelta(seconds=13),
    )
    registry.request_cancel(
        task_id=task_id,
        expected_task_hash=registry.task(task_id).record_hash,
        observed_at=now + timedelta(seconds=14),
    )
    registry.apply_cancel(task_id=task_id, observed_at=now + timedelta(seconds=15))
    with pytest.raises(TaskVersionStale, match=r"task_control.start_version_stale"):
        registry.start_next(
            compatibility=compatibility(plan3),
            worker_instance_id=worker_id,
            observed_at=now + timedelta(seconds=16),
            expected_task_id=third.task_id,
            expected_task_hash="0" * 64,
        )
    assert registry.task(third.task_id).lifecycle is TaskLifecycle.QUEUED
    started_third = registry.start_next(
        compatibility=compatibility(plan3),
        worker_instance_id=worker_id,
        observed_at=now + timedelta(seconds=17),
        expected_task_id=third.task_id,
        expected_task_hash=third.record_hash,
    )
    assert started_third is not None and started_third[0].lifecycle is TaskLifecycle.RUNNING


def test_the_queue_keeps_the_operators_places_and_a_recovery_holds_no_running_place(
    tmp_path: Path,
) -> None:
    """requirement (V100, LAWS PA2): how many Tasks wait is the operator's setting, a full queue
    is refused before a request's planning work, and a Task waiting for its recovery lets the
    queue's head run while a recovery asked for meanwhile is refused by name."""

    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    assert read_queue_setting(tmp_path / "runtime").tasks_waiting == "auto"
    assert waiting_places(read_queue_setting(tmp_path / "runtime"))[0] >= 1
    with pytest.raises(ValueError, match="tasks_waiting_invalid"):
        write_queue_setting(tmp_path / "runtime", 0, chosen_by="HUMAN", chosen_at=now)
    write_queue_setting(tmp_path / "runtime", "2", chosen_by="HUMAN", chosen_at=now)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    contracts = [task_contract(salt=name) for name in ("a", "b", "c", "d")]
    admitted = [
        registry.admit(
            input_envelope=contracts[0][0],
            goal=contracts[0][1],
            plan=contracts[0][2],
            observed_at=now,
        ).record
    ]
    first = registry.start_next(
        compatibility=compatibility(contracts[0][2]),
        worker_instance_id=uuid4(),
        observed_at=now + timedelta(seconds=1),
    )
    assert first is not None
    for i, (envelope, goal, plan) in enumerate(contracts[1:3], start=2):
        at = now + timedelta(seconds=i)
        admitted.append(
            registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=at).record
        )
    with pytest.raises(TaskQueueFull, match=r"2 Tasks wait in the queue's 2 places \(set to 2\)"):
        registry.check_capacity()
    with pytest.raises(TaskQueueFull, match="tasks_waiting"):
        registry.admit(
            input_envelope=contracts[3][0],
            goal=contracts[3][1],
            plan=contracts[3][2],
            observed_at=now + timedelta(seconds=4),
        )
    assert registry.queued_task() == registry.task(admitted[1].task_id)  # the queue's head

    # The Host that held the first Task stopped; the next one finds it owed a recovery.
    (interrupted,) = registry.reconcile_after_writer_acquisition(
        observed_at=now + timedelta(seconds=5)
    )
    assert interrupted.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
    second = registry.start_next(
        compatibility=compatibility(contracts[1][2]),
        worker_instance_id=uuid4(),
        observed_at=now + timedelta(seconds=6),
    )
    assert second is not None and second[0].task_id == admitted[1].task_id
    assert registry.active_task() == registry.task(admitted[1].task_id)
    with pytest.raises(TaskTransitionRejected, match="recovery_waits_for_the_running_task"):
        registry.restart_recovery(
            task_id=interrupted.task_id,
            compatibility=compatibility(contracts[0][2]),
            worker_instance_id=uuid4(),
            observed_at=now + timedelta(seconds=7),
        )
    registry.check_capacity()  # one Task waits in two places


def test_a_lost_ledger_lists_its_tasks_again_from_their_frozen_requests(tmp_path: Path) -> None:
    """requirement (V181, U33, LAWS DA2): each admission's request, with the session that
    submitted it, is written before its row; a store lost and made again lists the Task from it,
    blocked and not claimed done, and the same request is admitted anew."""

    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    agent = SubmittingAgent(vendor="claude-code", session="session-1", goal_id=None)
    database = resolve_task_control_database(tmp_path)
    registry = DuckDbTaskControlRegistry(
        database, gate=WorkspaceMutationGate(), submitted_by=lambda: agent
    )
    envelope, goal, plan = task_contract(salt="frozen")
    admitted = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    assert registry.submitted_by(admitted.record.task_id) == agent
    (request,) = read_requests(tmp_path / "runtime")
    assert (request.task_id, request.submitted_by, request.plan) == (
        admitted.record.task_id,
        agent,
        plan,
    )
    assert registry.rebuild_from_requests(observed_at=now) == ()  # nothing was lost

    for lost in database.parent.glob(f"{database.name}*"):
        lost.unlink()
    rebuilt_store = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    (rebuilt,) = rebuilt_store.rebuild_from_requests(observed_at=now + timedelta(minutes=1))
    assert rebuilt.task_id == admitted.record.task_id
    assert rebuilt.lifecycle is TaskLifecycle.BLOCKED
    assert rebuilt.failure_code == "task_control.ledger_rebuilt"
    assert rebuilt_store.safe_projection(rebuilt.task_id).cancel_available is True
    assert rebuilt_store.tasks() == (rebuilt,)
    assert rebuilt_store.submitted_by(rebuilt.task_id) == agent
    assert rebuilt_store.rebuild_from_requests(observed_at=now + timedelta(minutes=2)) == ()
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
    moved = next((tmp_path / "runtime" / "task-requests").glob("*.json"))
    moved.rename(moved.with_name("0" * 64 + ".json"))
    with pytest.raises(ValueError, match="admission_request_unreadable"):
        rebuilt_store.rebuild_from_requests(observed_at=now + timedelta(minutes=4))


@pytest.mark.parametrize("failure_code", ["task_control.ledger_rebuilt", "fixture.blocked"])
def test_only_a_ledger_rebuilt_block_can_close_without_discarding_its_history(
    tmp_path: Path, failure_code: str
) -> None:
    """BADGE regression: version-confirmed closing keeps the blocked reason and all
    stored work, evidence and events; an ordinary block still requires its recovery."""
    import duckdb

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
    _complete_stage(
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
    (request,) = read_requests(database.parent)
    with duckdb.connect(str(database), read_only=True) as connection:
        history = connection.execute(
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
    assert read_requests(database.parent) == (request,)
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
    with duckdb.connect(str(database), read_only=True) as connection:
        after_history = connection.execute(
            "SELECT * FROM workspace_task_event WHERE task_id = ? ORDER BY sequence",
            [str(blocked.task_id)],
        ).fetchall()
        commands = connection.execute(
            "SELECT command_hash, resolution, resolved_at FROM workspace_task_command "
            "WHERE task_id = ?",
            [str(blocked.task_id)],
        ).fetchall()
    assert after_history[:-1] == history
    assert after_history[-1][4] == "task.cancel_requested"
    assert json.loads(after_history[-1][5]) == {
        "command_hash": command.command_hash,
    }
    assert commands == [(command.command_hash, "CANCELLED", closed_at.replace(tzinfo=None))]


def test_registry_enforces_one_active_one_queued_and_safe_cancel(tmp_path: Path) -> None:
    now = datetime(2026, 8, 5, 15, tzinfo=UTC)
    write_queue_setting(tmp_path / "runtime", 1, chosen_by="HUMAN", chosen_at=now)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    first = task_contract(salt="first")
    registry.admit(input_envelope=first[0], goal=first[1], plan=first[2], observed_at=now)
    started = registry.start_next(
        compatibility=compatibility(first[2]),
        worker_instance_id=uuid4(),
        observed_at=now + timedelta(seconds=1),
    )
    assert started is not None
    second = task_contract(salt="second")
    queued = registry.admit(
        input_envelope=second[0],
        goal=second[1],
        plan=second[2],
        observed_at=now + timedelta(seconds=2),
    ).record
    third = task_contract(salt="third")
    with pytest.raises(TaskQueueFull, match=r"queue_full: 1 Tasks wait in the queue's 1 places"):
        registry.admit(
            input_envelope=third[0],
            goal=third[1],
            plan=third[2],
            observed_at=now + timedelta(seconds=3),
        )
    with pytest.raises(TaskTransitionRejected, match="stale"):
        registry.request_cancel(
            task_id=queued.task_id,
            expected_task_hash="0" * 64,
            observed_at=now + timedelta(seconds=4),
        )
    cancelled, command = registry.request_cancel(
        task_id=queued.task_id,
        expected_task_hash=queued.record_hash,
        observed_at=now + timedelta(seconds=5),
    )
    assert cancelled.lifecycle is TaskLifecycle.CANCELLED
    repeated, repeated_command = registry.request_cancel(
        task_id=queued.task_id,
        expected_task_hash=cancelled.record_hash,
        observed_at=now + timedelta(seconds=6),
    )
    assert repeated.lifecycle is TaskLifecycle.CANCELLED
    assert repeated_command.command_hash == command.command_hash


def test_status_and_cancel_stay_responsive_without_unfencing_admission(tmp_path):
    gate = WorkspaceMutationGate()
    registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
    other_reader = DuckDbTaskControlRegistry(registry.database_path, gate=gate)
    first = task_contract()
    now = datetime.now(UTC)
    registry.admit(input_envelope=first[0], goal=first[1], plan=first[2], observed_at=now)
    running, _ = registry.start_next(
        compatibility=compatibility(first[2]), worker_instance_id=uuid4(), observed_at=now
    )
    before = registry.safe_projection(running.task_id)
    held, release = Event(), Event()

    def data_transaction():
        with gate.hold():
            held.set()
            assert release.wait(10)

    with ThreadPoolExecutor(max_workers=3) as pool:
        writer = pool.submit(data_transaction)
        assert held.wait(2)
        try:
            assert (
                pool.submit(other_reader.safe_projection, running.task_id).result(timeout=2)
                == before
            )
            cancelled, _ = pool.submit(
                registry.request_cancel,
                task_id=running.task_id,
                expected_task_hash=running.record_hash,
                observed_at=now,
            ).result(timeout=2)
            assert cancelled.lifecycle is TaskLifecycle.CANCEL_REQUESTED
            next_input, next_goal, next_plan = task_contract(salt="next")
            admission = pool.submit(
                registry.admit,
                input_envelope=next_input,
                goal=next_goal,
                plan=next_plan,
                observed_at=now,
            )
            with pytest.raises(FutureTimeout):
                admission.result(timeout=0.2)
        finally:
            release.set()
        writer.result(timeout=2)
        assert admission.result(timeout=2).record.lifecycle is TaskLifecycle.QUEUED
    assert (
        registry.apply_cancel(task_id=running.task_id, observed_at=now).lifecycle
        is TaskLifecycle.CANCELLED
    )


def test_a_read_only_unit_never_holds_the_instance_a_starting_writer_waits_for(tmp_path):
    """A read only unit never holds the instance a starting writer waits for."""

    gate = WorkspaceMutationGate()
    registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
    first = task_contract()
    now = datetime.now(UTC)
    registry.admit(input_envelope=first[0], goal=first[1], plan=first[2], observed_at=now)
    inside, go_on = Event(), Event()

    def status_read():
        with registry.retain(read_only=True):
            inside.set()
            assert go_on.wait(10)
            return registry.tasks()

    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(status_read)
        assert inside.wait(2)
        writer = pool.submit(
            registry.start_next,
            compatibility=compatibility(first[2]),
            worker_instance_id=uuid4(),
            observed_at=now,
        )
        with pytest.raises(FutureTimeout):
            writer.result(timeout=0.3)  # it waits for the unit, as it should
        go_on.set()
        assert len(reader.result(timeout=5)) == 1
        running, _ = writer.result(timeout=5)
    assert running.lifecycle is TaskLifecycle.RUNNING


def test_registry_rejects_missing_required_evidence_and_stale_heartbeat(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 5, 16, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract()
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    worker_id = uuid4()
    started = registry.start_next(
        compatibility=compatibility(plan),
        worker_instance_id=worker_id,
        observed_at=now + timedelta(seconds=1),
    )
    assert started is not None
    _, execution = started
    item = registry.begin_work_item(
        task_id=task.task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        observed_at=now + timedelta(seconds=2),
    )
    wrong = (
        TaskEvidence(
            evidence_kind="wrong_kind",
            reference="playpen://task-evidence/wrong",
            content_hash=digest("wrong"),
        ),
    )
    registry.mark_ready(
        task_id=task.task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        evidence=wrong,
        observed_at=now + timedelta(seconds=3),
    )
    with pytest.raises(TaskTransitionRejected, match="required evidence"):
        registry.verify_work_item(
            TaskStageReceipt.from_identity(
                receipt_id=uuid4(),
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id="resolve_inputs",
                work_item_definition_hash=item.definition_hash,
                verifier_id="task-control.resolve_inputs",
                evidence=wrong,
                status="VERIFIED",
                failure_code=None,
                observed_at=now + timedelta(seconds=4),
            )
        )
    with pytest.raises(TaskTransitionRejected, match="worker"):
        registry.heartbeat(
            execution_id=execution.execution_id,
            worker_instance_id=uuid4(),
            observed_at=now + timedelta(seconds=5),
        )


class _InterruptedBeforeReceiptAdapter(_FixtureTaskAdapter):
    """The process stops between a stage's stored evidence and its receipt (V101)."""

    def __init__(self, *, accepts_at_recovery: bool) -> None:
        super().__init__()
        self.accepts_at_recovery = accepts_at_recovery
        self.verifications = 0

    def verify_stage(self, *, task, execution, work_item, evidence):
        if work_item.stage_id == "resolve_inputs":
            self.verifications += 1
            if self.verifications == 1:
                raise RuntimeError("simulated process interruption")
            if self.verifications == 2 and not self.accepts_at_recovery:
                raise ValueError("fixture.evidence_moved")
        return evidence


class _CancellingTaskAdapter(_FixtureTaskAdapter):
    def __init__(self, *, registry, observed_at: datetime, ready: bool = False) -> None:
        super().__init__()
        self.registry = registry
        self.observed_at = observed_at
        self.ready = ready

    def execute_stage(self, *, task, execution, work_item):
        current = self.registry.task(task.task_id)
        self.registry.request_cancel(
            task_id=task.task_id,
            expected_task_hash=current.record_hash,
            observed_at=self.observed_at,
        )
        if self.ready:
            return super().execute_stage(task=task, execution=execution, work_item=work_item)
        return StageExecutionResult(disposition=StageDisposition.CANCELLED)


class _AuthorityFailure(ValueError):
    failure_class = "AUTHORITY_FAILURE"


class _AuthorityVerifyTaskAdapter(_FixtureTaskAdapter):
    def verify_stage(self, *, task, execution, work_item, evidence):
        del task, execution, work_item, evidence
        raise _AuthorityFailure("FIXTURE_EVIDENCE_AUTHORITY_INVALID")


def test_runner_preserves_typed_failure_class_from_verification(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, 18, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="typed-authority-failure")
    task = registry.admit(
        input_envelope=envelope,
        goal=goal,
        plan=plan,
        observed_at=now,
    ).record
    adapter = _AuthorityVerifyTaskAdapter()
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now + timedelta(seconds=1),
        heartbeat_seconds=0.01,
    )
    try:
        with pytest.raises(_AuthorityFailure, match="FIXTURE_EVIDENCE_AUTHORITY_INVALID"):
            runner.run_next()
    finally:
        runner.close()

    failed = registry.task(task.task_id)
    assert failed.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
    assert failed.failure_code == "AUTHORITY_FAILURE:FIXTURE_EVIDENCE_AUTHORITY_INVALID"


@pytest.mark.parametrize("ready", [False, True])
def test_runner_applies_adapter_cancellation_at_internal_safe_checkpoint(
    tmp_path: Path,
    ready: bool,
) -> None:
    now = datetime(2026, 8, 10, 17, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="internal-safe-cancel")
    registry.admit(
        input_envelope=envelope,
        goal=goal,
        plan=plan,
        observed_at=now,
    )
    adapter = _CancellingTaskAdapter(
        registry=registry, observed_at=now + timedelta(seconds=1), ready=ready
    )
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now + timedelta(seconds=2),
        heartbeat_seconds=0.01,
    )
    try:
        result = runner.run_next()
    finally:
        runner.close()
    assert result is not None and result.lifecycle is TaskLifecycle.CANCELLED
    assert adapter.executed == (["resolve_inputs"] if ready else [])


def test_a_cancel_between_the_runner_s_read_and_its_claim_is_applied_at_the_claim(
    tmp_path: Path,
) -> None:
    """A cancel between the runner's read and its claim is applied at the claim."""

    now = datetime(2026, 8, 10, 18, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="cancel-between-read-and-claim")
    registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    adapter = _FixtureTaskAdapter()
    original_begin = registry.begin_work_item
    claims: list[str] = []

    def cancel_then_begin(**values):
        # The cancel lands after the runner read RUNNING and before the claim.
        claims.append(values["stage_id"])
        if len(claims) == 1:
            current = registry.task(values["task_id"])
            registry.request_cancel(
                task_id=values["task_id"],
                expected_task_hash=current.record_hash,
                observed_at=now + timedelta(seconds=1),
            )
        return original_begin(**values)

    registry.begin_work_item = cancel_then_begin  # type: ignore[method-assign]
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now + timedelta(seconds=2),
        heartbeat_seconds=0.01,
    )
    try:
        result = runner.run_next()
    finally:
        runner.close()
    assert result is not None and result.lifecycle is TaskLifecycle.CANCELLED
    assert result.failure_code != "TASK_EXECUTION_INTERRUPTED"
    assert claims == ["resolve_inputs"] and adapter.executed == []


def test_task_admission_order_survives_equal_clocks_and_reopen(tmp_path, monkeypatch):

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
    import duckdb

    with duckdb.connect(str(path)) as connection:
        originals = connection.execute(
            "SELECT task_id, record_json FROM workspace_task ORDER BY task_id"
        ).fetchall()
        connection.execute("DROP INDEX workspace_task_admission_order")
        connection.execute("ALTER TABLE workspace_task DROP COLUMN admission_sequence")
    assert [t.task_id for t in reopened.read_existing_tasks(path)] == sorted(expected, key=str)
    migrated = DuckDbTaskControlRegistry(path, gate=WorkspaceMutationGate())
    with duckdb.connect(str(path), read_only=True) as connection:
        assert (
            connection.execute(
                "SELECT task_id, record_json FROM workspace_task ORDER BY task_id"
            ).fetchall()
            == originals
        )
    envelope, goal, plan = task_contract(salt="after-legacy-bootstrap")
    added = migrated.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    assert migrated.tasks()[-1] == added


def test_runner_checkpoints_stages_and_recovers_from_verified_prefix(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 5, 17, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract()
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    adapter = _FixtureTaskAdapter()
    heartbeats = []
    with sqlite3.connect(tmp_path / "task-runtime-heartbeats.sqlite") as connection:
        connection.execute(
            """
            CREATE TABLE workspace_task_heartbeat (
                execution_id TEXT PRIMARY KEY,
                worker_instance_id TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL
            )
            """
        )
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now,
        heartbeat_seconds=0.01,
        heartbeat_sink=heartbeats.append,
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None and completed.lifecycle is TaskLifecycle.SUCCEEDED
    assert adapter.executed == ["resolve_inputs", "publish_result"]
    # The work board is the one record of progress: no graph checkpoint store is written (W10).
    assert not (tmp_path / "task-runtime.sqlite").exists()
    assert registry.safe_projection(task.task_id).verified_stage_count == 2
    assert heartbeats
    assert heartbeats[-1].task_id == task.task_id
    assert heartbeats[-1].execution_id == completed.latest_execution_id
    assert runner.heartbeat_signal(completed.latest_execution_id) == heartbeats[-1]

    second_envelope, second_goal, second_plan = task_contract(salt="recovery")
    recovering = registry.admit(
        input_envelope=second_envelope,
        goal=second_goal,
        plan=second_plan,
        observed_at=now + timedelta(seconds=1),
    ).record
    unstable = _FixtureTaskAdapter(fail_once=True)
    interrupted_runner = TaskControlRunner(
        registry=registry,
        adapters={unstable.task_kind: unstable},
        runtime_path=str(tmp_path / "task-runtime-2.sqlite"),
        clock=lambda: now + timedelta(seconds=2),
        heartbeat_seconds=0.01,
    )
    try:
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            interrupted_runner.run_next()
        assert registry.task(recovering.task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        recovered = interrupted_runner.recover(recovering.task_id)
    finally:
        interrupted_runner.close()
    assert recovered.lifecycle is TaskLifecycle.SUCCEEDED
    assert unstable.executed == ["resolve_inputs", "resolve_inputs", "publish_result"]


class _RaisingAdapter(_FixtureTaskAdapter):
    """A stage that raises an error its owner does not name, on every attempt."""

    def execute_stage(self, *, task, execution, work_item):
        del task, execution
        self.executed.append(work_item.stage_id)
        raise KeyError("a database error the owner does not classify")


def test_a_stage_raising_on_every_attempt_is_blocked_with_its_type_not_resumed_again(
    tmp_path: Path,
) -> None:
    """regression (LS1's acceptance): one raise is a stop the next run recovers from, and the
    same stage raising on each attempt is blocked on its third with the error's type, so a
    resume no longer loops on a failure its owner did not name."""

    from alphalattice.control.task_control.runner import REPEATED_STAGE_RAISES

    now = datetime(2026, 8, 5, 17, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="raising")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    adapter = _RaisingAdapter()
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now,
        heartbeat_seconds=0.01,
    )
    try:
        with pytest.raises(KeyError):
            runner.run_next()
        for _ in range(REPEATED_STAGE_RAISES - 2):
            assert registry.task(task.task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
            with pytest.raises(KeyError):
                runner.recover(task.task_id)
        stopped = runner.recover(task.task_id)
    finally:
        runner.close()
    assert stopped.lifecycle is TaskLifecycle.BLOCKED
    assert stopped.failure_code == "TASK_STAGE_RAISED:KeyError"
    assert adapter.executed == ["resolve_inputs"] * REPEATED_STAGE_RAISES
    # What was raised rides beside the code, as an owner's cause does (V444).
    (item,) = [
        state
        for state in registry.work_items(task.task_id)
        if state.lifecycle is WorkItemLifecycle.BLOCKED
    ]
    assert item.failure_cause is not None
    assert item.failure_cause.exception_type == "KeyError"
    assert "a database error the owner does not classify" in item.failure_cause.detail


class _DeferringThenRaisingAdapter(_FixtureTaskAdapter):
    """A stage that defers on its first two attempts, then raises until its cause is fixed."""

    def __init__(self) -> None:
        super().__init__()
        self.deferrals = 2
        self.fixed = False

    def execute_stage(self, *, task, execution, work_item):
        if work_item.stage_id == "resolve_inputs" and not self.fixed:
            self.executed.append(work_item.stage_id)
            if self.deferrals:
                self.deferrals -= 1
                return StageExecutionResult(
                    disposition=StageDisposition.DEFERRED,
                    failure_code="AGENT_TEMPORARILY_UNAVAILABLE",
                )
            raise KeyError("a database error the owner does not classify")
        return super().execute_stage(task=task, execution=execution, work_item=work_item)


def test_deferrals_spend_none_of_a_stages_raises_and_a_raised_block_reopens(
    tmp_path: Path,
) -> None:
    """Deferrals consume no stage retry allowance and a raised block reopens through its
    recovery."""

    from alphalattice.control.task_control.runner import REPEATED_STAGE_RAISES

    now = datetime(2026, 8, 5, 17, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="deferring")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    adapter = _DeferringThenRaisingAdapter()
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now,
        heartbeat_seconds=0.01,
    )

    def retry_deferred() -> None:
        registry.mark_recovery_required(
            task_id=task.task_id, failure_code="RETRY_DEFERRED_TASK", observed_at=now
        )

    try:
        assert runner.run_next().lifecycle is TaskLifecycle.DEFERRED
        retry_deferred()
        assert runner.recover(task.task_id).lifecycle is TaskLifecycle.DEFERRED
        retry_deferred()
        for _ in range(REPEATED_STAGE_RAISES - 1):
            with pytest.raises(KeyError):
                runner.recover(task.task_id)
            assert registry.task(task.task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        blocked = runner.recover(task.task_id)
        assert (blocked.lifecycle, blocked.failure_code) == (
            TaskLifecycle.BLOCKED,
            "TASK_STAGE_RAISED:KeyError",
        )
        assert adapter.executed == ["resolve_inputs"] * (2 + REPEATED_STAGE_RAISES)
        adapter.fixed = True
        reopened = registry.mark_recovery_required(
            task_id=task.task_id,
            failure_code=str(blocked.failure_code),
            observed_at=now,
            allow_blocked=True,
            expected_task_hash=blocked.record_hash,
        )
        assert reopened.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert runner.recover(task.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    finally:
        runner.close()


@pytest.mark.parametrize("accepts", [True, False])
def test_recovery_seals_stored_evidence_its_verifier_accepts_and_reruns_the_rest(
    tmp_path: Path, accepts: bool
) -> None:
    """regression (V101, EV1): a stage that stored its evidence and was interrupted before its
    receipt ran again on recovery; its verifier now decides, and only evidence it refuses runs
    again."""

    now = datetime(2026, 8, 5, 19, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="stored")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    adapter = _InterruptedBeforeReceiptAdapter(accepts_at_recovery=accepts)
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now,
        heartbeat_seconds=0.01,
    )
    try:
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            runner.run_next()
        stored = {item.stage_id: item for item in registry.work_items(task.task_id)}
        assert stored["resolve_inputs"].lifecycle is WorkItemLifecycle.READY_FOR_VERIFICATION
        recovered = runner.recover(task.task_id)
    finally:
        runner.close()
    assert recovered.lifecycle is TaskLifecycle.SUCCEEDED
    ran_again = [] if accepts else ["resolve_inputs"]
    assert adapter.executed == ["resolve_inputs", *ran_again, "publish_result"]


def test_deferred_stage_can_recover_without_leaving_a_blocked_work_item(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 5, 19, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract()
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    adapter = _FixtureTaskAdapter(defer_once=True)
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now,
        heartbeat_seconds=0.01,
    )
    try:
        deferred = runner.run_next()
        assert deferred is not None and deferred.lifecycle is TaskLifecycle.DEFERRED
        registry.mark_recovery_required(
            task_id=task.task_id,
            failure_code="RETRY_DEFERRED_TASK",
            observed_at=now + timedelta(seconds=1),
        )
        recovered = runner.recover(task.task_id)
    finally:
        runner.close()
    assert recovered.lifecycle is TaskLifecycle.SUCCEEDED
    assert all(
        item.lifecycle is WorkItemLifecycle.VERIFIED for item in registry.work_items(task.task_id)
    )


@pytest.mark.parametrize("cancel_while_returning", [False, True])
def test_dispatcher_finishes_deferred_cancel_only_after_command_returns(
    tmp_path: Path, cancel_while_returning: bool
) -> None:
    now = datetime.now(UTC)
    deferred, release = Event(), Event()
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        envelope, goal, plan = task_contract()
        adapter = _FixtureTaskAdapter(defer_once=True)

        class Command:
            command_kind = "factor_research"

            def admit(self):
                task = registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=now
                ).record
                return CommandAdmission(task.task_id, task.lifecycle)

            def execute(self, task_id):
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
            assert deferred.wait(10)
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
            release.set()
            dispatcher.close()


def test_a_held_recovery_reads_as_queued_and_a_stopped_admission_never_as_running(
    tmp_path: Path,
) -> None:
    """A held recovery reads as queued and a stopped admission never as running."""

    now = datetime.now(UTC)
    held, release = Event(), Event()
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry

        def admitted(salt: str) -> TaskRecord:
            envelope, goal, plan = task_contract(salt=salt)
            return registry.admit(
                input_envelope=envelope, goal=goal, plan=plan, observed_at=now
            ).record

        class Stops(_FixtureTaskAdapter):
            def execute_stage(self, *, task, execution, work_item):
                del task, execution, work_item
                return StageExecutionResult(
                    disposition=StageDisposition.BLOCKED, failure_code="fixture.stopped"
                )

        stopped = admitted("stopped")
        runner = TaskControlRunner(
            registry=registry,
            adapters={Stops.task_kind: Stops()},
            runtime_path=str(tmp_path / "task-runtime.sqlite"),
            clock=lambda: now,
        )
        try:
            assert runner.run_next(expected_task_id=stopped.task_id).lifecycle is (
                TaskLifecycle.BLOCKED
            )
        finally:
            runner.close()
        recovering = registry.mark_recovery_required(
            task_id=admitted("recovering").task_id,
            failure_code="fixture.interrupted",
            observed_at=now,
        )

        class Found:
            command_kind = "factor_research"

            def __init__(self, task: TaskRecord, *, hold: bool) -> None:
                self.task, self.hold = task, hold

            def admit(self) -> CommandAdmission:
                found = registry.task(self.task.task_id)
                return CommandAdmission(found.task_id, found.lifecycle.value)

            def execute(self, task_id):
                del task_id
                if self.hold:
                    held.set()
                    assert release.wait(10)

        dispatcher = LocalBackgroundDispatcher(registry)
        try:
            dispatcher.submit(Found(recovering, hold=True))
            assert held.wait(10)
            assert dispatcher.status(recovering.task_id).lifecycle is TaskLifecycle.QUEUED
            dispatcher.submit(Found(stopped, hold=False))
            assert not dispatcher.command_running(stopped.task_id)
            assert dispatcher.status(stopped.task_id).lifecycle is TaskLifecycle.BLOCKED
            release.set()
            dispatcher.drain_for_tests()
            assert dispatcher.status(recovering.task_id).lifecycle is (
                TaskLifecycle.RECOVERY_REQUIRED
            )
        finally:
            release.set()
            dispatcher.close()


@pytest.mark.parametrize("frees", ["RESUMED", "CANCELLED"])
def test_tasks_admitted_behind_a_deferral_start_once_it_frees_in_admission_order(
    tmp_path: Path, frees: str
) -> None:
    """Tasks admitted behind a deferral start once it frees in admission order."""

    now = datetime.now(UTC)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        registry = session.task_control_registry
        finished: list[str] = []

        class Command:
            command_kind = "factor_research"

            def __init__(self, salt: str, adapter: _FixtureTaskAdapter) -> None:
                self.salt, self.adapter = salt, adapter

            def admit(self) -> CommandAdmission:
                envelope, goal, plan = task_contract(salt=self.salt)
                task = registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=now
                ).record
                return CommandAdmission(task.task_id, task.lifecycle.value)

            def execute(self, task_id, *, expected_task_hash=None):
                ended = session.execute_admitted(
                    registry.task(task_id), self.adapter, lambda: now, expected_task_hash
                )
                if ended is not None and ended.lifecycle is TaskLifecycle.SUCCEEDED:
                    finished.append(self.salt)

        deferring = Command("deferred", _FixtureTaskAdapter(defer_once=True))
        dispatcher = LocalBackgroundDispatcher(registry, version_moved_error=TaskTransitionRejected)
        try:
            held = dispatcher.submit(deferring).task_id
            dispatcher.drain_for_tests()
            assert registry.task(held).lifecycle is TaskLifecycle.DEFERRED
            behind = {
                salt: dispatcher.submit(Command(salt, _FixtureTaskAdapter())).task_id
                for salt in ("second", "third")
            }
            dispatcher.drain_for_tests()
            for task_id in behind.values():
                # Its turn has not come: queued, owned, and no failure of its command.
                assert registry.task(task_id).lifecycle is TaskLifecycle.QUEUED
                assert dispatcher.command_running(task_id)
                assert dispatcher.failure(task_id) is None
            assert finished == []
            if frees == "RESUMED":
                registry.mark_recovery_required(
                    task_id=held, failure_code="RETRY_DUE", observed_at=now
                )
                assert dispatcher.resume({"factor_research": deferring}, only_task_id=held)
            else:
                assert dispatcher.request_cancel(held)
                assert registry.task(held).lifecycle is TaskLifecycle.CANCELLED
            dispatcher.drain_for_tests()
            assert finished == (
                ["deferred", "second", "third"] if frees == "RESUMED" else ["second", "third"]
            )
            for task_id in behind.values():
                assert registry.task(task_id).lifecycle is TaskLifecycle.SUCCEEDED
                assert dispatcher.failure(task_id) is None
                assert not dispatcher.command_running(task_id)
        finally:
            dispatcher.close()


class _RefusingContract(BaseModel):
    value: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def refuse(self) -> _RefusingContract:
        raise ValueError("risk_research.covariance_chunk_identity_invalid")


def test_failure_code_from_names_the_refusal_inside_a_validation_report() -> None:
    with pytest.raises(ValidationError) as caught:
        _RefusingContract(value="tampered")
    report = str(caught.value)
    assert "\n" in report and len(report) > FAILURE_CODE_MAX_LENGTH
    assert failure_code_from(caught.value) == "risk_research.covariance_chunk_identity_invalid"
    assert failure_code_from(ValueError("x" * 300)) == "x" * FAILURE_CODE_MAX_LENGTH
    assert failure_code_from(RuntimeError("plain refusal")) == "plain refusal"


class _RefusingTaskAdapter(_FixtureTaskAdapter):
    """A stage that reports a refusal longer than the work-item record accepts."""

    refusal = "risk_research.covariance_chunk_identity_invalid:" + "checkpoint " * 24

    def execute_stage(self, *, task, execution, work_item):
        del task, execution
        self.executed.append(work_item.stage_id)
        return StageExecutionResult(StageDisposition.BLOCKED, failure_code=self.refusal)


def test_runner_records_an_overlong_stage_refusal_as_a_block(tmp_path: Path) -> None:
    now = datetime(2026, 9, 16, 9, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="overlong-refusal")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    adapter = _RefusingTaskAdapter()
    assert len(adapter.refusal) > FAILURE_CODE_MAX_LENGTH
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now + timedelta(seconds=1),
        heartbeat_seconds=0.01,
    )
    try:
        result = runner.run_next()
    finally:
        runner.close()

    assert result is not None and result.lifecycle is TaskLifecycle.BLOCKED
    blocked = registry.task(task.task_id)
    assert blocked.lifecycle is TaskLifecycle.BLOCKED
    assert blocked.failure_code == adapter.refusal[:FAILURE_CODE_MAX_LENGTH]
    assert blocked.failure_code.startswith("risk_research.covariance_chunk_identity_invalid:")
    item = next(
        state for state in registry.work_items(task.task_id) if state.stage_id == "resolve_inputs"
    )
    assert item.lifecycle is WorkItemLifecycle.BLOCKED
    assert item.failure_code == blocked.failure_code
    assert adapter.executed == ["resolve_inputs"]


class _ExhaustedTaskAdapter(_FixtureTaskAdapter):
    """A stage whose owner blocks with what it saw: the machine out of memory."""

    cause = StageFailureCause.from_facts(
        {
            "exception_type": "ImportError",
            "detail": "DLL load failed while importing _multiarray_umath:\n  The paging file is "
            "too small for this operation to complete. " + "x" * 600,
            "step": "base_feature_materialization",
            "unit": "listing-a",
            "first_session": "2026-09-01",
            "last_session": "2026-09-15",
        }
    )

    def execute_stage(self, *, task, execution, work_item):
        del task, execution
        self.executed.append(work_item.stage_id)
        return StageExecutionResult(
            StageDisposition.BLOCKED,
            failure_code="feature.materialization_failed",
            failure_cause=self.cause,
        )


def test_a_blocked_stage_keeps_the_cause_its_owner_saw(tmp_path: Path) -> None:
    """A blocked stage keeps the cause its owner saw."""

    now = datetime(2026, 10, 2, 9, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="exhausted-stage")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    sealed_before = registry.work_items(task.task_id)[0]
    adapter = _ExhaustedTaskAdapter()
    assert adapter.cause is not None
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now + timedelta(seconds=1),
        heartbeat_seconds=0.01,
    )
    try:
        runner.run_next()
    finally:
        runner.close()

    item = next(
        state
        for state in registry.work_items(task.task_id)
        if state.lifecycle is WorkItemLifecycle.BLOCKED
    )
    assert item.failure_code == "feature.materialization_failed"
    assert item.failure_cause == adapter.cause
    assert item.failure_cause.detail.startswith("DLL load failed while importing _multiarray")
    assert len(item.failure_cause.detail) == 400 and "\n" not in item.failure_cause.detail
    assert item.failure_cause.first_session.isoformat() == "2026-09-01"
    details = registry._read(
        lambda connection: connection.execute(
            "SELECT details_json FROM workspace_task_event WHERE task_id = ? AND kind = ?",
            [str(task.task_id), "work_item.blocked"],
        ).fetchone()
    )
    assert json.loads(details[0])["failure_cause"] == adapter.cause.model_dump(mode="json")
    view = build_task_recovery_view(
        snapshot=registry.board(task.task_id),
        running=False,
        worker_failure=None,
        heartbeats=TaskHeartbeatReadout(signals=(), unreadable=()),
        recoverable_kinds=(),
        observed_at=now + timedelta(seconds=2),
        replans={},
    )
    assert view.stop is not None and view.stop.cause == adapter.cause

    # A state stored before causes existed has no such field, and its hash never covered one.
    stored = sealed_before.model_dump(mode="json")
    del stored["failure_cause"]
    sealed = stored.pop("state_hash")
    assert canonical_hash(stored) == sealed
    assert WorkItemState.model_validate({**stored, "state_hash": sealed}) == sealed_before
    # Facts without an exception type name no cause, and never stop the block being recorded.
    assert StageFailureCause.from_facts({"detail": "no type"}) is None
    assert StageFailureCause.from_facts({"exception_type": "E", "first_session": "x"}) is None


def test_blocked_task_recovery_requires_explicit_matching_failure(tmp_path: Path) -> None:
    now = datetime(2026, 8, 12, 5, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract()
    registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now)
    started = registry.start_next(
        compatibility=compatibility(plan),
        worker_instance_id=uuid4(),
        observed_at=now,
    )
    assert started is not None
    running, execution = started
    registry.begin_work_item(
        task_id=running.task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        observed_at=now,
    )
    blocked = registry.block_work_item(
        task_id=running.task_id,
        execution_id=execution.execution_id,
        stage_id="resolve_inputs",
        failure_code="FIXABLE_CONTRACT_WIRING",
        observed_at=now,
    )

    unchanged = registry.mark_recovery_required(
        task_id=blocked.task_id,
        failure_code="FIXABLE_CONTRACT_WIRING",
        observed_at=now + timedelta(seconds=1),
    )
    assert unchanged.lifecycle is TaskLifecycle.BLOCKED
    # A reopening confirmed against one version is written against that version
    # or not at all: the version the person saw is BLOCKED, a competing
    # confirmation moves the Task first, and the second confirmation finds the
    # Task gone from the version it confirmed and applies nothing.
    confirmed = blocked.record_hash
    with pytest.raises(TaskVersionStale):
        registry.mark_recovery_required(
            task_id=blocked.task_id,
            failure_code="FIXABLE_CONTRACT_WIRING",
            observed_at=now + timedelta(seconds=2),
            allow_blocked=True,
            expected_task_hash="0" * 64,
        )
    assert registry.task(blocked.task_id).lifecycle is TaskLifecycle.BLOCKED
    reopened = registry.mark_recovery_required(
        task_id=blocked.task_id,
        failure_code="FIXABLE_CONTRACT_WIRING",
        observed_at=now + timedelta(seconds=2),
        allow_blocked=True,
        expected_task_hash=confirmed,
    )
    assert reopened.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
    assert reopened.record_hash != confirmed
    with pytest.raises(TaskVersionStale):
        registry.mark_recovery_required(
            task_id=blocked.task_id,
            failure_code="FIXABLE_CONTRACT_WIRING",
            observed_at=now + timedelta(seconds=3),
            allow_blocked=True,
            expected_task_hash=confirmed,
        )
    assert registry.task(blocked.task_id) == reopened, "the stale loser changed nothing"


@pytest.mark.parametrize("state", (*TaskLifecycle, "RECOVERY_WITH_CANCEL"))
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
            task = _complete_stage(
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
            task = _complete_stage(
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


def test_a_versionless_cancel_aims_again_at_a_moving_task() -> None:
    """A versionless cancel aims again at a moving task."""

    class MovingPort:
        def __init__(self, moves: int, *, terminal: bool = False) -> None:
            self.version = 0
            self.moves = moves
            self.terminal = terminal
            self.requests: list[str] = []

        def safe_projection(self, task_id):
            return SimpleNamespace(
                task_id=task_id,
                task_record_hash=f"v{self.version}",
                lifecycle=TaskLifecycle.RUNNING,
            )

        def latest_safe_projections(self):
            return ()

        def tasks(self):
            return ()

        def request_cancel(self, *, task_id, expected_task_hash, observed_at):
            self.requests.append(expected_task_hash)
            if self.terminal:
                raise TaskTransitionRejected("terminal task cannot be cancelled")
            if self.moves > 0:
                self.moves -= 1
                self.version += 1
                raise TaskVersionStale("cancel command observed a stale task version")
            return object()

        def finalize_cancel_after_writer_stopped(self, **_values):
            raise AssertionError("no owned idle work here")

    def dispatcher_over(port: MovingPort) -> LocalBackgroundDispatcher:
        return LocalBackgroundDispatcher(
            status_port=port,  # type: ignore[arg-type]
            version_moved_error=TaskTransitionRejected,
            version_stale_error=TaskVersionStale,
        )

    moving = MovingPort(moves=3)
    assert dispatcher_over(moving).request_cancel(uuid4()) is True
    assert moving.requests == ["v0", "v1", "v2", "v3"], "aimed again at each version it found"
    runaway = MovingPort(moves=CANCEL_REAIM_LIMIT + 3)
    assert dispatcher_over(runaway).request_cancel(uuid4()) is False
    assert len(runaway.requests) == CANCEL_REAIM_LIMIT
    confirmed = MovingPort(moves=1)
    with pytest.raises(TaskVersionMovedError, match="cancel_version_stale"):
        dispatcher_over(confirmed).request_cancel(uuid4(), expected_task_hash="v0")
    assert confirmed.requests == ["v0"], "a confirmed version is aimed once"
    terminal = MovingPort(moves=0, terminal=True)
    assert dispatcher_over(terminal).request_cancel(uuid4()) is False
    assert len(terminal.requests) == 1, "a terminal refusal is not retried"


def _chains_contract(chains: int, *, salt: str):
    """`chains` independent two-stage chains (`c0_first` -> `c0_second`, ...)."""

    envelope = TaskInputEnvelope.create(
        task_kind="factor_research",
        input_schema_id="factor-research.confirmed-mandate",
        payload={"chains": chains, "salt": salt},
    )
    goal = ResearchGoal.create(
        goal_kind="RUN_FACTOR_RESEARCH",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchDeskFactorInput",
        summary="Independent chains.",
    )
    items = tuple(
        definition
        for index in range(chains)
        for definition in (
            work_item(f"c{index}_first"),
            work_item(f"c{index}_second", dependencies=(f"c{index}_first",)),
        )
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=digest("chains-workflow"),
        verifier_catalog_hash=digest("chains-verifiers"),
        work_items=items,
    )
    return envelope, goal, plan


class _ChainsAdapter:
    """Declares a width; records what ran at once, on which thread, after what."""

    task_kind = "factor_research"

    def __init__(self, registry, *, width: int, fail: str | None = None, block: str | None = None):
        self.registry = registry
        self.width = width
        self.fail = fail
        self.block = block
        self.lock = Lock()
        self.running = 0
        self.most = 0
        self.threads: set[int] = set()
        self.order: list[str] = []
        self.first_stages_met = Event()
        self.seen_verified: dict[str, set[str]] = {}

    def concurrent_work_items(self, task) -> int:
        del task
        return self.width

    def compatibility(self, task):
        return compatibility(task.plan)

    def execute_stage(self, *, task, execution, work_item):
        del execution
        stage = work_item.stage_id
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            self.threads.add(get_ident())
            if self.running >= min(2, self.width):
                self.first_stages_met.set()
        try:
            # Wait until two items run at once (a width of one never waits).
            self.first_stages_met.wait(timeout=5.0)
            self.seen_verified[stage] = {
                item.stage_id
                for item in self.registry.work_items(task.task_id)
                if item.lifecycle is WorkItemLifecycle.VERIFIED
            }
            if stage == self.fail:
                raise RuntimeError("simulated failure of one item")
            if stage == self.block:
                return StageExecutionResult(
                    disposition=StageDisposition.BLOCKED, failure_code="FIXTURE_BLOCKED"
                )
            return StageExecutionResult(
                disposition=StageDisposition.READY,
                evidence=(
                    TaskEvidence(
                        evidence_kind="fixture",
                        reference=f"playpen://chains/{stage}",
                        content_hash=digest(stage),
                    ),
                ),
            )
        finally:
            with self.lock:
                self.running -= 1
                self.order.append(stage)

    def verify_stage(self, *, task, execution, work_item, evidence):
        del task, execution, work_item
        return evidence


def _run_chains(tmp_path: Path, adapter_of, *, chains: int = 3, salt: str, width=None):
    now = datetime(2026, 9, 25, 12, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = _chains_contract(chains, salt=salt)
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    adapter = adapter_of(registry)
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "task-runtime.sqlite"),
        clock=lambda: now,
        heartbeat_seconds=0.01,
        width=width,
    )
    return registry, task, adapter, runner


def test_independent_work_items_run_at_once_within_the_declared_width(tmp_path: Path) -> None:
    """Independent work items run at once within the declared width."""

    registry, task, adapter, runner = _run_chains(
        tmp_path, lambda registry: _ChainsAdapter(registry, width=2), salt="width-two"
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None and completed.lifecycle is TaskLifecycle.SUCCEEDED
    assert completed.active_work_item_id is None
    assert adapter.most == 2, "two items ran at once, never three"
    assert get_ident() not in adapter.threads, "a declared width runs on worker threads"
    for index in range(3):
        assert f"c{index}_first" in adapter.seen_verified[f"c{index}_second"]
    assert all(
        item.lifecycle is WorkItemLifecycle.VERIFIED for item in registry.work_items(task.task_id)
    )
    assert len(registry.stage_receipts(task.task_id)) == 6


def test_a_tasks_width_stays_within_its_cpu_budget(tmp_path: Path) -> None:
    """A task's execution width stays within its CPU budget."""

    registry, task, adapter, runner = _run_chains(
        tmp_path, lambda registry: _ChainsAdapter(registry, width=3), salt="budget-two", width=2
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None and completed.lifecycle is TaskLifecycle.SUCCEEDED
    assert adapter.most == 2, "the budget's two cores, never the adapter's three"
    assert all(
        item.lifecycle is WorkItemLifecycle.VERIFIED for item in registry.work_items(task.task_id)
    )


def test_an_adapter_that_declares_no_width_runs_one_item_inline(tmp_path: Path) -> None:
    """requirement: every other domain is unchanged -- one item at a time, on
    the runner's own thread, in plan order."""

    _registry, _task, adapter, runner = _run_chains(
        tmp_path, lambda registry: _ChainsAdapter(registry, width=1), salt="width-one"
    )
    try:
        completed = runner.run_next()
    finally:
        runner.close()
    assert completed is not None and completed.lifecycle is TaskLifecycle.SUCCEEDED
    assert adapter.most == 1 and adapter.threads == {get_ident()}
    assert adapter.order == [
        "c0_first",
        "c0_second",
        "c1_first",
        "c1_second",
        "c2_first",
        "c2_second",
    ]


def test_a_failing_item_waits_for_the_others_and_leaves_the_task_to_recovery(
    tmp_path: Path,
) -> None:
    """requirement: an item that raises stops the run only after every item
    running beside it finished -- no worker outlives the execution -- and the
    Task is RECOVERY_REQUIRED; recovery completes it from the verified prefix."""

    registry, task, adapter, runner = _run_chains(
        tmp_path,
        lambda registry: _ChainsAdapter(registry, width=2, fail="c0_first"),
        chains=2,
        salt="width-failure",
    )
    try:
        with pytest.raises(RuntimeError, match="simulated failure of one item"):
            runner.run_next()
        assert adapter.running == 0, "the item beside the failure finished first"
        assert registry.task(task.task_id).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        adapter.fail = None
        recovered = runner.recover(task.task_id)
    finally:
        runner.close()
    assert recovered.lifecycle is TaskLifecycle.SUCCEEDED


def test_a_blocked_item_keeps_the_work_that_completed_beside_it(tmp_path: Path) -> None:
    """requirement: a block is recorded after the items running beside it
    finish; what they completed is verified (a later run carries it), and the
    Task is BLOCKED under the blocking item's code."""

    registry, task, _adapter, runner = _run_chains(
        tmp_path,
        lambda registry: _ChainsAdapter(registry, width=2, block="c0_first"),
        chains=2,
        salt="width-block",
    )
    try:
        blocked = runner.run_next()
    finally:
        runner.close()
    assert blocked is not None and blocked.lifecycle is TaskLifecycle.BLOCKED
    assert blocked.failure_code == "FIXTURE_BLOCKED"
    states = {item.stage_id: item.lifecycle for item in registry.work_items(task.task_id)}
    assert states["c0_first"] is WorkItemLifecycle.BLOCKED
    assert states["c1_first"] is WorkItemLifecycle.VERIFIED
    assert states["c0_second"] is WorkItemLifecycle.PENDING


def end_child_for_start_check(*, cancelled):
    import os

    os._exit(7)


@pytest.mark.parametrize(
    "task_kind",
    sorted(
        key.removeprefix("task_kind:")
        for key in json.loads(
            (Path(__file__).resolve().parents[2] / "config/registries/formats.json").read_text()
        )["entries"]
        if key.startswith("task_kind:")
    ),
)
def test_every_registered_task_kind_retains_a_child_start_stop_and_can_recover(
    task_kind, tmp_path, monkeypatch
):
    """V610/TE12: the shared runner holds the stop for the complete registered Task-kind class."""

    from alphalattice.control.task_control.child import ChildInterrupted, run_in_child

    with pytest.raises(ChildInterrupted):
        run_in_child(f"{__name__}:end_child_for_start_check", {})
    now = datetime(2026, 10, 3, 12, tzinfo=UTC)
    original, _, original_plan = task_contract(salt=task_kind)
    envelope = TaskInputEnvelope.create(
        task_kind=task_kind, input_schema_id=original.input_schema_id, payload=original.payload
    )
    goal = ResearchGoal.create(
        goal_kind="RUN_FACTOR_RESEARCH",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchDeskFactorInput",
        summary="Exercise the common child start seam.",
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=original_plan.workflow_definition_hash,
        verifier_catalog_hash=original_plan.verifier_catalog_hash,
        work_items=original_plan.work_items,
    )
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    admitted = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record

    class ChildStartAdapter(_FixtureTaskAdapter):
        needs_child = True

        def execute_stage(self, **arguments):
            if self.needs_child:
                run_in_child("builtins:str", {})
            return super().execute_stage(**arguments)

    adapter = ChildStartAdapter()
    adapter.task_kind = task_kind
    projections = []
    runner = TaskControlRunner(
        registry=registry,
        adapters={task_kind: adapter},
        runtime_path=str(tmp_path / "runtime.sqlite"),
        clock=lambda: now + timedelta(seconds=1),
        projection_sink=projections.append,
        heartbeat_seconds=0.01,
    )

    def cannot_start(process):
        raise OSError(1455, "fixture paging file too small")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(multiprocessing.get_context("spawn").Process, "start", cannot_start)
            stopped = runner.run_next()
        assert stopped is not None and stopped.task_id == admitted.task_id
        assert stopped.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert stopped.failure_code == "task_control.child_start_failed"
        assert registry.task(admitted.task_id) == stopped
        assert projections[-1].latest_failure_code == stopped.failure_code
        assert adapter.executed == []  # first attempt; no repeated untyped interruption
        adapter.needs_child = False
        recovered = runner.recover(admitted.task_id, expected_task_hash=stopped.record_hash)
        assert (
            recovered.task_id == admitted.task_id and recovered.lifecycle is TaskLifecycle.SUCCEEDED
        )
    finally:
        runner.close()


@pytest.mark.parametrize("phase", ["verify", "stored_recovery", "verified_prefix"])
def test_a_child_start_stop_preserves_verification_evidence(phase, tmp_path):
    """V610: unavailable verification is never evidence invalidation on either recovery path."""
    from alphalattice.control.task_control.child import ChildStartFailed

    now = datetime(2026, 10, 3, 12, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt=phase)
    admitted = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record

    class VerificationAdapter(_FixtureTaskAdapter):
        fail_verification = phase == "verify"
        interrupt_verification = phase == "stored_recovery"
        interrupt_execution = phase == "verified_prefix"

        def execute_stage(self, **arguments):
            if self.interrupt_execution and arguments["work_item"].stage_id == "publish_result":
                raise RuntimeError("fixture stop after verified prefix")
            return super().execute_stage(**arguments)

        def verify_stage(self, **arguments):
            if self.interrupt_verification:
                raise RuntimeError("fixture stop before receipt")
            if self.fail_verification:
                raise ChildStartFailed(OSError(1455, "fixture child cannot start"))
            return super().verify_stage(**arguments)

    adapter = VerificationAdapter()
    runner = TaskControlRunner(
        registry=registry,
        adapters={adapter.task_kind: adapter},
        runtime_path=str(tmp_path / "runtime.sqlite"),
        clock=lambda: now + timedelta(seconds=1),
        heartbeat_seconds=0.01,
    )
    try:
        if phase == "verify":
            stopped = runner.run_next()
        else:
            with pytest.raises(RuntimeError, match="fixture stop"):
                runner.run_next()
            before = registry.work_items(admitted.task_id)
            adapter.interrupt_verification = False
            adapter.interrupt_execution = False
            adapter.fail_verification = True
            stopped = runner.recover(admitted.task_id)
            after = registry.work_items(admitted.task_id)
            # Restart resets an interrupted, unverified item to pending before prefix checking.
            # The saved/verified item must keep its lifecycle and exact evidence through refusal.
            assert (after[0].lifecycle, after[0].evidence) == (
                before[0].lifecycle,
                before[0].evidence,
            )
            if phase == "verified_prefix":
                assert before[1].lifecycle is WorkItemLifecycle.IN_PROGRESS
                assert after[1].lifecycle is WorkItemLifecycle.PENDING
        assert stopped.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        assert stopped.failure_code == "task_control.child_start_failed"
        first = registry.work_items(admitted.task_id)[0]
        assert first.evidence
        assert first.lifecycle is (
            WorkItemLifecycle.VERIFIED
            if phase == "verified_prefix"
            else WorkItemLifecycle.READY_FOR_VERIFICATION
        )
        adapter.fail_verification = False
        finished = runner.recover(admitted.task_id)
        assert (
            finished.task_id == admitted.task_id and finished.lifecycle is TaskLifecycle.SUCCEEDED
        )
        assert adapter.executed.count("resolve_inputs") == 1
    finally:
        runner.close()


def test_recovery_links_seal_preview_and_confirmed_successor_in_the_event_journal(
    tmp_path: Path,
) -> None:
    import duckdb

    from alphalattice.control.task_control.registry import TaskControlDatabaseAuthorityError

    now = datetime(2026, 8, 21, 12, tzinfo=UTC)
    database = resolve_task_control_database(tmp_path)
    registry = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())

    source_input, source_goal, source_plan = task_contract(salt="recovery-source")
    source = registry.admit(
        input_envelope=source_input,
        goal=source_goal,
        plan=source_plan,
        observed_at=now,
    ).record
    stopped_source = registry.mark_recovery_required(
        task_id=source.task_id,
        failure_code="fixture.interrupted",
        observed_at=now + timedelta(seconds=1),
    )
    offered = {
        "operation": "EXPERIMENT_PLAN",
        "research_input_id": "research-input-1",
    }
    preview = registry.record_recovery_link(
        source_task_id=source.task_id,
        source_record_hash=stopped_source.record_hash,
        admission_request=offered,
        observed_at=now + timedelta(seconds=2),
    )
    repeated_preview = registry.record_recovery_link(
        source_task_id=source.task_id,
        source_record_hash=stopped_source.record_hash,
        admission_request=offered,
        observed_at=now + timedelta(seconds=3),
    )
    assert repeated_preview == preview
    assert preview.successor_task_id is None
    assert preview.link_hash == canonical_hash(
        preview.model_dump(mode="json", exclude={"link_hash"})
    )
    assert registry.task(source.task_id).record_hash == stopped_source.record_hash

    successor_input, successor_goal, successor_plan = task_contract(salt="recovery-successor")
    successor = registry.admit(
        input_envelope=successor_input,
        goal=successor_goal,
        plan=successor_plan,
        observed_at=now + timedelta(seconds=4),
    ).record

    with pytest.raises(
        TaskTransitionRejected, match=r"task_control\.recovery_link_preview_required"
    ):
        registry.record_recovery_link(
            source_task_id=source.task_id,
            source_record_hash=stopped_source.record_hash,
            admission_request={**offered, "research_input_id": "other-input"},
            successor_task_id=successor.task_id,
            observed_at=now + timedelta(seconds=5),
        )

    moved_source = registry.mark_recovery_required(
        task_id=source.task_id,
        failure_code="fixture.interrupted",
        observed_at=now + timedelta(seconds=6),
    )
    assert moved_source.record_hash != stopped_source.record_hash
    reopened = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    confirmed = reopened.record_recovery_link(
        source_task_id=source.task_id,
        source_record_hash=stopped_source.record_hash,
        admission_request=offered,
        successor_task_id=successor.task_id,
        observed_at=now + timedelta(seconds=7),
    )
    repeated_confirmation = reopened.record_recovery_link(
        source_task_id=source.task_id,
        source_record_hash=stopped_source.record_hash,
        admission_request=offered,
        successor_task_id=successor.task_id,
        observed_at=now + timedelta(seconds=8),
    )

    assert confirmed.successor_task_id == successor.task_id
    assert confirmed.source_record_hash == stopped_source.record_hash
    assert repeated_confirmation == confirmed
    assert reopened.task(source.task_id).record_hash == moved_source.record_hash
    assert reopened.recovery_links(source.task_id) == (preview, confirmed)
    assert reopened.recovery_links() == (preview, confirmed)
    returned = reopened.recovery_links(source.task_id)
    returned[0].admission_request["research_input_id"] = "caller-edited"
    assert reopened.recovery_links(source.task_id) == (preview, confirmed)
    with pytest.raises(TaskVersionStale, match=r"task_control\.recovery_link_source_version_stale"):
        reopened.record_recovery_link(
            source_task_id=source.task_id,
            source_record_hash=stopped_source.record_hash,
            admission_request=offered,
            observed_at=now + timedelta(seconds=8),
        )

    unrelated_input, unrelated_goal, unrelated_plan = task_contract(salt="unrelated-source")
    unrelated = registry.admit(
        input_envelope=unrelated_input,
        goal=unrelated_goal,
        plan=unrelated_plan,
        observed_at=now + timedelta(seconds=9),
    ).record
    unrelated_stopped = registry.mark_recovery_required(
        task_id=unrelated.task_id,
        failure_code="fixture.interrupted",
        observed_at=now + timedelta(seconds=10),
    )
    with pytest.raises(
        TaskTransitionRejected, match=r"task_control\.recovery_link_preview_required"
    ):
        registry.record_recovery_link(
            source_task_id=unrelated.task_id,
            source_record_hash=unrelated_stopped.record_hash,
            admission_request=offered,
            successor_task_id=successor.task_id,
            observed_at=now + timedelta(seconds=11),
        )

    # Corrupt fixture bytes through the store, then read with a fresh owner. A broken
    # link seal is an authority refusal, never a guessed successful continuation.
    damaged = preview.model_dump(mode="json")
    damaged["link_hash"] = "f" * 64
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE workspace_task_event SET details_json = ? WHERE task_id = ? AND kind = ?",
            [json.dumps(damaged), str(source.task_id), "task.recovery_link"],
        )
    fresh = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    with pytest.raises(
        TaskControlDatabaseAuthorityError, match=r"task_control\.recovery_link_invalid"
    ):
        fresh.recovery_links(source.task_id)
