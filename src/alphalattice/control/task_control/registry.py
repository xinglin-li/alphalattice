"""The durable authority of a workspace's research Tasks, on the machine's log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

import duckdb

from alphalattice.control.workspace_runtime.database import (
    WORKSPACE_MARKET_DATA_DATABASE_FILENAME,
    open_workspace_database,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.kernel.data_layer.history import held, workspace_on_log
from alphalattice.kernel.data_layer.log import uuid7
from alphalattice.kernel.execution import execution_port, preferred_task
from alphalattice.kernel.shared_kernel.environment import package_versions

from .contracts import (
    ResearchGoal,
    ResearchPlan,
    StageFailureCause,
    TaskCommand,
    TaskCommandKind,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskRecoveryLink,
    TaskSafeProjection,
    TaskStageReceipt,
    WorkItemLifecycle,
    WorkItemState,
)
from .ledger import SubmittingAgent, TaskAdmissionRequest
from .queue import active_same_work, read_queue_setting, running_places, waiting_places
from .upgrade import LEDGER_REBUILT_DETAIL as LEDGER_REBUILT_DETAIL  # a rebuilt Task's words
from .upgrade import LEDGER_REBUILT_NEXT as LEDGER_REBUILT_NEXT
from .upgrade import (
    TASK_CONTROL_DATABASE_FILENAME,
    TASK_CONTROL_DIRECTORY,
    import_ledgers,
    log_of,
    stored_records,
)

_TERMINAL_TASKS = frozenset(
    {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
)
_RECOVERY_LINK_EVENT = "task.recovery_link"
_ADMISSION_SUBJECT_EVENT = "task.admission_subject"
_RECOVERY_LINK_STOPPED = frozenset(
    {TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED, TaskLifecycle.RECOVERY_REQUIRED}
)
_NORMAL_ENDS = frozenset(
    {"work_item.ready_for_verification", "work_item.verified", "work_item.blocked"}
)
"""The events that end a stage's attempt normally: ready, verified, or blocked and deferred
alike (`block_work_item`), so a deferral ends a run of interrupted attempts."""

_WAKE_STATES = {
    "task.wake_registered": "PENDING",
    "task.wake_attempted": "ATTEMPTED",
    "task.wake_delivered": "DELIVERED",
    "task.wake_undelivered": "UNDELIVERED",
}
"""A Codex wake's journal events and the state each leaves it in (WAKE). They sit beside the
Task's record and change none of it."""
WAKE_UNCERTAIN = "CODEX_WAKE_DELIVERY_UNCERTAIN"
"""The failure of a send the Host began and did not finish: whether it was queued is unknown."""

_ACTIVE_TASKS = frozenset(
    {
        TaskLifecycle.RUNNING,
        TaskLifecycle.DEFERRED,
        TaskLifecycle.REVIEW_PENDING,
        TaskLifecycle.CANCEL_REQUESTED,
        TaskLifecycle.RECOVERY_REQUIRED,
    }
)
_RUNNING_PLACE = _ACTIVE_TASKS - {TaskLifecycle.RECOVERY_REQUIRED}
"""The Tasks that hold a running place. A Task waiting for its recovery holds none: it
waited for a person or a changed install while every Task behind it waited too. More than one
holds a place only when each may run beside others (`overlapping`)."""
_RUNNING_PLACE_SQL = ", ".join(f"'{lifecycle.value}'" for lifecycle in sorted(_RUNNING_PLACE))
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)  # the log keeps the registry clock in microseconds since

_QUEUE_HEAD = """
    SELECT task_id, record_json FROM workspace_task WHERE workspace_id = ? AND lifecycle = 'QUEUED'
    ORDER BY admission_sequence NULLS FIRST, admitted_us, task_id LIMIT 1
"""
_QUEUED_TASK = """
    SELECT task_id, record_json FROM workspace_task
    WHERE workspace_id = ? AND lifecycle = 'QUEUED' AND task_id = ?
"""


def _queued(connection: sqlite3.Connection, workspace_id: str) -> tuple[str, str] | None:
    chosen = preferred_task()
    if chosen is not None:
        row = connection.execute(_QUEUED_TASK, [workspace_id, str(chosen)]).fetchone()
    else:
        row = connection.execute(_QUEUE_HEAD, [workspace_id]).fetchone()
    return cast("tuple[str, str] | None", row)


def task_is_unrecoverable(task: TaskRecord) -> bool:
    """Whether a blocked Task carries the ledger rebuild's unrecoverable reason."""
    return (
        task.lifecycle is TaskLifecycle.BLOCKED
        and task.failure_code == "task_control.ledger_rebuilt"
    )


def _cancel_available(task: TaskRecord) -> bool:
    return task.lifecycle not in _TERMINAL_TASKS or task_is_unrecoverable(task)


class TaskQueueFull(RuntimeError):
    """Every waiting place is taken."""


def _queue_full(queued: int, places: int, reason: str) -> TaskQueueFull:
    return TaskQueueFull(
        f"task_control.queue_full: {queued} Tasks wait in the queue's {places} places ({reason}); "
        "tasks_waiting setting sets them; wait for a Task to finish before retrying"
    )


class TaskNotFoundError(KeyError):
    """Only the requested Task row is absent, not a dependent execution or stage."""

    def __init__(self, task_id: UUID):
        super().__init__(task_id)
        self.task_id = task_id


class TaskTransitionRejected(RuntimeError):
    """Refuse a task or work-item transition whose deterministic preconditions do not hold."""

    pass


class TaskVersionStale(TaskTransitionRejected):
    """A transition confirmed against one Task version found another.

    Raised inside the write transaction, so a cancel or a recovery confirmed
    against version A is refused atomically once the Task has moved to B;
    nothing is applied to whatever the Task became. Callers that carry no
    confirmed version never see it.
    """


@dataclass(frozen=True)
class TaskAdmission:
    """Return admitted durable state and whether a matching nonterminal request already existed.

    Attributes:
        record: Existing or newly admitted task record.
        duplicate_active: Whether matching nonterminal input/plan scope was reused.
    """

    record: TaskRecord
    duplicate_active: bool


@dataclass(frozen=True)
class TaskBoardSnapshot:
    """Retain one task board whose components share a single record version.

    One Task's record, work board, latest execution and safe projection, read
    under one connection lock so every part describes the same record version
    (`projection.task_record_hash == task.record_hash`).
    """

    task: TaskRecord
    work_items: tuple[WorkItemState, ...]
    execution: TaskExecution | None
    projection: TaskSafeProjection


@dataclass(frozen=True)
class TaskProjectionBatch:
    """Readable projections and exact Task ids whose damaged projection could not be rebuilt."""

    projections: tuple[TaskSafeProjection, ...]
    refused_task_ids: tuple[UUID, ...]


class TaskControlDatabaseAuthorityError(ValueError):
    """Stable refusal for a Task Control store that is not the workspace's own."""


@dataclass(frozen=True)
class TaskRecordBatch:
    """Canonical records and stored identities whose record could not be verified."""

    records: tuple[TaskRecord, ...]
    refused_task_ids: tuple[str, ...]


class TaskRecordAuthorityError(TaskControlDatabaseAuthorityError):
    """An incomplete canonical scan cannot authorize work or establish absence."""

    def __init__(self, refused_task_ids: tuple[str, ...]) -> None:
        super().__init__("task_control.database_authority_unreadable")
        self.refused_task_ids = refused_task_ids


class TaskQueueHeadAuthorityError(TaskControlDatabaseAuthorityError):
    """The canonical QUEUED head cannot be read to answer another Task's projection."""

    def __init__(self, task_id: str | None = None) -> None:
        """Name the unreadable queue head without attributing its failure to another Task.

        Args:
            task_id: The stored identity of the queued head, when it is readable.
        """
        super().__init__("task_control.database_authority_unreadable")
        self.task_id = task_id


def resolve_task_control_database(workspace: Path) -> Path:
    """The one Task Control store of a workspace, named from its root: `runtime/`.

    Sixteen production sites spelled this filename as a literal and one --
    ``WorkspaceRuntime`` -- passed the market-data database instead, which is
    how the same workspace ended up with two task schemas. Resolving through one
    function is what makes that a single fact rather than seventeen agreeing
    strings. The place is this function's too: its callers passed either
    the root or `runtime/`, and the workspaces grew a second, empty store at the
    root, so a caller passes the root and a `runtime/` directory is refused.
    """
    root = Path(workspace)
    if root.name == TASK_CONTROL_DIRECTORY:
        raise TaskControlDatabaseAuthorityError("task_control.workspace_root_expected")
    canonical = root / TASK_CONTROL_DIRECTORY / TASK_CONTROL_DATABASE_FILENAME
    for predecessor in (
        root / TASK_CONTROL_DATABASE_FILENAME,
        root / WORKSPACE_MARKET_DATA_DATABASE_FILENAME,
    ):
        if not predecessor.is_file() or _task_state_row_count(predecessor) == 0:
            continue
        canonical_rows = _task_state_row_count(canonical) if canonical.is_file() else 0
        if canonical_rows:
            raise TaskControlDatabaseAuthorityError("task_control.database_authorities_conflict")
        # Selecting the empty/new canonical store here would abandon durable state
        # in the predecessor. No migration or cross-store readback owner exists, so
        # even terminal rows are authority until one is installed deliberately.
        raise TaskControlDatabaseAuthorityError("task_control.legacy_database_state_present")
    return canonical


def _task_state_row_count(database_path: Path) -> int:
    """Count Task Control state without opening either store for writes."""

    try:
        connection = open_workspace_database(database_path, read_only=True)
        try:
            tables = tuple(
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'main'
                      AND table_name LIKE 'workspace_task%'
                    ORDER BY table_name
                    """
                ).fetchall()
            )
            return sum(
                int(
                    connection.execute(
                        "SELECT count(*) FROM query_table(?)",
                        [table],
                    ).fetchone()[0]
                )
                for table in tables
            )
        finally:
            connection.close()
    except duckdb.Error as error:
        raise TaskControlDatabaseAuthorityError(
            "task_control.database_authority_unreadable"
        ) from error


class DuckDbTaskControlRegistry:
    """One workspace's Task authority on the machine's log (LAWS DA2).

    Each write is an event carrying the sealed record it makes (`task.requested`, `task.saved`,
    `work_item.saved`, `task.noted`, `execution.started`, `stage.verified`, `command.accepted`,
    `command.resolved`); every read takes their log.sqlite projections, scoped to this workspace.
    An operation reads, decides and appends in one unit of the log's one writer, so no write
    lands between a read and the event it decided, and a failed one leaves no event. An
    execution's latest heartbeat and each Task's safe projection are working state beside them.
    """

    def __init__(
        self,
        database_path: Path,
        *,
        gate: WorkspaceMutationGate,
        submitted_by: Callable[[], SubmittingAgent | None] | None = None,
        workspace_id: str | None = None,
    ) -> None:
        """Bind the workspace's Task Control to the log its process holds.

        Args:
            database_path: The workspace's canonical Task Control place: its 0.2.7 store, if one
                is still there, is imported onto the log here, and the queue setting is read
                beside it.
            gate: Workspace mutation owner governing durable writes.
            submitted_by: Optional Host provenance callback, separate from admission authority.
            workspace_id: The id its manifest names the workspace by; else its folder's.

        Raises:
            TaskControlDatabaseAuthorityError: The path is not the canonical Task Control place.
            DataError: Another process holds the data root (`data_layer.host_running`).
        """
        self.database_path = database_path.resolve()
        # Who submits a Task, as the Host's boundary read its request: provenance only.
        self._submitted_by = submitted_by or (lambda: None)
        # Which Tasks may hold a running place beside others, as the Host composes it: none by
        # default, so one Task runs at a time; `started` is told each start, so a waiting Task
        # that may run beside it is driven at once.
        self.overlapping: Callable[[TaskRecord], bool] = lambda _task: False
        self.started: Callable[[], None] = lambda: None
        if self.database_path.name != TASK_CONTROL_DATABASE_FILENAME:
            # A Task authority beside the real one is refused by name: none is read or written.
            raise TaskControlDatabaseAuthorityError("task_control.registry_database_not_canonical")
        self.gate = gate
        self._answers: dict[object, tuple[tuple[object, ...], object]] = {}
        root = self.database_path.parent.parent
        self.workspace_id = workspace_on_log(root, root.name, workspace_id)
        unequal = import_ledgers(self)
        if unequal:  # the 0.2.7 files stay, for the next open to retry
            raise TaskControlDatabaseAuthorityError(
                "task_control.import_unequal:" + ",".join(unequal[:5])
            )

    @staticmethod
    def _json(value: object) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _model(model_type, value: str):
        return model_type.model_validate_json(value)

    @staticmethod
    def _us(value: datetime) -> int:
        """A registry clock as the log's rows keep it: UTC integer microseconds."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task-control clock must be timezone-aware")
        return (value - _EPOCH) // timedelta(microseconds=1)

    @staticmethod
    def _db_time(value: datetime) -> str:
        """A registry clock as its rows keep it: UTC, ISO text to the microsecond, so it sorts."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task-control clock must be timezone-aware")
        naive = value.astimezone(UTC).replace(tzinfo=None)
        return naive.isoformat(sep=" ", timespec="microseconds")

    def _record(self, type_: str, task_id: UUID, attrs: dict[str, Any], **more: Any) -> None:
        """Append one Task Control event about `task_id`, in this workspace; its text, when it
        has one, is the sealed record it carries."""
        held().record(
            type_,
            ("task", str(task_id)),
            attrs,
            workspace=self.workspace_id,
            node="task_control",
            **more,
        )

    def _write(self, operation, *, control_only: bool = False):
        def unit():
            with held().unit() as connection:
                return operation(connection)

        # Admission, execution and artifact-reference mutations retain the
        # workspace fence. Cancellation requests and derived projections add
        # no Data reference/work and may proceed while the Data writer is busy.
        if control_only:
            return unit()
        return self.gate.run(unit)

    def _stamp(self) -> tuple[object, ...]:
        """The log connection's count of changes: any write since a read changes it."""
        with held().reading() as connection:
            return (connection.total_changes,)

    def _answered(self, key: object, read):  # type: ignore[no-untyped-def]
        """A read answered again while the log shows no write since it was taken.

        Every write, this registry's or another's, changes the held log's count of changes, so
        the count taken before a read tells whether its answer still holds (W10, measured: STATUS
        p50 1.8 s at twenty requests a second on a nine-Task workspace when each read was taken
        again).
        """
        stamp = self._stamp()
        held_answer = self._answers.get(key)
        if held_answer is not None and held_answer[0] == stamp:
            return held_answer[1]
        value = read()
        self._answers[key] = (stamp, value)
        return value

    def _read(self, operation):
        with held().reading() as connection:
            return operation(connection)

    @contextmanager
    def retain(self, *, read_only: bool = False):
        """One bounded unit of work on the calling thread.

        The log stays open for the process's life, so a unit reopens nothing; a read-only unit is
        the read scope, and a writing one (a Task's whole execution) holds nothing, so the log's
        other writers never wait behind it.
        """
        if not read_only:
            yield
            return
        with self.read_scope():
            yield

    @contextmanager
    def read_scope(self):
        """Every read nested inside sees one state of the log: no write lands between them.

        The board and the status projection hold it; so does the application session's read
        boundary (``WorkspaceApplicationSession.reads``) for a projection that joins readers.
        """
        with held().reading():
            yield

    @staticmethod
    def _verified_record(stored_id: str, document: str) -> TaskRecord:
        try:
            record = TaskRecord.model_validate_json(document)
            if str(record.task_id) != stored_id:
                raise ValueError("canonical Task identity differs from its stored row")
        except (TypeError, ValueError) as error:
            raise TaskRecordAuthorityError((stored_id,)) from error
        return record

    @staticmethod
    def _task_row(connection, task_id: UUID) -> TaskRecord:
        row = connection.execute(
            "SELECT record_json FROM workspace_task WHERE task_id = ?", [str(task_id)]
        ).fetchone()
        if row is None:
            raise TaskNotFoundError(task_id)
        return DuckDbTaskControlRegistry._verified_record(str(task_id), row[0])

    @staticmethod
    def _work_item_row(connection, task_id: UUID, stage_id: str) -> WorkItemState:
        row = connection.execute(
            """
            SELECT state_json FROM workspace_task_work_item
            WHERE task_id = ? AND stage_id = ?
            """,
            [str(task_id), stage_id],
        ).fetchone()
        if row is None:
            raise KeyError((task_id, stage_id))
        return WorkItemState.model_validate_json(row[0])

    @staticmethod
    def _replace_task(
        current: TaskRecord,
        *,
        observed_at: datetime,
        lifecycle: TaskLifecycle | None = None,
        active_work_item_id: str | object | None = ...,
        latest_execution_id: UUID | object | None = ...,
        started_at: datetime | object | None = ...,
        failure_code: str | object | None = ...,
    ) -> TaskRecord:
        identity = current.model_dump(mode="python", exclude={"record_hash"})
        identity["lifecycle"] = lifecycle or current.lifecycle
        if active_work_item_id is not ...:
            identity["active_work_item_id"] = active_work_item_id
        if latest_execution_id is not ...:
            identity["latest_execution_id"] = latest_execution_id
        if started_at is not ...:
            identity["started_at"] = started_at
        if failure_code is not ...:
            identity["failure_code"] = failure_code
        identity["updated_at"] = observed_at
        identity["version"] = current.version + 1
        return TaskRecord.from_identity(**identity)

    @staticmethod
    def _replace_work_item(
        current: WorkItemState,
        *,
        observed_at: datetime,
        lifecycle: WorkItemLifecycle,
        attempt_count: int | None = None,
        evidence: tuple[TaskEvidence, ...] | None = None,
        failure_code: str | object | None = ...,
        failure_cause: StageFailureCause | None = None,
        started_at: datetime | object | None = ...,
    ) -> WorkItemState:
        identity = current.model_dump(mode="python", exclude={"state_hash"})
        identity["lifecycle"] = lifecycle
        identity["attempt_count"] = (
            current.attempt_count if attempt_count is None else attempt_count
        )
        if evidence is not None:
            identity["evidence"] = evidence
        if failure_code is not ...:
            # A cause is kept only with the code it was recorded beside.
            identity["failure_code"] = failure_code
            identity["failure_cause"] = failure_cause
        if started_at is not ...:
            identity["started_at"] = started_at
        identity["updated_at"] = observed_at
        identity["version"] = current.version + 1
        return WorkItemState.from_identity(**identity)

    def _save_task(self, connection, task: TaskRecord) -> None:
        attrs = {"lifecycle": task.lifecycle.value, "updated_at": self._db_time(task.updated_at)}
        self._record("task.saved", task.task_id, attrs, text=self._json(task))

    def _save_work_item(self, connection, item: WorkItemState, ordinal: int | None = None) -> None:
        attrs = {
            "stage_id": item.stage_id,
            "lifecycle": item.lifecycle.value,
            "updated_at": self._db_time(item.updated_at),
        }
        if ordinal is not None:
            attrs["ordinal"] = ordinal
        self._record("work_item.saved", item.task_id, attrs, text=self._json(item))

    def _event(
        self,
        connection,
        *,
        task_id: UUID,
        execution_id: UUID | None,
        kind: str,
        details: dict[str, object],
        observed_at: datetime,
    ) -> None:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM workspace_task_event WHERE task_id = ?",
            [str(task_id)],
        ).fetchone()[0]
        attrs = {
            "event_id": uuid7(),
            "execution_id": str(execution_id) if execution_id else None,
            "sequence": sequence,
            "kind": kind,
            "recorded_at": self._db_time(observed_at),
        }
        self._record("task.noted", task_id, attrs, text=self._json(details))

    def daily_health(self, task_id: UUID | None = None) -> dict[str, object] | None:
        """Read an exact Daily observation or the workspace's latest one."""
        subject, args = ("", ()) if task_id is None else (" AND e.subject_id = ?", (str(task_id),))
        rows = held().read(
            "SELECT x.text FROM event e JOIN event_text x ON x.seq = e.seq"
            " WHERE e.workspace_id = ? AND e.type = 'task.noted'"
            " AND json_extract(e.attrs, '$.kind') = 'daily_health'"
            + subject
            + " ORDER BY e.seq DESC LIMIT 1",
            (self.workspace_id, *args),
        )
        return cast(dict[str, object], json.loads(rows[0][0])) if rows else None

    def record_daily_health(
        self, task: TaskRecord, result: dict[str, object], *, observed_at: datetime
    ) -> None:
        """Keep the first observation with the Daily, including unavailable checks."""

        def operation(connection) -> None:
            if self.daily_health(task.task_id) is None:
                self._event(
                    connection,
                    task_id=task.task_id,
                    execution_id=task.latest_execution_id,
                    kind="daily_health",
                    details=result,
                    observed_at=observed_at,
                )

        self._write(operation, control_only=True)

    def _admitted(
        self,
        record: TaskRecord,
        items: Iterable[WorkItemState],
        *,
        admission_sequence: int | None,
        submitted_by: SubmittingAgent | None,
        frozen: str | None,
    ) -> None:
        """Record a Task's admission: its sealed record and work items, and as its text the
        frozen request it was admitted to do. Its parent is the goal its submitting agent
        declared when the log holds that goal in this workspace, else this workspace."""
        ((_, version),) = package_versions(("alphalattice",))
        goal = None if submitted_by is None else submitted_by.goal_id
        held_goal = "SELECT 1 FROM goal WHERE goal_id = ? AND workspace_id = ?"
        if goal is not None and not held().read(held_goal, (goal, self.workspace_id)):
            goal = None
        attrs = {
            "goal_id": goal,
            "type": record.task_kind,
            "product_version": str(version),
            "input_hash": record.input.input_hash,
            "plan_hash": record.plan.plan_hash,
            "workflow_definition_hash": record.plan.workflow_definition_hash,
            "admitted_at": self._db_time(record.admitted_at),
            "admission_sequence": admission_sequence,
            "submitted_by": None if submitted_by is None else submitted_by.model_dump(mode="json"),
        }
        if (
            record.lifecycle is not TaskLifecycle.QUEUED
        ):  # a 0.2.7 Task enters at its stored lifecycle
            attrs["lifecycle"] = record.lifecycle.value
        self._record("task.requested", record.task_id, attrs, text=frozen, version=2)
        self._save_task(None, record)
        for ordinal, item in enumerate(items):
            self._save_work_item(None, item, ordinal)

    def _started(self, execution: TaskExecution) -> None:
        attrs = {
            "execution_id": str(execution.execution_id),
            "graph_thread_id": execution.graph_thread_id,
            "worker_instance_id": str(execution.worker_instance_id),
            "compatibility_hash": execution.compatibility.compatibility_hash,
            "started_at": self._db_time(execution.started_at),
        }
        self._record("execution.started", execution.task_id, attrs, text=self._json(execution))

    def _resolved(self, task_id: UUID, command_id: object, at: str | None, resolution: str) -> None:
        attrs = {"command_id": str(command_id), "resolved_at": at, "resolution": resolution}
        self._record("command.resolved", task_id, attrs)

    @staticmethod
    def _beat(execution: TaskExecution, at: datetime) -> TaskExecution:
        identity = execution.model_dump(mode="python", exclude={"execution_hash"})
        identity["last_heartbeat_at"] = at
        return TaskExecution.from_identity(**identity)

    @classmethod
    def _execution_row(
        cls, connection: sqlite3.Connection, execution_id: object
    ) -> TaskExecution | None:
        """An execution as sealed at its start, resealed with its latest heartbeat if it beat."""
        row = connection.execute(
            "SELECT e.execution_json, h.last_heartbeat_us FROM workspace_task_execution e"
            " LEFT JOIN workspace_task_heartbeat_at h ON h.execution_id = e.execution_id"
            " WHERE e.execution_id = ?",
            [str(execution_id)],
        ).fetchone()
        if row is None:
            return None
        execution = TaskExecution.model_validate_json(row[0])
        if row[1] is None:
            return execution
        return cls._beat(execution, _EPOCH + timedelta(microseconds=row[1]))

    def admit(
        self,
        *,
        input_envelope: TaskInputEnvelope,
        goal: ResearchGoal,
        plan: ResearchPlan,
        observed_at: datetime,
        same_work: Callable[[TaskRecord], bool] | None = None,
    ) -> TaskAdmission:
        """Admit sealed work or reuse a matching nonterminal task before consuming a place.

        Args:
            input_envelope: Sealed input scope to admit.
            goal: Sealed goal bound to the input.
            plan: Sealed workflow/verifier plan bound to the goal.
            observed_at: Aware operational clock supplied by the caller.
            same_work: The owner's rule for other input that is the same work, asked of each
                unfinished Task of this kind inside this transaction, so two admissions in two
                processes can't both pass it.

        Returns:
            The task record and duplicate-admission disposition.

        Raises:
            TaskQueueFull: Every configured waiting place is taken.
            ValueError: Clock or request-derived record validation fails.
        """
        self._db_time(observed_at)

        def operation(connection):
            active = active_same_work(
                connection,
                self.workspace_id,
                self._verified_record,
                input_envelope,
                plan,
                same_work,
            )
            if active is not None:
                return TaskAdmission(record=active, duplicate_active=True)
            queued_count = connection.execute(
                "SELECT COUNT(*) FROM workspace_task"
                " WHERE workspace_id = ? AND lifecycle = 'QUEUED'",
                [self.workspace_id],
            ).fetchone()[0]
            places, reason = waiting_places(read_queue_setting(self.database_path.parent))
            if queued_count >= places:
                raise _queue_full(queued_count, places, reason)
            task_id = UUID(uuid7())  # Host-minted, time-ordered
            identity = {
                "task_id": task_id,
                "task_kind": input_envelope.task_kind,
                "input": input_envelope,
                "goal": goal,
                "plan": plan,
                "lifecycle": TaskLifecycle.QUEUED,
                "active_work_item_id": None,
                "latest_execution_id": None,
                "admitted_at": observed_at,
                "started_at": None,
                "updated_at": observed_at,
                "failure_code": None,
                "version": 1,
            }
            record = TaskRecord.from_identity(**identity)
            admission_sequence = connection.execute(
                "SELECT COALESCE(MAX(admission_sequence), 0) + 1 FROM workspace_task"
                " WHERE workspace_id = ?",
                [self.workspace_id],
            ).fetchone()[0]
            submitted_by = self._submitted_by()
            items = [
                WorkItemState.from_identity(
                    task_id=task_id,
                    stage_id=definition.stage_id,
                    definition_hash=definition.definition_hash,
                    lifecycle=WorkItemLifecycle.PENDING,
                    attempt_count=0,
                    evidence=(),
                    failure_code=None,
                    started_at=None,
                    updated_at=observed_at,
                    version=1,
                )
                for definition in plan.work_items
            ]
            # What the Task was admitted to do is its admission's text, frozen (DA2).
            frozen = TaskAdmissionRequest(
                task_id=task_id,
                task_kind=input_envelope.task_kind,
                input=input_envelope,
                goal=goal,
                plan=plan,
                admitted_at=observed_at,
                admission_sequence=admission_sequence,
                submitted_by=submitted_by,
            ).model_dump_json()
            self._admitted(
                record,
                items,
                admission_sequence=admission_sequence,
                submitted_by=submitted_by,
                frozen=frozen,
            )
            self._event(
                connection,
                task_id=task_id,
                execution_id=None,
                kind="task.admitted",
                details={
                    "input_hash": input_envelope.input_hash,
                    "goal_hash": goal.goal_hash,
                    "plan_hash": plan.plan_hash,
                },
                observed_at=observed_at,
            )
            return TaskAdmission(record=record, duplicate_active=False)

        return self._write(operation)

    def check_capacity(self) -> None:
        """Refuse as admission would when every waiting place is taken.

        Asked before a request does its planning work: Portfolio's RUN resolved its
        sources and caches first.
        """
        queued = self._read(
            lambda connection: connection.execute(
                "SELECT COUNT(*) FROM workspace_task"
                " WHERE workspace_id = ? AND lifecycle = 'QUEUED'",
                [self.workspace_id],
            ).fetchone()[0]
        )
        places, reason = waiting_places(read_queue_setting(self.database_path.parent))
        if queued >= places:
            raise _queue_full(queued, places, reason)

    def start_next(
        self,
        *,
        compatibility: TaskExecutionCompatibility,
        worker_instance_id: UUID,
        observed_at: datetime,
        expected_task_id: UUID | None = None,
        expected_task_hash: str | None = None,
    ) -> tuple[TaskRecord, TaskExecution] | None:
        """Claim the admission-ordered queue head when the workspace running place is free.

        Args:
            compatibility: Installed execution compatibility bindings.
            worker_instance_id: Worker UUID bound to the execution.
            observed_at: Aware operational clock supplied by the caller.
            expected_task_id: Optional queue-head identity the caller confirmed.
            expected_task_hash: Optional exact task version the caller confirmed.

        Returns:
            The started task/execution pair, or None when no queued work or free running place
            exists.

        Raises:
            TaskVersionStale: The confirmed queue-head version changed.
            TaskTransitionRejected: The confirmed task or installed workflow binding differs.
        """
        self._db_time(observed_at)

        def operation(connection):
            # The queue's head in admission order, the order the dispatcher runs them in.
            row = _queued(connection, self.workspace_id)
            if row is None:
                return None
            current = self._verified_record(str(row[0]), row[1])
            if not self._beside(connection, current):
                return None
            if expected_task_id is not None and current.task_id != expected_task_id:
                raise TaskTransitionRejected("task_control.queued_task_identity_mismatch")
            if expected_task_hash is not None and current.record_hash != expected_task_hash:
                raise TaskVersionStale("task_control.start_version_stale")
            if current.plan.workflow_definition_hash != compatibility.workflow_definition_hash:
                raise TaskTransitionRejected("task workflow compatibility changed before start")
            execution_id = UUID(uuid7())
            execution = TaskExecution.from_identity(
                execution_id=execution_id,
                task_id=current.task_id,
                graph_thread_id=f"workspace-task:{current.task_id}:{execution_id}",
                worker_instance_id=worker_instance_id,
                compatibility=compatibility,
                started_at=observed_at,
                last_heartbeat_at=observed_at,
                checkpoint_disposition="active",
            )
            task = self._replace_task(
                current,
                observed_at=observed_at,
                lifecycle=TaskLifecycle.RUNNING,
                latest_execution_id=execution_id,
                started_at=observed_at,
                failure_code=None,
            )
            self._save_task(connection, task)
            self._started(execution)
            self._event(
                connection,
                task_id=task.task_id,
                execution_id=execution_id,
                kind="task.started",
                details={"compatibility_hash": compatibility.compatibility_hash},
                observed_at=observed_at,
            )
            return task, execution

        started = self._write(operation)
        if started is not None:
            self.started()
        return started

    def may_start(self, task_id: UUID) -> bool:
        """Whether the Task would take a running place now (`start_next`, `restart_recovery`).

        It is owed its recovery or stands at the queue's head, and the places' holders let it
        run beside them.
        """

        def operation(connection) -> bool:  # type: ignore[no-untyped-def]
            current = self._task_row(connection, task_id)
            if current.lifecycle is TaskLifecycle.QUEUED:
                head = _queued(connection, self.workspace_id)
                if head is None or str(head[0]) != str(task_id):
                    return False
            elif current.lifecycle is not TaskLifecycle.RECOVERY_REQUIRED:
                return False
            return self._beside(connection, current)

        return bool(self._read(operation))

    def turn_stamp(self) -> tuple[str, ...]:
        """Who holds the running places and heads the queue, read together.

        A returned command's turn moved when this did while it ran.
        """

        def operation(connection) -> tuple[str, ...]:  # type: ignore[no-untyped-def]
            holders = connection.execute(
                "SELECT task_id FROM workspace_task WHERE workspace_id = ? AND lifecycle IN ("
                + _RUNNING_PLACE_SQL
                + ") ORDER BY task_id",
                [self.workspace_id],
            ).fetchall()
            head = connection.execute(_QUEUE_HEAD, [self.workspace_id]).fetchone()
            return (*(str(row[0]) for row in holders), "|", "" if head is None else str(head[0]))

        return tuple(self._read(operation))

    def _beside(self, connection, task: TaskRecord) -> bool:  # type: ignore[no-untyped-def]
        """Whether no other Task holds a running place, or a place is free and every holder and
        this Task may run beside others (`tasks_running`)."""
        rows = connection.execute(
            "SELECT task_id, record_json FROM workspace_task WHERE workspace_id = ?"
            " AND task_id <> ? AND lifecycle IN (" + _RUNNING_PLACE_SQL + ")",
            [self.workspace_id, str(task.task_id)],
        ).fetchall()
        if not rows:
            return True
        places, _reason = running_places(read_queue_setting(self.database_path.parent))
        holders = [self._verified_record(str(row[0]), row[1]) for row in rows]
        return len(holders) < places and all(map(self.overlapping, [task, *holders]))

    def restart_recovery(
        self,
        *,
        task_id: UUID,
        compatibility: TaskExecutionCompatibility,
        worker_instance_id: UUID,
        observed_at: datetime,
        expected_task_hash: str | None = None,
    ) -> tuple[TaskRecord, TaskExecution]:
        """Start a fresh compatible execution from the verified work-board prefix.

        A stage that stored its evidence before its receipt keeps it: the runner verifies it
        first and runs again only a stage whose evidence fails (EV1); a stage caught in
        progress or blocked starts over.

        `expected_task_hash` is the record version a person or Agent confirmed
        the recovery against; when it is given and the Task has moved, the
        recovery is refused here, inside the transaction, rather than started
        from whatever the Task became. Versionless callers are unchanged.
        """
        self._db_time(observed_at)

        def operation(connection):
            current = self._task_row(connection, task_id)
            if expected_task_hash is not None and current.record_hash != expected_task_hash:
                raise TaskVersionStale("task_control.recovery_version_stale")
            if current.lifecycle is not TaskLifecycle.RECOVERY_REQUIRED:
                raise TaskTransitionRejected("task does not require recovery")
            # It held no running place while it waited, so another Task may hold it now.
            if not self._beside(connection, current):
                raise TaskTransitionRejected("task_control.recovery_waits_for_the_running_task")
            if current.plan.workflow_definition_hash != compatibility.workflow_definition_hash:
                raise TaskTransitionRejected("task workflow compatibility changed before recovery")
            if current.latest_execution_id is None:
                raise TaskTransitionRejected("recovery task has no prior execution")
            prior = self._execution_row(connection, current.latest_execution_id)
            if prior is None:
                raise KeyError(current.latest_execution_id)
            if prior.compatibility != compatibility:
                raise TaskTransitionRejected("task execution compatibility changed before recovery")
            rows = connection.execute(
                """
                SELECT state_json FROM workspace_task_work_item
                WHERE task_id = ?
                  AND lifecycle IN ('IN_PROGRESS', 'BLOCKED')
                """,
                [str(task_id)],
            ).fetchall()
            for row in rows:
                item = WorkItemState.model_validate_json(row[0])
                reset = self._replace_work_item(
                    item,
                    observed_at=observed_at,
                    lifecycle=WorkItemLifecycle.PENDING,
                    evidence=(),
                    failure_code=None,
                    started_at=None,
                )
                self._save_work_item(connection, reset)
            execution_id = UUID(uuid7())
            execution = TaskExecution.from_identity(
                execution_id=execution_id,
                task_id=current.task_id,
                graph_thread_id=f"workspace-task:{current.task_id}:{execution_id}",
                worker_instance_id=worker_instance_id,
                compatibility=compatibility,
                started_at=observed_at,
                last_heartbeat_at=observed_at,
                checkpoint_disposition="active",
            )
            task = self._replace_task(
                current,
                observed_at=observed_at,
                lifecycle=TaskLifecycle.RUNNING,
                active_work_item_id=None,
                latest_execution_id=execution_id,
                failure_code=None,
            )
            self._save_task(connection, task)
            self._started(execution)
            self._event(
                connection,
                task_id=task_id,
                execution_id=execution_id,
                kind="task.recovery_started",
                details={"prior_execution_id": str(prior.execution_id)},
                observed_at=observed_at,
            )
            return task, execution

        return self._write(operation)

    def begin_work_item(
        self,
        *,
        task_id: UUID,
        execution_id: UUID,
        stage_id: str,
        observed_at: datetime,
    ) -> WorkItemState:
        """Begin a pending stage under the current running execution after all dependencies verify.

        Args:
            task_id: Task UUID whose durable record is addressed.
            execution_id: Execution UUID whose record or event is addressed.
            stage_id: Exact work-stage identifier in the sealed plan.
            observed_at: Aware operational clock supplied by the caller.

        Returns:
            The in-progress stage state with incremented attempt count.

        Raises:
            TaskTransitionRejected: Task/execution/stage state or dependency verification forbids
                beginning.
            KeyError: The requested stage is outside the plan.
        """

        def operation(connection):
            task = self._task_row(connection, task_id)
            if task.lifecycle is not TaskLifecycle.RUNNING:
                raise TaskTransitionRejected("only a running task can begin work")
            if task.latest_execution_id != execution_id:
                raise TaskTransitionRejected("work item execution is stale")
            definition = next(
                (item for item in task.plan.work_items if item.stage_id == stage_id), None
            )
            if definition is None:
                raise KeyError(stage_id)
            for dependency in definition.dependency_ids:
                dependency_state = self._work_item_row(connection, task_id, dependency)
                if dependency_state.lifecycle is not WorkItemLifecycle.VERIFIED:
                    raise TaskTransitionRejected("work item dependency is not verified")
            current = self._work_item_row(connection, task_id, stage_id)
            if current.lifecycle is not WorkItemLifecycle.PENDING:
                raise TaskTransitionRejected("work item is not pending")
            item = self._replace_work_item(
                current,
                observed_at=observed_at,
                lifecycle=WorkItemLifecycle.IN_PROGRESS,
                attempt_count=current.attempt_count + 1,
                started_at=observed_at,
                failure_code=None,
            )
            task = self._replace_task(
                task,
                observed_at=observed_at,
                active_work_item_id=stage_id,
            )
            self._save_work_item(connection, item)
            self._save_task(connection, task)
            self._event(
                connection,
                task_id=task_id,
                execution_id=execution_id,
                kind="work_item.started",
                details={"stage_id": stage_id, "attempt": item.attempt_count},
                observed_at=observed_at,
            )
            return item

        return self._write(operation)

    def mark_ready(
        self,
        *,
        task_id: UUID,
        execution_id: UUID,
        stage_id: str,
        evidence: tuple[TaskEvidence, ...],
        observed_at: datetime,
    ) -> WorkItemState:
        """Retain evidence for an in-progress stage under the current execution.

        Args:
            task_id: Task UUID whose durable record is addressed.
            execution_id: Execution UUID whose record or event is addressed.
            stage_id: Exact work-stage identifier in the sealed plan.
            evidence: Exact stage evidence references to retain or verify.
            observed_at: Aware operational clock supplied by the caller.

        Returns:
            The stage state ready for deterministic verification.

        Raises:
            TaskTransitionRejected: Execution is stale or the stage is not in progress.
        """

        def operation(connection):
            task = self._task_row(connection, task_id)
            if task.latest_execution_id != execution_id:
                raise TaskTransitionRejected("work item execution is stale")
            current = self._work_item_row(connection, task_id, stage_id)
            if current.lifecycle is not WorkItemLifecycle.IN_PROGRESS:
                raise TaskTransitionRejected("work item is not in progress")
            item = self._replace_work_item(
                current,
                observed_at=observed_at,
                lifecycle=WorkItemLifecycle.READY_FOR_VERIFICATION,
                evidence=evidence,
            )
            self._save_work_item(connection, item)
            self._event(
                connection,
                task_id=task_id,
                execution_id=execution_id,
                kind="work_item.ready_for_verification",
                details={"stage_id": stage_id, "evidence_count": len(evidence)},
                observed_at=observed_at,
            )
            return item

        return self._write(operation)

    def return_to_pending(
        self, *, task_id: UUID, stage_id: str, observed_at: datetime
    ) -> WorkItemState:
        """Send back to run a stage whose stored evidence did not verify at recovery."""
        self._db_time(observed_at)

        def operation(connection):
            task = self._task_row(connection, task_id)
            current = self._work_item_row(connection, task_id, stage_id)
            if current.lifecycle is not WorkItemLifecycle.READY_FOR_VERIFICATION:
                raise TaskTransitionRejected("work item is not ready for verification")
            item = self._replace_work_item(
                current,
                observed_at=observed_at,
                lifecycle=WorkItemLifecycle.PENDING,
                evidence=(),
                failure_code=None,
                started_at=None,
            )
            self._save_work_item(connection, item)
            self._event(
                connection,
                task_id=task_id,
                execution_id=task.latest_execution_id,
                kind="work_item.evidence_unverified",
                details={"stage_id": stage_id},
                observed_at=observed_at,
            )
            return item

        return self._write(operation)

    def verify_work_item(self, receipt: TaskStageReceipt) -> TaskRecord:
        """Verify receipt authority, evidence and required kinds before completing a stage.

        Args:
            receipt: Sealed verifier receipt for the stage.

        Returns:
            Updated task state, succeeded when every stage is verified.

        Raises:
            TaskTransitionRejected: Execution, definition, verifier, ready state, evidence or
                verified status disagrees.
        """

        def operation(connection):
            task = self._task_row(connection, receipt.task_id)
            if task.latest_execution_id != receipt.execution_id:
                raise TaskTransitionRejected("stage receipt execution is stale")
            definition = next(
                item for item in task.plan.work_items if item.stage_id == receipt.stage_id
            )
            if receipt.work_item_definition_hash != definition.definition_hash:
                raise TaskTransitionRejected("stage receipt definition is stale")
            if receipt.verifier_id != definition.verifier_id:
                raise TaskTransitionRejected("stage receipt verifier is unauthorized")
            current = self._work_item_row(connection, receipt.task_id, receipt.stage_id)
            if current.lifecycle is not WorkItemLifecycle.READY_FOR_VERIFICATION:
                raise TaskTransitionRejected("work item is not ready for verification")
            if receipt.evidence != current.evidence:
                raise TaskTransitionRejected("stage receipt evidence differs from work item")
            evidence_kinds = {item.evidence_kind for item in receipt.evidence}
            missing = set(definition.required_evidence_kinds) - evidence_kinds
            if missing:
                raise TaskTransitionRejected(
                    "stage receipt omits required evidence: " + ", ".join(sorted(missing))
                )
            if receipt.status != "VERIFIED":
                raise TaskTransitionRejected("only a verified receipt can complete work")
            verified = {
                "receipt_hash": receipt.receipt_hash,
                "execution_id": str(receipt.execution_id),
                "stage_id": receipt.stage_id,
                "observed_at": self._db_time(receipt.observed_at),
            }
            self._record("stage.verified", receipt.task_id, verified, text=self._json(receipt))
            item = self._replace_work_item(
                current,
                observed_at=receipt.observed_at,
                lifecycle=WorkItemLifecycle.VERIFIED,
            )
            self._save_work_item(connection, item)
            remaining = connection.execute(
                """
                SELECT COUNT(*) FROM workspace_task_work_item
                WHERE task_id = ? AND lifecycle != 'VERIFIED'
                """,
                [str(receipt.task_id)],
            ).fetchone()[0]
            # Items run at once (a runner's declared width): the active one stays
            # active while it runs, and a verified one hands over to the earliest
            # still in progress; one at a time, that is none.
            active = task.active_work_item_id
            if active == receipt.stage_id:
                row = connection.execute(
                    """
                    SELECT stage_id FROM workspace_task_work_item
                    WHERE task_id = ? AND lifecycle = 'IN_PROGRESS'
                    ORDER BY ordinal LIMIT 1
                    """,
                    [str(receipt.task_id)],
                ).fetchone()
                active = None if row is None else str(row[0])
            task = self._replace_task(
                task,
                observed_at=receipt.observed_at,
                lifecycle=(TaskLifecycle.SUCCEEDED if remaining == 0 else task.lifecycle),
                active_work_item_id=active,
            )
            self._save_task(connection, task)
            self._event(
                connection,
                task_id=receipt.task_id,
                execution_id=receipt.execution_id,
                kind="work_item.verified",
                details={"stage_id": receipt.stage_id, "receipt_hash": receipt.receipt_hash},
                observed_at=receipt.observed_at,
            )
            if task.lifecycle is TaskLifecycle.SUCCEEDED:
                self._event(
                    connection,
                    task_id=receipt.task_id,
                    execution_id=receipt.execution_id,
                    kind="task.succeeded",
                    details={"plan_hash": task.plan.plan_hash},
                    observed_at=receipt.observed_at,
                )
            return task

        return self._write(operation)

    def block_work_item(
        self,
        *,
        task_id: UUID,
        execution_id: UUID,
        stage_id: str,
        failure_code: str,
        observed_at: datetime,
        deferred: bool = False,
        failure_cause: StageFailureCause | None = None,
    ) -> TaskRecord:
        """Record an active stage failure and a deferred or blocked task disposition.

        Args:
            task_id: Task UUID whose durable record is addressed.
            execution_id: Execution UUID whose record or event is addressed.
            stage_id: Exact work-stage identifier in the sealed plan.
            failure_code: Stable failure cause to retain.
            observed_at: Aware operational clock supplied by the caller.
            deferred: Whether to retain a deferred task rather than a terminal block.
            failure_cause: What the owner saw beside the code, kept on the work item and its
                event.

        Returns:
            The updated task with its stable failure code.

        Raises:
            TaskTransitionRejected: The stage is neither in progress nor ready for verification.
        """

        def operation(connection):
            task = self._task_row(connection, task_id)
            current = self._work_item_row(connection, task_id, stage_id)
            if current.lifecycle not in {
                WorkItemLifecycle.IN_PROGRESS,
                WorkItemLifecycle.READY_FOR_VERIFICATION,
            }:
                raise TaskTransitionRejected("only active work can be blocked")
            item = self._replace_work_item(
                current,
                observed_at=observed_at,
                lifecycle=WorkItemLifecycle.BLOCKED,
                failure_code=failure_code,
                failure_cause=failure_cause,
            )
            task = self._replace_task(
                task,
                observed_at=observed_at,
                lifecycle=(TaskLifecycle.DEFERRED if deferred else TaskLifecycle.BLOCKED),
                failure_code=failure_code,
            )
            self._save_work_item(connection, item)
            self._save_task(connection, task)
            details: dict[str, object] = {"stage_id": stage_id, "failure_code": failure_code}
            if failure_cause is not None:
                details["failure_cause"] = failure_cause.model_dump(mode="json")
            self._event(
                connection,
                task_id=task_id,
                execution_id=execution_id,
                kind="work_item.blocked",
                details=details,
                observed_at=observed_at,
            )
            return task

        return self._write(operation)

    def request_cancel(
        self,
        *,
        task_id: UUID,
        expected_task_hash: str,
        observed_at: datetime,
    ) -> tuple[TaskRecord, TaskCommand]:
        """Record version-confirmed cancellation or reuse an existing cancellation command.

        Args:
            task_id: Task UUID whose durable record is addressed.
            expected_task_hash: Exact task version the caller confirmed.
            observed_at: Aware operational clock supplied by the caller.

        Returns:
            The task and command; queued and ledger-rebuilt work cancels immediately,
            executing work awaits enactment.

        Raises:
            TaskVersionStale: The confirmed task version changed.
            TaskTransitionRejected: The task is terminal and has no reusable cancellation command.
        """

        def operation(connection):
            current = self._task_row(connection, task_id)
            existing = connection.execute(
                """
                SELECT command_json FROM workspace_task_command
                WHERE task_id = ?
                  AND json_extract(command_json, '$.kind') = 'CANCEL_REQUESTED'
                ORDER BY accepted_us DESC LIMIT 1
                """,
                [str(task_id)],
            ).fetchone()
            if existing is not None and current.lifecycle in {
                TaskLifecycle.CANCEL_REQUESTED,
                TaskLifecycle.CANCELLED,
            }:
                return current, TaskCommand.model_validate_json(existing[0])
            if current.record_hash != expected_task_hash:
                raise TaskVersionStale("cancel command observed a stale task version")
            if not _cancel_available(current):
                raise TaskTransitionRejected("terminal task cannot be cancelled")
            command = TaskCommand.from_identity(
                command_id=UUID(uuid7()),
                task_id=task_id,
                expected_task_hash=expected_task_hash,
                kind=TaskCommandKind.CANCEL_REQUESTED,
                requested_at=observed_at,
            )
            if current.lifecycle in {TaskLifecycle.QUEUED, TaskLifecycle.BLOCKED}:
                task = self._replace_task(
                    current,
                    observed_at=observed_at,
                    lifecycle=TaskLifecycle.CANCELLED,
                    active_work_item_id=None,
                    failure_code=(
                        current.failure_code
                        if current.lifecycle is TaskLifecycle.BLOCKED
                        else "TASK_CANCELLED_BEFORE_START"
                    ),
                )
                resolution = "CANCELLED"
                resolved_at = self._db_time(observed_at)
            else:
                task = self._replace_task(
                    current,
                    observed_at=observed_at,
                    lifecycle=TaskLifecycle.CANCEL_REQUESTED,
                )
                resolution = None
                resolved_at = None
            self._save_task(connection, task)
            accepted = {
                "command_id": str(command.command_id),
                "command_hash": command.command_hash,
                "accepted_at": self._db_time(observed_at),
            }
            self._record("command.accepted", task_id, accepted, text=self._json(command))
            if resolution is not None:
                self._resolved(task_id, command.command_id, resolved_at, resolution)
            self._event(
                connection,
                task_id=task_id,
                execution_id=current.latest_execution_id,
                kind="task.cancel_requested",
                details={"command_hash": command.command_hash},
                observed_at=observed_at,
            )
            return task, command

        task, command = self._write(operation, control_only=True)
        if task.lifecycle is TaskLifecycle.CANCEL_REQUESTED:
            execution_port().cancel(task.latest_execution_id)
        return task, command

    def finalize_cancel_after_writer_stopped(
        self, *, task_id: UUID, expected_task_hash: str, observed_at: datetime
    ) -> TaskRecord:
        """The exclusive Host calls only after its command has returned.

        Writer inactivity is established by the dispatch owner under its enqueue
        lock, never from a PID, timeout or task label. The exact version and an
        accepted cancellation are checked again in the registry transaction.
        """

        def operation(connection):
            current = self._task_row(connection, task_id)
            if current.record_hash != expected_task_hash:
                raise TaskVersionStale("cancel finalization observed a stale task version")
            if current.lifecycle is not TaskLifecycle.CANCEL_REQUESTED:
                raise TaskTransitionRejected("task has no pending cancellation")
            accepted = connection.execute(
                """SELECT 1 FROM workspace_task_command WHERE task_id = ?
                   AND json_extract(command_json, '$.kind') = 'CANCEL_REQUESTED'
                   AND resolution IS NULL LIMIT 1""",
                [str(task_id)],
            ).fetchone()
            if accepted is None:
                raise TaskTransitionRejected("task_control.orphan_cancel_command_absent")
            return self._cancel_task(
                connection, current, observed_at=observed_at, boundary="WRITER_STOPPED"
            )

        return self._write(operation, control_only=True)

    def apply_cancel(self, *, task_id: UUID, observed_at: datetime) -> TaskRecord:
        """Enact the pending cancellation under the task mutation owner.

        Args:
            task_id: Task UUID whose durable record is addressed.
            observed_at: Aware operational clock supplied by the caller.

        Returns:
            The cancelled task record.

        Raises:
            TaskTransitionRejected: The task has no pending cancellation.
        """

        def operation(connection):
            current = self._task_row(connection, task_id)
            if current.lifecycle is not TaskLifecycle.CANCEL_REQUESTED:
                raise TaskTransitionRejected("task has no pending cancellation")
            return self._cancel_task(connection, current, observed_at=observed_at)

        return self._write(operation)

    def _cancel_task(
        self,
        connection,
        current: TaskRecord,
        *,
        observed_at: datetime,
        boundary: Literal["SAFE_CHECKPOINT", "WRITER_STOPPED"] = "SAFE_CHECKPOINT",
    ) -> TaskRecord:
        task_id = current.task_id
        failure_code = (
            "TASK_CANCELLED_AT_SAFE_CHECKPOINT"
            if boundary == "SAFE_CHECKPOINT"
            else "TASK_CANCELLED_AFTER_WRITER_STOPPED"
        )
        rows = connection.execute(
            """
                SELECT state_json FROM workspace_task_work_item
                WHERE task_id = ? AND lifecycle != 'VERIFIED'
                """,
            [str(task_id)],
        ).fetchall()
        for row in rows:
            item = WorkItemState.model_validate_json(row[0])
            cancelled = self._replace_work_item(
                item,
                observed_at=observed_at,
                lifecycle=WorkItemLifecycle.CANCELLED,
                failure_code=failure_code,
            )
            self._save_work_item(connection, cancelled)
        task = self._replace_task(
            current,
            observed_at=observed_at,
            lifecycle=TaskLifecycle.CANCELLED,
            active_work_item_id=None,
            failure_code=failure_code,
        )
        self._save_task(connection, task)
        for (command_id,) in connection.execute(
            "SELECT command_id FROM workspace_task_command WHERE task_id = ? AND resolution IS NULL"
            " ORDER BY accepted_us, command_id",
            [str(task_id)],
        ).fetchall():
            self._resolved(task_id, command_id, self._db_time(observed_at), "CANCELLED")
        self._event(
            connection,
            task_id=task_id,
            execution_id=task.latest_execution_id,
            kind="task.cancelled",
            details={"boundary": boundary},
            observed_at=observed_at,
        )
        return task

    def reconcile_after_writer_acquisition(
        self, *, observed_at: datetime
    ) -> tuple[TaskRecord, ...]:
        """Called only by a newly acquired exclusive workspace writer session.

        No timeout, PID guess, domain execution or implicit retry of waiting
        tasks. The transaction retains original inputs, receipts and execution
        identities; only orphan state and already-accepted cancellation move.
        """
        self._db_time(observed_at)

        def operation(connection):
            changed = []
            rows = connection.execute(
                "SELECT t.task_id, t.record_json FROM workspace_task t "
                "LEFT JOIN workspace_task_projection p ON p.task_id = t.task_id "
                "WHERE t.workspace_id = ? AND (t.lifecycle NOT IN ('SUCCEEDED', 'BLOCKED',"
                " 'CANCELLED') OR p.task_id IS NULL"
                " OR p.task_record_hash IS NOT json_extract(t.record_json, '$.record_hash'))",
                [self.workspace_id],
            ).fetchall()
            for row in rows:
                task = self._verified_record(str(row[0]), row[1])
                if task.lifecycle in _TERMINAL_TASKS:
                    continue
                if task.lifecycle in {TaskLifecycle.RUNNING, TaskLifecycle.CANCEL_REQUESTED}:
                    prior = self._execution_row(connection, task.latest_execution_id)
                    if prior is None:
                        raise TaskTransitionRejected("task_control.orphan_execution_absent")
                    if (
                        prior.task_id != task.task_id
                        or prior.execution_id != task.latest_execution_id
                    ):
                        raise TaskTransitionRejected("task_control.orphan_execution_mismatch")
                pending = connection.execute(
                    "SELECT command_json FROM workspace_task_command "
                    "WHERE task_id = ? AND resolution IS NULL",
                    [str(task.task_id)],
                ).fetchall()
                commands = tuple(TaskCommand.model_validate_json(value[0]) for value in pending)
                if any(command.task_id != task.task_id for command in commands):
                    raise TaskTransitionRejected("task_control.orphan_command_mismatch")
                cancel = any(
                    command.kind is TaskCommandKind.CANCEL_REQUESTED for command in commands
                )
                if task.lifecycle is TaskLifecycle.CANCEL_REQUESTED and not cancel:
                    raise TaskTransitionRejected("task_control.orphan_cancel_command_absent")
                if cancel:
                    changed.append(
                        self._cancel_task(
                            connection, task, observed_at=observed_at, boundary="WRITER_STOPPED"
                        )
                    )
                elif task.lifecycle is TaskLifecycle.RUNNING:
                    recovered = self._replace_task(
                        task,
                        observed_at=observed_at,
                        lifecycle=TaskLifecycle.RECOVERY_REQUIRED,
                        failure_code="TASK_WRITER_INTERRUPTED",
                    )
                    self._save_task(connection, recovered)
                    self._event(
                        connection,
                        task_id=task.task_id,
                        execution_id=task.latest_execution_id,
                        kind="task.recovery_required",
                        details={
                            "failure_code": "TASK_WRITER_INTERRUPTED",
                            "boundary": "WRITER_ACQUIRED",
                        },
                        observed_at=observed_at,
                    )
                    changed.append(recovered)
            return tuple(changed), tuple(
                self._verified_record(str(row[0]), row[1]).task_id for row in rows
            )

        changed, projection_ids = self._write(operation)
        # Repair derived status too, including a crash after state commit but
        # before projection publication. Terminal Task records remain untouched.
        for task_id in projection_ids:
            self.safe_projection(task_id)
        return changed

    def heartbeat(
        self, *, execution_id: UUID, worker_instance_id: UUID, observed_at: datetime
    ) -> TaskExecution:
        """Update the business heartbeat only for the execution's bound worker.

        Args:
            execution_id: Execution UUID whose record or event is addressed.
            worker_instance_id: Worker UUID bound to the execution.
            observed_at: Aware operational clock supplied by the caller.

        Returns:
            The resealed execution record.

        Raises:
            KeyError: The execution record is absent.
            TaskTransitionRejected: The worker identity is stale.
        """

        def operation(connection):
            current = self._execution_row(connection, execution_id)
            if current is None:
                raise KeyError(execution_id)
            if current.worker_instance_id != worker_instance_id:
                raise TaskTransitionRejected("heartbeat worker identity is stale")
            connection.execute(  # working state: a beat is not history (no event)
                "INSERT INTO workspace_task_heartbeat_at VALUES (?, ?) ON CONFLICT (execution_id)"
                " DO UPDATE SET last_heartbeat_us = excluded.last_heartbeat_us",
                [str(execution_id), self._us(observed_at)],
            )
            return self._beat(current, observed_at)

        return self._write(operation)

    def mark_recovery_required(
        self,
        *,
        task_id: UUID,
        failure_code: str,
        observed_at: datetime,
        allow_blocked: bool = False,
        expected_task_hash: str | None = None,
        failure_cause: StageFailureCause | None = None,
        failed_stage_id: str | None = None,
    ) -> TaskRecord:
        """Owe the Task a recovery; reopen a BLOCKED one only on its confirmed version.

        `expected_task_hash` is the record version the person confirmed the
        reopening against. It is checked inside this write, so a BLOCKED Task
        that moved meanwhile -- reopened by another confirmation, started,
        cancelled, or blocked again on a later attempt -- is refused here with
        nothing applied to whatever it became. Versionless callers (the runner's
        own failure path) are unchanged.
        """

        def operation(connection):
            current = self._task_row(connection, task_id)
            if expected_task_hash is not None and current.record_hash != expected_task_hash:
                raise TaskVersionStale("task_control.recovery_version_stale")
            if (
                failure_code == "HOST_STOP_REQUESTED"
                and current.lifecycle is TaskLifecycle.CANCEL_REQUESTED
            ):
                return self._cancel_task(connection, current, observed_at=observed_at)
            if current.lifecycle in _TERMINAL_TASKS and not (
                allow_blocked
                and current.lifecycle is TaskLifecycle.BLOCKED
                and current.failure_code == failure_code
            ):
                return current
            active_work_item_id = current.active_work_item_id
            if failure_cause is not None and failed_stage_id is not None:
                item = self._work_item_row(connection, task_id, failed_stage_id)
                if item.lifecycle in {
                    WorkItemLifecycle.IN_PROGRESS,
                    WorkItemLifecycle.READY_FOR_VERIFICATION,
                }:
                    item = self._replace_work_item(
                        item,
                        observed_at=observed_at,
                        lifecycle=item.lifecycle,
                        failure_code=failure_code,
                        failure_cause=failure_cause,
                    )
                    self._save_work_item(connection, item)
                    active_work_item_id = failed_stage_id
            task = self._replace_task(
                current,
                observed_at=observed_at,
                lifecycle=TaskLifecycle.RECOVERY_REQUIRED,
                active_work_item_id=active_work_item_id,
                failure_code=failure_code,
            )
            self._save_task(connection, task)
            self._event(
                connection,
                task_id=task_id,
                execution_id=task.latest_execution_id,
                kind="task.recovery_required",
                details={"failure_code": failure_code},
                observed_at=observed_at,
            )
            return task

        return self._write(operation)

    def task(self, task_id: UUID) -> TaskRecord:
        """Resolve the validated durable task record through the bounded read cache.

        Args:
            task_id: Task UUID whose durable record is addressed.

        Returns:
            The requested task record.

        Raises:
            TaskNotFoundError: The task row is absent.
        """
        return self._answered(
            ("task", task_id), lambda: self._read(lambda c: self._task_row(c, task_id))
        )

    def _wakes_from(
        self, connection: sqlite3.Connection, task_id: UUID | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Fold each Codex wake's journal events into its current receipt (WAKE)."""
        rows = connection.execute(
            "SELECT e.task_id, e.kind, e.details_json, e.recorded_us FROM workspace_task_event e"
            " JOIN workspace_task t ON t.task_id = e.task_id"
            f" WHERE t.workspace_id = ? AND e.kind IN ({', '.join('?' * len(_WAKE_STATES))})"
            + (" AND e.task_id = ?" if task_id is not None else "")
            + " ORDER BY e.task_id, e.sequence",
            [self.workspace_id, *_WAKE_STATES, *([str(task_id)] if task_id is not None else [])],
        ).fetchall()
        wakes: dict[str, dict[str, Any]] = {}
        for stored_task, kind, details_json, recorded_us in rows:
            details = json.loads(details_json)
            held = wakes.get(details["registration_id"])
            if held is None and kind == "task.wake_registered":
                held = wakes[details["registration_id"]] = {"task_id": stored_task}
            if held is None or held["task_id"] != stored_task:
                raise TaskControlDatabaseAuthorityError("task_control.wake_registration_invalid")
            held.update(
                details,
                state=_WAKE_STATES[kind],
                recorded_at=(_EPOCH + timedelta(microseconds=recorded_us)).isoformat(),
            )
        return tuple(wakes.values())

    def register_wake(
        self,
        task_id: UUID,
        thread_id: str,
        read_command: str,
        *,
        observed_at: datetime,
        codex_path: str | None = None,
    ) -> dict[str, Any]:
        """Hold one wake; update a pending command/path, reuse a settled lifecycle."""
        self._db_time(observed_at)

        def operation(connection):  # type: ignore[no-untyped-def]
            task = self._task_row(connection, task_id)
            registration_id = uuid7()
            held = [w for w in self._wakes_from(connection, task_id) if w["thread_id"] == thread_id]
            if held:
                last = held[-1]
                if last["state"] == "PENDING" and (
                    last["read_command"] != read_command or last.get("codex_path") != codex_path
                ):
                    registration_id = last["registration_id"]
                elif last["state"] == "PENDING" or last["lifecycle"] == task.lifecycle.value:
                    return last
            self._event(
                connection,
                task_id=task_id,
                execution_id=None,
                kind="task.wake_registered",
                details={
                    "registration_id": registration_id,
                    "thread_id": thread_id,
                    "read_command": read_command,
                    "codex_path": codex_path,
                    "lifecycle": task.lifecycle.value,
                },
                observed_at=observed_at,
            )
            return next(
                w
                for w in self._wakes_from(connection, task_id)
                if w["registration_id"] == registration_id
            )

        return self._write(operation, control_only=True)

    def wake_registrations(self, task_id: UUID | None = None) -> tuple[dict[str, Any], ...]:
        """Every Codex wake's current receipt, or one Task's."""
        return self._read(lambda connection: self._wakes_from(connection, task_id))

    def claim_wake(
        self,
        registration_id: str,
        task_id: UUID,
        *,
        event: str,
        lifecycle: str,
        observed_at: datetime,
    ) -> dict[str, Any] | None:
        """Commit one send attempt before the queue call, while the Task is still in `lifecycle`.

        A wake is attempted once: a send the Host does not finish reads as uncertain at its next
        start (`interrupt_wakes`), never as pending again.
        """
        self._db_time(observed_at)

        def operation(connection):  # type: ignore[no-untyped-def]
            held = next(
                (
                    w
                    for w in self._wakes_from(connection, task_id)
                    if w["registration_id"] == registration_id
                ),
                None,
            )
            if (
                held is None
                or held["state"] != "PENDING"
                or self._task_row(connection, task_id).lifecycle.value != lifecycle
            ):
                return None
            attempt: dict[str, object] = {
                "registration_id": registration_id,
                "event": event,
                "lifecycle": lifecycle,
            }
            self._event(
                connection,
                task_id=task_id,
                execution_id=None,
                kind="task.wake_attempted",
                details=attempt,
                observed_at=observed_at,
            )
            return {**held, **attempt, "state": "ATTEMPTED"}

        return self._write(operation, control_only=True)

    def finish_wake(
        self, registration_id: str, task_id: UUID, result: dict[str, Any], *, observed_at: datetime
    ) -> None:
        """Record an attempted wake's one outcome: queued, or its named failure."""
        self._db_time(observed_at)

        def operation(connection):  # type: ignore[no-untyped-def]
            held = next(
                (
                    w
                    for w in self._wakes_from(connection, task_id)
                    if w["registration_id"] == registration_id
                ),
                None,
            )
            if held is None or held["state"] != "ATTEMPTED":
                return
            self._event(
                connection,
                task_id=task_id,
                execution_id=None,
                kind="task.wake_delivered" if result["delivered"] else "task.wake_undelivered",
                details={"registration_id": registration_id, "result": result},
                observed_at=observed_at,
            )

        self._write(operation, control_only=True)

    def interrupt_wakes(self, *, observed_at: datetime) -> tuple[dict[str, Any], ...]:
        """Close each send a stopped Host left attempted as uncertain, without sending again."""
        result = {"channel": "codex-queue", "delivered": False, "failure": WAKE_UNCERTAIN}
        interrupted = [w for w in self.wake_registrations() if w["state"] == "ATTEMPTED"]
        for wake in interrupted:
            self.finish_wake(
                wake["registration_id"], UUID(wake["task_id"]), result, observed_at=observed_at
            )
        return tuple({**held, "state": "UNDELIVERED", "result": result} for held in interrupted)

    @staticmethod
    def _recovery_links_from(
        connection: sqlite3.Connection, workspace_id: str, task_id: UUID | None = None
    ) -> tuple[TaskRecoveryLink, ...]:
        """Read and verify links in one journal query, optionally by source Task."""
        if task_id is None:
            rows = connection.execute(
                """
                SELECT e.task_id, e.details_json, e.recorded_us FROM workspace_task_event e
                JOIN workspace_task t ON t.task_id = e.task_id
                WHERE t.workspace_id = ? AND e.kind = ?
                ORDER BY e.recorded_us, e.task_id, e.sequence
                """,
                [workspace_id, _RECOVERY_LINK_EVENT],
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT task_id, details_json, recorded_us FROM workspace_task_event
                WHERE kind = ? AND task_id = ? ORDER BY recorded_us, sequence
                """,
                [_RECOVERY_LINK_EVENT, str(task_id)],
            ).fetchall()
        links: list[TaskRecoveryLink] = []
        for stored_task_id, details_json, event_time in rows:
            try:
                link = TaskRecoveryLink.model_validate_json(details_json)
                if str(link.source_task_id) != str(stored_task_id):
                    raise ValueError("task recovery link source differs from its journal row")
                if (
                    not isinstance(event_time, int)
                    or DuckDbTaskControlRegistry._us(link.recorded_at) != event_time
                ):
                    raise ValueError("task recovery link clock differs from its journal row")
            except (TypeError, ValueError) as error:
                raise TaskControlDatabaseAuthorityError(
                    "task_control.recovery_link_invalid"
                ) from error
            links.append(link)
        return tuple(links)

    def record_admission_subject(
        self, task_id: UUID, document: dict[str, Any], *, observed_at: datetime
    ) -> None:
        """Keep an owner's sealed admission subject beside its canonical Task."""
        encoded = self._json(document)
        self._db_time(observed_at)
        if document.get("task_id") != str(task_id) or self._db_time(
            datetime.fromisoformat(document["admitted_at"])
        ) != self._db_time(observed_at):
            raise TaskControlDatabaseAuthorityError("task_control.database_authority_unreadable")

        def operation(connection):  # type: ignore[no-untyped-def]
            self._task_row(connection, task_id)
            if connection.execute(
                "SELECT 1 FROM workspace_task_event "
                "WHERE task_id = ? AND kind = ? AND details_json = ?",
                [str(task_id), _ADMISSION_SUBJECT_EVENT, encoded],
            ).fetchone():
                return
            self._event(
                connection,
                task_id=task_id,
                execution_id=None,
                kind=_ADMISSION_SUBJECT_EVENT,
                details=json.loads(encoded),
                observed_at=observed_at,
            )

        self._write(operation, control_only=True)

    def admission_subjects(self, task_id: UUID) -> tuple[dict[str, Any], ...]:
        """Read this Task's admission documents; the kind owner verifies their seals."""
        rows = self._read(
            lambda connection: connection.execute(
                "SELECT details_json, recorded_us FROM workspace_task_event "
                "WHERE task_id = ? AND kind = ? ORDER BY recorded_us, sequence",
                [str(task_id), _ADMISSION_SUBJECT_EVENT],
            ).fetchall()
        )
        subjects: list[dict[str, Any]] = []
        for document, recorded_us in rows:
            try:
                subject = json.loads(document)
                if (
                    subject["task_id"] != str(task_id)
                    or self._us(datetime.fromisoformat(subject["admitted_at"])) != recorded_us
                ):
                    raise ValueError("admission subject differs from its journal row")
            except (KeyError, TypeError, ValueError) as error:
                raise TaskControlDatabaseAuthorityError(
                    "task_control.database_authority_unreadable"
                ) from error
            subjects.append(subject)
        return tuple(subjects)

    def recovery_links(self, task_id: UUID | None = None) -> tuple[TaskRecoveryLink, ...]:
        """Read sealed recovery provenance from the existing Task event journal.

        With no Task id, one query reads the full link set for a batch projection.
        A supplied id selects links whose source is that Task.
        """
        links = self._answered(
            ("recovery_links", task_id),
            lambda: self._read(
                lambda connection: self._recovery_links_from(connection, self.workspace_id, task_id)
            ),
        )
        # A frozen contract can still carry a mutable request dictionary. Never expose
        # the owner's cached seal to a caller that may edit its returned request.
        return tuple(link.model_copy(deep=True) for link in links)

    def record_recovery_link(
        self,
        *,
        source_task_id: UUID,
        source_record_hash: str,
        admission_request: dict[str, Any],
        successor_task_id: UUID | None = None,
        observed_at: datetime,
    ) -> TaskRecoveryLink:
        """Append a sealed preview or its matching confirmed successor to the event journal.

        A preview is admitted only for the current stopped source version. A confirmed
        successor must match a prior preview and exist as a distinct canonical Task of
        the same kind. Its source may have advanced after admission; that link retains
        the confirmed source hash so readers can leave the newer version unresolved.
        """
        self._db_time(observed_at)
        normalized_request = json.loads(self._json(admission_request))
        if not isinstance(normalized_request, dict):
            raise ValueError("task recovery link request must be a mapping")
        link = TaskRecoveryLink.create(
            source_task_id=source_task_id,
            source_record_hash=source_record_hash,
            admission_request=normalized_request,
            successor_task_id=successor_task_id,
            recorded_at=observed_at.astimezone(UTC),
        )

        def operation(connection):  # type: ignore[no-untyped-def]
            source = self._task_row(connection, source_task_id)
            links = self._recovery_links_from(connection, self.workspace_id, source_task_id)
            existing = next(
                (
                    current
                    for current in links
                    if current.source_record_hash == link.source_record_hash
                    and current.admission_request == link.admission_request
                    and current.successor_task_id == link.successor_task_id
                ),
                None,
            )
            if successor_task_id is None:
                if source.record_hash != source_record_hash:
                    raise TaskVersionStale("task_control.recovery_link_source_version_stale")
                if source.lifecycle not in _RECOVERY_LINK_STOPPED:
                    raise TaskTransitionRejected("task_control.recovery_link_source_not_stopped")
                if existing is not None:
                    return existing
            else:
                successor = self._task_row(connection, successor_task_id)
                if source_task_id == successor_task_id or successor.task_kind != source.task_kind:
                    raise TaskTransitionRejected("task_control.recovery_link_successor_invalid")
                if not any(
                    current.successor_task_id is None
                    and current.source_record_hash == source_record_hash
                    and current.admission_request == normalized_request
                    for current in links
                ):
                    raise TaskTransitionRejected("task_control.recovery_link_preview_required")
                if existing is not None:
                    return existing

            self._event(
                connection,
                task_id=source_task_id,
                execution_id=None,
                kind=_RECOVERY_LINK_EVENT,
                details=link.model_dump(mode="json"),
                observed_at=observed_at,
            )
            return link

        return self._write(operation, control_only=True)

    def submitted_by(self, task_id: UUID) -> SubmittingAgent | None:
        """The agent session that submitted a Task, where its request named one."""

        def read() -> SubmittingAgent | None:
            row = self._read(
                lambda connection: connection.execute(
                    "SELECT submitted_by_json FROM workspace_task WHERE task_id = ?",
                    [str(task_id)],
                ).fetchone()
            )
            if row is None:
                raise TaskNotFoundError(task_id)
            return None if row[0] is None else SubmittingAgent.model_validate_json(row[0])

        return self._answered(("submitted_by", task_id), read)

    def submitted_by_session(self, session: str) -> tuple[tuple[UUID, SubmittingAgent], ...]:
        """The Tasks one agent session submitted, newest admitted first, each with its submitter.

        One read of the submitters' column; no Task record is read or validated here.
        """

        def read() -> tuple[tuple[UUID, SubmittingAgent], ...]:
            rows = self._read(
                lambda connection: connection.execute(
                    "SELECT task_id, submitted_by_json FROM workspace_task WHERE workspace_id = ?"
                    " AND json_extract(submitted_by_json, '$.session') = ?"
                    " ORDER BY admission_sequence DESC NULLS LAST, admitted_us DESC, task_id DESC",
                    [self.workspace_id, session],
                ).fetchall()
            )
            return tuple(
                (UUID(row[0]), SubmittingAgent.model_validate_json(row[1])) for row in rows
            )

        return self._answered(("submitted_by_session", session), read)

    def work_items(self, task_id: UUID) -> tuple[WorkItemState, ...]:
        """Read the task's work-item states from the durable authority.

        Args:
            task_id: Task UUID whose durable record is addressed.

        Returns:
            The recorded work-item states in plan order.
        """
        return self._read(lambda connection: self._work_items_from(connection, task_id))

    def consecutive_interruptions(self, task_id: UUID, stage_id: str) -> int:
        """Count consecutive interruptions, excluding the open attempt.

        Normal ends reset the count; an intentional Host stop does not spend it.
        """

        def read(connection) -> int:  # type: ignore[no-untyped-def]
            rows = connection.execute(
                """
                SELECT kind, details_json FROM workspace_task_event
                WHERE task_id = ? ORDER BY sequence
                """,
                [str(task_id)],
            ).fetchall()
            count, open_attempt = 0, False
            for kind, details in rows:
                stage = json.loads(details).get("stage_id") if details else None
                if kind == "work_item.started" and stage == stage_id:
                    open_attempt = True
                elif kind == "task.recovery_required" and open_attempt:
                    failure = json.loads(details).get("failure_code") if details else None
                    count, open_attempt = count + (failure != "HOST_STOP_REQUESTED"), False
                elif stage == stage_id and kind in _NORMAL_ENDS:
                    count, open_attempt = 0, False
            return count

        return int(self._read(read))

    def task_with_work_items(self, task_id: UUID) -> tuple[TaskRecord, tuple[WorkItemState, ...]]:
        """Read task and work-item state at one version under a single read scope.

        A Task's record and its work board read at one version, in one read scope, and
        answered again while the database files show no write since (W10).
        """

        def read() -> tuple[TaskRecord, tuple[WorkItemState, ...]]:
            with self.read_scope():
                return self._read(
                    lambda connection: (
                        self._task_row(connection, task_id),
                        self._work_items_from(connection, task_id),
                    )
                )

        return self._answered(("task_with_work_items", task_id), read)

    @staticmethod
    def _work_items_from(connection, task_id: UUID) -> tuple[WorkItemState, ...]:  # type: ignore[no-untyped-def]
        rows = connection.execute(
            """
            SELECT state_json FROM workspace_task_work_item
            WHERE task_id = ? ORDER BY ordinal
            """,
            [str(task_id)],
        ).fetchall()
        return tuple(WorkItemState.model_validate_json(row[0]) for row in rows)

    def execution(self, execution_id: UUID) -> TaskExecution:
        """Read and validate one durable execution record.

        Args:
            execution_id: Execution UUID whose record or event is addressed.

        Returns:
            The stored execution record.

        Raises:
            KeyError: The execution row is absent.
        """

        def operation(connection):
            execution = self._execution_row(connection, execution_id)
            if execution is None:
                raise KeyError(execution_id)
            return execution

        return self._read(operation)

    def tasks(self) -> tuple[TaskRecord, ...]:
        """Resolve durable task records through the bounded read cache.

        Returns:
            The recorded task sequence.
        """
        return self._answered(
            "tasks", lambda: self._read(lambda c: self._tasks_from(c, self.workspace_id))
        )

    def record_collection(self) -> TaskRecordBatch:
        """Read every canonical row, retaining peers beside named unreadable records."""
        return self._answered(
            "record_collection",
            lambda: self._read(lambda c: self._record_collection_from(c, self.workspace_id)),
        )

    def has_tasks(self, task_kind: str) -> bool:
        """Whether any Task of this kind is recorded: one count, no record validated."""

        def count(connection) -> bool:  # type: ignore[no-untyped-def]
            row = connection.execute(
                "SELECT count(*) FROM workspace_task WHERE workspace_id = ? AND task_kind = ?",
                [self.workspace_id, task_kind],
            ).fetchone()
            return bool(row[0])

        return bool(self._answered(("has_tasks", task_kind), lambda: self._read(count)))

    @staticmethod
    def _tasks_from(connection: sqlite3.Connection, workspace_id: str) -> tuple[TaskRecord, ...]:
        batch = DuckDbTaskControlRegistry._record_collection_from(connection, workspace_id)
        if batch.refused_task_ids:
            raise TaskRecordAuthorityError(batch.refused_task_ids)
        return batch.records

    @staticmethod
    def _record_collection_from(
        connection: sqlite3.Connection, workspace_id: str
    ) -> TaskRecordBatch:
        records: list[TaskRecord] = []
        refused_task_ids: list[str] = []
        for stored_id, document in connection.execute(
            "SELECT task_id, record_json FROM workspace_task WHERE workspace_id = ?"
            " ORDER BY admission_sequence NULLS FIRST, admitted_us, task_id",
            [workspace_id],
        ).fetchall():
            try:
                record = DuckDbTaskControlRegistry._verified_record(str(stored_id), document)
            except TaskRecordAuthorityError:
                refused_task_ids.append(str(stored_id))
                continue
            records.append(record)
        return TaskRecordBatch(tuple(records), tuple(refused_task_ids))

    @classmethod
    def read_existing_tasks(cls, database_path: Path) -> tuple[TaskRecord, ...]:
        """A workspace's Tasks, read without taking its data root: a 0.2.7 store's not yet moved
        onto the log, else the log's for its folder (none when it knows no such workspace)."""
        if (stored := stored_records(database_path)) is not None:
            return stored
        with log_of(database_path) as (connection, workspace_id):
            return () if workspace_id is None else cls._tasks_from(connection, workspace_id)

    def active_task(self) -> TaskRecord | None:
        """The first Task holding a running place, else the first one waiting for its recovery."""
        tasks = [task for task in self.tasks() if task.lifecycle in _ACTIVE_TASKS]
        holding = [task for task in tasks if task.lifecycle in _RUNNING_PLACE]
        return holding[0] if holding else (tasks[0] if tasks else None)

    def running_place_holder(self) -> TaskRecord | None:
        """The first Task holding a running place, if one does.

        A deferred one holds it too, which no command drives while it waits for its retry time.
        """
        active = self.active_task()
        return active if active is not None and active.lifecycle in _RUNNING_PLACE else None

    def queued_task(self) -> TaskRecord | None:
        """The selected interactive Task or the first Task waiting in admission order."""

        def head(connection) -> TaskRecord | None:  # type: ignore[no-untyped-def]
            row = _queued(connection, self.workspace_id)
            if row is None:
                return None
            stored_task_id = str(row[0])
            try:
                task = TaskRecord.model_validate_json(row[1])
            except (TypeError, ValueError) as error:
                raise TaskQueueHeadAuthorityError(stored_task_id) from error
            if str(task.task_id) != stored_task_id or task.lifecycle is not TaskLifecycle.QUEUED:
                raise TaskQueueHeadAuthorityError(stored_task_id)
            return task

        if preferred_task() is not None:
            return self._read(head)
        return self._answered("queued_task", lambda: self._read(head))

    def stage_receipts(self, task_id: UUID) -> tuple[TaskStageReceipt, ...]:
        """Read the task's committed verifier receipts in observation/stage order.

        Args:
            task_id: Task UUID whose durable record is addressed.

        Returns:
            The validated committed stage receipts.
        """
        return self._read(
            lambda connection: tuple(
                TaskStageReceipt.model_validate_json(row[0])
                for row in connection.execute(
                    """
                    SELECT receipt_json FROM workspace_task_stage_receipt
                    WHERE task_id = ? ORDER BY observed_us, stage_id
                    """,
                    [str(task_id)],
                ).fetchall()
            )
        )

    def safe_projection(self, task_id: UUID) -> TaskSafeProjection:
        # Hold one read/version boundary across the existing readers and its
        # derived publication. No partial combination of different Task versions.
        """Resolve one task-safe projection under a single read/version boundary.

        Args:
            task_id: Task UUID whose durable record is addressed.

        Returns:
            The projection bound to the same task-record version as its read inputs.
        """
        return self._answered(("projection", task_id), lambda: self._safe_projection(task_id))

    def board(self, task_id: UUID) -> TaskBoardSnapshot:
        """Read the task board and projection as one state of the log.

        The record, work board, latest execution and projection of one Task in one read scope,
        so a reader that explains the Task never combines parts of different versions.
        """

        with self.read_scope():
            task = self.task(task_id)
            work_items = self.work_items(task_id)
            execution = (
                self.execution(task.latest_execution_id)
                if task.latest_execution_id is not None
                else None
            )
            return TaskBoardSnapshot(
                task=task,
                work_items=work_items,
                execution=execution,
                projection=self._safe_projection(task_id),
            )

    def _safe_projection(self, task_id: UUID) -> TaskSafeProjection:
        with self.read_scope():
            projection, published_json = self._project(task_id)
        # A status read publishes its projection only when the projection
        # moved; an unchanged one (a poll between heartbeats) writes nothing.
        if published_json != self._json(projection):
            self._publish_projection(projection)
        return projection

    def _project(self, task_id: UUID) -> tuple[TaskSafeProjection, str | None]:
        task = self.task(task_id)
        work_items = self.work_items(task_id)
        execution = self.execution(task.latest_execution_id) if task.latest_execution_id else None
        queued = self.queued_task()
        current_stage = task.active_work_item_id
        if current_stage is None and task.lifecycle not in _TERMINAL_TASKS:
            current_stage = next(
                (
                    item.stage_id
                    for item in work_items
                    if item.lifecycle is WorkItemLifecycle.PENDING
                ),
                None,
            )
        receipts = self.stage_receipts(task_id)
        artifact_refs = tuple(
            dict.fromkeys(item.reference for receipt in receipts for item in receipt.evidence)
        )[:16]
        last_activity = max(
            task.updated_at,
            execution.last_heartbeat_at if execution else task.updated_at,
        )
        identity = {
            "task_id": task.task_id,
            "task_kind": task.task_kind,
            "goal_summary": task.goal.summary,
            "lifecycle": task.lifecycle,
            "current_stage": current_stage,
            "verified_stage_count": sum(
                item.lifecycle is WorkItemLifecycle.VERIFIED for item in work_items
            ),
            "total_stage_count": len(work_items),
            "running_since": task.started_at,
            "last_activity_at": last_activity,
            "cancel_available": _cancel_available(task),
            "cancel_pending": task.lifecycle is TaskLifecycle.CANCEL_REQUESTED,
            "queued_next_task_id": (
                queued.task_id if queued is not None and queued.task_id != task.task_id else None
            ),
            "latest_failure_code": task.failure_code,
            "artifact_refs": artifact_refs,
            "task_record_hash": task.record_hash,
        }
        projection = TaskSafeProjection.from_identity(**identity)
        published = self._read(
            lambda connection: connection.execute(
                "SELECT projection_json FROM workspace_task_projection WHERE task_id = ?",
                [str(task_id)],
            ).fetchone()
        )
        return projection, (str(published[0]) if published is not None else None)

    def _publish_projection(self, projection: TaskSafeProjection) -> None:
        def operation(connection):
            existing = connection.execute(
                "SELECT task_record_hash FROM workspace_task_projection WHERE task_id = ?",
                [str(projection.task_id)],
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO workspace_task_projection VALUES (?, ?, ?, ?)",
                    [
                        str(projection.task_id),
                        projection.task_record_hash,
                        self._json(projection),
                        self._us(projection.last_activity_at),
                    ],
                )
            else:
                connection.execute(
                    """
                    UPDATE workspace_task_projection
                    SET task_record_hash = ?, projection_json = ?, updated_us = ?
                    WHERE task_id = ?
                    """,
                    [
                        projection.task_record_hash,
                        self._json(projection),
                        self._us(projection.last_activity_at),
                        str(projection.task_id),
                    ],
                )

        self._write(operation, control_only=True)

    def latest_safe_projections(self) -> tuple[TaskSafeProjection, ...]:
        """Read persisted task-safe projections ordered by newest update and task identifier.

        Returns:
            The stored validated projection sequence.
        """
        return self._read(
            lambda connection: tuple(
                TaskSafeProjection.model_validate_json(row[0])
                for row in connection.execute(
                    """
                    SELECT p.projection_json FROM workspace_task_projection p
                    JOIN workspace_task t ON t.task_id = p.task_id WHERE t.workspace_id = ?
                    ORDER BY p.updated_us DESC, p.task_id
                    """,
                    [self.workspace_id],
                ).fetchall()
            )
        )

    def projection_collection(self, task_ids: Iterable[UUID]) -> TaskProjectionBatch:
        """Read a bounded projection page, rebuilding one damaged derived row at a time.

        The Task Control query itself remains one global read: a store/open/query failure still
        refuses the collection. JSON in the derived cache is parsed per row so one damaged cache
        entry cannot hide its readable peers. A bad or missing cache entry is reconstructed only
        from that Task's canonical records; if those records cannot support a complete projection,
        the batch names that Task for a caller-owned refusal rather than inventing state.

        Returns:
            The validated projections in stored order and IDs that could not be rebuilt.
        """
        wanted = sorted({str(task_id) for task_id in task_ids})
        if not wanted:
            return TaskProjectionBatch((), ())
        placeholders = ",".join("?" for _ in wanted)
        rows = self._read(
            lambda connection: connection.execute(
                f"""
                SELECT task_id, projection_json FROM workspace_task_projection
                WHERE task_id IN ({placeholders})
                ORDER BY updated_us DESC, task_id
                """,
                wanted,
            ).fetchall()
        )
        projections: list[TaskSafeProjection] = []
        refused_task_ids: list[UUID] = []
        found_task_ids: set[UUID] = set()
        for stored_task_id, projection_json in rows:
            task_id = UUID(str(stored_task_id))
            found_task_ids.add(task_id)
            try:
                projection = TaskSafeProjection.model_validate_json(projection_json)
                if projection.task_id != task_id:
                    raise ValueError("task control projection row identity differs")
            except (TypeError, ValueError):
                try:
                    projection = self.safe_projection(task_id)
                except TaskQueueHeadAuthorityError:
                    raise
                except (KeyError, ValueError):
                    refused_task_ids.append(task_id)
                    continue
            projections.append(projection)
        for task_id_text in wanted:
            task_id = UUID(task_id_text)
            if task_id in found_task_ids:
                continue
            try:
                projection = self.safe_projection(task_id)
            except TaskQueueHeadAuthorityError:
                raise
            except (KeyError, ValueError):
                refused_task_ids.append(task_id)
                continue
            projections.append(projection)
        return TaskProjectionBatch(tuple(projections), tuple(refused_task_ids))

    def stale_active_tasks(
        self, *, observed_at: datetime, threshold: timedelta = timedelta(seconds=30)
    ) -> tuple[TaskRecord, ...]:
        """Find active tasks whose recorded business heartbeat exceeds the age threshold.

        Args:
            observed_at: Aware operational clock supplied by the caller.
            threshold: Positive business-heartbeat age threshold.

        Returns:
            Active task records with a latest execution older than the threshold.

        Raises:
            ValueError: The age threshold is not positive.
        """
        if threshold <= timedelta():
            raise ValueError("stale task threshold must be positive")
        values: list[TaskRecord] = []
        for task in self.tasks():
            if task.lifecycle not in _ACTIVE_TASKS or task.latest_execution_id is None:
                continue
            execution = self.execution(task.latest_execution_id)
            if observed_at - execution.last_heartbeat_at > threshold:
                values.append(task)
        return tuple(values)


__all__ = [
    "TASK_CONTROL_DATABASE_FILENAME",
    "DuckDbTaskControlRegistry",
    "TaskAdmission",
    "TaskBoardSnapshot",
    "TaskControlDatabaseAuthorityError",
    "TaskQueueFull",
    "TaskQueueHeadAuthorityError",
    "TaskRecoveryLink",
    "TaskTransitionRejected",
    "TaskVersionStale",
    "resolve_task_control_database",
]
