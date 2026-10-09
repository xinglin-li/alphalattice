"""One background dispatcher for every local application command.

`PortfolioResearchApplication.run()` blocks through a whole Task. That is right
for a script and wrong for a service: a browser asking for a run has to be told
*which task it now owns* long before the book has been walked, and it has to be
told that without a second lifecycle appearing behind the answer.

So this owns exactly one thing -- when the work happens -- and delegates
everything else:

- **Task Control still admits.** Admission runs on the calling thread, because
  the caller has to be handed a real task id and because the workspace's
  one-active/one-queued rule is enforced there. A full queue refuses here for
  the same reason it refuses anywhere: it is the registry's answer, surfaced,
  not a capacity this dispatcher invented.
- **Task Control still executes.** The worker calls the command's own
  `execute`, which builds the same `TaskControlRunner` over the same checkpoint
  namespace and the same workspace lease.
- **Task Control still reports.** Status is `TaskSafeProjection`, read from the
  registry. Nothing here caches a lifecycle, and nothing here can report one the
  registry would not.

One worker thread, not a pool. The workspace admits one active task; a second
worker could only ever wait, and waiting threads that look like parallelism are
how a capacity rule stops being true. `close()` joins it, so no worker outlives
the service that started it.

Generic on purpose. A command is a name, an admission and an execution; nothing
below knows what a Portfolio is, and Gate 9C2 registers its own commands without
a branch here.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID

from alphalattice.control.task_control.contracts import (
    TaskLifecycle,
    TaskRecord,
    TaskSafeProjection,
)

CommandDisposition = Literal[
    "ADMITTED",
    "REFUSED_QUEUE_FULL",
    "REFUSED_INVALID_COMMAND",
]
"""What happened to a submission, before any numerical work.

`REFUSED_QUEUE_FULL` is the workspace's real answer and is reported as a
disposition rather than an exception, because a UI has to render it as a state
and a retry rather than as a fault.
"""


@dataclass(frozen=True, slots=True)
class CommandAdmission:
    """What an admission already knows, without a second read.

    The registry hands back the record it just wrote, so the task id and its
    lifecycle are in hand. Reading a full `TaskSafeProjection` here instead would
    open six more gated database connections while a worker is holding the write
    gate -- measured at roughly 190 ms of the 250 ms budget, for facts the caller
    is about to poll for anyway.
    """

    task_id: UUID
    lifecycle: str


class LocalApplicationCommand(Protocol):
    """One typed unit of local work: a name, an admission, an execution."""

    @property
    def command_kind(self) -> str:
        """Stable identity for the kind of work, for status and telemetry-free logs."""
        ...

    def admit(self) -> CommandAdmission:
        """Write one queued task and return what it wrote. No numerical work."""
        ...

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Drive that task through Task Control until it stops. The work.

        `expected_task_hash` is the record version a person or Agent confirmed a
        resume against; the command carries it to Task Control's own start
        boundary, which refuses a Task that has moved. `None` is the versionless
        path (a service restart, a caller without a confirmation).
        """
        ...


class TaskStatusPort(Protocol):
    """The registry surface a dispatcher is allowed to touch: read and cancel."""

    def safe_projection(self, task_id: UUID) -> TaskSafeProjection:
        """Read the safe projection for one Task."""
        ...

    def latest_safe_projections(self) -> tuple[TaskSafeProjection, ...]:
        """Read the registry's latest safe Task projections."""
        ...

    def tasks(self) -> tuple[TaskRecord, ...]:
        """Read Task records needed for cancellation decisions, in admission order."""
        ...

    def running_place_holder(self) -> TaskRecord | None:
        """Read the Task holding the workspace's one running place, if one does."""
        ...

    def request_cancel(
        self, *, task_id: UUID, expected_task_hash: str, observed_at: datetime
    ) -> object:
        """Request cancellation against the caller's confirmed Task version."""
        ...

    def finalize_cancel_after_writer_stopped(
        self, *, task_id: UUID, expected_task_hash: str, observed_at: datetime
    ) -> object:
        """Finalize cancellation after the owned writer has stopped."""
        ...


class TaskVersionMovedError(Exception):
    """Refuse a cancel aimed at a Task version that has already moved.

    The caller carried that version explicitly; nothing was applied.
    """


class TaskQueueFullError(Exception):
    """Raised by an admission when the workspace already holds its capacity.

    Declared here so the dispatcher can name the condition without importing the
    registry's own exception hierarchy, and matched structurally: any admission
    port may raise its own type as long as the dispatcher is told which one.
    """


@dataclass(frozen=True, slots=True)
class CommandSubmission:
    """What a caller gets back before the work starts.

    `task_id` is `None` exactly when the disposition is a refusal. `lifecycle` is
    the one the registry wrote, carried out of the admission itself -- so a
    caller never sees a lifecycle this dispatcher made up, and never waits on a
    projection read to learn it.
    """

    command_kind: str
    disposition: CommandDisposition
    submitted_at: datetime
    task_id: UUID | None = None
    lifecycle: str | None = None
    refusal_detail: str | None = None

    @property
    def admitted(self) -> bool:
        """Whether Task Control admitted the submitted command."""
        return self.disposition == "ADMITTED"


_SETTLED_LIFECYCLES = frozenset(
    {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
)
"""Lifecycles Task Control treats as final for the Task itself."""


CANCEL_REAIM_LIMIT = 8
"""How many times a versionless cancel re-reads a moving Task's version and
aims again before it keeps `False`: a running coverage Task moves its
version at every stage boundary, several times a second."""


@dataclass
class LocalBackgroundDispatcher:
    """Admit on the caller's thread; execute on one owned worker.

    Constructed with the queue-full exception type its admission port raises, so
    the capacity answer stays the registry's and this class stays generic.
    """

    status_port: TaskStatusPort
    queue_full_error: type[BaseException] = TaskQueueFullError
    version_moved_error: type[BaseException] = TaskVersionMovedError
    version_stale_error: type[BaseException] = TaskVersionMovedError
    """The registry's own stale-version refusal, matched structurally like the others: a
    caller that confirmed a version learns it moved; a versionless caller keeps the old
    `False`."""
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    on_idle: Callable[[], None] | None = None
    on_command_returned: Callable[[str, UUID, str | None], None] | None = None
    """Told `(command_kind, task_id, worker_failure)` once a command has returned.

    An observer, not a participant: it runs after the running set is updated,
    so a status read inside it already sees the settled lifecycle, and it can
    neither change the Task's outcome nor re-run the command. Its own failure is
    kept on this dispatcher (`observer_failures`), never on the Task, because an
    observer that could not write is not a worker that failed.
    """

    _work: queue.Queue[tuple[LocalApplicationCommand, UUID, str | None] | None] = field(
        default_factory=queue.Queue, init=False, repr=False
    )
    _worker: threading.Thread | None = field(default=None, init=False, repr=False)
    _failures: dict[UUID, str] = field(default_factory=dict, init=False, repr=False)
    _running: set[UUID] = field(default_factory=set, init=False, repr=False)
    """Work this dispatcher queued and whose command has not yet returned."""
    _waiting: dict[UUID, LocalApplicationCommand] = field(
        default_factory=dict, init=False, repr=False
    )
    """Commands that returned before their Task's turn came: another Task held the running
    place (a deferral, which no command drives while it waits, among them) or stood ahead of
    it in the queue. Each is driven again once the place is free, in admission order."""
    _owned_tasks: set[UUID] = field(default_factory=set, init=False, repr=False)
    """Submitted here or restored with an installed command under the Host lease."""
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    admissions: int = field(default=0, init=False)
    executions: int = field(default=0, init=False)
    observer_failures: int = field(default=0, init=False)
    last_observer_failure: str | None = field(default=None, init=False)
    """Exception class of the last hook failure; never its message."""

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Begin the worker. Idempotent, so a service may call it on every boot."""
        with self._lock:
            if self._closed:
                raise RuntimeError("local_application.dispatcher_closed")
            if self._worker is not None:
                return
            worker = threading.Thread(
                target=self._drain, name="local-application-dispatcher", daemon=False
            )
            self._worker = worker
        worker.start()

    def close(self, *, timeout: float | None = None) -> bool:
        """Stop accepting work and join the worker. True when the worker ended.

        Unbounded by default, and that is the safe default rather than a
        convenient one: a task that is mid-stage owns a workspace lease and a
        checkpoint, and a `close` that returned while it was still running would
        hand the caller a service it could believe was stopped.

        A caller that must bound the wait passes a timeout and reads the answer.
        `False` means a writer is still live, and the only correct response is to
        keep holding whatever that writer needs -- not to carry on with cleanup.
        Calling again resumes the join rather than reporting success.
        """
        with self._lock:
            if self._closed and self._worker is None:
                return True
            self._closed = True
            worker = self._worker
        if worker is None:
            return True
        self._work.put(None)
        worker.join(timeout=timeout)
        if worker.is_alive():
            return False
        with self._lock:
            self._worker = None
        return True

    @property
    def worker_alive(self) -> bool:
        """Whether the one owned worker is still running. Never inferred."""
        worker = self._worker
        return worker is not None and worker.is_alive()

    def __enter__(self) -> LocalBackgroundDispatcher:
        """Start the owned worker when entering the dispatch scope."""
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Join the owned worker when leaving the dispatch scope."""
        self.close()

    # -------------------------------------------------------------- submit

    def submit(self, command: LocalApplicationCommand) -> CommandSubmission:
        """Admit one command and return the task it owns. Never waits for the work.

        The admission itself is synchronous and that is deliberate: the task id,
        the queue position and the capacity refusal are all facts the caller
        needs *now*, and all three are Task Control's to state.
        """
        submitted_at = self.clock()
        with self._lock:
            if self._closed:
                raise RuntimeError("local_application.dispatcher_closed")
        try:
            admission = command.admit()
        except self.queue_full_error as error:
            return CommandSubmission(
                command_kind=command.command_kind,
                disposition="REFUSED_QUEUE_FULL",
                submitted_at=submitted_at,
                refusal_detail=str(error)[:200],
            )
        except (ValueError, KeyError) as error:
            return CommandSubmission(
                command_kind=command.command_kind,
                disposition="REFUSED_INVALID_COMMAND",
                submitted_at=submitted_at,
                refusal_detail=str(error)[:200],
            )
        self.admissions += 1
        # A Task found stopped or cancelled has nothing for its command to run: it is answered as
        # it stands, never read as running while that command waits its turn.
        stopped = admission.lifecycle in {
            TaskLifecycle.BLOCKED.value,
            TaskLifecycle.CANCELLED.value,
        }
        with self._lock:
            self._owned_tasks.add(admission.task_id)
            if not stopped:
                self._running.add(admission.task_id)
        self._work.put((command, admission.task_id, None))
        self.start()
        return CommandSubmission(
            command_kind=command.command_kind,
            disposition="ADMITTED",
            submitted_at=submitted_at,
            task_id=admission.task_id,
            lifecycle=admission.lifecycle,
        )

    def resume(
        self,
        commands: dict[str, LocalApplicationCommand],
        *,
        only_task_id: UUID | None = None,
        expected_task_hash: str | None = None,
        refusal: Callable[[TaskRecord], str | None] | None = None,
    ) -> tuple[UUID, ...]:
        """Re-enqueue work the workspace still owes after a restart.

        A service that comes back up inherits whatever the last one left
        `RECOVERY_REQUIRED` or still `QUEUED`. Re-enqueuing is not re-admitting: the task, its
        identity and its verified stage prefix are already durable, and the
        command supplied here only has to know how to drive it.

        `expected_task_hash` (with `only_task_id`) is the version a caller confirmed
        the resume against. It rides the queue item to the command, which hands it
        to Task Control's start boundary; a Task that moved meanwhile is refused
        there, inside the registry's transaction, not by a re-read here.

        `refusal` names a Task that cannot resume under the code installed now; it
        is not enqueued, and its code is this Task's failure, as for a kind this
        service does not run.
        """
        resumed: list[UUID] = []
        # A Task that never started has no stored status projection. Restore
        # from the registry's records, with the active recovery before the queue.
        records = sorted(
            self.status_port.tasks(),
            key=lambda task: task.lifecycle is not TaskLifecycle.RECOVERY_REQUIRED,
        )
        for task in records:
            if only_task_id is not None and task.task_id != only_task_id:
                continue
            if task.task_kind in commands:
                with self._lock:
                    self._owned_tasks.add(task.task_id)
            if task.lifecycle not in {TaskLifecycle.RECOVERY_REQUIRED, TaskLifecycle.QUEUED}:
                continue
            command = commands.get(task.task_kind)
            if command is None:
                # A task kind this service does not run. Leaving it alone is the
                # honest answer; another adapter owns it.
                with self._lock:
                    self._failures[task.task_id] = (
                        "local_application.recovery_command_not_installed:" + task.task_kind
                    )
                continue
            with self._lock:
                if task.task_id in self._running:
                    continue
            refused = None if refusal is None else refusal(task)
            if refused is not None:
                with self._lock:
                    self._failures[task.task_id] = refused
                continue
            with self._lock:
                if task.task_id in self._running:
                    continue
                self._running.add(task.task_id)
            confirmed = expected_task_hash if task.task_id == only_task_id else None
            self._work.put((command, task.task_id, confirmed))
            resumed.append(task.task_id)
        if resumed:
            self.start()
        return tuple(resumed)

    # -------------------------------------------------------------- status

    def status(self, task_id: UUID) -> TaskSafeProjection:
        """The registry's projection, held short of terminal until work returns.

        Task Control's lifecycle is about the Task. A reader of this route is
        asking about the operation, and the operation is not over until the
        command that publishes the result has returned. Reporting the first as
        if it were the second is what let a finished run be announced before it
        could be opened.
        """
        projection = self.status_port.safe_projection(task_id)
        return self.masked(projection)

    def masked(self, projection: TaskSafeProjection) -> TaskSafeProjection:
        """Apply the operation-not-over rule to a projection read elsewhere.

        A recovery this dispatcher holds waits its turn, as a queued Task does: it awaits no
        one's decision, so a reader waits on it rather than ending there.
        """
        if not self.command_running(projection.task_id):
            return projection
        if projection.lifecycle is TaskLifecycle.RECOVERY_REQUIRED:
            return projection.model_copy(update={"lifecycle": TaskLifecycle.QUEUED})
        if projection.lifecycle not in _SETTLED_LIFECYCLES:
            return projection
        return projection.model_copy(update={"lifecycle": TaskLifecycle.RUNNING})

    def command_running(self, task_id: UUID) -> bool:
        """Whether this dispatcher's command for the Task has not yet returned.

        A command kept until the running place is free counts as not returned.
        """
        with self._lock:
            return task_id in self._running or task_id in self._waiting

    def active(self) -> tuple[TaskSafeProjection, ...]:
        """Return the registry's latest safe Task projections."""
        return self.status_port.latest_safe_projections()

    def final_lifecycle(self, projection: TaskSafeProjection) -> TaskLifecycle | None:
        """The Task's final lifecycle: one Task Control holds final, once its command returned.

        Args:
            projection: The Task, as Task Control projected it.

        Returns:
            `SUCCEEDED`, `BLOCKED` or `CANCELLED`; None while it runs, waits or may resume.
        """
        settled = self.masked(projection)
        return settled.lifecycle if settled.lifecycle in _SETTLED_LIFECYCLES else None

    def failure(self, task_id: UUID) -> str | None:
        """Why the worker stopped, when the failure was not Task Control's own.

        A stage that blocks is recorded by the registry and reads back through
        the projection. This is for the other case -- the command itself raised
        -- which would otherwise be visible only as a task that stopped moving.
        """
        with self._lock:
            return self._failures.get(task_id)

    def request_cancel(self, task_id: UUID, *, expected_task_hash: str | None = None) -> bool:
        """Request cancellation; finalize only for owned work with no command running.

        Running commands retain safe-checkpoint cancellation. Waiting work has
        no worker to observe that checkpoint, so the exclusive Host proves its
        inactivity under the same lock used to enqueue, then asks Task Control
        to finalize the exact accepted version.

        The version handed to the registry is the caller's confirmed one when it
        carries one, else the record hash read immediately before the request;
        either way the registry's optimistic-concurrency guard decides: a cancel
        aimed at a version that has already moved is refused rather than applied
        to whatever the task became. A confirmed caller is told so
        (`TaskVersionMovedError`). A versionless caller means the Task as it is
        now: a running Task moves its version at every stage boundary, so the
        read and the command race, and the request is aimed again at the
        version it finds, `CANCEL_REAIM_LIMIT` times, before it keeps `False`
        (seen on the book journey: a cancel of a running eight-unit run was
        refused at the transaction and the run completed as if never asked).
        """
        with self._lock:
            attempts = 0
            while True:
                projection = self.status_port.safe_projection(task_id)
                confirmed = (
                    expected_task_hash
                    if expected_task_hash is not None
                    else projection.task_record_hash
                )
                try:
                    self.status_port.request_cancel(
                        task_id=task_id,
                        expected_task_hash=confirmed,
                        observed_at=self.clock(),
                    )
                    break
                except self.version_moved_error as error:
                    stale = isinstance(error, self.version_stale_error)
                    if expected_task_hash is not None:
                        if stale:
                            raise TaskVersionMovedError(
                                "task_control.cancel_version_stale"
                            ) from error
                        return False
                    attempts += 1
                    if not stale or attempts >= CANCEL_REAIM_LIMIT:
                        return False
            finalized = task_id in self._owned_tasks and task_id not in self._running
            if finalized:
                self._finalize_idle_cancel(task_id)
                self._waiting.pop(task_id, None)
        if finalized:
            # A cancelled deferral frees the running place no command held.
            self._drive_waiting()
        return True

    def _finalize_idle_cancel(self, task_id: UUID) -> None:
        """Caller holds the enqueue lock and has proved the owned command stopped."""
        projection = self.status_port.safe_projection(task_id)
        if projection.lifecycle is TaskLifecycle.CANCEL_REQUESTED:
            self.status_port.finalize_cancel_after_writer_stopped(
                task_id=task_id,
                expected_task_hash=projection.task_record_hash,
                observed_at=self.clock(),
            )

    def _waits_its_turn(self, task_id: UUID) -> bool:
        """Whether the Task's command returned before its turn came.

        Its Task is still queued or owed its recovery, and another Task holds the running place
        or, for a queued one, stands ahead of it in the queue. A deferral holds the place while
        no command drives it, so every Task admitted meanwhile met this; its command returned
        and nothing drove it again. Any other start that did not happen stays the command's
        failure.
        """
        try:
            tasks = self.status_port.tasks()
            own = next((task for task in tasks if task.task_id == task_id), None)
            if own is None or own.lifecycle not in {
                TaskLifecycle.QUEUED,
                TaskLifecycle.RECOVERY_REQUIRED,
            }:
                return False
            holder = self.status_port.running_place_holder()
        except Exception:
            return False
        if holder is not None and holder.task_id != task_id:
            return True
        head = next((task for task in tasks if task.lifecycle is TaskLifecycle.QUEUED), None)
        return (
            own.lifecycle is TaskLifecycle.QUEUED and head is not None and head.task_id != task_id
        )

    def _drive_waiting(self) -> None:
        """Drive again the commands whose turn had not come, once the running place is free.

        The recoveries first, then the queue in its admission order, the order Task Control
        starts them in.
        """
        with self._lock:
            if self._closed or not self._waiting:
                return
        try:
            if self.status_port.running_place_holder() is not None:
                return
            tasks = self.status_port.tasks()
        except Exception:
            return
        recoveries = [t.task_id for t in tasks if t.lifecycle is TaskLifecycle.RECOVERY_REQUIRED]
        queue = [t.task_id for t in tasks if t.lifecycle is TaskLifecycle.QUEUED]
        driven = False
        with self._lock:
            if self._closed:
                return
            for task_id in (*recoveries, *queue):
                if task_id in self._running:
                    continue
                command = self._waiting.pop(task_id, None)
                if command is not None:
                    self._running.add(task_id)
                    self._work.put((command, task_id, None))
                    driven = True
                elif task_id in queue:
                    # A queued Task no command here drives stands at the queue's head; the rest
                    # wait behind it, as Task Control starts them. Its recovery hands it back.
                    break
            # A Task no longer queued or owed its recovery has nothing left for its command.
            live = {*recoveries, *queue}
            for task_id in [task_id for task_id in self._waiting if task_id not in live]:
                del self._waiting[task_id]
        if driven:
            self.start()

    # -------------------------------------------------------------- worker

    def _drain(self) -> None:
        while True:
            item = self._work.get()
            try:
                if item is None:
                    return
                command, task_id, expected_task_hash = item
                try:
                    # Versionless work is called the way every command always was; only a
                    # confirmed version asks the command to carry it to Task Control.
                    if expected_task_hash is None:
                        command.execute(task_id)
                    else:
                        command.execute(task_id, expected_task_hash=expected_task_hash)
                    self.executions += 1
                    with self._lock:
                        self._failures.pop(task_id, None)
                except Exception as error:
                    with self._lock:
                        self._failures[task_id] = f"{type(error).__name__}: {error}"[:400]
                finally:
                    waits = self._waits_its_turn(task_id)
                    # Cleared whichever way it went: a caller waiting for a
                    # terminal answer must be released on failure too.
                    with self._lock:
                        self._running.discard(task_id)
                        if waits:
                            # Not a failure: its turn had not come.
                            self._waiting[task_id] = command
                            self._failures.pop(task_id, None)
                        try:
                            self._finalize_idle_cancel(task_id)
                        except Exception as error:
                            self._failures[task_id] = f"{type(error).__name__}: {error}"[:400]
                self._notify_returned(command.command_kind, task_id)
                self._drive_waiting()
            finally:
                self._work.task_done()
                with self._lock:
                    callback = self.on_idle if not self._running and not self._closed else None
                if callback is not None:
                    try:
                        callback()
                    except Exception:
                        # An optional wake observer is not allowed to kill the
                        # Task worker or change a published Task's lifecycle.
                        with self._lock:
                            if item is not None:
                                self._failures[item[1]] = "local_application.idle_observer_failed"

    def _notify_returned(self, command_kind: str, task_id: UUID) -> None:
        observer = self.on_command_returned
        if observer is None:
            return
        try:
            observer(command_kind, task_id, self.failure(task_id))
        except Exception as error:
            # The worker outlives its observer. The failure is visible on this
            # dispatcher and through the activity read, and stays off the Task:
            # its lifecycle and result were already established above.
            with self._lock:
                self.observer_failures += 1
                # The class, never the message: an observer's exception text is
                # not a safe fact and is not served or persisted anywhere.
                self.last_observer_failure = type(error).__name__

    def drain_for_tests(self, *, timeout: float = 120.0) -> None:
        """Block until the queue is empty. Test-only synchronisation.

        Named for what it is. Production callers poll status, because a caller
        that waits for the work has undone the reason this class exists.
        """
        del timeout
        self._work.join()


__all__ = [
    "CommandAdmission",
    "CommandDisposition",
    "CommandSubmission",
    "LocalApplicationCommand",
    "LocalBackgroundDispatcher",
    "TaskQueueFullError",
    "TaskStatusPort",
    "TaskVersionMovedError",
]
