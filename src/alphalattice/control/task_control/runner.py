"""Durable single-writer task runner: one loop over a Task's work board (W10).

The registry's work items are the only state a run keeps: a step claims what is ready,
runs it and records its receipt, and recovery restarts from the verified items, so no
second store (a graph checkpoint) holds a Task's progress.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from threading import Event, Thread
from typing import Final, Protocol
from uuid import UUID, uuid4

from alphalattice.kernel.shared_kernel.spans import SpanLedger, collect

from .child import ChildStartFailed
from .contracts import (
    FAILURE_CODE_MAX_LENGTH,
    StageFailureCause,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskHeartbeatSignal,
    TaskLifecycle,
    TaskRecord,
    TaskSafeProjection,
    TaskStageReceipt,
    WorkItemDefinition,
    WorkItemLifecycle,
)
from .registry import DuckDbTaskControlRegistry, TaskTransitionRejected
from .timing import record_stage_spans

_LOG = logging.getLogger(__name__)

REPEATED_STAGE_RAISES: Final = 3
"""A stage that raises on this many attempts in a row is blocked with its error's type, not
resumed again; an attempt that defers or completes ends the row (V475). One raise is a stop the
next run recovers from; the same stage raising on every attempt is a failure its owner did not
name, and resuming it would loop (found by LS1's acceptance: a Sector reader's database error
resumed nine times)."""


class StageDisposition(StrEnum):
    """Declare domain-stage readiness, deferral, block or checkpoint cancellation."""

    READY = "READY"
    DEFERRED = "DEFERRED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class StageExecutionResult:
    """Return a domain-stage disposition with retained evidence and optional stable cause.

    Attributes:
        disposition: Outcome of domain-stage execution.
        evidence: Evidence references proposed for verification.
        failure_code: Optional stable domain failure cause.
        failure_cause: What the owner saw beside that code, kept with the block (V444).
    """

    disposition: StageDisposition
    evidence: tuple[TaskEvidence, ...] = ()
    failure_code: str | None = None
    failure_cause: StageFailureCause | None = None


class TaskDomainAdapter(Protocol):
    """Define the domain boundary for compatibility, stage execution and verification."""

    task_kind: str

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Resolve installed domain/workflow/framework compatibility for this task.

        Args:
            task: Sealed task record whose domain work is requested.

        Returns:
            The sealed execution compatibility bindings.
        """
        ...

    def execute_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
    ) -> StageExecutionResult:
        """Execute the declared domain stage under its task/execution binding.

        Args:
            task: Sealed task record whose domain work is requested.
            execution: Bound execution record for this run.
            work_item: Sealed definition of the stage to execute or verify.

        Returns:
            The domain disposition and retained evidence to be verified.
        """
        ...

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Verify stage evidence under the declared deterministic domain verifier.

        Args:
            task: Sealed task record whose domain work is requested.
            execution: Bound execution record for this run.
            work_item: Sealed definition of the stage to execute or verify.
            evidence: Exact stage evidence references to retain or verify.

        Returns:
            The evidence references accepted by the domain verifier.
        """
        ...


def _bounded_raise(
    attempt: int,
    call: Callable[[], StageExecutionResult],
    scope: Callable[[], AbstractContextManager[object]] | None = None,
) -> StageExecutionResult:
    """Run one stage, inside `scope` when one is given; on its last allowed attempt, an
    unnamed raise blocks it with its type."""
    try:
        if scope is None:
            return call()
        with scope():
            return call()
    except Exception as error:
        if isinstance(error, ChildStartFailed):
            raise
        if getattr(error, "failure_class", None) or attempt < REPEATED_STAGE_RAISES:
            raise
        _LOG.exception("a stage raised on attempt %s; it is blocked, not resumed", attempt)
        return StageExecutionResult(
            StageDisposition.BLOCKED,
            failure_code=f"TASK_STAGE_RAISED:{type(error).__name__}",
            # What was raised rides beside the code, as an owner's cause does (V444).
            failure_cause=StageFailureCause.from_facts(
                {"exception_type": type(error).__name__, "detail": str(error)}
            ),
        )


def _spanned[T](
    runtime_path: Path,
    task_id: UUID,
    execution_id: UUID,
    stage_id: str,
    phase: str,
    call: Callable[[], T],
) -> T:
    """Run one phase of a stage inside a span ledger and keep what it spent its time on (A4).

    The readout is kept whether the phase returned or raised; keeping it never fails the stage.
    """
    ledger: SpanLedger | None = None
    try:
        with collect() as ledger:
            return call()
    finally:
        if ledger is not None and ledger.readout_value is not None:
            record_stage_spans(
                runtime_path,
                task_id=task_id,
                execution_id=execution_id,
                stage_id=stage_id,
                phase=phase,
                readout=ledger.readout_value,
            )


class _WorkPool:
    """The work items of one execution in flight: at most `width` at once.

    Width 1 runs each item inline on the runner's thread, as the runner always
    has. A wider pool runs each item's `execute_stage` on a worker thread of
    its own and nothing else there: every registry transition -- the claim,
    the receipt, a block -- stays on the runner's thread, one completion per
    step. An adapter declares the width (`concurrent_work_items`) when
    its items are independent and it keeps its own numerics and shared state
    under that concurrency; no other adapter is run concurrently.
    """

    def __init__(self, width: int, ordinals: Mapping[str, int]) -> None:
        self.width = max(1, width)
        self._ordinals = ordinals
        self._executor = (
            ThreadPoolExecutor(self.width, thread_name_prefix="task-work-item")
            if self.width > 1
            else None
        )
        self._running: dict[Future[StageExecutionResult], WorkItemDefinition] = {}
        self._finished: list[tuple[WorkItemDefinition, StageExecutionResult]] = []

    def __len__(self) -> int:
        return len(self._running) + len(self._finished)

    def stage_ids(self) -> set[str]:
        return {item.stage_id for item in self._running.values()} | {
            item.stage_id for item, _result in self._finished
        }

    def has_room(self) -> bool:
        return len(self) < self.width

    def start(
        self, definition: WorkItemDefinition, call: Callable[[], StageExecutionResult]
    ) -> None:
        if self._executor is None:
            self._finished.append((definition, call()))
        else:
            self._running[self._executor.submit(call)] = definition

    def next_done(self) -> tuple[WorkItemDefinition, StageExecutionResult]:
        """The next finished item, the earliest in the plan among those that
        finished together. An item that raised is re-raised only after every
        other running item has finished: no worker outlives its execution."""

        if self._finished:
            return self._finished.pop(0)
        done, _pending = wait(self._running, return_when=FIRST_COMPLETED)
        future = min(done, key=lambda value: self._ordinals[self._running[value].stage_id])
        definition = self._running.pop(future)
        try:
            return definition, future.result()
        except BaseException:
            self.drain()
            raise

    def drain(self) -> list[tuple[WorkItemDefinition, StageExecutionResult]]:
        """Wait for every running item; the results of those that returned, in
        plan order (an item that raised is left to the recovery that follows)."""

        wait(self._running)
        finished = [
            (definition, future.result())
            for future, definition in self._running.items()
            if future.exception() is None
        ]
        self._running.clear()
        finished += self._finished
        self._finished = []
        return sorted(finished, key=lambda value: self._ordinals[value[0].stage_id])

    def close(self) -> None:
        self.drain()
        if self._executor is not None:
            self._executor.shutdown(wait=True)


@dataclass(frozen=True)
class _RunnerContext:
    registry: DuckDbTaskControlRegistry
    adapter: TaskDomainAdapter
    clock: Callable[[], datetime]
    pool: _WorkPool


class _RecoveryEvidenceError(ValueError):
    failure_class = "TASK_RECOVERY_EVIDENCE_INVALID"


class _StoreUnreadable(RuntimeError):
    """One operational heartbeat sidecar exists but cannot be read; its message is the code."""


def heartbeat_store_path(runtime_path: Path) -> Path:
    """The runner's operational heartbeat sidecar, named from its runtime file."""
    return runtime_path.with_name(
        f"{runtime_path.stem}-heartbeats{runtime_path.suffix or '.sqlite'}"
    )


@dataclass(frozen=True)
class TaskHeartbeatReadout:
    """What the runners' sidecars hold under one execution id, and what could not be read.

    `signals` are the rows recorded under that id in every sidecar that could be read:
    the runner that executed it wrote exactly one, and a row elsewhere is not that
    runner's and does not bind to the execution's worker (the reader that explains the
    Task decides that). `unreadable` names, by failure code, each sidecar that exists
    but could not be read -- a filesystem refusal, a store without the heartbeat table,
    a row that is not a signal -- so an absent signal beside one is not a verdict: it
    may be there. Neither is telemetry from the runner; both are only what was read.
    """

    signals: tuple[TaskHeartbeatSignal, ...]
    unreadable: tuple[str, ...]


class TaskHeartbeatReader:
    """Read operational heartbeat sidecars without owning task execution.

    Read-only access to the runners' operational heartbeats, for a reader that
    explains a Task without owning it.

    The runner is the only writer; this opens its sidecars read-only, creates
    nothing, and answers by execution id -- the store's own key. Every applicable
    sidecar is inspected and the answer is a `TaskHeartbeatReadout`: a sidecar that
    cannot be read is reported there by failure code and never hides what another
    sidecar holds, and nothing that goes wrong inside a sidecar leaves this boundary
    as an exception. A missing sidecar or row is simply no signal.
    """

    def __init__(self, *runtime_paths: Path | str) -> None:
        """Resolve distinct read-only heartbeat sidecars from the supplied runtime files.

        Args:
            runtime_paths: Runner runtime files whose heartbeat sidecars are read, deduplicated by
                resolved path.
        """
        self.paths = tuple(
            dict.fromkeys(heartbeat_store_path(Path(value).resolve()) for value in runtime_paths)
        )

    def read(self, execution_id: UUID) -> TaskHeartbeatReadout:
        """Inspect all resolved sidecars for this execution and report unreadable stores separately.

        Args:
            execution_id: Execution UUID whose record or event is addressed.

        Returns:
            Recorded signals and stable unreadability causes; missing sidecars/rows yield no signal.
        """
        signals: list[TaskHeartbeatSignal] = []
        unreadable: list[str] = []
        for path in self.paths:
            try:
                signal = self._read_store(path, execution_id)
            except _StoreUnreadable as error:
                unreadable.append(str(error))
                continue
            if signal is not None:
                signals.append(signal)
        return TaskHeartbeatReadout(signals=tuple(signals), unreadable=tuple(unreadable))

    @staticmethod
    def _read_store(path: Path, execution_id: UUID) -> TaskHeartbeatSignal | None:
        # Filesystem access (the existence check included) and the read-only open are one
        # boundary: whatever the operating system or SQLite refuses is "unreadable", named,
        # and nothing else is caught here.
        try:
            if not path.is_file():
                return None
            with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5.0) as conn:
                row = conn.execute(
                    "SELECT signal_json FROM workspace_task_heartbeat WHERE execution_id = ?",
                    (str(execution_id),),
                ).fetchone()
        except (OSError, sqlite3.Error) as error:
            raise _StoreUnreadable("task_control.heartbeat_store_unreadable") from error
        if row is None or row[0] is None:
            return None
        try:
            return TaskHeartbeatSignal.model_validate_json(row[0])
        except ValueError as error:
            raise _StoreUnreadable("task_control.heartbeat_signal_invalid") from error


class TaskControlRunner:
    """Execute the next admitted task; DuckDB remains the business authority."""

    def __init__(
        self,
        *,
        registry: DuckDbTaskControlRegistry,
        adapters: Mapping[str, TaskDomainAdapter],
        runtime_path: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        heartbeat_seconds: float = 5.0,
        projection_sink: Callable[[TaskSafeProjection], None] | None = None,
        heartbeat_sink: Callable[[TaskHeartbeatSignal], None] | None = None,
        width: int | None = None,
        stage_gate: Callable[[TaskRecord], str | None] | None = None,
        stage_scope: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> None:
        """Bind the installed adapters, business authority and operational heartbeat owner.

        Args:
            registry: Durable Task Control business authority.
            adapters: Installed task-kind to domain-adapter mapping.
            runtime_path: Runtime filename used to name this runner's heartbeat sidecar.
            clock: Product clock used for execution and heartbeat observations.
            heartbeat_seconds: Positive operational heartbeat interval.
            projection_sink: Optional callback receiving the task-safe business projection.
            heartbeat_sink: Optional callback receiving operational heartbeat signals.
            width: The cores a Task may use at once, its workspace's CPU budget as found when
                it starts (an execution parameter, PA2); the host's processors when None.
            stage_gate: Asked before each stage starts; a refusal code it returns blocks the
                stage, recoverably, before any of its work (an execution check, never a
                result: the Host's memory check).
            stage_scope: Entered around each stage's work when the stage runs alone on the
                runner's thread (the Host's storage measurement); concurrent work items
                run without it, each exactly as before.

        Raises:
            ValueError: The heartbeat interval is not positive.
        """
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.registry = registry
        self.adapters = dict(adapters)
        self.clock = clock
        self.heartbeat_seconds = heartbeat_seconds
        self.projection_sink = projection_sink
        self.heartbeat_sink = heartbeat_sink
        self.width = width
        self.stage_gate = stage_gate
        self.stage_scope = stage_scope
        self.worker_instance_id = uuid4()
        # The runtime file's name names the heartbeat sidecar beside it; its folder keeps the
        # stages' spans.
        self._runtime_path = Path(runtime_path)
        self._heartbeat_store = _SqliteHeartbeatStore(heartbeat_store_path(self._runtime_path))

    def close(self) -> None:
        """Release what the runner holds between executions: nothing since W10's loop."""

    def run_next(
        self, *, expected_task_id: UUID | None = None, expected_task_hash: str | None = None
    ) -> TaskRecord | None:
        """Claim the confirmed queue head and invoke its installed domain adapter.

        Args:
            expected_task_id: Optional queue-head identity the caller confirmed.
            expected_task_hash: Optional exact task version the caller confirmed.

        Returns:
            The resulting task record, or None when no queued work or running place is available.

        Raises:
            ValueError: The confirmed queue-head identifier differs.
        """
        queued = self.registry.queued_task()
        if queued is None:
            return None
        if expected_task_id is not None and queued.task_id != expected_task_id:
            raise ValueError("task_control.queued_task_identity_mismatch")
        adapter = self._adapter(queued)
        started = self.registry.start_next(
            compatibility=adapter.compatibility(queued),
            worker_instance_id=self.worker_instance_id,
            observed_at=self._now(),
            expected_task_id=queued.task_id,
            expected_task_hash=expected_task_hash,
        )
        if started is None:
            return None
        task, execution = started
        return self._invoke(task, execution, adapter)

    def recover(self, task_id: UUID, *, expected_task_hash: str | None = None) -> TaskRecord:
        """Resume from the verified prefix; a confirmed version is enforced by the registry."""
        task = self.registry.task(task_id)
        adapter = self._adapter(task)
        task, execution = self.registry.restart_recovery(
            task_id=task_id,
            compatibility=adapter.compatibility(task),
            worker_instance_id=self.worker_instance_id,
            observed_at=self._now(),
            expected_task_hash=expected_task_hash,
        )
        return self._invoke(task, execution, adapter, verify_prefix=True)

    def _invoke(
        self,
        task: TaskRecord,
        execution: TaskExecution,
        adapter: TaskDomainAdapter,
        *,
        verify_prefix: bool = False,
    ) -> TaskRecord:
        stop = Event()
        heartbeat = Thread(
            target=self._heartbeat,
            args=(execution, stop),
            name=f"task-heartbeat-{task.task_id}",
            daemon=True,
        )
        heartbeat.start()
        # One task's execution is one bounded unit of registry work: every
        # stage's begin, receipt, projection and the adapter's own reads share
        # the instance opened here rather than each reopening the file.
        try:
            with self.registry.retain(read_only=False):
                return self._execute(task, execution, adapter, verify_prefix=verify_prefix)
        finally:
            stop.set()
            heartbeat.join(timeout=max(self.heartbeat_seconds * 2, 1.0))

    def _execute(
        self,
        task: TaskRecord,
        execution: TaskExecution,
        adapter: TaskDomainAdapter,
        *,
        verify_prefix: bool,
    ) -> TaskRecord:
        try:
            if verify_prefix:
                definitions = {item.stage_id: item for item in task.plan.work_items}
                for item in self.registry.work_items(task.task_id):
                    if item.lifecycle is not WorkItemLifecycle.VERIFIED:
                        continue
                    try:
                        verified = _spanned(
                            self._runtime_path,
                            task.task_id,
                            execution.execution_id,
                            item.stage_id,
                            "recovery_verify",
                            partial(
                                adapter.verify_stage,
                                task=task,
                                execution=execution,
                                work_item=definitions[item.stage_id],
                                evidence=item.evidence,
                            ),
                        )
                        if verified != item.evidence:
                            raise ValueError("verifier changed evidence identity")
                    except ChildStartFailed:
                        raise
                    except Exception as error:
                        raise _RecoveryEvidenceError(
                            f"recovery_prefix_invalid:{item.stage_id}"
                        ) from error
                self._keep_stored_evidence(task, execution, adapter, definitions)
            pool = _WorkPool(
                _declared_width(adapter, task, self.width),
                {item.stage_id: ordinal for ordinal, item in enumerate(task.plan.work_items)},
            )
            context = _RunnerContext(
                registry=self.registry, adapter=adapter, clock=self.clock, pool=pool
            )
            try:
                while self._step(task.task_id, execution.execution_id, context):
                    pass
            except BaseException:
                pool.drain()
                raise
            finally:
                pool.close()
            return self.registry.task(task.task_id)
        except ChildStartFailed as error:
            _LOG.exception("the OS refused a Task's child start; recovery is required")
            task = self.registry.mark_recovery_required(
                task_id=task.task_id, failure_code=str(error), observed_at=self._now()
            )
            self._publish_projection(task.task_id)
            return task
        except BaseException as error:
            declared_class = getattr(error, "failure_class", None)
            failure_code = (
                f"{declared_class}:{error}"[:120]
                if isinstance(declared_class, str) and declared_class
                else "TASK_EXECUTION_INTERRUPTED"
            )
            self.registry.mark_recovery_required(
                task_id=task.task_id,
                failure_code=failure_code,
                observed_at=self._now(),
            )
            self._publish_projection(task.task_id)
            raise

    def _heartbeat(self, execution: TaskExecution, stop: Event) -> None:
        sequence = 1
        self._publish_heartbeat(execution, sequence)
        while not stop.wait(self.heartbeat_seconds):
            sequence += 1
            self._publish_heartbeat(execution, sequence)

    def _publish_heartbeat(self, execution: TaskExecution, sequence: int) -> None:
        signal = TaskHeartbeatSignal.from_identity(
            task_id=execution.task_id,
            execution_id=execution.execution_id,
            worker_instance_id=self.worker_instance_id,
            sequence=sequence,
            observed_at=self._now(),
        )
        self._heartbeat_store.touch(signal)
        if self.heartbeat_sink is not None:
            with suppress(Exception):
                # Observation failure cannot become Task lifecycle authority or
                # stop the independent liveness writer.
                self.heartbeat_sink(signal)

    def heartbeat_signal(self, execution_id: UUID) -> TaskHeartbeatSignal | None:
        """Read the latest operational signal without changing Task lifecycle."""
        return self._heartbeat_store.read(execution_id)

    def stale_active_tasks(self, *, threshold_seconds: float = 30.0) -> tuple[TaskRecord, ...]:
        """Compare active execution age against the latest business or operational heartbeat.

        Args:
            threshold_seconds: Positive age threshold for business/operational heartbeat comparison.

        Returns:
            Active tasks whose latest available heartbeat exceeds the threshold.

        Raises:
            ValueError: The age threshold is not positive.
        """
        if threshold_seconds <= 0:
            raise ValueError("stale task threshold must be positive")
        now = self._now()
        stale: list[TaskRecord] = []
        for task in self.registry.tasks():
            if (
                task.lifecycle
                not in {
                    TaskLifecycle.RUNNING,
                    TaskLifecycle.DEFERRED,
                    TaskLifecycle.REVIEW_PENDING,
                    TaskLifecycle.CANCEL_REQUESTED,
                    TaskLifecycle.RECOVERY_REQUIRED,
                }
                or task.latest_execution_id is None
            ):
                continue
            execution = self.registry.execution(task.latest_execution_id)
            signal = self._heartbeat_store.read(execution.execution_id)
            operational = signal.observed_at if signal is not None else None
            last_seen = max(
                execution.last_heartbeat_at,
                operational or execution.last_heartbeat_at,
            )
            if (now - last_seen).total_seconds() > threshold_seconds:
                stale.append(task)
        return tuple(stale)

    def _adapter(self, task: TaskRecord) -> TaskDomainAdapter:
        try:
            return self.adapters[task.task_kind]
        except KeyError as exc:
            raise ValueError(f"no task adapter owns {task.task_kind!r}") from exc

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("task runner clock must return a timezone-aware value")
        return value.astimezone(UTC)

    def _step(self, task_id: UUID, execution_id: UUID, context: _RunnerContext) -> bool:
        """One step of the work board: claim what is ready, record what finished; go on or not."""
        registry = context.registry
        pool = context.pool
        task = registry.task(task_id)
        if task.lifecycle is TaskLifecycle.CANCEL_REQUESTED:
            pool.drain()
            registry.apply_cancel(task_id=task_id, observed_at=self._now())
            self._publish_projection(task_id)
            return False
        if task.lifecycle is not TaskLifecycle.RUNNING:
            pool.drain()
            return False
        work_states = registry.work_items(task_id)
        running = pool.stage_ids()
        if any(
            item.lifecycle not in {WorkItemLifecycle.VERIFIED, WorkItemLifecycle.PENDING}
            and item.stage_id not in running
            for item in work_states
        ):
            raise RuntimeError("task work-board contains an unreconciled active item")
        verified = {
            item.stage_id for item in work_states if item.lifecycle is WorkItemLifecycle.VERIFIED
        }
        definitions = {item.stage_id: item for item in task.plan.work_items}
        for item in work_states:
            if not pool.has_room():
                break
            definition = definitions[item.stage_id]
            if (
                item.lifecycle is not WorkItemLifecycle.PENDING
                or item.stage_id in running
                or any(value not in verified for value in definition.dependency_ids)
            ):
                continue
            try:
                registry.begin_work_item(
                    task_id=task_id,
                    execution_id=execution_id,
                    stage_id=definition.stage_id,
                    observed_at=self._now(),
                )
            except TaskTransitionRejected:
                # The Task moved between the read above and this claim -- a
                # cancel requested in that window is the one case a running
                # worker can meet here. The cancel is applied at this
                # boundary, as it would have been one read earlier; anything
                # else is the interruption it always was (seen on the book
                # journey: a cancel twelve milliseconds after the start left
                # the run RECOVERY_REQUIRED and the queue behind it waiting).
                moved = registry.task(task_id)
                if moved.lifecycle is TaskLifecycle.CANCEL_REQUESTED:
                    pool.drain()
                    registry.apply_cancel(task_id=task_id, observed_at=self._now())
                    self._publish_projection(task_id)
                    return False
                raise
            self._publish_projection(task_id)
            # This attempt, after the stage's interrupted ones in a row (V475).
            attempt = registry.consecutive_interruptions(task_id, definition.stage_id) + 1
            refused = None if self.stage_gate is None else self.stage_gate(task)
            pool.start(
                definition,
                partial(
                    _spanned,
                    self._runtime_path,
                    task_id,
                    execution_id,
                    definition.stage_id,
                    "execute",
                    partial(
                        _bounded_raise,
                        attempt,
                        partial(
                            context.adapter.execute_stage,
                            task=task,
                            execution=registry.execution(execution_id),
                            work_item=definition,
                        )
                        if refused is None
                        else partial(
                            StageExecutionResult,
                            StageDisposition.BLOCKED,
                            failure_code=refused,
                        ),
                        None if context.pool.width > 1 else self.stage_scope,
                    ),
                ),
            )
        if not pool:
            if len(verified) == len(work_states):
                return False
            raise RuntimeError("task work-board holds no runnable item")
        definition, result = pool.next_done()
        execution = registry.execution(execution_id)
        registry.heartbeat(
            execution_id=execution_id,
            worker_instance_id=execution.worker_instance_id,
            observed_at=self._now(),
        )
        if result.disposition is not StageDisposition.READY:
            # The items still running finish first: a cancelled Task keeps
            # none of their work (the cancel covers them), a blocked one keeps
            # what completed -- a later run carries a completed item, and one
            # that did not complete is left as it stood.
            finished = pool.drain()
            if result.disposition is StageDisposition.CANCELLED:
                current_task = registry.task(task_id)
                if current_task.lifecycle is not TaskLifecycle.CANCEL_REQUESTED:
                    raise RuntimeError(
                        "stage reported cancellation without a pending task cancellation"
                    )
                registry.apply_cancel(task_id=task_id, observed_at=self._now())
                self._publish_projection(task_id)
                return False
            for other, value in finished:
                if value.disposition is StageDisposition.READY:
                    self._complete(context, task_id, execution, other, value)
            # A domain adapter's refusal is recorded as the refusal it is,
            # whatever its length: a code longer than the record accepts
            # would fail this write and turn a block into an interruption.
            registry.block_work_item(
                task_id=task_id,
                execution_id=execution_id,
                stage_id=definition.stage_id,
                failure_code=(result.failure_code or "TASK_STAGE_DID_NOT_COMPLETE")[
                    :FAILURE_CODE_MAX_LENGTH
                ],
                failure_cause=result.failure_cause,
                observed_at=self._now(),
                deferred=result.disposition is StageDisposition.DEFERRED,
            )
            self._publish_projection(task_id)
            return False
        task = self._complete(context, task_id, execution, definition, result)
        return task.lifecycle in {TaskLifecycle.RUNNING, TaskLifecycle.CANCEL_REQUESTED}

    def _keep_stored_evidence(
        self,
        task: TaskRecord,
        execution: TaskExecution,
        adapter: TaskDomainAdapter,
        definitions: Mapping[str, WorkItemDefinition],
    ) -> None:
        """Seal what an interrupted execution stored before its receipt, where it verifies.

        The stage ran and stored its evidence, and the process stopped before the receipt: its
        own verifier decides, and only a stage whose evidence fails runs again (V101, EV1).
        """
        for item in self.registry.work_items(task.task_id):
            if item.lifecycle is not WorkItemLifecycle.READY_FOR_VERIFICATION:
                continue
            definition = definitions[item.stage_id]
            try:
                verified = _spanned(
                    self._runtime_path,
                    task.task_id,
                    execution.execution_id,
                    item.stage_id,
                    "recovery_verify",
                    partial(
                        adapter.verify_stage,
                        task=task,
                        execution=execution,
                        work_item=definition,
                        evidence=item.evidence,
                    ),
                )
            except ChildStartFailed:
                raise
            except Exception:
                verified = None
            if verified != item.evidence:
                self.registry.return_to_pending(
                    task_id=task.task_id, stage_id=item.stage_id, observed_at=self._now()
                )
                continue
            self.registry.verify_work_item(
                TaskStageReceipt.from_identity(
                    receipt_id=uuid4(),
                    task_id=task.task_id,
                    execution_id=execution.execution_id,
                    stage_id=definition.stage_id,
                    work_item_definition_hash=definition.definition_hash,
                    verifier_id=definition.verifier_id,
                    evidence=item.evidence,
                    status="VERIFIED",
                    failure_code=None,
                    observed_at=self._now(),
                )
            )
        self._publish_projection(task.task_id)

    def _complete(
        self,
        context: _RunnerContext,
        task_id: UUID,
        execution: TaskExecution,
        definition: WorkItemDefinition,
        result: StageExecutionResult,
    ) -> TaskRecord:
        registry = context.registry
        registry.mark_ready(
            task_id=task_id,
            execution_id=execution.execution_id,
            stage_id=definition.stage_id,
            evidence=result.evidence,
            observed_at=self._now(),
        )
        verified = _spanned(
            self._runtime_path,
            task_id,
            execution.execution_id,
            definition.stage_id,
            "verify",
            partial(
                context.adapter.verify_stage,
                task=registry.task(task_id),
                execution=execution,
                work_item=definition,
                evidence=result.evidence,
            ),
        )
        if verified != result.evidence:
            raise RuntimeError("stage verifier changed the proposed evidence identity")
        task = registry.verify_work_item(
            TaskStageReceipt.from_identity(
                receipt_id=uuid4(),
                task_id=task_id,
                execution_id=execution.execution_id,
                stage_id=definition.stage_id,
                work_item_definition_hash=definition.definition_hash,
                verifier_id=definition.verifier_id,
                evidence=verified,
                status="VERIFIED",
                failure_code=None,
                observed_at=self._now(),
            )
        )
        self._publish_projection(task_id)
        return task

    def _publish_projection(self, task_id: UUID) -> None:
        projection = self.registry.safe_projection(task_id)
        if self.projection_sink is not None:
            self.projection_sink(projection)


def _declared_width(adapter: TaskDomainAdapter, task: TaskRecord, width: int | None) -> int:
    """How many of this Task's independent work items may run at once: what
    the adapter declares (`concurrent_work_items(task)`), within the cores its
    workspace's CPU budget gives it when it starts, else the host's processors;
    1 for an adapter that declares nothing. Each work item may hold a worker
    process, so under several Hosts at once a width from the processors alone
    multiplied the machine's processes (V436)."""

    declared = getattr(adapter, "concurrent_work_items", None)
    if declared is None:
        return 1
    return max(1, min(int(declared(task)), width or os.cpu_count() or 1))


class _SqliteHeartbeatStore:
    """Operational liveness sidecar; it never owns task lifecycle."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=5.0) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_task_heartbeat (
                    execution_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    worker_instance_id TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    signal_json TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(workspace_task_heartbeat)"
                ).fetchall()
            }
            for name, definition in (
                ("task_id", "TEXT"),
                ("sequence", "INTEGER"),
                ("signal_json", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE workspace_task_heartbeat ADD COLUMN {name} {definition}"
                    )

    def touch(self, signal: TaskHeartbeatSignal) -> None:
        with sqlite3.connect(self.path, timeout=5.0) as connection:
            connection.execute(
                """
                INSERT INTO workspace_task_heartbeat (
                    execution_id, task_id, worker_instance_id, heartbeat_at,
                    sequence, signal_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    task_id = excluded.task_id,
                    worker_instance_id = excluded.worker_instance_id,
                    heartbeat_at = excluded.heartbeat_at,
                    sequence = excluded.sequence,
                    signal_json = excluded.signal_json
                """,
                (
                    str(signal.execution_id),
                    str(signal.task_id),
                    str(signal.worker_instance_id),
                    signal.observed_at.isoformat(),
                    signal.sequence,
                    signal.model_dump_json(),
                ),
            )

    def read(self, execution_id: UUID) -> TaskHeartbeatSignal | None:
        with sqlite3.connect(self.path, timeout=5.0) as connection:
            row = connection.execute(
                """
                SELECT signal_json FROM workspace_task_heartbeat
                WHERE execution_id = ?
                """,
                (str(execution_id),),
            ).fetchone()
        return (
            None
            if row is None or row[0] is None
            else TaskHeartbeatSignal.model_validate_json(row[0])
        )


__all__ = [
    "StageDisposition",
    "StageExecutionResult",
    "TaskControlRunner",
    "TaskDomainAdapter",
    "TaskHeartbeatReader",
    "TaskHeartbeatReadout",
    "heartbeat_store_path",
]
