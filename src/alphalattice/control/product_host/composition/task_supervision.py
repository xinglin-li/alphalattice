"""The Supervisor and the Recovery Center over Task Control (GY2, V83).

A Host thread, outside the worker process that runs a Task's numerics (W10), reads each
unfinished Task's recovery view and work board, lets Guanyin's rules classify it
(`guanyin/supervision.py`) and keeps the incident records: a new incident opens a record, a
seen one is marked seen again with the remedies offered now, and one whose condition cleared is
resolved. A Task's read carries its open incident, so an agent following the Task wakes on it
(WK).

The Recovery Center performs one remedy the Host offered for an open incident, only through the
existing operation (RECOVER or CANCEL, confirmed against the Task's version; REPLAN answers the
owner's re-plan request, which admits nothing), and records the attempt with what the Task became
afterwards. It adds no power: what a person or an agent may do is what the recovery view offers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from threading import Event, Lock, Thread
from typing import Literal, Protocol, cast
from uuid import UUID

from alphalattice.control.guanyin.tasks.supervision import (
    IncidentRecord,
    IncidentStore,
    OfferedRemedy,
    RecoveryAttempt,
    TaskFacts,
    classify,
    first_detection,
    incident_key,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.task_recovery import (
    TaskAttentionFact,
    task_attention,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord, WorkItemLifecycle
from alphalattice.control.task_control.registry import TaskNotFoundError
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)


class _Host(Protocol):
    """What the Supervisor reads and acts through: the Host's operations, as they answer."""

    @property
    def workspace_session(self) -> WorkspaceApplicationSession: ...

    @property
    def dispatcher(self) -> LocalBackgroundDispatcher: ...

    def recovery_view(self, task_id: UUID) -> dict[str, object]: ...

    def status(self, task_id: UUID, *, wait_seconds: float | None = None) -> dict[str, object]: ...

    def execute(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: Literal["HUMAN", "EXTERNAL_AUTOMATION"] = "HUMAN",
    ) -> dict[str, object]: ...


SUPERVISION_SECONDS = 30.0
"""How often the Supervisor reads the unfinished Tasks: a read of the Task registry and one
recovery view each, the heartbeat's own cadence being a few seconds."""

_FINISHED = frozenset({TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED})

Remedy = Literal["CANCEL", "RECOVER", "REPLAN"]


class TaskSupervisor:
    """Guanyin's Supervisor thread and Recovery Center for one Host's workspace."""

    def __init__(self, operations: _Host, *, interval: float = SUPERVISION_SECONDS) -> None:
        """Supervise the Tasks this Host's operations serve."""
        self.operations = operations
        self.store = IncidentStore(operations.workspace_session.workspace)
        self.interval = interval
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------ the thread

    def start(self) -> None:
        """Begin supervising; idempotent."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="guanyin-supervisor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop supervising and join the thread."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=max(self.interval, 1.0) * 2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.supervise_once()
            except Exception:
                continue

    # ------------------------------------------------------------------ the rules' pass

    def supervise_once(self) -> tuple[IncidentRecord, ...]:
        """Classify every unfinished Task once; answer the incidents that opened."""
        with self._lock:
            now = self.operations.dispatcher.clock()
            registry = self.operations.workspace_session.task_control_registry
            tasks = registry.tasks()
            observed_hashes = {str(task.task_id): task.record_hash for task in tasks}
            attention = self.attention(tasks)
            seen: set[str] = set()
            opened: list[IncidentRecord] = []
            for task in tasks:
                if task.lifecycle in _FINISHED:
                    continue
                view = self.operations.recovery_view(task.task_id)
                observed_hash = view.get("task_record_hash")
                if isinstance(observed_hash, str):
                    observed_hashes[str(task.task_id)] = observed_hash
                status = cast(Mapping[str, object], view["status"])
                liveness = cast(Mapping[str, object], view["liveness"])
                current_lifecycle = str(view["lifecycle"])
                fact = attention.get(task.task_id)
                if (
                    current_lifecycle in {"BLOCKED", "RECOVERY_REQUIRED"}
                    and fact is not None
                    and not fact.unresolved
                    and fact.task_record_hash == view.get("task_record_hash")
                ):
                    # A stopped Task with current owner proof no longer has a finding
                    # to reopen. RUNNING and QUEUED still need liveness/parked rules.
                    continue
                record, items = registry.task_with_work_items(task.task_id)
                current = cast(str | None, status.get("current_stage"))
                started = next(
                    (
                        item.started_at
                        for item in items
                        if item.stage_id == current
                        and item.lifecycle is WorkItemLifecycle.IN_PROGRESS
                    ),
                    None,
                )
                facts = TaskFacts(
                    task_id=str(task.task_id),
                    task_kind=task.task_kind,
                    execution_id=None
                    if record.latest_execution_id is None
                    else str(record.latest_execution_id),
                    lifecycle=str(view["lifecycle"]),
                    operation_running=bool(view["operation_running"]),
                    liveness=str(liveness["status"]),
                    telemetry=str(liveness["telemetry"]),
                    current_stage=current,
                    stage_started_at=started,
                )
                finding = classify(facts, now)
                if finding is None:
                    continue
                key = incident_key(facts, finding)
                seen.add(key)
                remedies = _remedies(view)
                kept = self.store.get(key)
                if kept is None or kept.state == "RESOLVED":
                    kept = IncidentRecord.create(
                        key=key,
                        task_id=facts.task_id,
                        task_kind=facts.task_kind,
                        execution_id=facts.execution_id,
                        code=finding.code,
                        incident=first_detection(facts, finding, now),
                        state="OPEN",
                        last_seen_at=now,
                        remedies=remedies,
                    )
                    opened.append(kept)
                else:
                    kept = kept.changed(last_seen_at=now, remedies=remedies)
                self.store.put(kept)
            for kept in self.store.records():
                if kept.state == "OPEN" and kept.key not in seen:
                    try:
                        resolved_task_id = UUID(kept.task_id)
                    except ValueError:
                        resolved_task = None
                    else:
                        try:
                            resolved_task = registry.task(resolved_task_id)
                        except TaskNotFoundError:
                            resolved_task = None
                    observed_hash = observed_hashes.get(kept.task_id)
                    self.store.put(
                        kept.changed(
                            state="RESOLVED",
                            resolved_at=now,
                            resolved_task_record_hash=(
                                None
                                if resolved_task is None
                                or resolved_task.record_hash != observed_hash
                                else resolved_task.record_hash
                            ),
                        )
                    )
            return tuple(opened)

    def attention(self, tasks: Iterable[TaskRecord]) -> dict[UUID, TaskAttentionFact]:
        """Resolve current attention from one link read, one incident read and canonical Tasks.

        Records already supplied by the caller are the canonical records for this
        projection batch. A child outside that batch is read through Task Control;
        a missing child or read-side OS error leaves its link unresolved. Typed Task
        Control authority refusals remain refusals. Recovery-link read errors are
        also allowed to retain Task Control's located refusal rather than being guessed.
        """
        records = tuple(tasks)
        registry = self.operations.workspace_session.task_control_registry
        canonical = registry.record_collection().records
        by_id = {record.task_id: record for record in canonical}
        admission_order = {record.task_id: index for index, record in enumerate(canonical)}
        by_id.update((record.task_id, record) for record in records)
        completed: dict[tuple[str, str], TaskRecord] = {}
        for record in by_id.values():
            if record.lifecycle is TaskLifecycle.SUCCEEDED:
                key = (record.task_kind, record.plan.plan_hash)
                previous = completed.get(key)
                if previous is None or (
                    admission_order.get(record.task_id, -1)
                    > admission_order.get(previous.task_id, -1)
                ):
                    completed[key] = record
        links = registry.recovery_links()
        incidents = self.store.records()
        successor_ids = {
            link.successor_task_id
            for link in links
            if link.successor_task_id is not None
            and link.source_task_id in by_id
            and link.source_record_hash == by_id[link.source_task_id].record_hash
        }
        successors = {
            task_id: record for task_id, record in by_id.items() if task_id in successor_ids
        }
        for task_id in sorted(successor_ids - successors.keys(), key=str):
            try:
                successors[task_id] = registry.task(task_id)
            except (TaskNotFoundError, OSError):
                # No readable canonical child means the explicit link cannot prove completion.
                continue
        return {
            record.task_id: task_attention(
                record,
                successors=successors,
                links=links,
                incidents=incidents,
                superseded_by=completed.get((record.task_kind, record.plan.plan_hash)),
                superseded_is_later=(
                    (candidate := completed.get((record.task_kind, record.plan.plan_hash)))
                    is not None
                    and record.task_id in admission_order
                    and candidate.task_id in admission_order
                    and admission_order[candidate.task_id] > admission_order[record.task_id]
                ),
            )
            for record in records
        }

    def open_incident(self, task_id: UUID) -> IncidentRecord | None:
        """The Task's open incident, as the rules last found it."""
        return next(
            (
                record
                for record in self.store.records()
                if record.task_id == str(task_id) and record.state == "OPEN"
            ),
            None,
        )

    # ------------------------------------------------------------------ the Recovery Center

    def incidents(self) -> dict[str, object]:
        """Every incident record, open ones first: what was found and what was done."""
        records = self.store.records()
        return {
            "status": "TASK_INCIDENTS",
            "incidents": [record.model_dump(mode="json") for record in records],
            "open": sum(record.state == "OPEN" for record in records),
            "detail": (
                f"{sum(record.state == 'OPEN' for record in records)} open of {len(records)}."
                if records
                else "No incident has been found."
            ),
        }

    def remediate(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        selected_by: Literal["USER_COMMAND", "AGENT_PROPOSAL"],
    ) -> dict[str, object]:
        """Perform one remedy the Host offers for an open incident, and record the attempt."""
        assert request.task_id is not None and request.incident_key is not None
        record = self.store.get(request.incident_key)
        if record is None or record.task_id != str(request.task_id):
            return _refused("guanyin.incident_unknown", request)
        if record.state != "OPEN":
            return _refused("guanyin.incident_resolved", request)
        view = self.operations.recovery_view(request.task_id)
        offered = {remedy.action: remedy for remedy in _remedies(view)}
        chosen = offered.get(cast(Remedy, request.remedy))
        if chosen is None or not chosen.available:
            return {
                **_refused("guanyin.remedy_not_offered", request),
                "offered": [
                    remedy.model_dump(mode="json")
                    for remedy in offered.values()
                    if remedy.available
                ],
            }
        if chosen.action == "REPLAN":
            answer: dict[str, object] = {
                "status": "REPLAN_OFFERED",
                "next_requests": view.get("next_requests") or {},
            }
        else:
            answer = self.operations.execute(
                PortfolioResearchOperationRequest(
                    operation=cast("Literal['RECOVER', 'CANCEL']", chosen.operation),
                    task_id=request.task_id,
                    expected_task_hash=request.expected_task_hash,
                ),
                caller="HUMAN" if selected_by == "USER_COMMAND" else "EXTERNAL_AUTOMATION",
            )
        after = self.operations.status(request.task_id)
        attempt = RecoveryAttempt(
            ordinal=len(record.attempts) + 1,
            action=chosen.action,
            selected_by=selected_by,
            expected_task_hash=request.expected_task_hash or str(view["task_record_hash"]),
            requested_at=self.operations.dispatcher.clock(),
            disposition=str(answer.get("disposition") or answer.get("status") or "ANSWERED"),
            lifecycle_after=cast(str | None, after.get("lifecycle")),
            verified_stages_after=cast(int | None, after.get("verified_stage_count")),
            failure_code=cast(str | None, answer.get("failure_code")),
        )
        record = record.changed(attempts=(*record.attempts, attempt))
        self.store.put(record)
        # A recovery or a cancellation acted on the Task: the receipt's outcome is the Task's
        # and its owner's, so `--wait` follows a resumed Task and a refused remedy is refused,
        # never read off the nested answer (V508). A re-plan leaves the Task as it was.
        acted = chosen.action != "REPLAN"
        code = answer.get("failure_code") if acted else None
        return {
            "status": "REMEDY_ATTEMPTED",
            **(
                {"task_id": str(request.task_id), "lifecycle": after.get("lifecycle")}
                if acted
                else {}
            ),
            **({"failure_code": code} if code else {}),
            "incident": record.model_dump(mode="json"),
            "attempt": attempt.model_dump(mode="json"),
            "answer": answer,
            "next_requests": {
                "status": {"operation": "STATUS", "task_id": str(request.task_id)},
                "incidents": {"operation": "TASK_INCIDENTS"},
            },
        }


def _remedies(view: Mapping[str, object]) -> tuple[OfferedRemedy, ...]:
    actions = cast(list[Mapping[str, object]], view["actions"])
    return tuple(
        OfferedRemedy(
            action=cast(Remedy, action["action"]),
            operation=str(action["operation"]),
            available=bool(action["available"]),
            reason=str(action["reason"])[:300],
        )
        for action in actions
    )


def _refused(code: str, request: PortfolioResearchOperationRequest) -> dict[str, object]:
    return {
        "status": "REFUSED",
        "failure_code": code,
        "next_requests": {"incidents": {"operation": "TASK_INCIDENTS"}},
        "task_id": None if request.task_id is None else str(request.task_id),
    }
