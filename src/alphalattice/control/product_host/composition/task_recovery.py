"""One Task's recovery view: what stopped, what stays verified, what its owners permit.

Project Guanyin at G0 for the Local Web product: a read-only composition of
facts that existing owners already hold, read as one snapshot. Task Control
owns the lifecycle, the work board with its verified evidence, the cancellation
flags and the record version, and hands all of them over under one lock
(`board`), so every part below describes the same record version; the
dispatcher owns whether this Host's command has returned and why a worker
stopped outside Task Control, and those two facts are kept apart from Task
Control's state rather than masked over it; the service owns which Task kinds
it can resume; the owner that admits each Task kind declares its re-PLAN (V188),
and the service hands those declarations over. Nothing here
repairs, retries, kills a process, restarts a native Agent or changes a
scientific policy -- `active_remediation_attempt_count` is 0 by contract, a
fact about this view -- and a timer establishes nothing: a heartbeat that is
late is reported as not recently observed, never as termination.

What this Host does not observe is said so: it records no model-execution
facts for its Tasks, so `model_facts` is NOT_OBSERVED and no generic
`RuntimeGuardianProjection` is invented from zeros; the generic pieces that
are source-backed (a `HealthSignal` from lifecycle and heartbeat, the
`RuntimeIncident` of a recorded stop) are carried as such. No domain guardian
projection is published for these Task kinds (the per-domain projections
retired with RT, 2026-09-29; workspace maintenance explains its own stop
codes), and this view does not transplant one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from alphalattice.control.guanyin.data.workspace_maintenance import maintenance_failure_detail
from alphalattice.control.guanyin.tasks.supervision import IncidentRecord
from alphalattice.control.observation_runtime.guardian import (
    GuardianHealth,
    HealthSignal,
    RuntimeGuardianProjection,
    RuntimeIncident,
)
from alphalattice.control.product_host.composition.plain_refusals import (
    STOP_WORDS_BOUND,
    explain,
)
from alphalattice.control.task_control.contracts import (
    PLAN_WORK_ITEM_LIMIT,
    StageFailureCause,
    TaskExecution,
    TaskHeartbeatSignal,
    TaskLifecycle,
    TaskRecord,
    TaskRecoveryLink,
    TaskReplan,
    TaskSafeProjection,
    WorkItemLifecycle,
)
from alphalattice.control.task_control.registry import (
    LEDGER_REBUILT_DETAIL,
    LEDGER_REBUILT_NEXT,
    TaskBoardSnapshot,
)
from alphalattice.control.task_control.runner import TaskHeartbeatReadout
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.failure_codes import (
    FAILURE_DETAIL_WITHHELD,
    safe_failure_code,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

STALE_HEARTBEAT_SECONDS = 30.0
"""The runner's own stale-execution threshold (`TaskControlRunner.stale_active_tasks`), reused,
not re-decided. Liveness is read the way that owner reads it: the later of the durable Task
Control timestamp (a stage boundary) and the runner's operational heartbeat for the same
execution, the latter through the read-only `TaskHeartbeatReader` over the runner's sidecar."""

_TERMINAL = frozenset({TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED})
_DATA_KINDS = frozenset(
    {"workspace_preparation", "workspace_data_update", "research_input_capture"}
)
"""Task kinds whose stop codes the workspace-maintenance guardian already explains."""

_MODEL_OPERATIONS = frozenset({"EVIDENCE_PREPARE", "EVIDENCE_REFRESH", "CRO_REVIEW"})
"""Admitting operations whose Tasks run an analyst or reviewer actor."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreservedStage(_Contract):
    """Retain the complete Task Control work-item evidence and stop/recovery state.

    One work item of the Task's plan as Task Control holds it, with every evidence
    reference its verification recorded (`evidence_count` is that same number; nothing
    is truncated here). A verified stage is kept through recovery; a cancelled or blocked
    stage is kept as the record of where the Task stopped.
    """

    stage_id: str
    lifecycle: str
    evidence: tuple[str, ...]
    evidence_count: int = Field(ge=0)


class StopFact(_Contract):
    """Retain Task Control's typed stop and bounded public explanation.

    The typed reason Task Control recorded for a stop, with a bounded user-safe
    explanation. `recoverable` is Task Control's own RECOVERY_REQUIRED, never a guess
    from the code. `cause` is what the stopped stage's owner saw beside the code, as its
    work item keeps it (V444).
    """

    code: str = Field(min_length=1, max_length=240)
    stage_id: str | None
    detail: str = Field(min_length=1, max_length=500)
    recoverable: bool
    cause: StageFailureCause | None = None


class WorkerStop(_Contract):
    """Retain the dispatcher's bounded operational worker-stop fact.

    The dispatcher's fact about the last time this Host's own command for the Task
    stopped outside Task Control: a typed code or the withheld marker, and the exception
    class -- never the message. Kept apart from Task Control's `stop`.
    """

    code: str = Field(min_length=1, max_length=240)
    failure_type: str | None = Field(default=None, max_length=80)
    detail: str = Field(min_length=1, max_length=500)


class LivenessFact(_Contract):
    """What the execution's heartbeats say, and only that, with their sources named.

    `telemetry` says what is available: the runner's operational signal bound to this
    Task, execution and worker (OPERATIONAL); Task Control's durable timestamp alone
    because no signal is recorded for this execution (DURABLE_ONLY), a sidecar could not
    be read so a signal may be hidden (UNREADABLE) or the recorded signal is not bound to
    this execution's worker (UNBOUND); or nothing at all before an execution exists
    (NONE). A signal of another execution never counts. `last_heartbeat_source` says
    which of the two timestamps -- the later one, the runner's own rule -- supplies
    `last_heartbeat_at` and `age_seconds`: an available signal is not necessarily the
    later timestamp, and an older signal is reported at its own age
    (`operational_age_seconds`), never as refreshed.
    """

    status: Literal["OBSERVED", "NOT_RECENT", "NOT_OBSERVED", "NOT_APPLICABLE"]
    telemetry: Literal["OPERATIONAL", "DURABLE_ONLY", "UNREADABLE", "UNBOUND", "NONE"]
    last_heartbeat_at: datetime | None
    last_heartbeat_source: Literal["OPERATIONAL_SIGNAL", "DURABLE_TIMESTAMP"] | None = None
    operational_at: datetime | None = None
    operational_age_seconds: float | None = Field(default=None, ge=0)
    signal_sequence: int | None = Field(default=None, ge=1)
    durable_at: datetime | None = None
    age_seconds: float | None = Field(default=None, ge=0)
    note: str = Field(min_length=1, max_length=400)


class PermittedAction(_Contract):
    """Name an owner-supported recovery action and its exact scope or refusal.

    One choice an owner supports right now, named by the operation that performs it,
    with what it touches and what it changes. An unavailable action says why.
    """

    action: Literal["CANCEL", "RECOVER", "REPLAN"]
    operation: str = Field(min_length=1, max_length=64)
    admits: bool
    """Whether performing `operation` admits or resumes work (a preview never does)."""
    available: bool
    requires_confirmation: bool
    scope: str = Field(min_length=1, max_length=400)
    expected_effect: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=300)


class TaskAttentionFact(_Contract):
    """Whether this exact Task version still waits on a recovery choice.

    An explicit owner-replan link resolves only when it names this exact source
    version and its canonical child has succeeded. A Guanyin resolution resolves
    only the exact Task version it observed; historical incidents without that
    binding cannot clear current attention.
    """

    unresolved: bool = Field(
        description="Whether this exact Task version still waits for a recovery choice."
    )
    resolution: Literal["NOT_STOPPED", "STOPPED", "SUCCESSOR_SUCCEEDED", "INCIDENT_RESOLVED"] = (
        Field(description="Which current owner fact resolves or still holds this attention.")
    )
    task_record_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="The canonical Task record version this attention describes.",
    )
    successor_task_id: str | None = Field(
        default=None,
        description="The exact linked successor Task id, when a confirmed link applies.",
    )
    successor_task_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="The canonical linked successor record hash, when it was readable.",
    )
    successor_lifecycle: str | None = Field(
        default=None,
        description="The linked successor's current lifecycle, when its record was readable.",
    )
    incident_key: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="The matching Guanyin incident that held or resolved this attention.",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_resolution(self) -> TaskAttentionFact:
        """Keep the resolution vocabulary consistent with its proof fields."""
        if self.unresolved != (self.resolution == "STOPPED"):
            raise ValueError("task attention unresolved state differs from its resolution")
        if self.resolution == "SUCCESSOR_SUCCEEDED" and (
            self.successor_task_id is None
            or self.successor_task_hash is None
            or self.successor_lifecycle != TaskLifecycle.SUCCEEDED.value
        ):
            raise ValueError("successful successor attention needs its canonical child facts")
        if self.resolution == "INCIDENT_RESOLVED" and self.incident_key is None:
            raise ValueError("resolved incident attention needs its incident key")
        return self


class TaskRecoveryView(_Contract):
    """Seal read-only recovery state, preserved evidence and owner-permitted actions.

    Seal a read-only task recovery projection with preserved evidence and owner-permitted actions.
    """

    kind: Literal["TaskRecoveryView"] = "TaskRecoveryView"
    guardian_mode: Literal["G0_READ_ONLY"] = "G0_READ_ONLY"
    active_remediation_attempt_count: Literal[0] = 0
    task_id: str
    task_kind: str
    lifecycle: str
    """Task Control's own lifecycle for this record version. The dispatcher's
    operation-return mask is not applied here; `operation_running` carries that fact."""
    operation_running: bool
    """This Host's command for the Task has not returned (a settled Task may still be
    publishing; a cancelled one may still be unwinding). Never a lifecycle."""
    cancellation: Literal["NOT_REQUESTED", "REQUESTED", "ACKNOWLEDGED"]
    stop: StopFact | None
    worker_failure: WorkerStop | None
    resume_refusal: str | None = Field(default=None, min_length=1, max_length=240)
    """Why this interrupted or parked Task cannot resume under what is installed now: its
    owner's typed refusal, or Task Control's when the compatibility moved against the latest
    execution. Asked before any resume; None when it would resume or nothing waits."""
    liveness: LivenessFact
    execution_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The compatibility binding of the latest execution; absent until a Task has started."""
    verified_stage_count: int = Field(ge=0)
    total_stage_count: int = Field(ge=1)
    stages: tuple[PreservedStage, ...] = Field(max_length=PLAN_WORK_ITEM_LIMIT)
    """Every work item of the plan, as many as Task Control's plan holds
    (`PLAN_WORK_ITEM_LIMIT`): a coverage run carries one set of its stages
    per unit."""
    artifact_refs: tuple[str, ...]
    """The safe projection's artifact references as Task Control publishes them (its own
    contract keeps the first sixteen distinct receipt references); the complete references
    are on `stages`."""
    actions: tuple[PermittedAction, ...]
    health: HealthSignal
    incidents: tuple[RuntimeIncident, ...]
    model_facts: Literal["NOT_OBSERVED"] = "NOT_OBSERVED"
    """This Host records no model-execution facts for its Tasks; a read-only view that
    called no model says nothing about what the Task used."""
    guardian: RuntimeGuardianProjection | None = None
    guardian_availability: Literal["PUBLISHED", "NOT_PUBLISHED"]
    """A domain Guanyin projection for this Task, when one is published by its owner."""
    task_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The exact Task version every Task Control fact above describes; a confirmation carries
    it back and the owner refuses to act on any other version."""
    observed_at: datetime
    attention: TaskAttentionFact | None = Field(
        default=None,
        description="Current recovery attention bound to task_record_hash, when requested.",
    )
    view_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def omit_absent_attention(self, handler):  # type: ignore[no-untyped-def]
        """Keep the pre-attention view identity when its optional fact is absent."""
        serialized = handler(self)
        if self.attention is None:
            serialized.pop("attention", None)
        return serialized

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> TaskRecoveryView:
        """Require aware observation time, bounded verified stages and exact recovery view identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Clock, stage counts, guardian availability or view_hash differs.
        """
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("task recovery view clock must be timezone-aware")
        if self.attention is not None and self.attention.task_record_hash != self.task_record_hash:
            raise ValueError("task recovery attention differs from the viewed Task version")
        if self.verified_stage_count > self.total_stage_count:
            raise ValueError("task recovery view verified count exceeds plan")
        if (self.guardian is None) != (self.guardian_availability == "NOT_PUBLISHED"):
            raise ValueError("task recovery view guardian availability is inconsistent")
        if self.view_hash != canonical_hash(self.model_dump(mode="json", exclude={"view_hash"})):
            raise ValueError("task recovery view hash is invalid")
        return self


def _with_hash(model, values: dict[str, object], field: str):  # type: ignore[no-untyped-def]
    provisional = model.model_construct(**values, **{field: "0" * 64})
    identity = provisional.model_dump(mode="json", exclude={field})
    return model(**values, **{field: canonical_hash(identity)})


def stop_detail(task_kind: str, code: str, source: str) -> str:
    """A bounded, user-safe explanation of one recorded stop code.

    Task Control's own codes and this Host's worker codes are explained here; the
    data Tasks' codes by the workspace-maintenance guardian that already owns
    them. Anything else is stated as what it is: the owner's typed code, with no
    retry implied. A code without installed words stays in the code field; it is never
    substituted for the person's explanation.
    """
    if code == "task_control.child_start_failed":
        return refusal_words(code)["detail"]
    if code == "task_control.ledger_rebuilt":
        return LEDGER_REBUILT_DETAIL
    if code == "TASK_WRITER_INTERRUPTED":
        return (
            "The writer stopped before the Task could record its next checkpoint. Stages "
            "already verified are kept; review the recovery view before resuming this Task."
        )
    if code.startswith("TASK_RECOVERY_EVIDENCE_INVALID:"):
        return (
            "A previously verified stage's evidence could not be verified during recovery, "
            "so the Task stopped before continuing. Restore the evidence or declare the work "
            "again through its owner's plan."
        )
    if code == "TASK_EXECUTION_INTERRUPTED":
        return (
            "The worker stopped before the current stage was verified. Verified stages and "
            "their artifacts are kept; resuming this same Task re-verifies them from their "
            "evidence and runs only the unverified stage again."
        )
    if code == "TASK_CANCELLED_BEFORE_START":
        return "Cancelled before any stage ran. Nothing was computed or published."
    if code in {"TASK_CANCELLED_AT_SAFE_CHECKPOINT", "TASK_CANCELLED_AFTER_WRITER_STOPPED"}:
        return (
            "The cancellation was acknowledged at a safe checkpoint. Stages verified before "
            "it are kept as recorded; the Task will not continue."
        )
    if code.startswith("TASK_STAGE_RAISED:"):
        return (
            "This stage raised the same unnamed error on three attempts in a row (its type and "
            "message are beside the code), so it was stopped rather than resumed again. Stages "
            "verified before it are kept. Once its cause is fixed, RECOVER with this version, "
            "as the recovery view offers it, reopens the same Task and runs the stage again."
        )
    if code == "TASK_STAGE_DID_NOT_COMPLETE":
        return (
            "A stage ended without completing and without a typed reason; its owner stopped "
            "the Task there. Nothing was retried."
        )
    if code in {"task_control.recovery_version_stale", "task_control.start_version_stale"}:
        return (
            "A confirmed resume reached Task Control after the Task had moved to another "
            "version and was refused there; nothing was started or changed. Review the Task "
            "again and confirm the version it shows now."
        )
    if code == "portfolio_application.task_not_dispatched":
        return (
            "The worker could not start this queued Task because the workspace's one active "
            "Task was still open (an interrupted Task holds that place). Nothing has run; resume "
            "it once the active Task has been resumed or cancelled."
        )
    if code.endswith("storage.managed_capacity_exceeded"):
        return (
            "The stage's write exceeds the workspace storage cap. Raise the cap in Settings "
            "or preview and confirm a cleanup, then resume the Task's offered request; "
            "its verified stages and retained results stay intact."
        )
    if code.endswith(":storage.disk_space_insufficient"):
        return (
            "The stage's write was refused by the workspace storage budget before anything "
            "was placed; nothing partial remains and no generation or receipt was published. "
            "Free managed space -- preview and confirm a cleanup plan, or review pinned "
            "inputs -- and prepare again; what is already sealed still reads."
        )
    if code.startswith("local_application.recovery_command_not_installed"):
        return (
            "This service has no recovery command for this Task kind; another adapter owns "
            "its resumption. The Task and its verified stages are unchanged."
        )
    if source == "WORKER":
        return (
            "This Host's command stopped outside Task Control with the recorded failure "
            "class; the Task's own lifecycle is the authority on what it may do next."
        )
    if task_kind in _DATA_KINDS:
        return maintenance_failure_detail(code)[:STOP_WORDS_BOUND]
    # An owner that words its code says what holds and the way on, which a Task stopped on it
    # reads here too, not only a refused request (V504: RR5's book read the bare code).
    worded = explain(code).get("detail")
    if isinstance(worded, str) and worded:
        return worded[:STOP_WORDS_BOUND]
    return (
        "The Task stopped at a governed boundary whose reason has no installed explanation. "
        "Nothing was retried, and its verified stages are kept as recorded. Read its recovery "
        "view for the actions its owner permits."
    )[:STOP_WORDS_BOUND]


def task_attention(
    task: TaskRecord,
    *,
    successors: Mapping[UUID, TaskRecord] | None = None,
    links: Iterable[TaskRecoveryLink] = (),
    incidents: Iterable[IncidentRecord] = (),
) -> TaskAttentionFact:
    """Project current stop attention from canonical Task, link and incident facts.

    A confirmed successor link is evidence only for its exact source record hash.
    A matching child must be a distinct canonical Task of the same kind in
    ``SUCCEEDED``. A pending, failed, unreadable or invalid child keeps the source
    stopped unless another exact-version confirmed child has succeeded or a
    current-version resolved incident supplies independent resolution evidence.
    Successful unrelated Tasks are never considered.

    A resolved incident is accepted only when it names this exact Task record hash
    and the same execution. Legacy resolved incidents without that hash are not
    proof of resolution and leave a stopped Task unresolved. An open incident for
    this Task's current execution takes precedence over a resolved incident.
    """
    if task.lifecycle not in {TaskLifecycle.BLOCKED, TaskLifecycle.RECOVERY_REQUIRED}:
        return TaskAttentionFact(
            unresolved=False,
            resolution="NOT_STOPPED",
            task_record_hash=task.record_hash,
        )

    successor_records = successors or {}
    current_links = sorted(
        (
            link
            for link in links
            if link.source_task_id == task.task_id
            and link.source_record_hash == task.record_hash
            and link.successor_task_id is not None
            and link.successor_task_id != task.task_id
        ),
        key=lambda link: (link.recorded_at, str(link.successor_task_id)),
    )
    pending_successor: TaskAttentionFact | None = None
    for link in current_links:
        successor_id = link.successor_task_id
        assert successor_id is not None
        child = successor_records.get(successor_id)
        canonical_child = (
            child is not None
            and child.task_id == successor_id
            and child.task_id != task.task_id
            and child.task_kind == task.task_kind
        )
        if canonical_child:
            assert child is not None
            if child.lifecycle is TaskLifecycle.SUCCEEDED:
                return TaskAttentionFact(
                    unresolved=False,
                    resolution="SUCCESSOR_SUCCEEDED",
                    task_record_hash=task.record_hash,
                    successor_task_id=str(child.task_id),
                    successor_task_hash=child.record_hash,
                    successor_lifecycle=child.lifecycle.value,
                )
        if pending_successor is None:
            pending_successor = TaskAttentionFact(
                unresolved=True,
                resolution="STOPPED",
                task_record_hash=task.record_hash,
                successor_task_id=str(successor_id),
                successor_task_hash=(child.record_hash if canonical_child and child else None),
                successor_lifecycle=(child.lifecycle.value if canonical_child and child else None),
            )

    current_execution = None if task.latest_execution_id is None else str(task.latest_execution_id)
    matching_incidents = tuple(
        incident
        for incident in incidents
        if incident.task_id == str(task.task_id) and incident.execution_id == current_execution
    )
    open_incident = next(
        (incident for incident in matching_incidents if incident.state == "OPEN"),
        None,
    )
    if open_incident is not None:
        return TaskAttentionFact(
            unresolved=True,
            resolution="STOPPED",
            task_record_hash=task.record_hash,
            successor_task_id=(
                None if pending_successor is None else pending_successor.successor_task_id
            ),
            successor_task_hash=(
                None if pending_successor is None else pending_successor.successor_task_hash
            ),
            successor_lifecycle=(
                None if pending_successor is None else pending_successor.successor_lifecycle
            ),
            incident_key=open_incident.key,
        )

    resolved_incident = next(
        (
            incident
            for incident in matching_incidents
            if incident.state == "RESOLVED"
            and incident.resolved_task_record_hash == task.record_hash
        ),
        None,
    )
    if resolved_incident is not None:
        return TaskAttentionFact(
            unresolved=False,
            resolution="INCIDENT_RESOLVED",
            task_record_hash=task.record_hash,
            incident_key=resolved_incident.key,
        )
    if pending_successor is not None:
        return pending_successor
    return TaskAttentionFact(
        unresolved=True,
        resolution="STOPPED",
        task_record_hash=task.record_hash,
    )


def _worker_stop(failure: str) -> tuple[str, str | None]:
    """The dispatcher's `Class: message` as a typed code and a class name, never the message."""

    head, _separator, tail = failure.partition(": ")
    if _separator == "":
        return (safe_failure_code(failure) or FAILURE_DETAIL_WITHHELD), None
    return (safe_failure_code(tail.strip()) or FAILURE_DETAIL_WITHHELD), head.strip()[:80] or None


def _bound_signal(
    execution: TaskExecution, readout: TaskHeartbeatReadout
) -> TaskHeartbeatSignal | None:
    """The signal the runner wrote for this execution: bound to its Task, execution and
    worker. A readable sidecar holds at most one; the latest wins should copies exist."""

    return max(
        (
            signal
            for signal in readout.signals
            if signal.task_id == execution.task_id
            and signal.execution_id == execution.execution_id
            and signal.worker_instance_id == execution.worker_instance_id
        ),
        key=lambda signal: (signal.observed_at, signal.sequence),
        default=None,
    )


def _liveness(
    projection: TaskSafeProjection,
    execution: TaskExecution | None,
    readout: TaskHeartbeatReadout,
    observed_at: datetime,
) -> LivenessFact:
    """The runner's own rule (`stale_active_tasks`): the later of the durable timestamp and
    the operational signal, the signal counted only when bound to this execution's identity.
    Availability and the source of the reported age are two facts: a bound signal older
    than the durable timestamp is available, reported at its own age, and not the source."""

    durable = execution.last_heartbeat_at if execution is not None else None
    signal = _bound_signal(execution, readout) if execution is not None else None
    telemetry: Literal["OPERATIONAL", "DURABLE_ONLY", "UNREADABLE", "UNBOUND", "NONE"] = (
        "NONE"
        if execution is None
        else "OPERATIONAL"
        if signal is not None
        else "UNREADABLE"  # a bound signal may sit in the sidecar that could not be read
        if readout.unreadable
        else "UNBOUND"
        if readout.signals
        else "DURABLE_ONLY"
    )
    operational_at = signal.observed_at if signal is not None else None
    sequence = signal.sequence if signal is not None else None
    operational_age = (
        None if operational_at is None else max(0.0, (observed_at - operational_at).total_seconds())
    )
    last_seen: datetime | None = None
    last_source: Literal["OPERATIONAL_SIGNAL", "DURABLE_TIMESTAMP"] | None = None
    if durable is not None:
        if operational_at is not None and operational_at >= durable:
            last_seen, last_source = operational_at, "OPERATIONAL_SIGNAL"
        else:
            last_seen, last_source = durable, "DURABLE_TIMESTAMP"
    age = None if last_seen is None else max(0.0, (observed_at - last_seen).total_seconds())
    source = (
        f"the runner's heartbeat #{sequence}"
        if last_source == "OPERATIONAL_SIGNAL"
        else "Task Control's durable timestamp (a stage boundary); the runner's heartbeat "
        f"#{sequence} is older, {int(operational_age or 0)} s ago"
        if telemetry == "OPERATIONAL"
        else "Task Control's durable timestamp only (a stage boundary; no runner heartbeat is "
        "recorded for this execution)"
        if telemetry == "DURABLE_ONLY"
        else "Task Control's durable timestamp only (a runner heartbeat store could not be "
        "read, so a signal for this execution may be hidden)"
        if telemetry == "UNREADABLE"
        else "Task Control's durable timestamp only (the recorded heartbeat is not bound to this "
        "execution's worker and is not counted)"
        if telemetry == "UNBOUND"
        else "nothing"
    )
    if projection.lifecycle not in {TaskLifecycle.RUNNING, TaskLifecycle.CANCEL_REQUESTED}:
        return LivenessFact(
            status="NOT_APPLICABLE",
            telemetry=telemetry,
            last_heartbeat_at=last_seen,
            last_heartbeat_source=last_source,
            operational_at=operational_at,
            operational_age_seconds=operational_age,
            signal_sequence=sequence,
            durable_at=durable,
            age_seconds=age,
            note=(
                "The Task is not executing; its lifecycle, not a heartbeat, says where it stands."
            ),
        )
    if age is None:
        return LivenessFact(
            status="NOT_OBSERVED",
            telemetry=telemetry,
            last_heartbeat_at=None,
            durable_at=None,
            age_seconds=None,
            note=(
                "No execution heartbeat has been recorded for this Task yet. Whether a worker "
                "is running is unknown to this view; nothing is inferred from time alone."
            ),
        )
    if age > STALE_HEARTBEAT_SECONDS:
        return LivenessFact(
            status="NOT_RECENT",
            telemetry=telemetry,
            last_heartbeat_at=last_seen,
            last_heartbeat_source=last_source,
            operational_at=operational_at,
            operational_age_seconds=operational_age,
            signal_sequence=sequence,
            durable_at=durable,
            age_seconds=age,
            note=(
                f"No heartbeat for {int(age)} s by {source}. The worker may still be running or "
                "may have stopped; this view does not decide, retry, kill or restart anything."
            ),
        )
    return LivenessFact(
        status="OBSERVED",
        telemetry=telemetry,
        last_heartbeat_at=last_seen,
        last_heartbeat_source=last_source,
        operational_at=operational_at,
        operational_age_seconds=operational_age,
        signal_sequence=sequence,
        durable_at=durable,
        age_seconds=age,
        note=f"A heartbeat was recorded {int(age)} s ago by {source}.",
    )


def _actions(
    task: TaskRecord,
    projection: TaskSafeProjection,
    recoverable_kinds: frozenset[str],
    running: bool,
    replans: Mapping[str, TaskReplan],
    blocked_retry_reason: str | None = None,
    resume_refusal: str | None = None,
    replan_refusal: str | None = None,
) -> tuple[PermittedAction, ...]:
    short = str(task.task_id)[:8]
    lifecycle = projection.lifecycle
    cancel_reason = (
        "Task Control admits a cancellation request for this version."
        if projection.cancel_available and not projection.cancel_pending
        else "Cancellation is already requested; the worker has not acknowledged it yet."
        if projection.cancel_pending
        else f"The Task has ended ({lifecycle.value}); there is nothing to cancel."
    )
    cancel = PermittedAction(
        action="CANCEL",
        operation="CANCEL",
        admits=False,
        available=projection.cancel_available and not projection.cancel_pending,
        requires_confirmation=True,
        scope=(
            f"This Task only ({short}). Verified stages, their artifacts and every published "
            "result stay exactly as recorded; nothing is deleted or recomputed."
        ),
        expected_effect=(
            "Cancelled at once: it never started."
            if lifecycle is TaskLifecycle.QUEUED
            else (
                "Cancellation is requested; the worker acknowledges it at its next safe "
                "checkpoint, and until then the Task keeps running. Requested is not "
                "acknowledged."
            )
        ),
        reason=cancel_reason,
    )
    # RECOVER is the owner's own re-enqueue: an interrupted Task, or a queued Task no command
    # is driving any more (its command returned without a dispatch while another Task held the
    # workspace's active place). A queued Task whose command is still waiting needs nothing.
    # It is an attempt: the Task may stop again, and Task Control records why.
    interrupted = lifecycle is TaskLifecycle.RECOVERY_REQUIRED
    parked = lifecycle is TaskLifecycle.QUEUED and not running
    # A BLOCKED Task is resumable only when its owner permits retrying its
    # repaired cause; the owner's words are the reason.
    retryable = lifecycle is TaskLifecycle.BLOCKED and blocked_retry_reason is not None
    resumable = interrupted or parked or retryable
    installed = task.task_kind in recoverable_kinds
    recover = PermittedAction(
        action="RECOVER",
        operation="RECOVER",
        admits=True,
        available=resumable and installed and resume_refusal is None,
        requires_confirmation=True,
        scope=(
            f"This same queued Task ({short}); no new Task and no new declaration. Nothing has "
            "run; it is handed back to the worker and starts once the workspace's active place "
            "is free."
            if lifecycle is TaskLifecycle.QUEUED
            else (
                f"This same Task ({short}); no new Task and no new declaration. Its "
                f"{projection.verified_stage_count} verified stage(s) and their artifacts are "
                "kept and re-verified from their evidence; only the unverified stage runs again."
                + (
                    " The blocked stage's checkpoint is re-verified and reused; only the "
                    "unfinished part is computed, after its owner re-checks the repaired cause."
                    if retryable
                    else ""
                )
            )
        ),
        expected_effect=(
            "An attempt: the Task starts under its first execution if the confirmed version "
            "still holds. If it completes it publishes once, and an identical declaration "
            "afterwards is reused exactly; it may stop again, and Task Control records why."
            if lifecycle is TaskLifecycle.QUEUED
            else (
                "An attempt: the Task returns to RUNNING under a new execution if the confirmed "
                "version still holds. If it completes it publishes once, and an identical "
                "declaration afterwards is reused exactly; it may stop again, and Task Control "
                "records why."
            )
        ),
        reason=(
            f"What this Task was admitted under has changed since ({resume_refusal}), so it "
            "cannot resume as recorded. Cancel it and plan the same work again."
            if resume_refusal is not None
            else (
                "Task Control holds this Task as interrupted and this service resumes its kind."
                if interrupted
                else str(blocked_retry_reason)
                if retryable
                else "No command is driving this queued Task any more; this service can hand "
                "it back to the worker."
            )
            if resumable and installed
            else (
                "The Task is queued and its command is still waiting its turn; nothing to resume."
                if lifecycle is TaskLifecycle.QUEUED
                else f"The Task is {lifecycle.value}; only an interrupted or parked queued Task "
                "can be resumed."
            )
            if not resumable
            else f"This service has no recovery command for {task.task_kind}."
        ),
    )
    declared = replans.get(task.task_kind)
    preview, admitting = (
        (None, "NONE") if declared is None else (declared.preview, declared.admitting)
    )
    actor = admitting in _MODEL_OPERATIONS
    direct = preview is not None and preview == admitting
    replan = PermittedAction(
        action="REPLAN",
        operation=preview or admitting,
        admits=preview is None or direct,
        available=preview is not None and replan_refusal is None,
        requires_confirmation=True,
        scope=(
            f"A new Task through {admitting}. This Task stays exactly as recorded: "
            "nothing here is changed, cancelled or deleted."
            if direct
            else (
                f"A new preview through {preview}, which records a plan and admits nothing. This "
                "Task stays exactly as recorded: nothing here is changed, cancelled or deleted."
            )
            if preview
            else (
                f"{admitting} admits a Task and runs its actor; it is that owner's own "
                "confirmed admission, not a re-plan. This Task stays exactly as recorded."
            )
        ),
        expected_effect=(
            LEDGER_REBUILT_NEXT
            if task.failure_code == "task_control.ledger_rebuilt"
            and preview is not None
            and replan_refusal is None
            else f"Starts a new Task through {admitting}; it does not resume or change this Task."
            if direct
            else (
                f"Nothing is admitted until {admitting} is confirmed with that owner"
                + (", which runs an analyst or reviewer actor" if actor else "")
                + ". An identical declaration is reused exactly once this Task has succeeded "
                "(no second Task or publication); a changed one admits a new Task beside this "
                "one; while this Task runs or waits, the owner answers with its own disposition."
            )
            if preview
            else (
                f"Only {admitting}, confirmed with its owner, admits work; it runs an actor and "
                "may publish. No confirmation-free re-plan exists for this kind."
            )
        ),
        reason=(
            replan_refusal
            if replan_refusal is not None
            else f"The owner admits this Task kind directly through {admitting}, "
            "without a plan preview."
            if direct
            else f"{task.task_kind} is previewed through {preview} and admitted through "
            f"{admitting}, each with that owner's own confirmation."
            if preview
            else f"No confirmation-free preview exists for {task.task_kind}; {admitting} admits."
        ),
    )
    return (cancel, recover, replan)


def _health_and_incidents(
    task: TaskRecord,
    projection: TaskSafeProjection,
    liveness: LivenessFact,
    stop: StopFact | None,
    observed_at: datetime,
) -> tuple[HealthSignal, tuple[RuntimeIncident, ...]]:
    lifecycle = projection.lifecycle
    if lifecycle is TaskLifecycle.SUCCEEDED:
        health = GuardianHealth.TERMINAL_SUCCEEDED
    elif lifecycle in {TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}:
        health = GuardianHealth.TERMINAL_BLOCKED
    elif lifecycle in {
        TaskLifecycle.DEFERRED,
        TaskLifecycle.REVIEW_PENDING,
        TaskLifecycle.RECOVERY_REQUIRED,
    }:
        health = GuardianHealth.TERMINAL_DEFERRED
    elif liveness.status == "NOT_RECENT":
        health = GuardianHealth.LIVENESS_STALE
    elif lifecycle is TaskLifecycle.QUEUED:
        health = GuardianHealth.STARTING
    else:
        health = GuardianHealth.HEALTHY
    signal = _with_hash(
        HealthSignal,
        {
            "task_id": str(task.task_id),
            "status": health,
            "observed_at": observed_at,
            "last_heartbeat_at": liveness.last_heartbeat_at,
            "heartbeat_age_seconds": liveness.age_seconds,
        },
        "signal_hash",
    )
    incidents: list[RuntimeIncident] = []
    if stop is not None:
        incidents.append(
            _with_hash(
                RuntimeIncident,
                {
                    "incident_code": stop.code[:120],
                    "detected_at": observed_at,
                    "stage_id": stop.stage_id,
                    "user_safe_detail": stop.detail,
                    "shadow_only": True,
                },
                "incident_id",
            )
        )
    if health is GuardianHealth.LIVENESS_STALE:
        incidents.append(
            _with_hash(
                RuntimeIncident,
                {
                    "incident_code": "TASK_LIVENESS_NOT_RECENT",
                    "detected_at": observed_at,
                    "stage_id": projection.current_stage,
                    "user_safe_detail": liveness.note,
                    "shadow_only": True,
                },
                "incident_id",
            )
        )
    elif (
        health in {GuardianHealth.TERMINAL_BLOCKED, GuardianHealth.TERMINAL_DEFERRED}
        and not incidents
    ):
        incidents.append(
            _with_hash(
                RuntimeIncident,
                {
                    "incident_code": f"TASK_{lifecycle.value}",
                    "detected_at": observed_at,
                    "stage_id": projection.current_stage,
                    "user_safe_detail": (
                        f"The Task is {lifecycle.value} and its owner recorded no code."
                    ),
                    "shadow_only": True,
                },
                "incident_id",
            )
        )
    return signal, tuple(incidents)


def build_task_recovery_view(
    *,
    snapshot: TaskBoardSnapshot,
    running: bool,
    worker_failure: str | None,
    heartbeats: TaskHeartbeatReadout,
    recoverable_kinds: Iterable[str],
    observed_at: datetime,
    replans: Mapping[str, TaskReplan],
    blocked_retry_reason: str | None = None,
    resume_refusal: str | None = None,
    replan_refusal: str | None = None,
    attention: TaskAttentionFact | None = None,
) -> TaskRecoveryView:
    """Project one exact Task Control snapshot with separate dispatcher and liveness facts.

    Describe one Task version from Task Control's one-lock snapshot (`registry.board`),
    the dispatcher's two facts and what the runners' heartbeat sidecars hold for the
    snapshot's execution (read beside the snapshot through `TaskHeartbeatReader`: the
    rows under its id and the sidecars that could not be read). Each is kept apart from
    the durable snapshot. `replans` holds each Task kind's re-PLAN as the owner that admits
    it declares it; a kind no owner declares has none.
    """
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("task recovery view clock must be timezone-aware")
    task, items, execution, projection = (
        snapshot.task,
        snapshot.work_items,
        snapshot.execution,
        snapshot.projection,
    )
    if projection.task_record_hash != task.record_hash:
        raise ValueError("task_control.board_snapshot_inconsistent")
    lifecycle = projection.lifecycle
    stop: StopFact | None = None
    if task.failure_code:
        stopped = next(
            (
                item
                for item in items
                if item.lifecycle in {WorkItemLifecycle.BLOCKED, WorkItemLifecycle.CANCELLED}
            ),
            None,
        )
        stop = StopFact(
            code=task.failure_code,
            stage_id=stopped.stage_id if stopped is not None else task.active_work_item_id,
            detail=stop_detail(task.task_kind, task.failure_code, "TASK_CONTROL"),
            recoverable=lifecycle is TaskLifecycle.RECOVERY_REQUIRED,
            cause=stopped.failure_cause if stopped is not None else None,
        )
    worker: WorkerStop | None = None
    if worker_failure and lifecycle not in _TERMINAL:
        code, failure_type = _worker_stop(worker_failure)
        worker = WorkerStop(
            code=code,
            failure_type=failure_type,
            detail=stop_detail(task.task_kind, code, "WORKER"),
        )
    liveness = _liveness(projection, execution, heartbeats, observed_at)
    actions = _actions(
        task,
        projection,
        frozenset(recoverable_kinds),
        running,
        replans,
        blocked_retry_reason,
        resume_refusal,
        replan_refusal,
    )
    health, incidents = _health_and_incidents(task, projection, liveness, stop, observed_at)
    values = {
        "task_id": str(task.task_id),
        "task_kind": task.task_kind,
        "lifecycle": lifecycle.value,
        "operation_running": running,
        "cancellation": (
            "REQUESTED"
            if lifecycle is TaskLifecycle.CANCEL_REQUESTED
            else "ACKNOWLEDGED"
            if lifecycle is TaskLifecycle.CANCELLED
            else "NOT_REQUESTED"
        ),
        "stop": stop,
        "worker_failure": worker,
        "resume_refusal": resume_refusal,
        "liveness": liveness,
        "execution_binding_hash": (
            execution.compatibility.compatibility_hash if execution is not None else None
        ),
        "verified_stage_count": projection.verified_stage_count,
        "total_stage_count": projection.total_stage_count,
        "stages": tuple(
            PreservedStage(
                stage_id=item.stage_id,
                lifecycle=item.lifecycle.value,
                evidence=tuple(evidence.reference for evidence in item.evidence),
                evidence_count=len(item.evidence),
            )
            for item in items
        ),
        "artifact_refs": projection.artifact_refs,
        "actions": actions,
        "health": health,
        "incidents": incidents,
        "model_facts": "NOT_OBSERVED",
        "guardian": None,
        "guardian_availability": "NOT_PUBLISHED",
        "task_record_hash": task.record_hash,
        "observed_at": observed_at,
        "attention": attention,
    }
    return _with_hash(TaskRecoveryView, values, "view_hash")


RecoverableKinds = Callable[[], frozenset[str]]

__all__ = [
    "STALE_HEARTBEAT_SECONDS",
    "LivenessFact",
    "PermittedAction",
    "PreservedStage",
    "RecoverableKinds",
    "StopFact",
    "TaskAttentionFact",
    "TaskRecoveryView",
    "WorkerStop",
    "build_task_recovery_view",
    "stop_detail",
    "task_attention",
]
