"""Workspace activity: safe observations of product operations, composed once.

The Host already owns three truths about work: the shared operation owner sees
every request and its answer, the dispatcher knows when a command has returned,
and Task Control and the artifact owners hold lifecycle and results. This
module composes the existing Unified Observation Ledger under the workspace so
those facts are *recorded* as they happen and can be *read back* in bounded
pages, without becoming a second store, a broker or a new authority.

What is recorded, and at which trust level, is decided here and nowhere else:

- `ProductOperationObserved` (OPERATIONAL_ASSERTION): the entry saw an operation
  requested, returned or fail, with the caller class the entry itself supplied.
- `TaskControlTransition` (TASK_CONTROL_ASSERTION): the registry's projection
  once the dispatcher's command returned; a settled lifecycle is only stated
  when the command is over.
- `ArtifactVerificationObserved` (ARTIFACT_ASSERTION): a result reference the
  owner *read back* after that return -- the sealed artifact itself was opened,
  not merely indexed. This is the only observation a reader may render as
  "result available".
- `ExternalActivityObserved` (AGENT_PROPOSAL): what an admitted external client
  declared about its own work. The boundary assigns the level; the client
  cannot.

Activity is optional infrastructure. A store that cannot be opened leaves the
product usable and the observer visibly UNAVAILABLE; a callback that fails
leaves the operation's result and the Task's lifecycle exactly as they were and
counts one missing observation that no later success conceals. Only typed
failure codes and exception class names are ever persisted: no exception
message, no worker error text, no owner refusal prose.
"""

from __future__ import annotations

import functools
import re
import subprocess
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

from alphalattice.control.observation_runtime.adapters import safe_observation_draft
from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthority,
    ObservationAvailability,
    ObservationRetentionClass,
)
from alphalattice.control.observation_runtime.ledger import (
    ObservationLedger,
    ObservationStorageError,
    UnifiedObservationPort,
)
from alphalattice.control.observation_runtime.policy import (
    EXTERNAL_ACTIVITY_SCHEMA,
    PRODUCT_OPERATION_SCHEMA,
    default_observation_policies,
)
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    ArtifactReference,
    PortfolioResearchOperations,
    task_status_body,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application.activity import (
    ActivityReadQuery,
    ExternalActivityEventDocument,
    ExternalActivityReadQuery,
    OperationSpan,
    activity_item,
    next_read,
    observed_operation,
    request_field_names,
    request_subject,
    returned_status,
    returned_subject,
    returned_task_id,
)
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    WAIT_EXITS,
    outcome_of,
    refusal_words,
)
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.interface.local_application.failure_codes import (
    FAILURE_DETAIL_WITHHELD,
    owner_failure_code,
    safe_failure_code,
)
from alphalattice.interface.local_application.portfolio_research import (
    OperationCaller,
    PortfolioResearchOperationRequest,
)

OBSERVATION_LEDGER_NAME = "observations.sqlite"
"""Under `<workspace>/runtime`, the same store the Desk publishers use."""

OPERATION_SOURCE_KIND = "PRODUCT_OPERATION"
TASK_SOURCE_KIND = "TASK_CONTROL"
ARTIFACT_SOURCE_KIND = "PRODUCT_ARTIFACT"
EXTERNAL_SOURCE_KIND = "EXTERNAL_CLIENT"
_ENUM_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,79}")
_REFERENCE_VALUE = re.compile(r"^[A-Za-z0-9_.:,\-/]{1,128}$")


def safe_enum_token(value: object) -> str | None:
    """Admit only bounded enum-token strings for activity persistence.

    Args:
        value: Explicit candidate operation status.

    Returns:
        Safe enum token or None; arbitrary object text is not persisted.
    """
    text = str(value) if isinstance(value, str) else None
    return text if text is not None and _ENUM_TOKEN.match(text) else None


def failure_class_name(text: object) -> str | None:
    """The exception class a worker failure was reported under, never its message."""
    if not isinstance(text, str):
        return None
    match = _CLASS_NAME.match(text)
    return match.group(0) if match else None


def safe_references(values: Mapping[str, Any]) -> dict[str, Any]:
    """Keep hashes, ids and enumeration tokens; drop anything shaped like prose."""

    def admitted(value: object) -> bool:
        if isinstance(value, bool | int):
            return True
        if isinstance(value, str):
            return _REFERENCE_VALUE.match(value) is not None
        return isinstance(value, list) and all(
            isinstance(item, str) and _REFERENCE_VALUE.match(item) for item in value
        )

    return {key: value for key, value in values.items() if admitted(value)}


def task_run_id(task_id: UUID | str) -> str:
    """The run under which this Host's observations of one Task are correlated."""
    return f"local-web:{task_id}"


ArtifactResolver = Callable[[str, UUID], ArtifactReference | None]
"""Given a command kind and a Task, open the owner's published result and answer
with its reference, or `None` when that kind publishes nothing this observer
names. Opening is the verification: an index entry alone must not answer."""


@dataclass
class WorkspaceActivity:
    """One recorder and one bounded reader over the workspace ledger."""

    workspace: Path
    workspace_id: str
    gate: WorkspaceMutationGate
    instance: str
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    storage_cap_reader: Callable[[], int] | None = None
    native_usage_state: Callable[[], dict[str, object]] | None = None
    """Optional Host-owned usage health read; only its safe status and reason are exposed."""
    store_failure: str | None = field(default=None, init=False)
    """The typed code of the storage failure that left this observer UNAVAILABLE."""

    _ledger: ObservationLedger | None = field(default=None, init=False, repr=False)
    _port: UnifiedObservationPort | None = field(default=None, init=False, repr=False)
    _sequences: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _dispatcher: LocalBackgroundDispatcher | None = field(default=None, init=False, repr=False)
    _registry: DuckDbTaskControlRegistry | None = field(default=None, init=False, repr=False)
    _wake_sender: ThreadPoolExecutor | None = field(default=None, init=False, repr=False)
    _wake_capability: dict[str, object] | None = field(default=None, init=False, repr=False)
    _operations: PortfolioResearchOperations | None = field(default=None, init=False, repr=False)
    _artifacts: ArtifactResolver | None = field(default=None, init=False, repr=False)
    _missing: int = field(default=0, init=False)
    _appends: int = field(default=0, init=False)
    _recording_failed: bool = field(default=False, init=False)
    _first_failure_at: str | None = field(default=None, init=False)
    _last_failure_code: str | None = field(default=None, init=False)
    _last_failure_type: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Open the optional observation ledger and its policy-bound event port.

        A storage refusal is retained as a visible failure code; recording is not silently retried
        and operation execution remains separate.
        """
        try:
            ledger = ObservationLedger(
                self.workspace / "runtime" / OBSERVATION_LEDGER_NAME,
                gate=self.gate,
                storage_cap_reader=self.storage_cap_reader,
            )
        except ObservationStorageError as error:
            # Optional infrastructure: the product stays usable and every read
            # says why nothing is being recorded. Nothing is retried silently.
            self.store_failure = error.failure_code
            self._note_failure(error.failure_code, type(error).__name__)
            return
        self._ledger = ledger
        self._port = UnifiedObservationPort(
            ledger=ledger, policies=default_observation_policies(), clock=self.clock
        )

    def attach(
        self,
        *,
        dispatcher: LocalBackgroundDispatcher,
        registry: DuckDbTaskControlRegistry,
        artifacts: ArtifactResolver,
        operations: PortfolioResearchOperations | None = None,
    ) -> None:
        """Bind the owners read at command return; installs the dispatcher hook.

        Each Codex wake a stopped Host left mid-send is named uncertain here, never sent again.
        """
        self._dispatcher = dispatcher
        self._registry = registry
        self._artifacts = artifacts
        self._operations = operations
        dispatcher.on_command_returned = self.command_returned
        self._wake_sender = ThreadPoolExecutor(max_workers=1, thread_name_prefix="codex-wake")
        for held in registry.interrupt_wakes(observed_at=self.clock()):
            self._isolated(functools.partial(self._record_wake, held))

    def replay_wakes(self) -> None:
        """Send every pending wake whose Task reached its event while no hook saw it.

        The start's replay, once the Host has resumed its Tasks.
        """
        self._wake_changed(None)

    def close(self) -> None:
        """Finish the wake being sent, then close the optional observation ledger."""
        sender, self._wake_sender = self._wake_sender, None
        if sender is not None:
            sender.shutdown(wait=True)
        if self._ledger is not None:
            self._ledger.close()

    # ------------------------------------------------------------ Codex wakes

    def _wake_changed(self, task_id: UUID | str | None) -> None:
        """A Task may have reached a registered wake's event: send it on the wake thread."""
        if self._wake_sender is not None:
            self._wake_sender.submit(self._isolated, lambda: self._send_wakes(task_id))

    def drain_wakes(self) -> None:
        """Wait for the wakes already handed to the wake thread (tests)."""
        if self._wake_sender is not None:
            self._wake_sender.submit(lambda: None).result(timeout=120)

    def _send_wakes(self, task_id: UUID | str | None) -> None:
        """Send each pending wake whose Task needs its lead now, once (WAKE, R1).

        A Codex turn ends its shell's children with it, so the Host, which outlives turns,
        queues the one line. The attempt is committed before the call; a failure is named in the
        Task's activity and never retried, and the lead's next read of the Task shows its state.
        """
        registry, dispatcher = self._registry, self._dispatcher
        if registry is None or dispatcher is None:
            return
        for held in registry.wake_registrations(None if task_id is None else UUID(str(task_id))):
            if held["state"] != "PENDING":
                continue
            task = UUID(held["task_id"])
            status = dispatcher.status(task)
            lifecycle = status.lifecycle.value
            event = WAIT_EXITS.get(lifecycle)
            if event is None and outcome_of({"lifecycle": lifecycle}) != "PENDING":
                event = "ENDED"
            claimed = (
                None
                if event is None
                else registry.claim_wake(
                    held["registration_id"],
                    task,
                    event=event,
                    lifecycle=lifecycle,
                    observed_at=self.clock(),
                )
            )
            if claimed is None or event is None:
                continue
            result = _queue_wake(
                claimed["thread_id"],
                _wake_line(event, status.task_kind, lifecycle, status.latest_failure_code)
                + f" Read and verify it with {claimed['read_command']}",
                claimed.get("codex_path"),
            )
            registry.finish_wake(held["registration_id"], task, result, observed_at=self.clock())
            self._record_wake({**claimed, "result": result})

    def _record_wake(self, held: Mapping[str, Any]) -> None:
        """Name a wake's outcome in its Task's activity: queued, or its failure."""
        task_id = str(held["task_id"])
        result = held["result"]
        self._append(
            schema_kind="TaskControlTransition",
            source_kind=TASK_SOURCE_KIND,
            source_id=self.source_id,
            authority=ObservationAuthority.TASK_CONTROL_ASSERTION,
            task_id=task_id,
            run_id=task_run_id(task_id),
            correlation_ids=(task_id,),
            payload={
                "task_lifecycle": held["lifecycle"],
                "disposition": "WAKE_DELIVERED" if result["delivered"] else "WAKE_UNDELIVERED",
                **({} if result["delivered"] else {"failure_code": result["failure"]}),
            },
        )

    @property
    def available(self) -> bool:
        """Read whether this activity ledger was successfully opened.

        Returns:
            Whether activity storage is available.
        """
        return self._ledger is not None

    @property
    def source_id(self) -> str:
        """Name this exact Local Web service instance as an activity source.

        Returns:
            Instance-bound local-web source identifier.
        """
        return f"local-web:{self.instance}"

    @property
    def store_epoch(self) -> str | None:
        """Read the optional activity store epoch.

        Returns:
            Exact ledger epoch string or None when storage is unavailable.
        """
        return None if self._ledger is None else str(self._ledger.store_epoch)

    # ----------------------------------------------------- operation observer

    def entered(
        self, request: PortfolioResearchOperationRequest, *, caller: OperationCaller
    ) -> OperationSpan | None:
        """Record admitted operation entry using bounded safe references and field names.

        Agent session and goal references are provenance alongside the operation and grant no
        identity or execution authority.

        Args:
            request: Validated operation declaration.
            caller: Declared operation caller.

        Returns:
            Operation span for observed operations, otherwise None; recording failures are isolated.
        """
        if not observed_operation(request.operation):
            return None
        span = OperationSpan(
            operation=request.operation,
            caller=caller,
            operation_ref=uuid4().hex,
            subject={},
            request_fields=[],
            started=perf_counter(),
        )
        # The agent session and goal the request's client named, read in the request's own
        # context: provenance beside what the request did, never an identity (OP13, ID6).
        provenance = REQUEST_PROVENANCE.get()

        def record() -> None:
            span.subject = safe_references(
                {
                    **request_subject(request),
                    **(provenance.subject() if provenance is not None else {}),
                }
            )
            span.request_fields = request_field_names(request)
            self._record_operation(span, phase="REQUESTED")

        self._isolated(record)
        return span

    def returned(self, span: OperationSpan, body: Mapping[str, Any]) -> None:
        """Record an operation return with safe status, references and next-read metadata.

        Recording failures are isolated from the operation result.

        Args:
            span: Exact entered operation span.
            body: Owner-supplied operation response.
        """

        def record() -> None:
            subject = {**span.subject, **safe_references(returned_subject(body))}
            conversation = body.get("conversation")
            receipt = body.get("receipt")
            if (
                span.operation == "AGENT_ANSWER_SUBMIT"
                and body.get("status") in {"ACCEPTED", "DONE"}
                and isinstance(conversation, dict)
                and conversation.get("status") == "DELIVERED"
                and isinstance(receipt, dict)
                and conversation.get("task_id") == receipt.get("task_id") == body.get("task_id")
                and conversation.get("answer_reference")
                == receipt.get("answer_reference")
                == body.get("answer_reference")
                and conversation.get("bundle_reference") == body.get("bundle_reference")
                and isinstance(conversation.get("observation_id"), str)
                and re.fullmatch(r"[0-9a-f]{64}", conversation["observation_id"]) is not None
                and conversation.get("original_session_host") in {"codex", "claude-code"}
                and isinstance(conversation.get("original_session_id"), str)
                and _REFERENCE_VALUE.fullmatch(conversation["original_session_id"]) is not None
                and "original_goal_id" in conversation
            ):
                original_goal = conversation["original_goal_id"]
                if original_goal is not None and str(UUID(str(original_goal))) != original_goal:
                    raise ValueError("activity.accepted_answer_binding_mismatch")
                subject.update(
                    agent_vendor=conversation["original_session_host"],
                    agent_session=conversation["original_session_id"],
                    goal_id=original_goal,
                    conversation_original_goal_id=original_goal,
                )
                # Exact retries reuse the first actual return correlation. They remain
                # ordinary recorded returns without adding a second purported first receipt.
                linked = (
                    None
                    if self._ledger is None
                    else self._ledger.unique_correlated_observation(
                        task_run_id(str(body["task_id"])),
                        PRODUCT_OPERATION_SCHEMA,
                        conversation["observation_id"],
                    )
                )
                if linked is None:
                    subject["conversation_observation_id"] = conversation["observation_id"]
            self._record_operation(
                span,
                phase="RETURNED",
                status=safe_enum_token(returned_status(body)),
                failure_code=_persisted_code(body.get("failure_code")),
                task_id=returned_task_id(body),
                lifecycle=safe_enum_token(body.get("lifecycle")),
                subject=subject,
                read=next_read(span.operation, body),
            )

        self._isolated(record)
        # A registration, a cancel or a decision may have reached a wake's event.
        task_id = returned_task_id(body)
        if task_id is not None:
            self._wake_changed(task_id)

    def failed(self, span: OperationSpan, error: BaseException) -> None:
        """Record a bounded safe operation failure while withholding arbitrary detail.

        The shared boundary reads an owner's admitted code independently of its exception base;
        arbitrary failure detail is withheld and recording failures are isolated.

        Args:
            span: Exact entered operation span.
            error: Explicit operation failure.
        """

        def record() -> None:
            code = owner_failure_code(error)
            self._record_operation(
                span,
                phase="FAILED",
                status="FAILED",
                failure_code=code or FAILURE_DETAIL_WITHHELD,
                failure_type=type(error).__name__,
            )

        self._isolated(record)

    def refused(self, operation: str, failure_code: str, *, caller: str) -> None:
        """Count one refusal, whatever the operation, reads included (AC, OP14).

        The count keeps the operation, the code, the caller kind and the agent vendor the
        request's session named: never the session, the request or any text, so it outlives
        the transient rows as the trace of where requests fail.

        Args:
            operation: The operation refused, or `UNKNOWN` when the request named none.
            failure_code: The typed code the refusal carried.
            caller: The caller kind the entry boundary supplied.
        """
        code = _persisted_code(failure_code)
        ledger = self._ledger
        if code is None or ledger is None:
            return
        provenance = REQUEST_PROVENANCE.get()
        self._isolated(
            lambda: ledger.count_refusal(
                operation=operation,
                failure_code=code,
                caller=caller,
                vendor=None if provenance is None else provenance.vendor,
                refused_at=self.clock(),
            )
        )

    def refusals(self, *, days: int) -> dict[str, object]:
        """Read the refusals counted over the last `days` days, the most frequent first.

        Args:
            days: How many days back, today included.

        Returns:
            The `ACTIVITY_REFUSALS` answer: the first day read, each counted row and the total,
            or the store's refusal when the observer is unavailable.
        """
        if self._ledger is None:
            return _refused("activity.storage_unavailable", "RESTART_HOST_AFTER_STORE_REPAIR")
        since = (self.clock().astimezone(UTC) - timedelta(days=days - 1)).date().isoformat()
        rows = self._ledger.refusal_counts(since_day=since)
        return {
            "status": "ACTIVITY_REFUSALS",
            "since_day": since,
            "days": days,
            "refusals": list(rows),
            "total": sum(int(cast(int, row["count"])) for row in rows),
        }

    def _record_operation(
        self,
        span: OperationSpan,
        *,
        phase: str,
        status: str | None = None,
        failure_code: str | None = None,
        failure_type: str | None = None,
        task_id: str | None = None,
        lifecycle: str | None = None,
        subject: dict[str, Any] | None = None,
        read: dict[str, str] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "operation": span.operation,
            "caller": span.caller,
            "phase": phase,
            "operation_ref": span.operation_ref,
            "subject": span.subject if subject is None else subject,
            "request_fields": span.request_fields,
        }
        if phase != "REQUESTED":
            payload["latency_milliseconds"] = int((perf_counter() - span.started) * 1000)
            payload["status"] = status
        if failure_code is not None:
            payload["failure_code"] = failure_code
        if failure_type is not None:
            payload["failure_type"] = failure_type
        if task_id is not None:
            payload["task_id"] = task_id
        if lifecycle is not None:
            payload["task_lifecycle"] = lifecycle
        if read is not None:
            payload["next_read"] = read
        correlations = {span.operation_ref}
        if subject is not None and isinstance(subject.get("conversation_observation_id"), str):
            correlations.add(subject["conversation_observation_id"])
        run_id = None
        if task_id is not None:
            run_id = task_run_id(task_id)
            correlations.update({task_id, run_id})
        self._append(
            schema_kind=PRODUCT_OPERATION_SCHEMA,
            source_kind=OPERATION_SOURCE_KIND,
            source_id=self.source_id,
            payload=payload,
            authority=ObservationAuthority.OPERATIONAL_ASSERTION,
            task_id=task_id,
            run_id=run_id,
            correlation_ids=tuple(sorted(correlations)),
        )

    # -------------------------------------------------------- command return

    def command_returned(self, command_kind: str, task_id: UUID, failure: str | None) -> None:
        """Record what Task Control and the artifact owner say now that the command is over.

        Called on the dispatcher's worker after the running set was updated, so
        the projection read here is the settled one. The worker failure text is
        reduced to its exception class; owner reads are typed and bounded; an
        owner that cannot answer degrades this observer, never the Task.
        """
        self._isolated(lambda: self._record_command_return(command_kind, task_id, failure))
        self._wake_changed(task_id)

    def _record_command_return(self, command_kind: str, task_id: UUID, failure: str | None) -> None:
        dispatcher = self._dispatcher
        if dispatcher is None or self._registry is None:
            return
        projection = dispatcher.status(task_id)
        run_id = task_run_id(task_id)
        payload: dict[str, Any] = {
            "task_lifecycle": projection.lifecycle.value,
            "task_class": command_kind,
            "disposition": "COMMAND_RETURNED",
            "stage_id": projection.current_stage,
            "verified_prefix_count": projection.verified_stage_count,
            "total_units": projection.total_stage_count,
            "projection_hash": projection.projection_hash,
            "last_verified_result_ref": (
                projection.artifact_refs[-1] if projection.artifact_refs else None
            ),
        }
        registry_code = _persisted_code(projection.latest_failure_code)
        if registry_code is not None:
            payload["failure_code"] = registry_code
        worker_failure = failure_class_name(failure)
        if worker_failure is not None:
            payload["failure_type"] = worker_failure
        self._append(
            schema_kind="TaskControlTransition",
            source_kind=TASK_SOURCE_KIND,
            source_id=self.source_id,
            payload=payload,
            authority=ObservationAuthority.TASK_CONTROL_ASSERTION,
            task_id=str(task_id),
            run_id=run_id,
            stage_id=projection.current_stage,
            correlation_ids=(run_id, str(task_id)),
        )
        if projection.lifecycle is not TaskLifecycle.SUCCEEDED or self._artifacts is None:
            return
        try:
            reference = self._artifacts(command_kind, task_id)
        except (ValueError, KeyError, OSError) as error:
            self._note_failure("activity.artifact_readback_failed", type(error).__name__)
            return
        if reference is None:
            return
        self._append(
            schema_kind="ArtifactVerificationObserved",
            source_kind=ARTIFACT_SOURCE_KIND,
            source_id=self.source_id,
            payload={
                "artifact_kind": reference.artifact_kind,
                "artifact_hash": reference.artifact_hash,
                "result_ref_hash": reference.artifact_hash,
                "availability": "AVAILABLE",
                "verified_prefix_count": projection.verified_stage_count,
            },
            authority=ObservationAuthority.ARTIFACT_ASSERTION,
            task_id=str(task_id),
            run_id=run_id,
            stage_id=projection.current_stage,
            correlation_ids=(run_id, str(task_id)),
        )

    # ------------------------------------------------------- external events

    def admit_external_event(self, document: ExternalActivityEventDocument) -> dict[str, object]:
        """Record one declared event at the external-client trust level.

        Identity is the producer's own `(producer_id, producer_session,
        producer_sequence)`: a replay of the same event reuses the row, a reused
        sequence with different content is refused by the ledger, and the
        producer restarts its sequence by opening a new session.
        """
        port = self._port
        if port is None:
            return _refused("activity.storage_unavailable", "RESTART_HOST_AFTER_STORE_REPAIR")
        registry = self._registry
        subject = dict(document.subject)
        task_id = subject.get("task_id")
        verified = False
        if task_id is not None:
            if registry is None:
                return _refused("activity.event_task_unverifiable", "START_HOST_BEFORE_EVENTS")
            try:
                registry.task(UUID(task_id))
            except KeyError:
                return _refused("activity.event_task_unknown", "NAME_AN_EXISTING_TASK_OR_OMIT_IT")
            verified = True
        summary, truncated = document.retained_summary()
        source_id = f"{document.producer_id}:{document.producer_session}"
        try:
            draft = safe_observation_draft(
                policies=port.policies,
                schema_kind=EXTERNAL_ACTIVITY_SCHEMA,
                occurred_at=document.occurred_at,
                source_kind=EXTERNAL_SOURCE_KIND,
                source_id=source_id,
                source_sequence=document.producer_sequence,
                payload={
                    "event_kind": document.event_kind,
                    "producer_id": document.producer_id,
                    "producer_session": document.producer_session,
                    "producer_sequence": document.producer_sequence,
                    "summary": summary,
                    "summary_truncated": truncated,
                    "subject": subject,
                    "task_verified": verified,
                },
                authority=ObservationAuthority.AGENT_PROPOSAL,
                retention_class=ObservationRetentionClass.TRANSIENT_OPERATIONAL,
                task_id=task_id if verified else None,
                correlation_ids=tuple(
                    sorted({*document.correlation_ids, *([str(task_id)] if verified else [])})
                ),
            )
            result = port.emit(draft)
        except ValueError as error:
            code = safe_failure_code(str(error)) or "activity.event_not_admitted"
            return _refused(code, "READ_ACTIVITY_EVENT_CONTRACT")
        except ObservationStorageError as error:
            self._note_failure(error.failure_code, type(error).__name__)
            return _refused(error.failure_code, "RETRY_SAME_SEQUENCE_LATER")
        self._note_append()
        return {
            "status": result.disposition,
            "observation_id": result.observation_id,
            "source_id": source_id,
            "source_sequence": result.source_sequence,
            "authority": ObservationAuthority.AGENT_PROPOSAL.value,
            "task_verified": verified,
            "summary_truncated": truncated,
            "cursor": self._cursor_at_head(),
        }

    # --------------------------------------------------------------- reads

    def read(self, query: ActivityReadQuery) -> dict[str, object]:
        """One bounded page plus the current projection of every Task it names.

        The Task join reads only the projections this page references (plus the
        watched ids), never the whole projection table; a Task admitted moments
        ago that has no row yet is read from the registry directly.
        """
        started = perf_counter()
        ledger = self._ledger
        page = None if ledger is None else ledger.observations_after(query.after, limit=query.limit)
        referenced = {str(value) for value in query.watch}
        items = [] if page is None else [activity_item(item) for item in page.items]
        if page is not None:
            referenced.update(
                item.envelope.task_id for item in page.items if item.envelope.task_id is not None
            )
        tasks = self._task_projections(referenced)
        return {
            "workspace_id": self.workspace_id,
            "disposition": "UNAVAILABLE" if page is None else page.disposition,
            "epoch": None if page is None else page.store_epoch,
            "cursor": None if page is None else page.next_cursor.encode(),
            "head": 0 if page is None else page.head_ordinal,
            "more": False if page is None else page.more,
            "items": items,
            "unavailable": 0 if page is None else page.unavailable_count,
            "tasks": tasks,
            "observer": self.observer_state(),
            "read_cost": {
                "observations": len(items),
                "projections": sum(
                    1
                    for value in tasks.values()
                    if isinstance(value, dict) and "lifecycle" in value
                ),
                "elapsed_ms": int((perf_counter() - started) * 1000),
            },
        }

    def recent(self, *, limit: int = 20) -> dict[str, object]:
        """The newest requests, grouped by agent session and goal, each with its read (OP13).

        One read finds what just happened, whichever session asks: the newest returned
        operations with the agent session and goal their rows recorded, and the read that
        opens each. The feed is transient; a goal keeps its own record.

        Args:
            limit: How many groups, from 1 to 50.

        Returns:
            The groups, newest first, each with up to five of its newest requests.
        """
        if not 1 <= limit <= 50:
            raise ValueError("activity.recent_limit_invalid")
        ledger = self._ledger
        if ledger is None:
            return {"status": "UNAVAILABLE", "groups": [], "observer": self.observer_state()}
        page = ledger.observations_of_kind(PRODUCT_OPERATION_SCHEMA, limit=500)
        groups: dict[tuple[str, str, str], dict[str, object]] = {}
        for item in reversed(page.items):
            row = activity_item(item)
            payload = row["payload"] or {}
            if payload.get("phase") != "RETURNED":
                continue
            subject = payload.get("subject") or {}
            key = (
                str(subject.get("agent_vendor", "")),
                str(subject.get("agent_session", "")),
                str(subject.get("goal_id", "")),
            )
            if key not in groups:
                if len(groups) == limit:
                    continue
                groups[key] = {
                    "agent": {"vendor": key[0], "session_id": key[1]} if key[1] else None,
                    "goal_id": key[2] or None,
                    "latest_at": row["occurred_at"],
                    "items": [],
                }
            items = groups[key]["items"]
            assert isinstance(items, list)
            if len(items) < 5:
                items.append(
                    {
                        "at": row["occurred_at"],
                        "operation": payload.get("operation"),
                        "status": payload.get("status"),
                        "task_id": payload.get("task_id"),
                        "read": payload.get("next_read"),
                    }
                )
        return {
            "status": "AVAILABLE",
            "groups": list(groups.values()),
            "claim": "The newest recorded requests by agent session and goal; each read opens "
            "the result. A goal's own record keeps what the transient feed may drop.",
        }

    def read_external(self, query: ExternalActivityReadQuery) -> dict[str, object]:
        """Read bounded external-client events for the explicitly declared sessions.

        The declared sessions' readback: the newest external-client events the
        store holds, whatever the product recorded between them.

        The feed's tail is the wrong owner for a conversation: two hundred of the
        product's own operations push a team's exchanges out of it. This read
        walks one kind in commit order; `before` pages further back. No Task
        join: a declared event names its Task as a verified flag, not a state.
        """
        if query.observation_id is not None:
            return self._accepted_answer_detail(query.observation_id)
        started = perf_counter()
        ledger = self._ledger
        page = (
            None
            if ledger is None
            else ledger.observations_of_kind(
                EXTERNAL_ACTIVITY_SCHEMA, limit=query.limit, before=query.before
            )
        )
        items = [] if page is None else [activity_item(item) for item in page.items]
        return {
            "workspace_id": self.workspace_id,
            "disposition": "UNAVAILABLE" if page is None else page.disposition,
            "epoch": None if page is None else page.store_epoch,
            "head": 0 if page is None else page.head_ordinal,
            "oldest": items[0]["ordinal"] if items else None,
            "more": False if page is None else page.more,
            "items": items,
            "unavailable": 0 if page is None else page.unavailable_count,
            "observer": self.observer_state(),
            "read_cost": {
                "observations": len(items),
                "elapsed_ms": int((perf_counter() - started) * 1000),
            },
        }

    def _accepted_answer_detail(self, observation_id: str) -> dict[str, object]:
        """Open one selected native event or owner return and verify its sealed answer."""
        started = perf_counter()
        ledger, operations = self._ledger, self._operations
        answer: dict[str, object] = {
            "status": "UNAVAILABLE",
            "reason": "activity.accepted_answer_unavailable",
            "observation_id": observation_id,
        }
        observations = 0
        if ledger is not None:
            try:
                envelope = ledger.read(observation_id)
            except (ObservationStorageError, ValueError):
                pass
            else:
                observations = 1
                payload = envelope.inline_safe_payload or {}
                subject = payload.get("subject")
                if (
                    envelope.schema_kind == EXTERNAL_ACTIVITY_SCHEMA
                    and envelope.availability is ObservationAvailability.AVAILABLE
                    and payload.get("event_kind") == "NATIVE_COORDINATION_MESSAGE"
                    and isinstance(subject, dict)
                    and subject.get("message_kind") == "answer"
                    and subject.get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
                    and operations is not None
                    and operations.review is not None
                ):
                    if subject.get("authorship_basis") == "NOT_OBSERVED":
                        answer, checked = self._product_accepted_event_detail(
                            observation_id, subject
                        )
                        observations += checked
                    else:
                        answer = {
                            **operations.review.read_accepted_answer(subject),
                            "observation_id": observation_id,
                        }
                elif (
                    envelope.schema_kind == PRODUCT_OPERATION_SCHEMA
                    and envelope.schema_version == 1
                    and envelope.source_kind == OPERATION_SOURCE_KIND
                    and (
                        envelope.source_id == self.source_id
                        or re.fullmatch(r"local-web:[0-9a-f]{32}", envelope.source_id) is not None
                    )
                    and envelope.authority is ObservationAuthority.OPERATIONAL_ASSERTION
                    and envelope.availability is ObservationAvailability.AVAILABLE
                    and payload.get("operation") == "AGENT_ANSWER_SUBMIT"
                    and payload.get("phase") == "RETURNED"
                    and payload.get("status") in {"ACCEPTED", "DONE"}
                    and isinstance(subject, dict)
                    and isinstance(subject.get("task_id"), str)
                    and envelope.task_id == payload.get("task_id") == subject["task_id"]
                    and envelope.run_id == task_run_id(subject["task_id"])
                    and operations is not None
                    and operations.review is not None
                ):
                    answer = {
                        **operations.review.read_product_accepted_answer(
                            subject, verdict=payload["status"]
                        ),
                        "observation_id": observation_id,
                        "source_kind": envelope.source_kind,
                        "source_id": envelope.source_id,
                        "authority": envelope.authority.value,
                    }
        return {
            "workspace_id": self.workspace_id,
            "observation_id": observation_id,
            "epoch": None if ledger is None else ledger.store_epoch,
            "accepted_answer": answer,
            "observer": self.observer_state(),
            "read_cost": {
                "observations": observations,
                "elapsed_ms": int((perf_counter() - started) * 1000),
            },
        }

    def _product_accepted_event_detail(
        self, observation_id: str, subject: Mapping[str, object]
    ) -> tuple[dict[str, object], int]:
        """Read a default relay only through its actual correlated owner-return envelope."""
        unavailable: dict[str, object] = {
            "status": "UNAVAILABLE",
            "reason": "activity.accepted_answer_unavailable",
            "observation_id": observation_id,
            "missing": ["accepted_operation_return"],
        }
        ledger, operations = self._ledger, self._operations
        if ledger is None or operations is None or operations.review is None:
            return unavailable, 0
        try:
            task = str(UUID(str(subject.get("reference"))))
            if (
                task != subject.get("reference")
                or subject.get("native_agent_id") != subject.get("native_session_id")
                or subject.get("role") != "research_lead"
                or subject.get("submitted_by") != subject.get("native_session_id")
            ):
                return {**unavailable, "reason": "activity.accepted_answer_binding_mismatch"}, 0
            receipt = ledger.unique_correlated_observation(
                task_run_id(task), PRODUCT_OPERATION_SCHEMA, observation_id
            )
        except (ObservationStorageError, ValueError):
            return unavailable, 0
        if receipt is None:
            return unavailable, 0
        payload = receipt.inline_safe_payload or {}
        accepted = payload.get("subject")
        if not (
            receipt.schema_kind == PRODUCT_OPERATION_SCHEMA
            and receipt.schema_version == 1
            and receipt.source_kind == OPERATION_SOURCE_KIND
            and (
                receipt.source_id == self.source_id
                or re.fullmatch(r"local-web:[0-9a-f]{32}", receipt.source_id) is not None
            )
            and receipt.authority is ObservationAuthority.OPERATIONAL_ASSERTION
            and receipt.availability is ObservationAvailability.AVAILABLE
            and observation_id in receipt.correlation_ids
            and receipt.task_id == payload.get("task_id") == task
            and receipt.run_id == task_run_id(task)
            and payload.get("operation") == "AGENT_ANSWER_SUBMIT"
            and payload.get("phase") == "RETURNED"
            and payload.get("status") in {"ACCEPTED", "DONE"}
            and isinstance(accepted, dict)
            and accepted.get("conversation_observation_id") == observation_id
            and accepted.get("task_id") == task
            and accepted.get("answer_reference") == subject.get("answer_reference")
            and accepted.get("bundle_reference") == subject.get("bundle_reference")
            and accepted.get("agent_role") == subject.get("bundle_role")
            and accepted.get("agent_vendor") == subject.get("native_host")
            and accepted.get("agent_session") == subject.get("native_session_id")
            and "conversation_original_goal_id" in accepted
            and accepted["conversation_original_goal_id"]
            == accepted.get("goal_id")
            == subject.get("goal_id")
        ):
            return {**unavailable, "reason": "activity.accepted_answer_binding_mismatch"}, 1
        return {
            **operations.review.read_product_accepted_answer(accepted, verdict=payload["status"]),
            "observation_id": observation_id,
            "receipt_observation_id": receipt.observation_id,
            "source_kind": receipt.source_kind,
            "source_id": receipt.source_id,
            "authority": receipt.authority.value,
        }, 1

    def _task_projections(self, referenced: set[str]) -> dict[str, object]:
        dispatcher, registry = self._dispatcher, self._registry
        if not referenced or dispatcher is None or registry is None:
            return {}
        ids = [UUID(value) for value in sorted(referenced)]
        projection_batch = registry.projection_collection(ids)
        projected = {str(value.task_id): value for value in projection_batch.projections}
        refused = {str(task_id) for task_id in projection_batch.refused_task_ids}
        tasks: dict[str, object] = {}
        for task_id in ids:
            if str(task_id) in refused:
                tasks[str(task_id)] = PortfolioResearchOperations.task_projection_refusal(task_id)
                continue
            projection = projected.get(str(task_id))
            try:
                shown = (
                    dispatcher.status(task_id)
                    if projection is None
                    else dispatcher.masked(projection)
                )
            except KeyError:
                continue  # a watched id this workspace never admitted
            body = task_status_body(shown)
            body["task_record_hash"] = shown.task_record_hash
            body["wakes"] = [
                {**held, **refusal_words(held["result"]["failure"])}
                if not held.get("result", {}).get("delivered", True)
                else held
                for held in registry.wake_registrations(task_id)
            ]
            failure = dispatcher.failure(shown.task_id)
            body["worker_failure"] = failure
            body["worker_failure_type"] = failure_class_name(failure)
            tasks[str(task_id)] = body
        return tasks

    def wake_readiness(self, path: str | None = None) -> dict[str, object]:
        """Share successful queue probes; retry a supplied path after failure."""
        from alphalattice.interface.local_application.native_setup import (
            codex_command,
            codex_queue_readiness,
        )

        with self._lock:
            command, source = codex_command(path)
            if (
                self._wake_capability is None
                or self._wake_capability.get("command") != command
                or (path and not self._wake_capability.get("present"))
            ):
                self._wake_capability = codex_queue_readiness(path)
            return {**self._wake_capability, "command_source": source}

    def observer_state(self) -> dict[str, object]:
        """Whether recording is trustworthy right now and whether anything was lost.

        `missing_observations` never decreases and keeps `status` DEGRADED after
        a later successful append: a gap in the record stays a gap. `recording`
        is the current append health. Neither says anything about execution.
        """
        dispatcher, operations = self._dispatcher, self._operations
        with self._lock:
            missing, appends = self._missing, self._appends
            recording_failed = self._recording_failed
            first_failure_at = self._first_failure_at
            last_code, last_type = self._last_failure_code, self._last_failure_type
        hook_failures = 0 if dispatcher is None else dispatcher.observer_failures
        hook_type = None if dispatcher is None else dispatcher.last_observer_failure
        entry_failures = 0 if operations is None else operations.observer_failures
        entry_type = None if operations is None else operations.last_observer_failure
        if self.store_failure is not None:
            status, recording = "UNAVAILABLE", "UNAVAILABLE"
        elif missing or hook_failures or entry_failures:
            status = "DEGRADED"
            recording = "FAILING" if recording_failed else "OK"
        else:
            status, recording = "OK", "OK"
        state: dict[str, object] = {
            "status": status,
            "recording": recording,
            "appends": appends,
            "missing_observations": missing + hook_failures + entry_failures,
            "first_failure_at": first_failure_at,
            "last_failure_code": last_code,
            "last_failure_type": last_type,
            "hook_failures": hook_failures,
            "hook_failure_type": hook_type,
            "entry_failures": entry_failures,
            "entry_failure_type": entry_type,
            "store_failure": self.store_failure,
            "claim": "OBSERVER_STATE_NOT_EXECUTION_STATE",
            "codex_queue": self.wake_readiness(),
        }
        if self.native_usage_state is not None:
            try:
                usage = self.native_usage_state()
                usage_status = safe_enum_token(usage.get("status"))
                if usage_status not in {"NOT_BOUND", "OFF", "OBSERVING", "STOPPED", "UNAVAILABLE"}:
                    usage_status = "UNAVAILABLE"
                state["native_usage"] = {
                    "status": usage_status,
                    "reason": _persisted_code(usage.get("reason")),
                }
            except Exception:
                state["native_usage"] = {
                    "status": "UNAVAILABLE",
                    "reason": "activity.observer_failed",
                }
            usage_state = cast(dict[str, object], state["native_usage"])
            reason = usage_state.get("reason")
            if isinstance(reason, str):
                words = refusal_words(reason)
                usage_state.update(
                    {key: words[key] for key in ("detail", "next_action") if key in words}
                )
        return state

    # ------------------------------------------------------------- internals

    def _isolated(self, record: Callable[[], None]) -> None:
        """Run one recording step; any failure is counted, typed and contained."""

        try:
            record()
        except Exception as error:
            # Never the operation's answer, never the Task's outcome. Visible on
            # every read as a missing observation with its exception class.
            self._note_failure("activity.observer_failed", type(error).__name__)

    def _append(
        self,
        *,
        schema_kind: str,
        source_kind: str,
        source_id: str,
        payload: dict[str, Any],
        authority: ObservationAuthority,
        task_id: str | None,
        run_id: str | None,
        correlation_ids: tuple[str, ...],
        stage_id: str | None = None,
    ) -> None:
        """Allocate this source's next sequence and append; failure only degrades."""

        port = self._port
        if port is None:
            self._note_failure("activity.storage_unavailable", None)
            return
        with self._lock:
            key = (source_kind, source_id)
            sequence = self._sequences.get(key, -1) + 1
            self._sequences[key] = sequence
        try:
            draft = safe_observation_draft(
                policies=port.policies,
                schema_kind=schema_kind,
                occurred_at=self.clock(),
                source_kind=source_kind,
                source_id=source_id,
                source_sequence=sequence,
                payload=payload,
                authority=authority,
                retention_class=ObservationRetentionClass.TRANSIENT_OPERATIONAL,
                task_id=task_id,
                run_id=run_id,
                stage_id=stage_id,
                correlation_ids=correlation_ids,
            )
            port.emit(draft)
        except ObservationStorageError as error:
            self._note_failure(error.failure_code, type(error).__name__)
            return
        except ValueError as error:
            self._note_failure(
                safe_failure_code(str(error)) or "activity.observation_not_admitted",
                type(error).__name__,
            )
            return
        self._note_append()

    def _note_failure(self, failure_code: str, failure_type: str | None) -> None:
        with self._lock:
            self._missing += 1
            self._recording_failed = True
            self._last_failure_code = failure_code
            self._last_failure_type = failure_type
            if self._first_failure_at is None:
                self._first_failure_at = self.clock().isoformat()

    def _note_append(self) -> None:
        with self._lock:
            self._appends += 1
            self._recording_failed = False

    def _cursor_at_head(self) -> str | None:
        ledger = self._ledger
        if ledger is None:
            return None
        return f"{ledger.store_epoch}:{ledger.head_ordinal()}"


def _wake_line(event: str, kind: str, lifecycle: str, failure: str | None) -> str:
    """What a wake tells its lead first: the Task, what happened in plain words and whose move
    it is, before the command that reads it.

    Args:
        event: The wait's event: ENDED, NEEDS_DECISION or DEFERRED.
        kind: The Task's kind.
        lifecycle: Its lifecycle now.
        failure: Its latest failure code, if any.

    Returns:
        One sentence, ending before the read command.
    """
    task = kind.replace("_", " ").capitalize()
    words = (refusal_words(failure).get("detail") if failure else None) or failure
    if event == "NEEDS_DECISION":
        happened = "needs a decision; yours, or the person's where it says so"
    elif event == "DEFERRED":
        happened = "is deferred and resumes by itself"
    elif lifecycle == "SUCCEEDED":
        return f"Host: {task} finished; its result is yours to read."
    else:
        happened = "stopped; its way on is yours, or the person's where it says so"
    return f"Host: {task} {happened}" + (f": {words.rstrip('.')}." if words else ".")


def _queue_wake(thread: str, message: str, path: str | None = None) -> dict[str, Any]:
    """Queue one line to a Codex thread, once, and name the outcome.

    A queued message arrives as a user message, so it carries no event body and no instruction:
    only what happened and the command that reads it. The command's success means Codex stored the
    line in its queue, not that the model read it: Codex takes its queue while the thread is idle,
    and an active goal's next turn starts first.
    """
    from alphalattice.interface.local_application.native_setup import codex_command

    codex, source = codex_command(path)
    result: dict[str, Any] = {
        "channel": "codex-queue",
        "delivered": False,
        "command": codex,
        "command_source": source,
    }
    if codex is None:
        return {**result, "failure": "CODEX_COMMAND_MISSING"}
    try:
        subprocess.run(
            [codex, "queue", "--thread", thread, "--message", message],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.CalledProcessError:
        return {**result, "failure": "CODEX_QUEUE_FAILED"}
    except subprocess.TimeoutExpired:
        return {**result, "failure": "CODEX_QUEUE_TIMED_OUT"}
    except OSError:
        return {**result, "failure": "CODEX_QUEUE_START_FAILED"}
    # `delivered` is the stored spelling of queued: it and WAKE_DELIVERED and task.wake_delivered
    # sit in people's activity ledgers, so they stay, and every word a reader sees says queued.
    return {**result, "delivered": True}


def _persisted_code(value: object) -> str | None:
    """A typed code as-is; any other failure text becomes the withheld marker."""

    if value is None or value == "":
        return None
    return safe_failure_code(value) or FAILURE_DETAIL_WITHHELD


def _refused(failure_code: str, next_action: str) -> dict[str, object]:
    answer: dict[str, object] = {
        "status": "REFUSED",
        "failure_code": failure_code,
        "next_action": next_action,
    }
    if failure_code == "activity.storage_unavailable":
        answer.update(refusal_words(failure_code))
        answer["next_action"] = next_action
        answer["next_requests"] = {
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
        }
    return answer


__all__ = [
    "ARTIFACT_SOURCE_KIND",
    "EXTERNAL_SOURCE_KIND",
    "FAILURE_DETAIL_WITHHELD",
    "OBSERVATION_LEDGER_NAME",
    "OPERATION_SOURCE_KIND",
    "TASK_SOURCE_KIND",
    "ArtifactReference",
    "ArtifactResolver",
    "WorkspaceActivity",
    "failure_class_name",
    "safe_enum_token",
    "safe_failure_code",
    "safe_references",
    "task_run_id",
]
