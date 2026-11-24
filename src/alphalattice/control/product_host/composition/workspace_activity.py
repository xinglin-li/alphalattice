"""Workspace activity: safe observations of product operations, composed once.

The Host already owns three truths about work: the shared operation owner sees
every request and its answer, the dispatcher knows when a command has returned,
and Task Control and the artifact owners hold lifecycle and results. This
module writes those facts into the Host's history as they happen (each one a
span or a log record in the machine's log, under this workspace) and reads them
back in bounded pages, without becoming a second store, a broker or a new
authority.

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

Activity is optional infrastructure. A Host whose history does not record leaves
the product usable and the observer visibly UNAVAILABLE; a callback that fails
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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, Final, cast
from uuid import UUID, uuid4

from pydantic_core import to_jsonable_python

from alphalattice.capabilities.domain_contracts import spawn
from alphalattice.capabilities.domain_contracts import telemetry_names as telemetry
from alphalattice.capabilities.domain_contracts.project_layout import runtime_json
from alphalattice.control.observation_runtime.adapters import safe_observation_draft
from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthority,
    ObservationAvailability,
    ObservationDraft,
    ObservationReadCursor,
    ObservationReadPage,
    ObservationRecord,
    ObservationRetentionClass,
)
from alphalattice.control.observation_runtime.log_reads import (
    INDEXED,
    MAXIMUM_READ_PAGE,
    OBSERVATION_ID,
    OBSERVATION_INDEXED,
    SUBJECT_INDEXED,
    LogObservations,
    ObservationStorageError,
    agent_attributes,
    agent_event,
    payload_of,
)
from alphalattice.control.observation_runtime.policy import (
    EXTERNAL_ACTIVITY_SCHEMA,
    PRODUCT_OPERATION_SCHEMA,
    default_observation_policies,
)
from alphalattice.control.observation_runtime.profile import align_clock_inputs, stage_profile
from alphalattice.control.product_host.composition.native_usage_reader import UNVERIFIED_LIMIT
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    ArtifactReference,
    PortfolioResearchOperations,
    task_status_body,
)
from alphalattice.control.product_host.publication.goals import sessions_of
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.control.task_control.timing import read_stage_spans, stage_spans_path
from alphalattice.control.task_control.upgrade import set_aside
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.admit import SECRET_NAME
from alphalattice.foundation.data_layer.chain import canonical
from alphalattice.foundation.data_layer.log import ATTRS_CHARACTERS
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
from alphalattice.interface.local_application.goals import Goal
from alphalattice.interface.local_application.native_bridge import (
    CHILD_USAGE_BINDING_UNVERIFIED as UNVERIFIED,
)
from alphalattice.interface.local_application.native_bridge import (
    HOSTS,
    codex_command,
    codex_queue_reach,
    first_line,
    joined_members,
)
from alphalattice.interface.local_application.portfolio_research import (
    OperationCaller,
    PortfolioResearchOperationRequest,
)
from alphalattice.interface.local_application.questions import asked_questions, person_question
from alphalattice.kernel.clock import clock_metadata, utc_now
from alphalattice.kernel.execution import execution_port
from alphalattice.kernel.resources import worker_width

"""Under `<workspace>/runtime`, the same store the Desk publishers use."""

OPERATION_SOURCE_KIND = "PRODUCT_OPERATION"
TASK_SOURCE_KIND = "TASK_CONTROL"
ARTIFACT_SOURCE_KIND = "PRODUCT_ARTIFACT"
EXTERNAL_SOURCE_KIND = "EXTERNAL_CLIENT"
_ENUM_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_CLASS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,79}")
_REFERENCE_VALUE = re.compile(r"^[A-Za-z0-9_.:,\-/]{1,128}$")
_SESSION_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


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


def _joined(
    records: list[dict[str, Any]], snapshots: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Both reads with each member's submits under that member, as its session file proves."""
    both = records + snapshots
    subjects = joined_members(
        [(item.get("payload") or {}).get("subject") or {} for item in both],
        agent="native_agent_id",
        session="native_session_id",
        members=True,
    )
    both = [
        {**item, "payload": {**item["payload"], "subject": subject}}
        if item.get("payload")
        else item
        for item, subject in zip(both, subjects, strict=True)
    ]
    return both[: len(records)], both[len(records) :]


@dataclass
class WorkspaceActivity:
    """One recorder and one bounded reader over the workspace's records in the Host's history."""

    workspace: Path
    workspace_id: str
    gate: WorkspaceMutationGate
    instance: str
    clock: Callable[[], datetime] = utc_now
    native_usage_state: Callable[[], dict[str, object]] | None = None
    serving_host: dict[str, object] = field(default_factory=dict, init=False)
    """Optional Host-owned usage health read; only its safe status and reason are exposed."""
    store_failure: str | None = field(default=None, init=False)
    """The typed code of the storage failure that left this observer UNAVAILABLE."""

    _keeping: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    """Held while an agent's event is checked against its replays and written, once."""
    _sequences: dict[tuple[str, str], int] = field(default_factory=dict, init=False, repr=False)
    _operation_spans: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    """Each entered operation's span in the history, by operation_ref, until it returns."""
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
        """Set the 0.2.7 observation ledger aside and record the replay clock's correlation, when
        there is one, in the Host's history.

        The old `runtime/observations.sqlite` retires without import (the person, 11-15): the
        upgrade keeps it under `retired-0.2.7/` and the live root no longer holds it. An activity
        built where no Host's history records (a Host refuses to start without it)
        keeps nothing: the failure code is retained, every read says why, and recording is not
        silently retried; operation execution remains separate.
        """
        runtime = self.workspace / "runtime"
        set_aside(runtime, (runtime / f"{LEDGER_0_2_7}{end}" for end in ("", "-wal", "-shm")))
        if not history.recording():
            self.store_failure = "activity.storage_unavailable"
            self._note_failure(self.store_failure, None)
            return
        if isinstance(correlation := getattr(self.clock, "correlation", None), dict):
            self._append(
                schema_kind="ClockCorrelationRecorded",
                source_kind=OPERATION_SOURCE_KIND,
                source_id=self.source_id,
                payload=correlation,
                authority=ObservationAuthority.OPERATIONAL_ASSERTION,
                task_id=None,
                run_id=None,
                correlation_ids=(),
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
        if operations is not None:
            operations.goals.profile = lambda goal: self.goal_profile(
                goal, operations.heartbeats.runtime_paths
            )
        dispatcher.on_command_returned = self.command_returned
        self._wake_sender = execution_port().register_pool(
            ThreadPoolExecutor(max_workers=worker_width(1), thread_name_prefix="codex-wake"),
            name="codex-wake",
        )
        for held in registry.interrupt_wakes(observed_at=self.clock()):
            self._isolated(functools.partial(self._record_wake, held))

    def replay_wakes(self) -> None:
        """Send every pending wake whose Task reached its event while no hook saw it.

        The start's replay, once the Host has resumed its Tasks.
        """
        self._wake_changed(None)

    def close(self) -> None:
        """Finish the wake being sent."""
        sender, self._wake_sender = self._wake_sender, None
        if sender is not None:
            sender.shutdown(wait=True)

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
        """Read whether this activity's records are kept: the Host's history records.

        Returns:
            Whether activity storage is available.
        """
        return self._reads is not None

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
            The history's epoch string, or None when storage is unavailable.
        """
        return None if self._reads is None else str(self._reads.store_epoch)

    # ----------------------------------------------------- operation observer

    def entered(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: OperationCaller,
        relayed: bool = False,
    ) -> OperationSpan | None:
        """Record admitted operation entry using bounded safe references and field names.

        Agent session and goal references are provenance alongside the operation and grant no
        identity or execution authority.

        Args:
            request: Validated operation declaration.
            caller: Declared operation caller.
            relayed: Relay admission already checked by the operation owner.

        Returns:
            Operation span for observed operations, otherwise None; recording failures are isolated.
        """
        if not observed_operation(request.operation):
            return None
        provenance = REQUEST_PROVENANCE.get()
        span = OperationSpan(
            operation=request.operation,
            caller=caller,
            operation_ref=uuid4().hex,
            subject={},
            request_fields=[],
            started=perf_counter(),
            # The operation owner already admitted the confirmation's relay.
            # HUMAN decisions ignore it and use their canonical direct code.
            confirmation=(request.person_confirmation if caller != "HUMAN" and relayed else None),
        )

        # The agent session and goal the request's client named, read in the request's own
        # context: provenance beside what the request did, never an identity (OP13, ID6).
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

    def _record_person_questions(
        self, span: OperationSpan, body: Mapping[str, Any], subject: dict[str, Any]
    ) -> None:
        asked = asked_questions(span.operation, body.get("person_questions"))
        confirmation = span.confirmation
        answered = outcome_of(dict(body)) == "OK" and (
            confirmation is not None
            or (span.caller == "HUMAN" and span.operation == "NETWORK_ACCESS_SET")
        )
        if asked or answered:
            code = confirmation.question_code if confirmation else "network.workspace_access"
            definitions = asked or [person_question(code) if code else {}]
            refs = safe_references(
                {
                    **subject,
                    "workspace_id": self.workspace_id,
                    "operation_ref": span.operation_ref,
                }
            )
            if confirmation is not None:
                refs["decision_hash"] = confirmation.decision_hash
            subject["person_questions"] = [
                {
                    "code": definition.get("code"),
                    "catalog_hash": definition.get("catalog_hash"),
                    "state": "ASKED" if asked else "ANSWERED" if definition else "LEGACY_UNCODED",
                    "references": refs,
                }
                for definition in definitions
            ]

    def returned(self, span: OperationSpan, body: Mapping[str, Any]) -> None:
        """Record an operation return with safe status, references and next-read metadata.

        Recording failures are isolated from the operation result.

        Args:
            span: Exact entered operation span.
            body: Owner-supplied operation response.
        """

        def record() -> None:
            subject = {**span.subject, **safe_references(returned_subject(body))}
            # Question enrichment must not erase the ordinary RETURNED row.
            self._isolated(lambda: self._record_person_questions(span, body, subject))
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
                and OBSERVATION_ID.fullmatch(conversation["observation_id"]) is not None
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
                    if self._reads is None
                    else self._reads.unique_correlated_observation(
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
                offered_requests=body.get("next_requests") or {},
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
        at = self.clock()
        provenance = REQUEST_PROVENANCE.get()
        if code is not None:
            vendor = None if provenance is None else provenance.vendor
            refusal = {"operation": operation, "failure_code": code, "caller": caller}
            refused = refusal | {"vendor": vendor}
            self._isolated(
                lambda: history.fact(
                    telemetry.EVENT_REQUEST_REFUSED, refused, at=at, workspace=self.workspace_id
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
        if self._reads is None:
            return _refused("activity.storage_unavailable", "RESTART_HOST_AFTER_STORE_REPAIR")
        since = (self.clock().astimezone(UTC) - timedelta(days=days - 1)).date().isoformat()
        rows = self._reads.refusal_counts(since_day=since)
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
        offered_requests: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "operation": span.operation,
            "caller": span.caller,
            "phase": phase,
            "operation_ref": span.operation_ref,
            "subject": span.subject if subject is None else subject,
            "request_fields": span.request_fields,
        }
        payload.update(clock_metadata(self.clock))
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
        if offered_requests is not None:
            payload["offered_requests"] = sorted(
                name
                for name in offered_requests
                if isinstance(name, str) and re.fullmatch(r"[a-z][a-z_]{0,47}", name)
            )[:16]
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
        sequence with different content is refused, and the producer restarts its
        sequence by opening a new session.
        """
        if self._reads is None:
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
                policies=_POLICIES,
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
            disposition, kept = self._keep(draft)
        except ValueError as error:
            code = safe_failure_code(str(error)) or "activity.event_not_admitted"
            return _refused(code, "READ_ACTIVITY_EVENT_CONTRACT")
        except ObservationStorageError as error:
            self._note_failure(error.failure_code, type(error).__name__)
            return _refused(error.failure_code, "RETRY_SAME_SEQUENCE_LATER")
        self._isolated(history.flush)  # the reply names the record: it is committed first
        self._note_append()
        return {
            "status": disposition,
            "observation_id": kept.observation_id,
            "source_id": source_id,
            "source_sequence": kept.source_sequence,
            "authority": ObservationAuthority.AGENT_PROPOSAL.value,
            "task_verified": verified,
            "summary_truncated": truncated,
            "cursor": self._cursor_at_head(),
        }

    # --------------------------------------------------------------- reads

    def read(self, query: ActivityReadQuery) -> dict[str, object]:
        """One bounded page, the current projection of every Task it names and the CLI call at the
        root of each of its traces.

        The Task join reads only the projections this page references (plus the
        watched ids), never the whole projection table; a Task admitted moments
        ago that has no row yet is read from the registry directly.
        """
        started = perf_counter()
        ledger = self._reads
        page = None if ledger is None else ledger.observations_after(query.after, limit=query.limit)
        referenced = {str(value) for value in query.watch}
        items = [] if page is None else [activity_item(item) for item in page.items]
        if page is not None:
            referenced.update(
                item.record.task_id for item in page.items if item.record.task_id is not None
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
            "calls": _calls(ledger, page),
            "observer": self.observer_state(),
            **({"workspace_writer": writer} if (writer := self.gate.waiting()) else {}),
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

    def _profile_rows(self, goal: Goal) -> tuple[list[Any], list[Any], int, list[Any]]:
        ledger, head = self._reads, 0
        rows: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        clocks: list[dict[str, Any]] = []
        if ledger is None:
            return rows, [{"code": "LEDGER_UNAVAILABLE"}], head, clocks
        cursor = ObservationReadCursor(store_epoch=ledger.store_epoch, ordinal=0)

        def page_at(at: ObservationReadCursor) -> ObservationReadPage | None:
            try:
                return ledger.observations_after(at, limit=MAXIMUM_READ_PAGE)
            except ObservationStorageError:
                gaps.append({"code": "LEDGER_ROWS_UNREADABLE"})
                return None

        page = page_at(cursor)
        while page is not None:
            head = head or page.head_ordinal
            for item in page.items:
                record, payload = item.record, item.record.inline_safe_payload
                if item.ordinal > head:
                    break
                if payload is None:
                    gaps.append({"code": "LEDGER_ROW_UNAVAILABLE", "source": record.observation_id})
                elif (
                    record.schema_kind in {PRODUCT_OPERATION_SCHEMA, "ClockCorrelationRecorded"}
                    and record.source_kind == OPERATION_SOURCE_KIND
                    and record.authority == ObservationAuthority.OPERATIONAL_ASSERTION
                ):
                    target = rows if record.schema_kind == PRODUCT_OPERATION_SCHEMA else clocks
                    target.append(
                        {
                            **payload,
                            "at": record.observed_at.isoformat(),
                            "source": record.source_id,
                            "provenance": record.observation_id,
                            "clock_domain": payload.get("clock_domain", "LEDGER_OBSERVED_AT"),
                        }
                    )
            if not page.more or page.next_cursor.ordinal >= head:
                break
            if not page.advances(cursor):
                gaps.append({"code": "LEDGER_CURSOR_STALLED"})
                break
            cursor = page.next_cursor
            page = page_at(cursor)
        owned = {
            (r["source"], r.get("operation_ref"))
            for r in rows
            if r.get("subject", {}).get("goal_id") == str(goal.goal_id)
        }
        return (
            [
                r
                for r in rows
                if (r["source"], r.get("operation_ref")) in owned
                and r.get("subject", {}).get("goal_id") in {None, str(goal.goal_id)}
            ],
            gaps,
            head,
            clocks,
        )

    def _profile_window(
        self,
        goal: Goal,
        rows: Sequence[Any],
        offsets: Mapping[str, float] | None,
        gaps: list[dict[str, Any]],
    ) -> tuple[str, str, str]:
        """Bind terminal cuts to their saved revision; active reads use one snapshot clock."""
        times = [datetime.fromisoformat(row["at"]) for row in rows]
        start = (
            min(times)
            if offsets is not None and times
            else min([goal.intent_registered_at, *times])
        )
        end = max(times, default=start) if offsets is not None else goal.recorded_at
        source = "RECORDED_RPC_BOUND" if offsets is not None else "GOAL_TERMINAL_RECORD"
        if goal.state == "OPEN":
            snapshot = self.clock()
            source = "HOST_READ_SNAPSHOT"
            if offsets is not None:
                offset = offsets.get(getattr(self.clock, "reference", ""))
                if offset is None or clock_metadata(self.clock).get("clock_drift") is not False:
                    gaps.append({"code": "CLOCK_WINDOW_UNOBSERVED"})
                    snapshot = end
                    source = "RECORDED_RPC_BOUND"
                else:
                    snapshot += timedelta(seconds=offset)
            end = max([end, snapshot, *times])
        else:
            closing = [
                at
                for at, row in zip(times, rows, strict=True)
                if row.get("phase") == "RETURNED"
                and (
                    row.get("status") == "GOAL_SAVED"
                    or (row.get("operation"), row.get("status")) == ("GOAL_SUBMIT", "COMPLETE")
                )
                and row.get("operation") in {"GOAL_SUBMIT", "GOAL_ABANDON"}
                and row.get("subject", {}).get("goal_hash") == goal.goal_hash
            ]
            end = min(closing, default=end)
            source = "GOAL_TERMINAL_RETURN" if closing else source
            if offsets is not None and not closing:
                gaps.append({"code": "CLOCK_WINDOW_UNOBSERVED"})
        return start.isoformat(), end.isoformat(), source

    def _goal_trace(self, goal: Goal) -> dict[str, Any] | None:
        """The goal's calls from the log: those naming it, and those of each session that worked
        for it (its record names them) without naming it."""
        operations = self._operations
        attributed = () if operations is None else operations.goals.store.attributed(goal.goal_id)
        sessions = [session["session_id"] for session in sessions_of(attributed)]
        trace: dict[str, Any] | None = history.goal_trace(
            str(goal.goal_id), sessions, workspace=self.workspace_id
        )
        return trace

    def goal_profile(self, goal: Goal, runtime_paths: tuple[Path, ...]) -> dict[str, Any]:
        """Read each scoped owner input once; the history's head fixes the answer's window."""
        rows, gaps, head, clocks = self._profile_rows(goal)
        phases: list[dict[str, Any]] = []
        for task in sorted({row["task_id"] for row in rows if row.get("task_id")}):
            items: Sequence[Any] = ()
            if self._registry is not None:
                try:
                    _record, items = self._registry.task_with_work_items(UUID(task))
                except KeyError:
                    gaps.append({"code": "TASK_RECORD_UNAVAILABLE", "source": task})
            observed: set[tuple[str, str]] = set()
            for parent, runtime in {path.parent: path for path in runtime_paths}.items():
                kept = read_stage_spans(runtime, UUID(task), diagnostics=gaps)
                if not kept:
                    gaps.append({"code": "UNKEPT_OR_INTERRUPTED_PHASE", "source": task})
                phases.extend(
                    {
                        **row,
                        "namespace": str(parent),
                        "provenance": str(stage_spans_path(runtime, UUID(task))),
                    }
                    for row in kept
                )
                observed.update((str(row.get("stage_id")), str(row.get("phase"))) for row in kept)
            for item in items:
                for phase in ("execute", "verify"):
                    if item.started_at is not None and (item.stage_id, phase) not in observed:
                        gaps.append(
                            {
                                "code": "UNKEPT_OR_INTERRUPTED_PHASE",
                                "source": [task, item.stage_id, phase, item.state_hash],
                            }
                        )
        recorded_clock = clocks or any(row.get("clock_ref") for row in (*rows, *phases))
        offsets = align_clock_inputs(rows, phases, clocks, gaps) if recorded_clock else None
        start, end, source = self._profile_window(goal, rows, offsets, gaps)
        if not rows:
            gaps.append({"code": "RPC_RECORDS_UNOBSERVED"})
        catalog = runtime_json("config/registries/messages.json")["entries"]
        return {
            **stage_profile(
                window=(start, end),
                window_source=source,
                rpc=rows,
                phases=phases,
                diagnostics=gaps,
            ),
            "status": "GOAL_PROFILE",
            "trace": self._goal_trace(goal),
            "goal_id": str(goal.goal_id),
            "goal_hash": goal.goal_hash,
            "goal_kind": goal.declaration.kind,
            "ledger_head": head,
            "ledger_epoch": self.store_epoch,
            "workspace_id": self.workspace_id,
            "runtime_origins": list(map(str, runtime_paths)),
            "labels": {
                code: row["words"]
                for code, row in catalog.items()
                if code.startswith("goal.profile.")
            },
            "stdout_hint": {
                "code": "cli.profile.full_output",
                **catalog["cli.profile.full_output"]["words"],
            },
        }

    def recent(self, *, limit: int = 20) -> dict[str, object]:
        """The newest requests, grouped by agent session and goal, each with its read (OP13).

        One read finds what just happened, whichever session asks: the newest returned
        operations with the agent session and goal their rows recorded, and the read that
        opens each. A missing or unsupported session identity is UNKNOWN, never inferred
        from workspace bindings. The feed is transient; a goal keeps its own record.

        Args:
            limit: How many groups, from 1 to 50.

        Returns:
            The groups, newest first, each with up to five of its newest requests.
        """
        if not 1 <= limit <= 50:
            raise ValueError("activity.recent_limit_invalid")
        ledger = self._reads
        writer = self.gate.waiting()
        if ledger is None:
            return {
                "status": "UNAVAILABLE",
                "groups": [],
                "observer": self.observer_state(),
                **({"workspace_writer": writer} if writer else {}),
            }
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
                    "agent_attribution": "RECORDED"
                    if key[0] in {"codex", "claude-code"} and key[1]
                    else "UNKNOWN",
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
                        "failure_code": payload.get("failure_code"),
                        "next_action": subject.get("next_action"),
                        "offered_requests": payload.get("offered_requests"),
                        "identifier_kinds": sorted(
                            name
                            for name in subject
                            if name.endswith(("_id", "_hash", "_reference"))
                        ),
                    }
                )
        return {
            "status": "AVAILABLE",
            "groups": list(groups.values()),
            **({"workspace_writer": writer} if writer else {}),
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
        A selected Session's latest member snapshots are independent of that
        exchange cursor; their separate bound and truncation are disclosed.
        """
        if query.observation_id is not None:
            return self._accepted_answer_detail(query.observation_id)
        started = perf_counter()
        ledger = self._reads
        page = (
            None
            if ledger is None
            else ledger.observations_of_kind(
                EXTERNAL_ACTIVITY_SCHEMA,
                limit=query.limit,
                before=query.before,
                native_session_id=query.native_session_id,
            )
        )
        snapshots = (
            ledger.observations_of_kind(
                EXTERNAL_ACTIVITY_SCHEMA,
                limit=MAXIMUM_READ_PAGE,
                native_session_id=query.native_session_id,
                snapshots=True,
                head_ordinal=page.head_ordinal,
            )
            if ledger is not None and page is not None and query.native_session_id is not None
            else None
        )
        records = [] if page is None else [activity_item(item) for item in page.items]
        snapshot_items = (
            [] if snapshots is None else [activity_item(item) for item in snapshots.items]
        )
        records, snapshot_items = _joined(records, snapshot_items)
        by_kind: dict[str, list[dict[str, Any]]] = {
            EXTERNAL_ACTIVITY_SCHEMA: [],
            PRODUCT_OPERATION_SCHEMA: [],
        }
        for item in records:
            by_kind[item["schema_kind"]].append(item)
        items, product_items = by_kind[EXTERNAL_ACTIVITY_SCHEMA], by_kind[PRODUCT_OPERATION_SCHEMA]
        return {
            "workspace_id": self.workspace_id,
            "disposition": "UNAVAILABLE" if page is None else page.disposition,
            "epoch": None if page is None else page.store_epoch,
            "head": 0 if page is None else page.head_ordinal,
            "oldest": records[0]["ordinal"] if records else None,
            "more": False if page is None else page.more,
            "items": items,
            **(
                {
                    "native_session_id": query.native_session_id,
                    "snapshots": snapshot_items,
                    "product_items": product_items,
                    "snapshots_more": False if snapshots is None else snapshots.more,
                }
                if query.native_session_id is not None
                else {}
            ),
            "unavailable": sum(
                current.unavailable_count for current in (page, snapshots) if current is not None
            ),
            "observer": self.observer_state(),
            "read_cost": {
                "observations": len(records) + len(snapshot_items),
                "elapsed_ms": int((perf_counter() - started) * 1000),
            },
        }

    def _accepted_answer_detail(self, observation_id: str) -> dict[str, object]:
        """Open one selected native event or owner return and verify its sealed answer."""
        started = perf_counter()
        ledger, operations = self._reads, self._operations
        answer: dict[str, object] = {
            "status": "UNAVAILABLE",
            "reason": "activity.accepted_answer_unavailable",
            "observation_id": observation_id,
        }
        observations = 0
        if ledger is not None:
            try:
                record = ledger.read(observation_id)
            except (ObservationStorageError, ValueError):
                pass
            else:
                observations = 1
                payload = record.inline_safe_payload or {}
                subject = payload.get("subject")
                if (
                    record.schema_kind == EXTERNAL_ACTIVITY_SCHEMA
                    and record.availability is ObservationAvailability.AVAILABLE
                    and payload.get("event_kind") == "NATIVE_COORDINATION_MESSAGE"
                    and isinstance(subject, dict)
                    and (subject.get("message_kind"), subject.get("input_channel"))
                    in {("answer", "PRODUCT_ACCEPTED_ANSWER"), ("acceptance", "PRODUCT_REVIEW")}
                    and operations is not None
                    and operations.review is not None
                ):
                    if subject.get("authorship_basis") == "NOT_OBSERVED":
                        answer, checked = self._product_accepted_event_detail(
                            observation_id, subject
                        )
                        observations += checked
                    else:  # a hook's, or a review thread's acceptance: its Task from the thread
                        if subject.get("input_channel") == "PRODUCT_REVIEW":
                            from . import committee  # lazy: research review on the activity feed

                            task = committee.accepted_task(
                                operations.goals.store, str(subject.get("reference"))
                            )
                            subject = {**subject, "reference": task}
                        answer = {
                            **operations.review.read_accepted_answer(subject),
                            "observation_id": observation_id,
                        }
                elif (
                    record.schema_kind == PRODUCT_OPERATION_SCHEMA
                    and record.source_kind == OPERATION_SOURCE_KIND
                    and (
                        record.source_id == self.source_id
                        or re.fullmatch(r"local-web:[0-9a-f]{32}", record.source_id) is not None
                    )
                    and record.authority is ObservationAuthority.OPERATIONAL_ASSERTION
                    and record.availability is ObservationAvailability.AVAILABLE
                    and payload.get("operation") == "AGENT_ANSWER_SUBMIT"
                    and payload.get("phase") == "RETURNED"
                    and payload.get("status") in {"ACCEPTED", "DONE"}
                    and isinstance(subject, dict)
                    and isinstance(subject.get("task_id"), str)
                    and record.task_id == payload.get("task_id") == subject["task_id"]
                    and record.run_id == task_run_id(subject["task_id"])
                    and operations is not None
                    and operations.review is not None
                ):
                    answer = {
                        **operations.review.read_product_accepted_answer(
                            subject, verdict=payload["status"]
                        ),
                        "observation_id": observation_id,
                        "source_kind": record.source_kind,
                        "source_id": record.source_id,
                        "authority": record.authority.value,
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
        """Read a default relay only through its actual correlated owner-return record."""
        unavailable: dict[str, object] = {
            "status": "UNAVAILABLE",
            "reason": "activity.accepted_answer_unavailable",
            "observation_id": observation_id,
            "missing": ["accepted_operation_return"],
        }
        ledger, operations = self._reads, self._operations
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
        """Share successful reach probes; retry a supplied path after failure."""
        with self._lock:
            command, source = codex_command(path)
            if (
                self._wake_capability is None
                or self._wake_capability.get("command") != command
                or (path and not self._wake_capability.get("present"))
            ):
                self._wake_capability = codex_queue_reach(path)
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
            "host": dict(self.serving_host),
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
                state["native_usage"] = _usage_view(usage)
                sessions = usage.get("sessions")
                if isinstance(sessions, dict):  # each Session's own, for its own page
                    cast(dict[str, object], state["native_usage"])["sessions"] = {
                        sid: _usage_view(own)
                        for sid, own in sessions.items()
                        if isinstance(sid, str)
                        and (":" not in sid or sid.split(":", 1)[0] in HOSTS)
                        and _SESSION_ID.fullmatch(sid.rsplit(":", 1)[-1])
                        and isinstance(own, dict)
                    }
            except Exception:
                state["native_usage"] = _usage_view(
                    {"status": "UNAVAILABLE", "reason": "activity.observer_failed"}
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
        if self._reads is None:
            self._note_failure("activity.storage_unavailable", None)
            return
        with self._lock:
            key = (source_kind, source_id)
            sequence = self._sequences.get(key, -1) + 1
            self._sequences[key] = sequence
        try:
            draft = safe_observation_draft(
                policies=_POLICIES,
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
            self._keep(draft)
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

    def _keep(self, draft: ObservationDraft) -> tuple[str, ObservationRecord]:
        """Validate a draft and write it into the Host's history once.

        Returns:
            APPENDED with the record written (its id the event's), or REUSED_EXACT with the one an
            agent's replayed sequence already wrote.

        Raises:
            ObservationStorageError: The Host's history does not record, or did not keep it.
            ValueError: The draft is refused, it is of a kind the history keeps no event for, or
                a replayed sequence changed its content.
        """
        reads = self._reads
        if reads is None:
            raise ObservationStorageError("activity.storage_unavailable", "history not recording")
        if draft.schema_kind not in (PRODUCT_OPERATION_SCHEMA, EXTERNAL_ACTIVITY_SCHEMA, *_FACTS):
            raise ValueError("observation.payload_not_admitted")
        kept, observed_at = _POLICIES.validate_and_redact(draft), self.clock()
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observation clock must be timezone-aware")
        observed_at = observed_at.astimezone(UTC)
        with self._keeping:
            if kept.schema_kind == EXTERNAL_ACTIVITY_SCHEMA:
                existing = reads.replayed(kept.source_id, kept.source_sequence)
                if existing is not None:
                    if _own(existing) != _own(kept):
                        raise ValueError("observation.source_sequence_collision")
                    return "REUSED_EXACT", existing
            try:
                event_id = trace_record(kept, observed_at, self._operation_spans, self.workspace_id)
            except ObservationStorageError:
                raise
            except Exception as error:  # the history did not keep it: never counted as kept
                raise ObservationStorageError(
                    "activity.observer_failed", type(error).__name__
                ) from error
        if event_id is None:
            raise ObservationStorageError("activity.storage_unavailable", "history not recording")
        return "APPENDED", record_of(kept, observed_at, event_id)

    @property
    def _reads(self) -> LogObservations | None:
        """This workspace's records in the Host's log; None while the history does not record."""
        opened = history.opened_log()
        return None if opened is None else LogObservations(opened[0], self.workspace_id)

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
        reads = self._reads
        if reads is None:
            return None
        return f"{reads.store_epoch}:{reads.head_ordinal()}"


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
        spawn.run(
            [codex, "queue", "--thread", thread, "--message", message],
            process_kind="codex_wake_queue",
            check=True,
            capture_output=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as error:
        return {**result, "failure": "CODEX_QUEUE_FAILED", "detail": first_line(error.stderr)}
    except subprocess.TimeoutExpired:
        return {**result, "failure": "CODEX_QUEUE_TIMED_OUT"}
    except OSError:
        return {**result, "failure": "CODEX_QUEUE_START_FAILED"}
    # `delivered` is the stored spelling of queued: it and WAKE_DELIVERED and task.wake_delivered
    # sit in people's activity histories, so they stay, and every word a reader sees says queued.
    return {**result, "delivered": True}


def _usage_view(usage: dict[str, Any]) -> dict[str, object]:
    """A usage standing as a page shows it: owner status, safe reason and its words, progress.

    A Session's unverified children are named by their safe ids, each its own row.
    """
    status = safe_enum_token(usage.get("status"))
    view: dict[str, object] = {
        "status": status
        if status in {"NOT_BOUND", "OFF", "OBSERVING", "STOPPED", "UNAVAILABLE"}
        else "UNAVAILABLE",
        "reason": _persisted_code(usage.get("reason")),
        **{
            key: value
            for key in ("read_through", "total_bytes", "bytes_read")
            if type(value := usage.get(key)) is int and value >= 0
        },
        **({"rescan": usage["rescan"]} if type(usage.get("rescan")) is bool else {}),
    }
    if isinstance(reason := view["reason"], str):
        words = refusal_words(reason)
        view.update({key: words[key] for key in ("detail", "next_action") if key in words})
    if isinstance(seconds := usage.get("model_seconds"), (float, int)) and 0 <= seconds < float(
        "inf"
    ):
        view["model_seconds"] = seconds
    links = usage.get("lineage")
    if type(unplaced := usage.get("unplaced")) is int and unplaced >= 0:
        view["unplaced"] = unplaced
    if isinstance(links, (list, tuple)):
        valid = [
            list(pair)
            for pair in links
            if isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(value, str) and _SESSION_ID.fullmatch(value) for value in pair)
        ]
        view["lineage"], view["lineage_count"] = valid[:UNVERIFIED_LIMIT], len(valid)
    unverified = usage.get("unverified")
    if isinstance(unverified, list) and unverified:
        view["unverified"] = [
            i
            for i in unverified[:UNVERIFIED_LIMIT]
            if isinstance(i, str) and _SESSION_ID.fullmatch(i)
        ]
        count = usage.get("unverified_count")
        view["unverified_count"] = (
            count if type(count) is int and count >= len(unverified) else len(unverified)
        )
        view["unverified_detail"] = refusal_words(UNVERIFIED).get("detail")
    return view


LEDGER_0_2_7 = "observations.sqlite"
"""The retired observation ledger's file in a 0.2.7 workspace's runtime folder."""
_POLICIES = default_observation_policies()
"""The schema policies every record is validated and redacted by before it is kept."""


OWN: Final = (
    "occurred_at",
    "source_kind",
    "source_id",
    "source_sequence",
    "task_id",
    "run_id",
    "stage_id",
    "correlation_ids",
    "authority",
    "retention_class",
)
"""A record's own fields, beside its payload and the clock the Host kept it at: what no column
of its event holds."""


def _calls(
    reads: LogObservations | None, page: ObservationReadPage | None
) -> dict[str, dict[str, object]]:
    """The CLI call at the root of each trace a page's items are in, by trace."""
    traces = {item.trace_id for item in page.items if item.trace_id} if page else set()
    return reads.calls(traces) if reads is not None and traces else {}


def _own(record: ObservationDraft | ObservationRecord) -> dict[str, Any]:
    """What a replayed agent event must repeat: its own fields and its payload, as the log reads
    an agent's event back (`payload_of`, which keeps no empty subject)."""
    own: dict[str, Any] = record.model_dump(mode="json", include=set(OWN))
    payload = dict(record.inline_safe_payload or {})
    return own | {
        "payload": payload_of(
            telemetry.EVENT_AGENT_ACTIVITY, agent_attributes(payload), payload.get("summary")
        )
    }


def record_of(kept: ObservationDraft, observed_at: datetime, event_id: str) -> ObservationRecord:
    """The record a kept draft is, kept at `observed_at` by the event `event_id`."""
    return ObservationRecord(
        **kept.model_dump(mode="python", include={*OWN, "schema_kind", "inline_safe_payload"}),
        observed_at=observed_at,
        observation_id=event_id,
    )


def trace_record(
    kept: ObservationDraft, observed_at: datetime, spans: dict[str, Any], workspace: str
) -> str | None:
    """Write a kept record into the Host's history in `workspace` as one event: its payload as the
    event's attributes and its own fields (`OWN`) with `observed_at` as the `observation`
    attribute. The event's type is its kind and the event's id its identity: that id, None outside
    a Host.

    An operation is one step: requested writes its started row, returned or failed its ended row.
    An agent's event is a fact with its GenAI names, its summary the text; every other record is
    a fact of its type, and each person question an operation asked or answered is its own fact."""
    payload = dict(kept.inline_safe_payload or {})
    own = kept.model_dump(mode="json", include=set(OWN))
    observation = {"observation": own | {"observed_at": to_jsonable_python(observed_at)}}
    kind = kept.schema_kind
    if kind == PRODUCT_OPERATION_SCHEMA:
        ref, (attributes, text) = (
            str(payload["operation_ref"]),
            bounded(payload | observation, None),
        )
        if payload["phase"] == "REQUESTED":
            step = history.begin(
                telemetry.SPAN_PRODUCT_OPERATION, attributes, text, workspace=workspace
            )
            spans[ref], event_id = step, None if step is None else str(step["event_id"])
        else:
            step, failed = spans.pop(ref, None), payload["phase"] == "FAILED"
            event_id = history.finish(
                step,
                telemetry.SPAN_PRODUCT_OPERATION,
                attributes,
                text,
                failed=failed,
                workspace=workspace,
            )
        for question in (payload.get("subject") or {}).get("person_questions") or ():
            asked = {"operation": payload["operation"], **question}
            history.fact(telemetry.EVENT_PERSON_QUESTION, asked, workspace=workspace)
        return event_id
    if kind == EXTERNAL_ACTIVITY_SCHEMA:
        attributes, text = bounded(agent_attributes(payload) | observation, payload.get("summary"))
        heard: str | None = history.fact(
            telemetry.EVENT_AGENT_ACTIVITY,
            attributes,
            subject=("agent_event", agent_event(kept.source_id, kept.source_sequence)),
            text=text,
            actor=("agent", str(payload["producer_id"])),
            at=kept.occurred_at,
            workspace=workspace,
        )
        return heard
    about = None
    if kind == "TaskControlTransition" and kept.task_id is not None:
        about = ("task", kept.task_id)
    elif kind == "ArtifactVerificationObserved":
        about = ("artifact", str(payload["artifact_hash"]))
    attributes, text = bounded(payload | observation, None)
    kept_as: str | None = history.fact(
        _FACTS[kind], attributes, subject=about, text=text, workspace=workspace
    )
    return kept_as


def bounded(attributes: dict[str, Any], text: str | None) -> tuple[dict[str, Any], str | None]:
    """An observation event's attributes and text as the log keeps them: as they are when the
    attributes fit its bound and no field's name reads as a credential's to it (a public
    `case_token` the observation policy admits); else the attributes the reads select on, marked
    `observation.in_text`, and the whole, attributes and text, as the event's text, which has no
    bound and which the log still refuses when a value is shaped like a credential."""
    fits = len(canonical(attributes).decode()) <= ATTRS_CHARACTERS
    if fits and not _credential_named(attributes):
        return attributes, text
    observation = attributes["observation"]
    index = {key: attributes[key] for key in INDEXED if key in attributes}
    subject = attributes.get("subject") or {}
    kept = {key: subject[key] for key in SUBJECT_INDEXED if key in subject}
    index |= {"subject": kept} if kept else {}
    keys = {key: observation[key] for key in OBSERVATION_INDEXED}
    index["observation"] = keys | {"in_text": True}
    return index, canonical({"attributes": attributes, "text": text}).decode()


def _credential_named(value: Any, name: str = "") -> bool:
    """Whether the log's name rule (`admit.SECRET_NAME`) reads a field of `value` as a credential:
    a named field with a value, unless it names a keychain entry (`*_ref`)."""
    if isinstance(value, dict):
        return any(_credential_named(item, str(key)) for key, item in value.items())
    if isinstance(value, list):
        return any(_credential_named(item, name) for item in value)
    named = bool(SECRET_NAME.search(name)) and not name.endswith("_ref")
    return named and isinstance(value, str) and bool(value)


_FACTS = {
    "TaskControlTransition": telemetry.EVENT_TASK_TRANSITION,
    "ArtifactVerificationObserved": telemetry.EVENT_ARTIFACT_VERIFIED,
    "ClockCorrelationRecorded": telemetry.EVENT_CLOCK_CORRELATED,
}
"""The record kinds kept as facts, by their event type."""


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
