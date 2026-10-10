"""Typed workspace-activity boundary shared by the Host, the browser and clients.

Three things live here and nothing else: which operations of the shared
vocabulary are *recorded* when they run, the typed documents a client may send
or read at the activity routes, and the projection of one observation into the
JSON a consumer renders. Recording itself, Task correlation and the ledger are
owned by the Host composition; this module decides no outcome.

Activity is observation, not authority. An item says what an entry saw an
operation do and what Task Control or an artifact owner later reported; opening
any reference still goes through that reference's own readback route.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, Protocol, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.observation_runtime.contracts import (
    ObservationReadCursor,
    ObservationReadItem,
)
from alphalattice.interface.local_application.failure_codes import (
    FAILURE_DETAIL_WITHHELD,
)
from alphalattice.interface.local_application.portfolio_research import (
    OperationCaller,
    PortfolioResearchOperationRequest,
)
from alphalattice.interface.local_application.web import LocalWebError

READ_OPERATIONS: frozenset[str] = frozenset(
    {
        "RESEARCH_STRATEGY_READBACK",
        "FEATURE_CATALOG_CONTROLS",
        "FEATURE_CATALOG_READBACK",
        "FEATURE_CATALOG_BUILD_READBACK",
        "FEATURE_TRIAL_READBACK",
        "FEATURE_TRIALS",
        "RESEARCH_STRATEGY_CONTROLS",
        "GOAL_SCHEMA",
        "GOAL_LIST",
        "GOAL_SHOW",
        "GOAL_NARRATIVE",
        "GOAL_REFERENCE",
        "GOAL_EXPORT",
        "GOAL_CONTINUE",
        "COMPARE",
        "CONTROLS",
        "CRO_REVIEW_DOSSIER",
        "CRO_REVIEW_FINDING",
        "DATA_ISSUES",
        "DATA_UPDATE_READBACK",
        "COMMITTEE_READ",
        "EVIDENCE_CRO",
        "EVIDENCE_CRO_EXPORT",
        "EVIDENCE_DOCUMENTS",
        "EVIDENCE_LEDGER",
        "EVIDENCE_PACKET",
        "EXPERIMENTS",
        "EXPERIMENT_COMPARE",
        "EXPERIMENT_ALPHA_COMPARE",
        "EXPERIMENT_CONTROLS",
        "EXPERIMENT_DECLARATION_CONVERT",
        "EXPERIMENT_CURATION",
        "EXPERIMENT_DELIVERY_EXPORT",
        "EXPERIMENT_DRAFT",
        "EXPERIMENT_EXPORT",
        "EXPERIMENT_FOUNDATIONS",
        "EXPERIMENT_FOUNDATION_DRAFT",
        "EXPERIMENT_FOUNDATION_EXPORT",
        "EXPERIMENT_FOUNDATION_READBACK",
        "EXPERIMENT_FOUNDATION_SUMMARY",
        "EXPERIMENT_PORTFOLIO_DRAFT",
        "EXPERIMENT_PREVIEW_READBACK",
        "EXPERIMENT_READBACK",
        "EXPERIMENT_SUMMARY",
        "EXPERIMENT_REPLAY",
        "EXPERIMENT_RISK_EXPORT",
        "EXPERIMENT_RISK_LINKS",
        "EXPORT",
        "FINALIZATION",
        "PORTFOLIO_UPDATE_READBACK",
        "PORTFOLIO_READBACK",
        "REPORT",
        "RESEARCH_HISTORY",
        "RESEARCH_INPUTS",
        "RESEARCH_INPUT_READBACK",
        "MODEL_TRAINING_INPUT_READBACK",
        "RESEARCH_UPDATE_AUTOMATION_READBACK",
        "RESEARCH_UPDATE_READBACK",
        "RESULTS",
        "STATUS",
        "STORAGE_CAP_SHOW",
        "STORAGE_READBACK",
        "TASK_RECOVERY",
        "TASK_GUARDIAN",
        "TASK_INCIDENTS",
        "STRATEGY_CALIBRATION_READBACK",
        "STRATEGY_SCORE_READBACK",
        "TASKS",
        "UPGRADE_OVERVIEW",
        "NETWORK_ACCESS",
        "EVIDENCE_CONSENT",
        "USAGE_READING",
        "PENDING_DECISIONS",
        "WORKSPACE_PREPARE_READBACK",
        "ACTIVITY_REFUSALS",
        "ACTIVITY_LIST",
        "ACTIVITY_RECENT",
        "CPU_BUDGET_SHOW",
        "WORKSPACE_BACKUPS",
        "MODEL_EXTENSIONS",
        "FEATURE_EXTENSIONS",
        "FEATURE_REVIEW",
        "OPERATION_LIST",
        "WORKSPACE_SHOW",
    }
)
"""Operations that change nothing and are never recorded as activity.

The browser polls several of these every few seconds; recording them would fill
the ledger with the observer's own reads. A refused read is still not activity:
the caller already holds the refusal.
"""

OBSERVED_OPERATIONS: frozenset[str] = frozenset(
    {
        "AGENT_ANSWER_SUBMIT",
        "AGENT_BUNDLE_PREPARE",
        "GOAL_OPEN",
        "GOAL_REVISE",
        "GOAL_ATTACH",
        "GOAL_NOTE",
        "GOAL_TAKE",
        "GOAL_SUBMIT",
        "GOAL_ABANDON",
        "CANCEL",
        "CRO_REVIEW",
        "CRO_REVIEW_SUBMIT",
        "DATA_CHANGE_CONFIRM",
        "DATA_ISSUE_CONFIRM",
        "DATA_ISSUE_DELEGATE",
        "DATA_ISSUE_REVOKE",
        "DATA_ISSUE_PREVIEW",
        "DATA_UPDATE_PLAN",
        "DATA_UPDATE_RUN",
        "EVIDENCE_ANALYSIS_SUBMIT",
        "EVIDENCE_CONTINUE",
        "EVIDENCE_PREPARE",
        "EVIDENCE_PREVIEW",
        "EVIDENCE_REFRESH",
        "EVIDENCE_SELECT",
        "EXPERIMENT_CURATE",
        "EXPERIMENT_FOUNDATION_PREVIEW",
        "EXPERIMENT_FOUNDATION_SEAL",
        "EXPERIMENT_HANDOFF_PREVIEW",
        "EXPERIMENT_LINK_RISK",
        "EXPERIMENT_PLAN",
        "EXPERIMENT_RUN",
        "EXPERIMENT_PROMOTE",
        "EXPERIMENT_CONTINUE",
        "EXPERIMENT_VERIFY_ALL",
        "MODEL_TRAINING_INPUT_PLAN",
        "FEATURE_CATALOG_PLAN",
        "FEATURE_CATALOG_BUILD",
        "FEATURE_TRIAL",
        "MODEL_TRAINING_INPUT_PREPARE",
        "RESEARCH_STRATEGY_PLAN",
        "RESEARCH_STRATEGY_PREPARE",
        "RESEARCH_STRATEGY_INSTALL",
        "FREEZE",
        "PLAN",
        "PORTFOLIO_UPDATE_PLAN",
        "PORTFOLIO_UPDATE_RUN",
        "RECOVER",
        "TASK_REMEDIATE",
        "RESEARCH_INPUT_CONFIRM",
        "RESEARCH_INPUT_PLAN",
        "RESEARCH_UPDATE_AUTOMATION_CONFIGURE",
        "RESEARCH_UPDATE_PLAN",
        "RESEARCH_UPDATE_RUN",
        "RUN",
        "STORAGE_CONFIRM",
        "STORAGE_EVIDENCE_REBUILD",
        "STORAGE_PIN",
        "STORAGE_CAP_SET",
        "STORAGE_PLAN",
        "STRATEGY_CALIBRATION_PLAN",
        "STRATEGY_CALIBRATION_RUN",
        "STRATEGY_SCORE_PLAN",
        "STRATEGY_SCORE_RUN",
        "UPGRADE_ACKNOWLEDGE",
        "NETWORK_ACCESS_SET",
        "EVIDENCE_CONSENT_SET",
        "EVIDENCE_INSTALL",
        "USAGE_READING_SET",
        "CPU_BUDGET_SET",
        "WORKSPACE_BACKUP",
        "MODEL_ACTIVATE",
        "MODEL_DEACTIVATE",
        "FEATURE_ACTIVATE",
        "FEATURE_DEACTIVATE",
        "STRATEGY_ACTIVATE",
        "STRATEGY_DEACTIVATE",
        "WORKSPACE_PREPARE_CONFIRM",
        "WORKSPACE_PREPARE_PLAN",
        "WAKE_REGISTER",
        "COMMITTEE_OPEN",
        "COMMITTEE_SUBMIT",
    }
)
"""Previews, submissions, confirmations and Task commands: recorded on entry and
on return, whatever their outcome. A preview with no Task is recorded as exactly
that; nothing here admits work or invents a Task."""


RECORDS_ITSELF: frozenset[str] = frozenset({"EVENT_DECLARE", "SESSION_USAGE_READ"})
"""Operations whose own record is what they write: a declared event is its row in the feed,
and a usage reading files its readings, so the observer records no operation beside them
(V266)."""


def observed_operation(operation: str) -> bool:
    """Check whether this operation is recorded on entry and return.

    Args:
        operation: The owner's operation name.

    Returns:
        Whether the operation belongs to the observed set.
    """
    return operation in OBSERVED_OPERATIONS


SUBJECT_FIELDS: tuple[str, ...] = (
    "agent_role",
    "goal_id",
    "goal_hash",
    "analysis_publication_hash",
    "automation_enabled",
    "automation_package_ids",
    "calibration_plan_hash",
    "candidate_hash",
    "candidate_id",
    "component_id",
    "curation_receipt_hash",
    "data_issue_case_token",
    "data_issue_grant_hash",
    "data_issue_evidence_hash",
    "data_issue_option_hash",
    "data_issue_option_id",
    "experiment_kind",
    "experiment_plan_hash",
    "experiment_receipt_hash",
    "experiment_task_id",
    "factor_task_id",
    "formation_session",
    "foundation_admission_hash",
    "handoff_hash",
    "input_binding_hash",
    "input_pinned",
    "left_result_hash",
    "left_task_id",
    "observed_through",
    "origin_task_id",
    "portfolio_session",
    "position_basis",
    "prepared_input_hash",
    "preparation_plan_hash",
    "research_input_id",
    "research_input_plan_hash",
    "result_hash",
    "review_dossier_hash",
    "review_policy_hash",
    "review_publication_hash",
    "review_schema_hash",
    "right_result_hash",
    "right_task_id",
    "risk_report_hash",
    "risk_task_id",
    "score_plan_hash",
    "score_snapshot_hash",
    "storage_plan_hash",
    "strategy_package_id",
    "task_id",
    "update_plan_hash",
    "update_publication_hash",
    "update_task_id",
)
"""Request fields that are references: hashes, ids, dates, enumerations.

Deliberately not here: `spec`, `experiment_document`, `experiment_yaml`,
`experiment_curation`, `review_answer`, `analysis_answer`, `delivery_question`
and `delivery_commentary`. Those are user-authored content and belong to the
owner that received them; activity carries the identity the owner returned.
"""

RETURNED_REFERENCE_FIELDS: tuple[str, ...] = (
    "agent_role",
    "answer_reference",
    "bundle_reference",
    "goal_id",
    "goal_hash",
    "admission_hash",
    "analysis_publication_hash",
    "candidate_hash",
    "cached_result_hash",
    "export_hash",
    "lifecycle",
    "next_action",
    "plan_hash",
    "publication_hash",
    "publication_task_id",
    "report_hash",
    "result_hash",
    "review_key",
    "review_publication_hash",
    "selection_hash",
    "spec_hash",
    "task_id",
)
"""Reference-shaped keys an owner may return; everything else in a response
stays with the caller. `task_id` and `lifecycle` are lifted to their own payload
fields as well, so a reader never has to open `subject` to find the Task."""

_PUBLICATION_READBACK: Mapping[str, str] = {
    "DATA_UPDATE_RUN": "DATA_UPDATE_READBACK",
    "EXPERIMENT_RUN": "EXPERIMENT_READBACK",
    "EXPERIMENT_PROMOTE": "EXPERIMENT_READBACK",
    "EXPERIMENT_CONTINUE": "EXPERIMENT_READBACK",
    "MODEL_TRAINING_INPUT_PREPARE": "MODEL_TRAINING_INPUT_READBACK",
    "RESEARCH_STRATEGY_PREPARE": "RESEARCH_STRATEGY_READBACK",
    "PORTFOLIO_UPDATE_RUN": "PORTFOLIO_UPDATE_READBACK",
    "RESEARCH_INPUT_CONFIRM": "RESEARCH_INPUT_READBACK",
    "RESEARCH_UPDATE_RUN": "RESEARCH_UPDATE_READBACK",
    "STRATEGY_CALIBRATION_RUN": "STRATEGY_CALIBRATION_READBACK",
    "STRATEGY_SCORE_RUN": "STRATEGY_SCORE_READBACK",
    "WORKSPACE_PREPARE_CONFIRM": "WORKSPACE_PREPARE_READBACK",
}


def request_subject(request: PortfolioResearchOperationRequest) -> dict[str, Any]:
    """The references a request named, and only those."""
    subject: dict[str, Any] = {}
    for name in SUBJECT_FIELDS:
        value = getattr(request, name)
        if value is None:
            continue
        if isinstance(value, UUID):
            subject[name] = str(value)
        elif isinstance(value, tuple):
            subject[name] = [str(item) for item in value]
        else:
            subject[name] = value
    return subject


def request_field_names(request: PortfolioResearchOperationRequest) -> list[str]:
    """Which fields were supplied, by name; never their values."""
    return sorted(
        field.name
        for field in fields(request)
        if field.name != "operation" and getattr(request, field.name) is not None
    )


def returned_subject(body: Mapping[str, Any]) -> dict[str, Any]:
    """The references an owner's response named, and only those."""
    subject: dict[str, Any] = {}
    for name in RETURNED_REFERENCE_FIELDS:
        value = body.get(name)
        if isinstance(value, (str, int, bool)):
            subject[name] = value
    intent = body.get("execution_intent")
    if isinstance(intent, Mapping) and isinstance(intent.get("task_id"), str):
        subject["existing_task_id"] = intent["task_id"]
        if isinstance(intent.get("lifecycle"), str):
            subject["existing_task_lifecycle"] = intent["lifecycle"]
    return subject


def returned_status(body: Mapping[str, Any]) -> str | None:
    """The owner's own word for what happened: `status` or `disposition`."""
    for key in ("status", "disposition"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def refusal_code(body: Mapping[str, Any]) -> str | None:
    """The code of an owner's answer that refused its request, or None for any other answer.

    A refusal is an answer whose own word (`status` or `disposition`) begins `REFUSED`; a Task
    readback that reports its Task's failure code is not one.

    Args:
        body: The owner answer.

    Returns:
        Its `failure_code` (or `refused`) when it refused, else None.
    """
    status = returned_status(body)
    if status is None or not status.startswith("REFUSED"):
        return None
    code = body.get("failure_code") or body.get("refused")
    return code if isinstance(code, str) and code else None


def returned_task_id(body: Mapping[str, Any]) -> str | None:
    """Read a nonempty Task identifier from an owner's answer.

    Args:
        body: The owner answer.

    Returns:
        The Task identifier, if supplied as a nonempty string.
    """
    value = body.get("task_id")
    return value if isinstance(value, str) and value else None


def next_read(operation: str, body: Mapping[str, Any]) -> dict[str, str] | None:
    """The one read that safely follows this response, when one exists.

    A Task is read through STATUS; an exact reuse is read through the kind's own
    readback; a result hash is read through REPORT. A preview has no read of its
    own yet (its cache is replaceable until card 32), so it returns nothing.
    """
    task_id = returned_task_id(body)
    if task_id is not None:
        return {"operation": "STATUS", "task_id": task_id}
    published = body.get("publication_task_id")
    if isinstance(published, str) and published:
        return {
            "operation": _PUBLICATION_READBACK.get(operation, "STATUS"),
            "task_id": published,
        }
    result = body.get("result_hash")
    if isinstance(result, str) and result:
        return {"operation": "REPORT", "result_hash": result}
    return None


@dataclass(slots=True)
class OperationSpan:
    """One recorded entry, handed back so the return can be joined to it.

    `operation_ref` is the correlation key of the REQUESTED and RETURNED
    observations; it is the only new identity activity introduces and it
    grants nothing.
    """

    operation: str
    caller: OperationCaller
    operation_ref: str
    subject: dict[str, Any]
    request_fields: list[str]
    started: float


class OperationObserver(Protocol):
    """The seam the shared operation owner calls around every dispatch.

    `entered` returns `None` for operations that are not recorded, and the owner
    then runs exactly as it does without an observer. A span is returned even
    when recording failed, so the return can still be attempted and the
    degradation stays visible to readers instead of silently ending the record.
    """

    def entered(
        self, request: PortfolioResearchOperationRequest, *, caller: OperationCaller
    ) -> OperationSpan | None:
        """Record a request before dispatch and return its correlation span."""
        ...

    def returned(self, span: OperationSpan, body: Mapping[str, Any]) -> None:
        """Record the owner's answer against the entry span."""
        ...

    def failed(self, span: OperationSpan, error: BaseException) -> None:
        """Record a dispatch failure against the entry span."""
        ...

    def refused(self, operation: str, failure_code: str, *, caller: str) -> None:
        """Count one refusal of any operation, reads included (AC, OP14)."""
        ...

    def refusals(self, *, days: int) -> dict[str, object]:
        """Read the refusals counted over the last `days` days."""
        ...

    def read(self, query: ActivityReadQuery) -> dict[str, object]:
        """Read one bounded page of the feed (`activity list`)."""
        ...

    def wake_readiness(self) -> dict[str, object]:
        """Whether this Host can deliver a Codex queue wake."""
        ...

    def recent(self, *, limit: int = 20) -> dict[str, object]:
        """Read the newest requests, by agent session and goal (`activity recent`)."""
        ...

    def admit_external_event(self, document: ExternalActivityEventDocument) -> dict[str, object]:
        """Record one declared event at the external-client trust level (`event declare`)."""
        ...

    def read_external(self, query: ExternalActivityReadQuery) -> dict[str, object]:
        """Read one bounded page of the external-client events, newest last."""
        ...


ACTIVITY_REFUSAL_DAYS = 30
"""How many days of refusal counts `activity refusals` reads when it names none."""
ACTIVITY_DEFAULT_LIMIT = 50
ACTIVITY_MAXIMUM_LIMIT = 200
ACTIVITY_MAXIMUM_WATCH = 16
SUMMARY_RETAINED_CHARACTERS = 500
SUMMARY_MAXIMUM_CHARACTERS = 4000
_EVENT_KIND = r"^[A-Z][A-Z0-9_]{2,63}$"
_PRODUCER_TOKEN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_SUBJECT_KEY = r"^[a-z][a-z0-9_]{0,63}$"


class ExternalActivityEventDocument(BaseModel):  # type: ignore[misc]
    """One event an admitted external client declares about its own work.

    The client names what it observed; the boundary that admits it names how far
    that can be trusted. There is deliberately no field for authority, caller,
    source or trust level: a document that carried one could carry the other.
    `subject.task_id` is verified against Task Control before the event is
    recorded; every other reference is carried as declared.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_kind: str = Field(pattern=_EVENT_KIND)
    producer_id: str = Field(pattern=_PRODUCER_TOKEN)
    producer_session: str = Field(pattern=_PRODUCER_TOKEN)
    producer_sequence: int = Field(ge=0, le=2**63 - 1, strict=True)
    occurred_at: datetime
    summary: str = Field(min_length=1, max_length=SUMMARY_MAXIMUM_CHARACTERS)
    subject: dict[str, str] = Field(default_factory=dict)
    correlation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_document(self) -> Self:
        """Reject out-of-scope event metadata before it can be recorded.

        Returns:
            This validated event document.

        Raises:
            ValueError: If time, subject, correlation, or Task fields are invalid.
        """
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("activity.event_time_not_aware")
        if len(self.subject) > 16:
            raise ValueError("activity.event_subject_too_large")
        for key, value in self.subject.items():
            if re.fullmatch(_SUBJECT_KEY, key) is None:
                raise ValueError("activity.event_subject_key_invalid")
            if not 1 <= len(value) <= 200:
                raise ValueError("activity.event_subject_value_invalid")
        if len(self.correlation_ids) > 8 or any(
            not 1 <= len(item) <= 128 for item in self.correlation_ids
        ):
            raise ValueError("activity.event_correlation_invalid")
        if "task_id" in self.subject:
            try:
                UUID(self.subject["task_id"])
            except ValueError as exc:
                raise ValueError("activity.event_task_id_invalid") from exc
        return self

    def retained_summary(self) -> tuple[str, bool]:
        """The bounded text the ledger keeps, and whether it was cut."""
        text = " ".join(self.summary.split())
        if len(text) <= SUMMARY_RETAINED_CHARACTERS:
            return text, False
        return text[:SUMMARY_RETAINED_CHARACTERS], True


@dataclass(frozen=True, slots=True)
class ActivityReadQuery:
    """A bounded read: where to continue from, how much, and which Tasks to join."""

    after: ObservationReadCursor | None = None
    limit: int = ACTIVITY_DEFAULT_LIMIT
    watch: tuple[UUID, ...] = ()

    @classmethod
    def from_query(cls, query: Mapping[str, list[str]]) -> ActivityReadQuery:
        """Parse bounded feed pagination and Task watches from query values.

        Args:
            query: The parsed URL query parameters.

        Returns:
            The validated read query.

        Raises:
            LocalWebError: If a field, cursor, limit, or watch is invalid.
        """
        unknown = set(query) - {"after", "limit", "watch"}
        if unknown:
            raise LocalWebError("activity.query_field_unknown")
        after = None
        raw_after = _single(query, "after")
        if raw_after:
            try:
                after = ObservationReadCursor.parse(raw_after)
            except ValueError as exc:
                raise LocalWebError("activity.cursor_invalid") from exc
        limit = ACTIVITY_DEFAULT_LIMIT
        raw_limit = _single(query, "limit")
        if raw_limit is not None:
            if not raw_limit.isdigit() or not 1 <= int(raw_limit) <= ACTIVITY_MAXIMUM_LIMIT:
                raise LocalWebError("activity.limit_invalid")
            limit = int(raw_limit)
        watch: tuple[UUID, ...] = ()
        raw_watch = _single(query, "watch")
        if raw_watch:
            parts = [part for part in raw_watch.split(",") if part]
            if len(parts) > ACTIVITY_MAXIMUM_WATCH:
                raise LocalWebError("activity.watch_invalid")
            try:
                watch = tuple(dict.fromkeys(UUID(part) for part in parts))
            except ValueError as exc:
                raise LocalWebError("activity.watch_invalid") from exc
        return cls(after=after, limit=limit, watch=watch)


@dataclass(frozen=True, slots=True)
class ExternalActivityReadQuery:
    """Describe a page of a declared session's external-client events.

    The readback includes every stored ``ExternalActivityObserved``, not just
    the feed's tail. ``before`` is the ordinal preceding the page in hand.
    """

    limit: int = ACTIVITY_MAXIMUM_LIMIT
    before: int | None = None
    observation_id: str | None = None

    @classmethod
    def from_query(cls, query: Mapping[str, list[str]]) -> ExternalActivityReadQuery:
        """Parse a bounded external-event page request.

        Args:
            query: The parsed URL query parameters.

        Returns:
            The validated page limit and older-page cursor.

        Raises:
            LocalWebError: If a field, limit, or cursor is invalid.
        """
        unknown = set(query) - {"limit", "before", "observation_id"}
        if unknown:
            raise LocalWebError("activity.query_field_unknown")
        observation_id = _single(query, "observation_id")
        if observation_id is not None:
            if re.fullmatch(r"[0-9a-f]{64}", observation_id) is None:
                raise LocalWebError("activity.observation_id_invalid")
            if "limit" in query or "before" in query:
                raise LocalWebError("activity.query_selection_conflict")
            return cls(observation_id=observation_id)
        limit = ACTIVITY_MAXIMUM_LIMIT
        raw_limit = _single(query, "limit")
        if raw_limit is not None:
            if not raw_limit.isdigit() or not 1 <= int(raw_limit) <= ACTIVITY_MAXIMUM_LIMIT:
                raise LocalWebError("activity.limit_invalid")
            limit = int(raw_limit)
        before = None
        raw_before = _single(query, "before")
        if raw_before is not None:
            if not raw_before.isdigit() or int(raw_before) < 1:
                raise LocalWebError("activity.cursor_invalid")
            before = int(raw_before)
        return cls(limit=limit, before=before)


def _single(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key) or []
    if not values:
        return None
    if len(values) != 1:
        raise LocalWebError(f"activity.query_parameter_invalid:{key}")
    return values[0]


def activity_item(item: ObservationReadItem) -> dict[str, Any]:
    """One observation as a consumer sees it: envelope facts and its safe payload.

    The payload is `None` when retention removed it; the row keeps its place so
    a reader that was away sees the gap where it happened.
    """
    envelope = item.envelope
    return {
        "ordinal": item.ordinal,
        "observation_id": envelope.observation_id,
        "occurred_at": envelope.occurred_at.isoformat(),
        "observed_at": envelope.observed_at.isoformat(),
        "schema_kind": envelope.schema_kind,
        "source_kind": envelope.source_kind,
        "source_id": envelope.source_id,
        "source_sequence": envelope.source_sequence,
        "task_id": envelope.task_id,
        "run_id": envelope.run_id,
        "stage_id": envelope.stage_id,
        "correlation_ids": list(envelope.correlation_ids),
        "authority": envelope.authority.value,
        "retention_class": envelope.retention_class.value,
        "availability": envelope.availability.value,
        "payload": envelope.inline_safe_payload,
    }


__all__ = [
    "ACTIVITY_DEFAULT_LIMIT",
    "ACTIVITY_MAXIMUM_LIMIT",
    "ACTIVITY_MAXIMUM_WATCH",
    "ACTIVITY_REFUSAL_DAYS",
    "FAILURE_DETAIL_WITHHELD",
    "OBSERVED_OPERATIONS",
    "READ_OPERATIONS",
    "RECORDS_ITSELF",
    "RETURNED_REFERENCE_FIELDS",
    "SUBJECT_FIELDS",
    "SUMMARY_RETAINED_CHARACTERS",
    "ActivityReadQuery",
    "ExternalActivityEventDocument",
    "ExternalActivityReadQuery",
    "OperationObserver",
    "OperationSpan",
    "activity_item",
    "next_read",
    "observed_operation",
    "refusal_code",
    "request_field_names",
    "request_subject",
    "returned_status",
    "returned_subject",
    "returned_task_id",
]
