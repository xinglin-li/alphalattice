"""Goals: what an agent is asked to achieve, the Host's record of it and its checked submission.

No numerical or execution authority (LAWS OP13): an agent's own goal mode runs a goal; the Host
keeps the declaration, records what the sessions bound to it did, and checks the submission
against that record, sealing it or naming each missing item.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Final, Literal, Self, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type ResearchStage = Literal[
    "DATA_FEATURES", "FACTOR_FOUNDATION", "ALPHA", "RISK", "PORTFOLIO", "EVIDENCE_CRO"
]
type GoalKind = Literal["RESEARCH", "DATA", "OPERATIONS", "REVIEW", "FIRST_USE"]
_ID = r"^[a-z][a-z0-9_-]{0,63}$"
_HASH = r"^[0-9a-f]{64}$"

FIRST_USE_HOURS: Final = 24
"""How long a first-use goal's delegation lasts from its opening (V452)."""


class GoalContract(BaseModel):  # type: ignore[misc]
    """Frozen, closed-field base for a goal's records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _distinct(values: tuple[Any, ...], field: str) -> None:
    names = [getattr(v, field) for v in values]
    if len(set(names)) != len(names):
        raise ValueError("goal.duplicate_identifier")


class GoalBudget(GoalContract):
    """Declared bounds, shown beside what was used; never a permission grant."""

    maximum_tasks: int = Field(ge=0, le=100)
    maximum_numerical_calls: int = Field(ge=0, le=1_000_000)
    maximum_model_calls: int = Field(ge=0, le=100)


class GoalCriterion(GoalContract):
    """One completion criterion: a sentence the submission answers with evidence."""

    criterion_id: str = Field(pattern=_ID)
    text: str = Field(min_length=1, max_length=600)


class GoalDeliverable(GoalContract):
    """A slot the submission fills with references."""

    deliverable_id: str = Field(pattern=_ID)
    kind: Literal["RESULT", "REPORT", "FINDINGS", "DECISION"]
    description: str = Field(min_length=1, max_length=600)
    required: bool = True


class GoalResearchDesign(GoalContract):
    """A research goal's design: its purpose, its comparison and the stages it must evidence."""

    purpose: Literal["NEW_RESEARCH", "EXISTING_RESULTS", "CONTINUATION"]
    comparison_design: str = Field(min_length=1, max_length=4000)
    required_stages: tuple[ResearchStage, ...] = Field(min_length=1, max_length=6)


class GoalDeclaration(GoalContract):
    """What is to be achieved and what completes it, fixed per revision."""

    title: str = Field(min_length=1, max_length=180)
    objective: str = Field(min_length=1, max_length=2400)
    kind: GoalKind
    scope: str = Field(default="", max_length=4000)
    constraints: tuple[str, ...] = Field(default=(), max_length=30)
    criteria: tuple[GoalCriterion, ...] = Field(min_length=1, max_length=20)
    deliverables: tuple[GoalDeliverable, ...] = Field(default=(), max_length=20)
    budget: GoalBudget | None = None
    research: GoalResearchDesign | None = None
    parent_goal_id: UUID | None = None
    target_date: date | None = Field(
        default=None,
        description="The date the person named for the positions: they are entered on the first "
        "XNYS/XNAS session on or after it, decided at the close before.",
    )

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def _serialize_named_date(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep an absent date absent, so every goal sealed before the field keeps its hash."""
        serialized: dict[str, Any] = handler(self)
        if self.target_date is None:
            serialized.pop("target_date", None)
        return serialized

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _consistent(self) -> Self:
        _distinct(self.criteria, "criterion_id")
        _distinct(self.deliverables, "deliverable_id")
        if self.research is not None and self.kind != "RESEARCH":
            raise ValueError("goal.research_design_needs_research_kind")
        return self

    def intent(self) -> dict[str, Any]:
        """The declared intent without its display title."""
        return cast(dict[str, Any], self.model_dump(mode="json", exclude={"title"}))


def update_offers(update: object, named: date | None, task: object) -> dict[str, object]:
    """An activation's update offers: its status once admitted, else the update to plan.

    Under a first use the update keeps the goal's named date, which never changes, and a date no
    session reads offers none, since the latest session's update is not the goal's.
    """
    if task:
        return {"update_status": {"operation": "STATUS", "task_id": str(task)}}
    if named is None or not isinstance(update, dict):
        return {} if update is None else {"update": update}
    try:
        formation = str(target_sessions(named)["formation_session"])
    except ValueError:
        return {}
    return {"update": {**update, "observed_through": formation}}


def target_sessions(named: date) -> dict[str, object]:
    """The sessions a named date's positions stand on, from the exchange calendars alone.

    Args:
        named: The date the person named.

    Returns:
        The named date, whether it is a session, the entry session (the first common XNYS/XNAS
        session on or after it) and the formation session before it, whose close is the
        information cutoff.

    Raises:
        ValueError: `first_use.date_outside_calendar` for a date the calendars do not plan.
    """
    # The calendars load only for a dated first use, never with every client command.
    from alphalattice.foundation.causal_outcomes.execution.readers import (
        planned_local_qa_schedule,
    )

    try:
        schedule = planned_local_qa_schedule(named - timedelta(days=14), named)
    except (ValueError, KeyError, LookupError) as error:
        raise ValueError("first_use.date_outside_calendar") from error
    point = next((p for p in schedule if p.entry_session >= named), None)
    if point is None:
        raise ValueError("first_use.date_outside_calendar")
    return {
        "named_date": named.isoformat(),
        "named_is_session": point.entry_session == named,
        "entry_session": point.entry_session.isoformat(),
        "formation_session": point.formation_session.isoformat(),
        "information_cutoff_at": point.formation_close_at.isoformat(),
    }


class GoalReferenceRequest(GoalContract):
    """A proposed link from a goal to an exact product read."""

    reference_id: str = Field(pattern=_ID)
    label: str = Field(min_length=1, max_length=180)
    stage: ResearchStage
    request: dict[str, Any]


class GoalReference(GoalReferenceRequest):
    """A reference sealed with its identity and timing relationship."""

    identity: dict[str, Any]
    reference_hash: str = Field(pattern=_HASH)
    intent_relation: Literal[
        "QUESTION_RECORDED_BEFORE_TASK_ADMISSION", "POST_HOC", "NO_EXECUTION_TIME_PROOF"
    ]


class GoalFile(GoalContract):
    """A text file the agent hands in, kept by its content hash."""

    reference_id: str = Field(pattern=_ID)
    name: str = Field(min_length=1, max_length=180)
    media_type: Literal["text/markdown", "text/plain", "application/json"]
    text: str = Field(min_length=1, max_length=200_000)


class GoalCriterionAnswer(GoalContract):
    """The agent's answer to one criterion, with the references that support it."""

    criterion_id: str = Field(pattern=_ID)
    answer: Literal["MET", "NOT_MET", "NOT_ASSESSED"]
    evidence: tuple[str, ...] = Field(default=(), max_length=32)
    note: str = Field(default="", max_length=2400)


class GoalDeliverableAnswer(GoalContract):
    """What fills one deliverable slot."""

    deliverable_id: str = Field(pattern=_ID)
    references: tuple[str, ...] = Field(min_length=1, max_length=32)


class GoalFinding(GoalContract):
    """One finding, with its evidence."""

    finding_id: str = Field(pattern=_ID)
    text: str = Field(min_length=1, max_length=2400)
    evidence: tuple[str, ...] = Field(default=(), max_length=32)


class GoalProblem(GoalContract):
    """What stayed unresolved, naming the Tasks that failed on the way."""

    problem_id: str = Field(pattern=_ID)
    text: str = Field(min_length=1, max_length=2400)
    task_ids: tuple[UUID, ...] = Field(default=(), max_length=64)


class GoalSubmission(GoalContract):
    """The agent's completion: its judgment only (OP11); the Host adds the facts."""

    outcome: Literal["ACHIEVED", "PARTLY_ACHIEVED", "NOT_ACHIEVED"]
    summary: str = Field(min_length=1, max_length=4000)
    criteria: tuple[GoalCriterionAnswer, ...] = Field(min_length=1, max_length=20)
    deliverables: tuple[GoalDeliverableAnswer, ...] = Field(default=(), max_length=20)
    references: tuple[GoalReferenceRequest, ...] = Field(default=(), max_length=32)
    files: tuple[GoalFile, ...] = Field(default=(), max_length=10)
    findings: tuple[GoalFinding, ...] = Field(default=(), max_length=32)
    problems: tuple[GoalProblem, ...] = Field(default=(), max_length=32)
    follow_ups: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _distinct_ids(self) -> Self:
        for values, field in (
            (self.criteria, "criterion_id"),
            (self.deliverables, "deliverable_id"),
            (self.references, "reference_id"),
            (self.files, "reference_id"),
            (self.findings, "finding_id"),
            (self.problems, "problem_id"),
        ):
            _distinct(values, field)
        return self


class GoalStatement(GoalContract):
    """An attributed decision or conclusion with explicit evidence standing."""

    statement_id: str = Field(pattern=_ID)
    kind: Literal["DECISION", "CONCLUSION", "PM_RESPONSE"]
    attribution: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=12000)
    evidence: tuple[str, ...] = Field(default=(), max_length=64)
    disposition: Literal["SUPPORTS", "DOES_NOT_SUPPORT", "INSUFFICIENT", "OPEN", "REJECTED"]
    responds_to: str | None = Field(default=None, max_length=64)


class GoalSession(GoalContract):
    """An agent session, as its vendor names it: provenance, never identity (ID6)."""

    vendor: Literal["claude-code", "codex"]
    session_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")


class GoalAcceptedAnswerContext(GoalContract):
    """The original Goal frozen at genuine first acceptance.

    The assignment fields read contexts written before FLOW-1, when a lead's assignment
    message could close at acceptance; nothing writes them now.
    """

    goal_id: UUID | None
    assignment_packet_hash: str | None = Field(default=None, pattern=_HASH)
    assignment_diagnostic: (
        Literal["native_bridge.assignment_not_observed", "native_bridge.assignment_ambiguous"]
        | None
    ) = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def exact_or_missing(self) -> Self:
        """An exact captured packet and a missing-link diagnostic cannot both be asserted."""
        if self.assignment_packet_hash is not None and self.assignment_diagnostic is not None:
            raise ValueError("goal.assignment_closure_invalid")
        return self


class GoalAcceptedAnswerReceipt(GoalAcceptedAnswerContext):
    """An owner-resolved accepted answer link, never a public event's assertion.

    The scientific owner has already filed the answer and its Task. The original Goal is
    retained separately before optional observation; an absent old receipt stays unknown.
    It files the accepted answer under that Goal without claiming native authorship (OP13, ID7).
    """

    session: GoalSession
    task_id: UUID
    bundle_reference: str = Field(pattern=_HASH)
    answer_reference: str = Field(pattern=_HASH)
    bundle_role: Literal["ALPHA", "ANALYST", "CRO", "DATA", "FACTOR", "PORTFOLIO", "RISK"]
    verdict: Literal["ACCEPTED", "DONE"]


class GoalTaskFact(GoalContract):
    """One Task a bound session started, as Task Control holds it."""

    task_id: UUID
    kind: str = Field(min_length=1, max_length=200)
    state: str = Field(min_length=1, max_length=64)
    updated_at: datetime | None = Field(
        default=None, description="Task Control's canonical last-update time, when available."
    )

    @model_serializer(mode="wrap")  # type: ignore[untyped-decorator]
    def _serialize_available_update(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep an absent clock absent, preserving the pre-field sealed Goal shape."""
        serialized: dict[str, Any] = handler(self)
        if self.updated_at is None:
            serialized.pop("updated_at", None)
        return serialized

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def update_time_is_aware(self) -> Self:
        """A present Task Control update clock identifies an unambiguous instant."""
        if self.updated_at is not None and (
            self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None
        ):
            raise ValueError("goal.task_updated_at_must_be_aware")
        return self


class GoalCompletion(GoalContract):
    """The Host's own facts when it sealed the goal: what the agent never writes."""

    checked_at: datetime
    sessions: tuple[GoalSession, ...] = Field(default=(), max_length=64)
    tasks: tuple[GoalTaskFact, ...] = Field(default=(), max_length=1000)
    request_count: int = Field(ge=0)


class Goal(GoalContract):
    """One immutable revision of a goal: its declaration, its attributed record and its end."""

    kind: Literal["Goal"] = "Goal"
    goal_id: UUID
    workspace_id: str
    parent_hash: str | None = Field(default=None, pattern=_HASH)
    revision: int = Field(ge=1)
    recorded_at: datetime
    intent_registered_at: datetime
    submitted_by: str
    change_reason: str = Field(min_length=1, max_length=2400)
    declaration: GoalDeclaration
    state: Literal["OPEN", "COMPLETE", "ABANDONED"] = "OPEN"
    references: tuple[GoalReference, ...] = Field(default=(), max_length=64)
    statements: tuple[GoalStatement, ...] = Field(default=(), max_length=64)
    submission: GoalSubmission | None = None
    completion: GoalCompletion | None = None
    goal_hash: str = Field(pattern=_HASH)

    def outcome(self) -> str:
        """Whether this revision records an attributed conclusion (the case page's word)."""
        return (
            "ATTRIBUTED_CONCLUSION_RECORDED"
            if any(s.kind == "CONCLUSION" for s in self.statements)
            else "QUESTION_OPEN"
        )

    @classmethod
    def seal(cls, **values: Any) -> Self:
        """Seal a revision with the canonical hash of its payload."""
        payload = cls.model_construct(**values, goal_hash="0" * 64).model_dump(
            mode="json", exclude={"goal_hash"}
        )
        return cls(**payload, goal_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def identity(self) -> Self:
        """Verify the revision's hash, clocks, identifiers and end."""
        if any(
            v.tzinfo is None or v.utcoffset() is None
            for v in (self.recorded_at, self.intent_registered_at)
        ):
            raise ValueError("goal.clock_invalid")
        if self.goal_hash != canonical_hash(self.model_dump(mode="json", exclude={"goal_hash"})):
            raise ValueError("goal.identity_mismatch")
        for values, field in (
            (self.references, "reference_id"),
            (self.statements, "statement_id"),
        ):
            _distinct(values, field)
        sealed = self.submission is not None and self.completion is not None
        if (self.state == "COMPLETE") != sealed:
            raise ValueError("goal.completion_inconsistent")
        return self
