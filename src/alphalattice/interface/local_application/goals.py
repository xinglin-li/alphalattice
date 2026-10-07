"""Goals: what an agent is asked to achieve, the Host's record of it and its checked submission.

No numerical or execution authority (LAWS OP13): an agent's own goal mode runs a goal; the Host
keeps the declaration, records what the sessions bound to it did, and checks the submission
against that record, sealing it or naming each missing item.
"""

from __future__ import annotations

from datetime import datetime
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
