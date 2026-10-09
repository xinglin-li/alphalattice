"""Host-owned durable task, work-board, and recovery contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)
from pydantic_core import to_jsonable_python

from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _hash(value: object) -> str:
    return canonical_hash(to_jsonable_python(value))


FAILURE_CODE_MAX_LENGTH = 120
"""The longest failure code a Task or work item records."""


def failure_code_from(error: BaseException) -> str:
    """The code a stage records for a refusal it caught, fitted to the contract.

    A contract that refuses raises a pydantic ``ValidationError`` whose text is
    a multi-line report; the code inside it is the message of its first error
    (``Value error, risk_research.covariance_chunk_identity_invalid``). Any
    other error is its own text. Either is cut to the length the work-item and
    Task records accept, because a code that does not fit is not recorded as a
    block at all -- the write fails and the runner classifies the whole stage as
    an interruption, which invites a recovery that would only block again.
    """
    text = str(error)
    if isinstance(error, ValidationError):
        errors = error.errors()
        if errors:
            message = str(errors[0].get("msg", "")).strip()
            text = message.removeprefix("Value error, ").strip() or text
    return text[:FAILURE_CODE_MAX_LENGTH]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StageFailureCause(ContractModel):
    """What a stage's owner saw when its work failed, kept beside the stable code.

    The code names the refusal; the cause says what was raised, in which step and on which
    unit of work, so a reader can tell an exhausted machine (a `MemoryError`, a paging file
    too small) from a defect. It holds only what the owner already keeps in its own outcome.

    Attributes:
        exception_type: The raised type's name.
        detail: The message, whitespace collapsed and cut to 400 characters.
        step: The owner's step that raised it.
        unit: The unit of work in hand, such as a listing.
        first_session: The first session that unit was computing.
        last_session: The last session that unit was computing.
        row_count: The source's recorded row count, or its recorded unknown marker.
        sanitizer_code: The source sanitizer's recorded subcode, when supplied.
    """

    exception_type: str = Field(min_length=1, max_length=120)
    detail: str = Field(default="", max_length=400)
    step: str | None = Field(default=None, max_length=120)
    unit: str | None = Field(default=None, max_length=160)
    first_session: date | None = None
    last_session: date | None = None
    row_count: Annotated[int, Field(ge=0, strict=True)] | Literal["UNKNOWN"] | None = None
    sanitizer_code: str | None = Field(default=None, max_length=120)

    @model_serializer(mode="wrap")
    def serialize_recorded_facts(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep causes recorded before source facts byte-equivalent and hash-compatible."""
        fields = handler(self)
        for key in ("row_count", "sanitizer_code"):
            if fields.get(key) is None:
                fields.pop(key, None)
        return fields

    @classmethod
    def from_facts(cls, facts: Mapping[str, object] | None) -> StageFailureCause | None:
        """The cause an owner's facts name, fitted to the record; None when they name none.

        A cause rides beside a block and never stops it being recorded: facts without an
        exception type, or that do not read as a cause, give none, and a long message or
        name is cut to what the record accepts.

        Args:
            facts: The owner's failure facts under this model's field names.

        Returns:
            The fitted cause, or None.
        """
        if not facts:
            return None

        def text(key: str, limit: int) -> str | None:
            value = facts.get(key)
            words = " ".join(str(value).split()) if value is not None else ""
            return words[:limit] or None

        kind = text("exception_type", 120)
        if kind is None:
            return None
        try:
            return cls.model_validate(
                {
                    "exception_type": kind,
                    "detail": text("detail", 400) or "",
                    "step": text("step", 120),
                    "unit": text("unit", 160),
                    "first_session": facts.get("first_session"),
                    "last_session": facts.get("last_session"),
                    "row_count": facts.get("row_count"),
                    "sanitizer_code": text("sanitizer_code", 120),
                }
            )
        except ValueError:
            return None


def _cause_absent(identity: dict[str, Any]) -> dict[str, Any]:
    """Omit an absent failure cause so states sealed without that field still verify."""
    if identity.get("failure_cause") is None:
        return {key: value for key, value in identity.items() if key != "failure_cause"}
    return identity


class TaskLifecycle(StrEnum):
    """Record queued, executing, deferred, review, cancellation, terminal and recovery states."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DEFERRED = "DEFERRED"
    REVIEW_PENDING = "REVIEW_PENDING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class WorkItemLifecycle(StrEnum):
    """Track pending/executing work through verification, block or cancellation."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    READY_FOR_VERIFICATION = "READY_FOR_VERIFICATION"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"


class TaskCommandKind(StrEnum):
    """Identify a cancellation request whose enactment belongs to the task owner."""

    CANCEL_REQUESTED = "CANCEL_REQUESTED"


class TaskEvidence(ContractModel):
    """Identify one kind of stage evidence by retained reference and content commitment.

    Attributes:
        evidence_kind: Declared evidence category.
        reference: Bounded retained evidence reference.
        content_hash: Commitment to the evidence content.
    """

    evidence_kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,119}$")
    reference: str = Field(min_length=1, max_length=2048)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class TaskInputEnvelope(ContractModel):
    """Seal declared domain input values under their task kind and input schema.

    Attributes:
        task_kind: Installed task domain.
        input_schema_id: Declared domain input schema.
        payload: Retained domain input values.
        input_hash: Canonical envelope identity excluding this hash.
    """

    task_kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,79}$")
    input_schema_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,119}$")
    payload: dict[str, Any]
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskInputEnvelope:
        """Validate the declared TaskInputEnvelope invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Canonical envelope identity differs.
        """
        identity = self.model_dump(mode="json", exclude={"input_hash"})
        if self.input_hash != _hash(identity):
            raise ValueError("task input envelope hash is invalid")
        return self

    @classmethod
    def create(
        cls, *, task_kind: str, input_schema_id: str, payload: dict[str, Any]
    ) -> TaskInputEnvelope:
        """Compute canonical identity and construct the validated TaskInputEnvelope.

        Args:
            task_kind: Declared installed task domain identifier.
            input_schema_id: Declared input schema identifier.
            payload: Domain input values retained in the envelope.

        Returns:
            The sealed and validated immutable record.
        """
        identity = {
            "task_kind": task_kind,
            "input_schema_id": input_schema_id,
            "payload": payload,
        }
        return cls(**identity, input_hash=_hash(identity))


class ResearchGoal(ContractModel):
    """Seal a declared research deliverable and its input binding.

    Attributes:
        goal_kind: Declared research goal identifier.
        input_hash: Sealed task input identity.
        deliverable_kind: Declared result category.
        summary: Bounded goal summary.
        attributes: Domain-owned goal metadata.
        goal_hash: Canonical goal identity excluding this hash.
    """

    goal_kind: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,119}$")
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    deliverable_kind: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{2,119}$")
    summary: str = Field(min_length=1, max_length=500)
    attributes: dict[str, Any] = Field(default_factory=dict)
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchGoal:
        """Validate the declared ResearchGoal invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Canonical goal identity differs.
        """
        identity = self.model_dump(mode="json", exclude={"goal_hash"})
        if self.goal_hash != _hash(identity):
            raise ValueError("research goal hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        goal_kind: str,
        input_hash: str,
        deliverable_kind: str,
        summary: str,
        attributes: dict[str, Any] | None = None,
    ) -> ResearchGoal:
        """Compute canonical identity and construct the validated ResearchGoal.

        Args:
            goal_kind: Declared research goal identifier.
            input_hash: Canonical task input identity.
            deliverable_kind: Declared deliverable identifier.
            summary: Bounded goal summary.
            attributes: Optional domain-owned goal metadata.

        Returns:
            The sealed and validated immutable record.
        """
        identity = {
            "goal_kind": goal_kind,
            "input_hash": input_hash,
            "deliverable_kind": deliverable_kind,
            "summary": summary,
            "attributes": attributes or {},
        }
        return cls(**identity, goal_hash=_hash(identity))


class WorkItemDefinition(ContractModel):
    """Seal one stage, its prior dependencies, verifier and required evidence kinds.

    Attributes:
        stage_id: Unique stage identifier within the plan.
        dependency_ids: Ordered distinct predecessor identifiers.
        verifier_id: Installed stage verifier.
        required_evidence_kinds: Evidence categories required for verification.
        cancellation_boundary: Whether this stage declares a safe cancellation boundary.
        definition_hash: Canonical definition identity excluding this hash.
    """

    stage_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")
    dependency_ids: tuple[str, ...] = ()
    verifier_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,119}$")
    required_evidence_kinds: tuple[str, ...] = ()
    cancellation_boundary: bool = True
    definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> WorkItemDefinition:
        """Validate the declared WorkItemDefinition invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: A dependency repeats or refers to the same stage, or definition identity
                differs.
        """
        if self.stage_id in self.dependency_ids:
            raise ValueError("work item cannot depend on itself")
        if self.dependency_ids != tuple(dict.fromkeys(self.dependency_ids)):
            raise ValueError("work item dependencies must be unique")
        identity = self.model_dump(mode="json", exclude={"definition_hash"})
        if self.definition_hash != _hash(identity):
            raise ValueError("work item definition hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        stage_id: str,
        dependency_ids: tuple[str, ...],
        verifier_id: str,
        required_evidence_kinds: tuple[str, ...] = (),
        cancellation_boundary: bool = True,
    ) -> WorkItemDefinition:
        """Compute canonical identity and construct the validated WorkItemDefinition.

        Args:
            stage_id: Exact work-stage identifier in the sealed plan.
            dependency_ids: Ordered distinct stage dependencies.
            verifier_id: Installed verifier declared for the stage.
            required_evidence_kinds: Evidence kinds the verifier receipt must carry.
            cancellation_boundary: Whether the stage declares a safe cancellation boundary.

        Returns:
            The sealed and validated immutable record.
        """
        identity = {
            "stage_id": stage_id,
            "dependency_ids": dependency_ids,
            "verifier_id": verifier_id,
            "required_evidence_kinds": required_evidence_kinds,
            "cancellation_boundary": cancellation_boundary,
        }
        return cls(**identity, definition_hash=_hash(identity))


PLAN_WORK_ITEM_LIMIT: Final = 4096
"""The most work items one plan carries: the widest plan a consumer
declares. A book-wide evidence coverage run carries one set of its
stages per unit, and its own contract admits 512 units of eight stages;
a consumer that needs more declares a successor here, never a second
plan for one Task. The bound admits; the plan hash binds the items."""


class ResearchPlan(ContractModel):
    """Seal a bounded dependency-ordered work graph against goal and installed verifier bindings.

    Attributes:
        goal_hash: Declared goal identity.
        workflow_definition_hash: Installed workflow definition identity.
        verifier_catalog_hash: Installed verifier catalog identity.
        work_items: Nonempty bounded stages in dependency order.
        plan_hash: Canonical plan identity excluding this hash.
    """

    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    work_items: tuple[WorkItemDefinition, ...] = Field(
        min_length=1, max_length=PLAN_WORK_ITEM_LIMIT
    )
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> ResearchPlan:
        """Validate the declared ResearchPlan invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Stages repeat, dependencies do not precede their children, or plan identity
                differs.
        """
        stage_ids = tuple(item.stage_id for item in self.work_items)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("research plan stage IDs must be unique")
        seen: set[str] = set()
        for item in self.work_items:
            if any(dependency not in seen for dependency in item.dependency_ids):
                raise ValueError("research plan dependencies must precede their child")
            seen.add(item.stage_id)
        identity = self.model_dump(mode="json", exclude={"plan_hash"})
        if self.plan_hash != _hash(identity):
            raise ValueError("research plan hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        goal_hash: str,
        workflow_definition_hash: str,
        verifier_catalog_hash: str,
        work_items: tuple[WorkItemDefinition, ...],
    ) -> ResearchPlan:
        """Compute canonical identity and construct the validated ResearchPlan.

        Args:
            goal_hash: Canonical research goal identity.
            workflow_definition_hash: Installed workflow definition commitment.
            verifier_catalog_hash: Installed verifier catalog commitment.
            work_items: Dependency-ordered sealed work definitions.

        Returns:
            The sealed and validated immutable record.
        """
        identity = {
            "goal_hash": goal_hash,
            "workflow_definition_hash": workflow_definition_hash,
            "verifier_catalog_hash": verifier_catalog_hash,
            "work_items": work_items,
        }
        return cls(**identity, plan_hash=_hash(identity))


class TaskRecord(ContractModel):
    """Seal durable task state with a consistent input, goal, plan and active-stage binding.

    Attributes:
        task_id: Task UUID.
        task_kind: Domain matching the sealed input envelope.
        input: Sealed domain input.
        goal: Sealed goal bound to that input.
        plan: Sealed plan bound to that goal.
        lifecycle: Durable task state.
        active_work_item_id: Optional active stage within the sealed plan.
        latest_execution_id: Optional latest execution UUID.
        admitted_at: Recorded admission clock.
        started_at: Optional initial execution clock.
        updated_at: Last task update clock.
        failure_code: Optional stable failure cause.
        version: Positive durable record version.
        record_hash: Canonical task-record identity.
    """

    task_id: UUID
    task_kind: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,79}$")
    input: TaskInputEnvelope
    goal: ResearchGoal
    plan: ResearchPlan
    lifecycle: TaskLifecycle
    active_work_item_id: str | None = None
    latest_execution_id: UUID | None = None
    admitted_at: datetime
    started_at: datetime | None = None
    updated_at: datetime
    failure_code: str | None = Field(default=None, max_length=120)
    version: int = Field(ge=1)
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskRecord:
        """Validate the declared TaskRecord invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Task/input/goal/plan bindings disagree, active stage is outside the plan, or
                identity differs.
        """
        if self.task_kind != self.input.task_kind:
            raise ValueError("task kind differs from its input envelope")
        if self.goal.input_hash != self.input.input_hash:
            raise ValueError("task goal differs from its input envelope")
        if self.plan.goal_hash != self.goal.goal_hash:
            raise ValueError("task plan differs from its goal")
        if self.active_work_item_id is not None and self.active_work_item_id not in {
            item.stage_id for item in self.plan.work_items
        }:
            raise ValueError("task active work item is outside its plan")
        identity = self.model_dump(mode="json", exclude={"record_hash"})
        if self.record_hash != _hash(identity):
            raise ValueError("task record hash is invalid")
        return self

    @classmethod
    def from_identity(cls, **identity: Any) -> TaskRecord:
        """Compute canonical identity and construct the validated TaskRecord.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, record_hash=_hash(identity))


class TaskRecoveryLink(ContractModel):
    """Seal an operational link from one stopped Task version to an owner replan.

    The request is the owner's normalized replan request before recovery context is
    added. A missing successor records a preview only; it cannot establish that work
    was admitted or completed.
    """

    source_task_id: UUID
    source_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    admission_request: dict[str, Any]
    successor_task_id: UUID | None = None
    recorded_at: datetime
    link_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskRecoveryLink:
        """Validate the link seal and its operational request boundary."""
        if self.recorded_at.tzinfo is None or self.recorded_at.utcoffset() is None:
            raise ValueError("task recovery link clock must be timezone-aware")
        operation = self.admission_request.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ValueError("task recovery link request must name an operation")
        if {"recovery_task_id", "recovery_task_hash"} & self.admission_request.keys():
            raise ValueError("task recovery link request must omit recovery context")
        if self.successor_task_id == self.source_task_id:
            raise ValueError("task recovery link successor must differ from its source")
        identity = self.model_dump(mode="json", exclude={"link_hash"})
        if self.link_hash != _hash(identity):
            raise ValueError("task recovery link hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        source_task_id: UUID,
        source_record_hash: str,
        admission_request: dict[str, Any],
        successor_task_id: UUID | None,
        recorded_at: datetime,
    ) -> TaskRecoveryLink:
        """Compute the canonical seal for one preview or confirmed admission."""
        identity = {
            "source_task_id": source_task_id,
            "source_record_hash": source_record_hash,
            "admission_request": admission_request,
            "successor_task_id": successor_task_id,
            "recorded_at": recorded_at,
        }
        return cls(**identity, link_hash=_hash(identity))


class WorkItemState(ContractModel):
    """Seal stage state, attempts and unique retained evidence references for one task.

    Attributes:
        task_id: Owning task UUID.
        stage_id: Work-stage identifier.
        definition_hash: Sealed stage-definition identity.
        lifecycle: Recorded stage state.
        attempt_count: Nonnegative stage execution count.
        evidence: Unique retained evidence references.
        failure_code: Optional stable failure cause.
        failure_cause: The owner's recorded explanation of that failure.
        started_at: Optional first stage execution clock.
        updated_at: Last stage-state clock.
        version: Positive state version.
        state_hash: Canonical stage-state identity.
    """

    task_id: UUID
    stage_id: str
    definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycle: WorkItemLifecycle
    attempt_count: int = Field(ge=0)
    evidence: tuple[TaskEvidence, ...] = ()
    failure_code: str | None = Field(default=None, max_length=120)
    failure_cause: StageFailureCause | None = None
    started_at: datetime | None = None
    updated_at: datetime
    version: int = Field(ge=1)
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> WorkItemState:
        """Validate the declared WorkItemState invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Evidence references repeat or canonical stage-state identity differs.
        """
        references = tuple(item.reference for item in self.evidence)
        if len(set(references)) != len(references):
            raise ValueError("work item evidence references must be unique")
        identity = _cause_absent(self.model_dump(mode="json", exclude={"state_hash"}))
        if self.state_hash != _hash(identity):
            raise ValueError("work item state hash is invalid")
        return self

    @classmethod
    def from_identity(cls, **identity: Any) -> WorkItemState:
        """Compute canonical identity and construct the validated WorkItemState.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, state_hash=_hash(_cause_absent(identity)))


class TaskExecutionCompatibility(ContractModel):
    """Seal contract, workflow, schema, domain policy and framework execution bindings.

    Attributes:
        task_contract_hash: Installed Task contract identity.
        workflow_definition_hash: Installed workflow definition identity.
        input_schema_id: Domain input schema identifier.
        domain_policy_hash: Installed domain policy identity.
        framework_identity_hash: Installed execution framework identity.
        compatibility_hash: Canonical binding identity.
    """

    task_contract_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_schema_id: str
    domain_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    framework_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    compatibility_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskExecutionCompatibility:
        """Validate the declared TaskExecutionCompatibility invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Canonical execution compatibility identity differs.
        """
        identity = self.model_dump(mode="json", exclude={"compatibility_hash"})
        if self.compatibility_hash != _hash(identity):
            raise ValueError("task execution compatibility hash is invalid")
        return self

    @classmethod
    def create(cls, **identity: Any) -> TaskExecutionCompatibility:
        """Compute canonical identity and construct the validated TaskExecutionCompatibility.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, compatibility_hash=_hash(identity))


class TaskExecution(ContractModel):
    """Seal execution/worker ownership and business-heartbeat/checkpoint state.

    Attributes:
        execution_id: Execution UUID.
        task_id: Owning task UUID.
        graph_thread_id: Task/execution graph thread identifier.
        worker_instance_id: Bound worker UUID.
        compatibility: Installed execution compatibility.
        started_at: Recorded execution start clock.
        last_heartbeat_at: Recorded business heartbeat clock.
        checkpoint_disposition: Active, diagnostic retention or pruning qualification.
        execution_hash: Canonical execution-record identity.
    """

    execution_id: UUID
    task_id: UUID
    graph_thread_id: str = Field(pattern=r"^workspace-task:[0-9a-f-]{36}:[0-9a-f-]{36}$")
    worker_instance_id: UUID
    compatibility: TaskExecutionCompatibility
    started_at: datetime
    last_heartbeat_at: datetime
    checkpoint_disposition: str = Field(
        pattern=r"^(active|retain_for_diagnosis|eligible_for_prune)$"
    )
    execution_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskExecution:
        """Validate the declared TaskExecution invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Canonical execution-record identity differs.
        """
        identity = self.model_dump(mode="json", exclude={"execution_hash"})
        if self.execution_hash != _hash(identity):
            raise ValueError("task execution hash is invalid")
        return self

    @classmethod
    def from_identity(cls, **identity: Any) -> TaskExecution:
        """Compute canonical identity and construct the validated TaskExecution.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, execution_hash=_hash(identity))


class TaskHeartbeatSignal(ContractModel):
    """Typed operational liveness for one exact Task Control execution."""

    task_id: UUID
    execution_id: UUID
    worker_instance_id: UUID
    sequence: int = Field(ge=1)
    observed_at: datetime
    signal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskHeartbeatSignal:
        """Validate the declared TaskHeartbeatSignal invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Observation clock is naive or canonical heartbeat-signal identity differs.
        """
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("task heartbeat clock must be timezone-aware")
        identity = self.model_dump(mode="json", exclude={"signal_hash"})
        if self.signal_hash != _hash(identity):
            raise ValueError("task heartbeat signal hash is invalid")
        return self

    @classmethod
    def from_identity(cls, **identity: Any) -> TaskHeartbeatSignal:
        """Compute canonical identity and construct the validated TaskHeartbeatSignal.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, signal_hash=_hash(identity))


class TaskCommand(ContractModel):
    """Seal a task cancellation request against the task version the caller confirmed.

    Attributes:
        command_id: Cancellation command UUID.
        task_id: Addressed task UUID.
        expected_task_hash: Exact confirmed task version.
        kind: Declared cancellation command.
        requested_at: Recorded request clock.
        command_hash: Canonical command identity.
    """

    command_id: UUID
    task_id: UUID
    expected_task_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: TaskCommandKind
    requested_at: datetime
    command_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskCommand:
        """Validate the declared TaskCommand invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Canonical cancellation command identity differs.
        """
        identity = self.model_dump(mode="json", exclude={"command_hash"})
        if self.command_hash != _hash(identity):
            raise ValueError("task command hash is invalid")
        return self

    @classmethod
    def from_identity(cls, **identity: Any) -> TaskCommand:
        """Compute canonical identity and construct the validated TaskCommand.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, command_hash=_hash(identity))


class TaskStageReceipt(ContractModel):
    """Seal a stage-verifier result against execution, definition and exact evidence.

    Attributes:
        receipt_id: Verifier receipt UUID.
        task_id: Owning task UUID.
        execution_id: Execution being verified.
        stage_id: Verified stage identifier.
        work_item_definition_hash: Sealed stage definition identity.
        verifier_id: Verifier authority identifier.
        evidence: Unique evidence references supplied for verification.
        status: Verified, blocked or cancelled receipt state.
        failure_code: Optional stable failure cause.
        observed_at: Recorded verifier observation clock.
        receipt_hash: Canonical verifier receipt identity.
    """

    receipt_id: UUID
    task_id: UUID
    execution_id: UUID
    stage_id: str
    work_item_definition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verifier_id: str
    evidence: tuple[TaskEvidence, ...]
    status: str = Field(pattern=r"^(VERIFIED|BLOCKED|CANCELLED)$")
    failure_code: str | None = Field(default=None, max_length=120)
    observed_at: datetime
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskStageReceipt:
        """Validate the declared TaskStageReceipt invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Evidence references repeat or canonical verifier receipt identity differs.
        """
        references = tuple(item.reference for item in self.evidence)
        if len(set(references)) != len(references):
            raise ValueError("task stage evidence references must be unique")
        identity = self.model_dump(mode="json", exclude={"receipt_hash"})
        if self.receipt_hash != _hash(identity):
            raise ValueError("task stage receipt hash is invalid")
        return self

    @classmethod
    def from_identity(cls, **identity: Any) -> TaskStageReceipt:
        """Compute canonical identity and construct the validated TaskStageReceipt.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, receipt_hash=_hash(identity))


class TaskSafeProjection(ContractModel):
    """Seal the task read model without exposing the domain input payload.

    Attributes:
        task_id: Projected task UUID.
        task_kind: Task domain.
        goal_summary: Bounded declared goal summary.
        lifecycle: Projected task state.
        current_stage: Optional current stage identifier.
        verified_stage_count: Verified work-item count, no greater than the total.
        total_stage_count: Positive sealed plan stage count.
        running_since: Optional recorded run start clock.
        last_activity_at: Last recorded task activity clock.
        cancel_available: Whether cancellation is available.
        cancel_pending: Whether cancellation awaits enactment.
        queued_next_task_id: Optional queued successor UUID.
        latest_failure_code: Optional stable failure cause.
        artifact_refs: Retained artifact references.
        task_record_hash: Task version represented by this projection.
        projection_hash: Canonical projection identity.
    """

    task_id: UUID
    task_kind: str
    goal_summary: str
    lifecycle: TaskLifecycle
    current_stage: str | None
    verified_stage_count: int = Field(ge=0)
    total_stage_count: int = Field(ge=1)
    running_since: datetime | None
    last_activity_at: datetime
    cancel_available: bool
    cancel_pending: bool
    queued_next_task_id: UUID | None
    latest_failure_code: str | None
    artifact_refs: tuple[str, ...] = ()
    task_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> TaskSafeProjection:
        """Validate the declared TaskSafeProjection invariants and canonical identity.

        Returns:
            This validated immutable record.

        Raises:
            ValueError: Verified count exceeds the plan total or canonical projection identity
                differs.
        """
        if self.verified_stage_count > self.total_stage_count:
            raise ValueError("task projection verified count exceeds plan")
        identity = self.model_dump(mode="json", exclude={"projection_hash"})
        if self.projection_hash != _hash(identity):
            raise ValueError("task safe projection hash is invalid")
        return self

    @classmethod
    def from_identity(cls, **identity: Any) -> TaskSafeProjection:
        """Compute canonical identity and construct the validated TaskSafeProjection.

        Args:
            identity: Declared model fields supplied to canonical sealing and validation.

        Returns:
            The sealed and validated immutable record.
        """
        return cls(**identity, projection_hash=_hash(identity))


class TaskReplan(ContractModel):
    """How the owner that admits a Task kind plans it again.

    A distinct `preview` records a plan and admits nothing; `admitting` admits a Task and
    needs that owner's own confirmation. When both name the same operation, the re-plan
    starts a new Task directly. A kind with no confirmation-free preview has no `preview`,
    and its re-plan is its owner's confirmed admission.
    """

    task_kind: str = Field(min_length=1)
    preview: str | None = Field(default=None, min_length=1)
    admitting: str = Field(min_length=1)


__all__ = [
    "FAILURE_CODE_MAX_LENGTH",
    "ResearchGoal",
    "ResearchPlan",
    "StageFailureCause",
    "TaskCommand",
    "TaskCommandKind",
    "TaskEvidence",
    "TaskExecution",
    "TaskExecutionCompatibility",
    "TaskHeartbeatSignal",
    "TaskInputEnvelope",
    "TaskLifecycle",
    "TaskRecord",
    "TaskRecoveryLink",
    "TaskReplan",
    "TaskSafeProjection",
    "TaskStageReceipt",
    "WorkItemDefinition",
    "WorkItemLifecycle",
    "WorkItemState",
    "failure_code_from",
]
