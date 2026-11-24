"""Typed public observation contracts for the local desktop runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.domain_contracts.identity import canonical_hash

Hash = str


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObservationAuthority(StrEnum):
    """Classify the owner whose assertion an observation carries.

    Operational, task, artifact and guardian assertions retain their owner boundaries.
    An agent proposal is a proposal; a host decision carries the host's authority.
    The enum label alone does not replace authoritative owner readback.
    """

    OPERATIONAL_ASSERTION = "OPERATIONAL_ASSERTION"
    TASK_CONTROL_ASSERTION = "TASK_CONTROL_ASSERTION"
    ARTIFACT_ASSERTION = "ARTIFACT_ASSERTION"
    GUARDIAN_ASSERTION = "GUARDIAN_ASSERTION"
    AGENT_PROPOSAL = "AGENT_PROPOSAL"
    HOST_DECISION = "HOST_DECISION"


class ObservationSensitivity(StrEnum):
    """Declare the payload's permitted exposure class.

    Public-safe, local-sensitive and private-runtime payloads remain subject to the
    recording policy. PROHIBITED payloads cannot be admitted as observation drafts.
    """

    PUBLIC_SAFE = "PUBLIC_SAFE"
    LOCAL_SENSITIVE = "LOCAL_SENSITIVE"
    PRIVATE_RUNTIME = "PRIVATE_RUNTIME"
    PROHIBITED = "PROHIBITED"


class ObservationRetentionClass(StrEnum):
    """Select the observation's recording and retention policy class.

    Operational, durable decision, artifact reference and eligible internal cognition
    records have distinct retention purposes. NEVER_RECORD is rejected at admission;
    eligibility does not grant independent authority to retain private content.
    """

    TRANSIENT_OPERATIONAL = "TRANSIENT_OPERATIONAL"
    RUN_OPERATIONAL = "RUN_OPERATIONAL"
    DURABLE_DECISION = "DURABLE_DECISION"
    ARTIFACT_REFERENCE = "ARTIFACT_REFERENCE"
    INTERNAL_COGNITION_ELIGIBLE = "INTERNAL_COGNITION_ELIGIBLE"
    NEVER_RECORD = "NEVER_RECORD"


class ObservationAvailability(StrEnum):
    """Describe payload availability while preserving observation identity.

    Compaction, retention eviction and missing or tampered content remain visible as
    unavailable observations rather than being presented as available payloads.
    """

    AVAILABLE = "AVAILABLE"
    COMPACTED = "COMPACTED"
    EVICTED_BY_RETENTION = "EVICTED_BY_RETENTION"
    MISSING_OR_TAMPERED = "MISSING_OR_TAMPERED"


class ObservationPayloadRef(_Contract):
    """Bind an artifact reference, its content identity and availability.

    Attributes:
        artifact_kind: Named artifact type used by its owning reader.
        content_hash: Expected content identity of the referenced payload.
        uri: Workspace playpen URI resolved by the artifact owner.
        availability: Recorded payload availability.
        byte_count: Optional nonnegative payload size.
        ref_hash: Canonical identity of all reference fields except this hash.
    """

    kind: Literal["ObservationPayloadRef"] = "ObservationPayloadRef"
    artifact_kind: str = Field(min_length=1)
    content_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str = Field(pattern=r"^playpen://")
    availability: ObservationAvailability
    byte_count: int | None = Field(default=None, ge=0)
    ref_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the canonical identity of the payload reference.

        Returns:
            This validated reference.

        Raises:
            ValueError: The reference hash differs from its fields.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"ref_hash"}))
        if self.ref_hash != expected:
            raise ValueError("observation payload reference hash is invalid")
        return self


class ObservationAuthorityReference(_Contract):
    """Typed pointer to a source-owner record that the UOL can re-read."""

    kind: Literal["ObservationAuthorityReference"] = "ObservationAuthorityReference"
    owner_kind: str = Field(min_length=1)
    record_kind: str = Field(min_length=1)
    record_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str | None = None
    run_id: str | None = None
    subject_id: str | None = None
    reference_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Validate complete owner scope and the reference identity.

        Returns:
            This validated owner reference.

        Raises:
            ValueError: Task/run scope is incomplete, subject scope is absent, or the hash differs.
        """
        if (self.task_id is None) != (self.run_id is None):
            raise ValueError("observation authority scope is incomplete")
        if self.subject_id is not None and self.task_id is None:
            raise ValueError("observation authority subject lacks a task/run scope")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"reference_hash"}))
        if self.reference_hash != expected:
            raise ValueError("observation authority reference hash is invalid")
        return self


class ObservationAuthorityReadback(_Contract):
    """Proof returned by the named external owner after authoritative readback."""

    kind: Literal["ObservationAuthorityReadback"] = "ObservationAuthorityReadback"
    reference_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    owner_kind: str = Field(min_length=1)
    record_kind: str = Field(min_length=1)
    record_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str | None = None
    run_id: str | None = None
    subject_id: str | None = None
    safe_claims: tuple[tuple[str, str | int | bool | None], ...] = ()
    readback_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require canonical safe-claim keys and a matching readback hash.

        Returns:
            This validated readback.

        Raises:
            ValueError: Claim keys are not sorted and unique, or the hash differs.
        """
        claim_keys = tuple(key for key, _value in self.safe_claims)
        if claim_keys != tuple(sorted(set(claim_keys))):
            raise ValueError("observation authority safe claims are not canonical")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"readback_hash"}))
        if self.readback_hash != expected:
            raise ValueError("observation authority readback hash is invalid")
        return self


class ObservationAuthorityReader(Protocol):
    """Read a typed record from its real owner; hashes alone are insufficient."""

    def read_observation_authority(
        self, reference: ObservationAuthorityReference
    ) -> ObservationAuthorityReadback:
        """Re-read an observation reference through its authoritative external owner.

        Args:
            reference: Typed owner record and optional task/run/subject scope.

        Returns:
            The real owner readback used to validate the assertion, including safe claims.
        """
        ...


class ObservationDraft(_Contract):
    """Declare a recordable observation before ledger admission.

    Admission requires an aware occurrence clock, sorted unique correlations and
    exactly one payload representation. Prohibited sensitivity and NEVER_RECORD
    retention are rejected. Declaring authority does not itself verify the owner.

    Attributes:
        schema_kind: Named payload schema interpreted by its owner.
        schema_version: Positive schema version of that payload.
        occurred_at: Timezone-aware clock recorded by the source.
        source_kind: Source-owner category.
        source_id: Stable source handle.
        source_sequence: Nonnegative sequence within this source.
        task_id: Optional task correlation.
        run_id: Optional run correlation.
        stage_id: Optional stage correlation.
        correlation_ids: Sorted unique correlation handles.
        authority: Declared assertion or decision owner category.
        sensitivity: Declared exposure class, excluding PROHIBITED.
        retention_class: Recording class, excluding NEVER_RECORD.
        inline_safe_payload: Inline payload when no payload reference is supplied.
        payload_ref: Artifact reference when no inline payload is supplied.
        policy_hash: Identity of the recording policy.
    """

    kind: Literal["ObservationDraft"] = "ObservationDraft"
    schema_kind: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    occurred_at: datetime
    source_kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    task_id: str | None = None
    run_id: str | None = None
    stage_id: str | None = None
    correlation_ids: tuple[str, ...] = ()
    authority: ObservationAuthority
    sensitivity: ObservationSensitivity
    retention_class: ObservationRetentionClass
    inline_safe_payload: dict[str, Any] | None = None
    payload_ref: ObservationPayloadRef | None = None
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_draft(self) -> Self:
        """Validate the draft clock, correlations and permitted payload representation.

        Returns:
            This admitted-shape draft.

        Raises:
            ValueError: The clock, correlations, representation or recording class is invalid.
        """
        _require_aware(self.occurred_at, "observation occurred_at")
        if self.correlation_ids != tuple(sorted(set(self.correlation_ids))):
            raise ValueError("observation correlation IDs must be sorted and unique")
        if (self.inline_safe_payload is None) == (self.payload_ref is None):
            raise ValueError("observation requires exactly one payload representation")
        if self.sensitivity is ObservationSensitivity.PROHIBITED:
            raise ValueError("observation.payload_not_admitted")
        if self.retention_class is ObservationRetentionClass.NEVER_RECORD:
            raise ValueError("observation.payload_not_admitted")
        return self


class ObservationRecord(_Contract):
    """One kept observation as the Host's log holds it: its own fields and its payload.

    Its identity is the log's: `observation_id` is the id of the event that keeps it, issued by
    the Host and opaque; the log's hash chain binds its content. Its kind is the event's type,
    its place the event's. Every record the log keeps holds its payload.

    Attributes:
        occurred_at: Aware clock its source recorded.
        observed_at: Aware clock the Host kept it at.
        source_sequence: Source-local nonnegative sequence, distinct from the log's order.
        correlation_ids: Sorted unique handles.
        observation_id: The event id that keeps it.
    """

    kind: Literal["ObservationRecord"] = "ObservationRecord"
    schema_kind: str = Field(min_length=1)
    occurred_at: datetime
    observed_at: datetime
    source_kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    task_id: str | None = None
    run_id: str | None = None
    stage_id: str | None = None
    correlation_ids: tuple[str, ...] = ()
    authority: ObservationAuthority
    retention_class: ObservationRetentionClass
    inline_safe_payload: dict[str, Any]
    availability: ObservationAvailability = ObservationAvailability.AVAILABLE
    observation_id: str = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_record(self) -> Self:
        """Validate its clocks and correlations.

        Returns:
            This validated record.

        Raises:
            ValueError: A clock is naive or the correlations are not sorted and unique.
        """
        _require_aware(self.occurred_at, "observation occurred_at")
        _require_aware(self.observed_at, "observation observed_at")
        if self.correlation_ids != tuple(sorted(set(self.correlation_ids))):
            raise ValueError("observation correlation IDs must be sorted and unique")
        return self


class ObservationReadCursor(_Contract):
    """Where a bounded reader stopped: one store epoch and its last seen ordinal.

    The ordinal is the store's commit order, not an observation identity; it is
    only meaningful inside the epoch that assigned it. A reader that presents a
    cursor from another epoch, or beyond the current head, is reset rather than
    silently continued.
    """

    kind: Literal["ObservationReadCursor"] = "ObservationReadCursor"
    store_epoch: str = Field(pattern=r"^[0-9a-f]{32}$")
    ordinal: int = Field(ge=0)

    def encode(self) -> str:
        """Serialize this validated store epoch and last-seen ordinal.

        Returns:
            The epoch:ordinal cursor text.
        """
        return f"{self.store_epoch}:{self.ordinal}"

    @classmethod
    def parse(cls, value: str) -> ObservationReadCursor:
        """Parse a cursor without granting validity against a particular store head.

        Args:
            value: Encoded epoch:ordinal cursor text.

        Returns:
            The validated epoch and nonnegative ordinal.

        Raises:
            ValueError: The separator, ordinal or typed cursor fields are invalid.
        """
        epoch, separator, ordinal = value.partition(":")
        if not separator or not ordinal.isdigit():
            raise ValueError("observation.cursor_invalid")
        try:
            return cls(store_epoch=epoch, ordinal=int(ordinal))
        except ValueError as exc:
            raise ValueError("observation.cursor_invalid") from exc


class ObservationReadItem(_Contract):
    """Pair a record with its positive place in the log and its place in the trace tree.

    Attributes:
        ordinal: The log's order within the store epoch, distinct from source sequence.
        record: The observation the event keeps.
        trace_id: The trace its event is in: one CLI call's, or the Host's own.
        span_id: Its own span (an operation's), or the span it happened in (a fact's).
        parent_span_id: Its span's parent: the Host step it ran in, None at a trace's root.
    """

    kind: Literal["ObservationReadItem"] = "ObservationReadItem"
    ordinal: int = Field(ge=1)
    record: ObservationRecord
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None


class ObservationReadPage(_Contract):
    """One bounded slice of the store in commit order.

    `TAIL` answers a reader with no cursor (the newest `limit` rows); `CONTINUED`
    answers a valid cursor; `RESET` answers a cursor this store cannot honour and
    carries a fresh tail so the reader can rebuild. Rows whose payload retention
    already removed remain in `items` with their availability, so a gap is
    visible rather than skipped.
    """

    kind: Literal["ObservationReadPage"] = "ObservationReadPage"
    disposition: Literal["TAIL", "CONTINUED", "RESET"]
    store_epoch: str = Field(pattern=r"^[0-9a-f]{32}$")
    head_ordinal: int = Field(ge=0)
    items: tuple[ObservationReadItem, ...]
    more: bool

    @property
    def next_cursor(self) -> ObservationReadCursor:
        """Return the last item cursor, or the store head when this page is empty.

        Returns:
            A cursor in this page's store epoch.
        """
        ordinal = self.items[-1].ordinal if self.items else self.head_ordinal
        return ObservationReadCursor(store_epoch=self.store_epoch, ordinal=ordinal)

    def advances(self, cursor: ObservationReadCursor) -> bool:
        """Whether reading on from this page moves past `cursor`: a reader that followed a page
        that does not would read the same rows again without end."""
        return self.next_cursor.ordinal > cursor.ordinal

    @property
    def unavailable_count(self) -> int:
        """Count page items whose records are not AVAILABLE.

        Returns:
            The number of compacted, evicted or missing/tampered payloads in this page.
        """
        return sum(
            item.record.availability is not ObservationAvailability.AVAILABLE for item in self.items
        )


def build_observation_authority_reference(
    *,
    owner_kind: str,
    record_kind: str,
    record_hash: Hash,
    task_id: str | None = None,
    run_id: str | None = None,
    subject_id: str | None = None,
) -> ObservationAuthorityReference:
    """Seal a scoped pointer to a record owned outside the Host's history.

    Args:
        owner_kind: Named external authority owner.
        record_kind: Named record type read by that owner.
        record_hash: Expected identity of the owner record.
        task_id: Optional task scope, paired with run_id.
        run_id: Optional run scope, paired with task_id.
        subject_id: Optional subject requiring task/run scope.

    Returns:
        The validated reference with its canonical identity.

    Raises:
        ValueError: Scope or typed identity fields are invalid.
    """
    values = {
        "owner_kind": owner_kind,
        "record_kind": record_kind,
        "record_hash": record_hash,
        "task_id": task_id,
        "run_id": run_id,
        "subject_id": subject_id,
    }
    provisional = ObservationAuthorityReference.model_construct(**values, reference_hash="")
    identity = provisional.model_dump(mode="json", exclude={"reference_hash"})
    return ObservationAuthorityReference(
        **identity,
        reference_hash=canonical_hash(identity),
    )


def build_observation_authority_readback(
    reference: ObservationAuthorityReference,
    *,
    safe_claims: dict[str, str | int | bool | None] | None = None,
) -> ObservationAuthorityReadback:
    """Seal owner-supplied safe claims for an existing reference.

    This builder copies reference scope and canonicalizes claims. The external reader
    must actually read the owner record; invoking this builder is not that readback.

    Args:
        reference: Owner pointer whose scope and identity are copied.
        safe_claims: Optional owner-derived safe scalar claims, sorted by key.

    Returns:
        The validated readback commitment.
    """
    values = {
        "reference_hash": reference.reference_hash,
        "owner_kind": reference.owner_kind,
        "record_kind": reference.record_kind,
        "record_hash": reference.record_hash,
        "task_id": reference.task_id,
        "run_id": reference.run_id,
        "subject_id": reference.subject_id,
        "safe_claims": tuple(sorted((safe_claims or {}).items())),
    }
    provisional = ObservationAuthorityReadback.model_construct(**values, readback_hash="")
    identity = provisional.model_dump(mode="json", exclude={"readback_hash"})
    return ObservationAuthorityReadback(**identity, readback_hash=canonical_hash(identity))


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    value.astimezone(UTC)


__all__ = [
    "ObservationAuthority",
    "ObservationAuthorityReadback",
    "ObservationAuthorityReader",
    "ObservationAuthorityReference",
    "ObservationAvailability",
    "ObservationDraft",
    "ObservationPayloadRef",
    "ObservationReadCursor",
    "ObservationReadItem",
    "ObservationReadPage",
    "ObservationRecord",
    "ObservationRetentionClass",
    "ObservationSensitivity",
    "build_observation_authority_readback",
    "build_observation_authority_reference",
]
