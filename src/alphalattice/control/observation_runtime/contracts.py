"""Typed public observation contracts for the local desktop runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

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
        span_id: Optional span handle.
        parent_span_id: Optional enclosing span handle.
        correlation_ids: Sorted unique correlation handles.
        authority: Declared assertion or decision owner category.
        sensitivity: Declared exposure class, excluding PROHIBITED.
        retention_class: Recording class, excluding NEVER_RECORD.
        inline_safe_payload: Inline payload when no payload reference is supplied.
        payload_ref: Artifact reference when no inline payload is supplied.
        policy_hash: Identity of the recording policy.
        supersedes_observation_id: Optional identity of an earlier observation.
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
    span_id: str | None = None
    parent_span_id: str | None = None
    correlation_ids: tuple[str, ...] = ()
    authority: ObservationAuthority
    sensitivity: ObservationSensitivity
    retention_class: ObservationRetentionClass
    inline_safe_payload: dict[str, Any] | None = None
    payload_ref: ObservationPayloadRef | None = None
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    supersedes_observation_id: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

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


class ObservationEnvelope(_Contract):
    """Bind a ledger observation to its source, payload and recording policy.

    Identity retains the occurrence clock, source and correlation scope, authority,
    sensitivity, retention class, payload hash and policy. It excludes the arrival
    clock and availability so retention can remove payload bytes without minting a
    different observation. Available envelopes carry exactly one representation.

    Attributes:
        occurred_at: Aware source occurrence clock included in identity.
        observed_at: Aware ledger arrival clock excluded from observation identity.
        source_sequence: Source-local nonnegative sequence, distinct from ledger ordinal.
        correlation_ids: Sorted unique handles included in identity.
        payload_hash: Payload commitment retained even when content is unavailable.
        inline_safe_payload: Inline content, forbidden when availability is not AVAILABLE.
        payload_ref: Referenced content whose owner retains availability information.
        availability: Readback state excluded from observation identity.
        policy_hash: Recording-policy commitment.
        observation_id: Canonical source/payload/policy identity.
    """

    kind: Literal["ObservationEnvelope"] = "ObservationEnvelope"
    schema_kind: str = Field(min_length=1)
    schema_version: int = Field(ge=1)
    occurred_at: datetime
    observed_at: datetime
    source_kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    task_id: str | None = None
    run_id: str | None = None
    stage_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    correlation_ids: tuple[str, ...] = ()
    authority: ObservationAuthority
    sensitivity: ObservationSensitivity
    retention_class: ObservationRetentionClass
    payload_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    inline_safe_payload: dict[str, Any] | None = None
    payload_ref: ObservationPayloadRef | None = None
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    supersedes_observation_id: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    availability: ObservationAvailability = ObservationAvailability.AVAILABLE
    observation_id: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_envelope(self) -> Self:
        """Validate clocks, availability shape and the observation identity.

        Returns:
            This validated envelope.

        Raises:
            ValueError: A clock, correlation, payload representation or identity is invalid.
        """
        _require_aware(self.occurred_at, "observation occurred_at")
        _require_aware(self.observed_at, "observation observed_at")
        if self.correlation_ids != tuple(sorted(set(self.correlation_ids))):
            raise ValueError("observation correlation IDs must be sorted and unique")
        if self.availability is ObservationAvailability.AVAILABLE:
            if (self.inline_safe_payload is None) == (self.payload_ref is None):
                raise ValueError("available observation requires one payload representation")
        elif self.inline_safe_payload is not None:
            raise ValueError("unavailable observation cannot retain inline payload")
        if self.observation_id != observation_identity(self):
            raise ValueError("observation envelope identity is invalid")
        return self


class ObservationAppendResult(_Contract):
    """Report a new append or exact reuse without changing source identity.

    Attributes:
        disposition: APPENDED for a new record or REUSED_EXACT for the same observation.
        observation_id: Identity of the appended or reused observation.
        source_sequence: Nonnegative source-local sequence.
        availability: Resulting payload availability.
        result_hash: Canonical identity of the other result fields.
    """

    kind: Literal["ObservationAppendResult"] = "ObservationAppendResult"
    disposition: Literal["APPENDED", "REUSED_EXACT"]
    observation_id: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    source_sequence: int = Field(ge=0)
    availability: ObservationAvailability
    result_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_result(self) -> Self:
        """Require the canonical append-result identity.

        Returns:
            This validated result.

        Raises:
            ValueError: The result hash differs from its fields.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"result_hash"}))
        if self.result_hash != expected:
            raise ValueError("observation append result hash is invalid")
        return self


class ObservationRetentionPolicy(_Contract):
    """Seal the workspace-relative observation retention budget.

    New ledger caps are one percent of the operator workspace cap. Historical receipts
    retain their original 64-256 MiB clamp. Cleanup starts at 90 percent and targets 80 percent.
    These are
    operational retention bounds; they do not change scientific artifact identities.

    Attributes:
        workspace_managed_cap_bytes: Positive managed workspace capacity.
        ledger_cap_bytes: Derived observation ledger capacity.
        high_water_bytes: Derived cleanup trigger.
        cleanup_target_bytes: Derived target after cleanup.
        transient_ring_size: Positive number of transient in-memory observations.
        policy_hash: Canonical policy identity excluding this hash.
    """

    kind: Literal["ObservationRetentionPolicy"] = "ObservationRetentionPolicy"
    workspace_managed_cap_bytes: int = Field(gt=0)
    ledger_cap_bytes: int = Field(gt=0)
    high_water_bytes: int = Field(ge=0)
    cleanup_target_bytes: int = Field(ge=0)
    transient_ring_size: int = Field(default=128, ge=1)
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    cap_basis: Literal["BOUNDED_LEGACY", "WORKSPACE_SHARE"] = "BOUNDED_LEGACY"
    """Historical receipts retain their exact clamp; new execution uses a workspace share."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> Self:
        """Require the fixed capacity formula, watermarks and policy hash.

        Returns:
            This validated policy.

        Raises:
            ValueError: A derived budget, watermark or identity differs from policy.
        """
        expected_cap = min(
            256 * 1024 * 1024,
            max(64 * 1024 * 1024, self.workspace_managed_cap_bytes // 100),
        )
        if self.cap_basis == "WORKSPACE_SHARE":
            expected_cap = max(1, self.workspace_managed_cap_bytes // 100)
        if self.ledger_cap_bytes != expected_cap:
            raise ValueError("observation retention cap is invalid")
        if self.high_water_bytes != int(expected_cap * 0.9):
            raise ValueError("observation retention high water is invalid")
        if self.cleanup_target_bytes != int(expected_cap * 0.8):
            raise ValueError("observation retention cleanup target is invalid")
        payload = self.model_dump(mode="json", exclude={"policy_hash"})
        if self.cap_basis == "BOUNDED_LEGACY":
            payload.pop("cap_basis")
        expected_hash = canonical_hash(payload)
        if self.policy_hash != expected_hash:
            raise ValueError("observation retention policy hash is invalid")
        return self


class ObservationProjectionCheckpoint(_Contract):
    """Bind a projection to the observations and source cursors it consumed.

    Attributes:
        projection_kind: Named projection type.
        projection_key: Projection-specific scope handle.
        through_observation_id: Last observation identity included in this checkpoint.
        observation_count: Number of consumed observations.
        source_cursors: Nonnegative source-local sequence cursors.
        policy_hash: Policy used to derive this projection.
        projection_hash: Identity of the resulting projection.
        checkpoint_hash: Canonical identity of the checkpoint fields.
    """

    kind: Literal["ObservationProjectionCheckpoint"] = "ObservationProjectionCheckpoint"
    projection_kind: str = Field(min_length=1)
    projection_key: str = Field(min_length=1)
    through_observation_id: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    observation_count: int = Field(ge=0)
    source_cursors: dict[str, int]
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    projection_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_checkpoint(self) -> Self:
        """Require nonnegative source cursors and the checkpoint identity.

        Returns:
            This validated checkpoint.

        Raises:
            ValueError: A cursor is negative or the checkpoint hash differs.
        """
        if any(value < 0 for value in self.source_cursors.values()):
            raise ValueError("observation projection cursor is invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"checkpoint_hash"}))
        if self.checkpoint_hash != expected:
            raise ValueError("observation projection checkpoint hash is invalid")
        return self


class ObservationRunSummary(_Contract):
    """Retain compacted-run counts and commitments without the removed payloads.

    Attributes:
        run_id: Run whose operational observations were compacted.
        compacted_observation_count: Positive number represented by the summary.
        retained_decision_ids: Decision observation identities preserved by retention.
        source_ranges: Nonnegative inclusive first/last sequences grouped by source.
        compacted_payload_digest: Commitment to the compacted payloads.
        summary_hash: Canonical identity of the summary fields.
    """

    kind: Literal["ObservationRunSummary"] = "ObservationRunSummary"
    run_id: str = Field(min_length=1)
    compacted_observation_count: int = Field(ge=1)
    retained_decision_ids: tuple[Hash, ...]
    source_ranges: dict[str, tuple[int, int]]
    compacted_payload_digest: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    summary_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_summary(self) -> Self:
        """Validate source-range ordering and the compacted-run summary hash.

        Returns:
            This validated summary.

        Raises:
            ValueError: A range is negative or reversed, or the summary hash differs.
        """
        if any(first < 0 or last < first for first, last in self.source_ranges.values()):
            raise ValueError("observation run source range is invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"summary_hash"}))
        if self.summary_hash != expected:
            raise ValueError("observation run summary hash is invalid")
        return self


class ObservationRetentionReport(_Contract):
    """Record retention effects separately from physical storage measurements.

    Logical payload bytes cannot increase during retention. The report does not
    require physical file bytes to shrink; compaction and filesystem allocation are
    distinct from logical payload removal.

    Attributes:
        policy_hash: Retention policy identity used for this pass.
        bytes_before: Logical payload size before cleanup.
        bytes_after: Logical payload size after cleanup.
        physical_bytes_before: Observed physical storage size before cleanup.
        physical_bytes_after: Observed physical storage size after cleanup.
        compacted_count: Number of observations compacted.
        evicted_count: Number of observations evicted by retention.
        protected_run_ids: Runs protected from this cleanup pass.
        run_summaries: Commitments retained for compacted runs.
        report_hash: Canonical identity of all report fields except this hash.
    """

    kind: Literal["ObservationRetentionReport"] = "ObservationRetentionReport"
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    bytes_before: int = Field(ge=0)
    bytes_after: int = Field(ge=0)
    physical_bytes_before: int = Field(ge=0)
    physical_bytes_after: int = Field(ge=0)
    compacted_count: int = Field(ge=0)
    evicted_count: int = Field(ge=0)
    protected_run_ids: tuple[str, ...]
    run_summaries: tuple[ObservationRunSummary, ...]
    report_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_report(self) -> Self:
        """Require nonincreasing logical bytes and a matching report hash.

        Returns:
            This validated report.

        Raises:
            ValueError: Logical payload bytes increased or the report hash differs.
        """
        if self.bytes_after > self.bytes_before:
            raise ValueError("observation retention increased logical payload bytes")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"report_hash"}))
        if self.report_hash != expected:
            raise ValueError("observation retention report hash is invalid")
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
    """Pair an envelope with its positive ledger commit ordinal.

    Attributes:
        ordinal: Commit order within the store epoch, distinct from source sequence.
        envelope: Observation metadata and current payload availability.
    """

    kind: Literal["ObservationReadItem"] = "ObservationReadItem"
    ordinal: int = Field(ge=1)
    envelope: ObservationEnvelope


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

    @property
    def unavailable_count(self) -> int:
        """Count page items whose envelopes are not AVAILABLE.

        Returns:
            The number of compacted, evicted or missing/tampered payloads in this page.
        """
        return sum(
            item.envelope.availability is not ObservationAvailability.AVAILABLE
            for item in self.items
        )


def build_retention_policy(*, workspace_managed_cap_bytes: int) -> ObservationRetentionPolicy:
    """Derive and seal the fixed workspace-relative retention policy.

    Args:
        workspace_managed_cap_bytes: Positive workspace-managed storage budget in bytes.

    Returns:
        The validated ledger cap, watermarks, transient ring size and policy identity.

    Raises:
        ValueError: The resulting policy fields do not satisfy the retention contract.
    """
    cap = max(1, workspace_managed_cap_bytes // 100)
    values = {
        "workspace_managed_cap_bytes": workspace_managed_cap_bytes,
        "ledger_cap_bytes": cap,
        "high_water_bytes": int(cap * 0.9),
        "cleanup_target_bytes": int(cap * 0.8),
        "transient_ring_size": 128,
        "cap_basis": "WORKSPACE_SHARE",
    }
    provisional = ObservationRetentionPolicy.model_construct(**values, policy_hash="")
    identity = provisional.model_dump(mode="json", exclude={"policy_hash"})
    return ObservationRetentionPolicy(**identity, policy_hash=canonical_hash(identity))


def build_observation_authority_reference(
    *,
    owner_kind: str,
    record_kind: str,
    record_hash: Hash,
    task_id: str | None = None,
    run_id: str | None = None,
    subject_id: str | None = None,
) -> ObservationAuthorityReference:
    """Seal a scoped pointer to a record owned outside the observation ledger.

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


def observation_identity(value: ObservationEnvelope) -> Hash:
    """Hash immutable source, scope, payload and policy fields of an envelope.

    Args:
        value: Envelope whose identity-bearing fields are selected.

    Returns:
        The canonical observation identity, excluding arrival time and availability.
    """
    return cast(
        str,
        canonical_hash(
            {
                "schema_kind": value.schema_kind,
                "schema_version": value.schema_version,
                "occurred_at": value.occurred_at,
                "source_kind": value.source_kind,
                "source_id": value.source_id,
                "source_sequence": value.source_sequence,
                "task_id": value.task_id,
                "run_id": value.run_id,
                "stage_id": value.stage_id,
                "span_id": value.span_id,
                "parent_span_id": value.parent_span_id,
                "correlation_ids": value.correlation_ids,
                "authority": value.authority,
                "sensitivity": value.sensitivity,
                "retention_class": value.retention_class,
                "payload_hash": value.payload_hash,
                "policy_hash": value.policy_hash,
                "supersedes_observation_id": value.supersedes_observation_id,
            }
        ),
    )


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    value.astimezone(UTC)


__all__ = [
    "ObservationAppendResult",
    "ObservationAuthority",
    "ObservationAuthorityReadback",
    "ObservationAuthorityReader",
    "ObservationAuthorityReference",
    "ObservationAvailability",
    "ObservationDraft",
    "ObservationEnvelope",
    "ObservationPayloadRef",
    "ObservationProjectionCheckpoint",
    "ObservationReadCursor",
    "ObservationReadItem",
    "ObservationReadPage",
    "ObservationRetentionClass",
    "ObservationRetentionPolicy",
    "ObservationRetentionReport",
    "ObservationRunSummary",
    "ObservationSensitivity",
    "build_observation_authority_readback",
    "build_observation_authority_reference",
    "build_retention_policy",
    "observation_identity",
]
