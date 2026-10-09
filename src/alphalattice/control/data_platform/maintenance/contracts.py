"""Typed, path-free contracts for one desktop workspace maintenance cycle."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.feature_engine.contracts import canonical_hash
from alphalattice.foundation.feature_engine.inputs.contracts import FeatureCandidateRecheck
from alphalattice.foundation.market_data_ops.sources.contracts import CandidateDataRecheck
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest


class MaintenanceTrigger(StrEnum):
    """Identify startup, onboarding, user request or deferred resume as the cycle trigger."""

    STARTUP = "startup"
    ONBOARDING = "onboarding"
    USER_REQUEST = "user_request"
    DEFERRED_RESUME = "deferred_resume"


def workspace_maintenance_data_policy_hash() -> str:
    """The installed current-workspace maintenance policy, not a UI preference."""
    return str(
        canonical_hash(
            {
                "rolling_days": 45,
                "standard_incremental_gap_days": 120,
                "maximum_automatic_catch_up_gap_days": 366,
                "provider_adjusted_close_required": True,
                "full_history_escalation": "EXPLICIT_LISTING_AUTHORIZATION_ONLY",
                "daily_bar_settlement_delay_minutes": 120,
                "historical_revision_detection": "EVIDENCE_TRIGGERED_NO_PROVIDER_CHANGE_FEED",
                "transport": "chunk25-workers2-to1-backoff5-15-45",
            }
        )
    )


class MaintenancePhase(StrEnum):
    """Identify progress through preflight, market data, quality, Feature and completion."""

    PREFLIGHT = "preflight"
    MARKET_DATA = "market_data"
    QUALITY = "quality"
    FEATURE = "feature"
    COMPLETED = "completed"


class MaintenanceStatus(StrEnum):
    """Record maintenance lifecycle, review, deferral, completion or safe cancellation."""

    NOOP = "noop"
    RUNNING = "running"
    DEFERRED = "deferred"
    REVIEW_PENDING = "review_pending"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    """The Task that ran this cycle was cancelled at a safe checkpoint: not active, not
    finished, not retried under this request. Listing work already completed belongs to
    the maintenance operation and stays available to the next plan's cycle."""


CANCELLED_AT_SAFE_CHECKPOINT = "workspace_maintenance.cancelled_at_safe_checkpoint"
"""The cycle's failure code once its Task's cancellation was enacted."""


class ActionAuditScope(StrEnum):
    """Distinguish bounded rolling action audits from explicitly admitted full-history audits."""

    ROLLING = "rolling"
    FULL = "full"


@dataclass(frozen=True)
class WorkspaceMaintenanceRequest:
    """Bind one maintenance scope to membership, installed policies and separate clocks.

    Attributes:
        market_profile_id: Market profile to maintain.
        target_market_session: Session the cycle intends to reach.
        knowledge_cutoff_at: Optional causal evidence cutoff.
        requested_at: Optional operational request clock.
        trigger: Declared start/resume trigger.
        membership_revision: Immutable membership revision.
        data_policy_hash: Installed Data policy commitment.
        feature_policy_hash: Installed Feature policy commitment.
        request_hash: Canonical request identity.
        full_history_listing_ids: Explicit full-history listing scope.
        candidate_recheck: Optional Feature recheck scope.
        candidate_data_recheck: Optional disjoint raw-data retry scope.
    """

    market_profile_id: str
    target_market_session: date
    knowledge_cutoff_at: datetime | None
    trigger: MaintenanceTrigger
    membership_revision: str
    data_policy_hash: str
    feature_policy_hash: str
    request_hash: str
    full_history_listing_ids: tuple[str, ...] = ()
    requested_at: datetime | None = field(
        default=None, metadata={"exclude_if": lambda value: value is None}
    )
    candidate_recheck: FeatureCandidateRecheck | None = field(
        default=None, metadata={"exclude_if": lambda value: value is None}
    )
    candidate_data_recheck: CandidateDataRecheck | None = field(
        default=None, metadata={"exclude_if": lambda value: value is None}
    )

    @property
    def request_clock(self) -> datetime:
        """Resolve the operational request clock, falling back to its evidence cutoff.

        Returns:
            The recorded request clock or cutoff.

        Raises:
            ValueError: Both recorded clocks are absent.
        """
        value = self.requested_at or self.knowledge_cutoff_at
        if value is None:
            raise ValueError("maintenance request clock is absent")
        return value

    @classmethod
    def create(
        cls,
        *,
        market_profile_id: str,
        target_market_session: date,
        knowledge_cutoff_at: datetime | None,
        trigger: MaintenanceTrigger,
        membership_revision: str,
        data_policy_hash: str,
        feature_policy_hash: str,
        full_history_listing_ids: tuple[str, ...] = (),
        requested_at: datetime | None = None,
        candidate_recheck: FeatureCandidateRecheck | None = None,
        candidate_data_recheck: CandidateDataRecheck | None = None,
    ) -> WorkspaceMaintenanceRequest:
        """Normalize clocks and listing scope and seal a maintenance request.

        Args:
            market_profile_id: Workspace market profile identifier.
            target_market_session: Market session the request intends to reach.
            knowledge_cutoff_at: Optional aware evidence cutoff retained separately from request
                time.
            trigger: Declared reason for starting or resuming maintenance.
            membership_revision: Immutable membership revision bound by the request.
            data_policy_hash: Installed Data policy commitment.
            feature_policy_hash: Installed Feature policy commitment.
            full_history_listing_ids: Listing scope explicitly admitted for full-history work.
            requested_at: Optional aware operational request clock.
            candidate_recheck: Optional Feature candidate recheck scope.
            candidate_data_recheck: Optional raw-data candidate retry scope, disjoint from the
                Feature recheck.

        Returns:
            The immutable canonical maintenance request.

        Raises:
            ValueError: Request clocks are absent/naive or candidate recheck listing scopes overlap.
        """
        if knowledge_cutoff_at is None and requested_at is None:
            raise ValueError("maintenance acquisition requires its request clock")
        if any(
            value.tzinfo is None
            for value in (knowledge_cutoff_at, requested_at)
            if value is not None
        ):
            raise ValueError("maintenance knowledge cutoff must be timezone-aware")
        payload: dict[str, Any] = {
            "market_profile_id": market_profile_id,
            "target_market_session": target_market_session.isoformat(),
            "knowledge_cutoff_at": knowledge_cutoff_at.astimezone(UTC).isoformat()
            if knowledge_cutoff_at
            else None,
            "trigger": trigger.value,
            "membership_revision": membership_revision,
            "data_policy_hash": data_policy_hash,
            "feature_policy_hash": feature_policy_hash,
            "full_history_listing_ids": tuple(sorted(set(full_history_listing_ids))),
        }
        if requested_at is not None:
            payload["requested_at"] = requested_at.astimezone(UTC).isoformat()
        if candidate_recheck is not None:
            payload["candidate_recheck"] = asdict(candidate_recheck)
        if candidate_data_recheck is not None:
            # Both rechecks may fall due on one day and a request then carries
            # both: the raw retry of failed candidates outside the parent and
            # the Feature/Sector recheck of parent members outside the current
            # membership. Their listings are disjoint by construction; a scope
            # that names a listing in both is not one request's work.
            if candidate_recheck is not None and set(candidate_recheck.listing_ids) & set(
                candidate_data_recheck.listing_ids
            ):
                raise ValueError("workspace_maintenance.candidate_recheck_scopes_overlap")
            payload["candidate_data_recheck"] = asdict(candidate_data_recheck)
        return cls(
            market_profile_id=market_profile_id,
            target_market_session=target_market_session,
            knowledge_cutoff_at=knowledge_cutoff_at.astimezone(UTC)
            if knowledge_cutoff_at
            else None,
            trigger=trigger,
            membership_revision=membership_revision,
            data_policy_hash=data_policy_hash,
            feature_policy_hash=feature_policy_hash,
            request_hash=canonical_hash(payload),
            full_history_listing_ids=tuple(payload["full_history_listing_ids"]),
            requested_at=requested_at.astimezone(UTC) if requested_at else None,
            candidate_recheck=candidate_recheck,
            candidate_data_recheck=candidate_data_recheck,
        )

    def with_changes(self, **changes: Any) -> WorkspaceMaintenanceRequest:
        """This request with the named fields changed, sealed again.

        Its other fields are passed as the values they are: the candidate scopes stay the
        scopes the seal reads, which a rebuild through `asdict` turned into plain dicts.

        Args:
            changes: Fields of `create` and their new values.

        Returns:
            The request sealed with its new values; with none, this request again.
        """
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "request_hash"
        }
        return type(self).create(**{**values, **changes})


@dataclass(frozen=True)
class ListingMarketDataChange:
    """Describe the exact new and corrected sessions observed for one listing.

    Attributes:
        listing_id: Listing whose inputs changed.
        new_session_start: First newly added market session, when present.
        new_sessions: Sorted unique added sessions.
        raw_correction_start: Earliest recorded raw correction.
        action_correction_start: Earliest corporate-action evidence correction.
        raw_correction_sessions: Sorted unique raw-input correction sessions.
        adjusted_return_change_sessions: Sorted unique adjusted-return correction sessions.
        source_receipt_hashes: Evidence receipt commitments retained with the listing changes.
    """

    listing_id: str
    new_session_start: date | None = None
    new_sessions: tuple[date, ...] = ()
    raw_correction_start: date | None = None
    action_correction_start: date | None = None
    raw_correction_sessions: tuple[date, ...] = ()
    adjusted_return_change_sessions: tuple[date, ...] = ()
    source_receipt_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require every declared session tuple to be sorted and unique.

        Raises:
            ValueError: Added, raw-correction or adjusted-return sessions repeat or are out of
                order.
        """
        for value in (
            self.new_sessions,
            self.raw_correction_sessions,
            self.adjusted_return_change_sessions,
        ):
            if value != tuple(sorted(set(value))):
                raise ValueError("market-data change sessions must be sorted and unique")


@dataclass(frozen=True)
class MarketDataChangeSet:
    """Canonicalize market-data, benchmark, sector and membership changes as one identity.

    Attributes:
        listing_changes: Listing changes stored in listing-identifier order.
        spy_correction_start: Earliest benchmark correction.
        spy_return_change_sessions: Sorted unique benchmark return changes.
        sector_revision_changed: Whether sector revision evidence changed.
        membership_additions: Sorted unique admitted additions.
        membership_removals: Sorted unique removals.
        receipt_hashes: Sorted unique source evidence receipt commitments.
        change_set_hash: Computed canonical change identity.
    """

    listing_changes: tuple[ListingMarketDataChange, ...] = ()
    spy_correction_start: date | None = None
    spy_return_change_sessions: tuple[date, ...] = ()
    sector_revision_changed: bool = False
    membership_additions: tuple[str, ...] = ()
    membership_removals: tuple[str, ...] = ()
    receipt_hashes: tuple[str, ...] = ()
    change_set_hash: str = ""

    def __post_init__(self) -> None:
        """Canonicalize change ordering and verify any supplied change-set identity.

        Raises:
            ValueError: Benchmark sessions are not sorted/unique or a supplied change identity
                differs.
        """
        ordered = tuple(sorted(self.listing_changes, key=lambda item: item.listing_id))
        if self.spy_return_change_sessions != tuple(sorted(set(self.spy_return_change_sessions))):
            raise ValueError("SPY return change sessions must be sorted and unique")
        payload: dict[str, Any] = {
            "listing_changes": [asdict(item) for item in ordered],
            "spy_correction_start": self.spy_correction_start,
            "spy_return_change_sessions": self.spy_return_change_sessions,
            "sector_revision_changed": self.sector_revision_changed,
            "membership_additions": sorted(set(self.membership_additions)),
            "membership_removals": sorted(set(self.membership_removals)),
            "receipt_hashes": sorted(set(self.receipt_hashes)),
        }
        expected = canonical_hash(payload)
        if self.change_set_hash and self.change_set_hash != expected:
            raise ValueError("market-data change-set hash does not match its content")
        object.__setattr__(self, "listing_changes", ordered)
        object.__setattr__(self, "membership_additions", tuple(payload["membership_additions"]))
        object.__setattr__(self, "membership_removals", tuple(payload["membership_removals"]))
        object.__setattr__(self, "receipt_hashes", tuple(payload["receipt_hashes"]))
        object.__setattr__(self, "change_set_hash", expected)

    @property
    def empty(self) -> bool:
        """Report whether any input or membership change is recorded.

        Returns:
            Whether listing, benchmark, sector and membership changes are all absent.
        """
        return not (
            self.listing_changes
            or self.spy_correction_start
            or self.spy_return_change_sessions
            or self.sector_revision_changed
            or self.membership_additions
            or self.membership_removals
        )


@dataclass(frozen=True)
class ActionAuditChainReceipt:
    """One listing's corporate-action audit, chained to the audit before it.

    It checks its hash whenever it is built or read, as the remediation failure receipt does
    (SC4, EV2): a receipt changed in place (its coverage moved, its old hash kept) is
    refused before any reuse by its date. The hash is the one ``create`` always sealed.
    """

    receipt_hash: str
    listing_id: str
    provider: str
    audit_scope: ActionAuditScope
    previous_receipt_hash: str | None
    full_anchor_receipt_hash: str
    history_start: date
    history_end: date
    covered_through_session: date
    window_action_hash: str
    action_set_hash: str
    raw_evidence_hash: str
    mapping_revision: str
    data_policy_hash: str
    provider_receipt_hash: str
    observed_at: datetime

    def __post_init__(self) -> None:
        """Refuse a receipt whose coverage, anchor or hash is not what it seals."""
        if self.observed_at.tzinfo is None:
            raise ValueError("action audit observation must be timezone-aware")
        if self.history_start > self.history_end or self.history_end > self.covered_through_session:
            raise ValueError("action audit chain has an invalid coverage range")
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in {"receipt_hash", "observed_at"}
        }
        if (
            self.audit_scope is ActionAuditScope.FULL
            and self.full_anchor_receipt_hash != self._full_anchor(values)
        ) or self.receipt_hash != self._identity(values):
            raise ValueError("action audit chain receipt hash is invalid")

    @staticmethod
    def _full_anchor(values: Mapping[str, object]) -> str:
        """The anchor a full-history audit is: its own coverage and evidence."""
        digest: str = canonical_hash(
            {
                key: values[key]
                for key in (
                    "listing_id",
                    "provider",
                    "history_start",
                    "history_end",
                    "covered_through_session",
                    "action_set_hash",
                    "raw_evidence_hash",
                    "mapping_revision",
                    "data_policy_hash",
                    "provider_receipt_hash",
                )
            }
        )
        return digest

    @staticmethod
    def _identity(values: Mapping[str, object]) -> str:
        """The receipt's hash: every field but itself and the time it was observed."""
        digest: str = canonical_hash(
            {**values, "audit_scope": ActionAuditScope(str(values["audit_scope"])).value}
        )
        return digest

    @classmethod
    def create(
        cls,
        *,
        listing_id: str,
        provider: str,
        audit_scope: ActionAuditScope,
        previous_receipt_hash: str | None,
        full_anchor_receipt_hash: str | None,
        history_start: date,
        history_end: date,
        covered_through_session: date,
        window_action_hash: str,
        action_set_hash: str,
        raw_evidence_hash: str,
        mapping_revision: str,
        data_policy_hash: str,
        provider_receipt_hash: str,
        observed_at: datetime,
    ) -> ActionAuditChainReceipt:
        """Seal an aware action-audit receipt with its required full-history anchor.

        Args:
            listing_id: Exact listing identifier whose evidence is recorded.
            provider: Source provider identifier recorded by the audit.
            audit_scope: Rolling or full-history action-audit scope.
            previous_receipt_hash: Optional predecessor receipt in the action-audit chain.
            full_anchor_receipt_hash: Full-history anchor required by a rolling receipt.
            history_start: First market session in the bounded history.
            history_end: Last market session in the bounded history.
            covered_through_session: Latest session covered by the action audit.
            window_action_hash: Commitment to actions observed in the audit window.
            action_set_hash: Commitment to the retained corporate-action set.
            raw_evidence_hash: Commitment to raw market evidence used by the audit.
            mapping_revision: Listing/provider mapping revision used by the audit.
            data_policy_hash: Installed Data policy commitment.
            provider_receipt_hash: Provider evidence receipt commitment.
            observed_at: Operational observation clock; callers supply an aware instant.

        Returns:
            The UTC-clocked immutable chain receipt.

        Raises:
            ValueError: Observation is naive or a rolling receipt lacks a full-history anchor.
        """
        if observed_at.tzinfo is None:
            raise ValueError("action audit observation must be timezone-aware")
        values: dict[str, object] = {
            "listing_id": listing_id,
            "provider": provider,
            "audit_scope": audit_scope,
            "previous_receipt_hash": previous_receipt_hash,
            "full_anchor_receipt_hash": full_anchor_receipt_hash,
            "history_start": history_start,
            "history_end": history_end,
            "covered_through_session": covered_through_session,
            "window_action_hash": window_action_hash,
            "action_set_hash": action_set_hash,
            "raw_evidence_hash": raw_evidence_hash,
            "mapping_revision": mapping_revision,
            "data_policy_hash": data_policy_hash,
            "provider_receipt_hash": provider_receipt_hash,
        }
        if audit_scope is ActionAuditScope.FULL:
            values["full_anchor_receipt_hash"] = cls._full_anchor(values)
        if values["full_anchor_receipt_hash"] is None:
            raise ValueError("rolling action audit requires a full-history anchor")
        return cls(
            receipt_hash=cls._identity(values),
            observed_at=observed_at.astimezone(UTC),
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class AgentExecutionBudget:
    """Bound model timeout, case lifetime and the fixed semantic/stale diagnosis allowance.

    Attributes:
        model_call_timeout_seconds: Timeout bounded by the overall case deadline.
        case_deadline_seconds: Total case deadline in seconds.
        maximum_model_calls: Declared model-call allowance.
        maximum_semantic_diagnoses: Exactly one initial semantic diagnosis.
        maximum_stale_rediagnoses: Exactly one stale-evidence rediagnosis.
    """

    model_call_timeout_seconds: int = 20
    case_deadline_seconds: int = 180
    maximum_model_calls: int = 6
    maximum_semantic_diagnoses: int = 1
    maximum_stale_rediagnoses: int = 1

    def __post_init__(self) -> None:
        """Validate timeout bounds and the fixed one-plus-one diagnosis policy.

        Raises:
            ValueError: Timeout is outside the case deadline or either fixed diagnosis allowance
                differs from one.
        """
        if not 1 <= self.model_call_timeout_seconds <= self.case_deadline_seconds:
            raise ValueError("agent timeout budget is invalid")
        if self.maximum_semantic_diagnoses != 1 or self.maximum_stale_rediagnoses != 1:
            raise ValueError("Data Engineer diagnosis budget is fixed at one plus one stale retry")

    @property
    def policy_hash(self) -> str:
        """Compute the canonical identity of all declared execution budget fields.

        Returns:
            The canonical budget policy commitment.
        """
        return str(canonical_hash(asdict(self)))


REMEDIATION_VALIDATION_FAILED = "DATA_REMEDIATION_BUSINESS_VALIDATION_FAILED"
"""The remediation stopped on an exception that names no stable reason of its own."""

_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+(:[A-Za-z0-9_.-]{1,80})?$")


def remediation_failure_reason(error: BaseException) -> tuple[str, str | None]:
    """A stable reason code and a bounded, safe explanation for a failed remediation.

    An actor failure keeps the code it declared; a typed workspace error
    (``AlphaLatticeError``) contributes its code and its authored message; a
    ``ValueError`` whose message is itself a dotted code (the convention of
    every owner on this path, ``feature_input.resolution_policy_changed`` and
    the like) contributes that code; anything else is the generic validation
    failure with no explanation, so no raw exception text, Provider payload or
    model response is ever persisted or shown.
    """
    declared = getattr(error, "failure_code", None)
    if isinstance(declared, str) and declared:
        return declared, None
    failure = getattr(error, "failure", None)
    code = getattr(failure, "code", None)
    if isinstance(code, str) and code:
        message = getattr(failure, "message", None)
        return code, (message[:200] if isinstance(message, str) and message else None)
    text = str(error)
    if isinstance(error, ValueError) and _REASON_CODE.match(text):
        return text, None
    return REMEDIATION_VALIDATION_FAILED, None


@dataclass(frozen=True)
class DataRemediationFailureReceipt:
    """Durable terminal evidence; it never authorizes a remediation effect.

    ``failure_code`` is the stable reason (``remediation_failure_reason``);
    ``explanation`` is the bounded safe text that reason allows, or nothing.
    Receipts recorded before ``explanation`` existed read back with it absent.
    """

    receipt_hash: str
    maintenance_id: str
    case_token: str | None
    evidence_hash: str | None
    failure_code: str
    retryable: bool
    execution_attempt_count: int
    prior_failure_codes: tuple[str, ...]
    agent_execution_hashes: tuple[str, ...]
    model_context_audit_hashes: tuple[str, ...]
    observed_at: datetime
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Require an aware clock, bounded attempt count and canonical failure identity.

        Raises:
            ValueError: Observation is naive, attempt count is outside zero through two, or receipt
                identity differs.
        """
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("Data remediation failure receipt clock must be timezone-aware")
        if self.execution_attempt_count < 0 or self.execution_attempt_count > 2:
            raise ValueError("Data remediation failure receipt attempt count is invalid")
        if self.receipt_hash != self._identity_hash(asdict(self)):
            raise ValueError("Data remediation failure receipt hash is invalid")

    @staticmethod
    def _identity_hash(values: Mapping[str, object]) -> str:
        # An absent explanation is not part of the identity, so receipts sealed
        # before the field existed keep verifying with the hash they carry.
        identity = {key: value for key, value in values.items() if key != "receipt_hash"}
        if identity.get("explanation") is None:
            identity.pop("explanation", None)
        return str(canonical_hash(identity))

    @classmethod
    def seal(cls, **values: object) -> DataRemediationFailureReceipt:
        """Compute the failure identity and construct the validated immutable receipt.

        Args:
            values: Declared fields supplied to the receipt sealing owner.

        Returns:
            The sealed failure receipt.
        """
        return cls(**values, receipt_hash=cls._identity_hash(values))  # type: ignore[arg-type]

    @classmethod
    def read_document(cls, document: Mapping[str, object]) -> DataRemediationFailureReceipt:
        """The receipt as the registry stored it (lists and ISO clocks in JSON).

        Every field the receipt had when it was sealed must be present with
        its recorded shape -- a sequence field is a JSON list of strings --
        before the identity is checked; a document that lost a field or holds
        another shape is refused, never repaired into a receipt whose hash
        would still verify. Only ``explanation`` may be absent (receipts sealed
        before it existed).
        """
        required = {field.name for field in fields(cls)} - {"explanation"}
        missing = sorted(required - set(document))
        if missing:
            raise ValueError(f"Data remediation failure receipt document lacks {missing}")
        values = dict(document)
        for key in (
            "prior_failure_codes",
            "agent_execution_hashes",
            "model_context_audit_hashes",
        ):
            recorded = values[key]
            if not isinstance(recorded, list | tuple) or not all(
                isinstance(item, str) for item in recorded
            ):
                raise ValueError(
                    f"Data remediation failure receipt document field {key} is invalid"
                )
            values[key] = tuple(recorded)
        if not isinstance(values["observed_at"], str | datetime):
            raise ValueError("Data remediation failure receipt document clock is invalid")
        observed = (
            values["observed_at"]
            if isinstance(values["observed_at"], datetime)
            else datetime.fromisoformat(values["observed_at"])
        )
        values["observed_at"] = observed if observed.tzinfo else observed.replace(tzinfo=UTC)
        values.setdefault("explanation", None)
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WorkspaceMaintenanceCycle:
    """Retain durable maintenance state and committed effects for one request.

    Attributes:
        cycle_id: Request-derived cycle identity.
        request: Sealed maintenance scope.
        phase: Recorded current phase.
        status: Recorded lifecycle state.
        market_data_change_set: Observed input changes when available.
        child_task_refs: Recorded child task references.
        effect_receipts: Committed effect receipt references.
        retry_after_at: Optional next retry clock.
        failure_code: Optional stable failure cause.
        transport_workers: Recorded transport concurrency.
        updated_at: Last cycle update clock.
    """

    cycle_id: str
    request: WorkspaceMaintenanceRequest
    phase: MaintenancePhase
    status: MaintenanceStatus
    market_data_change_set: MarketDataChangeSet | None
    child_task_refs: tuple[str, ...]
    effect_receipts: tuple[str, ...]
    retry_after_at: datetime | None
    failure_code: str | None
    transport_workers: int
    updated_at: datetime


@dataclass(frozen=True)
class WorkspaceMaintenanceOutcome:
    """Return bounded cycle status, progress, effect references and retry disposition.

    Attributes:
        cycle_id: Durable cycle identity.
        status: Maintenance disposition.
        phase: Last recorded phase.
        change_set_hash: Optional observed change identity.
        child_task_refs: Recorded child task references.
        retry_after_at: Optional next retry clock.
        failure_code: Optional stable failure cause.
        failure_cause: The failed step's recorded cause: its
            `exception_type`, `detail`, `step`, `unit`, `first_session` and `last_session`.
    """

    cycle_id: str
    status: MaintenanceStatus
    phase: MaintenancePhase
    change_set_hash: str | None = None
    child_task_refs: tuple[str, ...] = ()
    retry_after_at: datetime | None = None
    failure_code: str | None = None
    failure_cause: Mapping[str, object] | None = None


@dataclass(frozen=True)
class MaintenanceFreshnessProjection:
    """Project target/current sessions, readiness and maintenance disposition.

    Attributes:
        target_market_session: Requested market session.
        current_market_session: Active panel session when present.
        phase: Recorded maintenance phase.
        status: Recorded maintenance lifecycle.
        readiness: Workspace readiness status.
        retry_after_at: Optional next retry clock.
        failure_code: Optional stable failure cause.
    """

    target_market_session: date
    current_market_session: date | None
    phase: MaintenancePhase
    status: MaintenanceStatus
    readiness: str
    retry_after_at: datetime | None = None
    failure_code: str | None = None


class _SealedDataUpdate(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def seal(cls, **values: object) -> Self:
        payload = cls.model_construct(**values, content_hash="0" * 64).model_dump(
            mode="json", exclude={"content_hash"}
        )
        return cls(**payload, content_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def check_identity(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("workspace_data_update.identity_invalid")
        return self


class WorkspaceDataUpdateBinding(_SealedDataUpdate):
    """Bind Data updates to the installed market profile, Data policy and Feature catalog.

    Attributes:
        profile_file_hash: Installed profile-file commitment.
        data_policy_hash: Installed Data policy identity.
        feature_catalog_hash: Installed Feature catalog identity.
    """

    kind: Literal["WorkspaceDataUpdateBinding"] = "WorkspaceDataUpdateBinding"
    market_profile_id: Literal["us-current-index-research"] = "us-current-index-research"
    profile_file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkspaceInputStatus(_SealedDataUpdate):
    """Describe current Data, Feature and Foundation inputs with source-verification qualification.

    Attributes:
        manifest_revision: Active membership revision.
        data_revision_hash: Current market-data revision identity.
        data_through: Latest raw-data session when present.
        adjusted_through: Latest adjusted-data session when present.
        panel_hash: Active Feature panel identity.
        panel_through: Active panel session.
        readiness_status: Current readiness qualification.
        sources_checked_at: Last successful source check.
        source_check_failed_at: Optional last failed source check.
        source_disclosure: Optional last-known membership disclosure after source failure.
        foundation_hash: Optional current Foundation identity.
        foundation_panel_hash: Optional Foundation panel binding.
        foundation_disposition: Matching, prior, missing or unreadable Foundation state.
        panel_manifest_revision: Optional panel membership binding.
        listing_set_hash: Optional panel listing-set commitment.
    """

    kind: Literal["WorkspaceInputStatus"] = "WorkspaceInputStatus"
    manifest_revision: str
    data_revision_hash: str
    data_through: date | None
    adjusted_through: date | None
    panel_hash: str
    panel_through: date
    readiness_status: str
    sources_checked_at: datetime | None
    source_check_failed_at: datetime | None = Field(default=None, exclude_if=lambda v: v is None)
    source_disclosure: Literal["LAST_KNOWN_MEMBERSHIP_AFTER_SOURCE_FAILURE"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    foundation_hash: str | None
    foundation_panel_hash: str | None
    foundation_disposition: Literal["MATCHING_PANEL", "PRIOR_INPUTS", "MISSING", "UNREADABLE"]
    panel_manifest_revision: str | None = Field(default=None, exclude_if=lambda v: v is None)
    # The identity of the listings the active manifest admits, so a later
    # revision that admits the same listings (a governance-only change) is
    # told apart from one that changes the membership.
    listing_set_hash: str | None = Field(default=None, exclude_if=lambda v: v is None)


class WorkspaceDataChange(_SealedDataUpdate):
    """Exact Human-reviewable scope, not a user-supplied ticker permission."""

    action: Literal["UNIVERSE", "FULL_HISTORY_AUDIT"]
    transition_id: str | None = None
    candidate_document_hash: str | None = None
    additions: tuple[str, ...] = ()
    removals: tuple[str, ...] = ()
    full_history_listing_ids: tuple[str, ...] = ()
    valuation_manifest: UniverseManifest | None = None
    book_heads: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def scope(self) -> Self:
        """Validate universe-change or full-history-audit scope and book-head uniqueness.

        Returns:
            This validated contract.

        Raises:
            ValueError: Book roots repeat, action-specific fields conflict, valuation scope differs,
                or audit listings are not sorted/unique.
        """
        if len({r for r, _ in self.book_heads}) != len(self.book_heads):
            raise ValueError("workspace_data_update.change_book_scope_invalid")
        if self.action == "UNIVERSE":
            if (
                not self.transition_id
                or not self.candidate_document_hash
                or self.full_history_listing_ids
                or set(self.additions) & set(self.removals)
            ):
                raise ValueError("workspace_data_update.universe_change_invalid")
            if self.valuation_manifest is not None and any(
                v.symbol not in self.removals for v in self.valuation_manifest.listings
            ):
                raise ValueError("workspace_data_update.valuation_scope_invalid")
        elif (
            not self.full_history_listing_ids
            or self.transition_id is not None
            or self.additions
            or self.removals
            or self.valuation_manifest is not None
        ):
            raise ValueError("workspace_data_update.audit_change_invalid")
        if self.full_history_listing_ids != tuple(sorted(set(self.full_history_listing_ids))):
            raise ValueError("workspace_data_update.audit_scope_invalid")
        return self


class WorkspaceValuationGrant(_SealedDataUpdate):
    """Retain approved held-listing valuation scope against named book obligations.

    Attributes:
        approved_plan_hash: Approving immutable plan identity.
        manifest: Retained valuation membership manifest.
        book_roots: Books whose held-listing obligations this grant answers.
        listing_ids: Exact granted listing scope.
    """

    approved_plan_hash: str
    manifest: UniverseManifest
    book_roots: tuple[str, ...]
    listing_ids: tuple[str, ...]


class WorkspaceDataChangeExecution(_SealedDataUpdate):
    """Bind an approved Data update plan to its captured maintenance execution request.

    Attributes:
        plan_hash: Approved immutable plan identity.
        request: Captured execution request.
    """

    plan_hash: str
    request: WorkspaceMaintenanceRequest


class WorkspaceValuationReceipt(_SealedDataUpdate):
    """Seal completed held-listing maintenance evidence under its approved plan.

    Attributes:
        plan_hash: Approving plan identity.
        manifest: Retained valuation scope.
        maintenance_id: Maintenance operation identity.
        target_session: Market session covered by the valuation run.
        outcome_hash: Canonical maintenance outcome commitment.
    """

    plan_hash: str
    manifest: UniverseManifest
    maintenance_id: str
    target_session: date
    outcome_hash: str


class WorkspaceDataUpdatePlan(_SealedDataUpdate):
    """Seal update bindings, prior input status, maintenance scope and valuation obligations.

    Attributes:
        workspace_id: Workspace identity.
        workspace_manifest_hash: Workspace configuration commitment.
        binding: Installed Data update owner bindings.
        before: Input status observed before execution.
        request: Sealed maintenance request.
        change: Optional admitted universe/audit change.
        valuation_grants: Existing held-listing valuation grants to preserve.
    """

    kind: Literal["WorkspaceDataUpdatePlan"] = "WorkspaceDataUpdatePlan"
    workspace_id: str
    workspace_manifest_hash: str
    binding: WorkspaceDataUpdateBinding
    before: WorkspaceInputStatus
    request: WorkspaceMaintenanceRequest
    change: WorkspaceDataChange | None = Field(default=None, exclude_if=lambda v: v is None)
    valuation_grants: tuple[WorkspaceValuationGrant, ...] = Field(
        default=(), exclude_if=lambda v: not v
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def check_request(self) -> Self:
        """Recreate request identity and verify all plan/input/policy bindings.

        Returns:
            This validated contract.

        Raises:
            ValueError: Recreated request identity, membership, profile, Data/Feature policy or
                full-history change scope differs.
        """
        values = {item.name: getattr(self.request, item.name) for item in fields(self.request)}
        recorded = values.pop("request_hash")
        if WorkspaceMaintenanceRequest.create(**values).request_hash != recorded:
            raise ValueError("workspace_data_update.request_identity_invalid")
        if (
            self.request.membership_revision != self.before.manifest_revision
            or self.request.market_profile_id != self.binding.market_profile_id
            or self.request.data_policy_hash != self.binding.data_policy_hash
            or self.request.feature_policy_hash
            != canonical_hash(
                {"catalog": self.binding.feature_catalog_hash, "invalidation": "domain-topology"}
            )
            or (
                self.request.full_history_listing_ids
                and (
                    self.change is None
                    or self.request.full_history_listing_ids != self.change.full_history_listing_ids
                )
            )
        ):
            raise ValueError("workspace_data_update.request_binding_invalid")
        return self


class WorkspaceDataUpdateReceipt(_SealedDataUpdate):
    """Seal completed Data update lineage with before/after inputs and committed effects.

    Attributes:
        plan_hash: Approved plan identity.
        maintenance_request_hash: Executed maintenance request identity.
        cycle_id: Durable maintenance cycle identity.
        target_session: Requested market session.
        completed_at: Completion clock.
        before: Prior input status.
        after: Verified resulting input status.
        child_task_refs: Completed child task references.
        transition_receipt_hash: Optional universe-transition receipt.
        valuation_receipt_hashes: Committed held-listing valuation receipts.
        effect_receipts: Committed effect references.
    """

    kind: Literal["WorkspaceDataUpdateReceipt"] = "WorkspaceDataUpdateReceipt"
    plan_hash: str
    maintenance_request_hash: str
    cycle_id: str
    target_session: date
    completed_at: datetime
    before: WorkspaceInputStatus
    after: WorkspaceInputStatus
    child_task_refs: tuple[str, ...]
    transition_receipt_hash: str | None = Field(default=None, exclude_if=lambda v: v is None)
    valuation_receipt_hashes: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    effect_receipts: tuple[str, ...]
