"""Typed, host-owned remediation contracts for playpen Data Operations cases."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

_DATA_OPERATIONS_REQUEST_NAMESPACE = UUID("1c881a8f-9961-423b-9982-bf7ad17ac37d")


def canonical_hash(value: object) -> str:
    """Hash a JSON-mode model or JSON-shaped value without implementation state."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RemediationAction(StrEnum):
    """Available host-owned actions for a Data Operations case."""

    RETRY_PRIMARY = "retry_primary"
    REFRESH_ADJUSTMENT_DIAGNOSTIC = "refresh_adjustment_diagnostic"
    WAIT_THEN_RETRY = "wait_then_retry"
    QUARANTINE_LISTING = "quarantine_listing"
    EXCLUDE_FROM_NEXT_MANIFEST = "exclude_from_next_manifest"
    REQUALIFY_LISTING = "requalify_listing"
    USE_LAST_KNOWN_GOOD = "use_last_known_good"
    RETAIN_RAW_VALUE_WITH_CAVEAT = "retain_raw_value_with_caveat"
    USE_VERIFIED_ALIAS = "use_verified_alias"
    ESCALATE_FOR_HUMAN = "escalate_for_human"


class PolicyDisposition(StrEnum):
    """Whether a remediation option can proceed automatically."""

    AUTO = "auto"
    HUMAN_REVIEW = "human_review"


class RetryPrimaryArgs(BaseModel):
    """Bounded retry of the primary provider over an explicit range."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.RETRY_PRIMARY]
    provider: str = Field(min_length=1, max_length=80)
    range_start: date
    range_end: date
    max_attempts: int = Field(ge=1, le=3)


class RefreshAdjustmentDiagnosticArgs(BaseModel):
    """Re-fetch one full provider adjustment view; never modify action facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.REFRESH_ADJUSTMENT_DIAGNOSTIC]
    provider: str = Field(min_length=1, max_length=80)
    range_start: date
    range_end: date


class WaitThenRetryArgs(BaseModel):
    """Provider retry deferred for a bounded interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.WAIT_THEN_RETRY]
    retry_after_seconds: int = Field(ge=1, le=3_600)
    provider: str = Field(min_length=1, max_length=80)
    range_start: date
    range_end: date
    max_attempts: int = Field(ge=1, le=3)


class QuarantineListingArgs(BaseModel):
    """Quarantine a listing with a typed reason and optional recheck time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.QUARANTINE_LISTING]
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,119}$")
    recheck_after_at: datetime | None = None


class ExcludeFromNextManifestArgs(BaseModel):
    """Exclude a listing from a new revision; never mutate the active revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.EXCLUDE_FROM_NEXT_MANIFEST]
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,119}$")
    recheck_after_at: datetime


class RequalifyListingArgs(BaseModel):
    """Re-admit only after a host-owned deterministic qualification receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.REQUALIFY_LISTING]
    qualification_receipt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class UseLastKnownGoodArgs(BaseModel):
    """Keep an already-bound frozen snapshot; never advertise it as current."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.USE_LAST_KNOWN_GOOD]
    snapshot_ref: str = Field(pattern=r"^playpen://feature-panel/manifests/[a-f0-9]{64}$")
    bound_market_as_of_session: date


class RetainRawValueWithCaveatArgs(BaseModel):
    """Human-only decision for an unexplained isolated raw-price move."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.RETAIN_RAW_VALUE_WITH_CAVEAT]
    caveat_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{2,119}$")


class VerifiedAliasArgs(BaseModel):
    """Provider alias backed by a recorded listing identity check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.USE_VERIFIED_ALIAS]
    candidate_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,119}$")
    provider_symbol: str = Field(min_length=1, max_length=80)
    identity_evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class EscalateForHumanArgs(BaseModel):
    """Human review request for a typed remediation question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[RemediationAction.ESCALATE_FOR_HUMAN]
    review_kind: str = Field(pattern=r"^[a-z][a-z0-9_]{2,79}$")


PolicyArgs = Annotated[
    RetryPrimaryArgs
    | RefreshAdjustmentDiagnosticArgs
    | WaitThenRetryArgs
    | QuarantineListingArgs
    | ExcludeFromNextManifestArgs
    | RequalifyListingArgs
    | UseLastKnownGoodArgs
    | RetainRawValueWithCaveatArgs
    | VerifiedAliasArgs
    | EscalateForHumanArgs,
    Field(discriminator="action"),
]


class RemediationOption(BaseModel):
    """A host-built action with all executable parameters fixed before the Agent sees it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    option_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,119}$")
    target_listing_ids: tuple[str, ...] = Field(min_length=1)
    disposition: PolicyDisposition
    policy_args: PolicyArgs
    option_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_hash_and_disposition(self) -> RemediationOption:
        """Check that disposition and hash match the immutable option payload."""
        expected_disposition = (
            PolicyDisposition.HUMAN_REVIEW
            if self.policy_args.action
            in {
                RemediationAction.ESCALATE_FOR_HUMAN,
                RemediationAction.RETAIN_RAW_VALUE_WITH_CAVEAT,
            }
            else PolicyDisposition.AUTO
        )
        if self.disposition is not expected_disposition:
            raise ValueError("remediation option disposition does not match action")
        payload = self.model_dump(mode="json", exclude={"option_hash"})
        if self.option_hash != canonical_hash(payload):
            raise ValueError("remediation option hash does not match immutable payload")
        return self

    @classmethod
    def create(
        cls,
        *,
        option_id: str,
        target_listing_ids: tuple[str, ...],
        disposition: PolicyDisposition,
        policy_args: PolicyArgs,
    ) -> RemediationOption:
        """Build an option with a hash of its executable parameters."""
        payload = {
            "option_id": option_id,
            "target_listing_ids": target_listing_ids,
            "disposition": disposition.value,
            "policy_args": policy_args.model_dump(mode="json"),
        }
        return cls(option_hash=canonical_hash(payload), **payload)


class ResearchUniverseDisclosure(BaseModel):
    """Safe user-visible explanation of the admitted research scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int = Field(ge=1, le=2_000)
    admitted_count: int = Field(ge=1, le=2_000)
    quarantined_count: int = Field(ge=0, le=2_000)
    quarantine_reason_counts: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_counts(self) -> ResearchUniverseDisclosure:
        """Require admitted and quarantined counts to reconcile."""
        if self.admitted_count + self.quarantined_count != self.candidate_count:
            raise ValueError("research universe disclosure counts do not reconcile")
        if sum(self.quarantine_reason_counts.values()) < self.quarantined_count:
            raise ValueError("quarantine reason summary does not cover excluded listings")
        return self


class FeaturePanelInputBinding(BaseModel):
    """Immutable panel snapshot selected before a research task is admitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_ref: str = Field(pattern=r"^playpen://feature-panel/manifests/[a-f0-9]{64}$")
    snapshot_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    panel_binding_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    panel_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    sector_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    catalog_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    spy_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    temporal_identity_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    temporal_risk_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    as_of_session: date

    @model_validator(mode="after")
    def validate_snapshot_identity(self) -> FeaturePanelInputBinding:
        """Require the snapshot reference to contain its declared hash."""
        if not self.snapshot_ref.endswith(self.snapshot_hash):
            raise ValueError("panel snapshot ref does not match snapshot hash")
        return self


class DataOperationsRequest(BaseModel):
    """A stable, deterministic successor to a Front Desk handoff.

    This is business input, not an execution frame.  A durable Task may have
    several executions while it keeps this one immutable request identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: UUID
    source_handoff_id: UUID
    session_id: UUID
    market: Literal["US"]
    manifest_revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    research_scope: Literal["explicit_symbols", "active_research_universe"]
    listing_ids: tuple[str, ...] = Field(min_length=1, max_length=1_000)
    symbols: tuple[str, ...] = Field(min_length=1, max_length=1_000)
    universe_disclosure: ResearchUniverseDisclosure
    panel_input: FeaturePanelInputBinding | None = None
    requested_range_start: date
    as_of_session: date
    research_goal: str = Field(min_length=3, max_length=400)
    request_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_request(self) -> DataOperationsRequest:
        """Check request scope, dates, and deterministic payload hash."""
        if self.symbols != tuple(sorted(set(self.symbols))):
            raise ValueError("DataOperationsRequest symbols must be sorted and unique")
        if len(self.listing_ids) != len(self.symbols):
            raise ValueError("DataOperationsRequest listing identity does not match symbols")
        if self.requested_range_start > self.as_of_session:
            raise ValueError("DataOperationsRequest range start is after as_of session")
        if self.panel_input is not None and self.panel_input.as_of_session < self.as_of_session:
            raise ValueError("panel snapshot does not cover the requested as-of session")
        payload = self.model_dump(mode="json", exclude={"request_hash"})
        if self.request_hash != canonical_hash(payload):
            raise ValueError("DataOperationsRequest hash does not match immutable payload")
        return self

    @classmethod
    def create(
        cls,
        *,
        source_handoff_id: UUID,
        session_id: UUID,
        manifest_revision: str,
        listing_ids: tuple[str, ...],
        symbols: tuple[str, ...],
        requested_range_start: date,
        as_of_session: date,
        research_goal: str,
        research_scope: Literal[
            "explicit_symbols", "active_research_universe"
        ] = "explicit_symbols",
        universe_disclosure: ResearchUniverseDisclosure | None = None,
        panel_input: FeaturePanelInputBinding | None = None,
    ) -> DataOperationsRequest:
        """Create a stable request ID and hash from the admitted business input."""
        disclosure = universe_disclosure or ResearchUniverseDisclosure(
            candidate_count=len(symbols),
            admitted_count=len(symbols),
            quarantined_count=0,
            quarantine_reason_counts={},
        )
        request_identity = {
            "source_handoff_id": source_handoff_id,
            "session_id": session_id,
            "market": "US",
            "manifest_revision": manifest_revision,
            "research_scope": research_scope,
            "listing_ids": listing_ids,
            "symbols": symbols,
            "universe_disclosure": disclosure.model_dump(mode="json"),
            "panel_input": panel_input.model_dump(mode="json") if panel_input else None,
            "requested_range_start": requested_range_start,
            "as_of_session": as_of_session,
            "research_goal": research_goal,
        }
        payload = {
            "request_id": uuid5(
                _DATA_OPERATIONS_REQUEST_NAMESPACE,
                canonical_hash(request_identity),
            ),
            **request_identity,
        }
        return cls(request_hash=canonical_hash(payload), **payload)


class PolicyDecision(BaseModel):
    """Chosen remediation option and the case evidence it answers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_token: str = Field(pattern=r"^[a-f0-9]{64}$")
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    option_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,119}$")
    option_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    disposition: PolicyDisposition
