"""Deterministic Git-like state transition contracts for Pre-Research.

This module owns no market-data, feature, panel, or Factor calculation.  It
compares frozen semantic state, compiles the exact work authorities that those
owners may consume, and advances the verified HEAD with compare-and-swap.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListingDataDeltaKind(StrEnum):
    """Classify the semantic data work required for one observed listing.

    Distinguishes unchanged/evidence-only rows, bounded tails, full-history additions,
    raw or adjusted-return corrections, active removals, insufficient history and
    ambiguous sources. The classification determines owner work authority.
    """

    UNCHANGED = "UNCHANGED"
    TAIL_APPEND_REQUIRED = "TAIL_APPEND_REQUIRED"
    FULL_HISTORY_ADDITION_REQUIRED = "FULL_HISTORY_ADDITION_REQUIRED"
    RAW_SEMANTIC_CORRECTION = "RAW_SEMANTIC_CORRECTION"
    ADJUSTED_RETURN_CORRECTION = "ADJUSTED_RETURN_CORRECTION"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    REMOVE_FROM_ACTIVE = "REMOVE_FROM_ACTIVE"
    INSUFFICIENT_RESEARCH_HISTORY = "INSUFFICIENT_RESEARCH_HISTORY"
    BLOCKED_SOURCE_AMBIGUITY = "BLOCKED_SOURCE_AMBIGUITY"


class PanelDeltaDisposition(StrEnum):
    """Declare whether a Panel is reused, sparsely rebuilt, fully rebuilt or blocked.

    The disposition is derived from the observed transition rather than selected by
    a numerical producer.
    """

    NOOP = "NOOP"
    SPARSE_REBUILD = "SPARSE_REBUILD"
    FULL_HISTORY_REBUILD = "FULL_HISTORY_REBUILD"
    BLOCKED = "BLOCKED"


class FactorDeltaDisposition(StrEnum):
    """Declare the Factor evidence authority required by an observed transition.

    Exact reuse, a reuse assessment, fixed screening and a blocked transition are
    distinct dispositions; none grants a producer authority to broaden its scope.
    """

    REUSE_EXACT_ALLOWED = "REUSE_EXACT_ALLOWED"
    REUSE_ASSESSMENT_REQUIRED = "REUSE_ASSESSMENT_REQUIRED"
    FIXED_SCREENING_REQUIRED = "FIXED_SCREENING_REQUIRED"
    BLOCKED = "BLOCKED"


class PreResearchRevisionDisposition(StrEnum):
    """Classify the verified pre-research state transition.

    Tracks a no-op, a source change without active membership change, bounded
    maintenance, membership revision, initialization, or a blocked transition.
    """

    NOOP = "NOOP"
    SOURCE_CHANGED_ACTIVE_SET_UNCHANGED = "SOURCE_CHANGED_ACTIVE_SET_UNCHANGED"
    BOUNDED_MAINTENANCE = "BOUNDED_MAINTENANCE"
    MEMBERSHIP_REVISION = "MEMBERSHIP_REVISION"
    INITIALIZATION_REQUIRED = "INITIALIZATION_REQUIRED"
    BLOCKED = "BLOCKED"


class QuarantineReason(StrEnum):
    """State why a candidate listing cannot enter the qualified active set.

    History, source ambiguity, quality qualification and unavailable sector evidence
    remain explicit reasons rather than disappearing from the transition record.
    """

    INSUFFICIENT_RESEARCH_HISTORY = "INSUFFICIENT_RESEARCH_HISTORY"
    SOURCE_AMBIGUITY = "SOURCE_AMBIGUITY"
    QUALITY_NOT_QUALIFIED = "QUALITY_NOT_QUALIFIED"
    SECTOR_EVIDENCE_UNAVAILABLE = "SECTOR_EVIDENCE_UNAVAILABLE"


class PreResearchListingObservation(_Contract):
    """Freeze one listing's source membership, local history and semantic impact.

    History bounds occur together. A source ambiguity cannot claim a known session
    set; affected factors and sessions are canonical sorted unique axes. Optional
    impact fields distinguish numerical corrections from evidence-only changes.
    """

    listing_id: str = Field(min_length=1, max_length=200)
    source_member: bool
    local_history_start: date | None = None
    local_history_end: date | None = None
    data_state_hash: str | None = None
    raw_semantic_delta: bool = False
    adjusted_return_semantic_delta: bool = False
    evidence_only_delta: bool = False
    semantic_delta_kind: ListingDataDeltaKind | None = None
    affected_factor_ids: tuple[str, ...] = ()
    affected_sessions: tuple[date, ...] = ()
    session_set_known: bool = True
    source_ambiguous: bool = False

    @model_validator(mode="after")
    def validate_observation(self) -> PreResearchListingObservation:
        """Verify history bounds, source certainty and canonical impact axes.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: History bounds are incomplete/reversed, source ambiguity claims known
                sessions, or impact axes are noncanonical.
        """
        if (self.local_history_start is None) != (self.local_history_end is None):
            raise ValueError("listing local history bounds must be present together")
        if (
            self.local_history_start is not None
            and self.local_history_end is not None
            and self.local_history_start > self.local_history_end
        ):
            raise ValueError("listing local history interval is inverted")
        if self.source_ambiguous and self.session_set_known:
            raise ValueError("ambiguous source cannot claim a known session set")
        if self.affected_factor_ids != tuple(sorted(set(self.affected_factor_ids))):
            raise ValueError("affected factor IDs must be unique and sorted")
        if self.affected_sessions != tuple(sorted(set(self.affected_sessions))):
            raise ValueError("affected sessions must be unique and sorted")
        return self


class VerifiedPreResearchHead(_Contract):
    """Bind the complete verified research prefix behind the active HEAD pointer.

    Retains unique ordered source/active/factor axes, source and catalog revisions,
    Panel/Factor/execution/Foundation lineage, optional paired logical and adjusted
    return bindings, research validity and a verification clock. The clock is excluded
    from its content identity; current-universe evidence is not historical PIT evidence.
    """

    kind: Literal["VerifiedPreResearchHead"] = "VerifiedPreResearchHead"
    market_profile_id: str = Field(min_length=1, max_length=160)
    manifest_revision: str
    source_revision: str
    membership_fingerprint: str
    ordered_source_candidate_listing_ids: tuple[str, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    listing_set_hash: str
    research_history_start: date
    target_market_session: date
    sector_revision: str
    catalog_hash: str
    panel_snapshot_hash: str
    logical_panel_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    logical_semantic_index_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    factor_screening_result_hash: str
    factor_candidate_slate_hash: str
    research_desk_factor_input_hash: str
    causal_execution_outcome_snapshot_hash: str
    research_foundation_hash: str
    research_foundation_marker_hash: str
    ordered_factor_ids: tuple[str, ...] = Field(min_length=1, max_length=55)
    data_state_hash: str
    adjusted_return_revision_cursor: int | None = Field(default=None, ge=0)
    adjusted_return_revision_chain_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    validity_class: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"] = "CURRENT_UNIVERSE_RESEARCH_ONLY"
    is_point_in_time_historical: Literal[False] = False
    verified_at: datetime
    head_hash: str

    @model_validator(mode="after")
    def validate_head(self) -> VerifiedPreResearchHead:
        """Verify HEAD axes, complete lineage, research interval and canonical identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The clock, unique axes, membership hashes, optional paired bindings,
                interval, or content identity is invalid.
        """
        _require_aware(self.verified_at, "HEAD verification")
        if self.ordered_listing_ids != tuple(dict.fromkeys(self.ordered_listing_ids)):
            raise ValueError("HEAD listing IDs must be unique and ordered")
        if self.ordered_source_candidate_listing_ids != tuple(
            dict.fromkeys(self.ordered_source_candidate_listing_ids)
        ):
            raise ValueError("HEAD source candidate IDs must be unique and ordered")
        if self.membership_fingerprint != canonical_hash(self.ordered_source_candidate_listing_ids):
            raise ValueError("HEAD source membership fingerprint is invalid")
        if self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids)):
            raise ValueError("HEAD factor IDs must be unique and ordered")
        if (self.logical_panel_hash is None) != (self.logical_semantic_index_hash is None):
            raise ValueError("HEAD native logical Panel identity is incomplete")
        if self.listing_set_hash != canonical_hash(self.ordered_listing_ids):
            raise ValueError("HEAD listing-set hash is invalid")
        if self.research_history_start > self.target_market_session:
            raise ValueError("HEAD Research History Window is inverted")
        identity = self.model_dump(mode="json", exclude={"head_hash", "verified_at"})
        if self.logical_panel_hash is None:
            identity.pop("logical_panel_hash", None)
            identity.pop("logical_semantic_index_hash", None)
        if self.adjusted_return_revision_cursor is None:
            identity.pop("adjusted_return_revision_cursor", None)
        if self.adjusted_return_revision_chain_hash is None:
            identity.pop("adjusted_return_revision_chain_hash", None)
        if (self.adjusted_return_revision_cursor is None) != (
            self.adjusted_return_revision_chain_hash is None
        ):
            raise ValueError("HEAD adjusted-return revision binding is incomplete")
        if self.head_hash != canonical_hash(identity):
            raise ValueError("HEAD content identity is invalid")
        return self


class ObservedPreResearchCandidateSnapshot(_Contract):
    """Freeze the locally observed source candidate against its expected base HEAD.

    The candidate commits to source membership, sorted per-listing observations,
    history bounds, target session and installed policies. Its observation clock is
    excluded from identity; empty optional impact fields preserve older record forms.
    """

    kind: Literal["ObservedPreResearchCandidateSnapshot"] = "ObservedPreResearchCandidateSnapshot"
    market_profile_id: str = Field(min_length=1, max_length=160)
    base_head_hash: str | None
    source_revision: str
    source_membership_fingerprint: str
    target_market_session: date
    research_history_start: date
    ordered_candidate_listing_ids: tuple[str, ...] = Field(min_length=1)
    listing_observations: tuple[PreResearchListingObservation, ...] = Field(min_length=1)
    sector_revision: str | None
    adjusted_return_revision_cursor: int | None = Field(default=None, ge=0)
    adjusted_return_revision_chain_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_hashes: tuple[str, ...] = Field(min_length=1)
    observed_at: datetime
    candidate_snapshot_hash: str

    @model_validator(mode="after")
    def validate_snapshot(self) -> ObservedPreResearchCandidateSnapshot:
        """Verify candidate membership, observations and compatible canonical identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Clock/interval, unique source axis, observation membership/order,
                adjusted-return binding, or identity is invalid.
        """
        _require_aware(self.observed_at, "candidate observation")
        if self.research_history_start > self.target_market_session:
            raise ValueError("candidate Research History Window is inverted")
        if self.ordered_candidate_listing_ids != tuple(
            dict.fromkeys(self.ordered_candidate_listing_ids)
        ):
            raise ValueError("candidate listing IDs must be unique and ordered")
        if self.source_membership_fingerprint != canonical_hash(self.ordered_candidate_listing_ids):
            raise ValueError("candidate source membership fingerprint is invalid")
        observed_ids = tuple(value.listing_id for value in self.listing_observations)
        if observed_ids != tuple(sorted(observed_ids)) or len(observed_ids) != len(
            set(observed_ids)
        ):
            raise ValueError("candidate observations must be unique and sorted")
        source_ids = tuple(
            value.listing_id for value in self.listing_observations if value.source_member
        )
        if set(source_ids) != set(self.ordered_candidate_listing_ids):
            raise ValueError("candidate membership differs from listing observations")
        identity = self.model_dump(mode="json", exclude={"candidate_snapshot_hash", "observed_at"})
        _remove_empty_observation_impact(identity)
        if self.adjusted_return_revision_cursor is None:
            identity.pop("adjusted_return_revision_cursor", None)
        if self.adjusted_return_revision_chain_hash is None:
            identity.pop("adjusted_return_revision_chain_hash", None)
        if (self.adjusted_return_revision_cursor is None) != (
            self.adjusted_return_revision_chain_hash is None
        ):
            raise ValueError("candidate adjusted-return revision binding is incomplete")
        if self.candidate_snapshot_hash != canonical_hash(identity):
            raise ValueError("candidate snapshot identity is invalid")
        return self


class PreResearchListingDelta(_Contract):
    """Record one listing's classified work and known numerical impact.

    The source data-state reference, affected factors and affected sessions accompany
    the delta kind so downstream owners can receive the exact admitted scope.
    """

    listing_id: str
    kind: ListingDataDeltaKind
    affected_factor_ids: tuple[str, ...] = ()
    affected_sessions: tuple[date, ...] = ()
    source_data_state_hash: str | None = None


class PreResearchObservedDelta(_Contract):
    """Seal the difference between frozen source observations and verified research state.

    Retains source additions/removals, prior active membership, ordered listing deltas,
    required authorities, blockers and Panel/Factor/revision dispositions. Source
    membership change remains distinct from the qualified active-set outcome.
    """

    kind: Literal["PreResearchObservedDelta"] = "PreResearchObservedDelta"
    base_head_hash: str | None
    candidate_snapshot_hash: str
    research_history_start: date
    target_market_session: date
    base_adjusted_return_revision_cursor: int | None = Field(default=None, ge=0)
    base_adjusted_return_revision_chain_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    observed_adjusted_return_revision_cursor: int | None = Field(default=None, ge=0)
    observed_adjusted_return_revision_chain_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    prior_active_listing_ids: tuple[str, ...]
    additions: tuple[str, ...]
    removals: tuple[str, ...]
    retained: tuple[str, ...]
    listing_deltas: tuple[PreResearchListingDelta, ...]
    source_membership_changed: bool
    source_changed_active_set_may_remain_unchanged: bool
    panel_disposition: PanelDeltaDisposition
    factor_disposition: FactorDeltaDisposition
    revision_disposition: PreResearchRevisionDisposition
    blockers: tuple[str, ...]
    required_authorities: tuple[str, ...]
    delta_hash: str

    @model_validator(mode="after")
    def validate_delta(self) -> PreResearchObservedDelta:
        """Verify delta axes, work dispositions and optional revision bindings.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Addition/removal axes overlap, listing deltas are noncanonical, dispositions
                contradict work/blockers, or identity is invalid.
        """
        if set(self.additions) & set(self.removals):
            raise ValueError("source additions and removals must be disjoint")
        ids = tuple(value.listing_id for value in self.listing_deltas)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("listing deltas must be unique and sorted")
        if self.revision_disposition is PreResearchRevisionDisposition.NOOP and any(
            value.kind is not ListingDataDeltaKind.UNCHANGED for value in self.listing_deltas
        ):
            raise ValueError("NOOP delta contains work")
        if (
            self.blockers
            and self.revision_disposition is not PreResearchRevisionDisposition.BLOCKED
        ):
            raise ValueError("blocked delta must use BLOCKED disposition")
        identity = self.model_dump(mode="json", exclude={"delta_hash"})
        for cursor_name, chain_name in (
            (
                "base_adjusted_return_revision_cursor",
                "base_adjusted_return_revision_chain_hash",
            ),
            (
                "observed_adjusted_return_revision_cursor",
                "observed_adjusted_return_revision_chain_hash",
            ),
        ):
            cursor = getattr(self, cursor_name)
            chain = getattr(self, chain_name)
            if (cursor is None) != (chain is None):
                raise ValueError("delta adjusted-return revision binding is incomplete")
            if cursor is None:
                identity.pop(cursor_name, None)
                identity.pop(chain_name, None)
        if self.delta_hash != canonical_hash(identity):
            raise ValueError("observed delta identity is invalid")
        return self


class PreResearchQuarantine(_Contract):
    """Retain a listing's exclusion reason and supporting evidence identity.

    A quarantined listing cannot simultaneously belong to the staged qualified axis.
    """

    listing_id: str
    reason: QuarantineReason
    evidence_hash: str


class PreResearchStagedDelta(_Contract):
    """Bind qualification results and staged children to the observed delta.

    The qualified listing axis and its hash determine whether the active set changed.
    Quarantines remain explicit and optional adjusted-return revision bindings occur
    together; a source-only membership change has its own revision disposition.
    """

    kind: Literal["PreResearchStagedDelta"] = "PreResearchStagedDelta"
    observed_delta_hash: str
    ordered_qualified_listing_ids: tuple[str, ...] = Field(min_length=1)
    qualified_listing_set_hash: str
    quarantines: tuple[PreResearchQuarantine, ...]
    active_set_changed: bool
    revision_disposition: PreResearchRevisionDisposition
    adjusted_return_revision_cursor: int | None = Field(default=None, ge=0)
    adjusted_return_revision_chain_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    staged_child_hashes: tuple[str, ...]
    staged_delta_hash: str

    @model_validator(mode="after")
    def validate_staging(self) -> PreResearchStagedDelta:
        """Verify the qualified axis, quarantine exclusion and staging identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Qualified axes/hashes differ, quarantine overlaps active membership,
                revision binding is incomplete, or identity is invalid.
        """
        if self.ordered_qualified_listing_ids != tuple(
            dict.fromkeys(self.ordered_qualified_listing_ids)
        ):
            raise ValueError("qualified listing IDs must be unique and ordered")
        if self.qualified_listing_set_hash != canonical_hash(self.ordered_qualified_listing_ids):
            raise ValueError("qualified listing-set hash is invalid")
        if set(value.listing_id for value in self.quarantines) & set(
            self.ordered_qualified_listing_ids
        ):
            raise ValueError("quarantined listings cannot enter the active set")
        identity = self.model_dump(mode="json", exclude={"staged_delta_hash"})
        if (self.adjusted_return_revision_cursor is None) != (
            self.adjusted_return_revision_chain_hash is None
        ):
            raise ValueError("staged adjusted-return revision binding is incomplete")
        if self.adjusted_return_revision_cursor is None:
            identity.pop("adjusted_return_revision_cursor", None)
            identity.pop("adjusted_return_revision_chain_hash", None)
        if self.staged_delta_hash != canonical_hash(identity):
            raise ValueError("staged delta identity is invalid")
        return self


class PreResearchUpdatePlan(_Contract):
    """Compile disjoint owner I/O scopes and required transition authorities.

    Full-history additions, bounded retained work, removals and evidence-only work
    have disjoint axes. Rebuild/screening/publication requirements, user confirmation
    and installed policy identities are part of the sealed plan.
    """

    kind: Literal["PreResearchUpdatePlan"] = "PreResearchUpdatePlan"
    base_head_hash: str | None
    candidate_snapshot_hash: str
    delta_hash: str
    full_history_addition_listing_ids: tuple[str, ...]
    bounded_retained_listing_ids: tuple[str, ...]
    removed_listing_ids: tuple[str, ...]
    evidence_only_listing_ids: tuple[str, ...]
    require_full_panel_rebuild: bool
    require_fixed_factor_screening: bool
    require_foundation_publication: bool
    user_confirmation_required: bool
    operations: tuple[str, ...]
    policy_hashes: tuple[str, ...]
    update_plan_hash: str

    @model_validator(mode="after")
    def validate_plan(self) -> PreResearchUpdatePlan:
        """Verify disjoint owner I/O scopes and the canonical plan identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Work scopes overlap or the update-plan hash differs from its contents.
        """
        scopes = (
            self.full_history_addition_listing_ids
            + self.bounded_retained_listing_ids
            + self.removed_listing_ids
            + self.evidence_only_listing_ids
        )
        if len(scopes) != len(set(scopes)):
            raise ValueError("update plan I/O scopes must be disjoint")
        if self.update_plan_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"update_plan_hash"})
        ):
            raise ValueError("update plan identity is invalid")
        return self


class ConfirmedPreResearchUpdateMandate(_Contract):
    """Bind user confirmation to one exact frozen tree and compiled update plan.

    The base HEAD, candidate, delta, plan, installed policies and confirmation token
    belong to the mandate identity. Its timezone-aware confirmation clock is recorded
    separately from that identity.
    """

    kind: Literal["ConfirmedPreResearchUpdateMandate"] = "ConfirmedPreResearchUpdateMandate"
    base_head_hash: str | None
    candidate_snapshot_hash: str
    delta_hash: str
    update_plan_hash: str
    policy_hashes: tuple[str, ...]
    confirmed_at: datetime
    confirmation_token: str = Field(min_length=16, max_length=300)
    mandate_hash: str

    @model_validator(mode="after")
    def validate_mandate(self) -> ConfirmedPreResearchUpdateMandate:
        """Verify the confirmation clock and exact frozen-plan mandate identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The confirmation clock is naive or the mandate hash is invalid.
        """
        _require_aware(self.confirmed_at, "update confirmation")
        if self.mandate_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"mandate_hash", "confirmed_at"})
        ):
            raise ValueError("update mandate identity is invalid")
        return self


class PreResearchPendingTransition(_Contract):
    """Durable control pointer to one frozen, not-yet-activated transition."""

    kind: Literal["PreResearchPendingTransition"] = "PreResearchPendingTransition"
    base_head_hash: str | None
    candidate_snapshot_hash: str
    delta_hash: str
    update_plan_hash: str
    mandate_hash: str | None
    frozen_at: datetime
    pending_hash: str

    @model_validator(mode="after")
    def validate_pending(self) -> PreResearchPendingTransition:
        """Verify the frozen pending-transition clock and canonical child index.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The frozen clock is naive or the pending-transition hash is invalid.
        """
        _require_aware(self.frozen_at, "pending transition")
        if self.pending_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"pending_hash", "frozen_at"})
        ):
            raise ValueError("pending transition identity is invalid")
        return self


class PreResearchStorageEvidence(_Contract):
    """Reconcile Panel chunk disposition and measured reachable storage.

    New and reused chunks cover the active snapshot. Compressed/uncompressed sizes
    and cumulative reachable bytes remain explicit evidence, with HEAD, milestone
    or historical reachability classified separately.
    """

    compressed_bytes: int = Field(ge=0)
    uncompressed_bytes: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    newly_written_chunks: int = Field(ge=0)
    content_reused_chunks: int = Field(ge=0)
    cumulative_reachable_panel_bytes: int = Field(ge=0)
    reachability_class: Literal["HEAD", "MILESTONE", "HISTORY"]

    @model_validator(mode="after")
    def validate_counts(self) -> PreResearchStorageEvidence:
        """Reconcile new/reused chunks and reachable active storage.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Chunk dispositions do not cover the snapshot or reachable bytes are below
                active compressed bytes.
        """
        if self.newly_written_chunks + self.content_reused_chunks != self.chunk_count:
            raise ValueError("Panel storage chunk disposition does not cover the snapshot")
        if self.cumulative_reachable_panel_bytes < self.compressed_bytes:
            raise ValueError("reachable Panel bytes are smaller than the active snapshot")
        return self


class PreResearchRevisionMarker(_Contract):
    """Seal one verified HEAD transition and its durable children.

    Binds expected base and next HEAD, candidate/delta/staging/plan/mandate references,
    child identities and storage evidence. The aware publication clock is recorded
    but excluded from the marker's canonical content identity.
    """

    kind: Literal["PreResearchRevisionMarker"] = "PreResearchRevisionMarker"
    base_head_hash: str | None
    next_head_hash: str
    candidate_snapshot_hash: str
    delta_hash: str
    staged_delta_hash: str
    update_plan_hash: str
    mandate_hash: str | None
    child_hashes: tuple[str, ...]
    storage_evidence: PreResearchStorageEvidence
    published_at: datetime
    marker_hash: str

    @model_validator(mode="after")
    def validate_marker(self) -> PreResearchRevisionMarker:
        """Verify the publication clock and canonical transition-marker identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The publication clock is naive or the marker hash is invalid.
        """
        _require_aware(self.published_at, "revision marker publication")
        if self.marker_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"marker_hash", "published_at"})
        ):
            raise ValueError("revision marker identity is invalid")
        return self


class PreResearchCasError(RuntimeError):
    """The current verified HEAD differed at the final activation boundary."""


def seal_contract(model: type[_Contract], values: dict[str, object], hash_field: str):
    """Construct a transition contract with the canonical identity assigned to its hash field.

    Args:
        model: Immutable transition contract to construct.
        values: Declared field values used to form the contract.
        hash_field: Identity field to seal from those contents.

    Returns:
        (_Contract): Constructed contract carrying its canonical content identity.

    Raises:
        ValueError: Field values or the resulting contract fail validation.
    """
    return seal_model(model, values, field=hash_field)


def build_verified_head(**values: object) -> VerifiedPreResearchHead:
    """Seal verified HEAD contents without binding the observation clock or absent extensions.

    Args:
        **values: Verified HEAD fields, including its recorded verification clock.

    Returns:
        Validated HEAD; absent logical and adjusted-return bindings are omitted from identity.

    Raises:
        ValueError: Axes, paired lineage, history, clock, or resulting identity is invalid.
    """
    identity = VerifiedPreResearchHead.model_construct(**values, head_hash="").model_dump(
        mode="json", exclude={"head_hash", "verified_at"}
    )
    if identity.get("adjusted_return_revision_cursor") is None:
        identity.pop("adjusted_return_revision_cursor", None)
    if identity.get("adjusted_return_revision_chain_hash") is None:
        identity.pop("adjusted_return_revision_chain_hash", None)
    if identity.get("logical_panel_hash") is None:
        identity.pop("logical_panel_hash", None)
        identity.pop("logical_semantic_index_hash", None)
    return VerifiedPreResearchHead(**values, head_hash=canonical_hash(identity))


def confirm_update_plan(
    *,
    plan: PreResearchUpdatePlan,
    confirmed_at: datetime,
    confirmation_token: str,
) -> ConfirmedPreResearchUpdateMandate:
    """Bind one user confirmation to the exact frozen tree and compiled plan.

    Args:
        plan: Sealed update plan whose source tree and policy identities were reviewed.
        confirmed_at: Timezone-aware confirmation clock retained outside the mandate identity.
        confirmation_token: Bounded token recording confirmation of this exact plan.

    Returns:
        Validated immutable mandate for the frozen candidate, delta and plan.

    Raises:
        ValueError: Confirmation fields or the resulting canonical mandate are invalid.
    """
    values: dict[str, object] = {
        "kind": "ConfirmedPreResearchUpdateMandate",
        "base_head_hash": plan.base_head_hash,
        "candidate_snapshot_hash": plan.candidate_snapshot_hash,
        "delta_hash": plan.delta_hash,
        "update_plan_hash": plan.update_plan_hash,
        "policy_hashes": plan.policy_hashes,
        "confirmed_at": confirmed_at,
        "confirmation_token": confirmation_token,
    }
    identity = ConfirmedPreResearchUpdateMandate.model_construct(
        **values, mandate_hash=""
    ).model_dump(mode="json", exclude={"mandate_hash", "confirmed_at"})
    return ConfirmedPreResearchUpdateMandate(
        **values,
        mandate_hash=canonical_hash(identity),
    )


def _remove_empty_observation_impact(identity: dict[str, object]) -> None:
    """Keep pre-impact candidate artifacts readable while binding real impact."""
    observations = identity.get("listing_observations")
    if not isinstance(observations, list):
        return
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        if not observation.get("affected_factor_ids"):
            observation.pop("affected_factor_ids", None)
        if not observation.get("affected_sessions"):
            observation.pop("affected_sessions", None)
        if observation.get("semantic_delta_kind") is None:
            observation.pop("semantic_delta_kind", None)


def build_candidate_snapshot(**values: object) -> ObservedPreResearchCandidateSnapshot:
    """Seal frozen source observations using the supported candidate identity form.

    Args:
        **values: Candidate fields, including its recorded observation clock.

    Returns:
        Validated candidate with clock, empty impact and absent revision extensions
        omitted from the compatible content identity.

    Raises:
        ValueError: Source membership, observations, bounds, paired revision or clock is invalid.
    """
    identity = ObservedPreResearchCandidateSnapshot.model_construct(
        **values, candidate_snapshot_hash=""
    ).model_dump(mode="json", exclude={"candidate_snapshot_hash", "observed_at"})
    _remove_empty_observation_impact(identity)
    if identity.get("adjusted_return_revision_cursor") is None:
        identity.pop("adjusted_return_revision_cursor", None)
    if identity.get("adjusted_return_revision_chain_hash") is None:
        identity.pop("adjusted_return_revision_chain_hash", None)
    return ObservedPreResearchCandidateSnapshot(
        **values, candidate_snapshot_hash=canonical_hash(identity)
    )


def compute_observed_delta(
    *,
    head: VerifiedPreResearchHead | None,
    candidate: ObservedPreResearchCandidateSnapshot,
) -> PreResearchObservedDelta:
    """Compare one frozen source snapshot against one verified HEAD.

    Args:
        head: Current verified research prefix, or None for initialization.
        candidate: Frozen local source observations naming their expected base HEAD.

    Returns:
        Sealed membership/work delta, exact owner authorities and numerical dispositions.
        Ambiguous sources block work; evidence-only changes may preserve Panel and Factor values.

    Raises:
        ValueError: Base HEAD, market profile or history differs, an initial candidate
            names a base HEAD, or a derived delta fails contract validation.
    """
    if head is not None:
        if candidate.base_head_hash != head.head_hash:
            raise ValueError("CANDIDATE_BASE_HEAD_MISMATCH")
        if candidate.market_profile_id != head.market_profile_id:
            raise ValueError("CANDIDATE_MARKET_PROFILE_MISMATCH")
        if candidate.research_history_start != head.research_history_start:
            raise ValueError("CANDIDATE_HISTORY_WINDOW_CHANGED")
    elif candidate.base_head_hash is not None:
        raise ValueError("INITIAL_CANDIDATE_HAS_BASE_HEAD")

    base_ids = set(head.ordered_listing_ids if head is not None else ())
    base_source_ids = set(head.ordered_source_candidate_listing_ids if head is not None else ())
    candidate_ids = set(candidate.ordered_candidate_listing_ids)
    additions = tuple(sorted(candidate_ids - base_source_ids))
    removals = tuple(sorted(base_source_ids - candidate_ids))
    retained = tuple(sorted(base_ids & candidate_ids))
    observations = {value.listing_id: value for value in candidate.listing_observations}
    deltas: list[PreResearchListingDelta] = []
    blockers: list[str] = []

    for listing_id in sorted(base_ids | candidate_ids):
        observation = observations.get(listing_id)
        if listing_id in base_ids - candidate_ids:
            kind = ListingDataDeltaKind.REMOVE_FROM_ACTIVE
        elif (
            observation is None or observation.source_ambiguous or not observation.session_set_known
        ):
            kind = ListingDataDeltaKind.BLOCKED_SOURCE_AMBIGUITY
            blockers.append(f"SOURCE_AMBIGUITY:{listing_id}")
        elif listing_id in additions:
            if (
                observation.local_history_start is not None
                and observation.local_history_start > candidate.research_history_start
            ):
                kind = ListingDataDeltaKind.INSUFFICIENT_RESEARCH_HISTORY
            else:
                kind = ListingDataDeltaKind.FULL_HISTORY_ADDITION_REQUIRED
        elif listing_id not in base_ids and listing_id in base_source_ids:
            # A previously quality-rejected source member stays quarantined
            # without creating repeated Provider or Panel work every startup.
            kind = ListingDataDeltaKind.UNCHANGED
        elif observation.semantic_delta_kind is not None:
            kind = observation.semantic_delta_kind
        elif observation.adjusted_return_semantic_delta:
            kind = ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION
        elif observation.raw_semantic_delta:
            kind = ListingDataDeltaKind.RAW_SEMANTIC_CORRECTION
        elif observation.local_history_end is None or (
            observation.local_history_end < candidate.target_market_session
        ):
            kind = ListingDataDeltaKind.TAIL_APPEND_REQUIRED
        elif observation.evidence_only_delta:
            kind = ListingDataDeltaKind.EVIDENCE_ONLY
        else:
            kind = ListingDataDeltaKind.UNCHANGED
        deltas.append(
            PreResearchListingDelta(
                listing_id=listing_id,
                kind=kind,
                affected_factor_ids=(observation.affected_factor_ids if observation else ()),
                affected_sessions=(observation.affected_sessions if observation else ()),
                source_data_state_hash=(observation.data_state_hash if observation else None),
            )
        )

    membership_changed = bool(additions or removals)
    active_removals = tuple(sorted(set(removals) & base_ids))
    active_set_may_change = bool(additions or active_removals)
    meaningful = tuple(
        value
        for value in deltas
        if value.kind not in {ListingDataDeltaKind.UNCHANGED, ListingDataDeltaKind.EVIDENCE_ONLY}
    )
    evidence_only = any(value.kind is ListingDataDeltaKind.EVIDENCE_ONLY for value in deltas)
    if blockers:
        revision = PreResearchRevisionDisposition.BLOCKED
        panel = PanelDeltaDisposition.BLOCKED
        factor = FactorDeltaDisposition.BLOCKED
    elif head is None:
        revision = PreResearchRevisionDisposition.INITIALIZATION_REQUIRED
        panel = PanelDeltaDisposition.FULL_HISTORY_REBUILD
        factor = FactorDeltaDisposition.FIXED_SCREENING_REQUIRED
    elif active_set_may_change:
        revision = PreResearchRevisionDisposition.MEMBERSHIP_REVISION
        panel = PanelDeltaDisposition.FULL_HISTORY_REBUILD
        factor = FactorDeltaDisposition.FIXED_SCREENING_REQUIRED
    elif meaningful:
        revision = PreResearchRevisionDisposition.BOUNDED_MAINTENANCE
        panel = PanelDeltaDisposition.SPARSE_REBUILD
        factor = FactorDeltaDisposition.REUSE_ASSESSMENT_REQUIRED
    elif evidence_only or candidate.source_revision != head.source_revision:
        revision = PreResearchRevisionDisposition.SOURCE_CHANGED_ACTIVE_SET_UNCHANGED
        panel = PanelDeltaDisposition.NOOP
        factor = FactorDeltaDisposition.REUSE_EXACT_ALLOWED
    else:
        revision = PreResearchRevisionDisposition.NOOP
        panel = PanelDeltaDisposition.NOOP
        factor = FactorDeltaDisposition.REUSE_EXACT_ALLOWED

    authorities = []
    if any(value.kind is ListingDataDeltaKind.FULL_HISTORY_ADDITION_REQUIRED for value in deltas):
        authorities.append("DATA_FULL_HISTORY_ADDITION")
    if any(value.kind is ListingDataDeltaKind.TAIL_APPEND_REQUIRED for value in deltas):
        authorities.append("DATA_BOUNDED_TAIL")
    if panel is PanelDeltaDisposition.SPARSE_REBUILD:
        authorities.extend(("FEATURE_BOUNDED_INVALIDATION", "PANEL_SPARSE_REBUILD"))
    if panel is PanelDeltaDisposition.FULL_HISTORY_REBUILD:
        authorities.extend(("PANEL_FULL_HISTORY_REBUILD", "FACTOR_FIXED_SCREENING"))
    values: dict[str, object] = {
        "kind": "PreResearchObservedDelta",
        "base_head_hash": head.head_hash if head else None,
        "candidate_snapshot_hash": candidate.candidate_snapshot_hash,
        "research_history_start": candidate.research_history_start,
        "target_market_session": candidate.target_market_session,
        "base_adjusted_return_revision_cursor": (
            head.adjusted_return_revision_cursor if head else None
        ),
        "base_adjusted_return_revision_chain_hash": (
            head.adjusted_return_revision_chain_hash if head else None
        ),
        "observed_adjusted_return_revision_cursor": candidate.adjusted_return_revision_cursor,
        "observed_adjusted_return_revision_chain_hash": (
            candidate.adjusted_return_revision_chain_hash
        ),
        "prior_active_listing_ids": head.ordered_listing_ids if head else (),
        "additions": additions,
        "removals": removals,
        "retained": retained,
        "listing_deltas": tuple(deltas),
        "source_membership_changed": membership_changed,
        "source_changed_active_set_may_remain_unchanged": bool(additions),
        "panel_disposition": panel,
        "factor_disposition": factor,
        "revision_disposition": revision,
        "blockers": tuple(blockers),
        "required_authorities": tuple(authorities),
    }
    identity = PreResearchObservedDelta.model_construct(**values, delta_hash="").model_dump(
        mode="json", exclude={"delta_hash"}
    )
    for name in (
        "base_adjusted_return_revision_cursor",
        "base_adjusted_return_revision_chain_hash",
        "observed_adjusted_return_revision_cursor",
        "observed_adjusted_return_revision_chain_hash",
    ):
        if identity.get(name) is None:
            identity.pop(name, None)
    return PreResearchObservedDelta(**values, delta_hash=canonical_hash(identity))


def compile_update_plan(
    *, delta: PreResearchObservedDelta, policy_hashes: tuple[str, ...]
) -> PreResearchUpdatePlan:
    """Compile deterministic I/O scopes.  No owner may broaden these scopes.

    Args:
        delta: Sealed observed membership and semantic-work classifications.
        policy_hashes: Nonempty installed policy identities governing those operations.

    Returns:
        Sealed disjoint I/O scopes with required rebuilds/publication and confirmation
        for initialization or membership revisions.

    Raises:
        ValueError: Policies are absent or compiled scopes/identity are invalid.
    """
    if not policy_hashes:
        raise ValueError("update plan requires code-owned policy identities")
    by_kind: dict[ListingDataDeltaKind, list[str]] = {}
    for value in delta.listing_deltas:
        by_kind.setdefault(value.kind, []).append(value.listing_id)
    full_additions = tuple(by_kind.get(ListingDataDeltaKind.FULL_HISTORY_ADDITION_REQUIRED, ()))
    bounded = tuple(
        sorted(
            set(by_kind.get(ListingDataDeltaKind.TAIL_APPEND_REQUIRED, ()))
            | set(by_kind.get(ListingDataDeltaKind.RAW_SEMANTIC_CORRECTION, ()))
            | set(by_kind.get(ListingDataDeltaKind.ADJUSTED_RETURN_CORRECTION, ()))
        )
    )
    evidence = tuple(by_kind.get(ListingDataDeltaKind.EVIDENCE_ONLY, ()))
    operations = tuple(delta.required_authorities)
    requires_confirmation = delta.revision_disposition in {
        PreResearchRevisionDisposition.INITIALIZATION_REQUIRED,
        PreResearchRevisionDisposition.MEMBERSHIP_REVISION,
    }
    values: dict[str, object] = {
        "kind": "PreResearchUpdatePlan",
        "base_head_hash": delta.base_head_hash,
        "candidate_snapshot_hash": delta.candidate_snapshot_hash,
        "delta_hash": delta.delta_hash,
        "full_history_addition_listing_ids": full_additions,
        "bounded_retained_listing_ids": bounded,
        "removed_listing_ids": tuple(
            sorted(set(delta.removals) & set(delta.prior_active_listing_ids))
        ),
        "evidence_only_listing_ids": evidence,
        "require_full_panel_rebuild": (
            delta.panel_disposition is PanelDeltaDisposition.FULL_HISTORY_REBUILD
        ),
        "require_fixed_factor_screening": (
            delta.factor_disposition is FactorDeltaDisposition.FIXED_SCREENING_REQUIRED
        ),
        "require_foundation_publication": delta.revision_disposition
        in {
            PreResearchRevisionDisposition.INITIALIZATION_REQUIRED,
            PreResearchRevisionDisposition.MEMBERSHIP_REVISION,
            PreResearchRevisionDisposition.BOUNDED_MAINTENANCE,
        },
        "user_confirmation_required": requires_confirmation,
        "operations": operations,
        "policy_hashes": policy_hashes,
    }
    return seal_contract(PreResearchUpdatePlan, values, "update_plan_hash")


def stage_qualified_delta(
    *,
    delta: PreResearchObservedDelta,
    ordered_qualified_listing_ids: tuple[str, ...],
    quarantines: tuple[PreResearchQuarantine, ...],
    staged_child_hashes: tuple[str, ...] = (),
    adjusted_return_revision_cursor: int | None = None,
    adjusted_return_revision_chain_hash: str | None = None,
) -> PreResearchStagedDelta:
    """Seal the qualified active axis and quarantines against an observed transition.

    Args:
        delta: Frozen observed delta carrying prior active membership.
        ordered_qualified_listing_ids: Unique ordered axis admitted after qualification.
        quarantines: Exclusion evidence for candidates not on the qualified axis.
        staged_child_hashes: Durable child identities produced under the transition's scopes.
        adjusted_return_revision_cursor: Optional measured adjusted-return revision position.
        adjusted_return_revision_chain_hash: Paired chain commitment at that position.

    Returns:
        Validated staging record. Source membership changes which leave the active
        set unchanged use the explicit source-only revision disposition.

    Raises:
        ValueError: Qualified axes, quarantine exclusion, revision binding or identity is invalid.
    """
    prior = set(delta.prior_active_listing_ids)
    active_changed = set(ordered_qualified_listing_ids) != prior
    disposition = delta.revision_disposition
    if delta.source_membership_changed and not active_changed:
        disposition = PreResearchRevisionDisposition.SOURCE_CHANGED_ACTIVE_SET_UNCHANGED
    values: dict[str, object] = {
        "kind": "PreResearchStagedDelta",
        "observed_delta_hash": delta.delta_hash,
        "ordered_qualified_listing_ids": ordered_qualified_listing_ids,
        "qualified_listing_set_hash": canonical_hash(ordered_qualified_listing_ids),
        "quarantines": quarantines,
        "active_set_changed": active_changed,
        "revision_disposition": disposition,
        "adjusted_return_revision_cursor": adjusted_return_revision_cursor,
        "adjusted_return_revision_chain_hash": adjusted_return_revision_chain_hash,
        "staged_child_hashes": staged_child_hashes,
    }
    identity = PreResearchStagedDelta.model_construct(**values, staged_delta_hash="").model_dump(
        mode="json", exclude={"staged_delta_hash"}
    )
    if adjusted_return_revision_cursor is None:
        identity.pop("adjusted_return_revision_cursor", None)
        identity.pop("adjusted_return_revision_chain_hash", None)
    return PreResearchStagedDelta(**values, staged_delta_hash=canonical_hash(identity))


class PreResearchHeadStore:
    """Content-addressed HEAD/marker store with one atomic CAS pointer."""

    _NON_IDENTITY_FIELDS = frozenset({"verified_at", "observed_at", "confirmed_at", "published_at"})

    def __init__(self, *, artifact_root: Path, mutation_gate: WorkspaceMutationGate) -> None:
        """Bind the content-addressed transition store and its final activation gate.

        Args:
            artifact_root: Workspace artifact root containing pre-research-revisions.
            mutation_gate: Gate used for compare-and-swap HEAD activation and recovery.
        """
        self.root = artifact_root / "pre-research-revisions"
        self.mutation_gate = mutation_gate

    def publish_candidate(self, candidate: ObservedPreResearchCandidateSnapshot) -> str:
        """Publish and verify frozen source observations under its content identity.

        Args:
            candidate: Frozen source observations to publish immutably.

        Returns:
            Playpen reference to the verified durable record; equal content is reused.

        Raises:
            ValueError: Identity, existing content, or authoritative readback is inconsistent.
        """
        return self._publish("candidates", candidate.candidate_snapshot_hash, candidate)

    def publish_delta(self, delta: PreResearchObservedDelta) -> str:
        """Publish and verify sealed observed semantic/membership delta under its content identity.

        Args:
            delta: Sealed observed semantic/membership delta to publish immutably.

        Returns:
            Playpen reference to the verified durable record; equal content is reused.

        Raises:
            ValueError: Identity, existing content, or authoritative readback is inconsistent.
        """
        return self._publish("deltas", delta.delta_hash, delta)

    def publish_plan(self, plan: PreResearchUpdatePlan) -> str:
        """Publish and verify the compiled owner scopes and required authorities.

        Args:
            plan: Compiled owner scopes and required authorities to publish immutably.

        Returns:
            Playpen reference to the verified durable record; equal content is reused.

        Raises:
            ValueError: Identity, existing content, or authoritative readback is inconsistent.
        """
        return self._publish("plans", plan.update_plan_hash, plan)

    def publish_staged_delta(self, staged: PreResearchStagedDelta) -> str:
        """Publish and verify qualified active membership and staged children.

        Args:
            staged: Qualified active membership and staged children to publish immutably.

        Returns:
            Playpen reference to the verified durable record; equal content is reused.

        Raises:
            ValueError: Identity, existing content, or authoritative readback is inconsistent.
        """
        return self._publish("staging", staged.staged_delta_hash, staged)

    def publish_mandate(self, mandate: ConfirmedPreResearchUpdateMandate) -> str:
        """Publish and verify confirmation bound to the frozen tree and plan.

        Args:
            mandate: Confirmation bound to the frozen tree and plan to publish immutably.

        Returns:
            Playpen reference to the verified durable record; equal content is reused.

        Raises:
            ValueError: Identity, existing content, or authoritative readback is inconsistent.
        """
        return self._publish("mandates", mandate.mandate_hash, mandate)

    def set_pending_transition(
        self,
        *,
        candidate: ObservedPreResearchCandidateSnapshot,
        delta: PreResearchObservedDelta,
        plan: PreResearchUpdatePlan,
        frozen_at: datetime,
        mandate: ConfirmedPreResearchUpdateMandate | None = None,
    ) -> PreResearchPendingTransition:
        """Freeze one control-plane index without replacing a confirmed tree."""
        existing = self.pending_transition()
        if existing is not None and existing.mandate_hash is not None:
            if existing.base_head_hash == plan.base_head_hash:
                return existing
            raise PreResearchCasError("PENDING_TRANSITION_BASE_HEAD_STALE")
        values: dict[str, object] = {
            "kind": "PreResearchPendingTransition",
            "base_head_hash": plan.base_head_hash,
            "candidate_snapshot_hash": candidate.candidate_snapshot_hash,
            "delta_hash": delta.delta_hash,
            "update_plan_hash": plan.update_plan_hash,
            "mandate_hash": mandate.mandate_hash if mandate is not None else None,
            "frozen_at": frozen_at,
        }
        identity = PreResearchPendingTransition.model_construct(
            **values, pending_hash=""
        ).model_dump(mode="json", exclude={"pending_hash", "frozen_at"})
        pending = PreResearchPendingTransition(**values, pending_hash=canonical_hash(identity))
        self._publish("pending", pending.pending_hash, pending)
        self._replace_named_pointer(
            "pending-transition.json", {"pending_hash": pending.pending_hash}
        )
        return pending

    def pending_transition(self) -> PreResearchPendingTransition | None:
        """Reopen the pending pointer and validate every frozen transition child.

        Returns:
            Validated pending transition, or None when no pointer exists.

        Raises:
            ValueError: Pointer shape, pending identity, or candidate/delta/plan/mandate lineage
                differs.
            FileNotFoundError: A child named by the durable pointer is absent.
        """
        pointer = self.root / "pending-transition.json"
        if not pointer.exists():
            return None
        payload = self._read_json(pointer)
        if set(payload) != {"pending_hash"}:
            raise ValueError("PRE_RESEARCH_PENDING_POINTER_TAMPERED")
        pending_hash = str(payload["pending_hash"])
        pending = PreResearchPendingTransition.model_validate(
            self._read_json(self._path("pending", pending_hash))
        )
        if pending.pending_hash != pending_hash:
            raise ValueError("PRE_RESEARCH_PENDING_POINTER_CHILD_MISMATCH")
        candidate = self.load_candidate(pending.candidate_snapshot_hash)
        delta = self.load_delta(pending.delta_hash)
        plan = self.load_plan(pending.update_plan_hash)
        mandate = self.load_mandate(pending.mandate_hash) if pending.mandate_hash else None
        if (
            delta.candidate_snapshot_hash != candidate.candidate_snapshot_hash
            or plan.delta_hash != delta.delta_hash
            or plan.candidate_snapshot_hash != candidate.candidate_snapshot_hash
            or pending.base_head_hash != plan.base_head_hash
            or (mandate is not None and mandate.update_plan_hash != plan.update_plan_hash)
        ):
            raise ValueError("PRE_RESEARCH_PENDING_TRANSITION_CHILD_MISMATCH")
        return pending

    def clear_pending_transition(self, *, expected_pending_hash: str) -> None:
        """Remove only the pending pointer naming the expected frozen transition.

        Args:
            expected_pending_hash: Pending identity which the caller expects to clear.

        Raises:
            PreResearchCasError: A present pointer names a different pending transition.
            ValueError: The pending pointer or its durable lineage fails validation.
        """
        pointer = self.root / "pending-transition.json"
        if not pointer.exists():
            return
        pending = self.pending_transition()
        if pending is None or pending.pending_hash != expected_pending_hash:
            raise PreResearchCasError("PENDING_TRANSITION_COMPARE_AND_SWAP_FAILED")
        pointer.unlink()

    def clear_unconfirmed_pending_transition(self, *, base_head_hash: str | None) -> bool:
        """Supersede a stale proposal without disturbing a confirmed frozen run."""
        pending = self.pending_transition()
        if (
            pending is None
            or pending.mandate_hash is not None
            or pending.base_head_hash != base_head_hash
        ):
            return False
        self.clear_pending_transition(expected_pending_hash=pending.pending_hash)
        return True

    def load_candidate(self, content_hash: str) -> ObservedPreResearchCandidateSnapshot:
        """Reopen and validate the requested candidate source observations.

        Args:
            content_hash: Content-addressed record handle to read.

        Returns:
            The immutable candidate source observations with its internal contract validated.

        Raises:
            ValueError: The requested handle or loaded contract is invalid.
            FileNotFoundError: The requested record is absent.
        """
        return ObservedPreResearchCandidateSnapshot.model_validate(
            self._read_json(self._path("candidates", content_hash))
        )

    def load_delta(self, content_hash: str) -> PreResearchObservedDelta:
        """Reopen and validate the requested observed delta.

        Args:
            content_hash: Content-addressed record handle to read.

        Returns:
            The immutable observed delta with its internal contract validated.

        Raises:
            ValueError: The requested handle or loaded contract is invalid.
            FileNotFoundError: The requested record is absent.
        """
        return PreResearchObservedDelta.model_validate(
            self._read_json(self._path("deltas", content_hash))
        )

    def load_plan(self, content_hash: str) -> PreResearchUpdatePlan:
        """Reopen and validate the requested compiled update plan.

        Args:
            content_hash: Content-addressed record handle to read.

        Returns:
            The immutable compiled update plan with its internal contract validated.

        Raises:
            ValueError: The requested handle or loaded contract is invalid.
            FileNotFoundError: The requested record is absent.
        """
        return PreResearchUpdatePlan.model_validate(
            self._read_json(self._path("plans", content_hash))
        )

    def load_staged_delta(self, content_hash: str) -> PreResearchStagedDelta:
        """Reopen and validate the requested qualified staging record.

        Args:
            content_hash: Content-addressed record handle to read.

        Returns:
            The immutable qualified staging record with its internal contract validated.

        Raises:
            ValueError: The requested handle or loaded contract is invalid.
            FileNotFoundError: The requested record is absent.
        """
        return PreResearchStagedDelta.model_validate(
            self._read_json(self._path("staging", content_hash))
        )

    def load_mandate(self, content_hash: str) -> ConfirmedPreResearchUpdateMandate:
        """Reopen and validate the requested confirmed update mandate.

        Args:
            content_hash: Content-addressed record handle to read.

        Returns:
            The immutable confirmed update mandate with its internal contract validated.

        Raises:
            ValueError: The requested handle or loaded contract is invalid.
            FileNotFoundError: The requested record is absent.
        """
        return ConfirmedPreResearchUpdateMandate.model_validate(
            self._read_json(self._path("mandates", content_hash))
        )

    def current_head(self) -> VerifiedPreResearchHead | None:
        """Reopen current HEAD and verify its pointer, marker and complete transition lineage.

        Returns:
            Verified research HEAD, or None before any HEAD pointer is published.

        Raises:
            ValueError: Pointer shape, marker/HEAD identities or transition children disagree.
            FileNotFoundError: The pointer names an absent durable child.
        """
        pointer = self.root / "current-head.json"
        if not pointer.exists():
            return None
        payload = self._read_json(pointer)
        marker_hash = str(payload.get("marker_hash", ""))
        marker = PreResearchRevisionMarker.model_validate(
            self._read_json(self._path("markers", marker_hash))
        )
        head = VerifiedPreResearchHead.model_validate(
            self._read_json(self._path("heads", marker.next_head_hash))
        )
        candidate = ObservedPreResearchCandidateSnapshot.model_validate(
            self._read_json(self._path("candidates", marker.candidate_snapshot_hash))
        )
        delta = PreResearchObservedDelta.model_validate(
            self._read_json(self._path("deltas", marker.delta_hash))
        )
        staged = PreResearchStagedDelta.model_validate(
            self._read_json(self._path("staging", marker.staged_delta_hash))
        )
        plan = PreResearchUpdatePlan.model_validate(
            self._read_json(self._path("plans", marker.update_plan_hash))
        )
        mandate = (
            ConfirmedPreResearchUpdateMandate.model_validate(
                self._read_json(self._path("mandates", marker.mandate_hash))
            )
            if marker.mandate_hash is not None
            else None
        )
        if head.head_hash != marker.next_head_hash:
            raise ValueError("PRE_RESEARCH_HEAD_MARKER_CHILD_MISMATCH")
        if (
            candidate.candidate_snapshot_hash != marker.candidate_snapshot_hash
            or delta.delta_hash != marker.delta_hash
            or staged.staged_delta_hash != marker.staged_delta_hash
            or plan.update_plan_hash != marker.update_plan_hash
            or delta.candidate_snapshot_hash != candidate.candidate_snapshot_hash
            or plan.delta_hash != delta.delta_hash
            or staged.observed_delta_hash != delta.delta_hash
            or (mandate is not None and mandate.update_plan_hash != plan.update_plan_hash)
        ):
            raise ValueError("PRE_RESEARCH_HEAD_TRANSITION_CHILD_MISMATCH")
        if payload != {"head_hash": head.head_hash, "marker_hash": marker.marker_hash}:
            raise ValueError("PRE_RESEARCH_HEAD_POINTER_TAMPERED")
        return head

    def current_revision_marker(self) -> PreResearchRevisionMarker | None:
        """Read the marker behind the current CAS pointer for recovery decisions."""
        pointer = self.root / "current-head.json"
        if not pointer.exists():
            return None
        payload = self._read_json(pointer)
        marker = PreResearchRevisionMarker.model_validate(
            self._read_json(self._path("markers", str(payload.get("marker_hash", ""))))
        )
        if payload.get("head_hash") != marker.next_head_hash:
            raise ValueError("PRE_RESEARCH_HEAD_POINTER_MARKER_MISMATCH")
        return marker

    def restore_verified_head_pointer(
        self,
        *,
        expected_current_head_hash: str,
        replacement_head_hash: str,
        replacement_marker_hash: str,
    ) -> VerifiedPreResearchHead:
        """Reverse one unaccepted transition without overwriting a newer HEAD."""
        with self.mutation_gate.try_hold(timeout_seconds=30.0):
            current = self.current_head()
            if current is None or current.head_hash != expected_current_head_hash:
                raise PreResearchCasError("HEAD_COMPARE_AND_SWAP_FAILED")
            replacement_marker = PreResearchRevisionMarker.model_validate(
                self._read_json(self._path("markers", replacement_marker_hash))
            )
            replacement_head = VerifiedPreResearchHead.model_validate(
                self._read_json(self._path("heads", replacement_head_hash))
            )
            if (
                replacement_marker.next_head_hash != replacement_head.head_hash
                or replacement_head.head_hash != replacement_head_hash
            ):
                raise ValueError("PRE_RESEARCH_HEAD_ROLLBACK_CHILD_MISMATCH")
            self._replace_pointer(
                {
                    "head_hash": replacement_head.head_hash,
                    "marker_hash": replacement_marker.marker_hash,
                }
            )
            if self.current_head() != replacement_head:
                raise ValueError("PRE_RESEARCH_HEAD_ROLLBACK_READBACK_FAILED")
            return replacement_head

    def advance_head(
        self,
        *,
        expected_base_head_hash: str | None,
        next_head: VerifiedPreResearchHead,
        candidate_snapshot_hash: str,
        delta_hash: str,
        staged_delta_hash: str,
        update_plan_hash: str,
        mandate_hash: str | None,
        child_hashes: tuple[str, ...],
        storage_evidence: PreResearchStorageEvidence,
        published_at: datetime,
        before_pointer_commit: Callable[[PreResearchRevisionMarker], None] | None = None,
    ) -> PreResearchRevisionMarker:
        """Publish immutable children, then atomically advance only the expected HEAD."""
        _require_aware(published_at, "HEAD publication")
        with self.mutation_gate.try_hold(timeout_seconds=30.0):
            current = self.current_head()
            actual = current.head_hash if current else None
            if actual != expected_base_head_hash:
                raise PreResearchCasError("HEAD_COMPARE_AND_SWAP_FAILED")
            self._publish("heads", next_head.head_hash, next_head)
            marker_values: dict[str, object] = {
                "kind": "PreResearchRevisionMarker",
                "base_head_hash": expected_base_head_hash,
                "next_head_hash": next_head.head_hash,
                "candidate_snapshot_hash": candidate_snapshot_hash,
                "delta_hash": delta_hash,
                "staged_delta_hash": staged_delta_hash,
                "update_plan_hash": update_plan_hash,
                "mandate_hash": mandate_hash,
                "child_hashes": child_hashes,
                "storage_evidence": storage_evidence,
                "published_at": published_at,
            }
            identity = PreResearchRevisionMarker.model_construct(
                **marker_values, marker_hash=""
            ).model_dump(mode="json", exclude={"marker_hash", "published_at"})
            marker = PreResearchRevisionMarker(
                **marker_values, marker_hash=canonical_hash(identity)
            )
            self._publish("markers", marker.marker_hash, marker)
            durable_marker = PreResearchRevisionMarker.model_validate(
                self._read_json(self._path("markers", marker.marker_hash))
            )
            durable_head = VerifiedPreResearchHead.model_validate(
                self._read_json(self._path("heads", next_head.head_hash))
            )
            if durable_marker != marker or durable_head != next_head:
                raise ValueError("PRE_RESEARCH_HEAD_AUTHORITATIVE_READBACK_FAILED")
            if before_pointer_commit is not None:
                before_pointer_commit(marker)
            self._replace_pointer(
                {"head_hash": next_head.head_hash, "marker_hash": marker.marker_hash}
            )
            if self.current_head() != next_head:
                raise ValueError("PRE_RESEARCH_HEAD_ACTIVATION_READBACK_FAILED")
            return marker

    def _path(self, kind: str, content_hash: str) -> Path:
        if len(content_hash) != 64 or any(char not in "0123456789abcdef" for char in content_hash):
            raise ValueError("artifact content hash is invalid")
        return self.root / kind / f"{content_hash}.json"

    def _publish(self, kind: str, content_hash: str, value: _Contract) -> str:
        target = self._path(kind, content_hash)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = value.model_dump(mode="json")
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        staged = target.with_name(f".{content_hash}.{os.getpid()}.tmp")
        if target.exists():
            durable = type(value).model_validate(self._read_json(target))
            if durable.model_dump(
                mode="json", exclude=self._NON_IDENTITY_FIELDS
            ) != value.model_dump(mode="json", exclude=self._NON_IDENTITY_FIELDS):
                raise ValueError("content-addressed artifact identity collision")
        else:
            staged.write_bytes(serialized)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)
        durable = type(value).model_validate(self._read_json(target))
        if durable.model_dump(mode="json", exclude=self._NON_IDENTITY_FIELDS) != value.model_dump(
            mode="json", exclude=self._NON_IDENTITY_FIELDS
        ):
            raise ValueError("content-addressed artifact readback differs")
        return f"playpen://pre-research-revisions/{kind}/{content_hash}"

    def _replace_pointer(self, payload: dict[str, str]) -> None:
        self._replace_named_pointer("current-head.json", payload)

    def _replace_named_pointer(self, name: str, payload: dict[str, str]) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{name}.{os.getpid()}.tmp")
        staged.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(staged, target)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("artifact payload must be an object")
        return payload


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} clock must be timezone-aware")


__all__ = [
    "ConfirmedPreResearchUpdateMandate",
    "FactorDeltaDisposition",
    "ListingDataDeltaKind",
    "ObservedPreResearchCandidateSnapshot",
    "PanelDeltaDisposition",
    "PreResearchCasError",
    "PreResearchHeadStore",
    "PreResearchListingDelta",
    "PreResearchListingObservation",
    "PreResearchObservedDelta",
    "PreResearchPendingTransition",
    "PreResearchQuarantine",
    "PreResearchRevisionDisposition",
    "PreResearchRevisionMarker",
    "PreResearchStagedDelta",
    "PreResearchStorageEvidence",
    "PreResearchUpdatePlan",
    "QuarantineReason",
    "VerifiedPreResearchHead",
    "build_candidate_snapshot",
    "build_verified_head",
    "compile_update_plan",
    "compute_observed_delta",
    "confirm_update_plan",
    "seal_contract",
    "stage_qualified_delta",
]
