"""Path-free contracts for the playpen Feature Foundation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import ClassVar

from alphalattice.foundation.market_data_ops.sources.membership import membership_identity
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import SECTOR_HISTORY_BACKFILLED

PANEL_ROW_IDENTITY_BY_BINDING = "PANEL_BINDING"
"""The row-identity rule of Panels built before temporal membership.

Every row's hash bound the whole build's manifest and sector revisions, so a
listing joining at a later session changed the identity of every row of every session.
Recorded artifacts under this rule stay readable and recoverable as written.
"""

PANEL_ROW_IDENTITY_BY_CROSS_SECTION = "SESSION_CROSS_SECTION"
"""The row-identity rule of Panels with per-session membership.

A row's hash binds the identity of its own session's cross-section -- the
members that session actually holds, in calculation order, each with its
sector -- the catalog, the policy, the session, the listing and its values.
A membership change at one session moves the identity of that session
onward and of nothing before it.
"""


@dataclass(frozen=True)
class TemporalKnowledgeBoundary:
    """Bitemporal identity for a governed research input.

    ``market_as_of_session`` is the last market session represented by the
    input. ``knowledge_cutoff_at`` is the UTC instant after which newly learned
    provider facts must not leak into the input. ``materialized_at`` is an
    operational timestamp only and is deliberately excluded from
    :meth:`identity_hash`.
    """

    market_as_of_session: date
    knowledge_cutoff_at: datetime
    materialized_at: datetime
    universe_source_observed_at: datetime
    sector_source_observed_at: datetime
    universe_point_in_time_qualified: bool = False
    sector_point_in_time_qualified: bool = False

    def __post_init__(self) -> None:
        """Refuse timestamps outside the declared knowledge boundary.

        Raises:
            ValueError: A timestamp is naive, materialization precedes the cutoff,
                or a source observation falls after the cutoff.
        """
        for name in (
            "knowledge_cutoff_at",
            "materialized_at",
            "universe_source_observed_at",
            "sector_source_observed_at",
        ):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.knowledge_cutoff_at > self.materialized_at:
            raise ValueError("knowledge cutoff cannot be after materialization")
        if self.universe_source_observed_at > self.knowledge_cutoff_at:
            raise ValueError("universe observation is after the knowledge cutoff")
        if self.sector_source_observed_at > self.knowledge_cutoff_at:
            raise ValueError("sector observation is after the knowledge cutoff")

    def identity_payload(self) -> dict[str, object]:
        """Return the temporal inputs that define this research input.

        Returns:
            ISO dates and timestamps with the two point-in-time qualification flags;
            the operational materialization timestamp is excluded.
        """
        return {
            "market_as_of_session": self.market_as_of_session.isoformat(),
            "knowledge_cutoff_at": self.knowledge_cutoff_at.isoformat(),
            "universe_source_observed_at": self.universe_source_observed_at.isoformat(),
            "sector_source_observed_at": self.sector_source_observed_at.isoformat(),
            "universe_point_in_time_qualified": self.universe_point_in_time_qualified,
            "sector_point_in_time_qualified": self.sector_point_in_time_qualified,
        }

    def identity_hash(self) -> str:
        """Hash the temporal inputs, excluding the materialization timestamp.

        Returns:
            Canonical identity of the knowledge boundary and its qualification flags.
        """
        return canonical_hash(self.identity_payload())

    def audit_payload(self) -> dict[str, object]:
        """Include the operational materialization timestamp in an audit projection.

        Returns:
            Identity inputs, their hash, and materialization time as an ISO timestamp.
        """
        return {
            **self.identity_payload(),
            "materialized_at": self.materialized_at.isoformat(),
            "temporal_identity_hash": self.identity_hash(),
        }


@dataclass(frozen=True)
class FeatureCatalogBinding:
    """Independent desktop catalog, clock, and materialization semantics.

    The two clock authorities are bound separately and both are part of the
    binding. Every downstream identity quotes ``catalog_hash``, so a Panel built
    under a different Formula observation clock cannot present itself as the same
    catalog's output -- which is precisely what a Panel whose Features were one
    session stale used to be able to do.

    They are separate fields because they answer different questions and change
    for different reasons. ``formula_observation_policy_hash`` covers which
    session each Formula's value belongs to and what it consumed;
    ``source_availability_policy_hash`` covers when such a value may be read. A
    Provider that changes its publication schedule moves the second and must
    leave the first -- and every per-Formula method identity under it -- exactly
    where it was.

    ``source_authority_binding_hash`` is the third, and it splits the second
    question in half. *Which* owners a Formula reads is a property of the Formula
    and moves only when its ``required_fields`` do; *when* those owners publish is
    a property of the feed. Holding them in one hash meant a Provider schedule
    change and a Formula gaining a Sector dependency were indistinguishable, and
    a Panel could not say which had happened.
    """

    stable_id: str
    catalog_content_hash: str
    formula_implementation_hash: str
    materializer_policy_hash: str
    formula_observation_policy_hash: str
    source_availability_policy_hash: str
    source_authority_binding_hash: str
    catalog_hash: str

    @classmethod
    def create(
        cls,
        *,
        stable_id: str,
        catalog_content_hash: str,
        formula_implementation_hash: str,
        materializer_policy_hash: str,
        formula_observation_policy_hash: str,
        source_availability_policy_hash: str,
        source_authority_binding_hash: str,
    ) -> FeatureCatalogBinding:
        """Bind the catalog content to its separate implementation and clock authorities.

        Args:
            stable_id: Logical name of the installed catalog.
            catalog_content_hash: Identity of its declared recipe content.
            formula_implementation_hash: Identity of the installed formula implementation.
            materializer_policy_hash: Identity of feature materialization semantics.
            formula_observation_policy_hash: Authority for each formula's observation clock.
            source_availability_policy_hash: Authority for when source values may be read.
            source_authority_binding_hash: Authority for which source owners formulas read.

        Returns:
            Binding with a canonical hash of all seven supplied authorities.
        """
        payload = {
            "stable_id": stable_id,
            "catalog_content_hash": catalog_content_hash,
            "formula_implementation_hash": formula_implementation_hash,
            "materializer_policy_hash": materializer_policy_hash,
            "formula_observation_policy_hash": formula_observation_policy_hash,
            "source_availability_policy_hash": source_availability_policy_hash,
            "source_authority_binding_hash": source_authority_binding_hash,
        }
        return cls(catalog_hash=canonical_hash(payload), **payload)


@dataclass(frozen=True)
class FeatureInvalidation:
    """One domain-owned reason to recompute a bounded feature scope."""

    kind: str
    listing_id: str | None = None
    earliest_session: date | None = None
    affected_sessions: tuple[date, ...] = ()
    source_fields: tuple[str, ...] = ()
    factor_ids: tuple[str, ...] = ()
    source_receipt_hash: str | None = None

    _ALLOWED: ClassVar[frozenset[str]] = frozenset(
        {
            "normal_new_session",
            "raw_correction",
            "action_correction",
            "action_evidence_change",
            "adjusted_return_correction",
            "ohlc_correction",
            "volume_correction",
            "spy_correction",
            "sector_revision_change",
            "manifest_addition",
            "manifest_removal",
            "catalog_binding_change",
            "panel_binding_change",
            "initial_backfill",
        }
    )

    def __post_init__(self) -> None:
        """Qualify the invalidation reason, scope, and ordered source identities.

        Raises:
            ValueError: The reason is unknown, a listing-scoped reason lacks its
                listing, the receipt length is invalid, or an axis is not sorted and unique.
        """
        if self.kind not in self._ALLOWED:
            raise ValueError("unknown feature invalidation kind")
        if (
            self.kind in {"raw_correction", "action_correction", "manifest_addition"}
            and not self.listing_id
        ):
            raise ValueError("listing-scoped invalidation requires a listing_id")
        if self.source_receipt_hash is not None and len(self.source_receipt_hash) != 64:
            raise ValueError("feature invalidation receipt requires a SHA-256 identity")
        if self.affected_sessions != tuple(sorted(set(self.affected_sessions))):
            raise ValueError("feature invalidation sessions must be sorted and unique")
        if self.source_fields != tuple(sorted(set(self.source_fields))):
            raise ValueError("feature invalidation fields must be sorted and unique")
        if self.factor_ids != tuple(sorted(set(self.factor_ids))):
            raise ValueError("feature invalidation factors must be sorted and unique")


@dataclass(frozen=True)
class FeatureBuildRequest:
    """Stable deterministic input for a feature materialization task."""

    manifest_revision: str
    catalog: FeatureCatalogBinding
    spy_revision: str
    history_start: date
    as_of_session: date
    invalidations: tuple[FeatureInvalidation, ...]
    request_hash: str

    @classmethod
    def create(
        cls,
        *,
        manifest_revision: str,
        catalog: FeatureCatalogBinding,
        spy_revision: str,
        history_start: date,
        as_of_session: date,
        invalidations: tuple[FeatureInvalidation, ...] = (),
    ) -> FeatureBuildRequest:
        """Bind one history interval and its declared invalidations for materialization.

        Args:
            manifest_revision: Data manifest authority for the build.
            catalog: Qualified recipe, implementation, and clock binding.
            spy_revision: Revision of the market-reference inputs.
            history_start: First session of the requested history.
            as_of_session: Last session represented by the build.
            invalidations: Domain-owned reasons to recompute affected feature scope.

        Returns:
            Request with a canonical hash of the revisions, interval, and invalidations.

        Raises:
            ValueError: History starts after the as-of session.
        """
        if history_start > as_of_session:
            raise ValueError("feature build history start is after as-of session")
        hash_payload = {
            "manifest_revision": manifest_revision,
            "catalog_hash": catalog.catalog_hash,
            "spy_revision": spy_revision,
            "history_start": history_start.isoformat(),
            "as_of_session": as_of_session.isoformat(),
            "invalidations": [asdict(item) for item in invalidations],
        }
        return cls(
            manifest_revision=manifest_revision,
            catalog=catalog,
            spy_revision=spy_revision,
            history_start=history_start,
            as_of_session=as_of_session,
            invalidations=invalidations,
            request_hash=canonical_hash(hash_payload),
        )


class FeatureBuildStatus(StrEnum):
    """Describe whether a feature build is running, deferred, blocked, or completed."""

    RUNNING = "running"
    DEFERRED = "deferred"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class FeatureBuildStage(StrEnum):
    """Which durable stage a build has actually reached.

    A blocked build used to say only that it was blocked, so "the base closure is
    still incomplete" and "every listing is materialised and only composition
    failed" were the same answer. They call for opposite recoveries: the first
    has listings left to compute, the second must reuse the closure it already
    has. One remediation discarded 452 completed listings because the difference
    was not expressible.

    Every member below is one recovery, and no two of them are the same work.
    Corruption in particular is separated from a pending transition: reconciling
    a transition against damaged state is not a repair, and reporting damage as
    "recoverable" is how a workspace gets reset instead of investigated.
    """

    BASE_CLOSURE_INCOMPLETE = "base_closure_incomplete"
    BASE_CLOSURE_COMPLETE_PANEL_PENDING = "base_closure_complete_panel_pending"
    TRANSITION_RECOVERY_PENDING = "transition_recovery_pending"
    CLOSURE_AUTHORITY_UNAVAILABLE = "closure_authority_unavailable"
    PANEL_PUBLISHED_AWAITING_GATEWAY_ADMISSION = "panel_published_awaiting_gateway_admission"
    PANEL_SEMANTIC_INDEX_PENDING = "panel_semantic_index_pending"
    TERMINAL_PANEL_COMPLETE = "terminal_panel_complete"


@dataclass(frozen=True)
class FeatureBuildOutcome:
    """Safe build projection; frames, paths, SQL, and private messages stay local."""

    status: FeatureBuildStatus
    request_hash: str
    receipt_hash: str | None
    coverage_summary: dict[str, object]
    artifact_refs: tuple[str, ...] = ()
    retry_after_at: datetime | None = None
    failure_code: str | None = None
    build_stage: FeatureBuildStage | None = None

    def safe_hash(self) -> str:
        """Hash the safe projection of the recorded build outcome.

        Returns:
            Canonical identity of all outcome fields, including recovery stage and retry time.
        """
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class FeaturePanelBinding:
    """Bind a Panel's source revisions to its catalog and computation policy.

    Attributes:
        manifest_revision: Data manifest used to build the Panel.
        sector_revision: Sector classification authority.
        catalog_hash: Qualified feature catalog binding.
        spy_revision: Market-reference revision.
        policy_hash: Panel computation policy identity.
        panel_binding_hash: Canonical lineage binding, separate from materialized content.
    """

    manifest_revision: str
    sector_revision: str
    catalog_hash: str
    spy_revision: str
    policy_hash: str
    panel_binding_hash: str

    @classmethod
    def create(
        cls,
        *,
        manifest_revision: str,
        sector_revision: str,
        catalog_hash: str,
        spy_revision: str,
        policy_hash: str,
    ) -> FeaturePanelBinding:
        """Bind the five lineage authorities of one Panel build.

        Args:
            manifest_revision: Data manifest authority.
            sector_revision: Sector classification authority.
            catalog_hash: Qualified feature catalog identity.
            spy_revision: Market-reference revision.
            policy_hash: Panel computation policy identity.

        Returns:
            Lineage binding with a canonical hash of the supplied authorities.
        """
        payload = {
            "manifest_revision": manifest_revision,
            "sector_revision": sector_revision,
            "catalog_hash": catalog_hash,
            "spy_revision": spy_revision,
            "policy_hash": policy_hash,
        }
        return cls(panel_binding_hash=canonical_hash(payload), **payload)

    @property
    def panel_hash(self) -> str:
        """Compatibility alias for pre-closure playpen callers.

        New code must use ``panel_binding_hash`` so a lineage identity is not
        mistaken for the hash of materialized panel content.
        """
        return self.panel_binding_hash


def panel_cross_section_identity(
    members: Sequence[str],
    sector_by_listing_id: Mapping[str, str],
    *,
    source_exclusions: Sequence[str] = (),
) -> str:
    """The computational identity of one session's cross-section.

    Ordered: the members in calculation order with their sectors, because
    the sector demeaning sums in that order and a reordering is a different
    floating-point computation. Nothing about how the membership was
    decided enters here; that is the Universe journal's.

    Args:
        members: Nominal listing axis in calculation order.
        sector_by_listing_id: Sector authority for every nominal member.
        source_exclusions: Evidenced exclusions within the nominal member axis.

    Returns:
        Canonical identity of ordered listing-sector pairs and unique exclusions.

    Raises:
        ValueError: A member lacks a sector, the axis is empty, or an exclusion is outside it.
    """
    try:
        pairs = [[listing_id, sector_by_listing_id[listing_id]] for listing_id in members]
    except KeyError as exc:
        raise ValueError(f"cross-section member {exc.args[0]} has no sector") from exc
    if not pairs:
        raise ValueError("cross-section identity requires at least one member")
    payload: dict[str, object] = {"kind": "PanelCrossSection", "members": pairs}
    if source_exclusions:
        excluded = sorted(set(source_exclusions))
        if not set(excluded).issubset(members):
            raise ValueError("source exclusion is outside the nominal cross-section")
        payload["source_exclusions"] = excluded
    return canonical_hash(payload)


@dataclass(frozen=True)
class PanelSourceExclusion:
    """Inclusive date bounds applied to trading sessions, never a Universe ENTRY/EXIT."""

    listing_id: str
    first_session: date
    last_session: date
    evidence_hash: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate a listing's evidenced source exclusion over an inclusive interval.

        Raises:
            ValueError: The listing or reason is empty, dates are reversed, or
                evidence is not a lowercase SHA-256 identity.
        """
        if (
            not self.listing_id
            or self.first_session > self.last_session
            or not self.reason_codes
            or len(self.evidence_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.evidence_hash)
            or any(not reason for reason in self.reason_codes)
        ):
            raise ValueError("invalid Panel source exclusion")

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PanelSourceExclusion:
        """Read an evidenced source exclusion from its serialized fields.

        Args:
            payload: Listing, inclusive ISO date bounds, evidence hash, and reason codes.

        Returns:
            Validated source exclusion, without changing Universe membership.

        Raises:
            KeyError: A required field is absent.
            ValueError: Dates, reasons, evidence identity, or the reason sequence are invalid.
        """
        return cls(
            listing_id=str(payload["listing_id"]),
            first_session=date.fromisoformat(str(payload["first_session"])),
            last_session=date.fromisoformat(str(payload["last_session"])),
            evidence_hash=str(payload["evidence_hash"]),
            reason_codes=tuple(str(item) for item in _sequence(payload["reason_codes"])),
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the exclusion with inclusive ISO date bounds.

        Returns:
            Listing, interval, evidence identity, and reason codes.
        """
        return {
            **asdict(self),
            "first_session": self.first_session.isoformat(),
            "last_session": self.last_session.isoformat(),
        }


@dataclass(frozen=True)
class PanelMembershipEpoch:
    """A contiguous range of sessions whose cross-section holds one member set."""

    first_session: date
    last_session: date
    listing_ids: tuple[str, ...]
    """The members, as a subsequence of the Panel's calculation axis."""

    def __post_init__(self) -> None:
        """Validate one nonempty member set over ordered session bounds.

        Raises:
            ValueError: Bounds are reversed or the member axis is empty or duplicated.
        """
        if self.first_session > self.last_session:
            raise ValueError("Panel membership epoch range is reversed")
        if not self.listing_ids or len(self.listing_ids) != len(set(self.listing_ids)):
            raise ValueError("Panel membership epoch members must be non-empty and unique")


@dataclass(frozen=True)
class PanelMembershipBasisRange:
    """Name the membership qualification basis over inclusive session bounds.

    Attributes:
        first_session: First session covered by this basis.
        last_session: Last session covered by this basis.
        basis: Universe qualification under which the members were resolved.
    """

    first_session: date
    last_session: date
    basis: str


@dataclass(frozen=True)
class PanelCrossSectionRange:
    """Sessions sharing one cross-section identity: the unit of partition reuse."""

    first_session: date
    last_session: date
    cross_section_identity: str
    member_count: int

    def to_payload(self) -> dict[str, object]:
        """Serialize a range of sessions sharing one cross-section identity.

        Returns:
            ISO date bounds, cross-section identity, and nominal member count.
        """
        return {
            "first_session": self.first_session.isoformat(),
            "last_session": self.last_session.isoformat(),
            "cross_section_identity": self.cross_section_identity,
            "member_count": self.member_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PanelCrossSectionRange:
        """Read the serialized identity and population of a session range.

        Args:
            payload: ISO date bounds, cross-section identity, and member count.

        Returns:
            Session range with parsed dates and member count.

        Raises:
            KeyError: A required field is absent.
            ValueError: A date or member count cannot be parsed.
        """
        return cls(
            first_session=date.fromisoformat(str(payload["first_session"])),
            last_session=date.fromisoformat(str(payload["last_session"])),
            cross_section_identity=str(payload["cross_section_identity"]),
            member_count=int(str(payload["member_count"])),
        )


@dataclass(frozen=True)
class PanelMembership:
    """Which listings each session's cross-section holds, over one calculation axis.

    ``listing_ids`` is the axis: every listing that is a member on some
    session, in calculation order. Each epoch's members are a subsequence of
    it. The basis ranges say which promise a session's membership is under
    (the disclosed initial-cohort backfill before T0, the as-observed forward
    membership from T0 on); the bootstrap reference and journal sequence name
    the Universe record this was resolved from, without copying it.
    """

    listing_ids: tuple[str, ...]
    epochs: tuple[PanelMembershipEpoch, ...]
    basis_ranges: tuple[PanelMembershipBasisRange, ...]
    journal_sequence: int = 0
    bootstrap_record_hash: str | None = None
    bootstrap_t0_session: date | None = None
    source_exclusions: tuple[PanelSourceExclusion, ...] = ()
    _epoch_by_session: dict[date, PanelMembershipEpoch] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Qualify member axes, epoch order, and declared membership bases.

        Raises:
            ValueError: The axis is empty or duplicated, epochs are absent or overlap,
                an epoch violates axis membership or order, no basis is declared,
                or a source exclusion names a listing outside the historical axis.
        """
        if not self.listing_ids or len(self.listing_ids) != len(set(self.listing_ids)):
            raise ValueError("Panel membership axis must be non-empty and unique")
        if not self.epochs:
            raise ValueError("Panel membership requires at least one epoch")
        positions = {listing_id: index for index, listing_id in enumerate(self.listing_ids)}
        previous: PanelMembershipEpoch | None = None
        for epoch in self.epochs:
            if previous is not None and epoch.first_session <= previous.last_session:
                raise ValueError("Panel membership epochs must be sorted and disjoint")
            try:
                order = [positions[listing_id] for listing_id in epoch.listing_ids]
            except KeyError as exc:
                raise ValueError(
                    f"Panel membership epoch member {exc.args[0]} is outside the axis"
                ) from exc
            if order != sorted(order):
                raise ValueError("Panel membership epoch members must follow the axis order")
            previous = epoch
        if not self.basis_ranges:
            raise ValueError("Panel membership requires a basis for every session")
        if any(item.listing_id not in positions for item in self.source_exclusions):
            raise ValueError("Panel source exclusion is outside the historical axis")

    @classmethod
    def uniform(
        cls,
        listing_ids: Sequence[str],
        *,
        first_session: date,
        last_session: date,
        basis: str,
    ) -> PanelMembership:
        """One member set on every session: a Panel before or without a journal.

        Args:
            listing_ids: Nonempty unique calculation axis held on every session.
            first_session: First session of the uniform epoch and basis.
            last_session: Last session of the uniform epoch and basis.
            basis: Qualification under which membership was resolved.

        Returns:
            One-epoch membership with one basis range and no journal reference.

        Raises:
            ValueError: Bounds are reversed or members are empty or duplicated.
        """
        axis = tuple(listing_ids)
        return cls(
            listing_ids=axis,
            epochs=(PanelMembershipEpoch(first_session, last_session, axis),),
            basis_ranges=(PanelMembershipBasisRange(first_session, last_session, basis),),
        )

    @property
    def is_uniform(self) -> bool:
        """Report whether one epoch holds the whole calculation axis."""
        return len(self.epochs) == 1 and self.epochs[0].listing_ids == self.listing_ids

    def epoch(self, session: date) -> PanelMembershipEpoch:
        """Resolve and cache the membership epoch containing a session.

        Args:
            session: Session whose inclusive epoch is requested.

        Returns:
            Declared epoch with its members in calculation order.

        Raises:
            KeyError: No membership epoch contains the session.
        """
        cached = self._epoch_by_session.get(session)
        if cached is not None:
            return cached
        for epoch in self.epochs:
            if epoch.first_session <= session <= epoch.last_session:
                self._epoch_by_session[session] = epoch
                return epoch
        raise KeyError(f"session {session.isoformat()} is outside the Panel membership")

    def members(self, session: date) -> tuple[str, ...]:
        """Read one session's nominal members in calculation order.

        Args:
            session: Session whose cross-section is requested.

        Returns:
            Epoch members before applying evidenced source exclusions.

        Raises:
            KeyError: The session is outside the declared epochs.
        """
        return self.epoch(session).listing_ids

    def members_by_session(self, sessions: Sequence[date]) -> dict[date, tuple[str, ...]]:
        """Project nominal members onto the requested session axis.

        Args:
            sessions: Sessions to resolve against the declared epochs.

        Returns:
            Member tuples keyed by session.

        Raises:
            KeyError: A requested session is outside the declared epochs.
        """
        return {session: self.members(session) for session in sessions}

    def excluded_sources(self, session: date) -> tuple[str, ...]:
        """Read nominal members with an evidenced source exclusion at a session.

        Args:
            session: Session to resolve against membership and inclusive exclusion bounds.

        Returns:
            Sorted unique excluded listing IDs drawn from that session's nominal members.

        Raises:
            KeyError: The session is outside the declared epochs.
        """
        nominal = set(self.members(session))
        return tuple(
            sorted(
                {
                    item.listing_id
                    for item in self.source_exclusions
                    if item.first_session <= session <= item.last_session
                    and item.listing_id in nominal
                }
            )
        )

    def basis(self, session: date) -> str:
        """Read the declared membership qualification for one session.

        Args:
            session: Session whose membership basis is requested.

        Returns:
            First declared basis whose inclusive bounds contain the session.

        Raises:
            KeyError: No basis range contains the session.
        """
        for item in self.basis_ranges:
            if item.first_session <= session <= item.last_session:
                return item.basis
        raise KeyError(f"session {session.isoformat()} has no membership basis")

    def row_count(self, sessions: Sequence[date]) -> int:
        """Count nominal member rows across the supplied sessions.

        Args:
            sessions: Sessions contributing to the population count.

        Returns:
            Sum of nominal member counts; repeated sessions contribute repeatedly.

        Raises:
            KeyError: A requested session is outside the declared epochs.
        """
        return sum(len(self.members(session)) for session in sessions)

    def cross_sections(
        self, sessions: Sequence[date], sector_by_listing_id: Mapping[str, str]
    ) -> tuple[PanelCrossSectionRange, ...]:
        """The identity of every session's cross-section, as contiguous ranges.

        Args:
            sessions: Ordered session axis to group by shared cross-section identity.
            sector_by_listing_id: Sector authority for each nominal member.

        Returns:
            Consecutive supplied sessions sharing an identity, with nominal member counts.

        Raises:
            KeyError: A requested session is outside the membership epochs.
            ValueError: A member lacks its sector or a cross-section is empty.
        """
        ranges: list[PanelCrossSectionRange] = []
        identities: dict[tuple[PanelMembershipEpoch, tuple[str, ...]], str] = {}
        for session in sessions:
            epoch = self.epoch(session)
            exclusions = self.excluded_sources(session)
            identity = identities.get((epoch, exclusions))
            if identity is None:
                identity = panel_cross_section_identity(
                    epoch.listing_ids, sector_by_listing_id, source_exclusions=exclusions
                )
                identities[epoch, exclusions] = identity
            if ranges and ranges[-1].cross_section_identity == identity:
                ranges[-1] = PanelCrossSectionRange(
                    ranges[-1].first_session, session, identity, len(epoch.listing_ids)
                )
            else:
                ranges.append(
                    PanelCrossSectionRange(session, session, identity, len(epoch.listing_ids))
                )
        return tuple(ranges)

    def summary_payload(self) -> dict[str, object]:
        """The list-free description a publication records; lists live in the journal.

        Returns:
        Population counts and hashes, epoch and basis summaries, bootstrap and journal
        references, and any evidenced source exclusions; nominal member lists stay in
        the full membership record.
        """
        latest = self.epochs[-1].listing_ids
        payload: dict[str, object] = {
            "axis_listing_count": len(self.listing_ids),
            "axis_listing_set_hash": canonical_hash(sorted(self.listing_ids)),
            # Members effective at the latest session, not the latest admitted
            # roster (which may already include future-effective entries).
            "as_of_member_count": len(latest),
            "as_of_listing_set_hash": canonical_hash(sorted(latest)),
            "journal_sequence": self.journal_sequence,
            "bootstrap_record_hash": self.bootstrap_record_hash,
            "bootstrap_t0_session": (
                self.bootstrap_t0_session.isoformat()
                if self.bootstrap_t0_session is not None
                else None
            ),
            "basis_ranges": [
                {
                    "first_session": item.first_session.isoformat(),
                    "last_session": item.last_session.isoformat(),
                    "basis": item.basis,
                }
                for item in self.basis_ranges
            ],
            "epochs": [
                {
                    "first_session": epoch.first_session.isoformat(),
                    "last_session": epoch.last_session.isoformat(),
                    "member_count": len(epoch.listing_ids),
                    "membership_hash": membership_identity(epoch.listing_ids),
                }
                for epoch in self.epochs
            ],
        }
        if self.source_exclusions:
            payload["source_exclusions"] = [item.to_payload() for item in self.source_exclusions]
        return payload

    def to_payload(self) -> dict[str, object]:
        """The full record a recipe needs to rebuild every session's members.

        The axis once; each epoch as the axis members it does not hold, which
        is the small side of a membership that changes a few names a year.

        Returns:
        Full historical axis, each epoch's absent members, qualification bases,
        source exclusions, and journal and bootstrap references.
        """
        return {
            **self.summary_payload(),
            "listing_ids": list(self.listing_ids),
            "epochs": [
                {
                    "first_session": epoch.first_session.isoformat(),
                    "last_session": epoch.last_session.isoformat(),
                    "absent_listing_ids": [
                        listing_id
                        for listing_id in self.listing_ids
                        if listing_id not in set(epoch.listing_ids)
                    ],
                }
                for epoch in self.epochs
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PanelMembership:
        """Restore temporal membership from the historical axis and absent-member epochs.

        Args:
            payload: Full membership record, with optional journal, bootstrap, and exclusions.

        Returns:
            Qualified membership with each epoch reconstructed in calculation-axis order.

        Raises:
            KeyError: A required membership field is absent.
            ValueError: Payload containers, dates, counts, or membership invariants are invalid.
        """
        axis = tuple(str(value) for value in _sequence(payload["listing_ids"]))
        epochs = []
        for item in _sequence(payload["epochs"]):
            entry = _mapping(item)
            absent = {str(value) for value in _sequence(entry.get("absent_listing_ids", ()))}
            epochs.append(
                PanelMembershipEpoch(
                    first_session=date.fromisoformat(str(entry["first_session"])),
                    last_session=date.fromisoformat(str(entry["last_session"])),
                    listing_ids=tuple(value for value in axis if value not in absent),
                )
            )
        t0 = payload.get("bootstrap_t0_session")
        record = payload.get("bootstrap_record_hash")
        return cls(
            listing_ids=axis,
            epochs=tuple(epochs),
            basis_ranges=tuple(
                PanelMembershipBasisRange(
                    first_session=date.fromisoformat(str(_mapping(item)["first_session"])),
                    last_session=date.fromisoformat(str(_mapping(item)["last_session"])),
                    basis=str(_mapping(item)["basis"]),
                )
                for item in _sequence(payload["basis_ranges"])
            ),
            journal_sequence=int(str(payload.get("journal_sequence", 0))),
            bootstrap_record_hash=str(record) if record is not None else None,
            bootstrap_t0_session=date.fromisoformat(str(t0)) if t0 is not None else None,
            source_exclusions=tuple(
                PanelSourceExclusion.from_payload(_mapping(item))
                for item in _sequence(payload.get("source_exclusions", ()))
            ),
        )


def panel_member_counts(
    manifest: Mapping[str, object], sessions: Sequence[date]
) -> dict[date, int]:
    """Declared sample population per session, distinct from the historical coverage axis.

    Args:
        manifest: Panel manifest carrying per-session membership or a dense population.
        sessions: Sessions requiring declared population counts.

    Returns:
        Population count per session, using membership epochs when recorded.

    Raises:
        KeyError: A required manifest or epoch field is absent.
        ValueError: Dates or counts are invalid, epochs are missing or overlap,
            a population is nonpositive, or requested sessions lack coverage.
    """
    summary = manifest.get("safe_summary")
    membership = summary.get("membership") if isinstance(summary, Mapping) else None
    if not isinstance(membership, Mapping):
        count = int(str(manifest["active_listing_count"]))
        return dict.fromkeys(sessions, count)
    epochs = membership.get("epochs")
    if not isinstance(epochs, list):
        raise ValueError("feature_panel.membership_epochs_missing")
    result: dict[date, int] = {}
    for epoch in epochs:
        first = date.fromisoformat(str(epoch["first_session"]))
        last = date.fromisoformat(str(epoch["last_session"]))
        count = int(str(epoch["member_count"]))
        if count < 1:
            raise ValueError("feature_panel.membership_count_invalid")
        for session in sessions:
            if first <= session <= last:
                if session in result:
                    raise ValueError("feature_panel.membership_epochs_overlap")
                result[session] = count
    if set(result) != set(sessions):
        raise ValueError("feature_panel.membership_session_missing")
    return result


def panel_source_manifest_revision(manifest: Mapping[str, object]) -> str:
    """The Data authority this snapshot actually names, never the workspace's latest pointer.

    Args:
        manifest: Panel snapshot with a recorded safe-summary lineage.

    Returns:
        Data manifest revision that the snapshot records as its source authority.

    Raises:
        ValueError: The recorded source revision is absent or has an invalid length.
    """
    summary = manifest.get("safe_summary")
    lineage = summary.get("lineage") if isinstance(summary, Mapping) else None
    revision = lineage.get("manifest_revision") if isinstance(lineage, Mapping) else None
    if not isinstance(revision, str) or len(revision) != 64:
        raise ValueError("feature_panel.source_manifest_revision_missing")
    return revision


def panel_as_of_listing_identity(manifest: Mapping[str, object]) -> tuple[str, int]:
    """The listing-set hash and count a Panel snapshot admits at its as-of session.

    A dense Panel's axis is its members, so its ``listing_set_hash`` and
    ``active_listing_count`` answer directly. A Panel with per-session
    membership records the as-of members in its membership summary; its
    axis holds every listing some session held and is not the answer.

    Args:
        manifest: Panel snapshot carrying dense counts or temporal membership summaries.

    Returns:
        Listing-set identity and member count admitted at the snapshot's as-of session.

    Raises:
        KeyError: The appropriate population fields are absent.
        ValueError: The recorded population count cannot be parsed.
    """
    summary = manifest.get("safe_summary")
    membership = summary.get("membership") if isinstance(summary, Mapping) else None
    if isinstance(membership, Mapping):
        return (
            str(membership["as_of_listing_set_hash"]),
            int(str(membership["as_of_member_count"])),
        )
    return str(manifest["listing_set_hash"]), int(str(manifest["active_listing_count"]))


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, (list, tuple)):
        return value
    raise ValueError("Panel membership payload field is not a sequence")


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise ValueError("Panel membership payload entry is not a mapping")


@dataclass(frozen=True)
class PanelTemporalRisk:
    """Explicit limits on the historical claims supported by this panel."""

    UNIVERSE_TEMPORAL_SCOPE: ClassVar[str] = "CURRENT_ACTIVE_SURVIVORS"
    SECTOR_SOURCE: ClassVar[str] = "YAHOO_CURRENT_SECTOR"
    SECTOR_HISTORY_TREATMENT: ClassVar[str] = SECTOR_HISTORY_BACKFILLED
    RESEARCH_USE_CLASS: ClassVar[str] = "CURRENT_UNIVERSE_RESEARCH_ONLY"

    universe_temporal_scope: str
    universe_point_in_time_qualified: bool
    sector_source: str
    sector_observed_at: datetime
    sector_point_in_time_qualified: bool
    sector_history_treatment: str
    research_use_class: str

    @classmethod
    def current_yahoo(
        cls, *, sector_observed_at: datetime, sector_history_treatment: str | None = None
    ) -> PanelTemporalRisk:
        """The limits of a Panel over Yahoo's current classification.

        Args:
            sector_observed_at: When the current classification was observed.
            sector_history_treatment: What its sessions read (`sector_treatment`);
                the backfill when omitted.

        Returns:
            The limits.
        """
        return cls(
            universe_temporal_scope=cls.UNIVERSE_TEMPORAL_SCOPE,
            universe_point_in_time_qualified=False,
            sector_source=cls.SECTOR_SOURCE,
            sector_observed_at=sector_observed_at,
            sector_point_in_time_qualified=False,
            sector_history_treatment=sector_history_treatment or cls.SECTOR_HISTORY_TREATMENT,
            research_use_class=cls.RESEARCH_USE_CLASS,
        )

    def risk_hash(self) -> str:
        """Hash the historical-claim limits and sector observation timestamp.

        Returns:
            Canonical identity of the complete temporal-risk disclosure.
        """
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class PanelAdmissionSummary:
    """Host-owned readiness decision for one panel as-of session."""

    as_of_session: date
    expected_factor_count: int
    recorded_factor_count: int
    available_factor_count: int
    unavailable_factor_count: int
    structural_failure_reasons: tuple[str, ...]
    small_sector_warning_factors: tuple[str, ...]
    sector_distribution: dict[str, int]
    research_admissible: bool

    @classmethod
    def evaluate(
        cls,
        *,
        as_of_session: date,
        factor_ids: tuple[str, ...],
        availability: list[dict[str, object]],
        sector_distribution: dict[str, int],
    ) -> PanelAdmissionSummary:
        """Decide research admission from as-of availability and sector population.

        Args:
            as_of_session: Session at which readiness is assessed.
            factor_ids: Complete expected feature axis.
            availability: Per-session factor availability and warning records.
            sector_distribution: Active population by sector.

        Returns:
            Counts, warnings, and structural failures. Admission requires a complete
            factor axis, at least one available factor, every sector population at
            least five, and no missing active-manifest base row.

        Raises:
            KeyError: An availability record lacks its session or factor identity.
        """
        current = [
            item for item in availability if str(item["session_date"]) == as_of_session.isoformat()
        ]
        recorded = {str(item["factor_id"]) for item in current}
        available = {
            str(item["factor_id"]) for item in current if item.get("status") == "available"
        }
        structural = set()
        if recorded != set(factor_ids):
            structural.add("availability_record_incomplete")
        if any(count < 5 for count in sector_distribution.values()):
            structural.add("active_sector_sample_below_5")
        if any(item.get("reason") == "active_manifest_base_row_missing" for item in current):
            structural.add("active_manifest_base_row_missing")
        if not available:
            structural.add("no_available_factor_at_as_of")
        warning_factors = tuple(
            sorted(
                {
                    str(item["factor_id"])
                    for item in current
                    if bool(item.get("small_sector_warning"))
                }
            )
        )
        return cls(
            as_of_session=as_of_session,
            expected_factor_count=len(factor_ids),
            recorded_factor_count=len(recorded),
            available_factor_count=len(available),
            unavailable_factor_count=len(recorded - available),
            structural_failure_reasons=tuple(sorted(structural)),
            small_sector_warning_factors=warning_factors,
            sector_distribution=dict(sorted(sector_distribution.items())),
            research_admissible=not structural,
        )

    def summary_hash(self) -> str:
        """Hash the complete admission decision and its supporting counts.

        Returns:
            Canonical identity of the readiness summary.
        """
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class PanelInvalidationRange:
    """Declare inclusive session bounds that require Panel recomputation.

    Attributes:
        first_session: First affected session.
        last_session: Last affected session.
    """

    first_session: date
    last_session: date

    def __post_init__(self) -> None:
        """Refuse reversed invalidation bounds.

        Raises:
            ValueError: The first affected session follows the last.
        """
        if self.first_session > self.last_session:
            raise ValueError("panel invalidation range is reversed")


@dataclass(frozen=True)
class FactorSessionInvalidation:
    """Associate one factor with its nonempty recomputation intervals.

    Attributes:
        factor_id: Factor whose values require recomputation.
        ranges: Inclusive affected session ranges in the supplied order.
    """

    factor_id: str
    ranges: tuple[PanelInvalidationRange, ...]

    def __post_init__(self) -> None:
        """Require a factor identity and at least one affected interval.

        Raises:
            ValueError: The factor ID or interval sequence is empty.
        """
        if not self.factor_id or not self.ranges:
            raise ValueError("factor-session invalidation must be non-empty")


@dataclass(frozen=True)
class FactorSessionInvalidationPlan:
    """Bind factor recomputation scopes to their source and input evidence.

    Attributes:
        items: Per-factor intervals in factor-ID order when built by create.
        source_receipt_hashes: Source evidence identities, sorted and deduplicated by create.
        expected_input_hashes: Expected input identities, sorted and deduplicated by create.
        plan_hash: Canonical identity of the scopes and both evidence axes.
    """

    items: tuple[FactorSessionInvalidation, ...]
    source_receipt_hashes: tuple[str, ...]
    expected_input_hashes: tuple[str, ...]
    plan_hash: str

    @classmethod
    def create(
        cls,
        *,
        items: tuple[FactorSessionInvalidation, ...],
        source_receipt_hashes: tuple[str, ...] = (),
        expected_input_hashes: tuple[str, ...] = (),
    ) -> FactorSessionInvalidationPlan:
        """Order factor scopes and bind them to unique source and input identities.

        Args:
            items: One invalidation item per factor, preserving each item's interval order.
            source_receipt_hashes: Receipts supporting the recomputation decision.
            expected_input_hashes: Input identities expected by the recomputation.

        Returns:
            Canonically hashed plan with sorted factors and sorted unique evidence axes.

        Raises:
            ValueError: More than one item names the same factor.
        """
        ordered = tuple(sorted(items, key=lambda item: item.factor_id))
        if len({item.factor_id for item in ordered}) != len(ordered):
            raise ValueError("factor-session plan contains duplicate factor identities")
        payload = {
            "items": [
                {
                    "factor_id": item.factor_id,
                    "ranges": [asdict(value) for value in item.ranges],
                }
                for item in ordered
            ],
            "source_receipt_hashes": sorted(set(source_receipt_hashes)),
            "expected_input_hashes": sorted(set(expected_input_hashes)),
        }
        return cls(
            items=ordered,
            source_receipt_hashes=tuple(payload["source_receipt_hashes"]),
            expected_input_hashes=tuple(payload["expected_input_hashes"]),
            plan_hash=canonical_hash(payload),
        )

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Return the plan's factor IDs in stored item order."""
        return tuple(item.factor_id for item in self.items)
