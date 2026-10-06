"""Forward membership history of one workspace Universe: a cohort, then dated events.

The workspace's first successful research admission freezes an initial
qualified cohort U0 and its session T0. U0 is the fixed research cohort for
the historical backfill before T0 -- an explicit initialization assumption,
not historical point-in-time membership. From T0 onward membership is the
cohort evolved by recorded entries and exits, each with the session it
takes effect, the time the evidence was observed and the time the decision
was made. Earlier intervals are never rewritten: a listing joining later is
absent before its entry, a listing leaving keeps what it had, a re-entry is
a new interval and the gap stays.

The journal is incremental (the cohort once, then only actual changes) and
the interval index is a pure function of it, recomputed on demand. Each
contiguous range of sessions with the same members is an epoch carrying one
membership identity; a future event changes the identity of no session it
does not reach.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from itertools import pairwise
from typing import Literal

from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.shared_kernel.identity import canonical_hash

INITIAL_COHORT_BACKFILL = "INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME"
"""The one initialization assumption: U0 stands in for every session before T0."""

FORWARD_AS_OBSERVED = "FORWARD_AS_OBSERVED_GOVERNED_MEMBERSHIP"
"""The forward promise: the membership admitted at each session, as observed then."""

MembershipEventKind = Literal["ENTRY", "EXIT"]


@lru_cache(maxsize=32)
def _membership_clocks(start: date, end: date) -> tuple[tuple[date, datetime, datetime], ...]:
    """Planned exchange clocks, not a claim that future market data were observed."""
    calendar = materialize_calendar_schedule(
        ("XNYS", "XNAS"),
        start=start,
        end=end,
        as_of_timestamp=datetime.combine(end, time.max, tzinfo=UTC),
    )
    rows: dict[date, list[tuple[datetime, datetime]]] = {}
    for row in calendar.to_pylist():
        rows.setdefault(row["session_date"], []).append(
            (row["session_open_timestamp"], row["session_close_timestamp"])
        )
    return tuple(
        (session, min(opened for opened, _ in clocks), min(closed for _, closed in clocks))
        for session, clocks in sorted(rows.items())
        if len(clocks) == 2
    )


def membership_effective_session(
    *, observed_at: datetime, decided_at: datetime, not_before: date
) -> date:
    """Earliest formation whose known information and next-open deadline admit the change.

    An old catch-up target is not an observation timestamp. Facts must be
    observed by formation close; an overnight decision on already known facts
    is allowed before the next common open. Calendar enumeration uses future
    schedule clocks only, never a future Provider observation or readiness claim.
    """
    if any(value.tzinfo is None for value in (observed_at, decided_at)):
        raise ValueError("universe_membership.timestamp_timezone_required")
    observed = observed_at.astimezone(UTC)
    decided = decided_at.astimezone(UTC)
    if decided < observed:
        raise ValueError("universe_membership.decision_precedes_observation")
    start = max(not_before, observed.date() - timedelta(days=2))
    end = max(start, decided.date()) + timedelta(days=14)
    clocks = _membership_clocks(start, end)
    for (session, _opened, closed), (_next, next_open, _close) in pairwise(clocks):
        if observed <= closed and decided < next_open:
            return session
    raise ValueError("universe_membership.effective_session_unavailable")


@dataclass(frozen=True)
class UniverseSourceObservation:
    """A source check, including unchanged checks; contains references, never member lists."""

    market_profile_id: str
    observed_at: datetime
    candidate_membership_hash: str
    source_identity_hash: str
    first_eligible_session: date
    previous_observed_at: datetime | None = None
    observation_hash: str = ""

    def __post_init__(self) -> None:
        """Normalize observation clocks and verify the content identity.

        Raises:
            ValueError: A clock is naive, out of order, or the supplied hash differs.

        """
        for name in ("observed_at", "previous_observed_at"):
            value = getattr(self, name)
            if value is not None:
                if value.tzinfo is None:
                    raise ValueError("universe_membership.timestamp_timezone_required")
                object.__setattr__(self, name, value.astimezone(UTC))
        if self.previous_observed_at is not None and self.previous_observed_at > self.observed_at:
            raise ValueError("universe_membership.observation_precedes_previous")
        expected = canonical_hash(
            {
                "kind": "UniverseSourceObservation",
                "market_profile_id": self.market_profile_id,
                "observed_at": self.observed_at.isoformat(),
                "candidate_membership_hash": self.candidate_membership_hash,
                "source_identity_hash": self.source_identity_hash,
                "first_eligible_session": self.first_eligible_session.isoformat(),
                "previous_observed_at": (
                    self.previous_observed_at.isoformat()
                    if self.previous_observed_at is not None
                    else None
                ),
            }
        )
        if self.observation_hash and self.observation_hash != expected:
            raise ValueError("universe_membership.observation_identity_mismatch")
        object.__setattr__(self, "observation_hash", expected)


def membership_identity(listing_ids: Iterable[str]) -> str:
    """Hash one member set independently of how it was selected."""
    return str(
        canonical_hash({"kind": "UniverseMembership", "listing_ids": sorted(set(listing_ids))})
    )


@dataclass(frozen=True)
class UniverseBootstrapRecord:
    """T0, U0 and everything that makes the backfill assumption attributable."""

    market_profile_id: str
    t0_session: date
    history_start: date
    cohort_listing_ids: tuple[str, ...]
    cohort_hash: str
    manifest_revision: str
    candidate_manifest_hash: str
    qualification_policy_hash: str
    feature_input_policy_hash: str
    source_observed_at: datetime
    admitted_at: datetime
    panel_snapshot_hash: str
    derivation: Literal["FIRST_QUALIFIED_PUBLICATION", "RECONSTRUCTED_FROM_DURABLE_EVIDENCE"]
    initialization_assumption: str = INITIAL_COHORT_BACKFILL
    record_hash: str = ""

    def __post_init__(self) -> None:
        """Canonicalize the initial cohort and verify its recorded identities.

        Raises:
            ValueError: The cohort, clocks, assumption, or hashes are invalid.

        """
        if self.history_start > self.t0_session:
            raise ValueError("universe bootstrap history start is after T0")
        if self.source_observed_at.tzinfo is None or self.admitted_at.tzinfo is None:
            raise ValueError("universe bootstrap timestamps must be timezone-aware")
        cohort = tuple(sorted(set(self.cohort_listing_ids)))
        if not cohort:
            raise ValueError("universe bootstrap cohort is empty")
        if self.initialization_assumption != INITIAL_COHORT_BACKFILL:
            raise ValueError("universe bootstrap must declare the initial-cohort assumption")
        object.__setattr__(self, "cohort_listing_ids", cohort)
        expected_cohort = membership_identity(cohort)
        if self.cohort_hash and self.cohort_hash != expected_cohort:
            raise ValueError("universe bootstrap cohort hash does not match its listings")
        object.__setattr__(self, "cohort_hash", expected_cohort)
        expected = canonical_hash(
            {
                "kind": "UniverseBootstrapRecord",
                "market_profile_id": self.market_profile_id,
                "t0_session": self.t0_session.isoformat(),
                "history_start": self.history_start.isoformat(),
                "cohort_hash": expected_cohort,
                "manifest_revision": self.manifest_revision,
                "candidate_manifest_hash": self.candidate_manifest_hash,
                "qualification_policy_hash": self.qualification_policy_hash,
                "feature_input_policy_hash": self.feature_input_policy_hash,
                "source_observed_at": self.source_observed_at.isoformat(),
                "admitted_at": self.admitted_at.isoformat(),
                "panel_snapshot_hash": self.panel_snapshot_hash,
                "derivation": self.derivation,
                "initialization_assumption": self.initialization_assumption,
            }
        )
        if self.record_hash and self.record_hash != expected:
            raise ValueError("universe bootstrap record hash does not match its content")
        object.__setattr__(self, "record_hash", expected)


@dataclass(frozen=True)
class MembershipEvent:
    """One listing entering or leaving the Universe from an effective session on."""

    market_profile_id: str
    sequence: int
    listing_id: str
    kind: MembershipEventKind
    effective_session: date
    observed_at: datetime
    decided_at: datetime
    authority: str
    reference_hash: str
    manifest_revision: str
    event_hash: str = ""

    def __post_init__(self) -> None:
        """Normalize event clocks and verify the event identity.

        Raises:
            ValueError: The sequence, clocks, authority, or hash is invalid.

        """
        if self.sequence < 1:
            raise ValueError("membership event sequence must be positive")
        if self.observed_at.tzinfo is None or self.decided_at.tzinfo is None:
            raise ValueError("membership event timestamps must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        object.__setattr__(self, "decided_at", self.decided_at.astimezone(UTC))
        if not self.authority or not self.reference_hash:
            raise ValueError("membership event requires its authority and reference")
        expected = canonical_hash(
            {
                "kind": "UniverseMembershipEvent",
                "market_profile_id": self.market_profile_id,
                "sequence": self.sequence,
                "listing_id": self.listing_id,
                "event": self.kind,
                "effective_session": self.effective_session.isoformat(),
                "observed_at": self.observed_at.isoformat(),
                "decided_at": self.decided_at.isoformat(),
                "authority": self.authority,
                "reference_hash": self.reference_hash,
                "manifest_revision": self.manifest_revision,
            }
        )
        if self.event_hash and self.event_hash != expected:
            raise ValueError("membership event hash does not match its content")
        object.__setattr__(self, "event_hash", expected)


@dataclass(frozen=True)
class MembershipEpoch:
    """A contiguous range of sessions with one member set and one identity."""

    first_session: date
    last_session: date
    listing_ids: tuple[str, ...]
    membership_hash: str = ""

    def __post_init__(self) -> None:
        """Canonicalize the epoch's members and verify its identity.

        Raises:
            ValueError: The range is reversed or the supplied hash differs.

        """
        if self.first_session > self.last_session:
            raise ValueError("membership epoch range is reversed")
        listing_ids = tuple(sorted(set(self.listing_ids)))
        object.__setattr__(self, "listing_ids", listing_ids)
        expected = membership_identity(listing_ids)
        if self.membership_hash and self.membership_hash != expected:
            raise ValueError("membership epoch identity does not match its listings")
        object.__setattr__(self, "membership_hash", expected)


@dataclass(frozen=True)
class MembershipBasisRange:
    """Which promise a range of sessions is under."""

    first_session: date
    last_session: date
    basis: str


@dataclass(frozen=True)
class MembershipSchedule:
    """Per-session membership over a calendar, resolved from the journal."""

    sessions: tuple[date, ...]
    epochs: tuple[MembershipEpoch, ...]
    basis_ranges: tuple[MembershipBasisRange, ...]
    bootstrap: UniverseBootstrapRecord | None
    journal_sequence: int
    admitted_listing_ids: tuple[str, ...]
    """The journal's latest admitted roster, which may include future-effective decisions."""
    _by_session: dict[date, MembershipEpoch] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Index the calendar after checking coverage and unique ordering.

        Raises:
            ValueError: Sessions are invalid or epochs overlap or leave gaps.

        """
        if not self.sessions:
            raise ValueError("membership schedule requires sessions")
        if self.sessions != tuple(sorted(set(self.sessions))):
            raise ValueError("membership schedule sessions must be sorted and unique")
        covered: list[date] = []
        by_session: dict[date, MembershipEpoch] = {}
        for epoch in self.epochs:
            for session in self.sessions:
                if epoch.first_session <= session <= epoch.last_session:
                    if session in by_session:
                        raise ValueError("membership epochs overlap")
                    by_session[session] = epoch
                    covered.append(session)
        if tuple(sorted(covered)) != self.sessions:
            raise ValueError("membership epochs do not cover the schedule's sessions")
        object.__setattr__(self, "_by_session", by_session)

    def members(self, session: date) -> tuple[str, ...]:
        """Return the members effective for one session.

        Raises:
            KeyError: The session is outside this schedule.

        """
        return self._by_session[session].listing_ids

    def epoch(self, session: date) -> MembershipEpoch:
        """Return the membership epoch covering one session.

        Raises:
            KeyError: The session is outside this schedule.

        """
        return self._by_session[session]

    @property
    def union(self) -> tuple[str, ...]:
        """Return every member present in any epoch, in canonical order."""
        return tuple(sorted({listing for epoch in self.epochs for listing in epoch.listing_ids}))

    def members_by_session(self, sessions: Sequence[date]) -> dict[date, tuple[str, ...]]:
        """Map requested sessions to their effective member sets.

        Raises:
            KeyError: A requested session is outside this schedule.

        """
        return {session: self.members(session) for session in sessions}

    def identity_by_session(self, sessions: Sequence[date]) -> dict[date, str]:
        """Map requested sessions to their membership identities.

        Raises:
            KeyError: A requested session is outside this schedule.

        """
        return {session: self.epoch(session).membership_hash for session in sessions}

    def basis(self, session: date) -> str:
        """Return the promise that governs membership for one session.

        Raises:
            KeyError: No basis range covers the session.

        """
        for item in self.basis_ranges:
            if item.first_session <= session <= item.last_session:
                return item.basis
        raise KeyError(session)

    def row_count(self) -> int:
        """Count all session-member rows in the schedule."""
        return sum(len(self.members(session)) for session in self.sessions)


def resolve_membership_schedule(
    *,
    sessions: Sequence[date],
    bootstrap: UniverseBootstrapRecord | None,
    events: Sequence[MembershipEvent],
    fallback_listing_ids: Sequence[str],
) -> MembershipSchedule:
    """Replay the journal onto a calendar.

    Without a bootstrap record the workspace is still building its initial
    cohort: every session holds ``fallback_listing_ids`` (the manifest being
    built) and the whole calendar is under the initialization assumption.
    With one, sessions before T0 hold U0 and sessions from T0 on hold U0
    evolved by the events effective at or before them, in journal order.
    """
    calendar = tuple(sorted(set(sessions)))
    if not calendar:
        raise ValueError("membership schedule requires sessions")
    if bootstrap is None:
        listing_ids = tuple(sorted(set(fallback_listing_ids)))
        return MembershipSchedule(
            sessions=calendar,
            epochs=(
                MembershipEpoch(
                    first_session=calendar[0], last_session=calendar[-1], listing_ids=listing_ids
                ),
            ),
            basis_ranges=(
                MembershipBasisRange(calendar[0], calendar[-1], INITIAL_COHORT_BACKFILL),
            ),
            bootstrap=None,
            journal_sequence=0,
            admitted_listing_ids=listing_ids,
        )
    ordered = sorted(events, key=lambda item: item.sequence)
    if any(item.effective_session < bootstrap.t0_session for item in ordered):
        raise ValueError("membership event takes effect before the bootstrap session")
    members = set(bootstrap.cohort_listing_ids)
    epochs: list[MembershipEpoch] = []
    # Applied in effective order; the journal sequence breaks ties, so two
    # events for one session apply in the order they were decided.
    pending = sorted(ordered, key=lambda item: (item.effective_session, item.sequence))
    epoch_start = calendar[0]
    current_ids = tuple(sorted(members))
    for session in calendar:
        changed = False
        while pending and pending[0].effective_session <= session:
            event = pending.pop(0)
            if event.kind == "ENTRY":
                if event.listing_id in members:
                    raise ValueError("membership entry for a listing already present")
                members.add(event.listing_id)
            else:
                if event.listing_id not in members:
                    raise ValueError("membership exit for a listing not present")
                members.remove(event.listing_id)
            changed = True
        if changed and tuple(sorted(members)) != current_ids:
            previous = _previous_session(calendar, session)
            if previous is not None and previous >= epoch_start:
                epochs.append(
                    MembershipEpoch(
                        first_session=epoch_start, last_session=previous, listing_ids=current_ids
                    )
                )
            epoch_start = session
            current_ids = tuple(sorted(members))
    epochs.append(
        MembershipEpoch(
            first_session=epoch_start, last_session=calendar[-1], listing_ids=current_ids
        )
    )
    basis_ranges: list[MembershipBasisRange] = []
    backfill = tuple(session for session in calendar if session < bootstrap.t0_session)
    forward = tuple(session for session in calendar if session >= bootstrap.t0_session)
    if backfill:
        basis_ranges.append(
            MembershipBasisRange(backfill[0], backfill[-1], INITIAL_COHORT_BACKFILL)
        )
    if forward:
        basis_ranges.append(MembershipBasisRange(forward[0], forward[-1], FORWARD_AS_OBSERVED))
    return MembershipSchedule(
        sessions=calendar,
        epochs=tuple(epochs),
        basis_ranges=tuple(basis_ranges),
        bootstrap=bootstrap,
        journal_sequence=ordered[-1].sequence if ordered else 0,
        admitted_listing_ids=journal_members(bootstrap, ordered),
    )


def membership_events_for_transition(
    *,
    market_profile_id: str,
    current_listing_ids: Iterable[str],
    next_listing_ids: Iterable[str],
    effective_session: date,
    observed_at: datetime,
    decided_at: datetime,
    authority: str,
    reference_hash: str,
    manifest_revision: str,
    first_sequence: int,
) -> tuple[MembershipEvent, ...]:
    """List the exits and entries between member sets in journal order."""
    current = set(current_listing_ids)
    following = set(next_listing_ids)
    events: list[MembershipEvent] = []
    sequence = first_sequence
    for listing_id in sorted(current - following):
        events.append(
            MembershipEvent(
                market_profile_id=market_profile_id,
                sequence=sequence,
                listing_id=listing_id,
                kind="EXIT",
                effective_session=effective_session,
                observed_at=observed_at,
                decided_at=decided_at,
                authority=authority,
                reference_hash=reference_hash,
                manifest_revision=manifest_revision,
            )
        )
        sequence += 1
    for listing_id in sorted(following - current):
        events.append(
            MembershipEvent(
                market_profile_id=market_profile_id,
                sequence=sequence,
                listing_id=listing_id,
                kind="ENTRY",
                effective_session=effective_session,
                observed_at=observed_at,
                decided_at=decided_at,
                authority=authority,
                reference_hash=reference_hash,
                manifest_revision=manifest_revision,
            )
        )
        sequence += 1
    return tuple(events)


def journal_members(
    bootstrap: UniverseBootstrapRecord, events: Sequence[MembershipEvent]
) -> tuple[str, ...]:
    """Apply recorded events to obtain the journal's current member set."""
    members = set(bootstrap.cohort_listing_ids)
    for event in sorted(events, key=lambda item: item.sequence):
        if event.kind == "ENTRY":
            if event.listing_id in members:
                raise ValueError("membership entry for a listing already present")
            members.add(event.listing_id)
        else:
            if event.listing_id not in members:
                raise ValueError("membership exit for a listing not present")
            members.remove(event.listing_id)
    return tuple(sorted(members))


def _previous_session(calendar: Sequence[date], session: date) -> date | None:
    index = calendar.index(session)
    return calendar[index - 1] if index > 0 else None


__all__ = [
    "FORWARD_AS_OBSERVED",
    "INITIAL_COHORT_BACKFILL",
    "MembershipBasisRange",
    "MembershipEpoch",
    "MembershipEvent",
    "MembershipSchedule",
    "UniverseBootstrapRecord",
    "UniverseSourceObservation",
    "membership_effective_session",
    "membership_events_for_transition",
    "membership_identity",
    "resolve_membership_schedule",
]
