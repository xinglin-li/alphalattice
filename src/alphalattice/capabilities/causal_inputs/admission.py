"""The one place a strategy clock is compared against an input's own clock.

Every relation this capability enforces lives here and nowhere else. The Feature
owner, the Alpha producer, the Risk owner and the Portfolio Desk each supply a
thin resolver that turns their durable evidence into a
:class:`~alphalattice.capabilities.causal_inputs.contracts.CausalInputAuthority`;
none of them re-implements a comparison, and none of them owns a second copy of
the failure taxonomy.

**The relations.** For every formation, resolved as real timezone-aware instants
on the ordered exchange-session axis::

    input.observed_through_at   <= schedule.information_cutoff_at
    input.source_available_at   <= schedule.order_submission_deadline_at
    input.derived_ready_at      <= schedule.order_submission_deadline_at

    information_cutoff_at <= order_submission_deadline_at <= entry_at < exit_at

The second line is the one that did not exist before. Collapsing the deadline
into the cutoff would demand that everything be finished *at* ``close(T)``, which
is false for this strategy and is what made a producer reach for a session of
artificial lag. Keeping them apart lets a score that computes overnight be
admitted for what it is.

**The Target is checked separately and may not appear above.** It is a future
outcome, so ``target_available_at >= exit_at`` by construction and it would fail
the deadline relation on every formation. That is not a defect to be excused
with a special case: a Target in a decision input set is a leak, and
:func:`admit_decision_input_set` refuses one by role before any comparison runs.
Its maturity before a training cutoff is the causal outcome reader's to check.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta

from alphalattice.capabilities.causal_inputs.contracts import (
    AdmittedInformationSet,
    AnchoredInstantPolicy,
    CausalInputAuthority,
    SessionAnchor,
    StrategyDecisionSchedule,
    TemporalAdmissionError,
    TemporalAdmissionResult,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_EVENT_COLUMNS: Mapping[str, str] = {
    "OFFICIAL_OPEN": "session_open_timestamp",
    "OFFICIAL_CLOSE": "session_close_timestamp",
}
"""The calendar owner's own column names, so nothing here invents a clock."""


def resolve_session_instant(
    anchor: SessionAnchor,
    *,
    formation: date,
    axis_positions: Mapping[date, int],
    ordered_axis: Sequence[date],
    session_clocks: Mapping[date, Mapping[str, object]],
) -> datetime:
    """The real instant ``anchor`` names, relative to ``formation``.

    Offsets index the session axis, never the calendar, so a holiday or a half
    session moves the instant without moving the offset. A resolved value must
    be timezone-aware: a naive timestamp compared against an aware one raises in
    Python, and one silently coerced would compare wrongly by whole hours.
    """
    position = axis_positions.get(formation)
    if position is None:
        raise TemporalAdmissionError("causal_inputs.formation_off_session_axis")
    target = position + anchor.offset_sessions
    if target < 0 or target >= len(ordered_axis):
        raise TemporalAdmissionError("causal_inputs.anchor_off_session_axis")
    session = ordered_axis[target]
    clock = session_clocks.get(session)
    if clock is None:
        raise TemporalAdmissionError("causal_inputs.session_clock_absent")
    value = clock.get(_EVENT_COLUMNS[anchor.event])
    if not isinstance(value, datetime):
        raise TemporalAdmissionError("causal_inputs.session_clock_event_absent")
    if value.tzinfo is None or value.utcoffset() is None:
        raise TemporalAdmissionError("causal_inputs.session_clock_not_timezone_aware")
    return value


def resolve_policy_instant(
    policy: AnchoredInstantPolicy,
    *,
    formation: date,
    axis_positions: Mapping[date, int],
    ordered_axis: Sequence[date],
    session_clocks: Mapping[date, Mapping[str, object]],
) -> datetime:
    """A policy instant: a real anchor plus the offset somebody decided on."""
    anchored = resolve_session_instant(
        policy.anchor,
        formation=formation,
        axis_positions=axis_positions,
        ordered_axis=ordered_axis,
        session_clocks=session_clocks,
    )
    return anchored + timedelta(minutes=policy.minutes_after_anchor)


def require_schedule_ordering(
    *,
    schedule: StrategyDecisionSchedule,
    formation_sessions: Sequence[date],
    ordered_axis: Sequence[date],
    session_clocks: Mapping[date, Mapping[str, object]],
) -> None:
    """``cutoff <= deadline <= entry < exit`` on every formation, as instants.

    Checked per formation rather than once on the offsets. The offsets can be
    ordered while the instants are not: a deadline expressed as minutes before an
    open is an instant, and a long weekend or an early close moves the anchors
    around it. Once, on the conventions, is exactly the check that holds on
    average and fails next to the hole.
    """
    positions = {session: index for index, session in enumerate(ordered_axis)}
    for formation in formation_sessions:
        resolved = {
            name: resolve_policy_instant(
                policy,
                formation=formation,
                axis_positions=positions,
                ordered_axis=ordered_axis,
                session_clocks=session_clocks,
            )
            for name, policy in (
                ("cutoff", schedule.information_cutoff),
                ("deadline", schedule.order_submission_deadline),
                ("entry", schedule.entry),
                ("exit", schedule.exit),
            )
        }
        cutoff, deadline = resolved["cutoff"], resolved["deadline"]
        entry, exit_at = resolved["entry"], resolved["exit"]
        if not cutoff <= deadline:
            raise TemporalAdmissionError("causal_inputs.schedule_deadline_precedes_cutoff")
        if not deadline <= entry:
            raise TemporalAdmissionError("causal_inputs.schedule_entry_precedes_deadline")
        if not entry < exit_at:
            raise TemporalAdmissionError("causal_inputs.schedule_exit_not_after_entry")


def failed_decision_relation(
    *,
    observed_at: datetime,
    available_at: datetime,
    ready_at: datetime,
    cutoff_at: datetime,
    deadline_at: datetime,
    entry_at: datetime,
    exit_at: datetime,
) -> str | None:
    """The one implementation of the decision ordering. Returns what failed.

    Every caller that has resolved instants -- whether from session anchors here
    or from an execution owner's published rows elsewhere -- comes through this
    function, so there is exactly one place the inequalities live and exactly one
    vocabulary for what went wrong.

    A caller whose contract does not model a submission deadline passes the
    cutoff for both. That is strictly stronger than a schedule requires, and
    being able to say so precisely is the reason this takes instants rather than
    a schedule: the relation is the shared thing, the resolution is not.
    """
    # The input's own chain first. These are properties of the input alone and
    # hold under every strategy, so a violation here is a broken authority rather
    # than an unusable one -- and offsets cannot detect it: an input observed at
    # close(T) claiming availability at open(T) has both anchors on offset zero.
    if not observed_at <= available_at:
        return "SOURCE_AVAILABLE_BEFORE_OBSERVATION"
    if not available_at <= ready_at:
        return "DERIVED_READY_BEFORE_SOURCE_AVAILABLE"
    if not observed_at <= cutoff_at:
        return "OBSERVED_THROUGH_AFTER_INFORMATION_CUTOFF"
    if not available_at <= deadline_at:
        return "SOURCE_AVAILABLE_AFTER_ORDER_DEADLINE"
    if not ready_at <= deadline_at:
        return "DERIVED_READY_AFTER_ORDER_DEADLINE"
    if not deadline_at <= entry_at:
        return "ORDER_DEADLINE_AFTER_ENTRY"
    if not cutoff_at < entry_at:
        # Strict, and true of every schedule: deciding at the instant of
        # execution is not a decision. The deadline relation above is
        # deliberately non-strict -- an order lodged exactly at the cutoff is
        # lodged -- so this is the one that catches a fixture whose entry
        # collapsed onto its own cutoff.
        return "ENTRY_NOT_AFTER_INFORMATION_CUTOFF"
    if not entry_at < exit_at:
        return "EXIT_NOT_AFTER_ENTRY"
    return None


def _horizon_alignment(
    *, authority: CausalInputAuthority, schedule: StrategyDecisionSchedule
) -> str:
    """Does this input measure the interval the strategy is exposed to?

    Compared on the anchors, which is where the question is decidable. A surface
    with no declared start is not a window and is reported as such rather than
    being assumed to match.
    """
    if authority.observation_start is None:
        return "NOT_A_WINDOWED_OBSERVATION"
    start = authority.observation_start.anchor
    end = authority.observed_through.anchor
    target_start, target_end = schedule.target_start.anchor, schedule.target_end.anchor
    exchange_window = (
        authority.observation_start.minutes_after_anchor
        == schedule.target_start.minutes_after_anchor
        and authority.observed_through.minutes_after_anchor
        == schedule.target_end.minutes_after_anchor
    )
    if exchange_window and (start, end) == (target_start, target_end):
        return "MATCHED_TO_STRATEGY_WINDOW"
    same_unit = (
        start.event == target_start.event
        and end.event == target_end.event
        and end.offset_sessions - start.offset_sessions
        == target_end.offset_sessions - target_start.offset_sessions
    )
    return "SAME_UNIT_DIFFERENT_WINDOW" if same_unit else "DIFFERENT_UNIT_AND_WINDOW"


def admit_causal_input(
    *,
    authority: CausalInputAuthority,
    schedule: StrategyDecisionSchedule,
    formation_sessions: Sequence[date],
    ordered_axis: Sequence[date],
    session_clocks: Mapping[date, Mapping[str, object]],
) -> TemporalAdmissionResult:
    """Compare one input's clock against one strategy's clock, per formation.

    Streams the axis and keeps only counts, the first exception and the last
    admitted formation's instants. A 1,260-formation study over ten inputs
    therefore seals ten compact rows instead of 12,600 timestamps that the
    session axis already determines.
    """
    if authority.temporal_usage == "TARGET_OUTCOME":
        raise TemporalAdmissionError("causal_inputs.target_is_not_a_decision_input")
    if not formation_sessions:
        raise TemporalAdmissionError("causal_inputs.formation_axis_empty")

    positions = {session: index for index, session in enumerate(ordered_axis)}
    refused_relation: str | None = None
    first_refused: date | None = None
    refused_count = 0
    last: dict[str, datetime] = {}

    for formation in formation_sessions:

        def policy_instant(policy: AnchoredInstantPolicy, at: date = formation) -> datetime:
            return resolve_policy_instant(
                policy,
                formation=at,
                axis_positions=positions,
                ordered_axis=ordered_axis,
                session_clocks=session_clocks,
            )

        observed = policy_instant(authority.observed_through)
        available = policy_instant(authority.source_available)
        cutoff = policy_instant(schedule.information_cutoff)
        entry = policy_instant(schedule.entry)
        exit_at = policy_instant(schedule.exit)
        deadline = policy_instant(schedule.order_submission_deadline)
        ready = (
            available
            if authority.derived_ready is None
            else policy_instant(authority.derived_ready)
        )

        relation = failed_decision_relation(
            observed_at=observed,
            available_at=available,
            ready_at=ready,
            cutoff_at=cutoff,
            deadline_at=deadline,
            entry_at=entry,
            exit_at=exit_at,
        )
        if relation is not None:
            refused_count += 1
            if refused_relation is None:
                refused_relation, first_refused = relation, formation
            continue
        last = {
            "observed_through_at": observed,
            "source_available_at": available,
            "derived_ready_at": ready,
            "information_cutoff_at": cutoff,
            "order_submission_deadline_at": deadline,
            "entry_at": entry,
            "exit_at": exit_at,
        }

    admitted = refused_relation is None
    measured = (
        {
            "observation_staleness_at_decision_minutes": (
                last["information_cutoff_at"] - last["observed_through_at"]
            ).total_seconds()
            / 60.0,
            "ready_to_submission_deadline_minutes": (
                last["order_submission_deadline_at"] - last["derived_ready_at"]
            ).total_seconds()
            / 60.0,
        }
        if admitted and last
        else {}
    )
    return TemporalAdmissionResult.create(
        input_id=authority.input_id,
        input_kind=authority.input_kind,
        temporal_usage=authority.temporal_usage,
        input_authority_hash=authority.authority_hash,
        strategy_schedule_hash=schedule.schedule_hash,
        disposition="ADMITTED" if admitted else "REFUSED",
        refused_relation=refused_relation,
        formation_count=len(formation_sessions),
        refused_formation_count=refused_count,
        first_refused_formation=first_refused,
        point_in_time_disposition=authority.point_in_time_disposition,
        horizon_alignment=_horizon_alignment(authority=authority, schedule=schedule),
        **last,
        **measured,
    )


def admit_decision_input_set(
    *,
    authorities: Sequence[CausalInputAuthority],
    schedule: StrategyDecisionSchedule,
    formation_sessions: Sequence[date],
    ordered_axis: Sequence[date],
    session_clocks: Mapping[date, Mapping[str, object]],
) -> AdmittedInformationSet:
    """Admit a whole decision input set and seal the consumption binding.

    The result names the fact surfaces by their own unchanged hashes. Re-running
    these same facts under a different schedule yields a different
    ``admission_hash`` and moves no ``input_surface_hash`` at all, which is the
    firewall stated as an artifact rather than as a rule.
    """
    if not authorities:
        raise TemporalAdmissionError("causal_inputs.decision_input_set_empty")
    ids = [item.input_id for item in authorities]
    if len(set(ids)) != len(ids):
        raise TemporalAdmissionError("causal_inputs.decision_input_id_duplicated")
    leaked = tuple(item for item in authorities if item.temporal_usage == "TARGET_OUTCOME")
    if leaked:
        # Refused by role, before a single instant is resolved. A Target that
        # reached the comparisons would be refused for missing the deadline,
        # which reads as a timing defect rather than as the leak it is.
        raise TemporalAdmissionError("causal_inputs.target_is_not_a_decision_input")

    require_schedule_ordering(
        schedule=schedule,
        formation_sessions=formation_sessions,
        ordered_axis=ordered_axis,
        session_clocks=session_clocks,
    )
    results = tuple(
        admit_causal_input(
            authority=item,
            schedule=schedule,
            formation_sessions=formation_sessions,
            ordered_axis=ordered_axis,
            session_clocks=session_clocks,
        )
        for item in sorted(authorities, key=lambda value: value.input_id)
    )
    admitted = all(item.disposition == "ADMITTED" for item in results)
    ordered_formations = tuple(sorted(formation_sessions))
    return AdmittedInformationSet.create(
        strategy_schedule_hash=schedule.schedule_hash,
        input_surface_hashes={item.input_id: item.surface_hash for item in authorities},
        input_temporal_authority_hashes={
            item.input_id: item.authority_hash for item in authorities
        },
        results=results,
        admitted_sessions_hash=str(
            canonical_hash([value.isoformat() for value in ordered_formations])
            if admitted
            else canonical_hash([])
        ),
        admitted_session_count=len(ordered_formations) if admitted else 0,
        first_admitted_session=ordered_formations[0] if admitted else None,
        last_admitted_session=ordered_formations[-1] if admitted else None,
        formation_axis_hash=str(
            canonical_hash([value.isoformat() for value in ordered_formations])
        ),
        session_axis_hash=str(canonical_hash([value.isoformat() for value in ordered_axis])),
        disposition="ADMITTED" if admitted else "REFUSED",
    )


__all__ = [
    "admit_causal_input",
    "admit_decision_input_set",
    "failed_decision_relation",
    "require_schedule_ordering",
    "resolve_policy_instant",
    "resolve_session_instant",
]
