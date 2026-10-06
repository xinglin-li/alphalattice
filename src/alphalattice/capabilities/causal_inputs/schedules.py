"""Re-derive a strategy schedule from the installed execution method.

The execution owner in ``foundation.causal_outcomes.execution`` remains the
authority on entry, exit and information cutoff; nothing here decides them. This
module reads an installed ``ExecutionOutcomeMethodRecipe`` and restates it in the
neutral vocabulary Risk, Alpha and Portfolio can all consume, so a schedule is
never assembled from literals a caller typed.

It lives beside the contracts rather than inside a Desk because all three Desks
need the same schedule. Homing it in the Portfolio Desk would make Risk import
Portfolio to learn what session it may read, which is the coupling this whole
capability exists to remove. The dependency runs ``capabilities -> foundation``,
the direction the package layering already uses; the execution owner does not
import back.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Final, NamedTuple

import pyarrow as pa

from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    SessionAnchor,
    SessionEvent,
    StrategyDecisionSchedule,
    TemporalAdmissionError,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ONE_SESSION_RECIPE_ID,
    ExecutionOutcomeMethodRecipe,
)

NEXT_OPEN_AFTER_OFFICIAL_CLOSE: Final = "NEXT_OPEN_AFTER_OFFICIAL_CLOSE"
"""The strategy this study runs: decide at the close, trade the next opens.

A real business time point, which is why it may be a name. It says nothing about
a machine, an agent, a branch or an implementation stage.
"""

PRE_OPEN_SUBMISSION_DEADLINE: Final = "DEVELOPMENT_PRE_OPEN_SUBMISSION_DEADLINE"
PRE_OPEN_SUBMISSION_MINUTES_BEFORE_ENTRY: Final = 30

_DEADLINE_RATIONALE: Final = (
    "Installed development operational policy, not an exchange fact and not a "
    "broker-confirmed cutoff. No venue, broker or market-on-open deadline is "
    "published anywhere in this repository, so a submission time was decided "
    "rather than observed; it is stated here so a result can be re-read under a "
    "different one."
)

_EVENT_SUFFIXES: Final[tuple[SessionEvent, ...]] = ("OFFICIAL_OPEN", "OFFICIAL_CLOSE")


class InstalledScheduleBinding(NamedTuple):
    """One installed strategy clock: which method it trades, whose deadline it uses.

    A binding rather than a default. Previously one ``schedule_id`` defaulted for
    every recipe, so a five-session method silently resolved as the next-open
    strategy; now a caller names an installed binding and the resolver checks it
    against the recipe it claims.
    """

    schedule_id: str
    execution_recipe_id: str
    order_submission_policy_handle: str


INSTALLED_ORDER_SUBMISSION_POLICIES: Final[Mapping[str, tuple[int, str]]] = {
    PRE_OPEN_SUBMISSION_DEADLINE: (PRE_OPEN_SUBMISSION_MINUTES_BEFORE_ENTRY, _DEADLINE_RATIONALE),
}
"""Every order-submission policy this build installs, by handle.

Selected, never defaulted. A caller that names no handle gets a refusal rather
than thirty minutes, because a deadline nobody chose is a deadline nobody can
argue with -- and it lands in the schedule identity, so it must be a decision.
"""

INSTALLED_STRATEGY_SCHEDULES: Final[Mapping[str, InstalledScheduleBinding]] = {
    NEXT_OPEN_AFTER_OFFICIAL_CLOSE: InstalledScheduleBinding(
        schedule_id=NEXT_OPEN_AFTER_OFFICIAL_CLOSE,
        execution_recipe_id=ONE_SESSION_RECIPE_ID,
        order_submission_policy_handle=PRE_OPEN_SUBMISSION_DEADLINE,
    ),
}
"""The strategy clocks this build installs. One today, and that is the honest count.

The contracts can express a close-auction, intraday or weekly schedule, and the
resolver below handles any binding whose recipe resolves. What this build has
*installed* is one, and calling the schema's reach "production support" would be
the fixture-versus-route confusion this table exists to end.
"""


def session_clocks_from_execution_rows(table: pa.Table) -> dict[date, dict[str, object]]:
    """Session open and close instants, taken from the execution owner's own rows.

    The execution snapshot already publishes, per formation, the formation close,
    the entry open and the holding-end open -- the very instants a schedule is
    made of, on the calendar the realised returns were measured against. Deriving
    them here rather than re-materialising a calendar removes the second source
    of truth: a locally regenerated calendar can only agree with the durable rows
    or silently disagree with them, and there is no version of that where the
    local copy is the authority.

    A session appears with only the events its rows actually witnessed. Nothing
    is interpolated, so an anchor pointing at an instant this snapshot never
    published is a typed refusal rather than a guess.
    """
    columns = {
        name: table[name].to_pylist()
        for name in (
            "formation_session",
            "formation_close_at",
            "entry_session",
            "entry_open_at",
            "holding_end_session",
            "holding_end_open_at",
        )
    }
    clocks: dict[date, dict[str, object]] = {}

    def place(session: date, field: str, value: object) -> None:
        seen = clocks.setdefault(session, {}).setdefault(field, value)
        if seen != value:
            # Two rows disagree about one exchange instant. Every listing on a
            # formation shares one schedule point, so this is a snapshot
            # assembled from two calendars rather than a data quirk.
            raise TemporalAdmissionError("causal_inputs.execution_session_clock_disagrees")

    for index in range(len(columns["formation_session"])):
        place(
            columns["formation_session"][index],
            "session_close_timestamp",
            columns["formation_close_at"][index],
        )
        place(
            columns["entry_session"][index],
            "session_open_timestamp",
            columns["entry_open_at"][index],
        )
        place(
            columns["holding_end_session"][index],
            "session_open_timestamp",
            columns["holding_end_open_at"][index],
        )
    return clocks


def entry_session_by_formation(table: pa.Table) -> dict[date, date]:
    """Which session each formation's orders actually filled in.

    The projection a re-mark cannot be correct without, taken from the execution
    snapshot rather than inferred from an array position. A book filled at
    formation ``T`` is valued at ``entry_session(T)``, and re-marking it to a
    close means the close *of that session* -- so anything indexed beside the
    formation axis is one day early on the installed next-open method, which is
    exactly the error this projection exists to make unstateable.
    """
    formations = table.column("formation_session").to_pylist()
    entries = table.column("entry_session").to_pylist()
    projection: dict[date, date] = {}
    for formation, entry in zip(formations, entries, strict=True):
        seen = projection.setdefault(formation, entry)
        if seen != entry:
            # Every listing on one formation shares one schedule point, so rows
            # that disagree are a snapshot assembled from two schedules.
            raise TemporalAdmissionError("causal_inputs.execution_entry_session_disagrees")
        if entry <= formation:
            raise TemporalAdmissionError("causal_inputs.execution_entry_not_after_formation")
    return projection


def _event_of(timing: str) -> SessionEvent:
    """Which published instant an execution recipe's timing names.

    Refused rather than defaulted. A recipe naming a venue event the calendar
    does not publish -- an auction imbalance, a pre-open indication -- has no
    timestamp to resolve to, and quietly treating it as the official open would
    move a whole decision by minutes without saying so.
    """
    for suffix in _EVENT_SUFFIXES:
        if timing.endswith(suffix):
            return suffix
    raise TemporalAdmissionError("causal_inputs.execution_timing_event_unresolved")


def resolve_order_submission_policy(
    *, handle: str, entry_offset_sessions: int, entry_event: SessionEvent
) -> AnchoredInstantPolicy:
    """The named, installed instant orders must be in by.

    Anchored to the entry instant the execution owner published, so an early
    close or a holiday moves it correctly, and offset by an interval somebody
    chose. ``basis`` is ``INSTALLED_OPERATIONAL_POLICY`` and the contract refuses
    a non-zero offset that claims to be an exchange fact, so a decided number
    cannot borrow the venue's authority.

    There is no default. An unknown or absent handle is a refusal, because the
    policy lands inside ``schedule_hash`` and therefore inside every result
    downstream of it: a caller who never chose it would still be publishing it.
    """
    installed = INSTALLED_ORDER_SUBMISSION_POLICIES.get(handle)
    if installed is None:
        raise TemporalAdmissionError("causal_inputs.order_submission_policy_not_installed")
    minutes_before, rationale = installed
    if minutes_before <= 0:
        # A deadline at or after the entry instant is not a deadline; it would
        # admit an input that became ready while the order was being filled.
        raise TemporalAdmissionError("causal_inputs.order_deadline_not_before_entry")
    return AnchoredInstantPolicy.create(
        policy_id=handle,
        basis="INSTALLED_OPERATIONAL_POLICY",
        anchor=SessionAnchor(offset_sessions=entry_offset_sessions, event=entry_event),
        minutes_after_anchor=-minutes_before,
        rationale=rationale,
    )


def resolve_installed_schedule_handle(recipe_id: str) -> str:
    """Which installed strategy trades this method, when exactly one does.

    Not a default. A recipe with no installed strategy is a refusal, and a recipe
    two strategies claim is a refusal too -- a campaign cannot be given the one
    that happens to sort first, because the deadline policy differs between them
    and lands in every downstream identity. When a second binding for one recipe
    is installed, the campaign request grows a handle field and this resolves it
    instead of inferring it.
    """
    matches = tuple(
        handle
        for handle, binding in INSTALLED_STRATEGY_SCHEDULES.items()
        if binding.execution_recipe_id == recipe_id
    )
    if not matches:
        raise TemporalAdmissionError("causal_inputs.strategy_schedule_not_installed_for_recipe")
    if len(matches) > 1:
        raise TemporalAdmissionError("causal_inputs.strategy_schedule_ambiguous_for_recipe")
    return matches[0]


def resolve_strategy_decision_schedule(
    *,
    schedule_handle: str,
    recipe: ExecutionOutcomeMethodRecipe,
    method_binding_hash: str | None = None,
) -> StrategyDecisionSchedule:
    """One installed strategy clock, checked against the method it claims to trade.

    Every offset and every event comes from the recipe. Nothing is counted in
    calendar days, and the one-session and five-session methods both derive
    correctly because neither offset is written here.

    The Target window is the execution window by construction rather than by
    agreement -- ``target_start``/``target_end`` are the same objects as
    ``entry``/``exit``, so a Target measuring a different interval cannot be
    described by a schedule at all.
    """
    binding = INSTALLED_STRATEGY_SCHEDULES.get(schedule_handle)
    if binding is None:
        raise TemporalAdmissionError("causal_inputs.strategy_schedule_not_installed")
    if binding.execution_recipe_id != recipe.recipe_id:
        # The binding names the method it trades. Resolving it against a
        # different recipe would give a five-session method the next-open
        # strategy's identity, which is exactly what a global default did.
        raise TemporalAdmissionError("causal_inputs.strategy_schedule_recipe_mismatch")
    if recipe.information_cutoff != "FORMATION_OFFICIAL_CLOSE":
        raise TemporalAdmissionError("causal_inputs.execution_information_cutoff_unresolved")
    entry_event = _event_of(recipe.entry_timing)
    exit_event = _event_of(recipe.exit_timing)
    entry = AnchoredInstantPolicy.exchange_event(
        offset_sessions=recipe.entry_offset_sessions,
        event=entry_event,
        policy_id=recipe.entry_timing,
    )
    exit_at = AnchoredInstantPolicy.exchange_event(
        offset_sessions=recipe.exit_offset_sessions,
        event=exit_event,
        policy_id=recipe.exit_timing,
    )
    deadline = resolve_order_submission_policy(
        handle=binding.order_submission_policy_handle,
        entry_offset_sessions=recipe.entry_offset_sessions,
        entry_event=entry_event,
    )
    return StrategyDecisionSchedule.create(
        schedule_id=binding.schedule_id,
        information_cutoff=AnchoredInstantPolicy.exchange_event(
            offset_sessions=0, event="OFFICIAL_CLOSE", policy_id=recipe.information_cutoff
        ),
        order_submission_deadline=deadline,
        entry=entry,
        exit=exit_at,
        target_start=entry,
        target_end=exit_at,
        execution_recipe_id=recipe.recipe_id,
        execution_recipe_hash=recipe.recipe_hash,
        method_binding_hash=method_binding_hash,
    )


__all__ = [
    "INSTALLED_ORDER_SUBMISSION_POLICIES",
    "INSTALLED_STRATEGY_SCHEDULES",
    "NEXT_OPEN_AFTER_OFFICIAL_CLOSE",
    "PRE_OPEN_SUBMISSION_DEADLINE",
    "PRE_OPEN_SUBMISSION_MINUTES_BEFORE_ENTRY",
    "InstalledScheduleBinding",
    "entry_session_by_formation",
    "resolve_installed_schedule_handle",
    "resolve_order_submission_policy",
    "resolve_strategy_decision_schedule",
    "session_clocks_from_execution_rows",
]
