"""One clock owner: facts that do not move, and a route that actually consumes it.

Three requirement groups.

**Facts are strategy-neutral.** The same authority object -- one hash -- is
admitted or refused purely by which strategy asks. Nothing here edits a Formula
to make a strategy work, because that is the move the capability exists to make
unnecessary.

**An input's own chain is checked on real instants.** Session offsets cannot
decide it: an input observed at ``close(T)`` claiming availability at ``open(T)``
has both anchors on offset zero, and only the resolved timestamps say so.

**Refusals come from re-derivation, not from hash comparison.** Every forgery
below re-seals its own identity and each parent identity above it, so a check
that only compared hashes would pass all of them.
"""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime, timedelta

import pyarrow as pa
import pytest
from pydantic import ValidationError

from alphalattice.capabilities.causal_inputs.admission import (
    admit_causal_input,
    admit_decision_input_set,
    require_schedule_ordering,
)
from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    CausalInputAuthority,
    SessionAnchor,
    SessionEvent,
    StrategyDecisionSchedule,
    TemporalAdmissionError,
)
from alphalattice.capabilities.causal_inputs.schedules import (
    INSTALLED_ORDER_SUBMISSION_POLICIES,
    INSTALLED_STRATEGY_SCHEDULES,
    PRE_OPEN_SUBMISSION_DEADLINE,
    InstalledScheduleBinding,
    resolve_installed_schedule_handle,
    resolve_order_submission_policy,
    resolve_strategy_decision_schedule,
    session_clocks_from_execution_rows,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    FIVE_SESSION_RECIPE_ID,
    build_five_session_recipe,
    build_one_session_recipe,
)
from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    PANEL_SCORE_READINESS_POLICY,
    resolve_panel_score_readiness_policy,
)
from alphalattice.investment.risk_research.experiments.temporal import (
    RISK_READINESS_BUDGET,
    resolve_risk_readiness_policy,
    risk_covariance_authority,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

NEXT_OPEN = "NEXT_OPEN_AFTER_OFFICIAL_CLOSE"
_OPEN_HOUR, _CLOSE_HOUR = 13, 20


def _axis(count: int = 40) -> tuple[date, ...]:
    start = date(2024, 1, 2)
    return tuple(start + timedelta(days=index) for index in range(count))


AXIS = _axis()
FORMATIONS = AXIS[10:20]


def _clocks(
    axis: tuple[date, ...] = AXIS, *, open_hour: int = _OPEN_HOUR, close_hour: int = _CLOSE_HOUR
) -> dict[date, dict[str, object]]:
    return {
        session: {
            "session_open_timestamp": datetime(
                session.year, session.month, session.day, open_hour, 30, tzinfo=UTC
            ),
            "session_close_timestamp": datetime(
                session.year, session.month, session.day, close_hour, 0, tzinfo=UTC
            ),
        }
        for session in axis
    }


CLOCKS = _clocks()


def _fact(offset: int, event: SessionEvent, policy_id: str = "FIXTURE") -> AnchoredInstantPolicy:
    return AnchoredInstantPolicy.exchange_event(
        offset_sessions=offset, event=event, policy_id=policy_id
    )


def _authority(**overrides: object) -> CausalInputAuthority:
    values: dict[str, object] = {
        "input_id": "feature_panel",
        "input_kind": "BASE_FEATURE_PANEL",
        "temporal_usage": "DECISION_INPUT",
        "surface_hash": "a" * 64,
        "owner_authority_id": "feature_engine.catalog:daily-provider-session-close",
        "owner_identity_hash": "b" * 64,
        "observed_through": _fact(0, "OFFICIAL_CLOSE"),
        "source_available": _fact(0, "OFFICIAL_CLOSE"),
        "observation_semantics": "Feature values for session T at its official close.",
        "point_in_time_disposition": "REVISABLE_BACKWARD_PROJECTION",
    }
    values.update(overrides)
    return CausalInputAuthority.create(**values)


def _overnight(minutes: int = 240) -> AnchoredInstantPolicy:
    return AnchoredInstantPolicy.create(
        policy_id="DYNAMIC_LIGHTGBM_OVERNIGHT_FIT",
        basis="MEASURED_RUNTIME",
        anchor=SessionAnchor(offset_sessions=0, event="OFFICIAL_CLOSE"),
        minutes_after_anchor=minutes,
        rationale="measured five-fold fit wall time, rounded up to whole minutes",
    )


def _installed_schedule() -> StrategyDecisionSchedule:
    return resolve_strategy_decision_schedule(
        schedule_handle=NEXT_OPEN, recipe=build_one_session_recipe()
    )


def _admit(
    authority: CausalInputAuthority,
    schedule: StrategyDecisionSchedule | None = None,
    *,
    clocks: dict[date, dict[str, object]] | None = None,
):
    return admit_causal_input(
        authority=authority,
        schedule=schedule or _installed_schedule(),
        formation_sessions=FORMATIONS,
        ordered_axis=AXIS,
        session_clocks=clocks or CLOCKS,
    )


# --------------------------------------------------------------------------
# The input's own chain, on real instants
# --------------------------------------------------------------------------


def test_availability_before_observation_is_refused_though_offsets_agree() -> None:
    """Availability before observation is refused though offsets agree."""

    forged = _authority(source_available=_fact(0, "OFFICIAL_OPEN"))
    assert (
        forged.source_available.anchor.offset_sessions
        == forged.observed_through.anchor.offset_sessions
    )
    result = _admit(forged)
    assert result.disposition == "REFUSED"
    assert result.refused_relation == "SOURCE_AVAILABLE_BEFORE_OBSERVATION"
    assert result.refused_formation_count == len(FORMATIONS)


def test_readiness_before_its_own_source_is_refused() -> None:
    early = _authority(
        derived_ready=AnchoredInstantPolicy.create(
            policy_id="FINISHED_BEFORE_ITS_INPUTS",
            basis="MEASURED_RUNTIME",
            anchor=SessionAnchor(offset_sessions=0, event="OFFICIAL_OPEN"),
            minutes_after_anchor=0,
            rationale="a computation that finished before the data it reads",
        )
    )
    assert _admit(early).refused_relation == "DERIVED_READY_BEFORE_SOURCE_AVAILABLE"


def test_a_post_close_source_availability_is_expressible_and_admitted() -> None:
    """A post close source availability is expressible and admitted."""

    late = _authority(
        source_available=AnchoredInstantPolicy.create(
            policy_id="VENDOR_POST_CLOSE_DELIVERY",
            basis="INSTALLED_SOURCE_AVAILABILITY_POLICY",
            anchor=SessionAnchor(offset_sessions=0, event="OFFICIAL_CLOSE"),
            minutes_after_anchor=90,
            rationale="the provider asserts delivery ninety minutes after the close",
        )
    )
    result = _admit(late)
    assert result.disposition == "ADMITTED"
    assert result.observation_staleness_at_decision_minutes == 0.0
    assert result.ready_to_submission_deadline_minutes == pytest.approx(1020.0 - 90.0)


def test_the_two_reported_minutes_are_named_for_where_they_start() -> None:
    """The two reported minutes are named for where they start."""

    risk = _authority(
        input_id="risk",
        input_kind="RISK_COVARIANCE",
        observation_start=_fact(-1, "OFFICIAL_OPEN"),
        observed_through=_fact(0, "OFFICIAL_OPEN"),
        source_available=_fact(0, "OFFICIAL_CLOSE"),
        derived_ready=AnchoredInstantPolicy.create(
            policy_id="ONE_MINUTE",
            basis="INSTALLED_OPERATIONAL_POLICY",
            anchor=SessionAnchor(offset_sessions=0, event="OFFICIAL_CLOSE"),
            minutes_after_anchor=1,
            rationale="budget",
        ),
    )
    result = _admit(risk)
    assert result.observation_staleness_at_decision_minutes == 390.0
    assert result.ready_to_submission_deadline_minutes == 1019.0
    observed, deadline = result.observed_through_at, result.order_submission_deadline_at
    assert observed is not None and deadline is not None
    assert (deadline - observed).total_seconds() / 60.0 == 1410.0


# --------------------------------------------------------------------------
# Installed policy, never an implicit default
# --------------------------------------------------------------------------


def test_the_submission_deadline_must_be_named_and_installed() -> None:
    with pytest.raises(TemporalAdmissionError, match="order_submission_policy_not_installed"):
        resolve_order_submission_policy(
            handle="A_DEADLINE_NOBODY_INSTALLED",
            entry_offset_sessions=1,
            entry_event="OFFICIAL_OPEN",
        )
    installed = resolve_order_submission_policy(
        handle=PRE_OPEN_SUBMISSION_DEADLINE, entry_offset_sessions=1, entry_event="OFFICIAL_OPEN"
    )
    assert installed.basis == "INSTALLED_OPERATIONAL_POLICY"
    assert installed.minutes_after_anchor < 0


def test_no_caller_receives_thirty_minutes_without_choosing_it() -> None:
    """The production resolver has no deadline default left to inherit."""

    assert (
        inspect.signature(resolve_order_submission_policy).parameters["handle"].default
        is inspect.Parameter.empty
    )
    assert (
        inspect.signature(resolve_strategy_decision_schedule).parameters["schedule_handle"].default
        is inspect.Parameter.empty
    )
    assert _installed_schedule().order_submission_deadline.policy_id == PRE_OPEN_SUBMISSION_DEADLINE


def test_a_schedule_must_be_installed_and_must_match_its_recipe() -> None:
    with pytest.raises(TemporalAdmissionError, match="strategy_schedule_not_installed"):
        resolve_strategy_decision_schedule(
            schedule_handle="A_STRATEGY_NOBODY_INSTALLED", recipe=build_one_session_recipe()
        )
    # The binding names the method it trades; resolving it against another recipe
    # would give a five-session method the next-open strategy's identity.
    with pytest.raises(TemporalAdmissionError, match="strategy_schedule_recipe_mismatch"):
        resolve_strategy_decision_schedule(
            schedule_handle=NEXT_OPEN, recipe=build_five_session_recipe()
        )
    with pytest.raises(TemporalAdmissionError, match="not_installed_for_recipe"):
        resolve_installed_schedule_handle(FIVE_SESSION_RECIPE_ID)


def test_the_same_resolver_handles_a_second_installed_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same resolver handles a second installed binding."""

    monkeypatch.setitem(
        INSTALLED_ORDER_SUBMISSION_POLICIES,
        "DEVELOPMENT_HOUR_BEFORE_OPEN_SUBMISSION",
        (60, "an installed development policy that closes the book an hour earlier"),
    )
    monkeypatch.setitem(
        INSTALLED_STRATEGY_SCHEDULES,
        "WEEKLY_NEXT_OPEN_AFTER_OFFICIAL_CLOSE",
        InstalledScheduleBinding(
            schedule_id="WEEKLY_NEXT_OPEN_AFTER_OFFICIAL_CLOSE",
            execution_recipe_id=FIVE_SESSION_RECIPE_ID,
            order_submission_policy_handle="DEVELOPMENT_HOUR_BEFORE_OPEN_SUBMISSION",
        ),
    )
    weekly = resolve_strategy_decision_schedule(
        schedule_handle=resolve_installed_schedule_handle(FIVE_SESSION_RECIPE_ID),
        recipe=build_five_session_recipe(),
    )
    assert (weekly.entry.anchor.offset_sessions, weekly.exit.anchor.offset_sessions) == (1, 6)
    assert weekly.order_submission_deadline.minutes_after_anchor == -60
    assert weekly.schedule_hash != _installed_schedule().schedule_hash

    panel = _authority()
    both = {
        schedule.schedule_id: _admit(panel, schedule)
        for schedule in (_installed_schedule(), weekly)
    }
    assert {item.disposition for item in both.values()} == {"ADMITTED"}
    # One fact, two consumption bindings.
    assert {item.input_authority_hash for item in both.values()} == {panel.authority_hash}
    assert len({item.strategy_schedule_hash for item in both.values()}) == 2


def test_switching_the_deadline_policy_moves_no_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _authority()
    before = _admit(panel)
    monkeypatch.setitem(
        INSTALLED_ORDER_SUBMISSION_POLICIES, PRE_OPEN_SUBMISSION_DEADLINE, (90, "an hour earlier")
    )
    after = _admit(panel)
    assert before.input_authority_hash == after.input_authority_hash == panel.authority_hash
    assert before.strategy_schedule_hash != after.strategy_schedule_hash
    assert after.ready_to_submission_deadline_minutes == pytest.approx(
        (before.ready_to_submission_deadline_minutes or 0.0) - 60.0
    )


# --------------------------------------------------------------------------
# The execution owner is the only session-clock authority
# --------------------------------------------------------------------------


def _execution_table(formations: tuple[date, ...]) -> pa.Table:
    return pa.table(
        {
            "formation_session": list(formations),
            "formation_close_at": [
                CLOCKS[value]["session_close_timestamp"] for value in formations
            ],
            "entry_session": [AXIS[AXIS.index(value) + 1] for value in formations],
            "entry_open_at": [
                CLOCKS[AXIS[AXIS.index(value) + 1]]["session_open_timestamp"]
                for value in formations
            ],
            "holding_end_session": [AXIS[AXIS.index(value) + 2] for value in formations],
            "holding_end_open_at": [
                CLOCKS[AXIS[AXIS.index(value) + 2]]["session_open_timestamp"]
                for value in formations
            ],
        }
    )


def test_session_clocks_come_from_the_durable_execution_rows() -> None:
    """No second calendar. The instants are the snapshot's own."""

    formations = AXIS[5:9]
    derived = session_clocks_from_execution_rows(_execution_table(formations))
    assert (
        derived[formations[0]]["session_close_timestamp"]
        == CLOCKS[formations[0]]["session_close_timestamp"]
    )
    assert derived[AXIS[6]]["session_open_timestamp"] == CLOCKS[AXIS[6]]["session_open_timestamp"]
    # Nothing is interpolated: a session the rows never witnessed has no entry,
    # and an anchor pointing at it is a refusal rather than a guess.
    assert AXIS[0] not in derived


def test_a_snapshot_whose_rows_disagree_about_one_instant_is_refused() -> None:
    session = AXIS[5]
    close = CLOCKS[session]["session_close_timestamp"]
    assert isinstance(close, datetime)
    table = pa.table(
        {
            "formation_session": [session, session],
            "formation_close_at": [close, close + timedelta(minutes=30)],
            "entry_session": [AXIS[6], AXIS[6]],
            "entry_open_at": [CLOCKS[AXIS[6]]["session_open_timestamp"]] * 2,
            "holding_end_session": [AXIS[7], AXIS[7]],
            "holding_end_open_at": [CLOCKS[AXIS[7]]["session_open_timestamp"]] * 2,
        }
    )
    with pytest.raises(TemporalAdmissionError, match="execution_session_clock_disagrees"):
        session_clocks_from_execution_rows(table)


def test_nothing_hardcodes_a_utc_wall_clock() -> None:
    """Nothing hardcodes a UTC wall clock."""

    winter = _clocks(open_hour=14, close_hour=21)
    summer_result, winter_result = _admit(_authority()), _admit(_authority(), clocks=winter)
    assert summer_result.disposition == winter_result.disposition == "ADMITTED"
    assert winter_result.information_cutoff_at is not None
    assert winter_result.information_cutoff_at.hour == 21
    assert (
        winter_result.ready_to_submission_deadline_minutes
        == summer_result.ready_to_submission_deadline_minutes
    )
    # A clock the owner never published is refused rather than interpolated.
    holed = {key: value for key, value in CLOCKS.items() if key != AXIS[12]}
    with pytest.raises(TemporalAdmissionError, match="session_clock_absent"):
        _admit(_authority(), clocks=holed)


# --------------------------------------------------------------------------
# Risk readiness is a budget, not a measurement
# --------------------------------------------------------------------------


def test_risk_readiness_is_an_installed_budget_and_must_be_named() -> None:
    """No runtime receipt exists in this build, so nothing claims MEASURED_RUNTIME."""

    policy = resolve_risk_readiness_policy(RISK_READINESS_BUDGET)
    assert policy.basis == "INSTALLED_OPERATIONAL_POLICY"
    assert "not a measurement" in policy.rationale
    with pytest.raises(TemporalAdmissionError, match="readiness_budget_not_installed"):
        resolve_risk_readiness_policy("A_BUDGET_NOBODY_INSTALLED")
    assert (
        inspect.signature(risk_covariance_authority).parameters["readiness_handle"].default
        is inspect.Parameter.empty
    )


def test_alpha_readiness_budget_clears_the_installed_next_open_deadline() -> None:
    """The Alpha-owned budget is compared by the shared calendar admission owner."""

    score = _authority(
        input_id="dynamic_score",
        input_kind="RAW_SCORE",
        derived_ready=resolve_panel_score_readiness_policy(PANEL_SCORE_READINESS_POLICY),
    )
    result = _admit(score)
    assert result.disposition == "ADMITTED"
    assert result.ready_to_submission_deadline_minutes == pytest.approx(1020.0 - 60.0)


# --------------------------------------------------------------------------
# Adversarial: re-sealed forgeries
# --------------------------------------------------------------------------


def test_a_score_shifted_one_session_and_fully_re_sealed_is_refused() -> None:
    forged = _authority(
        input_id="score",
        input_kind="RAW_SCORE",
        observed_through=_fact(1, "OFFICIAL_CLOSE"),
        source_available=_fact(1, "OFFICIAL_CLOSE"),
    )
    assert forged.authority_hash == canonical_hash(
        forged.model_dump(mode="json", exclude={"authority_hash"})
    )
    assert _admit(forged).refused_relation == "OBSERVED_THROUGH_AFTER_INFORMATION_CUTOFF"


def test_a_substituted_surface_hash_needs_the_producer_owner_to_refuse_it() -> None:
    """A substituted surface hash needs the producer owner to refuse it."""

    honest = _authority(input_id="score", input_kind="RAW_SCORE")
    swapped = _authority(input_id="score", input_kind="RAW_SCORE", surface_hash="c" * 64)
    assert _admit(honest).disposition == _admit(swapped).disposition == "ADMITTED"
    assert honest.authority_hash != swapped.authority_hash
    # Same owner handle, different surface: the substitution is visible to a
    # verifier that reopens the owner, and invisible to one that only rehashes.
    assert honest.owner_identity_hash == swapped.owner_identity_hash
    assert honest.surface_hash != swapped.surface_hash


def test_availability_or_readiness_after_the_deadline_is_refused() -> None:
    late_source = _authority(source_available=_fact(1, "OFFICIAL_CLOSE"))
    assert _admit(late_source).refused_relation == "SOURCE_AVAILABLE_AFTER_ORDER_DEADLINE"
    slow = _authority(input_id="score", input_kind="RAW_SCORE", derived_ready=_overnight(1100))
    assert _admit(slow).refused_relation == "DERIVED_READY_AFTER_ORDER_DEADLINE"


def test_an_overnight_score_is_admitted_without_touching_its_observation() -> None:
    """The relation that did not exist before, and the workaround it retires."""

    score = _authority(input_id="dynamic_score", input_kind="RAW_SCORE", derived_ready=_overnight())
    result = _admit(score)
    assert result.disposition == "ADMITTED"
    assert result.observation_staleness_at_decision_minutes == 0.0
    assert result.ready_to_submission_deadline_minutes == pytest.approx(1020.0 - 240.0)
    assert score.observed_through.anchor.offset_sessions == 0


def _target(**overrides: object) -> CausalInputAuthority:
    values: dict[str, object] = {
        "input_id": "target",
        "temporal_usage": "TARGET_OUTCOME",
        "input_kind": "TARGET_OUTCOME",
        "observation_start": _fact(1, "OFFICIAL_OPEN"),
        "observed_through": _fact(2, "OFFICIAL_OPEN"),
        "source_available": _fact(2, "OFFICIAL_OPEN"),
    }
    values.update(overrides)
    return _authority(**values)


def test_a_target_is_refused_from_a_decision_input_set_by_usage() -> None:
    """Refused before an instant is resolved, so the reason reads as the leak."""

    with pytest.raises(TemporalAdmissionError, match="target_is_not_a_decision_input"):
        admit_decision_input_set(
            authorities=[_authority(), _target()],
            schedule=_installed_schedule(),
            formation_sessions=FORMATIONS,
            ordered_axis=AXIS,
            session_clocks=CLOCKS,
        )
    with pytest.raises(TemporalAdmissionError, match="target_is_not_a_decision_input"):
        _admit(_target())


def test_a_formation_outside_the_axis_cannot_be_padded_in() -> None:
    with pytest.raises(TemporalAdmissionError, match="formation_off_session_axis"):
        admit_causal_input(
            authority=_authority(),
            schedule=_installed_schedule(),
            formation_sessions=(*FORMATIONS, date(2019, 1, 1)),
            ordered_axis=AXIS,
            session_clocks=CLOCKS,
        )


def test_a_naive_timestamp_is_refused_rather_than_assumed_utc() -> None:
    naive = {
        session: {
            "session_open_timestamp": datetime(
                session.year, session.month, session.day, _OPEN_HOUR, 30
            ),
            "session_close_timestamp": datetime(
                session.year, session.month, session.day, _CLOSE_HOUR, 0
            ),
        }
        for session in AXIS
    }
    with pytest.raises(TemporalAdmissionError, match="not_timezone_aware"):
        _admit(_authority(), clocks=naive)


def test_a_deadline_after_its_own_entry_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        INSTALLED_ORDER_SUBMISSION_POLICIES,
        PRE_OPEN_SUBMISSION_DEADLINE,
        (-15, "a deadline that falls after the trade it governs"),
    )
    with pytest.raises(TemporalAdmissionError, match="order_deadline_not_before_entry"):
        _installed_schedule()


def test_schedule_ordering_is_checked_on_instants_per_formation() -> None:
    schedule = _installed_schedule()
    require_schedule_ordering(
        schedule=schedule,
        formation_sessions=FORMATIONS,
        ordered_axis=AXIS,
        session_clocks=CLOCKS,
    )
    late = StrategyDecisionSchedule.create(
        schedule_id=schedule.schedule_id,
        information_cutoff=schedule.information_cutoff,
        order_submission_deadline=AnchoredInstantPolicy.create(
            policy_id="AFTER_THE_OPEN",
            basis="INSTALLED_OPERATIONAL_POLICY",
            anchor=SessionAnchor(offset_sessions=1, event="OFFICIAL_OPEN"),
            minutes_after_anchor=15,
            rationale="a deadline that falls after the trade it governs",
        ),
        entry=schedule.entry,
        exit=schedule.exit,
        target_start=schedule.target_start,
        target_end=schedule.target_end,
    )
    with pytest.raises(TemporalAdmissionError, match="schedule_entry_precedes_deadline"):
        require_schedule_ordering(
            schedule=late,
            formation_sessions=FORMATIONS,
            ordered_axis=AXIS,
            session_clocks=CLOCKS,
        )


def test_the_authority_contract_cannot_carry_a_strategy_field() -> None:
    forbidden = {"entry", "exit", "entry_at", "entry_offset_sessions", "strategy_schedule_hash"}
    assert not forbidden & set(CausalInputAuthority.model_fields)
    with pytest.raises(TemporalAdmissionError, match="contract_field_unknown"):
        _authority(strategy_schedule_hash="c" * 64)


def test_an_offset_instant_may_not_claim_to_be_a_published_exchange_fact() -> None:
    with pytest.raises(ValidationError, match="anchored_instant_basis_inconsistent"):
        AnchoredInstantPolicy.create(
            policy_id="PRETENDS_TO_BE_THE_OPEN",
            basis="EXCHANGE_PUBLISHED_FACT",
            anchor=SessionAnchor(offset_sessions=1, event="OFFICIAL_OPEN"),
            minutes_after_anchor=-30,
            rationale="a decided cutoff wearing the calendar's authority",
        )
