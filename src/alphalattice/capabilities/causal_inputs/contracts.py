"""What an input says about its own time, and what a strategy asks of it.

**Facts report time; strategies verify time.** A fact never accommodates a
strategy and a strategy never rewrites a fact. That single rule decides the
shape of everything here, and in particular it decides what is *absent*:
``CausalInputAuthority`` has no entry, no exit, no rebalance clock and no order
deadline, because raw market data, corporate actions, Feature Formulas, Feature
values, the Base Panel, the Development Overlay and a general return surface are
strategy-neutral authorities. One Panel is consumable by many strategies, so a
strategy schedule may not enter its identity.

The schedule enters through a separate *consumption* binding instead::

    fact artifact identity  ---+
                               +--> AdmittedInformationSet
    strategy schedule identity -+

Switching strategy leaves the Panel hash and every Feature hash exactly where
they were, may change the admission result, and rotates only the strategy-bound
Programs downstream of it.

**Why this owner exists at all.** Three domains had already discovered parts of
the same clock. The Feature owner holds ``FeatureObservationClock`` and
``FeatureAvailabilityPolicy``; the execution owner holds an information cutoff
and real entry/exit timestamps; ``portfolio_inputs.signed_score`` joined the two
and checked ``available <= decision < entry < exit``. Risk held nothing at all --
a published covariance said nothing about when its information was knowable.

That three-way split left one relation unexpressible, and its absence was doing
real damage. A strategy that decides at ``close(T)`` and trades at ``open(T+1)``
has the whole night to compute, so a score that takes hours is perfectly
tradable. With only ``available <= decision`` to say it with, a producer whose
value is not finished *at* the close could only buy admissibility by moving its
own observation back a session -- the ``lag_sessions=1`` workaround, which is a
Formula edit made to satisfy a trading clock. The missing term is the order
submission deadline, and separating it from the information cutoff is what lets
a slow producer stay honest about the session it observed.

``lag`` accordingly keeps exactly one lawful meaning here: economic or transform
semantics owned by the Formula. A strategy that cannot use ``Feature(T)`` yields
``TEMPORAL_ADMISSION_REFUSED``, never a modified ``Feature(T)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"

SessionEvent = Literal["OFFICIAL_OPEN", "OFFICIAL_CLOSE"]
"""The two intra-session instants the market calendar actually publishes.

Deliberately not ``MarketPhase``. That enum is the Feature owner's *availability*
vocabulary -- it answers after which phase a value becomes selectable, and its
``PRE_OPEN``/``INTRADAY``/``POST_CLOSE`` members have no published timestamp to
resolve to. This answers a different question: which exchange instant was the
last thing an input saw. Both members here map onto a real column
(``session_open_timestamp``, ``session_close_timestamp``), so every anchor below
resolves to a timezone-aware instant rather than to a label a comparison would
then have to interpret.
"""

TemporalUsage = Literal["DECISION_INPUT", "TARGET_OUTCOME"]
"""The only distinction admission actually acts on.

This began as a closed enum of every input a pipeline might have -- panel,
overlay, score, filtered score, return surface, covariance, universe, sector,
holdings, benchmark, target. Every new input rotated a public schema, and all
eleven members were being consulted for one bit: may this be read at decision
time, or is it a future outcome? That bit is here; the descriptive name lives in
``input_kind``, which is free text because no admission rule reads it.

A Target is not a decision input, and ``admit_decision_input_set`` refuses one
structurally rather than by convention -- see the note on its own check.
"""


class TemporalAdmissionError(ValueError):
    """Stable fail-closed boundary for every temporal authority and admission."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def _seal[ContractT: _Contract](
    model: type[ContractT], field: str, /, **values: object
) -> ContractT:
    """Construct one content-addressed contract, sealing ``field`` over the rest.

    One helper for all six contracts below rather than six ``create``
    classmethods. The domains this capability serves each grew their own copy of
    that method; there is no reason for a seventh set inside one module.
    """
    unknown = sorted(set(values) - set(model.model_fields))
    if unknown:
        # `model_construct` does not validate, so an unknown key would be dropped
        # in silence and the seal would cover a field the caller thinks it set.
        # A producer smuggling a strategy field into a fact must fail loudly.
        raise TemporalAdmissionError(f"causal_inputs.contract_field_unknown:{unknown[0]}")
    draft = model.model_construct(**values, **{field: "0" * 64})
    identity = draft.model_dump(mode="json", exclude={field})
    payload = draft.model_dump(exclude={field})
    return cast(ContractT, model.model_validate({**payload, field: canonical_hash(identity)}))


class SessionAnchor(_Contract):
    """One instant, named as an offset onto the ordered exchange-session axis.

    ``offset_sessions`` is signed and counted from the formation session:
    negative is history, ``0`` is the formation itself, positive is the future
    an execution reaches into. Indexing the session axis rather than the
    calendar is the whole point -- a holiday, a half session or a DST boundary
    moves the real instant without moving the offset, and calendar arithmetic
    would silently disagree with the exchange on all three.
    """

    offset_sessions: int = Field(ge=-4096, le=4096)
    event: SessionEvent


class AnchoredInstantPolicy(_Contract):
    """An instant that is *not* an exchange fact, and says so.

    Two things need this shape and neither is published by any exchange: when a
    computed input finishes, and when orders must be submitted. Both are an
    offset in minutes from a real session anchor, and both are worthless without
    knowing which kind of claim they are -- hence ``basis``, which cannot be
    defaulted.

    ``INSTALLED_OPERATIONAL_POLICY`` means somebody decided it.
    ``MEASURED_RUNTIME`` means a run was timed. The difference decides what a
    result may claim, so it is a field rather than a comment.
    """

    policy_id: str = Field(min_length=1, max_length=96)
    basis: Literal[
        "EXCHANGE_PUBLISHED_FACT",
        "INSTALLED_SOURCE_AVAILABILITY_POLICY",
        "MEASURED_RUNTIME",
        "INSTALLED_OPERATIONAL_POLICY",
    ]
    """Four different kinds of claim, and none may impersonate another.

    An exchange event is published by the venue. A source availability policy is
    what a provider asserts about its own publication schedule -- twenty minutes
    after the close, post-close, T+1 morning -- and is an assertion, not an
    observation. A measured runtime is a timed computation. An operational
    deadline is a decision somebody made about their own process. Collapsing any
    two of these is how an assumed number acquires an observed one's authority.
    """
    anchor: SessionAnchor
    minutes_after_anchor: int = Field(ge=-1440, le=1440)
    """Signed. Negative is before the anchor, which is what a deadline is."""

    rationale: str = Field(min_length=1, max_length=400)
    policy_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal an instant policy from its declared values.

        Args:
            values: Policy fields excluding the computed hash.

        Returns:
            The policy with its canonical identity.
        """
        return _seal(cls, "policy_hash", **values)

    @classmethod
    def exchange_event(cls, *, offset_sessions: int, event: SessionEvent, policy_id: str) -> Self:
        """An instant that *is* a published exchange event, with no offset.

        Every schedule instant is one of these or a decided offset from one, so
        an official open and a broker cutoff are the same type and a reader can
        see which is which from ``basis`` instead of from where it is stored.
        """
        return cls.create(
            policy_id=policy_id,
            basis="EXCHANGE_PUBLISHED_FACT",
            anchor=SessionAnchor(offset_sessions=offset_sessions, event=event),
            minutes_after_anchor=0,
            rationale="the exchange publishes this instant on its own session calendar",
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Reject a false exchange fact or a changed policy identity.

        Returns:
            This validated policy.

        Raises:
            TemporalAdmissionError: If the basis or canonical hash disagrees.
        """
        if self.basis == "EXCHANGE_PUBLISHED_FACT" and self.minutes_after_anchor != 0:
            # An exchange fact is the anchor. Offsetting one and still calling it
            # published is how an assumed deadline acquires a fact's authority.
            raise TemporalAdmissionError("causal_inputs.anchored_instant_basis_inconsistent")
        if self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise TemporalAdmissionError("causal_inputs.anchored_instant_identity_invalid")
        return self


class CausalInputAuthority(_Contract):
    """One input's own clock, stated by the owner that produces it.

    **Strategy-neutral by construction.** There is no entry, no exit, no
    deadline and no rebalance field, and ``extra="forbid"`` means a producer
    cannot add one. A producer states what it observed and when that became
    usable; whether some strategy can use it is not its question.

    Stated as *rules* rather than as per-formation arrays. A study of 1,260
    formations across ten inputs would otherwise persist 12,600 timestamps that
    the session axis already determines, and every one of them would be a copy
    that could disagree with its owner. The anchors below resolve against the
    ordered axis at admission time and are not stored.
    """

    kind: Literal["CausalInputAuthority"] = "CausalInputAuthority"
    input_id: str = Field(min_length=1, max_length=160)
    input_kind: str = Field(min_length=1, max_length=96)
    """What this input is, for a reader. No admission rule branches on it."""

    temporal_usage: TemporalUsage

    surface_hash: str = Field(pattern=_HASH)
    """The fact artifact this clock describes. Unchanged by any strategy."""

    owner_authority_id: str = Field(min_length=1, max_length=160)
    owner_identity_hash: str = Field(pattern=_HASH)
    """The handle a verifier reopens to re-derive this clock at its owner.

    Without it an authority is self-asserted: a forged clock that re-seals its
    own hash verifies perfectly against itself. Every adversarial case this
    capability is tested against turns on re-derivation, not on hash equality.
    """

    observation_start: AnchoredInstantPolicy | None = None
    """Where a windowed observation begins, for an input that measures an interval.

    ``None`` for a point observation such as a Feature value or a covariance
    matrix. A return surface has one: its row ``T`` spans ``open(T-1)`` to
    ``open(T)``, and declaring both ends is what lets a consumer ask whether that
    interval is the one its strategy holds -- a question the end alone cannot
    answer, and the one Risk had no way to state at all.
    """

    observed_through: AnchoredInstantPolicy
    """The newest instant any value in this surface reflects.

    Not the session it is labelled with. A return surface whose row ``T`` is
    ``open(T-1) -> open(T)`` observes through ``OFFICIAL_OPEN`` at offset ``0``,
    and reporting ``close(T)`` because the row says ``T`` is the exact error this
    field exists to make unstateable.
    """

    source_available: AnchoredInstantPolicy
    """When the upstream source publishes what ``observed_through`` names.

    An instant policy rather than a bare exchange event, because availability is
    the field most likely to be something the exchange never publishes: a vendor
    that lands its file twenty minutes after the close, a post-close settlement
    feed, a source that only arrives the next morning. Forcing those into
    ``OFFICIAL_CLOSE`` would either overstate availability or buy safety with a
    whole session of lag -- the two failures this capability exists to prevent.
    """

    derived_ready: AnchoredInstantPolicy | None = None
    """When computation over the source finishes, for an input that is computed.

    ``None`` for a surface that is merely read. A fitted score, a filtered score
    and a covariance are all computed, and each is ready some real interval after
    its inputs land -- overnight, in this strategy's case, which is time the
    strategy genuinely has.
    """

    observation_semantics: str = Field(min_length=1, max_length=400)
    """What the owner says its window is, in the owner's own words."""

    point_in_time_disposition: Literal[
        "POINT_IN_TIME",
        "CURRENT_MEMBERSHIP_BACKFILLED",
        "REVISABLE_BACKWARD_PROJECTION",
        "UNVERIFIED",
    ]
    """Whether the values would have looked like this at the time.

    Distinct from causality on purpose. A current-membership universe is
    perfectly causal per session and still cannot have been known, and a
    surface that passes every timestamp comparison while carrying survivors is
    the failure this field refuses to leave implicit.
    """

    authority_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal a strategy-neutral input authority.

        Args:
            values: The producer's clock and source claims.

        Returns:
            The authority with its canonical identity.
        """
        return _seal(cls, "authority_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the authority hash without inferring temporal ordering.

        Returns:
            This validated authority.

        Raises:
            TemporalAdmissionError: If its fields no longer match the hash.
        """
        # Ordering is *not* checked here. Offsets cannot decide it: an input
        # observed at close(T) and claiming availability at open(T) has both
        # anchors on offset zero and is plainly impossible, and only the resolved
        # instants say so. `admit_causal_input` compares them per formation.
        if self.authority_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"authority_hash"})
        ):
            raise TemporalAdmissionError("causal_inputs.authority_identity_invalid")
        return self


class StrategyDecisionSchedule(_Contract):
    """One strategy's clock: what it may know, when it must act, what it holds.

    The rule, not the resolved instants. Per-formation timestamps belong to the
    execution owner that publishes them, and a schedule that copied them would
    be a second place for them to be wrong.

    ``target_start``/``target_end`` are here rather than on a Target artifact
    because the holding window *is* the strategy. A Target whose window differs
    from the one the strategy trades measures a different question.
    """

    kind: Literal["StrategyDecisionSchedule"] = "StrategyDecisionSchedule"
    schedule_id: str = Field(min_length=1, max_length=96)
    """A real business time point, never a machine, agent, branch or stage name."""

    information_cutoff: AnchoredInstantPolicy
    order_submission_deadline: AnchoredInstantPolicy
    entry: AnchoredInstantPolicy
    exit: AnchoredInstantPolicy
    target_start: AnchoredInstantPolicy
    target_end: AnchoredInstantPolicy
    """All six are the same type, and that is what makes other strategies sayable.

    A next-open strategy's cutoff is an exchange fact; a pre-close strategy's is
    ``OFFICIAL_CLOSE`` less ten minutes; an intraday entry is ``OFFICIAL_OPEN``
    plus thirty. One type covers all of them, so switching strategy changes the
    values here and never the schema -- and a producer downstream is not asked to
    understand a second vocabulary for the same instant.
    """

    execution_recipe_id: str | None = Field(default=None, min_length=1, max_length=96)
    execution_recipe_hash: str | None = Field(default=None, pattern=_HASH)
    method_binding_hash: str | None = Field(default=None, pattern=_HASH)
    """The installed execution method this schedule was re-derived from.

    Absent only for a fixture. A schedule a Host resolved for real evidence
    carries the seal, so a verifier can reopen the method and rebuild it.
    """

    schedule_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal a strategy's declared decision schedule.

        Args:
            values: The schedule and its installed method binding.

        Returns:
            The schedule with its canonical identity.
        """
        return _seal(cls, "schedule_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_schedule(self) -> Self:
        """Verify target alignment, method fields, and schedule identity.

        Returns:
            This validated schedule.

        Raises:
            TemporalAdmissionError: If the declared window, method, or hash conflicts.
        """
        if (self.target_start, self.target_end) != (self.entry, self.exit):
            raise TemporalAdmissionError("causal_inputs.schedule_target_window_mismatch")
        # Ordering is deliberately *not* checked here. Offsets alone cannot
        # decide it once an instant may carry minutes: a close-auction strategy
        # cuts off ten minutes before the close it enters at, so both sit at
        # offset zero and only the resolved instants are ordered. The check
        # belongs where the calendar is, and `require_schedule_ordering` runs it
        # per formation before any input is compared.
        # The recipe names the method; the binding hash names one snapshot's seal
        # of it. A schedule derived from an installed recipe legitimately has the
        # first pair and no seal yet, so only the pair is required to agree.
        if (self.execution_recipe_id is None) != (self.execution_recipe_hash is None):
            raise TemporalAdmissionError("causal_inputs.schedule_method_fields_inconsistent")
        if self.method_binding_hash is not None and self.execution_recipe_id is None:
            raise TemporalAdmissionError("causal_inputs.schedule_method_seal_unattributed")
        if self.schedule_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"schedule_hash"})
        ):
            raise TemporalAdmissionError("causal_inputs.schedule_identity_invalid")
        return self


class TemporalAdmissionResult(_Contract):
    """What one strategy's clock concluded about one input. A Clock Matrix row.

    Carries the resolved extremes rather than a verdict alone. A researcher
    needs to see *how stale* an admitted input is -- an input that clears the
    cutoff by five hours and one that clears it by four minutes are both
    ``ADMITTED``, and only one of them is comfortable.

    Exception evidence is bounded and compact: the first offending formation and
    the counts, never a per-formation array. The relation that failed is named,
    because "refused" without a relation sends a reader back to the arithmetic.
    """

    kind: Literal["TemporalAdmissionResult"] = "TemporalAdmissionResult"
    input_id: str = Field(min_length=1, max_length=160)
    input_kind: str = Field(min_length=1, max_length=96)
    """What this input is, for a reader. No admission rule branches on it."""

    temporal_usage: TemporalUsage
    input_authority_hash: str = Field(pattern=_HASH)
    strategy_schedule_hash: str = Field(pattern=_HASH)

    disposition: Literal["ADMITTED", "REFUSED"]
    refused_relation: (
        Literal[
            "SOURCE_AVAILABLE_BEFORE_OBSERVATION",
            "DERIVED_READY_BEFORE_SOURCE_AVAILABLE",
            "OBSERVED_THROUGH_AFTER_INFORMATION_CUTOFF",
            "SOURCE_AVAILABLE_AFTER_ORDER_DEADLINE",
            "DERIVED_READY_AFTER_ORDER_DEADLINE",
            "ORDER_DEADLINE_AFTER_ENTRY",
            "ENTRY_NOT_AFTER_INFORMATION_CUTOFF",
            "EXIT_NOT_AFTER_ENTRY",
            "TARGET_WINDOW_MISMATCH",
            "TARGET_AVAILABLE_BEFORE_EXIT",
            "TARGET_IN_DECISION_INPUT_SET",
            "AXIS_UNRESOLVABLE",
        ]
        | None
    ) = None

    formation_count: int = Field(ge=0)
    refused_formation_count: int = Field(ge=0)
    first_refused_formation: date | None = None

    observed_through_at: datetime | None = None
    source_available_at: datetime | None = None
    derived_ready_at: datetime | None = None
    information_cutoff_at: datetime | None = None
    order_submission_deadline_at: datetime | None = None
    entry_at: datetime | None = None
    exit_at: datetime | None = None
    """The resolved instants on the *last* admitted formation, for the matrix.

    One formation's worth, not 1,260. The relations are checked on every
    formation; these are what a reader is shown.
    """

    observation_staleness_at_decision_minutes: float | None = None
    """``information_cutoff_at - observed_through_at``. How old the newest fact was.

    Named for the instant it starts from, because it was previously called
    "slack" and printed beside ``observed_through`` in a matrix where the other
    column started somewhere else entirely. Larger is staler; zero means the
    input saw the decision instant itself.
    """

    ready_to_submission_deadline_minutes: float | None = None
    """``order_submission_deadline_at - derived_ready_at``. The room actually left.

    Starts at *readiness*, not at observation. For a covariance ready one minute
    after the close this is 1,019 minutes, while the distance from its own
    ``open(T)`` observation to the same deadline is 1,410 -- two different
    quantities that one word made look like the same column.
    """

    point_in_time_disposition: Literal[
        "POINT_IN_TIME",
        "CURRENT_MEMBERSHIP_BACKFILLED",
        "REVISABLE_BACKWARD_PROJECTION",
        "UNVERIFIED",
    ]

    horizon_alignment: Literal[
        "MATCHED_TO_STRATEGY_WINDOW",
        "SAME_UNIT_DIFFERENT_WINDOW",
        "NOT_A_WINDOWED_OBSERVATION",
        "DIFFERENT_UNIT_AND_WINDOW",
    ]
    """Whether a windowed input measures the interval this strategy holds.

    Separate from causality, and reported even when admission passes. A return
    surface can be perfectly causal, land comfortably inside the deadline, and
    still describe a different session's move than the one the portfolio is
    exposed to -- which is a scientific mismatch, not a firewall breach, and is
    useless to a reader if the two are collapsed into one verdict.
    """

    result_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one input's temporal admission result.

        Args:
            values: The measured relations and disposition.

        Returns:
            The result with its canonical identity.
        """
        return _seal(cls, "result_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_result(self) -> Self:
        """Verify refusal counts and the result's canonical identity.

        Returns:
            This validated result.

        Raises:
            TemporalAdmissionError: If the disposition, counts, or hash conflict.
        """
        if (self.disposition == "REFUSED") != (self.refused_relation is not None):
            raise TemporalAdmissionError("causal_inputs.admission_result_disposition_inconsistent")
        if (self.disposition == "REFUSED") != (self.refused_formation_count > 0):
            raise TemporalAdmissionError(
                "causal_inputs.admission_result_refusal_count_inconsistent"
            )
        if self.refused_formation_count > self.formation_count:
            raise TemporalAdmissionError("causal_inputs.admission_result_refusal_count_invalid")
        if self.result_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"result_hash"})
        ):
            raise TemporalAdmissionError("causal_inputs.admission_result_identity_invalid")
        return self


class AdmittedInformationSet(_Contract):
    """The consumption binding: unchanged facts, one schedule, what was admitted.

    This is the artifact a strategy-bound Program binds, and the reason no fact
    artifact needs to. It names the fact surfaces by their own unchanged hashes,
    names the schedule, and seals what the combination admitted.

    Re-running the same facts under a different schedule produces a different
    ``admission_hash`` and leaves every ``input_surface_hash`` exactly where it
    was. That asymmetry is the firewall: data and Features record what the world
    did, and a strategy only decides what it could see.
    """

    kind: Literal["AdmittedInformationSet"] = "AdmittedInformationSet"
    strategy_schedule_hash: str = Field(pattern=_HASH)
    input_surface_hashes: Mapping[str, str]
    input_temporal_authority_hashes: Mapping[str, str]
    results: tuple[TemporalAdmissionResult, ...] = Field(min_length=1)

    admitted_sessions_hash: str = Field(pattern=_HASH)
    admitted_session_count: int = Field(ge=0)
    first_admitted_session: date | None = None
    last_admitted_session: date | None = None
    """The formation axis, by identity plus bounds rather than by enumeration."""

    formation_axis_hash: str = Field(pattern=_HASH)
    session_axis_hash: str = Field(pattern=_HASH)

    disposition: Literal["ADMITTED", "REFUSED"]
    admission_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the joint input admission under one strategy schedule.

        Args:
            values: Input identities, per-input results, and axis evidence.

        Returns:
            The information set with its canonical admission identity.
        """
        return _seal(cls, "admission_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_admission(self) -> Self:
        """Verify the input axis, dispositions, bindings, and seal.

        Returns:
            This validated information set.

        Raises:
            TemporalAdmissionError: If its results or hashes are inconsistent.
        """
        refused = tuple(item for item in self.results if item.disposition == "REFUSED")
        if (self.disposition == "REFUSED") != bool(refused):
            raise TemporalAdmissionError("causal_inputs.admitted_set_disposition_inconsistent")
        if any(item.strategy_schedule_hash != self.strategy_schedule_hash for item in self.results):
            # One binding, one schedule. A set assembled from two would report a
            # single admission over inputs that were never checked together.
            raise TemporalAdmissionError("causal_inputs.admitted_set_schedule_mixed")
        ids = [item.input_id for item in self.results]
        if ids != sorted(set(ids)):
            raise TemporalAdmissionError("causal_inputs.admitted_set_input_axis_invalid")
        if set(ids) != set(self.input_surface_hashes) or set(ids) != set(
            self.input_temporal_authority_hashes
        ):
            raise TemporalAdmissionError("causal_inputs.admitted_set_input_map_incomplete")
        if any(
            self.input_temporal_authority_hashes[item.input_id] != item.input_authority_hash
            for item in self.results
        ):
            raise TemporalAdmissionError("causal_inputs.admitted_set_authority_map_inconsistent")
        if self.admission_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"admission_hash"})
        ):
            raise TemporalAdmissionError("causal_inputs.admitted_set_identity_invalid")
        return self


__all__ = [
    "AdmittedInformationSet",
    "AnchoredInstantPolicy",
    "CausalInputAuthority",
    "SessionAnchor",
    "SessionEvent",
    "StrategyDecisionSchedule",
    "TemporalAdmissionError",
    "TemporalAdmissionResult",
    "TemporalUsage",
]
