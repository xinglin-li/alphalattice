"""What a portfolio consumer requires of any signed score, whoever produced it.

A capability rather than a Desk module, and that placement is the boundary. The
Portfolio Desk states the requirement; a score producer satisfies it; neither
imports the other. A corrected Dynamic score implements this Protocol in its own
package and is consumed by handle -- no Risk or Portfolio source changes.

**Two authorities, and neither may state the other's terms.**

A score producer owns *observation* and *availability*: which session a value
belongs to, and when that value becomes usable. Those come from the Feature
owner's ``FeatureObservationClock`` and ``FeatureAvailabilityPolicy``, derived
from the recipe that already carries them.

A portfolio owns *execution*: which close it decides on, which open it enters at,
which open it exits at. Those come from
``foundation.causal_outcomes.execution`` and are read as published events with
real timestamps, not as offsets a consumer counted.

An earlier version of this module had one contract holding both, and the score
producer hard-coded ``FORMATION_SESSION_CLOSE``, ``entry_offset_sessions = 1``
and ``NEXT_COMMON_SESSION_OFFICIAL_OPEN`` to fill it. That is the responsibility
inversion the Feature remediation exists to end: a producer that states an entry
offset has taken a position on tradability, and a strategy clock that reaches
into a Formula is the defect one integer used to hide.

**This module no longer checks the ordering.** It used to, with a chain that
compared ``score_available_at <= decision_at < entry_at < exit_at`` and had no
way to say "orders are due at 09:00 tomorrow" -- so a score not finished at the
close could only be admitted by a producer moving its own observation back a
session. That chain is gone. The relation lives once, in
``capabilities.causal_inputs.admission``, and a campaign reaches it through
``admit_decision_input_set`` with a schedule resolved from the installed
execution method.

What remains here is the part that is genuinely this capability's: what a signed
score *is* (signed, dimensionless, finite, on declared axes), what a producer may
state about its own observation, the execution events as their owner published
them, and ``score_input_authority`` -- the thin resolver that restates a
producer's two session offsets in the shared vocabulary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Literal, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    CausalInputAuthority,
    SessionAnchor,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

_HASH = r"^[0-9a-f]{64}$"

SIGNED_SCORE_SEMANTICS = "CROSS_SECTIONAL_SIGNED_DIMENSIONLESS_SCORE"
"""The one unit this consumer accepts.

Not a return, not a probability, not a rank. Signed and dimensionless, so the
objective's preference coefficient is the only thing that gives it a scale.
"""


class SignedScoreClockError(ValueError):
    """A score cannot state, or does not satisfy, the causal event ordering."""


class ScoreObservationAuthority(BaseModel):  # type: ignore[misc]
    """What the score producer owns, and nothing else.

    Every field here traces to the Feature owner's installed clock and
    availability policy. There is no decision cutoff, no entry offset, no entry
    or exit timing and no rebalance clock -- the contract forbids extra fields,
    so one cannot be added by a producer that would like to decide when its
    values are tradable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ScoreObservationAuthority"] = "ScoreObservationAuthority"

    observation_session_offset_sessions: int = Field(ge=0, le=512)
    """Sessions back from the labelled session to the observed one.

    The Formula's own economic skip, taken from ``FeatureObservationClock``. For
    ``mom_252_21`` it is 21 and it is *the Formula's* semantics, not a delay a
    portfolio imposed for tradability -- which is why it may not be traded away
    or compensated for elsewhere.
    """

    availability_delay_sessions: int = Field(ge=0, le=512)
    """Sessions between the observation session and the value becoming usable.

    A property of the *source*. Zero for daily provider bars, which are complete
    at the observation session's own close. A producer that pays a session here
    to feel safe is paying for isolation the execution method already provides.
    """

    availability_policy_id: str = Field(min_length=1, max_length=96)
    availability_policy_hash: str = Field(pattern=_HASH)
    observation_clock_hash: str = Field(pattern=_HASH)
    """The Feature owner's own clock identity, so a changed clock is a changed score."""

    formula_observation_semantics: str = Field(min_length=1, max_length=200)
    """What the Formula says its window is, in the owner's own words."""

    source_authority_id: str = Field(min_length=1, max_length=160)
    methodology_identity: str = Field(pattern=_HASH)

    strategy_scope: Literal[
        "CONDITIONAL_FIXED_SCORE_NOT_A_STRATEGY_SIGNAL",
        "CORRECTED_STRATEGY_SIGNAL",
    ]
    """What a result measured against this score is allowed to conclude.

    A clean observation clock is not a strategy signal. ``mom_252_21`` is
    causally admissible under both the pre-successor and the successor Feature
    clock -- its 21-session skip exceeds the disputed one-session ambiguity by
    twenty -- and it is still a fixed instrument chosen to hold the signal still
    while Risk and Portfolio arms move.
    """

    authority_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal producer-owned observation and availability fields without execution policy.

        Args:
            values: Typed producer observation fields; an existing authority_hash is replaced by the
                derived hash.

        Returns:
            The validated observation authority with its canonical identity.
        """
        draft = dict(values)
        draft.pop("authority_hash", None)
        identity = cls.model_construct(**draft, authority_hash="0" * 64).model_dump(
            mode="json", exclude={"authority_hash"}
        )
        return cls(**draft, authority_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the canonical identity of producer-owned observation fields.

        Returns:
            This validated score observation authority.

        Raises:
            SignedScoreClockError: The authority hash differs from its fields.
        """
        if self.authority_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"authority_hash"})
        ):
            raise SignedScoreClockError("portfolio_strategy_lab.score_authority_identity_invalid")
        return self


class PortfolioExecutionEvents(BaseModel):  # type: ignore[misc]
    """The events a portfolio decides and trades on, as its owner published them.

    Resolved by the Host from ``causal_outcomes.execution`` -- the same rows the
    campaign's realized returns are measured on -- rather than reconstructed from
    offsets. A reconstruction that disagreed with the returns would pair a
    decision with a price it never got.

    **The handles are the point of this contract, not decoration.** An events
    object that carried only timestamps could be re-sealed by anybody: a replay
    would have nothing to reopen and would be reduced to reading the artifact
    back and agreeing with it. ``outcome_snapshot_hash`` and
    ``outcome_manifest_ref`` name the exact durable graph, and
    ``method_seal_disposition`` with ``method_binding_hash`` name the method
    authority that graph is sealed to -- so a verifier can go to the owner, read
    the same rows, resolve the same seal, rebuild this object and require it
    whole.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PortfolioExecutionEvents"] = "PortfolioExecutionEvents"

    outcome_snapshot_hash: str = Field(pattern=_HASH)
    outcome_manifest_ref: str = Field(min_length=1, max_length=256)
    method_seal_disposition: Literal["METHOD_BOUND", "LEGACY_READBACK_ONLY"]
    """Whether the outcome snapshot carries method authority at all.

    A disposition rather than an optional binding, following the owner's own
    contract: ``None`` invites a reader to treat "no seal" as "no constraint".
    A legacy snapshot's rows are readable and its method authority is absent,
    and a campaign built on one cannot claim its execution method was verified.
    """

    method_binding_hash: str | None = Field(default=None, pattern=_HASH)
    execution_recipe_id: str | None = Field(default=None, min_length=1, max_length=96)
    execution_recipe_hash: str | None = Field(default=None, pattern=_HASH)
    """The method the snapshot is sealed to, or absent for a legacy snapshot.

    Read from the owner's ``ExecutionOutcomeMethodBinding``, never assigned from
    a literal. An earlier version named ``NEXT_OPEN_TO_OPEN_ONE_SESSION`` in the
    builder, which meant every campaign reported that method whether or not the
    snapshot it consumed was published under it.
    """
    ordered_formation_sessions: tuple[date, ...] = Field(min_length=1)
    decision_at: tuple[datetime, ...] = Field(min_length=1)
    entry_at: tuple[datetime, ...] = Field(min_length=1)
    exit_at: tuple[datetime, ...] = Field(min_length=1)
    session_close_at: Mapping[date, datetime]
    """Official close per ordered session, for resolving an observation session.

    Wider than the formation axis on purpose: a Formula with an economic skip
    observes a session earlier than the formation it is labelled with, and the
    close of *that* session is when its value becomes usable.
    """

    events_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the owner-published event rows, durable graph handles and method disposition.

        Args:
            values: Typed event and owner-seal fields; events_hash is derived from their JSON form.

        Returns:
            The validated event contract with its canonical identity.
        """
        draft = dict(values)
        draft.pop("events_hash", None)
        identity = cls.model_construct(**draft, events_hash="0" * 64).model_dump(
            mode="json", exclude={"events_hash"}
        )
        return cls(**draft, events_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Validate event-axis lengths, method-bound seal completeness and event identity.

        Returns:
            This validated event contract; the causal admission owner checks decision ordering.

        Raises:
            SignedScoreClockError: Event lengths, disposition versus complete seal fields, or the
                hash is inconsistent.
        """
        lengths = {
            len(self.ordered_formation_sessions),
            len(self.decision_at),
            len(self.entry_at),
            len(self.exit_at),
        }
        if len(lengths) != 1:
            raise SignedScoreClockError("portfolio_strategy_lab.execution_events_axis_mismatch")
        bound = self.method_seal_disposition == "METHOD_BOUND"
        named = (self.method_binding_hash, self.execution_recipe_id, self.execution_recipe_hash)
        if bound != all(value is not None for value in named):
            # Method-bound and nameless, or legacy and named. Either way the
            # disposition and the fields disagree about whether an authority
            # exists, and a consumer would have to pick one.
            raise SignedScoreClockError(
                "portfolio_strategy_lab.execution_events_seal_fields_inconsistent"
            )
        if self.events_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"events_hash"})
        ):
            raise SignedScoreClockError("portfolio_strategy_lab.execution_events_identity_invalid")
        return self


class ResolvedSignedScore(BaseModel):  # type: ignore[misc]
    """One score surface, its axes, its identity and its observation authority.

    ``values`` is not on this model. The numbers travel beside it as a plain
    array so a contract that is cheap to publish stays cheap to publish, and
    ``values_identity`` is what binds the two: a consumer that wants to know it
    has the same numbers hashes them and compares.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["ResolvedSignedScore"] = "ResolvedSignedScore"
    semantics: Literal["CROSS_SECTIONAL_SIGNED_DIMENSIONLESS_SCORE"] = (
        "CROSS_SECTIONAL_SIGNED_DIMENSIONLESS_SCORE"
    )
    binding_hash: str = Field(pattern=_HASH)
    values_identity: str = Field(pattern=_HASH)
    ordered_formation_sessions: tuple[date, ...] = Field(min_length=1)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    observation: ScoreObservationAuthority
    rederived: bool
    disposition: str = Field(min_length=1, max_length=96)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axes(self) -> Self:
        """Require sorted unique score sessions and distinct listing handles.

        Returns:
            This validated score metadata; numerical values are bound separately.

        Raises:
            SignedScoreClockError: Sessions are not canonical or listing handles repeat.
        """
        sessions = self.ordered_formation_sessions
        if sessions != tuple(sorted(set(sessions))):
            raise SignedScoreClockError("portfolio_strategy_lab.score_session_axis_unordered")
        if len(set(self.ordered_listing_ids)) != len(self.ordered_listing_ids):
            raise SignedScoreClockError("portfolio_strategy_lab.score_listing_axis_duplicated")
        return self


@runtime_checkable
class SignedScoreSource(Protocol):
    """The one thing the Portfolio Desk needs from a score producer.

    Deliberately two methods. ``resolve`` re-derives and reports what it found;
    ``projection`` hands over the numbers on the axis the campaign decided on. A
    producer that could only do the second would be a file, not an authority.
    """

    def resolve(self, *, binding_hash: str) -> ResolvedSignedScore:
        """Re-derive a signed score authority from its exact producer binding.

        Args:
            binding_hash: Producer binding identity requested by the consumer.

        Returns:
            The resolved axes, value commitment, observation authority and rederivation disposition.
        """
        ...

    def projection(
        self,
        *,
        binding_hash: str,
        formation_sessions: Sequence[date],
        ordered_listing_ids: Sequence[str],
    ) -> FloatArray:
        """Project producer-owned score numbers onto explicit consumer axes.

        Args:
            binding_hash: Producer binding whose score values are requested.
            formation_sessions: Consumer formation-session order.
            ordered_listing_ids: Consumer listing order.

        Returns:
            The formation-by-listing signed dimensionless numerical surface.
        """
        ...


def score_input_authority(
    observation: ScoreObservationAuthority,
    *,
    surface_hash: str,
    input_id: str,
    input_kind: str = "RAW_SCORE",
    derived_ready: AnchoredInstantPolicy | None = None,
) -> CausalInputAuthority:
    """Restate a score producer's own clock in the shared vocabulary.

    The thin resolver, and nothing more: the two session offsets the producer
    already declares become two anchors on the exchange-session axis, and every
    comparison against a strategy happens at the shared owner.

    Both anchors sit at ``OFFICIAL_CLOSE`` because that is what the Feature
    owner's installed availability catalog says -- every source it holds publishes
    at the official close with no delay -- and ``source_available`` carries the
    basis that admits this is installed policy rather than an observed
    publication receipt. The economic skip reaches backwards and the publication
    delay reaches forwards, which is why they stay separate terms: they answer
    different questions and only one of them belongs to the Formula.

    ``derived_ready`` is the producer's own computation cost, and it is the field
    that makes a fitted score sayable at all. ``None`` for a score whose values
    are a direct transform of the observation; supplied, with a measured or
    budgeted basis, by a producer that fits a model overnight.
    """
    return CausalInputAuthority.create(
        input_id=input_id,
        input_kind=input_kind,
        temporal_usage="DECISION_INPUT",
        surface_hash=surface_hash,
        owner_authority_id=observation.source_authority_id,
        owner_identity_hash=observation.authority_hash,
        observed_through=AnchoredInstantPolicy.exchange_event(
            offset_sessions=-observation.observation_session_offset_sessions,
            event="OFFICIAL_CLOSE",
            policy_id="SCORE_OBSERVATION_SESSION_CLOSE",
        ),
        source_available=AnchoredInstantPolicy.create(
            policy_id=observation.availability_policy_id,
            basis="INSTALLED_SOURCE_AVAILABILITY_POLICY",
            anchor=SessionAnchor(
                offset_sessions=(
                    observation.availability_delay_sessions
                    - observation.observation_session_offset_sessions
                ),
                event="OFFICIAL_CLOSE",
            ),
            minutes_after_anchor=0,
            rationale=(
                "installed source availability policy "
                f"{observation.availability_policy_id}, "
                f"{observation.availability_delay_sessions} sessions after the "
                "observation session close"
            ),
        ),
        derived_ready=derived_ready,
        observation_semantics=observation.formula_observation_semantics,
        point_in_time_disposition="CURRENT_MEMBERSHIP_BACKFILLED",
    )


def require_signed_finite(values: FloatArray, *, minimum_finite_per_formation: int = 2) -> None:
    """The score keeps its sign, and a formation nobody could score is refused.

    Negative values are not clipped, not shifted and not ranked away. This is
    checked at the boundary rather than assumed, because every score defect this
    study has had -- ``score > 0`` selection, non-negative slope calibration,
    dispersion reconstruction -- was a producer quietly deciding that the
    negative half did not count.
    """
    if values.ndim != 2:
        raise SignedScoreClockError("portfolio_strategy_lab.score_values_not_a_surface")
    finite = np.isfinite(values)
    if int(finite.sum()) == 0:
        raise SignedScoreClockError("portfolio_strategy_lab.score_values_all_unresolved")
    if int(np.min(finite.sum(axis=1))) < minimum_finite_per_formation:
        raise SignedScoreClockError("portfolio_strategy_lab.score_formation_coverage_insufficient")
    resolved = values[finite]
    if float(np.max(np.abs(resolved))) == 0.0:
        raise SignedScoreClockError("portfolio_strategy_lab.score_values_identically_zero")
    if not bool(np.any(resolved < 0.0)):
        # A cross-sectionally standardized score is centred, so an all-positive
        # surface means something upstream removed the negative half. Stage 6
        # once ran a whole campaign on a signal that had been clipped this way
        # and reported the tie-break as a method comparison.
        raise SignedScoreClockError("portfolio_strategy_lab.score_values_not_signed")


__all__ = [
    "SIGNED_SCORE_SEMANTICS",
    "PortfolioExecutionEvents",
    "ResolvedSignedScore",
    "ScoreObservationAuthority",
    "SignedScoreClockError",
    "SignedScoreSource",
    "require_signed_finite",
    "score_input_authority",
]
