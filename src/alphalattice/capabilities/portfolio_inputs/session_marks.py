"""A published session mark, restated in the shared temporal vocabulary.

The mark decides the optimizer reference, so it is a **decision input** and not
provenance a reader glances at. It therefore belongs in the same admitted set as
the score, the covariance and the tradability universe, checked against the same
strategy clock by the same owner. Leaving it out meant a Campaign could name a
close-mark method it had never established was knowable.

This is the adapter and nothing else: it restates what the Market Data owner
sealed and what the Feature owner installs, and it copies no relation. Every
comparison happens in ``causal_inputs.admission``.

It lives here rather than beside the surface because the surface lives in
``market_data_ops``, which the Feature engine already depends on -- naming the
availability catalog from there would close a package cycle the structural guard
rejects, and is the reason the publisher takes the policy as an argument at all.
``portfolio_inputs`` is where the other three Portfolio decision inputs already
resolve their clocks, so the mark joins them.
"""

from __future__ import annotations

from typing import Final

from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    CausalInputAuthority,
    SessionAnchor,
    SessionEvent,
    TemporalAdmissionError,
)
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FeatureAvailabilityPolicy,
    installed_source_availability_catalog,
)
from alphalattice.foundation.market_data_ops.publication.session_marks import (
    SessionMarkSurface,
)
from alphalattice.protocols.research_authoring.contracts import MarketPhase

SESSION_MARK_INPUT_ID: Final = "session_mark"
SESSION_MARK_SOURCE_FIELDS: Final = ("open_split_adjusted", "close_split_adjusted")
"""The two published columns a mark is derived from, named for the catalog.

The availability owner is *resolved from what the surface reads* rather than
asserted. The Feature catalog maps every field to the authority that publishes
it, so a build that moved the split-adjusted prices to a source with a real
delay would move this answer without anything here being edited.
"""

SESSION_MARK_OBSERVATION_SEMANTICS: Final = (
    "open(T) to close(T) simple return on the split-adjusted series, within one "
    "session; complete at that session's official close"
)

_AVAILABLE_EVENT: Final[dict[MarketPhase, SessionEvent]] = {
    MarketPhase.OFFICIAL_CLOSE: "OFFICIAL_CLOSE",
}
"""The one availability phase that resolves to a published instant.

``PRE_OPEN``, ``INTRADAY``, ``POST_CLOSE`` and ``UNSPECIFIED`` are deliberately
absent. They are real availability phases and the exchange calendar publishes no
timestamp for any of them, so an owner declaring one gives a deadline comparison
nothing to compare -- and rounding to the nearest published event would either
invent availability the source did not have or throw away a session of it. Every
owner this build installs declares the official close, so the map is one entry
and the refusal below is what a future owner meets rather than a silent guess.
"""


def resolve_session_mark_availability() -> FeatureAvailabilityPolicy:
    """The installed owner of the columns a mark is derived from.

    Asked of the Feature engine's catalog, which is the authority. Both prices
    come from one source today, so the effective policy is that source's; the
    catalog still takes the maximum across whatever set the fields resolve to,
    because a mark is usable only once *both* of its prices are.
    """
    catalog = installed_source_availability_catalog()
    return catalog.effective_policy(catalog.authorities_for(SESSION_MARK_SOURCE_FIELDS))


def session_mark_input_authority(
    surface: SessionMarkSurface, *, input_id: str = SESSION_MARK_INPUT_ID
) -> CausalInputAuthority:
    """Restate one published mark surface's own clock, re-resolved at its owners.

    Two claims, from two owners, neither restated here:

    * ``observed_through`` is the exchange event the surface sealed -- the
      session's own official close, at offset zero. It is a market fact and the
      surface may state it.
    * ``source_available`` is the Feature engine's *installed policy*, resolved
      here from the live catalog rather than read off the surface. The surface's
      sealed id and hash are then compared against the answer, so a forged
      surface naming a policy this build does not install, or naming an installed
      policy under a hash it does not have, is refused before admission.

    That direction matters. If the anchor were built from the surface's own
    numbers, a publisher could assert any availability it liked and every
    downstream comparison would agree with it. Building it from the owner and
    checking the surface against that is what makes the availability term
    somebody else's answer -- and it is what makes a *changed* policy show up as
    a refused admission rather than as an unchanged pass.

    ``derived_ready`` is ``None``. A mark is one division of two published
    prices, not a fit: there is no computation interval to declare, and
    declaring a fictional one would put an invented number inside the deadline
    comparison.
    """
    installed = resolve_session_mark_availability()
    if surface.availability_policy_id != installed.policy_id:
        raise TemporalAdmissionError(
            "portfolio_inputs.session_mark_availability_owner_not_this_build"
        )
    if surface.availability_policy_hash != installed.policy_hash:
        # The surface names an installed policy under a hash the owner does not
        # agree with. A publisher could otherwise seal a real policy's name over
        # a different policy's identity and nothing on the route would look.
        raise TemporalAdmissionError(
            "portfolio_inputs.session_mark_availability_policy_not_this_build"
        )
    event = _AVAILABLE_EVENT.get(installed.available_after_phase)
    if event is None:
        raise TemporalAdmissionError("portfolio_inputs.session_mark_availability_phase_unresolved")
    return CausalInputAuthority.create(
        input_id=input_id,
        input_kind="SESSION_INTRADAY_MARK",
        temporal_usage="DECISION_INPUT",
        surface_hash=surface.surface_hash,
        owner_authority_id="market_data_ops.publication.session_marks",
        owner_identity_hash=surface.epoch.epoch_hash,
        observed_through=AnchoredInstantPolicy.exchange_event(
            offset_sessions=surface.observed_through_offset_sessions,
            event=surface.observed_through_event,
            policy_id="SESSION_MARK_OBSERVATION_SESSION_CLOSE",
        ),
        source_available=AnchoredInstantPolicy.create(
            policy_id=installed.policy_id,
            basis="INSTALLED_SOURCE_AVAILABILITY_POLICY",
            anchor=SessionAnchor(
                offset_sessions=(
                    surface.observed_through_offset_sessions + installed.publication_delay_sessions
                ),
                event=event,
            ),
            minutes_after_anchor=0,
            rationale=(
                f"installed source availability policy {installed.policy_id}, "
                f"{installed.publication_delay_sessions} sessions after the "
                "observation session"
            ),
        ),
        observation_semantics=SESSION_MARK_OBSERVATION_SEMANTICS,
        # The same disposition the Risk and tradability surfaces carry, and for
        # the same reason: the rows are causal on every session and the universe
        # behind them is today's membership backfilled.
        point_in_time_disposition="CURRENT_MEMBERSHIP_BACKFILLED",
    )


__all__ = [
    "SESSION_MARK_INPUT_ID",
    "SESSION_MARK_OBSERVATION_SEMANTICS",
    "SESSION_MARK_SOURCE_FIELDS",
    "resolve_session_mark_availability",
    "session_mark_input_authority",
]
