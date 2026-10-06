"""Resolve a Feature's observation clock through the Feature owner's own module.

Nothing here computes a factor, restates a formula, or decides a lag. It asks
``feature_engine.catalog.observation_clock`` for the clock it already derives
from the recipe, and reports it.

**Why this delegates rather than reads.** The Feature remediation gave Features a
true observation clock and split three questions one integer used to answer: the
Formula's own economic skip, source availability, and a model-side lag. That
module derives the clock from the recipe -- deliberately, because "a clock stored
beside the recipe is a number that can disagree with the window it describes".
Re-deriving it here would be a second declaration with exactly that failure mode.

**What this module must never grow.** A decision cutoff, an entry offset, an
entry or exit timing, a rebalance clock. Those belong to
``causal_outcomes.execution`` and are the Portfolio Host's to resolve. An earlier
version of this package hard-coded ``FORMATION_SESSION_CLOSE``,
``entry_offset_sessions = 1`` and ``NEXT_COMMON_SESSION_OFFICIAL_OPEN`` into a
score binding, which is a producer taking a position on when its own values are
tradable -- the responsibility inversion the Feature work exists to end.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FeatureObservationClock,
    installed_feature_availability_policy,
    observation_clock_for,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .standardize import SimpleSignalError


@dataclass(frozen=True, slots=True)
class FeatureClockProvenance:
    """One feature's observation semantics, as the Feature owner derives them."""

    feature_id: str
    formula_ref: str
    formula: str
    window_sessions: int
    observation_session_offset_sessions: int | None
    """The Formula's own economic skip, from ``FeatureObservationClock``.

    Named for what it is rather than for the field it came from. ``lag_sessions``
    was read at four call sites as four different things; this is the one the
    clock module derives and the only one a consumer may use.
    """

    availability_delay_sessions: int
    availability_policy_id: str
    availability_policy_hash: str
    observation_clock_hash: str
    source_interval: str
    """The Formula's ordered source rows relative to ``t``, measured not declared."""

    minimum_observations: int
    return_convention: str
    methodology_identity: str
    """Hash over the fields that can move the numbers or the availability."""

    catalog_binding_hash: str
    """Which revision of the Feature catalog said so."""


def resolve_feature_clock(feature_id: str) -> FeatureClockProvenance:
    """Resolve one installed feature's observation clock, or refuse by name.

    A feature the shipped catalog does not carry has no owner-derived clock, and
    a score built on it cannot state when its values were observable. That is a
    refusal here rather than a default of zero, because a defaulted skip is
    indistinguishable from a verified one at the point where it matters.
    """
    catalog = FeatureCatalog.load()
    matches = [value for value in catalog.factors if value.factor_id == feature_id]
    if len(matches) != 1:
        raise SimpleSignalError(
            "alpha_research.simple_signal_feature_clock_not_declared:" + feature_id
        )
    spec = matches[0]
    clock: FeatureObservationClock = observation_clock_for(spec)
    availability = installed_feature_availability_policy()
    # The same field set the Feature Desk uses to decide whether a recipe is
    # still the control it claims to be. Importing the contract's tuple keeps one
    # owner for "what can move the numbers"; restating it would let the two drift.
    from alphalattice.kernel.quant.factor_contracts import NUMERICAL_SPEC_FIELDS

    payload = spec.model_dump(mode="json")
    methodology = canonical_hash({key: payload[key] for key in NUMERICAL_SPEC_FIELDS})
    return FeatureClockProvenance(
        feature_id=str(spec.factor_id),
        formula_ref=str(spec.formula_ref),
        formula=str(spec.formula),
        window_sessions=int(clock.declared_window_sessions),
        observation_session_offset_sessions=clock.latest_consumed_session_offset,
        availability_delay_sessions=int(availability.publication_delay_sessions),
        availability_policy_id=str(availability.policy_id),
        availability_policy_hash=str(availability.policy_hash),
        observation_clock_hash=str(clock.clock_hash),
        source_interval=str(clock.source_interval_rendered),
        minimum_observations=int(clock.source_interval.minimum_history_rows),
        return_convention=str(spec.return_convention),
        methodology_identity=str(methodology),
        catalog_binding_hash=str(catalog.binding.catalog_hash),
    )


__all__ = ["FeatureClockProvenance", "resolve_feature_clock"]
