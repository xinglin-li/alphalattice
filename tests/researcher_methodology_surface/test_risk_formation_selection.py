"""Formation selection is authored, and capacity never decides which dates go.

Requirement tests for the split between the estimator's scientific lookback and
the publication's formation axis. The two used to be one number, so a study that
needed a wider axis could only be refused.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from alphalattice.investment.risk_research.experiments.formation_selection import (
    BOUND_AUTHORITY_FORMATIONS,
    DEFAULT_TRAILING_FORMATION_COUNT,
    DEVELOPMENT_FORMATION_CAPACITY,
    LEGACY_TRAILING_FIXED_FORMATIONS,
    FormationSelectionError,
    FormationSelectionPolicy,
    eligible_formation_sessions,
    select_formation_sessions,
)
from alphalattice.investment.risk_research.experiments.window import (
    REQUIRED_LOOKBACK_SESSIONS,
    REQUIRED_NEXT_SESSIONS,
)

_LOOKBACK = REQUIRED_LOOKBACK_SESSIONS
_NEXT = REQUIRED_NEXT_SESSIONS


def _axis(count: int) -> tuple[date, ...]:
    start = date(2016, 8, 1)
    return tuple(start + timedelta(days=index) for index in range(count))


def _available(formation_count: int) -> tuple[date, ...]:
    return _axis(_LOOKBACK + formation_count + _NEXT)


def test_the_scientific_lookback_is_unchanged() -> None:
    """The estimator's history requirement is not what this owner negotiates."""

    assert REQUIRED_LOOKBACK_SESSIONS == 314
    assert _LOOKBACK + 1 == 315


def test_legacy_default_still_selects_exactly_one_thousand() -> None:
    available = _available(1_264)
    selected = select_formation_sessions(
        policy=FormationSelectionPolicy.legacy(),
        available=available,
        lookback_sessions=_LOOKBACK,
        next_sessions=_NEXT,
    )
    assert DEFAULT_TRAILING_FORMATION_COUNT == 1_000
    assert len(selected) == 1_000
    eligible = eligible_formation_sessions(
        available=available, lookback_sessions=_LOOKBACK, next_sessions=_NEXT
    )
    assert len(eligible) == 1_264
    assert selected == eligible[-1_000:]


def test_bound_authority_publishes_exactly_the_declared_axis() -> None:
    available = _available(1_264)
    eligible = eligible_formation_sessions(
        available=available, lookback_sessions=_LOOKBACK, next_sessions=_NEXT
    )
    selected = select_formation_sessions(
        policy=FormationSelectionPolicy.bound_authority(),
        available=available,
        lookback_sessions=_LOOKBACK,
        next_sessions=_NEXT,
        requested_sessions=eligible,
    )
    assert len(selected) == 1_264
    assert selected == eligible


def test_both_policies_agree_on_their_shared_trailing_formations() -> None:
    """The numbers cannot differ because the axis does not differ.

    ``build_development_covariance_surface`` iterates the selected sessions and
    reads one window per session, so two policies that select the same sessions
    feed the estimator identical inputs in identical order.
    """

    available = _available(1_264)
    eligible = eligible_formation_sessions(
        available=available, lookback_sessions=_LOOKBACK, next_sessions=_NEXT
    )
    legacy = select_formation_sessions(
        policy=FormationSelectionPolicy.legacy(),
        available=available,
        lookback_sessions=_LOOKBACK,
        next_sessions=_NEXT,
    )
    bound = select_formation_sessions(
        policy=FormationSelectionPolicy.bound_authority(),
        available=available,
        lookback_sessions=_LOOKBACK,
        next_sessions=_NEXT,
        requested_sessions=eligible,
    )
    assert legacy == bound[-len(legacy) :]


def test_capacity_refuses_rather_than_truncating() -> None:
    available = _available(DEVELOPMENT_FORMATION_CAPACITY + 1)
    eligible = eligible_formation_sessions(
        available=available, lookback_sessions=_LOOKBACK, next_sessions=_NEXT
    )
    with pytest.raises(FormationSelectionError) as failure:
        select_formation_sessions(
            policy=FormationSelectionPolicy.bound_authority(),
            available=available,
            lookback_sessions=_LOOKBACK,
            next_sessions=_NEXT,
            requested_sessions=eligible,
        )
    assert "capacity_exceeded" in str(failure.value)


def test_a_requested_axis_the_surface_cannot_answer_is_refused() -> None:
    available = _available(600)
    eligible = eligible_formation_sessions(
        available=available, lookback_sessions=_LOOKBACK, next_sessions=_NEXT
    )
    outside = (*eligible, eligible[-1] + timedelta(days=365))
    with pytest.raises(FormationSelectionError) as failure:
        select_formation_sessions(
            policy=FormationSelectionPolicy.bound_authority(),
            available=available,
            lookback_sessions=_LOOKBACK,
            next_sessions=_NEXT,
            requested_sessions=outside,
        )
    assert "requested_axis_unavailable" in str(failure.value)


def test_an_unordered_request_is_refused() -> None:
    available = _available(600)
    eligible = eligible_formation_sessions(
        available=available, lookback_sessions=_LOOKBACK, next_sessions=_NEXT
    )
    with pytest.raises(FormationSelectionError):
        select_formation_sessions(
            policy=FormationSelectionPolicy.bound_authority(),
            available=available,
            lookback_sessions=_LOOKBACK,
            next_sessions=_NEXT,
            requested_sessions=tuple(reversed(eligible)),
        )


def test_policy_identity_is_sealed_and_parameters_match_the_method() -> None:
    legacy = FormationSelectionPolicy.legacy()
    bound = FormationSelectionPolicy.bound_authority()
    assert legacy.method_id == LEGACY_TRAILING_FIXED_FORMATIONS
    assert bound.method_id == BOUND_AUTHORITY_FORMATIONS
    assert legacy.policy_hash != bound.policy_hash
    assert bound.trailing_formation_count is None
    # A validator failure reaches the caller wrapped by pydantic, so the test
    # asserts what a caller actually catches and that the owner's own reason
    # survives inside it.
    with pytest.raises(ValidationError) as mismatched:
        FormationSelectionPolicy.create(
            method_id=BOUND_AUTHORITY_FORMATIONS, trailing_formation_count=1_000
        )
    assert "formation_selection_parameters_invalid" in str(mismatched.value)
    with pytest.raises(ValidationError) as absent:
        FormationSelectionPolicy.create(method_id=LEGACY_TRAILING_FIXED_FORMATIONS)
    assert "formation_selection_parameters_invalid" in str(absent.value)


def test_a_thousand_formation_binding_still_validates_after_the_capacity_change() -> None:
    """Historical bindings resolve unchanged; only the ceiling moved."""

    available = _available(1_000)
    selected = select_formation_sessions(
        policy=FormationSelectionPolicy.legacy(),
        available=available,
        lookback_sessions=_LOOKBACK,
        next_sessions=_NEXT,
    )
    assert len(selected) == 1_000
    assert DEVELOPMENT_FORMATION_CAPACITY > 1_264
