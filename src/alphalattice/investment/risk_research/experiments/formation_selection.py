"""Which formations a development covariance build is published for.

Two questions were answered by one number. ``REQUIRED_LOOKBACK_SESSIONS`` is the
estimator's scientific history requirement and is not negotiable. How many
formations a publication covers is a separate, authored decision, and until now
it was a bare ceiling named after the production path's own trailing slice --
so a study that needed a different axis could only be refused, never authored.

The production builder in ``surfaces/historical.py`` keeps its trailing slice
untouched. This owner exists for the development writer, which already receives
an exact axis on ``RiskDevelopmentInputBinding.formation_sessions`` and never
reads that slice.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

LEGACY_TRAILING_FIXED_FORMATIONS = "LEGACY_TRAILING_FIXED_FORMATIONS"
"""The trailing fixed count the development path has always defaulted to.

Its count stays 1000 and its identity does not move: a binding that declares no
method is this method, so every historical binding and artifact resolves
unchanged.
"""

BOUND_AUTHORITY_FORMATIONS = "BOUND_AUTHORITY_FORMATIONS"
"""Publish exactly the formations the resolved authority asked for.

No truncation, no padding, no substitution. A requested axis the return surface
cannot answer is a typed refusal, because a build that quietly published a
different axis would be attributed to the one that was requested.
"""

DEFAULT_TRAILING_FORMATION_COUNT = 1000
"""Unchanged, so the legacy method keeps producing exactly what it produced."""

DEVELOPMENT_FORMATION_CAPACITY = 4096
"""How many formations one development build may publish before refusing.

A capacity budget, not a selection rule. It answers whether a build may run; it
never decides which dates a build silently drops. The previous ceiling did both
jobs with one number, which is why widening an axis looked like weakening a
scientific bound.
"""


class FormationSelectionError(ValueError):
    """Stable fail-closed boundary for formation selection."""


class FormationSelectionPolicy(BaseModel):  # type: ignore[misc]
    """The authored rule that decides a development publication's formations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FormationSelectionPolicy"] = "FormationSelectionPolicy"
    method_id: Literal["LEGACY_TRAILING_FIXED_FORMATIONS", "BOUND_AUTHORITY_FORMATIONS"]
    trailing_formation_count: int | None = Field(default=None, ge=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> Self:
        trailing = self.method_id == LEGACY_TRAILING_FIXED_FORMATIONS
        if trailing != (self.trailing_formation_count is not None):
            raise FormationSelectionError("risk_research.formation_selection_parameters_invalid")
        if self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise FormationSelectionError("risk_research.formation_selection_identity_invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        provisional = cls.model_construct(**values, policy_hash="0" * 64)
        return cls(
            **values,
            policy_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"policy_hash"}))
            ),
        )

    @classmethod
    def legacy(cls, *, count: int = DEFAULT_TRAILING_FORMATION_COUNT) -> Self:
        return cls.create(
            method_id=LEGACY_TRAILING_FIXED_FORMATIONS, trailing_formation_count=count
        )

    @classmethod
    def bound_authority(cls) -> Self:
        return cls.create(method_id=BOUND_AUTHORITY_FORMATIONS)


def eligible_formation_sessions(
    *, available: Sequence[date], lookback_sessions: int, next_sessions: int
) -> tuple[date, ...]:
    """Formations the return surface can actually answer, in axis order.

    Restates ``historical.py``'s own arithmetic -- history behind, realised
    session ahead -- so both paths agree on eligibility even though they differ
    on how many of the eligible formations a publication covers.
    """

    if lookback_sessions < 0 or next_sessions < 0:
        raise FormationSelectionError("risk_research.formation_selection_bounds_invalid")
    stop = len(available) - next_sessions
    if stop <= lookback_sessions:
        return ()
    return tuple(available[lookback_sessions:stop])


def select_formation_sessions(
    *,
    policy: FormationSelectionPolicy,
    available: Sequence[date],
    lookback_sessions: int,
    next_sessions: int,
    requested_sessions: Sequence[date] | None = None,
    capacity: int = DEVELOPMENT_FORMATION_CAPACITY,
) -> tuple[date, ...]:
    """Resolve the exact formation axis, or refuse with a typed reason."""

    eligible = eligible_formation_sessions(
        available=available, lookback_sessions=lookback_sessions, next_sessions=next_sessions
    )
    if not eligible:
        raise FormationSelectionError("risk_research.formation_selection_no_eligible_formations")

    if policy.method_id == LEGACY_TRAILING_FIXED_FORMATIONS:
        count = policy.trailing_formation_count or DEFAULT_TRAILING_FORMATION_COUNT
        selected = eligible[-count:]
    else:
        if requested_sessions is None:
            raise FormationSelectionError("risk_research.formation_selection_request_absent")
        requested = tuple(requested_sessions)
        if requested != tuple(sorted(set(requested))):
            raise FormationSelectionError("risk_research.formation_selection_request_axis_invalid")
        missing = tuple(value for value in requested if value not in set(eligible))
        if missing:
            # Named rather than dropped: a build that published the intersection
            # would be attributed to the axis that was asked for.
            raise FormationSelectionError(
                "risk_research.formation_selection_requested_axis_unavailable"
            )
        selected = requested

    if len(selected) > capacity:
        raise FormationSelectionError("risk_research.formation_selection_capacity_exceeded")
    return selected


__all__ = [
    "BOUND_AUTHORITY_FORMATIONS",
    "DEFAULT_TRAILING_FORMATION_COUNT",
    "DEVELOPMENT_FORMATION_CAPACITY",
    "LEGACY_TRAILING_FIXED_FORMATIONS",
    "FormationSelectionError",
    "FormationSelectionPolicy",
    "eligible_formation_sessions",
    "select_formation_sessions",
]
