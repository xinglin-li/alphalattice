"""Deterministic Data diagnostics for failures, restatements, and divergence.

Everything here observes and classifies; nothing here writes, qualifies, or
propagates. Explicit qualification stays with the storage owner's
``apply_validated_batch`` and dependency propagation stays with the Host, so a
diagnostic can never silently become an admission decision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

from alphalattice.foundation.market_data_ops.sources.contracts import (
    RawDailyBar,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _bar_observation_identity(bar: RawDailyBar) -> str:
    """Observation identity of one raw bar's numeric content.

    Deliberately computed over the same value fields the storage owner hashes
    into ``payload_hash``, but spelled here as this diagnostic's own identity:
    an observation compares content, it does not borrow the store's
    qualification identity or pretend to be one.
    """
    return canonical_hash(
        {
            "kind": "RestatementBarObservation",
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
    )


@dataclass(frozen=True)
class RestatementCandidateObservation:
    """One already-qualified row a fresh fetch disagrees with, by content."""

    listing_id: str
    session_date: date
    prior_identity: str
    candidate_identity: str
    classification: Literal["PROVIDER_CORRECTION_OBSERVED"]


@dataclass(frozen=True)
class RestatementObservationReceipt:
    """Pure pre-admission comparison of candidate rows with qualified history.

    This receipt observes; it never claims. It records no mutation, mints no
    qualified Data identity, and names no downstream surface -- which artifacts
    bind an affected row is knowledge the Host owns, and explicit qualification
    (with its ``bar_revision`` trail and new qualified identity) remains
    ``apply_validated_batch``'s act alone. When ``corrections`` is non-empty the
    scoped before/after identities necessarily differ, which is exactly the
    signal that qualifying the candidate would produce a new Data identity.
    """

    provider: str
    listing_scope: tuple[str, ...]
    session_scope_start: date
    session_scope_end: date
    qualified_scope_identity: str
    candidate_scope_identity: str
    exact_reuse_count: int
    new_session_count: int
    qualified_only_count: int
    corrections: tuple[RestatementCandidateObservation, ...]
    receipt_hash: str


def audit_bounded_restatements(
    *,
    candidate_bars: Sequence[RawDailyBar],
    qualified_bars: Sequence[RawDailyBar],
    provider: str,
    listing_scope: tuple[str, ...],
    session_scope_start: date,
    session_scope_end: date,
) -> RestatementObservationReceipt:
    """Compare a fetched candidate batch with qualified history, read-only.

    Bounded by an explicit listing and session scope -- an empty or reversed
    scope is refused rather than widened, so this can never become an unbounded
    historical scan. Rows outside the scope are ignored on both sides.

    ``same content identity -> EXACT_REUSE``; ``different content identity ->
    PROVIDER_CORRECTION_OBSERVED``, carrying prior and candidate identities so
    the prior content stays nameable after any later explicit qualification.
    """
    if not listing_scope or listing_scope != tuple(sorted(set(listing_scope))):
        raise ValueError("restatement audit requires an explicit ordered listing scope")
    if session_scope_start > session_scope_end:
        raise ValueError("restatement audit session scope is reversed")
    scope = frozenset(listing_scope)

    def scoped(bars: Sequence[RawDailyBar]) -> dict[tuple[str, date], RawDailyBar]:
        rows: dict[tuple[str, date], RawDailyBar] = {}
        for bar in bars:
            if bar.listing_id not in scope:
                continue
            if not session_scope_start <= bar.session_date <= session_scope_end:
                continue
            key = (bar.listing_id, bar.session_date)
            if key in rows:
                raise ValueError("restatement audit received duplicate rows for one session")
            rows[key] = bar
        return rows

    qualified = scoped(qualified_bars)
    candidate = scoped(candidate_bars)
    exact_reuse = 0
    corrections: list[RestatementCandidateObservation] = []
    for key in sorted(qualified.keys() & candidate.keys()):
        prior_identity = _bar_observation_identity(qualified[key])
        candidate_identity = _bar_observation_identity(candidate[key])
        if prior_identity == candidate_identity:
            exact_reuse += 1
        else:
            corrections.append(
                RestatementCandidateObservation(
                    listing_id=key[0],
                    session_date=key[1],
                    prior_identity=prior_identity,
                    candidate_identity=candidate_identity,
                    classification="PROVIDER_CORRECTION_OBSERVED",
                )
            )

    def scope_identity(rows: Mapping[tuple[str, date], RawDailyBar]) -> str:
        return canonical_hash(
            {
                "kind": "RestatementScopeContent",
                "rows": [
                    (key[0], key[1], _bar_observation_identity(rows[key])) for key in sorted(rows)
                ],
            }
        )

    # "After" is the scoped content as explicit qualification would leave it:
    # candidate rows where the fetch covered a session, qualified rows
    # elsewhere. Any correction therefore forces the two identities apart.
    merged = dict(qualified)
    merged.update(candidate)
    qualified_scope_identity = scope_identity(qualified)
    candidate_scope_identity = scope_identity(merged)
    new_session_count = len(candidate.keys() - qualified.keys())
    qualified_only_count = len(qualified.keys() - candidate.keys())
    receipt_hash = canonical_hash(
        {
            "kind": "RestatementObservationReceipt",
            "provider": provider,
            "listing_scope": listing_scope,
            "session_scope_start": session_scope_start,
            "session_scope_end": session_scope_end,
            "qualified_scope_identity": qualified_scope_identity,
            "candidate_scope_identity": candidate_scope_identity,
            "exact_reuse_count": exact_reuse,
            "new_session_count": new_session_count,
            "qualified_only_count": qualified_only_count,
            "corrections": [
                (item.listing_id, item.session_date, item.prior_identity, item.candidate_identity)
                for item in corrections
            ],
        }
    )
    return RestatementObservationReceipt(
        provider=provider,
        listing_scope=listing_scope,
        session_scope_start=session_scope_start,
        session_scope_end=session_scope_end,
        qualified_scope_identity=qualified_scope_identity,
        candidate_scope_identity=candidate_scope_identity,
        exact_reuse_count=exact_reuse,
        new_session_count=new_session_count,
        qualified_only_count=qualified_only_count,
        corrections=tuple(corrections),
        receipt_hash=receipt_hash,
    )


@dataclass(frozen=True)
class UniverseSpyDivergencePolicy:
    """A deliberately broad, documented classification bound.

    0.05 of same-clock simple return in one session is far outside anything a
    broad equal-weight universe and SPY plausibly do apart; the bound exists to
    catch data that is wrong, not to grade strategies, and must never be tuned
    against research results.
    """

    absolute_divergence_bound: float = 0.05

    def __post_init__(self) -> None:
        """Require a finite classification bound strictly between zero and one."""
        if not 0.0 < self.absolute_divergence_bound < 1.0:
            raise ValueError("divergence bound must be between zero and one")

    @property
    def policy_hash(self) -> str:
        """Hash the divergence bound as the policy's deterministic identity."""
        return canonical_hash(
            {
                "kind": "UniverseSpyDivergencePolicy",
                "absolute_divergence_bound": self.absolute_divergence_bound,
            }
        )


@dataclass(frozen=True)
class SessionDivergenceObservation:
    """Same-session return comparison for a flagged divergence."""

    session_date: date
    universe_aggregate_return: float
    spy_return: float
    divergence: float


@dataclass(frozen=True)
class UniverseSpyDivergenceDiagnostic:
    """Advisory evidence that broad Universe data behaved plausibly against SPY.

    A sentinel, not a scientific admission threshold and not a Portfolio
    benchmark replacement. ``attribution`` is fixed at ``UNATTRIBUTED`` because
    a divergence proves inconsistency, never which side is wrong; the only
    consequence a consumer may attach is quarantine and review.
    """

    universe_source_identity: str
    spy_source_identity: str
    session_count: int
    classification: Literal["CONSISTENT", "IMPLAUSIBLE_DIVERGENCE_REVIEW_REQUIRED"]
    attribution: Literal["UNATTRIBUTED"]
    flagged_sessions: tuple[SessionDivergenceObservation, ...]
    largest_absolute_divergence: float
    policy_hash: str
    diagnostic_hash: str


def evaluate_universe_spy_divergence(
    *,
    universe_returns: Mapping[date, float],
    spy_returns: Mapping[date, float],
    universe_source_identity: str,
    spy_source_identity: str,
    policy: UniverseSpyDivergencePolicy | None = None,
) -> UniverseSpyDivergenceDiagnostic:
    """Compare qualified Universe aggregate returns with the same-clock SPY path.

    Session alignment is exact or refused: both series must cover the identical
    ordered session set, because a diagnostic that silently intersected axes
    would hide exactly the missing-session defects it exists to surface. Inputs
    arrive already computed under qualified Data authority, with their source
    identities carried into the evidence; this module never reaches upward to
    compose them.
    """
    active_policy = policy or UniverseSpyDivergencePolicy()
    if not universe_returns:
        raise ValueError("universe-spy divergence requires at least one session")
    universe_axis = tuple(sorted(universe_returns))
    spy_axis = tuple(sorted(spy_returns))
    if universe_axis != spy_axis:
        raise ValueError("data.universe_spy_axis_misaligned")
    flagged: list[SessionDivergenceObservation] = []
    largest = 0.0
    for session in universe_axis:
        universe_value = float(universe_returns[session])
        spy_value = float(spy_returns[session])
        if not math.isfinite(universe_value) or not math.isfinite(spy_value):
            raise ValueError("data.universe_spy_return_non_finite")
        divergence = universe_value - spy_value
        largest = max(largest, abs(divergence))
        if abs(divergence) > active_policy.absolute_divergence_bound:
            flagged.append(
                SessionDivergenceObservation(
                    session_date=session,
                    universe_aggregate_return=universe_value,
                    spy_return=spy_value,
                    divergence=divergence,
                )
            )
    classification: Literal["CONSISTENT", "IMPLAUSIBLE_DIVERGENCE_REVIEW_REQUIRED"] = (
        "IMPLAUSIBLE_DIVERGENCE_REVIEW_REQUIRED" if flagged else "CONSISTENT"
    )
    diagnostic_hash = canonical_hash(
        {
            "kind": "UniverseSpyDivergenceDiagnostic",
            "universe_source_identity": universe_source_identity,
            "spy_source_identity": spy_source_identity,
            "session_count": len(universe_axis),
            "classification": classification,
            "flagged": [
                (
                    item.session_date,
                    item.universe_aggregate_return,
                    item.spy_return,
                    item.divergence,
                )
                for item in flagged
            ],
            "largest_absolute_divergence": largest,
            "policy_hash": active_policy.policy_hash,
        }
    )
    return UniverseSpyDivergenceDiagnostic(
        universe_source_identity=universe_source_identity,
        spy_source_identity=spy_source_identity,
        session_count=len(universe_axis),
        classification=classification,
        attribution="UNATTRIBUTED",
        flagged_sessions=tuple(flagged),
        largest_absolute_divergence=largest,
        policy_hash=active_policy.policy_hash,
        diagnostic_hash=diagnostic_hash,
    )
