"""Host-only cross-domain Data-truth composition.

Two things live here that cannot live anywhere else without inverting a
dependency:

- **Qualified session authority.** Materializing an exchange calendar is a
  qualification act. ``market_data_ops`` must not do it for itself, or a refresh
  runner would be quarantining listings against a calendar nobody qualified; the
  Host resolves the axis and injects it.
- **Universe-versus-SPY composition.** Only the Host may legally combine
  qualified Universe Data with the Portfolio benchmark input path. The
  foundation evaluators stay pure and never import upward.

A Human, an Installed Agent, and External Automation are equal users here, and
all three may submit only a *request*. None may submit a computed result, a
policy hash, or an authority hash: those are what the Host establishes, and a
caller-supplied hash would be the caller asserting the very authority this
module exists to resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal, cast

from alphalattice.capabilities.portfolio_inputs.benchmark import (
    build_portfolio_benchmark_surface,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.returns.execution import open_to_open_simple_return
from alphalattice.foundation.market_data_ops.runtime.diagnostics import (
    UniverseSpyDivergenceDiagnostic,
    UniverseSpyDivergencePolicy,
    evaluate_universe_spy_divergence,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    TradingSessionAuthority,
    build_trading_session_authority,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.shared_kernel.identity import canonical_hash

#: The common US research venues. One spelling, shared by the execution axis and
#: this preflight, so a session is a session everywhere in the product.
COMMON_US_CALENDAR_IDS: tuple[str, str] = ("XNYS", "XNAS")

DEFAULT_UNIVERSE_SPY_DIVERGENCE_POLICY_ID = "UNIVERSE_SPY_ABSOLUTE_0P05"

#: How many recent formation sessions the routine Host preflight covers. Small
#: on purpose: this is a sentinel that must stay affordable on every run, not a
#: historical sweep. Widening it is a Data decision with its own cost, not a
#: default that quietly grows.
ROUTINE_PREFLIGHT_FORMATION_SESSIONS = 5

#: Explicitly installed diagnostic policies. A request names one by id; it never
#: supplies bounds or a hash. No discovery, no dynamic import.
_INSTALLED_DIVERGENCE_POLICIES: dict[str, UniverseSpyDivergencePolicy] = {
    DEFAULT_UNIVERSE_SPY_DIVERGENCE_POLICY_ID: UniverseSpyDivergencePolicy(),
}


class DataTruthPreflightError(ValueError):
    """Stable failure raised before a workspace may be called research-ready."""


class DataTruthScopeUnavailable(DataTruthPreflightError):
    """The workspace cannot yet express a scope to evaluate at all.

    Deliberately a distinct type, because it is the *only* preflight failure a
    caller may treat as "no verdict". A freshly initialized workspace has no
    matured formation axis, so there is genuinely nothing to compare; every
    other failure means the evidence should have existed and the authority
    could not be resolved, which is a blocking fact rather than an absence.
    """


def resolve_trading_session_authority(
    *,
    start: date,
    end: date,
    as_of_timestamp: datetime,
    calendar_ids: tuple[str, ...] = COMMON_US_CALENDAR_IDS,
) -> TradingSessionAuthority:
    """Resolve and seal the ordered common-session axis for one bounded range.

    Reuses ``materialize_calendar_schedule`` -- the same calendar the execution
    axis is built from -- rather than adding a second calendar path. A session
    is admitted only when every named venue agrees it is open with identical
    open and close clocks, so a partial or disagreeing venue day never becomes
    a qualified session.
    """
    if start > end:
        raise DataTruthScopeUnavailable("data_truth_preflight.session_range_reversed")
    calendar = materialize_calendar_schedule(
        calendar_ids,
        start=start,
        end=end,
        as_of_timestamp=as_of_timestamp,
    )
    venue_rows: dict[date, dict[str, dict[str, object]]] = {}
    for row in calendar.to_pylist():
        venue_rows.setdefault(cast(date, row["session_date"]), {})[str(row["calendar_id"])] = row
    sessions: list[date] = []
    for session, venues in sorted(venue_rows.items()):
        if set(venues) != set(calendar_ids):
            continue
        clocks = {
            (
                value["session_open_timestamp"],
                value["session_close_timestamp"],
            )
            for value in venues.values()
        }
        if len(clocks) != 1:
            continue
        sessions.append(session)
    if not sessions:
        raise DataTruthScopeUnavailable("data_truth_preflight.session_axis_empty")
    return build_trading_session_authority(calendar_ids=calendar_ids, sessions=sessions)


@dataclass(frozen=True)
class DataTruthPreflightRequest:
    """What a caller is entitled to assert: scope and an installed policy id.

    Deliberately carries no ``authority_hash`` and no ``policy_hash``. Those are
    result bindings the Host computes after resolving the workspace, the
    calendar, the qualified manifest and the installed policy -- not inputs a
    caller may hand in. Accepting them would let the submitter declare the
    authority the preflight exists to establish.
    """

    workspace: Path
    market_profile_id: str
    formation_sessions: tuple[date, ...]
    divergence_policy_id: str = DEFAULT_UNIVERSE_SPY_DIVERGENCE_POLICY_ID

    def __post_init__(self) -> None:
        """Require a nonempty ordered unique formation axis and a market profile.

        Raises:
            DataTruthPreflightError: Formation scope is invalid or the profile identifier is absent.
        """
        if not self.formation_sessions or self.formation_sessions != tuple(
            sorted(set(self.formation_sessions))
        ):
            raise DataTruthPreflightError("data_truth_preflight.formation_scope_invalid")
        if not self.market_profile_id:
            raise DataTruthPreflightError("data_truth_preflight.market_profile_required")

    @property
    def scope_hash(self) -> str:
        """Bind the declared formation scope to its installed divergence policy identifier.

        Returns:
            Canonical identity of the caller-declared scope without computed authorities.
        """
        return str(
            canonical_hash(
                {
                    "kind": "DataTruthPreflightRequestScope",
                    "market_profile_id": self.market_profile_id,
                    "formation_sessions": [value.isoformat() for value in self.formation_sessions],
                    "divergence_policy_id": self.divergence_policy_id,
                }
            )
        )


@dataclass(frozen=True)
class DataTruthPreflightResult:
    """Typed Host verdict: what blocks, what is merely advisory, under which identities."""

    request_scope_hash: str
    session_authority_hash: str | None
    """``None`` only when the run was blocked before an authority could resolve."""

    divergence_policy_hash: str | None
    divergence: UniverseSpyDivergenceDiagnostic | None
    """Absent when the scope was blocked: evaluating a divergence over sessions
    the calendar does not admit would describe a comparison that never
    legitimately existed."""

    disposition: Literal["RESEARCH_READY", "BLOCKED"]
    blocking_failure_code: str | None
    advisory_codes: tuple[str, ...]
    result_hash: str

    @property
    def blocks_research_ready(self) -> bool:
        """Report whether the deterministic preflight verdict blocks research readiness.

        Returns:
            Whether the disposition is BLOCKED.
        """
        return self.disposition == "BLOCKED"


def run_data_truth_preflight(
    request: DataTruthPreflightRequest,
    *,
    as_of_timestamp: datetime,
) -> DataTruthPreflightResult:
    """Resolve every authority, then evaluate the Data-owned sentinels.

    Calendar/session breach blocks research-ready: a formation session that is
    not a qualified common session means the scope itself is wrong, and no
    downstream evidence built on it can be trusted. Universe-versus-SPY
    divergence never blocks and never attributes -- it says the two sides
    disagree, not which one is wrong, so it surfaces as advisory for review.
    """
    policy = _INSTALLED_DIVERGENCE_POLICIES.get(request.divergence_policy_id)
    if policy is None:
        raise DataTruthPreflightError("data_truth_preflight.divergence_policy_not_installed")

    authority = resolve_trading_session_authority(
        start=min(request.formation_sessions),
        end=max(request.formation_sessions),
        as_of_timestamp=as_of_timestamp,
    )
    off_axis = tuple(
        value for value in request.formation_sessions if value not in authority.session_set
    )
    if off_axis:
        return _seal_result(
            request=request,
            authority=authority,
            policy=policy,
            divergence=None,
            blocking_failure_code="data_truth_preflight.formation_session_not_qualified",
            advisory_codes=(),
        )

    divergence = evaluate_universe_spy_divergence_preflight(
        workspace=request.workspace,
        formation_sessions=request.formation_sessions,
        market_profile_id=request.market_profile_id,
        policy=policy,
    )
    advisory = (
        ("data_truth_preflight.universe_spy_divergence_review",)
        if divergence.classification != "CONSISTENT"
        else ()
    )
    return _seal_result(
        request=request,
        authority=authority,
        policy=policy,
        divergence=divergence,
        blocking_failure_code=None,
        advisory_codes=advisory,
    )


def blocked_preflight_result(
    *,
    request: DataTruthPreflightRequest,
    failure_code: str,
) -> DataTruthPreflightResult:
    """Seal a blocking verdict for a failure that occurred before any authority resolved.

    A preflight that could not establish its authority has still reached a
    conclusion: this workspace may not be called research-ready. Returning that
    as a typed blocking result -- rather than as an absence -- is what stops a
    resolution failure from being indistinguishable from a clean run.
    """
    values: dict[str, object] = {
        "kind": "DataTruthPreflightResult",
        "request_scope_hash": request.scope_hash,
        "session_authority_hash": None,
        "divergence_policy_hash": None,
        "divergence_hash": None,
        "disposition": "BLOCKED",
        "blocking_failure_code": failure_code,
        "advisory_codes": [],
    }
    return DataTruthPreflightResult(
        request_scope_hash=request.scope_hash,
        session_authority_hash=None,
        divergence_policy_hash=None,
        divergence=None,
        disposition="BLOCKED",
        blocking_failure_code=failure_code,
        advisory_codes=(),
        result_hash=canonical_hash(values),
    )


def _seal_result(
    *,
    request: DataTruthPreflightRequest,
    authority: TradingSessionAuthority,
    policy: UniverseSpyDivergencePolicy,
    divergence: UniverseSpyDivergenceDiagnostic | None,
    blocking_failure_code: str | None,
    advisory_codes: tuple[str, ...],
) -> DataTruthPreflightResult:
    disposition: Literal["RESEARCH_READY", "BLOCKED"] = (
        "BLOCKED" if blocking_failure_code is not None else "RESEARCH_READY"
    )
    values = {
        "kind": "DataTruthPreflightResult",
        "request_scope_hash": request.scope_hash,
        "session_authority_hash": authority.authority_hash,
        "divergence_policy_hash": policy.policy_hash,
        "divergence_hash": divergence.diagnostic_hash if divergence is not None else None,
        "disposition": disposition,
        "blocking_failure_code": blocking_failure_code,
        "advisory_codes": list(advisory_codes),
    }
    return DataTruthPreflightResult(
        request_scope_hash=request.scope_hash,
        session_authority_hash=authority.authority_hash,
        divergence_policy_hash=policy.policy_hash,
        divergence=divergence,
        disposition=disposition,
        blocking_failure_code=blocking_failure_code,
        advisory_codes=advisory_codes,
        result_hash=canonical_hash(values),
    )


def evaluate_universe_spy_divergence_preflight(
    *,
    workspace: Path,
    formation_sessions: tuple[date, ...],
    market_profile_id: str = "us-current-index-research",
    policy: UniverseSpyDivergencePolicy | None = None,
) -> UniverseSpyDivergenceDiagnostic:
    """Compose the qualified Universe aggregate against the same-clock SPY path.

    The SPY side is the existing benchmark surface, unchanged. The Universe side
    is the equal-weight mean of the same open-to-open one-session return over
    every qualified listing whose axis covers the session, with membership
    counts folded into the source identity so two runs over different member
    sets cannot share it. Advisory: it attributes nothing.
    """
    if not formation_sessions or formation_sessions != tuple(sorted(set(formation_sessions))):
        raise DataTruthPreflightError("data_truth_preflight.formation_scope_invalid")
    spy_surface = build_portfolio_benchmark_surface(
        workspace=workspace,
        formation_sessions=formation_sessions,
    )
    market = MarketDataRepository(workspace.resolve())
    manifest = market.current_quality_filtered_research_manifest(
        market_profile_id=market_profile_id
    )
    if manifest is None:
        raise DataTruthPreflightError("data_truth_preflight.research_manifest_missing")
    # The manifest is Market Data's; the projected frame is the Feature state's.
    # Reading the second from the first is the same owner confusion the Portfolio
    # benchmark surface carried, and neither call site had been exercised.
    store = FeatureStateRepository(workspace.resolve(), market_data=market)
    per_session: dict[date, list[float]] = {session: [] for session in formation_sessions}
    lineage: list[tuple[str, str, str]] = []
    # One database instance for the whole read: each listing's frame is three
    # short reads, and an instance open per read (three per listing, a
    # metadata read each) was most of this preflight's time.
    with market.database.retain(read_only=True):
        for listing in sorted(manifest.listings, key=lambda item: item.listing_id):
            rows, raw_hash, action_hash = store.projected_feature_frame(
                manifest,
                listing_id=listing.listing_id,
                through=max(formation_sessions) + date.resolution * 10,
                start=min(formation_sessions),
            )
            lineage.append((listing.listing_id, raw_hash, action_hash))
            sessions = tuple(cast(date, row["session_date"]) for row in rows)
            position = {value: index for index, value in enumerate(sessions)}
            for formation in formation_sessions:
                index = position.get(formation)
                if index is None or index + 2 >= len(rows):
                    continue
                per_session[formation].append(
                    open_to_open_simple_return(
                        entry_open=float(rows[index + 1]["open_split_adjusted"]),
                        exit_open=float(rows[index + 2]["open_split_adjusted"]),
                        period_dividend=float(rows[index + 2]["cash_dividend"]),
                    )
                )
    universe_returns: dict[date, float] = {}
    for session in formation_sessions:
        values = per_session[session]
        if not values:
            raise DataTruthPreflightError("data_truth_preflight.session_has_no_qualified_members")
        universe_returns[session] = sum(values) / len(values)
    universe_source_identity = canonical_hash(
        {
            "kind": "UniverseAggregateReturnSource",
            "manifest_revision": manifest.revision_sha256,
            "formation_sessions": formation_sessions,
            "member_counts": {
                session.isoformat(): len(per_session[session]) for session in formation_sessions
            },
            "lineage": lineage,
        }
    )
    return evaluate_universe_spy_divergence(
        universe_returns=universe_returns,
        spy_returns=dict(zip(formation_sessions, spy_surface.simple_returns, strict=True)),
        universe_source_identity=universe_source_identity,
        spy_source_identity=spy_surface.surface_hash,
        policy=policy,
    )


def routine_preflight_formation_scope(
    *,
    as_of_session: date,
    as_of_timestamp: datetime,
    session_count: int = ROUTINE_PREFLIGHT_FORMATION_SESSIONS,
) -> tuple[date, ...]:
    """The most recent formation sessions whose one-session outcome has matured.

    A formation session needs an entry and an exit session after it, so the two
    trailing sessions of the axis can never be formations; taking them would ask
    the diagnostic about a return that has not happened yet.
    """
    authority = resolve_trading_session_authority(
        start=as_of_session - date.resolution * 120,
        end=as_of_session,
        as_of_timestamp=as_of_timestamp,
    )
    matured = authority.sessions[:-2]
    if len(matured) < session_count:
        raise DataTruthScopeUnavailable("data_truth_preflight.matured_session_axis_too_short")
    return tuple(matured[-session_count:])


__all__ = [
    "COMMON_US_CALENDAR_IDS",
    "DEFAULT_UNIVERSE_SPY_DIVERGENCE_POLICY_ID",
    "ROUTINE_PREFLIGHT_FORMATION_SESSIONS",
    "DataTruthPreflightError",
    "DataTruthPreflightRequest",
    "DataTruthPreflightResult",
    "DataTruthScopeUnavailable",
    "blocked_preflight_result",
    "evaluate_universe_spy_divergence_preflight",
    "resolve_trading_session_authority",
    "routine_preflight_formation_scope",
    "run_data_truth_preflight",
]
