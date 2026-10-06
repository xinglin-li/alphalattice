"""What the Risk surfaces actually observed, stated in the neutral vocabulary.

A thin resolver, and deliberately only that. Every relation is checked by
``capabilities.causal_inputs.admission``; this module owns one thing the shared
owner cannot know -- which exchange instants the Risk arithmetic really reaches.

Risk had no temporal vocabulary at all before this. A published covariance
carried its formation session and said nothing about when its information was
knowable, so "is this causal" could only be answered by reading the estimator.
The answers below are read off that arithmetic rather than inferred from a row
label, because the label and the observation genuinely differ here.

**The return surface's row ``T`` is not a fact about ``T``'s close.**
``derive_risk_return_rows`` pairs consecutive required sessions and computes
``open_to_open_log_return(entry_open=open(T-1), exit_open=open(T))``. Its newest
input is therefore the *open* of ``T``, some six and a half hours before the
close a decision is taken at. Reporting ``close(T)`` because the row is labelled
``T`` is precisely the error ``observed_through`` exists to make unstateable.

**The covariance at formation ``T`` ends at that same open.**
``build_development_covariance_surface`` slices
``returns[end - REQUIRED_LOOKBACK_SESSIONS : end + 1]`` where ``end`` is the
formation's own position, so the estimator sees rows ``T-314 .. T`` inclusive and
nothing newer. The row at ``end + 1`` is read too, but it is passed to
``evaluate_formation`` as the realised outcome and never to ``adapter.estimate``
-- an outcome, not an input, and it is classified as one here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    CausalInputAuthority,
    SessionAnchor,
    TemporalAdmissionError,
)
from alphalattice.investment.risk_research.experiments.window import REQUIRED_LOOKBACK_SESSIONS

RISK_RETURN_OBSERVATION_SEMANTICS: Final = (
    "Row T is open_to_open_log_return(entry_open=open(T-1), exit_open=open(T), "
    "period_dividend=dividend(T)). The newest instant it reflects is the official "
    "open of T, not its close. Unit is a log total return; the execution owner's "
    "realised outcome is a simple return, and neither is changed here."
)

RISK_COVARIANCE_OBSERVATION_SEMANTICS: Final = (
    "The estimate for formation T consumes return rows T-314..T inclusive, whose "
    "newest input is open(T). The row at T+1 is read only by evaluate_formation "
    "as a realised outcome and never reaches adapter.estimate."
)

RISK_READINESS_BUDGET: Final = "DEVELOPMENT_RISK_COVARIANCE_READINESS_BUDGET"
RISK_READINESS_BUDGET_MINUTES: Final = 30

_READINESS_RATIONALE: Final = (
    "Installed development operational budget, not a measurement. This build "
    "persists no runtime receipt for a covariance estimation, so there is "
    "nothing to bind a MEASURED_RUNTIME claim to. R0 and R1 took 166 s and "
    "181 s for 1,264 formations on one large development machine, which is "
    "context and not evidence about other hardware. Read this as the interval "
    "the strategy allots for the estimate."
)

INSTALLED_RISK_READINESS_BUDGETS: Final[Mapping[str, tuple[int, str]]] = {
    RISK_READINESS_BUDGET: (RISK_READINESS_BUDGET_MINUTES, _READINESS_RATIONALE),
}


def resolve_risk_readiness_policy(handle: str) -> AnchoredInstantPolicy:
    """The named, installed interval a covariance estimate is allotted.

    ``INSTALLED_OPERATIONAL_POLICY`` rather than ``MEASURED_RUNTIME``, and the
    distinction is the point. A per-formation cost divided out of one campaign
    wall time on one 32-core machine is not a fact about a desktop, and dividing
    a total by a formation count without binding the parallelism that produced
    it is not a latency at all. Until a runtime receipt is persisted -- estimator
    identity, capacity profile, thread policy, formation count, measured wall
    time -- this is a budget somebody chose, and it says so.
    """

    installed = INSTALLED_RISK_READINESS_BUDGETS.get(handle)
    if installed is None:
        raise TemporalAdmissionError("risk_research.readiness_budget_not_installed")
    minutes, rationale = installed
    return AnchoredInstantPolicy.create(
        policy_id=handle,
        basis="INSTALLED_OPERATIONAL_POLICY",
        anchor=SessionAnchor(offset_sessions=0, event="OFFICIAL_CLOSE"),
        minutes_after_anchor=minutes,
        rationale=rationale,
    )


def risk_covariance_authority(
    *,
    surface_hash: str,
    owner_identity_hash: str,
    input_id: str,
    readiness_handle: str,
    lookback_sessions: int = REQUIRED_LOOKBACK_SESSIONS,
) -> CausalInputAuthority:
    """One development covariance surface's clock.

    ``observation_start`` reaches one session further back than the lookback
    because the oldest *row* in the window is itself an interval: row
    ``T-lookback`` opens at ``T-lookback-1``. Counting rows and counting observed
    instants differ by exactly one here, and the difference is the kind of
    off-by-one that a session label hides.

    ``readiness_handle`` has no default. What a covariance costs to compute is
    the term that decides whether it clears the submission deadline, and this
    build has no runtime receipt to derive it from, so it is a budget a caller
    must name rather than a number this module quietly supplies.
    """

    return CausalInputAuthority.create(
        input_id=input_id,
        input_kind="RISK_COVARIANCE",
        temporal_usage="DECISION_INPUT",
        surface_hash=surface_hash,
        owner_authority_id="risk_research.experiments.development",
        owner_identity_hash=owner_identity_hash,
        observation_start=AnchoredInstantPolicy.exchange_event(
            offset_sessions=-(lookback_sessions + 1),
            event="OFFICIAL_OPEN",
            policy_id="RISK_ESTIMATION_WINDOW_START",
        ),
        observed_through=AnchoredInstantPolicy.exchange_event(
            offset_sessions=0, event="OFFICIAL_OPEN", policy_id="RISK_ESTIMATION_WINDOW_END"
        ),
        source_available=AnchoredInstantPolicy.exchange_event(
            offset_sessions=0, event="OFFICIAL_CLOSE", policy_id="PROVIDER_DAILY_BAR_CLOSE"
        ),
        derived_ready=resolve_risk_readiness_policy(readiness_handle),
        observation_semantics=RISK_COVARIANCE_OBSERVATION_SEMANTICS,
        point_in_time_disposition="CURRENT_MEMBERSHIP_BACKFILLED",
    )


__all__ = [
    "INSTALLED_RISK_READINESS_BUDGETS",
    "RISK_COVARIANCE_OBSERVATION_SEMANTICS",
    "RISK_READINESS_BUDGET",
    "RISK_READINESS_BUDGET_MINUTES",
    "RISK_RETURN_OBSERVATION_SEMANTICS",
    "resolve_risk_readiness_policy",
    "risk_covariance_authority",
]
