"""Each provider case the realistic replay injects reaches the store through the real client."""

from __future__ import annotations

from datetime import date
from functools import partial, partialmethod
from pathlib import Path

import pytest

from alphalattice.capabilities.data import universe as data_universe
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.capabilities.domain_contracts.project_layout import resolve_playpen_root
from alphalattice.control.data_platform.readiness import (
    WorkspaceReadinessGate,
    WorkspaceReadinessStatus,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import planned_local_qa_schedule
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources import universe as current_universe
from alphalattice.foundation.market_data_ops.sources.providers import ProviderFetchError
from alphalattice.foundation.market_data_ops.sources.universe import (
    bootstrap_from_candidate_manifest_document,
    candidate_manifest_document,
    membership_fingerprint,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    SplitBasisRevision,
    continue_issued_observations,
)
from tests.workspace_maintenance.wire_replay import (
    LISTING,
    ReplayedWorkspace,
    WireReplay,
    evening,
    install,
)

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/sources/providers.py"
PROFILE = (
    resolve_playpen_root(Path(__file__))
    / "config"
    / "market-profiles"
    / "us-current-index-research.yaml"
)
APPROVAL = "data.full_history_audit_approval_required"
"""The failure a daily records when a change reaches back into stored history."""


def _replayed(case, tmp_path, request, **load):
    replay = WireReplay.load(case, **load)
    install(request, replay, tmp_path)
    return replay, ReplayedWorkspace.first_use(replay, tmp_path)


def _actions(workspace):
    return [
        (item.effective_date, item.action_kind, item.cash_amount, item.new_shares_per_old_share)
        for item in workspace.market.actions(LISTING)
    ]


def _closes(workspace):
    return {bar.session_date: bar.close for bar in workspace.market.raw_bars(LISTING)}


def _adjusted(workspace, through):
    return {
        point.session_date: point.adjusted_close
        for point in workspace.market.provider_adjusted_closes(LISTING, through=through)
    }


def _observations(workspace, through):
    """The stored bars and actions through `through`, as the Portfolio's daily observations read
    them (the QA reader's own store reads, without its research-manifest epoch)."""
    start = workspace.replay.start
    bars = workspace.market.raw_bars_by_listing((LISTING,), start=start, through=through)[LISTING]
    actions = workspace.market.actions_by_listing((LISTING,))[LISTING]
    return LocalQAMarketSnapshot.create(
        source_hash=canonical_hash(["wire-replay"]),
        through=through,
        ordered_listing_ids=(LISTING,),
        schedule=planned_local_qa_schedule(start, through),
        bars=tuple(sorted(bars, key=lambda bar: bar.session_date)),
        actions=tuple(item for item in actions if item.effective_date <= through),
    )


@pytest.mark.proves("market_data_ops.provider_actions_reach_the_store_through_the_wire")
def test_a_dividend_the_provider_already_shows_is_kept_through_each_daily(tmp_path, request):
    """A dividend the provider shows from the first use on is stored once and kept by each daily
    with no approval asked, the first daily's chart followed through the provider's redirect."""
    replay, workspace = _replayed("dividend-stored", tmp_path, request)
    replay.requests.clear()
    states = [workspace.daily(day) for day in replay.days[1:]]
    assert [(state.state, state.failure_code) for state in states] == [("UPDATED", None)] * 2
    charts = [(host, query) for host, path, query in replay.requests if "/chart/" in path]
    assert [host for host, _query in charts[:2]] == [
        "query2.finance.yahoo.com",
        "query1.finance.yahoo.com",
    ]
    assert charts[0][1] == charts[1][1] and "crumb" in charts[0][1]
    assert _actions(workspace) == [(date(2026, 5, 12), "CASH_DIVIDEND", 0.26, None)]


@pytest.mark.proves("market_data_ops.provider_actions_reach_the_store_through_the_wire")
@pytest.mark.parametrize(
    ("case", "ex", "amount"),
    [
        ("dividend-new", date(2026, 7, 30), 0.26),
        ("dividend-late", date(2026, 7, 28), 0.26),
        ("dividend-corrected", date(2026, 7, 21), 0.31),
    ],
    ids=("new", "late", "corrected"),
)
def test_a_dividend_that_moves_stored_history_waits_for_approval_then_restates(
    tmp_path, request, case, ex, amount
):
    """A new, late or corrected dividend asks the person to approve a full-history audit; once
    approved, the daily stores the provider's amount and its restated adjusted closes."""
    replay, workspace = _replayed(case, tmp_path, request)
    first, approved = replay.days[1:]
    assert workspace.daily(first).failure_code == APPROVAL
    state = workspace.daily(approved, approved=True)
    assert (state.state, state.failure_code) == ("UPDATED", None)
    assert (ex, "CASH_DIVIDEND", amount, None) in _actions(workspace)
    sent = replay.history("AAPL", approved)
    before = max(item for item in sent if item < ex)
    stored = _adjusted(workspace, approved)
    assert stored[before] == pytest.approx(sent[before]["adjclose"], rel=1e-12)


@pytest.mark.proves("market_data_ops.provider_actions_reach_the_store_through_the_wire")
@pytest.mark.parametrize(
    ("case", "ex", "ratio"),
    [
        ("split-forward", date(2026, 7, 30), 4.0),
        ("split-reverse", date(2026, 7, 30), 0.5),
        ("split-late", date(2026, 7, 28), 2.0),
    ],
    ids=("forward", "reverse", "late"),
)
def test_a_split_waits_for_approval_then_rebases_the_stored_bars(
    tmp_path, request, case, ex, ratio
):
    """A forward, reverse or late-reported split asks for approval; once approved, it is stored
    and every bar before it is rebased by its ratio."""
    replay, workspace = _replayed(case, tmp_path, request)
    held = _closes(workspace)
    first, approved = replay.days[1:]
    assert workspace.daily(first).failure_code == APPROVAL
    assert workspace.daily(approved, approved=True).failure_code is None
    assert (ex, "SPLIT", None, ratio) in _actions(workspace)
    before = max(item for item in held if item < ex)
    assert _closes(workspace)[before] == pytest.approx(held[before] / ratio, rel=1e-12)


@pytest.mark.proves("market_data_ops.a_late_split_through_the_wire_refuses_the_continuation")
def test_a_split_reported_late_on_an_issued_session_is_refused_by_the_continuation(
    tmp_path, request
):
    """A split the provider reports after the session it moved was issued cannot continue the
    issued observations: the continuation refuses the split basis revision."""
    replay, workspace = _replayed("split-late", tmp_path, request)
    issued_through, first, approved = replay.days
    issued = _observations(workspace, issued_through)
    workspace.daily(first)
    workspace.daily(approved, approved=True)
    current = _observations(workspace, approved)
    with pytest.raises(SplitBasisRevision):
        continue_issued_observations(issued, current)


@pytest.mark.proves("market_data_ops.provider_actions_reach_the_store_through_the_wire")
def test_several_actions_in_one_history_are_each_stored_in_the_latest_units(tmp_path, request):
    """Two dividends around a split in one history are each stored, the dividend before the split
    in the units after it, and each daily keeps them with no approval asked."""
    replay, workspace = _replayed("actions-in-one-history", tmp_path, request)
    states = [workspace.daily(day) for day in replay.days[1:]]
    assert [state.failure_code for state in states] == [None, None]
    assert _actions(workspace) == [
        (date(2026, 2, 10), "CASH_DIVIDEND", pytest.approx(0.25 / 3, rel=1e-12), None),
        (date(2026, 3, 16), "SPLIT", None, 3.0),
        (date(2026, 5, 12), "CASH_DIVIDEND", 0.09, None),
    ]


@pytest.mark.proves("market_data_ops.provider_rerounding_stands_through_the_wire")
def test_a_provider_rerounding_within_the_rules_precision_leaves_the_stored_closes(
    tmp_path, request
):
    """Adjusted closes the provider sends again re-rounded by about 4e-7 stand as stored, and the
    daily discloses the difference it kept."""
    replay, workspace = _replayed("rerounding", tmp_path, request)
    held = _adjusted(workspace, replay.days[0])
    state = workspace.daily(replay.days[1])
    assert state.failure_code is None
    assert state.change_document["adjusted_precision_kept"] == pytest.approx(4e-7, rel=1e-3)
    after = _adjusted(workspace, replay.days[1])
    assert {day: after[day] for day in held} == held


@pytest.mark.proves("market_data_ops.provider_outages_end_as_their_failure_through_the_wire")
@pytest.mark.parametrize(
    ("case", "first_day", "waits", "crumbs"),
    [
        ("outage-rate-limited", None, [2.0], 1),
        ("outage-server-error", None, [], 1),
        # A transport timeout ends as a permanent fetch failure, not a retryable timeout: the
        # product's error mapping reads no timeout in libcurl's message, a defect still open.
        ("outage-timeout", "data.provider_fetch_failed", [], 0),
    ],
    ids=("rate-limited", "server-error", "timeout"),
)
def test_a_provider_outage_ends_as_its_failure_or_its_retry(
    tmp_path, monkeypatch, request, case, first_day, waits, crumbs
):
    """A rate limit is retried after the product's backoff and yfinance's other cookie strategy, a
    server error after that strategy alone, and a timed-out transfer fails its day; the next day
    updates."""
    slept: list[float] = []
    backed_off = partialmethod(ProviderFetchError.backed_off, sleep=slept.append)
    monkeypatch.setattr(ProviderFetchError, "backed_off", backed_off)
    replay, workspace = _replayed(case, tmp_path, request)
    replay.requests.clear()
    assert workspace.daily(replay.days[1]).failure_code == first_day
    asked = [path for _host, path, _query in replay.requests]
    assert (slept, asked.count("/v1/test/getcrumb")) == (waits, crumbs)
    assert workspace.daily(replay.days[2]).failure_code is None


@pytest.mark.proves("market_data_ops.provider_sessions_follow_the_calendar_through_the_wire")
def test_the_stored_sessions_are_the_calendars_with_its_half_days_and_no_holidays(
    tmp_path, request
):
    """A first use through the replay stores exactly the exchange calendar's sessions: its half
    days are stored and its holidays are not."""
    replay, workspace = _replayed(None, tmp_path, request)
    stored = sorted(_closes(workspace))
    assert stored == [day for day in replay.sessions if day <= replay.days[0]]
    assert {date(2025, 11, 28), date(2025, 12, 24)} <= set(stored)
    assert not {date(2025, 11, 27), date(2025, 12, 25), date(2026, 1, 1)} & set(stored)


@pytest.mark.proves("market_data_ops.membership_changes_through_the_universe_fetcher")
def test_an_index_membership_change_is_stored_as_a_pending_manifest_update(
    tmp_path, monkeypatch, request
):
    """A name leaving one index and another joining one, read through the real fetcher and parser
    after the source's Retry-After is waited out, are stored as the workspace's pending manifest
    update; the active membership stays until the person approves."""
    replay, workspace = _replayed("membership-change", tmp_path, request)
    slept: list[float] = []
    fetch = partial(
        data_universe.fetch_universe_source, fetcher=replay.universe_fetch, sleep=slept.append
    )
    monkeypatch.setattr(current_universe, "fetch_universe_source", fetch)
    market = workspace.market
    gate = WorkspaceReadinessGate(
        market_data=market,
        feature_state=FeatureStateRepository(market.database, market_data=market),
        panel_state=PanelStateRepository(market.database, market_data=market),
        profile_path=PROFILE,
        provider=workspace.provider,
    )
    first, second = replay.days[:2]
    active = current_universe.discover_current_universe_candidates(observed_at=evening(first))
    market.readiness.save(
        market_profile_id=gate.market_profile_id,
        status=WorkspaceReadinessStatus.RESEARCH_READY.value,
        active_manifest_id=workspace.manifest.manifest_id,
        active_manifest_revision=workspace.manifest.revision_sha256,
        active_membership_fingerprint=membership_fingerprint(active.source_manifest),
        active_candidate_manifest_document=candidate_manifest_document(active.source_manifest),
        pending_membership_fingerprint=None,
        pending_candidate_manifest_document=None,
        last_checked_at=evening(first),
        last_changed_at=evening(first),
        failure_code=None,
        observed_at=evening(first),
    )
    replay.today = second
    decision = gate.refresh_sources_if_due(observed_at=evening(second))

    record = market.readiness.load(gate.market_profile_id)
    assert record is not None and decision.proposal is not None
    pending = bootstrap_from_candidate_manifest_document(record.pending_candidate_manifest_document)
    stored = market.manifest_transition(decision.proposal.transition_id)
    assert (record.status, pending.candidate_symbols) == (
        "MANIFEST_UPDATE_PENDING",
        ("AAPL", "NVDA"),
    )
    assert (stored.lifecycle, stored.additions, stored.removals) == (
        "PROPOSED",
        ("NVDA",),
        ("MSFT",),
    )
    assert record.active_membership_fingerprint == membership_fingerprint(active.source_manifest)
    assert active.candidate_symbols == ("AAPL", "MSFT")
    assert slept == [7.0]
