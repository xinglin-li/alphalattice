"""Each provider case the realistic replay injects reaches the store through the real client."""

from __future__ import annotations

from datetime import UTC, date, datetime
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
from alphalattice.control.product_host.composition.portfolio_updates import (
    publish_decision_successor,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.publication.projection import (
    project_as_traded_series,
    project_research_series,
)
from alphalattice.foundation.market_data_ops.sources import universe as current_universe
from alphalattice.foundation.market_data_ops.sources.providers import ProviderFetchError
from alphalattice.foundation.market_data_ops.sources.universe import (
    bootstrap_from_candidate_manifest_document,
    candidate_manifest_document,
    membership_fingerprint,
)
from alphalattice.foundation.market_data_ops.storage.rebase import recorded as recorded_rebases
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    continue_issued_observations,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (
    FrozenHistoricalBookRecipe,
    FrozenHistoricalComponentRecipe,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from tests.portfolio_strategy_lab.local_web_support import TEST_PACKAGE
from tests.portfolio_strategy_lab.synthetic_numerical import HASH, build_numerical, prepared_for
from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest
from tests.workspace_maintenance.wire_replay import (
    LISTING,
    ReplayedWorkspace,
    WireReplay,
    _open_stamp,
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


def _observations(n, workspace, through):
    """The stored bars and actions from the book's first session through `through`, as the
    Portfolio's daily observations read them, on the book's own schedule."""
    index = n.sessions.index(through)
    start = n.sessions[n.first - 1]
    bars = workspace.market.raw_bars_by_listing((LISTING,), start=start, through=through)[LISTING]
    actions = workspace.market.actions_by_listing((LISTING,))[LISTING]
    return LocalQAMarketSnapshot.create(
        source_hash=canonical_hash(["wire-replay"]),
        through=through,
        ordered_listing_ids=(LISTING,),
        schedule=n.schedule[n.first - 1 : index + 3],
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
        ("special-dividend", date(2026, 7, 30), 28.0),
    ],
    ids=("new", "late", "corrected", "special"),
)
def test_a_dividend_that_moves_stored_history_waits_for_approval_then_restates(
    tmp_path, request, case, ex, amount
):
    """A new, late or corrected dividend asks the person to approve a full-history audit; once
    approved, the daily stores the provider's amount and its restated adjusted closes."""
    replay, workspace = _replayed(case, tmp_path, request)
    if case == "special-dividend":
        assert amount > _closes(workspace)[replay.days[0]] / 10
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
        ("split-reverse-fifth", date(2026, 7, 30), 0.2),
        ("spin-off-price-adjustment", date(2026, 7, 30), 1.589),
    ],
    ids=("forward", "reverse", "late", "reverse-fifth", "spin-off-price-only"),
)
def test_a_split_waits_for_approval_then_rebases_the_stored_bars(
    tmp_path, request, case, ex, ratio
):
    """An approved split rebases prior bars. Yahoo's fractional spin-off adjusts price only."""
    replay, workspace = _replayed(case, tmp_path, request)
    held = _closes(workspace)
    first, approved = replay.days[1:]
    assert workspace.daily(first).failure_code == APPROVAL
    state = workspace.daily(approved, approved=True)
    assert (state.state, state.failure_code) == ("UPDATED", None)
    assert (ex, "SPLIT", None, ratio) in _actions(workspace)
    if case == "spin-off-price-adjustment":
        assert _actions(workspace) == [(ex, "SPLIT", None, ratio)]
        bars = workspace.market.raw_bars(LISTING)
        actions = workspace.market.actions(LISTING)
        traded = project_as_traded_series(bars, actions, daily_price_basis="split_adjusted")
        previous = next(row for row in traded if row.session_date == replay.days[0])
        current = next(row for row in traded if row.session_date == ex)
        assert (previous.price_adjustment_index, current.price_adjustment_index) == (1.0, ratio)
        assert previous.share_count_index == current.share_count_index == 1.0
        projected = {
            bar.session_date: bar.close_total_return_adjusted
            for bar in project_research_series(bars, actions, daily_price_basis="split_adjusted")
        }
        assert abs(projected[ex] / projected[replay.days[0]] - 1.0) < 0.005
    else:
        before = max(item for item in held if item < ex)
        assert _closes(workspace)[before] == pytest.approx(held[before] / ratio, rel=1e-12)


@pytest.mark.proves("market_data_ops.a_late_split_through_the_wire_continues_the_issued_prefix")
def test_a_split_reported_late_on_an_issued_session_continues_the_book_disclosed(tmp_path, request):
    """A split reported after the session it moved was issued continues the book, its next
    publication disclosing it and settling at the rebased prices."""
    replay, workspace = _replayed("split-late", tmp_path, request)
    issued_through, first, approved = replay.days
    # A one-name book over the replay's one listing.
    component = FrozenHistoricalComponentRecipe(TEST_PACKAGE.component_ids[0], 10_000, "mu.iv0")
    n = build_numerical(
        listing_ids=(LISTING,),
        book_recipe=FrozenHistoricalBookRecipe(
            TEST_PACKAGE.strategy_id, (component,), top_k=1, exit_rank=1
        ),
    )
    ledger = PortfolioLedgerStore(tmp_path / "ledger")
    ledger.publish_decision_checkpoint(n.checkpoint)
    previous, basis = None, None
    for offset, index in enumerate(range(n.first, n.sessions.index(issued_through) + 1)):
        observed = _observations(n, workspace, n.sessions[index])
        ledger.publish_decision_observations(observed)
        previous = publish_decision_successor(
            ledger,
            workspace.market,
            checkpoint=n.checkpoint,
            previous=previous,
            prepared=prepared_for(n, index),
            observed=observed,
            raw=observed,
            basis=basis,
            plan_hash=canonical_hash(offset),
            published_at=evening(n.sessions[index]),
        )
        basis = observed
    chain = ledger.decision_history(n.checkpoint.history_hash)
    stored_bytes = [
        ledger.content.artifact_path(category="decision-updates", content_hash=v.content_hash)
        for v in chain
    ]
    issued_bytes = [path.read_bytes() for path in stored_bytes]
    issued = ledger.load_decision_observation(chain[-1].source_snapshot_hash)
    workspace.daily(first)
    workspace.daily(approved, approved=True)
    raw = _observations(n, workspace, approved)
    continued = continue_issued_observations(issued, raw)
    ledger.publish_decision_observations(raw, continued)

    publish_decision_successor(
        ledger,
        workspace.market,
        checkpoint=n.checkpoint,
        previous=chain[-1],
        prepared=None,
        observed=continued,
        raw=raw,
        basis=issued,
        plan_hash=HASH,
        published_at=evening(approved),
    )

    # The issued prefix continues: the bars before the split rebased as the store rebased them,
    # those from it as issued.
    with workspace.market.database.read_transaction() as read:
        (rebase,) = recorded_rebases(read, (LISTING,))
    ex = date(2026, 7, 28)
    assert (rebase.ex_date, rebase.price_factor) == (ex, 0.5)
    known = {v.session_date: v.close for v in raw.bars}
    for bar in issued.bars:
        moved = next(v for v in continued.bars if v.session_date == bar.session_date)
        scale = rebase.price_factor if bar.session_date < ex else 1.0
        assert moved.close == pytest.approx(bar.close * scale, rel=1e-12)
        assert moved.close == pytest.approx(known[bar.session_date], rel=1e-12)
    # The successor on {checkpoint, parent} discloses the split as the store recorded it.
    stored = ledger.decision_successor(n.checkpoint.history_hash, chain[-1].content_hash)
    disclosure = stored.source_revision.late_split
    assert [
        (v.listing_id, v.effective_date, v.new_shares_per_old_share, v.observed_at)
        for v in disclosure.splits
    ] == [(LISTING, ex, 2.0, rebase.observed_at)]
    assert disclosure.splits[0].rebase_receipt_hash == rebase.receipt_hash
    assert disclosure.affected_publication_hashes == tuple(
        v.content_hash for v in chain if v.observed_through >= ex
    )
    assert disclosure.unheld == ()
    (spanning,) = [
        event
        for publication in chain
        for event in publication.events
        if event.phase == "OUTCOME_SETTLED"
        and event.entry.schedule.entry_session < ex <= event.entry.schedule.holding_end_session
    ]
    (holding,) = disclosure.holdings
    opens = {v.session_date: v.open for v in raw.bars}
    entry, exit_ = opens[spanning.entry.schedule.entry_session], opens[ex]
    assert holding.proposal_hash == spanning.proposal_hash
    assert holding.issued_return == pytest.approx(exit_ / (entry / rebase.price_factor) - 1)
    assert holding.rebased_return == pytest.approx(exit_ / entry - 1)
    assert holding.weight == spanning.entry.weights[0] > 0
    assert holding.book_effect_bps == pytest.approx(
        holding.weight * (holding.rebased_return - holding.issued_return) * 10_000
    )
    # The prior open entry settles: its issued entry, its exit from the rebased prices.
    open_entry = chain[-1].active_entry
    (settled,) = [
        v
        for v in stored.events
        if v.phase == "OUTCOME_SETTLED" and v.proposal_hash == open_entry.proposal_hash
    ]
    schedule, weight = open_entry.entry.schedule, open_entry.entry.weights[0]
    assert settled.entry == open_entry.entry
    assert settled.gross_return == pytest.approx(
        weight * (opens[schedule.holding_end_session] / opens[schedule.entry_session] - 1)
    )
    # The issued publications keep every stored byte.
    assert [path.read_bytes() for path in stored_bytes] == issued_bytes
    assert ledger.decision_history(n.checkpoint.history_hash)[:-1] == chain


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


def _duplicate_chart_row(replay, order, monkeypatch):
    original = replay.chart

    def chart(symbol, query):
        answer = original(symbol, query)
        if symbol != "AAPL" or replay.today != replay.days[1] or not answer["timestamp"]:
            return answer
        answer["timestamp"].append(answer["timestamp"][-1])
        indicators = answer["indicators"]
        series = {
            **indicators["quote"][0],
            "adjclose": indicators["adjclose"][0]["adjclose"],
        }
        for name, values in series.items():
            value = values[-1]
            other = value if order == "identical" or name == "volume" else value * 1.08
            if order == "conflict-first":
                values.insert(len(values) - 1, other)
            else:
                values.append(other)
        return answer

    monkeypatch.setattr(replay, "chart", chart)


def _assert_conflict_detail(detail, session):
    description, _, fields = detail.rpartition(": ")
    assert "AAPL" in description and session.isoformat() in description, detail
    assert set(fields.split(", ")) == {"adjclose", "close", "high", "low", "open"}


def _assert_duplicate_chart_reads(replay, workspace, first_day):
    replay.today = first = replay.days[1]
    fetch_daily = partial(workspace.provider.fetch_daily, ("AAPL",), start=first, end=first)
    if first_day is None:
        rows = fetch_daily()["AAPL"]
        assert [row["session_date"] for row in rows] == [first.isoformat()]
        assert rows[0]["close"] == pytest.approx(replay.history("AAPL", first)[first]["close"])
        return

    for fetch in (
        fetch_daily,
        partial(
            workspace.provider.fetch_hydration,
            listing_id=LISTING,
            provider_symbol="AAPL",
            start=replay.start,
            end=first,
        ),
        partial(
            workspace.provider.fetch_adjusted_close_history,
            listing_id=LISTING,
            provider_symbol="AAPL",
            start=first,
            end=first,
        ),
        partial(
            workspace.provider.fetch_action_history,
            listing_id=LISTING,
            provider_symbol="AAPL",
            start=first,
            end=first,
        ),
    ):
        with pytest.raises(ProviderFetchError) as refused:
            fetch()
        error = refused.value
        assert (error.code, error.retryable) == (first_day, True)
        _assert_conflict_detail(str(error), first)


@pytest.mark.proves("market_data_ops.provider_outages_end_as_their_failure_through_the_wire")
@pytest.mark.parametrize(
    ("case", "first_day", "transport_retry", "duplicate"),
    [
        ("outage-rate-limited", None, ([2.0], 1), None),
        ("outage-server-error", None, ([], 1), None),
        # A transport timeout ends as a permanent fetch failure, not a retryable timeout: the
        # product's error mapping reads no timeout in libcurl's message, a defect still open.
        ("outage-timeout", "data.provider_fetch_failed", ([], 0), None),
        ("chart-empty-200", "data.empty_payload", None, None),
        ("chart-nan-close", "data.provider_fetch_failed", None, None),
        ("chart-zero-close", "data.invalid_adjusted_close_payload", None, None),
        ("cash-acquisition-delists", None, None, None),
        (None, "data.conflicting_session_rows", None, "conflict-first"),
        (None, "data.conflicting_session_rows", None, "conflict-last"),
        (None, None, None, "identical"),
    ],
    ids=(
        "rate-limited",
        "server-error",
        "timeout",
        "empty-http-200",
        "nan-close",
        "zero-close",
        "cash-acquisition",
        "conflict-first",
        "conflict-last",
        "identical",
    ),
)
def test_a_provider_outage_ends_as_its_failure_or_its_retry(
    tmp_path, monkeypatch, request, case, first_day, transport_retry, duplicate
):
    """Transport retries and bad chart answers preserve valid history. An unavailable delisted
    chart leaves the cash settlement and active manifest intact."""
    slept: list[float] = []
    backed_off = partialmethod(ProviderFetchError.backed_off, sleep=slept.append)
    monkeypatch.setattr(ProviderFetchError, "backed_off", backed_off)
    replay, workspace = _replayed(case, tmp_path, request)
    replay.requests.clear()
    before = _closes(workspace)
    initial, first, later = replay.days
    market = workspace.market
    if case == "cash-acquisition-delists":
        candidate = {"listing_ids": [LISTING]}
        market.admit_current_universe_onboarding(
            acquisition_manifest(),
            onboarding_id="cash-acquisition-setup",
            candidate_manifest_hash=canonical_hash(candidate),
            candidate_manifest_document=candidate,
            history_start=replay.start,
            as_of_session=initial,
            calendar_by_listing_id={LISTING: "XNAS"},
            observed_at=evening(initial),
            requested_listing_ids=(LISTING,),
        )
        market.complete_current_universe_onboarding(
            onboarding_id="cash-acquisition-setup",
            research_manifest=workspace.manifest,
            quality_admission_hash=canonical_hash([LISTING]),
            observed_at=evening(initial),
        )
    if duplicate is not None:
        _duplicate_chart_row(replay, duplicate, monkeypatch)
        _assert_duplicate_chart_reads(replay, workspace, first_day)
    state = workspace.daily(first)
    assert (state.state, state.failure_code) == (
        "UPDATED" if first_day is None else "FAILED",
        first_day,
    )
    if transport_retry is not None:
        asked = [path for _host, path, _query in replay.requests]
        assert (slept, asked.count("/v1/test/getcrumb")) == transport_retry
    if first_day is not None:
        assert _closes(workspace) == before
        if duplicate is not None:
            _assert_conflict_detail(state.change_document["failure_cause"]["detail"], first)

    if case == "cash-acquisition-delists":
        settled = _closes(workspace)
        assert settled == {**before, first: 205.0}
        refused = workspace.daily(later)
        assert (refused.state, refused.failure_code) == ("FAILED", "data.empty_payload")
        active = market.current_quality_filtered_research_manifest(
            market_profile_id=workspace.manifest.profile.market_profile_id,
        )
        assert active is not None
        assert tuple(listing.listing_id for listing in active.listings) == (LISTING,)
        assert _closes(workspace) == settled
    else:
        recovered = workspace.daily(later)
        assert (recovered.state, recovered.failure_code) == ("UPDATED", None)
        assert _closes(workspace)[later] == pytest.approx(
            replay.history("AAPL", later)[later]["close"]
        )


@pytest.mark.proves("market_data_ops.provider_sessions_follow_the_calendar_through_the_wire")
@pytest.mark.parametrize(
    "case", (None, "chart-unfinished-next-session"), ids=("calendar", "partial-bar")
)
def test_the_stored_sessions_are_the_calendars_with_its_half_days_and_no_holidays(
    tmp_path, request, case
):
    """The store follows the exchange calendar, including half days, holidays and in-progress
    sessions. The daylight-saving change preserves the local session dates."""
    replay, workspace = _replayed(case, tmp_path, request)
    stored = sorted(_closes(workspace))
    assert stored == [day for day in replay.sessions if day <= replay.days[0]]
    assert {date(2025, 11, 28), date(2025, 12, 24)} <= set(stored)
    assert not {date(2025, 11, 27), date(2025, 12, 25), date(2026, 1, 1)} & set(stored)
    friday, monday = date(2025, 10, 31), date(2025, 11, 3)
    closes = _closes(workspace)
    source = replay.history("AAPL", replay.days[0])
    assert datetime.fromtimestamp(_open_stamp(friday), UTC).hour == 13
    assert datetime.fromtimestamp(_open_stamp(monday), UTC).hour == 14
    assert [closes[day] for day in (friday, monday)] == pytest.approx(
        [source[day]["close"] for day in (friday, monday)]
    )
    if case is not None:
        completed, unfinished = replay.days[1:]
        midday = datetime(unfinished.year, unfinished.month, unfinished.day, 16, tzinfo=UTC)
        event = workspace.daily(completed, observed_at=midday)
        assert (event.state, event.failure_code) == ("UPDATED", None)
        stored = _closes(workspace)
        assert stored[completed] == pytest.approx(
            replay.history("AAPL", completed)[completed]["close"]
        )
        assert unfinished not in stored


@pytest.mark.proves("market_data_ops.membership_changes_through_the_universe_fetcher")
@pytest.mark.parametrize(
    ("case", "candidates", "added", "removed", "waits"),
    (
        ("membership-change", ("AAPL", "NVDA"), ("NVDA",), ("MSFT",), [7.0]),
        ("ticker-change", ("AAPX", "MSFT"), ("AAPX",), ("AAPL",), []),
    ),
    ids=("index-membership", "ticker-change"),
)
def test_an_index_membership_change_is_stored_as_a_pending_manifest_update(
    tmp_path, monkeypatch, request, case, candidates, added, removed, waits
):
    """Index changes go through the real fetcher and parser. An AAPL to AAPX rename carries the
    same chart history and stays a pending update until a person approves it."""
    replay, workspace = _replayed(case, tmp_path, request)
    before = _closes(workspace)
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
    assert decision.status is WorkspaceReadinessStatus.MANIFEST_UPDATE_PENDING
    assert (record.status, pending.candidate_symbols) == ("MANIFEST_UPDATE_PENDING", candidates)
    assert (stored.lifecycle, stored.additions, stored.removals) == ("PROPOSED", added, removed)
    assert record.active_membership_fingerprint == membership_fingerprint(active.source_manifest)
    assert active.candidate_symbols == ("AAPL", "MSFT")
    assert slept == waits

    if case == "ticker-change":
        assert record.active_manifest_id == workspace.manifest.manifest_id
        assert record.active_manifest_revision == workspace.manifest.revision_sha256
        assert _closes(workspace) == before
        replacement = workspace.provider.fetch_hydration(
            listing_id=LISTING, provider_symbol="AAPX", start=replay.start, end=second
        )
        closes = {
            date.fromisoformat(row["session_date"]): row["close"] for row in replacement.daily_rows
        }
        assert closes.pop(second) == pytest.approx(replay.history("AAPL", second)[second]["close"])
        assert closes == pytest.approx(before)
        assert any(path == "/v8/finance/chart/AAPX" for _, path, _ in replay.requests)
