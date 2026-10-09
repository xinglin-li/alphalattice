"""Pytest entry point for the runnable Local-first market-data operations case."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from alphalattice.foundation.market_data_ops.runtime.diagnostics import (
    evaluate_universe_spy_divergence,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import (
    KnownCorporateActionExpectation,
    build_trading_session_authority,
    evaluate_price_action_integrity_sentinels,
)


def _bar(session: date, *, close: float = 100.0) -> RawDailyBar:
    return RawDailyBar(
        listing_id="listing-aapl",
        provider="fixture",
        session_date=session,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=1_000_000,
    )


def test_price_action_sentinels_anchor_known_events_and_quarantine_extremes() -> None:
    """Price action sentinels anchor known events and quarantine extremes."""

    split_session = date(2020, 8, 31)
    dividend_session = date(2020, 8, 7)
    sessions = (
        dividend_session,
        date(2020, 8, 10),
        split_session,
        date(2020, 9, 1),
    )
    bars = tuple(_bar(session) for session in sessions)
    actions = (
        CorporateActionEvent(
            "listing-aapl", "fixture", split_session, "SPLIT", new_shares_per_old_share=4.0
        ),
        CorporateActionEvent(
            "listing-aapl", "fixture", dividend_session, "CASH_DIVIDEND", cash_amount=0.205
        ),
    )
    goldens = (
        KnownCorporateActionExpectation("listing-aapl", split_session, "SPLIT", 4.0),
        KnownCorporateActionExpectation("listing-aapl", dividend_session, "CASH_DIVIDEND", 0.205),
    )

    anchored = evaluate_price_action_integrity_sentinels(
        bars=bars,
        actions=actions,
        provider="fixture",
        as_of=date(2020, 9, 2),
        trading_sessions=build_trading_session_authority(
            calendar_ids=("XNAS", "XNYS"), sessions=sessions
        ),
        known_actions=goldens,
    )
    assert anchored.disposition == "ANCHORED"
    assert anchored.findings == ()
    replay = evaluate_price_action_integrity_sentinels(
        bars=bars,
        actions=actions,
        provider="fixture",
        as_of=date(2020, 9, 2),
        trading_sessions=build_trading_session_authority(
            calendar_ids=("XNAS", "XNYS"), sessions=sessions
        ),
        known_actions=goldens,
    )
    assert replay.report_hash == anchored.report_hash

    missing = evaluate_price_action_integrity_sentinels(
        bars=bars,
        actions=actions[1:],
        provider="fixture",
        as_of=date(2020, 9, 2),
        trading_sessions=build_trading_session_authority(
            calendar_ids=("XNAS", "XNYS"), sessions=sessions
        ),
        known_actions=goldens,
    )
    assert missing.disposition == "QUARANTINE_REVIEW_REQUIRED"
    assert {item.code for item in missing.findings} == {"data.sentinel.golden_action_missing"}

    distorted = evaluate_price_action_integrity_sentinels(
        bars=bars,
        actions=(
            CorporateActionEvent(
                "listing-aapl", "fixture", split_session, "SPLIT", new_shares_per_old_share=2.0
            ),
            actions[1],
        ),
        provider="fixture",
        as_of=date(2020, 9, 2),
        trading_sessions=build_trading_session_authority(
            calendar_ids=("XNAS", "XNYS"), sessions=sessions
        ),
        known_actions=goldens,
    )
    assert {item.code for item in distorted.findings} == {"data.sentinel.golden_action_mismatch"}

    extreme = evaluate_price_action_integrity_sentinels(
        bars=bars,
        actions=(
            actions[0],
            CorporateActionEvent(
                "listing-aapl", "fixture", dividend_session, "CASH_DIVIDEND", cash_amount=30.0
            ),
        ),
        provider="fixture",
        as_of=date(2020, 9, 2),
        trading_sessions=build_trading_session_authority(
            calendar_ids=("XNAS", "XNYS"), sessions=sessions
        ),
        known_actions=(goldens[0],),
    )
    assert {item.code for item in extreme.findings} == {"data.sentinel.extreme_dividend_ratio"}

    off_session = evaluate_price_action_integrity_sentinels(
        bars=(*bars, _bar(date(2020, 8, 8))),
        actions=actions,
        provider="fixture",
        as_of=date(2020, 9, 2),
        trading_sessions=build_trading_session_authority(
            calendar_ids=("XNAS", "XNYS"), sessions=sessions
        ),
        known_actions=goldens,
    )
    assert {item.code for item in off_session.findings} == {"data.sentinel.bar_session_not_trading"}

    degenerate = evaluate_price_action_integrity_sentinels(
        bars=(bars[0],),
        actions=(),
        provider="fixture",
        as_of=dividend_session,
        trading_sessions=build_trading_session_authority(
            calendar_ids=("XNAS", "XNYS"), sessions=(dividend_session,)
        ),
    )
    assert degenerate.disposition == "ANCHORED"


def test_universe_spy_divergence_is_advisory_and_exactly_aligned() -> None:
    """Placebo, breach, and misalignment: the three behaviours the sentinel owns."""

    axis = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    same = {session: 0.004 for session in axis}
    placebo = evaluate_universe_spy_divergence(
        universe_returns=same,
        spy_returns=dict(same),
        universe_source_identity="u" * 64,
        spy_source_identity="s" * 64,
    )
    assert placebo.classification == "CONSISTENT"
    assert placebo.attribution == "UNATTRIBUTED"
    assert placebo.flagged_sessions == ()

    breached = evaluate_universe_spy_divergence(
        universe_returns={**same, axis[1]: 0.09},
        spy_returns=dict(same),
        universe_source_identity="u" * 64,
        spy_source_identity="s" * 64,
    )
    assert breached.classification == "IMPLAUSIBLE_DIVERGENCE_REVIEW_REQUIRED"
    assert tuple(item.session_date for item in breached.flagged_sessions) == (axis[1],)
    assert breached.attribution == "UNATTRIBUTED"

    with pytest.raises(ValueError, match="universe_spy_axis_misaligned"):
        evaluate_universe_spy_divergence(
            universe_returns=same,
            spy_returns={session: 0.004 for session in axis[:-1]},
            universe_source_identity="u" * 64,
            spy_source_identity="s" * 64,
        )
    with pytest.raises(ValueError, match="universe_spy_return_non_finite"):
        evaluate_universe_spy_divergence(
            universe_returns={**same, axis[0]: float("nan")},
            spy_returns=dict(same),
            universe_source_identity="u" * 64,
            spy_source_identity="s" * 64,
        )


def test_session_authority_judges_only_the_range_it_covers() -> None:
    """Session authority judges only the range it covers."""

    axis = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    authority = build_trading_session_authority(calendar_ids=("XNAS", "XNYS"), sessions=axis)
    assert authority.first_session == axis[0] and authority.last_session == axis[-1]

    clean = evaluate_price_action_integrity_sentinels(
        bars=(_bar(axis[0]), _bar(axis[1])),
        actions=(),
        provider="fixture",
        as_of=date(2026, 1, 8),
        trading_sessions=authority,
    )
    assert clean.disposition == "ANCHORED"
    assert clean.session_authority_hash == authority.authority_hash

    # Before the axis begins: the authority never spoke for this date, so it is
    # silent rather than negative.
    earlier = evaluate_price_action_integrity_sentinels(
        bars=(_bar(date(2026, 1, 2)), _bar(axis[0])),
        actions=(),
        provider="fixture",
        as_of=date(2026, 1, 8),
        trading_sessions=authority,
    )
    assert earlier.disposition == "ANCHORED"

    # Inside the covered range but absent from the axis: a real defect, on both
    # the bar and the corporate action.
    gapped = build_trading_session_authority(
        calendar_ids=("XNAS", "XNYS"), sessions=(axis[0], axis[2])
    )
    breach = evaluate_price_action_integrity_sentinels(
        bars=(_bar(axis[0]), _bar(axis[1])),
        actions=(
            CorporateActionEvent(
                "listing-aapl", "fixture", axis[1], "CASH_DIVIDEND", cash_amount=0.2
            ),
        ),
        provider="fixture",
        as_of=date(2026, 1, 8),
        trading_sessions=gapped,
    )
    assert breach.disposition == "QUARANTINE_REVIEW_REQUIRED"
    assert {item.code for item in breach.findings} == {
        "data.sentinel.bar_session_not_trading",
        "data.sentinel.action_session_not_trading",
    }


def test_resolved_session_authority_excludes_weekends_and_is_content_addressed() -> None:
    """The Host resolves one calendar, and the same range always seals the same axis."""

    from alphalattice.control.data_platform.preflight import (
        resolve_trading_session_authority,
    )

    stamp = datetime(2026, 2, 1, tzinfo=UTC)
    first = resolve_trading_session_authority(
        start=date(2026, 1, 2), end=date(2026, 1, 20), as_of_timestamp=stamp
    )
    again = resolve_trading_session_authority(
        start=date(2026, 1, 2), end=date(2026, 1, 20), as_of_timestamp=stamp
    )
    assert first.authority_hash == again.authority_hash
    assert all(session.weekday() < 5 for session in first.sessions)
    assert date(2026, 1, 3) not in first.session_set  # a Saturday
    assert first.covers(date(2026, 1, 3)) is True  # covered, and not a session

    narrower = resolve_trading_session_authority(
        start=date(2026, 1, 2), end=date(2026, 1, 16), as_of_timestamp=stamp
    )
    assert narrower.authority_hash != first.authority_hash
