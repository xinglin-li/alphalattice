"""Current numerical source proofs for durable history reuse, through public owners."""

from __future__ import annotations

import struct
from datetime import UTC, date, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import SanitizedBatch
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

DAYS = (date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5))
NOW = datetime(2026, 8, 7, 22, tzinfo=UTC)
SCOPE = ("listing-a", "listing-b")


@pytest.fixture
def market_source(tmp_path):
    manifest = UniverseManifest(
        manifest_id="market-prefix-fixture",
        profile=MarketProfile(
            market_profile_id="market-prefix-fixture",
            display_name="Market prefix fixture",
            market="US",
            currency="USD",
            calendar_id="XNYS",
            provider="fixture",
            daily_price_basis="unadjusted",
            manifest_as_of=NOW.date(),
            data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
        ),
        listings=tuple(
            ManifestListing(listing_id=listing, symbol=symbol, mic="XNYS", provider_symbol=symbol)
            for listing, symbol in zip(SCOPE, ("AAA", "BBB"), strict=True)
        ),
        revision_sha256="1" * 64,
        universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
        is_point_in_time_historical=False,
    )
    market = MarketDataRepository(tmp_path / "workspace")
    market.bootstrap(manifest)
    market.apply_validated_batch(
        manifest,
        SanitizedBatch(
            "fixture",
            tuple(
                RawDailyBar(listing, "fixture", day, 100.0, 102.0, 99.0, 101.0, 2**53 + 7)
                for listing in SCOPE
                for day in DAYS
            ),
            (
                CorporateActionEvent(
                    SCOPE[0], "fixture", DAYS[0], "CASH_DIVIDEND", cash_amount=0.5
                ),
                CorporateActionEvent(
                    SCOPE[0], "fixture", DAYS[2], "CASH_DIVIDEND", cash_amount=0.75
                ),
            ),
        ),
        ingestion_id="market-prefix-fixture",
        observed_at=NOW,
    )
    return market, manifest


def _proof(market, *, listing_ids=SCOPE, start=DAYS[0], through=DAYS[1], connection=None):
    return market.market_data_source_prefix_proof(
        listing_ids=listing_ids, start=start, through=through, _connection=connection
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "UPDATE raw_daily_bar_current SET open = open + 1 WHERE session_date = ?",
        "UPDATE raw_daily_bar_current SET volume = volume + 1 WHERE session_date = ?",
        "UPDATE corporate_action_current SET cash_amount = cash_amount + 1 "
        "WHERE effective_date = ?",
    ),
    ids=("raw-price", "integer-volume-beyond-float-precision", "active-action"),
)
def test_market_prefix_proves_actual_values_when_evidence_and_journals_are_unchanged(
    market_source, mutation
):
    market, manifest = market_source
    before = _proof(market)
    watermark = market.execution_source_watermark(manifest, through=DAYS[1])
    with market.database.read_transaction() as connection:
        evidence = connection.execute(
            "SELECT payload_hash FROM raw_daily_bar_current ORDER BY listing_id, session_date"
        ).fetchall()
        actions = connection.execute(
            "SELECT payload_hash FROM corporate_action_current ORDER BY effective_date"
        ).fetchall()
        assert _proof(market, connection=connection) == before
    with market.database.connect(read_only=False) as connection:
        connection.execute(mutation, [DAYS[0]])
    with market.database.read_transaction() as connection:
        assert (
            connection.execute(
                "SELECT payload_hash FROM raw_daily_bar_current ORDER BY listing_id, session_date"
            ).fetchall()
            == evidence
        )
        assert (
            connection.execute(
                "SELECT payload_hash FROM corporate_action_current ORDER BY effective_date"
            ).fetchall()
            == actions
        )
    assert market.execution_source_watermark(manifest, through=DAYS[1]) == watermark
    assert _proof(market) != before


def test_market_prefix_respects_dates_active_actions_and_latest_provider_even_without_actions(
    market_source,
):
    market, _ = market_source
    before = _proof(market)
    later = _proof(market, through=DAYS[2])
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            "UPDATE raw_daily_bar_current SET open = open + 1 WHERE session_date = ?", [DAYS[2]]
        )
        connection.execute(
            "UPDATE corporate_action_current SET cash_amount = 9 WHERE effective_date = ?",
            [DAYS[2]],
        )
        connection.execute(
            """INSERT INTO corporate_action_current
               SELECT listing_id, 'other', effective_date, action_kind,
                      new_shares_per_old_share, cash_amount, provisional, provenance,
                      payload_hash, status, observed_at
               FROM corporate_action_current WHERE effective_date = ?""",
            [DAYS[0]],
        )
        connection.execute(
            """INSERT INTO corporate_action_current
               SELECT listing_id, provider, ?, action_kind,
                      new_shares_per_old_share, cash_amount, provisional, provenance,
                      payload_hash, 'RETRACTED', observed_at
               FROM corporate_action_current
               WHERE effective_date = ? AND provider = 'fixture'""",
            [DAYS[1], DAYS[0]],
        )
    assert _proof(market) == before
    assert _proof(market, through=DAYS[2]) != later
    assert _proof(market, listing_ids=tuple(reversed(SCOPE))) == before
    # A latest mapping is selected even when effective_from is after this
    # request's cutoff, just as the public actions reader selects it.
    assert market.actions(SCOPE[1]) == ()
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            """INSERT INTO provider_symbol_mapping
               VALUES (?, 'other', 'BBB', ?, NULL, 'ACTIVE')""",
            [SCOPE[1], date(2026, 8, 10)],
        )
    assert market.actions(SCOPE[1]) == ()
    assert _proof(market) != before
    # An action before the raw start remains part of the admitted action set.
    bounded = _proof(market, start=DAYS[1])
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            "UPDATE corporate_action_current SET status = 'RETRACTED' WHERE effective_date = ?",
            [DAYS[0]],
        )
    assert _proof(market, start=DAYS[1]) != bounded


def test_market_prefix_retains_raw_provider_rows_numeric_bits_and_missing_keys(market_source):
    market, manifest = market_source
    before = _proof(market)
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            """INSERT INTO raw_daily_bar_current
               SELECT listing_id, 'other', session_date, open, high, low, close, volume,
                      payload_hash, observed_at
               FROM raw_daily_bar_current WHERE listing_id = ? AND session_date = ?""",
            [SCOPE[0], DAYS[0]],
        )
    assert len(market.raw_bars(SCOPE[0], start=DAYS[0], through=DAYS[0])) == 2
    assert _proof(market) != before
    # An equal-numeric +0 -> -0 UPDATE may retain +0 in DuckDB. Use the
    # public bulk writer with a fractional neighbor and a nonzero transition,
    # then establish the actual reader's bits before comparing the proof.
    positive_zero = None
    for stage, opening in (("positive", 0.0), ("transition", 1.5), ("negative", -0.0)):
        market.apply_validated_batch(
            manifest,
            SanitizedBatch(
                "fixture",
                tuple(
                    RawDailyBar(
                        SCOPE[0],
                        "fixture",
                        day,
                        opening if day == DAYS[0] else 1.125,
                        2.5,
                        0.0,
                        1.25,
                        1000,
                    )
                    for day in DAYS
                ),
                (),
            ),
            ingestion_id=f"market-prefix-zero-{stage}",
            observed_at=NOW,
        )
        actual = next(
            bar.open
            for bar in market.raw_bars(SCOPE[0], start=DAYS[0], through=DAYS[0])
            if bar.provider == "fixture"
        )
        assert struct.pack("!d", actual) == struct.pack("!d", opening)
        if stage == "positive":
            positive_zero = _proof(market)
    assert _proof(market) != positive_zero
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            "UPDATE raw_daily_bar_current SET open = 'nan'::DOUBLE WHERE session_date = ?",
            [DAYS[0]],
        )
    nan_rows = _proof(market)
    with market.database.connect(read_only=False) as connection:
        connection.execute("DELETE FROM raw_daily_bar_current WHERE session_date = ?", [DAYS[0]])
    assert _proof(market) != nan_rows


def test_market_prefix_refuses_invalid_scopes_and_missing_provider(market_source):
    market, _ = market_source
    for scope in ((), (SCOPE[0], SCOPE[0])):
        with pytest.raises(ValueError, match=r"market_data_ops\.source_prefix_request_invalid"):
            _proof(market, listing_ids=scope)
    with pytest.raises(ValueError, match=r"market_data_ops\.source_prefix_request_invalid"):
        _proof(market, start=DAYS[2])
    with market.database.connect(read_only=False) as connection:
        connection.execute("DELETE FROM provider_symbol_mapping WHERE listing_id = ?", [SCOPE[1]])
    with pytest.raises(ValueError, match="listing has no provider mapping"):
        _proof(market)


def test_listing_set_reads_answer_each_listings_own_reads(market_source):
    """Set-scoped listing reads match each listing's individual read."""
    market, _ = market_source
    with market.database.connect(read_only=False) as connection:
        connection.execute(
            """INSERT INTO corporate_action_current
               SELECT listing_id, 'other', effective_date, action_kind,
                      new_shares_per_old_share, cash_amount, provisional, provenance,
                      payload_hash, status, observed_at
               FROM corporate_action_current WHERE effective_date = ?""",
            [DAYS[0]],
        )
        connection.execute(
            """INSERT INTO provider_symbol_mapping
               VALUES (?, 'other', 'BBB', ?, NULL, 'ACTIVE')""",
            [SCOPE[1], date(2026, 8, 10)],
        )
    for start, through in ((None, None), (DAYS[1], None), (DAYS[0], DAYS[1]), (DAYS[2], DAYS[1])):
        assert market.raw_bars_by_listing(SCOPE, start=start, through=through) == {
            listing: market.raw_bars(listing, start=start, through=through) for listing in SCOPE
        }
    assert market.raw_bars_by_listing(("missing",)) == {"missing": ()}
    assert market.actions_by_listing(tuple(reversed(SCOPE))) == {
        listing: market.actions(listing) for listing in SCOPE
    }
    with market.database.connect(read_only=False) as connection:
        connection.execute("DELETE FROM provider_symbol_mapping WHERE listing_id = ?", [SCOPE[1]])
    with pytest.raises(ValueError, match="listing has no provider mapping"):
        market.actions(SCOPE[1])
    with pytest.raises(ValueError, match="listing has no provider mapping"):
        market.actions_by_listing(SCOPE)


@settings(max_examples=300, deadline=None)
@given(st.data())
def test_a_rolling_refresh_compares_returns_from_the_session_before_it(data):
    """A rolling refresh compares returns from the session before it."""
    calendar = [date(2026, 1, 1) + timedelta(days=i) for i in range(data.draw(st.integers(2, 30)))]
    closes = st.floats(1.0, 500.0, allow_nan=False)
    kept = data.draw(st.lists(st.booleans(), min_size=len(calendar), max_size=len(calendar)))
    prior = {day: data.draw(closes) for day, keep in zip(calendar, kept, strict=True) if keep}
    first = data.draw(st.integers(0, len(calendar) - 1))
    observed = {
        day: prior[day] if day in prior and data.draw(st.booleans()) else data.draw(closes)
        for day in calendar[first:]
        if day in prior or data.draw(st.booleans())
    }
    for extra in range(data.draw(st.integers(0, 3))):
        observed[calendar[-1] + timedelta(days=extra + 1)] = data.draw(closes)
    if not prior or not observed:
        return
    current = {**prior, **observed}
    changes = MarketDataRepository._provider_adjusted_return_changes
    assert changes(prior, current, refreshed_from=min(observed)) == changes(prior, current)
