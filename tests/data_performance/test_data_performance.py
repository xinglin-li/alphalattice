"""Offline equivalence and boundary tests for the performance closure."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.producers.base_materializer import (
    BaseFeatureMaterializer,
)
from alphalattice.foundation.feature_engine.storage.contracts import FeatureMaterializationWrite
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import (
    ProviderFetchError,
    YFinanceMarketDataProvider,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    CorruptedPayload,
    SanitizedBatch,
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository, _bar_hash

NOW = datetime(2026, 8, 3, 22, tzinfo=UTC)


def _manifest() -> UniverseManifest:
    profile = MarketProfile(
        "performance-fixture",
        "Performance fixture",
        "US",
        "USD",
        "XNYS",
        "yfinance",
        "unadjusted",
        date(2026, 8, 3),
        "CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    return UniverseManifest(
        "performance-fixture",
        profile,
        (ManifestListing("listing-aapl", "AAPL", "XNAS", "AAPL"),),
        "a" * 64,
    )


def _rows(sessions: tuple[date, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "session_date": session.isoformat(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000 + index,
            "split_ratio": 2.0 if index == 1_000 else 0.0,
            "cash_dividend": 0.25 if index == 2_000 else 0.0,
            "capital_gain": 0.0,
        }
        for index, session in enumerate(sessions)
    )


def _legacy_bar_loop(bars) -> float:
    connection = duckdb.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE raw_daily_bar_current (
            listing_id VARCHAR, provider VARCHAR, session_date DATE,
            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT,
            payload_hash VARCHAR, observed_at TIMESTAMP,
            PRIMARY KEY (listing_id, provider, session_date)
        );
        CREATE TABLE data_quality (
            listing_id VARCHAR PRIMARY KEY, latest_session DATE,
            quality_state VARCHAR, summary VARCHAR, updated_at TIMESTAMP
        );
        """
    )
    started = time.perf_counter()
    connection.execute("BEGIN TRANSACTION")
    for bar in bars:
        payload_hash = _bar_hash(bar)
        current = connection.execute(
            """SELECT payload_hash FROM raw_daily_bar_current
            WHERE listing_id = ? AND provider = ? AND session_date = ?""",
            [bar.listing_id, bar.provider, bar.session_date],
        ).fetchone()
        if current is None:
            connection.execute(
                "INSERT INTO raw_daily_bar_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    bar.listing_id,
                    bar.provider,
                    bar.session_date,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    payload_hash,
                    NOW.replace(tzinfo=None),
                ],
            )
        connection.execute(
            """
            INSERT INTO data_quality VALUES (?, ?, 'READY', 'validated raw payload', ?)
            ON CONFLICT (listing_id) DO UPDATE SET
                latest_session = greatest(data_quality.latest_session, excluded.latest_session),
                updated_at = excluded.updated_at
            """,
            [bar.listing_id, bar.session_date, NOW.replace(tzinfo=None)],
        )
    connection.execute("COMMIT")
    elapsed = time.perf_counter() - started
    connection.close()
    return elapsed


def test_set_based_raw_persistence_is_equivalent_idempotent_and_faster(tmp_path) -> None:
    sessions = tuple(pd.bdate_range("2016-01-04", periods=2_520).date)
    manifest = _manifest()
    market_data = MarketDataRepository(tmp_path / "workspace")
    market_data.bootstrap(manifest)
    payload = {"AAPL": tuple(reversed(_rows(sessions)))}
    batch = sanitize_payload(manifest, "yfinance", payload, ("AAPL",))

    started = time.perf_counter()
    first = market_data.apply_validated_batch(
        manifest, batch, ingestion_id="initial", observed_at=NOW
    )
    set_seconds = time.perf_counter() - started
    legacy_seconds = _legacy_bar_loop(batch.bars)

    assert first == {
        "inserted": 2_520,
        "corrected": 0,
        "action_inserted": 2,
        "action_corrected": 0,
    }
    assert legacy_seconds / max(set_seconds, 1e-9) >= 20
    assert set_seconds * 500 < 120
    assert market_data.revision_count("listing-aapl") == 0
    assert market_data.action_revision_count("listing-aapl") == 0

    replay = market_data.apply_validated_batch(
        manifest, batch, ingestion_id="initial", observed_at=NOW
    )
    assert replay == {
        "inserted": 0,
        "corrected": 0,
        "action_inserted": 0,
        "action_corrected": 0,
    }

    corrected_bars = tuple(
        replace(bar, close=100.5) if bar.session_date == sessions[500] else bar
        for bar in batch.bars
    )
    corrected_actions = tuple(
        replace(action, cash_amount=0.5) if action.action_kind == "CASH_DIVIDEND" else action
        for action in batch.actions
    )
    corrected = SanitizedBatch("yfinance", corrected_bars, corrected_actions)
    outcome = market_data.apply_validated_batch(
        manifest, corrected, ingestion_id="correction", observed_at=NOW
    )
    assert outcome == {
        "inserted": 0,
        "corrected": 1,
        "action_inserted": 0,
        "action_corrected": 1,
    }
    assert market_data.revision_count("listing-aapl") == 1
    assert market_data.action_revision_count("listing-aapl") == 1
    assert market_data.apply_validated_batch(
        manifest, corrected, ingestion_id="correction", observed_at=NOW
    ) == {
        "inserted": 0,
        "corrected": 0,
        "action_inserted": 0,
        "action_corrected": 0,
    }
    connection = duckdb.connect(str(market_data.path), read_only=True)
    try:
        assert (
            connection.execute("SELECT count(*) FROM raw_daily_bar_current").fetchone()[0] == 2_520
        )
        assert connection.execute("SELECT latest_session FROM data_quality").fetchone()[0] == max(
            sessions
        )
        assert connection.execute("SELECT count(*) FROM provider_attempt").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT count(*) FROM corporate_action_current WHERE status = 'ACTIVE'"
            ).fetchone()[0]
            == 2
        )
    finally:
        connection.close()


def _canonical_digest(value: object) -> str:
    """The canonical JSON digest a raw bar's payload hash is defined as."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_bar_payload_hash_is_the_canonical_json_digest() -> None:
    fields = ("listing_id", "provider", "session_date", "open", "high", "low", "close", "volume")
    bar = RawDailyBar(
        "listing-aapl", "yfinance", date(2026, 1, 2), 100.0, 101.25, 99.5, 100.75, 1_000
    )
    shapes = (
        # Written directly: plain text (unescaped under ensure_ascii=False), a date,
        # finite floats at their repr's edges, ints.
        bar,
        replace(bar, open=-0.0, high=5e-324, low=1e-7, close=0.1 + 0.2, volume=0),
        replace(bar, high=1.7976931348623157e308, close=1e16, volume=1_000.0),
        replace(bar, listing_id="listing-é-東京\u2028\x7f", volume=-1),
        # Left to the encoder: non-finite and numpy numbers, a bool or null volume,
        # escaped text, a datetime session.
        replace(bar, close=float("nan")),
        replace(bar, high=float("inf"), low=float("-inf")),
        replace(bar, close=np.float64(100.75), volume=np.int64(7)),
        replace(bar, volume=True),
        replace(bar, volume=None),
        replace(bar, listing_id='listing-"aapl"'),
        replace(bar, provider="y\\finance"),
        replace(bar, provider="y\nfinance"),
        replace(bar, session_date=datetime(2026, 1, 2, 21, 30)),
    )
    for shape in shapes:
        expected = _canonical_digest({name: getattr(shape, name) for name in fields})
        assert _bar_hash(shape) == expected, shape


class _Cache:
    def set_cache_location(self, _location: str) -> None:
        return None


class _Ticker:
    def __init__(self, frame: pd.DataFrame, calls: list[dict[str, object]]) -> None:
        self.frame = frame
        self.calls = calls

    def history(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame

    @property
    def actions(self) -> pd.DataFrame:
        return self.frame


class _YFinance:
    cache = _Cache()

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict[str, object]] = []

    def Ticker(self, _symbol: str) -> _Ticker:
        return _Ticker(self.frame, self.calls)


def _history_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"])
    return pd.DataFrame(
        {
            "Open": [49.5, 50.0, 50.0, 51.0],
            "High": [50.5, 50.5, 51.0, 52.0],
            "Low": [49.0, 49.5, 49.0, 50.0],
            "Close": [50.0, 50.0, 50.0, 51.0],
            "Adj Close": [49.5, 49.75, 50.0, 51.0],
            "Volume": [1_000, 1_100, 2_000, 2_100],
            "Dividends": [0.0, 0.5, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 2.0, 0.0],
            "Capital Gains": [0.0, 0.0, 0.0, 0.25],
            "Repaired?": [False, False, True, False],
        },
        index=index,
    )


def test_one_response_hydration_splits_evidence_and_filters_window(tmp_path, monkeypatch) -> None:
    fake = _YFinance(_history_frame())
    monkeypatch.setattr(YFinanceMarketDataProvider, "_module", lambda _self: fake)
    provider = YFinanceMarketDataProvider(tmp_path / "cache")
    evidence = provider.fetch_hydration(
        listing_id="listing-aapl",
        provider_symbol="AAPL",
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )

    assert len(fake.calls) == 1
    assert fake.calls[0] == {
        "start": "2026-01-02",
        "end": "2026-01-07",
        "auto_adjust": False,
        "actions": True,
        "repair": True,
        "raise_errors": True,
    }
    assert tuple(row["session_date"] for row in evidence.daily_rows) == (
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
    )
    assert [(item.action_kind, item.effective_date) for item in evidence.actions] == [
        ("CASH_DIVIDEND", date(2026, 1, 2)),
        ("SPLIT", date(2026, 1, 5)),
        ("CAPITAL_GAIN", date(2026, 1, 6)),
    ]
    assert all(item.listing_id == "listing-aapl" for item in evidence.actions)
    assert len(evidence.adjusted_closes) == 3
    assert evidence.repaired_sessions == (date(2026, 1, 5),)
    assert evidence.provider_policy_hash is not None
    legacy_daily = provider.fetch_daily(("AAPL",), start=date(2026, 1, 2), end=date(2026, 1, 6))[
        "AAPL"
    ]
    legacy_actions = provider.fetch_action_history(
        listing_id="listing-aapl",
        provider_symbol="AAPL",
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    legacy_adjusted = provider.fetch_adjusted_close_history(
        listing_id="listing-aapl",
        provider_symbol="AAPL",
        start=date(2026, 1, 2),
        end=date(2026, 1, 6),
    )
    assert evidence.daily_rows == legacy_daily
    assert evidence.actions == legacy_actions
    assert evidence.adjusted_closes == legacy_adjusted


def test_provider_rejects_unrepaired_split_discontinuity(tmp_path, monkeypatch) -> None:
    frame = _history_frame().copy()
    frame.loc[frame.index[:2], ["Open", "High", "Low", "Close"]] *= 2.0
    fake = _YFinance(frame)
    monkeypatch.setattr(YFinanceMarketDataProvider, "_module", lambda _self: fake)

    with pytest.raises(ProviderFetchError) as captured:
        YFinanceMarketDataProvider(tmp_path / "cache").fetch_hydration(
            listing_id="listing-aapl",
            provider_symbol="AAPL",
            start=date(2026, 1, 1),
            end=date(2026, 1, 6),
        )

    assert captured.value.code == "data.unrepaired_split_adjustment"
    assert captured.value.retryable is False


def test_provider_rejects_other_split_day_price_discontinuity(tmp_path, monkeypatch) -> None:
    frame = _history_frame().copy()
    frame.loc[frame.index[2], ["Open", "High", "Low", "Close"]] *= 6.0
    fake = _YFinance(frame)
    monkeypatch.setattr(YFinanceMarketDataProvider, "_module", lambda _self: fake)

    with pytest.raises(ProviderFetchError) as captured:
        YFinanceMarketDataProvider(tmp_path / "cache").fetch_hydration(
            listing_id="listing-aapl",
            provider_symbol="AAPL",
            start=date(2026, 1, 1),
            end=date(2026, 1, 6),
        )

    assert captured.value.code == "data.unrepaired_split_adjustment"
    assert captured.value.retryable is False


def test_provider_allows_cash_event_explained_price_discontinuity(tmp_path, monkeypatch) -> None:
    frame = _history_frame().iloc[:2].copy()
    frame.loc[frame.index[0], ["Open", "High", "Low", "Close"]] = [
        123.0,
        124.0,
        122.0,
        123.5,
    ]
    frame.loc[frame.index[1], ["Open", "High", "Low", "Close"]] = [
        20.5,
        22.0,
        20.0,
        21.5,
    ]
    frame.loc[frame.index[1], "Dividends"] = 103.75
    frame.loc[:, "Stock Splits"] = 0.0
    fake = _YFinance(frame)
    monkeypatch.setattr(YFinanceMarketDataProvider, "_module", lambda _self: fake)

    evidence = YFinanceMarketDataProvider(tmp_path / "cache").fetch_hydration(
        listing_id="listing-kdp",
        provider_symbol="KDP",
        start=date(2026, 1, 1),
        end=date(2026, 1, 2),
    )

    assert len(evidence.daily_rows) == 2


@pytest.mark.parametrize("value", [float("nan"), 0.0, -1.0])
def test_one_response_hydration_rejects_invalid_adjusted_close(
    tmp_path, monkeypatch, value
) -> None:
    frame = _history_frame().iloc[1:].copy()
    frame.loc[frame.index[0], "Adj Close"] = value
    fake = _YFinance(frame)
    monkeypatch.setattr(YFinanceMarketDataProvider, "_module", lambda _self: fake)
    provider = YFinanceMarketDataProvider(tmp_path / "cache")
    with pytest.raises(ProviderFetchError) as captured:
        provider.fetch_hydration(
            listing_id="listing-aapl",
            provider_symbol="AAPL",
            start=date(2026, 1, 2),
            end=date(2026, 1, 6),
        )
    assert captured.value.code == "data.invalid_adjusted_close_payload"
    assert "2026-01-02" in str(captured.value)


def _seasonality_reference(
    sessions: pd.DatetimeIndex,
    closes: np.ndarray,
    *,
    years: int,
    minimum_observations: int,
) -> np.ndarray:
    output = np.full(len(sessions), np.nan)
    visible_month_closes: dict[tuple[int, int], float] = {}
    for position, evaluation in enumerate(sessions):
        if position:
            visible = sessions[position - 1]
            visible_month_closes[(visible.year, visible.month)] = float(closes[position - 1])
        if position + 1 < minimum_observations:
            continue
        ordered = tuple(sorted(visible_month_closes.items()))
        monthly_returns = {
            key: float(np.log(close / prior_close))
            for (key, close), (_prior_key, prior_close) in zip(
                ordered[1:], ordered[:-1], strict=True
            )
            if prior_close > 0.0
        }
        if years == 1:
            selected = ((evaluation.year - 1, evaluation.month),)
        else:
            selected = tuple(
                key
                for key in monthly_returns
                if key[1] == evaluation.month and key[0] < evaluation.year
            )[-years:]
        values = [monthly_returns[key] for key in selected if key in monthly_returns]
        if len(values) == years:
            output[position] = float(np.mean(values))
    return output


def _empty_feature_cutoff(catalog: FeatureCatalog) -> str:
    return json.dumps(
        {factor_id: None for factor_id in catalog.factor_ids},
        sort_keys=True,
        separators=(",", ":"),
    )


def _rle_workspace(
    tmp_path: Path,
    *,
    root_name: str,
    periods: int,
) -> tuple[
    tuple[date, ...],
    FeatureCatalog,
    MarketDataRepository,
    FeatureStateRepository,
    str,
]:
    sessions = tuple(pd.bdate_range("2026-01-02", periods=periods).date)
    catalog = FeatureCatalog.load()
    market_data = MarketDataRepository(tmp_path / root_name)
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(_manifest())
    feature_state.ensure_current_storage()
    return sessions, catalog, market_data, feature_state, _empty_feature_cutoff(catalog)


def _feature_rows(sessions: tuple[date, ...], cutoff: str) -> list[dict[str, object]]:
    return [
        {
            "listing_id": "listing-aapl",
            "session_date": session,
            "input_cutoffs_json": cutoff,
        }
        for session in sessions
    ]


def test_vectorized_seasonality_matches_full_history_reference_for_20_listings() -> None:
    sessions = pd.bdate_range("2016-01-04", periods=2_520)
    catalog = FeatureCatalog.load()
    specs = {item.factor_id: item for item in catalog.factors}
    materializer = BaseFeatureMaterializer(catalog)
    market_rng = np.random.default_rng(10_000)
    market_close = 100.0 * np.cumprod(1.0 + market_rng.normal(0.0002, 0.01, len(sessions)))
    market_frame = pd.DataFrame(
        {
            "session_date": sessions,
            "open_raw": market_close * 0.997,
            "high_raw": market_close * 1.01,
            "low_raw": market_close * 0.99,
            "close_raw": market_close,
            "volume_raw": market_rng.integers(1_000_000, 5_000_000, len(sessions)),
            "open_split_adjusted": market_close * 0.997,
            "high_split_adjusted": market_close * 1.01,
            "low_split_adjusted": market_close * 0.99,
            "close_split_adjusted": market_close,
            "provider_adjusted_close": market_close,
            "close_total_return_adjusted": market_close,
        }
    )
    for seed in range(20):
        rng = np.random.default_rng(seed)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(sessions)))
        frame = pd.DataFrame(
            {
                "session_date": sessions,
                "open_raw": close * 0.997,
                "high_raw": close * 1.01,
                "low_raw": close * 0.99,
                "close_raw": close,
                "volume_raw": rng.integers(1_000_000, 5_000_000, len(sessions)),
                "open_split_adjusted": close * 0.997,
                "high_split_adjusted": close * 1.01,
                "low_split_adjusted": close * 0.99,
                "close_split_adjusted": close,
                "provider_adjusted_close": close,
                "close_total_return_adjusted": close,
            }
        )
        block = materializer.materialize_listing(
            listing_id=f"seasonality-{seed}", projected_bars=frame, market_bars=market_frame
        )
        for factor_id, years in (("seasonality_12m", 1),):
            expected = _seasonality_reference(
                sessions,
                close,
                years=years,
                minimum_observations=specs[factor_id].minimum_observations,
            )
            actual = block.values[factor_id].to_numpy(dtype=float)
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_serialized_cutoff_optimization_preserves_every_feature_row_hash(tmp_path) -> None:
    sessions = pd.bdate_range("2020-01-02", periods=300)
    rng = np.random.default_rng(9)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(sessions)))
    frame = pd.DataFrame(
        {
            "session_date": sessions,
            "open_raw": close * 0.997,
            "high_raw": close * 1.01,
            "low_raw": close * 0.99,
            "close_raw": close,
            "volume_raw": rng.integers(1_000_000, 5_000_000, len(sessions)),
            "open_split_adjusted": close * 0.997,
            "high_split_adjusted": close * 1.01,
            "low_split_adjusted": close * 0.99,
            "close_split_adjusted": close,
            "provider_adjusted_close": close,
            "close_total_return_adjusted": close,
        }
    )
    catalog = FeatureCatalog.load()
    market_rng = np.random.default_rng(10)
    market_close = 100.0 * np.cumprod(1.0 + market_rng.normal(0.0002, 0.01, len(sessions)))
    market_frame = frame.copy()
    for field, multiplier in (
        ("open_raw", 0.997),
        ("high_raw", 1.01),
        ("low_raw", 0.99),
        ("close_raw", 1.0),
        ("open_split_adjusted", 0.997),
        ("high_split_adjusted", 1.01),
        ("low_split_adjusted", 0.99),
        ("close_split_adjusted", 1.0),
        ("provider_adjusted_close", 1.0),
        ("close_total_return_adjusted", 1.0),
    ):
        market_frame[field] = market_close * multiplier
    block = BaseFeatureMaterializer(catalog).materialize_listing(
        listing_id="listing-aapl", projected_bars=frame, market_bars=market_frame
    )
    string_rows = block.values.to_dict("records")
    mapping_rows = [
        {**row, "input_cutoffs_json": json.loads(str(row["input_cutoffs_json"]))}
        for row in string_rows
    ]
    hashes = []
    for name, rows, rows_are_canonical in (
        ("serialized", string_rows, False),
        ("mapping", mapping_rows, False),
        ("columnar", block.values, True),
    ):
        market_data = MarketDataRepository(tmp_path / name)
        feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
        market_data.bootstrap(_manifest())
        feature_state.ensure_current_storage()
        feature_state.upsert_feature_materialization(
            listing_id="listing-aapl",
            catalog_hash=catalog.binding.catalog_hash,
            rows=rows,
            ineligibility=block.ineligibility.to_dict("records"),
            raw_input_hash="raw",
            action_set_hash_value="actions",
            market_reference_revision="spy",
            idempotency_key=name,
            revision_reason="initial",
            observed_at=NOW,
            rows_are_canonical=rows_are_canonical,
        )
        connection = duckdb.connect(str(market_data.path), read_only=True)
        try:
            hashes.append(
                connection.execute(
                    "SELECT row_hash FROM feature_daily_current ORDER BY session_date"
                )
                .fetchnumpy()["row_hash"]
                .tolist()
            )
        finally:
            connection.close()
    assert hashes[0] == hashes[1]


def test_rle_preserves_prefix_interior_constant_and_single_day_runs(tmp_path) -> None:
    sessions, catalog, market_data, feature_state, cutoff = _rle_workspace(
        tmp_path, root_name="rle", periods=6
    )
    rows = _feature_rows(sessions, cutoff)
    ineligibility = [
        {
            "session_date": sessions[0],
            "factor_id": "rev_21",
            "reason": "insufficient_history",
            "observation_count": 1,
        },
        {
            "session_date": sessions[1],
            "factor_id": "rev_21",
            "reason": "insufficient_history",
            "observation_count": 2,
        },
        {
            "session_date": sessions[3],
            "factor_id": "rev_21",
            "reason": "zero_denominator",
            "observation_count": 5,
        },
        {
            "session_date": sessions[4],
            "factor_id": "rev_21",
            "reason": "zero_denominator",
            "observation_count": 5,
        },
        {
            "session_date": sessions[5],
            "factor_id": "beta_63",
            "reason": "market_alignment",
            "observation_count": 6,
        },
    ]
    feature_state.upsert_feature_materialization(
        listing_id="listing-aapl",
        catalog_hash=catalog.binding.catalog_hash,
        rows=rows,
        ineligibility=ineligibility,
        raw_input_hash="raw",
        action_set_hash_value="actions",
        market_reference_revision="spy",
        idempotency_key="rle-shapes",
        revision_reason="initial",
        observed_at=NOW,
    )
    connection = duckdb.connect(str(market_data.path), read_only=True)
    try:
        expanded = connection.execute(
            """
            SELECT session_date, factor_id, reason, observation_count
            FROM feature_ineligibility ORDER BY factor_id, session_date
            """
        ).fetchall()
        run_count = connection.execute("SELECT count(*) FROM feature_ineligibility_run").fetchone()[
            0
        ]
    finally:
        connection.close()
    expected = sorted(
        [
            (
                item["session_date"],
                item["factor_id"],
                item["reason"],
                item["observation_count"],
            )
            for item in ineligibility
        ],
        key=lambda item: (item[1], item[0]),
    )
    assert expanded == expected
    assert run_count == 3
    bounded = feature_state.find_feature_ineligibility_runs(
        listing_ids=("listing-aapl",),
        factor_ids=("rev_21",),
        start=sessions[3],
        end=sessions[4],
    )
    assert len(bounded) == 1
    assert bounded[0].reason == "zero_denominator"


def test_partial_rle_rebuild_clips_physical_runs_without_view_input(tmp_path) -> None:
    sessions, catalog, market_data, feature_state, cutoff = _rle_workspace(
        tmp_path, root_name="rle-overlap", periods=8
    )

    def feature_rows(selected: tuple[date, ...]) -> list[dict[str, object]]:
        return _feature_rows(selected, cutoff)

    initial_ineligibility = [
        {
            "session_date": session,
            "factor_id": "rev_21",
            "reason": "insufficient_history",
            "observation_count": index + 1,
        }
        for index, session in enumerate(sessions)
    ] + [
        {
            "session_date": session,
            "factor_id": "beta_63",
            "reason": "zero_denominator",
            "observation_count": 5,
        }
        for session in sessions
    ]
    feature_state.upsert_feature_materialization(
        listing_id="listing-aapl",
        catalog_hash=catalog.binding.catalog_hash,
        rows=feature_rows(sessions),
        ineligibility=initial_ineligibility,
        raw_input_hash="raw-initial",
        action_set_hash_value="actions",
        market_reference_revision="spy",
        idempotency_key="rle-initial",
        revision_reason="initial",
        observed_at=NOW,
    )
    changed_range = sessions[2:6]
    changed_ineligibility = [
        {
            "session_date": session,
            "factor_id": "rev_21",
            "reason": "zero_denominator",
            "observation_count": 3,
        }
        for session in changed_range[:2]
    ]
    feature_state.upsert_feature_materialization(
        listing_id="listing-aapl",
        catalog_hash=catalog.binding.catalog_hash,
        rows=feature_rows(changed_range),
        ineligibility=changed_ineligibility,
        raw_input_hash="raw-correction",
        action_set_hash_value="actions",
        market_reference_revision="spy",
        idempotency_key="rle-partial",
        revision_reason="raw_correction",
        observed_at=NOW,
    )
    connection = duckdb.connect(str(market_data.path), read_only=True)
    try:
        expanded = connection.execute(
            """
            SELECT session_date, factor_id, reason, observation_count
            FROM feature_ineligibility ORDER BY factor_id, session_date
            """
        ).fetchall()
        run_ids_before_replay = connection.execute(
            "SELECT run_id FROM feature_ineligibility_run ORDER BY run_id"
        ).fetchall()
    finally:
        connection.close()
    expected = sorted(
        [
            (sessions[0], "rev_21", "insufficient_history", 1),
            (sessions[1], "rev_21", "insufficient_history", 2),
            (sessions[2], "rev_21", "zero_denominator", 3),
            (sessions[3], "rev_21", "zero_denominator", 3),
            (sessions[6], "rev_21", "insufficient_history", 7),
            (sessions[7], "rev_21", "insufficient_history", 8),
            (sessions[0], "beta_63", "zero_denominator", 5),
            (sessions[1], "beta_63", "zero_denominator", 5),
            (sessions[6], "beta_63", "zero_denominator", 5),
            (sessions[7], "beta_63", "zero_denominator", 5),
        ],
        key=lambda item: (item[1], item[0]),
    )
    assert expanded == expected

    feature_state.upsert_feature_materialization(
        listing_id="listing-aapl",
        catalog_hash=catalog.binding.catalog_hash,
        rows=feature_rows(changed_range),
        ineligibility=changed_ineligibility,
        raw_input_hash="raw-correction",
        action_set_hash_value="actions",
        market_reference_revision="spy",
        idempotency_key="rle-partial",
        revision_reason="raw_correction",
        observed_at=NOW,
    )
    connection = duckdb.connect(str(market_data.path), read_only=True)
    try:
        assert (
            connection.execute(
                "SELECT run_id FROM feature_ineligibility_run ORDER BY run_id"
            ).fetchall()
            == run_ids_before_replay
        )
    finally:
        connection.close()


def test_feature_persistence_rolls_back_after_staged_rle_write(tmp_path) -> None:
    sessions, catalog, market_data, feature_state, cutoff = _rle_workspace(
        tmp_path, root_name="rle-rollback", periods=3
    )
    rows = _feature_rows(sessions, cutoff)
    ineligibility = [
        {
            "session_date": session,
            "factor_id": "rev_21",
            "reason": "insufficient_history",
            "observation_count": index + 1,
        }
        for index, session in enumerate(sessions)
    ]

    def fail_after_rle_write(stage: str, _elapsed: float) -> None:
        if stage == "rle_write":
            raise RuntimeError("fixture crash after staged RLE write")

    with pytest.raises(RuntimeError, match="fixture crash"):
        feature_state.upsert_feature_materialization(
            listing_id="listing-aapl",
            catalog_hash=catalog.binding.catalog_hash,
            rows=rows,
            ineligibility=ineligibility,
            raw_input_hash="raw",
            action_set_hash_value="actions",
            market_reference_revision="spy",
            idempotency_key="rle-crash",
            revision_reason="initial",
            observed_at=NOW,
            timing_sink=fail_after_rle_write,
        )
    connection = duckdb.connect(str(market_data.path), read_only=True)
    try:
        assert connection.execute("SELECT count(*) FROM feature_daily_current").fetchone()[0] == 0
        assert (
            connection.execute("SELECT count(*) FROM feature_ineligibility_run").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT count(*) FROM feature_materialization_receipt").fetchone()[0]
            == 0
        )
    finally:
        connection.close()


def test_feature_current_insert_skips_conflict_probe_until_a_revision_exists(tmp_path) -> None:
    session = date(2026, 1, 2)
    catalog = FeatureCatalog.load()
    market_data = MarketDataRepository(tmp_path / "insert-mode")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(_manifest())
    feature_state.ensure_current_storage()
    cutoff = json.dumps(
        {factor_id: None for factor_id in catalog.factor_ids},
        sort_keys=True,
        separators=(",", ":"),
    )

    class RecordingConnection:
        def __init__(self, inner) -> None:
            self.inner = inner
            self.statements: list[str] = []

        def execute(self, query, parameters=None):
            self.statements.append(" ".join(str(query).split()))
            if parameters is None:
                return self.inner.execute(query)
            return self.inner.execute(query, parameters)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    def write(*, value: float | None, identity: str) -> FeatureMaterializationWrite:
        row: dict[str, object] = {
            "listing_id": "listing-aapl",
            "session_date": session,
            "input_cutoffs_json": cutoff,
        }
        if value is not None:
            row["rev_21"] = value
        return FeatureMaterializationWrite(
            listing_id="listing-aapl",
            catalog_hash=catalog.binding.catalog_hash,
            rows=(row,),
            ineligibility=(),
            raw_input_hash="raw",
            action_set_hash_value="actions",
            market_reference_revision="spy",
            idempotency_key=identity,
            revision_reason="fixture",
            observed_at=NOW,
        )

    with feature_state.feature_build_connection() as inner:
        connection = RecordingConnection(inner)
        feature_state.upsert_feature_materialization_batch(
            (write(value=None, identity="initial"),),
            _connection=connection,
        )
        current_writes = [
            statement
            for statement in connection.statements
            if "INTO feature_daily_current" in statement
        ]
        assert len(current_writes) == 1
        assert current_writes[0].startswith("INSERT INTO feature_daily_current")

        connection.statements.clear()
        feature_state.upsert_feature_materialization_batch(
            (write(value=1.0, identity="revision"),),
            _connection=connection,
        )
        current_writes = [
            statement
            for statement in connection.statements
            if "INTO feature_daily_current" in statement
        ]
        assert len(current_writes) == 1
        assert current_writes[0].startswith("INSERT OR REPLACE INTO feature_daily_current")

    assert feature_state.feature_revision_count("listing-aapl") == 1


def test_shared_session_cutoff_sets_stay_one_relation_per_batch(tmp_path) -> None:
    """Shared session cutoff sets stay one relation per batch."""

    sessions = tuple(pd.bdate_range("2026-01-02", periods=40).date)
    catalog = FeatureCatalog.load()
    writes = tuple(
        FeatureMaterializationWrite(
            listing_id=f"listing-{index}",
            catalog_hash=catalog.binding.catalog_hash,
            rows=tuple(
                {
                    "listing_id": f"listing-{index}",
                    "session_date": session,
                    "input_cutoffs_json": json.dumps(
                        {factor_id: session.isoformat() for factor_id in catalog.factor_ids},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                for session in sessions
            ),
            ineligibility=(),
            raw_input_hash=f"raw-{index}",
            action_set_hash_value="actions",
            market_reference_revision="spy",
            idempotency_key=f"write-{index}",
            revision_reason="initial",
            observed_at=NOW,
        )
        for index in range(4)
    )

    class CountingConnection:
        def __init__(self, inner) -> None:
            self.inner = inner
            self.statements: list[str] = []

        def execute(self, query, parameters=None):
            self.statements.append(" ".join(str(query).split()))
            if parameters is None:
                return self.inner.execute(query)
            return self.inner.execute(query, parameters)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    market_data = MarketDataRepository(tmp_path / "cutoff-relation")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(_manifest())
    feature_state.ensure_current_storage()
    with feature_state.feature_build_connection() as inner:
        connection = CountingConnection(inner)
        feature_state.upsert_feature_materialization_batch(writes, _connection=connection)
        cutoff_writes = [
            statement
            for statement in connection.statements
            if "INTO feature_input_cutoff_set" in statement
        ]
    # One relation for the batch, never one statement per cutoff row: the per-row
    # shape would have issued len(writes) * len(sessions) = 160 probed inserts here,
    # and 1.17M over a full universe rebuild. The writes share their sessions, so
    # the first offers every set and the others none.
    assert len(cutoff_writes) == 1
    assert all("SELECT cutoff_set_hash" in statement for statement in cutoff_writes)
    assert all("VALUES" not in statement for statement in cutoff_writes)
    stored = duckdb.connect(str(feature_state.path), read_only=True)
    try:
        count, factor_count = stored.execute(
            "SELECT count(*), max(factor_count) FROM feature_input_cutoff_set"
        ).fetchone()
    finally:
        stored.close()
    assert count == len(sessions)
    assert factor_count == len(catalog.factor_ids)


@pytest.mark.parametrize("batch_size", (1, 4, 8))
def test_batched_feature_persistence_matches_single_listing_reference(
    tmp_path, batch_size: int
) -> None:
    sessions = tuple(pd.bdate_range("2026-01-02", periods=6).date)
    catalog = FeatureCatalog.load()
    cutoff = _empty_feature_cutoff(catalog)
    writes = tuple(
        FeatureMaterializationWrite(
            listing_id=f"listing-{index}",
            catalog_hash=catalog.binding.catalog_hash,
            rows=tuple(
                {
                    "listing_id": f"listing-{index}",
                    "session_date": session,
                    "input_cutoffs_json": cutoff,
                }
                for session in sessions
            ),
            ineligibility=tuple(
                {
                    "session_date": session,
                    "factor_id": "rev_21",
                    "reason": "insufficient_history",
                    "observation_count": position + 1,
                }
                for position, session in enumerate(sessions)
            ),
            raw_input_hash=f"raw-{index}",
            action_set_hash_value="actions",
            market_reference_revision="spy",
            idempotency_key=f"write-{index}",
            revision_reason="initial",
            observed_at=NOW,
        )
        for index in range(8)
    )
    single_market = MarketDataRepository(tmp_path / "single")
    single = FeatureStateRepository(single_market.database, market_data=single_market)
    batched_market = MarketDataRepository(tmp_path / "batched")
    batched = FeatureStateRepository(batched_market.database, market_data=batched_market)
    for market_data, feature_state in (
        (single_market, single),
        (batched_market, batched),
    ):
        market_data.bootstrap(_manifest())
        feature_state.ensure_current_storage()
    for write in writes:
        single.upsert_feature_materialization(
            listing_id=write.listing_id,
            catalog_hash=write.catalog_hash,
            rows=write.rows,
            ineligibility=write.ineligibility,
            raw_input_hash=write.raw_input_hash,
            action_set_hash_value=write.action_set_hash_value,
            market_reference_revision=write.market_reference_revision,
            idempotency_key=write.idempotency_key,
            revision_reason=write.revision_reason,
            observed_at=write.observed_at,
        )
    batch_receipts = tuple(
        receipt
        for start in range(0, len(writes), batch_size)
        for receipt in batched.upsert_feature_materialization_batch(
            writes[start : start + batch_size]
        )
    )
    assert len(batch_receipts) == 8

    def durable_rows(feature_state: FeatureStateRepository) -> tuple[list[tuple], ...]:
        connection = duckdb.connect(str(feature_state.path), read_only=True)
        try:
            return tuple(
                connection.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall()
                for table in (
                    "feature_daily_current",
                    "feature_daily_revision",
                    "feature_materialization_receipt",
                    "feature_ineligibility_run",
                    "feature_ineligibility",
                )
            )
        finally:
            connection.close()

    assert durable_rows(single) == durable_rows(batched)
    before_replay = durable_rows(batched)
    replay_receipts = tuple(
        receipt
        for start in range(0, len(writes), batch_size)
        for receipt in batched.upsert_feature_materialization_batch(
            writes[start : start + batch_size]
        )
    )
    assert replay_receipts == batch_receipts
    assert durable_rows(batched) == before_replay


@pytest.mark.parametrize(
    ("stage", "crash_at"),
    (
        # The second listing's rows, the first listing's already in the batch's transaction.
        ("current_write", 2),
        # The batch's runs, inserted once after every listing's rows and receipts.
        ("rle_write", 1),
    ),
)
def test_batched_feature_persistence_rolls_back_the_bounded_batch(
    tmp_path, stage: str, crash_at: int
) -> None:
    sessions = tuple(pd.bdate_range("2026-01-02", periods=3).date)
    catalog = FeatureCatalog.load()
    cutoff = _empty_feature_cutoff(catalog)
    writes = tuple(
        FeatureMaterializationWrite(
            listing_id=f"listing-{index}",
            catalog_hash=catalog.binding.catalog_hash,
            rows=tuple(
                {
                    "listing_id": f"listing-{index}",
                    "session_date": session,
                    "input_cutoffs_json": cutoff,
                }
                for session in sessions
            ),
            ineligibility=(
                {
                    "session_date": sessions[0],
                    "factor_id": "rev_21",
                    "reason": "insufficient_history",
                    "observation_count": 1,
                },
            ),
            raw_input_hash=f"raw-{index}",
            action_set_hash_value="actions",
            market_reference_revision="spy",
            idempotency_key=f"write-{index}",
            revision_reason="initial",
            observed_at=NOW,
        )
        for index in range(4)
    )
    market_data = MarketDataRepository(tmp_path / "batch-crash")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(_manifest())
    feature_state.ensure_current_storage()
    observed = 0

    def fail_during_batch(name: str, _elapsed: float) -> None:
        nonlocal observed
        if name == stage:
            observed += 1
            if observed == crash_at:
                raise RuntimeError("fixture crash during batch")

    with pytest.raises(RuntimeError, match="fixture crash"):
        feature_state.upsert_feature_materialization_batch(writes, timing_sink=fail_during_batch)
    connection = duckdb.connect(str(market_data.path), read_only=True)
    try:
        for table in (
            "feature_daily_current",
            "feature_daily_revision",
            "feature_materialization_receipt",
            "feature_ineligibility_run",
        ):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    finally:
        connection.close()


def test_one_response_hydration_rejects_empty_payload(tmp_path, monkeypatch) -> None:
    fake = _YFinance(_history_frame().iloc[0:0])
    monkeypatch.setattr(YFinanceMarketDataProvider, "_module", lambda _self: fake)
    with pytest.raises(ProviderFetchError) as captured:
        YFinanceMarketDataProvider(tmp_path / "cache").fetch_hydration(
            listing_id="listing-aapl",
            provider_symbol="AAPL",
            start=date(2026, 1, 2),
            end=date(2026, 1, 6),
        )
    assert captured.value.code == "data.empty_payload"


def test_hydration_host_rejects_ticker_identity_mismatch() -> None:
    with pytest.raises(CorruptedPayload) as captured:
        sanitize_payload(
            _manifest(),
            "yfinance",
            {"MSFT": _rows((date(2026, 1, 2),))},
            ("AAPL",),
        )
    assert captured.value.code == "SYMBOL_MISMATCH"


def test_one_response_hydration_maps_shared_session_instability(tmp_path, monkeypatch) -> None:
    class UnstableTicker:
        def history(self, **_kwargs):
            raise RuntimeError("curl multi handle session already in use")

    class UnstableYFinance(_YFinance):
        def Ticker(self, _symbol: str):
            return UnstableTicker()

    fake = UnstableYFinance(_history_frame())
    monkeypatch.setattr(YFinanceMarketDataProvider, "_module", lambda _self: fake)
    with pytest.raises(ProviderFetchError) as captured:
        YFinanceMarketDataProvider(tmp_path / "cache").fetch_hydration(
            listing_id="listing-aapl",
            provider_symbol="AAPL",
            start=date(2026, 1, 2),
            end=date(2026, 1, 6),
        )
    assert captured.value.code == "data.provider_session_unstable"
