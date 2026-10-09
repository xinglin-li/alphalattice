"""Fresh selected-value proofs over public Feature source storage."""

from __future__ import annotations

import json
import struct
from contextlib import nullcontext
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import duckdb
import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.database import (
    WorkspaceDatabase,
    live_workspace_connections,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

DAYS = (date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5))
NOW = datetime(2026, 8, 7, 22, tzinfo=UTC)
SCOPE = ("listing-a", "listing-b")


@pytest.fixture
def feature_source(tmp_path):
    catalog = FeatureCatalog.load()
    market = MarketDataRepository(tmp_path / "workspace")
    market.bootstrap(
        UniverseManifest(
            manifest_id="feature-prefix-fixture",
            profile=MarketProfile(
                market_profile_id="feature-prefix-fixture",
                display_name="Feature prefix fixture",
                market="US",
                currency="USD",
                calendar_id="XNYS",
                provider="fixture",
                daily_price_basis="unadjusted",
                manifest_as_of=NOW.date(),
                data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
            ),
            listings=tuple(
                ManifestListing(
                    listing_id=listing, symbol=symbol, mic="XNYS", provider_symbol=symbol
                )
                for listing, symbol in zip(SCOPE, ("AAA", "BBB"), strict=True)
            ),
            revision_sha256="1" * 64,
            universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
            is_point_in_time_historical=False,
        )
    )
    feature = FeatureStateRepository(market.database, market_data=market)
    feature.ensure_current_storage()
    for listing in SCOPE:
        feature.upsert_feature_materialization(
            listing_id=listing,
            catalog_hash=catalog.binding.catalog_hash,
            rows=tuple(
                {
                    "listing_id": listing,
                    "session_date": day,
                    "input_cutoffs_json": {factor: None for factor in catalog.factor_ids},
                    **{factor: float(index + 1) for index, factor in enumerate(catalog.factor_ids)},
                }
                for day in DAYS
            ),
            ineligibility=(),
            raw_input_hash="2" * 64,
            action_set_hash_value="3" * 64,
            market_reference_revision="4" * 64,
            idempotency_key=listing,
            revision_reason="fixture",
            observed_at=NOW,
        )
    return feature, catalog


def _proof(
    feature,
    catalog,
    *,
    listing_ids=SCOPE,
    factors=None,
    start=DAYS[0],
    end=DAYS[1],
    connection=None,
):
    return feature.feature_source_prefix_proof(
        listing_ids=listing_ids,
        catalog_hash=catalog.binding.catalog_hash,
        start=start,
        end=end,
        factor_ids=factors if factors is not None else catalog.factor_ids[:3],
        _connection=connection,
    )


def _batch_proofs(
    feature,
    catalog,
    *,
    listing_ids=SCOPE,
    factors=None,
    start=DAYS[0],
    ends=DAYS,
    connection=None,
):
    return feature.feature_source_prefix_proofs(
        listing_ids=listing_ids,
        catalog_hash=catalog.binding.catalog_hash,
        start=start,
        ends=ends,
        factor_ids=factors if factors is not None else catalog.factor_ids[:3],
        _connection=connection,
    )


def _original_v1_proof(connection, *, listing_ids, catalog_hash, start, end, factor_ids):
    """Frozen pre-batch values hash, in the year-sealed envelope of a prefix in one year.

    This oracle queries each bound directly and normalizes its Arrow IPC; it does not call the
    batch/scalar implementation or a private production hash helper. It retains the original SQL
    column order, null bits, IEEE values and UTF-8 request envelope, including empty results. A
    prefix within its cutoff's year holds no closed year, so the envelope binds just these values.
    """
    assert start.year == end.year
    scope, factors = tuple(sorted(listing_ids)), tuple(sorted(factor_ids))
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info('feature_daily_runtime')").fetchall()
    }
    verification = (
        "source_verification_receipt_hash"
        if "source_verification_receipt_hash" in columns
        else "NULL::VARCHAR"
    )
    projection = ", ".join(
        f'COALESCE("{factor}", 0::DOUBLE) AS value_{index}, "{factor}" IS NULL AS null_{index}'
        for index, factor in enumerate(factors)
    )
    rows = connection.execute(
        f"""
        SELECT listing_id, session_date, catalog_hash,
               COALESCE(raw_input_hash, '') AS raw_input_hash,
               raw_input_hash IS NULL AS raw_input_is_null,
               COALESCE(action_set_hash, '') AS action_set_hash,
               action_set_hash IS NULL AS action_set_is_null,
               COALESCE(market_reference_revision, '') AS market_reference_revision,
               market_reference_revision IS NULL AS market_reference_is_null,
               COALESCE(input_cutoffs_json, '') AS input_cutoffs_json,
               input_cutoffs_json IS NULL AS cutoffs_is_null,
               COALESCE({verification}, '') AS source_verification_receipt_hash,
               {verification} IS NULL AS verification_is_null,
               {projection}
        FROM feature_daily_runtime
        WHERE listing_id IN (SELECT unnest(?)) AND catalog_hash = ?
          AND session_date BETWEEN ? AND ?
        ORDER BY session_date, listing_id
        """,
        [scope, catalog_hash, start, end],
    ).to_arrow_table()
    normalized = rows.replace_schema_metadata(None).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, normalized.schema) as writer:
        writer.write_table(normalized)
    values_hash = sha256(sink.getvalue()).hexdigest()
    payload = {
        "kind": "FeatureSourcePrefixProofV2",
        "listing_ids": scope,
        "catalog_hash": catalog_hash,
        "start": start,
        "end": end,
        "factor_ids": factors,
        "closed_years": [],
        "open_values": values_hash,
    }
    return sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _assert_batch_matches_original(feature, catalog, **request):
    listing_ids = request.get("listing_ids", SCOPE)
    factors = request.get("factors", catalog.factor_ids[:3])
    start = request.get("start", DAYS[0])
    ends = request.get("ends", DAYS)
    with feature.database.read_transaction() as connection:
        expected = tuple(
            _original_v1_proof(
                connection,
                listing_ids=listing_ids,
                catalog_hash=catalog.binding.catalog_hash,
                start=start,
                end=end,
                factor_ids=factors,
            )
            for end in ends
        )
        actual = _batch_proofs(feature, catalog, **request, connection=connection)
        assert actual == expected
        assert actual == tuple(
            _proof(
                feature,
                catalog,
                listing_ids=listing_ids,
                factors=factors,
                start=start,
                end=end,
                connection=connection,
            )
            for end in ends
        )
    return actual


def test_feature_prefix_detects_value_tamper_without_row_hash_or_revision_change(feature_source):
    feature, catalog = feature_source
    before = _proof(feature, catalog)
    rows = feature.feature_rows(
        listing_ids=SCOPE,
        catalog_hash=catalog.binding.catalog_hash,
        start=DAYS[0],
        end=DAYS[1],
        include_values=False,
    )
    revisions = tuple(feature.feature_revision_count(listing) for listing in SCOPE)
    with feature.database.read_transaction() as connection:
        assert _proof(feature, catalog, connection=connection) == before
    factor = catalog.factor_ids[0]
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            f'UPDATE feature_daily_current SET "{factor}" = "{factor}" + 1 WHERE session_date = ?',
            [DAYS[0]],
        )
    assert (
        feature.feature_rows(
            listing_ids=SCOPE,
            catalog_hash=catalog.binding.catalog_hash,
            start=DAYS[0],
            end=DAYS[1],
            include_values=False,
        )
        == rows
    )
    assert tuple(feature.feature_revision_count(listing) for listing in SCOPE) == revisions
    assert _proof(feature, catalog) != before


@pytest.mark.parametrize(
    "column",
    (
        "raw_input_hash",
        "action_set_hash",
        "market_reference_revision",
        "source_verification_receipt_hash",
    ),
)
def test_feature_prefix_binds_actual_source_and_verification_lineage(feature_source, column):
    feature, catalog = feature_source
    before = _proof(feature, catalog)
    values = feature.feature_rows(
        listing_ids=SCOPE,
        catalog_hash=catalog.binding.catalog_hash,
        start=DAYS[0],
        end=DAYS[1],
        factor_ids=catalog.factor_ids[:3],
        include_lineage=False,
    )
    with feature.database.connect(read_only=False) as connection:
        connection.execute(f"UPDATE feature_daily_current SET {column} = ?", ["f" * 64])
    assert (
        feature.feature_rows(
            listing_ids=SCOPE,
            catalog_hash=catalog.binding.catalog_hash,
            start=DAYS[0],
            end=DAYS[1],
            factor_ids=catalog.factor_ids[:3],
            include_lineage=False,
        )
        == values
    )
    assert _proof(feature, catalog) != before


def test_feature_prefix_binds_actual_cutoffs_without_trusting_their_stored_hash(feature_source):
    feature, catalog = feature_source
    before = _proof(feature, catalog)
    cutoffs = dict.fromkeys(catalog.factor_ids)
    cutoffs[catalog.factor_ids[0]] = DAYS[0].isoformat()
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            "UPDATE feature_input_cutoff_set SET input_cutoffs_json = ?",
            [json.dumps(cutoffs, sort_keys=True, separators=(",", ":"))],
        )
    assert _proof(feature, catalog) != before


def test_feature_prefix_reads_only_declared_values_dates_and_listings(feature_source):
    feature, catalog = feature_source
    before = _proof(feature, catalog, listing_ids=(SCOPE[0],))
    chosen, unchosen = catalog.factor_ids[0], catalog.factor_ids[3]
    with feature.database.connect(read_only=False) as connection:
        connection.execute(f'UPDATE feature_daily_current SET "{unchosen}" = 98.0')
        connection.execute(
            f'UPDATE feature_daily_current SET "{chosen}" = 99.0 '
            "WHERE session_date = ? OR listing_id = ?",
            [DAYS[2], SCOPE[1]],
        )
    assert _proof(feature, catalog, listing_ids=(SCOPE[0],)) == before
    assert _proof(feature, catalog, listing_ids=(SCOPE[0],), end=DAYS[2]) != before
    assert _proof(feature, catalog, listing_ids=(SCOPE[1],)) != before
    assert _proof(feature, catalog, listing_ids=(SCOPE[0],), factors=(chosen,)) != before
    full = _proof(feature, catalog)
    assert (
        _proof(
            feature,
            catalog,
            listing_ids=tuple(reversed(SCOPE)),
            factors=tuple(reversed(catalog.factor_ids[:3])),
        )
        == full
    )


def test_feature_prefix_keeps_null_nan_negative_zero_and_absent_rows_distinct(feature_source):
    feature, catalog = feature_source
    factors = catalog.factor_ids[:3]
    # DuckDB can retain +0 on an equal-numeric UPDATE. Ingest heterogeneous
    # doubles through the public writer, force a nonzero transition, and pin
    # the real public reader's IEEE bits rather than assuming the write changed them.
    positive_zero = None
    positive_zero_batch = None
    for stage, value in (("positive", 0.0), ("transition", 1.5), ("negative", -0.0)):
        feature.upsert_feature_materialization(
            listing_id=SCOPE[0],
            catalog_hash=catalog.binding.catalog_hash,
            rows=tuple(
                {
                    "listing_id": SCOPE[0],
                    "session_date": day,
                    "input_cutoffs_json": {factor: None for factor in catalog.factor_ids},
                    **{factor: float(index + 1) for index, factor in enumerate(catalog.factor_ids)},
                    factors[0]: value if day == DAYS[0] else 1.125,
                }
                for day in DAYS
            ),
            ineligibility=(),
            raw_input_hash="2" * 64,
            action_set_hash_value="3" * 64,
            market_reference_revision="4" * 64,
            idempotency_key=f"feature-prefix-zero-{stage}",
            revision_reason="fixture",
            observed_at=NOW,
        )
        actual = feature.feature_rows(
            listing_ids=(SCOPE[0],),
            catalog_hash=catalog.binding.catalog_hash,
            start=DAYS[0],
            end=DAYS[0],
            factor_ids=(factors[0],),
        )[0][factors[0]]
        assert struct.pack("!d", actual) == struct.pack("!d", value)
        if stage == "positive":
            positive_zero = _proof(feature, catalog)
            positive_zero_batch = _assert_batch_matches_original(feature, catalog)
    assert _proof(feature, catalog) != positive_zero
    assert _assert_batch_matches_original(feature, catalog) != positive_zero_batch
    with feature.database.connect(read_only=False) as connection:
        connection.execute(f'UPDATE feature_daily_current SET "{factors[1]}" = NULL')
    null_cells = _proof(feature, catalog)
    null_cells_batch = _assert_batch_matches_original(feature, catalog)
    with feature.database.connect(read_only=False) as connection:
        connection.execute(f"UPDATE feature_daily_current SET \"{factors[1]}\" = 'nan'::DOUBLE")
    assert _proof(feature, catalog) != null_cells
    assert _assert_batch_matches_original(feature, catalog) != null_cells_batch
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            "UPDATE feature_daily_current SET "
            + ", ".join(f'"{factor}" = NULL' for factor in factors)
        )
    null_rows = _proof(feature, catalog)
    null_rows_batch = _assert_batch_matches_original(feature, catalog)
    with feature.database.connect(read_only=False) as connection:
        connection.execute("DELETE FROM feature_daily_current WHERE session_date = ?", [DAYS[0]])
    assert _proof(feature, catalog) != null_rows
    assert _assert_batch_matches_original(feature, catalog) != null_rows_batch


def test_feature_prefix_refuses_invalid_or_unadmitted_selection(feature_source):
    feature, catalog = feature_source
    for scope in ((), (SCOPE[0], SCOPE[0])):
        with pytest.raises(ValueError, match=r"feature\.source_prefix_request_invalid"):
            _proof(feature, catalog, listing_ids=scope)
    for factors in ((), (catalog.factor_ids[0], catalog.factor_ids[0])):
        with pytest.raises(ValueError, match=r"feature\.source_prefix_request_invalid"):
            _proof(feature, catalog, factors=factors)
    with pytest.raises(ValueError, match=r"feature\.source_prefix_request_invalid"):
        _proof(feature, catalog, start=DAYS[2])
    with pytest.raises(ValueError, match="feature row projection contains an unknown factor"):
        _proof(feature, catalog, factors=("unadmitted-factor",))


PREFIX_DAYS = tuple(date(2025, 1, 1) + timedelta(days=index) for index in range(67))
PREFIX_COUNTS = (*range(1, 10), 15, 16, 17, 63, 64, 65)
PREFIX_ENDS = tuple(PREFIX_DAYS[count - 1] for count in PREFIX_COUNTS)


@pytest.fixture
def feature_prefixes(feature_source):
    feature, catalog = feature_source
    first, second = catalog.factor_ids[:2]
    feature.upsert_feature_materialization(
        listing_id=SCOPE[0],
        catalog_hash=catalog.binding.catalog_hash,
        rows=tuple(
            {
                "listing_id": SCOPE[0],
                "session_date": day,
                "input_cutoffs_json": {factor: None for factor in catalog.factor_ids},
                **{factor: float(index + 1) for index, factor in enumerate(catalog.factor_ids)},
                first: -0.0 if position % 2 else float(position) / 8.0,
                second: None if position % 3 == 0 else float(position) / 16.0,
            }
            for position, day in enumerate(PREFIX_DAYS)
        ),
        ineligibility=(),
        raw_input_hash="2" * 64,
        action_set_hash_value="3" * 64,
        market_reference_revision="4" * 64,
        idempotency_key="feature-prefix-padding",
        revision_reason="fixture",
        observed_at=NOW,
    )
    # A valid NaN and a null are different stored cells; retain the former in
    # DuckDB explicitly instead of relying on a writer's pandas conversion.
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            f"UPDATE feature_daily_current SET \"{second}\" = 'nan'::DOUBLE "
            "WHERE listing_id = ? AND session_date IN (SELECT unnest(?))",
            [SCOPE[0], tuple(PREFIX_DAYS[::5])],
        )
    return feature, catalog


def _padding_request(catalog):
    return {
        "listing_ids": (SCOPE[0],),
        "factors": catalog.factor_ids[:2],
        "start": PREFIX_DAYS[0],
        "ends": PREFIX_ENDS,
    }


def test_feature_prefix_batch_keeps_order_duplicate_ends_and_sorted_axes(feature_source):
    feature, catalog = feature_source
    ends = (DAYS[2], DAYS[0], DAYS[1], DAYS[0], DAYS[2])
    expected = _assert_batch_matches_original(feature, catalog, ends=ends)
    assert isinstance(expected, tuple)
    assert expected[0] == expected[4]
    assert expected[1] == expected[3]
    assert len(set(expected)) == 3
    assert (
        _assert_batch_matches_original(
            feature,
            catalog,
            listing_ids=tuple(reversed(SCOPE)),
            factors=tuple(reversed(catalog.factor_ids[:3])),
            ends=ends,
        )
        == expected
    )
    empty = _assert_batch_matches_original(
        feature, catalog, listing_ids=("absent-listing-汉",), ends=ends
    )
    assert len(set(empty)) == 3
    assert empty != expected


def test_feature_prefix_batch_preserves_original_boolean_padding_at_each_bound(feature_prefixes):
    feature, catalog = feature_prefixes
    request = _padding_request(catalog)
    proofs = _assert_batch_matches_original(feature, catalog, **request)
    assert len(proofs) == len(PREFIX_COUNTS)
    with feature.database.read_transaction() as connection:
        counts = tuple(
            connection.execute(
                "SELECT count(*) FROM feature_daily_runtime WHERE listing_id = ? "
                "AND catalog_hash = ? AND session_date BETWEEN ? AND ?",
                [SCOPE[0], catalog.binding.catalog_hash, PREFIX_DAYS[0], end],
            ).fetchone()[0]
            for end in PREFIX_ENDS
        )
    assert counts == PREFIX_COUNTS


def test_feature_prefix_batch_corrections_enter_only_their_exact_cutoffs(feature_prefixes):
    feature, catalog = feature_prefixes
    request = _padding_request(catalog)
    before = _assert_batch_matches_original(feature, catalog, **request)
    chosen, unchosen = catalog.factor_ids[0], catalog.factor_ids[2]
    with feature.database.connect(read_only=False) as connection:
        connection.execute(f'UPDATE feature_daily_current SET "{unchosen}" = 81.0')
    assert _assert_batch_matches_original(feature, catalog, **request) == before
    future_before = _assert_batch_matches_original(
        feature, catalog, **{**request, "ends": (PREFIX_DAYS[-1],)}
    )
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            f'UPDATE feature_daily_current SET "{chosen}" = 99.0 '
            "WHERE listing_id = ? AND session_date = ?",
            [SCOPE[0], PREFIX_DAYS[-1]],
        )
    assert _assert_batch_matches_original(feature, catalog, **request) == before
    assert (
        _assert_batch_matches_original(feature, catalog, **{**request, "ends": (PREFIX_DAYS[-1],)})
        != future_before
    )
    corrected = PREFIX_DAYS[8]
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            f'UPDATE feature_daily_current SET "{chosen}" = -3.0 '
            "WHERE listing_id = ? AND session_date = ?",
            [SCOPE[0], corrected],
        )
    after = _assert_batch_matches_original(feature, catalog, **request)
    assert tuple(old != new for old, new in zip(before, after, strict=True)) == tuple(
        end >= corrected for end in PREFIX_ENDS
    )


@pytest.mark.parametrize(
    "column",
    (
        "raw_input_hash",
        "action_set_hash",
        "market_reference_revision",
        "source_verification_receipt_hash",
    ),
)
def test_feature_prefix_batch_lineage_enters_only_the_recorded_session(feature_prefixes, column):
    feature, catalog = feature_prefixes
    request = _padding_request(catalog)
    before = _assert_batch_matches_original(feature, catalog, **request)
    corrected = PREFIX_DAYS[4]
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            f"UPDATE feature_daily_current SET {column} = ? "
            "WHERE listing_id = ? AND session_date = ?",
            ["f" * 64, SCOPE[0], corrected],
        )
    after = _assert_batch_matches_original(feature, catalog, **request)
    assert tuple(old != new for old, new in zip(before, after, strict=True)) == tuple(
        end >= corrected for end in PREFIX_ENDS
    )


def test_feature_prefix_batch_reads_actual_cutoffs_without_trusting_the_stored_hash(
    feature_prefixes,
):
    feature, catalog = feature_prefixes
    request = _padding_request(catalog)
    before = _assert_batch_matches_original(feature, catalog, **request)
    corrected = PREFIX_DAYS[8]
    cutoffs = dict.fromkeys(catalog.factor_ids)
    cutoffs[catalog.factor_ids[0]] = PREFIX_DAYS[2].isoformat()
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            "INSERT INTO feature_input_cutoff_set VALUES (?, ?, ?)",
            ["e" * 64, json.dumps(cutoffs, sort_keys=True, separators=(",", ":")), len(cutoffs)],
        )
        connection.execute(
            "UPDATE feature_daily_current SET cutoff_set_hash = ? "
            "WHERE listing_id = ? AND session_date = ?",
            ["e" * 64, SCOPE[0], corrected],
        )
    after = _assert_batch_matches_original(feature, catalog, **request)
    assert tuple(old != new for old, new in zip(before, after, strict=True)) == tuple(
        end >= corrected for end in PREFIX_ENDS
    )
    # The same stored hash now names different actual JSON; no counter decides
    # which prefix moved, and a later proof cannot trust that stored hash.
    cutoffs[catalog.factor_ids[0]] = PREFIX_DAYS[3].isoformat()
    with feature.database.connect(read_only=False) as connection:
        connection.execute(
            "UPDATE feature_input_cutoff_set SET input_cutoffs_json = ? WHERE cutoff_set_hash = ?",
            [json.dumps(cutoffs, sort_keys=True, separators=(",", ":")), "e" * 64],
        )
    changed = _assert_batch_matches_original(feature, catalog, **request)
    assert tuple(old != new for old, new in zip(after, changed, strict=True)) == tuple(
        end >= corrected for end in PREFIX_ENDS
    )


@pytest.mark.parametrize(
    "case",
    (
        "empty-listings",
        "duplicate-listings",
        "empty-listing-name",
        "non-string-listing",
        "empty-factors",
        "duplicate-factors",
        "empty-factor-name",
        "empty-ends",
        "end-before-start",
    ),
)
def test_feature_prefix_batch_refuses_invalid_requests(feature_source, case):
    feature, catalog = feature_source
    requests = {
        "empty-listings": {"listing_ids": ()},
        "duplicate-listings": {"listing_ids": (SCOPE[0], SCOPE[0])},
        "empty-listing-name": {"listing_ids": ("",)},
        "non-string-listing": {"listing_ids": (1,)},
        "empty-factors": {"factors": ()},
        "duplicate-factors": {"factors": (catalog.factor_ids[0], catalog.factor_ids[0])},
        "empty-factor-name": {"factors": ("",)},
        "empty-ends": {"ends": ()},
        "end-before-start": {"ends": (DAYS[1], DAYS[0] - timedelta(days=1))},
    }
    with pytest.raises(ValueError) as seen:
        _batch_proofs(feature, catalog, **requests[case])
    assert str(seen.value) == "feature.source_prefix_request_invalid"


def test_feature_prefix_batch_refuses_unadmitted_factors_and_catalog(feature_source):
    feature, catalog = feature_source
    with pytest.raises(ValueError, match="feature row projection contains an unknown factor"):
        _batch_proofs(feature, catalog, factors=("unadmitted-factor",))
    with pytest.raises(ValueError, match="feature catalog hash does not match the active binding"):
        feature.feature_source_prefix_proofs(
            listing_ids=SCOPE,
            catalog_hash="a" * 64,
            start=DAYS[0],
            ends=DAYS,
            factor_ids=catalog.factor_ids[:3],
        )


@pytest.mark.parametrize("writable_retention", (False, True))
def test_feature_prefix_batch_uses_the_supplied_snapshot_without_write_authority(
    feature_source, writable_retention
):
    feature, catalog = feature_source
    before = _assert_batch_matches_original(feature, catalog)
    stored_bytes = feature.database.path.read_bytes()
    retention = feature.database.retain(read_only=False) if writable_retention else nullcontext()
    with (
        retention,
        pytest.raises(duckdb.Error, match="read-only"),
        feature.database.read_transaction() as connection,
    ):
        assert _batch_proofs(feature, catalog, connection=connection) == before
        assert _proof(feature, catalog, connection=connection) == before[1]
        connection.execute("DELETE FROM feature_daily_current")
    assert _assert_batch_matches_original(feature, catalog) == before
    assert feature.database.path.read_bytes() == stored_bytes


@pytest.mark.parametrize(
    "counts,cutoffs_text",
    (
        ((999_999, 1_000_000, 1_000_001), "{}"),
        (
            (1_000_001, 1_000_007, 1_000_017),
            '{"scope":"' + "示例-" * 64 + '","nullable":null}',
        ),
    ),
)
def test_feature_prefix_batch_matches_original_v1_across_the_arrow_chunk_boundary(
    tmp_path, counts, cutoffs_text
):
    """Keep original small boundaries and independently prove multi-chunk earlier prefixes; as
    a proof reads only its cutoff's year, the rows past the chunk boundary are listings of three
    sessions, the first holding the smallest count, each later one the rows up to the next."""
    catalog = FeatureCatalog.load()
    first, second = catalog.factor_ids[:2]
    database = WorkspaceDatabase(tmp_path / "chunk-boundary")
    feature = FeatureStateRepository(database, installed_catalog=catalog)
    start = date(2024, 1, 1)
    small, middle, _large = counts
    with database.connect(read_only=False) as connection:
        connection.execute(
            f"""
            CREATE VIEW feature_daily_runtime AS
            SELECT 'L' || lpad(CAST(i AS VARCHAR), 7, '0') AS listing_id,
                   DATE '{start.isoformat()}'
                       + CASE WHEN i < {small} THEN 0 WHEN i < {middle} THEN 1 ELSE 2 END
                       AS session_date,
                   '{catalog.binding.catalog_hash}'::VARCHAR AS catalog_hash,
                   CASE WHEN i % 3 = 0 THEN NULL ELSE 'raw' END::VARCHAR AS raw_input_hash,
                   CASE WHEN i % 5 = 0 THEN NULL ELSE 'action' END::VARCHAR AS action_set_hash,
                   CASE WHEN i % 7 = 0 THEN NULL ELSE 'market' END::VARCHAR
                       AS market_reference_revision,
                   CASE WHEN i % 11 = 0 THEN NULL ELSE '{cutoffs_text}' END::VARCHAR
                       AS input_cutoffs_json,
                   CASE WHEN i % 2 = 0 THEN NULL ELSE 'verified' END::VARCHAR
                       AS source_verification_receipt_hash,
                   CASE WHEN i % 17 = 0 THEN NULL ELSE i / 8.0 END::DOUBLE AS "{first}",
                   CASE WHEN i % 19 = 0 THEN 'nan'::DOUBLE
                        WHEN i % 5 = 0 THEN '-0.0'::DOUBLE
                        ELSE ((i % 64) - 32) / 16.0 END::DOUBLE AS "{second}"
            FROM range({max(counts)}) AS generated(i)
            """
        )
    ends = tuple(start + timedelta(days=day) for day in range(3))
    result = _assert_batch_matches_original(
        feature,
        catalog,
        listing_ids=tuple(f"L{i:07d}" for i in range(max(counts))),
        factors=(first, second),
        start=start,
        ends=ends,
    )
    assert len(set(result)) == 3
    with database.read_transaction() as connection:
        assert (
            tuple(
                connection.execute(
                    "SELECT count(*) FROM feature_daily_runtime WHERE session_date <= ?", [end]
                ).fetchone()[0]
                for end in ends
            )
            == counts
        )


OLDER_DAYS = (date(2025, 12, 30), date(2025, 12, 31))
"""A closed year beside the fixture's 2026 sessions."""


def _write_older_year(feature, catalog, *, offset=1.0, listings=SCOPE):
    for listing in listings:
        feature.upsert_feature_materialization(
            listing_id=listing,
            catalog_hash=catalog.binding.catalog_hash,
            rows=tuple(
                {
                    "listing_id": listing,
                    "session_date": day,
                    "input_cutoffs_json": {factor: None for factor in catalog.factor_ids},
                    **{factor: index + offset for index, factor in enumerate(catalog.factor_ids)},
                }
                for day in OLDER_DAYS
            ),
            ineligibility=(),
            raw_input_hash="2" * 64,
            action_set_hash_value="3" * 64,
            market_reference_revision="4" * 64,
            idempotency_key=f"older-{listing}-{offset}",
            revision_reason="fixture",
            observed_at=NOW,
        )


def _seal(feature, catalog):
    with feature.database.connect(read_only=False) as connection:
        return feature.seal_closed_feature_years(
            catalog.binding.catalog_hash, _connection=connection
        )


def _sealed_years(feature):
    with feature.database.read_transaction() as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT year FROM verified_year_fact WHERE kind = 'feature_year'"
            ).fetchall()
        ]


def _across_years(feature, catalog):
    return _proof(feature, catalog, start=OLDER_DAYS[0], end=DAYS[1])


def _without_seals(feature, catalog):
    with feature.database.connect(read_only=False) as connection:
        connection.execute("DROP TABLE verified_year_fact")  # a store from before the facts
    return _across_years(feature, catalog)


def test_a_closed_year_answers_by_its_seal_as_its_values_would(feature_source):
    """requirement: a proof reads only its cutoff's year; each earlier
    year answers by its seal, which equals a digest of that year's values, written once."""
    feature, catalog = feature_source
    _write_older_year(feature, catalog)
    unsealed = _across_years(feature, catalog)
    assert _seal(feature, catalog) == (2025,)
    assert _seal(feature, catalog) == ()
    assert _sealed_years(feature) == [2025]
    assert _across_years(feature, catalog) == unsealed


def test_a_write_to_a_closed_year_removes_its_seal_and_moves_the_proof(feature_source):
    """requirement: every Feature writer ends its years' seals in its own
    transaction, so a correction to a closed year moves the proof the next day."""
    feature, catalog = feature_source
    _write_older_year(feature, catalog)
    assert _seal(feature, catalog) == (2025,)
    before = _across_years(feature, catalog)
    _write_older_year(feature, catalog, offset=2.0, listings=SCOPE[:1])
    assert _sealed_years(feature) == []
    after = _across_years(feature, catalog)
    assert after != before
    assert _seal(feature, catalog) == (2025,)
    assert _across_years(feature, catalog) == after == _without_seals(feature, catalog)


@pytest.mark.parametrize("record", ["kept", "deleted"])
def test_an_edit_while_the_store_is_closed_ends_every_seal(feature_source, record):
    """tamper: another program's edit while the file was closed ends every
    seal at the next open, read only or writable, with its seal record kept or deleted (a
    counted epoch restarted from a deleted record and revived old seals)."""
    feature, catalog = feature_source
    _write_older_year(feature, catalog)
    assert _seal(feature, catalog) == (2025,)
    before = _across_years(feature, catalog)
    assert live_workspace_connections(feature.database.path) is None
    factor = catalog.factor_ids[0]
    outside = duckdb.connect(str(feature.database.path))
    try:
        outside.execute(
            f'UPDATE feature_daily_current SET "{factor}" = "{factor}" + 1 WHERE session_date = ?',
            [OLDER_DAYS[0]],
        )
    finally:
        outside.close()
    if record == "deleted":
        Path(f"{feature.database.path}.seal.json").unlink()
    after = _across_years(feature, catalog)
    assert after != before
    assert _seal(feature, catalog) == (2025,)
    assert _across_years(feature, catalog) == after == _without_seals(feature, catalog)
