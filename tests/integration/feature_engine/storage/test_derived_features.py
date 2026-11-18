"""Feature values as a derived method: the build's writes reach the workspace's copy in the
catalog, and every read takes the same values from it as from the workspace's table."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.root import create_data_root
from alphalattice.kernel.data_layer.store import StorePort

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/storage/derived_features.py"
DAYS = (date(2025, 12, 30), date(2025, 12, 31), date(2026, 1, 2))
NOW = datetime(2026, 1, 3, 22, tzinfo=UTC)
SCOPE = (str(uuid.UUID(int=7)), str(uuid.UUID(int=8)))
OTHER = (SCOPE[0], str(uuid.UUID(int=9)))  # another workspace: one listing shared, one its own
SYMBOLS = dict(zip((*SCOPE, OTHER[1]), ("AAA", "BBB", "CCC"), strict=True))


def _upsert(feature, catalog, listing, value, reason):
    feature.upsert_feature_materialization(
        listing_id=listing,
        catalog_hash=catalog.binding.catalog_hash,
        rows=tuple(
            {
                "listing_id": listing,
                "session_date": day,
                "input_cutoffs_json": {factor: None for factor in catalog.factor_ids},
                **{factor: value + index for index, factor in enumerate(catalog.factor_ids)},
            }
            for day in DAYS
        ),
        ineligibility=(),
        raw_input_hash="2" * 64,
        action_set_hash_value="3" * 64,
        market_reference_revision="4" * 64,
        idempotency_key=f"{listing}-{reason}",
        revision_reason=reason,
        observed_at=NOW,
    )


def _reads(feature, catalog, monkeypatch, setting, scope=SCOPE):
    monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", setting)
    rows = feature.feature_rows(
        listing_ids=scope,
        catalog_hash=catalog.binding.catalog_hash,
        start=DAYS[0],
        end=DAYS[-1],
        as_frame=True,
    )
    proof = feature.feature_source_prefix_proof(
        listing_ids=scope,
        catalog_hash=catalog.binding.catalog_hash,
        start=DAYS[0],
        end=DAYS[-1],
        factor_ids=catalog.factor_ids,
    )
    return rows, proof


def _workspace(folder, scope=SCOPE):
    """A workspace holding the scope's listings, its Feature tables ready."""
    market = MarketDataRepository(folder)
    market.bootstrap(
        UniverseManifest(
            manifest_id="derived-features",
            profile=MarketProfile(
                market_profile_id="derived-features",
                display_name="Derived features",
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
                for listing, symbol in ((listing, SYMBOLS[listing]) for listing in scope)
            ),
            revision_sha256="1" * 64,
            universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
            is_point_in_time_historical=False,
        )
    )
    feature = FeatureStateRepository(market.database, market_data=market)
    feature.ensure_current_storage()
    return feature


@pytest.mark.proves("feature_engine.derived_feature_reads_equal_the_stored")
def test_the_builds_writes_reach_the_catalog_and_every_read_takes_equal_values(
    tmp_path, monkeypatch
):
    """Under derived, each workspace's reads equal its own table's, whatever another builds; a
    build's start maintains the catalog, its listing batches never."""
    root, folder, other = tmp_path / "data", tmp_path / "workspace", tmp_path / "other"
    create_data_root(root)
    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(root))
    catalog = FeatureCatalog.load()
    feature, neighbour = _workspace(folder), _workspace(other, OTHER)
    stranger = _workspace(tmp_path / "stranger", SCOPE[:1])
    history.open_history("derived-features", "derived-features", folder, root)
    maintained, maintain = [], StorePort.maintain
    monkeypatch.setattr(
        StorePort, "maintain", lambda port, now=None: maintained.append(1) or maintain(port, now)
    )
    try:
        monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
        for listing in SCOPE:
            _upsert(feature, catalog, listing, 1.0, "fixture")
        seeded = [_reads(feature, catalog, monkeypatch, s) for s in ("stored", "derived")]
        monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
        _upsert(feature, catalog, SCOPE[0], 9.0, "correction")
        corrected = [_reads(feature, catalog, monkeypatch, s) for s in ("stored", "derived")]
        with feature.database.transaction() as connection:  # a write the Host stopped uncopied
            connection.execute(
                f'UPDATE feature_daily_current SET "{catalog.factor_ids[0]}" = 7.0,'
                " row_hash = sha256(row_hash)"
            )
        stopped = _reads(feature, catalog, monkeypatch, "derived")[0]
        batches = len(maintained)  # every write so far was a listing batch
        monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
        with feature.database.transaction() as connection:  # the next build's start reconciles
            feature.retire_rows_outside_layer(_connection=connection)
        reconciled = [_reads(feature, catalog, monkeypatch, s) for s in ("stored", "derived")]
        monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
        with feature.database.transaction() as connection:  # a start with every batch mirrored
            feature.retire_rows_outside_layer(_connection=connection)
        started = len(maintained) - batches
    finally:
        history.close_history()
    history.open_history("other-features", "other-features", other, root)
    try:  # the same feature set and a shared listing, other values: never the first's rows
        monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
        _upsert(neighbour, catalog, OTHER[0], 5.0, "neighbour")  # its build part way
        partial = [_reads(neighbour, catalog, monkeypatch, s, OTHER) for s in ("stored", "derived")]
        built = _reads(feature, catalog, monkeypatch, "derived")
        monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
        _upsert(neighbour, catalog, OTHER[1], 6.0, "neighbour")
        with neighbour.database.transaction() as connection:  # it drops the shared listing
            connection.execute("DELETE FROM feature_daily_current WHERE listing_id = ?", [OTHER[0]])
        with neighbour.database.transaction() as connection:
            neighbour.retire_rows_outside_layer(_connection=connection)
        dropped = [_reads(neighbour, catalog, monkeypatch, s, OTHER) for s in ("stored", "derived")]
        after = _reads(feature, catalog, monkeypatch, "derived")
        monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
        with pytest.raises(DataError) as unknown:
            _upsert(stranger, catalog, SCOPE[0], 3.0, "stranger")
    finally:
        history.close_history()
    monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
    with pytest.raises(DataError) as refused:
        _upsert(feature, catalog, SCOPE[1], 5.0, "outside")
    for (stored_rows, stored_proof), (derived_rows, derived_proof) in (
        seeded,
        corrected,
        reconciled,
        partial,
        dropped,
        (reconciled[0], built),
        (reconciled[0], after),
    ):
        assert stored_rows.equals(derived_rows) and stored_proof == derived_proof
    assert len({partial[0][1], dropped[0][1], reconciled[0][1]}) == 3  # each its own rows
    assert not stopped.equals(reconciled[0][0])  # the stopped copy read stale until reconciled
    assert seeded[0][1] != corrected[0][1]  # the correction moved the values both read
    assert (batches, started) == (0, 2)  # each build's start maintained, moved or not
    assert refused.value.failure.code == "feature.derived_reads_need_host"
    assert unknown.value.failure.code == "feature.derived_reads_need_workspace"
