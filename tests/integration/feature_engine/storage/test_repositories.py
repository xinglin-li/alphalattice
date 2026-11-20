"""The Feature store's opening on a real workspace database: 0.2.8 upgrades only from 0.2.7, so a
workspace an earlier release last opened is refused by name before its schema is touched."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.errors import DataError

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/storage/repositories.py"


@pytest.mark.proves("feature_engine.workspace_before_0_2_7_refused_by_name")
@pytest.mark.parametrize(
    "retired",
    [
        pytest.param("feature_ineligibility", id="ineligibility-rows"),
        pytest.param("feature_year_seal", id="year-seals"),
    ],
)
def test_a_workspace_an_earlier_release_last_opened_is_refused_by_name(
    tmp_path: Path, retired: str
) -> None:
    """A table 0.2.7 retired at its open marks a workspace no 0.2.7 opened: its Feature storage
    is refused by name, with the release to open it with first, and its tables are left as
    they were."""
    market = MarketDataRepository(tmp_path)
    market.bootstrap(
        UniverseManifest(
            manifest_id="before-0-2-7",
            profile=MarketProfile(
                market_profile_id="before-0-2-7",
                display_name="Before 0.2.7",
                market="US",
                currency="USD",
                calendar_id="XNYS",
                provider="fixture",
                daily_price_basis="unadjusted",
                manifest_as_of=date(2026, 1, 2),
                data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
            ),
            listings=(
                ManifestListing(listing_id="L1", symbol="AAA", mic="XNYS", provider_symbol="AAA"),
            ),
            revision_sha256="1" * 64,
            universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
            is_point_in_time_historical=False,
        )
    )

    def tables() -> list[tuple[str, str]]:
        with duckdb.connect(str(market.path), read_only=True) as connection:
            return connection.execute(
                "SELECT table_name, table_type FROM information_schema.tables ORDER BY table_name"
            ).fetchall()

    with duckdb.connect(str(market.path)) as connection:
        connection.execute(f"CREATE TABLE {retired} (listing_id VARCHAR)")
    before = tables()
    feature = FeatureStateRepository(market.database, market_data=market)
    with pytest.raises(DataError, match=r"open it with 0\.2\.7 first") as refused:
        feature.ensure_current_storage()
    assert refused.value.failure.code == "feature_storage.workspace_before_0_2_7"
    assert tables() == before
