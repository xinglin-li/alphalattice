"""A sector map's entries take world's names, and a map world does not cover is refused."""

import pytest

from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionEntry
from alphalattice.foundation.feature_engine.panels.sector_classification import renamed

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/sector_classification.py"
ENTRY = SectorRevisionEntry(
    listing_id="listing-aapl",
    provider="fixture",
    provider_symbol="AAPL",
    sector_name="Technology",
    payload_hash="blake3:" + "a" * 64,
    evidence_hash="blake3:" + "b" * 64,
)


@pytest.mark.proves("feature_engine.sector_map_entries_take_world_names_or_are_refused")
@pytest.mark.parametrize(
    "names",
    [{"listing-aapl": "Technology"}, {}, None],
    ids=["world's names", "a listing world misses", "no world catalog"],
)
def test_a_maps_entries_take_worlds_names_or_are_refused(names: dict[str, str] | None) -> None:
    """Entries take world's names with their provenance; an uncovered listing is refused."""
    if names != {"listing-aapl": "Technology"}:
        with pytest.raises(ValueError, match=r"feature_closure.sector_map_unserved"):
            renamed((ENTRY,), names)
        return
    assert renamed((ENTRY,), names) == (ENTRY,)
