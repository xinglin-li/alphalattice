"""A sector map's names through world: selected at its first Host read only when world holds
exactly its names, read at that cutoff after, and its own entries otherwise."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.database import open_workspace_database
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    SectorRevisionEntry,
    SectorRevisionMap,
    durable_payload,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    identified,
    sector_history_as_of,
)
from alphalattice.foundation.feature_engine.panels.sector_classification import classified
from alphalattice.foundation.market_data_ops.storage import world, world_reads
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests._support.identity_fixtures import fixture_digest
from tests.workspace_maintenance.acquisition_manifest import NOW, acquisition_manifest

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/sector_classification.py"
NAMES = {"listing-aapl": "Technology", "listing-msft": "Technology"}
_PAYLOAD = fixture_digest("sector-payload")
_EVIDENCE = fixture_digest("sector-evidence")


def _map(names: dict[str, str]) -> SectorRevisionMap:
    """A map over `names`, as an activation publishes one."""
    entries = tuple(
        SectorRevisionEntry(
            listing_id=listing,
            provider="fixture",
            provider_symbol=listing.removeprefix("listing-").upper(),
            sector_name=name,
            payload_hash=_PAYLOAD,
            evidence_hash=_EVIDENCE,
        )
        for listing, name in sorted(names.items())
    )
    return identified(
        SectorRevisionMap,
        {
            "manifest_revision": fixture_digest("manifest-revision"),
            "sector_revision": fixture_digest("sector-revision"),
            "entries": entries,
        },
        "map_hash",
    )


def _classify(database: Path, names: dict[str, str]) -> None:
    """The workspace's current classifications set to `names`, then admitted into world."""
    connection = open_workspace_database(database, read_only=False)
    try:
        connection.execute("DELETE FROM sector_classification_current")
        for listing, name in sorted(names.items()):
            connection.execute(
                "INSERT INTO sector_classification_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    listing,
                    "fixture",
                    listing,
                    name,
                    None,
                    _PAYLOAD,
                    _EVIDENCE,
                    fixture_digest("admitted-sector-revision"),
                    NOW,
                ],
            )
    finally:
        connection.close()
    world.admit_sectors(database, "w1")


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    """A workspace's listings admitted into world, in a Host on a data root of its own."""
    store = MarketDataRepository(tmp_path / "workspace", universe_reader="workspace")
    store.bootstrap(acquisition_manifest())
    history.open_history("w1", "workspace", tmp_path / "workspace", root=tmp_path / "data")
    database = tmp_path / "workspace" / "market-data.duckdb"
    world.admit(database, "w1")
    try:
        yield database
    finally:
        history.release_history()


def _selections() -> list[tuple[str]]:
    port = history.store()
    assert port is not None
    return [
        tuple(row)
        for row in port.log.read("SELECT json_extract(spec, '$.map_hash') FROM selection")
    ]


@pytest.mark.proves("feature_engine.sector_maps_read_their_names_through_world")
@pytest.mark.parametrize(
    "case",
    [
        "world holds its names",
        "a world revision after its cutoff",
        "read first after a world revision",
        "outside the Host",
    ],
)
def test_a_sector_map_reads_its_names_through_world_at_its_selection(
    workspace: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """A map selected at its first Host read takes world's names at its cutoff, else its own."""
    database = workspace
    sector_map = _map(NAMES)
    revised = NAMES | {"listing-aapl": "Energy"}
    artifacts = PanelClosureArtifactStore(ArtifactResolver(tmp_path / "workspace" / "artifacts"))
    artifacts.publish_json(
        category="sector-maps",
        content_hash=sector_map.map_hash,
        payload=durable_payload(sector_map),
    )
    _classify(database, revised if case == "read first after a world revision" else NAMES)
    if case == "outside the Host":
        history.release_history()

    assert classified(sector_map) == sector_map.entries
    if case != "world holds its names" and case != "a world revision after its cutoff":
        if case != "outside the Host":
            assert _selections() == []
        return
    assert _selections() == [(sector_map.map_hash,)]
    if case == "a world revision after its cutoff":
        _classify(database, revised)
        assert world.sector_map(tuple(NAMES), session=NOW.date(), cutoff_us=world_reads.LATEST) == (
            revised
        )
    assert classified(sector_map) == sector_map.entries  # its names at its cutoff
    assert dict(sector_history_as_of(artifacts, sector_map.sector_revision)) == NAMES
    assert _selections() == [(sector_map.map_hash,)]  # selected once
