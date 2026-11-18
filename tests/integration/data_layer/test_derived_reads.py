import uuid

import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime import derived_views
from alphalattice.control.workspace_runtime.database import WorkspaceDatabase
from alphalattice.kernel.data_layer import derived_reads  # noqa: F401  (registers the setup)
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.root import create_data_root
from alphalattice.kernel.data_layer.store import StorePort, feature_table

TEST_SOURCE = "src/alphalattice/kernel/data_layer/derived_reads.py"
LISTING = str(uuid.UUID(int=7))
TABLE = feature_table(definition_id("feature_set", "desktop"))


@pytest.mark.proves("data_layer.a_workspace_reads_derived_rows_under_its_names")
def test_a_workspace_reader_takes_the_catalogs_rows_and_its_writer_the_tables(
    tmp_path, monkeypatch
):
    """Under derived, a workspace's read-only connection reads a feature set's catalog rows under
    the workspace table's name, a persisted view's included, and a writable one its own table."""
    root, folder = tmp_path / "data", tmp_path / "workspace"
    create_data_root(root)
    rows = pa.table(
        {
            "recipe_key": ["r1"],
            "input_digest": ["d1"],
            "listing_id": [LISTING],
            "session_date": ["2024-01-02"],
            "momentum": [0.25],
        }
    )
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        named = pa.table({"part": [f"2024-01/{LISTING}"], "input_digest": ["d1"]})
        port.derive(TABLE, rows, recipe_key="r1", parts=named, compute_ms=1)
        port.close()
    database = WorkspaceDatabase(folder)
    with database.transaction() as connection:
        connection.execute(
            "CREATE TABLE feature_daily_current (listing_id VARCHAR, session_date DATE,"
            " catalog_hash VARCHAR, momentum DOUBLE)"
        )
        connection.execute(
            f"INSERT INTO feature_daily_current VALUES ('{LISTING}', '2024-01-02', 'r1', 0.5)"
        )
        connection.execute(
            "CREATE VIEW feature_daily_runtime AS SELECT * FROM feature_daily_current"
        )
        body = "SELECT CAST(listing_id AS VARCHAR) AS listing_id, session_date,"
        body += f" recipe_key AS catalog_hash, momentum FROM lake.derived.{TABLE}"
        derived_views.declare(
            connection, "feature_daily_current", setting="ALPHALATTICE_FEATURE_READS", body=body
        )
    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(root))
    monkeypatch.setenv("ALPHALATTICE_FEATURE_READS", "derived")
    read = "SELECT listing_id, catalog_hash, momentum FROM feature_daily_runtime"
    with database.read_transaction() as connection:
        derived = connection.execute(read).fetchall()
    with database.transaction() as connection:
        stored = connection.execute(read).fetchall()
    assert derived == [(LISTING, "r1", 0.25)] and stored == [(LISTING, "r1", 0.5)]
