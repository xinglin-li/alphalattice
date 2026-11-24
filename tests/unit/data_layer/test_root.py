"""How a data root's path enters DuckDB SQL."""

from __future__ import annotations

from pathlib import PureWindowsPath

import pytest

from alphalattice.foundation.data_layer.log import envelope
from alphalattice.foundation.data_layer.root import bound_reads, sql_path

TEST_SOURCE = "src/alphalattice/foundation/data_layer/root.py"


@pytest.mark.proves("data_layer.paths_are_quoted_sql_literals")
@pytest.mark.parametrize(
    ("spelled", "literal"),
    [
        (r"D:\research\o'brien\lake.ducklake", "D:/research/o''brien/lake.ducklake"),
        (r"\\?\D:\research\lake.duckdb", "D:/research/lake.duckdb"),
        (r"\\?\UNC\server\share\lake.duckdb", "//server/share/lake.duckdb"),
    ],
)
def test_a_path_is_a_sql_literal_with_forward_slashes_and_quotes_doubled(spelled, literal):
    """A Windows path, an extended-length one in its ordinary spelling, reads with forward slashes
    and each quote doubled, so the literal attaches and neither ends early nor changes meaning."""
    assert sql_path(PureWindowsPath(spelled)) == literal


@pytest.mark.proves("data_layer.an_event_names_its_type_version")
def test_an_event_is_of_its_types_current_version_unless_it_names_another():
    """A new event's envelope names no version, the log writing its type's current one, or the
    version its writer names."""
    assert envelope("task.saved")["type_version"] is None
    assert envelope("task.requested", version=2)["type_version"] == 2


@pytest.mark.proves("data_layer.a_lake_session_bounds_its_read_ahead")
def test_a_lake_session_reads_ahead_no_scan_job():
    """A session over the lake turns DuckDB's read-ahead off: its default queue holds a Parquet
    file open per queued job, so a scan's descriptors grow with the lake's files."""
    run = []

    class Session:
        def execute(self, sql):
            run.append(sql)
            return self

        def complete(self):
            return None

    bound_reads(Session())
    assert run == ["SET read_ahead_depth = 0"]
