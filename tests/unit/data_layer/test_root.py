"""How a data root's path enters DuckDB SQL."""

from __future__ import annotations

from pathlib import PureWindowsPath

import pytest

from alphalattice.kernel.data_layer.log import envelope
from alphalattice.kernel.data_layer.root import sql_path

TEST_SOURCE = "src/alphalattice/kernel/data_layer/root.py"


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
def test_an_event_is_of_its_types_first_version_unless_it_names_another():
    """A new event's envelope carries version 1 of its type, or the version its writer names."""
    assert envelope("task.saved")["type_version"] == 1
    assert envelope("task.requested", version=2)["type_version"] == 2
