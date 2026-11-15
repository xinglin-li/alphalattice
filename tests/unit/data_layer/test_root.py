"""How a data root's path enters DuckDB SQL."""

from __future__ import annotations

from pathlib import PureWindowsPath

import pytest

from alphalattice.kernel.data_layer.root import sql_path

TEST_SOURCE = "src/alphalattice/kernel/data_layer/root.py"


@pytest.mark.proves("data_layer.paths_are_quoted_sql_literals")
def test_a_path_is_a_sql_literal_with_forward_slashes_and_quotes_doubled():
    """A Windows path reads with forward slashes, and a quote in a folder's name is doubled, so the
    literal it is placed in neither ends early nor changes meaning."""
    path = PureWindowsPath("D:\\research\\o'brien\\lake.ducklake")
    assert sql_path(path) == "D:/research/o''brien/lake.ducklake"
