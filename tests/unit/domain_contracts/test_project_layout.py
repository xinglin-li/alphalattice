"""uv's tool folder, read from the caller's settings as uv reads it, without starting uv."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphalattice.capabilities.domain_contracts.project_layout import tool_bin

TEST_SOURCE = "src/alphalattice/capabilities/domain_contracts/project_layout.py"


@pytest.mark.proves("domain_contracts.uv_tool_folder_read_as_uv_reads_it")
def test_the_tool_folder_follows_uvs_order_and_a_relative_override_is_the_callers(tmp_path):
    """A nonempty UV_TOOL_BIN_DIR counts relative to the caller's directory; XDG only absolute."""
    home = {"HOME": str(tmp_path / "home"), "USERPROFILE": str(tmp_path / "home")}
    assert tool_bin({"UV_TOOL_BIN_DIR": "tools", **home}, tmp_path) == tmp_path / "tools"
    assert (
        tool_bin({"UV_TOOL_BIN_DIR": str(tmp_path / "set")}, Path("elsewhere")) == tmp_path / "set"
    )
    assert (
        tool_bin({"XDG_BIN_HOME": "relative", **home}, tmp_path)
        == tmp_path / "home" / ".local" / "bin"
    )
    assert tool_bin({"XDG_BIN_HOME": str(tmp_path / "xdg"), **home}, tmp_path) == tmp_path / "xdg"
    assert (
        tool_bin({"XDG_DATA_HOME": str(tmp_path / "share"), **home}, tmp_path) == tmp_path / "bin"
    )
