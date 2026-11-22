"""Where the Host's research history lives, read from the environment alone."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alphalattice.kernel.data_layer.history import DATA_ROOT_ENV, data_root, store

TEST_SOURCE = "src/alphalattice/kernel/data_layer/history.py"


@pytest.mark.proves("data_layer.data_root_is_set_or_the_application_folder")
def test_the_data_root_is_its_setting_else_the_platforms_application_folder(monkeypatch, tmp_path):
    """A set ALPHALATTICE_DATA_ROOT wins; unset, the root is the platform's application data
    folder (Windows: LOCALAPPDATA/AlphaLattice; elsewhere XDG_DATA_HOME/alphalattice)."""
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "set"))
    assert data_root() == tmp_path / "set"
    monkeypatch.delenv(DATA_ROOT_ENV)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    expected = tmp_path / ("local/AlphaLattice" if os.name == "nt" else "xdg/alphalattice")
    assert data_root() == Path(expected)


@pytest.mark.proves("data_layer.only_a_host_opens_its_store")
def test_a_process_with_no_host_history_opens_no_store():
    """A process whose Host history is not open gets no StorePort, so its workspace opens admit
    nothing."""
    assert store() is None
