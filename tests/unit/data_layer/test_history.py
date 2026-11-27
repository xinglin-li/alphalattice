"""Where the Host's research history lives, read from the environment alone."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alphalattice.foundation.data_layer.history import DATA_ROOT_ENV, data_root, store
from alphalattice.foundation.data_layer.history_runtime import otlp_refusal

TEST_SOURCE = "src/alphalattice/foundation/data_layer/history.py"


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


@pytest.mark.proves("data_layer.offline_otlp_only_to_loopback")
@pytest.mark.parametrize(
    ("endpoint", "refusal"),
    [
        pytest.param("http://127.0.0.1:4318/v1/traces", None, id="loopback-ipv4"),
        pytest.param("http://127.8.9.10:4318", None, id="loopback-ipv4-block"),
        pytest.param("http://[::1]:4318/v1/metrics", None, id="loopback-ipv6"),
        pytest.param("http://localhost:4318", "localhost", id="a-name-is-never-resolved"),
        pytest.param("http://192.168.1.20:4318", "192.168.1.20", id="a-private-address"),
        pytest.param("https://8.8.8.8/v1/traces", "8.8.8.8", id="a-public-address"),
    ],
)
def test_offline_admits_an_otlp_receiver_only_at_a_literal_loopback_address(endpoint, refusal):
    """Offline, an OTLP endpoint is admitted only at a literal loopback address, and any other host
    is refused by name; online, each of them is admitted."""
    expected = None if refusal is None else f"otlp.offline_host_refused: {refusal}"
    assert otlp_refusal(endpoint, offline=True) == expected
    assert otlp_refusal(endpoint, offline=False) is None
