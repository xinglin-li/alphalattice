"""The workspace network control reads closed on anything its writer does not write (V271)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.network_access import (
    CONTROL_PATH,
    network_access,
    set_network_access,
)


def test_the_written_control_opens_and_closes_the_network(tmp_path: Path) -> None:
    """requirement (C1 rule 7): the written control decides, the operator's switch first."""

    assert network_access(tmp_path, environment={}).decided_by == "DEFAULT"
    set_network_access(tmp_path, enabled=True)
    assert network_access(tmp_path, environment={}).allowed
    assert not network_access(tmp_path, environment={"ALPHALATTICE_NETWORK_DISABLED": "1"}).allowed
    set_network_access(tmp_path, enabled=False)
    assert not network_access(tmp_path, environment={}).allowed


@pytest.mark.parametrize(
    "document",
    [
        "[]",
        '"open"',
        "7",
        "null",
        json.dumps({"schema": "network-access", "version": 999, "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": True, "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": "1", "network_enabled": True}),
        json.dumps({"schema": "network-access", "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": 1, "network_enabled": 1}),
        json.dumps({"schema": "other", "version": 1, "network_enabled": True}),
        json.dumps({"schema": "network-access", "version": 2, "network_enabled": True}),
        json.dumps(
            {
                "schema": "network-access",
                "version": 2,
                "network_enabled": True,
                "delegation": "first-use-goal:x",
                "until": "2999-01-01T00:00:00",
            }
        ),
        json.dumps(
            {"schema": "network-access", "version": 1, "network_enabled": True, "hosts": ["*"]}
        ),
        "{not json",
    ],
)
def test_anything_but_a_version_one_mapping_with_a_boolean_reads_closed(
    tmp_path: Path, document: str
) -> None:
    """regression (V271, OP5): a JSON array raised AttributeError and version 999 with
    ``network_enabled: true`` opened the network; each reads as the control, closed."""

    path = tmp_path / CONTROL_PATH
    path.parent.mkdir(parents=True)
    path.write_text(document, encoding="utf-8")
    access = network_access(tmp_path, environment={})
    assert access.decided_by == "WORKSPACE_CONTROL" and not access.allowed


def test_a_delegated_setting_holds_until_its_end_with_no_write(tmp_path: Path) -> None:
    """regression (V452, OP5, OP19; an outside review at 118f6378): a first-use goal's network
    stayed open after its hours, since only a later write closed it. The delegation's setting
    carries its end, and reads closed after it at every read, idle or not; the page reads who
    set it."""

    end = datetime(2026, 10, 3, tzinfo=UTC)
    set_network_access(tmp_path, enabled=True, delegation="first-use-goal:g", until=end)
    before = network_access(tmp_path, environment={}, now=end - timedelta(seconds=1))
    after = network_access(tmp_path, environment={}, now=end + timedelta(hours=1))
    assert before.allowed and not after.allowed
    assert before.body()["set_by"] == {"delegation": "first-use-goal:g", "until": end.isoformat()}
    # The operator's switch decides, and who set the control is still named (U70).
    held = network_access(tmp_path, environment={"ALPHALATTICE_NETWORK_DISABLED": "1"}, now=end)
    assert held.decided_by == "OPERATOR_OFFLINE_SWITCH" and held.body()["set_by"]
    set_network_access(tmp_path, enabled=True)
    assert network_access(tmp_path, environment={}, now=end + timedelta(days=9)).allowed
    assert "set_by" not in network_access(tmp_path, environment={}).body()
