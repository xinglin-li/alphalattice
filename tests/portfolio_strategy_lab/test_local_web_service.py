"""Local Web service writes take admission and operator settings."""

from __future__ import annotations

import json

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from tests.portfolio_strategy_lab.cli_support import _cli
from tests.portfolio_strategy_lab.local_web_support import _json, _request


@pytest.mark.parametrize(
    "path",
    [
        "/api/experiments/training-inputs/prepare",
        "/api/workspace/storage/plan",
        "/api/workspace/storage/cap",
        "/api/research-inputs/plan",
        "/api/experiments/training-inputs/plan",
        "/api/research-strategies/plan",
        "/api/workspace/data-issues/preview",
        "/api/workspace/preparation/plan",
        "/api/experiments/plan",
        "/api/experiments/handoff",
        "/api/experiments/foundations/preview",
    ],
)
def test_a_route_that_writes_takes_the_write_check(
    live: LocalPortfolioWebSession, path: str
) -> None:
    """A route that writes takes the write check."""

    status, _headers, body = _request(live, path, method="POST", payload={}, token=None)
    assert status == 403, (status, body[:300])
    assert "session_token_absent" in json.loads(body)["refused"]

    if path == "/api/workspace/storage/cap":
        cap = str(20 * 1024**3)
        changed = _json(
            live, "/api/workspace/storage/cap", method="POST", payload={"storage_cap_bytes": cap}
        )
        assert changed["status"] == "CONFIGURED"
        assert changed["capacity"]["cap_bytes"] == int(cap)
        status, _headers, body = _request(
            live, "/api/workspace/storage/cap", method="POST", payload={"storage_cap_bytes": "0"}
        )
        assert status == 200 and json.loads(body)["failure_code"] == "storage.cap_setting_invalid"
        assert _json(live, "/api/workspace/storage/cap")["capacity"]["cap_bytes"] == int(cap)
        human = _json(
            live,
            "/api/workspace/storage/cap",
            method="POST",
            payload={"storage_cap_bytes": str(25 * 1024**3)},
        )
        assert human["capacity"]["setting"]["chosen_by"] == "HUMAN"
        code, shown, _ = _cli(live.workspace, "storage", "cap")
        assert code == 0 and shown["data"]["capacity"]["cap_bytes"] == 25 * 1024**3
