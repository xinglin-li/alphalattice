"""An installed strategy activated on a workspace copy and served offline: the Host, the real CLI
on it and its Tasks' readings, shared by the strategy-activation journeys whatever copy they serve
(the QA store's in the public module, the strategy-book road's in the private one).
"""

from __future__ import annotations

import gc
import json
import socket
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.capabilities.alpha_modeling.runtime.service import AlphaModelRuntimeService
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.foundation.market_data_ops.sources.providers import MarketDataProvider
from alphalattice.interface.local_application.cli import main
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.kernel.shared_kernel.waits import wait_until
from devtools.evaluation.golden_workspaces import remove_tree
from tests.portfolio_strategy_lab.local_web_support import _json

PACKAGE = "RETURN_G6_MU_ONLY"
"""The installed book both copies hold (the Rebound Return Book)."""


@contextmanager
def forward_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    copy: Callable[[Path], Path],
    provider: MarketDataProvider | None = None,
    provider_from: Callable[[Path], MarketDataProvider] | None = None,
    clock: Callable[[], datetime],
) -> Iterator[
    tuple[
        LocalPortfolioWebSession,
        Callable[..., tuple[int, dict]],
        Callable[[str], dict[str, int]],
    ]
]:
    """A copy of the research installation, its strategy activated by a person, offline: the
        provider's yfinance module absent, the membership source refusing, no socket off the machine
        and no model fitted. Yields the Host, the real CLI on it and a Task's attempts by stage
    . `provider_from` builds the provider from the copy before the Host opens it."""

    # The provider's yfinance module is absent: every fetch refuses, `data.provider_unavailable`.
    monkeypatch.setitem(sys.modules, "yfinance", None)
    connect = socket.socket.connect

    def loopback(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else address
        assert host in {"127.0.0.1", "localhost", "::1"}, f"the test reached {address}"
        connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback)

    def no_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a forward update's resume fits no model")

    monkeypatch.setattr(AlphaModelRuntimeService, "fit", no_fit)
    backup_root = tmp_path / "backups"
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(backup_root))
    workspace = copy(tmp_path / "workspace")
    (workspace / "runtime/local-research-connection.json").unlink(missing_ok=True)
    if provider_from is not None:
        provider = provider_from(workspace)
    live = LocalPortfolioWebSession.from_workspace(workspace, clock=clock)
    live.data_provider = provider
    live.data_source_loader = no_membership_source
    live.start()
    try:
        registry = live.session.task_control_registry
        offered = live.operations.execute(
            PortfolioResearchOperationRequest(operation="CONTROLS", strategy_package_id=PACKAGE)
        )["activation"]
        activated = _json(
            live,
            "/api/strategy/activate",
            method="POST",
            payload={"task_id": offered["next_requests"]["activate"]["task_id"]},
            timeout=300.0,
        )
        assert activated["status"] == "ACTIVATED", activated

        def cli(*argv: str) -> tuple[int, dict]:
            code = main(
                ["--workspace", str(workspace), "--view", "full", *argv], serve=lambda _: 99
            )
            return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])

        def attempts(task: str) -> dict[str, int]:
            _record, items = registry.task_with_work_items(UUID(task))
            return {item.stage_id: item.attempt_count for item in items if item.attempt_count}

        yield live, cli, attempts
    finally:
        live.stop()
        gc.collect()
        assert not (left := remove_tree(workspace, tmp_path)), left
        if backup_root.exists():
            assert not (left := remove_tree(backup_root, tmp_path)), left


def no_membership_source(**_kwargs: object) -> object:
    raise ConnectionError("an offline test reads no membership source")


def with_lifecycle(answer: dict, registry: DuckDbTaskControlRegistry, task: str | None) -> str:
    """A wait's answer whole, as text, beside its Task's lifecycle as Task Control holds it at
    the same instant: the lifecycle the answer read, its wait event and incident, so
    a Task read as running after it ended is told from an incident's wake."""
    held = None if task is None else registry.task(UUID(task)).lifecycle.value
    return json.dumps({"registry_lifecycle": held, "answer": answer}, indent=1, default=str)


def lifecycle_settled(
    session: LocalPortfolioWebSession, task_id: str, timeout: float = 1800.0
) -> str:
    registry = session.session.task_control_registry

    def ended(_left: float) -> str | None:
        state = str(registry.task(UUID(task_id)).lifecycle.value)
        return state if state in {"SUCCEEDED", "BLOCKED", "CANCELLED"} else None

    return wait_until(ended, timeout=timeout, what=f"Task {task_id} settles")
