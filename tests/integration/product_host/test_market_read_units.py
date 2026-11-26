"""The market store's read-only units ask for nothing a waiting writer holds (W0).

The Task registry's status read once retained its instance and then asked for the
registry's lock, which a starting Task held while it waited for that instance to
close (`31c49cf3`). Each unit here retains the live market store read-only across
many reads, and a gated writer that arrives meanwhile holds the mutation gate while
it waits for the instance. So each unit is stopped inside its retention, a writer
takes the gate and asks for the writable instance, and the unit then finishes: were
it to ask for the gate (or anything else the writer holds), both would wait until
the writer's wait expired.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.control.data_platform import preflight
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.data_preparation.remediation import (
    WorkspaceDataIssueApplication,
)
from alphalattice.control.product_host.maintenance.data_update import (
    installed_data_update_binding,
    read_workspace_inputs,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.workspace_task_runner.lock_order import LockOrder, lock_order

__all__ = ["lock_order"]  # the fixture, used below by name: no pair of locks in both orders


def _unit_lets_the_writer_in(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    gate: WorkspaceMutationGate,
    unit: Callable[[], Any],
    pause: tuple[object, str],
) -> Any:
    """Run `unit` to its first call of `pause` (inside its retention), start a gated
    writer, then let the unit finish; the writer gets the store once it has."""

    owner, name = pause
    original = getattr(owner, name)
    inside, go_on = Event(), Event()

    def paused(*args: Any, **kwargs: Any) -> Any:
        if not inside.is_set():
            inside.set()
            assert go_on.wait(30)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, paused)

    def write() -> bool:
        with gate.hold(), MarketDataRepository(workspace).database.retain(read_only=False):
            return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(unit)
        assert inside.wait(60)
        writer = pool.submit(write)
        with pytest.raises(FutureTimeout):
            writer.result(timeout=0.3)  # it holds the gate and waits for the unit's instance
        go_on.set()
        value = reading.result(timeout=60)
        assert writer.result(timeout=60)
    return value


@pytest.fixture
def workspace(qualified: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "workspace"
    shutil.copytree(qualified, copy)
    return copy


def test_the_workspace_inputs_read_holds_nothing_a_writer_waits_with(
    monkeypatch, workspace, lock_order: LockOrder
):
    binding = installed_data_update_binding()
    expected = read_workspace_inputs(workspace, binding)
    readiness = type(MarketDataRepository(workspace).readiness)
    status = _unit_lets_the_writer_in(
        monkeypatch,
        workspace,
        WorkspaceMutationGate(),
        lambda: read_workspace_inputs(workspace, binding),
        (readiness, "load"),
    )
    assert status == expected


def test_the_divergence_preflight_holds_nothing_a_writer_waits_with(
    monkeypatch, workspace, lock_order: LockOrder
):
    status = read_workspace_inputs(workspace, installed_data_update_binding())
    sessions = preflight.routine_preflight_formation_scope(
        as_of_session=status.panel_through, as_of_timestamp=datetime.now(UTC)
    )
    # The SPY side is read before the retention; only the Universe side is held.
    monkeypatch.setattr(
        preflight,
        "build_portfolio_benchmark_surface",
        lambda *, workspace, formation_sessions: SimpleNamespace(
            simple_returns=tuple(0.0 for _ in formation_sessions), surface_hash="5" * 64
        ),
    )
    diagnostic = _unit_lets_the_writer_in(
        monkeypatch,
        workspace,
        WorkspaceMutationGate(),
        lambda: preflight.evaluate_universe_spy_divergence_preflight(
            workspace=workspace, formation_sessions=sessions
        ),
        (FeatureStateRepository, "projected_feature_frame"),
    )
    assert diagnostic.session_count == len(sessions)


def test_the_data_issue_readiness_holds_nothing_a_writer_waits_with(
    monkeypatch, workspace, lock_order: LockOrder
):
    with WorkspaceApplicationSession.acquire(workspace) as session:
        issues = WorkspaceDataIssueApplication(session, lambda: datetime.now(UTC))
        expected = issues.current_decisions_ready()
        ready = _unit_lets_the_writer_in(
            monkeypatch,
            workspace,
            session.mutation_gate,
            issues.current_decisions_ready,
            (WorkspaceDataIssueApplication, "_cases"),
        )
    assert ready == expected
