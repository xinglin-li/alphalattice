"""The runtime hazards RH's inventory ranked, each checked the way it would bite (RH2).

The private RH inventory names five two-path suspects. Each is checked here with a second party
arriving while the first holds what it holds. A check passes where the owner prevents the
hazard. Where it does not yet, the check is marked with the phase-4 fix that owns it, and it
fails once that fix lands, so the mark is removed with the fix. The recorder (`lock_order.py`)
names any pair of the product's locks a check took in both orders.

The market store's read units, suspect 2, are W0's checks
(`tests/workspace_maintenance/test_market_read_units.py`).
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Timer
from time import monotonic
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.database import retain_workspace_database
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from tests.alternative_evidence_desk.document_intelligence_support import (
    _built,
    _counted_runtime,
)
from tests.alternative_evidence_desk.planted_corpus import _NOW
from tests.workspace_task_runner.lock_order import LockOrder, lock_order, record_lock_order
from tests.workspace_task_runner.task_control_support import compatibility, task_contract

__all__ = ["lock_order"]  # the fixture, used below by name


def test_the_recorder_names_a_pair_taken_in_both_orders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with record_lock_order(monkeypatch) as recorder:
        gate = WorkspaceMutationGate()
        registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
        with gate.hold(), registry._connection_lock:
            pass
        assert recorder.inversions() == []
        with registry._connection_lock, gate.hold():
            pass
    assert recorder.inversions() == [("gate", "registry")]


def test_the_recorder_names_the_order_that_stalled_the_workbench(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Before `31c49cf3` a status read retained the registry read-only and then asked for its
    connection lock, while a writer takes the lock and then the file read-write. The recorder
    names that pair even when the two never meet on two threads."""

    with record_lock_order(monkeypatch) as recorder:
        registry = DuckDbTaskControlRegistry(
            resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
        )
        first = task_contract()
        now = datetime.now(UTC)
        registry.admit(input_envelope=first[0], goal=first[1], plan=first[2], observed_at=now)
        assert recorder.inversions() == []
        with (
            retain_workspace_database(registry.database_path, read_only=True),
            registry._connection_lock,
        ):
            pass
    assert recorder.inversions() == [("instance:research-task-control.duckdb", "registry")]


def test_a_status_read_beside_a_starting_task_takes_no_pair_in_both_orders(
    tmp_path: Path, lock_order: LockOrder
) -> None:
    """Suspect 1: the Task registry's read-only unit against a starting Task (`31c49cf3`).
    The unit takes the connection lock before its instance, so the writer waits for it and
    both finish; the recorder sees one order."""

    gate = WorkspaceMutationGate()
    registry = DuckDbTaskControlRegistry(resolve_task_control_database(tmp_path), gate=gate)
    first = task_contract()
    now = datetime.now(UTC)
    registry.admit(input_envelope=first[0], goal=first[1], plan=first[2], observed_at=now)
    inside, go_on = Event(), Event()

    def status_read() -> int:
        with registry.retain(read_only=True):
            inside.set()
            assert go_on.wait(10)
            return len(registry.tasks())

    with ThreadPoolExecutor(max_workers=2) as pool:
        reader = pool.submit(status_read)
        assert inside.wait(5)
        writer = pool.submit(
            registry.start_next,
            compatibility=compatibility(first[2]),
            worker_instance_id=uuid4(),
            observed_at=now,
        )
        with pytest.raises(FutureTimeout):
            writer.result(timeout=0.3)
        go_on.set()
        assert reader.result(timeout=10) == 1
        running, _ = writer.result(timeout=10)
    assert running.lifecycle is TaskLifecycle.RUNNING


def test_a_session_read_defers_while_a_writer_holds_the_gate(
    tmp_path: Path, lock_order: LockOrder
) -> None:
    """Suspect 3: the quality step holds the gate over its whole list (about 38 s for 473 names,
    V109). A session's read of the context asks for the gate with a timeout and defers, so a
    page is not held for the hold; the hold itself is RH's phase-4 fix."""

    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        held, release = Event(), Event()

        def write() -> None:
            with session.mutation_gate.hold():
                held.set()
                assert release.wait(10)

        with ThreadPoolExecutor(max_workers=1) as pool:
            writer = pool.submit(write)
            assert held.wait(5)
            started = monotonic()
            with session.reads(timeout_seconds=0.2) as acquired:
                assert acquired is False
            assert monotonic() - started < 5
            release.set()
            writer.result(timeout=10)


def test_the_workbench_build_replaces_an_asset_a_reader_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Suspect 4 (RH, V170): the build replaces the assets a starting Host reads; Windows
    refuses to replace a file a reader holds open, so the build retries the replace while the
    reader finishes."""

    from scripts import build_local_web_ui

    for name in build_local_web_ui.FONT_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build_local_web_ui.ASSETS / name, target)
    monkeypatch.setattr(build_local_web_ui, "ASSETS", tmp_path)
    (tmp_path / "workbench.html").write_bytes(b"<!doctype html>")
    reading = (tmp_path / "workbench.html").open("rb")  # a Host reading its page at start
    finished = Timer(0.3, reading.close)
    finished.start()
    try:
        build_local_web_ui.build()
    finally:
        finished.join()
        reading.close()
    page = (tmp_path / "workbench.html").read_bytes()
    assert page.startswith(b"<!DOCTYPE html>") and b'<script src="/workbench.' in page


def test_an_index_a_session_reads_is_not_evicted_from_under_it(tmp_path: Path) -> None:
    """Suspect 5: a storage clean-up evicts a hybrid index while a retriever holds it open.
    The retrieval service refuses while any session on the generation is open, under its
    lease lock, and evicts once the session ends."""

    runtime, _passes = _counted_runtime(tmp_path)
    request, document_set, generation = _built(runtime)
    (database,) = sorted((tmp_path / "workspace" / ".system" / "knowledge-indexes").rglob("*.db"))
    relative = database.resolve().relative_to((tmp_path / "workspace").resolve()).as_posix()
    session = runtime.retrieval.open_session(
        document_set=document_set, generation=generation, evidence_as_of=request.evidence_as_of
    )
    try:
        with pytest.raises(ValueError, match="retrieval_index_in_use"):
            runtime.retrieval.evict_generation(relative, plan_hash="a" * 64, evicted_at=_NOW)
        assert database.is_file()
    finally:
        session.close()
    runtime.retrieval.evict_generation(relative, plan_hash="a" * 64, evicted_at=_NOW)
    assert not database.is_file()
    runtime.close()
