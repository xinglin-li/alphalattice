"""Where a Task's stages spend their time: spans, their ledger and their readback (A4)."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from alphalattice.control.task_control.child import child_calls, run_in_child
from alphalattice.control.task_control.contracts import TaskEvidence, TaskLifecycle
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
    TaskControlRunner,
)
from alphalattice.control.task_control.timing import (
    read_stage_spans,
    stage_spans_path,
    task_timing,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.kernel.shared_kernel.spans import (
    collect,
    readout,
    span,
    spanned,
)
from tests.workspace_task_runner.task_control_support import compatibility, digest, task_contract

HERE = "tests.workspace_task_runner.test_stage_spans"


def _rows(value: dict[str, object]) -> dict[tuple[object, object, object], dict[str, object]]:
    return {
        (row["category"], row["detail"], row["origin"]): row
        for row in value["spans"]  # type: ignore[union-attr]
    }


def test_a_span_counts_its_own_time_apart_from_the_spans_it_holds() -> None:
    """requirement (A4): exclusive time is a span's own, less the spans it holds on its thread;
    a span never swallows the exception its block raises, and still counts its time."""

    @spanned("verify", "fixture")
    def checked() -> None:
        time.sleep(0.02)

    with collect() as ledger:
        with span("compute", "outer"):
            time.sleep(0.02)
            with span("read"):
                time.sleep(0.03)
            checked()
        with pytest.raises(ZeroDivisionError), span("compute", "outer"):
            _ = 1 / 0
    rows = _rows(readout(ledger))
    outer = rows["compute", "outer", "host"]
    assert outer["count"] == 2
    held = (
        rows["read", None, "host"]["inclusive_seconds"]
        + rows["verify", "fixture", "host"]["inclusive_seconds"]
    )
    assert held >= 0.05
    assert outer["exclusive_seconds"] == pytest.approx(outer["inclusive_seconds"] - held, abs=0.005)
    assert readout(ledger)["wall_seconds"] >= outer["inclusive_seconds"]
    with span("compute"):  # outside a ledger: measured by nothing, harmless
        pass


def test_a_thread_an_owner_starts_counts_toward_the_one_open_ledger() -> None:
    """requirement (A4): a thread an owner starts inherits no context, so its spans count
    toward the one open ledger; with two open, toward neither, and both say they overlapped."""

    def work() -> None:
        with span("hash", "thread"):
            time.sleep(0.01)

    with collect() as ledger:
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
    assert ("hash", "thread", "host") in _rows(readout(ledger))
    assert readout(ledger)["process"]["overlapped"] is False

    with collect() as first, collect() as second:
        thread = threading.Thread(target=work)
        thread.start()
        thread.join()
    assert not readout(first)["spans"] and not readout(second)["spans"]
    assert readout(first)["process"]["overlapped"] and readout(second)["process"]["overlapped"]


def spanned_work(*, seconds: float, cancelled) -> int:  # type: ignore[no-untyped-def]
    with span("fit", "fixture"):
        time.sleep(seconds)
    return 1


def test_a_worker_call_brings_its_spans_back_to_its_stage() -> None:
    """requirement (A4): a Task's call in a worker process keeps its spans, its CPU seconds and
    bytes, and they come back with the answer into the calling stage's ledger, beside the time
    the stage waited for it."""
    with collect() as ledger:
        assert run_in_child(f"{HERE}:spanned_work", {"seconds": 0.05}) == 1
        with child_calls(f"{HERE}:spanned_work", workers=2) as calls:
            for _ in range(3):
                calls.make({"seconds": 0.02})
            assert [calls.answer() for _ in range(3)] == [1, 1, 1]
    value = readout(ledger)
    rows = _rows(value)
    assert rows["fit", "fixture", "worker"]["count"] == 4
    assert rows["fit", "fixture", "worker"]["inclusive_seconds"] >= 0.11
    assert rows["wait", "worker", "host"]["count"] == 4
    assert value["workers"]["calls"] == 4
    assert value["workers"]["wall_seconds"] >= 0.11


class _SpannedAdapter:
    task_kind = "factor_research"

    def compatibility(self, task):  # type: ignore[no-untyped-def]
        return compatibility(task.plan)

    def execute_stage(self, *, task, execution, work_item):  # type: ignore[no-untyped-def]
        del task, execution
        with span("compute", work_item.stage_id):
            time.sleep(0.01)
        kind = {"resolve_inputs": "input_binding", "publish_result": "screening_report"}
        return StageExecutionResult(
            disposition=StageDisposition.READY,
            evidence=(
                TaskEvidence(
                    evidence_kind=kind[work_item.stage_id],
                    reference=f"playpen://runner/{work_item.stage_id}",
                    content_hash=digest(work_item.stage_id),
                ),
            ),
        )

    def verify_stage(self, *, task, execution, work_item, evidence):  # type: ignore[no-untyped-def]
        del task, execution, work_item
        with span("verify", "fixture"):
            return evidence


def test_each_stage_phase_keeps_its_spans_and_its_timing_reads_them_back(tmp_path: Path) -> None:
    """requirement (A4): the runner keeps each stage's execution and verification spans beside
    the Task, and the Task's timing lists them under their stage; a torn line is skipped and a
    Task without spans reads as before."""
    now = datetime(2026, 10, 7, 18, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="stage-spans")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    runtime = tmp_path / "runtime" / "task-runtime.sqlite"
    runtime.parent.mkdir(exist_ok=True)
    runner = TaskControlRunner(
        registry=registry,
        adapters={"factor_research": _SpannedAdapter()},
        runtime_path=str(runtime),
        clock=lambda: now + timedelta(seconds=1),
        heartbeat_seconds=0.01,
    )
    try:
        runner.run_next()
    finally:
        runner.close()

    path = stage_spans_path(runtime, task.task_id)
    assert path == tmp_path / "runtime" / "execution" / "spans" / f"{task.task_id}.jsonl"
    with path.open("a", encoding="utf-8") as sink:
        sink.write('{"torn": \n')
    kept = read_stage_spans(runtime, task.task_id)
    assert [(value["stage_id"], value["phase"]) for value in kept] == [
        ("resolve_inputs", "execute"),
        ("resolve_inputs", "verify"),
        ("publish_result", "execute"),
        ("publish_result", "verify"),
    ]
    record, items = registry.task_with_work_items(task.task_id)
    timing = task_timing(record, items, now=now + timedelta(seconds=2), spans=kept)
    stages = {stage["stage_id"]: stage for stage in timing["stages"]}  # type: ignore[union-attr]
    first = stages["resolve_inputs"]["phases"]
    assert [phase["phase"] for phase in first] == ["execute", "verify"]
    assert ("compute", "resolve_inputs", "host") in _rows(first[0])
    assert ("verify", "fixture", "host") in _rows(first[1])
    assert all(phase["execution_id"] for phase in first)
    plain = task_timing(record, items, now=now + timedelta(seconds=2))
    assert all(stage["phases"] == [] for stage in plain["stages"])  # type: ignore[union-attr]


def test_a_stage_gate_refusal_blocks_the_stage_before_its_work(tmp_path: Path) -> None:
    """requirement (PERF-1): the Host's memory check blocks a stage, recoverably, unexecuted."""
    now = datetime(2026, 10, 8, 11, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="stage-gate")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    runtime = tmp_path / "runtime" / "task-runtime.sqlite"
    runtime.parent.mkdir(exist_ok=True)
    executed: list[str] = []

    class Recording(_SpannedAdapter):
        def execute_stage(self, *, task, execution, work_item):  # type: ignore[no-untyped-def]
            executed.append(work_item.stage_id)
            return super().execute_stage(task=task, execution=execution, work_item=work_item)

    runner = TaskControlRunner(
        registry=registry,
        adapters={"factor_research": Recording()},
        runtime_path=str(runtime),
        clock=lambda: now + timedelta(seconds=1),
        heartbeat_seconds=0.01,
        stage_gate=lambda _task: "execution.memory_insufficient",
    )
    try:
        runner.run_next()
    finally:
        runner.close()
    record = registry.task(task.task_id)
    assert record.lifecycle is TaskLifecycle.BLOCKED
    assert record.failure_code == "execution.memory_insufficient"
    assert executed == []


def test_each_stage_on_the_runners_thread_measures_storage_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: the runner opens the Host's storage scope around each stage
    it runs alone, so a stage's admissions share one walk and the next stage walks again."""
    from alphalattice.control.product_host.storage import inventory

    now = datetime(2026, 10, 8, 14, tzinfo=UTC)
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract(salt="storage-scope")
    task = registry.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    workspace = tmp_path / "workspace"
    (workspace / "artifacts").mkdir(parents=True)
    (workspace / "artifacts/panel.bin").write_bytes(b"p" * 100)
    walks: list[Path] = []
    walk = inventory.managed_file_inventory
    monkeypatch.setattr(
        inventory, "managed_file_inventory", lambda root: walks.append(root) or walk(root)
    )

    class Admitting(_SpannedAdapter):
        def execute_stage(self, *, task, execution, work_item):  # type: ignore[no-untyped-def]
            for _ in range(3):
                inventory.require_storage_capacity(workspace, additional_bytes=10)
            return super().execute_stage(task=task, execution=execution, work_item=work_item)

    runtime = tmp_path / "runtime" / "task-runtime.sqlite"
    runtime.parent.mkdir(exist_ok=True)
    runner = TaskControlRunner(
        registry=registry,
        adapters={"factor_research": Admitting()},
        runtime_path=str(runtime),
        clock=lambda: now + timedelta(seconds=1),
        heartbeat_seconds=0.01,
        stage_scope=inventory.storage_capacity_scope,
    )
    try:
        runner.run_next()
    finally:
        runner.close()
    assert registry.task(task.task_id).lifecycle is TaskLifecycle.SUCCEEDED
    assert len(walks) == 2  # one walk for each of the two stages, not one per admission
