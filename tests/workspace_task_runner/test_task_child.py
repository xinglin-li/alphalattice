"""A Task's numerical work in the Host's worker process (W10)."""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alphalattice.control.task_control.child import (
    ONE_THREAD_VARIABLES,
    ChildInterrupted,
    ChildRefused,
    ChildStartFailed,
    child_calls,
    run_in_child,
)

HERE = "tests.workspace_task_runner.test_task_child"


@pytest.mark.parametrize("spread", [False, True])
@pytest.mark.parametrize("at", ["constructor", "start"])
@pytest.mark.parametrize("error_type", [OSError, MemoryError])
def test_child_creation_refuses_by_name_and_the_next_call_replaces_it(
    monkeypatch, spread, at, error_type
):
    """V610: both child APIs normalize constructor/start resource refusals and can recover."""
    # Retire even a warm worker through the public boundary before injecting at its next start.
    with pytest.raises(ChildInterrupted):
        run_in_child(f"{HERE}:ended", {})
    error = error_type("fixture child start refused")

    def fail(*args, **kwargs):
        raise error

    with monkeypatch.context() as patch:
        if at == "constructor":
            patch.setattr(ProcessPoolExecutor, "__init__", fail)
        else:
            patch.setattr(multiprocessing.get_context("spawn").Process, "start", fail)
        with pytest.raises(
            ChildStartFailed, match=r"^task_control\.child_start_failed$"
        ) as stopped:
            if spread:
                with child_calls(f"{HERE}:doubled", workers=1) as calls:
                    calls.make({"value": 21})
            else:
                run_in_child(f"{HERE}:doubled", {"value": 21})
        assert stopped.value.__cause__ is error
        assert stopped.value.type_name == error_type.__name__
    assert run_in_child(f"{HERE}:doubled", {"value": 21})["value"] == 42


class _Declared(RuntimeError):
    failure_class = "OPERATIONAL_FAILURE"


def doubled(*, value: int, cancelled) -> dict[str, int]:  # type: ignore[no-untyped-def]
    return {"value": value * 2, "pid": os.getpid()}


def thread_variables(*, cancelled) -> dict[str, str | None]:  # type: ignore[no-untyped-def]
    return {name: os.environ.get(name) for name in ONE_THREAD_VARIABLES}


def numerical_product(*, cancelled) -> dict[str, object]:  # type: ignore[no-untyped-def]
    import numpy as np

    from alphalattice.kernel.shared_kernel.environment import numerical_thread_counts

    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    before = numerical_thread_counts()
    product = values @ values.T
    return {"counts": before, "after": numerical_thread_counts(), "product": product.tolist()}


def refused(*, cancelled) -> None:  # type: ignore[no-untyped-def]
    raise ValueError("fixture.refused_by_name")


def interrupted(*, cancelled) -> None:  # type: ignore[no-untyped-def]
    raise _Declared("fixture.interrupted")


def until_cancelled(*, seconds: float, cancelled) -> str:  # type: ignore[no-untyped-def]
    ends = time.monotonic() + seconds
    while time.monotonic() < ends:
        if cancelled():
            return "cancelled"
        time.sleep(0.05)
    return "ran out"


def ended(*, cancelled) -> None:  # type: ignore[no-untyped-def]
    os._exit(7)


def sized(*, value: int, size: int, cancelled) -> dict[str, object]:  # type: ignore[no-untyped-def]
    if value < 0:
        raise ValueError("fixture.refused_by_name")
    return {"value": value, "pid": os.getpid(), "payload": b"x" * size}


def test_calls_spread_over_kept_workers_answer_in_order_and_bring_back_large_answers() -> None:
    """requirement (PF #1, W10): independent calls run on up to N kept workers and answer in
    the order made; an answer of megabytes comes back whole, through the calls' folder, where a
    worker's pipe refused one under load (WinError 1450); a refusal answers in its turn."""

    with child_calls(f"{HERE}:sized", workers=3) as calls:
        for value in range(6):
            calls.make({"value": value, "size": 12_000_000 if value == 4 else 8})
        calls.make({"value": -1, "size": 8})
        answers = [calls.answer() for _ in range(6)]
        with pytest.raises(ChildRefused, match=r"^fixture.refused_by_name$"):
            calls.answer()
    assert [answer["value"] for answer in answers] == list(range(6))
    assert len({answer["pid"] for answer in answers}) == 3
    assert answers[4]["payload"] == b"x" * 12_000_000


def test_a_call_runs_in_one_kept_worker_and_answers_its_failures_by_kind() -> None:
    """requirement (W10, LAWS OW2): the fit's Python work leaves the Host's process; the worker
    answers a result, a refusal as a refusal and anything else as an interruption with its
    declared class, and a second call finds the same warm worker."""

    first = run_in_child(f"{HERE}:doubled", {"value": 21})
    second = run_in_child(f"{HERE}:doubled", {"value": 1})
    assert (first["value"], second["value"]) == (42, 2)
    assert first["pid"] == second["pid"] != os.getpid()
    with pytest.raises(ChildRefused, match=r"^fixture.refused_by_name$") as refusal:
        run_in_child(f"{HERE}:refused", {})
    assert refusal.value.type_name == "ValueError"
    with pytest.raises(ChildInterrupted, match=r"^fixture.interrupted$") as failure:
        run_in_child(f"{HERE}:interrupted", {})
    assert failure.value.type_name == "_Declared"
    assert failure.value.failure_class == "OPERATIONAL_FAILURE"


def test_a_cancellation_reaches_the_worker_and_a_worker_that_ends_is_replaced() -> None:
    """requirement (W10): the parent relays the Task's cancellation while the call runs; a
    worker that ends without answering is an interruption, and the next call starts a new one."""

    asked = []

    def cancelled() -> bool:
        asked.append(True)
        return len(asked) >= 2

    answer = run_in_child(f"{HERE}:until_cancelled", {"seconds": 30.0}, cancelled=cancelled)
    assert answer == "cancelled"
    with pytest.raises(ChildInterrupted, match="child_process_ended"):
        run_in_child(f"{HERE}:ended", {})
    assert run_in_child(f"{HERE}:doubled", {"value": 5})["value"] == 10


def test_a_cached_read_sees_a_write_another_registry_made(tmp_path: Path) -> None:
    """regression (W10): Task reads are answered again while the database files show no write;
    a write through another registry on the same file is seen by the next read."""

    from alphalattice.control.task_control.registry import (
        TASK_CONTROL_DATABASE_FILENAME,
        DuckDbTaskControlRegistry,
    )
    from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
    from tests.workspace_task_runner.task_control_support import task_contract

    now = datetime(2026, 9, 29, tzinfo=UTC)
    database = tmp_path / "runtime" / TASK_CONTROL_DATABASE_FILENAME
    first = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    envelope, goal, plan = task_contract()
    admitted = first.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    assert first.tasks() == (admitted,)
    assert first.task(admitted.task_id) is first.task(admitted.task_id)  # answered again
    second = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    second.request_cancel(
        task_id=admitted.task_id, expected_task_hash=admitted.record_hash, observed_at=now
    )
    assert first.task(admitted.task_id).lifecycle != admitted.lifecycle
    record, items = first.task_with_work_items(admitted.task_id)
    assert record == first.task(admitted.task_id) and len(items) == len(plan.work_items)


def test_a_fit_log_records_the_numerical_thread_counts(tmp_path: Path) -> None:
    """requirement (V68, LAWS PA3): the BLAS thread count can move an Alpha metric's last bits,
    so the fit log keeps the numerical libraries' counts beside the run; a record written before
    it was kept still reads, with none."""

    import numpy  # noqa: F401 -- the Host's numerical libraries are loaded when it fits

    from alphalattice.evidence.alternative_evidence.runtime.execution import ModelFitExecution
    from alphalattice.kernel.shared_kernel.environment import numerical_thread_counts

    counts = numerical_thread_counts()
    assert counts and all(isinstance(api, str) and threads >= 1 for api, threads in counts)
    record = {
        "program_hash": "a" * 64,
        "cpu_budget": "auto",
        "cores": 4,
        "lightgbm_threads": 4,
        "reason": "fixture",
        "machine": {
            "processors": 8,
            "busy_processors": 1.0,
            "total_memory_bytes": 1,
            "available_memory_bytes": 1,
            "preparations_running": 0,
        },
        "planned_at": "2026-09-29T00:00:00+00:00",
    }
    assert ModelFitExecution.model_validate(record).numerical_threads == ()
    kept = ModelFitExecution.model_validate({**record, "numerical_threads": counts})
    assert kept.numerical_threads == counts


def test_a_spread_worker_loads_its_numerical_libraries_on_one_thread() -> None:
    """regression (V436, the fork's measurement): every kept worker reserved its numerical
    libraries' buffers for one thread per processor at import, about 1.5 GB of commit each, so
    a few Hosts' first builds exhausted the machine. A spread worker, whose calls run on one
    thread, is told so before its libraries load; the Task's own worker keeps the default its
    calls run under, on which a Risk number can depend."""

    parent = {name: os.environ.get(name) for name in ONE_THREAD_VARIABLES}
    assert run_in_child(f"{HERE}:thread_variables", {}) == parent
    with child_calls(f"{HERE}:thread_variables", workers=2) as calls:
        for _ in range(2):
            calls.make({})
        answers = [calls.answer() for _ in range(2)]
    spread = [a for a in answers if a != parent]
    assert spread and all(set(a.values()) == {"1"} for a in spread)


def test_spread_numerical_calls_hold_one_thread_and_leave_normal_calls_unchanged() -> None:
    """PATTERN: the shared limiter reaches real BLAS work, with the same exact result.

    Spread calls hold one thread. The ordinary Task worker has no thread scope,
    so the same call retains its ambient numerical counts before and afterward.
    """
    import numpy as np

    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    expected = (values @ values.T).tolist()
    ordinary = run_in_child(f"{HERE}:numerical_product", {})
    with child_calls(f"{HERE}:numerical_product", workers=2) as calls:
        for _ in range(2):
            calls.make({})
        spread = [calls.answer() for _ in range(2)]
    for answer in spread:
        assert answer["product"] == expected
        assert answer["counts"] and all(threads == 1 for _api, threads in answer["counts"])
        assert answer["after"] == answer["counts"]
    again = run_in_child(f"{HERE}:numerical_product", {})
    assert ordinary["counts"] and ordinary["counts"] == ordinary["after"]
    assert again["counts"] == again["after"] == ordinary["counts"]
    assert ordinary["product"] == again["product"] == expected


def observe_cancellation(*, started: str, stopped: str, cancelled):
    Path(started).write_text("running", encoding="utf-8")
    deadline = time.monotonic() + 10
    while not cancelled() and time.monotonic() < deadline:
        time.sleep(0.02)
    Path(stopped).write_text("cancelled" if cancelled() else "expired", encoding="utf-8")


def test_a_later_start_refusal_cancels_an_already_running_child(monkeypatch, tmp_path):
    """V610: a second worker start refusal drains a real first call through its cancel flag."""
    # Retire all workers prior child tests may have kept, through their public call boundary.
    with child_calls(f"{HERE}:ended", workers=3) as calls:
        for _ in range(3):
            calls.make({})
        for _ in range(3):
            with pytest.raises(ChildInterrupted):
                calls.answer()
    start = multiprocessing.get_context("spawn").Process.start
    processes = []

    def fail_second(process):
        if processes:
            raise OSError(1455, "fixture second worker start refused")
        start(process)
        processes.append(process)

    running = tmp_path / "running"
    stopped = tmp_path / "stopped"
    with monkeypatch.context() as patch:
        patch.setattr(multiprocessing.get_context("spawn").Process, "start", fail_second)
        with (
            pytest.raises(ChildStartFailed, match=r"^task_control\.child_start_failed$"),
            child_calls(f"{HERE}:observe_cancellation", workers=2) as calls,
        ):
            calls.make({"started": str(running), "stopped": str(stopped)})
            deadline = time.monotonic() + 10
            while not running.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert running.exists()
            refused_at = time.monotonic()
            calls.make({"started": str(running), "stopped": str(stopped)})
    assert time.monotonic() - refused_at < 3
    assert stopped.read_text(encoding="utf-8") == "cancelled"
    # The first worker is intentionally kept idle; its next answer proves no owned call remains.
    assert run_in_child(f"{HERE}:doubled", {"value": 21})["pid"] == processes[0].pid
    with child_calls(f"{HERE}:doubled", workers=2) as calls:
        calls.make({"value": 21})
        calls.make({"value": 22})
        assert [calls.answer()["value"] for _ in range(2)] == [42, 44]


class _Counted:
    """A payload that counts, in the parent, each time it is pickled for a worker."""

    pickled = 0

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __reduce__(self) -> tuple[type[_Counted], tuple[bytes]]:
        type(self).pickled += 1
        return (_Counted, (self.data,))


def shared_length(value: int, payload: _Counted, cancelled) -> dict[str, int]:
    return {"value": value, "length": len(payload.data)}


def test_a_shared_argument_ships_once_to_each_worker_of_a_call_set() -> None:
    """requirement: a shared argument goes to each worker once per call set, by its
    content key after that; another set ships its own, even interleaved on one worker, so
    nothing one set shipped answers another."""
    payload = _Counted(b"x" * 1_000_000)
    _Counted.pickled = 0
    with child_calls(f"{HERE}:shared_length", workers=2) as calls:
        for value in range(6):
            calls.make({"value": value}, shared={"payload": ("payload-key", payload)})
        answers = [calls.answer() for _ in range(6)]
    assert [answer["value"] for answer in answers] == list(range(6))
    assert {answer["length"] for answer in answers} == {1_000_000}
    assert _Counted.pickled == 2
    with child_calls(f"{HERE}:shared_length", workers=2) as calls:
        calls.make({"value": 6}, shared={"payload": ("payload-key", payload)})
        assert calls.answer() == {"value": 6, "length": 1_000_000}
    assert _Counted.pickled == 3
    small = _Counted(b"y" * 10)
    with (
        child_calls(f"{HERE}:shared_length", workers=1) as first,
        child_calls(f"{HERE}:shared_length", workers=1) as second,
    ):
        for value in range(2):
            first.make({"value": value}, shared={"payload": ("payload-key", payload)})
            assert first.answer()["length"] == 1_000_000
            second.make({"value": value}, shared={"payload": ("payload-key", small)})
            assert second.answer()["length"] == 10
