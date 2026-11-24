from concurrent.futures import Future, ProcessPoolExecutor
from itertools import count
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from alphalattice.control.task_control import child
from alphalattice.kernel.execution import (
    CANCELLED,
    cancel_execution,
    child_process,
    execution_cancellation,
    stop_reason,
)

TEST_SOURCE = "src/alphalattice/control/task_control/child.py"


@pytest.mark.proves("execution.cancellation_matches_one_execution")
def test_cancellation_belongs_to_the_named_execution():
    """Only an accepted signal naming the active execution reaches its work."""
    execution = UUID(int=1)
    with execution_cancellation(execution):
        cancel_execution(UUID(int=2))
        assert stop_reason() is None
        cancel_execution(execution)
        assert stop_reason() == CANCELLED
    assert stop_reason() is None


class _Ended(Exception):
    """Stands in for the worker's exit, so the test process itself goes on."""


@pytest.mark.proves("task_control.a_worker_ends_with_its_parent")
@pytest.mark.parametrize(
    "parent",
    [
        pytest.param(SimpleNamespace(sentinel=7), id="blocks-on-the-parents-sentinel"),
        pytest.param(None, id="no-parent-ends-at-once"),
    ],
)
def test_a_kept_worker_ends_when_its_parent_ends(monkeypatch, parent):
    """A kept worker blocks on its parent's sentinel, with no loop or timer, and ends with
    code 3 when it returns; a worker with no parent ends at once."""
    waited: list[list[int]] = []

    def ended(code: int) -> None:
        raise _Ended(code)

    monkeypatch.setattr(child_process.multiprocessing, "parent_process", lambda: parent)
    monkeypatch.setattr(child_process.multiprocessing.connection, "wait", waited.append)
    monkeypatch.setattr(child_process, "os", SimpleNamespace(_exit=ended))  # its exit, recorded
    with pytest.raises(_Ended, match="3"):
        child_process.watch_parent()
    assert waited == ([] if parent is None else [[7]])


@pytest.mark.proves("task_control.a_wedged_worker_is_ended_by_name")
@pytest.mark.parametrize(
    ("used", "ended"),
    [
        pytest.param([3.0] * 9, ["terminated", "shut down"], id="no-cpu-past-the-bound-is-ended"),
        pytest.param([1.0, 2.0, 3.0, 4.0], [], id="a-call-using-cpu-is-waited-for"),
        pytest.param([None] * 9, ["terminated", "shut down"], id="a-worker-never-started-is-ended"),
    ],
)
def test_a_worker_that_makes_no_progress_is_ended_and_its_call_interrupted(
    tmp_path, monkeypatch, used, ended
):
    """A call whose worker uses no CPU for longer than its bound ends that worker's process and
    is interrupted by name; a worker that keeps using CPU is waited for to its answer."""
    clock, cpu, ends = count(0, 100), iter(used), []
    monkeypatch.setattr(child.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(child, "cpu_seconds_of", lambda _pid: next(cpu))
    monkeypatch.setattr(child.spawn, "stop_pool", lambda _pool: ends.append("terminated"))

    class _Running:
        def result(self, timeout: float) -> dict:
            if next(polls) == len(used) - 1:
                return {"status": "OK"}
            raise child.FutureTimeout

    class _Pool:
        def __init__(self) -> None:
            self._processes = {7: SimpleNamespace(pid=7)}

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            ends.append("shut down")

    polls = count()
    pool = cast("ProcessPoolExecutor", _Pool())
    monkeypatch.setitem(child._BUSY, pool, 1)
    submitted = child._Submitted(2, pool, cast("Future", _Running()))
    if ended:
        with pytest.raises(child.ChildInterrupted, match=r"task_control\.child_wedged"):
            child._waited(submitted, tmp_path / "cancel", None)
    else:
        assert child._waited(submitted, tmp_path / "cancel", None) == {"status": "OK"}
    assert ends == ended
    assert (pool in child._BUSY) == (not ended)
