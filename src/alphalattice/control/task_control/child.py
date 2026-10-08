"""A Task's numerical work in the Host's worker processes (W10; LAWS OW2).

The Host keeps the Task: its registry, its receipts and every database write, so it stays
DuckDB's one writer, and its light reads no longer wait for the GIL behind a fit's Python work
(measured: a Host's STATUS p95 went from 174 ms idle to 2.2 s during an Alpha fit, its request
threads queued on the registry's lock whose holds the fit's row hashing stretched). One worker
process, started when the Host first needs it and kept, runs one call at a time, named by its
module and function, on arguments the parent hands it (frozen records, specs and paths): it
writes only the files its owner writes and answers with its result or with its failure's kind,
type and text. Kept warm, it pays its imports once and keeps what it verified. The parent
relays a Task's cancellation through a flag file the call reads; a worker whose parent is gone
stops.

Work of many independent units -- a first Feature build's listings -- spreads over up to N of
these kept workers (`ChildCalls`): worker 0 is the one a Task's call uses, and the others are
started beside it the first time they are needed and kept the same way. Each still runs one call
at a time; each such call holds the numerical libraries to one thread, so the workers do not
oversubscribe the processors; and the answers come back in the order the calls were made, so the
work's result does not depend on how many workers ran it.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
import pickle
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from concurrent import futures
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alphalattice.kernel.shared_kernel.environment import held_offline
from alphalattice.kernel.shared_kernel.spans import (
    SpanLedger,
    absorb,
    collect,
    span,
)

POLL_SECONDS = 0.5
"""How often the waiting parent reads the call's answer and the Task's cancellation."""
PARENT_WATCH_SECONDS = 2.0
"""How often the worker looks for its parent; a worker whose parent is gone ends at once."""

_WORKERS: dict[int, ProcessPoolExecutor] = {}
"""The kept workers by position; 0 is the Task's."""
_WORKER_LOCK = threading.Lock()


class ChildRefused(ValueError):
    """The worker's call refused its inputs (a `ValueError` there): its text, as a refusal here."""

    def __init__(self, message: str, *, type_name: str) -> None:
        """Keep the call's text as this refusal's and its exception's type name."""
        super().__init__(message)
        self.type_name = type_name


class ChildInterrupted(RuntimeError):
    """The worker's call failed otherwise, or the worker ended without answering."""

    def __init__(self, message: str, *, type_name: str, failure_class: str | None) -> None:
        """Keep the call's text, its exception's type name and its declared failure class."""
        super().__init__(message)
        self.type_name = type_name
        if failure_class is not None:
            self.failure_class = failure_class


class ChildStartFailed(ChildInterrupted):
    """The OS refused a worker's creation; no call was handed to that worker."""

    def __init__(self, error: OSError | MemoryError) -> None:
        """Keep a stable stop code and the original exception as the raised cause."""
        super().__init__(
            "task_control.child_start_failed", type_name=type(error).__name__, failure_class=None
        )


def run_in_child(
    target: str,
    arguments: Mapping[str, Any],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> Any:
    """Run ``module:function`` in the Host's worker process on `arguments`; its answer.

    The function takes the arguments as keywords and ``cancelled``, a callable that reads true
    once the parent has relayed the Task's cancellation. The arguments and the answer cross by
    pickle; calls run one at a time.

    Args:
        target: The function, as ``package.module:name``.
        arguments: Its keyword arguments.
        cancelled: The Task's cancellation, read while the call runs.

    Returns:
        What the function returned.

    Raises:
        ChildRefused: The call raised a `ValueError`.
        ChildStartFailed: The OS refused the worker's creation.
        ChildInterrupted: The call raised anything else, or the worker ended without answering.
    """
    with tempfile.TemporaryDirectory(prefix="alphalattice-call-") as folder:
        flag = Path(folder) / "cancel"
        with span("wait", "worker"):
            answer = _waited(_submitted(0, target, arguments, flag, threads=None), flag, cancelled)
    absorb(answer.get("spans"))
    return _answered(answer)


class ChildCalls:
    """Calls of one function over the first `workers` kept workers, answered in the order made.

    A call goes to the first of those workers with no call of these pending, so a caller that
    keeps k calls ahead of the answers it takes keeps k workers busy and starts no more. Each
    call holds the numerical libraries to one thread. Its answer comes back through a file in
    the calls' own folder rather than the worker's pipe: a Windows pipe refuses a write of
    megabytes when the machine runs short of kernel resources (WinError 1450, measured under
    load with a first build's ~10 MB answers), where a file write does not. Use it as a context
    manager (`child_calls`): leaving it withdraws the calls not started, waits for the running
    ones, told to cancel, and removes the folder.
    """

    def __init__(
        self, target: str, *, workers: int, cancelled: Callable[[], bool] | None = None
    ) -> None:
        """Bind the function and how many kept workers its calls spread over."""
        if workers < 1:
            raise ValueError("task_control.child_workers_invalid")
        self._target = target
        self._workers = workers
        self._cancelled = cancelled
        self._made = 0
        self._pending: deque[_Submitted] = deque()
        self._folder = tempfile.TemporaryDirectory(prefix="alphalattice-calls-")
        self._flag = Path(self._folder.name) / "cancel"

    def make(self, arguments: Mapping[str, Any]) -> None:
        """Make the next call on `arguments` without waiting for it.

        It goes to the first worker with none of these calls pending; with all of them busy,
        behind the oldest call, whose worker frees first.
        """
        busy = {item.worker for item in self._pending}
        worker = next(
            (index for index in range(self._workers) if index not in busy),
            self._pending[0].worker if self._pending else 0,
        )
        answer = Path(self._folder.name) / f"{self._made}.answer"
        self._made += 1
        self._pending.append(
            _submitted(worker, self._target, arguments, self._flag, threads=1, answer=answer)
        )

    def answer(self) -> Any:
        """The oldest unanswered call's answer, raising its failure as `run_in_child` does."""
        submitted = self._pending.popleft()
        with span("wait", "worker"):
            answer = _waited(submitted, self._flag, self._cancelled)
        absorb(answer.get("spans"))
        if answer["status"] != "OK" or submitted.answer is None:
            return _answered(answer)
        with submitted.answer.open("rb") as source:
            result = pickle.load(source)
        submitted.answer.unlink()
        return result

    def close(self) -> None:
        """Withdraw the calls not started; tell the running ones to cancel and wait for them."""
        pending, self._pending = self._pending, deque()
        running = [item.future for item in pending if not item.future.cancel()]
        if running:
            self._flag.touch()
            futures.wait(running)
        self._folder.cleanup()


@contextmanager
def child_calls(
    target: str, *, workers: int, cancelled: Callable[[], bool] | None = None
) -> Iterator[ChildCalls]:
    """`ChildCalls` of ``module:function`` over `workers` kept workers, closed on leaving."""
    calls = ChildCalls(target, workers=workers, cancelled=cancelled)
    try:
        yield calls
    finally:
        calls.close()


@dataclass(frozen=True)
class _Submitted:
    """One call handed to a kept worker: which worker, the process pool it had then, the call,
    and the file its answer comes back through, when it does not come through the pipe."""

    worker: int
    pool: ProcessPoolExecutor
    future: Future[dict[str, Any]]
    answer: Path | None = None


def _submitted(
    worker: int,
    target: str,
    arguments: Mapping[str, Any],
    flag: Path,
    *,
    threads: int | None,
    answer: Path | None = None,
) -> _Submitted:
    """Hand one call to a kept worker; one that has ended answers as an interruption."""
    pool = None
    try:
        pool = _worker(worker)
        future = pool.submit(
            _call,
            target,
            dict(arguments),
            str(flag),
            threads,
            None if answer is None else str(answer),
        )
    except BrokenProcessPool:
        assert pool is not None
        raise _ended(worker, pool) from None
    except (OSError, MemoryError) as error:
        if pool is not None:
            _retire_worker(worker, pool)
        raise ChildStartFailed(error) from error
    return _Submitted(worker, pool, future, answer)


def _waited(
    submitted: _Submitted, flag: Path, cancelled: Callable[[], bool] | None
) -> dict[str, Any]:
    """Wait for a call's answer, relaying the Task's cancellation while it runs."""
    try:
        while True:
            try:
                return submitted.future.result(timeout=POLL_SECONDS)
            except FutureTimeout:
                if cancelled is not None and not flag.exists() and cancelled():
                    flag.touch()
    except BrokenProcessPool:
        raise _ended(submitted.worker, submitted.pool) from None


def _ended(worker: int, pool: ProcessPoolExecutor) -> ChildInterrupted:
    """The interruption for a worker that ended; the next call to its place starts a new one."""
    _retire_worker(worker, pool)
    return ChildInterrupted(
        "task_control.child_process_ended", type_name="BrokenProcessPool", failure_class=None
    )


def _answered(answer: dict[str, Any]) -> Any:
    """A call's result, or its failure raised by kind."""
    if answer["status"] == "OK":
        return answer["result"]
    if answer["kind"] == "REFUSED":
        raise ChildRefused(answer["message"], type_name=answer["type"])
    raise ChildInterrupted(
        answer["message"], type_name=answer["type"], failure_class=answer["failure_class"]
    )


def _worker(worker: int) -> ProcessPoolExecutor:
    with _WORKER_LOCK:
        kept = _WORKERS.get(worker)
        if kept is None:
            kept = _WORKERS[worker] = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_started,
                initargs=(worker,),
            )
        return kept


def _retire_worker(worker: int, pool: ProcessPoolExecutor) -> None:
    """Retire the worker at `worker` if it is still the one that ended, never its replacement."""
    with _WORKER_LOCK:
        if _WORKERS.get(worker) is pool:
            del _WORKERS[worker]
    pool.shutdown(wait=False, cancel_futures=True)


ONE_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
"""The numerical libraries' thread variables, each read once, when its library loads."""


def _worker_started(worker: int = 0) -> None:
    # A spread worker runs every call on one thread (`threads=1`): its libraries are told so
    # before they load, and reserve one thread's buffers rather than one per processor, about
    # 1.5 GB of commit a worker (V436). The Task's own worker keeps the default its calls run
    # under (`threads=None`), on which a Risk number can depend (LAWS PA3).
    if worker > 0:
        for name in ONE_THREAD_VARIABLES:
            os.environ[name] = "1"
    threading.Thread(target=_watch_parent, name="parent-watch", daemon=True).start()


def _call(
    target: str,
    arguments: dict[str, Any],
    flag: str,
    threads: int | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    """One call in the worker, held offline: its result, or its failure's kind, type and text.

    The worker runs a Task's numerical work, which reads only sealed inputs, so each call holds
    its reads offline whatever the workspace allows, as the parent's run does (V116). With
    `threads`, the numerical libraries are held to that many threads for the call; with
    `answer`, the result is written to that file and the pipe carries only that it was. The
    call's spans, with the worker's CPU seconds and bytes over it, ride beside the answer for
    the parent's stage ledger (A4).
    """
    ledger: SpanLedger | None = None
    try:
        module_name, _, name = target.partition(":")
        function = getattr(importlib.import_module(module_name), name)
        with held_offline(), _numerical_threads(threads), collect() as ledger:
            result = function(**arguments, cancelled=lambda: os.path.exists(flag))
        if answer is not None:
            with open(answer, "wb") as sink:
                pickle.dump(result, sink, protocol=pickle.HIGHEST_PROTOCOL)
            result = None
    except BaseException as error:
        declared = getattr(error, "failure_class", None)
        return {
            "status": "FAILED",
            "kind": "REFUSED" if isinstance(error, ValueError) else "INTERRUPTED",
            "type": type(error).__name__,
            "message": str(error)[:2000],
            "failure_class": declared if isinstance(declared, str) else None,
            "spans": None if ledger is None else ledger.readout_value,
        }
    return {
        "status": "OK",
        "result": result,
        "spans": None if ledger is None else ledger.readout_value,
    }


@contextmanager
def _numerical_threads(threads: int | None) -> Iterator[None]:
    """The numerical libraries held to `threads` threads for a block; unchanged without one."""
    if threads is None:
        yield
        return
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    with threadpool_limits(limits=threads):
        yield


def _watch_parent() -> None:
    parent = multiprocessing.parent_process()
    while True:
        time.sleep(PARENT_WATCH_SECONDS)
        if parent is None or not parent.is_alive():
            os._exit(3)


__all__ = [
    "ChildCalls",
    "ChildInterrupted",
    "ChildRefused",
    "ChildStartFailed",
    "child_calls",
    "run_in_child",
]
