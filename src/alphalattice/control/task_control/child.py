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
import sys
import tempfile
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent import futures
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alphalattice.capabilities.domain_contracts import spawn
from alphalattice.capabilities.domain_contracts import telemetry_names as telemetry
from alphalattice.capabilities.domain_contracts.environment import (
    held_offline,
    numerical_thread_limit,
)
from alphalattice.capabilities.domain_contracts.spans import (
    SpanLedger,
    absorb,
    collect,
    cpu_seconds_of,
    span,
)
from alphalattice.kernel import storage
from alphalattice.kernel.config import Change, declare
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.execution import (
    CANCELLED,
    Bell,
    ExecutionStopped,
    child_process,
    execution_port,
    interrupt_on_execution_stop,
    stop_reason,
    worker_cancellation,
)
from alphalattice.kernel.resources import (
    ResourceLease,
    bound_native_threads,
    cpu_width,
    current_resource_lease,
    use_resource_lease,
    worker_resource_lease,
    worker_width,
)

from .scratch import task_scratch_directory

if TYPE_CHECKING:  # the SDK is imported by a traced call alone
    from opentelemetry.sdk.trace import TracerProvider

    from alphalattice.foundation.data_layer.spans import RowsExporter

_TRACER: tuple[TracerProvider, RowsExporter] | None = None
"""This worker process's tracer provider and the exporter keeping its spans' rows."""
POLL_SECONDS = 0.5
"""How often the waiting parent reads the call's answer and the Task's cancellation."""

WEDGED_SECONDS = 120.0
"""How long the worker owing a call may use no CPU at all before it is taken as wedged: a process
blocked on a lock, a wait or I/O that never returns, suspended, or never started, is ended and
its call interrupted, where a long numerical call keeps using CPU and is waited for."""

KEPT_WORKER_IDLE_SECONDS = 180.0
"""How long a kept worker waits with no call before it retires; the next call starts it again.
An execution parameter: an idle Host keeps no workers the person would see for its whole life."""

_WORKERS: dict[int, ProcessPoolExecutor] = {}
"""Kept workers by position. The ordinary call owns 0 and spread calls own 1 onward."""
_TASK_WORKERS: dict[ResourceLease, dict[int, ProcessPoolExecutor]] = {}
"""Workers started ahead for a Task, owned by its exact grant until it ends."""
_WORKER_LOCK = threading.Lock()
_BUSY: dict[ProcessPoolExecutor, int] = {}
"""Each kept worker's calls in flight; one with none retires after `KEPT_WORKER_IDLE_SECONDS`."""
_BELLS: weakref.WeakKeyDictionary[ProcessPoolExecutor, Bell] = weakref.WeakKeyDictionary()
"""Each kept worker's doorbell: the parent rings it once a call's cancel flag is set, and the
worker's call looks at its flag (`worker_cancellation`)."""
_IDLE: dict[ProcessPoolExecutor, threading.Timer] = {}
"""Each idle kept worker's one pending retirement. A new call or the worker's retirement cancels
it: a pending timer holds its executor, whose semaphores and pipes stay open while it waits."""


def _cancel_idle(pool: ProcessPoolExecutor) -> None:
    """Cancel the pool's pending idle retirement, with `_WORKER_LOCK` held."""
    if (timer := _IDLE.pop(pool, None)) is not None:
        timer.cancel()


_BELL: Bell | None = None
"""In a kept worker, its doorbell."""
_SHARED: dict[str, dict[str, Any]] = {}
"""In a kept worker, by call set, the values that set shipped once, by content key. Sets may
interleave on one worker; each set's close forgets its own."""


@dataclass(frozen=True)
class _Shared:
    """An argument a call set ships to each worker once: its content key, and its value on the
    first call to that worker (`carried`), resolved from the worker's holding after that."""

    key: str
    value: Any = None
    carried: bool = False


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
    with tempfile.TemporaryDirectory(
        prefix="alphalattice-call-", dir=task_scratch_directory()
    ) as folder:
        flag = Path(folder) / "cancel"
        with span("wait", "worker"):
            answer = _waited(_submitted(0, target, arguments, flag, threads=None), flag, cancelled)
    absorb(answer.get("spans"))
    _traced(answer)
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
    ones, told to cancel, and removes the folder. `opens` names the workspace files the function
    opens (None, the default: any), and only those are let go while a call runs (`hand_off`).
    """

    def __init__(
        self,
        target: str,
        *,
        workers: int,
        cancelled: Callable[[], bool] | None = None,
        preload_models: bool = True,
        opens: Iterable[Path] | None = None,
    ) -> None:
        """Bind the function, how many kept workers its calls spread over, and the files it
        opens."""
        if workers < 1:
            raise ValueError("task_control.child_workers_invalid")
        self._target = target
        self._workers = worker_width(workers)
        self._cancelled = cancelled
        self._preload_models = preload_models
        self._opens = None if opens is None else tuple(opens)
        self._made = 0
        self._pending: deque[_Submitted] = deque()
        self._folder = tempfile.TemporaryDirectory(
            prefix="alphalattice-calls-", dir=task_scratch_directory()
        )
        self._shipped: dict[tuple[int, str], ProcessPoolExecutor] = {}
        self._flag = Path(self._folder.name) / "cancel"

    def make(
        self, arguments: Mapping[str, Any], *, shared: Mapping[str, tuple[str, Any]] | None = None
    ) -> None:
        """Make the next call on `arguments` without waiting for it.

        It goes to the first worker with none of these calls pending; with all of them busy,
        behind the oldest call, whose worker frees first. Each of `shared` -- an argument's
        name to its content key and value, the same for many calls -- goes to each worker
        once in these calls: later calls name it by its key, and the worker holds it until
        these calls close. A worker started again receives it again.
        """
        busy = {item.worker for item in self._pending}
        worker = next(
            (place for place in range(1, self._workers + 1) if place not in busy),
            self._pending[0].worker if self._pending else 1,
        )
        if shared:
            pool = _worker(worker, threads=1)
            sent: dict[str, Any] = {"__call_set__": self._folder.name}
            for name, (key, value) in shared.items():
                carried = self._shipped.get((worker, key)) is not pool
                self._shipped[(worker, key)] = pool
                sent[name] = _Shared(key, value if carried else None, carried)
            arguments = {**arguments, **sent}
        answer = Path(self._folder.name) / f"{self._made}.answer"
        self._made += 1
        self._pending.append(
            _submitted(
                worker,
                self._target,
                arguments,
                self._flag,
                threads=1,
                answer=answer,
                preload_models=self._preload_models,
                opens=self._opens,
            )
        )

    def answer(self) -> Any:
        """The oldest unanswered call's answer, raising its failure as `run_in_child` does."""
        submitted = self._pending.popleft()
        with span("wait", "worker"):
            answer = _waited(
                submitted,
                self._flag,
                self._cancelled,
                lambda: [item.pool for item in self._pending],
            )
        absorb(answer.get("spans"))
        _traced(answer)
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
            _relay(self._flag, {item.pool for item in pending})
            if stop_reason() is not None:
                for pool in {item.pool for item in pending if _task_owns_pool(item.pool)}:
                    spawn.stop_pool(pool)
            futures.wait(running)
        for item in pending:
            if not item.future.cancelled() and isinstance(
                item.future.exception(), BrokenProcessPool
            ):
                _retire_worker(item.worker, item.pool, wait_for_exit=True)
        # Each worker these calls shipped to forgets them, after their calls, in its own order.
        for worker in {worker for worker, _key in self._shipped}:
            if _kept_workers().get(worker) in self._shipped.values():
                # A worker that has ended took its holding with it.
                with suppress(ChildInterrupted):
                    _submitted(
                        worker,
                        f"{__name__}:_forget",
                        {"call_set": self._folder.name},
                        self._flag,
                        threads=1,
                        preload_models=self._preload_models,
                        opens=(),
                    )
        self._folder.cleanup()


@contextmanager
def child_calls(
    target: str,
    *,
    workers: int,
    cancelled: Callable[[], bool] | None = None,
    preload_models: bool = True,
    opens: Iterable[Path] | None = None,
) -> Iterator[ChildCalls]:
    """`ChildCalls` of ``module:function`` over `workers` kept workers, closed on leaving."""
    calls = ChildCalls(
        target, workers=workers, cancelled=cancelled, preload_models=preload_models, opens=opens
    )
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
    preload_models: bool = True,
    opens: Iterable[Path] | None = None,
) -> _Submitted:
    """Hand one call to a kept worker; one that has ended answers as an interruption. The worker
    may open the files this process holds that `opens` names (None: any), so none of them is kept
    open until the call ends."""
    from alphalattice.control.workspace_runtime.database import hand_off  # lazy: DuckDB
    from alphalattice.foundation.data_layer.root import DATA_ROOT_ENV  # lazy: the data layer

    pool = None
    ended = hand_off(opens)
    try:
        pool = _worker(worker, threads=threads)
        future = pool.submit(
            _call,
            target,
            dict(arguments),
            str(flag),
            threads,
            None if answer is None else str(answer),
            _carrier(),
            process_config().get(DATA_ROOT_ENV),
            preload_models,
            worker_resource_lease(threads),
        )
    except BrokenProcessPool:
        ended()
        assert pool is not None
        raise _ended(worker, pool) from None
    except (OSError, MemoryError) as error:
        ended()
        if pool is not None:
            _retire_worker(worker, pool)
        raise ChildStartFailed(error) from error
    except BaseException:
        ended()
        raise
    future.add_done_callback(lambda _done: ended())
    _hold(worker, pool, future)
    return _Submitted(worker, pool, future, answer)


def _hold(worker: int, pool: ProcessPoolExecutor, future: Future[Any]) -> None:
    """Count a call in flight on a kept worker; its last call's end starts the idle wait."""
    with _WORKER_LOCK:
        if _WORKERS.get(worker) is not pool:
            return  # The Task owns its ahead workers through the end of its grant.
        _BUSY[pool] = _BUSY.get(pool, 0) + 1
        _cancel_idle(pool)

    def ended(_done: Future[Any]) -> None:
        with _WORKER_LOCK:
            if pool not in _BUSY:
                return
            _BUSY[pool] -= 1
            if _BUSY[pool]:
                return
            _cancel_idle(pool)
            timer = _IDLE[pool] = threading.Timer(
                KEPT_WORKER_IDLE_SECONDS, _retire_idle, (worker, pool)
            )
            timer.daemon = True
        timer.start()

    future.add_done_callback(ended)


def _retire_idle(worker: int, pool: ProcessPoolExecutor) -> None:
    """Retire a kept worker that took no call since its idle wait began."""
    with _WORKER_LOCK:
        if _IDLE.get(pool) is not threading.current_thread():
            return  # Cancelled while it waited for the lock: a call took the pool, or it retired.
        del _IDLE[pool]
        if _BUSY.get(pool) or _WORKERS.get(worker) is not pool:
            return
        del _WORKERS[worker]
        _BUSY.pop(pool, None)
    pool.shutdown(wait=False, cancel_futures=False)


def _waited(
    submitted: _Submitted,
    flag: Path,
    cancelled: Callable[[], bool] | None,
    sharing: Callable[[], Iterable[ProcessPoolExecutor]] = tuple,
) -> dict[str, Any]:
    """Wait for a call's answer, relaying the Task's cancellation while it runs to its worker
    and to the workers of ``sharing``, the other calls running under the same flag."""
    progress: tuple[float | None, float] = (None, time.monotonic())
    # The Task's stop reaches the workers when it is accepted, through the execution's interrupt;
    # one accepted before this wait began is relayed at once.
    try:
        release = interrupt_on_execution_stop(lambda: _relay(flag, {submitted.pool, *sharing()}))
    except ExecutionStopped:  # already timed out
        release = _unheld
    if stop_reason() is not None:
        _relay(flag, {submitted.pool, *sharing()})
    try:
        # The answer ends the wait at once; each POLL_SECONDS without one relays a cancel only
        # the Task's predicate knows and samples the worker's CPU for the wedge.
        while (answer := _answer(submitted.future)) is None:
            reason = stop_reason()
            if reason is not None or (cancelled is not None and cancelled()):
                _relay(flag, {submitted.pool, *sharing()})
                if _task_owns_pool(submitted.pool):
                    spawn.stop_pool(submitted.pool)
                    _retire_worker(submitted.worker, submitted.pool, wait_for_exit=True)
                    raise ExecutionStopped(reason or CANCELLED) from None
            progress = _progressed(submitted.pool, progress)
            if time.monotonic() - progress[1] > WEDGED_SECONDS:
                raise _ended_wedged(submitted) from None
        return answer
    except BrokenProcessPool as broken:
        # The future keeps this exception; its traceback holds these frames, which hold the
        # future and its retired pool, a cycle only a collection would end.
        broken.__traceback__ = None
        raise _ended(submitted.worker, submitted.pool) from None
    finally:
        release()


def _unheld() -> None:
    """No interrupt to release."""


def _relay(flag: Path, pools: Iterable[ProcessPoolExecutor]) -> None:
    """Tell the calls under ``flag`` to cancel: set the flag, then ring each worker's bell."""
    if not flag.exists():
        flag.touch()
    for pool in pools:
        if (bell := _BELLS.get(pool)) is not None:
            bell.set()


def _answer(future: Future[Any]) -> dict[str, Any] | None:
    """The call's answer, or None while it has none after POLL_SECONDS."""
    try:
        answer: dict[str, Any] = future.result(timeout=POLL_SECONDS)
    except FutureTimeout:
        return None
    return answer


def _progressed(
    pool: ProcessPoolExecutor, last: tuple[float | None, float]
) -> tuple[float | None, float]:
    """The worker's CPU seconds and when they last moved. Whatever the worker runs for this
    call -- the call itself, or one queued before it -- uses CPU while it works; a worker with
    no process to read (one that never started) never moves."""
    used = _cpu_seconds(pool)
    return last if used is None or used == last[0] else (used, time.monotonic())


def _cpu_seconds(pool: ProcessPoolExecutor) -> float | None:
    """The CPU seconds the pool's worker process has used, or None when it has none to read."""
    for process in list(getattr(pool, "_processes", {}).values()):
        if process.pid is not None:
            return cpu_seconds_of(process.pid)
    return None


def _ended_wedged(submitted: _Submitted) -> ChildInterrupted:
    """End a wedged worker's process and retire its pool; its call is interrupted by name."""
    _terminate_processes(submitted.pool)
    _retire_worker(submitted.worker, submitted.pool, wait_for_exit=True)
    return ChildInterrupted(
        "task_control.child_wedged", type_name="ChildWedged", failure_class=None
    )


def _terminate_processes(pool: ProcessPoolExecutor) -> None:
    """Signal the processes this exact pool still owns before joining its manager."""
    spawn.stop_pool(pool)


def _ended(worker: int, pool: ProcessPoolExecutor) -> ChildInterrupted:
    """The interruption for a worker that ended; the next call to its place starts a new one."""
    _retire_worker(worker, pool, wait_for_exit=True)
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


def _kept_workers() -> dict[int, ProcessPoolExecutor]:
    lease = current_resource_lease()
    return _WORKERS if lease is None else _TASK_WORKERS.get(lease, _WORKERS)


def _task_owns_pool(pool: ProcessPoolExecutor) -> bool:
    lease = current_resource_lease()
    if lease is None:
        return False
    with _WORKER_LOCK:
        return pool in _TASK_WORKERS.get(lease, {}).values()


def stop_task_workers() -> None:
    """Join this Task's workers before a terminal outcome can release its running place."""
    lease = current_resource_lease()
    with _WORKER_LOCK:
        owned = _TASK_WORKERS.get(lease) if lease is not None else None
        pools = tuple(owned.values()) if owned is not None else ()
    for pool in pools:
        _terminate_processes(pool)
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)
    if owned is not None:
        with _WORKER_LOCK:
            owned.clear()


@contextmanager
def warm_child_workers(workers: int, *, modules: tuple[str, ...]) -> Iterator[Callable[[], None]]:
    """Hold a Task's kept workers and give its runner the startup after claim."""
    from alphalattice.foundation.data_layer.root import DATA_ROOT_ENV  # lazy: the data layer

    lease = current_resource_lease()
    assert lease is not None
    with _WORKER_LOCK:
        _TASK_WORKERS[lease] = {}

    def start() -> None:
        warm = (modules, process_config().get(DATA_ROOT_ENV), worker_resource_lease(1))
        try:
            for worker in range(1, worker_width(workers) + 1):
                _worker(worker, threads=1, warm=warm).submit(os.getpid)
        except (OSError, MemoryError) as error:
            raise ChildStartFailed(error) from error

    try:
        yield start
    finally:
        try:
            stop_task_workers()
        finally:
            with _WORKER_LOCK:
                del _TASK_WORKERS[lease]


type _WarmImports = tuple[tuple[str, ...], str | None, ResourceLease | None]


def _worker(
    worker: int, *, threads: int | None, warm: _WarmImports | None = None
) -> ProcessPoolExecutor:
    """The kept worker at `worker`, started if none; a kept one leaves its idle wait, so it is not
    retired between its selection and the call `_hold` counts."""
    with _WORKER_LOCK:
        workers = _kept_workers()
        kept = workers.get(worker)
        if kept is not None:
            _cancel_idle(kept)
        else:
            context = multiprocessing.get_context("spawn")
            bell = context.Event()
            kept = workers[worker] = execution_port().register_pool(
                ProcessPoolExecutor(
                    max_workers=worker_width(1),
                    mp_context=context,
                    initializer=_worker_started,
                    initargs=(cpu_width(threads), warm, bell),
                ),
                name="task-child",
            )
            _BELLS[kept] = bell
        return kept


def _retire_worker(worker: int, pool: ProcessPoolExecutor, *, wait_for_exit: bool = False) -> None:
    """Retire the worker at `worker` if it is still the one that ended, never its replacement."""
    # On a broken or terminated pool the manager joins the process. The Task keeps
    # its grant until that exit completes, including when a call loses its reply.
    if wait_for_exit:
        pool.shutdown(wait=True, cancel_futures=False)
    with _WORKER_LOCK:
        workers = _kept_workers()
        if workers.get(worker) is pool:
            del workers[worker]
        _BUSY.pop(pool, None)
        _cancel_idle(pool)
    if not wait_for_exit:
        pool.shutdown(wait=False, cancel_futures=False)


def _worker_started(threads: int, warm: _WarmImports | None, bell: Bell) -> None:
    global _BELL
    _BELL = bell
    bound_native_threads(threads)
    threading.Thread(target=child_process.watch_parent, name="parent-watch", daemon=True).start()
    # The worker's DuckDB driver, installed in the storage port before any call opens a file.
    from alphalattice.control.workspace_runtime import reader_threads  # lazy: the worker's own

    storage.install_driver(reader_threads.DRIVER)
    if warm is not None:
        modules, root, grant = warm
        with use_resource_lease(grant), held_offline():
            _follow_root(root)
            for module in modules:
                importlib.import_module(module)
            with _numerical_threads(1, preload_models=False):
                pass


_RECORDED: set[Path] = set()
"""The data roots this worker has recorded itself in (`record_worker`)."""


def _follow_root(root: str | None) -> None:
    """Name the caller's data root in this worker's environment, which `data_root` reads, and
    record this worker in it once, so a Host started after this one's Host ends waits for it."""
    from alphalattice.foundation.data_layer.log import record_worker  # lazy: the data layer
    from alphalattice.foundation.data_layer.root import DATA_ROOT_ENV, data_root  # lazy: the same

    declare(Change("task-worker-data-root", {DATA_ROOT_ENV: root}), native=True)
    where = data_root()
    host = spawn.worker_host_pid()
    if where not in _RECORDED and host is not None:
        record_worker(where, host=host)
        _RECORDED.add(where)


def _call(
    target: str,
    arguments: dict[str, Any],
    flag: str,
    threads: int | None = None,
    answer: str | None = None,
    trace: Mapping[str, str] | None = None,
    root: str | None = None,
    preload_models: bool = True,
    grant: ResourceLease | None = None,
) -> dict[str, Any]:
    """One call in the worker, held offline: its result, or its failure's kind, type and text.

    The worker runs a Task's numerical work, which reads only sealed inputs, so each call holds
    its reads offline whatever the workspace allows, as the parent's run does. With
    `threads`, the numerical libraries are held to that many threads for the call; with
    `answer`, the result is written to that file and the pipe carries only that it was. The
    call's spans, with the worker's CPU seconds and bytes over it, ride beside the answer for
    the parent's stage ledger (A4). With `trace`, the W3C context of the step that sent the call,
    the call is one `worker.call` span in that trace, handed back as its step event (`trace`).
    The call reads the data root its caller names (`root`, unset for the default): a kept worker
    started under another root, before the caller's history opened, reads the caller's.
    """
    ledger: SpanLedger | None = None
    rows: list[dict[str, Any]] = []
    try:
        assert _BELL is not None, "a call runs in a kept worker, which holds its bell"
        with use_resource_lease(grant), worker_cancellation(flag, _BELL):
            _follow_root(root)
            arguments = _held(arguments)
            module_name, _, name = target.partition(":")
            function = getattr(importlib.import_module(module_name), name)
            with (
                held_offline(),
                _numerical_threads(threads, preload_models=preload_models),
                collect() as ledger,
                _worker_span(target, trace, rows),
            ):
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
            "trace": rows or None,
        }
    return {
        "status": "OK",
        "result": result,
        "spans": None if ledger is None else ledger.readout_value,
        "trace": rows or None,
    }


def _carrier() -> dict[str, str]:
    """The trace of the step sending a call. Only a Host loads its history, so a process that has
    not cannot be recording, and pays no import for it."""
    history = sys.modules.get("alphalattice.foundation.data_layer.history")
    return {} if history is None else history.trace_carrier()


def _traced(answer: Mapping[str, Any]) -> None:
    """Hand the worker's spans to the Host's log, when the call carried a trace."""
    if answer.get("trace"):
        from alphalattice.foundation.data_layer.history import traced  # lazy: a traced answer alone

        traced(answer["trace"])


@contextmanager
def _worker_span(
    target: str, trace: Mapping[str, str] | None, rows: list[dict[str, Any]]
) -> Iterator[None]:
    """The call as one `worker.call` span in the trace `trace` names; its step event goes to
    `rows`. Without a trace, nothing: the SDK is imported only by a traced call."""
    if not trace:
        yield
        return
    from alphalattice.foundation.data_layer.spans import (  # lazy: the SDK, for a traced call alone
        RowsExporter,
        context_of,
        tracer_provider,
    )

    global _TRACER
    if _TRACER is None:  # one per worker process, kept with it
        kept = RowsExporter()
        _TRACER = (tracer_provider(kept), kept)
    provider, exporter = _TRACER
    tracer = provider.get_tracer("worker")
    try:
        with tracer.start_as_current_span(
            telemetry.SPAN_WORKER_CALL, context_of(trace), attributes={telemetry.CALL: target}
        ):
            yield
    finally:
        rows.extend(exporter.rows)
        exporter.rows.clear()


def _held(arguments: dict[str, Any]) -> dict[str, Any]:
    """A call's arguments with each shipped-once value resolved from this worker's holding."""
    call_set = arguments.pop("__call_set__", None)
    if call_set is None:
        return arguments
    holding = _SHARED.setdefault(call_set, {})
    for name, value in arguments.items():
        if isinstance(value, _Shared):
            if value.carried:
                holding[value.key] = value.value
            arguments[name] = holding[value.key]
    return arguments


def _forget(call_set: str, cancelled: Callable[[], bool]) -> None:
    """In a kept worker, drop what a closed call set shipped (`ChildCalls.close`)."""
    _SHARED.pop(call_set, None)


@contextmanager
def _numerical_threads(threads: int | None, *, preload_models: bool) -> Iterator[None]:
    """The numerical libraries held to `threads` threads for a block; unchanged without one."""
    if current_resource_lease() is not None:
        threads = cpu_width(threads)
    if threads is None:
        yield
        return
    with numerical_thread_limit(threads, preload_models=preload_models):
        yield


__all__ = [
    "ChildCalls",
    "ChildInterrupted",
    "ChildRefused",
    "ChildStartFailed",
    "child_calls",
    "run_in_child",
    "warm_child_workers",
]
