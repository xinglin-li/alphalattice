"""Task admission and the owner directory of independently held worker pools."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context
from threading import Condition, Event, Lock, Thread, local
from typing import Any
from uuid import UUID
from weakref import WeakKeyDictionary

from .cancel import cancel_execution, execution_cancellation
from .port import ExecutionHeartbeat, ExecutionLane, ExecutionPort, PublicationCheck

_DEFAULT_EXECUTION_TIMEOUT_SECONDS = 24 * 60 * 60
_POOLS: WeakKeyDictionary[Any, str] = WeakKeyDictionary()
_POOL_LOCK = Lock()
_process_pool_owner: Callable[[ProcessPoolExecutor, str], None] | None = None


def install_process_pool_owner(owner: Callable[[ProcessPoolExecutor, str], None]) -> None:
    """Bind the product's process observations at the execution boundary."""
    global _process_pool_owner
    _process_pool_owner = owner


class ExecutionPool:
    """Own the admission lanes for a Host's Task dispatch workers."""

    def __init__(self, width: int = 1) -> None:
        """Create a scheduler that starts its workers on demand."""
        if width < 1:
            raise ValueError("the execution width must be positive")
        self._pool = self.register_pool(
            ThreadPoolExecutor(max_workers=sys.maxsize, thread_name_prefix="task-dispatch"),
            name="task-dispatch",
        )
        self._width = width
        self._condition = Condition()
        self._pending: dict[ExecutionLane, deque[Callable[[], None]]] = {
            "interactive": deque(),
            "background": deque(),
        }
        self._running = 0
        self._closing = False
        self._local = local()

    def register_pool[T](self, pool: T, *, name: str) -> T:
        """Record a pool for its lifetime. Its caller owns its workers and shutdown."""
        if isinstance(pool, ProcessPoolExecutor) and _process_pool_owner is not None:
            _process_pool_owner(pool, name)
        with _POOL_LOCK:
            _POOLS[pool] = name
        return pool

    def registered_pool_names(self) -> tuple[str, ...]:
        """Return the names of the pools currently held in this process."""
        with _POOL_LOCK:
            return tuple(sorted(_POOLS.values()))

    def submit[T](
        self, action: Callable[[], T], *, lane: ExecutionLane = "background"
    ) -> Future[T]:
        """Queue independent work with its own context and return its completion."""
        future: Future[T] = Future()
        context = Context()

        def run() -> None:
            if future.set_running_or_notify_cancel():
                try:
                    result = context.run(action)
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)

        with self._condition:
            if self._closing:
                raise RuntimeError("the execution pool is closed")
            self._pending[lane].append(run)
            self._start_ready()
        return future

    def grow(self, width: int) -> None:
        """Admit an additional running place supplied by the Host."""
        with self._condition:
            self._width = max(self._width, width)
            self._start_ready()

    def _start_ready(self) -> None:
        """Give the next free place to interactive work, FIFO within each lane."""
        interactive = self._pending["interactive"]
        background = self._pending["background"]
        while self._running < self._width and (queue := interactive or background):
            self._running += 1
            self._pool.submit(self._drive, queue.popleft())

    def _drive(self, action: Callable[[], None]) -> None:
        self._local.running = True
        try:
            action()
        finally:
            self._local.running = False
            with self._condition:
                self._running -= 1
                self._start_ready()
                self._condition.notify_all()

    @contextmanager
    def running(
        self,
        execution_id: UUID,
        timeout: float | None,
        heartbeat: ExecutionHeartbeat,
        *,
        publication_check: PublicationCheck | None = None,
    ) -> Iterator[None]:
        """Apply the per-execution deadline and settle the heartbeat on exit."""
        stop = Event()

        def pulse() -> None:
            sequence = 1
            heartbeat.emit(sequence)
            while not stop.wait(heartbeat.interval):
                sequence += 1
                heartbeat.emit(sequence)

        if timeout is None:
            timeout = _DEFAULT_EXECUTION_TIMEOUT_SECONDS
        with execution_cancellation(execution_id, timeout, publication_check=publication_check):
            watcher = Thread(target=pulse, name=heartbeat.name, daemon=True)
            watcher.start()
            try:
                yield
            finally:
                stop.set()
                watcher.join(timeout=max(heartbeat.interval * 2, 1.0))

    def cancel(self, execution_id: UUID | None) -> None:
        """Stop the matching live execution."""
        cancel_execution(execution_id)

    def close(self) -> None:
        """Drain admitted work and join the Host's workers."""
        if getattr(self._local, "running", False):
            raise RuntimeError("a worker cannot close its execution pool")
        with self._condition:
            self._closing = True
            self._condition.wait_for(lambda: self._running == 0)
        self._pool.shutdown()


_PROCESS_LOCK = Lock()
_PROCESS: ExecutionPool | None = None


def process_execution() -> ExecutionPool:
    """Return the process lifetime execution owner without starting idle threads."""
    global _PROCESS
    with _PROCESS_LOCK:
        if _PROCESS is None:
            _PROCESS = ExecutionPool()
        return _PROCESS


def execution_port() -> ExecutionPort:
    """Expose the process owner's Task lifecycle to its domain callers."""
    return process_execution()
