"""Exact execution cancellation and its per-operation interrupt observers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from threading import Condition, Lock, Thread, Timer
from time import monotonic
from uuid import UUID

CANCELLED = "execution.cancelled"
TIMED_OUT = "execution.timeout"
_INTERRUPT_INTERVAL = 0.025


class ExecutionStopped(RuntimeError):
    """The current execution reached its accepted cancellation or deadline."""


class _Cancellation:
    def __init__(self, flag: Path | None = None) -> None:
        self.reason: str | None = None
        self._flag = flag
        self._condition = Condition()
        self._interrupts: dict[object, Callable[[], object]] = {}
        self._watcher: Thread | None = None
        self._closed = False
        self._deadline = 0.0

    def stop(self, reason: str) -> None:
        with self._condition:
            self._stop(reason)
            if self._interrupts:
                self._start()
                self._condition.notify_all()

    def _stop(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
            self._deadline = monotonic() + 1.0

    def _start(self) -> None:
        if self._watcher is None:
            self._watcher = Thread(target=self._watch, name="execution-interrupt", daemon=True)
            self._watcher.start()

    def _watch(self) -> None:
        with self._condition:
            try:
                while not self._closed:
                    if not self._interrupts:
                        if self._flag is None:
                            return
                        self._condition.wait(timeout=1.0)
                        continue
                    if self.reason is None and self._flag is not None and self._flag.exists():
                        self._stop(CANCELLED)
                    if self.reason is not None:
                        if monotonic() >= self._deadline:
                            return
                        # Keep attempting only while the same operation holds its registration.
                        # An interrupt observed just before engine entry can otherwise be lost.
                        for interrupt in self._interrupts.values():
                            with suppress(Exception):
                                interrupt()
                    self._condition.wait(_INTERRUPT_INTERVAL)
            finally:
                self._watcher = None

    def watch(self, interrupt: Callable[[], object]) -> object | None:
        with self._condition:
            if self.reason is None and self._flag is not None and self._flag.exists():
                self._stop(CANCELLED)
            if self.reason == TIMED_OUT:
                raise ExecutionStopped(TIMED_OUT)
            if self._closed or self.reason is not None:
                # A cancelled Task can finish its current chunk's writes before seeing the stop.
                return None
            first = not self._interrupts
            registration = object()
            self._interrupts[registration] = interrupt
            if self._flag is not None and first:
                # One observer waits between queries within the same kept worker call.
                self._start()
                self._condition.notify()
            return registration

    def unwatch(self, registration: object) -> None:
        # Serial with the last interrupt, so a returned cursor cannot be hit again.
        with self._condition:
            self._interrupts.pop(registration)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            watcher = self._watcher
            self._condition.notify()
        if watcher is not None:
            watcher.join(timeout=1.0)


_CURRENT: ContextVar[_Cancellation | None] = ContextVar("execution_cancellation", default=None)
_RUNNING: dict[UUID, _Cancellation] = {}
_RUNNING_LOCK = Lock()


@contextmanager
def execution_cancellation(execution_id: UUID, timeout: float | None = None) -> Iterator[None]:
    """Bind the execution's signal for its lifetime, including its nested thread work."""
    signal = _Cancellation()
    with _RUNNING_LOCK:
        _RUNNING[execution_id] = signal
    reset = _CURRENT.set(signal)
    timer = None if timeout is None else Timer(timeout, signal.stop, (TIMED_OUT,))
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        yield
    finally:
        if timer is not None:
            timer.cancel()
            timer.join(timeout=1.0)
        signal.close()
        _CURRENT.reset(reset)
        with _RUNNING_LOCK:
            del _RUNNING[execution_id]


def cancel_execution(execution_id: UUID | None) -> None:
    """Signal only the matching live execution, after Task Control accepted the cancel."""
    with _RUNNING_LOCK:
        signal = _RUNNING.get(execution_id) if execution_id is not None else None
    if signal is not None:
        signal.stop(CANCELLED)


@contextmanager
def worker_cancellation(flag: str) -> Iterator[None]:
    """Honor the existing per-call flag in a kept process, with no shared cursor references."""
    signal = _Cancellation(Path(flag))
    reset = _CURRENT.set(signal)
    try:
        yield
    finally:
        _CURRENT.reset(reset)
        signal.close()


def stop_reason() -> str | None:
    """Return the accepted stop reason for the current execution, if any."""
    signal = _CURRENT.get()
    return None if signal is None else signal.reason


def interrupt_on_execution_stop(interrupt: Callable[[], object]) -> Callable[[], None]:
    """Register a resource stop and return its owner-lifetime release."""
    signal = _CURRENT.get()
    if signal is None or (registration := signal.watch(interrupt)) is None:
        return lambda: None
    return lambda: signal.unwatch(registration)


def carry_cancellation[T](call: Callable[[], T]) -> Callable[[], T]:
    """Give the Task's work-item thread the same cancellation signal."""
    signal = _CURRENT.get()

    def run() -> T:
        reset = _CURRENT.set(signal)
        try:
            return call()
        finally:
            _CURRENT.reset(reset)

    return run


@contextmanager
def interruptible(interrupt: Callable[[], object]) -> Iterator[None]:
    """Register one active raw operation for interrupt, only while it owns its handle."""
    signal = _CURRENT.get()
    if signal is None:
        yield
        return
    registration = signal.watch(interrupt)
    try:
        yield
    finally:
        if registration is not None:
            signal.unwatch(registration)
