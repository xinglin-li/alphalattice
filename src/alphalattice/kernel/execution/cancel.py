"""Exact execution cancellation and its per-operation interrupt observers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from pathlib import Path
from threading import Condition, Event, Lock, Thread, Timer
from time import monotonic
from typing import Protocol
from uuid import UUID

from .port import PublicationCheck

CANCELLED = "execution.cancelled"
TIMED_OUT = "execution.timeout"
_INTERRUPT_INTERVAL = 0.025
_BELL_BOUND = 60.0
"""How long a kept call's bell listener waits between looks at whether its call has ended; a
ring (the parent's cancel, the call's own end) wakes it at once."""


class ExecutionStopped(RuntimeError):
    """The current execution reached its accepted cancellation or deadline."""


class Bell(Protocol):
    """A kept worker's doorbell, shared with its parent: a process-wide event either side sets."""

    def set(self) -> None:
        """Ring it: the listener looks at its call's flag."""
        ...

    def clear(self) -> None:
        """Quiet it before a look, so a ring after the look is still heard."""
        ...

    def wait(self, timeout: float | None = None) -> bool:
        """Block until it rings or ``timeout`` passes; whether it rang."""
        ...


class _Cancellation:
    def __init__(self, publication_check: PublicationCheck | None = None) -> None:
        self.reason: str | None = None
        self._publication_check = publication_check
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
            self._deadline = monotonic() + 1.0
            self.reason = reason

    def _start(self) -> None:
        if self._watcher is None:
            self._watcher = Thread(target=self._watch, name="execution-interrupt", daemon=True)
            self._watcher.start()

    def _watch(self) -> None:
        """After the stop, interrupt each registered operation until none is left or its second
        passes: keep attempting only while the same operation holds its registration, as an
        interrupt observed just before engine entry can otherwise be lost."""
        with self._condition:
            try:
                while not self._closed and self._interrupts and monotonic() < self._deadline:
                    for interrupt in self._interrupts.values():
                        with suppress(Exception):
                            interrupt()
                    self._condition.wait(_INTERRUPT_INTERVAL)
            finally:
                self._watcher = None

    def watch(self, interrupt: Callable[[], object]) -> object | None:
        with self._condition:
            if self.reason == TIMED_OUT:
                raise ExecutionStopped(TIMED_OUT)
            if self._closed or self.reason is not None:
                # A cancelled Task can finish its current chunk's writes before seeing the stop.
                return None
            registration = object()
            self._interrupts[registration] = interrupt
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
def execution_cancellation(
    execution_id: UUID,
    timeout: float | None = None,
    *,
    publication_check: PublicationCheck | None = None,
) -> Iterator[None]:
    """Bind the execution's signal for its lifetime, including its nested thread work."""
    signal = _Cancellation(publication_check)
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
def worker_cancellation(flag: str, bell: Bell) -> Iterator[None]:
    """Honor the per-call flag in a kept process, with no shared cursor references.

    The parent touches the flag and rings the worker's bell; the call stops when the bell rings
    with its own flag set. A ring for an earlier call, or this call's own end, only makes the
    listener look again.
    """
    signal = _Cancellation()
    reset = _CURRENT.set(signal)
    ended = Event()
    listener = Thread(
        target=_listen, args=(signal, Path(flag), bell, ended), name="execution-bell", daemon=True
    )
    listener.start()
    try:
        yield
    finally:
        ended.set()
        bell.set()
        listener.join(timeout=1.0)
        _CURRENT.reset(reset)
        signal.close()


def _listen(signal: _Cancellation, flag: Path, bell: Bell, ended: Event) -> None:
    """Stop the call once its flag is set, looking at each ring of its worker's bell. The bell is
    cleared before each look at the call's end and its flag, and the parent and the teardown each
    write before they ring, so a ring after the clear is heard by the wait and none is lost."""
    bell.clear()
    while not ended.is_set():
        if flag.exists():
            signal.stop(CANCELLED)
            break
        bell.wait(_BELL_BOUND)
        bell.clear()


def stop_reason() -> str | None:
    """Return the accepted stop reason for the current execution, if any."""
    signal = _CURRENT.get()
    return None if signal is None else signal.reason


def stop_deadline() -> float | None:
    """The absolute settlement deadline of this execution's accepted stop."""
    signal = _CURRENT.get()
    return signal._deadline if signal is not None and signal.reason is not None else None


@contextmanager
def execution_publication() -> Iterator[None]:
    """Order an irreversible publication against acceptance of this execution's stop."""
    signal = _CURRENT.get()
    if signal is None:
        yield
        return
    check = signal._publication_check
    guard = check() if check is not None else nullcontext(False)
    # Task Control takes the durable authority before it signals the execution.
    with guard as cancelled, signal._condition:
        if cancelled:
            signal.stop(CANCELLED)
        if signal.reason is not None:
            raise ExecutionStopped(signal.reason)
        yield


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
