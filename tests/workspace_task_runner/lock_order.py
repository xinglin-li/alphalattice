"""A lock-order recorder for the product's stores (RH2).

A unit that holds one of the product's locks and asks for another while a second thread
holds them the other way round waits until a deadline expires: the Task registry's status
read did so with a starting Task (`31c49cf3`). The recorder names every pair of locks a
test took in both orders, whichever threads took them, so the order is caught when it is
written rather than when two threads happen to meet.

The locks it sees: each workspace mutation gate (`gate`), each Task registry's connection
lock (`registry`), and each workspace database file (`instance:<file>`). A file is a lock only
the way the database owner makes it one: a thread that retains it read-only holds it, and a
thread that asks for it read-write waits for every such holder; readers attach to a writer's
instance, and writers to each other's, without waiting. A lock created before
`record_lock_order` starts is not seen; the fixture that uses it starts it before the owners
it watches are built.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.task_control import registry as task_registry
from alphalattice.control.workspace_runtime import database, mutation_gate


@dataclass
class LockOrder:
    """The ordered pairs of locks taken so far, each with where it was first taken."""

    edges: dict[tuple[str, str], str] = field(default_factory=dict)
    _held: threading.local = field(default_factory=threading.local)

    def _stack(self) -> list[str]:
        stack: list[str] | None = getattr(self._held, "stack", None)
        if stack is None:
            stack = []
            self._held.stack = stack
        return stack

    def held_now(self) -> list[str]:
        """The locks the calling thread holds: those it took and the files it retains read-only."""

        me = threading.get_ident()
        with database._STATE:
            retained = [
                f"instance:{path.name}"
                for (path, owner), retention in database._RETENTIONS.items()
                if owner == me and retention.root._instance.read_only
            ]
        return [*self._stack(), *retained]

    def taking(self, name: str) -> None:
        """Record that the calling thread asks for `name` while holding what it holds."""

        for outer in self.held_now():
            if outer != name and (outer, name) not in self.edges:
                self.edges[(outer, name)] = "".join(traceback.format_stack(limit=12)[:-2])

    def took(self, name: str) -> None:
        self._stack().append(name)

    def released(self, name: str) -> None:
        stack = self._stack()
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] == name:
                del stack[index]
                return

    def inversions(self) -> list[tuple[str, str]]:
        """Every pair of locks taken in both orders, each named once."""

        return sorted((a, b) for a, b in self.edges if a < b and (b, a) in self.edges)

    def report(self) -> str:
        return "\n\n".join(
            f"{a} then {b}:\n{self.edges[(a, b)]}\n{b} then {a}:\n{self.edges[(b, a)]}"
            for a, b in self.inversions()
        )


class _RecordedRLock:
    """An `RLock` that tells the recorder when it is asked for, held and let go."""

    def __init__(self, recorder: LockOrder, name: str) -> None:
        self._lock = threading.RLock()
        self._recorder = recorder
        self._name = name

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking:
            self._recorder.taking(self._name)
        acquired = self._lock.acquire(blocking, timeout)
        if acquired:
            self._recorder.took(self._name)
        return acquired

    def release(self) -> None:
        self._lock.release()
        self._recorder.released(self._name)

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.release()


@contextmanager
def record_lock_order(monkeypatch: pytest.MonkeyPatch) -> Iterator[LockOrder]:
    """Watch the gates, registry locks and database instances created from here on."""

    recorder = LockOrder()
    monkeypatch.setattr(mutation_gate, "RLock", lambda: _RecordedRLock(recorder, "gate"))
    monkeypatch.setattr(task_registry, "RLock", lambda: _RecordedRLock(recorder, "registry"))
    monkeypatch.setattr(task_registry, "_CONNECTION_LOCKS", type(task_registry._CONNECTION_LOCKS)())
    acquire = database._acquire

    def recorded_acquire(path: Path, *, read_only: bool, wait_seconds: float) -> Any:
        if not read_only:  # a writer waits for the file's read-only holders
            recorder.taking(f"instance:{path.name}")
        return acquire(path, read_only=read_only, wait_seconds=wait_seconds)

    monkeypatch.setattr(database, "_acquire", recorded_acquire)
    yield recorder


@pytest.fixture
def lock_order(monkeypatch: pytest.MonkeyPatch) -> Iterator[LockOrder]:
    """Fail the test if it took two of the product's locks in both orders."""

    with record_lock_order(monkeypatch) as recorder:
        yield recorder
    assert not recorder.inversions(), recorder.report()
