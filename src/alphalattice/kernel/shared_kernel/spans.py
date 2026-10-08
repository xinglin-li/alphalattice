"""Where a long stage spends its time: execution spans, never evidence or identity (A4).

An owner marks a category of its work with ``with span("verify"):`` or ``@spanned("fit")``.
The Task runner opens a ledger around each stage's execution and its verification
(``collect``), and every span the stage's threads close inside it is added up by category:
how many, and their inclusive and exclusive seconds (a span's time less the spans it holds,
on its own thread). A Task's call in a worker process is collected there and its readout
travels back with the call's answer (``absorb``), so a stage's ledger covers its workers too.
The runner keeps each readout beside the Task's records; ``task show --section timing``
reads them back with the stage they belong to.

A span measures and never decides. It never suppresses an exception, reads nothing a result
depends on and changes no value; the rule closure hashes ``with span(...)`` as its body and
leaves out ``@spanned(...)`` and their import (``source_identity``), so adding or moving a
span moves no identity (LAWS ID3). Outside a ledger a span costs one context-variable read.
Spans go around units of work -- a fit, a verification, a publication -- never inside a
per-row loop.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final, ParamSpec, TypeVar

SPAN_CLASSES: Final[Mapping[str, str]] = {
    "materialize": "compute",
    "features": "compute",
    "fit": "compute",
    "predict": "compute",
    "score": "compute",
    "compute": "compute",
    "read": "io",
    "write": "io",
    "serialize": "io",
    "copy": "io",
    "verify": "verify",
    "hash": "verify",
    "network": "network",
    "wait": "wait",
}
"""Each span category and the class of time a ledger counts it under.

``materialize`` builds a stage's input arrays from sealed sources; ``features`` computes
Feature values; ``fit``, ``predict`` and ``score`` are model work; ``read`` and ``write`` move
bytes to and from storage, ``serialize`` turns values into bytes or back and ``copy``
duplicates stored bytes; ``verify`` re-checks sealed content and ``hash`` digests it;
``network`` waits on a provider and ``wait`` on another process or a lock."""

SPAN_READOUT_SCHEMA: Final = "alphalattice.stage-spans.v1"
MAXIMUM_SPAN_KEYS: Final = 256
"""Distinct (category, detail, origin) rows one ledger keeps; later new rows are counted only."""
MAXIMUM_DETAIL_CHARACTERS: Final = 64

_P = ParamSpec("_P")
_R = TypeVar("_R")

_ACTIVE: ContextVar[SpanLedger | None] = ContextVar("alphalattice_span_ledger", default=None)
_OPEN: list[SpanLedger] = []
"""The ledgers open in this process. A thread an owner starts does not inherit the context
variable, so a span there counts toward the one open ledger, and toward none when several are."""
_OPEN_LOCK = threading.Lock()


class SpanLedger:
    """The spans of one stage phase, added up by category, detail and origin."""

    def __init__(self) -> None:
        """Start an empty ledger; ``collect`` opens it and reads its resources."""
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str | None, str], list[float]] = {}
        self._dropped = 0
        self._local = threading.local()
        self._workers = {"calls": 0, "wall_seconds": 0.0, "cpu_seconds": 0.0}
        self._worker_bytes = {"read_bytes": 0, "write_bytes": 0}
        self.overlapped = False
        self.readout_value: dict[str, Any] | None = None

    def _stack(self) -> list[float]:
        stack: list[float] | None = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        return stack

    def add(
        self,
        category: str,
        detail: str | None,
        *,
        origin: str,
        count: int,
        inclusive: float,
        exclusive: float,
    ) -> None:
        """Add closed spans to one row; a row beyond the bound is counted, never kept."""
        key = (category, detail, origin)
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                if len(self._rows) >= MAXIMUM_SPAN_KEYS:
                    self._dropped += count
                    return
                row = self._rows[key] = [0, 0.0, 0.0]
            row[0] += count
            row[1] += inclusive
            row[2] += exclusive

    def absorb(self, readout: Mapping[str, Any]) -> None:
        """Add a worker call's readout: its spans as the worker's, its resources as workers'."""
        for row in readout.get("spans", ()):
            self.add(
                str(row["category"]),
                row.get("detail"),
                origin="worker",
                count=int(row["count"]),
                inclusive=float(row["inclusive_seconds"]),
                exclusive=float(row["exclusive_seconds"]),
            )
        process = readout.get("process", {})
        workers = readout.get("workers", {})
        with self._lock:
            self._dropped += int(readout.get("dropped_spans", 0))
            self._workers["calls"] += 1 + int(workers.get("calls", 0))
            self._workers["wall_seconds"] += float(readout.get("wall_seconds", 0.0))
            self._workers["cpu_seconds"] += float(process.get("cpu_seconds", 0.0)) + float(
                workers.get("cpu_seconds", 0.0)
            )
            for name in ("read_bytes", "write_bytes"):
                self._worker_bytes[name] += int(process.get(name, 0)) + int(workers.get(name, 0))

    def rows(self) -> list[dict[str, Any]]:
        """The kept rows, largest exclusive time first."""
        with self._lock:
            items = sorted(self._rows.items(), key=lambda item: -item[1][2])
        return [
            {
                "category": category,
                "detail": detail,
                "origin": origin,
                "count": int(count),
                "inclusive_seconds": round(inclusive, 6),
                "exclusive_seconds": round(exclusive, 6),
            }
            for (category, detail, origin), (count, inclusive, exclusive) in items
        ]


class span:
    """Count the block's time under ``category`` (and ``detail``) in the open ledger."""

    __slots__ = ("_category", "_detail", "_ledger", "_started")

    def __init__(self, category: str, detail: str | None = None) -> None:
        """Name the work; ``category`` is one of ``SPAN_CLASSES``, ``detail`` a short label."""
        self._category = category
        self._detail = None if detail is None else detail[:MAXIMUM_DETAIL_CHARACTERS]
        self._ledger: SpanLedger | None = None
        self._started = 0.0

    def __enter__(self) -> None:
        """Start the block's clock when a ledger is open; nothing otherwise."""
        ledger = _ACTIVE.get() or _sole_open()
        self._ledger = ledger
        if ledger is not None:
            ledger._stack().append(0.0)
            self._started = time.perf_counter()

    def __exit__(self, *_exc: object) -> None:
        """Add the block's time to its row; returns None, so every exception goes through."""
        ledger = self._ledger
        if ledger is not None:
            elapsed = time.perf_counter() - self._started
            stack = ledger._stack()
            held = stack.pop() if stack else 0.0
            if stack:
                stack[-1] += elapsed
            ledger.add(
                self._category,
                self._detail,
                origin="host",
                count=1,
                inclusive=elapsed,
                exclusive=max(0.0, elapsed - held),
            )


def spanned(
    category: str, detail: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Count each call of the decorated function as one ``span(category, detail)``."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(function)
        def measured(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with span(category, detail):
                return function(*args, **kwargs)

        return measured

    return decorate


def _sole_open() -> SpanLedger | None:
    with _OPEN_LOCK:
        return _OPEN[0] if len(_OPEN) == 1 else None


def active_ledger() -> SpanLedger | None:
    """The ledger this thread's spans count toward, if any."""
    return _ACTIVE.get() or _sole_open()


def absorb(readout: Mapping[str, Any] | None) -> None:
    """Add a worker call's readout to the open ledger; nothing without one or a readout."""
    ledger = active_ledger()
    if ledger is not None and readout:
        ledger.absorb(readout)


def _resources() -> tuple[float, int, int]:
    times = os.times()
    try:
        import psutil  # type: ignore[import-untyped]

        counters = psutil.Process().io_counters()
        return float(times.user + times.system), int(counters.read_bytes), int(counters.write_bytes)
    except (ImportError, OSError, AttributeError):  # pragma: no cover - platform without counters
        return float(times.user + times.system), 0, 0


@contextmanager
def collect() -> Iterator[SpanLedger]:
    """Open a ledger for the block; ``readout`` reads it, with the process's own resources.

    The process's CPU seconds and read/write bytes are deltas over the block. They are the
    whole process's, so a ledger that ``overlapped`` another open one shares them with it.
    """
    ledger = SpanLedger()
    started_at = datetime.now(UTC)
    cpu, read, written = _resources()
    started = time.perf_counter()
    with _OPEN_LOCK:
        if _OPEN:
            ledger.overlapped = True
            for other in _OPEN:
                other.overlapped = True
        _OPEN.append(ledger)
    token = _ACTIVE.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE.reset(token)
        with _OPEN_LOCK:
            _OPEN.remove(ledger)
        wall = time.perf_counter() - started
        cpu_after, read_after, written_after = _resources()
        ledger.readout_value = {
            "schema": SPAN_READOUT_SCHEMA,
            "started_at": started_at.isoformat(),
            "wall_seconds": round(wall, 6),
            "process": {
                "pid": os.getpid(),
                "cpu_seconds": round(max(0.0, cpu_after - cpu), 6),
                "read_bytes": max(0, read_after - read),
                "write_bytes": max(0, written_after - written),
                "overlapped": ledger.overlapped,
            },
            "workers": {
                **{key: round(value, 6) for key, value in ledger._workers.items()},
                **ledger._worker_bytes,
            },
            "spans": ledger.rows(),
            "dropped_spans": ledger._dropped,
        }


def readout(ledger: SpanLedger) -> dict[str, Any]:
    """What a closed ledger read: its wall, resources and span rows."""
    value = ledger.readout_value
    if value is None:
        # A caller error, never a refusal a person or an agent meets: the ledger is still open.
        raise RuntimeError("a span ledger is read only after its collect() block ends")
    return value


__all__ = [
    "MAXIMUM_SPAN_KEYS",
    "SPAN_CLASSES",
    "SPAN_READOUT_SCHEMA",
    "SpanLedger",
    "absorb",
    "active_ledger",
    "collect",
    "readout",
    "span",
    "spanned",
]
