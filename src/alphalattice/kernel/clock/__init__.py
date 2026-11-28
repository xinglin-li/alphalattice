"""Wall time and replay correlation for the product."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from threading import Lock
from time import get_clock_info, monotonic, time, time_ns
from typing import Any
from uuid import uuid4

from .port import ClockPort

__all__ = [
    "ClockPort",
    "IncreasingClock",
    "ReplayClock",
    "carry_observation",
    "clock_metadata",
    "hold_writer_clock",
    "observation_time",
    "task_observation",
    "utc_now",
    "wall_time",
    "wall_time_ns",
    "writing",
]


_OBSERVED: ContextVar[datetime | None] = ContextVar("task_observed_at", default=None)
_TICK = timedelta(microseconds=1)
KEEP_AHEAD = timedelta(milliseconds=1)
"""How far past the last time handed out a writer's kept bound may lie: under one wall-clock tick,
and one persisted bound for each millisecond of times handed out at most."""


def _wall() -> datetime:
    return datetime.now(UTC)


class IncreasingClock:
    """Strictly increasing times over a source clock: each is the larger of the source's reading
    and the last time handed out plus one microsecond, so two writes never tie and a clock stepped
    back never inverts their order; a reading past the last is the source's own, true to its tick.

    `seed` is a bound at or above every time an earlier owner of the same data root handed out, and
    `keep` persists such a bound (at most `KEEP_AHEAD` past the last time) before a time beyond the
    last kept bound is handed out, so the next owner's seed holds it after a crash too.
    """

    def __init__(
        self,
        source: ClockPort = _wall,
        *,
        seed: datetime | None = None,
        keep: Callable[[datetime], None] | None = None,
    ) -> None:
        """Wrap `source`, starting above `seed` when one is given."""
        self._source: ClockPort = source
        self._last: datetime | None = seed
        self._kept: datetime | None = seed
        self._keep: Callable[[datetime], None] | None = keep
        self._lock = Lock()

    def __call__(self) -> datetime:
        """Hand out the next time."""
        with self._lock:
            now: datetime = self._source()
            if self._last is not None and now <= self._last:
                now = self._last + _TICK
            self._last = now
            if self._keep is not None and (self._kept is None or now > self._kept):
                bound = now + KEEP_AHEAD
                self._keep(bound)
                self._kept = bound
            return now

    def end(self) -> None:
        """Stop keeping bounds: no `keep` runs after this returns."""
        with self._lock:
            self._keep = None


_HELD: list[IncreasingClock] = []
_held = Lock()
_WRITING: ContextVar[IncreasingClock | None] = ContextVar("writer_clock", default=None)


def hold_writer_clock(
    seed: datetime | None, keep: Callable[[datetime], None]
) -> tuple[IncreasingClock, Callable[[], None]]:
    """One data root's increasing clock while its writer holds the root, seeded and kept by that
    writer, with the end (once) that stops it keeping and lets it go. Each root keeps its own."""
    owner = IncreasingClock(_wall, seed=seed, keep=keep)
    with _held:
        _HELD.append(owner)

    def end() -> None:
        owner.end()
        with _held:
            if owner in _HELD:
                _HELD.remove(owner)

    return owner, end


@contextmanager
def writing(owner: IncreasingClock) -> Iterator[None]:
    """`utc_now` answers with the clock of the root these writes go to."""
    token = _WRITING.set(owner)
    try:
        yield
    finally:
        _WRITING.reset(token)


def utc_now() -> datetime:
    """Read aware UTC wall time at an operational boundary: the clock of the root being written
    (`writing`), else the first root its process holds, a Host's own workspace; so every time a
    root's writes carry is distinct, in write order and above that root's earlier times."""
    owner = _WRITING.get()
    if owner is None:
        with _held:
            owner = _HELD[0] if _HELD else None
    return _wall() if owner is None else owner()


def wall_time() -> float:
    """Read Unix wall time in seconds."""
    return time()


def wall_time_ns() -> int:
    """Read Unix wall time in nanoseconds."""
    return time_ns()


def observation_time() -> datetime:
    """Use the task's sampled time for business observations, or sample at this boundary."""
    return _OBSERVED.get() or utc_now()


@contextmanager
def task_observation(observed_at: datetime | None) -> Iterator[None]:
    """Carry the task's start timestamp through its business observations."""
    token = _OBSERVED.set(observed_at)
    try:
        yield
    finally:
        _OBSERVED.reset(token)


def carry_observation[T](call: Callable[[], T]) -> Callable[[], T]:
    """Carry only the task's observation across a worker-thread boundary."""
    observed_at = _OBSERVED.get()

    def run() -> T:
        with task_observation(observed_at):
            return call()

    return run


class ReplayClock:
    """Advance a declared anchor while retaining the uncertainty of its UTC correlation."""

    def __init__(
        self,
        anchor: datetime,
        *,
        ticks: Callable[[], float] = monotonic,
        wall: ClockPort = utc_now,
    ) -> None:
        """Sample one declared anchor and its bracketed process-clock correlation."""
        if anchor.utcoffset() is None:
            raise ValueError("clock.anchor_naive")
        self.anchor, self.reference = anchor.astimezone(UTC), uuid4().hex
        self._ticks, self._wall = ticks, wall
        self._resolution = {
            "time": get_clock_info("time").resolution,
            "monotonic": get_clock_info("monotonic").resolution,
        }
        self._precision = sum(self._resolution.values()) + datetime.resolution.total_seconds()
        lo, self._origin, hi = self._sample()
        self._tick = (lo + hi) / 2
        self._increasing = IncreasingClock(self._elapsed)
        self._uncertainty = (hi - lo) / 2 + self._precision
        self.correlation = dict(
            schema="alphalattice.clock-correlation.v1",
            clock_ref=self.reference,
            replay_anchor=self.anchor.isoformat(),
            system_anchor=self._origin.isoformat(),
            monotonic_anchor=self._tick,
            uncertainty_seconds=self._uncertainty,
            resolution_seconds=self._resolution,
        )

    def _sample(self) -> tuple[float, datetime, float]:
        return self._ticks(), self._wall(), self._ticks()

    def _elapsed(self) -> datetime:
        return self.anchor + timedelta(seconds=self._ticks() - self._tick)

    def __call__(self) -> datetime:
        """Advance replay time solely by monotonic elapsed, increasing across reads in one tick."""
        return self._increasing()

    def phase_metadata(self) -> dict[str, Any]:
        """Bind a phase or RPC and report drift beyond measured sampling uncertainty."""
        lo, wall, hi = self._sample()
        drift = (wall - self._origin).total_seconds() - ((lo + hi) / 2 - self._tick)
        bound = self._uncertainty + (hi - lo) / 2 + self._precision
        return dict(clock_ref=self.reference, clock_drift=abs(drift) > bound)


def clock_metadata(clock: object) -> dict[str, Any]:
    """Optional timing metadata cannot change a Task's disposition or overwrite its fields."""
    try:
        match getattr(clock, "phase_metadata", lambda: {})():
            case {"clock_ref": str() as reference, "clock_drift": bool() as drift}:
                return {"clock_ref": reference, "clock_drift": drift}
        if getattr(clock, "clock_domain", None) == "UTC":
            return {"clock_domain": "UTC"}
    except Exception:
        pass
    return {}
