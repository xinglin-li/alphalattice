"""Wall time and replay correlation for the product."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from time import get_clock_info, monotonic, time, time_ns
from typing import Any
from uuid import uuid4

from .port import ClockPort

__all__ = [
    "ClockPort",
    "ReplayClock",
    "carry_observation",
    "clock_metadata",
    "observation_time",
    "task_observation",
    "utc_now",
    "wall_time",
    "wall_time_ns",
]


_OBSERVED: ContextVar[datetime | None] = ContextVar("task_observed_at", default=None)


def utc_now() -> datetime:
    """Read aware UTC wall time at an operational boundary."""
    return datetime.now(UTC)


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

    def __call__(self) -> datetime:
        """Advance replay time solely by monotonic elapsed."""
        return self.anchor + timedelta(seconds=self._ticks() - self._tick)

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
