"""Replay time and its sampled correlation to the process clocks."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any
from uuid import uuid4


class ReplayClock:
    """Advance a declared anchor while retaining the uncertainty of its UTC correlation."""

    def __init__(
        self,
        anchor: datetime,
        *,
        ticks: Callable[[], float] = monotonic,
        wall: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Sample one declared anchor and its bracketed process-clock correlation."""
        if anchor.utcoffset() is None:
            raise ValueError("clock.anchor_naive")
        self.anchor, self.reference = anchor.astimezone(UTC), uuid4().hex
        self._ticks, self._wall = ticks, wall
        lo, self._origin, hi = self._sample()
        self._tick = (lo + hi) / 2
        self._uncertainty = (hi - lo) / 2 + datetime.resolution.total_seconds()
        self.correlation = dict(
            schema="alphalattice.clock-correlation.v1",
            clock_ref=self.reference,
            replay_anchor=self.anchor.isoformat(),
            system_anchor=self._origin.isoformat(),
            monotonic_anchor=self._tick,
            uncertainty_seconds=self._uncertainty,
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
        bound = self._uncertainty + (hi - lo) / 2 + datetime.resolution.total_seconds()
        return dict(clock_ref=self.reference, clock_drift=abs(drift) > bound)


def clock_metadata(clock: object) -> dict[str, Any]:
    """Optional timing metadata cannot change a Task's disposition or overwrite its fields."""
    try:
        match getattr(clock, "phase_metadata", lambda: {})():
            case {"clock_ref": str() as reference, "clock_drift": bool() as drift}:
                return {"clock_ref": reference, "clock_drift": drift}
    except Exception:
        pass
    return {}
