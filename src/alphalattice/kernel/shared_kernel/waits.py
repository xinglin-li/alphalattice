"""The one way to wait: every wait has a finite deadline on the monotonic clock.

``wait_until`` calls a check until it returns something truthy, passing it the seconds left so a
blocking call inside never outlives the wait; between checks it pauses with capped, jittered
exponential backoff, and a set stop event ends the wait at once. ``pause`` is a bounded pause that
the same stop event ends. A deadline is refused unless it is finite, positive and within its
ceiling. The commit's wait rules (devtools.architecture.waits) refuse a bare sleep elsewhere.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

CEILING = 3600.0
"""The longest any one wait may be: an hour. A longer need is a scheduled job's, never a wait's."""
clock = time.monotonic
"""The clock every deadline reads."""


class WaitTimeout(TimeoutError):
    """A wait that reached its deadline; its message names what was awaited and for how long."""


class WaitStopped(Exception):
    """A wait its stop event ended before its deadline; its message names what was awaited."""


def bound(seconds: float, *, ceiling: float = CEILING) -> float:
    """``seconds`` as a deadline's length, refused unless finite, positive and within ceiling."""
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= ceiling
    ):
        raise ValueError(
            f"wait.bound_invalid: {seconds!r} s; a wait takes a finite positive bound "
            f"within {ceiling:g} s"
        )
    return float(seconds)


def jitter(span: float) -> float:
    """Up to ``span`` seconds, so two processes polling the same thing drift apart.

    The amount is the monotonic clock's nanosecond phase, which differs between processes.
    """
    return span * (time.monotonic_ns() % 1_000_003) / 1_000_003


def pause(seconds: float, *, stop: threading.Event | None = None) -> bool:
    """Pause ``seconds`` (bounded as any wait), or less if ``stop`` is set; True when it was."""
    seconds = bound(seconds)
    if stop is not None:
        return stop.wait(seconds)
    time.sleep(seconds)
    return False


def wait_until[T](
    check: Callable[[float], T],
    *,
    timeout: float,
    what: str,
    interval: float = 0.05,
    max_interval: float = 2.0,
    stop: threading.Event | None = None,
    ceiling: float = CEILING,
) -> T:
    """The first truthy result of ``check(seconds_left)``, until a deadline or a stop.

    The check is called until ``timeout`` seconds pass (WaitTimeout) or ``stop`` is set
    (WaitStopped). Pauses start at ``interval``, double up to ``max_interval``, carry up to
    ``interval`` of jitter, and never run past the deadline.
    """
    timeout = bound(timeout, ceiling=ceiling)
    interval, max_interval = bound(interval), bound(max_interval)
    deadline, delay = clock() + timeout, interval
    while True:
        if stop is not None and stop.is_set():
            raise WaitStopped(f"wait.stopped: {what} was stopped")
        left = deadline - clock()
        if left <= 0:
            raise WaitTimeout(f"wait.timeout: {what} not reached within {timeout:g} s")
        if result := check(left):
            return result
        left = deadline - clock()
        if left <= 0:
            raise WaitTimeout(f"wait.timeout: {what} not reached within {timeout:g} s")
        if pause(min(delay + jitter(interval), left), stop=stop):
            raise WaitStopped(f"wait.stopped: {what} was stopped")
        delay = min(delay * 2, max_interval)
