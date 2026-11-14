"""Bounded waiting: every wait has a finite deadline on the monotonic clock.

`wait_until` calls a check until it returns something truthy, passing it the seconds left so a
blocking call inside never outlives the wait; between checks it pauses with capped exponential
backoff, and a set stop event ends the wait at once. `pause` is a bounded pause that
the same stop event ends. A bound is refused unless it is finite, positive and within its ceiling.
Standard library only, so it runs alike on macOS, Linux and Windows.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable

CEILING = 3600.0
"""The longest any one wait may be: an hour. A longer need is a scheduled job's, never a wait's."""


class WaitTimeout(TimeoutError):
    """A wait that reached its deadline: what it waited for and for how long."""

    code = "wait.timeout"

    def __init__(self, what: str, seconds: float) -> None:
        """Name what was not reached and the bound it had."""
        super().__init__(f"wait.timeout: {what} not reached within {seconds:g} s")
        self.what, self.seconds = what, seconds


class WaitStopped(Exception):
    """A wait its stop event ended before its deadline."""

    code = "wait.stopped"

    def __init__(self, what: str) -> None:
        """Name the wait its stop event ended."""
        super().__init__(f"wait.stopped: {what} was stopped")
        self.what = what


def bound(seconds: float, *, ceiling: float = CEILING) -> float:
    """Return `seconds` as a wait's length.

    Args:
        seconds: The wait's length.
        ceiling: The longest length admitted.

    Returns:
        The length as a float.

    Raises:
        ValueError: `wait.bound_invalid`, unless the length is finite, positive and at most
            `ceiling`.
    """
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


def pause(seconds: float, *, stop: threading.Event | None = None) -> bool:
    """Pause `seconds`, bounded as any wait, or less if `stop` is set.

    Args:
        seconds: The pause's length.
        stop: An event that ends the pause early.

    Returns:
        True when `stop` ended the pause.
    """
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
    """Return the first truthy result of `check(seconds_left)`.

    Pauses between checks start at `interval`, double up to `max_interval` and never run past
    the deadline; `check` is given the seconds left so a blocking call inside it never outlives
    the wait.

    Args:
        check: Called with the seconds left until it returns something truthy.
        timeout: The wait's bound, in seconds.
        what: What the wait is for, as its errors name it.
        interval: The first pause.
        max_interval: The longest pause.
        stop: An event that ends the wait.
        ceiling: The longest bound admitted.

    Returns:
        The check's first truthy result.

    Raises:
        WaitTimeout: The bound passed first.
        WaitStopped: `stop` was set first.
    """
    timeout = bound(timeout, ceiling=ceiling)
    interval, max_interval = bound(interval), bound(max_interval)
    deadline, delay = time.monotonic() + timeout, interval
    while True:
        if stop is not None and stop.is_set():
            raise WaitStopped(what)
        left = deadline - time.monotonic()
        if left <= 0:
            raise WaitTimeout(what, timeout)
        if result := check(left):
            return result
        left = deadline - time.monotonic()
        if left <= 0:
            raise WaitTimeout(what, timeout)
        if pause(min(delay, left), stop=stop):
            raise WaitStopped(what)
        delay = min(delay * 2, max_interval)


__all__ = ["CEILING", "WaitStopped", "WaitTimeout", "bound", "pause", "wait_until"]
