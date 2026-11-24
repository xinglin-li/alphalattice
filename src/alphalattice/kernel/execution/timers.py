"""One timer: every periodic job's next due time, and the wake at it. A Host owns one for its
jobs and closes it at its stop and on a failed start; `TIMERS` serves jobs no Host owns.

A job is declared once (`Timers.job`) with its clock owner, a timezone-aware wall clock, and the
call that wakes its worker. Its owner arms it with the next due time it computed; one thread waits
for the earliest due time among the armed jobs and fires each job once per arming. The job's own
worker then runs it, so a long run never delays another job's wake.

The thread trusts the monotonic clock for at most `WALL_RECHECK_SECONDS` before it reads the wall
clocks again, since a monotonic wait may not count the time a machine sleeps: a Host asleep past a
due time fires that job once at wake, never once for each time it missed. That is the longest it
ever blocks, armed or not.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from threading import Condition, Event, Thread

WALL_RECHECK_SECONDS = 60.0
"""The longest the timer waits before it reads the wall clocks again."""


class Timer:
    """One declared job's wake: its next due time while armed, None once fired or disarmed."""

    def __init__(
        self,
        owner: Timers,
        name: str,
        clock: Callable[[], datetime],
        fire: Callable[[], None],
    ) -> None:
        """Declared by `Timers.job`."""
        self.name, self.clock, self.fire = name, clock, fire
        self._owner = owner
        self.due: datetime | None = None

    def arm(self, at: datetime | None) -> None:
        """Fire once at ``at`` (at once when it has passed); None disarms. Each arming fires once.

        Raises:
            ValueError: ``timers.due_naive``, for a due time without a timezone.
        """
        if at is not None and at.tzinfo is None:
            raise ValueError(f"timers.due_naive: {self.name} was armed with a naive time")
        self._owner._arm(self, at)


class Timers:
    """The armed jobs and the one thread that wakes each at its due time."""

    def __init__(self, *, recheck_seconds: float = WALL_RECHECK_SECONDS) -> None:
        """No job yet; the thread starts with the first arming."""
        self._condition = Condition()
        self._armed: set[Timer] = set()
        self._recheck = recheck_seconds
        self._thread: Thread | None = None
        self._stop = Event()

    def job(self, name: str, clock: Callable[[], datetime], fire: Callable[[], None]) -> Timer:
        """Declare a job: ``clock`` is its clock owner, ``fire`` wakes its worker."""
        return Timer(self, name, clock, fire)

    def close(self, *, timeout: float | None = 5.0) -> bool:
        """Stop the thread; armed jobs no longer fire, and a later arming starts none. Whether it
        stopped within ``timeout`` (None: until it has): a fire in hand (a job's wake call)
        finishes first. Closing again answers the same."""
        with self._condition:
            self._stop.set()
            self._armed.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)
            return not thread.is_alive()
        return True

    def _arm(self, timer: Timer, at: datetime | None) -> None:
        with self._condition:
            timer.due = at
            if at is None:
                self._armed.discard(timer)
            else:
                self._armed.add(timer)
                if self._thread is None and not self._stop.is_set():
                    thread = Thread(target=self._run, name="timers", daemon=True)
                    # Published once started: a start the system refuses raises here and leaves
                    # no thread for `close` to join, so the caller's own cleanup still runs.
                    thread.start()
                    self._thread = thread
            self._condition.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                if self._stop.is_set():  # under the lock: an arming after close fires nothing
                    return
                due, left = [], self._recheck
                for timer in self._armed:
                    if timer.due is None:  # armed jobs always carry one; read once, typed
                        continue
                    seconds = (timer.due - timer.clock()).total_seconds()
                    if seconds <= 0:
                        due.append(timer)
                    else:
                        left = min(left, seconds)
                for timer in due:
                    self._armed.discard(timer)
                    timer.due = None
                if not due:
                    # Until an arming, the earliest due time or the wall clocks' recheck.
                    self._condition.wait(left)
                    continue
            for timer in due:
                timer.fire()


TIMERS = Timers()
"""The timer of jobs no Host owns. A Host owns its own, which its stop and every failed start
close: its maintenance, update and supervision jobs arm that one."""
