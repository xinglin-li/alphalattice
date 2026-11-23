"""The log owner's events, dispatched in process to the subscribers declared for their subject.

An event is a row the Host's log admits (`HostLog.append` publishes its subject once the row is
written), or a change its owner keeps in memory and publishes the same way (the dispatcher's
command returning). A waiter declares itself for one subject with `subscribed`, reads the state it
waits on, and blocks until an event for that subject arrives or its deadline passes: a wait ends on
its owner's event, never on a sleep or a poll, and an event that never comes is refused by name.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from threading import Condition


class EventDeadline(TimeoutError):
    """A subscriber's wait reached its deadline with no event for its subject; its message names
    what it awaited and for how long."""


class Subscription:
    """One waiter's place on a subject: how many events have reached it."""

    def __init__(self, hub: Events) -> None:
        """Start at no events heard."""
        self._hub = hub
        self.heard = 0

    def wait(self, seen: int, *, timeout: float, what: str) -> int:
        """The count of events heard once it passes ``seen``, at once when one already has.

        Raises:
            EventDeadline: ``events.deadline``, when none arrives within ``timeout`` seconds.
            ValueError: ``events.timeout_invalid``, for a deadline that is not finite and positive.
        """
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"events.timeout_invalid: {timeout!r} s; a wait has a deadline")
        with self._hub._condition:
            if not self._hub._condition.wait_for(lambda: self.heard > seen, timeout):
                raise EventDeadline(f"events.deadline: {what} had no event within {timeout:g} s")
            return self.heard


class Events:
    """The subscribers of each subject and the condition they wait on."""

    def __init__(self) -> None:
        """No subscriber yet."""
        self._condition = Condition()
        self._subscribers: dict[Hashable, list[Subscription]] = {}

    @contextmanager
    def subscribed(self, subject: Hashable) -> Iterator[Subscription]:
        """A subscription to ``subject`` for the block: declare it before reading the state the
        wait is about, so an event between that read and the wait is still heard."""
        subscription = Subscription(self)
        with self._condition:
            self._subscribers.setdefault(subject, []).append(subscription)
        try:
            yield subscription
        finally:
            with self._condition:
                held = self._subscribers[subject]
                held.remove(subscription)
                if not held:
                    del self._subscribers[subject]

    def publish(self, subject: Hashable) -> None:
        """Tell each subscriber of ``subject`` that an event for it arrived. A subject no one
        waits on costs one dictionary read: the log's writes never wait on a follower."""
        if subject not in self._subscribers:
            return
        with self._condition:
            for subscription in self._subscribers.get(subject, ()):
                subscription.heard += 1
            self._condition.notify_all()


EVENTS = Events()
"""This process's events: the Host's log and its dispatcher publish, its waiters subscribe."""
