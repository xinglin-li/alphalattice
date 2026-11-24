"""Wall time and calendar-date access."""

from datetime import datetime
from typing import Protocol


class ClockPort(Protocol):
    """A callable that supplies an aware wall-clock observation."""

    def __call__(self) -> datetime:
        """Return the current aware wall-clock observation."""
        ...
