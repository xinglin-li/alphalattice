"""Wall time and calendar-date access."""

from typing import Protocol


class ClockPort(Protocol):
    """The kernel port for wall time and calendar-date access."""
