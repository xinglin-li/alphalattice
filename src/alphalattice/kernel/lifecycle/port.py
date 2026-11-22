"""Host lifetime and supervision."""

from typing import Protocol


class LifecyclePort(Protocol):
    """The kernel port for host lifetime and supervision."""
