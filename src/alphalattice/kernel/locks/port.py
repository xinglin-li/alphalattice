"""Lock and lease primitives."""

from typing import Protocol


class LocksPort(Protocol):
    """The kernel port for lock and lease primitives."""
