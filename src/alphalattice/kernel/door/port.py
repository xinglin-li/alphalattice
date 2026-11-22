"""Operation admission and dispatch boundary."""

from typing import Protocol


class DoorPort(Protocol):
    """The kernel port for operation admission and dispatch boundary."""
