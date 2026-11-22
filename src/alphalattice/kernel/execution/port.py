"""Task execution, scheduling and cancellation."""

from typing import Protocol


class ExecutionPort(Protocol):
    """The kernel port for task execution, scheduling and cancellation."""
