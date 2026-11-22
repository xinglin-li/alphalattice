"""Persistent storage sessions and objects."""

from typing import Protocol


class StoragePort(Protocol):
    """The kernel port for persistent storage sessions and objects."""
