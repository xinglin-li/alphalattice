"""Ordered durable event records."""

from typing import Protocol


class LedgerPort(Protocol):
    """The kernel port for ordered durable event records."""
