"""Outbound access boundary."""

from typing import Protocol


class EgressPort(Protocol):
    """The kernel port for outbound access boundary."""
