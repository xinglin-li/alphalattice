"""Instrumentation and exporter boundary."""

from typing import Protocol


class TelemetryPort(Protocol):
    """The kernel port for instrumentation and exporter boundary."""
