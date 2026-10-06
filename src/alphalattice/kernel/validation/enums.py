"""Stable vocabulary for Validation protocol state."""

from enum import StrEnum


class SplitMode(StrEnum):
    """Choose expanding or rolling training windows for a research split."""

    EXPANDING = "EXPANDING"
    ROLLING = "ROLLING"


class RebalanceFrequency(StrEnum):
    """Select the daily, weekly or monthly cadence of a proposed rebalance schedule."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"


__all__ = [
    "RebalanceFrequency",
    "SplitMode",
]
