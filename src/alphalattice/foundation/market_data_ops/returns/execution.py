"""Shared execution-return mathematics for Data-owned research surfaces."""

from __future__ import annotations

import math


def open_to_open_total_return_gross(
    *, entry_open: float, exit_open: float, period_dividend: float
) -> float:
    """Return the dividend-aware gross return on one adjusted-open interval.

    Args:
        entry_open: Positive adjusted open at the interval's entry.
        exit_open: Positive adjusted open at its exit.
        period_dividend: Nonnegative dividend attributable to the interval.

    Returns:
        Positive gross return including the dividend.

    Raises:
        ValueError: An input is nonfinite or invalid, or the gross is invalid.

    """
    values = (float(entry_open), float(exit_open), float(period_dividend))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("market_data.execution_return_non_finite")
    if values[0] <= 0.0 or values[1] <= 0.0 or values[2] < 0.0:
        raise ValueError("market_data.execution_return_input_invalid")
    gross = (values[1] + values[2]) / values[0]
    if not math.isfinite(gross) or gross <= 0.0:
        raise ValueError("market_data.execution_return_gross_invalid")
    return gross


def open_to_open_simple_return(
    *, entry_open: float, exit_open: float, period_dividend: float
) -> float:
    """Return gross return less one for an adjusted-open interval.

    Raises:
        ValueError: The inputs cannot form a valid gross return.

    """
    return (
        open_to_open_total_return_gross(
            entry_open=entry_open,
            exit_open=exit_open,
            period_dividend=period_dividend,
        )
        - 1.0
    )


def open_to_open_log_return(
    *, entry_open: float, exit_open: float, period_dividend: float
) -> float:
    """Return the natural log of an adjusted-open gross return.

    Raises:
        ValueError: The inputs cannot form a valid gross return.

    """
    return math.log(
        open_to_open_total_return_gross(
            entry_open=entry_open,
            exit_open=exit_open,
            period_dividend=period_dividend,
        )
    )


__all__ = [
    "open_to_open_log_return",
    "open_to_open_simple_return",
    "open_to_open_total_return_gross",
]
