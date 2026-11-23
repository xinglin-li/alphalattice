"""The performance holder's judges: an operation's mean call against its budget, and a hot loop's
complete cost with owners plugged in against the same loop without them.

Each verdict is given only once its 99% interval clears the bound: wholly under it passes, wholly
over it is refused by name, and an interval that still holds the bound is refused by name as
unsettled, with the repair. An operation's interval is its pytest-benchmark mean's with Student's
t at the rounds it ran, one look after the tool's own time-bound rounds
(tests/integration/testing/test_performance.py); the loop's is the median of complete-loop ratios,
looked at no more than LOOKS times with the confidence split over every look (Bonferroni), so
stopping at any of them keeps 99%.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil, floor, sqrt
from pathlib import Path

LOOP_BOUND = 0.02
"""The 0.2.9 done-when: a loop costs at most 2% more with the kernel's owners plugged in."""
CONFIDENCE, LOOKS = 0.99, 10
BUDGETS = "ALPHALATTICE_PERFORMANCE_BUDGETS"


@dataclass(frozen=True)
class Operation:
    """One kernel operation, its call and its budget: the mean seconds a call may take. A call that
    needs a test's fixtures (a running Host) takes pytest's request and returns the call measured.
    """

    name: str
    call: Callable[..., object]
    budget: float


def _z(looks: int = 1) -> float:
    return statistics.NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / (2 * looks))


def _t(df: int) -> float:
    """Student's t two-sided quantile at CONFIDENCE for an integer df: bisection on its closed-form
    distribution (Abramowitz and Stegun 26.7.3-4), equal to scipy's to 1e-4."""

    def inside(value: float) -> float:
        theta = math.atan(value / sqrt(df))
        sine, cosine, odd = math.sin(theta), math.cos(theta), df % 2
        if df == 1:
            return 2 * theta / math.pi
        term = total = cosine if odd else 1.0
        for k in range(1, (df - 1) // 2 if odd else df // 2):
            term *= cosine * cosine * ((2 * k) / (2 * k + 1) if odd else (2 * k - 1) / (2 * k))
            total += term
        return 2 / math.pi * (theta + sine * total) if odd else sine * total

    low, high = 0.0, 1e4
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if inside(middle) < CONFIDENCE else (low, middle)
    return high


def _verdict(low: float, high: float, bound: float, over: str, unsettled: str, said: str) -> None:
    if low > bound:
        raise AssertionError(f"performance.{over}: {said}")
    if high > bound:
        raise AssertionError(
            f"performance.{unsettled}: {said}, which does not settle within the time bound; "
            "run the performance rows on an idle machine"
        )


def node_budget(name: str, default: float) -> float | None:
    """An operation's budget on the node that runs it: from the node's record, {name: seconds},
    which the nightly names in BUDGETS; None when that record does not list it (measured, not
    judged); without the variable, the holder's own budget, the workshop's main machine's."""
    if not (path := os.environ.get(BUDGETS)):
        return default
    try:
        budgets = json.loads(Path(path).read_text(encoding="utf-8"))
        if (budget := budgets.get(name)) is None:
            return None
        if isinstance(budget, bool) or not isinstance(budget, int | float):
            raise ValueError(f"{name}'s budget {budget!r} is not a JSON number of seconds")
        if not (math.isfinite(seconds := float(budget)) and seconds > 0):
            raise ValueError(
                f"{name}'s budget {budget!r} is not a finite positive number of seconds"
            )
        return seconds
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise AssertionError(
            f"performance.budgets_unreadable: {path}: {error}; restore the node's budget record"
        ) from error


def judged(name: str, mean: float, stddev: float, rounds: int, budget: float) -> None:
    """Judge an operation's benchmark mean, with its 99% interval at the rounds it ran (Student's t
    with rounds - 1 degrees of freedom: a five-round sample's deviation is itself an estimate),
    against its budget."""
    half = _t(rounds - 1) * stddev / sqrt(rounds)
    low, high = mean - half, mean + half
    said = (
        f"{name} takes {mean * 1e6:.1f} us a call (99% interval {low * 1e6:.1f} to "
        f"{high * 1e6:.1f} us, {rounds} rounds) against its budget of {budget * 1e6:.1f} us"
    )
    _verdict(low, high, budget, "over_budget", "budget_unsettled", said)


def median_interval(ratios: Sequence[float], looks: int = LOOKS) -> tuple[float, float]:
    """The distribution-free interval of the ratios' median: the order statistics at
    n/2 -+ z*sqrt(n)/2, at the confidence split over `looks`."""
    ranked, n, z = sorted(ratios), len(ratios), _z(looks)
    return ranked[max(0, floor(n / 2 - z * sqrt(n) / 2) - 1)], ranked[
        min(n - 1, ceil(n / 2 + z * sqrt(n) / 2))
    ]


def overhead[T](
    bare: Callable[[T], object],
    plugged: Callable[[T], object],
    items: Sequence[T],
    *,
    step: int = 12,
    max_seconds: float = 20.0,
) -> tuple[float, float, float, int]:
    """The owners' extra cost on the complete loop, as the median of complete-loop ratios, its 99%
    interval and the loops.

    A loop pass runs every item bare and plugged, each item's pair alternating which runs first, so
    drift and load fall on both; its ratio is the plugged items' total time over the bare ones', so
    a cost on any one item counts in the loop it recurs in. The median of passes holds against the
    heavy-tailed pauses a loaded machine adds (a 5.5% stdev per loop at the workshop's usual load,
    11-22), where a mean does not settle. Each `step` passes is one look; the passes stop at the
    first look whose interval clears the bound, at the last look, or at the time bound."""
    ratios: list[float] = []
    deadline = time.perf_counter() + max_seconds
    for _look in range(LOOKS):
        for _pass in range(step):
            seconds = [0.0, 0.0]
            for index, item in enumerate(items):
                for side in (0, 1) if (len(ratios) + index) % 2 == 0 else (1, 0):
                    begun = time.perf_counter()
                    (bare, plugged)[side](item)
                    seconds[side] += time.perf_counter() - begun
            ratios.append(seconds[1] / seconds[0])
        low, high = median_interval(ratios)
        if low - 1 > LOOP_BOUND or high - 1 <= LOOP_BOUND or time.perf_counter() > deadline:
            break
    return statistics.median(ratios) - 1, low - 1, high - 1, len(ratios)


def within_bound(cost: float, low: float, high: float, loops: int, loop: str) -> None:
    """Judge the owners' cost on a loop, with its 99% interval, against the 2% bound."""
    said = (
        f"the owners add {cost:.2%} to {loop} (99% interval {low:.2%} to {high:.2%}, "
        f"{loops} complete loops) against its 2% bound"
    )
    _verdict(low, high, LOOP_BOUND, "loop_overhead", "loop_unsettled", said)
