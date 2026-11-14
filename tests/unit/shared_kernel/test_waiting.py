import itertools
import math
import threading

import pytest

from alphalattice.kernel.shared_kernel import waiting

TEST_SOURCE = "src/alphalattice/kernel/shared_kernel/waiting.py"


class _Clock:
    """The monotonic clock and sleep, driven: a sleep advances the clock and is recorded."""

    def __init__(self) -> None:
        self.now, self.sleeps = 100.0, list[float]()

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.proves("shared_kernel.a_wait_ends_at_its_bound")
@pytest.mark.parametrize(
    "seconds",
    [
        pytest.param(math.inf, id="infinite"),
        pytest.param(math.nan, id="not-a-number"),
        pytest.param(0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(True, id="a-bool"),
        pytest.param(3600.5, id="past-the-hour"),
    ],
)
def test_a_wait_refuses_a_bound_that_is_not_finite_positive_and_within_the_hour(seconds):
    """A wait's length is finite, positive and at most the hour, or the wait is refused by name."""
    with pytest.raises(ValueError, match=r"wait.bound_invalid"):
        waiting.wait_until(lambda _left: True, timeout=seconds, what="x")


@pytest.mark.proves("shared_kernel.a_wait_ends_at_its_bound")
@pytest.mark.parametrize(
    ("ready_at", "stop_at", "outcome"),
    [
        pytest.param(3, None, "ready", id="first-truthy-result"),
        pytest.param(None, None, "timeout", id="deadline-reached"),
        pytest.param(None, 2, "stopped", id="stop-event-set"),
    ],
)
def test_a_wait_ends_at_its_result_its_deadline_or_its_stop(
    monkeypatch, ready_at, stop_at, outcome
):
    """A wait returns its check's first truthy result, given the seconds left; it ends at its
    deadline with WaitTimeout, never pausing past it, and at a set stop event with WaitStopped."""
    clock = _Clock()
    monkeypatch.setattr(waiting.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(waiting.time, "sleep", clock.sleep)
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda seconds: clock.sleep(seconds) or stop.is_set())
    lefts = list[float]()

    def check(left: float) -> str | None:
        lefts.append(left)
        if len(lefts) == stop_at:
            stop.set()
        return "done" if len(lefts) == ready_at else None

    expected = {"ready": None, "timeout": waiting.WaitTimeout, "stopped": waiting.WaitStopped}
    if expected[outcome] is None:
        assert waiting.wait_until(check, timeout=10, what="x", stop=stop) == "done"
        assert len(lefts) == ready_at
    else:
        with pytest.raises(expected[outcome], match=rf"wait.{outcome}: x"):
            waiting.wait_until(check, timeout=10, what="x", stop=stop)
    assert lefts[0] == 10 and all(a > b for a, b in itertools.pairwise(lefts))
    assert clock.now <= 110 and all(0 < pause <= 2 for pause in clock.sleeps)
