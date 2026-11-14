"""The one way to wait always ends: at its result, its deadline or its stop."""

import threading

import pytest

from alphalattice.kernel.shared_kernel import waits

TEST_SOURCE = "src/alphalattice/kernel/shared_kernel/waits.py"


@pytest.fixture
def paused(monkeypatch):
    """A clock that moves only by the pauses taken; a pause says whether the stop is set."""
    now, pauses = [0.0], []

    def pause(seconds, *, stop=None):
        pauses.append(seconds)
        now[0] += seconds
        return bool(stop and stop.is_set())

    monkeypatch.setattr(waits, "clock", lambda: now[0])
    monkeypatch.setattr(waits, "pause", pause)
    return pauses


@pytest.mark.proves("shared_kernel.a_wait_ends_by_its_deadline_or_its_stop")
def test_a_wait_returns_its_result_or_ends_by_name_at_its_deadline_or_stop(paused):
    """requirement: a wait hands each check the time left, backs off, and ends by name."""
    seen, results = [], iter([None, 0, "ready"])
    got = waits.wait_until(
        lambda left: seen.append(left) or next(results), timeout=5, what="ready", interval=0.1
    )
    assert got == "ready" and seen == [5, 5 - paused[0], 5 - paused[0] - paused[1]]
    assert 0.1 <= paused[0] <= 0.2 and 0.2 <= paused[1] <= 0.3  # the backoff doubles
    paused.clear()
    with pytest.raises(waits.WaitTimeout, match="the lane frees not reached within 10 s"):
        waits.wait_until(lambda left: False, timeout=10, what="the lane frees", max_interval=30)
    assert sum(paused) == pytest.approx(10)  # the last pause is cut to the time left
    stop, checks = threading.Event(), []
    with pytest.raises(waits.WaitStopped, match="the drain was stopped"):
        waits.wait_until(
            lambda left: checks.append(left) or stop.set(), timeout=60, what="the drain", stop=stop
        )
    assert len(checks) == 1


@pytest.mark.proves("shared_kernel.a_wait_ends_by_its_deadline_or_its_stop")
@pytest.mark.parametrize(
    "seconds",
    [float("inf"), float("nan"), 0, -1, 3601, None, True],
    ids=["inf", "nan", "zero", "negative", "over-ceiling", "none", "bool"],
)
def test_a_wait_without_a_finite_capped_bound_is_refused_before_it_starts(seconds):
    """requirement: an infinite, empty, negative, uncapped or non-number bound is refused."""
    with pytest.raises(ValueError, match=r"wait\.bound_invalid"):
        waits.wait_until(lambda left: True, timeout=seconds, what="x")
    with pytest.raises(ValueError, match=r"wait\.bound_invalid"):
        waits.pause(seconds)
