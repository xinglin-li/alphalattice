from __future__ import annotations

from contextlib import nullcontext

import pytest

from devtools.testing.performance import BUDGETS, judged, median_interval, node_budget, within_bound

TEST_SOURCE = "src/devtools/testing/performance.py"


def _unsettled(name):
    return pytest.raises(AssertionError, match=rf"{name}_unsettled: .* idle machine")


@pytest.mark.proves("testing.a_performance_judge_refuses_by_name")
@pytest.mark.parametrize(
    ("judge", "outcome"),
    [
        pytest.param(lambda: judged("digest", 0.0004, 0.00001, 100, 0.001), nullcontext(), id="op"),
        pytest.param(
            lambda: judged("digest", 0.0015, 0.00001, 100, 0.001),
            pytest.raises(AssertionError, match=r"over_budget: digest takes 1500\.0 us"),
            id="op-over",
        ),
        # A 995 us mean whose 99% interval reaches 1020.8 us does not pass a 1000 us budget.
        pytest.param(
            lambda: judged("digest", 0.000995, 0.0001, 100, 0.001), _unsettled("budget"), id="op-?"
        ),
        # Five rounds at 985 us with a 10 us deviation: Student's t reaches 1005.6 us, not 996.5.
        pytest.param(
            lambda: judged("digest", 0.000985, 0.00001, 5, 0.001), _unsettled("budget"), id="op-5"
        ),
        pytest.param(
            lambda: within_bound(0.01, 0.005, 0.015, 400, "the loop"), nullcontext(), id="loop"
        ),
        pytest.param(
            lambda: within_bound(0.03, 0.025, 0.035, 400, "the loop"),
            pytest.raises(AssertionError, match=r"loop_overhead: the owners add 3\.00% to the"),
            id="loop-over",
        ),
        pytest.param(
            lambda: within_bound(0.019, 0.015, 0.023, 400, "the loop"),
            _unsettled("loop"),
            id="loop-?",
        ),
    ],
)
def test_a_performance_judge_passes_within_its_bound_and_refuses_over_it_by_name(judge, outcome):
    """An operation or a loop passes only when its 99% interval is under its bound; one wholly
    over it, or one whose interval still holds it, is refused by name."""
    with outcome:
        judge()


@pytest.mark.proves("testing.a_performance_judge_refuses_by_name")
def test_the_median_interval_widens_with_the_looks_it_is_split_over():
    """The median's interval is the order statistics n/2 -+ z*sqrt(n)/2, and splitting 99% over
    ten looks widens it, so stopping at any look keeps the stated confidence."""
    ratios = [1 + index / 1000 for index in range(100)]
    assert median_interval(ratios, looks=1) == pytest.approx((1.036, 1.063))
    assert median_interval(ratios, looks=10) == pytest.approx((1.032, 1.067))


@pytest.mark.proves("testing.a_node_judges_its_operations_by_its_own_budgets")
@pytest.mark.parametrize(
    ("record", "outcome"),
    [
        pytest.param(None, 0.001, id="no-record"),
        pytest.param('{"digest": 0.0008}', 0.0008, id="listed"),
        pytest.param('{"grant": 0.0002}', None, id="unlisted"),
        pytest.param("not json", "budgets_unreadable", id="unreadable"),
        pytest.param('{"digest": 1e309}', "budgets_unreadable", id="infinite"),
        pytest.param('{"digest": 0}', "budgets_unreadable", id="zero"),
        # A JSON true is no one-second budget, and a string is no number.
        pytest.param('{"digest": true}', "budgets_unreadable", id="bool"),
        pytest.param('{"digest": "0.001"}', "budgets_unreadable", id="string"),
    ],
)
def test_an_operation_is_judged_by_its_nodes_budget(tmp_path, monkeypatch, record, outcome):
    """The node's record decides an operation's budget; one it does not list is not judged."""
    if record is not None:
        (tmp_path / "budgets.json").write_text(record, encoding="utf-8")
        monkeypatch.setenv(BUDGETS, str(tmp_path / "budgets.json"))
    else:
        monkeypatch.delenv(BUDGETS, raising=False)
    if outcome == "budgets_unreadable":
        with pytest.raises(
            AssertionError, match=r"performance\.budgets_unreadable: .*budgets\.json"
        ):
            node_budget("digest", 0.001)
    else:
        assert node_budget("digest", 0.001) == outcome
