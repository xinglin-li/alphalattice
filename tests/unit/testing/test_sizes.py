import pytest

from devtools.architecture.git_inputs import carried
from devtools.testing.sizes import budget_rises, validate_policy

TEST_SOURCE = "src/devtools/testing/sizes.py"

ROW = dict(pattern="tests/a/test_x.py::t*", owner="tests", reason="Builds a world.")
POLICY = dict(
    schema="alphalattice.test-sizes.v1",
    sizes=["small", "medium", "large"],
    defaults=[dict(pattern="tests/*", owner="tests", size="small")],
    large=[dict(ROW, timeout_seconds=300)],
    limits=dict(small=1, medium=60, large=900),
)
NAME = "large:tests/a/test_x.py::t*"


def admitted(budget: str, value: float) -> dict:
    return dict(budget=budget, value=value, admitted="2026-11-14: measured at 90 s")


def timed(seconds: float | None, **extra) -> dict:
    row = dict(ROW) if seconds is None else dict(ROW, timeout_seconds=seconds)
    return dict(POLICY, large=[row], **extra)


@pytest.mark.proves("testing.test_budgets_only_fall")
@pytest.mark.parametrize(
    ("after", "refused"),
    [
        pytest.param(POLICY, False, id="an unchanged budget passes"),
        pytest.param(timed(120), False, id="a lowered timeout needs no admission"),
        pytest.param(dict(POLICY, large=[]), False, id="a removed Large row needs no admission"),
        pytest.param(timed(600), True, id="a raised timeout without an admission is refused"),
        pytest.param(timed(None), True, id="a dropped timeout rises to the Large limit"),
        pytest.param(
            timed(600, budget_admissions=[admitted(NAME, 600)]),
            False,
            id="a raised timeout with its admission passes",
        ),
        pytest.param(
            timed(600, budget_admissions=[admitted(NAME, 450)]),
            True,
            id="an admission below the new value is refused",
        ),
        pytest.param(
            dict(POLICY, limits=dict(small=1, medium=90, large=900)),
            True,
            id="a raised size limit is refused",
        ),
        pytest.param(
            dict(POLICY, large=[*POLICY["large"], dict(ROW, pattern="tests/b.py::u")]),
            True,
            id="an added Large row is an added budget",
        ),
        pytest.param(
            dict(POLICY, growth=dict(floor_lines=30)),
            True,
            id="an added growth floor is an added budget",
        ),
    ],
)
def test_a_test_budget_only_falls_unless_a_dated_admission_holds_it(after, refused) -> None:
    """Every raised or added budget names its admission; lowering or removing one needs none."""
    assert bool(budget_rises(POLICY, after)) is refused


@pytest.mark.proves("testing.test_budgets_only_fall")
def test_a_moved_large_row_keeps_its_budget_and_its_admissions_stay_current() -> None:
    """A refusal names the budget and its rise; a byte-identical move carries its budget; an
    admission names one current budget once, dated."""
    (rise,) = budget_rises(POLICY, timed(600))
    assert rise.startswith(f"test budget {NAME} rose 300 -> 600;")
    moves = {"tests/b/test_x.py": "tests/a/test_x.py"}
    moved = dict(POLICY, large=[dict(ROW, pattern="tests/b/test_x.py::t*", timeout_seconds=300)])
    assert budget_rises(POLICY, moved)  # a rename without its move is an added budget
    assert not budget_rises(POLICY, moved, lambda v: next(iter(carried([v], moves)), v))
    assert validate_policy(timed(600, budget_admissions=[admitted(NAME, 600)]))
    stale = admitted("large:tests/gone.py::t", 600)
    undated = [dict(admitted(NAME, 600), admitted=text) for text in ("why", "2026-02-30: why")]
    for rows in ([stale], [admitted(NAME, 600)] * 2, *([row] for row in undated)):
        with pytest.raises(ValueError, match=r"test_size\.budget_admission"):
            validate_policy(timed(600, budget_admissions=rows))
