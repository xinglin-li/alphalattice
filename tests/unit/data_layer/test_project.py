"""A recipe key is computed from its registered lineage and typed content."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.project import (
    definition_names,
    dims,
    input_digest,
    recipe_key,
    registry_us,
    task_parent,
)

TEST_SOURCE = "src/alphalattice/kernel/data_layer/project.py"
MOMENTUM = "fixture-lineage-a"


@pytest.mark.proves("data_layer.a_number_has_one_typed_form")
def test_a_number_has_one_typed_form_in_a_recipe_key_and_in_dims():
    """5 and 5.0 are one parameter and one dimension; a value with no typed form is refused."""
    assert recipe_key(MOMENTUM, 1, {"lookback": 5}) == recipe_key(MOMENTUM, 1, {"lookback": 5.0})
    assert (
        dims({"side": "long", "h": 21.0})
        == dims({"h": 21, "side": "long"})
        == '{"h":21,"side":"long"}'
    )
    for code, value in (
        ("value_not_finite", {"h": float("nan")}),
        ("measure_dims_untyped", "h=21"),
    ):
        with pytest.raises(DataError) as refused:
            dims(value)
        assert refused.value.failure.code == f"data_layer.{code}"


def _admitted(admission, *parts, dataset="raw_bar", rule=None, version=None):
    """One admission as input_digest reads it from the log: its parts' knowledge ranges."""
    ranges = [{"part": p, "min_known_asof_us": lo, "max_known_asof_us": hi} for p, lo, hi in parts]
    return (admission, dataset, json.dumps(ranges), rule, version)


def _digest(*admissions, cutoff=25, parts=("2023",)):
    log = SimpleNamespace(execute=lambda sql: SimpleNamespace(fetchall=lambda: list(admissions)))
    return input_digest(log, {"raw_bar": parts}, cutoff)


@pytest.mark.proves("data_layer.a_cache_is_valid_for_exactly_its_inputs")
def test_an_input_digest_moves_only_with_the_admissions_its_parts_see():
    """A part's digest moves with a correction it sees, a lag rule's version or a cutoff inside a
    range; never with another part's admission, a later one or the order admissions are read in."""
    first = _admitted("a1", ("2023", 10, 20), rule="yf-close", version=1)
    other = _admitted("a2", ("2024", 30, 40))
    held = _digest(first, other)
    assert held.startswith("blake3:") and held == _digest(other, first)
    assert held == _digest(first, other, _admitted("a3", ("2024", 22, 22)))  # another part
    assert held == _digest(first, other, _admitted("a3", ("2023", 30, 30)))  # after the cutoff
    assert held == _digest(first, cutoff=24)  # past every range: its latest knowledge time
    assert held != _digest(first, other, _admitted("a3", ("2023", 22, 22)))  # a correction
    assert held != _digest(_admitted("a1", ("2023", 10, 20), rule="yf-close", version=2), other)
    assert _digest(first, cutoff=15) != _digest(first, cutoff=18)  # inside a range: the cutoff


@pytest.mark.proves("data_layer.recipe_key_is_its_content")
def test_a_recipe_key_is_its_method_version_parameters_and_upstream():
    """The key ignores the order parameters and upstream recipes are written in, and changes
    with the method's lineage, its version, a parameter's value or an upstream recipe."""
    key = recipe_key(MOMENTUM, 1, {"lookback": 12, "skip": 1}, ["b", "a"])
    assert key == recipe_key(MOMENTUM, 1, {"skip": 1, "lookback": 12}, ["a", "b"])
    reversal = "fixture-lineage-b"
    others = (
        recipe_key(MOMENTUM, 2, {"lookback": 12, "skip": 1}, ["a", "b"]),
        recipe_key(MOMENTUM, 1, {"lookback": 6, "skip": 1}, ["a", "b"]),
        recipe_key(MOMENTUM, 1, {"lookback": 12, "skip": 1}, ["a"]),
        recipe_key(reversal, 1, {"lookback": 12, "skip": 1}, ["a", "b"]),
    )
    assert key not in others and len(set(others)) == 4


@pytest.mark.proves("data_layer.lineage_id_is_its_first_registration")
@pytest.mark.parametrize(
    ("kind", "phase", "attrs", "text", "expected"),
    [
        (
            "definition.registered",
            None,
            {},
            '{"family":"method","name":"m","version":1}',
            [("method", "m")],
        ),
        ("definition.registered", None, {}, '{"family":"method","name":"m","version":2}', []),
        ("task.requested", None, {"type": "factor.run"}, None, [("event_type", "factor.run")]),
        (
            "wait.started",
            None,
            {"event_type": "factor.done"},
            None,
            [("event_type", "factor.done")],
        ),
        ("wait.started", None, {}, None, []),
        ("trial", "completed", {}, None, [("node", "worker")]),
        ("trial", "started", {}, None, []),
    ],
    ids=(
        "first-version",
        "later-version",
        "task-type",
        "wait-type",
        "timer",
        "span-end",
        "span-start",
    ),
)
def test_a_definition_is_first_named_by_the_event_that_registers_it(
    kind, phase, attrs, text, expected
):
    """Find an event's explicit or implicit first registration names from its own payload."""
    assert list(definition_names(kind, phase, "worker", attrs, text)) == expected


@pytest.mark.proves("data_layer.a_task_parent_is_its_nearest")
@pytest.mark.parametrize(
    ("version", "attrs", "parent"),
    [
        (2, {"parent_task_id": "k0", "goal_id": "g1"}, ("k0", None, None)),
        (2, {"goal_id": "g1"}, (None, "g1", None)),
        (2, {}, (None, None, "w1")),
        (1, {"submitted_by": {"goal_id": "g1"}}, (None, "g1", None)),
        (1, {"submitted_by": {"goal_id": "g2"}}, (None, None, "w1")),  # another workspace's
        (1, {"submitted_by": None}, (None, None, "w1")),
    ],
    ids=["sub-task", "goal", "workspace", "v1-goal", "v1-other-goal", "v1-no-agent"],
)
def test_a_task_stores_its_nearest_parent(version, attrs, parent):
    """Version 2 names a parent task, else a goal, else neither; version 1's goal is its
    agent's, when the log holds it in the task's own workspace."""
    event = {"type_version": version, "workspace_id": "w1"}
    assert task_parent(event, attrs, {"g1"}.__contains__) == parent


SECOND = 1_763_078_400_000_000  # 2025-11-14 00:00:00 UTC, in microseconds


@pytest.mark.proves("data_layer.a_task_control_time_is_its_microseconds")
@pytest.mark.parametrize(
    ("text", "us"),
    [
        pytest.param("2025-11-14 00:00:00.000007", SECOND + 7, id="registry"),
        pytest.param("2025-11-14 00:00:00.000007+00:00", SECOND + 7, id="offset"),
        pytest.param("2025-11-14T00:00:00Z", SECOND, id="whole-second"),
        pytest.param("2025-11-14T00:00:00.123Z", None, id="three-digits"),
        pytest.param("2025-11-14 00:00:00+05:00", None, id="not-utc"),
        pytest.param("2025-11-14", None, id="a-date"),
    ],
)
def test_a_task_control_time_is_read_to_the_microsecond(text, us):
    """The second, then six fraction digits or none, in UTC: anything else is refused by name."""
    if us is None:
        with pytest.raises(DataError, match="UTC ISO"):
            registry_us(text)
    else:
        assert registry_us(text) == us
