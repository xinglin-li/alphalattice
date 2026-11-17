"""A recipe's identity, and a definition lineage's, computed from their content alone."""

from __future__ import annotations

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.project import (
    definition_id,
    recipe_key,
    registry_us,
    task_parent,
)

TEST_SOURCE = "src/alphalattice/kernel/data_layer/project.py"
MOMENTUM = definition_id("method", "momentum")


@pytest.mark.proves("data_layer.recipe_key_is_its_content")
def test_a_recipe_key_is_its_method_version_parameters_and_upstream():
    """The key ignores the order parameters and upstream recipes are written in, and changes
    with the method's lineage, its version, a parameter's value or an upstream recipe."""
    key = recipe_key(MOMENTUM, 1, {"lookback": 12, "skip": 1}, ["b", "a"])
    assert key == recipe_key(MOMENTUM, 1, {"skip": 1, "lookback": 12}, ["a", "b"])
    reversal = definition_id("method", "reversal")
    others = (
        recipe_key(MOMENTUM, 2, {"lookback": 12, "skip": 1}, ["a", "b"]),
        recipe_key(MOMENTUM, 1, {"lookback": 6, "skip": 1}, ["a", "b"]),
        recipe_key(MOMENTUM, 1, {"lookback": 12, "skip": 1}, ["a"]),
        recipe_key(reversal, 1, {"lookback": 12, "skip": 1}, ["a", "b"]),
    )
    assert key not in others and len(set(others)) == 4


@pytest.mark.proves("data_layer.lineage_id_is_its_first_registration")
def test_a_lineage_id_is_its_family_and_first_name_and_never_a_name():
    """A lineage's id is a digest of its family and the name it was first registered under: the
    same on every machine, different across families or names, and never the name itself."""
    assert definition_id("method", "momentum") == MOMENTUM
    named = {MOMENTUM, definition_id("metric", "momentum"), definition_id("method", "reversal")}
    assert len(named) == 3
    assert MOMENTUM.startswith("blake3:") and "momentum" not in MOMENTUM


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
