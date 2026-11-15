"""A recipe's identity, computed from its content alone."""

from __future__ import annotations

import pytest

from alphalattice.kernel.data_layer.project import recipe_key

TEST_SOURCE = "src/alphalattice/kernel/data_layer/project.py"


@pytest.mark.proves("data_layer.recipe_key_is_its_content")
def test_a_recipe_key_is_its_method_version_parameters_and_upstream():
    """The key ignores the order parameters and upstream recipes are written in, and changes
    with the method's version, a parameter's value or an upstream recipe."""
    key = recipe_key("momentum", 1, {"lookback": 12, "skip": 1}, ["b", "a"])
    assert key == recipe_key("momentum", 1, {"skip": 1, "lookback": 12}, ["a", "b"])
    others = (
        recipe_key("momentum", 2, {"lookback": 12, "skip": 1}, ["a", "b"]),
        recipe_key("momentum", 1, {"lookback": 6, "skip": 1}, ["a", "b"]),
        recipe_key("momentum", 1, {"lookback": 12, "skip": 1}, ["a"]),
    )
    assert key not in others and len(set(others)) == 3
