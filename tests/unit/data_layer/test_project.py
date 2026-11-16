"""A recipe's identity, and a definition lineage's, computed from their content alone."""

from __future__ import annotations

import pytest

from alphalattice.kernel.data_layer.project import definition_id, recipe_key

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
