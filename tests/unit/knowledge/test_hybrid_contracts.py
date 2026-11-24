"""A new Evidence authority takes the int8 retrieval recipe; an installed one keeps its own."""

from __future__ import annotations

import pytest

from alphalattice.capabilities.knowledge.hybrid_contracts import (
    NEW_BUILD_RECIPE,
    RECIPE_MINILM_CPU,
    RECIPE_MINILM_INT8_CPU,
    setup_recipe,
)


@pytest.mark.proves("knowledge.a_new_authority_takes_the_int8_recipe")
def test_a_new_authority_takes_the_int8_recipe_and_an_installed_one_keeps_its_own() -> None:
    assert NEW_BUILD_RECIPE == RECIPE_MINILM_INT8_CPU
    assert setup_recipe(None, installed=None, retained_layout=False) == RECIPE_MINILM_INT8_CPU
    # A pack directory of the retained layout is read only by the retained recipe.
    assert setup_recipe(None, installed=None, retained_layout=True) == RECIPE_MINILM_CPU
    assert (
        setup_recipe(None, installed=RECIPE_MINILM_CPU, retained_layout=False) == RECIPE_MINILM_CPU
    )
    named = "hybrid-v3-bge-small-en"
    assert setup_recipe(named, installed=RECIPE_MINILM_CPU, retained_layout=True) == named
