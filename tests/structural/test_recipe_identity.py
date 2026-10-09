"""A recipe's identity is its content, never its names (NM1, LAWS.md ID10)."""

from __future__ import annotations

import re
import typing
from dataclasses import replace

from pydantic import BaseModel

from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    BROAD_FEATURE_PACKAGE,
    FROZEN_RESEARCH_BOOK_RECIPES,
    INSTALLED_HETEROGENEOUS_BOOK_RECIPE,
    heterogeneous_package,
)
from alphalattice.kernel.shared_kernel.recipe_identity import recipe_identity


def _renamed(value: object) -> object:
    """The value with every label it lists renamed, at every depth."""
    if isinstance(value, BaseModel):
        labels = getattr(type(value), "LABELS", frozenset())
        update: dict[str, object] = {}
        for name in type(value).model_fields:
            item = getattr(value, name)
            if name in labels:
                update[name] = _relabelled(item)
            elif isinstance(item, BaseModel | tuple):
                update[name] = _renamed(item)
        return value.model_copy(update=update)
    if isinstance(value, tuple):
        return tuple(_renamed(item) for item in value)
    return value


def _relabelled(value: object) -> object:
    if isinstance(value, str):
        return value + " renamed"
    if isinstance(value, tuple):
        return tuple(_relabelled(item) for item in value)
    return value


def _sealed() -> list[tuple[BaseModel, str]]:
    strategy = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY
    return [
        (INSTALLED_ALPHA_PRODUCT_RECIPE, "recipe_hash"),
        (strategy, "strategy_hash"),
        *((component, "recipe_hash") for component in strategy.components),
        (INSTALLED_HETEROGENEOUS_BOOK_RECIPE, "recipe_hash"),
        (BROAD_FEATURE_PACKAGE, "package_hash"),
        (heterogeneous_package(), "package_hash"),
    ]


def test_a_renamed_label_moves_no_identity() -> None:
    """A renamed label moves no identity."""

    for model, seal in _sealed():
        seal_field = frozenset({seal})
        identity = recipe_identity(model, exclude=seal_field)
        assert identity == getattr(model, seal), type(model).__name__
        renamed = _renamed(model)
        assert isinstance(renamed, BaseModel)
        assert renamed != model, type(model).__name__
        assert recipe_identity(renamed, exclude=seal_field) == identity, type(model).__name__

    product = INSTALLED_ALPHA_PRODUCT_RECIPE
    moved = product.model_copy(update={"purge_sessions": product.purge_sessions + 1})
    assert recipe_identity(moved, exclude=frozenset({"recipe_hash"})) != product.recipe_hash

    for book in FROZEN_RESEARCH_BOOK_RECIPES:
        renamed_book = replace(
            book,
            strategy_id=book.strategy_id + " renamed",
            components=tuple(
                replace(value, component_id=value.component_id + " renamed")
                for value in book.components
            ),
        )
        assert renamed_book.recipe_hash == book.recipe_hash
        assert replace(book, top_k=book.top_k + 1).recipe_hash != book.recipe_hash


def test_the_label_table_names_every_installed_strategy_component_and_rule() -> None:
    """The label table names every installed strategy component and rule."""

    from alphalattice.interface.local_application.labels import label, label_table, title
    from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
        COMPONENT_IDS,
        HETEROGENEOUS_STRATEGY_ID,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies import (
        buffered_equal_weight,
        buffered_inverse_volatility,
        buffered_rank_return,
        installed_strategies,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
        TrancheWeightRule,
    )

    labels = label_table().labels
    assert len({value.id for value in labels}) == len(labels)
    installed = (
        installed_strategies.BROAD_FEATURE_STRATEGY_ID,
        HETEROGENEOUS_STRATEGY_ID,
        installed_strategies.REBOUND_RETURN_STRATEGY_ID,
        installed_strategies.TREND_REBOUND_STRATEGY_ID,
        *COMPONENT_IDS,
        *typing.get_args(TrancheWeightRule.__value__),
        buffered_equal_weight.WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT_POLICY_ID,
        buffered_inverse_volatility.INVERSE_VOLATILITY_POLICY_ID,
        buffered_rank_return.CAUSAL_RANK_MU_POLICY_ID,
    )
    assert [code for code in installed if label(code) is None] == []
    stage_code = re.compile(r"\b(?:G\d+|C\d|R\d|IW184|Gate [A-Z]|Stage \d+|Policy [A-Z])\b")
    assert [value.id for value in labels if stage_code.search(value.title)] == []
    assert title("RETURN_G6_MU_ONLY") != "RETURN_G6_MU_ONLY"
    assert title("no such id") == "no such id"
