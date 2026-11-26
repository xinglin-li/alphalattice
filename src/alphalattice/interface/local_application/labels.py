"""The product's names for what it installs, from one table (NM1, LAWS.md ID10).

``labels.json`` holds, by the id the code and the workspaces keep, a title and a summary in English
and Chinese for each strategy, component, recipe, policy, risk method and rule. The CLI and the
docs read titles here and the Workbench reads the same table; nothing that decides a number or an
identity reads it, so renaming a title moves neither.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_TABLE = Path(__file__).with_name("labels.json")

LabelKind = Literal[
    "book_recipe",
    "component",
    "feature_axis",
    "model_family",
    "model_set",
    "objective",
    "policy",
    "recipe",
    "risk_method",
    "score_consumer",
    "strategy",
    "target",
    "target_cross_section",
    "weight_rule",
]


class Label(BaseModel):  # type: ignore[misc]
    """One installed thing's name: its kind, the id it is stored by, and its words."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: LabelKind
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    title_zh: str = Field(min_length=1)
    summary_zh: str = Field(min_length=1)


class LabelTable(BaseModel):  # type: ignore[misc]
    """The whole table, one label per id."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    table_schema: Literal["alphalattice.labels"] = Field(alias="schema")
    version: Literal[1]
    note: str = Field(min_length=1)
    labels: tuple[Label, ...] = Field(min_length=1)


@cache
def label_table() -> LabelTable:
    """The installed label table, read once."""
    table: LabelTable = LabelTable.model_validate_json(_TABLE.read_text(encoding="utf-8"))
    return table


@cache
def _by_id() -> dict[str, Label]:
    return {value.id: value for value in label_table().labels}


def label(code: str) -> Label | None:
    """The label of one installed id, or none when the table names no such id."""
    return _by_id().get(code)


def title(code: str, *, chinese: bool = False) -> str:
    """The title a reader sees for an id: the label's, or the id itself when it has none."""
    value = label(code)
    if value is None:
        return code
    return value.title_zh if chinese else value.title


def titles_in(answer: object, *, chinese: bool = False) -> dict[str, str]:
    """The title of each installed id an answer names, as a key or a value, in id order.

    The CLI shows them beside the answer, whose ids stay as their owner wrote them.

    Args:
        answer: An owner's answer.
        chinese: Whether the titles are the Chinese ones.

    Returns:
        Each named id's title; empty when the answer names none.
    """
    known = _by_id()
    found: dict[str, str] = {}
    pending = [answer]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            named = known.get(value)
            if named is not None:
                found[value] = named.title_zh if chinese else named.title
        elif isinstance(value, dict):
            pending.extend(value)
            pending.extend(value.values())
        elif isinstance(value, list | tuple):
            pending.extend(value)
    return dict(sorted(found.items()))


__all__ = ["Label", "LabelKind", "LabelTable", "label", "label_table", "title", "titles_in"]
