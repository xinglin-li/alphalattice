"""The product's names for what it installs, from one table (NM1, LAWS.md ID10).

``labels.json`` holds, by the id the code and the workspaces keep, a title and a summary in English
and Chinese for each strategy, component, recipe, policy, risk method and rule. The CLI and the
docs read titles here and the Workbench reads the same table; nothing that decides a number or an
identity reads it, so renaming a title moves neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any, Literal, cast, get_args

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


@dataclass(frozen=True, slots=True)
class Label:
    """One installed thing's name: its kind, the id it is stored by, and its words."""

    kind: LabelKind
    id: str
    title: str
    summary: str
    title_zh: str
    summary_zh: str


@dataclass(frozen=True, slots=True)
class LabelTable:
    """The whole table, one label per id."""

    table_schema: Literal["alphalattice.labels"]
    version: Literal[1]
    note: str
    labels: tuple[Label, ...]


_LABEL_FIELDS = tuple(value.name for value in fields(Label))


def _words(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"labels.table_unreadable: {_TABLE}: {where} is not words")
    return value


def _label(value: Any, index: int) -> Label:
    if not isinstance(value, dict) or set(value) != set(_LABEL_FIELDS):
        raise ValueError(f"labels.table_unreadable: {_TABLE}: label {index} has other fields")
    kind = value["kind"]
    if kind not in get_args(LabelKind):
        raise ValueError(f"labels.table_unreadable: {_TABLE}: label {index} has an unknown kind")
    words = {name: _words(value[name], f"label {index} {name}") for name in _LABEL_FIELDS[1:]}
    return Label(kind=cast(LabelKind, kind), **words)


def _read(text: str) -> LabelTable:
    """The table as labels.json holds it: exactly its fields, each label's words non-empty.

    A decoded table of another shape is refused by name (`labels.table_unreadable`); text that is
    not JSON raises the decoder's own error, and an unreadable file the reader's."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"schema", "version", "note", "labels"}:
        raise ValueError(f"labels.table_unreadable: {_TABLE}: the table has other fields")
    if (
        value["schema"] != "alphalattice.labels"
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise ValueError(f"labels.table_unreadable: {_TABLE}: not version 1 of alphalattice.labels")
    rows = value["labels"]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"labels.table_unreadable: {_TABLE}: the table has no labels")
    return LabelTable(
        table_schema="alphalattice.labels",
        version=1,
        note=_words(value["note"], "the note"),
        labels=tuple(_label(row, index) for index, row in enumerate(rows)),
    )


@cache
def _installed() -> tuple[LabelTable, dict[str, Label]]:
    """The installed label table and its labels by id, read and indexed once."""
    table = _read(_TABLE.read_text(encoding="utf-8"))
    return table, {value.id: value for value in table.labels}


def label_table() -> LabelTable:
    """The installed label table, read once."""
    return _installed()[0]


def label(code: str) -> Label | None:
    """The label of one installed id, or none when the table names no such id."""
    return _installed()[1].get(code)


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
    known = _installed()[1]
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
