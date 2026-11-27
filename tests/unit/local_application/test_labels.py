"""The label table is read as labels.json holds it, and refused by name when it does not."""

import json

import pytest

from alphalattice.interface.local_application import labels

TEST_SOURCE = "src/alphalattice/interface/local_application/labels.py"

LABEL = {
    "kind": "strategy",
    "id": "BOOK",
    "title": "Book",
    "summary": "A book.",
    "title_zh": "组合",
    "summary_zh": "一个组合。",
}
TABLE = {"schema": "alphalattice.labels", "version": 1, "note": "Names.", "labels": [LABEL]}


def _changed(**change: object) -> str:
    table = {**TABLE, "labels": [dict(LABEL)]}
    for where, value in change.items():
        if where.startswith("label_"):
            table["labels"][0][where.removeprefix("label_")] = value
        else:
            table[where] = value
    return json.dumps(table)


@pytest.mark.proves("local_application.the_label_table_reads_as_its_schema_or_is_refused_by_name")
@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"version": True},
        {"labels": []},
        {"label_kind": "unknown"},
        {"label_title": ""},
        {"label_extra": "words"},
    ],
    ids=[
        "table-field",
        "boolean-version",
        "no-labels",
        "unknown-kind",
        "empty-title",
        "label-field",
    ],
)
def test_a_table_other_than_its_schema_is_refused_by_name(change: dict[str, object]) -> None:
    """The table reads whole as its schema holds it; a field the schema lacks, a non-integer
    version, no labels, an unknown kind or empty words each refuse it, naming the file."""
    assert labels._read(_changed()).labels == (labels.Label(**LABEL),)
    with pytest.raises(ValueError, match=r"labels\.table_unreadable") as refused:
        labels._read(_changed(**change))
    assert str(labels._TABLE) in str(refused.value)
