"""A goal is found again by its declaration's typed, versioned binary hash and its opener."""

from __future__ import annotations

from datetime import date

import pytest

from alphalattice.interface.local_application import goals
from alphalattice.interface.local_application.goals import GoalDeclaration, declaration_hash

TEST_SOURCE = "src/alphalattice/interface/local_application/goals.py"
DECLARED = {"title": "Value", "objective": "Does value pay?", "kind": "RESEARCH"}
DECLARED |= {"criteria": [{"criterion_id": "c1", "text": "A result."}], "constraints": ["a", "b"]}


def _hash(opener: object = "HUMAN", **changed: object) -> str:
    return declaration_hash(GoalDeclaration.model_validate(DECLARED | changed), opener)


@pytest.mark.proves("local_application.a_goal_is_found_by_its_declaration_hash")
@pytest.mark.parametrize(
    ("opener", "changed", "equal"),
    [
        pytest.param("HUMAN", {}, True, id="same"),
        pytest.param({"session_id": "s1", "vendor": "codex"}, {}, False, id="opener"),
        pytest.param("HUMAN", {"title": "Momentum"}, False, id="title"),
        pytest.param("HUMAN", {"target_date": date(2026, 11, 16)}, False, id="date"),
        pytest.param("HUMAN", {"constraints": ["b", "a"]}, False, id="order"),
    ],
)
def test_equal_declarations_and_openers_hash_equal(opener, changed, equal) -> None:
    """The same declaration and opener give one hash; another opener, field or order another."""
    assert (_hash() == _hash(opener, **changed)) is equal


@pytest.mark.proves("local_application.a_goal_is_found_by_its_declaration_hash")
def test_the_encoding_version_is_part_of_the_hash(monkeypatch) -> None:
    """A new encoding version names every declaration anew; a type it does not know is refused."""
    before = _hash()
    monkeypatch.setattr(goals, "DECLARATION_ENCODING", 2)
    assert _hash() != before
    with pytest.raises(TypeError, match=r"goal\.declaration_untyped"):
        goals._typed(object())
