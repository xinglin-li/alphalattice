"""Current observed questions retain their scope and recorded provenance."""

import pytest

from alphalattice.interface.local_application import questions
from tests.portfolio_strategy_lab.person_questions_support import question_row


@pytest.mark.parametrize("corruption", ["null_goal", "empty_correlation"])
def test_a_later_question_reopens_only_its_goal_or_session(corruption):
    """A prior answer cannot clear a later ask or another Goal's pending question."""

    project = questions.current_person_questions
    history = [
        question_row("old-ask", "ASKED"),
        question_row("old-answer", "ANSWERED"),
        question_row("new-ask", "ASKED"),
    ]
    assert project(history)["pending"]
    assert not project(history[:2])["pending"]
    assert project([*history, question_row("other-answer", "ANSWERED", goal="goal-b")])["pending"]
    assert not project([*history, question_row("new-answer", "ANSWERED")])["pending"]
    unscoped = [question_row("ask", "ASKED", goal=None)]
    assert project([*unscoped, question_row("answer", "ANSWERED", goal=None, session="session-b")])[
        "pending"
    ]
    assert not project([*unscoped, question_row("answer", "ANSWERED", goal=None)])["pending"]
    assert project([question_row("historical", "ASKED", catalog="1" * 64)])["pending"]
    for bad in (
        question_row("bad", "ASKED") | {"correlation_ids": []},
        question_row("bad-hash", "ASKED", catalog="not-a-hash"),
        question_row("bad-scope", "ASKED", goal=42),
        question_row("bad-correlation", "ASKED") | {"correlation_ids": "bad-correlation"},
    ):
        assert project([bad])["invalid"]
    if corruption == "null_goal":
        pair = [question_row("ask", "ASKED"), question_row("answer", "ANSWERED", goal="goal-b")]
        for observed in pair:
            subject = observed["payload"]["subject"]
            refs = subject["person_questions"][0]["references"]
            subject["goal_id"], refs["goal_id"] = refs["goal_id"], None
        assert project(pair)["invalid"]
    else:
        current = project([history[-1], question_row("", "ANSWERED")])
        assert current["pending"] and current["invalid"]
