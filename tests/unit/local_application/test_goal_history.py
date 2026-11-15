"""Each goal revision's research-history events, as the goal store publishes it."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.interface.local_application import goal_history
from alphalattice.interface.local_application.goals import GoalStatement

TEST_SOURCE = "src/alphalattice/interface/local_application/goal_history.py"


def _statement(kind: str, disposition: str) -> GoalStatement:
    """One statement a revision adds."""
    text = f"A {kind.lower()}."
    return GoalStatement(
        statement_id=kind.lower(),
        kind=kind,
        attribution="Researcher",
        text=text,
        disposition=disposition,
    )


@pytest.mark.proves("local_application.goal_revisions_enter_the_history")
def test_a_goal_its_conclusion_and_the_persons_responses(monkeypatch) -> None:
    """A new goal is recorded once; a conclusion is drawn; a supporting response is adopted
    feedback, a rejecting one is not, and an open one is no feedback."""
    events: list[tuple[Any, ...]] = []
    held = {"g-known"}  # a goal the history already holds

    def record(*event: Any, **more: Any) -> int:
        events.append((*event, more.get("text")))
        return len(events)

    def record_new(exists: str, args: tuple[Any, ...], *event: Any, **more: Any) -> int | None:
        return None if args[0] in held else record(*event, **more)

    monkeypatch.setattr(goal_history.history, "record", record)
    monkeypatch.setattr(goal_history.history, "record_new", record_new)
    declaration = SimpleNamespace(title="A study", objective="Does it survive costs?")
    goal: Any = SimpleNamespace(goal_id="g1", declaration=declaration, statements=())
    goal_history.saved(None, goal, "HUMAN")
    known: Any = SimpleNamespace(goal_id="g-known", declaration=declaration)
    goal_history.saved(None, known, "HUMAN")
    prior = goal
    for kind, disposition in (
        ("CONCLUSION", "OPEN"),
        ("PM_RESPONSE", "SUPPORTS"),
        ("PM_RESPONSE", "REJECTED"),
        ("PM_RESPONSE", "OPEN"),
    ):
        revised: Any = SimpleNamespace(
            goal_id="g1", statements=(*prior.statements, _statement(kind, disposition))
        )
        goal_history.saved(prior, revised, "HUMAN")
        prior = revised
    assert events == [
        ("goal.set", ("goal", "g1"), {"title": "A study"}, "Does it survive costs?"),
        ("conclusion.drawn", ("goal", "g1"), "A conclusion."),
        ("feedback.given", ("goal", "g1"), {"adopted": True}, "A pm_response."),
        ("feedback.given", ("goal", "g1"), {"adopted": False}, "A pm_response."),
    ]
