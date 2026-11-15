"""A goal's research-history events, recorded beside each revision the goal store publishes."""

from __future__ import annotations

from alphalattice.interface.local_application.goals import Goal, GoalStatement
from alphalattice.kernel.data_layer import history

_ADOPTED = {"SUPPORTS": True, "DOES_NOT_SUPPORT": False, "REJECTED": False}
"""A person's response to a conclusion, as feedback: adopted when it supports it."""


def saved(prior: Goal | None, goal: Goal, caller: str) -> None:
    """A new goal's title, with its objective as its text; a statement a revision adds."""
    if prior is None:
        opened(goal, caller)
    elif len(goal.statements) > len(prior.statements):
        noted(str(goal.goal_id), goal.statements[-1])


def opened(goal: Goal, caller: str) -> None:
    """A goal id is its declaration's and opener's digest, so a goal the history already holds
    (another workspace's same declaration, or a workspace made again) is not recorded twice."""
    goal_id = str(goal.goal_id)
    actor = ("person", caller) if caller == "HUMAN" else ("agent", caller)
    title, text = {"title": goal.declaration.title}, goal.declaration.objective
    exists = "SELECT 1 FROM goal WHERE goal_id = ?"
    history.record_new(
        exists, (goal_id,), "goal.set", ("goal", goal_id), title, text=text, actor=actor
    )


def noted(goal_id: str, statement: GoalStatement) -> None:
    """A conclusion as the history's conclusion; the person's response to one as feedback."""
    subject = ("goal", goal_id)
    if statement.kind == "CONCLUSION":
        actor = ("agent", statement.attribution)
        history.record("conclusion.drawn", subject, text=statement.text, actor=actor)
    elif statement.kind == "PM_RESPONSE" and statement.disposition in _ADOPTED:
        adopted = {"adopted": _ADOPTED[statement.disposition]}
        actor = ("person", statement.attribution)
        history.record("feedback.given", subject, adopted, text=statement.text, actor=actor)
