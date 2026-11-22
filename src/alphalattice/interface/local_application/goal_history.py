"""A goal's research-history events, recorded beside each revision the goal store publishes."""

from __future__ import annotations

from typing import Any

from alphalattice.interface.local_application.goals import Goal, GoalStatement
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.log import uuid7

_ADOPTED = {"SUPPORTS": True, "DOES_NOT_SUPPORT": False, "REJECTED": False}
"""A person's response to a conclusion, as feedback: adopted when it supports it."""


def prepare_note(document: dict[str, Any] | None) -> dict[str, Any]:
    """Keep the supplied statement name, or issue one when none was given."""
    value = dict(document or {})
    if value.get("kind") == "HYPOTHESIS":
        if not history.recording():
            raise ValueError("experience.history_unavailable")
        if "statement_id" not in value:
            value["statement_id"] = "s" + uuid7().replace("-", "")
    return value


def replay_note(
    document: dict[str, Any] | None, prior: Goal, store: Any, caller: str, reason: str | None
) -> tuple[Goal, dict[str, Any] | None]:
    """Recover the prior revision and issued statement name after a lost note reply."""
    if not document or document.get("kind") != "HYPOTHESIS":
        return prior, document
    head = store.head(prior.goal_id)
    if head is None or head.parent_hash is None or head.submitted_by != caller:
        return prior, document
    if head.change_reason != reason:
        return prior, document
    if head.goal_hash == prior.goal_hash:
        before = store.load(head.parent_hash)
    elif head.parent_hash == prior.goal_hash:
        before = prior
    else:
        return prior, document
    if (
        len(head.statements) != len(before.statements) + 1
        or head.statements[:-1] != before.statements
    ):
        return prior, document
    last = head.statements[-1]
    candidate = dict(document)
    candidate.setdefault("statement_id", last.statement_id)
    if GoalStatement.model_validate(candidate) == last:
        return before, candidate
    return prior, document


def saved(prior: Goal | None, goal: Goal, caller: str) -> str | None:
    """A new goal's title, with its objective as its text; a statement a revision adds."""
    if prior is None:
        opened(goal, caller)
    elif len(goal.statements) > len(prior.statements):
        return noted(str(goal.goal_id), goal.statements[-1])
    return None


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


def noted(goal_id: str, statement: GoalStatement) -> str | None:
    """A conclusion as the history's conclusion; the person's response to one as feedback."""
    subject = ("goal", goal_id)
    if statement.kind == "HYPOTHESIS":
        return str(
            history.record_hypothesis(
                goal_id, statement.statement_id, statement.text, statement.attribution
            )
        )
    elif statement.kind == "CONCLUSION":
        actor = ("agent", statement.attribution)
        history.record("conclusion.drawn", subject, text=statement.text, actor=actor)
    elif statement.kind == "PM_RESPONSE" and statement.disposition in _ADOPTED:
        adopted = {"adopted": _ADOPTED[statement.disposition]}
        actor = ("person", statement.attribution)
        history.record("feedback.given", subject, adopted, text=statement.text, actor=actor)
    return None
