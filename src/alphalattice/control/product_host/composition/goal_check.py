"""The Host's check of a goal's submission: only what it can decide (LAWS OP13)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alphalattice.interface.local_application.goals import Goal, GoalSubmission, GoalTaskFact

# Task Control's lifecycle: a moving Task must end before a goal completes; one waiting on
# someone (a recovery, a review, a block, a deferral) is named in the submission's problems.
MOVING = frozenset({"QUEUED", "RUNNING", "CANCEL_REQUESTED"})
WAITING = frozenset({"DEFERRED", "REVIEW_PENDING", "BLOCKED", "RECOVERY_REQUIRED"})


def missing_items(
    goal: Goal,
    submission: GoalSubmission,
    *,
    tasks: tuple[GoalTaskFact, ...],
    verify: Callable[[str], str | None],
    cited_known: frozenset[str],
) -> list[dict[str, Any]]:
    """Each item a submission lacks, with its code; empty when the record is complete.

    Args:
        goal: The open goal the submission answers.
        submission: The agent's completion document.
        tasks: The Tasks the goal's sessions started, with their lifecycles.
        verify: Re-reads one reference at its owner; answers a failure code, or nothing.
        cited_known: Every reference id the goal and the submission hold.

    Returns:
        The missing items in a stable order. Nothing here judges whether a summary is true.
    """
    missing: list[dict[str, Any]] = []

    def need(code: str, **where: Any) -> None:
        missing.append({"code": code, **where})

    answers = {a.criterion_id: a for a in submission.criteria}
    declared = {c.criterion_id for c in goal.declaration.criteria}
    for criterion_id in sorted(declared - set(answers)):
        need("goal.criterion_unanswered", criterion_id=criterion_id)
    for criterion_id in sorted(set(answers) - declared):
        need("goal.criterion_unknown", criterion_id=criterion_id)
    for answer in submission.criteria:
        if answer.answer in {"MET", "NOT_MET"} and not answer.evidence:
            need("goal.criterion_evidence_required", criterion_id=answer.criterion_id)
        if answer.answer == "NOT_ASSESSED" and not answer.note.strip():
            need("goal.criterion_reason_required", criterion_id=answer.criterion_id)
    if submission.outcome == "ACHIEVED" and any(a.answer != "MET" for a in submission.criteria):
        need("goal.outcome_contradicts_criteria")
    delivered = {d.deliverable_id for d in submission.deliverables}
    for slot in goal.declaration.deliverables:
        if slot.required and slot.deliverable_id not in delivered:
            need("goal.deliverable_missing", deliverable_id=slot.deliverable_id)
    for slot_id in sorted(delivered - {d.deliverable_id for d in goal.declaration.deliverables}):
        need("goal.deliverable_unknown", deliverable_id=slot_id)
    cited = {
        *(ref for a in submission.criteria for ref in a.evidence),
        *(ref for d in submission.deliverables for ref in d.references),
        *(ref for f in submission.findings for ref in f.evidence),
    }
    for reference_id in sorted(cited - cited_known):
        need("goal.reference_absent", reference_id=reference_id)
    for reference_id in sorted(cited & cited_known):
        code = verify(reference_id)
        if code is not None:
            need("goal.reference_unverified", reference_id=reference_id, failure_code=code)
    named = {task for p in submission.problems for task in p.task_ids}
    for task in tasks:
        if task.state in MOVING:
            need("goal.task_running", task_id=str(task.task_id), state=task.state)
        elif task.state in WAITING and task.task_id not in named:
            need("goal.task_unaccounted", task_id=str(task.task_id), state=task.state)
    return missing
