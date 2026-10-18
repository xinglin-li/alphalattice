"""The Host's check of a goal's submission: only what it can decide (LAWS OP13)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from uuid import UUID

from pydantic import ValidationError

from alphalattice.interface.local_application.failure_codes import within
from alphalattice.interface.local_application.goals import (
    Goal,
    GoalReferenceRequest,
    GoalSubmission,
    GoalTaskFact,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)

# Task Control's lifecycle: a moving Task must end before a goal completes; one waiting on
# someone (a recovery, a review, a block, a deferral) is named in the submission's problems,
# unless a successor that succeeded continued it.
MOVING = frozenset({"QUEUED", "RUNNING", "CANCEL_REQUESTED"})
NO_EVIDENCE = "goal.criterion_evidence_required"
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
            need(NO_EVIDENCE, criterion_id=answer.criterion_id)
        if answer.answer == "NOT_ASSESSED" and not answer.note.strip():
            need("goal.criterion_reason_required", criterion_id=answer.criterion_id)
    outcome = criteria_outcome(submission)
    if outcome is not None and submission.outcome != outcome:
        need("goal.outcome_contradicts_criteria", outcome=outcome)
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
    return missing + _task_items(tasks, named)


def _task_items(tasks: tuple[GoalTaskFact, ...], named: set[UUID]) -> list[dict[str, Any]]:
    """Each Task still moving, or waiting with neither a problem naming it nor a successor."""
    items = []
    for task in tasks:
        where = {"task_id": str(task.task_id), "state": task.state}
        if task.state in MOVING:
            items.append({"code": "goal.task_running", **where})
        elif task.state in WAITING and task.task_id not in named and not task.continued_by:
            items.append({"code": "goal.task_unaccounted", **where})
    return items


def criteria_outcome(submission: GoalSubmission) -> str | None:
    """The outcome a submission's criteria require; None where they leave it to the agent.

    A formula, and so the Host's. Every criterion MET gives ACHIEVED, every one NOT_MET gives
    NOT_ACHIEVED, and MET beside NOT_MET gives PARTLY_ACHIEVED. A criterion NOT_ASSESSED rules
    out only ACHIEVED: the agent judges PARTLY_ACHIEVED or NOT_ACHIEVED, and an ACHIEVED is
    corrected to the one its MET give.
    """
    said = {a.answer for a in submission.criteria}
    if not said:
        return None
    if "NOT_ASSESSED" in said:
        if submission.outcome != "ACHIEVED":
            return None
        return "PARTLY_ACHIEVED" if "MET" in said else "NOT_ACHIEVED"
    if said == {"MET"}:
        return "ACHIEVED"
    return "NOT_ACHIEVED" if said == {"NOT_MET"} else "PARTLY_ACHIEVED"


def corrected(
    submission: GoalSubmission, missing: list[dict[str, Any]], offered: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The submission as the Host can correct it, the agent's words unchanged.

    Its outcome is the one its criteria give, and a criterion citing no evidence, or only
    references the goal lacks, cites each attach offered beside it.

    Args:
        submission: The agent's completion document.
        missing: Its missing items.
        offered: The attaches offered beside it, by `attach:<reference_id>`.

    Returns:
        The corrected document, or None when there is nothing the Host can correct.
    """
    sent = submission.model_dump(mode="json", exclude_unset=True)
    document = submission.model_dump(mode="json", exclude_unset=True)
    for item in missing:
        if item["code"] == "goal.outcome_contradicts_criteria":
            document["outcome"] = item["outcome"]
    cited = [a["goal_reference"]["reference_id"] for a in offered.values()]
    absent = {m["reference_id"] for m in missing if m["code"] == "goal.reference_absent"}
    bare = {m["criterion_id"] for m in missing if m["code"] == "goal.criterion_evidence_required"}
    for answer in document.get("criteria", ()) if cited else ():
        evidence = answer.get("evidence") or []
        if answer["criterion_id"] in bare or (evidence and set(evidence) <= absent):
            answer["evidence"] = cited
    return None if document == sent else document


UPDATE_KIND = "portfolio_public_watermark_advancement"
"""A research update's Task kind: the date's published positions a first use completes on."""


def offered_attach(
    goal: Goal,
    missing: list[dict[str, Any]],
    tasks: tuple[GoalTaskFact, ...],
    resolve: Callable[[GoalReferenceRequest], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Each attach a submission citing an absent reference needs, ready to send as it stands.

    Read from the goal's own Tasks: each succeeded research update's positions (PORTFOLIO) and,
    once published, their review (EVIDENCE_CRO), by a slug of the update's Task.

    Args:
        goal: The open goal.
        missing: The submission's missing items.
        tasks: The Tasks the goal's sessions started.
        resolve: Reads one reference's request at its owner, as an attach would.

    Returns:
        The attach requests by `attach:<reference_id>`; none when no reference is absent.
    """
    if not any(m["code"] in {"goal.reference_absent", NO_EVIDENCE} for m in missing):
        return {}
    held, offered = {r.reference_id for r in goal.references}, {}
    for task in tasks:
        if task.kind != UPDATE_KIND or task.state != "SUCCEEDED":
            continue
        slug, read = str(task.task_id)[:8], {"operation": "RESEARCH_UPDATE_READBACK"}
        refs = [("positions", "PORTFOLIO", read | {"task_id": str(task.task_id)})]
        try:
            probe = {"reference_id": "probe", "label": "probe", "stage": "PORTFOLIO"}
            body = resolve(GoalReferenceRequest(**probe, request=refs[0][2]))
        except (ValueError, KeyError, OSError):
            continue
        standing = body.get("review_standing") or {}
        if standing.get("status") == "REVIEWED" and body.get("review_selector"):
            export = {"operation": "EVIDENCE_CRO_EXPORT", **body["review_selector"]}
            export["review_publication_hash"] = standing["review_publication_hash"]
            refs.append(("review", "EVIDENCE_CRO", export))
        for kind, stage, request in refs:
            reference_id = f"{kind}-{slug}"
            if reference_id not in held:
                offered[f"attach:{reference_id}"] = {
                    "operation": "GOAL_ATTACH",
                    "goal_id": str(goal.goal_id),
                    "change_reason": f"The submission cites the update's {kind}.",
                    "goal_reference": {
                        "reference_id": reference_id,
                        "label": f"The update's {kind}",
                        "stage": stage,
                        "request": request,
                    },
                }
    return offered


def submission_grammar(reference_operations: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    """The words a submission is checked in, short enough to show whole in a compact view.

    Args:
        reference_operations: Each read a reference may cite, with the stages it may stand for.

    Returns:
        The reference id grammar, a problem's fields and each stage's admitted reads.
    """
    stage_reads: dict[str, list[str]] = {}
    for operation, stages in reference_operations.items():
        for stage in stages:
            stage_reads.setdefault(stage, []).append(operation)
    (pattern,) = (m.pattern for m in GoalReferenceRequest.model_fields["reference_id"].metadata)
    return {
        "reference_id": f"{pattern}: an id of the goal's own, never a Task or result id; "
        "attach a stage's read under it first, or declare it in the submission's references.",
        "problem": {
            "problem_id": "An id in the reference_id grammar.",
            "text": "What stayed unresolved.",
            "task_ids": "The whole Task ids it accounts for (`goal show --view full`); a stop a "
            "later Task continued and finished needs none.",
        },
        "stage_reads": stage_reads,
    }


def resubmit(
    goal: Goal, submission: GoalSubmission, missing: list[dict[str, Any]], offered: dict[str, Any]
) -> dict[str, Any]:
    """The resubmit an INCOMPLETE answer offers: the submission as the Host can correct it.

    Left open when nothing is correctable; bound to the goal's revision unless an offered
    attach revises the goal first.
    """
    request = {
        "operation": "GOAL_SUBMIT",
        "goal_id": str(goal.goal_id),
        "goal_submission": corrected(submission, missing, offered),
    }
    if not offered:
        request["goal_hash"] = goal.goal_hash
    return request


def reference_request(document: dict[str, Any]) -> PortfolioResearchOperationRequest:
    """A reference's read as its request, a refusal located under the reference's `request`."""
    try:
        read = PortfolioResearchRequestDocument.model_validate(document)
        request: PortfolioResearchOperationRequest = read.to_operation_request()
    except ValidationError as error:
        raise within(error, "request") from error
    return request
