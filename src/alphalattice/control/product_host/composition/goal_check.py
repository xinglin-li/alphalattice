"""The Host's check of a goal's submission: only what it can decide (LAWS OP13)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError

from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.publication.goals import GoalRefusal
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.failure_codes import located_failure, within
from alphalattice.interface.local_application.goals import (
    Goal,
    GoalDeclaration,
    GoalReferenceRequest,
    GoalStatement,
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
    goal: Goal,
    submission: GoalSubmission,
    missing: list[dict[str, Any]],
    offered: dict[str, Any],
    *,
    inline: bool = False,
) -> dict[str, Any]:
    """The resubmit an INCOMPLETE answer offers: the submission as the Host can correct it.

    Left open when nothing is correctable; bound to the goal's revision unless an offered
    attach revises the goal first.
    """
    request: dict[str, Any] = {
        "operation": "GOAL_SUBMIT",
        "goal_id": str(goal.goal_id),
        "goal_submission": corrected(submission, missing, offered),
    }
    if inline:
        document = request["goal_submission"] or submission.model_dump(mode="json")
        held = {r["reference_id"] for r in document.get("references", [])}
        document["references"] = document.get("references", []) + [
            v["goal_reference"]
            for v in offered.values()
            if v["goal_reference"]["reference_id"] not in held
        ]
        request["goal_submission"] = document
    if not offered or inline:
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


def goal_refusal(
    error: Exception,
    request: PortfolioResearchOperationRequest,
    *,
    attributed_goal_id: str | None = None,
) -> dict[str, object]:
    """A Goal refusal with exact read and recovery forms, reusing its owner's snapshot."""
    failure = located_failure(error, "goal.refused")
    code = str(failure["failure_code"])
    answer: dict[str, object] = {
        "status": "REFUSED",
        **failure,
        **refusal_words(code),
        **explain(code),
    }
    if code.partition(":")[0] == "goal.closed_open_a_follow_up":
        named = {
            name: str(value)
            for name, value in (
                ("goal_id", attributed_goal_id or request.goal_id),
                ("goal_hash", request.goal_hash),
            )
            if value is not None
        }
        offered = cast(
            dict[str, object],
            answer.get("next_requests") or {"schema": {"operation": "GOAL_SCHEMA"}},
        )
        offers = {"show": {"operation": "GOAL_SHOW", **named}, **offered}
        if attributed_goal_id is not None:
            _required, allowed = request.field_contract(request.operation)
            retry = PortfolioResearchRequestDocument.model_validate(
                {
                    "operation": request.operation,
                    **{name: getattr(request, name) for name in allowed},
                }
            ).model_dump(mode="json", exclude_none=True, exclude_defaults=True)
            offers["retry"] = retry
        answer["next_requests"] = offers
    elif code == "goal.conclusion_requires_evidence" and isinstance(error, GoalRefusal):
        goal = error.goal
        identity = {"goal_id": str(goal.goal_id), "goal_hash": goal.goal_hash}
        answer["expected"] = {"attached_reference_ids": [r.reference_id for r in goal.references]}
        answer["next_requests"] = {
            "show": {"operation": "GOAL_SHOW", "goal_id": str(goal.goal_id)},
            "schema": {"operation": "GOAL_SCHEMA"},
            "attach": {
                "operation": "GOAL_ATTACH",
                **identity,
                "goal_reference": None,
                "change_reason": request.change_reason,
            },
            "note": {
                "operation": "GOAL_NOTE",
                **identity,
                "change_reason": request.change_reason,
                "goal_statement": {**(request.goal_statement or {}), "evidence": None},
            },
        }
    elif code == "goal.reference_not_found" and request.goal_hash is not None:
        answer["next_requests"] = {
            "narrative": {"operation": "GOAL_NARRATIVE", "goal_hash": request.goal_hash}
        }
    elif code == "goal.revision_conflict_read_latest" and request.goal_id is not None:
        # Read this Goal's head: its hash is a replaced revision, not the session's own Goal.
        offered = cast(dict[str, object], answer.get("next_requests") or {})
        answer["next_requests"] = {
            "show": {"operation": "GOAL_SHOW", "goal_id": str(request.goal_id)},
            **offered,
        }
    return answer


def completion_template(goal: Goal, references: Sequence[dict[str, Any]] = ()) -> str:
    """The open goal's completion to fill: the Host's slots prefilled, the judgment left.

    A GOAL_SUBMIT bound to the goal's revision: each criterion and deliverable slot by
    id, its words as a comment, the references to cite; the judgment is a placeholder the
    contract refuses until written.

    Args:
        goal: The goal's head revision.
        references: Exact owner-selected reads to submit inline.

    Returns:
        The request's YAML text, `request --file` sends it once filled.
    """

    def said(text: str) -> str:
        return " ".join(text.split())[:160]

    def quoted(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    declared = goal.declaration
    selected = [r.reference_id for r in goal.references] + [
        str(r["reference_id"]) for r in references
    ]
    lines = [
        f"# The completion of goal {quoted(said(declared.title))}, revision {goal.revision}: your",
        "# judgment only; the Host adds its own facts (sessions, Tasks, requests). Replace each",
        "# <...>, then `request --file <this file>`: it completes this goal's revision, and a",
        "# later revision refuses it (read the goal again). Evidence names a reference by its id:",
        *(f"#   {ref.reference_id}: {said(ref.label)} ({ref.stage})" for ref in goal.references),
        "# A read the goal does not hold yet goes under `references:` with its own id.",
        "operation: GOAL_SUBMIT",
        f"goal_id: {quoted(str(goal.goal_id))}",
        f"goal_hash: {quoted(goal.goal_hash)}",
        "goal_submission:",
        "  outcome: <ACHIEVED, PARTLY_ACHIEVED or NOT_ACHIEVED>",
        "  summary: <what the goal reached, in your words>",
        "  criteria:",
    ]
    for criterion in declared.criteria:
        lines += [
            f"  # {said(criterion.text)}",
            f"  - criterion_id: {quoted(criterion.criterion_id)}",
            "    answer: <MET, NOT_MET or NOT_ASSESSED>",
            f"    evidence: {json.dumps(selected)}",
            '    note: ""',
        ]
    if declared.deliverables:
        lines.append("  deliverables:")
    for deliverable in declared.deliverables:
        need = "required" if deliverable.required else "optional"
        lines += [
            f"  # {deliverable.kind}, {need}: {said(deliverable.description)}",
            f"  - deliverable_id: {quoted(deliverable.deliverable_id)}",
            f"    references: {json.dumps(selected or ['<reference id>'])}",
        ]
    if references:
        lines += ["  references:", *(f"  - {json.dumps(ref)}" for ref in references)]
    return "\n".join(lines) + "\n"


def goal_schema(operations: Mapping[str, tuple[str, ...]], template: object) -> dict[str, Any]:
    """The existing declaration and submission schema, composed beside its grammar."""
    return {
        "status": "AVAILABLE",
        "grammar": submission_grammar(operations),
        "schemas": {
            "declaration": GoalDeclaration.model_json_schema(),
            "reference": GoalReferenceRequest.model_json_schema(),
            "statement": GoalStatement.model_json_schema(),
            "submission": GoalSubmission.model_json_schema(),
        },
        "reference_operations": operations,
        "template": GoalDeclaration.model_validate(template).model_dump(
            mode="json", exclude_defaults=True
        ),
        "claim": "Contracts and declared budgets are not permission grants; the Host "
        "checks a submission against its own record, never the truth of its summary.",
    }


def offer_completion(
    goal: Goal, body: dict[str, Any], resolve: Callable[[GoalReferenceRequest], dict[str, Any]]
) -> None:
    """Select exact reads once and leave only the author's judgments to fill."""
    import yaml  # type: ignore[import-untyped]

    refs = offered_attach(
        goal,
        [{"code": NO_EVIDENCE}],
        tuple(GoalTaskFact.model_validate(t) for t in body["record"]["tasks"]),
        resolve,
    )
    body["yaml"] = completion_template(goal, [v["goal_reference"] for v in refs.values()])
    if not goal.references and not refs:
        return
    request = yaml.safe_load(body["yaml"])
    request["goal_submission"].update(outcome=None, summary=None)
    for criterion in request["goal_submission"]["criteria"]:
        criterion["answer"] = None
    body["next_requests"]["complete"] = request
    body["detail"] = "goal.closure.ready"
