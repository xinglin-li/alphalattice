"""What waits on a person, and what the agent carries, in one read (binding plan, N5).

A Task that stopped and needs a person's choice; a data issue awaiting the authorized caller's
confirmation; an upgrade not yet acknowledged; a Factor study nobody has curated; an explored study
nobody has promoted; a review sealed under an earlier Evidence binding; the CRO's latest
review of a book when its recommendation asks a person to act; a workspace with no verified
research input; data newer than the newest input; a PLAN previewed and not run (V45). Each owner
already
answers its own part and names its next request; this reads those answers, never recomputes
them, and lists each decision with the request that takes it. Nothing here verifies evidence
or starts work.

Each item says whom it waits on (STOPS-1, the person's hands-off rule). A step only a person
may take waits on the PERSON; a step the agent may take, by its own operations or the
first-use goal's delegation, waits on the AGENT, which takes it and tells the person.
Advice stops nothing: the agent reports it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final
from uuid import UUID

from alphalattice.control.product_host.composition.task_recovery import (
    TaskAttentionFact,
    task_attention,
)
from alphalattice.control.task_control.contracts import TaskRecord

PERSON: Final = "PERSON"
AGENT: Final = "AGENT"

WAITS_ON: Final[dict[str, str]] = {
    "FIRST_USE": AGENT,
    "INPUT_VERSION": PERSON,
    "STOPPED_TASK": AGENT,
    "UPGRADE": AGENT,
    "PLAN_PREVIEW": AGENT,
    "CURATION": AGENT,
    "PROMOTION": AGENT,
    "REVIEW_CHANGED": AGENT,
    "CRO_RECOMMENDATION": AGENT,
}
"""Whom each kind waits on. A workspace's preparation and its data issues wait on the agent
while a first-use goal delegates them, else on the person (`pending_decisions`)."""


def _tasks(
    tasks: Iterable[TaskRecord], attention: Mapping[UUID, TaskAttentionFact]
) -> list[dict[str, Any]]:
    return [
        {
            "kind": "STOPPED_TASK",
            "task_id": str(task.task_id),
            "task_kind": task.task_kind,
            "lifecycle": task.lifecycle.value,
            "attention": fact.model_dump(mode="json"),
            "detail": "This Task stopped; its recovery view says what stopped it and what may "
            "resume it, and the agent resumes it where it may.",
            "next_requests": {
                "recovery": {"operation": "TASK_RECOVERY", "task_id": str(task.task_id)}
            },
        }
        for task in tasks
        if (fact := attention.get(task.task_id) or task_attention(task)).unresolved
    ]


def _data_issues(readback: Mapping[str, Any]) -> list[dict[str, Any]]:
    offered = readback.get("next_requests") or {}
    cases: dict[str, dict[str, Any]] = {}
    for name, request in offered.items():
        if name.startswith("preview:"):
            _prefix, case, _option = name.split(":", 2)
            cases.setdefault(case, {})[name] = request
    return [
        {
            "kind": "DATA_ISSUE",
            "case_token": case,
            "detail": "A data issue offers options; preview one to see its consequences first.",
            "next_requests": requests,
        }
        for case, requests in sorted(cases.items())
    ]


def all_data_issues(read_page: Callable[[str | None], Mapping[str, Any]]) -> dict[str, Any]:
    """Every page of the data issues' readback, its offered requests merged (V186).

    The owner pages its cases; a case past the first page that needs a person is
    pending too, so every page is read by the owner's own cursor.

    Args:
        read_page: The owner's readback for one cursor, None for the first page.

    Returns:
        One readback whose `next_requests` hold every page's.
    """
    offered: dict[str, Any] = {}
    cursor: str | None = None
    seen: set[str] = set()
    while True:
        page = read_page(cursor)
        offered.update(page.get("next_requests") or {})
        following = page.get("next_cursor")
        if not isinstance(following, str) or following in seen:
            return {"next_requests": offered}
        seen.add(following)
        cursor = following


def _upgrade(overview: Mapping[str, Any]) -> list[dict[str, Any]]:
    # The overview's own standing (V185): an upgrade stands until this one is
    # acknowledged, whatever an earlier acknowledgement recorded.
    offered = overview.get("next_requests") or {}
    if not overview.get("show") or "acknowledge" not in offered:
        return []
    return [
        {
            "kind": "UPGRADE",
            "detail": "The installed code changed since this workspace last acknowledged it; "
            "the overview says what each saved object is now.",
            "next_requests": {
                "overview": {"operation": "UPGRADE_OVERVIEW"},
                "acknowledge": offered["acknowledge"],
            },
        }
    ]


def _reviews(overview: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "REVIEW_CHANGED",
            "review_publication_hash": row.get("review_publication_hash"),
            "review_key": row.get("review_key"),
            "detail": "This review was sealed under an earlier Evidence binding; it reads back as "
            "recorded, and its book can be reviewed again.",
            "next_requests": dict(row.get("next_requests") or {}),
        }
        for row in overview.get("reviews") or []
        if row.get("state") == "CHANGED"
    ]


def _recommendations(overview: Mapping[str, Any]) -> list[dict[str, Any]]:
    # The CRO's newest review of each book (V187), as the CRO chooses it: by
    # publication time, then hash, among the reviews of one book.
    newest: dict[str, Mapping[str, Any]] = {}
    for row in overview.get("reviews") or []:
        key = row.get("book_key") or row.get("review_key")
        order = (str(row.get("published_at")), str(row.get("review_publication_hash")))
        held = newest.get(key) if key is not None else None
        if key is not None and (
            held is None
            or order >= (str(held.get("published_at")), str(held.get("review_publication_hash")))
        ):
            newest[key] = row
    return [
        {
            "kind": "CRO_RECOMMENDATION",
            "review_publication_hash": row.get("review_publication_hash"),
            "review_key": row.get("review_key"),
            "person_action": row["person_action"],
            "detail": "The CRO's latest review of this book recommends a person weigh its cited "
            "sources before relying on it; the agent reports it, and it stops nothing.",
            "next_requests": dict(row.get("next_requests") or {}),
        }
        for _key, row in sorted(newest.items())
        if row.get("state") == "CURRENT" and row.get("person_action")
    ]


def _studies(awaiting: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    return [
        *(
            {
                "kind": "CURATION",
                "task_id": task_id,
                "detail": "This Factor study's factors are not curated yet; an Alpha study builds "
                "on a curation decision.",
                "next_requests": {
                    "curation": {"operation": "EXPERIMENT_CURATION", "task_id": task_id}
                },
            }
            for task_id in awaiting.get("curation", [])
        ),
        *(
            {
                "kind": "PROMOTION",
                "task_id": task_id,
                "detail": "This study ran on a sample of names; its promotion runs the same "
                "declaration on the whole universe.",
                "next_requests": {
                    "promote": {"operation": "EXPERIMENT_PROMOTE", "task_id": task_id}
                },
            }
            for task_id in awaiting.get("promotion", [])
        ),
    ]


def _workspace(
    preparation: Mapping[str, Any], inputs: Mapping[str, Any], data_update: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The workspace's own state, from its owners' answers (V45): no verified research input and
    no preparation under way, or market data newer than the newest input version's end."""
    versions = [v for item in inputs.get("inputs") or () for v in item.get("versions") or ()]
    if not versions:
        if preparation.get("task_id") or preparation.get("status") in {None, "REFUSED"}:
            return []
        return [
            {
                "kind": "WORKSPACE_PREPARATION",
                "detail": "No verified research input exists in this workspace; one explicit "
                "preparation makes it research-ready.",
                "next_requests": {"plan": {"operation": "WORKSPACE_PREPARE_PLAN"}},
            }
        ]
    current = data_update.get("after") or data_update.get("inputs") or {}
    through = str(current.get("market_through") or current.get("data_through") or "")
    newest = max(
        (
            (str(v["end"]), str(item.get("input_id")))
            for item in inputs.get("inputs") or ()
            for v in item.get("versions") or ()
            if v.get("end")
        ),
        default=None,
    )
    if not through or newest is None or through <= newest[0]:
        return []
    receipt = data_update.get("receipt") or {}
    published = receipt.get("after") or {}
    source_task = (
        data_update.get("task_id")
        if data_update.get("task_lifecycle") == "SUCCEEDED"
        and isinstance(data_update.get("plan_hash"), str)
        and bool(data_update["plan_hash"])
        and data_update.get("plan_hash") == receipt.get("plan_hash")
        and published.get("data_through") == through
        and all(
            current.get(key) is not None and current[key] == published.get(key)
            for key in ("data_revision_hash", "panel_hash")
        )
        else None
    )
    return [
        {
            "kind": "INPUT_VERSION",
            **({"task_id": source_task} if source_task else {}),
            "data_through": through,
            "input_through": newest[0],
            "research_input_id": newest[1],
            "detail": "The workspace holds data past its newest research input; a new input "
            "version brings it to research.",
            "next_requests": {
                "plan": {"operation": "RESEARCH_INPUT_PLAN", "research_input_id": newest[1]}
            },
        }
    ]


def _previews(previews: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "PLAN_PREVIEW",
            **preview,
            "detail": "This PLAN was previewed and not run; reading it grants nothing, and "
            "confirming it admits a Task.",
            "next_requests": {
                "preview": {
                    "operation": "EXPERIMENT_PREVIEW_READBACK",
                    "experiment_plan_hash": preview["plan_hash"],
                },
                "run": {
                    "operation": "EXPERIMENT_RUN",
                    "experiment_plan_hash": preview["plan_hash"],
                },
            },
        }
        for preview in previews
    ]


def pending_decisions(
    *,
    tasks: Iterable[TaskRecord],
    awaiting: Mapping[str, list[str]],
    data_issues: Mapping[str, Any],
    overview: Mapping[str, Any],
    previews: Iterable[Mapping[str, Any]] = (),
    preparation: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    data_update: Mapping[str, Any] | None = None,
    first_use: Mapping[str, Any] | None = None,
    attention: Mapping[UUID, TaskAttentionFact] | None = None,
    goal_attribution: Mapping[tuple[str, str], set[str]] | None = None,
) -> dict[str, object]:
    """Compose owner-supplied workspace/task/data/review decisions into one bounded read view.

    Args:
        tasks: Retained task declarations.
        awaiting: Study decisions grouped by owner.
        data_issues: Explicit data owner issues.
        overview: Explicit upgrade/review/recommendation overview.
        previews: Optional prepared decision previews.
        preparation: Optional workspace preparation state.
        inputs: Optional verified input state.
        data_update: Optional data update state.
        first_use: The workspace's first-use goal while its delegation is active (U70).
        attention: Current owner resolution facts, separate from retained Task history.
        goal_attribution: Exact selectors and the Goals their retained requests counted toward.

    Returns:
        Ordered pending decisions, counts by kind and a plain summary; no decision is executed.
    """
    delegated = first_use is not None
    items = [
        *_first_use(first_use),
        *_workspace(preparation or {}, inputs or {}, data_update or {}),
        *_tasks(tasks, attention or {}),
        *_data_issues(data_issues),
        *_upgrade(overview),
        *_previews(previews),
        *_studies(awaiting),
        *_reviews(overview),
        *_recommendations(overview),
    ]
    counts: dict[str, int] = {}
    for item in items:
        goal_ids = {str(item["goal_id"])} if item["kind"] == "FIRST_USE" else set()
        for key in ("task_id", "plan_hash", "review_publication_hash", "review_key", "case_token"):
            if (value := item.get(key)) is not None:
                goal_ids.update((goal_attribution or {}).get((key, str(value)), ()))
        if item["kind"] == "DATA_ISSUE":
            for request in (data_issues.get("next_requests") or {}).values():
                if (
                    request.get("operation") == "DATA_ISSUE_DELEGATE"
                    and request.get("data_issue_case_token") == item["case_token"]
                ):
                    goal_ids.update(
                        (goal_attribution or {}).get(("task_id", str(request["task_id"])), ())
                    )
        item["goal_ids"] = sorted(goal_ids)
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
        item["waits_on"] = WAITS_ON.get(item["kind"], AGENT if delegated else PERSON)
    person = sum(item["waits_on"] == PERSON for item in items)
    return {
        "status": "PENDING_DECISIONS",
        "decisions": items,
        "counts": counts,
        "detail": (
            f"{person} decision{' waits' if person == 1 else 's wait'} on a person."
            if person
            else "Nothing waits on a person."
        ),
    }


def _first_use(first_use: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The person's first use, run by their agent: its steps taken and the Stop (U70, V452)."""
    if first_use is None:
        return []
    return [
        {
            "kind": "FIRST_USE",
            **first_use,
            "detail": "Your agent runs your first use from your sentence: each step it takes "
            "for you is listed here, until its hours end; Stop ends it now.",
            "next_requests": {
                "stop": {
                    "operation": "GOAL_ABANDON",
                    "goal_id": first_use["goal_id"],
                    "change_reason": "Stopped by the person in the Workbench",
                }
            },
        }
    ]


__all__ = ["all_data_issues", "pending_decisions"]
