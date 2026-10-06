"""What a research flow needs before it runs, what the workspace holds of it, what to ask next.

An agent's way from a research intent to the right request cost more than the work (V367: AX11
spent 26 CLI launches before its first goal opened, re-reading help, schemas, controls and
handoffs to find each entry and its prerequisites; AX10 built a feature before it learned that
its trial needed an Alpha study). The flows' order is fixed -- a Factor study; an Alpha study
from a curated Factor study; a Risk study beside them on the same input; a book from an Alpha
study, sized by a Risk study when it names one; the book's Evidence and CRO review; a formula
factor's trial against an Alpha study handed off from a Factor study -- so each flow's
prerequisite results are known. Every controls answer, and every refusal that means one is
missing, states them for its input: the completed results the workspace holds, newest first and
at most five of each, the one the flow still needs, and the requests allowed next, each ready
to send. Nothing is created or chosen here: a request that leaves a choice names it as None.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final, Literal

from alphalattice.foundation.factor_research.experiments.authoring import FACTOR_EXPERIMENT_KIND
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FACTOR_DEVELOPMENT_CURATION_CATEGORY,
)
from alphalattice.investment.alpha_research.experiments.authoring import ALPHA_EXPERIMENT_KIND
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND

Flow = Literal["FACTOR_STUDY", "ALPHA_STUDY", "RISK_STUDY", "BOOK", "FEATURE_TRIAL", "BOOK_REVIEW"]

FLOW_OF_KIND: Final[Mapping[str, Flow]] = {
    FACTOR_EXPERIMENT_KIND: "FACTOR_STUDY",
    ALPHA_EXPERIMENT_KIND: "ALPHA_STUDY",
    RISK_EXPERIMENT_KIND: "RISK_STUDY",
}
"""The flow an experiment kind's controls begin."""

FLOW_OF_REFUSAL: Final[Mapping[str, Flow]] = {
    "research_experiment.factor_parent_required": "ALPHA_STUDY",
    "factor_research.completed_experiment_required": "ALPHA_STUDY",
    "research_experiment.prepared_features_begin_with_factor": "FACTOR_STUDY",
    "portfolio_research.alpha_parent_required": "BOOK",
    "portfolio_research.alpha_development_required": "BOOK",
    "portfolio_research.completed_alpha_required": "BOOK",
    "portfolio_research.risk_study_required": "BOOK",
    "risk_report.completed_risk_required": "BOOK",
    "risk_report.completed_portfolio_required": "BOOK_REVIEW",
}
"""The refusals that mean a flow's prerequisite result is missing, and the flow they stop."""

FLOWS: Final[tuple[Flow, ...]] = (
    "FACTOR_STUDY",
    "ALPHA_STUDY",
    "RISK_STUDY",
    "BOOK",
    "BOOK_REVIEW",
    "FEATURE_TRIAL",
)
"""The standard flows, in the research order (V376): Factor, then Alpha, Risk beside them, a
book from both, its review, and a formula factor's trial."""

_SHOWN: Final = 5
_NEEDS: Final[Mapping[Flow, tuple[str, ...]]] = {
    "FACTOR_STUDY": (),
    "RISK_STUDY": (),
    "ALPHA_STUDY": ("CURATED_FACTOR_STUDY",),
    "BOOK": ("ALPHA_STUDY",),
    "FEATURE_TRIAL": ("ALPHA_STUDY",),
    "BOOK_REVIEW": ("BOOK",),
}
"""Each flow's prerequisite results on its input. A book sized by risk also names a Risk study,
which the book's declaration chooses (`portfolio.risk_task_id`); it is shown, never required."""

_SHOWS: Final[Mapping[Flow, tuple[str, ...]]] = {
    "FACTOR_STUDY": ("FACTOR_STUDY",),
    "RISK_STUDY": ("RISK_STUDY",),
    "ALPHA_STUDY": ("FACTOR_STUDY", "ALPHA_STUDY"),
    "BOOK": ("ALPHA_STUDY", "RISK_STUDY", "FACTOR_STUDY"),
    "FEATURE_TRIAL": ("ALPHA_STUDY", "FACTOR_STUDY"),
    "BOOK_REVIEW": ("BOOK", "ALPHA_STUDY"),
}
"""The results each flow's answer lists: its prerequisites and what opens them."""

_DETAIL: Final[Mapping[Flow, tuple[str, str]]] = {
    "FACTOR_STUDY": (
        "A Factor study needs only its research input; plan the controls' declaration.",
        "A Factor study needs only its research input; plan the controls' declaration.",
    ),
    "RISK_STUDY": (
        "A Risk study needs only its research input, beside the Factor and Alpha studies on it; "
        "plan the controls' declaration.",
        "A Risk study needs only its research input, beside the Factor and Alpha studies on it; "
        "plan the controls' declaration.",
    ),
    "ALPHA_STUDY": (
        "An Alpha study begins from a curated Factor study on its input; preview a curated "
        "one's handoff and plan the Alpha declaration it returns.",
        "An Alpha study begins from a curated Factor study on its input, and this input has "
        "none: {way}",
    ),
    "BOOK": (
        "A book is drafted from a completed Alpha study on its input, one candidate chosen; "
        "name a Risk study as `portfolio.risk_task_id` to size it by risk.",
        "A book is drafted from a completed Alpha study on its input, and this input has none: "
        "{way}",
    ),
    "FEATURE_TRIAL": (
        "A formula factor's trial runs against a completed Alpha study handed off from a Factor "
        "study on its input; the feature plan's answer offers the trial with the study to "
        "choose.",
        "A formula factor's trial runs against a completed Alpha study handed off from a Factor "
        "study on its input, and this input has none: {way}",
    ),
    "BOOK_REVIEW": (
        "A book's Evidence and CRO review reads a completed book; its readback offers the "
        "review, the dossier and the CRO's bundle.",
        "A book's Evidence and CRO review reads a completed book, and this input has none: {way}",
    ),
}
"""Each flow's sentence: its prerequisites met, and missing, `{way}` the way to them."""

_WAY: Final[Mapping[str, str]] = {
    "factor": "run a Factor study first (its controls are offered).",
    "curation": "curate a Factor study (offered), then preview its handoff.",
    "handoff": "preview a curated Factor study's handoff (offered) and plan its Alpha study.",
    "portfolio-draft": "draft a book from an Alpha study (offered).",
}


def holdings(
    tasks: Iterable[Any],
    plan_of: Callable[[Any], Any],
    *,
    binding_hash: str,
    workspace: Path,
) -> dict[str, list[dict[str, Any]]]:
    """The completed studies on one input, by result, newest first, at most five of each.

    Args:
        tasks: The workspace's research-experiment Tasks that succeeded, any order.
        plan_of: A Task's experiment plan.
        binding_hash: The input revision the flow runs on.
        workspace: The workspace, whose study folders hold the Factor studies' curations.

    Returns:
        `FACTOR_STUDY` (each its curation receipts), `ALPHA_STUDY` (each a development study
        handed off from a Factor study, with that study), `RISK_STUDY` and `BOOK` (each with its
        Alpha study); and `CURATED_FACTOR_STUDY`, the newest curated Factor studies among them
        all, since a prerequisite is judged over every study and only shown for the newest
        (V497).
    """
    held: dict[str, list[dict[str, Any]]] = {
        "FACTOR_STUDY": [],
        "ALPHA_STUDY": [],
        "RISK_STUDY": [],
        "BOOK": [],
        "CURATED_FACTOR_STUDY": [],
    }
    for task in sorted(tasks, key=lambda value: value.admitted_at, reverse=True):
        plan = plan_of(task)
        if plan.binding.binding_hash != binding_hash:
            continue
        kind = plan.program.kind
        row: dict[str, Any] = {"task_id": str(task.task_id)}
        if plan.portfolio_source is not None:
            result = "BOOK"
            row["alpha_task_id"] = str(plan.portfolio_source.alpha_task_id)
        elif kind == FACTOR_EXPERIMENT_KIND:
            result = "FACTOR_STUDY"
            folder = workspace / plan.document["experiment"]["output_workspace"]
            row["curation_receipt_hashes"] = sorted(
                path.stem
                for path in (folder / FACTOR_DEVELOPMENT_CURATION_CATEGORY).rglob("*.json")
            )
            if row["curation_receipt_hashes"] and len(held["CURATED_FACTOR_STUDY"]) < _SHOWN:
                held["CURATED_FACTOR_STUDY"].append(row)
        elif kind == ALPHA_EXPERIMENT_KIND and plan.alpha_source is not None:
            result = "ALPHA_STUDY"
            row["factor_task_id"] = str(plan.alpha_source.factor_task_id)
        elif kind == RISK_EXPERIMENT_KIND:
            result = "RISK_STUDY"
        else:
            continue
        if len(held[result]) < _SHOWN:
            held[result].append(row)
    return held


def prerequisites(
    flow: Flow,
    held: Mapping[str, list[dict[str, Any]]],
    *,
    input_id: str,
    binding_hash: str,
) -> dict[str, Any]:
    """A flow's prerequisites on its input: what it needs, holds and misses, what to ask next.

    Args:
        flow: The flow the answer begins or refused.
        held: The input's completed studies (`holdings`).
        input_id: The research input.
        binding_hash: Its revision.

    Returns:
        `flow`, `needs`, `present` (the results the flow reads, by result), `missing`, `detail`
        and `next_requests`, each request ready to send, a choice it leaves named as None.
    """
    factors = held.get("FACTOR_STUDY", [])
    # Over every study, not the newest shown: an older curation still opens Alpha (V497).
    curated = held.get("CURATED_FACTOR_STUDY", [])
    alphas = held.get("ALPHA_STUDY", [])
    books = held.get("BOOK", [])
    selector = {"research_input_id": input_id, "input_binding_hash": binding_hash}

    def toward_alpha() -> dict[str, dict[str, Any]]:
        if curated:
            receipts = curated[0]["curation_receipt_hashes"]
            return {
                "handoff": {
                    "operation": "EXPERIMENT_HANDOFF_PREVIEW",
                    "task_id": curated[0]["task_id"],
                    # Several curations of one study are a choice, left to make.
                    "curation_receipt_hash": receipts[0] if len(receipts) == 1 else None,
                }
            }
        if factors:
            return {
                "curation": {"operation": "EXPERIMENT_CURATION", "task_id": factors[0]["task_id"]}
            }
        return {
            "factor": {
                "operation": "EXPERIMENT_CONTROLS",
                **selector,
                "experiment_kind": FACTOR_EXPERIMENT_KIND,
            }
        }

    missing: list[str] = []
    following: dict[str, dict[str, Any]]
    if flow in {"FACTOR_STUDY", "RISK_STUDY"}:
        # A study is planned from a declaration: its controls write one to edit, and the plan
        # leaves it to give, a template never printed as runnable (V433).
        following = {
            "controls": {
                "operation": "EXPERIMENT_CONTROLS",
                **selector,
                "experiment_kind": FACTOR_EXPERIMENT_KIND
                if flow == "FACTOR_STUDY"
                else RISK_EXPERIMENT_KIND,
            },
            "plan": {"operation": "EXPERIMENT_PLAN", **selector, "experiment_document": None},
        }
    elif flow == "ALPHA_STUDY":
        missing = [] if curated else ["CURATED_FACTOR_STUDY"]
        following = toward_alpha()
    elif flow in {"BOOK", "FEATURE_TRIAL"}:
        missing = [] if alphas else ["ALPHA_STUDY"]
        following = (
            (
                {
                    "portfolio-draft": {
                        "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
                        "task_id": alphas[0]["task_id"],
                        "candidate_id": None,
                    }
                }
                if flow == "BOOK"
                else {}
            )
            if alphas
            else toward_alpha()
        )
        if flow == "BOOK" and not held.get("RISK_STUDY"):
            following["risk"] = {
                "operation": "EXPERIMENT_CONTROLS",
                **selector,
                "experiment_kind": RISK_EXPERIMENT_KIND,
            }
    else:
        missing = [] if books else ["BOOK"]
        following = (
            {"book": {"operation": "EXPERIMENT_READBACK", "task_id": books[0]["task_id"]}}
            if books
            else {
                "portfolio-draft": {
                    "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
                    "task_id": alphas[0]["task_id"],
                    "candidate_id": None,
                }
            }
            if alphas
            else toward_alpha()
        )
    met, unmet = _DETAIL[flow]
    way = next((_WAY[name] for name in following if name in _WAY), "")
    return {
        "flow": flow,
        "needs": list(_NEEDS[flow]),
        "present": {result: list(held.get(result, [])) for result in _SHOWS[flow]},
        "missing": missing,
        "detail": unmet.format(way=way) if missing else met,
        "next_requests": following,
    }


__all__ = ["FLOWS", "FLOW_OF_KIND", "FLOW_OF_REFUSAL", "Flow", "holdings", "prerequisites"]
