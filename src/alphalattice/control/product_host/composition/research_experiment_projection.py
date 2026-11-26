"""Read-only public facts from an already verified research Task.

This module assembles responses. It does not admit, execute, or verify a Program;
those owners remain in the execution source closure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from alphalattice.control.product_host.research_authoring.authority import (
    exploration_sample_size,
)
from alphalattice.control.task_control.contracts import TaskRecord
from alphalattice.evidence.alternative_evidence.runtime.execution import CpuBudgetStore
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
from alphalattice.protocols.research_authoring.contracts import (
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
)

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.research_experiment_plan import (
        ExperimentPlan,
    )
    from alphalattice.protocols.actor_execution.contracts import ActorSubmissionBinding


def _field_changes(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for key in sorted(before.keys() | after.keys()):
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            changes.extend(
                {**change, "path": [key, *change["path"]]} for change in _field_changes(old, new)
            )
        elif key not in before or key not in after or old != new:
            changes.append(
                {
                    "path": [key],
                    "before_present": key in before,
                    "before": old,
                    "after_present": key in after,
                    "after": new,
                }
            )
    return changes


def _generated_output_change(change: dict[str, Any]) -> bool:
    return change["path"] == ["experiment", "output_workspace"] and all(
        isinstance(change[side], str) and change[side].startswith("research-experiments/")
        for side in ("before", "after")
    )


def plan_impact(
    plan: ExperimentPlan, origin: ExperimentPlan | None, match: TaskRecord | None
) -> dict[str, object]:
    """Explain the sealed declaration's possible work without admitting a Task."""
    kind = plan.program.kind
    all_changes = _field_changes(origin.document, plan.document) if origin is not None else []
    changes = [change for change in all_changes if not _generated_output_change(change)]
    if plan.portfolio_source is not None:
        source = plan.portfolio_source
        upstream: dict[str, object] = {
            "alpha_task_id": source.alpha_task_id,
            "candidate_id": source.candidate_id,
            "score_value_hash": source.score_value_hash,
            "score_ref_count": len(source.score_refs),
            "disposition": "VERIFIED_SAVED_ALPHA_SCORES_NO_REFIT",
        }
        work = "PORTFOLIO_MARKET_PREPARATION_AND_BOOK_REPLAY"
        preserved = ["ALPHA_FIT", "FACTOR_SCREEN", "FEATURE_VALUES", "SOURCE_PRICES"]
    elif plan.alpha_source is not None:
        upstream = {
            "factor_task_id": str(plan.alpha_source.factor_task_id),
            "curation_receipt_hash": plan.alpha_source.curation_receipt_hash,
            "disposition": "SAVED_FACTOR_AND_FOUNDATION_REVALIDATED_AT_RUN",
        }
        work = "ALPHA_FIT_AND_SCORE"
        preserved = ["FACTOR_SCREEN", "FEATURE_VALUES", "SOURCE_PRICES"]
    elif plan.model_training_source is not None:
        upstream = {
            "source_handle": plan.model_training_source.source_handle,
            "disposition": "SEALED_TRAINING_INPUT_REVALIDATED_AT_RUN",
        }
        work = "MODEL_FIT_AND_SCORE"
        preserved = ["SOURCE_PRICES"]
    elif kind == RISK_EXPERIMENT_KIND:
        upstream = {"disposition": "SEALED_RISK_INPUT_REVALIDATED_AT_RUN"}
        work = "RISK_MODEL"
        preserved = ["ALPHA_FIT", "FEATURE_VALUES", "SOURCE_PRICES"]
    else:
        upstream = {"disposition": "SEALED_FACTOR_INPUT_REVALIDATED_AT_RUN"}
        work = "FACTOR_SCREEN"
        preserved = ["FEATURE_VALUES", "SOURCE_PRICES"]
    generated_output = (
        next(
            (
                {"before": change["before"], "after": change["after"]}
                for change in all_changes
                if _generated_output_change(change)
            ),
            None,
        )
        if origin is not None
        else None
    )
    return {
        "source": {
            "input_id": plan.binding.input_id,
            "input_binding_hash": plan.binding.binding_hash,
            "data_snapshot_handle": plan.document["experiment"]["data_snapshot_handle"],
            "upstream": upstream,
        },
        "support": {
            "sessions": plan.document["experiment"]["sessions"],
            "preview": plan.execution_preview,
            "admission": "RUN_REVALIDATES_BINDINGS_AND_SUPPORT",
        },
        "change": {
            "origin_task_id": str(plan.origin_task_id) if plan.origin_task_id else None,
            "declared_fields": changes,
            "generated_output_workspace": generated_output,
            "input_changed": (origin is not None and origin.binding != plan.binding),
            "program_changed": (origin is not None and origin.program != plan.program),
            "implementation_changed": (
                origin is not None and origin.implementation_hash != plan.implementation_hash
            ),
            "origin_result": "PRESERVED_HISTORICAL" if origin is not None else None,
        },
        "execution": {
            "disposition": "EXISTING_EXECUTION_CANDIDATE" if match else "NEW_EXECUTION",
            "task_id": str(match.task_id) if match else None,
            "work_if_new": work,
            "expected_numerical_calls_if_new": plan.execution_preview.get(
                "expected_numerical_calls"
            ),
            "preserved_upstream": preserved,
            "verification": "RUN_REVALIDATES_BEFORE_EXACT_REUSE_OR_ADMISSION",
        },
    }


def method_currency(verifier: object) -> dict[str, object]:
    """What a run would be told about the method a verified study was sealed under.

    A Desk verifier that compares a sealed method with this build's installed
    catalog states its verdict. A recorded readback opens either way, so the
    verdict travels with the result instead of refusing it. Empty for a Desk
    whose verifier states none.
    """
    standing = getattr(verifier, "method_standing", None)
    if standing is None:
        return {}
    return {
        "method_standing": standing,
        "method_refusal": getattr(verifier, "method_refusal", None),
    }


def recorded_work(
    original: ResearchExecutionEvidence, *, current_numerical_calls: int
) -> dict[str, object]:
    """Keep the immutable execution and this operation's work distinct."""
    return {
        "execution_evidence_hash": original.evidence_hash,
        "execution_numerical_call_count": original.numerical_call_count,
        "numerical_call_count": current_numerical_calls,
    }


SAMPLE_SCHEME_SUPERSEDED = "research_lane.sample_scheme_superseded"


def sample_standing(document: Any, authority: ResolvedResearchAuthority) -> dict[str, object]:
    """Read historical exploration sample standing under the installed sampling rule.

    An exploration study whose names were sampled before a sample kept each Sector's
    share reads back as recorded and is not current: the same size now samples other names
    (binding plan, decision 4). Empty for every other study.
    """
    if research_lane(document) != "EXPLORATION" or authority.listing_sample is not None:
        return {}
    return {"method_standing": "NOT_CURRENT", "method_refusal": SAMPLE_SCHEME_SUPERSEDED}


def research_lane(document: Any) -> str:
    """Project the study's declared exploration or promotion lane.

    A study's lane (binding plan, B17): EXPLORATION when it ran on a sample of its
    input's names, else PROMOTION. A Portfolio study carries its Alpha study's universe,
    so an exploration Alpha's Portfolio is exploration too.
    """
    handle = str(document["experiment"]["universe_handle"])
    return "EXPLORATION" if exploration_sample_size(handle) is not None else "PROMOTION"


def lane_fields(document: Any, task_id: UUID | str | None = None) -> dict[str, Any]:
    """The lane a body states, and for an exploration study its way across."""
    lane = research_lane(document)
    if lane == "PROMOTION":
        return {"research_lane": lane}
    return {
        "research_lane": lane,
        "lane_detail": (
            "This study ran on a sample of its input's names to answer fast. Its results "
            "are development evidence and cannot become a strategy until the same "
            "declaration runs on the whole universe."
        ),
        **(
            {"promotion": {"operation": "EXPERIMENT_PROMOTE", "task_id": str(task_id)}}
            if task_id is not None
            else {}
        ),
    }


def realization(store: CpuBudgetStore, task_id: UUID, program_hash: str) -> dict[str, Any]:
    """Read recorded execution settings and machine provenance for one study.

    Where and how a study's numbers were computed: the settings and machine its Task
    started with and, for an Alpha study, the threads its fits used. A record, never an
    identity: the same numbers come from any of them (binding plan, N4).
    """
    task = store.task_execution(task_id)
    fits = store.model_fit_execution(program_hash)
    if task is None and fits is None:
        return {"recorded": False}
    return {
        "recorded": True,
        **(
            {
                "started_at": task.started_at.isoformat(),
                "cpu_budget": task.cpu_budget,
                "cores": task.cores,
                "reader_threads": task.reader_threads,
                "processors": task.machine.processors,
                "platform": None if task.platform is None else task.platform.model_dump(),
            }
            if task is not None
            else {}
        ),
        **({"lightgbm_threads": fits.lightgbm_threads} if fits is not None else {}),
    }


def reuse_words(realization: dict[str, Any]) -> str:
    """B9's reuse wording: what was reused, where and how it was computed."""
    if not realization.get("recorded"):
        return (
            "Reused exactly: this Program's evidence is saved and verified, so nothing was "
            "recomputed. Where and how it was computed was not recorded when it ran."
        )
    platform = realization.get("platform") or {}
    where = (
        " ".join(str(v) for v in (platform.get("system"), platform.get("release")) if v)
        or "this machine"
    )
    how = (
        f"{realization.get('cores')} of {realization.get('processors')} processors"
        if realization.get("cores")
        else "its recorded threads"
    )
    fits = realization.get("lightgbm_threads")
    return (
        f"Reused exactly: computed {str(realization.get('started_at', ''))[:10]} on {where}"
        f" with {how}"
        + (f" and {fits} LightGBM thread{'' if fits == 1 else 's'}" if fits else "")
        + "; nothing was recomputed, and the same numbers come from any setting."
    )


def published_result_header(
    *,
    task_id: UUID,
    plan: ExperimentPlan,
    actor: ActorSubmissionBinding,
    readback: ResearchExecutionEvidence,
    original: ResearchExecutionEvidence,
) -> dict[str, Any]:
    """Project facts only after the caller proved the original Task binding."""
    return {
        "status": "EXPERIMENT_PUBLISHED",
        "evidence_verification": "COMPLETE",
        "task_id": str(task_id),
        "program": plan.program.model_dump(mode="json"),
        "document": plan.document,
        "actor": actor.model_dump(mode="json"),
        "evidence": readback.model_dump(mode="json"),
        "research_input_id": plan.binding.input_id,
        "input_binding_hash": plan.binding.binding_hash,
        **({"origin_task_id": str(plan.origin_task_id)} if plan.origin_task_id else {}),
        **lane_fields(plan.document, task_id),
        **recorded_work(original, current_numerical_calls=readback.numerical_call_count),
    }
