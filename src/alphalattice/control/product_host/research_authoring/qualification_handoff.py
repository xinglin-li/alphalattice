"""Bind a family of development studies to the installed Alpha qualification (GR3).

The family is the Host's: every Alpha development study on the question admitted from the
goal's opening on, read from Task Control whether a goal's session ran it or not. The document
names the goal, one study on the question and the candidates it nominates; the envelope is the
question's own, with a budget the qualification's calls fit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from alphalattice.capabilities.alpha_modeling.catalog import build_installed_alpha_model_catalog
from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    PreparedInputAuthority,
)
from alphalattice.control.product_host.research_authoring.model_extensions import (
    activated_models,
)
from alphalattice.control.research_program.authoring.dispatcher import SealedSubmission
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.investment.alpha_research.candidates.qualification_task import (
    AlphaFamilyQualification,
)
from alphalattice.investment.alpha_research.experiments.family_qualification import (
    METHOD,
    AlphaQualificationFamily,
    AlphaQuestionPreparation,
    planned_calls,
    qualification_section,
)
from alphalattice.investment.alpha_research.experiments.mandate import (
    build_current_alpha_research_model_mandate,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
)


def qualification_document(
    document: Mapping[str, Any],
    question_envelope: Mapping[str, Any],
    family: AlphaQualificationFamily,
) -> dict[str, Any]:
    """The document a qualification runs: the question's envelope with the calls it needs."""
    section = qualification_section(document)
    if section is None:
        raise AuthoringError(
            "alpha_research.qualification_declaration_invalid:methodology_id",
            expected={"methodology_id": METHOD},
        )
    nominated = section.get("nominated_candidate_ids")
    count = len(nominated) if isinstance(nominated, list | tuple) else 0
    return {
        "experiment": {
            **question_envelope,
            "budget": {
                "maximum_candidates": len(family.members),
                "maximum_numerical_calls": planned_calls(family, ("",) * count),
            },
        },
        "alpha": dict(section),
    }


def qualification_workflow(
    *,
    workspace: Path,
    document: dict[str, Any],
    family: AlphaQualificationFamily,
    prepare_question: Callable[[], AlphaQuestionPreparation],
    authority: ResolvedResearchAuthority,
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[ResearchProgramWorkflow, SealedSubmission, dict[str, Any]]:
    """The workflow, the sealed submission and the preview of one qualification.

    A family holding an agent's model a person activated is qualified with the rest: its current
    refit seals the state its adapter projects.
    """
    catalog = build_installed_alpha_model_catalog(activated_models(workspace))
    method = AlphaFamilyQualification(
        family=family,
        prepare_question=prepare_question,
        workspace=workspace,
        model_mandate=build_current_alpha_research_model_mandate(catalog=catalog),
        model_catalog=catalog,
        cancelled=cancelled,
    )
    nominated = method.validate_declaration(document)
    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    # The question's own authority: the envelope differs from its study's in the budget alone,
    # and the handles and sessions it resolved are the family's.
    workflow = build_research_program_workflow(
        workspace=workspace / "artifacts" / "alpha-research",
        workspace_root=workspace,
        executors=(method,),
        compilers=(method,),
        authority=PreparedInputAuthority(envelope, authority),
        verifier_kinds=(method.kind,),
    )
    sealed = workflow.prepare(document, actor_kind=ActorKind.HUMAN, actor_id="preview")
    preview = {
        # The question's admitted sessions: the window every study of the family read.
        "statistical_start": str(authority.sessions[0]),
        "statistical_end": str(authority.sessions[-1]),
        "statistical_session_count": len(authority.sessions),
        "interval_semantics": "THE_QUESTIONS_ADMITTED_SESSIONS",
        "listing_count": len(authority.ordered_listing_ids),
        "methodology_id": "ALPHA_FAMILY_QUALIFICATION",
        "question_hash": family.question_hash,
        "goal_id": family.goal_id,
        "family_opened_at": family.opened_at.isoformat(),
        "family_hash": family.content_hash,
        "study_count": len(family.members),
        "attempted_candidate_ids": list(family.candidate_ids),
        "unfinished_task_ids": list(family.unfinished_task_ids),
        "nominated_candidate_ids": list(nominated),
        "expected_numerical_calls": planned_calls(family, nominated),
        "sealed_holdout": "UNREAD",
    }
    return workflow, sealed, preview


__all__ = ["qualification_document", "qualification_workflow"]
