"""The Alpha owner's qualification Task (GR3, V77).

A family of development studies in, a candidate set or an evidence-complete stop out.

It keeps the family's sealed children in the owner's store by their content names, seals the
qualification Program the Portfolio reads, refits the nominated candidates on the window the
studies trained on, applies the benchmark, Holm across the whole family and the stability rule
(``qualification``), and has the committer seal its end. The sealed holdout stays unread: the
question's fold plan never materializes it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    DeskProgramCompilation,
    NumericalCallRecorder,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

from ..experiments.authoring import ALPHA_EXPERIMENT_KIND
from ..experiments.contracts import AlphaDevelopmentProgram, candidate_id_for_spec
from ..experiments.development_artifacts import AlphaDevelopmentArtifactStore
from ..experiments.family_qualification import (
    CATEGORY,
    METHOD,
    AlphaQualificationFamily,
    AlphaQualificationReceipt,
    AlphaQuestionPreparation,
    alpha_question_hash,
    planned_calls,
    qualification_section,
    read_qualification_receipt,
)
from ..experiments.mandate import AlphaResearchModelMandate
from ..inputs.surfaces import prepare_alpha_program_array_workspace
from ..publication.artifacts import AlphaCurrentArtifactStore
from .artifacts import AlphaGoalResearchArtifactStore
from .committer import AlphaGoalResultCommitter
from .contracts import AlphaGoalDisposition, AlphaResearchProgram, seal_contract
from .control import build_goal_criteria, family_registry, resolve_goal_progress
from .qualification import current_stability_policy_hash, qualify_alpha_model_candidates

_SECTION_KEYS = frozenset(
    {"methodology_id", "goal_id", "question_task_id", "nominated_candidate_ids"}
)


class AlphaFamilyQualification:
    """Compile and execute one qualification; the family is the Host's, never the document's."""

    kind = ALPHA_EXPERIMENT_KIND

    def __init__(
        self,
        *,
        family: AlphaQualificationFamily,
        prepare_question: Callable[[], AlphaQuestionPreparation],
        workspace: Path,
        model_mandate: AlphaResearchModelMandate,
        model_catalog: AlphaModelCatalog,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        """Bind the Host's family, the question's preparation and the models its studies ran."""
        self.family = family
        self.prepare_question = prepare_question
        self.workspace = workspace.resolve()
        try:
            # The models the family ran, never the whole installed menu (V118, V299).
            self.model_mandate = model_mandate.admitting_domains(
                member.recipe.search_domain_hash for member in family.members
            )
        except ValueError as error:
            raise AuthoringError("alpha_research.qualification_model_not_installed") from error
        self.model_catalog = model_catalog
        self.cancelled = cancelled

    @property
    def compiler(self) -> AlphaFamilyQualification:
        """The method compiles its own declarations."""
        return self

    def validate_declaration(self, document: Mapping[str, Any]) -> tuple[str, ...]:
        """The nominated candidates: distinct, and each one the family attempted."""
        section = qualification_section(document)
        # A refusal names the rule it breaks and what may be written (V302, V292).
        if section is None:
            raise AuthoringError(
                "alpha_research.qualification_declaration_invalid:methodology_id",
                expected={"methodology_id": METHOD},
            )
        if set(section) != _SECTION_KEYS:
            raise AuthoringError(
                "alpha_research.qualification_declaration_invalid:alpha",
                expected={"alpha": sorted(_SECTION_KEYS)},
            )
        if str(section["goal_id"]) != self.family.goal_id:
            raise AuthoringError("alpha_research.qualification_family_goal_mismatch")
        nominated = section["nominated_candidate_ids"]
        rule = (
            "empty"
            if not isinstance(nominated, list | tuple) or not nominated
            else "repeated"
            if len(set(nominated)) != len(nominated)
            else "not_in_family"
            if not set(nominated) <= set(self.family.candidate_ids)
            else None
        )
        if rule is not None:
            # The family's candidates are what may be nominated (V292).
            raise AuthoringError(
                f"alpha_research.qualification_nomination_invalid:{rule}",
                expected={"qualification.nominated_candidate_ids": self.family.candidate_ids},
            )
        return tuple(str(value) for value in nominated)

    def method_binding_hash(self) -> str:
        """What the Program binds: the family, the stability rule and the goal's criteria."""
        return str(
            canonical_hash(
                [
                    METHOD,
                    self.family.content_hash,
                    current_stability_policy_hash(),
                    build_goal_criteria(self.model_mandate).criteria_hash,
                ]
            )
        )

    def compile_desk_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        """Seal what the qualification binds: its family, its nominations and the rule."""
        nominated = self.validate_declaration(document)
        if planned_calls(self.family, nominated) > envelope.budget.maximum_numerical_calls:
            raise AuthoringError("alpha_research.qualification_budget_exceeded")
        method = self.method_binding_hash()
        return DeskProgramCompilation(
            desk_program_hash=canonical_hash([authority.authority_hash, method, list(nominated)]),
            catalog_hash=canonical_hash([self.model_mandate.mandate_hash, METHOD]),
            method_binding_hash=method,
            parameter_domain_hash=canonical_hash({"nominated_candidate_ids": list(nominated)}),
        )

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        """Qualify the family and seal its end, a candidate set or a no-stable stop."""
        envelope = ResearchExperimentEnvelope.create(**document["experiment"])
        compiled = self.compile_desk_program(
            envelope=envelope, document=document, authority=authority
        )
        if (
            compiled.desk_program_hash != program.desk_program_hash
            or program.authority_hash != authority.authority_hash
        ):
            raise AuthoringError("alpha_research.qualification_sealed_program_mismatch")
        nominated = self.validate_declaration(document)
        artifact_root = self.workspace / "artifacts"
        current = AlphaCurrentArtifactStore(artifact_root)
        for member in self.family.members:
            if self.cancelled():
                raise AuthoringError("alpha_research.qualification_cancelled_at_checkpoint")
            _import_study(self.workspace / member.output_workspace, current)
        batch_results = tuple(
            current.load_batch_result(member.batch_result_hash) for member in self.family.members
        )
        question = self.prepare_question()
        if alpha_question_hash(question.development_program) != self.family.question_hash:
            raise AuthoringError("alpha_research.qualification_question_changed")
        criteria = build_goal_criteria(self.model_mandate)
        qualification_program = _qualification_program(
            question.development_program,
            family=self.family,
            authority_hash=authority.authority_hash,
            mandate=self.model_mandate,
            criteria_hash=criteria.criteria_hash,
        )
        goal_store = AlphaGoalResearchArtifactStore(artifact_root)
        goal_store.publish_program(qualification_program)
        registry = family_registry(
            program_hash=qualification_program.program_hash,
            recipes={
                candidate_id_for_spec(m.recipe): (m.recipe, r.batch_hash)
                for m, r in zip(self.family.members, batch_results, strict=True)
            },
            batch_results=batch_results,
        )
        fold_plan = question.fold_plan
        # The current refit trains on the window the studies' folds trained on.
        train_sessions = question.development_program.split_policy.train_sessions
        lanes = {
            m.recipe.target_lane for m in self.family.members if m.recipe.target_lane is not None
        }
        with prepare_alpha_program_array_workspace(
            fold_plan, train_session_count=train_sessions
        ) as arrays:
            qualified = qualify_alpha_model_candidates(
                program=qualification_program,
                registry=registry,
                batch_results=batch_results,
                nominated_candidate_ids=nominated,
                # A lane question trains on its lane's plan alone; a lane-free one on the plan.
                fold_plan=None if lanes else fold_plan,
                train_session_count=train_sessions,
                store=current,
                array_workspace=None if lanes else arrays,
                # A lane question's candidates carry its one lane; the fold plan is that lane's.
                fold_plans={lane: fold_plan for lane in lanes} or None,
                array_workspaces={lane: arrays for lane in lanes} or None,
                model_catalog=self.model_catalog,
                model_mandate=self.model_mandate,
            )
        goal_store.publish_registry(qualified.registry)
        if qualified.oos_evidence is not None:
            goal_store.publish_oos_evidence(qualified.oos_evidence)
        goal_store.publish_qualification(qualified.qualification)
        # The family is the whole attempt: no batch follows it, so the progress is terminal.
        progress = resolve_goal_progress(
            criteria=criteria,
            registry=qualified.registry,
            qualification=qualified.qualification,
            completed_batch_count=criteria.max_batch_count,
        )
        goal_store.publish_goal_progress(progress)
        qualified_ids = set(qualified.qualification.current_qualified_ids)
        selected = (
            tuple(value for value in nominated if value in qualified_ids)[
                : criteria.target_current_qualified_candidates
            ]
            if progress.disposition is AlphaGoalDisposition.SATISFIED
            else ()
        )
        committed = AlphaGoalResultCommitter(
            goal_store, numerical_store=current, model_mandate=self.model_mandate
        ).commit(
            program=qualification_program,
            selected_candidate_ids=selected,
            registry=qualified.registry,
            qualification=qualified.qualification,
            progress=progress,
            limitations=_limitations(self.family),
            science_policy_hash=current_stability_policy_hash(),
            completed_batch_count=len(self.family.members),
            criteria=criteria,
        )
        receipt = AlphaQualificationReceipt.create(
            program_hash=program.program_hash,
            authority_hash=authority.authority_hash,
            method_binding_hash=program.method_binding_hash,
            family_hash=self.family.content_hash,
            qualification_program_hash=qualification_program.program_hash,
            registry_hash=committed.marker.registry_hash,
            qualification_hash=committed.marker.qualification_hash,
            marker_hash=committed.marker.marker_hash,
            disposition=committed.marker.disposition,
            candidate_set_hash=committed.marker.candidate_set_snapshot_hash,
            scientific_stop_hash=committed.marker.scientific_stop_hash,
            selected_candidate_ids=selected,
            attempted_candidate_ids=qualified.qualification.attempted_candidate_ids,
            current_qualified_ids=qualified.qualification.current_qualified_ids,
            fit_call_count=qualified.fit_call_count,
            predict_call_count=qualified.predict_call_count,
            metric_call_count=qualified.metric_call_count,
        )
        store = AlphaDevelopmentArtifactStore(output_workspace / "alpha-qualification")
        uri = store._publish_identity_json(
            category=CATEGORY, value=receipt, identity_field="content_hash"
        )
        calls = qualified.fit_call_count + qualified.predict_call_count
        calls += qualified.metric_call_count
        for _ in range(calls):
            if recorder is not None:
                recorder.record(capability="alpha_model.qualification")
        return DeskExecutionResult(
            disposition="COMPUTED",
            artifact_uris=(uri,),
            formation_sessions=authority.sessions,
            numerical_call_count=calls,
            desk_input_binding_hash=authority.authority_hash,
        )


def verify_qualification(
    *,
    program: SealedResearchProgram,
    evidence: ResearchExecutionEvidence,
    output_workspace: Path,
    artifact_root: Path,
) -> AlphaQualificationReceipt:
    """Read the receipt and every terminal child it names, at the Alpha owner's store."""
    receipt = read_qualification_receipt(
        program=program, evidence=evidence, output_workspace=output_workspace
    )
    goal_store = AlphaGoalResearchArtifactStore(artifact_root)
    marker = goal_store.load_marker(receipt.marker_hash)
    goal_store.validate_terminal_lineage(marker)
    if (
        marker.program_hash != receipt.qualification_program_hash
        or marker.registry_hash != receipt.registry_hash
        or marker.qualification_hash != receipt.qualification_hash
        or marker.candidate_set_snapshot_hash != receipt.candidate_set_hash
        or marker.scientific_stop_hash != receipt.scientific_stop_hash
        or marker.disposition != receipt.disposition
    ):
        raise AuthoringError("alpha_research.qualification_marker_mismatch")
    return receipt


def _qualification_program(
    question: AlphaDevelopmentProgram,
    *,
    family: AlphaQualificationFamily,
    authority_hash: str,
    mandate: AlphaResearchModelMandate,
    criteria_hash: str,
) -> AlphaResearchProgram:
    """The Program the qualification seals and the Portfolio reads: the question's identities
    with the mandate narrowed to the models its family ran (V299), the stability rule and the
    family it concludes."""
    return seal_contract(
        AlphaResearchProgram,
        {
            "foundation_hash": question.foundation_hash,
            "logical_panel_hash": question.logical_panel_hash,
            "logical_semantic_index_hash": question.logical_semantic_index_hash,
            "causal_outcome_snapshot_hash": question.causal_outcome_snapshot_hash,
            # The family the qualification concludes stands where a goal's plan stood.
            "pm_plan_hash": family.content_hash,
            "ordered_listing_ids_hash": question.ordered_listing_ids_hash,
            "ordered_factor_ids_hash": question.ordered_feature_ids_hash,
            "split_policy_hash": question.split_policy_hash,
            "metric_policy_hash": question.metric_policy_hash,
            "package_identity_hash": question.package_identity_hash,
            "stability_policy_hash": current_stability_policy_hash(),
            "goal_criteria_hash": criteria_hash,
            "user_authorization_hash": authority_hash,
            "research_goal_hash": canonical_hash({"goal_id": family.goal_id}),
            "model_mandate_hash": mandate.mandate_hash,
            "model_catalog_hash": mandate.catalog_binding.catalog_hash,
            "target_policy_hashes": question.target_policy_hashes,
        },
        "program_hash",
    )


def _limitations(family: AlphaQualificationFamily) -> tuple[str, ...]:
    values = ["QUALIFIED_OVER_THE_WHOLE_ATTEMPTED_FAMILY", "SEALED_HOLDOUT_UNREAD"]
    if family.unfinished_task_ids:
        values.append(f"UNFINISHED_STUDIES_WITHOUT_EVIDENCE:{len(family.unfinished_task_ids)}")
    return tuple(values)


def _import_study(output: Path, current: AlphaCurrentArtifactStore) -> None:
    """Keep a study's sealed children in the Alpha owner's store, by their content names.

    Every file is content-addressed, so a name present already holds the same bytes; a link
    keeps one copy on the volume and a copy stands in where links are not supported.
    """
    source = AlphaDevelopmentArtifactStore(output / "alpha-development").root
    if not source.is_dir():
        raise AuthoringError("alpha_research.qualification_study_evidence_unavailable")
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = current.root / path.relative_to(source)
        if target.exists():
            if target.stat().st_size != path.stat().st_size:
                raise AuthoringError("alpha_research.qualification_import_conflict")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            os.link(path, staged)
        except OSError:
            shutil.copyfile(path, staged)
        os.replace(staged, target)


__all__ = ["AlphaFamilyQualification", "verify_qualification"]
