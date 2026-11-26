"""Prepare an explicit Factor decision for development Alpha, never execute it.

A Task runs the prepared study's fits in a child process (W10): the handoff keeps the arguments
it was prepared from (`FactorHandoffRecipe`), the child prepares it again from them and runs the
executor, and the Host publishes what the child's files make.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import yaml  # type: ignore[import-untyped]

from alphalattice.capabilities.alpha_modeling.runtime.lightgbm_threads import (
    lightgbm_fit_threads,
    lightgbm_threads,
)
from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    ResearchWorkspaceManifest,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
    universe_profile,
)
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    compatible_factor_inputs,
    factor_input_paths,
    logical_panel_owner,
    read_factor_bundle,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.control.task_control.child import ChildInterrupted, ChildRefused, run_in_child
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FactorDevelopmentCurationReader,
    FactorDevelopmentReceipt,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.research_loop.decisions import (
    verify_factor_research_curation,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.investment.alpha_research.experiments.development_execution import (
    AlphaDevelopmentCancelled,
    AlphaExperimentExecutor,
)
from alphalattice.investment.alpha_research.experiments.policies import load_alpha_split_policy
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExecutionResult,
    NumericalCallRecorder,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)


@dataclass(frozen=True)
class FactorHandoffRecipe:
    """The arguments a handoff was prepared from, which a child process prepares it again with.

    Frozen records, documents and paths; the feature input's reader holds paths and its own
    memo, so the recipe crosses to the child by pickle, as spawn hands any argument.
    """

    workspace: Path
    manifest: ResearchWorkspaceManifest
    evidence_root: Path
    original_binding_hash: str
    original_document: dict[str, Any]
    receipt: FactorDevelopmentReceipt
    checkpoint: FactorResearchDeterministicEvidence
    curation_receipt_hash: str
    input_id: str | None
    document: dict[str, Any] | None
    yaml_text: str | None
    available_bindings: tuple[ResearchWorkspaceExperimentInput, ...] | None
    feature_input: ResolvedDevelopmentFeatureInput | None
    recorded_readback: bool

    def prepare(self) -> PreparedFactorHandoff:
        """The handoff these arguments prepare.

        Returns:
            The prepared handoff, its executor built again in this process.
        """
        return prepare_factor_handoff(
            workspace=self.workspace,
            manifest=self.manifest,
            evidence_root=self.evidence_root,
            original_binding_hash=self.original_binding_hash,
            original_document=self.original_document,
            receipt=self.receipt,
            checkpoint=self.checkpoint,
            curation_receipt_hash=self.curation_receipt_hash,
            input_id=self.input_id,
            document=self.document,
            yaml_text=self.yaml_text,
            available_bindings=self.available_bindings,
            recorded_readback=self.recorded_readback,
            feature_input=self.feature_input,
        )


@dataclass
class ChildAlphaExecutor:
    """The handoff's Alpha executor, run in a child process (W10).

    The child prepares the handoff again from its recipe and runs the executor at the Host's
    admitted LightGBM thread count; it reads its inputs and writes the study's content-addressed
    files. A Task's cancellation reaches it at the next fold boundary, relayed by the parent;
    its refusal comes back as a refusal and its cancellation as the Alpha cancellation.
    """

    recipe: FactorHandoffRecipe
    kind: str
    cancelled: Callable[[], bool] | None = field(default=None)

    def execute(
        self,
        *,
        program: SealedResearchProgram,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
        output_workspace: Path,
        recorder: NumericalCallRecorder | None = None,
    ) -> DeskExecutionResult:
        """Run the fits in a child; what the executor returned there.

        Args:
            program: The sealed Program.
            document: Its document.
            authority: The authority it resolved.
            output_workspace: The study's output workspace.
            recorder: Refused: a recorder is a callback into this process.

        Returns:
            The executor's result.

        Raises:
            AlphaDevelopmentCancelled: The Task was cancelled at a fold boundary.
        """
        if recorder is not None:
            raise AuthoringError("research_authoring.child_execution_records_no_calls")
        try:
            result = run_in_child(
                f"{__name__}:execute_alpha_in_child",
                {
                    "recipe": self.recipe,
                    "program": program,
                    "document": document,
                    "authority": authority,
                    "output_workspace": output_workspace,
                    "fit_threads": lightgbm_fit_threads(),
                },
                cancelled=self.cancelled,
            )
        except (ChildRefused, ChildInterrupted) as failure:
            if failure.type_name == AlphaDevelopmentCancelled.__name__:
                raise AlphaDevelopmentCancelled() from None
            raise
        return cast(DeskExecutionResult, result)


def execute_alpha_in_child(
    *,
    recipe: FactorHandoffRecipe,
    program: SealedResearchProgram,
    document: Mapping[str, Any],
    authority: ResolvedResearchAuthority,
    output_workspace: Path,
    fit_threads: int,
    cancelled: Callable[[], bool],
) -> DeskExecutionResult:
    """The child's side: prepare the handoff again and run its executor on the sealed Program.

    Args:
        recipe: What the Host prepared the handoff from.
        program: The sealed Program.
        document: Its document.
        authority: The authority it resolved.
        output_workspace: The study's output workspace.
        fit_threads: The LightGBM thread count the Host admitted for this run.
        cancelled: Reads true once the Host relayed the Task's cancellation.

    Returns:
        The executor's result.
    """
    executor = recipe.prepare().executor
    executor.set_cancellation_check(cancelled)
    with lightgbm_threads(fit_threads):
        return cast(
            DeskExecutionResult,
            executor.execute(
                program=program,
                document=document,
                authority=authority,
                output_workspace=output_workspace,
                recorder=None,
            ),
        )


@dataclass(frozen=True)
class PreparedFactorHandoff:
    """Retain exact curated Factor source, normalized Alpha draft and compiler preflight state."""

    workspace: Path
    source: Path
    binding: ResearchWorkspaceExperimentInput
    document: dict[str, Any]
    curation: dict[str, Any]
    executor: AlphaExperimentExecutor
    missing_fields: tuple[str, ...]
    feature_input: ResolvedDevelopmentFeatureInput | None = None
    recipe: FactorHandoffRecipe | None = None

    def in_child(self, cancelled: Callable[[], bool] | None) -> ChildAlphaExecutor:
        """This handoff's executor, run in a child process by a Task (W10).

        Args:
            cancelled: The Task's cancellation, relayed to the child.

        Returns:
            The executor a Task's workflow installs.
        """
        if self.recipe is None:
            raise AuthoringError("research_authoring.handoff_recipe_unavailable")
        return ChildAlphaExecutor(recipe=self.recipe, kind=self.executor.kind, cancelled=cancelled)

    def workflow(
        self, executor: AlphaExperimentExecutor | ChildAlphaExecutor | None = None
    ) -> ResearchProgramWorkflow:
        """The workflow that runs this handoff, with its own executor or the one given.

        Args:
            executor: The executor installed in place of the handoff's own (a child's).

        Returns:
            The workflow.
        """
        return build_research_program_workflow(
            workspace=self.source,
            workspace_root=self.workspace,
            executors=(executor or self.executor,),
            alpha_compiler=self.executor.compiler,
            verifier_kinds=("alpha.model-development",),
            authority=WorkspaceResearchAuthorityResolver(
                workspace=self.source, feature_input=self.feature_input
            ),
        )


def prepare_factor_handoff(
    *,
    workspace: Path,
    manifest: ResearchWorkspaceManifest,
    evidence_root: Path,
    original_binding_hash: str,
    original_document: dict[str, Any],
    receipt: FactorDevelopmentReceipt,
    checkpoint: FactorResearchDeterministicEvidence,
    curation_receipt_hash: str,
    input_id: str | None,
    document: dict[str, Any] | None,
    yaml_text: str | None,
    available_bindings: tuple[ResearchWorkspaceExperimentInput, ...] | None = None,
    recorded_readback: bool = False,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> PreparedFactorHandoff:
    """Revalidate independently named evidence, then use the shipped compiler."""
    decisions = FactorDevelopmentCurationReader(evidence_root).submissions(
        checkpoint.checkpoint_hash
    )
    decision = next((v for v in decisions if v.receipt_hash == curation_receipt_hash), None)
    if decision is None:
        raise AuthoringError("factor_research.curation_selection_unavailable")
    verify_factor_research_curation(
        decision=decision,
        checkpoint=checkpoint,
        review_binding_hash=receipt.method_binding_hash,
        recorded_readback=recorded_readback,
    )
    selected = {v.factor_id for v in decision.submission.proposal.choices}
    if not selected:
        raise AuthoringError("factor_research.handoff_no_selected_factors")
    if not selected.issubset(receipt.selected_factor_ids):
        raise AuthoringError("factor_research.curation_outside_report_axis")
    axis = tuple(v for v in receipt.selected_factor_ids if v in selected)
    original = read_factor_bundle(workspace, original_binding_hash)
    eligible = []
    for binding in (
        available_bindings if available_bindings is not None else manifest.experiment_inputs or ()
    ):
        if input_id is not None and binding.input_id != input_id:
            continue
        # The Factor input itself is a candidate: its bundle was just verified.
        bundle = (
            original
            if binding.binding_hash == original_binding_hash
            else read_factor_bundle(workspace, binding.binding_hash)
        )
        if compatible_factor_inputs(original, bundle):
            eligible.append((binding, bundle))
    if len(eligible) != 1:
        raise AuthoringError(
            "factor_research.handoff_input_ambiguous"
            if eligible
            else "factor_research.handoff_input_unavailable"
        )
    binding, bundle = eligible[0]
    if feature_input is not None and (
        feature_input.input_binding_hash != binding.binding_hash
        or feature_input.base_panel_snapshot_hash != bundle.panel_snapshot_hash
    ):
        raise AuthoringError("factor_research.prepared_features_new_input_requires_build")
    source, _artifact_root = factor_input_paths(workspace, binding.binding_hash)
    logical_panel_owner(source).verify_snapshot(bundle.panel_snapshot_hash)
    output = f"research-experiments/alpha-{decision.submission.research_input.input_hash}"
    if document is not None and yaml_text is not None:
        raise AuthoringError("research_experiment.one_document_required")
    draft = deepcopy(
        dict(load_authoring_document(yaml_text)) if yaml_text is not None else document or {}
    )
    if set(draft) - {"experiment", "alpha"}:
        raise AuthoringError("factor_research.handoff_document_section_unknown")
    experiment = draft.get("experiment", {})
    alpha = draft.get("alpha", {})
    if not isinstance(experiment, dict) or not isinstance(alpha, dict):
        raise AuthoringError("research_experiment.document_not_a_mapping")
    # Only the already installed one-session development path is handed off.
    # Model and target choices remain the author's, never a test fixture default.
    if set(alpha) - {
        "target_recipe_id",
        "model_capability_handle",
        "model_parameters",
        "ordered_feature_ids",
        "factor_evidence_handle",
        "foundation_admission_hash",
    }:
        raise AuthoringError("factor_research.handoff_method_not_supported")
    if alpha.get("foundation_admission_hash") is not None:
        from alphalattice.foundation.research_foundation.publication.sponsorship import (
            read_foundation_admission,
        )

        admission = read_foundation_admission(
            workspace / "artifacts", str(alpha["foundation_admission_hash"])
        )
        if (
            admission.curation_receipt_hash != curation_receipt_hash
            or admission.factor_receipt_hash != receipt.receipt_hash
            or admission.input_binding_hash != binding.binding_hash
        ):
            raise AuthoringError("research_foundation.handoff_source_mismatch")
        axis = admission.foundation.ordered_factor_ids
    for key, value in {
        "ordered_feature_ids": list(axis),
        "factor_evidence_handle": receipt.receipt_hash,
    }.items():
        if key in alpha and alpha[key] != value:
            raise AuthoringError("factor_research.handoff_selection_mismatch")
        alpha[key] = value
    authored = {**original_document["experiment"], **experiment}
    fixed = {
        "kind": "alpha.model-development",
        "data_snapshot_handle": feature_input.source_handle
        if feature_input
        else bundle.panel_snapshot_hash,
        "output_workspace": output,
        "baseline_workspace": f"research-inputs/{binding.binding_hash}/source",
        "publication_intent": "DEVELOPMENT_EVIDENCE_ONLY",
    }
    for key, value in fixed.items():
        if key in experiment and experiment[key] != value:
            raise AuthoringError(
                f"factor_research.handoff_authority_field_mismatch:experiment.{key}",
                expected={f"experiment.{key}": value},
            )
        authored[key] = value
    # The Factor study's universe, or a declared exploration sample of it (binding plan,
    # B17): an Alpha study may run on fewer names than its Factor evidence, never others.
    factor_universe = original_document["experiment"]["universe_handle"]
    universe = experiment.get("universe_handle", factor_universe)
    if universe_profile(str(universe)) != factor_universe:
        raise AuthoringError(
            "factor_research.handoff_authority_field_mismatch:experiment.universe_handle",
            expected={"experiment.universe_handle": factor_universe},
        )
    authored["universe_handle"] = universe
    draft = {"experiment": authored, "alpha": alpha}
    envelope = ResearchExperimentEnvelope.create(**authored)
    factor_envelope = ResearchExperimentEnvelope.create(**original_document["experiment"])
    if (envelope.sessions.as_of.session, envelope.sessions.as_of.phase) < (
        factor_envelope.sessions.as_of.session,
        factor_envelope.sessions.as_of.phase,
    ):
        raise AuthoringError("factor_research.handoff_evidence_after_cutoff")
    missing = [
        v
        for v in ("target_recipe_id", "model_capability_handle", "model_parameters")
        if v not in alpha
    ]
    executor = cast(
        AlphaExperimentExecutor,
        build_installed_desk_executors(
            envelope=envelope,
            workspace=source,
            artifact_root=source / "artifacts",
            workspace_root=workspace,
            document=draft,
            factor_evidence_root=evidence_root,
            alpha_split_policy=load_alpha_split_policy(),
            feature_authority_outcome_snapshot_handle=bundle.outcome_snapshot_hash,
            feature_input=feature_input,
        )[0],
    )
    return PreparedFactorHandoff(
        workspace,
        source,
        binding,
        draft,
        decision.model_dump(mode="json"),
        executor,
        tuple(missing),
        feature_input,
        FactorHandoffRecipe(
            workspace=workspace,
            manifest=manifest,
            evidence_root=evidence_root,
            original_binding_hash=original_binding_hash,
            original_document=original_document,
            receipt=receipt,
            checkpoint=checkpoint,
            curation_receipt_hash=curation_receipt_hash,
            input_id=input_id,
            document=document,
            yaml_text=yaml_text,
            available_bindings=available_bindings,
            feature_input=feature_input,
            recorded_readback=recorded_readback,
        ),
    )


MODEL_CHOICE_WORDS: Final = (
    "The agent fills the open target, model and parameters within the person's bounds, "
    "preferring the dynamic-panel LightGBM capability with light settings, and tells the "
    "person in one line what it chose and how to change it; it does not stop to ask. A "
    "research strategy's required Alpha study is planned from strategy controls and needs no "
    "model choice."
)
"""Whose decision a handoff's open fields are (FLOW-3)."""


def preview_factor_handoff(
    *,
    workspace: Path,
    manifest: ResearchWorkspaceManifest,
    evidence_root: Path,
    original_binding_hash: str,
    original_document: dict[str, Any],
    receipt: FactorDevelopmentReceipt,
    checkpoint: FactorResearchDeterministicEvidence,
    curation_receipt_hash: str,
    input_id: str | None,
    document: dict[str, Any] | None,
    yaml_text: str | None,
    available_bindings: tuple[ResearchWorkspaceExperimentInput, ...] | None = None,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> dict[str, Any]:
    """Preview the exact curated Factor-to-Alpha draft and input/compiler preflight.

    A complete draft seals its preview program. This step publishes no Foundation and grants no
    training authority or numerical/scientific validation.

    Args:
        workspace: Caller-owned admitted workspace.
        manifest: Exact workspace declaration.
        evidence_root: Exact retained Factor evidence root.
        original_binding_hash: Exact source input revision.
        original_document: Origin study declaration.
        receipt: Exact Factor development receipt.
        checkpoint: Verified deterministic Factor evidence.
        curation_receipt_hash: Exact selected curation receipt.
        input_id: Optional explicit handoff input.
        document: Optional authored Alpha draft object.
        yaml_text: Optional authored Alpha YAML.
        available_bindings: Optional explicitly available input revisions.
        feature_input: Optional exact prepared feature source.

    Returns:
        Incomplete draft or compiler-preflight projection with normalized document/YAML and zero
        numerical calls.
    """
    prepared = prepare_factor_handoff(
        workspace=workspace,
        manifest=manifest,
        evidence_root=evidence_root,
        original_binding_hash=original_binding_hash,
        original_document=original_document,
        receipt=receipt,
        checkpoint=checkpoint,
        curation_receipt_hash=curation_receipt_hash,
        input_id=input_id,
        document=document,
        yaml_text=yaml_text,
        available_bindings=available_bindings,
        feature_input=feature_input,
    )
    result: dict[str, Any] = {
        "status": "DRAFT_INCOMPLETE"
        if prepared.missing_fields
        else "INPUT_COMPILER_PREFLIGHT_PASSED",
        "missing_fields": list(prepared.missing_fields),
        "curation": prepared.curation,
        "input_binding_hash": prepared.binding.binding_hash,
        "foundation": prepared.executor.foundation.model_dump(mode="json"),
        "authoring_options": prepared.executor.compiler.authoring_options(),
        "ordered_feature_ids": prepared.document["alpha"]["ordered_feature_ids"],
        "document": prepared.document,
        "limitations": [
            "DEVELOPMENT_ONLY",
            "NO_FOUNDATION_PUBLICATION",
            "NO_TRAINING_AUTHORITY",
            "INPUT_AND_COMPILER_CHECK_ONLY_NOT_NUMERICAL_OR_SCIENTIFIC_VALIDATION",
        ],
        "task": None,
        "numerical_call_count": 0,
        # Whose decision the open fields are, in the answer itself (FLOW-3).
        "choice_owner": "AGENT_DISCLOSES",
    }
    if prepared.missing_fields:
        result["detail"] = MODEL_CHOICE_WORDS
    if not prepared.missing_fields:
        sealed = prepared.workflow().prepare(
            prepared.document, actor_kind=ActorKind.HUMAN, actor_id="preview"
        )
        result["program"] = sealed.program.model_dump(mode="json")
    result["yaml"] = yaml.safe_dump(prepared.document, sort_keys=False)
    return result
