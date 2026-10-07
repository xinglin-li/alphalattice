"""Manual authored experiments over the existing Task and evidence owners."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import date, datetime
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Any, cast
from uuid import UUID

from alphalattice.capabilities.alpha_modeling.runtime.lightgbm_threads import (
    LightGBMThreads,
    lightgbm_threads,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.plain_refusals import (
    explain,
    task_record_refusal,
)
from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
)
from alphalattice.control.product_host.composition.research_experiment_plan import (
    AlphaExperimentSource,
    ExperimentPlan,
)
from alphalattice.control.product_host.composition.research_experiment_projection import (
    SAMPLE_SCHEME_SUPERSEDED,
    lane_fields,
    method_currency,
    plan_impact,
    published_result_header,
    realization,
    recorded_work,
    research_lane,
    reuse_words,
    sample_standing,
)
from alphalattice.control.product_host.composition.research_prerequisites import (
    FLOW_OF_KIND,
    FLOW_OF_REFUSAL,
    FLOWS,
    Flow,
    holdings,
    prerequisites,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    ResearchWorkspaceManifest,
    ResearchWorkspaceManifestHolder,
    held,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.result_standing import study_standing
from alphalattice.control.product_host.data_preparation.feature_research import (
    ResearchFeatureBuildApplication,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.product_host.research_authoring.authority import (
    universe_control,
    universe_profile,
)
from alphalattice.control.product_host.research_authoring.comparison import POSITION_UNITS
from alphalattice.control.product_host.research_authoring.factor_authoring import (
    describe_feature_input,
)
from alphalattice.control.product_host.research_authoring.factor_curation import (
    submit_report_curation,
)
from alphalattice.control.product_host.research_authoring.factor_handoff import (
    PreparedFactorHandoff,
    prepare_factor_handoff,
    preview_factor_handoff,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    confined,
    factor_input_paths,
    factor_template,
    factor_workflow,
    normalize_factor_document,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.input_revisions import (
    ResearchInputRevisions,
)
from alphalattice.control.product_host.research_authoring.lifecycle_handoff import (
    lifecycle_controls,
    lifecycle_workflow,
)
from alphalattice.control.product_host.research_authoring.portfolio_handoff import (
    PortfolioRiskLink,
    PreparedPortfolioHandoff,
    portfolio_draft,
    prepare_portfolio_handoff,
)
from alphalattice.control.product_host.research_authoring.qualification_handoff import (
    qualification_document,
    qualification_workflow,
)
from alphalattice.control.product_host.research_authoring.risk_handoff import (
    risk_controls,
    risk_workflow,
)
from alphalattice.control.product_host.research_authoring.risk_reports import RiskReportLinks
from alphalattice.control.product_host.research_authoring.timing import research_timing
from alphalattice.control.product_host.storage.inventory import (
    StorageInventoryError,
    require_storage_capacity,
)
from alphalattice.control.product_host.storage.plan_previews import (
    PLAN_PREVIEWS_DIRECTORY,
    PREVIEW_TTL,
    PreviewRegistry,
    RetainedPreview,
)
from alphalattice.control.research_program.authoring.dispatcher import SealedSubmission
from alphalattice.control.research_program.authoring.document import (
    load_authoring_document,
    require_authoring_document,
)
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.control.task_control.child import run_in_child
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskReplan,
    WorkItemDefinition,
    failure_code_from,
)
from alphalattice.control.task_control.registry import TaskNotFoundError, TaskVersionStale
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.content_store import replace_shared_file
from alphalattice.evidence.alternative_evidence.runtime.execution import (
    CpuBudgetStore,
    ModelFitExecution,
    budget_cores,
    machine_load,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    FACTOR_EXPERIMENT_KIND,
    INSTALLED_REDUNDANCY_POLICIES,
    INSTALLED_SCREENING_POLICIES,
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FACTOR_DEVELOPMENT_CURATION_CATEGORY,
    FactorDevelopmentReceipt,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.research_loop.contracts import (
    FactorResearchReviewDecisionReceipt,
)
from alphalattice.foundation.factor_research.research_loop.development_curation import (
    curation_readback,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.research_foundation.contracts import ResearchFoundationAdmission
from alphalattice.foundation.research_foundation.publication.sponsorship import (
    admit_selected_factor_foundation,
    publish_foundation_admission,
    read_foundation_admission,
)
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    LocalBackgroundDispatcher,
)
from alphalattice.interface.local_application.experiment_report import (
    render_experiment_report,
    render_handoff_report,
)
from alphalattice.interface.local_application.failure_codes import (
    located_failure,
    public_failure,
    safe_failure_code,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.investment.alpha_research.candidates.qualification_task import (
    verify_qualification,
)
from alphalattice.investment.alpha_research.experiments.authoring import ALPHA_EXPERIMENT_KIND
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaDevelopmentExecutionReceipt,
)
from alphalattice.investment.alpha_research.experiments.development_evidence import (
    AlphaDevelopmentReceiptReader,
    AlphaDevelopmentVerifiedGraph,
    alpha_development_receipt_handle,
)
from alphalattice.investment.alpha_research.experiments.development_execution import (
    AlphaDevelopmentCancelled,
)
from alphalattice.investment.alpha_research.experiments.family_qualification import (
    METHOD as QUALIFICATION_METHOD,
)
from alphalattice.investment.alpha_research.experiments.family_qualification import (
    AlphaQualificationFamily,
    AlphaQuestionPreparation,
    AlphaStudy,
    alpha_family,
    alpha_question_fields,
    alpha_question_hash,
    qualification_section,
)
from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
    lifecycle_section,
    read_lifecycle_research_receipt,
)
from alphalattice.investment.alpha_research.experiments.mandate import AlphaResearchModelRecipe
from alphalattice.investment.alpha_research.experiments.verification import AlphaEvidenceVerifier
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    lifecycle_implementation_hash,
)
from alphalattice.investment.portfolio_strategy_lab.application.availability import (
    NOT_AVAILABLE,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    KIND as PORTFOLIO_EXPERIMENT_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    PortfolioExperimentCancelled,
    PortfolioExperimentSpec,
    PortfolioExperimentVerifier,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    implementation_hash as portfolio_implementation_hash,
)
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
from alphalattice.investment.risk_research.experiments.development import RiskDevelopmentCancelled
from alphalattice.investment.risk_research.experiments.identity import (
    risk_development_source_closure_hash,
    risk_numerical_source_closure_hash,
)
from alphalattice.investment.risk_research.experiments.replay_identity import (
    replay_follows_recorded_moves,
)
from alphalattice.investment.risk_research.experiments.verification import RiskEvidenceVerifier
from alphalattice.kernel.shared_kernel.environment import numerical_thread_counts
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash
from alphalattice.protocols.actor_execution.contracts import (
    ActorKind,
    ActorSubmissionBinding,
    AgentExecutionBinding,
    seal_actor_submission,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
)

TASK_KIND = "research_experiment"
STAGES = ("execute_sealed_experiment", "verify_experiment_evidence")

RETRYABLE_ARTIFACT_BLOCKS: frozenset[str] = frozenset(
    {
        "risk_research.covariance_chunk_identity_invalid",
        "risk_research.covariance_chunk_tampered",
        "risk_research.development_checkpoint_matrix_mismatch",
        "risk_research.development_checkpoint_scope_mismatch",
        "risk_research.development_checkpoint_axis_mismatch",
        "research_authoring.evidence_artifact_unverifiable",
    }
)
"""Stage refusals that name a required artifact that was missing or did not read.

Once the artifact is back, the same execution contract can continue from its
verified prefix and its checkpoint, so a person may ask for that explicitly. A
scientific refusal, a permission refusal, an exhausted budget or a source that
has not been repaired is not in this set and stays a stop.
"""


RETRYABLE_CAPACITY_BLOCK = "storage.managed_capacity_exceeded"
"""An operator capacity stop; the same declaration may continue after capacity is restored."""


def retryable_block(failure_code: str | None) -> str | None:
    """The repaired-artifact or operator-capacity stop this owner permits retrying."""
    if not failure_code:
        return None
    code = failure_code.split(":", 1)[0]
    return code if code in RETRYABLE_ARTIFACT_BLOCKS or code == RETRYABLE_CAPACITY_BLOCK else None


_VERIFIED_EVIDENCE: ContextVar[dict[tuple[object, ...], object] | None] = ContextVar(
    "research_experiment_verified_evidence", default=None
)
_REUSE_VERIFIED: ContextVar[bool] = ContextVar("research_experiment_reuse_verified", default=False)

LEDGER_READ_OPERATIONS = frozenset(
    {
        "EXPERIMENT_READBACK",
        "EXPERIMENT_DRAFT",
        "EXPERIMENT_PORTFOLIO_DRAFT",
        "EXPERIMENT_CURATION",
        "EXPERIMENT_HANDOFF_PREVIEW",
    }
)
"""The reads that may answer from a study's earlier full verification (binding plan, L1).

Fail-closed: every other operation that reads a study (export, publication, admission,
comparison, replay, a run, a stage's verification) verifies it in full.
"""

_LEDGER_ENTRIES = 8
"""Studies whose verification is kept; each keeps only what a read shows (an Alpha study's
verified graph holds about 130 MB of chunk tables on the real workspace, none of it kept)."""

VERIFICATION_LEDGER_DIRECTORY = "verification-ledger"
"""Under the workspace's `runtime/`: each Alpha study's last full verification, kept on disk as
a read shows it, so a restart does not verify it in full again (V89, decision 5). Only there,
since only there is hashing a measured cost (an Alpha readback re-derives every sealed chunk's
hash, about 2 s on the real workspace; the other kinds read in 0.1-0.5 s)."""


def _sealed_admission(workspace: Path, admission_hash: str) -> ResearchFoundationAdmission:
    """A sealed Foundation admission as its store reads it. A stored record that no longer
    matches its name (the store says so in a sentence, inside the Alpha study's closure) is
    refused by the admission's own code, never the sentence (V449, OP4)."""

    try:
        return read_foundation_admission(workspace / "artifacts", admission_hash)
    except OSError as error:
        raise AuthoringError("research_foundation.admission_artifact_unavailable") from error
    except ValueError as error:
        if safe_failure_code(str(error)) is not None:
            raise
        raise AuthoringError("research_foundation.admission_identity_invalid") from error


@dataclass(frozen=True, slots=True)
class _VerifiedStudy:
    """What one full verification of a study proved, in the terms a read shows it."""

    evidence: ResearchExecutionEvidence
    original: ResearchExecutionEvidence
    currency: dict[str, object]
    view: object
    record_hash: str
    fingerprint: str
    verified_at: datetime
    files: int
    implementation: str


@contextmanager
def verified_study_evidence(*, reuse_verified: bool = False) -> Iterator[None]:
    """Verify each study's evidence once per operation, then release.

    A study page or a research case reads several studies, and a comparison inside
    it reads two of them again; each read walked the whole sealed graph and
    re-derived every chunk's content hash (binding plan, B16: a case gathering
    studies verified the same two Alpha studies twice in one request). Inside this
    scope a Task's evidence is verified once per (Task, record version, readback
    mode) and the verified result reused for the rest of the operation. An
    operation run inside another (a case reading its studies) joins the scope
    already open.

    Across requests, a read in ``LEDGER_READ_OPERATIONS`` (``reuse_verified``) may
    answer from the owner's ledger: the last full verification of the same Task
    record, while no file under the study's output or input bundle changed its
    path, size, modification time or file identity (binding plan, L1). A change that
    keeps all four is caught by every export, publication and admission, which
    verify in full, and by the sweep (V89). Its answer says so (``verification_basis``).
    """
    if _VERIFIED_EVIDENCE.get() is not None:
        yield
        return
    token = _VERIFIED_EVIDENCE.set({})
    reuse = _REUSE_VERIFIED.set(reuse_verified)
    try:
        yield
    finally:
        scope = _VERIFIED_EVIDENCE.get()
        if scope is not None:
            scope.clear()
        _REUSE_VERIFIED.reset(reuse)
        _VERIFIED_EVIDENCE.reset(token)


def implementation_role(kind: str, *, lifecycle: bool = False) -> str:
    """The role a study kind's implementation is recorded under in the identity successors.

    A lifecycle Alpha study has a role of its own: its Program bound the implementation
    until the binding plan's P, so a lifecycle study sealed before P is historical while
    every other Alpha study stays current across the same move.
    """
    return f"research_experiment.{kind}{'.lifecycle' if lifecycle else ''}.implementation"


def plan_implementation_role(plan: ExperimentPlan) -> str:
    """The role this plan's implementation hash is recorded under."""
    if plan.qualification_family is not None:
        return "research_experiment.alpha.qualification.implementation"
    return implementation_role(plan.program.kind, lifecycle=plan.model_training_source is not None)


def plan_implementation_hash(plan: ExperimentPlan) -> str:
    """The implementation this plan's role installs now."""
    if plan.qualification_family is not None:
        return _qualification_implementation_hash()
    return _implementation_hash(plan.program.kind)


def _qualification_implementation_hash() -> str:
    """The code that turns a qualification's family into its candidate set or stop (GR3):
    the method, the qualification it applies and the committer that seals its end."""

    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="ALPHA_QUALIFICATION",
        tracked_paths=tuple(
            "src/alphalattice/investment/alpha_research/" + path
            for path in (
                "experiments/family_qualification.py",
                "candidates/qualification_task.py",
                "candidates/qualification.py",
                "candidates/committer.py",
                "candidates/control.py",
            )
        ),
    )


def _implementation_hash(kind: str = FACTOR_EXPERIMENT_KIND) -> str:
    """The code that turns a sealed Program and its inputs into this kind's numbers.

    This composition file is not in it. What it composes -- the workflow, the
    executor and the inputs -- is re-derived at run time and must equal the sealed
    Program, execution preview and input authority (``_current``), so a change here
    that moved a number would be refused through them (binding plan, B5).
    """
    if kind == RISK_EXPERIMENT_KIND:
        root = resolve_playpen_root(Path(__file__))
        return str(
            canonical_hash(
                {
                    "risk_development": risk_development_source_closure_hash(root),
                    # The numerical code the run executes, which its Program no longer
                    # seals, and which its authority no longer names (binding plan, P).
                    "risk_numerics": risk_numerical_source_closure_hash(root),
                    "host": source_rule_closure_hash(
                        root=root,
                        semantic_owner="product_host",
                        numerical_role="RISK_DEVELOPMENT_HANDOFF",
                        tracked_paths=tuple(
                            "src/alphalattice/" + p
                            for p in (
                                "control/product_host/research_authoring/risk_handoff.py",
                                "control/research_program/authoring/workflow.py",
                            )
                        ),
                    ),
                }
            )
        )
    if kind == PORTFOLIO_EXPERIMENT_KIND:
        return portfolio_implementation_hash()
    closure = source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="product_host",
        numerical_role="FACTOR_DEVELOPMENT_EXPERIMENT"
        if kind == FACTOR_EXPERIMENT_KIND
        else "ALPHA_DEVELOPMENT_EXPERIMENT",
        # Which inputs and sessions a study reads is bound by content in its Program (the
        # input binding and its authority); the Host code that chooses, prepares or verifies
        # them decides no number of the study, so its files (`factor_inputs.py`,
        # `execution.py`, `authority.py`, `input_revisions.py`, the feature preparation's) are
        # not entries, and a development overlay is the Feature engine's sealed result, which
        # the study reads through `development_input.py` (UC, LAWS.md ID8). An Alpha study
        # reads the Factor study's sealed result, so the Factor Desk's code is the Factor
        # kind's entry and not the Alpha kind's.
        tracked_paths=tuple(
            "src/alphalattice/" + p
            for p in (
                "foundation/feature_engine/panels/development_input.py",
                "control/research_program/authoring/workflow.py",
            )
        )
        + (
            tuple(
                "src/alphalattice/foundation/factor_research/experiments/" + path
                for path in ("execution.py", "authoring.py", "verification.py")
            )
            if kind == FACTOR_EXPERIMENT_KIND
            else ()
        )
        + (
            tuple(
                "src/alphalattice/" + path
                for path in (
                    "control/product_host/research_authoring/factor_handoff.py",
                    "control/product_host/research_authoring/lifecycle_handoff.py",
                    "investment/alpha_research/experiments/lifecycle_authoring.py",
                    "control/product_host/research_authoring/foundation.py",
                    "investment/alpha_research/experiments/development_execution.py",
                    "investment/alpha_research/experiments/execution.py",
                    "investment/alpha_research/experiments/development_evidence.py",
                    "investment/alpha_research/experiments/authoring.py",
                    "investment/alpha_research/inputs/development_foundation.py",
                    "foundation/research_foundation/contracts.py",
                    "foundation/research_foundation/mandate/foundation.py",
                    "foundation/research_foundation/publication/sponsorship.py",
                )
            )
            if kind == ALPHA_EXPERIMENT_KIND
            else ()
        ),
    )
    if kind == ALPHA_EXPERIMENT_KIND:
        # A lifecycle Program binds its method only; the code that runs it is bound here,
        # with the rest of the kind's implementation (binding plan, P).
        return str(canonical_hash({"alpha": closure, "lifecycle": lifecycle_implementation_hash()}))
    return closure


PREVIEWS_DIRECTORY = "experiment-previews"
"""Under the workspace's `runtime/`: each preview sealed by its plan hash until it expires."""

ENVELOPE_SCHEMA_ID = "research-experiment-envelope"
"""The one envelope schema the Host's research Desks install and their templates write."""


def refuse_authored_identity(section: Mapping[str, object]) -> None:
    """Refuse what an author may not write into a research envelope (LAWS OP8).

    The Host computes ``envelope_hash``: carried in a document, it would shape the managed
    output path before the envelope recomputes it, and so move the Program and the Plan for
    no change of selection. ``schema_id`` must name the installed envelope schema, since any
    other well-formed string would move them the same way (V128).

    Args:
        section: The authored document's ``experiment`` section.

    Raises:
        AuthoringError: The section carries ``envelope_hash`` or another schema.
    """
    if "envelope_hash" in section:
        raise AuthoringError("research_authoring.resolved_identity_authored:envelope_hash")
    if section.get("schema_id") != ENVELOPE_SCHEMA_ID:
        raise AuthoringError("research_authoring.schema_id_not_installed")


def _task_contract(
    plan: ExperimentPlan, actor: ActorSubmissionBinding
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    if actor.submission_hash != plan.program.program_hash:
        raise AuthoringError("research_experiment.actor_program_mismatch")
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id="research-experiment-task-input",
        payload={"plan": plan.model_dump(mode="json"), "actor": actor.model_dump(mode="json")},
    )
    # Each summary is bytes of every stored Task's goal, which `_checked_plan` rebuilds and
    # compares by equality: a changed one refuses every stored Task of its kind (V518 was
    # reverted for it), so the texts are pinned with their reason in the tests.
    goal = ResearchGoal.create(
        goal_kind="EXECUTE_RESEARCH_EXPERIMENT",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchExecutionEvidence",
        summary={
            FACTOR_EXPERIMENT_KIND: (
                "Run an admitted Factor experiment, not a strategy or current publication."
            ),
            ALPHA_EXPERIMENT_KIND: (
                "Run an admitted Alpha development experiment, "
                "not a strategy or current publication."
            ),
            PORTFOLIO_EXPERIMENT_KIND: (
                "Replay selected Alpha evidence through an EW research book; no activation."
            ),
            RISK_EXPERIMENT_KIND: (
                "Run installed Risk diagnostics over a sealed research input; "
                "no allocation or activation."
            ),
        }[plan.program.kind],
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(STAGES),
        verifier_catalog_hash=canonical_hash(STAGES),
        work_items=tuple(
            WorkItemDefinition.create(
                stage_id=stage,
                dependency_ids=STAGES[:i],
                verifier_id=f"research_experiment.{stage}",
            )
            for i, stage in enumerate(STAGES)
        ),
    )
    return envelope, goal, workflow


class ResearchExperimentApplication:
    """Own explicit previews, admitted experiment tasks and exact durable study readback."""

    task_kind = TASK_KIND
    replans = (
        TaskReplan(task_kind=TASK_KIND, preview="EXPERIMENT_PLAN", admitting="EXPERIMENT_RUN"),
    )
    """The re-plan of the Task kind this owner admits, which the recovery view offers
    (V188)."""

    @property
    def manifest(self) -> ResearchWorkspaceManifest:
        """The workspace manifest, read from the one holder the Host refreshes (V182)."""
        return self._manifests.current

    def __init__(
        self,
        *,
        session: WorkspaceApplicationSession,
        manifest: ResearchWorkspaceManifest | ResearchWorkspaceManifestHolder,
        dispatcher: LocalBackgroundDispatcher,
        clock: Callable[[], datetime],
    ):
        """Wire retained workspace, manifest, dispatcher and bounded preview ownership.

        Preview retention is separate from durable task recovery. Per-attempt workflow/evidence
        caches and installed-role identities belong to this service instance.

        Args:
            session: Retained workspace writer/task session.
            manifest: Held workspace declaration.
            dispatcher: Bounded local task dispatcher.
            clock: Explicit observed-time source.
        """
        self.session, self.dispatcher, self.clock = session, dispatcher, clock
        self._manifests = held(manifest)
        self._previews: PreviewRegistry[ExperimentPlan] = PreviewRegistry(
            model=ExperimentPlan,
            clock=clock,
            root=session.workspace / "runtime" / PREVIEWS_DIRECTORY,
        )
        self._curation_lock = Lock()
        # A Foundation's preview is kept by its admission hash until it expires, sealed on
        # disk, so its seal finds it beside a later preview and after a Host restart (V543).
        self._foundation_previews: PreviewRegistry[ResearchFoundationAdmission] = PreviewRegistry(
            model=ResearchFoundationAdmission,
            clock=clock,
            root=session.workspace / "runtime" / PLAN_PREVIEWS_DIRECTORY / "research-foundation",
            hash_field="admission_hash",
        )
        # The workflow `compatibility` verified for the execution it admits,
        # consumed by the first stage of that same execution (see `execute`).
        self._admitted_workflows: dict[UUID, Any] = {}
        # The evidence a running Task's own checks verified, once per execution attempt.
        self._attempt_evidence: dict[UUID, ResearchExecutionEvidence] = {}
        # Each Task's plan as its own contract checked it, keyed by that contract (`_of`).
        self._task_plans: dict[
            tuple[UUID, str, str, str], tuple[ExperimentPlan, ActorSubmissionBinding]
        ] = {}
        self._ledger: dict[tuple[UUID, bool], _VerifiedStudy] = {}
        self._ledger_lock = Lock()
        # Each study role's installed implementation, asked once: the code this Host serves.
        self._installed: dict[str, str] = {}

    def _foundation_candidate(
        self,
        task_id: UUID,
        decision_hash: str,
        binding: ResearchWorkspaceExperimentInput,
        *,
        recorded_admission: ResearchFoundationAdmission | None = None,
    ) -> ResearchFoundationAdmission:
        parent, receipt, checkpoint = self._factor_parent(
            task_id, recorded_readback=recorded_admission is not None
        )
        prepared = prepare_factor_handoff(
            workspace=self.session.workspace,
            manifest=self.manifest,
            evidence_root=confined(
                self.session.workspace, parent.document["experiment"]["output_workspace"]
            ),
            original_binding_hash=parent.binding.binding_hash,
            original_document=parent.document,
            receipt=receipt,
            checkpoint=checkpoint,
            curation_receipt_hash=decision_hash,
            input_id=binding.input_id,
            document=None,
            yaml_text=None,
            available_bindings=(binding,),
            recorded_readback=recorded_admission is not None,
            feature_input=self._feature_input(
                parent.document["experiment"]["data_snapshot_handle"],
                binding.binding_hash,
                current_policy=recorded_admission is None,
            ),
        )
        return self._admitted_candidate(
            prepared,
            parent=parent,
            receipt=receipt,
            checkpoint=checkpoint,
            task_id=task_id,
            binding=binding,
            recorded_admission=recorded_admission,
        )

    @staticmethod
    def _admitted_candidate(
        prepared: PreparedFactorHandoff,
        *,
        parent: ExperimentPlan,
        receipt: FactorDevelopmentReceipt,
        checkpoint: FactorResearchDeterministicEvidence,
        task_id: UUID,
        binding: ResearchWorkspaceExperimentInput,
        recorded_admission: ResearchFoundationAdmission | None = None,
    ) -> ResearchFoundationAdmission:
        """The Foundation this Host would admit today from one prepared handoff.

        Built from the Task-verified receipt and checkpoint the handoff was
        prepared on, the curation it selected and the Panel identities its
        executor resolved; the caller compares it with a stored admission.
        """

        development = prepared.executor.foundation
        resolver = ArtifactResolver(prepared.source / "artifacts")
        panel = (
            prepared.feature_input.panel_manifest
            if prepared.feature_input
            else resolver.load_feature_panel_manifest(
                resolver.feature_panel_manifest_uri(development.feature_panel_snapshot_hash)
            )
        )
        return admit_selected_factor_foundation(
            receipt=receipt,
            checkpoint=checkpoint,
            decision=FactorResearchReviewDecisionReceipt.model_validate(prepared.curation),
            panel_manifest=dict(panel),
            artifact_root=prepared.source / "artifacts",
            logical_panel_hash=development.logical_panel_hash,
            logical_semantic_index_hash=development.logical_semantic_index_hash,
            factor_task_id=str(task_id),
            input_id=binding.input_id,
            input_binding_hash=binding.binding_hash,
            factor_input_binding_hash=parent.binding.binding_hash,
            recorded_admission=recorded_admission,
        )

    def _foundation(
        self, admission_hash: str, *, recorded_readback: bool = False
    ) -> ResearchFoundationAdmission:
        admission = _sealed_admission(self.session.workspace, admission_hash)
        binding = self._binding(admission.input_id, admission.input_binding_hash)
        if (
            self._foundation_candidate(
                UUID(admission.factor_task_id),
                admission.curation_receipt_hash,
                binding,
                recorded_admission=admission if recorded_readback else None,
            )
            != admission
        ):
            raise AuthoringError("research_foundation.source_changed")
        return admission

    def _foundation_listed(self, admission_hash: str) -> dict[str, object]:
        """One sealed admission with its standing (V279). Verified under the installed curation
        policy it is `CURRENT`, and new work may start on it; otherwise it reads by its recorded
        graph, `HISTORICAL` with the code that refuses a draft or a plan on it."""

        try:
            admission = self._foundation(admission_hash)
        except (ValueError, TaskNotFoundError) as error:
            try:
                recorded = self._foundation(admission_hash, recorded_readback=True)
            except (ValueError, TaskNotFoundError) as recorded_error:
                return self._foundation_refused(admission_hash, recorded_error)
            return self._foundation_body(
                recorded,
                "FOUNDATION_SEALED",
                recorded_readback=True,
                standing={
                    "standing": "HISTORICAL",
                    "standing_code": public_failure(error, "research_foundation.not_current"),
                },
            )
        return self._foundation_body(
            admission, "FOUNDATION_SEALED", standing={"standing": "CURRENT"}
        )

    def _foundation_refused(
        self, admission_hash: str, error: ValueError | TaskNotFoundError
    ) -> dict[str, object]:
        """Keep one unreadable admission's refusal local, without claiming its graph verified."""
        code = (
            "task_control.task_not_found"
            if isinstance(error, TaskNotFoundError)
            else public_failure(error, "research_foundation.not_current")
        )
        body: dict[str, object] = {
            "status": "REFUSED",
            "foundation_admission_hash": admission_hash,
            "failure_code": code,
            "numerical_call_count": 0,
            "detail": "The admission's recorded sources could not be verified. Read Research "
            "inputs to plan a new Factor study.",
            **explain(code),
            "next_action": "RESEARCH_INPUTS",
            "next_requests": {"inputs": {"operation": "RESEARCH_INPUTS"}},
        }
        try:
            admission = _sealed_admission(self.session.workspace, admission_hash)
        except ValueError:
            return body
        body.update(
            admission=admission.model_dump(mode="json"),
            next_action="EXPERIMENT_CONTROLS",
            next_requests={
                "factor": {
                    "operation": "EXPERIMENT_CONTROLS",
                    "experiment_kind": FACTOR_EXPERIMENT_KIND,
                    "research_input_id": admission.input_id,
                    "input_binding_hash": admission.input_binding_hash,
                }
            },
        )
        if isinstance(error, TaskNotFoundError):
            body.update(
                missing_factor_task_id=admission.factor_task_id,
                detail=(
                    f"The recorded Factor Task {admission.factor_task_id} is absent from Task "
                    "Control. Open Factor controls on this admission's recorded input to plan "
                    "a new study; the admission does not retain its original declaration."
                ),
            )
        return body

    def _development_routes(self, input_binding_hash: str) -> dict[str, object]:
        """With no prepared component source, an Alpha study begins at a sealed Foundation of the
        input (V303): each current one's draft, or the list that says how one is sealed."""

        root = (
            self.session.workspace / "artifacts/factor-research/research-desk/foundation-admissions"
        )
        drafts: dict[str, dict[str, str]] = {}
        for path in sorted(root.glob("*.json")):
            listed = self._foundation_listed(path.stem)
            if listed.get("standing") != "CURRENT":
                continue
            admission = cast(dict[str, object], listed["admission"])
            if admission["input_binding_hash"] == input_binding_hash:
                drafts[f"alpha-draft-{path.stem[:12]}"] = {
                    "operation": "EXPERIMENT_FOUNDATION_DRAFT",
                    "foundation_admission_hash": path.stem,
                }
        return {
            "detail": "No prepared component source serves a lifecycle study of this input; an "
            "Alpha development study begins at a sealed Foundation of it, drafted by "
            "EXPERIMENT_FOUNDATION_DRAFT"
            + (
                "."
                if drafts
                else ", and this input has none current: EXPERIMENT_FOUNDATIONS "
                "lists the sealed ones, and a curated Factor study previews and seals one."
            ),
            "next_action": "DRAFT_FROM_A_FOUNDATION" if drafts else "SEAL_A_FOUNDATION",
            "next_requests": drafts or {"foundations": {"operation": "EXPERIMENT_FOUNDATIONS"}},
        }

    def _foundation_replan(self, admission_hash: str) -> dict[str, object]:
        """A seal whose preview is past its hour or was never kept here: preview it again, the
        request bound to the kept preview's study, decision and input where one verifies (V543)."""
        kept = self._foundation_previews.get(admission_hash)
        return {
            "status": "REFUSED",
            "failure_code": "research_foundation.preview_required",
            **(
                {}
                if kept is None
                else {
                    "next_requests": {
                        "replan": {
                            "operation": "EXPERIMENT_FOUNDATION_PREVIEW",
                            "task_id": kept.plan.factor_task_id,
                            "curation_receipt_hash": kept.plan.curation_receipt_hash,
                            "research_input_id": kept.plan.input_id,
                            "input_binding_hash": kept.plan.input_binding_hash,
                        }
                    }
                }
            ),
        }

    @staticmethod
    def _foundation_body(
        admission: ResearchFoundationAdmission,
        status: str,
        *,
        recorded_readback: bool = False,
        standing: dict[str, str] | None = None,
    ) -> dict[str, object]:
        historical = standing is not None and standing["standing"] == "HISTORICAL"
        return {
            "status": status,
            "admission": admission.model_dump(mode="json"),
            "numerical_call_count": 0,
            "task_id": None,
            **(
                {"verification": "RECORDED_GRAPH_NOT_CURRENT_POLICY_ADMISSION"}
                if recorded_readback
                else {}
            ),
            **(standing or {}),
            "next_requests": {
                name: {
                    "operation": operation,
                    "foundation_admission_hash": admission.admission_hash,
                }
                for name, operation in (
                    (("seal", "EXPERIMENT_FOUNDATION_SEAL"),)
                    if status == "FOUNDATION_PREVIEWED"
                    else (("export", "EXPERIMENT_FOUNDATION_EXPORT"),)
                    if historical
                    else (
                        ("alpha-draft", "EXPERIMENT_FOUNDATION_DRAFT"),
                        ("export", "EXPERIMENT_FOUNDATION_EXPORT"),
                    )
                )
            },
            "limitations": [
                "RESEARCH_ONLY_NOT_CURRENT",
                "NO_PROSPECTIVE_VALIDATION",
                "RISK_AND_STRATEGY_NOT_ADMITTED",
                "SEALED_HOLDOUT_UNREAD",
            ],
        }

    def _operate_foundation(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: str,
        agent_execution: AgentExecutionBinding | None,
    ) -> dict[str, object]:
        op = request.operation
        if op == "EXPERIMENT_FOUNDATIONS":
            root = (
                self.session.workspace
                / "artifacts/factor-research/research-desk/foundation-admissions"
            )
            return {
                "status": "AVAILABLE",
                "foundations": [
                    self._foundation_listed(p.stem) for p in sorted(root.glob("*.json"))
                ],
            }
        if op == "EXPERIMENT_FOUNDATION_PREVIEW":
            assert request.task_id is not None and request.curation_receipt_hash is not None
            binding = self._binding(request.research_input_id, request.input_binding_hash)
            previewed = self._foundation_candidate(
                request.task_id, request.curation_receipt_hash, binding
            )
            self._foundation_previews.remember(previewed)
            return self._foundation_body(previewed, "FOUNDATION_PREVIEWED")
        assert request.foundation_admission_hash is not None
        if op == "EXPERIMENT_FOUNDATION_SEAL":
            prior = self._foundation_previews.runnable(request.foundation_admission_hash)
            if prior is None:
                return self._foundation_replan(request.foundation_admission_hash)
            actor = seal_actor_submission(
                actor_kind=ActorKind(caller),
                actor_id=f"local-web-{caller.lower()}",
                submission_hash=prior.admission_hash,
                agent_execution=agent_execution,
            )
            with self._curation_lock:
                binding = self._binding(prior.input_id, prior.input_binding_hash)
                current = self._foundation_candidate(
                    UUID(prior.factor_task_id), prior.curation_receipt_hash, binding
                )
                if current != prior:
                    raise AuthoringError("research_foundation.preview_stale")
                status = publish_foundation_admission(
                    self.session.workspace / "artifacts",
                    current,
                    published_at=self.clock(),
                    actor=actor,
                )
            return {
                **self._foundation_body(current, status),
                "actor": actor.model_dump(mode="json"),
            }
        recorded = op in {"EXPERIMENT_FOUNDATION_READBACK", "EXPERIMENT_FOUNDATION_EXPORT"}
        try:
            admission = self._foundation(
                request.foundation_admission_hash, recorded_readback=recorded
            )
        except (ValueError, TaskNotFoundError) as error:
            return self._foundation_refused(request.foundation_admission_hash, error)
        body = self._foundation_body(admission, "FOUNDATION_SEALED", recorded_readback=recorded)
        if op == "EXPERIMENT_FOUNDATION_EXPORT":
            return {**body, "json": json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)}
        if op == "EXPERIMENT_FOUNDATION_DRAFT":
            task = self.session.task_control_registry.task(UUID(admission.factor_task_id))
            parent, _actor = self._of(task)
            _parent, receipt, checkpoint = self._factor_parent(task.task_id)
            draft = preview_factor_handoff(
                workspace=self.session.workspace,
                manifest=self.manifest,
                evidence_root=confined(
                    self.session.workspace, parent.document["experiment"]["output_workspace"]
                ),
                original_binding_hash=parent.binding.binding_hash,
                original_document=parent.document,
                receipt=receipt,
                checkpoint=checkpoint,
                curation_receipt_hash=admission.curation_receipt_hash,
                input_id=admission.input_id,
                document={"alpha": {"foundation_admission_hash": admission.admission_hash}},
                yaml_text=None,
                available_bindings=(
                    self._binding(admission.input_id, admission.input_binding_hash),
                ),
                feature_input=self._feature_input(
                    parent.document["experiment"]["data_snapshot_handle"],
                    parent.binding.binding_hash,
                ),
            )
            return {
                **draft,
                "plan_request": {
                    "operation": "EXPERIMENT_PLAN",
                    "research_input_id": admission.input_id,
                    "input_binding_hash": admission.input_binding_hash,
                    "factor_task_id": admission.factor_task_id,
                    "curation_receipt_hash": admission.curation_receipt_hash,
                },
            }
        return body

    def _binding(
        self, input_id: str | None, binding_hash: str | None = None
    ) -> ResearchWorkspaceExperimentInput:
        if read_research_workspace_manifest(self.session.workspace) != self.manifest:
            raise AuthoringError("research_experiment.workspace_changed")
        values = self.manifest.experiment_inputs or ()
        matches = [v for v in values if input_id is None or v.input_id == input_id]
        if len(matches) != 1:
            raise AuthoringError(
                "research_experiment.input_selection_required"
                if matches
                else "research_experiment.input_not_admitted"
            )
        return ResearchInputRevisions(self.session).select(
            matches[0].input_id,
            binding_hash,
            verify=False,
        )

    def _feature_input(
        self, handle: str, binding_hash: str, *, current_policy: bool = True
    ) -> ResolvedDevelopmentFeatureInput | None:
        if not handle.startswith("research-features@"):
            return None
        identity = handle.removeprefix("research-features@")
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise AuthoringError("research_experiment.prepared_features_handle_invalid")
        return ResearchFeatureBuildApplication(self.session, clock=self.clock).prepared_source(
            identity, input_binding_hash=binding_hash, current_policy=current_policy
        )

    def controls(
        self,
        input_id: str | None,
        binding_hash: str | None = None,
        kind: str = FACTOR_EXPERIMENT_KIND,
        component_id: str | None = None,
        feature_preparation_hash: str | None = None,
    ) -> dict[str, object]:
        """Project controls from one exact admitted input and installed experiment kind.

        Args:
            input_id: Explicit input selection, or None when uniquely declared.
            binding_hash: Optional exact input revision.
            kind: Installed experiment kind.
            component_id: Optional declared model component.
            feature_preparation_hash: Optional exact prepared factor features.

        Returns:
            Input-selection refusal or installed controls/template and declared research limits.

        Raises:
            AuthoringError: Kind/input/prepared feature selection is not admitted.
        """
        inputs = [v.model_dump(mode="json") for v in self.manifest.experiment_inputs or ()]
        if not inputs:
            return {"status": "RESEARCH_INPUT_NOT_ADMITTED", "inputs": [], "actions": []}
        if input_id is None and len(inputs) > 1:
            return {"status": "INPUT_SELECTION_REQUIRED", "inputs": inputs}
        try:
            binding = self._binding(input_id, binding_hash)
        except FileNotFoundError:
            revisions = ResearchInputRevisions(self.session)
            try:
                anchor = revisions.anchor(
                    input_id if input_id is not None else str(inputs[0]["input_id"])
                )
                requested = binding_hash or anchor.binding_hash
                groups = cast(list[dict[str, Any]], revisions.versions()["inputs"])
                missing = [
                    version
                    for group in groups
                    if group["input_id"] == anchor.input_id
                    for version in group["versions"]
                    if version["binding_hash"] == requested
                ]
                if (
                    len(missing) == 1
                    and missing[0].get("available") is False
                    and missing[0].get("unreadable") == "research_input.manifest_missing"
                ):
                    return {
                        "status": "REFUSED",
                        "failure_code": missing[0]["unreadable"],
                        "input_id": anchor.input_id,
                        "input_binding_hash": requested,
                        "detail": missing[0]["detail"],
                        "next_requests": missing[0]["next_requests"],
                    }
            except (OSError, ValueError, KeyError, TypeError):
                pass
            raise
        if feature_preparation_hash is not None and kind != FACTOR_EXPERIMENT_KIND:
            raise AuthoringError("research_experiment.prepared_features_begin_with_factor")
        if kind not in FLOW_OF_KIND:
            raise AuthoringError("research_experiment.standalone_kind_not_installed")
        # What the flow needs on this input, what the workspace holds of it, what next (V367).
        needed = {"prerequisites": self._prerequisites(FLOW_OF_KIND[kind], binding)}
        if kind == ALPHA_EXPERIMENT_KIND:
            answer = lifecycle_controls(self.session.workspace, binding, component_id)
            if (
                answer["status"] == "MODEL_TRAINING_SOURCE_SELECTION_REQUIRED"
                and not answer["sources"]
            ):
                return {**answer, **self._development_routes(binding.binding_hash), **needed}
            return {**answer, **needed}
        if kind == RISK_EXPERIMENT_KIND:
            return {**risk_controls(self.session, binding), **needed}
        bundle = read_factor_bundle(self.session.workspace, binding.binding_hash)
        root = confined(
            self.session.workspace, f"research-inputs/{binding.binding_hash}/source/artifacts"
        )
        panel = ArtifactResolver(root).load_feature_panel_manifest(
            ArtifactResolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
        )
        prepared = (
            self._feature_input(
                f"research-features@{feature_preparation_hash}", binding.binding_hash
            )
            if feature_preparation_hash is not None
            else None
        )
        if prepared is not None:
            panel = prepared.panel_manifest
        inventory = factor_inventory_from_panel_manifest(panel)
        template = factor_template(bundle)
        if prepared is not None:
            template["experiment"]["data_snapshot_handle"] = prepared.source_handle
        template["factor"]["factor_ids"] = [v.factor_id for v in inventory]
        import yaml  # type: ignore[import-untyped]

        controls = [
            {
                "path": ["factor", "factor_ids"],
                "label": "Factors to inspect",
                "type": "multiple",
                "options": [v.factor_id for v in inventory],
                "value": template["factor"]["factor_ids"],
                "help": "All context factors remain in the multiple-testing denominator.",
            },
            *(
                {
                    "path": ["experiment", "sessions", key],
                    "label": label,
                    "type": "date",
                    "value": template["experiment"]["sessions"][key],
                }
                for key, label in (
                    ("start", "Authority interval start"),
                    ("end", "Authority interval end"),
                )
            ),
            {
                "path": ["experiment", "sessions", "as_of", "session"],
                "label": "Decision cutoff (official close)",
                "type": "date",
                "value": template["experiment"]["sessions"]["as_of"]["session"],
            },
            universe_control(
                str(template["experiment"]["universe_handle"]),
                len(
                    FeaturePanelReader(ArtifactResolver(root)).listing_ids(
                        ArtifactResolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
                    )
                ),
            ),
            {
                "path": ["factor", "screening_policy"],
                "label": "Screening policy",
                "type": "select",
                "options": INSTALLED_SCREENING_POLICIES,
                "value": INSTALLED_SCREENING_POLICIES[0],
            },
            {
                "path": ["factor", "redundancy_policy"],
                "label": "Redundancy policy",
                "type": "select",
                "options": INSTALLED_REDUNDANCY_POLICIES,
                "value": INSTALLED_REDUNDANCY_POLICIES[0],
            },
        ]
        return {
            "status": "READY",
            "inputs": inputs,
            "input_id": binding.input_id,
            "input_binding_hash": binding.binding_hash,
            "method": FACTOR_EXPERIMENT_KIND,
            "template": template,
            "yaml": yaml.safe_dump(template, sort_keys=False),
            "controls": controls,
            "factor_options": [v.factor_id for v in inventory],
            "feature_input": {
                **describe_feature_input(
                    input_binding_hash=binding.binding_hash,
                    panel=panel,
                    workspace=self.session.workspace,
                ),
                **({"definition_plan_hash": prepared.definition_plan_hash} if prepared else {}),
            },
            "screening_options": INSTALLED_SCREENING_POLICIES,
            "redundancy_options": INSTALLED_REDUNDANCY_POLICIES,
            "limits": [
                "Development evidence only; no Foundation admission or strategy activation.",
                "Selection scopes the report, not the full multiple-testing context.",
            ],
            **needed,
        }

    def _prerequisites(
        self, flow: Flow, binding: ResearchWorkspaceExperimentInput
    ) -> dict[str, Any]:
        """A flow's prerequisites on one input, from one read of the Tasks (V367)."""
        succeeded = [
            task
            for task in self.session.task_control_registry.tasks()
            if task.task_kind == TASK_KIND and task.lifecycle is TaskLifecycle.SUCCEEDED
        ]
        held = holdings(
            succeeded,
            lambda task: self._of(task)[0],
            binding_hash=binding.binding_hash,
            workspace=self.session.workspace,
        )
        return prerequisites(
            flow, held, input_id=binding.input_id, binding_hash=binding.binding_hash
        )

    def intents(self) -> list[dict[str, object]]:
        """Each research input's standard flows, what each needs, holds and asks next (V376).

        An agent's first read (`workspace show`) is its path: on each input, every flow's
        prerequisite results, the completed studies the workspace holds, what is missing and the
        requests allowed next, each ready to send, from one read of the Tasks. An input whose
        revision does not resolve is named with its refusal and no flows.
        """
        succeeded = [
            task
            for task in self.session.task_control_registry.tasks()
            if task.task_kind == TASK_KIND and task.lifecycle is TaskLifecycle.SUCCEEDED
        ]
        found: list[dict[str, object]] = []
        for value in self.manifest.experiment_inputs or ():
            try:
                binding = self._binding(value.input_id)
            except (AuthoringError, KeyError, ValueError, OSError) as error:
                found.append(
                    {
                        "research_input_id": value.input_id,
                        **located_failure(error, "research_experiment.input_not_resolved"),
                    }
                )
                continue
            held = holdings(
                succeeded,
                lambda task: self._of(task)[0],
                binding_hash=binding.binding_hash,
                workspace=self.session.workspace,
            )
            found.append(
                {
                    "research_input_id": binding.input_id,
                    "input_binding_hash": binding.binding_hash,
                    "flows": {
                        flow: prerequisites(
                            flow,
                            held,
                            input_id=binding.input_id,
                            binding_hash=binding.binding_hash,
                        )
                        for flow in FLOWS
                    },
                }
            )
        return found

    def prerequisites_of(
        self,
        flow: Flow,
        *,
        task_id: UUID | None = None,
        input_id: str | None = None,
        binding_hash: str | None = None,
    ) -> dict[str, object]:
        """A flow's `prerequisites` on the input a request names, or nothing when none resolves.

        Args:
            flow: The flow.
            task_id: A study whose input it is.
            input_id: The research input, when the request names one.
            binding_hash: Its revision, or with no input the revision alone.

        Returns:
            `{"prerequisites": ...}`, or empty when the request's input does not resolve.
        """
        try:
            if task_id is not None:
                binding = self._of(self.session.task_control_registry.task(task_id))[0].binding
            elif input_id is None and binding_hash is not None:
                binding = next(
                    value
                    for value in (
                        self._revision(v.input_id, binding_hash)
                        for v in self.manifest.experiment_inputs or ()
                    )
                    if value is not None
                )
            else:
                binding = self._binding(input_id, binding_hash)
        except (StopIteration, AuthoringError, KeyError, ValueError, OSError):
            return {}
        return {"prerequisites": self._prerequisites(flow, binding)}

    def refusal_prerequisites(self, code: str, request: Any) -> dict[str, object]:
        """A refusal meaning a prerequisite result is missing names the flow's (V367)."""
        flow = FLOW_OF_REFUSAL.get(code)
        if flow is None:
            return {}
        return self.prerequisites_of(
            flow,
            task_id=getattr(request, "task_id", None),
            input_id=getattr(request, "research_input_id", None),
            binding_hash=getattr(request, "input_binding_hash", None),
        )

    def _revision(
        self, input_id: str, binding_hash: str
    ) -> ResearchWorkspaceExperimentInput | None:
        try:
            return ResearchInputRevisions(self.session).select(input_id, binding_hash, verify=False)
        except (AuthoringError, KeyError, ValueError, OSError):
            return None

    def _factor_parent(
        self,
        task_id: UUID,
        *,
        recorded_readback: bool = False,
    ) -> tuple[ExperimentPlan, FactorDevelopmentReceipt, FactorResearchDeterministicEvidence]:
        task = self.session.task_control_registry.task(task_id)
        plan, _actor = self._of(task)
        if plan.program.kind != FACTOR_EXPERIMENT_KIND:
            raise AuthoringError("research_experiment.factor_parent_required")
        body = self.readback(task_id, current_policy=not recorded_readback)
        if body["status"] != "EXPERIMENT_PUBLISHED":
            raise AuthoringError("factor_research.completed_experiment_required")
        return (
            plan,
            FactorDevelopmentReceipt.model_validate_json(json.dumps(body["receipt"])),
            FactorResearchDeterministicEvidence.model_validate_json(json.dumps(body["result"])),
        )

    def _question_preparation(
        self, plan: ExperimentPlan
    ) -> tuple[AlphaQuestionPreparation, AlphaResearchModelRecipe]:
        """A development study's fold plan, Program and recipe, prepared as its run prepares
        them; a lifecycle or qualification study asks no development question."""
        if plan.alpha_source is None:
            raise AuthoringError("alpha_research.qualification_question_not_a_development_study")
        prepared, _source = self._prepare_alpha(
            plan.binding,
            plan.document,
            plan.alpha_source.factor_task_id,
            plan.alpha_source.curation_receipt_hash,
        )
        sealed = prepared.workflow().prepare(
            plan.document, actor_kind=ActorKind.HUMAN, actor_id="preview"
        )
        preparation = prepared.executor.prepare_execution(
            program=sealed.program, document=plan.document, authority=sealed.authority
        )
        return (
            AlphaQuestionPreparation(
                fold_plan=preparation.fold_plan,
                development_program=preparation.development_program,
            ),
            preparation.model_recipe,
        )

    def _goal_opened_at(self, goal_id: UUID) -> datetime:
        """When a goal was opened: its first revision's time, whatever it was revised to."""
        store = GoalStore(self.session.workspace / "artifacts", self.manifest.workspace_id)
        goal = store.head(goal_id)
        if goal is None:
            raise AuthoringError("alpha_research.qualification_goal_not_found")
        while goal.parent_hash is not None:
            goal = store.load(goal.parent_hash)
        return goal.recorded_at

    def _qualification(
        self, document: Mapping[str, Any], *, cancelled: Callable[[], bool] = lambda: False
    ) -> tuple[
        ResearchProgramWorkflow,
        SealedSubmission,
        dict[str, Any],
        dict[str, Any],
        AlphaQualificationFamily,
        ResearchWorkspaceExperimentInput,
        tuple[dict[str, object], ...],
    ]:
        """An Alpha qualification's family, read from Task Control (GR3, V77).

        Every Alpha development study on the question admitted from the goal's opening on is a
        member, whether a goal's session ran it or not: the whole attempted family enters the
        Holm correction. A study still moving or waiting on someone leaves the family
        unsettled, one cancelled before its result is named without evidence, and a recipe
        studied twice counts once, its first study standing.
        """
        section = qualification_section(document)
        if section is None:
            raise AuthoringError(
                "alpha_research.qualification_declaration_invalid:methodology_id",
                expected={"methodology_id": QUALIFICATION_METHOD},
            )
        identifiers: dict[str, UUID] = {}
        for name in ("goal_id", "question_task_id"):
            try:
                identifiers[name] = UUID(str(section.get(name)))
            except ValueError as error:
                raise AuthoringError(
                    f"alpha_research.qualification_declaration_invalid:{name}"
                ) from error
        goal_id, anchor_id = identifiers["goal_id"], identifiers["question_task_id"]
        opened_at = self._goal_opened_at(goal_id)
        registry = self.session.task_control_registry
        try:
            anchor_task = registry.task(anchor_id)
        except KeyError as error:
            raise AuthoringError("alpha_research.qualification_question_not_found") from error
        if anchor_task.task_kind != TASK_KIND:
            raise AuthoringError("alpha_research.qualification_question_not_a_development_study")
        anchor, _actor = self._of(anchor_task)
        anchor_preparation, _recipe = self._question_preparation(anchor)
        question = alpha_question_hash(anchor_preparation.development_program)
        question_fields = alpha_question_fields(anchor_preparation.development_program)
        assert anchor.alpha_source is not None
        workspace = self.session.workspace.resolve()

        def sealed_study(plan: ExperimentPlan) -> tuple[str, str, str]:
            output = confined(workspace, plan.document["experiment"]["output_workspace"])
            receipt = AlphaDevelopmentReceiptReader(output / "alpha-development").load(
                self._stored_evidence(plan).artifact_uris[0]
            )
            return (
                output.resolve().relative_to(workspace).as_posix(),
                receipt.receipt_hash,
                receipt.batch_result_hash,
            )

        studies: list[AlphaStudy] = []
        excluded: list[dict[str, object]] = []
        for task in registry.tasks():
            if task.task_kind != TASK_KIND or task.admitted_at < opened_at:
                continue
            plan, _actor = self._of(task)
            # The studies of this input and Foundation; the owner's rule chooses among them.
            if (
                plan.alpha_source is None
                or plan.binding != anchor.binding
                or plan.alpha_source.foundation_hash != anchor.alpha_source.foundation_hash
            ):
                continue
            preparation, recipe = self._question_preparation(plan)
            fields = alpha_question_fields(preparation.development_program)
            if fields != question_fields:
                # A study of the goal the family leaves out, and the fields that split it off
                # (V293): the family is one question's, and this study asked another.
                excluded.append(
                    {
                        "task_id": str(task.task_id),
                        "differs_in": sorted(
                            key
                            for key in fields.keys() | question_fields.keys()
                            if fields.get(key) != question_fields.get(key)
                        ),
                    }
                )
            studies.append(
                AlphaStudy(
                    task_id=str(task.task_id),
                    admitted_at=task.admitted_at,
                    lifecycle=task.lifecycle.value,
                    question_hash=alpha_question_hash(preparation.development_program),
                    recipe=recipe,
                    sealed=(
                        partial(sealed_study, plan)
                        if task.lifecycle is TaskLifecycle.SUCCEEDED
                        else None
                    ),
                )
            )
        family = alpha_family(
            studies=studies, question_hash=question, goal_id=str(goal_id), opened_at=opened_at
        )
        normalized = qualification_document(document, anchor.document["experiment"], family)
        workflow, sealed, preview = qualification_workflow(
            workspace=self.session.workspace,
            document=normalized,
            family=family,
            prepare_question=lambda: anchor_preparation,
            authority=anchor.authority,
            cancelled=cancelled,
        )
        return workflow, sealed, preview, normalized, family, anchor.binding, tuple(excluded)

    def _prepare_alpha(
        self,
        binding: ResearchWorkspaceExperimentInput,
        document: dict[str, Any],
        factor_task_id: UUID,
        curation_receipt_hash: str,
    ) -> tuple[PreparedFactorHandoff, AlphaExperimentSource]:
        parent, receipt, checkpoint = self._factor_parent(factor_task_id)
        prepared = prepare_factor_handoff(
            workspace=self.session.workspace,
            manifest=self.manifest,
            evidence_root=confined(
                self.session.workspace, parent.document["experiment"]["output_workspace"]
            ),
            original_binding_hash=parent.binding.binding_hash,
            original_document=parent.document,
            receipt=receipt,
            checkpoint=checkpoint,
            curation_receipt_hash=curation_receipt_hash,
            input_id=binding.input_id,
            document=document,
            yaml_text=None,
            available_bindings=(binding,),
            feature_input=self._feature_input(
                parent.document["experiment"]["data_snapshot_handle"], binding.binding_hash
            ),
        )
        if prepared.missing_fields:
            raise AuthoringError(
                "alpha_research.explicit_declaration_required:" + ",".join(prepared.missing_fields)
            )
        admission_hash = document.get("alpha", {}).get("foundation_admission_hash")
        if admission_hash is not None:
            # The admission the declaration names, proved against this very
            # handoff: the Task-verified receipt and checkpoint, the selected
            # curation and the Panel identities the executor resolved are the
            # ones the run would read. Verified here, once, rather than on a
            # second handoff of the same sources prepared only to be compared.
            admission = _sealed_admission(self.session.workspace, str(admission_hash))
            if (
                admission.factor_task_id != str(factor_task_id)
                or admission.curation_receipt_hash != curation_receipt_hash
                or admission.input_id != binding.input_id
                or admission.input_binding_hash != binding.binding_hash
            ):
                raise AuthoringError("research_foundation.handoff_source_mismatch")
            candidate = self._admitted_candidate(
                prepared,
                parent=parent,
                receipt=receipt,
                checkpoint=checkpoint,
                task_id=factor_task_id,
                binding=binding,
            )
            if candidate != admission:
                raise AuthoringError("research_foundation.source_changed")
        return prepared, AlphaExperimentSource(
            factor_task_id=factor_task_id,
            factor_binding=parent.binding,
            factor_receipt_hash=receipt.receipt_hash,
            curation_receipt_hash=curation_receipt_hash,
            foundation_hash=prepared.executor.foundation.foundation_hash,
            foundation_admission_hash=admission_hash,
        )

    def _prepare_portfolio(
        self,
        alpha_task_id: UUID,
        candidate_id: str,
        document: dict[str, Any] | None = None,
        task_id: UUID | None = None,
    ) -> tuple[PreparedPortfolioHandoff, ResearchWorkspaceExperimentInput]:
        parent, _actor = self._of(self.session.task_control_registry.task(alpha_task_id))
        if parent.program.kind != ALPHA_EXPERIMENT_KIND:
            raise AuthoringError("portfolio_research.alpha_parent_required")
        if parent.alpha_source is None:
            # A model lifecycle replay holds models, not candidates: a book is drafted from a
            # development study's candidate, which the refusal's prerequisites name (V486).
            raise AuthoringError("portfolio_research.alpha_development_required")
        body = self.readback(alpha_task_id)
        if body["status"] != "EXPERIMENT_PUBLISHED":
            raise AuthoringError("portfolio_research.completed_alpha_required")
        binding = self._binding(parent.binding.input_id, parent.binding.binding_hash)
        evidence = ResearchExecutionEvidence.model_validate(body["evidence"])
        prepared = prepare_portfolio_handoff(
            workspace=self.session.workspace,
            input_binding_hash=binding.binding_hash,
            alpha_task_id=str(alpha_task_id),
            alpha_program_hash=parent.program.program_hash,
            alpha_document=parent.document,
            alpha_receipt_handle=evidence.artifact_uris[0],
            # The receipt the readback above walked and proved, handed to the
            # consumer of this same request rather than walked a second time.
            alpha_receipt=AlphaDevelopmentExecutionReceipt.model_validate(body["receipt"]),
            candidate_id=candidate_id,
            document=document,
            risk=self._portfolio_risk_link(document),
            cancellation=lambda: (
                task_id is not None
                and self.session.task_control_registry.task(task_id).lifecycle
                is TaskLifecycle.CANCEL_REQUESTED
            ),
        )
        return prepared, binding

    def _risk_studies(self, binding_hash: str) -> tuple[str, ...]:
        """Offer only Risk studies the Portfolio link's own reader admits on this input."""

        found = []
        for task in self.session.task_control_registry.tasks():
            if task.task_kind != TASK_KIND or task.lifecycle is not TaskLifecycle.SUCCEEDED:
                continue
            plan, _actor = self._of(task)
            if (
                plan.program.kind == RISK_EXPERIMENT_KIND
                and plan.binding.binding_hash == binding_hash
            ):
                try:
                    self._portfolio_risk_link({"portfolio": {"risk_task_id": str(task.task_id)}})
                except AuthoringError as error:
                    if str(error) in {
                        "portfolio_research.risk_study_scoped",
                        "portfolio_research.risk_study_not_completed",
                    }:
                        continue
                    raise
                found.append(str(task.task_id))
        return tuple(found)

    def _portfolio_risk_link(self, document: dict[str, Any] | None) -> PortfolioRiskLink | None:
        """The completed Risk study a Portfolio declaration names, read as its readback shows it."""

        section = (document or {}).get("portfolio")
        named = section.get("risk_task_id") if isinstance(section, dict) else None
        if named is None:
            return None
        try:
            task_id = UUID(str(named))
        except ValueError as error:
            raise AuthoringError("portfolio_research.risk_study_invalid") from error
        risk, _actor = self._of(self.session.task_control_registry.task(task_id))
        if risk.program.kind != RISK_EXPERIMENT_KIND:
            raise AuthoringError("portfolio_research.risk_study_invalid")
        body = self.readback(task_id)
        surface, risk_input = body.get("risk_surface"), body.get("risk_input")
        if body.get("status") != "EXPERIMENT_PUBLISHED" or not isinstance(surface, dict):
            raise AuthoringError("portfolio_research.risk_study_not_completed")
        if surface.get("scope_surfaces") or not isinstance(risk_input, dict):
            raise AuthoringError("portfolio_research.risk_study_scoped")
        return PortfolioRiskLink(
            task_id=str(task_id),
            program_hash=risk.program.program_hash,
            surface_hash=str(surface["surface_hash"]),
            evidence_root=confined(
                self.session.workspace, risk.document["experiment"]["output_workspace"]
            ),
            input_binding_hash=risk.binding.binding_hash,
            panel_snapshot_hash=str(risk_input["panel_snapshot_hash"]),
            universe_revision=str(risk_input["universe_revision_sha256"]),
            formation_sessions=tuple(
                date.fromisoformat(str(v)) for v in surface["formation_sessions"]
            ),
            ordered_listing_ids=tuple(str(v) for v in surface["ordered_listing_ids"]),
        )

    def plan(
        self,
        input_id: str | None,
        document: dict[str, Any] | None,
        yaml_text: str | None,
        binding_hash: str | None = None,
        origin_task_id: UUID | None = None,
        factor_task_id: UUID | None = None,
        curation_receipt_hash: str | None = None,
        *,
        caller: str = "HUMAN",
    ) -> dict[str, object]:
        """Prepare one explicit experiment declaration and retain its exact bounded preview.

        Args:
            input_id: Explicit declared research input.
            document: Authored declaration object, exclusive with yaml_text.
            yaml_text: Authored YAML, exclusive with document.
            binding_hash: Optional exact input revision.
            origin_task_id: Optional completed study used to describe declaration changes.
            factor_task_id: Optional exact curated Factor parent for Alpha development.
            curation_receipt_hash: Optional exact Factor curation receipt.
            caller: Declared preview caller.

        Returns:
            Sealed program/document, exact plan, execution preview, change impact and legal next
            requests; numerical_call_count is zero.

        Raises:
            AuthoringError: Document, typed upstream selection, origin, input or authority is
                inadmissible.
        """
        if (document is None) == (yaml_text is None):
            raise AuthoringError("research_experiment.one_document_required")
        # One check for both entries, the YAML text and the object (V134).
        selected = (
            load_authoring_document(yaml_text)
            if yaml_text is not None
            else require_authoring_document(document)
        )
        assert selected is not None
        origin = self.readback(origin_task_id) if origin_task_id is not None else None
        if origin is not None and origin["status"] != "EXPERIMENT_PUBLISHED":
            raise AuthoringError("research_experiment.completed_origin_required")
        origin_plan = (
            self._of(self.session.task_control_registry.task(origin_task_id))[0]
            if origin_task_id is not None
            else None
        )
        alpha_source = None
        portfolio_source = None
        model_training_source = None
        qualification_family = None
        excluded: tuple[dict[str, object], ...] = ()
        section = selected.get("experiment")
        if not isinstance(section, dict):
            raise AuthoringError("research_authoring.experiment_section_missing")
        refuse_authored_identity(section)
        kind = section.get("kind")
        if kind == RISK_EXPERIMENT_KIND:
            if factor_task_id is not None or curation_receipt_hash is not None:
                raise AuthoringError("risk_research.unexpected_parent_selection")
            binding = self._binding(input_id, binding_hash)
            workflow, authority, preview, normalized = risk_workflow(
                session=self.session,
                binding=binding,
                document=selected,
            )
            sealed = workflow.prepare(normalized, actor_kind=ActorKind.HUMAN, actor_id="preview")
        elif kind == PORTFOLIO_EXPERIMENT_KIND:
            if factor_task_id is not None or curation_receipt_hash is not None:
                raise AuthoringError("portfolio_research.source_fields_invalid")
            spec = PortfolioExperimentSpec.model_validate(selected.get("portfolio"))
            prepared_portfolio, binding = self._prepare_portfolio(
                UUID(spec.alpha_task_id), spec.candidate_id, dict(selected)
            )
            if input_id not in {None, binding.input_id} or binding_hash not in {
                None,
                binding.binding_hash,
            }:
                raise AuthoringError("portfolio_research.input_not_alpha_source")
            normalized = prepared_portfolio.document
            portfolio_source = prepared_portfolio.source
            workflow = prepared_portfolio.workflow()
            sealed = workflow.prepare(normalized, actor_kind=ActorKind.HUMAN, actor_id="preview")
            authority, preview = sealed.authority, prepared_portfolio.preview
        elif kind == ALPHA_EXPERIMENT_KIND and qualification_section(selected) is not None:
            if factor_task_id is not None or curation_receipt_hash is not None:
                # A qualification's parent is its goal's family, named in its section.
                raise AuthoringError(
                    "alpha_research.qualification_unexpected_parent:"
                    + ("factor_task_id" if factor_task_id is not None else "curation_receipt_hash"),
                    expected={"factor_task_id": None, "curation_receipt_hash": None},
                )
            workflow, sealed, preview, normalized, qualification_family, binding, excluded = (
                self._qualification(selected)
            )
            if input_id not in {None, binding.input_id} or binding_hash not in {
                None,
                binding.binding_hash,
            }:
                raise AuthoringError("alpha_research.qualification_input_not_the_question")
            authority = sealed.authority
        elif kind == ALPHA_EXPERIMENT_KIND and lifecycle_section(selected) is not None:
            if factor_task_id is not None or curation_receipt_hash is not None:
                raise AuthoringError("alpha_research.lifecycle_unexpected_factor_parent")
            binding = self._binding(input_id, binding_hash)
            workflow, sealed, preview, normalized, model_training_source = lifecycle_workflow(
                workspace=self.session.workspace,
                binding=binding,
                document=dict(selected),
            )
            authority = sealed.authority
        elif kind == ALPHA_EXPERIMENT_KIND:
            admission_hash = selected.get("alpha", {}).get("foundation_admission_hash")
            if admission_hash is not None:
                # Read for the selection it carries; ``_prepare_alpha`` proves it
                # against the handoff the declaration is prepared on.
                admission = _sealed_admission(self.session.workspace, str(admission_hash))
                if (
                    factor_task_id not in {None, UUID(admission.factor_task_id)}
                    or curation_receipt_hash not in {None, admission.curation_receipt_hash}
                    or input_id not in {None, admission.input_id}
                    or binding_hash not in {None, admission.input_binding_hash}
                ):
                    raise AuthoringError("research_foundation.handoff_source_mismatch")
                factor_task_id, curation_receipt_hash = (
                    UUID(admission.factor_task_id),
                    admission.curation_receipt_hash,
                )
                input_id, binding_hash = admission.input_id, admission.input_binding_hash
            if factor_task_id is None or curation_receipt_hash is None:
                raise AuthoringError("alpha_research.factor_decision_selection_required")
            parent, _actor = self._of(self.session.task_control_registry.task(factor_task_id))
            binding = self._binding(
                input_id or parent.binding.input_id,
                binding_hash or parent.binding.binding_hash,
            )
            prepared, alpha_source = self._prepare_alpha(
                binding, dict(selected), factor_task_id, curation_receipt_hash
            )
            normalized = prepared.document
            workflow = prepared.workflow()
            sealed = workflow.prepare(normalized, actor_kind=ActorKind.HUMAN, actor_id="preview")
            authority = sealed.authority
            preview = prepared.executor.prepare_execution(
                program=sealed.program,
                document=normalized,
                authority=authority,
            ).describe()
        else:
            if factor_task_id is not None or curation_receipt_hash is not None:
                raise AuthoringError("research_experiment.alpha_source_on_factor")
            binding = self._binding(input_id, binding_hash)
            bundle = read_factor_bundle(self.session.workspace, binding.binding_hash)
            feature_input = self._feature_input(
                str(selected["experiment"].get("data_snapshot_handle", "")), binding.binding_hash
            )
            normalized = normalize_factor_document(
                selected, binding, bundle, feature_input=feature_input
            )
            workflow, authority, preview = factor_workflow(
                workspace=self.session.workspace,
                binding_hash=binding.binding_hash,
                document=normalized,
                bundle=bundle,
                feature_input=feature_input,
            )
            sealed = workflow.prepare(normalized, actor_kind=ActorKind.HUMAN, actor_id="preview")
        plan = ExperimentPlan.create(
            workspace_id=self.manifest.workspace_id,
            binding=binding,
            document=normalized,
            program=sealed.program,
            authority=authority,
            execution_preview=preview,
            implementation_hash=(
                _implementation_hash(sealed.program.kind)
                if qualification_family is None
                else _qualification_implementation_hash()
            ),
            origin_task_id=origin_task_id,
            alpha_source=alpha_source,
            portfolio_source=portfolio_source,
            model_training_source=model_training_source,
            qualification_family=qualification_family,
        )
        # A bounded set of unadmitted previews, each addressable by its own hash,
        # so a second actor's PLAN does not make the first one unrunnable. Durable
        # Task inputs, never this registry, own recovery and completed readback.
        self._previews.remember(plan, caller=caller)
        match = self._matching_task(plan)
        # The PLAN body is a function of the declaration alone (the YAML, the
        # document and the Agent bridge paths answer identically); who previewed
        # it and until when are service facts, read through the inspect request.
        impact = plan_impact(plan, origin_plan, match)
        change = cast(dict[str, object], impact["change"])
        return {
            "status": "PLANNED",
            "plan_hash": plan.plan_hash,
            "preview": {
                "retention": self._previews.retention,
                "expires_after_minutes": int(PREVIEW_TTL.total_seconds() // 60),
            },
            "program": plan.program.model_dump(mode="json"),
            "document": normalized,
            "numerical_call_count": 0,
            "execution_preview": preview,
            # The goal's studies the qualification's family leaves out, and why (V293).
            **({"family_excluded_studies": list(excluded)} if excluded else {}),
            "task_id": None,
            "timing": self._timing(plan),
            "impact": impact,
            "execution_intent": {
                "disposition": "EXISTING_EXECUTION_CANDIDATE" if match else "NEW_EXECUTION",
                "task_id": str(match.task_id) if match else None,
                "lifecycle": match.lifecycle.value if match else None,
                "verification": "RUN_REVALIDATES_BEFORE_REUSE_OR_ADMISSION",
                "upstream": "SAVED_ALPHA_SCORES_NO_REFIT"
                if portfolio_source
                else "DECLARED_METHOD_ON_SEALED_INPUT",
            },
            "declaration_changes": change["declared_fields"] if origin is not None else None,
            **({"origin_task_id": str(origin_task_id)} if origin_task_id else {}),
            "selected_factors": (
                normalized["factor"]["factor_ids"]
                if kind == FACTOR_EXPERIMENT_KIND
                else normalized["alpha"]["ordered_feature_ids"]
                if alpha_source
                else []
            ),
            **(
                {"portfolio_source": portfolio_source.model_dump(mode="json")}
                if portfolio_source
                else {}
            ),
            **({"alpha_source": alpha_source.model_dump(mode="json")} if alpha_source else {}),
            **(
                {"model_training_source": model_training_source.model_dump(mode="json")}
                if model_training_source
                else {}
            ),
            "next_action": "EXPERIMENT_RUN",
            "next_requests": {
                "run": {"operation": "EXPERIMENT_RUN", "experiment_plan_hash": plan.plan_hash},
                "inspect": _preview_readback_request(plan.plan_hash),
            },
            **lane_fields(normalized),
            "limitations": ["DEVELOPMENT_EVIDENCE_ONLY"]
            + (
                ["EXPLORATION_SAMPLE_NOT_ADMISSIBLE"]
                if research_lane(normalized) == "EXPLORATION"
                else []
            ),
        }

    def _timing(
        self,
        plan: ExperimentPlan,
        formations: tuple[str, ...] = (),
        selected_session: str | None = None,
    ) -> dict[str, Any]:
        value: dict[str, Any] = dict(
            research_timing(
                workspace=self.session.workspace,
                binding_hash=plan.binding.binding_hash,
                document=plan.document,
                preview=plan.execution_preview,
                formations=formations,
                selected_session=selected_session,
            )
        )
        if plan.portfolio_source is not None:
            source = plan.portfolio_source
            parent, _ = self._of(self.session.task_control_registry.task(source.alpha_task_id))
            value["training"].update(
                alpha_task_id=str(source.alpha_task_id),
                source="RECORDED_ALPHA_TASK_PLAN",
                split_policy=parent.execution_preview.get("split_policy"),
                folds=parent.execution_preview.get("folds"),
                maturity_lag_sessions=parent.execution_preview.get("maturity_lag_sessions"),
                lifecycle=parent.document.get("alpha", {}).get("lifecycle"),
            )
            day = selected_session or plan.execution_preview.get("statistical_start")
            folds = [
                fold
                for fold in parent.execution_preview.get("folds", ())
                if day is not None and fold["validation_start"] <= day <= fold["validation_end"]
            ]
            value["training"]["selected_fold"] = folds[0] if len(folds) == 1 else None
            clock = source.clock.binding
            value["rebalance"] = {
                "clock_id": clock.clock_id,
                "parameters": dict(clock.parameters),
                "score_session_count": plan.execution_preview.get("score_session_count"),
                "hold_session_count": plan.execution_preview.get("hold_session_count"),
                "tranches": plan.document["portfolio"].get("tranches"),
            }
        return value

    def _preview_facts(self, retained: RetainedPreview[ExperimentPlan]) -> dict[str, object]:
        return {
            "previewed_at": retained.previewed_at.isoformat(),
            "expires_at": retained.expires_at.isoformat(),
            "caller": retained.caller,
            "retention": self._previews.retention,
        }

    def preview_readback(self, plan_hash: str) -> dict[str, object]:
        """One retained preview, by exact hash, with what may lawfully follow it.

        Four honest answers and never a substitute: AVAILABLE (retained, current
        and runnable), EXPIRED (retained, past its expiry), INVALID (retained but
        the workspace moved beneath it), MISSING (this service holds no such
        preview -- evicted, never previewed here, or forgotten by a restart). An
        admitted Task carrying the same plan is named beside any of them, and a
        forgotten preview whose Task exists answers ADMITTED_AS_TASK rather than
        MISSING; the Task is read through its own readback, not through this one.
        (`ADMITTED` alone is the RUN disposition for work just sent, so a read
        never says it.)
        """
        import yaml

        entry = self._previews.get(plan_hash)
        task = self._task_for_plan_hash(plan_hash)
        body: dict[str, object] = {
            "plan_hash": plan_hash,
            "retention": self._previews.retention,
            "retained_previews": len(self._previews),
            "preview_capacity": self._previews.capacity,
        }
        if task is not None:
            body["existing_task"] = {
                "task_id": str(task.task_id),
                "lifecycle": task.lifecycle.value,
                "readback": {"operation": "EXPERIMENT_READBACK", "task_id": str(task.task_id)},
            }
        if entry is None:
            body["status"] = "ADMITTED_AS_TASK" if task is not None else "MISSING"
            body["next_action"] = "EXPERIMENT_READBACK" if task is not None else "EXPERIMENT_PLAN"
            body["explanation"] = (
                "This workspace holds no preview under this hash. A preview is kept "
                "until it expires; PLAN the declaration again explicitly."
            )
            return body
        plan = entry.plan
        status, invalid_code = "AVAILABLE", None
        if self._previews.expired(entry):
            status = "EXPIRED"
        else:
            try:
                self._current(plan)
            except AuthoringError as error:
                status, invalid_code = "INVALID", str(error)
        body.update(
            {
                "status": status,
                **self._preview_facts(entry),
                "program": plan.program.model_dump(mode="json"),
                "document": plan.document,
                "yaml": yaml.safe_dump(plan.document, sort_keys=False),
                "research_input_id": plan.binding.input_id,
                "input_binding_hash": plan.binding.binding_hash,
                "execution_preview": plan.execution_preview,
                "timing": self._timing(plan),
                "impact": plan_impact(
                    plan,
                    self._of(self.session.task_control_registry.task(plan.origin_task_id))[0]
                    if plan.origin_task_id is not None
                    else None,
                    task,
                ),
                **(
                    {"model_training_source": plan.model_training_source.model_dump(mode="json")}
                    if plan.model_training_source
                    else {}
                ),
                **({"origin_task_id": str(plan.origin_task_id)} if plan.origin_task_id else {}),
                **(
                    {"alpha_source": plan.alpha_source.model_dump(mode="json")}
                    if plan.alpha_source
                    else {}
                ),
                **(
                    {"portfolio_source": plan.portfolio_source.model_dump(mode="json")}
                    if plan.portfolio_source
                    else {}
                ),
            }
        )
        if invalid_code is not None:
            # A successful read of a preview that can no longer run. Not a
            # refusal of this read, so not `failure_code`, which every client
            # treats as one; the code names what moved beneath the preview.
            body["invalidated_by"] = invalid_code
        if status == "AVAILABLE":
            body["next_action"] = "EXPERIMENT_RUN"
            body["next_requests"] = {
                "run": {"operation": "EXPERIMENT_RUN", "experiment_plan_hash": plan_hash},
                "replan": self._replan_request(plan),
            }
        else:
            body["next_action"] = "EXPERIMENT_PLAN"
            body["next_requests"] = {"replan": self._replan_request(plan)}
        return body

    @staticmethod
    def _replan_request(plan: ExperimentPlan) -> dict[str, object]:
        """The exact PLAN request that re-creates this preview; nothing is guessed."""

        request: dict[str, object] = {
            "operation": "EXPERIMENT_PLAN",
            "research_input_id": plan.binding.input_id,
            "input_binding_hash": plan.binding.binding_hash,
            "experiment_document": plan.document,
        }
        if plan.origin_task_id is not None:
            request["origin_task_id"] = str(plan.origin_task_id)
        if plan.alpha_source is not None:
            request["factor_task_id"] = str(plan.alpha_source.factor_task_id)
            request["curation_receipt_hash"] = plan.alpha_source.curation_receipt_hash
        return request

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the retained study's exact declaration, including its parent selections."""
        plan, _actor = self._of(task)
        return self._replan_request(plan)

    def _replan_refusal(
        self, failure_code: str, entry: RetainedPreview[ExperimentPlan]
    ) -> dict[str, object]:
        return {
            "status": "REFUSED",
            "failure_code": failure_code,
            "plan_hash": entry.plan.plan_hash,
            "expires_at": entry.expires_at.isoformat(),
            "next_action": "EXPERIMENT_PLAN",
            "next_requests": {
                "replan": self._replan_request(entry.plan),
                "inspect": _preview_readback_request(entry.plan.plan_hash),
            },
        }

    def _task_for_plan_hash(self, plan_hash: str) -> TaskRecord | None:
        """The newest non-cancelled Task admitted from exactly this plan hash."""

        found = [
            task
            for task in self.session.task_control_registry.tasks()
            if task.task_kind == TASK_KIND
            and task.lifecycle is not TaskLifecycle.CANCELLED
            and cast(dict[str, Any], task.input.payload.get("plan", {})).get("plan_hash")
            == plan_hash
        ]
        return max(found, key=lambda task: task.updated_at) if found else None

    def blocked_retry_reason(self, task: TaskRecord) -> str | None:
        """Why a BLOCKED Task of this kind may be retried, or nothing.

        Read by the recovery view; it admits nothing and runs nothing.
        """
        if task.task_kind != TASK_KIND or task.lifecycle is not TaskLifecycle.BLOCKED:
            return None
        code = retryable_block(task.failure_code)
        if code is None:
            return None
        if code == RETRYABLE_CAPACITY_BLOCK:
            return (
                "The stage stopped on storage.managed_capacity_exceeded. Raise the cap in "
                "Settings or confirm a cleanup, then RECOVER with this version re-checks the "
                "current capacity and plan and reopens the same Task; its verified stages and "
                "checkpoint are kept."
            )
        return (
            f"The stage stopped on {code}: a required artifact was missing or did not read. "
            "Once it is restored, RECOVER with this version re-checks the plan against the "
            "workspace and reopens the same Task; its verified stages and checkpoint are kept."
        )

    def retry_blocked(
        self, task_id: UUID, *, now: datetime, expected_task_hash: str
    ) -> dict[str, object]:
        """Reopen a BLOCKED Task whose retryable cause a person says is repaired.

        The Task must be BLOCKED on an artifact or capacity stop this owner retries; its
        plan must still be current (input binding, implementation, program); it
        is then handed back to Task Control as recovery-required, keeping its
        BLOCKED history and every verified stage. Nothing numerical runs here:
        the runner's next attempt re-verifies the prefix and the checkpoint
        before any estimate, and stops again, by name, if the cause remains.
        A plan that is no longer current is not reopened; the answer names the
        new plan the person would have to make.

        `expected_task_hash` is the version the person confirmed. It is the
        version the reopening is written against, inside Task Control's own
        transaction: a Task that moved while the plan was being re-checked --
        reopened by a competing confirmation, started, cancelled, or blocked
        again -- is refused as stale with nothing applied, never reset to
        recovery from whatever it became.
        """
        registry = self.session.task_control_registry
        task = registry.task(task_id)
        if task.task_kind != TASK_KIND:
            raise AuthoringError("research_experiment.task_kind_mismatch")
        if task.record_hash != expected_task_hash:
            return self._stale_retry(task)
        if task.lifecycle is not TaskLifecycle.BLOCKED:
            return {
                "disposition": "NOT_BLOCKED",
                "task_id": str(task_id),
                "lifecycle": task.lifecycle.value,
                "failure_code": task.failure_code,
            }
        code = retryable_block(task.failure_code)
        if code is None:
            return {
                "disposition": "REFUSED",
                "failure_code": "research_experiment.block_not_retryable",
                "task_id": str(task_id),
                "blocked_on": task.failure_code,
                "detail": (
                    "This Task stopped on a refusal that is not a missing or unreadable "
                    "artifact; repairing files does not answer it. A changed declaration is "
                    "a new PLAN."
                ),
                "next_requests": {
                    "recovery": {"operation": "TASK_RECOVERY", "task_id": str(task_id)}
                },
            }
        plan, _actor = self._of(task)
        try:
            self._current(plan, task_id=task_id)
        except AuthoringError as error:
            return {
                "disposition": "REFUSED",
                **located_failure(error, "research_experiment.refused"),
                "task_id": str(task_id),
                "blocked_on": task.failure_code,
                "detail": (
                    "The execution binding of this Task no longer holds (the input, source or "
                    "implementation moved); it cannot be reopened. Plan the same declaration "
                    "again under the current binding."
                ),
                # The same request an ordinary re-PLAN offers, the upstream selection kept
                # (V140).
                "next_requests": {"plan": self._replan_request(plan)},
            }
        finally:
            self._admitted_workflows.pop(task_id, None)
        if code == RETRYABLE_CAPACITY_BLOCK:
            try:
                require_storage_capacity(self.session.workspace, additional_bytes=0)
            except StorageInventoryError as error:
                return {
                    "disposition": "REFUSED",
                    "failure_code": error.failure_code,
                    "detail": str(error),
                    "task_id": str(task_id),
                    "task_record_hash": task.record_hash,
                    "next_requests": {
                        "recovery": {"operation": "TASK_RECOVERY", "task_id": str(task_id)}
                    },
                }
        try:
            reopened = registry.mark_recovery_required(
                task_id=task_id,
                failure_code=task.failure_code or code,
                observed_at=now,
                allow_blocked=True,
                expected_task_hash=expected_task_hash,
            )
        except TaskVersionStale:
            return self._stale_retry(registry.task(task_id))
        if reopened.lifecycle is not TaskLifecycle.RECOVERY_REQUIRED:
            return {
                "disposition": "REFUSED",
                "failure_code": "research_experiment.block_not_reopened",
                "task_id": str(task_id),
                "lifecycle": reopened.lifecycle.value,
            }
        return {
            "disposition": "RETRY_ADMITTED",
            "task_id": str(task_id),
            "lifecycle": reopened.lifecycle.value,
            "task_record_hash": reopened.record_hash,
            "blocked_on": task.failure_code,
            "detail": (
                "The Task is reopened for a new attempt under the same execution contract; its "
                "verified stages and checkpoint are re-verified and reused, and only the "
                "unfinished part is computed."
            ),
        }

    @staticmethod
    def _stale_retry(current: TaskRecord) -> dict[str, object]:
        """The confirmed version is gone; nothing was applied. The current one is named."""

        return {
            "disposition": "REFUSED",
            "failure_code": "task_control.recovery_version_stale",
            "task_id": str(current.task_id),
            "lifecycle": current.lifecycle.value,
            "task_record_hash": current.record_hash,
        }

    def _of(self, task: TaskRecord) -> tuple[ExperimentPlan, ActorSubmissionBinding]:
        """A Task's plan and actor, checked against the Task's own contract once per Task.

        A Task's input, goal and plan never change, so their hashes key the check: a listing
        re-parsed and re-hashed every Task's plan on every call (W10, measured: a third of a
        light read's time on fifteen Tasks).
        """
        key = (task.task_id, task.input.input_hash, task.goal.goal_hash, task.plan.plan_hash)
        held = self._task_plans.get(key)
        if held is not None:
            return held
        checked = self._checked_plan(task)
        self._task_plans[key] = checked
        return checked

    def _checked_plan(self, task: TaskRecord) -> tuple[ExperimentPlan, ActorSubmissionBinding]:
        if task.task_kind != TASK_KIND:
            raise AuthoringError("research_experiment.task_kind_mismatch")
        plan = ExperimentPlan.model_validate(task.input.payload.get("plan"))
        actor = ActorSubmissionBinding.model_validate(task.input.payload.get("actor"))
        envelope, goal, workflow = _task_contract(plan, actor)
        if (
            task.input != envelope
            or task.goal != goal
            or task.plan != workflow
            or plan.workspace_id != self.manifest.workspace_id
        ):
            raise AuthoringError("research_experiment.task_binding_mismatch")
        return plan, actor

    def _current(  # type: ignore[no-untyped-def]
        self, plan: ExperimentPlan, *, task_id: UUID | None = None, replay: bool = False
    ):
        if sample_standing(plan.document, plan.authority):
            raise AuthoringError(SAMPLE_SCHEME_SUPERSEDED)
        if self._binding(
            plan.binding.input_id, plan.binding.binding_hash
        ) != plan.binding or not is_current(
            plan_implementation_role(plan),
            plan.implementation_hash,
            plan_implementation_hash(plan),
        ):
            raise AuthoringError("research_experiment.execution_binding_changed")
        if plan.qualification_family is not None:
            workflow, sealed, preview, _normalized, family, _binding, _excluded = (
                self._qualification(
                    plan.document,
                    cancelled=lambda: (
                        task_id is not None
                        and self.session.task_control_registry.task(task_id).lifecycle
                        is TaskLifecycle.CANCEL_REQUESTED
                    ),
                )
            )
            if (
                family != plan.qualification_family
                or preview != plan.execution_preview
                or sealed.program != plan.program
            ):
                raise AuthoringError("alpha_research.qualification_family_changed")
            authority = sealed.authority
        elif plan.model_training_source is not None:
            workflow, sealed, preview, _normalized, source = lifecycle_workflow(
                workspace=self.session.workspace,
                binding=plan.binding,
                document=plan.document,
                cancelled=lambda: (
                    task_id is not None
                    and self.session.task_control_registry.task(task_id).lifecycle
                    is TaskLifecycle.CANCEL_REQUESTED
                ),
            )
            if (
                source != plan.model_training_source
                or preview != plan.execution_preview
                or sealed.program != plan.program
            ):
                raise AuthoringError("alpha_research.lifecycle_execution_plan_changed")
            authority = sealed.authority
        elif plan.program.kind == RISK_EXPERIMENT_KIND:
            workflow, authority, preview, normalized = risk_workflow(
                session=self.session,
                binding=plan.binding,
                document=plan.document,
                cancelled=None
                if task_id is None
                else lambda: (
                    self.session.task_control_registry.task(task_id).lifecycle
                    is TaskLifecycle.CANCEL_REQUESTED
                ),
            )
            sealed = workflow.prepare(normalized, actor_kind=ActorKind.HUMAN, actor_id="preview")
            # A replay reads sealed evidence, so the estimators' recorded moves carry its Program
            # (V314); a run stays exact, since its executor seals what it computes.
            if preview != plan.execution_preview or not (
                sealed.program == plan.program
                or (
                    replay
                    and replay_follows_recorded_moves(
                        plan.program,
                        sealed.program,
                        envelope=sealed.envelope,
                        document=sealed.document,
                        authority=sealed.authority,
                    )
                )
            ):
                raise AuthoringError("risk_research.execution_plan_changed")
        elif plan.portfolio_source is not None:
            prepared_portfolio, _binding = self._prepare_portfolio(
                UUID(plan.portfolio_source.alpha_task_id),
                plan.portfolio_source.candidate_id,
                plan.document,
                task_id,
            )
            if (
                prepared_portfolio.source != plan.portfolio_source
                or prepared_portfolio.preview != plan.execution_preview
            ):
                raise AuthoringError("portfolio_research.source_changed")
            workflow = prepared_portfolio.workflow()
            sealed = workflow.prepare(plan.document, actor_kind=ActorKind.HUMAN, actor_id="preview")
            if sealed.program != plan.program:
                raise AuthoringError("portfolio_research.program_changed")
            authority = sealed.authority
        elif plan.alpha_source is not None:
            # A named Foundation admission is proved inside ``_prepare_alpha``,
            # on the handoff this check prepares.
            prepared, source = self._prepare_alpha(
                plan.binding,
                plan.document,
                plan.alpha_source.factor_task_id,
                plan.alpha_source.curation_receipt_hash,
            )
            if source != plan.alpha_source:
                raise AuthoringError("alpha_research.parent_evidence_changed")
            # A Task's fits run in a child process, so the Host's reads never wait behind
            # them for the GIL (W10); the Host relays the Task's cancellation to it.
            workflow = (
                prepared.workflow()
                if task_id is None
                else prepared.workflow(
                    prepared.in_child(
                        lambda: (
                            self.session.task_control_registry.task(task_id).lifecycle
                            is TaskLifecycle.CANCEL_REQUESTED
                        )
                    )
                )
            )
            sealed = workflow.prepare(plan.document, actor_kind=ActorKind.HUMAN, actor_id="preview")
            authority = sealed.authority
            preview = prepared.executor.prepare_execution(
                program=sealed.program,
                document=plan.document,
                authority=authority,
            ).describe()
            if preview != plan.execution_preview or sealed.program != plan.program:
                raise AuthoringError("alpha_research.execution_plan_changed")
        else:
            workflow, authority, _preview = factor_workflow(
                workspace=self.session.workspace,
                binding_hash=plan.binding.binding_hash,
                document=plan.document,
                feature_input=self._feature_input(
                    plan.document["experiment"]["data_snapshot_handle"], plan.binding.binding_hash
                ),
            )
        if authority != plan.authority:
            raise AuthoringError("research_experiment.input_authority_changed")
        return workflow

    def book_source(self, task_id: UUID) -> dict[str, str | None] | None:
        """The Alpha study and candidate a Portfolio study's plan names (V511).

        The way on from its stopped Task is bound to them: its draft request named neither.

        Args:
            task_id: The stopped Task.

        Returns:
            The Alpha study's Task id and the candidate, or none for another Task.
        """
        try:
            task = self.session.task_control_registry.task(task_id)
            if task.task_kind != TASK_KIND:
                return None
            plan = ExperimentPlan.model_validate(task.input.payload["plan"])
        except (KeyError, ValueError):
            return None
        source = plan.portfolio_source
        if source is None:
            return None
        return {"task_id": source.alpha_task_id, "candidate_id": source.candidate_id}

    def study_facts(self, task_id: UUID | None, *, moved: bool = False) -> dict[str, object]:
        """Read declared study facts and installed-binding differences without verification.

        What a refusal about this study can say without verifying anything: its ID, its
        kind and the numerical calls its continuation declares; empty when unknown. With
        `moved`, which of its recorded identities differ from what is installed: the kind's
        implementation (successors consulted) and its research input's binding.
        """
        if task_id is None:
            return {}
        try:
            task = self.session.task_control_registry.task(task_id)
            if task.task_kind != TASK_KIND:
                # Another owner's Task: its own kind, which the refusal names (V490).
                return {"task_id": str(task_id), "kind": task.task_kind}
            plan = ExperimentPlan.model_validate(task.input.payload["plan"])
        except (KeyError, ValueError):
            return {"task_id": str(task_id)}
        facts: dict[str, object] = {
            "task_id": str(task_id),
            "kind": plan.program.kind,
            "refresh_calls": plan.execution_preview.get("expected_numerical_calls"),
        }
        if moved:
            changes: list[dict[str, str | None]] = []
            installed = _implementation_hash(plan.program.kind)
            role = plan_implementation_role(plan)
            if not is_current(role, plan.implementation_hash, installed):
                changes.append(
                    {
                        "identity": "implementation",
                        "recorded": plan.implementation_hash,
                        "installed": installed,
                    }
                )
            try:
                binding = self._binding(plan.binding.input_id, plan.binding.binding_hash)
            except (ValueError, OSError):
                binding = None
            if binding != plan.binding:
                changes.append(
                    {
                        "identity": "research_input",
                        "recorded": plan.binding.binding_hash,
                        "installed": None if binding is None else binding.binding_hash,
                    }
                )
            facts["moved"] = changes
        return facts

    @staticmethod
    def _same_execution(left: ExperimentPlan, right: ExperimentPlan) -> bool:
        # Derivation metadata cannot force duplicate numerical work, and neither can
        # an implementation recorded as the predecessor of the other's.
        fields = {"plan_hash", "origin_task_id", "implementation_hash"}
        role = plan_implementation_role(left)
        return bool(
            left.model_dump(mode="json", exclude=fields)
            == right.model_dump(mode="json", exclude=fields)
            and (
                is_current(role, left.implementation_hash, right.implementation_hash)
                or is_current(role, right.implementation_hash, left.implementation_hash)
            )
        )

    def draft(
        self, task_id: UUID, input_id: str | None, binding_hash: str | None
    ) -> dict[str, object]:
        """Copy a completed declaration onto an explicitly selected admitted input.

        Args:
            task_id: Exact completed origin study.
            input_id: Optional selected input; defaults to the origin input.
            binding_hash: Optional exact selected input revision.

        Returns:
            Authored document/YAML and a new PLAN request with dates preserved and exact upstream
            rechecks.

        Raises:
            AuthoringError: Origin is incomplete or a prepared/lifecycle/Portfolio source cannot
                move to this input.
        """
        body = self.readback(task_id)
        if body["status"] != "EXPERIMENT_PUBLISHED":
            raise AuthoringError("research_experiment.completed_origin_required")
        prior, _actor = self._of(self.session.task_control_registry.task(task_id))
        binding = self._binding(
            *draft_revision(
                (prior.binding.input_id, prior.binding.binding_hash), input_id, binding_hash
            )
        )
        if prior.portfolio_source is not None and binding != prior.binding:
            raise AuthoringError("portfolio_research.input_not_alpha_source")
        # An Alpha draft's handoff below verifies this bundle's files; its own read only
        # names the Panel (binding plan, L2: the bundle is digested once per draft).
        bundle = read_factor_bundle(
            self.session.workspace, binding.binding_hash, verify=prior.alpha_source is None
        )
        document = json.loads(json.dumps(prior.document))
        # A document authored before V128 may carry the Host's own identity; a draft never does.
        document["experiment"].pop("envelope_hash", None)
        document["experiment"].update(
            data_snapshot_handle=bundle.panel_snapshot_hash,
            output_workspace="managed",
            baseline_workspace="managed",
        )
        if prior.document["experiment"]["data_snapshot_handle"].startswith("research-features@"):
            if binding != prior.binding:
                raise AuthoringError(
                    "research_experiment.prepared_features_new_input_requires_build"
                )
            document["experiment"]["data_snapshot_handle"] = prior.document["experiment"][
                "data_snapshot_handle"
            ]
        if prior.model_training_source is not None:
            if binding != prior.binding:
                raise AuthoringError("alpha_research.lifecycle_new_input_requires_preparation")
            document["experiment"]["data_snapshot_handle"] = (
                prior.model_training_source.source_handle
            )
        if prior.alpha_source is not None:
            document["experiment"].update(
                output_workspace=prior.document["experiment"]["output_workspace"],
                baseline_workspace=f"research-inputs/{binding.binding_hash}/source",
            )
            self._prepare_alpha(
                binding,
                document,
                prior.alpha_source.factor_task_id,
                prior.alpha_source.curation_receipt_hash,
            )
        import yaml

        return {
            "status": "DRAFT_READY",
            "plan_request": {
                "operation": "EXPERIMENT_PLAN",
                "research_input_id": binding.input_id,
                "input_binding_hash": binding.binding_hash,
                "origin_task_id": str(task_id),
                **(
                    {
                        "factor_task_id": str(prior.alpha_source.factor_task_id),
                        "curation_receipt_hash": prior.alpha_source.curation_receipt_hash,
                    }
                    if prior.alpha_source
                    else {}
                ),
            },
            "origin_task_id": str(task_id),
            **(
                {"alpha_source": prior.alpha_source.model_dump(mode="json")}
                if prior.alpha_source
                else {}
            ),
            "research_input_id": binding.input_id,
            "input_binding_hash": binding.binding_hash,
            "document": document,
            "yaml": yaml.safe_dump(document, sort_keys=False),
            "before": {
                "input_binding_hash": prior.binding.binding_hash,
                "sessions": prior.document["experiment"]["sessions"],
                "execution_preview": prior.execution_preview,
            },
            "after": {
                "input_binding_hash": binding.binding_hash,
                "sessions": document["experiment"]["sessions"],
                "input_start": str(bundle.sessions[0]),
                "input_end": str(bundle.sessions[-1]),
            },
            "numerical_call_count": 0,
            "limitations": [
                "DATES_PRESERVED",
                "PREVIEW_REQUIRED",
                "FACTOR_DECISION_REVERIFIED" if prior.alpha_source else "NO_CURATION_INHERITED",
                "POLICY_STATISTICAL_WINDOW_MAY_DIFFER_FROM_AUTHORED_INTERVAL",
            ],
        }

    def promote(
        self, task_id: UUID, *, caller: str, agent_execution: AgentExecutionBinding | None
    ) -> dict[str, object]:
        """Run an explored declaration on its exact input's whole universe.

        Run an explored study's declaration on its input's whole universe: the one step
        from the exploration lane to the promotion lane (binding plan, B17).

        An Alpha study is planned again with its universe widened and run, in this one
        request; a Portfolio study builds on its Alpha study's promotion, which this
        request runs first when it has not run. Reuse answers as `run` does, so asking
        twice admits nothing twice.
        """
        plan, _actor = self._of(self.session.task_control_registry.task(task_id))
        if research_lane(plan.document) == "PROMOTION":
            return {
                "status": "ALREADY_PROMOTION",
                "task_id": str(task_id),
                "research_lane": "PROMOTION",
                "numerical_call_count": 0,
            }
        if plan.alpha_source is not None:
            draft = self.draft(task_id, None, None)
            document = cast(dict[str, Any], draft["document"])
            document["experiment"]["universe_handle"] = universe_profile(
                str(document["experiment"]["universe_handle"])
            )
            planned = self.plan(
                str(draft["research_input_id"]),
                document,
                None,
                str(draft["input_binding_hash"]),
                task_id,
                plan.alpha_source.factor_task_id,
                plan.alpha_source.curation_receipt_hash,
                caller=caller,
            )
        elif plan.portfolio_source is not None:
            alpha_task_id = UUID(plan.portfolio_source.alpha_task_id)
            alpha = self.promote(alpha_task_id, caller=caller, agent_execution=agent_execution)
            promoted_alpha = (
                alpha_task_id
                if alpha["status"] == "ALREADY_PROMOTION"
                else UUID(str(alpha.get("task_id") or alpha.get("publication_task_id")))
            )
            if alpha["status"] not in {"ALREADY_PROMOTION", "REUSED_EXACT"}:
                return {
                    "status": "UPSTREAM_PROMOTION_ADMITTED",
                    "task_id": str(task_id),
                    # The Task this answer started, which a wait follows (V137).
                    "follow_task_id": str(promoted_alpha),
                    "alpha_promotion": alpha,
                    "detail": (
                        "The Alpha study this Portfolio study builds on is running on the whole "
                        "universe first. Ask again once it has succeeded; the Portfolio study "
                        "then runs on it."
                    ),
                    "next_requests": {
                        # The Task this answer started, which a saved answer's wait follows
                        # as `--wait` does (V137, V446).
                        "task": {"operation": "STATUS", "task_id": str(promoted_alpha)},
                        "alpha": {
                            "operation": "EXPERIMENT_READBACK",
                            "task_id": str(promoted_alpha),
                        },
                        "promote": {"operation": "EXPERIMENT_PROMOTE", "task_id": str(task_id)},
                    },
                    "numerical_call_count": alpha.get("numerical_call_count", 0),
                }
            document, binding = self.portfolio_document(promoted_alpha, plan.document["portfolio"])
            planned = self.plan(
                binding.input_id, document, None, binding.binding_hash, task_id, caller=caller
            )
        else:
            raise AuthoringError(f"research_lane.promotion_kind_not_supported:{plan.program.kind}")
        if planned.get("status") != "PLANNED":
            return planned
        sent = self.run(str(planned["plan_hash"]), caller=caller, agent_execution=agent_execution)
        return {
            **sent,
            "promoted_from_task_id": str(task_id),
            "promotion_plan_hash": planned["plan_hash"],
            "research_lane": "PROMOTION",
        }

    def continue_study(
        self, task_id: UUID, *, caller: str, agent_execution: AgentExecutionBinding | None
    ) -> dict[str, object]:
        """Continue a saved study as a new draft, planned and run in one request (CLI-2).

        The same three steps a person takes -- its continuation draft, that draft's plan, and
        the run -- so the answer is exactly RUN's (reuse included: asking twice admits nothing
        twice), with the plan it admitted beside it.
        """
        draft = self.draft(task_id, None, None)
        request = cast(dict[str, Any], draft["plan_request"])
        planned = self.plan(
            str(request["research_input_id"]),
            cast(dict[str, Any], draft["document"]),
            None,
            str(request["input_binding_hash"]),
            UUID(str(request["origin_task_id"])) if request.get("origin_task_id") else None,
            UUID(str(request["factor_task_id"])) if request.get("factor_task_id") else None,
            cast(str | None, request.get("curation_receipt_hash")),
            caller=caller,
        )
        if planned.get("status") != "PLANNED":
            return planned
        sent = self.run(str(planned["plan_hash"]), caller=caller, agent_execution=agent_execution)
        return {
            **sent,
            "continued_from_task_id": str(task_id),
            "continuation_plan_hash": planned["plan_hash"],
            "continuation_plan": {
                k: planned[k]
                for k in ("execution_preview", "declaration_changes", "research_lane")
                if k in planned
            },
        }

    def portfolio_document(
        self,
        alpha_task_id: UUID,
        spec: Mapping[str, Any],
        *,
        candidate_id: str | None = None,
    ) -> tuple[dict[str, Any], ResearchWorkspaceExperimentInput]:
        """Carry a Portfolio declaration onto an explicitly selected Alpha study.

        A Portfolio study's declaration, carried onto another Alpha study: the envelope
        that Alpha derives, and the same policy on it (its candidate, unless another is
        named). What a promotion and a feature trial plan.
        """
        candidate = candidate_id or str(spec["candidate_id"])
        prepared, binding = self._prepare_portfolio(alpha_task_id, candidate)
        document = {
            **cast(dict[str, Any], portfolio_draft(prepared)["document"]),
            "portfolio": {
                **dict(spec),
                "alpha_task_id": str(alpha_task_id),
                "candidate_id": candidate,
            },
        }
        return document, binding

    def _matching_task(self, plan: ExperimentPlan) -> TaskRecord | None:
        return next(
            (
                task
                for task in self.session.task_control_registry.tasks()
                if task.task_kind == TASK_KIND
                and self._same_execution(self._of(task)[0], plan)
                and task.lifecycle is not TaskLifecycle.CANCELLED
            ),
            None,
        )

    def run(
        self, plan_hash: str, *, caller: str, agent_execution: AgentExecutionBinding | None
    ) -> dict[str, object]:
        """Admit the caller and reuse or submit only the exact previewed experiment.

        Args:
            plan_hash: Exact bounded preview or durable admitted plan identity.
            caller: Declared operation caller.
            agent_execution: Exact installed-agent execution binding when applicable.

        Returns:
            Existing task, explicit replan refusal or bounded dispatcher admission.

        Raises:
            AuthoringError: Neither an exact preview nor its admitted task exists, or caller
                authority is absent.
        """
        entry = self._previews.get(plan_hash)
        # No preview here: after a restart or eviction the only lawful answers
        # are the Task this exact plan already admitted, or a fresh explicit
        # PLAN. Never the newest preview of someone else.
        admitted = None if entry is not None else self._task_for_plan_hash(plan_hash)
        if entry is None and admitted is None:
            raise AuthoringError("research_experiment.preview_required")
        plan = entry.plan if entry is not None else self._of(admitted)[0]
        # The caller is admitted before any answer -- reuse of an existing
        # Task, a re-PLAN refusal or a submission -- whatever the preview's
        # retention, expiry or the service's restarts. Reading stays separate:
        # EXPERIMENT_READBACK and EXPERIMENT_PREVIEW_READBACK need no binding.
        actor = self._admit_actor(plan, caller=caller, agent_execution=agent_execution)
        if entry is None:
            assert admitted is not None
            return self._existing_task_answer(admitted)
        if self._previews.expired(entry):
            return self._replan_refusal("research_experiment.preview_expired", entry)
        try:
            self._current(plan)
        except AuthoringError as error:
            return self._replan_refusal(str(error), entry)
        task = self._matching_task(plan)
        if task is not None:
            return self._existing_task_answer(task)
        sent = self.dispatcher.submit(ResearchExperimentCommand(self, plan, actor))
        return {
            "status": sent.disposition,
            "task_id": str(sent.task_id) if sent.task_id else None,
            "lifecycle": sent.lifecycle,
            "failure_code": sent.refusal_detail,
        }

    @staticmethod
    def _admit_actor(
        plan: ExperimentPlan, *, caller: str, agent_execution: AgentExecutionBinding | None
    ) -> ActorSubmissionBinding:
        """Who may RUN this exact program: an installed Agent only with its execution binding."""

        if caller == "INSTALLED_AGENT" and agent_execution is None:
            raise AuthoringError("research_experiment.agent_execution_not_admitted")
        return seal_actor_submission(
            actor_kind=ActorKind(caller),
            actor_id=f"local-web-{caller.lower()}",
            submission_hash=plan.program.program_hash,
            agent_execution=agent_execution,
        )

    def _existing_task_answer(self, task: TaskRecord) -> dict[str, object]:
        """The one answer for a plan that already has a Task: reuse, join or report."""

        if task.lifecycle is TaskLifecycle.SUCCEEDED:
            reused = self.readback(task.task_id)
            recorded = cast(dict[str, Any], reused.get("realization") or {"recorded": False})
            return {
                "status": "REUSED_EXACT",
                "task_id": None,
                "publication_task_id": str(task.task_id),
                "numerical_call_count": 0,
                "realization": recorded,
                "detail": reuse_words(recorded),
            }
        answer: dict[str, object] = {
            "status": (
                "REUSED_IN_FLIGHT"
                if task.lifecycle
                in {
                    TaskLifecycle.QUEUED,
                    TaskLifecycle.RUNNING,
                    TaskLifecycle.RECOVERY_REQUIRED,
                }
                else task.lifecycle.value
            ),
            "task_id": str(task.task_id),
            "lifecycle": task.lifecycle.value,
            "failure_code": task.failure_code,
        }
        if task.lifecycle is TaskLifecycle.BLOCKED:
            # RUN never reopens a BLOCKED Task on its own. The same declaration
            # is that Task; what a person can do with it is named here, so the
            # answer is a next step rather than a wall.
            reason = self.blocked_retry_reason(task)
            answer["next_requests"] = (
                {
                    "recover": {
                        "operation": "RECOVER",
                        "task_id": str(task.task_id),
                        "expected_task_hash": task.record_hash,
                    }
                }
                if reason is not None
                else {"recovery": {"operation": "TASK_RECOVERY", "task_id": str(task.task_id)}}
            )
            answer["detail"] = reason or (
                "This declaration is the BLOCKED Task above; its refusal is not a repaired "
                "artifact, so it is not retried. A changed declaration is a new PLAN."
            )
        return answer

    def admit(self, plan: ExperimentPlan, actor: ActorSubmissionBinding) -> CommandAdmission:
        """Admit the plan `run` verified in this same request; never a second verification.

        The dispatcher calls this synchronously from `run`'s submission, on the
        caller's thread, right after `run` proved the plan current
        (`_current`) and admitted the actor; `ResearchExperimentCommand` is
        the only route here with a plan. The execution verifies again before
        any numerical work (`compatibility`, then the first stage).
        """
        envelope, goal, workflow = _task_contract(plan, actor)
        registry = self.session.task_control_registry
        for task in registry.tasks():
            if (
                task.task_kind == TASK_KIND
                and task.lifecycle is not TaskLifecycle.CANCELLED
                and self._same_execution(self._of(task)[0], plan)
            ):
                return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)
        task = registry.admit(
            input_envelope=envelope, goal=goal, plan=workflow, observed_at=self.clock()
        ).record
        return CommandAdmission(task_id=task.task_id, lifecycle=task.lifecycle.value)

    def _evidence(
        self, task: TaskRecord, *, recorded_readback: bool = False
    ) -> ResearchExecutionEvidence:
        return self._verified_evidence(task, recorded_readback=recorded_readback)[0]

    def _verified_evidence(
        self,
        task: TaskRecord,
        *,
        recorded_readback: bool = False,
        plan_actor: tuple[ExperimentPlan, ActorSubmissionBinding] | None = None,
    ) -> tuple[ResearchExecutionEvidence, ResearchProgramWorkflow, ResearchExecutionEvidence]:
        """The Task's evidence read back through its Desk verifier, with the workflow
        that holds that verifier, so the request that had the graph walked can
        project what the walk proved instead of walking it a second time."""

        scope = _VERIFIED_EVIDENCE.get()
        key = (task.task_id, task.record_hash, recorded_readback)
        if scope is not None and key in scope:
            return cast(
                tuple[
                    ResearchExecutionEvidence, ResearchProgramWorkflow, ResearchExecutionEvidence
                ],
                scope[key],
            )
        plan, actor = plan_actor if plan_actor is not None else self._of(task)
        stored: list[ResearchExecutionEvidence] = []
        workflow = build_research_program_workflow(
            workspace=factor_input_paths(self.session.workspace, plan.binding.binding_hash)[0],
            workspace_root=self.session.workspace,
            executors=(),
            verifier_kinds=(plan.program.kind,),
        )
        evidence = workflow.readback_sealed(
            plan.document,
            program_hash=plan.program.program_hash,
            actor_kind=actor.actor_kind,
            actor_id=actor.actor_id,
            agent_execution=actor.agent_execution,
            authority=plan.authority,
            recorded_readback=recorded_readback,
            stored_sink=stored,
        )[0]
        _check_evidence_window(plan, evidence)
        if scope is not None:
            scope[key] = (evidence, workflow, stored[0])
        return evidence, workflow, stored[0]

    def _study_fingerprint(self, plan: ExperimentPlan) -> tuple[str, int]:
        """Every file a study's verification reads, by path, size, modification time and file
        identity (decision 5: a copy, a restore or a sync changes one of them), and how many."""

        roots = (
            confined(self.session.workspace, plan.document["experiment"]["output_workspace"]),
            factor_input_paths(self.session.workspace, plan.binding.binding_hash)[0].parent,
        )
        rows = []
        for root in roots:
            for directory, _dirs, names in os.walk(root):
                for name in names:
                    stat = os.stat(os.path.join(directory, name))
                    rows.append(
                        f"{directory}/{name}\t{stat.st_size}\t{stat.st_mtime_ns}\t{stat.st_ino}"
                    )
        digest = hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()
        return digest, len(rows)

    def _installed_implementation(self, plan: ExperimentPlan) -> str:
        role = plan_implementation_role(plan)
        if role not in self._installed:
            self._installed[role] = plan_implementation_hash(plan)
        return self._installed[role]

    def _ledger_path(self, key: tuple[UUID, bool]) -> Path:
        mode = "recorded" if key[1] else "current"
        return (
            self.session.workspace
            / "runtime"
            / VERIFICATION_LEDGER_DIRECTORY
            / f"{key[0]}-{mode}.json"
        )

    def _kept_on_disk(self, key: tuple[UUID, bool]) -> _VerifiedStudy | None:
        """An Alpha study's verification kept on disk; a damaged entry costs a full check."""
        try:
            entry = json.loads(self._ledger_path(key).read_text(encoding="utf-8"))
            if entry["schema_version"] != 1 or entry["task_id"] != str(key[0]):
                return None
            return _VerifiedStudy(
                evidence=ResearchExecutionEvidence.model_validate(entry["evidence"]),
                original=ResearchExecutionEvidence.model_validate(entry["original"]),
                currency=dict(entry["currency"]),
                view=dict(entry["view"]),
                record_hash=str(entry["record_hash"]),
                fingerprint=str(entry["fingerprint"]),
                verified_at=datetime.fromisoformat(entry["verified_at"]),
                files=int(entry["files"]),
                implementation=str(entry["implementation"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _keep_on_disk(self, key: tuple[UUID, bool], study: _VerifiedStudy) -> None:
        """Keep an Alpha study's full verification as its read shows it (V89)."""
        try:
            text = json.dumps(
                {
                    "schema_version": 1,
                    "task_id": str(key[0]),
                    "recorded_readback": key[1],
                    "record_hash": study.record_hash,
                    "fingerprint": study.fingerprint,
                    "files": study.files,
                    "implementation": study.implementation,
                    "verified_at": study.verified_at.isoformat(),
                    "evidence": study.evidence.model_dump(mode="json"),
                    "original": study.original.model_dump(mode="json"),
                    "currency": study.currency,
                    "view": study.view,
                },
                sort_keys=True,
            )
        except (TypeError, ValueError):
            return  # a view that is not plain data is not kept; the next read verifies it
        path = self._ledger_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Each writer stages its own file; a read another request holds open on Windows may
        # still keep the entry from being replaced, and then this read stands verified and
        # the next one verifies again: keeping is a cache, never a reason to refuse (V477).
        staged = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
        try:
            staged.write_text(text, encoding="utf-8")
            replace_shared_file(staged, path)
        except OSError:
            staged.unlink(missing_ok=True)

    def _verified_study(
        self,
        task: TaskRecord,
        plan: ExperimentPlan,
        actor: ActorSubmissionBinding,
        *,
        recorded_readback: bool,
        output: Path,
        graph_sink: list[AlphaDevelopmentVerifiedGraph] | None,
    ) -> tuple[_VerifiedStudy, str]:
        """The study as its last full verification proved it, or verified in full now.

        Answers with the basis: ``FULL``, or ``FILES_UNCHANGED ...`` when a ledger read
        answered from an earlier full verification of the same Task record under the same
        installed implementation, whose files all kept their path, size, modification time
        and file identity: what was checked, and no more (V265). A caller that takes the
        Alpha graph itself (``graph_sink``, a comparison) always verifies in full.
        """

        key = (task.task_id, recorded_readback)
        fingerprint, files = self._study_fingerprint(plan)
        installed = self._installed_implementation(plan)
        if _REUSE_VERIFIED.get() and graph_sink is None:
            with self._ledger_lock:
                kept = self._ledger.get(key)
            if kept is None and plan.alpha_source is not None:
                kept = self._kept_on_disk(key)
            if (
                kept is not None
                and kept.record_hash == task.record_hash
                and kept.fingerprint == fingerprint
                and kept.implementation == installed
            ):
                return kept, (
                    f"FILES_UNCHANGED {kept.files} files kept their path, size, time and "
                    f"identity since the full check at {kept.verified_at.isoformat()}"
                )
        evidence, workflow, original = self._verified_evidence(
            task, recorded_readback=recorded_readback, plan_actor=(plan, actor)
        )
        study = _VerifiedStudy(
            evidence=evidence,
            original=original,
            currency=method_currency(workflow.verifier(plan.program.kind)),
            view=self._verified_view(plan, evidence, workflow, output, graph_sink),
            record_hash=task.record_hash,
            fingerprint=fingerprint,
            verified_at=self.clock(),
            files=files,
            implementation=installed,
        )
        with self._ledger_lock:
            self._ledger.pop(key, None)
            self._ledger[key] = study
            while len(self._ledger) > _LEDGER_ENTRIES:
                self._ledger.pop(next(iter(self._ledger)))
        if plan.alpha_source is not None:
            self._keep_on_disk(key, study)
        return study, "FULL"

    def _verified_view(
        self,
        plan: ExperimentPlan,
        evidence: ResearchExecutionEvidence,
        workflow: ResearchProgramWorkflow,
        output: Path,
        graph_sink: list[AlphaDevelopmentVerifiedGraph] | None,
    ) -> object:
        """What a read shows of the kind's verified evidence, taken from its Desk verifier."""

        if plan.program.kind == RISK_EXPERIMENT_KIND:
            risk_verifier = workflow.verifier(RISK_EXPERIMENT_KIND)
            if not isinstance(risk_verifier, RiskEvidenceVerifier):
                raise AuthoringError("risk_research.verified_readback_unavailable")
            return risk_verifier.projection(evidence=evidence, output_workspace=output)
        if plan.portfolio_source is not None:
            verifier = workflow.verifier(PORTFOLIO_EXPERIMENT_KIND)
            if (
                not isinstance(verifier, PortfolioExperimentVerifier)
                or verifier.verified_readback is None
            ):
                raise AuthoringError("portfolio_research.verified_readback_unavailable")
            return verifier.verified_readback
        if plan.model_training_source is not None or plan.qualification_family is not None:
            return None
        if plan.alpha_source is not None:
            return self._alpha_projection(workflow, evidence, output, graph_sink)
        verified_factor = getattr(
            workflow.verifier(FACTOR_EXPERIMENT_KIND), "verified_readback", None
        )
        if verified_factor is None:
            raise AuthoringError("factor_research.verified_readback_unavailable")
        return verified_factor

    @staticmethod
    def _alpha_projection(
        workflow: ResearchProgramWorkflow,
        evidence: ResearchExecutionEvidence,
        output: Path,
        graph_sink: list[AlphaDevelopmentVerifiedGraph] | None = None,
    ) -> dict[str, object]:
        """The verified report facts of the Alpha graph this readback walked.

        ``_verified_evidence`` had the Desk verifier resolve every child of the
        receipt and re-hash every chunk. The verifier keeps that walk's result
        as a value, and the same request projects it; the graph's receipt is
        matched to the evidence's handle, so nothing a different verification
        walked could be reported here. A verifier that walked another shape of
        Alpha evidence keeps no graph, and the reader walks once more. A caller
        that will consume the graph itself in this request (the saved
        comparison) receives it through ``graph_sink``: the one walk's value,
        never a second walk.
        """

        reader = AlphaDevelopmentReceiptReader(output / "alpha-development")
        handle = evidence.artifact_uris[0]
        verifier = workflow.verifier(ALPHA_EXPERIMENT_KIND)
        graph = verifier.verified_graph if isinstance(verifier, AlphaEvidenceVerifier) else None
        if graph is None or graph.receipt.receipt_hash != alpha_development_receipt_handle(handle):
            graph = reader.read(handle)
        if graph_sink is not None:
            graph_sink.append(graph)
        return reader.projection_of(graph)

    def awaiting(self, tasks: Iterable[TaskRecord]) -> dict[str, list[str]]:
        """Read metadata for retained studies awaiting curation or promotion.

        Saved studies that wait on a person (N5): Factor studies nobody has curated and
        explored studies nobody has promoted. Read from the plans and the curation folders
        alone, among the request's one reading of the Tasks (V119): nothing is verified.
        """
        succeeded: list[tuple[TaskRecord, ExperimentPlan]] = []
        promoted: set[str] = set()
        admitted: list[str] = []
        for task in tasks:
            if task.task_kind != TASK_KIND:
                continue
            plan, _actor = self._of(task)
            admitted.append(plan.plan_hash)
            if plan.origin_task_id is not None and research_lane(plan.document) == "PROMOTION":
                promoted.add(str(plan.origin_task_id))
            if task.lifecycle is TaskLifecycle.SUCCEEDED:
                succeeded.append((task, plan))
        curation, promotion = [], []
        for task, plan in succeeded:
            if plan.program.kind == FACTOR_EXPERIMENT_KIND:
                folder = (
                    confined(
                        self.session.workspace, plan.document["experiment"]["output_workspace"]
                    )
                    / FACTOR_DEVELOPMENT_CURATION_CATEGORY
                )
                if not any(folder.rglob("*.json")):
                    curation.append(str(task.task_id))
            if research_lane(plan.document) == "EXPLORATION" and str(task.task_id) not in promoted:
                promotion.append(str(task.task_id))
        return {"curation": curation, "promotion": promotion, "admitted": admitted}

    def waiting_previews(self, admitted: Iterable[str]) -> list[dict[str, object]]:
        """Each PLAN previewed and not run while it is still runnable (V45).

        Args:
            admitted: The plans the one read of the Tasks found admitted (`awaiting`).

        Returns:
            Each waiting preview's plan hash, study kind, who previewed it and its times.
        """
        taken = set(admitted)
        return [
            {
                "plan_hash": entry.plan.plan_hash,
                "study_kind": entry.plan.program.kind,
                "previewed_by": entry.caller,
                "previewed_at": entry.previewed_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
            }
            for entry in self._previews.waiting()
            if entry.plan.plan_hash not in taken
        ]

    def listing(self) -> dict[str, object]:
        """List retained experiment task metadata and exact declared study selections.

        Returns:
            Task lifecycle, input/program/source identities, research lane and declared
            factors/policy; descendant verification is separate.
        """
        rows = []
        batch = self.session.task_control_registry.record_collection()
        refusals = [task_record_refusal(task_id) for task_id in batch.refused_task_ids]
        for task in batch.records:
            if task.task_kind != TASK_KIND:
                continue
            try:
                plan, _actor = self._of(task)
            except ValueError as error:
                refusals.append(
                    {
                        "status": "REFUSED",
                        "task_id": str(task.task_id),
                        "failure_code": public_failure(error, "research_experiment.refused"),
                        "detail": "The recorded study plan could not be read. Inspect this "
                        "Task's recorded state, or start a new study from Research inputs.",
                        "next_requests": {
                            "task": {
                                "operation": "TASK_RECOVERY",
                                "task_id": str(task.task_id),
                            },
                            "inputs": {"operation": "RESEARCH_INPUTS"},
                        },
                    }
                )
                continue
            rows.append(
                {
                    "task_id": str(task.task_id),
                    "lifecycle": task.lifecycle.value,
                    "input_id": plan.binding.input_id,
                    "input_binding_hash": plan.binding.binding_hash,
                    "kind": plan.program.kind,
                    **(
                        {
                            "model_adapter_id": plan.execution_preview.get("model_adapter_id"),
                            "component_recipe_id": plan.document["alpha"].get(
                                "component_recipe_id"
                            ),
                        }
                        if plan.program.kind == ALPHA_EXPERIMENT_KIND
                        else {}
                    ),
                    **(
                        {
                            "risk_capability_handle": plan.document["risk"]["estimator"][
                                "capability"
                            ],
                            "risk_parameters": plan.document["risk"]["estimator"]["parameters"],
                        }
                        if plan.program.kind == RISK_EXPERIMENT_KIND
                        else {}
                    ),
                    **(
                        {
                            "target_recipe_id": plan.document["alpha"]["target_recipe_id"],
                            "model_parameters": plan.document["alpha"]["model_parameters"],
                        }
                        if plan.alpha_source
                        else {}
                    ),
                    **({"origin_task_id": str(plan.origin_task_id)} if plan.origin_task_id else {}),
                    "program_hash": plan.program.program_hash,
                    "research_lane": research_lane(plan.document),
                    "sessions": plan.document["experiment"]["sessions"],
                    "factor_ids": (
                        plan.document["factor"]["factor_ids"]
                        if plan.program.kind == FACTOR_EXPERIMENT_KIND
                        else plan.document["alpha"]["ordered_feature_ids"]
                        if plan.alpha_source
                        else plan.execution_preview.get("ordered_feature_ids", [])
                        if plan.model_training_source
                        else []
                    ),
                    **(
                        {
                            "candidate_id": plan.portfolio_source.candidate_id,
                            # the declared policy the workbench names a book by (law 134)
                            "portfolio_policy": {
                                key: plan.document["portfolio"].get(key)
                                for key in ("top_k", "tranches", "weight_rule")
                            },
                        }
                        if plan.portfolio_source
                        else {}
                    ),
                }
            )
        return {
            "status": "AVAILABLE",
            "experiments": rows,
            **({"refusals": refusals} if refusals else {}),
        }

    def risk_report_selection(self, task_id: UUID) -> dict[str, object]:
        """Read a selected report subject's kind without opening its numerical result."""
        task = self.session.task_control_registry.task(task_id)
        completed_portfolio = False
        if task.task_kind == TASK_KIND and task.lifecycle is TaskLifecycle.SUCCEEDED:
            plan, _actor = self._of(task)
            completed_portfolio = plan.portfolio_source is not None
        return {
            "task_id": str(task_id),
            **RiskReportLinks.selection(
                task_kind=task.task_kind, completed_portfolio=completed_portfolio
            ),
        }

    def readback(
        self,
        task_id: UUID,
        portfolio_session: str | None = None,
        *,
        current_policy: bool = False,
        include_alpha_projection: bool = True,
    ) -> dict[str, object]:
        """Read one exact retained experiment through its deterministic evidence verifier.

        Args:
            task_id: Exact experiment task identity.
            portfolio_session: Optional declared Portfolio session.
            current_policy: Whether to require installed policy during readback.
            include_alpha_projection: Whether to include the verified Alpha projection.

        Returns:
            Task state or verified study publication and report projection, with the study's
            `standing` (V368).
        """
        body = self._readback(
            task_id,
            portfolio_session,
            current_policy=current_policy,
            include_alpha_projection=include_alpha_projection,
        )
        # What the study can claim, one standing from its marks (V368).
        return {**body, "standing": study_standing(body).model_dump(mode="json")}

    def _verified_alpha_side(
        self, task_id: UUID
    ) -> tuple[dict[str, object], AlphaDevelopmentVerifiedGraph | None]:
        """One side of a saved comparison: the Task's readback body without the
        rendered projection, and the graph that readback verified (None when
        the Task published no Alpha graph, which the comparison reports)."""

        sink: list[AlphaDevelopmentVerifiedGraph] = []
        body = self._readback(
            task_id, None, current_policy=False, include_alpha_projection=False, graph_sink=sink
        )
        return body, sink[0] if sink else None

    def summary(self, task_id: UUID) -> dict[str, object]:
        """Read recorded Alpha facts without claiming a fresh bulk-evidence proof."""
        task = self.session.task_control_registry.task(task_id)
        plan, _actor = self._of(task)
        if plan.program.kind != ALPHA_EXPERIMENT_KIND:
            raise AuthoringError("research_experiment.summary_kind_not_installed")
        if task.lifecycle is not TaskLifecycle.SUCCEEDED:
            return {
                "status": task.lifecycle.value,
                "task_id": str(task_id),
                "failure_code": task.failure_code,
                # The block's own words when its code has them, else the lifecycle's.
                **(
                    explain(task.failure_code or "", task_id=str(task_id))
                    or explain(
                        "task_not_succeeded", task_id=str(task_id), lifecycle=task.lifecycle.value
                    )
                ),
            }
        evidence = self._stored_evidence(plan)
        final = next(
            v
            for v in self.session.task_control_registry.work_items(task_id)
            if v.stage_id == STAGES[-1]
        )
        if (
            not final.evidence
            or final.evidence[0].content_hash != evidence.evidence_hash
            or evidence.program_hash != plan.program.program_hash
            or evidence.kind != plan.program.kind
            or evidence.desk_program_hash != plan.program.desk_program_hash
            or evidence.method_binding_hash != plan.program.method_binding_hash
            or evidence.authority_hash != plan.authority.authority_hash
        ):
            raise AuthoringError("research_experiment.publication_evidence_mismatch")
        output = confined(self.session.workspace, plan.document["experiment"]["output_workspace"])
        body: dict[str, Any] = {
            "status": "EXPERIMENT_SUMMARY",
            "task_id": str(task_id),
            "evidence_verification": "METADATA_ONLY_BULK_EVIDENCE_NOT_CHECKED",
            "program": plan.program.model_dump(mode="json", exclude={"resolved_sessions"}),
            "document": plan.document,
            "execution_preview": plan.execution_preview,
            "research_input_id": plan.binding.input_id,
            "input_binding_hash": plan.binding.binding_hash,
            **recorded_work(evidence, current_numerical_calls=0),
            "next_requests": {
                "verify": {"operation": "EXPERIMENT_READBACK", "task_id": str(task_id)}
            },
            "limitations": [
                "SAVED_SUMMARY_NOT_FULL_GRAPH_VERIFICATION",
                "NO_CURRENT_STRATEGY_ACTIVATION",
            ],
        }
        if plan.qualification_family is not None:
            body["alpha_qualification"] = verify_qualification(
                program=plan.program,
                evidence=evidence,
                output_workspace=output,
                artifact_root=self.session.workspace / "artifacts",
            ).model_dump(mode="json")
            body["qualification_family"] = plan.qualification_family.model_dump(mode="json")
        elif plan.model_training_source is not None:
            receipt = read_lifecycle_research_receipt(
                program=plan.program, evidence=evidence, output_workspace=output
            )
            body["lifecycle_research"] = {
                **receipt.model_dump(
                    mode="json",
                    exclude={"model_sets", "projections", "projection_files", "formation_sessions"},
                ),
                "formation_count": len(receipt.formation_sessions),
            }
            body["model_training_source"] = plan.model_training_source.model_dump(mode="json")
        else:
            body.update(
                AlphaDevelopmentReceiptReader(output / "alpha-development").summary(evidence)
            )
            if plan.alpha_source is not None:
                body["alpha_source"] = plan.alpha_source.model_dump(mode="json")
        return body

    def _readback(
        self,
        task_id: UUID,
        portfolio_session: str | None,
        *,
        current_policy: bool,
        include_alpha_projection: bool,
        graph_sink: list[AlphaDevelopmentVerifiedGraph] | None = None,
    ) -> dict[str, object]:
        task = self.session.task_control_registry.task(task_id)
        plan, actor = self._of(task)
        if portfolio_session is not None and plan.portfolio_source is None:
            raise AuthoringError("portfolio_research.date_selector_not_portfolio")
        if task.lifecycle is not TaskLifecycle.SUCCEEDED:
            return {
                "status": task.lifecycle.value,
                "task_id": str(task_id),
                "failure_code": task.failure_code,
                # The block's own words when its code has them, else the lifecycle's.
                **(
                    explain(task.failure_code or "", task_id=str(task_id))
                    or explain(
                        "task_not_succeeded", task_id=str(task_id), lifecycle=task.lifecycle.value
                    )
                ),
            }
        output = confined(self.session.workspace, plan.document["experiment"]["output_workspace"])
        study, basis = self._verified_study(
            task,
            plan,
            actor,
            recorded_readback=not current_policy,
            output=output,
            graph_sink=graph_sink,
        )
        evidence, original = study.evidence, study.original
        items = self.session.task_control_registry.work_items(task_id)
        final = next(v for v in items if v.stage_id == STAGES[-1])
        # Replay changes disposition/hash; the receipt's immutable Program is the
        # binding. Stage verification below compares the originally filed record.
        if not final.evidence or original.evidence_hash != final.evidence[0].content_hash:
            raise AuthoringError("research_experiment.publication_evidence_mismatch")
        shared = {
            **published_result_header(
                task_id=task_id, plan=plan, actor=actor, readback=evidence, original=original
            ),
            **study.currency,
            "verification_basis": basis,
            **sample_standing(plan.document, plan.authority),
            "realization": realization(
                CpuBudgetStore(self.session.workspace / "runtime"),
                task_id,
                plan.program.program_hash,
            ),
        }
        next_requests = {
            "continue": {"operation": "EXPERIMENT_DRAFT", "task_id": str(task_id)},
            "export": {
                "operation": "EXPERIMENT_EXPORT",
                "task_id": str(task_id),
                **({"portfolio_session": portfolio_session} if portfolio_session else {}),
            },
        }
        if plan.portfolio_source is None:
            shared["timing"] = self._timing(plan, tuple(map(str, evidence.formation_sessions)))
        if plan.program.kind == RISK_EXPERIMENT_KIND:
            return {
                **shared,
                "next_requests": next_requests,
                "report_reference_selection": {
                    "admitted_subject": "COMPLETED_PORTFOLIO_STUDY",
                    "installed_book": RiskReportLinks.selection(
                        task_kind="INSTALLED_STRATEGY_BOOK", completed_portfolio=False
                    ),
                    "next_requests": {"studies": {"operation": "EXPERIMENTS"}},
                },
                "execution_preview": plan.execution_preview,
                "input_binding_hash": plan.binding.binding_hash,
                **cast(dict[str, object], study.view),
            }
        if plan.portfolio_source is not None:
            receipt, rows = cast(tuple[Any, list[dict[str, Any]]], study.view)
            if not evidence.artifact_uris[0].endswith("/" + receipt.receipt_hash):
                raise AuthoringError("portfolio_research.verified_readback_receipt_mismatch")
            data_quality = receipt.data_quality
            position = (
                rows[-1]
                if portfolio_session is None
                else next((r for r in rows if r["session"] == portfolio_session), None)
            )
            if position is None:
                raise AuthoringError("portfolio_research.session_outside_report")
            selected_index = next(i for i, row in enumerate(rows) if row is position)
            shared["timing"] = self._timing(
                plan, tuple(map(str, evidence.formation_sessions)), position["session"]
            )
            review_selector = {
                "experiment_task_id": str(task_id),
                "experiment_receipt_hash": receipt.receipt_hash,
                "portfolio_session": position["session"],
            }
            return {
                **shared,
                "next_requests": {
                    **next_requests,
                    "export": {
                        "operation": "EXPERIMENT_EXPORT",
                        "task_id": str(task_id),
                        "portfolio_session": position["session"],
                    },
                    "risk-links": {"operation": "EXPERIMENT_RISK_LINKS", "task_id": str(task_id)},
                    **review_requests(review_selector),
                    "delivery": {
                        "operation": "EXPERIMENT_DELIVERY_EXPORT",
                        "task_id": str(task_id),
                        "experiment_receipt_hash": receipt.receipt_hash,
                        "portfolio_session": position["session"],
                    },
                },
                "receipt": receipt.model_dump(mode="json"),
                "result": receipt.result,
                "listing_labels": receipt.listing_labels or receipt.source.ordered_listing_ids,
                "portfolio_source": plan.portfolio_source.model_dump(mode="json"),
                "data_quality": data_quality,
                "execution_preview": plan.execution_preview,
                "position": position,
                # Each metric's unit by its path, which a scoped read names (V343).
                "metric_units": {f"position.{k}": unit for k, unit in POSITION_UNITS.items()},
                "preceding_position": rows[selected_index - 1] if selected_index else None,
                "review_selector": review_selector,
                "series": [
                    {k: v for k, v in r.items() if k not in {"targets", "weights"}} for r in rows
                ],
                "limitations": [
                    "POST_OBSERVED_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION",
                    "CURRENT_UNIVERSE_NON_PIT",
                    "PREDICTED_RISK_AND_DECOMPOSITION_NOT_ADMITTED",
                    "NO_CAPACITY_OR_IMPACT_ESTIMATE",
                    "NO_STRATEGY_ACTIVATION",
                    *([data_quality["notice"]] if data_quality["notice"] else []),
                ],
            }
        if plan.qualification_family is not None:
            qualification = verify_qualification(
                program=plan.program,
                evidence=evidence,
                output_workspace=output,
                artifact_root=self.session.workspace / "artifacts",
            )
            return {
                **shared,
                "next_requests": next_requests,
                "alpha_qualification": qualification.model_dump(mode="json"),
                "qualification_family": plan.qualification_family.model_dump(mode="json"),
                "execution_preview": plan.execution_preview,
                "limitations": [
                    "QUALIFIED_OVER_THE_WHOLE_ATTEMPTED_FAMILY",
                    "SEALED_HOLDOUT_UNREAD",
                    "NO_CURRENT_STRATEGY_ACTIVATION",
                ],
            }
        if plan.model_training_source is not None:
            receipt = read_lifecycle_research_receipt(
                program=plan.program, evidence=evidence, output_workspace=output
            )
            return {
                **shared,
                "next_requests": next_requests,
                "model_training_source": plan.model_training_source.model_dump(mode="json"),
                "lifecycle_research": receipt.model_dump(mode="json"),
                "execution_preview": plan.execution_preview,
                "limitations": [
                    "LOCAL_MODEL_LIFECYCLE_RESEARCH_NOT_ORIGINAL_RESEARCH_RESULTS",
                    "CURRENT_UNIVERSE_NON_PIT",
                    "NO_CURRENT_STRATEGY_ACTIVATION",
                ],
            }
        if plan.alpha_source is not None:
            projection = cast(dict[str, object], study.view)
            if not include_alpha_projection:
                # Verified as always -- the Desk verifier walked the graph -- and
                # not rendered: the comparison consumes the graph itself.
                return {
                    **shared,
                    "receipt": {
                        "receipt_hash": alpha_development_receipt_handle(evidence.artifact_uris[0])
                    },
                    "alpha_source": plan.alpha_source.model_dump(mode="json"),
                }
            return {
                **shared,
                "next_requests": {
                    **next_requests,
                    # The candidate is the reader's choice, named as one (V136).
                    "portfolio-draft": {
                        "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
                        "task_id": str(task_id),
                        "candidate_id": None,
                    },
                },
                **projection,
                "alpha_source": plan.alpha_source.model_dump(mode="json"),
                "execution_preview": plan.execution_preview,
                "limitations": [
                    "DEVELOPMENT_EVIDENCE_ONLY",
                    "NO_FOUNDATION_OR_STRATEGY_ACTIVATION",
                    "NO_INDEPENDENT_SCIENTIFIC_VALIDATION",
                ],
            }
        receipt, child = cast(tuple[Any, Any], study.view)
        if not evidence.artifact_uris[0].endswith("/" + receipt.receipt_hash):
            raise AuthoringError("factor_research.verified_readback_receipt_mismatch")
        return {
            **shared,
            # The declaration's selection before the context it was screened in, which is
            # its multiple-testing denominator, not the choice (V252, SC3).
            "declared_selection": {
                "selected_factor_ids": list(receipt.selected_factor_ids),
                "screened_factor_count": len(child.program.factor_ids),
            },
            "next_requests": {
                **next_requests,
                "curation": {"operation": "EXPERIMENT_CURATION", "task_id": str(task_id)},
            },
            "receipt": receipt.model_dump(mode="json"),
            "result": child.model_dump(mode="json"),
            "limitations": [
                "DEVELOPMENT_EVIDENCE_ONLY",
                "Selected factors do not reduce the multiple-testing context.",
                *(["RECORDED_CURATION_POLICY_NOT_CURRENT_ADMISSION"] if not current_policy else []),
            ],
        }

    def _stored_evidence(self, plan: ExperimentPlan) -> ResearchExecutionEvidence:
        from alphalattice.control.research_program.authoring.workflow import ResearchEvidenceStore

        output = confined(self.session.workspace, plan.document["experiment"]["output_workspace"])
        stored = ResearchEvidenceStore(output).load_for_program(plan.program.program_hash)
        if stored is None:
            raise AuthoringError("research_experiment.evidence_absent")
        return stored

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """One execution attempt: the runner asks `compatibility`, then runs the stages.

        `compatibility` verifies the plan against the workspace before the
        execution is recorded; the first stage consumes that verification
        (the same workflow object) instead of repeating it seconds later, and
        anything not consumed -- a recovery whose first stage is already
        verified, an admission the registry refused -- is dropped here.
        """
        task = self.session.task_control_registry.task(task_id)
        try:
            self.session.execute_admitted(task, self, self.clock, expected_task_hash)
        finally:
            self._admitted_workflows.pop(task_id, None)
            self._attempt_evidence.pop(task_id, None)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Verify and retain the exact workflow before binding task execution compatibility.

        Args:
            task: Exact admitted experiment task.

        Returns:
            Schema, workflow, input, method and installed execution identity binding.
        """
        plan, _actor = self._of(task)
        self._admitted_workflows[task.task_id] = self._current(plan, task_id=task.task_id)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(schema_structure(ExperimentPlan)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=plan.program.method_binding_hash,
            framework_identity_hash=self.session.execution_identity(plan.implementation_hash),
        )

    @staticmethod
    def _stage_evidence(evidence: ResearchExecutionEvidence) -> tuple[TaskEvidence, ...]:
        return (
            TaskEvidence(
                evidence_kind="research_experiment.evidence",
                reference=f"playpen://research-evidence/{evidence.evidence_hash}",
                content_hash=evidence.evidence_hash,
            ),
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Execute one exact experiment stage and retain deterministic cancellation/refusal state.

        Args:
            task: Exact admitted experiment task.
            execution: Current execution declaration.
            work_item: Exact workflow stage.

        Returns:
            Stage result; typed numerical cancellation becomes CANCELLED and ValueError becomes
            BLOCKED with its public failure code.
        """
        try:
            return self._execute_stage(task=task, work_item=work_item)
        except (AlphaDevelopmentCancelled, PortfolioExperimentCancelled, RiskDevelopmentCancelled):
            return StageExecutionResult(StageDisposition.CANCELLED)
        except ValueError as error:
            return StageExecutionResult(
                StageDisposition.BLOCKED, failure_code=failure_code_from(error)
            )

    def _execute_stage(
        self, *, task: TaskRecord, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        plan, actor = self._of(task)
        if work_item.stage_id == STAGES[0]:
            workflow = self._admitted_workflows.pop(task.task_id, None)
            if workflow is None:
                workflow = self._current(plan, task_id=task.task_id)
            sealed = SealedSubmission(
                document=plan.document,
                envelope=ResearchExperimentEnvelope.create(**plan.document["experiment"]),
                authority=plan.authority,
                program=plan.program,
                binding=actor,
            )
            if plan.program.kind == ALPHA_EXPERIMENT_KIND:
                with self._fit_threads(plan.program.program_hash):
                    workflow.execute_prepared(sealed)
            else:
                workflow.execute_prepared(sealed)
            read_factor_bundle(self.session.workspace, plan.binding.binding_hash)
        elif work_item.stage_id != STAGES[1]:
            raise AuthoringError("research_experiment.stage_unknown")
        self._attempt_verified(task)
        return StageExecutionResult(
            StageDisposition.READY, evidence=self._stage_evidence(self._stored_evidence(plan))
        )

    @contextmanager
    def _fit_threads(self, program_hash: str) -> Iterator[LightGBMThreads]:
        """The workspace's CPU budget, as the threads of one Alpha run's LightGBM fits.

        The fits run one at a time, so the run takes the budget's cores; LightGBM proves
        its sealed canary at that count first and refuses by name when it differs. What
        the fits used is appended to the workspace's execution log, beside the result and
        never in it (binding plan, B3), with the thread counts of the numerical libraries the
        fits and the fold metrics run at, which can move a metric's last bits (PA3, V68).
        """

        store = CpuBudgetStore(self.session.workspace / "runtime")
        machine = machine_load()
        budget = store.read()
        cores, reason = budget_cores(budget, machine)
        with lightgbm_threads(cores) as used:
            store.record(
                ModelFitExecution(
                    program_hash=program_hash,
                    cpu_budget=budget.cpu_budget,
                    cores=cores,
                    lightgbm_threads=used.threads,
                    reason=f"{reason}; {used.reason}",
                    machine=machine,
                    planned_at=self.clock(),
                    numerical_threads=numerical_thread_counts(),
                )
            )
            yield used

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Require the installed stage and its exact stored evidence tuple.

        Args:
            task: Exact retained experiment task.
            execution: Current execution declaration.
            work_item: Exact workflow stage.
            evidence: Evidence tuple supplied by Task Control.

        Returns:
            Unchanged verified evidence.

        Raises:
            AuthoringError: Stage is unknown or stored evidence does not match.
        """
        del execution
        if work_item.stage_id not in STAGES:
            raise AuthoringError("research_experiment.stage_unknown")
        plan, _actor = self._of(task)
        self._attempt_verified(task)
        if evidence != self._stage_evidence(self._stored_evidence(plan)):
            raise AuthoringError("research_experiment.stage_evidence_mismatch")
        return evidence

    def verified_evidence_hash(self, task_id: UUID) -> str | None:
        """The evidence a Task's last stage verified and sealed, by its VERIFIED receipt.

        Args:
            task_id: A research experiment Task.

        Returns:
            Its evidence's hash, or ``None`` before that stage's receipt is sealed.
        """
        for receipt in self.session.task_control_registry.stage_receipts(task_id):
            if receipt.stage_id == STAGES[1] and receipt.status == "VERIFIED":
                for item in receipt.evidence:
                    if item.evidence_kind == "research_experiment.evidence":
                        return str(item.content_hash)
        return None

    def _attempt_verified(self, task: TaskRecord) -> ResearchExecutionEvidence:
        """A running Task's evidence, read back by its verifier in the worker, once per attempt.

        Sealed files do not change within an attempt, so the first verification answers the
        rest (the first stage's check, the second stage and its check).
        """

        held = self._attempt_evidence.get(task.task_id)
        if held is not None:
            return held
        evidence = self._verify_in_worker(task)
        self._attempt_evidence[task.task_id] = evidence
        return evidence

    def _verify_in_worker(self, task: TaskRecord) -> ResearchExecutionEvidence:
        """A study's sealed evidence read back by its verifier in the Host's worker (W10).

        The readback re-hashes every sealed chunk the study wrote, seconds of Python that held
        the Host's GIL while its reads waited.
        """
        plan, actor = self._of(task)
        evidence = cast(
            ResearchExecutionEvidence,
            run_in_child(
                f"{__name__}:verify_task_evidence",
                {
                    "workspace": self.session.workspace,
                    "binding_hash": plan.binding.binding_hash,
                    "kind": plan.program.kind,
                    "document": plan.document,
                    "program_hash": plan.program.program_hash,
                    "actor": actor,
                    "authority": plan.authority,
                },
            ),
        )
        _check_evidence_window(plan, evidence)
        return evidence

    def verify_saved_study(self, task: TaskRecord) -> None:
        """Verify a saved study's sealed evidence in full, in the worker (the sweep, V89).

        What reads keep of the study stands where its files are still the ones just verified:
        the kept check's time moves to now. A study that does not verify raises.
        """
        plan, _actor = self._of(task)
        fingerprint, _files = self._study_fingerprint(plan)
        self._verify_in_worker(task)
        now = self.clock()
        for key in ((task.task_id, True), (task.task_id, False)):
            with self._ledger_lock:
                kept = self._ledger.get(key)
                if kept is not None and kept.fingerprint == fingerprint:
                    self._ledger[key] = replace(kept, verified_at=now)
            if plan.alpha_source is not None:
                on_disk = self._kept_on_disk(key)
                if on_disk is not None and on_disk.fingerprint == fingerprint:
                    self._keep_on_disk(key, replace(on_disk, verified_at=now))

    def forget_verification(self, task_id: UUID) -> None:
        """Drop what reads keep of a study, so its next read verifies it in full."""
        for key in ((task_id, True), (task_id, False)):
            with self._ledger_lock:
                self._ledger.pop(key, None)
            self._ledger_path(key).unlink(missing_ok=True)

    def operate(
        self,
        request: PortfolioResearchOperationRequest,
        *,
        caller: str,
        agent_execution: AgentExecutionBinding | None = None,
    ) -> dict[str, object]:
        """Route an explicit experiment operation to its deterministic owner.

        Saved comparisons verify each selected side. Curation publishes immutable metadata under its
        lock, grants no numerical authority and requires the agent binding when the caller is an
        installed agent.

        Args:
            request: Validated experiment operation declaration.
            caller: Declared operation caller.
            agent_execution: Optional exact installed-agent execution binding.

        Returns:
            Controls, draft/preview, bounded execution, verified readback/comparison/export or exact
            curation/handoff result.
        """
        op = request.operation
        if op == "EXPERIMENT_ALPHA_COMPARE":
            from alphalattice.investment.alpha_research.experiments.comparison import (
                compare_saved_alpha_candidates,
            )

            assert (
                request.left_task_id is not None
                and request.right_task_id is not None
                and request.left_candidate_id is not None
                and request.right_candidate_id is not None
            )
            if request.left_task_id == request.right_task_id:
                raise AuthoringError("alpha_research.saved_comparison_requires_two_tasks")
            left_plan, _left_actor = self._of(
                self.session.task_control_registry.task(request.left_task_id)
            )
            right_plan, _right_actor = self._of(
                self.session.task_control_registry.task(request.right_task_id)
            )
            if left_plan.alpha_source is None or right_plan.alpha_source is None:
                raise AuthoringError("alpha_research.saved_comparison_alpha_tasks_required")
            # One comparison read: each side is verified once by its readback and
            # the comparison consumes that graph -- its receipt, children and the
            # score rows the walk proved -- walking nothing again.
            left, left_graph = self._verified_alpha_side(request.left_task_id)
            right, right_graph = self._verified_alpha_side(request.right_task_id)
            return compare_saved_alpha_candidates(
                left=left,
                left_candidate_id=request.left_candidate_id,
                left_graph=left_graph,
                right=right,
                right_candidate_id=request.right_candidate_id,
                right_graph=right_graph,
            )
        if op == "EXPERIMENT_COMPARE":
            from alphalattice.control.product_host.research_authoring.comparison import (
                compare_experiment_reports,
            )

            assert request.left_task_id is not None and request.right_task_id is not None
            if request.left_task_id == request.right_task_id:
                raise AuthoringError("portfolio_research.comparison_requires_two_results")
            left = self.readback(request.left_task_id, request.portfolio_session)
            right = self.readback(request.right_task_id, request.portfolio_session)
            return compare_experiment_reports(left, right)
        if op in {"EXPERIMENT_LINK_RISK", "EXPERIMENT_RISK_LINKS", "EXPERIMENT_RISK_EXPORT"}:
            assert request.task_id is not None
            selection = self.risk_report_selection(request.task_id)
            if not selection["available"]:
                return {"status": "REFUSED", **selection}
            links = RiskReportLinks(self.session.workspace, self.readback)
            if op == "EXPERIMENT_LINK_RISK":
                assert request.risk_task_id is not None
                return links.attach(
                    request.task_id,
                    request.risk_task_id,
                    caller=caller,
                    portfolio_window=request.risk_report_scope == "POST_OBSERVED_PORTFOLIO_WINDOW",
                )
            if op == "EXPERIMENT_RISK_LINKS":
                return links.listing(request.task_id)
            assert request.risk_report_hash is not None
            return links.export(request.task_id, request.risk_report_hash)
        if op.startswith("EXPERIMENT_FOUNDATION"):
            return self._operate_foundation(request, caller=caller, agent_execution=agent_execution)
        if op == "EXPERIMENT_PORTFOLIO_DRAFT":
            assert request.task_id is not None and request.candidate_id is not None
            prepared_portfolio, binding = self._prepare_portfolio(
                request.task_id, request.candidate_id
            )
            return {
                **portfolio_draft(prepared_portfolio, self._risk_studies(binding.binding_hash)),
                # What the catalog holds and a study may not declare yet, and why (V313).
                "not_available": list(NOT_AVAILABLE),
                "research_input_id": binding.input_id,
                "origin_task_id": str(request.task_id),
                "plan_request": {
                    "operation": "EXPERIMENT_PLAN",
                    "research_input_id": binding.input_id,
                    "input_binding_hash": binding.binding_hash,
                },
            }
        if op == "EXPERIMENT_CONTROLS":
            return self.controls(
                request.research_input_id,
                request.input_binding_hash,
                request.experiment_kind or FACTOR_EXPERIMENT_KIND,
                request.component_id,
                request.feature_preparation_hash,
            )
        if op == "EXPERIMENT_DRAFT":
            assert request.task_id is not None
            return self.draft(
                request.task_id, request.research_input_id, request.input_binding_hash
            )
        if op == "EXPERIMENT_PROMOTE":
            assert request.task_id is not None
            return self.promote(request.task_id, caller=caller, agent_execution=agent_execution)
        if op == "EXPERIMENT_CONTINUE":
            assert request.task_id is not None
            return self.continue_study(
                request.task_id, caller=caller, agent_execution=agent_execution
            )
        if op == "EXPERIMENT_PLAN":
            return self.plan(
                request.research_input_id,
                request.experiment_document,
                request.experiment_yaml,
                request.input_binding_hash,
                request.origin_task_id,
                request.factor_task_id,
                request.curation_receipt_hash,
                caller=caller,
            )
        if op == "EXPERIMENT_PREVIEW_READBACK":
            assert request.experiment_plan_hash is not None
            return self.preview_readback(request.experiment_plan_hash)
        if op == "EXPERIMENT_RUN":
            assert request.experiment_plan_hash is not None
            return self.run(
                request.experiment_plan_hash, caller=caller, agent_execution=agent_execution
            )
        if op == "EXPERIMENTS":
            return self.listing()
        assert request.task_id is not None
        if op == "EXPERIMENT_SUMMARY":
            return self.summary(request.task_id)
        if op == "EXPERIMENT_REPLAY":
            task = self.session.task_control_registry.task(request.task_id)
            self._current(self._of(task)[0], replay=True)
        body = self.readback(
            request.task_id,
            request.portfolio_session,
            current_policy=op == "EXPERIMENT_REPLAY",
        )
        if op in {"EXPERIMENT_CURATION", "EXPERIMENT_CURATE", "EXPERIMENT_HANDOFF_PREVIEW"}:
            if body["status"] != "EXPERIMENT_PUBLISHED":
                raise AuthoringError("factor_research.completed_experiment_required")
            task = self.session.task_control_registry.task(request.task_id)
            plan, _actor = self._of(task)
            if plan.program.kind != FACTOR_EXPERIMENT_KIND:
                raise AuthoringError("research_experiment.factor_parent_required")
            root = confined(self.session.workspace, plan.document["experiment"]["output_workspace"])
            receipt = FactorDevelopmentReceipt.model_validate_json(json.dumps(body["receipt"]))
            checkpoint = FactorResearchDeterministicEvidence.model_validate_json(
                json.dumps(body["result"])
            )
            if op == "EXPERIMENT_CURATION":
                return {
                    **curation_readback(root, receipt, checkpoint),
                    "inputs": [v.input_id for v in self.manifest.experiment_inputs or ()],
                    "next_requests": {
                        "curate": {
                            "operation": "EXPERIMENT_CURATE",
                            "task_id": str(request.task_id),
                            "experiment_curation": {"expected_receipt_hash": receipt.receipt_hash},
                        }
                    },
                }
            if op == "EXPERIMENT_CURATE":
                if caller == "INSTALLED_AGENT" and agent_execution is None:
                    raise AuthoringError("research_experiment.agent_execution_not_admitted")
                assert request.experiment_curation is not None
                # Synchronous immutable metadata, not a second Task runtime.
                with self._curation_lock:
                    previous = curation_readback(root, receipt, checkpoint)["decisions"]
                    result = submit_report_curation(
                        root=root,
                        receipt=receipt,
                        checkpoint=checkpoint,
                        request=request.experiment_curation,
                        actor_kind=ActorKind(caller),
                        actor_id=f"local-web-{caller.lower()}",
                        agent_execution=agent_execution,
                    )
                return {
                    "status": "REUSED_EXACT"
                    if result.decision.model_dump(mode="json") in previous
                    else "CURATION_PUBLISHED",
                    "decision": result.decision.model_dump(mode="json"),
                    "next_requests": {
                        # The Alpha handoff on this decision, filled, so the receipt travels
                        # by --from and is never copied (V391).
                        "handoff": {
                            "operation": "EXPERIMENT_HANDOFF_PREVIEW",
                            "task_id": str(request.task_id),
                            "curation_receipt_hash": result.decision.receipt_hash,
                            "research_input_id": plan.binding.input_id,
                            "input_binding_hash": plan.binding.binding_hash,
                        },
                        "foundation-preview": {
                            "operation": "EXPERIMENT_FOUNDATION_PREVIEW",
                            "task_id": str(request.task_id),
                            "curation_receipt_hash": result.decision.receipt_hash,
                            "research_input_id": plan.binding.input_id,
                            "input_binding_hash": plan.binding.binding_hash,
                        },
                    },
                    "task": None,
                    "numerical_call_count": 0,
                }
            assert request.curation_receipt_hash is not None
            selected_binding = None
            if request.input_binding_hash is not None:
                selected_binding = self._binding(
                    request.research_input_id or plan.binding.input_id,
                    request.input_binding_hash,
                )
            elif any(
                v.binding_hash == plan.binding.binding_hash
                for v in ResearchInputRevisions(self.session).lineage(plan.binding.input_id)
            ):
                selected_binding = self._binding(
                    request.research_input_id or plan.binding.input_id,
                    plan.binding.binding_hash
                    if request.research_input_id in {None, plan.binding.input_id}
                    else None,
                )
            preview = preview_factor_handoff(
                workspace=self.session.workspace,
                manifest=self.manifest,
                evidence_root=root,
                original_binding_hash=plan.binding.binding_hash,
                original_document=plan.document,
                receipt=receipt,
                checkpoint=checkpoint,
                curation_receipt_hash=request.curation_receipt_hash,
                input_id=request.research_input_id,
                document=request.experiment_document,
                yaml_text=request.experiment_yaml,
                available_bindings=(selected_binding,) if selected_binding else None,
                feature_input=self._feature_input(
                    plan.document["experiment"]["data_snapshot_handle"], plan.binding.binding_hash
                ),
            )
            # The plan this handoff starts, as a Foundation draft offers its own, so
            # `study plan --from` plans it with its Factor source kept (V361).
            preview["plan_request"] = {
                "operation": "EXPERIMENT_PLAN",
                # The input the handoff chose, beside its binding: the plan never falls back to
                # the Factor study's own input under another input's binding (V438).
                "research_input_id": selected_binding.input_id
                if selected_binding is not None
                else request.research_input_id or plan.binding.input_id,
                "input_binding_hash": preview["input_binding_hash"],
                "factor_task_id": str(request.task_id),
                "curation_receipt_hash": request.curation_receipt_hash,
            }
            preview["json"] = json.dumps(preview, ensure_ascii=False, sort_keys=True, indent=2)
            preview["html"] = render_handoff_report(
                {k: v for k, v in preview.items() if k != "json"}
            )
            return preview
        if op == "EXPERIMENT_REPLAY" and body["status"] == "EXPERIMENT_PUBLISHED":
            return {**body, "status": "REUSED_EXACT", "numerical_call_count": 0}
        if op == "EXPERIMENT_EXPORT":
            import yaml

            document = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2)
            exported = {
                "status": body["status"],
                "json": document,
                "yaml": yaml.safe_dump(body.get("document", {}), sort_keys=False),
                "html": render_experiment_report(body),
            }
            return {**exported, "export_hash": canonical_hash(exported)}
        return body


def draft_revision(
    prior: tuple[str, str], input_id: str | None, binding_hash: str | None
) -> tuple[str, str | None]:
    """The input and revision a draft copies its study onto (V498).

    The origin's exact revision while its input stays; another input named without a revision
    takes that input's declared anchor (none, which its revisions resolve); a revision named
    always wins. The moves a study's sources refuse are refused after, as before.

    Args:
        prior: The origin study's input and revision.
        input_id: The input the draft names, if any.
        binding_hash: The revision the draft names, if any.

    Returns:
        The input, and the revision or none for the input's declared anchor.
    """
    prior_input, prior_hash = prior
    chosen = input_id or prior_input
    if binding_hash is not None:
        return chosen, binding_hash
    return chosen, prior_hash if chosen == prior_input else None


def review_requests(selector: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """The requests that take a book to its Evidence and CRO review, bound to the book.

    Every answer that names a book for its review offers them -- a study book's readback, an
    update's positions, a recorded choice of analysis -- so the agent continues the book it
    read, never the workspace's default (V473, V483, V487).
    """
    return {
        "review": {"operation": "EVIDENCE_CRO", **selector},
        # The book's Evidence begins at its preview (V473).
        "evidence_preview": {"operation": "EVIDENCE_PREVIEW", **selector},
        # Read the current Evidence state before it offers CRO work: a generic
        # book read cannot know whether its bounded continuation is settled (P1).
    }


def _preview_readback_request(plan_hash: str) -> dict[str, object]:
    return {"operation": "EXPERIMENT_PREVIEW_READBACK", "experiment_plan_hash": plan_hash}


@dataclass(frozen=True)
class ResearchExperimentCommand:
    """Dispatch a sealed experiment plan with its exact admitted actor binding."""

    application: ResearchExperimentApplication
    plan: ExperimentPlan | None = None
    actor: ActorSubmissionBinding | None = None
    command_kind: str = TASK_KIND

    def admit(self) -> CommandAdmission:
        """Require the exact preview and actor before deterministic experiment admission.

        Returns:
            Task identity and admitted lifecycle.

        Raises:
            AuthoringError: Plan or actor is absent.
        """
        if self.plan is None or self.actor is None:
            raise AuthoringError("research_experiment.preview_required")
        return self.application.admit(self.plan, self.actor)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Execute one exact admitted experiment task.

        Args:
            task_id: Exact task identity.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.execute(task_id, expected_task_hash=expected_task_hash)


def _check_evidence_window(plan: ExperimentPlan, evidence: ResearchExecutionEvidence) -> None:
    """Refuse evidence whose sessions are not the window the plan previewed."""
    if (
        str(evidence.formation_sessions[0]) != plan.execution_preview["statistical_start"]
        or str(evidence.formation_sessions[-1]) != plan.execution_preview["statistical_end"]
        or len(evidence.formation_sessions) != plan.execution_preview["statistical_session_count"]
        or (
            plan.program.kind == FACTOR_EXPERIMENT_KIND
            and plan.execution_preview["latest_outcome_session"]
            > plan.document["experiment"]["sessions"]["as_of"]["session"]
        )
    ):
        raise AuthoringError("research_experiment.evidence_window_mismatch")


def verify_task_evidence(
    *,
    workspace: Path,
    binding_hash: str,
    kind: str,
    document: dict[str, Any],
    program_hash: str,
    actor: ActorSubmissionBinding,
    authority: ResolvedResearchAuthority | None,
    cancelled: Callable[[], bool],
) -> ResearchExecutionEvidence:
    """The worker's side of a Task's check: its sealed evidence read back through its verifier.

    Args:
        workspace: The research workspace.
        binding_hash: The Task's research input.
        kind: The study's kind, whose Desk verifier reads it back.
        document: The study's document.
        program_hash: Its sealed Program.
        actor: Who submitted it.
        authority: The authority the plan resolved.
        cancelled: Unused: a verification is not cancelled midway.

    Returns:
        The verified evidence.
    """
    del cancelled
    workflow = build_research_program_workflow(
        workspace=factor_input_paths(workspace, binding_hash)[0],
        workspace_root=workspace,
        executors=(),
        verifier_kinds=(kind,),
    )
    return workflow.readback_sealed(
        document,
        program_hash=program_hash,
        actor_kind=actor.actor_kind,
        actor_id=actor.actor_id,
        agent_execution=actor.agent_execution,
        authority=authority,
        recorded_readback=False,
        stored_sink=[],
    )[0]
