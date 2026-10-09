"""Task Control owner for the Portfolio evidence review. Three stages.

1. **admit** the resolved dossier the Host built;
2. **seal** one actor submission against it;
3. **publish** exactly one recommendation for the review key.

The Task never starts or waits on an evidence loop. Evidence arrives as an
already-published, already-verified artifact, or the review does not run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic_core import to_jsonable_python

from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskRecord,
    WorkItemDefinition,
)
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
from alphalattice.control.workspace_runtime.content_store import verified_model_read_scope
from alphalattice.evidence.alternative_evidence.publication.analysis import (
    AlternativeEvidenceAnalysisPublicationService,
)
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
)
from alphalattice.evidence.alternative_evidence.publication.usage import (
    record_provider_stage_usage,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewAnswer,
    PortfolioReviewAssessmentSubmission,
    PortfolioReviewDossier,
    PortfolioReviewPublication,
    PortfolioReviewReceipt,
    compile_portfolio_review_recommendation,
    translate_submission,
)
from alphalattice.oversight.chief_risk_officer.decision.submissions import (
    PORTFOLIO_REVIEW_POLICY_ROLE,
    CROHostPolicyBinding,
    build_portfolio_review_policy_binding,
    seal_portfolio_review,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    PortfolioExperimentReviewSubject,
    PortfolioUpdateReviewSubject,
)
from alphalattice.oversight.chief_risk_officer.publication.portfolio_review import (
    PortfolioReviewPublicationService,
)
from alphalattice.protocols.actor_execution import ActorKind, AgentExecutionBinding
from alphalattice.protocols.actor_execution.answers import (
    ANSWER_CORRECTION_BOUND,
    AnswerProblem,
)
from alphalattice.protocols.actor_execution.usage import ProviderStageUsage

TASK_KIND = "chief_risk_officer.portfolio_review"
INPUT_SCHEMA_ID = "cro-portfolio-review-authority"

# The reviewer's stage budget. The actor takes `min(remaining, profile
# review timeout)`, so this is a ceiling over the admitted profile and not a
# second policy: at ten seconds it overrode every profile and no real review
# could complete -- one measured call over a three-issuer, nineteen-finding
# dossier takes 26.4 s against an endpoint that answers a trivial call in
# 1.15 s. Set to the analyst's budget so the profile is what binds.
ACTOR_STAGE_BUDGET_SECONDS: float = 90.0
"""The whole actor stage, including its corrections."""

_STAGES = (
    ("admit_portfolio_review", (), "cro_review_dossier"),
    ("seal_portfolio_review_assessment", ("admit_portfolio_review",), "cro_review_receipt"),
    (
        "publish_portfolio_review",
        ("seal_portfolio_review_assessment",),
        "cro_review_publication",
    ),
)


@dataclass(frozen=True, slots=True)
class PortfolioReviewActorResult:
    """Record an accepted actor answer and its observed execution evidence.

    One actor's answer -- its accepted part -- plus the execution facts the
    Host records about it: what was dropped, calls and repairs, usage.
    """

    answer: PortfolioReviewAnswer
    actor_kind: ActorKind
    actor_id: str
    dropped: tuple[AnswerProblem, ...] = ()
    agent_execution: AgentExecutionBinding | None = None
    model_call_count: int = 0
    protocol_repair_count: int = 0
    provider_usage: ProviderStageUsage | None = None
    """What the Provider reported for this stage, or nothing when it reported none."""
    carried: PortfolioReviewAssessmentSubmission | None = None
    """The earlier assessment this carries forward, in this dossier's handles (W3)."""

    def __post_init__(self) -> None:
        """Require protocol repair accounting within the installed answer-correction bound.

        Raises:
            ValueError: protocol_repair_count exceeds ANSWER_CORRECTION_BOUND.
        """
        if self.protocol_repair_count > ANSWER_CORRECTION_BOUND:
            raise ValueError("chief_risk_officer.review_repair_budget_exceeded")


class PortfolioReviewActor(Protocol):
    """One semantic execution over one dossier. No tools, no retrieval, no loop."""

    @property
    def process_binding_hash(self) -> str:
        """Identify the exact semantic actor process used by Task compatibility.

        Returns:
            Actor-owned process binding; it carries no Portfolio decision authority.
        """
        ...

    def __call__(
        self, *, dossier: PortfolioReviewDossier, deadline_seconds: float
    ) -> PortfolioReviewActorResult:
        """Execute one tool-free semantic review of an already admitted dossier.

        Args:
            dossier: Exact Host-resolved book and published issuer evidence to interpret.
            deadline_seconds: Remaining admitted actor-stage time budget.

        Returns:
            Accepted answer plus actor identity, usage, call and repair facts for Host sealing.
        """
        ...


class PortfolioReviewEvidenceExpired(RuntimeError):
    """The evidence expired before the actor stage was reached."""


class PortfolioReviewTimeout(TimeoutError):
    """The actor stage exhausted its budget. No recommendation is published."""


class SubmittedPortfolioReviewActor:
    """One already-screened answer, from whichever actor produced it.

    Human, Installed Agent and External Automation all arrive here; the kind
    is recorded and the Installed Agent additionally carries its execution
    binding. A browser, a CLI and a notebook are transports, not actor kinds.
    """

    def __init__(
        self,
        *,
        answer: PortfolioReviewAnswer,
        actor_kind: ActorKind,
        actor_id: str,
        dropped: tuple[AnswerProblem, ...] = (),
        agent_execution: AgentExecutionBinding | None = None,
        model_call_count: int = 0,
        protocol_repair_count: int = 0,
        durable_submission: bool = False,
    ) -> None:
        """Hold an already screened human, installed-agent or external-automation answer.

        Args:
            answer: Typed accepted Portfolio review answer.
            actor_kind: Admitted human, installed-agent or external-automation kind.
            actor_id: Identifier of the actor that supplied the answer.
            dropped: Screened answer problems retained as execution evidence.
            agent_execution: Required exactly for an installed-agent submission.
            model_call_count: Observed model calls reported for this actor execution.
            protocol_repair_count: Observed protocol corrections reported for this execution.
            durable_submission: External durable submission flag; installed agents and nonzero
                call/repair counts are refused.

        Raises:
            ValueError: Actor/execution binding or durable external call accounting is inconsistent.
            pydantic.ValidationError: The supplied answer violates its typed contract.
        """
        if actor_kind not in {
            ActorKind.HUMAN,
            ActorKind.INSTALLED_AGENT,
            ActorKind.EXTERNAL_AUTOMATION,
        } or (agent_execution is not None) != (actor_kind is ActorKind.INSTALLED_AGENT):
            raise ValueError("chief_risk_officer.actor_execution_evidence_invalid")
        self.answer = PortfolioReviewAnswer.model_validate(answer)
        self.dropped = dropped
        self.actor_kind = actor_kind
        self.actor_id = actor_id
        self.agent_execution = agent_execution
        self.model_call_count = model_call_count
        self.protocol_repair_count = protocol_repair_count
        if durable_submission and (
            actor_kind is ActorKind.INSTALLED_AGENT or model_call_count or protocol_repair_count
        ):
            raise ValueError("chief_risk_officer.external_submission_call_accounting_invalid")
        self.durable_submission = durable_submission
        self.observed_deadlines: list[float] = []

    @property
    def process_binding_hash(self) -> str:
        """Bind submitted-answer execution to its response structure and tool-free process.

        Returns:
            Canonical submitted-review process identity using the structural answer schema.
        """
        return _hash(
            {
                "owner": "chief_risk_officer.runtime.submitted_portfolio_review",
                "submission": schema_structure(PortfolioReviewAnswer),
                "execution": "TOOL_FREE_SINGLE_SEMANTIC_EXECUTION",
            }
        )

    def __call__(
        self, *, dossier: PortfolioReviewDossier, deadline_seconds: float
    ) -> PortfolioReviewActorResult:
        """Return the held answer and execution facts without invoking a model.

        Args:
            dossier: Protocol input; this adapter holds an answer already screened by its owner.
            deadline_seconds: Observed admitted deadline recorded for execution evidence.

        Returns:
            Held accepted answer and actor/call/repair facts; Host validation remains downstream.
        """
        del dossier
        self.observed_deadlines.append(deadline_seconds)
        return PortfolioReviewActorResult(
            answer=self.answer,
            dropped=self.dropped,
            actor_kind=self.actor_kind,
            actor_id=self.actor_id,
            agent_execution=self.agent_execution,
            model_call_count=self.model_call_count,
            protocol_repair_count=self.protocol_repair_count,
        )


class CarriedPortfolioReviewActor:
    """The last fresh review of the same basis, carried forward (W3).

    Nothing the reviewer read has changed -- the holdings, their bands, the
    findings and the open issues are the ones it assessed -- so its answer is
    the review as of this later day. The Host states it: no model call, no
    new statement to the issuers' register.
    """

    def __init__(self, receipt: PortfolioReviewReceipt, dossier: PortfolioReviewDossier) -> None:
        """Hold one fresh answered receipt and the exact dossier it assessed.

        Args:
            receipt: Earlier fresh receipt with an accepted answer.
            dossier: Source dossier whose identity matches the receipt.

        Raises:
            ValueError: The receipt lacks an answer, is already carried or binds another dossier.
        """
        if (
            receipt.answer is None
            or receipt.carried_forward
            or receipt.dossier_hash != dossier.dossier_hash
        ):
            raise ValueError("chief_risk_officer.carried_review_invalid")
        self.receipt = receipt
        self.dossier = dossier
        self.answer = receipt.answer
        self.actor_kind = ActorKind.HOST_FALLBACK
        self.actor_id = "carried-forward"

    @property
    def process_binding_hash(self) -> str:
        """Bind carried review execution to the original receipt and zero model calls.

        Returns:
            Canonical carried-review process identity, including carried_from receipt hash.
        """
        return _hash(
            {
                "owner": "chief_risk_officer.runtime.carried_portfolio_review",
                "carried_from": self.receipt.receipt_hash,
                "execution": "NO_MODEL_CALL",
            }
        )

    def __call__(
        self, *, dossier: PortfolioReviewDossier, deadline_seconds: float
    ) -> PortfolioReviewActorResult:
        """Translate the earlier submission into the target dossier without a model call.

        Args:
            dossier: Target admitted dossier on the same review basis.
            deadline_seconds: Protocol deadline; no semantic execution is performed here.

        Returns:
            Earlier accepted answer plus a carried assessment using target finding/issue handles.

        Raises:
            ValueError: The earlier assessment cannot be translated into the target dossier.
        """
        del deadline_seconds
        carried = translate_submission(self.receipt.submission, source=self.dossier, target=dossier)
        if carried is None:
            raise ValueError("chief_risk_officer.carried_review_untranslatable")
        return PortfolioReviewActorResult(
            answer=self.answer,
            actor_kind=self.actor_kind,
            actor_id=self.actor_id,
            carried=carried,
        )


@dataclass(frozen=True, slots=True)
class PortfolioReviewTaskResources:
    """What the Task is allowed to reach. No closure decides anything."""

    actor: PortfolioReviewActor
    decision_policy: CROHostPolicyBinding
    playpen_root: Path
    publications: PortfolioReviewPublicationService
    evidence_publications: AlternativeEvidenceAnalysisPublicationService
    response_schema_hash: str
    typed_user_authority: str
    actor_kind: str
    actor_id: str
    clock: Callable[[], datetime]
    monotonic: Callable[[], float] = time.monotonic
    verify_update_subject: Callable[[PortfolioUpdateReviewSubject], None] | None = None
    verify_experiment_subject: Callable[[PortfolioExperimentReviewSubject], None] | None = None
    verify_external_review_context: Callable[[PortfolioReviewDossier], None] | None = None


def portfolio_review_task_contract(
    *, review_key_payload: dict[str, object]
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    """The Task input envelope whose `input_hash` *is* the review key."""
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND, input_schema_id=INPUT_SCHEMA_ID, payload=review_key_payload
    )
    goal = ResearchGoal.create(
        goal_kind="REVIEW_PORTFOLIO_AGAINST_ALTERNATIVE_EVIDENCE",
        input_hash=envelope.input_hash,
        deliverable_kind="PortfolioReviewPublication",
        summary=(
            "Interpret admitted issuer evidence against one sealed Portfolio book and return "
            "one Host-routed recommendation without action authority."
        ),
        attributes={"decision_policy_hash": str(review_key_payload["decision_policy_hash"])},
    )
    work_items = tuple(
        WorkItemDefinition.create(
            stage_id=stage,
            dependency_ids=dependencies,
            verifier_id=f"chief-risk-officer.{stage}",
            required_evidence_kinds=(kind,),
        )
        for stage, dependencies, kind in _STAGES
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=_hash(tuple(value[0] for value in _STAGES)),
        verifier_catalog_hash=_hash(tuple(value.verifier_id for value in work_items)),
        work_items=work_items,
    )
    return envelope, goal, plan


class PortfolioReviewTaskAdapter:
    """Three business stages over Task Control's existing recovery boundaries."""

    task_kind = TASK_KIND

    def __init__(
        self,
        *,
        registry: DuckDbTaskControlRegistry,
        artifacts: AlternativeEvidenceArtifactStore,
        resources: PortfolioReviewTaskResources,
    ) -> None:
        """Bind the CRO three-stage workflow to Task Control and admitted review resources.

        Args:
            registry: Task Control registry owning state/recovery transitions.
            artifacts: Store owning exact CRO evidence and publication records.
            resources: Admitted policy, actor, clocks and source-context verification dependencies.
        """
        self.registry = registry
        self.artifacts = artifacts
        self.resources = resources
        self.model_call_count = 0
        self.protocol_repair_count = 0
        self.actor_stage_seconds: float | None = None

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Seal current response, workflow, input, policy and actor-process compatibility.

        Args:
            task: Task whose admitted plan and input schema contribute to compatibility.

        Returns:
            Exact compatibility binding using structural dossier schema and installed owner
            identities.
        """
        return TaskExecutionCompatibility.create(
            task_contract_hash=_hash(schema_structure(PortfolioReviewDossier)),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=self.resources.decision_policy.binding_hash,
            framework_identity_hash=self.resources.actor.process_binding_hash,
        )

    @verified_model_read_scope(reuse_verified=True)
    def execute_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
    ) -> StageExecutionResult:
        """Execute one installed CRO stage and convert failures to a blocked disposition.

        Args:
            task: Admitted Task record naming the exact review input.
            execution: Task Control execution context; this adapter uses the business task/stage.
            work_item: Installed admit, seal or publish stage to execute.

        Returns:
            READY with one exact artifact reference, or BLOCKED with a classified failure code.
        """
        del execution
        try:
            evidence = self._execute(task=task, stage=work_item.stage_id)
        except Exception as error:
            return StageExecutionResult(
                disposition=StageDisposition.BLOCKED,
                failure_code=_failure_code(error),
            )
        return StageExecutionResult(disposition=StageDisposition.READY, evidence=(evidence,))

    @verified_model_read_scope(reuse_verified=True)
    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Load the single artifact required by an installed CRO stage.

        Args:
            task: Task Control protocol record; artifact verification is keyed by evidence identity.
            execution: Task Control protocol execution context.
            work_item: Installed stage selecting dossier, receipt or publication verification.
            evidence: Exactly one artifact reference produced by that stage.

        Returns:
            The unchanged evidence tuple after the corresponding typed artifact readback.

        Raises:
            ValueError: The stage has other than one evidence item.
            AlternativeEvidencePublicationError: The typed artifact cannot be loaded or verified.
        """
        del execution, task
        if len(evidence) != 1:
            raise ValueError("chief_risk_officer.review_stage_evidence_invalid")
        stage = work_item.stage_id
        identity = evidence[0].content_hash
        if stage == "admit_portfolio_review":
            self.artifacts.load("cro-review-dossiers", identity, PortfolioReviewDossier)
        elif stage == "seal_portfolio_review_assessment":
            self.artifacts.load("cro-review-receipts", identity, PortfolioReviewReceipt)
        elif stage == "publish_portfolio_review":
            self.artifacts.load("cro-review-publications", identity, PortfolioReviewPublication)
        return evidence

    # ------------------------------------------------------------- stages

    def _execute(self, *, task: TaskRecord, stage: str) -> TaskEvidence:
        if stage == "admit_portfolio_review":
            dossier = self._admit(task)
            return _evidence("cro_review_dossier", dossier.dossier_hash)

        if stage == "seal_portfolio_review_assessment":
            dossier = self._dossier(task)
            if not self._evidence_is_current(dossier):
                raise PortfolioReviewEvidenceExpired("chief_risk_officer.review_evidence_expired")
            started = self.resources.monotonic()
            result = self.resources.actor(
                dossier=dossier, deadline_seconds=ACTOR_STAGE_BUDGET_SECONDS
            )
            self.actor_stage_seconds = self.resources.monotonic() - started
            if self.actor_stage_seconds > ACTOR_STAGE_BUDGET_SECONDS:
                raise PortfolioReviewTimeout("chief_risk_officer.review_actor_deadline_exceeded")
            self.model_call_count += result.model_call_count
            self.protocol_repair_count += result.protocol_repair_count
            receipt, recommendation = seal_portfolio_review(
                dossier=dossier,
                answer=result.answer,
                dropped=result.dropped,
                evidence_is_current=True,
                decision_policy=self.resources.decision_policy,
                playpen_root=self.resources.playpen_root,
                actor_kind=result.actor_kind,
                actor_id=result.actor_id,
                agent_execution=result.agent_execution,
                model_call_count=result.model_call_count,
                protocol_repair_count=result.protocol_repair_count,
                carried=result.carried,
            )
            self.artifacts.publish(
                "cro-review-recommendations", recommendation.recommendation_hash, recommendation
            )
            self.artifacts.publish("cro-review-receipts", receipt.receipt_hash, receipt)
            record_provider_stage_usage(
                self.artifacts,
                stage="CHIEF_RISK_OFFICER_REVIEWER",
                subject_hash=receipt.receipt_hash,
                usage=result.provider_usage,
                model_call_count=result.model_call_count,
            )
            return _evidence("cro_review_receipt", receipt.receipt_hash)

        if stage == "publish_portfolio_review":
            dossier = self._dossier(task)
            self._verify_external_context(task, dossier)
            receipt = self._receipt(task)
            recommendation = compile_portfolio_review_recommendation(
                dossier=dossier, receipt=receipt
            )
            from alphalattice.evidence.alternative_evidence.contracts import seal_contract

            publication = seal_contract(
                PortfolioReviewPublication,
                "publication_hash",
                review_key=task.input.input_hash,
                book_authority=dossier.book_authority,
                report_hash=dossier.report_hash,
                update_subject=dossier.update_subject,
                experiment_subject=dossier.experiment_subject,
                result_hash=dossier.result_hash,
                issuer_scope_hash=dossier.issuer_scope_hash,
                analysis_publication_hash=dossier.analysis_publication_hash,
                dossier_hash=dossier.dossier_hash,
                decision_receipt_hash=receipt.receipt_hash,
                recommendation_hash=recommendation.recommendation_hash,
                decision_policy_hash=receipt.decision_policy_hash,
                published_at=self.resources.clock(),
            )
            self.resources.publications.publish(publication=publication)
            return _evidence("cro_review_publication", publication.publication_hash)

        raise ValueError("chief_risk_officer.review_stage_unknown")

    # ------------------------------------------------------------ admission

    def _admit(self, task: TaskRecord) -> PortfolioReviewDossier:
        """Load the dossier the Task input names, and check every other binding."""

        payload = task.input.payload
        dossier = self.artifacts.load(
            "cro-review-dossiers", str(payload["dossier_hash"]), PortfolioReviewDossier
        )
        expected = {
            "report_hash": dossier.report_hash,
            "book_authority": str(dossier.book_authority),
            "issuer_scope_hash": dossier.issuer_scope_hash,
            "analysis_publication_hash": dossier.analysis_publication_hash,
            "cro_package_hash": dossier.cro_package_hash,
            "obligation_hash": dossier.obligation_hash,
            "dossier_hash": dossier.dossier_hash,
            "typed_user_authority": self.resources.typed_user_authority,
            "actor_kind": self.resources.actor_kind,
            "actor_id": self.resources.actor_id,
            "process_binding_hash": self.resources.actor.process_binding_hash,
            "response_schema_hash": self.resources.response_schema_hash,
            "execution_semantics": "TOOL_FREE_SINGLE_SEMANTIC_EXECUTION",
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(f"chief_risk_officer.review_input_binding_mismatch:{name}")
        # The policy a Task was admitted under runs while it names the installed one: a
        # recorded move (a comment, the rule's own rotation) keeps the review's meaning.
        if not is_current(
            PORTFOLIO_REVIEW_POLICY_ROLE,
            str(payload.get("decision_policy_hash")),
            self.resources.decision_policy.binding_hash,
        ):
            raise ValueError(
                "chief_risk_officer.review_input_binding_mismatch:decision_policy_hash"
            )
        subject = dossier.update_subject
        if payload.get("update_subject") != (
            None if subject is None else subject.model_dump(mode="json")
        ):
            raise ValueError("chief_risk_officer.review_input_binding_mismatch:update_subject")
        if payload.get("experiment_subject") != (
            None
            if dossier.experiment_subject is None
            else dossier.experiment_subject.model_dump(mode="json")
        ):
            raise ValueError("chief_risk_officer.review_input_binding_mismatch:experiment_subject")
        self._verify_subject(dossier)
        self._verify_external_context(task, dossier)
        prepared = payload.get("submission_hash")
        if prepared is not None and prepared != self._prepared_submission_hash():
            raise ValueError("chief_risk_officer.review_input_binding_mismatch:submission_hash")
        self.artifacts.publish("cro-review-dossiers", dossier.dossier_hash, dossier)
        return dossier

    def _verify_external_context(self, task: TaskRecord, dossier: PortfolioReviewDossier) -> None:
        if {"prepared_answer", "prepared_submission"} & set(task.input.payload):
            if self.resources.decision_policy != build_portfolio_review_policy_binding(
                self.resources.playpen_root
            ):
                raise ValueError("chief_risk_officer.external_review_policy_changed")
            if self.resources.verify_external_review_context is None:
                raise ValueError("chief_risk_officer.external_review_context_reader_missing")
            self.resources.verify_external_review_context(dossier)

    def _prepared_submission_hash(self) -> str | None:
        prepared = getattr(self.resources.actor, "answer", None)
        if prepared is None:
            return None
        return str(canonical_hash(prepared.model_dump(mode="json")))

    def _evidence_is_current(self, dossier: PortfolioReviewDossier) -> bool:
        """Every publication the dossier reads at its cutoff is current: one for
        the ordinary dossier, one per unit for a book analysed as several. A
        carried child is an earlier reading of filings still in the window,
        carried whatever its analysis's own expiry (W3), and is not asked to be
        current."""

        now = self.resources.clock()
        return all(
            self.resources.evidence_publications.replay(publication_hash, now=now).is_current
            for publication_hash in (
                tuple(
                    child.analysis_publication_hash
                    for child in dossier.evidence_children
                    if child.read_as_of is None
                )
                if dossier.evidence_children
                else dossier.evidence_publication_hashes
            )
        )

    def _dossier(self, task: TaskRecord) -> PortfolioReviewDossier:
        dossier = self.artifacts.load(
            "cro-review-dossiers",
            self._stage_identity(task, "admit_portfolio_review"),
            PortfolioReviewDossier,
        )
        self._verify_subject(dossier)
        return dossier

    def _verify_subject(self, dossier: PortfolioReviewDossier) -> None:
        if dossier.experiment_subject is not None:
            if self.resources.verify_experiment_subject is None:
                raise ValueError("chief_risk_officer.review_experiment_reader_missing")
            self.resources.verify_experiment_subject(dossier.experiment_subject)
        if dossier.update_subject is not None:
            if self.resources.verify_update_subject is None:
                raise ValueError("chief_risk_officer.review_update_reader_missing")
            self.resources.verify_update_subject(dossier.update_subject)

    def _receipt(self, task: TaskRecord) -> PortfolioReviewReceipt:
        return self.artifacts.load(
            "cro-review-receipts",
            self._stage_identity(task, "seal_portfolio_review_assessment"),
            PortfolioReviewReceipt,
        )

    def _stage_identity(self, task: TaskRecord, stage: str) -> str:
        matches = [
            value for value in self.registry.stage_receipts(task.task_id) if value.stage_id == stage
        ]
        if len(matches) != 1 or len(matches[0].evidence) != 1:
            raise ValueError(f"chief_risk_officer.review_stage_absent:{stage}")
        return str(matches[0].evidence[0].content_hash)


def _evidence(kind: str, identity: str) -> TaskEvidence:
    return TaskEvidence(
        evidence_kind=kind,
        reference=f"semantic://chief-risk-officer/{kind}/{identity}",
        content_hash=identity,
    )


def _hash(value: object) -> str:
    return str(canonical_hash(to_jsonable_python(value)))


def _failure_code(error: Exception) -> str:
    text = str(error)
    return (
        text
        if text.startswith("chief_risk_officer.")
        else f"chief_risk_officer.review_failed.{type(error).__name__.lower()}"
    )


__all__ = [
    "ACTOR_STAGE_BUDGET_SECONDS",
    "INPUT_SCHEMA_ID",
    "TASK_KIND",
    "CarriedPortfolioReviewActor",
    "PortfolioReviewActor",
    "PortfolioReviewActorResult",
    "PortfolioReviewEvidenceExpired",
    "PortfolioReviewTaskAdapter",
    "PortfolioReviewTaskResources",
    "PortfolioReviewTimeout",
    "SubmittedPortfolioReviewActor",
    "portfolio_review_task_contract",
]
