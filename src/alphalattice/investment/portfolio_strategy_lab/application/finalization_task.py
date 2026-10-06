"""One Portfolio finalization Task: permit, continuation, package, closure, release.

Five work items under the one Task Control runner that already exists, and one
task id across all of them. That is the whole point of the shape: the plan asks
that a single finalization identity survive permit issuance, the protected
numerical continuation, the pending package, closure and the handoff, and the
only way a runner can express "resume at closure" is if closure is its own
verified item.

The development Task is never touched. It is referenced by exact result, ledger
and program identities on the frozen candidate, and a finalization is a *new*
admission over those identities rather than an extension of a completed plan.

Every stage is idempotent against its own durable artifact, because recovery
lands in the middle of a protocol where repeating a step is not merely wasteful:
a second permit would be a second evaluation, a second continuation would charge
costs twice, and a second handoff would give downstream work two answers.

The rule that makes that true is narrow and absolute: **a stage returns an
artifact hash only after that exact artifact is durable.** Returning the hash of
something still in memory is what turns a crash into a repeated protected
evaluation -- the runner records the stage as reached, recovery looks for the
artifact, finds nothing, and computes it again.
"""

from __future__ import annotations

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
from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    FinalizationReleaseAuthority,
    FinalPortfolioEvaluationPackage,
    FrozenPortfolioCandidate,
    PortfolioFinalizationError,
    PortfolioValidationReceipt,
    ProtectedContinuationResult,
    ProtectedEvaluationPermit,
    ProtectedEvaluationPort,
    ProtectedFixture,
    ProtectedPathContinuation,
    ReleasedPortfolioArtifacts,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure

FINALIZATION_TASK_KIND = "portfolio_public_protected_finalization"
FINALIZATION_INPUT_SCHEMA = "portfolio-public-protected-finalization-candidate"
FINALIZATION_EVIDENCE_KIND = "portfolio-public.finalization-stage"

FINALIZATION_STAGES: tuple[str, ...] = (
    "claim_protected_permit",
    "continue_protected_path",
    "seal_pending_package",
    "verify_protected_closure",
    "release_validated_handoff",
)


def portfolio_finalization_task_contract(
    *, workspace_id: str, candidate: FrozenPortfolioCandidate, fixture: ProtectedFixture
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    """Admit one finalization over an immutable candidate and one fixture.

    The payload names the frozen candidate, the exact originating development
    identities and the fixture. It carries no metric, no window and no control:
    a finalization is not configurable, and a payload that could be steered would
    be a second research request wearing this task's name.
    """
    envelope = TaskInputEnvelope.create(
        task_kind=FINALIZATION_TASK_KIND,
        input_schema_id=FINALIZATION_INPUT_SCHEMA,
        payload={
            "workspace_id": workspace_id,
            "candidate_hash": candidate.candidate_hash,
            "configuration_hash": candidate.spec_hash,
            "program_hash": candidate.program_hash,
            "numerical_input_assembly_hash": candidate.numerical_input_assembly_hash,
            "development_task_id": candidate.development_task_id,
            "development_result_hash": candidate.development_result_hash,
            "pre_protected_state_hash": candidate.pre_protected_state.state_hash,
            "fixture_hash": fixture.fixture_hash,
            "fixture_disposition": fixture.disposition,
        },
    )
    goal = ResearchGoal.create(
        goal_kind="FINALIZE_FROZEN_PORTFOLIO_CANDIDATE_ON_PROTECTED_FIXTURE",
        input_hash=envelope.input_hash,
        deliverable_kind="ValidatedPortfolioHandoff",
        summary="Continue the sealed book across one permitted fixture and release it once.",
        attributes={
            "permits_issued": 1,
            "protected_evaluations": 1,
            "validation_task_adapters": 0,
            "validation_task_ids": 0,
        },
    )
    items = tuple(
        WorkItemDefinition.create(
            stage_id=stage,
            dependency_ids=() if index == 0 else (FINALIZATION_STAGES[index - 1],),
            verifier_id="portfolio-strategy-lab.finalization-stage-readback",
            required_evidence_kinds=(FINALIZATION_EVIDENCE_KIND,),
        )
        for index, stage in enumerate(FINALIZATION_STAGES)
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(
            {"stages": FINALIZATION_STAGES, "recovery": "RESUME_VERIFIED_FINALIZATION_PREFIX"}
        ),
        verifier_catalog_hash=canonical_hash(tuple({value.verifier_id for value in items})),
        work_items=items,
    )
    return envelope, goal, plan


class PortfolioFinalizationTaskAdapter:
    """Drive the five stages; own no numerics, no gate authority and no release."""

    task_kind = FINALIZATION_TASK_KIND

    def __init__(
        self,
        *,
        workspace_id: str,
        candidate: FrozenPortfolioCandidate,
        fixture: ProtectedFixture,
        gate: ProtectedEvaluationPort,
        continuation: ProtectedPathContinuation,
        release: FinalizationReleaseAuthority,
        store: PortfolioFinalizationStore,
    ) -> None:
        """Bind controlled finalization stages to exact candidate, permit and release owners.

        Stage products start absent and are rehydrated from durable evidence on resume. Operation
        counters record permits, continuations, publications, closures and release commits.

        Args:
            workspace_id: Explicit workspace identity.
            candidate: Exact frozen development candidate.
            fixture: Controlled fixture authority supplied by its owner.
            gate: Permit and exact-closure owner.
            continuation: Controlled path continuation owner.
            release: Validated artifact release authority.
            store: Durable finalization publication/readback store.
        """
        self.workspace_id = workspace_id
        self.candidate = candidate
        self.fixture = fixture
        self.gate = gate
        self.continuation = continuation
        self.release = release
        self.store = store
        self.permit: ProtectedEvaluationPermit | None = None
        self.continuation_result: ProtectedContinuationResult | None = None
        self.package: FinalPortfolioEvaluationPackage | None = None
        self.receipt: PortfolioValidationReceipt | None = None
        self.handoff_hash: str | None = None
        self.released_report_hash: str | None = None
        # Measured, not asserted: the acceptance matrix reads these back.
        self.permits_issued = 0
        self.protected_continuations = 0
        self.package_publications = 0
        self.closures = 0
        self.handoff_publications = 0
        self.release_commits = 0

    # ------------------------------------------------------------- binding

    def _require_task(self, task: TaskRecord) -> None:
        expected = {
            "workspace_id": self.workspace_id,
            "candidate_hash": self.candidate.candidate_hash,
            "configuration_hash": self.candidate.spec_hash,
            "program_hash": self.candidate.program_hash,
            "numerical_input_assembly_hash": self.candidate.numerical_input_assembly_hash,
            "development_task_id": self.candidate.development_task_id,
            "development_result_hash": self.candidate.development_result_hash,
            "pre_protected_state_hash": self.candidate.pre_protected_state.state_hash,
            "fixture_hash": self.fixture.fixture_hash,
            "fixture_disposition": self.fixture.disposition,
        }
        if (
            task.task_kind != self.task_kind
            or task.input.input_schema_id != FINALIZATION_INPUT_SCHEMA
            or task.input.payload != expected
        ):
            raise PortfolioFinalizationError("portfolio_finalization.task_binding_mismatch")

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Describe exact candidate schema, holdings policy and verified-prefix recovery authority.

        Args:
            task: Task whose finalization input and workflow must match this adapter.

        Returns:
            Typed compatibility binding for structural candidate schema, workflow/input route,
            holdings policy and candidate-specific recovery.
        """
        self._require_task(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(
                {"candidate": schema_structure(FrozenPortfolioCandidate)}
            ),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=self.candidate.holdings_spec_hash,
            framework_identity_hash=canonical_hash(
                {
                    "adapter": "PortfolioFinalizationTaskAdapter",
                    "candidate_hash": self.candidate.candidate_hash,
                    "recovery": "RESUME_VERIFIED_FINALIZATION_PREFIX",
                }
            ),
        )

    # ------------------------------------------------------------- stages

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Resume verified finalization state before dispatching an admitted stage.

        Args:
            task: Exact finalization task declaration.
            execution: Task execution record; stage dispatch does not consume this value.
            work_item: Declared finalization stage.

        Returns:
            READY with one durable stage artifact identity, or BLOCKED for an unknown stage.
        """
        del execution
        self._require_task(task)
        stage = work_item.stage_id
        if stage not in FINALIZATION_STAGES:
            return StageExecutionResult(
                disposition=StageDisposition.BLOCKED,
                failure_code="portfolio_finalization.task_stage_unknown",
            )
        # Rehydrate first. A resumed run is a new object in a new process, and
        # every step after the first depends on artifacts the previous attempt
        # already made durable.
        self._resume()
        content = getattr(self, f"_stage_{stage}")(str(task.task_id))
        return StageExecutionResult(
            disposition=StageDisposition.READY,
            evidence=(
                TaskEvidence(
                    evidence_kind=FINALIZATION_EVIDENCE_KIND,
                    reference=(f"playpen://portfolio-strategy-lab/finalization/{stage}/{content}"),
                    content_hash=content,
                ),
            ),
        )

    def rehydrate(self) -> None:
        """Recover durable state without running a stage.

        Needed when a caller drives a task that is *already* succeeded: the
        runner has nothing to dispatch, so nothing would populate this adapter
        and completion checks would run against an empty object and report the
        finalization incomplete. Reading is not running.
        """
        self._resume()

    def _resume(self) -> None:
        """Recover every durable artifact this finalization already made, computing nothing.

        Runs before stage execution and verification. A resumed run
        is a new object in a new process, and the cheapest way to guarantee it
        never repeats an expensive step is to make each stage look for the answer
        before it considers producing one.
        """

        if self.permit is None:
            self.permit = self.gate.reopen_permit(candidate=self.candidate)
        if self.permit is None:
            return
        if self.continuation_result is None:
            self.continuation_result = self.store.find_continuation_for_permit(
                self.permit.permit_hash
            )
        if self.package is None:
            self.package = self.store.find_package_for_permit(self.permit.permit_hash)
        if self.receipt is None:
            self.receipt = self.gate.reopen_receipt(permit=self.permit)

    def _stage_claim_protected_permit(self, task_id: str) -> str:
        del task_id
        if self.permit is None:
            # Reopened above when one exists; the Gate refuses a second issuance,
            # and a resumed run must not turn that refusal into a failure.
            self.permit = self.gate.admit(candidate=self.candidate, fixture=self.fixture)
            self.permits_issued += 1
        return self.permit.permit_hash

    def _stage_continue_protected_path(self, task_id: str) -> str:
        del task_id
        permit = self._require_permit()
        if self.continuation_result is None:
            # Reopened above when one exists. Running the continuation again
            # would charge the same costs a second time under one permit, which
            # is the failure the whole one-time protocol is built to prevent.
            result = self.continuation.evaluate(permit=permit, candidate=self.candidate)
            self.protected_continuations += 1
            if result.continued_from_state_hash != self.candidate.pre_protected_state.state_hash:
                raise PortfolioFinalizationError(
                    "portfolio_finalization.continuation_did_not_continue_the_sealed_state"
                )
            # Committed before the hash is returned. The window between computing
            # a protected path and recording that it exists is exactly where a
            # crash would cost a second evaluation.
            self.store.publish_continuation(result)
            self.continuation_result = result
        return self.continuation_result.result_hash

    def _stage_seal_pending_package(self, task_id: str) -> str:
        del task_id
        permit = self._require_permit()
        result = self._require_continuation()
        if self.package is None:
            package = FinalPortfolioEvaluationPackage.create(
                permit_hash=permit.permit_hash,
                candidate_hash=self.candidate.candidate_hash,
                continued_from_state_hash=result.continued_from_state_hash,
                fixture_hash=self.fixture.fixture_hash,
                protected_result_hash=result.protected_result_hash,
                protected_execution_ledger_hash=result.execution_ledger_hash,
                protected_economic_ledger_hash=result.economic_ledger_hash,
                protected_report_hash=result.report_hash,
                protected_formation_count=result.formation_count,
                protected_listing_count=result.listing_count,
            )
            # Determined entirely by the durable continuation, so a crash before
            # this line costs one cheap rebuild and never a rerun.
            self.store.publish_package(package)
            self.package_publications += 1
            self.package = package
        return self.package.package_hash

    def _stage_verify_protected_closure(self, task_id: str) -> str:
        del task_id
        permit = self._require_permit()
        package = self._require_package()
        if self.receipt is None:
            self.receipt = self.gate.close(permit=permit, candidate=self.candidate, package=package)
            self.closures += 1
        if not self.receipt.closed:
            raise PortfolioFinalizationError(
                "portfolio_finalization.closure_refused:" + self.receipt.closure
            )
        return self.receipt.receipt_hash

    def _stage_release_validated_handoff(self, task_id: str) -> str:
        """Three steps, in the only order that keeps every window recoverable.

        The handoff is minted and committed first, because it is the one piece
        that cannot be recomputed: it carries a release instant, so a recovery
        that re-minted it would hand a reader a different identity than the one
        the previous attempt might already have written down.

        Then the artifacts are copied. That is idempotent preparation, writes no
        index, and releases nothing on its own.

        Then the release marker is committed. *That* is the release: one atomic
        fact binding the package, the receipt, the handoff and the exact result,
        and the thing every public reader of a protected result has to find
        before it may read one.
        """

        package = self._require_package()
        handoff = self.store.find_handoff_for_package(package.package_hash)
        if handoff is None:
            handoff = self.release.release(
                package=package,
                receipt=self._require_receipt(),
                candidate=self.candidate,
                finalization_task_id=task_id,
            )
            self.store.publish_handoff(handoff)
            self.handoff_publications += 1
        self.handoff_hash = handoff.handoff_hash
        self.released_report_hash = handoff.released_report_hash
        if self.store.find_release_for_package(package.package_hash) is None:
            self.release.promote(package)
            self.store.publish_release(
                ReleasedPortfolioArtifacts.create(
                    workspace_id=self.workspace_id,
                    package_hash=package.package_hash,
                    validation_receipt_hash=self._require_receipt().receipt_hash,
                    handoff_hash=handoff.handoff_hash,
                    released_result_hash=package.protected_result_hash,
                    released_report_hash=package.protected_report_hash,
                    released_at=handoff.released_at,
                )
            )
            self.release_commits += 1
        return handoff.handoff_hash

    # ---------------------------------------------------------- verification

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Reopen stage authority and prove durable permit, closure, continuation or release.

        Verification reads retained authority without issuing a permit, evaluating a path or
        releasing artifacts. A release verifies both the handoff and reopened marker against the
        exact package and closure.

        Args:
            task: Exact finalization task declaration.
            execution: Task execution record; verification does not consume this value.
            work_item: Admitted finalization stage whose proof is checked.
            evidence: Exactly one stage-kind evidence record.

        Returns:
            Original evidence after exact durable readback checks.

        Raises:
            PortfolioFinalizationError: Stage/evidence shape, durable identity, pending visibility,
                closure or handoff/release marker binding differs.
        """
        del execution
        self._require_task(task)
        if (
            work_item.stage_id not in FINALIZATION_STAGES
            or len(evidence) != 1
            or evidence[0].evidence_kind != FINALIZATION_EVIDENCE_KIND
        ):
            raise PortfolioFinalizationError("portfolio_finalization.task_evidence_invalid")
        # Prefix verification can be the first call on a freshly composed
        # adapter. Read existing authority; do not issue, evaluate or release.
        self._resume()
        if (
            work_item.stage_id == "claim_protected_permit"
            and self._require_permit().permit_hash != evidence[0].content_hash
        ):
            raise PortfolioFinalizationError("portfolio_finalization.permit_evidence_not_durable")
        if work_item.stage_id == "verify_protected_closure":
            receipt = self._require_receipt()
            if not receipt.closed or receipt.receipt_hash != evidence[0].content_hash:
                raise PortfolioFinalizationError(
                    "portfolio_finalization.closure_evidence_not_durable"
                )
        if work_item.stage_id == "continue_protected_path":
            # Opened, not remembered: the point of the stage is that the result
            # survived the process that produced it.
            durable_result = self.store.find_continuation_for_permit(
                self._require_permit().permit_hash
            )
            if durable_result is None or durable_result.result_hash != evidence[0].content_hash:
                raise PortfolioFinalizationError(
                    "portfolio_finalization.continuation_evidence_not_durable"
                )
        if work_item.stage_id == "seal_pending_package":
            durable = self.store.load_package(evidence[0].content_hash)
            if durable.visibility != "PENDING_CLOSURE":
                raise PortfolioFinalizationError(
                    "portfolio_finalization.package_released_before_closure"
                )
        if work_item.stage_id == "release_validated_handoff":
            package = self._require_package()
            durable = self.store.load_handoff(evidence[0].content_hash)
            if (
                durable.package_hash != package.package_hash
                or durable.validation_receipt_hash != self._require_receipt().receipt_hash
            ):
                raise PortfolioFinalizationError("portfolio_finalization.handoff_binding_invalid")
            # And the marker, reopened. A verified release stage that checked only
            # the handoff would pass on a finalization whose artifacts were
            # adopted and never released.
            marker = self.store.find_release_for_package(package.package_hash)
            if marker is None:
                raise PortfolioFinalizationError("portfolio_finalization.release_marker_absent")
            if (
                marker.handoff_hash != durable.handoff_hash
                or marker.validation_receipt_hash != durable.validation_receipt_hash
                or marker.released_result_hash != package.protected_result_hash
                or marker.released_report_hash != package.protected_report_hash
                or marker.released_report_hash != durable.released_report_hash
            ):
                raise PortfolioFinalizationError(
                    "portfolio_finalization.release_marker_binding_invalid"
                )
        return evidence

    def _require_continuation(self) -> ProtectedContinuationResult:
        if self.continuation_result is None:
            raise PortfolioFinalizationError(
                "portfolio_finalization.continuation_stage_not_reached"
            )
        return self.continuation_result

    def _require_permit(self) -> ProtectedEvaluationPermit:
        if self.permit is None:
            raise PortfolioFinalizationError("portfolio_finalization.permit_stage_not_reached")
        return self.permit

    def _require_package(self) -> FinalPortfolioEvaluationPackage:
        if self.package is None:
            raise PortfolioFinalizationError("portfolio_finalization.package_stage_not_reached")
        return self.package

    def _require_receipt(self) -> PortfolioValidationReceipt:
        if self.receipt is None:
            raise PortfolioFinalizationError("portfolio_finalization.closure_stage_not_reached")
        return self.receipt


__all__ = [
    "FINALIZATION_EVIDENCE_KIND",
    "FINALIZATION_INPUT_SCHEMA",
    "FINALIZATION_STAGES",
    "FINALIZATION_TASK_KIND",
    "PortfolioFinalizationTaskAdapter",
    "portfolio_finalization_task_contract",
]
