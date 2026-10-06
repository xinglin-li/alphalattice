"""One Portfolio domain adapter under the shared Task Control runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from alphalattice.control.task_control.runner import (
    StageDisposition,
    StageExecutionResult,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
    PortfolioExecutionProgram,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    ScoreSourceMode,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    PortfolioResearchExecutor,
    ResolvedPortfolioExecution,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    EligiblePoolShort,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure

PORTFOLIO_PUBLIC_TASK_KIND = "portfolio_public_development_replay"
PORTFOLIO_PUBLIC_INPUT_SCHEMA = "portfolio-research-durable-request"
PORTFOLIO_PUBLIC_STAGE = "execute_and_publish_declared_path"
PORTFOLIO_PUBLIC_EVIDENCE_KIND = "portfolio-public.result"


class PortfolioResearchTaskInput(BaseModel):  # type: ignore[misc]
    """The complete durable request from which one Portfolio command resumes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace_id: str
    workspace_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec: PortfolioResearchSpec
    selected_strategy_package_id: str
    selected_score_source_mode: ScoreSourceMode
    selected_strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program: PortfolioExecutionProgram
    admission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_selection(self) -> Self:
        """Require task spec, selected package/mode and program holdings identity to agree.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Explicit/default strategy selection, score-source mode, program package/mode
                or holdings spec identity differs.
        """
        if (
            self.spec.strategy_package_id != WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID
            and self.spec.strategy_package_id != self.selected_strategy_package_id
        ):
            raise ValueError("portfolio_application.task_strategy_selection_mismatch")
        if self.spec.score_source_mode != self.selected_score_source_mode:
            raise ValueError("portfolio_application.task_score_source_mode_mismatch")
        if (
            self.program.strategy_package_hash != self.selected_strategy_package_hash
            or self.program.score_source_mode != self.selected_score_source_mode
        ):
            raise ValueError("portfolio_application.task_program_strategy_mismatch")
        if self.program.holdings_spec_hash != self.spec.holdings_spec_hash:
            raise ValueError("portfolio_application.task_program_spec_mismatch")
        return self


def portfolio_research_task_input(task: TaskRecord) -> PortfolioResearchTaskInput:
    """Read the typed request owned by one durable Task record."""
    if (
        task.task_kind != PORTFOLIO_PUBLIC_TASK_KIND
        or task.input.input_schema_id != PORTFOLIO_PUBLIC_INPUT_SCHEMA
    ):
        raise ValueError("portfolio_application.task_input_schema_mismatch")
    return cast(
        PortfolioResearchTaskInput,
        PortfolioResearchTaskInput.model_validate(task.input.payload),
    )


def portfolio_research_task_contract(
    *,
    workspace_id: str,
    spec: PortfolioResearchSpec,
    program: PortfolioExecutionProgram,
    admission_hash: str,
    authorities_hash: str,
    workspace_manifest_hash: str,
    strategy_catalog_hash: str,
    selected_strategy_package_id: str,
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    """Bind the plan the run was authorised against, not only the request.

    A task keyed on the spec and the program alone can be recovered against a
    different plan taken over different owner artifacts, and would then publish a
    result no plan ever described. What is bound is the plan's `admission_hash`
    -- request, authorities, catalog, window and guards -- rather than the whole
    preview, because the preview also carries cache state and a work estimate,
    and those change the moment the run publishes.
    """
    if program.authorities_hash != authorities_hash:
        raise ValueError("portfolio_application.task_program_authorities_mismatch")

    task_input = PortfolioResearchTaskInput(
        workspace_id=workspace_id,
        workspace_manifest_hash=workspace_manifest_hash,
        spec=spec,
        selected_strategy_package_id=selected_strategy_package_id,
        selected_score_source_mode=program.score_source_mode,
        selected_strategy_package_hash=program.strategy_package_hash,
        strategy_catalog_hash=strategy_catalog_hash,
        program=program,
        admission_hash=admission_hash,
    )
    envelope = TaskInputEnvelope.create(
        task_kind=PORTFOLIO_PUBLIC_TASK_KIND,
        input_schema_id=PORTFOLIO_PUBLIC_INPUT_SCHEMA,
        payload=task_input.model_dump(mode="json"),
    )
    goal = ResearchGoal.create(
        goal_kind="EXECUTE_PUBLIC_PORTFOLIO_DEVELOPMENT_REPLAY",
        input_hash=envelope.input_hash,
        deliverable_kind="PortfolioResearchResult",
        summary="Execute or exactly reuse the frozen public Portfolio path and publish its report.",
        attributes={
            "optimizer_calls": 0,
            "alpha_fits": 0,
            "dense_covariance_materializations": 0,
            "sector_forecast_calls": 0,
        },
    )
    item = WorkItemDefinition.create(
        stage_id=PORTFOLIO_PUBLIC_STAGE,
        dependency_ids=(),
        verifier_id="portfolio-strategy-lab.public-result-readback",
        required_evidence_kinds=(PORTFOLIO_PUBLIC_EVIDENCE_KIND,),
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash(
            {
                "stages": (PORTFOLIO_PUBLIC_STAGE,),
                "recovery": "IDEMPOTENT_PROGRAM_RESULT_REOPEN",
            }
        ),
        verifier_catalog_hash=canonical_hash((item.verifier_id,)),
        work_items=(item,),
    )
    return envelope, goal, plan


class PortfolioResearchTaskAdapter:
    """Execute one already-compiled Program; Task Control owns lifecycle."""

    task_kind = PORTFOLIO_PUBLIC_TASK_KIND

    def __init__(
        self,
        *,
        workspace_id: str,
        spec: PortfolioResearchSpec,
        program: PortfolioExecutionProgram,
        resolved: Callable[[], ResolvedPortfolioExecution],
        coverage: PortfolioSupportCoverage,
        authorities_hash: str,
        admission_hash: str,
        workspace_manifest_hash: str,
        strategy_catalog_hash: str,
        selected_strategy_package_id: str,
        executor: PortfolioResearchExecutor,
    ) -> None:
        """Bind one portfolio task to exact admitted spec/program/input and executor owners.

        Args:
            workspace_id: Explicit workspace identity.
            spec: Admitted public research controls.
            program: Exact sealed execution program.
            resolved: Deterministic input resolver invoked by the executor.
            coverage: Exact common source support.
            authorities_hash: Exact shared authority binding.
            admission_hash: Exact task admission identity.
            workspace_manifest_hash: Workspace manifest authority.
            strategy_catalog_hash: Installed strategy declaration catalog.
            selected_strategy_package_id: Explicit selected installed package.
            executor: Numerical execution and durable readback owner.
        """
        self.workspace_id = workspace_id
        self.spec = spec
        self.program = program
        self.resolved = resolved
        self.coverage = coverage
        self.authorities_hash = authorities_hash
        self.admission_hash = admission_hash
        self.workspace_manifest_hash = workspace_manifest_hash
        self.strategy_catalog_hash = strategy_catalog_hash
        self.selected_strategy_package_id = selected_strategy_package_id
        self.executor = executor

    def _require_task(self, task: TaskRecord) -> None:
        if self.program.authorities_hash != self.authorities_hash:
            raise ValueError("portfolio_application.task_program_authorities_mismatch")
        expected = PortfolioResearchTaskInput(
            workspace_id=self.workspace_id,
            workspace_manifest_hash=self.workspace_manifest_hash,
            spec=self.spec,
            selected_strategy_package_id=self.selected_strategy_package_id,
            selected_score_source_mode=self.program.score_source_mode,
            selected_strategy_package_hash=self.program.strategy_package_hash,
            strategy_catalog_hash=self.strategy_catalog_hash,
            program=self.program,
            admission_hash=self.admission_hash,
        ).model_dump(mode="json")
        if (
            task.task_kind != self.task_kind
            or task.input.input_schema_id != PORTFOLIO_PUBLIC_INPUT_SCHEMA
            or task.input.payload != expected
        ):
            raise ValueError("portfolio_application.task_binding_mismatch")

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Describe structural spec/program contracts and exact-result reopen recovery.

        Args:
            task: Task whose input/workflow must match this adapter.

        Returns:
            Compatibility binding for structural schemas, workflow/input route, policy recipe and
            exact program recovery.
        """
        self._require_task(task)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash(
                {
                    "spec": schema_structure(PortfolioResearchSpec),
                    "program": schema_structure(PortfolioExecutionProgram),
                }
            ),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=self.program.policy_recipe_hash,
            framework_identity_hash=canonical_hash(
                {
                    "executor": "PortfolioResearchExecutor",
                    "program_hash": self.program.program_hash,
                    "recovery": "EXACT_RESULT_REOPEN",
                }
            ),
        )

    def execute_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
    ) -> StageExecutionResult:
        """Execute the admitted public portfolio stage and retain one result identity.

        Args:
            task: Exact admitted portfolio task.
            execution: Task execution record unused by this dispatch.
            work_item: Declared public portfolio stage.

        Returns:
            READY with one durable result URI/hash, or BLOCKED for an unknown stage.
        """
        del execution
        self._require_task(task)
        if work_item.stage_id != PORTFOLIO_PUBLIC_STAGE:
            return StageExecutionResult(
                disposition=StageDisposition.BLOCKED,
                failure_code="portfolio_application.task_stage_unknown",
            )
        try:
            result = self.executor.execute(
                workspace_id=self.workspace_id,
                spec=self.spec,
                program=self.program,
                resolved=self.resolved,
                coverage=self.coverage,
                authorities_hash=self.authorities_hash,
            )
        except EligiblePoolShort as refused:
            # The book refused before it opened, by the formation too short for a rebalance: the
            # owner's stop, which a resume of the same Program would only meet again (V519).
            return StageExecutionResult(
                disposition=StageDisposition.BLOCKED, failure_code=str(refused)
            )
        return StageExecutionResult(
            disposition=StageDisposition.READY,
            evidence=(
                TaskEvidence(
                    evidence_kind=PORTFOLIO_PUBLIC_EVIDENCE_KIND,
                    reference=self.executor.store.content.uri("results", result.result_hash),
                    content_hash=result.result_hash,
                ),
            ),
        )

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Reopen result, program, execution, economics and report for exact stage evidence.

        Args:
            task: Exact admitted portfolio task.
            execution: Task execution record unused by verification.
            work_item: Declared public portfolio stage.
            evidence: Exactly one result-kind evidence record.

        Returns:
            Original evidence after durable identity and associated-artifact readback.

        Raises:
            ValueError: Stage/evidence shape, result program/hash or result URI differs.
        """
        del execution
        self._require_task(task)
        if (
            work_item.stage_id != PORTFOLIO_PUBLIC_STAGE
            or len(evidence) != 1
            or evidence[0].evidence_kind != PORTFOLIO_PUBLIC_EVIDENCE_KIND
        ):
            raise ValueError("portfolio_application.task_evidence_invalid")
        durable = self.executor.store.load_result(evidence[0].content_hash)
        if (
            durable.program_hash != self.program.program_hash
            or durable.result_hash != evidence[0].content_hash
            or evidence[0].reference
            != self.executor.store.content.uri("results", durable.result_hash)
        ):
            raise ValueError("portfolio_application.task_evidence_invalid")
        self.executor.store.load_program(durable.program_hash)
        self.executor.store.load_execution(durable.execution_ledger_hash)
        self.executor.store.load_economics(durable.economic_ledger_hash)
        self.executor.store.load_report(durable.report_hash)
        return evidence


__all__ = [
    "PORTFOLIO_PUBLIC_TASK_KIND",
    "PortfolioResearchTaskAdapter",
    "PortfolioResearchTaskInput",
    "portfolio_research_task_contract",
    "portfolio_research_task_input",
]
