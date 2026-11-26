"""Application composition for PLAN, RUN/reuse, recovery, REPORT, and EXPORT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.publication.portfolio_research import (
    PortfolioResearchPipelineManifest,
    PortfolioResearchPipelineStore,
)
from alphalattice.control.product_host.storage.input_references import ResearchInputStorage
from alphalattice.control.product_host.storage.inventory import storage_capacity_scope
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord, TaskReplan
from alphalattice.control.task_control.runner import TaskControlRunner
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioControlReceipt,
    PortfolioDeclaredPathReport,
    PortfolioExecutionProgram,
    PortfolioPlanPreview,
    PortfolioResearchResult,
    PortfolioResearchSpec,
    build_schedule_guard,
    export_command,
)
from alphalattice.investment.portfolio_strategy_lab.application.controls import (
    INSTALLED_PUBLIC_CONTROL_CATALOG,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    PortfolioResearchExecutor,
    ResolvedPortfolioExecution,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    PortfolioExecutionResolver,
    PortfolioResearchCompiler,
    ResolvedPortfolioAuthorities,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PortfolioResearchTaskAdapter,
    portfolio_research_task_contract,
    portfolio_research_task_input,
)
from alphalattice.investment.portfolio_strategy_lab.inputs.shared_lanes import (
    PortfolioResearchCompositionError,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.static import PortfolioReportContext

PUBLIC_REFUSALS: tuple[str, ...] = tuple(
    value.refusal_code for value in INSTALLED_PUBLIC_CONTROL_CATALOG.refusals
)
"""Rendered with every preview so a refusal is visible before a request, not after."""

PLAN_FIELDS = ("default_strategy_package_id", "default_score_source_mode", "strategy_installation")
"""The workspace manifest's fields a Portfolio research plan reads: the strategy the resolver
serves. A plan binds these, never the whole manifest, so a capture, a training or a scoring
publication leaves an admitted research Task recoverable (V284)."""


@dataclass(frozen=True, slots=True)
class PlannedPortfolioResearch:
    """Retain the exact plan preview and resolved input/strategy authority."""

    preview: PortfolioPlanPreview
    authorities: ResolvedPortfolioAuthorities


@dataclass(frozen=True, slots=True)
class CompletedPortfolioResearch:
    """Retain the completed research result and sealed pipeline manifest."""

    result: PortfolioResearchResult
    pipeline_manifest: PortfolioResearchPipelineManifest


@dataclass(frozen=True, slots=True)
class AdmittedPortfolioResearch:
    """A run that exists in Task Control and has not been executed.

    The split `run()` used to hide. Admission compiles or reopens a Program and
    writes one queued task; execution walks the book. A caller that wants to
    return before the numerical work -- a local service answering a browser --
    needs the first half on its own, and needs it to have already produced the
    task identity everything afterwards is projected from.

    Nothing here is a second lifecycle. The task is the same task, admitted
    through the same registry, and `execute()` drives the same runner it always
    did.
    """

    task_id: UUID
    spec: PortfolioResearchSpec
    planned: PlannedPortfolioResearch
    program: PortfolioExecutionProgram
    authorities: ResolvedPortfolioAuthorities
    resolve_once: Callable[[], ResolvedPortfolioExecution]
    already_succeeded: bool
    lifecycle: str
    """The lifecycle the registry wrote, carried out of the admission itself."""


class PortfolioResearchApplication:
    """One application session and one shared Task Control runner per command."""

    replans = (
        TaskReplan(
            task_kind=PortfolioResearchTaskAdapter.task_kind, preview="PLAN", admitting="RUN"
        ),
    )
    """The re-plan of the development replay this application admits, which the
    recovery view offers (V188)."""

    def __init__(
        self,
        *,
        workspace_id: str,
        workspace: Path,
        manifest_binding: Callable[[], str],
        session: WorkspaceApplicationSession,
        resolver: PortfolioExecutionResolver,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        report_context: Callable[[PortfolioExecutionProgram, date], PortfolioReportContext]
        | None = None,
    ) -> None:
        """Compose exact portfolio input, task, storage, execution and report owners.

        Args:
            workspace_id: Explicit workspace identity.
            workspace: Caller-owned workspace root.
            manifest_binding: Owner callback for the manifest plan-field binding.
            session: Retained writer/task session.
            resolver: Deterministic strategy/input resolution owner.
            clock: Explicit observed-time source.
            report_context: Optional deterministic report context resolver.
        """
        self.workspace_id = workspace_id
        self.workspace = workspace.resolve()
        self.manifest_binding = manifest_binding
        """The plan fields' hash as the manifest stands now (`PLAN_FIELDS`), read from the
        Host's one manifest holder, so a publication another owner makes is seen (V284)."""
        self.session = session
        self.resolver = resolver
        self.clock = clock
        self.ledger = PortfolioLedgerStore.for_workspace(
            self.workspace, capacity=ResearchInputStorage(session).admit_evidence_bytes
        )
        artifact_root = self.ledger.artifact_root
        self.pipeline = PortfolioResearchPipelineStore(artifact_root)
        # The installed declarations, so a report rebuilt from a stored ledger
        # can still say which strategy produced it without resolving anything.
        self.executor = PortfolioResearchExecutor(
            self.ledger, packages=resolver.installed_packages(), report_context=report_context
        )
        self.compiler = PortfolioResearchCompiler()

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Replan the durable book's exact controls and selected installed package.

        Args:
            task: Durable Task whose declaration supplies the next request.

        Returns:
            The filled book planning request, independent of a changed workspace default.

        Raises:
            ValueError: The Task's input schema or retained selection does not validate.
        """
        durable = portfolio_research_task_input(task)
        return {
            "operation": "PLAN",
            "spec": {
                **dict(PortfolioControlReceipt.of(durable.spec).selected),
                "strategy_package_id": durable.selected_strategy_package_id,
                "score_source_mode": durable.selected_score_source_mode,
            },
        }

    def plan(self, spec: PortfolioResearchSpec) -> PlannedPortfolioResearch:
        """Resolve authorities, bound the request, and estimate. Run nothing.

        The window is checked here, before any numerical path is entered, and an
        out-of-support request names the owner that ran out rather than failing
        somewhere inside a replay.
        """
        authorities = self.resolver.resolve_authorities(workspace=self.workspace, spec=spec)
        coverage = authorities.coverage
        sessions = authorities.candidate_sessions
        selected_start = spec.study_start or sessions[0]
        selected_end = spec.study_end or sessions[-1]
        if selected_start > selected_end:
            raise PortfolioResearchCompositionError(
                "portfolio_application.study_window_axis_invalid"
            )
        limiting = coverage.limiting_owners(start=selected_start, end=selected_end)
        if limiting:
            raise PortfolioResearchCompositionError(
                "portfolio_application.study_window_outside_support:" + ",".join(limiting)
            )
        if selected_start < sessions[0] or selected_end > sessions[-1]:
            raise PortfolioResearchCompositionError(
                "portfolio_application.study_window_outside_common_watermark"
            )
        session_axis = frozenset(sessions)
        missing_sessions = tuple(
            dict.fromkeys(
                value for value in (selected_start, selected_end) if value not in session_axis
            )
        )
        if missing_sessions:
            raise PortfolioResearchCompositionError(
                "portfolio_application.study_window_session_absent:"
                + ",".join(value.isoformat() for value in missing_sessions)
            )
        ceiling = min(spec.exit_rank_band()[1], authorities.eligible_count)
        if spec.exit_rank > ceiling:
            raise PortfolioResearchCompositionError(
                f"portfolio_application.exit_rank_exceeds_eligible_count:{spec.exit_rank}>{ceiling}"
            )
        cached = self.ledger.find_planned_result(
            spec_hash=spec.spec_hash, authorities_hash=authorities.authorities_hash
        )
        # Three states, resolved in order of how little they cost. A published
        # result answers everything; otherwise a materialized path for this exact
        # holdings configuration over these exact authorities still saves the
        # walk; only then is it a full miss.
        known_program = self.ledger.find_program_for(
            holdings_spec_hash=spec.holdings_spec_hash,
            authorities_hash=authorities.authorities_hash,
        )
        materialized = None
        if cached is not None:
            materialized = self.ledger.load_execution(cached.execution_ledger_hash)
        elif known_program is not None:
            materialized = self.ledger.find_execution_for_program(known_program.program_hash)
        cache_state = (
            "RESULT_HIT"
            if cached is not None
            else ("EXECUTION_LEDGER_HIT" if materialized is not None else "FULL_NUMERICAL_MISS")
        )
        ledger_coverage: tuple[str, str, int] | None = None
        if materialized is not None:
            # Read from the materialized path itself, so ledger coverage is a
            # fact about what exists rather than a restatement of owner support.
            ledger_coverage = (
                materialized.formation_sessions[0].isoformat(),
                materialized.formation_sessions[-1].isoformat(),
                len(materialized.formation_sessions),
            )
        preview = PortfolioPlanPreview.create(
            workspace_id=self.workspace_id,
            workspace_manifest_hash=self.manifest_binding(),
            spec_hash=spec.spec_hash,
            holdings_spec_hash=spec.holdings_spec_hash,
            authorities_hash=authorities.authorities_hash,
            strategy_catalog_hash=self.resolver.strategy_catalog_hash,
            strategy_package_id=authorities.strategy_package_id,
            strategy_package_hash=authorities.strategy_package_hash,
            score_source_mode=authorities.score_source_mode,
            control_catalog_hash=INSTALLED_PUBLIC_CONTROL_CATALOG.catalog_hash,
            coverage=coverage,
            schedule_guard=build_schedule_guard(tranches=spec.tranches),
            selected_study_start=selected_start.isoformat(),
            selected_study_end=selected_end.isoformat(),
            full_support_start=sessions[0].isoformat(),
            full_support_end=sessions[-1].isoformat(),
            candidate_formation_count=len(sessions),
            listing_count=len(authorities.ordered_listing_ids),
            eligible_count=authorities.eligible_count,
            cache_state=cache_state,
            exact_cache_hit=cached is not None,
            cached_result_hash=None if cached is None else cached.result_hash,
            execution_ledger_coverage=ledger_coverage,
            # Only a full miss walks the book. An execution-ledger hit rebuilds
            # descendants from the stored path and replays nothing.
            prefix_work_formation_count=len(sessions) if materialized is None else 0,
            estimated_score_replays=len(sessions) if materialized is None else 0,
            estimated_risk_surface_builds=len(sessions) if materialized is None else 0,
            legal_recovery=(
                "REOPEN_EXACT_PUBLISHED_RESULT",
                "RESUME_RECOVERY_REQUIRED_TASK_WITH_SAME_PROGRAM",
            ),
            refusals=PUBLIC_REFUSALS,
        )
        return PlannedPortfolioResearch(preview=preview, authorities=authorities)

    def run(
        self,
        *,
        spec: PortfolioResearchSpec,
        planned: PlannedPortfolioResearch | None = None,
    ) -> CompletedPortfolioResearch:
        """Admit and execute in one call; an exact reuse answers without a Task (V195)."""
        work = planned or self.plan(spec)
        reused = self.reused(spec, work)
        return reused if reused is not None else self.execute(self.admit(spec=spec, planned=work))

    def reused(
        self, spec: PortfolioResearchSpec, planned: PlannedPortfolioResearch
    ) -> CompletedPortfolioResearch | None:
        """An exact reuse, answered from the stored result without a Task.

        The one reuse rule `strategy-book run` and the synchronous route (the standalone script)
        keep (V195): the plan decides reuse, and the stored result answers under the Task that first
        produced it, taking no Task, no place in the queue and no second publication. A result
        the by-result index does not name (published before the index) answers None, so one Task
        is admitted and its publication names it.
        """
        if planned.preview.spec_hash != spec.spec_hash:
            raise ValueError("portfolio_application.plan_spec_mismatch")
        cached = planned.preview.cached_result_hash
        if not planned.preview.exact_cache_hit or cached is None:
            return None
        manifest = self.pipeline.find_for_result(cached)
        if manifest is None:
            return None
        result = self.ledger.load_result(cached)
        return CompletedPortfolioResearch(
            result=PortfolioResearchResult.model_validate(
                result.model_copy(update={"action": "REUSED_EXACT"})
            ),
            pipeline_manifest=manifest,
        )

    def admit(
        self,
        *,
        spec: PortfolioResearchSpec,
        planned: PlannedPortfolioResearch | None = None,
    ) -> AdmittedPortfolioResearch:
        """Everything up to and including the queued task, and nothing after it.

        Deliberately cheap. It resolves *authorities* -- manifests, axes and
        receipts -- and compiles a Program only when neither the result nor the
        program is already known. The numerical resolution stays behind a thunk
        that admission never calls, so an admission costs no score replay and no
        Risk surface even on a full miss.
        """
        work = planned or self.plan(spec)
        if work.preview.spec_hash != spec.spec_hash:
            raise ValueError("portfolio_application.plan_spec_mismatch")
        # The plan already resolved these. Resolving a second time to compare a
        # receipt against itself is pure cost -- 29 seconds of it on the real
        # development root, twice per admission -- and it does not answer the
        # question it looks like it answers: whether the owners moved is decided
        # against the *numerical* resolution, and the executor checks the Program
        # against that before it walks anything.
        authorities = work.authorities
        if authorities.authorities_hash != work.preview.authorities_hash:
            raise ValueError("portfolio_application.plan_authorities_moved")

        # Numerical resolution is deferred. On an exact cache hit the executor
        # reopens the published result and the thunk is never called, so a reuse
        # costs no score replay and no Risk surface -- which is what "exact
        # reuse" has to mean if the word is doing any work.
        resolution: ResolvedPortfolioExecution | None = None

        def resolve_once() -> ResolvedPortfolioExecution:
            nonlocal resolution
            if resolution is None:
                resolution = self.resolver.resolve(workspace=self.workspace, spec=spec)
                authorities.require_resolution(resolution)
            return resolution

        cached = (
            None
            if work.preview.cached_result_hash is None
            else self.ledger.load_result(work.preview.cached_result_hash)
        )
        known = self.ledger.find_program_for(
            holdings_spec_hash=spec.holdings_spec_hash,
            authorities_hash=authorities.authorities_hash,
        )
        if cached is not None:
            program = self.ledger.load_program(cached.program_hash)
        elif known is not None:
            # A descendant control binds nothing the program does, so the same
            # holdings configuration over the same authorities is the same
            # program. Compiling it again would mean resolving to rediscover it.
            program = known
        else:
            # From authorities, not from a resolution: admission must not pay
            # for a score replay or a Risk surface to discover an axis the
            # authority receipt already states.
            program = self.compiler.compile_from_authorities(spec=spec, authorities=authorities)
        if program.authorities_hash != authorities.authorities_hash:
            raise ValueError("portfolio_application.program_authorities_mismatch")
        envelope, goal, task_plan = portfolio_research_task_contract(
            workspace_id=self.workspace_id,
            spec=spec,
            program=program,
            admission_hash=work.preview.admission_hash,
            authorities_hash=authorities.authorities_hash,
            workspace_manifest_hash=work.preview.workspace_manifest_hash,
            strategy_catalog_hash=self.resolver.strategy_catalog_hash,
            selected_strategy_package_id=authorities.strategy_package_id,
        )
        admission = self.session.task_control_registry.admit(
            input_envelope=envelope,
            goal=goal,
            plan=task_plan,
            observed_at=self.clock(),
        )
        return AdmittedPortfolioResearch(
            task_id=admission.record.task_id,
            spec=spec,
            planned=work,
            program=program,
            authorities=authorities,
            resolve_once=resolve_once,
            already_succeeded=(
                admission.record.lifecycle is not TaskLifecycle.RECOVERY_REQUIRED
                and admission.record.lifecycle is not TaskLifecycle.QUEUED
            ),
            lifecycle=admission.record.lifecycle.value,
        )

    def execute(
        self,
        admitted: AdmittedPortfolioResearch,
        *,
        expected_task_hash: str | None = None,
    ) -> CompletedPortfolioResearch:
        """Drive the admitted task to a published result. This is the work.

        A confirmed Task version travels to the registry's own start boundary
        (`restart_recovery` / `start_next`), which refuses a moved Task there.
        """
        spec = admitted.spec
        work = admitted.planned
        program = admitted.program
        authorities = admitted.authorities
        adapter = PortfolioResearchTaskAdapter(
            workspace_id=self.workspace_id,
            spec=spec,
            program=program,
            resolved=admitted.resolve_once,
            coverage=authorities.coverage,
            authorities_hash=authorities.authorities_hash,
            admission_hash=work.preview.admission_hash,
            workspace_manifest_hash=work.preview.workspace_manifest_hash,
            strategy_catalog_hash=self.resolver.strategy_catalog_hash,
            selected_strategy_package_id=authorities.strategy_package_id,
            executor=self.executor,
        )
        runner = TaskControlRunner(
            registry=self.session.task_control_registry,
            adapters={adapter.task_kind: adapter},
            runtime_path=str(self.session.runtime_path),
            clock=self.clock,
            stage_scope=storage_capacity_scope,
        )
        try:
            record = self.session.task_control_registry.task(admitted.task_id)
            if record.lifecycle is TaskLifecycle.RECOVERY_REQUIRED:
                record = runner.recover(admitted.task_id, expected_task_hash=expected_task_hash)
            elif record.lifecycle is TaskLifecycle.QUEUED:
                dispatched = runner.run_next(
                    expected_task_id=admitted.task_id, expected_task_hash=expected_task_hash
                )
                if dispatched is None:
                    raise ValueError("portfolio_application.task_not_dispatched")
                record = dispatched
        finally:
            runner.close()
        if record.lifecycle is not TaskLifecycle.SUCCEEDED:
            raise ValueError(f"portfolio_application.task_not_succeeded:{record.lifecycle.value}")
        result = self.ledger.find_result_for(
            program_hash=program.program_hash, spec_hash=spec.spec_hash
        )
        if result is None:
            raise ValueError("portfolio_application.task_result_absent")
        if work.preview.exact_cache_hit:
            result = PortfolioResearchResult.model_validate(
                result.model_copy(update={"action": "REUSED_EXACT"})
            )
        manifest = PortfolioResearchPipelineManifest.create(
            workspace_id=self.workspace_id,
            task_id=record.task_id,
            task_record_hash=record.record_hash,
            program_hash=program.program_hash,
            result_hash=result.result_hash,
            report_hash=result.report_hash,
            completed_at=self.clock(),
        )
        self.pipeline.publish(manifest)
        return CompletedPortfolioResearch(result=result, pipeline_manifest=manifest)

    def recover(
        self,
        *,
        task_id: UUID,
        expected_task_hash: str | None = None,
    ) -> CompletedPortfolioResearch:
        """Replan and require exact admitted workspace, strategy, program and task authority.

        Args:
            task_id: Exact durable task to recover.
            expected_task_hash: Optional optimistic task identity.

        Returns:
            Completed result from execution of the exact reopened admission.

        Raises:
            ValueError: Workspace/manifest/catalog/admission/authority/strategy or exact
                program/task binding differs.
        """
        task = self.session.task_control_registry.task(task_id)
        durable = portfolio_research_task_input(task)
        if durable.workspace_id != self.workspace_id:
            raise ValueError("portfolio_application.recovery_workspace_mismatch")
        if durable.workspace_manifest_hash != self.manifest_binding():
            raise ValueError("portfolio_application.recovery_workspace_manifest_mismatch")
        if durable.strategy_catalog_hash != self.resolver.strategy_catalog_hash:
            raise ValueError("portfolio_application.recovery_strategy_catalog_mismatch")
        spec = durable.spec
        planned = self.plan(spec)
        # Recovery continues the run that was admitted, so it must match the plan
        # that admitted it. A different preview -- or the same preview taken over
        # moved authorities -- is a different run wearing this task's id.
        if durable.admission_hash != planned.preview.admission_hash:
            raise ValueError("portfolio_application.recovery_admission_mismatch")
        if durable.program.authorities_hash != planned.authorities.authorities_hash:
            raise ValueError("portfolio_application.recovery_authorities_mismatch")
        if (
            durable.selected_strategy_package_id != planned.authorities.strategy_package_id
            or durable.selected_score_source_mode != planned.authorities.score_source_mode
            or durable.selected_strategy_package_hash != planned.authorities.strategy_package_hash
        ):
            raise ValueError("portfolio_application.recovery_strategy_selection_mismatch")
        admitted = self.admit(spec=spec, planned=planned)
        if (
            admitted.task_id != task_id
            or admitted.program.program_hash != durable.program.program_hash
        ):
            raise ValueError("portfolio_application.recovery_task_identity_mismatch")
        return self.execute(admitted, expected_task_hash=expected_task_hash)

    def report(self, result_hash: str) -> PortfolioDeclaredPathReport:
        """Reopen the exact declared-path report bound to one result.

        Args:
            result_hash: Exact stored research result identity.

        Returns:
            The declared-path report from durable result lineage.
        """
        result = self.ledger.load_result(result_hash)
        return self.ledger.load_report(result.report_hash)

    def originating_task(self, result_hash: str) -> UUID | None:
        """The task this result came out of, read from the durable manifest.

        A researcher holds a result hash and nothing else, so the report and the
        export have to be able to name their own run from that alone. `None` is
        truthful for a result published before the index existed, and is not an
        invitation to guess.
        """
        manifest = self.pipeline.find_for_result(result_hash)
        return None if manifest is None else manifest.task_id

    def export(self, result_hash: str) -> str:
        """The command that reproduces one result's request.

        A result published before V403 holds the command it was published with, which its
        identity covers; a later one's is composed from its originating Task's request.

        Args:
            result_hash: Exact stored research result identity.

        Returns:
            The command, every non-default control written out.

        Raises:
            ValueError: `portfolio_application.export_task_absent` when no Task's completion names
                the result; `portfolio_application.export_spec_mismatch` when that Task's request
                is not the result's.
        """
        result = self.ledger.load_result(result_hash)
        if result.export_command is not None:
            return result.export_command
        manifest = self.pipeline.find_for_result(result_hash)
        if manifest is None:
            raise ValueError("portfolio_application.export_task_absent")
        task = self.session.task_control_registry.task(manifest.task_id)
        spec = portfolio_research_task_input(task).spec
        if spec.spec_hash != result.spec_hash:
            raise ValueError("portfolio_application.export_spec_mismatch")
        return export_command(workspace_id=self.workspace_id, spec=spec)

    def program(self, result_hash: str) -> PortfolioExecutionProgram:
        """Reopen the exact execution program bound to one result.

        Args:
            result_hash: Exact stored research result identity.

        Returns:
            The execution program from durable result lineage.
        """
        return self.ledger.load_program(self.ledger.load_result(result_hash).program_hash)


__all__ = [
    "PUBLIC_REFUSALS",
    "AdmittedPortfolioResearch",
    "CompletedPortfolioResearch",
    "PlannedPortfolioResearch",
    "PortfolioPlanPreview",
    "PortfolioResearchApplication",
]
