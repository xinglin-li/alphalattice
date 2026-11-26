"""Every saved study verified again in full: the sweep and the verify-all request.

A study read may answer from its last full verification while its files keep their path, size,
modification time and file identity (decision 5). A change that keeps all four is caught by the
next export, publication or admission, and by this sweep, which verifies every saved study's
sealed evidence in full in the Host's worker process, so the Host's reads do not wait on it.

It is a Task, the workspace's lowest priority: admitted when the Host goes idle after an
upgrade or a week, it stops as soon as another Task waits and the next sweep takes up the rest;
a person or an agent asks for a whole one at once (`EXPERIMENT_VERIFY_ALL`). A study whose
evidence does not verify is named in the sweep's report with its refusal, its kept
verifications are dropped so its next read verifies it in full and refuses by name, and the
upgrade overview lists it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta
from threading import Lock
from typing import Literal, cast
from uuid import UUID

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_experiments import (
    TASK_KIND as STUDY_TASK_KIND,
)
from alphalattice.control.product_host.composition.research_experiments import (
    VERIFICATION_LEDGER_DIRECTORY,
    ResearchExperimentApplication,
)
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
)
from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.kernel.shared_kernel.identity import canonical_hash

SWEEP_TASK_KIND = "study_verification_sweep"
SWEEP_STAGE = "verify_saved_studies"
SWEEP_INTERVAL = timedelta(days=7)
"""How long after a whole sweep the next one is due, unless an upgrade makes it due sooner."""
REPORTS_DIRECTORY = "sweeps"
"""Under the verification ledger: each sweep's report, named by its content hash."""

Scope = Literal["ALL", "DUE"]


def _task_contract(
    scope: Scope, installed: str, requested_at: datetime
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    envelope = TaskInputEnvelope.create(
        task_kind=SWEEP_TASK_KIND,
        input_schema_id="study-verification-sweep",
        payload={"scope": scope, "installed": installed, "requested_at": requested_at.isoformat()},
    )
    goal = ResearchGoal.create(
        goal_kind="VERIFY_SAVED_STUDIES",
        input_hash=envelope.input_hash,
        deliverable_kind="StudyVerificationSweepReport",
        summary="Verify every saved study's sealed evidence in full; compute nothing.",
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash((SWEEP_STAGE,)),
        verifier_catalog_hash=canonical_hash((f"{SWEEP_TASK_KIND}.{SWEEP_STAGE}",)),
        work_items=(
            WorkItemDefinition.create(
                stage_id=SWEEP_STAGE,
                dependency_ids=(),
                verifier_id=f"{SWEEP_TASK_KIND}.{SWEEP_STAGE}",
            ),
        ),
    )
    return envelope, goal, plan


class StudyVerificationSweep:
    """The sweep's Task: its admission, its one stage and the report it seals."""

    task_kind = SWEEP_TASK_KIND
    replans = (
        TaskReplan(
            task_kind=SWEEP_TASK_KIND,
            preview="EXPERIMENT_VERIFY_ALL",
            admitting="EXPERIMENT_VERIFY_ALL",
        ),
    )
    """The recovery view asks for a sweep again without a separate plan."""

    def __init__(
        self,
        *,
        session: WorkspaceApplicationSession,
        experiments: ResearchExperimentApplication,
        installed: Callable[[], str],
        clock: Callable[[], datetime],
    ) -> None:
        """Bind the sweep to the session's Tasks, the studies' verifier and the installed code."""
        self.session, self.experiments, self.clock = session, experiments, clock
        self.installed = installed
        # Whether a sweep is due and its admission are one step: an idle hook and a start may ask
        # at once, and only one of them admits it.
        self.admission = Lock()
        self.root = (
            session.workspace / "runtime" / VERIFICATION_LEDGER_DIRECTORY / REPORTS_DIRECTORY
        )

    # ------------------------------------------------------------------ admission

    def admit(self, scope: Scope) -> CommandAdmission:
        """Admit one sweep, whole (`ALL`) or taking up the last one where it stopped (`DUE`)."""
        envelope, goal, plan = _task_contract(scope, self.installed(), self.clock())
        admission = self.session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=self.clock()
        )
        return CommandAdmission(
            task_id=admission.record.task_id, lifecycle=admission.record.lifecycle.value
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Run or recover an admitted sweep through Task Control's runner."""
        task = self.session.task_control_registry.task(task_id)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def due(self) -> bool:
        """Whether an idle Host admits a sweep.

        It is due a week after the last one, at once after an upgrade (the installed study
        code moved since it ran) or when it stopped for waiting work; before the first, a week
        after the oldest saved study; never while one is admitted and not ended.
        """
        if any(
            task.task_kind == SWEEP_TASK_KIND
            and task.lifecycle
            not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
            for task in self.session.task_control_registry.tasks()
        ):
            return False
        studies = self.saved_studies()
        if not studies:
            return False
        last = self.last_report()
        if last is None:
            oldest: datetime = min(study.updated_at for study in studies)
            return self.clock() - oldest >= SWEEP_INTERVAL
        return (
            last["installed"] != self.installed()
            or bool(last["remaining"])
            or self.clock() - datetime.fromisoformat(str(last["finished_at"])) >= SWEEP_INTERVAL
        )

    def saved_studies(self) -> tuple[TaskRecord, ...]:
        """The workspace's saved studies: its succeeded study Tasks."""
        return tuple(
            task
            for task in self.session.task_control_registry.tasks()
            if task.task_kind == STUDY_TASK_KIND and task.lifecycle is TaskLifecycle.SUCCEEDED
        )

    # ------------------------------------------------------------------ the Task

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """A sweep resumes only under the study code it was admitted to verify with."""
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash({"study_verification_sweep": 1}),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=canonical_hash({"interval_days": SWEEP_INTERVAL.days}),
            framework_identity_hash=self.session.execution_identity(self.installed()),
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Verify each saved study in full until another Task waits; seal the report."""
        del execution, work_item
        started_at = self.clock()
        scope = str(task.input.payload["scope"])
        installed = self.installed()
        # A sweep that stopped for waiting work is taken up where it stopped (DUE), keeping what
        # it verified and the refusals it named.
        last = self.last_report() if scope == "DUE" else None
        verified: list[str] = []
        failed: list[dict[str, str]] = []
        if last is not None and last["installed"] == installed and last["remaining"]:
            verified = [str(value) for value in cast(list[object], last["verified"])]
            failed = [dict(row) for row in cast(list[dict[str, str]], last["failed"])]
        done = {*verified, *(row["task_id"] for row in failed)}
        studies = [study for study in self.saved_studies() if str(study.task_id) not in done]
        remaining = 0
        for index, study in enumerate(studies):
            if self.work_waits():
                remaining = len(studies) - index
                break
            try:
                self.experiments.verify_saved_study(study)
            except (ValueError, KeyError) as error:
                code = public_failure(error, "study_verification_sweep.study_unverified")
                failed.append({"task_id": str(study.task_id), "failure_code": code})
                self.experiments.forget_verification(study.task_id)
                continue
            verified.append(str(study.task_id))
        report = {
            "schema_version": 1,
            "task_id": str(task.task_id),
            "scope": scope,
            "installed": installed,
            "started_at": started_at.isoformat(),
            "finished_at": self.clock().isoformat(),
            "verified": sorted(set(verified)),
            "failed": failed,
            "remaining": remaining,
        }
        digest = self._seal(report)
        return StageExecutionResult(
            disposition=StageDisposition.READY,
            evidence=(
                TaskEvidence(
                    evidence_kind=f"{SWEEP_TASK_KIND}.report",
                    reference=f"playpen://study-verification-sweep/{digest}",
                    content_hash=digest,
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
        """The sealed report reads back under its content hash."""
        del task, execution, work_item
        (item,) = evidence
        text = (self.root / f"{item.content_hash}.json").read_text(encoding="utf-8")
        if canonical_hash(json.loads(text)) != item.content_hash:
            raise ValueError("study_verification_sweep.report_tampered")
        return evidence

    # ------------------------------------------------------------------ reports

    def last_report(self) -> dict[str, object] | None:
        """The newest sealed report that reads as sealed; a damaged one is passed over."""
        if not self.root.is_dir():
            return None
        newest: dict[str, object] | None = None
        for path in self.root.glob("*.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if canonical_hash(report) != path.stem:
                continue
            if newest is None or str(report["finished_at"]) > str(newest["finished_at"]):
                newest = report
        return newest

    def failed_studies(self) -> dict[str, str]:
        """The studies the newest sweep could not verify, with their refusals."""
        last = self.last_report()
        rows = [] if last is None else last["failed"]
        assert isinstance(rows, list)
        return {str(row["task_id"]): str(row["failure_code"]) for row in rows}

    def _seal(self, report: dict[str, object]) -> str:
        digest = canonical_hash(report)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{digest}.json"
        staged = path.with_name(f"{path.name}.partial")
        staged.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        os.replace(staged, path)
        return str(digest)

    def work_waits(self) -> bool:
        """Whether another Task waits for the running place the sweep holds."""
        return any(
            task.lifecycle is TaskLifecycle.QUEUED
            for task in self.session.task_control_registry.tasks()
        )


class StudyVerificationSweepCommand:
    """One sweep, as the Host's dispatcher runs it: whole on request, due when idle."""

    command_kind = SWEEP_TASK_KIND

    def __init__(self, sweep: StudyVerificationSweep, scope: Scope = "DUE") -> None:
        """One sweep of the given scope; a recovery rebuilds it with the default."""
        self.sweep, self.scope = sweep, scope

    def admit(self) -> CommandAdmission:
        """Admit the sweep's Task."""
        return self.sweep.admit(self.scope)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Run or recover the sweep's Task."""
        self.sweep.execute(task_id, expected_task_hash=expected_task_hash)
