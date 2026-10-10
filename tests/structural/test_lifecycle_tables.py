"""Every set of Task lifecycle states in src is registered with its reason (V524, S2).

A consumer that sorts Task states decides each state on purpose. The sorts are: in flight, a
waiter's exit, finished, owed, resumable. A state the Task never leaves by itself, put on the
wrong side, kept a confirm in flight (V506), a waiter past a deferral (V507) and a clock running
through a wait (V520). The scan finds every set of two or more states in src, named or written
inline in a comparison. Each is pinned here by its module, scope and members, with the reason
for what it holds and what it leaves out. A new or changed set fails until it is registered.

Who can reach a state is pinned too:
- only the preparation and data update owners defer a Task, and the research update through
  the data update's stages (V601);
- no Task Control transition sets REVIEW_PENDING, which the client's tables keep for answers'
  `status`;
- every kind that can defer reads its deferral in STATUS.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
STATES = frozenset(
    {
        "QUEUED",
        "RUNNING",
        "DEFERRED",
        "REVIEW_PENDING",
        "CANCEL_REQUESTED",
        "CANCELLED",
        "SUCCEEDED",
        "BLOCKED",
        "RECOVERY_REQUIRED",
    }
)

RESUMABLE = "resumable: a queued or interrupted Task is (re)driven; every other state is not"
FINISHED = "finished: its readback answers for it; every other state is not done"
ENDED = "ended: nothing more happens to it; a waiting state may still move"
EXECUTING = "executing now; a waiting or ended Task does no work"
STOPPED = "stopped and owed a way on; a deferral has its own branch (V507)"

REGISTER: dict[tuple[str, str, str], str] = {
    (
        "control/product_host/composition/model_sandbox.py",
        "_TERMINAL",
        "BLOCKED,CANCELLED,RECOVERY_REQUIRED,SUCCEEDED",
    ): "the sandbox's existing wait ends on success or a stopped/interrupted attempt; "
    "queued, running, cancelling and deferred states are not a finished sandbox attempt",
    ("control/data_platform/task_telemetry.py", "EXECUTING", "CANCEL_REQUESTED,RUNNING"): EXECUTING,
    (
        "control/product_host/composition/application_session.py",
        "WorkspaceApplicationSession.execute_admitted",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "control/product_host/composition/decision_advancement.py",
        "DecisionAdvancementApplication._plan",
        "DEFERRED,QUEUED,RECOVERY_REQUIRED,RUNNING",
    ): "a pending advancement of the strategy answers its own plan whatever target is asked, a "
    "deferred one too, whose run resumes it once due (V601, V604); a cancelling one is not pending",
    (
        "control/product_host/composition/decision_advancement.py",
        "DecisionAdvancementApplication.in_flight",
        "CANCEL_REQUESTED,QUEUED,RECOVERY_REQUIRED,RUNNING",
    ): "every state the Task leaves by itself answers the admitted Task; a deferral is not in "
    "flight: its rerun resumes it once due and is refused before then (V601)",
    (
        "control/product_host/composition/evidence_review_application.py",
        "EvidenceReviewApplication._active_refresh_task.active",
        "QUEUED,RECOVERY_REQUIRED,RUNNING",
    ): "an evidence refresh still owed; evidence Tasks never defer, and a cancelling one is not",
    (
        "control/product_host/composition/evidence_review_application.py",
        "EvidenceReviewApplication.recovery_commands",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "control/product_host/composition/evidence_review_projection.py",
        "EvidenceCroProjector._coverage_progress",
        "CANCEL_REQUESTED,RUNNING",
    ): EXECUTING,
    (
        "control/product_host/composition/feature_trials.py",
        "_TERMINAL",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/composition/resource_estimates.py",
        "_HOLDING_MEMORY",
        "CANCEL_REQUESTED,RUNNING",
    ): "Tasks still executing hold the memory a refused heavy run waits for (PERF-1)",
    (
        "control/product_host/composition/goal_check.py",
        "MOVING",
        "CANCEL_REQUESTED,QUEUED,RUNNING",
    ): "a moving Task must end before its goal completes (OP13)",
    (
        "control/product_host/composition/goal_check.py",
        "WAITING",
        "BLOCKED,DEFERRED,RECOVERY_REQUIRED,REVIEW_PENDING",
    ): "a Task waiting on someone or on time is named in the submission's problems",
    (
        "control/product_host/composition/local_web_session.py",
        "LocalPortfolioWebSession.resume_refusal",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "control/product_host/composition/local_web_session.py",
        "LocalPortfolioWebSession._recovery_commands",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "PortfolioRunCommand.execute",
        "CANCELLED,CANCEL_REQUESTED",
    ): "cancelled or cancelling: the run stops at its boundary",
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "PortfolioResearchOperations._tasks_wait",
        "DEFERRED,QUEUED,RUNNING",
    ): "a busy queue puts the verification sweep and the held-state backup off, a deferral "
    "holding the running place too, which they would only queue behind (V604)",
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "PortfolioResearchOperations._workspace_operation",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "PortfolioResearchOperations._stopped_answer",
        "BLOCKED,RECOVERY_REQUIRED",
    ): STOPPED,
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "PortfolioResearchOperations._resume_refusal",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "PortfolioResearchOperations._task_replan",
        "BLOCKED,CANCELLED,RECOVERY_REQUIRED",
    ): "a distinct replan carries the exact blocked, cancelled or interrupted source id and "
    "record hash (P3a); moving, deferred, review and succeeded states gain no stopped-source "
    "provenance",
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "PortfolioResearchOperations._recovery_context",
        "BLOCKED,CANCELLED,RECOVERY_REQUIRED",
    ): "a recovery preview or its offered admission requires the current blocked, cancelled "
    "or interrupted source version (P3a); every other lifecycle refuses that provenance",
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "_LIVE_LIFECYCLES",
        "CANCEL_REQUESTED,DEFERRED,QUEUED,RUNNING",
    ): "a status long poll waits for these to move; a deferral moves when its resume is sent",
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "_PREPARATION_IN_FLIGHT",
        "CANCEL_REQUESTED,QUEUED,RUNNING",
    ): "a confirm answers a preparation on its way; a deferred one goes to its owner (V506)",
    (
        "control/product_host/composition/portfolio_research_operations.py",
        "_FINISHED_LIFECYCLES",
        "CANCELLED,SUCCEEDED",
    ): FINISHED,
    (
        "control/product_host/composition/portfolio_updates.py",
        "PortfolioUpdateApplication.admit",
        "QUEUED,RECOVERY_REQUIRED,RUNNING",
    ): "another update in progress refuses a second; updates never defer, and a cancelling one "
    "runs no further step",
    (
        "control/product_host/composition/research_experiments.py",
        "ResearchExperimentApplication._existing_task_answer",
        "QUEUED,RECOVERY_REQUIRED,RUNNING",
    ): "in flight answers REUSED_IN_FLIGHT; any other state answers as itself",
    (
        "control/product_host/composition/result_standing.py",
        "_RUNNING",
        "CANCEL_REQUESTED,DEFERRED,QUEUED,RUNNING",
    ): "a result's Task still on its way; studies never defer",
    (
        "control/product_host/composition/task_recovery.py",
        "_TERMINAL",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/composition/task_recovery.py",
        "_liveness",
        "CANCEL_REQUESTED,RUNNING",
    ): EXECUTING,
    (
        "control/product_host/composition/task_recovery.py",
        "_health_and_incidents",
        "BLOCKED,CANCELLED",
    ): "terminal and blocked: the Guardian's TERMINAL_BLOCKED",
    (
        "control/product_host/composition/task_recovery.py",
        "_health_and_incidents",
        "DEFERRED,RECOVERY_REQUIRED,REVIEW_PENDING",
    ): "waiting on someone or on time: the Guardian's TERMINAL_DEFERRED",
    (
        "control/product_host/composition/task_recovery.py",
        "task_attention",
        "BLOCKED,CANCELLED,RECOVERY_REQUIRED",
    ): "blocked or interrupted Tasks owe recovery attention until an exact-version successful "
    "successor or resolved incident proves resolution (P3a), and a ledger-rebuilt one, which "
    "cannot resume, closes as UNRECOVERABLE; a cancelled Task owes none but folds under an "
    "exact-version successor that succeeded (BADGE: unrecoverable stops close, resolved stops "
    "fold under their successor); deferral waits on time, and other states owe no stopped "
    "attention; this owns the former pending_decisions._STOPPED classification",
    (
        "control/product_host/composition/task_supervision.py",
        "_FINISHED",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/composition/task_supervision.py",
        "TaskSupervisor.supervise_once",
        "BLOCKED,RECOVERY_REQUIRED",
    ): "only blocked or interrupted views may suppress reopening a finding through resolved "
    "attention bound to that exact Task hash (P3a); queued and running liveness remain "
    "classified, while other states gain no resolution shortcut",
    (
        "control/product_host/composition/upgrade_overview.py",
        "upgrade_overview",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "control/product_host/composition/verification_sweep.py",
        "StudyVerificationSweep.due",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/data_preparation/application.py",
        "WorkspacePreparationApplication.plan",
        "CANCELLED,SUCCEEDED",
    ): "an unfinished preparation is one not succeeded or cancelled",
    (
        "control/product_host/data_preparation/application.py",
        "WorkspacePreparationApplication.plan",
        "BLOCKED,CANCELLED",
    ): "a stopped preparation stays the named predecessor of its recovery plan",
    (
        "control/product_host/data_preparation/application.py",
        "WorkspacePreparationApplication.confirm",
        "BLOCKED,DEFERRED",
    ): "the retry path: refused before the retry time, resumed after it (V375, V506)",
    (
        "control/product_host/data_preparation/application.py",
        "WorkspacePreparationApplication.confirm",
        "CANCELLED,SUCCEEDED",
    ): FINISHED,
    (
        "control/product_host/data_preparation/application.py",
        "WorkspacePreparationApplication._predecessor",
        "BLOCKED,CANCELLED",
    ): "a predecessor the next preparation may continue from",
    (
        "control/product_host/data_preparation/application.py",
        "WorkspacePreparationApplication._listing_activity",
        "CANCEL_REQUESTED,RUNNING",
    ): EXECUTING,
    (
        "control/product_host/data_preparation/feature_research.py",
        "ResearchFeatureBuildApplication._submit",
        "BLOCKED,RECOVERY_REQUIRED",
    ): "a stopped build is answered, not submitted again",
    (
        "control/product_host/data_preparation/feature_research.py",
        "ResearchFeatureBuildApplication.admit",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/data_preparation/input_capture.py",
        "ResearchInputCaptureApplication.plan",
        "CANCELLED,SUCCEEDED",
    ): FINISHED,
    (
        "control/product_host/data_preparation/input_capture.py",
        "ResearchInputCaptureApplication.admit",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/data_preparation/model_training.py",
        "ModelTrainingInputApplication.admit",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/data_preparation/remediation.py",
        "WorkspaceDataIssueApplication.readback",
        "BLOCKED,DEFERRED,REVIEW_PENDING",
    ): "a preparation or update waiting on a data decision or on time, which the data issues "
    "continue",
    (
        "control/product_host/data_preparation/research_strategy.py",
        "ResearchStrategyPreparation.prepare",
        "BLOCKED,RECOVERY_REQUIRED",
    ): "a stopped preparation is answered, not prepared again",
    (
        "control/product_host/data_preparation/research_strategy.py",
        "ResearchStrategyPreparation.admit",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/data_preparation/research_strategy.py",
        "ResearchStrategyPreparation.readback",
        "BLOCKED,CANCELLED",
    ): "an ended unsuccessful preparation offers its verified durable declaration for a "
    "new preview (V615); a succeeded one is read back and an interrupted one is recoverable",
    (
        "control/product_host/maintenance/data_update.py",
        "bind_existing_data_workspace",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/maintenance/data_update.py",
        "WorkspaceDataUpdateApplication.confirm.approve",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "control/product_host/maintenance/data_update.py",
        "WorkspaceDataUpdateApplication._supplemented",
        "CANCELLED,SUCCEEDED",
    ): "an audit's confirmation joins the membership change it was admitted onto while that "
    "change has not succeeded or been cancelled; a stopped one still takes it on its resume",
    (
        "control/product_host/maintenance/data_update.py",
        "WorkspaceDataUpdateApplication._waiting",
        "DEFERRED,QUEUED,RECOVERY_REQUIRED,RUNNING",
    ): "a data update that has not ended answers its own plan, a deferred one too, whose run "
    "resumes it once due; a cancelling one is not waiting (V604)",
    (
        "control/product_host/storage/evidence_references.py",
        "_IN_FLIGHT",
        "CANCEL_REQUESTED,DEFERRED,QUEUED,RECOVERY_REQUIRED,REVIEW_PENDING,RUNNING",
    ): "every unended state keeps the evidence it references",
    (
        "control/product_host/storage/input_references.py",
        "ResearchInputStorage._references",
        "CANCELLED,SUCCEEDED",
    ): FINISHED,
    ("control/task_control/registry.py", "_TERMINAL_TASKS", "BLOCKED,CANCELLED,SUCCEEDED"): ENDED,
    (
        "control/task_control/registry.py",
        "_RECOVERY_LINK_STOPPED",
        "BLOCKED,CANCELLED,RECOVERY_REQUIRED",
    ): "a new recovery preview requires the current blocked, cancelled or interrupted source "
    "version (P3a); moving, deferred, review and succeeded sources cannot create it, while "
    "confirmed successors retain the prior preview's source hash",
    (
        "control/task_control/registry.py",
        "_ACTIVE_TASKS",
        "CANCEL_REQUESTED,DEFERRED,RECOVERY_REQUIRED,REVIEW_PENDING,RUNNING",
    ): "started and unended: the one active Task; a queued one is not started",
    (
        "control/task_control/registry.py",
        "DuckDbTaskControlRegistry.request_cancel.operation",
        "CANCELLED,CANCEL_REQUESTED",
    ): "a cancel already asked for or done answers as it stands",
    (
        "control/task_control/registry.py",
        "DuckDbTaskControlRegistry.request_cancel.operation",
        "BLOCKED,QUEUED",
    ): "work that never started, or a ledger-rebuilt stop that cannot resume, cancels at once and "
    "keeps the stop's failure code (BADGE: unrecoverable stops close); running work is asked to "
    "cancel",
    (
        "control/task_control/registry.py",
        "DuckDbTaskControlRegistry.reconcile_after_writer_acquisition.operation",
        "CANCEL_REQUESTED,RUNNING",
    ): EXECUTING,
    (
        "control/task_control/runner.py",
        "TaskControlRunner.stale_active_tasks",
        "CANCEL_REQUESTED,DEFERRED,RECOVERY_REQUIRED,REVIEW_PENDING,RUNNING",
    ): "started and unended, whose liveness the runner judges",
    ("control/task_control/runner.py", "TaskControlRunner._step", "CANCEL_REQUESTED,RUNNING"): (
        "the runner steps on while executing"
    ),
    (
        "control/task_control/timing.py",
        "_ENDED",
        "BLOCKED,CANCELLED,DEFERRED,RECOVERY_REQUIRED,REVIEW_PENDING,SUCCEEDED",
    ): "the clock stops: ended, or waiting on someone or on time (V520)",
    (
        "foundation/feature_engine/producers/reference_data.py",
        "SectorRefreshStager._load",
        "BLOCKED,DEFERRED,RUNNING",
    ): "the sector staging file's own statuses, not a Task's",
    (
        "interface/local_application/cli_contract.py",
        "_PENDING_STATES",
        "CANCEL_REQUESTED,DEFERRED,QUEUED,RECOVERY_REQUIRED,REVIEW_PENDING,RUNNING",
    ): "pending: the outcome's exit code 3; a waiter ends on WAIT_EXITS among them",
    ("interface/local_application/cli_contract.py", "_REFUSED_STATES", "BLOCKED,CANCELLED"): (
        "refused: the outcome's exit code 2"
    ),
    (
        "interface/local_application/cli_contract.py",
        "ACTION_STATES",
        "RECOVERY_REQUIRED,REVIEW_PENDING",
    ): "waiting on a request: a review to decide (an answer's status), a Task to recover",
    (
        "interface/local_application/dispatcher.py",
        "_SETTLED_LIFECYCLES",
        "BLOCKED,CANCELLED,SUCCEEDED",
    ): ENDED,
    (
        "interface/local_application/dispatcher.py",
        "LocalBackgroundDispatcher.resume",
        "QUEUED,RECOVERY_REQUIRED",
    ): RESUMABLE,
    (
        "interface/local_application/dispatcher.py",
        "LocalBackgroundDispatcher.submit",
        "BLOCKED,CANCELLED",
    ): "found stopped or cancelled when admitted: its command has nothing to run, so it is "
    "never read as running while that command waits its turn; a succeeded one still is, its "
    "command publishing (V600)",
    (
        "interface/local_application/dispatcher.py",
        "LocalBackgroundDispatcher._waits_its_turn",
        "QUEUED,RECOVERY_REQUIRED",
    ): "a command that returned before its Task's turn came is kept and driven again once the "
    "running place is free; a Task in any other state has nothing left for it (V604)",
    (
        "control/product_host/composition/research_update_automation.py",
        "_MOVING",
        "CANCEL_REQUESTED,QUEUED,RECOVERY_REQUIRED,RUNNING",
    ): "an update the automation attends still on its way: read again at the next idle; a "
    "deferral waits on its retry time instead (V604)",
    (
        "control/product_host/composition/research_update_automation.py",
        "_STOPPED",
        "BLOCKED,CANCELLED",
    ): "an attended update that stopped: its words name the way on, and no plan follows at once "
    "(V604)",
}


def _members(node: ast.AST) -> frozenset[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            if sub.value.id == "TaskLifecycle" and sub.attr in STATES:
                found.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value in STATES:
            found.add(sub.value)
    return frozenset(found)


def _literal(node: ast.AST) -> bool:
    return isinstance(node, (ast.Set, ast.Tuple, ast.List)) or (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set"}
    )


def _sets() -> set[tuple[str, str, str]]:
    """Each set of two or more lifecycle states in src: its module, scope and members."""
    found: set[tuple[str, str, str]] = set()
    for path in sorted(ROOT.rglob("*.py")):
        module = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(node: ast.AST, scope: str, module: str = module) -> None:
            for child in ast.iter_child_nodes(node):
                inner = scope
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    inner = f"{scope}.{child.name}" if scope else child.name
                if isinstance(child, (ast.Assign, ast.AnnAssign)) and child.value is not None:
                    held = _members(child.value)
                    if _literal(child.value) and len(held) >= 2:
                        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                        name = ",".join(ast.unparse(target) for target in targets)
                        where = f"{scope}.{name}" if scope else name
                        found.add((module, where, ",".join(sorted(held))))
                        continue
                if isinstance(child, ast.Compare) and any(
                    isinstance(op, (ast.In, ast.NotIn)) for op in child.ops
                ):
                    for compared in child.comparators:
                        held = _members(compared)
                        if _literal(compared) and len(held) >= 2:
                            found.add((module, scope or "<module>", ",".join(sorted(held))))
                visit(child, inner)

        visit(tree, "")
    return found


def test_every_lifecycle_set_is_registered_with_its_reason() -> None:
    """requirement (V524, S2): each set of Task states in src is registered with what it holds
    and leaves out, so a new or changed one is classified on purpose before it lands."""

    found = _sets()
    unregistered = sorted(found - set(REGISTER))
    gone = sorted(set(REGISTER) - found)
    assert not unregistered, f"register each with its reason: {unregistered}"
    assert not gone, f"drop the register's entries no longer in src: {gone}"
    assert all(reason.strip() for reason in REGISTER.values())


def test_only_the_data_owners_defer_and_no_transition_sets_review_pending() -> None:
    """Only the data owners defer and no transition sets review pending."""

    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )

    deferring = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if "StageDisposition.DEFERRED" in path.read_text(encoding="utf-8")
    )
    assert deferring == [
        "control/product_host/composition/decision_advancement.py",  # its data stage's (V601)
        "control/product_host/data_preparation/application.py",
        "control/product_host/maintenance/data_update.py",
        "control/task_control/runner.py",  # maps the disposition to the Task's lifecycle
    ], deferring
    registry = (ROOT / "control/task_control/registry.py").read_text(encoding="utf-8")
    assert "lifecycle=TaskLifecycle.REVIEW_PENDING" not in registry
    reader = inspect.getsource(PortfolioResearchOperations._deferred_way)
    assert "PREPARATION_TASK_KIND" in reader and "DATA_UPDATE_TASK_KIND" in reader
    assert "DecisionAdvancementApplication.task_kind" in reader
