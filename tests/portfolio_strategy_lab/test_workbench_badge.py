"""BADGE: durable Task records through Local Web's APIs and the built Workbench."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.verification_sweep import SWEEP_TASK_KIND
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskEvidence,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskStageReceipt,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from tests.portfolio_strategy_lab.local_web_support import _json, _run_badge_browser
from tests.workspace_task_runner.task_control_support import digest, task_contract

pytestmark = pytest.mark.usefixtures("workbench_build")


def test_real_task_records_drive_home_badges_and_grouped_history(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """One producer-to-consumer regression: canonical equal-clock admissions drive Home's
    person decision count and Tasks/History's disclosed earlier stop; closing preserves the reason.

    These are labelled QA Task Control checkpoints of the registered verification kind.
    They execute no scientific adapter and publish no scientific result.
    """
    assert live.session is not None
    registry = live.session.task_control_registry
    now = live.clock()

    def record(salt: str, failure_code: str | None) -> TaskRecord:
        _input, _goal, template = task_contract(salt=salt)
        envelope = TaskInputEnvelope.create(
            task_kind=SWEEP_TASK_KIND,
            input_schema_id="study-verification-sweep",
            payload={"qa_fixture": salt},
        )
        goal = ResearchGoal.create(
            goal_kind="VERIFY_SAVED_STUDIES",
            input_hash=envelope.input_hash,
            deliverable_kind="StudyVerificationSweepReport",
            summary="QA fixture: preserve Task Control checkpoints without scientific work.",
        )
        plan = ResearchPlan.create(
            goal_hash=goal.goal_hash,
            workflow_definition_hash=template.workflow_definition_hash,
            verifier_catalog_hash=template.verifier_catalog_hash,
            work_items=template.work_items,
        )
        admitted = registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=now
        ).record
        started = registry.start_next(
            compatibility=TaskExecutionCompatibility.create(
                task_contract_hash=digest("qa-badge-contract"),
                workflow_definition_hash=plan.workflow_definition_hash,
                input_schema_id=envelope.input_schema_id,
                domain_policy_hash=digest("qa-badge-policy"),
                framework_identity_hash=digest("qa-badge-framework"),
            ),
            worker_instance_id=uuid4(),
            expected_task_id=admitted.task_id,
            observed_at=now,
        )
        assert started is not None
        task, execution = started
        for definition in plan.work_items:
            item = registry.begin_work_item(
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id=definition.stage_id,
                observed_at=now,
            )
            if failure_code is not None:
                task = registry.block_work_item(
                    task_id=task.task_id,
                    execution_id=execution.execution_id,
                    stage_id=definition.stage_id,
                    failure_code=failure_code,
                    observed_at=now,
                )
                break
            evidence = tuple(
                TaskEvidence(
                    evidence_kind=kind,
                    reference=f"qa://badge/{salt}/{kind}",
                    content_hash=digest(f"badge:{salt}:{kind}"),
                )
                for kind in definition.required_evidence_kinds
            )
            registry.mark_ready(
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id=definition.stage_id,
                evidence=evidence,
                observed_at=now,
            )
            task = registry.verify_work_item(
                TaskStageReceipt.from_identity(
                    receipt_id=uuid4(),
                    task_id=task.task_id,
                    execution_id=execution.execution_id,
                    stage_id=definition.stage_id,
                    work_item_definition_hash=item.definition_hash,
                    verifier_id=definition.verifier_id,
                    evidence=evidence,
                    status="VERIFIED",
                    failure_code=None,
                    observed_at=now,
                )
            )
        # The runner publishes this public readout at a checkpoint. TASKS reads
        # persisted projections, so the Registry fixture must publish it as well.
        projection = registry.safe_projection(task.task_id)
        assert projection.task_record_hash == task.record_hash
        return task

    rebuilt = record("badge-unrecoverable", "task_control.ledger_rebuilt")
    first = _json(live, "/api/decisions")
    assert first["counts"].get("STOPPED_TASK", 0) == 0
    peer = record("badge-normal-stop", "study_verification_sweep.not_admitted")
    source = record("badge-exact-plan", "study_verification_sweep.not_admitted")
    successor = record("badge-exact-plan", None)
    assert source.admitted_at == successor.admitted_at
    assert source.plan.plan_hash == successor.plan.plan_hash
    assert successor.lifecycle is TaskLifecycle.SUCCEEDED
    listing = _json(live, "/api/tasks")
    tasks = {row["task_id"]: row for row in listing["tasks"]}
    expected_ids = {str(row.task_id) for row in (rebuilt, peer, source, successor)}
    assert expected_ids <= tasks.keys(), {
        "expected_ids": sorted(expected_ids),
        "task_ids": sorted(tasks),
        "refusals": [
            (row.get("task_id"), row.get("failure_code")) for row in listing.get("refusals", [])
        ],
    }
    assert tasks[str(rebuilt.task_id)]["attention"]["resolution"] == "UNRECOVERABLE"
    assert tasks[str(source.task_id)]["attention"]["successor_task_hash"] == successor.record_hash
    decisions = _json(live, "/api/decisions")
    stopped = [row for row in decisions["decisions"] if row["kind"] == "STOPPED_TASK"]
    assert [row["task_id"] for row in stopped] == [str(peer.task_id)]
    assert stopped[0]["waits_on"] == "AGENT"
    assert decisions["counts"]["STOPPED_TASK"] == 1
    assert not [row for row in decisions["decisions"] if row["waits_on"] == "PERSON"]
    history = {row["task_id"]: row for row in _json(live, "/api/research-history")["entries"]}
    assert {str(row.task_id) for row in (rebuilt, peer, source, successor)} <= history.keys()
    assert history[str(rebuilt.task_id)]["failure_code"] == "task_control.ledger_rebuilt"
    assert history[str(source.task_id)]["status"] == "BLOCKED"
    assert history[str(successor.task_id)]["status"] == "SUCCEEDED"
    context = _json(live, "/api/session?context=1")["research_context"]["tasks"]["tasks"]
    context_versions = {row["task_id"]: row["task_record_hash"] for row in context}
    assert context_versions == {
        str(row.task_id): row.record_hash for row in (rebuilt, peer, source, successor)
    }
    _run_badge_browser(
        live,
        tmp_path / "badge-browser",
        "badge",
        {
            "rebuilt": str(rebuilt.task_id),
            "rebuilt_hash": rebuilt.record_hash,
            "peer": str(peer.task_id),
            "peer_hash": peer.record_hash,
            "source": str(source.task_id),
            "successor": str(successor.task_id),
            "successor_hash": successor.record_hash,
        },
    )
    cancelled = registry.task(rebuilt.task_id)
    assert cancelled.lifecycle is TaskLifecycle.CANCELLED
    assert cancelled.failure_code == "task_control.ledger_rebuilt"
    assert registry.task(source.task_id) == source
    assert registry.task(successor.task_id) == successor
    agent_stop = record("badge-agent-unrecoverable", "task_control.ledger_rebuilt")
    assert live.operations is not None
    agent_closed = live.operations.execute(
        PortfolioResearchOperationRequest(
            operation="CANCEL",
            task_id=agent_stop.task_id,
            expected_task_hash=agent_stop.record_hash,
        ),
        caller="EXTERNAL_AUTOMATION",
    )
    assert agent_closed["lifecycle"] == "CANCELLED"
    assert registry.task(agent_stop.task_id).failure_code == "task_control.ledger_rebuilt"


def test_activity_task_merge_keeps_only_same_version_attention() -> None:
    """Sparse activity projections cannot ungroup a resolved stop or retain stale attention."""
    import shutil
    import subprocess

    node = shutil.which("node")
    assert node is not None, "BADGE requires the UI's Node.js development runtime"
    source = Path(__file__).resolve().parents[2] / (
        "src/alphalattice/interface/local_application/assets/workbench-source/js/app/data.js"
    )
    code = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const c = {window: {ALPHA_PRODUCT: true}, app: {}, URLSearchParams};
vm.createContext(c);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8') + ';globalThis.data=Data;', c);
const D = c.data, sourceHash = 'a'.repeat(64), successorHash = 'b'.repeat(64);
const source = {task_id: 'source', task_kind: 'workspace_preparation', lifecycle: 'BLOCKED',
  task_record_hash: sourceHash};
const successor = {task_id: 'successor', task_kind: 'workspace_preparation', lifecycle: 'SUCCEEDED',
  task_record_hash: successorHash};
const attention = {task_record_hash: sourceHash, unresolved: false,
  resolution: 'SUCCESSOR_SUCCEEDED', successor_task_id: 'successor',
  successor_task_hash: successorHash, successor_lifecycle: 'SUCCEEDED'};
const seed = () => D.setTasks([{...source, attention}, successor]);
const grouped = () => D.groupTaskSuccessors(D.tasks());
seed();
D.mergeTasks([source, successor]);
assert.equal(D.taskAttention(D.taskOf('source')), attention,
  'the same Task version keeps its owner fact');
assert.equal(grouped().length, 1, 'the sparse activity report keeps one canonical successor');
assert.equal(grouped()[0].earlierStops[0].task_id, 'source', 'the earlier stop remains disclosed');
for (const version of [undefined, 'c'.repeat(64)]) {
  seed();
  const incoming = {...source};
  if (version === undefined) delete incoming.task_record_hash;
  else incoming.task_record_hash = version;
  D.mergeTasks([incoming]);
  assert.equal(D.taskAttention(D.taskOf('source')), null,
    'an absent or changed version drops attention');
  assert.equal(Object.hasOwn(D.taskOf('source'), 'attention'), false,
    'the old fact is not retained');
  assert.equal(grouped().length, 2, 'an unbound earlier stop cannot remain grouped');
}
seed();
D.mergeTasks([{task_id: 'source', task_kind: source.task_kind, lifecycle: source.lifecycle}]);
D.mergeTasks([{task_id: 'source', task_kind: source.task_kind, lifecycle: source.lifecycle}]);
assert.equal(Object.hasOwn(D.taskOf('source'), 'attention'), false,
  'two absent versions cannot establish equality');
const replacements = [
  {task_record_hash: sourceHash, unresolved: true, resolution: 'STOPPED'}, null];
for (const replacement of replacements) {
  seed();
  D.mergeTasks([{...source, attention: replacement}]);
  assert.equal(D.taskOf('source').attention, replacement, 'explicit owner attention always wins');
  assert.equal(grouped().length, 2, 'an explicit changed fact removes the earlier grouping');
}
seed();
D.mergeTasks([{...successor, task_record_hash: 'd'.repeat(64)}]);
assert.equal(D.taskSuccessor(D.taskOf('source')), null,
  'a changed successor version invalidates the relation');
assert.equal(grouped().length, 2, 'successor identity remains exact after an activity merge');
"""
    subprocess.run([node, "-e", code, str(source)], check=True, timeout=10)
