"""UIFOLLOW: retained Task checkpoints and attribution drive the real page readers.

The Task/Goal/activity producers are real. Scientific readbacks are labelled transport
fixtures: this proves routing and race safety, not scientific or native acceptance.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    ArtifactReference,
)
from alphalattice.control.product_host.composition.research_experiment_plan import ExperimentPlan
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
)
from alphalattice.control.task_control.contracts import (
    TaskEvidence,
    TaskExecutionCompatibility,
    TaskStageReceipt,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputGateway,
    FeatureInputGovernanceService,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    PortfolioExperimentSource,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    portfolio_review_task_contract,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind, seal_actor_submission
from alphalattice.protocols.research_authoring.contracts import (
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)
from tests.feature_input_gateway.gateway_support import (
    _evidence,
    _manifest,
    _panel_impact,
    _sectors,
    _temporal,
)
from tests.portfolio_strategy_lab.local_web_support import _json, run_node
from tests.workspace_task_runner.task_control_support import digest


def _checkpoint_experiment(workspace_id: str, salt: str):
    """A canonical owner input for metadata-only checkpoints, never scientific evidence."""
    session = date(2024, 8, 12)
    kind = "portfolio.policy-development" if salt == "portfolio" else "factor.screening-development"
    envelope = ResearchExperimentEnvelope.create(
        kind=kind,
        schema_id="research-experiment-envelope",
        data_snapshot_handle="qa-follow-fixture",
        universe_handle="qa-follow-fixture",
        output_workspace="qa-follow/" + salt,
        sessions={
            "start": session,
            "end": session,
            "as_of": {"session": session, "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 1, "maximum_numerical_calls": 1},
        determinism={"seed": 1, "thread_limit": 1, "network_disabled": True},
    )
    authority = ResolvedResearchAuthority.create(
        data_snapshot_handle=envelope.data_snapshot_handle,
        universe_handle=envelope.universe_handle,
        panel_snapshot_hash=digest("qa-panel"),
        panel_manifest_ref="qa-follow/panel.json",
        universe_revision_sha256=digest("qa-universe"),
        ordered_listing_ids=("qa-listing",),
        sessions=(session,),
        source_watermark_hash=digest("qa-watermark"),
    )
    program = SealedResearchProgram.create(
        kind=kind,
        envelope_hash=envelope.envelope_hash,
        authority_hash=authority.authority_hash,
        desk_program_hash=digest("qa-desk:" + salt),
        resolved_sessions=(session,),
        catalog_hash=digest("qa-catalog"),
        method_binding_hash=digest("qa-method"),
        parameter_domain_hash=digest("qa-domain"),
    )
    source = None
    if salt == "portfolio":
        source = PortfolioExperimentSource.create(
            alpha_task_id=str(UUID(int=17)),
            alpha_program_hash=digest("qa-alpha-program"),
            alpha_receipt_hash=digest("qa-alpha-receipt"),
            candidate_id="QA",
            target_recipe_id="QA",
            target_binding_hash=digest("qa-target"),
            input_binding_hash=digest("qa-input"),
            panel_snapshot_hash=authority.panel_snapshot_hash,
            outcome_snapshot_hash=digest("qa-outcome"),
            universe_revision=authority.universe_revision_sha256,
            ordered_listing_ids=("qa-listing",),
            formation_sessions=(session,),
            score_sessions=(session,),
            score_refs=(),
            score_value_hash=digest("qa-scores"),
        )
    plan = ExperimentPlan.create(
        workspace_id=workspace_id,
        binding=ResearchWorkspaceExperimentInput(
            input_id="qa-follow-fixture", binding_hash=digest("qa-input")
        ),
        document={"experiment": envelope.model_dump(mode="json", exclude={"envelope_hash"})},
        program=program,
        authority=authority,
        execution_preview={},
        implementation_hash=digest("qa-implementation"),
        portfolio_source=source,
    )
    actor = seal_actor_submission(
        actor_kind=ActorKind.HUMAN, actor_id="QA fixture", submission_hash=program.program_hash
    )
    return plan, actor


def test_real_task_events_follow_the_attributed_goal_and_preserve_manual_reading(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    """Real checkpoints reach actual UI readers for progress, results, a decision and review."""
    assert live.session is not None and live.operations is not None and live.activity is not None
    assert live.dispatcher is not None
    registry, goals, observer = (
        live.session.task_control_registry,
        live.operations.goals,
        live.activity,
    )
    goal_id, publication = uuid4(), digest("qa-follow-publication")
    opened = goals.operate(
        PortfolioResearchOperationRequest(
            operation="GOAL_OPEN",
            goal_id=goal_id,
            goal_declaration={
                "kind": "OPERATIONS",
                "title": "QA fixture: follow exact owner checkpoints",
                "objective": "Read checkpoint pages; no scientific or native claim.",
                "criteria": [
                    {
                        "criterion_id": "routes",
                        "text": "Every owner checkpoint opens its exact reader.",
                    }
                ],
            },
        ),
        "HUMAN",
    )
    goal = goals.store.load(opened["goal_hash"])
    artifacts: dict[UUID, ArtifactReference] = {}
    observer.attach(
        dispatcher=live.dispatcher,
        registry=registry,
        artifacts=lambda _kind, task_id: artifacts.get(task_id),
        operations=live.operations,
    )
    checkpoints: list[dict[str, object]] = []
    cursor = _json(live, "/api/activity")["cursor"]

    def admit(salt: str, kind: str = "research_experiment"):
        if kind == "research_experiment":
            experiment, actor = _checkpoint_experiment(live.workspace_manifest.workspace_id, salt)
            admission = live.operations.experiments.admit(experiment, actor)
            admitted = registry.task(admission.task_id)
        elif kind == "chief_risk_officer.portfolio_review":
            envelope, intent, plan = portfolio_review_task_contract(
                review_key_payload={"decision_policy_hash": digest("qa-review-policy")}
            )
            admitted = registry.admit(
                input_envelope=envelope, goal=intent, plan=plan, observed_at=live.clock()
            ).record
        else:
            assert kind == "study_verification_sweep"
            admission = live.operations.sweep.admit("ALL")
            admitted = registry.task(admission.task_id)
        envelope, plan = admitted.input, admitted.plan
        request = (
            PortfolioResearchOperationRequest(
                operation="EXPERIMENT_RUN",
                experiment_plan_hash=envelope.payload["plan"]["plan_hash"],
            )
            if kind == "research_experiment"
            else PortfolioResearchOperationRequest(
                operation="CRO_REVIEW"
                if kind == "chief_risk_officer.portfolio_review"
                else "EXPERIMENT_VERIFY_ALL"
            )
        )
        answer = {
            "disposition": "ADMITTED",
            "task_id": str(admitted.task_id),
            "task_lifecycle": "QUEUED",
        }
        span = observer.entered(request, caller="EXTERNAL_AUTOMATION")
        assert span is not None
        observer.returned(span, answer)
        goals.attribute(goal, request, answer, None)
        started = registry.start_next(
            compatibility=TaskExecutionCompatibility.create(
                task_contract_hash=digest("qa-follow-contract"),
                workflow_definition_hash=plan.workflow_definition_hash,
                input_schema_id=envelope.input_schema_id,
                domain_policy_hash=digest("qa-follow-policy"),
                framework_identity_hash=digest("qa-follow-framework"),
            ),
            worker_instance_id=uuid4(),
            expected_task_id=admitted.task_id,
            observed_at=live.clock(),
        )
        assert started is not None
        task, execution = started
        item = registry.begin_work_item(
            task_id=task.task_id,
            execution_id=execution.execution_id,
            stage_id=plan.work_items[0].stage_id,
            observed_at=live.clock(),
        )
        return task, execution, item, plan

    def complete(checkpoint, *, review: bool = False):
        task, execution, item, plan = checkpoint
        content_hash = publication if review else digest("qa-follow:" + str(task.task_id))
        for index, definition in enumerate(plan.work_items):
            if index:
                item = registry.begin_work_item(
                    task_id=task.task_id,
                    execution_id=execution.execution_id,
                    stage_id=definition.stage_id,
                    observed_at=live.clock(),
                )
            evidence_kind = (
                definition.required_evidence_kinds[0]
                if definition.required_evidence_kinds
                else "qa_receipt"
            )
            evidence = (
                TaskEvidence(
                    evidence_kind=evidence_kind,
                    content_hash=content_hash,
                    reference=(
                        "cro_review_publication/"
                        if review and evidence_kind == "cro_review_publication"
                        else "qa_follow_checkpoint/" + definition.stage_id + "/"
                    )
                    + content_hash,
                ),
            )
            registry.mark_ready(
                task_id=task.task_id,
                execution_id=execution.execution_id,
                stage_id=item.stage_id,
                evidence=evidence,
                observed_at=live.clock(),
            )
            finished = registry.verify_work_item(
                TaskStageReceipt.from_identity(
                    receipt_id=uuid4(),
                    task_id=task.task_id,
                    execution_id=execution.execution_id,
                    stage_id=item.stage_id,
                    work_item_definition_hash=item.definition_hash,
                    verifier_id=definition.verifier_id,
                    evidence=evidence,
                    status="VERIFIED",
                    failure_code=None,
                    observed_at=live.clock(),
                )
            )
        artifacts[task.task_id] = ArtifactReference(
            artifact_kind="cro_review_publication" if review else "ResearchExecutionEvidence",
            artifact_hash=content_hash,
        )
        return finished

    def capture(phase: str, task_id: UUID):
        nonlocal cursor
        registry.safe_projection(task_id)
        observer.command_returned(registry.task(task_id).task_kind, task_id, None)
        activity = _json(live, f"/api/activity?after={cursor}&watch={task_id}&limit=100")
        cursor = activity["cursor"]
        assert activity["observer"]["status"] == "OK"
        assert any(row["schema_kind"] == "TaskControlTransition" for row in activity["items"])
        narrative = _json(live, f"/api/goals/narrative?goal_id={goal_id}")
        task_ids = [row["task_id"] for row in narrative["record"]["tasks"]]
        checkpoints.append(
            {
                "phase": phase,
                "task_id": str(task_id),
                "activity": activity,
                "narrative": narrative,
                "listing": _json(live, "/api/goals"),
                "tasks": _json(live, "/api/tasks")["tasks"],
                "decisions": _json(live, "/api/decisions")["decisions"],
                "recoveries": {
                    task: _json(live, f"/api/tasks/recovery?task_id={task}") for task in task_ids
                },
            }
        )

    factor = admit("factor")
    capture("running", factor[0].task_id)
    complete(factor)
    capture("factor", factor[0].task_id)
    portfolio = admit("portfolio")
    complete(portfolio)
    capture("portfolio", portfolio[0].task_id)
    blocked = admit("decision", "study_verification_sweep")
    registry.block_work_item(
        task_id=blocked[0].task_id,
        execution_id=blocked[1].execution_id,
        stage_id=blocked[2].stage_id,
        failure_code="study_verification_sweep.not_admitted",
        observed_at=live.clock(),
    )
    # Retain an actual data owner's issue and preview. Its exact case is attributed
    # to this Goal; STOPPED_TASK is agent work, never forced into a person decision.
    issues = live.operations.data_issues
    manifest = _manifest(20)
    manifest = replace(
        manifest, profile=replace(manifest.profile, market_profile_id="us-current-index-research")
    )
    sectors = _sectors(manifest)
    assessed = FeatureInputGovernanceService(
        issues.market, issues.panel, live.session.mutation_gate, FeatureInputGateway()
    ).assess_and_record(
        candidate_manifest=manifest,
        evidence=_evidence(
            manifest, failed={0}, failure_code="data.unexplained_raw_move", extreme=True
        ),
        sector_by_listing_id=sectors,
        temporal_boundary=_temporal(),
        panel_impact=_panel_impact(manifest, sectors),
        observed_at=live.clock(),
    )
    case = assessed.agent_cases[0]
    issues.market.readiness.save(
        market_profile_id=manifest.profile.market_profile_id,
        status="FEATURE_BUILDING",
        active_manifest_id=manifest.manifest_id,
        active_manifest_revision=manifest.revision_sha256,
        active_membership_fingerprint=None,
        active_candidate_manifest_document=None,
        pending_membership_fingerprint=None,
        pending_candidate_manifest_document=None,
        last_checked_at=live.clock(),
        last_changed_at=live.clock(),
        failure_code=None,
        observed_at=live.clock(),
    )
    offered = next(
        value
        for value in issues.readback()["next_requests"].values()
        if value["operation"] == "DATA_ISSUE_PREVIEW"
    )
    request = PortfolioResearchOperationRequest(**offered)
    answer = live.operations.execute(request, caller="HUMAN")
    assert answer["status"] == "CONFIRMATION_REQUIRED"
    # The same real attribution writer used for the retained checkpoints records
    # this exact owner preview. No case/source membership is inferred by the UI.
    goals.attribute(goal, request, answer, None)
    assert str(goal_id) in goals.decision_attribution()["case_token", case.case_token]
    capture("decision", blocked[0].task_id)
    stop = next(
        row for row in checkpoints[-1]["decisions"] if row.get("task_id") == str(blocked[0].task_id)
    )
    assert stop["waits_on"] == "AGENT"
    person = next(
        row for row in checkpoints[-1]["decisions"] if row.get("case_token") == case.case_token
    )
    assert person["waits_on"] == "PERSON" and str(goal_id) in person["goal_ids"]
    review = admit("review", "chief_risk_officer.portfolio_review")
    complete(review, review=True)
    capture("review", review[0].task_id)
    # The review has no open earlier decision in this transport fixture; the producer's
    # stopped checkpoint remains retained and is checked separately above.
    fixture = {
        "goal_id": str(goal_id),
        "factor": str(factor[0].task_id),
        "portfolio": str(portfolio[0].task_id),
        "blocked": str(blocked[0].task_id),
        "case_token": case.case_token,
        "review": str(review[0].task_id),
        "publication": publication,
        "book": {
            "experiment_task_id": str(portfolio[0].task_id),
            "experiment_receipt_hash": digest("qa-follow-book"),
            "portfolio_session": "2024-08-12",
        },
        "checkpoints": checkpoints,
    }
    path = tmp_path / "goal-follow-fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8", newline="\n")
    root = Path(__file__).resolve().parents[2]
    run_node(
        [
            str(Path(__file__).with_name("workbench_goal_follow.cjs")),
            str(
                root / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
            ),
            str(path),
        ],
        missing="The Workbench holder requires the installed Node runtime.",
        required=True,
        check=True,
        timeout=15,
    )
