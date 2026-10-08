"""The owner returns a filed specialist contribution through its parent to Team (V691)."""

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from alphalattice.control.observation_runtime.adapters import safe_observation_draft
from alphalattice.control.observation_runtime.contracts import (
    ObservationAuthority,
    ObservationRetentionClass,
)
from alphalattice.control.observation_runtime.ledger import (
    ObservationLedger,
    UnifiedObservationPort,
)
from alphalattice.control.observation_runtime.policy import (
    PRODUCT_OPERATION_SCHEMA,
    default_observation_policies,
)
from alphalattice.control.product_host.composition.evidence_review_application import (
    ANSWER_CATEGORY,
)
from alphalattice.control.product_host.composition.evidence_review_bundles import (
    BUNDLE_CATEGORY,
    EvidenceReviewBundles,
)
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskInputEnvelope,
)
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
    AlternativeEvidenceResearchObligation,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceAdmission,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSourcePolicy,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    INPUT_SCHEMA_ID as ANALYST_INPUT_SCHEMA,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    TASK_KIND as ANALYST_TASK_KIND,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    SubmittedEvidenceAnalysis,
)
from alphalattice.interface.local_application.activity import ExternalActivityReadQuery
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
)
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeResearchBinding,
)
from alphalattice.interface.local_application.native_setup import (
    PROJECT_DECLARATION_NAME,
    bind_session,
    declare_project,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    assessment_schema_hash,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewAnswer,
    PortfolioReviewCoverage,
    PortfolioReviewDossier,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import BookAuthority
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    INPUT_SCHEMA_ID as CRO_INPUT_SCHEMA,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    TASK_KIND as CRO_TASK_KIND,
)
from alphalattice.protocols.actor_execution import (
    ActorKind,
    seal_actor_submission,
)
from alphalattice.protocols.actor_execution.answers import (
    AgentAnswerRecord,
    AgentRun,
    AnswerProblem,
    AnswerVerdict,
    answer_digest,
    answer_slot,
)
from alphalattice.protocols.actor_execution.bundles import (
    AgentBundleRecord,
    bundle_directory_key,
    bundle_slot,
)
from tests.portfolio_strategy_lab.local_web_support import _json
from tests.workspace_task_runner.task_control_support import task_contract

JUDGMENT_ROLES = {
    "ALPHA": ("alphalattice_alpha",),
    "ANALYST": ("alphalattice_evidence_analyst",),
    "CRO": ("alphalattice_cro",),
    "DATA": ("alphalattice_data",),
    "FACTOR": ("alphalattice_factor",),
    "PORTFOLIO": ("alphalattice_portfolio",),
    "RISK": ("alphalattice_risk",),
}
"""Each bundle role's shipped card."""


def open_goal(live, title):
    """Declare one synthetic Goal using its public owner."""
    return live.operations.goals.operate(
        PortfolioResearchRequestDocument(
            operation="GOAL_OPEN",
            goal_id=str(uuid4()),
            change_reason="Synthetic declaration",
            goal_declaration={
                "title": title,
                "objective": "Retain the exact accepted exchange.",
                "kind": "RESEARCH",
                "criteria": [{"criterion_id": "read", "text": "Read it."}],
                "deliverables": [
                    {"deliverable_id": "result", "kind": "RESULT", "description": "The receipt."}
                ],
            },
        ).to_operation_request(),
        "HUMAN",
        None,
    )["goal_id"]


@pytest.fixture
def answer_scene(live, tmp_path, monkeypatch):
    """Synthetic acceptance seam: no evidence, specialist or scientific work executes."""
    envelope, goal, plan = task_contract(salt="accepted-answer-conversation")
    task = live.session.task_control_registry.admit(
        input_envelope=envelope,
        goal=goal,
        plan=plan,
        observed_at=datetime(2026, 10, 4, tzinfo=UTC),
    ).record
    directory = bundle_directory_key(str(tmp_path / "bundle"))
    bundle = AgentBundleRecord(
        role="ANALYST",
        bundle_directory=directory,
        submission={
            "operation": "EVIDENCE_ANALYSIS_SUBMIT",
            "result_hash": "d" * 64,
            "task_id": str(task.task_id),
            "analysis_context_hash": "c" * 64,
        },
        files=("README.md",),
        record_hash=bundle_slot(directory),
    )
    accepted = AlternativeEvidenceAnalystAnswer.model_validate(
        {
            "findings": [
                {
                    "issuer": "SYNTHETIC",
                    "topic": "LIQUIDITY_GOING_CONCERN",
                    "direction": "ADVERSE",
                    "summary": "Distinctive authored finding: synthetic liquidity pressure.",
                    "cite": ["S1"],
                }
            ],
        }
    )
    host, parent, child = "codex", "fixture-parent", "fixture-child"
    role = "alphalattice_evidence_analyst"
    project = live.workspace.parent
    (project / ".codex").mkdir(exist_ok=True)
    (project / ".codex/config.toml").write_text(
        '[[hooks.SubagentStart]]\nmatcher = "^alphalattice_.*$"\n',
        encoding="utf-8",
    )
    declare_project(project, "codex")
    (project / ".codex" / BINDING_NAME).write_text(
        json.dumps(
            {
                "session_id": parent,
                "workspace": str(live.workspace),
                "host": host,
                "roles": [role, "alphalattice_cro"],
                "usage": "OFF",
            }
        ),
        encoding="utf-8",
    )
    run = AgentRun(
        host=host,
        session_id=parent,
        agent_id=child,
        role=role,
        model="synthetic-model",
        basis="HOOK",
    )
    record = AgentAnswerRecord(
        bundle_key="b" * 64,
        number=1,
        verdict="ACCEPTED",
        corrections_used=0,
        problems=(),
        accepted_items=(1,),
        answer_digest=answer_digest(accepted),
        agent_run=run,
        record_hash=answer_slot("b" * 64, 1),
    )
    events = [
        (
            "NATIVE_SUBAGENT_START_HOOK",
            {"native_agent_id": child, "role": role, "hook_model": "synthetic-model"},
        ),
        (
            "NATIVE_COORDINATION_MESSAGE",
            {
                "native_agent_id": parent,
                "role": "research_lead",
                "message_kind": "assignment",
                "message_id": "fixture-assignment",
                "recipient_id": child,
                "reference": bundle.record_hash,
            },
        ),
    ]

    def publish_events():
        for sequence, (kind, subject) in enumerate(events):
            assert (
                live.operations.declare_event(
                    live.operations.observer,
                    {
                        "event_kind": kind,
                        "producer_id": "fixture-native",
                        "producer_session": parent,
                        "producer_sequence": sequence,
                        "occurred_at": task.admitted_at.isoformat(),
                        "summary": "Synthetic hook or assignment.",
                        "subject": {"native_host": host, "native_session_id": parent, **subject},
                    },
                )["status"]
                == "APPENDED"
            )

    def submit(
        *,
        filed=record,
        retry=None,
        verdict="ACCEPTED",
        disposition="ADMITTED",
        newly_filed=True,
        goal_id=None,
    ):
        live.review.artifacts.publish(ANSWER_CATEGORY, filed.record_hash, filed)
        delivery = live.review.accepted_answer_delivery(
            retry or filed,
            accepted.model_dump(mode="json"),
            task_id=task.task_id,
            newly_filed=newly_filed,
        )
        body = {
            "disposition": disposition,
            "task_id": str(task.task_id),
            "lifecycle": "QUEUED",
            "answer": {
                "verdict": verdict,
                "number": filed.number,
                "accepted_items": list(filed.accepted_items),
                "dropped": [],
            },
        }
        if delivery is not None:
            body["answer"]["accepted_delivery"] = delivery
        monkeypatch.setattr(EvidenceReviewBundles, "agent_bundle", lambda self, directory: bundle)
        monkeypatch.setattr(live.review, "submit_analysis", lambda **kwargs: body)
        token = REQUEST_PROVENANCE.set(
            RequestProvenance(vendor=host, session=parent, goal_id=goal_id)
        )
        try:
            return live.operations.execute(
                PortfolioResearchRequestDocument(
                    operation="AGENT_ANSWER_SUBMIT",
                    bundle_directory=directory,
                    agent_answer=accepted.model_dump(mode="json"),
                ).to_operation_request(),
                caller="EXTERNAL_AUTOMATION",
            )
        finally:
            REQUEST_PROVENANCE.reset(token)

    def answers():
        return [
            item
            for item in live.operations.observer.read_external(ExternalActivityReadQuery())["items"]
            if (item.get("payload") or {}).get("subject", {}).get("input_channel")
            == "PRODUCT_ACCEPTED_ANSWER"
        ]

    return live, bundle, accepted, record, events, publish_events, submit, answers


@pytest.fixture
def product_answer_scene(live, tmp_path):
    """Use the actual generic owners and sealed records without native author claims."""
    envelope, goal, plan = task_contract(salt="product-accepted-answer-read")
    task = live.session.task_control_registry.admit(
        input_envelope=envelope,
        goal=goal,
        plan=plan,
        observed_at=datetime(2026, 10, 6, tzinfo=UTC),
    ).record
    directory = str(tmp_path / "owner-risk-bundle")
    prepared = live.operations.execute(
        PortfolioResearchRequestDocument(
            operation="AGENT_BUNDLE_PREPARE",
            agent_role="RISK",
            task_id=task.task_id,
            bundle_directory=directory,
        ).to_operation_request(),
        caller="EXTERNAL_AUTOMATION",
    )
    assert prepared["status"] == "AGENT_BUNDLE_READY"
    bundle = EvidenceReviewBundles(live.review).agent_bundle(directory)
    assert bundle is not None
    authored = {
        "text": "Public accepted Risk conclusion: the synthetic Task has no numerical report.",
        "references": [str(task.task_id)],
    }

    def submit(*, session="owner-submit-first", goal_id=None, vendor="codex"):
        client = LocalResearchClient(live.workspace)
        cursor = client.activity(limit=200)["cursor"]
        token = REQUEST_PROVENANCE.set(
            RequestProvenance(
                vendor=None if session is None else vendor, session=session, goal_id=goal_id
            )
        )
        try:
            result = live.operations.execute(
                PortfolioResearchRequestDocument(
                    operation="AGENT_ANSWER_SUBMIT",
                    bundle_directory=directory,
                    agent_answer=authored,
                ).to_operation_request(),
                caller="EXTERNAL_AUTOMATION",
            )
        finally:
            REQUEST_PROVENANCE.reset(token)
        page = client.activity(after=cursor, limit=200)
        (row,) = [
            item
            for item in page["items"]
            if item["schema_kind"] == PRODUCT_OPERATION_SCHEMA
            and item["payload"].get("operation") == "AGENT_ANSWER_SUBMIT"
            and item["payload"].get("phase") == "RETURNED"
        ]
        return result, row

    return SimpleNamespace(live=live, task=task, bundle=bundle, authored=authored, submit=submit)


@pytest.fixture(params=["codex", "claude-code"])
def no_hook_answer_scene(product_answer_scene, monkeypatch, request):
    """Public generic answer and Goal owners, with no lifecycle observation or message."""
    scene = product_answer_scene
    scene.host = request.param
    scene.project = scene.live.workspace.parent
    scene.parent = "owner-submit-first"
    directory = ".claude" if scene.host == "claude-code" else ".codex"
    declaration = "settings.json" if scene.host == "claude-code" else "config.toml"
    source = Path(__file__).resolve().parents[2] / directory
    shutil.copytree(source / "agents", scene.project / directory / "agents")
    (scene.project / directory / declaration).write_bytes((source / declaration).read_bytes())
    declare_project(scene.project, scene.host)
    environment = "CLAUDE_CODE_SESSION_ID" if scene.host == "claude-code" else "CODEX_THREAD_ID"
    other = "CODEX_THREAD_ID" if scene.host == "claude-code" else "CLAUDE_CODE_SESSION_ID"
    monkeypatch.delenv(other, raising=False)
    monkeypatch.setenv(environment, scene.parent)
    bind_session(
        scene.project,
        host=scene.host,
        session_id=scene.parent,
        workspace=scene.live.workspace,
        usage="off",
    )
    scene.binding = NativeResearchBinding.read(scene.project)
    scene.client = LocalResearchClient(scene.live.workspace)
    original_submit = scene.submit

    def host_submit(**kwargs):
        return original_submit(vendor=scene.host, **kwargs)

    scene.submit = host_submit

    def take(goal):
        assert (
            scene.client.request({"operation": "GOAL_TAKE", "goal_id": goal})["status"]
            == "GOAL_TAKEN"
        )

    scene.take = take
    return scene


def _accepted_rows(record):
    return [
        row
        for row in record["conversation"]
        if row.get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
    ]


def test_an_accepted_answer_is_filed_as_the_leads_product_fact(no_hook_answer_scene):
    """FLOW-1: with no hook and no dispatch message, the product's acceptance is filed in the
    Goal's conversation as the lead's fact, its author unobserved; a retry is the same row."""
    scene = no_hook_answer_scene
    goal = open_goal(scene.live, "Default accepted contribution")
    scene.take(goal)
    result, returned = scene.submit(goal_id=goal)
    conversation = result["conversation"]
    assert result["status"] == "ACCEPTED" and conversation["status"] == "DELIVERED"
    assert result["recorded_agent"]["basis"] == "NOT_OBSERVED"
    assert "native_authorship" not in conversation and "assignment_closure" not in conversation
    assert conversation["original_goal_id"] == goal
    assert (conversation["original_session_host"], conversation["original_session_id"]) == (
        scene.host,
        scene.parent,
    )
    record = scene.client.request({"operation": "GOAL_SHOW", "goal_id": goal})["record"]
    (answer,) = _accepted_rows(record)
    assert answer["authorship_basis"] == "NOT_OBSERVED" and answer["agent_id"] == scene.parent
    assert answer["bundle_role"] == "RISK"
    selected = scene.client.read_external(observation_id=conversation["observation_id"])
    detail = selected["accepted_answer"]
    assert detail["status"] == "AVAILABLE"
    assert detail["recorded_agent"]["basis"] == "NOT_OBSERVED"
    assert detail["contribution"] == scene.authored
    assert detail["receipt_observation_id"] == returned["observation_id"]
    assert selected["read_cost"]["observations"] == 2
    assert not any(
        "HOOK" in row["payload"]["event_kind"] for row in scene.client.read_external()["items"]
    )
    retry, _ = scene.submit(goal_id=goal)
    assert retry["conversation"] == conversation
    after = scene.client.request({"operation": "GOAL_SHOW", "goal_id": goal})["record"]
    assert _accepted_rows(after) == [answer]


def test_default_retry_keeps_original_goal(no_hook_answer_scene):
    scene = no_hook_answer_scene
    original = open_goal(scene.live, "Original accepted receipt")
    scene.take(original)
    original_binding = scene.binding.record_path(scene.project)
    original_bytes = original_binding.read_bytes()
    other_session = str(UUID(int=910))
    assert other_session != scene.parent
    assert (
        bind_session(
            scene.project,
            host=scene.host,
            session_id=other_session,
            workspace=scene.live.workspace,
            usage="off",
        )["status"]
        == "BOUND"
    )
    assert len(NativeResearchBinding.bindings(scene.project)) == 2
    assert original_binding.read_bytes() == original_bytes
    first, returned = scene.submit(goal_id=original)
    later = open_goal(scene.live, "Later work")
    scene.take(later)
    retried, _ = scene.submit(goal_id=later)
    assert retried["conversation"] == first["conversation"]
    assert retried["recorded_agent"] == first["recorded_agent"]
    later_record = scene.client.request({"operation": "GOAL_SHOW", "goal_id": later})["record"]
    assert _accepted_rows(later_record) == []
    detail = scene.client.read_external(observation_id=first["conversation"]["observation_id"])[
        "accepted_answer"
    ]
    assert detail["status"] == "AVAILABLE" and detail["goal_id"] == original
    assert detail["receipt_observation_id"] == returned["observation_id"]
    assert original_binding.read_bytes() == original_bytes
    assert (
        NativeResearchBinding.read(scene.project, session=(scene.host, other_session)).session_id
        == other_session
    )


def test_optional_missing_project_observation_recovers_under_original_accepted_goal(
    no_hook_answer_scene,
):
    scene = no_hook_answer_scene
    original = open_goal(scene.live, "Original optional observation")
    scene.take(original)
    directory = ".claude" if scene.host == "claude-code" else ".codex"
    (scene.project / directory / PROJECT_DECLARATION_NAME).unlink()
    first, _ = scene.submit(goal_id=original)
    assert first["status"] == "ACCEPTED" and first["conversation"]["status"] == "UNAVAILABLE"
    later = open_goal(scene.live, "Independent retry Goal")
    scene.take(later)
    declare_project(scene.project, scene.host)
    repaired, _ = scene.submit(goal_id=later)
    assert repaired["status"] == "ACCEPTED" and repaired["conversation"]["status"] == "DELIVERED"
    assert repaired["conversation"]["original_goal_id"] == original
    assert repaired["answer_reference"] == first["answer_reference"]
    assert repaired["recorded_agent"] == first["recorded_agent"]
    later_record = scene.client.request({"operation": "GOAL_SHOW", "goal_id": later})["record"]
    assert _accepted_rows(later_record) == []
    original_record = scene.client.request({"operation": "GOAL_SHOW", "goal_id": original})[
        "record"
    ]
    assert len(_accepted_rows(original_record)) == 1


def test_unreadable_first_accepted_context_keeps_science_and_cannot_be_recaptured(
    no_hook_answer_scene,
):
    scene = no_hook_answer_scene
    original = open_goal(scene.live, "Acceptance context cannot be read")
    scene.take(original)
    goals = scene.live.operations.goals
    context = goals.accepted_answer_context

    def unreadable(**_kwargs):
        raise ValueError("PRIVATE-CONTEXT-READ-ERROR")

    goals.accepted_answer_context = unreadable
    first, _ = scene.submit(goal_id=original)
    assert first["status"] == "ACCEPTED"
    assert first["conversation"]["status"] == "UNAVAILABLE"
    assert first["conversation"]["reason"] == "native_bridge.accepted_context_unavailable"
    assert first["conversation"]["missing"] == ["first_accepted_context"]
    assert "PRIVATE-CONTEXT-READ-ERROR" not in str(first)
    goals.accepted_answer_context = context
    recovered, _ = scene.submit(goal_id=original)
    assert recovered["status"] == "ACCEPTED"
    assert recovered["answer_reference"] == first["answer_reference"]
    assert recovered["recorded_agent"] == first["recorded_agent"]
    assert recovered["conversation"]["status"] == "DELIVERED"
    assert recovered["conversation"]["original_goal_id"] is None


def test_public_accepted_label_cannot_borrow_the_real_owner_return(no_hook_answer_scene):
    scene = no_hook_answer_scene
    goal = open_goal(scene.live, "Exact selected owner receipt")
    scene.take(goal)
    accepted, _ = scene.submit(goal_id=goal)
    original_id = accepted["conversation"]["observation_id"]
    original = next(
        row for row in scene.client.read_external()["items"] if row["observation_id"] == original_id
    )
    declared = {
        "event_kind": original["payload"]["event_kind"],
        "producer_id": "public-declared-copy",
        "producer_session": original["payload"]["producer_session"],
        "producer_sequence": 0,
        "occurred_at": original["occurred_at"],
        "summary": original["payload"]["summary"],
        "subject": {
            key: value for key, value in original["payload"]["subject"].items() if key != "goal_id"
        },
        "correlation_ids": original["correlation_ids"],
    }
    posted = scene.client.publish_event(declared)
    assert posted["status"] == "APPENDED"
    selected = scene.client.read_external(observation_id=posted["observation_id"])
    assert selected["accepted_answer"]["status"] == "UNAVAILABLE"
    assert selected["accepted_answer"]["missing"] == ["accepted_operation_return"]
    real = scene.client.read_external(observation_id=original_id)["accepted_answer"]
    assert real["status"] == "AVAILABLE" and real["recorded_agent"]["basis"] == "NOT_OBSERVED"


@pytest.mark.parametrize("session", ["owner-submit-first", None], ids=["not-observed", "no-run"])
def test_selected_product_return_reads_the_exact_accepted_contribution(
    product_answer_scene, session
):
    """Owner acceptance opens public content without creating native authorship."""
    scene = product_answer_scene
    goal_id = open_goal(scene.live, "Actual accepted product return")
    result, row = scene.submit(session=session, goal_id=goal_id)
    assert result["status"] == "ACCEPTED"
    assert result["bundle_reference"] == scene.bundle.record_hash
    assert result["receipt"]["task_id"] == str(scene.task.task_id)
    record = scene.live.review.artifacts.load(
        ANSWER_CATEGORY, result["answer_reference"], AgentAnswerRecord
    )
    client = LocalResearchClient(scene.live.workspace)
    page = client.activity(limit=200)
    assert "schema_version" not in row, "ActivityItem is the versionless public list projection"
    assert row["availability"] == "AVAILABLE"
    selected = client.read_external(observation_id=row["observation_id"])
    detail = selected["accepted_answer"]
    assert selected["read_cost"]["observations"] == 1 and "items" not in selected
    assert detail["status"] == "AVAILABLE" and detail["record_kind"] == "PRODUCT_OPERATION"
    assert detail["observation_id"] == row["observation_id"]
    assert detail["authority"] == "OPERATIONAL_ASSERTION"
    assert detail["source_id"] == row["source_id"]
    assert detail["agent_session"] == session and detail["goal_id"] == goal_id
    assert detail["agent_vendor"] == (None if session is None else "codex")
    assert detail["task_id"] == str(scene.task.task_id)
    assert detail["answer_reference"] == record.record_hash
    assert detail["answer_digest"] == record.answer_digest == answer_digest(scene.authored)
    assert detail["bundle_reference"] == scene.bundle.record_hash
    assert detail["contribution"] == scene.authored
    if session is None:
        assert record.agent_run is None and detail["recorded_agent"] is None
    else:
        assert detail["recorded_agent"] == record.agent_run.model_dump(mode="json")
        assert detail["recorded_agent"]["basis"] == "NOT_OBSERVED"
        assert detail["recorded_agent"]["agent_id"] is None
        assert detail["recorded_agent"]["role"] is None
    assert (
        not {"native_host", "native_session_id", "native_agent_id", "authorship_basis"}
        & detail.keys()
    )
    # The operation rows carry no contribution text; a self-bound Session's Conversation row
    # holds only its bounded preview (AUTOBIND).
    operations = [item for item in page["items"] if item["schema_kind"] == PRODUCT_OPERATION_SCHEMA]
    assert scene.authored["text"] not in json.dumps(operations)
    assert client.activity(limit=200)["items"] == page["items"]
    external = client.read_external()["items"]
    if session is None:
        assert external == []
    else:
        # The Session bound itself at this request: its bound fact and its delivered answer.
        kinds = sorted(row["payload"]["subject"]["message_kind"] for row in external)
        assert kinds == ["answer", "session_bound"]
    assert scene.live.session.task_control_registry.tasks() == (scene.task,)


@pytest.mark.parametrize(
    "first_session", ["owner-submit-first", None], ids=["not-observed", "no-original-run"]
)
def test_selected_product_return_reuse_keeps_original_run_and_actual_new_submitter(
    product_answer_scene, first_session
):
    """An exact reused seal keeps its original run beside the new operation's provenance."""
    scene = product_answer_scene
    first_goal = open_goal(scene.live, "Original accepted submitter")
    second_goal = open_goal(scene.live, "Exact reused submitter")
    first, first_row = scene.submit(session=first_session, goal_id=first_goal)
    original = scene.live.review.artifacts.load(
        ANSWER_CATEGORY, first["answer_reference"], AgentAnswerRecord
    )
    second, second_row = scene.submit(session="owner-submit-second", goal_id=second_goal)
    assert first["status"] == second["status"] == "ACCEPTED"
    assert first["answer_reference"] == second["answer_reference"]
    assert first_row["observation_id"] != second_row["observation_id"]
    client = LocalResearchClient(scene.live.workspace)
    first_read = client.read_external(observation_id=first_row["observation_id"])["accepted_answer"]
    second_read = client.read_external(observation_id=second_row["observation_id"])[
        "accepted_answer"
    ]
    assert first_read["status"] == second_read["status"] == "AVAILABLE"
    assert (first_read["agent_session"], first_read["goal_id"]) == (
        first_session,
        first_goal,
    )
    assert (second_read["agent_session"], second_read["goal_id"]) == (
        "owner-submit-second",
        second_goal,
    )
    original_run = (
        None if original.agent_run is None else original.agent_run.model_dump(mode="json")
    )
    assert first_read["recorded_agent"] == second_read["recorded_agent"] == original_run
    if first_session is None:
        assert original.agent_run is None and second_read["recorded_agent"] is None
    else:
        assert second_read["recorded_agent"]["session_id"] == first_session
    assert (
        scene.live.review.artifacts.load(
            ANSWER_CATEGORY, second["answer_reference"], AgentAnswerRecord
        )
        == original
    )


def test_selected_product_return_survives_host_instance_restart(product_answer_scene):
    """A retained actual owner receipt is independent of the current Host instance."""
    scene = product_answer_scene
    result, row = scene.submit()
    scene.live.stop()
    scene.live.start()
    assert row["source_id"] != scene.live.activity.source_id
    detail = LocalResearchClient(scene.live.workspace).read_external(
        observation_id=row["observation_id"]
    )["accepted_answer"]
    assert detail["status"] == "AVAILABLE"
    assert detail["source_id"] == row["source_id"]
    assert detail["answer_reference"] == result["answer_reference"]
    assert detail["contribution"] == scene.authored


@pytest.mark.parametrize("session", ["owner-submit-first", None], ids=["not-observed", "no-run"])
def test_native_selected_event_cannot_borrow_product_acceptance_authorship(
    product_answer_scene, session
):
    """A claimed native answer event cannot make an accepted unknown author into a child."""
    scene = product_answer_scene
    result, _row = scene.submit(session=session)
    client = LocalResearchClient(scene.live.workspace)
    claimed = client.publish_event(
        {
            "event_kind": "NATIVE_COORDINATION_MESSAGE",
            "producer_id": "fixture-claimed-native",
            "producer_session": "owner-submit-first",
            "producer_sequence": 0,
            "occurred_at": datetime(2026, 10, 6, tzinfo=UTC).isoformat(),
            "summary": "Synthetic claimed native answer; no hook delivery authority.",
            "subject": {
                "message_kind": "answer",
                "input_channel": "PRODUCT_ACCEPTED_ANSWER",
                "native_host": "codex",
                "native_session_id": "owner-submit-first",
                "native_agent_id": "invented-child",
                "role": "alphalattice_risk",
                "submitted_by": "owner-submit-first",
                "authorship_basis": "HOOK",
                "reference": str(scene.task.task_id),
                "task_id": str(scene.task.task_id),
                "answer_reference": result["answer_reference"],
                "bundle_reference": scene.bundle.record_hash,
            },
        }
    )
    assert claimed["status"] == "APPENDED" and claimed["authority"] == "AGENT_PROPOSAL"
    detail = client.read_external(observation_id=claimed["observation_id"])["accepted_answer"]
    assert detail["status"] == "UNAVAILABLE"
    assert detail["reason"] == "activity.accepted_answer_unavailable"
    assert "contribution" not in detail and scene.authored["text"] not in json.dumps(detail)


@pytest.mark.parametrize(
    "case",
    [
        "source_kind",
        "source_id",
        "authority",
        "operation",
        "phase",
        "status",
        "verdict",
        "answer_reference",
        "bundle_reference",
        "old_missing_bundle",
        "role",
        "payload_task",
        "subject_task",
        "envelope_task",
        "foreign_task",
        "run",
        "submitter_pair",
        "goal",
    ],
)
def test_selected_product_return_rejects_malformed_owner_binding(product_answer_scene, case):
    """Only one exact owner source, terminal verdict, seal and Task can expose content."""
    scene = product_answer_scene
    _result, actual = scene.submit()
    payload = json.loads(json.dumps(actual["payload"]))
    subject = payload["subject"]
    source_kind, source_id = "PRODUCT_OPERATION", "local-web:" + "f" * 32
    authority = ObservationAuthority.OPERATIONAL_ASSERTION
    task_id, run_id = str(scene.task.task_id), actual["run_id"]
    if case == "source_kind":
        source_kind = "EXTERNAL_CLIENT"
    elif case == "source_id":
        source_id = "external-client:fixture"
    elif case == "authority":
        authority = ObservationAuthority.AGENT_PROPOSAL
    elif case in {"operation", "phase", "status", "verdict"}:
        key, value = {
            "operation": ("operation", "AGENT_BUNDLE_PREPARE"),
            "phase": ("phase", "REQUESTED"),
            "status": ("status", "CORRECT"),
            "verdict": ("status", "DONE"),
        }[case]
        payload[key] = value
    elif case in {"answer_reference", "bundle_reference"}:
        subject[case] = "e" * 64
    elif case == "old_missing_bundle":
        del subject["bundle_reference"]
    elif case == "role":
        subject["agent_role"] = "FACTOR"
    elif case == "payload_task":
        payload["task_id"] = str(uuid4())
    elif case == "subject_task":
        subject["task_id"] = str(uuid4())
    elif case == "envelope_task":
        task_id = str(uuid4())
    elif case == "foreign_task":
        envelope, goal, plan = task_contract(salt="foreign-product-accepted-task")
        foreign = scene.live.session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=scene.task.admitted_at
        ).record
        task_id = payload["task_id"] = subject["task_id"] = str(foreign.task_id)
        run_id = f"local-web:{task_id}"
    elif case == "run":
        run_id = "local-web:foreign-task"
    elif case == "submitter_pair":
        del subject["agent_vendor"]
    elif case == "goal":
        subject["goal_id"] = "invalid-goal"
    policies = default_observation_policies()
    with ObservationLedger(
        scene.live.workspace / "runtime/observations.sqlite", gate=scene.live.session.mutation_gate
    ) as ledger:
        receipt = UnifiedObservationPort(ledger=ledger, policies=policies).emit(
            safe_observation_draft(
                policies=policies,
                schema_kind=PRODUCT_OPERATION_SCHEMA,
                occurred_at=datetime(2026, 10, 6, tzinfo=UTC),
                source_kind=source_kind,
                source_id=source_id,
                source_sequence=0,
                payload=payload,
                authority=authority,
                retention_class=ObservationRetentionClass.TRANSIENT_OPERATIONAL,
                task_id=task_id,
                run_id=run_id,
            )
        )
    detail = LocalResearchClient(scene.live.workspace).read_external(
        observation_id=receipt.observation_id
    )["accepted_answer"]
    assert detail["status"] == "UNAVAILABLE"
    assert detail["reason"] in {
        "activity.accepted_answer_unavailable",
        "activity.accepted_answer_binding_mismatch",
    }
    assert "contribution" not in detail and scene.authored["text"] not in json.dumps(detail)


@pytest.mark.parametrize(
    "case", ["missing_answer", "missing_bundle", "text_tail", "digest", "bundle_binding"]
)
def test_selected_product_return_reopens_and_verifies_its_sealed_content(
    product_answer_scene, case
):
    """A genuine receipt cannot expose bytes that no longer match its exact sealed answer."""
    scene = product_answer_scene
    result, row = scene.submit()
    answer_path = (
        scene.live.review.artifacts.root / ANSWER_CATEGORY / f"{result['answer_reference']}.json"
    )
    bundle_path = (
        scene.live.review.artifacts.root / BUNDLE_CATEGORY / f"{scene.bundle.record_hash}.json"
    )
    if case == "missing_answer":
        answer_path.unlink()
    elif case == "missing_bundle":
        bundle_path.unlink()
    else:
        path = bundle_path if case == "bundle_binding" else answer_path
        stored = json.loads(path.read_text(encoding="utf-8"))
        if case == "bundle_binding":
            stored["submission"]["task_record_hash"] = "e" * 64
        elif case == "text_tail":
            stored["accepted_text"] += " Unaccepted appended text."
        else:
            stored["answer_digest"] = "e" * 64
        path.write_text(json.dumps(stored), encoding="utf-8", newline="\n")
    detail = LocalResearchClient(scene.live.workspace).read_external(
        observation_id=row["observation_id"]
    )["accepted_answer"]
    assert detail["status"] == "UNAVAILABLE"
    assert "contribution" not in detail and scene.authored["text"] not in json.dumps(detail)


def specialist_scene(answer_scene, tmp_path, monkeypatch, role):
    """Prepare and submit through the real shared generic doors on one synthetic Task."""
    live, original, _accepted, _record, events, _publish, _submit, answers = answer_scene
    task_id = UUID(original.submission["task_id"])
    task = live.session.task_control_registry.task(task_id)
    directory = str(tmp_path / f"{role.lower()}-bundle")
    prepared = live.operations.execute(
        PortfolioResearchRequestDocument(
            operation="AGENT_BUNDLE_PREPARE",
            agent_role=role,
            task_id=task_id,
            bundle_directory=directory,
        ).to_operation_request(),
        caller="EXTERNAL_AUTOMATION",
    )
    assert prepared["status"] == "AGENT_BUNDLE_READY"
    bundle = EvidenceReviewBundles(live.review).agent_bundle(directory)
    card = JUDGMENT_ROLES[role][0]
    binding_path = live.workspace.parent / ".codex" / BINDING_NAME
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["roles"].append(card)
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    events[0][1]["role"] = card
    events[1][1]["reference"] = bundle.record_hash
    accepted_at = datetime(2026, 10, 5, 3, 45, tzinfo=UTC)
    monkeypatch.setattr(live.review, "clock", lambda: accepted_at)
    authored = {
        "text": f"Distinctive {role} interpretation: the listed Task has no numerical report.",
        "references": [str(task_id)],
        "read": [value["name"] for value in prepared["files"]],
    }

    def submit(answer=None, *, goal_id=None):
        token = REQUEST_PROVENANCE.set(
            RequestProvenance(vendor="codex", session="fixture-parent", goal_id=goal_id)
        )
        try:
            return live.operations.execute(
                PortfolioResearchRequestDocument(
                    operation="AGENT_ANSWER_SUBMIT",
                    bundle_directory=directory,
                    agent_answer=authored if answer is None else answer,
                ).to_operation_request(),
                caller="EXTERNAL_AUTOMATION",
            )
        finally:
            REQUEST_PROVENANCE.reset(token)

    return live, task, prepared, bundle, authored, accepted_at, submit, answers


@pytest.mark.parametrize("role", ["ALPHA", "DATA", "FACTOR", "PORTFOLIO", "RISK"])
def test_generic_specialist_doors_seal_exact_references_without_scientific_admission(
    answer_scene, tmp_path, monkeypatch, role
):
    """BEHAVIOUR: all five roles keep actual text and source Task; the conversation row is the
    lead's product fact, its author unobserved (FLOW-1)."""
    live, task, prepared, bundle, authored, accepted_at, submit, answers = specialist_scene(
        answer_scene, tmp_path, monkeypatch, role
    )
    authored["text"] = (
        f"Distinctive {role} interpretation: the listed Task has no numerical report. " * 20
    )
    assert prepared["task_id"] == str(task.task_id)
    assert bundle.submission == {
        "task_id": str(task.task_id),
        "task_record_hash": task.record_hash,
    }
    assert {str(task.task_id), task.record_hash, task.input.input_hash}.issubset(
        bundle.allowed_references
    )
    assert "does not contain numerical diagnostics" in prepared["files"][0]["text"]
    result = submit()
    assert result["status"] == "ACCEPTED"
    assert result["receipt"]["task_id"] == str(task.task_id)
    assert result["answer_reference"] == result["receipt"]["answer_reference"]
    record = live.review.artifacts.load(
        ANSWER_CATEGORY, result["answer_reference"], AgentAnswerRecord
    )
    assert record.accepted_text == authored["text"]
    assert record.accepted_references == tuple(authored["references"])
    assert record.read_files == tuple(authored["read"])
    assert record.accepted_at == accepted_at
    assert live.session.task_control_registry.tasks() == (task,)
    assert result["recorded_agent"]["basis"] == "NOT_OBSERVED"
    assert result["conversation"]["status"] == "DELIVERED"
    assert result["conversation"]["task_id"] == str(task.task_id)
    assert result["conversation"]["bundle_reference"] == bundle.record_hash
    assert result["conversation"]["session_id"] == "fixture-parent"
    (row,) = answers()
    subject = row["payload"]["subject"]
    assert len(row["payload"]["summary"]) <= 500
    assert row["payload"]["summary"] != authored["text"]
    assert (subject["native_agent_id"], subject["role"]) == ("fixture-parent", "research_lead")
    assert subject["authorship_basis"] == "NOT_OBSERVED" and subject["bundle_role"] == role
    assert subject["submitted_by"] == "fixture-parent"
    assert subject["reference"] == str(task.task_id)
    assert subject["answer_reference"] == record.record_hash
    assert subject["bundle_reference"] == bundle.record_hash
    assert subject["source_time_kind"] == "PRODUCT_ACCEPTED_AT"
    assert row["occurred_at"] == accepted_at.isoformat()

    def refuse_page_scan(*_args, **_kwargs):
        pytest.fail("A selected answer must not scan external observation pages.")

    with monkeypatch.context() as selected_read:
        selected_read.setattr(ObservationLedger, "observations_of_kind", refuse_page_scan)
        selected = _json(live, f"/api/activity/external?observation_id={row['observation_id']}")
    assert selected["observation_id"] == row["observation_id"]
    assert "items" not in selected
    detail = selected["accepted_answer"]
    assert detail["status"] == "AVAILABLE"
    assert detail["observation_id"] == row["observation_id"]
    assert detail["recorded_agent"]["basis"] == "NOT_OBSERVED"
    assert detail["task_id"] == str(task.task_id)
    assert detail["contribution"] == {
        "text": authored["text"],
        "references": authored["references"],
    }
    page = _json(live, "/api/activity/external")
    assert "accepted_answer" not in page
    assert page["epoch"] == selected["epoch"]
    assert answers() == [row]


@pytest.mark.parametrize(
    "invalid",
    [
        {"text": " "},
        {"text": "x" * 4001},
        {"text": "A bounded interpretation.", "references": ["unassigned-reference"]},
        {"text": "A bounded interpretation.", "references": ["x" * 201]},
    ],
)
def test_generic_specialist_shape_or_unassigned_reference_is_corrected_before_acceptance(
    answer_scene, tmp_path, monkeypatch, invalid
):
    """BEHAVIOUR: bounded prose is accepted only against its exact prepared reference list."""
    live, task, _prepared, _bundle, _authored, _accepted_at, submit, answers = specialist_scene(
        answer_scene, tmp_path, monkeypatch, "RISK"
    )
    correction = submit(invalid)
    assert correction["status"] == "CORRECT" and correction["accepted_items"] == []
    assert answers() == [] and live.session.task_control_registry.tasks() == (task,)
    accepted = submit()
    assert accepted["status"] == "ACCEPTED" and accepted["receipt"]["accepted_items"] == 1
    assert len(answers()) == 1


@pytest.mark.parametrize(
    "case",
    [
        "missing_answer",
        "missing_bundle",
        "historical_record",
        "text_tail_tampered",
        "author",
        "session",
        "submitter",
        "role",
        "bundle_binding",
        "task",
    ],
)
def test_selected_answer_refuses_missing_tampered_or_mismatched_seals(
    answer_scene, tmp_path, monkeypatch, case
):
    """A selected observation cannot grant prose from a missing or differently bound seal."""
    live, task, _prepared, bundle, authored, _at, submit, answers = specialist_scene(
        answer_scene, tmp_path, monkeypatch, "RISK"
    )
    authored["text"] = "The accepted diagnostics have an exact retained interpretation. " * 20
    result = submit()
    assert result["status"] == "ACCEPTED"
    (row,) = answers()
    subject = dict(row["payload"]["subject"])
    page = _json(live, "/api/activity/external")
    answer_path = (
        live.review.artifacts.root / ANSWER_CATEGORY / f"{result['answer_reference']}.json"
    )
    bundle_path = live.review.artifacts.root / BUNDLE_CATEGORY / f"{bundle.record_hash}.json"
    if case == "missing_answer":
        answer_path.unlink()
    elif case == "missing_bundle":
        bundle_path.unlink()
    elif case in {"historical_record", "text_tail_tampered"}:
        stored = json.loads(answer_path.read_text(encoding="utf-8"))
        if case == "historical_record":
            stored.update(accepted_text=None, accepted_references=[], accepted_at=None)
        else:
            stored["accepted_text"] = authored["text"] + "An unaccepted tail."
        answer_path.write_text(json.dumps(stored), encoding="utf-8", newline="\n")
    elif case == "bundle_binding":
        stored = json.loads(bundle_path.read_text(encoding="utf-8"))
        stored["submission"]["task_record_hash"] = "f" * 64
        bundle_path.write_text(json.dumps(stored), encoding="utf-8", newline="\n")
    elif case == "task":
        envelope, goal, plan = task_contract(salt="foreign-selected-answer-task")
        other = live.session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=task.admitted_at
        ).record
        subject.update(task_id=str(other.task_id), reference=str(other.task_id))
    else:
        key = {
            "author": "native_agent_id",
            "session": "native_session_id",
            "submitter": "submitted_by",
            "role": "role",
        }[case]
        subject[key] = "alphalattice_factor" if case == "role" else "foreign-participant"
    detail = live.review.read_accepted_answer(subject)
    assert detail["status"] == "UNAVAILABLE"
    assert detail["reason"] in {
        "activity.accepted_answer_unavailable",
        "activity.accepted_answer_binding_mismatch",
    }
    assert "contribution" not in detail
    assert authored["text"] not in json.dumps(detail)
    assert _json(live, "/api/activity/external")["items"] == page["items"]


@pytest.mark.parametrize("explicit_references", [False, True])
def test_selected_answer_keeps_both_legal_empty_reference_forms(
    answer_scene, tmp_path, monkeypatch, explicit_references
):
    """An optional empty reference field keeps the exact accepted text and retained digest."""
    live, _task, _prepared, _bundle, _authored, _at, submit, answers = specialist_scene(
        answer_scene, tmp_path, monkeypatch, "RISK"
    )
    written = {"text": "An exact empty-reference interpretation. " * 20}
    if explicit_references:
        written["references"] = []
    assert submit(written)["status"] == "ACCEPTED"
    (row,) = answers()
    selected = _json(live, f"/api/activity/external?observation_id={row['observation_id']}")
    assert selected["accepted_answer"]["status"] == "AVAILABLE"
    assert selected["accepted_answer"]["contribution"] == {
        "text": written["text"],
        "references": [],
    }


@pytest.mark.parametrize("role", ["ANALYST", "CRO"])
def test_selected_typed_answer_reads_the_whole_accepted_task_contribution(
    answer_scene, monkeypatch, role
):
    """The typed readers reopen admitted input; this seam executes no scientific work. A row
    filed now is the lead's fact and reads through its operation return, which this direct
    seam has none of; a stored pre-FLOW-1 row with its HOOK credit still reads whole."""
    live, bundle, analyst, record, events, publish, _submit, answers = answer_scene
    prepared_id = UUID(bundle.submission["task_id"])
    prepared = live.session.task_control_registry.task(prepared_id)
    text = "Exact accepted typed prose is retained beyond the conversation summary. " * 18
    if role == "ANALYST":
        accepted = analyst.model_copy(update={"notes": text})
        submitted = SubmittedEvidenceAnalysis(
            prepared_task_id=prepared_id,
            packet_hash="a" * 64,
            analysis_policy_hash="b" * 64,
            decision_policy_hash="c" * 64,
            answer=accepted,
            actor_submission=seal_actor_submission(
                actor_kind=ActorKind.EXTERNAL_AUTOMATION,
                actor_id="synthetic-typed-answer",
                submission_hash=canonical_hash(accepted.model_dump(mode="json")),
            ),
        )
        request = seal_contract(
            AlternativeEvidenceRequest,
            "request_hash",
            ordered_entity_ids=("SYNTHETIC",),
            evidence_as_of=prepared.admitted_at,
            acquisition_deadline=prepared.admitted_at + timedelta(hours=1),
            evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
            source_policy=AlternativeEvidenceSourcePolicy(),
            ttl_seconds=3600,
            mode=AlternativeEvidenceMode.RECORDED,
        )
        admission = seal_contract(
            AlternativeEvidenceAdmission,
            "admission_hash",
            request_hash=request.request_hash,
            network_consent=False,
            admit_live_official=False,
            admit_model_review=False,
            admitted_at=prepared.admitted_at,
        )
        obligation = seal_contract(
            AlternativeEvidenceResearchObligation,
            "obligation_hash",
            question="Read synthetic retained evidence.",
            ordered_entity_ids=("SYNTHETIC",),
            evidence_as_of=prepared.admitted_at,
            approved_source_families=("SEC_EDGAR_OFFICIAL",),
            required_checks=("READ_SYNTHETIC",),
        )
        payload = {
            "purpose": "SUBMITTED_ANALYSIS",
            "request": request.model_dump(mode="json"),
            "admission": admission.model_dump(mode="json"),
            "obligation": obligation.model_dump(mode="json"),
            "resource_binding_hash": "d" * 64,
            "submitted_analysis": submitted.model_dump(mode="json"),
        }
        binding = {
            "role": "alternative_evidence.analyst",
            "task_id": str(prepared_id),
            "unit_id": None,
            "analysis_context_hash": "c" * 64,
        }

        def analysis_context(task_id, *, now, unit_id=None):
            assert task_id == prepared_id and unit_id is None
            assert now == prepared.admitted_at
            return None, {
                "analysis_context_hash": binding["analysis_context_hash"],
                "packet_hash": submitted.packet_hash,
                "analysis_policy_hash": submitted.analysis_policy_hash,
                "decision_policy_hash": submitted.decision_policy_hash,
            }

        monkeypatch.setattr(
            live.review, "evidence_task_adapter", SimpleNamespace(analysis_context=analysis_context)
        )
        task_kind, schema = ANALYST_TASK_KIND, ANALYST_INPUT_SCHEMA
    else:
        accepted = PortfolioReviewAnswer(summary=text)
        dossier = seal_contract(
            PortfolioReviewDossier,
            "dossier_hash",
            book_authority=BookAuthority.DEVELOPMENT_RESULT,
            report_hash="a" * 64,
            result_hash="b" * 64,
            issuer_scope_hash="c" * 64,
            exposure_projection_hash="d" * 64,
            registry_hash="e" * 64,
            analysis_publication_hash="f" * 64,
            analyst_brief_hash="a" * 64,
            cro_package_hash="b" * 64,
            obligation_hash="c" * 64,
            evidence_as_of=prepared.admitted_at,
            evidence_expires_at=prepared.admitted_at + timedelta(hours=1),
            issuers=(),
            findings=(),
            citations=(),
            coverage=PortfolioReviewCoverage(
                reviewed_ending_weight_coverage=0,
                reviewed_absolute_change_coverage=0,
                mapping_coverage=0,
                selected_issuer_coverage=0,
            ),
            mapping_failure_count=0,
            held_count=0,
            window_end_effective_n=0,
            claim_limits=("Synthetic read fixture.",),
            limitations=("No scientific work was executed.",),
            portfolio_report_link="synthetic-report",
        )
        live.review.artifacts.publish("cro-review-dossiers", dossier.dossier_hash, dossier)
        binding = {
            "role": "chief_risk_officer.reviewer",
            "dossier_hash": dossier.dossier_hash,
            "policy_hash": "c" * 64,
            "schema_hash": assessment_schema_hash(dossier),
        }
        bundle = bundle.model_copy(
            update={
                "role": "CRO",
                "submission": {
                    "operation": "CRO_REVIEW_SUBMIT",
                    "review_dossier_hash": binding["dossier_hash"],
                    "review_policy_hash": binding["policy_hash"],
                    "review_schema_hash": binding["schema_hash"],
                },
            }
        )
        payload = {
            "prepared_answer": accepted.model_dump(mode="json"),
            "submission_hash": answer_digest(accepted),
            "actor_kind": "EXTERNAL_AUTOMATION",
            "actor_id": "synthetic-typed-answer",
            "dossier_hash": binding["dossier_hash"],
            "decision_policy_hash": binding["policy_hash"],
        }
        task_kind, schema = CRO_TASK_KIND, CRO_INPUT_SCHEMA
        events[0][1]["role"] = "alphalattice_cro"
        record = record.model_copy(
            update={"agent_run": record.agent_run.model_copy(update={"role": "alphalattice_cro"})}
        )
    envelope = TaskInputEnvelope.create(
        task_kind=task_kind, input_schema_id=schema, payload=payload
    )
    goal = ResearchGoal.create(
        goal_kind="READ_RETAINED_ANSWER",
        input_hash=envelope.input_hash,
        deliverable_kind="SyntheticAcceptedAnswer",
        summary="Reopen one admitted answer.",
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=prepared.plan.workflow_definition_hash,
        verifier_catalog_hash=prepared.plan.verifier_catalog_hash,
        work_items=prepared.plan.work_items,
    )
    task = live.session.task_control_registry.admit(
        input_envelope=envelope, goal=goal, plan=plan, observed_at=prepared.admitted_at
    ).record
    bundle_key = str(canonical_hash(binding))
    record = record.model_copy(
        update={
            "bundle_key": bundle_key,
            "record_hash": answer_slot(bundle_key, record.number),
            "answer_digest": answer_digest(accepted),
        }
    )
    live.review.artifacts.publish(BUNDLE_CATEGORY, bundle.record_hash, bundle)
    live.review.artifacts.publish(ANSWER_CATEGORY, record.record_hash, record)
    publish()
    delivery = live.review.accepted_answer_delivery(
        record, accepted.model_dump(mode="json"), task_id=task.task_id, newly_filed=True
    )
    token = REQUEST_PROVENANCE.set(RequestProvenance(vendor="codex", session="fixture-parent"))
    try:
        body = {"disposition": "ADMITTED", "answer": {"accepted_delivery": delivery}}
        with live.session.mutation_gate.hold():
            receipt = live.operations.capture_accepted_answer_context(bundle, body)
        assert receipt is not None
        assert receipt.task_id == task.task_id and receipt.answer_reference == record.record_hash
        assert (
            live.operations.record_agent_answer(bundle, body, accepted_receipt=receipt)["status"]
            == "DELIVERED"
        )
    finally:
        REQUEST_PROVENANCE.reset(token)
    (row,) = answers()
    assert len(row["payload"]["summary"]) <= 500 and text not in row["payload"]["summary"]
    assert row["payload"]["subject"]["authorship_basis"] == "NOT_OBSERVED"
    current = _json(live, f"/api/activity/external?observation_id={row['observation_id']}")
    assert current["accepted_answer"]["status"] == "UNAVAILABLE"
    assert current["accepted_answer"]["missing"] == ["accepted_operation_return"]
    run = record.agent_run
    legacy = live.operations.declare_event(
        live.operations.observer,
        {
            "event_kind": row["payload"]["event_kind"],
            "producer_id": "pre-flow-1-native",
            "producer_session": row["payload"]["producer_session"],
            "producer_sequence": 1,
            "occurred_at": row["occurred_at"],
            "summary": row["payload"]["summary"],
            "subject": {
                **{k: v for k, v in row["payload"]["subject"].items() if k != "goal_id"},
                "native_agent_id": run.agent_id,
                "role": run.role,
                "authorship_basis": "HOOK",
            },
            "correlation_ids": row["correlation_ids"],
        },
    )
    assert legacy["status"] == "APPENDED"
    row = next(item for item in answers() if item["observation_id"] == legacy["observation_id"])
    selected = _json(live, f"/api/activity/external?observation_id={row['observation_id']}")
    assert selected["accepted_answer"]["status"] == "AVAILABLE"
    assert selected["accepted_answer"]["contribution"] == accepted.model_dump(mode="json")
    assert selected["accepted_answer"]["task_id"] == str(task.task_id)
    assert selected["accepted_answer"]["answer_reference"] == record.record_hash
    assert selected["accepted_answer"]["answer_digest"] == record.answer_digest
    assert selected["accepted_answer"]["bundle_reference"] == bundle.record_hash
    assert len(answers()) == 2


def test_generic_retry_repairs_original_acceptance_after_source_task_changes(
    answer_scene, tmp_path, monkeypatch
):
    """BEHAVIOUR: a missed projection repairs the same filed answer and original acceptance time."""
    live, task, _prepared, _bundle, authored, accepted_at, submit, answers = specialist_scene(
        answer_scene, tmp_path, monkeypatch, "RISK"
    )
    observer = live.operations.observer
    original = observer.admit_external_event
    first_goal, retry_goal = (
        open_goal(live, "Original specialist Goal"),
        open_goal(live, "Retry specialist Goal"),
    )

    def unavailable(document):
        raise OSError("SYNTHETIC-PROJECTION-ERROR")

    monkeypatch.setattr(observer, "admit_external_event", unavailable)
    first = submit(goal_id=first_goal)
    assert first["status"] == "ACCEPTED" and first["conversation"]["status"] == "UNAVAILABLE"
    original_record = live.review.artifacts.load(
        ANSWER_CATEGORY, first["answer_reference"], AgentAnswerRecord
    )
    live.session.task_control_registry.request_cancel(
        task_id=task.task_id,
        expected_task_hash=task.record_hash,
        observed_at=accepted_at + timedelta(minutes=1),
    )
    monkeypatch.setattr(observer, "admit_external_event", original)
    monkeypatch.setattr(live.review, "clock", lambda: accepted_at + timedelta(hours=1))
    repaired = submit(goal_id=retry_goal)
    assert repaired["status"] == "ACCEPTED" and repaired["conversation"]["status"] == "DELIVERED"
    assert repaired["answer_reference"] == first["answer_reference"]
    assert (
        live.review.artifacts.load(ANSWER_CATEGORY, repaired["answer_reference"], AgentAnswerRecord)
        == original_record
    )
    (row,) = answers()
    assert row["occurred_at"] == accepted_at.isoformat()
    assert row["payload"]["subject"]["goal_id"] == first_goal
    (goal_row,) = [
        row
        for row in live.operations.goals.store.attributed(UUID(first_goal))
        if row.get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
    ]
    assert goal_row["source_time_kind"] == "PRODUCT_ACCEPTED_AT"
    assert goal_row["occurred_at"] == accepted_at.isoformat()
    assert not any(
        row.get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
        for row in live.operations.goals.store.attributed(UUID(retry_goal))
    )
    assert authored["text"] in row["payload"]["summary"]
    assert submit()["conversation"]["status"] == "DELIVERED" and answers() == [row]
    changed = submit({"text": "A new interpretation after the assigned Task changed."})
    assert changed["status"] == "REFUSED"
    assert changed["failure_code"] == "agent_bundle.specialist_task_changed"
    assert answers() == [row]


def test_generic_prepare_requires_exact_assigned_task(answer_scene, tmp_path):
    """BEHAVIOUR: generic preparation has no role/time/book fallback for a missing Task."""
    live, *_ = answer_scene
    result = live.operations.execute(
        PortfolioResearchRequestDocument(
            operation="AGENT_BUNDLE_PREPARE",
            agent_role="RISK",
            bundle_directory=str(tmp_path / "missing-task-bundle"),
        ).to_operation_request(),
        caller="EXTERNAL_AUTOMATION",
    )
    assert result["status"] == "REFUSED"
    assert result["failure_code"] == "agent_bundle.specialist_task_required"


@pytest.mark.parametrize("basis", ["NOT_OBSERVED", "ROLE_CARD", "HOOK"])
def test_an_answer_of_any_recorded_basis_is_the_leads_fact_without_child_credit(
    answer_scene, basis
):
    """BEHAVIOUR (FLOW-1): the lead submitted it; no recorded basis, a stored HOOK credit
    among them, names a child in the conversation."""
    _live, bundle, _accepted, record, _events, _publish, submit, answers = answer_scene
    run = (
        AgentRun(host="codex", session_id="fixture-parent", basis="NOT_OBSERVED")
        if basis == "NOT_OBSERVED"
        else record.agent_run.model_copy(update={"basis": basis})
    )
    result = submit(filed=record.model_copy(update={"agent_run": run}))
    assert result["status"] == "ACCEPTED"
    assert result["conversation"]["status"] == "DELIVERED"
    assert result["conversation"]["bundle_reference"] == bundle.record_hash
    (row,) = answers()
    assert row["payload"]["subject"]["authorship_basis"] == "NOT_OBSERVED"
    assert row["payload"]["subject"]["native_agent_id"] == "fixture-parent"


def test_unfiled_and_correction_records_supply_no_accepted_delivery(answer_scene):
    """BEHAVIOUR: only stored terminal acceptance can supply a contribution."""
    live, _bundle, accepted, record, _events, _publish, _submit, _answers = answer_scene
    assert (
        live.review.accepted_answer_delivery(
            record, accepted.model_dump(mode="json"), task_id=uuid4()
        )
        is None
    )
    correction = record.model_copy(update={"verdict": AnswerVerdict.CORRECT})
    live.review.artifacts.publish(ANSWER_CATEGORY, correction.record_hash, correction)
    assert (
        live.review.accepted_answer_delivery(
            correction, accepted.model_dump(mode="json"), task_id=uuid4()
        )
        is None
    )


@pytest.mark.parametrize("operation", ["EVIDENCE_ANALYSIS_SUBMIT", "CRO_REVIEW_SUBMIT"])
def test_direct_answer_doors_keep_unknown_author_accepted_without_child_credit(
    answer_scene, monkeypatch, operation
):
    """BEHAVIOUR: each direct registered answer door uses the same optional return wiring."""
    live, bundle, accepted, record, _events, _publish, _submit, answers = answer_scene
    unknown = record.model_copy(update={"agent_run": None})
    live.review.artifacts.publish(ANSWER_CATEGORY, unknown.record_hash, unknown)
    body = {
        "disposition": "ADMITTED",
        "task_id": bundle.submission["task_id"],
        "answer": {"verdict": "ACCEPTED", "accepted_items": [1]},
    }
    if operation == "EVIDENCE_ANALYSIS_SUBMIT":
        request = {
            **bundle.submission,
            "analysis_answer": accepted.model_dump(mode="json"),
        }
        monkeypatch.setattr(live.review, "submit_analysis", lambda **kwargs: body)
    else:
        request = {
            "operation": operation,
            "review_dossier_hash": "a" * 64,
            "review_policy_hash": "b" * 64,
            "review_schema_hash": "c" * 64,
            "review_answer": {"risks": []},
        }
        monkeypatch.setattr(live.review, "submit_assessment", lambda **kwargs: body)
    result = live.operations.execute(
        PortfolioResearchRequestDocument.model_validate(request).to_operation_request(),
        caller="EXTERNAL_AUTOMATION",
    )
    assert result["disposition"] == "ADMITTED" and result["answer"]["verdict"] == "ACCEPTED"
    assert result["conversation"]["status"] == "UNAVAILABLE"
    assert result["conversation"]["reason"] == "native_bridge.accepted_delivery_unavailable"
    assert result["conversation"]["missing"] == ["accepted_delivery"]
    assert result["conversation"]["task_id"] == bundle.submission["task_id"]
    assert answers() == []
    assert (
        live.review.artifacts.load(ANSWER_CATEGORY, unknown.record_hash, AgentAnswerRecord)
        == unknown
    )


def test_done_preview_keeps_only_accepted_prose_and_names_drops(answer_scene):
    """BEHAVIOUR: a bounded final answer displays accepted text with its drop count."""
    _live, _bundle, _accepted, record, _events, publish, submit, answers = answer_scene
    final = record.model_copy(
        update={
            "verdict": AnswerVerdict.DONE,
            "problems": (AnswerProblem(item=2, text="SYNTHETIC-DROPPED-ITEM"),),
        }
    )
    publish()
    assert submit(filed=final, verdict="DONE")["status"] == "DONE"
    (row,) = answers()
    assert "DONE; 1 dropped" in row["payload"]["summary"]
    assert "Distinctive authored finding" in row["payload"]["summary"]
    assert "SYNTHETIC-DROPPED-ITEM" not in row["payload"]["summary"]


def test_observation_failure_preserves_acceptance_and_reports_missing_conversation(
    answer_scene, monkeypatch
):
    """BEHAVIOUR: an optional observer cannot undo an already accepted answer."""
    live, bundle, _accepted, record, _events, publish, submit, answers = answer_scene
    publish()
    original = live.operations.observer.admit_external_event

    def unavailable(document):
        raise OSError("SYNTHETIC-PRIVATE-ERROR")

    monkeypatch.setattr(live.operations.observer, "admit_external_event", unavailable)
    result = submit()
    assert result["status"] == "ACCEPTED"
    assert result["conversation"]["status"] == "UNAVAILABLE"
    assert result["conversation"]["reason"] == "native_bridge.accepted_delivery_failed"
    assert result["conversation"]["missing"] == ["accepted_answer_delivery"]
    assert result["conversation"]["task_id"] == bundle.submission["task_id"]
    assert result["conversation"]["bundle_reference"] == bundle.record_hash
    assert result["conversation"]["answer_reference"] == record.record_hash
    assert answers() == []
    assert "SYNTHETIC-PRIVATE-ERROR" not in str(result)
    assert (
        live.review.artifacts.load(ANSWER_CATEGORY, record.record_hash, AgentAnswerRecord) == record
    )
    monkeypatch.setattr(live.operations.observer, "admit_external_event", original)
    repaired = submit(disposition="REUSED_EXACT")
    assert repaired["status"] == "ACCEPTED" and repaired["conversation"]["status"] == "DELIVERED"
    assert submit(disposition="REUSED_EXACT")["conversation"]["status"] == "DELIVERED"


def test_delivery_read_failure_preserves_the_already_filed_acceptance(answer_scene, monkeypatch):
    """BEHAVIOUR: optional post-admission metadata reads cannot replace the accepted receipt."""
    live, bundle, _accepted, record, _events, publish, submit, answers = answer_scene
    publish()
    original = live.review.artifacts.load

    def unreadable(*args):
        raise OSError("SYNTHETIC-PRIVATE-READ-ERROR")

    monkeypatch.setattr(live.review.artifacts, "load", unreadable)
    result = submit()
    assert result["status"] == "ACCEPTED"
    assert result["conversation"]["status"] == "UNAVAILABLE"
    assert result["conversation"]["reason"] == "native_bridge.accepted_delivery_unavailable"
    assert result["conversation"]["missing"] == ["accepted_delivery"]
    assert result["conversation"]["task_id"] == bundle.submission["task_id"]
    assert result["conversation"]["bundle_reference"] == bundle.record_hash
    assert live.operations.observer_failures == 1
    assert "SYNTHETIC-PRIVATE-READ-ERROR" not in str(result)
    assert answers() == []
    monkeypatch.setattr(live.review.artifacts, "load", original)
    assert original(ANSWER_CATEGORY, record.record_hash, AgentAnswerRecord) == record


def test_cro_accepted_risk_and_publication_retry_keep_the_same_stored_author(answer_scene):
    """BEHAVIOUR: CRO's own typed risk fields and exact publication Task produce one exchange."""
    live, bundle, _accepted, record, events, publish, _submit, answers = answer_scene
    accepted = PortfolioReviewAnswer.model_validate(
        {
            "risks": [
                {
                    "findings": ["F1"],
                    "why": "Distinctive authored risk: synthetic liquidity pressure.",
                    "severity": "HIGH",
                    "confidence": "SUPPORTED",
                    "recommendation": "Inspect liquidity.",
                }
            ]
        }
    )
    record = record.model_copy(
        update={
            "answer_digest": answer_digest(accepted),
            "agent_run": record.agent_run.model_copy(update={"role": "alphalattice_cro"}),
        }
    )
    live.review.artifacts.publish(ANSWER_CATEGORY, record.record_hash, record)
    events[0][1]["role"] = "alphalattice_cro"
    publish()
    task = live.session.task_control_registry.task(UUID(bundle.submission["task_id"]))
    delivery = live.review.accepted_answer_delivery(
        record, accepted.model_dump(mode="json"), input_hash=task.input.input_hash
    )
    assert delivery["task_id"] == str(task.task_id)
    body = {"disposition": "ADMITTED", "answer": {"accepted_delivery": delivery}}
    token = REQUEST_PROVENANCE.set(RequestProvenance(vendor="codex", session="fixture-parent"))
    try:
        accepted_bundle = bundle.model_copy(update={"role": "CRO"})
        with live.session.mutation_gate.hold():
            receipt = live.operations.capture_accepted_answer_context(accepted_bundle, body)
        assert receipt is not None
        assert receipt.task_id == task.task_id and receipt.answer_reference == record.record_hash
        assert (
            live.operations.record_agent_answer(accepted_bundle, body, accepted_receipt=receipt)[
                "status"
            ]
            == "DELIVERED"
        )
        (row,) = answers()
        assert accepted.risks[0].why in row["payload"]["summary"]
        assert row["payload"]["subject"]["bundle_role"] == "CRO"
        body["disposition"] = "REUSED_EXACT"
        assert (
            live.operations.record_agent_answer(accepted_bundle, body, accepted_receipt=receipt)[
                "status"
            ]
            == "DELIVERED"
        )
        assert answers() == [row]
    finally:
        REQUEST_PROVENANCE.reset(token)
    assert live.review.accepted_answer_delivery(
        record, accepted.model_dump(mode="json"), input_hash="f" * 64
    ) == {
        "status": "UNAVAILABLE",
        "reason": "native_bridge.accepted_answer_not_recorded",
    }


@pytest.mark.parametrize("fail_before_binding", [False, True])
def test_goal_retry_keeps_original_binding_or_stays_unbound(
    answer_scene, monkeypatch, fail_before_binding
):
    """BEHAVIOUR: retry cannot borrow its caller's newer Goal or infer an original one."""
    live, _bundle, _accepted, _record, _events, publish, submit, answers = answer_scene

    first, second = open_goal(live, "Original Goal"), open_goal(live, "Retry Goal")
    publish()
    owner, field = (
        (live.review.artifacts, "load")
        if fail_before_binding
        else (live.operations.observer, "admit_external_event")
    )
    original = getattr(owner, field)

    def fail(*args):
        raise OSError("SYNTHETIC-PROJECTION-UNAVAILABLE")

    monkeypatch.setattr(owner, field, fail)
    assert submit(goal_id=first)["conversation"]["status"] == "UNAVAILABLE"
    monkeypatch.setattr(owner, field, original)
    repaired = submit(goal_id=second, newly_filed=False, disposition="REUSED_EXACT")
    assert repaired["status"] == "ACCEPTED" and repaired["conversation"]["status"] == "DELIVERED"
    (row,) = answers()
    subject = row["payload"]["subject"]
    original_conversation = [
        entry
        for entry in live.operations.goals.store.attributed(UUID(first))
        if entry.get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
    ]
    retry_conversation = [
        entry
        for entry in live.operations.goals.store.attributed(UUID(second))
        if entry.get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
    ]
    assert retry_conversation == []
    if fail_before_binding:
        assert "goal_id" not in subject and original_conversation == []
    else:
        assert subject["goal_id"] == first
        (entry,) = original_conversation
        assert entry["input_channel"] == "PRODUCT_ACCEPTED_ANSWER"
        assert entry["source_time_kind"] == "TASK_ADMISSION"
        assert entry["occurred_at"] == row["occurred_at"]
        for name in ("submitted_by", "answer_reference", "bundle_reference", "authorship_basis"):
            assert entry[name] == subject[name]
    assert (
        submit(goal_id=second, newly_filed=False, disposition="REUSED_EXACT")["conversation"][
            "status"
        ]
        == "DELIVERED"
    )
    assert answers() == [row]


def test_an_answer_from_a_session_that_bound_itself_is_delivered(product_answer_scene):
    """regression (AX's REACCEPT, 2026-10-08 00:21): five accepted Analyst answers each read
    `native_bridge.accepted_delivery_failed`, retries too: no configured project held a binding
    of their Session. The submission's own request binds its Session first (AUTOBIND), so the
    accepted answer is filed in its Conversation, and a retry is the same row."""
    from alphalattice.interface.local_application.native_setup import autobind_root

    scene = product_answer_scene
    first, _row = scene.submit(session="self-bound-lead", vendor="codex")
    assert first["status"] == "ACCEPTED"
    assert first["conversation"]["status"] == "DELIVERED", first["conversation"]
    root = autobind_root(scene.live.workspace)
    assert NativeResearchBinding.read(root, session=("codex", "self-bound-lead")) is not None
    retry, _row = scene.submit(session="self-bound-lead", vendor="codex")
    assert retry["conversation"]["observation_id"] == first["conversation"]["observation_id"]
