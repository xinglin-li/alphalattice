"""The owner returns a filed specialist contribution through its parent to Team (V691)."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from alphalattice.control.observation_runtime.ledger import ObservationLedger
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
from alphalattice.interface.local_application.native_bridge import BINDING_NAME, JUDGMENT_ROLES
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


@pytest.mark.parametrize("path_assignment", [False, True])
def test_accepted_authored_contribution_has_exact_child_parent_and_one_retry(
    answer_scene, path_assignment
):
    """BEHAVIOUR: acceptance projects authored prose, exact attribution and stable retry."""
    live, bundle, accepted, record, events, publish, submit, answers = answer_scene
    if path_assignment:
        events[0][1].update(
            native_agent_path="/root/analyst", native_agent_path_basis="CODEX_SESSION_META"
        )
        events[1][1]["recipient_id"] = "/root/analyst"
    publish()
    result = submit()
    assert result["status"] == "ACCEPTED" and result["conversation"]["status"] == "DELIVERED"
    (row,) = answers()
    subject = row["payload"]["subject"]
    assert row["authority"] == "AGENT_PROPOSAL"
    assert accepted.findings[0].summary in row["payload"]["summary"]
    assert row["payload"]["summary"].startswith("Product accepted deliverable (ACCEPTED):")
    assert (subject["native_agent_id"], subject["submitted_by"], subject["recipient_id"]) == (
        "fixture-child",
        "fixture-parent",
        "fixture-parent",
    )
    assert subject["reply_to"] == "fixture-assignment"
    assert subject["bundle_reference"] == bundle.record_hash
    assert subject["answer_reference"] == record.record_hash
    assert subject["source_time_kind"] == "TASK_ADMISSION"
    assert (
        row["occurred_at"]
        == live.session.task_control_registry.task(
            UUID(subject["reference"])
        ).admitted_at.isoformat()
    )
    # The next trial's apparent author cannot replace the already filed author's credit.
    retry = record.model_copy(
        update={
            "number": 2,
            "record_hash": answer_slot(record.bundle_key, 2),
            "agent_run": record.agent_run.model_copy(update={"agent_id": "different-child"}),
        }
    )
    assert submit(retry=retry)["conversation"]["status"] == "DELIVERED"
    assert answers() == [row]
    assert submit(disposition="REUSED_EXACT")["conversation"]["status"] == "DELIVERED"
    assert answers() == [row]


def specialist_scene(answer_scene, tmp_path, monkeypatch, role, *, proved=True):
    """Prepare and submit through the real shared generic doors on one synthetic Task."""
    live, original, _accepted, _record, events, publish, _submit, answers = answer_scene
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
    if proved:
        publish()
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
@pytest.mark.parametrize("proved", [False, True])
def test_generic_specialist_doors_seal_exact_references_without_scientific_admission(
    answer_scene, tmp_path, monkeypatch, role, proved
):
    """BEHAVIOUR: all five roles keep actual text, source Task and exact HOOK-only credit."""
    live, task, prepared, bundle, authored, accepted_at, submit, answers = specialist_scene(
        answer_scene, tmp_path, monkeypatch, role, proved=proved
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
    if proved:
        assert result["conversation"]["status"] == "DELIVERED"
        (row,) = answers()
        subject = row["payload"]["subject"]
        assert len(row["payload"]["summary"]) <= 500
        assert row["payload"]["summary"] != authored["text"]
        assert subject["native_agent_id"] == "fixture-child"
        assert subject["role"] == JUDGMENT_ROLES[role][0]
        assert subject["submitted_by"] == "fixture-parent"
        assert subject["reference"] == str(task.task_id)
        assert subject["answer_reference"] == record.record_hash
        assert subject["bundle_reference"] == bundle.record_hash
        assert subject["source_time_kind"] == "PRODUCT_ACCEPTED_AT"
        assert row["occurred_at"] == accepted_at.isoformat()
        lookups = []
        read_observation = ObservationLedger.read

        def exact_read(ledger, observation_id):
            lookups.append(observation_id)
            return read_observation(ledger, observation_id)

        def refuse_page_scan(*_args, **_kwargs):
            pytest.fail("A selected answer must not scan external observation pages.")

        with monkeypatch.context() as selected_read:
            selected_read.setattr(ObservationLedger, "read", exact_read)
            selected_read.setattr(ObservationLedger, "observations_of_kind", refuse_page_scan)
            selected = _json(live, f"/api/activity/external?observation_id={row['observation_id']}")
        assert lookups == [row["observation_id"]]
        assert selected["read_cost"]["observations"] == 1
        assert selected["observation_id"] == row["observation_id"]
        assert "items" not in selected
        detail = selected["accepted_answer"]
        assert detail["status"] == "AVAILABLE"
        assert detail["observation_id"] == row["observation_id"]
        assert detail["native_session_id"] == "fixture-parent"
        assert detail["native_agent_id"] == "fixture-child"
        assert detail["submitted_by"] == "fixture-parent"
        assert detail["task_id"] == str(task.task_id)
        assert detail["answer_reference"] == record.record_hash
        assert detail["answer_digest"] == record.answer_digest
        assert detail["bundle_reference"] == bundle.record_hash
        assert detail["contribution"] == {
            "text": authored["text"],
            "references": authored["references"],
        }
        artifact_reads = live.review.artifacts.read_count
        page = _json(live, "/api/activity/external")
        assert "accepted_answer" not in page
        assert live.review.artifacts.read_count == artifact_reads
        assert page["epoch"] == selected["epoch"]
        assert answers() == [row]
    else:
        assert result["recorded_agent"]["basis"] == "NOT_OBSERVED"
        assert result["conversation"]["status"] == "UNAVAILABLE"
        assert result["conversation"]["reason"] == "native_bridge.start_not_observed"
        assert result["conversation"]["missing"] == ["native_subagent_start", "exact_assignment"]
        assert result["conversation"]["task_id"] == str(task.task_id)
        assert result["conversation"]["bundle_reference"] == bundle.record_hash
        assert result["conversation"]["session_id"] == "fixture-parent"
        assert answers() == []


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
    """The typed readers reopen admitted input; this seam executes no scientific work."""
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
    history = live.operations.observer.read_external(ExternalActivityReadQuery())["items"]
    token = REQUEST_PROVENANCE.set(RequestProvenance(vendor="codex", session="fixture-parent"))
    try:
        assert (
            live.operations.record_agent_answer(
                bundle,
                {"disposition": "ADMITTED", "answer": {"accepted_delivery": delivery}},
                history,
            )["status"]
            == "DELIVERED"
        )
    finally:
        REQUEST_PROVENANCE.reset(token)
    (row,) = answers()
    assert len(row["payload"]["summary"]) <= 500 and text not in row["payload"]["summary"]
    selected = _json(live, f"/api/activity/external?observation_id={row['observation_id']}")
    assert selected["accepted_answer"]["status"] == "AVAILABLE"
    assert selected["accepted_answer"]["contribution"] == accepted.model_dump(mode="json")
    assert selected["accepted_answer"]["task_id"] == str(task.task_id)
    assert selected["accepted_answer"]["answer_reference"] == record.record_hash
    assert selected["accepted_answer"]["answer_digest"] == record.answer_digest
    assert selected["accepted_answer"]["bundle_reference"] == bundle.record_hash
    assert answers() == [row]


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


@pytest.mark.parametrize(
    "case", ["unknown", "role_card", "wrong_bundle", "wrong_child", "ambiguous"]
)
def test_unproved_answer_is_accepted_without_a_child_conversation_credit(answer_scene, case):
    """BEHAVIOUR: absent, mismatched or conflicting authorship never guesses a child."""
    _live, bundle, _accepted, record, events, publish, submit, answers = answer_scene
    if case in {"unknown", "role_card"}:
        run = (
            AgentRun(host="codex", session_id="fixture-parent", basis="NOT_OBSERVED")
            if case == "unknown"
            else record.agent_run.model_copy(update={"basis": "ROLE_CARD"})
        )
        record = record.model_copy(update={"agent_run": run})
    elif case == "wrong_bundle":
        events[1][1]["reference"] = "f" * 64
    elif case == "wrong_child":
        events[1][1]["recipient_id"] = "different-child"
    else:
        events.append(
            (
                "NATIVE_COORDINATION_MESSAGE",
                {
                    **events[1][1],
                    "message_id": "other-assignment",
                    "recipient_id": "different-child",
                },
            )
        )
    publish()
    result = submit(filed=record)
    assert result["status"] == "ACCEPTED"
    assert result["conversation"]["status"] == "UNAVAILABLE"
    assert (
        result["conversation"]["reason"]
        == {
            "unknown": "native_bridge.accepted_author_not_observed",
            "role_card": "native_bridge.accepted_author_not_observed",
            "wrong_bundle": "native_bridge.assignment_not_observed",
            "wrong_child": "native_bridge.start_not_observed",
            "ambiguous": "native_bridge.assignment_ambiguous",
        }[case]
    )
    assert result["conversation"]["session_id"] == "fixture-parent"
    assert result["conversation"]["bundle_reference"] == bundle.record_hash
    assert result["conversation"]["answer_reference"] == record.record_hash
    assert result["conversation"]["task_id"] == bundle.submission["task_id"]
    assert answers() == []


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


def test_failed_history_read_is_unavailable_and_preserves_filed_acceptance(
    answer_scene, monkeypatch
):
    """BEHAVIOUR: a failed optional ledger read cannot become an empty record or child credit."""
    live, bundle, _accepted, record, _events, publish, submit, answers = answer_scene
    publish()
    before = live.operations.observer.read_external(ExternalActivityReadQuery())["items"]
    original = live.operations.observer.read_external

    def unavailable(query):
        raise OSError("SYNTHETIC-PRIVATE-HISTORY-ERROR")

    with monkeypatch.context() as history:
        history.setattr(live.operations.observer, "read_external", unavailable)
        result = submit()
    assert result["status"] == "ACCEPTED"
    assert result["conversation"]["status"] == "UNAVAILABLE"
    assert result["conversation"]["reason"] == "native_bridge.history_unavailable"
    assert result["conversation"]["missing"] == ["native_history"]
    assert result["conversation"]["session_id"] == "fixture-parent"
    assert result["conversation"]["bundle_reference"] == bundle.record_hash
    assert result["conversation"]["task_id"] == bundle.submission["task_id"]
    assert "SYNTHETIC-PRIVATE-HISTORY-ERROR" not in str(result)
    assert (
        live.review.artifacts.load(ANSWER_CATEGORY, record.record_hash, AgentAnswerRecord) == record
    )
    assert original(ExternalActivityReadQuery())["items"] == before
    assert answers() == []


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
    history = live.operations.observer.read_external(ExternalActivityReadQuery())["items"]
    body = {"disposition": "ADMITTED", "answer": {"accepted_delivery": delivery}}
    token = REQUEST_PROVENANCE.set(RequestProvenance(vendor="codex", session="fixture-parent"))
    try:
        assert (
            live.operations.record_agent_answer(
                bundle.model_copy(update={"role": "CRO"}), body, history
            )["status"]
            == "DELIVERED"
        )
        (row,) = answers()
        assert accepted.risks[0].why in row["payload"]["summary"]
        assert row["payload"]["subject"]["role"] == "alphalattice_cro"
        body["disposition"] = "REUSED_EXACT"
        assert (
            live.operations.record_agent_answer(
                bundle.model_copy(update={"role": "CRO"}), body, history
            )["status"]
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
