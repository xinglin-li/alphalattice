"""Evidence and CRO bindings, submission and recovery through the real HTTP Host."""

from __future__ import annotations

import json
import re
import shlex
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.evidence_review_application import (
    ANSWER_CATEGORY,
    PortfolioReviewCommand,
    agent_answer_result,
)
from alphalattice.control.product_host.composition.evidence_review_bundles import (
    EvidenceReviewBundles,
)
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceArtifact,
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    BROAD_FEATURE_PACKAGE,
    PRODUCT_EVIDENCE_ROOT_KEY,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewPublication,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    PortfolioReviewTaskAdapter,
)
from alphalattice.protocols.actor_execution.answers import AgentAnswerRecord, answer_slot
from alphalattice.protocols.actor_execution.bundles import bundle_slot
from tests.alternative_evidence_desk.planted_corpus import _NOW
from tests.alternative_evidence_desk.review_dossiers import controlled_answer
from tests.alternative_evidence_desk.review_http_support import (
    _raise_interruption,
    _Service,
    _service,
    build_authority,
    start_service,
)
from tests.alternative_evidence_desk.review_package import _package
from tests.structural.source_shape_samples import bundle_refusal_declarations

ONE_UNIT = "u01"
"""The harness book's only unit: every book is prepared as a coverage run (C2)."""


def test_every_bundle_manual_refusal_registers_its_words_and_way_on() -> None:
    """OP4: the real door guard accepts every manual way the bundle owner offers."""

    from alphalattice.control.product_host.composition.evidence_review_bundles import (
        PACKET_SELECTOR_CODES,
        agent_bundle_refusal,
    )
    from alphalattice.interface.local_application.cli_contract import (
        refusal_problem,
        refusal_words,
        worded_refusal,
    )

    # Census the owner's literal namespace declarations, then exercise its public
    # refusal entry and the real door guard; no private import or patched state.
    declarations = bundle_refusal_declarations()
    for code, message in {k: v for words in declarations for k, v in words.items()}.items():
        variants = (code, f"{code}:{ONE_UNIT}") if code in PACKET_SELECTOR_CODES else (code,)
        for variant in variants:
            answer = agent_bundle_refusal(variant)
            words = refusal_words(variant)
            assert words.get("detail") == message, variant
            assert words.get("next_action") == answer["next_action"], variant
            assert refusal_problem(worded_refusal(answer)) is None, variant


def _packet(body: dict[str, Any]) -> dict[str, str]:
    """The one packet request a body hands out, selected as a client selects
    it: by its operation, whatever key names the unit."""

    (request,) = [
        value
        for value in body["next_requests"].values()
        if isinstance(value, dict) and value.get("operation") == "EVIDENCE_PACKET"
    ]
    return request


@pytest.fixture
def service(tmp_path: Path, http_book) -> Iterator[_Service]:
    yield from _service(tmp_path, *http_book, with_runtime=True, with_actor=True)


@pytest.mark.parametrize("via_http", (False, True))
def test_native_analysis_cli_prepares_submits_and_reviews_without_managed_actors(
    service, tmp_path, capsys, via_http
):

    from alphalattice.interface.local_application.cli import main
    from alphalattice.interface.local_application.client import LocalResearchClient
    from tests.alternative_evidence_desk.planted_corpus import _CitingActor

    review_risks = service.review.review_actor.risks
    adapter = service.review.evidence_task_adapter
    topics = adapter.resources.analysis_actor.topics
    adapter.resources = replace(adapter.resources, analysis_actor=None)
    service.review.model_authority_admitted = False
    service.review.review_actor = None
    client = LocalResearchClient(service.session.workspace)
    selected = {"result_hash": service.result_hash()}
    count = len(service.registry.tasks())
    assert (
        service.post("/api/evidence-refresh", selected)["disposition"]
        == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED"
    )
    writes = service.review.artifacts.write_count
    # The section's state before any work: what is missing is evidence, not a
    # credential, and the step out of it is the native preview -- the same
    # request the client sends next.
    section = service.evidence_cro()
    assert section["state"] == "AWAITING_ALTERNATIVE_EVIDENCE"
    assert section["available_actions"] == []
    assert section["next_requests"] == {"preview": {"operation": "EVIDENCE_PREVIEW", **selected}}
    preview = client.request(section["next_requests"]["preview"])
    assert service.get("/api/evidence/preview?" + urllib.parse.urlencode(selected)) == preview
    assert preview["status"] == "EVIDENCE_PREPARATION_READY", preview
    assert preview["managed_model_required"] is False
    assert preview["recorded_candidate_count"] > 0
    assert (len(service.registry.tasks()), service.review.artifacts.write_count) == (count, writes)
    prepared = (
        service.post("/api/evidence/prepare", selected)
        if via_http
        else client.request(preview["next_requests"]["prepare"])
    )
    assert prepared["disposition"] == "ADMITTED"
    service.drain()
    source_id = UUID(prepared["task_id"])
    assert service.registry.task(source_id).lifecycle is TaskLifecycle.SUCCEEDED
    assert len(service.registry.tasks()) == count + 1
    # The same preparation asked for again is the completed one: no Task, no
    # canonicalization, no corpus pass, and the same packet to continue from.
    again = (
        service.post("/api/evidence/prepare", selected)
        if via_http
        else client.request(preview["next_requests"]["prepare"])
    )
    assert again["disposition"] == "REUSED_EXACT" and again["task_id"] == str(source_id)
    assert _packet(again) == _packet(prepared)
    assert len(service.registry.tasks()) == count + 1
    assert adapter.runtime.retrieval.passage_embedding_pass_count == 1
    # With a packet prepared and no analysis, the section names that packet
    # and hands out the bound read of it; no state re-verifies or embeds.
    section = service.evidence_cro()
    assert section["state"] == "ANALYST_PACKET_PREPARED"
    assert section["task_id"] == str(source_id)
    assert section["evidence_as_of"] == prepared["evidence_as_of"]
    assert _packet(section) == _packet(prepared)
    assert set(section["next_requests"]) == {f"packet_{ONE_UNIT}", "preview"}
    assert adapter.runtime.retrieval.passage_embedding_pass_count == 1
    exported = client.request(_packet(section))
    assert (
        service.get(
            "/api/evidence/packet?"
            + urllib.parse.urlencode(
                {**selected, "task_id": str(source_id), "evidence_unit_id": ONE_UNIT}
            )
        )
        == exported
    )
    assert exported.get("status") == "EVIDENCE_ANALYST_PACKET_READY", exported
    assert exported["external_host_usage"] == "UNAVAILABLE"
    assert exported["submission_template"]["result_hash"] == selected["result_hash"]
    assert adapter.resources.analysis_actor is None
    packet = adapter.prepared_packet(source_id, now=service.review.clock(), unit_id=ONE_UNIT)
    answer = _CitingActor(topics=topics)(packet=packet).answer.model_dump(mode="json")
    document = {**exported["submission_template"], "analysis_answer": answer}
    writes = service.review.artifacts.write_count
    bad = client.request({**document, "analysis_context_hash": "a" * 64})
    assert "external_analysis_binding_changed" in bad["refused"]
    status, http_bad = service.request(
        "/api/evidence/analysis",
        method="POST",
        payload={
            **selected,
            "task_id": str(source_id),
            "evidence_unit_id": ONE_UNIT,
            "analysis_context_hash": "a" * 64,
            "analysis_answer": answer,
        },
    )
    assert status == 400 and "external_analysis_binding_changed" in http_bad["refused"]
    invented = json.loads(json.dumps(answer))
    invented["findings"][0]["cite"] = ["S999"]
    refused = client.request({**document, "analysis_answer": invented})
    assert refused["status"] == "CORRECT" and refused["task_id"] is None
    returned = refused["answer"]
    assert returned["problems"] == [{"item": 1, "text": "S999 is not an excerpt of this bundle."}]
    assert returned["accepted_items"] == list(range(2, len(answer["findings"]) + 1))
    assert (returned["number"], returned["rounds_left"]) == (1, 2)
    assert "Item 1: S999 is not an excerpt of this bundle." in returned["message"]
    assert len(service.registry.tasks()) == count + 1
    # Only the answer's durable record is written; nothing is admitted.
    assert service.review.artifacts.write_count == writes + 1
    if via_http:
        submitted = service.post(
            "/api/evidence/analysis", {k: v for k, v in document.items() if k != "operation"}
        )
    else:
        # The lead writes the Analyst's bundle; the Analyst sends its answer
        # file for that bundle with the one command and reads the verdict.
        workspace = ["--workspace", str(service.session.workspace), "--view", "full"]
        prepare = [
            *("bundle", "prepare", "--role", "ANALYST"),
            *("--task", str(source_id), "--unit", ONE_UNIT),
        ]
        directory = ["--dir", str(tmp_path / "analyst-bundle")]
        selector = ["--result", selected["result_hash"]]
        assert main([*workspace, *prepare, *directory, *selector], serve=lambda _: 99) == 0
        bundle = json.loads(capsys.readouterr().out)["data"]
        read = [item["name"] for item in bundle["files"]]
        Path(bundle["answer_file"]).write_text(
            json.dumps({**answer, "read": read}), encoding="utf-8"
        )
        # The answer's Task is pending until it runs, so the submit exits 3 (OP3, V580).
        assert main(["--view", "full", *bundle["submit_arguments"]], serve=lambda _: 99) == 3
        verdict = json.loads(capsys.readouterr().out)["data"]
        assert verdict["status"] == "ACCEPTED" and verdict["agent_role"] == "ANALYST"
        assert verdict["task_lifecycle"] == "QUEUED"
        assert verdict["receipt"]["accepted_items"] == len(answer["findings"])
        submitted = {"task_id": verdict["receipt"]["task_id"]}
    service.drain()
    task_id = UUID(submitted["task_id"])
    task = service.registry.task(task_id)
    assert task.lifecycle is TaskLifecycle.SUCCEEDED, task.failure_code
    view = adapter.published_analysis(task_id, now=service.review.clock())
    assert view.lineage.analyst_receipt.actor_submission.actor_kind.value == (
        "HUMAN" if via_http else "EXTERNAL_AUTOMATION"
    )
    assert view.lineage.analyst_receipt.model_call_count == 0
    before = len(service.registry.tasks()), service.review.artifacts.write_count
    reused = (
        service.post(
            "/api/evidence/analysis", {k: v for k, v in document.items() if k != "operation"}
        )
        if via_http
        else client.request(document)
    )
    assert reused["disposition"] == "REUSED_EXACT" and reused["task_id"] is None
    assert reused["analysis_publication_hash"] == view.publication.publication_hash
    assert (len(service.registry.tasks()), service.review.artifacts.write_count) == before
    # A current analysis and no review: the dossier is the step, and the
    # managed review is the one thing the missing credential withholds.
    section = service.evidence_cro()
    assert section["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    assert section["available_actions"] == []
    assert section["next_requests"] == {"dossier": {"operation": "CRO_REVIEW_DOSSIER", **selected}}
    assert section["evidence_as_of"] == prepared["evidence_as_of"]
    dossier = client.request(section["next_requests"]["dossier"])
    assert service.get("/api/cro/dossier?" + urllib.parse.urlencode(selected)) == dossier
    assert dossier["status"] == "CRO_DOSSIER_READY"
    assessment_request = {
        **dossier["submission_template"],
        "review_answer": controlled_answer(dossier["dossier"], *review_risks).model_dump(
            mode="json"
        ),
    }
    assessed = (
        service.post(
            "/api/cro/assessment", {k: v for k, v in assessment_request.items() if k != "operation"}
        )
        if via_http
        else client.request(assessment_request)
    )
    service.drain()
    assert service.registry.task(UUID(assessed["task_id"])).lifecycle is TaskLifecycle.SUCCEEDED
    section = service.evidence_cro()
    assert section["state"] == "REVIEW_PUBLISHED"
    assert list(section["next_requests"]) == ["export"]
    published = section["review_publication_hash"]
    assert section["next_requests"]["export"] == {
        "operation": "EVIDENCE_CRO_EXPORT",
        **selected,
        "review_publication_hash": published,
    }
    exported_review = client.request(section["next_requests"]["export"])
    assert exported_review["review"]["publication"]["publication_hash"] == published
    # The same review pinned by handle reads back exactly and grants nothing.
    pinned = service.evidence_cro(review_publication_hash=published)
    assert pinned["state"] == "REVIEW_PUBLISHED"
    assert pinned["available_actions"] == []
    assert pinned["next_requests"] == section["next_requests"]
    assert adapter.runtime.retrieval.passage_embedding_pass_count == 1


def test_a_captured_preparation_is_reused_after_time_moves_on(tmp_path: Path, http_book) -> None:
    """A captured preparation is reused after time moves on."""

    from alphalattice.interface.local_application.client import LocalResearchClient

    workspace, report = http_book
    authority = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    now = [_NOW]
    service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
    try:
        client = LocalResearchClient(service.session.workspace)
        selected = {"result_hash": service.result_hash()}
        preview = client.request({"operation": "EVIDENCE_PREVIEW", **selected})
        captured = preview["next_requests"]["prepare"]
        assert captured["evidence_as_of"] == _NOW.isoformat()
        assert captured["preparation_binding_hash"] == preview["preparation_binding_hash"]
        assert preview["prepared_task_id"] is None
        assert not [
            value
            for value in preview["next_requests"].values()
            if value.get("operation") == "EVIDENCE_PACKET"
        ]
        count = len(service.registry.tasks())
        prepared = client.request(captured)
        assert prepared["disposition"] == "ADMITTED"
        assert prepared["evidence_as_of"] == _NOW.isoformat()
        service.drain()
        task_id = prepared["task_id"]
        adapter = service.review.evidence_task_adapter
        assert adapter.runtime.retrieval.passage_embedding_pass_count == 1

        # Two hours later the same captured intent is the completed preparation:
        # no Task, no corpus pass, and the packet it names still answers as of
        # the original cutoff.
        now[0] = _NOW + timedelta(hours=2)
        again = client.request(captured)
        assert again["disposition"] == "REUSED_EXACT" and again["task_id"] == task_id
        assert again["evidence_as_of"] == _NOW.isoformat()
        assert _packet(again)["task_id"] == task_id
        assert len(service.registry.tasks()) == count + 1
        assert adapter.runtime.retrieval.passage_embedding_pass_count == 1
        packet = client.request(_packet(again))
        assert packet["status"] == "EVIDENCE_ANALYST_PACKET_READY"
        later_preview = client.request({"operation": "EVIDENCE_PREVIEW", **selected})
        assert later_preview["prepared_task_id"] is None, "a new cutoff is a new intent"
        assert later_preview["next_requests"]["prepare"]["evidence_as_of"] == now[0].isoformat()

        # A binding the workspace no longer describes is refused by name and
        # points back to the preview; nothing is admitted.
        moved = client.request({**captured, "preparation_binding_hash": "a" * 64})
        assert moved.get("disposition") == "REFUSED_PREPARATION_BINDING_CHANGED", moved
        assert moved["failure_code"] == "alternative_evidence.preparation_binding_changed"
        assert moved["next_requests"]["preview"]["operation"] == "EVIDENCE_PREVIEW"
        assert "task_id" not in moved
        # A bare request an hour later is a new preparation as of that moment:
        # its own Task, cutoff and packet -- over the same eligible bytes, so
        # the generation is the one already committed, nothing is embedded,
        # and the selection reused whole from the sealed receipt builds and
        # proves no index (record section Y).
        now[0] = _NOW + timedelta(hours=3)
        bare = client.request({"operation": "EVIDENCE_PREPARE", **selected})
        assert bare["disposition"] == "ADMITTED" and bare["task_id"] != task_id
        assert bare["evidence_as_of"] == now[0].isoformat()
        service.drain()
        assert adapter.runtime.retrieval.passage_embedding_pass_count == 1
        assert adapter.runtime.retrieval.generation_reuse_count == 0
        assert adapter.runtime.generation_builds_avoided == 1
        assert len(service.registry.tasks()) == count + 2
        # A preparation sealed under a canonicalization binding this code has
        # superseded is history, not the completed preparation of the same
        # intent: the packet is not offered, and a new request prepares its
        # own artifacts instead of copying the superseded ones -- which every
        # later analysis would have refused as a binding mismatch.
        current_binding = adapter.runtime.document_binding_hash
        adapter._completed_units.clear()
        adapter._absent_units.clear()
        adapter.runtime.document_binding_hash = "b" * 64
        try:
            with pytest.raises(ValueError, match="preparation_superseded"):
                adapter.preparation_window(service.registry.task(UUID(task_id)), ONE_UNIT)
            # The packet read refuses by the same name, never as a lineage
            # mismatch (seen on the retained journey fixture at a later tree).
            with pytest.raises(ValueError, match="preparation_superseded"):
                adapter.prepared_packet(UUID(task_id), now=now[0], unit_id=ONE_UNIT)
            first = service.registry.task(UUID(task_id))
            (unit,) = adapter.run_of(first).units
            assert adapter.completed_unit(unit.preparation_intent_hash) is None
            superseded = client.request({"operation": "EVIDENCE_PREPARE", **selected})
            assert superseded["disposition"] == "ADMITTED" and superseded["task_id"] != task_id
            service.drain()
            fresh = service.registry.task(UUID(superseded["task_id"]))
            assert fresh.lifecycle is TaskLifecycle.SUCCEEDED
            assert adapter._document_set(fresh, ONE_UNIT).canonicalization_binding_hash == "b" * 64
            first_set = adapter._document_set(service.registry.task(UUID(task_id)), ONE_UNIT)
            assert adapter._document_set(fresh, ONE_UNIT).document_set_hash != (
                first_set.document_set_hash
            )
        finally:
            adapter.runtime.document_binding_hash = current_binding
            adapter._completed_units.clear()
            adapter._absent_units.clear()
        assert len(service.registry.tasks()) == count + 3
        # The intent previewed at two hours was never prepared; after its
        # acquisition window the refusal names the window, not a stale packet.
        stale_intent = later_preview["next_requests"]["prepare"]
        now[0] = _NOW + timedelta(hours=4)
        closed = client.request(stale_intent)
        assert closed["disposition"] == "REFUSED_ACQUISITION_WINDOW_CLOSED"
        assert closed["failure_code"] == "alternative_evidence.acquisition_deadline_exceeded"
        assert len(service.registry.tasks()) == count + 3
        service.session.stop()

        # A new process over the same workspace, still the same day: the
        # captured intent reads back the same Task; a day later it is expired.
        now[0] = _NOW + timedelta(hours=6)
        service = start_service(workspace, authority, tmp_path, clock=lambda: now[0])
        client = LocalResearchClient(service.session.workspace)
        reopened = client.request(captured)
        assert reopened["disposition"] == "REUSED_EXACT" and reopened["task_id"] == task_id
        assert reopened["evidence_as_of"] == _NOW.isoformat()
        # The authority (and its runtime counters) outlives the service: the
        # reopen added no corpus pass to the one the first service made.
        reopened_adapter = service.review.evidence_task_adapter
        assert reopened_adapter.runtime.retrieval.passage_embedding_pass_count == 1
        now[0] = _NOW + timedelta(days=1, minutes=1)
        expired = client.request(captured)
        assert expired["disposition"] == "REFUSED_PREPARATION_EXPIRED"
        assert expired["failure_code"] == "alternative_evidence.brief_source_stale"
        assert expired["next_requests"]["preview"]["operation"] == "EVIDENCE_PREVIEW"
        assert len(service.registry.tasks()) == count + 3
    finally:
        service.session.stop()


def test_external_dossier_submission_is_bound_and_recovers_each_own_answer(
    service, monkeypatch, tmp_path, capsys
):

    from alphalattice.interface.local_application.cli import main
    from alphalattice.interface.local_application.client import LocalResearchClient
    from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
        PortfolioReviewAnswer,
    )
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        portfolio_review_task_contract,
    )

    service.post("/api/evidence-refresh", {"result_hash": service.result_hash()})
    service.drain()
    client = LocalResearchClient(service.session.workspace)
    review_risks = service.review.review_actor.risks
    service.review.model_authority_admitted = False
    service.review.review_actor = None
    count, writes = len(service.registry.tasks()), service.review.artifacts.write_count
    exported = client.request(
        {"operation": "CRO_REVIEW_DOSSIER", "result_hash": service.result_hash()}
    )
    assert exported["status"] == "CRO_DOSSIER_READY"
    assert service.review.artifacts.write_count == writes and len(service.registry.tasks()) == count
    schema = exported["assessment_schema"]
    risk_ref = schema["properties"]["risks"]["items"]["$ref"].split("/")[-1]
    aliases = [f"F{index}" for index in range(1, len(exported["dossier"]["findings"]) + 1)]
    assert schema["$defs"][risk_ref]["properties"]["findings"]["items"]["enum"] == aliases
    assert list(exported["finding_aliases"]) == aliases
    assert (
        "enum"
        not in PortfolioReviewAnswer.model_json_schema()["$defs"][risk_ref]["properties"][
            "findings"
        ]["items"]
    )
    submission = controlled_answer(exported["dossier"], *review_risks).model_dump(mode="json")
    document = {**exported["submission_template"], "review_answer": submission}
    assert document["result_hash"] == service.result_hash()
    for field in ("review_dossier_hash", "review_policy_hash", "review_schema_hash"):
        assert client.request({**document, field: "a" * 64})["refused"].endswith(
            "chief_risk_officer.external_review_binding_changed"
        )
    invented = json.loads(json.dumps(submission))
    invented["risks"][0]["findings"] = ["F999"]
    refused = client.request({**document, "review_answer": invented})
    assert refused["failure_code"] == "chief_risk_officer.answer_problems"
    assert refused["answer"]["problems"] == [
        {"item": 1, "text": "F999 is not a finding of this bundle."}
    ]
    assert refused["task_id"] is None
    # The answer names no issuer: an invented field is named, never read.
    refused = client.request({**document, "review_answer": {**submission, "issuers": ["UNKNOWN"]}})
    assert refused["answer"]["problems"] == [
        {"item": None, "text": "'issuers' is not a field of this answer; use 'risks', 'summary'."}
    ]
    assert (refused["answer"]["number"], refused["answer"]["rounds_left"]) == (2, 1)
    assert client.request({**document, "caller": "HUMAN"})["status"] == "REFUSED"
    assert len(service.registry.tasks()) == count
    assert service.review.artifacts.write_count == writes + 2
    original_contract = service.review.review_task_contract

    def changed_contract(**facts):
        envelope, _, _ = original_contract(**facts)
        return portfolio_review_task_contract(
            review_key_payload={**envelope.payload, "decision_policy_hash": "a" * 64}
        )

    with monkeypatch.context() as patch:
        patch.setattr(service.review, "review_task_contract", changed_contract)
        assert "external_review_binding_changed" in client.request(document)["refused"]
    assert len(service.registry.tasks()) == count
    assert service.review.artifacts.write_count == writes + 2
    # Recovery uses each immutable answer, not the last command of its kind.
    workspace = ["--workspace", str(service.session.workspace), "--view", "full"]
    prepare = [
        *("bundle", "prepare", "--role", "CRO"),
        *("--dir", str(tmp_path / "cro-bundle")),
    ]
    assert main([*workspace, *prepare], serve=lambda _: 99) == 0
    bundle = json.loads(capsys.readouterr().out)["data"]
    read = [item["name"] for item in bundle["files"]]
    Path(bundle["answer_file"]).write_text(
        json.dumps({**submission, "read": read}), encoding="utf-8"
    )
    contract_calls = []
    capacity = client.request({"operation": "CPU_BUDGET_SET", "tasks_waiting": "1"})
    assert capacity["task_queue"]["tasks_waiting"] == 1

    def counted_contract(**facts):
        contract_calls.append(1)
        return original_contract(**facts)

    with monkeypatch.context() as patch:
        patch.setattr(PortfolioReviewCommand, "execute", lambda *_: None)
        patch.setattr(service.review, "review_task_contract", counted_contract)
        assert main(["--view", "full", *bundle["submit_arguments"]], serve=lambda _: 99) == 3
        first = json.loads(capsys.readouterr().out)["data"]
        second_submission = {**submission, "summary": "A second reading of the same findings."}
        second_document = {**document, "review_answer": second_submission}
        full = client.request(second_document)
        assert full["disposition"] == "REFUSED_QUEUE_FULL" and "task_id" not in full
        assert full["failure_code"] == "task_control.queue_full" and full["detail"]
        capacity = client.request(full["next_requests"]["capacity"])
        assert capacity["task_queue"]["tasks_waiting"] == 1
        capacity = client.request({"operation": "CPU_BUDGET_SET", "tasks_waiting": "2"})
        assert capacity["task_queue"]["tasks_waiting"] == 2
        second = client.request(second_document)
    assert len(contract_calls) == 3  # One frozen contract per attempt, including the refusal.
    ids = (UUID(first["task_id"]), UUID(second["task_id"]))
    assert first["status"] == "ACCEPTED" and first["receipt"]["task_id"] == first["task_id"]
    assert second["disposition"] == "ADMITTED" and ids[0] != ids[1]
    for task_id, answer in zip(ids, (submission, second_submission), strict=True):
        task = service.registry.task(task_id)
        assert task.input.payload["prepared_answer"] == answer
        assert task.input.payload["actor_kind"] == "EXTERNAL_AUTOMATION"
        assert task.input.payload["actor_id"] == "local-research-external"
    service.session.stop()
    service.session.review_authority = replace(
        service.session.review_authority, model_authority_admitted=False, review_actor=None
    )
    service.session.start()
    assert not service.review.model_authority_admitted and service.review.review_actor is None
    service.drain()
    client = LocalResearchClient(service.session.workspace)
    publications = service.review.artifacts.values(
        "cro-review-publications", PortfolioReviewPublication
    )
    assert len(publications) == 2
    for task_id, answer in zip(ids, (submission, second_submission), strict=True):
        task = service.registry.task(task_id)
        assert task.lifecycle is TaskLifecycle.SUCCEEDED
        view = service.review.review_publications.find_for_review_key(task.input.input_hash)
        assert view.receipt.answer is not None
        assert view.receipt.answer.model_dump(mode="json") == answer
        assert view.receipt.actor_submission.actor_kind.value == "EXTERNAL_AUTOMATION"
        assert view.receipt.model_call_count == 0  # Product-managed calls only.
        body = client.request(
            {
                "operation": "EVIDENCE_CRO_EXPORT",
                "result_hash": service.result_hash(),
                "review_publication_hash": view.publication.publication_hash,
            }
        )
        assert body["review"]["dossier"]["dossier_hash"] == document["review_dossier_hash"]
        assert "Assessed by EXTERNAL_AUTOMATION: local-research-external" in body["html"]
        assert "Product-managed model calls: 0" in body["html"]
        assert "cost are unavailable" in body["html"]
    projection = client.request({"operation": "EVIDENCE_CRO", "result_hash": service.result_hash()})
    assert any("EXTERNAL_AUTOMATION" in line for line in projection["review_attribution"])
    reused = client.request(document)
    assert reused["disposition"] == "REUSED_EXACT"
    assert "history=review%3A" + reused["review_publication_hash"] in client.selected_url(
        document, reused
    )
    assert len(service.registry.tasks()) == count + 2


def test_a_third_answer_with_problems_is_done_and_its_valid_part_admitted(service):
    """Two corrections at most, counted durably; the third answer is kept whole
    where it is acceptable and its problem recorded as dropped, never stuck."""

    from alphalattice.interface.local_application.client import LocalResearchClient

    service.post("/api/evidence-refresh", {"result_hash": service.result_hash()})
    service.drain()
    risks = service.review.review_actor.risks
    service.review.model_authority_admitted = False
    service.review.review_actor = None
    client = LocalResearchClient(service.session.workspace)
    exported = client.request({"operation": "CRO_REVIEW_DOSSIER"})
    submission = controlled_answer(exported["dossier"], *risks).model_dump(mode="json")
    stray = {**submission["risks"][0], "findings": ["F999"]}
    answer = {**submission, "risks": [*submission["risks"], stray]}
    document = {**exported["submission_template"], "review_answer": answer}
    problem = {"item": len(answer["risks"]), "text": "F999 is not a finding of this bundle."}
    count = len(service.registry.tasks())
    for number in (1, 2):
        returned = client.request(document)
        assert returned["status"] == "CORRECT" and returned["task_id"] is None
        assert returned["answer"]["number"] == number
        assert returned["answer"]["rounds_left"] == 3 - number
        assert returned["answer"]["problems"] == [problem]
        assert returned["answer"]["accepted_items"] == list(range(1, len(answer["risks"])))
        if number == 1:
            # The count is durable: a restarted Host keeps it.
            service.session.stop()
            service.session.review_authority = replace(
                service.session.review_authority,
                model_authority_admitted=False,
                review_actor=None,
            )
            service.session.start()
            client = LocalResearchClient(service.session.workspace)
    done = client.request(document)
    assert done["disposition"] == "ADMITTED" and len(service.registry.tasks()) == count + 1
    binding = {
        "role": "chief_risk_officer.reviewer",
        "dossier_hash": exported["submission_template"]["review_dossier_hash"],
        "policy_hash": exported["submission_template"]["review_policy_hash"],
        "schema_hash": exported["submission_template"]["review_schema_hash"],
    }
    answer_reference = answer_slot(str(canonical_hash(binding)), 3)
    record = service.review.artifacts.load(ANSWER_CATEGORY, answer_reference, AgentAnswerRecord)
    delivery = {
        "answer_record": record.model_dump(mode="json"),
        "contribution": submission,
        "task_id": done["task_id"],
        "first_submission": True,
    }
    assert done["answer"] == {
        "verdict": "DONE",
        "number": 3,
        "rounds_left": 0,
        "accepted_items": list(range(1, len(answer["risks"]))),
        "answer_reference": answer_reference,
        "dropped": [problem],
        "accepted_delivery": delivery,
    }
    service.drain()
    # Settled (V417): the same answer again reads its receipt; another, even a clean one, is
    # refused by name and starts no Task.
    again = client.request(document)
    assert again["disposition"] == "REUSED_EXACT", again
    assert again["answer"] == {
        **done["answer"],
        "accepted_delivery": {**delivery, "first_submission": False},
    }, again
    clean = client.request({**exported["submission_template"], "review_answer": submission})
    assert "agent_bundle.answer_settled" in json.dumps(clean), clean
    assert len(service.registry.tasks()) == count + 1
    view = service.review.review_publications.find_for_review_key(
        service.registry.task(UUID(done["task_id"])).input.input_hash
    )
    assert view.receipt.answer.model_dump(mode="json") == submission
    assert [value.model_dump(mode="json") for value in view.receipt.dropped] == [problem]
    # What was dropped is never silent: the review is partial and says so,
    # and the page reads the same line.
    line = (
        "The reviewer's last answer had 1 item(s) the Host could not admit after two "
        f"corrections; they are not part of this review. Item {problem['item']}: "
        f"{problem['text']}"
    )
    assert view.receipt.outcome.review_state.value == "PARTIAL"
    assert line in view.receipt.outcome.limitations
    assert line in service.evidence_cro()["limitations"]


def test_a_dropped_analyst_item_reaches_the_review_as_a_gap(service):
    """A finding the Host dropped from the Analyst's third answer is not
    silently absent: the published analysis carries it as its first gap, and
    the CRO's bundle names it before the other gaps."""

    from alphalattice.interface.local_application.client import LocalResearchClient
    from alphalattice.oversight.chief_risk_officer.decision.views import render_review_bundle
    from tests.alternative_evidence_desk.planted_corpus import _CitingActor

    adapter = service.review.evidence_task_adapter
    topics = adapter.resources.analysis_actor.topics
    adapter.resources = replace(adapter.resources, analysis_actor=None)
    service.review.model_authority_admitted = False
    service.review.review_actor = None
    client = LocalResearchClient(service.session.workspace)
    selected = {"result_hash": service.result_hash()}
    preview = client.request({"operation": "EVIDENCE_PREVIEW", **selected})
    prepared = client.request(preview["next_requests"]["prepare"])
    service.drain()
    source_id = UUID(prepared["task_id"])
    exported = client.request(_packet(service.evidence_cro()))
    packet = adapter.prepared_packet(source_id, now=service.review.clock(), unit_id=ONE_UNIT)
    answer = _CitingActor(topics=topics)(packet=packet).answer.model_dump(mode="json")
    stray = {**answer["findings"][0], "cite": ["S999"]}
    written = {**answer, "findings": [*answer["findings"], stray]}
    document = {**exported["submission_template"], "analysis_answer": written}
    problem = {"item": len(written["findings"]), "text": "S999 is not an excerpt of this bundle."}
    for _number in (1, 2):
        assert client.request(document)["status"] == "CORRECT"
    done = client.request(document)
    assert done["disposition"] == "ADMITTED" and done["answer"]["dropped"] == [problem]
    service.drain()
    view = adapter.published_analysis(UUID(done["task_id"]), now=service.review.clock())
    line = (
        f"The Analyst's answer lost item {problem['item']} after two corrections: {problem['text']}"
    )
    assert view.lineage.cro_package.missing_evidence[0] == line
    assert len(view.lineage.brief.findings) == len(answer["findings"])
    dossier = service.review._resolve_review_dossier(service.review._review_selector(None))
    assert dossier.coverage.missing_evidence[0].endswith(line)
    files = render_review_bundle(dossier, task_procedure="").files
    bundle = "\n".join(text for _name, text in files)
    assert bundle.index(line) < bundle.index(dossier.coverage.missing_evidence[1])


def test_the_two_commands_carry_each_specialist_from_its_bundle_to_a_receipt(
    service, tmp_path, capsys, monkeypatch
):
    """The two commands carry each specialist from its bundle to a receipt."""

    from alphalattice.evidence.alternative_evidence.analysis.views import render_analyst_bundle
    from alphalattice.interface.local_application.cli import main
    from alphalattice.interface.local_application.client import LocalResearchClient
    from alphalattice.oversight.chief_risk_officer.decision.views import render_review_bundle
    from tests.alternative_evidence_desk.planted_corpus import _CitingActor
    from tests.alternative_evidence_desk.review_dossiers import REPO_ROOT

    adapter = service.review.evidence_task_adapter
    topics = adapter.resources.analysis_actor.topics
    adapter.resources = replace(adapter.resources, analysis_actor=None)
    service.review.model_authority_admitted = False
    service.review.review_actor = None
    client = LocalResearchClient(service.session.workspace)
    selected = {"result_hash": service.result_hash()}
    start = client.activity()

    def run(*arguments: str) -> tuple[int, dict[str, Any]]:
        code = main(["--view", "full", *arguments], serve=lambda _: 99)
        return code, json.loads(capsys.readouterr().out)["data"]

    def files_of(bundle: dict[str, Any]) -> dict[str, str]:
        directory = Path(bundle["bundle_directory"])
        return {path.name: path.read_text(encoding="utf-8") for path in directory.iterdir()}

    def skill(path: str) -> str:
        return (REPO_ROOT / "skills" / path / "SKILL.md").read_text(encoding="utf-8")

    workspace = ("--workspace", str(service.session.workspace))
    # Commands printed for a POSIX shell, which `shlex` reads back; the PowerShell form is
    # held by OP16's own test (V449).
    monkeypatch.setenv("ALPHALATTICE_SHELL", "posix")
    # The CPU budget (F1): the lead reads the machine and its load, and sets a
    # budget, with one verb; a budget that is not one is refused by name.
    code, budget = run(*workspace, "cpu-budget", "show")
    assert code == 0 and budget["cpu_budget"] == "auto" and budget["machine"]["processors"] >= 1
    code, budget = run(*workspace, "cpu-budget", "set", "--cores", "2")
    assert code == 0 and budget["cpu_budget"] == 2
    code, refused = run(*workspace, "cpu-budget", "set", "--cores", "all")
    assert code == 2 and "execution.cpu_budget_invalid" in json.dumps(refused)
    assert run(*workspace, "cpu-budget", "set", "--cores", "auto")[0] == 0
    # The Task queue's waiting places are set beside it, one setting a request (V100).
    assert budget["task_queue"]["tasks_waiting"] == "auto" and budget["task_queue"]["places"] >= 1
    code, queue = run(*workspace, "cpu-budget", "set", "--queue", "3")
    assert code == 0 and queue["task_queue"]["places"] == 3 and queue["cpu_budget"] == "auto"
    code, refused = run(*workspace, "cpu-budget", "set", "--queue", "0")
    assert code == 2 and "task_control.tasks_waiting_invalid" in json.dumps(refused)
    preview = client.request({"operation": "EVIDENCE_PREVIEW", **selected})
    prepared = client.request(preview["next_requests"]["prepare"])
    service.drain()
    source_id = prepared["task_id"]

    # Every book is a run: the Analyst's packet is one unit's, and a bundle
    # asked for without its unit is refused naming the units there are.
    code, unnamed = run(
        *workspace,
        *("bundle", "prepare", "--role", "ANALYST", "--task", source_id),
        *("--result", selected["result_hash"]),
        *("--dir", str(tmp_path / "unnamed")),
    )
    assert code == 2 and f"agent_bundle.analyst_unit_required:{ONE_UNIT}" in json.dumps(unnamed)
    assert not (tmp_path / "unnamed").exists()
    # Each says how to name the packet, where it answered with no way on (V242).
    assert unnamed["next_action"] == "GIVE_THE_PACKETS_TASK_AND_UNIT"
    assert "`evidence_unit_id`" in unnamed["message"]
    code, untasked = run(
        *workspace,
        *("bundle", "prepare", "--role", "ANALYST"),
        *("--result", selected["result_hash"]),
        *("--dir", str(tmp_path / "untasked")),
    )
    assert code == 2 and untasked["failure_code"] == "agent_bundle.analyst_task_required"
    assert untasked["next_action"] == "GIVE_THE_PACKETS_TASK_AND_UNIT"

    # The Analyst: a subset, one invalid item corrected, then accepted.
    code, analyst = run(
        *workspace,
        *("bundle", "prepare", "--role", "ANALYST"),
        *("--task", source_id, "--unit", ONE_UNIT),
        *("--result", selected["result_hash"]),
        *("--dir", str(tmp_path / "analyst")),
    )
    assert code == 0 and analyst["status"] == "AGENT_BUNDLE_READY"
    assert analyst["files"][0]["name"] == "README.md"
    packet = adapter.prepared_packet(UUID(source_id), now=service.review.clock(), unit_id=ONE_UNIT)
    view = render_analyst_bundle(
        packet, task_procedure=skill("alternative-evidence/governed-evidence-analysis")
    )
    assert files_of(analyst) == dict(view.files)
    readme = files_of(analyst)["README.md"]
    assert "`findings`: at most three an issuer and 32 in one answer" in readme
    full = _CitingActor(topics=topics)(packet=packet).answer.model_dump(mode="json")
    subset = {**full, "findings": full["findings"][:2]}
    assert len(subset["findings"]) == 2
    invented = json.loads(json.dumps(subset))
    invented["findings"][1]["cite"] = ["S999"]
    answer_file = Path(analyst["answer_file"])
    # An answer names the files it read whole; the list is its own word, kept with the
    # answer as provenance, and the answer is read whatever it names (V260, OP11).
    read = [item["name"] for item in analyst["files"]]
    answer_file.write_text(json.dumps({**invented, "read": read[:1]}), encoding="utf-8")
    code, verdict = run(*analyst["submit_arguments"])
    assert code == 0 and verdict["status"] == "CORRECT" and "receipt" not in verdict
    bundle = EvidenceReviewBundles(service.review).agent_bundle(str(tmp_path / "analyst"))
    assert bundle is not None
    binding = {
        "role": "alternative_evidence.analyst",
        "task_id": source_id,
        "unit_id": ONE_UNIT,
        "analysis_context_hash": bundle.submission["analysis_context_hash"],
    }
    kept = service.review.artifacts.load(
        ANSWER_CATEGORY, answer_slot(str(canonical_hash(binding)), 1), AgentAnswerRecord
    )
    assert kept.read_files == tuple(read[:1])
    assert verdict["problems"] == [{"item": 2, "text": "S999 is not an excerpt of this bundle."}]
    assert (verdict["answer_number"], verdict["rounds_left"], verdict["accepted_items"]) == (
        1,
        2,
        [1],
    )
    answer_file.write_text(json.dumps({**subset, "read": read}), encoding="utf-8")
    code, verdict = run(*analyst["submit_arguments"])
    # Accepted, its Task pending: the receipt names the Task's state (V580).
    assert code == 3 and verdict["status"] == "ACCEPTED"
    assert verdict["task_lifecycle"] in {"QUEUED", "RUNNING"}
    assert verdict["receipt"] == {
        "agent_role": "ANALYST",
        "verdict": "ACCEPTED",
        "answer_reference": answer_slot(str(canonical_hash(binding)), 2),
        "task_id": verdict["task_id"],
        "accepted_items": 2,
        "dropped_items": 0,
    }
    service.drain()

    # The CRO: one invalid risk kept three times, the third answer DONE.
    # The book named by its selector's field (an experiment or update book
    # names its own fields); the reply carries the one command ready to run.
    code, cro = run(
        *workspace,
        *("bundle", "prepare", "--role", "CRO"),
        *("--result", selected["result_hash"]),
        *("--dir", str(tmp_path / "c")),
    )
    assert code == 0 and cro["status"] == "AGENT_BUNDLE_READY"
    cro_bundle = EvidenceReviewBundles(service.review).agent_bundle(str(tmp_path / "c"))
    cro_binding = {
        "role": "chief_risk_officer.reviewer",
        "dossier_hash": cro_bundle.submission["review_dossier_hash"],
        "policy_hash": cro_bundle.submission["review_policy_hash"],
        "schema_hash": cro_bundle.submission["review_schema_hash"],
    }
    # The bundle offers its submit, the directory bound and the answer left (V406).
    offered = cro["next_requests"]["submit"]
    assert (offered["operation"], offered["agent_answer"]) == ("AGENT_ANSWER_SUBMIT", None)
    assert Path(offered["bundle_directory"]) == Path(cro["bundle_directory"])
    command = shlex.split(cro["submit_command"])
    # The installed command, as every printed command starts (V429), then its arguments.
    assert command[0] == "alphalattice" and command[1:] == cro["submit_arguments"]
    dossier = service.review._resolve_review_dossier(service.review._review_selector(None))
    view = render_review_bundle(
        dossier, task_procedure=skill("chief-risk-officer/review-portfolio-evidence")
    )
    assert files_of(cro) == dict(view.files)
    assert "`risks`: at most 16 in one answer" in files_of(cro)["README.md"]
    risk = {
        "findings": ["F1"],
        "why": "The cited finding is adverse for a held issuer.",
        "severity": "HIGH",
        "confidence": "SUPPORTED",
        "recommendation": "Size the position down.",
    }
    submission = {"risks": [risk], "summary": "One major negative in the evidence read."}
    stray = {**risk, "findings": ["F999"]}
    Path(cro["answer_file"]).write_text(
        json.dumps(
            {
                **submission,
                "risks": [*submission["risks"], stray],
                "read": [item["name"] for item in cro["files"]],
            }
        ),
        encoding="utf-8",
    )
    problem = {
        "item": len(submission["risks"]) + 1,
        "text": "F999 is not a finding of this bundle.",
    }
    for rounds_left in (2, 1):
        code, verdict = run(*cro["submit_arguments"])
        assert code == 0 and verdict["status"] == "CORRECT"
        assert (verdict["rounds_left"], verdict["problems"]) == (rounds_left, [problem])
    code, verdict = run(*cro["submit_arguments"])
    assert code == 3 and verdict["status"] == "DONE" and verdict["dropped"] == [problem]
    assert verdict["receipt"] == {
        "agent_role": "CRO",
        "verdict": "DONE",
        "answer_reference": answer_slot(str(canonical_hash(cro_binding)), 3),
        "task_id": verdict["task_id"],
        "accepted_items": len(submission["risks"]),
        "dropped_items": 1,
    }
    # The lead goes on from this answer: its Task, and the book's Evidence and CRO (V406).
    assert verdict["next_requests"] == {
        "task": {"operation": "STATUS", "task_id": verdict["task_id"]},
        "book": {"operation": "EVIDENCE_CRO", "result_hash": selected["result_hash"]},
    }
    service.drain()
    review = service.review.review_publications.find_for_review_key(
        service.registry.task(UUID(verdict["task_id"])).input.input_hash
    )
    assert review.receipt.answer.model_dump(mode="json") == submission

    # What the agents read carries no hash; what the lead holds is the receipt.
    for text in (*files_of(analyst).values(), *files_of(cro).values()):
        assert re.search(r"\b[0-9a-f]{64}\b", text) is None
    page = client.activity(after=start["cursor"], limit=200)
    submitted = [
        (item["payload"]["subject"].get("agent_role"), item["payload"]["status"])
        for item in page["items"]
        if item["payload"]
        and item["payload"].get("operation") == "AGENT_ANSWER_SUBMIT"
        and item["payload"]["phase"] == "RETURNED"
    ]
    assert submitted == [
        ("ANALYST", "CORRECT"),
        ("ANALYST", "ACCEPTED"),
        ("CRO", "CORRECT"),
        ("CRO", "CORRECT"),
        ("CRO", "DONE"),
    ]
    # A directory the Host never prepared, or a bundle whose evidence moved
    # on, is refused by name and nothing is read.
    stranger = tmp_path / "elsewhere"
    stranger.mkdir()
    (stranger / "answer.json").write_text("{}", encoding="utf-8")
    code, refused = run(
        *workspace,
        *("bundle", "submit", "--dir", str(stranger)),
        *("--file", str(stranger / "answer.json")),
    )
    assert code == 2 and refused["failure_code"] == "agent_bundle.not_prepared"


def test_external_review_expiry_before_publication_keeps_the_assessment_but_not_a_publication(
    service, monkeypatch
):
    from alphalattice.interface.local_application.client import LocalResearchClient

    service.post("/api/evidence-refresh", {"result_hash": service.result_hash()})
    service.drain()
    client = LocalResearchClient(service.session.workspace)
    exported = client.request({"operation": "CRO_REVIEW_DOSSIER"})
    document = {
        "operation": "CRO_REVIEW_SUBMIT",
        "review_dossier_hash": exported["dossier"]["dossier_hash"],
        "review_policy_hash": exported["decision_policy_hash"],
        "review_schema_hash": exported["assessment_schema_hash"],
        "review_answer": controlled_answer(
            exported["dossier"], *service.review.review_actor.risks
        ).model_dump(mode="json"),
    }
    verify = PortfolioReviewTaskAdapter.verify_stage

    def expire_after_assessment(adapter, **kwargs):
        result = verify(adapter, **kwargs)
        if kwargs["work_item"].stage_id == "seal_portfolio_review_assessment":
            service.review.clock = lambda: _NOW + timedelta(days=10)
        return result

    monkeypatch.setattr(PortfolioReviewTaskAdapter, "verify_stage", expire_after_assessment)
    sent = client.request(document)
    service.drain()
    task = service.registry.task(UUID(sent["task_id"]))
    assert task.lifecycle is TaskLifecycle.BLOCKED
    assert "external_review_context_changed" in task.failure_code
    assert task.input.payload["prepared_answer"] == document["review_answer"]
    assert not service.review.artifacts.values(
        "cro-review-publications", PortfolioReviewPublication
    )
    count = len(service.registry.tasks())
    assert client.request(document)["disposition"] == "REFUSED_ALTERNATIVE_EVIDENCE_EXPIRED"
    assert len(service.registry.tasks()) == count


@pytest.fixture
def credential_free_service(tmp_path: Path, http_book) -> Iterator[_Service]:
    """Valid workspace; no Provider credential.

    Everything a deterministic product needs is admitted: the evidence runtime,
    the recorded documents, the registry and the listing authority. Only the
    two model actors are absent, which is what an admission with no
    `DEEPSEEK_API_KEY` now produces.
    """

    yield from _service(
        tmp_path, *http_book, with_runtime=True, with_actor=True, model_authority_admitted=False
    )


@pytest.fixture
def bare_service(tmp_path: Path, http_book) -> Iterator[_Service]:
    """Authority admitted, but neither an evidence runtime nor a review actor."""

    yield from _service(tmp_path, *http_book, with_runtime=False, with_actor=False)


def _no_identities(body: dict[str, Any], *, allowed: set[str]) -> bool:
    return set(re.findall(r"[0-9a-f]{64}", json.dumps(body))) <= allowed


def _drive_whole_route(service: _Service) -> str:
    result_hash = service.result_hash()
    refreshed = service.post("/api/evidence-refresh", {"result_hash": result_hash})
    assert refreshed["disposition"] == "ADMITTED"
    service.drain()
    reviewed = service.post("/api/cro-review", {"result_hash": result_hash})
    assert reviewed["disposition"] == "ADMITTED"
    service.drain()
    return result_hash


# ============================================================ the whole route


def test_the_two_buttons_drive_the_whole_route_over_http(service: _Service) -> None:
    result_hash = service.result_hash()
    section = service.evidence_cro()
    assert section["state"] == "AWAITING_ALTERNATIVE_EVIDENCE"
    assert section["book"]["authority"] == "DEVELOPMENT_RESULT"
    assert section["book"]["result_hash"] == result_hash

    refreshed = service.post("/api/evidence-refresh", {"result_hash": result_hash})
    assert refreshed["disposition"] == "ADMITTED"
    assert service.evidence_cro()["state"] == "EVIDENCE_REFRESH_IN_PROGRESS"
    service.drain()
    assert service.registry.task(UUID(refreshed["task_id"])).lifecycle is TaskLifecycle.SUCCEEDED
    assert service.evidence_cro()["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"

    reviewed = service.post("/api/cro-review", {"result_hash": result_hash})
    assert reviewed["disposition"] == "ADMITTED"
    service.drain()
    assert service.registry.task(UUID(reviewed["task_id"])).lifecycle is TaskLifecycle.SUCCEEDED

    published = service.evidence_cro(result_hash)
    assert published["state"] == "REVIEW_PUBLISHED"
    assert published["book"]["authority"] == "DEVELOPMENT_RESULT"
    assert published["required_actions"][0]["action"] == "RECONSIDER_CANDIDATE"
    assert published["evidence_selection"] == "UNIQUE_CURRENT"
    assert len(published["issuer_rows"]) == 5
    assert published["issue_cards"]
    assert published["citations"]
    # The publication identity crosses the HTTP boundary for the installed
    # result and agrees with the history entry that names the same review.
    recorded = next(
        row["review_publication_hash"]
        for row in service.get("/api/research-history?history_kind=CRO_REVIEW")["entries"]
        if row["book"].get("result_hash") == result_hash
    )
    assert published["review_publication_hash"] == recorded
    assert _no_identities(published, allowed={result_hash, recorded})
    # The only identities handed to the page are the book's and the handle of
    # the review on display, which its export request is bound to.
    review_hash = published["review_publication_hash"]
    assert _no_identities(published, allowed={result_hash, review_hash})
    assert published["next_requests"]["export"]["review_publication_hash"] == review_hash
    assert published == service.evidence_cro(), "the latest result is the default book"


def test_freezing_the_book_asks_for_a_new_review_and_reuses_the_evidence(
    service: _Service,
) -> None:
    """A freeze changes what a review means, not what the issuers disclosed."""

    result_hash = _drive_whole_route(service)
    previous = next(
        row["review_publication_hash"]
        for row in service.get("/api/research-history?history_kind=CRO_REVIEW")["entries"]
        if row["book"].get("result_hash") == result_hash
    )
    export_query = (
        f"/api/evidence-cro/export?result_hash={result_hash}&review_publication_hash={previous}"
    )
    historical = service.get(export_query)
    frozen = service.agent(
        PortfolioResearchAgentRequest(operation="FREEZE", result_hash=result_hash)
    )
    assert frozen["candidate_hash"]
    assert service.get(export_query) == historical

    section = service.evidence_cro(result_hash)
    assert section["book"]["authority"] == "FROZEN_CANDIDATE"
    assert section["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
    reviewed = service.post("/api/cro-review", {"result_hash": result_hash})
    assert reviewed["disposition"] == "ADMITTED"
    service.drain()
    published = service.evidence_cro(result_hash)
    assert published["state"] == "REVIEW_PUBLISHED"
    assert published["book"]["authority"] == "FROZEN_CANDIDATE"
    assert published["required_actions"][0]["action"] == "RECONSIDER_CANDIDATE"

    analyses = service.review.artifacts.values(
        "analysis-publications", AlternativeEvidenceAnalysisPublication
    )
    assert len(analyses) == 1, "the evidence was reused, not re-acquired"
    reviews = service.review.review_publications.store.values(
        "cro-review-publications", PortfolioReviewPublication
    )
    assert {value.book_authority for value in reviews} == {
        "DEVELOPMENT_RESULT",
        "FROZEN_CANDIDATE",
    }


def test_an_exact_reuse_over_http_admits_no_task_and_calls_no_actor(service: _Service) -> None:
    result_hash = _drive_whole_route(service)
    tasks = len(service.registry.tasks())
    calls = len(service.review.review_actor.observed_deadlines)

    again = service.post("/api/cro-review", {"result_hash": result_hash})
    assert again["disposition"] == "REUSED_EXACT"
    assert again.get("task_id") is None
    assert len(service.registry.tasks()) == tasks
    assert len(service.review.review_actor.observed_deadlines) == calls


def test_expiry_is_a_typed_zero_work_state_over_http(tmp_path: Path, http_book) -> None:
    workspace, report = http_book
    now = [_NOW]
    service = start_service(
        workspace, build_authority(tmp_path=tmp_path, report=report), tmp_path, clock=lambda: now[0]
    )
    try:
        result_hash = service.result_hash()
        assert (
            service.post("/api/evidence-refresh", {"result_hash": result_hash})["disposition"]
            == "ADMITTED"
        )
        service.drain()
        assert service.evidence_cro()["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"

        now[0] = _NOW + timedelta(days=2)
        expired = service.evidence_cro()
        assert expired["state"] == "ALTERNATIVE_EVIDENCE_EXPIRED"
        assert expired["required_actions"][0]["action"] == "PREPARE_EVIDENCE"
        # The state says which analysis expired and when, and the way forward
        # is a new preparation (the managed refresh only because a credential
        # is admitted here), never the stale review.
        assert expired["evidence_expires_at"] in expired["explanation"]
        assert expired["evidence_as_of"] == _NOW.isoformat()
        assert set(expired["next_requests"]) == {"preview", "refresh"}
        assert expired["next_requests"]["preview"] == {
            "operation": "EVIDENCE_PREVIEW",
            "result_hash": result_hash,
        }
        tasks = len(service.registry.tasks())
        refused = service.post("/api/cro-review", {"result_hash": result_hash})
        assert refused["disposition"] == "REFUSED_ALTERNATIVE_EVIDENCE_EXPIRED"
        assert len(service.registry.tasks()) == tasks
        assert service.review.review_actor.observed_deadlines == []
    finally:
        service.session.stop()


def test_a_service_without_a_runtime_or_an_actor_refuses_with_no_work(
    bare_service: _Service,
) -> None:
    result_hash = bare_service.result_hash()
    tasks = len(bare_service.registry.tasks())

    section = bare_service.evidence_cro()
    assert section["state"] == "EVIDENCE_AUTHORITY_NOT_ADMITTED"
    assert "source package" in section["explanation"]
    assert section["next_requests"] == {} and section["available_actions"] == []
    refresh = bare_service.post("/api/evidence-refresh", {"result_hash": result_hash})
    assert refresh["disposition"] == "REFUSED_NO_ADMITTED_EVIDENCE_RUNTIME"
    review = bare_service.post("/api/cro-review", {"result_hash": result_hash})
    assert review["disposition"] == "REFUSED_NO_ADMITTED_REVIEW_ACTOR"
    assert len(bare_service.registry.tasks()) == tasks


# ======================================== the real composition, no credential


def _install_semantic_pack(monkeypatch: pytest.MonkeyPatch, model_root: Path) -> None:
    """Stand in for the one artifact this environment cannot hold.

    `probe_semantic_pack` verifies a pinned MiniLM ONNX pack by sha256, then
    checks onnxruntime, sentencepiece and sqlite-vec versions, the CPU
    execution provider and the loaded vector extension. Everything but the two
    model files is really installed here and really runs; the pack itself is
    about half a gigabyte, is absent from this machine, and can only be
    obtained over a network this gate forbids. So exactly two file hashes are
    substituted and nothing else about the probe is bypassed.
    """

    from alphalattice.kernel.knowledge import _embeddings as embeddings
    from alphalattice.kernel.knowledge.hybrid_contracts import (
        MODEL_ARTIFACT_SHA256,
        TOKENIZER_SHA256,
    )

    (model_root / "onnx").mkdir(parents=True, exist_ok=True)
    (model_root / "onnx" / "model.onnx").write_bytes(b"absent-model-pack-placeholder")
    (model_root / "sentencepiece.bpe.model").write_bytes(b"absent-tokenizer-placeholder")
    real = embeddings._hash_file
    pinned = {"model.onnx": MODEL_ARTIFACT_SHA256, "sentencepiece.bpe.model": TOKENIZER_SHA256}

    def hash_file(path: Path) -> str:
        return pinned.get(path.name) or str(real(path))

    monkeypatch.setattr(embeddings, "_hash_file", hash_file)

    # The cross-encoder pack is bound the same way, and stood in for the same
    # way: its two file hashes are substituted, and the adapter that would load
    # the real ONNX graph is replaced by the deterministic stand-in the unit
    # tests use, so nothing about the probe, the identity or the packet path is
    # bypassed except the model bytes this environment cannot hold.
    from alphalattice.kernel.knowledge import _reranking as reranking
    from alphalattice.kernel.knowledge.hybrid_contracts import RERANKER_PACK
    from tests.alternative_evidence_desk.planted_corpus import _FakeReranker

    # The reranker pack is stood in for the same way and no further: every file
    # the loader consumes is really created, so presence, the refusal of an
    # unadmitted extra file, each per-file comparison and the manifest hash all
    # really run. Only the bytes-to-digest mapping and the two library version
    # strings are substituted, because this environment holds neither the pack
    # nor FastEmbed; the private real-evidence acceptance suite runs all of it for
    # real in the declared environment.
    pack = model_root / "reranker"
    reranker_pinned: dict[str, str] = {}
    for name, digest in RERANKER_PACK:
        artifact = pack / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"absent-reranker-artifact:{name}".encode())
        reranker_pinned[str(artifact)] = digest
    real_reranker_hash = reranking._hash_file

    def reranker_hash_file(path: Path) -> str:
        return reranker_pinned.get(str(path)) or str(real_reranker_hash(path))

    monkeypatch.setattr(reranking, "_hash_file", reranker_hash_file)
    monkeypatch.setattr(reranking, "_runtime_versions", lambda: ("0.8.0", "0.23.2"))
    monkeypatch.setattr(
        reranking, "LocalCrossEncoderAdapter", lambda _root, _spec, **_: _FakeReranker()
    )


def _forbid_outbound_network(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Any connection off the loopback fails the test and is recorded."""

    import socket

    attempts: list[Any] = []
    real_connect = socket.socket.connect

    def connect(self: Any, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host not in {"127.0.0.1", "::1", "localhost"}:
            attempts.append(address)
            raise AssertionError(f"outbound network call: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def test_the_real_composition_starts_and_refuses_without_a_provider_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, http_book
) -> None:
    """The real composition starts and refuses without a provider credential."""

    workspace = http_book[0]
    authority_root = workspace / "authority" / "semantic-model"
    authority_root.mkdir(parents=True, exist_ok=True)
    _install_semantic_pack(monkeypatch, authority_root)

    # The manifest must declare the capability the real reader computes.
    from alphalattice.evidence.alternative_evidence.runtime.service import (
        AlternativeEvidenceDocumentIntelligenceRuntime,
    )

    probe = AlternativeEvidenceDocumentIntelligenceRuntime(
        artifact_root=workspace / "runtime" / "artifacts",
        workspace_root=workspace / "runtime" / "evidence-knowledge",
        model_root=authority_root,
    )
    try:
        capability = probe.retrieval.capability()
        assert capability.status == "READY", capability.status
        capability_hash = str(capability.logical_hash)
    finally:
        probe.close()

    binding, _registry_path = _package(workspace, semantic_capability_hash=capability_hash)
    # The manifest carries the strategy artifact roots the installed catalog
    # needs, exactly as a shipped workspace does.
    (workspace / "artifacts" / "iw184").mkdir(parents=True, exist_ok=True)
    publish_research_workspace_manifest(
        workspace,
        ResearchWorkspaceManifest.create(
            workspace_id="qa-gate-9c6a-no-credential",
            default_strategy_package_id=BROAD_FEATURE_PACKAGE.strategy_id,
            default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
            strategy_artifacts=(
                ResearchWorkspaceArtifact(
                    artifact_key=PRODUCT_EVIDENCE_ROOT_KEY,
                    relative_path="artifacts/iw184",
                ),
            ),
            evidence_review=binding,
        ),
    )

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    attempts = _forbid_outbound_network(monkeypatch)

    session = LocalPortfolioWebSession.from_workspace(workspace)
    url = session.start()
    service = _Service(session, tmp_path)
    try:
        admitted = session.review_authority
        assert admitted is not None
        # There is no actor to call, which is a stronger statement than
        # counting calls that were never made.
        assert admitted.model_authority_admitted is False
        assert admitted.review_actor is None
        assert admitted.evidence_resources is not None
        assert admitted.evidence_resources.analysis_actor is None
        # The deterministic half is fully admitted, so this is a degraded
        # section rather than a degraded workspace.
        assert admitted.evidence_runtime is not None
        assert admitted.registry.entries[0].ticker == "AAPL"

        # Portfolio readback is untouched.
        result_hash = service.result_hash()
        report = service.get(f"/api/report?result_hash={result_hash}")
        assert report["report_hash"]
        assert report["book"]["positions"]

        tasks = len(service.registry.tasks())
        projection = service.evidence_cro(result_hash)
        # The state is what the evidence says -- nothing prepared yet -- and
        # the next step is the native one; the missing credential removes only
        # the managed actions.
        assert projection["state"] == "AWAITING_ALTERNATIVE_EVIDENCE"
        assert projection["available_actions"] == []
        assert set(projection["next_requests"]) == {"preview"}
        assert projection["next_requests"]["preview"] == {
            "operation": "EVIDENCE_PREVIEW",
            "result_hash": result_hash,
        }

        refresh = service.post("/api/evidence-refresh", {"result_hash": result_hash})
        assert refresh["disposition"] == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED"
        review = service.post("/api/cro-review", {"result_hash": result_hash})
        assert review["disposition"] == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED"
        for answer, role in ((refresh, "ANALYST"), (review, "CRO")):
            assert answer["failure_code"] == "evidence_review.model_authority_not_admitted"
            (bundle,) = answer["next_requests"].values()
            assert bundle == dict(
                operation="AGENT_BUNDLE_PREPARE", agent_role=role, result_hash=result_hash
            )
            command = re.search(r"\(([^)]+)\)", answer["detail"])
            assert command and shlex.split(command[1]) == ["bundle", "prepare"]

        assert len(service.registry.tasks()) == tasks
        service.drain()
        assert len(service.registry.tasks()) == tasks
        assert attempts == []
    finally:
        session.stop()

    # A clean close keeps nothing: no listener, no worker, no lease.
    assert session.web is None
    assert session.dispatcher is None
    assert session.session is None
    assert session.review is None
    assert session.review_authority is None
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"{url}/api/results", timeout=5)
    reacquired = WorkspaceApplicationSession.acquire(workspace)
    with reacquired:
        pass
    assert attempts == []


def test_local_web_starts_and_serves_portfolio_without_a_provider_credential(
    credential_free_service: _Service,
) -> None:
    """Local web starts and serves portfolio without a provider credential."""

    result_hash = credential_free_service.result_hash()
    assert len(result_hash) == 64
    report = credential_free_service.get(f"/api/report?result_hash={result_hash}")
    assert report["report_hash"]
    assert report["book"]["positions"]


def test_evidence_cro_refuses_with_zero_work_when_no_credential_is_admitted(
    credential_free_service: _Service,
) -> None:
    """Evidence CRO refuses with zero work when no credential is admitted."""

    service = credential_free_service
    result_hash = service.result_hash()
    tasks = len(service.registry.tasks())

    projection = service.evidence_cro(result_hash)
    assert projection["state"] == "AWAITING_ALTERNATIVE_EVIDENCE"
    assert projection["available_actions"] == []
    assert list(projection["next_requests"]) == ["preview"]
    # The state explains itself without naming a key, a header or a URL, and
    # says which work the missing credential withholds.
    assert "DEEPSEEK_API_KEY" not in projection["explanation"]
    assert "credential" in projection["explanation"]
    assert "Prepare" in projection["explanation"]

    refresh = service.post("/api/evidence-refresh", {"result_hash": result_hash})
    assert refresh["disposition"] == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED"
    review = service.post("/api/cro-review", {"result_hash": result_hash})
    assert review["disposition"] == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED"

    # No Task was admitted, so nothing is queued that could later call a model.
    assert len(service.registry.tasks()) == tasks
    service.drain()
    assert len(service.registry.tasks()) == tasks


def test_a_published_review_reopens_with_no_provider_credential(tmp_path: Path, http_book) -> None:
    """A published review reopens with no provider credential."""

    workspace, report = http_book
    online = build_authority(tmp_path=tmp_path, report=report)
    service = start_service(workspace, online, tmp_path)
    try:
        result_hash = _drive_whole_route(service)
        published = service.evidence_cro(result_hash)
        assert published["state"] == "REVIEW_PUBLISHED"
    finally:
        service.session.stop()

    offline = build_authority(tmp_path=tmp_path, report=report, model_authority_admitted=False)
    reopened_service = start_service(workspace, offline, tmp_path)
    try:
        before = len(reopened_service.registry.tasks())
        reopened = reopened_service.evidence_cro(result_hash)

        assert reopened["state"] == "REVIEW_PUBLISHED"
        for field in (
            "issue_cards",
            "citations",
            "issuer_rows",
            "coverage",
            "scope_coverage",
            "evidence_selection",
            "evidence_selection_detail",
            "reasons",
            "rule_ids",
            "policy_version",
            "disposition",
            "review_state",
        ):
            assert reopened.get(field) == published.get(field), field

        # Reading admitted nothing and called nobody.
        assert len(reopened_service.registry.tasks()) == before
        reopened_service.drain()
        assert len(reopened_service.registry.tasks()) == before

        publication_hash = next(
            iter(
                reopened_service.review.artifacts.values(
                    "cro-review-publications", PortfolioReviewPublication
                )
            )
        ).publication_hash
        historical_url = "/api/evidence-cro?" + urllib.parse.urlencode(
            {
                "result_hash": result_hash,
                "review_publication_hash": publication_hash,
            }
        )
        old_clock = reopened_service.review.clock
        reopened_service.review.clock = lambda: (
            datetime.fromisoformat(published["evidence_expires_at"]) + timedelta(seconds=1)
        )
        pinned = reopened_service.get(historical_url)
        assert pinned["state"] == "REVIEW_PUBLISHED"
        assert pinned["available_actions"] == [] and pinned["scope_coverage"] is None
        for field in (
            "issue_cards",
            "citations",
            "issuer_rows",
            "coverage",
            "review_state",
            "disposition",
            "review_attribution",
        ):
            assert pinned[field] == published[field], field
        assert len(reopened_service.registry.tasks()) == before
        reopened_service.review.clock = old_clock

        # New work still refuses, before any Task is admitted.
        for path in ("/api/evidence-refresh", "/api/cro-review"):
            answer = reopened_service.post(path, {"result_hash": result_hash})
            assert answer["disposition"] == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED", path
            assert answer.get("task_id") is None, path
        assert len(reopened_service.registry.tasks()) == before
    finally:
        reopened_service.session.stop()


def test_a_person_and_the_installed_agent_choose_and_are_recorded_apart(
    service: _Service,
) -> None:
    """A person and the installed agent choose and are recorded apart."""

    result_hash = service.result_hash()
    assert service.post("/api/evidence-refresh", {"result_hash": result_hash})["disposition"] == (
        "ADMITTED"
    )
    service.drain()
    application = service.review
    resolved = application.resolve_book(application.default_selector())
    eligible = application.eligible_evidence(resolved)
    assert eligible, "the refresh published something to choose"
    chosen = eligible[0][1].publication.publication_hash

    human = service.post(
        "/api/evidence-select",
        {"result_hash": result_hash, "analysis_publication_hash": chosen},
    )
    agent = service.agent(
        PortfolioResearchAgentRequest(
            operation="EVIDENCE_SELECT",
            result_hash=result_hash,
            analysis_publication_hash=chosen,
        )
    )

    assert human["chosen_by"] == "HUMAN"
    assert agent["chosen_by"] == "INSTALLED_AGENT"
    # Equivalent product outcomes; they differ only where provenance should.
    assert human["analysis_publication_hash"] == agent["analysis_publication_hash"] == chosen
    assert human["issuer_scope_hash"] == agent["issuer_scope_hash"]
    assert human["evidence_as_of"] == agent["evidence_as_of"]
    assert human["selection_hash"] != agent["selection_hash"]
    # The receipt names the book the choice was recorded for and goes on to that book's review,
    # never the default book's (V487, the user's review).
    assert agent["review_selector"]["result_hash"] == result_hash
    assert agent["next_requests"]["dossier"] == {
        "operation": "CRO_REVIEW_DOSSIER",
        **agent["review_selector"],
    }
    assert agent["next_requests"]["review"] == {
        "operation": "EVIDENCE_CRO",
        **agent["review_selector"],
    }

    # A page cannot name itself the Agent, and the envelope has no actor field.
    forged = service.post(
        "/api/evidence-select",
        {
            "result_hash": result_hash,
            "analysis_publication_hash": chosen,
            "chosen_by": "INSTALLED_AGENT",
        },
    )
    assert forged["chosen_by"] == "HUMAN", "the route states the caller, not the body"
    with pytest.raises(ValidationError):
        PortfolioResearchAgentRequest.model_validate(
            {
                "operation": "EVIDENCE_SELECT",
                "analysis_publication_hash": chosen,
                "chosen_by": "HUMAN",
            }
        )

    # An ineligible publication is still refused for either caller.
    status, body = service.request(
        "/api/evidence-select",
        method="POST",
        payload={"result_hash": result_hash, "analysis_publication_hash": "0" * 64},
    )
    assert status == 400, (status, body)
    refused = service.agent(
        PortfolioResearchAgentRequest(
            operation="EVIDENCE_SELECT",
            result_hash=result_hash,
            analysis_publication_hash="0" * 64,
        )
    )
    assert "refused" in refused


def test_the_actions_still_require_the_session_token(service: _Service) -> None:
    for path in ("/api/evidence-refresh", "/api/cro-review"):
        status, _body = service.request(path, method="POST", payload={}, token=None)
        assert status in {401, 403}, (path, status)


def test_the_agent_tool_and_the_buttons_share_one_operation_owner(service: _Service) -> None:
    """Human and Agent see the same states and receive the same refusals and reuse."""

    result_hash = service.result_hash()
    assert service.agent(PortfolioResearchAgentRequest(operation="EVIDENCE_CRO")) == (
        service.evidence_cro()
    )
    refreshed = service.agent(
        PortfolioResearchAgentRequest(operation="EVIDENCE_REFRESH", result_hash=result_hash)
    )
    assert refreshed["disposition"] == "ADMITTED"
    service.drain()
    reviewed = service.agent(
        PortfolioResearchAgentRequest(operation="CRO_REVIEW", result_hash=result_hash)
    )
    assert reviewed["disposition"] == "ADMITTED"
    service.drain()

    published = service.evidence_cro(result_hash)
    assert published["state"] == "REVIEW_PUBLISHED"
    assert (
        service.agent(
            PortfolioResearchAgentRequest(operation="EVIDENCE_CRO", result_hash=result_hash)
        )
        == published
    )
    assert (
        service.post("/api/cro-review", {"result_hash": result_hash})["disposition"]
        == "REUSED_EXACT"
    )
    assert (
        service.agent(
            PortfolioResearchAgentRequest(operation="CRO_REVIEW", result_hash=result_hash)
        )["disposition"]
        == "REUSED_EXACT"
    )


# ================================================================== restart


class _patched:
    """Swap one attribute for the duration of a step, and put it back."""

    def __init__(self, owner: object, name: str, value: object) -> None:
        self.owner = owner
        self.name = name
        self.original = getattr(owner, name)
        setattr(owner, name, value)

    def undo(self) -> None:
        setattr(self.owner, self.name, self.original)


def test_a_real_restart_resumes_both_task_kinds_exactly_once(tmp_path: Path, http_book) -> None:
    """A real restart resumes both task kinds exactly once."""

    workspace, report = http_book
    authority = build_authority(tmp_path=tmp_path, report=report)

    first = start_service(workspace, authority, tmp_path)
    result_hash = first.result_hash()
    interrupted = _patched(
        AlternativeEvidenceDocumentTaskAdapter, "verify_stage", _raise_interruption
    )
    try:
        refreshed = first.post("/api/evidence-refresh", {"result_hash": result_hash})
        first.drain()
    finally:
        interrupted.undo()
    evidence_task = UUID(refreshed["task_id"])
    assert first.registry.task(evidence_task).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
    assert first.evidence_cro()["state"] == "EVIDENCE_REFRESH_IN_PROGRESS"
    first.session.stop()
    assert first.session.dispatcher is None
    assert first.session.web is None
    assert first.session.session is None

    second = start_service(workspace, authority, tmp_path)
    try:
        assert second.session.review is not first.session.review
        assert second.session.resumed_task_ids == (evidence_task,)
        second.drain()
        assert second.registry.task(evidence_task).lifecycle is TaskLifecycle.SUCCEEDED
        assert second.evidence_cro()["state"] == "ALTERNATIVE_EVIDENCE_READY_FOR_REVIEW"
        analyses = second.review.artifacts.values(
            "analysis-publications", AlternativeEvidenceAnalysisPublication
        )
        assert len(analyses) == 1, "resumed once, published once"

        interrupted = _patched(PortfolioReviewTaskAdapter, "verify_stage", _raise_interruption)
        try:
            reviewed = second.post("/api/cro-review", {"result_hash": result_hash})
            second.drain()
        finally:
            interrupted.undo()
        review_task = UUID(reviewed["task_id"])
        assert second.registry.task(review_task).lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        actor_calls = len(second.review.review_actor.observed_deadlines)
        second.session.stop()

        third = start_service(workspace, authority, tmp_path)
        try:
            assert third.session.resumed_task_ids == (review_task,)
            third.drain()
            assert third.registry.task(review_task).lifecycle is TaskLifecycle.SUCCEEDED
            assert third.evidence_cro()["state"] == "REVIEW_PUBLISHED"
            stored = third.review.review_publications.store.values(
                "cro-review-publications", PortfolioReviewPublication
            )
            assert len(stored) == 1, "resumed once, published once"
            assert len(third.review.review_actor.observed_deadlines) == actor_calls + 1
        finally:
            third.session.stop()
    finally:
        if second.session.web is not None:
            second.session.stop()


def test_a_request_that_names_no_book_is_told_what_names_one() -> None:
    """regression (V242): a bundle asked for with its Task alone read "This workspace holds no
    sealed book", though the workspace held one its request did not name; the refusal says what
    names a book and which request writes the whole selector."""

    from alphalattice.control.product_host.composition.evidence_review_application import (
        EvidenceReviewApplication,
    )

    app = SimpleNamespace(default_selector=lambda _selector: None, has_evidence_authority=True)
    outcome = EvidenceReviewApplication._review_selector(app, None)  # type: ignore[arg-type]
    assert outcome.disposition == "REFUSED_NO_BOOK_TO_REVIEW"
    assert "`experiment_task_id`" in outcome.detail and "`packet_<unit>`" in outcome.detail
    # The way on is a request, not only words (V295): the history read lists the books.
    assert outcome.next_requests == {"history": {"operation": "RESEARCH_HISTORY"}}


def test_each_packet_comes_with_its_analyst_bundle_request() -> None:
    """requirement (V295; LAWS OP12): after O2 a lead assembled the five-field selector of an
    Analyst's bundle by hand; every packet request now comes with the bundle request beside
    it, the book, Task and unit filled and only the directory left to choose."""

    from alphalattice.control.product_host.composition.evidence_review_application import (
        with_analyst_bundles,
    )
    from alphalattice.interface.local_application.cli_contract import choices

    book = {
        "experiment_task_id": str(UUID(int=1)),
        "experiment_receipt_hash": "a" * 64,
        "portfolio_session": "2026-09-01",
    }
    packet = {
        "operation": "EVIDENCE_PACKET",
        **book,
        "task_id": str(UUID(int=2)),
        "evidence_unit_id": "unit-1",
        "evidence_as_of": "2026-09-01T00:00:00+00:00",
    }
    prepare = {"operation": "EVIDENCE_PREPARE", **book}
    offered = with_analyst_bundles({"packet_unit-1": packet, "prepare": prepare})

    assert set(offered) == {"packet_unit-1", "prepare", "analyst_bundle_unit-1"}
    bundle = offered["analyst_bundle_unit-1"]
    assert bundle == {
        "operation": "AGENT_BUNDLE_PREPARE",
        "agent_role": "ANALYST",
        **book,
        "task_id": str(UUID(int=2)),
        "evidence_unit_id": "unit-1",
    }
    assert choices(bundle) == ["bundle_directory"]
    single = with_analyst_bundles({"packet": {**packet, "evidence_unit_id": "unit-1"}})
    assert single["analyst_bundle"]["task_id"] == str(UUID(int=2))


def test_a_position_basis_beside_another_book_is_refused_by_its_name() -> None:
    """regression (V290; LAWS OP12): an experiment's selector with the position basis its
    experiment recorded was refused `evidence_review_update_selector_invalid`, naming neither
    the field nor the way on; the refusal names the field and offers the request without it."""

    from alphalattice.control.product_host.composition.plain_refusals import refused
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
        BookSelector,
        PortfolioEvidenceReviewError,
    )

    book = {
        "experiment_task_id": UUID(int=1),
        "experiment_receipt_hash": "a" * 64,
        "portfolio_session": "2026-09-01",
    }
    with pytest.raises(PortfolioEvidenceReviewError) as caught:
        BookSelector(**book, position_basis="OBSERVED_RESEARCH_ENTRY")
    code = str(caught.value)
    assert code == "product_host.evidence_review_update_selector_invalid:position_basis"
    with pytest.raises(PortfolioEvidenceReviewError, match=r"update_selector_invalid$"):
        BookSelector(update_task_id=UUID(int=3), position_basis="OBSERVED_RESEARCH_ENTRY")
    retry = {"operation": "AGENT_BUNDLE_PREPARE", "agent_role": "CRO"} | {
        key: str(value) for key, value in book.items()
    }
    answer = refused(code, book=retry)
    assert answer["next_requests"] == {"without_position_basis": retry}
    assert answer["next_action"] == "SEND_THE_OFFERED_REQUEST"
    assert "`position_basis`" in answer["detail"]


def test_an_unprepared_units_packet_says_why_and_offers_the_coverage_read():
    """An unprepared units packet says why and offers the coverage read."""

    from alphalattice.control.product_host.composition.plain_refusals import (
        EVIDENCE_UNIT_CODES,
        unit_refusal,
    )

    book = {"result_hash": "a" * 64}
    empty = unit_refusal(
        "alternative_evidence.unit_not_prepared:alternative_evidence.document_set_empty", book
    )
    assert empty is not None and empty["status"] == "REFUSED"
    assert "None of this unit's issuers holds a source document" in empty["detail"]
    assert empty["next_requests"] == {"coverage": {"operation": "EVIDENCE_CRO", **book}}
    for code in EVIDENCE_UNIT_CODES:
        answer = unit_refusal(code, None)
        assert answer is not None and answer["detail"], code
        assert answer["next_requests"] == {"coverage": {"operation": "EVIDENCE_CRO"}}
    assert unit_refusal("alternative_evidence.delivery_continuation_stale", book) is None


def test_a_review_request_naming_a_book_the_workspace_does_not_hold_is_refused_in_words(
    tmp_path: Path,
    http_book,
) -> None:
    """A review request naming a book the workspace does not hold is refused in words."""

    from alphalattice.interface.local_application.cli_contract import refusal_words

    workspace, report = http_book
    service = start_service(workspace, build_authority(tmp_path=tmp_path, report=report), tmp_path)
    try:
        session = "2026-09-29"
        cases = {
            "product_host.evidence_review_result_unknown": {"result_hash": "f" * 64},
            "product_host.evidence_review_handoff_unknown": {"handoff_hash": "e" * 64},
            "product_host.evidence_review_experiment_not_published": {
                "experiment_task_id": str(UUID(int=7)),
                "experiment_receipt_hash": "d" * 64,
                "portfolio_session": session,
            },
            "product_host.evidence_review_update_task_mismatch": {
                "update_task_id": str(UUID(int=9)),
                "update_publication_hash": "c" * 64,
                "position_basis": "CONDITIONAL_ESTIMATE",
            },
        }
        for code, selector in cases.items():
            for route in ("/api/evidence-cro", "/api/cro/dossier", "/api/evidence/preview"):
                status, body = service.request(route + "?" + urllib.parse.urlencode(selector))
                assert (status, body["failure_code"]) == (400, code), (route, body)
                assert body["detail"] == refusal_words(code)["detail"], (route, body)
                assert body["next_action"] == refusal_words(code)["next_action"]
        # Present but not a result: the store's refusal stands, never read as unknown.
        result = service.result_hash()
        (stored,) = [p for p in workspace.rglob(f"{result}.json") if p.parent.name == "results"]
        stored.write_bytes(b"{}")
        status, body = service.request("/api/evidence-cro?result_hash=" + result)
        assert (status, body["refused"]) == (400, "content_store.artifact_tampered"), body
    finally:
        service.session.stop()


def test_the_report_door_distinguishes_unknown_missing_child_and_corrupt_content(
    tmp_path: Path, http_book
):
    """CONTRACT: an unknown request offers kept results; lost sealed work keeps recovery."""
    workspace, report = http_book
    service = start_service(workspace, build_authority(tmp_path=tmp_path, report=report), tmp_path)
    try:
        status, body = service.request("/api/report?result_hash=" + "f" * 64)
        assert (status, body["failure_code"]) == (200, "portfolio_research.result_not_found")
        assert body["next_requests"] == {"results": {"operation": "RESULTS"}}
        identity = service.result_hash()
        (child,) = [
            p for p in workspace.rglob(f"{report.report_hash}.json") if p.parent.name == "reports"
        ]
        original = child.read_bytes()
        child.unlink()
        status, body = service.request("/api/report?result_hash=" + identity)
        assert status == 200 and body["failure_code"] == (
            f"content_store.artifact_missing:{report.report_hash}"
        ), body
        assert report.report_hash in body["detail"] and "restore a backup" in body["detail"]
        assert body["next_requests"] == {"workspace": {"operation": "WORKSPACE_SHOW"}}
        child.write_bytes(b"{}")
        status, body = service.request("/api/report?result_hash=" + identity)
        assert (status, body["failure_code"]) == (200, "content_store.artifact_tampered"), body
        child.write_bytes(original)
    finally:
        service.session.stop()


def _invented(full: dict[str, Any], handle: str) -> dict[str, Any]:
    """The Analyst's answer with its first finding citing an excerpt the bundle does not hold."""
    value = json.loads(json.dumps(full))
    value["findings"][0]["cite"] = [handle]
    return value


def test_two_answers_to_one_bundle_sent_at_once_are_numbered_one_and_two_and_both_kept(
    service: _Service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two answers to one bundle sent at once are numbered one and two and both kept."""

    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector
    from alphalattice.protocols.actor_execution.answers import AgentRun, answer_digest
    from tests.alternative_evidence_desk.planted_corpus import _CitingActor

    selected = {"result_hash": service.result_hash()}
    prepared = service.post("/api/evidence/prepare", selected)
    service.drain()
    task_id = UUID(prepared["task_id"])
    adapter = service.review.evidence_task_adapter
    packet, context = adapter.analysis_context(
        task_id, now=service.review.clock(), unit_id=ONE_UNIT
    )
    full = _CitingActor(topics=adapter.resources.analysis_actor.topics)(packet=packet)
    answers = {
        name: _invented(full.answer.model_dump(mode="json"), handle)
        for name, handle in (("a", "S998"), ("b", "S999"))
    }
    runs = {
        name: AgentRun(
            host="claude-code",
            session_id="lead-session",
            agent_id=f"analyst-{name}",
            role="alphalattice_evidence_analyst",
            basis="NOT_OBSERVED",
        )
        for name in answers
    }
    read = {"a": ("README.md",), "b": ("README.md", "material-1.md")}
    store = service.review.artifacts
    original = store.publish
    held, release = threading.Event(), threading.Event()

    def publish(category: str, identity: str, model: Any) -> None:
        # The first answer stops at its write, counted and not yet filed, while the second is sent.
        if category == ANSWER_CATEGORY and not held.is_set():
            held.set()
            assert release.wait(60)
        original(category, identity, model)

    monkeypatch.setattr(store, "publish", publish)
    results: dict[str, Any] = {}

    def send(name: str) -> None:
        results[name] = service.review.submit_analysis(
            dispatcher=service.session.dispatcher,
            selector=BookSelector(**selected),
            task_id=task_id,
            context_hash=str(context["analysis_context_hash"]),
            answer=answers[name],
            caller="EXTERNAL_AUTOMATION",
            unit_id=ONE_UNIT,
            read_files=read[name],
            agent_run=runs[name],
        )

    first = threading.Thread(target=send, args=("a",))
    first.start()
    assert held.wait(60)
    second = threading.Thread(target=send, args=("b",))
    second.start()
    second.join(0.5)  # sent while the first is still being filed
    release.set()
    first.join(60)
    second.join(60)
    assert {name: value["answer"]["number"] for name, value in results.items()} == {"a": 1, "b": 2}
    assert (results["a"]["answer"]["rounds_left"], results["b"]["answer"]["rounds_left"]) == (2, 1)
    binding = {
        "role": "alternative_evidence.analyst",
        "task_id": str(task_id),
        "unit_id": ONE_UNIT,
        "analysis_context_hash": context["analysis_context_hash"],
    }
    bundle_key = str(canonical_hash(binding))
    for number, (name, handle) in enumerate((("a", "S998"), ("b", "S999")), start=1):
        kept = store.load(ANSWER_CATEGORY, answer_slot(bundle_key, number), AgentAnswerRecord)
        assert kept.answer_digest == answer_digest(answers[name]), name
        problems = [value.model_dump(mode="json") for value in kept.problems]
        assert problems == [{"item": 1, "text": f"{handle} is not an excerpt of this bundle."}]
        assert (kept.read_files, kept.agent_run, kept.corrections_used) == (
            read[name],
            runs[name],
            number,
        )
    assert not store.exists(ANSWER_CATEGORY, answer_slot(bundle_key, 3))


def test_no_analyst_is_credited_with_another_bundles_answer(
    service: _Service, tmp_path: Path
) -> None:
    """No analyst is credited with another bundle's answer."""

    from alphalattice.interface.local_application.cli_contract import (
        REQUEST_PROVENANCE,
        RequestProvenance,
    )
    from tests.alternative_evidence_desk.planted_corpus import _CitingActor
    from tests.alternative_evidence_desk.portfolio_coverage_support import run_one

    selected = {"result_hash": service.result_hash()}
    prepared = service.post("/api/evidence/prepare", selected)
    service.drain()
    names = ("one", "two", "three")
    directories = {name: tmp_path / f"analyst-{name}" for name in names}
    for directory in directories.values():
        bundle = run_one(
            service,
            {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "ANALYST",
                "task_id": prepared["task_id"],
                "evidence_unit_id": ONE_UNIT,
                "bundle_directory": str(directory),
                **selected,
            },
        )
        assert bundle["status"] == "AGENT_BUNDLE_READY", bundle
        assert bundle["bundle_reference"] == bundle_slot(str(directory)), bundle
    session = "lead-session"
    adapter = service.review.evidence_task_adapter
    packet = adapter.prepared_packet(
        UUID(prepared["task_id"]), now=service.review.clock(), unit_id=ONE_UNIT
    )
    full = _CitingActor(topics=adapter.resources.analysis_actor.topics)(packet=packet)
    handles = {"one": "S997", "two": "S998", "three": "S999"}
    answered: dict[str, Any] = {}
    token = REQUEST_PROVENANCE.set(RequestProvenance(vendor="claude-code", session=session))
    try:
        for name in ("two", "one", "three"):
            answered[name] = run_one(
                service,
                {
                    "operation": "AGENT_ANSWER_SUBMIT",
                    "bundle_directory": str(directories[name]),
                    "agent_answer": _invented(full.answer.model_dump(mode="json"), handles[name]),
                },
            )
    finally:
        REQUEST_PROVENANCE.reset(token)
    credited = {name: value["recorded_agent"] for name, value in answered.items()}
    for name in names:
        assert credited[name]["basis"] == "NOT_OBSERVED", credited
        assert (credited[name]["agent_id"], credited[name]["model"]) == (None, None), credited
        assert (credited[name]["host"], credited[name]["session_id"]) == ("claude-code", session)
    context = EvidenceReviewBundles(service.review).agent_bundle(str(directories["one"]))
    assert context is not None
    binding = {
        "role": "alternative_evidence.analyst",
        "task_id": prepared["task_id"],
        "unit_id": ONE_UNIT,
        "analysis_context_hash": context.submission["analysis_context_hash"],
    }
    bundle_key = str(canonical_hash(binding))
    for number, name in enumerate(("two", "one", "three"), start=1):
        kept = service.review.artifacts.load(
            ANSWER_CATEGORY, answer_slot(bundle_key, number), AgentAnswerRecord
        )
        assert kept.agent_run is not None
        assert kept.agent_run.model_dump(mode="json") == credited[name], name


def test_a_specialists_receipt_names_its_tasks_state_so_a_wait_follows_it() -> None:
    """A specialist's receipt names its task's state so a wait follows it."""

    from alphalattice.interface.local_application.cli_contract import outcome_of

    answer = {"verdict": "ACCEPTED", "accepted_items": [1], "dropped": []}
    for role in ("ANALYST", "CRO"):
        for lifecycle, expected in (("QUEUED", "PENDING"), ("BLOCKED", "REFUSED")):
            admitted = agent_answer_result(
                role,
                {
                    "disposition": "ADMITTED",
                    "task_id": "t-1",
                    "lifecycle": lifecycle,
                    "answer": answer,
                },
            )
            assert admitted["status"] == admitted["answer_status"] == "ACCEPTED"
            assert (admitted["task_lifecycle"], outcome_of(admitted)) == (lifecycle, expected)
            assert all(word in admitted["message"] for word in ("accepted", "t-1", lifecycle))
            assert admitted["next_requests"]["task"] == {"operation": "STATUS", "task_id": "t-1"}
        reused = agent_answer_result(
            role,
            {
                "disposition": "REUSED_EXACT",
                "task_id": None,
                "publication_task_id": "t-0",
                "answer": answer,
            },
        )
        assert "task_lifecycle" not in reused and outcome_of(reused) == "OK"


def test_review_continue_carries_a_real_book_from_its_analyst_to_its_published_review(
    service, tmp_path, capsys
):
    """Review continue carries a real book from its analyst to its published review."""

    from alphalattice.interface.local_application.cli import main
    from alphalattice.interface.local_application.client import LocalResearchClient
    from tests.alternative_evidence_desk.planted_corpus import _CitingActor

    adapter = service.review.evidence_task_adapter
    topics = adapter.resources.analysis_actor.topics
    adapter.resources = replace(adapter.resources, analysis_actor=None)
    service.review.model_authority_admitted = False
    service.review.review_actor = None
    client = LocalResearchClient(service.session.workspace)
    selected = {"result_hash": service.result_hash()}
    workspace = ("--workspace", str(service.session.workspace))
    prepared = client.request(
        client.request({"operation": "EVIDENCE_PREVIEW", **selected})["next_requests"]["prepare"]
    )
    service.drain()
    source_id = prepared["task_id"]

    def run(*arguments: str) -> tuple[int, dict[str, Any]]:
        code = main(["--view", "full", *arguments], serve=lambda _: 99)
        return code, json.loads(capsys.readouterr().out)["data"]

    def run_draining(*arguments: str) -> tuple[int, dict[str, Any]]:
        result: list[tuple[int, dict[str, Any]]] = []
        worker = threading.Thread(target=lambda: result.append(run(*arguments)))
        worker.start()
        deadline = time.monotonic() + 240
        while worker.is_alive() and time.monotonic() < deadline:
            service.drain()
            worker.join(0.2)
        assert not worker.is_alive(), "review continue did not end"
        return result[0]

    analysts = tmp_path / "analysts"
    code, analyst = run(
        *workspace,
        *("bundle", "prepare", "--role", "ANALYST"),
        *("--task", source_id, "--unit", ONE_UNIT, "--result", selected["result_hash"]),
        *("--dir", str(analysts / "analyst-u01")),
    )
    assert code == 0, analyst
    packet = adapter.prepared_packet(UUID(source_id), now=service.review.clock(), unit_id=ONE_UNIT)
    full = _CitingActor(topics=topics)(packet=packet).answer.model_dump(mode="json")
    Path(analyst["answer_file"]).write_text(
        json.dumps({**full, "read": [item["name"] for item in analyst["files"]]}),
        encoding="utf-8",
    )
    code, carried = run_draining(
        *workspace, "review", "continue", "--dir", str(analysts), "--cro-dir", str(tmp_path / "cro")
    )
    assert code == 0 and carried["status"] == "CRO_BUNDLE_READY", carried
    assert [receipt["agent_role"] for receipt in carried["receipts"]] == ["ANALYST"]
    cro = carried["cro_bundle"]
    assert Path(cro["bundle_directory"]) == (tmp_path / "cro").resolve()
    assert (tmp_path / "cro" / "README.md").is_file()
    risk = {
        "findings": ["F1"],
        "why": "The cited finding is adverse for a held issuer.",
        "severity": "HIGH",
        "confidence": "SUPPORTED",
        "recommendation": "Size the position down.",
    }
    Path(cro["answer_file"]).write_text(
        json.dumps(
            {
                "risks": [risk],
                "summary": "One major negative in the evidence read.",
                "read": [item["name"] for item in cro["files"]],
            }
        ),
        encoding="utf-8",
    )
    code, published = run_draining(*workspace, "review", "continue", "--dir", str(tmp_path / "cro"))
    assert code == 0 and published["status"] == "REVIEW_PUBLISHED", published
    assert [receipt["agent_role"] for receipt in published["receipts"]] == ["CRO"]
    assert published["evidence"]["state"]
