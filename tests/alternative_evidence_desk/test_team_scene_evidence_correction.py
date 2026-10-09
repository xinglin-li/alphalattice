"""Card 33 (engineering consumer): one same-context refusal, correction and owner readback.

The controlled Analyst/CRO fixtures of the evidence review route, driven the way
external actors drive them: the Analyst's brief is refused twice by its domain
owner (a changed analysis binding; an invented span citation), the corrected
brief is admitted and published, the CRO assesses the same dossier and the
review is published. The team's messages name these objects through the
accepted public ingress, and the scene the real consumer modules derive from
the real feed shows the refusals, the admission, the exact reuse and the
publications on the same references, in recorded order, with the original
objection retained. Fixture actor output is fixture output, not native-host
proof; no model or Provider is involved.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application.client import LocalResearchClient
from tests.alternative_evidence_desk.planted_corpus import _CitingActor
from tests.alternative_evidence_desk.review_dossiers import controlled_answer
from tests.alternative_evidence_desk.review_http_support import (
    _Service,
    build_authority,
    build_workspace,
    start_service,
)
from tests.portfolio_strategy_lab.local_web_support import run_node

MODULES = (
    Path(__file__).resolve().parents[2]
    / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
)
RUNNER = Path(__file__).resolve().parents[1] / "portfolio_strategy_lab/workbench_team_readback.cjs"
SESSION = "019a0b1c-parent-session"
SCOPE = sha256(b"parent-session|evidence").hexdigest()


@pytest.fixture
def service(tmp_path: Path) -> Iterator[_Service]:
    workspace, report = build_workspace(tmp_path)
    authority = build_authority(
        tmp_path=tmp_path, report=report, with_runtime=True, with_actor=True
    )
    started = start_service(workspace, authority, tmp_path)
    try:
        yield started
    finally:
        started.session.stop()


class _Producer:
    """Documents shaped like the lead's bridge emits them; the sequence is the producer's."""

    def __init__(self, client: LocalResearchClient) -> None:
        self.client, self.sequence = client, 0

    def _publish(self, kind: str, summary: str, subject: dict[str, str]) -> dict[str, Any]:
        self.sequence += 1
        answer = self.client.publish_event(
            {
                "event_kind": kind,
                "producer_id": "codex-native",
                "producer_session": SCOPE,
                "producer_sequence": self.sequence,
                "occurred_at": datetime(2026, 9, 14, 13, 0, self.sequence, tzinfo=UTC).isoformat(),
                "summary": summary,
                "subject": {
                    **subject,
                    "source_time_kind": "BRIDGE_RECEIVED",
                    "native_event_id": sha256(f"{kind}|{self.sequence}".encode()).hexdigest(),
                },
                "correlation_ids": [SESSION],
            }
        )
        assert answer["status"] == "APPENDED", answer
        return answer

    def hook(self, event: str, agent: str, role: str, turn: str) -> dict[str, Any]:
        return self._publish(
            "NATIVE_SUBAGENT_START_HOOK"
            if event == "SubagentStart"
            else "NATIVE_SUBAGENT_STOP_HOOK",
            f"{role}: {event} hook observed.",
            {
                "native_session_id": SESSION,
                "native_turn_id": turn,
                "native_agent_id": agent,
                "role": role,
                "input_channel": "CODEX_HOOK",
                "native_hook_event": event,
                "terminal_state": "NOT_ESTABLISHED",
                "stop_hook_active": "unknown",
            },
        )

    def message(self, agent: str, role: str, kind: str, text: str, **extra: str) -> dict[str, Any]:
        return self._publish(
            "NATIVE_COORDINATION_MESSAGE",
            text,
            {
                "native_session_id": SESSION,
                "native_agent_id": agent,
                "role": role,
                "message_kind": kind,
                "message_id": extra.pop("message_id", f"m-{self.sequence + 1}"),
                "message_sha256": sha256(text.encode()).hexdigest(),
                "message_bytes": str(len(text.encode())),
                "input_channel": "ACTOR_DECLARED",
                **extra,
            },
        )


def _consumer(feed: dict[str, Any], tmp: Path, *selection: str) -> dict[str, Any]:
    run_node(None)
    path = tmp / "feed.json"
    path.write_text(json.dumps(feed), encoding="utf-8")
    result = run_node(
        [str(RUNNER), str(MODULES), str(path), *selection],
        capture_output=True,
        encoding="utf-8",
        timeout=20,
        check=True,
    )
    return json.loads(result.stdout)


def test_scene_shows_the_same_context_refusals_correction_and_owner_readback(
    service: _Service, tmp_path: Path
) -> None:
    review_risks = service.review.review_actor.risks
    adapter = service.review.evidence_task_adapter
    topics = adapter.resources.analysis_actor.topics
    adapter.resources = replace(adapter.resources, analysis_actor=None)
    service.review.model_authority_admitted = False
    service.review.review_actor = (
        None  # external actors only: the team submits through the CLI/client
    )
    client = LocalResearchClient(service.session.workspace)
    producer = _Producer(client)
    lead, analyst, cro = SESSION, "child-analyst-e1", "child-cro-e2"
    selected = {"result_hash": service.result_hash()}
    result_ref = "result:" + selected["result_hash"]
    start = client.activity()
    tasks_before = len(service.registry.tasks())

    # The PM assigns the saved book by its History reference; the host names the Analyst child.
    producer.message(
        lead,
        "research_lead",
        "assignment",
        "Prepare and submit an evidence brief on the saved book.",
        recipient_id=analyst,
        reference=result_ref,
    )
    producer.hook("SubagentStart", analyst, "alternative_analyst", "turn-e1")
    # Preparation through the product: the packet's source Task is the same context for all
    # that follows.
    preview = client.request({"operation": "EVIDENCE_PREVIEW", **selected})
    assert preview["status"] == "EVIDENCE_PREPARATION_READY"
    prepared = client.request(preview["next_requests"]["prepare"])
    assert prepared["disposition"] == "ADMITTED"
    service.drain()
    source_id = UUID(prepared["task_id"])
    assert service.registry.task(source_id).lifecycle is TaskLifecycle.SUCCEEDED
    # Every book is a run: the packet request is the one the reply hands
    # out, selected by its operation as a client selects it.
    (packet_request,) = [
        value
        for value in prepared["next_requests"].values()
        if value.get("operation") == "EVIDENCE_PACKET"
    ]
    exported = client.request(packet_request)
    assert exported["status"] == "EVIDENCE_ANALYST_PACKET_READY"
    packet = adapter.prepared_packet(
        source_id, now=service.review.clock(), unit_id=packet_request["evidence_unit_id"]
    )
    answer = _CitingActor(topics=topics)(packet=packet).answer.model_dump(mode="json")
    document = {**exported["submission_template"], "analysis_answer": answer}

    # The Analyst declares the packet Task (verified by Task Control) and submits; the domain
    # owner refuses twice in this same context.
    verified = producer.message(
        analyst,
        "alternative_analyst",
        "answer",
        "Brief drafted against the packet; submitting.",
        task_id=str(source_id),
        message_id="answer-1",
    )
    assert verified["task_verified"] is True
    binding_changed = client.request({**document, "analysis_context_hash": "a" * 64})
    assert "external_analysis_binding_changed" in binding_changed["refused"]
    invented = json.loads(json.dumps(answer))
    invented["findings"][0]["cite"] = ["S999"]
    bad_span = client.request({**document, "analysis_answer": invented})
    assert bad_span["status"] == "CORRECT"
    assert bad_span["answer"]["problems"] == [
        {"item": 1, "text": "S999 is not an excerpt of this bundle."}
    ]
    producer.message(
        cro,
        "independent_cro",
        "objection",
        "Two submissions were refused by the evidence owner; resubmit citing the packet's spans.",
        task_id=str(source_id),
        recipient_id=lead,
        message_id="objection-1",
    )
    # The corrected submission in the same context is admitted, published and exactly reusable.
    submitted = client.request(document)
    assert submitted["disposition"] == "ADMITTED"
    service.drain()
    analysis_task = UUID(submitted["task_id"])
    assert service.registry.task(analysis_task).lifecycle is TaskLifecycle.SUCCEEDED
    reused = client.request(document)
    assert reused["disposition"] == "REUSED_EXACT" and reused["task_id"] is None
    publication = reused["analysis_publication_hash"]
    view = adapter.published_analysis(analysis_task, now=service.review.clock())
    assert (
        view.publication.publication_hash == publication
    )  # the owner's own readback of the accepted output
    producer.message(
        analyst,
        "alternative_analyst",
        "answer",
        "Resubmitted brief admitted and published; the publication hash is referenced.",
        reference=publication,
        message_id="answer-2",
    )
    # The CRO assesses the same dossier; the review is published.
    dossier = client.request({"operation": "CRO_REVIEW_DOSSIER", **selected})
    assert dossier["status"] == "CRO_DOSSIER_READY"
    assessed = client.request(
        {
            **dossier["submission_template"],
            "review_answer": controlled_answer(dossier["dossier"], *review_risks).model_dump(
                mode="json"
            ),
        }
    )
    assert assessed["disposition"] == "ADMITTED"
    service.drain()
    review_task = UUID(assessed["task_id"])
    assert service.registry.task(review_task).lifecycle is TaskLifecycle.SUCCEEDED
    cro_view = service.evidence_cro()
    assert cro_view["state"] == "REVIEW_PUBLISHED"
    history = client.request({"operation": "RESEARCH_HISTORY", "history_limit": 50})["entries"]
    review_entry = next(e for e in history if e.get("review_publication_hash"))
    review_ref = "review:" + review_entry["review_publication_hash"]
    producer.message(
        cro,
        "independent_cro",
        "answer",
        "Assessment submitted and published; the review is referenced.",
        reference=review_ref,
        message_id="answer-3",
    )
    producer.message(
        lead,
        "research_lead",
        "pm_response",
        "Accepted. The two refusals and the objection stay on record.",
        recipient_id=cro,
    )
    assert len(service.registry.tasks()) == tasks_before + 3

    # The real feed, History and owner answers go to the real consumer modules.
    page = client.activity(after=start["cursor"], limit=200)
    assert page["disposition"] == "CONTINUED"
    reads = {
        "/api/report?result_hash=" + selected["result_hash"]: client.request(
            {"operation": "REPORT", **selected}
        ),
        "/api/status?task_id=" + str(analysis_task): client.request(
            {"operation": "STATUS", "task_id": str(analysis_task)}
        ),
    }
    scene = _consumer(
        {
            "pages": [start, page],
            "history": history,
            "reads": reads,
            "tasks": [],
            "resolve": [result_ref, publication, str(analysis_task), review_ref],
            "verify": [result_ref, str(analysis_task)],
        },
        tmp_path,
    )
    (session,) = scene["sessions"]
    assert session["id"] == SESSION
    assert set(session["references"]) == {
        result_ref,
        selected["result_hash"],
        str(source_id),
        publication,
        review_ref,
        review_entry["review_publication_hash"],
    }
    facts = [
        (
            f["operation"] or f["lifecycle"],
            f["phase"],
            f["status"],
            f["failure_code"],
            [r["operation"] for r in f["followed"]],
        )
        for f in session["facts"]
    ]
    assert facts == [
        ("EVIDENCE_PREVIEW", "REQUESTED", None, None, []),
        ("EVIDENCE_PREVIEW", "RETURNED", "EVIDENCE_PREPARATION_READY", None, []),
        ("EVIDENCE_PREPARE", "REQUESTED", None, None, []),
        ("EVIDENCE_PREPARE", "RETURNED", "ADMITTED", None, []),
        ("SUCCEEDED", None, None, None, []),  # the packet Task: declared by the Analyst, verified
        ("EVIDENCE_ANALYSIS_SUBMIT", "REQUESTED", None, None, []),
        (
            "EVIDENCE_ANALYSIS_SUBMIT",
            "FAILED",
            "FAILED",
            "alternative_evidence.external_analysis_binding_changed",
            [],
        ),
        ("EVIDENCE_ANALYSIS_SUBMIT", "REQUESTED", None, None, []),
        # The invented citation is an answer returned for correction, not a failure.
        (
            "EVIDENCE_ANALYSIS_SUBMIT",
            "RETURNED",
            "CORRECT",
            "alternative_evidence.answer_problems",
            [],
        ),
        ("EVIDENCE_ANALYSIS_SUBMIT", "REQUESTED", None, None, []),
        ("EVIDENCE_ANALYSIS_SUBMIT", "RETURNED", "ADMITTED", None, []),
        ("SUCCEEDED", None, None, None, ["EVIDENCE_ANALYSIS_SUBMIT"]),  # one owner-named hop
        ("EVIDENCE_ANALYSIS_SUBMIT", "REQUESTED", None, None, []),
        (
            "EVIDENCE_ANALYSIS_SUBMIT",
            "RETURNED",
            "REUSED_EXACT",
            None,
            ["EVIDENCE_ANALYSIS_SUBMIT"],
        ),
        ("CRO_REVIEW_SUBMIT", "REQUESTED", None, None, []),
        ("CRO_REVIEW_SUBMIT", "RETURNED", "ADMITTED", None, []),
        ("SUCCEEDED", None, None, None, ["CRO_REVIEW_SUBMIT"]),
    ]
    withheld = session["facts"][8]
    assert withheld["observation"] and "unknown_span_reference" not in json.dumps(
        withheld
    )  # the detail stays with the owner
    assert publication in session["facts"][13]["named"]  # the exact reuse names the publication
    assert [e["messageKind"] for e in session["entries"] if e["kind"] == "message"] == [
        "assignment",
        "answer",
        "objection",
        "answer",
        "answer",
        "pm_response",
    ]
    objection = next(e for e in session["entries"] if e["messageKind"] == "objection")
    assert objection["taskId"] == str(source_id) and objection["replayOf"] is None
    # Discovery, verification and the honest limits of this reader's owners.
    resolved = scene["resolved"]
    assert (
        resolved[result_ref]["level"] == "verified"
        and resolved[result_ref]["owner"] == "Portfolio result"
    )
    analysis = resolved[str(analysis_task)]
    assert analysis["level"] == "discovered" and analysis["target"]["ref"] == str(analysis_task)
    assert analysis["limit"] == "no owner readback through this reader; open the Task"
    assert resolved[publication]["level"] == "unresolved"
    assert (
        resolved[review_ref]["level"] == "discovered" and resolved[review_ref]["owner"] == "History"
    )
    # The page: the product evidence -- the binding change a failure, the invented citation an
    # answer returned with its problems, the admission, the exact reuse and the review -- in
    # recorded order with its detail in its own block after the thread; in the thread (laws 125,
    # 140) the same operations are counted one-line runs at their recorded places between the
    # member statements, never inside one and never with their detail; the CRO's objection
    # retained among the statements, apart from them.
    html = scene["html"]
    for text in (
        "EVIDENCE_ANALYSIS_SUBMIT · product refusal",
        "alternative_evidence.external_analysis_binding_changed",
        "EVIDENCE_ANALYSIS_SUBMIT · returned for correction",
        "alternative_evidence.answer_problems",
        "EVIDENCE_ANALYSIS_SUBMIT · product admission",
        "EVIDENCE_ANALYSIS_SUBMIT · exact reuse",
        "CRO_REVIEW_SUBMIT · product admission",
        "Task named by EVIDENCE_ANALYSIS_SUBMIT ADMITTED for",
        "Two submissions were refused by the evidence owner",
    ):
        assert text in html, text
    panel = html[html.index('class="panel team-product"') :]
    order = [
        panel.index("EVIDENCE_ANALYSIS_SUBMIT · product refusal"),
        panel.index("EVIDENCE_ANALYSIS_SUBMIT · returned for correction"),
        panel.index("EVIDENCE_ANALYSIS_SUBMIT · product admission"),
        panel.index("EVIDENCE_ANALYSIS_SUBMIT · exact reuse"),
        panel.index("CRO_REVIEW_SUBMIT · product admission"),
    ]
    assert order == sorted(order), "recorded order: refusal, correction, admission, reuse, review"
    assert panel.count("EVIDENCE_ANALYSIS_SUBMIT · product refusal") == 1  # the binding change
    # the invented citation is an answer returned for correction, not a refusal (contract 10.4)
    assert panel.count("EVIDENCE_ANALYSIS_SUBMIT · returned for correction") == 1
    details = (
        "alternative_evidence.external_analysis_binding_changed",
        "alternative_evidence.answer_problems",
        "Task named by EVIDENCE_ANALYSIS_SUBMIT ADMITTED for",
    )
    for detail in details:
        assert detail in panel, detail
    thread = html[html.index('id="teamThread"') : html.index('class="panel team-product"')]
    product_line = r'<li class="team-event team-product-event[^"]*"[^>]*>.*?</li>'
    statements = re.sub(product_line, "", thread, flags=re.S)
    assert "Two submissions were refused by the evidence owner" in statements
    for word in (
        "product refusal",
        "returned for correction",
        "product admission",
        "exact reuse",
        *details,
    ):
        assert word not in statements, word  # an operation is a run of its own, never a statement
    for detail in details:
        assert detail not in thread, detail  # the detail stays in the product's block
    assert thread.count('id="teamThread"') == 1  # one thread: the Participants page retired
    places = [
        thread.index("Brief drafted against the packet"),
        thread.index("EVIDENCE_ANALYSIS_SUBMIT · product refusal"),
        thread.index("EVIDENCE_ANALYSIS_SUBMIT · returned for correction"),
        thread.index("Two submissions were refused by the evidence owner"),
        thread.index("EVIDENCE_ANALYSIS_SUBMIT · product admission"),
        thread.index("EVIDENCE_ANALYSIS_SUBMIT · exact reuse"),
        thread.index("Resubmitted brief admitted"),
        thread.index("CRO_REVIEW_SUBMIT · product admission"),
        thread.index("Assessment submitted and published"),
    ]
    assert places == sorted(places), "each run at its recorded place between the statements"
    # Each declared reference reads back in its exchange's contribution reader with the owner's
    # answer: the saved book verified by the Portfolio owner, the review discovered by History
    # and not verified, the analysis publication unresolved by every owner this reader has.
    by_reference = {
        e["reference"]: scene["readers"][e["observation"]]
        for e in session["entries"]
        if e["reference"]
    }
    # a header names the speaker and the kind, the addressee as the thread writes it (@mention)
    assert 'aria-label="Main PM · Assignment"' in by_reference[result_ref]
    assert "→ @Alternative Analyst</span>" in by_reference[result_ref]
    assert "[verified:verified by Portfolio result]" in by_reference[result_ref]
    assert "CRO · Answer" in by_reference[review_ref]
    assert "[metadata:discovered by History · not verified]" in by_reference[review_ref]
    assert "Alternative Analyst · Answer" in by_reference[publication]
    assert "[metadata:unresolved]" in by_reference[publication]
    objection_reader = scene["readers"][objection["observation"]]
    assert 'aria-label="CRO · Objection"' in objection_reader
    assert "→ @Main PM</span>" in objection_reader
    assert "Two submissions were refused by the evidence owner" in objection_reader

    # The invented citation's verdict is the product's own CORRECT, said as that
    # observation's title and state ("returned for correction"); beyond it the scene
    # states no correction of its own.
    def states_no_correction(text: str) -> bool:
        own = (
            text.lower()
            .replace("returned_for_correction", "")
            .replace("returned for correction", "")
        )
        return "correct" not in own

    assert "returned for correction" in html.lower()
    assert states_no_correction(html), "the scene states no correction of its own"
    assert all(states_no_correction(r) for r in scene["readers"].values())
    assert len(service.registry.tasks()) == tasks_before + 3
