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
import shlex
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.evidence_review_application import (
    ANSWER_CATEGORY,
)
from alphalattice.control.product_host.composition.evidence_review_bundles import (
    EvidenceReviewBundles,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.answers import AgentAnswerRecord, answer_slot
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


def test_the_two_commands_carry_each_specialist_from_its_bundle_to_a_receipt(
    service, tmp_path, capsys, monkeypatch
):
    """A served specialist correction-and-acceptance loop reopens its exact sealed answer."""

    from alphalattice.evidence.alternative_evidence.analysis.views import render_analyst_bundle
    from alphalattice.interface.local_application.cli import main
    from alphalattice.oversight.chief_risk_officer.decision.views import render_review_bundle
    from tests.alternative_evidence_desk.review_dossiers import REPO_ROOT

    monkeypatch.setenv("CODEX_THREAD_ID", "bundle-review-lead")
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
        files = {path.name: path.read_text(encoding="utf-8") for path in directory.iterdir()}
        assert json.loads(files.pop("workspace.json")) == {
            "workspace": str(service.session.workspace.resolve())
        }
        return files

    def unrouted(bundle: dict[str, Any]) -> dict[str, str]:
        """Its files as rendered, the route its README ends with set apart."""
        cut = "\n\n## Your route"
        return {k: v.split(cut)[0].rstrip() for k, v in files_of(bundle).items()}

    def keyed(bundle: dict[str, Any]) -> list[str]:
        """The member's own submit: the bundle's submit with the key its route names."""
        key = re.search(r"--key ([0-9a-f]{32})", files_of(bundle)["README.md"])
        return [*bundle["submit_arguments"], "--key", key.group(1) if key else ""]

    def skill(path: str) -> str:
        return (REPO_ROOT / "skills" / path / "SKILL.md").read_text(encoding="utf-8")

    workspace = ("--workspace", str(service.session.workspace))
    # Commands printed for a POSIX shell, which `shlex` reads back; the PowerShell form is
    # held by its own test.
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
    # The Task queue's waiting places are set beside it, one setting a request.
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
    assert code == 2 and "agent_bundle.analyst_unit_required:u01" in json.dumps(unnamed)
    assert not (tmp_path / "unnamed").exists()
    # Each says how to name the packet, where it answered with no way on.
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
        *("--task", source_id, "--unit", "u01"),
        *("--result", selected["result_hash"]),
        *("--dir", str(tmp_path / "analyst")),
    )
    assert code == 0 and analyst["status"] == "AGENT_BUNDLE_READY"
    assert analyst["files"][0]["name"] == "README.md"
    packet = adapter.prepared_packet(UUID(source_id), now=service.review.clock(), unit_id="u01")
    view = render_analyst_bundle(
        packet, task_procedure=skill("alternative-evidence/governed-evidence-analysis")
    )
    assert unrouted(analyst) == {k: v.rstrip() for k, v in view.files}
    readme = files_of(analyst)["README.md"]
    assert "`findings`: at most three an issuer and 32 in one answer" in readme
    full = _CitingActor(topics=topics)(packet=packet).answer.model_dump(mode="json")
    subset = {**full, "findings": full["findings"][:2]}
    assert len(subset["findings"]) == 2
    invented = json.loads(json.dumps(subset))
    invented["findings"][1]["cite"] = ["S999"]
    answer_file = Path(analyst["answer_file"])
    # An answer names the files it read whole; the list is its own word, kept with the
    # answer as provenance, and the answer is read whatever it names.
    read = [item["name"] for item in analyst["files"]]
    answer_file.write_text(json.dumps({**invented, "read": read[:1]}), encoding="utf-8")
    code, verdict = run(*keyed(analyst))
    assert code == 0 and verdict["status"] == "CORRECT" and "receipt" not in verdict
    bundle = EvidenceReviewBundles(service.review).agent_bundle(str(tmp_path / "analyst"))
    assert bundle is not None
    binding = {
        "role": "alternative_evidence.analyst",
        "task_id": source_id,
        "unit_id": "u01",
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
    code, verdict = run(*keyed(analyst))
    # Accepted, its Task pending: the receipt names the Task's state.
    assert code == 3 and verdict["status"] == "ACCEPTED"
    assert verdict["bundle_reference"] == analyst["bundle_reference"]
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
    # The bundle offers its submit, the directory bound and the answer left.
    offered = cro["next_requests"]["submit"]
    assert (offered["operation"], offered["agent_answer"]) == ("AGENT_ANSWER_SUBMIT", None)
    assert Path(offered["bundle_directory"]) == Path(cro["bundle_directory"])
    command = shlex.split(cro["submit_command"])
    # The installed command, as every printed command starts, then its arguments.
    assert command[0] == "alphalattice" and command[1:] == cro["submit_arguments"]
    dossier = service.review._resolve_review_dossier(service.review._review_selector(None))
    view = render_review_bundle(
        dossier, task_procedure=skill("chief-risk-officer/review-portfolio-evidence")
    )
    assert unrouted(cro) == {k: v.rstrip() for k, v in view.files}
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
        code, verdict = run(*keyed(cro))
        assert code == 0 and verdict["status"] == "CORRECT"
        assert (verdict["rounds_left"], verdict["problems"]) == (rounds_left, [problem])
    code, verdict = run(*keyed(cro))
    assert code == 3 and verdict["status"] == "DONE" and verdict["dropped"] == [problem]
    assert verdict["bundle_reference"] == cro["bundle_reference"]
    assert verdict["receipt"] == {
        "agent_role": "CRO",
        "verdict": "DONE",
        "answer_reference": answer_slot(str(canonical_hash(cro_binding)), 3),
        "task_id": verdict["task_id"],
        "accepted_items": len(submission["risks"]),
        "dropped_items": 1,
    }
    # The lead goes on from this answer: its Task, and the book's Evidence and CRO.
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
    # Each member's acceptance is a row of its own thread, never a relay as the lead; it opens
    # the full accepted answer.
    rows = [
        item
        for item in client.read_external()["items"]
        if item["payload"]["subject"].get("input_channel") == "PRODUCT_REVIEW"
        and item["payload"]["subject"]["message_kind"] == "acceptance"
    ]
    relayed = [
        item
        for item in client.read_external()["items"]
        if item["payload"]["subject"].get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
    ]
    assert len(rows) == 2 and relayed == []
    for row in rows:
        subject = row["payload"]["subject"]
        assert subject["authorship_basis"] == "HOST_ACCEPTED"
        detail = client.read_external(observation_id=row["observation_id"])["accepted_answer"]
        assert detail["status"] == "AVAILABLE", detail
        assert detail["bundle_reference"] == subject["bundle_reference"]
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
