"""Card 33 (engineering consumer): a research team's declared work becomes a readable scene.

Transport evidence over one booted product, driven through the real CLI: the
main PM's assignment, a hook observation, permitted messages, a real
domain-owner refusal on a declared reference, an unrelated Portfolio admission
with an owner-verified result, an explicit producer retry, missing metadata,
truncation, an unknown kind and a reconnect -- all through
`LocalResearchClient.publish_event` and the existing activity feed. The feed's
own pages are then read by the real consumer modules (`live-activity.js` +
`live-team.js`) in Node, and the scene they derive is asserted here. Events are
shaped exactly like the lead's producer; they are a fixture, not native-host
proof, and the boundary assigns EXTERNAL_CLIENT / AGENT_PROPOSAL to every one.
The refusal here (an EXPERIMENT_RUN of an unretained PLAN) and the admission
(a Portfolio replay) are different contexts: this module proves that both are
recorded and read truthfully, not that one corrected the other. The coherent
same-context refusal, corrected submission and owner readback are proved in
`tests/alternative_evidence_desk/test_team_scene_evidence_correction.py`.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from tests.portfolio_strategy_lab.local_web_support import _json, _request

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"
MODULES = (
    Path(__file__).resolve().parents[2]
    / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
)
SESSION = "019a0b1c-parent-session"
SCOPE = sha256(b"parent-session|workspace").hexdigest()


def _cli(live: LocalPortfolioWebSession, *arguments: str) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(live.workspace),
            "--view",
            "full",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert live.web.application.external_token not in result.stdout  # type: ignore[union-attr]
    return result.returncode, json.loads(result.stdout)


class _Producer:
    """Documents shaped like the lead's bridge emits them; the sequence is the producer's."""

    def __init__(self, tmp: Path) -> None:
        self.tmp, self.sequence = tmp, 0

    def _document(self, kind: str, summary: str, subject: dict[str, str]) -> dict[str, Any]:
        self.sequence += 1
        return {
            "event_kind": kind,
            "producer_id": "codex-native",
            "producer_session": SCOPE,
            "producer_sequence": self.sequence,
            "occurred_at": datetime(2026, 9, 14, 12, 0, self.sequence, tzinfo=UTC).isoformat(),
            "summary": summary,
            "subject": {
                **subject,
                "source_time_kind": "BRIDGE_RECEIVED",
                "native_event_id": sha256(f"{kind}|{self.sequence}".encode()).hexdigest(),
            },
            "correlation_ids": [SESSION],
        }

    def hook(self, event: str, agent: str, role: str, turn: str, stop: str = "unknown") -> dict:
        return self._document(
            "NATIVE_SUBAGENT_START_HOOK"
            if event == "SubagentStart"
            else "NATIVE_SUBAGENT_STOP_HOOK",
            f"{role}: {event} hook observed. The child may continue; this is not proof of "
            "agent exit, Task completion or publication.",
            {
                "native_session_id": SESSION,
                "native_turn_id": turn,
                "native_agent_id": agent,
                "role": role,
                "input_channel": "CODEX_HOOK",
                "native_hook_event": event,
                "terminal_state": "NOT_ESTABLISHED",
                "stop_hook_active": stop,
            },
        )

    def message(self, agent: str, role: str, kind: str, text: str, **extra: str) -> dict[str, Any]:
        subject = {
            "native_session_id": SESSION,
            "native_agent_id": agent,
            "message_id": extra.pop("message_id", f"m-{self.sequence + 1}"),
            "message_sha256": sha256(text.encode()).hexdigest(),
            "message_bytes": str(len(text.encode())),
            "input_channel": "ACTOR_DECLARED",
            **extra,
        }
        if role:  # an absent field is declared absence; an empty value is refused by the boundary
            subject["role"] = role
        if kind:
            subject["message_kind"] = kind
        return self._document("NATIVE_COORDINATION_MESSAGE", text, subject)

    def publish(self, live: LocalPortfolioWebSession, document: dict[str, Any]) -> dict[str, Any]:
        path = self.tmp / f"event-{self.sequence}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        _code, body = _cli(live, "event", "declare", "--file", f"{path}")
        return body["data"]


def _words(markup: str) -> str:
    """What a reader reads of a stretch of markup: its words, one space between them."""
    return " ".join(re.sub(r"<[^>]+>", " ", markup).split())


def _consumer(feed: dict[str, Any], tmp: Path, *selection: str) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js development runtime required")
    path = tmp / "pages.json"
    path.write_text(json.dumps(feed), encoding="utf-8")
    result = subprocess.run(
        [
            node,
            str(Path(__file__).with_name("workbench_team_readback.cjs")),
            str(MODULES),
            str(path),
            *selection,
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=20,
        check=True,
    )
    return json.loads(result.stdout)


def test_team_scene_reads_declared_work_and_owner_facts_from_the_real_feed(
    live: LocalPortfolioWebSession, tmp_path: Path
) -> None:
    producer = _Producer(tmp_path)
    tasks_before = len(live.session.task_control_registry.tasks())  # type: ignore[union-attr]
    start = _json(live, "/api/activity")
    cursor = start["cursor"]
    lead, analyst, cro = SESSION, "child-analyst-7", "child-cro-9"

    # The foreground PM assigns; the host names the child; the child asks; the PM answers.
    assignment = producer.publish(
        live,
        producer.message(
            lead,
            "research_lead",
            "assignment",
            "Replay the retained book on the July input and report what survives.",
            recipient_id=analyst,
        ),
    )
    assert assignment["status"] == "APPENDED" and assignment["authority"] == "AGENT_PROPOSAL"
    assert (
        assignment["source_id"] == f"codex-native:{SCOPE}" and assignment["task_verified"] is False
    )
    producer.publish(live, producer.hook("SubagentStart", analyst, "alternative_analyst", "turn-1"))
    producer.publish(
        live,
        producer.message(
            analyst,
            "alternative_analyst",
            "question",
            "Which holdings date should the replay declare?",
            recipient_id=lead,
        ),
    )
    producer.publish(
        live,
        producer.message(
            lead,
            "research_lead",
            "pm_response",
            "The input default; change nothing else.",
            recipient_id=analyst,
        ),
    )

    # The Analyst's long answer names a PLAN this service never previewed: the retained preview
    # is 500 characters and marked, and the referenced original is not this feed's to hold.
    bogus_plan = "b" * 64
    long_answer = " ".join(f"finding-{i}" for i in range(120))
    answer = producer.publish(
        live,
        producer.message(
            analyst,
            "alternative_analyst",
            "answer",
            long_answer,
            reference=bogus_plan,
            message_id="answer-1",
        ),
    )
    assert answer["summary_truncated"] is True
    objection = producer.message(
        cro,
        "independent_cro",
        "objection",
        "That PLAN is not retained here; it cannot be run as stated.",
        reference=bogus_plan,
        recipient_id=lead,
    )
    first = producer.publish(live, objection)
    replay = producer.publish(live, objection)  # the bridge's explicit retry: same sequence
    assert first["status"] == "APPENDED" and replay["status"] == "REUSED_EXACT"
    assert replay["observation_id"] == first["observation_id"]

    # A real domain-owner refusal on that exact reference; then an unrelated Portfolio replay
    # is admitted (generic transport: the two are different contexts, not a correction).
    code, refused = _cli(live, "study", "run", "--plan", bogus_plan)
    assert code == 2 and refused["data"]["failure_code"] == "research_experiment.preview_required"
    spec = tmp_path / "spec.yaml"
    spec.write_text("{}", encoding="utf-8")
    code, sent = _cli(live, "strategy-book", "run", "--file", str(spec))
    assert code == 3 and sent["data"]["disposition"] == "ADMITTED"
    task_id = sent["data"]["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    corrected = producer.publish(
        live,
        producer.message(
            analyst,
            "alternative_analyst",
            "answer",
            "The replay was admitted as a product Task; its result is the owner's.",
            task_id=task_id,
            reference=task_id,
            message_id="answer-2",
        ),
    )
    assert corrected["task_verified"] is True
    producer.publish(
        live,
        producer.message(
            lead,
            "research_lead",
            "pm_response",
            "Accepted as evidence; the objection stays on record.",
            recipient_id=cro,
        ),
    )
    producer.publish(
        live, producer.hook("SubagentStop", analyst, "alternative_analyst", "turn-1", stop="false")
    )
    # Missing metadata and an unknown kind are admitted as declared, not filled in.
    producer.publish(
        live, producer.message("child-unknown", "", "", "no kind, no recipient, no role")
    )
    unknown = producer.publish(
        live,
        producer._document(
            "NATIVE_SOMETHING_NEW",
            "a kind this consumer does not know",
            {"native_session_id": SESSION, "native_agent_id": analyst},
        ),
    )
    assert unknown["status"] == "APPENDED"
    assert len(live.session.task_control_registry.tasks()) == tasks_before + 1  # type: ignore[union-attr]

    # The consumer reads the feed the way the workbench does: one page from the old cursor.
    page = _json(live, f"/api/activity?after={cursor}&limit=100")
    assert page["disposition"] == "CONTINUED" and page["more"] is False
    declared = [i for i in page["items"] if i["schema_kind"] == "ExternalActivityObserved"]
    assert len(declared) == 11 and all(i["authority"] == "AGENT_PROPOSAL" for i in declared)
    assert all(i["source_kind"] == "EXTERNAL_CLIENT" for i in declared)
    assert (
        sum(i["task_id"] == task_id for i in declared) == 1
    )  # only the verified reference joins the Task
    scene = _consumer({"pages": [start, page]}, tmp_path)
    (session,) = scene["sessions"]
    assert session["id"] == SESSION and session["unknownKinds"] == ["NATIVE_SOMETHING_NEW"]
    assert scene["unknown"] == 0
    assert sorted(session["references"]) == sorted([bogus_plan, task_id])
    states = {p["id"]: p["state"] for p in session["participants"]}
    assert states[lead] == "declared only · no host event observed"
    assert states[analyst] == "1 stop hook observed · terminal state not established"
    assert states[cro] == "declared only · no host event observed"
    assert states["child-unknown"] == "declared only · no host event observed"
    kinds = [(e["kind"], e["messageKind"], e["replayOf"]) for e in session["entries"]]
    assert kinds == [
        ("message", "assignment", None),
        ("hook", None, None),
        ("message", "question", None),
        ("message", "pm_response", None),
        ("message", "answer", None),
        ("message", "objection", None),
        ("message", "answer", None),
        ("message", "pm_response", None),
        ("hook", None, None),
        ("message", None, None),
    ]
    answered = session["entries"][4]
    assert answered["truncated"] is True and answered["textLength"] == 500
    assert (
        answered["bytes"] == str(len(long_answer.encode())) and answered["reference"] == bogus_plan
    )
    assert answered["qualified"] == "hash"
    stop = session["entries"][8]
    assert stop["hookEvent"] == "SubagentStop" and stop["terminal"] == "NOT_ESTABLISHED"
    assert stop["stopActive"] == "false" and stop["channel"] == "CODEX_HOOK"
    assert stop["timeKind"] == "BRIDGE_RECEIVED" and stop["authority"] == "AGENT_PROPOSAL"
    assert session["entries"][6]["taskId"] == task_id
    assert session["entries"][6]["reference"] == task_id
    assert session["entries"][6]["qualified"] == "task"
    assert session["entries"][9]["declaredKind"] is None and session["entries"][9]["role"] is None
    # Product observations on the same references, each on its own, in recorded order: the
    # refusal names the declared hash; the Portfolio RUN's observations name the verified Task.
    # The two contexts differ and no correction is stated between them.
    facts = [
        (
            f["operation"] or f["lifecycle"] or f["artifact"],
            f["phase"],
            f["status"],
            f["failure_code"],
            f["named"],
        )
        for f in session["facts"]
    ]
    assert facts == [
        ("EXPERIMENT_RUN", "REQUESTED", None, None, [bogus_plan]),
        (
            "EXPERIMENT_RUN",
            "RETURNED",
            "REFUSED",
            "research_experiment.preview_required",
            [bogus_plan],
        ),
        ("RUN", "RETURNED", "ADMITTED", None, [task_id]),  # the request row names no Task yet
        ("SUCCEEDED", None, None, None, [task_id]),
        ("PortfolioResearchResult", None, None, None, [task_id]),
    ]
    assert all(f["followed"] == [] for f in session["facts"]), "no hop was needed or inferred"
    # The Research Team page: the declared foreground PM on its card, every retained exchange in
    # the thread with its kind told apart (an undeclared kind named as such), the unknown event
    # kind named, and the product evidence -- the refusal, the admission, the owner-verified
    # artifact and the current projection -- in its own block after the thread, in recorded
    # order, none of it among the member statements.
    html = scene["html"]
    for text in (
        "Main PM · declared foreground conversation",
        "That PLAN is not retained here; it cannot be run as stated.",
        "Message kind not declared",
        "NATIVE_SOMETHING_NEW (codex-native)",
        "EXPERIMENT_RUN · product refusal",
        "RUN · product admission",
        "PortfolioResearchResult · owner-verified artifact",
        "Task Control projection, read now",
        f"<activity-open:{task_id}>",
    ):
        assert text in html, text
    # the conversation is one column (C4): the thread ends with its section; since N5 (law 125) the
    # product's observations stand in the thread as its own one-line events, never among the
    # member statements
    thread_at = html.index('id="teamThread"')
    thread = html[thread_at : html.index("</section>", thread_at)]
    # a single product line, or consecutive ones as one counted line (C4 item 1)
    product_line = r'<li class="team-event team-product-event[^"]*"[^>]*>.*?</li>'
    statements = re.sub(product_line, "", thread, flags=re.S)
    assert "Objection" in statements and "product refusal" in thread
    for word in ("product refusal", "product admission", "owner-verified artifact"):
        assert word not in statements, word
    # the evidence page, after the workroom
    evidence = html[html.index('class="panel team-product"') :]
    assert html.index('id="teamThread"') < html.index('class="panel team-product"')
    assert (
        evidence.index("product refusal")
        < evidence.index("product admission")
        < evidence.index("owner-verified artifact")
    )
    assert "correct" not in html.lower()
    assert "running" not in html.lower().replace("not proof of a running", "")
    # Each exchange read in place, reached the way the page reaches it (C4 item 7): its line names
    # the member, the addressee and the kind once, the verification opens under it; the assignment
    # names its recipient; the long answer is a marked preview whose original this feed does not
    # hold, its reference declared and unresolved until asked; the objection is retained with the
    # same declared reference; the stop hook keeps its provenance and its unestablished terminal
    # state.
    reader = {e["observation"]: scene["readers"][e["observation"]] for e in session["entries"]}
    assignment_reader = reader[session["entries"][0]["observation"]]
    assert "Main PM → @Alternative Analyst · Assignment" in _words(assignment_reader)
    assert "Actor-declared text · not host-verified" in assignment_reader
    truncated_reader = reader[answered["observation"]]
    assert "Alternative Analyst · Answer" in _words(truncated_reader)
    for text in (
        "[partial:the first 500 characters]",
        "the full original is not held by this feed",
        "declared · not resolved",
        f"<team-resolve:{bogus_plan}>",
    ):
        assert text in truncated_reader, text
    objection_reader = reader[session["entries"][5]["observation"]]
    assert "CRO → @Main PM · Objection" in _words(objection_reader)
    assert f"<team-resolve:{bogus_plan}>" in objection_reader
    hook_reader = reader[stop["observation"]]
    assert "Alternative Analyst SubagentStop hook observed" in _words(hook_reader)
    for text in (
        'Terminal state: <span class="mono">NOT_ESTABLISHED</span>',
        'Stop hook active: <span class="mono">false',
        "Codex hook input · not host-authenticated",
        "not proof of exit, Task completion or publication",
    ):
        assert text in hook_reader, text
    assert all("product refusal" not in r and "product admission" not in r for r in reader.values())

    # The installed Portfolio replay's verified reference, resolved and verified the way the
    # scene's buttons do it, with the Host's own answers recorded as the owner readbacks: Task
    # Control discovers the Task (a projection); the Portfolio owner's index names its result and
    # REPORT opens that exact result, which names this Task as its origin; the reader offered is
    # the saved `result:` entry History lists. The Portfolio route reads that result as its book
    # (the product, 3de3f325: an installed result, never an authored experiment's reading).
    (artifact,) = [f for f in session["facts"] if f["artifact"] == "PortfolioResearchResult"]
    (artifact_item,) = [i for i in page["items"] if i["observation_id"] == artifact["observation"]]
    result_hash = artifact_item["payload"]["artifact_hash"]
    report = _json(live, f"/api/report?result_hash={result_hash}")
    assert report["result_hash"] == result_hash and report["originating_task_id"] == task_id
    results = _json(live, "/api/results")
    assert [r["task_id"] for r in results["results"] if r["result_hash"] == result_hash] == [
        task_id
    ]
    status, _headers, book_body = _request(live, f"/api/workbench/portfolio?task_id={task_id}")
    book = json.loads(book_body)
    assert status == 200 and book["subject"]["source_kind"] == "INSTALLED_RESULT"
    assert book["subject"]["result_hash"] == result_hash and book["subject"]["receipt_hash"] is None
    history = _json(live, "/api/research-history?history_limit=50")["entries"]
    assert [e["kind"] for e in history if e["entry_id"] == f"result:{result_hash}"] == [
        "INSTALLED_RESULT"
    ]
    asked = _consumer(
        {
            "pages": [start, page],
            "history": history,
            "tasks": [_json(live, f"/api/status?task_id={task_id}")],
            "reads": {
                "/api/results": results,
                f"/api/report?result_hash={result_hash}": report,
                "/api/workbench/portfolio": {"__refused": "research_experiment.task_kind_mismatch"},
            },
            "resolve": [task_id],
            "verify": [task_id],
        },
        tmp_path,
    )
    verified = asked["resolved"][task_id]
    assert verified["level"] == "verified" and verified["owner"] == "Portfolio result"
    assert verified["target"] == {
        "kind": "task",
        "ref": task_id,
        "taskKind": "portfolio_public_development_replay",
    }
    assert verified["open"] == ["history-open", f"result:{result_hash}"]
    assert verified["failure"] is None and verified["limit"] is None
    assert asked["requested"] == ["/api/results", f"/api/report?result_hash={result_hash}"]
    # The answer that declared the Task reads back with the owner's verification and the reader
    # History offers; no experiment entry is invented for an installed Task anywhere on the page.
    verified_reader = asked["readers"][session["entries"][6]["observation"]]
    assert f"[verified:verified by Portfolio result] result {result_hash[:8]}" in verified_reader
    assert f"<history-open:result:{result_hash}>" in verified_reader
    assert f"experiment:{task_id}" not in asked["html"]
    assert all(f"experiment:{task_id}" not in r for r in asked["readers"].values())

    # Reconnect: a fresh reader joins at the tail and continues; the scene it derives from its
    # own pages is the same scene, and a participant selection travels in the route.
    tail = _json(live, "/api/activity?limit=100")
    assert tail["disposition"] == "TAIL"
    producer.publish(
        live,
        producer.message(
            cro,
            "independent_cro",
            "question",
            "Is the July input the sealed one?",
            recipient_id=lead,
        ),
    )
    later = _json(live, f"/api/activity?after={tail['cursor']}")
    assert later["disposition"] == "CONTINUED" and len(later["items"]) == 1
    again = _consumer({"pages": [tail, later]}, tmp_path, SESSION, cro)
    assert again["feed"]["cursor"] == later["cursor"]
    assert [p["id"] for p in again["sessions"][0]["participants"] if p["messages"]] == [
        lead,
        analyst,
        cro,
        "child-unknown",
    ]
    assert "Is the July input the sealed one?" in again["html"]
    assert "Which holdings date" not in again["html"], (
        "the selected participant filters the timeline"
    )
    assert all("Which holdings date" not in r for r in again["readers"].values())
    assert "Objection" in again["html"]
    (reply_to_cro,) = [
        e
        for e in again["sessions"][0]["entries"]
        if e["messageKind"] == "pm_response" and e["recipient"] == cro
    ]
    assert "Main PM → @CRO · PM response" in _words(again["readers"][reply_to_cro["observation"]])
    # Reading and rendering admitted nothing and ran nothing.
    assert len(live.session.task_control_registry.tasks()) == tasks_before + 1  # type: ignore[union-attr]
    assert _json(live, "/api/activity")["head"] == later["head"]
