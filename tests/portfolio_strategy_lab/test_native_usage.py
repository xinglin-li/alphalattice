"""What an agent ran and spent, read from synthetic session files of both hosts (AU, V300)."""

import json
import os
from pathlib import Path

import pytest

from alphalattice.interface.local_application import client as client_module
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    hook_reply,
    pin_differs,
)
from alphalattice.interface.local_application.native_observation_sequence import SEQUENCE_NAME
from alphalattice.interface.local_application.native_usage import (
    HOST_CLAUDE_CODE,
    HOST_CODEX,
    ModelUsage,
    admitted_session_file,
    codex_session_file,
    read_session,
)

SECRET = "SECRET-TRANSCRIPT-TEXT"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_answer_authorship_reads_older_team_pages_and_keeps_conflicting_assignments_unknown(
    ambiguous,
):
    """CONTRACT: author attribution must use the whole retained Team record, not its tail."""
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.interface.local_application.cli_contract import (
        REQUEST_PROVENANCE,
        RequestProvenance,
    )

    # The prepared bundle is now cited by its record hash, not its directory.
    bundle_reference = "a" * 64

    def event(ordinal, kind, **subject):
        return {
            "ordinal": ordinal,
            "payload": {
                "event_kind": kind,
                "subject": {
                    "native_host": "claude-code",
                    "native_session_id": "lead",
                    **subject,
                },
            },
        }

    older = [
        event(
            1,
            "NATIVE_SUBAGENT_START_HOOK",
            native_agent_id="analyst",
            role="alphalattice_evidence_analyst",
            hook_model="synthetic-model",
        ),
        event(
            2,
            "NATIVE_COORDINATION_MESSAGE",
            native_agent_id="lead",
            message_kind="assignment",
            recipient_id="analyst",
            reference=bundle_reference,
        ),
    ]
    newer = [event(3, "NATIVE_SUBAGENT_STOP_HOOK", native_agent_id="analyst")]
    if ambiguous:
        newer.append(
            event(
                4,
                "NATIVE_COORDINATION_MESSAGE",
                native_agent_id="lead",
                message_kind="assignment",
                recipient_id="another-analyst",
                reference=bundle_reference,
            )
        )

    class Pages:
        def __init__(self):
            self.cursors = []

        def read_external(self, query):
            self.cursors.append(query.before)
            return (
                {"items": newer, "oldest": 3, "more": True}
                if query.before is None
                else {"items": older, "oldest": 1, "more": False}
            )

    operations = object.__new__(PortfolioResearchOperations)
    pages = Pages()
    operations.observer = pages
    token = REQUEST_PROVENANCE.set(RequestProvenance(vendor="claude-code", session="lead"))
    try:
        run = operations._agent_run("ANALYST", bundle_reference)
    finally:
        REQUEST_PROVENANCE.reset(token)
    assert pages.cursors == [None, 3]
    assert run is not None
    assert (run.agent_id, run.basis) == (
        (None, "NOT_OBSERVED") if ambiguous else ("analyst", "HOOK")
    )


def _write(path: Path, records: list[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record) for record in records] + ["{not json"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


def _assistant(ident, model, *, output, at, effort="high", text=SECRET, usage=None):
    counts = {
        "input_tokens": 3,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 200,
        "output_tokens": output,
    }
    return {
        "type": "assistant",
        "timestamp": at,
        "effort": effort,
        "message": {
            "id": ident,
            "model": model,
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "usage": counts if usage is None else usage,
        },
    }


def _lead_records():
    return [
        {"type": "summary", "summary": SECRET},
        {"type": "user", "timestamp": "2026-09-29T01:00:00Z", "message": {"content": SECRET}},
        _assistant("msg_1", "claude-opus-5-5", output=5, at="2026-09-29T01:00:05Z"),
        # The same response's later block carries its final counts.
        _assistant("msg_1", "claude-opus-5-5", output=40, at="2026-09-29T01:00:06Z"),
        _assistant("msg_x", "<synthetic>", output=9, at="2026-09-29T01:00:07Z"),
        _assistant("msg_2", "claude-opus-5-5", output=7, at="2026-09-29T01:00:09Z"),
        _assistant("msg_3", "claude-opus-5-5", output=1, at="2026-09-29T01:00:10Z", usage={}),
    ]


def test_a_claude_session_counts_each_response_once_and_keeps_no_text(tmp_path):
    usage = read_session(_write(tmp_path / "lead.jsonl", _lead_records()), host=HOST_CLAUDE_CODE)

    assert [row.response_id for row in usage.responses] == ["msg_1", "msg_2"]
    assert usage.incomplete == 1
    assert (usage.first_at, usage.last_at) == ("2026-09-29T01:00:00Z", "2026-09-29T01:00:10Z")
    (row,) = usage.by_model()
    assert (row.model, row.efforts, row.responses) == ("claude-opus-5-5", ("high",), 2)
    assert (row.input_tokens, row.cache_read_tokens, row.cache_write_tokens) == (6, 2000, 400)
    assert row.output_tokens == 47
    assert (row.first_at, row.last_at) == ("2026-09-29T01:00:06Z", "2026-09-29T01:00:09Z")
    assert SECRET not in repr(usage)


def test_a_codex_rollout_names_the_model_by_turn_and_leaves_the_uncached_input(tmp_path):
    def spent(ident, total, cached, written, output):
        usage = {
            "input_tokens": total,
            "cached_input_tokens": cached,
            "cache_write_input_tokens": written,
            "output_tokens": output,
        }
        record = {"type": "token_usage_record", "timestamp": "2026-09-29T02:00:05Z"}
        return {**record, "payload": {"response_id": ident, "usage": usage}}

    def turn(effort):
        return {"type": "turn_context", "payload": {"model": "gpt-6-luna", "effort": effort}}

    path = _write(
        tmp_path / "rollout.jsonl",
        [
            {"type": "session_meta", "timestamp": "2026-09-29T02:00:00Z", "payload": {}},
            {"type": "response_item", "payload": {"type": "message", "content": SECRET}},
            spent("r0", 10, 0, 0, 1),  # before any turn names a model
            turn("xhigh"),
            spent("r1", 100, 40, 20, 10),
            spent("r1", 100, 40, 20, 10),
            turn("high"),
            spent("r2", 50, 0, 0, 5),
            spent("r3", 10, 20, 0, 1),  # more cached than sent: impossible
        ],
    )
    usage = read_session(path, host=HOST_CODEX)

    assert [row.response_id for row in usage.responses] == ["r1", "r2"]
    assert usage.incomplete == 2
    (row,) = usage.by_model()
    assert (row.model, row.efforts, row.responses) == ("gpt-6-luna", ("high", "xhigh"), 2)
    assert (row.input_tokens, row.cache_read_tokens, row.cache_write_tokens) == (90, 40, 20)
    assert row.output_tokens == 15
    assert SECRET not in repr(usage)


def test_only_the_hosts_own_file_of_that_session_is_admitted(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    own = _write(tmp_path / "claude/projects/p/lead.jsonl", _lead_records())
    elsewhere = _write(tmp_path / "elsewhere/lead.jsonl", _lead_records())
    (tmp_path / "claude/projects/p/dir.jsonl").mkdir()

    def admit(path, stem="lead"):
        return admitted_session_file(path, host=HOST_CLAUDE_CODE, stem=stem)

    assert admit(str(own)) == own.resolve()
    assert admit(str(own), stem="other") is None
    assert admit(str(elsewhere)) is None
    assert admit("lead.jsonl") is None
    assert admit(str(tmp_path / "claude/projects/p/dir.jsonl"), stem="dir") is None
    assert admit(str(tmp_path / "claude/projects/p/../../../elsewhere/lead.jsonl")) is None
    assert admit(None) is None
    link = tmp_path / "claude/projects/p/linked.jsonl"
    try:
        os.symlink(elsewhere, link)
    except OSError:
        pass
    else:
        assert admit(str(link), stem="linked") is None

    older = _write(tmp_path / "codex/sessions/2026/09/28/rollout-a-01a0-thread.jsonl", [])
    newer = _write(tmp_path / "codex/sessions/2026/09/29/rollout-b-01a0-thread.jsonl", [])
    assert codex_session_file("01a0-thread") == newer.resolve() != older.resolve()
    assert codex_session_file("*") is None
    assert codex_session_file("../thread") is None
    assert codex_session_file("missing") is None


@pytest.fixture
def receiver(monkeypatch):
    seen = []

    class AcceptedClient:
        def __init__(self, workspace, *, timeout):
            assert timeout == 2.0

        def publish_event(self, body):
            seen.append(body)
            return {
                "status": "APPENDED",
                "observation_id": f"observation-{len(seen)}",
                "source_id": body["producer_id"] + ":" + body["producer_session"],
                "source_sequence": body["producer_sequence"],
                "authority": "AGENT_PROPOSAL",
                "summary_truncated": False,
            }

    monkeypatch.setattr(client_module, "LocalResearchClient", AcceptedClient)
    return seen


def test_a_subagent_stop_delivers_what_it_and_the_lead_spent_without_text(
    tmp_path, monkeypatch, receiver
):
    config = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    binding = {
        "session_id": "lead-session",
        "workspace": str(tmp_path / "workspace"),
        "roles": ["alphalattice_cro"],
        "host": "claude-code",
    }
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(binding))
    (project / ".claude/agents").mkdir(parents=True)
    (project / ".claude/agents/alphalattice_cro.md").write_text(
        "---\nname: alphalattice_cro\nmodel: claude-sonnet-5-5\neffort: high\n---\nThe card.\n"
    )
    lead = _write(config / "projects/p/lead-session.jsonl", _lead_records())
    # The card pins claude-sonnet-5-5; the host ran an older model.
    child = [_assistant("msg_c", "claude-sonnet-5", output=3, at="2026-09-29T01:00:08Z")]
    agent = _write(config / "projects/p/lead-session/subagents/agent-child.jsonl", child)
    hook = json.dumps(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "lead-session",
            "prompt_id": "prompt-1",
            "agent_id": "child",
            "agent_type": "alphalattice_cro",
            "model": "sonnet",
            "cwd": str(project),
            "stop_hook_active": False,
            "transcript_path": str(lead),
            "agent_transcript_path": str(agent),
            "last_assistant_message": SECRET,
        }
    ).encode()

    assert hook_reply(project, hook) == {}
    stop, spent, led = receiver
    assert [stop["event_kind"], spent["event_kind"], led["event_kind"]] == [
        "NATIVE_SUBAGENT_STOP_HOOK",
        "NATIVE_AGENT_USAGE",
        "NATIVE_AGENT_USAGE",
    ]
    assert {key: stop["subject"][key] for key in ("hook_model", "role_model", "role_effort")} == {
        "hook_model": "sonnet",
        "role_model": "claude-sonnet-5-5",
        "role_effort": "high",
    }
    assert spent["subject"]["native_agent_id"] == "child"
    assert spent["subject"]["role"] == "alphalattice_cro"
    assert (spent["subject"]["model"], spent["subject"]["output_tokens"]) == (
        "claude-sonnet-5",
        "3",
    )
    assert spent["subject"]["pin_differs"] == "model"
    assert "pin_differs" not in led["subject"]  # the lead has no role card
    assert led["subject"]["native_agent_id"] == "lead-session"
    assert led["subject"]["role"] == "research_lead"
    assert led["subject"]["input_channel"] == "CLAUDE_CODE_SESSION_FILE"
    assert (led["subject"]["responses"], led["subject"]["output_tokens"]) == ("2", "47")
    # Room is left for the goal the Host files the event under (16 keys at most).
    assert all(len(item["subject"]) <= 15 for item in receiver)
    sent = json.dumps(receiver).encode() + (project / ".codex" / SEQUENCE_NAME).read_bytes()
    assert SECRET.encode() not in sent
    assert str(config).encode() not in sent and b"transcript" not in sent

    # The same counts read again are the same events; a new lead response is a new reading.
    assert hook_reply(project, hook) == {}
    assert receiver[3:6] == receiver[0:3]
    records = [*_lead_records(), _assistant("msg_4", "claude-opus-5-5", output=2, at="t")]
    _write(lead, records)
    assert hook_reply(project, hook) == {}
    assert receiver[7] == receiver[1]
    assert receiver[8]["subject"]["responses"] == "3"
    assert receiver[8]["producer_sequence"] > receiver[2]["producer_sequence"]


def test_a_binding_that_turns_usage_off_reads_no_session_file(tmp_path, monkeypatch, receiver):
    """requirement (RR, privacy): a session bound with `"usage": "OFF"` reads none of the host's
    session files at a subagent's stop; the stop itself is still observed."""

    config = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    binding = {
        "session_id": "lead-session",
        "workspace": str(tmp_path / "workspace"),
        "roles": ["alphalattice_cro"],
        "host": "claude-code",
        "usage": "OFF",
    }
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(binding))
    (project / ".claude/agents").mkdir(parents=True)
    (project / ".claude/agents/alphalattice_cro.md").write_text(
        "---\nname: alphalattice_cro\nmodel: claude-sonnet-5-5\neffort: high\n---\nThe card.\n"
    )
    lead = _write(config / "projects/p/lead-session.jsonl", _lead_records())
    child = [_assistant("msg_c", "claude-sonnet-5", output=3, at="2026-09-29T01:00:08Z")]
    agent = _write(config / "projects/p/lead-session/subagents/agent-child.jsonl", child)
    hook = json.dumps(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "lead-session",
            "prompt_id": "prompt-1",
            "agent_id": "child",
            "agent_type": "alphalattice_cro",
            "model": "sonnet",
            "cwd": str(project),
            "stop_hook_active": False,
            "transcript_path": str(lead),
            "agent_transcript_path": str(agent),
            "last_assistant_message": SECRET,
        }
    ).encode()

    assert hook_reply(project, hook) == {}
    assert [item["event_kind"] for item in receiver] == ["NATIVE_SUBAGENT_STOP_HOOK"]


@pytest.mark.parametrize(
    ("pins", "model", "efforts", "differs"),
    [
        (
            {"role_model": "claude-sonnet-5-5", "role_effort": "high"},
            "claude-sonnet-5-5",
            ("high",),
            [],
        ),
        ({"role_model": "claude-sonnet-5-5"}, "claude-sonnet-5", ("high",), ["model"]),
        ({"role_model": "sonnet"}, "claude-sonnet-5-5", ("high",), []),
        ({"role_model": "sonnet"}, "claude-opus-5-5", ("high",), ["model"]),
        ({"role_model": "gpt-6-luna", "role_effort": "xhigh"}, "gpt-6-luna", ("high",), ["effort"]),
        ({"role_model": "inherit", "role_effort": "inherit"}, "claude-opus-5-5", ("max",), []),
        ({}, "claude-opus-5-5", ("max",), []),
    ],
)
def test_a_reading_differs_from_its_card_by_an_exact_id_an_alias_or_an_effort(
    pins, model, efforts, differs
):
    reading = ModelUsage(model, efforts, 1, 1, 0, 0, 1, None, None)
    assert pin_differs(pins, reading) == differs


def test_a_lone_lead_is_read_by_its_own_command_under_its_binding(tmp_path, monkeypatch, receiver):
    """regression (V301): a session without subagents recorded no reading, since the bridge read
    the session files only at a subagent's stop; its own goal and answer commands read the lead
    through the bridge, for the bound session only and never where the binding reads nothing."""

    from alphalattice.interface.local_application.native_bridge import lead_readings
    from alphalattice.interface.local_application.native_usage import session_file

    config = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    binding = {
        "session_id": "lead-session",
        "workspace": str(tmp_path / "workspace"),
        "roles": ["alphalattice_cro"],
        "host": "claude-code",
    }
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(binding))
    lead = _write(config / "projects/p/lead-session.jsonl", _lead_records())
    child = [_assistant("msg_c", "claude-sonnet-5-5", output=3, at="2026-09-29T01:00:08Z")]
    agent = _write(config / "projects/p/lead-session/subagents/agent-child.jsonl", child)

    assert session_file(HOST_CLAUDE_CODE, "lead-session") == lead.resolve()
    assert session_file(HOST_CLAUDE_CODE, "lead-session", "child") == agent.resolve()
    assert session_file(HOST_CLAUDE_CODE, "../lead-session") is None
    assert lead_readings(project, {"CLAUDE_CODE_SESSION_ID": "another-session"}) == []
    assert receiver == []
    lead_readings(project, {"CLAUDE_CODE_SESSION_ID": "lead-session"})
    [reading] = receiver
    assert reading["event_kind"] == "NATIVE_AGENT_USAGE"
    subject = reading["subject"]
    assert (subject["native_agent_id"], subject["role"]) == ("lead-session", "research_lead")
    assert (subject["responses"], subject["output_tokens"]) == ("2", "47")
    assert SECRET not in json.dumps(receiver)
    (project / ".codex" / BINDING_NAME).write_text(json.dumps({**binding, "usage": "OFF"}))
    assert lead_readings(project, {"CLAUDE_CODE_SESSION_ID": "lead-session"}) == []


def test_an_answer_is_credited_to_the_specialist_its_bundle_was_assigned_to() -> None:
    """requirement (AU3, V300, V555, V574, LAWS.md ID7): a request names its session, never its
    subagent, so an answer's author is the specialist the session's lead assigned the answered
    bundle to -- an assignment whose reference is the bundle's key, exactly -- checked by that
    specialist's start hook in the session under the bundle's role, its model the hook's or else
    its card's pin. Two Analysts of one role, one stopped and one running, are each credited with
    their own bundle's answer, whatever order the events arrive in; a bundle no assignment
    settles is credited to no one -- never the running Analyst, never the lead, never by time;
    the record keeps it beside the answer, never in its identity."""

    import os

    from alphalattice.interface.local_application.native_bridge import judgment_agent
    from alphalattice.protocols.actor_execution.answers import AgentAnswerRecord, AgentRun
    from alphalattice.protocols.actor_execution.bundles import bundle_slot

    session = "lead-session"
    root = os.path.abspath("analyst-bundles")
    first, second, third = (os.path.join(root, name) for name in ("u01", "u02", "u03"))

    def item(ordinal, kind, **subject):
        base = {"native_session_id": session, "native_host": "claude-code"}
        return {"ordinal": ordinal, "payload": {"event_kind": kind, "subject": {**base, **subject}}}

    def started(ordinal, agent, role="alphalattice_evidence_analyst", **pins):
        return item(ordinal, "NATIVE_SUBAGENT_START_HOOK", native_agent_id=agent, role=role, **pins)

    def assigned(ordinal, recipient, directory, sender=session, reference=None):
        return item(
            ordinal,
            "NATIVE_COORDINATION_MESSAGE",
            native_agent_id=sender,
            role="research_lead",
            message_kind="assignment",
            recipient_id=recipient,
            reference=reference or bundle_slot(directory),
        )

    events = [
        item(
            1,
            "NATIVE_AGENT_USAGE",
            native_agent_id=session,
            role="research_lead",
            model="claude-opus-5-5",
            efforts="max",
            last_at="2026-09-29T01:00:09Z",
        ),
        started(2, "analyst-1", role_model="claude-sonnet-5-5", role_effort="medium"),
        assigned(3, "analyst-1", first),
        started(4, "analyst-2", hook_model="sonnet"),
        assigned(5, "analyst-2", second),
        item(
            6,
            "NATIVE_SUBAGENT_STOP_HOOK",
            native_agent_id="analyst-1",
            role="alphalattice_evidence_analyst",
        ),
    ]

    def credited(directory, found=events, role="ANALYST"):
        value = judgment_agent(
            found,
            host="claude-code",
            session_id=session,
            bundle_role=role,
            bundle_reference=None if directory is None else bundle_slot(directory),
        )
        return AgentRun.model_validate(value)

    # The stopped Analyst's answer, sent while the other runs, is its own; the running one's too.
    one = credited(first)
    assert (one.agent_id, one.role, one.model, one.efforts, one.basis) == (
        "analyst-1",
        "alphalattice_evidence_analyst",
        "claude-sonnet-5-5",
        ("medium",),
        "ROLE_CARD",
    )
    two = credited(second)
    assert (two.agent_id, two.model, two.basis) == ("analyst-2", "sonnet", "HOOK")
    assert credited(first, list(reversed(events))) == one
    unknown = AgentRun(host="claude-code", session_id=session, basis="NOT_OBSERVED")
    # No assignment of the bundle settles no author: never the running Analyst, never the lead.
    assert credited(third) == unknown and credited(None) == unknown
    # The match is exact, by the bundle's key: its directory, however spelled, names it not.
    by_key = [*events, started(7, "analyst-3"), assigned(8, "analyst-3", third)]
    assert credited(third, by_key).agent_id == "analyst-3"
    for spelled in (third, third + os.sep):
        by_directory = [*events, started(7, "analyst-3"), assigned(8, "analyst-3", third, spelled)]
        assert credited(third, by_directory) == unknown
    # Assigned to two, to one never started, by another than the lead, or in another session.
    assert credited(first, [*events, assigned(7, "analyst-2", first)]) == unknown
    assert credited(third, [*events, assigned(7, "ghost", third)]) == unknown
    assert (
        credited(third, [*events, assigned(7, "analyst-2", third, sender="analyst-1")]) == unknown
    )
    elsewhere = assigned(7, "analyst-2", third)
    elsewhere["payload"]["subject"]["native_session_id"] = "another-session"
    assert credited(third, [*events, elsewhere]) == unknown
    # A recipient started under another role answers no Analyst's bundle, and a CRO's it does.
    cro = [
        *events,
        started(7, "cro-1", role="alphalattice_cro_medium"),
        assigned(8, "cro-1", third),
    ]
    assert credited(third, cro) == unknown
    assert credited(third, cro, role="CRO").agent_id == "cro-1"
    running = one
    record = {
        "bundle_key": "a" * 64,
        "number": 1,
        "verdict": "DONE",
        "corrections_used": 0,
        "problems": [],
        "accepted_items": [1],
        "answer_digest": "b" * 64,
    }
    from alphalattice.protocols.actor_execution.answers import answer_slot

    slot = answer_slot("a" * 64, 1)
    kept = AgentAnswerRecord.model_validate(
        {**record, "agent_run": running.model_dump(mode="json"), "record_hash": slot}
    )
    assert kept.agent_run == running and kept.record_hash == slot
    assert AgentAnswerRecord.model_validate({**record, "record_hash": slot}).agent_run is None


def test_a_codex_thread_names_its_spawn_in_its_first_record_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (V568, AU1): a Codex specialist runs in a thread of its own whose rollout
    opens with its `session_meta`, naming the thread that spawned it. The one transcript reader
    reads that first record alone and keeps the parent, role and optional canonical path; it gives
    nothing for a thread with no rollout, a top-level thread, a first record of another thread
    or one past its bound, and never reads a turn."""

    from dataclasses import asdict
    from uuid import uuid4

    from alphalattice.interface.local_application import native_usage
    from alphalattice.interface.local_application.native_usage import codex_thread_spawn

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    day = tmp_path / "sessions" / "2026" / "10" / "03"
    day.mkdir(parents=True)
    lead, child, other = (str(uuid4()) for _ in range(3))

    def rollout(thread: str, *records: dict[str, object]) -> None:
        name = f"rollout-2026-10-03T00-00-00-{thread}.jsonl"
        (day / name).write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")

    spawn = {"parent_thread_id": lead, "agent_role": "alphalattice_cro", "agent_path": "SECRET"}
    source = {"subagent": {"thread_spawn": spawn}}
    meta = {"id": child, "base_instructions": "SECRET", "source": source}
    rollout(child, {"type": "session_meta", "payload": meta}, {"payload": {"text": "SECRET"}})
    rollout(lead, {"type": "session_meta", "payload": {"id": lead, "source": "cli"}})
    found = codex_thread_spawn(child)
    assert found is not None
    assert asdict(found) == {
        "parent_thread_id": lead,
        "agent_role": "alphalattice_cro",
        "agent_path": None,
    }
    assert codex_thread_spawn(lead) is None and codex_thread_spawn(other) is None
    # A first record naming another thread is not this thread's.
    rollout(other, {"type": "session_meta", "payload": {**meta, "id": lead}})
    assert codex_thread_spawn(other) is None
    monkeypatch.setattr(native_usage, "FIRST_RECORD_BYTES", 64)
    assert codex_thread_spawn(child) is None


@pytest.mark.parametrize(
    ("spawn_path", "top_path", "expected"),
    (
        ("/root/evidence_analyst", "/root/evidence_analyst", "/root/evidence_analyst"),
        ("/root/evidence_analyst", "absent", "/root/evidence_analyst"),
        ("/root/evidence_analyst", "/root/another", None),
        (None, "/root/evidence_analyst", None),
        ("/root/../evidence_analyst", "absent", None),
        ("/root/" + "a" * 65, "absent", None),
        ("/root/" + "/".join(["a" * 64] * 3), "absent", "/root/" + "/".join(["a" * 64] * 3)),
        ("/root/" + "/".join(["a" * 64] * 3) + "/b", "absent", None),
    ),
)
def test_codex_spawn_path_is_exact_bounded_header_metadata(
    tmp_path, monkeypatch, spawn_path, top_path, expected
):
    from uuid import uuid4

    from alphalattice.interface.local_application.native_usage import codex_thread_spawn

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    child, parent = str(uuid4()), str(uuid4())
    spawn = {
        "parent_thread_id": parent,
        "agent_role": "alphalattice_evidence_analyst",
        "agent_path": spawn_path,
    }
    meta = {"id": child, "source": {"subagent": {"thread_spawn": spawn}}}
    if top_path != "absent":
        meta["agent_path"] = top_path
    file = tmp_path / "sessions/2026/10/04" / f"rollout-2026-10-04T00-00-00-{child}.jsonl"
    _write(file, [{"type": "session_meta", "payload": meta}])
    found = codex_thread_spawn(child)
    assert found is not None
    assert (found.parent_thread_id, found.agent_role, found.agent_path) == (
        parent,
        "alphalattice_evidence_analyst",
        expected,
    )


def test_codex_spawn_path_reader_never_reads_a_turn(tmp_path, monkeypatch):
    import io
    from uuid import uuid4

    from alphalattice.interface.local_application import native_usage

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    child, parent = str(uuid4()), str(uuid4())
    record = {
        "type": "session_meta",
        "payload": {
            "id": child,
            "agent_path": "/root/evidence_analyst",
            "base_instructions": SECRET,
            "source": {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": parent,
                        "agent_role": "alphalattice_evidence_analyst",
                        "agent_path": "/root/evidence_analyst",
                    }
                }
            },
        },
    }
    file = tmp_path / "sessions/2026/10/04" / f"rollout-2026-10-04T00-00-00-{child}.jsonl"
    _write(file, [record, {"type": "turn_context", "payload": {"text": SECRET}}])
    data = file.read_bytes()
    original = Path.open
    reads = []

    class FirstRecordOnly(io.BytesIO):
        def readline(self, size=-1):
            assert not reads, "Only the first metadata record may be read"
            assert size == native_usage.FIRST_RECORD_BYTES + 1
            reads.append(size)
            return super().readline(size)

        def read(self, *_):
            pytest.fail("The spawn owner must never read the session body")

    def opened(path, *args, **kwargs):
        return FirstRecordOnly(data) if path == file else original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    found = native_usage.codex_thread_spawn(child)
    assert found is not None and found.agent_path == "/root/evidence_analyst"
    assert len(reads) == 1 and SECRET not in repr(found)


@pytest.mark.parametrize(
    "case",
    (
        "path",
        "compact",
        "ancestor",
        "legacy",
        "ambiguous",
        "uuid_ambiguous",
        "wrong_role",
        "other_session",
        "two_paths",
    ),
)
def test_header_proved_path_authorship_keeps_uuid_contract_and_no_file_reads(monkeypatch, case):
    from alphalattice.interface.local_application.native_bridge import judgment_agent
    from alphalattice.protocols.actor_execution.answers import AgentRun

    child = "00000000-0000-0000-0000-000000000001"
    other = "00000000-0000-0000-0000-000000000002"
    parent = "00000000-0000-0000-0000-000000000003"
    path = "/root/evidence_analyst"
    proof = {
        "native_agent_path": path,
        "native_agent_path_basis": "CODEX_SESSION_META",
        "native_agent_path_scope": "BOUND_SESSION_ANCESTRY",
        "native_parent_thread_id": other if case == "ancestor" else parent,
        "native_spawn_role": "alphalattice_evidence_analyst",
    }
    if case == "legacy":
        proof = {}
    if case == "compact":
        proof = {key: proof[key] for key in ("native_agent_path", "native_agent_path_basis")}
    if case == "wrong_role":
        proof["native_spawn_role"] = "alphalattice_cro"

    def row(ordinal, kind, **subject):
        return {
            "ordinal": ordinal,
            "payload": {
                "event_kind": kind,
                "subject": {
                    "native_session_id": parent,
                    "native_host": "codex",
                    **subject,
                },
            },
        }

    first = row(
        1,
        "NATIVE_SUBAGENT_START_HOOK",
        native_agent_id=child,
        role="alphalattice_evidence_analyst",
        hook_model="synthetic-model",
        **proof,
    )
    if case == "other_session":
        first["payload"]["subject"]["native_session_id"] = other
    events = [
        first,
        row(
            2,
            "NATIVE_COORDINATION_MESSAGE",
            native_agent_id=parent,
            message_kind="assignment",
            reference="a" * 64,
            recipient_id=child if case == "uuid_ambiguous" else path,
        ),
    ]
    if case in {"ambiguous", "uuid_ambiguous", "two_paths"}:
        events.append(
            row(
                3,
                "NATIVE_SUBAGENT_START_HOOK",
                native_agent_id=child if case == "two_paths" else other,
                role="alphalattice_evidence_analyst",
                hook_model="synthetic-model",
                **{**proof, "native_agent_path": "/root/other" if case == "two_paths" else path},
            )
        )
    before = json.dumps(events)

    def forbidden_open(*_, **__):
        pytest.fail("Authorship reads sealed hook facts, never a session file")

    monkeypatch.setattr(Path, "open", forbidden_open)
    result = judgment_agent(
        events,
        host="codex",
        session_id=parent,
        bundle_role="ANALYST",
        bundle_reference="a" * 64,
    )
    assert json.dumps(events) == before
    if case in {"path", "compact", "ancestor", "uuid_ambiguous"}:
        assert result["agent_id"] == child and result["basis"] == "HOOK"
        assert set(result) == {
            "host",
            "session_id",
            "agent_id",
            "role",
            "model",
            "efforts",
            "basis",
        }
        assert AgentRun.model_validate(result).agent_id == child
    else:
        assert result == {"host": "codex", "session_id": parent, "basis": "NOT_OBSERVED"}
