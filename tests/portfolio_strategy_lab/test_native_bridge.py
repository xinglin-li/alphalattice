"""Native delivery contracts, including persistence through a fixture Host."""

import importlib.util
import io
import json
import os
import sqlite3
import subprocess
import tomllib
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import pytest

from alphalattice.control.workspace_runtime.lock import WorkspaceLock
from alphalattice.interface.local_application import client as client_module
from alphalattice.interface.local_application.client import LocalResearchClient
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeBridgeError,
    NativeResearchBinding,
    coordination_event,
    deliver,
    hook_reply,
    observation_request,
)
from alphalattice.interface.local_application.native_observation_sequence import (
    LEGACY_SEQUENCE_NAME,
    LOCK_NAME,
    SEQUENCE_NAME,
    NativeSequenceError,
    reserve_sequence,
)
from tests.portfolio_strategy_lab.local_web_support import _json

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("character", ("x", "中", "😀"), ids=("ascii", "cjk", "astral"))
def test_widest_coordination_event_delivers_and_reads_back(live, tmp_path, character):
    project = tmp_path / "project"
    binding = _bind(project, live.workspace)
    event = coordination_event(
        binding,
        agent_id=character * 200,
        role="alphalattice_cro",
        kind="answer",
        message_id=character * 200,
        message=(character * 4000).encode(),
        reference=character * 200,
        recipient_id=character * 200,
        reply_to=character * 200,
    )
    result = deliver(project, binding, event)
    assert result["status"] == "DELIVERED"
    assert result["summary_truncated"] is True
    assert result["authority"] == "AGENT_PROPOSAL"
    (stored,) = _json(live, "/api/activity/external")["items"]
    assert stored["observation_id"] == result["observation_id"]
    assert stored["payload"]["summary"] == character * 500
    for key in ("native_agent_id", "message_id", "reference", "recipient_id", "reply_to"):
        assert stored["payload"]["subject"][key] == character * 200
    assert deliver(project, binding, event) == result  # exact retry after the wide append
    assert len(_json(live, "/api/activity/external")["items"]) == 1


def test_host_refusal_is_named_at_the_bridge_entry(live, tmp_path):
    project = tmp_path / "project"
    binding = _bind(project, live.workspace)
    event = coordination_event(binding, kind="plan", message_id="plan", message=b"A plan.")
    with WorkspaceLock(project / ".codex" / LOCK_NAME):
        document = observation_request(project, binding, event)
    client = LocalResearchClient(live.workspace)
    assert client.publish_event({**document, "summary": "A conflicting claim."})["status"] == (
        "APPENDED"
    )
    refused = deliver(project, binding, event)
    assert refused["status"] == "REFUSED"
    assert refused["reason"] == "observation.source_sequence_collision"
    assert refused["next_action"] == "READ_ACTIVITY_EVENT_CONTRACT"
    assert len(_json(live, "/api/activity/external")["items"]) == 1


def _entry():
    spec = importlib.util.spec_from_file_location(
        "native_entry", ROOT / "src/alphalattice/interface/local_application/native_setup.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def _bind(project, workspace):
    (project / ".codex").mkdir(parents=True)
    value = {"session_id": "parent", "workspace": str(workspace), "roles": ["alphalattice_cro"]}
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(value))
    return NativeResearchBinding.read(project)


@pytest.mark.parametrize("stop_active", (False, True, None))
def test_activity_mapping_retries_are_stable_and_not_research(tmp_path, monkeypatch, stop_active):
    # Accepted31 interface, not an installed product/HTTP/host proof. The shared
    # public client is absent on the lead line until the final merge.
    seen = []
    replies = {"accept": True}

    class AcceptedClient:
        def __init__(self, workspace, *, timeout):
            assert timeout == 2.0

        def publish_event(self, body):
            seen.append(body)
            return {
                "status": "APPENDED" if replies["accept"] else "REFUSED",
                "observation_id": "fixture-observation",
                "source_id": body["producer_id"] + ":" + body["producer_session"],
                "source_sequence": body["producer_sequence"],
                "authority": "AGENT_PROPOSAL",
                "summary_truncated": len(" ".join(body["summary"].split())) > 500,
            }

    monkeypatch.setattr(client_module, "LocalResearchClient", AcceptedClient)
    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    binding = _bind(project, workspace)
    raw = json.dumps(
        {
            "hook_event_name": "SubagentStop",
            "session_id": "parent",
            "turn_id": "turn",
            "agent_id": "child",
            "agent_type": "alphalattice_cro",
            "cwd": str(project),
            "stop_hook_active": stop_active,
            "last_assistant_message": "MUST NOT SEND",
            "transcript_path": "private.jsonl",
        }
    ).encode()
    assert hook_reply(project, raw) == {}
    assert hook_reply(project, raw) == {}
    assert seen[0] == seen[1]  # Includes the exact original sequence and timestamp.
    assert "MUST NOT SEND" not in json.dumps(seen)
    assert "private.jsonl" not in json.dumps(seen)
    assert all(item["event_kind"] == "NATIVE_SUBAGENT_STOP_HOOK" for item in seen)
    assert all(item["subject"]["native_hook_event"] == "SubagentStop" for item in seen)
    assert all(item["subject"]["terminal_state"] == "NOT_ESTABLISHED" for item in seen)
    assert seen[0]["subject"]["stop_hook_active"] == (
        "unknown" if stop_active is None else str(stop_active).lower()
    )
    assert all("task_id" not in item["subject"] and "authority" not in item for item in seen)
    replies["accept"] = False
    reply = hook_reply(project, raw)
    assert set(reply) == {"systemMessage"}
    assert len(seen) == 3  # No automatic retry on product refusal.
    event = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="objection",
        message_id="review-objection",
        message=b"Limited coverage; not a clearance.",
    )
    assert event["source"] == "actor_declared"
    assert event["message_sha256"] == sha256(event["message"].encode()).hexdigest()
    assert deliver(project, binding, event)["status"] == "UNAVAILABLE"
    assert seen[-1]["producer_sequence"] == 1
    assert seen[-1]["producer_session"] == seen[0]["producer_session"]
    metadata = (project / ".codex" / SEQUENCE_NAME).read_bytes()
    assert event["message"].encode() not in metadata
    assert b"private.jsonl" not in metadata
    assert not workspace.exists()  # No product data/Task writes by this adapter.
    # Separate checkouts have separate sequence files, even for one foreground
    # session/workspace. They must not reuse the receiver's same source/sequence.
    other = tmp_path / "other-project"
    other_binding = _bind(other, workspace)
    deliver(other, other_binding, event)
    assert seen[-1]["producer_sequence"] == 0
    assert seen[-1]["producer_session"] != seen[0]["producer_session"]
    assert (project / ".codex" / SEQUENCE_NAME).read_bytes() == metadata
    assert str(other) not in json.dumps(seen[-1])


def test_binding_and_hook_errors_do_not_leak_or_steer(tmp_path):
    assert hook_reply(tmp_path, b"not-json") == {}  # Unbound: no capture or transport.
    binding = _bind(tmp_path, tmp_path / "workspace")
    lead = coordination_event(
        binding,
        agent_id="parent",
        role="research_lead",
        kind="pm_response",
        message_id="response",
        message=b"Keep the review's unresolved objection.",
    )
    assert lead["agent_id"] == lead["session_id"] == "parent"
    assert lead["source"] == "actor_declared"
    with pytest.raises(NativeBridgeError):
        coordination_event(
            binding,
            agent_id="child",
            role="research_lead",
            kind="pm_response",
            message_id="response",
            message=b"Cannot impersonate the foreground lead.",
        )
    with pytest.raises(NativeBridgeError):
        coordination_event(
            binding,
            agent_id="parent",
            role="alphalattice_cro",
            kind="answer",
            message_id="answer",
            message=b"The parent is not a separate reviewer.",
        )
    reply = hook_reply(tmp_path, b'{"secret":"DO NOT ECHO"')
    assert set(reply) == {"systemMessage"}
    assert "DO NOT ECHO" not in json.dumps(reply)
    with pytest.raises(NativeBridgeError):
        NativeResearchBinding.from_document({**asdict(binding), "authority": "HUMAN"})
    with pytest.raises(NativeBridgeError):
        coordination_event(
            binding,
            agent_id="child",
            role="alphalattice_cro",
            kind="objection",
            message_id="objection",
            message=b"x" * 8193,
        )
    with pytest.raises(NativeBridgeError):
        coordination_event(
            binding,
            agent_id="child",
            role="unknown",
            kind="objection",
            message_id="x",
            message=b"text",
        )
    for text, code in (
        (b" " * 600, "message_empty"),
        (b"x" * 501, "long_message_reference_required"),
    ):
        with pytest.raises(NativeBridgeError, match=code):
            coordination_event(
                binding,
                agent_id="child",
                role="alphalattice_cro",
                kind="answer",
                message_id="bounded",
                message=text,
            )
    original = ("A bounded observation. " * 30).encode()
    long = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="answer",
        message_id="long",
        message=original,
        reference="playpen://controlled-assessment/" + "a" * 64,
    )
    assert long["message"].encode() == original
    with pytest.raises(NativeBridgeError, match="assignment_recipient_required"):
        coordination_event(
            binding,
            agent_id="parent",
            role="research_lead",
            kind="assignment",
            message_id="assignment",
            message=b"Read the admitted packet.",
        )
    # Decision notes (GR2): either role, at a plan, a decision, a dead end or a surprise; a
    # reply names what it answers, and an assignment is never a reply.
    for agent_id, role in (("parent", "research_lead"), ("child", "alphalattice_cro")):
        for kind in ("plan", "decision", "dead_end", "surprise"):
            note = coordination_event(
                binding,
                agent_id=agent_id,
                role=role,
                kind=kind,
                message_id=f"{kind}-1",
                message=b"Costs first; the fold sweep waits.",
            )
            assert note["kind"] == kind and note["reply_to"] is None
    answer = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="answer",
        message_id="answer-1",
        message=b"The folds hold.",
        reply_to="assignment-1",
    )
    assert answer["reply_to"] == "assignment-1"
    with pytest.raises(NativeBridgeError, match="assignment_is_not_a_reply"):
        coordination_event(
            binding,
            agent_id="parent",
            role="research_lead",
            kind="assignment",
            message_id="assignment-2",
            message=b"Read the admitted packet.",
            recipient_id="child",
            reply_to="assignment-1",
        )


def test_missing_activity_client_and_wrong_ack_do_not_claim_delivery(tmp_path, monkeypatch):
    binding = _bind(tmp_path, tmp_path / "workspace")
    event = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="answer",
        message_id="a",
        message=b"Scoped assessment returned.",
    )

    class MissingClient:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(client_module, "LocalResearchClient", MissingClient)
    assert (
        deliver(tmp_path, binding, event)["reason"] == "native_bridge.activity_client_unavailable"
    )
    assert not (tmp_path / ".codex" / SEQUENCE_NAME).exists()

    class WrongAck(MissingClient):
        def publish_event(self, document):
            return {
                "status": "APPENDED",
                "source_id": "other",
                "source_sequence": 0,
                "authority": "AGENT_PROPOSAL",
                "observation_id": "o",
            }

    monkeypatch.setattr(client_module, "LocalResearchClient", WrongAck)
    assert deliver(tmp_path, binding, event)["status"] == "UNAVAILABLE"
    with WorkspaceLock(tmp_path / ".codex" / LOCK_NAME):
        assert deliver(tmp_path, binding, event)["reason"] == "native_bridge.observation_busy"


def test_retry_metadata_collision_and_corruption_refuse_without_reset(tmp_path):
    (tmp_path / ".codex").mkdir()

    def reserve(identity="b" * 64, content="c" * 64):
        with WorkspaceLock(tmp_path / ".codex" / LOCK_NAME):
            return reserve_sequence(
                tmp_path, scope="a" * 64, event_id=identity, fingerprint=content
            )

    first = reserve()
    assert reserve() == first
    with pytest.raises(NativeSequenceError, match="event_identity_collision"):
        reserve(content="d" * 64)
    metadata = tmp_path / ".codex" / SEQUENCE_NAME
    original = metadata.read_bytes()
    metadata.write_bytes(b"not-sqlite")
    with pytest.raises(NativeSequenceError, match="metadata_invalid"):
        reserve()
    assert metadata.read_bytes() == b"not-sqlite"
    metadata.write_bytes(original)
    # Even future pruning cannot reset the allocator to a sequence the Host holds.
    with closing(sqlite3.connect(metadata)) as store, store:
        store.execute("DELETE FROM native_observation_sequence")
    assert reserve(identity="e" * 64)[0] == 1
    with closing(sqlite3.connect(metadata)) as store, store:
        store.execute("PRAGMA user_version=2")
    wrong_version = metadata.read_bytes()
    with pytest.raises(NativeSequenceError, match="metadata_invalid"):
        reserve()
    assert metadata.read_bytes() == wrong_version


def test_checkout_delivers_beyond_1024_across_sessions_without_reusing_host_sequences(
    live, tmp_path
):
    project = tmp_path / "project"
    initial_binding = _bind(project, live.workspace)
    sequences, ids = [], []
    first = None
    for index in range(1030):
        binding = NativeResearchBinding(
            f"session-{index // 250}", initial_binding.workspace, initial_binding.roles
        )
        event = coordination_event(
            binding, kind="plan", message_id=f"plan-{index}", message=b"A bounded plan."
        )
        result = deliver(project, binding, event)
        assert result["status"] == "DELIVERED", (index, result)
        sequences.append(result["source_sequence"])
        ids.append(result["observation_id"])
        if first is None:
            first = (binding, event, result)
    assert sequences == list(range(1030))
    assert len(set(ids)) == 1030
    assert deliver(project, *first[:2]) == first[2]  # first session still retries exactly
    stored = []
    route = "/api/activity/external?limit=200"
    while True:
        page = _json(live, route)
        stored.extend(page["items"])
        if not page["more"]:
            break
        route = f"/api/activity/external?limit=200&before={page['oldest']}"
    assert len(stored) == 1030
    assert {item["observation_id"] for item in stored} == set(ids)
    assert sorted(item["payload"]["producer_sequence"] for item in stored) == sequences
    assert len({item["source_id"] for item in stored}) == 5


def test_full_legacy_store_migrates_held_host_sequences_and_first_times(live, tmp_path):
    project = tmp_path / "project"
    binding = _bind(project, live.workspace)
    metadata = project / ".codex" / SEQUENCE_NAME
    legacy = project / ".codex" / LEGACY_SEQUENCE_NAME
    stamp = "2026-10-01T00:00:00+00:00"
    rows = [["a" * 64, sha256(str(i).encode()).hexdigest(), "b" * 64, stamp] for i in range(1024)]
    held = []
    client = LocalResearchClient(live.workspace)
    # Generate the hashes with the bridge, then model the old JSON format at its
    # first and last row positions. The Host already holds those exact documents.
    for index in (0, 1023):
        event = coordination_event(
            binding, kind="plan", message_id=f"old-plan-{index}", message=b"An old plan."
        )
        with WorkspaceLock(project / ".codex" / LOCK_NAME):
            document = observation_request(project, binding, event)
        with closing(sqlite3.connect(metadata)) as store:
            row = store.execute(
                "SELECT scope, event_id, fingerprint, occurred_at FROM native_observation_sequence "
                "WHERE sequence=?",
                (document["producer_sequence"] + 1,),
            ).fetchone()
        rows[index] = [*(value.hex() for value in row[:3]), row[3]]
        document["producer_sequence"] = index
        admitted = client.publish_event(document)
        assert admitted["status"] == "APPENDED"
        held.append((event, document, admitted))
    # Only this synthetic fixture is converted; never delete a person's sequence file.
    metadata.unlink()
    original = json.dumps(rows, separators=(",", ":")).encode()
    legacy.write_bytes(original)
    for event, document, admitted in held:
        result = deliver(project, binding, event)
        assert result["status"] == "DELIVERED"
        assert result["source_sequence"] == document["producer_sequence"]
        assert result["observation_id"] == admitted["observation_id"]
        with WorkspaceLock(project / ".codex" / LOCK_NAME):
            assert observation_request(project, binding, event) == document
    next_event = coordination_event(binding, kind="plan", message_id="new-plan", message=b"New.")
    result = deliver(project, binding, next_event)
    assert result["status"] == "DELIVERED" and result["source_sequence"] == 1024
    assert len(_json(live, "/api/activity/external")["items"]) == 3
    assert legacy.read_bytes() == original


@pytest.mark.parametrize("rows", ("not-json", "[[]]", "[[" + ",".join(['"a"'] * 4) + "]]"))
def test_invalid_legacy_metadata_never_allocates_a_replacement_sequence(tmp_path, rows):
    (tmp_path / ".codex").mkdir()
    legacy = tmp_path / ".codex" / LEGACY_SEQUENCE_NAME
    legacy.write_bytes(rows.encode())
    for _ in range(2):
        with (
            WorkspaceLock(tmp_path / ".codex" / LOCK_NAME),
            pytest.raises(NativeSequenceError, match="metadata_invalid"),
        ):
            reserve_sequence(tmp_path, scope="a" * 64, event_id="b" * 64, fingerprint="c" * 64)
        assert legacy.read_bytes() == rows.encode()


def test_configuration_and_detach_preserve_unrelated_settings(tmp_path, monkeypatch, capsys):
    entry = _entry()
    roles = entry._roles()
    assert set(roles) == {
        "alphalattice_data",
        "alphalattice_factor",
        "alphalattice_alpha",
        "alphalattice_risk",
        "alphalattice_portfolio",
        "alphalattice_evidence_analyst",
        "alphalattice_cro",
    }
    # The Claude host's binding also names the evidence specialists' medium cards (X9).
    assert set(entry._roles("claude-code")) == set(roles) | {
        "alphalattice_evidence_analyst_medium",
        "alphalattice_cro_medium",
    }
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    binding = NativeResearchBinding("parent", tmp_path, tuple(roles))
    # The specialists run on one model at one reasoning effort. Both are the cards' own settings
    # (a change is made in them alone); a card left on another model or effort than the rest
    # fails here.
    models = set()
    efforts = set()
    for role in roles:
        card = tomllib.loads((ROOT / ".codex" / config["agents"][role]["config_file"]).read_text())
        assert card["name"] == role
        assert card["model"] and card["model_reasoning_effort"]
        models.add(card["model"])
        efforts.add(card["model_reasoning_effort"])
        # Every specialist writes: a stage card runs its path and saves the answers it
        # continues from (V384), the two evidence specialists their answer file.
        assert (card["sandbox_mode"], card["approval_policy"]) == ("workspace-write", "never")
        event = coordination_event(
            binding,
            agent_id=f"child/{role}",
            role=role,
            kind="answer",
            message_id=role,
            message=b"Advisory only.",
        )
        assert event["role"] == role and event["source"] == "actor_declared"
    assert len(models) == 1, f"the specialists' cards name different models: {sorted(models)}"
    assert len(efforts) == 1, f"the specialists' cards name different efforts: {sorted(efforts)}"
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    (tmp_path / ".codex").mkdir()
    hooks = tmp_path / ".codex/hooks.json"
    hooks.write_text('{"hooks":{"Stop":[]}}')
    before = hooks.read_bytes()
    with pytest.raises(NativeBridgeError, match="existing_configuration_differs"):
        entry._create_or_match("hooks.json", {"hooks": {}})
    assert hooks.read_bytes() == before
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(entry, "_roles", lambda host="codex", root=None: ["alphalattice_cro"])
    monkeypatch.setattr(
        entry.sys,
        "argv",
        ["native", "bind", "--workspace", str(workspace), "--session-id", "parent"],
    )
    assert entry.main() == 0
    assert entry.main() == 0
    monkeypatch.setattr(entry.sys, "argv", ["native", "unbind", "--session-id", "wrong"])
    assert entry.main() == 2
    assert NativeResearchBinding.read(tmp_path).session_id == "parent"
    monkeypatch.setattr(entry.sys, "argv", ["native", "unbind", "--session-id", "parent"])
    assert entry.main() == 0
    assert NativeResearchBinding.read(tmp_path) is None
    assert hooks.read_bytes() == before
    assert "native_bridge.session_mismatch" in capsys.readouterr().out


@pytest.mark.parametrize("command", ("configure", "doctor"))
@pytest.mark.parametrize("nested", (False, True))
def test_linked_worktree_setup_refuses_before_writing_or_claiming_local_hooks(
    tmp_path, monkeypatch, capsys, command, nested
):
    """Codex 0.156.1 reads linked-worktree hooks from the corresponding main folder."""
    main = tmp_path / "main"
    checkout = tmp_path / "linked"
    checkout.mkdir()
    directory = main / ".git/worktrees/linked"
    directory.mkdir(parents=True)
    (main / ".git/HEAD").write_text("ref: refs/heads/main\n", newline="\n")
    marker = checkout / ".git"
    marker.write_text(f"gitdir: {directory}\n", newline="\n")
    (directory / "gitdir").write_text(str(marker) + "\n", newline="\n")
    (directory / "commondir").write_text("../..\n", newline="\n")
    project = checkout / "nested" if nested else checkout
    (project / ".codex").mkdir(parents=True)
    (project / ".codex/config.toml").write_bytes((ROOT / ".codex/config.toml").read_bytes())
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    entry = _entry()
    monkeypatch.setattr(entry, "INSTALLED", True)
    monkeypatch.setattr(entry.sys, "argv", ["native", "--project", str(project), command])
    assert entry.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "REFUSED"
    assert result["failure_code"] == "native_bridge.configuration_path_invalid"
    assert result["hook_declaration_root"] == str(main / "nested" if nested else main)
    assert result["hook_declarations_effective"] is False
    assert result["host_trust"] == "NOT_CHECKED"
    assert result["foreground_attachment"] == "NOT_PROVED"
    assert result["trust_changed"] is False
    assert "independent ordinary project" in result["detail"] and "/hooks" in result["detail"]
    assert result["next_action"] == "USE_LOCAL_DECLARATIONS_AND_METADATA_IN_THE_EXACT_PROJECT"
    assert "next_commands" not in result
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not (main / ".codex").exists()


@pytest.mark.parametrize("git_checkout", (False, True))
def test_ordinary_project_setup_still_validates_local_declarations_without_trust(
    tmp_path, monkeypatch, capsys, git_checkout
):
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    config = project / ".codex/config.toml"
    config.write_bytes((ROOT / ".codex/config.toml").read_bytes())
    python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    if git_checkout:
        (project / ".git").mkdir()
        (project / ".git/HEAD").write_text("ref: refs/heads/main\n", newline="\n")
    entry = _entry()
    for command, status in (
        ("configure", "LOCAL_DECLARATIONS_VALIDATED"),
        ("doctor", "LOCAL_CONFIGURATION_ONLY"),
    ):
        monkeypatch.setattr(entry.sys, "argv", ["native", "--project", str(project), command])
        assert entry.main() == 0
        result = json.loads(capsys.readouterr().out)
        assert result["status"] == status
        if command == "doctor":
            assert result["host_trust"] == "NOT_CHECKED"
            assert result["foreground_attachment"] == "NOT_PROVED"
    assert not (project / ".codex" / BINDING_NAME).exists()


def test_a_lead_names_only_the_kind_and_a_message_is_named_by_its_content(
    tmp_path, monkeypatch, capsys
):
    """requirement (V420, an outside review at 3fa785fd): a delegation cost the lead its own id,
    its role and a message id beside the relation, call after call. The lead's id and role come
    from its binding and a message's id from its content: an assignment is its kind and
    recipient, its close its kind and the id the assignment answered, and the same message
    sent again is the same message while a changed text is a new one."""

    entry = _entry()
    binding = _bind(tmp_path, tmp_path)
    assigned = coordination_event(
        binding, kind="assignment", message=b"Challenge the folds.", recipient_id="child"
    )
    assert (assigned["agent_id"], assigned["role"]) == ("parent", "research_lead")
    again = coordination_event(
        binding, kind="assignment", message=b"Challenge the folds.", recipient_id="child"
    )
    assert again == assigned
    changed = coordination_event(
        binding, kind="assignment", message=b"Challenge the costs.", recipient_id="child"
    )
    assert changed["message_id"] != assigned["message_id"]
    closed = coordination_event(
        binding, kind="pm_response", message=b"Held.", reply_to=str(assigned["message_id"])
    )
    assert closed["reply_to"] == assigned["message_id"] and closed["role"] == "research_lead"
    # A specialist names itself and its role; a sender with no role is not the lead.
    with pytest.raises(NativeBridgeError, match="message_kind_or_role_invalid"):
        coordination_event(binding, agent_id="child", kind="answer", message=b"Held.")
    # Through the command, the answer names the id a reply sends back.
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    monkeypatch.setattr(
        entry, "deliver", lambda _root, _binding, _event, **_kwargs: {"status": "DELIVERED"}
    )
    monkeypatch.setattr(
        entry.sys, "argv", ["native", "message", "--kind", "assignment", "--to", "child"]
    )
    monkeypatch.setattr(entry.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"Challenge the folds.")))
    assert entry.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "DELIVERED",
        "message_id": assigned["message_id"],
    }


def test_claude_host_configures_binds_and_inspects_without_the_codex_files(
    tmp_path, monkeypatch, capsys
):
    # A Claude-only tree ships the bridge, the Claude cards and settings, and no Codex file.
    project = tmp_path / "project"
    (project / ".claude/agents").mkdir(parents=True)
    for card in (ROOT / ".claude/agents").glob("*.md"):
        (project / ".claude/agents" / card.name).write_bytes(card.read_bytes())
    settings = (ROOT / ".claude/settings.json").read_bytes()
    (project / ".claude/settings.json").write_bytes(settings)
    python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    (project / ".codex").mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    entry = _entry()
    monkeypatch.setattr(entry, "ROOT", project)

    def run(*argv):
        monkeypatch.setattr(entry.sys, "argv", ["native", *argv])
        code = entry.main()
        return code, json.loads(capsys.readouterr().out)

    cards = sorted(path.stem for path in (ROOT / ".claude/agents").glob("alphalattice_*.md"))
    assert {"alphalattice_evidence_analyst_medium", "alphalattice_cro_medium"} <= set(cards)
    assert run("configure", "--host", "claude-code")[1]["status"] == "LOCAL_DECLARATIONS_VALIDATED"
    code, bound = run(
        "bind", "--host", "claude-code", "--session-id", "parent", "--workspace", str(workspace)
    )
    assert (code, bound["status"]) == (0, "BOUND_NOT_ATTACHED")
    assert NativeResearchBinding.read(project).roles == tuple(cards)
    code, report = run("doctor")
    assert (code, report["host"], report["roles"]) == (0, "claude-code", cards)
    assert report["hook_declarations_present"] is True
    assert not (project / ".codex/config.toml").exists()
    # The Codex host still reads its own file, and refuses without it.
    assert run("configure")[1]["status"] == "REFUSED"


def test_hook_bootstrap_failure_cannot_request_subagent_continuation(tmp_path):
    config = tomllib.loads((ROOT / ".codex/config.toml").read_text())
    command = config["hooks"]["SubagentStop"][0]["hooks"][0]["command"]
    fake_uv = tmp_path / ("uv.cmd" if os.name == "nt" else "uv")
    fake_uv.write_text("@exit /b 2\n" if os.name == "nt" else "#!/bin/sh\nexit 2\n")
    fake_uv.chmod(0o700)
    result = subprocess.run(
        command,
        shell=True,
        input=b"{}",
        capture_output=True,
        env={**os.environ, "PATH": str(tmp_path)},
        cwd=tmp_path,
        timeout=5,
    )
    assert result.returncode == 0  # Exit2 would ask the host to continue the child.


def test_claude_code_host_binds_delivers_and_is_kept_apart_from_codex(tmp_path, monkeypatch):
    # A binding names its host; a hook from the other host is out of scope, and the producer
    # id / channel name the host as source information only.
    seen = []

    class AcceptedClient:
        def __init__(self, workspace, *, timeout):
            pass

        def publish_event(self, body):
            seen.append(body)
            return {
                "status": "APPENDED",
                "observation_id": "fixture-observation",
                "source_id": body["producer_id"] + ":" + body["producer_session"],
                "source_sequence": body["producer_sequence"],
                "authority": "AGENT_PROPOSAL",
                "summary_truncated": False,
            }

    monkeypatch.setattr(client_module, "LocalResearchClient", AcceptedClient)
    workspace = tmp_path / "workspace"
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    document = {
        "session_id": "parent",
        "workspace": str(workspace),
        "roles": ["alphalattice_cro"],
        "host": "claude-code",
    }
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(document))
    binding = NativeResearchBinding.read(project)
    assert binding.host == "claude-code"
    with pytest.raises(NativeBridgeError, match="binding_invalid"):
        NativeResearchBinding.from_document({**document, "host": "other-host"})
    assert (
        NativeResearchBinding.from_document({k: v for k, v in document.items() if k != "host"}).host
        == "codex"
    )
    claude_payload = {
        "hook_event_name": "SubagentStart",
        "session_id": "parent",
        "prompt_id": "prompt-1",
        "agent_id": "child",
        "agent_type": "alphalattice_cro",
        "cwd": str(project),
        "permission_mode": "default",
        "transcript_path": "private.jsonl",
    }
    assert hook_reply(project, json.dumps(claude_payload).encode()) == {}
    assert seen[-1]["producer_id"] == "claude-code-native"
    assert seen[-1]["subject"]["input_channel"] == "CLAUDE_CODE_HOOK"
    assert seen[-1]["subject"]["native_host"] == "claude-code"
    assert seen[-1]["subject"]["native_turn_id"] == "prompt-1"
    assert "private.jsonl" not in json.dumps(seen)
    codex_payload = {**claude_payload, "turn_id": "turn"}
    del codex_payload["prompt_id"]
    reply = hook_reply(project, json.dumps(codex_payload).encode())
    assert set(reply) == {"systemMessage"} and "event_scope_invalid" in reply["systemMessage"]
    assert len(seen) == 1
    message = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="answer",
        message_id="answer-1",
        message=b"One assessment, not a route.",
    )
    assert deliver(project, binding, message)["status"] == "DELIVERED"
    assert seen[-1]["producer_id"] == "claude-code-native"
    assert seen[-1]["subject"]["native_host"] == "claude-code"
    assert seen[-1]["subject"]["input_channel"] == "ACTOR_DECLARED"


def test_claude_hook_contract_examples_replay_through_host_and_credit_exact_assignment(
    live, monkeypatch
):
    """CONTRACT (V677): Claude Start/Stop examples reach the real observation owner and
    credit only the bundle assigned to that child. These are the existing synthetic Claude
    contract examples above, extended with Stop; they are not recorded live Claude payloads.
    Live Claude authentication and capture remain separate evidence (V697).
    """
    from alphalattice.interface.local_application.native_bridge import judgment_agent
    from alphalattice.protocols.actor_execution.answers import AgentRun

    project = live.workspace.parent
    (project / ".claude/agents").mkdir(parents=True)
    for relative in ("settings.json", "agents/alphalattice_cro.md"):
        (project / ".claude" / relative).write_bytes((ROOT / ".claude" / relative).read_bytes())
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent")
    client = LocalResearchClient(live.workspace)
    assert client.bind_native_session(project, usage="off")["status"] == "BOUND_NOT_ATTACHED"
    binding = NativeResearchBinding.read(project)
    assert binding is not None and binding.host == "claude-code"

    start = {
        "hook_event_name": "SubagentStart",
        "session_id": "parent",
        "prompt_id": "prompt-1",
        "agent_id": "child",
        "agent_type": "alphalattice_cro",
        "model": "sonnet",
        "cwd": str(project),
        "permission_mode": "default",
        "transcript_path": "PRIVATE-CONTRACT-TRANSCRIPT.jsonl",
    }
    stop = {
        **start,
        "hook_event_name": "SubagentStop",
        "stop_hook_active": False,
        "agent_transcript_path": "PRIVATE-CONTRACT-CHILD.jsonl",
        "last_assistant_message": "PRIVATE-CONTRACT-ANSWER",
    }
    raw_start, raw_stop = (json.dumps(payload).encode() for payload in (start, stop))
    assert hook_reply(project, raw_start) == {}
    assigned = coordination_event(
        binding,
        kind="assignment",
        message=b"Review this contract example's bundle.",
        recipient_id="child",
        reference="a" * 64,
    )
    receipt = deliver(project, binding, assigned, host_owned=True)
    assert receipt["status"] == "DELIVERED"
    assert hook_reply(project, raw_stop) == {}

    rows = _json(live, "/api/activity/external")["items"]
    ordered = sorted(rows, key=lambda row: row["ordinal"])
    assert [row["payload"]["event_kind"] for row in ordered] == [
        "NATIVE_SUBAGENT_START_HOOK",
        "NATIVE_COORDINATION_MESSAGE",
        "NATIVE_SUBAGENT_STOP_HOOK",
    ]
    assert all(row["authority"] == "AGENT_PROPOSAL" for row in rows)
    hooks = [row["payload"] for row in rows if "HOOK" in row["payload"]["event_kind"]]
    for payload in hooks:
        subject = payload["subject"]
        assert len(subject) <= 16  # The existing owner contract remains the admission bound.
        assert subject["native_host"] == "claude-code"
        assert subject["input_channel"] == "CLAUDE_CODE_HOOK"
        assert subject["native_turn_id"] == "prompt-1"
        assert subject["terminal_state"] == "NOT_ESTABLISHED"
    (stopped,) = [
        payload for payload in hooks if payload["event_kind"] == "NATIVE_SUBAGENT_STOP_HOOK"
    ]
    assert stopped["subject"]["stop_hook_active"] == "false"
    assert "PRIVATE-CONTRACT" not in json.dumps(rows)

    def credited(reference):
        return AgentRun.model_validate(
            judgment_agent(
                rows,
                host="claude-code",
                session_id="parent",
                bundle_role="CRO",
                bundle_reference=reference,
            )
        )

    author = credited("a" * 64)
    assert (author.agent_id, author.role, author.model, author.basis) == (
        "child",
        "alphalattice_cro",
        "sonnet",
        "HOOK",
    )
    assert credited("b" * 64) == AgentRun(
        host="claude-code", session_id="parent", basis="NOT_OBSERVED"
    )
    assert hook_reply(project, raw_start) == hook_reply(project, raw_stop) == {}
    assert deliver(project, binding, assigned, host_owned=True) == receipt
    assert _json(live, "/api/activity/external")["items"] == rows


def test_every_bound_the_bridge_applies_is_the_contract_of_what_it_carries(tmp_path):
    """requirement (V574, an outside review at 5f7e7375): an assignment naming a prepared
    bundle by its directory was refused once the directory passed 200 characters, though a
    bundle's directory may be 1,024: the bridge's bounds had drifted from the contracts it
    carries. Each bound it applies is pinned here against that contract, as the contract's own
    validator admits it: a bound tighter than its contract refuses what the contract admits, a
    looser one passes what the Host then refuses under another word. Every length bound in the
    bridge is a named constant, and every named one is a row of this table."""

    import ast

    from pydantic import ValidationError

    from alphalattice.interface.local_application import native_bridge as bridge
    from alphalattice.interface.local_application import native_hook_input as hook_input
    from alphalattice.interface.local_application.activity import (
        SUMMARY_MAXIMUM_CHARACTERS,
        SUMMARY_RETAINED_CHARACTERS,
        ExternalActivityEventDocument,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash
    from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord, bundle_slot

    def admitted(subject=None, correlation_ids=()):
        try:
            ExternalActivityEventDocument.model_validate(
                {
                    "event_kind": "NATIVE_COORDINATION_MESSAGE",
                    "producer_id": "codex-native",
                    "producer_session": "a" * 64,
                    "producer_sequence": 0,
                    "occurred_at": "2026-10-03T00:00:00+00:00",
                    "summary": "A message.",
                    "subject": subject or {},
                    "correlation_ids": list(correlation_ids),
                }
            )
        except ValidationError:
            return False
        return True

    def edge(admits):
        """The longest length a contract admits; one more it refuses."""
        longest = max(n for n in range(1, 513) if admits(n))
        assert not admits(longest + 1)
        return longest

    subject_value = edge(lambda n: admitted({"reference": "r" * n}))
    correlation = edge(lambda n: admitted(correlation_ids=["s" * n]))
    subject_keys = edge(lambda n: admitted({f"k{i}": "v" for i in range(n)}))

    # The widest event of each kind the bridge sends, as it sends it.
    project = tmp_path / "project"
    (project / ".claude" / "agents").mkdir(parents=True)
    (project / ".codex").mkdir()
    (project / ".claude" / "agents" / "alphalattice_cro.md").write_text(
        "---\nname: alphalattice_cro\nmodel: claude-opus-5-5\neffort: medium\n---\nCard.\n"
    )
    binding = NativeResearchBinding(
        session_id="parent", workspace=tmp_path, roles=("alphalattice_cro",), host="claude-code"
    )
    hook = {
        "hook_event_name": "SubagentStart",
        "session_id": "parent",
        "prompt_id": "prompt-1",
        "agent_id": "child",
        "agent_type": "alphalattice_cro",
        "cwd": str(project),
        "model": "claude-opus-5-5",
    }
    message = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="answer",
        message=b"Answered.",
        reference="r" * 64,
        recipient_id="parent",
        reply_to="m-assignment",
    )
    usage = {
        "session_id": "parent",
        "agent_id": "child",
        "role": "alphalattice_cro",
        "host": "claude-code",
        "model": "claude-opus-5-5",
        "efforts": ["medium", "high"],
        "pin_differs": ["model", "effort"],
        "last_at": "2026-10-03T00:00:00Z",
        "responses": 1,
        "input_tokens": 1,
        "cache_read_tokens": 1,
        "cache_write_tokens": 1,
        "output_tokens": 1,
    }
    widest = max(
        len(bridge.observation_request(project, binding, event)["subject"])
        for event in (
            bridge.lifecycle_event(project, binding, json.dumps(hook).encode()),
            message,
            {"source": "native_usage", "usage": usage},
        )
    )
    # The widest binding `native_research.py bind` writes, as it writes it (`_create_or_match`).
    document = {
        "session_id": "s" * bridge.CORRELATION_CHARACTERS,
        "workspace": "D:\\" + "\u00e9" * (bridge.TEXT_CHARACTERS - 3),
        "roles": [f"r{index:02d}" + "x" * 61 for index in range(bridge.BINDING_ROLES)],
        "host": "claude-code",
        "usage": "OFF",
    }
    NativeResearchBinding.from_document(document)
    written = len((json.dumps(document, indent=2) + "\n").encode())
    directory = AgentBundleRecord.model_fields["bundle_directory"].metadata
    longest_directory = max(getattr(item, "max_length", 0) or 0 for item in directory)
    entry = _entry()
    shipped = {host: entry._roles(host) for host in ("codex", "claude-code")}
    # name: (the bridge's bound, how it stands to its contract, the contract's, what it bounds)
    table = {
        "SUBJECT_VALUE_CHARACTERS": (
            bridge.SUBJECT_VALUE_CHARACTERS,
            "==",
            subject_value,
            "each id, role, locator and reference an event carries; a message's at entry",
        ),
        "CORRELATION_CHARACTERS": (
            bridge.CORRELATION_CHARACTERS,
            "==",
            correlation,
            "the bound session's id, every event's correlation id, at binding",
        ),
        "MESSAGE_CHARACTERS": (
            bridge.MESSAGE_CHARACTERS,
            "==",
            SUMMARY_MAXIMUM_CHARACTERS,
            "a message, the event's summary",
        ),
        "MESSAGE_KEPT_CHARACTERS": (
            bridge.MESSAGE_KEPT_CHARACTERS,
            "==",
            SUMMARY_RETAINED_CHARACTERS,
            "what the Host keeps of a message; a longer one cites a reference",
        ),
        "MAX_MESSAGE_BYTES": (
            bridge.MAX_MESSAGE_BYTES,
            ">=",
            4 * SUMMARY_MAXIMUM_CHARACTERS,
            "the message read: the longest summary in UTF-8, four bytes a character at most",
        ),
        "PIN_CHARACTERS": (
            bridge.PIN_CHARACTERS,
            "<=",
            subject_value,
            "a card's model or effort pin, carried in a start hook's subject",
        ),
        "OBSERVATION_ID_CHARACTERS": (
            bridge.OBSERVATION_ID_CHARACTERS,
            ">=",
            len(canonical_hash({"observation": 1})),
            "an acknowledgment's observation id: the Host's are hashes",
        ),
        "FIELD_CHARACTERS": (
            hook_input.FIELD_CHARACTERS,
            ">=",
            subject_value,
            "a hook's field, read whole; a carried one meets the subject bound at the event",
        ),
        "MAX_HOOK_INPUT_BYTES": (
            hook_input.MAX_HOOK_INPUT_BYTES,
            ">=",
            hook_input.FIELD_CHARACTERS * 4 * 8,
            "a hook's whole input: its eight read fields at their widest, beside what it drops",
        ),
        "BINDING_BYTES": (
            bridge.BINDING_BYTES,
            ">=",
            written,
            "the binding file: the widest binding `bind` writes",
        ),
        "BINDING_ROLES": (
            bridge.BINDING_ROLES,
            ">=",
            max(len(roles) for roles in shipped.values()),
            "a binding's roles: the cards a host ships",
        ),
        "TEXT_CHARACTERS": (
            bridge.TEXT_CHARACTERS,
            ">=",
            260,
            "a binding's host, reading and workspace, never carried: a Windows path",
        ),
        "SPAWN_HOPS": (
            bridge.SPAWN_HOPS,
            ">=",
            2,
            "the rollouts a Codex specialist's spawn chain reads, a first record each: its "
            "lead's own and a specialist's specialist's (V568)",
        ),
    }
    relations = {"==": int.__eq__, "<=": int.__le__, ">=": int.__ge__}
    for name, (bound, relation, contract, _what) in table.items():
        assert relations[relation](bound, contract), (name, bound, relation, contract)
    assert widest <= subject_keys, widest
    # A bundle is named by its key, which any message carries; its directory may not fit one.
    assert len(bundle_slot("D:\\" + "d" * 1020)) <= subject_value < longest_directory
    for roles in shipped.values():
        assert roles and all(bridge._ROLE_NAME.match(role) for role in roles), roles

    # Every length bound is named, and every named one is a row.
    for module in (bridge, hook_input):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        named = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and type(getattr(module, target.id)) is int
        }
        assert named <= set(table), named - set(table)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left, *node.comparators]
            if any(
                isinstance(item, ast.Call) and getattr(item.func, "id", "") == "len"
                for item in operands
            ):
                literals = [
                    item.value
                    for item in operands
                    if isinstance(item, ast.Constant) and item.value not in (0, 1)
                ]
                assert not literals, (module.__name__, node.lineno, literals)

    # At its edges: a message's ids and references are refused by name where they enter.
    fields = {
        "agent_id": "child",
        "message_id": "m-1",
        "reference": "r" * 64,
        "recipient_id": "parent",
        "reply_to": "m-0",
    }
    for name in fields:
        at_bound = {**fields, name: "x" * subject_value}
        sent = coordination_event(
            binding, role="alphalattice_cro", kind="answer", message=b"Read.", **at_bound
        )
        assert sent[name] == "x" * subject_value
        past = {**fields, name: "x" * (subject_value + 1)}
        with pytest.raises(NativeBridgeError, match=f"message_field_invalid:{name}$"):
            coordination_event(
                binding, role="alphalattice_cro", kind="answer", message=b"Read.", **past
            )
    # The longest message of the widest characters is read whole; one character more is not,
    # by the character bound, the bytes being within the read's.
    widest_text = "\U0001f600" * bridge.MESSAGE_CHARACTERS
    read = coordination_event(
        binding,
        agent_id="child",
        role="alphalattice_cro",
        kind="answer",
        message=widest_text.encode(),
        reference="r" * 64,
    )
    assert read["message"] == widest_text
    longer = (widest_text[:-1] + "xx").encode()
    assert len(longer) <= bridge.MAX_MESSAGE_BYTES
    with pytest.raises(NativeBridgeError, match="message_character_limit"):
        coordination_event(
            binding,
            agent_id="child",
            role="alphalattice_cro",
            kind="answer",
            message=longer,
            reference="r" * 64,
        )
    # A binding the Host could never correlate, or naming no card, is refused when bound.
    for refused in (
        {**document, "session_id": "s" * (correlation + 1)},
        {**document, "roles": ["Alphalattice-CRO"]},
    ):
        with pytest.raises(NativeBridgeError, match="binding_invalid"):
            NativeResearchBinding.from_document(refused)
    # A hook's carried value past the subject bound is refused by the bridge, before the Host.
    long_child = {**hook, "agent_id": "c" * (subject_value + 1)}
    event = bridge.lifecycle_event(project, binding, json.dumps(long_child).encode())
    with pytest.raises(NativeBridgeError, match="activity_subject_invalid"):
        bridge.observation_request(project, binding, event)


def test_a_binding_is_found_up_from_any_folder_and_serves_its_session_and_specialists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (V568): one bind, the one `native_setup bind` and `session bind` share, writes
    the binding in the project holding its host's declarations, the nearest up from where it
    runs, making a Claude Code project's `.codex` folder; a project bound otherwise refuses it.
    The binding is found from any folder within the project, as git finds its .git, and serves
    the bound session, a Claude Code subagent that carries its id, and a Codex specialist whose
    own rollout names a spawn chain reaching the bound session within `SPAWN_HOPS`; no other
    session, of either host, whatever its id."""

    import shutil
    from uuid import uuid4

    from alphalattice.interface.local_application import native_bridge as bridge
    from alphalattice.interface.local_application.native_setup import (
        bind_session,
        session_project,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    claude = tmp_path / "claude-project"
    (claude / ".claude" / "agents").mkdir(parents=True)
    shutil.copyfile(ROOT / ".claude/settings.json", claude / ".claude/settings.json")
    shutil.copyfile(ROOT / ".claude/agents/alphalattice_cro.md", claude / ".claude/agents/c.md")
    (claude / ".claude/agents/c.md").rename(claude / ".claude/agents/alphalattice_cro.md")
    below = claude / "deep" / "er"
    below.mkdir(parents=True)
    lead = str(uuid4())
    assert session_project(below, "claude-code") == claude
    with pytest.raises(NativeBridgeError, match="hook_declaration_missing"):
        session_project(below, "codex")
    written = bind_session(claude, host="claude-code", session_id=lead, workspace=workspace)
    assert written["workspace"] == str(workspace.resolve()) and (claude / ".codex").is_dir()
    assert bind_session(claude, host="claude-code", session_id=lead, workspace=workspace)
    with pytest.raises(NativeBridgeError, match="existing_configuration_differs"):
        bind_session(claude, host="claude-code", session_id=str(uuid4()), workspace=workspace)
    found = bridge.NativeResearchBinding.find(below)
    assert found is not None and found[0] == claude
    binding = found[1]
    assert binding.serves(("claude-code", lead)) and not binding.serves(("codex", lead))
    assert not binding.serves(("claude-code", str(uuid4())))
    assert bridge.NativeResearchBinding.find(tmp_path) is None

    # A Codex lead's specialists, by their own rollouts' spawn chains.
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    day = tmp_path / "codex-home" / "sessions" / "2026" / "10" / "03"
    day.mkdir(parents=True)

    def spawned(parent: str) -> str:
        thread = str(uuid4())
        meta = {
            "id": thread,
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent}}},
        }
        record = json.dumps({"type": "session_meta", "payload": meta})
        (day / f"rollout-2026-10-03T00-00-00-{thread}.jsonl").write_text(record + "\n", "utf-8")
        return thread

    codex = bridge.NativeResearchBinding(
        session_id=lead, workspace=workspace, roles=("alphalattice_cro",), host="codex"
    )
    chain = [lead]
    for _hop in range(bridge.SPAWN_HOPS + 1):
        chain.append(spawned(chain[-1]))
    served = [codex.serves(("codex", thread)) for thread in chain]
    assert served == [True] * (bridge.SPAWN_HOPS + 1) + [False]
    assert not codex.serves(("codex", spawned(str(uuid4()))))
    assert not codex.serves(("claude-code", chain[1]))


@pytest.mark.parametrize("kind", ("SubagentStart", "SubagentStop"))
@pytest.mark.parametrize("relation", ("direct", "ancestor", "wrong_parent", "wrong_role"))
def test_new_hook_path_uses_exact_header_role_and_admitted_ancestry_with_usage_off(
    tmp_path, monkeypatch, kind, relation
):
    from uuid import uuid4

    from pydantic import ValidationError

    from alphalattice.interface.local_application import native_bridge as bridge
    from alphalattice.interface.local_application.activity import ExternalActivityEventDocument

    role, path = "alphalattice_evidence_analyst", "/root/evidence_analyst"
    project = tmp_path / "project"
    cards = project / ".codex/agents"
    cards.mkdir(parents=True)
    (cards / f"{role}.toml").write_text(
        'model = "synthetic-model"\nmodel_reasoning_effort = "xhigh"\n', encoding="utf-8"
    )
    lead, child, intermediate = (str(uuid4()) for _ in range(3))
    binding = bridge.NativeResearchBinding(
        session_id=lead,
        workspace=tmp_path / "workspace",
        roles=(role,),
        host="codex",
        usage="OFF",
    )
    (project / ".codex" / BINDING_NAME).write_text(
        json.dumps(
            {
                "session_id": lead,
                "workspace": str(binding.workspace),
                "roles": [role],
                "host": "codex",
                "usage": "OFF",
            }
        ),
        encoding="utf-8",
    )
    home = tmp_path / "codex-home"
    day = home / "sessions/2026/10/04"
    day.mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(home))

    def header(thread, parent, agent_role, agent_path):
        meta = {
            "id": thread,
            "agent_path": agent_path,
            "source": {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": parent,
                        "agent_role": agent_role,
                        "agent_path": agent_path,
                    }
                }
            },
        }
        (day / f"rollout-2026-10-04T00-00-00-{thread}.jsonl").write_text(
            json.dumps({"type": "session_meta", "payload": meta})
            + "\n"
            + json.dumps({"type": "turn_context", "payload": {"text": "PRIVATE TURN"}})
            + "\n",
            encoding="utf-8",
        )

    parent = intermediate if relation in {"ancestor", "wrong_parent"} else lead
    if relation == "ancestor":
        header(intermediate, lead, "alphalattice_cro", "/root/review_lead")
    header(child, parent, "alphalattice_cro" if relation == "wrong_role" else role, path)

    def no_usage(*_, **__):
        pytest.fail("Usage OFF must never read responses or the session body")

    monkeypatch.setattr(bridge, "read_session", no_usage)
    delivered = []

    def capture(_project, _binding, event):
        delivered.append(event)
        return {"status": "DELIVERED"}

    monkeypatch.setattr(bridge, "deliver", capture)
    hook = json.dumps(
        {
            "hook_event_name": kind,
            "session_id": lead,
            "turn_id": "turn",
            "agent_id": child,
            "agent_type": role,
            "model": "synthetic-model",
            "cwd": str(project),
            "agent_path": "/root/forged",
            "last_assistant_message": "PRIVATE TURN",
        }
    ).encode()
    assert bridge.handle_hook(project, hook) == {"status": "DELIVERED"}
    assert len(delivered) == 1
    with WorkspaceLock(project / ".codex" / LOCK_NAME):
        request = bridge.observation_request(project, binding, delivered[0])
    subject = request["subject"]
    # Validate the complete delivered document, including the event id added after
    # lifecycle projection. The alias is only two fields: ancestry and exact role
    # have already been proved by the producer, and the original 16-field bound stays.
    ExternalActivityEventDocument.model_validate(request)
    assert subject["hook_model"] == subject["role_model"] == "synthetic-model"
    assert subject["role_effort"] == "xhigh"
    assert len(subject["native_event_id"]) == 64
    assert len(subject) <= 16
    assert {"native_agent_path_scope", "native_parent_thread_id", "native_spawn_role"}.isdisjoint(
        subject
    )
    if relation in {"direct", "ancestor"}:
        assert subject["native_agent_path"] == path
        assert subject["native_agent_path_basis"] == "CODEX_SESSION_META"
        assert len(subject) == 16
        with pytest.raises(ValidationError, match="event_subject_too_large"):
            ExternalActivityEventDocument.model_validate(
                {**request, "subject": {**subject, "unexpected_field": "extra"}}
            )
    else:
        assert "native_agent_path" not in subject
    assert subject["native_agent_id"] == child
    assert "PRIVATE TURN" not in json.dumps(request) and "/root/forged" not in json.dumps(request)
