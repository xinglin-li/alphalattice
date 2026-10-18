"""What an agent ran and spent, read from synthetic session files of both hosts (AU, V300)."""

import json
import os
from dataclasses import asdict
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import pytest

from alphalattice.interface.local_application import client as client_module
from alphalattice.interface.local_application import native_bridge as bridge_module
from alphalattice.interface.local_application.activity import ExternalActivityEventDocument
from alphalattice.interface.local_application.failure_codes import owner_failure_code
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    NativeResearchBinding,
    pin_differs,
    read_session_usage,
)
from alphalattice.interface.local_application.native_setup import declare_project
from alphalattice.interface.local_application.native_usage import (
    HOST_CLAUDE_CODE,
    HOST_CODEX,
    ModelUsage,
    NativeUsageReadLimitError,
    admitted_session_file,
    claude_children,
    claude_thread_spawn,
    codex_session_file,
    member_claims,
    read_session,
    session_file,
)
from alphalattice.protocols.actor_execution.bundles import bundle_slot
from devtools.evaluation.native_usage import STEPS, walk

SECRET = "SECRET-TRANSCRIPT-TEXT"


@pytest.mark.parametrize(
    "problem",
    [
        "exact",
        "session",
        "agent",
        "sidechain",
        "role",
        "nested",
        "parent_agent",
        "missing_meta",
        "invalid_meta",
        "large_meta",
        "large_header",
        "duplicate_lead",
    ],
)
def test_claude_child_metadata_names_one_exact_direct_parent_and_explicit_role(
    tmp_path, monkeypatch, problem
):
    """Only bounded sidecar/header metadata binds an exact child of the lead to its counts."""
    native = tmp_path / "synthetic-claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(native))
    lead = _write(native / "projects/fixture-project/lead-session.jsonl", _lead_records())
    child = _write(
        lead.parent / "lead-session/subagents/agent-exact-child.jsonl",
        [
            {
                "type": "user",
                "sessionId": "lead-session",
                "agentId": "exact-child",
                "isSidechain": True,
                "message": {"content": SECRET},
            },
            _assistant("response", "synthetic-model", output=3, at="2026-10-06T00:00:00Z"),
        ],
    )
    header = json.loads(child.read_text().splitlines()[0])
    meta = {
        "agentType": "alphalattice_risk",
        "spawnDepth": 1,
        "toolUseId": "synthetic-tool-use",
        "description": SECRET,
    }
    if problem in {"session", "agent", "sidechain"}:
        field, value = {
            "session": ("sessionId", "another-parent"),
            "agent": ("agentId", "another-child"),
            "sidechain": ("isSidechain", False),
        }[problem]
        header[field] = value
    elif problem == "role":
        del meta["agentType"]
    elif problem == "nested":
        meta["spawnDepth"] = 2
    elif problem == "parent_agent":
        meta["parentAgentId"] = "another-child"
    elif problem == "large_meta":
        meta["description"] = SECRET * 2000
    elif problem == "large_header":
        header["message"] = {"content": SECRET * 100_000}
    elif problem == "duplicate_lead":
        _write(native / "projects/other-project/lead-session.jsonl", _lead_records())
    child.write_text(json.dumps(header) + "\n", encoding="utf-8")
    if problem != "missing_meta":
        child.with_suffix(".meta.json").write_text(
            "{" if problem == "invalid_meta" else json.dumps(meta), encoding="utf-8"
        )
    result = claude_thread_spawn("lead-session", "exact-child")
    if problem in {"exact", "role"}:
        assert result is not None
        assert (result.parent_thread_id, result.agent_role) == (
            "lead-session",
            "alphalattice_risk" if problem == "exact" else None,
        )
        assert SECRET not in repr(result)
    else:
        assert result is None
    assert claude_thread_spawn("lead-session", "/root/alias") is None
    assert claude_thread_spawn("lead-session", "lead-session") is None


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


KEY, THREAD = "0123456789abcdef0123456789abcdef", "afc0f906-28c2-4c6d-9d0b-fd249bdf88d4"
CLAIMS = [
    (f"x committee submit --update {THREAD} --role ALPHA --key={KEY}", ("ALPHA", True)),
    (f"x committee wait --update '{THREAD}' --role RISK --key \"{KEY}\"", ("RISK", False)),
    (f'x bundle submit --dir "out/a 1" --key {KEY}', "out/a 1"),
    (f"x bundle submit --dir /d/w/out --key {KEY}", "D:/w/out" if os.name == "nt" else "/d/w/out"),
    (f"x --key {KEY}; x committee submit --update {THREAD} --role ALPHA --key {KEY[::-1]}", None),
    (f"x committee submit --update {THREAD} --role ALPHA --key $key", None),
]


@pytest.mark.parametrize(("command", "claimed"), CLAIMS)
def test_a_member_claim_keeps_only_its_keys_hash_its_thread_and_what_it_spoke(
    tmp_path, command, claimed
):
    """One key per command, quoted or `=`, a relative or MSYS bundle path; never the key."""
    claims: list[dict] = []
    use = {"type": "tool_use", "id": "u1", "input": {"command": command}}
    member_claims({"cwd": str(tmp_path), "message": {"content": [use]}}, claims)
    said = '{"status": "COMMITTEE_MESSAGE_ACCEPTED", "message_id": "M3"}'
    result = {"type": "tool_result", "tool_use_id": "u1", "content": said}
    member_claims({"message": {"content": [result]}}, claims)
    if claimed is None:
        assert claims == []
        return
    (claim,) = claims
    assert claim["key"] == sha256(KEY.encode()).hexdigest() and KEY not in json.dumps(claims)
    if isinstance(claimed, str):
        folder = Path(claimed) if Path(claimed).is_absolute() else tmp_path / claimed
        assert (claim["reference"], claim["spoke"]) == (bundle_slot(str(folder)), ["M3"])
    else:
        assert (claim["thread"], claim["role"], claim["submit"]) == (THREAD, *claimed)
        assert claim["spoke"] == (["M3"] if claimed[1] else [])


def test_a_claude_session_counts_each_response_once_and_keeps_no_text(tmp_path):
    records = _lead_records()
    records[2]["message"]["id"] = records[3]["message"]["id"] = SECRET * 10_000
    usage = read_session(_write(tmp_path / "lead.jsonl", records), host=HOST_CLAUDE_CODE)

    assert [row.response_id for row in usage.responses] == [
        sha256(i.encode()).hexdigest() for i in (SECRET * 10_000, "msg_2")
    ]
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
        return {
            **record,
            "payload": {"response_id": SECRET * 10_000 if ident == "r1" else ident, "usage": usage},
        }

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

    assert [row.response_id for row in usage.responses] == [
        sha256(i.encode()).hexdigest() for i in (SECRET * 10_000, "r2")
    ]
    assert usage.incomplete == 2
    (row,) = usage.by_model()
    assert (row.model, row.efforts, row.responses) == ("gpt-6-luna", ("high", "xhigh"), 2)
    assert (row.input_tokens, row.cache_read_tokens, row.cache_write_tokens) == (90, 40, 20)
    assert row.output_tokens == 15
    assert SECRET not in repr(usage)


@pytest.mark.parametrize("host", [HOST_CODEX, HOST_CLAUDE_CODE])
def test_malformed_selected_usage_is_incomplete_instead_of_a_complete_prefix(tmp_path, host):
    """An unfinished host usage JSON record cannot turn earlier counts into a full reading."""
    if host == HOST_CLAUDE_CODE:
        records = [_assistant("first", "synthetic-model", output=3, at="2026-10-06T00:00:00Z")]
        malformed = b'{"type":"assistant","message":{"usage":'
    else:
        records = [
            {"type": "turn_context", "payload": {"model": "synthetic-model"}},
            {
                "type": "token_usage_record",
                "payload": {
                    "response_id": "first",
                    "usage": {
                        "input_tokens": 3,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 3,
                    },
                },
            },
        ]
        malformed = b'{"type":"token_usage_record","payload":'
    path = _write(tmp_path / "synthetic-session.jsonl", records)
    with path.open("ab") as stream:
        stream.write(malformed)
    reading = read_session(path, host=host)
    assert len(reading.responses) == 1 and reading.incomplete == 1
    assert reading.responses[0].output_tokens == 3


@pytest.mark.parametrize("host", [HOST_CODEX, HOST_CLAUDE_CODE])
def test_bounded_usage_admits_a_large_valid_native_line_and_preserves_legacy_totals(tmp_path, host):
    """BEHAVIOUR: a valid private line larger than 1 MiB does not erase complete usage."""
    ignored = {
        "type": "response_item" if host == HOST_CODEX else "user",
        "payload": {"content": SECRET * 60000},
    }
    records = (
        [
            {"type": "turn_context", "payload": {"model": "gpt-6-luna", "effort": "high"}},
            {
                "type": "token_usage_record",
                "timestamp": "2026-10-06T00:00:00Z",
                "payload": {
                    "response_id": "complete-response",
                    "usage": {
                        "input_tokens": 20,
                        "cached_input_tokens": 5,
                        "cache_write_input_tokens": 0,
                        "output_tokens": 3,
                    },
                },
            },
        ]
        if host == HOST_CODEX
        else [
            _assistant(
                "complete-response", "claude-sonnet-5-5", output=3, at="2026-10-06T00:00:00Z"
            )
        ]
    )
    path = _write(tmp_path / "session.jsonl", [ignored, *records])
    maximum_line = max(len(line) for line in path.read_bytes().splitlines(keepends=True))
    assert 1024 * 1024 < maximum_line < 8 * 1024 * 1024
    bounded = read_session(
        path, host=host, max_bytes=path.stat().st_size, max_line_bytes=maximum_line
    )
    assert bounded == read_session(path, host=host)
    assert bounded.incomplete == 0 and len(bounded.responses) == 1
    assert bounded.responses[0].response_id == sha256(b"complete-response").hexdigest()
    assert SECRET not in repr(bounded)


@pytest.mark.parametrize("limit", ["max_bytes", "max_line_bytes"])
def test_an_incomplete_bounded_usage_scan_returns_no_prefix_totals(tmp_path, limit):
    """A byte slice stays partial; an oversized record retains its typed refusal."""
    records = [
        _assistant("complete-response", "claude-sonnet-5-5", output=3, at="2026-10-06T00:00:00Z"),
        {"type": "user", "payload": {"content": SECRET * 100}},
    ]
    path = _write(tmp_path / "session.jsonl", records)
    data = path.read_bytes()
    limits = {
        "max_bytes": len(data),
        "max_line_bytes": max(map(len, data.splitlines(keepends=True))),
    }
    assert read_session(path, host=HOST_CLAUDE_CODE, **limits).incomplete == 0
    limits[limit] -= 1
    if limit == "max_bytes":
        partial = read_session(path, host=HOST_CLAUDE_CODE, **limits)
        assert partial.read_through < partial.total_bytes and partial.bytes_read == limits[limit]
        assert SECRET not in repr(partial)
        return
    with pytest.raises(NativeUsageReadLimitError) as caught:
        read_session(path, host=HOST_CLAUDE_CODE, **limits)
    assert caught.value.limit == limit and SECRET not in str(caught.value)
    assert str(caught.value) == "native_usage_limit_exceeded"
    assert owner_failure_code(caught.value) == "native_usage.read_limit_exceeded"
    assert (
        read_session(path, host=HOST_CLAUDE_CODE).responses[0].response_id
        == sha256(b"complete-response").hexdigest()
    )


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
        def __init__(self, workspace, *, timeout, goal=None):
            assert timeout == 2.0
            assert goal is None
            self.workspace = workspace

        def publish_native_event(self, project, event):
            assert event == {"source": "native_usage_read"}
            binding = NativeResearchBinding.read(project)
            assert binding is not None and binding.workspace == self.workspace
            return read_session_usage(project, binding, publish=self.publish_event)

        def publish_event(self, body):
            seen.append(body)
            return {"status": "APPENDED", "observation_id": f"observation-{len(seen)}"}

    monkeypatch.setattr(client_module, "LocalResearchClient", AcceptedClient)
    return seen


def _claude_project(tmp_path, monkeypatch, *, usage=None, roles=("alphalattice_cro",)):
    """A configured Claude project bound to ``lead-session`` and its synthetic session root."""
    config = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    declare_project(project, HOST_CLAUDE_CODE)
    binding = {
        "session_id": "lead-session",
        "workspace": str(tmp_path / "workspace"),
        "roles": list(roles),
        "host": "claude-code",
        **({"usage": usage} if usage else {}),
    }
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(binding))
    (project / ".claude/agents").mkdir(parents=True)
    (project / ".claude/agents/alphalattice_cro.md").write_text(
        "---\nname: alphalattice_cro\nmodel: claude-sonnet-5-5\neffort: high\n---\nThe card.\n"
    )
    return config, project


def _claude_child(config, agent, role, records):
    """A child the lead keeps under its own Session directory, with its sidecar."""
    path = config / f"projects/p/lead-session/subagents/agent-{agent}.jsonl"
    header = {"type": "user", "sessionId": "lead-session", "agentId": agent, "isSidechain": True}
    _write(path, [header, *records])
    path.with_suffix(".meta.json").write_text(
        json.dumps({"agentType": role, "spawnDepth": 1, "description": SECRET})
    )
    return path


def _incremental_probe(tmp_path, monkeypatch, records, *, host=HOST_CLAUDE_CODE):
    native, project = tmp_path / "native", tmp_path / "project"
    monkeypatch.setenv("CODEX_HOME" if host == HOST_CODEX else "CLAUDE_CONFIG_DIR", str(native))
    (project / ".codex").mkdir(parents=True)
    declare_project(project, host)
    session = str(uuid4()) if host == HOST_CODEX else "lead-session"
    binding = NativeResearchBinding(session, tmp_path / "workspace", (), host)
    path = native / (
        f"sessions/2026/10/11/rollout-fixture-{session}.jsonl"
        if host == HOST_CODEX
        else f"projects/p/{session}.jsonl"
    )
    _write(path, records)
    filed = []

    def pull():
        filed.clear()
        return read_session_usage(
            project, binding, publish=lambda d: filed.append(d) or {"status": "APPENDED"}
        )

    return path, binding, pull, filed


def _incremental_scenario_inputs(host):
    def context(model):
        return [{"type": "turn_context", "payload": {"model": model}}] if host == HOST_CODEX else []

    def spent(ident, model, output):
        if host == HOST_CLAUDE_CODE:
            return _assistant(ident, model, output=output, at="2026-10-11T00:00:00Z")
        counts = {
            "input_tokens": 10,
            "cached_input_tokens": 4,
            "cache_write_input_tokens": 0,
            "output_tokens": output,
        }
        return {
            "type": "token_usage_record",
            "payload": {"response_id": ident, "usage": counts},
        }

    ignored = {"type": "user", "message": {"content": SECRET}}
    records = [
        ignored,
        *context("first-model"),
        *[spent(str(i), "first-model", 2) for i in range(8)],
        *context("second-model"),
        *[spent(str(i), "second-model", 2) for i in range(8, 12)],
    ]
    mutations = {
        "append": (
            [ignored] * 15 + [spent("new", "second-model", 5)],
            {"first-model": (8, 16), "second-model": (5, 13)},
        ),
        "response_replacement": (
            [*context("first-model"), spent("0", "first-model", 1)],
            {"first-model": (8, 15), "second-model": (5, 13)},
        ),
        "file_replacement": (
            [*context("replacement"), spent("replacement", "replacement", 13)],
            {"replacement": (1, 13)},
        ),
        "file_shrink": (
            [*context("shrink"), spent("shrink", "shrink", 11)],
            {"shrink": (1, 11)},
        ),
    }
    return records, mutations


def test_incremental_usage_scenario_keeps_bounded_exact_model_counts(tmp_path, monkeypatch):
    """The regenerated scenario holds exact counts across bounded reads and source changes."""
    monkeypatch.setattr(bridge_module, "USAGE_READ_BYTES", 450)
    owner_read, scans = bridge_module.read_session, []

    def observe(path, **limits):
        scans.append(limits["checkpoint"].get("offset", 0))
        return owner_read(path, **limits)

    monkeypatch.setattr(bridge_module, "read_session", observe)

    def run(host):

        records, mutations = _incremental_scenario_inputs(host)
        path, binding, pull, filed = _incremental_probe(
            tmp_path / host, monkeypatch, records, host=host
        )
        state = {
            "step": "",
            "cursor": 0,
            "cache": binding.workspace / "runtime/native-usage",
            "expected": {"first-model": (8, 16), "second-model": (4, 8)},
        }

        def mutate(step):
            state["step"] = step
            cache = next(state["cache"].glob("*.json"), None)
            if step == "unterminated_final_line":
                with path.open("ab") as stream:
                    stream.write(json.dumps({"type": "user", "message": SECRET}).encode())
            elif step == "cumulative_mismatch":
                record = mutations["file_shrink"][0][-1]
                state["mismatch"] = host == HOST_CODEX
                if state["mismatch"]:
                    record["payload"]["thread_token_usage"] = {
                        **record["payload"]["usage"],
                        "output_tokens": 99,
                    }
                with path.open("ab") as stream:
                    stream.write(("\n" + json.dumps(record) + "\n").encode())
            elif step == "corrupt_checkpoint":
                checkpoint = json.loads(cache.read_text())
                checkpoint["responses"][0]["output_tokens"] = 12345
                cache.write_text(json.dumps(checkpoint))  # Digest must refuse this edit.
            elif step == "same_size_mtime":
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            if step not in mutations:
                return
            records, state["expected"] = mutations[step]
            if step in {"append", "response_replacement"}:
                with path.open("ab") as stream:
                    stream.write("".join(json.dumps(r) + "\n" for r in records).encode())
            else:
                target = path.with_suffix(".replacement") if step == "file_replacement" else path
                _write(target, records)
                if target != path:
                    os.replace(target, path)

        def read():
            step, previous, progress = state["step"], state["cursor"], []
            failed = state.get("mismatch", False)
            pending = step in {"cold_cap", "unterminated_final_line"}
            resets = step in {
                "file_replacement",
                "file_shrink",
                "corrupt_checkpoint",
                "same_size_mtime",
            }
            for _ in range(30):
                receipt = pull()
                checkpoint = json.loads(next(state["cache"].glob("*.json")).read_text())
                member = receipt["participants"][0]
                progress.append(member)
                assert (
                    receipt["status"] == member["status"] in {"PARTIAL", "DELIVERED", "UNAVAILABLE"}
                )
                assert (
                    member["total_bytes"] == path.stat().st_size
                    and 0 <= member["bytes_read"] <= 450
                )
                assert member["rescan"] == (resets and len(progress) == 1)
                through = member["read_through"]
                if not member.get("rescan", False) and step not in {
                    "corrupt_checkpoint",
                    "same_size_mtime",
                }:
                    assert scans[-1] == previous and through >= previous
                previous = through
                if member["status"] == "PARTIAL":
                    assert member["reason"] == "native_bridge.usage_catching_up" and not filed
                    assert (
                        "models" not in member and receipt["read_through"] < member["total_bytes"]
                    )
                if pending or member["status"] != "PARTIAL":
                    break
            expected_status = "UNAVAILABLE" if failed else "PARTIAL" if pending else "DELIVERED"
            assert member["status"] == expected_status
            if failed:
                assert member["reason"] == "native_bridge.usage_cumulative_mismatch" and not filed
                assert checkpoint["responses"][0]["output_tokens"] == 11
                again = pull()
                warm = json.loads(next(state["cache"].glob("*.json")).read_text())
                assert again["reason"] == receipt["reason"] and warm["bytes_read"] == 0
                assert warm["offset"] == checkpoint["offset"] and not filed
            assert all(m["read_through"] > p["read_through"] for p, m in pairwise(progress))
            if not pending and not failed:
                assert member["read_through"] == member["total_bytes"]
                subjects = (d["subject"] for d in filed)
                assert {
                    s["model"]: (int(s["responses"]), int(s["output_tokens"])) for s in subjects
                } == state["expected"]
            if step == "unchanged":
                assert len(progress) == 1 and member["bytes_read"] == 0
            state["cursor"] = previous
            return {"receipt": receipt, "reads": len(progress)}

        scenario, trace = walk(read, mutate)
        assert scenario.id == "native_usage_incremental"
        assert [e.attributes["alphalattice.mutation"] for e in trace.events] == list(STEPS)
        saved = list((binding.workspace / "runtime/native-usage").glob("*.json"))
        assert saved and all(SECRET not in p.read_text(encoding="utf-8") for p in saved)
        assert all(
            len(r["response_id"]) == 64
            for p in saved
            for r in json.loads(p.read_text())["responses"]
        )

    for host in (HOST_CODEX, HOST_CLAUDE_CODE):
        run(host)


def test_the_host_reads_the_lead_and_the_children_its_own_session_records(tmp_path, monkeypatch):
    """The Host reads lead and child provenance from its own native session records."""
    config, project = _claude_project(tmp_path, monkeypatch)
    _write(config / "projects/p/lead-session.jsonl", _lead_records()[:-1])
    # The card pins claude-sonnet-5-5; the host ran an older model.
    spent = [_assistant("msg_c", "claude-sonnet-5", output=3, at="2026-09-29T01:00:08Z")]
    _claude_child(config, "child", "alphalattice_cro", spent)
    _claude_child(config, "stranger", "general-purpose", spent)
    binding = NativeResearchBinding.read(project)
    filed = []

    def publish(document):
        ExternalActivityEventDocument.model_validate(document)
        filed.append(document)
        return {"status": "APPENDED", "observation_id": f"observation-{len(filed)}"}

    receipt = read_session_usage(project, binding, publish=publish)

    assert receipt["status"] == "DELIVERED" and "reason" not in receipt
    members = {m["agent_id"]: m for m in receipt["participants"]}
    assert members["lead-session"]["status"] == "DELIVERED"
    assert members["child"]["status"] == members["stranger"]["status"] == "DELIVERED"
    assert members["child"]["specialist_verified"] is True
    assert members["stranger"]["role"] == "general-purpose"
    assert members["stranger"]["specialist_verified"] is False
    led, child, helper = filed
    assert "role" not in helper["subject"]
    assert helper["subject"]["observed_role"] == "general-purpose"
    assert helper["subject"]["specialist_verified"] == "false"
    assert (led["subject"]["native_agent_id"], led["subject"]["role"]) == (
        "lead-session",
        "research_lead",
    )
    assert (led["subject"]["responses"], led["subject"]["output_tokens"]) == ("2", "47")
    assert "pin_differs" not in led["subject"]  # the lead has no role card
    assert (child["subject"]["native_agent_id"], child["subject"]["role"]) == (
        "child",
        "alphalattice_cro",
    )
    assert child["subject"]["pin_differs"] == "model"
    assert child["subject"]["input_channel"] == "CLAUDE_CODE_SESSION_FILE"
    assert child["subject"]["last_at"] == "2026-09-29T01:00:08Z"
    assert all(len(item["subject"]) <= 16 for item in filed)
    sent = json.dumps(filed)
    assert SECRET not in sent and str(config) not in sent and "transcript" not in sent
    # The same counts read again are the same events: their sequence is their content's.
    again = []
    read_session_usage(
        project, binding, publish=lambda d: again.append(d) or {"status": "APPENDED"}
    )
    assert again == filed
    records = [*_lead_records()[:-1], _assistant("msg_4", "claude-opus-5-5", output=2, at="t")]
    _write(config / "projects/p/lead-session.jsonl", records)
    later = []
    read_session_usage(
        project, binding, publish=lambda d: later.append(d) or {"status": "APPENDED"}
    )
    assert later[0]["subject"]["responses"] == "3"
    assert later[0]["producer_sequence"] != filed[0]["producer_sequence"]
    assert later[1] == filed[1]
    for number in range(65):
        _claude_child(config, f"many-{number}", "alphalattice_cro", spent)
    many = []
    receipt = read_session_usage(
        project, binding, publish=lambda d: many.append(d) or {"status": "APPENDED"}
    )
    assert receipt["status"] == "DELIVERED"
    assert len(many) == len(receipt["participants"]) == 68


def test_a_codex_lead_names_its_children_in_its_own_rollout(tmp_path, monkeypatch):
    """Recorded exact-parent children retain independent usage when their lead fails."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    day = tmp_path / "sessions" / "2026" / "10" / "07"
    day.mkdir(parents=True)
    lead, child, unknown, stray, unnamed = (str(uuid4()) for _ in range(5))

    def rollout(thread, *records):
        name = f"rollout-2026-10-07T00-00-00-{thread}.jsonl"
        (day / name).write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")

    def activity(thread, kind):
        item = {"type": "SubAgentActivity", "id": SECRET, "kind": kind, "agent_thread_id": thread}
        return {"type": "event_msg", "payload": {"type": "item_completed", "item": item}}

    def meta(thread, parent=lead, role="alphalattice_risk"):
        spawn = {"parent_thread_id": parent, "agent_role": role}
        source = {"subagent": {"thread_spawn": spawn}}
        return {"type": "session_meta", "payload": {"id": thread, "source": source}}

    def usage(ident):
        counts = {
            "input_tokens": 10,
            "cached_input_tokens": 4,
            "cache_write_input_tokens": 0,
            "output_tokens": 2,
        }
        return {"type": "token_usage_record", "payload": {"response_id": ident, "usage": counts}}

    turn = {"type": "turn_context", "payload": {"model": "gpt-6-luna", "effort": "high"}}
    rollout(
        lead,
        {"type": "session_meta", "payload": {"id": lead, "source": "cli"}},
        turn,
        usage("r1"),
        activity(child, "started"),
        activity(child, "completed"),
        activity(unknown, "future-kind"),
        activity(stray, "started"),
        activity(unnamed, "interacted"),
        activity("not-a-thread", "started"),
        {"type": "event_msg", "payload": {"type": "item_completed", "item": {"type": "Other"}}},
    )
    rollout(child, meta(child), turn, usage("c1"), activity(lead, "interacted"))
    rollout(unknown, meta(unknown, role=None), turn, usage("u1"))
    rollout(stray, meta(stray, str(uuid4())))
    rollout(unnamed, meta(unnamed), turn, usage("not-recorded"))

    reading = read_session(codex_session_file(lead), host=HOST_CODEX)
    assert reading.children == (child, unknown, stray)
    checkpoint = {}
    path = codex_session_file(child)
    assert read_session(path, host=HOST_CODEX, checkpoint=checkpoint).children == ()
    for children, expected_bytes in (([lead], path.stat().st_size), ([], 0)):
        checkpoint.pop("child_discovery")
        checkpoint["children"] = children
        body = {k: v for k, v in checkpoint.items() if k != "digest"}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        checkpoint["digest"] = sha256(encoded).hexdigest()
        rebuilt = read_session(path, host=HOST_CODEX, checkpoint=checkpoint)
        assert (
            not rebuilt.rescan and rebuilt.children == () and rebuilt.bytes_read == expected_bytes
        )
        warm = read_session(path, host=HOST_CODEX, checkpoint=checkpoint)
        assert not warm.rescan and warm.bytes_read == 0
    assert reading.incomplete == 0 and [r.response_id for r in reading.responses] == [
        sha256(b"r1").hexdigest()
    ]
    assert SECRET not in repr(reading)

    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    declare_project(project, HOST_CODEX)
    binding = {
        "session_id": lead,
        "workspace": str(tmp_path / "workspace"),
        "roles": ["alphalattice_risk"],
        "host": "codex",
    }
    (project / ".codex" / BINDING_NAME).write_text(json.dumps(binding))
    filed = []
    receipt = read_session_usage(
        project,
        NativeResearchBinding.read(project),
        publish=lambda d: filed.append(d) or {"status": "APPENDED"},
    )
    members = {m["agent_id"]: m for m in receipt["participants"]}
    assert {a: m["status"] for a, m in members.items()} == {
        lead: "DELIVERED",
        child: "DELIVERED",
        unknown: "DELIVERED",
        stray: "UNAVAILABLE",
    }
    assert members[child]["specialist_verified"] is True
    assert members[unknown]["role"] is None and members[unknown]["specialist_verified"] is False
    assert [d["subject"]["native_agent_id"] for d in filed] == [lead, child, unknown]
    assert filed[1]["subject"]["role"] == "alphalattice_risk"
    assert "role" not in filed[2]["subject"] and "observed_role" not in filed[2]["subject"]
    assert filed[2]["subject"]["specialist_verified"] == "false"
    for number in range(65):
        extra = str(uuid4())
        rollout(extra, meta(extra), turn, usage(f"many-{number}"))
        with codex_session_file(lead).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(activity(extra, "started")) + "\n")
    many = []
    receipt = read_session_usage(
        project,
        NativeResearchBinding.read(project),
        publish=lambda d: many.append(d) or {"status": "APPENDED"},
    )
    assert len(many) == 68 and len(receipt["participants"]) == 69
    assert receipt["status"] == "PARTIAL"  # The earlier stray names another parent.
    with codex_session_file(lead).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "user", "message": {"content": SECRET * 40}}) + "\n")
    monkeypatch.setattr(bridge_module, "USAGE_LINE_BYTES", 512)
    failed = read_session_usage(
        project, NativeResearchBinding.read(project), publish=lambda d: {"status": "APPENDED"}
    )
    cached = {m["agent_id"]: m for m in failed["participants"]}
    assert (
        cached[lead]["status"] == "UNAVAILABLE" and cached[lead]["read_limit"] == "max_line_bytes"
    )
    assert cached[child]["status"] == cached[unknown]["status"] == "DELIVERED"
    assert cached[stray]["status"] == "UNAVAILABLE" and unnamed not in cached
    assert len(cached) == 69
    codex_session_file(lead).unlink()
    missing = read_session_usage(
        project, NativeResearchBinding.read(project), publish=lambda d: {"status": "APPENDED"}
    )
    missing_members = {m["agent_id"]: m for m in missing["participants"]}
    assert missing_members[lead]["reason"] == "native_bridge.lead_usage_file_missing"
    assert missing_members[child]["status"] == missing_members[unknown]["status"] == "DELIVERED"


def test_claude_children_are_listed_from_the_leads_own_directory_alone(tmp_path, monkeypatch):
    """requirement (FLOW-1): only ``<session>/subagents/agent-<id>.jsonl`` beside the bound
    lead's own file names children; another Session's directory is never listed."""
    config = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    assert claude_children("lead-session") == ()
    _write(config / "projects/p/lead-session.jsonl", _lead_records())
    assert claude_children("lead-session") == ()
    for name in ("agent-b.jsonl", "agent-a.jsonl", "agent-a.meta.json", "notes.txt"):
        _write(config / "projects/p/lead-session/subagents" / name, [])
    _write(config / "projects/p/other-session/subagents/agent-c.jsonl", [])
    assert claude_children("lead-session") == ("a", "b")
    for number in range(65):
        _write(config / f"projects/p/lead-session/subagents/agent-many{number}.jsonl", [])
    assert len(claude_children("lead-session")) == 67


@pytest.mark.parametrize("off", ["binding", "person", "unknown_control", "unreadable_control"])
def test_reading_off_opens_no_session_file(tmp_path, monkeypatch, off):
    """requirement (privacy, FLOW-1): a binding with ``"usage": "OFF"``, or the person's
    workspace switch turned off, reads nothing at all; a switch record this owner did not
    write, or cannot read, reads as off."""
    from alphalattice.interface.local_application import native_bridge

    config, project = _claude_project(
        tmp_path, monkeypatch, usage="OFF" if off == "binding" else None
    )
    _write(config / "projects/p/lead-session.jsonl", _lead_records())
    workspace = tmp_path / "workspace"
    assert native_bridge.usage_reading(workspace) is True
    control = workspace / native_bridge.USAGE_CONTROL_PATH
    if off == "person":
        answer = native_bridge.set_usage_reading(workspace, enabled=False)
        assert answer["usage_reading"] == "OFF"
        assert answer["next_requests"]["set"]["usage_reading_enabled"] is True
    elif off != "binding":
        control.parent.mkdir(parents=True)
        control.write_text(
            "{" if off == "unreadable_control" else '{"reading": true}', encoding="utf-8"
        )
    assert native_bridge.usage_reading(workspace) is (off == "binding")

    def opened(*_args, **_kwargs):
        raise AssertionError("a session file was read")

    monkeypatch.setattr(native_bridge, "read_session", opened)
    monkeypatch.setattr(native_bridge, "session_file", opened)
    monkeypatch.setattr(native_bridge, "claude_children", opened)
    filed = []
    receipt = read_session_usage(
        project, NativeResearchBinding.read(project), publish=lambda d: filed.append(d) or {}
    )
    assert receipt == {"status": "SKIPPED", "reason": "native_bridge.usage_disabled"}
    assert filed == []


@pytest.mark.parametrize("host", [HOST_CODEX, HOST_CLAUDE_CODE])
def test_an_unknown_record_shape_reads_as_unavailable_never_as_zero(tmp_path, monkeypatch, host):
    """requirement (FLOW-1): a host that changed its usage record files nothing: no zero total,
    no partial count, and no refusal of research."""
    from alphalattice.interface.local_application import native_bridge

    if host == HOST_CODEX:
        records = [
            {"type": "turn_context", "payload": {"model": "gpt-6-luna"}},
            {"type": "token_usage_record", "payload": {"response_id": "r", "spent": {"n": 9}}},
        ]
    else:
        records = [{"type": "assistant", "message": {"id": "m", "model": "claude-x", "cost": 9}}]
    path = _write(tmp_path / "lead.jsonl", records)
    monkeypatch.setattr(native_bridge, "session_file", lambda *_args: path)
    binding = NativeResearchBinding(
        session_id="lead-session", workspace=tmp_path, roles=("alphalattice_cro",), host=host
    )
    filed = []
    receipt = read_session_usage(tmp_path, binding, publish=lambda d: filed.append(d) or {})
    assert receipt["status"] == "UNAVAILABLE"
    (lead,) = receipt["participants"]
    assert lead["reason"] in {
        "native_bridge.lead_usage_incomplete",
        "native_bridge.lead_usage_not_observed",
    }
    assert filed == []


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
    """The lead's own commands read only its bound Session and publish only complete usage."""

    from alphalattice.interface.local_application.client import lead_readings

    config = tmp_path / "claude"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    project = tmp_path / "project"
    (project / ".codex").mkdir(parents=True)
    declare_project(project, HOST_CLAUDE_CODE)
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
    (receipt,) = lead_readings(project, {"CLAUDE_CODE_SESSION_ID": "lead-session"})
    assert receipt["status"] == "UNAVAILABLE"
    member = receipt["participants"][0]
    assert tuple(member[key] for key in ("agent_id", "role", "status", "reason", "incomplete")) == (
        "lead-session",
        "research_lead",
        "UNAVAILABLE",
        "native_bridge.lead_usage_incomplete",
        1,
    )
    size = lead.stat().st_size
    assert tuple(
        member[key] for key in ("read_through", "total_bytes", "bytes_read", "rescan")
    ) == (size, size, size, False)
    assert receiver == []
    # The parser fixture's final empty-usage response is incomplete. Removing only that
    # synthetic record gives a complete source with the same two actual response counts.
    _write(lead, _lead_records()[:-1])
    (receipt,) = lead_readings(project, {"CLAUDE_CODE_SESSION_ID": "lead-session"})
    # The child holds no sidecar, so it names this Session nowhere and is not read.
    assert receipt["status"] == "PARTIAL"
    assert [m["status"] for m in receipt["participants"]] == ["DELIVERED", "UNAVAILABLE"]
    [reading] = receiver
    assert reading["event_kind"] == "NATIVE_AGENT_USAGE"
    subject = reading["subject"]
    assert (subject["native_agent_id"], subject["role"]) == ("lead-session", "research_lead")
    assert (subject["responses"], subject["output_tokens"]) == ("2", "47")
    assert SECRET not in json.dumps(receiver)
    (project / ".codex" / BINDING_NAME).write_text(json.dumps({**binding, "usage": "OFF"}))
    assert lead_readings(project, {"CLAUDE_CODE_SESSION_ID": "lead-session"}) == []


def test_a_codex_thread_names_its_spawn_in_its_first_record_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Codex thread names its spawn in its first record alone."""

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
    assert asdict(found) == {"parent_thread_id": lead, "agent_role": "alphalattice_cro"}
    assert codex_thread_spawn(lead) is None and codex_thread_spawn(other) is None
    # A first record naming another thread is not this thread's.
    rollout(other, {"type": "session_meta", "payload": {**meta, "id": lead}})
    assert codex_thread_spawn(other) is None
    monkeypatch.setattr(native_usage, "FIRST_RECORD_BYTES", 64)
    assert codex_thread_spawn(child) is None
