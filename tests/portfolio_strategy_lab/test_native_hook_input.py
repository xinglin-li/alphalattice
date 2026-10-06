"""Native lifecycle decoding neither copies private content nor grants authority."""

import json
from dataclasses import asdict

import pytest

from alphalattice.interface.local_application.native_hook_input import (
    MAX_HOOK_INPUT_BYTES,
    NativeHookInputError,
    read_subagent_lifecycle,
)


def _payload(root, **changes):
    return {
        "hook_event_name": "SubagentStart",
        "session_id": "parent",
        "turn_id": "turn",
        "agent_id": "child",
        "agent_type": "alphalattice_cro",
        "cwd": str(root),
        **changes,
    }


def _read(root, payload):
    return read_subagent_lifecycle(
        json.dumps(payload).encode(), expected_cwd=root, expected_session_id="parent"
    )


def test_only_lifecycle_metadata_survives_additive_host_changes(tmp_path):
    expected = _read(tmp_path, _payload(tmp_path))
    incoming = _payload(
        tmp_path,
        transcript_path=str(tmp_path / "must-not-open"),
        agent_transcript_path="private-rollout.jsonl",
        last_assistant_message="private final output",
        future_field={"secret": "must-not-retain"},
        version="some-future-host-version",
        authority="PRODUCT_VERIFIED",
    )
    assert _read(tmp_path, incoming) == expected
    assert set(asdict(expected)) == {
        "hook_event_name",
        "host",
        "session_id",
        "turn_id",
        "agent_id",
        "agent_type",
        "model",
        "permission_mode",
        "stop_hook_active",
    }
    assert expected.host == "codex"
    assert not list(tmp_path.iterdir())
    assert _read(tmp_path, _payload(tmp_path, hook_event_name="FutureHook")) is None


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"session_id": "other-parent"}, "session_mismatch"),
        ({"cwd": "relative/project"}, "cwd_mismatch"),
        ({"turn_id": None}, "invalid_turn_id"),
        ({"agent_id": True}, "invalid_agent_id"),
        ({"agent_type": "bad\nrole"}, "invalid_agent_type"),
        ({"hook_event_name": "SubagentStop", "stop_hook_active": 1}, "invalid_stop_hook_active"),
    ],
)
def test_known_events_refuse_invalid_identity_without_echoing_input(tmp_path, changes, code):
    with pytest.raises(NativeHookInputError) as caught:
        _read(tmp_path, _payload(tmp_path, **changes))
    assert str(caught.value) == "native_hook." + code


def test_binding_and_stop_semantics_are_not_inferred(tmp_path):
    payload = _payload(
        tmp_path,
        hook_event_name="SubagentStop",
        stop_hook_active=True,
        permission_mode="futureMode",
        model="future-model",
    )
    observed = _read(tmp_path, payload)
    assert observed.hook_event_name == "SubagentStop"
    assert observed.stop_hook_active is True
    assert observed.permission_mode == "futureMode"
    assert "status" not in asdict(observed)
    with pytest.raises(NativeHookInputError, match="agent_mismatch"):
        read_subagent_lifecycle(
            json.dumps(payload).encode(),
            expected_cwd=tmp_path,
            expected_session_id="parent",
            expected_agent_id="different-child",
        )
    with pytest.raises(NativeHookInputError, match="cwd_mismatch"):
        _read(tmp_path / "different", payload)


@pytest.mark.parametrize(
    "data",
    [b'"scalar"', b'{"secret":', b"\xff", b'{"x":1,"x":2}', b" " * (MAX_HOOK_INPUT_BYTES + 1)],
    ids=["scalar", "incomplete-json", "invalid-utf8", "duplicate-key", "oversized"],
)
def test_invalid_or_oversized_transport_refuses_without_payload(tmp_path, data):
    with pytest.raises(NativeHookInputError) as caught:
        read_subagent_lifecycle(data, expected_cwd=tmp_path, expected_session_id="parent")
    assert str(caught.value) in {
        "native_hook.object_required",
        "native_hook.invalid_json",
        "native_hook.input_too_large",
    }


def test_claude_code_names_the_turn_prompt_id_and_the_host_follows_the_key(tmp_path):
    codex = _read(tmp_path, _payload(tmp_path))
    payload = _payload(tmp_path, prompt_id="prompt-1", permission_mode="default")
    del payload["turn_id"]
    claude = _read(tmp_path, payload)
    assert (claude.host, claude.turn_id) == ("claude-code", "prompt-1")
    assert (codex.host, codex.turn_id) == ("codex", "turn")
    assert asdict(claude) == {
        **asdict(codex),
        "host": "claude-code",
        "turn_id": "prompt-1",
        "permission_mode": "default",
    }
    # Both keys present: the Codex name wins; an empty Claude prompt refuses as the turn.
    assert _read(tmp_path, _payload(tmp_path, prompt_id="ignored")).host == "codex"
    with pytest.raises(NativeHookInputError, match="invalid_prompt_id"):
        _read(tmp_path, {**payload, "prompt_id": ""})
