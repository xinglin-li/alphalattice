"""Select scoped lifecycle metadata, not transcripts or product authority.

This is the native-input side of the adapter. Its output retains the host's field
meaning: a SubagentStop hook can request continuation, so its input does not
establish a terminal turn, agent or product Task. Two hosts send the same shape
under one different name: Codex identifies the turn as `turn_id`, Claude Code as
`prompt_id`; the key that arrived names the host.
Shape/scope validation cannot authenticate a host; the admitted producer and
product boundary must establish provenance separately. No host version pin.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

MAX_HOOK_INPUT_BYTES = 1_048_576
FIELD_CHARACTERS = 512
"""The longest field read from a hook: its directory is checked, never carried; a carried id,
role or model meets the Host's subject bound where the bridge builds the event (V574)."""


class NativeHookInputError(ValueError):
    """A safe failure code with no rejected input or parser excerpt."""


def definition_digest(project: Path, host: str) -> str:
    """Fingerprint the local lifecycle declarations; this establishes no runtime trust.

    Args:
        project: The exact admitted native project.
        host: The host whose two lifecycle definitions are selected.

    Returns:
        The SHA-256 of those local definitions, independent of other configuration.

    Raises:
        NativeHookInputError: A declaration is unsafe, missing or malformed.
    """
    path = project / (".codex/config.toml" if host == "codex" else ".claude/settings.json")
    if host not in {"codex", "claude-code"} or path.is_symlink() or path.parent.is_symlink():
        raise NativeHookInputError("native_hook.definition_unreadable")
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_HOOK_INPUT_BYTES + 1)
        if len(data) > MAX_HOOK_INPUT_BYTES:
            raise ValueError
        document = (
            tomllib.loads(data.decode("utf-8"))
            if host == "codex"
            else json.loads(data, object_pairs_hook=_unique_object)
        )
        hooks = document["hooks"]
        selected = {
            event: [
                group
                for group in hooks[event]
                if isinstance(group, dict) and group.get("matcher") == "^alphalattice_.*$"
            ]
            for event in ("SubagentStart", "SubagentStop")
        }
        if any(not groups for groups in selected.values()):
            raise ValueError
        encoded = json.dumps({"host": host, "hooks": selected}, sort_keys=True).encode("utf-8")
    except (OSError, KeyError, TypeError, ValueError, UnicodeError, RecursionError):
        raise NativeHookInputError("native_hook.definition_unreadable") from None
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeSubagentLifecycle:
    """Scoped lifecycle metadata from an admitted native subagent hook."""

    hook_event_name: Literal["SubagentStart", "SubagentStop"]
    host: Literal["codex", "claude-code"]
    session_id: str
    turn_id: str
    agent_id: str
    agent_type: str
    model: str | None
    permission_mode: str | None
    stop_hook_active: bool | None


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise NativeHookInputError("native_hook.duplicate_field")
        result[key] = value
    return result


def _field(payload: dict[str, object], key: str, *, optional: bool = False) -> str | None:
    value = payload.get(key)
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > FIELD_CHARACTERS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise NativeHookInputError(f"native_hook.invalid_{key}")
    return value


def read_subagent_lifecycle(
    data: bytes,
    *,
    expected_cwd: Path,
    expected_session_id: str,
    expected_agent_id: str | None = None,
) -> NativeSubagentLifecycle | None:
    """Drop unknown events; reject malformed or out-of-scope known events.

    Expected scope comes from the caller's admitted binding, never from data.
    Unknown fields are discarded. In particular, transcript paths and the last
    assistant message are neither opened nor returned; output handling belongs
    to the separate explicitly scoped analysis/submission path.
    """
    if len(data) > MAX_HOOK_INPUT_BYTES:
        raise NativeHookInputError("native_hook.input_too_large")
    try:
        payload = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError):
        # Includes duplicate-field errors: do not expose decoder input excerpts.
        raise NativeHookInputError("native_hook.invalid_json") from None
    if not isinstance(payload, dict):
        raise NativeHookInputError("native_hook.object_required")
    event = payload.get("hook_event_name")
    if event not in ("SubagentStart", "SubagentStop"):
        return None
    session_id = cast(str, _field(payload, "session_id"))
    agent_id = cast(str, _field(payload, "agent_id"))
    cwd = cast(str, _field(payload, "cwd"))
    if session_id != expected_session_id:
        raise NativeHookInputError("native_hook.session_mismatch")
    if expected_agent_id is not None and agent_id != expected_agent_id:
        raise NativeHookInputError("native_hook.agent_mismatch")
    try:
        cwd_matches = Path(cwd).is_absolute() and Path(cwd).resolve() == expected_cwd.resolve()
    except (OSError, ValueError, RuntimeError):
        cwd_matches = False
    if not cwd_matches:
        raise NativeHookInputError("native_hook.cwd_mismatch")
    stop_active = payload.get("stop_hook_active") if event == "SubagentStop" else None
    if stop_active is not None and not isinstance(stop_active, bool):
        raise NativeHookInputError("native_hook.invalid_stop_hook_active")
    # Codex names the turn `turn_id`; Claude Code names it `prompt_id`. Exactly one arrives.
    if payload.get("turn_id") is None and payload.get("prompt_id") is not None:
        host: Literal["codex", "claude-code"] = "claude-code"
        turn_id = cast(str, _field(payload, "prompt_id"))
    else:
        host = "codex"
        turn_id = cast(str, _field(payload, "turn_id"))
    return NativeSubagentLifecycle(
        hook_event_name=event,
        host=host,
        session_id=session_id,
        turn_id=turn_id,
        agent_id=agent_id,
        agent_type=cast(str, _field(payload, "agent_type")),
        model=_field(payload, "model", optional=True),
        permission_mode=_field(payload, "permission_mode", optional=True),
        stop_hook_active=stop_active,
    )
