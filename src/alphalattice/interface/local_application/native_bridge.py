"""Project-scoped native observation delivery; never a research executor.

The native wire mapping uses the accepted external activity event contract.
Transport authentication is not authentication of an actor's claims. At a subagent's stop
the bridge also delivers what the subagent and the lead have run and spent so far, read from
the host's own session files by `native_usage` (AU, V300; LAWS OP13, OP14), unless the
session's binding turned that reading off (`"usage": "OFF"`, `native_research.py bind --usage
off`); every other observation goes on as before.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from alphalattice.control.workspace_runtime.lock import WorkspaceLock
from alphalattice.interface.local_application.cli_contract import agent_session
from alphalattice.interface.local_application.failure_codes import safe_failure_code
from alphalattice.interface.local_application.native_hook_input import (
    NativeHookInputError,
    read_subagent_lifecycle,
)
from alphalattice.interface.local_application.native_observation_sequence import (
    LOCK_NAME,
    NativeSequenceError,
    metadata_path,
    reserve_sequence,
)
from alphalattice.interface.local_application.native_usage import (
    ModelUsage,
    canonical_agent_path,
    codex_thread_spawn,
    hook_session_files,
    read_session,
    session_file,
)
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
from alphalattice.protocols.actor_execution.answers import AgentAnswerRecord, AnswerVerdict
from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord

BINDING_NAME = "native-research.local.json"
# Every bound the bridge applies is the contract of what it carries, so it never refuses what
# that contract admits nor passes what the Host then refuses under another word; the table
# beside its tests pins each against its contract (V574).
SUBJECT_VALUE_CHARACTERS = 200
"""The longest value the Host's activity event holds in its subject: each id, role, locator
and reference the bridge carries is bounded by it where it enters."""
CORRELATION_CHARACTERS = 128
"""The longest correlation id the event holds: the bound session's id is one."""
MESSAGE_CHARACTERS = 4000
"""The longest message: the event's summary bound."""
MESSAGE_KEPT_CHARACTERS = 500
"""What the Host keeps of a message, its whitespace folded; a longer one cites a reference."""
MAX_MESSAGE_BYTES = 4 * MESSAGE_CHARACTERS
"""The most bytes a message of ``MESSAGE_CHARACTERS`` takes in UTF-8: the read never refuses
a message the character bound admits."""
TEXT_CHARACTERS = 512
"""The longest field a binding holds that is not carried: its host, reading and workspace."""
BINDING_BYTES = 8192
"""The longest binding file: the widest binding `native_research.py bind` writes fits it."""
BINDING_ROLES = 32
"""The most roles a binding names: the project's cards, each named by ``_ROLE_NAME``."""
PIN_CHARACTERS = 64
"""The longest model or effort a role card pins that the bridge carries."""
OBSERVATION_ID_CHARACTERS = 200
"""The longest observation id an acknowledgment may name; the Host's are 64-hex hashes."""
SPAWN_HOPS = 4
"""The most threads a Codex specialist's spawn chain climbs to reach the bound session: a
specialist's specialist and two more (V568)."""
# One producer identity per foreground host; the Team page shows the host as source
# information, never as authentication (the product philosophy, 2026-09-22).
HOSTS = ("codex", "claude-code")
USAGE_READINGS = ("READ", "OFF")
"""Whether a subagent's stop reads the host's session files for what it ran and spent (AU):
the binding's ``usage``, ``READ`` when it names none."""
PRODUCERS = {"codex": "codex-native", "claude-code": "claude-code-native"}
CHANNELS = {"codex": "CODEX_HOOK", "claude-code": "CLAUDE_CODE_HOOK"}
USAGE_CHANNELS = {"codex": "CODEX_SESSION_FILE", "claude-code": "CLAUDE_CODE_SESSION_FILE"}
LEAD_ROLE = "research_lead"
_USAGE_COUNTS = (
    "responses",
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
)
_ROLE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# Decision notes (决策笔记): written only at a plan, a decision, a dead end or a surprise,
# never raw reasoning; any participant may write one (GR2).
DECISION_NOTE_KINDS = frozenset({"plan", "decision", "dead_end", "surprise"})
LEAD_KINDS = frozenset({"assignment", "question", "pm_response"}) | DECISION_NOTE_KINDS
SPECIALIST_KINDS = frozenset({"question", "answer", "objection"}) | DECISION_NOTE_KINDS


class NativeBridgeError(ValueError):
    """Only code-owned error text is allowed across the hook boundary."""


def _text(value: object, bound: int = TEXT_CHARACTERS) -> str:
    if not isinstance(value, str) or not value or len(value) > bound:
        raise NativeBridgeError("native_bridge.binding_invalid")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise NativeBridgeError("native_bridge.binding_invalid")
    return value


def _carried(name: str, value: str) -> None:
    """Refuse a message's id or reference the Host's activity event could not hold (V574)."""
    try:
        _text(value, SUBJECT_VALUE_CHARACTERS)
    except NativeBridgeError:
        raise NativeBridgeError(f"native_bridge.message_field_invalid:{name}") from None


@dataclass(frozen=True, slots=True)
class NativeResearchBinding:
    """Admitted native session, workspace, roles, and producer host."""

    session_id: str
    workspace: Path
    roles: tuple[str, ...]
    host: str = "codex"
    usage: str = "READ"

    @classmethod
    def read(cls, project: Path) -> NativeResearchBinding | None:
        """Read and validate the project's native research binding.

        Args:
            project: The admitted project root.

        Returns:
            The binding, or ``None`` when no binding file exists.

        Raises:
            NativeBridgeError: If the file is unsafe, unreadable, or invalid.
        """
        path = project / ".codex" / BINDING_NAME
        try:
            if (
                path.parent.is_symlink()
                or path.is_symlink()
                or not path.resolve().is_relative_to(project.resolve())
            ):
                raise NativeBridgeError("native_bridge.binding_path_invalid")
            with path.open("rb") as stream:
                data = stream.read(BINDING_BYTES + 1)
        except FileNotFoundError:
            return None
        except OSError:
            raise NativeBridgeError("native_bridge.binding_unreadable") from None
        if len(data) > BINDING_BYTES:
            raise NativeBridgeError("native_bridge.binding_invalid")
        try:
            value = json.loads(data)
        except (ValueError, RecursionError):
            raise NativeBridgeError("native_bridge.binding_invalid") from None
        return cls.from_document(value)

    @classmethod
    def from_document(cls, value: Any) -> NativeResearchBinding:
        """Validate a binding document without trusting its declared fields.

        Args:
            value: The decoded binding document.

        Returns:
            A binding with admitted scalar fields and roles.

        Raises:
            NativeBridgeError: If required fields or scope values are invalid.
        """
        try:
            if not isinstance(value, dict):
                raise ValueError
            if set(value) - {"host", "usage"} != {"session_id", "workspace", "roles"}:
                raise ValueError
            host = _text(value.get("host", "codex"))
            if host not in HOSTS:
                raise ValueError
            usage = _text(value.get("usage", "READ"))
            if usage not in USAGE_READINGS:
                raise ValueError
            # The session's id is every event's correlation id, and a role a card's name; a
            # binding holding a longer one would deliver nothing (V574).
            session_id = _text(value["session_id"], CORRELATION_CHARACTERS)
            workspace = Path(_text(value["workspace"]))
            roles = value["roles"]
            if not workspace.is_absolute() or not isinstance(roles, list) or not roles:
                raise ValueError
            admitted_roles = tuple(_text(role) for role in roles)
            if any(_ROLE_NAME.match(role) is None for role in admitted_roles):
                raise ValueError
            duplicated = len(set(admitted_roles)) != len(admitted_roles)
            if duplicated or len(admitted_roles) > BINDING_ROLES:
                raise ValueError
        except (ValueError, TypeError, KeyError):
            raise NativeBridgeError("native_bridge.binding_invalid") from None
        return cls(
            session_id=session_id,
            workspace=workspace,
            roles=admitted_roles,
            host=host,
            usage=usage,
        )

    @classmethod
    def find(cls, start: Path) -> tuple[Path, NativeResearchBinding] | None:
        """The nearest binding from a directory upward, as git finds its `.git` (V568).

        A checkout is one such project; an installed agent project is another.

        Args:
            start: The directory a command runs in.

        Returns:
            The project holding the nearest binding, with it; None when no folder up holds one.

        Raises:
            NativeBridgeError: If the nearest binding is unsafe, unreadable or invalid.
        """
        for folder in (start, *start.parents):
            path = folder / ".codex" / BINDING_NAME
            if path.is_symlink() or path.exists():
                binding = cls.read(folder)
                if binding is not None:
                    return folder, binding
        return None

    def serves(self, session: tuple[str, str]) -> bool:
        """Whether a command of this agent session works in this binding's workspace (V568).

        The bound session's own commands do, and its own specialists': a Claude Code subagent
        carries its lead's session id, and a Codex specialist runs in a thread of its own whose
        rollout names the thread that spawned it (`codex_thread_spawn`), a chain that reaches
        the bound session within `SPAWN_HOPS`. No other session does, whatever its time.

        Args:
            session: The vendor and the session id the command's environment names.

        Returns:
            Whether the session is the bound one or one of its own specialists.
        """
        vendor, thread = session
        if vendor != self.host:
            return False
        if thread == self.session_id:
            return True
        if vendor != "codex":
            return False
        for _hop in range(SPAWN_HOPS):
            spawn = codex_thread_spawn(thread)
            if spawn is None:
                return False
            if spawn.parent_thread_id == self.session_id:
                return True
            thread = spawn.parent_thread_id
        return False


def lifecycle_event(
    project: Path, binding: NativeResearchBinding, data: bytes
) -> dict[str, object] | None:
    """Project an admitted subagent hook into one lifecycle event.

    Args:
        project: The project expected by the hook binding.
        binding: The admitted native session and role scope.
        data: The hook's bounded input bytes.

    Returns:
        A scoped event, or ``None`` for an unknown or unadmitted role.
    """
    observed = read_subagent_lifecycle(
        data, expected_cwd=project, expected_session_id=binding.session_id
    )
    if observed is None or observed.agent_type not in binding.roles:
        return None
    # A producer-channel label, not permission to self-assert HOST_VERIFIED.
    event: dict[str, object] = {
        "source": "native_hook",
        "lifecycle": asdict(observed),
        "role_pin": role_pin(project, observed.host, observed.agent_type),
    }
    if observed.host == binding.host == "codex":
        # An alias leaves only the admitted first-record owner, never hook input or a turn.
        spawn = codex_thread_spawn(observed.agent_id)
        if (
            spawn is not None
            and spawn.agent_role == observed.agent_type
            and spawn.agent_path is not None
            and binding.serves(("codex", observed.agent_id))
        ):
            event["thread_spawn"] = asdict(spawn)
            event["thread_spawn_bound_session"] = binding.session_id
    return event


def role_pin(project: Path, host: str, role: str) -> dict[str, str]:
    """The model and effort the project's role card pins, as the card writes them (AU).

    A Claude Code card may pin an alias (``sonnet``) that the host resolves; the reader of
    the pin compares it with the model the session file shows.

    Args:
        project: The project whose cards the host started the agent from.
        host: ``claude-code`` (``.claude/agents/<role>.md``) or ``codex``
            (``.codex/agents/<role>.toml``).
        role: The agent's role, a card's name.

    Returns:
        ``role_model`` and ``role_effort`` where the card names them.
    """
    if _ROLE_NAME.match(role) is None:
        return {}
    pins: dict[str, object]
    try:
        if host == "claude-code":
            card = (project / ".claude" / "agents" / f"{role}.md").read_text("utf-8")
            lines = card.split("\n")
            head = lines[1 : lines.index("---", 1)] if lines[0].strip() == "---" else []
            fields = {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in head)}
            pins = {"role_model": fields.get("model"), "role_effort": fields.get("effort")}
        else:
            with (project / ".codex" / "agents" / f"{role}.toml").open("rb") as stream:
                config = tomllib.load(stream)
            pins = {
                "role_model": config.get("model"),
                "role_effort": config.get("model_reasoning_effort"),
            }
    except (OSError, ValueError, UnicodeError):
        return {}
    return {
        key: value
        for key, value in pins.items()
        if isinstance(value, str) and 0 < len(value) <= PIN_CHARACTERS and value.isprintable()
    }


def pin_differs(pins: dict[str, str], usage: ModelUsage) -> list[str]:
    """Which of a role card's pins one model's reading does not match (AU).

    A pinned id (``claude-sonnet-5-5``, ``gpt-6-luna``) must be the model that ran; an alias
    (``sonnet``) must name one of its parts, so an older model under an alias shows only as
    its id. A pinned effort must be every effort the agent ran at; ``inherit`` pins nothing.

    Args:
        pins: The role card's ``role_model`` and ``role_effort`` (`role_pin`).
        usage: One model's reading of the agent's session file.

    Returns:
        ``model`` and ``effort`` where the reading differs from the pin.
    """
    differs = []
    model = pins.get("role_model", "inherit")
    exact = "-" in model or any(char.isdigit() for char in model)
    if model != "inherit" and (
        usage.model != model if exact else model not in usage.model.split("-")
    ):
        differs.append("model")
    effort = pins.get("role_effort", "inherit")
    if effort != "inherit" and usage.efforts and set(usage.efforts) != {effort}:
        differs.append("effort")
    return differs


def usage_events(
    binding: NativeResearchBinding,
    lifecycle: dict[str, Any],
    data: bytes,
    pins: dict[str, str],
) -> list[dict[str, object]]:
    """At a subagent's stop, what it and the lead have run and spent, one event per model.

    Only the session files the hook points at are read, through `native_usage`, and only
    counts, ids, models, efforts and times leave it; a file not admitted gives no event. The
    subagent's readings say where they differ from its role card's pins.

    Args:
        binding: The admitted native session and role scope.
        lifecycle: The admitted SubagentStop hook's lifecycle metadata.
        data: The hook's bounded input bytes, already admitted by the lifecycle read.
        pins: The subagent's role card's pins (`role_pin`).

    Returns:
        The usage events, the subagent's first; none when the hook is not a stop, or when the
        binding turned the reading off.
    """
    if lifecycle["hook_event_name"] != "SubagentStop" or binding.usage == "OFF":
        return []
    files = hook_session_files(
        json.loads(data),
        host=lifecycle["host"],
        session_id=lifecycle["session_id"],
        agent_id=lifecycle["agent_id"],
    )
    events: list[dict[str, object]] = []
    for name, agent_id, role in (
        ("agent", lifecycle["agent_id"], lifecycle["agent_type"]),
        ("lead", binding.session_id, LEAD_ROLE),
    ):
        if name not in files:
            continue
        for row in read_session(files[name], host=lifecycle["host"]).by_model():
            usage = {
                "session_id": lifecycle["session_id"],
                "agent_id": agent_id,
                "role": role,
                "host": lifecycle["host"],
                **asdict(row),
                "pin_differs": pin_differs(pins, row) if name == "agent" else [],
            }
            events.append({"source": "native_usage", "usage": usage})
    return events


def coordination_event(
    binding: NativeResearchBinding,
    *,
    kind: str,
    message: bytes,
    agent_id: str | None = None,
    role: str | None = None,
    message_id: str | None = None,
    reference: str | None = None,
    recipient_id: str | None = None,
    reply_to: str | None = None,
) -> dict[str, object]:
    """Explicitly supplied user-safe text; never read an arbitrary transcript.

    A reply names the message it answers (``reply_to``); an answer, an objection or the
    lead's response naming an assignment closes it in the goal the Host files it under.
    The lead names neither itself nor its role: a message naming no sender is the bound
    session's, as ``research_lead``. A message naming no id is named by its own content, so
    the same message sent again is the same message and a changed text a new one (V420).
    """
    if agent_id is None:
        agent_id = binding.session_id
    if role is None and agent_id == binding.session_id:
        role = LEAD_ROLE
    permitted = (role == LEAD_ROLE and agent_id == binding.session_id and kind in LEAD_KINDS) or (
        role in binding.roles and agent_id != binding.session_id and kind in SPECIALIST_KINDS
    )
    if not permitted:
        raise NativeBridgeError("native_bridge.message_kind_or_role_invalid")
    fields = {
        "agent_id": agent_id,
        "message_id": message_id,
        "reference": reference,
        "recipient_id": recipient_id,
        "reply_to": reply_to,
    }
    for name, value in fields.items():
        if value is not None:
            _carried(name, value)
    if kind == "assignment" and (recipient_id is None or recipient_id == agent_id):
        raise NativeBridgeError("native_bridge.assignment_recipient_required")
    if kind == "assignment" and reply_to is not None:
        raise NativeBridgeError("native_bridge.assignment_is_not_a_reply")
    if not message or len(message) > MAX_MESSAGE_BYTES:
        raise NativeBridgeError("native_bridge.message_size_invalid")
    try:
        body = message.decode("utf-8")
    except UnicodeError:
        raise NativeBridgeError("native_bridge.message_encoding_invalid") from None
    if not body.strip():
        raise NativeBridgeError("native_bridge.message_empty")
    if len(body) > MESSAGE_CHARACTERS:
        raise NativeBridgeError("native_bridge.message_character_limit")
    if len(" ".join(body.split())) > MESSAGE_KEPT_CHARACTERS and reference is None:
        raise NativeBridgeError("native_bridge.long_message_reference_required")
    if message_id is None:
        content = [agent_id, role, kind, body, reference, recipient_id, reply_to]
        message_id = "m-" + _digest(content)[:24]
    return {
        "source": "actor_declared",
        "session_id": binding.session_id,
        "agent_id": agent_id,
        "role": role,
        "kind": kind,
        "message_id": message_id,
        "message": body,
        "message_sha256": sha256(message).hexdigest(),
        "reference": reference,
        "recipient_id": recipient_id,
        "reply_to": reply_to,
    }


def _digest(event: object) -> str:
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def admitted_coordination_event(
    binding: NativeResearchBinding, event: dict[str, Any]
) -> dict[str, object]:
    """Revalidate an explicit message at the Host; retain only the existing selected shape."""
    try:
        selected = coordination_event(
            binding,
            kind=event["kind"],
            message=event["message"].encode("utf-8"),
            agent_id=event.get("agent_id"),
            role=event.get("role"),
            message_id=event.get("message_id"),
            reference=event.get("reference"),
            recipient_id=event.get("recipient_id"),
            reply_to=event.get("reply_to"),
        )
    except (KeyError, TypeError, AttributeError):
        raise NativeBridgeError("native_bridge.event_invalid") from None
    if selected != event:
        raise NativeBridgeError("native_bridge.event_invalid")
    return selected


def _activity_content(
    binding: NativeResearchBinding, event: dict[str, Any]
) -> tuple[str, str, dict[str, str]]:
    """Select the accepted event's fields; source labels never confer trust."""
    if event["source"] == "native_hook":
        native = event["lifecycle"]
        kind = native["hook_event_name"]
        if native["host"] != binding.host:
            raise NativeBridgeError("native_bridge.event_scope_invalid")
        subject = {
            "native_session_id": native["session_id"],
            "native_turn_id": native["turn_id"],
            "native_agent_id": native["agent_id"],
            "role": native["agent_type"],
            "native_host": native["host"],
            "input_channel": CHANNELS[native["host"]],
            "native_hook_event": kind,
            "terminal_state": "NOT_ESTABLISHED",
            "stop_hook_active": (
                "unknown"
                if native["stop_hook_active"] is None
                else str(native["stop_hook_active"]).lower()
            ),
        }
        if native["model"] is not None:
            subject["hook_model"] = native["model"]
        spawn = event.get("thread_spawn")
        if isinstance(spawn, Mapping) and native["host"] == "codex":
            agent_path = canonical_agent_path(spawn.get("agent_path"))
            if (
                agent_path is not None
                and event.get("thread_spawn_bound_session") == binding.session_id
                and isinstance(spawn.get("parent_thread_id"), str)
                and spawn.get("agent_role") == native["agent_type"]
            ):
                subject.update(
                    native_agent_path=agent_path,
                    native_agent_path_basis="CODEX_SESSION_META",
                )
        subject.update(event.get("role_pin") or {})
        if native["session_id"] != binding.session_id or native["agent_type"] not in binding.roles:
            raise NativeBridgeError("native_bridge.event_scope_invalid")
        if kind not in {"SubagentStart", "SubagentStop"}:
            raise NativeBridgeError("native_bridge.event_kind_invalid")
        event_kind = (
            "NATIVE_SUBAGENT_START_HOOK" if kind == "SubagentStart" else "NATIVE_SUBAGENT_STOP_HOOK"
        )
        summary = (
            f"{native['agent_type']}: {kind} hook observed. "
            "The child may continue; this is not proof of agent exit, "
            "Task completion or publication."
        )
    elif event["source"] in {"actor_declared", "product_accepted"}:
        if event["session_id"] != binding.session_id:
            raise NativeBridgeError("native_bridge.event_scope_invalid")
        event_kind = "NATIVE_COORDINATION_MESSAGE"
        summary = event["message"]
        subject = {
            "native_session_id": event["session_id"],
            "native_agent_id": event["agent_id"],
            "role": event["role"],
            "message_kind": event["kind"],
            "message_id": event["message_id"],
            "native_host": binding.host,
        }
        if event["source"] == "product_accepted":
            subject.update(
                input_channel="PRODUCT_ACCEPTED_ANSWER",
                source_time_kind=event["source_time_kind"],
                bundle_reference=event["bundle_reference"],
                answer_reference=event["answer_reference"],
                submitted_by=event["submitted_by"],
                authorship_basis="HOOK",
            )
        else:
            subject.update(
                input_channel="ACTOR_DECLARED",
                message_sha256=event["message_sha256"],
                message_bytes=str(len(summary.encode("utf-8"))),
            )
        for name in ("reference", "recipient_id", "reply_to"):
            if event.get(name) is not None:
                subject[name] = event[name]
    elif event["source"] == "native_usage":
        return _usage_content(binding, event["usage"])
    else:
        raise NativeBridgeError("native_bridge.event_source_invalid")
    subject.setdefault("source_time_kind", "BRIDGE_RECEIVED")
    if any(
        not isinstance(v, str) or not 1 <= len(v) <= SUBJECT_VALUE_CHARACTERS
        for v in subject.values()
    ):
        raise NativeBridgeError("native_bridge.activity_subject_invalid")
    return event_kind, summary, subject


def _usage_content(
    binding: NativeResearchBinding, usage: dict[str, Any]
) -> tuple[str, str, dict[str, str]]:
    """One agent's counts for one model, read so far; no time kind, the counts carry theirs."""
    if usage["session_id"] != binding.session_id or usage["host"] != binding.host:
        raise NativeBridgeError("native_bridge.event_scope_invalid")
    subject = {
        "native_session_id": usage["session_id"],
        "native_agent_id": usage["agent_id"],
        "role": usage["role"],
        "native_host": usage["host"],
        "input_channel": USAGE_CHANNELS[usage["host"]],
        "model": usage["model"],
        **{name: str(usage[name]) for name in _USAGE_COUNTS},
    }
    if usage["efforts"]:
        subject["efforts"] = ",".join(usage["efforts"])
    if usage["pin_differs"]:
        subject["pin_differs"] = ",".join(usage["pin_differs"])
    if usage["last_at"] is not None:
        subject["last_at"] = usage["last_at"]
    if any(
        not isinstance(v, str) or not 1 <= len(v) <= SUBJECT_VALUE_CHARACTERS
        for v in subject.values()
    ):
        raise NativeBridgeError("native_bridge.activity_subject_invalid")
    summary = (
        f"{usage['role']}: {usage['model']}, {usage['responses']} responses, "
        f"{usage['input_tokens']} input, {usage['cache_read_tokens']} cache-read, "
        f"{usage['cache_write_tokens']} cache-write and {usage['output_tokens']} output tokens "
        "so far, from the host's session file."
    )
    return "NATIVE_AGENT_USAGE", summary, subject


def observation_request(
    project: Path, binding: NativeResearchBinding, event: dict[str, Any]
) -> dict[str, object]:
    """Map one selected event, retaining stable producer identity for explicit retry."""
    kind, summary, subject = _activity_content(binding, event)
    if len(binding.session_id) > CORRELATION_CHARACTERS:
        raise NativeBridgeError("native_bridge.activity_correlation_invalid")
    # Sequence reservations belong to this checkout, not to every checkout that
    # observes the same session/workspace. Keep independent senders disjoint.
    scope = _digest([str(project.resolve()), binding.session_id, str(binding.workspace.resolve())])
    # A repeated hook callback has no independent event id in the native input.
    # Deduplicate its declared turn/child/state; do not invent a new host event.
    # A usage event is a reading so far: the same counts read again are the same event.
    identity = (
        [kind, subject["native_agent_id"], subject["native_turn_id"], subject["stop_hook_active"]]
        if event["source"] == "native_hook"
        else [kind, subject["native_agent_id"], subject["message_id"]]
        if event["source"] in {"actor_declared", "product_accepted"}
        else [kind, subject["native_agent_id"], subject["model"]]
        + [subject[name] for name in _USAGE_COUNTS]
        + [subject.get("pin_differs", "")]
    )
    event_id = _digest(identity)
    sequence, occurred_at = reserve_sequence(
        project, scope=scope, event_id=event_id, fingerprint=_digest(event)
    )
    if event["source"] != "product_accepted":
        subject["native_event_id"] = event_id
    return {
        "event_kind": kind,
        "producer_id": PRODUCERS[binding.host],
        "producer_session": scope,
        "producer_sequence": sequence,
        "occurred_at": event["occurred_at"]
        if event["source"] == "product_accepted"
        else occurred_at,
        "summary": summary,
        "subject": subject,
        "correlation_ids": [binding.session_id],
    }


def acknowledged(request: dict[str, object], response: dict[str, Any]) -> bool:
    """Check whether the owner acknowledged the exact producer event.

    Args:
        request: The submitted observation request.
        response: The owner's answer.

    Returns:
        Whether the answer binds the same producer, sequence, and authority.
    """
    return (
        response.get("status") in {"APPENDED", "REUSED_EXACT"}
        and response.get("source_id") == f"{request['producer_id']}:{request['producer_session']}"
        and type(response.get("source_sequence")) is int
        and response["source_sequence"] == request["producer_sequence"]
        and response.get("authority") == "AGENT_PROPOSAL"
        and isinstance(response.get("observation_id"), str)
        and 1 <= len(response["observation_id"]) <= OBSERVATION_ID_CHARACTERS
        and type(response.get("summary_truncated")) is bool
        and not response.get("refused")
        and not response.get("failure_code")
    )


def deliver(
    project: Path,
    binding: NativeResearchBinding,
    event: dict[str, object],
    *,
    host_owned: bool = False,
) -> dict[str, object]:
    """Send one scoped observation through the product's local client.

    Args:
        project: The project holding the producer sequence.
        binding: The admitted native session and workspace.
        event: The bounded event to submit.
        host_owned: The explicit message door asks the Host to reserve its sequence.

    Returns:
        A delivered receipt or a safe unavailable status.
    """
    # Reuse the sole token/loopback/instance/redirect-safe product transport.
    # Keep import off the unbound hook's fast path.
    from alphalattice.interface.local_application.client import (
        LocalResearchClient,
        LocalResearchClientError,
    )

    try:
        client = LocalResearchClient(binding.workspace, timeout=2.0)
        if host_owned:
            return cast(dict[str, object], client.publish_native_event(project, event))
        publish = getattr(client, "publish_event", None)
        if not callable(publish):
            return {"status": "UNAVAILABLE", "reason": "native_bridge.activity_client_unavailable"}
        return deliver_owned(project, binding, event, publish=publish)
    except LocalResearchClientError:
        return {"status": "UNAVAILABLE", "reason": "native_bridge.transport_unavailable"}


def deliver_owned(
    project: Path,
    binding: NativeResearchBinding,
    event: dict[str, Any],
    *,
    publish: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, object]:
    """Reserve and publish one event through the existing sequence and observation owners."""
    from alphalattice.interface.local_application.client import LocalResearchClientError

    try:
        with WorkspaceLock(metadata_path(project, LOCK_NAME)):
            request = observation_request(project, binding, event)
            response = publish(request)
    except WorkspaceConflictError:
        return {"status": "UNAVAILABLE", "reason": "native_bridge.observation_busy"}
    except NativeSequenceError as error:
        return {"status": "UNAVAILABLE", "reason": str(error)}
    except LocalResearchClientError:
        return {"status": "UNAVAILABLE", "reason": "native_bridge.transport_unavailable"}
    code = safe_failure_code(response.get("failure_code"))
    if response.get("status") == "REFUSED" and code is not None:
        return {
            "status": "REFUSED",
            "reason": code,
            **{
                key: response[key]
                for key in ("detail", "next_action")
                if isinstance(response.get(key), str)
            },
        }
    if not acknowledged(request, response):
        return {"status": "UNAVAILABLE", "reason": "native_bridge.product_contract_unavailable"}
    return {
        "status": "DELIVERED",
        "observation_id": response["observation_id"],
        "source_id": response["source_id"],
        "source_sequence": response["source_sequence"],
        "authority": response["authority"],
        "summary_truncated": response.get("summary_truncated"),
        # The goal the Host filed the event under, and an assignment's packet (GR2).
        **{key: response[key] for key in ("goal_id", "packet_hash") if response.get(key)},
    }


def deliver_accepted_answer(
    project: Path,
    binding: NativeResearchBinding,
    *,
    bundle: AgentBundleRecord,
    answer: AgentAnswerRecord,
    contribution: Mapping[str, object],
    parent: Mapping[str, object],
    task_id: str,
    admitted_at: datetime,
    events: Iterable[Mapping[str, Any]],
    publish: Callable[[dict[str, Any]], dict[str, Any]],
    before_publish: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, object] | None:
    """Publish one accepted deliverable without a role-specific answer interpretation.

    The caller supplies its sealed generic answer, screened accepted contribution and
    immutable Task admission. The saved author must still have exact HOOK/assignment
    proof for this bundle and original parent. This is a product record relayed by the
    parent, never a native spoken transcript. It uses the existing producer sequence,
    observation and Goal owners; optional observation failure preserves acceptance.
    """
    run = answer.agent_run
    if answer.verdict not in {AnswerVerdict.ACCEPTED, AnswerVerdict.DONE} or run is None:
        return None
    if run.basis != "HOOK" or run.agent_id is None or run.role is None:
        return None
    unavailable: dict[str, object] = {
        "status": "UNAVAILABLE",
        "reason": "native_bridge.accepted_answer_not_recorded",
    }
    try:
        if (parent.get("vendor"), parent.get("session")) != (run.host, run.session_id) or (
            binding.host,
            binding.session_id,
        ) != (run.host, run.session_id):
            return unavailable
        if run.role not in binding.roles or not binding.workspace.resolve().is_relative_to(
            project.resolve()
        ):
            return unavailable
        history = list(events)
        proved = judgment_agent(
            history,
            host=run.host,
            session_id=run.session_id,
            bundle_role=bundle.role,
            bundle_reference=bundle.record_hash,
        )
        if (proved.get("basis"), proved.get("agent_id"), proved.get("role")) != (
            "HOOK",
            run.agent_id,
            run.role,
        ):
            return unavailable
        starts = [
            item
            for item in history
            if (item.get("payload") or {}).get("event_kind") == "NATIVE_SUBAGENT_START_HOOK"
        ]
        assignments = {
            str(subject["message_id"])
            for item in history
            if isinstance(subject := (item.get("payload") or {}).get("subject"), dict)
            and subject.get("message_kind") == "assignment"
            and subject.get("message_id")
            and judgment_agent(
                [*starts, item],
                host=run.host,
                session_id=run.session_id,
                bundle_role=bundle.role,
                bundle_reference=bundle.record_hash,
            ).get("agent_id")
            == run.agent_id
        }
        parts: list[str] = []

        def fields(value: object, path: str) -> None:
            if isinstance(value, Mapping):
                for name, child in value.items():
                    fields(child, f"{path}.{name}" if path else str(name))
            elif isinstance(value, (list, tuple)):
                for number, child in enumerate(value):
                    fields(child, f"{path}[{number}]")
            elif isinstance(value, str) and value.strip():
                parts.append(f"{path}: {value}")

        fields(contribution, "")
        preview = " ".join("; ".join(parts).split())
        drops = len(answer.problems) if answer.verdict is AnswerVerdict.DONE else 0
        dropped = f"; {drops} dropped" if drops else ""
        summary = f"Product accepted deliverable ({answer.verdict}{dropped}): {preview}"
        if len(summary) > MESSAGE_KEPT_CHARACTERS:
            summary = summary[: MESSAGE_KEPT_CHARACTERS - 1] + "…"
        identity = _digest([bundle.record_hash, answer.record_hash])
        event: dict[str, Any] = {
            "source": "product_accepted",
            "session_id": run.session_id,
            "agent_id": run.agent_id,
            "role": run.role,
            "kind": "answer",
            "message_id": "accepted-" + identity[:24],
            "message": summary,
            "reference": task_id,
            "bundle_reference": bundle.record_hash,
            "answer_reference": answer.record_hash,
            "submitted_by": run.session_id,
            "recipient_id": run.session_id,
            "occurred_at": (answer.accepted_at or admitted_at).isoformat(),
            "source_time_kind": "PRODUCT_ACCEPTED_AT"
            if answer.accepted_at is not None
            else "TASK_ADMISSION",
        }
        if len(assignments) == 1:
            event["reply_to"] = next(iter(assignments))

        def file(document: dict[str, Any]) -> dict[str, Any]:
            if before_publish is not None:
                before_publish(document)
            return publish(document)

        return deliver_owned(project, binding, event, publish=file)
    except Exception:
        return unavailable


def lead_readings(project: Path, environ: Mapping[str, str]) -> list[dict[str, object]]:
    """The lead's readings, delivered where its own command asks (V301).

    A session with no subagent had no reading, since the bridge read the hosts' session files
    only at a subagent's stop. A goal's take and submission and an agent's answer are rare, so
    the client reads the lead there, through this bridge and under its binding: for the bound
    session only, never where the binding turned readings off, and never in the request's way.

    Args:
        project: The project whose binding names the session.
        environ: The command's environment, naming the session it runs in.

    Returns:
        Each delivery's answer, one per model; none when nothing was read.
    """
    binding = NativeResearchBinding.read(project)
    if binding is None or binding.usage == "OFF":
        return []
    # The session as every reader of it reads it (V583): an agent inside another names none.
    if agent_session(environ) != (binding.host, binding.session_id):
        return []
    path = session_file(binding.host, binding.session_id)
    if path is None:
        return []
    answers = []
    for row in read_session(path, host=binding.host).by_model():
        usage = {
            "session_id": binding.session_id,
            "agent_id": binding.session_id,
            "role": LEAD_ROLE,
            "host": binding.host,
            **asdict(row),
            "pin_differs": [],
        }
        answers.append(deliver(project, binding, {"source": "native_usage", "usage": usage}))
    return answers


JUDGMENT_ROLES = {
    "ALPHA": ("alphalattice_alpha",),
    "ANALYST": ("alphalattice_evidence_analyst", "alternative_analyst"),
    "CRO": ("alphalattice_cro", "independent_cro"),
    "DATA": ("alphalattice_data",),
    "FACTOR": ("alphalattice_factor",),
    "PORTFOLIO": ("alphalattice_portfolio",),
    "RISK": ("alphalattice_risk",),
}
"""The role cards an answer to each bundle role may come from; a card's variants (``_medium``)
share its role."""


def _proved_agent_path(subject: Mapping[str, Any], *, host: str, session_id: str) -> str | None:
    """Read a hook's selected first-record path admitted through the bound ancestry."""
    if (
        host != "codex"
        or subject.get("native_agent_path_basis") != "CODEX_SESSION_META"
        or subject.get("native_session_id") != session_id
        or subject.get("native_host") != host
        # Older hooks carried the selected header role as well. A conflicting
        # retained fact cannot confer a path alias, even after payload compaction.
        or ("native_spawn_role" in subject and subject["native_spawn_role"] != subject.get("role"))
    ):
        return None
    return canonical_agent_path(subject.get("native_agent_path"))


def judgment_agent(
    events: Iterable[Mapping[str, Any]],
    *,
    host: str,
    session_id: str,
    bundle_role: str,
    bundle_reference: str | None,
) -> dict[str, object]:
    """Who made an answer to one bundle, as the session's own records settle it (AU3, V555).

    A request names its session, never its subagent, so the author is settled by a link: the
    session's lead assigned the bundle to one specialist -- an assignment message whose
    reference is the bundle's own, the key its prepare answer gave (``bundle_reference``, V574),
    exactly -- and that specialist started in the session under a card of the bundle's role.
    An assignment may name its native id or a unique canonical path that the new hook recorded
    from its admitted first-record parent and role. Ambiguous paths settle no author; an exact
    native-id assignment remains valid. The returned agent id is always the hook's native id,
    preserving the existing ``AgentRun`` contract.
    Its model is its start hook's where the host named one, else its card's pin. Anything less
    settles no author: no assignment of the bundle, assignments naming more than one
    specialist, a recipient the session never started under the bundle's role, or a session
    event whose content retention removed, as the record is then no longer whole (V570). The
    author is then unknown -- the host and the session alone -- never another specialist, never
    the lead, and never a guess by time. Nothing here opens a file: it reads the hook events and
    messages this bridge delivered.

    Args:
        events: The Host's external activity items, every one it holds, in any order.
        host: The session's host.
        session_id: The session the answer's request named.
        bundle_role: The answered bundle's role, ``ANALYST`` or ``CRO``.
        bundle_reference: The answered bundle's key, its record's hash: a 64-hex reference
            any message can carry, where its directory may be longer than one can.

    Returns:
        The fields of an ``AgentRun``.
    """
    cards = JUDGMENT_ROLES.get(bundle_role, ())
    run: dict[str, object] = {"host": host, "session_id": session_id}
    started: dict[str, Mapping[str, Any]] = {}
    assigned: set[str] = set()
    path_agents: dict[str, set[str]] = {}
    agent_paths: dict[str, set[str]] = {}
    for item in sorted(events, key=lambda value: int(value.get("ordinal") or 0)):
        if item.get("payload") is None and session_id in (item.get("correlation_ids") or ()):
            # Retention keeps a session's newest events once the store is full: the one it
            # emptied may be the assignment, or a second one that unsettles it.
            return {**run, "basis": "NOT_OBSERVED"}
        payload = item.get("payload") or {}
        subject = payload.get("subject") or {}
        if subject.get("native_session_id") != session_id or subject.get("native_host") != host:
            continue
        kind, role = payload.get("event_kind"), str(subject.get("role", ""))
        if kind == "NATIVE_SUBAGENT_START_HOOK":
            agent_id = str(subject.get("native_agent_id", ""))
            agent_path = _proved_agent_path(subject, host=host, session_id=session_id)
            if agent_id and agent_path is not None:
                path_agents.setdefault(agent_path, set()).add(agent_id)
                agent_paths.setdefault(agent_id, set()).add(agent_path)
            if any(role == card or role.startswith(f"{card}_") for card in cards):
                started[agent_id] = subject
        elif (
            kind == "NATIVE_COORDINATION_MESSAGE"
            and subject.get("message_kind") == "assignment"
            and subject.get("native_agent_id") == session_id
            and bundle_reference is not None
            and subject.get("reference") == bundle_reference
        ):
            assigned.add(str(subject.get("recipient_id", "")))
    resolved: set[str] = set()
    for recipient in assigned:
        if canonical_agent_path(recipient) is not None:
            candidates = path_agents.get(recipient, set())
            if len(candidates) != 1:
                return {**run, "basis": "NOT_OBSERVED"}
            candidate = next(iter(candidates))
            if agent_paths.get(candidate) != {recipient}:
                return {**run, "basis": "NOT_OBSERVED"}
        else:
            candidate = recipient
        if candidate not in started:
            return {**run, "basis": "NOT_OBSERVED"}
        resolved.add(candidate)
    agent = next(iter(resolved)) if len(resolved) == 1 else None
    subject = None if agent is None else started.get(agent)
    if subject is None:
        return {**run, "basis": "NOT_OBSERVED"}
    pinned = subject.get("role_effort")
    run |= {
        "agent_id": agent,
        "role": subject.get("role"),
        "efforts": [pinned] if pinned not in (None, "inherit") else [],
    }
    if subject.get("hook_model"):
        return {**run, "model": subject["hook_model"], "basis": "HOOK"}
    if subject.get("role_model") not in (None, "inherit"):
        return {**run, "model": subject["role_model"], "basis": "ROLE_CARD"}
    return {**run, "basis": "NOT_OBSERVED"}


def handle_hook(project: Path, data: bytes) -> dict[str, object]:
    """One bounded attempt, no disk queue/retry and no research side effect."""
    binding = NativeResearchBinding.read(project)
    if binding is None:
        return {"status": "IGNORED", "reason": "native_bridge.not_bound"}
    event = lifecycle_event(project, binding, data)
    if event is None:
        return {"status": "IGNORED", "reason": "native_bridge.event_not_selected"}
    result = deliver(project, binding, event)
    if result["status"] != "DELIVERED":
        return result
    lifecycle = cast(dict[str, Any], event["lifecycle"])
    for usage in usage_events(binding, lifecycle, data, cast(dict[str, str], event["role_pin"])):
        answer = deliver(project, binding, usage)
        if answer["status"] != "DELIVERED":
            return answer
    return result


def hook_reply(project: Path, data: bytes) -> dict[str, str]:
    """Advisory JSON only: never continue:false, decision:block, or exit2."""
    try:
        result = handle_hook(project, data)
        if result["status"] in {"IGNORED", "DELIVERED"}:
            return {}
        reason = result["reason"]
    except (NativeBridgeError, NativeHookInputError) as error:
        reason = str(error)
    except Exception:
        reason = "native_bridge.internal_failure"
    return {
        "systemMessage": f"AlphaLattice observation unavailable ({reason}); do not rerun research."
    }
