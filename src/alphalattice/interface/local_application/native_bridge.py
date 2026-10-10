"""Project-scoped native Session bindings and Host-read observations; never a research executor.

A binding names the agent Session a workspace serves. The Host reads that Session's usage, and
its children's, from the host's own session files at the moments research reaches -- a goal's
take or submission, an answer's submission, a Goal or Team page opening -- through
`native_usage` (AU, V300; LAWS OP13, OP14), unless the binding turned reading off
(``"usage": "OFF"``). Each reading and each accepted answer is filed through the Host's own
activity owner. No hook, message or always-running reader is involved.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import UUID

from alphalattice.control.workspace_runtime.content_store import replace_shared_file
from alphalattice.interface.local_application.cli_contract import join, refusal_words, shell
from alphalattice.interface.local_application.failure_codes import (
    owner_failure_code,
    public_failure,
    safe_failure_code,
)
from alphalattice.interface.local_application.native_usage import (
    ModelUsage,
    NativeUsageReadLimitError,
    SessionUsage,
    claude_children,
    claude_thread_spawn,
    codex_thread_spawn,
    read_session,
    session_file,
)
from alphalattice.protocols.actor_execution.answers import AgentAnswerRecord, AnswerVerdict
from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord


def command_readiness(command: list[str], *, timeout: int = 30) -> dict[str, Any]:
    """Probe a dependency's public command without installing or changing anything."""
    executable = shutil.which(command[0])
    result: dict[str, Any] = {"present": False, "command": executable, "reason": None}
    if executable is None:
        return {**result, "reason": "COMMAND_MISSING"}
    try:
        completed = subprocess.run(
            [executable, *command[1:]], capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return {**result, "reason": "COMMAND_TIMED_OUT"}
    except OSError:
        return {**result, "reason": "COMMAND_START_FAILED"}
    result.update(
        present=completed.returncode == 0,
        reason=None if completed.returncode == 0 else "COMMAND_FAILED",
    )
    if result["present"] and command[1:] == ["--version"]:
        output = getattr(completed, "stdout", None) or b""
        text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
        version = re.search(r"(?<![\w.])(\d+\.\d+\.\d+(?:[-+][\w.-]+)?)(?![\w.])", text)
        result["version"] = None if version is None else version[1]
    return result


def codex_command(path: str | None = None) -> tuple[str | None, str]:
    """An admitted client executable, otherwise this process's PATH."""
    offered = None if not path else Path(path)
    if (
        offered is not None
        and offered.is_absolute()
        and offered.is_file()
        and offered.name.lower() in {"codex", "codex.exe", "codex.cmd"}
    ):
        return str(offered), "CLIENT_PATH"
    found = shutil.which("codex")
    return (None if found is None else str(Path(found).absolute())), "HOST_PATH"


def codex_queue_readiness(path: str | None = None) -> dict[str, Any]:
    """Whether this process can start the background queue the Host must deliver through."""
    command, source = codex_command(path)
    return {
        **command_readiness([command or "codex", "queue", "--help"], timeout=5),
        "command_source": source,
    }


BINDING_NAME = "native-research.local.json"
BINDING_DIRECTORY = "native-research.local.d"
BINDING_RECORDS = 128
"""The bounded collection of exact host/Session bindings in one configured project."""
# Every bound the bridge applies is the contract of what it carries, so it never refuses what
# that contract admits nor passes what the Host then refuses under another word; the table
# beside its tests pins each against its contract (V574).
SUBJECT_VALUE_CHARACTERS = 200
"""The longest value the Host's activity event holds in its subject: each id, role, locator
and reference the bridge carries is bounded by it where it enters."""
CORRELATION_CHARACTERS = 128
"""The longest correlation id the event holds: the bound session's id is one."""
MESSAGE_KEPT_CHARACTERS = 500
"""What the Host keeps of an accepted deliverable's preview, its whitespace folded."""
TEXT_CHARACTERS = 512
"""The longest field a binding holds that is not carried: its host, reading and workspace."""
BINDING_BYTES = 8192
"""The longest binding file: the widest binding `session bind` writes fits it."""
BINDING_ROLES = 32
"""The most roles a binding names: the project's cards, each named by ``_ROLE_NAME``."""
PIN_CHARACTERS = 64
"""The longest model or effort a role card pins that the bridge carries."""
SPAWN_HOPS = 4
"""The most threads a Codex specialist's spawn chain climbs to reach the bound session: a
specialist's specialist and two more (V568)."""
# One producer identity per foreground host; the Team page shows the host as source
# information, never as authentication (the product philosophy, 2026-09-22).
HOSTS = ("codex", "claude-code")
USAGE_READINGS = ("READ", "OFF")
"""Whether the Host reads this Session's usage from the host's session files (AU): the
binding's ``usage``, ``READ`` when it names none."""
USAGE_READ_BYTES = 512 * 1024 * 1024
"""The complete optional scan's 512 MiB cap leaves room beyond a measured 112 MiB session."""
USAGE_LINE_BYTES = 8 * 1024 * 1024
"""The optional scan's 8 MiB line cap admits a measured valid native line larger than 2 MiB."""
USAGE_MODEL_ROWS = 32
"""The most model rows a complete optional reading publishes; overflow publishes no rows."""
PRODUCERS = {"codex": "codex-native", "claude-code": "claude-code-native"}
USAGE_CHANNELS = {"codex": "CODEX_SESSION_FILE", "claude-code": "CLAUDE_CODE_SESSION_FILE"}
LEAD_ROLE = "research_lead"
PRODUCT_ROLE_PREFIX = "alphalattice_"
"""Every shipped card's name begins so; a child of any other role is its lead's helper."""
_USAGE_COUNTS = (
    "responses",
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
)
_ROLE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# Written by configure beside the host settings; the one mark of a product project.
PROJECT_DECLARATION_NAME = "alphalattice-project.local.json"
PROJECT_DECLARATION_SCHEMA = "alphalattice.native-project.v1"


class NativeBridgeError(ValueError):
    """Only code-owned error text is allowed across the bridge boundary."""


def declares_product(folder: Path, host: str) -> bool:
    """Read the bounded configure-owned project declaration, never hook configuration."""
    declaration = folder / (
        ".claude/settings.json" if host == "claude-code" else ".codex/config.toml"
    )
    marker = declaration.parent / PROJECT_DECLARATION_NAME
    if any(
        path.is_symlink() or (os.name == "nt" and path.is_junction())
        for path in (folder, declaration.parent, declaration, marker)
    ):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    try:
        with marker.open("rb") as stream:
            data = stream.read(64 * 1024 + 1)
    except FileNotFoundError:
        return False
    if len(data) > 64 * 1024:
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    try:
        document = json.loads(data)
    except (ValueError, UnicodeError):
        raise NativeBridgeError("native_bridge.configuration_path_invalid") from None
    if document != {"schema": PROJECT_DECLARATION_SCHEMA, "host": host}:
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    return True


def session_project(start: Path, host: str | None) -> Path:
    """The agent project a session binds in, found up from where the binding runs (V568).

    It is the nearest folder with this host's explicit configure-owned project declaration.
    Ordinary host settings and hook matchers do not identify a product project.

    Args:
        start: The directory the binding command runs in.
        host: ``codex`` or ``claude-code``; None, outside any agent session, for either.

    Returns:
        The project's folder.

    Raises:
        NativeBridgeError: ``native_bridge.project_declaration_missing`` when no parent declares it.
    """
    hosts = HOSTS if host is None else (host,)
    for folder in (start, *start.parents):
        if any(declares_product(folder, each) for each in hosts):
            return folder
    raise NativeBridgeError("native_bridge.project_declaration_missing")


def _text(value: object, bound: int = TEXT_CHARACTERS) -> str:
    if not isinstance(value, str) or not value or len(value) > bound:
        raise NativeBridgeError("native_bridge.binding_invalid")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise NativeBridgeError("native_bridge.binding_invalid")
    return value


@dataclass(frozen=True, slots=True)
class NativeResearchBinding:
    """Admitted native session, workspace, roles, and producer host."""

    session_id: str
    workspace: Path
    roles: tuple[str, ...]
    host: str = "codex"
    usage: str = "READ"
    observation_started_at: datetime | None = None

    @classmethod
    def slot_path(cls, project: Path, session: tuple[str, str]) -> Path:
        """The exact owner slot; a host/Session is never a path supplied by a caller."""
        host, session_id = session
        if host not in HOSTS:
            raise NativeBridgeError("native_bridge.binding_invalid")
        _text(session_id, CORRELATION_CHARACTERS)
        key = sha256(f"{host}\0{session_id}".encode()).hexdigest()
        return project / ".codex" / BINDING_DIRECTORY / f"{key}.json"

    @classmethod
    def _read_path(cls, project: Path, path: Path) -> NativeResearchBinding | None:
        """Read one bounded regular owner record, rejecting links and aliases."""
        try:
            parents = (
                path,
                *(
                    parent
                    for parent in path.parents
                    if parent == project or parent.is_relative_to(project)
                ),
            )
            if any(
                parent.is_symlink() or (os.name == "nt" and parent.is_junction())
                for parent in parents
            ) or not path.resolve().is_relative_to(project.resolve()):
                raise NativeBridgeError("native_bridge.binding_path_invalid")
            if not stat.S_ISREG(path.stat().st_mode):
                raise NativeBridgeError("native_bridge.binding_path_invalid")
            with path.open("rb") as stream:
                data = stream.read(BINDING_BYTES + 1)
        except FileNotFoundError:
            return None
        except (OSError, RuntimeError):
            raise NativeBridgeError("native_bridge.binding_unreadable") from None
        if len(data) > BINDING_BYTES:
            raise NativeBridgeError("native_bridge.binding_invalid")
        try:
            value = json.loads(data)
        except (ValueError, RecursionError):
            raise NativeBridgeError("native_bridge.binding_invalid") from None
        binding = cls.from_document(value)
        if path.name != BINDING_NAME and path != cls.slot_path(
            project, (binding.host, binding.session_id)
        ):
            raise NativeBridgeError("native_bridge.binding_invalid")
        return binding

    @classmethod
    def binding_entries(
        cls, project: Path
    ) -> tuple[tuple[NativeResearchBinding, ...], tuple[dict[str, str], ...]]:
        """Enumerate owner records without making one bad item erase readable bindings.

        Diagnostics retain the bounded record key and typed code, never a guessed Session.
        Native session files are not discovered or read by this collection.
        """
        paths = [project / ".codex" / BINDING_NAME]
        directory = project / ".codex" / BINDING_DIRECTORY
        if directory.is_symlink() or (os.name == "nt" and directory.is_junction()):
            raise NativeBridgeError("native_bridge.binding_path_invalid")
        if directory.exists():
            if (
                not directory.is_dir()
                or directory.parent.is_symlink()
                or (os.name == "nt" and directory.parent.is_junction())
            ):
                raise NativeBridgeError("native_bridge.binding_path_invalid")
            try:
                children = list(islice(directory.iterdir(), BINDING_RECORDS + 1))
            except OSError:
                raise NativeBridgeError("native_bridge.binding_unreadable") from None
            if len(children) > BINDING_RECORDS:
                raise NativeBridgeError("native_bridge.binding_limit_exceeded")
            paths.extend(sorted(children))
        values: dict[tuple[str, str], NativeResearchBinding] = {}
        refusals: list[dict[str, str]] = []
        for path in paths:
            try:
                if (
                    path.name != BINDING_NAME
                    and re.fullmatch(r"[0-9a-f]{64}\.json", path.name) is None
                ):
                    raise NativeBridgeError("native_bridge.binding_path_invalid")
                binding = cls._read_path(project, path)
                if binding is None:
                    continue
                key = (binding.host, binding.session_id)
                if key in values:
                    raise NativeBridgeError("native_bridge.binding_invalid")
                values[key] = binding
            except NativeBridgeError as error:
                refusals.append(
                    {
                        "binding_key": path.name[:72],
                        "failure_code": public_failure(error, "native_bridge.binding_invalid"),
                    }
                )
        if len(values) > BINDING_RECORDS:
            raise NativeBridgeError("native_bridge.binding_limit_exceeded")
        return tuple(values.values()), tuple(refusals)

    @classmethod
    def bindings(cls, project: Path) -> tuple[NativeResearchBinding, ...]:
        """The strict bounded collection; readers needing per-item refusals use entries."""
        bindings, refusals = cls.binding_entries(project)
        if refusals:
            raise NativeBridgeError(refusals[0]["failure_code"])
        return bindings

    @classmethod
    def read(
        cls, project: Path, *, session: tuple[str, str] | None = None
    ) -> NativeResearchBinding | None:
        """Read an exact host/Session; an omitted selector admits only one record.

        Args:
            project: The admitted project root.
            session: The real host/Session selector; no newest or first selection.

        Returns:
            The binding, or ``None`` when no binding file exists.

        Raises:
            NativeBridgeError: For an unsafe record or an ambiguous omitted selector.
        """
        if session is None:
            bindings = cls.bindings(project)
            if len(bindings) > 1:
                raise NativeBridgeError("native_bridge.binding_ambiguous")
            return bindings[0] if bindings else None
        binding = cls._read_path(project, cls.slot_path(project, session))
        try:
            legacy = cls._read_path(project, project / ".codex" / BINDING_NAME)
        except NativeBridgeError:
            if binding is None:
                raise
            legacy = None
        if legacy is not None and (legacy.host, legacy.session_id) == session:
            if binding is not None:
                raise NativeBridgeError("native_bridge.binding_invalid")
            return legacy
        return binding

    def record_path(self, project: Path) -> Path:
        """The selected existing record; compatibility records are never migrated silently."""
        slot_path = self.slot_path(project, (self.host, self.session_id))
        if self._read_path(project, slot_path) == self:
            return slot_path
        legacy_path = project / ".codex" / BINDING_NAME
        if self._read_path(project, legacy_path) != self:
            raise NativeBridgeError("native_bridge.binding_invalid")
        return legacy_path

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
            if set(value) - {"host", "usage", "observation_started_at"} != {
                "session_id",
                "workspace",
                "roles",
            }:
                raise ValueError
            host = _text(value.get("host", "codex"))
            if host not in HOSTS:
                raise ValueError
            usage = _text(value.get("usage", "READ"))
            if usage not in USAGE_READINGS:
                raise ValueError
            checkpoint = value.get("observation_started_at")
            observed_at = None if checkpoint is None else datetime.fromisoformat(_text(checkpoint))
            if observed_at is not None and observed_at.utcoffset() is None:
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
            observation_started_at=observed_at,
        )

    @classmethod
    def find(
        cls, start: Path, *, session: tuple[str, str] | None = None
    ) -> tuple[Path, NativeResearchBinding] | None:
        """The nearest binding from a directory upward, as git finds its `.git` (V568).

        A checkout is one such project; an installed agent project is another.

        Args:
            start: The directory a command runs in.
            session: The actual host/Session; exact binding precedes bounded ancestry.

        Returns:
            The project holding the nearest binding, with it; None when no folder up holds one.

        Raises:
            NativeBridgeError: If the nearest binding is unsafe, unreadable or invalid.
        """
        # The configure-owned declaration bounds the project; legacy records predate it.
        configured = None
        try:
            configured = session_project(start, None)
        except NativeBridgeError as error:
            if str(error) != "native_bridge.project_declaration_missing":
                raise
        if configured is not None and session is not None:
            try:
                host_project = session_project(start, session[0])
            except NativeBridgeError as error:
                if str(error) != "native_bridge.project_declaration_missing":
                    raise
                raise NativeBridgeError("native_bridge.project_mismatch") from None
            if host_project != configured:
                raise NativeBridgeError("native_bridge.project_mismatch")
        folders = (configured,) if configured is not None else (start, *start.parents)
        for folder in folders:
            path = folder / ".codex" / BINDING_NAME
            directory = folder / ".codex" / BINDING_DIRECTORY
            if path.is_symlink() or path.exists() or directory.is_symlink() or directory.exists():
                binding = cls.read(folder, session=session)
                if binding is not None:
                    return folder, binding
                if session is not None:
                    bindings = cls.bindings(folder)
                    # OFF is a privacy choice, not permission to discover this child's
                    # parent through another Session that happens to allow reading.
                    if any(each.usage == "OFF" for each in bindings):
                        return None
                    serving = [each for each in bindings if each.serves(session)]
                    if len(serving) > 1:
                        raise NativeBridgeError("native_bridge.binding_ambiguous")
                    if serving:
                        return folder, serving[0]
                    return None
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
        if vendor != "codex" or self.usage == "OFF":
            return False
        for _hop in range(SPAWN_HOPS):
            spawn = codex_thread_spawn(thread)
            if spawn is None:
                return False
            if spawn.parent_thread_id == self.session_id:
                return True
            thread = spawn.parent_thread_id
        return False


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


def _digest(event: object) -> str:
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _subject_bounded(subject: dict[str, str]) -> dict[str, str]:
    if any(
        not isinstance(v, str) or not 1 <= len(v) <= SUBJECT_VALUE_CHARACTERS
        for v in subject.values()
    ):
        raise NativeBridgeError("native_bridge.activity_subject_invalid")
    return subject


def _usage_content(
    binding: NativeResearchBinding, usage: dict[str, Any]
) -> tuple[str, str, dict[str, str]]:
    """One agent's counts for one model, with their actual source and latest record time."""
    if usage["session_id"] != binding.session_id or usage["host"] != binding.host:
        raise NativeBridgeError("native_bridge.event_scope_invalid")
    subject = {
        "native_session_id": usage["session_id"],
        "native_agent_id": usage["agent_id"],
        "role": usage["role"],
        "native_host": usage["host"],
        "input_channel": USAGE_CHANNELS[usage["host"]],
        "sample_time_kind": "LATEST_USAGE_RECORD_AT"
        if usage["last_at"] is not None
        else "NOT_OBSERVED",
        "model": usage["model"],
        **{name: str(usage[name]) for name in _USAGE_COUNTS},
    }
    if usage["efforts"]:
        subject["efforts"] = ",".join(usage["efforts"])
    if usage["pin_differs"]:
        subject["pin_differs"] = ",".join(usage["pin_differs"])
    if usage["last_at"] is not None:
        subject["last_at"] = usage["last_at"]
    summary = (
        f"{usage['role']}: {usage['model']}, {usage['responses']} responses, "
        f"{usage['input_tokens']} input, {usage['cache_read_tokens']} cache-read, "
        f"{usage['cache_write_tokens']} cache-write and {usage['output_tokens']} output tokens "
        "so far, from the host's session file."
    )
    return "NATIVE_AGENT_USAGE", summary, _subject_bounded(subject)


def _accepted_content(
    binding: NativeResearchBinding, event: dict[str, Any]
) -> tuple[str, str, dict[str, str]]:
    """A product-accepted deliverable, filed under the lead's Session as a product fact."""
    if event["session_id"] != binding.session_id:
        raise NativeBridgeError("native_bridge.event_scope_invalid")
    subject = {
        "native_session_id": event["session_id"],
        "native_agent_id": event["agent_id"],
        "role": event["role"],
        "message_kind": "answer",
        "message_id": event["message_id"],
        "native_host": binding.host,
        "input_channel": "PRODUCT_ACCEPTED_ANSWER",
        "source_time_kind": event["source_time_kind"],
        "bundle_reference": event["bundle_reference"],
        "answer_reference": event["answer_reference"],
        "submitted_by": event["submitted_by"],
        "authorship_basis": "NOT_OBSERVED",
        "bundle_role": event["bundle_role"],
        "reference": event["reference"],
        "recipient_id": event["recipient_id"],
    }
    return "NATIVE_COORDINATION_MESSAGE", event["message"], _subject_bounded(subject)


def _aware(value: object) -> str | None:
    """A recorded time the activity contract accepts, as written; None for any other."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return str(value) if parsed.utcoffset() is not None else None


def producer_scope(project: Path, binding: NativeResearchBinding) -> str:
    """Read this exact project's native producer scope, shared by writers and readers.

    Args:
        project: The admitted project whose observations are filed.
        binding: The exact native parent and served workspace.

    Returns:
        The existing stable producer scope without changing observation identity.
    """
    return _digest([str(project.resolve()), binding.session_id, str(binding.workspace.resolve())])


def observation_request(
    project: Path,
    binding: NativeResearchBinding,
    event: dict[str, Any],
    *,
    usage_goal_id: str | None = None,
) -> dict[str, object]:
    """Map one Host-read event to the activity contract, named by its own content.

    The sequence is the event's digest, so the same snapshot or accepted answer filed again
    is the same observation and nothing needs a stored counter. ``usage_goal_id`` is the
    Goal resolved by the Host, never a native file's claim.
    """
    if event["source"] == "product_accepted":
        kind, summary, subject = _accepted_content(binding, event)
        identity: list[object] = [kind, subject["native_agent_id"], subject["message_id"]]
        occurred_at = str(event["occurred_at"])
    elif event["source"] == "product_bound":
        goal_hash = event.get("goal_hash")
        kind, summary = MESSAGE_EVENT_KIND, BOUND_WORDS if goal_hash is None else GOAL_HELD_WORDS
        subject = _subject_bounded(
            {
                "native_session_id": binding.session_id,
                "native_agent_id": binding.session_id,
                "role": LEAD_ROLE,
                "message_kind": "session_bound",
                "message_id": "bound-"
                + _digest(
                    [
                        binding.host,
                        binding.session_id,
                        str(binding.workspace),
                        *([] if goal_hash is None else [goal_hash]),
                    ]
                )[:24],
                "native_host": binding.host,
                "input_channel": "PRODUCT_OPERATION",
                "source_time_kind": "BOUND_AT",
                # The goal the Session holds, as Team reads a lead's research question.
                **({} if goal_hash is None else {"reference": f"case:{goal_hash}"}),
            }
        )
        identity = [kind, subject["native_agent_id"], subject["message_id"]]
        occurred_at = str(event["occurred_at"])
    elif event["source"] == "native_usage":
        kind, summary, subject = _usage_content(binding, event["usage"])
        identity = [
            kind,
            event["usage"],
            None if usage_goal_id is None else str(UUID(usage_goal_id)),
        ]
        occurred_at = _aware(event["usage"]["last_at"]) or datetime.now(UTC).isoformat()
    else:
        raise NativeBridgeError("native_bridge.event_source_invalid")
    if len(binding.session_id) > CORRELATION_CHARACTERS:
        raise NativeBridgeError("native_bridge.activity_correlation_invalid")
    return {
        "event_kind": kind,
        "producer_id": PRODUCERS[binding.host],
        "producer_session": producer_scope(project, binding),
        "producer_sequence": int(_digest(identity)[:15], 16),
        "occurred_at": occurred_at,
        "summary": summary,
        "subject": subject,
        "correlation_ids": [binding.session_id],
    }


def file_observation(
    project: Path,
    binding: NativeResearchBinding,
    event: dict[str, Any],
    *,
    publish: Callable[[dict[str, Any]], dict[str, Any]],
    usage_goal_id: str | None = None,
    failure: str = "native_bridge.product_contract_unavailable",
) -> dict[str, object]:
    """File one Host-read event through the Host's own observation owner.

    A sequence the ledger already holds is this same event filed before, so it reads as
    filed; any other refusal is named, an owner's exception as ``failure``, and none stops
    research.
    """
    request = observation_request(project, binding, event, usage_goal_id=usage_goal_id)
    try:
        response = publish(request)
    except Exception as error:
        return {"status": "UNAVAILABLE", "reason": owner_failure_code(error) or failure}
    code = safe_failure_code(response.get("failure_code"))
    if response.get("status") in {"APPENDED", "REUSED_EXACT"} or (
        code == "observation.source_sequence_collision"
    ):
        return {
            "status": "DELIVERED",
            **{
                key: response[key]
                for key in ("observation_id", "goal_id")
                if isinstance(response.get(key), str)
            },
        }
    return {
        "status": "UNAVAILABLE",
        "reason": code or "native_bridge.product_contract_unavailable",
        **{
            key: response[key]
            for key in ("detail", "next_action")
            if isinstance(response.get(key), str)
        },
    }


MESSAGE_EVENT_KIND = "NATIVE_COORDINATION_MESSAGE"
BOUND_WORDS = (
    "The Host bound this session to the workspace at its first request; its usage is read when "
    "goals, answers and pages ask, and the workspace's reading switch governs it."
)
GOAL_HELD_WORDS = (
    "This session holds the goal its reference names; its requests and Tasks count toward it."
)


def file_bound_fact(
    project: Path,
    binding: NativeResearchBinding,
    *,
    at: datetime,
    publish: Callable[[dict[str, Any]], dict[str, Any]],
    goal_hash: str | None = None,
) -> dict[str, object]:
    """File one product fact: the Host bound this Session at its first request (AUTOBIND).

    It names the Session in Team and the Goal's Conversation from the first call, whether or
    not its usage file can be read yet. Filed again, it is the same observation. With
    `goal_hash` it names the goal the Session holds, which its Sessions row reads as its
    question (STOPS-1).
    """
    event: dict[str, Any] = {"source": "product_bound", "occurred_at": at.isoformat()}
    if goal_hash is not None:
        event["goal_hash"] = goal_hash
    return file_observation(project, binding, event, publish=publish)


def deliver_accepted_answer(
    project: Path,
    binding: NativeResearchBinding,
    *,
    bundle: AgentBundleRecord,
    answer: AgentAnswerRecord,
    contribution: Mapping[str, object],
    task_id: str,
    admitted_at: datetime,
    publish: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, object] | None:
    """File one accepted deliverable as a product fact of the lead's Session.

    The lead submitted it; who among its children wrote it is not observed, and the record
    says so. This is a product record, never a native transcript; a failure to file it
    leaves the acceptance as it is.
    """
    if answer.verdict not in {AnswerVerdict.ACCEPTED, AnswerVerdict.DONE}:
        return None
    references: dict[str, object] = {
        "host": binding.host,
        "session_id": binding.session_id,
        "bundle_reference": bundle.record_hash,
        "answer_reference": answer.record_hash,
        "task_id": task_id,
    }

    def unavailable(reason: str, *missing: str) -> dict[str, object]:
        return {
            "status": "UNAVAILABLE",
            "reason": reason,
            "missing": list(missing),
            **(refusal_words(reason) or refusal_words("native_bridge.accepted_delivery_failed")),
            **references,
        }

    try:
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
            "session_id": binding.session_id,
            "agent_id": binding.session_id,
            "role": LEAD_ROLE,
            "bundle_role": bundle.role,
            "message_id": "accepted-" + identity[:24],
            "message": summary,
            "reference": task_id,
            "bundle_reference": bundle.record_hash,
            "answer_reference": answer.record_hash,
            "submitted_by": binding.session_id,
            "recipient_id": binding.session_id,
            "occurred_at": (answer.accepted_at or admitted_at).isoformat(),
            "source_time_kind": "PRODUCT_ACCEPTED_AT"
            if answer.accepted_at is not None
            else "TASK_ADMISSION",
        }
        result = file_observation(
            project,
            binding,
            event,
            publish=publish,
            failure="native_bridge.accepted_delivery_failed",
        )
        if result.get("status") != "DELIVERED":
            return unavailable(
                str(result.get("reason") or "native_bridge.accepted_delivery_failed"),
                "accepted_answer_delivery",
            )
        return {**result, **references}
    except Exception:
        return unavailable("native_bridge.accepted_delivery_failed", "accepted_answer_delivery")


def _reading(
    project: Path,
    binding: NativeResearchBinding,
    *,
    agent_id: str,
    role: str,
    path: Path | None,
    publish: Callable[[dict[str, Any]], dict[str, Any]],
    goal_id: str | None,
) -> tuple[dict[str, object], SessionUsage | None]:
    """One participant's latest reading, filed per model; unavailable rather than wrong."""
    child = agent_id != binding.session_id
    prefix = "child" if child else "lead"
    member = {"agent_id": agent_id, "role": role}
    if path is None:
        return {
            **member,
            "status": "UNAVAILABLE",
            "reason": f"native_bridge.{prefix}_usage_file_missing",
        }, None
    try:
        reading = read_session(
            path, host=binding.host, max_bytes=USAGE_READ_BYTES, max_line_bytes=USAGE_LINE_BYTES
        )
    except NativeUsageReadLimitError as error:
        return {
            **member,
            "status": "UNAVAILABLE",
            "reason": f"native_bridge.{prefix}_usage_read_failed",
            "read_limit": error.limit,
        }, None
    except (OSError, ValueError):
        return {
            **member,
            "status": "UNAVAILABLE",
            "reason": f"native_bridge.{prefix}_usage_read_failed",
        }, None
    rows = reading.by_model()
    # An unknown or changed record shape is unavailable, never a partial or zero total.
    if reading.incomplete:
        return {
            **member,
            "status": "UNAVAILABLE",
            "reason": f"native_bridge.{prefix}_usage_incomplete",
            "incomplete": reading.incomplete,
        }, reading
    if len(rows) > USAGE_MODEL_ROWS:
        return {
            **member,
            "status": "UNAVAILABLE",
            "reason": f"native_bridge.{prefix}_usage_read_failed",
            "read_limit": "models",
        }, reading
    if not rows:
        return {
            **member,
            "status": "UNAVAILABLE",
            "reason": f"native_bridge.{prefix}_usage_not_observed",
        }, reading
    pins = role_pin(project, binding.host, role) if child else {}
    for row in rows:
        usage = {
            "session_id": binding.session_id,
            "agent_id": agent_id,
            "role": role,
            "host": binding.host,
            **asdict(row),
            "pin_differs": pin_differs(pins, row) if child else [],
        }
        filed = file_observation(
            project,
            binding,
            {"source": "native_usage", "usage": usage},
            publish=publish,
            usage_goal_id=goal_id,
        )
        if filed["status"] != "DELIVERED":
            return {**member, **filed}, reading
    return {**member, "status": "DELIVERED", "models": len(rows)}, reading


# A person's workspace switch for reading the bound Sessions' usage, beside the network's.
USAGE_CONTROL_PATH = Path("runtime") / "native-usage-reading.json"
_USAGE_CONTROL_SCHEMA = "native-usage-reading"
_USAGE_WORDS = {
    True: "The Host reads the usage of the agent Sessions bound to this workspace from their own "
    "session files when a goal is taken or submitted, an answer is submitted or a Team or Goal "
    "page opens. It keeps models, efforts, token counts and times, never conversation text.",
    False: "Usage reading is off: the Host reads no agent Session's file. Earlier readings stay.",
}


def usage_reading(workspace: Path) -> bool:
    """Whether the Host may read this workspace's bound Sessions' usage; on by default.

    A person turns it off on Settings. A record this owner did not write, or one it cannot
    read, reads as off: privacy fails closed.
    """
    try:
        record = json.loads((workspace / USAGE_CONTROL_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return False
    return (
        isinstance(record, dict)
        and set(record) == {"schema", "version", "reading"}
        and record["schema"] == _USAGE_CONTROL_SCHEMA
        and record["version"] == 1
        and record["reading"] is True
    )


def usage_reading_answer(workspace: Path) -> dict[str, object]:
    """The switch's standing, what it means and the request that turns it the other way."""
    enabled = usage_reading(workspace)
    return {
        "status": "USAGE_READING",
        "usage_reading": "READ" if enabled else "OFF",
        "detail": _USAGE_WORDS[enabled],
        "next_requests": {
            "set": {"operation": "USAGE_READING_SET", "usage_reading_enabled": not enabled}
        },
    }


def set_usage_reading(workspace: Path, *, enabled: bool) -> dict[str, object]:
    """Atomically write a person's setting, then read it back."""
    path = workspace / USAGE_CONTROL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".json.tmp")
    staged.write_text(
        json.dumps({"schema": _USAGE_CONTROL_SCHEMA, "version": 1, "reading": enabled}), "utf-8"
    )
    replace_shared_file(staged, path)
    return usage_reading_answer(workspace)


def read_session_usage(
    project: Path,
    binding: NativeResearchBinding,
    *,
    publish: Callable[[dict[str, Any]], dict[str, Any]],
    goal_id: str | None = None,
) -> dict[str, object]:
    """Read the bound Session and the children its own files record, once, at the Host.

    Only the bound Session's own file is opened, then each child it names: a Codex lead's
    ``SubAgentActivity`` items, or a Claude Code lead's ``subagents`` directory. A child is
    read only when its own metadata names this Session as its direct parent and one of the
    binding's roles. Nothing runs between readings; each participant stands alone, and a
    failure of one is named beside the others, never a refusal of research.

    Args:
        project: The Host's admitted project.
        binding: The exact parent Session, workspace and roles admitted by that Host.
        publish: The Host's existing observation owner.
        goal_id: The exact Goal the Host resolved for this reading.

    Returns:
        ``DELIVERED``, ``PARTIAL`` or ``UNAVAILABLE`` with one entry per participant.
    """
    if binding.usage == "OFF" or not usage_reading(binding.workspace):
        return {"status": "SKIPPED", "reason": "native_bridge.usage_disabled"}
    members: list[dict[str, object]] = []
    try:
        lead_path = session_file(binding.host, binding.session_id)
        lead, reading = _reading(
            project,
            binding,
            agent_id=binding.session_id,
            role=LEAD_ROLE,
            path=lead_path,
            publish=publish,
            goal_id=goal_id,
        )
        members.append(lead)
        if binding.host == "codex":
            children = reading.children if reading is not None else ()
        else:
            children = claude_children(binding.session_id)
        for child in children:
            spawn = (
                codex_thread_spawn(child)
                if binding.host == "codex"
                else claude_thread_spawn(binding.session_id, child)
            )
            if (
                spawn is not None
                and spawn.parent_thread_id == binding.session_id
                and spawn.agent_role is not None
                and not spawn.agent_role.startswith(PRODUCT_ROLE_PREFIX)
            ):
                # The lead's own helper, no product card (a general subagent, or a card started
                # from its text in the installing session): named, never read, and no failure
                # of the reading (STOPS-1).
                members.append(
                    {
                        "agent_id": child,
                        "status": "NOT_READ",
                        "reason": "native_bridge.child_not_a_specialist",
                    }
                )
                continue
            if (
                spawn is None
                or spawn.parent_thread_id != binding.session_id
                or spawn.agent_role not in binding.roles
            ):
                members.append(
                    {
                        "agent_id": child,
                        "status": "UNAVAILABLE",
                        "reason": "native_bridge.child_usage_binding_unverified",
                    }
                )
                continue
            member, _ = _reading(
                project,
                binding,
                agent_id=child,
                role=spawn.agent_role,
                path=session_file(binding.host, binding.session_id, child),
                publish=publish,
                goal_id=goal_id,
            )
            members.append(member)
    except NativeUsageReadLimitError as error:
        members.append(
            {
                "status": "UNAVAILABLE",
                "reason": "native_bridge.child_usage_read_failed",
                "read_limit": error.limit,
            }
        )
    except Exception:
        members.append({"status": "UNAVAILABLE", "reason": "native_bridge.lead_usage_read_failed"})
    read = [member for member in members if member.get("status") != "NOT_READ"]
    delivered = sum(1 for member in read if member.get("status") == "DELIVERED")
    status = "DELIVERED" if delivered == len(read) else "PARTIAL" if delivered else "UNAVAILABLE"
    first = next((m for m in read if m.get("status") != "DELIVERED"), None)
    return {
        "status": status,
        "participants": members,
        **({"reason": first["reason"]} if first is not None and "reason" in first else {}),
    }


def readiness(project: Path, binding: NativeResearchBinding | None) -> dict[str, Any]:
    """Whether this project's Session binding is usable for research.

    It reads only the project declaration and the supplied exact binding; no native file,
    hook definition or host trust.

    Args:
        project: The agent project.
        binding: The caller's exact binding, when bound.

    Returns:
        ``READY``, or ``REFUSED`` naming what is missing; research never waits on it.
    """
    host = "codex" if binding is None else binding.host
    missing: list[str] = []
    failure: str | None = None
    try:
        declared = session_project(project, host)
        if declared.resolve() != project.resolve():
            raise ValueError("native_bridge.project_mismatch")
    except (OSError, ValueError) as error:
        missing.append("project_declaration")
        failure = (
            str(error)
            if isinstance(error, ValueError)
            else "native_bridge.configuration_path_invalid"
        )
    if binding is None:
        missing.append("native_session_binding")
        failure = failure or "native_bridge.not_bound"
    return {
        "status": "READY" if not missing else "REFUSED",
        "failure_code": failure,
        "host": host,
        "session_id": None if binding is None else binding.session_id,
        "missing": missing,
        "research_nonblocking": True,
    }


def session_setup(project: Path, session: tuple[str, str] | None) -> dict[str, Any]:
    """An agent Session's setup: its binding preflight, and the retrieval runtime it lacks.

    Outside a Session there is no binding to check. The retrieval runtime is the lock's own
    dependency, the agent's to fill before the Evidence setup meets it.
    """
    from alphalattice.interface.local_application.retrieval_environment import fill_command, load

    answer: dict[str, Any] = {"status": "NO_AGENT_SESSION"}
    if session is not None:
        try:
            binding = NativeResearchBinding.read(project, session=session)
        except (OSError, ValueError):
            binding = None
        answer = attachment_preflight(project, binding)
    if not load():
        command = join(fill_command(), shell())
        answer["retrieval_runtime"] = {"present": False, "who_decides": "AGENT", "command": command}
    return answer


def attachment_preflight(
    project: Path, binding: NativeResearchBinding | None = None
) -> dict[str, Any]:
    """Read the binding's readiness and attach the named way forward."""
    result = readiness(project, binding)
    return {
        **result,
        **(refusal_words(result["failure_code"]) if result.get("failure_code") else {}),
        **(
            {
                "detail": "The Session binding is ready for research.",
                "next_action": "Continue research with this Session binding.",
            }
            if not result.get("failure_code")
            else {}
        ),
    }
