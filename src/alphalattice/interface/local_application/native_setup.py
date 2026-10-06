"""Configure, inspect and invoke the project-local native observation bridge."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator

from alphalattice.interface.local_application.failure_codes import setup_failure
from alphalattice.interface.local_application.native_bridge import (
    BINDING_NAME,
    HOSTS,
    MAX_MESSAGE_BYTES,
    USAGE_READINGS,
    NativeBridgeError,
    NativeResearchBinding,
    coordination_event,
    deliver,
    hook_reply,
)
from alphalattice.interface.local_application.native_hook_input import (
    MAX_HOOK_INPUT_BYTES,
)
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PRODUCT_HOOK_MATCHER = "^alphalattice_.*$"
"""The matcher of the hooks the product declares to a host: the mark of an agent project."""
RESOURCE_ROOT = resolve_playpen_root(Path(__file__))
INSTALLED = not (RESOURCE_ROOT / "pyproject.toml").is_file()
ROOT = Path.cwd() if INSTALLED else RESOURCE_ROOT


def _absolute_project(value: Path) -> Path:
    """Validate a bounded absolute project path without inspecting its contents.

    Args:
        value: Project path supplied at the typed native session door.

    Returns:
        The unchanged path, ready for the Host's independent admission.

    Raises:
        ValueError: The path is relative, unbounded or contains control characters.
    """
    if (
        not value.is_absolute()
        or len(str(value)) > 512
        or any(ord(character) < 32 for character in str(value))
    ):
        raise ValueError("native_bridge.project_path_invalid")
    return value


class NativeSessionBindRequest(BaseModel):  # type: ignore[misc]
    """A binding request names a project and usage; its session is the request's provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    project: Path
    usage: Literal["read", "off"] = "read"

    @field_validator("project")  # type: ignore[untyped-decorator]
    @classmethod
    def absolute_project(cls, value: Path) -> Path:
        """Refuse relative or unbounded paths before the Host inspects its own project."""
        return _absolute_project(value)


class NativeEventDeliveryRequest(BaseModel):  # type: ignore[misc]
    """One selected safe message; no session, workspace or producer sequence is supplied."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    project: Path
    event: dict[str, Any]

    @field_validator("project")  # type: ignore[untyped-decorator]
    @classmethod
    def absolute_project(cls, value: Path) -> Path:
        """Apply the binding door's path contract to an explicit coordination message."""
        return _absolute_project(value)


def admitted_session_project(workspace: Path, requested: Path, host: str) -> Path:
    """Derive a served workspace's exact configured project independently of the client path.

    Args:
        workspace: The Host's own served workspace.
        requested: The client's nominated absolute project path.
        host: The admitted native host identifier.

    Returns:
        The independently resolved project containing the served workspace.

    Raises:
        NativeBridgeError: The project differs, aliases a path or has unsafe declarations.
    """
    project = session_project(workspace.resolve(), host)
    if requested.is_symlink() or requested.absolute() != requested.resolve():
        raise NativeBridgeError("native_bridge.project_path_invalid")
    if requested.resolve() != project.resolve():
        raise NativeBridgeError("native_bridge.project_mismatch")
    declaration = project / (
        ".claude/settings.json" if host == "claude-code" else ".codex/config.toml"
    )
    if any(path.is_symlink() for path in (project, declaration.parent, declaration)):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    if not workspace.resolve().is_relative_to(project.resolve()):
        raise NativeBridgeError("native_bridge.project_mismatch")
    return project.resolve()


def attachment_preflight(
    project: Path | None = None,
    binding: NativeResearchBinding | None = None,
    events: Iterable[Mapping[str, Any]] | None = None,
    *,
    goal_id: str | None = None,
    runtime: dict[str, Any] | None = None,
    read_external: Callable[..., Mapping[str, Any]] | None = None,
    history_available: bool = True,
) -> dict[str, Any]:
    """Read the common native readiness owner and attach the named way forward."""
    from alphalattice.interface.local_application.cli_contract import refusal_words
    from alphalattice.interface.local_application.native_runtime import readiness

    result = readiness(
        ROOT if project is None else project,
        binding,
        events,
        goal_id=goal_id,
        runtime=runtime,
        read_external=read_external,
        history_available=history_available,
    )
    return {
        **result,
        **(refusal_words(result["failure_code"]) if result.get("failure_code") else {}),
    }


def files_unavailable(
    error: OSError, *, project: Path, workspace: Path | None = None
) -> dict[str, Any]:
    """Name only an admitted local path and its category, never the exception's prose.

    Args:
        error: The failed local file operation, used only for its class and filename.
        project: The admitted absolute project root bounding safe path disclosure.
        workspace: The served workspace, when separately available.

    Returns:
        A refusal with the failure category, admitted path when known and remedy.
    """
    from alphalattice.interface.local_application.cli_contract import refusal_words

    candidate = Path(error.filename) if isinstance(error.filename, str) else None
    safe_path: Path | None = None
    category = "PROJECT_FILES"
    if candidate is not None:
        candidate = candidate.absolute()
        if candidate.is_relative_to(project):
            safe_path = candidate
            category = (
                "PROJECT_BINDING"
                if candidate == project / ".codex" / BINDING_NAME
                else "HOST_DECLARATIONS"
                if candidate.is_relative_to(project / ".codex")
                or candidate.is_relative_to(project / ".claude")
                else "PROJECT_FILES"
            )
        elif workspace is not None and candidate.is_relative_to(workspace):
            safe_path, category = candidate, "WORKSPACE"
    code = f"native_bridge.files_unavailable:{type(error).__name__}"
    return {
        "status": "REFUSED",
        "failure_code": code,
        "reason": code,
        "path_category": category,
        **({"path": str(safe_path)} if safe_path is not None else {}),
        **refusal_words(code),
    }


def _declarations() -> Path:
    return (
        RESOURCE_ROOT
        if INSTALLED
        and not (ROOT / ".codex/config.toml").is_file()
        and not (ROOT / ".claude/settings.json").is_file()
        else ROOT
    )


def _codex_hook_root(project: Path) -> Path:
    """Resolve Codex's linked-worktree hook root from bounded Git metadata only."""

    def metadata(path: Path) -> str:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        with path.open("rb") as stream:
            data = stream.read(64 * 1024 + 1)
        if len(data) > 64 * 1024:
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        return os.fsdecode(data).strip()

    def git_pointer(path: Path) -> Path:
        value = metadata(path)
        if not value.startswith("gitdir:") or not value[7:].strip():
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        return (path.parent / value[7:].strip()).resolve(strict=True)

    project = project.resolve()
    for checkout in (project, *project.parents):
        marker = checkout / ".git"
        if marker.is_symlink():
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        if marker.is_dir():
            if (marker / "HEAD").exists():
                return project
            continue
        if not marker.exists():
            continue
        directory = git_pointer(marker)
        if not directory.is_dir() or directory.parent.name != "worktrees":
            return project
        common = directory.parent.parent
        registered = (directory / metadata(directory / "gitdir")).resolve(strict=True)
        linked_common = (directory / metadata(directory / "commondir")).resolve(strict=True)
        if registered.name != ".git" or registered.parent != checkout or linked_common != common:
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        main = common.parent
        main_marker = main / ".git"
        main_directory = main_marker.resolve() if main_marker.is_dir() else git_pointer(main_marker)
        if main_directory != common:
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        return main / project.relative_to(checkout)
    return project


def _hook_root_refusal(host: str) -> dict[str, Any] | None:
    """Refuse ineffective local hooks before configuration can write or claim readiness."""
    if host != "codex":
        return None
    root = _codex_hook_root(ROOT)
    if root == ROOT.resolve():
        return None
    code = "native_bridge.configuration_path_invalid"
    return {
        "status": "REFUSED",
        "failure_code": code,
        "reason": code,
        "host": host,
        "hook_declaration_root": str(root),
        "hook_declarations_effective": False,
        "host_trust": "NOT_CHECKED",
        "foreground_attachment": "NOT_PROVED",
        "trust_changed": False,
        "detail": (
            "Codex reads this linked worktree's hooks from the main checkout. "
            "The local declarations do not configure that hook root, which is outside "
            "this project. Use an independent ordinary project, run native_research.py "
            "configure there, then have the person review the actual definitions in /hooks. "
            "This command has changed no declarations or trust."
        ),
        "next_action": "USE_LOCAL_DECLARATIONS_AND_METADATA_IN_THE_EXACT_PROJECT",
    }


def _install_declarations(host: str) -> None:
    """Copy shipped agent guidance unchanged, and bind hooks to this installed interpreter."""
    manifest = json.loads(
        (RESOURCE_ROOT / "config/release/runtime-resources.json").read_text(encoding="utf-8")
    )
    host_directory = ".claude" if host == "claude-code" else ".codex"
    inputs = ["AGENTS.md"]
    if host == "claude-code":
        inputs.append("CLAUDE.md")
    for directory in manifest["directories"]:
        if directory.startswith(host_directory + "/") or directory.startswith(".agents/"):
            inputs.extend(
                path.relative_to(RESOURCE_ROOT).as_posix()
                for path in (RESOURCE_ROOT / directory).rglob("*")
                if path.is_file()
            )
    declaration = ".claude/settings.json" if host == "claude-code" else ".codex/config.toml"
    inputs.append(declaration)
    arguments = [
        sys.executable,
        "-m",
        "alphalattice.interface.local_application.native_setup",
        "--project",
        str(ROOT),
        "hook",
    ]
    command = shlex.join(arguments) if os.name != "nt" else subprocess.list2cmdline(arguments)
    command += " || exit 0"
    documents: dict[Path, bytes] = {}
    for relative in sorted(set(inputs)):
        data = (RESOURCE_ROOT / relative).read_bytes()
        if relative == declaration:
            if host == "claude-code":
                document = json.loads(data)
                for groups in document["hooks"].values():
                    for group in groups:
                        for hook in group["hooks"]:
                            hook["command"] = command
                data = (json.dumps(document, indent=2) + "\n").encode()
            else:
                data = re.sub(
                    r"^command = .+$",
                    lambda _: "command = " + json.dumps(command),
                    data.decode(),
                    flags=re.MULTILINE,
                ).encode()
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        if path.exists() and path.read_bytes() != data:
            raise NativeBridgeError("native_bridge.existing_configuration_differs")
        documents[path] = data
    for path, data in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)


def _roles(host: str = "codex", root: Path | None = None) -> list[str]:
    declarations = _declarations() if root is None else root
    if host == "claude-code":
        # The Claude host's own cards (scripts/materialize_claude_host.py derives them), with
        # the medium-effort card of each evidence specialist; no Codex file is read.
        return sorted(
            path.stem for path in (declarations / ".claude/agents").glob("alphalattice_*.md")
        )
    with (declarations / ".codex/config.toml").open("rb") as stream:
        agents = tomllib.load(stream).get("agents", {})
    return sorted(name for name in agents if name.startswith("alphalattice_"))


def _declares_product(folder: Path, host: str) -> bool:
    """Whether a folder holds this host's declarations of the product's hooks (V568)."""
    declaration = folder / (
        ".claude/settings.json" if host == "claude-code" else ".codex/config.toml"
    )
    if declaration.is_symlink() or declaration.parent.is_symlink():
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    try:
        if host == "claude-code":
            settings = json.loads(declaration.read_text(encoding="utf-8"))
            hooks = settings.get("hooks", {}) if isinstance(settings, dict) else {}
        else:
            with declaration.open("rb") as stream:
                hooks = tomllib.load(stream).get("hooks", {})
    except (OSError, ValueError):
        return False
    return isinstance(hooks, dict) and any(
        isinstance(group, dict) and group.get("matcher") == PRODUCT_HOOK_MATCHER
        for groups in hooks.values()
        if isinstance(groups, list)
        for group in groups
    )


def session_project(start: Path, host: str | None) -> Path:
    """The agent project a session binds in, found up from where the binding runs (V568).

    It is the nearest folder up from ``start`` whose host declarations declare the product's
    hooks, as `configure` writes them and a checkout ships them; a host's own settings, a home
    folder's ``.codex/config.toml`` or ``.claude/settings.json``, declare none and are no
    project.

    Args:
        start: The directory the binding command runs in.
        host: ``codex`` or ``claude-code``; None, outside any agent session, for either.

    Returns:
        The project's folder.

    Raises:
        NativeBridgeError: ``native_bridge.hook_declaration_missing`` when no folder up holds
            them.
    """
    hosts = HOSTS if host is None else (host,)
    for folder in (start, *start.parents):
        if any(_declares_product(folder, each) for each in hosts):
            return folder
    raise NativeBridgeError("native_bridge.hook_declaration_missing")


def bind_session(
    project: Path, *, host: str, session_id: str, workspace: Path, usage: str = "read"
) -> dict[str, Any]:
    """Bind one agent session to a workspace in a project, once (V568).

    The binding is the one the bridge's hooks read and the CLI finds from any folder within
    the project.

    Args:
        project: The agent project, a checkout or a configured folder.
        host: The host running the session.
        session_id: The session, as its host names it.
        workspace: The workspace's folder.
        usage: ``off`` keeps a subagent's stop from reading the host's session files.

    Returns:
        The binding as written, with its project.

    Raises:
        NativeBridgeError: For a missing workspace, an invalid binding or a project bound
            otherwise.
    """
    if not workspace.is_dir():
        raise NativeBridgeError("native_bridge.workspace_missing")
    folder = str(workspace.resolve())
    document = {
        "session_id": session_id,
        "workspace": folder,
        "roles": _roles(host, project),
        "host": host,
        # Named only when off, so a binding written before the switch reads the same.
        **({"usage": "OFF"} if usage == "off" else {}),
    }
    existing = NativeResearchBinding.read(project)
    if existing is None:
        document["observation_started_at"] = datetime.now(UTC).isoformat()
    else:
        if (
            existing.session_id != session_id
            or existing.workspace.resolve() != workspace.resolve()
            or existing.host != host
            or existing.roles != tuple(document["roles"])
            or existing.usage != ("OFF" if usage == "off" else "READ")
        ):
            raise NativeBridgeError("native_bridge.existing_configuration_differs")
        if existing.observation_started_at is not None:
            document["observation_started_at"] = existing.observation_started_at.isoformat()
    NativeResearchBinding.from_document(document)
    _create_or_match(BINDING_NAME, document, project)
    binding = NativeResearchBinding.read(project)
    preflight = attachment_preflight(
        project,
        binding,
        runtime={"status": "NOT_CHECKED", "host_trust": "NOT_CHECKED", "trust_changed": False},
    )
    return {
        "status": "BOUND_NOT_ATTACHED",
        "session_id": session_id,
        "host": host,
        "workspace": folder,
        "project": str(project),
        "host_trust": "NOT_CHECKED",
        "foreground_attachment": "NOT_PROVED",
        "attachment_preflight": preflight,
        "detail": preflight["detail"],
        "next_action": preflight["next_action"],
    }


def _hooks(host: str) -> dict[str, Any]:
    if host == "claude-code":
        # The derived host files (scripts/materialize_claude_host.py) declare the same hooks.
        settings = json.loads(
            (_declarations() / ".claude/settings.json").read_text(encoding="utf-8")
        )
        return cast(dict[str, Any], settings.get("hooks", {}))
    with (_declarations() / ".codex/config.toml").open("rb") as stream:
        return cast(dict[str, Any], tomllib.load(stream).get("hooks", {}))


def _configuration(host: str = "codex") -> dict[str, Any]:
    python = (
        Path(sys.executable)
        if INSTALLED
        else ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if not python.is_file():
        raise NativeBridgeError("native_bridge.local_environment_missing")
    if not _roles(host):
        raise NativeBridgeError("native_bridge.roles_missing")
    if host == "claude-code":
        try:
            hooks = _hooks(host)
        except (OSError, ValueError, AttributeError):
            raise NativeBridgeError("native_bridge.hook_declaration_missing") from None
        if not all(
            any(
                isinstance(group, dict) and group.get("matcher") == PRODUCT_HOOK_MATCHER
                for group in hooks.get(event) or []
            )
            for event in ("SubagentStart", "SubagentStop")
        ):
            raise NativeBridgeError("native_bridge.hook_declaration_missing")
        return hooks
    hooks = _hooks(host)
    if not all(hooks.get(event) for event in ("SubagentStart", "SubagentStop")):
        raise NativeBridgeError("native_bridge.hook_declaration_missing")
    return hooks


def unbind_session(project: Path, *, session_id: str | None) -> dict[str, object]:
    """Remove a project's binding, once: the bound session's own, or the person's (V586).

    An agent session removes only the binding that names it; the person, outside any agent
    session (``session_id`` None), removes the project's binding whichever session it names.
    The research is unchanged: a binding is a session's default workspace and the scope of its
    observation, never a record.

    Args:
        project: The agent project holding the binding.
        session_id: The session removing it; None for the person.

    Returns:
        ``NOT_BOUND``, or ``DETACHED`` with the session the removed binding named.

    Raises:
        NativeBridgeError: ``native_bridge.session_mismatch`` when the binding names another
            session than the one removing it; the binding's own read refusals.
    """
    binding = NativeResearchBinding.read(project)
    if binding is None:
        return {"status": "NOT_BOUND", "project": str(project)}
    if session_id is not None and binding.session_id != session_id:
        raise NativeBridgeError("native_bridge.session_mismatch")
    (project / ".codex" / BINDING_NAME).unlink()
    return {
        "status": "DETACHED",
        "project": str(project),
        "session_id": binding.session_id,
        "research_unchanged": True,
    }


def _create_or_match(name: str, document: dict[str, Any], project: Path | None = None) -> None:
    root = ROOT if project is None else project
    path = root / ".codex" / name
    if (
        path.parent.is_symlink()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    data = (json.dumps(document, indent=2) + "\n").encode()
    # A Claude Code project configured from the installed command holds no `.codex` yet.
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        with path.open("rb") as stream:
            existing = stream.read(len(data) + 1)
        if existing != data:
            raise NativeBridgeError("native_bridge.existing_configuration_differs") from None


def main() -> int:
    """Configure or invoke the native bridge on the selected project.

    Returns:
        Zero on success, two on a named refusal, three when delivery is unavailable.
    """
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, help="The writable agent project; cwd when installed."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    configure = commands.add_parser(
        "configure", help="Check shipped hook declarations and the local environment."
    )
    configure.add_argument("--host", choices=HOSTS, default="codex")
    bind = commands.add_parser("bind", help="Bind this bridge to an explicit session/workspace.")
    bind.add_argument("--session-id", required=True)
    bind.add_argument("--workspace", required=True, type=Path)
    bind.add_argument(
        "--host", choices=HOSTS, default="codex", help="The foreground host sending the hooks."
    )
    bind.add_argument(
        "--usage",
        choices=[value.lower() for value in USAGE_READINGS],
        default="read",
        help="off: a subagent's stop reads no session file of the host for what it ran and spent.",
    )
    unbind = commands.add_parser(
        "unbind", help="Detach only the named session; research is unchanged."
    )
    unbind.add_argument("--session-id", required=True)
    doctor = commands.add_parser(
        "doctor",
        help="Read runtime definitions and prospective native evidence; no model operation.",
    )
    doctor.add_argument(
        "--host", choices=HOSTS, help="The host to inspect; by default the bound one, else codex."
    )
    commands.add_parser("hook", help="Native lifecycle stdin; advisory success/refusal uses exit0.")
    message = commands.add_parser(
        "message", help="Deliver explicitly supplied safe UTF-8 text on stdin."
    )
    message.add_argument(
        "--agent-id", help="The sender; by default the bound session, the research lead."
    )
    message.add_argument(
        "--role", help="The sender's role; by default research_lead for the bound session."
    )
    message.add_argument("--kind", required=True)
    message.add_argument(
        "--message-id",
        help="This message's id; by default its content's, so a retry is the same message.",
    )
    message.add_argument("--reference")
    message.add_argument("--to", dest="recipient_id")
    message.add_argument(
        "--reply-to", help="The message id this one answers; an answer to an assignment closes it."
    )
    args = parser.parse_args()
    if args.project is not None:
        ROOT = args.project.resolve()
    result: dict[str, Any]
    if args.command == "hook":
        data = sys.stdin.buffer.read(MAX_HOOK_INPUT_BYTES + 1)
        print(json.dumps(hook_reply(ROOT, data)))
        return 0
    try:
        if args.command == "configure":
            refusal = _hook_root_refusal(args.host)
            if refusal is not None:
                print(json.dumps(refusal))
                return 2
            if INSTALLED:
                _install_declarations(args.host)
            _configuration(args.host)
            result = {
                "status": "LOCAL_DECLARATIONS_VALIDATED",
                "host": args.host,
                "next": (
                    "Inspect actual project hook presence and trust in the host's /hooks. "
                    "No trust was changed."
                ),
            }
        elif args.command == "bind":
            result = bind_session(
                ROOT,
                host=args.host,
                session_id=args.session_id,
                workspace=args.workspace,
                usage=args.usage,
            )
        elif args.command == "unbind":
            # The checkout's shim of `alphalattice session unbind`, by the session it names.
            result = unbind_session(ROOT, session_id=args.session_id)
        elif args.command == "doctor":
            from alphalattice.interface.local_application.client import LocalResearchClient
            from alphalattice.interface.local_application.native_runtime import retained_history

            binding = NativeResearchBinding.read(ROOT)
            host = args.host or (binding.host if binding is not None else "codex")
            refusal = _hook_root_refusal(host)
            if refusal is not None:
                print(json.dumps(refusal))
                return 2

            def read_external(**selector: Any) -> dict[str, Any]:
                assert binding is not None
                return LocalResearchClient(binding.workspace, timeout=5).read_external(**selector)

            history, history_status = (
                retained_history(read_external)
                if binding is not None
                else (None, {"status": "UNAVAILABLE"})
            )
            preflight = attachment_preflight(
                ROOT,
                binding,
                history,
                read_external=read_external if binding is not None else None,
                history_available=history_status["status"] == "AVAILABLE",
            )
            result = {
                **preflight,
                "hook_declarations_present": bool(_hooks(host)),
                "session_bound": binding is not None,
                "host": binding.host if binding is not None else None,
                "usage_reading": binding.usage if binding is not None else None,
                "roles": _roles(host),
                "attachment_preflight": preflight,
                "history_read": history_status,
                "activity_client_supported": callable(
                    getattr(LocalResearchClient, "publish_event", None)
                ),
            }
        else:
            binding = NativeResearchBinding.read(ROOT)
            if binding is None:
                raise NativeBridgeError("native_bridge.not_bound")
            event = coordination_event(
                binding,
                agent_id=args.agent_id,
                role=args.role,
                kind=args.kind,
                message_id=args.message_id,
                message=sys.stdin.buffer.read(MAX_MESSAGE_BYTES + 1),
                reference=args.reference,
                recipient_id=args.recipient_id,
                reply_to=args.reply_to,
            )
            # Its id, for a reply's --reply-to, whichever way the delivery went.
            result = {
                **deliver(ROOT, binding, event, host_owned=True),
                "message_id": event["message_id"],
            }
        print(json.dumps(result))
        return 3 if result["status"] == "UNAVAILABLE" else 2 if result["status"] == "REFUSED" else 0
    except NativeBridgeError as error:
        print(
            json.dumps(
                {**setup_failure(error), "reason": str(error), "next_commands": _doctor_command()}
            )
        )
        return 2
    except OSError as error:
        # Its files are the checkout's host directories and the workspace's; one it cannot read
        # or write is said with its way on, never a bare code (V539), and the check its words
        # name is offered (V590).
        refusal = {
            **setup_failure(error),
            **files_unavailable(error, project=ROOT, workspace=getattr(args, "workspace", None)),
            "next_commands": _doctor_command(),
        }
        print(json.dumps(refusal))
        return 2
    except Exception as error:
        print(json.dumps({**setup_failure(error), "next_commands": _doctor_command()}))
        return 2


def _doctor_command() -> dict[str, list[str]]:
    """The bridge's own read-only check, every refusal's way on, as the person runs it."""
    return {
        "doctor": [
            sys.executable,
            "-m",
            "alphalattice.interface.local_application.native_setup",
            "--project",
            str(ROOT),
            "doctor",
        ]
        if INSTALLED
        else [sys.executable, "scripts/native_research.py", "doctor"]
    }


if __name__ == "__main__":
    raise SystemExit(main())
