"""Configure, bind and inspect a project's native Session for research; no hooks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, field_validator

from alphalattice.control.product_host.storage.inventory import RECOVERY_HEADROOM_BYTES
from alphalattice.interface.local_application.failure_codes import setup_failure
from alphalattice.interface.local_application.native_bridge import (
    BINDING_DIRECTORY,
    BINDING_NAME,
    BINDING_RECORDS,
    HOSTS,
    PROJECT_DECLARATION_NAME,
    PROJECT_DECLARATION_SCHEMA,
    SPAWN_HOPS,
    USAGE_READINGS,
    NativeBridgeError,
    NativeResearchBinding,
    declares_product,
    session_project,
)
from alphalattice.interface.local_application.native_usage import codex_thread_spawn
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

PRODUCT_HOOK_MATCHER = "^alphalattice_.*$"
"""The matcher of the product's retired lifecycle hooks; configure removes groups with it."""
PRODUCT_HOOK_EVENTS = ("SubagentStart", "SubagentStop")
RESOURCE_ROOT = resolve_playpen_root(Path(__file__))
INSTALLED = not (RESOURCE_ROOT / "pyproject.toml").is_file()
ROOT = Path.cwd() if INSTALLED else RESOURCE_ROOT


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
    return result


def codex_queue_readiness() -> dict[str, Any]:
    """Whether this process can start the background queue the Host must deliver through."""
    return command_readiness(["codex", "queue", "--help"], timeout=5)


def _installed_leg() -> dict[str, Any]:
    """Compare the active distribution with this checkout, naming unproved PATH ownership."""
    command = shutil.which("alphalattice")
    expected = None
    if (RESOURCE_ROOT / "pyproject.toml").is_file():
        expected = tomllib.loads((RESOURCE_ROOT / "pyproject.toml").read_text("utf-8"))["project"][
            "version"
        ]
    try:
        distribution = importlib.metadata.distribution("alphalattice")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        url = urlsplit(direct.get("url", ""))
        source = (
            Path(unquote(url.path).lstrip("/") if os.name == "nt" else unquote(url.path))
            if url.scheme == "file"
            else None
        )
        matches = (expected is None or distribution.version == expected) and (
            source is None or source.resolve() == RESOURCE_ROOT.resolve()
        )
        paired = command is not None and Path(command).parent == Path(sys.executable).parent
        return {
            "present": matches if paired else False if command is None or not matches else None,
            "command": command,
            "version": distribution.version,
            "checkout_version": expected,
            "source": None if source is None else str(source),
            "path_owner": "ACTIVE_ENVIRONMENT" if paired else "NOT_PROVED",
        }
    except importlib.metadata.PackageNotFoundError:
        return {"present": False, "command": command, "reason": "DISTRIBUTION_MISSING"}


def _host_facts(binding: NativeResearchBinding | None) -> dict[str, Any]:
    """Read existing Host facts when bound; an absent Host supplies no made-up measurements."""
    from alphalattice.interface.local_application.client import (
        LocalResearchClient,
        LocalResearchClientError,
    )

    if binding is None:
        return {"scope": "LOCAL_PROCESS", "reason": "SESSION_NOT_BOUND"}
    try:
        client = LocalResearchClient(Path(binding.workspace), timeout=5)
        return {
            "scope": "WORKSPACE_HOST",
            "capacity": client.request({"operation": "STORAGE_CAP_SHOW"}).get("capacity"),
            "cpu": client.request({"operation": "CPU_BUDGET_SHOW"}),
            "network": client.request({"operation": "NETWORK_ACCESS"}),
            "queue": client.activity(limit=1).get("observer", {}).get("codex_queue"),
        }
    except LocalResearchClientError as error:
        return {"scope": "LOCAL_PROCESS", "reason": str(error).partition(":")[0]}


HOST_DEPENDENCIES = {
    "python": ("Run locked Python", "Use Python 3.12 and uv sync --locked; disclose it.", "AGENT"),
    "uv": (
        "Install the lock's dependencies",
        "Ask once to install uv, then the agent installs it; without it the locked "
        "environment cannot be repaired.",
        "PERSON",
    ),
    "installed_leg": (
        "Resolve the clean alphalattice command",
        "Run uv sync --locked and the documented tool install; disclose it. "
        "Check an unproved PATH owner first.",
        "AGENT",
    ),
    "retrieval_runtime": (
        "Read Evidence with its retrieval recipe",
        "Use the retrieval environment setup and install_retrieval_pack "
        "--install hybrid-v2-minilm; disclose the pinned repair. Keep packs outside "
        "the workspace; network acquisition still needs its permission.",
        "AGENT",
    ),
    "background_notices": (
        "Continue after background work ends",
        "Ask once to install the Codex CLI outside the lock; the agent installs it "
        "and checks codex queue --help. Restart its idle Host to refresh the check. "
        "Until available, WAIT_IN_THE_TURN occupies this turn until work ends.",
        "PERSON",
    ),
    "disk": (
        "Keep admitted data within disk and storage capacity",
        "Read storage cap show and storage plan; ask once before cleanup or changing "
        "the cap. Bind/start the Host to measure its cap; free space cannot prove it fits.",
        "PERSON",
    ),
    "cpu": (
        "Budget Task workers",
        "Read cpu-budget show; the agent sets workers and tasks_waiting within "
        "the reported machine budget.",
        "AGENT",
    ),
    "network": (
        "Know whether acquisition is admitted",
        "Read network-access show; relay the person's permission before opening "
        "the network. A closed network remains a valid offline setting.",
        "PERSON",
    ),
    "user_layer": (
        "Local user tuning",
        "Use maintenance Skill; disclose the optional layer.",
        "AGENT",
    ),
    "user_backup": (
        "Recover the person's layer before an upgrade",
        "Back up the optional user layer before upgrading, using the maintenance "
        "Skill; disclose it. A missing backup does not refuse research.",
        "AGENT",
    ),
}


def host_readiness(
    project: Path, host: str, binding: NativeResearchBinding | None
) -> list[dict[str, Any]]:
    """One advisory dependency list: the measured facts, what they enable and who fills them."""
    from alphalattice.interface.local_application.retrieval_environment import interpreter_path
    from alphalattice.kernel.knowledge.model_store import default_store_root, recipe_readiness

    facts = _host_facts(binding)
    retrieval = command_readiness([str(interpreter_path()), "-c", "import fastembed, onnxruntime"])
    packs = recipe_readiness(default_store_root(), "hybrid-v2-minilm")
    queue = (
        {"present": True, "scope": "HOST_FEATURE"}
        if host == "claude-code"
        else {
            "scope": facts["scope"],
            **(facts.get("queue") or {"present": None, "reason": "HOST_CHECK_NOT_REPORTED"}),
        }
        if facts["scope"] == "WORKSPACE_HOST"
        else {**codex_queue_readiness(), "scope": "LOCAL_PROCESS"}
    )
    capacity = facts.get("capacity")
    free = shutil.disk_usage(project).free
    disk_ready = (
        None
        if capacity is None
        else (
            capacity["measured_data_bytes"] <= capacity["cap_bytes"]
            and capacity["free_disk_bytes"] >= RECOVERY_HEADROOM_BYTES
        )
    )
    user = project / ".alphalattice/user"
    backups = sorted(path.name for path in (user / "backups").glob("*") if path.is_dir())
    specifications = (
        (
            "python",
            sys.version_info[:2] == (3, 12),
            {"version": sys.version.split()[0], "interpreter": sys.executable},
        ),
        ("uv", (uv := command_readiness([os.environ.get("UV", "uv"), "--version"]))["present"], uv),
        ("installed_leg", (installed := _installed_leg())["present"], installed),
        (
            "retrieval_runtime",
            retrieval["present"] and packs["ready"],
            {**retrieval, "packs": packs},
        ),
        (
            "background_notices",
            queue["present"],
            {"host": host, **queue, "delivery_proved": False},
        ),
        (
            "disk",
            disk_ready,
            {"free_disk_bytes": free, "capacity": capacity, "capacity_scope": facts["scope"]},
        ),
        (
            "cpu",
            (os.cpu_count() or 0) > 0,
            {
                "logical_cores": os.cpu_count(),
                "host": facts.get("cpu"),
                "host_scope": facts["scope"],
            },
        ),
        (
            "network",
            None if facts.get("network") is None else True,
            {
                "scope": facts["scope"],
                "setting": facts.get("network"),
                "operator_offline": os.environ.get("ALPHALATTICE_NETWORK_DISABLED") == "1",
            },
        ),
        ("user_layer", user.is_dir(), {"path": str(user)}),
        ("user_backup", bool(backups), {"latest": backups[-1] if backups else None}),
    )
    guidance = dict(HOST_DEPENDENCIES)
    if host == "claude-code":
        guidance["background_notices"] = (
            "Continue after background work ends",
            "Use background Bash completion notices, without another CLI.",
            "AGENT",
        )
    return [
        {
            "key": key,
            "present": present,
            "what_it_enables": guidance[key][0],
            "how_to_fill": guidance[key][1],
            "who_decides": guidance[key][2],
            "facts": {"scope": "LOCAL_PROCESS", **measured},
        }
        for key, present, measured in specifications
    ]


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
    """A milestone's request that the Host read the caller's Session usage; it carries none."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    project: Path
    event: dict[str, Any]

    @field_validator("project")  # type: ignore[untyped-decorator]
    @classmethod
    def absolute_project(cls, value: Path) -> Path:
        """Apply the binding door's path contract to a milestone's reading request."""
        return _absolute_project(value)


def admitted_session_project(workspace: Path, requested: Path, host: str) -> Path:
    """Admit a served workspace's configured project independently of the client's word.

    A workspace inside its agent project (``workspaces/`` of a checkout) finds that project
    up from itself, and the client's project must be it. A workspace kept elsewhere -- a
    second drive, an installed wheel's own folder -- finds none; the client's project is then
    admitted only when it holds this host's configure-owned declaration itself (FLOW-1).

    Args:
        workspace: The Host's own served workspace.
        requested: The client's nominated absolute project path.
        host: The admitted native host identifier.

    Returns:
        The admitted project.

    Raises:
        NativeBridgeError: The project differs, aliases a path, declares nothing or has
            unsafe declarations.
    """
    if requested.is_symlink() or requested.absolute() != requested.resolve():
        raise NativeBridgeError("native_bridge.project_path_invalid")
    try:
        project = session_project(workspace.resolve(), host)
    except NativeBridgeError as error:
        if str(error) != "native_bridge.project_declaration_missing":
            raise
        if not declares_product(requested.resolve(), host):
            raise
        project = requested.resolve()
    if requested.resolve() != project.resolve():
        raise NativeBridgeError("native_bridge.project_mismatch")
    declaration = project / (
        ".claude/settings.json" if host == "claude-code" else ".codex/config.toml"
    )
    if any(path.is_symlink() for path in (project, declaration.parent, declaration)):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    return project.resolve()


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


def attachment_preflight(
    project: Path | None = None, binding: NativeResearchBinding | None = None
) -> dict[str, Any]:
    """Read the binding's readiness and attach the named way forward."""
    from alphalattice.interface.local_application.cli_contract import refusal_words

    result = readiness(ROOT if project is None else project, binding)
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


def _install_declarations(host: str) -> None:
    """Copy shipped guidance unchanged; default host declarations register no product hooks."""
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
    documents: dict[Path, bytes] = {}
    for relative in sorted(set(inputs)):
        data = (RESOURCE_ROOT / relative).read_bytes()
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        if relative == declaration and path.exists():
            data = _merge_installed_declaration(data, path.read_bytes(), host)
        elif path.exists() and path.read_bytes() != data:
            raise NativeBridgeError("native_bridge.existing_configuration_differs")
        documents[path] = data
    for path, data in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
        elif path == ROOT / declaration and path.read_bytes() != data:
            path.write_bytes(data)


def _merge_installed_declaration(source: bytes, existing: bytes, host: str) -> bytes:
    """Admit product roles without replacing unrelated configuration or existing hooks."""
    if host == "claude-code":
        if not isinstance(json.loads(existing), dict):
            raise NativeBridgeError("native_bridge.configuration_path_invalid")
        return existing
    expected = tomllib.loads(source.decode()).get("agents", {})
    current = tomllib.loads(existing.decode()).get("agents", {})
    if not isinstance(current, dict):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    for name, profile in expected.items():
        if name in current and current[name] != profile:
            raise NativeBridgeError("native_bridge.existing_configuration_differs")
    additions = []
    for section in re.split(r"(?=^\[agents\.)", source.decode(), flags=re.MULTILINE):
        if section.startswith("[agents."):
            name = section.split("]", 1)[0].removeprefix("[agents.")
            if name not in current:
                additions.append(section)
    return existing if not additions else existing + b"\n" + "".join(additions).encode()


def strip_product_hooks(project: Path, host: str) -> int:
    """Remove the product's own retired lifecycle hook groups from a project it configures.

    An earlier configure could add ``SubagentStart`` and ``SubagentStop`` groups matching
    `PRODUCT_HOOK_MATCHER`; they called a command that no longer exists and kept the host's
    trust prompts. Only groups with exactly that matcher are removed; every other hook and
    setting stays as written.

    Args:
        project: The agent project being configured.
        host: ``claude-code`` or ``codex``.

    Returns:
        How many product hook groups were removed.

    Raises:
        NativeBridgeError: The declaration is a link or not the expected document.
    """
    path = project / (".claude/settings.json" if host == "claude-code" else ".codex/config.toml")
    if path.is_symlink() or path.parent.is_symlink():
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    if not path.is_file():
        return 0
    removed = 0
    if host == "claude-code":
        document = json.loads(path.read_text(encoding="utf-8"))
        hooks = document.get("hooks") if isinstance(document, dict) else None
        if not isinstance(hooks, dict):
            return 0
        for event in PRODUCT_HOOK_EVENTS:
            groups = hooks.get(event)
            if not isinstance(groups, list):
                continue
            kept = [
                group
                for group in groups
                if not (isinstance(group, dict) and group.get("matcher") == PRODUCT_HOOK_MATCHER)
            ]
            removed += len(groups) - len(kept)
            if kept:
                hooks[event] = kept
            else:
                hooks.pop(event, None)
        if removed:
            if not hooks:
                document.pop("hooks", None)
            path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        return removed
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    kept_lines: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        table = next(
            (name for name in PRODUCT_HOOK_EVENTS if line.strip() == f"[[hooks.{name}]]"), None
        )
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if table is not None and following == f"matcher = {json.dumps(PRODUCT_HOOK_MATCHER)}":
            removed += 1
            index += 2
            # The group's own command tables belong to it until the next other table.
            while index < len(lines) and (
                not lines[index].strip().startswith("[")
                or lines[index].strip() == f"[[hooks.{table}.hooks]]"
            ):
                index += 1
            while kept_lines and not kept_lines[-1].strip():
                kept_lines.pop()
            if kept_lines and index < len(lines):
                kept_lines.append("")
            continue
        kept_lines.append(line)
        index += 1
    if removed:
        result = "\n".join(kept_lines).rstrip("\n") + "\n"
        tomllib.loads(result)
        path.write_text(result, encoding="utf-8", newline="\n")
    return removed


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


def continue_here(project: Path, host: str) -> dict[str, Any]:
    """How the session that installed AlphaLattice continues the research (STOPS-1).

    A host loads AGENTS.md, the research Skill and the specialist cards on its own only in a
    session started in the configured folder. The installing session reads the first two by
    path and starts each specialist as a general subagent from its card, so a card's tool
    limits hold there by instruction; `open_session` is the command for a session where the
    host enforces them, offered as an option and never a step.
    """
    folder = project.resolve()
    claude = host == "claude-code"
    command = f'cd "{folder}"; claude' if claude else f'codex -C "{folder}"'
    skill = folder / (".claude" if claude else ".agents") / "skills/alphalattice-research"
    return {
        "read": [str(folder / "AGENTS.md"), str(skill / "SKILL.md")],
        "specialists": str(folder / (".claude/agents" if claude else ".codex/agents")),
        "open_session": command,
        "disclosure": (
            "AlphaLattice is set up and the research continues in this session. Its specialists "
            "start here from their cards as general subagents, so their tool limits hold by "
            f"instruction; for limits the host enforces, open a session with `{command}`."
        ),
    }


def declare_project(project: Path, host: str) -> None:
    """Write exact host-local project metadata; it grants neither trust nor native authorship."""
    if host not in HOSTS:
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    _create_or_match(
        PROJECT_DECLARATION_NAME,
        {"schema": PROJECT_DECLARATION_SCHEMA, "host": host},
        project,
        directory=".claude" if host == "claude-code" else ".codex",
    )


def bind_session(
    project: Path,
    *,
    host: str,
    session_id: str,
    workspace: Path,
    usage: str = "read",
    roles: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Bind this exact host/Session without replacing another Session's workspace (V568).

    The CLI and independent usage observer find this exact binding within the project.

    Args:
        project: The agent project, a checkout or a configured folder.
        host: The host running the session.
        session_id: The session, as its host names it.
        workspace: The workspace's folder.
        usage: ``off`` prevents native Session usage reads.
        roles: The admitted specialist roles; the project's own cards when omitted.

    Returns:
        The binding as written, with its project.

    Raises:
        NativeBridgeError: For a missing workspace, unsafe records, or a changed scope
            already bound to this exact host/Session.
    """
    if host not in HOSTS or usage not in {"read", "off"}:
        raise NativeBridgeError("native_bridge.binding_invalid")
    if (
        workspace.is_symlink()
        or (os.name == "nt" and workspace.is_junction())
        or workspace.absolute() != workspace.resolve()
    ):
        raise NativeBridgeError("native_bridge.binding_path_invalid")
    if not workspace.is_dir():
        raise NativeBridgeError("native_bridge.workspace_missing")
    folder = str(workspace.resolve())
    document = {
        "session_id": session_id,
        "workspace": folder,
        "roles": list(roles) if roles is not None else _roles(host, project),
        "host": host,
        # Named only when off, so a binding written before the switch reads the same.
        **({"usage": "OFF"} if usage == "off" else {}),
    }
    selected = (host, session_id)
    existing = NativeResearchBinding.read(project, session=selected)
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
    if existing is None:
        bindings, refusals = NativeResearchBinding.binding_entries(project)
        if len(bindings) + len(refusals) >= BINDING_RECORDS:
            raise NativeBridgeError("native_bridge.binding_limit_exceeded")
        legacy = project / ".codex" / BINDING_NAME
        # The first binding retains the established compatibility path. A later
        # Session gets its own slot; nothing migrates, replaces or detaches the first.
        if not legacy.exists() and not legacy.is_symlink() and not bindings and not refusals:
            name, directory = BINDING_NAME, ".codex"
        else:
            name = NativeResearchBinding.slot_path(project, selected).name
            directory = f".codex/{BINDING_DIRECTORY}"
        try:
            _create_or_match(name, document, project, directory=directory)
        except NativeBridgeError as error:
            # A concurrent bind of this exact Session may already have admitted the
            # same scope. Keep its checkpoint rather than generating a new one.
            if str(error) != "native_bridge.existing_configuration_differs":
                raise
            winner = NativeResearchBinding.read(project, session=selected)
            if winner is None and name == BINDING_NAME:
                # Two distinct Sessions can both see the project's first empty
                # record. The loser gets its own slot, preserving the winner's file.
                try:
                    _create_or_match(
                        NativeResearchBinding.slot_path(project, selected).name,
                        document,
                        project,
                        directory=f".codex/{BINDING_DIRECTORY}",
                    )
                except NativeBridgeError as slot_error:
                    if str(slot_error) != "native_bridge.existing_configuration_differs":
                        raise
                winner = NativeResearchBinding.read(project, session=selected)
            if winner is None or (
                winner.workspace.resolve() != workspace.resolve()
                or winner.roles != tuple(document["roles"])
                or winner.usage != ("OFF" if usage == "off" else "READ")
            ):
                raise
    binding = NativeResearchBinding.read(project, session=selected)
    preflight = attachment_preflight(project, binding)
    return {
        "status": "BOUND",
        "session_id": session_id,
        "host": host,
        "workspace": folder,
        "project": str(project),
        "attachment_preflight": preflight,
        "detail": preflight["detail"],
        "next_action": preflight["next_action"],
    }


AUTOBIND_ROOT = Path("runtime") / "native-sessions"
"""Where the Host keeps the bindings it makes itself, inside the workspace (AUTOBIND)."""


def autobind_root(workspace: Path) -> Path:
    """The workspace's own binding folder, read like a project's."""
    return workspace / AUTOBIND_ROOT


def autobind_session(
    workspace: Path, host: str, session_id: str, *, project: Path | None = None
) -> tuple[Path, str]:
    """Bind the Session a request names to this workspace, with no step and no configure.

    Under the person's hands-off rule a Session is bound by working: the Host writes the
    binding in the workspace's own folder, with the product's shipped specialist roles and
    reading on (the workspace switch still governs reading). A Codex child binds its lead,
    found up its own rollout's spawn chain; a Claude Code subagent carries its lead's id.

    Args:
        workspace: The workspace the Session works on.
        host: Its host.
        session_id: The Session the request names.
        project: The configured project above the workspace, where the binding is kept with
            its own cards so commands from it may omit the workspace; else the
            workspace's own folder with the shipped roles.

    Returns:
        The binding folder and the Session bound.
    """
    lead = session_id
    if host == "codex":
        for _ in range(SPAWN_HOPS):
            spawn = codex_thread_spawn(lead)
            if spawn is None:
                break
            lead = spawn.parent_thread_id
    if project is not None:
        bind_session(project, host=host, session_id=lead, workspace=workspace)
        return project, lead
    root = autobind_root(workspace)
    bind_session(root, host=host, session_id=lead, workspace=workspace, roles=tuple(_roles(host)))
    return root, lead


def _configuration(host: str = "codex") -> None:
    python = (
        Path(sys.executable)
        if INSTALLED
        else ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if not python.is_file():
        raise NativeBridgeError("native_bridge.local_environment_missing")
    if not _roles(host):
        raise NativeBridgeError("native_bridge.roles_missing")


def unbind_session(
    project: Path, *, session_id: str | None, host: str | None = None
) -> dict[str, object]:
    """Detach this Session's own record; other Session bindings and research stay (V586).

    An agent removes only its exact host/Session record. Outside any agent Session,
    a person may detach the sole record; multiple records require an exact owning Session.
    The research is unchanged: a binding is a session's default workspace and the scope of its
    observation, never a record.

    Args:
        project: The agent project holding the binding.
        session_id: The session removing it; None for the person.
        host: The actual host; older shims may omit it only for an unambiguous Session.

    Returns:
        ``NOT_BOUND``, or ``DETACHED`` with the session the removed binding named.

    Raises:
        NativeBridgeError: ``native_bridge.session_mismatch`` when the binding names another
            session than the one removing it; the binding's own read refusals.
    """
    if session_id is None:
        binding = NativeResearchBinding.read(project)
    elif host is not None:
        binding = NativeResearchBinding.read(project, session=(host, session_id))
    else:
        # Compatibility shims did not supply a host. The Session must still name
        # one exact record; an identifier shared by hosts cannot select either.
        candidates = [
            each
            for each in NativeResearchBinding.bindings(project)
            if each.session_id == session_id
        ]
        if len(candidates) > 1:
            raise NativeBridgeError("native_bridge.binding_ambiguous")
        binding = candidates[0] if candidates else None
    if binding is None:
        if session_id is not None and NativeResearchBinding.bindings(project):
            raise NativeBridgeError("native_bridge.session_mismatch")
        return {"status": "NOT_BOUND", "project": str(project)}
    if session_id is not None and binding.session_id != session_id:
        raise NativeBridgeError("native_bridge.session_mismatch")
    binding.record_path(project).unlink()
    return {
        "status": "DETACHED",
        "project": str(project),
        "session_id": binding.session_id,
        "host": binding.host,
        "research_unchanged": True,
    }


def _create_or_match(
    name: str,
    document: dict[str, Any],
    project: Path | None = None,
    *,
    directory: str = ".codex",
) -> None:
    root = ROOT if project is None else project
    path = root / directory / name
    parents = (
        path,
        *(parent for parent in path.parents if parent == root or parent.is_relative_to(root)),
    )
    if any(
        parent.is_symlink() or (os.name == "nt" and parent.is_junction()) for parent in parents
    ) or not path.resolve().is_relative_to(root.resolve()):
        raise NativeBridgeError("native_bridge.configuration_path_invalid")
    data = (json.dumps(document, indent=2) + "\n").encode()
    # A Claude Code project configured from the installed command holds no `.codex` yet.
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        with path.open("rb") as stream:
            existing = stream.read(len(data) + 1)
        if existing != data:
            raise NativeBridgeError("native_bridge.existing_configuration_differs") from None


def main() -> int:
    """Configure, bind or inspect the project: zero on success, two on refusal."""
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, help="The writable agent project; cwd when installed."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    configure = commands.add_parser(
        "configure", help="Install role guidance and declare a project without product hooks."
    )
    configure.add_argument("--host", choices=HOSTS, default="codex")
    bind = commands.add_parser("bind", help="Bind this bridge to an explicit session/workspace.")
    bind.add_argument("--session-id", required=True)
    bind.add_argument("--workspace", required=True, type=Path)
    bind.add_argument("--host", choices=HOSTS, default="codex", help="The foreground native host.")
    bind.add_argument(
        "--usage",
        choices=[value.lower() for value in USAGE_READINGS],
        default="read",
        help="off: read no native Session file for usage.",
    )
    unbind = commands.add_parser(
        "unbind", help="Detach only the named session; research is unchanged."
    )
    unbind.add_argument("--session-id", required=True)
    unbind.add_argument("--host", choices=HOSTS, help="The exact native host of this Session.")
    doctor = commands.add_parser("doctor", help="Read this Session's binding and readiness.")
    doctor.add_argument(
        "--host", choices=HOSTS, help="The host to inspect; by default the bound one, else codex."
    )
    args = parser.parse_args()
    if args.project is not None:
        ROOT = args.project.resolve()
    result: dict[str, Any]
    doctor_command = _doctor_command()
    try:
        if args.command == "configure":
            if INSTALLED:
                _install_declarations(args.host)
            removed = strip_product_hooks(ROOT, args.host)
            _configuration(args.host)
            declare_project(ROOT, args.host)
            result = {
                "status": "LOCAL_DECLARATIONS_VALIDATED",
                "host": args.host,
                "retired_hook_groups_removed": removed,
                "trust_changed": False,
                "next": (
                    "Continue the research in this session: read the files in `read` by path, "
                    "start each specialist from its card in `specialists`, and tell the person "
                    "`disclosure` in one line."
                ),
                "continue_here": continue_here(ROOT, args.host),
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
            result = unbind_session(ROOT, session_id=args.session_id, host=args.host)
        else:
            from alphalattice.interface.local_application.cli_contract import agent_session

            session = agent_session(os.environ)
            if session is not None and args.host is not None and args.host != session[0]:
                raise NativeBridgeError("native_bridge.session_mismatch")
            binding = NativeResearchBinding.read(ROOT, session=session)
            host = args.host or (binding.host if binding is not None else "codex")
            preflight = attachment_preflight(ROOT, binding)
            result = {
                **preflight,
                "session_bound": binding is not None,
                "host": binding.host if binding is not None else None,
                "usage_reading": binding.usage if binding is not None else None,
                "roles": _roles(host),
                "attachment_preflight": preflight,
                "host_readiness": host_readiness(ROOT, host, binding),
            }
        print(json.dumps(result))
        return 2 if result["status"] == "REFUSED" else 0
    except NativeBridgeError as error:
        print(
            json.dumps(
                {**setup_failure(error), "reason": str(error), "next_commands": doctor_command}
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
            "next_commands": doctor_command,
        }
        print(json.dumps(refusal))
        return 2
    except Exception as error:
        print(json.dumps({**setup_failure(error), "next_commands": doctor_command}))
        return 2


def _doctor_command() -> dict[str, list[str]]:
    """The binding's own read-only check, every refusal's way on, as the person runs it."""
    command = (
        [
            sys.executable,
            "-m",
            "alphalattice.interface.local_application.native_setup",
            "--project",
            str(ROOT),
            "doctor",
        ]
        if INSTALLED
        else [sys.executable, "scripts/native_research.py", "doctor"]
    )
    return {"doctor": command}


if __name__ == "__main__":
    raise SystemExit(main())
