"""The console's standard-library client for finite, caller-independent Host commands."""

from __future__ import annotations

import base64
import http.client
import json
import os
import shutil
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Final, NamedTuple
from urllib.parse import urlsplit

CONNECTION_NAME: Final = "local-research-connection.json"
CLIENT_HEADER: Final = "X-Alphalattice-Client"
WORKSPACE_HEADER: Final = "X-Alphalattice-Workspace"
INSTANCE_HEADER: Final = "X-Alphalattice-Instance"
OPERATIONS_PATH: Final = "/api/client/operations"
CLI_OPERATION: Final = "CLI_INVOKE"
AGENT_SESSION_VARIABLES: Final = (
    ("claude-code", "CLAUDE_CODE_SESSION_ID"),
    ("codex", "CODEX_THREAD_ID"),
)
"""The native environment variable naming each vendor's session."""
CLI_ENVIRONMENT: Final = ("ALPHALATTICE_SHELL", "MSYSTEM", "TRACEPARENT", "BAGGAGE")
MAXIMUM_REQUEST_BODY_BYTES: Final = 128 * 1024
DEFAULT_HTTP_WAIT_SECONDS: Final = 120.0
_DELIVERY_LOST: Final = "local_client.connection_lost_task_may_still_run"

# Extra positional arguments and options are opt-in at the same boundary as the command.
# No input/output files, follows, launchers or Session binding work cross this operation.
_COMMANDS: Final = {
    ("--help",): (0, frozenset()),
    ("-h",): (0, frozenset()),
    ("activity", "list"): (0, frozenset({"--after", "--limit"})),
    ("goal", "show"): (1, frozenset()),
    ("input", "list"): (0, frozenset()),
    ("network", "show"): (0, frozenset()),
    ("operation", "list"): (0, frozenset({"--schema-operation"})),
    ("study", "list"): (0, frozenset()),
    ("study", "show"): (1, frozenset()),
    ("task", "list"): (0, frozenset()),
    ("task", "show"): (1, frozenset()),
    ("workspace", "show"): (0, frozenset()),
}
_WORKSPACE_OPTION: Final = "--workspace"
_GLOBAL_VALUES: Final = frozenset(
    {_WORKSPACE_OPTION, "--lang", "--view", "--goal", "--request-timeout"}
)
_VIEW_VALUES: Final = frozenset({"--section", "--next-from"})
_VIEW_SWITCHES: Final = frozenset({"--list-next"})


class Invocation(NamedTuple):
    """A CLI form whose streams and effects fit an ordinary Host response."""

    workspace: Path
    timeout: float


class DeliveryLost(ValueError):
    """The Host may have executed the command, so it cannot be run again."""


def workspace_connection_key(workspace: Path) -> str:
    """Bind the request to the canonical workspace location."""
    return sha256(os.fsencode(os.path.normcase(str(workspace.resolve())))).hexdigest()


def connection_path(workspace: Path) -> Path:
    """Locate the descriptor without following a link outside the workspace."""
    root = workspace.resolve()
    path = root / "runtime" / CONNECTION_NAME
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("local_client.connection_path_escapes_workspace")
    return path


def connection_port(
    schema_id: str, workspace: str, workspace_id: str, url: str, instance: str, token: str
) -> int:
    """Validate the connection record and admit only its plain IPv4 loopback port."""
    values = (schema_id, workspace, workspace_id, url, instance, token)
    if (
        not all(isinstance(value, str) for value in values)
        or schema_id != "local-research-connection"
        or not workspace_id
        or not 16 <= len(instance) <= 128
        or not 32 <= len(token) <= 128
    ):
        raise ValueError("local_client.connection_unreadable")
    target = urlsplit(url)
    if (
        not Path(workspace).is_absolute()
        or target.scheme != "http"
        or target.hostname != "127.0.0.1"
        or not target.port
        or target.username is not None
        or target.password is not None
        or target.path not in ("", "/")
        or target.query
        or target.fragment
        or url.rstrip("/") != f"http://127.0.0.1:{target.port}"
    ):
        raise ValueError("local_client.connection_not_local")
    return target.port


def select(argv: list[str], cwd: Path) -> Invocation | None:
    """Select a declared form and its workspace before either side runs a command."""
    words: list[str] = []
    options: dict[str, str] = {}
    remaining = iter(argv)
    for argument in remaining:
        name, equal, value = argument.partition("=")
        if name in _VIEW_SWITCHES:
            if equal:
                return None
            options[name] = ""
        elif name.startswith("--") and name != "--help":
            value = value if equal else next(remaining, "")
            if not value or value.startswith(("-", "@")):
                return None
            options[name] = value
        else:
            if argument.startswith("@") or (
                name.startswith("-") and argument not in {"-h", "--help"}
            ):
                return None
            words.append(argument)
    command = tuple(words[:2])
    rule = _COMMANDS.get(command)
    if rule is None or len(words) - len(command) > rule[0]:
        return None
    help_only = command in {("-h",), ("--help",)}
    admitted = _GLOBAL_VALUES if help_only else _GLOBAL_VALUES | _VIEW_VALUES | _VIEW_SWITCHES
    if options.keys() - admitted - rule[1]:
        return None
    named = options.get(_WORKSPACE_OPTION)
    if named is None and not help_only:
        return None
    try:
        timeout = float(options.get("--request-timeout", DEFAULT_HTTP_WAIT_SECONDS))
        workspace = (cwd / named if named is not None else cwd).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not 0 < timeout <= 600:
        return None
    return Invocation(workspace, timeout)


def agent_session_values(environ: Mapping[str, str]) -> dict[str, str]:
    """Read each vendor's raw session value before its caller interprets it."""
    return {vendor: environ.get(name, "") for vendor, name in AGENT_SESSION_VARIABLES}


def _settings() -> dict[str, str]:
    return {key: os.environ[key] for key in CLI_ENVIRONMENT if key in os.environ}


def _styles() -> list[list[str | None]]:
    streams = (sys.stdout, sys.stderr)  # noqa: TID251  # match the console encoding
    return [[stream.encoding, stream.errors] for stream in streams]


def _refused_before_dispatch(status: int, response: dict[str, object]) -> bool:
    return (
        status in {403, 404, 413}
        or response.get("cli") == "local"
        or response.get("failure_code") == "local_client.request_invalid"
    )


def invoke(argv: list[str]) -> int | None:
    """Print the Host's bytes and return its exit code, or select the full CLI."""
    if any(agent_session_values(os.environ).values()):
        # Launch counting and named native bindings are caller-side file operations.
        return None
    # The portable launcher is the caller's existing prefix in this case. Probing
    # an off-PATH tool can start uv, so that caller-side action stays on the full path.
    if (
        "UV_RUN_RECURSION_DEPTH" in os.environ or shutil.which("alphalattice") is None
    ) and shutil.which("uv") is not None:
        return None
    try:
        cwd = Path.cwd()
        selected = select(argv, cwd)
        if selected is None:
            return None
        with connection_path(selected.workspace).open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            return None
        record = json.loads(raw)
        port = connection_port(
            record.get("schema_id", "local-research-connection"),
            record["workspace"],
            record["workspace_id"],
            record["url"],
            record["instance"],
            record["token"],
        )
        if Path(record["workspace"]).resolve() != selected.workspace:
            return None
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None

    body = json.dumps(
        {
            "operation": CLI_OPERATION,
            "argv": argv,
            "cwd": str(cwd),
            "stdin": None,  # Admitted one-response forms consume no input.
            "streams": _styles(),
            "environment": _settings(),
        }
    ).encode("ascii")
    if len(body) > MAXIMUM_REQUEST_BODY_BYTES:
        return None
    headers = {
        "Content-Type": "application/json",
        CLIENT_HEADER: record["token"],
        WORKSPACE_HEADER: workspace_connection_key(selected.workspace),
        INSTANCE_HEADER: record["instance"],
    }
    exchange = http.client.HTTPConnection("127.0.0.1", port, timeout=selected.timeout)
    try:
        exchange.connect()
    except OSError:
        exchange.close()
        return None
    try:
        exchange.request("POST", OPERATIONS_PATH, body=body, headers=headers)
        response = exchange.getresponse()
        answer = json.loads(response.read())
        if not isinstance(answer, dict):
            raise ValueError(_DELIVERY_LOST)
        if _refused_before_dispatch(response.status, answer):
            return None
        code = answer["exit"]
        if response.status != 200 or answer.get("cli") != "complete" or type(code) is not int:
            raise ValueError(_DELIVERY_LOST)
        stdout = base64.b64decode(answer["stdout"], validate=True)
        stderr = base64.b64decode(answer["stderr"], validate=True)
    except (KeyError, OSError, TypeError, ValueError, http.client.HTTPException) as error:
        raise DeliveryLost(_DELIVERY_LOST) from error
    finally:
        exchange.close()
    streams = (sys.stdout, sys.stderr)  # noqa: TID251  # write the returned console bytes
    for output, data in zip(streams, (stdout, stderr), strict=True):
        if data:
            output.buffer.write(data)
            output.flush()
    return code
