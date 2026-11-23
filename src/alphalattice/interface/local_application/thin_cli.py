"""The console's standard-library client for finite, caller-independent Host commands.

One Host stays resident per data root: it holds the root's lock (machine/host.lock) and writes
its endpoint record beside it (machine/host-endpoints/<workspace key>.json), so every checkout's
CLI on that root finds it. A command none answers starts the Host detached here (`start_host`,
the CLI's own `serve --detach`, which returns once the Host serves) and then sends, so a cold
command loads no product module either; a command this client declines (a bound session's) takes
the same start in the full CLI's Host-backed operation. The Host retires after its idle bound
(web.DEFAULT_IDLE_TIMEOUT_SECONDS), never while a Task runs, and the next command starts it.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Final, NamedTuple
from urllib.parse import urlsplit

from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.config import data_root
from alphalattice.kernel.door import (
    CLI_OPERATION as CLI_OPERATION,
)
from alphalattice.kernel.door import (
    OPERATIONS_PATH as OPERATIONS_PATH,
)
from alphalattice.kernel.door import (
    request_bytes,
)
from alphalattice.kernel.identity import identity_sha256

ENDPOINT_FOLDER: Final = "host-endpoints"
"""The data root's endpoint records, one per served workspace, beside its lock."""
MANIFEST_NAME: Final = "research-workspace.json"
"""A workspace's manifest: only an initialized workspace starts a Host from a read."""
START_SECONDS: Final = 330.0
"""The detached start's own ready wait (web_launcher.READY_SECONDS, 300 s) and its exit."""
CLIENT_HEADER: Final = "X-Alphalattice-Client"
WORKSPACE_HEADER: Final = "X-Alphalattice-Workspace"
INSTANCE_HEADER: Final = "X-Alphalattice-Instance"
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
    starts: bool
    """Whether no answer starts the Host: a named workspace's command, never `--help`."""


class DeliveryLost(ValueError):
    """The Host may have executed the command, so it cannot be run again."""


def workspace_connection_key(workspace: Path) -> str:
    """Bind the request to the canonical workspace location."""
    return identity_sha256(os.fsencode(os.path.normcase(str(workspace.resolve())))).hexdigest()


def connection_path(workspace: Path) -> Path:
    """Locate the endpoint record of `workspace`'s Host on this process's data root without
    following a link out of its folder."""
    return endpoint_path(data_root(), workspace)


def endpoint_path(root: Path, workspace: Path) -> Path:
    """The endpoint record a Host serving `workspace` on data root `root` writes, beside the
    root's lock under the workspace's key; the record names the workspace it serves."""
    folder = root.resolve() / "machine" / ENDPOINT_FOLDER
    path = folder / f"{workspace_connection_key(workspace)}.json"
    if path.is_symlink() or not path.resolve().is_relative_to(folder):
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
    return Invocation(workspace, timeout, starts=not help_only)


def agent_session_values(environ: Mapping[str, str]) -> dict[str, str]:
    """Read each vendor's raw session value before its caller interprets it."""
    return {vendor: environ.get(name, "") for vendor, name in AGENT_SESSION_VARIABLES}


def _settings() -> dict[str, str]:
    values = process_config().public
    return {key: values[key] for key in CLI_ENVIRONMENT if key in values}


def _styles() -> list[list[str | None]]:
    streams = (sys.stdout, sys.stderr)  # noqa: TID251  # match the console encoding
    return [[stream.encoding, stream.errors] for stream in streams]


def _endpoint(path: Path) -> dict[str, str] | None:
    """An endpoint record, validated; None when no Host has written it."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(4097)
    except FileNotFoundError:
        return None
    if len(raw) > 4096:
        raise ValueError("local_client.connection_unreadable")
    record = json.loads(raw)
    connection_port(
        record.get("schema_id", "local-research-connection"),
        record["workspace"],
        record["workspace_id"],
        record["url"],
        record["instance"],
        record["token"],
    )
    return dict(record)


def _connect(record: Mapping[str, str], timeout: float) -> http.client.HTTPConnection | None:
    """A connection to the record's Host, or None when nothing answers at its port."""
    port = urlsplit(record["url"]).port
    exchange = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        exchange.connect()
    except OSError:
        exchange.close()
        return None
    return exchange


def answers(url: str, timeout: float) -> bool:
    """Whether a Host listens at an endpoint record's `url`: a crashed Host's record outlives it.
    A TCP connect, opened and closed, asks it; the request keeps its own connection."""
    port = urlsplit(url).port
    if port is None:
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


_ATTEMPTS: ContextVar[set[tuple[Path, Path]] | None] = ContextVar("host_starts", default=None)
"""The (data root, workspace) pairs the running console command has started a Host for."""


@contextmanager
def one_command() -> Iterator[None]:
    """One console command's scope, which the entry holds: it makes one Host start attempt per
    workspace, whichever of its paths (this one, or the full CLI's) reaches the start first."""
    scope = _ATTEMPTS.set(set())
    try:
        yield
    finally:
        _ATTEMPTS.reset(scope)


def start_host(workspace: Path) -> bool:
    """Start the data root's Host detached for `workspace`, and return once it serves.

    It runs the CLI's own `serve --detach` as a child, so this process loads no product module;
    that start returns when the Host serves (its ready signal) or refuses, the root's lock
    deciding whether this workspace may have it. Only an initialized workspace starts a Host
    from a command, once in a console command's scope (`one_command`).
    """
    attempts, attempt = _ATTEMPTS.get(), (data_root().resolve(), workspace)
    if not (workspace / MANIFEST_NAME).is_file():
        return False
    if attempts is not None:
        if attempt in attempts:
            return False
        attempts.add(attempt)
    command = [
        sys.executable,
        "-m",
        "alphalattice.control.product_host.composition.entry",
        "serve",
        "--workspace",
        str(workspace),
        "--detach",
        "--no-browser",
    ]
    try:
        started = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,  # the launch line carries the session's key
            stderr=subprocess.DEVNULL,
            timeout=START_SECONDS,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return started.returncode == 0


def _serving(selected: Invocation) -> tuple[dict[str, str], http.client.HTTPConnection] | None:
    """The record and a connection to the Host serving the selected workspace, starting the data
    root's Host when none answers and the command starts one; None when no Host serves it, and
    the full CLI answers."""
    found = _answering(selected.workspace, selected.timeout)
    if found is None and selected.starts and start_host(selected.workspace):
        found = _answering(selected.workspace, selected.timeout)
    return found


def _answering(
    workspace: Path, timeout: float
) -> tuple[dict[str, str], http.client.HTTPConnection] | None:
    """The record and a connection to the Host serving `workspace` on this data root, or None."""
    record = _endpoint(connection_path(workspace))
    exchange = None if record is None else _connect(record, timeout)
    if record is None or exchange is None:
        return None
    if Path(record["workspace"]).resolve() != workspace:
        exchange.close()  # a record under this key naming another workspace is not this one's
        return None
    return record, exchange


def _refused_before_dispatch(status: int, response: dict[str, object]) -> bool:
    return (
        status in {403, 404, 413}
        or response.get("cli") == "local"
        or response.get("failure_code") == "local_client.request_invalid"
    )


def invoke(argv: list[str]) -> int | None:
    """Print the Host's bytes and return its exit code, or select the full CLI."""
    if any(agent_session_values(process_config().public).values()):
        # Launch counting and named native bindings are caller-side file operations.
        return None
    # The portable launcher is the caller's existing prefix in this case. Probing
    # an off-PATH tool can start uv, so that caller-side action stays on the full path.
    if (
        "UV_RUN_RECURSION_DEPTH" in process_config().public or shutil.which("alphalattice") is None
    ) and shutil.which("uv") is not None:
        return None
    try:
        cwd = Path.cwd()
        selected = select(argv, cwd)
        if selected is None:
            return None
        serving = _serving(selected)
    except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if serving is None:
        return None
    return send(*serving, argv, cwd, selected.workspace)


def send(
    record: Mapping[str, str],
    exchange: http.client.HTTPConnection,
    argv: list[str],
    cwd: Path,
    workspace: Path,
) -> int | None:
    """Send one command to the Host `record` names over `exchange`, print its bytes and return
    its exit code; None when the Host refused it before dispatch (the full CLI answers)."""
    body = request_bytes(
        {
            "operation": CLI_OPERATION,
            "argv": argv,
            "cwd": str(cwd),
            "stdin": None,  # Admitted one-response forms consume no input.
            "streams": _styles(),
            "environment": _settings(),
        }
    )
    if len(body) > MAXIMUM_REQUEST_BODY_BYTES:
        exchange.close()
        return None
    headers = {
        "Content-Type": "application/json",
        CLIENT_HEADER: record["token"],
        WORKSPACE_HEADER: workspace_connection_key(workspace),
        INSTANCE_HEADER: record["instance"],
    }
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
