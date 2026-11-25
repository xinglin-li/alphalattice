"""Start the local Portfolio Research service and open it in the system browser.

A development launcher and nothing more. It is not an installer, a native shell,
an updater, a bundle, a signing path or a distribution claim; Gate 9D owns all of
those and none of them are started here.

    python scripts/run_local_portfolio_web.py --workspace <path>

The workspace's identity-bound manifest owns its id, installed artifacts,
catalog and default strategy selection. The launcher accepts none of those as a
second configuration surface.

The service binds loopback on an operating-system-chosen port and prints the URL
it actually got. `--no-browser` prints the URL and waits, which is what a headless
check wants; `--port` pins a port for someone who needs a stable bookmark and
accepts the collision risk that comes with it.

The workspace's Evidence and CRO section admits the official SEC source while it runs,
from the workspace's consent and network control (`alphalattice evidence-consent set`,
`alphalattice network set`); nothing about it is a launch option.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

from alphalattice.capabilities.domain_contracts import spawn
from alphalattice.capabilities.domain_contracts.project_layout import runtime_json
from alphalattice.capabilities.domain_contracts.waits import pause
from alphalattice.interface.local_application.cli_contract import (
    ANSWER_LANGUAGE as ANSWER_LANGUAGE,
)
from alphalattice.interface.local_application.cli_contract import (
    terminal_serve_command,
    worded,
)
from alphalattice.interface.local_application.failure_codes import safe_failure_code
from alphalattice.interface.local_application.web import (
    DEFAULT_IDLE_TIMEOUT_SECONDS,
    LAUNCH_KEY_VARIABLE,
    OPEN_THE_WORKBENCH,
)
from alphalattice.kernel.config import consume_secret
from alphalattice.kernel.config import current as process_config

from .process_job import ChildJob
from .workspace_folder import RESEARCH_WORKSPACE_MANIFEST_NAME, holds_foreign_files

# The Host's session (and the numerical libraries it brings) is imported in `main`, never here:
# a Host run as `-m` this module has each spawned worker re-import it first, before the worker
# sets its numerical libraries to one thread (child._worker_started), and a library loaded then
# keeps a thread and its buffers for every processor (about 1.8 GB of commit a worker).

STOPPING: dict[str, object] = {
    "status": "HOST_STOPPING",
    "detail": "This Host is stopping its Tasks at safe points and will keep their completed work "
    "for resumption. It joins the workers before releasing the workspace. Read Task status "
    "while it finishes; serve it again to resume.",
    "finishing_task_ids": [],
    "next_action": "SERVE_AGAIN_WHEN_WORK_NEEDS_IT",
}
"""The answer to `host stop`, given before the Host begins to stop."""


def _windows_stdin(descriptor: int) -> tuple[Any, Any, int]:
    if sys.platform != "win32":
        raise OSError("Windows stdin inspection is unavailable")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
    return kernel32, handle, kernel32.GetFileType(handle)


def _stdin_ties() -> bool:
    """Whether something holds standard input open, so `--stop-on-stdin` can tie the Host's life
    to it: a console, or a pipe whose writer is still there.

    An agent's background shell gives a launch none (the NUL device, a pipe closed at once), and
    a Host that took that end for a stop would stop as it started. With nothing held there is no
    tie: the Host serves until its process ends, as without the option.
    """
    try:
        descriptor = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):  # a stream handed in, read as it comes
        return True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32, handle, kind = _windows_stdin(descriptor)
        if kind == 3:  # FILE_TYPE_PIPE: PeekNamedPipe fails once the writer is gone
            return bool(kernel32.PeekNamedPipe(handle, None, 0, None, None, None))
        # FILE_TYPE_CHAR: a console takes a console mode; the NUL device does not.
        return kind == 2 and bool(kernel32.GetConsoleMode(handle, ctypes.byref(wintypes.DWORD())))
    if os.isatty(descriptor):
        return True
    import fcntl
    import select
    import stat
    import struct
    import termios

    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        return False
    if not select.select([descriptor], [], [], 0)[0]:
        return True  # open and quiet
    waiting = fcntl.ioctl(descriptor, termios.FIONREAD, bytes(4))
    return bool(struct.unpack("i", waiting)[0])  # readable with nothing waiting is the end


def _wait_for_stop_on_stdin() -> None:
    """Return at a 'stop' line or at the end of standard input.

    On Windows a synchronous read left pending on a pipe holds every other operation on it,
    and a process the Host starts queries the standard input it inherits as it starts: the
    Task worker (W10) waited for the read, and the read for the stop. So a pipe is peeked and
    read only when bytes wait; anything else is read as it always was.
    """
    try:
        descriptor: int | None = sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):  # a stream with no descriptor
        descriptor = None
    if sys.platform == "win32" and descriptor is not None:
        import ctypes
        from ctypes import wintypes

        kernel32, handle, kind = _windows_stdin(descriptor)
        if kind == 3:  # FILE_TYPE_PIPE
            waiting, pending = wintypes.DWORD(), b""
            while kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(waiting), None):
                if not waiting.value:
                    pause(0.2)
                    continue
                chunk = os.read(descriptor, waiting.value)
                if not chunk:
                    return
                *lines, pending = (pending + chunk).split(b"\n")
                if any(line.strip() == b"stop" for line in lines):
                    return
            return  # the writer closed the pipe: the end of standard input
    for line in sys.stdin:
        if line.strip() == "stop":
            return


READY_SECONDS = 300
"""How long a detached launch waits for its Host to report it serves."""
HANDED_KEY = "key=<from the launcher>"
"""What a detached Host prints in place of the launch key its launcher made and holds."""
READY_VARIABLE = "ALPHALATTICE_HOST_READY"
"""Where a detached launch names the pipe its Host writes one byte to once it serves."""
_OBSERVED = tuple(
    words.partition("{reason}")[0]
    for words in runtime_json("config/registries/messages.json")["entries"]["host.memory_observed"][
        "words"
    ].values()
)
"""How a Host's start line that observes its memory budget begins, in each language."""


def _refusal(
    failure_code: str,
    detail: str,
    next_action: str,
    *,
    message_id: str | None = None,
    terminal_command: list[str] | None = None,
    terminal_routes: list[dict[str, str]] | None = None,
    launch_context: dict[str, int | None] | None = None,
    cause: str | None = None,
    host_log: str | None = None,
    workspaces: list[str] | None = None,
) -> int:
    """Print a launch's refusal as an answer: its code, its words and its way on."""
    answer: dict[str, object] = {
        "status": "REFUSED",
        "failure_code": failure_code,
        "detail": detail,
        "next_action": next_action,
    }
    if message_id is not None:
        answer["message_id"] = message_id
    if terminal_command is not None:
        answer["terminal_command"] = terminal_command
    if terminal_routes is not None:
        answer["terminal_routes"] = terminal_routes
    if launch_context is not None:
        answer["launch_context"] = launch_context
    if cause is not None:
        answer["cause"] = cause
    if host_log is not None:
        answer["host_log"] = host_log
    if workspaces is not None:
        answer["workspaces"] = workspaces
    print(json.dumps(answer), flush=True)
    return 2


def attached_child(command: list[str], *, env: dict[str, str]) -> int:
    """Run the retrieval child with its caller's streams and parent-owned lifetime."""
    try:
        return spawn.attached(
            command,
            env=env,
            job_factory=ChildJob,
            process_kind="attached_retrieval_host_child",
        )
    except OSError:
        return _refusal(
            "local_web.child_ownership_unavailable",
            str(worded("host.child_ownership.unavailable")),
            "RUN_TERMINAL_COMMAND_FROM_ANOTHER_TERMINAL",
            message_id="host.child_ownership.unavailable",
            terminal_command=command,
        )


def _spawn_detached(
    command: list[str], log: Any, key: str, ready: int | None = None
) -> subprocess.Popen[bytes]:
    """Start the Host apart from this process: its own process group (session), no console
    window and no standard input, and out of this process's Job where the Job allows it (a
    Codex turn ends its shell's children, by its tree or its Job).

    Denied Job breakaway raises; no still-held child is started. ``ready``, a pipe's write
    end, is the one descriptor the Host inherits: it writes there once it serves.
    """
    # The new interpreter imports this same source, as `entry.serve` gives its child.
    source = str(Path(__file__).resolve().parents[4])
    environment = process_config().child_environment()
    inherited = environment.get("PYTHONPATH")
    environment |= {
        "PYTHONPATH": source + os.pathsep + inherited if inherited else source,
        "PYTHONFAULTHANDLER": "1",
        LAUNCH_KEY_VARIABLE: key,
        **({} if ready is None else {READY_VARIABLE: spawn.descriptor_name(ready)}),
    }
    return spawn.detached(command, log=log, env=environment, inherit=ready)


def _names_a_host(workspace: Path) -> bool:
    """Whether a Host left its connection here: whether that Host serves is the launch's to read."""
    from alphalattice.interface.local_application import client  # lazy: only a refusal's path.

    try:
        return client.LocalResearchConnection.path(workspace).exists()
    except client.LocalResearchClientError:
        return True  # a record that leaves the workspace is refused by name at that read


def _workspace_refusal(workspace: Path) -> int | None:
    """Refuse an unusable workspace before a detached Host is launched."""
    # Checked before any launch: a containment refusal would hand on a command that cannot serve.
    try:
        if holds_foreign_files(workspace) and not _names_a_host(workspace):
            message_id = "host.start.not_a_workspace"
            inside = workspace.resolve()
            found = inside.glob(f"workspaces/*/{RESEARCH_WORKSPACE_MANIFEST_NAME}")
            return _refusal(
                "research_workspace.nonempty_manifest_absent",
                str(worded(message_id)),
                "SERVE_A_RESEARCH_WORKSPACE",
                message_id=message_id,
                workspaces=sorted(str(manifest.parent) for manifest in found),
            )
    except OSError as error:
        return _refusal(
            "local_web.service_not_started",
            f"Cannot inspect workspace {workspace}: {error}. "
            "Restore access to the folder and retry; nothing was started.",
            "RESTORE_WORKSPACE_ACCESS",
        )
    return None


def _detach(
    arguments: argparse.Namespace, host_interpreter: Path | None = None, new_research: bool = False
) -> int:
    """Start a detached Host and return once it serves, as a daemon's start does.

    This launch makes the Host's launch key and hands it over in the environment; the Host
    prints its launch line without it to its log, and this launch relays the line with it, so
    the key never reaches disk. It waits `READY_SECONDS` at most and ties nothing to this
    shell: `alphalattice host stop` stops the Host. The log sits in the temporary directory,
    since a new workspace must be empty until the Host initializes it.
    """
    if arguments.stop_on_stdin:
        return _refusal(
            "local_web.detach_with_stop_on_stdin",
            "A detached Host has no standard input to stop on; `alphalattice host stop` stops "
            "it. Drop --stop-on-stdin, or --detach for an attached Host.",
            "DROP_ONE_OF_DETACH_AND_STOP_ON_STDIN",
        )
    if (workspace_refusal := _workspace_refusal(arguments.workspace)) is not None:
        return workspace_refusal
    descriptor, name = tempfile.mkstemp(prefix="alphalattice-host-", suffix=".log")
    os.close(descriptor)
    log_path, key = Path(name), secrets.token_urlsafe(32)
    command = [
        str(host_interpreter) if host_interpreter is not None else sys.executable,
        "-m",
        "alphalattice.control.product_host.composition.web_launcher",
        "--workspace",
        str(arguments.workspace),
        "--port",
        str(arguments.port),
        *(["--no-browser"] if arguments.no_browser else []),
    ]
    reader, writer = os.pipe()
    try:
        with log_path.open("wb") as log:
            host = _spawn_detached(command, log, key, writer)
    except PermissionError as error:
        os.close(reader)
        from alphalattice.interface.local_application import client

        with suppress(client.LocalResearchClientError, OSError):
            if client.LocalResearchConnection.path(arguments.workspace).exists():
                with log_path.open("a", encoding="utf-8") as failed:
                    failed.write("local_web.detached_breakaway_denied\n")
                return _failed_start(arguments.workspace, None, "", log_path)
        log_path.unlink(missing_ok=True)
        panel_host = process_config().get("ALPHALATTICE_TERMINAL_PANEL_HOST", "")
        capability = runtime_json("config/registries/host-capabilities.json")["entries"].get(
            f"{panel_host}.product-host.terminal-panel", {}
        )
        panel = os.name == "nt" and capability.get("availability") == "probed"
        message_id = "host.start.breakaway_denied" + (".panel" if panel else "")
        if sys.platform == "win32":
            win32_error = error.winerror
        else:
            win32_error = None
        return _refusal(
            "local_web.detached_breakaway_denied",
            str(worded(message_id)),
            "SERVE_FROM_THE_HOST_TERMINAL_PANEL" if panel else "SERVE_FROM_A_PERSON_OWNED_TERMINAL",
            message_id=message_id,
            launch_context={
                "win32_error": win32_error,
                "job_limit_flags": ChildJob.current_limit_flags()
                if sys.platform == "win32"
                else None,
            },
            terminal_routes=[
                dict(route)
                for route in runtime_json("config/registries/refusals.json")["entries"][
                    "local_web.detached_breakaway_denied"
                ]["terminal_routes"]
            ],
            terminal_command=terminal_serve_command(
                arguments.workspace, port=arguments.port, new_research=new_research
            ),
        )
    except BaseException:
        os.close(reader)
        raise
    finally:
        os.close(writer)  # the Host holds the one write end left, so its end closes the pipe
    return _heard_from(arguments, host, reader, log_path, key)


def _heard_from(
    arguments: argparse.Namespace,
    host: subprocess.Popen[bytes],
    reader: int,
    log_path: Path,
    key: str,
) -> int:
    """The detached start's answer from its Host: served once its ready byte came and it serves
    `HOST_STOP`; its own refusal, read from the log it left, once it ended; else stopped as not
    ready. One deadline holds the byte, the Host's exit and the serving check together."""
    deadline = time.monotonic() + READY_SECONDS
    said = _ready(reader, READY_SECONDS)
    if said == b"":  # the Host ended without serving: its pipe closed with it
        with suppress(subprocess.TimeoutExpired):
            host.wait(timeout=max(deadline - time.monotonic(), 0.0))
        if host.returncode is not None:  # its last words are in the log once it has ended
            text = log_path.read_text(encoding="utf-8", errors="replace")
            return _failed_start(
                arguments.workspace, host, text, log_path, deadline - time.monotonic()
            )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    launch = next(
        (line for line in text.splitlines() if line.startswith("Portfolio Research")), None
    )
    if said and launch is not None and (left := deadline - time.monotonic()) > 0:
        from alphalattice.interface.local_application import client  # lazy: the serving check only

        with suppress(client.LocalResearchClientError):  # its bound is the client's: no floor
            answered = client.LocalResearchClient(arguments.workspace, timeout=left).request(
                {"operation": "OPERATION_LIST", "schema_operation": "HOST_STOP"}
            )
            if "HOST_STOP" in answered.get("operation_fields", {}):
                observed = [line for line in text.splitlines() if line.startswith(_OBSERVED)]
                launch = launch.replace(HANDED_KEY, f"key={key}")
                return _served(arguments, "\n".join([launch, *observed[:1]]), host, log_path)
    spawn.stop_tree(host.pid)
    return _refusal(
        "local_web.detached_host_not_ready",
        f"The Host did not become ready within {READY_SECONDS} seconds and was stopped. "
        f"Its log is {log_path}. Run alphalattice serve again.",
        "SERVE_AGAIN",
    )


def _ready(reader: int, seconds: float) -> bytes | None:
    """The detached Host's ready byte; b"" once it ended without one, its pipe closed with it;
    None when it said neither within ``seconds``. The read blocks in its own thread, which ends
    on the byte, on the Host's end, or on its stop after the bound."""
    heard: list[bytes] = []

    def read() -> None:
        with os.fdopen(reader, "rb", buffering=0) as pipe:
            heard.append(pipe.read(1))

    reading = threading.Thread(target=read, name="host-ready", daemon=True)
    reading.start()
    reading.join(seconds)
    return heard[0] if heard else None


def _served(
    arguments: argparse.Namespace,
    launch: str,
    host: subprocess.Popen[bytes],
    log_path: Path,
) -> int:
    """Say plainly that it STARTED, where and at which address; relay the launch line, say how
    it lives, and name the next step."""
    address = re.search(r"http://[^/\s?#]+/", launch)
    workspace = arguments.workspace.resolve()
    stream = sys.stdout  # noqa: TID251  # bootstrap/lifecycle stream ownership
    if hasattr(stream, "reconfigure"):
        # Lifecycle text is UTF-8 even when Python chose a locale encoding for the pipe.
        stream.reconfigure(encoding="utf-8")
    print(f"STARTED: a Host now serves {workspace} at {address[0] if address else 'its link'}.")
    print(launch)
    print(
        f"Loopback only. This Host (process {host.pid}) runs apart from the shell that "
        f"started it and outlives it; `alphalattice host stop` stops it. Log: {log_path}\n"
        f"{str(worded('host.idle')).format(seconds=DEFAULT_IDLE_TIMEOUT_SECONDS)}",
        flush=True,
    )
    if arguments.no_browser:
        print(OPEN_THE_WORKBENCH, flush=True)
    return 0


def _failed_start(
    workspace: Path,
    host: subprocess.Popen[bytes] | None,
    text: str,
    log_path: Path,
    seconds: float = READY_SECONDS,
) -> int:
    """Reuse only the exact authenticated serving Host; preserve unverified discovery and lease.
    Its verification spends only the ``seconds`` its start has left, and none left is unverified."""
    if host is None or "workspace_runtime.writer_already_owned" in text:
        from alphalattice.interface.local_application import client

        try:
            if seconds <= 0:  # nothing of the start's deadline is left to verify with
                raise TimeoutError
            reader = client.LocalResearchClient(workspace, timeout=seconds)
            body = reader.activity(limit=1)
            observer = body.get("observer")
            serving = observer.get("host") if isinstance(observer, dict) else None
            if not isinstance(serving, dict):
                raise ValueError("local_client.connection_unreadable")
            pid, since = serving.get("pid"), serving.get("started_at")
            if (
                body.get("workspace_id") != reader.connection.workspace_id
                or serving.get("instance") != reader.connection.instance
                or type(pid) is not int
                or pid <= 0
                or not isinstance(since, str)
            ):
                raise ValueError("local_client.connection_unreadable")
            if datetime.fromisoformat(since).tzinfo is None:
                raise ValueError("local_client.connection_unreadable")
        except (client.LocalResearchClientError, OSError, ValueError, TypeError) as error:
            message_id = "host.reuse.unverified"
            return _refusal(
                "local_web.serving_host_unverified",
                str(worded(message_id)),
                "READ_THE_HOST_LOG",
                message_id=message_id,
                cause="local_web.detached_host_not_ready"
                if seconds <= 0
                else safe_failure_code(str(error)) or "local_client.connection_unreadable",
                host_log=str(log_path),
            )
        log_path.unlink(missing_ok=True)
        profile = {key: serving[key] for key in ("instance", "pid", "started_at")}
        message_id = "host.serving"
        served = {
            "workspace": str(workspace.resolve()),
            "address": reader.connection.url,
            "host": {**profile, "memory": serving.get("memory")},
            "message_id": message_id,
            "detail": str(worded(message_id)).format(**profile),
        }
        if "workbench" in body:
            served["workbench"] = body["workbench"]
        print(
            json.dumps(
                {"status": "ALREADY_SERVING", "next_action": "USE_THE_SERVING_HOST", **served}
            ),
            flush=True,
        )
        return 0
    return _refusal(
        "local_web.detached_host_exited",
        f"The detached Host exited (code {host.returncode}) before it served; its log, which "
        f"holds no launch key, says why: {log_path}",
        "READ_THE_HOST_LOG",
    )


def main(
    argv: list[str] | None = None,
    *,
    host_interpreter: Path | None = None,
    new_research: bool = False,
    own_process_tree: bool = False,
) -> int:
    """Run the declared command and return its exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="0 lets the operating system choose, which is the safe default",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--stop-on-stdin",
        action="store_true",
        help="For an attached host: stop on a 'stop' line or EOF; join Tasks before exit. With "
        "no standard input held open, nothing ties it and it serves until its process ends.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Start the Host as its own process, apart from this shell and its Job, and return "
        "once it serves; `alphalattice host stop` stops it.",
    )
    arguments = parser.parse_args(argv)
    if arguments.detach:
        return _detach(arguments, host_interpreter, new_research)
    if own_process_tree:
        if sys.platform == "win32" and process_config().secret(LAUNCH_KEY_VARIABLE) is not None:
            # The detached launcher verifies it escaped its caller's Job before
            # we enter the serving process's own Job.
            sys.stdin.read()
        with spawn.current_tree("host"):
            return _serve(arguments)
    return _serve(arguments)


def _serve(arguments: argparse.Namespace) -> int:
    """Run one Host and drain its work before releasing the service."""
    handed = process_config().secret(LAUNCH_KEY_VARIABLE) is not None
    # A detached launch's ready pipe, taken before this Host starts any child of its own.
    named = consume_secret(READY_VARIABLE)
    ready = None if named is None else spawn.inherited(named.reveal(), os.O_WRONLY)

    from alphalattice.control.product_host.composition.local_web_session import (  # lazy: serve
        LocalPortfolioWebSession,
    )

    session = LocalPortfolioWebSession.from_workspace(
        arguments.workspace,
        port=arguments.port,
    )
    if not handed:
        session.idle_timeout_seconds = 0
    # `start` is inside the cleanup boundary, not before it. It acquires a
    # workspace lease and starts a worker before it binds a socket, so a failure
    # part-way through is released by `finally`.
    tied = arguments.stop_on_stdin and _stdin_ties()
    stop = threading.Event()

    def stopping() -> dict[str, object]:
        stop.set()
        if session.dispatcher is not None:
            session.dispatcher._begin_drain()
        return {
            **STOPPING,
            "finishing_task_ids": (
                [] if session.dispatcher is None else session.dispatcher.finishing()
            ),
        }

    def watched() -> None:
        _wait_for_stop_on_stdin()
        stopping()

    # The stop is read from the start, so a stop during a slow start (its resume) ends it at the
    # next step instead of after it, past a caller's wait, with its connection left behind.
    session.stop_requested = stop.is_set
    session.stop_host = stopping
    try:
        if tied:
            # Either way of stopping ends the wait: the input's own, or `host stop`.
            threading.Thread(target=watched, daemon=True).start()
        session.start()
        if stop.is_set():  # stopped while starting: no launch line, no browser; `finally` stops it
            return 0
        # The launch URL gives the browser that opens it the session, and only it (HB):
        # a script or the QA kit takes it from this line.
        url = session.launch_url

        def needed() -> bool:  # read each time: a Host owing its upgrade composes later
            operations = session.operations
            if operations is None or operations.automation is None:
                return False
            return operations.automation.keeps_host(operations.goals)

        # A detached launch holds the key it handed over and prints it itself (never on disk).
        shown = re.sub(r"key=[^&#]+", HANDED_KEY, url) if handed else url
        print(f"Portfolio Research workspace: {shown}")
        print(
            (
                "Loopback only. Send 'stop' or close stdin, or run "
                "`alphalattice host stop`, to stop."
                if tied
                else "Loopback only. No standard input holds this Host, so it serves until "
                "`alphalattice host stop` or its process ends."
                if arguments.stop_on_stdin
                else "Loopback only. Ctrl-C or `alphalattice host stop` to stop."
            )
            + (
                "\n" + str(worded("host.idle")).format(seconds=session.idle_timeout_seconds)
                if handed
                else ""
            ),
            flush=True,
        )
        held = spawn.held_memory()
        if held is not None and held["unproved"]:  # the start says what its budget does
            print(str(worded("host.memory_observed")).format(reason=held["unproved"]), flush=True)
        if ready is not None:  # the launch line is in the log: the launcher may relay it now,
            # before any browser work, which can take longer than the launcher waits
            with suppress(OSError), os.fdopen(ready, "wb", buffering=0) as pipe:
                ready = None
                pipe.write(b"1")
        if arguments.no_browser:
            print(OPEN_THE_WORKBENCH, flush=True)
        else:
            webbrowser.open(url)
        while not stop.wait(1):  # a timed wait keeps Ctrl-C heard
            if session.dispatcher is not None and session.dispatcher._idle_expired(
                session.idle_timeout_seconds, needed
            ):
                stopping()
    except KeyboardInterrupt:
        stopping()
        print("\nStopping.")
    finally:
        # Unbounded on purpose: it joins every request thread and the dispatcher
        # worker before releasing the lease, and a launcher that gave up early
        # would leave a second writer behind for the next one to meet.
        session.stop()
        if ready is not None:  # never served: the launcher reads the pipe's end as the exit
            os.close(ready)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(own_process_tree=True))

__all__ = ["main"]
