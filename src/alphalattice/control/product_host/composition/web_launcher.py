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
from pathlib import Path
from typing import Any

from alphalattice.interface.local_application.web import LAUNCH_KEY_VARIABLE, OPEN_THE_WORKBENCH

# The Host's session (and the numerical libraries it brings) is imported in `main`, never here:
# a Host run as `-m` this module has each spawned worker re-import it first, before the worker
# sets its numerical libraries to one thread (child._worker_started), and a library loaded then
# keeps a thread and its buffers for every processor (about 1.8 GB of commit a worker).

STOPPING: dict[str, object] = {
    "status": "HOST_STOPPING",
    "detail": "This Host stops once its running work has ended: it answers this, lets its Tasks "
    "finish, joins its workers and releases the workspace; nothing is cancelled. Serve it again "
    "when work needs it.",
    "next_action": "SERVE_AGAIN_WHEN_WORK_NEEDS_IT",
}
"""The answer to `host stop`, given before the Host begins to stop."""


def _windows_stdin(descriptor: int) -> tuple[Any, Any, int]:
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
                    time.sleep(0.2)
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


def _refusal(
    failure_code: str,
    detail: str,
    next_action: str,
    *,
    message_id: str | None = None,
    terminal_command: list[str] | None = None,
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
    print(json.dumps(answer), flush=True)
    return 2


def _spawn_detached(command: list[str], log: Any, key: str) -> subprocess.Popen[bytes]:
    """Start the Host apart from this process: its own process group (session), no console
    window and no standard input, and out of this process's Job where the Job allows it (a
    Codex turn ends its shell's children, by its tree or its Job).

    Denied Job breakaway raises; no still-held child is started.
    """
    # The new interpreter imports this same source, as `entry.serve` gives its child.
    source = str(Path(__file__).resolve().parents[4])
    inherited = os.environ.get("PYTHONPATH")
    environment = {
        **os.environ,
        "PYTHONPATH": source + os.pathsep + inherited if inherited else source,
        LAUNCH_KEY_VARIABLE: key,
    }
    common: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": log,
        "env": environment,
    }
    if sys.platform != "win32":
        return subprocess.Popen(command, start_new_session=True, **common)
    # No console window: the Host's workers start in its hidden console, never in new windows.
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.IsProcessInJob.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    in_job = ctypes.c_int()
    child = subprocess.Popen(
        command, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **common
    )
    # Windows Popen exposes a native handle absent from the cross-platform stub.
    native: Any = child
    handle = ctypes.c_void_p(int(native._handle))
    if not kernel32.IsProcessInJob(handle, None, ctypes.byref(in_job)) or in_job.value:
        child.terminate()
        child.wait()
        raise PermissionError("local_web.detached_breakaway_denied")
    return child


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
    try:
        with log_path.open("wb") as log:
            host = _spawn_detached(command, log, key)
    except PermissionError:
        from alphalattice.interface.local_application.cli_contract import ANSWER_LANGUAGE, worded

        log_path.unlink(missing_ok=True)
        message_id = "host.start.breakaway_denied"
        return _refusal(
            "local_web.detached_breakaway_denied",
            str(worded(message_id)),
            "SERVE_FROM_A_PERSON_OWNED_TERMINAL",
            message_id=message_id,
            terminal_command=[
                "alphalattice",
                "--lang",
                ANSWER_LANGUAGE.get(),
                "--workspace",
                str(arguments.workspace.resolve()),
                "serve",
                "--port",
                str(arguments.port),
                "--no-browser",
                "--detach",
                *(["--new-research"] if new_research else []),
            ],
        )
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        launch = next(
            (line for line in text.splitlines() if line.startswith("Portfolio Research")), None
        )
        if launch is not None and "Loopback only." in text:
            from alphalattice.interface.local_application import client

            with suppress(client.LocalResearchClientError):
                ready = client.LocalResearchClient(
                    arguments.workspace, timeout=READY_SECONDS
                ).request(
                    {"operation": "OPERATION_LIST", "schema_operation": "HOST_STOP"},
                    timeout=deadline - time.monotonic(),
                )
                if "HOST_STOP" in ready.get("operation_fields", {}):
                    return _served(
                        arguments, launch.replace(HANDED_KEY, f"key={key}"), host, log_path
                    )
        if host.poll() is not None:
            return _exited(arguments.workspace, host, text, log_path)
        time.sleep(0.2)
    return _refusal(
        "local_web.detached_host_not_ready",
        f"The detached Host (process {host.pid}) has not reported serving after "
        f"{READY_SECONDS} s; it may still be starting. Read its log, {log_path}, or stop it "
        "with `alphalattice host stop`.",
        "READ_THE_HOST_LOG_OR_STOP_IT",
    )


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
    print(f"STARTED: a Host now serves {workspace} at {address[0] if address else 'its link'}.")
    print(launch)
    print(
        f"Loopback only. This Host (process {host.pid}) runs apart from the shell that "
        f"started it and outlives it; `alphalattice host stop` stops it. Log: {log_path}",
        flush=True,
    )
    if arguments.no_browser:
        print(OPEN_THE_WORKBENCH, flush=True)
    return 0


def _exited(workspace: Path, host: subprocess.Popen[bytes], text: str, log_path: Path) -> int:
    """A detached Host that ended before it served: plainly ALREADY_SERVING where another Host
    serves the workspace (with its address, from its connection record), else its log's reason."""
    if "workspace_runtime.writer_already_owned" in text:
        from alphalattice.interface.local_application import client

        log_path.unlink(missing_ok=True)
        address = None
        with suppress(client.LocalResearchClientError, OSError):
            address = client.LocalResearchConnection.read(workspace).url
        detail = (
            "A Host already serves this workspace, so no second one started. Use it: your "
            "Session's first answer to it offers the Workbench (`workbench`); "
            "`alphalattice host stop` stops it."
        )
        served = {"workspace": str(workspace.resolve()), "address": address, "detail": detail}
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
    handed = LAUNCH_KEY_VARIABLE in os.environ

    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )

    session = LocalPortfolioWebSession.from_workspace(
        arguments.workspace,
        port=arguments.port,
    )
    # `start` is inside the cleanup boundary, not before it. It acquires a
    # workspace lease and starts a worker before it binds a socket, so a failure
    # part-way through still has something to release -- and this `finally` is
    # what releases it.
    try:
        session.start()
        # The launch URL gives the browser that opens it the session, and only it (HB):
        # a script or the QA kit takes it from this line.
        url = session.launch_url
        tied = arguments.stop_on_stdin and _stdin_ties()
        stop = threading.Event()

        def stopping() -> dict[str, object]:
            stop.set()
            return dict(STOPPING)

        def watched() -> None:
            _wait_for_stop_on_stdin()
            stop.set()

        if session.operations is not None:
            session.operations.stop_host = stopping
        # A detached launch holds the key it handed over and prints it itself (never on disk).
        shown = re.sub(r"key=[^&#]+", HANDED_KEY, url) if handed else url
        print(f"Portfolio Research workspace: {shown}")
        print(
            "Loopback only. Send 'stop' or close stdin, or run `alphalattice host stop`, to stop."
            if tied
            else "Loopback only. No standard input holds this Host, so it serves until "
            "`alphalattice host stop` or its process ends."
            if arguments.stop_on_stdin
            else "Loopback only. Ctrl-C or `alphalattice host stop` to stop.",
            flush=True,
        )
        if arguments.no_browser:
            print(OPEN_THE_WORKBENCH, flush=True)
        else:
            webbrowser.open(url)
        if tied:
            # Either way of stopping ends the wait: the input's own, or `host stop`.
            threading.Thread(target=watched, daemon=True).start()
        while not stop.wait(1):  # a timed wait keeps Ctrl-C heard
            pass
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        # Unbounded on purpose: it joins every request thread and the dispatcher
        # worker before releasing the lease, and a launcher that gave up early
        # would leave a second writer behind for the next one to meet.
        session.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["main"]
