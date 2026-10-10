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
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)


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


def main(argv: list[str] | None = None) -> int:
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
    arguments = parser.parse_args(argv)

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
        print(f"Portfolio Research workspace: {url}")
        print(
            "Loopback only. Send 'stop' or close stdin to stop."
            if tied
            else "Loopback only. No standard input holds this Host, so it serves until its "
            "process ends."
            if arguments.stop_on_stdin
            else "Loopback only. Ctrl-C to stop.",
            flush=True,
        )
        if not arguments.no_browser:
            webbrowser.open(url)
        if tied:
            _wait_for_stop_on_stdin()
        else:
            threading.Event().wait()
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

__all__ = ["LocalPortfolioWebSession", "main"]
