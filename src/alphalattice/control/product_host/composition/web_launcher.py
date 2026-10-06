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

`--sec-network-consent` is the operator's explicit consent to acquire SEC
filings from the official endpoints for the workspace's Evidence and CRO
section; without it the section stays on its recorded package, offline. The
consent alone composes nothing: the environment must allow the network
through the workspace network control and name the SEC contact (`SEC_USER_AGENT`),
and a refusal is printed by name. `--sec-max-document-bytes` declares the
per-document cap for that source, never above the contract's maximum.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.evidence.alternative_evidence.sources.admission import (
    admit_official_source,
)


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
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
        if kernel32.GetFileType(handle) == 3:  # FILE_TYPE_PIPE
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
        help="For an attached host: stop on a 'stop' line or EOF; join Tasks before exit.",
    )
    parser.add_argument(
        "--sec-network-consent",
        action="store_true",
        help="Explicit consent to acquire SEC filings from the official endpoints.",
    )
    parser.add_argument(
        "--sec-max-document-bytes",
        type=int,
        default=None,
        help="The per-document cap declared for the official source (contract maximum bound).",
    )
    parser.add_argument(
        "--sec-acquisition-window-seconds",
        type=int,
        default=None,
        help="The acquisition window declared for the official source's requests, from "
        "each request's cutoff; the admitted policy's window otherwise.",
    )
    parser.add_argument(
        "--sec-max-total-attempts",
        type=int,
        default=None,
        help="Campaign bound on HTTP attempts of every kind, enforced before each request.",
    )
    parser.add_argument(
        "--sec-max-total-response-bytes",
        type=int,
        default=None,
        help="Campaign bound on decoded response bytes, partial transfers included.",
    )
    parser.add_argument(
        "--sec-max-body-resources",
        type=int,
        default=None,
        help="Campaign bound on distinct filing bodies fetched.",
    )
    parser.add_argument(
        "--sec-campaign-id",
        default=None,
        help="Name the durable campaign these bounds admit (all three bounds and the "
        "document cap required); a later launch with the same id and bounds resumes "
        "it with only what remains, from its ledger under the workspace.",
    )
    arguments = parser.parse_args(argv)

    official_source = None
    if arguments.sec_network_consent or arguments.sec_max_document_bytes is not None:
        official_source = admit_official_source(
            network_consent=bool(arguments.sec_network_consent),
            maximum_document_bytes=arguments.sec_max_document_bytes,
            acquisition_window_seconds=arguments.sec_acquisition_window_seconds,
            maximum_total_attempts=arguments.sec_max_total_attempts,
            maximum_total_response_bytes=arguments.sec_max_total_response_bytes,
            maximum_body_resources=arguments.sec_max_body_resources,
            campaign_id=arguments.sec_campaign_id,
            workspace_root=arguments.workspace,
        )
        if official_source.transport_origin == "DENIED":
            print(
                "Official SEC source admitted with the network disabled: live "
                "preparations read back, no request leaves this process, and a new "
                f"source check is refused ({official_source.refusal_code})."
            )
        elif official_source.admitted:
            print(
                "Official SEC source admitted "
                f"({official_source.transport_origin}; per-document cap "
                f"{official_source.maximum_document_bytes or 'recorded policy'})."
            )
            campaign = official_source.campaign
            if campaign is not None:
                print(
                    f"Campaign {campaign['campaign_id']} "
                    f"{'resumed' if campaign['resumed'] else 'admitted'}: "
                    f"{campaign['attempts_remaining']} attempts, "
                    f"{campaign['bytes_remaining']:,} decoded bytes and "
                    f"{campaign['body_resources_remaining']} bodies remain; "
                    f"{campaign['unsettled_reservations']} unsettled reservation(s)."
                )
        else:
            print(
                f"Official SEC source not admitted: {official_source.refusal_code}; "
                "the Evidence and CRO section stays on its recorded package."
            )
    session = LocalPortfolioWebSession.from_workspace(
        arguments.workspace,
        port=arguments.port,
        official_source=official_source,
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
        print(f"Portfolio Research workspace: {url}")
        print(
            "Loopback only. Send 'stop' or close stdin to stop."
            if arguments.stop_on_stdin
            else "Loopback only. Ctrl-C to stop.",
            flush=True,
        )
        if not arguments.no_browser:
            webbrowser.open(url)
        if arguments.stop_on_stdin:
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
        if official_source is not None and official_source.source is not None:
            # The official client, and with it the campaign's ownership.
            official_source.source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["LocalPortfolioWebSession", "main"]
