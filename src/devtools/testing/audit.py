"""One process audit hook with scoped subscribers and reentrant per-thread dispatch."""

from __future__ import annotations

import ntpath
import os
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

# NumPy's locked import initializes platform metadata. Complete it before either
# a layer contract or a result recorder subscribes to this process's audit hook.
import numpy.testing  # noqa: F401

_INSTALLED: bool = globals().get("_INSTALLED", False)
_LOCK = globals().get("_LOCK") or threading.RLock()
_SUBSCRIBERS: list[tuple[object, Callable[[str, tuple[Any, ...]], None]]] = globals().get(
    "_SUBSCRIBERS", []
)
_DISPATCH = globals().get("_DISPATCH") or threading.local()
_PROJECT = Path(__file__).absolute().parents[3]


def dispatch(event: str, args: tuple[Any, ...]) -> None:
    if not getattr(_DISPATCH, "busy", False):
        _DISPATCH.busy = True
        try:
            for _, callback in tuple(_SUBSCRIBERS):
                callback(event, args)
        finally:
            _DISPATCH.busy = False


def _install_audit() -> None:
    global _INSTALLED
    with _LOCK:
        if not _INSTALLED:
            _INSTALLED = True  # a vetoed installation cannot be retried as a second hook
            sys.addaudithook(dispatch)


@contextmanager
def subscribe(callback: Callable[[str, tuple[Any, ...]], None]) -> Iterator[None]:
    """Own a subscription to the single process hook, verifying actual dispatch."""
    if not callable(callback):
        raise ValueError("testing.layer_contract_unmeasured: subscriber must be callable")
    token, acknowledged = object(), []

    def receive(event: str, args: tuple[Any, ...]) -> None:
        if event == "alphalattice.audit.subscribed" and len(args) == 1 and args[0] is token:
            acknowledged.append(True)
        else:
            callback(event, args)

    with _LOCK:
        _install_audit()
        _SUBSCRIBERS.append((token, receive))
    try:
        sys.audit("alphalattice.audit.subscribed", token)
        if not acknowledged:
            raise RuntimeError(
                "testing.layer_contract_unmeasured: audit dispatch unavailable; "
                "restore the process hook"
            )
        yield
    finally:
        with _LOCK:
            _SUBSCRIBERS[:] = [row for row in _SUBSCRIBERS if row[0] is not token]


class ResourceObservation:
    """Count audited attempts for one pytest node without deciding its outcome."""

    def __init__(self) -> None:
        self.thread = threading.get_ident()
        self.phase = "setup"
        self.files = self.database = self.network = self.background = False
        self.subprocesses: dict[str, dict[str, Any]] = {}
        self.errors = 0
        self.available = True

    def note(self, event: str, args: tuple[Any, ...]) -> None:
        if event not in {"open", "subprocess.Popen", "sqlite3.connect"} and not event.startswith(
            "socket."
        ):
            return
        background = threading.get_ident() != self.thread
        if background:
            self.background = True
            if event != "subprocess.Popen":
                return
        try:
            if event == "subprocess.Popen":
                self._subprocess(args, background)
            elif event == "sqlite3.connect":
                self.database = True
            elif event.startswith("socket.") and event != "socket.gethostname":
                self.network = True
            elif event == "open" and args and not isinstance(args[0], int):
                self._file()
        except Exception:
            self.errors += 1

    def _subprocess(self, args: tuple[Any, ...], background: bool) -> None:
        executable = args[0]
        if executable is None and len(args) > 1:
            command = args[1]
            if isinstance(command, (tuple, list)):
                executable = command[0] if command else None
            elif isinstance(command, str):
                if command.startswith('"'):
                    end = command.find('"', 1)
                    executable = command[1:end] if end > 1 else None
                else:
                    executable = command.split(maxsplit=1)[0] if command.strip() else None
        executable = ntpath.basename(os.fsdecode(executable)).lower() if executable else "unknown"
        executable = executable or "unknown"
        callsite = self._callsite()
        with _LOCK:
            row = self.subprocesses.setdefault(
                executable,
                {
                    "count": 0,
                    "background_count": 0,
                    "sampled_seconds": None,
                    "waited_seconds": None,
                    "reason": "no profile",
                    "phases": {},
                    "callsites": {},
                },
            )
            row["count"] += 1
            row["background_count"] += int(background)
            phases = row["phases"]
            phases[self.phase] = phases.get(self.phase, 0) + 1
            sites = row["callsites"]
            sites[callsite] = sites.get(callsite, 0) + 1

    def _callsite(self) -> tuple[str, str, int, str]:
        frame: FrameType | None = sys._getframe(1)
        ignored = (__name__, "subprocess", "threading", "devtools.pipeline.receipt")
        while frame:
            module = frame.f_globals.get("__name__", "")
            if module not in ignored and not module.startswith(("_pytest.", "pytest.", "execnet.")):
                path = Path(frame.f_code.co_filename)
                try:
                    name, scope = path.relative_to(_PROJECT).as_posix(), "project"
                except ValueError:
                    name, scope = path.name, "external"
                return name, frame.f_code.co_name, frame.f_lineno, scope
            frame = frame.f_back
        return "unknown", "unknown", 0, "unknown"

    def _file(self) -> None:
        frame: FrameType | None = sys._getframe(1)
        ignored = (__name__, "devtools.pipeline.receipt", "pathlib", "tempfile", "os", "io")
        while frame and frame.f_globals.get("__name__", "") in ignored:
            frame = frame.f_back
        name = frame.f_globals.get("__name__", "") if frame else ""
        if not name.startswith(("_pytest.", "pytest.", "importlib.", "_frozen_importlib")):
            self.files = True

    def snapshot(self, declared: str | None = None) -> dict[str, Any]:
        unknown = ["duckdb", "in-process Host", "native I/O"]
        if not self.available:
            unknown.append("audit unavailable")
        if self.subprocesses:
            unknown.append("descendant subprocesses")
        if "unknown" in self.subprocesses:
            unknown.append("subprocess executable")
        if self.background:
            unknown.append("background-thread attribution")
        if self.errors:
            unknown.append("audit event decoding")
        if not self.available:
            layer = "unknown"
        elif self.network:
            layer = "e2e"
        elif self.files or self.database:
            layer = "integration"
        elif (
            declared == "unit" and not self.subprocesses and not self.background and not self.errors
        ):
            layer = "unit"
        else:
            layer = "unknown"
        with _LOCK:
            processes = {}
            for name, row in self.subprocesses.items():
                sites = [
                    dict(file=file, function=function, line=line, scope=scope, count=count)
                    for (file, function, line, scope), count in row["callsites"].items()
                ]
                processes[name] = {**row, "phases": dict(row["phases"]), "callsites": sites}
        return dict(
            observed_layer=layer,
            basis="audited attempts; unit also requires its enforced contract",
            files=self.files,
            database=self.database,
            network=self.network,
            subprocesses=processes,
            unknown=unknown,
        )
