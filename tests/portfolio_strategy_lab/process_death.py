"""The process-death recovery harness shared by the Portfolio and workspace owners.

Two halves, deliberately in two places. The spawner and handshake live here,
once. The phase table -- what the child does before it reports its boundary --
is each owner's own ``_crash_child(workspace, phase)`` in its test module,
because the phases are the owner's recovery contract, not a shared fixture.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import signal
import subprocess
import sys
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Thread

from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    _process_tree_ids,
)


def _creation_filtered_tree(root: int, parents: dict[int, int], birth) -> tuple[int, ...]:
    """A reused parent PID does not adopt children older than that process."""
    selected = [root]
    pending = [root]
    while pending:
        parent = pending.pop()
        parent_birth = birth(parent)
        if parent_birth is None:
            continue
        for pid, parent_pid in parents.items():
            if parent_pid != parent or pid in selected:
                continue
            created = birth(pid)
            if created is not None and created >= parent_birth:
                selected.append(pid)
                pending.append(pid)
    return tuple(selected)


def _terminate_windows_owned_tree(child) -> None:
    """Only lifetime-verified descendants; terminate the same handles verified.

    Telemetry's PID-only graph is deliberately NOT destructive authority. A
    real PID reuse made it include an unrelated, older OneDrive service here.
    Keep this stricter cleanup in the test owner, not in scientific source pins.
    """
    from ctypes import wintypes as w

    class Entry(ctypes.Structure):
        _fields_ = [
            ("size", w.DWORD),
            ("usage", w.DWORD),
            ("pid", w.DWORD),
            ("heap", ctypes.c_size_t),
            ("module", w.DWORD),
            ("threads", w.DWORD),
            ("parent", w.DWORD),
            ("priority", w.LONG),
            ("flags", w.DWORD),
            ("name", w.WCHAR * 260),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.CloseHandle.restype = w.BOOL
    kernel.GetProcessTimes.argtypes = [w.HANDLE, *([ctypes.POINTER(w.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = w.BOOL
    kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    kernel.GetExitCodeProcess.restype = w.BOOL
    kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
    kernel.TerminateProcess.restype = w.BOOL
    for name in ("Process32FirstW", "Process32NextW"):
        method = getattr(kernel, name)
        method.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
        method.restype = w.BOOL

    def parents() -> dict[int, int]:
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = Entry(size=ctypes.sizeof(Entry))
            result = {}
            more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while more:
                result[int(entry.pid)] = int(entry.parent)
                more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
            return result
        finally:
            kernel.CloseHandle(snapshot)

    def created(handle):
        times = [w.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime

    handles = {}
    births = {child.pid: created(int(child._handle))}

    def birth(pid):
        if pid not in births:
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return None
            handles[pid] = handle
            births[pid] = created(handle)
        return births[pid]

    try:
        admitted = _creation_filtered_tree(child.pid, parents(), birth)
        # Recheck parent links after opening handles: a PID may have been reused
        # between the first snapshot and OpenProcess.
        current = {pid: parent for pid, parent in parents().items() if pid in admitted}
        admitted = _creation_filtered_tree(child.pid, current, births.get)
        if child.poll() is not None:
            return  # the anchor exited; its numeric PID is no longer an ownership root
        for pid in reversed(admitted):
            if pid == child.pid:
                if child.poll() is None:
                    child.terminate()  # Popen owns this handle, not a PID lookup.
                continue
            handle = kernel.OpenProcess(0x1001, False, pid)
            if not handle:
                if ctypes.get_last_error() == 87:  # exited before handle acquisition
                    continue
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if created(handle) != births[pid]:
                    continue  # another process now owns the numeric PID
                code = w.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    raise ctypes.WinError(ctypes.get_last_error())
                if code.value == 259 and not kernel.TerminateProcess(handle, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(handle)
    finally:
        for handle in handles.values():
            kernel.CloseHandle(handle)


def child_import_paths(module_dir: Path | None = None) -> list[str]:
    """The ``sys.path`` prefix a spawned child needs to import a test module by name.

    The worktree root (for ``tests``), ``src`` and ``scripts`` -- nothing else.
    The flat layout gave a self-spawned test file its own directory for free
    and each module patched the rest in; the package tree gives neither, so a
    child is started through ``-c`` with this prefix and the module's package
    name.
    """
    root = Path(__file__).resolve().parents[2]
    paths = [str(root / "src"), str(root / "scripts"), str(root)]
    if module_dir is not None:
        paths.append(str(module_dir))
    return paths


@contextmanager
def forced_process(
    workspace: Path,
    module: str,
    phase: str,
    *,
    module_dir: Path | None = None,
    boundary_timeout_seconds: float = 300,
):  # type: ignore[no-untyped-def]
    """Run ``module._crash_child(workspace, phase)`` in a child and stop at its boundary.

    The child prints one ``CRASH_BOUNDARY:<json>`` line when it reaches the phase
    under test; the parent verifies the reporting PID is a descendant of the
    process it started, yields that message, and afterwards terminates only that
    descendant tree. Only descendants of a process this test started are ever
    signalled, and only after the handshake.

    ``boundary_timeout_seconds`` is a hang budget, not a performance bound: the
    wait returns the moment the boundary line arrives, and a child that exits
    early fails at once through the EOF sentinel, so a liberal default costs
    nothing on the healthy path. It was sixty seconds, which the routed lane at
    eight workers proved is a budget for a lightly loaded machine -- a child
    that builds a real workspace before its boundary can legitimately take
    longer under contention -- so it is three hundred, and a caller with a
    heavier phase still raises it explicitly.
    """
    paths = child_import_paths(module_dir)
    command = (
        f"import sys; sys.path[:0] = {paths!r}; from {module} import _crash_child; "
        "_crash_child(sys.argv[1], sys.argv[2])"
    )
    env = dict(os.environ, ALPHALATTICE_NETWORK_DISABLED="1")
    env.pop("PYTHONPATH", None)
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", command, str(workspace), phase],
        cwd=workspace.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    messages = queue.Queue()
    child_output = []

    def read():  # type: ignore[no-untyped-def]
        assert child.stdout is not None
        for line in child.stdout:
            child_output.append(line)
            if len(child_output) > 40:
                child_output.pop(0)
            if line.startswith("CRASH_BOUNDARY:"):
                messages.put(json.loads(line.removeprefix("CRASH_BOUNDARY:")))
        messages.put(None)

    reader = Thread(target=read, daemon=True)
    reader.start()
    try:
        observed = messages.get(timeout=boundary_timeout_seconds)
        if observed is None:
            # stdout EOF can precede the Windows venv launcher's exit by a few
            # milliseconds. Reap the owned handle before trying to kill its PID.
            with suppress(subprocess.TimeoutExpired):
                child.wait(timeout=5)
        assert observed is not None, (
            f"child exited before boundary: {child.poll()} {''.join(child_output)}"
        )
        descendants, limitation = _process_tree_ids(child.pid)
        assert limitation is None and observed["pid"] in descendants
        yield observed
    finally:
        if child.poll() is None:
            if os.name == "nt":
                _terminate_windows_owned_tree(child)
            else:
                descendants, limitation = _process_tree_ids(child.pid)
                assert limitation is None
                for pid in reversed(descendants):
                    with suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGTERM)
        child.wait(timeout=30)
        reader.join(timeout=5)
        assert not reader.is_alive()
        assert child.stdout is not None
        child.stdout.close()


__all__ = ["child_import_paths", "forced_process"]
