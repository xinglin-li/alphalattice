"""Admit a Linux child to a user cgroup before releasing its first instruction."""

from __future__ import annotations

import ctypes
import errno
import os
import select
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

_CGROUP = Path("/sys/fs/cgroup")
_BOOT = """import os, sys
ready, start = map(int, sys.argv[1:3])
with open("/proc/self/cgroup", "rb") as record:
    os.write(ready, record.read())
os.close(ready)
if os.read(start, 1) != b"1":
    raise SystemExit(1)
os.close(start)
os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
"""


def available() -> bool:
    """Find a user manager on a unified cgroup v2 hierarchy."""
    return (
        sys.platform == "linux"
        and shutil.which("systemd-run") is not None
        and (_CGROUP / "cgroup.controllers").is_file()
        and Path(f"/run/user/{os.getuid()}/systemd/private").is_socket()
    )


def enable_child_subreaper() -> None:
    """Let a dedicated Linux launcher adopt its own descendants after reparenting."""
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    set_child_subreaper = 36
    if prctl(set_child_subreaper, 1, 0, 0, 0) != 0:
        raise OSError("spawn.child_subreaper_unavailable")


class LinuxScope:
    """Hold accounting and termination for one new, verified cgroup."""

    def __init__(self, directory: Path, *, subgroup: bool = False) -> None:
        """Hold the kernel memory event descriptor while this scope exists."""
        self.directory = directory
        self.subgroup = subgroup
        self.events = None if subgroup else (directory / "memory.events").open("rb", buffering=0)

    def sample(self) -> tuple[int, bool]:
        """Read the cgroup peak and whether an allocation reached its hard limit."""
        assert self.events is not None
        try:
            self.events.seek(0)
            events = dict(line.split() for line in self.events.read().splitlines())
            peak = int((self.directory / "memory.peak").read_bytes())
        except OSError as error:
            if error.errno not in {errno.ENOENT, errno.ENODEV}:
                raise
            return 0, False  # A zero-member transient scope was removed by its manager.
        return peak, int(events.get(b"max", b"0")) != 0

    def wait(self, *stops: int) -> None:
        """Wait on native memory changes or the explicit watcher teardown."""
        assert self.events is not None
        native = vars(select)  # These Linux interfaces are absent from Windows's select.
        watcher = native["poll"]()
        watcher.register(self.events, native["POLLPRI"])
        for stop in stops:
            watcher.register(stop, native["POLLIN"])
        watcher.poll(1000)  # Observe has no kernel threshold and samples cumulative peak.

    def stop(self) -> None:
        """Kill the whole cgroup, including descendants in other sessions."""
        native = vars(select)
        try:
            (self.directory / "cgroup.kill").write_bytes(b"1")
            with (self.directory / "cgroup.events").open("rb", buffering=0) as events:
                watcher = native["poll"]()
                watcher.register(events, native["POLLPRI"])
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    events.seek(0)
                    if b"populated 0" in events.read().splitlines():
                        return
                    watcher.poll(max(1, int((deadline - time.monotonic()) * 1000)))
        except FileNotFoundError:
            return
        raise RuntimeError("spawn.tree_not_stopped: Linux cgroup still populated")

    def close(self) -> None:
        """Release the held memory event descriptor."""
        if self.events is not None:
            self.events.close()
        if self.subgroup:
            with suppress(FileNotFoundError):
                self.directory.rmdir()


def start(
    args: Any, *, limit: int | None, options: dict[str, Any]
) -> tuple[subprocess.Popen[Any], LinuxScope]:
    """Start a child behind a pipe until its scope and memory limit are verified."""
    name = f"alphalattice-{uuid4().hex}.scope"
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={name}",
        "--property=MemoryAccounting=yes",
    ]
    if limit is not None:
        command.extend((f"--property=MemoryMax={limit}", "--property=OOMPolicy=kill"))
    original = list(args) if isinstance(args, list | tuple) else [args]
    ready_read, ready_write = os.pipe()
    start_read, start_write = os.pipe()
    with (
        os.fdopen(ready_read, "rb") as ready,
        os.fdopen(ready_write, "wb") as ready_child,
        os.fdopen(start_read, "rb") as start_child,
        os.fdopen(start_write, "wb") as release,
    ):
        pass_fds = set(options.pop("pass_fds", ())) | {ready_write, start_read}
        process = subprocess.Popen(
            [
                *command,
                "--",
                sys.executable,
                "-I",
                "-c",
                _BOOT,
                str(ready_write),
                str(start_read),
                *original,
            ],
            pass_fds=tuple(sorted(pass_fds)),
            creationflags=options.pop("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            **options,
        )
        ready_child.close()
        start_child.close()
        scope = None
        try:
            if not select.select([ready], [], [], 5)[0]:
                raise OSError("spawn.cgroup_admission_unproved: launch acknowledgment timed out")
            scope = _verify(ready.read(4096).decode().strip(), name, limit)
            release.write(b"1")
            release.flush()
            process.args = args
            return process, scope
        except BaseException:
            if scope is not None:
                with suppress(OSError):
                    scope.stop()
                scope.close()
            with suppress(ProcessLookupError):
                process.kill()
            process.wait(timeout=5)
            raise


def _scope_directory(record: str, name: str) -> Path:
    path = (_CGROUP / record.removeprefix("0::/")).resolve()
    if not record.startswith("0::/") or not path.is_relative_to(_CGROUP) or path.name != name:
        raise OSError("spawn.cgroup_admission_unproved")
    return path


def _verify(record: str, name: str, limit: int | None) -> LinuxScope:
    return _accounting(_scope_directory(record, name), limit)


def _accounting(path: Path, limit: int | None) -> LinuxScope:
    if limit is not None:
        maximum = (path / "memory.max").read_text().strip()
        page_size = int(vars(os)["sysconf"]("SC_PAGE_SIZE"))
        if maximum == "max" or not 0 <= limit - int(maximum) < page_size:
            raise OSError("spawn.cgroup_admission_unproved: MemoryMax differs")
    return LinuxScope(path)


def attach(pid: int, limit: int | None = None, *, supervisor: bool = False) -> LinuxScope:
    """Admit the serving process to a verified native scope."""
    name = f"alphalattice-{uuid4().hex}.scope"
    pids = ["2", str(pid), str(os.getpid())] if supervisor else ["1", str(pid)]
    properties = ["PIDs", "au", *pids, "MemoryAccounting", "b", "true"]
    if supervisor:
        properties += ["Delegate", "b", "true"]
    subprocess.run(
        [
            "busctl",
            "--user",
            "call",
            "org.freedesktop.systemd1",
            "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager",
            "StartTransientUnit",
            "ssa(sv)a(sa(sv))",
            name,
            "fail",
            "3" if supervisor else "2",
            *properties,
            "0",
        ],
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=True,
        timeout=5,
    )
    record = Path(f"/proc/{pid}/cgroup")
    # The synchronous start joins the manager's queued start job for this unit.
    subprocess.run(
        ["systemctl", "--user", "start", name],
        stdout=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=True,
        timeout=5,
    )
    path = _scope_directory(record.read_text().strip(), name)
    if not supervisor:
        return LinuxScope(path)

    # Keep the uncapped observer in a sibling cgroup. Its live scope retains the
    # serving cgroup's native memory peak and event counters after a kernel OOM.
    observer, serving = path / "observer", path / "serving"
    observer.mkdir()
    serving.mkdir()
    (observer / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
    (serving / "cgroup.procs").write_text(str(pid), encoding="ascii")
    (path / "cgroup.subtree_control").write_text("+memory", encoding="ascii")
    if limit is not None:
        (serving / "memory.max").write_text(str(limit), encoding="ascii")
        (serving / "memory.oom.group").write_text("1", encoding="ascii")
    if _scope_directory(record.read_text().strip(), "serving") != serving:
        raise OSError("spawn.cgroup_admission_unproved: Host is outside the capped cgroup")
    return _accounting(serving, limit)


def worker_scope(pid: int) -> LinuxScope:
    """Keep worker ownership below the serving Host's budget when it has a scope."""
    record = Path(f"/proc/{pid}/cgroup")
    serving = (_CGROUP / record.read_text().strip().removeprefix("0::/")).resolve()
    if (
        serving.is_relative_to(_CGROUP)
        and serving.name == "serving"
        and serving.parent.name.startswith("alphalattice-")
        and (serving.parent / "observer").is_dir()
    ):
        worker = serving / f"worker-{uuid4().hex}"
        worker.mkdir()
        try:
            (worker / "cgroup.procs").write_text(str(pid), encoding="ascii")
            if _scope_directory(record.read_text().strip(), worker.name) != worker:
                raise OSError("spawn.cgroup_admission_unproved: misplaced worker")
        except BaseException:
            with suppress(OSError):
                worker.rmdir()
            raise
        return LinuxScope(worker, subgroup=True)
    return attach(pid)


def is_host_observer(pid: int) -> bool:
    """Recognize the external reporter in the managed serving scope's sibling."""
    try:
        group = Path(f"/proc/{pid}/cgroup").read_text().strip()
    except FileNotFoundError:
        return False
    path = (_CGROUP / group.removeprefix("0::/")).resolve()
    return (
        group.startswith("0::/")
        and path.is_relative_to(_CGROUP)
        and path.name == "observer"
        and path.parent.name.startswith("alphalattice-")
        and path.parent.name.endswith(".scope")
        and (path.parent / "serving").is_dir()
    )
