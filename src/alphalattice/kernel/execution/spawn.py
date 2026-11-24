"""Start external processes with bounded, quiet, isolated execution.

``scripts/create_gpu_environment.py`` remains outside this owner under the no-GPU fence.
The ambient Job probe adds no Job, so its measurement sees the caller's Job.
UI QA scene services use ready-service breakaway and are stopped by ``scenes.py down``.
"""

from __future__ import annotations

import codecs
import ctypes
import logging
import math
import multiprocessing
import multiprocessing.connection
import os
import select
import signal
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, closing, contextmanager, suppress
from ctypes import wintypes
from typing import Any, Protocol, Self

from .cancel import ExecutionStopped, interruptible, stop_reason
from .darwin_tree import DarwinTree
from .darwin_tree import start as start_darwin
from .linux_pool import GuardedWorker, start_guarded_worker
from .linux_scope import LinuxScope, enable_child_subreaper, is_host_observer
from .linux_scope import attach as attach_linux_scope
from .linux_scope import available as linux_scope_available
from .linux_scope import start as start_linux_scope
from .linux_scope import worker_scope as linux_worker_scope

if sys.platform == "win32":
    import _winapi

DEFAULT_TIMEOUT = 300.0
REAP_TIMEOUT = 5.0
_CREATE_SUSPENDED = 0x4
_LOG = logging.getLogger(__name__)


class TreeMemoryLimit(RuntimeError):
    """The tree reached its memory budget and was stopped."""

    failure_class = "spawn.tree_memory_limit"


class TreeMemory:
    """Track one owned tree's aggregate peak and apply its declared policy."""

    def __init__(self, kind: str, budget_bytes: int | None, mode: str) -> None:
        """Bind a kind's declared budget and the mode for this tree."""
        self.kind = kind
        self.budget_bytes = budget_bytes
        self.mode = mode
        self.peak_bytes = 0
        self.breached = False

    @property
    def enforce(self) -> bool:
        """A tree with no declared budget only observes, regardless of mode."""
        return self.mode == "enforce" and self.budget_bytes is not None

    def _details(self) -> str:
        budget = "undeclared" if self.budget_bytes is None else self.budget_bytes
        return (
            f"kind={self.kind} budget_bytes={budget} peak_bytes={self.peak_bytes} mode={self.mode}"
        )

    def observe(self, measured_bytes: int, *, limit_event: bool = False) -> bool:
        """Record an aggregate charge and tell the owner whether to end the tree."""
        self.peak_bytes = max(self.peak_bytes, measured_bytes)
        if self.budget_bytes is None or (not limit_event and self.peak_bytes <= self.budget_bytes):
            return False
        if not self.breached:
            self.breached = True
            _LOG.warning("spawn.tree_memory_limit: %s", self._details())
        return self.enforce

    def finish(self) -> None:
        """Record the measured peak and policy at the tree's end."""
        _LOG.info("spawn.tree_memory: %s", self._details())

    def check(self) -> None:
        """Raise the named refusal for a tree ended by its budget."""
        if self.enforce and self.breached:
            raise TreeMemoryLimit(f"spawn.tree_memory_limit: {self._details()}")


def start_piped(args: list[str]) -> subprocess.Popen[bytes]:
    """Start a quiet binary request/reply service with the caller's environment."""
    return subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def watch_parent() -> None:
    """End a kept worker after its parent's sentinel becomes ready."""
    parent = multiprocessing.parent_process()
    if parent is not None:
        multiprocessing.connection.wait([parent.sentinel])
    os._exit(3)


class ProcessObservations(Protocol):
    """Use the caller's process inventory and wait implementation."""

    @property
    def NoSuchProcess(self) -> type[Exception]:
        """Identify a process that exited before inspection."""
        ...

    @property
    def STATUS_ZOMBIE(self) -> str:
        """Identify an exited process whose parent has yet to reap it."""
        ...

    def Process(self, pid: int, /) -> Any:
        """Return the process identified by its PID."""
        ...

    def process_iter(self) -> Iterable[Any]:
        """Enumerate observed processes."""
        ...

    def wait_procs(self, members: list[Any], /, *, timeout: float) -> tuple[list[Any], list[Any]]:
        """Return the exited and live members within a bounded wait."""
        ...


class TreeNotStopped(RuntimeError):
    """The child owner could not stop its complete process tree."""


def _command(args: Any) -> Any:
    if (
        isinstance(args, list | tuple)
        and args
        and isinstance(args[0], str)
        and args[0].lower() in {"python", "python.exe", "python3", "python3.exe"}
    ):
        return type(args)((sys.executable, *args[1:]))
    if isinstance(args, str) and args.lower() in {"python", "python.exe", "python3", "python3.exe"}:
        return sys.executable
    return args


def attached(
    args: list[str],
    *,
    env: dict[str, str],
    job_factory: Any,
    processes: ProcessObservations,
    default_environment: Callable[[], dict[str, str]],
    timeout: float = DEFAULT_TIMEOUT,
    memory: TreeMemory | None = None,
) -> int:
    """Run an interactive child with its caller's streams and owned lifetime."""
    args = _command(args)
    if sys.platform != "win32":
        return run(
            processes,
            default_environment,
            args,
            env=env,
            stdin=sys.stdin,
            timeout=timeout,
            memory=memory,
        ).returncode
    with (
        closing(job_factory() if memory is None else job_factory(memory=memory)) as job,
        ExitStack() as streams,
    ):
        startup = subprocess.STARTUPINFO()
        startup.dwFlags = subprocess.STARTF_USESTDHANDLES
        current, handles = _winapi.GetCurrentProcess(), []
        for field, kind in (
            ("hStdInput", _winapi.STD_INPUT_HANDLE),
            ("hStdOutput", _winapi.STD_OUTPUT_HANDLE),
            ("hStdError", _winapi.STD_ERROR_HANDLE),
        ):
            handle = _winapi.GetStdHandle(kind)
            if handle not in (None, -1):
                handle = _winapi.DuplicateHandle(
                    current, handle, current, 0, True, _winapi.DUPLICATE_SAME_ACCESS
                )
                streams.callback(_winapi.CloseHandle, handle)
                if handle & 3 != 3 or _winapi.GetFileType(handle) != _winapi.FILE_TYPE_CHAR:
                    handles.append(handle)
            setattr(startup, field, handle)
        startup.lpAttributeList = {"handle_list": handles}
        process, thread, pid, _ = _winapi.CreateProcess(
            None,
            subprocess.list2cmdline(args),
            None,
            None,
            bool(handles),
            _CREATE_SUSPENDED,
            {**env, "GIT_TERMINAL_PROMPT": "0"},
            None,
            startup,
        )
        try:
            job.assign_and_resume(process, thread)
            if _winapi.WaitForSingleObject(process, int(timeout * 1000)) == _winapi.WAIT_TIMEOUT:
                raise subprocess.TimeoutExpired(args, timeout)
            result = int(_winapi.GetExitCodeProcess(process))
        except BaseException:
            stop_tree(pid, processes=processes)
            raise
        finally:
            _winapi.CloseHandle(thread)
            _winapi.CloseHandle(process)
    if memory is not None:
        memory.check()
    return result


def ambient_job_probe(
    args: list[str], *, creationflags: int, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run the leaf Job probe hidden and bounded without changing its Job membership."""
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        _command(args),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        creationflags=creationflags | subprocess.CREATE_NO_WINDOW,
        env=env,
    )


def detached(
    args: list[str], *, log: Any, env: dict[str, str], processes: ProcessObservations
) -> subprocess.Popen[bytes]:
    """Start a workspace-owned Host outside the launching turn's process group."""
    args = _command(args)
    common = {
        "stdin": subprocess.DEVNULL,
        "stdout": log,
        "stderr": log,
        "env": {**env, "GIT_TERMINAL_PROMPT": "0"},
    }
    if sys.platform != "win32":
        return subprocess.Popen(
            args,
            start_new_session=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            **common,
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.IsProcessInJob.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    in_job = ctypes.c_int()
    flags = (
        subprocess.CREATE_NO_WINDOW
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_BREAKAWAY_FROM_JOB
    )
    common["stdin"] = subprocess.PIPE
    child = subprocess.Popen(args, creationflags=flags, **common)
    native: Any = child
    proved = kernel32.IsProcessInJob(
        ctypes.c_void_p(int(native._handle)), None, ctypes.byref(in_job)
    )
    error_code = ctypes.get_last_error() if not proved else 0
    if not proved or in_job.value:
        stop_tree(child.pid, processes=processes)
        child.wait(timeout=REAP_TIMEOUT)
        error = PermissionError("local_web.detached_breakaway_denied")
        error.winerror = error_code
        raise error
    if child.stdin is not None:
        child.stdin.close()  # Release a detached Host only after ambient breakaway is proved.
    return child


def stop_tree(
    pid: int, *, processes: ProcessObservations, stop_root: Callable[[], object] | None = None
) -> None:
    """Stop a process and its descendants, children first."""
    try:
        parent = processes.Process(pid)
        members = parent.children(recursive=True)
    except processes.NoSuchProcess:
        return
    if stop_root is None:
        members.append(parent)
    for member in reversed(members) if stop_root is not None else members:
        with suppress(processes.NoSuchProcess):
            member.kill()
    if stop_root is not None:
        # multiprocessing has the root's wait handle and reaps it itself.
        with suppress(ProcessLookupError):
            stop_root()
    _, alive = processes.wait_procs(members, timeout=REAP_TIMEOUT)
    if alive:
        raise TreeNotStopped(f"spawn.tree_not_stopped: {[p.pid for p in alive]}")


def stop_pool(pool: Any, *, processes: ProcessObservations) -> None:
    """Stop each exact pool root and its descendants without taking the manager's wait."""
    if (owners := getattr(pool, "_owned_process_trees", None)) is not None:
        for owner in owners.copy():
            owner._stop()
        return
    for child in list((getattr(pool, "_processes", None) or {}).values()):
        if child.pid is not None and child.exitcode is None:
            stop_tree(child.pid, processes=processes, stop_root=child.kill)


@contextmanager
def reap_detached_sessions(*, processes: ProcessObservations) -> Iterator[Callable[[int], None]]:
    """A dedicated Linux launcher adopts and reaps descendants across session boundaries.

    It yields `keep`: given the launcher's own child once started, an adopted orphan is reaped the
    moment it exits, as init reaps one, so a process that ended is gone to every reader rather than
    a zombie of the launcher until its close (a timed-out descendant, a stopped Host, 11-23)."""
    if sys.platform != "linux":
        yield lambda _pid: None
        return
    owner = processes.Process(os.getpid())
    if owner.children():
        raise TreeNotStopped("spawn.subreaper_requires_dedicated_launcher")
    try:
        enable_child_subreaper()
    except OSError as error:
        raise TreeNotStopped(str(error)) from error
    try:
        yield lambda kept: threading.Thread(
            target=_reap_adopted, args=(kept,), name="spawn-reap-adopted", daemon=True
        ).start()
    finally:
        # The caller closes its ordinary process group first. A detached Host
        # whose launcher has exited is still a child of this subreaper.
        deadline = time.monotonic() + REAP_TIMEOUT
        while members := owner.children(recursive=True):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TreeNotStopped(
                    f"spawn.tree_not_stopped: detached session members {[p.pid for p in members]}"
                )
            for member in reversed(members):
                with suppress(processes.NoSuchProcess):
                    member.kill()
            processes.wait_procs(members, timeout=min(remaining, 0.1))


def _reap_adopted(kept: int) -> None:
    """Reap each child of this subreaper as it exits, except `kept`, the launcher's own child,
    whose status its owner reads; the watch ends when `kept` exits or no child is left, and the
    launcher's close then stops and reaps what remains. waitid blocks, so it never spins."""
    for pid in iter(lambda: _next_exited(kept), kept):
        with suppress(ChildProcessError):
            os.waitpid(pid, 0)


def _next_exited(kept: int) -> int:
    """The pid of a child that has exited, left unreaped (WNOWAIT); `kept` when none is left."""
    linux = vars(os)  # Linux names, read where the platform has them, as getpgid is
    try:
        ended = linux["waitid"](linux["P_ALL"], 0, linux["WEXITED"] | linux["WNOWAIT"])
    except ChildProcessError:
        return kept
    return kept if ended is None else int(ended.si_pid)


def _in_group(member: Any, pid: int) -> bool:
    try:
        return bool(vars(os)["getpgid"](member.pid) == pid)
    except ProcessLookupError:
        return False


def _running(member: Any, processes: ProcessObservations) -> bool:
    try:
        return bool(member.status() != processes.STATUS_ZOMBIE)
    except processes.NoSuchProcess:
        return False


def _stop_group(
    pid: int,
    *,
    processes: ProcessObservations,
    observed: Iterable[Any] = (),
    lineage: DarwinTree | None = None,
) -> None:
    """Stop an owned POSIX group and its descendants in other sessions."""
    descendants = list(observed)
    if lineage is None:
        group_owned = True
        with suppress(processes.NoSuchProcess):
            descendants.extend(processes.Process(pid).children(recursive=True))
        candidates = processes.process_iter()
    else:
        root_live, descendants_by_origin = lineage.members(processes.process_iter())
        descendants.extend(descendants_by_origin)
        group_owned = root_live or any(_in_group(member, pid) for member in descendants)
        candidates = descendants
    # Capture before signalling: a dying member can leave the group before it exits.
    members = [member for member in candidates if member.pid != pid and _in_group(member, pid)]
    if group_owned:
        with suppress(ProcessLookupError):
            vars(os)["killpg"](pid, vars(signal)["SIGKILL"])
            members.extend(
                member
                for member in processes.process_iter()
                if member.pid != pid and member not in members and _in_group(member, pid)
            )
    if lineage is not None:
        _, late = lineage.members(processes.process_iter())
        members.extend(
            member for member in late if member not in members and _in_group(member, pid)
        )
        descendants.extend(late)
    for descendant in dict.fromkeys(descendants):
        # Once the group signal is sent, parent links can disappear. Use the
        # captured process identities and their PID-reuse-safe kill operation.
        with suppress(processes.NoSuchProcess):
            descendant.kill()
        if descendant not in members:
            members.append(descendant)
    deadline = time.monotonic() + REAP_TIMEOUT
    while members:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TreeNotStopped(
                f"spawn.tree_not_stopped: group members {[p.pid for p in members]}"
            )
        _, alive = processes.wait_procs(members, timeout=min(remaining, 0.1))
        members = []
        members.extend(member for member in alive if _running(member, processes))


class _PosixMemoryWatch:
    """Observe owned process identities and aggregate memory while the tree lives."""

    def __init__(
        self,
        pid: int,
        processes: ProcessObservations,
        memory: TreeMemory | None,
        scope: LinuxScope | None,
        *,
        current: bool = False,
        lineage: DarwinTree | None = None,
        pool_child: Any = None,
    ) -> None:
        self.pid, self.processes, self.memory = pid, processes, memory
        self.scope = scope
        self.current = current
        self.lineage = lineage
        self._pool_child = pool_child
        self._ending = threading.Event()
        self._stopped = False
        self._lock = threading.RLock()
        self._members: list[Any] = []
        self._failure: BaseException | None = None
        self._wake = os.pipe() if scope is not None and scope.events is not None else None
        if self._wake is None:
            with suppress(processes.NoSuchProcess):
                self._members.append(processes.Process(pid))
        self._thread = threading.Thread(target=self._watch, name="spawn-tree-memory", daemon=True)
        self._thread.start()

    def _sample(self) -> int:
        with self._lock:
            return self._sample_owned()

    def _sample_owned(self) -> int:
        live = [
            member
            for member in self._members
            if member.is_running() and _running(member, self.processes)
        ]
        for ancestor in live.copy():
            with suppress(self.processes.NoSuchProcess):
                for member in ancestor.children(recursive=True):
                    if member not in live:
                        live.append(member)
        self._members = live
        if self.lineage is not None:
            self.lineage.remember(live)
            _, descendants = self.lineage.members(self.processes.process_iter())
            live.extend(member for member in descendants if member not in live)
            self.lineage.remember(descendants)
        used = 0
        if self.memory is not None:
            for member in live:
                with suppress(self.processes.NoSuchProcess):
                    used += int(member.memory_info().rss)
        return used

    def _watch(self) -> None:
        try:
            while True:
                if self._ending.is_set():
                    return
                amount, reached = (
                    self.scope.sample()
                    if self.scope is not None and self._wake is not None
                    else (self._sample(), False)
                )
                if self.memory is not None and self.memory.observe(amount, limit_event=reached):
                    break
                if self.scope is not None and self._wake is not None:
                    self.scope.wait(self._wake[0])
                else:
                    self._ending.wait(1.0)
        except Exception as error:
            self._failure = error
        if (
            self.memory is not None
            and self.memory.enforce
            and (self.memory.breached or self._failure is not None)
        ):
            try:
                self._stop()
            except Exception as error:
                self._failure = error

    def _stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            if self.scope is not None:
                self.scope.stop()
            elif self._pool_child is not None:
                self._stop_pool_child()
            elif self.current:
                members = self._members
                if self.lineage is not None:
                    _, descendants = self.lineage.members(self.processes.process_iter())
                    members = [*members, *descendants]
                for member in reversed(members):
                    if member.pid != self.pid:
                        with suppress(self.processes.NoSuchProcess):
                            member.kill()
                os._exit(1)
            else:
                _stop_group(
                    self.pid,
                    processes=self.processes,
                    observed=(member for member in self._members if member.pid != self.pid),
                    lineage=self.lineage,
                )
            self._stopped = True

    def _stop_pool_child(self) -> None:
        if isinstance(self._pool_child, GuardedWorker):
            self._pool_child.stop()
            return
        self._sample()
        if self._pool_child.exitcode is None:
            with suppress(ProcessLookupError):
                self._pool_child.kill()
        for member in self._members:
            if member.pid != self.pid:
                with suppress(self.processes.NoSuchProcess):
                    member.kill()
        _, alive = self.processes.wait_procs(
            [member for member in self._members if member.pid != self.pid], timeout=REAP_TIMEOUT
        )
        if live := [member.pid for member in alive if _running(member, self.processes)]:
            raise TreeNotStopped(f"spawn.tree_not_stopped: {live}")

    def close(self, *, finish_host_report: bool = False) -> None:
        """End the scope and keep the observed peak before handing back its result."""
        self._ending.set()
        if self._wake is not None:
            os.write(self._wake[1], b"1")
        try:
            self._thread.join(timeout=REAP_TIMEOUT + 1.0)
            if not self.current:
                if finish_host_report and not self._stopped:
                    reporters = [
                        member
                        for member in self.processes.process_iter()
                        if _in_group(member, self.pid) and is_host_observer(member.pid)
                    ]
                    self.processes.wait_procs(reporters, timeout=REAP_TIMEOUT + 1.0)
                self._stop()
            if self.memory is not None:
                self.memory.finish()
            if self._thread.is_alive() or self._failure is not None:
                raise TreeNotStopped("spawn.tree_not_stopped: memory watch unconfirmed") from (
                    self._failure
                )
        finally:
            if self.scope is not None:
                self.scope.close()
            if isinstance(self._pool_child, GuardedWorker):
                self._pool_child.close_control()
            if self._wake is not None:
                for descriptor in self._wake:
                    os.close(descriptor)


def own_pool_trees(pool: Any, processes: ProcessObservations, memory: TreeMemory) -> None:
    """Anchor the worker before its initializer, keeping ownership after manager retirement."""
    launch = pool._spawn_process
    arguments = pool._initargs
    owners: list[_PosixMemoryWatch] = []
    pool._owned_process_trees = owners

    def observe(worker: Any, scope: LinuxScope | None = None) -> None:
        lineage = DarwinTree(worker.pid) if sys.platform == "darwin" else None
        owners.append(
            _PosixMemoryWatch(
                worker.pid,
                processes,
                TreeMemory(memory.kind, memory.budget_bytes, memory.mode),
                scope,
                lineage=lineage,
                pool_child=worker,
            )
        )

    def start() -> None:
        has_linux_scope = memory.mode == "enforce" and linux_scope_available()
        if sys.platform == "linux" and not has_linux_scope:
            observe(start_guarded_worker(pool, launch, arguments))
            return
        previous = set(pool._processes)
        reader, release = pool._mp_context.Pipe(duplex=False)
        with reader, release:
            pool._initargs = (*arguments, reader)
            try:
                launch()
            finally:
                pool._initargs = arguments
            worker = next(child for pid, child in pool._processes.items() if pid not in previous)
            observe(worker, linux_worker_scope(worker.pid) if has_linux_scope else None)
            release.send_bytes(b"1")

    pool._spawn_process = start


def close_pool_trees(pool: Any) -> None:
    """Collect the retained POSIX trees after their executor joined the workers."""
    for owner in getattr(pool, "_owned_process_trees", ()):
        owner.close()
    if hasattr(pool, "_owned_process_trees"):
        pool._owned_process_trees.clear()


def _start_posix(
    args: Any, processes: ProcessObservations, memory: TreeMemory | None, **options: Any
) -> tuple[subprocess.Popen[Any], _PosixMemoryWatch | None]:
    scope = None
    lineage = None
    if sys.platform == "darwin":
        process, lineage = start_darwin(args, options=options)
    elif (
        memory is not None
        and memory.enforce
        and not options.get("shell")
        and linux_scope_available()
    ):
        process, scope = start_linux_scope(args, limit=memory.budget_bytes, options=options)
    else:
        process = subprocess.Popen(
            args,
            creationflags=options.pop("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            **options,
        )
    try:
        watch = (
            _PosixMemoryWatch(process.pid, processes, memory, scope, lineage=lineage)
            if memory is not None or lineage is not None
            else None
        )
    except BaseException:
        try:
            if scope is not None:
                scope.stop()
            else:
                _stop_group(process.pid, processes=processes, lineage=lineage)
        finally:
            if scope is not None:
                scope.close()
            process.wait(timeout=REAP_TIMEOUT)
        raise
    return process, watch


@contextmanager
def current_tree(processes: ProcessObservations, memory: TreeMemory) -> Iterator[None]:
    """Own a real serving process throughout its operating-system lifetime."""
    if sys.platform == "win32":
        job = ChildJob(memory=memory)
        try:
            job.assign(_winapi.GetCurrentProcess())
        except BaseException:
            job.close()
            raise
        try:
            yield
        finally:
            job.finish_memory()
            # The serving process is in its own Job. Windows releases its last
            # handle, and kills any remaining member, when this process ends.
    elif memory.enforce and linux_scope_available():
        with _supervised_linux_host(memory):
            yield
    else:
        # Observe a Linux Host in place. Only enforce can move it into a cgroup
        # since an attach must finish before the serving code starts children.
        optional = sys.platform == "linux" and memory.mode == "observe"
        lineage = DarwinTree(os.getpid()) if sys.platform == "darwin" else None
        try:
            watch = _PosixMemoryWatch(
                os.getpid(), processes, memory, None, current=True, lineage=lineage
            )
        except (OSError, RuntimeError):
            if not optional:
                raise
            _LOG.warning("spawn.host_observer_unavailable: kind=%s phase=start", memory.kind)
            yield
            return
        try:
            yield
        finally:
            try:
                watch.close()
            except TreeNotStopped:
                if not optional:
                    raise
                _LOG.warning("spawn.host_observer_unavailable: kind=%s phase=finish", memory.kind)


def _observe_linux_host(
    pid: int, memory: TreeMemory, host: multiprocessing.connection.Connection
) -> None:
    """Report a hard-limit event from outside the serving process's cgroup."""
    with host, ExitStack() as resources:
        departure = vars(os)["pidfd_open"](pid)
        resources.callback(os.close, departure)
        limit = memory.budget_bytes if memory.enforce else None
        scope = attach_linux_scope(pid, limit, supervisor=True)
        stopped = False
        try:
            host.send_bytes(b"ready")
            while not host.poll() and not select.select([departure], [], [], 0)[0]:
                peak, reached = scope.sample()
                if memory.observe(peak, limit_event=reached):
                    return
                scope.wait(host.fileno(), departure)
            if host.poll():
                with suppress(EOFError):
                    stopped = host.recv_bytes() == b"done"
            peak, reached = scope.sample()
            memory.observe(peak, limit_event=reached)
        finally:
            try:
                if not stopped or (memory.enforce and memory.breached):
                    scope.stop()
            finally:
                scope.close()
                memory.finish()


@contextmanager
def _supervised_linux_host(memory: TreeMemory) -> Iterator[None]:
    """Keep the native hard limit's reporter alive if its Host is killed by the kernel."""
    context = multiprocessing.get_context("spawn")
    owner, channel = context.Pipe()
    observer = context.Process(target=_observe_linux_host, args=(os.getpid(), memory, channel))
    with owner, channel:
        observer.start()
        channel.close()
        admitted = False
        try:
            with suppress(EOFError):
                admitted = owner.poll(2 * REAP_TIMEOUT + 1) and owner.recv_bytes() == b"ready"
            if not admitted:
                raise OSError("spawn.cgroup_admission_unproved: missing Host observer")
            yield
        finally:
            with suppress(BrokenPipeError, ConnectionResetError):
                owner.send_bytes(b"done")
            owner.close()
            observer.join(timeout=2 * REAP_TIMEOUT)
            if observer.is_alive():
                observer.kill()
                observer.join(timeout=REAP_TIMEOUT)
                raise TreeNotStopped("spawn.tree_not_stopped: Host observer did not retire")
            exit_code = observer.exitcode
            observer.close()
            if admitted and exit_code != 0:
                raise TreeNotStopped("spawn.tree_not_stopped: Host observer failed")


class JobProcess(subprocess.Popen[Any]):
    """Keep Popen's pipe and wait ownership while admitting its primary thread to a Job."""

    def __init__(
        self,
        job: ChildJob,
        default_environment: Callable[[], dict[str, str]],
        args: Any,
        *,
        creationflags: int,
        **kwargs: Any,
    ) -> None:
        """Start ``args`` suspended, so ``job`` admits it before its first instruction."""
        self._job = job
        self._default_environment = default_environment
        super().__init__(args, creationflags=creationflags | _CREATE_SUSPENDED, **kwargs)

    if sys.platform == "win32":

        def _execute_child(
            self,
            args: Any,
            executable: str | bytes | os.PathLike[str] | None,
            _preexec_fn: Any,
            close_fds: bool,
            pass_fds: tuple[int, ...],
            cwd: str | bytes | os.PathLike[str] | None,
            env: Any,
            startupinfo: subprocess.STARTUPINFO | None,
            creationflags: int,
            shell: bool,
            p2cread: Any,
            p2cwrite: Any,
            c2pread: Any,
            c2pwrite: Any,
            errread: Any,
            errwrite: Any,
            *_posix_options: Any,
        ) -> None:
            # CPython 3.12 Popen supplies the prepared pipe handles here. Its Windows
            # implementation closes the primary thread handle before returning.
            if pass_fds:
                raise AssertionError("pass_fds not supported on Windows.")
            if not isinstance(args, str):
                if shell and isinstance(args, (bytes, os.PathLike)):
                    raise TypeError("bytes and path-like args cannot be used with shell on Windows")
                args = subprocess.list2cmdline(
                    [args] if isinstance(args, (bytes, os.PathLike)) else args
                )
            program = None if executable is None else os.fsdecode(executable)
            directory = None if cwd is None else os.fsdecode(cwd)
            startup = subprocess.STARTUPINFO() if startupinfo is None else startupinfo.copy()
            stream_handles = (p2cread, c2pwrite, errwrite)
            redirected = -1 not in stream_handles
            if redirected:
                startup.dwFlags |= _winapi.STARTF_USESTDHANDLES
                startup.hStdInput, startup.hStdOutput, startup.hStdError = stream_handles
            native: Any = self
            attributes = startup.lpAttributeList or {}
            handles = list(attributes.get("handle_list", []))
            if handles or (redirected and close_fds):
                if redirected:
                    handles.extend(map(int, stream_handles))
                handles = native._filter_handle_list(handles)
                startup.lpAttributeList = {**attributes, "handle_list": handles}
                if handles and not close_fds:
                    warnings.warn(
                        "startupinfo.lpAttributeList['handle_list'] overriding close_fds",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if shell:
                startup.dwFlags |= _winapi.STARTF_USESHOWWINDOW
                startup.wShowWindow = _winapi.SW_HIDE
                parent_env = self._default_environment() if not program else {}
                command_shell = program or parent_env.get("COMSPEC")
                if not command_shell:
                    command_shell = os.path.join(
                        parent_env.get("SYSTEMROOT", ""), "System32", "cmd.exe"
                    )
                    if not os.path.isabs(command_shell):
                        raise FileNotFoundError(
                            "shell not found: neither %ComSpec% nor %SystemRoot% is set"
                        )
                if not program and os.path.isabs(command_shell):
                    program = command_shell
                args = f'{command_shell} /c "{args}"'
            sys.audit("subprocess.Popen", program, args, directory, env)

            process = thread = None
            try:
                try:
                    process, thread, pid, _ = _winapi.CreateProcess(
                        program,
                        args,
                        None,
                        None,
                        bool(handles) or not close_fds,
                        creationflags,
                        env,
                        directory,
                        startup,
                    )
                finally:
                    native._close_pipe_fds(p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite)
                self._job.assign_and_resume(process, thread)
                native._handle = vars(subprocess)["Handle"](process)
                self.pid = pid
                native._child_created = True
            except BaseException:
                if process is not None:
                    try:
                        if _winapi.WaitForSingleObject(process, 0) == _winapi.WAIT_TIMEOUT:
                            _winapi.TerminateProcess(process, 1)
                            if (
                                _winapi.WaitForSingleObject(process, int(REAP_TIMEOUT * 1000))
                                == _winapi.WAIT_TIMEOUT
                            ):
                                raise TreeNotStopped("spawn.tree_not_stopped: Job admission failed")
                    finally:
                        _winapi.CloseHandle(process)
                raise
            finally:
                if thread is not None:
                    _winapi.CloseHandle(thread)


def run(
    processes: ProcessObservations,
    default_environment: Callable[[], dict[str, str]],
    /,
    args: Any,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run one child until it exits or its finite deadline ends its tree."""
    if (reason := stop_reason()) is not None:
        raise ExecutionStopped(reason)
    args = _command(args)
    timeout = kwargs.pop("timeout", DEFAULT_TIMEOUT)
    input = kwargs.pop("input", None)
    capture_output = kwargs.pop("capture_output", False)
    check = kwargs.pop("check", False)
    memory = kwargs.pop("memory", None)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("spawn.timeout_invalid: provide a finite positive timeout")
    supplied = kwargs.pop("env", None)
    environment = default_environment() if supplied is None else dict(supplied)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    kwargs["env"] = environment
    creationflags = kwargs.pop("creationflags", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    encoding = kwargs.get("encoding")
    if encoding is not None:
        try:
            valid_encoding = codecs.lookup(encoding).name == "utf-8"
        except LookupError:
            valid_encoding = False
        if not valid_encoding:
            raise ValueError("spawn.encoding_invalid: text children must use UTF-8")
    if (
        kwargs.get("text")
        or kwargs.get("universal_newlines")
        or encoding is not None
        or kwargs.get("errors") is not None
    ):
        kwargs["encoding"] = "utf-8"
    if input is not None:
        if "stdin" in kwargs:
            raise ValueError("stdin and input arguments may not both be used")
        kwargs["stdin"] = subprocess.PIPE
    else:
        if kwargs.get("stdin") is None:
            kwargs["stdin"] = subprocess.DEVNULL
        elif kwargs["stdin"] == subprocess.PIPE:
            raise ValueError("spawn.stdin_unfed: supply input or close stdin")
    if capture_output:
        if "stdout" in kwargs or "stderr" in kwargs:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    job = ChildJob(memory=memory) if sys.platform == "win32" else None
    if job is None:
        kwargs["start_new_session"] = True
    watch = None
    try:
        if job is None:
            process, watch = _start_posix(
                args, processes, memory, creationflags=creationflags, **kwargs
            )
        else:
            process = JobProcess(
                job, default_environment, args, creationflags=creationflags, **kwargs
            )
    except BaseException:
        if job is not None:
            job.close()
        raise

    group_stopped = False
    group_lock = threading.Lock()

    def stop_group() -> None:
        nonlocal group_stopped
        with group_lock:
            if not group_stopped:
                _stop_group(process.pid, processes=processes)
                group_stopped = True

    def end_owned_tree(*, finish_host_report: bool) -> None:
        if job is not None:
            job.close()  # the Job still owns descendants after the direct parent exits
        elif watch is not None:
            watch.close(finish_host_report=finish_host_report)
        else:
            stop_group()

    def request_stop() -> None:
        if job is not None:
            job.terminate()
        elif watch is not None:
            watch._stop()
        else:
            stop_group()

    ended_normally = False
    try:
        with interruptible(request_stop):
            if (reason := stop_reason()) is not None:
                raise ExecutionStopped(reason)
            stdout, stderr = process.communicate(input=input, timeout=timeout)
        if (reason := stop_reason()) is not None:
            raise ExecutionStopped(reason)
        ended_normally = True
    except BaseException as error:
        try:
            request_stop()
        finally:
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate(timeout=REAP_TIMEOUT)
        if isinstance(error, subprocess.TimeoutExpired):
            error.output, error.stderr = stdout, stderr
        raise
    finally:
        try:
            end_owned_tree(
                finish_host_report=(
                    ended_normally
                    and sys.platform == "linux"
                    and memory is not None
                    and memory.kind == "attached_retrieval_host_child"
                )
            )
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
    result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if memory is not None:
        memory.check()
    if check:
        result.check_returncode()
    return result


class InteractiveSession:
    """Own a bounded multi-turn child; its streams and stop scope are explicit."""

    def __init__(
        self,
        *,
        processes: ProcessObservations,
        default_environment: Callable[[], dict[str, str]],
        timeout: float,
        stop_scope: str,
        affinity: int | None = None,
        priority: int | None = None,
        exit_code: int = 1,
        memory: TreeMemory | None = None,
    ) -> None:
        """Set the finite per-turn bound and whether teardown owns a tree or its root."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int | float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("spawn.timeout_invalid: provide a finite positive timeout")
        if stop_scope not in {"tree", "root"}:
            raise ValueError("spawn.stop_scope_invalid")
        self._processes = processes
        self._default_environment = default_environment
        self.timeout, self.stop_scope = timeout, stop_scope
        self.affinity, self.priority, self.exit_code = affinity, priority, exit_code
        self.memory = memory
        self.child: subprocess.Popen[Any] | None = None
        self.job: ChildJob | None = None
        self._memory_watch: _PosixMemoryWatch | None = None
        self.stopped = False
        self.timed_out = False
        self.failure: BaseException | None = None
        self.timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._interrupts = ExitStack()
        self._generation = 0

    def start(self, args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        """Start once, admitting the suspended child to its Job before execution."""
        if self.stopped or self.child is not None:
            raise RuntimeError("spawn.session_terminal")
        stdin = kwargs.pop("stdin", subprocess.PIPE)
        if stdin not in (subprocess.PIPE, subprocess.DEVNULL):
            raise ValueError("spawn.session_owns_stdin")
        supplied = kwargs.pop("env", None)
        env = self._default_environment() if supplied is None else dict(supplied)
        env["GIT_TERMINAL_PROMPT"] = "0"
        kwargs.update(stdin=stdin, env=env)
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        kwargs.setdefault("stderr", subprocess.DEVNULL)
        creationflags = kwargs.pop("creationflags", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if (
            kwargs.get("text")
            or kwargs.get("encoding") is not None
            or kwargs.get("errors") is not None
        ):
            if (
                kwargs.get("encoding") is not None
                and codecs.lookup(kwargs["encoding"]).name != "utf-8"
            ):
                raise ValueError("spawn.encoding_invalid: text children must use UTF-8")
            kwargs["encoding"] = "utf-8"
        if self.stop_scope == "tree":
            if sys.platform == "win32":
                if self.job is None:
                    self.job = ChildJob(
                        affinity=self.affinity,
                        priority=self.priority,
                        memory=self.memory,
                    )
            else:
                kwargs["start_new_session"] = True
        try:
            command = _command(args)
            if self.job is not None:
                self.child = JobProcess(
                    self.job,
                    self._default_environment,
                    command,
                    creationflags=creationflags,
                    **kwargs,
                )
            elif self.stop_scope == "tree":
                self.child, self._memory_watch = _start_posix(
                    command,
                    self._processes,
                    self.memory,
                    creationflags=creationflags,
                    **kwargs,
                )
            else:
                self.child = subprocess.Popen(command, creationflags=creationflags, **kwargs)
        except BaseException:
            if self.job is not None:
                ChildJob.close(self.job)
            if self.child is not None and self.child.poll() is None:
                self.child.kill()
                self.child.wait(timeout=REAP_TIMEOUT)
            raise
        self._arm()
        self._interrupts.enter_context(interruptible(self.stop))
        if (reason := stop_reason()) is not None:
            self.stop()
            raise ExecutionStopped(reason)
        return self.child

    def _arm(self) -> None:
        with self._lock:
            if self.stopped:
                raise RuntimeError("spawn.session_terminal")
            if self.timer is not None:
                self.timer.cancel()
            self._generation += 1
            self.timer = threading.Timer(self.timeout, self._expire, args=(self._generation,))
            self.timer.daemon = True
            self.timer.start()

    def _expire(self, generation: int) -> None:
        with self._lock:
            if self.stopped or generation != self._generation:
                return
            self.timed_out = True
        try:
            self.stop()
        except BaseException as error:
            self.failure = error

    def check(self) -> None:
        """Report an expired deadline or an unproved stop to the caller."""
        if self.failure is not None:
            raise TreeNotStopped("spawn.session_stop_unconfirmed") from self.failure
        if self.memory is not None:
            self.memory.check()
        if (reason := stop_reason()) is not None:
            raise ExecutionStopped(reason)
        if self.timed_out:
            raise subprocess.TimeoutExpired(self.child.args if self.child else [], self.timeout)

    def send(self, data: Any) -> None:
        """Feed one turn and renew its finite response deadline."""
        self.check()
        if self.child is None or self.child.stdin is None or self.child.stdin.closed:
            raise RuntimeError("spawn.session_input_closed")
        self.child.stdin.write(data)
        self.child.stdin.flush()
        self.check()
        self._arm()

    def close_input(self) -> None:
        """Send EOF when no further turns will be supplied."""
        if self.child is not None and self.child.stdin is not None:
            self.child.stdin.close()

    def stop(self) -> None:
        """Stop the owned scope and wait for its bounded termination barrier."""
        with self._lock:
            first = not self.stopped
            if first:
                self.stopped = True
                if self.timer is not None:
                    self.timer.cancel()
            child = self.child
        if not first:
            if not self._finished.wait(timeout=REAP_TIMEOUT * 3):
                raise TreeNotStopped("spawn.session_stop_unconfirmed")
            if self.failure is not None:
                raise TreeNotStopped("spawn.session_stop_unconfirmed") from self.failure
            return
        try:
            if child is not None:
                if self.job is not None:
                    self.job.terminate(exit_code=self.exit_code)
                elif self._memory_watch is not None:
                    self._memory_watch.close()
                elif self.stop_scope == "tree":
                    _stop_group(child.pid, processes=self._processes)
                elif child.poll() is None:
                    child.kill()
                if child.poll() is None:
                    child.wait(timeout=REAP_TIMEOUT)
        except BaseException as error:
            self.failure = error
            raise
        finally:
            try:
                self.close_input()
                if child is not None:
                    for stream in (child.stdout, child.stderr):
                        if stream is not None:
                            stream.close()
            except BaseException as error:
                self.failure = error
                raise
            finally:
                self._finished.set()

    def __enter__(self) -> Self:
        """Keep the session within its caller's cleanup scope."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Stop the child and release its Job even when the caller fails."""
        try:
            self._interrupts.close()
            self.stop()
        finally:
            if self.job is not None:
                ChildJob.close(self.job)


def run_root(
    processes: ProcessObservations,
    default_environment: Callable[[], dict[str, str]],
    /,
    args: Any,
    *,
    timeout: float,
    check: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Bound a normal-Host launcher while leaving its ready Host outside its root scope."""
    with InteractiveSession(
        processes=processes,
        default_environment=default_environment,
        timeout=timeout,
        stop_scope="root",
    ) as session:
        child = session.start(args, **kwargs)
        child.wait(timeout=timeout)
        session.check()
        result: subprocess.CompletedProcess[Any] = subprocess.CompletedProcess(
            args, child.returncode
        )
        if check:
            result.check_returncode()
        return result


class _Limits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _Limits),
        ("IoInfo", ctypes.c_ulonglong * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _CompletionPort(ctypes.Structure):
    _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", wintypes.HANDLE)]


class _NotificationLimits(ctypes.Structure):
    _fields_ = [
        ("IoReadBytesLimit", ctypes.c_ulonglong),
        ("IoWriteBytesLimit", ctypes.c_ulonglong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("JobMemoryLimit", ctypes.c_ulonglong),
        ("RateControlTolerance", ctypes.c_int),
        ("RateControlToleranceInterval", ctypes.c_int),
        ("LimitFlags", wintypes.DWORD),
    ]


_JOB_MEMORY_LIMIT = 0x200
_JOB_MEMORY_MESSAGE = 10
_JOB_NOTIFICATION_MESSAGE = 11
_JOB_COMPLETION_KEY = 1


class ChildJob:
    """Keep the sole non-inheritable kill-on-close Job handle in its parent owner."""

    @staticmethod
    def current_limit_flags() -> int | None:
        """Read this process's current Job limits; unknown facts remain unknown."""
        if sys.platform != "win32":
            return None
        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.QueryInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_void_p,
            ]
            kernel.QueryInformationJobObject.restype = wintypes.BOOL
            limits = _ExtendedLimits()
            if kernel.QueryInformationJobObject(
                None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None
            ):
                return int(limits.BasicLimitInformation.LimitFlags)
        except OSError:
            pass
        return None

    def __init__(
        self,
        *,
        affinity: int | None = None,
        priority: int | None = None,
        memory: TreeMemory | None = None,
    ) -> None:
        """Create this parent's Job with its caller's optional affinity and priority."""
        if sys.platform != "win32":
            raise OSError("Windows Jobs are unavailable")
        self._lock = threading.Lock()
        self.memory = memory
        self._memory_port: int | None = None
        self._memory_thread: threading.Thread | None = None
        self._memory_failure: OSError | None = None
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.WaitForMultipleObjects.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self.kernel.WaitForMultipleObjects.restype = wintypes.DWORD
        self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.ResumeThread.restype = wintypes.DWORD
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("owned Job unavailable")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if affinity is not None:
            limits.BasicLimitInformation.LimitFlags |= 0x10
            limits.BasicLimitInformation.Affinity = affinity
        if priority is not None:
            limits.BasicLimitInformation.LimitFlags |= 0x20
            limits.BasicLimitInformation.PriorityClass = priority
        if memory is not None and memory.enforce:
            assert memory.budget_bytes is not None
            limits.BasicLimitInformation.LimitFlags |= _JOB_MEMORY_LIMIT
            limits.JobMemoryLimit = memory.budget_bytes
        try:
            if not self.kernel.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise OSError("owned Job limit unavailable")
            if memory is not None and memory.budget_bytes is not None:
                self._watch_job_memory(memory.budget_bytes, enforce=memory.enforce)
        except BaseException:
            if self._memory_port is not None:
                _winapi.CloseHandle(self._memory_port)
            _winapi.CloseHandle(self.handle)
            self.handle = None
            raise

    def _watch_job_memory(self, budget: int, *, enforce: bool) -> None:
        self.kernel.CreateIoCompletionPort.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.c_size_t,
            wintypes.DWORD,
        ]
        self.kernel.CreateIoCompletionPort.restype = wintypes.HANDLE
        self.kernel.GetQueuedCompletionStatus.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.DWORD,
        ]
        self.kernel.PostQueuedCompletionStatus.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        port = self.kernel.CreateIoCompletionPort(ctypes.c_void_p(-1), None, 0, 1)
        if not port:
            raise OSError("spawn.job_memory_unavailable: open completion port")
        self._memory_port = port
        association = _CompletionPort(_JOB_COMPLETION_KEY, port)
        if not self.kernel.SetInformationJobObject(
            self.handle, 7, ctypes.byref(association), ctypes.sizeof(association)
        ):
            raise OSError("spawn.job_memory_unavailable: associate completion port")
        if not enforce:
            notice = _NotificationLimits(JobMemoryLimit=budget, LimitFlags=_JOB_MEMORY_LIMIT)
            if not self.kernel.SetInformationJobObject(
                self.handle, 12, ctypes.byref(notice), ctypes.sizeof(notice)
            ):
                raise OSError("spawn.job_memory_unavailable: set notification limit")
        watch = threading.Thread(target=self._read_job_memory, args=(port,), daemon=True)
        watch.start()
        self._memory_thread = watch

    def _job_peak(self) -> int:
        value = _ExtendedLimits()
        if not self.kernel.QueryInformationJobObject(
            self.handle, 9, ctypes.byref(value), ctypes.sizeof(value), None
        ):
            raise OSError("spawn.job_memory_unavailable: read peak")
        return int(value.PeakJobMemoryUsed)

    def _read_job_memory(self, port: int) -> None:
        memory = self.memory
        assert memory is not None
        code, key, process = wintypes.DWORD(), ctypes.c_size_t(), ctypes.c_void_p()
        try:
            while self.kernel.GetQueuedCompletionStatus(
                port, ctypes.byref(code), ctypes.byref(key), ctypes.byref(process), 0xFFFFFFFF
            ):
                if key.value == 0:
                    return  # Close posts specifically to this port.
                if key.value == _JOB_COMPLETION_KEY and code.value in {
                    _JOB_MEMORY_MESSAGE,
                    _JOB_NOTIFICATION_MESSAGE,
                }:
                    if memory.observe(self._job_peak(), limit_event=True):
                        self.terminate()
                    return  # One breach verdict per tree.
            raise OSError("spawn.job_memory_unavailable: receive notification")
        except OSError as error:
            self._memory_failure = error
            if memory.enforce:
                with suppress(OSError, TreeNotStopped):
                    self.terminate()

    def assign(self, process: int) -> None:
        """Assign the caller-owned process handle to this Job."""
        if sys.platform != "win32":
            raise OSError("Windows Jobs are unavailable")
        if not self.kernel.AssignProcessToJobObject(self.handle, process):
            raise ctypes.WinError(ctypes.get_last_error())

    def assign_and_resume(self, process: int, thread: int) -> None:
        """Admit the suspended child before it can run any Host code."""
        if sys.platform != "win32":
            raise OSError("Windows Jobs are unavailable")
        self.assign(process)
        if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self, *, exit_code: int = 1) -> None:
        """Stop this Job and prove each captured member exited, keeping the Job readable."""
        with self._lock:
            self._terminate(exit_code)

    def _terminate(self, exit_code: int) -> None:
        if sys.platform != "win32":
            raise OSError("Windows Jobs are unavailable")
        if not self.handle:
            return
        deadline = time.monotonic() + REAP_TIMEOUT
        held: dict[int, int] = {}

        def accounting() -> _Accounting:
            value = _Accounting()
            if not self.kernel.QueryInformationJobObject(
                self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return value

        def capture(count: int) -> set[int]:
            # A bounded buffer also admits members racing the accounting read.
            size = 8 + ctypes.sizeof(ctypes.c_size_t) * max(count + 64, 64)
            members = ctypes.create_string_buffer(size)
            if not self.kernel.QueryInformationJobObject(self.handle, 3, members, size, None):
                raise TreeNotStopped("spawn.tree_not_stopped: Job members unmeasured")
            length = wintypes.DWORD.from_buffer(members, 4).value
            pids = (ctypes.c_size_t * length).from_buffer(members, 8)
            acquired = set()
            for pid in pids:
                if pid in held:
                    continue
                handle = self.kernel.OpenProcess(0x00100000, False, pid)
                if handle:
                    held[pid] = handle
                    acquired.add(pid)
                elif ctypes.get_last_error() != 87:
                    raise TreeNotStopped(f"spawn.tree_not_stopped: cannot open member {pid}")
            return acquired

        try:
            before = accounting()
            if not before.ActiveProcesses:
                return
            capture(before.ActiveProcesses)
            limits = _ExtendedLimits()
            if not self.kernel.QueryInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), None
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            limits.BasicLimitInformation.LimitFlags |= 0x8
            limits.BasicLimitInformation.ActiveProcessLimit = before.ActiveProcesses
            if not self.kernel.SetInformationJobObject(
                self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.TerminateJobObject(self.handle, exit_code):
                raise ctypes.WinError(ctypes.get_last_error())
            while True:
                handles = list(held.values())
                for start in range(0, len(handles), 64):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TreeNotStopped("spawn.tree_not_stopped: Job deadline")
                    batch = handles[start : start + 64]
                    result = self.kernel.WaitForMultipleObjects(
                        len(batch),
                        (wintypes.HANDLE * len(batch))(*batch),
                        True,
                        max(1, int(remaining * 1000)),
                    )
                    if result != 0:
                        raise TreeNotStopped(f"spawn.tree_not_stopped: Job wait {result}")
                # The Job's active count proves the stop. A member that joined after the first
                # capture and already exited (or one the active-process limit refused) is counted
                # in TotalProcesses but never captured, so only an active member is waited for.
                after = accounting()
                if not after.ActiveProcesses:
                    break
                if not capture(after.ActiveProcesses) and time.monotonic() >= deadline:
                    raise TreeNotStopped("spawn.tree_not_stopped: uncaptured Job members")
        finally:
            for handle in held.values():
                _winapi.CloseHandle(handle)

    def finish_memory(self) -> None:
        """Drain the memory verdict before releasing the completion port."""
        if self._memory_thread is not None:
            self.kernel.PostQueuedCompletionStatus(self._memory_port, 0, 0, None)
            self._memory_thread.join(timeout=REAP_TIMEOUT + 1.0)
            if self._memory_thread.is_alive():
                raise TreeNotStopped("spawn.tree_not_stopped: Job memory watch unconfirmed")
            self._memory_thread = None
        if self._memory_port is not None:
            _winapi.CloseHandle(self._memory_port)
            self._memory_port = None
        if self.memory is not None:
            self.memory.observe(self._job_peak())
            self.memory.finish()
        if self._memory_failure is not None:
            raise self._memory_failure

    def close(self) -> None:
        """Stop the Job before releasing its handle."""
        if not self.handle:
            return
        try:
            self.finish_memory()
            self.terminate()
        finally:
            if self._memory_port is not None:
                _winapi.CloseHandle(self._memory_port)
            _winapi.CloseHandle(self.handle)
            self.handle = None
