"""Start external processes with bounded, quiet, isolated execution.

The product's compute worker pool in ``task_control.child`` owns its own workers.
``scripts/create_gpu_environment.py`` remains outside this owner under the no-GPU fence.
The ambient Job probe adds no Job, so its measurement sees the caller's Job.
UI QA scene services use ready-service breakaway and are stopped by ``scenes.py down``.
"""

from __future__ import annotations

import codecs
import ctypes
import math
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, closing, suppress
from ctypes import wintypes
from typing import Any, Literal, overload

import psutil

if sys.platform == "win32":
    import _winapi

DEFAULT_TIMEOUT = 300.0
REAP_TIMEOUT = 5.0


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
    args: list[str], *, env: dict[str, str], job_factory: Any, timeout: float = DEFAULT_TIMEOUT
) -> int:
    """Run an interactive child with its caller's streams and owned lifetime."""
    args = _command(args)
    if sys.platform != "win32":
        return run(args, env=env, stdin=sys.stdin, timeout=timeout).returncode
    with closing(job_factory()) as job, ExitStack() as streams:
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
            0x4,
            {**env, "GIT_TERMINAL_PROMPT": "0"},
            None,
            startup,
        )
        try:
            job.assign_and_resume(process, thread)
            if _winapi.WaitForSingleObject(process, int(timeout * 1000)) == _winapi.WAIT_TIMEOUT:
                raise subprocess.TimeoutExpired(args, timeout)
            return int(_winapi.GetExitCodeProcess(process))
        except BaseException:
            stop_tree(pid)
            raise
        finally:
            _winapi.CloseHandle(thread)
            _winapi.CloseHandle(process)


def ambient_job_probe(args: list[str], *, creationflags: int) -> subprocess.CompletedProcess[str]:
    """Run the leaf Job probe hidden and bounded without changing its Job membership."""
    env = os.environ.copy()
    env.pop("ALPHALATTICE_DATA_ROOT", None)
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


def detached(args: list[str], *, log: Any, env: dict[str, str]) -> subprocess.Popen[bytes]:
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
    child = subprocess.Popen(args, creationflags=flags, **common)
    native: Any = child
    proved = kernel32.IsProcessInJob(
        ctypes.c_void_p(int(native._handle)), None, ctypes.byref(in_job)
    )
    error_code = ctypes.get_last_error() if not proved else 0
    if not proved or in_job.value:
        stop_tree(child.pid)
        child.wait(timeout=REAP_TIMEOUT)
        error = PermissionError("local_web.detached_breakaway_denied")
        error.winerror = error_code
        raise error
    return child


def stop_tree(pid: int) -> None:
    """Stop a process and its descendants, children first."""
    try:
        parent = psutil.Process(pid)
        members = [*parent.children(recursive=True), parent]
    except psutil.NoSuchProcess:
        return
    for member in members:
        with suppress(psutil.NoSuchProcess):
            member.kill()
    _, alive = psutil.wait_procs(members, timeout=REAP_TIMEOUT)
    if alive:
        raise TreeNotStopped(f"spawn.tree_not_stopped: {[p.pid for p in alive]}")


def _in_group(member: Any, pid: int) -> bool:
    try:
        return bool(vars(os)["getpgid"](member.pid) == pid)
    except ProcessLookupError:
        return False


def _running(member: Any) -> bool:
    try:
        return bool(member.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def _stop_group(pid: int) -> None:
    """Kill an owned POSIX group and wait for its surviving descendants to exit."""
    members = [
        member for member in psutil.process_iter() if member.pid != pid and _in_group(member, pid)
    ]  # Capture before signalling: a dying member can leave the group before it exits.
    try:
        vars(os)["killpg"](pid, vars(signal)["SIGKILL"])
    except ProcessLookupError:
        pass
    else:
        members.extend(
            member
            for member in psutil.process_iter()
            if member.pid != pid and member not in members and _in_group(member, pid)
        )  # Include members admitted between capture and the group signal.
    deadline = time.monotonic() + REAP_TIMEOUT
    while members:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TreeNotStopped(
                f"spawn.tree_not_stopped: group members {[p.pid for p in members]}"
            )
        _, alive = psutil.wait_procs(members, timeout=min(remaining, 0.1))
        members = []
        members.extend(member for member in alive if _running(member))


@overload
def run(args: Any, *, text: Literal[True], **kwargs: Any) -> subprocess.CompletedProcess[str]: ...


@overload
def run(args: Any, *, encoding: str, **kwargs: Any) -> subprocess.CompletedProcess[str]: ...


@overload
def run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]: ...


def run(
    args: Any,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    input: Any = None,
    capture_output: bool = False,
    check: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run one child until it exits or its finite deadline ends its tree."""
    args = _command(args)
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("spawn.timeout_invalid: provide a finite positive timeout")
    supplied = kwargs.pop("env", None)
    environment = dict(os.environ if supplied is None else supplied)
    if supplied is None:  # only an explicitly supplied environment may carry the person's data root
        environment.pop("ALPHALATTICE_DATA_ROOT", None)
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
    job = ChildJob() if sys.platform == "win32" else None
    if job is not None:
        creationflags |= 0x4  # CREATE_SUSPENDED: admit before any child code runs
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(args, creationflags=creationflags, **kwargs)
    except BaseException:
        if job is not None:
            job.close()
        raise

    group_stopped = False

    def end_owned_tree() -> None:
        nonlocal group_stopped
        if job is not None:
            job.close()  # the Job still owns descendants after the direct parent exits
        elif not group_stopped:
            group_stopped = True
            _stop_group(process.pid)

    try:
        if job is not None:
            native: Any = process
            job.assign(int(native._handle))
            job.resume_pid(process.pid)
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except BaseException as error:
        try:
            end_owned_tree()
        finally:
            if process.poll() is None:
                process.kill()
            stdout, stderr = process.communicate(timeout=REAP_TIMEOUT)
        if isinstance(error, subprocess.TimeoutExpired):
            error.output, error.stderr = stdout, stderr
        raise
    finally:
        end_owned_tree()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


class InteractiveSession:
    """Own a bounded multi-turn child; its streams and stop scope are explicit."""

    def __init__(
        self,
        *,
        timeout: float,
        stop_scope: str,
        affinity: int | None = None,
        priority: int | None = None,
        exit_code: int = 1,
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
        self.timeout, self.stop_scope = timeout, stop_scope
        self.affinity, self.priority, self.exit_code = affinity, priority, exit_code
        self.child: subprocess.Popen[Any] | None = None
        self.job: ChildJob | None = None
        self.stopped = False
        self.timed_out = False
        self.failure: BaseException | None = None
        self.timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._generation = 0

    def start(self, args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        """Start once, admitting the suspended child to its Job before execution."""
        if self.stopped or self.child is not None:
            raise RuntimeError("spawn.session_terminal")
        stdin = kwargs.pop("stdin", subprocess.PIPE)
        if stdin not in (subprocess.PIPE, subprocess.DEVNULL):
            raise ValueError("spawn.session_owns_stdin")
        supplied = kwargs.pop("env", None)
        env = dict(os.environ if supplied is None else supplied)
        if supplied is None:
            env.pop("ALPHALATTICE_DATA_ROOT", None)
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
                    self.job = ChildJob(affinity=self.affinity, priority=self.priority)
                creationflags |= 0x4
            else:
                kwargs["start_new_session"] = True
        try:
            self.child = subprocess.Popen(_command(args), creationflags=creationflags, **kwargs)
            if self.job is not None:
                native: Any = self.child
                self.job.assign(int(native._handle))
                self.job.resume_pid(self.child.pid)
        except BaseException:
            if self.job is not None:
                ChildJob.close(self.job)
            if self.child is not None and self.child.poll() is None:
                self.child.kill()
                self.child.wait(timeout=REAP_TIMEOUT)
            raise
        self._arm()
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
                elif self.stop_scope == "tree":
                    _stop_group(child.pid)
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

    def __enter__(self) -> InteractiveSession:
        """Keep the session within its caller's cleanup scope."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Stop the child and release its Job even when the caller fails."""
        try:
            self.stop()
        finally:
            if self.job is not None:
                ChildJob.close(self.job)


def run_root(
    args: Any, *, timeout: float, check: bool = False, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Bound a normal-Host launcher while leaving its ready Host outside its root scope."""
    with InteractiveSession(timeout=timeout, stop_scope="root") as session:
        child = session.start(args, **kwargs)
        child.wait(timeout=timeout)
        session.check()
        result: subprocess.CompletedProcess[Any] = subprocess.CompletedProcess(
            args, child.returncode
        )
        if check:
            result.check_returncode()
        return result


@overload
def check_output(args: Any, *, text: Literal[True], **kwargs: Any) -> str: ...


@overload
def check_output(args: Any, **kwargs: Any) -> bytes: ...


def check_output(args: Any, **kwargs: Any) -> Any:
    """Return a bounded child's output, refusing its nonzero exit."""
    return run(args, stdout=subprocess.PIPE, check=True, **kwargs).stdout


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

    def __init__(self, *, affinity: int | None = None, priority: int | None = None) -> None:
        """Create this parent's Job with its caller's optional affinity and priority."""
        if sys.platform != "win32":
            raise OSError("Windows Jobs are unavailable")
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
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise OSError("owned Job limit unavailable")

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

    def resume_pid(self, pid: int) -> None:
        """Resume the sole initial thread of an already assigned suspended child."""
        threads = psutil.Process(pid).threads()
        if len(threads) != 1:
            raise OSError("spawn.initial_thread_unavailable")
        self.kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenThread.restype = wintypes.HANDLE
        thread = self.kernel.OpenThread(0x0002, False, threads[0].id)  # THREAD_SUSPEND_RESUME
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _winapi.CloseHandle(thread)

    def terminate(self, *, exit_code: int = 1) -> None:
        """Stop this Job and prove each captured member exited, keeping the Job readable."""
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
            total = before.TotalProcesses
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
                after = accounting()
                if after.TotalProcesses == total:
                    break
                new = capture(after.ActiveProcesses)
                if len(new) < after.TotalProcesses - total:
                    raise TreeNotStopped("spawn.tree_not_stopped: uncaptured Job members")
                total = after.TotalProcesses
                if time.monotonic() >= deadline:
                    raise TreeNotStopped("spawn.tree_not_stopped: Job deadline")
        finally:
            for handle in held.values():
                _winapi.CloseHandle(handle)

    def close(self) -> None:
        """Stop the Job before releasing its handle."""
        if not self.handle:
            return
        try:
            self.terminate()
        finally:
            _winapi.CloseHandle(self.handle)
            self.handle = None
