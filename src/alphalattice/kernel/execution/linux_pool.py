"""Keep fallback Linux pool workers under a dedicated child subreaper."""

from __future__ import annotations

import multiprocessing
import os
import select
import signal
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, suppress
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

from .linux_scope import enable_child_subreaper

_STOP_SECONDS = 5.0
_ADMISSION_SECONDS = 30.0


class _ProcessContext(Protocol):
    Process: type[BaseProcess]


def _drain_adopted() -> None:
    """Stop and reap the guardian's children, including newly adopted descendants."""
    native = vars(os)
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    deadline = time.monotonic() + _STOP_SECONDS
    while time.monotonic() < deadline:
        try:
            ended, _ = native["waitpid"](-1, native["WNOHANG"])
        except ChildProcessError:
            return
        if ended:
            continue
        # Only this guardian reaps these PIDs. They cannot be reused between
        # reading its direct children and stopping them.
        for word in children.read_text(encoding="ascii").split():
            with suppress(ProcessLookupError):
                os.kill(int(word), vars(signal)["SIGKILL"])
        select.select([], [], [], min(0.1, max(0.0, deadline - time.monotonic())))
    raise RuntimeError("spawn.tree_not_stopped: Linux adopted pool descendants still live")


def _guard_worker(
    method: str,
    target: Callable[..., None],
    arguments: tuple[Any, ...],
    control: Connection,
    parent_channel: Connection,
) -> None:
    parent_channel.close()
    with control:
        enable_child_subreaper()
        # A forkserver would parent the job outside this subreaper. Its launch
        # arguments can travel through spawn, which also preserves spawn pools.
        context = multiprocessing.get_context("fork" if method == "fork" else "spawn")
        worker = cast(_ProcessContext, context).Process(target=target, args=arguments)
        worker.start()
        assert worker.pid is not None
        with ExitStack() as opened:
            try:
                native = vars(os)
                departure, wake = native["pipe2"](native["O_NONBLOCK"] | native["O_CLOEXEC"])
                opened.callback(os.close, departure)
                opened.callback(os.close, wake)
                child_exit = vars(signal)["SIGCHLD"]
                previous = signal.signal(child_exit, lambda *_: None)
                opened.callback(signal.signal, child_exit, previous)
                opened.callback(signal.set_wakeup_fd, signal.set_wakeup_fd(wake))
            except (OSError, ValueError):
                _drain_adopted()
                raise
            try:
                control.send_bytes(str(worker.pid).encode("ascii"))
                # SIGCHLD wakes the guardian even when a descendant holds the
                # multiprocessing sentinel open. Only the worker's own exit counts.
                native_select = vars(select)
                events = native_select["poll"]()
                controller = control.fileno()
                events.register(controller, native_select["POLLIN"])
                events.register(departure, native_select["POLLIN"])
                while worker.exitcode is None:
                    readable = dict(events.poll())
                    if controller in readable:
                        with suppress(EOFError):
                            control.recv_bytes()
                        break
                    if departure in readable:
                        os.read(departure, 4096)
            finally:
                if worker.exitcode is None:
                    with suppress(ProcessLookupError):
                        worker.kill()
                    deadline = time.monotonic() + _STOP_SECONDS
                    while worker.exitcode is None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not wait([departure], timeout=remaining):
                            raise RuntimeError(
                                "spawn.tree_not_stopped: Linux pool worker still live"
                            )
                        os.read(departure, 4096)
                worker.join(timeout=_STOP_SECONDS)
                status = worker.exitcode
                assert status is not None
                _drain_adopted()
                with suppress(BrokenPipeError, ConnectionResetError):
                    control.send_bytes(b"settled")
    raise SystemExit(status if status >= 0 else 128 - status)


class GuardedWorker:
    """Expose the real worker PID and keep the guardian's join handle."""

    def __init__(self, launcher: Any, control: Connection) -> None:
        """Read the worker's identity from its guardian before admitting it."""
        self._launcher = launcher
        self._control = control
        self._lock = threading.Lock()
        self._requested = False
        self._settled = False
        with suppress(EOFError):
            if control.poll(_ADMISSION_SECONDS):
                word = control.recv_bytes()
                if word.isdigit() and int(word) > 0:
                    self.pid = int(word)
                    return
        raise OSError("spawn.worker_tree_admission_unproved")

    def __getattr__(self, name: str) -> Any:
        """Let the executor join and observe the guardian through the worker handle."""
        return getattr(self._launcher, name)

    def _request(self) -> None:
        if not self._requested and not self._control.poll():
            with suppress(BrokenPipeError, ConnectionResetError):
                self._control.send_bytes(b"stop")
        self._requested = True

    def terminate(self) -> None:
        """Let the executor retire a broken worker without killing its guardian."""
        with self._lock:
            self._request()

    def stop(self) -> None:
        """Require the guardian's acknowledgment before the Task can finish stopping."""
        with self._lock:
            if not self._settled:
                self._request()
                with suppress(EOFError):
                    self._settled = self._control.poll(2 * _STOP_SECONDS + 1) and (
                        self._control.recv_bytes() == b"settled"
                    )
            if not self._settled:
                raise RuntimeError("spawn.tree_not_stopped: Linux pool guardian did not settle")

    def close_control(self) -> None:
        """Release the controller channel after that pool has joined."""
        self._control.close()


def start_guarded_worker(
    pool: Any, launch: Callable[[], None], initial_arguments: tuple[Any, ...]
) -> GuardedWorker:
    """Use the executor's native launcher with a guardian around its ordinary target."""
    context = pool._mp_context
    owner, control = context.Pipe()
    previous = set(pool._processes)
    launcher = None

    def process(*, target: Callable[..., None], args: tuple[Any, ...]) -> Any:
        return context.Process(
            target=_guard_worker,
            args=(context.get_start_method(), target, args, control, owner),
        )

    try:
        with control:
            pool._mp_context = SimpleNamespace(Process=process)
            pool._initargs = (*initial_arguments, None, os.getpid())
            try:
                launch()
            finally:
                pool._mp_context = context
                pool._initargs = initial_arguments
        launcher_pid = next(pid for pid in pool._processes if pid not in previous)
        launcher = pool._processes.pop(launcher_pid)
        handle = GuardedWorker(launcher, owner)
        pool._processes[handle.pid] = handle
        return handle
    except BaseException:
        owner.close()
        if launcher is not None:
            launcher.join(timeout=2 * _STOP_SECONDS + 1)
            if launcher.is_alive():
                raise RuntimeError(
                    "spawn.tree_not_stopped: Linux pool guardian still live"
                ) from None
        raise
