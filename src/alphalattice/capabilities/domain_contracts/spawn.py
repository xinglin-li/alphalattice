"""Bind kernel child execution to the product's environment and process observations."""

from __future__ import annotations

import multiprocessing
import multiprocessing.util
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import AbstractContextManager
from multiprocessing.connection import Connection
from threading import Lock
from typing import Any, Literal, overload

import psutil

from alphalattice.capabilities.domain_contracts.runtime_resources import runtime_json
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.execution import (
    ExecutionStopped,
    child_process,
    install_process_pool_owner,
    interrupt_on_execution_stop,
    stop_reason,
)

DEFAULT_TIMEOUT = child_process.DEFAULT_TIMEOUT
REAP_TIMEOUT = child_process.REAP_TIMEOUT
TreeNotStopped = child_process.TreeNotStopped
TreeMemoryLimit = child_process.TreeMemoryLimit
_worker_host: int | None = None


def tree_memory(kind: str) -> child_process.TreeMemory:
    """Bind a registered process kind's tree budget and the operator's current mode."""
    rows = runtime_json("config/registries/process-budgets.json")["kinds"]
    if kind not in rows:
        raise ValueError(f"spawn.process_kind_unknown: {kind}")
    budget = rows[kind].get("tree_memory_bytes")
    if budget == "undeclared":
        budget = None
    elif type(budget) is not int or budget <= 0:
        raise ValueError(f"spawn.tree_memory_budget_invalid: {kind}")
    return child_process.TreeMemory(kind, budget, process_config().settings.process_memory_mode)


def current_tree(kind: str) -> AbstractContextManager[None]:
    """Keep the installed process in its own registered tree for its real lifetime."""
    return child_process.current_tree(psutil, tree_memory(kind))


def _start_pool_worker(
    kind: str,
    initialize: Callable[..., None] | None,
    arguments: tuple[Any, ...],
    release: Connection | None = None,
    host_pid: int | None = None,
) -> None:
    global _worker_host
    _worker_host = host_pid
    if release is not None:
        with release:
            if not release.poll(REAP_TIMEOUT) or release.recv_bytes() != b"1":
                raise OSError("spawn.worker_tree_admission_unproved")
    elif sys.platform == "win32" and multiprocessing.parent_process() is not None:
        owner = current_tree(kind)
        owner.__enter__()
        multiprocessing.util.Finalize(
            None,
            owner.__exit__,
            args=(None, None, None),
            exitpriority=10,
        )
    if initialize is not None:
        initialize(*arguments)


def worker_host_pid() -> int | None:
    """Keep a pool worker's Host identity across its private Linux guardian."""
    parent = multiprocessing.parent_process()
    return _worker_host if _worker_host is not None else (parent.pid if parent else None)


def _own_process_pool(pool: ProcessPoolExecutor, name: str) -> None:
    kind = {"task-child": "kept_worker", "causal-derivation": "causal_derivation_worker"}.get(name)
    if kind is None:
        return
    native: Any = pool
    native._initargs = (kind, native._initializer, native._initargs)
    native._initializer = _start_pool_worker
    if sys.platform != "win32":
        child_process.own_pool_trees(pool, psutil, tree_memory(kind))
    admission = Lock()
    submit, shutdown = pool.submit, pool.shutdown

    def stop() -> None:
        with admission:
            stop_pool(pool)

    release = interrupt_on_execution_stop(stop) if name == "causal-derivation" else None

    def submit_owned[T](call: Callable[..., T], /, *args: Any, **kwargs: Any) -> Future[T]:
        with admission:
            if (reason := stop_reason()) is not None:
                raise ExecutionStopped(reason)
            return submit(call, *args, **kwargs)

    def shutdown_owned(*, wait: bool = True, cancel_futures: bool = False) -> None:
        try:
            shutdown(wait=wait or sys.platform != "win32", cancel_futures=cancel_futures)
        finally:
            if release is not None:
                release()
        child_process.close_pool_trees(pool)

    if name == "causal-derivation":
        native.submit = submit_owned
    native.shutdown = shutdown_owned


install_process_pool_owner(_own_process_pool)


def _default_environment() -> dict[str, str]:
    env = process_config().child_environment()
    # Only an explicitly supplied environment may carry the person's data root.
    env.pop("ALPHALATTICE_DATA_ROOT", None)
    return env


def attached(
    args: list[str],
    *,
    env: dict[str, str],
    job_factory: Any,
    timeout: float = DEFAULT_TIMEOUT,
    process_kind: str | None = None,
) -> int:
    """Run an interactive child with its caller's streams and owned lifetime."""
    return child_process.attached(
        args,
        env=env,
        job_factory=job_factory,
        processes=psutil,
        default_environment=_default_environment,
        timeout=timeout,
        memory=None if process_kind is None else tree_memory(process_kind),
    )


def ambient_job_probe(args: list[str], *, creationflags: int) -> subprocess.CompletedProcess[str]:
    """Run the leaf Job probe hidden and bounded without changing its Job membership."""
    return child_process.ambient_job_probe(
        args, creationflags=creationflags, env=_default_environment()
    )


def detached(args: list[str], *, log: Any, env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start a workspace-owned Host outside the launching turn's process group."""
    return child_process.detached(args, log=log, env=env, processes=psutil)


def stop_tree(pid: int) -> None:
    """Stop a process and its descendants, children first."""
    child_process.stop_tree(pid, processes=psutil)


def stop_pool(pool: Any) -> None:
    """Stop the trees of a process pool whose caller owns its shutdown."""
    child_process.stop_pool(pool, processes=psutil)


def reap_detached_sessions() -> AbstractContextManager[Callable[[int], None]]:
    """Reap Linux descendants owned by this dedicated launcher."""
    return child_process.reap_detached_sessions(processes=psutil)


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
    process_kind: str | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run one child until it exits or its finite deadline ends its tree."""
    return child_process.run(
        psutil,
        _default_environment,
        args,
        timeout=timeout,
        input=input,
        capture_output=capture_output,
        check=check,
        memory=None if process_kind is None else tree_memory(process_kind),
        **kwargs,
    )


class InteractiveSession(child_process.InteractiveSession):
    """Own a bounded multi-turn child with the product's process observations."""

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
        super().__init__(
            processes=psutil,
            default_environment=_default_environment,
            timeout=timeout,
            stop_scope=stop_scope,
            affinity=affinity,
            priority=priority,
            exit_code=exit_code,
        )


def run_root(
    args: Any, *, timeout: float, check: bool = False, **kwargs: Any
) -> subprocess.CompletedProcess[Any]:
    """Bound a normal-Host launcher while leaving its ready Host outside its root scope."""
    return child_process.run_root(
        psutil, _default_environment, args, timeout=timeout, check=check, **kwargs
    )


@overload
def check_output(args: Any, *, text: Literal[True], **kwargs: Any) -> str: ...


@overload
def check_output(args: Any, **kwargs: Any) -> bytes: ...


def check_output(args: Any, **kwargs: Any) -> Any:
    """Return a bounded child's output, refusing its nonzero exit."""
    return run(args, stdout=subprocess.PIPE, check=True, **kwargs).stdout


class ChildJob(child_process.ChildJob):
    """Create a kernel Job for the product's process owner."""

    def __init__(
        self,
        *,
        affinity: int | None = None,
        priority: int | None = None,
        memory: child_process.TreeMemory | None = None,
    ) -> None:
        """Create this parent's Job with its caller's optional affinity and priority."""
        super().__init__(affinity=affinity, priority=priority, memory=memory)
