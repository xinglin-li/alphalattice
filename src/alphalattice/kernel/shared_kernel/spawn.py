"""Bind kernel child execution to the product's environment and process observations."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Literal, overload

import psutil

from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.execution import child_process

DEFAULT_TIMEOUT = child_process.DEFAULT_TIMEOUT
REAP_TIMEOUT = child_process.REAP_TIMEOUT
TreeNotStopped = child_process.TreeNotStopped


def _default_environment() -> dict[str, str]:
    env = process_config().child_environment()
    # Only an explicitly supplied environment may carry the person's data root.
    env.pop("ALPHALATTICE_DATA_ROOT", None)
    return env


def attached(
    args: list[str], *, env: dict[str, str], job_factory: Any, timeout: float = DEFAULT_TIMEOUT
) -> int:
    """Run an interactive child with its caller's streams and owned lifetime."""
    return child_process.attached(
        args,
        env=env,
        job_factory=job_factory,
        processes=psutil,
        default_environment=_default_environment,
        timeout=timeout,
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
    """Create a kernel Job backed by the product's process observations."""

    def __init__(self, *, affinity: int | None = None, priority: int | None = None) -> None:
        """Create this parent's Job with its caller's optional affinity and priority."""
        super().__init__(processes=psutil, affinity=affinity, priority=priority)
