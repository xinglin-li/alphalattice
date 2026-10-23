"""Invocation-scoped golden ownership; domain builders and copy validators stay with callers."""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldenRun:
    """The canonical invocation root shared by its workers, never a persistent cache."""

    root: Path


_RUN: ContextVar[GoldenRun | None] = ContextVar("golden_run", default=None)


@contextmanager
def golden_run(root: Path) -> Iterator[Path]:
    """Bind acquisition and project discovery to one controller-provided invocation root."""
    run = GoldenRun(root.parent if os.environ.get("PYTEST_XDIST_WORKER") else root)
    token = _RUN.set(run)
    try:
        yield run.root
    finally:
        _RUN.reset(token)


def golden_root(fallback: Path) -> Path:
    """Select the invocation's golden directory, retaining explicit non-pytest callers."""
    run = _RUN.get()
    return run.root / "golden-workspaces" if run is not None else fallback


@contextmanager
def build_lock(path: Path) -> Iterator[None]:
    """Wait for OS build ownership; the caller's existing producer budget bounds the work."""
    if sys.platform != "win32":
        import fcntl  # lazy: POSIX-only lock backend, loaded where the OS lock is taken.

        with path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
        return

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    name = (
        "Local\\AlphaLatticeTestWorkspace-"
        + hashlib.sha256(str(path.resolve()).casefold().encode()).hexdigest()
    )
    handle = kernel.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        status = kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
        if status not in (0, 0x80):  # acquired, or released by a dead builder
            raise OSError(f"session workspace build lock failed: {path.name}: {status}")
        try:
            yield
        finally:
            kernel.ReleaseMutex(handle)
    finally:
        kernel.CloseHandle(handle)
