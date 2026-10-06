"""Product-written research prerequisites, once per pytest session across workers.

Only the builder sees the seed. Tests receive a private database and record copy;
read-only Portfolio consumers may link the seed's immutable Parquet objects.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest


@contextmanager
def _build_lock(path: Path) -> Iterator[None]:
    """Wait for the builder's OS lock, without polling or a fixed sleep."""
    if os.name != "nt":
        import fcntl

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
        status = kernel.WaitForSingleObject(handle, 900_000)
        if status not in (0, 0x80):  # acquired, or released by a dead builder
            raise TimeoutError(f"session workspace builder did not release {path.name}: {status}")
        try:
            yield
        finally:
            kernel.ReleaseMutex(handle)
    finally:
        kernel.CloseHandle(handle)


def _files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        with path.open("rb") as stream:
            result[path.relative_to(root).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return result


def session_workspace(
    factory: pytest.TempPathFactory,
    name: str,
    build: Callable[[Path], dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    """Publish one complete seed; a failed or changed seed is refused, never repaired."""
    if not name.isidentifier():
        raise ValueError("session workspace name must be an identifier")
    base = factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    cache = base / "research-prerequisites"
    cache.mkdir(exist_ok=True)
    target = cache / name
    record = target / "complete.json"
    started = time.perf_counter()
    with _build_lock(cache / f"{name}.lock"):
        built = not target.exists()
        if built:
            target.mkdir()
            metadata = build(target / "workspace")
            record.write_text(
                json.dumps({"metadata": metadata, "files": _files(target / "workspace")}) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        if not record.is_file():
            raise RuntimeError(f"incomplete session workspace: {name}")
        result = json.loads(record.read_text(encoding="utf-8"))
        if _files(target / "workspace") != result["files"]:
            raise RuntimeError(f"session workspace was changed: {name}")
    print(
        f"PLAYPEN_SESSION_WORKSPACE {'BUILD' if built else 'REUSE'} {name} "
        f"{time.perf_counter() - started:.3f}s",
        flush=True,
    )
    return target / "workspace", result["metadata"]


def copy_workspace(source: Path, target: Path, *, link_parquet: bool = False) -> Path:
    """Private writable files; only callers that never edit Parquet opt into links.

    The session seed remains alive for the entire run, including every linked
    consumer. Databases, records and model files always get independent copies.
    """

    def copy_file(src: str, dst: str) -> str:
        if link_parquet and Path(src).suffix == ".parquet":
            os.link(src, dst)
            return dst
        return shutil.copy2(src, dst)

    shutil.copytree(source, target, copy_function=copy_file)
    return target
