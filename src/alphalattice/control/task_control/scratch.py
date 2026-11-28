"""Task Control owns the temporary call files of one workspace execution."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from threading import Lock
from uuid import UUID

_CURRENT: ContextVar[Path | None] = ContextVar("task_scratch", default=None)
_OWNER = "owner.json"
_OPEN_SCOPES: set[Path] = set()
_SCOPE_LOCK = Lock()


def task_scratch_directory() -> Path | None:
    """Return the current Task's admitted directory for its temporary child calls."""
    return _CURRENT.get()


def carry_task_scratch[T](call: Callable[[], T]) -> Callable[[], T]:
    """Carry only this Task's directory to one of its parallel work items."""
    folder = _CURRENT.get()

    def run() -> T:
        held = _CURRENT.set(folder)
        try:
            return call()
        finally:
            _CURRENT.reset(held)

    return run


def _root(workspace: Path) -> Path:
    return workspace / "staging" / "task-scratch"


def _require_real(path: Path) -> None:
    if path.is_symlink() or path.is_junction() or path.resolve(strict=True) != path:
        raise ValueError(f"task_scratch.path_redirected: {path}")


def _remove_member(path: Path, root: Path) -> None:
    _require_real(path.parent)
    if not path.is_relative_to(root):
        raise ValueError("Task scratch path is outside its owner.")
    if path.is_symlink():
        path.unlink()
    elif path.is_junction():
        path.rmdir()
    else:
        _require_real(path)
        if path.is_dir():
            for child in path.iterdir():
                _remove_member(child, root)
            path.rmdir()
        else:
            path.unlink()


def _remove_owned(path: Path) -> None:
    _require_real(path)
    for child in path.iterdir():
        if child.name != _OWNER:
            _remove_member(child, path)
    # Retain proof if any payload member could not be removed.
    _remove_member(path / _OWNER, path)
    path.rmdir()


@contextmanager
def _open_scope(folder: Path) -> Iterator[None]:
    with _SCOPE_LOCK:
        if folder in _OPEN_SCOPES:
            raise ValueError(f"task_scratch.scope_open: {folder}")
        _OPEN_SCOPES.add(folder)
    try:
        yield
    finally:
        with _SCOPE_LOCK:
            _OPEN_SCOPES.remove(folder)


def _reclaim_for_rerun(folder: Path, lease: tuple[str, int, float]) -> None:
    try:
        recorded = _proved_lease(folder)
    except (OSError, ValueError, OverflowError) as error:
        raise ValueError(f"task_scratch.owner_unproved: {folder}") from error
    if recorded != lease:
        raise ValueError(f"task_scratch.owner_mismatch: {folder}")
    _remove_owned(folder)


@contextmanager
def own_task_scratch(workspace: Path, task_id: UUID, epoch: str) -> Iterator[None]:
    """Bind child scratch only after publishing the writer epoch and exact directory identity."""
    import psutil  # lazy: a Task records its current process before using scratch

    root = _root(workspace)
    root.parent.mkdir(exist_ok=True)
    _require_real(root.parent)
    root.mkdir(exist_ok=True)
    _require_real(root)
    folder = root / str(task_id)
    with _open_scope(folder):
        process = psutil.Process()
        created = process.create_time()
        try:
            folder.mkdir()
        except FileExistsError:
            _reclaim_for_rerun(folder, (epoch, process.pid, created))
            folder.mkdir()
        _require_real(folder)
        info = folder.stat()
        with (folder / _OWNER).open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "task_id": folder.name,
                    "host_epoch": epoch,
                    "pid": process.pid,
                    "created": created,
                    "device": info.st_dev,
                    "inode": info.st_ino,
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        held = _CURRENT.set(folder)
        try:
            yield
        finally:
            _CURRENT.reset(held)
            with suppress(OSError, ValueError, OverflowError):
                if _proved_lease(folder)[0] == epoch:
                    _remove_owned(folder)


def _proved_lease(path: Path) -> tuple[str, int, float]:
    _require_real(path)
    owner = path / _OWNER
    _require_real(owner)
    if not owner.is_file() or owner.stat().st_nlink != 1:
        raise ValueError("Task scratch owner record is unproved.")
    value = json.loads(owner.read_text(encoding="utf-8"))
    info = path.stat()
    if (
        not isinstance(value, dict)
        or not path.is_dir()
        or str(UUID(path.name)) != path.name
        or value.get("task_id") != path.name
        or value.get("device") != info.st_dev
        or value.get("inode") != info.st_ino
    ):
        raise ValueError("Task scratch does not match its owner record.")
    epoch, pid, created = value.get("host_epoch"), value.get("pid"), value.get("created")
    if (
        not isinstance(epoch, str)
        or str(UUID(epoch)) != epoch
        or type(pid) is not int
        or pid <= 0
        or type(created) not in (int, float)
        or not math.isfinite(created)
        or created <= 0
    ):
        raise ValueError("Task scratch does not name a valid process lease.")
    return epoch, pid, created


def _ended(pid: int, created: float) -> bool:
    import psutil  # lazy: only a lease in the current epoch needs a process read

    try:
        process = psutil.Process(pid)
        return process.create_time() != created or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True
    except (psutil.AccessDenied, OSError) as error:
        raise ValueError("Task scratch process identity is unproved.") from error


def reclaim_task_scratch(workspace: Path, epoch: str) -> dict[str, object]:
    """Under the acquired workspace writer, only reap proved ended Task directories."""
    removed: list[str] = []
    unproved: dict[str, str] = {}
    root = _root(workspace)
    result: dict[str, object] = {"removed": removed, "unproved": unproved}
    try:
        _require_real(root)
        paths = sorted(root.iterdir())
    except FileNotFoundError:
        return result
    except (OSError, ValueError):
        unproved[root.name] = "task_scratch.root_unproved"
        return result
    for path in paths:
        try:
            holder, pid, created = _proved_lease(path)
            if holder != epoch or _ended(pid, created):
                _remove_owned(path)
                removed.append(path.name)
        except (OSError, ValueError, OverflowError):
            unproved[path.name] = "task_scratch.owner_or_removal_unproved"
    return result
