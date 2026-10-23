"""Invocation-scoped golden ownership and immutable payloads; domain builders stay with callers."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import stat
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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


def _manifest(
    root: Path,
    parts: tuple[str, ...],
    *,
    filtered: bool = True,
    read: Callable[[Path], str] | None = None,
) -> dict[str, str | None]:
    contents: dict[str, str | None] = {}
    pending = [(root / name, name) for name in parts]
    while pending:
        path, name = pending.pop()
        if not filtered and name == "./.alphalattice-writer.lock":
            continue  # OS ownership is not a restorable payload file.
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            contents[name] = None
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise RuntimeError(f"golden input link is not admitted: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            contents[name + "/"] = ""
            with os.scandir(path) as entries:
                pending.extend(
                    (Path(entry.path), name + "/" + entry.name)
                    for entry in entries
                    if not filtered
                    or (
                        entry.name not in {"__pycache__", ".scratch"}
                        and not entry.name.endswith(".egg-info")
                        and not entry.name.startswith(".venv")
                    )
                )
        else:
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"golden input is not a regular file: {path}")
            if read:
                contents[name] = read(path)
            else:
                with path.open("rb") as stream:
                    contents[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return contents


def _long(path: Path) -> Path:
    path = path.absolute()
    for ancestor in (path, *path.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise RuntimeError(f"golden payload path is link-like: {ancestor}")
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\" + ("UNC\\" + value[2:] if value.startswith("\\\\") else value)
    return Path(value)


def seal_payload(
    workspace: Path, *, closed: Callable[[Path], AbstractContextManager[None]]
) -> None:
    """Seal only a closed, regular-file payload; immutable bytes never include outward links."""
    workspace = _long(workspace)
    try:
        with closed(workspace):
            contents = _manifest(workspace, (".",), filtered=False)
            files = [
                (workspace / name, (workspace / name).stat())
                for name in contents
                if not name.endswith("/")
            ]
            counts = Counter((info.st_dev, info.st_ino) for _, info in files)
            if any(info.st_nlink > counts[info.st_dev, info.st_ino] for _, info in files):
                raise RuntimeError("payload has a hard link outside its root")
            for path, info in files:
                path.chmod(info.st_mode & ~stat.S_IWRITE)
            (workspace.parent / "payload.json").write_text(
                json.dumps(contents, sort_keys=True), encoding="utf-8"
            )
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"golden payload cannot be sealed; close its writer and use a fresh run: {workspace}"
        ) from error


def copy_payload(
    workspace: Path, target: Path, *, closed: Callable[[Path], AbstractContextManager[None]]
) -> None:
    """Validate and copy each immutable source byte once into a private writable destination."""
    workspace, target = _long(workspace), _long(target)
    staged = target.with_name(target.name + ".copying")
    try:
        if target.exists():
            raise RuntimeError("payload destination already exists")
        expected = json.loads((workspace.parent / "payload.json").read_bytes())
        staged.mkdir()

        def copy_one(path: Path) -> str:
            if path.stat().st_mode & stat.S_IWRITE:
                raise RuntimeError("payload is not read-only")
            destination = staged / path.relative_to(workspace)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with path.open("rb") as source, destination.open("xb") as output:
                while chunk := source.read(io.DEFAULT_BUFFER_SIZE):
                    digest.update(chunk)
                    output.write(chunk)
            return digest.hexdigest()

        with closed(workspace):
            actual = _manifest(workspace, (".",), filtered=False, read=copy_one)
            if actual != expected:
                raise RuntimeError("payload disagrees with its manifest")
        for name in actual:
            if name.endswith("/"):
                (staged / name).mkdir(parents=True, exist_ok=True)
        staged.rename(target)
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"golden payload restore refused; use a fresh run: {workspace}"
        ) from error


@dataclass
class GoldenInputs:
    """Raw declared source and external-file content for one acquisition tree."""

    root: Path
    parts: tuple[str, ...]
    contents: dict[str, str | None]
    external: dict[Path, str | None] = field(default_factory=dict)

    def key(self, recipe: Mapping[str, object], files: Sequence[Path]) -> str:
        """Bind effective recipe values and only its requested external dependencies."""
        selected: dict[str, str | None] = {}
        for path in files:
            relative = (
                path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else ""
            )
            if relative in self.contents:
                selected[str(path)] = self.contents[relative]
                continue
            if path not in self.external:
                self.external[path] = _manifest(path.parent, (path.name,))[path.name]
            selected[str(path)] = self.external[path]
        value = json.dumps([self.contents, recipe, selected], sort_keys=True).encode()
        return hashlib.sha256(value).hexdigest()

    def verify(self) -> None:
        """Refuse publication if a declared input changed while its real builder ran."""
        if _manifest(self.root, self.parts) != self.contents or any(
            _manifest(path.parent, (path.name,))[path.name] != digest
            for path, digest in self.external.items()
        ):
            raise RuntimeError("golden inputs changed during build")


def golden_inputs(root: Path, parts: tuple[str, ...]) -> GoldenInputs | None:
    """Capture managed-run inputs once; native compatibility waits for its adapter cut."""
    return GoldenInputs(root, parts, _manifest(root, parts)) if _RUN.get() is not None else None


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
