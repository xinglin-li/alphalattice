"""Invocation-scoped golden ownership and immutable payloads; domain builders stay with callers."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import logging
import os
import re
import shutil
import stat
import threading
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class GoldenRuntime:
    """Caller-supplied OS ownership primitives, never product imports in evaluation."""

    lock: Callable[..., AbstractContextManager[object]]
    birth: Callable[[int], float | None]
    held: Callable[[Path], bool]
    confined: Callable[[Path, str], Path]

    @classmethod
    def from_os(
        cls,
        processes: Any,
        locking: Any,
        confined: Callable[[Path, str], Path],
    ) -> GoldenRuntime:
        """Compose caller-owned OS drivers without product or private implementation imports."""

        @contextmanager
        def build_lock(
            path: Path, *, shared: bool = False, blocking: bool = True, ready: Path | None = None
        ) -> Iterator[bool | None]:
            """Share complete entries; exclusively own an incomplete entry after releasing it."""
            path = confined(payload_path(path.parent), path.name)
            locker = locking.portalocker.Win32Locker() if os.name == "nt" else locking
            nonblocking = locking.LockFlags(0) if blocking else locking.LOCK_NB
            flags = locking.LOCK_SH if shared or ready is not None else locking.LOCK_EX
            with path.open("a+b") as stream:
                locker.lock(stream, flags | nonblocking)
                held = True
                try:
                    complete = ready.is_file() if ready is not None else None
                    if ready is not None and not complete:
                        locker.unlock(stream)
                        held = False
                        locker.lock(stream, locking.LOCK_EX | nonblocking)
                        held = True
                        complete = ready.is_file()
                    yield complete
                finally:
                    if held:
                        locker.unlock(stream)

        def birth(pid: int) -> float | None:
            try:
                return cast(float, processes.Process(pid).create_time())
            except processes.NoSuchProcess:
                return None
            except processes.Error as error:
                raise RuntimeError("golden controller birth is unverified") from error

        def held(path: Path) -> bool:
            try:
                with build_lock(path, blocking=False):
                    return False
            except locking.AlreadyLocked:
                return True

        return cls(build_lock, birth, held, confined)


@dataclass(frozen=True)
class GoldenRun:
    """The canonical invocation root shared by its workers, never a persistent cache."""

    root: Path
    runtime: GoldenRuntime
    readers: ExitStack = field(default_factory=ExitStack, compare=False, repr=False)
    generations: set[Path] = field(default_factory=set, compare=False, repr=False)
    releases: list[Callable[[], None]] = field(default_factory=list, compare=False, repr=False)
    state: dict[str, bool] = field(
        default_factory=lambda: {"closed": False}, compare=False, repr=False
    )
    mutex: Any = field(default_factory=threading.RLock, compare=False, repr=False)

    def hold(self, generation: Path, release: Callable[[], None]) -> bool:
        """Keep an OS reader lease until this invocation releases its linked files."""
        generation = payload_path(generation)
        with self.mutex:
            if self.state["closed"]:
                raise RuntimeError("golden reader invocation is closed")
            if generation in self.generations:
                return False
            lock = generation.parent / f".{generation.name}.readers.lock"
            self.readers.enter_context(self.runtime.lock(lock, shared=True))
            self.generations.add(generation)
            self.releases.append(release)
            return True

    def close_readers(self) -> None:
        """Record each last use while still leased; retain a marker if recording fails."""
        with self.mutex:
            if self.state["closed"]:
                return
            self.state["closed"] = True
            try:
                for release in self.releases:
                    try:
                        release()
                    except (OSError, RuntimeError, ValueError) as error:
                        logging.getLogger(__name__).warning("golden release unmeasured: %s", error)
            finally:
                self.readers.close()

    def pin(self, key: str) -> None:
        with self.runtime.lock(self.root / "index.lock"):
            state = _run_state(self.root, self.runtime)
            if (
                state["closed"]
                or self.runtime.birth(state["pid"]) != state["birth"]
                or not self.runtime.held(self.root / "controller.lock")
            ):
                raise RuntimeError("golden controller is not live; start a fresh run")
            if key not in state["keys"]:
                state["keys"].append(key)
                self.runtime.confined(self.root, "run.json").write_text(
                    json.dumps(state), encoding="utf-8"
                )


def _run_state(root: Path, runtime: GoldenRuntime) -> dict[str, Any]:
    state: dict[str, Any] = json.loads(runtime.confined(root, "run.json").read_bytes())
    types = {"pid": int, "birth": float, "run": str, "keys": list, "closed": bool}
    if (
        not isinstance(state, dict)
        or state.get("schema") != "alphalattice.golden-run.v1"
        or any(type(state.get(key)) is not kind for key, kind in types.items())
    ):
        raise RuntimeError("golden controller record is unverified")
    if (
        state["run"] != str(payload_path(root))
        or state["pid"] <= 0
        or not 0 < state["birth"] < float("inf")
        or not all(isinstance(key, str) for key in state["keys"])
    ):
        raise RuntimeError("golden controller identity is unverified")
    return state


@contextmanager
def golden_controller(root: Path, *, runtime: GoldenRuntime) -> Iterator[Path]:
    """Keep the controller's actual birth and requested keys until its explicit close or death."""
    root = payload_path(root)
    root.mkdir(parents=True, exist_ok=True)
    with runtime.lock(root / "controller.lock", blocking=False):
        with runtime.lock(root / "index.lock"):
            path = runtime.confined(root, "run.json")
            if path.exists():
                raise RuntimeError("golden controller root already used; choose a fresh run")
            state = dict(
                schema="alphalattice.golden-run.v1",
                run=str(root),
                pid=os.getpid(),
                birth=runtime.birth(os.getpid()),
                keys=[],
                closed=False,
            )
            path.write_text(json.dumps(state), encoding="utf-8")
        try:
            yield root
        finally:
            with runtime.lock(root / "index.lock"):
                state = _run_state(root, runtime)
                state["closed"] = True
                path.write_text(json.dumps(state), encoding="utf-8")


def controller_reclaimable(root: Path, *, runtime: GoldenRuntime) -> bool:
    """Retain busy or unprovable controllers; PID reuse never revives a closed owner."""
    with runtime.lock(root / "index.lock"):
        state = _run_state(root, runtime)
        if runtime.held(root / "controller.lock"):
            return False
        return bool(state["closed"] or runtime.birth(state["pid"]) != state["birth"])


_RUN: ContextVar[GoldenRun | None] = ContextVar("golden_run", default=None)
_ACQUIRED: list[dict[str, Any]] = []


def note_acquisition(row: dict[str, Any]) -> None:
    """Record what one acquisition cost, for the test that made it (`take_acquisitions`)."""
    _ACQUIRED.append(row)


def take_acquisitions() -> list[dict[str, Any]]:
    """This process's acquisitions since the last read: the test plugin reads them per test."""
    rows = _ACQUIRED[:]
    _ACQUIRED.clear()
    return rows


@contextmanager
def golden_run(root: Path, *, runtime: GoldenRuntime, controller: bool = True) -> Iterator[Path]:
    """Bind acquisition and discovery to the controller's invocation and lifetime lease."""
    run = GoldenRun(root.parent if os.environ.get("PYTEST_XDIST_WORKER") else root, runtime)
    with golden_controller(run.root, runtime=runtime) if controller else nullcontext():
        token = _RUN.set(run)
        try:
            yield run.root
        finally:
            try:
                run.close_readers()
            finally:
                _RUN.reset(token)


def hold_generation(generation: Path, release: Callable[[], None]) -> bool:
    """Pin a shared payload for the bound invocation, recording its last use on release."""
    run = _RUN.get()
    if run is None:
        raise RuntimeError("golden reader has no invocation; start a managed run")
    return run.hold(generation, release)


def generation_held(generation: Path) -> bool:
    """Whether this invocation has already pinned the generation it will link."""
    run = _RUN.get()
    if run is None:
        return False
    with run.mutex:
        return not run.state["closed"] and payload_path(generation) in run.generations


def golden_root(fallback: Path) -> Path:
    """Select the invocation's golden directory, retaining explicit non-pytest callers."""
    run = _RUN.get()
    return run.root / "golden-workspaces" if run is not None else fallback


GOLDEN_ROOT_VARIABLE = "ALPHALATTICE_TEST_GOLDEN_ROOT"


def persistent_golden_root(default: Path) -> Path:
    """The cross-invocation golden cache: the operator's shared root when set, else the checkout's.

    Its entries are keyed by their build closure and leased, so checkouts may share one root.
    """
    return Path(configured) if (configured := os.environ.get(GOLDEN_ROOT_VARIABLE)) else default


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


def payload_path(path: Path) -> Path:
    """Admit a no-follow payload or lock path using the platform's long-path spelling."""
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
    workspace: Path,
    *,
    closed: Callable[[Path], AbstractContextManager[None]],
    parent: Path | None = None,
) -> None:
    """Seal only a closed, regular-file payload; immutable bytes never include outward links.

    A derived golden's `parent` is the sealed payload it was built from: the files it links from
    there are that payload's own sealed files, so they must already be read-only and are left as
    they are (a derived build copied its parent's 3.6 GB whole, mac1 11-07). Every other file keeps
    the rule, so tests linking from the parent meanwhile change nothing here."""
    workspace = payload_path(workspace)
    try:
        inherited: set[tuple[int, int]] = set()
        if parent is not None:
            for folder, _, names in os.walk(payload_path(parent)):
                for name in names:
                    info = os.lstat(os.path.join(folder, name))
                    if stat.S_ISREG(info.st_mode):
                        inherited.add((info.st_dev, info.st_ino))
        with closed(workspace):
            contents = _manifest(workspace, (".",), filtered=False)
            files = [
                (workspace / name, (workspace / name).stat())
                for name in contents
                if not name.endswith("/")
            ]
            if any(
                info.st_mode & 0o222 for _, info in files if (info.st_dev, info.st_ino) in inherited
            ):
                raise RuntimeError("payload links a writable file of its parent")
            files = [(p, info) for p, info in files if (info.st_dev, info.st_ino) not in inherited]
            counts = Counter((info.st_dev, info.st_ino) for _, info in files)
            if any(info.st_nlink > counts[info.st_dev, info.st_ino] for _, info in files):
                raise RuntimeError("payload has a hard link outside its root")
            for path, info in files:
                path.chmod(info.st_mode & ~0o222)
            manifest = workspace.parent / "payload.json"
            manifest.write_text(json.dumps(contents, sort_keys=True), encoding="utf-8")
            manifest.chmod(stat.S_IREAD)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"golden payload cannot be sealed; close its writer and use a fresh run: {workspace}"
        ) from error


class GoldenPayloadCorrupt(RuntimeError):
    """A readable payload differs from the sealed evidence recorded for it."""


def verified_payload(workspace: Path) -> dict[str, str]:
    """Read a sealed golden through its no-follow manifest and refuse changed or writable files."""
    workspace = payload_path(workspace)
    try:
        manifest = workspace.parent / "payload.json"
        if manifest.is_symlink() or manifest.is_junction():
            raise ValueError("payload manifest is a link")
        if manifest.stat().st_mode & 0o222:
            raise ValueError("payload manifest is writable")
        expected = json.loads(manifest.read_bytes())
        actual = _manifest(workspace, (".",), filtered=False)
        if not isinstance(expected, dict) or actual != expected:
            raise ValueError("payload differs from its manifest")
        files = {
            name.removeprefix("./"): digest
            for name, digest in actual.items()
            if not name.endswith("/")
        }
        if any((workspace / name).stat().st_mode & 0o222 for name in files):
            raise ValueError("payload contains a writable file")
        return cast(dict[str, str], files)
    except (FileNotFoundError, ValueError, TypeError, UnicodeError) as error:
        raise GoldenPayloadCorrupt(
            f"golden payload verification refused: {workspace}: {error}"
        ) from error
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"golden payload verification unmeasured: {workspace}: {error}"
        ) from error


def _kernel32() -> Any:
    """Windows: kernel32 with CreateFileW, SetFileInformationByHandle and CloseHandle typed."""
    import ctypes.wintypes as wintypes  # lazy: the Windows path alone uses wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel


# Windows: each session seed file read in place, held open so that no writer can open it.
_HELD: dict[str, int] = {}


def refuse_writers(path: str) -> None:
    """A file read in place refuses every writer, through any of its links, and each link
    still goes with its test's directory.

    POSIX: its write bits go off; unlinking a name needs only its directory. Windows keeps
    one read-only attribute per file, shared by every link, so a link could not be deleted
    without making the source writable: one handle that shares reading and deletion, never
    writing, is held instead, and a writer's open fails with a sharing violation.
    """
    if os.name != "nt":
        os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) & ~0o222)
        return
    if path in _HELD:
        return

    kernel = _kernel32()
    # GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, OPEN_EXISTING
    handle = kernel.CreateFileW(str(Path(path).absolute()), 0x80000000, 0x1 | 0x4, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error(), f"test_datasets.hold_refused: {path}")
    _HELD[path] = handle  # closed with the process; the source outlives every reader of it


def remove_tree(path: Path, root: Path) -> list[str]:
    """Remove a passed test's own scratch `path`, inside `root` (its run's basetemp).

    A link (symlink or junction) is removed itself, never followed; a sealed read-only file, or a
    hard link to one, is deleted without touching the attribute every link of it shares. On Windows
    pytest's own cleanup (rmtree, ignoring errors) cannot delete such a file, so each passed
    journey's whole scratch stayed to the run's end (11.5 GB unique in four runs).
    Returns each entry it could not remove with its error, for the caller to name; never raises.
    """
    try:
        if (
            ".." in path.parts
            or ".." in root.parts
            or payload_path(path) == payload_path(root)
            or not payload_path(path).is_relative_to(payload_path(root))
            or path.is_symlink()
            or path.is_junction()
        ):
            return [f"refused: {path} is not inside {root}"]
    except (OSError, RuntimeError) as error:
        return [f"refused: {path}: {error}"]
    if os.name == "nt":
        return _remove_windows(path, root)
    if not shutil.rmtree.avoids_symlink_attacks:
        return [f"refused: safe directory traversal is unavailable: {path}"]
    left: list[str] = []
    try:
        parts = path.relative_to(root).parts
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        before = os.stat(root, follow_symlinks=False)
        with ExitStack() as held:
            parent = os.open(root, flags)
            held.callback(os.close, parent)
            after = os.fstat(parent)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError("basetemp identity changed")
            for part in parts[:-1]:
                parent = os.open(part, flags, dir_fd=parent)
                held.callback(os.close, parent)
            if stat.S_ISLNK(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False).st_mode):
                os.unlink(parts[-1], dir_fd=parent)
            else:
                shutil.rmtree(
                    parts[-1],
                    dir_fd=parent,
                    onexc=lambda _, name, error: left.append(f"{name}: {error}"),
                )
    except (OSError, ValueError, RuntimeError) as error:
        left.append(f"refused: {path}: {error}")
    return left


def _remove_windows(path: Path, root: Path) -> list[str]:
    """Pin every traversed directory against rename; delete each name by its own handle."""

    kernel = _kernel32()
    left: list[str] = []

    def opened(name: Path, stack: ExitStack) -> tuple[Any, int]:
        # DELETE | LIST_DIRECTORY | READ_ATTRIBUTES; share read/write, never delete.
        text = str(name.absolute())
        if not text.startswith("\\\\?\\"):
            text = "\\\\?\\" + ("UNC\\" + text[2:] if text.startswith("\\\\") else text)
        handle = kernel.CreateFileW(text, 0x10000 | 0x1 | 0x80, 0x3, None, 3, 0x02200000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        stack.callback(kernel.CloseHandle, handle)
        info = (ctypes.c_ulong * 2)()
        if not kernel.GetFileInformationByHandleEx(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return handle, info[0]

    def final(handle: Any) -> Path:
        buf = ctypes.create_unicode_buffer(32768)
        n = kernel.GetFinalPathNameByHandleW(handle, buf, len(buf), 0)
        if not n or n >= len(buf):
            raise ctypes.WinError(ctypes.get_last_error())
        return Path(buf.value)

    def delete(handle: Any) -> None:
        flags = ctypes.c_ulong(0x1 | 0x2 | 0x10)
        if not kernel.SetFileInformationByHandle(handle, 21, ctypes.byref(flags), 4):
            raise ctypes.WinError(ctypes.get_last_error())

    def walk(folder: Path, handle: Any) -> None:
        try:
            with os.scandir(str(payload_path(folder))) as listed:
                names = [entry.name for entry in listed]
            for name in names:
                child = folder / name
                try:
                    with ExitStack() as stack:
                        opened_child, attrs = opened(child, stack)
                        if attrs & 0x10 and not attrs & 0x400:
                            walk(child, opened_child)
                        else:
                            delete(opened_child)
                except (OSError, RuntimeError) as error:
                    left.append(f"{child}: {error}")
            delete(handle)
        except (OSError, RuntimeError) as error:
            left.append(f"{folder}: {error}")

    try:
        with ExitStack() as stack:
            base, attrs = opened(root, stack)
            if not attrs & 0x10 or attrs & 0x400:
                raise ValueError("basetemp is not a plain directory")
            current = root
            for part in path.relative_to(root).parts:
                current /= part
                handle, attrs = opened(current, stack)
                if not attrs & 0x10 or attrs & 0x400:
                    raise ValueError(f"scratch ancestor is not a plain directory: {current}")
            if not final(handle).is_relative_to(final(base)) or final(handle) == final(base):
                raise ValueError("scratch handle is outside basetemp")
            walk(path, handle)
    except (OSError, RuntimeError, ValueError) as error:
        left.append(f"{path}: {error}")
    return left


def remove_idle_reader_lock(path: Path) -> None:
    """Under the key lock and after reader exclusion, remove only its plain lock name."""
    path = payload_path(path)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or (getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    ):
        raise RuntimeError("golden reader lock is not an owned regular file")
    path.unlink()


def in_invocation(path: Path, *, shared: Path | None = None) -> bool:
    """A managed test may read its own golden or an explicitly owned shared root in place."""
    run = _RUN.get()
    if run is None:
        return False
    source = payload_path(path)
    return source.is_relative_to(payload_path(run.root)) or (
        shared is not None and source.is_relative_to(payload_path(shared))
    )


def read_in_place(relative: str, size: int) -> bool:
    """The payload classes no test writes (measured on the explorer goldens, 11-03): research
    inputs, and content-addressed files of 1 MB or more; everything else stays a private copy."""
    stem = relative.rsplit("/", 1)[-1].split(".", 1)[0]
    return relative.startswith("research-inputs/") or (
        size >= 1 << 20 and re.fullmatch(r"[0-9a-f]{64}", stem) is not None
    )


def copy_payload(
    workspace: Path,
    target: Path,
    *,
    closed: Callable[[Path], AbstractContextManager[None]],
    link: bool = False,
) -> None:
    """Validate and copy each immutable source byte once into a private writable destination.

    With `link` (a test's acquisition), the classes no test writes (read_in_place) are
    linked instead, each hashed against the manifest all the same. A link is the sealed file
    itself, so the seal's protection holds whatever process lives: POSIX write bits off, and on
    Windows the read-only attribute, shared by every link, refusing every writer even after
    the acquiring process exits. The cost, on Windows only: a plain unlink cannot remove the
    link, so pytest's tmp cleanup leaves it (it ignores errors) and the run's remover, which
    removes read-only names, takes it with the run. Without `link`, nothing is linked.
    """
    workspace, target = payload_path(workspace), payload_path(target)
    created = False
    try:
        if target.exists():
            raise RuntimeError("payload destination already exists")
        try:
            manifest = workspace.parent / "payload.json"
            if manifest.is_symlink() or manifest.is_junction():
                raise RuntimeError("payload manifest is a link")
            if manifest.stat().st_mode & 0o222:
                raise GoldenPayloadCorrupt("payload manifest is writable")
            expected = json.loads(manifest.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise GoldenPayloadCorrupt("golden payload manifest is missing or malformed") from error
        # A derived variant's own folder, under a parent that exists, never through a link.
        target.parent.mkdir(exist_ok=True)
        if target.parent.is_symlink() or target.parent.is_junction():
            raise RuntimeError("payload destination is reached through a link")
        # The target is private to this acquisition until return; a derived golden
        # remains under its build lock without a completion marker until sealed.
        target.mkdir()
        created = True

        def copy_one(path: Path) -> str:
            info = path.stat()
            if info.st_mode & 0o222:
                raise GoldenPayloadCorrupt("payload is not read-only")
            relative = path.relative_to(workspace).as_posix()
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            if link and read_in_place(relative, info.st_size):
                with path.open("rb") as source:
                    while chunk := source.read(io.DEFAULT_BUFFER_SIZE):
                        digest.update(chunk)
                os.link(path, destination)  # the seal refuses its writers
                return digest.hexdigest()
            with path.open("rb") as source, destination.open("xb") as output:
                while chunk := source.read(io.DEFAULT_BUFFER_SIZE):
                    digest.update(chunk)
                    output.write(chunk)
            return digest.hexdigest()

        with closed(workspace):
            actual = _manifest(workspace, (".",), filtered=False, read=copy_one)
            if actual != expected:
                raise GoldenPayloadCorrupt("payload disagrees with its manifest")
        for name in actual:
            if name.endswith("/"):
                (target / name).mkdir(parents=True, exist_ok=True)
    except (GoldenPayloadCorrupt, OSError, ValueError, RuntimeError) as error:
        # The destination was created by this call; never remove an existing one.
        left = remove_tree(target, target.parent) if created else []
        detail = f"; incomplete destination retained: {left}" if left else ""
        if isinstance(error, GoldenPayloadCorrupt):
            raise GoldenPayloadCorrupt(
                f"golden payload restore refused: {workspace}: {error}{detail}"
            ) from error
        raise RuntimeError(
            f"golden payload restore refused; use a fresh run: {workspace}{detail}"
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
        key = hashlib.sha256(value).hexdigest()
        if run := _RUN.get():
            run.pin(key)
        return key

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
