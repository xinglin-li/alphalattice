"""Invocation-scoped golden ownership and immutable payloads; domain builders stay with callers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
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


@contextmanager
def golden_run(root: Path, *, runtime: GoldenRuntime, controller: bool = True) -> Iterator[Path]:
    """Bind acquisition and discovery to the controller's invocation and lifetime lease."""
    run = GoldenRun(root.parent if os.environ.get("PYTEST_XDIST_WORKER") else root, runtime)
    with golden_controller(run.root, runtime=runtime) if controller else nullcontext():
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
    workspace: Path, *, closed: Callable[[Path], AbstractContextManager[None]]
) -> None:
    """Seal only a closed, regular-file payload; immutable bytes never include outward links."""
    workspace = payload_path(workspace)
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
    workspace, target = payload_path(workspace), payload_path(target)
    staged = target.with_name(target.name + ".copying")
    try:
        if target.exists():
            raise RuntimeError("payload destination already exists")
        expected = json.loads((workspace.parent / "payload.json").read_bytes())
        # A derived variant's own folder, under a parent that exists, never through a link.
        staged.parent.mkdir(exist_ok=True)
        if staged.parent.is_symlink() or staged.parent.is_junction():
            raise RuntimeError("payload destination is reached through a link")
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
