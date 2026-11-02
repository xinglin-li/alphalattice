"""Process-scoped cache inputs; opaque reads decline reuse, never the group's operation."""

from __future__ import annotations

import builtins
import genericpath
import io
import os
import stat
import sys
import threading
import weakref
from contextlib import ExitStack
from functools import wraps
from importlib.util import source_from_cache
from pathlib import Path
from types import FrameType
from typing import Any
from unittest.mock import patch

from devtools.pipeline.cache import Tree, digest

FORBIDDEN = frozenset(
    [
        "subprocess.Popen",
        "os.system",
        "os.startfile",
        "socket.connect",
        "socket.sendto",
        "socket.sendmsg",
        "ctypes.dlopen",
        "os.putenv",
        "os.unsetenv",
    ]
)
WRITES = frozenset(
    ["remove", "rename", "mkdir", "rmdir", "chmod", "utime", "link", "symlink", "truncate", "chown"]
)
_LISTDIR = globals().get("_LISTDIR") or os.listdir  # the original, bound before any recorder


_PATH_STAT = Path.stat.__code__
_PREDICATES = frozenset(
    function.__code__
    for function in (
        genericpath.exists,
        genericpath.isfile,
        genericpath.isdir,
        os.path.lexists,
        Path.exists,
        Path.is_file,
        Path.is_dir,
    )
)


_ACTIVE: weakref.ReferenceType[Recorder] | None = globals().get("_ACTIVE")
_INSTALLED: bool = globals().get("_INSTALLED", False)
_LOCK = globals().get("_LOCK") or threading.RLock()


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if _ACTIVE is not None and (owner := _ACTIVE()) is not None:
        owner.note(event, args)


class Recorder:
    def __init__(self, tree: Tree, temp: Path, native_readers: tuple[str, ...]) -> None:
        self.tree, self.temp, self.native = tree, temp.resolve(), native_readers
        self.inputs: dict[str, dict[str, Any]] = {
            k: {} for k in ("files", "probes", "listings", "env")
        }
        self.previous: weakref.ReferenceType[Recorder] | None = None
        self.reason: str | None = None
        self.environ = os.environ
        self.local, self.written, self.active = threading.local(), set[str](), False
        self.roots = (
            ("locked", Path(sys.prefix)),
            ("locked", Path(sys.base_prefix)),
            ("repo", self.tree.root),
            ("temp", self.temp),
        )

    def __enter__(self) -> Recorder:
        global _ACTIVE, _INSTALLED
        if self.active:
            self.reason = "result_cache.undeclared_input"
            return self
        self.stack = ExitStack()
        try:
            with _LOCK:
                self.previous = _ACTIVE
                prior = _ACTIVE() if _ACTIVE else None
                if prior is not None and prior.active:
                    prior.reason = self.reason = "result_cache.undeclared_input"
                for target in (os, builtins, io):
                    names = ("listdir", "open") if target is os else ("open",)
                    for name in names:
                        self.stack.enter_context(
                            patch.object(target, name, self.wrap(getattr(target, name), name))
                        )
                for name in ("stat", "lstat"):
                    self.stack.enter_context(patch.object(os, name, self.stat(getattr(os, name))))
                for name in ("exists", "isfile", "isdir"):
                    self.stack.enter_context(
                        patch.object(os.path, name, getattr(genericpath, name))
                    )
                for name in ("__getitem__", "__iter__", "__len__", "__repr__"):
                    environment_type = type(self.environ)
                    self.stack.enter_context(
                        patch.object(
                            environment_type,
                            name,
                            self.environment(getattr(environment_type, name), name),
                        )
                    )
                _ACTIVE = weakref.ref(self)
                if not _INSTALLED:
                    _INSTALLED = True  # failed or vetoed installation stays ineligible
                    sys.addaudithook(_audit)
                self.active, self.observed = True, False
                sys.audit("alphalattice.result_cache.ready", self)
                if not self.observed:
                    self.reason = "result_cache.undeclared_input"
        except BaseException as error:
            self.__exit__()
            if not isinstance(error, Exception):
                raise
            self.reason = "result_cache.undeclared_input"
        return self

    def __exit__(self, *exc: Any) -> None:
        global _ACTIVE
        with _LOCK:
            if os.environ is not self.environ:
                self.reason = self.reason or "result_cache.undeclared_input"
            self.active = False
            if _ACTIVE is not None and _ACTIVE() is self:
                _ACTIVE = self.previous
            try:
                self.stack.close()
            except Exception:
                self.reason = "result_cache.undeclared_input"

    def wrap(self, original: Any, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            owner = _ACTIVE() if _ACTIVE else None
            if owner is None or not owner.active:
                return original(*args, **kwargs)
            if name == "open":
                owner.local.pending = None
            owner.local.listing = name == "listdir"  # its own audit event is this call's
            try:
                result = original(*args, **kwargs)
            except BaseException:
                owner.reason = owner.reason or "result_cache.undeclared_input"
                raise
            finally:
                owner.local.listing = False
            if name == "open" and (pending := getattr(owner.local, "pending", None)):
                owner.written.add(pending)
            elif name == "listdir":
                path = args[0] if args else kwargs.get("path")
                owner.note(name, ("." if path is None else path,), result)
            return result

        return call

    def stat(self, original: Any) -> Any:
        @wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            owner = _ACTIVE() if _ACTIVE else None
            if owner is None or not owner.active or getattr(owner.local, "busy", False):
                return original(*args, **kwargs)
            caller: FrameType | None = sys._getframe(1)
            if caller is not None and caller.f_code is _PATH_STAT:
                caller = caller.f_back
            if caller is None or caller.f_code not in _PREDICATES:
                owner.reason = owner.reason or "result_cache.metadata_read"
                return original(*args, **kwargs)
            path = args[0] if args else kwargs.get("path")
            try:
                value = original(*args, **kwargs)
            except OSError as error:
                owner.note("probe", (path,), error)
                raise
            except BaseException:
                owner.reason = owner.reason or "result_cache.undeclared_input"
                raise
            owner.note("probe", (path,), value)
            return value

        return call

    def environment(self, original: Any, name: str) -> Any:
        @wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            owner = _ACTIVE() if _ACTIVE else None
            if (
                owner is None
                or not owner.active
                or getattr(owner.local, "busy", False)
                or (args[0] is not owner.environ and args[0] is not getattr(os, "environb", None))
            ):
                return original(*args, **kwargs)
            if name != "__getitem__":
                owner.reason = owner.reason or "result_cache.undeclared_input"
                return original(*args, **kwargs)
            key = args[1] if len(args) > 1 else kwargs.get("key")
            try:
                value = original(*args, **kwargs)
            except KeyError:
                owner.note("env", (key,), None)
                raise
            except BaseException:
                owner.reason = owner.reason or "result_cache.undeclared_input"
                raise
            owner.note("env", (key,), value)
            return value

        return call

    def probe(self, raw: Any, value: Any) -> None:
        missing = isinstance(value, (FileNotFoundError, NotADirectoryError))
        if isinstance(value, OSError) and not missing:
            self.reason = "result_cache.undeclared_input"
            return
        kind, path = self.location(raw)
        if kind == "locked" or (kind == "temp" and path in self.written):
            return
        if kind != "repo":
            self.reason = "result_cache.undeclared_input"
        elif missing:
            self.inputs["probes"][path] = None
        elif stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode):
            self.inputs["probes"][path] = "dir" if stat.S_ISDIR(value.st_mode) else "file"
        else:
            self.reason = "result_cache.undeclared_input"

    def environment_value(self, raw: Any, value: Any) -> None:
        key = os.fsdecode(raw)
        key.encode(
            "utf-8"
        )  # entry keys must serialize; the original lookup still returns its value
        found = digest(None if value is None else os.fsdecode(value))
        if key in self.inputs["env"] and self.inputs["env"][key] != found:
            self.reason = "result_cache.undeclared_input"
        self.inputs["env"][key] = found

    def location(self, raw: Any) -> tuple[str, str]:
        path = Path(os.fsdecode(raw)).absolute()
        if os.path.realpath(path) != os.path.normpath(path):
            return "outside", str(path)
        for kind, root in self.roots:
            if path.is_relative_to(root):
                return kind, path.relative_to(root).as_posix()
        return "outside", str(path)

    def file(self, raw: Any, flags: int) -> None:
        kind, relative = self.location(raw)
        writing = flags & (os.O_WRONLY | os.O_RDWR)
        if kind == "repo" and not writing and "__pycache__" in Path(relative).parts:
            try:
                source = Path(source_from_cache(str(self.tree.root / relative)))
                key = source.relative_to(self.tree.root).as_posix()
                caller = sys._getframe(1).f_back
                while caller and caller.f_globals.get("__name__") == __name__:
                    caller = caller.f_back
                if (
                    key in self.tree.blobs
                    and caller
                    and caller.f_globals.get("__name__")
                    in ("importlib._bootstrap_external", "_frozen_importlib_external")
                ):
                    self.inputs["files"][key] = self.tree.blobs[key]
                    return
            except ValueError:
                pass  # a loose bytecode-looking file is still an untracked input
        if kind == "temp":
            if writing and flags & (os.O_TRUNC | os.O_EXCL):
                self.local.pending = relative
            elif relative not in self.written:
                self.reason = "result_cache.undeclared_input"
        elif kind == "repo" and not writing and relative in self.tree.blobs:
            self.inputs["files"][relative] = self.tree.blobs[relative]
        elif kind != "locked" or writing:
            self.reason = "result_cache.undeclared_input"

    def note(self, event: str, args: tuple[Any, ...], value: Any = None) -> None:
        if not self.active or self.reason or getattr(self.local, "busy", False):
            return
        self.local.busy = True
        try:
            self.capture(event, args, value)
        except Exception:
            self.reason = "result_cache.undeclared_input"
        finally:
            self.local.busy = False

    def capture(self, event: str, args: tuple[Any, ...], value: Any) -> None:
        if event == "alphalattice.result_cache.ready" and args == (self,):
            self.observed = True
        elif event == "open":
            self.file(args[0], args[2])
        elif event == "probe":
            self.probe(args[0], value)
        elif event == "env":
            self.environment_value(args[0], value)
        elif event == "listdir":
            kind, path = self.location(args[0] if args else ".")
            if kind == "repo":
                self.inputs["listings"][path] = digest(sorted(set(value) - {"__pycache__"}))
            elif kind != "locked":
                self.reason = "result_cache.undeclared_input"
        elif event == "os.listdir" and not getattr(self.local, "listing", False):
            path = args[0] if args and args[0] is not None else "."  # a borrowed or bound listdir
            self.capture("listdir", (path,), _LISTDIR(path))
        elif event == "os.scandir":
            self.reason = "result_cache.metadata_read"
        elif self.opaque(event, args):
            self.reason = "result_cache.undeclared_input"

    def opaque(self, event: str, args: tuple[Any, ...]) -> bool:
        """A native reader's import, a child process or socket, or a write outside owned temp."""
        if event == "import":
            return any(args[0] == n or args[0].startswith(n + ".") for n in self.native)
        if event in FORBIDDEN or event.startswith("os.spawn"):
            return True
        return event.removeprefix("os.") in WRITES and any(
            self.location(p)[0] != "temp"
            for p in args[:2]
            if p is args[0] or isinstance(p, (str, bytes))
        )
