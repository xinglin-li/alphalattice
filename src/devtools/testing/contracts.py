"""Explicit test layers subscribe to the process audit owner and the import contract."""

from __future__ import annotations

import builtins
import dis
import importlib
import inspect
import ipaddress
import os
import socket
import sys
import sysconfig
import tempfile
import threading
import time
import zoneinfo
from collections.abc import Iterator, MutableMapping, MutableSequence, MutableSet
from contextlib import ExitStack, contextmanager
from functools import partial
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import resolve_name
from pathlib import Path
from types import CodeType, FrameType, ModuleType
from typing import Any, NoReturn
from unittest.mock import patch

import pytest_socket
from _pytest.capture import FDCaptureBase
from pytest_socket import disable_socket

from devtools.testing.audit import subscribe
from devtools.testing.layout import Backends, backend_catalog, import_violations

_IMPORT = builtins.__import__
_IMPORT_MODULE = importlib.import_module
_FRAMEWORK_IO = {FDCaptureBase.__init__.__code__, FDCaptureBase.done.__code__}


def _loopback_host(host: object) -> bool:
    """A name or address that stays on this machine; anything unreadable does not."""
    host = host.decode("ascii", "replace") if isinstance(host, bytes) else host
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return False


def _stays_on_this_machine(event: str, args: tuple[Any, ...]) -> bool:
    """Whether one socket audit event is an in-process Host's, which integration admits. Making a
    socket or reading this machine's name sends nothing; every address is read, never assumed."""
    if event in {"socket.__new__", "socket.gethostname"}:
        return True
    if event in {"socket.bind", "socket.connect", "socket.sendto"}:
        family = getattr(args[0], "family", None)
        if family not in (socket.AF_INET, socket.AF_INET6):
            return family == getattr(socket, "AF_UNIX", object())
        return _loopback_host(args[-1][0])
    return event in {"socket.getaddrinfo", "socket.gethostbyaddr"} and _loopback_host(args[0])


_INSTALLED = tuple(
    Path(root).resolve()
    for root in (
        *dict.fromkeys(sysconfig.get_paths()[key] for key in ("purelib", "platlib", "stdlib")),
        *zoneinfo.TZPATH,
    )
)
"""What the interpreter installed: packages, its library and the timezone database."""
_LIBRARIES = tuple(os.path.normcase(str(root)) + os.sep for root in _INSTALLED)
"""Where an installed library's modules are compiled from, as a module's file name starts."""


def _steps(code: CodeType) -> Iterator[tuple[str, str]]:
    """Each instruction's operation and name, a fused fast-local pair read as its two halves."""
    for instruction in dis.get_instructions(code):
        halves = _FUSED_FAST.get(instruction.opname)
        if halves is None:
            yield instruction.opname, str(instruction.argval)
        else:
            names = instruction.argval if len(halves) > 1 else (instruction.argval,)
            yield from zip(halves, map(str, names), strict=True)


class _LibraryImportClock:
    """A meta path finder timing an installed library's first import while a unit test runs: the
    library's own execution, outermost only and paid by the measured phase's own thread, so the
    test's own-phase budget leaves out only what delayed that phase
    (`LayerContract.library_import_seconds`). Another thread's import is not timed. It finds each
    module as the finders after it would and changes nothing they find."""

    def __init__(self, contract: LayerContract) -> None:
        self.contract = contract
        self.phase = threading.get_ident()  # the thread the measured phase runs on
        self.timing = False

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> Any:
        if threading.get_ident() != self.phase or self.timing or self not in sys.meta_path:
            return None  # another thread's, or inside a timed import whose imports are its time
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            find = getattr(finder, "find_spec", None)
            spec = None if find is None else find(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if (
            spec.origin is None
            or not os.path.normcase(spec.origin).startswith(_LIBRARIES)
            or isinstance(loader, type)
            or not hasattr(loader, "__dict__")
        ):
            return spec
        inner = loader.exec_module

        def exec_module(module: ModuleType) -> None:
            del loader.exec_module  # the loader's own method again
            self.timing = True
            started = time.perf_counter()
            try:
                inner(module)
            finally:
                self.timing = False
                self.contract.library_import_seconds += time.perf_counter() - started

        loader.exec_module = exec_module
        return spec


_PROCESS_STATE = (MutableMapping, MutableSequence, MutableSet)
# Python 3.13 and 3.14 fuse fast-local loads and stores into one instruction; each half is a step.
_FUSED_FAST = {
    "LOAD_FAST_BORROW": ("LOAD_FAST",),
    "LOAD_FAST_LOAD_FAST": ("LOAD_FAST", "LOAD_FAST"),
    "LOAD_FAST_BORROW_LOAD_FAST_BORROW": ("LOAD_FAST", "LOAD_FAST"),
    "STORE_FAST_LOAD_FAST": ("STORE_FAST", "LOAD_FAST"),
    "STORE_FAST_STORE_FAST": ("STORE_FAST", "STORE_FAST"),
}
"""A module global of these kinds holds what the process put there (a cache earlier tests
filled), not code: the walk never reads its contents, so a unit test's verdict never depends on
which tests ran before it in its worker. Its members are judged when they run."""


class LayerContract:
    """One collected test's resource boundary; contexts own only their acquired subscriptions."""

    def __init__(self, layer: str, temp: Path, backends: Backends | None = None) -> None:
        self.layer, self.temp = layer, temp.resolve()
        self.report_temp = Path(tempfile.gettempdir()).resolve()
        self._descriptors: dict[int, tuple[Path, tuple[int, int, int]]] = {}
        self._descriptor_lock = threading.RLock()
        self._opening = threading.local()
        self.library_import_seconds = 0.0
        """The time a library's first import took while this test ran (`_LibraryImportClock`)."""
        self.backends = (
            backend_catalog() if layer == "unit" and backends is None else backends or ()
        )
        if layer == "unit" and not self.backends:
            raise ValueError("testing.layer_contract_unmeasured: restore the unit backend catalog")

    def refuse(self, dependency: str) -> NoReturn:
        raise AssertionError(
            f"testing.layer_contract: {self.layer} {dependency}; use its temporary directory "
            "or move the test to integration; network journeys belong in e2e"
        )

    def dependency(self, name: str) -> None:
        if self.layer != "unit" or self.library_importing():
            return
        if violations := import_violations((name,), self.layer, self.backends):
            raise AssertionError(violations[0])

    def references(
        self, values: list[Any], *, module_inventory: frozenset[CodeType] = frozenset()
    ) -> None:
        """Inspect only selected callables and their referenced bindings, never sys.modules."""
        if self.layer != "unit":
            return
        seen: set[int] = set()
        while values:
            value = values.pop()
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, partial):
                values.extend([value.func, *value.args, *value.keywords.values()])
                continue
            if inspect.ismethod(value):
                values.extend([value.__func__, value.__self__])
                continue
            if isinstance(value, dict):
                values.extend(value.values())
                continue
            if isinstance(value, (tuple, list, set, frozenset)):
                values.extend(value)
                continue
            values.extend(self._bindings(value, module_inventory))

    def _bindings(self, value: Any, module_inventory: frozenset[CodeType]) -> list[Any]:
        """Validate a concrete binding and project the inputs its selected callable can reach."""
        name = (
            value.__name__
            if isinstance(value, ModuleType)
            else getattr(value, "__module__", type(value).__module__)
        )
        if not isinstance(name, str) or not name:
            self.refuse("unproved dependency " + type(value).__qualname__)
        self.dependency(name)
        callable_name = getattr(value, "__name__", "")
        if callable_name:
            module_name = "os" if name in {"nt", "posix"} else name
            self.dependency(module_name + "." + callable_name)
        if value is _IMPORT or value is _IMPORT_MODULE:
            self.refuse("unproved bound import loader " + name)
        if inspect.isfunction(value) and not name.startswith(("_pytest.", "pytest.")):
            if name.partition(".")[0] in sys.stdlib_module_names:
                return []
            return self._function_bindings(value, module_inventory)
        elif inspect.isclass(value) and name.startswith(("tests.", "alphalattice.", "devtools.")):
            return list(vars(value).values())
        return []

    def _function_bindings(self, value: Any, module_inventory: frozenset[CodeType]) -> list[Any]:
        """The selected function's captured values and referenced module members, read once."""
        self._local_bindings(value.__code__)
        if {"sys", "modules"} <= set(
            value.__code__.co_names
        ) and value.__code__ not in module_inventory:
            self.refuse("unproved loaded-module inventory reader " + value.__qualname__)
        closed = inspect.getclosurevars(value)
        dependencies = [
            *(held for held in closed.globals.values() if not isinstance(held, _PROCESS_STATE)),
            *closed.nonlocals.values(),
            *(value.__defaults__ or ()),
            *(value.__kwdefaults__ or {}).values(),
        ]
        for candidate in tuple(dependencies):
            if isinstance(candidate, ModuleType):
                namespace = vars(candidate)
                for key in value.__code__.co_names:
                    if candidate is sys and key == "modules":
                        if value.__code__ not in module_inventory:
                            self.refuse(
                                "unproved loaded-module inventory reader " + value.__qualname__
                            )
                        continue
                    if (
                        key in namespace
                        and not isinstance(namespace[key], _PROCESS_STATE)
                        and not (candidate is importlib and key == "import_module")
                        and not (candidate is builtins and key == "__import__")
                    ):
                        dependencies.append(namespace[key])
        return dependencies

    def _local_bindings(self, code: CodeType) -> None:
        """Follow explicit import aliases without reading source or guessing from word pairs."""
        aliases: dict[str, str] = {}
        owner = reference = ""
        for op, name in _steps(code):
            if op == "IMPORT_NAME":
                owner = reference = name
            elif op == "IMPORT_FROM":
                reference = owner + "." + name
            elif op in {"STORE_FAST", "STORE_DEREF", "STORE_NAME", "STORE_GLOBAL"}:
                if reference:
                    aliases[name] = reference
                else:
                    aliases.pop(name, None)
                reference = ""
            elif op in {"LOAD_FAST", "LOAD_DEREF", "LOAD_NAME", "LOAD_GLOBAL"}:
                reference = aliases.get(name, "")
            elif op in {"LOAD_ATTR", "LOAD_METHOD"}:
                reference = reference + "." + name if reference else ""
            else:
                reference = ""
            if reference and any(
                row["kind"] == "callable" and row["name"] == reference for row in self.backends
            ):
                self.dependency(reference)

    def _forget_closed(self) -> None:
        if self.layer != "unit" or os.name != "posix" or getattr(self._opening, "active", False):
            return
        with self._descriptor_lock:
            for fd in tuple(self._descriptors):
                try:
                    os.fstat(fd)
                except OSError:
                    self._descriptors.pop(fd, None)

    def note(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            self._forget_closed()
        if event in {"open", "os.remove"} and self.framework_io(args[0]):
            return
        if event.startswith("socket.") and not (
            self.layer == "integration" and _stays_on_this_machine(event, args)
        ):
            self.refuse(event)
        if self.layer != "unit":
            return
        if event == "import":
            self.dependency(str(args[0]))
        elif event in {"subprocess.Popen", "os.system", "os.startfile", "ctypes.dlopen"}:
            if event == "ctypes.dlopen" and (
                self.file_stamp_load(args[0]) or self.library_importing()
            ):
                return
            self.refuse(event + ": " + str(args[0]))
        elif event in {"os.listdir", "os.scandir"}:
            if not self.import_read():
                self.path(args[0])
        elif event == "open":
            path, _mode, flags = args
            writing = bool(
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            importing_source = isinstance(path, (str, bytes)) and os.fsdecode(path).endswith(
                (".py", ".pyc", *EXTENSION_SUFFIXES)
            )
            if not writing and importing_source and self.import_read():
                return
            if not writing and self.installed_read(path):
                return
            self.path(getattr(self._opening, "path", path))
        elif event.startswith("os.") and event[3:] in {
            "remove",
            "rename",
            "mkdir",
            "rmdir",
            "chmod",
            "utime",
            "link",
            "symlink",
            "truncate",
            "chown",
        }:
            self._mutation(event, args)

    def _mutation(self, event: str, args: tuple[Any, ...]) -> None:
        if event in {"os.rename", "os.link"}:
            self.path(args[0], dir_fd=args[2])
            self.path(args[1], dir_fd=args[3])
        elif event == "os.symlink":
            self.path(args[0])
            self.path(args[1], dir_fd=args[2])
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.chown", "os.utime"}:
            self.path(args[0], dir_fd=args[-1])
        else:
            self.path(args[0])

    def framework_io(self, raw: Any) -> bool:
        """Only pytest's exact capture constructors and closers may touch report files."""
        frame: FrameType | None = sys._getframe(1)
        while frame and frame.f_globals.get("__name__") in {
            __name__,
            "devtools.testing.audit",
            "tempfile",
        }:
            frame = frame.f_back
        if frame is None or frame.f_code not in _FRAMEWORK_IO:
            return False
        return isinstance(raw, int) or Path(os.fsdecode(raw)).resolve().is_relative_to(
            self.report_temp
        )

    def file_stamp_load(self, raw: Any) -> bool:
        """The identity owner's file stamp reading the Windows change clock: a file stat, which
        needs kernel32 for GetFileInformationByHandleEx. Only that library, loaded directly by
        that one caller; every other native load stays refused."""
        if str(raw).lower() not in {"kernel32", "kernel32.dll"}:
            return False
        frame: FrameType | None = sys._getframe(1)
        while frame and frame.f_globals.get("__name__") in {
            __name__,
            "devtools.testing.audit",
            "ctypes",
        }:
            frame = frame.f_back
        return bool(
            frame
            and frame.f_globals.get("__name__") == "alphalattice.kernel.identity.file_facts"
            and frame.f_code.co_name == "handle_stamp"
        )

    def library_importing(self) -> bool:
        """Whether an installed library (never the product's src/) is being imported, so the
        native load or import now made is that library's own, as pandas loads user32 through
        dateutil and imports subprocess at its import: a library's first use, which lands in
        whichever test imports it first. The module whose import is nearest decides; what a
        product module's import makes, and every load or import outside an import, stay
        refused."""
        frame: FrameType | None = sys._getframe(1)
        while frame:
            caller = frame.f_back
            if (
                frame.f_code.co_name == "<module>"
                and caller is not None
                and caller.f_globals.get("__name__", "").startswith(
                    ("importlib.", "_frozen_importlib")
                )
            ):
                # The import system compiles a module from its absolute file name; a name
                # spelled otherwise matches no root and stays refused.
                return os.path.normcase(frame.f_code.co_filename).startswith(_LIBRARIES)
            frame = caller
        return False

    def import_read(self) -> bool:
        """Only the importing reader itself, not arbitrary code called during an import."""
        frame: FrameType | None = sys._getframe(1)
        while frame and frame.f_globals.get("__name__") in {__name__, "devtools.testing.audit"}:
            frame = frame.f_back
        return bool(
            frame
            and frame.f_globals.get("__name__", "").startswith(
                ("importlib.", "_frozen_importlib", "_pytest.assertion.rewrite")
            )
        )

    def installed_read(self, raw: Any) -> bool:
        """A library's first-use read of what the interpreter installed, which lands in whichever
        test comes first: package metadata by `importlib.metadata`, installed packages and the
        timezone database. A project or user file read any other way stays refused."""
        frame: FrameType | None = sys._getframe(1)
        while frame and frame.f_globals.get("__name__") in {
            __name__,
            "devtools.testing.audit",
            "pathlib",
            "io",
        }:
            frame = frame.f_back
        if frame and frame.f_globals.get("__name__") == "importlib.metadata":
            return True
        if isinstance(raw, int):
            return False
        try:
            path = Path(os.fsdecode(raw)).resolve()
        except (OSError, TypeError, ValueError):
            return False
        return any(path.is_relative_to(root) for root in _INSTALLED)

    def path(self, raw: Any, *, dir_fd: int | None = None) -> Path:
        if isinstance(raw, int):
            with self._descriptor_lock:
                record = self._descriptors.get(raw)
                if record is None:
                    self.refuse("unproved file descriptor " + str(raw))
                try:
                    stat = os.fstat(raw)
                    identity = (stat.st_dev, stat.st_ino, stat.st_mode & 0o170000)
                except OSError:
                    identity = None
                if record[1] != identity:
                    self._descriptors.pop(raw, None)
                    self.refuse("unproved file descriptor " + str(raw))
                return record[0]
        try:
            path = Path(os.fsdecode(raw))
            if dir_fd not in (None, -1):
                path = self.path(dir_fd) / path
            path = path.resolve()
        except (OSError, TypeError, ValueError) as error:
            self.refuse("unreadable file path: " + str(error))
        if not path.is_relative_to(self.temp):
            self.refuse("file outside temporary root: " + str(path))
        return path

    @contextmanager
    def active(self) -> Iterator[None]:
        """Install transparent import checks and restore them even after a refusal."""
        with ExitStack() as stack:
            if self.layer == "unit":
                # Keep the audit subscriber too: it catches already-bound socket objects and
                # operations that pytest-socket does not wrap, including bind and sendto.
                contract = self

                class ContractSocketBlocked(RuntimeError):
                    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                        contract.refuse("socket")

                # Translate at the attempted call, so a caller can catch the same named
                # assertion as it could when the audit owner supplied the refusal.
                stack.enter_context(
                    patch.object(pytest_socket, "SocketBlockedError", ContractSocketBlocked)
                )
                for name in ("socket", "getaddrinfo", "gethostbyname"):
                    stack.callback(setattr, socket, name, getattr(socket, name))
                disable_socket()
                clock = _LibraryImportClock(self)
                sys.meta_path.insert(0, clock)
                stack.callback(sys.meta_path.remove, clock)

                def importing(
                    name: str,
                    globals: Any = None,
                    locals: Any = None,
                    fromlist: Any = (),
                    level: int = 0,
                ) -> Any:
                    package = (globals or {}).get("__package__", "")
                    self.dependency(resolve_name("." * level + name, package) if level else name)
                    return _IMPORT(name, globals, locals, fromlist, level)

                def importing_module(name: str, package: str | None = None) -> Any:
                    self.dependency(resolve_name(name, package) if name.startswith(".") else name)
                    return _IMPORT_MODULE(name, package)

                if os.name == "posix":
                    opened, closed, stated, duplicated = os.open, os.close, os.stat, os.dup2

                    def open_fd(
                        path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
                    ) -> int:
                        with self._descriptor_lock:
                            target = self.path(path, dir_fd=dir_fd) if dir_fd is not None else None
                            previous = getattr(self._opening, "path", None)
                            was_active = getattr(self._opening, "active", False)
                            self._opening.active = True
                            if target is not None:
                                self._opening.path = target
                            try:
                                fd = opened(path, flags, mode, dir_fd=dir_fd)
                            finally:
                                self._opening.active = was_active
                                if previous is None:
                                    self._opening.__dict__.pop("path", None)
                                else:
                                    self._opening.path = previous
                            try:
                                target = target or Path(os.fsdecode(path)).resolve()
                                stat = os.fstat(fd)
                                if target.is_relative_to(self.temp):
                                    self._descriptors[fd] = (
                                        target,
                                        (stat.st_dev, stat.st_ino, stat.st_mode & 0o170000),
                                    )
                                else:
                                    self._descriptors.pop(fd, None)
                            except (OSError, TypeError, ValueError):
                                self._descriptors.pop(fd, None)
                            return fd

                    def close_fd(fd: int) -> None:
                        with self._descriptor_lock:
                            try:
                                closed(fd)
                            finally:
                                self._descriptors.pop(fd, None)

                    def dup2_fd(fd: int, target: int, inheritable: bool = True) -> int:
                        with self._descriptor_lock:
                            try:
                                return duplicated(fd, target, inheritable=inheritable)
                            finally:
                                if fd != target:
                                    self._descriptors.pop(target, None)

                    def stat_fd(
                        path: Any, *, dir_fd: int | None = None, follow_symlinks: bool = True
                    ) -> Any:
                        if isinstance(path, int) or dir_fd is not None:
                            self.path(path, dir_fd=dir_fd)
                        return stated(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

                    for name, wrapper in (
                        ("open", open_fd),
                        ("close", close_fd),
                        ("stat", stat_fd),
                        ("dup2", dup2_fd),
                    ):
                        stack.enter_context(patch.object(os, name, wrapper))
                    for name in ("supports_fd", "supports_dir_fd", "supports_follow_symlinks"):
                        original = getattr(os, name)
                        added = {
                            new
                            for old, new in ((opened, open_fd), (stated, stat_fd))
                            if old in original
                        }
                        if added:
                            stack.enter_context(patch.object(os, name, original | added))

                stack.enter_context(patch.object(builtins, "__import__", importing))
                stack.enter_context(patch.object(importlib, "import_module", importing_module))
                stack.enter_context(patch.object(sys, "dont_write_bytecode", True))
            stack.enter_context(subscribe(self.note))
            yield
