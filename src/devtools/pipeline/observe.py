"""Process-scoped cache inputs; opaque reads decline reuse, never the group's operation."""

from __future__ import annotations

import builtins
import genericpath
import io
import json
import logging
import os
import stat
import sys
import threading
import weakref
from contextlib import ExitStack
from fnmatch import fnmatchcase
from functools import wraps
from importlib.util import source_from_cache
from pathlib import Path
from types import FrameType
from typing import Annotated, Any, Literal
from unittest.mock import patch

import pytest
from _pytest.assertion.rewrite import _read_pyc, _rewrite_test
from _pytest.config import _get_directory
from _pytest.logging import LoggingPlugin
from _pytest.main import validate_basetemp
from _pytest.pathlib import cleanup_dead_symlinks, samefile_nofollow
from _pytest.runner import _update_current_test_var
from _pytest.tmpdir import TempPathFactory
from _pytest.tmpdir import pytest_sessionfinish as _temp_finish
from pydantic import BaseModel, ConfigDict, Field

from devtools.pipeline.cache import Entry, Hex64, Store, Text, Tree, digest
from devtools.pipeline.receipt import TestResult, register
from devtools.pipeline.receipt import pytest_configure as receipt_configure
from devtools.testing.audit import dispatch
from devtools.testing.audit import subscribe as subscribe
from devtools.testing.requirements import file_holders
from devtools.testing.sizes import classify, load_policy

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
_DEVNULL = os.path.normcase(os.path.realpath(os.devnull))


_PATH_STAT = Path.stat.__code__
_FRAMEWORK_STATS = frozenset(
    f.__code__
    for f in (
        LoggingPlugin.__init__,
        _rewrite_test,
        _read_pyc,
        validate_basetemp,
        samefile_nofollow,
        _get_directory,
        TempPathFactory.getbasetemp,
        TempPathFactory._ensure_relative_to_basetemp,
        _temp_finish,
        cleanup_dead_symlinks,
    )
)
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


_audit = dispatch  # existing observer seam; the kernel owns the only dispatcher

_ACTIVE: weakref.ReferenceType[Recorder] | None = globals().get("_ACTIVE")
_LOCK = globals().get("_LOCK") or threading.RLock()


def _caller() -> FrameType | None:
    frame: FrameType | None = sys._getframe(1)
    while frame and frame.f_globals.get("__name__") in {
        __name__,
        "devtools.testing.audit",
        "pathlib",
    }:
        frame = frame.f_back
    return frame


def _pytest_current(key: Any) -> bool:
    if key not in ("PYTEST_CURRENT_TEST", b"PYTEST_CURRENT_TEST"):
        return False
    frame = _caller()
    while frame and frame.f_globals.get("__name__") in {
        "os",
        "_collections_abc",
        "collections.abc",
    }:
        frame = frame.f_back
    return frame is not None and frame.f_code is _update_current_test_var.__code__


class Recorder:
    def __init__(self, tree: Tree, temp: Path, native_readers: tuple[str, ...]) -> None:
        self.tree, self.temp, self.native = tree, temp.resolve(), native_readers
        self.inputs: dict[str, dict[str, Any]] = {
            k: {} for k in ("files", "probes", "listings", "env")
        }
        self.previous: weakref.ReferenceType[Recorder] | None = None
        self.reason: str | None = None
        self.output: tuple[Path, Any, Any] | None = None
        self.log: Path | None = None
        self.fresh_temp = False  # only the pytest owner may claim an absent basetemp
        self.environ = os.environ
        self.local, self.written, self.active = threading.local(), set[str](), False
        self.roots = (
            ("locked", Path(sys.prefix)),
            ("locked", Path(sys.base_prefix)),
            ("repo", self.tree.root),
            ("temp", self.temp),
        )

    def __enter__(self) -> Recorder:
        global _ACTIVE
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
                if self.fresh_temp:
                    self.stack.enter_context(
                        patch("_pytest.assertion.rewrite._read_pyc", lambda *args: None)
                    )
                _ACTIVE = weakref.ref(self)
                self.active, self.observed = True, False
                self.stack.enter_context(subscribe(self.note))
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
                owner.local.known_open = False
            owner.local.listing = name == "listdir"  # its own audit event is this call's
            try:
                result = original(*args, **kwargs)
            except BaseException:
                if name != "open" or not getattr(owner.local, "known_open", False):
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
            outer = _caller() if owner.fresh_temp else None
            while outer and outer.f_code in _PREDICATES:
                outer = outer.f_back
            framework = outer is not None and outer.f_code in _FRAMEWORK_STATS
            if not framework and (caller is None or caller.f_code not in _PREDICATES):
                owner.reason = owner.reason or "result_cache.metadata_read"
                return original(*args, **kwargs)
            path = args[0] if args else kwargs.get("path")
            try:
                value = original(*args, **kwargs)
            except OSError as error:
                owner.note("framework_probe" if framework else "probe", (path,), error)
                raise
            except BaseException:
                owner.reason = owner.reason or "result_cache.undeclared_input"
                raise
            owner.note("framework_probe" if framework else "probe", (path,), value)
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
            if _pytest_current(key):
                return original(*args, **kwargs)
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

    def probe(self, raw: Any, value: Any, framework: bool = False) -> None:
        missing = isinstance(value, (FileNotFoundError, NotADirectoryError))
        if isinstance(value, OSError) and not missing:
            self.reason = "result_cache.undeclared_input"
            return
        if framework and self.framework_path(raw, value):
            return
        kind, path = self.location(raw)
        if kind == "locked" or (kind == "temp" and (path in self.written or framework)):
            return
        if kind != "repo":
            self.reason = "result_cache.undeclared_input"
        elif missing:
            self.inputs["probes"][path] = None
        elif stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode):
            self.inputs["probes"][path] = "dir" if stat.S_ISDIR(value.st_mode) else "file"
            if framework and path in self.tree.blobs:
                self.inputs["files"][path] = self.tree.blobs[path]
        else:
            self.reason = "result_cache.undeclared_input"

    def framework_path(self, raw: Any, value: Any) -> bool:
        """Only owned temporary outputs or entailed directory ancestors need no cache input."""
        absolute = Path(os.fsdecode(raw)).absolute()
        if absolute.is_relative_to(self.temp) and Path(os.path.realpath(absolute)).is_relative_to(
            self.temp
        ):
            return True
        return (
            not isinstance(value, OSError)
            and stat.S_ISDIR(value.st_mode)
            and (
                absolute in self.tree.root.parents
                or absolute in self.temp.parents
                or (self.log is not None and absolute == self.log.parent)
            )
        )

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
        real = os.path.realpath(path)
        if os.path.normcase(real) == _DEVNULL:
            return "null", ""
        if real != os.path.normpath(path):
            return "outside", str(path)
        for kind, root in self.roots:
            if path.is_relative_to(root):
                return kind, path.relative_to(root).as_posix()
        return "outside", str(path)

    def file(self, raw: Any, flags: int) -> None:
        kind, relative = self.location(raw)
        writing = flags & (os.O_WRONLY | os.O_RDWR)
        self.local.known_open = kind == "null" or (kind == "locked" and not writing)
        if kind == "null":
            return  # the process's exact null device carries no mutable input
        if writing and Path(relative).suffix == ".pyc":
            self.reason = "result_cache.undeclared_input"
            return
        if (
            kind == "repo"
            and not writing
            and "__pycache__" in Path(relative).parts
            and self.bytecode(relative)
        ):
            return
        if kind == "temp":
            if writing and flags & (os.O_TRUNC | os.O_EXCL):
                self.local.pending = relative
            elif relative not in self.written:
                self.reason = "result_cache.undeclared_input"
        elif kind == "repo" and not writing and relative in self.tree.blobs:
            self.inputs["files"][relative] = self.tree.blobs[relative]
        elif kind != "locked" or writing:
            self.reason = "result_cache.undeclared_input"

    def bytecode(self, relative: str) -> bool:
        """Bind only import-system cache reads to their tracked source, never loose bytecode."""
        try:
            caller = _caller()
            context = caller.f_globals if caller else {}
            if context.get("__name__") == "_pytest.assertion.rewrite" and caller:
                source = caller.f_locals["source"]
                expected = context["get_cache_dir"](source) / (
                    source.name[:-3] + context["PYC_TAIL"]
                )
                if expected != self.tree.root / relative:
                    raise ValueError("not the assertion import's bytecode")
            else:
                source = Path(source_from_cache(str(self.tree.root / relative)))
            key = source.relative_to(self.tree.root).as_posix()
            if (
                key in self.tree.blobs
                and caller
                and caller.f_globals.get("__name__")
                in (
                    "importlib._bootstrap_external",
                    "_frozen_importlib_external",
                    "_pytest.assertion.rewrite",
                )
            ):
                self.local.known_open = True
                self.inputs["files"][key] = self.tree.blobs[key]
                return True
        except (KeyError, ValueError):
            pass  # a loose bytecode-looking file is still an untracked input
        return False

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

    def output_file(self, raw: Any, flags: int) -> bool:
        """Only the framework's exact output opener may write its declared sink."""
        if not flags & os.O_WRONLY:
            return False
        caller = _caller()
        if caller is None:
            return False
        if self.output and caller.f_code is self.output[1]:
            return Path(raw).absolute() == self.output[0]
        return (
            self.log is not None
            and caller.f_code is logging.FileHandler._open.__code__
            and Path(raw).absolute() == self.log
        )

    def receipt_environment(self) -> bool:
        frame = _caller()
        while frame and frame.f_globals.get("__name__") in {
            "os",
            "_collections_abc",
            "collections.abc",
        }:
            frame = frame.f_back
        return bool(self.output and frame and frame.f_code is self.output[2])

    def capture(self, event: str, args: tuple[Any, ...], value: Any) -> None:
        if event == "alphalattice.result_cache.ready" and args == (self,):
            self.observed = True
        elif event == "open":
            if (self.output or self.log) and self.output_file(args[0], args[2]):
                return
            self.file(args[0], args[2])
        elif event == "probe":
            self.probe(args[0], value)
        elif event == "framework_probe":
            self.probe(args[0], value, framework=True)
        elif event == "env":
            if (
                self.output
                and args[0] == "ALPHALATTICE_PIPELINE_EVENTS"
                and self.receipt_environment()
            ):
                return
            self.environment_value(args[0], value)
        elif event == "listdir":
            self.listing(args[0] if args else ".", value)
        elif event == "os.listdir" and not getattr(self.local, "listing", False):
            path = args[0] if args and args[0] is not None else "."  # a borrowed or bound listdir
            self.capture("listdir", (path,), _LISTDIR(path))
        elif event == "os.scandir":
            self.scandir(args[0])
        elif self.opaque(event, args):
            self.reason = "result_cache.undeclared_input"

    def listing(self, raw: Any, value: Any) -> None:
        kind, path = self.location(raw)
        if kind == "repo":
            self.inputs["listings"][path] = digest(sorted(set(value) - {"__pycache__"}))
        elif kind != "locked" and not (kind == "temp" and self.fresh_temp):
            self.reason = "result_cache.undeclared_input"

    def scandir(self, raw: Any) -> None:
        """Only pytest collection may bind native directory kinds to verified tree probes."""
        kind, path = self.location(raw)
        caller = _caller()
        if self.fresh_temp and kind == "temp":
            return
        if (
            self.fresh_temp
            and kind == "repo"
            and caller
            and caller.f_globals.get("__name__") == "_pytest.pathlib"
            and caller.f_code.co_name in {"scandir", "visit"}
        ):
            names = _LISTDIR(raw)  # collection owns kinds; an extra enumeration binds names
            self.capture("listdir", (raw,), names)
            for name in names:
                if name != "__pycache__":
                    key = (Path(path) / name).as_posix()
                    probe = self.tree._look("probe", key)
                    if probe not in {"file", "dir", None}:
                        self.reason = "result_cache.metadata_read"
                    self.inputs["probes"][key] = probe
        else:
            self.reason = "result_cache.metadata_read"

    def opaque(self, event: str, args: tuple[Any, ...]) -> bool:
        """A native reader's import, a child process or socket, or a write outside owned temp."""
        if event == "import":
            return any(args[0] == n or args[0].startswith(n + ".") for n in self.native)
        if event in {"os.putenv", "os.unsetenv"} and args and _pytest_current(args[0]):
            return False
        if event in FORBIDDEN or event.startswith("os.spawn"):
            return True
        return event.removeprefix("os.") in WRITES and any(
            self.location(p)[0] != "temp"
            for p in args[:2]
            if p is args[0] or isinstance(p, (str, bytes))
        )


class _Request(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    store: Path
    cap: Annotated[int, Field(gt=0)] | None = None
    events: Path | None = None
    log: Path | None = None
    key: Hex64
    writer: Annotated[dict[Literal["line", "head", "invocation", "at"], Text], Field(min_length=4)]
    native_readers: tuple[Text, ...]
    basetemp: Path
    verdict: Path


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--result-cache-record", help="result_cache.request_invalid")


@pytest.hookimpl(tryfirst=True)  # type: ignore[untyped-decorator]
def pytest_load_initial_conftests(early_config: Any) -> None:
    """Opt in before collection imports; requests belong to the runner, never test state."""
    options = early_config.known_args_namespace
    if not options.result_cache_record:
        return
    try:
        request = _Request.model_validate_json(Path(options.result_cache_record).read_bytes())
        paths = (
            request.basetemp,
            request.store,
            request.verdict,
            *([request.events] if request.events is not None else []),
            *([request.log] if request.log is not None else []),
        )
        if (
            not sys.dont_write_bytecode
            or not early_config.pluginmanager.is_blocked("cacheprovider")
            or any(not p.is_absolute() or p.absolute() != p.resolve() for p in paths)
            or request.basetemp.exists()
            or request.basetemp.is_relative_to(early_config.rootpath)
            or Path(options.basetemp or "").absolute() != request.basetemp
            or any(
                p.is_relative_to(request.basetemp) or p.is_relative_to(early_config.rootpath)
                for p in paths[1:]
            )
            or any(
                not all(part.isidentifier() for part in n.split("."))
                for n in request.native_readers
            )
        ):
            raise ValueError("recording context")
        tree = Tree.read(early_config.rootpath)
        owner = _PytestRecording(request, tree)
        early_config.pluginmanager.register(owner, "result-cache-recording")
        early_config.add_cleanup(owner.finish)
        if owner.seen is not None:
            owner.seen.__enter__()
    except (OSError, ValueError, TypeError) as error:
        raise pytest.UsageError("result_cache.request_invalid") from error


class _PytestRecording:
    def __init__(self, request: _Request, tree: Tree | str) -> None:
        self.request = request
        self.unavailable = tree if isinstance(tree, str) else None
        self.seen = (
            None
            if isinstance(tree, str)
            else Recorder(tree, request.basetemp, request.native_readers)
        )
        if self.seen is not None:
            self.seen.fresh_temp = True
            self.seen.log = request.log
            if request.events is not None:
                self.seen.output = (request.events, register.__code__, receipt_configure.__code__)
        self.session: Any = None
        self.results: dict[str, tuple[str, float]] = {}
        self.properties: dict[str, dict[str, Any]] = {}
        self.phases: dict[str, set[str]] = {}
        self.nodes: set[str] = set()
        self.bad, self.closed = False, False

    def pytest_collection_finish(self, session: Any) -> None:
        """The registered size owner excludes Large, quarantined and unknown holders."""
        if self.seen is None:
            return
        try:
            root = self.seen.tree.root
            policy, holders = load_policy(root), {}
            for item in session.items:
                path, _, name = item.nodeid.partition("::")
                if path not in holders:
                    holders[path] = {
                        h.node for h in file_holders(path, (root / path).read_text("utf-8"))
                    }
                declared = [m.name for m in item.iter_markers()]
                size = classify(item.nodeid, set(declared) & {"small", "medium", "large"}, policy)[
                    0
                ]
                self.bad |= (
                    not name
                    or name in self.nodes
                    or path + "::" + name.split("[", 1)[0] not in holders[path]
                    or size == "large"
                    or bool(set(declared) & {"e2e", "native", "real_evidence"})
                    or bool(set(Path(path).parts) & {"e2e", "native"})
                    or any(
                        r["node"] in {item.nodeid, item.nodeid.partition("[")[0]}
                        or fnmatchcase(item.nodeid, r["node"])
                        for r in policy.get("quarantine", [])
                    )
                )
                self.nodes.add(name)
            self.bad |= not self.nodes
        except Exception:
            self.bad = True  # an unreadable policy or unparseable holder cannot be cached

    def pytest_runtest_logreport(self, report: Any) -> None:
        name = report.nodeid.partition("::")[2]
        self.properties[name] = {
            k: v
            for k, v in report.user_properties
            if k in {"setup_seconds", "call_seconds", "teardown_seconds"}
        }
        phases = self.phases.setdefault(name, set())
        self.bad |= (
            report.when in phases
            or report.failed
            or hasattr(report, "wasxfail")
            or report.outcome not in {"passed", "skipped"}
            or bool(getattr(report, "rerun", 0))
        )
        phases.add(report.when)
        outcome, seconds = self.results.get(name, ("PASSED", 0.0))
        self.results[name] = ("SKIPPED" if report.skipped else outcome, seconds + report.duration)

    def modules(self) -> None:
        """One loaded-module inventory also catches readers imported before observation."""
        if self.seen is None:
            return
        native = sys.modules.get(os.name)
        fast = tuple(
            f
            for n in (
                "_path_exists",
                "_path_isfile",
                "_path_isdir",
                "_path_islink",
                "_path_lexists",
            )
            if (f := getattr(native, n, None)) is not None
        )
        for name, module in tuple(sys.modules.items()):
            if any(name == n or name.startswith(n + ".") for n in self.seen.native):
                self.seen.reason = "result_cache.native_reader"
            raw = getattr(module, "__file__", None)
            if not isinstance(raw, str) or raw.startswith("<"):
                continue
            if raw.endswith(".pyc"):
                raw = source_from_cache(raw)
            self.seen.note("open", (raw, "r", os.O_RDONLY))
            path = Path(os.path.realpath(raw))
            if path.is_relative_to(self.seen.tree.root):
                key = path.relative_to(self.seen.tree.root).as_posix()
                if key in self.seen.tree.blobs and any(
                    value is predicate
                    for value in tuple(vars(module).values())
                    for predicate in fast
                ):
                    self.seen.reason = self.seen.reason or "result_cache.undeclared_input"

    def pytest_sessionfinish(self, session: Any) -> None:
        self.session = session  # its final exitstatus includes every session-finish hook

    def finish(self) -> None:
        if self.closed:
            return
        seen = self.seen
        reason: str | None = self.unavailable or "result_cache.undeclared_input"
        try:
            if seen is not None:
                self.modules()
                incomplete = self.nodes != self.results.keys() or any(
                    not {"setup", "teardown"} <= phases for phases in self.phases.values()
                )
                reason = (
                    "result_cache.not_passing"
                    if self.bad
                    or self.session is None
                    or self.session.exitstatus
                    or incomplete
                    or sys.exception() is not None
                    else seen.reason
                )
        except Exception:
            pass  # a recorder failure declines reuse, never the group's real outcome
        finally:
            self.close()
        self.publish(reason, seen)

    def publish(self, reason: str | None, seen: Recorder | None) -> None:
        """The existing store owns publication; an unusable verdict never changes test outcomes."""
        request, entry = self.request, None
        try:
            if reason is None and seen is not None:
                store = Store.open(request.store, request.cap)
                if isinstance(store, str):
                    reason = store
                else:
                    details = {}
                    if request.events is not None:
                        for line in request.events.read_text("utf-8").splitlines():
                            result = TestResult.model_validate_json(line).model_dump(
                                mode="json", by_alias=True
                            )
                            node = result.pop("node")
                            name = node.partition("::")[2]
                            details[name] = {
                                **result,
                                "_node": node,
                                "_properties": self.properties.get(name, {}),
                            }
                        if details.keys() != self.results.keys():
                            raise ValueError("incomplete receipt outcomes")
                    candidate = Entry.sealed(
                        details=details,
                        key=request.key,
                        writer=request.writer,
                        results=self.results,
                        **seen.inputs,
                    )
                    store.write(candidate)
                    entry = store.lookup(request.key, seen.tree)
                    if entry and entry.digest != candidate.digest:
                        entry = None
                    reason = None if entry is not None else "result_cache.unwritable"
        except Exception:
            reason, entry = "result_cache.unwritable", None
        try:
            verdict = {
                "decision": "STORED" if entry else "UNCACHEABLE",
                "reason": reason,
                "entry": {"key": entry.key, "digest": entry.digest} if entry else None,
                "properties": self.properties,
            }
            request.verdict.write_text(json.dumps(verdict) + "\n", "utf-8")
        except Exception:
            pass  # missing verdict is an unusable recording to the caller, not a test failure

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self.seen is not None:
                self.seen.__exit__()
