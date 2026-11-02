"""Explicit test layers subscribe to the process audit owner and the import contract."""

from __future__ import annotations

import builtins
import dis
import importlib
import inspect
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import cache, partial
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import resolve_name
from pathlib import Path
from types import CodeType, FrameType, ModuleType
from typing import Any
from unittest.mock import patch

from _pytest.capture import FDCaptureBase

from devtools.testing.audit import subscribe
from devtools.testing.sizes import load_policy

_IMPORT = builtins.__import__
_IMPORT_MODULE = importlib.import_module
_FRAMEWORK_IO = {FDCaptureBase.__init__.__code__, FDCaptureBase.done.__code__}
Backends = Sequence[Mapping[str, str]]


@cache
def backend_catalog() -> Backends:
    """Freeze the static guard's catalog once; collection passes its already-read policy."""
    rows = load_policy(Path(__file__).resolve().parents[3]).get("unit_backends", [])
    if not rows:
        raise ValueError("testing.layer_contract_unmeasured: restore the unit backend catalog")
    return tuple(rows)


def import_violations(
    names: tuple[str, ...], layer: str, backends: Backends | None = None
) -> tuple[str, ...]:
    """One import contract for the existing static guard and the runtime subscriber."""
    if layer != "unit":
        return ()
    catalog = backend_catalog() if backends is None else backends
    return tuple(
        f"testing.layer_contract: unit dependency {name}; move the test to integration"
        for name in names
        if any(
            name == row["name"] or (row["kind"] == "module" and name.startswith(row["name"] + "."))
            for row in catalog
        )
    )


class LayerContract:
    """One collected test's resource boundary; contexts own only their acquired subscriptions."""

    def __init__(self, layer: str, temp: Path, backends: Backends | None = None) -> None:
        self.layer, self.temp = layer, temp.resolve()
        self.report_temp = Path(tempfile.gettempdir()).resolve()
        self.backends = (
            backend_catalog() if layer == "unit" and backends is None else backends or ()
        )
        if layer == "unit" and not self.backends:
            raise ValueError("testing.layer_contract_unmeasured: restore the unit backend catalog")

    def refuse(self, dependency: str) -> None:
        raise AssertionError(
            f"testing.layer_contract: {self.layer} {dependency}; use its temporary directory "
            "or move the test to integration; network journeys belong in e2e"
        )

    def dependency(self, name: str) -> None:
        if self.layer != "unit":
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
            *closed.globals.values(),
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
                        and not (candidate is importlib and key == "import_module")
                        and not (candidate is builtins and key == "__import__")
                    ):
                        dependencies.append(namespace[key])
        return dependencies

    def _local_bindings(self, code: CodeType) -> None:
        """Follow explicit import aliases without reading source or guessing from word pairs."""
        aliases: dict[str, str] = {}
        owner = reference = ""
        for instruction in dis.get_instructions(code):
            op, name = instruction.opname, str(instruction.argval)
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

    def note(self, event: str, args: tuple[Any, ...]) -> None:
        if event in {"open", "os.remove"} and self.framework_io(args[0]):
            return
        if event.startswith("socket."):
            self.refuse(event)
        if self.layer != "unit":
            return
        if event == "import":
            self.dependency(str(args[0]))
        elif event in {"subprocess.Popen", "os.system", "os.startfile", "ctypes.dlopen"}:
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
            self.path(path)
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
            self.path(args[0])
            if event in {"os.rename", "os.link", "os.symlink"}:
                self.path(args[1])

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

    def path(self, raw: Any) -> None:
        if isinstance(raw, int):
            self.refuse("unproved file descriptor " + str(raw))
        try:
            path = Path(os.fsdecode(raw)).resolve()
        except (OSError, TypeError, ValueError) as error:
            self.refuse("unreadable file path: " + str(error))
            return
        if not path.is_relative_to(self.temp):
            self.refuse("file outside temporary root: " + str(path))

    @contextmanager
    def active(self) -> Iterator[None]:
        """Install transparent import checks and restore them even after a refusal."""
        with ExitStack() as stack:
            if self.layer == "unit":

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

                stack.enter_context(patch.object(builtins, "__import__", importing))
                stack.enter_context(patch.object(importlib, "import_module", importing_module))
                stack.enter_context(patch.object(sys, "dont_write_bytecode", True))
            stack.enter_context(subscribe(self.note))
            yield
