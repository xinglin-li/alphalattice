"""One process environment snapshot and the declarations that can change its settings."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Protocol, overload

CPU_SHARE = "OMP_NUM_THREADS"
NETWORK_DISABLED = "ALPHALATTICE_NETWORK_DISABLED"
RETENTION_DAYS = "ALPHALATTICE_RETENTION_DAYS"
FILE_SCRUB_DAYS = "ALPHALATTICE_FILE_SCRUB_DAYS"
PROCESS_MEMORY_MODE = "ALPHALATTICE_PROCESS_MEMORY_MODE"

_PUBLIC_NAMES = frozenset(
    {
        CPU_SHARE,
        FILE_SCRUB_DAYS,
        NETWORK_DISABLED,
        PROCESS_MEMORY_MODE,
        RETENTION_DAYS,
        "ALPHALATTICE_BACKUP_ROOT",
        "ALPHALATTICE_DATA_ROOT",
        "ALPHALATTICE_FILE_TRUST",
        "ALPHALATTICE_MODEL_STORE",
        "ALPHALATTICE_SHELL",
        "ALPHALATTICE_TERMINAL_PANEL_HOST",
        "ALPHALATTICE_WORK_CENSUS",
        "BAGGAGE",
        "CLAUDE_CODE_SESSION_ID",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "CODEX_THREAD_ID",
        "LOCALAPPDATA",
        "MSYSTEM",
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
        "PATH",
        "PROGRAMDATA",
        "SEC_USER_AGENT",
        "TRACEPARENT",
        "UV",
        "UV_RUN_RECURSION_DEPTH",
        "VIRTUAL_ENV",
        "XDG_DATA_HOME",
    }
)


def _count(value: str) -> int | None:
    value = value.strip()
    if value.isdecimal() and (number := int(value)) > 0:
        return number
    return None


@dataclass(frozen=True)
class Settings:
    """The operator's typed process policy. Unset limits leave their owner's default in force."""

    cpu_cores: int | None
    retention_days: int | None
    network_disabled: bool
    file_scrub_days: int = 7
    process_memory_mode: str = "enforce"

    @classmethod
    def from_values(cls, values: Mapping[str, str]) -> Settings:
        """Project the operator's settings from an admitted environment snapshot."""
        process_memory_mode = values.get(PROCESS_MEMORY_MODE, "enforce")
        if process_memory_mode not in {"observe", "enforce"}:
            raise ValueError("spawn.memory_mode_invalid: use observe or enforce")
        return cls(
            cpu_cores=_count(values.get(CPU_SHARE, "")),
            retention_days=_count(values.get(RETENTION_DAYS, "")),
            network_disabled=values.get(NETWORK_DISABLED) == "1",
            file_scrub_days=_count(values.get(FILE_SCRUB_DAYS, "")) or 7,
            process_memory_mode=process_memory_mode,
        )


@dataclass(frozen=True, repr=False)
class Secret:
    """A process secret that formats as redacted in diagnostics."""

    _value: str

    def reveal(self) -> str:
        """Give the secret to the consumer authorized to use it."""
        return self._value

    def __repr__(self) -> str:
        """Keep the value out of diagnostics."""
        return "<redacted>"

    __str__ = __repr__


class ConfigPort(Protocol):
    """The kernel port for typed policy, public values and protected settings."""

    @property
    def settings(self) -> Settings:
        """The operator's typed policy."""
        ...

    def get(self, name: str, default: str | None = None) -> str | None:
        """Read an ordinary setting."""
        ...

    def secret(self, name: str) -> Secret | None:
        """Read a protected value in its diagnostic-safe form."""
        ...


@dataclass(frozen=True)
class Change:
    """Declare which environment values a trusted caller changes or clears."""

    kind: str
    values: Mapping[str, str | None] = field(repr=False)


@dataclass(frozen=True)
class ConfigEvent:
    """The declaration a consumer can record without retaining any values."""

    kind: str
    names: tuple[str, ...]


class _ChildEnvironment(dict[str, str]):
    """Raw values for process transport with a redacted diagnostic representation."""

    def __repr__(self) -> str:
        return "<private child environment>"

    __str__ = __repr__


@dataclass(frozen=True)
class Config:
    """A sealed process view. Public reads and diagnostic representation omit secrets."""

    _values: Mapping[str, str] = field(repr=False)
    public: Mapping[str, str] = field(init=False, repr=False)
    settings: Settings = field(init=False)
    event: ConfigEvent | None = None

    def __post_init__(self) -> None:
        """Hold independent raw values and their public projection."""
        raw = dict(self._values)
        public = {key: value for key, value in raw.items() if key.upper() in _PUBLIC_NAMES}
        object.__setattr__(self, "_values", MappingProxyType(raw))
        object.__setattr__(self, "public", MappingProxyType(public))
        object.__setattr__(self, "settings", Settings.from_values(public))

    @classmethod
    def capture(cls, read: Callable[[], Mapping[str, str]]) -> Config:
        """Call the startup reader once and hold an independent snapshot."""
        return cls(read())

    @overload
    def get(self, name: str, default: str) -> str: ...

    @overload
    def get(self, name: str, default: None = None) -> str | None: ...

    def get(self, name: str, default: str | None = None) -> str | None:
        """Read one ordinary setting. Protected names are served by `secret`."""
        return self.public.get(name, default)

    def secret(self, name: str) -> Secret | None:
        """Return the protected value in a type safe to format or log."""
        value = self._values.get(name)
        return Secret(value) if value is not None and name.upper() not in _PUBLIC_NAMES else None

    def child_environment(self) -> dict[str, str]:
        """Make the private transport payload for an authorized child process."""
        return _ChildEnvironment(self._values)

    def apply(self, change: Change) -> Config:
        """Make a new view from a declaration without rereading the process environment."""
        values = dict(self._values)
        for name, value in change.values.items():
            if value is None:
                values.pop(name, None)
            else:
                values[name] = value
        return Config(values, event=ConfigEvent(change.kind, tuple(sorted(change.values))))


_scoped: ContextVar[Config | None] = ContextVar("alphalattice_config")
_startup_lock = Lock()
_startup: Config | None = None


def current() -> Config:
    """The process's startup snapshot or the calling scope's declared view."""
    scope = _scoped.get(None)
    if scope is not None:
        return scope
    global _startup
    if _startup is not None:
        return _startup
    with _startup_lock:
        return _start()


DATA_ROOT_ENV = "ALPHALATTICE_DATA_ROOT"
"""Names the folder of this process's data root; unset, the application data folder."""


def data_root() -> Path:
    """The process's data root: `ALPHALATTICE_DATA_ROOT`, else the application data folder.

    It is read here, with the standard library alone, so the console's thin path can find the
    data root's Host (kernel.data_layer.root re-exports it for the data layer)."""
    values = current()
    override = values.get(DATA_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        local = values.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "AlphaLattice"
    data = values.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "alphalattice"


def _start() -> Config:
    """Capture the process environment once with the startup lock held."""
    global _startup
    if _startup is None:
        _startup = Config.capture(lambda: os.environ)
    return _startup


def _apply_process(change: Change, *, native: bool) -> Config:
    """Publish a process declaration with the startup lock held."""
    global _startup
    scope = _scoped.get(None)
    changed = (scope if scope is not None else _start()).apply(change)
    if native:
        # Some third-party libraries read their own platform environment during import.
        for name, value in change.values.items():
            if value is None:
                with suppress(KeyError):
                    del os.environ[name]
            else:
                os.environ[name] = value
    if scope is None:
        _startup = changed
    else:
        _scoped.set(changed)
    return changed


def declare(change: Change, *, native: bool = False) -> Config:
    """Apply a declared process change without silently rereading the environment."""
    with _startup_lock:
        return _apply_process(change, native=native)


def consume_secret(name: str) -> Secret | None:
    """Take a protected handoff once and remove it from subsequent child environments."""
    with _startup_lock:
        scope = _scoped.get(None)
        held = (scope if scope is not None else _start()).secret(name)
        if held is not None:
            _apply_process(Change("secret-consumed", {name: None}), native=scope is None)
        return held


@contextmanager
def process_override(change: Change, *, native: bool = False) -> Iterator[Config]:
    """Hold a declared change while a process-owned operation borrows its settings."""
    with _startup_lock:
        scope = _scoped.get(None)
        before = scope if scope is not None else _start()
        changed = _apply_process(change, native=native)
    try:
        yield changed
    finally:
        prior = {name: before._values.get(name) for name in change.values}
        declare(Change("restore-" + change.kind, prior), native=native)


@contextmanager
def using(config: Config) -> Iterator[Config]:
    """Bind an explicitly admitted view to a caller, including its nested reads."""
    token = _scoped.set(config)
    try:
        yield config
    finally:
        _scoped.reset(token)


@contextmanager
def override(change: Change) -> Iterator[Config]:
    """Declare a scoped change. Exiting restores the enclosing view."""
    with using(current().apply(change)) as config:
        yield config
