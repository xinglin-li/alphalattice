"""Explicit Host capabilities. A workspace's `plugins.toml` has an `[enabled]`
table of capability names to tagged BLAKE3 hashes. Each name refers to
`plugins/<name>/plugin.json`, which declares an entry and its source files.
The workspace setting is read when the Host composes. Restart after changing it.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Lock
from types import MappingProxyType
from typing import Annotated, Final, Protocol
from uuid import uuid7

from pydantic import StringConstraints, ValidationError

from alphalattice.capabilities.domain_contracts.domain.base import DomainModel, ShortString
from alphalattice.kernel.door import DECISIONS_DATABASE, Decision, Decisions
from alphalattice.kernel.identity import integrity_blake3_canonical, is_blake3
from alphalattice.protocols.plugin_contract import ActionSpec, EffectMode, ExecutionMode, Lease
from alphalattice.protocols.plugin_contract.contracts import JsonObject

_GROUP: Final = "alphalattice.capabilities"
_PACKAGE: Final = Path(__file__).resolve().parents[3]
_NAME = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)*")
_ENTRY = re.compile(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*:[a-zA-Z_]\w*", re.ASCII)
_SourcePath = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_./-]+$")]


class PluginState(StrEnum):
    """A static registration, its verified closure and its worker lifecycle."""

    DISCOVERED = "DISCOVERED"
    VERIFIED = "VERIFIED"
    ACTIVE = "ACTIVE"
    EVICTED = "EVICTED"


class PluginRefused(ValueError):
    """A refusal identifies the registration the caller needs to correct."""

    def __init__(
        self,
        code: str,
        name: str,
        *,
        fingerprint: str | None = None,
        effects: tuple[str, ...] = (),
        decision: Decision | None = None,
    ) -> None:
        """Carry the registration name and, for an approval refusal, its decision fields."""
        self.code, self.name = code, name
        self.fingerprint = fingerprint
        self.effects = effects
        self.decision = decision
        super().__init__(f"{code}: {name}")


class _Manifest(DomainModel):
    """Static ActionSpec fields plus the exact source files that form the plugin."""

    effect_mode: EffectMode
    lease: Lease
    input_schema: JsonObject
    output_schema: JsonObject
    inputs: tuple[ShortString, ...] = ()
    as_of_param: ShortString | None = None
    entry: str | None = None  # Only a workspace manifest supplies its own target.
    files: tuple[_SourcePath, ...]


class PluginWorker(Protocol):
    """A runtime returns this handle after loading, and owns the process and its stop."""

    def is_alive(self) -> bool:
        """Report whether the worker that loaded the closure is still running."""
        ...


@dataclass(frozen=True, slots=True)
class VerifiedPlugin:
    """The worker's exact action spec, entry target and verified source bytes."""

    spec: ActionSpec
    entry: str
    sources: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class _Registration:
    name: str
    entry: str
    folder: Path
    manifest_path: Path
    manifest: _Manifest
    manifest_bytes: bytes
    expected_hash: str | None

    def verify(self) -> VerifiedPlugin:
        """Bind the same bytes the worker gets to the manifest checked at discovery."""
        from alphalattice.control.workspace_runtime.verified_facts import (  # lazy: activation only
            full_content_proof,
            verified_bytes,
        )

        try:
            with full_content_proof():
                manifest, manifest_digest = verified_bytes(_file(self.folder, self.manifest_path))
                sources = {
                    path: verified_bytes(_file(self.folder, Path(path)))
                    for path in self.manifest.files
                }
        except (OSError, ValueError) as error:
            raise PluginRefused("plugin_registry.closure_changed", self.name) from error
        role = integrity_blake3_canonical(
            {
                "form": "plugin-closure-v1",
                "name": self.name,
                "entry": self.entry,
                "manifest": manifest_digest,
                "sources": {path: digest for path, (_, digest) in sources.items()},
            }
        )
        if manifest != self.manifest_bytes or (
            self.expected_hash is not None and role != self.expected_hash
        ):
            raise PluginRefused("plugin_registry.closure_changed", self.name)
        spec = ActionSpec(
            capability_id=self.name,
            plugin_id=self.name,
            code_role_hash=role,
            effect_mode=self.manifest.effect_mode,
            execution_mode=(
                ExecutionMode.TRUSTED_POOL
                if self.expected_hash is None
                else ExecutionMode.SANDBOX_SUBPROCESS
            ),
            **self.manifest.model_dump(exclude={"schema_version", "effect_mode", "entry", "files"}),
        )
        return VerifiedPlugin(
            spec, self.entry, MappingProxyType({path: body for path, (body, _) in sources.items()})
        )


@dataclass(slots=True)
class _Activation:
    registration: _Registration
    state: PluginState = PluginState.DISCOVERED
    worker: PluginWorker | None = None
    lock: Lock = field(default_factory=Lock)

    def live_worker(self) -> PluginWorker | None:
        if self.worker is not None and not self.worker.is_alive():
            self.worker = None
            self.state = PluginState.EVICTED
        return self.worker


class PluginRegistry:
    """One Host's installed and explicitly enabled capabilities."""

    def __init__(self, workspace: Path) -> None:
        """Read the installed group and only the workspace's enabled manifests."""
        root = workspace.resolve()
        self._decisions = Decisions(root / DECISIONS_DATABASE, uuid7)
        self._actions: dict[str, _Activation] = {}
        self._refusals: dict[str, PluginRefused] = {}
        for entry in _builtin_entries():
            self._add(
                _registration(
                    entry.name,
                    _PACKAGE,
                    Path("control/product_host/plugins") / f"{entry.name}.json",
                    entry=entry.value,
                )
            )
        for name, digest in _enabled(root).items():
            try:
                registration = _registration(
                    name, root, Path("plugins") / name / "plugin.json", expected_hash=digest
                )
                self._add(registration)
            except PluginRefused as error:
                self._refusals[name] = error

    @property
    def capabilities(self) -> tuple[str, ...]:
        """The names from the static manifests, before any worker is started."""
        return tuple(sorted(self._actions))

    def _add(self, registration: _Registration) -> None:
        capability = registration.name
        if capability in self._actions:
            raise PluginRefused("plugin_registry.capability_conflict", registration.name)
        self._actions[capability] = _Activation(registration)

    def _action(self, capability: str) -> _Activation:
        if capability in self._refusals:
            refusal = self._refusals[capability]
            raise PluginRefused(refusal.code, refusal.name)
        if capability in self._actions:
            return self._actions[capability]
        raise PluginRefused("plugin_registry.capability_unlisted", capability)

    def state(self, capability: str) -> PluginState:
        """Observe the process owner before reporting whether a loaded plugin is active."""
        action = self._action(capability)
        with action.lock:
            action.live_worker()
            return action.state

    def verify(self, capability: str) -> VerifiedPlugin:
        """Read the verified closure without seeking a workspace decision or loading a worker."""
        return self._action(capability).registration.verify()

    def activate(
        self,
        capability: str,
        load_in_worker: Callable[[VerifiedPlugin], PluginWorker],
        *,
        attended: bool = False,
    ) -> PluginWorker:
        """Load or reuse the runtime's worker after admitting its verified workspace effects."""
        action = self._action(capability)
        with action.lock:
            worker = action.live_worker()
            pinned = action.registration.expected_hash
            if worker is not None and pinned is None:
                return worker
            try:
                verified = action.registration.verify()
            except PluginRefused as error:
                if error.code == "plugin_registry.closure_changed" and pinned is not None:
                    self._decisions.lapse_plugin_confirmation(capability, pinned)
                raise
            if worker is None:
                action.state = PluginState.VERIFIED
            if pinned is not None:
                self._confirm(verified.spec, attended=attended)
            if worker is not None:
                return worker
            worker = load_in_worker(verified)
            if not worker.is_alive():
                raise PluginRefused("plugin_registry.worker_exited", verified.spec.plugin_id)
            action.worker = worker
            action.state = PluginState.ACTIVE
            return worker

    def _confirm(self, spec: ActionSpec, *, attended: bool) -> None:
        effects = tuple(
            name
            for name, declared in (
                ("result", spec.effect_mode is EffectMode.STAGING_WRITE),
                ("network", spec.lease.network),
                ("data_layer", spec.effect_mode is EffectMode.LEDGER_COMMIT),
            )
            if declared
        )
        allowed, decision = self._decisions.plugin_confirmation(
            spec.plugin_id, spec.code_role_hash, effects, attended=attended
        )
        if not allowed:
            raise PluginRefused(
                "plugin_registry.confirmation_required",
                spec.plugin_id,
                fingerprint=spec.code_role_hash,
                effects=effects,
                decision=decision,
            )


def _builtin_entries() -> metadata.EntryPoints:
    try:
        installed = metadata.distribution("alphalattice")
    except metadata.PackageNotFoundError:
        # The historical source runner installs the locked dependencies alone.
        return metadata.EntryPoints()
    return installed.entry_points.select(group=_GROUP)


def _file(root: Path, relative: Path) -> Path:
    path = root / relative
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError(relative)
    return path


def _enabled(workspace: Path) -> dict[str, str]:
    """A missing config enables nothing; the plugins directory is never enumerated."""
    path = Path("plugins.toml")
    try:
        content = _file(workspace, path).read_bytes()
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        raise PluginRefused("plugin_registry.configuration_invalid", path.name) from error
    try:
        settings = tomllib.loads(content.decode())
    except ValueError as error:
        raise PluginRefused("plugin_registry.configuration_invalid", path.name) from error
    enabled = settings.get("enabled", {})
    if settings.keys() - {"enabled"} or not isinstance(enabled, dict):
        raise PluginRefused("plugin_registry.configuration_invalid", path.name)
    for name, digest in enabled.items():
        if (
            not _NAME.fullmatch(name)
            or PureWindowsPath(name).is_reserved()
            or not is_blake3(digest)
        ):
            raise PluginRefused("plugin_registry.configuration_invalid", name)
    return enabled


def _registration(
    name: str,
    root: Path,
    path: Path,
    *,
    entry: str | None = None,
    expected_hash: str | None = None,
) -> _Registration:
    try:
        if not _NAME.fullmatch(name) or PureWindowsPath(name).is_reserved():
            raise ValueError
        body = _file(root, path).read_bytes()
        try:
            manifest = _Manifest.model_validate_json(body)
        except ValidationError as error:
            if any(
                item["loc"] == ("effect_mode",) and item["type"] == "enum"
                for item in error.errors(include_input=False)
            ):
                raise PluginRefused("plugin_registry.effect_unknown", name) from error
            raise
        target = entry or manifest.entry
        if target is None or not _ENTRY.fullmatch(target) or (entry and manifest.entry):
            raise ValueError
        # For workspace plugins the declared module and its helpers are relative to plugin.json.
        if (
            entry is None
            and target.partition(":")[0].replace(".", "/") + ".py" not in manifest.files
        ):
            raise ValueError
        for source in manifest.files:
            pure = PurePosixPath(source)
            if (
                pure.as_posix() != source
                or pure.is_absolute()
                or ".." in pure.parts
                or not pure.parts
            ):
                raise ValueError
            if any(PureWindowsPath(part).is_reserved() for part in pure.parts):
                raise ValueError
        folder, manifest_path = (root, path) if entry else (root / path.parent, Path(path.name))
        return _Registration(name, target, folder, manifest_path, manifest, body, expected_hash)
    except PluginRefused:
        raise
    except (OSError, ValueError) as error:
        raise PluginRefused("plugin_registry.manifest_invalid", name) from error
