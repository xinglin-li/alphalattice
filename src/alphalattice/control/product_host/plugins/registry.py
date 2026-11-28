"""Explicit Host capabilities. A workspace's `plugins.toml` has an `[enabled]`
table of capability names to tagged BLAKE3 hashes. Each name refers to
`plugins/<name>/plugin.json`, which declares an entry and its source files.
The workspace setting is read when the Host composes. Restart after changing it.
"""

from __future__ import annotations

import re
import sys
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module, metadata
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from threading import Lock
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Annotated, Final, Protocol
from uuid import uuid4, uuid7

from pydantic import JsonValue, StringConstraints, ValidationError

from alphalattice.capabilities.domain_contracts.domain.base import DomainModel, ShortString
from alphalattice.kernel.door import DECISIONS_DATABASE, Decision, Decisions
from alphalattice.kernel.identity import integrity_blake3_canonical, is_blake3
from alphalattice.protocols.plugin_contract import (
    ActionInvocation,
    ActionReceipt,
    ActionSpec,
    CatalogRead,
    EffectMode,
    ExecutionContext,
    ExecutionMode,
    Lease,
)
from alphalattice.protocols.plugin_contract.contracts import JsonObject

if TYPE_CHECKING:
    from datetime import date
    from importlib.machinery import ModuleSpec

    from alphalattice.control.task_control.plugin_staging import PluginStaging

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
class PluginListing:
    """One capability as the Host lists it: its static declaration and its worker state."""

    capability: str
    plugin: str
    effect_mode: EffectMode
    workspace: bool
    state: PluginState
    input_schema: Mapping[str, object]


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
    trusted_role: str | None = None
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
            package = _PACKAGE
            if entry.dist is not None:
                installed = entry.dist.locate_file("alphalattice")
                if isinstance(installed, Path) and installed.is_dir():
                    package = installed.resolve()
            self._add(
                _registration(
                    entry.name,
                    package,
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

    def listing(self) -> tuple[PluginListing, ...]:
        """Each capability as its static manifest declares it, with its state; no plugin code is
        read or imported and no worker started."""
        return tuple(
            PluginListing(
                capability=name,
                plugin=name,
                effect_mode=action.registration.manifest.effect_mode,
                workspace=action.registration.expected_hash is not None,
                state=self.state(name),
                input_schema=action.registration.manifest.input_schema,
            )
            for name, action in sorted(self._actions.items())
        )

    def invoke(
        self,
        capability: str,
        params: Mapping[str, JsonValue],
        *,
        catalog: Callable[[tuple[str, ...]], CatalogRead],
        as_of: date,
        attended: bool,
        staging: Callable[[], PluginStaging],
    ) -> ActionReceipt:
        """Run one capability on request and answer its receipt.

        Refused by name before any worker starts: a capability not listed, a ledger write, a
        workspace plugin (`admit_workspace`) and parameters its input schema does not admit, a
        property it does not declare included unless the schema says otherwise. A built-in then
        runs in a kept worker; a staging write commits through the Host's staging owner.
        """
        from alphalattice.control.task_control.plugin_staging import (  # lazy: a run only
            StagedCommit,
            check_plugin_input,
        )

        action = self._action(capability)
        if action.registration.manifest.effect_mode is EffectMode.LEDGER_COMMIT:
            raise PluginRefused("plugin_registry.ledger_commit_unavailable", capability)
        self.admit_workspace(capability, attended=attended)
        spec = self.trusted_spec(capability)
        invocation = spec.invoke(params, catalog(spec.inputs), as_of=as_of)
        schema = dict(spec.input_schema)
        schema.setdefault("additionalProperties", False)
        check_plugin_input(invocation, spec.model_copy(update={"input_schema": schema}))
        if spec.effect_mode is EffectMode.STAGING_WRITE:
            commit = self.run_trusted(invocation, staging=staging())
            assert isinstance(commit, StagedCommit)
            receipt: ActionReceipt = commit.receipt
            return receipt
        began = time.perf_counter()
        output = self.run_trusted(invocation)
        assert isinstance(output, dict)
        return invocation.answered(output, wall_ms=round((time.perf_counter() - began) * 1000))

    def admit_workspace(self, capability: str, *, attended: bool) -> None:
        """Refuse a workspace plugin by name: unconfirmed, its fingerprint's confirmation is
        required (asked when attended); confirmed, it waits for its guarded child process."""
        action = self._action(capability)
        if action.registration.expected_hash is None:
            return
        with action.lock:
            self._confirm(self._verify(action).spec, attended=attended)
        raise PluginRefused("plugin_registry.workspace_run_unavailable", capability)

    def trusted_spec(self, capability: str) -> ActionSpec:
        """Admit a built-in closure for invocation keying, without starting a worker."""
        action = self._action(capability)
        with action.lock:
            return self._verify_trusted(action).spec

    def run_trusted(
        self,
        invocation: ActionInvocation,
        *,
        staging: PluginStaging | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> object:
        """Run a verified built-in, committing staged effects through the Host's staging owner."""
        if not isinstance(invocation, ActionInvocation):
            name = getattr(invocation, "__name__", type(invocation).__name__)
            raise PluginRefused("plugin_registry.dispatch_unverified", name)
        name = invocation.capability_id
        action = self._action(name)
        with action.lock:
            verified = self._verify_trusted(action)
            if verified.spec.code_role_hash != invocation.code_role_hash:
                raise PluginRefused("plugin_registry.closure_changed", name)
        if verified.spec.effect_mode is EffectMode.STAGING_WRITE and staging is not None:

            @contextmanager
            def stage(context: ExecutionContext) -> Iterator[ActionReceipt]:
                with self._run_worker(action, verified, invocation, context, cancelled) as receipt:
                    assert receipt is not None
                    yield receipt

            return staging.request(
                verified.spec,
                invocation,
                stage,
            ).result(timeout=verified.spec.lease.timeout_ms / 1000)
        if verified.spec.effect_mode is not EffectMode.READ_ONLY:
            raise PluginRefused("plugin_registry.dispatch_unverified", name)

        from alphalattice.control.task_control.plugin_staging import (  # lazy: read verified IPC
            ARROW_OUTPUT,
            check_plugin_input,
            checked_plugin_output,
        )
        from alphalattice.control.task_control.scratch import (  # lazy: own this read's scratch
            task_scratch_directory,
        )
        from alphalattice.foundation.data_layer.root import data_root  # lazy: bind the read root

        check_plugin_input(invocation, verified.spec)
        with TemporaryDirectory(prefix="plugin-read-", dir=task_scratch_directory()) as folder:
            context = ExecutionContext(
                invocation_id=uuid4(),
                catalog_snapshot_id=invocation.catalog_snapshot_id,
                readonly_data_root=data_root(),
                staging_dir=Path(folder),
                lease=verified.spec.lease,
                is_speculative=invocation.is_speculative,
            )
            with self._run_worker(action, verified, invocation, context, cancelled):
                return checked_plugin_output(context.staging_dir / ARROW_OUTPUT, verified.spec)

    @staticmethod
    def _verify_trusted(action: _Activation) -> VerifiedPlugin:
        registration = action.registration
        if registration.expected_hash is not None:
            raise PluginRefused("plugin_registry.dispatch_unverified", registration.name)
        verified = registration.verify()
        module = verified.entry.partition(":")[0]
        if not module.startswith("alphalattice.") or _source_path(module) not in verified.sources:
            raise PluginRefused("plugin_registry.dispatch_unverified", registration.name)
        if action.trusted_role is not None and action.trusted_role != verified.spec.code_role_hash:
            raise PluginRefused("plugin_registry.closure_changed", registration.name)
        action.trusted_role = verified.spec.code_role_hash
        if action.state is PluginState.DISCOVERED:
            action.state = PluginState.VERIFIED
        return verified

    @staticmethod
    @contextmanager
    def _run_worker(
        action: _Activation,
        verified: VerifiedPlugin,
        invocation: ActionInvocation,
        context: ExecutionContext,
        cancelled: Callable[[], bool] | None,
    ) -> Iterator[ActionReceipt | None]:
        from alphalattice.control.task_control.child import KeptWorker  # lazy: actual dispatch

        with action.lock:
            worker = action.live_worker()
            if worker is None:
                worker = action.worker = KeptWorker()
            if not isinstance(worker, KeptWorker):
                raise PluginRefused("plugin_registry.dispatch_unverified", invocation.capability_id)
            with worker.calling(
                f"{__name__}:_invoke_verified",
                {
                    "entry": verified.entry,
                    "package": action.registration.folder,
                    "sources": tuple(verified.sources.items()),
                    "context_json": context.model_dump_json(),
                    "invocation_json": invocation.model_dump_json(),
                    "staged": verified.spec.effect_mode is EffectMode.STAGING_WRITE,
                },
                cancelled=cancelled,
            ) as receipt:
                action.state = PluginState.ACTIVE
                yield None if receipt is None else ActionReceipt.model_validate_json(receipt)

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
            verified = self._verify(action)
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

    def _verify(self, action: _Activation) -> VerifiedPlugin:
        """The verified closure; a pinned plugin whose closure changed lapses its approval."""
        try:
            return action.registration.verify()
        except PluginRefused as error:
            pinned = action.registration.expected_hash
            if error.code == "plugin_registry.closure_changed" and pinned is not None:
                self._decisions.lapse_plugin_confirmation(action.registration.name, pinned)
            raise

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


def _source_path(module: str) -> str:
    """Locate a declared module within the installed alphalattice package."""
    return module.removeprefix("alphalattice.").replace(".", "/") + ".py"


class _VerifiedSources(Loader, MetaPathFinder):
    """Use the registry's frozen bytes for declared modules, including lazy imports."""

    def __init__(self, package: Path, sources: tuple[tuple[str, bytes], ...]) -> None:
        self.package = package
        self.sources = sources
        self.modules = {
            "alphalattice." + path.removesuffix(".py").replace("/", "."): (path, source)
            for path, source in sources
            if path.endswith(".py")
        }

    def find_spec(
        self, fullname: str, path: object = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname not in self.modules:
            return None
        origin = str(self.package / self.modules[fullname][0])
        return spec_from_loader(fullname, self, origin=origin)

    def exec_module(self, module: ModuleType) -> None:
        path, source = self.modules[module.__name__]
        module.__file__ = str(self.package / path)
        exec(compile(source, module.__file__, "exec"), module.__dict__)


def _invoke_verified(
    *,
    entry: str,
    package: Path,
    sources: tuple[tuple[str, bytes], ...],
    context_json: str,
    invocation_json: str,
    staged: bool,
    cancelled: Callable[[], bool],
) -> str | None:
    """Run the named entry and return only its small control receipt through the pipe."""
    from alphalattice.control.task_control.plugin_staging import (  # lazy: write in the worker
        ARROW_OUTPUT,
        write_arrow_output,
    )
    from alphalattice.kernel.execution import CANCELLED, ExecutionStopped  # lazy: worker stop

    if cancelled():
        raise ExecutionStopped(CANCELLED)
    context = ExecutionContext.model_validate_json(context_json)
    invocation = ActionInvocation.model_validate_json(invocation_json)
    module, _, function = entry.partition(":")
    if module not in sys.modules:
        loader = _VerifiedSources(package, sources)
    else:
        existing_loader = getattr(sys.modules[module], "__loader__", None)
        if (
            not isinstance(existing_loader, _VerifiedSources)
            or existing_loader.package != package
            or existing_loader.sources != sources
        ):
            raise PluginRefused("plugin_registry.closure_changed", invocation.capability_id)
        loader = existing_loader
    if any(
        name in sys.modules and getattr(sys.modules[name], "__loader__", None) is not loader
        for name in loader.modules
    ):
        raise PluginRefused("plugin_registry.closure_changed", invocation.capability_id)
    sys.meta_path.insert(0, loader)
    try:
        call = getattr(import_module(module), function)
        output = context.staging_dir / ARROW_OUTPUT
        if staged:
            receipt: ActionReceipt = call(context, invocation)
            if receipt.output is not None:
                write_arrow_output(output, receipt.model_dump(mode="json")["output"])
            control_receipt: str = receipt.model_dump_json(exclude={"output"})
            return control_receipt
        write_arrow_output(output, call(invocation.params))
        return None
    finally:
        sys.meta_path.remove(loader)


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
