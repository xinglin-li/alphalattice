"""Task Control owns a plugin run's staging folder and the Host's commit of what it staged.

A request is admitted first: the capability's effect is STAGING_WRITE, the invocation was keyed
for the capability's own code (`code_role_hash`), and its read snapshot and declared input
versions are one read of the catalog owner's. An admitted request whose key is already running
attaches to that run: it receives the same future, already running so no caller can cancel it,
which holds the first run's result or refusal, and starts nothing.

An admitted run writes only its own folder, `<workspace>/staging/plugin-runs/<invocation_id>/`,
beside the Tasks' scratch, staging its answer as `output.json` with any regular files beside it.
The folder is created and every staged entry read through the workspace's path confinement from
the workspace root, so no component from the workspace down is a link, reparse point or hard
link, and every entry is a regular file or folder. The Host then checks what it staged: the
receipt names this invocation and its key, and the answer meets the capability's output schema,
whose references resolve only inside it. It writes the run's provenance beside the answer, a name
reserved to it, and commits the folder only if every declared input's version still equals the
one in the run's key, under the workspace mutation gate, through the artifact store's one atomic
replace (`ArtifactResolver.publish_plugin_result`). The answer returned is the committed
artifact's, so every caller for a key sees one result. Every refusal is by name with its next
step, and the folder is gone after a commit or a refusal alike.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections.abc import Callable, Collection
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from jsonschema import exceptions as schema_errors
from jsonschema import validators
from referencing import Registry
from referencing import exceptions as reference_errors

from alphalattice.capabilities.domain_contracts.domain.errors import WorkspaceSecurityError
from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor, ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.paths import (
    make_confined_parents,
    resolve_confined,
)
from alphalattice.protocols.plugin_contract import (
    ActionInvocation,
    ActionReceipt,
    ActionSpec,
    CatalogRead,
    EffectMode,
    ExecutionContext,
    ReceiptStatus,
)

OUTPUT = "output.json"
"""The answer a plugin stages in its folder, checked against its capability's output schema."""

PROVENANCE = "provenance.json"
"""The invocation and its receipt, written by the Host beside the answer it commits."""

Plugin = Callable[[ExecutionContext], ActionReceipt]
"""One run of a capability's code in the context it is given; an executor supplies it."""


class StagingRefused(ValueError):
    """A staged run the Host does not commit, by name, with the step that recovers it."""

    def __init__(self, code: str, message: str) -> None:
        """Name the refusal and its next step.

        Args:
            code: The registered refusal code.
            message: The step that recovers it, which the caller is shown.
        """
        super().__init__(f"{code}: {message}")
        self.code = code
        self.next_step = message


@dataclass(frozen=True)
class StagedCommit:
    """A committed run: its key, the published folder's descriptor and the answer it staged."""

    idempotency_key: str
    artifact: ArtifactDescriptor
    output: object


class PluginStaging:
    """Run a staging capability in its own folder and commit what it staged, once per key."""

    def __init__(
        self,
        workspace: Path,
        data_root: Path,
        resolver: ArtifactResolver,
        gate: WorkspaceMutationGate,
        catalog: Callable[[Collection[str], str | None], CatalogRead],
    ) -> None:
        """Bind the staging folders, the catalog read and the one commit path.

        Args:
            workspace: The workspace whose staging folder holds each run's folder.
            data_root: The data root a run reads, read-only.
            resolver: The artifact store a staged run is published to.
            gate: The workspace's single-writer boundary the commit holds.
            catalog: The catalog owner's read of declared inputs at a snapshot it named, or the
                latest for None (`StorePort.catalog_read`).
        """
        self._workspace = workspace
        self._data_root = data_root
        self._resolver = resolver
        self._gate = gate
        self._catalog = catalog
        self._lock = Lock()
        self._flights: dict[str, Future[StagedCommit]] = {}

    def request(
        self, spec: ActionSpec, invocation: ActionInvocation, plugin: Plugin
    ) -> Future[StagedCommit]:
        """The run of `invocation`: started here, or the running one with the same key.

        The request is admitted first, by name. The first admitted request runs the plugin and
        returns its finished future; one made while that run is in flight returns the same
        future, running and so not cancellable, and runs nothing.
        """
        self._admit(spec, invocation)
        key = invocation.idempotency_key
        with self._lock:
            running = self._flights.get(key)
            if running is not None:
                return running
            flight: Future[StagedCommit] = Future()
            flight.set_running_or_notify_cancel()  # owned by this run before any caller sees it
            self._flights[key] = flight
        try:
            flight.set_result(self._run(spec, invocation, plugin))
        except Exception as error:  # the attached requests receive the same refusal
            flight.set_exception(error)
        finally:
            with self._lock:
                del self._flights[key]
        return flight

    def _admit(self, spec: ActionSpec, invocation: ActionInvocation) -> None:
        """Refuse by name a request this owner may not run or attach."""
        if spec.effect_mode is not EffectMode.STAGING_WRITE:
            raise StagingRefused(
                "task_control.plugin_not_staging_write",
                "run a capability that stages its output through its own effect's owner",
            )
        if (invocation.capability_id, invocation.code_role_hash) != (
            spec.capability_id,
            spec.code_role_hash,
        ):
            raise StagingRefused(
                "task_control.plugin_role_hash_mismatch",
                "invoke the capability's current code again",
            )
        read = self._catalog(spec.inputs, invocation.catalog_snapshot_id)
        if (read.snapshot_id, read.input_versions) != (
            invocation.catalog_snapshot_id,
            invocation.input_versions,
        ):
            raise StagingRefused(
                "task_control.plugin_read_unbound",
                "invoke the capability again from one catalog read",
            )

    def _run(self, spec: ActionSpec, invocation: ActionInvocation, plugin: Plugin) -> StagedCommit:
        invocation_id = uuid4()
        run = f"staging/plugin-runs/{invocation_id}"
        try:  # each folder from the workspace root down, checked as it is created
            folder = make_confined_parents(self._workspace, f"{run}/{OUTPUT}").parent
        except (OSError, WorkspaceSecurityError) as error:
            raise StagingRefused(
                "task_control.plugin_staging_unproved",
                "replace the link in the workspace's staging folder with a folder and rerun",
            ) from error
        try:
            context = ExecutionContext(
                invocation_id=invocation_id,
                catalog_snapshot_id=invocation.catalog_snapshot_id,
                readonly_data_root=self._data_root,
                staging_dir=folder,
                lease=spec.lease,
                is_speculative=invocation.is_speculative,
            )
            receipt = plugin(context)
            if (
                receipt.status is not ReceiptStatus.OK
                or receipt.invocation_id != context.invocation_id
                or receipt.idempotency_key != invocation.idempotency_key
            ):
                raise StagingRefused(
                    "task_control.plugin_provenance_mismatch", "rerun the invocation"
                )
            staged = _staged_files(self._workspace, run)
            if PROVENANCE in staged or OUTPUT not in staged:
                raise StagingRefused(
                    "task_control.plugin_staged_entry_invalid",
                    f"stage {OUTPUT} and never {PROVENANCE}, which the Host writes",
                )
            _checked_output(resolve_confined(self._workspace, f"{run}/{OUTPUT}"), spec)
            provenance = resolve_confined(self._workspace, f"{run}/{PROVENANCE}")
            with provenance.open("x", encoding="utf-8") as record:
                json.dump(
                    {
                        "invocation": invocation.model_dump(mode="json"),
                        "receipt": receipt.model_dump(mode="json"),
                    },
                    record,
                    sort_keys=True,
                )
            with self._gate.hold():
                if self._catalog(spec.inputs, None).input_versions != invocation.input_versions:
                    raise StagingRefused(
                        "task_control.plugin_inputs_changed",
                        "rerun the invocation against the current catalog",
                    )
                artifact = self._resolver.publish_plugin_result(folder, invocation.idempotency_key)
            committed = self._resolver.resolve_plugin_result(artifact) / OUTPUT
            output = json.loads(committed.read_text(encoding="utf-8"))
            return StagedCommit(invocation.idempotency_key, artifact, output)
        finally:
            if folder.exists():
                shutil.rmtree(folder)


def _staged_files(workspace: Path, run: str) -> set[str]:
    """Each regular file staged in the run folder `run` of `workspace`, as a path relative to it,
    each confined from the workspace root; a link, reparse point, hard-linked file or other
    entry, or a name the workspace's paths refuse, is refused by name."""
    folder = workspace / run
    found: set[str] = set()
    pending = [""]
    while pending:
        prefix = pending.pop()
        try:
            with os.scandir(folder / prefix if prefix else folder) as entries:
                names = sorted(entry.name for entry in entries)
            for name in names:
                relative = f"{prefix}{name}"
                mode = resolve_confined(workspace, f"{run}/{relative}").lstat().st_mode
                if stat.S_ISDIR(mode):
                    pending.append(f"{relative}/")
                elif stat.S_ISREG(mode):
                    found.add(relative)
                else:
                    raise WorkspaceSecurityError(f"staged entry is not a regular file: {relative}")
        except (OSError, WorkspaceSecurityError) as error:
            raise StagingRefused(
                "task_control.plugin_staged_entry_invalid",
                "stage only regular files and folders inside the run's folder",
            ) from error
    return found


def _checked_output(path: Path, spec: ActionSpec) -> None:
    """Refuse by name a staged answer that is not JSON meeting the output schema, whose references
    resolve only inside the schema itself: an empty registry retrieves nothing."""
    schema = spec.model_dump(mode="json")["output_schema"]
    try:
        output = json.loads(path.read_text(encoding="utf-8"))
        validators.validator_for(schema)(schema, registry=Registry()).validate(output)
    except (
        OSError,
        ValueError,
        schema_errors.ValidationError,
        reference_errors.Unresolvable,
    ) as error:
        raise StagingRefused(
            "task_control.plugin_output_invalid", "repair the plugin's output and rerun"
        ) from error
