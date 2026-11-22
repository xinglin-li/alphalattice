"""Admit a local Factor input closure and prepare its existing authoring owners."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal, Self, cast
from uuid import UUID

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    current_allowed_logical_processors,
)
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    ResearchWorkspaceManifest,
    read_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.product_host.storage.retention import require_available_input
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.control.task_control.upgrade import RETIRED_DIRECTORY, set_aside
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.content_store import staging_directory
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.foundation.causal_outcomes.execution.methods import ONE_SESSION_RECIPE_ID
from alphalattice.foundation.causal_outcomes.execution.publication import (
    CausalExecutionOutcomePublisher,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.causal_outcomes.selection.listing_set import (
    resolve_published_execution_outcome,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    FACTOR_EXPERIMENT_KIND,
    INSTALLED_REDUNDANCY_POLICIES,
    INSTALLED_SCREENING_POLICIES,
    FactorExperimentCompiler,
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.factor_research.experiments.execution import (
    FactorExperimentExecutor,
    factor_policies_at_scale,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_files import FolderFiles
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.panels.logical_identity import (
    PanelLogicalArtifactStore,
    PanelLogicalIdentityPublisher,
)
from alphalattice.foundation.feature_engine.panels.observation_clock_authority import (
    FeaturePanelObservationClockVerifier,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.sources.universe import (
    bootstrap_from_candidate_manifest_document,
)
from alphalattice.foundation.market_data_ops.storage import world_reads
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import history, inputs
from alphalattice.kernel.data_layer.backup import (
    backup_data_root,
    generation_reads,
    verify_data_root,
)
from alphalattice.kernel.data_layer.inputs import folder_files
from alphalattice.kernel.data_layer.upgrade import Family
from alphalattice.kernel.identity import integrity_sha256
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
)

PROFILE = "us-current-index-research"


def _windows_long_path(path: Path) -> Path:
    """Use Windows extended paths without changing the workspace's recorded path."""
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def _reported_path(path: Path) -> Path:
    """Return the ordinary Windows spelling of a path used for file I/O."""
    value = str(path)
    if os.name == "nt" and value.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + value[8:])
    if os.name == "nt" and value.startswith("\\\\?\\"):
        return Path(value[4:])
    return path


def confined(root: Path, relative: str) -> Path:
    """Resolve an explicit relative path and refuse workspace escape.

    Args:
        root: Explicit admitted confinement root.
        relative: Declared relative path.

    Returns:
        Resolved path beneath the root.

    Raises:
        AuthoringError: Path is absolute or resolves outside the workspace.
    """
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise AuthoringError("research_experiment.path_escapes_workspace")
    return path


def _digest_stream(stream: BinaryIO) -> str:
    """The content digest of one open file object, read whole from its start."""

    digest = integrity_sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    """Stream the exact file bytes into their SHA-256 identity.

    Args:
        path: Explicit file to read.

    Returns:
        Lowercase SHA-256 digest.
    """
    with path.open("rb") as stream:
        return _digest_stream(stream)


_DIGEST_WORKERS = 4


def _file_object(stream: BinaryIO) -> tuple[int, int] | None:
    """The identity of the file object behind an open handle, or None where none is reported.

    Read off the handle, never off the path: a path can be pointed at another
    object at any time, an open handle cannot. Identity says two handles read
    the same bytes; it is never a substitute for reading them.
    """

    stat = os.fstat(stream.fileno())
    return (stat.st_dev, stat.st_ino) if stat.st_ino else None


def file_digests(paths: Iterable[Path]) -> dict[Path, str]:
    """Digest several files at once; every file object is read whole, by the same digest.

    Every path is opened first and held open until the call returns; the
    grouping, the reads and the check below all concern those open objects,
    and nothing outlives the call. Two handles that report one file object --
    the pooled DuckDB object and the bundle's link to it -- are read once
    and answer the same digest, because one object holds one set of bytes
    while it is held open; a platform that reports no object identity reads
    every path itself. Hashing releases the interpreter lock, so the objects
    of one verification are read in parallel.

    The boundary of what a returned digest states: at the end of the call
    every path still names the object whose bytes were read. A path that
    then names another object (replaced beneath the call) or no object
    (removed) is refused by name rather than answered with another object's
    digest; a caller wanting the new object verifies again.
    """
    ordered = tuple(dict.fromkeys(paths))
    with ExitStack() as stack:
        streams: dict[Path, BinaryIO] = {}
        for path in ordered:
            try:
                streams[path] = stack.enter_context(path.open("rb"))
            except FileNotFoundError as error:
                raise AuthoringError("research_experiment.input_file_unavailable") from error
        objects = {path: _file_object(stream) for path, stream in streams.items()}
        representative: dict[tuple[int, int], Path] = {}
        first_of: dict[Path, Path] = {}
        for path in ordered:
            identity = objects[path]
            first_of[path] = path if identity is None else representative.setdefault(identity, path)
        distinct = tuple(dict.fromkeys(first_of.values()))
        if len(distinct) <= 1:
            digests = {path: _digest_stream(streams[path]) for path in distinct}
        else:
            with ThreadPoolExecutor(max_workers=min(_DIGEST_WORKERS, len(distinct))) as pool:
                digests = dict(
                    zip(
                        distinct,
                        pool.map(lambda path: _digest_stream(streams[path]), distinct),
                        strict=True,
                    )
                )
        for path in ordered:
            try:
                current = path.stat()
            except FileNotFoundError as error:
                raise AuthoringError(
                    "research_experiment.input_file_removed_during_verification"
                ) from error
            if objects[path] is not None and (current.st_dev, current.st_ino) != objects[path]:
                raise AuthoringError("research_experiment.input_file_replaced_during_verification")
    return {path: digests[first_of[path]] for path in ordered}


def _closed_files(root: Path) -> set[str]:
    """No unadmitted WAL, sidecar or link may change what an owner opens."""
    pending = [root]
    found: set[str] = set()
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_symlink() or path.is_junction():
                raise AuthoringError("research_experiment.input_link_not_admitted")
            if path.is_dir():
                pending.append(path)
            elif path.name != ".alphalattice-writer.lock":
                found.add(path.relative_to(root).as_posix())
    return found


class FactorInputBundle(BaseModel):  # type: ignore[misc]
    """Manifest-last local bytes, not new scientific authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sessions: tuple[date, ...]
    files: tuple[tuple[str, str], ...]
    database_snapshot_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    # A 0.2.7 file-bound bundle's manifest carries no binding version.
    binding_version: Literal[2] | None = Field(default=None, exclude_if=lambda v: v is None)
    outcomes: tuple[tuple[str, str], ...] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal the exact Factor input bundle fields and file/session axes.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical binding_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = cls.model_construct(**values, binding_hash="0" * 64)
        return cls(
            **values,
            binding_hash=canonical_hash(draft.model_dump(mode="json", exclude={"binding_hash"})),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require ordered unique nonempty sessions/files and exact bundle/database bindings.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AuthoringError: Session/file axis, canonical identity or the declared database digest
                differs.
        """
        if (
            not self.sessions
            or self.sessions != tuple(sorted(set(self.sessions)))
            or not self.files
            or self.files != tuple(sorted(set(self.files)))
            or len({p for p, _ in self.files}) != len(self.files)
            or self.binding_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"binding_hash"}))
        ):
            raise AuthoringError("research_experiment.input_manifest_invalid")
        if (self.binding_version is None) != (self.outcomes is None):
            raise AuthoringError("research_experiment.input_manifest_invalid")
        if self.outcomes is not None and (
            not self.outcomes
            or self.outcomes != tuple(sorted(set(self.outcomes)))
            or len({artifact for artifact, _ in self.outcomes}) != len(self.outcomes)
            or any(
                not re.fullmatch(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                    artifact,
                )
                or len(content) != 64
                or any(character not in "0123456789abcdef" for character in content)
                for artifact, content in self.outcomes
            )
        ):
            raise AuthoringError("research_experiment.input_manifest_invalid")
        if (
            self.database_snapshot_hash is not None
            and dict(self.files).get("market-data.duckdb") != self.database_snapshot_hash
        ):
            raise AuthoringError("research_experiment.input_database_binding_mismatch")
        return self


def input_absent(error: BaseException) -> bool:
    """Whether a read failed because the binding, or a file of it, is not there: absent from the
    workspace's copies, or not kept on this data root (`data_layer.input_missing`)."""
    return isinstance(error, FileNotFoundError) or (
        isinstance(error, DataError) and error.failure.code == "data_layer.input_missing"
    )


def read_factor_bundle(
    workspace: Path, binding_hash: str, *, verify: bool = True
) -> FactorInputBundle:
    """Reopen an exact admitted bundle and optionally verify its complete closed byte set.

    Args:
        workspace: Caller-owned admitted workspace.
        binding_hash: Exact lowercase bundle identity.
        verify: Whether to require availability, closed files and every recorded file/database
            digest.

    Returns:
        Identity-validated bundle; verify=False reads metadata without whole-file verification.

    Raises:
        AuthoringError: Identity/path, availability, closed file set or exact file/database bytes
            differ.
    """
    workspace = _windows_long_path(workspace)
    if len(binding_hash) != 64 or any(v not in "0123456789abcdef" for v in binding_hash):
        raise AuthoringError("research_experiment.input_hash_invalid")
    inputs = _inputs_root(workspace, binding_hash)
    root = confined(inputs, binding_hash)
    manifest = confined(root, "manifest.json")
    try:
        manifest_bytes = manifest.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            error.errno, error.strerror, str(_reported_path(manifest))
        ) from error
    bundle = FactorInputBundle.model_validate_json(manifest_bytes)
    if bundle.binding_hash != binding_hash:
        raise AuthoringError("research_experiment.input_manifest_identity_mismatch")
    if verify:
        source = confined(root, "source")
        for name, _expected in bundle.files:
            target = confined(source, name)
            if not target.exists():
                require_available_input(workspace, _released_name(inputs, target))
        if _closed_files(source) != {name for name, _ in bundle.files}:
            raise AuthoringError("research_experiment.input_file_set_changed")
        database: Path | None = None
        if bundle.database_snapshot_hash is not None:
            database = confined(
                inputs, f"databases/{bundle.database_snapshot_hash}/market-data.duckdb"
            )
            if not database.is_file():
                require_available_input(workspace, _released_name(inputs, database))
                raise AuthoringError("research_experiment.input_database_unavailable")
        targets = {name: confined(source, name) for name, _expected in bundle.files}
        digests = file_digests((*targets.values(), *((database,) if database else ())))
        for name, expected in bundle.files:
            if digests[targets[name]] != expected:
                raise AuthoringError("research_experiment.input_file_tampered")
        if database is not None and digests[database] != bundle.database_snapshot_hash:
            raise AuthoringError("research_experiment.input_database_tampered")
    return cast(FactorInputBundle, bundle)


def factor_input_paths(workspace: Path, binding_hash: str) -> tuple[Path, Path]:
    """Resolve legacy inline and referenced snapshots through one path owner."""
    workspace = _windows_long_path(workspace)
    if len(binding_hash) != 64 or any(v not in "0123456789abcdef" for v in binding_hash):
        raise AuthoringError("research_experiment.input_hash_invalid")
    source = confined(_inputs_root(workspace, binding_hash), f"{binding_hash}/source")
    # Referenced bundles expose a local alias of a sealed DB object. No alias
    # ever points at the mutable market database or another workspace.
    return source, source / "artifacts"


def _released_name(inputs: Path, path: Path) -> str:
    """The name a 0.2.7 cleanup's receipt gives a released input file: its workspace path under
    `research-inputs/`, whichever store the binding is read from now."""
    return "research-inputs/" + path.relative_to(inputs.resolve()).as_posix()


def _inputs_root(workspace: Path, binding_hash: str) -> Path:
    """The folder laid out as `research-inputs/` that a binding is read from: its kept view on the
    data root (kernel.data_layer.inputs)."""
    root: Path = history.research_input_root(binding_hash, workspace)
    return root


def _copy_input(workspace: Path, source: Path, target: Path) -> None:
    """Stage one input file: a workspace-owned immutable Parquet linked where the volume allows,
    anything else (the mutable database first) copied. The binding is then kept on the data root,
    which keeps equal bytes once (kernel.data_layer.inputs), so the workspace pools nothing."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".parquet" and source.resolve().is_relative_to(workspace.resolve()):
        try:
            os.link(source, target)
            return
        except OSError:  # another volume, or no links: a copy
            pass
    require_storage_capacity(workspace, additional_bytes=source.stat().st_size)
    shutil.copy2(source, target)


def logical_panel_owner(source: Path) -> PanelLogicalIdentityPublisher:
    """Compose the existing logical Panel identity publisher beneath an exact source root.

    Args:
        source: Explicit retained source workspace.

    Returns:
        Panel logical identity owner with its deterministic artifact store.
    """
    resolver = ArtifactResolver(source / "artifacts")
    return PanelLogicalIdentityPublisher(
        resolver=resolver, store=PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver))
    )


def _logical_files(source: Path, snapshot: str) -> tuple[Path, ...]:
    """Copy only this verified snapshot's closure, never its current pointer."""
    closure = PanelClosureArtifactStore(ArtifactResolver(source / "artifacts"))
    store = PanelLogicalArtifactStore(closure)
    marker = store.marker_for_snapshot(snapshot)
    if marker is None:
        return ()
    logical_panel_owner(source).verify_snapshot(snapshot)
    if not isinstance(closure.files, FolderFiles):
        return ()  # kept on the log: the staged copy's publisher writes this snapshot's closure
    refs = {
        "publication-markers": marker.marker_hash,
        "revisions": marker.logical_panel_hash,
        "semantic-indexes": marker.logical_semantic_index_hash,
        "derivation-bindings": marker.derivation_binding_hash,
        "physical-materializations": marker.physical_materialization_hash,
    }
    revision = closure.load_json(
        category="logical/revisions", content_hash=marker.logical_panel_hash
    )
    refs["content-manifests"] = str(revision["logical_manifest_hash"])
    paths = [
        closure.root / "logical" / kind / f"{identity}.json" for kind, identity in refs.items()
    ]
    paths.append(closure.root / "logical/operational/snapshot-mappings" / f"{snapshot}.json")
    return tuple(confined(source, p.relative_to(source).as_posix()) for p in paths)


def compatible_factor_inputs(
    original: FactorInputBundle,
    candidate: FactorInputBundle,
    *,
    read_legacy: Callable[[str], Any] | None = None,
    read_catalog: Callable[[str], Any] | None = None,
) -> bool:
    """A mixed handoff is compatible only after the old and catalog rows compare equal."""

    def outcome_file(name: str) -> bool:
        return (
            name.startswith("artifacts/data-operations/execution-outcomes/")
            and "/chunks/" in name
            and name.endswith(".parquet")
        )

    if not (
        original.panel_snapshot_hash == candidate.panel_snapshot_hash
        and original.outcome_snapshot_hash == candidate.outcome_snapshot_hash
        and original.sessions == candidate.sessions
        and dict(original.files).get("market-data.duckdb", original.database_snapshot_hash)
        == dict(candidate.files).get("market-data.duckdb", candidate.database_snapshot_hash)
    ):
        return False
    if original.outcomes is None and candidate.outcomes is None:
        return {(p, h) for p, h in original.files if p != "market-data.duckdb"}.issubset(
            {(p, h) for p, h in candidate.files if p != "market-data.duckdb"}
        )
    if not {
        (p, h) for p, h in original.files if p != "market-data.duckdb" and not outcome_file(p)
    }.issubset(
        {(p, h) for p, h in candidate.files if p != "market-data.duckdb" and not outcome_file(p)}
    ):
        return False
    if original.outcomes is not None and candidate.outcomes is not None:
        return original.outcomes == candidate.outcomes
    if read_legacy is None or read_catalog is None:
        return False
    legacy = original if original.outcomes is None else candidate
    current = candidate if original.outcomes is None else original
    files = {Path(name).stem: name for name, _ in legacy.files if outcome_file(name)}
    artifacts = {digest: artifact for artifact, digest in current.outcomes or ()}
    return (
        bool(files)
        and files.keys() == artifacts.keys()
        and all(
            (old := read_legacy(files[digest])).equals(
                read_catalog(artifacts[digest]).cast(old.schema)
            )
            for digest in files
        )
    )


def bind_factor_inputs(
    *,
    workspace: Path,
    source: Path,
    input_id: str = "factor-development",
    panel_snapshot_hash: str | None = None,
    outcome_snapshot_hash: str | None = None,
    include_alpha_handoff: bool = False,
) -> ResearchWorkspaceExperimentInput:
    """Copy only the selected public-development input graph while both writers are stopped."""
    workspace, source = workspace.resolve(), source.resolve()
    with ExitStack() as stack:
        for root in sorted({workspace, source}):
            lease = WorkspaceWriterLease.acquire(root)
            stack.callback(lease.close)
        return _bind_factor_inputs(
            workspace=_windows_long_path(workspace),
            source=_windows_long_path(source),
            input_id=input_id,
            panel_snapshot_hash=panel_snapshot_hash,
            outcome_snapshot_hash=outcome_snapshot_hash,
            include_alpha_handoff=include_alpha_handoff,
            # Both writers' leases are held, so this process is the only writer.
            gate=WorkspaceMutationGate(),
        )


def publish_prepared_factor_inputs(
    session: WorkspaceApplicationSession,
    task_id: UUID,
    *,
    input_id: str,
    panel_snapshot_hash: str,
    completed_at: datetime,
    bind_configuration: bool,
    claim: Callable[[ResearchWorkspaceManifest], None] | None = None,
) -> ResearchWorkspaceExperimentInput:
    """One Outcome/logical-Panel/bundle handoff for first use and later capture; `claim` records
    the Host's ownership of the manifest that binds it, under the gate before it is written."""
    task = session.task_control_registry.task(task_id)
    expected_kind = "workspace_preparation" if bind_configuration else "research_input_capture"
    if not session.writer_lease.held or task.task_kind != expected_kind:
        raise AuthoringError("research_experiment.preparation_writer_required")
    root = session.workspace
    manifest_ref = ArtifactResolver.feature_panel_manifest_uri(panel_snapshot_hash)
    outcome = CausalExecutionOutcomePublisher(
        store=MarketDataRepository(root),
        resolver=ArtifactResolver(root / "artifacts"),
        artifact_root=root / "artifacts",
        mutation_gate=session.mutation_gate,
        # The processors this Host is allowed, never more: a bound job or
        # the half-machine rule narrows the derivation's workers with it;
        # the publisher applies its own ceiling under that allowance.
        derivation_workers=len(current_allowed_logical_processors()) or 1,
    ).publish_daily(
        panel_manifest_ref=manifest_ref,
        completed_at=completed_at,
        recipe_id=ONE_SESSION_RECIPE_ID,
    )
    logical_panel_owner(root).publish_snapshot(manifest_ref)
    return _bind_factor_inputs(
        workspace=_windows_long_path(root),
        source=_windows_long_path(root),
        input_id=input_id,
        panel_snapshot_hash=panel_snapshot_hash,
        outcome_snapshot_hash=outcome.manifest.snapshot_hash,
        include_alpha_handoff=True,
        gate=session.mutation_gate,
        bind_configuration=bind_configuration,
        claim=claim,
    )


def _bind_factor_inputs(
    *,
    workspace: Path,
    source: Path,
    input_id: str,
    panel_snapshot_hash: str | None,
    outcome_snapshot_hash: str | None,
    include_alpha_handoff: bool,
    gate: WorkspaceMutationGate,
    bind_configuration: bool = True,
    claim: Callable[[ResearchWorkspaceManifest], None] | None = None,
) -> ResearchWorkspaceExperimentInput:
    manifest = read_research_workspace_manifest(workspace)
    if not confined(source, "market-data.duckdb").is_file():
        raise AuthoringError("research_experiment.source_database_absent")
    if (source / "market-data.duckdb.wal").exists():
        raise AuthoringError("research_experiment.source_database_not_quiescent")
    market = MarketDataRepository(source)
    if panel_snapshot_hash is None:
        snapshot = PanelStateRepository(
            market.database, market_data=market
        ).feature_panel_snapshot_for_active(PROFILE)
        if snapshot is None:
            raise AuthoringError("research_experiment.panel_absent")
        panel_snapshot_hash = str(snapshot["snapshot_hash"])
    artifacts = source / "artifacts"
    resolver = ArtifactResolver(artifacts)
    confined(
        source,
        str(resolver._feature_panel_manifest_path(panel_snapshot_hash).relative_to(source)),
    )
    panel = resolver.load_feature_panel_manifest(
        resolver.feature_panel_manifest_uri(panel_snapshot_hash)
    )
    confined(source, "artifacts/feature-panel/semantic-index")
    for path in (artifacts / "feature-panel/semantic-index").glob("*.json"):
        confined(source, str(path.relative_to(source)))
    for chunk in panel["chunks"]:
        confined(
            source,
            str(resolver._feature_panel_chunk_path(chunk["chunk_hash"]).relative_to(source)),
        )
    verified = FeaturePanelObservationClockVerifier(resolver=resolver).verify(panel_snapshot_hash)
    if not verified.verified:
        raise AuthoringError(str(verified.failure_code))
    found = resolver.find_feature_panel_semantic_index(panel_snapshot_hash=panel_snapshot_hash)
    if found is None:
        raise AuthoringError("research_experiment.semantic_index_absent")
    index, index_uri = found
    sessions = tuple(
        date.fromisoformat(str(r["session_date"]))
        for r in cast(list[dict[str, Any]], index["sessions"])
    )
    for category in ("manifests", "markers", "method-bindings", "method-seal-markers"):
        directory = confined(source, f"artifacts/data-operations/execution-outcomes/{category}")
        for path in directory.glob("*.json"):
            confined(source, str(path.relative_to(source)))
    outcome = resolve_published_execution_outcome(
        artifact_root=artifacts,
        listing_set_hash=str(panel["listing_set_hash"]),
        snapshot_handle=outcome_snapshot_hash,
    )
    reader = CausalExecutionOutcomeDevelopmentReader(artifacts)
    seal = reader.resolve_method_seal(outcome.snapshot_hash)
    if (
        seal.disposition != "METHOD_BOUND"
        or seal.binding is None
        or seal.seal_marker is None
        or seal.outcome_marker is None
    ):
        raise AuthoringError("research_experiment.outcome_method_not_admitted")
    paths = [
        source / "market-data.duckdb",
        resolver._feature_panel_manifest_path(panel_snapshot_hash),
        resolver._feature_panel_semantic_index_path(index_uri.rsplit("/", 1)[-1]),
    ]
    confined(source, str(resolver._feature_panel_lifecycle_path().relative_to(source)))
    resolver.feature_panel_snapshot_lifecycle(
        resolver.feature_panel_manifest_uri(panel_snapshot_hash)
    )
    paths.append(resolver._feature_panel_lifecycle_path())
    for chunk in panel["chunks"]:
        paths.append(
            resolver.resolve_feature_panel_chunk_ref(
                uri=chunk["uri"],
                content_hash=chunk["chunk_hash"],
                metadata_hash=chunk["metadata_hash"],
            )
        )
    store = reader.artifacts
    outcome_refs = []
    for chunk in outcome.development_chunks:
        if store.read_chunk(chunk, columns=["listing_id"]).num_rows != chunk.row_count:
            raise AuthoringError("research_experiment.outcome_body_incomplete")
        outcome_refs.append(
            (history.execution_outcome_artifact_id(chunk.content_hash), chunk.content_hash)
        )
    paths.extend(
        (
            store._json_path("manifests", outcome.snapshot_hash),
            store._json_path("markers", seal.outcome_marker.marker_hash),
            store._json_path("method-bindings", seal.binding.binding_hash),
            store._json_path("method-seal-markers", seal.seal_marker.seal_marker_hash),
        )
    )
    if include_alpha_handoff:
        paths.extend(_logical_files(source, panel_snapshot_hash))
    source_digests = file_digests(paths)
    files = tuple(
        sorted({(p.resolve().relative_to(source).as_posix(), source_digests[p]) for p in paths})
    )
    bundle = FactorInputBundle.create(
        panel_snapshot_hash=panel_snapshot_hash,
        outcome_snapshot_hash=outcome.snapshot_hash,
        sessions=tuple(sorted(set(sessions))),
        files=files,
        database_snapshot_hash=dict(files)["market-data.duckdb"],
        binding_version=2,
        outcomes=tuple(sorted(outcome_refs)),
    )
    # The explicitly named installed input is the idempotency anchor, not a
    # search for a latest compatible directory. Verify it before reuse.
    if include_alpha_handoff:
        prior = next((v for v in manifest.experiment_inputs or () if v.input_id == input_id), None)
        if prior is not None:
            candidate = read_factor_bundle(workspace, prior.binding_hash)
            prior_source = factor_input_paths(workspace, prior.binding_hash)[0]

            def legacy_rows(name: str) -> Any:
                return pq.read_table(confined(prior_source, name))

            def catalog_rows(artifact: str) -> Any:
                digests = dict(bundle.outcomes or candidate.outcomes or ())
                return history.read_execution_outcome(digests[artifact])

            if compatible_factor_inputs(
                bundle,
                candidate,
                read_legacy=legacy_rows,
                read_catalog=catalog_rows,
            ) and _logical_files(prior_source, panel_snapshot_hash):
                bundle = candidate
    binding = ResearchWorkspaceExperimentInput(input_id=input_id, binding_hash=bundle.binding_hash)
    parent = confined(workspace, "research-inputs")  # the binding's staging place only
    parent.mkdir(parents=True, exist_ok=True)
    kept = history.research_input_kept(bundle.binding_hash)
    needs_logical = include_alpha_handoff and not _logical_files(
        factor_input_paths(workspace, bundle.binding_hash)[0] if kept else source,
        panel_snapshot_hash,
    )
    if kept and not needs_logical:
        read_factor_bundle(workspace, bundle.binding_hash)
    else:
        market.load_universe_manifest_revision(
            str(panel["safe_summary"]["lineage"]["manifest_revision"])
        )
        with staging_directory(parent) as probe:
            first = Path(probe) / "first"
            first.write_bytes(b"")
            try:
                os.link(first, Path(probe) / "second")
                links_available = True
            except OSError:
                links_available = False
        # The staged copy: a workspace-owned Parquet is linked where the volume allows
        # (`_copy_input`); the binding is then kept on the data root, never pooled here.
        staged_items = (confined(source, name) for name, _digest in files)
        required = sum(
            0
            if links_available and item.suffix == ".parquet" and item.is_relative_to(workspace)
            else item.stat().st_size
            for item in staged_items
        )
        require_storage_capacity(workspace, additional_bytes=required)
        with staging_directory(parent, ".binding-") as temporary:
            staged = Path(temporary) / "complete"
            # Every staged file is verified once by reading the bytes it will
            # keep (a link's are the source's own). The source is read again,
            # whole, at the end of the binding; a change to it anywhere in
            # between is refused there.
            for name, expected in files:
                src = confined(source, name)
                target = confined(staged / "source", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                _copy_input(workspace, src, target)
                if file_digest(target) != expected:
                    raise AuthoringError("research_experiment.source_changed_during_binding")
            if include_alpha_handoff:
                owner = logical_panel_owner(staged / "source")
                owner.publish_snapshot(resolver.feature_panel_manifest_uri(panel_snapshot_hash))
                owner.verify_snapshot(panel_snapshot_hash)
                # The publisher adds the logical files under the staged
                # artifacts; the copied objects were verified above and are
                # named by their digests, so only what it added is read.
                verified_names = dict(files)
                added = tuple(
                    sorted(
                        name
                        for name in _closed_files(staged / "source")
                        if name not in verified_names
                    )
                )
                added_digests = file_digests(confined(staged / "source", name) for name in added)
                files_with_logical = tuple(
                    sorted(
                        (
                            *files,
                            *(
                                (name, added_digests[confined(staged / "source", name)])
                                for name in added
                            ),
                        )
                    )
                )
                bundle = FactorInputBundle.create(
                    panel_snapshot_hash=bundle.panel_snapshot_hash,
                    outcome_snapshot_hash=bundle.outcome_snapshot_hash,
                    sessions=bundle.sessions,
                    files=files_with_logical,
                    database_snapshot_hash=bundle.database_snapshot_hash,
                    binding_version=bundle.binding_version,
                    outcomes=bundle.outcomes,
                )
                binding = ResearchWorkspaceExperimentInput(
                    input_id=input_id, binding_hash=bundle.binding_hash
                )
            if (source / "market-data.duckdb.wal").exists():
                raise AuthoringError("research_experiment.source_changed_during_binding")
            final_digests = file_digests(confined(source, name) for name, _expected in files)
            if any(final_digests[confined(source, name)] != expected for name, expected in files):
                raise AuthoringError("research_experiment.source_changed_during_binding")
            require_storage_capacity(workspace, additional_bytes=0)
            (staged / "manifest.json").write_text(
                bundle.model_dump_json(indent=2), encoding="utf-8"
            )
            if not history.research_input_kept(bundle.binding_hash):
                history.keep_research_input(
                    bundle.binding_hash, _kept_files(staged, bundle.binding_hash, bundle)
                )
            read_factor_bundle(workspace, bundle.binding_hash)
    history.hold_research_input(bundle.binding_hash)  # kept for another workspace: held here too
    select_research_inputs(
        factor_input_paths(workspace, bundle.binding_hash)[0].parent,
        source,
        bundle,
        str(panel["safe_summary"]["lineage"]["manifest_revision"]),
    )
    if not bind_configuration:
        return binding

    # Onto the manifest as it stands under the gate (WM): a writer since keeps what it wrote.
    def bind(current: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
        existing = tuple(v for v in current.experiment_inputs or () if v.input_id != input_id)
        updated = current.with_bindings(experiment_inputs=(*existing, binding))
        if updated != current:
            _record_input_publication(parent, input_id, binding, current, updated, claim)
        return updated

    update_research_workspace_manifest(workspace, bind, gate=gate)
    return binding


def _kept_files(
    staged: Path, binding_hash: str, bundle: FactorInputBundle
) -> list[tuple[str, Path]]:
    """A binding's files at their paths under `research-inputs/`: its folder, and the database
    the bundle names at the pooled path the door reads it from, while the folder holds it (a
    0.2.7 copy a cleanup released keeps only what it has; the door refuses the rest by name)."""
    files = [(f"{binding_hash}/{relative}", path) for relative, path in folder_files(staged)]
    database = confined(staged / "source", "market-data.duckdb")
    if bundle.database_snapshot_hash is not None and database.is_file():
        files.append((f"databases/{bundle.database_snapshot_hash}/market-data.duckdb", database))
    return files


def legacy_bindings(workspace: Path) -> list[tuple[str, list[tuple[str, Path]]]]:
    """Each 0.2.7 binding `workspace` copied under `research-inputs/<binding_hash>/`, with its
    files laid out as the door keeps them. The pools those copies link, and the publication
    records (`publications/`, which stay in the workspace), are not bindings."""
    folder = _windows_long_path(workspace) / "research-inputs"
    try:
        names = sorted(os.listdir(folder))
    except FileNotFoundError:
        return []
    found = []
    for name in names:
        if len(name) != 64 or any(c not in "0123456789abcdef" for c in name):
            continue
        tree = folder / name
        if tree.is_symlink() or not tree.is_dir():
            raise AuthoringError("research_experiment.input_link_not_admitted")
        bundle = FactorInputBundle.model_validate_json((tree / "manifest.json").read_bytes())
        if bundle.binding_hash != name:
            raise AuthoringError("research_experiment.input_manifest_identity_mismatch")
        found.append((name, _kept_files(tree, name, bundle)))
    return found


def _reads_equal(root: Path, read: Any, binding_hash: str, files: list[tuple[str, Path]]) -> bool:
    """Whether the binding kept on `root` (as `read` records it) is the copy's files, byte for
    byte: the same paths, each body's BLAKE3 the copy's own."""
    kept = inputs.members(root, read, binding_hash)
    copy = dict(files)
    return [relative for relative, _ in kept] == sorted(copy) and all(
        inputs.file_digest(body) == inputs.file_digest(copy[relative]) for relative, body in kept
    )


def research_input_upgrade_family(holder: Callable[[Path], str | None]) -> Family:
    """Each 0.2.7 binding copy kept once on the data root behind the door, held by its workspace
    (`holder` names its id on the log; a workspace with copies and no id is refused by name, so
    nothing is kept unheld); ALL EQUAL when every kept binding reads back as its copy's bytes.
    The upgrade reads the copies themselves."""

    def step(runner: Any) -> None:
        for workspace in runner.workspaces:
            held = holder(workspace)
            bindings = legacy_bindings(workspace)
            if bindings and held is None:
                raise DataError(
                    f"the workspace at {workspace} has research input copies and no id on the "
                    "log to hold them; open it once so the Host records it",
                    code="upgrade.refused",
                )
            for binding_hash, files in bindings:
                inputs.keep_binding(
                    runner.root, runner.log, binding_hash, files, workspace=held, durable=False
                )
            runner.log.flush()

    def verify(runner: Any) -> bool:
        return all(
            _reads_equal(runner.root, runner.log.read, binding_hash, files)
            for workspace in runner.workspaces
            for binding_hash, files in legacy_bindings(workspace)
        )

    def holds(workspace: Path) -> bool:
        return any(not history.research_input_kept(b) for b, _ in legacy_bindings(workspace))

    return Family("research-inputs", step, verify, holds)


_POOLS: Final = ("databases", "parquet")
"""The 0.2.7 pools under `research-inputs/` that its binding copies link."""


def _copies_left(workspace: Path) -> bool:
    """Whether `workspace` still holds a 0.2.7 binding copy, or a removal of them not finished
    (`set_aside`'s `retired-0.2.7/` under `research-inputs/`, recorded or not yet deleted)."""
    folder = _windows_long_path(workspace) / "research-inputs"
    return bool(legacy_bindings(workspace)) or (folder / RETIRED_DIRECTORY).exists()


def _remove_copies(folder: Path, binding_hashes: list[str]) -> None:
    """Retire one workspace's verified copies through the task upgrade's removal boundary: the
    set (both pools, then each binding tree) moves whole under `retired-0.2.7/`, its directory
    recorded before its first move, so a stop leaves each tree at its old name or retired, never
    part of one, and a resumed removal finishes the recorded set first; then it is deleted. The
    pools name only the copies they link, so they go with the set."""
    set_aside(folder, [*(folder / pool for pool in _POOLS), *(folder / b for b in binding_hashes)])
    shutil.rmtree(folder / RETIRED_DIRECTORY)


def research_input_copies_family(repository: Path) -> Family:
    """The 0.2.7 copies removed: first a machine backup in `repository` that holds their kept
    bodies, its seals verified, and every binding replayed from it (its saved log, its pooled
    bodies: what a restore places) ALL EQUAL with its copy; then each workspace's copies and its
    pools (`_remove_copies`). A replay that differs keeps every copy and stops the upgrade by
    name. A stop part-way resumes: a removal begun finishes before any other step. It selects no
    store."""

    def step(runner: Any) -> None:
        for workspace in runner.workspaces:  # a removal a stop left part-way, first
            folder = _windows_long_path(workspace) / "research-inputs"
            if (folder / RETIRED_DIRECTORY).exists():
                _remove_copies(folder, [b for b, _ in legacy_bindings(workspace)])
        held = [(workspace, legacy_bindings(workspace)) for workspace in runner.workspaces]
        if not any(bindings for _, bindings in held):
            return
        runner.log.flush()  # the generation holds every kept body and its pin
        generation = backup_data_root(runner.root, repository)
        verify_data_root(generation, source=runner.root)
        with generation_reads(generation) as (pool, read):
            for _workspace, bindings in held:
                for binding_hash, files in bindings:
                    if not _reads_equal(pool, read, binding_hash, files):
                        raise DataError(
                            f"the backup of research input {binding_hash} does not read back as "
                            "its copy; every copy is kept",
                            code="upgrade.refused",
                        )
        for workspace, bindings in held:
            if bindings:
                folder = _windows_long_path(workspace) / "research-inputs"
                _remove_copies(folder, [binding_hash for binding_hash, _files in bindings])

    def verify(runner: Any) -> bool:
        return not any(_copies_left(workspace) for workspace in runner.workspaces)

    return Family("research-input-copies", step, verify, _copies_left)


def select_research_inputs(
    destination: Path, source: Path, bundle: FactorInputBundle, universe: str
) -> None:
    """Fix what the bundle's research may see, once per binding: a selection at a moment world
    holds exactly the copy's market values, which the copy then reads through. None outside a
    Host, or while world does not hold them as of now (the source's tables, which their stages
    publish, are not the copy's, or world holds another value of one of their keys, another
    workspace's later one), and the copy reads its own tables."""
    host = history.opened_log()
    copy = MarketDataRepository(destination / "source")
    if host is None or world_reads.selected(copy.path) is not None:
        return
    cutoff = host[0].clock()
    if not world_reads.holds(copy.path, cutoff):
        return
    sessions = (bundle.sessions[0].isoformat(), bundle.sessions[-1].isoformat())
    recipes = {"feature_panel": bundle.panel_snapshot_hash}
    recipes["execution_outcome"] = bundle.outcome_snapshot_hash
    window = {"cutoff_us": cutoff, "universe": universe, "valid_from": sessions[0]}
    window |= {"valid_to": sessions[1]}
    history.selection(
        window | {"spec": {"binding_hash": bundle.binding_hash, "recipes": recipes}},
        actor=("host", "host"),
    )


def _record_input_publication(
    parent: Path,
    input_id: str,
    binding: ResearchWorkspaceExperimentInput,
    manifest: ResearchWorkspaceManifest,
    updated: ResearchWorkspaceManifest,
    claim: Callable[[ResearchWorkspaceManifest], None] | None,
) -> None:
    """Seal the receipt of one input's publication, and its writer's claim on the manifest that
    names it, before that manifest is written."""
    prior_hash = next(
        (v.binding_hash for v in manifest.experiment_inputs or () if v.input_id == input_id),
        None,
    )
    values = {
        "input_id": input_id,
        "prior_binding_hash": prior_hash,
        "binding_hash": binding.binding_hash,
        "prior_manifest_hash": manifest.manifest_hash,
        "next_manifest_hash": updated.manifest_hash,
    }
    identity = canonical_hash(values)
    folder = parent / "publications"
    folder.mkdir(exist_ok=True)
    target = folder / f"{identity}.json"
    payload = json.dumps(
        {**values, "receipt_hash": identity}, sort_keys=True, separators=(",", ":")
    )
    if target.exists() and target.read_text(encoding="utf-8") != payload:
        raise AuthoringError("research_experiment.input_publication_conflict")
    if not target.exists():
        with staging_directory(folder) as temporary:
            staged_receipt = Path(temporary) / "receipt.json"
            staged_receipt.write_text(payload, encoding="utf-8")
            staged_receipt.rename(target)
    if claim is not None:
        claim(updated)


@dataclass(frozen=True)
class PreparedInputAuthority:
    """Retain one exact prepared envelope and its resolved research authority."""

    envelope: ResearchExperimentEnvelope
    authority: ResolvedResearchAuthority

    def resolve(self, envelope: ResearchExperimentEnvelope) -> ResolvedResearchAuthority:
        """Require the exact prepared envelope before returning retained authority.

        Args:
            envelope: Explicit sealed declaration.

        Returns:
            Exact prepared research authority.

        Raises:
            AuthoringError: Envelope differs from the one prepared.
        """
        if envelope != self.envelope:
            raise AuthoringError("research_experiment.prepared_envelope_mismatch")
        return self.authority


def factor_workflow(
    *,
    workspace: Path,
    binding_hash: str,
    document: Mapping[str, Any],
    bundle: FactorInputBundle | None = None,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> tuple[ResearchProgramWorkflow, ResolvedResearchAuthority, dict[str, Any]]:
    """Resolve once; compiler, executor and caller consume this same authority.

    ``bundle`` is the input bundle the caller verified moments earlier in the
    same request (``read_factor_bundle`` digests every file of the input); a
    caller that has not read it leaves it unset and it is read here. It must
    be the bundle of ``binding_hash``.
    """
    if bundle is None:
        bundle = read_factor_bundle(workspace, binding_hash)
    elif bundle.binding_hash != binding_hash:
        raise AuthoringError("research_experiment.input_manifest_identity_mismatch")
    source, _artifact_root = factor_input_paths(workspace, binding_hash)
    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    if envelope.sessions.as_of.phase.name != "OFFICIAL_CLOSE":
        raise AuthoringError("research_experiment.official_close_cutoff_required")
    if envelope.kind != FACTOR_EXPERIMENT_KIND or envelope.data_snapshot_handle != (
        feature_input.source_handle if feature_input else bundle.panel_snapshot_hash
    ):
        raise AuthoringError("research_experiment.method_or_input_not_installed")
    if feature_input is not None and (
        feature_input.input_binding_hash != binding_hash
        or feature_input.base_panel_snapshot_hash != bundle.panel_snapshot_hash
    ):
        raise AuthoringError("research_experiment.prepared_features_input_mismatch")
    authority = WorkspaceResearchAuthorityResolver(
        workspace=source, feature_input=feature_input
    ).resolve(envelope)
    resolver = ArtifactResolver(source / "artifacts")
    panel = (
        feature_input.panel_manifest
        if feature_input
        else dict(resolver.load_feature_panel_manifest(str(authority.panel_manifest_ref)))
    )
    inventory = factor_inventory_from_panel_manifest(panel)
    executor = build_factor_executor(
        workspace=source,
        artifact_root=source / "artifacts",
        envelope=envelope,
        causal_outcome_snapshot_handle=bundle.outcome_snapshot_hash,
        resolved_panel=panel,
        feature_input=feature_input,
    )
    authority = ResolvedResearchAuthority.create(
        **{
            **authority.model_dump(
                mode="python", exclude={"authority_hash", "execution_input_binding_hash"}
            ),
            "execution_input_binding_hash": executor.execution_input_binding_hash,
        }
    )
    if len(authority.ordered_listing_ids) < executor.minimum_required_listings:
        raise AuthoringError(
            f"research_experiment.insufficient_listing_support:"
            f"{len(authority.ordered_listing_ids)}<{executor.minimum_required_listings}"
        )
    from alphalattice.control.product_host.composition.research_authoring import (
        build_research_program_workflow,
    )

    workflow = build_research_program_workflow(
        workspace=source,
        workspace_root=workspace,
        executors=(executor,),
        authority=PreparedInputAuthority(envelope, authority),
        compilers=(FactorExperimentCompiler(inventory),),
        verifier_kinds=(FACTOR_EXPERIMENT_KIND,),
    )
    return (
        workflow,
        authority,
        executor.describe_execution_window(str(authority.panel_manifest_ref)),
    )


def build_factor_executor(
    *,
    workspace: Path,
    artifact_root: Path,
    envelope: ResearchExperimentEnvelope,
    evidence_policy: Any | None = None,
    redundancy_policy: Any | None = None,
    outcome_artifact_root: Path | None = None,
    causal_outcome_snapshot_handle: str | None = None,
    feature_catalog: FeatureCatalog | None = None,
    feature_kernels: FeatureKernelRegistry | None = None,
    resolved_panel: dict[str, Any] | None = None,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
    walk_forward_policy: Any | None = None,
) -> FactorExperimentExecutor:
    """The Factor factory shared by legacy Desk installation and Local Web. A caller that
    states no policy gets the workspace standard's (`factor_scale_policies`)."""
    if evidence_policy is None and redundancy_policy is None and walk_forward_policy is None:
        scale = factor_scale_policies(workspace, envelope.universe_handle)
        evidence_policy = scale.get("evidence_policy")
        redundancy_policy = scale.get("redundancy_policy")
        walk_forward_policy = scale.get("walk_forward_policy")
    if resolved_panel is None:
        authority = WorkspaceResearchAuthorityResolver(
            workspace=workspace,
            artifact_root=artifact_root,
            feature_catalog=feature_catalog,
            feature_kernels=feature_kernels,
            feature_input=feature_input,
        ).resolve(envelope)
        if authority.panel_manifest_ref is None:
            raise AuthoringError("research_authoring.panel_authority_required")
        resolved_panel = (
            feature_input.panel_manifest
            if feature_input
            else dict(
                ArtifactResolver(artifact_root).load_feature_panel_manifest(
                    authority.panel_manifest_ref
                )
            )
        )
    outcome_root = outcome_artifact_root or artifact_root
    reader = CausalExecutionOutcomeDevelopmentReader(outcome_root)
    outcome = resolve_published_execution_outcome(
        artifact_root=outcome_root,
        listing_set_hash=str(resolved_panel.get("listing_set_hash", "")),
        snapshot_handle=causal_outcome_snapshot_handle,
    )
    return FactorExperimentExecutor(
        panel_manifest=resolved_panel,
        feature_reader=feature_input.reader
        if feature_input
        else FeaturePanelReader(ArtifactResolver(artifact_root)),
        outcome_reader=reader,
        outcome_snapshot_hash=outcome.snapshot_hash,
        outcome_manifest_ref=reader.manifest_uri(outcome.snapshot_hash),
        frozen_at=datetime.combine(
            envelope.sessions.as_of.session, datetime.min.time(), tzinfo=UTC
        ),
        inventory=factor_inventory_from_panel_manifest(resolved_panel),
        evidence_policy=evidence_policy,
        redundancy_policy=redundancy_policy,
        walk_forward_policy=walk_forward_policy,
    )


def factor_scale_policies(workspace: Path, market_profile_id: str) -> dict[str, Any]:
    """The Factor policies of the workspace's active universe standard: its listing floor and
    history years (`factor_policies_at_scale`); none for a workspace with no active candidate set,
    or at the shipped scale."""
    record = MarketDataRepository(workspace).readiness.load(market_profile_id)
    document = record.active_candidate_manifest_document if record else None
    if document is None:
        return {}
    standard = bootstrap_from_candidate_manifest_document(dict(document)).standard
    policies: dict[str, Any] = factor_policies_at_scale(
        minimum_listings=standard.minimum_factor_listings,
        history_years=standard.minimum_history_calendar_years,
    )
    return policies


def normalize_factor_document(
    document: Mapping[str, Any],
    binding: ResearchWorkspaceExperimentInput,
    bundle: FactorInputBundle,
    *,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> dict[str, Any]:
    """Normalize Factor authoring against one exact bundle and optional prepared source.

    Args:
        document: Explicit authored Factor declaration.
        binding: Exact admitted input binding.
        bundle: Exact retained input bundle.
        feature_input: Optional exact resolved prepared feature source.

    Returns:
        Normalized Factor document supplied by the existing input authoring owner.
    """
    return normalize_input_document(
        document,
        binding,
        bundle,
        section_name="factor",
        source_handle=feature_input.source_handle if feature_input else None,
    )


def normalize_input_document(
    document: Mapping[str, Any],
    binding: ResearchWorkspaceExperimentInput,
    bundle: FactorInputBundle,
    *,
    section_name: str,
    source_handle: str | None = None,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    """One managed envelope/path rule for input-bound standalone Desk documents."""
    value = json.loads(json.dumps(document, default=str))
    if set(value) != {"experiment", section_name} or not isinstance(value["experiment"], dict):
        raise AuthoringError("research_experiment.document_invalid")
    section = value["experiment"]
    handles = (
        {source_handle}
        if source_handle is not None
        else {"current", binding.input_id, bundle.panel_snapshot_hash}
    )
    if section.get("data_snapshot_handle") not in handles:
        raise AuthoringError("research_experiment.input_not_selected")
    section["data_snapshot_handle"] = source_handle or bundle.panel_snapshot_hash
    output = section.pop("output_workspace", "managed")
    baseline = section.pop("baseline_workspace", "managed")
    key = canonical_hash({"document": value, "input": binding.binding_hash})
    expected_output = f"research-experiments/{key}"
    expected_baseline = baseline_path or f"research-inputs/{binding.binding_hash}/source"
    if output not in {"managed", expected_output} or baseline not in {"managed", expected_baseline}:
        raise AuthoringError("research_experiment.output_not_host_managed")
    section.update(output_workspace=expected_output, baseline_workspace=expected_baseline)
    return cast(dict[str, Any], value)


def factor_template(bundle: FactorInputBundle) -> dict[str, Any]:
    """Build an explicit Factor draft from one exact input bundle and installed policies.

    Args:
        bundle: Exact retained input bundle.

    Returns:
        Factor envelope with empty authored factor selection and installed screening/redundancy
        defaults.
    """
    return {
        "experiment": input_experiment_envelope(
            bundle, kind=FACTOR_EXPERIMENT_KIND, maximum_candidates=256, seed=20260816
        ),
        "factor": {
            "factor_ids": [],
            "screening_policy": INSTALLED_SCREENING_POLICIES[0],
            "redundancy_policy": INSTALLED_REDUNDANCY_POLICIES[0],
        },
    }


def input_experiment_envelope(
    bundle: FactorInputBundle, *, kind: str, maximum_candidates: int, seed: int
) -> dict[str, Any]:
    """Common host-managed envelope; each Desk supplies its own method and seed."""
    return {
        "kind": kind,
        "schema_id": "research-experiment-envelope",
        "data_snapshot_handle": bundle.panel_snapshot_hash,
        "universe_handle": PROFILE,
        "sessions": {
            "start": str(bundle.sessions[0]),
            "end": str(bundle.sessions[-1]),
            "as_of": {"session": str(bundle.sessions[-1]), "phase": "OFFICIAL_CLOSE"},
        },
        "budget": {"maximum_candidates": maximum_candidates, "maximum_numerical_calls": 5000},
        "determinism": {"seed": seed, "thread_limit": 1, "network_disabled": True},
        "output_workspace": "managed",
        "baseline_workspace": "managed",
        "publication_intent": "DEVELOPMENT_EVIDENCE_ONLY",
    }
