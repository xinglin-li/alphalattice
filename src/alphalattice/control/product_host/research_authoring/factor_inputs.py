"""Admit a local Factor input closure and prepare its existing authoring owners."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Self, cast
from uuid import UUID

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
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
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
from alphalattice.foundation.factor_research.experiments.execution import FactorExperimentExecutor
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
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
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
)

PROFILE = "us-current-index-research"


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

    digest = sha256()
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
        if (
            self.database_snapshot_hash is not None
            and dict(self.files).get("market-data.duckdb") != self.database_snapshot_hash
        ):
            raise AuthoringError("research_experiment.input_database_binding_mismatch")
        return self


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
    if len(binding_hash) != 64 or any(v not in "0123456789abcdef" for v in binding_hash):
        raise AuthoringError("research_experiment.input_hash_invalid")
    root = confined(workspace, f"research-inputs/{binding_hash}")
    bundle = FactorInputBundle.model_validate_json(confined(root, "manifest.json").read_bytes())
    if bundle.binding_hash != binding_hash:
        raise AuthoringError("research_experiment.input_manifest_identity_mismatch")
    if verify:
        source = confined(root, "source")
        for name, _expected in bundle.files:
            target = confined(source, name)
            if not target.exists():
                require_available_input(
                    workspace, target.relative_to(workspace.resolve()).as_posix()
                )
        if _closed_files(source) != {name for name, _ in bundle.files}:
            raise AuthoringError("research_experiment.input_file_set_changed")
        database: Path | None = None
        if bundle.database_snapshot_hash is not None:
            database = confined(
                workspace,
                f"research-inputs/databases/{bundle.database_snapshot_hash}/market-data.duckdb",
            )
            if not database.is_file():
                require_available_input(
                    workspace, database.relative_to(workspace.resolve()).as_posix()
                )
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
    if len(binding_hash) != 64 or any(v not in "0123456789abcdef" for v in binding_hash):
        raise AuthoringError("research_experiment.input_hash_invalid")
    source = confined(workspace, f"research-inputs/{binding_hash}/source")
    # Referenced bundles expose a local alias of a sealed DB object. No alias
    # ever points at the mutable market database or another workspace.
    return source, source / "artifacts"


def _copy_input(
    workspace: Path,
    source: Path,
    target: Path,
    expected: str,
) -> bool:
    """Share only workspace-owned immutable Parquet, never the mutable database.

    Answers whether ``target`` holds bytes this call verified: a link to a
    pool object whose digest was read here (staged from the source, or already
    pooled) is verified; a copied file is not, and its caller must read it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    database = source.name == "market-data.duckdb"
    if source.suffix != ".parquet" and not database:
        shutil.copy2(source, target)
        return False
    relative = (
        f"databases/{expected}/market-data.duckdb" if database else f"parquet/{expected}.parquet"
    )
    pool = confined(workspace, f"research-inputs/{relative}")
    if not pool.exists():
        pool.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=pool.parent) as temporary:
            staged = Path(temporary) / "object"
            if not database and source.resolve().is_relative_to(workspace.resolve()):
                try:
                    os.link(source, staged)
                except OSError:
                    require_storage_capacity(workspace, additional_bytes=source.stat().st_size)
                    shutil.copy2(source, staged)
            else:
                shutil.copy2(source, staged)
            if file_digest(staged) != expected:
                raise AuthoringError("research_experiment.source_changed_during_binding")
            staged.rename(pool)
    elif file_digest(pool) != expected:
        raise AuthoringError("research_experiment.input_file_tampered")
    try:
        os.link(pool, target)
    except OSError:
        require_storage_capacity(workspace, additional_bytes=pool.stat().st_size)
        shutil.copy2(pool, target)
        return False
    return True


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


def compatible_factor_inputs(original: FactorInputBundle, candidate: FactorInputBundle) -> bool:
    """A handoff may add logical metadata, never replace the original inputs."""
    return (
        original.panel_snapshot_hash == candidate.panel_snapshot_hash
        and original.outcome_snapshot_hash == candidate.outcome_snapshot_hash
        and original.sessions == candidate.sessions
        and dict(original.files).get("market-data.duckdb", original.database_snapshot_hash)
        == dict(candidate.files).get("market-data.duckdb", candidate.database_snapshot_hash)
        and {(p, h) for p, h in original.files if p != "market-data.duckdb"}.issubset(
            {(p, h) for p, h in candidate.files if p != "market-data.duckdb"}
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
            workspace=workspace,
            source=source,
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
) -> ResearchWorkspaceExperimentInput:
    """One Outcome/logical-Panel/bundle handoff for first use and later capture."""
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
        workspace=root,
        source=root,
        input_id=input_id,
        panel_snapshot_hash=panel_snapshot_hash,
        outcome_snapshot_hash=outcome.manifest.snapshot_hash,
        include_alpha_handoff=True,
        gate=session.mutation_gate,
        bind_configuration=bind_configuration,
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
    for chunk in outcome.development_chunks:
        confined(
            source,
            str(store._parquet_path(chunk.split, chunk.content_hash).relative_to(source)),
        )
        paths.append(store.resolve_chunk(chunk))
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
    )
    # The explicitly named installed input is the idempotency anchor, not a
    # search for a latest compatible directory. Verify it before reuse.
    if include_alpha_handoff:
        prior = next((v for v in manifest.experiment_inputs or () if v.input_id == input_id), None)
        if prior is not None:
            candidate = read_factor_bundle(workspace, prior.binding_hash)
            prior_source = confined(workspace, f"research-inputs/{prior.binding_hash}/source")
            if compatible_factor_inputs(bundle, candidate) and _logical_files(
                prior_source, panel_snapshot_hash
            ):
                bundle = candidate
    binding = ResearchWorkspaceExperimentInput(input_id=input_id, binding_hash=bundle.binding_hash)
    parent = confined(workspace, "research-inputs")
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / bundle.binding_hash
    needs_logical = include_alpha_handoff and not _logical_files(
        destination / "source" if destination.exists() else source, panel_snapshot_hash
    )
    if destination.exists() and not needs_logical:
        read_factor_bundle(workspace, bundle.binding_hash)
    else:
        market.load_universe_manifest_revision(
            str(panel["safe_summary"]["lineage"]["manifest_revision"])
        )
        with TemporaryDirectory(dir=parent) as probe:
            first = Path(probe) / "first"
            first.write_bytes(b"")
            try:
                os.link(first, Path(probe) / "second")
                links_available = True
            except OSError:
                links_available = False
        required = 0
        for name, digest in files:
            item = confined(source, name)
            if item.name == "market-data.duckdb":
                pool = parent / "databases" / digest / "market-data.duckdb"
                required += 0 if pool.exists() else item.stat().st_size
            elif item.suffix == ".parquet":
                pool = parent / "parquet" / f"{digest}.parquet"
                required += (
                    0
                    if pool.exists() or (links_available and item.is_relative_to(workspace))
                    else item.stat().st_size
                )
            else:
                required += item.stat().st_size
                continue
            if not links_available:
                required += item.stat().st_size
        require_storage_capacity(workspace, additional_bytes=required)
        with TemporaryDirectory(dir=parent, prefix=".binding-") as temporary:
            staged = Path(temporary) / "complete"
            # Every staged object is verified once by the bytes it will keep:
            # a pooled object by its digest when staged or already pooled
            # (the target is a link to it), a copied file by reading the
            # copy. The source is read again, whole, at the end of the
            # binding; a change to it anywhere in between is refused there.
            for name, expected in files:
                src = confined(source, name)
                target = confined(staged / "source", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                linked = _copy_input(workspace, src, target, expected)
                if not linked and file_digest(target) != expected:
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
                )
                binding = ResearchWorkspaceExperimentInput(
                    input_id=input_id, binding_hash=bundle.binding_hash
                )
                destination = parent / bundle.binding_hash
            if (source / "market-data.duckdb.wal").exists():
                raise AuthoringError("research_experiment.source_changed_during_binding")
            final_digests = file_digests(confined(source, name) for name, _expected in files)
            if any(final_digests[confined(source, name)] != expected for name, expected in files):
                raise AuthoringError("research_experiment.source_changed_during_binding")
            require_storage_capacity(workspace, additional_bytes=0)
            (staged / "manifest.json").write_text(
                bundle.model_dump_json(indent=2), encoding="utf-8"
            )
            if destination.exists():
                read_factor_bundle(workspace, bundle.binding_hash)
            else:
                staged.rename(destination)
    if not bind_configuration:
        return binding

    # Onto the manifest as it stands under the gate, not the one read before the copy:
    # a writer in between keeps what it wrote (WM).
    def bind(current: ResearchWorkspaceManifest) -> ResearchWorkspaceManifest:
        existing = tuple(v for v in current.experiment_inputs or () if v.input_id != input_id)
        updated = current.with_bindings(experiment_inputs=(*existing, binding))
        if updated != current:
            _record_input_publication(parent, input_id, binding, current, updated)
        return updated

    update_research_workspace_manifest(workspace, bind, gate=gate)
    return binding


def _record_input_publication(
    parent: Path,
    input_id: str,
    binding: ResearchWorkspaceExperimentInput,
    manifest: ResearchWorkspaceManifest,
    updated: ResearchWorkspaceManifest,
) -> None:
    """Seal the receipt of one input's publication before the manifest names it."""
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
        with TemporaryDirectory(dir=folder) as temporary:
            staged_receipt = Path(temporary) / "receipt.json"
            staged_receipt.write_text(payload, encoding="utf-8")
            staged_receipt.rename(target)


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
) -> FactorExperimentExecutor:
    """The Factor factory shared by legacy Desk installation and Local Web."""
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
    )


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
