"""One explicit composition root for a writable local market-data workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from alphalattice.control.data_platform.maintenance.contracts import AgentExecutionBudget
from alphalattice.control.data_platform.maintenance.coordinator import (
    WorkspaceMaintenanceCoordinator,
)
from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.data_platform.preflight import (
    resolve_trading_session_authority,
)
from alphalattice.control.data_platform.runtime import PreFactorDataOperations
from alphalattice.control.observation_runtime.telemetry.progress import WorkspaceProgressPublisher
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.database import WorkspaceDatabase
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.layer import FeatureCatalogLayer
from alphalattice.foundation.feature_engine.inputs.closure_source import (
    FeatureClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputGateway,
    FeatureInputGovernanceService,
)
from alphalattice.foundation.feature_engine.panels.artifacts import PanelArtifactCompositionOwner
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_coordinator import (
    FeatureBaseClosureCoordinator,
    FeatureLayerClosures,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
)
from alphalattice.foundation.feature_engine.panels.logical_identity import (
    PanelLogicalArtifactStore,
    PanelLogicalIdentityPublisher,
)
from alphalattice.foundation.feature_engine.panels.recovery_binding import (
    PanelRecoveryBindingPublisher,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.foundation.feature_engine.publication.sector_map_activation import (
    SectorRevisionMapActivationCoordinator,
)
from alphalattice.foundation.feature_engine.publication.snapshots import (
    FeaturePanelSnapshotPublisher,
)
from alphalattice.foundation.feature_engine.runtime.service import FeatureFoundationService
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.returns.semantic_revisions import (
    AdjustedReturnSemanticRevisionPublisher,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.sources.providers import (
    MarketDataProvider,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.research_foundation.runtime.delta_service import (
    PreResearchDeltaService,
)


class WorkspaceApplicationControls(Protocol):
    """Share the admitted workspace mutation gate, writer lease and task registry."""

    workspace: Path
    mutation_gate: WorkspaceMutationGate
    writer_lease: WorkspaceWriterLease
    task_control_registry: DuckDbTaskControlRegistry


@dataclass
class WorkspaceRuntime:
    """The sole in-process composition root allowed to mutate one workspace.

    This class deliberately owns construction and dependency wiring only.  It
    does not own task lifecycle, market-data policy, or a background scheduler.
    DuckDB still rejects another read-write process at its database boundary.
    """

    database: WorkspaceDatabase
    market_data: MarketDataRepository
    feature_state: FeatureStateRepository
    panel_state: PanelStateRepository
    manifest: UniverseManifest
    provider: MarketDataProvider
    artifact_resolver: ArtifactResolver
    mutation_gate: WorkspaceMutationGate
    writer_lease: WorkspaceWriterLease
    task_control_registry: DuckDbTaskControlRegistry
    data_operations: PreFactorDataOperations
    feature_foundation: FeatureFoundationService
    sector_activation: SectorRevisionMapActivationCoordinator
    maintenance_registry: DuckDbWorkspaceMaintenanceRegistry
    progress_publisher: WorkspaceProgressPublisher
    closure_ledger: FeatureClosureLedger
    feature_catalog: FeatureCatalog = field(default_factory=FeatureCatalog.load)
    feature_kernels: FeatureKernelRegistry = field(
        default_factory=default_extension_kernel_registry
    )
    owns_writer_lease: bool = True

    @classmethod
    def create(
        cls,
        *,
        workspace: Path,
        manifest: UniverseManifest,
        provider: MarketDataProvider,
        artifact_root: Path,
        max_live_symbols: int = 5,
        retry_budget: int = 2,
        feature_catalog: FeatureCatalog | None = None,
        feature_kernels: FeatureKernelRegistry | None = None,
        application_controls: WorkspaceApplicationControls | None = None,
    ) -> WorkspaceRuntime:
        # One catalog revision and one kernel set for every writer this runtime
        # composes.  Resolving them here rather than inside each writer is what
        # keeps the materializer, the persistence factor axis, and the panel
        # publisher from disagreeing about which factors exist.
        """Compose Data and Feature runtime owners with one catalog, kernel set and writer lease.

        A newly acquired lease is closed on construction failure. Shared application controls retain
        ownership of their existing lease.

        Args:
            workspace: Caller-owned workspace root.
            manifest: Explicit admitted universe manifest.
            provider: Explicit market data provider.
            artifact_root: Explicit artifact root.
            max_live_symbols: Declared source concurrency bound.
            retry_budget: Declared retry bound.
            feature_catalog: Optional exact installed feature catalog.
            feature_kernels: Optional exact installed extension kernels.
            application_controls: Optional shared admitted writer controls.

        Returns:
            Runtime with coherent catalog/materializer/storage axes and retained mutation authority.

        Raises:
            ValueError: Shared controls belong to another workspace.
        """
        catalog = feature_catalog if feature_catalog is not None else FeatureCatalog.load()
        kernels = (
            feature_kernels if feature_kernels is not None else default_extension_kernel_registry()
        )
        root = workspace.resolve()
        if application_controls is not None and application_controls.workspace.resolve() != root:
            raise ValueError("WorkspaceRuntime application session workspace does not match")
        owns_lease = application_controls is None
        lease = (
            WorkspaceWriterLease.acquire(root)
            if application_controls is None
            else application_controls.writer_lease
        )
        try:
            database = WorkspaceDatabase(root)
            market_data = MarketDataRepository(database)
            # Storage is shaped by the same revision as the materializer and the
            # publisher. Left to load the shipped catalog, an installed revision
            # would compute an extension factor and then have nowhere to persist
            # it: the current-feature table would have no column of that name.
            feature_state = FeatureStateRepository(
                database, market_data=market_data, installed_catalog=catalog
            )
            panel_state = PanelStateRepository(database, market_data=market_data)
            mutation_gate = (
                WorkspaceMutationGate()
                if application_controls is None
                else application_controls.mutation_gate
            )
            progress_publisher = WorkspaceProgressPublisher(artifact_root)
            artifact_resolver = ArtifactResolver(artifact_root)
            panel_artifacts = PanelArtifactCompositionOwner(artifact_resolver)
            closure_ledger = FeatureClosureLedger(PanelClosureArtifactStore(artifact_resolver))
            # One closure per part of the catalog's layer: the shipped base and a column per
            # activated factor.
            feature_persistence = FeatureLayerClosures(
                store=feature_state,
                source=FeatureClosureSourceRepository(database.path),
                ledger=closure_ledger,
                layer=FeatureCatalogLayer.over(catalog),
            )
            sector_activation = SectorRevisionMapActivationCoordinator(
                store=feature_state, mutation_gate=mutation_gate, ledger=closure_ledger
            )
            operations = PreFactorDataOperations(
                market_data=market_data,
                panel_state=panel_state,
                manifest=manifest,
                provider=provider,
                snapshot_root=artifact_root,
                max_live_symbols=max_live_symbols,
                retry_budget=retry_budget,
                artifact_resolver=artifact_resolver,
            )
            feature_foundation = FeatureFoundationService(
                market_data=market_data,
                feature_state=feature_state,
                panel_state=panel_state,
                manifest=manifest,
                provider=provider,
                mutation_gate=mutation_gate,
                panel_artifacts=panel_artifacts,
                feature_persistence=feature_persistence,
                sector_activation=sector_activation,
                progress_sink=progress_publisher.publish,
                installed_catalog=catalog,
                kernel_registry=kernels,
                session_authority_resolver=resolve_trading_session_authority,
            )
            return cls(
                database=database,
                market_data=market_data,
                feature_state=feature_state,
                panel_state=panel_state,
                manifest=manifest,
                provider=provider,
                artifact_resolver=artifact_resolver,
                mutation_gate=mutation_gate,
                writer_lease=lease,
                feature_catalog=catalog,
                feature_kernels=kernels,
                task_control_registry=(
                    DuckDbTaskControlRegistry(
                        resolve_task_control_database(database.workspace), gate=mutation_gate
                    )
                    if application_controls is None
                    else application_controls.task_control_registry
                ),
                data_operations=operations,
                feature_foundation=feature_foundation,
                sector_activation=sector_activation,
                maintenance_registry=DuckDbWorkspaceMaintenanceRegistry(
                    database.path, gate=mutation_gate
                ),
                progress_publisher=progress_publisher,
                closure_ledger=closure_ledger,
                owns_writer_lease=owns_lease,
            )
        except Exception:
            if owns_lease:
                lease.close()
            raise

    @classmethod
    def from_operations(
        cls,
        operations: PreFactorDataOperations,
        *,
        manifest: UniverseManifest,
        application_controls: WorkspaceApplicationControls | None = None,
    ) -> WorkspaceRuntime:
        """Adopt an already-built deterministic owner during terminal migration."""
        if operations.manifest.revision_sha256 != manifest.revision_sha256:
            raise ValueError("WorkspaceRuntime manifest does not match Data Operations")
        root = operations.market_data.workspace.resolve()
        if application_controls is not None and application_controls.workspace.resolve() != root:
            raise ValueError("WorkspaceRuntime application session workspace does not match")
        owns_lease = application_controls is None
        lease = (
            WorkspaceWriterLease.acquire(root)
            if application_controls is None
            else application_controls.writer_lease
        )
        try:
            database = operations.market_data.database
            market_data = operations.market_data
            feature_state = FeatureStateRepository(database, market_data=market_data)
            panel_state = operations.panel_state
            gate = (
                WorkspaceMutationGate()
                if application_controls is None
                else application_controls.mutation_gate
            )
            progress_publisher = WorkspaceProgressPublisher(operations.snapshot_root)
            resolver = operations.artifact_resolver or ArtifactResolver(operations.snapshot_root)
            panel_artifacts = PanelArtifactCompositionOwner(resolver)
            closure_ledger = FeatureClosureLedger(PanelClosureArtifactStore(resolver))
            feature_persistence = FeatureBaseClosureCoordinator(
                store=feature_state,
                source=FeatureClosureSourceRepository(database.path),
                ledger=closure_ledger,
                factor_ids=FeatureCatalog.load().factor_ids,
            )
            sector_activation = SectorRevisionMapActivationCoordinator(
                store=feature_state, mutation_gate=gate, ledger=closure_ledger
            )
            operations.artifact_resolver = resolver
            feature_foundation = FeatureFoundationService(
                market_data=market_data,
                feature_state=feature_state,
                panel_state=panel_state,
                manifest=manifest,
                provider=operations.provider,
                mutation_gate=gate,
                panel_artifacts=panel_artifacts,
                feature_persistence=feature_persistence,
                sector_activation=sector_activation,
                progress_sink=progress_publisher.publish,
                session_authority_resolver=resolve_trading_session_authority,
            )
            return cls(
                database=database,
                market_data=market_data,
                feature_state=feature_state,
                panel_state=panel_state,
                manifest=manifest,
                provider=operations.provider,
                artifact_resolver=resolver,
                mutation_gate=gate,
                writer_lease=lease,
                task_control_registry=(
                    DuckDbTaskControlRegistry(
                        resolve_task_control_database(database.workspace), gate=gate
                    )
                    if application_controls is None
                    else application_controls.task_control_registry
                ),
                data_operations=operations,
                feature_foundation=feature_foundation,
                sector_activation=sector_activation,
                maintenance_registry=DuckDbWorkspaceMaintenanceRegistry(database.path, gate=gate),
                progress_publisher=progress_publisher,
                closure_ledger=closure_ledger,
                owns_writer_lease=owns_lease,
            )
        except Exception:
            if owns_lease:
                lease.close()
            raise

    def prepare_feature_closure(self) -> None:
        """Open the existing genesis owner only on an uninitialized publication plane.

        Each part of the catalog's layer opens its own closure: a person's
        activation opens its column's beside the held base's.
        """
        from alphalattice.foundation.feature_engine.runtime.closure_genesis import (
            FeatureClosureGenesisService,
        )

        for part in self.feature_foundation.layer.parts:
            if self.closure_ledger.current_head(part.binding.catalog_hash) is None:
                FeatureClosureGenesisService(
                    panel_state=self.panel_state,
                    source=FeatureClosureSourceRepository(self.database.path),
                    ledger=self.closure_ledger,
                ).open_genesis(manifest=self.manifest, catalog=part)

    def maintenance_coordinator(
        self,
        *,
        readiness_gate: Any,
        diagnose: Any | None = None,
        agent_budget: AgentExecutionBudget | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> WorkspaceMaintenanceCoordinator:
        """Compose the concrete task-plane coordinator without another graph."""
        governance = FeatureInputGovernanceService(
            market_data=self.market_data,
            panel_state=self.panel_state,
            mutation_gate=self.mutation_gate,
            gateway=FeatureInputGateway(),
        )
        adjusted_return_revisions = AdjustedReturnSemanticRevisionPublisher(
            store=self.market_data,
            resolver=self.artifact_resolver,
        )
        return WorkspaceMaintenanceCoordinator(
            market_data=self.market_data,
            feature_state=self.feature_state,
            panel_state=self.panel_state,
            manifest=self.manifest,
            provider=self.provider,
            mutation_gate=self.mutation_gate,
            readiness_gate=readiness_gate,
            feature_foundation=self.feature_foundation,
            registry=self.maintenance_registry,
            feature_input=governance,
            snapshot_publisher=self.panel_snapshot_publisher(),
            diagnose=diagnose,
            publish_adjusted_return_revision=adjusted_return_revisions.refresh,
            clock=clock,
            agent_budget=agent_budget or AgentExecutionBudget(),
        )

    def panel_snapshot_publisher(self) -> FeaturePanelSnapshotPublisher:
        """Compose the one panel snapshot writer under this runtime's authority."""
        return FeaturePanelSnapshotPublisher(
            feature_state=self.feature_state,
            panel_state=self.panel_state,
            resolver=self.artifact_resolver,
            mutation_gate=self.mutation_gate,
            recovery_binding=PanelRecoveryBindingPublisher(
                ledger=self.closure_ledger,
                resolver=self.artifact_resolver,
            ),
            logical_identity=PanelLogicalIdentityPublisher(
                resolver=self.artifact_resolver,
                store=PanelLogicalArtifactStore(PanelClosureArtifactStore(self.artifact_resolver)),
                closure_ledger=self.closure_ledger,
            ),
            catalog=self.feature_catalog,
            kernel_registry=self.feature_kernels,
        )

    def pre_research_delta_service(self, *, profile_path: Path) -> PreResearchDeltaService:
        """Return the startup delta adapter under this runtime's writer authority."""
        return PreResearchDeltaService(
            market_data=self.market_data,
            feature_state=self.feature_state,
            panel_state=self.panel_state,
            profile_path=profile_path,
            artifact_root=self.artifact_resolver.root,
            mutation_gate=self.mutation_gate,
            writer_lease=self.writer_lease,
            progress_publisher=self.progress_publisher,
        )

    def close(self) -> None:
        """Release the workspace writer role after runner/checkpoint shutdown."""
        if self.owns_writer_lease:
            self.writer_lease.close()
