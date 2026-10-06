"""Immutable, year-chunked Sector-Neutral Panel snapshot publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Protocol, cast

import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor, ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.layer import FeatureCatalogLayer
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_BINDING,
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    PanelCrossSectionRange,
    PanelTemporalRisk,
    TemporalKnowledgeBoundary,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
    PreparedPanelComposition,
    manifest_partition_origins,
    manifest_row_identity_basis,
)
from alphalattice.foundation.feature_engine.panels.history import (
    derive_feature_panel_history_eligibility,
)
from alphalattice.foundation.feature_engine.panels.recovery_binding import (
    PanelRecoveryBindingPublisher,
)
from alphalattice.foundation.feature_engine.panels.semantic_index import (
    FeaturePanelSemanticIndexService,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    catalog_implementation_hashes,
    catalog_methodology_hashes,
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
    PanelClippingEvidence,
    PanelPreprocessingBinding,
    PanelPreprocessingSealMarker,
)
from alphalattice.foundation.feature_engine.storage.contracts import PanelContentIdentity
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.kernel.quant.sector_history import SectorReclassification
from alphalattice.kernel.shared_kernel.sector_treatment import sector_treatment


@dataclass(frozen=True)
class FeaturePanelChunk:
    """One persisted yearly Panel chunk and the identity of its source rows."""

    year: int
    first_session: date
    last_session: date
    row_count: int
    chunk_hash: str
    metadata_hash: str
    uri: str
    # The binding of the build that wrote the chunk; ``chunk_hash`` is
    # evaluated with it. Absent in every manifest written before partition
    # reuse, where it is the snapshot's own binding, and carried exactly as
    # recorded when an existing manifest is re-sourced.
    origin_binding_hash: str | None = None
    # Under the session-cross-section rule: the identity each range of the
    # chunk's sessions was computed over, with its member count. Absent from
    # every manifest under the binding rule and carried exactly as recorded
    # when an existing manifest is re-sourced.
    cross_sections: tuple[PanelCrossSectionRange, ...] = ()

    def to_payload(self) -> dict[str, object]:
        """Serialize chunk sessions and include only recorded optional bindings."""
        payload = asdict(self)
        payload["first_session"] = self.first_session.isoformat()
        payload["last_session"] = self.last_session.isoformat()
        if self.origin_binding_hash is None:
            del payload["origin_binding_hash"]
        if self.cross_sections:
            payload["cross_sections"] = [item.to_payload() for item in self.cross_sections]
        else:
            del payload["cross_sections"]
        return payload


@dataclass(frozen=True)
class FeaturePanelSnapshotManifest:
    """Immutable identity, coverage and chunk index of a published Panel."""

    snapshot_hash: str
    panel_binding_hash: str
    panel_content_hash: str
    history_start: date
    as_of_session: date
    knowledge_cutoff_at: datetime
    temporal_identity_hash: str
    active_listing_count: int
    listing_set_hash: str
    schema_hash: str
    chunks: tuple[FeaturePanelChunk, ...]
    safe_summary: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        """Serialize the manifest with ISO dates and its named snapshot kind."""
        return {
            "snapshot_hash": self.snapshot_hash,
            "kind": "FeaturePanelSnapshotManifest",
            "panel_binding_hash": self.panel_binding_hash,
            "panel_content_hash": self.panel_content_hash,
            "history_start": self.history_start.isoformat(),
            "as_of_session": self.as_of_session.isoformat(),
            "knowledge_cutoff_at": self.knowledge_cutoff_at.isoformat(),
            "temporal_identity_hash": self.temporal_identity_hash,
            "active_listing_count": self.active_listing_count,
            "listing_set_hash": self.listing_set_hash,
            "schema_hash": self.schema_hash,
            "chunks": [item.to_payload() for item in self.chunks],
            "safe_summary": self.safe_summary,
        }


@dataclass(frozen=True)
class PublishedFeaturePanelSnapshot:
    """Published Panel manifest, artifact descriptor and observation time."""

    manifest: FeaturePanelSnapshotManifest
    artifact: ArtifactDescriptor
    materialized_at: datetime


@dataclass(frozen=True)
class _FeaturePanelSnapshotSource:
    active: dict[str, object]
    temporal_risk: PanelTemporalRisk
    temporal_boundary: TemporalKnowledgeBoundary
    content: PanelContentIdentity
    listing_ids: tuple[str, ...]
    chunks: tuple[FeaturePanelChunk, ...]
    schema_hash: str
    availability: list[dict[str, object]]
    # binding -> {spy_revision, materialization_receipt_hash, and under the
    # cross-section rule manifest_revision, sector_revision} for every build
    # whose cells the Panel holds; None when re-sourcing a manifest that
    # recorded none, so its payload is reproduced exactly.
    partition_origins: dict[str, dict[str, object]] | None
    # Which row-identity rule the rows follow, and under the cross-section
    # rule the list-free membership record the composition sealed (or the
    # one the re-sourced manifest recorded).
    row_identity_basis: str = PANEL_ROW_IDENTITY_BY_BINDING
    membership: dict[str, object] | None = None
    # The reclassifications the Panel's sessions read (V346), over its calculation axis.
    sector_reclassifications: tuple[SectorReclassification, ...] = ()


class PanelLogicalPublicationOwner(Protocol):
    """Publish and read back native logical/physical identity before activation."""

    def publish_snapshot(
        self, manifest_ref: str, *, published_at: datetime | None = None
    ) -> object:
        """Publish logical and physical identity for a native Panel snapshot."""
        ...


def resolve_panel_preprocessing_lineage(
    resolver: ArtifactResolver, *, panel_content_hash: str
) -> dict[str, object] | None:
    """Resolve, from a published Panel, the method that actually built it.

    Walks marker -> binding -> evidence and re-derives each document's identity
    from its own fields, because agreement between two documents proves only
    that one writer produced both. The binding is the edge that used to be
    missing: both the marker and the receipt quoted ``preprocessing_binding_hash``
    at each other while nothing held the binding those hashes named, so a
    self-consistent forged pair resolved cleanly. Loading it turns that hash into
    a claim that can fail.

    Returns ``None`` only when no marker exists -- a Panel built before the seam.
    That is legacy readback and never an active method. A marker naming a binding
    this workspace does not hold is a different thing entirely: a broken graph,
    which raises.
    """
    payload = resolver.load_panel_preprocessing_marker(panel_content_hash)
    if payload is None:
        return None
    marker = PanelPreprocessingSealMarker.model_validate(payload)
    if marker.panel_content_hash != panel_content_hash:
        raise ValueError("panel preprocessing marker does not name this panel")

    binding_payload = resolver.load_panel_preprocessing_binding(marker.preprocessing_binding_hash)
    if binding_payload is None:
        raise ValueError("panel preprocessing binding named by the marker is missing")
    binding = PanelPreprocessingBinding.model_validate(binding_payload)
    if (
        binding.binding_hash != marker.preprocessing_binding_hash
        or binding.panel_binding_hash != marker.panel_binding_hash
        or binding.recipe_id != marker.recipe_id
        or binding.recipe_hash != marker.recipe_hash
        or binding.implementation_id != marker.implementation_id
        or binding.implementation_binding_hash != marker.implementation_binding_hash
        or binding.implementation.implementation_binding_hash != marker.implementation_binding_hash
        or binding.catalog_hash != marker.catalog_hash
    ):
        raise ValueError("panel preprocessing binding contradicts the marker that names it")

    evidence = PanelClippingEvidence.model_validate(
        resolver.load_panel_clipping_evidence(
            resolver.panel_clipping_evidence_uri(marker.clipping_evidence_hash)
        )
    )
    if (
        evidence.preprocessing_binding_hash != binding.binding_hash
        or evidence.panel_binding_hash != binding.panel_binding_hash
        or evidence.recipe_id != binding.recipe_id
        or evidence.recipe_hash != binding.recipe_hash
    ):
        raise ValueError("panel preprocessing evidence does not describe this panel")

    return {
        "recipe_id": marker.recipe_id,
        "recipe_hash": marker.recipe_hash,
        "implementation_id": marker.implementation_id,
        "implementation_binding_hash": marker.implementation_binding_hash,
        # The lineage terminates in facts rather than in another digest: a reader
        # holding this can say which modules were hashed and which versions were
        # pinned, without holding any further document.
        "implementation_owners": list(binding.implementation.implementation_owners),
        "implementation_content_hash": binding.implementation.implementation_content_hash,
        # A binding sealed before E0 carries the environment it folded in; a new one
        # carries none, the environment being the closure receipt's provenance.
        **(
            {}
            if binding.implementation.numerical_environment_hash is None
            else {"numerical_environment_hash": binding.implementation.numerical_environment_hash}
        ),
        "catalog_hash": marker.catalog_hash,
        "policy_hash": binding.policy_hash,
        "preprocessing_binding_hash": marker.preprocessing_binding_hash,
        "clipping_evidence_hash": marker.clipping_evidence_hash,
        "preprocessing_identity": marker.preprocessing_identity,
    }


def reconcile_feature_panel_lifecycles(
    *,
    panel_state: PanelStateRepository,
    resolver: ArtifactResolver,
    mutation_gate: WorkspaceMutationGate,
    observed_at: datetime,
) -> None:
    """Feature's one Panel lifecycle function: the rows, then the projection research reads.

    Market Data's bootstrap no longer supersedes Panels by its own SQL; a Feature build runs this
    after it, so a publication that failed leaves no row its projection contradicts (V176).

    Args:
        panel_state: The Panel's lifecycle rows.
        resolver: Where the read-side projection is published.
        mutation_gate: The workspace's one writer.
        observed_at: When the rule ran.
    """
    mutation_gate.run(
        panel_state.supersede_unbound_feature_panel_snapshots, observed_at=observed_at
    )
    resolver.publish_feature_panel_lifecycle_projection(
        snapshots=panel_state.feature_panel_snapshot_lifecycles()
    )


class FeaturePanelSnapshotPublisher:
    """Trusted publisher; mutable DuckDB and physical paths stay behind it."""

    def __init__(
        self,
        *,
        feature_state: FeatureStateRepository,
        panel_state: PanelStateRepository,
        resolver: ArtifactResolver,
        mutation_gate: WorkspaceMutationGate,
        recovery_binding: PanelRecoveryBindingPublisher,
        logical_identity: PanelLogicalPublicationOwner,
        panel_artifacts: PanelArtifactCompositionOwner | None = None,
        catalog: FeatureCatalog | None = None,
        kernel_registry: FeatureKernelRegistry | None = None,
        semantic_index: FeaturePanelSemanticIndexService | None = None,
    ) -> None:
        """Compose the verified Panel source and its publication owners."""
        self.feature_state = feature_state
        self.panel_state = panel_state
        self.resolver = resolver
        self.mutation_gate = mutation_gate
        self.recovery_binding = recovery_binding
        self.logical_identity = logical_identity
        self.panel_artifacts = panel_artifacts or PanelArtifactCompositionOwner(resolver)
        # The existing semantic-index owner, composed rather than reimplemented:
        # publication is not complete until the Panel can be resolved, and a
        # second writer for the same artifact would be a second identity.
        self.semantic_index = semantic_index or FeaturePanelSemanticIndexService(resolver)
        self.catalog = catalog if catalog is not None else FeatureCatalog.load()
        self.layer = FeatureCatalogLayer.over(self.catalog)
        self.kernel_registry = (
            kernel_registry if kernel_registry is not None else default_extension_kernel_registry()
        )

    def publish(
        self,
        *,
        manifest: UniverseManifest,
        history_start: date,
        as_of_session: date,
        knowledge_cutoff_at: datetime | None = None,
        observed_at: datetime | None = None,
    ) -> PublishedFeaturePanelSnapshot:
        """Publish a verified immutable Panel snapshot for the requested history.

        Raises:
            ValueError: Publication time is naive or the source cannot be verified.

        """
        now = observed_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("snapshot publication time must be timezone-aware")
        # Before anything is composed: a row of the right shape carrying another
        # catalog would be published as this catalog's output without any method
        # having recomputed it. Rows the rotation left behind are absent from the
        # runtime view and recomputed by maintenance; rows that are present and
        # wrong are refused here, by identity, not by count.
        self.feature_state.assert_installed_catalog_rows(
            catalog_hash=self.catalog.binding.catalog_hash
        )
        source = self._read_consistent_source(
            manifest=manifest,
            history_start=history_start,
            as_of_session=as_of_session,
            knowledge_cutoff_at=knowledge_cutoff_at,
            materialized_at=now,
        )
        active = source.active
        temporal_risk = source.temporal_risk
        temporal_boundary = source.temporal_boundary
        content = source.content
        listing_ids = source.listing_ids
        chunks = list(source.chunks)
        schema_hash = source.schema_hash
        availability = source.availability
        cutoff = temporal_boundary.knowledge_cutoff_at
        sector = self.feature_state.current_sector_state(manifest)
        if sector is None:
            raise ValueError("sector state disappeared after snapshot publication read")
        # Under the binding rule the listing axis is the manifest's. Under the
        # cross-section rule it is the calculation axis: every listing some
        # session holds, as the composition recorded it. The latest manifest
        # may also admit future-effective members; ``membership`` records the
        # actual dated population rather than backdating those admissions.
        listing_set_hash = canonical_hash(listing_ids)
        preprocessing_lineage = resolve_panel_preprocessing_lineage(
            self.resolver, panel_content_hash=content.panel_content_hash
        )
        factor_summary = _factor_summary(
            availability,
            factor_ids=self.catalog.factor_ids,
            implementation_hashes=catalog_implementation_hashes(
                self.catalog, registry=self.kernel_registry
            ),
            methodology_hashes=catalog_methodology_hashes(
                self.catalog, registry=self.kernel_registry
            ),
            observation_clock_hashes={
                factor_id: clock.clock_hash
                for factor_id, clock in self.catalog.clocks_by_factor.items()
            },
            source_authority_ids=self.catalog.formula_source_authorities,
        )
        history_eligibility = derive_feature_panel_history_eligibility(
            availability,
            factor_ids=self.catalog.factor_ids,
        )
        temporal_payload = {
            **asdict(temporal_risk),
            "sector_observed_at": temporal_risk.sector_observed_at.isoformat(),
        }
        temporal_identity_payload = {
            **temporal_boundary.identity_payload(),
            "temporal_identity_hash": temporal_boundary.identity_hash(),
        }
        quality_governance = self.panel_state.feature_input_quality_disclosure(
            result_manifest_revision=manifest.revision_sha256
        )
        safe_summary: dict[str, object] = {
            "date_range": [history_start.isoformat(), as_of_session.isoformat()],
            "as_of_session": as_of_session.isoformat(),
            "active_listing_count": len(listing_ids),
            "listing_set_hash": listing_set_hash,
            "factor_catalog_summary": factor_summary,
            "model_history_eligibility": history_eligibility.safe_summary(),
            "sector_distribution": sector.sector_distribution,
            "temporal_risk": temporal_payload,
            "temporal_boundary": temporal_identity_payload,
            "quality_governance": quality_governance,
            "universe_policy": {
                "type": manifest.universe_policy_type,
                "components": list(manifest.universe_components),
                "survivorship_bias_warning": manifest.survivorship_bias_warning,
                "research_use_class": manifest.research_use_class,
            },
            "lineage": {
                "manifest_revision": str(active["manifest_revision"]),
                "sector_revision": str(active["sector_revision"]),
                "catalog_hash": str(active["catalog_hash"]),
                "spy_revision": str(active["spy_revision"]),
                "policy_hash": str(active["policy_hash"]),
                "panel_binding_hash": str(active["panel_binding_hash"]),
                "panel_content_hash": content.panel_content_hash,
                # Bound into the snapshot payload, which is what ``snapshot_hash``
                # is computed over, so Panel identity depends on the preprocessing
                # methodology. Without it, two Panels whose numbers coincide are
                # indistinguishable even when different methods produced them --
                # and a record claiming identity "rotates" would describe a
                # property the manifest formula did not actually have.
                "preprocessing": preprocessing_lineage,
                # Both clock authorities, bound separately because they answer
                # different questions and change for different reasons. The
                # Formula observation authority says which session each value
                # belongs to; the source availability authority says when such a
                # value may be read. A terminal Panel needs both, and a reader
                # must be able to tell which one moved.
                "formula_observation_policy_hash": (
                    self.catalog.binding.formula_observation_policy_hash
                ),
                "source_availability": self.catalog.source_availability.model_dump(mode="json"),
                # Every installed owner and the field map that resolves them, so
                # a reader can re-derive each Formula's authority set rather than
                # trust the per-Factor list above, and so an owner's schedule
                # change is visible as a change to this and to nothing else.
                "source_authorities": self.catalog.source_authorities.model_dump(mode="json"),
                "source_authority_binding_hash": (
                    self.catalog.binding.source_authority_binding_hash
                ),
            },
            "row_count": content.row_count,
            "availability_count": content.availability_count,
            "chunk_count": len(chunks),
            "schema_hash": schema_hash,
        }
        sector_progress = self.feature_state.sector_progress(manifest.revision_sha256)
        if sector_progress and sector_progress.get("status") == "DEFERRED":
            failed_at = cast(datetime, sector_progress["updated_at"])
            safe_summary["sector_reference"] = {
                "status": "PRIOR_VERIFIED_REFERENCE",
                "sector_revision": sector.sector_revision,
                "source_observed_at": sector.sector_observed_at.isoformat(),
                "refresh_failed_at": failed_at.replace(tzinfo=UTC).isoformat(),
                "failure_code": sector_progress["failure_code"],
            }
        if source.sector_reclassifications:
            # Each reclassification the sessions read from its effective session (V346);
            # absent while none is in force, so such a Panel's lineage is what it was.
            lineage = cast(dict[str, object], safe_summary["lineage"])
            lineage["sector_reclassifications"] = [
                {
                    "listing_id": item.listing_id,
                    "effective_session": item.effective_session.isoformat(),
                    "prior_sector": item.prior_sector,
                    "sector": item.sector,
                }
                for item in source.sector_reclassifications
            ]
        if source.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
            lineage = cast(dict[str, object], safe_summary["lineage"])
            lineage["row_identity_basis"] = source.row_identity_basis
            if source.membership is None:
                raise ValueError("feature panel under the cross-section rule has no membership")
            # The list-free membership record: which promise each range of
            # sessions is under, the epochs with their member counts and
            # identities, and the Universe record they resolve from. The
            # member lists themselves live once, in the Universe journal.
            safe_summary["membership"] = dict(source.membership)
        if source.partition_origins is not None:
            # Provenance of every partition and cell: which builds computed
            # them. The binding-in-the-name rule kept this implicitly (one
            # build per snapshot); a snapshot that reuses partitions says it.
            lineage = cast(dict[str, object], safe_summary["lineage"])
            lineage["partition_origins"] = {
                key: dict(value) for key, value in sorted(source.partition_origins.items())
            }
            own_binding = str(active["panel_binding_hash"])
            safe_summary["partition_reuse"] = {
                "reused_partition_count": sum(
                    1 for item in chunks if item.origin_binding_hash not in {None, own_binding}
                ),
                "composed_partition_count": sum(
                    1 for item in chunks if item.origin_binding_hash in {None, own_binding}
                ),
            }
        manifest_payload = {
            "kind": "FeaturePanelSnapshotManifest",
            "panel_binding_hash": str(active["panel_binding_hash"]),
            "panel_content_hash": content.panel_content_hash,
            "history_start": history_start.isoformat(),
            "as_of_session": as_of_session.isoformat(),
            "knowledge_cutoff_at": cutoff.isoformat(),
            "temporal_identity_hash": temporal_boundary.identity_hash(),
            "active_listing_count": len(listing_ids),
            "listing_set_hash": listing_set_hash,
            "schema_hash": schema_hash,
            "chunks": [item.to_payload() for item in chunks],
            "safe_summary": safe_summary,
        }
        snapshot_hash = canonical_hash(manifest_payload)
        snapshot_manifest = FeaturePanelSnapshotManifest(
            snapshot_hash=snapshot_hash,
            panel_binding_hash=str(active["panel_binding_hash"]),
            panel_content_hash=content.panel_content_hash,
            history_start=history_start,
            as_of_session=as_of_session,
            knowledge_cutoff_at=cutoff,
            temporal_identity_hash=temporal_boundary.identity_hash(),
            active_listing_count=len(listing_ids),
            listing_set_hash=listing_set_hash,
            schema_hash=schema_hash,
            chunks=tuple(chunks),
            safe_summary=safe_summary,
        )
        descriptor = self.resolver.publish_feature_panel_manifest(
            payload=snapshot_manifest.to_payload(), snapshot_hash=snapshot_hash
        )
        self.recovery_binding.publish(
            snapshot_hash=snapshot_hash,
            panel_content_hash=content.panel_content_hash,
            panel_binding_hash=str(active["panel_binding_hash"]),
            catalog_hash=str(active["catalog_hash"]),
            sector_revision=str(active["sector_revision"]),
            listing_ids=listing_ids,
            factor_ids=self.catalog.factor_ids,
            chunks=tuple(item.to_payload() for item in chunks),
            panel_source_state_hash=str(active["source_state_hash"]),
            parts=(
                self.layer.part_hashes
                if self.layer.layered
                and str(active["catalog_hash"]) == self.catalog.binding.catalog_hash
                else ()
            ),
        )
        self.logical_identity.publish_snapshot(descriptor.uri, published_at=now)
        self.mutation_gate.run(
            self.panel_state.register_feature_panel_snapshot,
            snapshot_hash=snapshot_hash,
            panel_binding_hash=str(active["panel_binding_hash"]),
            panel_content_hash=content.panel_content_hash,
            manifest_uri=descriptor.uri,
            metadata_hash=descriptor.metadata_hash,
            history_start=history_start,
            as_of_session=as_of_session,
            knowledge_cutoff_at=cutoff,
            temporal_identity_hash=temporal_boundary.identity_hash(),
            active_listing_count=len(listing_ids),
            chunk_count=len(chunks),
            observed_at=now,
        )
        self.resolver.publish_feature_panel_lifecycle_projection(
            snapshots=self.panel_state.feature_panel_snapshot_lifecycles()
        )
        # Terminal publication ends here, not at the manifest. The research
        # authority resolver needs a semantic index to name a session axis, and
        # until now a freshly published Panel had none, so it refused with
        # `snapshot_handle_unresolved` until somebody called this by hand. The
        # index owner derives its identity from the durable manifest that was
        # just verified -- no caller-supplied hash -- and `obtain` returns the
        # existing index unchanged, so replay produces no second one. It runs
        # after the lifecycle projection because the reader it scans through is
        # ACTIVE-gated.
        #
        # Only for a gateway-qualified Panel. Without a Feature Input Gateway
        # admission the reader refuses by design -- research must not read a
        # Panel whose inputs were never admitted -- and building an index anyway
        # would defeat that gate rather than complete a handoff. Such a Panel is
        # unresolvable for a reason, not for a missing artifact.
        if bool(quality_governance.get("gateway_qualified")):
            self.semantic_index.obtain(descriptor.uri)
            if (
                self.resolver.find_feature_panel_semantic_index(panel_snapshot_hash=snapshot_hash)
                is None
            ):
                raise ValueError("feature_panel.semantic_index_handoff_incomplete")
        return PublishedFeaturePanelSnapshot(snapshot_manifest, descriptor, now)

    def complete_semantic_index_handoff(self, *, manifest: UniverseManifest) -> str | None:
        """Finish an already-published Panel's handoff, and nothing else.

        Two situations leave a Panel published and unresolvable, and both are
        repaired by exactly this: publication that ran before the Feature Input
        Gateway admitted the inputs, where withholding the index was correct at
        the time, and a handoff that failed after the lifecycle projection.

        No Feature value is recomputed and none is republished. The Panel's
        immutable manifest and chunks already exist and are the input; the index
        owner reads row identities out of them and derives its own identity from
        the durable manifest. Nothing here touches ``feature_daily_runtime``,
        composes a Panel, or moves any lifecycle. Recovering a missing index by
        rebuilding the Panel would discard correct work to produce bytes that are
        already correct.

        Returns the snapshot the handoff completed for, or ``None`` when there is
        nothing to do -- no active snapshot, an index that already resolves, or
        inputs the Gateway has still not admitted. The last of those is the gate
        doing its job, not a repair this may force.
        """
        snapshot = self.panel_state.feature_panel_snapshot_for_active(
            manifest.profile.market_profile_id
        )
        if snapshot is None:
            return None
        snapshot_hash = str(snapshot["snapshot_hash"])
        if (
            self.resolver.find_feature_panel_semantic_index(panel_snapshot_hash=snapshot_hash)
            is not None
        ):
            return None
        disclosure = self.panel_state.feature_input_quality_disclosure(
            result_manifest_revision=manifest.revision_sha256
        )
        if not bool(disclosure.get("gateway_qualified")):
            return None
        self.semantic_index.obtain(str(snapshot["manifest_uri"]))
        if (
            self.resolver.find_feature_panel_semantic_index(panel_snapshot_hash=snapshot_hash)
            is None
        ):
            raise ValueError("feature_panel.semantic_index_handoff_incomplete")
        return snapshot_hash

    def quarantine(
        self,
        *,
        snapshot_hash: str,
        reason: str,
        observed_at: datetime | None = None,
    ) -> None:
        """Fail closed on one unsafe snapshot before changing mutable authority."""
        now = observed_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("snapshot lifecycle time must be timezone-aware")
        projected = []
        found = False
        for item in self.panel_state.feature_panel_snapshot_lifecycles():
            entry = dict(item)
            if entry["snapshot_hash"] == snapshot_hash:
                entry["lifecycle"] = "QUARANTINED"
                entry["reason"] = reason
                found = True
            projected.append(entry)
        if not found:
            raise KeyError(f"unknown feature panel snapshot: {snapshot_hash}")
        self.resolver.publish_feature_panel_lifecycle_projection(snapshots=projected)
        self.mutation_gate.run(
            self.panel_state.set_feature_panel_snapshot_lifecycle,
            snapshot_hash=snapshot_hash,
            lifecycle="QUARANTINED",
            reason=reason,
            observed_at=now,
        )
        self.resolver.publish_feature_panel_lifecycle_projection(
            snapshots=self.panel_state.feature_panel_snapshot_lifecycles()
        )

    def _read_consistent_source(
        self,
        *,
        manifest: UniverseManifest,
        history_start: date,
        as_of_session: date,
        knowledge_cutoff_at: datetime | None,
        materialized_at: datetime,
    ) -> _FeaturePanelSnapshotSource:
        with self.panel_state.feature_panel_read_transaction() as connection:
            active = self.panel_state.active_feature_panel(
                manifest.profile.market_profile_id, _connection=connection
            )
            if active is None:
                raise ValueError("feature panel snapshot requires an active panel")
            required = (
                "manifest_revision",
                "sector_revision",
                "catalog_hash",
                "spy_revision",
                "policy_hash",
                "panel_binding_hash",
                "panel_content_hash",
                "history_start",
                "as_of_session",
                "temporal_risk_hash",
                "temporal_identity_hash",
                "knowledge_cutoff_at",
                "source_state_hash",
            )
            if any(active.get(field) in {None, ""} for field in required):
                raise ValueError("active feature panel predates content-identity closure")
            active_cutoff_value = active["knowledge_cutoff_at"]
            assert isinstance(active_cutoff_value, datetime)
            active_cutoff = (
                active_cutoff_value.replace(tzinfo=UTC)
                if active_cutoff_value.tzinfo is None
                else active_cutoff_value.astimezone(UTC)
            )
            cutoff = knowledge_cutoff_at or active_cutoff
            if cutoff.tzinfo is None or cutoff.astimezone(UTC) != active_cutoff:
                raise ValueError("snapshot knowledge cutoff does not match the active panel")
            if active["manifest_revision"] != manifest.revision_sha256:
                raise ValueError("feature panel manifest revision is stale")
            if history_start < active["history_start"] or as_of_session > active["as_of_session"]:
                raise ValueError("snapshot request is outside the active panel range")
            sector = self.feature_state.current_sector_state(manifest, _connection=connection)
            if sector is None or sector.sector_revision != active["sector_revision"]:
                raise ValueError("active sector revision does not match the feature panel")
            temporal_boundary = TemporalKnowledgeBoundary(
                market_as_of_session=as_of_session,
                knowledge_cutoff_at=cutoff,
                materialized_at=materialized_at,
                universe_source_observed_at=datetime.combine(
                    manifest.profile.manifest_as_of,
                    datetime.min.time(),
                    tzinfo=UTC,
                ),
                sector_source_observed_at=sector.sector_observed_at,
                universe_point_in_time_qualified=manifest.is_point_in_time_historical,
                sector_point_in_time_qualified=False,
            )
            if temporal_boundary.identity_hash() != active["temporal_identity_hash"]:
                raise ValueError(
                    "active panel temporal identity changed before snapshot publication"
                )
            if history_start != active["history_start"] or as_of_session != active["as_of_session"]:
                raise ValueError("artifact-backed snapshot publication requires the active range")
            backed = self._artifact_backed_source(active)
            content, chunks, schema_hash, partition_origins = (
                backed.content,
                backed.chunks,
                backed.schema_hash,
                backed.partition_origins,
            )
            if backed.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
                cross_sections = tuple(item for chunk in chunks for item in chunk.cross_sections)
                if not cross_sections:
                    raise ValueError("feature panel under the cross-section rule has no ranges")
                availability = self.panel_state.panel_availability_rows(
                    cross_sections=cross_sections,
                    catalog_hash=str(active["catalog_hash"]),
                    policy_hash=str(active["policy_hash"]),
                    start=history_start,
                    end=as_of_session,
                    _connection=connection,
                )
            else:
                availability = self.panel_state.panel_availability_rows(
                    manifest_revision=str(active["manifest_revision"]),
                    sector_revision=str(active["sector_revision"]),
                    catalog_hash=str(active["catalog_hash"]),
                    policy_hash=str(active["policy_hash"]),
                    start=history_start,
                    end=as_of_session,
                    _connection=connection,
                )
        active_after = self.panel_state.active_feature_panel(manifest.profile.market_profile_id)
        if active_after is None or _active_source_identity(active_after) != _active_source_identity(
            active
        ):
            raise ValueError("active panel source changed during snapshot publication")
        sector_after = self.feature_state.current_sector_state(manifest)
        if sector_after is None or sector_after.sector_revision != active["sector_revision"]:
            raise ValueError("sector source changed during snapshot publication")
        if backed.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
            listing_ids = backed.listing_ids
            if (
                backed.membership is None
                or backed.membership.get("axis_listing_count") != len(listing_ids)
                or backed.membership.get("axis_listing_set_hash") != canonical_hash(listing_ids)
            ):
                raise ValueError("feature panel calculation axis differs from recorded membership")
        else:
            listing_ids = tuple(sorted(item.listing_id for item in manifest.listings))
        # The Sector each session read, over the axis the build read it for (V346).
        history = self.feature_state.sector_history(manifest, listing_ids=listing_ids)
        if history is None:
            raise ValueError("sector source changed during snapshot publication")
        temporal_risk = PanelTemporalRisk.current_yahoo(
            sector_observed_at=sector.sector_observed_at,
            sector_history_treatment=sector_treatment(reclassified=bool(history.reclassifications)),
        )
        if temporal_risk.risk_hash() != active["temporal_risk_hash"]:
            raise ValueError("feature panel temporal risk contract changed")
        return _FeaturePanelSnapshotSource(
            active=active,
            temporal_risk=temporal_risk,
            temporal_boundary=temporal_boundary,
            content=content,
            listing_ids=listing_ids,
            chunks=tuple(chunks),
            schema_hash=schema_hash,
            availability=availability,
            partition_origins=partition_origins,
            row_identity_basis=backed.row_identity_basis,
            membership=backed.membership,
            sector_reclassifications=history.reclassifications,
        )

    def _artifact_backed_source(self, active: Mapping[str, object]) -> _ArtifactBackedSource:
        snapshot = self.panel_state.feature_panel_snapshot_for_active(
            str(active["market_profile_id"])
        )
        if snapshot is not None:
            payload = self.resolver.load_feature_panel_manifest(str(snapshot["manifest_uri"]))
            if (
                payload.get("panel_binding_hash") != active["panel_binding_hash"]
                or payload.get("panel_content_hash") != active["panel_content_hash"]
            ):
                raise ValueError("active feature Panel snapshot identity changed")
            return _source_identity_from_manifest(payload, resolver=self.resolver)
        prepared = self.panel_artifacts.load(
            panel_binding_hash=str(active["panel_binding_hash"]),
            panel_content_hash=str(active["panel_content_hash"]),
        )
        content, chunks, schema_hash = _source_identity_from_composition(prepared)
        return _ArtifactBackedSource(
            content=content,
            chunks=chunks,
            schema_hash=schema_hash,
            partition_origins=self._partition_origins(prepared, active),
            row_identity_basis=prepared.binding.row_identity_basis,
            membership=(
                prepared.membership.summary_payload() if prepared.membership is not None else None
            ),
            listing_ids=tuple(sorted(prepared.listing_ids)),
        )

    def _partition_origins(
        self, prepared: PreparedPanelComposition, active: Mapping[str, object]
    ) -> dict[str, dict[str, object]]:
        """Resolve the composition's origins to SPY revisions and build receipts.

        The composition names each binding and, for those it knew, the SPY
        revision. This build's own binding resolves through the active Panel
        row (which also holds its materialization receipt); an earlier
        binding's receipt is carried from the snapshot it was recorded in --
        the one still ACTIVE before this publication supersedes it -- and is
        None when no such record exists. The receipt is provenance, never an
        input to recovery.
        """
        own_binding = str(active["panel_binding_hash"])
        previous: dict[str, dict[str, object]] = {}
        for item in self.panel_state.feature_panel_snapshot_lifecycles():
            if item.get("lifecycle") != "ACTIVE":
                continue
            manifest_uri = self.resolver.feature_panel_manifest_uri(str(item["snapshot_hash"]))
            try:
                previous.update(
                    manifest_partition_origins(
                        self.resolver.load_feature_panel_manifest(manifest_uri)
                    )
                )
            except (FileNotFoundError, ValueError):
                continue
        origins: dict[str, dict[str, object]] = {}
        for binding_hash, record in prepared.partition_origins.items():
            if binding_hash == own_binding:
                origins[binding_hash] = {
                    "spy_revision": str(active["spy_revision"]),
                    "materialization_receipt_hash": active.get("materialization_receipt_hash"),
                }
                if prepared.binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
                    origins[binding_hash]["manifest_revision"] = str(active["manifest_revision"])
                    origins[binding_hash]["sector_revision"] = str(active["sector_revision"])
                continue
            recorded = previous.get(binding_hash, {})
            resolved_spy = record.spy_revision
            if resolved_spy is None:
                resolved_spy = cast(str | None, recorded.get("spy_revision"))
            if resolved_spy is None:
                raise ValueError(
                    f"feature Panel partition origin {binding_hash} has no SPY revision"
                )
            origins[binding_hash] = {
                "spy_revision": str(resolved_spy),
                "materialization_receipt_hash": recorded.get("materialization_receipt_hash"),
            }
            if record.catalog_hash is not None:
                # Cells a catalog this one only adds columns to computed (V92).
                origins[binding_hash]["catalog_hash"] = record.catalog_hash
            if prepared.binding.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
                manifest_revision = record.manifest_revision or recorded.get("manifest_revision")
                sector_revision = record.sector_revision or recorded.get("sector_revision")
                if manifest_revision is None or sector_revision is None:
                    raise ValueError(
                        f"feature Panel partition origin {binding_hash} has no recorded revisions"
                    )
                origins[binding_hash]["manifest_revision"] = str(manifest_revision)
                origins[binding_hash]["sector_revision"] = str(sector_revision)
        return origins


def _active_source_identity(active: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        active.get(field)
        for field in (
            "manifest_revision",
            "sector_revision",
            "catalog_hash",
            "spy_revision",
            "policy_hash",
            "panel_binding_hash",
            "panel_content_hash",
            "source_state_hash",
            "history_start",
            "as_of_session",
            "temporal_identity_hash",
        )
    )


@dataclass(frozen=True)
class _ArtifactBackedSource:
    content: PanelContentIdentity
    chunks: list[FeaturePanelChunk]
    schema_hash: str
    partition_origins: dict[str, dict[str, object]] | None
    row_identity_basis: str
    membership: dict[str, object] | None
    listing_ids: tuple[str, ...]


def _source_identity_from_manifest(
    payload: Mapping[str, object], *, resolver: ArtifactResolver
) -> _ArtifactBackedSource:
    safe_summary = payload.get("safe_summary")
    if not isinstance(safe_summary, Mapping):
        raise ValueError("feature Panel snapshot has no safe summary")
    chunks_payload = payload.get("chunks")
    if not isinstance(chunks_payload, list) or not chunks_payload:
        raise ValueError("feature Panel snapshot has no chunks")
    chunks = [_feature_panel_chunk(item) for item in chunks_payload]
    content = PanelContentIdentity(
        panel_content_hash=str(payload["panel_content_hash"]),
        row_count=int(str(safe_summary["row_count"])),
        availability_count=int(str(safe_summary["availability_count"])),
        history_start=date.fromisoformat(str(payload["history_start"])),
        as_of_session=date.fromisoformat(str(payload["as_of_session"])),
    )
    lineage = safe_summary.get("lineage")
    recorded = lineage.get("partition_origins") if isinstance(lineage, Mapping) else None
    origins = manifest_partition_origins(payload) if isinstance(recorded, Mapping) else None
    basis = manifest_row_identity_basis(payload)
    membership = safe_summary.get("membership")
    listing_ids: tuple[str, ...] = ()
    if basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
        if not isinstance(membership, Mapping):
            raise ValueError(
                "feature Panel snapshot under the cross-section rule has no membership"
            )
        # The axis is not listed in the manifest; the rows hold it. Read
        # once from the identity columns, which the reader projects anyway.
        listing_ids = _listing_axis_from_chunks(chunks, payload, resolver=resolver)
    return _ArtifactBackedSource(
        content=content,
        chunks=chunks,
        schema_hash=str(payload["schema_hash"]),
        partition_origins=origins,
        row_identity_basis=basis,
        membership=dict(membership) if isinstance(membership, Mapping) else None,
        listing_ids=listing_ids,
    )


def _listing_axis_from_chunks(
    chunks: list[FeaturePanelChunk],
    payload: Mapping[str, object],
    *,
    resolver: ArtifactResolver,
) -> tuple[str, ...]:
    """Return the sorted listing axis held by a re-sourced manifest's rows.

    Re-derived rather than trusted: the manifest's ``listing_set_hash`` must
    be the hash of exactly this axis, or the manifest is not describing the
    rows it names.
    """
    listing_ids: set[str] = set()
    for chunk in chunks:
        path = resolver.resolve_feature_panel_chunk_ref(
            uri=chunk.uri, content_hash=chunk.chunk_hash, metadata_hash=chunk.metadata_hash
        )
        table = pq.read_table(path, columns=["listing_id"])
        listing_ids.update(str(value) for value in table.column("listing_id").to_pylist())
    axis = tuple(sorted(listing_ids))
    if canonical_hash(axis) != str(payload["listing_set_hash"]):
        raise ValueError("feature Panel snapshot listing axis differs from its rows")
    return axis


def _source_identity_from_composition(
    prepared: PreparedPanelComposition,
) -> tuple[PanelContentIdentity, list[FeaturePanelChunk], str]:
    chunks = [
        FeaturePanelChunk(
            year=item.year,
            first_session=item.first_session,
            last_session=item.last_session,
            row_count=item.row_count,
            chunk_hash=item.chunk_hash,
            metadata_hash=item.metadata_hash,
            uri=item.uri,
            origin_binding_hash=(
                item.origin_binding_hash
                if item.origin_binding_hash is not None
                else prepared.binding.panel_binding_hash
            ),
            cross_sections=item.cross_sections,
        )
        for item in prepared.chunks
    ]
    return prepared.content, chunks, prepared.schema_hash


def _feature_panel_chunk(payload: object) -> FeaturePanelChunk:
    if not isinstance(payload, Mapping):
        raise ValueError("feature Panel chunk entry is invalid")
    origin = payload.get("origin_binding_hash")
    recorded = payload.get("cross_sections") or ()
    if not isinstance(recorded, list | tuple):
        raise ValueError("feature Panel chunk cross-sections are invalid")
    return FeaturePanelChunk(
        year=int(str(payload["year"])),
        first_session=date.fromisoformat(str(payload["first_session"])),
        last_session=date.fromisoformat(str(payload["last_session"])),
        row_count=int(str(payload["row_count"])),
        chunk_hash=str(payload["chunk_hash"]),
        metadata_hash=str(payload["metadata_hash"]),
        uri=str(payload["uri"]),
        origin_binding_hash=str(origin) if origin is not None else None,
        cross_sections=tuple(
            PanelCrossSectionRange.from_payload(cast(Mapping[str, object], item))
            for item in recorded
        ),
    )


def _factor_summary(
    availability: list[dict[str, object]],
    *,
    factor_ids: tuple[str, ...],
    implementation_hashes: Mapping[str, str],
    methodology_hashes: Mapping[str, str],
    observation_clock_hashes: Mapping[str, str],
    source_authority_ids: Mapping[str, tuple[str, ...]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for factor_id in factor_ids:
        items = [item for item in availability if str(item["factor_id"]) == factor_id]
        denominator = sum(int(item["universe_size"]) for item in items)
        computed = sum(int(item["computed_count"]) for item in items)
        result[factor_id] = {
            "available_session_count": sum(item["status"] == "available" for item in items),
            "unavailable_session_count": sum(item["status"] != "available" for item in items),
            "null_ratio": 1.0 - (computed / denominator) if denominator else 1.0,
            "zero_mad_count": sum(
                item["reason"] in {"feature_mad_zero", "residual_mad_zero"} for item in items
            ),
            "coverage_failure_count": sum(
                item["reason"] == "coverage_below_98_percent" for item in items
            ),
            "small_sector_warning_count": sum(bool(item["small_sector_warning"]) for item in items),
            # Which code computed this factor, not merely which ID it answers to.
            # Written by this publisher onward; panels published before it read
            # back unchanged under their own schema, so every consumer treats
            # this as present-or-absent rather than required.
            "implementation_hash": implementation_hashes[factor_id],
            # Which *method*, which is strictly more than which code: the same
            # kernel at a different window or lag is a different factor, and
            # binding only the implementation left that change invisible.
            #
            # This key participates in ``availability_semantics_hash``, so a
            # panel published from here has a different logical revision than one
            # published before it. That is the intended new-writer identity move,
            # not a migration: a panel already on disk recomputes its hashes from
            # its own stored summary and reads back exactly as it always did.
            "methodology_hash": methodology_hashes[factor_id],
            # Which observation session the value belongs to and what it actually
            # consumed. The methodology identity already binds window and skip;
            # this binds the *semantics* those numbers carry -- the interval kind,
            # the newest source event and whether the row minimum is mechanical --
            # none of which lives in a ``FactorSpec``. It participates in
            # ``availability_semantics_hash``, so a Panel published under a
            # different Formula clock cannot present the same logical revision.
            #
            "observation_clock_hash": observation_clock_hashes[factor_id],
            # *Which* source owners this Formula reads -- not when they publish.
            # The schedules live once in the lineage below, so a Provider that
            # changes a publication time does not rotate fifty-three per-Factor
            # identities that compute exactly what they computed before. What
            # belongs here is the dependency itself: a Formula reading the Sector
            # classification map depends on a current classification backfilled
            # across history, and one reading another Formula's output inherits
            # that Formula's parents. A Panel that recorded only the price feed's
            # schedule could not distinguish either case from a plain OHLCV
            # Formula, which is what made the catalog-wide policy look sufficient.
            "source_authority_ids": list(source_authority_ids[factor_id]),
        }
    return result
