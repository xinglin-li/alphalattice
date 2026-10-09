"""Resolve authored handles against a real workspace.

Everything here reads artifacts the product already published: the Panel snapshot
lifecycle, the Panel manifest, the Panel semantic index, the current
quality-filtered research manifest, and the market-data source watermark. Nothing
is generated, and nothing is trusted from the document beyond the handles
themselves.

This module lives in ``product_host`` because it is deliberately Desk- and
storage-aware. ``product_host`` has zero legal in-degree -- no package under
``src/`` may import it -- so a concrete resolver placed here can only ever reach
the Desk-neutral dispatcher by injection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FeatureAvailabilityError,
    installed_feature_availability_policy,
    latest_selectable_observation_session,
)
from alphalattice.foundation.feature_engine.contracts import panel_source_manifest_revision
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.panels.observation_clock_authority import (
    FeaturePanelObservationClockVerifier,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.evaluation.contracts import AlphaMetricPolicy
from alphalattice.kernel.quant.sector_history import SectorHistory
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
)

CURRENT_SNAPSHOT_HANDLE = "current"

EXPLORATION_SAMPLE_MARK = ".sample-"
"""A universe handle `<profile>.sample-<n>` names a sample of the profile's names: the
exploration lane's small universe (binding plan, B17)."""
MINIMUM_EXPLORATION_SAMPLE = int(AlphaMetricPolicy.model_fields["minimum_cross_section"].default)
"""An Alpha study ranks a session only on this many names (its metric policy's minimum
cross-section); a smaller sample would publish no rank IC."""
EXPLORATION_SAMPLE_KINDS = frozenset(
    {"factor.screening-development", "alpha.model-development", "portfolio.policy-development"}
)
"""The studies whose arrays read the authority's listing axis. A Risk study estimates on
its own axis, so it does not run on a sample."""


def universe_profile(handle: str) -> str:
    """The market profile a universe handle names: itself, or the profile it samples."""
    profile, mark, _size = handle.rpartition(EXPLORATION_SAMPLE_MARK)
    return profile if mark else handle


def universe_control(handle: str, listing_count: int) -> dict[str, object]:
    """A draft's universe: the whole profile, or a sample of its names (R4).

    The sizes are the ones the authority admits: at least the metric policy's minimum
    cross-section, and fewer than the Panel's names.

    Args:
        handle: The draft's universe handle.
        listing_count: The names on the Panel's listing axis.

    Returns:
        The control.
    """
    return {
        "path": ["experiment", "universe_handle"],
        "label": "Universe",
        "type": "universe",
        "value": handle,
        "whole": universe_profile(handle),
        "sample_mark": EXPLORATION_SAMPLE_MARK,
        "sample_min": MINIMUM_EXPLORATION_SAMPLE,
        "sample_max": listing_count - 1,
        "help": (
            "The whole universe, or a sample of its names for the exploration lane; a study "
            "on a sample is promoted onto every name before a strategy is prepared from it."
        ),
    }


def exploration_sample_size(handle: str) -> int | None:
    """The sample size a universe handle declares; None for a whole profile."""
    _profile, mark, size = handle.rpartition(EXPLORATION_SAMPLE_MARK)
    if not mark:
        return None
    if not size.isdigit() or str(int(size)) != size:
        raise AuthoringError("research_lane.sample_handle_invalid")
    return int(size)


def _rank(listing_id: str) -> str:
    return sha256(f"alphalattice.exploration-sample:{listing_id}".encode()).hexdigest()


def panel_sector_labels(
    store: MarketDataRepository,
    manifest: UniverseManifest,
    panel_manifest: Mapping[str, Any],
    listing_ids: tuple[str, ...],
) -> tuple[SectorHistory, str | None]:
    """Each name's Sector as the selected Panel was built with it, and the coverage hash.

    Sealed inputs retain the database's exact bytes. An exited name's last
    observed Sector can therefore serve its historical samples without making
    it a current member. Bind any wider label coverage separately from the
    Panel's own Sector revision; never pass today's smaller roster as the array axis.

    Args:
        store: The Panel's source market-data store.
        manifest: The universe manifest revision the Panel was built from.
        panel_manifest: The Panel's manifest; its lineage names its Sector revision.
        listing_ids: The names to label, in the axis order.

    Returns:
        The label of every name, in the axis order, and the hash binding labels taken
        beyond the current roster (None when every name is a current member).

    Raises:
        AuthoringError: The Panel's Sector revision is no longer the current one, or a
            name has no label.
    """
    feature_state = FeatureStateRepository(store.database, market_data=store)
    sector = feature_state.current_sector_state(manifest)
    expected = panel_manifest["safe_summary"]["lineage"]["sector_revision"]
    if sector is None or sector.sector_revision != expected:
        raise AuthoringError("research_authoring.sector_evidence_unavailable")
    labels = {
        listing: sector.sector_by_listing_id[listing]
        for listing in listing_ids
        if listing in sector.sector_by_listing_id
    }
    missing = tuple(listing for listing in listing_ids if listing not in labels)
    if missing:
        labels.update(feature_state.sector_classifications(missing))
    if set(labels) != set(listing_ids):
        raise AuthoringError("research_authoring.sector_coverage_incomplete")
    coverage_hash = (
        None
        if set(labels) == set(sector.sector_by_listing_id)
        else str(canonical_hash({"source_sector_revision": expected, "sector_labels": labels}))
    )
    # The reclassifications the Panel's sessions read, from its lineage.
    return (
        SectorHistory.of_panel(
            panel_manifest["safe_summary"]["lineage"],
            {listing: labels[listing] for listing in listing_ids},
        ),
        coverage_hash,
    )


def exploration_sample(
    listing_ids: tuple[str, ...], size: int, sectors: Mapping[str, str]
) -> tuple[str, ...]:
    """A sample of a Panel's listings by Sector, in the axis order (binding plan, decision 4).

    The names are given out one at a time, each to the Sector with the highest Sainte-Lague
    quotient (its names over twice its names taken plus one; a Sector with none taken first;
    ties by Sector name), so each Sector takes its share of ``size``, every Sector is in the
    sample once it holds as many names as there are Sectors (a Sector-residual target keeps
    its Sectors), and a larger sample keeps a smaller one's names. Inside a Sector each name
    is ranked by a hash of its listing ID alone, so one input and one size always sample the
    same names; the labels are the Panel's own, as the Alpha study reads them.
    """
    groups: dict[str, list[str]] = {}
    for listing in listing_ids:
        groups.setdefault(sectors[listing], []).append(listing)
    take = dict.fromkeys(groups, 0)
    for _ in range(min(size, len(listing_ids))):
        name = min(
            (n for n in groups if take[n] < len(groups[n])),
            key=lambda n: (take[n] > 0, -Fraction(len(groups[n]), 2 * take[n] + 1), n),
        )
        take[name] += 1
    chosen = {
        v for name, members in groups.items() for v in sorted(members, key=_rank)[: take[name]]
    }
    return tuple(v for v in listing_ids if v in chosen)


"""The one symbolic snapshot handle: whichever Panel is ACTIVE right now.

Any other handle must be a full snapshot hash. There is deliberately no prefix
matching and no "most recent like this" search: a handle either names exactly one
published snapshot or it resolves to nothing.
"""


@dataclass(frozen=True, slots=True)
class InstalledResearchSnapshot:
    """Host-installed semantic handle for one immutable published Panel.

    This binding is runtime composition, never authored YAML.  It lets a
    development Program say ``panel-remediation-baseline`` while the Host
    resolves the content identity and the Panel's own universe revision.  No
    current pointer is consulted for an installed or exact-hash handle.
    """

    handle: str
    panel_snapshot_hash: str

    def __post_init__(self) -> None:
        """Require a concrete immutable snapshot handle and exact lowercase digest.

        Raises:
            ValueError: Handle is empty/current or the Panel snapshot digest is invalid.
        """
        if (
            not self.handle
            or len(self.panel_snapshot_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.panel_snapshot_hash)
            or self.handle == CURRENT_SNAPSHOT_HANDLE
        ):
            raise ValueError("research_authoring.installed_snapshot_invalid")


class WorkspaceResearchAuthorityResolver:
    """Resolve one envelope's handles against a workspace on disk."""

    def __init__(
        self,
        *,
        workspace: Path,
        artifact_root: Path | None = None,
        installed_snapshots: tuple[InstalledResearchSnapshot, ...] = (),
        feature_catalog: FeatureCatalog | None = None,
        feature_kernels: FeatureKernelRegistry | None = None,
        feature_input: ResolvedDevelopmentFeatureInput | None = None,
    ) -> None:
        """Compose exact snapshot locations and deterministic Feature authority owners.

        Compose workspace, installed snapshot locations and deterministic Feature authority owners.

        Installed handles resolve location only. Feature clocks, implementations and methodology
        identities are independently recomputed by their owners.

        Args:
            workspace: Caller-owned admitted workspace root.
            artifact_root: Optional explicit artifact root.
            installed_snapshots: Explicit unique immutable handle/location bindings.
            feature_catalog: Optional installed Feature catalog owner.
            feature_kernels: Optional installed Feature kernel owner.
            feature_input: Optional exact resolved development overlay.

        Raises:
            ValueError: Installed snapshot handles are duplicated.
        """
        self._workspace = Path(workspace)
        self._artifact_root = (
            Path(artifact_root) if artifact_root else self._workspace / "artifacts"
        )
        index = {value.handle: value.panel_snapshot_hash for value in installed_snapshots}
        if len(index) != len(installed_snapshots):
            raise ValueError("research_authoring.installed_snapshot_duplicate")
        self._installed_snapshots: Mapping[str, str] = index
        self._feature_catalog = feature_catalog
        self._feature_kernels = feature_kernels
        self._feature_input = feature_input
        """The Feature owners this resolver re-derives a Panel's clock authority from.

        Composition state, exactly as it already is for every writer a workspace
        runtime builds: the materializer, the persistence factor axis and the
        Panel publisher all take an installed catalog revision and kernel set, and
        a reader that verifies what they wrote has to be able to take the same
        ones. Left unset they resolve to the shipped catalog and the shipped
        registry, so production composes as before.

        This is not a caller-supplied expectation. The verifier is handed owners
        and recomputes every clock, implementation and methodology identity from
        them; it is never handed a hash to compare against.
        """

    def resolve(self, envelope: ResearchExperimentEnvelope) -> ResolvedResearchAuthority:
        """Resolve exact research authority from verified snapshot lineage.

        Verify exact snapshot lineage and resolve declared universe, sessions and source authority.

        Prepared features and component training sources use their explicit admitted handles.
        Exploration samples require an installed supported kind and a bounded sector-share
        selection.

        Args:
            envelope: Explicit sealed experiment declaration.

        Returns:
            Validated research authority with exact Panel, universe, listing/session axis and source
            watermark.

        Raises:
            AuthoringError: Prepared source, snapshot, Feature clock, universe lineage, session
                range or sample is inadmissible.
        """
        if envelope.data_snapshot_handle.startswith("research-features@"):
            source = self._feature_input
            if source is None or source.source_handle != envelope.data_snapshot_handle:
                raise AuthoringError("research_authoring.prepared_features_not_resolved")
            if envelope.kind not in {"factor.screening-development", "alpha.model-development"}:
                raise AuthoringError("research_authoring.prepared_features_method_not_supported")
            parent = self.resolve(
                ResearchExperimentEnvelope.create(
                    **{
                        **envelope.model_dump(mode="python", exclude={"envelope_hash"}),
                        "data_snapshot_handle": source.base_panel_snapshot_hash,
                    }
                )
            )
            return ResolvedResearchAuthority.create(
                **{
                    **parent.model_dump(mode="python", exclude={"authority_hash"}),
                    "data_snapshot_handle": source.source_handle,
                    "panel_snapshot_hash": source.panel_manifest["snapshot_hash"],
                    "panel_manifest_ref": source.manifest_ref,
                }
            )
        if envelope.data_snapshot_handle.startswith("component-training."):
            return self._training_authority(envelope)
        profile = universe_profile(envelope.universe_handle)
        sample = exploration_sample_size(envelope.universe_handle)
        if sample is not None and envelope.kind not in EXPLORATION_SAMPLE_KINDS:
            raise AuthoringError(f"research_lane.sample_not_supported:{envelope.kind}")
        store = MarketDataRepository(self._workspace)
        current_manifest = None
        if envelope.data_snapshot_handle == CURRENT_SNAPSHOT_HANDLE:
            # Preserve the generic current-route refusal order: an unknown
            # Universe is rejected before a snapshot lookup that depends on it.
            # Installed remediation handles never enter this pointer-reading
            # branch and instead derive their immutable Universe revision below.
            current_manifest = store.current_quality_filtered_research_manifest(
                market_profile_id=profile
            )
            if current_manifest is None:
                raise AuthoringError("research_authoring.universe_handle_unresolved")
        resolver = ArtifactResolver(self._artifact_root)
        snapshot_hash = self._snapshot_hash(store, envelope)
        # Feature authority is checked before any semantic/session relation is
        # consumed.  Runtime-installed snapshot handles resolve location only;
        # they never substitute for the Feature owner's independent proof.
        self._require_feature_clock_authority(resolver, snapshot_hash)
        try:
            panel_manifest = resolver.load_feature_panel_manifest(
                resolver.feature_panel_manifest_uri(snapshot_hash)
            )
        except (FileNotFoundError, KeyError, ValueError) as error:
            raise AuthoringError("research_authoring.snapshot_handle_unresolved") from error
        try:
            manifest_revision = panel_source_manifest_revision(panel_manifest)
        except ValueError as error:
            raise AuthoringError(
                "research_authoring.snapshot_universe_lineage_unavailable"
            ) from error
        try:
            manifest = store.load_universe_manifest_revision(manifest_revision)
        except ValueError as error:
            raise AuthoringError("research_authoring.universe_handle_unresolved") from error
        if manifest.profile.market_profile_id != profile:
            raise AuthoringError("research_authoring.snapshot_universe_handle_mismatch")
        found = resolver.find_feature_panel_semantic_index(panel_snapshot_hash=snapshot_hash)
        if found is None:
            # A Panel with no semantic index has no session axis that can be
            # named, so there is nothing to resolve the requested range against.
            raise AuthoringError("research_authoring.snapshot_handle_unresolved")
        index, _index_uri = found

        sessions = self._sessions_in_range(index, envelope)
        if not sessions:
            raise AuthoringError("research_authoring.no_sessions_resolved")

        panel_ref = resolver.feature_panel_manifest_uri(snapshot_hash)
        listing_ids = FeaturePanelReader(resolver).listing_ids(panel_ref)
        if sample is not None:
            if not MINIMUM_EXPLORATION_SAMPLE <= sample < len(listing_ids):
                raise AuthoringError(
                    f"research_lane.sample_size_invalid:{sample}:"
                    f"{MINIMUM_EXPLORATION_SAMPLE}-{len(listing_ids) - 1}"
                )
            labels, _coverage = panel_sector_labels(store, manifest, panel_manifest, listing_ids)
            listing_ids = exploration_sample(listing_ids, sample, labels)
        watermark = store.execution_source_watermark(
            manifest, through=max(sessions), listing_ids=listing_ids
        )
        return ResolvedResearchAuthority.create(
            data_snapshot_handle=envelope.data_snapshot_handle,
            universe_handle=envelope.universe_handle,
            panel_snapshot_hash=snapshot_hash,
            panel_manifest_ref=panel_ref,
            universe_revision_sha256=manifest.revision_sha256,
            # The verified, sorted historical coverage axis, not today's roster.
            # Per-session sampling remains the paired Panel's responsibility.
            ordered_listing_ids=listing_ids,
            listing_sample=None if sample is None else "SECTOR_SHARES",
            sessions=sessions,
            source_watermark_hash=str(canonical_hash(watermark)),
        )

    def _require_feature_clock_authority(
        self, resolver: ArtifactResolver, snapshot_hash: str
    ) -> None:
        """Refuse a Panel the installed Feature owners cannot answer for.

        The verifier takes a snapshot handle, a resolver and the installed owners.
        A caller cannot hand it an expected identity, so "verified" can only ever
        mean re-derived from a catalog, a kernel registry and an availability
        policy this process actually holds.
        """

        outcome = FeaturePanelObservationClockVerifier(
            resolver=resolver,
            catalog=self._feature_catalog,
            kernel_registry=self._feature_kernels,
        ).verify(snapshot_hash)
        if outcome.verified:
            return
        if outcome.failure_code == "feature_panel.manifest_unresolved":
            raise AuthoringError("research_authoring.snapshot_handle_unresolved")
        if outcome.failure_code == "feature_panel.observation_clock_authority_absent":
            raise AuthoringError("research_authoring.panel_observation_clock_absent")
        raise AuthoringError("research_authoring.panel_clock_authority_unverified")

    def _training_authority(
        self, envelope: ResearchExperimentEnvelope
    ) -> ResolvedResearchAuthority:
        from alphalattice.control.product_host.composition.research_workspace import (
            component_training_selection,
            resolve_workspace_model_lifecycle,
        )
        from alphalattice.investment.alpha_research.scores.model_renewal import (
            AlphaTrainingObservations,
        )

        if (
            envelope.kind != "alpha.model-development"
            or envelope.universe_handle != "us-current-index-research"
        ):
            raise AuthoringError("research_authoring.training_source_not_admitted")
        component, training_authority = component_training_selection(envelope.data_snapshot_handle)
        store, admitted = resolve_workspace_model_lifecycle(
            self._workspace,
            component_id=component,
            artifact_root=self._artifact_root,
            training_authority_hash=training_authority,
        )
        observations = store._load(
            "lifecycle-training-observations",
            admitted.observations_hash,
            "content_hash",
            AlphaTrainingObservations,
        )
        snapshot = store.load_frozen_observation_snapshot(observations.observation_hash)
        # Only exact admitted sessions, bounded by the authored decision cutoff.
        requested = envelope.sessions
        if (
            requested.as_of.phase.name != "OFFICIAL_CLOSE"
            or requested.end > requested.as_of.session
        ):
            raise AuthoringError("research_authoring.training_cutoff_not_admitted")
        sessions = tuple(
            day for day in snapshot.formation_sessions if requested.start <= day <= requested.end
        )
        if not sessions:
            raise AuthoringError("research_authoring.no_sessions_resolved")
        return ResolvedResearchAuthority.create(
            data_snapshot_handle=envelope.data_snapshot_handle,
            universe_handle=envelope.universe_handle,
            training_snapshot_hash=observations.content_hash,
            training_manifest_ref=store.uri(
                "current/lifecycle-training-observations", observations.content_hash
            ),
            universe_revision_sha256=canonical_hash(
                [observations.content_hash, snapshot.ordered_listing_ids]
            ),
            ordered_listing_ids=snapshot.ordered_listing_ids,
            sessions=sessions,
            source_watermark_hash=canonical_hash(
                [snapshot.snapshot_hash, requested.as_of.model_dump(mode="json")]
            ),
        )

    def _snapshot_hash(
        self, store: MarketDataRepository, envelope: ResearchExperimentEnvelope
    ) -> str:
        from alphalattice.foundation.feature_engine.storage.repositories import (
            PanelStateRepository,
        )

        handle = envelope.data_snapshot_handle
        if handle == CURRENT_SNAPSHOT_HANDLE:
            panel_state = PanelStateRepository(store.database, market_data=store)
            snapshot = panel_state.feature_panel_snapshot_for_active(envelope.universe_handle)
            if snapshot is None:
                raise AuthoringError("research_authoring.snapshot_handle_unresolved")
            return str(snapshot["snapshot_hash"])
        installed = self._installed_snapshots.get(handle)
        if installed is not None:
            return installed
        if len(handle) != 64 or any(character not in "0123456789abcdef" for character in handle):
            raise AuthoringError("research_authoring.snapshot_handle_unresolved")
        return str(handle)

    @staticmethod
    def _sessions_in_range(
        index: dict[str, object], envelope: ResearchExperimentEnvelope
    ) -> tuple[date, ...]:
        """Intersect the requested range with the Panel's own session axis.

        The axis comes from the index rather than a calendar, so a request can
        only ever name sessions the Panel actually materialized.

        The upper bound is the installed availability policy's answer applied to
        a *typed* decision event, not to a bare date. A date cannot distinguish a
        pre-open decision from a post-close one, and the difference is a whole
        session of information: a daily Feature for session ``T`` exists only
        after ``close(T)``. An unstated phase is refused rather than assumed
        closed.

        The bound is the narrower of the requested ``end`` and what the decision
        event can actually see.
        """

        rows = index.get("sessions")
        if not isinstance(rows, list):
            raise AuthoringError("research_authoring.snapshot_handle_unresolved")
        request = envelope.sessions
        axis = []
        for row in rows:
            if not isinstance(row, dict):
                raise AuthoringError("research_authoring.snapshot_handle_unresolved")
            axis.append(date.fromisoformat(str(row["session_date"])))
        try:
            selectable = latest_selectable_observation_session(
                axis,
                decision_cutoff=request.as_of,
                availability=installed_feature_availability_policy(),
            )
        except FeatureAvailabilityError as error:
            raise AuthoringError("research_authoring.decision_cutoff_phase_ambiguous") from error
        if selectable is None:
            return ()
        # The narrower of what was asked for and what is available. The envelope
        # already requires ``as_of.session >= end``, so the old ``min(end, as_of)``
        # was always ``end`` and never compared anything -- availability had no
        # way to remove a session however early in the day the decision was made.
        bound = min(request.end, selectable)
        return tuple(sorted({item for item in axis if request.start <= item <= bound}))


__all__ = [
    "CURRENT_SNAPSHOT_HANDLE",
    "InstalledResearchSnapshot",
    "WorkspaceResearchAuthorityResolver",
]
