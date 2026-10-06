"""Open a catalog's Feature closure on a workspace that has never published.

Every other closure entry point assumes a Panel already exists: persistence and
the Panel recovery binding both require a head, and the only head producer,
``FeatureClosureAdmissionService``, requires an active Panel snapshot. A new
workspace therefore cannot build its first Panel at all.

This service opens the closure and nothing else. It does not build features and
it does not publish a Panel; once the genesis head exists the ordinary build,
persist, and publish path produces the first real Panel with no special casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.inputs.closure_source import (
    FeatureClosureSourceRepository,
    empty_feature_row_hash_digest,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    FeatureBaseClosureGenesisRoot,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
    identified,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest

GENESIS_ORIGIN = "GENESIS_PANEL_PUBLICATION"


class FeatureClosureAuthorityFailure(ValueError):
    """The publication plane holds state that no genesis closure can explain."""


@dataclass(frozen=True, slots=True)
class GenesisClosureOutcome:
    """Genesis root and head identities with an exact-reuse disposition."""

    catalog_hash: str
    root_hash: str
    head_hash: str
    disposition: Literal["GENESIS_READY", "REUSED_EXACT"]


class FeatureClosureGenesisService:
    """Establish genesis closure authority, exactly once, for one catalog."""

    def __init__(
        self,
        *,
        panel_state: PanelStateRepository,
        source: FeatureClosureSourceRepository,
        ledger: FeatureClosureLedger,
    ) -> None:
        """Compose the Panel store, source and closure ledger for genesis."""
        self._store = panel_state
        self._source = source
        self._ledger = ledger

    def open_genesis(
        self, *, manifest: UniverseManifest, catalog: FeatureCatalog
    ) -> GenesisClosureOutcome:
        """Open or exactly reuse a genesis closure for an empty publication plane."""
        catalog_hash = catalog.binding.catalog_hash
        expected = self.expected_genesis_root(manifest=manifest, catalog=catalog)
        head = self._ledger.current_head(catalog_hash)
        if head is not None:
            return self._verified_reuse(head_root_hash=head.root_hash, expected=expected)

        self._require_uninitialized_publication_plane(manifest, catalog_hash)
        self._require_empty_closure(catalog_hash)
        # Immutable artifact first: a failure before the swap leaves it
        # unreachable, and a retry recomputes the same hash.
        self._ledger.publish_genesis_root(expected)
        try:
            installed = self._ledger.install_genesis_head(expected)
        except ValueError as error:
            if "head_compare_and_swap_failed" not in str(error):
                raise
            # A concurrent bootstrap won the swap. Re-read and prove it is ours.
            contender = self._ledger.current_head(catalog_hash)
            if contender is None:
                raise
            return self._verified_reuse(head_root_hash=contender.root_hash, expected=expected)
        return GenesisClosureOutcome(
            catalog_hash=catalog_hash,
            root_hash=expected.root_hash,
            head_hash=installed.head_hash,
            disposition="GENESIS_READY",
        )

    def _verified_reuse(
        self, *, head_root_hash: str, expected: FeatureBaseClosureGenesisRoot
    ) -> GenesisClosureOutcome:
        if not self._ledger.is_genesis_root(head_root_hash):
            raise FeatureClosureAuthorityFailure("feature_closure.genesis_closure_already_admitted")
        if head_root_hash != expected.root_hash:
            raise FeatureClosureAuthorityFailure("feature_closure.genesis_root_mismatch")
        head = self._ledger.require_head(expected.catalog_hash)
        return GenesisClosureOutcome(
            catalog_hash=expected.catalog_hash,
            root_hash=head_root_hash,
            head_hash=head.head_hash,
            disposition="REUSED_EXACT",
        )

    def _require_uninitialized_publication_plane(
        self, manifest: UniverseManifest, catalog_hash: str
    ) -> None:
        """Refuse to invent an ancestor for a Panel that already exists.

        A completed onboarding and a universe manifest are expected here and are
        not conflicts. An active or snapshotted Panel of this catalog without a
        closure head is not a resumable genesis: it is a publication plane whose
        lineage this service cannot honestly reconstruct, so it fails loudly
        instead of adopting the Panel under a fabricated root. A Panel of another
        catalog is not adopted: after a person's activation the new catalog's
        closure starts empty beside it, its rows naming their own catalog, and
        its first Panel replaces the other on publication.
        """
        market_profile_id = manifest.profile.market_profile_id
        for panel in (
            self._store.active_feature_panel(market_profile_id),
            self._store.feature_panel_snapshot_for_active(market_profile_id),
        ):
            if panel is not None and str(panel.get("catalog_hash")) == catalog_hash:
                raise FeatureClosureAuthorityFailure("feature_closure.genesis_authority_failure")

    def _require_empty_closure(self, catalog_hash: str) -> None:
        """Refuse persisted rows without a head as unexplained lineage."""
        if self._source.feature_row_count(catalog_hash=catalog_hash) != 0:
            raise FeatureClosureAuthorityFailure("feature_closure.genesis_authority_failure")

    def expected_genesis_root(
        self, *, manifest: UniverseManifest, catalog: FeatureCatalog
    ) -> FeatureBaseClosureGenesisRoot:
        """Derive the expected genesis root from manifest and catalog identity."""
        catalog_hash = catalog.binding.catalog_hash
        return identified(
            FeatureBaseClosureGenesisRoot,
            {
                "catalog_hash": catalog_hash,
                "manifest_revision": manifest.revision_sha256,
                "listing_ids": tuple(sorted(item.listing_id for item in manifest.listings)),
                "factor_ids": catalog.factor_ids,
                "feature_row_hash_digest": empty_feature_row_hash_digest(),
            },
            "root_hash",
        )


__all__ = [
    "GENESIS_ORIGIN",
    "FeatureClosureAuthorityFailure",
    "FeatureClosureGenesisService",
    "GenesisClosureOutcome",
]
