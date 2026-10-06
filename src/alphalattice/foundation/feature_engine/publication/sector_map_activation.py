"""Map-before-mutation authority for future current-sector revisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    SectorRevisionEntry,
    SectorRevisionMap,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    SectorRevisionMapActivationReceipt,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
    identified,
)
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.returns.sector_revision_identity import (
    sector_revision_hash,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _utc(value: datetime) -> datetime:
    """Attach UTC to persisted naive times or convert aware times to UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class SectorRevisionMapActivationCoordinator:
    """Publish a complete immutable map before activating the Store revision."""

    def __init__(
        self,
        *,
        store: FeatureStateRepository,
        mutation_gate: WorkspaceMutationGate,
        ledger: FeatureClosureLedger,
    ) -> None:
        """Compose the store, mutation gate and closure ledger for publication."""
        self._store = store
        self._mutation_gate = mutation_gate
        self._ledger = ledger

    def activate(
        self,
        *,
        manifest: UniverseManifest,
        observations: Sequence[Mapping[str, object]],
        observed_at: datetime,
    ) -> tuple[str, bool, str]:
        """Publish the complete sector map before activating its store revision."""
        entries = tuple(
            SectorRevisionEntry(
                listing_id=str(item["listing_id"]),
                provider=str(item["provider"]),
                provider_symbol=str(item["provider_symbol"]),
                sector_name=str(item["sector_name"]),
                sector_key=str(item["sector_key"]) if item.get("sector_key") is not None else None,
                payload_hash=str(item["payload_hash"]),
                evidence_hash=str(
                    item.get("evidence_hash")
                    or canonical_hash(
                        {
                            "source": "YAHOO_CURRENT_SECTOR",
                            "payload": item["payload_hash"],
                            "symbol": item["provider_symbol"],
                        }
                    )
                ),
            )
            for item in sorted(observations, key=lambda value: str(value["listing_id"]))
        )
        safe_observations = tuple(item.model_dump(mode="python") for item in entries)
        expected_revision = sector_revision_hash(
            manifest_revision=manifest.revision_sha256,
            observations=safe_observations,
        )
        sector_map = identified(
            SectorRevisionMap,
            {
                "manifest_revision": manifest.revision_sha256,
                "sector_revision": expected_revision,
                "entries": entries,
            },
            "map_hash",
        )
        _published(self._ledger.publish_sector_map, sector_map)
        revision, changed, store_receipt, effective_session = self._mutation_gate.run(
            self._store.activate_sector_revision,
            manifest=manifest,
            observations=safe_observations,
            observed_at=observed_at,
        )
        state = self._store.current_sector_state(manifest)
        if revision != expected_revision or state is None or state.sector_revision != revision:
            raise ValueError("feature_closure.sector_map_capture_failed")
        activation = identified(
            SectorRevisionMapActivationReceipt,
            {
                "manifest_revision": manifest.revision_sha256,
                "sector_revision": revision,
                "sector_map_hash": sector_map.map_hash,
                "store_receipt_hash": store_receipt,
                "changed": changed,
                "observed_at": observed_at,
                "effective_session": effective_session,
            },
            "receipt_hash",
        )
        _published(self._ledger.publish_sector_activation, activation)
        return revision, changed, store_receipt

    def bind_to_manifest(self, manifest: UniverseManifest) -> tuple[str, str] | None:
        """Rebind unchanged sector evidence onto a derived manifest, map first.

        The sector revision is bound to manifest identity, so a quality-
        governance child activates a revision that has never been observed as
        such. Left to the repository alone that revision reaches the store with
        no map in the ledger, and the next panel recovery binding cannot
        resolve one -- an admitted closure would fall back to a recipe carrying
        the parent's revision, and a genesis closure has no recipe at all.

        Returns ``None`` when the evidence is not carryable, which is the
        repository's own answer and not a failure.
        """
        evidence = self._store.bindable_sector_evidence(manifest)
        if evidence is None:
            return None
        sector_map = identified(
            SectorRevisionMap,
            {
                "manifest_revision": manifest.revision_sha256,
                "sector_revision": evidence.sector_revision,
                "entries": tuple(
                    SectorRevisionEntry(**dict(item)) for item in evidence.observations
                ),
            },
            "map_hash",
        )
        _published(self._ledger.publish_sector_map, sector_map)
        bound = self._mutation_gate.run(
            self._store.bind_current_sector_evidence_to_manifest,
            manifest,
        )
        if bound is None:
            return None
        revision, receipt_hash = bound
        state = self._store.current_sector_state(manifest)
        if (
            revision != evidence.sector_revision
            or state is None
            or state.sector_revision != revision
        ):
            raise ValueError("feature_closure.sector_map_capture_failed")
        activation = identified(
            SectorRevisionMapActivationReceipt,
            {
                "manifest_revision": manifest.revision_sha256,
                "sector_revision": revision,
                "sector_map_hash": sector_map.map_hash,
                "store_receipt_hash": receipt_hash,
                # Rebinding carries evidence across a membership change rather
                # than observing anything, so the receipt keeps the original
                # observation time instead of stamping a new one.
                "changed": False,
                "observed_at": _utc(evidence.source_observed_at),
            },
            "receipt_hash",
        )
        _published(self._ledger.publish_sector_activation, activation)
        return revision, receipt_hash


def _published[Published](publish: Callable[[Published], None], value: Published) -> None:
    """Publish through the ledger, naming an I/O refusal apart from bad evidence.

    A pointer or artifact the ledger could not place on disk (a sharing
    violation that outlasted its retry bound, any other OS error) leaves the
    previous state intact and is ``feature_closure.publication_blocked``, with
    the typed cause chained; the ledger refusing the evidence itself stays
    ``feature_closure.sector_map_capture_failed``. This classification does
    not roll back earlier successful publications.
    """
    try:
        publish(value)
    except (WorkspaceConflictError, OSError) as exc:
        raise ValueError("feature_closure.publication_blocked") from exc
    except Exception as exc:
        raise ValueError("feature_closure.sector_map_capture_failed") from exc


__all__ = ["SectorRevisionMapActivationCoordinator"]
