"""Content-addressed ledger and operational CAS pointers for Feature closure."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import cast

import pyarrow as pa
from pydantic import BaseModel

from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
    arrow_table_logical_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    PanelBaseValueClosureManifest,
    PanelDerivationRecipe,
    PanelRetentionAssessment,
    SectorRevisionMap,
    durable_payload,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    FeatureBaseClosureCatalogRoot,
    FeatureBaseClosureGenesisRoot,
    FeatureBaseClosureHead,
    FeatureBaseClosurePatch,
    FeatureBaseClosureTransitionMarker,
    FeatureBaseClosureTransitionReceipt,
    FeatureClosureFirstLoad,
    PanelRecoveryClosureBinding,
    PreparedFeatureBaseClosureTransition,
    SectorRevisionMapActivationReceipt,
)
from alphalattice.kernel.quant.sector_history import SectorHistory, SectorReclassification
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.persistence import (
    DURABLE_REPLACE_DELAYS,
    replace_with_retry,
)


class FeatureClosureLedger:
    """Own immutable ledger artifacts plus one CAS pointer per catalog."""

    def __init__(self, artifact_store: PanelClosureArtifactStore) -> None:
        """Bind immutable closure storage and operational pointer paths."""
        self._store = artifact_store
        self._operational_root = artifact_store.root / "feature-ledger" / "operational"

    def admit_catalog_root(
        self,
        *,
        catalog_hash: str,
        active_snapshot_hash: str,
        phase_one_failure_receipt_hash: str,
        retention_assessment_hash: str,
        feature_row_hash_digest: str,
    ) -> tuple[FeatureBaseClosureCatalogRoot, FeatureBaseClosureHead]:
        """Admit only the verified ACTIVE Phase-1 recipe as a new root."""
        failure = self._store.load_json(
            category="runs/failures", content_hash=phase_one_failure_receipt_hash
        )
        raw_verified = failure.get("verified_snapshot_hashes")
        raw_blocked = failure.get("blocked_snapshot_hashes")
        if not isinstance(raw_verified, list) or not isinstance(raw_blocked, list):
            raise ValueError("feature_closure.root_unavailable")
        verified = tuple(str(value) for value in raw_verified)
        blocked = tuple(str(value) for value in raw_blocked)
        if active_snapshot_hash not in verified or active_snapshot_hash in blocked:
            raise ValueError("feature_closure.root_unavailable")
        assessment = self._store.load_model(
            category="retention-assessments",
            content_hash=retention_assessment_hash,
            model=PanelRetentionAssessment,
        )
        if (
            assessment.snapshot_hash != active_snapshot_hash
            or assessment.disposition != "REMATERIALIZATION_VERIFIED_CURRENT_ENVIRONMENT"
            or assessment.recipe_hash is None
        ):
            raise ValueError("feature_closure.root_unavailable")
        recipe = self._store.load_model(
            category="recipes",
            content_hash=assessment.recipe_hash,
            model=PanelDerivationRecipe,
        )
        base_manifest = self._store.load_model(
            category="base-manifests",
            content_hash=recipe.base_closure_hash,
            model=PanelBaseValueClosureManifest,
        )
        if (
            recipe.snapshot_hash != active_snapshot_hash
            or recipe.catalog_hash != catalog_hash
            or base_manifest.catalog_hash != catalog_hash
            or base_manifest.factor_ids != recipe.factor_ids
        ):
            raise ValueError("feature_closure.root_unavailable")
        root = _identified(
            FeatureBaseClosureCatalogRoot,
            {
                "catalog_hash": catalog_hash,
                "active_panel_snapshot_hash": active_snapshot_hash,
                "phase_one_failure_receipt_hash": phase_one_failure_receipt_hash,
                "retention_assessment_hash": assessment.assessment_hash,
                "derivation_recipe_hash": recipe.recipe_hash,
                "base_closure_manifest_hash": base_manifest.manifest_hash,
                "listing_ids": recipe.listing_ids,
                "sessions": recipe.sessions,
                "factor_ids": recipe.factor_ids,
                "feature_row_hash_digest": feature_row_hash_digest,
            },
            "root_hash",
        )
        self.publish_model("feature-ledger/roots", root.root_hash, root)
        existing = self.current_head(catalog_hash)
        if existing is not None:
            if existing.root_hash != root.root_hash:
                raise ValueError("feature_closure.pending_transition_conflict")
            return root, existing
        head = _identified(
            FeatureBaseClosureHead,
            {
                "catalog_hash": catalog_hash,
                "root_hash": root.root_hash,
                "predecessor_head_hash": None,
                "last_transition_hash": None,
                "transition_cursor": 0,
                "feature_row_hash_digest": feature_row_hash_digest,
            },
            "head_hash",
        )
        self.publish_model("feature-ledger/heads", head.head_hash, head)
        self._cas_head(catalog_hash=catalog_hash, expected=None, next_head=head)
        return root, head

    def current_head(self, catalog_hash: str) -> FeatureBaseClosureHead | None:
        """Load the catalog's current closure head if one is published."""
        pointer = self._pointer_path("heads", catalog_hash)
        if not pointer.exists():
            return None
        payload = self._read_pointer(pointer)
        head_hash = str(payload["head_hash"])
        return self._store.load_model(
            category="feature-ledger/heads",
            content_hash=head_hash,
            model=FeatureBaseClosureHead,
        )

    def require_head(self, catalog_hash: str) -> FeatureBaseClosureHead:
        """Return the catalog head or reject an unopened closure."""
        head = self.current_head(catalog_hash)
        if head is None:
            raise ValueError("feature_closure.catalog_root_required")
        return head

    def is_genesis_root(self, root_hash: str) -> bool:
        """Discriminate by content-addressed category, never by lenient parsing."""
        path = Path(self._store.root) / "feature-ledger" / "genesis-roots" / f"{root_hash}.json"
        return bool(path.exists())

    def publish_genesis_root(self, root: FeatureBaseClosureGenesisRoot) -> None:
        """Write the immutable genesis root before any pointer can reach it."""
        self.publish_model("feature-ledger/genesis-roots", root.root_hash, root)

    def install_genesis_head(self, root: FeatureBaseClosureGenesisRoot) -> FeatureBaseClosureHead:
        """Open the closure with one compare-and-swap from no head at all."""
        head = _identified(
            FeatureBaseClosureHead,
            {
                "catalog_hash": root.catalog_hash,
                "root_hash": root.root_hash,
                "predecessor_head_hash": None,
                "last_transition_hash": None,
                "transition_cursor": 0,
                "feature_row_hash_digest": root.feature_row_hash_digest,
            },
            "head_hash",
        )
        self.publish_model("feature-ledger/heads", head.head_hash, head)
        self._cas_head(catalog_hash=root.catalog_hash, expected=None, next_head=head)
        return head

    def closure_root_for_head(
        self, head: FeatureBaseClosureHead
    ) -> FeatureBaseClosureCatalogRoot | FeatureBaseClosureGenesisRoot:
        """Load whichever root opened this closure, each through its own schema."""
        if self.is_genesis_root(head.root_hash):
            return self._store.load_model(
                category="feature-ledger/genesis-roots",
                content_hash=head.root_hash,
                model=FeatureBaseClosureGenesisRoot,
            )
        return self._store.load_model(
            category="feature-ledger/roots",
            content_hash=head.root_hash,
            model=FeatureBaseClosureCatalogRoot,
        )

    def root_for_head(self, head: FeatureBaseClosureHead) -> FeatureBaseClosureCatalogRoot:
        """Require an admitted root.

        A genesis closure has no Panel snapshot, retention assessment, or
        derivation recipe, so every caller that needs those is refused here
        rather than being handed a root that cannot answer them.
        """
        if self.is_genesis_root(head.root_hash):
            raise ValueError("feature_closure.genesis_root_not_admissible")
        return self._store.load_model(
            category="feature-ledger/roots",
            content_hash=head.root_hash,
            model=FeatureBaseClosureCatalogRoot,
        )

    def recipe_for_root(self, root: FeatureBaseClosureCatalogRoot) -> PanelDerivationRecipe:
        """Load the derivation recipe named by an admitted catalog root."""
        return self._store.load_model(
            category="recipes",
            content_hash=root.derivation_recipe_hash,
            model=PanelDerivationRecipe,
        )

    def pending_transition(self, catalog_hash: str) -> PreparedFeatureBaseClosureTransition | None:
        """Load the catalog's prepared transition if one is pending."""
        pointer = self._pointer_path("pending", catalog_hash)
        if not pointer.exists():
            return None
        payload = self._read_pointer(pointer)
        return self._store.load_model(
            category="feature-ledger/transitions/prepared",
            content_hash=str(payload["transition_hash"]),
            model=PreparedFeatureBaseClosureTransition,
        )

    def clear_finalized_pending(self, transition: PreparedFeatureBaseClosureTransition) -> bool:
        """Clear only the crash residue already named by the current head."""
        head = self.require_head(transition.catalog_hash)
        if head.last_transition_hash != transition.transition_hash:
            return False
        self._pointer_path("pending", transition.catalog_hash).unlink(missing_ok=True)
        return True

    def prepare_first_load(self, first_load: FeatureClosureFirstLoad) -> None:
        """Record a column part's first load while the genesis head it starts from is current."""
        head = self.require_head(first_load.catalog_hash)
        if head.head_hash != first_load.base_head_hash:
            raise ValueError("feature_closure.head_compare_and_swap_failed")
        pending = self.pending_first_load(first_load.catalog_hash)
        if self.pending_transition(first_load.catalog_hash) is not None or (
            pending is not None and pending.first_load_hash != first_load.first_load_hash
        ):
            raise ValueError("feature_closure.pending_transition_conflict")
        self.publish_model("feature-ledger/first-loads", first_load.first_load_hash, first_load)
        self._write_pointer(
            self._pointer_path("first-loads", first_load.catalog_hash),
            {"first_load_hash": first_load.first_load_hash},
        )

    def pending_first_load(self, catalog_hash: str) -> FeatureClosureFirstLoad | None:
        """Load the part's first load if one is pending."""
        pointer = self._pointer_path("first-loads", catalog_hash)
        if not pointer.exists():
            return None
        payload = self._read_pointer(pointer)
        return self._store.load_model(
            category="feature-ledger/first-loads",
            content_hash=str(payload["first_load_hash"]),
            model=FeatureClosureFirstLoad,
        )

    def complete_first_load(
        self, first_load: FeatureClosureFirstLoad, *, feature_row_hash_digest: str
    ) -> FeatureBaseClosureHead:
        """Advance the head to the digest of the rows the load wrote, naming it, and clear it."""
        current = self.require_head(first_load.catalog_hash)
        if current.head_hash != first_load.base_head_hash:
            raise ValueError("feature_closure.head_compare_and_swap_failed")
        next_head = _identified(
            FeatureBaseClosureHead,
            {
                "catalog_hash": current.catalog_hash,
                "root_hash": current.root_hash,
                "predecessor_head_hash": current.head_hash,
                "last_transition_hash": first_load.first_load_hash,
                "transition_cursor": current.transition_cursor + 1,
                "feature_row_hash_digest": feature_row_hash_digest,
            },
            "head_hash",
        )
        self.publish_model("feature-ledger/heads", next_head.head_hash, next_head)
        self._cas_head(
            catalog_hash=current.catalog_hash, expected=current.head_hash, next_head=next_head
        )
        self._pointer_path("first-loads", current.catalog_hash).unlink(missing_ok=True)
        return next_head

    def clear_first_load(self, first_load: FeatureClosureFirstLoad) -> None:
        """Clear a first load its head already names, or one whose rows were discarded."""
        head = self.require_head(first_load.catalog_hash)
        if head.head_hash != first_load.base_head_hash and (
            head.last_transition_hash != first_load.first_load_hash
        ):
            raise ValueError("feature_closure.recovery_required")
        self._pointer_path("first-loads", first_load.catalog_hash).unlink(missing_ok=True)

    def load_transition(self, transition_hash: str) -> PreparedFeatureBaseClosureTransition:
        """Load a prepared transition by its content hash."""
        return self._store.load_model(
            category="feature-ledger/transitions/prepared",
            content_hash=transition_hash,
            model=PreparedFeatureBaseClosureTransition,
        )

    def load_patch(self, patch_hash: str) -> FeatureBaseClosurePatch:
        """Load a Feature closure patch by its content hash."""
        return self._store.load_model(
            category="feature-ledger/patches",
            content_hash=patch_hash,
            model=FeatureBaseClosurePatch,
        )

    def publish_patch(
        self,
        *,
        catalog_hash: str,
        base_head_hash: str,
        idempotency_key: str,
        factor_ids: tuple[str, ...],
        listing_ids: tuple[str, ...],
        first_session: date,
        last_session: date,
        table: pa.Table,
    ) -> FeatureBaseClosurePatch:
        """Publish immutable patch rows and their sealed patch record."""
        logical_hash = arrow_table_logical_hash(table)
        artifact = self._store.publish_parquet(
            category="feature-ledger/patch-rows",
            content_hash=logical_hash,
            kind="FeatureBaseClosurePatchRows",
            table=table,
        )
        patch = _identified(
            FeatureBaseClosurePatch,
            {
                "catalog_hash": catalog_hash,
                "base_head_hash": base_head_hash,
                "idempotency_key": idempotency_key,
                "factor_ids": factor_ids,
                "row_count": table.num_rows,
                "first_session": first_session,
                "last_session": last_session,
                "listing_ids": tuple(sorted(set(listing_ids))),
                "artifact": artifact,
            },
            "patch_hash",
        )
        self.publish_model("feature-ledger/patches", patch.patch_hash, patch)
        return patch

    def prepare(self, transition: PreparedFeatureBaseClosureTransition) -> None:
        """Persist a transition only while its expected head is current."""
        head = self.require_head(transition.catalog_hash)
        if head.head_hash != transition.base_head_hash:
            raise ValueError("feature_closure.head_compare_and_swap_failed")
        pending = self.pending_transition(transition.catalog_hash)
        if pending is not None and pending.transition_hash != transition.transition_hash:
            raise ValueError("feature_closure.pending_transition_conflict")
        self.publish_model(
            "feature-ledger/transitions/prepared", transition.transition_hash, transition
        )
        self._write_pointer(
            self._pointer_path("pending", transition.catalog_hash),
            {"transition_hash": transition.transition_hash},
        )

    def complete(
        self,
        *,
        transition: PreparedFeatureBaseClosureTransition,
        receipt: FeatureBaseClosureTransitionReceipt,
        feature_row_hash_digest: str,
    ) -> tuple[FeatureBaseClosureHead, FeatureBaseClosureTransitionMarker]:
        """Publish verified receipt, next head, marker, and CAS the pointer."""
        current = self.require_head(transition.catalog_hash)
        if current.head_hash != transition.base_head_hash:
            raise ValueError("feature_closure.head_compare_and_swap_failed")
        if receipt.transition_hash != transition.transition_hash:
            raise ValueError("feature_closure.store_readback_mismatch")
        self.publish_model("feature-ledger/transitions/receipts", receipt.receipt_hash, receipt)
        next_head = _identified(
            FeatureBaseClosureHead,
            {
                "catalog_hash": current.catalog_hash,
                "root_hash": current.root_hash,
                "predecessor_head_hash": current.head_hash,
                "last_transition_hash": transition.transition_hash,
                "transition_cursor": current.transition_cursor + 1,
                "feature_row_hash_digest": feature_row_hash_digest,
            },
            "head_hash",
        )
        self.publish_model("feature-ledger/heads", next_head.head_hash, next_head)
        marker = _identified(
            FeatureBaseClosureTransitionMarker,
            {
                "transition_hash": transition.transition_hash,
                "receipt_hash": receipt.receipt_hash,
                "next_head_hash": next_head.head_hash,
            },
            "marker_hash",
        )
        self.publish_model("feature-ledger/transitions/markers", marker.marker_hash, marker)
        self._cas_head(
            catalog_hash=current.catalog_hash,
            expected=current.head_hash,
            next_head=next_head,
        )
        self._pointer_path("pending", current.catalog_hash).unlink(missing_ok=True)
        return next_head, marker

    def publish_panel_binding(self, binding: PanelRecoveryClosureBinding) -> None:
        """Publish a recovery binding and index it by snapshot."""
        self.publish_model("panel-recovery-bindings", binding.binding_hash, binding)
        self._write_pointer(
            self._pointer_path("panel-bindings", binding.snapshot_hash),
            {"binding_hash": binding.binding_hash},
        )

    def panel_binding(self, snapshot_hash: str) -> PanelRecoveryClosureBinding | None:
        """Load the recovery closure binding indexed by snapshot."""
        pointer = self._pointer_path("panel-bindings", snapshot_hash)
        if not pointer.exists():
            return None
        payload = self._read_pointer(pointer)
        return self._store.load_model(
            category="panel-recovery-bindings",
            content_hash=str(payload["binding_hash"]),
            model=PanelRecoveryClosureBinding,
        )

    def publish_sector_map(self, sector_map: SectorRevisionMap) -> None:
        """Publish an immutable sector revision map."""
        self.publish_model("sector-maps", sector_map.map_hash, sector_map)

    def publish_sector_activation(self, receipt: SectorRevisionMapActivationReceipt) -> None:
        """Publish activation evidence and point the revision to its map."""
        self.publish_model("sector-activation-receipts", receipt.receipt_hash, receipt)
        marker_hash = canonical_hash(
            {
                "kind": "SectorRevisionMapActivationMarker",
                "sector_revision": receipt.sector_revision,
                "sector_map_hash": receipt.sector_map_hash,
                "receipt_hash": receipt.receipt_hash,
            }
        )
        self._store.publish_json(
            category="sector-activation-markers",
            content_hash=marker_hash,
            payload={
                "kind": "SectorRevisionMapActivationMarker",
                "sector_revision": receipt.sector_revision,
                "sector_map_hash": receipt.sector_map_hash,
                "receipt_hash": receipt.receipt_hash,
                "marker_hash": marker_hash,
            },
        )
        self._write_pointer(
            self._pointer_path("sector-maps", receipt.sector_revision),
            {"sector_map_hash": receipt.sector_map_hash},
        )

    def sector_map_for_head(
        self, *, head: FeatureBaseClosureHead, sector_revision: str
    ) -> SectorRevisionMap:
        """Resolve the captured sector map from the head or its root recipe."""
        pointer = self._pointer_path("sector-maps", sector_revision)
        if pointer.exists():
            map_hash = str(self._read_pointer(pointer)["sector_map_hash"])
        else:
            root = self.closure_root_for_head(head)
            if isinstance(root, FeatureBaseClosureGenesisRoot):
                # A genesis closure carries no derivation recipe, so there is
                # no second place a sector map could be recovered from. Its
                # absence is the capture failure itself rather than a reason to
                # look somewhere that structurally cannot answer.
                raise ValueError("feature_closure.sector_map_capture_failed")
            recipe = self._store.load_model(
                category="recipes",
                content_hash=root.derivation_recipe_hash,
                model=PanelDerivationRecipe,
            )
            if recipe.sector_revision != sector_revision:
                raise ValueError("feature_closure.sector_map_capture_failed")
            map_hash = recipe.sector_map_hash
        sector_map = self._store.load_model(
            category="sector-maps", content_hash=map_hash, model=SectorRevisionMap
        )
        if sector_map.sector_revision != sector_revision:
            raise ValueError("feature_closure.sector_map_capture_failed")
        return sector_map

    def publish_model(self, category: str, content_hash: str, model: BaseModel) -> None:
        """Publish and read back a content-addressed closure model."""
        self._store.publish_json(
            category=category,
            content_hash=content_hash,
            payload=durable_payload(model),
        )
        loaded = self._store.load_model(
            category=category, content_hash=content_hash, model=type(model)
        )
        if loaded != model:
            raise ValueError("feature_closure.closure_readback_failed")

    def _cas_head(
        self,
        *,
        catalog_hash: str,
        expected: str | None,
        next_head: FeatureBaseClosureHead,
    ) -> None:
        pointer = self._pointer_path("heads", catalog_hash)
        current = self._read_pointer(pointer).get("head_hash") if pointer.exists() else None
        if current != expected:
            if current == next_head.head_hash:
                return
            raise ValueError("feature_closure.head_compare_and_swap_failed")
        self._write_pointer(pointer, {"head_hash": next_head.head_hash})

    def _pointer_path(self, group: str, identity: str) -> Path:
        if len(identity) != 64 or any(
            character not in "0123456789abcdef" for character in identity
        ):
            raise ValueError("Feature closure pointer identity is invalid")
        return self._operational_root / group / f"{identity}.json"

    @staticmethod
    def _read_pointer(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Feature closure pointer is invalid")
        return payload

    @classmethod
    def _write_pointer(cls, path: Path, payload: Mapping[str, object]) -> None:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        # A pointer republished with the bytes it already holds is left as it
        # is. The daily update rebinds the same sector evidence on every
        # one-listing cycle, and each needless replace is one more instant in
        # which a momentary Windows sharing violation can stop the Task.
        if path.exists() and path.read_bytes() == serialized:
            return
        # A changed pointer is staged and replaced atomically; a momentary
        # sharing violation is outlived within a bound, a persistent one is
        # raised as a typed conflict with the previous pointer still in place.
        # Remove this attempt's staged file even when replacement is refused.
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        staged.write_bytes(serialized)
        try:
            replace_with_retry(staged, path, delays=DURABLE_REPLACE_DELAYS)
        finally:
            staged.unlink(missing_ok=True)
        # Published only once the existing reader accepts what is on disk.
        if cls._read_pointer(path) != dict(payload):
            raise ValueError("Feature closure pointer readback failed")


def identified[Model: BaseModel](
    model: type[Model], values: Mapping[str, object], identity_field: str
) -> Model:
    """Validate a model with a content hash derived from its payload."""
    return _identified(model, values, identity_field)


def _identified[Model: BaseModel](
    model: type[Model], values: Mapping[str, object], identity_field: str
) -> Model:
    provisional = model.model_construct(**values, **{identity_field: "0" * 64})
    identity = durable_payload(provisional, exclude={identity_field})
    return cast(Model, model.model_validate({**values, identity_field: canonical_hash(identity)}))


def sector_history_as_of(store: PanelClosureArtifactStore, sector_revision: str) -> SectorHistory:
    """The Sector each session reads as of one revision: its map, then the reclassifications.

    Rebuilt from the maps the ledger keeps with their activation receipts, in observation order
    a receipt with an effective session records each listing whose Sector its map
    changed from the one before; a receipt without one, written before the forward rule, folds
    its changes into the backfill, as the Panels built then published them.

    Args:
        store: The Panel closure store.
        sector_revision: The revision the history ends at.

    Returns:
        The history over that revision's listings.

    Raises:
        ValueError: `feature_closure.sector_map_capture_failed` when the revision has no single
            map, a receipt names a map the store does not hold, or no receipt activated the
            revision while a later one recorded a reclassification.
    """
    maps = store.find_models(category="sector-maps", model=SectorRevisionMap, matches=_every)
    by_hash = {value.map_hash: value for value in maps}
    target = [value for value in maps if value.sector_revision == sector_revision]
    if len(target) != 1:
        raise ValueError("feature_closure.sector_map_capture_failed")
    receipts = sorted(
        store.find_models(
            category="sector-activation-receipts",
            model=SectorRevisionMapActivationReceipt,
            matches=_every,
        ),
        key=lambda value: (value.observed_at, value.receipt_hash),
    )
    current = {entry.listing_id: entry.sector_name for entry in target[0].entries}
    if not any(value.sector_revision == sector_revision for value in receipts):
        if any(value.effective_session is not None for value in receipts):
            raise ValueError("feature_closure.sector_map_capture_failed")
        return SectorHistory(current_revision=sector_revision, current=current)
    held: dict[str, str] = {}
    found: list[SectorReclassification] = []
    for receipt in receipts:
        activated = by_hash.get(receipt.sector_map_hash)
        if activated is None:
            raise ValueError("feature_closure.sector_map_capture_failed")
        entries = {entry.listing_id: entry.sector_name for entry in activated.entries}
        if receipt.effective_session is not None:
            found.extend(
                SectorReclassification(
                    listing_id=listing_id,
                    effective_session=receipt.effective_session,
                    prior_sector=held[listing_id],
                    sector=sector,
                )
                for listing_id, sector in sorted(entries.items())
                if listing_id in held and held[listing_id] != sector
            )
        held.update(entries)
        if receipt.sector_revision == sector_revision:
            break
    return SectorHistory(
        current_revision=sector_revision,
        current=current,
        reclassifications=tuple(
            sorted(
                (item for item in found if item.listing_id in current),
                key=lambda item: (item.effective_session, item.listing_id),
            )
        ),
    )


def copy_sector_history(
    source: PanelClosureArtifactStore, target: PanelClosureArtifactStore, sector_revision: str
) -> SectorHistory:
    """Copy what `sector_history_as_of` reads for one revision, so the target answers the same.

    The revision's map, and each activation receipt observed up to the revision's own with the
    map it names; a store whose receipts do not name the revision gives its map alone.

    Args:
        source: The store the history is read from.
        target: The store it is copied into.
        sector_revision: The revision the history ends at.

    Returns:
        The history, as both stores answer it.

    Raises:
        ValueError: `feature_closure.sector_map_capture_failed` as `sector_history_as_of` raises
            it, or when the target then answers another history.
    """
    history = sector_history_as_of(source, sector_revision)
    maps = {
        value.map_hash: value
        for value in source.find_models(
            category="sector-maps", model=SectorRevisionMap, matches=_every
        )
    }
    receipts = sorted(
        source.find_models(
            category="sector-activation-receipts",
            model=SectorRevisionMapActivationReceipt,
            matches=_every,
        ),
        key=lambda value: (value.observed_at, value.receipt_hash),
    )
    named = [value for value in maps.values() if value.sector_revision == sector_revision]
    if any(value.sector_revision == sector_revision for value in receipts):
        for receipt in receipts:
            named.append(maps[receipt.sector_map_hash])
            target.publish_json(
                category="sector-activation-receipts",
                content_hash=receipt.receipt_hash,
                payload=durable_payload(receipt),
            )
            if receipt.sector_revision == sector_revision:
                break
    for value in named:
        target.publish_json(
            category="sector-maps", content_hash=value.map_hash, payload=durable_payload(value)
        )
    if sector_history_as_of(target, sector_revision) != history:
        raise ValueError("feature_closure.sector_map_capture_failed")
    return history


def _every(_value: object) -> bool:
    return True


__all__ = ["FeatureClosureLedger", "copy_sector_history", "identified", "sector_history_as_of"]
