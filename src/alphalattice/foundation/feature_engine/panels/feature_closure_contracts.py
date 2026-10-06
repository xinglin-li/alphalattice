"""Typed contracts for append-only Feature inputs used by future Panels."""

from __future__ import annotations

from datetime import date, datetime
from operator import itemgetter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.feature_engine.panels.closure_contracts import ClosureArtifactRef
from alphalattice.kernel.shared_kernel.identity import canonical_hash

Hash = str


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureBaseClosureCatalogRoot(_Contract):
    """Bind a catalog to its admitted Panel and base Feature closure."""

    kind: Literal["FeatureBaseClosureCatalogRoot"] = "FeatureBaseClosureCatalogRoot"
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    active_panel_snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    phase_one_failure_receipt_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    retention_assessment_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_recipe_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    base_closure_manifest_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    listing_ids: tuple[str, ...]
    sessions: tuple[date, ...]
    factor_ids: tuple[str, ...]
    feature_row_hash_digest: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    root_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_root(self) -> FeatureBaseClosureCatalogRoot:
        """Check the closure axes and the root content hash."""
        _validate_axes(self.listing_ids, self.sessions, self.factor_ids)
        _validate_hash(self, "root_hash")
        return self


class FeatureBaseClosureGenesisRoot(_Contract):
    """Opens a catalog's closure on a workspace that has never published a Panel.

    ``FeatureBaseClosureCatalogRoot`` records the admission of Phase-1 evidence
    for a Panel that already existed, so it requires a failure receipt, a
    retention assessment, and an active snapshot. None of those exist before the
    first Panel, and a workspace cannot build one without an open closure.

    This root carries no predecessor and no prior Panel, and says so by
    structure rather than by zero-filling the fields it cannot honour. It is a
    separate schema in a separate category, so the frozen roots keep their exact
    identity and a loader never has to guess which contract it is holding.
    """

    kind: Literal["FeatureBaseClosureGenesisRoot"] = "FeatureBaseClosureGenesisRoot"
    schema_id: Literal["alphalattice.feature-closure.genesis-root.v1"] = (
        "alphalattice.feature-closure.genesis-root.v1"
    )
    origin: Literal["GENESIS_PANEL_PUBLICATION"] = "GENESIS_PANEL_PUBLICATION"
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    listing_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    feature_row_hash_digest: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    root_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_genesis_root(self) -> FeatureBaseClosureGenesisRoot:
        """Check first-publication axes and the genesis root hash."""
        if not self.listing_ids or self.listing_ids != tuple(sorted(set(self.listing_ids))):
            raise ValueError("genesis closure listing axis must be sorted and unique")
        if not self.factor_ids or len(self.factor_ids) != len(set(self.factor_ids)):
            raise ValueError("genesis closure factor axis must be non-empty and unique")
        _validate_hash(self, "root_hash")
        return self


class FeatureBaseClosureHead(_Contract):
    """Identify the current catalog closure root and transition cursor."""

    kind: Literal["FeatureBaseClosureHead"] = "FeatureBaseClosureHead"
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    root_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    predecessor_head_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    last_transition_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    transition_cursor: int = Field(ge=0)
    feature_row_hash_digest: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    head_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_head(self) -> FeatureBaseClosureHead:
        """Check the head content hash."""
        _validate_hash(self, "head_hash")
        return self


class FeatureBaseClosurePatch(_Contract):
    """Describe a row patch against one catalog closure head."""

    kind: Literal["FeatureBaseClosurePatch"] = "FeatureBaseClosurePatch"
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    base_head_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1)
    factor_ids: tuple[str, ...]
    row_count: int = Field(gt=0)
    first_session: date
    last_session: date
    listing_ids: tuple[str, ...]
    artifact: ClosureArtifactRef
    patch_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_patch(self) -> FeatureBaseClosurePatch:
        """Check patch range, axes, and content hash."""
        if self.first_session > self.last_session:
            raise ValueError("Feature closure patch range is inverted")
        if not self.factor_ids or len(self.factor_ids) != len(set(self.factor_ids)):
            raise ValueError("Feature closure patch factor axis is invalid")
        if not self.listing_ids or self.listing_ids != tuple(sorted(set(self.listing_ids))):
            raise ValueError("Feature closure patch listing axis must be sorted and unique")
        _validate_hash(self, "patch_hash")
        return self


class PreparedFeatureBaseClosureTransition(_Contract):
    """Record the expected row and Store receipts before closure changes."""

    kind: Literal["PreparedFeatureBaseClosureTransition"] = "PreparedFeatureBaseClosureTransition"
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    base_head_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    patch_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1)
    prior_row_hashes: tuple[tuple[str, str, Hash | None], ...]
    next_row_hashes: tuple[tuple[str, str, Hash], ...]
    expected_receipt_hashes: tuple[Hash, ...]
    prepared_at: datetime
    transition_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_transition(self) -> PreparedFeatureBaseClosureTransition:
        """Check transition clock, row keys, receipts, and hash."""
        if self.prepared_at.tzinfo is None or self.prepared_at.utcoffset() is None:
            raise ValueError("Feature closure transition clock must be timezone-aware")
        keys = list(map(_ROW_KEY, self.next_row_hashes))
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("Feature closure transition row keys are invalid")
        if list(map(_ROW_KEY, self.prior_row_hashes)) != keys:
            raise ValueError("Feature closure transition prior and next keys differ")
        if not self.expected_receipt_hashes:
            raise ValueError("Feature closure transition has no expected Store receipt")
        _validate_hash(self, "transition_hash")
        return self


class FeatureBaseClosureTransitionReceipt(_Contract):
    """Record Store readback and revision identities for a transition."""

    kind: Literal["FeatureBaseClosureTransitionReceipt"] = "FeatureBaseClosureTransitionReceipt"
    transition_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    patch_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    store_receipt_hashes: tuple[Hash, ...]
    revision_ids: tuple[Hash, ...]
    readback_disposition: Literal["NEXT_STATE_VERIFIED", "PRIOR_STATE_SAFE_TO_RETRY"]
    observed_at: datetime
    receipt_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> FeatureBaseClosureTransitionReceipt:
        """Check receipt clock and content hash."""
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("Feature closure receipt clock must be timezone-aware")
        _validate_hash(self, "receipt_hash")
        return self


class FeatureBaseClosureTransitionMarker(_Contract):
    """Commit a verified transition to its next closure head."""

    kind: Literal["FeatureBaseClosureTransitionMarker"] = "FeatureBaseClosureTransitionMarker"
    transition_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    next_head_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    marker_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> FeatureBaseClosureTransitionMarker:
        """Check the marker content hash."""
        _validate_hash(self, "marker_hash")
        return self


class PanelColumnClosureHead(_Contract):
    """The closure head of one column catalog a layered Panel's rows were composed from (V92)."""

    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    closure_head_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    closure_transition_cursor: int = Field(ge=0)


class PanelRecoveryClosureBinding(_Contract):
    """Bind a Panel snapshot to the closure and source state used to recover it.

    A layered catalog's Panel binds its base closure's head as every Panel binds its own, and
    each column catalog's head beside it; any other binding holds no column and keeps its hash.
    """

    kind: Literal["PanelRecoveryClosureBinding"] = "PanelRecoveryClosureBinding"
    snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    panel_content_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    panel_binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    closure_head_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    closure_transition_cursor: int = Field(ge=0)
    sector_map_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    listing_ids: tuple[str, ...]
    sessions: tuple[date, ...]
    factor_ids: tuple[str, ...]
    panel_source_state_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    column_closures: tuple[PanelColumnClosureHead, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_binding(self) -> PanelRecoveryClosureBinding:
        """Check recovery axes and binding content hash."""
        _validate_axes(self.listing_ids, self.sessions, self.factor_ids)
        _validate_hash(self, "binding_hash")
        return self


class SectorRevisionMapActivationReceipt(_Contract):
    """Record activation of a sector revision map and its Store receipt."""

    kind: Literal["SectorRevisionMapActivationReceipt"] = "SectorRevisionMapActivationReceipt"
    manifest_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    sector_map_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    store_receipt_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    changed: bool
    observed_at: datetime
    effective_session: date | None = Field(default=None, exclude_if=lambda value: value is None)
    """The session its reclassifications take effect from (V346); absent before the rule."""
    receipt_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> SectorRevisionMapActivationReceipt:
        """Check activation clock and receipt content hash."""
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("sector activation receipt clock must be timezone-aware")
        _validate_hash(self, "receipt_hash")
        return self


_ROW_KEY = itemgetter(0, 1)
"""A transition row's key: its session and listing."""


class FeatureClosureFirstLoad(_Contract):
    """A column part's first build, its rows written without per-key transitions (V92).

    Prepared while the part's head is its genesis head and the part holds no row: the build
    writes the part's rows as it computes them, and completion advances the head to the digest
    of every row the part then holds, naming this load as its last transition. A first load a
    later build finds pending was interrupted: the part's rows are discarded, back to the empty
    part its genesis head attests, and the part is built again.
    """

    kind: Literal["FeatureClosureFirstLoad"] = "FeatureClosureFirstLoad"
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    base_head_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str = Field(min_length=1)
    prepared_at: datetime
    first_load_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_first_load(self) -> FeatureClosureFirstLoad:
        """Check the load's clock and content hash."""
        if self.prepared_at.tzinfo is None or self.prepared_at.utcoffset() is None:
            raise ValueError("Feature closure first load clock must be timezone-aware")
        _validate_hash(self, "first_load_hash")
        return self


def _validate_axes(
    listing_ids: tuple[str, ...], sessions: tuple[date, ...], factor_ids: tuple[str, ...]
) -> None:
    if not listing_ids or len(listing_ids) != len(set(listing_ids)):
        raise ValueError("closure listing axis must be non-empty and unique")
    if not sessions or sessions != tuple(sorted(set(sessions))):
        raise ValueError("closure session axis must be sorted and unique")
    if not factor_ids or len(factor_ids) != len(set(factor_ids)):
        raise ValueError("closure factor axis must be non-empty and unique")


def _validate_hash(model: BaseModel, field_name: str) -> None:
    expected = canonical_hash(model.model_dump(mode="json", exclude={field_name}))
    if getattr(model, field_name) != expected:
        raise ValueError(f"{model.__class__.__name__} identity is invalid")


__all__ = [
    "FeatureBaseClosureCatalogRoot",
    "FeatureBaseClosureGenesisRoot",
    "FeatureBaseClosureHead",
    "FeatureBaseClosurePatch",
    "FeatureBaseClosureTransitionMarker",
    "FeatureBaseClosureTransitionReceipt",
    "FeatureClosureFirstLoad",
    "PanelColumnClosureHead",
    "PanelRecoveryClosureBinding",
    "PreparedFeatureBaseClosureTransition",
    "SectorRevisionMapActivationReceipt",
]
