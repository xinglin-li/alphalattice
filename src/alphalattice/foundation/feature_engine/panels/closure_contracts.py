"""Typed contracts for immutable Feature Panel rematerialization closures.

These contracts describe recovery evidence.  They never replace the active
Feature Panel snapshot or authorize eviction of an existing physical chunk.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from datetime import date, datetime
from typing import ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_BINDING,
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    PanelSourceExclusion,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

Hash = str


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(frozen=True, extra="forbid")


def durable_payload(
    model: BaseModel, *, exclude: AbstractSet[str] = frozenset()
) -> dict[str, object]:
    """The JSON payload a closure document is hashed over and published as.

    A model whose identity excludes absent optional values
    (``IDENTITY_EXCLUDES_NONE``) is serialized without them too: the bytes
    published under a content hash are that identity payload plus the hash,
    so the shape a historical hash was computed over is the shape written
    back under it. A writer that emitted ``null`` for a field the historical
    document never had would publish different bytes under the same hash,
    and the store rightly refuses that. Every sealer and writer of closure
    documents reads the rule here rather than restating it.
    """
    return cast(
        dict[str, object],
        model.model_dump(
            mode="json",
            exclude=set(exclude) or None,
            exclude_none=bool(getattr(model, "IDENTITY_EXCLUDES_NONE", False)),
        ),
    )


class ClosureArtifactRef(_Contract):
    """Locate an immutable artifact by URI, bytes, and digest."""

    kind: str = Field(min_length=1)
    content_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    physical_sha256: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    uri: str = Field(min_length=1)


class SectorRevisionEntry(_Contract):
    """Map a sector source revision to its ordered membership."""

    listing_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    sector_name: str = Field(min_length=1)
    sector_key: str | None = None
    payload_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")


class SectorRevisionMap(_Contract):
    """Record the sector revisions used by a Panel closure."""

    kind: Literal["SectorRevisionMap"] = "SectorRevisionMap"
    manifest_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[SectorRevisionEntry, ...]
    map_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_map(self) -> SectorRevisionMap:
        """Check sector entries and the map content hash."""
        listing_ids = tuple(item.listing_id for item in self.entries)
        if not listing_ids or listing_ids != tuple(sorted(set(listing_ids))):
            raise ValueError("sector revision entries must be sorted and unique")
        identity = self.model_dump(mode="json", exclude={"map_hash"})
        if self.map_hash != canonical_hash(identity):
            raise ValueError("sector revision map hash is invalid")
        return self


class SectorRevisionBackfillReceipt(_Contract):
    """Record a verified sector revision backfill."""

    kind: Literal["SectorRevisionBackfillReceipt"] = "SectorRevisionBackfillReceipt"
    sector_map_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    revision_ledger_row_count: int = Field(ge=0)
    correction_row_count: int = Field(ge=0)
    reconstruction_basis: Literal[
        "CURRENT_ROWS_EXACT",
        "CURRENT_ROWS_WITH_ZERO_CORRECTION_LEDGER",
    ]
    receipt_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> SectorRevisionBackfillReceipt:
        """Check backfill evidence and receipt content hash."""
        if (
            self.reconstruction_basis == "CURRENT_ROWS_WITH_ZERO_CORRECTION_LEDGER"
            and self.correction_row_count != 0
        ):
            raise ValueError("historical sector map backfill requires a zero-correction ledger")
        identity = self.model_dump(mode="json", exclude={"receipt_hash"})
        if self.receipt_hash != canonical_hash(identity):
            raise ValueError("sector revision backfill receipt hash is invalid")
        return self


class PanelBaseKeyChunk(_Contract):
    """Locate the annual key axis of a base value closure."""

    year: int = Field(ge=1900, le=2200)
    key_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    first_session: date
    last_session: date
    listing_count: int = Field(gt=0)
    session_count: int = Field(gt=0)
    artifact: ClosureArtifactRef


class PanelBaseValueChunk(_Contract):
    """Locate one factor's values on an annual key axis."""

    year: int = Field(ge=1900, le=2200)
    factor_id: str = Field(min_length=1)
    key_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    value_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    null_count: int = Field(ge=0)
    artifact: ClosureArtifactRef


class PanelBaseValueClosureManifest(_Contract):
    """Bind annual key and factor chunks into a recoverable base closure."""

    kind: Literal["PanelBaseValueClosureManifest"] = "PanelBaseValueClosureManifest"
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    history_start: date
    as_of_session: date
    factor_ids: tuple[str, ...]
    key_chunks: tuple[PanelBaseKeyChunk, ...]
    value_chunks: tuple[PanelBaseValueChunk, ...]
    absent_source_keys: ClosureArtifactRef | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    manifest_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_manifest(self) -> PanelBaseValueClosureManifest:
        """Check chunk coverage, key alignment, and manifest identity."""
        if self.absent_source_keys is not None and (
            self.absent_source_keys.kind != "PanelBaseKeyChunk"
            or self.absent_source_keys.row_count < 1
        ):
            raise ValueError("base closure source absence reference is invalid")
        if not self.factor_ids or len(self.factor_ids) != len(set(self.factor_ids)):
            raise ValueError("base closure factor IDs must be non-empty and unique")
        years = tuple(item.year for item in self.key_chunks)
        if not years or years != tuple(sorted(set(years))):
            raise ValueError("base closure key years must be sorted and unique")
        expected = {(year, factor_id) for year in years for factor_id in self.factor_ids}
        actual = {(item.year, item.factor_id) for item in self.value_chunks}
        if actual != expected or len(actual) != len(self.value_chunks):
            raise ValueError("base closure value chunks are incomplete or duplicated")
        keys = {item.year: item.key_hash for item in self.key_chunks}
        if any(item.key_hash != keys[item.year] for item in self.value_chunks):
            raise ValueError("base closure value chunk uses the wrong key axis")
        identity = self.model_dump(mode="json", exclude={"manifest_hash"})
        if self.manifest_hash != canonical_hash(identity):
            raise ValueError("base closure manifest hash is invalid")
        return self


class PanelAvailabilityClosure(_Contract):
    """Record availability rows for a Panel binding and session range."""

    kind: Literal["PanelAvailabilityClosure"] = "PanelAvailabilityClosure"
    panel_binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    history_start: date
    as_of_session: date
    availability_count: int = Field(gt=0)
    artifact: ClosureArtifactRef
    closure_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_closure(self) -> PanelAvailabilityClosure:
        """Check the availability closure content hash."""
        identity = self.model_dump(mode="json", exclude={"closure_hash"})
        if self.closure_hash != canonical_hash(identity):
            raise ValueError("Panel availability closure hash is invalid")
        return self


class PanelRowReceiptAssignment(_Contract):
    """Bind a Panel content identity to its row receipt artifact."""

    kind: Literal["PanelRowReceiptAssignment"] = "PanelRowReceiptAssignment"
    panel_content_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(gt=0)
    artifact: ClosureArtifactRef
    assignment_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_assignment(self) -> PanelRowReceiptAssignment:
        """Check the row receipt assignment content hash."""
        identity = self.model_dump(mode="json", exclude={"assignment_hash"})
        if self.assignment_hash != canonical_hash(identity):
            raise ValueError("Panel row receipt assignment hash is invalid")
        return self


class PanelCrossSectionRecord(_Contract):
    """Sessions of a chunk computed over one cross-section identity."""

    first_session: date
    last_session: date
    cross_section_identity: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    member_count: int = Field(gt=0)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_record(self) -> PanelCrossSectionRecord:
        """Check that the cross-section session range is forward."""
        if self.first_session > self.last_session:
            raise ValueError("cross-section record range is inverted")
        return self


class ExpectedPanelChunk(_Contract):
    """Describe a year's expected physical Panel chunk and origin."""

    year: int = Field(ge=1900, le=2200)
    first_session: date
    last_session: date
    row_count: int = Field(gt=0)
    chunk_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    physical_sha256: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)
    uri: str = Field(min_length=1)
    # The binding the chunk was hashed and written under, when it is not the
    # recipe's own. Absent for every chunk of a recipe written before
    # partition reuse and for chunks the snapshot's build composed itself;
    # the identity rule excludes absent values, so legacy recipes re-derive
    # their recorded hash unchanged.
    origin_binding_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Under the session-cross-section rule: the identity each range of the
    # chunk's sessions was computed over, as the snapshot manifest recorded
    # it. Absent under the binding rule.
    cross_sections: tuple[PanelCrossSectionRecord, ...] | None = None


class PanelPartitionOrigin(_Contract):
    """One build whose cells a snapshot holds: the inputs of its binding.

    Under the binding rule an origin differs from the recipe's binding only
    in the SPY revision (a compatible base shares membership, sector map,
    catalog and policy). Under the cross-section rule its manifest and
    sector revisions may differ too, and are recorded, so a rematerializer
    can rebuild the exact binding a receipt group or chunk was produced
    under. An origin under a catalog the recipe's only adds columns to
    records that catalog: its batches computed cells the recipe's
    catalog computes alike, and are replayed under their own binding.
    """

    panel_binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    spy_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_revision: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sector_revision: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    catalog_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PanelMembershipBasisRecord(_Contract):
    """Name the membership rule active over a session range."""

    first_session: date
    last_session: date
    basis: str = Field(min_length=1)


class PanelMembershipEpochRecord(_Contract):
    """Sessions with one member set, as the axis listings the set does not hold."""

    first_session: date
    last_session: date
    absent_listing_ids: tuple[str, ...]


class PanelMembershipRecord(_Contract):
    """Which members each session's cross-section held, over the recipe's axis.

    The axis is the recipe's ``listing_ids`` once; each epoch names only the
    axis listings it does not hold, which is the small side of a membership
    that changes a few names a year. The basis ranges say which promise the
    membership of a session is under; the bootstrap reference and journal
    sequence name the Universe record it was resolved from.
    """

    basis_ranges: tuple[PanelMembershipBasisRecord, ...] = Field(min_length=1)
    epochs: tuple[PanelMembershipEpochRecord, ...] = Field(min_length=1)
    journal_sequence: int = Field(ge=0)
    bootstrap_record_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bootstrap_t0_session: date | None = None
    source_exclusions: tuple[PanelSourceExclusion, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_record(self) -> PanelMembershipRecord:
        """Check membership epochs are ordered, disjoint, and unique."""
        previous: PanelMembershipEpochRecord | None = None
        for epoch in self.epochs:
            if epoch.first_session > epoch.last_session:
                raise ValueError("membership epoch range is inverted")
            if previous is not None and epoch.first_session <= previous.last_session:
                raise ValueError("membership epochs must be sorted and disjoint")
            if len(set(epoch.absent_listing_ids)) != len(epoch.absent_listing_ids):
                raise ValueError("membership epoch absent listings must be unique")
            previous = epoch
        return self

    def epoch(self, session: date) -> PanelMembershipEpochRecord:
        """Return the membership epoch covering a session."""
        for epoch in self.epochs:
            if epoch.first_session <= session <= epoch.last_session:
                return epoch
        raise ValueError("feature_panel.recipe_input_mismatch")


class PanelSectorReclassification(_Contract):
    """One listing's move to another Sector, read from its effective session on."""

    listing_id: str = Field(min_length=1)
    effective_session: date
    prior_sector: str = Field(min_length=1)
    sector: str = Field(min_length=1)


class PanelDerivationRecipe(_Contract):
    """Bind Panel source inputs to expected chunks for exact recovery."""

    kind: Literal["PanelDerivationRecipe"] = "PanelDerivationRecipe"
    snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    panel_content_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    panel_binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    schema_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    sector_map_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    spy_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    history_start: date
    as_of_session: date
    listing_ids: tuple[str, ...]
    sessions: tuple[date, ...]
    factor_ids: tuple[str, ...]
    row_hash_factor_ids: tuple[str, ...]
    session_batch_size: int = Field(gt=0)
    base_closure_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    base_value_catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    availability_closure_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    row_receipt_assignment_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    expected_chunks: tuple[ExpectedPanelChunk, ...]
    rematerialization_policy: Literal["CURRENT_ENVIRONMENT_BYTE_PARITY_REQUIRED"]
    # Every build whose cells the snapshot holds, as its manifest recorded
    # them; absent for a snapshot whose manifest recorded none (one build).
    partition_origins: tuple[PanelPartitionOrigin, ...] | None = None
    # The row-identity rule the rows follow and, under the cross-section
    # rule, the membership each session's rows were computed over. Absent
    # for every recipe under the binding rule, whose rows are the dense
    # grid of ``sessions`` times ``listing_ids``.
    row_identity_basis: str | None = None
    membership: PanelMembershipRecord | None = None
    # The reclassifications its sessions read, in the order the Panel recorded them
    # absent while none is in force, so a recipe of such a Panel is what it was.
    sector_reclassifications: tuple[PanelSectorReclassification, ...] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    recipe_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    # The identity and the published shape exclude absent optional values
    # (``durable_payload``), so a recipe written before those fields existed
    # re-derives the hash it recorded and republishes its exact bytes.
    IDENTITY_EXCLUDES_NONE: ClassVar[bool] = True

    @classmethod
    def seal(cls, values: dict[str, object]) -> PanelDerivationRecipe:
        """Validate and hash a derivation recipe from its declared values."""
        provisional = cls.model_construct(**values, recipe_hash="0" * 64)
        identity = durable_payload(provisional, exclude={"recipe_hash"})
        return cast(
            PanelDerivationRecipe,
            cls.model_validate({**values, "recipe_hash": canonical_hash(identity)}),
        )

    def origin_spy_revision(self, panel_binding_hash: Hash) -> Hash:
        """The SPY revision a recorded origin binding was created from."""
        return self.origin_revisions(panel_binding_hash)[2]

    def origin_revisions(self, panel_binding_hash: Hash) -> tuple[Hash, Hash, Hash]:
        """The (manifest, sector, SPY) revisions an origin binding was created from.

        The recipe's own binding resolves to its own revisions; a recorded
        origin under the binding rule shares the recipe's manifest and sector
        revisions and names its SPY revision; one under the cross-section
        rule names all three.
        """
        if panel_binding_hash == self.panel_binding_hash:
            return self.manifest_revision, self.sector_revision, self.spy_revision
        for origin in self.partition_origins or ():
            if origin.panel_binding_hash == panel_binding_hash:
                return (
                    origin.manifest_revision or self.manifest_revision,
                    origin.sector_revision or self.sector_revision,
                    origin.spy_revision,
                )
        raise ValueError("feature_panel.recipe_input_mismatch")

    def origin_catalog_hash(self, panel_binding_hash: Hash) -> Hash:
        """The catalog an origin binding was created from: the recipe's unless it records one."""
        if panel_binding_hash == self.panel_binding_hash:
            return self.catalog_hash
        for origin in self.partition_origins or ():
            if origin.panel_binding_hash == panel_binding_hash:
                return origin.catalog_hash or self.catalog_hash
        raise ValueError("feature_panel.recipe_input_mismatch")

    @property
    def identity_basis(self) -> str:
        """Return the recipe's row identity rule, defaulting to binding identity."""
        return self.row_identity_basis or PANEL_ROW_IDENTITY_BY_BINDING

    def members(self, session: date) -> tuple[str, ...]:
        """The session's members in calculation order; the whole axis under the binding rule."""
        if self.membership is None:
            return self.listing_ids
        absent = set(self.membership.epoch(session).absent_listing_ids)
        return tuple(listing_id for listing_id in self.listing_ids if listing_id not in absent)

    def members_by_session(self) -> dict[date, tuple[str, ...]]:
        """Return each session's members in calculation order."""
        by_epoch: dict[int, tuple[str, ...]] = {}
        result: dict[date, tuple[str, ...]] = {}
        for session in self.sessions:
            if self.membership is None:
                result[session] = self.listing_ids
                continue
            epoch = self.membership.epoch(session)
            key = id(epoch)
            if key not in by_epoch:
                by_epoch[key] = self.members(session)
            result[session] = by_epoch[key]
        return result

    def cross_section_identity(self, session: date) -> str:
        """The recorded cross-section identity of one session's rows."""
        for chunk in self.expected_chunks:
            for item in chunk.cross_sections or ():
                if item.first_session <= session <= item.last_session:
                    return item.cross_section_identity
        raise ValueError("feature_panel.recipe_input_mismatch")

    def expected_row_count(self, sessions: tuple[date, ...] | None = None) -> int:
        """Count rows across the selected sessions and their member sets."""
        selected = self.sessions if sessions is None else sessions
        return sum(len(self.members(session)) for session in selected)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recipe(self) -> PanelDerivationRecipe:
        """Check recipe axes, membership, origins, chunks, and hash."""
        if not self.listing_ids or len(self.listing_ids) != len(set(self.listing_ids)):
            raise ValueError("recipe listing IDs must be non-empty and unique")
        if self.sessions != tuple(sorted(set(self.sessions))):
            raise ValueError("recipe sessions must be sorted and unique")
        if not self.factor_ids or len(self.factor_ids) != len(set(self.factor_ids)):
            raise ValueError("recipe factor IDs must be non-empty and unique")
        if (
            not self.row_hash_factor_ids
            or len(self.row_hash_factor_ids) != len(set(self.row_hash_factor_ids))
            or not set(self.row_hash_factor_ids).issubset(self.factor_ids)
        ):
            raise ValueError("recipe row-hash factor axis is invalid")
        years = tuple(item.year for item in self.expected_chunks)
        if years != tuple(sorted(set(years))):
            raise ValueError("recipe annual chunks must be sorted and unique")
        if self.row_identity_basis not in {None, PANEL_ROW_IDENTITY_BY_CROSS_SECTION}:
            raise ValueError("recipe row identity basis is unknown")
        if (self.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION) != (
            self.membership is not None
        ):
            raise ValueError("recipe membership must accompany the cross-section rule")
        if self.membership is not None:
            axis = set(self.listing_ids)
            if any(item.listing_id not in axis for item in self.membership.source_exclusions or ()):
                raise ValueError("recipe source exclusion names a listing outside its axis")
            for epoch in self.membership.epochs:
                if not set(epoch.absent_listing_ids) <= axis:
                    raise ValueError("recipe membership names a listing outside its axis")
            for chunk in self.expected_chunks:
                if not chunk.cross_sections:
                    raise ValueError("recipe chunk under the cross-section rule has no ranges")
            for session in self.sessions:
                self.cross_section_identity(session)
        if sum(item.row_count for item in self.expected_chunks) != self.expected_row_count():
            raise ValueError("recipe expected row count does not match its axes")
        origins = {item.panel_binding_hash for item in self.partition_origins or ()}
        if self.partition_origins is not None and len(origins) != len(self.partition_origins):
            raise ValueError("recipe partition origins must be unique")
        for chunk in self.expected_chunks:
            if (
                chunk.origin_binding_hash is not None
                and chunk.origin_binding_hash != self.panel_binding_hash
                and chunk.origin_binding_hash not in origins
            ):
                raise ValueError("recipe chunk origin is not a recorded partition origin")
        if self.recipe_hash != canonical_hash(durable_payload(self, exclude={"recipe_hash"})):
            raise ValueError("Panel derivation recipe hash is invalid")
        return self


class PanelRetentionAssessment(_Contract):
    """Record recoverability and potential reclaim for a Panel snapshot."""

    kind: Literal["PanelRetentionAssessment"] = "PanelRetentionAssessment"
    snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    recipe_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    disposition: Literal[
        "REMATERIALIZATION_VERIFIED_CURRENT_ENVIRONMENT",
        "PHYSICAL_RETENTION_REQUIRED",
        "BLOCKED_UNRECOVERABLE_INPUT",
    ]
    existing_panel_bytes: int = Field(ge=0)
    closure_bytes: int = Field(ge=0)
    future_reclaim_candidate_bytes: int = Field(ge=0)
    eviction_authorized: Literal[False] = False
    verified_at: datetime
    assessment_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_assessment(self) -> PanelRetentionAssessment:
        """Check retention disposition, recipe evidence, and hash."""
        if (
            self.disposition == "REMATERIALIZATION_VERIFIED_CURRENT_ENVIRONMENT"
            and self.recipe_hash is None
        ):
            raise ValueError("verified retention assessment requires a derivation recipe")
        if self.disposition == "BLOCKED_UNRECOVERABLE_INPUT" and (
            self.recipe_hash is not None or self.future_reclaim_candidate_bytes != 0
        ):
            raise ValueError("blocked retention assessment cannot claim a recipe or reclaim bytes")
        identity = self.model_dump(mode="json", exclude={"assessment_hash"})
        if self.assessment_hash != canonical_hash(identity):
            raise ValueError("Panel retention assessment hash is invalid")
        return self


__all__ = [
    "ClosureArtifactRef",
    "ExpectedPanelChunk",
    "PanelAvailabilityClosure",
    "PanelBaseKeyChunk",
    "PanelBaseValueChunk",
    "PanelBaseValueClosureManifest",
    "PanelCrossSectionRecord",
    "PanelDerivationRecipe",
    "PanelMembershipBasisRecord",
    "PanelMembershipEpochRecord",
    "PanelMembershipRecord",
    "PanelPartitionOrigin",
    "PanelRetentionAssessment",
    "PanelRowReceiptAssignment",
    "SectorRevisionBackfillReceipt",
    "SectorRevisionEntry",
    "SectorRevisionMap",
    "durable_payload",
]
