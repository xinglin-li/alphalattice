"""Typed logical, derivation, and physical identities for Feature Panels."""

from __future__ import annotations

from datetime import date, datetime
from typing import ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

Hash = str


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Optional fields a model gained after documents were sealed without
    # them: absent values are excluded from its identity, so a document
    # written before the field re-derives the hash it recorded.
    IDENTITY_EXCLUDES_NONE_FIELDS: ClassVar[frozenset[str]] = frozenset()


class PanelLogicalFactorDigest(_Contract):
    """Record one factor's value and null digest in logical content."""

    factor_id: str = Field(min_length=1)
    value_null_digest: Hash = Field(pattern=r"^[0-9a-f]{64}$")


class PanelLogicalAnnualContent(_Contract):
    """Describe a year's ordered keys, row count, and factor digests."""

    year: int = Field(ge=1900, le=2200)
    first_session: date
    last_session: date
    session_count: int = Field(gt=0)
    row_count: int = Field(gt=0)
    ordered_key_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    factor_digests: tuple[PanelLogicalFactorDigest, ...]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_annual_content(self) -> PanelLogicalAnnualContent:
        """Check annual session range and unique factor digests."""
        if self.first_session > self.last_session:
            raise ValueError("logical Panel annual range is inverted")
        factor_ids = tuple(item.factor_id for item in self.factor_digests)
        if not factor_ids or len(factor_ids) != len(set(factor_ids)):
            raise ValueError("logical Panel annual factor digests are invalid")
        return self


class PanelLogicalMembershipEpoch(_Contract):
    """Sessions whose cross-section holds one member set, and how many."""

    first_session: date
    last_session: date
    member_count: int = Field(gt=0)
    membership_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_epoch(self) -> PanelLogicalMembershipEpoch:
        """Check the membership epoch has a forward session range."""
        if self.first_session > self.last_session:
            raise ValueError("logical Panel membership epoch range is inverted")
        return self


def expected_logical_rows(
    sessions: tuple[date, ...],
    listing_ids: tuple[str, ...],
    membership_epochs: tuple[PanelLogicalMembershipEpoch, ...] | None,
) -> int:
    """How many rows a logical Panel holds: each session's members, or the dense grid."""
    if membership_epochs is None:
        return len(sessions) * len(listing_ids)
    total = 0
    for session in sessions:
        for epoch in membership_epochs:
            if epoch.first_session <= session <= epoch.last_session:
                total += epoch.member_count
                break
        else:
            raise ValueError("logical Panel session has no membership epoch")
    return total


class PanelLogicalContentManifest(_Contract):
    """Identify Panel content independently of its physical chunks."""

    kind: Literal["PanelLogicalContentManifest"] = "PanelLogicalContentManifest"
    sessions: tuple[date, ...]
    listing_ids: tuple[str, ...]
    factor_ids: tuple[str, ...]
    annual_content: tuple[PanelLogicalAnnualContent, ...]
    availability_semantics_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    # Present for a Panel whose sessions hold their own members: ``listing_ids``
    # is then the calculation axis and each session's row count is its
    # epoch's member count. Absent for a dense Panel, where every session
    # holds the whole axis, and excluded from its identity so a manifest
    # published before the field re-derives the hash it recorded.
    membership_epochs: tuple[PanelLogicalMembershipEpoch, ...] | None = None
    logical_manifest_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    IDENTITY_EXCLUDES_NONE_FIELDS: ClassVar[frozenset[str]] = frozenset({"membership_epochs"})

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_manifest(self) -> PanelLogicalContentManifest:
        """Check ordered axes, annual coverage, row counts, and hash."""
        if not self.sessions or self.sessions != tuple(sorted(set(self.sessions))):
            raise ValueError("logical Panel session axis must be sorted and unique")
        if not self.listing_ids or len(self.listing_ids) != len(set(self.listing_ids)):
            raise ValueError("logical Panel listing axis must be non-empty and unique")
        if not self.factor_ids or len(self.factor_ids) != len(set(self.factor_ids)):
            raise ValueError("logical Panel factor axis must be non-empty and unique")
        years = tuple(item.year for item in self.annual_content)
        if years != tuple(sorted(set(years))):
            raise ValueError("logical Panel annual content must be sorted and unique")
        if (
            tuple(item.factor_id for item in self.annual_content[0].factor_digests)
            != self.factor_ids
        ):
            raise ValueError("logical Panel annual content differs from the factor axis")
        if any(
            tuple(value.factor_id for value in item.factor_digests) != self.factor_ids
            for item in self.annual_content
        ):
            raise ValueError("logical Panel factor axis changes across years")
        if sum(item.session_count for item in self.annual_content) != len(self.sessions):
            raise ValueError("logical Panel annual sessions do not cover the session axis")
        expected_rows = expected_logical_rows(
            self.sessions, self.listing_ids, self.membership_epochs
        )
        if sum(item.row_count for item in self.annual_content) != expected_rows:
            raise ValueError("logical Panel row count does not match its axes")
        _validate_hash(self, "logical_manifest_hash")
        return self


class PanelLogicalRevision(_Contract):
    """Bind logical Panel content to catalog and policy semantics."""

    kind: Literal["PanelLogicalRevision"] = "PanelLogicalRevision"
    logical_manifest_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_semantics_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    cross_section_policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    listing_set_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    calendar_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_revision(self) -> PanelLogicalRevision:
        """Check the logical Panel revision content hash."""
        _validate_hash(self, "logical_panel_hash")
        return self


class PanelDerivationBinding(_Contract):
    """Record the source identities used to derive a logical Panel."""

    kind: Literal["PanelDerivationBinding"] = "PanelDerivationBinding"
    legacy_snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_panel_content_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_panel_binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    sector_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    sector_map_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    catalog_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    policy_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    spy_revision: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    temporal_identity_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    closure_head_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A legacy binding's closure ledger head, which the write batch decides;
    the ledger keeps it by snapshot, outside the Panel's identity."""
    recovery_recipe_hash: Hash | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recovery_disposition: Literal["RECOVERY_CLOSURE_VERIFIED", "PHYSICAL_RETENTION_REQUIRED"]
    derivation_binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_binding(self) -> PanelDerivationBinding:
        """Check derivation inputs and binding content hash."""
        if self.recovery_disposition == "RECOVERY_CLOSURE_VERIFIED" and (
            self.sector_map_hash is None or self.recovery_recipe_hash is None
        ):
            raise ValueError("verified Panel recovery requires a map and recipe")
        if self.recovery_disposition == "PHYSICAL_RETENTION_REQUIRED" and (
            self.recovery_recipe_hash is not None
        ):
            raise ValueError("physically pinned Panel cannot claim a recovery recipe")
        _validate_hash(self, "derivation_binding_hash")
        return self


class PanelPhysicalChunk(_Contract):
    """Locate a physical chunk of a logical Panel materialization."""

    year: int = Field(ge=1900, le=2200)
    logical_annual_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_chunk_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str = Field(min_length=1)
    raw_sha256: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)
    row_count: int = Field(gt=0)
    schema_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    compression_codecs: tuple[str, ...]
    parquet_created_by: str | None = None


class PanelPhysicalMaterializationManifest(_Contract):
    """Associate physical chunks with their logical Panel identity."""

    kind: Literal["PanelPhysicalMaterializationManifest"] = "PanelPhysicalMaterializationManifest"
    logical_panel_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    physical_chunks: tuple[PanelPhysicalChunk, ...]
    encoder_provenance: tuple[tuple[str, str], ...]
    physical_materialization_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_materialization(self) -> PanelPhysicalMaterializationManifest:
        """Check physical chunk coverage and manifest identity."""
        years = tuple(item.year for item in self.physical_chunks)
        if not years or years != tuple(sorted(set(years))):
            raise ValueError("physical Panel chunks must be sorted and unique")
        _validate_hash(self, "physical_materialization_hash")
        return self


class PanelPublicationMarker(_Contract):
    """Record the published logical and physical Panel identities."""

    kind: Literal["PanelPublicationMarker"] = "PanelPublicationMarker"
    legacy_snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_binding_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    physical_materialization_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    logical_semantic_index_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    retention_disposition: Literal[
        "REMATERIALIZATION_VERIFIED_CURRENT_ENVIRONMENT",
        "PHYSICAL_RETENTION_REQUIRED",
    ]
    published_at: datetime
    marker_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> PanelPublicationMarker:
        """Check publication references and marker identity."""
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("Panel publication marker clock must be timezone-aware")
        _validate_hash(self, "marker_hash")
        return self


class LegacyPanelLogicalEquivalenceAttestation(_Contract):
    """Attest that a legacy Panel has the same logical content."""

    kind: Literal["LegacyPanelLogicalEquivalenceAttestation"] = (
        "LegacyPanelLogicalEquivalenceAttestation"
    )
    legacy_snapshot_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    legacy_panel_content_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    logical_panel_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")
    axes_equal: Literal[True] = True
    ieee_values_and_null_masks_equal: Literal[True] = True
    availability_semantics_equal: Literal[True] = True
    verified_annual_hashes: tuple[Hash, ...]
    attestation_hash: Hash = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_attestation(self) -> LegacyPanelLogicalEquivalenceAttestation:
        """Check the legacy equivalence evidence and its identity."""
        if not self.verified_annual_hashes:
            raise ValueError("logical equivalence requires annual verification evidence")
        _validate_hash(self, "attestation_hash")
        return self


def identity_payload(model: BaseModel, field_name: str) -> dict[str, object]:
    """The payload a logical document's identity is computed over."""
    excluded = {
        field_name,
        *(
            name
            for name in getattr(model, "IDENTITY_EXCLUDES_NONE_FIELDS", frozenset())
            if getattr(model, name, None) is None
        ),
    }
    return cast(dict[str, object], model.model_dump(mode="json", exclude=excluded))


def _validate_hash(model: BaseModel, field_name: str) -> None:
    if getattr(model, field_name) != canonical_hash(identity_payload(model, field_name)):
        raise ValueError(f"{model.__class__.__name__} identity is invalid")


__all__ = [
    "LegacyPanelLogicalEquivalenceAttestation",
    "PanelDerivationBinding",
    "PanelLogicalAnnualContent",
    "PanelLogicalContentManifest",
    "PanelLogicalFactorDigest",
    "PanelLogicalMembershipEpoch",
    "PanelLogicalRevision",
    "PanelPhysicalChunk",
    "PanelPhysicalMaterializationManifest",
    "PanelPublicationMarker",
    "expected_logical_rows",
    "identity_payload",
]
