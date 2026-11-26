"""Strict persisted contracts for data truth and provenance."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, model_validator

from alphalattice.kernel.data.enums import (
    LicenseClass,
    UniverseIndex,
)
from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
    NonEmptyString,
    Sha256Hex,
    ShortString,
    UtcDatetime,
    Uuid4,
)
from alphalattice.kernel.shared_kernel.domain.enums import DataValidityClass

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
MissingRatio = Annotated[float, Field(ge=0.0, le=1.0)]
DailyFrequency = Literal["daily"]


class UniverseSource(DomainModel):
    """Retrieved index source with provenance, content hash, and license."""

    source_id: Uuid4
    index: UniverseIndex
    source_uri: NonEmptyString
    selector: NonEmptyString
    retrieved_at: UtcDatetime
    response_hash: Sha256Hex
    license_class: LicenseClass


class UniverseMemberDecision(DomainModel):
    """Eligibility decision for one current-universe symbol."""

    symbol: ShortString
    provider_symbol: ShortString
    company_name: NonEmptyString
    source_indices: tuple[UniverseIndex, ...] = Field(min_length=1)
    calendar_id: Literal["XNYS", "XNAS"]
    eligible: bool
    reasons: tuple[NonEmptyString, ...] = ()


class UniverseManifest(DomainModel):
    """Current-universe source decisions and sorted eligible members."""

    manifest_id: Uuid4
    created_at: UtcDatetime
    as_of_timestamp: UtcDatetime
    construction_rule: NonEmptyString
    sources: tuple[UniverseSource, ...] = Field(min_length=1)
    decisions: tuple[UniverseMemberDecision, ...] = Field(min_length=1)
    members: tuple[ShortString, ...]
    data_validity_class: DataValidityClass
    content_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_members(self) -> UniverseManifest:
        """Require members to match the unique eligible decisions."""
        expected = tuple(sorted(item.symbol for item in self.decisions if item.eligible))
        if self.members != expected:
            raise ValueError("members must be sorted eligible decisions")
        if len(set(self.members)) != len(self.members):
            raise ValueError("universe members must be unique")
        return self


class SymbolQualityReport(DomainModel):
    """Coverage and row-quality diagnostics for one symbol."""

    symbol: ShortString
    calendar_id: Literal["XNYS", "XNAS"]
    expected_sessions: NonNegativeInt
    observed_sessions: NonNegativeInt
    missing_sessions: NonNegativeInt
    missing_ratio: MissingRatio
    maximum_consecutive_gap: NonNegativeInt
    duplicate_rows: NonNegativeInt
    non_monotonic_rows: NonNegativeInt
    invalid_ohlc_rows: NonNegativeInt
    invalid_volume_rows: NonNegativeInt
    invalid_corporate_action_rows: NonNegativeInt
    extreme_move_rows: NonNegativeInt
    first_session: date | None = None
    last_session: date | None = None
    eligible: bool
    reasons: tuple[NonEmptyString, ...] = ()


class DataQualityReport(DomainModel):
    """Per-symbol quality findings and their aggregate eligibility."""

    report_id: Uuid4
    created_at: UtcDatetime
    as_of_timestamp: UtcDatetime
    symbol_reports: tuple[SymbolQualityReport, ...]
    all_eligible: bool
    caveats: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_summary(self) -> DataQualityReport:
        """Reconcile aggregate eligibility with every symbol report."""
        if self.all_eligible != all(item.eligible for item in self.symbol_reports):
            raise ValueError("all_eligible differs from symbol reports")
        return self
