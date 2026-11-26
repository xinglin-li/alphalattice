"""Typed records persisted by Feature and Panel repositories."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class SectorReferenceState:
    """Current sector names and revision for a manifest's listings."""

    sector_revision: str
    sector_by_listing_id: dict[str, str]
    sector_observed_at: datetime
    sector_distribution: dict[str, int]


@dataclass(frozen=True)
class BindableSectorEvidence:
    """The exact evidence a derived manifest would inherit, read before mutating.

    ``sector_revision`` is bound to manifest identity, so rebinding activates a
    revision the closure ledger has not yet seen. Publishing this first is what
    lets the ledger keep its map-before-mutation order.
    """

    observations: tuple[Mapping[str, object], ...]
    sector_revision: str
    receipt_hash: str
    source_observed_at: datetime


@dataclass(frozen=True)
class PanelContentIdentity:
    """Content hash and coverage of a materialized feature panel."""

    panel_content_hash: str
    row_count: int
    availability_count: int
    history_start: date
    as_of_session: date


@dataclass(frozen=True)
class FeatureIneligibilityRun:
    """Stored span and cause of a listing's feature ineligibility."""

    run_id: str
    listing_id: str
    catalog_hash: str
    factor_id: str
    reason: str
    first_session: date
    last_session: date
    first_observation_count: int
    observation_cap: int
    materialization_receipt_hash: str
    updated_at: datetime


@dataclass(frozen=True)
class FeatureSourceWindow:
    """The exact numerical input window of a completed Feature computation."""

    first_output_session: date
    input_start: date
    input_end: date
    stock_input_hash: str
    market_input_hash: str

    def __post_init__(self) -> None:
        """Reject an invalid input window or noncanonical input hashes."""
        if not self.input_start <= self.first_output_session <= self.input_end or any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (self.stock_input_hash, self.market_input_hash)
        ):
            raise ValueError("feature.source_window_invalid")

    def to_payload(self) -> dict[str, object]:
        """Serialize the window with ISO-formatted session dates."""
        return {
            "first_output_session": self.first_output_session.isoformat(),
            "input_start": self.input_start.isoformat(),
            "input_end": self.input_end.isoformat(),
            "stock_input_hash": self.stock_input_hash,
            "market_input_hash": self.market_input_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> FeatureSourceWindow:
        """Parse an exact window payload and validate its bounds and hashes."""
        if set(payload) != {
            "first_output_session",
            "input_start",
            "input_end",
            "stock_input_hash",
            "market_input_hash",
        }:
            raise ValueError("feature.source_window_invalid")
        return cls(
            first_output_session=date.fromisoformat(str(payload["first_output_session"])),
            input_start=date.fromisoformat(str(payload["input_start"])),
            input_end=date.fromisoformat(str(payload["input_end"])),
            stock_input_hash=str(payload["stock_input_hash"]),
            market_input_hash=str(payload["market_input_hash"]),
        )


@dataclass(frozen=True)
class FeatureRowIdentity:
    """What one canonical Feature row is, computed once for one persistence.

    Derived from the write's own rows (the canonical cutoff payload and its set
    hash, and the row content hash) by the Host worker that computed them, or by
    the closure coordinator when the write carries none. The coordinator uses
    them to decide what changed and to seal its transition; the store consumes
    the same values for the same rows in the same batch instead of deriving them
    a second time. They name nothing outside that batch.
    """

    session_date: date
    canonical_cutoffs: str
    cutoff_set_hash: str
    row_hash: str


@dataclass(frozen=True)
class FeatureMaterializationWrite:
    """One listing-scoped Feature write; batch membership is never identity."""

    listing_id: str
    catalog_hash: str
    rows: Sequence[Mapping[str, object]] | pd.DataFrame
    ineligibility: Sequence[Mapping[str, object]] | pd.DataFrame
    raw_input_hash: str
    action_set_hash_value: str
    market_reference_revision: str
    idempotency_key: str
    revision_reason: str
    observed_at: datetime
    factor_ids: Sequence[str] | None = None
    rows_are_canonical: bool = False
    source_window: FeatureSourceWindow | None = None
    # One identity per canonical row, in row order; only with canonical rows.
    row_identities: tuple[FeatureRowIdentity, ...] | None = None
    # The runtime rows the store holds at this write's sessions, by session, as the closure
    # coordinator read them for its whole transition under the writer's hold (a session
    # absent holds no row); given, the store consumes them instead of reading them again.
    existing_rows: Mapping[date, Mapping[str, object]] | None = None


__all__ = [
    "BindableSectorEvidence",
    "FeatureIneligibilityRun",
    "FeatureMaterializationWrite",
    "FeatureRowIdentity",
    "FeatureSourceWindow",
    "PanelContentIdentity",
    "SectorReferenceState",
]
