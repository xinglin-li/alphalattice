"""Typed records shared by the local-first market-data prototype."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class CandidateDataRecheck:
    """A failed acquisition scope under an already approved candidate document."""

    prior_onboarding_id: str
    candidate_manifest_hash: str
    qualified_parent_revision: str
    listing_ids: tuple[str, ...]
    history_start: date

    def __post_init__(self) -> None:
        """Reject noncanonical recheck identities and listing order.

        Raises:
            ValueError: If the scope cannot identify one approved recheck.

        """
        if (
            any(
                re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in (
                    self.prior_onboarding_id,
                    self.candidate_manifest_hash,
                    self.qualified_parent_revision,
                )
            )
            or not self.listing_ids
            or self.listing_ids != tuple(sorted(set(self.listing_ids)))
        ):
            raise ValueError("workspace_maintenance.candidate_data_recheck_scope_invalid")


@dataclass(frozen=True)
class RawDailyBar:
    """Canonical provider OHLCV observation, excluding mutable adjustment views."""

    listing_id: str
    provider: str
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class ProviderAdjustedClosePoint:
    """Ephemeral provider adjustment evidence used only by a full audit.

    Providers can recalculate every historical adjusted close after a later
    corporate-action correction.  It is deliberately separate from a
    canonical daily bar, so that recalculation is neither a raw-bar mutation
    nor a ``bar_revision`` storm.
    """

    listing_id: str
    provider: str
    session_date: date
    adjusted_close: float


@dataclass(frozen=True)
class CorporateActionEvent:
    """Record a provider-observed action without treating it as final truth."""

    listing_id: str
    provider: str
    effective_date: date
    action_kind: Literal["SPLIT", "CASH_DIVIDEND", "CAPITAL_GAIN", "SPIN_OFF"]
    # This is deliberately not called ``split_ratio`` internally.  Its value
    # is new shares per old share: 2.0 means a 2-for-1 split and 0.1 means a
    # 1-for-10 reverse split.  Feature snapshots retain ``split_ratio`` as a
    # compatibility alias for the established research vocabulary.
    new_shares_per_old_share: float | None = None
    cash_amount: float | None = None
    provisional: bool = True
    provenance: str = "provider-observation"


class HealthSubject(BaseModel):
    """Describe one listing's readiness in a Data health report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    listing_id: str
    symbol: str
    latest_session: date | None
    expected_session: date
    quality_state: Literal[
        "READY",
        "DATA_UNREADY",
        "LISTING_REQUIRES_REVIEW",
        "QUALITY_INELIGIBLE",
    ]
    provider_failure: str | None = None


class DataHealthReport(BaseModel):
    """Group listing health under one manifest and market session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    case_token: str
    market_profile_id: str
    manifest_revision: str
    as_of_session: date
    subjects: tuple[HealthSubject, ...]


class DataRemediationProposal(BaseModel):
    """An option selection, never a capability to mutate the local data store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    case_token: str
    evidence_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    option_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,119}$")
    rationale: str = Field(min_length=1, max_length=1200)


@dataclass(frozen=True)
class FailureEvidence:
    """Bind a data failure to a listing, market profile, range, and time."""

    listing_id: str
    market_profile_id: str
    failure_code: str
    range_start: date
    range_end: date
    observed_at: datetime
