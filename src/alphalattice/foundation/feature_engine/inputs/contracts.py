"""Stable Feature Input evidence contracts shared with storage codecs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from alphalattice.kernel.shared_kernel.identity import canonical_hash

type QualificationDomain = Literal["MARKET_DATA", "BASE_FEATURES", "SECTOR_REFERENCE"]
FEATURE_QUALIFICATION_DOMAINS: tuple[QualificationDomain, ...] = (
    "BASE_FEATURES",
    "SECTOR_REFERENCE",
)


@dataclass(frozen=True)
class FeatureCandidateRecheck:
    """A small work scope over an already approved, raw-qualified candidate root."""

    parent_manifest_revision: str
    listing_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            re.fullmatch(r"[0-9a-f]{64}", self.parent_manifest_revision) is None
            or not self.listing_ids
            or self.listing_ids != tuple(sorted(set(self.listing_ids)))
            or any(not value for value in self.listing_ids)
        ):
            raise ValueError("workspace_maintenance.candidate_recheck_scope_invalid")


def quarantine_qualification_domain(reasons: tuple[str, ...]) -> QualificationDomain:
    """Raw, baseline-Feature and Sector debts require their own evidence."""
    if reasons and all(reason == "SECTOR_CLASSIFICATION_UNAVAILABLE" for reason in reasons):
        return "SECTOR_REFERENCE"
    return (
        "BASE_FEATURES"
        if reasons
        and all(
            reason.startswith("BASE_FEATURE_UNAVAILABLE:")
            or reason == "BASE_MARKET_OBSERVATION_UNAVAILABLE"
            for reason in reasons
        )
        else "MARKET_DATA"
    )


@dataclass(frozen=True)
class ListingQuarantine:
    """A dated usability exclusion, recoverable through fresh qualification.

    ``execution_receipt_hash`` is the decision that authorized it. A row that
    carries a standing quarantine to its next recheck over re-examined evidence
    names the row it continues in ``continued_from_quarantine_hash`` and keeps
    the original receipt: the continuation is not a new decision. Rows without
    it (every row written before continuation existed) hash as they always
    did.
    """

    listing_id: str
    reason_codes: tuple[str, ...]
    evidence_hash: str
    agent_proposal_hash: str | None
    execution_receipt_hash: str
    recheck_after_at: datetime
    quarantine_hash: str
    continued_from_quarantine_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        listing_id: str,
        reason_codes: tuple[str, ...],
        evidence_hash: str,
        execution_receipt_hash: str,
        recheck_after_at: datetime,
        agent_proposal_hash: str | None = None,
        continued_from_quarantine_hash: str | None = None,
    ) -> ListingQuarantine:
        """Seal a dated quarantine and its decision receipt into one identity."""
        if recheck_after_at.tzinfo is None:
            raise ValueError("quarantine recheck time must be timezone-aware")
        payload = {
            "listing_id": listing_id,
            "reason_codes": sorted(set(reason_codes)),
            "evidence_hash": evidence_hash,
            "agent_proposal_hash": agent_proposal_hash,
            "execution_receipt_hash": execution_receipt_hash,
            "recheck_after_at": recheck_after_at.isoformat(),
        }
        if continued_from_quarantine_hash is not None:
            payload["continued_from_quarantine_hash"] = continued_from_quarantine_hash
        return cls(
            listing_id=listing_id,
            reason_codes=tuple(payload["reason_codes"]),
            evidence_hash=evidence_hash,
            agent_proposal_hash=agent_proposal_hash,
            execution_receipt_hash=execution_receipt_hash,
            recheck_after_at=recheck_after_at,
            quarantine_hash=canonical_hash(payload),
            continued_from_quarantine_hash=continued_from_quarantine_hash,
        )


@dataclass(frozen=True)
class MaterializedFeatureQualification:
    """Dated computability of a named Feature scope, not a perpetual stock guarantee."""

    session: date
    catalog_hash: str
    factor_ids: tuple[str, ...]
    eligible_listing_ids: tuple[str, ...]
    pending_listing_ids: tuple[str, ...]
    exclusions: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    evidence_hash: str

    def summary(self) -> dict[str, object]:
        """Only failed/pending names, not another complete daily Universe copy."""
        return {
            "session": self.session.isoformat(),
            "catalog_hash": self.catalog_hash,
            "factor_ids": self.factor_ids,
            "eligible_count": len(self.eligible_listing_ids),
            "pending_listing_ids": self.pending_listing_ids,
            "exclusions": self.exclusions,
            "nominal_count": len(self.eligible_listing_ids)
            + len(self.pending_listing_ids)
            + len(self.exclusions),
            "evidence_hash": self.evidence_hash,
        }


__all__ = ["ListingQuarantine", "MaterializedFeatureQualification"]
