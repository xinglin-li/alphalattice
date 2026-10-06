"""The listing authority, issuer topics and recorded documents of the evidence
review route, shared with its HTTP composition suite.

Test support beside its owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

from datetime import timedelta

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    EvidenceTopic,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    SecIssuerRegistryEntry,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import RecordedEvidenceDocument
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    AdmittedListingTicker,
    AdmittedListingTickerAuthority,
    seal_portfolio_evidence_contract,
)
from tests.alternative_evidence_desk.planted_corpus import (
    _NOW,
    CIKS,
    PLANTED_CLAIMS,
    _recorded_document,
)

TICKERS: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN")

ISSUER_TOPICS: dict[str, EvidenceTopic] = {
    "AAPL": EvidenceTopic.LIQUIDITY_GOING_CONCERN,
    "MSFT": EvidenceTopic.LEGAL_REGULATORY,
    "NVDA": EvidenceTopic.CAPITAL_DILUTION,
    "TSLA": EvidenceTopic.GOVERNANCE_CONTROLS,
    "AMZN": EvidenceTopic.OPERATIONS_SUPPLY,
}


def _documents(
    entities: tuple[str, ...],
    *,
    revisions: tuple[str, ...] = ("release", "transcript"),
    suffix: str = "",
) -> tuple[RecordedEvidenceDocument, ...]:
    """Two official documents per issuer, each stating that issuer's planted claim.

    Two, because a corroborated finding needs two: one document is
    `SINGLE_SOURCE` by construction and can never reach an objection. Both
    are current reports, the form the integrated selection reads (the
    revision names which is the release and which the transcript).
    """

    return tuple(
        _recorded_document(
            entity,
            claims=(PLANTED_CLAIMS[ISSUER_TOPICS[entity]] + suffix,),
            revision=f"{entity.casefold()}-2026-q3-{revision}",
            document_type="8-K",
        )
        for entity in entities
        for revision in revisions
    )


def _registry() -> SecIssuerRegistrySnapshot:
    return seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=_NOW - timedelta(days=1),
        entries=tuple(
            SecIssuerRegistryEntry(
                entity_id=ticker, ticker=ticker, cik=CIKS[ticker], legal_name=f"{ticker} Inc."
            )
            for ticker in TICKERS
        ),
        source_content_hash="1" * 64,
    )


def _listing_authority(
    listing_ids: tuple[str, ...], *, omit: str | None = None
) -> AdmittedListingTickerAuthority:
    """Map the book's own listings onto admitted symbols, deterministically.

    `omit` drops one listing from the authority: that row then has no admitted
    symbol, which is the first of the three mapping gaps.
    """

    return seal_portfolio_evidence_contract(
        AdmittedListingTickerAuthority,
        "authority_hash",
        entries=tuple(
            AdmittedListingTicker(listing_id=listing_id, ticker=TICKERS[index % len(TICKERS)])
            for index, listing_id in enumerate(sorted(listing_ids))
            if listing_id != omit
        ),
    )
