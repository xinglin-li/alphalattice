"""Synthetic listing authority, issuer topics and documents shared by review fixtures."""

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
    """Two current official releases per issuer corroborate its planted claim."""

    return tuple(
        _recorded_document(
            entity,
            claims=(
                PLANTED_CLAIMS[ISSUER_TOPICS.get(entity, EvidenceTopic.OPERATIONS_SUPPLY)] + suffix,
            ),
            revision=f"{entity.casefold()}-2026-q3-{revision}",
            document_type="8-K",
        )
        for entity in entities
        for revision in revisions
    )


def _registry(tickers: tuple[str, ...] = TICKERS) -> SecIssuerRegistrySnapshot:
    return seal_contract(
        SecIssuerRegistrySnapshot,
        "registry_hash",
        captured_at=_NOW - timedelta(days=1),
        entries=tuple(
            SecIssuerRegistryEntry(
                entity_id=ticker,
                ticker=ticker,
                cik=CIKS.get(ticker, f"{index % 5 + 1:010d}"),
                legal_name=f"{ticker} Inc.",
            )
            for index, ticker in enumerate(tickers)
        ),
        source_content_hash="1" * 64,
    )


def _listing_authority(
    listing_ids: tuple[str, ...], *, omit: str | None = None, symbols: dict[str, str] | None = None
) -> AdmittedListingTickerAuthority:
    """Admit the book's symbols or a deterministic fixture map, with an optional missing row."""

    return seal_portfolio_evidence_contract(
        AdmittedListingTickerAuthority,
        "authority_hash",
        entries=tuple(
            AdmittedListingTicker(
                listing_id=listing_id,
                ticker=symbols[listing_id]
                if symbols is not None
                else TICKERS[index % len(TICKERS)],
            )
            for index, listing_id in enumerate(sorted(listing_ids))
            if listing_id != omit
        ),
    )
