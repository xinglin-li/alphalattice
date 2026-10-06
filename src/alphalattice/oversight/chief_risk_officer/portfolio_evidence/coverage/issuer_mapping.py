"""One deterministic listing-to-issuer mapping for the issuer scope compiler.

A ticker absent from the admitted registry is a gap, a ticker that resolves to
more than one issuer is a gap, and an issuer that a second listing would bind to
a different CIK is a gap. A gap is never replaced by a guess.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from alphalattice.evidence.alternative_evidence.contracts import SecIssuerRegistrySnapshot
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    PortfolioIssuerMappingFailure,
)


@dataclass(frozen=True, slots=True)
class ListingIssuerResolution:
    """One listing's answer: an issuer, or the reason there is not one."""

    listing_id: str
    ticker: str | None
    entity_id: str | None
    cik: str | None
    failure: PortfolioIssuerMappingFailure | None

    @property
    def mapped(self) -> bool:
        """Report whether this resolution identifies an admitted issuer.

        Returns:
            True when entity_id is present; False for an unresolved mapping gap.
        """
        return self.entity_id is not None


@dataclass(frozen=True, slots=True)
class PortfolioIssuerMapping:
    """Every listing's resolution in the order it was supplied, plus the gaps."""

    resolutions: tuple[ListingIssuerResolution, ...]
    failures: tuple[PortfolioIssuerMappingFailure, ...]
    cik_by_entity: dict[str, str]


def resolve_portfolio_issuer_mapping(
    *,
    registry: SecIssuerRegistrySnapshot,
    listings: tuple[tuple[str, str | None], ...],
) -> PortfolioIssuerMapping:
    """Map ordered `(listing_id, ticker)` pairs onto admitted registry issuers.

    Order is preserved so a caller can zip the answers back onto whatever it was
    iterating -- scenarios, or one book's rows -- without this function needing
    to know which of those it is serving.
    """
    entities_by_cik: dict[str, list[str]] = defaultdict(list)
    for entry in registry.entries:
        entities_by_cik[entry.cik].append(entry.entity_id)
    canonical_entity_by_cik = {cik: min(entity_ids) for cik, entity_ids in entities_by_cik.items()}
    registry_by_ticker: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for entry in registry.entries:
        registry_by_ticker[entry.ticker].append((canonical_entity_by_cik[entry.cik], entry.cik))

    cik_by_entity: dict[str, str] = {}
    resolutions: list[ListingIssuerResolution] = []
    failures: list[PortfolioIssuerMappingFailure] = []
    for listing_id, ticker in listings:
        # A listing with no admitted symbol fails first and separately. It is a
        # different gap from a symbol the registry does not carry, and telling a
        # reader which one it hit is the difference between "we cannot identify
        # this holding" and "this issuer does not file".
        matches = [] if ticker is None else registry_by_ticker.get(ticker, [])
        reason: str | None = None
        if ticker is None:
            reason = "LISTING_NOT_IN_ADMITTED_AUTHORITY"
        elif not matches:
            reason = "TICKER_NOT_IN_SEC_REGISTRY"
        elif len(matches) != 1:
            reason = "TICKER_CIK_AMBIGUOUS"
        if reason is None:
            entity_id, cik = matches[0]
            if cik_by_entity.setdefault(entity_id, cik) != cik:
                reason = "TICKER_CIK_AMBIGUOUS"
        if reason is not None:
            failure = PortfolioIssuerMappingFailure(
                listing_id=listing_id,
                ticker=ticker,
                reason=reason,
            )
            failures.append(failure)
            resolutions.append(
                ListingIssuerResolution(
                    listing_id=listing_id,
                    ticker=ticker,
                    entity_id=None,
                    cik=None,
                    failure=failure,
                )
            )
            continue
        entity_id, cik = matches[0]
        resolutions.append(
            ListingIssuerResolution(
                listing_id=listing_id,
                ticker=ticker,
                entity_id=entity_id,
                cik=cik,
                failure=None,
            )
        )
    return PortfolioIssuerMapping(
        resolutions=tuple(resolutions),
        failures=tuple(failures),
        cik_by_entity=cik_by_entity,
    )


def ordered_mapping_failures(
    failures: tuple[PortfolioIssuerMappingFailure, ...],
) -> tuple[PortfolioIssuerMappingFailure, ...]:
    """The one stable order both compilers publish gaps in."""
    return tuple(
        sorted(failures, key=lambda value: (value.ticker or "", value.listing_id, value.reason))
    )


__all__ = [
    "ListingIssuerResolution",
    "PortfolioIssuerMapping",
    "ordered_mapping_failures",
    "resolve_portfolio_issuer_mapping",
]
