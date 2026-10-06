"""Runtime discovery of the original current-US research-universe candidates.

This reuses the production project's frozen source definitions and parsers
rather than introducing a second index-scraping implementation. It is a
bootstrap/rebuild helper, never an application-startup network call.

The returned symbols are *candidates*. A symbol becomes a research-whitelist
member only after the deterministic ten-year quality gate records it as
eligible in a frozen workspace manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from alphalattice.kernel.data.enums import UniverseIndex
from alphalattice.kernel.data.universe import (
    FROZEN_UNIVERSE_SOURCES,
    FetchedUniverseSource,
    fetch_universe_source,
    parse_membership,
    provider_symbol,
)


@dataclass(frozen=True)
class ResearchWhitelistStandard:
    """The original project's current-universe construction and quality rules."""

    construction_rule: str
    minimum_history_calendar_years: int
    maximum_missing_ratio: float
    maximum_consecutive_missing_sessions: int
    data_validity_class: str


ORIGINAL_RESEARCH_WHITELIST_STANDARD = ResearchWhitelistStandard(
    construction_rule="current SP500 union NASDAQ100 union DJIA; quality filtered",
    minimum_history_calendar_years=10,
    maximum_missing_ratio=0.02,
    maximum_consecutive_missing_sessions=20,
    data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
)


@dataclass(frozen=True)
class CandidateSource:
    """One exact current-index response used to construct a candidate union."""

    index: str
    source_uri: str
    selector: str
    retrieved_at: datetime
    response_hash: str
    license_class: str


@dataclass(frozen=True)
class CandidateMembershipEvidence:
    """A source display label for one ticker; this is not issuer identity."""

    index: str
    company_name: str


@dataclass(frozen=True)
class CurrentUniverseCandidate:
    """A ticker appearing in at least one current index source."""

    symbol: str
    provider_symbol: str
    source_memberships: tuple[CandidateMembershipEvidence, ...]


@dataclass(frozen=True)
class CurrentUniverseCandidateManifest:
    """A frozen source-membership union awaiting independent quality admission."""

    created_at: datetime
    as_of_timestamp: datetime
    construction_rule: str
    data_validity_class: str
    sources: tuple[CandidateSource, ...]
    candidates: tuple[CurrentUniverseCandidate, ...]
    content_hash: str

    @property
    def members(self) -> tuple[str, ...]:
        """Return candidate symbols in manifest order."""
        return tuple(candidate.symbol for candidate in self.candidates)


@dataclass(frozen=True)
class CurrentUniverseBootstrap:
    """One provenance-bearing candidate set for an explicit full-data bootstrap."""

    source_manifest: CurrentUniverseCandidateManifest
    standard: ResearchWhitelistStandard

    @property
    def candidate_symbols(self) -> tuple[str, ...]:
        """Sorted current-index candidates, before historical-quality admission."""
        return self.source_manifest.members


def membership_fingerprint(manifest: CurrentUniverseCandidateManifest) -> str:
    """Hash stable candidate membership, not volatile source provenance.

    A fresh source observation always has new retrieval timestamps and often a
    new response hash. Those facts are retained in the audit document, but
    they must not make an unchanged whitelist look like a membership change.
    """
    return _canonical_hash(
        {
            "construction_rule": manifest.construction_rule,
            "data_validity_class": manifest.data_validity_class,
            "candidates": [
                {
                    "symbol": candidate.symbol,
                    "provider_symbol": candidate.provider_symbol,
                    "indices": sorted(
                        {membership.index for membership in candidate.source_memberships}
                    ),
                }
                for candidate in manifest.candidates
            ],
        }
    )


def candidate_manifest_document(manifest: CurrentUniverseCandidateManifest) -> dict[str, object]:
    """Return canonical JSON-shaped provenance suitable for the workspace authority."""
    return {
        "created_at": manifest.created_at.isoformat(),
        "as_of_timestamp": manifest.as_of_timestamp.isoformat(),
        "construction_rule": manifest.construction_rule,
        "data_validity_class": manifest.data_validity_class,
        "sources": [
            {**asdict(source), "retrieved_at": source.retrieved_at.isoformat()}
            for source in manifest.sources
        ],
        "candidates": [asdict(candidate) for candidate in manifest.candidates],
        "content_hash": manifest.content_hash,
    }


def bootstrap_from_candidate_manifest_document(
    document: dict[str, object],
) -> CurrentUniverseBootstrap:
    """Restore a persisted candidate manifest after a desktop restart.

    The document is treated as untrusted durable input: its content hash and
    immutable standard are checked before it can recreate an onboarding task.
    """
    try:
        content_hash = str(document["content_hash"])
        if len(content_hash) != 64:
            raise ValueError("candidate manifest hash has the wrong length")
        sources = tuple(
            CandidateSource(
                index=str(item["index"]),
                source_uri=str(item["source_uri"]),
                selector=str(item["selector"]),
                retrieved_at=datetime.fromisoformat(str(item["retrieved_at"])),
                response_hash=str(item["response_hash"]),
                license_class=str(item["license_class"]),
            )
            for item in document["sources"]  # type: ignore[index, union-attr]
        )
        candidates = tuple(
            CurrentUniverseCandidate(
                symbol=str(item["symbol"]),
                provider_symbol=str(item["provider_symbol"]),
                source_memberships=tuple(
                    CandidateMembershipEvidence(
                        index=str(membership["index"]),
                        company_name=str(membership["company_name"]),
                    )
                    for membership in item["source_memberships"]  # type: ignore[index, union-attr]
                ),
            )
            for item in document["candidates"]  # type: ignore[index, union-attr]
        )
        manifest = CurrentUniverseCandidateManifest(
            created_at=datetime.fromisoformat(str(document["created_at"])),
            as_of_timestamp=datetime.fromisoformat(str(document["as_of_timestamp"])),
            construction_rule=str(document["construction_rule"]),
            data_validity_class=str(document["data_validity_class"]),
            sources=sources,
            candidates=candidates,
            content_hash=content_hash,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("persisted candidate manifest is malformed") from exc
    if (
        manifest.construction_rule != ORIGINAL_RESEARCH_WHITELIST_STANDARD.construction_rule
        or manifest.data_validity_class != ORIGINAL_RESEARCH_WHITELIST_STANDARD.data_validity_class
        or manifest.candidates != tuple(sorted(manifest.candidates, key=lambda item: item.symbol))
    ):
        raise ValueError("persisted candidate manifest violates the original whitelist standard")
    expected_document = candidate_manifest_document(
        CurrentUniverseCandidateManifest(
            created_at=manifest.created_at,
            as_of_timestamp=manifest.as_of_timestamp,
            construction_rule=manifest.construction_rule,
            data_validity_class=manifest.data_validity_class,
            sources=manifest.sources,
            candidates=manifest.candidates,
            content_hash="",
        )
    )
    expected_document.pop("content_hash")
    if _canonical_hash(expected_document) != content_hash:
        raise ValueError("persisted candidate manifest content hash does not match")
    return CurrentUniverseBootstrap(
        source_manifest=manifest,
        standard=ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    )


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_candidate_manifest(
    fetched: dict[UniverseIndex, FetchedUniverseSource],
    *,
    timestamp: datetime,
) -> CurrentUniverseCandidateManifest:
    """Union source symbols without misusing volatile display names as identity.

    The production helper's strict company-name reconciliation is appropriate
    as a final identity guard, but current index pages use different short and
    legal labels for the same ticker. At candidate stage a ticker merely bounds
    data acquisition. Every source label remains evidence for later identity
    admission; it cannot create a durable listing identity.
    """
    memberships_by_symbol: dict[str, list[CandidateMembershipEvidence]] = {}
    sources: list[CandidateSource] = []
    for index in sorted(fetched, key=lambda item: item.value):
        response = fetched[index]
        config = FROZEN_UNIVERSE_SOURCES[index]
        if response.config != config:
            raise ValueError(f"{index.value} does not match the frozen source definition")
        sources.append(
            CandidateSource(
                index=index.value,
                source_uri=config.uri,
                selector=config.selector,
                retrieved_at=response.retrieved_at,
                response_hash=response.response_hash,
                license_class=config.license_class.value,
            )
        )
        for symbol, company_name in parse_membership(config, response.content):
            memberships_by_symbol.setdefault(symbol, []).append(
                CandidateMembershipEvidence(index=index.value, company_name=company_name)
            )
    candidates = tuple(
        CurrentUniverseCandidate(
            symbol=symbol,
            provider_symbol=provider_symbol(symbol),
            source_memberships=tuple(
                sorted(
                    memberships_by_symbol[symbol],
                    key=lambda item: (item.index, item.company_name),
                )
            ),
        )
        for symbol in sorted(memberships_by_symbol)
    )
    provisional = CurrentUniverseCandidateManifest(
        created_at=timestamp,
        as_of_timestamp=timestamp,
        construction_rule=ORIGINAL_RESEARCH_WHITELIST_STANDARD.construction_rule,
        data_validity_class=ORIGINAL_RESEARCH_WHITELIST_STANDARD.data_validity_class,
        sources=tuple(sources),
        candidates=candidates,
        content_hash="",
    )
    logical = candidate_manifest_document(provisional)
    logical.pop("content_hash")
    return CurrentUniverseCandidateManifest(
        created_at=provisional.created_at,
        as_of_timestamp=provisional.as_of_timestamp,
        construction_rule=provisional.construction_rule,
        data_validity_class=provisional.data_validity_class,
        sources=provisional.sources,
        candidates=provisional.candidates,
        content_hash=_canonical_hash(logical),
    )


def discover_current_universe_candidates(
    *,
    observed_at: datetime | None = None,
) -> CurrentUniverseBootstrap:
    """Fetch all three original sources and return their deterministic union.

    ``fetch_universe_source`` is bounded, validates the expected HTML table,
    and fails closed on source/schema errors. The caller must persist the
    resulting manifest before it is used as a workspace's bootstrap scope.
    """
    timestamp = observed_at or datetime.now(UTC)
    fetched: dict[UniverseIndex, FetchedUniverseSource] = {
        index: fetch_universe_source(config) for index, config in FROZEN_UNIVERSE_SOURCES.items()
    }
    return CurrentUniverseBootstrap(
        source_manifest=_build_candidate_manifest(fetched, timestamp=timestamp),
        standard=ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    )
