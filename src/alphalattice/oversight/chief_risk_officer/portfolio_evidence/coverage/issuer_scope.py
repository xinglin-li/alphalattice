"""Compile the one Portfolio-to-Evidence binding for a single sealed book.

A deterministic compiler, not a Task: given the same exposure projection and the
same registry it returns the same scope, which is what lets a reviewer check the
selection by hand instead of trusting a score.

Every mapped issuer is in scope. The scope used to stop at eight -- the
Alternative Evidence request axis -- and report the rest of the book as
"outside the review budget"; that made an execution bound into a coverage
bound, and a fifty-name book was reviewed as eight. Eight is now the size of
one execution unit (`runtime/coverage.py`), and priority decides the order
units are prepared in, never whether a held issuer is prepared.
"""

from __future__ import annotations

import math
from collections import defaultdict

from alphalattice.evidence.alternative_evidence.contracts import SecIssuerRegistrySnapshot
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    POSITION_EPSILON,
    IssuerSelectionReason,
    PortfolioExposureProjection,
    PortfolioIssuerMappingFailure,
    PortfolioIssuerPosition,
    PortfolioIssuerScope,
    exposure_band,
    seal_portfolio_evidence_contract,
    transition_of,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.coverage.issuer_mapping import (
    ordered_mapping_failures,
    resolve_portfolio_issuer_mapping,
)


def compile_portfolio_issuer_scope(
    *,
    projection: PortfolioExposureProjection,
    registry: SecIssuerRegistrySnapshot,
) -> PortfolioIssuerScope:
    """Map the book onto admitted issuers and rank every one of them for review.

    The ranking is lexicographic:

    1. issuers whose aggregate position grew, by descending positive change;
    2. remaining unchanged holdings, by descending ending weight;
    3. issuers that shrank or left, by descending absolute change;
    4. issuer id, ascending, to break every tie.

    Size and change decide *priority* here and the exposure band on each
    selected issuer; they never decide how severe a finding is, and they
    never leave a mapped issuer out.
    """
    mapping = resolve_portfolio_issuer_mapping(
        registry=registry,
        listings=tuple((value.listing_id, value.ticker) for value in projection.positions),
    )

    ending: dict[str, float] = defaultdict(float)
    preceding: dict[str, float] = defaultdict(float)
    tickers: dict[str, set[str]] = defaultdict(set)
    listings: dict[str, set[str]] = defaultdict(set)
    entity_by_listing: dict[str, str] = {}
    weighted_failures: list[PortfolioIssuerMappingFailure] = []
    for position, resolution in zip(projection.positions, mapping.resolutions, strict=True):
        entity_id = resolution.entity_id
        if entity_id is None:
            failure = resolution.failure
            assert failure is not None
            weighted_failures.append(
                failure.model_copy(
                    update={
                        "ending_weight": position.ending_weight,
                        "absolute_change": abs(position.signed_change),
                    }
                )
            )
            continue
        ending[entity_id] += position.ending_weight
        preceding[entity_id] += position.preceding_weight
        if position.ticker is not None:
            tickers[entity_id].add(position.ticker)
        listings[entity_id].add(position.listing_id)
        entity_by_listing[position.listing_id] = entity_id

    # Every coverage ratio is a correctly rounded sum over the same raw listing
    # weights, so a book that is fully mapped or fully reviewed reads exactly 1.0
    # rather than an ulp below it. Per-issuer aggregates are for the rows only.
    mapped = tuple(value for value in projection.positions if value.listing_id in entity_by_listing)
    unmapped = tuple(
        value for value in projection.positions if value.listing_id not in entity_by_listing
    )
    book_ending = math.fsum(value.ending_weight for value in projection.positions)
    book_absolute_change = math.fsum(abs(value.signed_change) for value in projection.positions)
    mapped_ending = math.fsum(value.ending_weight for value in mapped)
    unmapped_ending = math.fsum(value.ending_weight for value in unmapped)
    unmapped_change = math.fsum(abs(value.signed_change) for value in unmapped)

    # Weight rank over every mapped issuer, largest ending weight first, ties by id.
    by_weight = sorted(ending, key=lambda value: (-ending[value], value))
    weight_rank = {entity_id: rank for rank, entity_id in enumerate(by_weight, start=1)}

    ranked: list[tuple[int, float, str, IssuerSelectionReason]] = []
    for entity_id in ending:
        change = ending[entity_id] - preceding[entity_id]
        if change > POSITION_EPSILON:
            ranked.append((0, -change, entity_id, "OPENED_OR_INCREASED_BY_POSITIVE_CHANGE"))
        elif change < -POSITION_EPSILON:
            ranked.append((2, -abs(change), entity_id, "REDUCED_OR_EXITED_BY_ABSOLUTE_CHANGE"))
        else:
            ranked.append((1, -ending[entity_id], entity_id, "HELD_BY_ENDING_WEIGHT"))
    ranked.sort(key=lambda value: (value[0], value[1], value[2]))

    selected = tuple(
        PortfolioIssuerPosition(
            entity_id=entity_id,
            cik=mapping.cik_by_entity[entity_id],
            tickers=tuple(sorted(tickers[entity_id])),
            listing_ids=tuple(sorted(listings[entity_id])),
            ending_weight=ending[entity_id],
            preceding_weight=preceding[entity_id],
            signed_change=ending[entity_id] - preceding[entity_id],
            transition=transition_of(ending=ending[entity_id], preceding=preceding[entity_id]),
            weight_rank=weight_rank[entity_id],
            exposure_band=exposure_band(
                ending_weight=ending[entity_id],
                signed_change=ending[entity_id] - preceding[entity_id],
                weight_rank=weight_rank[entity_id],
            ),
            selection_reason=reason,
            selection_rank=rank,
        )
        for rank, (_tier, _key, entity_id, reason) in enumerate(ranked, start=1)
    )

    selected_ids = {value.entity_id for value in selected}
    reviewed = tuple(
        value for value in mapped if entity_by_listing[value.listing_id] in selected_ids
    )
    reviewed_ending = math.fsum(value.ending_weight for value in reviewed)
    reviewed_change = math.fsum(abs(value.signed_change) for value in reviewed)
    unavailable: list[str] = []
    if weighted_failures:
        still_held = sum(1 for value in weighted_failures if value.ending_weight > 0.0)
        exited = len(weighted_failures) - still_held
        # Named only where the admitted authority actually carries a symbol.
        # The rest are listing ids, and they belong in the structured failures a
        # reader can expand, not in a sentence.
        named = sorted({value.ticker for value in weighted_failures if value.ticker})
        parts = [
            f"{len(weighted_failures)} position(s) worth {unmapped_ending:.4f} of the book "
            f"did not map to an admitted issuer: {still_held} still held, {exited} exited"
        ]
        if named:
            shown = ", ".join(named[:8])
            more = "" if len(named) <= 8 else f", and {len(named) - 8} more"
            parts.append(f"; with an admitted symbol: {shown}{more}")
        unavailable.append("".join(parts))
    if not selected:
        unavailable.append("no listing in this book mapped to an admitted issuer")

    return seal_portfolio_evidence_contract(
        PortfolioIssuerScope,
        "scope_hash",
        book_authority=projection.book_authority,
        exposure_projection_hash=projection.projection_hash,
        registry_hash=registry.registry_hash,
        selected_issuers=selected,
        mapping_failures=ordered_mapping_failures(tuple(weighted_failures)),
        reviewed_ending_weight_coverage=_ratio(reviewed_ending, book_ending),
        reviewed_absolute_change_coverage=_ratio(reviewed_change, book_absolute_change),
        mapping_coverage=_ratio(mapped_ending, book_ending),
        selected_issuer_coverage=_ratio(float(len(selected)), float(len(ranked))),
        mapped_issuer_count=len(ranked),
        unmapped_ending_weight=min(1.0, unmapped_ending),
        unmapped_absolute_change=min(1.0, unmapped_change),
        unavailable_reasons=tuple(unavailable),
    )


def _ratio(part: float, whole: float) -> float:
    """Coverage over an empty book is zero, not one."""

    if whole <= POSITION_EPSILON:
        return 0.0
    return min(1.0, part / whole)


__all__ = ["compile_portfolio_issuer_scope"]
