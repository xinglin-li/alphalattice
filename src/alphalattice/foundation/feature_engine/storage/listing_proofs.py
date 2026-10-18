"""Prove each listing's selected Feature values over a prefix, its closed years apart.

The scope's proof (`FeatureStateRepository.feature_source_prefix_proofs`) answers closed
years by their seals, so any restatement moves it for every listing. A reader of the kept
score-input history compares these per-listing proofs to rebuild only the listings whose own
values moved.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import date
from typing import TYPE_CHECKING

import duckdb
import numpy as np

from alphalattice.kernel.shared_kernel.identity import canonical_hash

if TYPE_CHECKING:
    from alphalattice.foundation.feature_engine.storage.repositories import (
        FeatureStateRepository,
    )


def feature_source_listing_proofs(
    feature: FeatureStateRepository,
    *,
    listing_ids: Sequence[str],
    catalog_hash: str,
    start: date,
    ends: Sequence[date],
    factor_ids: Sequence[str],
    closed: Sequence[str] = (),
    _connection: duckdb.DuckDBPyConnection | None = None,
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...]]:
    """Each listing's selected Feature values from ``start`` to each cutoff, and its closed years.

    The proof (``FeatureSourceListingProofV2``) binds the listing, the cutoff, the selection,
    the digest of its rows before the cutoffs' calendar year (its closed years) and the
    sessions, values and null flags of its rows in that year, in the scope proof's own
    projection; a restatement of one listing moves its own proof only. As the scope proof
    answers closed years by their seals, a caller that proved them unchanged passes the closed
    digests it holds and only the open year is read. Values, not stored payload hashes, are the
    evidence, as the scope's proof holds.

    Args:
        feature: The workspace's Feature store.
        listing_ids: Nonempty unique listing scope, normalized to sorted order.
        catalog_hash: Exact installed computing catalog.
        start: First included session.
        ends: Nonempty requested cutoff sequence, in one calendar year when ``closed`` is given.
        factor_ids: Nonempty unique stored factor selection, normalized to sorted order.
        closed: The scope's closed-year digests, in its sorted order, proved unchanged.
        _connection: Optional connection already held in a consistent read snapshot.

    Returns:
        Each cutoff's proof of every listing in the requested cutoff order, and the closed-year
        digests in the scope's sorted order.

    Raises:
        ValueError: The request or selected catalog factors are invalid.
    """
    scope, factors, cutoffs = tuple(sorted(listing_ids)), tuple(sorted(factor_ids)), tuple(ends)
    if not scope or not factors or not cutoffs or any(start > end for end in cutoffs):
        raise ValueError("feature.source_prefix_request_invalid")
    if not set(factors) <= set(feature._feature_factor_ids(catalog_hash)):
        raise ValueError("feature row projection contains an unknown factor")
    opening = date(max(cutoffs).year, 1, 1)
    if closed and (len(closed) != len(scope) or min(cutoffs) < opening):
        raise ValueError("feature.source_prefix_request_invalid")
    boundary = (
        feature.database.read_transaction() if _connection is None else nullcontext(_connection)
    )
    names = [f"{kind}_{index}" for index in range(len(factors)) for kind in ("value", "null")]
    with boundary as connection:
        # The scope proof's own projection; only its values are read, ordered by listing.
        statement = feature._feature_prefix_statement(connection, factors, scoped=True)
        rows = connection.execute(
            f"SELECT listing_id, session_date, {', '.join(names)} FROM ({statement}) "
            "ORDER BY listing_id, session_date",
            [scope, catalog_hash, max(start, opening) if closed else start, max(cutoffs)],
        ).to_arrow_table()
    ids = rows.column("listing_id").to_numpy(zero_copy_only=False)
    days = rows.column("session_date").to_numpy(zero_copy_only=False).astype("datetime64[D]")
    values = [
        np.ascontiguousarray(rows.column(name).to_numpy(zero_copy_only=False)) for name in names
    ]
    keys = np.asarray(scope, dtype=object)
    bounds: list[int] = np.asarray(np.searchsorted(ids, keys, side="left"), np.int64).tolist()
    stops: list[int] = np.asarray(np.searchsorted(ids, keys, side="right"), np.int64).tolist()

    def digest(first: int, last: int) -> str:
        rows_hash = hashlib.sha256(days[first:last].astype(np.int64).tobytes())
        for column in values:
            rows_hash.update(column[first:last].tobytes())
        return rows_hash.hexdigest()

    def split(year: int) -> list[int]:
        first_day = np.datetime64(date(year, 1, 1), "D")
        return [
            first + int(np.searchsorted(days[first:stop], first_day))
            for first, stop in zip(bounds, stops, strict=True)
        ]

    # Each cutoff's closed years end at its own year; a year's digests are computed once.
    years = {end.year: split(end.year) for end in cutoffs}
    priors = {
        year: tuple(closed)
        or tuple(digest(first, cut) for first, cut in zip(bounds, cuts, strict=True))
        for year, cuts in years.items()
    }
    proofs: list[dict[str, str]] = []
    for end in cutoffs:
        bound, cuts = np.datetime64(end, "D"), years[end.year]
        proofs.append(
            {
                listing: str(
                    canonical_hash(
                        {
                            "kind": "FeatureSourceListingProofV2",
                            "listing_id": listing,
                            "catalog_hash": catalog_hash,
                            "start": start,
                            "end": end,
                            "factor_ids": factors,
                            "closed": prior,
                            "rows": digest(
                                cut, cut + int(np.searchsorted(days[cut:stop], bound, "right"))
                            ),
                        }
                    )
                )
                for listing, prior, cut, stop in zip(
                    scope, priors[end.year], cuts, stops, strict=True
                )
            }
        )
    return tuple(proofs), priors[max(cutoffs).year]
