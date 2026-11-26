"""Prove a market source prefix for a scope and for each listing in it, from one scan.

The workspace store's raw bars and cutoff-admitted actions are the evidence; a reader of the
kept score-input history compares the scope's proof and rebuilds only the listings whose own
proof moved (`market_data_source_prefix_proofs`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from datetime import date
from functools import cache
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pyarrow as pa

from alphalattice.foundation.market_data_ops.storage.duckdb import (
    _SELECTED_PROVIDER_SQL,
    _canonical_hash,
    _source_prefix_arrow_hash,
)

if TYPE_CHECKING:
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository


def _listing_arrow_hashes(table: pa.Table) -> dict[str, str]:
    """Each listing's rows of a table ordered by listing: every column's values, hashed.

    Fixed-width columns hash their bytes and text columns their unit-separated reprs, one
    conversion a column; one listing's rows alone decide its digest.
    """
    ids = table.column("listing_id").to_numpy(zero_copy_only=False)
    columns = [table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names]
    keys = np.unique(ids)
    bounds: list[int] = np.asarray(np.searchsorted(ids, keys, side="left"), np.int64).tolist()
    stops: list[int] = np.asarray(np.searchsorted(ids, keys, side="right"), np.int64).tolist()
    hashes: dict[str, str] = {}
    for listing, first, stop in zip(map(str, keys), bounds, stops, strict=True):
        digest = hashlib.sha256(table.schema.to_string().encode())
        for column in columns:
            part = column[first:stop]
            digest.update(
                "\x1f".join(map(repr, part)).encode() if part.dtype == object else part.tobytes()
            )
        hashes[listing] = digest.hexdigest()
    return hashes


def market_data_source_prefix_proofs(
    market: MarketDataRepository,
    *,
    listing_ids: Sequence[str],
    start: date,
    through: date,
    _connection: duckdb.DuckDBPyConnection | None = None,
) -> tuple[str, Callable[[], dict[str, str]]]:
    """Prove exact current raw values and cutoff-admitted actions, for the scope and each listing.

    Raw rows have the same listing/date selection as ``raw_bars``, including every stored
    provider. Actions use ``actions``' latest provider mapping, ACTIVE status and effective
    dates through the cutoff, with no lower action-date bound. Stored payload hashes and
    revision journals are not evidence that these current values still equal a previous
    prefix. One listing's correction moves its own proof and the scope's, never another's.

    Args:
        market: The workspace store.
        listing_ids: Nonempty unique listing scope, normalized to sorted order.
        start: First included raw session.
        through: Last included raw session and admitted action effective date.
        _connection: Optional connection already held in a consistent read snapshot.

    Returns:
        SHA256 binding request scope, actual keys, values, nulls and selected providers, and
        each listing's own proof of its provider, raw rows and actions, computed once when first
        asked for (a reader whose whole proof holds never needs them).

    Raises:
        ValueError: The request is invalid or a listing has no provider mapping.
    """
    scope = tuple(listing_ids)
    if (
        not scope
        or any(not isinstance(value, str) or not value for value in scope)
        or len(set(scope)) != len(scope)
        or start > through
    ):
        raise ValueError("market_data_ops.source_prefix_request_invalid")
    scope = tuple(sorted(scope))
    providers_sql = _SELECTED_PROVIDER_SQL
    boundary = (
        market.database.read_transaction() if _connection is None else nullcontext(_connection)
    )
    with boundary as connection:
        providers = connection.execute(
            providers_sql
            + "SELECT listing_id, provider FROM selected_provider ORDER BY listing_id",
            [scope],
        ).to_arrow_table()
        if providers.column("provider").null_count:
            raise ValueError("listing has no provider mapping")
        raw = connection.execute(
            """
            SELECT listing_id, provider, session_date,
                   COALESCE(open, 0::DOUBLE) AS open,
                   COALESCE(high, 0::DOUBLE) AS high,
                   COALESCE(low, 0::DOUBLE) AS low,
                   COALESCE(close, 0::DOUBLE) AS close,
                   COALESCE(volume, 0::BIGINT) AS volume,
                   open IS NULL AS open_is_null, high IS NULL AS high_is_null,
                   low IS NULL AS low_is_null, close IS NULL AS close_is_null,
                   volume IS NULL AS volume_is_null
            FROM raw_daily_bar_current
            WHERE listing_id IN (SELECT unnest(?))
              AND session_date BETWEEN ? AND ?
            ORDER BY listing_id, session_date, provider
            """,
            [scope, start, through],
        ).to_arrow_table()
        actions = connection.execute(
            providers_sql
            + """
            SELECT action.listing_id, action.provider, action.effective_date,
                   action.action_kind,
                   COALESCE(action.new_shares_per_old_share, 0::DOUBLE)
                       AS new_shares_per_old_share,
                   COALESCE(action.cash_amount, 0::DOUBLE) AS cash_amount,
                   action.provisional, action.provenance,
                   action.new_shares_per_old_share IS NULL AS shares_is_null,
                   action.cash_amount IS NULL AS cash_is_null
            FROM corporate_action_current AS action
            JOIN selected_provider AS selected
              ON selected.listing_id = action.listing_id
             AND selected.provider = action.provider
            WHERE action.status = 'ACTIVE' AND action.effective_date <= ?
            ORDER BY action.listing_id, action.effective_date, action.action_kind
            """,
            [scope, through],
        ).to_arrow_table()
    whole = _canonical_hash(
        {
            "kind": "MarketSourcePrefixProofV1",
            "listing_ids": scope,
            "start": start,
            "through": through,
            "providers": _source_prefix_arrow_hash(providers),
            "raw": _source_prefix_arrow_hash(raw),
            "actions": _source_prefix_arrow_hash(actions),
        }
    )

    @cache
    def listings() -> dict[str, str]:
        parts = [_listing_arrow_hashes(table) for table in (providers, raw, actions)]
        return {
            listing: _canonical_hash(
                {
                    "kind": "MarketSourceListingProofV1",
                    "listing_id": listing,
                    "start": start,
                    "through": through,
                    "parts": [part.get(listing) for part in parts],
                }
            )
            for listing in scope
        }

    return whole, listings
