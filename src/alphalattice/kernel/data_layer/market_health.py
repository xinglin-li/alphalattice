"""Advisory health counts for the world catalog's requested Daily session."""

from __future__ import annotations

from datetime import date
from typing import Literal, TypedDict

from alphalattice.kernel.data_layer.history_runtime import catalog_projection

CHECK_IDS = (
    "source_freshness",
    "not_null",
    "unique",
    "accepted_values",
    "relationships",
)
"""Stable identifiers in recorded Daily results."""


class MarketHealthCheck(TypedDict):
    """The stable identity, verdict and failure count of one Daily check."""

    id: str
    status: Literal["PASS", "FAIL", "UNAVAILABLE"]
    count: int | None


# Freshness checks the current raw-bar source against the completed market session, so
# weekends and holidays do not age a fresh source. The integrity checks use physical
# rows in that session. A new version of the same bar is a distinct legitimate row.
_COUNTS = """
WITH bars AS MATERIALIZED (
    SELECT provider_id, listing_id, session_date, version_id
    FROM lake.world.raw_bar WHERE session_date = ?
)
SELECT
    CASE WHEN EXISTS (SELECT 1 FROM raw_bar_latest WHERE session_date = ?)
        THEN 0 ELSE 1 END,
    (SELECT count(*) FROM bars
        WHERE provider_id IS NULL OR listing_id IS NULL OR version_id IS NULL),
    (SELECT count(*) FROM (
        SELECT 1 FROM bars
        WHERE provider_id IS NOT NULL AND listing_id IS NOT NULL AND version_id IS NOT NULL
        GROUP BY provider_id, listing_id, session_date, version_id HAVING count(*) > 1
    )),
    (SELECT count(*) FROM lake.world.corporate_action
        WHERE effective_date = ?
        AND kind NOT IN ('split', 'dividend', 'spinoff', 'delisting')),
    (SELECT count(*) FROM bars b WHERE b.listing_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM lake.world.listing l WHERE l.listing_id = b.listing_id
    ))
"""


def unavailable_market_health() -> list[MarketHealthCheck]:
    """Represent unavailable readings without reporting false zero counts."""
    return [{"id": name, "status": "UNAVAILABLE", "count": None} for name in CHECK_IDS]


def market_health(session: date) -> list[MarketHealthCheck]:
    """Read the five failure counts using the Host's held StorePort projection."""
    from duckdb import Error  # lazy: only the data extra has a catalog to check

    try:
        with catalog_projection("daily_market_health", _COUNTS, (session,) * 3) as result:
            rows = result.fetchall() if result is not None else ()
    except (Error, OSError):
        rows = ()
    if not rows:
        return unavailable_market_health()
    return [
        {"id": name, "status": "PASS" if count == 0 else "FAIL", "count": int(count)}
        for name, count in zip(CHECK_IDS, rows[0], strict=True)
    ]
