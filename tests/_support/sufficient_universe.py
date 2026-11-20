"""The sufficient test data set: the development generator scaled to the product's floors.

The development universe (development_universe.py) holds every case the full data holds, by name, at
the smallest size development needs. A test whose requirement is a product floor needs that floor's
scale instead, and nothing more: this set is sized by the floors, never by a real universe.

- Listings: the seventeen named development cases plus 91 generated members over eleven Sectors and
  both indexes, so 104 are eligible. The Factor executor's floor is 100 eligible listings
  (`minimum_factor_listings`), and calibration's is 80 members.
- History: eight calendar years before `AS_OF`. The daily threshold profile asks seven calendar
  years of Factor observations (schedule_contracts.py: 756 training, three 252-session folds, a
  252-session holdout), and a Factor's first observation waits for the catalog's longest lookback
  (252 sessions, 253 observations). At seven years the walk cannot fill its folds; the model
  lifecycle's 1,260-session window fits inside.

It is generated and synthetic, never a cut of real data. The nightly builds it once into the
golden workspace cache (`ALPHALATTICE_TEST_GOLDEN_ROOT`), whose key binds these listings and this
standard, and every scale test copies it from there.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date, datetime

from alphalattice.foundation.market_data_ops.sources.universe import (
    ORIGINAL_RESEARCH_WHITELIST_STANDARD,
    CurrentUniverseBootstrap,
)
from tests._support import development_universe as development
from tests._support.development_universe import OBSERVED_AT, THROUGH, DevelopmentProvider, Listing

SECTORS = (
    "Technology",
    "Healthcare",
    "Industrials",
    "Financials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Materials",
    "Utilities",
    "Real Estate",
    "Communication Services",
)
GENERATED = 91
"""Members beyond the named cases: every fourth pays dividends and every seventeenth splits, so a
Factor's adjusted history meets them across the cross-section, not on one name."""
STANDARD = replace(ORIGINAL_RESEARCH_WHITELIST_STANDARD, minimum_history_calendar_years=8)
"""The original standard's Factor floor (100 listings) over the eight years a walk needs."""
ROW_CAP = 250_000
"""About 2,040 sessions for 109 series, the reference included: 222,000 bars."""


def _generated(index: int) -> Listing:
    cases = (*(("dividends",) if index % 4 == 1 else ()), *(("split",) if index % 17 == 5 else ()))
    return Listing(
        symbol=f"S{chr(65 + index // 26)}{chr(65 + index % 26)}",
        index="NASDAQ100" if index % 2 else "SP500",
        sector=SECTORS[index % len(SECTORS)],
        cases=cases,
        price=20.0 + (index * 37) % 280,
    )


LISTINGS: tuple[Listing, ...] = (*development.LISTINGS, *map(_generated, range(GENERATED)))
SYMBOLS = tuple(item.symbol for item in LISTINGS)


def calendar_sessions(end: date = THROUGH) -> tuple[date, ...]:
    """The US sessions from eight years before `AS_OF` through `end`."""
    return development.calendar_sessions(end=end, standard=STANDARD)


def bootstrap(observed_at: datetime = OBSERVED_AT) -> CurrentUniverseBootstrap:
    """The candidate set the universe source answers at `observed_at`, under this standard."""
    return development.bootstrap(observed_at, listings=LISTINGS, standard=STANDARD)


def source_loader(**kwargs: object) -> CurrentUniverseBootstrap:
    """The universe source: the candidate set as observed at the Host's `observed_at`."""
    observed_at = kwargs.get("observed_at")
    return bootstrap(observed_at if isinstance(observed_at, datetime) else OBSERVED_AT)


def sufficient_provider(
    *,
    clock: Callable[[], datetime] = lambda: OBSERVED_AT,
    moved: Sequence[str] = (),
    moved_sessions: int = 1,
    through: date = THROUGH,
) -> DevelopmentProvider:
    """The sufficient set's provider: the development provider over these listings and years,
    served through `through` (a dated journey's own last session)."""
    return DevelopmentProvider(
        clock=clock,
        moved=moved,
        moved_sessions=moved_sessions,
        sessions=calendar_sessions(through),
        listings=LISTINGS,
        row_cap=ROW_CAP,
    )
