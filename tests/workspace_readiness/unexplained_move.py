"""A preparation that meets unexplained moves: listings whose close doubles for single sessions."""

from __future__ import annotations

from typing import Any

from alphalattice.foundation.market_data_ops.sources.universe import ResearchWhitelistStandard
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from tests._support import development_universe
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    OBSERVED_AT,
    SeededWalkProvider,
    history_start,
)


def unexplained_move(
    count: int = 120,
    listings: int = 1,
    sessions: int = 1,
    sector_size: int = 5,
    standard: ResearchWhitelistStandard = development_universe.STANDARD,
) -> tuple[SeededWalkProvider, tuple[str, ...]]:
    """A `count`-listing provider whose first `listings` listings each double their close on
    `sessions` sessions a week apart, with no event that explains them; a larger `sector_size`
    leaves room to set one aside above the five-name sector floor; its walk covers `standard`'s
    history."""
    symbols = tuple(f"F{i:03d}" for i in range(count))
    schedule = materialize_calendar_schedule(
        ("XNAS",),
        start=history_start(standard),
        end=AS_OF,
        as_of_timestamp=OBSERVED_AT,
    )
    calendar = tuple(value["session_date"] for value in schedule.to_pylist())
    moved = {calendar[len(calendar) // 2 + 5 * n] for n in range(sessions)}

    class UnexplainedMove(SeededWalkProvider):
        """The moved listings' closes double for their sessions, with no event for them."""

        def _closes(self, symbol: str, end: Any) -> dict[Any, float]:
            closes = super()._closes(symbol, end)
            if symbol in symbols[:listings]:
                for session in moved & closes.keys():
                    closes[session] *= 2.0
            return closes

    return UnexplainedMove(symbols, calendar, sector_size=sector_size), symbols
