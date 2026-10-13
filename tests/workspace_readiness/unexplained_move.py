"""A preparation that meets unexplained moves: listings whose close doubles for single sessions."""

from __future__ import annotations

from typing import Any

from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    HISTORY_START,
    OBSERVED_AT,
    SeededWalkProvider,
)


def unexplained_move(
    count: int = 120, listings: int = 1, sessions: int = 1, sector_size: int = 5
) -> tuple[SeededWalkProvider, tuple[str, ...]]:
    """A `count`-listing provider whose first `listings` listings each double their close on
    `sessions` sessions a week apart, with no event that explains them; a larger `sector_size`
    leaves room to set one aside above the five-name sector floor."""
    symbols = tuple(f"F{i:03d}" for i in range(count))
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=AS_OF, as_of_timestamp=OBSERVED_AT
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
