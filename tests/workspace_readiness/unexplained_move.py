"""A preparation that stops for one data decision: one listing's close doubles for one session."""

from __future__ import annotations

from typing import Any

from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from tests.researcher_methodology_surface.real_workspace import (
    AS_OF,
    HISTORY_START,
    OBSERVED_AT,
    SeededWalkProvider,
)


def unexplained_move(count: int = 120) -> tuple[SeededWalkProvider, tuple[str, ...]]:
    """A `count`-listing provider whose first listing's close doubles once, with no event for it."""
    symbols = tuple(f"F{i:03d}" for i in range(count))
    schedule = materialize_calendar_schedule(
        ("XNAS",), start=HISTORY_START, end=AS_OF, as_of_timestamp=OBSERVED_AT
    )
    sessions = tuple(value["session_date"] for value in schedule.to_pylist())
    moved = sessions[len(sessions) // 2]

    class UnexplainedMove(SeededWalkProvider):
        """One listing's close doubles for one session, with no event that explains it."""

        def _closes(self, symbol: str, end: Any) -> dict[Any, float]:
            closes = super()._closes(symbol, end)
            if symbol == symbols[0] and moved in closes:
                closes[moved] *= 2.0
            return closes

    return UnexplainedMove(symbols, sessions), symbols
