"""From which session a Sector reclassification is in force: the forward rule's clock.

Under DA12, the approximate point-in-time contract holds for the Sector as for the Universe:
the classification the workspace
holds when the rule starts stands in for every earlier session (a disclosed backfill, not point
in time), and each later reclassification of a listing takes effect from the session its update
observed it -- the trading day of the observation, the next session when that day has none -- and
never on a session a Panel already published. A refresh therefore changes the sessions from its
effective session on and none before it. The history it builds is `kernel/quant/sector_history`'s,
the treatment its records state `kernel/shared_kernel/sector_treatment`'s;
the Universe's own forward history, the template, is `membership.py`'s.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pandas as pd

from alphalattice.kernel.data.calendar import materialize_calendar_schedule

_EXCHANGE_ZONE = "America/New_York"


def sector_effective_session(observed_at: datetime, *, first_unpublished: date) -> date:
    """The session a reclassification observed at `observed_at` takes effect from.

    The exchange's trading day of the observation, the next session when that day is none, and
    never before the first session no Panel has published.

    Args:
        observed_at: When the refresh observed the classification (timezone-aware).
        first_unpublished: The first session after the last one a Panel published.

    Returns:
        The effective session.

    Raises:
        ValueError: `sector_history.effective_session_unavailable` for a naive clock or when the
            calendar names no session within two weeks.
    """
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("sector_history.effective_session_unavailable")
    observed_day = pd.Timestamp(observed_at).tz_convert(_EXCHANGE_ZONE).date()
    start = max(observed_day, first_unpublished)
    end = start + timedelta(days=14)
    schedule = materialize_calendar_schedule(
        ("XNYS", "XNAS"),
        start=start,
        end=end,
        as_of_timestamp=datetime.combine(end, time.max, tzinfo=UTC),
    )
    sessions: list[date] = sorted({row["session_date"] for row in schedule.to_pylist()})
    if not sessions:
        raise ValueError("sector_history.effective_session_unavailable")
    first: date = sessions[0]
    return first


__all__ = [
    "sector_effective_session",
]
