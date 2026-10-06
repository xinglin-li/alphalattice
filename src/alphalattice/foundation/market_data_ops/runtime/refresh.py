"""Bounded deterministic refresh planning for the local-first provider adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

REFRESH_OVERLAP_DAYS = 45
STANDARD_INCREMENTAL_GAP_DAYS = 120
MAXIMUM_AUTOMATIC_CATCH_UP_GAP_DAYS = 366
MAXIMUM_PROVIDER_WINDOW_DAYS = 184


class RefreshGapDisposition(StrEnum):
    """Planning outcome for the requested refresh gap."""

    INITIAL_HISTORY_REQUIRED = "INITIAL_HISTORY_REQUIRED"
    UP_TO_DATE = "UP_TO_DATE"
    NORMAL_INCREMENT = "NORMAL_INCREMENT"
    EXTENDED_CATCH_UP = "EXTENDED_CATCH_UP"
    EXPLICIT_APPROVAL_REQUIRED = "EXPLICIT_APPROVAL_REQUIRED"


@dataclass(frozen=True)
class RefreshWindow:
    """Inclusive provider window within a bounded refresh plan."""

    start: date
    end: date

    def __post_init__(self) -> None:
        """Reject a window whose start follows its end."""
        if self.start > self.end:
            raise ValueError("refresh window start is after end")


@dataclass(frozen=True)
class NormalRefreshPlan:
    """Local coverage, gap classification, and admitted provider windows."""

    latest_local_session: date | None
    requested_as_of: date
    missing_calendar_days: int | None
    overlap_days: int
    disposition: RefreshGapDisposition
    windows: tuple[RefreshWindow, ...]

    @property
    def fetch_start(self) -> date | None:
        """Return the first provider date, or ``None`` when no window is admitted."""
        return self.windows[0].start if self.windows else None

    @property
    def requires_explicit_approval(self) -> bool:
        """Report whether the gap exceeds automatic catch-up admission."""
        return self.disposition is RefreshGapDisposition.EXPLICIT_APPROVAL_REQUIRED


def normal_refresh_plan(
    latest_session: date | None,
    requested_as_of: date,
) -> NormalRefreshPlan:
    """Plan a complete missing interval plus a bounded correction overlap.

    The 45 days are an overlap before the last local session, not the maximum
    request size. A three-month absence therefore remains one ordinary
    incremental catch-up. Longer admitted ranges are split into resumable
    provider windows; a gap beyond one year requires a new user admission.
    """
    if latest_session is None:
        return NormalRefreshPlan(
            latest_local_session=None,
            requested_as_of=requested_as_of,
            missing_calendar_days=None,
            overlap_days=REFRESH_OVERLAP_DAYS,
            disposition=RefreshGapDisposition.INITIAL_HISTORY_REQUIRED,
            windows=(),
        )
    missing_days = max(0, (requested_as_of - latest_session).days)
    if missing_days > MAXIMUM_AUTOMATIC_CATCH_UP_GAP_DAYS:
        return NormalRefreshPlan(
            latest_local_session=latest_session,
            requested_as_of=requested_as_of,
            missing_calendar_days=missing_days,
            overlap_days=REFRESH_OVERLAP_DAYS,
            disposition=RefreshGapDisposition.EXPLICIT_APPROVAL_REQUIRED,
            windows=(),
        )
    if missing_days == 0:
        disposition = RefreshGapDisposition.UP_TO_DATE
    elif missing_days <= STANDARD_INCREMENTAL_GAP_DAYS:
        disposition = RefreshGapDisposition.NORMAL_INCREMENT
    else:
        disposition = RefreshGapDisposition.EXTENDED_CATCH_UP
    start = min(latest_session, requested_as_of) - timedelta(days=REFRESH_OVERLAP_DAYS)
    windows: list[RefreshWindow] = []
    cursor = start
    while cursor <= requested_as_of:
        window_end = min(
            requested_as_of,
            cursor + timedelta(days=MAXIMUM_PROVIDER_WINDOW_DAYS - 1),
        )
        windows.append(RefreshWindow(cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return NormalRefreshPlan(
        latest_local_session=latest_session,
        requested_as_of=requested_as_of,
        missing_calendar_days=missing_days,
        overlap_days=REFRESH_OVERLAP_DAYS,
        disposition=disposition,
        windows=tuple(windows),
    )


def normal_refresh_start(latest_session: date | None, requested_as_of: date) -> date | None:
    """Return the inclusive normal-fetch start, retaining a 45-day overlap.

    ``None`` means that no local coverage exists and the host must choose an
    explicit initial-history request.  This planner never performs a provider
    fetch and should not be used as evidence that corporate actions are fresh.
    """
    return normal_refresh_plan(latest_session, requested_as_of).fetch_start
