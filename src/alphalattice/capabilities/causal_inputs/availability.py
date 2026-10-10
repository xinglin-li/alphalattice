"""When a session's information is available: the one owner every reader of a named date uses.

A daily bar settles a while after the exchange close. Live use saw a mixed cohort for about half
an hour after the close and same-session revisions on the next pull, so the installed source
availability waits two hours. The date a person names, the Host's update automation and its held
dated update all read that instant here, so a declared strategy schedule
(`INSTALLED_SOURCE_AVAILABILITY_POLICY`) can set it later in one place.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final

SOURCE_READY_AFTER_CLOSE: Final = timedelta(hours=2)
"""The installed source-availability default: a session's data settles two hours after its close."""


def information_available_at(close: datetime) -> datetime:
    """When a session's information is available: its close plus the installed source lag.

    Args:
        close: The session's exchange close, timezone-aware.

    Returns:
        The instant its data is ready to plan on.

    Raises:
        ValueError: The close is not timezone-aware.
    """
    if close.tzinfo is None or close.utcoffset() is None:
        raise ValueError("workspace_readiness.session_close_timestamp_not_timezone_aware")
    return close + SOURCE_READY_AFTER_CLOSE
