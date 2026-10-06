"""What a study's sessions read of the Sector classification, as its records state it (V346).

Under the forward rule (`market_data_ops/sources/sector_forward.py`) a listing reads the
classification first recorded for it until its first reclassification, then each later one from
its effective session. A study none of whose sessions reads a reclassification reads one
classification, the current one, as every study did before the rule; its records say so, and
their identities stay what they were. One vocabulary for every record that states it: the Panel's
risk block and statement, the Alpha, Sector and Risk policies, the research foundation.
"""

from __future__ import annotations

from typing import Final, Literal

SECTOR_HISTORY_BACKFILLED: Final = "CURRENT_CLASSIFICATION_BACKFILLED"
"""One classification, the current one, read by every session: no reclassification in force."""

SECTOR_HISTORY_FORWARD: Final = "FIRST_RECORDED_CLASSIFICATION_BACKFILLED_THEN_AS_OBSERVED"
"""The backfill before each listing's first reclassification, then each as observed."""

type SectorHistoryTreatment = Literal[
    "CURRENT_CLASSIFICATION_BACKFILLED",
    "FIRST_RECORDED_CLASSIFICATION_BACKFILLED_THEN_AS_OBSERVED",
]
"""A record's statement of what its sessions read."""


def sector_treatment(*, reclassified: bool) -> SectorHistoryTreatment:
    """What a history's sessions read: the backfill alone, or the backfill then reclassifications.

    A history with no reclassification has every session read the current classification, as
    every study did before the forward rule, so its treatment and identities are theirs.

    Args:
        reclassified: Whether the history holds a reclassification.

    Returns:
        `SECTOR_HISTORY_FORWARD` or `SECTOR_HISTORY_BACKFILLED`.
    """
    return SECTOR_HISTORY_FORWARD if reclassified else SECTOR_HISTORY_BACKFILLED


__all__ = [
    "SECTOR_HISTORY_BACKFILLED",
    "SECTOR_HISTORY_FORWARD",
    "SectorHistoryTreatment",
    "sector_treatment",
]
