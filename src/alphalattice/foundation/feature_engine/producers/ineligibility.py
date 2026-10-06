"""Bounded physical-run access for feature eligibility diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from alphalattice.foundation.feature_engine.storage.contracts import FeatureIneligibilityRun
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository


@dataclass(frozen=True)
class FeatureIneligibilityStore:
    """Internal engines read compact intervals, never the expansion view."""

    store: FeatureStateRepository

    def find_runs(
        self,
        *,
        listing_ids: Sequence[str],
        factor_ids: Sequence[str],
        start: date,
        end: date,
    ) -> tuple[FeatureIneligibilityRun, ...]:
        """Read bounded ineligibility intervals for the requested listings and factors."""
        return self.store.find_feature_ineligibility_runs(
            listing_ids=listing_ids,
            factor_ids=factor_ids,
            start=start,
            end=end,
        )
