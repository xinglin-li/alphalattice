"""Model-eligible history boundary derived from authoritative Panel availability."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

from alphalattice.foundation.feature_engine.contracts import canonical_hash


@dataclass(frozen=True)
class FeaturePanelHistoryEligibility:
    """Record the first fully available session and its warmup history."""

    panel_history_start: date
    model_eligible_history_start: date
    warmup_session_count: int
    observed_session_count: int
    factor_count: int
    eligibility_hash: str

    def safe_summary(self) -> dict[str, object]:
        """Return the eligibility dates and counts for a public summary."""
        return {
            "panel_history_start": self.panel_history_start.isoformat(),
            "model_eligible_history_start": self.model_eligible_history_start.isoformat(),
            "warmup_session_count": self.warmup_session_count,
            "observed_session_count": self.observed_session_count,
            "factor_count": self.factor_count,
            "eligibility_hash": self.eligibility_hash,
        }


def derive_feature_panel_history_eligibility(
    availability: Sequence[Mapping[str, object]],
    *,
    factor_ids: tuple[str, ...],
) -> FeaturePanelHistoryEligibility:
    """Return the first session with a usable value surface for every factor."""
    expected = set(factor_ids)
    if not expected or len(expected) != len(factor_ids):
        raise ValueError("feature panel eligibility factor axis is invalid")
    by_session: dict[date, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for item in availability:
        raw_session = item.get("session_date")
        session = (
            raw_session if isinstance(raw_session, date) else date.fromisoformat(str(raw_session))
        )
        factor_id = str(item.get("factor_id"))
        if factor_id not in expected:
            continue
        if factor_id in by_session[session]:
            raise ValueError("feature panel eligibility contains duplicate availability")
        by_session[session][factor_id] = item
    sessions = tuple(sorted(by_session))
    if not sessions:
        raise ValueError("feature panel eligibility has no admitted sessions")
    eligible = tuple(
        session
        for session in sessions
        if set(by_session[session]) == expected
        and all(
            item.get("status") == "available"
            and float(cast(float | int | str, item.get("coverage", 0.0))) > 0.0
            for item in by_session[session].values()
        )
    )
    if not eligible:
        raise ValueError("feature panel has no model-eligible complete-factor session")
    first = eligible[0]
    panel_history_start = sessions[0]
    warmup_session_count = sum(session < first for session in sessions)
    observed_session_count = len(sessions)
    factor_count = len(factor_ids)
    values = {
        "panel_history_start": panel_history_start,
        "model_eligible_history_start": first,
        "warmup_session_count": warmup_session_count,
        "observed_session_count": observed_session_count,
        "factor_count": factor_count,
    }
    return FeaturePanelHistoryEligibility(
        panel_history_start=panel_history_start,
        model_eligible_history_start=first,
        warmup_session_count=warmup_session_count,
        observed_session_count=observed_session_count,
        factor_count=factor_count,
        eligibility_hash=canonical_hash(values),
    )


__all__ = [
    "FeaturePanelHistoryEligibility",
    "derive_feature_panel_history_eligibility",
]
