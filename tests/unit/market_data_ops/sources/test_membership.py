"""Membership transitions as events, judged without a store."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from alphalattice.foundation.market_data_ops.sources.membership import (
    membership_events_for_transition,
)

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/sources/membership.py"


@pytest.mark.proves("market_data_ops.a_membership_transition_is_its_exits_then_entries")
def test_transition_events_are_the_exits_then_entries_between_two_sets() -> None:
    events = membership_events_for_transition(
        market_profile_id="p",
        current_listing_ids=("A", "B", "C"),
        next_listing_ids=("A", "C", "D", "E"),
        effective_session=date(2026, 1, 5),
        observed_at=datetime(2026, 1, 8, 22, tzinfo=UTC),
        decided_at=datetime(2026, 1, 8, 22, tzinfo=UTC),
        authority="feature_input_gateway",
        reference_hash="7" * 64,
        manifest_revision="8" * 64,
        first_sequence=4,
    )
    assert [(item.sequence, item.listing_id, item.kind) for item in events] == [
        (4, "B", "EXIT"),
        (5, "D", "ENTRY"),
        (6, "E", "ENTRY"),
    ]
