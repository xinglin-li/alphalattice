"""A 0.2.7 corporate action as its world row."""

from __future__ import annotations

from typing import Any

import pytest

from alphalattice.foundation.market_data_ops.storage.world import world_action, world_listing
from alphalattice.kernel.data.errors import DataError

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/world.py"
AAPL = "listing-aapl"
WRITTEN = "2024-02-10 00:00:00.000000+00"
ACTION = (AAPL, "yfinance", "2024-02-09", "CASH_DIVIDEND", None, 0.24, False, "provider", "ACTIVE")
ACTION += (WRITTEN,)
ROW = {"listing_id": world_listing(AAPL), "effective_date": "2024-02-09", "kind": "dividend"}
ROW |= {"action_id": "CASH_DIVIDEND", "ratio": None, "amount": 0.24, "currency": "USD"}
ROW |= {"provisional": False, "provenance": "provider", "retracted": False}
ROW |= {"source_observed_at": WRITTEN}
CASES: dict[str, tuple[dict[int, Any], dict[str, Any] | str]] = {
    "dividend": ({}, ROW),
    "split": (
        {3: "SPLIT", 4: 4.0, 5: None},
        ROW
        | {"kind": "split", "action_id": "SPLIT", "ratio": 4.0, "amount": None}
        | {"currency": None},
    ),
    "capital gain": ({3: "CAPITAL_GAIN"}, ROW | {"action_id": "CAPITAL_GAIN"}),
    "retracted": ({8: "RETRACTED"}, ROW | {"retracted": True}),
    "unknown kind": ({3: "RIGHTS_ISSUE"}, "market_data.action_kind_unknown"),
}


@pytest.mark.proves("market_data_ops.an_action_is_one_world_row")
@pytest.mark.parametrize(("change", "expected"), list(CASES.values()), ids=list(CASES))
def test_a_0_2_7_action_is_one_world_row_or_refused_by_name(
    change: dict[int, Any], expected: dict[str, Any] | str
) -> None:
    """A 0.2.7 action keeps its kind as its id under its world listing, or is refused by name."""
    action = tuple(change.get(i, value) for i, value in enumerate(ACTION))
    if isinstance(expected, str):
        with pytest.raises(DataError) as refused:
            world_action(action, world_listing(AAPL), "USD")
        assert refused.value.failure.code == expected
        return
    assert world_listing(AAPL).replace("-", "") == "32e379d73d7c53f7ad63982fd6dade63"  # pinned
    assert world_action(action, world_listing(AAPL), "USD") == expected
