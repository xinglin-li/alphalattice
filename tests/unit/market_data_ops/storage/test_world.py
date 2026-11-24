"""A 0.2.7 corporate action as its world row, and the lag rules world counts knowledge by."""

from __future__ import annotations

from typing import Any

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer.store import lag_rule
from alphalattice.foundation.market_data_ops.storage import world
from alphalattice.foundation.market_data_ops.storage.world import (
    admissions,
    world_action,
    world_listing,
)

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/world.py"
AAPL = "listing-aapl"
WRITTEN = "2024-02-10 00:00:00.000000+00"
ACTION = (AAPL, "yfinance", "2024-02-09", "CASH_DIVIDEND", None, 0.24, False, "provider", "ACTIVE")
ACTION += (WRITTEN,)
ROW = {"listing_id": world_listing(AAPL), "effective_date": "2024-02-09", "kind": "dividend"}
ROW |= {"action_id": "CASH_DIVIDEND", "ratio": None, "amount": 0.24, "currency": "USD"}
ROW |= {"provisional": False, "provenance": "provider", "retracted": False}
ROW |= {"source_observed_at": WRITTEN, "source_payload_digest": "original-payload-hash"}
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
            world_action(
                action, world_listing(AAPL), "USD", source_payload_digest="original-payload-hash"
            )
        assert refused.value.failure.code == expected
        return
    assert world_listing(AAPL).replace("-", "") == "32e379d73d7c53f7ad63982fd6dade63"  # pinned
    assert (
        world_action(
            action, world_listing(AAPL), "USD", source_payload_digest="original-payload-hash"
        )
        == expected
    )


@pytest.mark.proves("market_data_ops.bars_are_admitted_in_whole_years_under_a_bound")
@pytest.mark.parametrize(
    ("years", "expected"),
    [
        pytest.param([("yf", 2024, 4), ("yf", 2025, 5)], [("yf", 2024, 2025)], id="all in one"),
        pytest.param(
            [("yf", 2023, 6), ("yf", 2024, 4), ("yf", 2025, 5)],
            [("yf", 2023, 2024), ("yf", 2025, 2025)],
            id="over the bound",
        ),
        pytest.param([("yf", 2024, 12)], [("yf", 2024, 2024)], id="a year over it, alone"),
        pytest.param(
            [("av", 2024, 1), ("yf", 2024, 1)], [("av", 2024, 2024), ("yf", 2024, 2024)], id="two"
        ),
    ],
)
def test_bars_are_admitted_in_whole_years_while_they_fit_the_bound(years, expected):
    """A provider's consecutive years go in one admission while their bars fit the bound; a year
    over it goes alone, and each provider's bars are their own admissions."""
    assert admissions(years, bound=10) == expected


@pytest.mark.proves("market_data_ops.reference_rows_are_known_from_their_valid_date")
def test_reference_rows_are_known_from_their_valid_date_and_bars_from_their_close():
    """StorePort reads world's reference rule as counted from the valid date, the bars' from the
    close."""
    read = {rule: lag_rule(spec, *rule) for rule, (_text, spec) in world.LAG_RULES.items()}
    assert read == {world.MARKET_CLOSE: (0, "close"), world.MARKET_OBSERVED: (0, "valid")}
