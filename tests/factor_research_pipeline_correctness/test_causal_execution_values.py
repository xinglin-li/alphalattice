"""Inference values keep the full causal row's admissions and exact economics."""

from datetime import UTC, date, datetime
from struct import pack

import pytest

from alphalattice.foundation.causal_outcomes.execution.compile import (
    causal_execution_simple_return,
    derive_causal_execution_row,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionSchedulePoint,
)
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.kernel.data.enums import ExecutionVenueStatus


def _bar(price: float, volume: int) -> RawDailyBar:
    return RawDailyBar(
        "listing-a", "synthetic", date(2026, 1, 5), price, price, price, price, volume
    )


@pytest.mark.parametrize("dividend", [0.0, 2.0])
@pytest.mark.parametrize(
    "entry,holding,entry_venue,holding_venue,eligible",
    [
        (_bar(100.0, 10), _bar(110.0, 10), None, None, True),
        (
            _bar(100.0, 10),
            _bar(110.0, 10),
            ExecutionVenueStatus.VERIFIED_ELIGIBLE,
            ExecutionVenueStatus.VERIFIED_ELIGIBLE,
            True,
        ),
        (None, _bar(110.0, 10), None, None, False),
        (_bar(100.0, 10), None, None, None, False),
        (_bar(100.0, 0), _bar(110.0, 10), None, None, False),
        (_bar(100.0, 10), _bar(110.0, 0), None, None, False),
        (_bar(0.0, 10), _bar(110.0, 10), None, None, False),
        (_bar(100.0, 10), _bar(float("nan"), 10), None, None, False),
        (
            _bar(100.0, 10),
            _bar(110.0, 10),
            ExecutionVenueStatus.HALTED_OR_MARKET_RESTRICTED,
            None,
            False,
        ),
        (
            _bar(100.0, 10),
            _bar(110.0, 10),
            None,
            ExecutionVenueStatus.HALTED_OR_MARKET_RESTRICTED,
            False,
        ),
    ],
)
def test_inference_return_keeps_the_evidence_rows_exact_value_and_eligibility(
    entry, holding, entry_venue, holding_venue, eligible, dividend
):
    point = CausalExecutionSchedulePoint(
        sequence=1,
        formation_session=date(2026, 1, 2),
        formation_close_at=datetime(2026, 1, 2, 21, tzinfo=UTC),
        entry_session=date(2026, 1, 5),
        entry_open_at=datetime(2026, 1, 5, 14, 30, tzinfo=UTC),
        holding_end_session=date(2026, 1, 6),
        holding_end_open_at=datetime(2026, 1, 6, 14, 30, tzinfo=UTC),
        actual_session_span=2,
    )
    inputs = dict(
        entry_bar=entry,
        holding_bar=holding,
        period_dividend_split_adjusted=dividend,
        entry_venue_status=entry_venue,
        holding_venue_status=holding_venue,
    )
    value = causal_execution_simple_return(**inputs)
    row = derive_causal_execution_row(listing_id="listing-a", symbol="A", point=point, **inputs)
    assert value == row["simple_return"]
    if eligible:
        expected = 0.10000000000000009 if dividend == 0.0 else 0.1200000000000001
        assert pack("!d", value) == pack("!d", expected)
    else:
        assert value is None
    assert len(row["row_hash"]) == len(row["entry_source_row_hash"]) == 64
    assert len(row["holding_end_source_row_hash"]) == 64


@pytest.mark.parametrize("dividend", [-1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("bar", [None, _bar(100.0, 0), _bar(100.0, 10)])
def test_invalid_dividends_are_refused_even_when_the_bar_cannot_execute(bar, dividend):
    with pytest.raises(ValueError, match=r"^causal execution dividend is invalid$"):
        causal_execution_simple_return(
            entry_bar=bar, holding_bar=bar, period_dividend_split_adjusted=dividend
        )
