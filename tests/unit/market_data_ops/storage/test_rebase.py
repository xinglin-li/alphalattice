"""A split-adjusting provider's resend of a listing's bars, recognized as rebases."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.storage.rebase import recognize

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/storage/rebase.py"
SESSIONS = tuple(date(2026, 5, 1) + timedelta(n) for n in range(10))
NOW = datetime(2026, 8, 3, 22, tzinfo=UTC)


def _bar(day: date, scale: float = 1.0) -> RawDailyBar:
    price = (100.0 + day.day / 7.0) * scale
    return RawDailyBar("L", "p", day, price, price * 1.01, price * 0.99, price, round(1e6 / scale))


def _split(day: date, ratio: float) -> CorporateActionEvent:
    return CorporateActionEvent("L", "p", day, "SPLIT", ratio)


@pytest.mark.proves("market_data_ops.split_rebases_compose_in_session_order")
def test_rebases_compose_in_session_order_and_refuse_what_no_split_explains() -> None:
    """Each bar before an ex-date is the stored one times the factors after it, in session order,
    for any ratio; a move no split explains is not a rebase."""
    stored = tuple(_bar(day) for day in SESSIONS[:10])
    first, second = SESSIONS[4], SESSIONS[8]

    def sent(bar: RawDailyBar, ratios: tuple[float, float]) -> RawDailyBar:
        scale = (1 / ratios[0] if bar.session_date < first else 1.0) * (
            1 / ratios[1] if bar.session_date < second else 1.0
        )
        return _bar(bar.session_date, scale) if scale != 1.0 else bar

    splits = (_split(second, 2.0), _split(first, 4.0))
    resent = [sent(b, (4.0, 2.0)) for b in stored]
    agreement = (1e-6, 1e-12)
    rebased = recognize(stored, resent, splits, agreement, NOW)
    assert rebased is not None
    bars, rebases, whole = rebased
    assert whole and [r.ex_date for r in rebases] == [first, second]
    factors = [0.125] * 4 + [0.5] * 4 + [1.0] * 2  # before both ex-dates, before the second
    assert [bar.close for bar in bars] == [
        b.close * f for b, f in zip(stored, factors, strict=True)
    ]
    gap = resent[:2] + resent[3:]  # a stored session the resend left out
    gapped = recognize(stored, gap, splits, agreement, NOW)
    assert gapped is not None and gapped[2] is False
    for ratios in ((3.0, 2.0), (4.0, 0.5)):  # a 3-for-1, a 1-for-2 reverse: rebased as well
        odd = (_split(second, ratios[1]), _split(first, ratios[0]))
        assert recognize(stored, [sent(b, ratios) for b in stored], odd, agreement, NOW)
    moved = list(resent)
    moved[2] = replace(moved[2], close=moved[2].close * 1.01)
    assert recognize(stored, moved, splits, agreement, NOW) is None
    assert recognize(stored, resent, (), agreement, NOW) is None
    assert recognize(stored, resent, splits, (0.0, 0.0), NOW) is None
