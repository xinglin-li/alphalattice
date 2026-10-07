"""Batched decision windows retain the scalar owner's exact qualification."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    TradabilitySurfaceError,
    decision_eligible_at_close,
    decision_eligible_at_closes,
)
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar


def _source(length: int):
    sessions = tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(length))
    listings = ("listing-A", "listing-B", "listing-C")
    bars = {
        (day, listing): RawDailyBar(
            listing_id=listing,
            provider="SYNTHETIC_QA",
            session_date=day,
            open=100.0 + column,
            high=102.0 + column,
            low=99.0 + column,
            close=101.0 + column,
            volume=1000 + row,
        )
        for row, day in enumerate(sessions)
        for column, listing in enumerate(listings)
    }
    return sessions, listings, bars


def _scalar_values(formations, listings, sessions, bars):
    positions = {day: index for index, day in enumerate(sessions)}
    values = [
        [
            decision_eligible_at_close(
                listing_id=listing,
                formation_session=day,
                history=sessions[max(0, positions[day] - 19) : positions[day] + 1],
                bars=bars,
            )
            for listing in listings
        ]
        for day in formations
    ]
    return np.asarray(values, dtype=np.bool_).reshape(len(formations), len(listings))


def _assert_scalar_cells(formations, listings, sessions, bars):
    before = dict(bars)
    actual = decision_eligible_at_closes(
        formation_sessions=formations, listing_ids=listings, sessions=sessions, bars=bars
    )
    expected = _scalar_values(formations, listings, sessions, bars)
    assert actual.dtype == expected.dtype == np.dtype(np.bool_)
    assert actual.shape == expected.shape == (len(formations), len(listings))
    assert actual.tobytes(order="C") == expected.tobytes(order="C")
    assert dict(bars) == before
    assert all(bars[key] is value for key, value in before.items())
    return actual


@pytest.mark.parametrize("length", [0, 1, 19, 20, 21, 40, 63])
def test_every_batched_cell_has_the_scalar_qualification_on_short_and_long_axes(length):
    """requirement: only a complete twenty-session history can qualify a decision."""

    sessions, listings, bars = _source(length)
    actual = _assert_scalar_cells(sessions, listings, sessions, bars)
    assert not actual[:19].any()
    assert actual[19:].all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("open", 0.0),
        ("open", -1.0),
        ("open", np.nan),
        ("open", np.inf),
        ("open", -np.inf),
        ("close", 0.0),
        ("close", -1.0),
        ("close", np.nan),
        ("close", np.inf),
        ("close", -np.inf),
        ("volume", 0),
        ("volume", -1),
        ("missing", None),
    ],
)
def test_an_invalid_bar_keeps_every_overlapping_window_false_until_it_leaves(field, value):
    """requirement: a missing or inadmissible source row has the scalar window's exact extent."""

    sessions, listings, bars = _source(41)
    key = (sessions[18], listings[1])
    if field == "missing":
        del bars[key]
    else:
        bars[key] = replace(bars[key], **{field: value})
    actual = _assert_scalar_cells(sessions, listings, sessions, bars)
    assert not actual[19:38, 1].any()
    assert actual[38:, 1].all()
    assert actual[19:, [0, 2]].all()


def test_formation_subsets_and_listing_permutations_keep_their_requested_cell_axis():
    """requirement: selection and repeated formations do not reorder a qualified cell."""

    sessions, listings, bars = _source(63)
    formations = (sessions[40], sessions[19], sessions[5], sessions[40])
    selected = (listings[2], listings[0])
    actual = _assert_scalar_cells(formations, selected, sessions, bars)
    assert actual[:, 0].tolist() == [True, True, False, True]
    empty_listings = _assert_scalar_cells(formations, (), sessions, bars)
    assert empty_listings.shape == (4, 0)
    empty_formations = _assert_scalar_cells((), listings, sessions, bars)
    assert empty_formations.shape == (0, 3)
    absent = _assert_scalar_cells(formations, ("listing-absent", listings[0]), sessions, bars)
    assert not absent[:, 0].any()


class _ReadCountingBars(Mapping[tuple[date, str], RawDailyBar]):
    def __init__(self, values, *, through):
        self.source = values
        self.through = through
        self.reads = []

    def __getitem__(self, key):
        assert key[0] <= self.through, "a future bar cannot qualify a past decision"
        self.reads.append(key)
        return self.source[key]

    def __iter__(self) -> Iterator[tuple[date, str]]:
        return iter(self.source)

    def __len__(self) -> int:
        return len(self.source)


def test_each_source_bar_is_read_once_and_future_invalid_bars_are_never_consulted():
    """requirement: reuse is one bounded source pass, not twenty reads per decision cell."""

    sessions, listings, values = _source(63)
    through = sessions[40]
    for key, value in tuple(values.items()):
        if key[0] > through:
            values[key] = replace(value, open=np.nan, close=np.inf, volume=0)
    before = dict(values)
    formations = sessions[:41]
    batched = _ReadCountingBars(values, through=through)
    actual = decision_eligible_at_closes(
        formation_sessions=formations, listing_ids=listings, sessions=sessions, bars=batched
    )
    scalar = _ReadCountingBars(values, through=through)
    expected = _scalar_values(formations, listings, sessions, scalar)
    assert actual.tobytes(order="C") == expected.tobytes(order="C")
    assert len(batched.reads) == 41 * len(listings)
    assert len(set(batched.reads)) == len(batched.reads)
    assert len(scalar.reads) > len(batched.reads)
    assert values == before
    assert all(values[key] is value for key, value in before.items())


@pytest.mark.parametrize(
    "bad_axis,code",
    [
        ("unsorted", "data_tradability.schedule_axis_invalid"),
        ("duplicate-session", "data_tradability.schedule_axis_invalid"),
        ("duplicate-listing", "data_tradability.listing_axis_invalid"),
        ("formation-absent", "data_tradability.source_axis_mismatch"),
    ],
)
def test_invalid_axes_are_refused_before_any_source_read(bad_axis, code):
    """requirement: calendar and selection admission precede source work."""

    sessions, listings, values = _source(40)
    formations = sessions[-2:]
    if bad_axis == "unsorted":
        sessions = tuple(reversed(sessions))
    elif bad_axis == "duplicate-session":
        sessions = (*sessions, sessions[-1])
    elif bad_axis == "duplicate-listing":
        listings = (*listings, listings[0])
    else:
        formations = (sessions[-1] + timedelta(days=1),)
    bars = _ReadCountingBars(values, through=max(sessions))
    with pytest.raises(TradabilitySurfaceError, match=code):
        decision_eligible_at_closes(
            formation_sessions=formations, listing_ids=listings, sessions=sessions, bars=bars
        )
    assert bars.reads == []
