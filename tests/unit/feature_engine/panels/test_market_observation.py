from datetime import date, timedelta

import pyarrow as pa
import pytest

from alphalattice.foundation.feature_engine.panels.market_observation import (
    PanelMarketObservation,
    panel_market_observation,
    sessions_without_a_bar,
)

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/market_observation.py"

AXIS = tuple(date(2025, 5, 12) + timedelta(days=day) for day in range(8))


def _bars(**held: tuple[int, ...]) -> pa.Table:
    pairs = [(listing, AXIS[index]) for listing, indexes in held.items() for index in indexes]
    return pa.table(
        {
            "listing_id": pa.array([listing for listing, _ in pairs], type=pa.string()),
            "session_date": pa.array([day for _, day in pairs], type=pa.date32()),
        }
    )


@pytest.mark.proves("feature_engine.panel_records_sessions_without_a_bar")
@pytest.mark.parametrize(
    ("bars", "absent"),
    [
        pytest.param(_bars(A=tuple(range(8))), {}, id="traded"),
        pytest.param(_bars(A=(0, 1, 5, 6, 7)), {"A": AXIS[2:5]}, id="halted"),
        pytest.param(_bars(A=(1, 2, 3, 4, 5, 6)), {}, id="listed-late-and-left-early"),
        pytest.param(
            _bars(B=tuple(range(8)), A=(0, 2, 3, 4, 5, 7)),
            {"A": (AXIS[1], AXIS[6])},
            id="two-gaps-beside-a-traded-listing",
        ),
        pytest.param(_bars(), {}, id="no-bar"),
    ],
)
def test_a_listing_misses_only_the_sessions_inside_its_own_span(bars, absent):
    """Only a session between a listing's first and last bar on the axis is without a bar."""
    assert sessions_without_a_bar(AXIS, bars) == absent


@pytest.mark.proves("feature_engine.panel_records_sessions_without_a_bar")
@pytest.mark.parametrize(
    ("edit", "outcome"),
    [
        pytest.param(None, "recorded", id="as-written"),
        pytest.param({"version": "panel-market-observation/v0"}, "refused", id="another-version"),
        pytest.param(
            {"absent_sessions": {"A": ["2025-05-15", "2025-05-14"]}}, "refused", id="reordered"
        ),
        pytest.param({"absent_sessions": {}}, "refused", id="empty"),
        pytest.param({"through": None}, "refused", id="malformed"),
    ],
)
def test_a_panel_record_reads_back_only_as_written(edit, outcome):
    """A record reads back as written; another version or a record off its form is refused."""
    written = PanelMarketObservation(
        through=AXIS[-1], source_watermark_hash="a" * 64, absent_sessions={"A": AXIS[2:4]}
    )
    record = written.to_payload() | (edit or {})
    manifest = {"safe_summary": {"market_observation": record}}
    if outcome == "recorded":
        assert panel_market_observation(manifest) == written
        assert panel_market_observation({"safe_summary": {}}) is None
        return
    with pytest.raises(ValueError, match=r"feature_panel.market_observation_invalid"):
        panel_market_observation(manifest)
