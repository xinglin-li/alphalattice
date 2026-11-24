"""A trial's kept return series: its form, and the effective N its class's series give."""

from __future__ import annotations

import json

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer import return_series

TEST_SOURCE = "src/alphalattice/foundation/data_layer/return_series.py"


@pytest.mark.proves("data_layer.a_return_series_is_kept_in_its_form")
@pytest.mark.parametrize(
    ("months", "returns", "refused"),
    [
        pytest.param(["2015-01", "2015-02"], [0.01, -0.02], None, id="kept"),
        pytest.param([], [], "1 to 600 months", id="empty"),
        pytest.param(["2015-01"] * 601, [0.0] * 601, "1 to 600 months", id="too-long"),
        pytest.param(["2015-01"], [0.0, 0.1], "one return each", id="unpaired"),
        pytest.param(["2015-13"], [0.0], "YYYY-MM", id="not-a-month"),
        pytest.param(["2015-02", "2015-01"], [0.0, 0.0], "strictly increasing", id="out-of-order"),
        pytest.param(["2015-01", "2015-01"], [0.0, 0.0], "strictly increasing", id="repeated"),
        pytest.param(["2015-01"], [float("nan")], "finite", id="not-finite"),
    ],
)
def test_a_return_series_is_kept_only_as_ordered_months_of_finite_returns(months, returns, refused):
    """A series is kept as its months and returns, or refused by name for its first fault."""
    if refused is None:
        body = json.loads(return_series.encode(months, returns))
        assert body == {"months": months, "returns": returns}
        return
    with pytest.raises(DataError, match=refused) as error:
        return_series.encode(months, returns)
    assert error.value.failure.code == "data_layer.return_series_invalid"


@pytest.mark.proves("data_layer.effective_n_follows_its_formula")
@pytest.mark.parametrize(
    ("rho", "members", "n"),
    [
        pytest.param(0.0, 10, 10.0, id="unrelated-counts-every-trial"),
        pytest.param(1.0, 10, 1.0, id="identical-counts-one"),
        pytest.param(0.5, 10, 5.5, id="between"),
        pytest.param(-0.3, 10, 10.0, id="negative-held-at-zero"),
        pytest.param(0.4, 1, None, id="one-series"),
        pytest.param(None, 5, None, id="no-pair-correlated"),
    ],
)
def test_effective_n_runs_from_one_to_the_number_of_series(rho, members, n):
    """N = rho + (1 - rho) * M, with rho held to [0, 1]; none below two series or with no pair."""
    assert return_series.effective_n(rho, members) == n
