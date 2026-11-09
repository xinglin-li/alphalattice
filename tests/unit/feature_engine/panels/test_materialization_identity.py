from datetime import date

import numpy as np
import pandas as pd
import pytest

from alphalattice.foundation.feature_engine.panels.materialization_identity import (
    feature_source_values_hash,
)

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/panels/materialization_identity.py"


@pytest.mark.proves("feature_engine.materialized_source_identity_binds_missing_positions")
def test_the_source_identity_distinguishes_missing_and_zero_at_each_session():
    sessions = (date(2026, 1, 2), date(2026, 1, 5))
    zero_then_missing = pd.DataFrame({"session_date": sessions, "close": [0.0, np.nan]})
    missing_then_zero = pd.DataFrame({"session_date": sessions, "close": [np.nan, 0.0]})
    zeroes = pd.DataFrame({"session_date": sessions, "close": [0.0, 0.0]})
    first = feature_source_values_hash(zero_then_missing, ("session_date", "close"))
    assert first == feature_source_values_hash(zero_then_missing.copy(), ("close",))
    assert first != feature_source_values_hash(missing_then_zero, ("close",))
    assert first != feature_source_values_hash(zeroes, ("close",))
