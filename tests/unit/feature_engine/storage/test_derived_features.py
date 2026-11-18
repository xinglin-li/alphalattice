import pytest

from alphalattice.foundation.feature_engine.storage import derived_features
from alphalattice.kernel.data.errors import DataError

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/storage/derived_features.py"


@pytest.mark.proves("feature_engine.the_feature_read_setting_names_one_choice")
@pytest.mark.parametrize(
    ("setting", "chosen"),
    [
        pytest.param(None, "stored", id="unset-is-the-rollback"),
        pytest.param("derived", "derived", id="derived"),
        pytest.param("store", "feature.feature_reads_unknown", id="misspelt"),
    ],
)
def test_the_setting_names_one_choice(monkeypatch, setting, chosen):
    """The setting is stored when unset, derived when set so, and refused by name otherwise."""
    if setting is None:
        monkeypatch.delenv(derived_features.SETTING, raising=False)
    else:
        monkeypatch.setenv(derived_features.SETTING, setting)
    try:
        found = derived_features.chosen()
    except DataError as error:
        found = error.failure.code
    assert found == chosen
