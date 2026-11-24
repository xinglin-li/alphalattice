import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.feature_engine.publication.current_storage import refuse_before_0_2_7

TEST_SOURCE = "src/alphalattice/foundation/feature_engine/publication/current_storage.py"


@pytest.mark.proves("feature_engine.retired_tables_mark_a_workspace_before_0_2_7")
@pytest.mark.parametrize(
    ("tables", "refused"),
    [
        pytest.param({"feature_ineligibility": "BASE TABLE"}, True, id="ineligibility-rows"),
        pytest.param({"feature_year_seal": "BASE TABLE"}, True, id="year-seals"),
        pytest.param({"feature_ineligibility": "VIEW"}, False, id="the-0.2.7-view"),
        pytest.param({"feature_ineligibility_run": "BASE TABLE"}, False, id="the-0.2.7-runs"),
    ],
)
def test_only_a_table_0_2_7_retired_marks_a_workspace_before_it(
    tables: dict[str, str], refused: bool
) -> None:
    """The ineligibility rows as a base table, or the year seals' table, mark a workspace no 0.2.7
    opened and are refused by name; what 0.2.7 left (the view, the runs) passes."""
    if not refused:
        refuse_before_0_2_7(tables)
        return
    with pytest.raises(DataError) as error:
        refuse_before_0_2_7(tables)
    assert error.value.failure.code == "feature_storage.workspace_before_0_2_7"
