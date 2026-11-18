from types import SimpleNamespace

import pytest

from alphalattice.control.workspace_runtime import derived_views

TEST_SOURCE = "src/alphalattice/control/workspace_runtime/derived_views.py"
SETTING = "ALPHALATTICE_TEST_READS"
VIEW = ("feature_daily_current", SETTING, "SELECT 42 AS x")


def _workspace(views):
    """A connection whose file declares `views` (None: no declaration table at all)."""

    def execute(sql, parameters=None):
        if "duckdb_tables" in sql:
            return SimpleNamespace(fetchall=lambda: [] if views is None else [(1,)])
        return SimpleNamespace(fetchall=lambda: list(views))

    return SimpleNamespace(execute=execute)


@pytest.mark.proves("workspace_runtime.a_derived_view_answers_its_readers")
@pytest.mark.parametrize(
    ("views", "setting", "registered", "served"),
    [
        pytest.param(
            [VIEW], "derived", True, [("feature_daily_current", "SELECT 42 AS x")], id="served"
        ),
        pytest.param([VIEW], None, True, [], id="setting-unset"),
        pytest.param([VIEW], "stored", True, [], id="stored"),
        pytest.param(None, "derived", False, [], id="nothing-declared"),
        pytest.param(
            [VIEW], "derived", False, "workspace_database.derived_setup_absent", id="no-setup"
        ),
    ],
)
def test_a_declared_view_is_served_under_its_setting_and_never_silently_skipped(
    monkeypatch, views, setting, registered, served
):
    """A declared view is served only while its setting is derived, and a connection that would
    serve one with no setup registered is refused by name."""
    seen = []
    held = derived_views.registered()
    derived_views.register((lambda connection, chosen: seen.extend(chosen)) if registered else None)
    if setting is None:
        monkeypatch.delenv(SETTING, raising=False)
    else:
        monkeypatch.setenv(SETTING, setting)
    try:
        derived_views.serve(_workspace(views))
    except RuntimeError as error:
        seen = str(error)
    finally:
        derived_views.register(held)
    assert seen == served
