from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.kernel.data_layer import derived_reads

TEST_SOURCE = "src/alphalattice/kernel/data_layer/derived_reads.py"
VIEWS = [("feature_daily_current", "SELECT 1"), ("factor_cov_current", "SELECT 2")]


@pytest.mark.proves("data_layer.a_workspace_attaches_the_catalog_once")
@pytest.mark.parametrize(
    ("held", "attached"),
    [pytest.param(False, 1, id="first-connection"), pytest.param(True, 0, id="instance-holds-it")],
)
def test_the_catalog_is_attached_read_only_once_and_each_view_made_per_connection(
    monkeypatch, held, attached
):
    """A connection attaches the data root's catalog read-only only when its instance lacks it,
    and makes every served view a TEMP view of its own."""
    run = []

    def execute(sql, parameters=None):
        run.append(sql)
        rows = [(1,)] if held and "duckdb_databases" in sql else []
        return SimpleNamespace(fetchall=lambda: rows, complete=lambda: None)

    monkeypatch.setattr(derived_reads, "data_root", lambda: Path("D:/root"))
    monkeypatch.setattr(derived_reads, "load_extensions", lambda connection: run.append("LOAD"))
    derived_reads.serve(SimpleNamespace(execute=execute), VIEWS)
    attaches = [sql for sql in run if sql.startswith("ATTACH")]
    assert len(attaches) == attached and run.count("LOAD") == attached
    assert all("READ_ONLY" in sql and "D:/root/machine/catalog.sqlite" in sql for sql in attaches)
    views = [sql for sql in run if sql.startswith("CREATE OR REPLACE TEMP VIEW")]
    assert views == [f"CREATE OR REPLACE TEMP VIEW {name} AS {body}" for name, body in VIEWS]
