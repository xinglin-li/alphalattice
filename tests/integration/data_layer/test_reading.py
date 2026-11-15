"""The reading surface the Host opens at start: views.sql over the log in DuckDB, and `compare`."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing

import pytest

from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.reading import compare, reading_session
from tests.integration.data_layer.test_project import AGENT, T0, _aftermath, _define, _research

TEST_SOURCE = "src/alphalattice/kernel/data_layer/reading.py"


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    """The reference history's data root with every artifact body removed: the inline bodies
    deleted from the log and the root's Parquet and blob folders emptied."""
    root = tmp_path_factory.mktemp("bare") / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        artifacts = _research(log)
        _aftermath(log, artifacts)
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con, con:
        con.execute("DELETE FROM artifact_body")
    for folder in ("data", "blobs"):
        shutil.rmtree(root / folder)
        (root / folder).mkdir()
    return root


@pytest.mark.proves("data_layer.experience_reads_without_bodies")
def test_experience_returns_each_trial_with_its_measures_and_no_body(bare):
    """Every trial's state, action and effect read from profiles and measures alone."""
    session = reading_session(bare)
    assert session is not None, "the reading views need DuckDB's sqlite extension"
    with closing(session):
        rows = session.execute(
            "SELECT trial_id, outcome, method, body_state, len(measures) FROM experience ORDER BY 1"
        ).fetchall()
        assert rows == [
            ("t1", "completed", "momentum", "present", 4),
            ("t2", "completed", "momentum", "never_stored", 3),
            ("t3", "completed", "momentum", "collected", 3),
        ]
        spread = "SELECT n FROM approach_spread WHERE metric = 'sharpe' ORDER BY n"
        assert [r[0] for r in session.execute(spread).fetchall()] == [1, 2]
        assert session.execute("SELECT trials FROM search_size").fetchall() == [(3,)]
        assert session.execute("SELECT count(*) FROM span WHERE type = 'trial'").fetchone() == (3,)


@pytest.mark.proves("data_layer.experience_reads_without_bodies")
def test_compare_converts_an_older_metric_version_to_the_newer(tmp_path):
    """Two artifacts that share no version of a metric compare after the older one's values are
    converted through the newer version's registered conversion."""
    root = tmp_path / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
        conversion = {"scale": 2.0, "offset": 0.0}
        text = "Sharpe on another scale."
        _define(
            log, "metric", "sharpe", 2, text=text, relation="convertible", conversion=conversion
        )
        meaning = {"type": "backtest", "identity_kind": "content", "storage": "none"}
        sides = {"a": [["sharpe", 1, "", 0.5]], "b": [["sharpe", 2, "", 1.5]]}
        for side, measures in sides.items():
            log.record(
                "artifact.produced", ("artifact", side), meaning | {"meaning": side}, **AGENT
            )
            log.record("result.recorded", ("artifact", side), {"measures": measures}, **AGENT)
        log.flush()
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        (row,) = compare(con, "a", "b")
    assert (row["version"], row["a"], row["b"], row["diff"]) == (2, 1.0, 1.5, 0.5)
    assert row["relation"] == "convertible"
