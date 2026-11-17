"""The reading surface the Host opens at start: views.sql over the log in DuckDB, and `compare`."""

from __future__ import annotations

import itertools
import json
import shutil
import sqlite3
import uuid
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from alphalattice.kernel.data_layer import create_data_root, reading
from alphalattice.kernel.data_layer import log as log_module
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.reading import compare, reading_session
from tests.integration.data_layer import test_project as project_fixture
from tests.integration.data_layer.analytics_view_values import EXPECTED
from tests.integration.data_layer.test_project import AGENT, T0, _aftermath, _define, _research

TEST_SOURCE = "src/alphalattice/kernel/data_layer/reading.py"


@pytest.fixture(scope="module")
def analytics_log(tmp_path_factory):
    """A fixed Host-written research log, shared by all four complete-view cases."""
    root = tmp_path_factory.mktemp("analytics") / "data"
    sequence = itertools.count(1)
    ids = SimpleNamespace(UUID=uuid.UUID, uuid4=lambda: uuid.UUID(int=next(sequence) << 64))
    with (
        patch.object(log_module, "time", SimpleNamespace(time_ns=lambda: T0 * 1000)),
        patch.object(log_module, "uuid", ids),
        patch.object(project_fixture, "uuid", ids),
    ):
        create_data_root(root)
        with HostLog.open(root, now_us=T0) as log:
            _research(log)
            prediction = {
                "target": {"what": "synthetic"},
                "horizon_us": T0 + 86_400_000_000,
                "confidence": 0.6,
                "trial_id": "t1",
            }
            log.record(
                "prediction.registered",
                ("prediction", "p1"),
                prediction,
                text="Synthetic prediction.",
                **AGENT,
            )
            log.record(
                "prediction.scored",
                ("prediction", "p1"),
                {"score": 0.16, "metric": "brier", "metric_version": 1},
                **AGENT,
            )
            log.record(
                "agent.turn",
                phase="completed",
                span="1234567890abcdef",
                started_us=T0 - 1_000_000,
                attrs={"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 2},
                **AGENT,
            )
            log.flush()
    return root


@pytest.mark.parametrize(
    ("view", "order"),
    [
        pytest.param("decision_scorecard", "prediction_id", id="decision-scorecard"),
        pytest.param("agent_daily", "day,actor_id", id="agent-daily"),
        pytest.param("node_io", "day,node", id="node-io"),
        pytest.param("experience", "trial_id", id="experience"),
    ],
)
@pytest.mark.proves("data_layer.analytics_views_preserve_values")
@pytest.mark.proves("data_layer.experience_reads_without_bodies")
def test_experience_returns_each_trial_with_its_measures_and_no_body(
    bare, analytics_log, view, order
):
    """DuckDB reads every analytics value and experience remains complete without bodies."""
    if view == "experience":
        session = reading_session(bare)
        assert session is not None, "the reading views need DuckDB's sqlite extension"
        with closing(session):
            rows = session.execute(
                "SELECT trial_id, outcome, method, body_state, len(measures)"
                " FROM experience ORDER BY 1"
            ).fetchall()
            assert rows == [
                ("t1", "completed", "price_momentum", "present", 4),  # its name since the rename
                ("t2", "completed", "price_momentum", "never_stored", 3),
                ("t3", "completed", "price_momentum", "collected", 3),
            ]
            view_rows = session.execute(
                "SELECT * FROM experience ORDER BY registered_at DESC, trial_id"
            ).fetchall()
            names = [column[0] for column in session.description]
            trials = reading.experience(session, limit=50)
            assert trials == [dict(zip(names, row, strict=True)) for row in view_rows]
            assert (
                reading.experience(
                    session,
                    goal="g1",
                    hypothesis="h1",
                    method="price_momentum",
                    workspace="w1",
                    limit=2,
                )
                == trials[:2]
            )
            assert reading.experience(session, goal="absent", limit=50) == []
            collected = next(trial for trial in trials if trial["trial_id"] == "t3")
            assert collected["body_state"] == "collected" and len(collected["measures"]) == 3
            assert collected["goal_conclusions"] == ["The goal's conclusion."]
            assert collected["goal_lessons"] == ["The goal's lesson."]
            spread = "SELECT n FROM approach_spread WHERE metric = 'sharpe_ratio' ORDER BY n"
            assert [r[0] for r in session.execute(spread).fetchall()] == [1, 2]
            assert session.execute("SELECT trials FROM search_size").fetchall() == [(3,)]
            assert session.execute("SELECT count(*) FROM span WHERE type = 'trial'").fetchone() == (
                3,
            )
    session = reading_session(analytics_log)
    assert session is not None, "the reading views need DuckDB's sqlite extension"
    with closing(session):
        rows = session.execute(f"SELECT * FROM {view} ORDER BY {order}").fetchall()
        actual = {"columns": [column[0] for column in session.description], "rows": rows}
    assert json.loads(json.dumps(actual, default=lambda value: value.isoformat())) == EXPECTED[view]


@pytest.fixture(scope="module")
def bare(tmp_path_factory):
    """The reference history's data root with every artifact body removed: the inline bodies
    deleted from the log and the root's Parquet and blob folders emptied."""
    root = tmp_path_factory.mktemp("bare") / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        artifacts = _research(log)
        _aftermath(root, log, artifacts)
        log.record("conclusion.drawn", ("goal", "g1"), text="The goal's conclusion.", **AGENT)
        log.record("lesson.recorded", ("goal", "g1"), text="The goal's lesson.", **AGENT)
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con, con:
        con.execute("DELETE FROM artifact_body")
    for folder in ("data", "blobs"):
        shutil.rmtree(root / folder)
        (root / folder).mkdir()
    return root


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
