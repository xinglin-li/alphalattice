"""Every state table of log.sqlite is a projection of its events: the 0.2.8 reference history,
written through the Host's one write path, replays ALL EQUAL into an empty log."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer import project as projections
from alphalattice.kernel.data_layer.chain import CHAIN_VERSION, digest
from alphalattice.kernel.data_layer.log import HostLog, replay, verify_whole
from alphalattice.kernel.data_layer.project import (
    PROJECTED,
    definition_id,
    differences,
    recipe_key,
)
from alphalattice.kernel.data_layer.reading import reading_session
from alphalattice.kernel.data_layer.store import StorePort, feature_table

TEST_SOURCE = "src/alphalattice/kernel/data_layer/project.py"
T0 = 1_794_700_000_000_000  # 2026-11-14, UTC
AGENT = {"workspace": "w1", "actor": ("agent", "claude"), "node": "research"}
PERSON = {"workspace": "w1", "actor": ("person", "me")}
N = 10**9  # ledger quantities are nano-units
MOMENTUM, OUTCOME = definition_id("method", "momentum"), definition_id("method", "realized_outcome")
SHARPE, FACTORS = definition_id("metric", "sharpe"), definition_id("feature_set", "momentum_set")


def _define(log, family, name, version=1, *, text, **more):
    """A definition entered by its event; `more` carries status, relation, conversion and who."""
    who = {k: more.pop(k) for k in ("actor", "node", "workspace") if k in more}
    body = {"family": family, "name": name, "version": version, "spec": more.pop("spec", {})}
    body |= {"status": more.pop("status", "standard"), "text": text, **more}
    key = {"family": family, "name": name, "version": version}
    text = json.dumps(body, sort_keys=True)
    return log.record(
        "definition.registered", ("definition", f"{family}/{name}"), key, text=text, **who
    )


def _trial(log, trial, recipe, sharpe, storage):
    """One trial from registration to its headline result; returns its result artifact."""
    registered = log.record(
        "trial.registered",
        ("trial", trial),
        {
            "goal_id": "g1",
            "hypothesis_id": "h1",
            "selection_id": "s1",
            "recipe_key": recipe,
            "seeds": [7],
            "product_version": "0.2.8",
            "context": {"regime": {"vix": 14.2}},
            "tags": {"instrument": ["AAPL", "MSFT"], "theme": "momentum"},
        },
        **AGENT,
    )
    span = uuid.uuid4().hex[:16]
    log.record("trial", ("trial", trial), phase="started", span=span, cause=registered, **AGENT)
    artifact = digest(f"{recipe}/{span}".encode())
    log.record(
        "artifact.produced",
        ("artifact", artifact),
        {
            "type": "backtest",
            "meaning": "Does 12-1 momentum earn a premium on the S&P 500, 2015 to 2024?",
            "recipe_key": recipe,
            "selection_id": "s1",
            "trial_id": trial,
            "storage": storage,
            "identity_kind": "recipe",
        },
        span=span,
        **AGENT,
    )
    measures = [["sharpe", 1, "", sharpe], ["max_drawdown", 1, "", 0.2]]
    measures.append(["ic_mean", 1, "horizon=21d", 0.03])
    log.record(
        "result.recorded",
        ("artifact", artifact),
        {"trial_id": trial, "measures": measures},
        span=span,
        **AGENT,
    )
    log.record(
        "trial",
        ("trial", trial),
        {"result_artifact": artifact},
        phase="completed",
        span=span,
        **AGENT,
    )
    return artifact


def _research(log):
    """A goal, a hypothesis, a selection, two recipes and three trials."""
    log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
    log.record("goal.set", ("goal", "g1"), {"title": "Find a momentum signal"}, **PERSON)
    claim = "12-1 momentum predicts next-month returns."
    log.record("hypothesis.stated", ("hypothesis", "h1"), {"goal_id": "g1"}, text=claim, **AGENT)
    window = {"cutoff_us": 1_767_225_600_000_000, "universe": "sp500"}
    window |= {"valid_from": "2015-01-01", "valid_to": "2024-12-31"}
    log.record("selection.made", ("selection", "s1"), window, **AGENT)
    _define(log, "method", "momentum", text="Past return over a lookback.", **AGENT)
    _define(log, "method", "realized_outcome", text="What a signal earned after its cutoff.")
    keys = {}
    for lookback in (12, 6):
        keys[lookback] = recipe_key(MOMENTUM, 1, {"lookback": lookback})
        attrs = {"method": "momentum", "method_version": 1, "params": {"lookback": lookback}}
        log.record("recipe.defined", ("recipe", keys[lookback]), attrs, **AGENT)
    trials = (("t1", 12, 0.9, "table"), ("t2", 12, 1.3, "none"), ("t3", 6, 0.4, "table"))
    return {t: _trial(log, t, keys[lookback], s, storage) for t, lookback, s, storage in trials}


def _aftermath(root, log, artifacts):
    """What follows the trials: notes, a prediction, a kept trial and its outcome, tags, a task,
    a book and its ledger, a setting, a new metric version, a forgotten note, a checkpoint."""
    text = "Positive but weak after costs."
    note = log.record("conclusion.drawn", ("trial", "t1"), text=text, **AGENT)
    log.record("feedback.given", ("trial", "t1"), {"adopted": True}, **PERSON)
    standing = {"standing": "supported"}
    log.record("hypothesis.standing_changed", ("hypothesis", "h1"), standing, **AGENT)
    target = {"what": "sp500 up over 21 sessions"}
    bet = {"target": target, "horizon_us": 1_770_000_000_000_000, "confidence": 0.6}
    claim = "The S&P 500 closes higher 21 sessions from now."
    bet |= {"trial_id": "t1"}
    log.record("prediction.registered", ("prediction", "p1"), bet, text=claim, **AGENT)
    score = {"score": 0.16, "metric": "brier", "metric_version": 1}
    log.record("prediction.scored", ("prediction", "p1"), score)
    log.record("trial.kept", ("trial", "t1"), **PERSON)
    horizon = recipe_key(
        OUTCOME, 1, {"horizon": "63d"}, [recipe_key(MOMENTUM, 1, {"lookback": 12})]
    )
    attrs = {"method": "realized_outcome", "method_version": 1, "params": {"horizon": "63d"}}
    attrs["upstream"] = [recipe_key(MOMENTUM, 1, {"lookback": 12})]  # the signal it scores
    log.record("recipe.defined", ("recipe", horizon), attrs)
    outcome = digest(f"{horizon}/t1".encode())
    meaning = "What the t1 signal earned over the 63 sessions after its cutoff."
    produced = {"type": "outcome", "identity_kind": "recipe", "recipe_key": horizon}
    produced |= {"trial_id": "t1", "storage": "none", "meaning": meaning}
    log.record("artifact.produced", ("artifact", outcome), produced)
    measures = [["sharpe", 1, "horizon=63d", 0.7]]
    log.record("result.recorded", ("artifact", outcome), {"trial_id": "t1", "measures": measures})
    log.record("tag.added", ("trial", "t3"), {"key": "theme", "value": "reversal"}, **AGENT)
    log.record("tag.removed", ("trial", "t3"), {"key": "instrument", "value": "MSFT"}, **AGENT)
    log.record("artifact.collected", ("artifact", artifacts["t3"]))
    requested = {"type": "evidence.prepare", "product_version": "0.2.8", "input_hash": "i"}
    requested |= {
        "plan_hash": "p",
        "workflow_definition_hash": "f",
        "admitted_at": "2026-11-14 00:00:00",
    }
    log.record("task.requested", ("task", "k1"), requested, text='{"asked": "evidence"}', **AGENT)
    span = uuid.uuid4().hex[:16]
    log.record("task", ("task", "k1"), phase="started", span=span, **AGENT)
    step = {"step": "download", "output": {"docs": 12}, "used": [artifacts["t1"]]}
    log.record("step.completed", ("task", "k1"), step, span=span, **AGENT)
    wait = {"task_id": "k1", "kind": "event", "event_type": "decision.made", "purpose": "approval"}
    log.record("wait.started", ("wait", "q1"), wait, span=span, **AGENT)
    log.record("wait.resolved", ("wait", "q1"), {}, span=span, **AGENT)
    log.record("task", ("task", "k1"), phase="completed", span=span, **AGENT)
    log.record("book.activated", ("book", "b1"), {"mode": "paper"}, **PERSON)
    lines = [["book", "AAPL", 10 * N], ["market", "AAPL", -10 * N]]
    lines += [["book", "CASH:USD", -1500 * N], ["market", "CASH:USD", 1500 * N]]
    log.record("ledger.posted", ("book", "b1"), {"lines": lines})
    setting = {"scope": "machine", "key": "provider", "value": "yfinance"}
    log.record("setting.changed", ("setting", "machine/provider"), setting, **PERSON)
    spec = {"unit": "ratio", "direction": "higher", "params": {"periods": 365}}
    conversion = {"scale": 1.2035, "offset": 0.0}
    text = "Annualized over 365 days."
    _define(
        log,
        "metric",
        "sharpe",
        2,
        text=text,
        spec=spec,
        relation="convertible",
        conversion=conversion,
    )
    moved = {"family": "metric", "name": "sharpe", "version": 1, "status": "deprecated"}
    log.record("definition.status_changed", ("definition", "metric/sharpe"), moved)
    log.record("record.forgotten", ("event", str(note)), {"seq": note}, **PERSON)
    _renames(root, log)
    seq, head = log.read("SELECT seq, head FROM chain_head")[0]
    mark = {"kind": "backup", "seq": seq, "head": head, "chain_version": CHAIN_VERSION}
    log.record("chain.checkpointed", attrs=mark)


def _task_control(log):
    """One Task as Task Control writes it: each event carries its sealed record as its text."""
    task, at = ("task", "k2"), "2026-11-14 00:00:01.000000"
    admitted = {"type": "evidence.prepare", "product_version": "0.2.8", "input_hash": "i2"}
    admitted |= {"plan_hash": "p2", "workflow_definition_hash": "f2", "admitted_at": at}
    log.record("task.requested", task, admitted | {"admission_sequence": 1}, text="{}", **AGENT)
    saved = {"lifecycle": "RUNNING", "updated_at": at}
    log.record("task.saved", task, saved, text='{"version": 2}', **AGENT)
    item = {"stage_id": "verify", "ordinal": 0, "lifecycle": "PENDING", "updated_at": at}
    log.record("work_item.saved", task, item, text='{"version": 1}', **AGENT)
    started = {"execution_id": "x1", "graph_thread_id": "g1", "worker_instance_id": "w1"}
    started |= {"compatibility_hash": "c1", "started_at": at}
    log.record("execution.started", task, started, text='{"execution_id": "x1"}', **AGENT)
    noted = {"event_id": "n1", "execution_id": "x1", "sequence": 1, "kind": "task.started"}
    log.record("task.noted", task, noted | {"recorded_at": at}, text="{}", **AGENT)
    verified = {"receipt_hash": "r1", "execution_id": "x1", "stage_id": "verify"}
    log.record("stage.verified", task, verified | {"observed_at": at}, text="{}", **AGENT)
    accepted = {"command_id": "c1", "command_hash": "h1", "accepted_at": at}
    log.record("command.accepted", task, accepted, text='{"kind": "CANCEL_REQUESTED"}', **AGENT)
    resolved = {"command_id": "c1", "resolved_at": at, "resolution": "CANCELLED"}
    log.record("command.resolved", task, resolved, **AGENT)
    log.record("task.saved", task, {"lifecycle": "CANCELLED", "updated_at": at}, text="{}", **AGENT)


def _renames(root, log):
    """A feature set over momentum, its cache entry and derived partition, then a method, a metric
    and that feature set renamed; momentum is defined again under its new name: the same recipe."""
    spec = {"method": "momentum", "method_version": 1, "factors": ["mom_12_1"]}
    _define(log, "feature_set", "momentum_set", text="Momentum factors.", spec=spec, **AGENT)
    _cached(root, log, recipe_key(definition_id("method", "momentum"), 1, {"lookback": 12}))
    for family, name, to, who in (
        ("method", "momentum", "price_momentum", AGENT),
        ("metric", "sharpe", "sharpe_ratio", PERSON),
        ("feature_set", "momentum_set", "momentum_factors", AGENT),
    ):
        renamed = {"family": family, "name": name, "to": to}
        log.record("definition.renamed", ("definition", f"{family}/{name}"), renamed, **who)
    again = {"method": "price_momentum", "method_version": 1, "params": {"lookback": 12}}
    log.record(
        "recipe.defined", ("recipe", recipe_key(MOMENTUM, 1, {"lookback": 12})), again, **AGENT
    )


def _cached(root, log, key):
    """The cache entry and derived partition a first computation leaves, as their writers keep them:
    the entry keyed by the recipe, the partition's table named by the feature set's lineage."""
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con, con:
        entry = "INSERT INTO cache_entry VALUES (?, '2024', 'rows', 1, 8, 1, ?, ?)"
        con.execute(entry, (key, T0, T0))
    port = StorePort.open(root, log)
    try:
        table = f"lake.derived.{feature_table(definition_id('feature_set', 'momentum_set'))}"
        columns = "recipe_key VARCHAR, input_digest VARCHAR, session_date DATE, mom_12_1 DOUBLE"
        port.lake.execute(f"CREATE TABLE {table} ({columns})").complete()
        port.lake.execute(f"ALTER TABLE {table} SET PARTITIONED BY (year(session_date))").complete()
        # past the catalog's 1000-row inlining, so the year's rows land as a Parquet partition
        days = "DATE '2024-01-02' + CAST(i % 300 AS INTEGER)"
        rows = f"INSERT INTO {table} SELECT ?, 'rows', {days}, 0.5 FROM range(1200) r(i)"
        port.lake.execute(rows, [key]).complete()
    finally:
        port.close()


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    """A data root whose log holds the reference history, and its result artifacts."""
    root = tmp_path_factory.mktemp("history") / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        artifacts = _research(log)
        _aftermath(root, log, artifacts)
        _task_control(log)
    return root, artifacts


def _log(root):
    """A connection to `root`'s log."""
    con = sqlite3.connect(root / "machine" / "log.sqlite")
    con.row_factory = sqlite3.Row
    return con


@pytest.mark.proves("data_layer.replay_rebuilds_every_projection")
def test_the_reference_history_replays_all_equal(history, tmp_path):
    """Every projected table, the chain head included, is rebuilt from the events alone."""
    root, _ = history
    with closing(_log(root)) as source, closing(replay(source, tmp_path / "r.sqlite")) as target:
        assert differences(source, target) == []
        assert verify_whole(source) and verify_whole(target)
        empty = [t for t in PROJECTED if not source.execute(f"SELECT 1 FROM {t}").fetchone()]
    assert empty == []  # the history reaches every projected table


@pytest.mark.proves("data_layer.replay_rebuilds_every_projection")
def test_the_projections_hold_the_research_memory(history):
    """Trials are found by tag; kept, never-stored and collected artifacts keep their measures."""
    root, artifacts = history
    with closing(_log(root)) as con:
        found = con.execute(
            "SELECT subject_id FROM tag WHERE key_id = ? AND value = 'AAPL' ORDER BY 1",
            (definition_id("tag", "instrument"),),
        ).fetchall()
        assert [r[0] for r in found] == ["t1", "t2", "t3"]
        state = con.execute("SELECT state, kept FROM trial WHERE trial_id = 't1'").fetchone()
        assert tuple(state) == ("completed", 1)
        bodies = {
            a: con.execute(
                "SELECT body_state FROM artifact WHERE artifact_id = ?", (a,)
            ).fetchone()[0]
            for a in artifacts.values()
        }
        assert sorted(bodies.values()) == ["collected", "never_stored", "present"]
        measured = "SELECT count(*) FROM measure JOIN artifact USING (artifact_id) WHERE trial_id"
        assert con.execute(f"{measured} IN ('t1', 't2', 't3')").fetchone()[0] == 10
        note = "SELECT text FROM research_note WHERE subject_id = 't1'"
        assert con.execute(note).fetchone()[0] is None  # forgotten; its event stays


@pytest.mark.proves("data_layer.a_rename_moves_no_key")
def test_a_rename_moves_no_recipe_measure_cache_or_partition_key(history):
    """A renamed method, metric and feature set keep their lineage's id: every recipe, measure,
    cache entry and derived partition keyed by it stays, every old name still finds it, and the
    reading views show the new names."""
    root, _ = history
    with closing(_log(root)) as con:
        lineages = dict(con.execute("SELECT name, definition_id FROM definition_lineage"))
        assert (lineages["price_momentum"], lineages["sharpe_ratio"]) == (MOMENTUM, SHARPE)
        assert lineages["momentum_factors"] == FACTORS
        assert "momentum" not in lineages and "sharpe" not in lineages
        recipes = con.execute("SELECT recipe_key FROM recipe WHERE method_id = ?", (MOMENTUM,))
        keys = {recipe_key(MOMENTUM, 1, {"lookback": n}) for n in (12, 6)}
        assert {row[0] for row in recipes} == keys  # one approach under either name: one recipe
        held = "SELECT definition_id FROM definition WHERE family = ? AND name = ? AND version = 1"
        assert con.execute(held, ("metric", "sharpe")).fetchone()[0] == SHARPE
        measured = "SELECT count(*) FROM measure WHERE metric_id = ? AND metric_version = 1"
        assert con.execute(measured, (SHARPE,)).fetchone()[0] == 4
        refs = "SELECT ref_id, ref_version FROM definition_ref WHERE definition_id = ?"
        assert [tuple(r) for r in con.execute(refs, (FACTORS,))] == [(MOMENTUM, 1)]
        shown = "SELECT DISTINCT metric FROM metric WHERE metric_id = ?"
        assert [tuple(r) for r in con.execute(shown, (SHARPE,))] == [("sharpe_ratio",)]
        cached = "SELECT input_digest FROM cache_entry WHERE recipe_key = ? AND part = '2024'"
        renamed = recipe_key(lineages["price_momentum"], 1, {"lookback": 12})
        assert con.execute(cached, (renamed,)).fetchone()[0] == "rows"
    reading = reading_session(root)
    try:  # the partition written under the old names is found, in place, under the new
        table = feature_table(lineages["momentum_factors"])
        listed = "SELECT data_file FROM ducklake_list_files('lake', ?, schema => 'derived')"
        [(path,)] = reading.execute(listed, [table]).fetchall()
        assert Path(path).parent.name == "year=2024" and Path(path).parent.parent.name == table
        held = f"SELECT count(*), min(recipe_key), max(recipe_key) FROM lake.derived.{table}"
        assert reading.execute(held).fetchone() == (1200, renamed, renamed)
    finally:
        reading.close()


@pytest.mark.proves("data_layer.replay_rebuilds_every_projection")
def test_a_projection_that_reads_the_clock_fails_replay_by_name(tmp_path, monkeypatch):
    """A handler that writes the wall clock rebuilds a different goal row, and replay says so."""
    honest = projections.HANDLERS["goal.set"]

    def clocked(con, e, a, text):
        honest(con, e, a | {"title": f"{a['title']} at {time.time_ns()}"}, text)

    monkeypatch.setitem(projections.HANDLERS, "goal.set", clocked)
    copy = tmp_path / "data"
    create_data_root(copy)
    with HostLog.open(copy, now_us=T0) as log:
        log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
        log.record("goal.set", ("goal", "g1"), {"title": "Find a momentum signal"}, **PERSON)
    with closing(_log(copy)) as source, closing(replay(source, tmp_path / "r.sqlite")) as target:
        assert differences(source, target) == ["goal"]


@pytest.mark.proves("data_layer.replay_rebuilds_every_projection")
def test_a_projection_refusal_leaves_no_event(tmp_path):
    """An unbalanced posting, a recipe under another key or reading an unknown one, an untagged
    key, a trial or task whose parent, selection or goal is another workspace's or hypothesis's,
    and a rename onto a name another lineage holds leave no trace."""
    copy = tmp_path / "data"
    create_data_root(copy)
    with HostLog.open(copy, now_us=T0) as log:
        for workspace in ("w1", "w2"):
            log.record("workspace.created", ("workspace", workspace), {"name": workspace})
        log.record("book.activated", ("book", "b1"), {"mode": "paper"}, **PERSON)
        _define(log, "method", "momentum", text="Past return over a lookback.", **AGENT)
        window = {"cutoff_us": 1, "universe": "sp500"}
        there = AGENT | {"workspace": "w2"}
        log.record("selection.made", ("selection", "s2"), window, **there)
        log.record(
            "goal.set",
            ("goal", "g2"),
            {"title": "Find a value signal"},
            **PERSON | {"workspace": "w2"},
        )
        log.record(
            "hypothesis.stated",
            ("hypothesis", "h2"),
            {"goal_id": "g2"},
            text="Value pays.",
            **there,
        )
        requested = {"type": "factor.run", "product_version": "0.2.8", "input_hash": "i"}
        requested |= {
            "plan_hash": "p",
            "workflow_definition_hash": "d",
            "admitted_at": "2026-11-14 00:00:00",
        }
        log.record("task.requested", ("task", "k1"), requested, workspace="w1", version=2)
        under_k1, under_g2 = requested | {"parent_task_id": "k1"}, requested | {"goal_id": "g2"}
        item = {"stage_id": "s", "lifecycle": "PENDING", "updated_at": "2026-11-14 00:00:00"}
        log.record("work_item.saved", ("task", "k1"), item | {"ordinal": 0}, text="{}", **AGENT)
        late = requested | {"admitted_at": "2026-11-14T00:00:00.123Z"}  # three fraction digits
        v2, V, T = (
            {"version": 2},
            {"lifecycle": "VERIFIED"},
            {"text": "{}"},
        )  # PENDING never VERIFIED
        before = log.read("SELECT count(*) FROM event")[0][0]
        reads = recipe_key(MOMENTUM, 1, {}, ["no-such-recipe"])
        mine = {"selection_id": "s2", "recipe_key": "r", "seeds": [], "product_version": "0.2.8"}
        renamed = {"family": "metric", "name": "sharpe", "to": "ann_return"}
        for code, type_, subject, attrs, who in (
            (
                "ledger_unbalanced",
                "ledger.posted",
                ("book", "b1"),
                {"lines": [["fees", "USD", N]]},
                AGENT,
            ),
            (
                "recipe_key_mismatch",
                "recipe.defined",
                ("recipe", "not-its-digest"),
                {"method": "momentum", "method_version": 1, "params": {}},
                AGENT,
            ),
            (
                "recipe_upstream_unknown",
                "recipe.defined",
                ("recipe", reads),
                {
                    "method": "momentum",
                    "method_version": 1,
                    "params": {},
                    "upstream": ["no-such-recipe"],
                },
                AGENT,
            ),
            (
                "tag_unregistered",
                "tag.added",
                ("trial", "t1"),
                {"key": "no_such", "value": 1},
                AGENT,
            ),
            ("trial_workspace_disagrees", "trial.registered", ("trial", "t9"), mine, AGENT),
            (
                "trial_workspace_disagrees",
                "trial.registered",
                ("trial", "t9"),
                mine | {"task_id": "k1"},
                there,
            ),
            ("task_workspace_disagrees", "task.requested", ("task", "k9"), under_k1, there | v2),
            ("task_time_unreadable", "task.requested", ("task", "k9"), late, AGENT | v2),
            ("work_item_move_undeclared", "work_item.saved", ("task", "k1"), item | V, AGENT | T),
            ("task_workspace_disagrees", "task.requested", ("task", "k9"), under_g2, AGENT | v2),
            (
                "trial_goal_disagrees",
                "trial.registered",
                ("trial", "t9"),
                mine | {"goal_id": "g1"},
                there,
            ),
            (
                "definition_name_taken",
                "definition.renamed",
                ("definition", "metric/sharpe"),
                renamed,
                PERSON,
            ),
        ):
            with pytest.raises(DataError) as refused:
                log.record(type_, subject, attrs, **who)
            assert refused.value.failure.code == f"data_layer.{code}"
        assert log.read("SELECT count(*) FROM event")[0][0] == before


# Each unit of research work stores one parent; its workspace and goal are derived up it.
REQUESTED = {"type": "factor.run", "product_version": "0.2.8", "input_hash": "i", "plan_hash": "p"}
REQUESTED |= {"workflow_definition_hash": "d", "admitted_at": "2026-11-14T00:00:00Z"}
RECIPE = recipe_key(MOMENTUM, 1, {})

# kind, id, the columns it stores its parent in, those columns, its derived workspace and goal
UNITS = (
    ("task", "k-goal", "parent_task_id, goal_id, workspace_id", (None, "g1", None), "w1", "g1"),
    ("task", "k-free", "parent_task_id, goal_id, workspace_id", (None, None, "w1"), "w1", None),
    ("task", "k-sub", "parent_task_id, goal_id, workspace_id", ("k-goal", None, None), "w1", "g1"),
    ("hypothesis", "h1", "goal_id, workspace_id", ("g1", None), "w1", "g1"),
    ("trial", "t-task", "task_id, hypothesis_id", ("k-sub", "h1"), "w1", "g1"),
    ("trial", "t-sub", "task_id, hypothesis_id", ("k-sub", None), "w1", "g1"),
    ("trial", "t-free", "task_id, hypothesis_id", (None, None), "w1", None),
)
SCOPES = {
    "task": "SELECT workspace_id, goal_id FROM task_workspace JOIN task_scope USING (task_id)"
    " WHERE task_id = ?",
    "hypothesis": "SELECT workspace_id, goal_id FROM hypothesis_scope WHERE hypothesis_id = ?",
    "trial": "SELECT workspace_id, goal_id FROM trial_scope WHERE trial_id = ?",
}


@pytest.fixture(scope="module")
def units(tmp_path_factory):
    """A log holding every kind of unit under each parent it may have: tasks under a goal, under
    no goal and under a task, a hypothesis, trials under a task with and without a hypothesis and
    one under none, and two reviews of the first trial."""
    root = tmp_path_factory.mktemp("units")
    create_data_root(root)
    agent = ("agent", "claude")
    with HostLog.open(root, now_us=T0) as log:
        for workspace in ("w1", "w2"):
            log.record("workspace.created", ("workspace", workspace), {"name": workspace})
        for goal, workspace in (("g1", "w1"), ("g2", "w2")):
            log.record(
                "goal.set", ("goal", goal), {"title": goal}, **PERSON | {"workspace": workspace}
            )
        w1 = {"workspace": "w1", "actor": agent}
        for task, parent in (("k-goal", {"goal_id": "g1"}), ("k-free", {})):
            log.record("task.requested", ("task", task), REQUESTED | parent, version=2, **w1)
        sub = REQUESTED | {"parent_task_id": "k-goal"}
        log.record("task.requested", ("task", "k-sub"), sub, version=2, **w1)
        log.record("hypothesis.stated", ("hypothesis", "h1"), {"goal_id": "g1"}, text="H", **w1)
        window = {"cutoff_us": 1, "universe": "sp500"}
        log.record("selection.made", ("selection", "s1"), window, **w1)
        _define(log, "method", "momentum", text="Past return over a lookback.")
        defined = {"method": "momentum", "method_version": 1, "params": {}}
        log.record("recipe.defined", ("recipe", RECIPE), defined, **w1)
        trial = {"selection_id": "s1", "recipe_key": RECIPE, "seeds": [], "product_version": "x"}
        tested = trial | {"task_id": "k-sub", "hypothesis_id": "h1"}
        log.record("trial.registered", ("trial", "t-task"), tested, **w1)
        log.record("trial.registered", ("trial", "t-sub"), trial | {"task_id": "k-sub"}, **w1)
        log.record("trial.registered", ("trial", "t-free"), trial, **w1)
        for _ in range(2):  # the first review's note is the experience, kept by the second
            log.record("review.completed", ("trial", "t-task"), {"verdict": "ok"}, text="R", **w1)
        log.flush()
    return root / "machine" / "log.sqlite"


@pytest.mark.proves("data_layer.each_work_unit_stores_one_parent")
@pytest.mark.parametrize(
    ("kind", "unit", "columns", "stored", "workspace", "goal"), UNITS, ids=[u[1] for u in UNITS]
)
def test_a_unit_stores_one_parent_and_reaches_its_chain(
    units, kind, unit, columns, stored, workspace, goal
):
    """Each unit stores exactly one parent; its workspace and goal are joined up from it."""
    with closing(sqlite3.connect(units)) as con:
        sql = f"SELECT {columns} FROM {kind} WHERE {kind}_id = ?"
        assert con.execute(sql, (unit,)).fetchone() == stored
        assert con.execute(SCOPES[kind], (unit,)).fetchone() == (workspace, goal)


@pytest.mark.proves("data_layer.each_work_unit_stores_one_parent")
def test_a_review_sets_the_experience_and_the_history_replays_equal(units, tmp_path):
    """The first review's note is its subject's experience; the log replays ALL EQUAL."""
    with closing(sqlite3.connect(units)) as con:
        notes = con.execute(
            "SELECT note_id FROM research_note WHERE subject_id = 't-task' ORDER BY seq"
        ).fetchall()
        experience = con.execute("SELECT experience_id FROM trial WHERE trial_id = 't-task'")
        assert experience.fetchone() == notes[0]
        with closing(replay(con, tmp_path / "replayed.sqlite")) as again:
            assert differences(con, again) == []
