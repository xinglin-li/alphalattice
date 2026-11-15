"""Every state table of log.sqlite is a projection of its events: the 0.2.8 reference history,
written through the Host's one write path, replays ALL EQUAL into an empty log."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer import project as projections
from alphalattice.kernel.data_layer.chain import CHAIN_VERSION, digest
from alphalattice.kernel.data_layer.log import HostLog, replay, verify_whole
from alphalattice.kernel.data_layer.project import PROJECTED, differences, recipe_key

TEST_SOURCE = "src/alphalattice/kernel/data_layer/project.py"
T0 = 1_794_700_000_000_000  # 2026-11-14, UTC
AGENT = {"workspace": "w1", "actor": ("agent", "claude"), "node": "research"}
PERSON = {"workspace": "w1", "actor": ("person", "me")}
N = 10**9  # ledger quantities are nano-units


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
        keys[lookback] = recipe_key("momentum", 1, {"lookback": lookback})
        attrs = {"method": "momentum", "method_version": 1, "params": {"lookback": lookback}}
        log.record("recipe.defined", ("recipe", keys[lookback]), attrs, **AGENT)
    trials = (("t1", 12, 0.9, "table"), ("t2", 12, 1.3, "none"), ("t3", 6, 0.4, "table"))
    return {t: _trial(log, t, keys[lookback], s, storage) for t, lookback, s, storage in trials}


def _aftermath(log, artifacts):
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
    horizon = recipe_key("realized_outcome", 1, {"horizon": "63d"})
    attrs = {"method": "realized_outcome", "method_version": 1, "params": {"horizon": "63d"}}
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
    requested = {"type": "evidence.prepare", "product_version": "0.2.8"}
    log.record("task.requested", ("task", "k1"), requested, **AGENT)
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
    seq, head = log.read("SELECT seq, head FROM chain_head")[0]
    mark = {"kind": "backup", "seq": seq, "head": head, "chain_version": CHAIN_VERSION}
    log.record("chain.checkpointed", attrs=mark)


@pytest.fixture(scope="module")
def history(tmp_path_factory):
    """A data root whose log holds the reference history, and its result artifacts."""
    root = tmp_path_factory.mktemp("history") / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        artifacts = _research(log)
        _aftermath(log, artifacts)
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
            "SELECT subject_id FROM tag WHERE key = 'instrument' AND value = 'AAPL' ORDER BY 1"
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
    """An unbalanced posting, a recipe under another key and an untagged key leave no trace."""
    copy = tmp_path / "data"
    create_data_root(copy)
    with HostLog.open(copy, now_us=T0) as log:
        log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
        log.record("book.activated", ("book", "b1"), {"mode": "paper"}, **PERSON)
        before = log.read("SELECT count(*) FROM event")[0][0]
        for code, type_, subject, attrs in (
            ("ledger_unbalanced", "ledger.posted", ("book", "b1"), {"lines": [["fees", "USD", N]]}),
            (
                "recipe_key_mismatch",
                "recipe.defined",
                ("recipe", "not-its-digest"),
                {"method": "momentum", "method_version": 1, "params": {}},
            ),
            ("tag_unregistered", "tag.added", ("trial", "t1"), {"key": "no_such", "value": 1}),
        ):
            with pytest.raises(DataError) as refused:
                log.record(type_, subject, attrs, **AGENT)
            assert refused.value.failure.code == f"data_layer.{code}"
        assert log.read("SELECT count(*) FROM event")[0][0] == before
