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

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer import create_data_root, experiences
from alphalattice.foundation.data_layer import project as projections
from alphalattice.foundation.data_layer.chain import CHAIN_VERSION
from alphalattice.foundation.data_layer.log import HostLog, envelope, replay, uuid7, verify_whole
from alphalattice.foundation.data_layer.project import (
    PROJECTED,
    differences,
    lineage,
    recipe_key,
)
from alphalattice.foundation.data_layer.reading import reading_session
from alphalattice.foundation.data_layer.root import seed_rows
from alphalattice.foundation.data_layer.store import StorePort, feature_table

TEST_SOURCE = "src/alphalattice/foundation/data_layer/project.py"
T0 = 1_794_700_000_000_000  # 2026-11-14, UTC
AGENT = {"workspace": "w1", "actor": ("agent", "claude"), "node": "research"}
PERSON = {"workspace": "w1", "actor": ("person", "me")}
N = 10**9  # ledger quantities are nano-units


def _identity(log, family, name):
    with log.reading() as con:
        found = lineage(con, family, name)
    assert found is not None
    return found


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


def _trial(log, trial, recipe, sharpe, storage, selection="s1"):
    """One trial from registration to its headline result; returns its result artifact."""
    registered = log.record(
        "trial.registered",
        ("trial", trial),
        {
            "goal_id": "g1",
            "hypothesis_id": "h1",
            "selection_id": selection,
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
    artifact = uuid7()
    log.record(
        "artifact.produced",
        ("artifact", artifact),
        {
            "type": "backtest",
            "type_version": 1,
            "meaning": "Does 12-1 momentum earn a premium on the S&P 500, 2015 to 2024?",
            "recipe_key": recipe,
            "selection_id": selection,
            "trial_id": trial,
            "storage": storage,
            "identity_kind": "recipe",
        },
        span=span,
        **AGENT,
    )
    measures = [["sharpe", 1, {}, sharpe], ["max_drawdown", 1, {}, 0.2]]
    measures.append(["ic_mean", 1, {"horizon_days": 21}, 0.03])
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
        started_us=T0,
        **AGENT,
    )
    return artifact


def _research(log):
    """A goal, a hypothesis, a selection of a registered universe, two recipes and three
    trials."""
    log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
    log.record("goal.set", ("goal", "g1"), {"title": "Find a momentum signal"}, **PERSON)
    claim = "12-1 momentum predicts next-month returns."
    log.record("hypothesis.stated", ("hypothesis", "h1"), {"goal_id": "g1"}, text=claim, **AGENT)
    _define(log, "universe", "sp500", text="The S&P 500's members.", **AGENT)
    window = {"cutoff_us": 1_767_225_600_000_000, "universe": "sp500"}
    window |= {"valid_from": "2015-01-01", "valid_to": "2024-12-31"}
    log.record("selection.made", ("selection", "s1"), window, **AGENT)
    _define(log, "method", "momentum", text="Past return over a lookback.", **AGENT)
    _define(log, "method", "realized_outcome", text="What a signal earned after its cutoff.")
    momentum = _identity(log, "method", "momentum")
    keys = {}
    for lookback in (12, 6):
        keys[lookback] = recipe_key(momentum, 1, {"lookback": lookback})
        attrs = {"method": "momentum", "method_version": 1, "params": {"lookback": lookback}}
        attrs["recipe_key"] = keys[lookback]
        log.record("recipe.defined", ("recipe", uuid7()), attrs, **AGENT)
    trials = (("t1", 12, 0.9, "table"), ("t2", 12, 1.3, "none"), ("t3", 6, 0.4, "table"))
    made = {t: _trial(log, t, keys[lookback], s, storage) for t, lookback, s, storage in trials}
    note = experiences.note_text(claim, "12-1 momentum", "It paid in 2015-2024.", ["Costs matter."])
    experiences.record_note(
        log, "trial", "t1", "success", note, workspace="w1", actor=AGENT["actor"]
    )
    return made


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
        _identity(log, "method", "realized_outcome"),
        1,
        {"horizon": "63d"},
        [recipe_key(_identity(log, "method", "momentum"), 1, {"lookback": 12})],
    )
    attrs = {"method": "realized_outcome", "method_version": 1, "params": {"horizon": "63d"}}
    attrs |= {"recipe_key": horizon, "reads": [artifacts["t1"]]}  # the signal it scores
    log.record("recipe.defined", ("recipe", uuid7()), attrs)
    outcome = uuid7()
    meaning = "What the t1 signal earned over the 63 sessions after its cutoff."
    produced = {"type": "outcome", "type_version": 1, "identity_kind": "recipe"}
    produced |= {"recipe_key": horizon}
    produced |= {"trial_id": "t1", "storage": "none", "meaning": meaning}
    log.record("artifact.produced", ("artifact", outcome), produced)
    measures = [["sharpe", 1, {"horizon_days": 63}, 0.7]]
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
    log.record("task", ("task", "k1"), phase="completed", span=span, started_us=T0, **AGENT)
    log.record("book.activated", ("book", "b1"), {"mode": "paper"}, **PERSON)
    lines = [["book", "listing", "AAPL", 10 * N], ["market", "listing", "AAPL", -10 * N]]
    lines += [["book", "cash", "USD", -1500 * N], ["market", "cash", "USD", 1500 * N]]
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
    review = requested | {"type": "review", "input_hash": "r"}  # a review of two trials
    log.record("task.requested", ("task", "k-review"), review, **AGENT)
    for phase in ("started", "completed"):
        log.record(
            "task", ("task", "k-review"), phase=phase, span=f"{2:016x}", started_us=T0, **AGENT
        )
    note = experiences.note_text("Momentum pays", "Both lookbacks", "The 12-1 one paid.", [])
    experiences.record_note(
        log,
        "task",
        "k-review",
        "success",
        note,
        workspace="w1",
        actor=AGENT["actor"],
        trials=["t1", "t2"],
    )
    experiences.render_pending(
        log, root / "blobs"
    )  # the reviews' experiences, off the agent's path
    seq, head = log.read("SELECT seq, head FROM chain_head")[0]
    mark = {"kind": "backup", "seq": seq, "head": head, "chain_version": CHAIN_VERSION}
    log.record("chain.checkpointed", attrs=mark)


def _knowledge(log):
    """A reference revision the retrieval store publishes, and the snapshot that freezes it."""
    from alphalattice.foundation.data_layer.knowledge import KnowledgeStore, SourceOrigin

    store, metadata = KnowledgeStore(log), {"document": {"document_id": "d1"}, "revision": 1}
    origin = SourceOrigin("d1", "1", "reference", T0)
    store.publish_revision(workspace_id="w1", metadata=metadata, content="A note.", origin=origin)
    store.publish_snapshot(
        workspace_id="w1", metadata={"snapshot_id": "s1", "revisions": [metadata]}
    )


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
    and that feature set renamed; momentum defined again under its new name is the same recipe,
    so the log refuses it as a split."""
    spec = {"method": "momentum", "method_version": 1, "factors": ["mom_12_1"]}
    _define(log, "feature_set", "momentum_set", text="Momentum factors.", spec=spec, **AGENT)
    _cached(root, log, recipe_key(_identity(log, "method", "momentum"), 1, {"lookback": 12}))
    for family, name, to, who in (
        ("method", "momentum", "price_momentum", AGENT),
        ("metric", "sharpe", "sharpe_ratio", PERSON),
        ("feature_set", "momentum_set", "momentum_factors", AGENT),
    ):
        renamed = {"family": family, "name": name, "to": to}
        log.record("definition.renamed", ("definition", f"{family}/{name}"), renamed, **who)
    again = {"method": "price_momentum", "method_version": 1, "params": {"lookback": 12}}
    again["recipe_key"] = recipe_key(_identity(log, "method", "momentum"), 1, {"lookback": 12})
    with pytest.raises(DataError) as split:
        log.record("recipe.defined", ("recipe", uuid7()), again, **AGENT)
    assert split.value.failure.code == "data_layer.recipe_split"


def _cached(root, log, key):
    """The cache entry and derived partition a first computation leaves, as their writers keep them:
    the entry keyed by the recipe, the partition's table named by the feature set's lineage."""
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con, con:
        entry = (
            "INSERT INTO cache_entry VALUES (?, 'feature_set', '2024', 'rows', 1, 8, 1, ?, ?, NULL)"
        )
        con.execute(entry, (key, T0, T0))
    port = StorePort.open(root, log)
    try:
        table = f"lake.derived.{feature_table(_identity(log, 'feature_set', 'momentum_set'))}"
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
        _knowledge(log)
    return root, artifacts


def _log(root):
    """A connection to `root`'s log."""
    con = sqlite3.connect(root / "machine" / "log.sqlite")
    con.row_factory = sqlite3.Row
    return con


@pytest.mark.proves("data_layer.replay_rebuilds_every_projection")
@pytest.mark.parametrize(
    "historical_seed", [None, 1, 2], ids=["minted", "legacy-ledger-only", "pre-pin-minted"]
)
def test_the_reference_history_replays_all_equal(request, tmp_path, historical_seed):
    """Every projected table, the chain head included, is rebuilt from the events alone."""
    if historical_seed is not None:
        _replay_historical_ledger(tmp_path, historical_seed)
        return
    root, _ = request.getfixturevalue("history")
    with closing(_log(root)) as source, closing(replay(source, tmp_path / "r.sqlite")) as target:
        assert differences(source, target) == []
        assert verify_whole(source) and verify_whole(target)
        births = source.execute(
            "SELECT n.family, n.name, n.definition_id, l.registered_seq FROM definition_name n"
            " JOIN definition_lineage l USING (definition_id) WHERE n.named_seq = l.registered_seq"
        ).fetchall()
        for _, _, identity, _ in births:
            assert len(identity) == 36 and uuid.UUID(identity).version == 7, identity
        seeded = source.execute("SELECT text FROM event_text WHERE seq = 1").fetchone()
        seed_ids = {tuple(row) for row in json.loads(seeded[0])}
        for family, name, identity, seq in births:
            if seq == 1:
                assert (family, name, identity) in seed_ids
            else:
                attrs = source.execute("SELECT attrs FROM event WHERE seq = ?", (seq,)).fetchone()
                assert [family, name, identity] in json.loads(attrs[0])["definition_ids"]
        for family, first, current in (
            ("method", "momentum", "price_momentum"),
            ("metric", "sharpe", "sharpe_ratio"),
            ("feature_set", "momentum_set", "momentum_factors"),
        ):
            identity = lineage(source, family, first)
            assert identity is not None and lineage(source, family, current) == identity
            assert lineage(target, family, current) == identity
        empty = [
            t
            for t in PROJECTED
            if t != "status_refusal" and not source.execute(f"SELECT 1 FROM {t}").fetchone()
        ]
    assert empty == []  # the history reaches every projected table


def _replay_historical_ledger(tmp_path, seed_version):
    """Restore a pre-pin registry and its keys from the original ledger tables."""
    count = 105 if seed_version == 1 else 106
    seed_text = None
    expected = "blake3:156a8bf14d8a9a1afd9ba8f0321e6b17aeae697f50e6aee6e1192e2ac03d99b6"
    key = {"family": "method", "name": "old-method", "version": 1}
    attrs = key.copy()
    if seed_version == 2:
        seed_text = json.dumps(
            [
                (family, name, uuid7())
                for family, name, version, *_ in seed_rows(historical_count=count)
                if version == 1
            ]
        )
        expected = uuid7()
        attrs["definition_ids"] = [("method", "old-method", expected)]
    seed = envelope("registry.seeded", attrs={"count": count}, text=seed_text, version=seed_version)
    text = json.dumps(key | {"spec": {}, "status": "standard", "text": "Synthetic method."})
    event = envelope(
        "definition.registered", ("definition", "method/old-method"), attrs, text=text, version=1
    )
    columns = ", ".join(seed)
    marks = ", ".join("?" for _ in seed)
    with closing(sqlite3.connect(":memory:")) as source:
        source.executescript(
            f"CREATE TABLE event (seq INTEGER PRIMARY KEY, {columns});"
            "CREATE TABLE event_text (seq INTEGER PRIMARY KEY, text TEXT);"
            "CREATE TABLE artifact_body (artifact_id TEXT PRIMARY KEY, body BLOB);"
        )
        source.executemany(
            f"INSERT INTO event ({columns}) VALUES ({marks})",
            [tuple(seed.values()), tuple(event.values())],
        )
        if seed_text is not None:
            source.execute("INSERT INTO event_text VALUES (1, ?)", (seed_text,))
        source.execute("INSERT INTO event_text VALUES (2, ?)", (text,))
        try:
            rebuilt = replay(source, tmp_path / "replayed.sqlite")
        except sqlite3.DatabaseError as error:
            pytest.fail(f"ledger-only replay failed: {error}")
        with closing(rebuilt):
            assert lineage(rebuilt, "method", "old-method") == expected
            assert lineage(rebuilt, "event_type", "artifact.pinned") is None
            assert verify_whole(rebuilt)


@pytest.mark.proves("data_layer.replay_rebuilds_every_projection")
def test_the_projections_hold_the_research_memory(history):
    """Trials are found by tag; kept, never-stored and collected artifacts keep their measures."""
    root, artifacts = history
    with closing(_log(root)) as con:
        found = con.execute(
            "SELECT subject_id FROM tag WHERE key_id = ? AND value = 'AAPL' ORDER BY 1",
            (lineage(con, "tag", "instrument"),),
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
        momentum = lineage(con, "method", "momentum")
        sharpe = lineage(con, "metric", "sharpe")
        factors = lineage(con, "feature_set", "momentum_set")
        lineages = dict(con.execute("SELECT name, definition_id FROM definition_lineage"))
        assert (lineages["price_momentum"], lineages["sharpe_ratio"]) == (momentum, sharpe)
        assert lineages["momentum_factors"] == factors
        assert "momentum" not in lineages and "sharpe" not in lineages
        recipes = con.execute("SELECT recipe_key FROM recipe WHERE method_id = ?", (momentum,))
        keys = {recipe_key(momentum, 1, {"lookback": n}) for n in (12, 6)}
        assert {row[0] for row in recipes} == keys  # one approach under either name: one recipe
        held = "SELECT definition_id FROM definition WHERE family = ? AND name = ? AND version = 1"
        assert con.execute(held, ("metric", "sharpe")).fetchone()[0] == sharpe
        measured = "SELECT count(*) FROM measure WHERE metric_id = ? AND metric_version = 1"
        assert con.execute(measured, (sharpe,)).fetchone()[0] == 4
        refs = "SELECT ref_id, ref_version FROM definition_ref WHERE definition_id = ?"
        assert [tuple(r) for r in con.execute(refs, (factors,))] == [(momentum, 1)]
        shown = "SELECT DISTINCT metric FROM metric WHERE metric_id = ?"
        assert [tuple(r) for r in con.execute(shown, (sharpe,))] == [("sharpe_ratio",)]
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


def _terminal_goal_projection_refusal(tmp_path):
    root = tmp_path / "data"
    create_data_root(root)
    goal = ("goal", "g1")
    with HostLog.open(root, now_us=T0) as log:
        log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
        log.record("goal.set", goal, {"title": "Read a result"}, **PERSON)
        closed = log.record("goal.closed", goal, {"state": "done"}, **PERSON)
        refused = log.record("goal.closed", goal, {"state": "dropped"}, **PERSON)
        state = "SELECT state, closed_seq FROM goal WHERE goal_id = 'g1'"
        assert tuple(log.read(state)[0]) == ("done", closed)
        repeated = log.record("goal.closed", goal, {"state": "done"}, **PERSON)
        assert tuple(log.read(state)[0]) == ("done", closed)
        assert log.read("SELECT seq FROM event WHERE seq = ?", (refused,))
        counts = log.read("SELECT seq, name FROM status_refusal ORDER BY seq")
        assert [tuple(row) for row in counts] == [(refused, "goal.state")]
        assert log.read("SELECT seq FROM event WHERE seq = ?", (repeated,))


def _task_projection_refusals(tmp_path):
    root = tmp_path / "data"
    create_data_root(root)
    task = ("task", "k1")
    now = "2026-11-14 00:00:01.000000"
    with HostLog.open(root, now_us=T0) as log:
        log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
        attrs = {
            "type": "evidence.prepare",
            "product_version": "0.2.8",
            "input_hash": "i",
            "plan_hash": "p",
            "workflow_definition_hash": "f",
            "admitted_at": now,
        }
        log.record("task.requested", task, attrs, text="{}", **AGENT)

        def save(lifecycle):
            return log.record(
                "task.saved",
                task,
                {"lifecycle": lifecycle, "updated_at": now},
                text=json.dumps({"lifecycle": lifecycle}),
                **AGENT,
            )

        def started():
            return log.record("task", task, phase="started", span=uuid.uuid4().hex[:16], **AGENT)

        state = (
            "SELECT state, state_seq, lifecycle, lifecycle_seq, attempts, started_seq, ended_seq"
            " FROM task WHERE task_id = ?"
        )
        first = save("RUNNING")
        started()
        started()
        assert tuple(log.read(state, (task[1],))[0]) == (
            "running",
            first,
            "RUNNING",
            first,
            1,
            first,
            None,
        )
        wait = {
            "task_id": task[1],
            "kind": "event",
            "event_type": "decision.made",
            "purpose": "review",
        }
        waiting = log.record("wait.started", ("wait", "q1"), wait, **AGENT)
        review = save("REVIEW_PENDING")
        refused = log.record("wait.resolved", ("wait", "q1"), {}, **AGENT)
        assert tuple(log.read(state, (task[1],))[0]) == (
            "waiting",
            waiting,
            "REVIEW_PENDING",
            review,
            1,
            first,
            None,
        )
        assert log.read("SELECT resolved_seq FROM wait WHERE wait_id = 'q1'")[0][0] == refused
        completed = save("SUCCEEDED")
        save("SUCCEEDED")
        late_start = started()
        late_save = save("RUNNING")
        assert tuple(log.read(state, (task[1],))[0]) == (
            "completed",
            completed,
            "SUCCEEDED",
            completed,
            1,
            first,
            completed,
        )
        counts = log.read("SELECT seq, name FROM status_refusal ORDER BY seq")
        assert [tuple(row) for row in counts] == [
            (refused, "task.state"),
            (late_start, "task.state"),
            (late_save, "task.lifecycle"),
        ]
        assert [
            log.read("SELECT count(*) FROM event WHERE seq = ?", (seq,))[0][0]
            for seq in (refused, late_start, late_save)
        ] == [1, 1, 1]
        task = ("task", "k2")
        log.record("task.requested", task, attrs, text="{}", **AGENT)
        blocked = save("BLOCKED")
        recovered = save("RECOVERY_REQUIRED")
        assert tuple(log.read(state, (task[1],))[0]) == (
            "waiting",
            blocked,
            "RECOVERY_REQUIRED",
            recovered,
            0,
            None,
            None,
        )
        started_again = save("RUNNING")
        assert tuple(log.read(state, (task[1],))[0]) == (
            "running",
            started_again,
            "RUNNING",
            started_again,
            1,
            started_again,
            None,
        )


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


def _projection_refusals_roll_back(tmp_path):
    """An ordinary projection refusal rolls back the event and its effects."""
    copy = tmp_path / "data"
    create_data_root(copy)
    with HostLog.open(copy, now_us=T0) as log:
        for workspace in ("w1", "w2"):
            log.record("workspace.created", ("workspace", workspace), {"name": workspace})
        log.record("book.activated", ("book", "b1"), {"mode": "paper"}, **PERSON)
        _define(log, "method", "momentum", text="Past return over a lookback.", **AGENT)
        window = {"cutoff_us": 1, "valid_from": "2015-01-01"}
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
        bare = {"method": "momentum", "method_version": 1, "params": {}}
        bare["recipe_key"] = recipe_key(_identity(log, "method", "momentum"), 1, {})
        log.record("recipe.defined", ("recipe", uuid7()), bare, **AGENT)
        mine = {"selection_id": "s2", "recipe_key": "r", "seeds": [], "product_version": "0.2.8"}
        theirs = mine | {"recipe_key": bare["recipe_key"], "hypothesis_id": "h2", "goal_id": "g2"}
        log.record("trial.registered", ("trial", "t2"), theirs, **there)
        before = log.read("SELECT count(*) FROM event")[0][0]
        bet = {"target": {"what": "up"}, "horizon_us": 1, "confidence": 0.5, "trial_id": "t2"}
        renamed = {"family": "metric", "name": "sharpe", "to": "ann_return"}
        for code, type_, subject, attrs, who in (
            (
                "ledger_unbalanced",
                "ledger.posted",
                ("book", "b1"),
                {"lines": [["fees", "cash", "USD", N]]},
                AGENT,
            ),
            (
                "recipe_key_mismatch",
                "recipe.defined",
                ("recipe", uuid7()),
                bare | {"recipe_key": "not-its-digest"},
                AGENT,
            ),
            (
                "recipe_upstream_given",
                "recipe.defined",
                ("recipe", uuid7()),
                bare | {"upstream": [bare["recipe_key"]]},
                AGENT,
            ),
            (
                "recipe_reads_unknown",
                "recipe.defined",
                ("recipe", uuid7()),
                bare | {"reads": ["no-such-artifact"]},
                AGENT,
            ),
            ("selection_split", "selection.made", ("selection", "s3"), window, there),
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
                "prediction_workspace_disagrees",
                "prediction.registered",
                ("prediction", "p9"),
                bet,
                AGENT | {"text": "It closes higher."},
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


@pytest.mark.proves("data_layer.replay_rebuilds_every_projection")
@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(_projection_refusals_roll_back, id="ordinary-refusals"),
        pytest.param(_terminal_goal_projection_refusal, id="terminal-goal"),
        pytest.param(_task_projection_refusals, id="task-status"),
    ],
)
def test_a_projection_refusal_follows_its_log_policy(tmp_path, scenario):
    """The Host transaction rolls ordinary failures back and counts status refusals."""
    scenario(tmp_path)


# Each unit of research work stores one parent; its workspace and goal are derived up it.
REQUESTED = {"type": "factor.run", "product_version": "0.2.8", "input_hash": "i", "plan_hash": "p"}
REQUESTED |= {"workflow_definition_hash": "d", "admitted_at": "2026-11-14T00:00:00Z"}

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
        log.record("selection.made", ("selection", "s1"), {"cutoff_us": 1}, **w1)
        _define(log, "method", "momentum", text="Past return over a lookback.")
        recipe = recipe_key(_identity(log, "method", "momentum"), 1, {})
        defined = {"method": "momentum", "method_version": 1, "params": {}, "recipe_key": recipe}
        log.record("recipe.defined", ("recipe", uuid7()), defined, **w1)
        trial = {"selection_id": "s1", "recipe_key": recipe, "seeds": [], "product_version": "x"}
        tested = trial | {"task_id": "k-sub", "hypothesis_id": "h1"}
        log.record("trial.registered", ("trial", "t-task"), tested, **w1)
        log.record("trial.registered", ("trial", "t-sub"), trial | {"task_id": "k-sub"}, **w1)
        log.record("trial.registered", ("trial", "t-free"), trial, **w1)
        for phase in ("started", "completed"):
            log.record(
                "trial", ("trial", "t-task"), phase=phase, span="1" * 16, started_us=T0, **w1
            )
        for _ in range(2):  # the first review closes the trial; the second keeps that
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
def test_a_review_closes_its_trial_and_the_history_replays_equal(units, tmp_path):
    """The first review's note closes its trial and a later one keeps that; the log replays ALL
    EQUAL."""
    with closing(sqlite3.connect(units)) as con:
        notes = con.execute(
            "SELECT seq FROM research_note WHERE subject_id = 't-task' ORDER BY seq"
        ).fetchall()
        reviewed = con.execute("SELECT reviewed_seq FROM trial WHERE trial_id = 't-task'")
        assert reviewed.fetchone() == notes[0]
        with closing(replay(con, tmp_path / "replayed.sqlite")) as again:
            assert differences(con, again) == []


# One row per research-record defect, each on a log of its own.
W2 = AGENT | {"workspace": "w2"}
WINDOW = {"cutoff_us": 1, "universe": "sp500"}
CONTENT = {"type": "backtest", "type_version": 1, "meaning": "m", "storage": "none"}
CONTENT |= {"identity_kind": "content", "content_digest": "blake3:same"}


def _code(code, write):
    with pytest.raises(DataError) as refused:
        write()
    assert refused.value.failure.code == f"data_layer.{code}"


def _made(log, attrs):
    log.record("artifact.produced", ("artifact", made := uuid7()), attrs, **AGENT)
    return made


def _recipe(log, version, params, reads=()):
    with log.reading() as con:
        key = recipe_key(
            _identity(log, "method", "momentum"), version, params, projections.upstream(con, reads)
        )
    attrs = {"method": "momentum", "method_version": version, "params": params, "recipe_key": key}
    log.record("recipe.defined", ("recipe", uuid7()), attrs | {"reads": list(reads)}, **AGENT)
    return key


def _selection_per_workspace(log, read, root):
    for who in (AGENT, W2):
        log.record("selection.made", ("selection", uuid7()), WINDOW, **who)
    assert read("SELECT count(DISTINCT workspace_id) FROM selection") == [(2,)]


def _artifact_per_event(log, read, root):
    _made(log, CONTENT), _made(log, CONTENT)
    assert read("SELECT count(*), count(DISTINCT content_digest) FROM artifact") == [(2, 1)]


def _recipe_typed_numbers(log, read, root):
    assert _recipe(log, 1, {"lookback": 5}).startswith("blake3:")  # 5.0 is 5: one recipe
    _code("recipe_split", lambda: _recipe(log, 1, {"lookback": 5.0}))


def _no_version_default(log, read, root):
    bare = {k: v for k, v in CONTENT.items() if k != "type_version"}
    _code("event_attrs_missing", lambda: _made(log, bare))
    log.record("selection.made", ("selection", uuid7()), WINDOW, **AGENT)  # names no version
    sql = "SELECT e.type_version, c.version FROM event e JOIN definition_current c"
    sql += " ON c.family = 'event_type' AND c.name = e.type WHERE e.type = 'selection.made'"
    assert [(given, given) for given, _ in read(sql)] == read(sql)


def _dims_canonical(log, read, root):
    made = _made(log, CONTENT)
    dims = [["sharpe", 1, {"side": "long", "h": 21.0}, 0.1], ["sharpe", 1, {}, 0.2]]
    log.record("result.recorded", ("artifact", made), {"measures": dims}, **AGENT)
    assert read("SELECT dims FROM measure ORDER BY dims") == [('{"h":21,"side":"long"}',), ("{}",)]


def _tag_without_version(log, read, root):
    tag = {"key": "theme", "value": "momentum"}
    log.record("tag.added", ("trial", "t1"), tag, **AGENT)
    _define(log, "tag", "theme", 2, text="Again.", relation="not_comparable", **AGENT)
    log.record("tag.added", ("trial", "t1"), tag, **AGENT)
    assert read("SELECT tag_version FROM tag") == [(1,)]
    log.record("tag.removed", ("trial", "t1"), tag, **AGENT)
    assert read("SELECT count(*) FROM tag") == [(0,)]


def _ledger_asset(log, read, root):
    log.record("book.activated", ("book", "b1"), {"mode": "paper"}, **PERSON)
    lines = [["book", "listing", "L1", N], ["market", "listing", "L1", -N]]
    lines += [["book", "cash", "USD", -N], ["market", "cash", "USD", N]]
    log.record("ledger.posted", ("book", "b1"), {"lines": lines}, **AGENT)
    assets = read("SELECT DISTINCT asset_kind, asset_id FROM ledger_entry ORDER BY 1")
    assert assets == [("cash", "USD"), ("listing", "L1")]
    old = {"lines": [["book", "cash", "CASH:USD", N], ["market", "cash", "CASH:USD", -N]]}
    with pytest.raises(sqlite3.IntegrityError, match="asset_id GLOB"):
        log.record("ledger.posted", ("book", "b1"), old, **AGENT)


def _names_give_way_to_lineages(log, read, root):
    log.record("selection.made", ("selection", uuid7()), WINDOW, **AGENT)
    moved = {"family": "universe", "name": "sp500", "to": "spx"}
    log.record("definition.renamed", ("definition", "universe/sp500"), moved, **AGENT)
    assert read("SELECT universe_id FROM selection") == [(_identity(log, "universe", "sp500"),)]
    log.record("task.requested", ("task", uuid7()), REQUESTED, version=2, **AGENT)
    assert "task_kind" not in {row[1] for row in read("PRAGMA table_info(task)")}
    assert read("SELECT task_kind FROM workspace_task") == [("factor.run",)]


def _recorded_at(log, read, root):
    log.record("goal.set", ("goal", "g1"), {"title": "t"}, **PERSON)
    log.record("conclusion.drawn", ("goal", "g1"), {}, text="Yes.", occurred_us=T0, **AGENT)
    [(recorded,)] = read("SELECT max(recorded_us) FROM event")
    log.close()
    with closing(reading_session(root)) as reading:
        noted = reading.execute("SELECT epoch_us(recorded_at) FROM goal_note").fetchall()
    assert noted == [(recorded,)] and recorded > T0


def _step_unstarted(log, read, root):
    _code("step_unstarted", lambda: log.record("trial", ("trial", "t1"), phase="completed"))


def _upstream_from_reads(log, read, root):
    source = _recipe(log, 1, {"lookback": 12})
    made = _made(log, CONTENT | {"identity_kind": "recipe", "recipe_key": source})
    derived = _recipe(log, 1, {"lookback": 6}, [made])
    assert read("SELECT recipe_key, upstream_key FROM recipe_upstream") == [(derived, source)]
    given = {"method": "momentum", "method_version": 1, "params": {}, "upstream": [source]}
    given["recipe_key"] = recipe_key(_identity(log, "method", "momentum"), 1, {}, [source])
    _code("recipe_upstream_given", lambda: log.record("recipe.defined", ("recipe", "r9"), given))


def _prediction_one_parent(log, read, root):
    bet = {"target": {"what": "up"}, "horizon_us": 1, "confidence": 0.5}
    log.record("prediction.registered", ("prediction", "p1"), bet, text="Up.", **AGENT)
    assert read("SELECT workspace_id, trial_id FROM prediction") == [("w1", None)]
    score = {"score": 1.0, "metric": "brier", "metric_version": 9}  # no such definition version
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        log.record("prediction.scored", ("prediction", "p1"), score, **AGENT)


DEFECTS = [_selection_per_workspace, _artifact_per_event, _recipe_typed_numbers]
DEFECTS += [_no_version_default, _dims_canonical, _tag_without_version]
DEFECTS += [_ledger_asset, _names_give_way_to_lineages, _recorded_at, _step_unstarted]
DEFECTS += [_upstream_from_reads, _prediction_one_parent]


@pytest.mark.proves("data_layer.research_records_are_named_by_ids")
@pytest.mark.parametrize("defect", DEFECTS, ids=lambda d: d.__name__.strip("_").replace("_", "-"))
def test_each_research_record_is_named_by_an_id(tmp_path, defect):
    """Each defect's facts, written through the Host's log, read back fixed and never split."""
    root = tmp_path / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        for workspace in ("w1", "w2"):
            log.record("workspace.created", ("workspace", workspace), {"name": workspace})
        _define(log, "universe", "sp500", text="The S&P 500's members.", **AGENT)
        _define(log, "method", "momentum", text="Past return over a lookback.", **AGENT)
        defect(log, lambda sql: [tuple(row) for row in log.read(sql)], root)
