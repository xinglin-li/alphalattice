"""A data root's databases hold their own rules: the reference check.py, on the product's DDL."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer.project import definition_id, register
from alphalattice.kernel.data_layer.reading import reading_session

TEST_SOURCE = "src/alphalattice/kernel/data_layer/root.py"
N = 10**9
T0 = 1_767_225_600_000_000
SPAN = "a" * 16
TRACE = "0" * 32
EVENT = {"event_id": "x", "type": "tag.added", "type_version": 1, "occurred_us": 1}
EVENT |= {"recorded_us": 1, "actor_kind": "host", "actor_id": "h", "node": "n", "trace_id": TRACE}


def _insert(table, **values):
    """An INSERT of `values` into `table` and its arguments."""
    names = ", ".join(values)
    return f"INSERT INTO {table} ({names}) VALUES ({', '.join('?' * len(values))})", (
        *values.values(),
    )


def _row(**values):
    """An INSERT of one event with `values` over the minimal one."""
    return _insert("event", **EVENT | values)


def _event(con, type_, subject, *, phase=None, attrs=None, span=None, text=None):
    """Append one event as an agent in workspace w1 would, and its text when it has one."""
    at = T0 + con.execute("SELECT count(*) FROM event").fetchone()[0]
    values = {**EVENT, "event_id": str(uuid.uuid4()), "type": type_, "occurred_us": at}
    values |= {"recorded_us": at, "actor_kind": "agent", "actor_id": "claude", "node": "research"}
    values |= {"workspace_id": "w1", "subject_kind": subject[0], "subject_id": subject[1]}
    values |= {"trace_id": TRACE, "attrs": json.dumps(attrs or {})}
    if phase:
        values |= {"phase": phase, "span_id": span}
        values |= {"started_us": at - 2_000_000} if phase != "started" else {}
    if text:
        values["text_digest"] = "blake3:t"
    seq = con.execute(*_insert("event", **values)).lastrowid
    if text:
        con.execute("INSERT INTO event_text VALUES (?, ?)", (seq, text))
    return seq


METHOD, METRIC = definition_id("method", "momentum"), definition_id("metric", "sharpe")
BACKTEST = definition_id("artifact_type", "backtest")
MOMENTUM = (
    "standard",
    None,
    None,
    "{}",
    "Past return over a lookback.",
    None,
    None,
    "research",
    1,
    1,
)


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    """A new data root's log holding one of each record the refusals act on."""
    root = tmp_path_factory.mktemp("data-root")
    create_data_root(root)
    con = sqlite3.connect(root / "machine" / "log.sqlite", isolation_level=None)
    con.execute("PRAGMA foreign_keys = ON")  # log.sql's first pragma holds per connection
    first = _event(con, "workspace.created", ("workspace", "w1"), attrs={"name": "main"})
    _event(con, "task", ("task", "t1"), phase="started", span=SPAN)
    _event(con, "task", ("task", "t1"), phase="completed", span=SPAN)
    note = _event(con, "lesson.recorded", ("goal", "g0"), text="Check the calendar first.")
    con.execute("INSERT INTO workspace VALUES ('w1', 'main', NULL, 'active', 1, NULL)")
    made = _event(con, "selection.made", ("selection", "s1"), attrs={"cutoff_us": 1})
    selection = ("s1", "w1", T0, "sp500", "2015-01-01", "2024-12-31", "{}", made)
    con.execute(f"INSERT INTO selection VALUES ({', '.join('?' * 8)})", selection)
    register(con, "method", "momentum", 1, *MOMENTUM, None)
    recipe = _event(con, "recipe.defined", ("recipe", "r1"))
    con.execute("INSERT INTO recipe VALUES ('r1', ?, 1, '{}', ?)", (METHOD, recipe))
    registered = _event(con, "trial.registered", ("trial", "t1"))
    trial = {"trial_id": "t1", "workspace_id": "w1", "selection_id": "s1", "recipe_key": "r1"}
    trial |= {"seeds": "[7]", "product_version": "0.2.8", "state": "registered"}
    con.execute(*_insert("trial", **trial, registered_seq=registered))
    produced = _event(con, "artifact.produced", ("artifact", "a1"), attrs={"type": "backtest"})
    artifact = {"artifact_id": "a1", "identity_kind": "recipe", "type_id": BACKTEST}
    artifact |= {"type_version": 1, "meaning": "A fixture backtest.", "recipe_key": "r1"}
    artifact |= {"selection_id": "s1", "trial_id": "t1", "storage": "table"}
    con.execute(*_insert("artifact", **artifact, body_state="present", produced_seq=produced))
    result = _event(con, "result.recorded", ("artifact", "a1"))
    con.execute("INSERT INTO measure VALUES ('a1', ?, 1, '', 0.9, ?)", (METRIC, result))
    active = _event(con, "book.activated", ("book", "b1"))
    con.execute("INSERT INTO book VALUES ('b1', 'w1', 'paper', 'active', NULL, ?, NULL)", (active,))
    fill = _event(con, "ledger.posted", ("book", "b1"))
    con.executemany(
        "INSERT INTO ledger_entry VALUES (?, ?, 'b1', ?, ?, ?)",
        [(fill, 1, "book", "AAPL", 10 * N), (fill, 2, "market", "AAPL", -10 * N)],
    )
    yield con, {"first": first, "note": note, "registered": registered, "metric": METRIC}
    con.close()


SHARPE = {"definition_id": METRIC, "version": 2, "status": "standard"}
SHARPE |= {"relation": "convertible", "conversion": '{"offset": 0.0, "scale": 1.2035}'}
SHARPE |= {"spec": '{"unit": "ratio", "params": {"periods": 365}}', "text": "Annualized."}
SHARPE |= {"owner": "kernel", "registered_seq": 1, "status_seq": 1}
LATE_TRIAL = (
    "INSERT INTO trial (trial_id, workspace_id, selection_id, recipe_key, product_version, state,"
    " registered_seq, ended_seq) VALUES ('bad', 'w1', 's1', 'r1', '0.2.8', 'completed',"
    " :registered, :registered - 1)"
)
REFUSALS = {
    "an event is never edited": ("UPDATE event SET attrs = '{}' WHERE seq = :first", "append-only"),
    "an event is never deleted": ("DELETE FROM event WHERE seq = :first", "append-only"),
    "an unregistered event type is refused": (
        _row(type="made.up"),
        "not registered",
    ),
    "a step ends once": (
        _row(type="task", phase="failed", started_us=0, span_id=SPAN),
        "UNIQUE",
    ),
    "a step starts once": (_row(type="task", phase="started", span_id=SPAN), "UNIQUE"),
    "a cause precedes its effect": (_row(cause_seq=10**9), "FOREIGN KEY"),
    "a cause is an event": (_row(cause_seq=-5), "FOREIGN KEY"),
    "attributes stay under 4 KiB": (
        _row(attrs=json.dumps({"x": "y" * 5000})),
        r"length\(attrs\) <= 4096",
    ),
    "a finished step names its start": (
        _row(type="task", phase="completed", span_id="f" * 16),
        "started_us IS NOT NULL",
    ),
    "a text needs its event's digest": (
        "INSERT INTO event_text VALUES (:first, 'x')",
        "no text digest",
    ),
    "a text is never edited": (
        "UPDATE event_text SET text = 'edited' WHERE seq = :note",
        "never edited",
    ),
    "a definition is never edited": (
        "UPDATE definition_version SET spec = '{}' WHERE definition_id = :metric",
        "never edited",
    ),
    "a definition is never deleted": (
        "DELETE FROM definition_version WHERE definition_id = :metric",
        "never deleted",
    ),
    "a version follows the one before it": (
        _insert("definition_version", **SHARPE | {"version": 3}),
        "follows the one before",
    ),
    "a later version says how it relates": (
        _insert("definition_version", **SHARPE | {"relation": None, "conversion": None}),
        r"\(version = 1\) = \(relation IS NULL\)",
    ),
    "a conversion is a scale and an offset": (
        _insert("definition_version", **SHARPE | {"conversion": "old * 1.2"}),
        r"json_valid\(conversion\)",
    ),
    "a name names one lineage forever": (
        "UPDATE definition_name SET definition_id = 'x' WHERE definition_id = :metric",
        "never moved",
    ),
    "a lineage keeps its id": (
        "UPDATE definition_lineage SET definition_id = 'x' WHERE definition_id = :metric",
        "keeps its id",
    ),
    "a selection is never edited": ("UPDATE selection SET cutoff_us = 0", "never edited"),
    "a recipe names a registered method": (
        "INSERT INTO recipe VALUES ('r0', 'reversal', 1, '{}', 1)",
        "FOREIGN KEY",
    ),
    "a recipe reads registered recipes": (
        "INSERT INTO recipe_upstream VALUES ('r1', 0, 'no-such-recipe')",
        "FOREIGN KEY",
    ),
    "a trial ends after its registration": (LATE_TRIAL, "ended_seq > registered_seq"),
    "a measure names a registered metric version": (
        "INSERT INTO measure VALUES ('a1', :metric, 9, '', 1.0, 1)",
        "FOREIGN KEY",
    ),
    "a recorded measure never moves": ("UPDATE measure SET value = 9", "never moves"),
    "an artifact's profile is kept forever": ("DELETE FROM artifact", "kept forever"),
    "the ledger is append-only": ("DELETE FROM ledger_entry", "append-only"),
}


@pytest.mark.proves("data_layer.log_refuses_what_its_rules_forbid")
@pytest.mark.parametrize(("statement", "why"), REFUSALS.values(), ids=REFUSALS)
def test_the_log_refuses_what_its_rules_forbid(log, statement, why):
    """Each rule of log.sql refuses its breach by name and leaves the log as it was."""
    con, seqs = log
    sql, args = statement if isinstance(statement, tuple) else (statement, seqs)
    before = con.execute("SELECT total_changes()").fetchone()[0]
    with pytest.raises(sqlite3.DatabaseError, match=why):
        con.execute(sql, args)
    assert con.execute("SELECT total_changes()").fetchone()[0] == before


@pytest.mark.proves("data_layer.seed_registry_is_whole")
def test_a_new_log_holds_the_seed_registry_whole(log):
    """Each seed row is a registered family's version 1, registered at sequence 1; registry.seeded,
    the four research tags and each artifact type's measures exist."""
    con, _ = log
    rows = con.execute(
        "SELECT family, name, version, registered_seq, status_seq, spec FROM definition"
        " WHERE owner = 'kernel'"
    ).fetchall()
    named = {family: {n for f, n, *_ in rows if f == family} for family, *_ in rows}
    measures = {
        m for f, *_, spec in rows if f == "artifact_type" for m in json.loads(spec)["measures"]
    }
    assert (len(rows), {row[2:5] for row in rows}) == (94, {(1, 1, 1)})
    assert named["tag"] == {"instrument", "theme", "regime", "quality"}
    assert set(named) - {"family"} <= named["family"]
    assert "registry.seeded" in named["event_type"]
    assert measures <= named["metric"]


@pytest.fixture(scope="module")
def settled(tmp_path_factory):
    """A new log where sharpe's version 2 replaced version 1, and a book bought 10 shares at 150
    with one fee posted on its own: the reference's view state."""
    root = tmp_path_factory.mktemp("settled-root")
    create_data_root(root)
    con = sqlite3.connect(root / "machine" / "log.sqlite", isolation_level=None)
    con.execute("PRAGMA foreign_keys = ON")  # log.sql's first pragma holds per connection
    con.execute(*_insert("definition_version", **SHARPE))
    con.execute(
        "UPDATE definition_version SET status = 'deprecated', status_seq = 2"
        " WHERE definition_id = ? AND version = 1",
        (METRIC,),
    )
    con.execute("INSERT INTO workspace VALUES ('w1', 'main', NULL, 'active', 1, NULL)")
    active = _event(con, "book.activated", ("book", "b1"))
    con.execute("INSERT INTO book VALUES ('b1', 'w1', 'paper', 'active', NULL, ?, NULL)", (active,))
    fill = _event(con, "ledger.posted", ("book", "b1"))
    legs = [("book", "AAPL", 10), ("market", "AAPL", -10)]
    legs += [("book", "CASH:USD", -1500), ("market", "CASH:USD", 1500)]
    con.executemany(
        "INSERT INTO ledger_entry VALUES (?, ?, 'b1', ?, ?, ?)",
        [(fill, line, account, asset, n * N) for line, (account, asset, n) in enumerate(legs, 1)],
    )
    fee = _event(con, "ledger.posted", ("book", "b1"))
    con.execute("INSERT INTO ledger_entry VALUES (?, 1, 'b1', 'fees', 'CASH:USD', ?)", (fee, N))
    lesson = _event(con, "lesson.recorded", ("goal", "g0"), text="Check the calendar first.")
    con.execute("DELETE FROM event_text WHERE seq = ?", (lesson,))
    yield con, fee
    con.close()


PLANS = {
    "a conclusion by subject": (
        "event WHERE type = 'conclusion.drawn' AND subject_id = 't1'",
        "event_type_subject",
    ),
    "a subject's events": (
        "event WHERE subject_kind = 'trial' AND subject_id = 't1'",
        "event_subject",
    ),
    "a metric's measures": (
        f"measure WHERE metric_id = '{METRIC}' AND metric_version = 2",
        "measure_by_metric",
    ),
    "a recipe's trials": ("trial WHERE recipe_key = 'r1'", "trial_recipe"),
}
READS = {
    "the current definition is the newest standing version": (
        "SELECT version FROM definition_current WHERE family = 'metric' AND name = 'sharpe'",
        [(2,)],
    ),
    "every metric version stays readable": (
        "SELECT count(*) FROM metric WHERE metric = 'sharpe'",
        [(2,)],
    ),
    "the event type manual reads its spec": (
        "SELECT durable FROM event_type WHERE type = 'decision.made'",
        [(1,)],
    ),
    "a forgotten text leaves its event and digest": (
        "SELECT e.text_digest, count(t.seq) FROM event e LEFT JOIN event_text t USING (seq)"
        " WHERE e.type = 'lesson.recorded'",
        [("blake3:t", 0)],
    ),
    "a position sums the book's legs": (
        "SELECT quantity_n FROM position WHERE asset = 'AAPL'",
        [(10 * N,)],
    ),
    "only the unbalanced posting shows": (
        "SELECT seq, asset, off_n FROM ledger_unbalanced",
        [("fee", "CASH:USD", N)],
    ),
}


@pytest.mark.proves("data_layer.log_views_read_as_defined")
@pytest.mark.parametrize(("sql", "expected"), READS.values(), ids=READS)
def test_the_log_reads_through_its_views(settled, sql, expected):
    """Each view of log.sql reads the reference's settled state as it says."""
    con, fee = settled
    assert con.execute(sql).fetchall() == [
        tuple(fee if v == "fee" else v for v in row) for row in expected
    ]


@pytest.mark.proves("data_layer.log_views_read_as_defined")
@pytest.mark.parametrize(("source", "index"), PLANS.values(), ids=PLANS)
def test_each_hot_read_uses_its_index(settled, source, index):
    """SQLite plans each hot read of the log through the index made for it."""
    con, _ = settled
    plan = " ".join(row[3] for row in con.execute(f"EXPLAIN QUERY PLAN SELECT * FROM {source}"))
    assert index in plan, plan


def _tables(path: Path) -> set[str]:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        return {
            name for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        con.close()


@pytest.mark.proves("data_layer.vector_table_defers_without_extension_loading")
def test_without_extension_loading_only_the_vector_table_is_deferred(tmp_path, monkeypatch):
    """An interpreter that refuses extension loading still gets every other index table, and the
    root names the vector table deferred rather than refusing."""
    import sqlite_vec

    def refuse(connection):
        raise sqlite3.OperationalError("not authorized")

    monkeypatch.setattr(sqlite_vec, "load", refuse)
    made = create_data_root(tmp_path)
    tables = _tables(tmp_path / "machine" / "index.sqlite")
    assert "machine/index.sqlite" in made.created and "passage_vec" not in tables
    assert {"passage", "index_meta"} <= tables
    assert made.deferred[-1] == "index: vector table deferred (extension loading unavailable)"


def _files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_file()
        else "folder"
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.proves("data_layer.root_created_twice_is_all_equal")
def test_a_data_root_created_twice_is_all_equal_and_refuses_another_version(tmp_path):
    """The second creation makes and changes nothing; another schema version is refused by name."""
    first = create_data_root(tmp_path)
    assert "machine/catalog.sqlite" in first.created
    reading = reading_session(tmp_path)  # the Host's views over the log, the lake attached
    assert reading is not None
    with closing(reading):
        assert reading.execute("SELECT count(*) FROM raw_bar_asof(1)").fetchall() == [(0,)]
        assert reading.execute("SELECT count(*) FROM corporate_action_asof(1)").fetchall() == [(0,)]
        assert reading.execute("SELECT count(*) FROM experience").fetchall() == [(0,)]
    files = _files(tmp_path)
    second = create_data_root(tmp_path)
    made = ("machine", "data", "blobs", "staging", "extensions")
    made += ("machine/log.sqlite", "machine/index.sqlite", "machine/catalog.sqlite")
    assert (first.created, second.created, _files(tmp_path)) == (made, (), files)
    assert second.deferred == ()
    assert _tables(tmp_path / "machine" / "index.sqlite") >= {
        "passage",
        "passage_vec",
        "index_meta",
    }
    con = sqlite3.connect(tmp_path / "machine" / "log.sqlite")
    con.execute("PRAGMA user_version = 9")
    con.close()
    with pytest.raises(DataError, match="schema version 9") as refused:
        create_data_root(tmp_path)
    assert refused.value.failure.code == "data_layer.schema_mismatch"
