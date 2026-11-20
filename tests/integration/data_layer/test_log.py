"""The Host's one write path on a real data root: one unit per event, the chain, the lock, the
batches, and a kill inside an append."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root
from alphalattice.kernel.data_layer import log as host
from alphalattice.kernel.data_layer.log import HostLog, attach, verify_from_checkpoint, verify_whole
from alphalattice.kernel.shared_kernel import spawn

TEST_SOURCE = "src/alphalattice/kernel/data_layer/log.py"
T0 = 1_794_700_000_000_000  # 2026-11-14, UTC: every open here is in this month
DAY = 86_400_000_000
RESEARCH = {"actor": ("agent", "claude"), "node": "research"}


def _define(log, name, **who):
    """Register metric `name` v1 through its event, as an agent's own metric is."""
    spec = {"family": "metric", "name": name, "version": 1, "status": "standard"}
    text = json.dumps(spec | {"spec": {"unit": "ratio"}, "text": f"The {name} metric."})
    key = {"family": "metric", "name": name, "version": 1}
    return log.record(
        "definition.registered", ("definition", f"metric/{name}"), key, text=text, **who
    )


def _count(con, table="event"):
    """Rows in `table` as `con` reads them."""
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def _events(root):
    """The committed events of `root`'s log, as a reader sees them."""
    with closing(attach(root)) as con:
        return _count(con)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """A data root whose log holds its seed, a month's checkpoint, a definition and a step."""
    root = tmp_path_factory.mktemp("template") / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        _define(log, "probe", **RESEARCH)
        log.record(
            "tool.call", attrs={"tool": "grep"}, phase="completed", started_us=T0, **RESEARCH
        )
    return root


@pytest.fixture
def root(template, tmp_path):
    """A private copy of the template data root."""
    copy = tmp_path / "data"
    shutil.copytree(template, copy)
    return copy


REFUSED = {
    "refused at admission": (("goal.set", ("goal", "g9"), {"note": "no title"}), {}),
    "refused by its projection": (
        ("definition.status_changed", ("definition", "metric/probe")),
        {"attrs": {"family": "metric", "name": "probe", "version": 1, "status": "superseded"}}
        | {"actor": ("agent", "x"), "node": "plugin"},
    ),
    "refused after the chain folded": (
        ("artifact.produced", ("artifact", "r1")),
        {
            "attrs": {
                "type": "report",
                "meaning": "m",
                "storage": "inline",
                "identity_kind": "content",
                "type_version": 1,
                "content_digest": "blake3:r1",
            }
        },
    ),
    "refused by the log's own rule": (("lesson.recorded",), {"text": "no cause", "cause": -5}),
}


@pytest.mark.proves("data_layer.host_append_is_all_or_nothing")
@pytest.mark.parametrize(("args", "more"), REFUSED.values(), ids=REFUSED)
def test_a_refused_event_leaves_no_trace_and_its_batch_whole(root, args, more):
    """An event refused at any step of its append leaves no event, text, projection or chain
    change, and the ordinary event before it in the same batch is committed whole."""
    tables = ("event", "definition", "event_text")
    with HostLog.open(root, now_us=T0) as log:
        before = {table: log.read(f"SELECT count(*) FROM {table}")[0][0] for table in tables}
        log.record("tool.call", attrs={"tool": "ls"}, phase="completed", started_us=T0, **RESEARCH)
        with pytest.raises((DataError, sqlite3.DatabaseError)):
            log.record(*args, **more)
    with closing(attach(root)) as con:
        after = {table: _count(con, table) for table in tables}
        status = con.execute("SELECT status FROM definition WHERE name = 'probe'").fetchone()[0]
        assert verify_whole(con)
    assert after == before | {"event": before["event"] + 1}
    assert status == "standard"


@pytest.mark.proves("data_layer.host_append_is_all_or_nothing")
def test_an_accepted_definition_writes_its_event_text_head_and_row_together(root):
    """A registered definition's event, its text, the chain head at its seq and its registry row,
    with its registration seq, are all in the log once its append commits."""
    with HostLog.open(root, now_us=T0) as log:
        seq = _define(log, "breadth", **RESEARCH)
    with closing(attach(root)) as con:
        text = con.execute("SELECT text FROM event_text WHERE seq = ?", (seq,)).fetchone()[0]
        row = con.execute(
            "SELECT registered_seq, owner, text FROM definition WHERE name = 'breadth'"
        ).fetchone()
        head = con.execute("SELECT seq FROM chain_head").fetchone()[0]
    assert json.loads(text)["name"] == "breadth"
    assert tuple(row) == (seq, "research", "The breadth metric.")
    assert head == seq


def _admission_summary_cases(log):
    """Write legacy and bound provider lists through the same log."""
    prior = log.read(
        "SELECT spec FROM definition WHERE family = 'event_type'"
        " AND name = 'admission' AND version = 1"
    )[0][0]
    key = {"family": "event_type", "name": "admission", "version": 2}
    log.record(
        "definition.registered",
        ("definition", "event_type/admission"),
        key,
        text=json.dumps(
            key
            | {
                "relation": "equal",
                "text": "This admission can keep bound provider text.",
                "spec": json.loads(prior) | {"text": "optional"},
            }
        ),
    )
    providers = [f"development-alias-{i:04d}" for i in range(200)]
    summary_text = json.dumps(providers)
    assert len(summary_text) > host.ATTRS_CHARACTERS
    candidate = {"count": len(providers), "digest": host.digest(summary_text.encode())}
    summaries = {}
    for dataset, names, text in (
        ("market", providers[:1], None),
        ("sector", providers[:1], None),
        ("market", candidate, summary_text),
        ("sector", candidate, summary_text),
    ):
        attrs = {"provider": names, "dataset": dataset, "marks": {"fixture": [1, 2]}}
        summary_seq = log.record(
            "admission",
            ("workspace", "admission-fixture"),
            attrs,
            text=text,
            version=2 if text else 1,
        )
        summaries[summary_seq] = (attrs, text)
        if text is None:
            log.record("record.forgotten", attrs={"seq": summary_seq})
        else:
            with pytest.raises(DataError) as protected_summary:
                log.record("record.forgotten", attrs={"seq": summary_seq})
            assert protected_summary.value.failure.code == "data_layer.record_unforgettable"
    encoded_fixture = r'["\u0073k-' + "a" * 20 + '"]'
    for text, refusal in (
        (summary_text, "data_layer.event_attrs_form"),
        (encoded_fixture, "data_layer.event_secret"),
    ):
        with pytest.raises(DataError) as malformed:
            log.record(
                "admission",
                ("workspace", "admission-fixture"),
                {
                    "provider": {"count": 1, "digest": host.digest(text.encode())},
                    "dataset": "market",
                },
                text=text,
                version=2,
            )
        assert malformed.value.failure.code == refusal
    return summaries, summary_seq


@pytest.mark.proves("data_layer.host_append_is_all_or_nothing")
def test_an_inline_artifact_writes_its_profile_and_body_together(root):
    """The log projects both profile forms in the event unit, independently of the inline body."""
    report = {
        "type": "report",
        "meaning": "the study's report",
        "storage": "inline",
        "identity_kind": "content",
        "type_version": 1,
        "content_digest": "blake3:r1",
    }
    small = {"columns": ["value"]}
    large = {"listings": [f"listing-development-{i:04d}" for i in range(200)]}
    text = json.dumps(large)
    assert len(text) > host.ATTRS_CHARACTERS
    expected = {}
    with HostLog.open(root, now_us=T0) as log:
        for name, version, attrs, shape, profile in (
            ("legacy", 1, report | {"shape": small}, small, None),
            ("text", 2, report, large, text),
            ("attrs", None, report | {"shape": small}, small, None),
        ):
            try:
                seq = log.record(
                    "artifact.produced",
                    ("artifact", name),
                    attrs,
                    text=profile,
                    body=b"# Report\n",
                    version=version,
                )
            except DataError as error:
                if name != "text" or error.failure.code != "data_layer.event_type_unregistered":
                    raise
                pytest.fail("the first profile text write must register its event version")
            expected[name] = (seq, "present", shape, b"# Report\n")
        seq = expected["text"][0]
        projected = log.read("SELECT shape FROM artifact WHERE artifact_id = 'text'")[0][0]
        assert json.loads(projected or "null") == large
        log.record("record.forgotten", attrs={"seq": expected["legacy"][0]})
        with pytest.raises(DataError) as protected:
            log.record("record.forgotten", attrs={"seq": seq})
        assert protected.value.failure.code == "data_layer.record_unforgettable"
        summaries, summary_seq = _admission_summary_cases(log)

    query = (
        "SELECT a.artifact_id, a.produced_seq, a.body_state, a.shape, b.body"
        " FROM artifact a JOIN artifact_body b USING (artifact_id)"
    )
    with (
        closing(attach(root)) as con,
        closing(host.replay(con, root.parent / "profiles-replay.sqlite")) as rebuilt,
    ):
        for ledger in (con, rebuilt):
            assert {
                row[0]: (row[1], row[2], json.loads(row[3]), row[4])
                for row in ledger.execute(query)
            } == expected
            assert {
                row[0]: (json.loads(row[1]), row[2])
                for row in ledger.execute(
                    "SELECT e.seq, e.attrs, t.text FROM event e LEFT JOIN event_text t USING(seq)"
                    " WHERE e.type = 'admission'"
                )
            } == summaries
            assert verify_whole(ledger) and verify_from_checkpoint(ledger)

    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        con.row_factory = sqlite3.Row
        with con:
            con.execute("DELETE FROM event_text WHERE seq = ?", (summary_seq,))
            assert not verify_whole(con) and not verify_from_checkpoint(con)
            with pytest.raises(DataError) as missing_summary:
                host.replay(con, root.parent / "missing-admission-summary.sqlite")
            assert missing_summary.value.failure.code == "data_layer.event_text_missing"
            con.rollback()

    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        con.row_factory = sqlite3.Row
        con.execute("DELETE FROM event_text WHERE seq = ?", (seq,))
        assert not verify_whole(con) and not verify_from_checkpoint(con)
        with pytest.raises(DataError) as missing:
            host.replay(con, root.parent / "missing-profile.sqlite")
        assert missing.value.failure.code == "data_layer.artifact_profile_invalid"


@pytest.mark.proves("data_layer.chain_catches_a_rewritten_event")
@pytest.mark.parametrize(
    ("event_type", "at_start"),
    [
        pytest.param("tool.call", False, id="after the last checkpoint: start-up refuses"),
        pytest.param("registry.seeded", True, id="before it: the whole scrub catches it"),
    ],
)
def test_the_chain_catches_a_rewritten_event(root, event_type, at_start):
    """Both verifications pass on a whole log; an event rewritten past the append-only trigger
    breaks the whole fold, and start-up's fold from the last checkpoint when it follows it."""
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        con.row_factory = sqlite3.Row
        assert verify_whole(con) and verify_from_checkpoint(con)
        con.execute("DROP TRIGGER event_no_update")
        con.execute("UPDATE event SET attrs = '{\"rewritten\":1}' WHERE type = ?", (event_type,))
        con.commit()
        whole, start = verify_whole(con), verify_from_checkpoint(con)
    assert (whole, start) == (False, at_start)
    if not at_start:
        with pytest.raises(DataError, match="does not match its chain head"):
            HostLog.open(root, now_us=T0)


@pytest.mark.proves("data_layer.chain_catches_a_rewritten_event")
def test_a_new_log_is_seeded_and_checkpointed_once_a_month(root):
    """The first event is registry.seeded counting the seed registry; the head is checkpointed in
    the first month and again only in a month that has none."""
    HostLog.open(root, now_us=T0 + DAY).close()
    with closing(attach(root)) as con:
        seeded = con.execute("SELECT seq, type, attrs FROM event ORDER BY seq LIMIT 1").fetchone()
        definitions = _count(con, "definition")
        first = [r[0] for r in con.execute("SELECT seq FROM chain_checkpoint")]
    assert (seeded["seq"], seeded["type"]) == (1, "registry.seeded")
    assert json.loads(seeded["attrs"])["count"] == definitions - 1  # the probe followed the seed
    assert first == [1]
    HostLog.open(root, now_us=T0 + 31 * DAY).close()
    with closing(attach(root)) as con:
        months = con.execute("SELECT kind, seq FROM chain_checkpoint ORDER BY seq").fetchall()
        last = _count(con) - 1  # the head the new month's checkpoint event recorded
    assert [tuple(m) for m in months] == [("month", 1), ("month", last)]


KILL = """
import json, os, sys
from pathlib import Path
from alphalattice.kernel.data_layer import log as host
root, point, now = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
log = host.HostLog.open(root, now_us=now)
def die(*args):
    os._exit(9)
if point == "_end":
    def die(self, statement, real=host.HostLog._end):
        os._exit(9) if statement == "COMMIT" else real(self, statement)
    host.HostLog._end = die
else:
    setattr(host, point, die)
spec = {"family": "metric", "name": "killed", "version": 1, "status": "standard", "text": "k"}
key = {"family": "metric", "name": "killed", "version": 1}
log.record("definition.registered", ("definition", "metric/killed"), key, text=json.dumps(spec))
log.flush()
"""


@pytest.mark.proves("data_layer.kill_inside_append_leaves_the_log_whole")
@pytest.mark.parametrize(
    "point",
    [
        pytest.param("_fold", id="after the event row"),
        pytest.param("project", id="after the chain fold"),
        pytest.param("_end", id="after the projection, before the commit"),
    ],
)
def test_a_kill_inside_an_append_leaves_the_log_whole(root, point):
    """A process killed at each point inside an append leaves its copy of the log whole: the next
    Host opens it, the chain verifies, and the killed event left no row anywhere."""
    events = _events(root)
    source = Path(host.__file__).parents[3]
    child = spawn.run(
        [sys.executable, "-c", KILL, str(root), point, str(T0)],
        env=os.environ | {"PYTHONPATH": str(source)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no window on Windows
    )
    assert child.returncode == 9, child.stderr
    with HostLog.open(root, now_us=T0) as log:
        assert log.read("SELECT count(*) FROM event")[0][0] == events
        assert not log.read("SELECT 1 FROM definition WHERE name = 'killed'")
        log.record("tool.call", attrs={"tool": "after"}, phase="completed", started_us=T0)
    with closing(attach(root)) as con:
        assert verify_whole(con)
    assert _events(root) == events + 1


@pytest.mark.proves("data_layer.one_host_writes_the_data_root")
def test_a_second_host_is_refused_by_name_and_others_attach_read_only(root):
    """While a Host holds the data root, another is refused with the code and the holder's pid;
    a reader attaches read-only and its write is refused by SQLite."""
    with HostLog.open(root, now_us=T0):
        with pytest.raises(DataError, match=f"pid {os.getpid()}") as refused:
            HostLog.open(root, now_us=T0)
        assert refused.value.failure.code == "data_layer.host_running"
        with (
            closing(attach(root)) as reader,
            pytest.raises(sqlite3.OperationalError, match="readonly"),
        ):
            reader.execute("INSERT INTO event_text VALUES (1, 'x')")
    HostLog.open(root, now_us=T0).close()  # released at close


REPLY = """
import os, sys
from pathlib import Path
from alphalattice.kernel.data_layer.log import HostLog
log = HostLog.open(Path(sys.argv[1]), batch=1000, seconds=60, now_us=int(sys.argv[2]))
log.record("tool.call", attrs={"tool": "replied"}, phase="completed", started_us=int(sys.argv[2]))
log.flush()  # a reply waits for its commit
os._exit(9)
"""


@pytest.mark.proves("data_layer.durable_events_commit_at_once")
def test_a_durable_event_commits_at_once_and_an_ordinary_one_with_its_batch(root):
    """An ordinary event is committed by its batch's size, its deadline, a durable event after it
    or the flush a reply waits for, which a killed Host keeps; a durable one when appended."""
    with HostLog.open(root, now_us=T0) as log:  # a goal is set in its workspace
        log.record("workspace.created", ("workspace", "w1"), {"name": "main"})
    base = _events(root)
    with HostLog.open(root, batch=2, seconds=60, now_us=T0) as log:
        log.record("tool.call", attrs={"tool": "a"}, phase="completed", started_us=T0)
        assert _events(root) == base
        goal = {"title": "durable"}
        log.record("goal.set", ("goal", "g1"), goal, workspace="w1", actor=("person", "me"))
        assert _events(root) == base + 2
        log.record("tool.call", attrs={"tool": "b"}, phase="completed", started_us=T0)
        log.record("tool.call", attrs={"tool": "c"}, phase="completed", started_us=T0)
        assert _events(root) == base + 4
    with HostLog.open(root, seconds=0.01, now_us=T0) as log:
        log.record("tool.call", attrs={"tool": "d"}, phase="completed", started_us=T0)
        deadline = log._timer  # None once it has already fired
        if deadline is not None:
            deadline.join(timeout=10)
        assert _events(root) == base + 5
    kwargs = {"stdin": subprocess.DEVNULL, "capture_output": True, "timeout": 60}
    child = spawn.run(
        [sys.executable, "-c", REPLY, str(root), str(T0)],
        env=os.environ | {"PYTHONPATH": str(Path(host.__file__).parents[3])},
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no window on Windows
        **kwargs,
    )
    assert (child.returncode, _events(root)) == (9, base + 6)
