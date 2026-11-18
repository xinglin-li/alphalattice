"""A review's note closes what it reviews in one unit with its run's aggregates, an ended trial
with none awaits review, and the Host renders each note into an experience it versions and a later
note replaces, the old one kept; a grown log replays its experiences equal."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root, experiences
from alphalattice.kernel.data_layer.chain import digest
from alphalattice.kernel.data_layer.log import HostLog, replay, uuid7
from alphalattice.kernel.data_layer.project import definition_id, differences, recipe_key
from alphalattice.kernel.data_layer.reading import reading_session

TEST_SOURCE = "src/alphalattice/kernel/data_layer/experiences.py"
T0 = 1_794_700_000_000_000  # 2026-11-14, UTC
AGENT = {"workspace": "w1", "actor": ("agent", "claude")}
REQUESTED = {"type": "factor.run", "product_version": "0.2.8", "input_hash": "i", "plan_hash": "p"}
REQUESTED |= {
    "workflow_definition_hash": "d",
    "admitted_at": "2026-11-14T00:00:00Z",
    "goal_id": "g1",
}
RECIPE = recipe_key(definition_id("method", "momentum"), 1, {})
USAGE = {"gen_ai.usage.input_tokens": 1200, "gen_ai.usage.output_tokens": 300, "cost_usd": 0.25}
NOTE = experiences.note_text(
    "Momentum pays", "12-1 momentum on the S&P 500", "It paid, thinly", ["Mind costs"]
)


def _world(log: HostLog, trials: int = 3) -> None:
    """A goal g1 with an ended task k1 whose trials t0.. ran under it (each ended but the last,
    t<trials-1>, which still runs), every run in its own trace with the agent's usage."""
    log.record("workspace.created", ("workspace", "w1"), {"name": "w1"})
    log.record("goal.set", ("goal", "g1"), {"title": "g1"}, workspace="w1", actor=("person", "me"))
    log.record("task.requested", ("task", "k1"), REQUESTED, version=2, trace="a" * 32, **AGENT)
    log.record("task", ("task", "k1"), phase="started", trace="a" * 32, span="1" * 16, **AGENT)
    for family, name, said in (
        ("method", "momentum", "Past return over a lookback."),
        ("universe", "sp500", "The S&P 500's members."),
    ):
        named = {"family": family, "name": name, "version": 1}
        spec = named | {"spec": {}, "status": "standard", "text": said}
        text = json.dumps(spec, sort_keys=True)
        log.record(
            "definition.registered", ("definition", f"{family}/{name}"), named, text=text, **AGENT
        )
    log.record(
        "selection.made", ("selection", "s1"), {"cutoff_us": 1, "universe": "sp500"}, **AGENT
    )
    defined = {"method": "momentum", "method_version": 1, "params": {}, "recipe_key": RECIPE}
    log.record("recipe.defined", ("recipe", uuid7()), defined, **AGENT)
    for n in range(trials):
        trial, trace = f"t{n}", f"{n + 1:032x}"
        registered = {
            "selection_id": "s1",
            "recipe_key": RECIPE,
            "seeds": [n],
            "product_version": "x",
            "task_id": "k1",
        }
        log.record("trial.registered", ("trial", trial), registered, **AGENT)
        log.record(
            "trial", ("trial", trial), phase="started", trace=trace, span=f"{n + 1:016x}", **AGENT
        )
        if n == trials - 1:
            continue
        artifact = digest(f"{trial}/result".encode())
        produced = {
            "type": "backtest",
            "type_version": 1,
            "meaning": "Does momentum pay?",
            "recipe_key": RECIPE,
            "selection_id": "s1",
            "trial_id": trial,
            "storage": "none",
            "identity_kind": "recipe",
        }
        log.record("artifact.produced", ("artifact", artifact), produced, **AGENT)
        measures = {
            "trial_id": trial,
            "measures": [["sharpe", 1, {}, 0.5 + n], ["ic_mean", 1, {"horizon_days": 21}, 0.03]],
        }
        log.record("result.recorded", ("artifact", artifact), measures, **AGENT)
        log.record(
            "trial",
            ("trial", trial),
            {"result_artifact": artifact, **USAGE},
            phase="completed",
            trace=trace,
            span=f"{n + 1:016x}",
            started_us=T0 + n,
            occurred_us=T0 + n + 4_000_000,
            **AGENT,
        )
    log.record(
        "task",
        ("task", "k1"),
        USAGE,
        phase="completed",
        trace="a" * 32,
        span="1" * 16,
        started_us=T0,
        **AGENT,
    )


@pytest.fixture
def world(tmp_path):
    """A data root holding the world, and its open log."""
    root = tmp_path / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        _world(log)
        yield root, log


def _note(log: HostLog, layer: str, unit: str, verdict: str = "success", text: str = NOTE) -> int:
    return experiences.record_note(
        log, layer, unit, verdict, text, workspace="w1", actor=("agent", "claude")
    )


def _review_task(log: HostLog, task: str, trials: list[str]) -> int:
    """A task of kind review that ran in its own trace, and its note naming the trials it
    reviewed."""
    requested, trace = REQUESTED | {"type": "review", "input_hash": task}, "b" * 32
    log.record("task.requested", ("task", task), requested, version=2, trace=trace, **AGENT)
    for phase in ("started", "completed"):
        log.record(
            "task", ("task", task), phase=phase, trace=trace, span="2" * 16, started_us=T0, **AGENT
        )
    return experiences.record_note(
        log, "task", task, "success", NOTE, workspace="w1", actor=("agent", "claude"), trials=trials
    )


def _one(log: HostLog, sql: str, *args):
    return tuple(log.read(sql, args)[0])


def _review(root: Path, log: HostLog) -> dict[str, str | None]:
    """Each trial's review state as the experience view reads it."""
    log.flush()
    session = reading_session(root)
    try:
        return dict(session.execute("SELECT trial_id, review FROM experience").fetchall())
    finally:
        session.close()


@pytest.mark.proves("data_layer.a_note_closes_what_it_reviews")
def test_a_note_closes_what_it_reviews_in_one_unit_with_its_aggregates(world):
    """A trial's note is one event holding its verdict and run aggregates (its spans' usage and
    seconds, its headline measures' place in its recipe's spread) and closes the trial; a task's
    and a goal's close theirs; a note on a running trial leaves nothing at all."""
    _, log = world
    events = log.read("SELECT count(*) FROM event")[0][0]
    with pytest.raises(DataError) as refused:
        _note(log, "trial", "t2")
    assert refused.value.failure.code == "data_layer.review_before_its_end"
    log.record("workspace.created", ("workspace", "w2"), {"name": "w2"})
    for layer, unit, code in (  # another workspace's units: refused, nothing read or recorded
        ("trial", "t1", "data_layer.review_workspace_disagrees"),
        ("task", "k1", "data_layer.review_workspace_disagrees"),
        ("goal", "g1", "data_layer.review_unknown_goal"),
    ):
        with pytest.raises(DataError) as refused:
            experiences.record_note(
                log, layer, unit, "success", NOTE, workspace="w2", actor=AGENT["actor"]
            )
        assert refused.value.failure.code == code
    events += 1  # w2's creation
    assert log.read("SELECT count(*) FROM event")[0][0] == events
    seq = _note(log, "trial", "t1")
    attrs = json.loads(_one(log, "SELECT attrs FROM event WHERE seq = ?", seq)[0])
    assert attrs == {
        "verdict": "success",
        **USAGE,
        "seconds": 4.0,
        "place": [{"metric": "sharpe", "version": 1, "n": 2, "z": 0.707107}],
    }
    assert _one(log, "SELECT reviewed_seq FROM trial WHERE trial_id = 't1'") == (seq,)
    task = _note(log, "task", "k1", "inconclusive")
    assert _one(log, "SELECT reviewed_seq FROM task WHERE task_id = 'k1'") == (task,)
    goal = _note(log, "goal", "g1", "failure")
    assert _one(log, "SELECT state, closed_seq FROM goal WHERE goal_id = 'g1'") == ("done", goal)
    goal_attrs = json.loads(_one(log, "SELECT attrs FROM event WHERE seq = ?", goal)[0])
    assert (
        goal_attrs["gen_ai.usage.input_tokens"] == 3 * 1200
    )  # the task's trace and both ended trials'


@pytest.mark.proves("data_layer.an_ended_trial_awaits_its_review")
def test_an_ended_trial_with_no_note_awaits_review(world):
    """The experience view reads an ended trial with no note as awaiting review, a running one as
    neither, and a reviewed one as reviewed."""
    root, log = world
    assert _review(root, log) == {"t0": "awaiting", "t1": "awaiting", "t2": None}
    _note(log, "trial", "t0")
    assert _review(root, log) == {"t0": "reviewed", "t1": "awaiting", "t2": None}


@pytest.mark.proves("data_layer.each_note_renders_a_versioned_experience")
@pytest.mark.parametrize(
    ("path", "notes", "renderers", "versions"),
    [
        pytest.param("render-and-version", 1, (1, 2), {1: [1, 2]}, id="render-and-version"),
        pytest.param(
            "replace-with-the-old-kept", 2, (1,), {1: [1], 2: [1]}, id="replace-with-the-old-kept"
        ),
        pytest.param("multi-trial-review", 0, (1,), {}, id="multi-trial-review"),
    ],
)
def test_the_host_renders_a_note_into_an_experience_it_versions_and_replaces(
    world, monkeypatch, path, notes, renderers, versions
):
    """Each note becomes an experience of its own, versioned by renderer and replaced on its layer
    with the old one kept; a review task's experience names each trial it reviewed."""
    root, log = world
    if path == "multi-trial-review":
        with pytest.raises(DataError) as refused:  # only a review task names trials
            experiences.record_note(
                log,
                "task",
                "k1",
                "success",
                NOTE,
                workspace="w1",
                actor=AGENT["actor"],
                trials=["t0"],
            )
        assert refused.value.failure.code == "data_layer.review_trials_need_a_review_task"
        log.record("workspace.created", ("workspace", "w2"), {"name": "w2"})
        elsewhere = {k: v for k, v in REQUESTED.items() if k != "goal_id"} | {"type": "review"}
        other = {"workspace": "w2", "actor": AGENT["actor"], "trace": "c" * 32}
        log.record("task.requested", ("task", "r2"), elsewhere, version=2, **other)
        for phase in ("started", "completed"):
            log.record("task", ("task", "r2"), phase=phase, span="3" * 16, started_us=T0, **other)
        with pytest.raises(DataError) as refused:  # another workspace's trial is not its to name
            experiences.record_note(
                log,
                "task",
                "r2",
                "success",
                NOTE,
                workspace="w2",
                actor=AGENT["actor"],
                trials=["t0"],
            )
        assert refused.value.failure.code == "data_layer.review_workspace_disagrees"
        _review_task(log, "r1", ["t1", "t0"])
        (made,) = experiences.render_pending(log, root / "blobs")[-1:]
        assert _one(log, "SELECT experience_id FROM task WHERE task_id = 'r1'") == (made,)
        assert _one(log, "SELECT task_id FROM experience_made WHERE experience_id = ?", made) == (
            "r1",
        )
        named = "SELECT trial_id, role FROM trial_experience WHERE experience_id = ? ORDER BY 1"
        assert [tuple(r) for r in log.read(named, (made,))] == [
            ("t0", "reviewed"),
            ("t1", "reviewed"),
        ]
        return
    seqs = [
        _note(log, "trial", "t1", text=experiences.note_text("H", "M", f"conclusion {n}", []))
        for n in range(notes)
    ]
    made = []
    for renderer in renderers:
        monkeypatch.setattr(experiences, "RENDERER_VERSION", renderer)
        made += experiences.render_pending(log, root / "blobs")
    ids = list(dict.fromkeys(made))
    assert [
        list(r)
        for r in log.read(
            "SELECT experience_id, version, note_seq, renderer_version FROM experience_version"
            " ORDER BY recorded_seq"
        )
    ] == [
        [ids[n - 1], v, seqs[n - 1], renderers[v - 1] if path == "render-and-version" else 1]
        for n, vs in versions.items()
        for v in vs
    ]
    current = _one(log, "SELECT experience_id FROM trial WHERE trial_id = 't1'")[0]
    assert current == ids[-1]
    rows = log.read(
        "SELECT experience_id, task_id, replaced_seq FROM experience_made ORDER BY made_seq"
    )
    assert [(r[0], r[1], r[2] is not None) for r in rows] == [(i, "k1", i != current) for i in ids]
    for body, size in log.read("SELECT body, size FROM experience_version"):
        kept = (root / "blobs" / body.removeprefix("blake3:")).read_bytes()
        assert (digest(kept), len(kept)) == (body, size)
    assert experiences.render_pending(log, root / "blobs") == []  # nothing is rendered twice


@pytest.mark.proves("data_layer.an_experience_renders_again_equal")
def test_a_grown_log_replays_its_experiences_equal(tmp_path):
    """A log grown to many trials, each reviewed, rendered, re-rendered and some replaced, replays
    ALL EQUAL from its events, and each body renders again to the bytes its event named."""
    root = tmp_path / "data"
    create_data_root(root)
    with HostLog.open(root, now_us=T0) as log:
        _world(log, trials=201)
        for n in range(200):
            _note(
                log,
                "trial",
                f"t{n}",
                text=experiences.note_text("H", "M", f"trial {n}", [f"lesson {n}"]),
            )
        for n in range(0, 200, 10):
            _note(log, "trial", f"t{n}", "failure")
        _note(log, "task", "k1")
        _review_task(log, "r1", [f"t{n}" for n in range(0, 200, 7)])
        made = experiences.render_pending(log, root / "blobs")
        log.flush()
    assert len(made) == 222
    with (
        closing(sqlite3.connect(root / "machine" / "log.sqlite")) as source,
        closing(replay(source, tmp_path / "replayed.sqlite")) as again,
    ):
        assert differences(source, again) == []
        replaced = again.execute(
            "SELECT count(*) FROM experience_made WHERE replaced_seq IS NOT NULL"
        ).fetchone()
        assert tuple(replaced) == (20,)
        named = again.execute("SELECT count(*) FROM trial_experience WHERE role = 'reviewed'")
        assert tuple(named.fetchone()) == (29,)
