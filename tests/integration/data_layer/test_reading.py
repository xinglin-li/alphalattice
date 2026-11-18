"""The reading surface the Host opens at start: views.sql over the log in DuckDB, and `compare`."""

from __future__ import annotations

import itertools
import json
import math
import shutil
import sqlite3
import statistics
import uuid
from contextlib import closing
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import create_data_root, history, reading, return_series
from alphalattice.kernel.data_layer import log as log_module
from alphalattice.kernel.data_layer.chain import canonical
from alphalattice.kernel.data_layer.log import HostLog, uuid7
from alphalattice.kernel.data_layer.project import definition_id
from alphalattice.kernel.data_layer.reading import compare, reading_session
from alphalattice.kernel.data_layer.store import StorePort, as_columns, stage
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
    minted = itertools.count(1)  # the artifact and recipe ids the history mints, in order

    def uuid7():
        return str(uuid.UUID(int=next(minted)))

    with (
        patch.object(log_module, "time", SimpleNamespace(time_ns=lambda: T0 * 1000)),
        patch.object(log_module, "uuid", ids),
        patch.object(project_fixture, "uuid", ids),
        patch.object(project_fixture, "uuid7", uuid7),
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


@pytest.mark.proves("data_layer.a_class_read_is_cut_at_its_known_time")
@pytest.mark.parametrize(
    ("width", "cut", "members", "states", "sharpe_n", "unknown_then"),
    [
        pytest.param("recipe", None, ["t1", "t2"], ["completed"] * 2, 2, [], id="recipe-now"),
        pytest.param(
            "method_version", None, ["t1", "t2", "t3"], ["completed"] * 3, 3, [], id="version-now"
        ),
        pytest.param("method", None, ["t1", "t2", "t3"], ["completed"] * 3, 3, [], id="method-now"),
        pytest.param(
            "method",
            "t3",
            ["t1", "t2", "t3"],
            ["completed", "completed", "running"],
            2,
            [],
            id="cut-before-t3-ended",
        ),
        pytest.param(
            "method_version", "t1", ["t1"], ["running"], 0, ["t2", "t3"], id="cut-before-t1-ended"
        ),
    ],
)
def test_a_class_read_returns_its_members_and_spread_as_known_at_its_cutoff(
    analytics_log, width, cut, members, states, sharpe_n, unknown_then
):
    """A class read cut at a known time holds only what was known then: its trials, their states,
    their end's fields and their measures; a trial registered later is refused by name."""
    with closing(sqlite3.connect(analytics_log / "machine" / "log.sqlite")) as con:
        ended = dict(
            con.execute(
                "SELECT t.trial_id, e.recorded_us FROM trial t JOIN event e ON e.seq = t.ended_seq"
            ).fetchall()
        )
    session = reading_session(analytics_log)
    assert session is not None, "the reading views need DuckDB's sqlite extension"
    cutoff = None if cut is None else ended[cut] - 1
    with closing(session):
        read = reading.experience_class(session, trial="t1", width=width, cutoff_us=cutoff)
        for later in unknown_then:
            with pytest.raises(DataError, match="as known then") as refused:
                reading.experience_class(session, trial=later, width=width, cutoff_us=cutoff)
            assert refused.value.failure.code == "experience.trial_unknown"
    assert [m["trial_id"] for m in read["members"]] == members
    assert [m["state_then"] for m in read["members"]] == states
    ended_then = [state == "completed" for state in states]
    for field in ("result_artifact", "ended_seq"):  # set at the end, so unknown before it
        assert [m[field] is not None for m in read["members"]] == ended_then
    sharpe = [row for row in read["spread"] if row["metric"] == "sharpe" and row["dims"] == "{}"]
    assert [row["n"] for row in sharpe] == ([sharpe_n] if sharpe_n else [])
    assert read["search"]["trials"] == len(members)
    assert sum(len(m["measures"]) > 0 for m in read["members"]) == states.count("completed")


def _brief_log(log, crowd=0):
    """Two goals of one method: g1's three trials (the shared research) and g2's one, then `crowd`
    more of g2's and, when crowded, g1's t9 last; two decisions of one kind on g2 and one of
    another, lessons on g2, on g1's trial and on g1 for another method, and a conclusion on g1's
    trial; returns the seq of g2's newer decision of its kind."""
    _research(log)
    log.record("goal.set", ("goal", "g2"), {"title": "Momentum elsewhere"}, **AGENT)
    log.record("hypothesis.stated", ("hypothesis", "h2"), {"goal_id": "g2"}, text="H2", **AGENT)
    (recipe,) = log.read("SELECT recipe_key FROM trial WHERE trial_id = 't1'")
    registered = {"goal_id": "g2", "hypothesis_id": "h2", "selection_id": "s1"}
    registered |= {"recipe_key": recipe[0], "seeds": [7], "product_version": "0.2.8"}
    log.record("trial.registered", ("trial", "t4"), registered, **AGENT)
    for n in range(crowd):  # g2's own trials, filling the class's first page
        log.record("trial.registered", ("trial", f"c{n:03}"), registered, **AGENT)
    if crowd:  # a later trial of another goal, past the first page
        late = registered | {"goal_id": "g1", "hypothesis_id": "h1"}
        log.record("trial.registered", ("trial", "t9"), late, **AGENT)
    for kind, chosen, why in (("universe", "sp400", "First"), ("horizon", "21d", "Monthly")):
        decided = {"kind": kind, "chosen": chosen, "alternatives": ["x"], "goal_id": "g2"}
        log.record("decision.made", ("decision", f"d-{kind}"), decided, text=why, **AGENT)
    decided = {"kind": "universe", "chosen": "sp500", "alternatives": ["sp400"], "goal_id": "g2"}
    newer = log.record(
        "decision.made", ("decision", "d-universe-2"), decided, text="Wider", **AGENT
    )
    log.record("lesson.recorded", ("goal", "g2"), text="Check the universe first.", **AGENT)
    log.record("lesson.recorded", ("trial", "t1"), text="Costs matter.", **AGENT)
    other = {"method": "reversal"}
    log.record("lesson.recorded", ("goal", "g1"), other, text="Not this method.", **AGENT)
    log.record("conclusion.drawn", ("trial", "t1"), text="It paid in 2015-2024.", **AGENT)
    log.flush()
    return newer


MONTHS = [f"{2015 + i // 12}-{i % 12 + 1:02}" for i in range(24)]
WALSH = {  # three mutually orthogonal, zero-mean patterns: their correlations are exactly 0
    "w1": [0.01, -0.01] * 12,
    "w2": [0.01, 0.01, -0.01, -0.01] * 6,
    "w3": ([0.01] * 4 + [-0.01] * 4) * 3,
}


@pytest.mark.proves("data_layer.a_class_read_gives_its_effective_n")
@pytest.mark.parametrize(
    ("series", "cut_before", "members", "pairs", "n"),
    [
        pytest.param({}, None, 0, 0, None, id="no-series"),
        pytest.param({"t1": "w1"}, None, 1, 0, None, id="one-series"),
        pytest.param({"t1": "w1", "t2": "w1", "t3": "w1"}, None, 3, 3, 1.0, id="identical"),
        pytest.param({"t1": "w1", "t2": "w2", "t3": "w3"}, None, 3, 3, 3.0, id="independent"),
        pytest.param({"t1": "w1", "t2": "w1", "t3": "w3:6"}, None, 3, 1, 1.0, id="short-overlap"),
        pytest.param({"t1": "w1", "t2": "w2", "t3": "w3"}, "t3", 2, 1, 2.0, id="known-later"),
    ],
)
def test_a_class_read_gives_the_effective_n_of_its_members_kept_series(
    tmp_path, series, cut_before, members, pairs, n
):
    """A class's effective N is rho + (1 - rho) * M over the members' kept monthly series known
    at the cut, pairs with under 12 common months left out, and none below two series; a member
    with no goal is a member like any other."""
    root = tmp_path / "data"
    create_data_root(root)
    kept = {"storage": "inline", "table": None, "measures": [], "keep": "forever"}
    recorded = {}
    with HostLog.open(root) as log:
        _research(log)
        (recipe,) = log.read("SELECT recipe_key FROM trial WHERE trial_id = 't1'")
        goalless = {"selection_id": "s1", "recipe_key": recipe[0], "seeds": [7]}
        goalless["product_version"] = "0.2.8"  # no hypothesis and no task: its goal is none
        log.record("trial.registered", ("trial", "t0"), goalless, **AGENT)
        _define(log, "artifact_type", return_series.TYPE, text="Kept returns.", spec=kept)
        for trial, pattern in series.items():
            name, _, length = pattern.partition(":")
            months = MONTHS[: int(length or 24)]
            (recipe,) = log.read("SELECT recipe_key FROM trial WHERE trial_id = ?", (trial,))
            produced, body = return_series.produced(
                trial, recipe[0], months, WALSH[name][: len(months)], 1
            )
            recorded[trial] = log.record(
                "artifact.produced", ("artifact", uuid7()), produced, body=body, **AGENT
            )
        log.flush()
        cutoff = None
        if cut_before is not None:
            (at,) = log.read("SELECT recorded_us FROM event WHERE seq = ?", (recorded[cut_before],))
            cutoff = at[0] - 1
    session = reading_session(root)
    assert session is not None, "the reading views need DuckDB's sqlite extension"
    with closing(session):
        read = reading.experience_class(session, trial="t1", width="method", cutoff_us=cutoff)
    assert [m["trial_id"] for m in read["members"]] == ["t1", "t2", "t3", "t0"]
    assert read["search"]["trials"] == 4
    found = read["effective_n"]
    assert (found["members_with_series"], found["pairs"]) == (members, pairs)
    assert found["n"] == (None if n is None else pytest.approx(n))


@pytest.mark.proves("data_layer.a_brief_fits_its_budget_in_priority_order")
@pytest.mark.parametrize(
    ("budget", "cut", "crowd", "decisions", "lessons", "similar", "omitted"),
    [
        pytest.param(
            2000,
            False,
            0,
            ["d-universe-2", "d-horizon"],
            ["Check the universe first.", "Costs matter."],
            ["t3", "t2", "t1"],
            [0, 0, 0],
            id="all-fit",
        ),
        pytest.param(
            2000,
            False,
            200,
            ["d-universe-2", "d-horizon"],
            ["Check the universe first.", "Costs matter."],
            ["t9", "t3", "t2", "t1"],
            [0, 0, 0],
            id="own-trials-crowd-the-class",
        ),
        pytest.param(
            200,
            False,
            0,
            ["d-universe-2", "d-horizon"],
            ["Check the universe first.", "Costs matter."],
            [],
            [0, 0, 3],
            id="experiences-left-out",
        ),
        pytest.param(
            100, False, 0, ["d-universe-2", "d-horizon"], [], [], [0, 2, 3], id="lessons-left-out"
        ),
        pytest.param(
            2000,
            True,
            0,
            ["d-horizon", "d-universe"],
            [],
            ["t3", "t2", "t1"],
            [0, 0, 0],
            id="known-before-the-newer-decision",
        ),
    ],
)
def test_a_brief_fits_decisions_then_lessons_then_similar_experiences_to_its_budget(
    tmp_path, budget, cut, crowd, decisions, lessons, similar, omitted
):
    """A brief takes the goal's current decision per kind, then lessons (its own first), then
    other goals' trials of its method, each whole while it fits the budget, as known at its cut;
    a goal unknown then is refused by name."""
    root = tmp_path / "data"
    create_data_root(root)
    with HostLog.open(root) as log:
        newer = _brief_log(log, crowd)
        (at,) = log.read("SELECT recorded_us FROM event WHERE seq = ?", (newer,))
    cutoff = at[0] - 1 if cut else None
    session = reading_session(root)
    assert session is not None, "the reading views need DuckDB's sqlite extension"
    with closing(session):
        read = reading.brief(session, goal="g2", budget=budget, cutoff_us=cutoff)
        with pytest.raises(DataError, match="as known then") as refused:
            reading.brief(session, goal="g9", cutoff_us=cutoff)
    assert refused.value.failure.code == "experience.goal_unknown"
    parts = {part["part"]: part for part in read["parts"]}
    assert list(parts) == ["decisions", "lessons", "experiences"]
    assert [d["decision_id"] for d in parts["decisions"]["items"]] == decisions
    assert [n["text"] for n in parts["lessons"]["items"]] == lessons
    assert [e["trial_id"] for e in parts["experiences"]["items"]] == similar
    assert [part["omitted"] for part in read["parts"]] == omitted
    assert read["used"] <= budget


ALTERNATING = [0.01 if t % 2 == 0 else -0.01 for t in range(1, 260)]


def _world(root, returns, sessions=260):
    """A world of four listings of universe sp500, each `returns[i](t)` a day from 2024-01-01 at
    a price of 100; the fourth left the universe in June and rose half a day, so it shows only if
    membership at the cutoff is ignored. Returns the log's moments just before and after it was
    admitted, every world row being known at its admission."""
    with HostLog.open(root) as log:
        for family, name in (("provider", "yf"), ("universe", "sp500")):
            key = {"family": family, "name": name, "version": 1}
            text = canonical(key | {"text": f"The {name} {family}."}).decode()
            log.record("definition.registered", ("definition", f"{family}/{name}"), key, text=text)
        before = log.clock()
        port = StorePort.open(root, log)
        try:
            listings = [str(uuid.UUID(int=i + 1)) for i in range(4)]
            daily = [*returns, lambda t: 0.5]
            bars, prices = [], dict.fromkeys(listings, 100.0)
            for t in range(sessions):
                day = (date(2024, 1, 1) + timedelta(days=t)).isoformat()
                for listing, rule in zip(listings, daily, strict=True):
                    if t:
                        prices[listing] *= 1 + rule(t)
                    bars.append(
                        {"listing_id": listing, "session_date": day, "close": prices[listing]}
                        | {"adjusted_close": prices[listing], "source_observed_at": "2024-11-01"}
                    )
            members = [
                {"universe_id": UNIVERSE, "listing_id": listing, "valid_from": "2023-01-01"}
                | {"valid_to": "2024-06-01" if listing == listings[3] else None}
                for listing in listings
            ]
            provider = definition_id("provider", "yf")
            for table, rows in (("raw_bar", bars), ("universe_member", members)):
                with stage(root, table, provider, pa.table(as_columns(rows))) as staged:
                    port.commit(staged)
        finally:
            port.close()
        return before, log.clock()


UNIVERSE = definition_id("universe", "sp500")


@pytest.mark.proves("data_layer.market_condition_is_measured_at_its_cutoff")
@pytest.mark.parametrize(
    ("returns", "sessions", "known", "vol", "trend", "dispersion"),
    [
        pytest.param([lambda t: 0.001] * 3, 260, True, 0.0, 1.001**252 - 1, 0.0, id="steady"),
        pytest.param(
            [lambda t: ALTERNATING[t - 1]] * 3,
            260,
            True,
            statistics.stdev(ALTERNATING[-63:]) * math.sqrt(252),
            math.prod(1 + r for r in ALTERNATING[-252:]) - 1,
            0.0,
            id="alternating",
        ),
        pytest.param(
            [lambda t: 0.01, lambda t: -0.01, lambda t: 0.0],
            260,
            True,
            0.0,
            0.0,
            0.01,
            id="members-cancel-out",
        ),
        pytest.param([lambda t: 0.001] * 3, 30, True, None, None, 0.0, id="too-few-sessions"),
        pytest.param([lambda t: 0.001] * 3, 260, False, None, None, None, id="known-after-cutoff"),
    ],
)
def test_a_markets_condition_is_measured_from_what_world_knew_at_the_cutoff(
    tmp_path, monkeypatch, returns, sessions, known, vol, trend, dispersion
):
    """A market's condition at a cutoff is its members' equal-weighted volatility over 63
    sessions, return over 252 and cross-sectional dispersion over 21, by adjusted close, from
    members and bars known then, and a Host records it on a new selection of that cutoff."""
    root = tmp_path / "data"
    create_data_root(root)
    before, after = _world(root, returns, sessions)
    session = reading_session(root)
    assert session is not None, "the reading views need DuckDB's sqlite extension"
    with closing(session):
        cutoff = after if known else before
        found = reading.market_condition(session, universe=UNIVERSE, cutoff_us=cutoff)
    expected = {"market_vol_63d": vol, "market_return_252d": trend}
    expected["market_dispersion_21d"] = dispersion
    assert found == {
        name: None if value is None else pytest.approx(value, abs=1e-12)
        for name, value in expected.items()
    }
    history.close_history()  # the process's own root, if a test before this one took it
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(root))
    history.open_history("w1", "one", tmp_path / "workspace", root)
    try:
        made = history.selection({"cutoff_us": cutoff, "universe": "sp500"}, actor=AGENT["actor"])
        history.flush()
    finally:
        history.close_history()
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        recorded = dict(
            con.execute(
                "SELECT l.name, m.value FROM artifact a JOIN measure m USING (artifact_id)"
                " JOIN definition_lineage l ON l.definition_id = m.metric_id"
                " WHERE a.selection_id = ?",
                (made,),
            ).fetchall()
        )
    assert recorded == found


@pytest.mark.proves("data_layer.a_class_read_splits_its_effect_by_market_condition")
@pytest.mark.parametrize(
    ("cut", "bands"),
    [
        pytest.param(False, [("low", 3, 2.6 / 3), ("high", 2, 2.5)], id="now"),
        pytest.param(True, [("low", 3, 2.6 / 3), ("high", 1, 2.0)], id="before-s3-condition"),
    ],
)
def test_a_class_read_splits_each_effect_by_its_members_market_condition(tmp_path, cut, bands):
    """A class read bands its members by each condition metric of their selection, known at its
    cut, at the members' terciles, and gives each band's spread of each headline effect; a
    member whose condition was not known then is in no band."""
    root = tmp_path / "data"
    create_data_root(root)
    with HostLog.open(root) as log:
        _research(log)
        for metric, (spec, words) in reading.CONDITION_METRICS.items():
            _define(log, "metric", metric, text=words, spec=spec)
        kind = {"storage": "none", "table": None, "keep": "forever"}
        kind["measures"] = sorted(reading.CONDITION_METRICS)
        _define(log, "artifact_type", reading.CONDITION_TYPE, text="Condition.", spec=kind)
        (recipe,) = log.read("SELECT recipe_key FROM trial WHERE trial_id = 't1'")
        for selection, trial, sharpe in (("s2", "t5", 2.0), ("s3", "t6", 3.0)):
            window = {"cutoff_us": int(sharpe), "universe": "sp500"}  # a window of its own
            log.record("selection.made", ("selection", selection), window, **AGENT)
            project_fixture._trial(log, trial, recipe[0], sharpe, "none", selection)
        seqs = {}
        for selection, vol in (("s1", 0.1), ("s2", 0.2), ("s3", 0.3)):
            values = dict.fromkeys(reading.CONDITION_METRICS) | {"market_vol_63d": vol}
            produced, result = reading.condition_events(selection, "u", 1, values, 1)
            artifact = ("artifact", uuid7())
            seqs[selection] = log.record("artifact.produced", artifact, produced, **AGENT)
            log.record("result.recorded", artifact, result, **AGENT)
        log.flush()
        (at,) = log.read("SELECT recorded_us FROM event WHERE seq = ?", (seqs["s3"],))
    session = reading_session(root)
    assert session is not None, "the reading views need DuckDB's sqlite extension"
    with closing(session):
        read = reading.experience_class(
            session, trial="t1", width="method", cutoff_us=at[0] - 1 if cut else None
        )
    sharpe = [
        (row["band"], row["n"], pytest.approx(row["mean"]))
        for row in read["by_condition"]
        if row["condition"] == "market_vol_63d" and row["metric"] == "sharpe"
    ]
    assert sharpe == bands
    assert {row["condition"] for row in read["by_condition"]} == {"market_vol_63d"}


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
        meaning = {"type": "backtest", "type_version": 1, "identity_kind": "content"}
        meaning["storage"] = "none"
        sides = {"a": [["sharpe", 1, {}, 0.5]], "b": [["sharpe", 2, {}, 1.5]]}
        for side, measures in sides.items():
            log.record(
                "artifact.produced",
                ("artifact", side),
                meaning | {"meaning": side, "content_digest": f"blake3:{side}"},
                **AGENT,
            )
            log.record("result.recorded", ("artifact", side), {"measures": measures}, **AGENT)
        log.flush()
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        (row,) = compare(con, "a", "b")
    assert (row["version"], row["a"], row["b"], row["diff"]) == (2, 1.0, 1.5, 0.5)
    assert row["relation"] == "convertible"
