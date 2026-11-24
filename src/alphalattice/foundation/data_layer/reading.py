"""The data root's reading surface: views.sql in DuckDB over the log, and two artifacts compared.

`reading_session` is DuckDB with the log attached read-only and every view and macro of views.sql
defined (span, experience, approach_spread, search_size and the rest), the world's point-in-time
macros only when the catalog exists. The Host never opens it with the log: it writes the log through
Python's sqlite3, and DuckDB's sqlite extension is a second SQLite library, whose closing descriptor
releases every POSIX lock the process holds on the file, its writer's included. The Host reads the
views through `LogReader`, a process of their own that alone attaches the log (`serve`). Nothing
in it is stored; each view is computed on read, and `experience` reads profiles and measures only,
so it is whole whether or not any artifact body still exists. `compare` reads `measure` through
SQLite by its primary key, where DuckDB's SQLite scan would read every measure.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import struct
import subprocess
import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, suppress
from pathlib import Path
from threading import Lock
from typing import IO, Any

import alphalattice.foundation.data_layer.return_series as return_series
from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.waits import WaitTimeout, wait_until
from alphalattice.foundation.data_layer.chain import canonical, digest
from alphalattice.foundation.data_layer.log import attach
from alphalattice.foundation.data_layer.root import (
    catalog_matches,
    configure_read_cache,
    duckdb_session,
    sql_path,
    view_statements,
    world_table_ids,
)
from alphalattice.kernel.execution import child_process, execution_port, interruptible
from alphalattice.kernel.resources import worker_width

READING_DEFERRED = "reading views: deferred (DuckDB's sqlite extension absent)"
READ_SECONDS = 60
"""How long a read of the views waits for the reader's answer: a stuck reader is ended past it,
so its caller, which holds the history's lock, never waits on it for ever. The experience read
of a 200,000-event log answers in 0.2 s after the reader starts."""


def reading_session(root: Path, *, log: bool = True, owner: Any = None) -> Any | None:
    """DuckDB with `root`'s log attached read-only and views.sql defined, in UTC; None when
    `duckdb_session` defers (READING_DEFERRED). The caller closes it. Without `log`, only the
    catalog and the views that read no log: a process that opens the log through Python's
    sqlite3 opens it with no second SQLite library, whose closing descriptor would release the
    process's locks on the file."""
    connection = duckdb_session(root) if owner is None else owner.cursor()
    if connection is None:
        return None
    try:
        _configure_reading(connection, root, log=log, shared=owner is not None)
    except BaseException:
        connection.close()
        raise
    return connection


def _configure_reading(connection: Any, root: Path, *, log: bool, shared: bool) -> None:
    """Attach the log and catalog as they stand when the reader first needs their views."""
    machine = root / "machine"
    alias = "read_lake" if shared else "lake"
    configure_read_cache(connection)
    connection.execute("SET TimeZone = 'UTC'").complete()
    if log:
        path = sql_path(machine / "log.sqlite")
        connection.execute(f"ATTACH '{path}' AS log (TYPE SQLITE, READ_ONLY)").complete()
    lake = (machine / "catalog.sqlite").exists()
    if lake:
        data, catalog = sql_path(root / "data"), sql_path(machine / "catalog.sqlite")
        connection.execute(
            f"ATTACH 'ducklake:sqlite:{catalog}' AS {alias} (DATA_PATH '{data}/', READ_ONLY)"
        ).complete()
        catalog_matches(connection, name=alias)
    table_ids = {}
    if lake and log:
        table_ids = world_table_ids(connection, attached=True)
    elif lake:
        with closing(attach(root)) as con:
            table_ids = world_table_ids(con)
    if shared:
        # The read-only views and writer macros have their own in-memory namespaces.
        connection.execute("ATTACH ':memory:' AS read_views").complete()
        connection.execute("USE read_views").complete()
    for statement in view_statements(table_ids):
        if (lake or "lake." not in statement) and (log or "log." not in statement):
            connection.execute(statement.replace("lake.", f"{alias}.")).complete()


class CatalogReader:
    """Hold the catalog and one prepared SQL shape per named stage for its lifetime."""

    def __init__(self, connection: Any, *, catalog: str = "lake") -> None:
        """Take ownership of the catalog connection and its prepared statements."""
        self._connection = connection
        self._catalog = catalog
        self._connection.execute("SET TimeZone = 'UTC'").complete()
        self._lock = Lock()
        self._plans: dict[str, tuple[str, str]] = {}

    def cursor(self) -> Any:
        """Give a caller a separate short-lived cursor for an unprepared read."""
        with self._lock:
            return self._connection.cursor()

    @contextmanager
    def projection(self, stage: str, query: str, parameters: Sequence[Any]) -> Iterator[Any]:
        """Execute on a fresh statement snapshot and consume it before sharing the cursor."""
        from duckdb import ConstantExpression  # lazy: the catalog needs the optional data extra

        with self._lock, interruptible(self._connection.interrupt):
            previous, name = self._plans.get(stage, ("", f"result_{len(self._plans)}"))
            if previous != query:
                statement = query.replace("lake.", f"{self._catalog}.")
                self._connection.execute(f"PREPARE {name} AS {statement}").complete()
                self._plans[stage] = (query, name)
            # DuckDB's SQL EXECUTE cannot itself bind placeholders. Use the engine's typed
            # literal rendering so caller values never enter the statement as SQL.
            arguments = ", ".join(str(ConstantExpression(value)) for value in parameters)
            call = f"EXECUTE {name}({arguments})" if arguments else f"EXECUTE {name}"
            yield self._connection.sql(call)

    def close(self) -> None:
        """Close the held connection after its last stage consumes its result."""
        with self._lock:
            self._connection.close()


class LogReader:
    """The reading views of the data root's log, read in a process of their own (`serve`) that
    alone opens the log in DuckDB: started ahead by the Host, restarted when it has ended, and
    ended by `close`. A read takes `experience` or `goal_notes` by name with their filters and
    returns their rows as those functions do. The history serializes its calls under the log's
    guard."""

    def __init__(self, root: Path) -> None:
        """A reader of `root`'s log views and the executor for its request pipe."""
        self.root = root
        self._child: subprocess.Popen[bytes] | None = None
        self._used = False
        self._answers = execution_port().register_pool(
            ThreadPoolExecutor(max_workers=worker_width(1), thread_name_prefix="log-reader"),
            name="log-reader",
        )

    def start_ahead(self) -> None:
        """Launch once ahead of the first read, without waiting for the child to import."""
        if self._child is None:
            with suppress(OSError, MemoryError):
                self._child = self._launch()  # a refused spawn is still retried on read

    def _launch(self) -> subprocess.Popen[bytes]:
        return child_process.start_piped([sys.executable, "-m", __name__, str(self.root)])

    def read(self, name: str, **filters: Any) -> Any:
        """The answer of the view read `name` (`serve`'s reads) with `filters`, as that read
        returns it; a read's own refusal by its code; refused as `data_layer.history_unavailable`
        when the reader cannot answer, or has not within READ_SECONDS, when its process is ended
        and the next read starts another."""
        self._used = True
        child = self._child
        if child is None or child.poll() is not None:
            child = self._child = self._launch()
        return _ask(child, (name, filters), self._answers, self.root)

    def close(self) -> None:
        """End the reader's process: its input closed, it returns; one that has not within ten
        seconds is ended."""
        child, self._child = self._child, None
        if child is not None:
            _end(child, kill=not self._used)
        self._answers.shutdown(wait=False)


def _ask(
    child: subprocess.Popen[bytes], request: Any, answers: ThreadPoolExecutor, root: Path
) -> Any:
    """One read's exchange with the reader's process `child`, its answer awaited on `answers`:
    a read's own refusal raised again by its code; refused as `data_layer.history_unavailable`
    when the process cannot answer, or has not within READ_SECONDS, when it is ended and reaped,
    so the next read starts another."""
    assert child.stdin is not None and child.stdout is not None
    try:
        _send(child.stdin, request)
        ok, value = answers.submit(_receive, child.stdout).result(READ_SECONDS)
    except TimeoutError as error:
        _end(child, kill=True)
        raise DataError(
            f"the reading views at {root} did not answer within {READ_SECONDS} s",
            code="data_layer.history_unavailable",
        ) from error
    except (OSError, EOFError) as error:
        raise DataError(
            f"the reading views' process at {root} ended ({error})",
            code="data_layer.history_unavailable",
        ) from error
    if not ok and isinstance(value, tuple):  # the read's own refusal: its code and message
        raise DataError(value[1], code=value[0])
    if not ok:
        raise DataError(
            f"the reading views at {root} did not answer ({value})",
            code="data_layer.history_unavailable",
        )
    return value


def _end(child: subprocess.Popen[bytes], *, kill: bool) -> None:
    """End the reader's process and reap it: killed at once, or with its input closed, killed if
    it has not returned within ten seconds."""
    assert child.stdin is not None
    if kill:
        child.kill()
    else:
        child.stdin.close()
    try:
        wait_until(lambda _left: child.poll() is not None, timeout=10, what="the reader to end")
    except WaitTimeout:
        child.kill()
        wait_until(lambda _left: child.poll() is not None, timeout=10, what="the reader killed")


def _send(stream: IO[bytes], value: Any) -> None:
    body = pickle.dumps(value)
    stream.write(struct.pack("<Q", len(body)) + body)
    stream.flush()


def _requests(stream: IO[bytes]) -> Iterator[Any]:
    """Each request on `stream` until it ends."""
    header = stream.read(8)
    while len(header) == 8:
        yield pickle.loads(stream.read(struct.unpack("<Q", header)[0]))  # our own Host's request
        header = stream.read(8)


def _receive(stream: IO[bytes]) -> Any:
    header = stream.read(8)
    if len(header) < 8:
        raise EOFError("no answer")
    return pickle.loads(stream.read(struct.unpack("<Q", header)[0]))  # our own child's answer


def serve(root: Path, requests: IO[bytes], answers: IO[bytes]) -> None:
    """Answer each read `LogReader` sends, by name with its filters, from `reading_session` over
    `root` until the requests end. This process holds no HostLog, so its DuckDB is the only SQLite
    library that opens the log here."""
    session = duckdb_session(root)
    views_pending = session is not None
    reads = {
        "experience": experience,
        "goal_notes": goal_notes,
        "experience_class": experience_class,
        "brief": brief,
        "market_condition": market_condition,
    }
    try:
        for name, filters in _requests(requests):
            if views_pending:
                # A first-use Task may create the catalog after the Host starts this child.
                _configure_reading(session, root, log=True, shared=False)
                views_pending = False
            try:
                _send(answers, (True, reads[name](session, **filters)))
            except DataError as error:  # a read's own refusal, by its code
                _send(answers, (False, (error.failure.code, str(error))))
            except Exception as error:  # the Host refuses it by name
                _send(answers, (False, f"{type(error).__name__}: {error}"))
    finally:
        if session is not None:
            session.close()


def experience(
    con: Any,
    *,
    goal: str | None = None,
    hypothesis: str | None = None,
    method: str | None = None,
    workspace: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Read bounded trials from the experience view, newest first."""
    return _experience_rows(
        con,
        "experience",
        "registered_at DESC, trial_id",
        goal=goal,
        hypothesis=hypothesis,
        method=method,
        workspace=workspace,
        limit=limit,
    )


def goal_notes(con: Any, **filters: Any) -> list[dict[str, Any]]:
    """Read bounded goal conclusions and lessons, including goals without trials."""
    return _experience_rows(con, "goal_note", "recorded_at DESC, note_id", **filters)


def _experience_rows(
    con: Any,
    view: str,
    order: str,
    *,
    goal: str | None = None,
    hypothesis: str | None = None,
    method: str | None = None,
    workspace: str | None = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    filters = {
        "goal_id": goal,
        "hypothesis_id": hypothesis,
        "method": method,
        "workspace_id": workspace,
    }
    clauses = [f"{field} = ?" for field, value in filters.items() if value is not None]
    values = [value for value in filters.values() if value is not None]
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    result = con.execute(
        f"SELECT * FROM {view}{where} ORDER BY {order} LIMIT ?",
        [*values, limit],
    )
    names = [column[0] for column in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


CLASS_WIDTHS = {
    "recipe": "c.recipe_key = a.recipe_key",
    "method_version": "c.method_id = a.method_id AND c.method_version = a.method_version",
    "method": "c.method_id = a.method_id",
}
"""An experiment's class at each width the agent may pick: the same recipe; the same method
version with other parameters; any version of the method."""


def experience_class(
    con: Any,
    *,
    trial: str,
    width: str = "method_version",
    cutoff_us: int | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Every trial of `trial`'s class at `width`, across goals, workspaces and agents, as known at
    `cutoff_us` (now when None): those registered by then, each in the state it had then, its
    end's fields (result, candidates, end) only once it had ended, with the headline measures
    recorded by then; the class's spread per metric and its search size, over every member,
    failed and cancelled ones included. Refused by name for an unknown width, or a trial the
    history did not hold by then."""
    if width not in CLASS_WIDTHS:
        raise DataError(
            f"a class's width is one of {', '.join(CLASS_WIDTHS)}, not {width!r}",
            code="experience.class_width_unknown",
        )
    known = _known(con, cutoff_us)
    _class_now(con, trial, width, known)
    try:
        members = _members(con, known, limit=limit)
        measured = _MEASURED
        spread = _rows(
            con,
            f"SELECT metric, metric_version, dims, count(*) AS n, avg(value) AS mean,"
            f" stddev_samp(value) AS sd, min(value) AS lo, median(value) AS mid, max(value) AS hi"
            f" FROM ({measured}) GROUP BY ALL ORDER BY metric, metric_version, dims",
            {"known": known},
        )
        search = _rows(
            con,
            "SELECT count(*) AS trials, sum(candidates) AS candidates,"
            " count(DISTINCT recipe_key) AS recipes, count(DISTINCT goal_id) AS goals,"
            " count(DISTINCT workspace_id) AS workspaces, count(DISTINCT actor_id) AS actors,"
            " count(*) FILTER (WHERE state_then = 'completed') AS completed,"
            " count(*) FILTER (WHERE state_then IN ('failed', 'cancelled')) AS failed"
            " FROM class_now",
        )[0]
        independent = _effective_n(con, known)
        by_condition = _rows(
            con,
            f"WITH cond AS (SELECT c.trial_id, ml.name AS condition, m.value AS level"
            " FROM class_now c JOIN log.artifact a ON a.selection_id = c.selection_id"
            " AND a.produced_seq <= $known JOIN log.definition_name tn"
            " ON tn.definition_id = a.type_id AND tn.family = 'artifact_type' AND tn.name = $type"
            " JOIN log.measure m ON m.artifact_id = a.artifact_id AND m.recorded_seq <= $known"
            " AND m.value IS NOT NULL"
            " JOIN log.definition_lineage ml ON ml.definition_id = m.metric_id"
            " QUALIFY row_number() OVER (PARTITION BY c.trial_id, ml.name"
            " ORDER BY a.produced_seq DESC) = 1),"
            " cuts AS (SELECT condition, quantile_cont(level, 1 / 3) AS q1,"
            " quantile_cont(level, 2 / 3) AS q2 FROM cond GROUP BY condition),"
            " bucket AS (SELECT trial_id, condition, level, CASE WHEN level <= q1 THEN 'low'"
            " WHEN level <= q2 THEN 'mid' ELSE 'high' END AS band"
            " FROM cond JOIN cuts USING (condition))"
            " SELECT b.condition, b.band, min(b.level) AS level_lo, max(b.level) AS level_hi,"
            " e.metric, e.metric_version, count(*) AS n, avg(e.value) AS mean,"
            " stddev_samp(e.value) AS sd FROM bucket b"
            f" JOIN ({measured}) e USING (trial_id) WHERE e.dims = '{{}}'"
            " GROUP BY b.condition, b.band, e.metric, e.metric_version"
            " ORDER BY b.condition, min(b.level), e.metric, e.metric_version",
            {"known": known, "type": CONDITION_TYPE},
        )
    finally:
        con.execute("DROP TABLE IF EXISTS class_now")
    return {
        "trial": trial,
        "width": width,
        "cutoff_us": cutoff_us,
        "known_seq": known,
        "members": members,
        "spread": spread,
        "search": search,
        "effective_n": independent,
        "by_condition": by_condition,
    }


CONDITION_TYPE = "market_condition"
"""The Host's artifact type of a market's condition at a selection's cutoff: its measures are
CONDITION_METRICS, recorded once per selection, so every trial of that selection reads it."""
CONDITION_METRICS = {
    "market_vol_63d": (
        {"unit": "ratio", "direction": "none", "params": {"sessions": 63, "periods": 252}},
        "The universe's equal-weighted daily return's standard deviation over the 63 sessions to"
        " the cutoff, annualized.",
    ),
    "market_return_252d": (
        {"unit": "ratio", "direction": "none", "params": {"sessions": 252}},
        "The universe's equal-weighted return compounded over the 252 sessions to the cutoff.",
    ),
    "market_dispersion_21d": (
        {"unit": "ratio", "direction": "none", "params": {"sessions": 21}},
        "The mean, over the 21 sessions to the cutoff, of the standard deviation across the"
        " universe's members of their daily returns.",
    ),
}
"""A market's condition as registered, versioned metrics (version 1 each), never a tag: each with
its spec and its definition's words. A metric with too few sessions known is undefined (null)."""


def market_condition(con: Any, *, universe: str, cutoff_us: int) -> dict[str, float | None]:
    """The condition of `universe` (its lineage id) as known at `cutoff_us`: CONDITION_METRICS
    over its members on the cutoff's date and their daily returns by the provider's adjusted
    close, from world's point-in-time reads alone, so nothing known later moves it."""
    found = _rows(
        con,
        "WITH day AS (SELECT CAST(make_timestamptz($cutoff) AS DATE) AS d),"
        " members AS (SELECT DISTINCT listing_id FROM universe_member_asof($cutoff), day"
        " WHERE universe_id = $universe AND valid_from <= d"
        " AND (valid_to IS NULL OR valid_to > d)),"
        " bars AS (SELECT b.listing_id, b.session_date, b.adjusted_close"
        " FROM raw_bar_asof($cutoff) b JOIN members USING (listing_id), day"
        " WHERE b.session_date <= d AND b.adjusted_close > 0"
        " QUALIFY row_number() OVER (PARTITION BY b.listing_id, b.session_date"
        " ORDER BY b.provider_id) = 1),"
        " days AS (SELECT DISTINCT session_date FROM bars ORDER BY session_date DESC LIMIT 253),"
        " r AS (SELECT session_date, adjusted_close / lag(adjusted_close)"
        " OVER (PARTITION BY listing_id ORDER BY session_date) - 1 AS r FROM bars"
        " WHERE session_date >= (SELECT min(session_date) FROM days)),"
        " d AS (SELECT session_date, avg(r) AS m, stddev_samp(r) AS x,"
        " row_number() OVER (ORDER BY session_date DESC) AS back"
        " FROM r WHERE r IS NOT NULL GROUP BY session_date)"
        " SELECT CASE WHEN count(*) FILTER (WHERE back <= 63) = 63"
        " THEN stddev_samp(m) FILTER (WHERE back <= 63) * sqrt(252) END AS market_vol_63d,"
        " CASE WHEN count(*) FILTER (WHERE back <= 252) = 252"
        " THEN exp(sum(ln(1 + m)) FILTER (WHERE back <= 252)) - 1 END AS market_return_252d,"
        " CASE WHEN count(x) FILTER (WHERE back <= 21) = 21"
        " THEN avg(x) FILTER (WHERE back <= 21) END AS market_dispersion_21d FROM d",
        {"universe": universe, "cutoff": cutoff_us},
    )[0]
    return {name: found[name] for name in CONDITION_METRICS}


def condition_events(
    selection: str,
    universe: str,
    cutoff_us: int,
    values: dict[str, float | None],
    type_version: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The `artifact.produced` and `result.recorded` attributes that record `values`, a market's
    condition at `selection`'s cutoff: one artifact of the selection, its content digest over
    the values and what gives them their meaning (this universe at this cutoff, each metric at
    version 1), so two different results of one question are two contents."""
    asked = {"universe_id": universe, "cutoff_us": cutoff_us, "metric_version": 1}
    asked["values"] = values
    produced = {
        "type": CONDITION_TYPE,
        "type_version": type_version,
        "identity_kind": "content",
        "content_digest": digest(canonical(asked)),
        "storage": "none",
        "meaning": f"The market's condition at selection {selection}'s cutoff.",
        "selection_id": selection,
    }
    measures = [[name, 1, {}, value] for name, value in sorted(values.items())]
    return produced, {"measures": measures}


_MEASURED = (  # each class member's headline measures recorded by then
    "SELECT c.trial_id, ml.name AS metric, m.metric_version, m.dims, m.value"
    " FROM class_now c JOIN log.measure m ON m.artifact_id = c.result_artifact"
    " JOIN log.definition_lineage ml ON ml.definition_id = m.metric_id"
    " WHERE m.recorded_seq <= $known AND m.value IS NOT NULL"
)


def _class_now(con: Any, trial: str, width: str, known: int) -> None:
    """Build `class_now`: every trial of `trial`'s class at `width` registered by seq `known`,
    each end's fields only once it had ended then. Refused by name, with nothing built, for a
    trial the history did not hold then, since a trial known then is a member of its own class."""
    then = "CASE WHEN c.ended_seq <= $known THEN c.{0} END AS {0}"  # an end's fields, once known
    con.execute(
        "CREATE OR REPLACE TEMP TABLE class_now AS SELECT c.* REPLACE ("
        + ", ".join(then.format(name) for name in ("result_artifact", "candidates", "ended_seq"))
        + "), CASE WHEN c.ended_seq <= $known THEN c.state ELSE 'running' END AS state_then"
        f" FROM class_member a JOIN class_member c ON {CLASS_WIDTHS[width]}"
        " WHERE a.trial_id = $trial AND a.registered_seq <= $known"
        " AND c.registered_seq <= $known",
        {"known": known, "trial": trial},
    )
    if con.execute("SELECT count(*) FROM class_now").fetchone()[0] == 0:
        con.execute("DROP TABLE class_now")
        raise DataError(
            f"trial {trial!r} is not in the Host's history as known then",
            code="experience.trial_unknown",
        )


def _members(
    con: Any, known: int, *, limit: int, other_than: str | None = None, newest: bool = False
) -> list[dict[str, Any]]:
    """A page of `class_now`'s members, each with its measures known then: the oldest first, or
    the `newest` first, chosen from the whole class, leaving out goal `other_than`'s, so the page
    bound applies after the choice."""
    other = "" if other_than is None else " WHERE goal_id IS DISTINCT FROM $goal"
    members = _rows(
        con,
        f"SELECT * EXCLUDE (state) FROM class_now{other}"
        f" ORDER BY registered_seq {'DESC' if newest else 'ASC'}, trial_id LIMIT $limit",
        {"limit": limit} | ({} if other_than is None else {"goal": other_than}),
    )
    by_trial: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(
        con,
        f"{_MEASURED} AND list_contains($trials, c.trial_id)"
        " ORDER BY c.trial_id, metric, metric_version, dims",
        {"known": known, "trials": [member["trial_id"] for member in members]},
    ):
        by_trial.setdefault(row.pop("trial_id"), []).append(row)
    for member in members:
        member["measures"] = by_trial.get(member["trial_id"], [])
    return members


def _known(con: Any, cutoff_us: int | None) -> int:
    """The last event recorded by `cutoff_us` (every event when None): a known time read as a
    seq, so every join cuts by seq exactly."""
    return int(
        con.execute(
            "SELECT coalesce(max(seq), 0) FROM log.event WHERE ? IS NULL OR recorded_us <= ?",
            [cutoff_us, cutoff_us],
        ).fetchone()[0]
    )


BRIEF_BUDGET = 2000
"""A brief's default budget in tokens: about a page an agent reads before it registers a trial."""
BRIEF_BUDGETS = range(100, 32_001)
"""The budgets a brief may be given, in tokens."""
CHARS_PER_TOKEN = 4
"""The estimate a brief counts tokens by: about four characters of English text a token, the
providers' published rule of thumb. An estimate, so a budget bounds a brief to within it."""
_PART_ROWS = 500
"""The most rows each part reads before it is fitted; the budget, not this, bounds the brief."""


def brief(
    con: Any, *, goal: str, budget: int = BRIEF_BUDGET, cutoff_us: int | None = None
) -> dict[str, Any]:
    """What an agent reads before new work on `goal`, as known at `cutoff_us`, fitted to
    `budget` tokens in priority order: the goal's current decision of each kind, newest first;
    then lessons, the goal's own before those of other goals that tried its methods; then similar
    experiences, the other goals' trials of its newest trial's method (`experience_class` at
    width method), with their conclusions. Each item is taken whole while it fits and counted as
    omitted when it does not. Refused by name for a budget out of range or a goal the history
    did not hold then."""
    if budget not in BRIEF_BUDGETS:
        raise DataError(
            f"a brief's budget is {BRIEF_BUDGETS.start} to {BRIEF_BUDGETS.stop - 1} tokens",
            code="experience.brief_budget_invalid",
        )
    known = _known(con, cutoff_us)
    asked = {"goal": goal, "known": known}
    if not _rows(con, "SELECT 1 FROM log.goal WHERE goal_id = $goal AND set_seq <= $known", asked):
        raise DataError(
            f"goal {goal!r} is not in the Host's history as known then",
            code="experience.goal_unknown",
        )
    decisions = _rows(
        con,
        "SELECT e.subject_id AS decision_id, json_extract_string(e.attrs, '$.kind') AS kind,"
        " json_extract_string(e.attrs, '$.chosen') AS chosen,"
        " CAST(json_extract(e.attrs, '$.alternatives') AS VARCHAR) AS alternatives,"
        " x.text AS reasons, make_timestamptz(e.recorded_us) AS made_at FROM log.event e"
        " JOIN log.event_type_name k ON k.type = e.type AND k.named = 'decision.made'"
        " LEFT JOIN log.event_text x ON x.seq = e.seq"
        " WHERE json_extract_string(e.attrs, '$.goal_id') = $goal AND e.seq <= $known"
        " QUALIFY row_number() OVER (PARTITION BY kind ORDER BY e.seq DESC) = 1"
        " ORDER BY e.seq DESC LIMIT $rows",
        asked | {"rows": _PART_ROWS},
    )
    lessons = _rows(
        con,
        "SELECT note_id, goal_id, hypothesis_id, method, text, recorded_at FROM note_index"
        " WHERE kind = 'lesson.recorded' AND seq <= $known AND text IS NOT NULL"
        " AND (goal_id = $goal OR method IN (SELECT method FROM class_member"
        " WHERE goal_id = $goal AND registered_seq <= $known))"
        " ORDER BY goal_id IS NOT DISTINCT FROM $goal DESC, seq DESC LIMIT $rows",
        asked | {"rows": _PART_ROWS},
    )
    newest = _rows(
        con,
        "SELECT trial_id FROM class_member WHERE goal_id = $goal AND registered_seq <= $known"
        " ORDER BY registered_seq DESC LIMIT 1",
        asked,
    )
    similar: list[dict[str, Any]] = []
    if newest:  # the other goals' trials of its method, newest first, from the whole class
        _class_now(con, newest[0]["trial_id"], "method", known)
        try:
            members = _members(con, known, limit=_PART_ROWS, other_than=goal, newest=True)
        finally:
            con.execute("DROP TABLE IF EXISTS class_now")
        similar = [_experience(member) for member in members]
        _conclude(con, similar, known)
    used, parts = 0, []
    for name, items in (("decisions", decisions), ("lessons", lessons), ("experiences", similar)):
        taken, omitted = [], 0
        for item in items:
            cost = _tokens(item)
            if used + cost <= budget:
                taken.append(item)
                used += cost
            else:
                omitted += 1
        parts.append({"part": name, "items": taken, "omitted": omitted})
    return {
        "goal": goal,
        "cutoff_us": cutoff_us,
        "known_seq": known,
        "budget": budget,
        "used": used,
        "parts": parts,
    }


def _experience(member: dict[str, Any]) -> dict[str, Any]:
    """A class member as a brief shows it: what was tried, how it stood then, its measures."""
    measures = {
        f"{m['metric']}@{m['metric_version']}" + ("" if m["dims"] == "{}" else m["dims"]): m[
            "value"
        ]
        for m in member["measures"]
    }
    keep = ("trial_id", "goal_id", "workspace_id", "method", "method_version", "params")
    return {key: member[key] for key in keep} | {
        "state": member["state_then"],
        "measures": measures,
    }


def _conclude(con: Any, experiences: list[dict[str, Any]], known: int) -> None:
    """Each experience's conclusions known then, from the index."""
    found: dict[str, list[str]] = {}
    for row in _rows(
        con,
        "SELECT subject_id, text FROM note_index WHERE kind = 'conclusion.drawn'"
        " AND subject_kind = 'trial' AND list_contains($trials, subject_id)"
        " AND seq <= $known AND text IS NOT NULL ORDER BY seq",
        {"trials": [e["trial_id"] for e in experiences], "known": known},
    ):
        found.setdefault(row["subject_id"], []).append(row["text"])
    for experience in experiences:
        experience["conclusions"] = found.get(experience["trial_id"], [])


def _tokens(item: dict[str, Any]) -> int:
    """An item's cost in the budget: its JSON's characters over `CHARS_PER_TOKEN`, rounded up."""
    return -(-len(json.dumps(item, default=str)) // CHARS_PER_TOKEN)


_SERIES = (  # each member's newest return series known then, one row per month
    "SELECT trial_id, registered_seq, unnest(CAST(j->'$.months' AS VARCHAR[])) AS month,"
    " unnest(CAST(j->'$.returns' AS DOUBLE[])) AS r FROM ("
    " SELECT c.trial_id, c.registered_seq, CAST(decode(b.body) AS JSON) AS j FROM class_now c"
    " JOIN log.artifact a ON a.trial_id = c.trial_id AND a.produced_seq <= $known"
    " JOIN log.definition_name tn ON tn.definition_id = a.type_id AND tn.family = 'artifact_type'"
    " AND tn.name = $type JOIN log.artifact_body b ON b.artifact_id = a.artifact_id"
    " QUALIFY row_number() OVER (PARTITION BY c.trial_id ORDER BY a.produced_seq DESC) = 1)"
)


def _effective_n(con: Any, known: int) -> dict[str, Any]:
    """The class's effective number of independent trials (return_series.effective_n): M members
    with a kept series, their mean pairwise correlation over common months estimated on the
    newest `return_series.SAMPLE` of them, pairs under `MIN_OVERLAP` common months left out."""
    found = _rows(
        con,
        f"WITH s AS ({_SERIES}),"
        " sample AS (SELECT * FROM s WHERE trial_id IN (SELECT trial_id FROM s GROUP BY trial_id"
        " ORDER BY max(registered_seq) DESC, trial_id LIMIT $sample)),"
        " pairs AS (SELECT corr(x.r, y.r) AS rho FROM sample x JOIN sample y"
        " ON x.month = y.month AND x.trial_id < y.trial_id GROUP BY x.trial_id, y.trial_id"
        " HAVING count(*) >= $overlap)"
        " SELECT (SELECT count(DISTINCT trial_id) FROM s) AS members_with_series,"
        " count(rho) AS pairs, avg(rho) AS mean_correlation FROM pairs WHERE isfinite(rho)",
        {
            "known": known,
            "type": return_series.TYPE,
            "sample": return_series.SAMPLE,
            "overlap": return_series.MIN_OVERLAP,
        },
    )[0]
    members = int(found["members_with_series"])
    return found | {
        "n": return_series.effective_n(found["mean_correlation"], members),
        "sample": min(members, return_series.SAMPLE),
        "min_overlap_months": return_series.MIN_OVERLAP,
        "method": return_series.METHOD,
    }


def _rows(con: Any, sql: str, parameters: Any = ()) -> list[dict[str, Any]]:
    result = con.execute(sql, parameters)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def compare(con: sqlite3.Connection, a: str, b: str) -> list[dict[str, Any]]:
    """Two artifacts by their measures, bodies present or not. A version both hold compares
    directly; with none shared, a's newest version is converted to b's when every version between
    them is equal or convertible; otherwise each version is its own row, never merged."""
    sides: dict[str, dict[tuple[str, str], dict[int, float | None]]] = {"a": {}, "b": {}}
    for side, artifact in (("a", a), ("b", b)):
        rows = con.execute(
            "SELECT l.name, m.metric_version, m.dims, m.value FROM measure m"
            " JOIN definition_lineage l ON l.definition_id = m.metric_id WHERE m.artifact_id = ?",
            (artifact,),
        )
        for metric, version, dims, value in rows:
            sides[side].setdefault((metric, dims), {})[version] = value
    out: list[dict[str, Any]] = []
    for key in sorted(set(sides["a"]) | set(sides["b"])):
        out += _compare_metric(con, key, sides["a"].get(key, {}), sides["b"].get(key, {}))
    return out


def _compare_metric(
    con: sqlite3.Connection,
    key: tuple[str, str],
    a: dict[int, float | None],
    b: dict[int, float | None],
) -> list[dict[str, Any]]:
    metric, dims = key
    bridge = None if (set(a) & set(b)) or not a or not b else _bridge(con, metric, max(a), max(b))
    if bridge is None:
        versions = sorted(set(a) | set(b))
        return [_compared(con, metric, v, dims, a.get(v), b.get(v), v) for v in versions]
    scale, offset = bridge
    va, vb = max(a), max(b)
    value = a[va]
    converted = None if value is None else scale * value + offset
    return [_compared(con, metric, vb, dims, converted, b[vb], va)]


def _definition(con: sqlite3.Connection, metric: str, version: int) -> Any:
    return con.execute(
        "SELECT relation, conversion FROM definition"
        " WHERE family = 'metric' AND name = ? AND version = ?",
        (metric, version),
    ).fetchone()


def _compared(
    con: sqlite3.Connection,
    metric: str,
    version: int,
    dims: str,
    va: float | None,
    vb: float | None,
    a_version: int,
) -> dict[str, Any]:
    found = _definition(con, metric, version)
    return {
        "metric": metric,
        "version": version,
        "dims": dims,
        "a": va,
        "b": vb,
        "diff": vb - va if va is not None and vb is not None else None,
        "a_version": a_version,
        "relation": found[0] if found else None,
        "conversion": found[1] if found else None,
    }


def _bridge(con: sqlite3.Connection, metric: str, src: int, dst: int) -> tuple[float, float] | None:
    """The affine map from version `src`'s values to `dst`'s, or None past a non-comparable step."""
    scale, offset = 1.0, 0.0
    for version in range(min(src, dst) + 1, max(src, dst) + 1):
        found = _definition(con, metric, version)
        if found is None or found[0] not in ("equal", "convertible"):
            return None
        step = json.loads(found[1]) if found[0] == "convertible" else {"scale": 1.0, "offset": 0.0}
        scale, offset = step["scale"] * scale, step["scale"] * offset + step["offset"]
    return (scale, offset) if src <= dst else (1 / scale, -offset / scale)


if __name__ == "__main__":
    # The reading process's entry: the product's DuckDB driver, installed in the storage port.
    from alphalattice.control.workspace_runtime import reader_threads
    from alphalattice.kernel import storage

    storage.install_driver(reader_threads.DRIVER)
    serve(Path(sys.argv[1]), sys.stdin.buffer, sys.stdout.buffer)  # noqa: TID251  # answer pipe
