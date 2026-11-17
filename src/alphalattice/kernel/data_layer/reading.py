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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO, Any

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.root import (
    catalog_matches,
    duckdb_session,
    schema_text,
    sql_path,
    statements,
)
from alphalattice.kernel.shared_kernel.waits import WaitTimeout, wait_until

READING_DEFERRED = "reading views: deferred (DuckDB's sqlite extension absent)"
READ_SECONDS = 60
"""How long a read of the views waits for the reader's answer: a stuck reader is ended past it,
so its caller, which holds the history's lock, never waits on it for ever. The experience read
of a 200,000-event log answers in 0.2 s and a cold reader starts in 0.4 s."""


def reading_session(root: Path, *, log: bool = True) -> Any | None:
    """DuckDB with `root`'s log attached read-only and views.sql defined, in UTC; None when
    `duckdb_session` defers (READING_DEFERRED). The caller closes it. Without `log`, only the
    catalog and the views that read no log: a process that opens the log through Python's
    sqlite3 opens it with no second SQLite library, whose closing descriptor would release the
    process's locks on the file."""
    connection = duckdb_session(root)
    if connection is None:
        return None
    machine = root / "machine"
    try:
        connection.execute("SET TimeZone = 'UTC'").complete()
        if log:
            path = sql_path(machine / "log.sqlite")
            connection.execute(f"ATTACH '{path}' AS log (TYPE SQLITE, READ_ONLY)").complete()
        lake = (machine / "catalog.sqlite").exists()
        if lake:
            data, catalog = sql_path(root / "data"), sql_path(machine / "catalog.sqlite")
            connection.execute(
                f"ATTACH 'ducklake:sqlite:{catalog}' AS lake (DATA_PATH '{data}/', READ_ONLY)"
            ).complete()
            catalog_matches(connection)
        for statement in statements(schema_text("views.sql")):
            if (lake or "lake." not in statement) and (log or "log." not in statement):
                connection.execute(statement).complete()
    except BaseException:
        connection.close()
        raise
    return connection


class LogReader:
    """The reading views of the data root's log, read in a process of their own (`serve`) that
    alone opens the log in DuckDB: started at the first read, restarted when it has ended, and
    ended by `close`. A read takes `experience` or `goal_notes` by name with their filters and
    returns their rows as those functions do. One caller at a time: its owner, the history, reads
    it under its own lock."""

    def __init__(self, root: Path) -> None:
        """A reader of `root`'s log views; its process starts at the first read."""
        self.root = root
        self._child: subprocess.Popen[bytes] | None = None
        self._answers = ThreadPoolExecutor(max_workers=1, thread_name_prefix="log-reader")

    def read(self, name: str, **filters: Any) -> list[dict[str, Any]]:
        """The rows of the view read `name` (experience or goal_notes) with `filters`; refused as
        `data_layer.history_unavailable` when the reader cannot answer, or has not within
        READ_SECONDS, when its process is ended and the next read starts another."""
        child = self._child
        if child is None or child.poll() is not None:
            child = self._child = subprocess.Popen(
                [sys.executable, "-m", __name__, str(self.root)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        return _ask(child, (name, filters), self._answers, self.root)

    def close(self) -> None:
        """End the reader's process: its input closed, it returns; one that has not within ten
        seconds is ended."""
        child, self._child = self._child, None
        if child is not None:
            _end(child, kill=False)
        self._answers.shutdown(wait=False)


def _ask(
    child: subprocess.Popen[bytes], request: Any, answers: ThreadPoolExecutor, root: Path
) -> list[dict[str, Any]]:
    """One read's exchange with the reader's process `child`, its answer awaited on `answers`:
    refused as `data_layer.history_unavailable` when the process cannot answer, or has not within
    READ_SECONDS, when it is ended and reaped, so the next read starts another."""
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
    if not ok:
        raise DataError(
            f"the reading views at {root} did not answer ({value})",
            code="data_layer.history_unavailable",
        )
    return list(value)


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
    session = reading_session(root)
    reads = {"experience": experience, "goal_notes": goal_notes}
    try:
        for name, filters in _requests(requests):
            try:
                _send(answers, (True, reads[name](session, **filters)))
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
    serve(Path(sys.argv[1]), sys.stdin.buffer, sys.stdout.buffer)  # noqa: TID251  # answer pipe
