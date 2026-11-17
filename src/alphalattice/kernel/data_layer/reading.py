"""The data root's reading surface: views.sql in DuckDB over the log, and two artifacts compared.

`reading_session` is what the Host opens at start: DuckDB with the log attached read-only and every
view and macro of views.sql defined (span, experience, approach_spread, search_size and the rest),
the world's point-in-time macros only when the catalog exists. Nothing in it is stored; each view is
computed on read, and `experience` reads profiles and measures only, so it is whole whether or not
any artifact body still exists. `compare` reads `measure` through SQLite by its primary key, where
DuckDB's SQLite scan would read every measure.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from alphalattice.kernel.data_layer.root import duckdb_session, schema_text, sql_path, statements

READING_DEFERRED = "reading views: deferred (DuckDB's sqlite extension absent)"


def reading_session(root: Path) -> Any | None:
    """DuckDB with `root`'s log attached read-only and views.sql defined, in UTC; None when
    `duckdb_session` defers (READING_DEFERRED). The caller closes it."""
    connection = duckdb_session(root)
    if connection is None:
        return None
    machine = root / "machine"
    try:
        connection.execute("SET TimeZone = 'UTC'").complete()
        log = sql_path(machine / "log.sqlite")
        connection.execute(f"ATTACH '{log}' AS log (TYPE SQLITE, READ_ONLY)").complete()
        lake = (machine / "catalog.sqlite").exists()
        if lake:
            data, catalog = sql_path(root / "data"), sql_path(machine / "catalog.sqlite")
            connection.execute(
                f"ATTACH 'ducklake:sqlite:{catalog}' AS lake (DATA_PATH '{data}/', READ_ONLY)"
            ).complete()
        for statement in statements(schema_text("views.sql")):
            if lake or "lake." not in statement:
                connection.execute(statement).complete()
    except BaseException:
        connection.close()
        raise
    return connection


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
