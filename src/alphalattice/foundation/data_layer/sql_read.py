"""An agent's own read-only SQL over its history (doc 09, how an agent queries, step 3).

The reading process (`reading.serve`) answers it from a session of its own: the log's SQLite and
the DuckLake catalog attached READ_ONLY with every view of views.sql, then every file and network
read turned off (`enable_external_access`) and the configuration locked, so no statement reaches
a file, the network or another database, and none turns either back on. One statement is read, and
only when DuckDB's own parser classifies it SELECT. Its surface is the views and table macros of
views.sql, the stable interface, never the log's or the catalog's tables (a Host-only column such
as workspace.folder among them): every relation it names, read from DuckDB's own parse tree, is
one of those views, one of its own CTEs or a row generator, and it calls no function that reads
the engine's settings or catalog. Its answer is cut at `ROWS` rows and runs under the reader's
deadline (`reading.READ_SECONDS`). DuckDB's own words on a refused statement are withheld, its
error's kind alone given, and an answer holding a path of this machine is refused by name.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

from alphalattice.capabilities.data.errors import DataError

ROWS: Final = 1000
"""The most rows a statement answers with; a longer answer is cut there and says so."""
CHARACTERS: Final = 8192
"""The longest statement read."""
_LOCKED: Final = ("SET enable_external_access = false", "SET lock_configuration = true")
_GENERATORS: Final = frozenset({"range", "generate_series", "unnest"})
"""Table functions besides the views' macros a statement may read: they make rows, read none."""
_ENGINE_READS: Final = ("current_setting", "getvariable", "duckdb_", "pragma_")
"""Scalar functions, by name or prefix, that read the engine's settings or catalog."""
_SURFACE: Final = (
    "SELECT (SELECT list(view_name) FROM duckdb_views() WHERE database_name = current_database()"
    " AND NOT internal), (SELECT list(function_name) FROM duckdb_functions()"
    " WHERE database_name = current_database() AND function_type = 'table_macro')"
)
"""The surface's names in one row: the session's own views, and its table macros."""


def lock(session: Any) -> Any:
    """Turn `session`'s file and network reads off and lock its configuration, once its log and
    catalog are attached; returns it."""
    for statement in _LOCKED:
        session.execute(statement).complete()
    return session


def answer(con: Any, *, statement: str, root: Path, rows: int = ROWS) -> dict[str, Any]:
    """The columns and first `rows` rows of one SELECT on the locked session `con`, and whether
    more were cut. Refused by name: `experience.sql_invalid` unless it is one statement of at most
    `CHARACTERS` that DuckDB classifies SELECT; `experience.sql_off_the_surface` when it reads
    past the views' surface (`_on_the_surface`); `experience.sql_refused` when DuckDB refuses or
    fails it; `experience.sql_names_a_path` when its answer holds a path of this machine."""
    import duckdb  # lazy: the reading process holds the data extra

    if not isinstance(statement, str) or not 0 < len(statement.strip()) <= CHARACTERS:
        raise DataError(
            f"a statement is 1 to {CHARACTERS} characters", code="experience.sql_invalid"
        )
    try:
        parsed = con.extract_statements(statement)
    except duckdb.Error as error:
        raise _refused(error, "experience.sql_invalid") from None
    if len(parsed) != 1 or parsed[0].type is not duckdb.StatementType.SELECT:
        kinds = ", ".join(each.type.name for each in parsed) or "none"
        raise DataError(f"one SELECT is read; this is {kinds}", code="experience.sql_invalid")
    _on_the_surface(con, statement)
    try:
        result = con.execute(statement)
        columns = [column[0] for column in result.description]
        found = result.fetchmany(rows + 1)
    except duckdb.Error as error:
        raise _refused(error, "experience.sql_refused") from None
    table = [[_plain(value) for value in row] for row in found[:rows]]
    if _names_a_path([columns, table], _needles(root)):
        raise DataError(
            "the answer holds a path of this machine, which is never given",
            code="experience.sql_names_a_path",
        )
    return {"columns": columns, "rows": table, "truncated": len(found) > rows}


def _on_the_surface(con: Any, statement: str) -> None:
    """Refused as `experience.sql_off_the_surface` unless every relation the SELECT names is a
    view of views.sql or its own CTE, every table function a views' macro or a row generator, and
    no function reads the engine's settings or catalog; read from DuckDB's parse tree."""
    named_views, named_macros = con.execute(_SURFACE).fetchone()
    views, macros = set(named_views or ()), set(named_macros or ())
    (tree,) = con.execute("SELECT json_serialize_sql(?)", [statement]).fetchone()
    parsed = json.loads(tree)
    if parsed.get("error"):
        raise DataError(
            "its form is not one this surface reads", code="experience.sql_off_the_surface"
        )
    ctes = {entry["key"] for found in _nodes(parsed, "cte_map") for entry in found.get("map", [])}
    for node in _nodes(parsed, "type"):
        kind, name = node["type"], ""
        if kind == "BASE_TABLE":
            name = node.get("table_name", "")
            named = node.get("catalog_name") or node.get("schema_name")
            allowed = not named and name in views | ctes
        elif kind == "TABLE_FUNCTION":
            name = (node.get("function") or {}).get("function_name", "")
            allowed = name in macros | _GENERATORS
        elif kind == "SHOW_REF":
            name, allowed = "SHOW, DESCRIBE or SUMMARIZE", False
        elif node.get("class") == "FUNCTION":
            name = node.get("function_name", "")
            allowed = not name.startswith(_ENGINE_READS)
        else:
            continue
        if not allowed:
            raise DataError(
                f"{name} is off the surface: read the history's views, their macros, your own"
                " CTEs and range, generate_series or unnest",
                code="experience.sql_off_the_surface",
            )


def _nodes(value: Any, key: str) -> Any:
    """Every mapping in a parse tree holding `key` (a string for `type`), in document order."""
    if isinstance(value, dict):
        if key in value and (key != "type" or isinstance(value[key], str)):
            yield value[key] if key == "cte_map" else value
        for item in value.values():
            yield from _nodes(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _nodes(item, key)


def _refused(error: Exception, code: str) -> DataError:
    """DuckDB's refusal by its kind alone: its words may name anything the engine holds."""
    if type(error).__name__ == "PermissionException":
        return DataError("file and network reads are off for an agent's statement", code=code)
    return DataError(
        f"DuckDB refused it ({type(error).__name__}); its words are withheld", code=code
    )


def _needles(root: Path) -> list[str]:
    """This machine's paths an answer must not hold, lower-cased with forward slashes: the data
    root, the home, the working and temporary folders and the interpreter's."""
    places = {
        root.resolve(),
        Path.home(),
        Path.cwd(),
        Path(tempfile.gettempdir()),
        Path(sys.prefix),
        Path(sys.base_prefix),
    }
    named = {place for place in places if place != Path(place.anchor)}  # a bare drive names none
    return sorted(str(place).replace(os.sep, "/").lower().rstrip("/") for place in named)


def _names_a_path(value: Any, needles: list[str]) -> bool:
    if isinstance(value, str):
        text = value.replace("\\", "/").lower()
        return any(needle in text for needle in needles)
    if isinstance(value, dict):
        return any(_names_a_path(item, needles) for item in (*value.keys(), *value.values()))
    if isinstance(value, list | tuple):
        return any(_names_a_path(item, needles) for item in value)
    return False


def _plain(value: Any) -> Any:
    """A cell in JSON's types: times in ISO form, other scalars DuckDB gives as their text."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, bytes | bytearray | memoryview):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return str(value)  # Decimal, UUID and DuckDB's other scalars
