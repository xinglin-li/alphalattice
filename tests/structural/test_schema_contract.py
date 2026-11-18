"""The data model's contract: kernel/data_layer/schema/contract.json against the DDL beside it.

The registry declares, for every table in log.sql, catalog.sql and index.sql, its key and where
each key part comes from, its one upward link, its home, each time column's clock, each status
column's states and moves, and each version column's binding. This test reads the DDL through the
engines that run it (SQLite for the log and the index, DuckDB for the catalog), finds every way the
two disagree, and requires exactly the registry's known breaks: a new break fails by its rule, and
a known break whose rule now holds fails until its handle is removed. The planted cases break each
rule on purpose and name what the check then reports.
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from functools import cache
from pathlib import Path
from typing import Any

import pytest

TEST_SCOPE = "whole-tree"

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "src" / "alphalattice" / "kernel" / "data_layer" / "schema"
FILES = ("log", "catalog", "index")
HOMES = {"log", "definition", "projection", "body", "working", "cache", "index", "world", "derived",
         "results"}  # fmt: skip
REBUILT = {"cache", "index", "derived"}  # a recomputed home may be keyed by a content address
STAMP = "TIMESTAMP WITH TIME ZONE"
CLOCKS = {
    "utc_us": {"INTEGER"},
    "log_seq": {"INTEGER"},
    "utc_timestamp": {STAMP},
    "source_clock": {STAMP},
    "market_date": {"DATE", "TEXT"},
    "iso_text": {"TEXT"},
}
TIME = re.compile(r"(_us|_seq|_at|_asof|_date|_session)$|^(seq|day|as_of|valid_from|valid_to)$")
STATUS = {"status", "state", "standing", "lifecycle", "body_state"}
VERSION = re.compile(r"(^|_)version$")
OWN_ID = {"uuid7", "seq", "link", "singleton"}
BREAK_ID = {"digest", "caller", "packed"}
DIGEST_NAME = re.compile(r"(_key|_digest|_hash)$")
KEY_COMMENT = re.compile(r"^-- key: \(([^)]*)\)[^\n]*\n(?:--[^\n]*\n)*CREATE TABLE (\S+) \(", re.M)
CHECK_IN = re.compile(r"CHECK \((\w+) IN \(([^)]*)\)\)")


def _texts() -> dict[str, str]:
    return {name: (SCHEMA / f"{name}.sql").read_text(encoding="utf-8") for name in FILES}


def _registry() -> dict[str, Any]:
    return json.loads((SCHEMA / "contract.json").read_text(encoding="utf-8"))


def _statements(text: str) -> list[str]:
    from alphalattice.kernel.data_layer.root import statements

    return list(statements(text))


def _sqlite_tables(prefix: str, text: str) -> dict[str, dict[str, Any]]:
    """Each table as SQLite builds it; a virtual table is named and keyed by its entry alone."""
    with closing(sqlite3.connect(":memory:")) as con:
        kept = [s for s in _statements(text) if not s.lstrip().startswith("CREATE VIRTUAL")]
        con.executescript("\n".join(s + ";" for s in kept))
        tables: dict[str, dict[str, Any]] = {
            f"{prefix}.{name}": {"virtual": True}
            for name in re.findall(r"^CREATE VIRTUAL TABLE (\w+)", text, re.M)
        }
        triggers: dict[str, dict[str, str]] = {}
        for trigger, table, sql in con.execute(
            "SELECT name, tbl_name, sql FROM sqlite_master WHERE type = 'trigger'"
        ):
            triggers.setdefault(table, {})[trigger] = sql
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
        for name, sql in rows.fetchall():
            info = con.execute(f"PRAGMA table_info({name})").fetchall()
            tables[f"{prefix}.{name}"] = {
                "key": [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]],
                "columns": {row[1]: {"type": row[2], "default": row[4]} for row in info},
                "links": {
                    row[3]: row[2] for row in con.execute(f"PRAGMA foreign_key_list({name})")
                },
                "checks": {
                    col: sorted(re.findall(r"'([^']*)'", values))
                    for col, values in CHECK_IN.findall(sql)
                },
                "triggers": triggers.get(name, {}),
            }
    return tables


def _catalog_tables(text: str) -> dict[str, dict[str, Any]]:
    """Each table as DuckDB builds it; DuckLake enforces no keys, so its key comment is the key."""
    import duckdb

    with closing(duckdb.connect()) as con:
        for statement in _statements(text):
            if statement.lstrip().upper().startswith("CREATE"):
                con.execute(statement)
        said = {
            table: [part.strip() for part in key.split(",")]
            for key, table in KEY_COMMENT.findall(text)
        }
        columns = (
            "SELECT column_name, data_type, is_nullable, column_default"
            " FROM information_schema.columns"
        )
        tables = {}
        for schema, name in con.execute(
            "SELECT table_schema, table_name FROM information_schema.tables"
        ).fetchall():
            rows = con.execute(
                f"{columns} WHERE table_schema = ? AND table_name = ?", [schema, name]
            ).fetchall()
            tables[f"catalog.{schema}.{name}"] = {
                "said_key": said.get(f"{schema}.{name}"),
                "columns": {
                    c: {"type": t, "null": n == "YES", "default": d} for c, t, n, d in rows
                },
                "links": {},
                "checks": {},
                "triggers": {},
            }
    return tables


def ddl(texts: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Every table of the three files, keyed `file.table`."""
    return {
        **_sqlite_tables("log", texts["log"]),
        **_catalog_tables(texts["catalog"]),
        **_sqlite_tables("index", texts["index"]),
    }


@cache
def _real() -> tuple[str, str]:
    return json.dumps(ddl(_texts())), json.dumps(_registry())


def _up(entry: dict[str, Any]) -> list[str]:
    """The one upward link: a column, or the columns of one composite link."""
    up = entry.get("up")
    return [] if up is None else [up] if isinstance(up, str) else list(up)


def _ancestors(name: str, tables: dict[str, Any], entries: dict[str, Any]) -> set[str]:
    """The tables a record's upward chain reaches."""
    seen, at = set(), name
    while at in entries and (up := _up(entries[at])):
        target = tables[at].get("links", {}).get(up[0])
        if target is None or f"{at.split('.')[0]}.{target}" in seen:
            break
        at = f"{at.split('.')[0]}.{target}"
        seen.add(at)
    return seen


def _keys(name: str, table: dict[str, Any], entry: dict[str, Any]) -> Iterator[str]:
    key, columns = entry.get("key", []), table["columns"]
    if "key" in table and table["key"] != key:
        yield f"key: {name}: the DDL keys {table['key']}, the registry {key}"
    if name.startswith("catalog.") and table.get("said_key") is None:
        yield f"key: {name}: the DDL states no key, and DuckLake enforces none"
    elif name.startswith("catalog.") and table["said_key"] != key:
        yield f"key: {name}: the DDL's key comment says {table['said_key']}, the registry {key}"
    for part in key:
        if part not in columns:
            yield f"key: {name}.{part}: a key part with no column"
        elif columns[part].get("null"):
            yield f"key: {name}.{part}: a nullable key part"


def _ids(name: str, table: dict[str, Any], entry: dict[str, Any]) -> Iterator[str]:
    for part in entry.get("key", []):
        kind = entry.get("id", {}).get(part, "")
        head, _, reason = (s.strip() for s in kind.partition(":"))
        if not kind:
            yield f"id: {name}.{part}: no declared source"
        elif head in BREAK_ID:
            yield f"id: {name}.{part}: a {head} id"
        elif head == "content" and entry.get("home") not in REBUILT:
            yield f"id: {name}.{part}: a content address keys a record that is not recomputed"
        elif head not in OWN_ID | {"standard", "content"} or (
            head in {"standard", "content"} and not reason
        ):
            yield f"id: {name}.{part}: '{kind}' is not a source with its reason"
        elif head in {"uuid7", "seq", "standard"} and DIGEST_NAME.search(part):
            yield f"id: {name}.{part}: a digest-named part declared as {head}"


def _links(
    name: str, table: dict[str, Any], entry: dict[str, Any], above: set[str]
) -> Iterator[str]:
    up, links, columns = _up(entry), set(entry.get("links", [])), table["columns"]
    yield from (
        f"up: {name}.{column}: an upward link with no column"
        for column in up
        if column not in columns
    )
    for column in sorted(table.get("links", {}).keys() - links - set(up)):
        yield f"up: {name}.{column}: a link the registry does not declare"
    for column, target in table.get("links", {}).items():
        edge = entry.get("edges", {}).get(column)
        if column not in up and not edge and f"{name.split('.')[0]}.{target}" in above:
            reach = "which its upward chain already reaches"
            yield f"up: {name}.{column}: a second upward link, to {target}, {reach}"
    for column in sorted(entry.get("edges", {}).keys() - table.get("links", {}).keys()):
        yield f"up: {name}.{column}: a declared edge with no link"
    if entry.get("home") not in HOMES:
        yield f"home: {name}: '{entry.get('home')}' is not a home"
    if entry.get("copy_of"):
        yield f"home: {name}: a second home of {entry['copy_of']}'s facts, kept as a table"


def _times(name: str, table: dict[str, Any], entry: dict[str, Any]) -> Iterator[str]:
    times, columns = entry.get("times", {}), table["columns"]
    for column in sorted(c for c in columns if TIME.search(c)):
        clock, kind = times.get(column), columns[column]["type"]
        if clock is None:
            yield f"time: {name}.{column}: no declared clock"
        elif clock not in CLOCKS:
            yield f"time: {name}.{column}: '{clock}' is not a clock"
        elif kind not in CLOCKS[clock]:
            yield f"time: {name}.{column}: a {clock} column typed {kind}"
        elif clock == "iso_text":
            yield f"time: {name}.{column}: a time kept as text on its writer's clock"
    yield from (
        f"time: {name}.{c}: a declared clock with no column" for c in times.keys() - columns.keys()
    )


def _statuses(name: str, table: dict[str, Any], entry: dict[str, Any]) -> Iterator[str]:
    for column in sorted(c for c in table["columns"] if c in STATUS):
        rule = entry.get("status", {}).get(column)
        if rule is None:
            yield f"status: {name}.{column}: no declared states"
            continue
        states, terminal, moves = set(rule["states"]), set(rule["terminal"]), rule["moves"]
        checked = table["checks"].get(column)
        if checked is None:
            yield f"status: {name}.{column}: the DDL checks no states"
        elif set(checked) != states:
            allows = f"the DDL allows {sorted(checked)}, the registry {sorted(states)}"
            yield f"status: {name}.{column}: {allows}"
        reached = set(moves).union(*map(set, moves.values())) if moves else set()
        if not terminal | reached <= states:
            yield f"status: {name}.{column}: a move or terminal state outside its states"
        if terminal & {state for state, targets in moves.items() if targets}:
            yield f"status: {name}.{column}: a terminal state moves"
        guard = rule.get("enforced_by")
        if not guard:
            yield f"status: {name}.{column}: its moves are declared, not enforced"
        elif not _guards(name, table["triggers"], guard, column, rule):
            yield f"status: {name}.{column}: enforced by '{guard}', which refuses other moves"


def _refusal(sql: str | None, event: str, table: str) -> str | None:
    """The WHEN clause of a trigger on this table whose whole body refuses, else None."""
    shape = (
        rf"CREATE TRIGGER \w+ BEFORE {event} ON {table}\s+WHEN (?P<when>.+?)\s+"
        r"BEGIN\s+SELECT RAISE\(ABORT, '[^']*'\);\s*END"
    )
    found = re.fullmatch(shape, sql.strip(), re.S) if sql else None
    return found and found["when"]


def _refuses(prefix: str, when: str, values: dict[str, Any], seed: tuple[str, ...] = ()) -> bool:
    """Whether the clause refuses these OLD and NEW values, run in the file's own schema."""
    return _run(prefix, when, tuple(sorted(values.items())), seed)


@cache
def _run(
    prefix: str, when: str, values: tuple[tuple[str, Any], ...], seed: tuple[str, ...]
) -> bool:
    clause = re.sub(r"\b(OLD|NEW)\.(\w+)", lambda m: f":{m[1].lower()}_{m[2]}", when)
    with closing(sqlite3.connect(":memory:")) as con:
        con.deserialize(_built(prefix))
        for statement in seed:
            con.execute(statement)
        query = f"SELECT CASE WHEN {clause} THEN 1 ELSE 0 END"
        return bool(con.execute(query, dict(values)).fetchone()[0])


@cache
def _built(prefix: str) -> bytes:
    """The file's schema, built once from the tree, for running a trigger's clause."""
    kept = [s for s in _statements(_texts()[prefix]) if not s.lstrip().startswith("CREATE VIRTUAL")]
    with closing(sqlite3.connect(":memory:")) as con:
        con.executescript("\n".join(s + ";" for s in kept))
        return con.serialize()


def _guards(name: str, triggers: dict[str, str], guard: str, column: str, rule: dict) -> bool:
    """The named trigger on this table refuses exactly the moves the registry does not declare:
    each change of the column to another state, run through the trigger's own clause."""
    prefix, table = name.split(".", 1)
    sql = triggers.get(guard.removeprefix("trigger ")) if guard.startswith("trigger ") else None
    when = _refusal(sql, rf"UPDATE OF (?:[\w, ]*\b)?{column}\b[\w, ]*", table)
    if not when or set(re.findall(r"\b(?:OLD|NEW)\.(\w+)", when)) != {column}:
        return False
    return all(
        _refuses(prefix, when, {f"old_{column}": old, f"new_{column}": new})
        == (new != old and new not in rule["moves"].get(old, []))
        for old in rule["states"]
        for new in rule["states"]
    )


# Three lineages: alpha holds version 1 in its family, beta 1 and 2, and another family's alpha
# holds 1 and 2; a trigger admits alpha 1, beta 1 and beta 2, and refuses alpha 0, alpha 2, beta 3
# and gamma, so an inequality or another lineage's version is caught.
LINEAGES = (("id-alpha", "{family}", "alpha", 1), ("id-beta", "{family}", "beta", 2),
            ("id-other", "{family}-other", "alpha", 2))  # fmt: skip
CASES = (("alpha", 0, True), ("alpha", 1, False), ("alpha", 2, True), ("beta", 1, False),
         ("beta", 2, False), ("beta", 3, True), ("gamma", 1, True))  # fmt: skip


def _seed(family: str) -> tuple[str, ...]:
    rows = []
    for at, (lineage, kin, label, top) in enumerate(LINEAGES, 1):
        kin = kin.format(family=family)
        rows += [
            f"INSERT INTO definition_lineage VALUES ('{lineage}', '{kin}', '{label}', {at}, {at})",
            f"INSERT INTO definition_name VALUES ('{kin}', '{label}', '{lineage}', {at})",
        ]
        rows += [
            "INSERT INTO definition_version (definition_id, version, status, relation, spec, text,"
            f" owner, registered_seq, status_seq) VALUES ('{lineage}', {v}, 'standard',"
            f" {'NULL' if v == 1 else repr('equal')}, '{{}}', 't', 'o', {at}, {at})"
            for v in range(1, top + 1)
        ]
    return tuple(rows)


def _registers(name: str, triggers: dict[str, str], binding: str, column: str) -> bool:
    """The named trigger on this table refuses an insert exactly when no definition registers its
    version, run through the trigger's own clause over seeded lineages. The binding names how the
    insert names its definition: `trigger T: family F, name C` or `trigger T: definition_id C`."""
    prefix, table = name.split(".", 1)
    head, _, names = binding.partition(":")
    said = dict(part.split() for part in names.split(",") if part.strip())
    when = _refusal(triggers.get(head.removeprefix("trigger ").strip()), "INSERT", table)
    by = said.get("name") or said.get("definition_id")
    if not when or not by or set(re.findall(r"\bNEW\.(\w+)", when)) != {column, by}:
        return False
    seed = _seed(said.get("family", "planted"))
    return all(
        _refuses(prefix, when, {f"new_{by}": label if "name" in said else f"id-{label}",
                                f"new_{column}": version}, seed) == refused
        for label, version, refused in CASES
    )  # fmt: skip


def _versions(name: str, table: dict[str, Any], entry: dict[str, Any]) -> Iterator[str]:
    columns, triggers = table["columns"], table["triggers"]
    for column in sorted(c for c in columns if VERSION.search(c)):
        binding = entry.get("versions", {}).get(column, "")
        trigger = binding.removeprefix("trigger ") if binding.startswith("trigger ") else None
        if not binding:
            yield f"version: {name}.{column}: no declared binding"
        elif trigger is not None and not _registers(name, triggers, binding, column):
            admits = "which does not refuse exactly the unregistered versions"
            yield f"version: {name}.{column}: bound by {binding.split(':')[0]}, {admits}"
        elif (
            binding == "definition"
            and not name.startswith("catalog.")
            and (table["links"].get(column) != "definition_version")
        ):
            unlinked = "a definition version with no link to definition_version"
            yield f"version: {name}.{column}: {unlinked}"
        elif trigger is None and binding.split(":")[0] not in {"definition", "own"}:
            yield f"version: {name}.{column}: '{binding}' is not a binding"
        if columns[column].get("default") is not None:
            yield f"version: {name}.{column}: a version with a default"


def problems(tables: dict[str, dict[str, Any]], registry: dict[str, Any]) -> set[str]:
    """Every way the DDL and the registry disagree, each as `rule: table.column: what`."""
    entries = registry["tables"]
    found = {f"coverage: {name}: a table with no entry" for name in tables.keys() - entries.keys()}
    found |= {
        f"coverage: {name}: an entry with no table" for name in entries.keys() - tables.keys()
    }
    for name in sorted(tables.keys() & entries.keys()):
        table, entry = tables[name], entries[name]
        if table.get("virtual"):
            continue
        above = _ancestors(name, tables, entries)
        for rule in (_keys, _ids, _times, _statuses, _versions):
            found.update(rule(name, table, entry))
        found.update(_links(name, table, entry, above))
    return found


def known(registry: dict[str, Any]) -> set[str]:
    """Today's breaks, each under its handle."""
    return {problem for handle in registry["known"].values() for problem in handle["problems"]}


@pytest.mark.proves("data_layer.the_schema_meets_its_contract")
def test_the_schema_meets_its_registry_with_only_its_known_breaks() -> None:
    """Every table is registered, and the DDL breaks exactly the rules the registry names as known;
    a known break whose rule now holds is named, so its handle leaves with the change mending it."""
    tables, registry = map(json.loads, _real())
    found, expected = problems(tables, registry), known(registry)
    assert {name.split(".")[0] for name in tables} == set(FILES)
    assert sorted(found - expected) == [], "new breaks of the data model's rules"
    assert sorted(expected - found) == [], "known breaks that now hold: remove their handles"
    assert all(handle["says"] and handle["problems"] for handle in registry["known"].values())


@pytest.mark.proves("data_layer.the_schema_meets_its_contract")
@pytest.mark.parametrize(
    ("changed", "runs"),
    [
        pytest.param("src/alphalattice/kernel/data_layer/schema/log.sql", True, id="ddl"),
        pytest.param(
            "src/alphalattice/kernel/data_layer/schema/contract.json", True, id="registry"
        ),
        pytest.param("src/alphalattice/kernel/data_layer/log.py", False, id="data-layer-code"),
        pytest.param("README.md", False, id="elsewhere"),
    ],
)
def test_the_commit_stage_runs_the_contract_for_a_schema_change_alone(
    changed: str, runs: bool
) -> None:
    """The merge lane admits the contract when the DDL or its registry changes, and leaves it to
    the nightly otherwise."""
    from devtools.testing import sizes

    text = (ROOT / "config/registries/test-sizes.json").read_text("utf-8")
    policy = sizes.validate_policy(json.loads(text))
    test = test_the_schema_meets_its_registry_with_only_its_known_breaks.__name__
    node = f"{Path(__file__).relative_to(ROOT).as_posix()}::{test}"
    assert sizes.included(node, "medium", "merge", policy, frozenset({changed})) is runs


def _with_column(table: str, column: str) -> Callable[[dict[str, str]], None]:
    def plant(texts: dict[str, str]) -> None:
        head = f"CREATE TABLE {table} (\n"
        texts["log"] = texts["log"].replace(head, f"{head}  {column},\n", 1)

    return plant


def _replace(old: str, new: str, file: str = "log") -> Callable[[dict[str, str]], None]:
    def plant(texts: dict[str, str]) -> None:
        assert old in texts[file]
        texts[file] = texts[file].replace(old, new, 1)

    return plant


def _set(path: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def plant(registry: dict[str, Any]) -> None:
        *head, last = path
        at = registry
        for part in head:
            at = at[part]
        if value is None:
            del at[last]
        else:
            at[last] = value

    return plant


BOOK = ("tables", "log.book")
NEW_TABLE = "\nCREATE TABLE planted (planted_id TEXT PRIMARY KEY) STRICT, WITHOUT ROWID;\n"
# The plants' book.state is guarded by a real trigger, so a status plant breaks one thing.
GUARD = """
CREATE TRIGGER book_state_moves BEFORE UPDATE OF state ON book
WHEN OLD.state = 'retired' AND NEW.state <> OLD.state
BEGIN SELECT RAISE(ABORT, 'a retired book never moves'); END;
"""
# case: (DDL plant, registry plants, the one problem it adds)
DdlPlant = Callable[[dict[str, str]], None] | None
RegistryPlant = Callable[[dict[str, Any]], None]
PLANTED: dict[str, tuple[DdlPlant, list[RegistryPlant], str]] = {
    "unregistered table": (
        lambda texts: texts.__setitem__("log", texts["log"] + NEW_TABLE), [],
        "coverage: log.planted: a table with no entry"),
    "stale entry": (
        None,
        [lambda r: r["tables"].update({"log.planted": copy.deepcopy(r["tables"]["log.book"])})],
        "coverage: log.planted: an entry with no table"),
    "key differs": (
        None, [_set((*BOOK, "key"), ["book_id", "workspace_id"]),
               _set((*BOOK, "id", "workspace_id"), "link")],
        "key: log.book: the DDL keys ['book_id'], the registry ['book_id', 'workspace_id']"),
    "digest declared meaningless": (
        None, [_set(("tables", "log.recipe_upstream", "id", "recipe_key"), "uuid7")],
        "id: log.recipe_upstream.recipe_key: a digest-named part declared as uuid7"),
    "id without its reason": (
        None, [_set(("tables", "log.task_step", "id", "step"), "standard")],
        "id: log.task_step.step: 'standard' is not a source with its reason"),
    "undeclared link": (
        None, [_set((*BOOK, "up"), None)],
        "up: log.book.workspace_id: a link the registry does not declare"),
    "link to an ancestor": (
        _with_column("wait", "workspace_id TEXT REFERENCES workspace(workspace_id)"),
        [_set(("tables", "log.wait", "links"), ["event_type_id", "workspace_id"])],
        "up: log.wait.workspace_id: a second upward link, to workspace, which its upward chain"
        " already reaches"),
    "renamed link to an ancestor": (
        _with_column("wait", "root_owner_id TEXT REFERENCES workspace(workspace_id)"),
        [_set(("tables", "log.wait", "links"), ["event_type_id", "root_owner_id"])],
        "up: log.wait.root_owner_id: a second upward link, to workspace, which its upward chain"
        " already reaches"),
    "edge with no link": (
        None, [_set((*BOOK, "edges"), {"planted_id": "another book"})],
        "up: log.book.planted_id: a declared edge with no link"),
    "second home": (
        None, [_set((*BOOK, "copy_of"), "log.workspace")],
        "home: log.book: a second home of log.workspace's facts, kept as a table"),
    "undeclared clock": (
        None, [_set((*BOOK, "times", "activated_seq"), None)],
        "time: log.book.activated_seq: no declared clock"),
    "time as text": (
        _with_column("book", "planted_at TEXT"), [_set((*BOOK, "times", "planted_at"), "iso_text")],
        "time: log.book.planted_at: a time kept as text on its writer's clock"),
    "time typed wrong": (
        _with_column("book", "planted_us TEXT"), [_set((*BOOK, "times", "planted_us"), "utc_us")],
        "time: log.book.planted_us: a utc_us column typed TEXT"),
    "status outside its states": (
        _replace("(state IN ('active', 'retired'))", "(state IN ('active', 'paused', 'retired'))"),
        [],
        "status: log.book.state: the DDL allows ['active', 'paused', 'retired'], the registry"
        " ['active', 'retired']"),
    "terminal state moves": (  # its guard refuses nothing, as its declared moves now allow
        _replace("WHEN OLD.state = 'retired' AND NEW.state <> OLD.state",
                 "WHEN OLD.state <> OLD.state AND NEW.state <> NEW.state"),
        [_set((*BOOK, "status", "state", "moves", "retired"), ["active"])],
        "status: log.book.state: a terminal state moves"),
    "unenforced status": (
        None, [_set((*BOOK, "status", "state", "enforced_by"), None)],
        "status: log.book.state: its moves are declared, not enforced"),
    "status enforced by words": (
        None, [_set((*BOOK, "status", "state", "enforced_by"), "the plants' own enforcement")],
        "status: log.book.state: enforced by 'the plants' own enforcement', which refuses other"
        " moves"),
    "status guard inverted": (
        _replace("WHEN OLD.state = 'retired'", "WHEN OLD.state != 'retired'"), [],
        "status: log.book.state: enforced by 'trigger book_state_moves', which refuses other"
        " moves"),
    "status guard on another table": (
        None, [_set((*BOOK, "status", "state", "enforced_by"), "trigger event_no_update")],
        "status: log.book.state: enforced by 'trigger event_no_update', which refuses other moves"),
    "unbound version": (
        None, [_set(("tables", "log.recipe", "versions", "method_version"), None)],
        "version: log.recipe.method_version: no declared binding"),
    "version trigger inverted": (
        _replace("version = NEW.type_version", "version != NEW.type_version"), [],
        "version: log.event.type_version: bound by trigger event_type_registered, which does not"
        " refuse exactly the unregistered versions"),
    "version trigger names another definition": (
        _replace("name = NEW.type AND", "name != NEW.type AND"), [],
        "version: log.event.type_version: bound by trigger event_type_registered, which does not"
        " refuse exactly the unregistered versions"),
    "version trigger on another table": (
        None,
        [_set(("tables", "log.recipe", "versions", "method_version"),
              "trigger event_type_registered")],
        "version: log.recipe.method_version: bound by trigger event_type_registered, which does"
        " not refuse exactly the unregistered versions"),
    "catalog key unstated": (
        _replace("-- key: (listing_id, version_id)\n", "", "catalog"), [],
        "key: catalog.world.listing: the DDL states no key, and DuckLake enforces none"),
    "defaulted version": (
        _replace("method_version INTEGER NOT NULL,", "method_version INTEGER NOT NULL DEFAULT 1,"),
        [],
        "version: log.recipe.method_version: a version with a default"),
}  # fmt: skip


@cache
def _guarded() -> str:
    texts = _texts()
    texts["log"] += GUARD
    return json.dumps(ddl(texts))


@pytest.mark.proves("data_layer.the_schema_meets_its_contract")
@pytest.mark.parametrize("case", list(PLANTED), ids=lambda case: case.replace(" ", "-"))
def test_each_rule_broken_on_purpose_is_named(case: str) -> None:
    """A planted break of one rule is reported by that rule, naming its table and column."""
    tables, registry = json.loads(_guarded()), json.loads(_real()[1])
    registry["tables"]["log.book"]["status"]["state"]["enforced_by"] = "trigger book_state_moves"
    before = problems(tables, registry)
    in_ddl, in_registry, expected = PLANTED[case]
    for plant in in_registry:
        plant(registry)
    if in_ddl is not None:
        texts = _texts()
        texts["log"] += GUARD
        in_ddl(texts)
        tables = ddl(texts)
    assert problems(tables, registry) - before == {expected}
