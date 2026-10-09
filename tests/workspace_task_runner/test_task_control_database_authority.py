"""One Task Control store per workspace, and the store it must refuse.

`WorkspaceRuntime` built its `DuckDbTaskControlRegistry` against the market-data
database while sixteen other production sites spelled `research-task-control.duckdb`
as a literal. Nothing failed: the registry bootstraps its own schema, so the
wrong pointer quietly opened a second, empty task authority beside the real one.
A read-only inventory of the live workspace found thirteen task-shaped tables at
zero rows there against 74 tasks and 1,317 events in the store everyone else used.

These tests hold the two halves of the repair: one resolver so the filename is a
single fact, and a refusal so the market-data store cannot be adopted again.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from alphalattice.control.task_control.registry import (
    TASK_CONTROL_DATABASE_FILENAME,
    DuckDbTaskControlRegistry,
    TaskControlDatabaseAuthorityError,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.database import (
    WORKSPACE_MARKET_DATA_DATABASE_FILENAME,
    WorkspaceDatabase,
)
from alphalattice.control.workspace_runtime.mutation_gate import (
    WorkspaceMutationGate,
)

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def test_the_resolver_names_the_one_task_store() -> None:
    workspace = Path("/workspace")
    resolved = resolve_task_control_database(workspace)
    assert resolved.name == TASK_CONTROL_DATABASE_FILENAME
    assert resolved.parent == workspace / "runtime"
    # The two stores are different files, which is the whole point.
    assert TASK_CONTROL_DATABASE_FILENAME != WORKSPACE_MARKET_DATA_DATABASE_FILENAME


def test_the_resolver_takes_the_root_and_refuses_runtime_or_a_root_store_with_tasks(
    tmp_path: Path,
) -> None:
    """Regression: callers passed the root or `runtime/`, so a workspace grew a second,
    empty store at its root; a `runtime/` directory is refused, an empty root store is left
    alone, and one that holds a Task is refused rather than abandoned."""

    with pytest.raises(TaskControlDatabaseAuthorityError, match="workspace_root_expected"):
        resolve_task_control_database(tmp_path / "runtime")
    root_store = tmp_path / TASK_CONTROL_DATABASE_FILENAME
    connection = duckdb.connect(str(root_store))
    try:
        connection.execute("CREATE TABLE workspace_task (task_id VARCHAR)")
    finally:
        connection.close()
    assert resolve_task_control_database(tmp_path) == tmp_path / "runtime" / root_store.name
    _seed_task_state(root_store, table="workspace_task_event")
    with pytest.raises(TaskControlDatabaseAuthorityError, match="legacy_database_state_present"):
        resolve_task_control_database(tmp_path)


def test_the_registry_refuses_the_market_data_store(tmp_path: Path) -> None:
    """The refusal fires on the name, before anything is opened or created."""

    database = WorkspaceDatabase(tmp_path)
    assert database.path.name == WORKSPACE_MARKET_DATA_DATABASE_FILENAME
    with pytest.raises(
        TaskControlDatabaseAuthorityError,
        match="registry_database_not_canonical",
    ):
        DuckDbTaskControlRegistry(database.path, gate=WorkspaceMutationGate())
    # Nothing was written: the market-data file is still whatever it was, and no
    # task schema was bootstrapped into it.
    assert not database.path.exists()


def test_the_registry_refuses_every_noncanonical_database_name(tmp_path: Path) -> None:
    arbitrary = tmp_path / "another-task-authority.duckdb"
    with pytest.raises(
        TaskControlDatabaseAuthorityError,
        match="registry_database_not_canonical",
    ):
        DuckDbTaskControlRegistry(arbitrary, gate=WorkspaceMutationGate())
    assert not arbitrary.exists()


def _seed_task_state(database: Path, *, table: str = "workspace_task") -> None:
    connection = duckdb.connect(str(database))
    try:
        connection.execute(f'CREATE TABLE "{table}" (task_id VARCHAR)')
        connection.execute(f'INSERT INTO "{table}" VALUES (?)', ["task-1"])
    finally:
        connection.close()


def test_the_resolver_refuses_legacy_task_state_instead_of_abandoning_it(
    tmp_path: Path,
) -> None:
    market = tmp_path / WORKSPACE_MARKET_DATA_DATABASE_FILENAME
    _seed_task_state(market)

    with pytest.raises(
        TaskControlDatabaseAuthorityError,
        match="legacy_database_state_present",
    ):
        resolve_task_control_database(tmp_path)
    assert not (tmp_path / "runtime" / TASK_CONTROL_DATABASE_FILENAME).exists()


def test_unrelated_research_task_state_does_not_become_task_control_authority(
    tmp_path: Path,
) -> None:
    _seed_task_state(tmp_path / WORKSPACE_MARKET_DATA_DATABASE_FILENAME, table="research_task")

    assert resolve_task_control_database(tmp_path) == (
        tmp_path / "runtime" / TASK_CONTROL_DATABASE_FILENAME
    )


def test_the_resolver_refuses_two_populated_task_authorities(tmp_path: Path) -> None:
    _seed_task_state(tmp_path / WORKSPACE_MARKET_DATA_DATABASE_FILENAME)
    (tmp_path / "runtime").mkdir()
    _seed_task_state(tmp_path / "runtime" / TASK_CONTROL_DATABASE_FILENAME)

    with pytest.raises(
        TaskControlDatabaseAuthorityError,
        match="database_authorities_conflict",
    ):
        resolve_task_control_database(tmp_path)


def test_an_empty_legacy_schema_does_not_override_the_canonical_store(tmp_path: Path) -> None:
    market = tmp_path / WORKSPACE_MARKET_DATA_DATABASE_FILENAME
    connection = duckdb.connect(str(market))
    try:
        connection.execute("CREATE TABLE workspace_task (task_id VARCHAR)")
    finally:
        connection.close()

    assert resolve_task_control_database(tmp_path) == (
        tmp_path / "runtime" / TASK_CONTROL_DATABASE_FILENAME
    )


def test_the_registry_accepts_the_resolved_store_and_no_other_file_appears(
    tmp_path: Path,
) -> None:
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
    )
    assert registry.database_path.name == TASK_CONTROL_DATABASE_FILENAME
    assert registry.active_task() is None
    created = {
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if path.suffix == ".duckdb"
    }
    assert created == {f"runtime/{TASK_CONTROL_DATABASE_FILENAME}"}


def test_the_workspace_runtime_composes_the_canonical_store() -> None:
    """The composition owner must resolve, not pass its market-data path.

    Asserted on the source rather than by building a runtime, because building
    one takes a real workspace, a writer lease and a universe manifest -- and the
    fact under test is which path expression is written at those two call sites.
    """

    source = (
        PLAYPEN_ROOT
        / "src"
        / "alphalattice"
        / "control"
        / "product_host"
        / "composition"
        / "workspace.py"
    ).read_text(encoding="utf-8")
    assert "DuckDbTaskControlRegistry(database.path" not in source
    assert source.count("resolve_task_control_database(database.workspace)") == 2


def test_no_production_site_spells_the_task_database_as_a_literal() -> None:
    """Seventeen agreeing strings were what allowed one of them to disagree."""

    owner = (
        PLAYPEN_ROOT / "src" / "alphalattice" / "control" / "task_control" / "registry.py"
    ).resolve()
    offenders = [
        path.relative_to(PLAYPEN_ROOT).as_posix()
        for directory in ("src", "scripts")
        for path in (PLAYPEN_ROOT / directory).rglob("*.py")
        if path.resolve() != owner
        and f'"{TASK_CONTROL_DATABASE_FILENAME}"' in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
