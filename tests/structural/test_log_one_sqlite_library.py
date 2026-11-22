"""One SQLite library opens the log in a process. The Host writes it through Python's sqlite3, and a
second library's closing descriptor would release the writer's POSIX locks on it, so the log is
attached in DuckDB only by the reading views' own process (`reading.serve`)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "alphalattice"
READING = SOURCE / "kernel" / "data_layer" / "reading.py"
TEST_SCOPE = "whole-tree"


class _Sessions(ast.NodeVisitor):
    """Each reading_session call that keeps the log, by the function that makes it."""

    def __init__(self) -> None:
        self.owners: list[str] = ["<module>"]
        self.with_log: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.owners.append(node.name)
        self.generic_visit(node)
        self.owners.pop()

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        log = next((k.value for k in node.keywords if k.arg == "log"), None)
        if name == "reading_session" and not (isinstance(log, ast.Constant) and log.value is False):
            self.with_log.append(self.owners[-1])
        self.generic_visit(node)


@pytest.mark.proves("data_layer.one_sqlite_library_opens_the_log")
def test_only_the_reading_views_process_attaches_the_log_in_duckdb() -> None:
    """CONTRACT: every reading_session but the reader's own leaves the log out, and no other module
    attaches a SQLite file through DuckDB's sqlite extension."""
    with_log, attaching = [], []
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        where = path.relative_to(ROOT).as_posix()
        if "TYPE SQLITE" in text and path != READING:
            attaching.append(where)
        sessions = _Sessions()
        sessions.visit(ast.parse(text))
        with_log += [(where, owner) for owner in sessions.with_log]
    assert attaching == []
    assert with_log == [("src/alphalattice/kernel/data_layer/reading.py", "serve")]
