"""One SQLite library opens the log in a process. The Host writes it through Python's sqlite3, and a
second library's closing descriptor would release the writer's POSIX locks on it, so the log is
attached in DuckDB only by the reading views' own process (`reading.serve`)."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "alphalattice"
READING = "src/alphalattice/foundation/data_layer/reading.py"
TEST_SCOPE = "whole-tree"


class _Sessions(ast.NodeVisitor):
    """Find calls that enable the log and SQL that attaches a SQLite database."""

    def __init__(self, where: str) -> None:
        self.where = where
        self.owners: list[str] = []
        self.with_log: list[tuple[str, str]] = []
        self.attaching: list[tuple[str, str]] = []

    def _site(self) -> tuple[str, str]:
        return self.where, ".".join(self.owners) or "<module>"

    def visit_FunctionDef(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        self.owners.append(node.name)
        self.generic_visit(node)
        self.owners.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name in {"reading_session", "_configure_reading"}:
            log = next((k.value for k in node.keywords if k.arg == "log"), None)
            forwards_log = (
                name == "_configure_reading"
                and self._site() == (READING, "reading_session")
                and isinstance(log, ast.Name)
                and log.id == "log"
            )
            if not forwards_log and not (isinstance(log, ast.Constant) and log.value is False):
                self.with_log.append(self._site())
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and "TYPE SQLITE" in node.value:
            self.attaching.append(self._site())


def _require_reader(with_log: list[tuple[str, str]], attaching: list[tuple[str, str]]) -> None:
    assert attaching == [(READING, "_configure_reading")], (
        f"DuckDB SQLite attach sites: {attaching}"
    )
    assert with_log == [(READING, "serve")], f"DuckDB log reader sites: {with_log}"


@pytest.mark.proves("data_layer.one_sqlite_library_opens_the_log")
def test_only_the_reading_views_process_attaches_the_log_in_duckdb() -> None:
    """CONTRACT: the installed reading process attaches the log, and a second module is refused."""
    with_log, attaching = [], []
    for path in sorted(SOURCE.rglob("*.py")):
        sessions = _Sessions(path.relative_to(ROOT).as_posix())
        sessions.visit(ast.parse(path.read_text(encoding="utf-8")))
        with_log.extend(sessions.with_log)
        attaching.extend(sessions.attaching)
    _require_reader(with_log, attaching)

    outside = "src/alphalattice/second_reader.py"
    for source in (
        "def read(): _configure_reading(connection, root, log=True, shared=False)",
        """def read(): connection.execute("ATTACH 'log.sqlite' AS log (TYPE SQLITE)")""",
    ):
        sessions = _Sessions(outside)
        sessions.visit(ast.parse(source))
        with pytest.raises(AssertionError, match=r"second_reader\.py.*read"):
            _require_reader(with_log + sessions.with_log, attaching + sessions.attaching)
