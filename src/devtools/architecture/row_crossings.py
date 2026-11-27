"""Row-wise crossings out of the engines: each ``to_pylist()`` and ``fetchall()`` call, counted.

Bulk data stays columnar (engineering.md): a result leaves DuckDB or Arrow as Arrow or NumPy,
and Python touches single rows only for a decision or for the person. The commit holds these
counts per file to a baseline that only falls (scripts/check_counts.py), so a new crossing comes
in only when another is removed. Calls are read with Python's ast: the name inside a string or a
comment is never one, and an Arrow fetch (``to_arrow_table``, ``fetch_arrow_table``, ``fetchnumpy``)
is not counted.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

CROSSINGS = frozenset({"to_pylist", "fetchall"})
"""The calls that turn a table into Python rows."""


@dataclass(frozen=True)
class Finding:
    """One row-wise crossing: where, and its rule ``row-crossing[call]``."""

    path: str
    line: int
    rule: str

    @property
    def message(self) -> str:
        """The repair, as the refusal says it."""
        call = self.rule.removeprefix("row-crossing[").removesuffix("]")
        return f"a new {call}(): keep the result as Arrow or NumPy, or remove another crossing"


def findings(path: str, text: str) -> list[Finding]:
    """Every ``to_pylist()`` and ``fetchall()`` call in one Python source."""
    return [
        Finding(path, node.lineno, f"row-crossing[{node.func.attr}]")
        for node in ast.walk(ast.parse(text, path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in CROSSINGS
    ]
