"""Every write to rows a year fact vouches for ends or renews the facts of their years."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
# The schema functions change no values: each recreates the runtime view, whose definition hash
# keys every seal (runtime_view_hash).
SCHEMA = {
    ("foundation/feature_engine/publication/current_storage.py", "ensure_feature_current_schema"),
    ("foundation/feature_engine/publication/current_storage.py", "_key_rows_by_catalog"),
}
ADJUSTED_WRITE = re.compile(
    r"\b(?:INSERT(?:\s+OR\s+REPLACE)?\s+INTO|MERGE\s+INTO|DELETE\s+FROM|UPDATE)\s+"
    # the table by name, or the physical table its read overlay names (market_snapshots)
    r"(?:\"\s*\+\s*market_snapshots\.pending_table\([^,]+,\s*\")?"
    r"provider_adjusted_close_current\b"
)


def _writers(table: str, pattern: re.Pattern[str], *calls: str) -> dict[tuple[str, str], bool]:
    found: dict[tuple[str, str], bool] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if table not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body = ast.get_source_segment(text, node) or ""
                if table in body and pattern.search(body):
                    found[(path.relative_to(SOURCE).as_posix(), node.name)] = all(
                        call in body for call in calls
                    )
    return found


WRITE_TARGET = re.compile(
    r"\b(?:INSERT(?:\s+OR\s+REPLACE)?\s+INTO|MERGE\s+INTO|DELETE\s+FROM|UPDATE"
    r"|CREATE(?:\s+OR\s+REPLACE)?(?:\s+TEMP(?:ORARY)?)?\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?"
    r"|DROP\s+TABLE(?:\s+IF\s+EXISTS)?|ALTER\s+TABLE)\s+(\{[^}]*\}|[\w.]+)"
)


def _row_writers(table: str, call: str) -> dict[tuple[str, str], bool]:
    """Each function with a statement writing `table`: its target is `table`, or a placeholder
    its module does not bind to another table's name (a loop's `{table}` counts); a write to
    another table that only reads `table` (a copy, a temp table) does not."""
    found: dict[tuple[str, str], bool] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if table not in text:
            continue
        tree = ast.parse(text)
        names = {
            target.id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign | ast.AnnAssign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body = ast.get_source_segment(text, node) or ""
                targets = [match.group(1) for match in WRITE_TARGET.finditer(body)]
                if table in body and any(
                    name == table or (name[0] == "{" and names.get(name[1:-1], table) == table)
                    for name in targets
                ):
                    found[(path.relative_to(SOURCE).as_posix(), node.name)] = call in body
    return found


def test_every_feature_row_write_ends_its_years_seals() -> None:
    """requirement: a Feature proof trusts a closed year's seal because
    every function writing ``feature_daily_current`` ends the seals of the years it touches, in
    its own transaction; only the schema functions, whose changes key the seals, are exempt."""

    # A function that only reads its rows into another table writes none of them.
    writers = _row_writers("feature_daily_current", "_clear_feature_year_seals")
    assert {
        ("foundation/feature_engine/storage/repositories.py", name)
        for name in ("discard_part_rows", "retire_rows_outside_layer")
    } <= set(writers)  # a loop's `{table}` still counts as writing it
    assert set(writers) >= SCHEMA
    unsealed = sorted(key for key, clears in writers.items() if not clears and key not in SCHEMA)
    assert not unsealed, f"each must call _clear_feature_year_seals: {unsealed}"


def test_the_adjusted_close_writer_keeps_its_years_facts() -> None:
    """requirement: the action audit trusts a closed year's adjusted-series fact because the
    one function writing adjusted closes records the years it writes and forgets those it
    replaces."""

    writers = _writers(
        "provider_adjusted_close_current",
        ADJUSTED_WRITE,
        "record_year_facts_by_scope(",
        "forget_year_facts(",
    )
    market = "foundation/market_data_ops/storage/duckdb.py"
    assert writers == {(market, "_reconcile_provider_adjusted_series"): True}
