"""Every write to rows a year fact vouches for ends or renews the facts of their years."""

from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "alphalattice"
WRITE = re.compile(r"\b(INSERT|UPDATE|DELETE|REPLACE|MERGE|DROP|ALTER|CREATE TABLE)\b")
# The schema functions change no values: each recreates the runtime view, whose definition hash
# keys every seal (runtime_view_hash).
SCHEMA = {
    ("foundation/feature_engine/publication/current_storage.py", "ensure_feature_current_schema"),
    ("foundation/feature_engine/publication/current_storage.py", "_key_rows_by_catalog"),
}
ADJUSTED_WRITE = re.compile(
    r"\b(?:INSERT(?:\s+OR\s+REPLACE)?\s+INTO|MERGE\s+INTO|DELETE\s+FROM|UPDATE)\s+"
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


def test_every_feature_row_write_ends_its_years_seals() -> None:
    """requirement: a Feature proof trusts a closed year's seal because
    every function writing ``feature_daily_current`` ends the seals of the years it touches, in
    its own transaction; only the schema functions, whose changes key the seals, are exempt."""

    writers = _writers("feature_daily_current", WRITE, "_clear_feature_year_seals")
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
        "record_year_facts(",
        "forget_year_facts(",
    )
    market = "foundation/market_data_ops/storage/duckdb.py"
    assert writers == {(market, "_reconcile_provider_adjusted_series"): True}
