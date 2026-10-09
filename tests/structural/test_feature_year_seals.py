"""Every write to Feature rows ends the seals of their years."""

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


def _writers() -> dict[tuple[str, str], bool]:
    found: dict[tuple[str, str], bool] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "feature_daily_current" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                body = ast.get_source_segment(text, node) or ""
                if "feature_daily_current" in body and WRITE.search(body):
                    found[(path.relative_to(SOURCE).as_posix(), node.name)] = (
                        "_clear_feature_year_seals" in body
                    )
    return found


def test_every_feature_row_write_ends_its_years_seals() -> None:
    """requirement: a Feature proof trusts a closed year's seal because
    every function writing ``feature_daily_current`` ends the seals of the years it touches, in
    its own transaction; only the schema functions, whose changes key the seals, are exempt."""

    writers = _writers()
    assert set(writers) >= SCHEMA
    unsealed = sorted(key for key, clears in writers.items() if not clears and key not in SCHEMA)
    assert not unsealed, f"each must call _clear_feature_year_seals: {unsealed}"
