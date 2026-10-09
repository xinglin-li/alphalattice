"""The spellings NM2 retired are the ones nothing tracked still uses.

`RETIRED_SPELLINGS` is the class a workspace prepared before the 2026-10-02 renames is refused
by. It names a spelling only when no product source, test, script or registry uses it any more,
and each one's successor only when the source uses that instead; the records (the evidence
roots' retired reasons, the plans, the census) keep the spellings they recorded.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from alphalattice.kernel.shared_kernel.retired_spellings import RETIRED_SPELLINGS

ROOT = Path(__file__).resolve().parents[2]
TABLE = "src/alphalattice/kernel/shared_kernel/retired_spellings.py"
SUFFIXES = (".py", ".json", ".yaml", ".yml")


def _tracked(*folders: str) -> dict[str, str]:
    listed = subprocess.run(
        ["git", "ls-files", "--", *folders],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {
        path: (ROOT / path).read_text(encoding="utf-8")
        for path in listed
        if path.endswith(SUFFIXES)
        and path != TABLE
        and path != "tests/structural/test_retired_spellings.py"
        and (ROOT / path).is_file()
    }


def _pattern(spelling: str) -> re.Pattern[str]:
    if spelling.endswith(":"):
        return re.compile(re.escape(spelling))
    if spelling == "IW184":
        # The family's bare name, as a value; the ids that keep it inside them are.
        return re.compile(r"[\"']IW184[\"']")
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(spelling) + r"(?![A-Za-z0-9_])")


def test_no_tracked_source_uses_a_retired_spelling_and_each_successor_is_used() -> None:
    """Requirement: the class the reader refuses by is exact: every spelling it names is
    gone from the source, tests, scripts and the shipped registries, and its successor is the
    one the source uses."""

    used = _tracked("src", "tests", "scripts")
    still = {
        spelling: sorted(path for path, text in used.items() if _pattern(spelling).search(text))
        for spelling in RETIRED_SPELLINGS
    }
    assert {spelling: paths for spelling, paths in still.items() if paths} == {}
    source = {path: text for path, text in used.items() if path.startswith("src/")}
    unused = [
        successor
        for successor in RETIRED_SPELLINGS.values()
        if not any(_pattern(successor).search(text) for text in source.values())
    ]
    assert unused == []
