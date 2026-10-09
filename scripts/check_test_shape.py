"""Refuse a staged test change that breaks the writing rules in tests/README.md.

Checked on what the change adds, so a test is held to the rules when it is edited:
a test file over 100 KB that grows, source text read outside tests/structural, a sleep
of a second or more, and a test docstring that grows past three lines.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP_BYTES = 100 * 1024
SLEEP = re.compile(r"\bsleep\(\s*(\d+(?:\.\d+)?)\s*\)")


def _docstring_lines(text: str) -> dict[str, int]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    return {
        node.name: len((ast.get_docstring(node) or "").splitlines())
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
    }


def problems(path: str, old: str | None, new: str) -> list[str]:
    """The writing rules a change to one test file breaks; `old` is None for a new file."""
    found = []
    size = len(new.encode("utf-8"))
    if size > CAP_BYTES and size > len((old or "").encode("utf-8")):
        found.append(f"{path}: grows to {size // 1024} KB; a test file stays under 100 KB (rule 9)")
    if not path.endswith(".py"):
        return found
    before = set((old or "").splitlines())
    added = [line for line in new.splitlines() if line not in before]
    if not path.startswith("tests/structural/") and any("inspect.getsource(" in x for x in added):
        found.append(f"{path}: reads source text; test the behaviour at the owner's seam (rule 3)")
    for line in added:
        match = SLEEP.search(line)
        if match and float(match.group(1)) >= 1:
            found.append(f"{path}: sleeps {match.group(1)} s; drive the clock or the step (rule 7)")
    was = _docstring_lines(old or "")
    for name, count in _docstring_lines(new).items():
        if count > 3 and count > was.get(name, 0):
            found.append(
                f"{path}::{name}: a {count}-line docstring; one requirement sentence (rule 1)"
            )
    return found


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def main() -> int:
    staged = _git("diff", "--cached", "--name-only", "--diff-filter=AM").stdout.splitlines()
    # A merge adds only what neither parent holds: what the merged branch brings was checked there.
    merging = _git("rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0
    parents = ("HEAD", "MERGE_HEAD") if merging else ("HEAD",)
    found = []
    for path in staged:
        if path.startswith("tests/") and path.endswith((".py", ".cjs")):
            new = _git("show", f":{path}").stdout
            each = []
            for parent in parents:
                shown = _git("show", f"{parent}:{path}")
                old = shown.stdout if shown.returncode == 0 else None
                each.append(set(problems(path, old, new)))
            found += sorted(set.intersection(*each))
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
