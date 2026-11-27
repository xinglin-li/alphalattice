"""Refuse a count that grew where it is written: the wait rules, strict typing's escapes and the
row-wise crossings out of the engines.

Each ratchet counts its findings per ``path::rule`` against its own baseline: the wait rules
(devtools.architecture.waits, over src/ and scripts/), each ``type: ignore`` per error code
(devtools.architecture.type_ignores, over every tracked Python file) and each ``to_pylist()`` and
``fetchall()`` call (devtools.architecture.row_crossings, over src/alphalattice/). A commit's
staged files are read as staged, in one Git read for all; a file's findings above its baseline are
refused, each by line with its repair. A baseline holds the cases that existed when its rules came
in: it only falls,
a change that raises it above its base (HEAD for a commit, the merge base with --base otherwise) is
refused, and ``--update`` lowers it. A byte-identical move keeps its old path's counts; the commit
that moves a counted file carries them to its new path with ``--update``. A file the public
manifest's rules make private has its count in the baseline's private side, its name ending
``-baseline-private.json``, which the manifest's PRIVATE_ALLOWANCE rule keeps private; the public
baseline names no private file, and a checkout without private files has no private side.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.release.public_manifest import classify  # noqa: E402

from alphalattice.capabilities.domain_contracts import spawn  # noqa: E402
from devtools.architecture import row_crossings, type_ignores, waits  # noqa: E402
from devtools.architecture.git_inputs import read_blobs, staged_changes, staged_moves  # noqa: E402


@dataclass(frozen=True)
class Ratchet:
    """One count that only falls: the files it reads, its findings, its baseline and its repair."""

    name: str
    baseline: str
    scope: tuple[str, ...]  # path prefixes; empty for every tracked Python file
    findings: Callable[[str, str], list]
    rule: str
    repair: str

    @property
    def sides(self) -> tuple[str, str]:
        """The public baseline and its private side."""
        return self.baseline, self.baseline.removesuffix(".json") + "-private.json"

    def reads(self, path: str) -> bool:
        """Whether ``path`` is one of this ratchet's files."""
        return path.endswith(".py") and path.startswith(self.scope or ("",))


WAITS = Ratchet(
    "wait_rules",
    "config/wait-baseline.json",
    ("src/", "scripts/"),
    waits.findings,
    "devtools.architecture.waits over src/ and scripts/; path::rule counts; lower only",
    f"repair each through {waits.WAIT}, never raise the baseline",
)
TYPE_IGNORES = Ratchet(
    "type_ignores",
    "config/type-ignore-baseline.json",
    (),
    type_ignores.findings,
    "devtools.architecture.type_ignores over every tracked Python file; "
    "path::type-ignore[code] counts; lower only",
    "type the code instead of suppressing it, never raise the baseline",
)
ROW_CROSSINGS = Ratchet(
    "row_crossings",
    "config/row-crossing-baseline.json",
    ("src/alphalattice/",),
    row_crossings.findings,
    "devtools.architecture.row_crossings over src/alphalattice/; "
    "path::row-crossing[call] counts; lower only",
    "keep the result columnar (Arrow or NumPy), never raise the baseline",
)
RATCHETS = (WAITS, TYPE_IGNORES, ROW_CROSSINGS)


def _git(*words: str) -> str:
    import subprocess  # here only: a unit test importing this script imports no process module

    done = spawn.run(
        ["git", *words],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if done.returncode:
        raise RuntimeError(f"counts.unreadable: git {words[0]}: {done.stderr.strip()[-200:]}")
    return done.stdout


def _side(ratchet: Ratchet, key: str) -> str:
    """The baseline file that holds a ``path::rule`` key: the private one for a private path."""
    private = classify(key.split("::")[0], set(), set(), set())[0] == "PRIVATE"
    return ratchet.sides[private]


def _counts(texts: list[str | None]) -> dict[str, int] | None:
    """Both sides' counts, or none without the public one."""
    if texts[0] is None:
        return None
    return {k: n for text in texts if text for k, n in json.loads(text)["counts"].items()}


def _moved(counts: dict[str, int] | None, moves: dict[str, str]) -> dict[str, int] | None:
    """``counts`` with each moved file's keys under its new path."""
    if counts is None or not moves:
        return counts
    new = {old: path for path, old in moves.items()}
    moved: dict[str, int] = {}
    for key, n in counts.items():
        path, rule = key.split("::", 1)
        moved[f"{new.get(path, path)}::{rule}"] = n
    return moved


def write(ratchet: Ratchet, counts: dict[str, int]) -> None:
    """Each key to its own baseline file; the private file only when it holds a key or exists."""
    for name in ratchet.sides:
        side = {k: n for k, n in sorted(counts.items()) if _side(ratchet, k) == name}
        if name == ratchet.sides[1] and not side and not (ROOT / name).is_file():
            continue
        payload = {
            "schema": f"alphalattice.{Path(ratchet.baseline).stem}.v1",
            "rule": ratchet.rule,
            "counts": side,
        }
        (ROOT / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )


def judge(
    ratchet: Ratchet,
    rows: list,
    held: dict[str, int],
    before: dict[str, int] | None,
    *,
    on_file: dict[str, int],
    scanned: set[str],
    gone: set[str],
    update: bool,
) -> bool:
    """Refuse ``rows`` above ``held`` and ``held`` above ``before``; on ``update``, write ``held``
    lowered to ``rows`` in the ``scanned`` files, without the files ``gone``, where it differs from
    the baseline ``on_file`` (a move's counts are held under its new path, still filed under its
    old one)."""
    current = Counter(f"{row.path}::{row.rule}" for row in rows)
    risen = {key for key, n in current.items() if n > held.get(key, 0)}
    for row in rows:
        if f"{row.path}::{row.rule}" in risen:
            print(f"{row.path}:{row.line}: {row.rule}: {row.message}", file=sys.stderr)
    raised = [k for k, n in held.items() if before is not None and n > before.get(k, 0)]
    for key in raised:
        print(
            f"{_side(ratchet, key)}: {key} rose above its base's count; it only falls",
            file=sys.stderr,
        )
    if risen or raised:
        print(
            f"{ratchet.name}.refused: findings above {ratchet.baseline}; {ratchet.repair}",
            file=sys.stderr,
        )
        return False
    lower = {}
    for key, n in held.items():
        path = key.split("::")[0]
        left = min(n, current[key]) if path in scanned else 0 if path in gone else n
        if left:
            lower[key] = left
    if lower != on_file and update:
        write(ratchet, lower)
        print(f"{ratchet.name} baseline lowered: {len(lower)} keys, {sum(lower.values())}")
    elif lower != on_file:
        print(f"{ratchet.name}: a file has fewer findings; run scripts/check_counts.py --update")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="the commit: staged files")
    mode.add_argument("--all", action="store_true", help="every tracked file")
    parser.add_argument("--update", action="store_true", help="write a lower baseline only")
    parser.add_argument(
        "--base",
        default="HEAD",
        help="outside a commit, a change may not raise the baseline "
        "above its merge base with this revision's",
    )
    args = parser.parse_args(argv)
    if args.staged:
        base, pairs = "HEAD", staged_changes(ROOT, timeout=60)
        paths = [path for meta, path in pairs if meta[4] != "D"]
    else:
        base = _git("merge-base", args.base, "HEAD").strip()
        pairs = staged_changes(ROOT, base=base, tree="HEAD", timeout=60)
        paths = [path for path in _git("ls-files", "-z").split("\0") if path]
    moves = staged_moves(ROOT, pairs=pairs)
    gone = {path for meta, path in pairs if meta[4] == "D"} if not args.staged else set()
    sources = sorted({path for path in paths if any(r.reads(path) for r in RATCHETS)})
    sides = [name for ratchet in RATCHETS for name in ratchet.sides]
    specs = [f"{base}:{name}" for name in sides]
    if args.staged:  # one read for both baselines, here and at the base, and every staged file
        specs += [":" + name for name in sides] + [":" + path for path in sources]
    blobs = read_blobs(specs, ROOT, timeout=60)
    if args.staged:
        texts = {path: blobs[":" + path] for path in sources if ":" + path in blobs}
    else:
        texts = {path: (ROOT / path).read_text(encoding="utf-8-sig") for path in sources}
    passed = True
    for ratchet in RATCHETS:
        rows = [
            row
            for p, text in texts.items()
            if ratchet.reads(p)
            for row in ratchet.findings(p, text)
        ]
        if args.staged:
            held = _counts([blobs.get(":" + name) for name in ratchet.sides])
        else:
            files = [ROOT / name for name in ratchet.sides]
            held = _counts([f.read_text(encoding="utf-8") if f.is_file() else None for f in files])
        if held is None:
            if args.update and args.all:  # the rules' first count: the existing cases, held
                write(ratchet, dict(Counter(f"{row.path}::{row.rule}" for row in rows)))
                print(f"{ratchet.name} baseline seeded: {len(rows)} findings")
                continue
            print(
                f"{ratchet.name}.unmeasured: no {ratchet.baseline}; seed it with --all --update",
                file=sys.stderr,
            )
            passed = False
            continue
        unmoved = [key for key in held if key.split("::")[0] in moves.values()]
        if args.staged and unmoved and not args.update:
            print(
                f"{ratchet.name}.refused: {unmoved[0].split('::')[0]} moved; carry its counts "
                "with scripts/check_counts.py --update and stage the baseline",
                file=sys.stderr,
            )
            passed = False
            continue
        before = _counts([blobs.get(f"{base}:{name}") for name in ratchet.sides])
        passed &= judge(
            ratchet,
            rows,
            _moved(held, moves),
            _moved(before, moves),
            on_file=held,
            scanned={p for p in texts if ratchet.reads(p)},
            gone=gone,
            update=args.update,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
