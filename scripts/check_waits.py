"""Refuse an unbounded wait or loop where it is written (devtools.architecture.waits).

A commit's staged Python sources under src/ and scripts/ are read as staged; a rule's findings in
a file above its baseline are refused, each by line with its repair. The baseline holds the cases
that existed when the rules came in; it only falls, and ``--update`` lowers it. A file the public
manifest's rules make private has its count in the baseline's private side, its name ending
``-baseline-private.json``, which the manifest's PRIVATE_ALLOWANCE rule keeps private; the public
baseline names no private file, and a checkout without private files has no private side.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "config/wait-baseline.json"
PRIVATE = BASELINE.removesuffix(".json") + "-private.json"
SIDES = (BASELINE, PRIVATE)
SCOPE = ("src/", "scripts/")
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.release.public_manifest import classify  # noqa: E402

from devtools.architecture import waits  # noqa: E402
from devtools.architecture.git_inputs import read_blobs  # noqa: E402


def _git(*words: str) -> str:
    done = subprocess.run(
        ["git", *words],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if done.returncode:
        raise RuntimeError(f"wait_rules.unreadable: git {words[0]}: {done.stderr.strip()[-200:]}")
    return done.stdout


def _side(key: str) -> str:
    """The baseline file that holds a ``path::rule`` key: the private one for a private path."""
    private = classify(key.split("::")[0], set(), set(), set())[0] == "PRIVATE"
    return PRIVATE if private else BASELINE


def _baseline(spec: str | None) -> dict[str, int] | None:
    """Both baselines' counts at a Git spec (``HEAD:``, ``:`` for the index, None for the
    checkout), or none without the public one; a checkout without private files has no private
    baseline."""
    if spec is None:
        files = [ROOT / name for name in SIDES]
        texts = [f.read_text(encoding="utf-8") if f.is_file() else None for f in files]
    else:
        blobs = read_blobs([spec + BASELINE, spec + PRIVATE], ROOT, timeout=60)
        texts = [blobs.get(spec + BASELINE), blobs.get(spec + PRIVATE)]
    if texts[0] is None:
        return None
    return {k: n for text in texts if text for k, n in json.loads(text)["counts"].items()}


def scan(paths: list[str], *, staged: bool) -> list[waits.Finding]:
    """Every finding in ``paths``, read from the index when ``staged``, else the checkout."""
    scoped = [p for p in paths if p.endswith(".py") and p.startswith(SCOPE)]
    if staged:
        texts = read_blobs([":" + p for p in scoped], ROOT, timeout=60)
        sources = {p: texts[":" + p] for p in scoped if ":" + p in texts}
    else:
        sources = {p: (ROOT / p).read_text(encoding="utf-8-sig") for p in scoped}
    return [row for path, text in sources.items() for row in waits.findings(path, text)]


def write(counts: dict[str, int]) -> None:
    """Each key to its own baseline file; the private file only when it holds a key or exists."""
    for name in SIDES:
        side = {k: n for k, n in sorted(counts.items()) if _side(k) == name}
        if name == PRIVATE and not side and not (ROOT / name).is_file():
            continue
        payload = {
            "schema": "alphalattice.wait-baseline.v1",
            "rule": "devtools.architecture.waits over src/ and scripts/; path::rule counts; "
            "lower only",
            "counts": side,
        }
        (ROOT / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="the commit: staged sources")
    mode.add_argument("--all", action="store_true", help="every tracked source")
    parser.add_argument("--update", action="store_true", help="write a lower baseline only")
    args = parser.parse_args(argv)
    if args.staged:
        paths = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split("\0")
    else:
        paths = _git("ls-files", "-z", "--", *SCOPE).split("\0")
    paths = [p for p in paths if p]
    rows = scan(paths, staged=args.staged)
    current = waits.counts(rows)
    held = _baseline(":" if args.staged else None)
    if held is None:
        if args.update and args.all:  # the rules' first count: the existing cases, held
            write(current)
            print(f"wait baseline seeded: {len(current)} keys, {sum(current.values())} findings")
            return 0
        print(f"wait_rules.unmeasured: no {BASELINE}; seed it with --all --update", file=sys.stderr)
        return 1
    risen = {key for key, n in current.items() if n > held.get(key, 0)}
    for row in rows:
        if f"{row.path}::{row.rule}" in risen:
            print(f"{row.path}:{row.line}: {row.rule}: {row.message}", file=sys.stderr)
    before = _baseline("HEAD:")
    raised = [k for k, n in held.items() if before is not None and n > before.get(k, 0)]
    for key in raised:
        print(f"{_side(key)}: {key} rose above HEAD's count; it only falls", file=sys.stderr)
    if risen or raised:
        print(
            f"wait_rules.refused: findings above {BASELINE}; repair each through {waits.WAIT}, "
            "never raise the baseline",
            file=sys.stderr,
        )
        return 1
    scanned = set(paths)
    lower = {
        key: min(n, current.get(key, 0)) if key.split("::")[0] in scanned else n
        for key, n in held.items()
    }
    lower = {key: n for key, n in lower.items() if n}
    if lower != held and args.update:
        write(lower)
        print(f"wait baseline lowered: {len(lower)} keys, {sum(lower.values())} findings")
    elif lower != held:
        print("wait rules: a file has fewer findings; run scripts/check_waits.py --update")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
