"""``python scripts/tests_registry.py find "<words>"``: the registered requirements and the tests
that hold them, best first, before a test is written; ``shrink``: lower each file's count of tests
older than the registry to what the tree holds (devtools.testing)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # devtools is checkout tooling, not an installed package

from devtools.testing.requirements import find, shrink  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/tests_registry.py", description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("find", help="find before you write a test").add_argument("words")
    lower = commands.add_parser("shrink", help="lower each file's legacy count")
    lower.add_argument("--seed", action="store_true", help="count the tree once, first")
    lower.add_argument("--to", type=Path, help="with --seed: the private place for the first count")
    args = parser.parse_args(argv)
    if args.command == "shrink":
        try:
            counted = shrink(args.root, seed=args.seed, to=args.to)
            print(f"{sum(counted.values())} unregistered tests counted")
        except ValueError as refused:
            print(refused)
            return 1
        return 0
    print("\n".join(find(args.root, args.words)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
