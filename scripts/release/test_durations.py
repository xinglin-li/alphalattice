"""The public tests' durations for pytest-split, and their totals, from JUnit records.

  python scripts/release/test_durations.py OUT JUNIT [JUNIT ...] [--tree DIR] [--summary FILE]
      [--expect SHARD [SHARD ...]]

Each test that ran keeps its seconds under its pytest node id; a later record of the same node
wins, and a skipped one is left out, so pytest-split gives it the average. ``--tree`` keeps only
nodes whose file exists there (the public tree), so a private test's name never enters the public
file, and a node the export's own scan would flag is left out too, taking the average.
``--summary`` writes the counts and seconds the public-tests job reports. A record that cannot be
read refuses by name. ``--expect`` refuses unless each declared shard has exactly one record,
its shard read from the record's folder (``public-tests-junit-<shard>``, as CI downloads it).
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # as a script in CI, or imported in tests
from public_manifest import PRIVATE_PREFIXES, SCANS  # the export's own line rules


def node_of(case: ET.Element) -> str:
    """A testcase's node id: its classname to the ``test_`` module is the file, then its classes."""
    parts = case.get("classname", "").split(".")
    end = next((i for i, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1)
    return "::".join(["/".join(parts[: end + 1]) + ".py", *parts[end + 1 :], case.get("name", "")])


def publishable(node: str, tree: Path | None) -> bool:
    """Whether the export's scan passes this node's line: no flagged literal, no internal tree,
    and no path-like token outside the tree (a parametrize id may name a private file)."""
    line = json.dumps(node)
    normalized = line.replace("\\\\", "/").replace("\\", "/")
    if any(p.search(line) for p in SCANS.values()) or any(
        p in normalized for p in PRIVATE_PREFIXES
    ):
        return False
    tokens = re.findall(r"[\w.-]+(?:/[\w.-]+)+\.\w+", node.split("::", 1)[-1])
    return tree is None or all((tree / token).is_file() for token in tokens)


def read(records: list[Path], tree: Path | None) -> tuple[dict[str, float], dict[str, int]]:
    durations: dict[str, float] = {}
    outcomes: dict[str, str] = {}  # each node once, by its latest record
    for record in records:
        try:
            cases = list(ET.parse(record).iter("testcase"))
        except (OSError, ET.ParseError) as error:
            raise SystemExit(
                f"test_durations.record_unreadable: {record}: {error}; restore the shard's JUnit"
            ) from error
        seen: dict[str, str] = {}
        for case in cases:
            node = node_of(case)
            if tree is not None and not (tree / node.split("::", 1)[0]).is_file():
                continue
            outcome = next(
                (tag for tag in ("skipped", "failure", "error") if case.find(tag) is not None),
                "passed",
            )
            # within one record a teardown error outlives its pass; a later record replaces it
            if outcome != "passed" or seen.get(node, "skipped") == "skipped":
                seen[node] = outcome
            if outcome != "skipped" and publishable(node, tree):
                durations[node] = round(float(case.get("time") or 0), 3)
        outcomes.update(seen)
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for outcome in outcomes.values():
        counts[{"failure": "failed", "error": "errors"}.get(outcome, outcome)] += 1
    return dict(sorted(durations.items())), counts


ARTIFACT = "public-tests-junit-"
"""The shard artifacts' name before the shard, as ci.yml uploads them."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "out", type=Path, help="the durations file to write (pytest-split's .test_durations)"
    )
    parser.add_argument("records", type=Path, nargs="+", help="JUnit files, oldest first")
    parser.add_argument(
        "--tree", type=Path, help="keep only nodes whose file exists under this tree"
    )
    parser.add_argument("--summary", type=Path, help="write the counts and seconds here as JSON")
    parser.add_argument(
        "--expect", nargs="+", default=[], help="the shards that must each have one JUnit here"
    )
    args = parser.parse_args(argv)
    # a shard is its folder's whole name past the prefix; another folder names none
    shards = [
        name.removeprefix(ARTIFACT) if name.startswith(ARTIFACT) else f"<{name}>"
        for name in (record.parent.name for record in args.records)
    ]
    if args.expect and (
        len(set(args.expect)) < len(args.expect) or sorted(shards) != sorted(args.expect)
    ):
        # every declared shard's own record, each once: a full run, not merely as many files
        raise SystemExit(
            f"test_durations.records_mismatch: JUnit records for shards {' '.join(shards)}, "
            f"expected each of {' '.join(args.expect)} once; rerun the full run"
        )
    durations, counts = read(args.records, args.tree)
    args.out.write_text(json.dumps(durations, indent=1) + "\n", encoding="utf-8", newline="\n")
    summary = {**counts, "nodes": len(durations), "test_seconds": round(sum(durations.values()), 1)}
    if args.summary:
        args.summary.write_text(
            json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
