"""The public tests' durations for pytest-split, and their totals, from JUnit records.

  python scripts/release/test_durations.py OUT JUNIT [JUNIT ...] [--tree DIR] [--summary FILE]
      [--expect SHARD [SHARD ...]] [--os windows|linux --sha SHA --pass-dir DIR --result FILE]

Each test that ran keeps its seconds under its pytest node id; a later record of the same node
wins, and a skipped one is left out, so pytest-split gives it the average. ``--tree`` keeps only
nodes whose file exists there (the public tree), so a private test's name never enters the public
file, and a node the export's own scan would flag is left out too, taking the average.
``--summary`` writes the counts and seconds the public-tests job reports. A record that cannot be
read refuses by name. ``--expect`` refuses unless each declared shard has exactly one record,
its shard read from the record's folder (``public-tests-junit-<shard>`` or
``public-tests-junit-<os>-<shard>``). Windows is the default reader. The per-OS result mode
checks the exact four JUnits and successful-shard identities before writing its result.
"""

import argparse
import json
import math
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


def read(
    records: list[Path], tree: Path | None, *, proof: tuple[str, str, Path] | None = None
) -> tuple[dict[str, float], dict[str, int]]:
    durations: dict[str, float] = {}
    outcomes: dict[str, str] = {}  # each node once, by its latest record
    prior: set[str] = set()
    if proof:
        os_name, sha, pass_dir = proof
        if pass_dir.is_symlink() or pass_dir.is_junction():
            raise ValueError(f"test_durations.pass_unreadable: {pass_dir}: linked directory")
        try:
            markers = {entry.name: entry for entry in pass_dir.iterdir()}
        except FileNotFoundError:
            markers = {}
        allowed = {f"public-tests-pass-{os_name}-{i}" for i in range(1, 5)}
        if markers.keys() - allowed:
            raise ValueError(f"test_durations.pass_mismatch: {pass_dir}: unexpected shard artifact")
    for record in records:
        try:
            document = ET.parse(record)
            cases = list(document.iter("testcase"))
        except (OSError, ET.ParseError) as error:
            raise SystemExit(
                f"test_durations.record_unreadable: {record}: {error}; restore the shard's JUnit"
            ) from error
        if proof:
            if (
                record.is_symlink()
                or record.is_junction()
                or record.parent.is_symlink()
                or record.parent.is_junction()
                or record.name != "junit.xml"
                or document.getroot().tag not in {"testsuite", "testsuites"}
                or not any(case.find("skipped") is None for case in cases)
            ):
                raise ValueError(
                    f"test_durations.record_unreadable: {record}: no valid executed shard"
                )
            group = int(record.parent.name.removeprefix(f"{ARTIFACT}{os_name}-"))
            bad = [
                case
                for case in cases
                if case.find("failure") is not None or case.find("error") is not None
            ]
            marker = markers.get(f"public-tests-pass-{os_name}-{group}")
            if bad and marker is not None:
                raise ValueError(
                    f"test_durations.pass_conflict: {record}: failed shard has a success marker"
                )
            if not bad:
                if (
                    marker is None
                    or marker.is_symlink()
                    or marker.is_junction()
                    or not marker.is_dir()
                ):
                    raise ValueError(
                        f"test_durations.pass_unreadable: {record}: missing physical success marker"
                    )
                file = marker / "passed.json"
                if (
                    file.is_symlink()
                    or file.is_junction()
                    or {item.name for item in marker.iterdir()} != {"passed.json"}
                ):
                    raise ValueError(
                        f"test_durations.pass_unreadable: {marker}: malformed artifact"
                    )
                payload = json.loads(file.read_text(encoding="utf-8"))
                if (
                    not isinstance(payload, dict)
                    or type(payload.get("group")) is not int
                    or payload != {"os": os_name, "sha": sha, "group": group}
                ):
                    raise ValueError(
                        f"test_durations.pass_mismatch: {file}: wrong OS, SHA or group"
                    )
            names = [node_of(case) for case in cases]
            if any(
                not case.get("classname") or not case.get("name") for case in cases
            ) or prior.intersection(names):
                raise ValueError(
                    f"test_durations.record_mismatch: {record}: "
                    "unnamed or repeated cross-shard node"
                )
            prior.update(names)
            if any(
                not math.isfinite(float(case.get("time") or 0)) or float(case.get("time") or 0) < 0
                for case in cases
            ):
                raise ValueError(f"test_durations.record_unreadable: {record}: invalid test time")
        seen: dict[str, str] = {}
        for case in cases:
            node = node_of(case)
            visible = tree is None or (tree / node.split("::", 1)[0]).is_file()
            if not visible and not proof:
                continue
            outcome = next(
                (tag for tag in ("error", "failure", "skipped") if case.find(tag) is not None),
                "passed",
            )
            # Failures/errors survive skips within a record; a later record still replaces it.
            if outcome in {"failure", "error"} or (
                seen.get(node) not in {"failure", "error"}
                and (outcome != "passed" or seen.get(node, "skipped") == "skipped")
            ):
                seen[node] = outcome
            if visible and outcome != "skipped" and publishable(node, tree):
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
    parser.add_argument(
        "--os",
        choices=("windows", "linux"),
        default="windows",
        help="the artifact OS (default: windows)",
    )
    parser.add_argument("--sha", help="the exact tested commit in per-OS result mode")
    parser.add_argument("--pass-dir", type=Path, help="the downloaded per-shard success artifacts")
    parser.add_argument("--result", type=Path, help="write the per-OS result identity and status")
    args = parser.parse_args(argv)
    per_os = args.result is not None or args.sha is not None or args.pass_dir is not None
    if per_os and (
        args.result is None
        or args.pass_dir is None
        or not re.fullmatch(r"[0-9a-f]{40}", args.sha or "")
    ):
        parser.error("per-OS results require --sha, --pass-dir and --result")

    def result(status: str) -> None:
        if args.result is not None:
            args.result.write_text(
                json.dumps({"os": args.os, "sha": args.sha, "status": status}) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    # a shard is its folder's whole name past the prefix; another folder names none
    prefix = ARTIFACT + args.os + "-"
    shards = [
        name.removeprefix(prefix)
        if name.startswith(prefix)
        else name.removeprefix(ARTIFACT)
        if args.os == "windows" and not per_os and name.startswith(ARTIFACT)
        else f"<{name}>"
        for name in (record.parent.name for record in args.records)
    ]
    try:
        if (per_os and sorted(args.expect) != ["1", "2", "3", "4"]) or (
            args.expect
            and (len(set(args.expect)) < len(args.expect) or sorted(shards) != sorted(args.expect))
        ):
            raise ValueError(
                f"test_durations.records_mismatch: JUnit records for shards {' '.join(shards)}, "
                f"expected each of {' '.join(args.expect)} once; rerun the full run"
            )
        durations, counts = read(
            args.records, args.tree, proof=(args.os, args.sha, args.pass_dir) if per_os else None
        )
        args.out.write_text(json.dumps(durations, indent=1) + "\n", encoding="utf-8", newline="\n")
        summary = {
            **counts,
            "nodes": len(durations),
            "test_seconds": round(sum(durations.values()), 1),
        }
        if args.summary:
            args.summary.write_text(
                json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
            )
        print(json.dumps(summary))
        status = "failed" if counts["failed"] or counts["errors"] else "success"
        result(status)
        return int(per_os and status == "failed")
    except (OSError, UnicodeError, ValueError, TypeError, SystemExit) as error:
        if not per_os:
            raise SystemExit(str(error)) from error
        result("unknown")
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
