"""The daily test-health report the test manager publishes: the test rows of the quality metrics,
from the checkout, its test-size registry and the commit-stage receipts it is given, each read
once through the receipt's own reader."""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from devtools.pipeline.definition import DefinitionError
from devtools.pipeline.receipt import ReceiptError, classed, read
from devtools.testing.requirements import file_holders, index
from devtools.testing.sizes import load_policy

SLOW_SECONDS = 10  # the quality metrics' row: commit-stage tests of 10 s or more
LARGE_BYTES = 100 * 1024  # the shape guard's ceiling for any tracked file under tests/


def _tracked(root: Path, top: str) -> list[Path]:
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", top],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [root / path for path in listed.split("\0") if path and (root / path).is_file()]


def _texts(paths: list[Path]) -> dict[Path, str]:
    """Each tracked Python file's text, read once."""
    texts = {}
    for path in paths:
        if path.suffix == ".py":
            try:
                texts[path] = path.read_bytes().decode("utf-8")
            except (OSError, UnicodeError) as error:
                raise DefinitionError(
                    "pipeline.workspace_unreadable",
                    f"{path}: {error}; restore this UTF-8 file and retry health",
                ) from None
    return texts


def tree(root: Path, today: date) -> dict[str, Any]:
    """Test bloat and the quarantine, from the checkout itself, each file read once."""
    files = _tracked(root, "tests")
    texts, source = _texts(files), _texts(_tracked(root, "src"))
    tests = [
        holder
        for path, text in sorted(texts.items())
        if path.name.startswith("test_")
        for holder in file_holders(path.relative_to(root).as_posix(), text)
    ]
    held = index(root, tests=tests)
    test_lines = sum(text.count("\n") for text in texts.values())
    source_lines = sum(text.count("\n") for text in source.values())
    quarantine = [
        {
            "node": row["node"],
            "owner": row["owner"],
            "until": row["until"],
            "days_left": (date.fromisoformat(row["until"]) - today).days,
        }
        for row in load_policy(root).get("quarantine", [])
    ]
    return {
        "test_lines": test_lines,
        "source_lines": source_lines,
        "test_to_source": round(test_lines / source_lines, 3) if source_lines else None,
        "test_functions": len(tests),
        "registered_functions": sum(1 for test in tests if test.ids),
        "requirements_with_several_holders": sorted({key for key, _ in held.duplicates()}),
        "files_over_100kb": [
            p.relative_to(root).as_posix() for p in files if p.stat().st_size > LARGE_BYTES
        ],
        "quarantine": sorted(quarantine, key=lambda row: row["days_left"]),
    }


def _stage_inputs(
    receipts: Sequence[Path], unreadable: list[dict[str, str]]
) -> Iterator[tuple[Path, list[Any], datetime]]:
    """Read each receipt identity once and retain unavailable inputs by their named refusal."""
    seen_paths: set[Path] = set()
    seen_receipts: set[bytes] = set()
    for path in receipts:
        try:
            canonical = path.resolve()
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            data, events = read(path)
            digest = sha256(data).digest()
            if digest in seen_receipts:
                continue
            seen_receipts.add(digest)
            moment = datetime.fromisoformat(events[-1].finished.finished_at)
            if moment.utcoffset() is None:
                raise ValueError("finished_at needs a timezone offset")
        except ReceiptError as refused:
            unreadable.append({"receipt": str(path), "code": refused.code})
            continue
        except (ValueError, OSError, RuntimeError) as error:
            unreadable.append(
                {"receipt": str(path), "code": "pipeline.receipt_malformed", "reason": str(error)}
            )
            continue
        yield path, events, moment


def stages(receipts: Sequence[Path]) -> dict[str, Any]:
    """Speed and reliability: results, flakes and the latest passing stage's slow tests. A stage
    with no verdict (an ERROR, or a receipt its reader refuses) counts toward the error rate; with
    no receipt each rate is unknown, never zero."""
    results: Counter[str] = Counter()
    flakes: Counter[str] = Counter()
    nodes = 0
    unreadable: list[dict[str, str]] = []
    latest: tuple[datetime | None, str, list[dict[str, Any]]] = (None, "", [])
    for path, events, moment in _stage_inputs(receipts, unreadable):
        finished = events[-1].finished
        results[finished.result] += 1
        tested = [e.test_result.model_dump() for e in events if "test_result" in e.id]
        nodes += len({result["node"] for result in tested})
        flakes.update(node for node, kind in classed(tested)[1].items() if kind == "flake")
        if finished.result == "PASS" and tested and (latest[0] is None or moment > latest[0]):
            latest = (moment, str(path), tested)
    if unreadable:
        results["UNREADABLE"] = len(unreadable)
    total = sum(result["seconds"] for result in latest[2])
    slow = [result for result in latest[2] if result["seconds"] >= SLOW_SECONDS]
    count = sum(results.values())
    return {
        "stages": count,
        "by_result": dict(results),
        "error_rate": round((results["ERROR"] + len(unreadable)) / count, 3) if count else None,
        "unreadable": unreadable,
        "flake_rate": round(sum(flakes.values()) / nodes, 4) if nodes else None,
        "flaked_nodes": sorted(flakes),
        "slow_from": latest[1] or None,
        "slow_tests": len(slow) if total else None,
        "slow_share": round(sum(r["seconds"] for r in slow) / total, 3) if total else None,
    }


def report(root: Path, receipts: Sequence[Path], today: date) -> dict[str, Any]:
    """The day's report: the tree's rows, then the stages'."""
    try:
        return {"date": today.isoformat(), "tree": tree(root, today), "stages": stages(receipts)}
    except DefinitionError:
        raise
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        raise DefinitionError(
            "pipeline.workspace_unreadable",
            f"{root}: {error}; repair the checkout inputs and retry health",
        ) from None
