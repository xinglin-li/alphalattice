"""Ratchet Ruff's Google-style public API docstring findings by source file."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "src/alphalattice"
DEFAULT_BASELINE = ROOT / "config/docstring-baseline.json"
RUFF_CONFIG = (
    'lint.pydocstyle.convention="google"',
    "lint.pydocstyle.ignore-var-parameters=true",
)


def public_file(path: Path, source_root: Path) -> bool:
    """Exclude private modules and packages from the public API ratchet."""
    return all(
        not part.startswith("_") or part == "__init__.py"
        for part in path.relative_to(source_root).parts
    )


def private_spans(path: Path) -> list[tuple[int, int]]:
    """Find private definitions and local functions by their source line spans."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    spans: list[tuple[int, int]] = []

    def visit(nodes: list[ast.stmt], *, inside_function: bool = False) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                is_dunder = node.name.startswith("__") and node.name.endswith("__")
                if inside_function or (node.name.startswith("_") and not is_dunder):
                    spans.append((node.lineno, node.end_lineno or node.lineno))
                    continue
                visit(node.body, inside_function=not isinstance(node, ast.ClassDef))
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.stmt):
                        visit([child], inside_function=inside_function)

    visit(tree.body)
    return spans


def counts(source_root: Path) -> dict[str, int]:
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "D",
        "--output-format",
        "json",
    ]
    for option in RUFF_CONFIG:
        command.extend(("--config", option))
    command.append(str(source_root))
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or f"Ruff exited {result.returncode}")
    findings = json.loads(result.stdout)
    spans: dict[Path, list[tuple[int, int]]] = {}
    tally: Counter[str] = Counter()
    for item in findings:
        path = Path(item["filename"]).resolve()
        if not public_file(path, source_root):
            continue
        if path not in spans:
            spans[path] = private_spans(path)
        row = item["location"]["row"]
        if any(start <= row <= end for start, end in spans[path]):
            continue
        tally[path.relative_to(ROOT.resolve()).as_posix()] += 1
    return dict(sorted(tally.items()))


def write(path: Path, current: dict[str, int]) -> None:
    payload = {
        "schema": "alphalattice.docstring-baseline.v1",
        "rule": (
            "Ruff D rules; Google convention; ignore-var-parameters=true; public API; tests exempt"
        ),
        "origin": (
            "Initial develop backlog counted on 2026-09-26; "
            "public modules and definitions only; lower only"
        ),
        "counts": current,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="write only a lower baseline")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve()
    baseline = args.baseline.resolve()
    if not source_root.is_dir():
        parser.error(f"source root does not exist: {source_root}")
    current = counts(source_root)
    if not baseline.exists():
        if not args.update:
            parser.error(f"baseline does not exist: {baseline}; initialize with --update")
        write(baseline, current)
        print(f"initialized {len(current)} files, {sum(current.values())} findings")
        return 0
    previous = json.loads(baseline.read_text(encoding="utf-8"))["counts"]
    growth = {
        path: (previous.get(path, 0), count)
        for path, count in current.items()
        if count > previous.get(path, 0)
    }
    if growth:
        for path, (before, after) in sorted(growth.items()):
            print(f"{path}: docstring findings grew {before} -> {after}", file=sys.stderr)
        return 1
    if args.update:
        write(baseline, current)
        print(f"lowered baseline to {len(current)} files, {sum(current.values())} findings")
    else:
        print(f"docstring baseline holds: {len(current)} files, {sum(current.values())} findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
