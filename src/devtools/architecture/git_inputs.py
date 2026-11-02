"""Shared Git blob input for registry and test-shape guards."""

import io
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

from devtools.testing.layout import rule_patterns  # the mover's own rule projection


def read_blobs(specs: list[str], root: Path | None = None) -> dict[str, str]:
    """Read distinct Git specifications in one batch, omitting missing blobs."""
    specs = list(dict.fromkeys(specs))
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=("\n".join(specs) + "\n").encode(),
        cwd=root or Path(__file__).resolve().parents[3],
        capture_output=True,
        check=True,
    )
    stream, found = io.BytesIO(result.stdout), {}
    for spec in specs:
        header = stream.readline().split()
        if header[-1:] == [b"missing"]:
            continue
        data = stream.read(int(header[-1]))
        stream.read(1)
        found[spec] = data.decode("utf-8", errors="surrogateescape")
    return found


Moves = dict[str, str]  # a staged new path to the deleted path it moved from


def staged_moves(root: Path | None = None) -> Moves:
    """Each staged path whose blob equals a path the change deletes, to that path: a move."""
    rows = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--raw",
            "-z",
            "--no-renames",
            "--full-index",
            "--diff-filter=AD",
        ],
        cwd=root or Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.split("\0")
    pairs = [
        (meta.split(), path)
        for meta, path in zip(rows[::2], rows[1::2], strict=False)
        if meta[:1] == ":"
    ]
    gone = {meta[2]: path for meta, path in pairs if meta[4] == "D"}
    return {path: gone[meta[3]] for meta, path in pairs if meta[4] == "A" and meta[3] in gone}


def carried(keys: Iterable[str], moves: Moves) -> set[str]:
    """Each key naming a moved file's old path, restated at its new path: what a move carries."""
    if not moves:
        return set()
    keys = list(keys)
    paths = sorted(map(re.escape, moves.values()), key=len, reverse=True)
    old = re.compile(r"(?<![\w./-])(" + "|".join(paths) + r")(?![\w.-])")  # whole paths only
    back = {before: after for after, before in moves.items()}
    found = {old.sub(lambda m: back[m.group(1)], key) for key in keys if old.search(key)}
    for key in keys:  # a path rule covering an old path, restated as the mover restates it
        head, *rule = re.split(r"(?<=\w):(?!:)", key, maxsplit=1)
        if rule and " " not in rule[0] and any(c in rule[0] for c in "*?["):
            found |= {head + ":" + p for p in rule_patterns(rule[0], (), back)} - {key}
    return found
