"""Shared Git blob input for registry and test-shape guards."""

import io
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from functools import cached_property
from pathlib import Path

from devtools.testing.layout import rule_patterns  # the mover's own rule projection
from devtools.testing.requirements import LAYOUT_SCHEMA, schema_file


def read_blobs(
    specs: list[str],
    root: Path | None = None,
    *,
    object_ids: dict[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, str]:
    """Read distinct Git specifications in one batch, omitting missing blobs."""
    specs = list(dict.fromkeys(specs))
    if not specs:
        return {}
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=("\n".join(specs) + "\n").encode(),
        cwd=root or Path(__file__).resolve().parents[3],
        capture_output=True,
        check=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    stream, found = io.BytesIO(result.stdout), {}
    for spec in specs:
        header = stream.readline().split()
        if header[-1:] == [b"missing"]:
            continue
        if object_ids is not None:
            object_ids[spec] = header[0].decode()
        data = stream.read(int(header[-1]))
        stream.read(1)
        found[spec] = data.decode("utf-8", errors="surrogateescape")
    return found


Moves = dict[str, str]  # a staged new path to the deleted path it moved from


def staged_changes(
    root: Path | None = None,
    *,
    tree: str | None = None,
    base: str | None = None,
    timeout: float | None = None,
) -> list[tuple[list[str], str]]:
    """One raw diff supplies both changed paths and exact move blobs."""
    if base is not None and tree is None:
        raise ValueError("a base revision needs a target tree")
    rows = subprocess.run(
        [
            "git",
            "diff",
            *([base or "HEAD", tree] if tree else ["--cached"]),
            "--raw",
            "-z",
            "--no-renames",
            "--no-abbrev",
        ],
        cwd=root or Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout.split("\0")
    return [(meta.split(), path) for meta, path in zip(rows[::2], rows[1::2], strict=False)]


def staged_moves(
    root: Path | None = None, *, pairs: list[tuple[list[str], str]] | None = None
) -> Moves:
    """Each staged path whose blob equals a path the change deletes, to that path: a move. A row of
    the staged move table pairs an edited move too: the same test, its added lines still judged."""
    pairs = staged_changes(root) if pairs is None else pairs
    gone = {meta[2]: path for meta, path in pairs if meta[4] == "D"}
    moves = {path: gone[meta[3]] for meta, path in pairs if meta[4] == "A" and meta[3] in gone}
    names = [":" + path for meta, path in pairs if meta[4] != "D" and path.startswith("config/")]
    configs = {name[1:]: text for name, text in read_blobs(names, root).items()}
    return moves | table_moves(pairs, configs)


def table_moves(pairs: list[tuple[list[str], str]], configs: Mapping[str, str]) -> Moves:
    """The edited moves a change's own move table pairs: each row names a path this change adds
    and one it deletes. `configs` holds the change's config files by path."""
    table = schema_file(configs, configs.__getitem__, LAYOUT_SCHEMA)
    if table is None:
        return {}
    added = {path for meta, path in pairs if meta[4] == "A"}
    deleted = {path for meta, path in pairs if meta[4] == "D"}
    rows = json.loads(configs[table])["files"]
    return {new: row["old"] for new, row in rows.items() if new in added and row["old"] in deleted}


def unique_moves(pairs: list[tuple[list[str], str]]) -> Moves:
    """Pair only unique deleted/added regular-file blobs with identical modes."""
    removed: dict[tuple[str, str], list[str]] = {}
    added: dict[tuple[str, str], list[str]] = {}
    for meta, path in pairs:
        if len(meta) != 5:
            raise ValueError("Git move metadata is malformed")
        if meta[4] == "D" and meta[0][1:] in {"100644", "100755"}:
            removed.setdefault((meta[0][1:], meta[2]), []).append(path)
        elif meta[4] == "A" and meta[1] in {"100644", "100755"}:
            added.setdefault((meta[1], meta[3]), []).append(path)
    return {
        paths[0]: removed[key][0]
        for key, paths in added.items()
        if len(paths) == 1 and len(removed.get(key, [])) == 1
    }


class CostIdentity:
    """Immutable test-file identity and unique exact moves between two ancestor revisions."""

    def __init__(self, root: Path, source: str, target: str) -> None:
        if not isinstance(root, Path) or any(
            not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha)
            for sha in (source, target)
        ):
            raise ValueError("cost identity needs a Git root and two exact commit SHAs")
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source, target],
            cwd=root,
            capture_output=True,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if ancestry.returncode == 1:
            raise ValueError("cost source head is not an ancestor of the target")
        if ancestry.returncode:
            raise ValueError("cost ancestry is unavailable; restore the source and target commits")
        self.source = self._files(root, source)
        self.target = self._files(root, target)
        pairs = staged_changes(root, tree=target, base=source, timeout=60)
        removed: dict[tuple[str, str], list[str]] = {}
        added: dict[tuple[str, str], list[str]] = {}
        for meta, path in pairs:
            if len(meta) != 5:
                raise ValueError("cost move diff is malformed")
            if meta[4] == "D" and path in self.source:
                removed.setdefault(self.source[path], []).append(path)
            elif meta[4] == "A" and path in self.target:
                added.setdefault(self.target[path], []).append(path)
        self.moves = {
            old[0]: added[key][0]
            for key, old in removed.items()
            if len(old) == 1 and len(added.get(key, [])) == 1
        }

    @staticmethod
    def _files(root: Path, revision: str) -> dict[str, tuple[str, str]]:
        raw = subprocess.run(
            ["git", "ls-tree", "-r", "-z", "--full-tree", revision, "--", "tests/"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        found: dict[str, tuple[str, str]] = {}
        for entry in raw.split(b"\0"):
            if not entry:
                continue
            meta, path = entry.split(b"\t", 1)
            mode, kind, blob = meta.decode("ascii").split()
            name = path.decode("utf-8", errors="surrogateescape")
            if kind == "blob" and mode in {"100644", "100755"}:
                found[name] = (mode, blob)
        return found

    def matches(self, source_node: str, target_node: str) -> bool:
        """Match literal node suffixes and real regular files; moves must be one-to-one."""
        if not isinstance(source_node, str) or not isinstance(target_node, str):
            return False
        old, sep, suffix = source_node.partition("::")
        new, target_sep, target_suffix = target_node.partition("::")
        if not sep or not target_sep or not suffix or suffix != target_suffix:
            return False
        if old not in self.source or new not in self.target:
            return False
        return old == new or self.moves.get(old) == new


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


class CensusInputs:
    """One invocation's immutable tree and lazy, revision-keyed Git inputs."""

    def __init__(self, root: Path, baseline: str | None, fixed: set[str], seed: set[str]) -> None:
        self.root, self.baseline, self.fixed = root, baseline, fixed
        self.stage = (
            subprocess.check_output(
                ["git", "write-tree"],
                cwd=root,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).strip()
            + ":"
        )
        self.paths = {path for _, path in self.changes} if baseline is not None else set()
        listing = (
            subprocess.check_output(
                ["git", "ls-tree", "-r", "-z", "--name-only", self.stage[:-1]],
                cwd=root,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            .decode("utf-8")
            .split("\0")
            if baseline is not None
            else []
        )
        self.sources = {p for p in listing if p.startswith("src/") and p.endswith(".py")}
        self.registries = [
            p
            for p in listing
            if p.startswith("config/registries/")
            and p.endswith(".json")
            and Path(p).name
            not in {"refusals.json", "messages.json", "agent-surface.json", "identity-reach.json"}
        ]
        self.blobs: dict[str, str] = {}
        self.objects: dict[str, str] = {}
        self.known: set[str] = set()
        self.load(
            [
                *(
                    self.stage + p
                    for p in self.paths | fixed | self.sources | set(self.registries) | seed
                ),
                *(
                    "HEAD:" + p
                    for p in seed
                    | set(self.registries)
                    | ({baseline, "config/import-hygiene.json"} if baseline else set())
                ),
                "HEAD",
                "MERGE_HEAD",
            ]
        )
        # The captured raw diff proves unchanged source paths share their HEAD blob.
        for path in self.sources - self.paths:
            staged, prior = self.stage + path, "HEAD:" + path
            if staged in self.objects and prior not in self.known:
                self.blobs[prior] = self.blobs[staged]
                self.objects[prior] = self.objects[staged]
                self.known.add(prior)
        self.pending = self.objects.get("MERGE_HEAD")

    @cached_property
    def parents(self) -> list[str]:
        return subprocess.check_output(
            ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
            cwd=self.root,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).split()

    @cached_property
    def changes(self) -> list[tuple[list[str], str]]:
        return staged_changes(self.root, tree=self.stage[:-1])

    def load(self, specs: Iterable[str]) -> None:
        needed = [key for key in dict.fromkeys(specs) if key not in self.known]
        self.blobs.update(read_blobs(needed, self.root, object_ids=self.objects))
        self.known.update(needed)

    def view(self, specs: Iterable[str]) -> dict[str, str]:
        """Only this reader's keys are visible; prefetched siblings never change membership."""
        keys = list(dict.fromkeys(specs))
        self.load(keys)
        return {key: self.blobs[key] for key in keys if key in self.blobs}
