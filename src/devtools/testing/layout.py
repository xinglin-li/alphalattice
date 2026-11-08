"""Test placement and the recorded-path projections used by current guards."""

from __future__ import annotations

import ast
import json
import keyword
from collections.abc import Iterable, Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from .requirements import (
    LAYERS,
    LAYOUT_SCHEMA,
    file_holders,
    schema_path,
)
from .sizes import _declarations, whole_tree_scope

REGISTRIES = ("config/registries/test-sizes.json", "config/registries/test-private.json")  # public


def placed_tests(root: Path) -> dict[str, str]:
    """Where each moved test is now, by its old path. A source inside an identity closure keeps
    naming the path it froze, since rewriting it would rotate that identity for a test move."""
    table = schema_path(root, LAYOUT_SCHEMA)  # a published tree has none
    rows = json.loads(table.read_text("utf-8"))["files"] if table else {}
    return {row["old"]: path for path, row in rows.items()}


def _target(layer: str, domain: str, stem: str, scope: Sequence[str]) -> str:
    """The shared layout map for both existing moves and newly added tests."""
    return "/".join(["tests", layer, domain, *scope, "test_" + stem + ".py"])


def _source_module(source: str, sources: Iterable[str]) -> None:
    parts = PurePosixPath(source).parts
    module = [*parts[1:-1], PurePosixPath(source).stem]
    if (
        not source.startswith(("src/", "scripts/"))
        or "\\" in source
        or PurePosixPath(source).as_posix() != source
        or any(not part.isidentifier() or keyword.iskeyword(part) for part in module)
        or not source.endswith(".py")
        or source not in sources
    ):
        raise ValueError(
            f"source {source!r} is not a staged source module; "
            "declare one existing src/*.py or scripts/*.py module"
        )


def _placement_owner(
    path: str, text: str, rows: Mapping[str, Mapping[str, Any]]
) -> tuple[str, str]:
    holders = file_holders(path, text)
    ids = {key for holder in holders for key in holder.ids}
    if (
        not holders
        or not ids
        or any(not holder.ids for holder in holders)
        or any(key not in rows or rows[key].get("retired") for key in ids)
    ):
        raise ValueError(
            "register the tests' requirement owner and layer and mark each test with proves"
        )
    choices = {(rows[key].get("owner"), rows[key].get("layer")) for key in ids}
    if len(choices) != 1:
        raise ValueError(
            "tests have mixed owners or layers; split the file by registered owner and layer"
        )
    owner, layer = next(iter(choices))
    if (
        not isinstance(owner, str)
        or not isinstance(layer, str)
        or layer not in LAYERS
        or not all(part.isidentifier() and not keyword.iskeyword(part) for part in owner.split("."))
    ):
        raise ValueError("invalid registered owner or layer; repair the requirement row")
    return owner, layer


def new_test_file_problems(
    path: str, text: str, rows: Mapping[str, Mapping[str, Any]], sources: Iterable[str]
) -> list[str]:
    """Apply the layout map to a newly added file using its single declared source."""
    try:
        tree = ast.parse(text)
        if path.startswith("tests/structural/"):
            if not whole_tree_scope(tree):
                raise ValueError("declare TEST_SCOPE = 'whole-tree' for this structural file")
            return []
        stated = _declarations(tree, "TEST_SOURCE")
        if (
            len(stated) != 1
            or not isinstance(stated[0], ast.Constant)
            or not isinstance(stated[0].value, str)
        ):
            raise ValueError(
                "declare exactly one literal TEST_SOURCE as a src/ or scripts/ Python path"
            )
        source = stated[0].value
        _source_module(source, sources)
        owner, layer = _placement_owner(path, text, rows)
        prefix = "src/" + owner.replace(".", "/")
        if source.startswith("scripts/"):
            relative = PurePosixPath(source.removeprefix("scripts/"))
        else:
            if not source.startswith(prefix + "/"):
                raise ValueError(
                    f"source {source} is outside owner {owner}; "
                    "declare its owned source or correct the requirement owner"
                )
            relative = PurePosixPath(source[len(prefix) + 1 :])
        if relative.name == "__init__.py":
            stem = relative.parts[-2] if len(relative.parts) > 1 else owner.rsplit(".", 1)[-1]
            scope = relative.parts[:-2]
        else:
            stem, scope = relative.stem, relative.parts[:-1]
        target = _target(layer, owner.rsplit(".", 1)[-1], stem, scope)
        if path != target:
            raise ValueError(f"new test file is outside its source owner; move it to {target}")
    except (SyntaxError, UnicodeError, ValueError) as error:
        return [f"TEST-LAYOUT {path}: {error}; repair the declaration or move the file"]
    return []


def new_data_file_problems(paths: Iterable[str], policy: Mapping[str, Any]) -> list[str]:
    """Refuse each new data file a test brings until the size policy names it with its reason."""
    registered = {row["path"] for row in policy.get("data_files", [])}
    return [
        f"TEST-DATA {path}: a test's new data file needs a reason; add "
        f'{{"path": "{path}", "reason": "why a test needs these bytes"}} to '
        f"data_files in {REGISTRIES[0]}, or build the data inside the test"
        for path in paths
        if path not in registered
    ]


def rule_patterns(pattern: str, files: Iterable[str], moves: dict[str, str]) -> list[str]:
    """Split a path rule without changing the remaining files or its node suffix."""
    path, separator, suffix = pattern.partition("::")
    changed = [old for old in moves if fnmatchcase(old, path)]
    remaining = any(p not in moves and fnmatchcase(p, path) for p in files)
    return ([pattern] if not changed or remaining else []) + [
        moves[old]
        + (separator + suffix if separator else "*" if any(c in path for c in "*?[") else "")
        for old in changed
    ]
