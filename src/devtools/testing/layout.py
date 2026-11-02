"""One-time mover: retire with old selectors unless a current caller remains; saved maps stay."""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import tarfile
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast

from .requirements import LAYERS

REGISTRIES = (
    "config/registries/test-sizes.json",
    "config/registries/test-private.json",
    "config/registries/tests-unregistered.json",
    "config/agent-eval/host-facts.json",
)
DISCOVERY = "from tests._support.project_discovery import project_discovery as project_discovery\n"


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


def _rules(
    rows: list[dict[str, Any]], key: str, tests: Iterable[str], moves: dict[str, str]
) -> list[dict[str, Any]]:
    """Preserve row payloads and refuse conflicting destination declarations."""
    mapped: dict[str, dict[str, Any]] = {}
    widths: dict[str, int] = {}
    positions: dict[str, int] = {}
    for index, row in sorted(enumerate(rows), key=lambda p: (-len(p[1][key]), p[1][key])):
        for pattern in rule_patterns(row[key], tests, moves):
            entry = row | {key: pattern}
            if "inputs" in entry:
                entry["inputs"] = [
                    v for p in entry["inputs"] for v in rule_patterns(p, tests, moves)
                ]
            if pattern in mapped:
                existing = mapped[pattern]
                if widths[pattern] == len(row[key]) and (
                    existing.get("owner"),
                    existing.get("size", "large"),
                ) != (entry.get("owner"), entry.get("size", "large")):
                    raise ValueError("test_layout.registry_collision:" + pattern)
                continue
            mapped[pattern] = entry
            widths[pattern] = len(row[key])
            positions[pattern] = index
    return sorted(mapped.values(), key=lambda row: positions[row[key]])


def _host_fact_image(document: Any, modules: dict[str, str]) -> dict[str, Any]:
    """Move declaration identities; observations, payloads and ordering stay intact."""
    path = REGISTRIES[3]
    if not isinstance(document, dict) or not isinstance(document.get("fakes"), list):
        raise ValueError("test_layout.registry_invalid:" + path + "; restore fake declarations")
    renamed = {a.removesuffix(".__init__"): b.removesuffix(".__init__") for a, b in modules.items()}
    mapped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in document["fakes"]:
        key = row.get("fake") if isinstance(row, dict) else None
        if (
            not isinstance(key, str)
            or key.count(":") != 1
            or not key.startswith("tests.")
            or not all(key.partition(":")[::2])
        ):
            raise ValueError(
                "test_layout.registry_invalid:" + path + "; use tests.module:symbol keys"
            )
        module, _, symbol = key.partition(":")
        target = renamed.get(module, module) + ":" + symbol
        if target in seen:
            raise ValueError("test_layout.registry_collision:" + target)
        seen.add(target)
        mapped.append(row | {"fake": target})
    return document | {"fakes": mapped}


def _registry_images(
    files: dict[str, str], moves: dict[str, str], modules: dict[str, str]
) -> dict[str, list[str | None]]:
    """Carry size rules, private-use counts and legacy ownership in the same footprint."""
    images: dict[str, list[str | None]] = {}
    tests = [
        p
        for p in files
        if p.startswith("tests/") and Path(p).name.startswith("test_") and p.endswith(".py")
    ]
    for path in REGISTRIES:
        if path not in files:
            continue
        document = json.loads(files[path])
        original = document.copy() if isinstance(document, dict) else document
        if path == REGISTRIES[3]:
            document = _host_fact_image(document, modules)
        elif path != REGISTRIES[0]:
            field = "entries" if path == REGISTRIES[1] else "files"
            mapped = {}
            for key, value in document[field].items():
                old, separator, suffix = key.partition("::")
                target = moves.get(old, old) + separator + suffix
                if target in mapped:
                    raise ValueError("test_layout.registry_collision:" + target)
                mapped[target] = value
            document[field] = mapped
        else:
            for field, key in (
                ("defaults", "pattern"),
                ("large", "pattern"),
                ("reach", "pattern"),
                ("quarantine", "node"),
            ):
                if field in document:
                    document[field] = _rules(document[field], key, tests, moves)
            if "mandatory" in document:
                document["mandatory"] = list(
                    dict.fromkeys(
                        v for p in document["mandatory"] for v in rule_patterns(p, tests, moves)
                    )
                )
        after = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        if document != original:
            images[path] = [files[path], after]
    return images


def _retire_bindings(
    files: dict[str, str],
    moves: dict[str, str],
    images: dict[str, list[str | None]],
    contexts: Iterable[str],
) -> None:
    """The last test leaving an old area retires its temporary discovery binding."""
    for context in contexts:
        area = str(PurePosixPath(context).parent) + "/"
        if not any(
            p.startswith(area) and Path(p).name.startswith("test_") and p not in moves
            for p in files
        ):
            before, after = images.get(context, [files[context], files[context]])
            images[context] = [before, after.replace(DISCOVERY, "") if after is not None else None]


def snapshot(root: Path, revision: str) -> tuple[str, dict[str, str]]:
    """Read one immutable Git tree once, without checking out or importing it."""
    source = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=root, text=True
    ).strip()
    raw = subprocess.check_output(
        [
            "git",
            "archive",
            source,
            "src",
            "scripts",
            "tests",
            "config/registries",
            *(
                [REGISTRIES[3]]
                if subprocess.check_output(
                    ["git", "ls-tree", "--name-only", source, "--", REGISTRIES[3]], cwd=root
                ).strip()
                else []
            ),
        ],
        cwd=root,
    )
    with tarfile.open(fileobj=io.BytesIO(raw)) as tree:
        files = {
            p.name: cast(BinaryIO, tree.extractfile(p)).read().decode("utf-8")
            for p in tree
            if p.isfile() and (p.name.endswith(".py") or p.name in REGISTRIES)
        }
    return source, files


def _fixture_exports(
    tree: ast.Module, functions: list[ast.FunctionDef | ast.AsyncFunctionDef]
) -> list[str]:
    """Keep Python export names distinct from pytest's optional fixture aliases."""
    exported = [n.name for n in functions]
    exported += [
        name
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
        for name in ast.literal_eval(n.value)
    ]
    return sorted(set(exported))


def _fixtures(text: str, consumed: set[str] | None = None) -> tuple[list[str], bool, bool]:
    tree = ast.parse(text)
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(ast.unparse(d).startswith("pytest.fixture") for d in n.decorator_list)
    ]
    session: set[str] = set()
    for n in functions:
        for d in n.decorator_list:
            if not (isinstance(d, ast.Call) and ast.unparse(d.func) == "pytest.fixture"):
                continue
            if any(k.arg is None for k in d.keywords):
                raise ValueError("test_layout.fixture_scope:declaration")
            keywords = {k.arg: k.value for k in d.keywords}
            try:
                scope = ast.literal_eval(keywords.get("scope", ast.Constant("function")))
                name = ast.literal_eval(keywords.get("name", ast.Constant(n.name)))
            except (ValueError, TypeError):
                raise ValueError("test_layout.fixture_scope:declaration") from None
            if (scope is not None and not isinstance(scope, str)) or (
                name is not None and not isinstance(name, str)
            ):
                raise ValueError("test_layout.fixture_scope:declaration")
            if scope == "session":
                session.add(name or n.name)
    return (
        _fixture_exports(tree, functions),
        "autouse=True" in text,
        bool(session if consumed is None else session.intersection(consumed)),
    )


def _fixture_consumers(
    closures: dict[str, list[str]] | None, contexts: dict[str, list[str]]
) -> tuple[dict[str, list[str]] | None, dict[str, set[str]]]:
    """Validate complete collected evidence and project it once to each inherited provider."""
    if closures is None:
        return None, {}
    if (
        not isinstance(closures, dict)
        or set(closures) != set(contexts)
        or any(
            not isinstance(names, list)
            or any(not isinstance(name, str) or not name for name in names)
            for names in closures.values()
        )
    ):
        raise ValueError("test_layout.fixture_scope:collected_closures")
    observed = {p: sorted(set(names)) for p, names in closures.items()}
    consumers: dict[str, set[str]] = {}
    for path, names in observed.items():
        for context in contexts[path]:
            consumers.setdefault(context, set()).update(names)
    return observed, consumers


def _wire(text: str, package: str, modules: dict[str, str]) -> str:

    def absolute(m: re.Match[str]) -> str:
        if not package:
            return m[0]
        base = package.split(".")[: len(package.split(".")) - len(m[2]) + 1]
        return m[1] + ".".join([*base, *filter(None, [m[3]])]) + m[4]

    text = re.sub(r"(?m)^(\s*from )(\.+)([\w.]*)( import .*)$", absolute, text)
    for before, after in sorted(modules.items(), key=lambda p: -len(p[0])):
        text = re.sub(
            r"(?m)^(\s*(?:from|import)\s+)" + re.escape(before) + r"(?=[.\s]|$)",
            r"\g<1>" + after,
            text,
        )
    return text


def plan(
    source: str,
    files: dict[str, str],
    placements: dict[str, list[str]],
    *,
    fixture_closures: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Render whole-file images; collisions and incompatible fixture visibility refuse."""
    images: dict[str, list[str | None]] = {}
    moves: dict[str, str] = {}
    contexts: dict[str, list[str]] = {}
    bindings: dict[tuple[str, str], str] = {}

    def put(path: str, text: str | None) -> None:
        if path in images and images[path][1] != text:
            raise ValueError("test_layout.path_collision:" + path)
        images[path] = [files.get(path), text]

    def add_move(old: str, choice: list[str]) -> None:
        layer, domain, stem, *scope = choice
        if layer not in LAYERS or not all(
            re.fullmatch(r"[a-z][a-z0-9_]*", p) for p in [domain, stem, *scope]
        ):
            raise ValueError("test_layout.path_invalid")
        target = "/".join(["tests", layer, domain, *scope, "test_" + stem + ".py"])
        if (
            old not in files
            or not old.startswith("tests/")
            or target in files
            or target.casefold() in {p.casefold() for p in moves.values()}
        ):
            raise ValueError("test_layout.path_collision:" + target)
        moves[old] = target
        put(old, None)
        put(target, files[old])
        contexts[old] = [
            str(p / "conftest.py")
            for p in PurePosixPath(old).parents
            if str(p) not in {".", "tests"} and str(p / "conftest.py") in files
        ]

    for old, choice in placements.items():
        add_move(old, choice)
    observed, consumers = _fixture_consumers(fixture_closures, contexts)
    modules = {p[:-3].replace("/", "."): q[:-3].replace("/", ".") for p, q in moves.items()}
    inherited = {
        c: {str(PurePosixPath(moves[p]).parent) for p in moves if c in contexts[p]}
        for c in {c for values in contexts.values() for c in values}
    }

    def inherit(context: str, destinations: set[str]) -> None:
        if files[context].strip() == DISCOVERY.strip():
            return
        provider = "tests/_support/" + PurePosixPath(context).parent.name + "_fixtures.py"
        raw = files.get(provider, files[context])
        names, autouse, session = _fixtures(raw, consumers.get(context))
        export = "".join(
            "from " + provider[:-3].replace("/", ".") + " import " + n + " as " + n + "\n"
            for n in names
        )
        if "__file__" in raw or not names or (session and len(destinations) > 1):
            raise ValueError("test_layout.fixture_scope:" + context)
        if provider in files and files[context].replace(DISCOVERY, "") not in {
            raw.replace(DISCOVERY, ""),
            export,
        }:
            raise ValueError("test_layout.path_collision:" + provider)
        put(provider, _wire(raw, context[:-3].replace("/", ".").rpartition(".")[0], modules))
        put(context, export + (DISCOVERY if DISCOVERY in files[context] else ""))
        modules[context[:-3].replace("/", ".")] = provider[:-3].replace("/", ".")
        for destination in destinations:
            if autouse and any(
                context not in contexts[p]
                for p in moves
                if str(PurePosixPath(moves[p]).parent) == destination
            ):
                raise ValueError("test_layout.fixture_visibility:" + destination)
            for name in names:
                key = destination, name
                if key in bindings and bindings[key] != context:
                    raise ValueError("test_layout.fixture_collision:" + name)
                bindings[key] = context
            path = destination + "/conftest.py"
            prior = images.get(path, [None, files.get(path, "")])[1] or ""
            if path in files and set(names).intersection(_fixtures(files[path])[0]):
                raise ValueError("test_layout.fixture_collision:" + path)
            images[path] = [files.get(path), prior + export]

    for context, destinations in inherited.items():
        inherit(context, destinations)
    for old, text in files.items():
        if not old.endswith(".py"):
            continue
        target = moves.get(old, old)
        current = images.get(target, [None, text])[1]
        if current is not None:
            wired = _wire(
                current,
                old[:-3].replace("/", ".").rpartition(".")[0] if old in moves else "",
                modules,
            )
            if any(before in wired for before in modules):
                raise ValueError("test_layout.unmapped_reference:" + old)
            if wired != current:
                images[target] = [files.get(target), wired]
    for path in list(images):
        for parent in PurePosixPath(path).parents:
            init = str(parent / "__init__.py")
            if str(parent) not in {".", "tests"} and init not in files and init not in images:
                put(init, "")
    _retire_bindings(files, moves, images, inherited)
    images.update(_registry_images(files, moves, modules))
    return {
        "schema": "alphalattice.test-layout.v1",
        "source": source,
        "fixture_closures": observed,
        "moves": moves,
        "modules": modules,
        "files": images,
    }


def inverse(value: dict[str, Any]) -> dict[str, Any]:
    """Invert exactly the recorded footprint, retaining the source revision and decisions."""
    return value | {
        "moves": {b: a for a, b in value["moves"].items()},
        "modules": {b: a for a, b in value["modules"].items()},
        "files": {p: list(reversed(v)) for p, v in value["files"].items()},
    }


def apply(root: Path, value: dict[str, Any]) -> list[str]:
    """Preflight every image and reject aliases before any file changes."""
    root = root.resolve()
    paths = {}
    for name, (before, _after) in value["files"].items():
        path = root / name
        resolved = path.resolve()
        if (
            not (name.startswith(("tests/", "src/", "scripts/")) or name in REGISTRIES)
            or "\\" in name
            or ":" in name
            or ".." in PurePosixPath(name).parts
            or not resolved.is_relative_to(root)
            or resolved != path.absolute()
            or any(p.is_symlink() for p in [path, *path.parents])
        ):
            raise ValueError("test_layout.path_invalid:" + name)
        if (path.read_bytes().decode("utf-8") if path.is_file() else None) != before or (
            path.exists() and not path.is_file()
        ):
            raise ValueError("test_layout.preimage_changed:" + name)
        paths[name] = path
    for name, (_before, after) in sorted(value["files"].items(), key=lambda p: p[1][1] is None):
        path = paths[name]
        if after is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(after.encode("utf-8"))
    return list(paths)
