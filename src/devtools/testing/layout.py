"""One-time mover: retire with old selectors unless a current caller remains; saved maps stay."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, cast


def snapshot(root: Path, revision: str) -> tuple[str, dict[str, str]]:
    """Read one immutable Git tree once, without checking out or importing it."""
    source = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=root, text=True
    ).strip()
    raw = subprocess.check_output(["git", "archive", source, "src", "scripts", "tests"], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tree:
        files = {
            p.name: cast(BinaryIO, tree.extractfile(p)).read().decode("utf-8")
            for p in tree
            if p.isfile() and p.name.endswith(".py")
        }
    return source, files


def _fixtures(text: str) -> tuple[list[str], bool, bool]:
    tree = ast.parse(text)
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any(ast.unparse(d).startswith("pytest.fixture") for d in n.decorator_list)
    ]
    exported = [n.name for n in functions]
    exported += [
        name
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
        for name in ast.literal_eval(n.value)
    ]
    return (
        sorted(set(exported)),
        "autouse=True" in text,
        bool(re.search(r"scope=[\"']session[\"']", text)),
    )


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


def plan(source: str, files: dict[str, str], placements: dict[str, list[str]]) -> dict[str, Any]:
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
        if layer not in {"unit", "integration", "contract", "e2e", "native"} or not all(
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
    modules = {p[:-3].replace("/", "."): q[:-3].replace("/", ".") for p, q in moves.items()}
    inherited = {
        c: {str(PurePosixPath(moves[p]).parent) for p in moves if c in contexts[p]}
        for c in {c for values in contexts.values() for c in values}
    }

    def inherit(context: str, destinations: set[str]) -> None:
        provider = "tests/_support/" + PurePosixPath(context).parent.name + "_fixtures.py"
        raw = files.get(provider, files[context])
        names, autouse, session = _fixtures(raw)
        export = "".join(
            "from " + provider[:-3].replace("/", ".") + " import " + n + " as " + n + "\n"
            for n in names
        )
        if "__file__" in raw or not names or (session and len(destinations) > 1):
            raise ValueError("test_layout.fixture_scope:" + context)
        if provider in files and files[context] not in {raw, export}:
            raise ValueError("test_layout.path_collision:" + provider)
        put(provider, _wire(raw, context[:-3].replace("/", ".").rpartition(".")[0], modules))
        put(context, export)
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
    return {
        "schema": "alphalattice.test-layout.v1",
        "source": source,
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
            not name.startswith(("tests/", "src/", "scripts/"))
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
