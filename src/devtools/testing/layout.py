"""Test placement map and whole-file mover; added files use the same path owner."""

from __future__ import annotations

import ast
import io
import json
import keyword
import ntpath
import re
import subprocess
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, cast

from .layout_references import declarations as reference_declarations
from .layout_references import images as reference_images
from .layout_references import provenance as reference_provenance
from .layout_references import source_paths as reference_source_paths
from .requirements import (
    FACTS_SCHEMA,
    LAYERS,
    LAYOUT_SCHEMA,
    LEGACY_SCHEMA,
    file_holders,
    schema_file,
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


def registries(files: Mapping[str, str]) -> dict[str, str]:
    """Each registry a move rewrites, by role: the public ones by path, the private legacy count
    and host facts by their schema when this tree holds them (never named by path)."""
    found = {"sizes": REGISTRIES[0], "private": REGISTRIES[1]}
    for role, schema in (("legacy", LEGACY_SCHEMA), ("facts", FACTS_SCHEMA)):
        if (path := schema_file(files, files.__getitem__, schema)) is not None:
            found[role] = path
    return found


DISCOVERY = "from tests._support.project_discovery import project_discovery as project_discovery\n"


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


def _host_fact_image(document: Any, modules: dict[str, str], path: str) -> dict[str, Any]:
    """Move declaration identities; observations, payloads and ordering stay intact."""
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


def _post_merge_paths(paths: list[str], moves: dict[str, str]) -> list[str]:
    """Carry exact deferred paths without merging distinct registered files."""
    result = [moves.get(path, path) for path in paths]
    if len(result) != len(set(result)):
        raise ValueError("test_layout.registry_collision:post_merge_only")
    return result


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
    for role, path in registries(files).items():
        if path not in files:
            continue
        document = json.loads(files[path])
        original = document.copy() if isinstance(document, dict) else document
        if role == "facts":
            document = _host_fact_image(document, modules, path)
        elif role != "sizes":
            field = "entries" if role == "private" else "files"
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
            if "post_merge_only" in document:
                document["post_merge_only"] = _post_merge_paths(document["post_merge_only"], moves)
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


def snapshot(
    root: Path, revision: str, *, reference_paths: Sequence[str] = ()
) -> tuple[str, dict[str, str]]:
    """Read one immutable Git tree once, without checking out or importing it."""
    source = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=root, text=True
    ).strip()
    paths = reference_source_paths(reference_paths)
    available = set(
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", source, "--", *sorted(paths)],
            cwd=root,
            text=True,
        ).splitlines()
    )
    if paths - available:
        raise ValueError(
            "test_layout.registry_invalid: reference source absent: "
            + ",".join(sorted(paths - available))
        )
    roots = ("src", "scripts", "tests", "config")  # all of config: private data is found by schema
    extra = sorted(
        path for path in paths if not any(path == p or path.startswith(p + "/") for p in roots)
    )
    raw = subprocess.check_output(
        [
            "git",
            "archive",
            source,
            *roots,
            *extra,
        ],
        cwd=root,
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as tree:
        for p in tree:  # Python, the config JSON a registry may be (by schema), and named inputs
            schema = p.name.startswith("config/") and p.name.endswith(".json")
            if p.isfile() and (p.name.endswith(".py") or schema or p.name in paths):
                try:
                    files[p.name] = cast(BinaryIO, tree.extractfile(p)).read().decode("utf-8")
                except UnicodeError as error:
                    raise ValueError(f"test_layout.input_unreadable:{p.name}: {error}") from None
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


def _wire_images(
    files: dict[str, str],
    moves: dict[str, str],
    modules: dict[str, str],
    images: dict[str, list[str | None]],
) -> None:
    """Rewrite imported identities without changing the previously checked image footprint."""
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


def plan(
    source: str,
    files: dict[str, str],
    placements: dict[str, list[str]],
    *,
    fixture_closures: dict[str, list[str]] | None = None,
    references: Sequence[Mapping[str, Any]] = (),
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
    coupled, source_references = reference_images(
        files,
        moves,
        modules,
        reference_declarations(references),
        blocked=(*registries(files).values(), *images),
    )
    images.update(coupled)
    _wire_images(files, moves, modules, images)
    for path in list(images):
        if not path.startswith("tests/"):
            continue
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
        "source_references": source_references,
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


def _check_windows_image_name(path: str) -> None:
    """The shared Win32 component-name preflight for every layout image."""
    reference_source_paths((path,))
    reserved = getattr(ntpath, "isreserved", None)
    if any(
        part.rstrip(" .") != part
        or any(ord(char) < 32 or char in '<>"|?*' for char in part)
        or (reserved(part) if reserved else PureWindowsPath(part).is_reserved())
        for part in PurePosixPath(path).parts
    ):
        raise ValueError(
            "test_layout.registry_invalid: reference move images need canonical Windows names"
        )


def _reference_python_module(path: str) -> str:
    """Keep Python file and dotted module identities losslessly equivalent."""
    _check_windows_image_name(path)
    module = path[:-3].replace("/", ".")
    if not path.endswith(".py") or module.replace(".", "/") + ".py" != path:
        raise ValueError(
            "test_layout.registry_invalid: reference move images need canonical Python paths"
        )
    return module


def _check_reference_modules(
    value: dict[str, Any], source_moves: dict[str, str], forward: bool
) -> None:
    """Derive allowed module identities from move and fixture images."""
    moves, modules = value["moves"], value["modules"]
    expected = {_reference_python_module(p): _reference_python_module(q) for p, q in moves.items()}
    for context, pair in value["files"].items():
        if not context.endswith("/conftest.py"):
            continue
        provider = "tests/_support/" + PurePosixPath(context).parent.name + "_fixtures.py"
        exported = pair[1 if forward else 0] if isinstance(pair, list) and len(pair) == 2 else None
        if (
            any(p.startswith(str(PurePosixPath(context).parent) + "/") for p in source_moves)
            and provider in value["files"]
            and isinstance(exported, str)
            and ("from " + provider[:-3].replace("/", ".") + " import ") in exported
        ):
            before, after = _reference_python_module(context), _reference_python_module(provider)
            expected[before if forward else after] = after if forward else before
    if modules != expected:
        raise ValueError(
            "test_layout.registry_invalid: reference module map does not derive from moves"
        )
    source_modules = modules if forward else {b: a for a, b in modules.items()}
    _check_reference_move_images(value, source_moves, source_modules, forward)


def _check_reference_move_images(
    value: dict[str, Any],
    source_moves: dict[str, str],
    source_modules: dict[str, str],
    forward: bool,
) -> None:
    """Bind one-to-one origins and destinations to their checked source images."""
    unique = len({path.casefold() for path in source_moves.values()}) == len(source_moves) and len(
        {path.casefold() for path in source_moves}
    ) == len(source_moves)
    for old, new in source_moves.items():
        origin, destination = value["files"].get(old, []), value["files"].get(new, [])
        if (
            not unique
            or not all(isinstance(pair, list) and len(pair) == 2 for pair in (origin, destination))
            or origin[1 if forward else 0] is not None
            or destination[0 if forward else 1] is not None
            or not isinstance(origin[0 if forward else 1], str)
            or not isinstance(destination[1 if forward else 0], str)
            or _wire(
                origin[0 if forward else 1],
                old[:-3].replace("/", ".").rpartition(".")[0],
                source_modules,
            )
            != destination[1 if forward else 0]
        ):
            raise ValueError(
                "test_layout.registry_invalid: reference move images do not bind module identities"
            )


def _reference_direction(
    value: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, str], bool]:
    """Bind recorded source identities to one forward or inverse move direction."""
    moves, modules = value["moves"], value["modules"]
    targets = {row["target"] for row in rows}
    forward = targets <= moves.keys() and not targets & set(moves.values())
    backward = targets <= set(moves.values()) and not targets & moves.keys()
    if not (forward or backward):
        raise ValueError(
            "test_layout.registry_invalid: reference direction does not bind the moves"
        )
    source_moves = moves if forward else {b: a for a, b in moves.items()}
    _check_reference_modules(value, source_moves, forward)
    source_modules = modules if forward else {b: a for a, b in modules.items()}
    return source_moves, source_modules, forward


def _checked_reference_paths(value: dict[str, Any]) -> set[str]:
    """Replay coupled source images before admitting any additional write footprint."""
    records = value.get("source_references", [])
    rows = reference_provenance(records)
    if not rows:
        return set()
    if len({path.casefold() for path in value["files"]}) != len(value["files"]):
        raise ValueError("test_layout.registry_invalid: reference move images have aliased paths")
    source_moves, source_modules, forward = _reference_direction(value, rows)
    moves = value["moves"]
    paths = {row["path"] for row in rows}
    if paths & (moves.keys() | set(moves.values())):
        raise ValueError("test_layout.registry_invalid: prepare moving-body references first")
    sources = {}
    for path in paths:
        pair = value["files"].get(path)
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(v, str) for v in pair)
        ):
            raise ValueError(
                "test_layout.registry_invalid: reference images cannot create or delete files"
            )
        sources[path] = pair[0 if forward else 1]
    expected, observed = reference_images(sources, source_moves, source_modules, rows)
    if observed != records:
        raise ValueError("test_layout.registry_invalid: reference source records changed")
    for path, pair in expected.items():
        if path.endswith(".py"):
            pair[1] = _wire(pair[1] or "", "", source_modules)
        if value["files"][path] != (pair if forward else list(reversed(pair))):
            raise ValueError(
                "test_layout.registry_invalid: non-location reference payload changed:" + path
            )
    return paths


def _checked_image_path(root: Path, name: str, allowed: set[str]) -> Path:
    """The file one image rewrites: under the code roots or allowed by name, never an alias."""
    if (
        not (name.startswith(("tests/", "src/", "scripts/")) or name in allowed)
        or "\\" in name
        or ":" in name
        or ".." in PurePosixPath(name).parts
    ):
        raise ValueError("test_layout.path_invalid:" + name)
    _check_windows_image_name(name)
    path = root / name
    resolved = path.resolve()
    if (
        not resolved.is_relative_to(root)
        or resolved != path.absolute()
        or any(p.is_symlink() for p in [path, *path.parents])
    ):
        raise ValueError("test_layout.path_invalid:" + name)
    return path


def apply(root: Path, value: dict[str, Any]) -> list[str]:
    """Preflight every image and reject aliases before any file changes."""
    root = root.resolve()
    references = _checked_reference_paths(value)
    images = {n: b or a or "" for n, (b, a) in value["files"].items() if n.startswith("config/")}
    allowed = set(registries(images).values()) | set(references)  # private ones by their schema
    paths = {}
    for name, (before, _after) in value["files"].items():
        path = _checked_image_path(root, name, allowed)
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
