"""The registries the gate holds: what the tree may add only by registering it (binding plan G0).

Six registries name what the product is made of. Three are already held where they live: the
packages (``config/package-architecture.json``, the structural guards), the operations
(``operations.py``, ``operation_registry``) and the identity roles (``config/identity-roles.json``,
the source-identity guards). The other three are held here, each against a baseline in
``config/registries/`` that may only shrink:

- ``parameters``: every numeric literal of the number-deciding code, by file, the name it is bound
  to and its value, each in one class -- METHOD (it decides a result), EXECUTION (how fast or how
  quietly, never what; LAWS.md PA2) or LIMIT (a bound the product keeps);
- ``refusals``: every failure code an owner raises or answers, by its owner;
- ``formats``: every persisted schema name and Task kind, with the file that declares it.

And the tests' ratchet: no new import of a private name, and no new patch of a private attribute,
in ``tests/`` (LAWS.md TE5); and the owners' (G2): no new decision of a mapped noun outside its
owner (``devtools.architecture.nouns``, against ``noun-definitions``), and no new module in the
Host unless it is registered as its composition (``host-modules``; LAWS.md OW6): a module the
owner map moves out is DOMAIN, and those only go; and no table written outside its registered
owners (``config/storage-authorities.json``; another package's data writes held in
``table-writers``, only shrinking). A new entry is refused with the registry and the
line to add; one that went away is dropped by ``--shrink``. The check reads only the files it is
given (the gate gives the changed ones), so it costs what the change costs.

    python -m devtools.architecture.registries [--shrink] [FILE ...]
    python -m devtools.architecture.where <noun>
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from devtools.architecture.nouns import definitions, owner_of
from devtools.architecture.structural import (
    discover_table_writes,
    load_storage_authority_registry,
    table_writer_violations,
)

REGISTRIES = Path("config") / "registries"
_CLASSES = frozenset({"METHOD", "EXECUTION", "LIMIT"})
_PARAMETER_ROOTS = (
    "src/alphalattice/capabilities/",
    "src/alphalattice/foundation/",
    "src/alphalattice/investment/",
    "src/alphalattice/evidence/",
    "src/alphalattice/oversight/",
)
_TRIVIAL = frozenset({0, 1, -1})
_CODE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")
_EXECUTION_WORDS = ("thread", "worker", "batch", "chunk", "cache", "timeout", "seconds", "poll")
_LIMIT_WORDS = ("max", "min", "limit", "bound", "capacity", "budget", "cap", "size", "length")


def _bound_name(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    """The name a literal is bound to: its assignment target, keyword, or enclosing function."""

    current: ast.AST | None = node
    while current is not None:
        parent = parents.get(id(current))
        if isinstance(parent, ast.keyword) and parent.arg:
            return parent.arg
        if isinstance(parent, ast.Assign | ast.AnnAssign):
            target = parent.targets[0] if isinstance(parent, ast.Assign) else parent.target
            if isinstance(target, ast.Name):
                return target.id
            if isinstance(target, ast.Attribute):
                return target.attr
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            return parent.name
        current = parent
    return "<module>"


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def parameters(relative: str, tree: ast.AST) -> Counter[str]:
    """``path::name::value`` of each numeric literal a number-deciding module binds."""

    if not relative.startswith(_PARAMETER_ROOTS):
        return Counter()
    parents = _parents(tree)
    found: Counter[str] = Counter()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
            and node.value not in _TRIVIAL
            and not isinstance(parents.get(id(node)), ast.Subscript | ast.Slice)
        ):
            value = node.value
            parent = parents.get(id(node))
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
                value = -value
            found[f"{relative}::{_bound_name(node, parents)}::{value!r}"] += 1
    return found


def refusals(relative: str, tree: ast.AST) -> Counter[str]:
    """Each failure code a module raises or answers, by its text."""

    if not relative.startswith("src/"):
        return Counter()
    parents = _parents(tree)
    found: Counter[str] = Counter()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        code = node.value.split(":", 1)[0]
        if not _CODE.match(code) or code.startswith(("alphalattice.", "src.")):
            continue
        parent = parents.get(id(node))
        raised = isinstance(parent, ast.Call) and isinstance(parents.get(id(parent)), ast.Raise)
        keyed = isinstance(parent, ast.keyword) and parent.arg == "failure_code"
        valued = isinstance(parent, ast.Dict) and any(
            isinstance(key, ast.Constant) and key.value == "failure_code" and value is node
            for key, value in zip(parent.keys, parent.values, strict=True)
        )
        if raised or keyed or valued:
            found[code] += 1
    return found


def formats(relative: str, tree: ast.AST) -> Counter[str]:
    """Each persisted schema name and Task kind a module declares."""

    if not relative.startswith("src/"):
        return Counter()
    found: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                if target.id == "TASK_KIND" or target.id.endswith("_TASK_KIND"):
                    found[f"task_kind:{node.value.value}"] += 1
                elif target.id.endswith("_SCHEMA") and node.value.value.startswith("alphalattice."):
                    found[f"schema:{node.value.value}"] += 1
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "schema"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value.startswith("alphalattice.")
                ):
                    found[f"schema:{value.value}"] += 1
    return found


def test_private(relative: str, tree: ast.AST) -> Counter[str]:
    """``path::name`` of each private name a test imports or patches."""

    if not relative.startswith("tests/"):
        return Counter()
    found: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("alphalattice"):
            for alias in node.names:
                if alias.name.startswith("_"):
                    found[f"{relative}::import {alias.name}"] += 1
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "setattr" and len(node.args) >= 2:
                attribute = node.args[1]
                if (
                    isinstance(attribute, ast.Constant)
                    and isinstance(attribute.value, str)
                    and attribute.value.startswith("_")
                    and not attribute.value.startswith("__")
                ):
                    found[f"{relative}::patch {attribute.value}"] += 1
            elif name == "patch" and node.args:
                target = node.args[0]
                if (
                    isinstance(target, ast.Constant)
                    and isinstance(target.value, str)
                    and target.value.startswith("alphalattice.")
                    and target.value.rsplit(".", 1)[-1].startswith("_")
                ):
                    found[f"{relative}::patch {target.value.rsplit('.', 1)[-1]}"] += 1
    return found


def owners(relative: str, tree: ast.AST) -> Counter[str]:
    """Each semantic owner a source-identity site names; each must be a registered package."""

    if not relative.startswith("src/"):
        return Counter()
    found: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                value = keyword.value
                if (
                    keyword.arg == "semantic_owner"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    found[value.value] += 1
    return found


def areas(root: Path) -> list[str]:
    """Each registered package whose product area is not the namespace it lives in."""

    document = json.loads((root / "config" / "package-architecture.json").read_text("utf-8"))
    return sorted(
        package["package_id"]
        for package in document["packages"]
        if package["physical_namespace_target"].split(".")[1] != package["product_area"]
    )


_SCANNERS = {
    "parameters": parameters,
    "refusals": refusals,
    "formats": formats,
    "test-private": test_private,
}


def inventory(root: Path, files: Iterable[str]) -> dict[str, Counter[str]]:
    """Each registry's entries in ``files`` (relative, POSIX)."""

    result: dict[str, Counter[str]] = {name: Counter() for name in _SCANNERS}
    for relative in files:
        path = root / relative
        if not relative.endswith(".py") or not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for name, scan in _SCANNERS.items():
            result[name].update(scan(relative, tree))
    return result


def baseline(root: Path, name: str) -> dict[str, object]:
    path = root / REGISTRIES / f"{name}.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return dict(document["entries"])


def _count(value: object) -> int:
    counted = value.get("count", 0) if isinstance(value, dict) else value
    return counted if isinstance(counted, int) else 0


def private_test_growth(base: Mapping[str, object], current: Mapping[str, object]) -> list[str]:
    """What the tests' ratchet holds beyond the registry its change started from (LAWS.md TE5).

    The ratchet only shrinks: a new test reaches the product through a public entry, so an entry
    registered by hand is refused here. The scan alone let five through on 2026-09-30, each
    registered beside its test (V16).
    """

    return [
        f"test-private: {key} is registered {_count(value)} times where the base held "
        f"{_count(base.get(key, 0))}; write the test through a public entry (LAWS.md TE5)"
        for key, value in sorted(current.items())
        if _count(value) > _count(base.get(key, 0))
    ]


def _format_problems(root: Path, known: dict[str, object]) -> list[str]:
    """A format that does not name its version, readers and upgraders, or names a file that is
    not there (LAWS.md DA6, V267)."""

    out: list[str] = []
    for key, value in sorted(known.items()):
        entry = value if isinstance(value, dict) else {}
        if not isinstance(entry.get("version"), list) or not isinstance(
            entry.get("upgraders"), list
        ):
            out.append(f"formats: {key} names no version or upgraders")
        readers = entry.get("readers")
        if not isinstance(readers, list) or not readers:
            out.append(f"formats: {key} names no reader")
            continue
        for path in (*readers, *entry.get("upgraders", ())):
            if not (root / str(path)).is_file():
                out.append(f"formats: {key} names {path}, which is not a file")
    return out


def problems(root: Path, files: Sequence[str]) -> list[str]:
    """What ``files`` add to a registry without registering it."""

    found = inventory(root, files)
    out: list[str] = []
    for name, entries in found.items():
        known = baseline(root, name)
        for key, count in sorted(entries.items()):
            registered = known.get(key)
            if name in {"refusals", "formats"}:
                # A code or a format is registered once; its uses are not counted.
                if registered is None:
                    out.append(f"{name}: {key} is not registered in {REGISTRIES}/{name}.json")
                continue
            allowed = 0 if registered is None else _count(registered)
            if count > allowed:
                out.append(
                    f"{name}: {key} occurs {count} times, {allowed} registered "
                    f"({REGISTRIES}/{name}.json)"
                )
        if name == "parameters":
            for key, value in sorted(known.items()):
                if isinstance(value, dict) and value.get("class") not in _CLASSES:
                    out.append(f"parameters: {key} has no class (METHOD, EXECUTION or LIMIT)")
        if name == "formats":
            out.extend(_format_problems(root, known))
    registered = {
        package["package_id"]
        for package in json.loads(
            (root / "config" / "package-architecture.json").read_text("utf-8")
        )["packages"]
    }
    for relative in files:
        path = root / relative
        if relative.endswith(".py") and path.is_file():
            for owner in owners(relative, ast.parse(path.read_text(encoding="utf-8"))):
                # A package, or one of its areas (`alpha_research.calibration`); an owner that
                # names neither is refused unless the baseline already held it (its value is
                # hashed into an identity, so renaming it is a successor of its own).
                if owner.split(".")[0] not in registered and owner not in baseline(root, "owners"):
                    out.append(f"{relative}: semantic_owner {owner!r} is not a registered package")
    known_definitions = baseline(root, "noun-definitions")
    for key in sorted(_definitions(root, files)):
        if key not in known_definitions:
            maker, noun, made = key.split("::")
            out.append(
                f"{maker}: makes {made}, a decision of {noun}; write it in "
                f"{owner_of(root, key)} (the owner map, config/registries/owners-map.json)"
            )
    sources = tuple(
        root / relative
        for relative in files
        if relative.startswith("src/alphalattice/")
        and relative.endswith(".py")
        and (root / relative).is_file()
    )
    storage = root / "config" / "storage-authorities.json"
    if sources and storage.is_file():
        out.extend(
            table_writer_violations(
                discover_table_writes(sources, source_root=root / "src"),
                load_storage_authority_registry(storage),
                other_data_writers=baseline(root, "table-writers"),
                whole_tree=False,
            )
        )
    known_host = baseline(root, "host-modules")
    for relative in files:
        if (
            relative.startswith(_HOST)
            and relative.endswith(".py")
            and (root / relative).is_file()
            and relative not in known_host
        ):
            out.append(
                f"{relative}: a new module in the Host, which keeps composition only (LAWS.md "
                "OW6): write a noun's decisions in its owner (devtools where <noun>), or register "
                f"this module as COMPOSITION in {REGISTRIES}/host-modules.json"
            )
    drifted = set(areas(root)) - set(baseline(root, "areas"))
    out.extend(
        f"package {package}: its product_area is not the namespace it lives in"
        for package in sorted(drifted)
    )
    return out


def _proposed_class(key: str) -> str:
    name = key.split("::")[1].lower()
    if any(word in name for word in _EXECUTION_WORDS):
        return "EXECUTION"
    if any(word in name for word in _LIMIT_WORDS):
        return "LIMIT"
    return "METHOD"


def _declaring_files(root: Path, files: Sequence[str], key: str) -> list[str]:
    """The files among ``files`` that declare the format ``key``."""

    return [
        relative
        for relative in files
        if relative.endswith(".py")
        and (root / relative).is_file()
        and key in formats(relative, ast.parse((root / relative).read_text(encoding="utf-8")))
    ]


def shrink(root: Path, files: Sequence[str], *, seed: bool = False) -> dict[str, int]:
    """Rewrite each baseline to what the tree holds, never growing it unless ``seed``."""

    found = inventory(root, files)
    sizes: dict[str, int] = {}
    for name, entries in found.items():
        known = baseline(root, name)
        kept: dict[str, object] = {}
        for key, count in sorted(entries.items()):
            registered = known.get(key)
            if registered is None and not seed:
                continue
            if name == "parameters":
                chosen = (
                    registered.get("class")
                    if isinstance(registered, dict)
                    else _proposed_class(key)
                )
                limit = count if seed else min(count, _count(registered))
                kept[key] = {"class": chosen, "count": limit}
            elif name == "test-private":
                kept[key] = count if seed else min(count, _count(registered))
            elif name == "formats" and registered is None:
                # A seeded format names the file that declares it as its first reader; its
                # version and upgraders are the author's to fill (DA6).
                kept[key] = {
                    "owner": key.split(".")[0],
                    "version": [],
                    "readers": _declaring_files(root, files, key),
                    "upgraders": [],
                }
            else:
                kept[key] = registered if registered is not None else {"owner": key.split(".")[0]}
        path = root / REGISTRIES / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema": f"alphalattice.playpen.registry.{name}",
            "note": __doc__.split("\n\n")[1].replace("\n", " ") if name == "parameters" else "",
            "entries": kept,
        }
        path.write_bytes((json.dumps(document, indent=0, sort_keys=True) + "\n").encode("utf-8"))
        sizes[name] = len(kept)
    known_definitions = baseline(root, "noun-definitions")
    kept_definitions = {
        key: {"owner": owner_of(root, key)}
        for key in sorted(_definitions(root, files))
        if seed or key in known_definitions
    }
    path = root / REGISTRIES / "noun-definitions.json"
    document = {
        "schema": "alphalattice.playpen.registry.noun-definitions",
        "note": "G2: each module that makes a mapped noun's decision outside its owner, as the "
        "tree held them; it only shrinks as the owner cards move them.",
        "entries": kept_definitions,
    }
    path.write_bytes((json.dumps(document, indent=0, sort_keys=True) + "\n").encode("utf-8"))
    sizes["noun-definitions"] = len(kept_definitions)
    known_host = baseline(root, "host-modules")
    moving = _moving_modules(root)
    kept_host = {
        relative: (
            known_host[relative]
            if relative in known_host and not seed
            else ("DOMAIN" if relative in moving else "COMPOSITION")
        )
        for relative in files
        if relative.startswith(_HOST) and (seed or relative in known_host)
    }
    path = root / REGISTRIES / "host-modules.json"
    document = {
        "schema": "alphalattice.playpen.registry.host-modules",
        "note": "G2 (LAWS.md OW6): the Host's modules. DOMAIN ones the owner map moves to their "
        "owners and only go; a new module is registered here as COMPOSITION, or goes to its owner.",
        "entries": kept_host,
    }
    path.write_bytes((json.dumps(document, indent=0, sort_keys=True) + "\n").encode("utf-8"))
    sizes["host-modules"] = len(kept_host)
    known_areas = baseline(root, "areas")
    drift = {
        package: known_areas.get(package, "drifted when the registries were seeded")
        for package in areas(root)
        if seed or package in known_areas
    }
    path = root / REGISTRIES / "areas.json"
    document = {"schema": "alphalattice.playpen.registry.areas", "note": "", "entries": drift}
    path.write_bytes((json.dumps(document, indent=0, sort_keys=True) + "\n").encode("utf-8"))
    sizes["areas"] = len(drift)
    registered = {
        package["package_id"]
        for package in json.loads(
            (root / "config" / "package-architecture.json").read_text("utf-8")
        )["packages"]
    }
    known_owners = baseline(root, "owners")
    unowned: dict[str, object] = {}
    for relative in files:
        path = root / relative
        if relative.endswith(".py") and path.is_file():
            for owner in owners(relative, ast.parse(path.read_text(encoding="utf-8"))):
                if owner.split(".")[0] not in registered and (seed or owner in known_owners):
                    unowned[owner] = known_owners.get(owner, relative)
    path = root / REGISTRIES / "owners.json"
    document = {"schema": "alphalattice.playpen.registry.owners", "note": "", "entries": unowned}
    path.write_bytes((json.dumps(document, indent=0, sort_keys=True) + "\n").encode("utf-8"))
    sizes["owners"] = len(unowned)
    return sizes


_HOST = "src/alphalattice/control/product_host/"


def _moving_modules(root: Path) -> set[str]:
    """The Host's module paths the owner map folds into their owners, a package taking its tree."""

    path = root / REGISTRIES / "owners-map.json"
    if not path.is_file():
        return set()
    document = json.loads(path.read_text("utf-8"))
    moving: set[str] = set()
    for row in document["nouns"].values():
        for item in row["folds_in"]:
            base = root.joinpath("src", *str(item["from"]).split("."))
            if base.with_suffix(".py").is_file():
                moving.add(base.with_suffix(".py").relative_to(root).as_posix())
            elif base.is_dir():
                moving.update(
                    path.relative_to(root).as_posix()
                    for path in base.rglob("*.py")
                    if "__pycache__" not in path.parts
                )
    return moving


def _definitions(root: Path, files: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for relative in files:
        path = root / relative
        if relative.endswith(".py") and path.is_file():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            found |= definitions(root, relative, tree)
    return found


def tracked(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for base in ("src", "tests")
        for path in (root / base).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--shrink", action="store_true", help="drop entries the tree no longer has")
    parser.add_argument("--seed", action="store_true", help="write the first baselines")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)
    if args.shrink or args.seed:
        # A baseline is the whole tree's: shrinking from a few files would drop the rest.
        print(json.dumps(shrink(args.root, tracked(args.root), seed=args.seed), sort_keys=True))
        return 0
    found = problems(args.root, args.files or tracked(args.root))
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
