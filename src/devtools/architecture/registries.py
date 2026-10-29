"""Original registry contracts and the staged census no-growth check.

Six registries name what the product is made of. Three are already held where they live: the
packages (``config/package-architecture.json``, the structural guards), the operations
(``operations.py``, ``operation_registry``) and the identity roles (``config/identity-roles.json``,
the source-identity guards). The other three are held here, each against a baseline in
``config/registries/`` that may only shrink:

Use --staged [--update]; existing registry guards retain their contracts.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import string
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from alphalattice.interface.local_application.message_catalog import english_message_ids
from devtools.architecture.git_inputs import read_blobs
from devtools.architecture.nouns import definitions, owner_of
from devtools.architecture.structural import (
    ImportHygiene,
    PackageArchitectureRegistry,
    discover_table_writes,
    import_contract_violations,
    load_storage_authority_registry,
    table_writer_violations,
)
from devtools.architecture.test_sizes import changed_paths, classify, included, load_policy, scoped

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


def refusals(relative: str, tree: ast.AST, *, locations: bool = False) -> Counter[str]:
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
            found[code] = node.lineno if locations else found[code] + 1
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
    # Evaluation is engineering tooling; other namespaces keep their physical area.
    return sorted(
        package["package_id"]
        for package in document["packages"]
        if (
            "engineering"
            if package["physical_namespace_target"] == "devtools.evaluation"
            else package["physical_namespace_target"].split(".")[1]
        )
        != package["product_area"]
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


def baseline(
    root: Path,
    name: str,
    *,
    cached: Mapping[str, dict[str, Any]] = {},
    errors: list[str] | None = None,
) -> dict[str, Any]:
    if name in cached:
        return cached[name]
    path = root / REGISTRIES / f"{name}.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if name == "refusals":
        try:
            return refusal_entries(document)
        except ValueError as error:
            if errors is None:
                raise
            errors.append(str(error))
            return {}
    return dict(document["entries"])


def _category_row(value: Any, fields: Mapping[str, type[Any]]) -> bool:
    return isinstance(value, dict) and all(
        isinstance(value.get(key), kind) and bool(value[key]) for key, kind in fields.items()
    )


def _refusal_entry(code: str, entry: Any, declared: set[str], notices: Any) -> None:
    if not _category_row(entry, {"owner": str}):
        raise ValueError(f"refusals: {code} category owner missing")
    if entry.get("kind") == "notice" or code in notices:
        if (
            code not in notices
            or entry.get("kind") != "notice"
            or "category" in entry
            or notices[code].get("owner") != entry["owner"]
        ):
            raise ValueError(f"refusals: {code} category notice exemption invalid")
        return
    category = entry.get("category")
    if not isinstance(category, str) or category not in declared:
        raise ValueError(f"refusals: {code} category missing or unknown")


def _check_word_selectors(document: Mapping[str, Any], entries: dict[str, Any]) -> None:
    """A selector is non-emitted word metadata, disjoint from every emitted or legacy key."""
    policy = document.get("classification", {})
    selectors = policy.get("word_selectors", {}) if isinstance(policy, dict) else None
    legacy = document.get("legacy_unregistered_words", [])
    if not isinstance(selectors, dict) or any(
        not isinstance(key, str)
        or not key
        or not _category_row(row, {"owner": str, "reason": str})
        or set(row) != {"owner", "reason"}
        or any(
            not isinstance(group, (dict, list)) or key in group
            for group in (entries, policy.get("notices", {}), legacy)
        )
        for key, row in selectors.items()
    ):
        raise ValueError("refusals: word selector shape or overlap invalid")


def refusal_entries(document: Mapping[str, Any]) -> dict[str, Any]:
    """Check committed categories; assignment rules remain source-review provenance."""
    entries = document.get("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("refusals: entries must be a mapping")
    _check_word_selectors(document, entries)
    if not entries:
        return {}
    categories = document.get("categories", [])
    policy = document.get("classification", {})
    if not _category_row(document, {"categories": list, "classification": dict}):
        raise ValueError("refusals: category declaration invalid")
    if any(not isinstance(c, str) or not c for c in categories) or len(set(categories)) != len(
        categories
    ):
        raise ValueError("refusals: category declaration invalid")
    notices = policy.get("notices", {})
    if not isinstance(notices, dict) or any(
        not _category_row(n, {"owner": str, "reason": str}) for n in notices.values()
    ):
        raise ValueError("refusals: category notice declaration invalid")
    declared = set(categories)
    for code, entry in entries.items():
        _refusal_entry(code, entry, declared, notices)
    return dict(entries)


def _count(value: object) -> int:
    counted = value.get("count", 0) if isinstance(value, dict) else value
    return counted if isinstance(counted, int) else 0


def private_test_growth(base: Mapping[str, object], current: Mapping[str, object]) -> list[str]:
    """What the tests' ratchet holds beyond the registry its change started from (LAWS.md TE5).

    Counts follow the access kind and symbol across files: moving an existing access is not
    growth. A new access or a second use still needs the product's public entry. The scan alone
    let five through on 2026-09-30, each registered beside its test (V16).
    """

    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    for entries, totals in ((base, before), (current, after)):
        for key, value in entries.items():
            totals[key.partition("::")[2]] += _count(value)

    return [
        f"test-private: {key} is registered {after[key.partition('::')[2]]} times where the "
        f"base held {before[key.partition('::')[2]]}; write the test through a public entry "
        "(LAWS.md TE5)"
        for key in sorted(current)
        if after[key.partition("::")[2]] > before[key.partition("::")[2]]
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
    out: list[str] = message_problems(root, files)
    for name, entries in found.items():
        known = baseline(root, name, errors=out)
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


def _refusals_for_shrink(
    root: Path, codes: Any, seed: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    refusal_path = root / REGISTRIES / "refusals.json"
    refusal_document = (
        json.loads(refusal_path.read_bytes()) if refusal_path.is_file() else {"entries": {}}
    )
    registered_refusals = refusal_entries(refusal_document)
    if seed and codes - registered_refusals.keys():
        raise ValueError("refusals: register owner and category before seeding")
    return refusal_document, registered_refusals


def shrink(root: Path, files: Sequence[str], *, seed: bool = False) -> dict[str, int]:
    """Rewrite each baseline to what the tree holds, never growing it unless ``seed``."""

    found = inventory(root, files)
    refusal_document, registered_refusals = _refusals_for_shrink(
        root, found["refusals"].keys(), seed
    )
    sizes: dict[str, int] = {}
    for name, entries in found.items():
        known = baseline(root, name, cached={"refusals": registered_refusals})
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
            **{"refusals": refusal_document}.get(name, {}),
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


TARGETS = {
    "nouns": ("config/registries/owners-map.json", "nouns"),
    "nouns/no_owner": ("config/package-architecture.json", "packages"),
    "operations": ("src/alphalattice/interface/local_application/operations.json", "fields"),
    "message-kinds": ("config/registries/refusals.json", "entries"),
}


def g1_matches(prefix: str, nouns: dict[str, Any]) -> list[dict[str, Any]]:
    """Use declared noun owners and folds; do not infer an owner from package names."""
    matches = []
    for noun, item in nouns.items():
        if item["owner"] == prefix:
            matches.append({"noun": noun, "relation": "OWNER", "owner": item["owner"]})
        for fold in item.get("folds_in", []):
            source = fold["from"]
            if source == prefix or source.startswith(prefix + "."):
                matches.append(
                    {
                        "noun": noun,
                        "relation": "FOLDS_IN",
                        "owner": fold.get("to", item["owner"]),
                        "card": fold["card"],
                        "source": source,
                    }
                )
    return matches


def _census_file_facts(path: str, text: str) -> dict[str, int]:
    tree = ast.parse(text, filename=path)
    return {
        "message-kinds|" + code: line for code, line in refusals(path, tree, locations=True).items()
    } | {
        "operations|" + n.value: n.lineno
        for alias in ast.walk(tree)
        if isinstance(alias, ast.TypeAlias) and alias.name.id == "PortfolioResearchOperation"
        for n in ast.walk(alias.value)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def _census_gaps(facts: dict[str, Any], data: dict[str, Any], text: str) -> dict[str, str]:
    packages = data["config/package-architecture.json"]["packages"]
    nouns = data[TARGETS["nouns"][0]]["nouns"]
    gaps: dict[str, str] = {}
    sites: dict[str, str] = {
        match[1]: f"config/package-architecture.json:{i}"
        for i, line in enumerate(text.splitlines(), 1)
        for match in re.finditer(r'"physical_namespace_target"\s*:\s*"([^"\\]+)"', line)
    }
    for path, items in facts.items():
        module = path.removesuffix(".py")[4:].replace("/", ".").removesuffix(".__init__")
        unit = ".".join(module.split(".")[: 3 if module.startswith("alphalattice.") else 2])
        sites.setdefault(unit, path + ":1")
        if not any(
            (module + ".").startswith(p["physical_namespace_target"] + ".")
            or (p["physical_namespace_target"] + ".").startswith(module + ".")
            for p in packages
        ):
            gaps.setdefault("nouns/no_owner|" + unit, path + ":1")
        for key, line in items.items():
            dimension, item = key.split("|", 1)
            registry, member = TARGETS[dimension]
            if item not in data[registry][member]:
                gaps[key] = f"{path}:{line}"
    for package in packages:
        prefix = package["physical_namespace_target"]
        if not g1_matches(prefix, nouns):
            gaps["nouns|" + package["package_id"]] = sites.get(
                prefix, "config/package-architecture.json:1"
            )
    return gaps


def _message_forms(code: str, row: dict[str, Any]) -> list[str]:
    question = row["kind"] == "PERSON_QUESTION"
    arguments = row.get("arguments", [] if question else None)
    if (
        not isinstance(arguments, list)
        or any(not isinstance(a, str) or not a.isidentifier() for a in arguments)
        or len(set(arguments)) != len(arguments)
    ):
        return [f"messages: {code} invalid arguments"]
    fields = ("words", "reason", "scope", "way_on") if question else ("words",)
    out = []
    for field in fields:
        forms = row.get(field, {})
        for locale in ("en", "zh-CN"):
            text = forms.get(locale) if isinstance(forms, dict) else None
            try:
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("missing form")
                parts = list(string.Formatter().parse(text))
                names = {name for _, name, _, _ in parts if name is not None}
                if names != set(arguments) or any(
                    spec or conversion for _, _, spec, conversion in parts
                ):
                    raise ValueError("named arguments differ")
            except ValueError as error:
                out.append(f"messages: {code}{'.' if question else ' '}{field}.{locale}: {error}")
    return out


def _unique_message_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate member {key}")
        seen[key] = value
    return seen


def _message_aliases(code: str, legacy: Any, aliases: set[str]) -> list[str]:
    if not isinstance(legacy, list):
        return [f"messages: {code} invalid aliases"]
    out = []
    for alias in legacy:
        if not isinstance(alias, str) or not alias.strip() or alias in aliases:
            out.append(f"messages: {code} duplicate or conflicting alias {alias!r}")
        else:
            aliases.add(alias)
    return out


def message_problems(text: str | Path, files: Sequence[str] = ()) -> list[str]:
    """Refuse ambiguous ids and aliases, incomplete locales and incompatible named arguments."""

    if isinstance(text, Path):
        path = text / "config/registries/messages.json"
        if "config/registries/messages.json" not in files or not path.is_file():
            return []
        text = path.read_text("utf-8")
    try:
        entries = json.loads(text, object_pairs_hook=_unique_message_members)["entries"]
        if not isinstance(entries, dict):
            raise ValueError("entries must be an object")
    except (ValueError, KeyError, TypeError) as error:
        return [f"messages: {error}"]
    out, aliases = [], set(entries)
    for code, row in entries.items():
        if not _CODE.fullmatch(code) or not isinstance(row, dict):
            out.append(f"messages: {code} invalid entry")
            continue
        kind, owner = row.get("kind"), row.get("owner")
        if (
            kind not in ("MESSAGE", "PERSON_QUESTION")
            or not isinstance(owner, str)
            or not owner.strip()
        ):
            out.append(f"messages: {code} missing owner or unknown kind")
            continue
        operations = row.get("operations")
        if (kind == "MESSAGE" and row.keys() & {"operations", "reason", "scope", "way_on"}) or (
            kind == "PERSON_QUESTION"
            and (
                not isinstance(operations, list)
                or not operations
                or any(not isinstance(op, str) or not op for op in operations)
            )
        ):
            out.append(f"messages: {code} invalid question contract")
        out.extend(_message_forms(code, row))
        out.extend(_message_aliases(code, row.get("legacy_keys", []), aliases))
    try:
        english_message_ids(entries)
    except ValueError as error:
        out.append(str(error))
    return out


def _legacy_message_keys(text: str) -> Counter[str]:
    match = re.fullmatch(r"\s*(?:/\*.*?\*/\s*)?window\.ALPHA_ZH\s*=\s*(\{.*\});?\s*", text, re.S)
    if not match:
        raise ValueError("unsupported legacy catalog declaration")
    tokens = "\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*'"
    body = re.sub(
        r"/\*.*?\*/|//[^\n]*|" + tokens,
        lambda m: (
            ""
            if m[0].startswith("/")
            else m[0]
            if m[0][0] == '"'
            else json.dumps(ast.literal_eval(m[0]))
        ),
        match[1],
        flags=re.S,
    )
    pairs = json.loads(re.sub(r",\s*}$", "}", body), object_pairs_hook=list)
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in pairs):
        raise ValueError("legacy catalog needs literal string pairs")
    return Counter(k for k, _ in pairs)


def _catalog_factory(tree: ast.Module, names: frozenset[str], spec: Any, historical: bool) -> None:
    factories = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name + "()" in names
    ]
    if spec is None and (historical or not factories):
        return
    if not isinstance(spec, dict) or spec.get("read_only") is not True or len(factories) != 1:
        raise ValueError(
            "legacy factory outside its canonical owner; use config/registries/messages.json"
        )
    factory = factories[0]
    body = factory.body[1:] if ast.get_docstring(factory) is not None else factory.body[:]
    returned = body[-1].value if body and isinstance(body[-1], ast.Return) else None
    match returned:
        case ast.Call(func=ast.Name(id="MappingProxyType"), args=[value], keywords=[]):
            wrapped = True
            body[-1] = ast.Return(value=value)
        case _:
            wrapped = False
    if (
        (not historical and not wrapped)
        or factory.name != spec.get("name")
        or ast.dump(ast.Module(body=body, type_ignores=[])) != spec.get("canonical_body")
    ):
        raise ValueError(
            "legacy factory changed its canonical form; use config/registries/messages.json"
        )


def _consumer_message_keys(
    path: str, text: str, shapes: dict[str, Any], *, historical: bool = False
) -> Counter[str]:
    if not text:
        return Counter()
    if path.endswith("js/data/zh.js"):
        footer = (
            shapes.get("publication_footer")
            or "window.ALPHA_ZH_READY = true;\nObject.freeze(window.ALPHA_ZH);"
        )
        if shapes.get("immutable") and not historical and not text.rstrip().endswith(footer):
            raise ValueError(
                "legacy owner must freeze its table; use config/registries/messages.json"
            )
        return _legacy_message_keys(text.rstrip().removesuffix(footer))
    if path.endswith("zh_literals.json"):
        return Counter(k for r in json.loads(text)["entries"] for k in r.get("keys", []))
    if path.endswith(".py"):
        _catalog_factory(ast.parse(text), shapes["writers"], shapes.get("factory"), historical)
    lines = text.splitlines(keepends=True)
    for read in shapes.get("read_calls", []):
        index = read["line"] - 1
        if (
            index >= len(lines)
            or lines[index].rstrip("\r\n") != read["text"]
            or text.count(read["text"]) != 1
        ):
            if historical and read.get("verification"):
                continue
            raise ValueError(
                f"{path}: registered catalog read changed; use config/registries/messages.json"
            )
        lines[index] = "\n"
    return _catalog_literal_keys(path, "".join(lines), shapes["writers"])


def _catalog_literal_keys(path: str, text: str, writers: frozenset[str]) -> Counter[str]:
    strings: list[str] = []
    quoted = r"""("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')"""

    def token(match: re.Match[str]) -> str:
        if match[0].startswith(("/", "#")):
            return " "
        strings.append(match[0])
        return '"' + str(len(strings) - 1) + '"'

    names = {n for n in writers if not n.endswith("()")}
    if path.endswith(".py"):
        names &= {n.id for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Name)}
    text = re.sub(quoted + r"|/\*.*?\*/|//[^\n]*|#[^\n]*", token, text, flags=re.S)
    found: Counter[str] = Counter()
    declaration = r'\b(\w+)\s*=\s*\{\s*((?:"\d+"\s*:\s*"\d+"\s*,?\s*)*)\}'
    for match in re.finditer(declaration, text):
        if match[1] in names:
            found.update(
                ast.literal_eval(strings[int(k)]) for k in re.findall(r'"(\d+)"\s*:', match[2])
            )
    remainder = (
        re.sub(declaration, lambda m: " " if m[1] in names else m[0], text)
        if path.endswith(".py")
        else text + " ".join(s for s in strings if s[1:-1] in names)
    )
    if any(re.search(r"\b" + re.escape(name) + r"\b", remainder) for name in names):
        raise ValueError(
            f"{path}: catalog outside owner {strings[-10:]}; use config/registries/messages.json"
        )
    extra: Counter[str] = Counter()
    for key, value in re.findall(r'"(\d+)"\s*:\s*"(\d+)"', text):
        k, v = (ast.literal_eval(strings[int(n)]) for n in (key, value))
        if k == v or re.search(r"[^\x00-\x7f]", v):
            extra[k] += 1
    return found | extra


def _catalog_writers(path: str, shapes: dict[str, Any]) -> frozenset[str]:
    default = ["WORDS", "_zh()"] if path.endswith(".py") else ["ALPHA_ZH", "catalog", "WORDS"]
    raw = shapes.get("writers", default)
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(n, str) or not n.strip() for n in raw)
    ):
        raise ValueError(
            "catalog writer coverage is empty or invalid; use config/registries/messages.json"
        )
    names = frozenset(ast.unparse(ast.parse(n.strip(), mode="eval").body) for n in raw)
    if any(re.fullmatch(r"[A-Za-z_]\w*(?:\(\))?", n) is None for n in names):
        raise ValueError("catalog writer must name its owner; use config/registries/messages.json")
    return names


def _catalog_read_contract(
    path: str, before: str, shape: dict[str, Any], previous: dict[str, Any], *, registered: bool
) -> frozenset[str]:
    names = _catalog_writers(path, shape)
    old = (
        names
        if not registered or previous.get("writers") == shape.get("writers")
        else _catalog_writers(path, previous)
    )
    if old - names:
        raise ValueError(
            "catalog writer coverage cannot shrink; use config/registries/messages.json"
        )
    held = previous.get("read_calls", [])
    factory = previous.get("factory")
    if factory:
        factory = {**factory, "canonical_body": (shape.get("factory") or {}).get("canonical_body")}
    if (held and shape.get("read_calls", [])[: len(held)] != held) or any(
        value and shape.get(key) != value
        for key, value in (("factory", factory), ("immutable", previous.get("immutable")))
    ):
        raise ValueError("registered catalog reads are frozen; use config/registries/messages.json")
    lines = before.splitlines()
    for read in shape.get("read_calls", []):
        if (
            read not in held
            and not read.get("verification")
            and (read["line"] > len(lines) or lines[read["line"] - 1] != read["text"])
        ):
            raise ValueError(
                "a new catalog read exemption must be an existing read; "
                "use config/registries/messages.json"
            )
    return names


def _catalog_census(root: Path) -> tuple[list[str], dict[str, Any]]:
    catalog, owners = "config/registries/messages.json", "config/registries/owners-map.json"
    legacy = "src/alphalattice/interface/local_application/assets/workbench-source/js/data/zh.js"
    blobs = read_blobs([":" + catalog, "HEAD:" + catalog, ":" + owners, "HEAD:" + owners], root)
    out = message_problems(blobs[":" + catalog]) if ":" + catalog in blobs else []
    if "HEAD:" + catalog in blobs and ":" + catalog not in blobs:
        out.append("messages: missing catalog; restore config/registries/messages.json")
    row = json.loads(blobs.get(":" + owners, "{}")).get("nouns", {}).get("message catalog", {})
    paths = dict.fromkeys([legacy, *row.get("gate_inputs", row.get("pending_consumers", []))])
    prior = (
        json.loads(blobs.get("HEAD:" + owners, "{}")).get("nouns", {}).get("message catalog", {})
    )
    if not set(prior.get("gate_inputs", prior.get("pending_consumers", []))) <= paths.keys():
        out.append("messages: restore the registered catalog gate inputs")
    basis = "HEAD:"
    pending = subprocess.run(
        ["git", "rev-parse", "--verify", "MERGE_HEAD"], cwd=root, capture_output=True, text=True
    )
    if pending.returncode == 0 and pending.stdout.strip() == row.get("integration_base"):
        basis = pending.stdout.strip() + ":"
    blobs.update(read_blobs([prefix + p for p in paths for prefix in (":", basis)], root))
    if (":" + legacy in blobs or "HEAD:" + legacy in blobs) and not row:
        out.append("messages: missing message catalog owner; restore the owner registry")
    census, consumers = {}, {}
    for path in paths:
        if ":" + path not in blobs and "HEAD:" + path not in blobs:
            continue
        before, after = blobs.get(basis + path, ""), blobs.get(":" + path, "")
        try:
            shape = row.get("gate_inputs", {}).get(path, {})
            previous = prior.get("gate_inputs", {}).get(path, {})
            unchanged = before == after and shape == previous
            shape = {
                **shape,
                "writers": _catalog_read_contract(
                    path, before, shape, previous, registered=path in prior.get("gate_inputs", {})
                ),
            }
            old_shape = shape | previous
            old_shape["writers"] = shape["writers"]
            old_shape["factory"] = previous.get("factory")
            old_shape["publication_footer"] = previous.get("publication_footer")
            old = _consumer_message_keys(path, before, old_shape, historical=True)
            current = old if unchanged else _consumer_message_keys(path, after, shape)
        except (ValueError, SyntaxError) as error:
            out.append(f"messages: {path}: {error}")
            continue
        out.extend(
            f"messages: {path}: new English key {key!r}; use a stable message id"
            for key in current - old
        )
        if path == legacy:
            census = {
                "legacy_message_keys": {
                    "count": len(current),
                    "declarations": current.total(),
                    "keys": dict(sorted(current.items())),
                }
            }
        else:
            consumers[path] = dict(sorted(current.items()))
    return out, census | ({"legacy_catalog_consumers": consumers} if consumers else {})


def _refusal_word_problems(root: Path, stage: str, document: Mapping[str, Any]) -> list[str]:
    """New refusal words require an entry; reviewed non-refusal allowances only shrink."""
    path = TARGETS["message-kinds"][0]
    words = "src/alphalattice/interface/local_application/refusal_words.json"
    blobs = read_blobs(["HEAD:" + path, stage + words], root)
    previous = json.loads(blobs.get("HEAD:" + path, "{}"))
    legacy = set(document.get("legacy_unregistered_words", []))
    selectors = document.get("classification", {}).get("word_selectors", {})
    out = [
        f"refusals: {key} word requires a registered owner and category"
        for key in sorted(
            json.loads(blobs.get(stage + words, "{}")).keys()
            - document.get("entries", {}).keys()
            - legacy
            - selectors.keys()
        )
    ]
    # Current-map disjointness cannot see a selector removed in this same change.
    reviewed_selectors = previous.get("classification", {}).get("word_selectors", {})
    out.extend(
        f"refusals: word selector {key} cannot become an emitted reason"
        for key in sorted(reviewed_selectors.keys() & document.get("entries", {}).keys())
    )
    for name, current, before, container in (
        ("legacy_unregistered_words", legacy, previous, document),
        (
            "word_selectors",
            set(selectors),
            previous.get("classification", {}),
            document.get("classification", {}),
        ),
    ):
        if name in before:
            out.extend(
                f"refusals: {name} cannot widen to {key}"
                for key in sorted(current - set(before[name]))
            )
            if name not in container:
                out.append(f"refusals: {name} declaration cannot be removed")
    return out


def _description_rows(document: Mapping[str, Any]) -> dict[str, Any]:
    """Name registry declarations independently of dictionary or list ordering."""
    rows = {}
    for member in ("entries", "nouns", "kinds", "defaults", "large"):
        values = document.get(member, {})
        if isinstance(values, list):
            # A test size's reason is its purpose sentence; it is not written twice.
            values = {row["pattern"]: {"description": row.get("reason"), **row} for row in values}
        rows.update({f"{member}:{key}": row for key, row in values.items()})
    return rows


def _has_description(row: Any) -> bool:
    """Accept one nonempty sentence; the owner's review proves its meaning."""
    text = row.get("description") if isinstance(row, dict) else None
    return (
        isinstance(text, str)
        and bool(text.strip())
        and "\n" not in text
        and text.rstrip().endswith((".", "!", "?"))
        and re.search(r"[.!?]\s+\S", text) is None
    )


def _description_census(
    root: Path, stage: str, blobs: dict[str, str], data: dict[str, Any], base: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Read staged declarations once and ratchet their remaining explanation debt."""
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", stage[:-1], "--", "config/registries"],
        cwd=root,
        text=True,
    ).splitlines()
    excluded = {"refusals.json", "messages.json", "agent-surface.json", "identity-reach.json"}
    paths = [p for p in paths if p.endswith(".json") and Path(p).name not in excluded]
    dimension = "undescribed_registry_entries"
    bootstrap = dimension not in base
    inputs = [
        prefix + path for path in paths for prefix in ([stage, "HEAD:"] if bootstrap else [stage])
    ]
    blobs.update(read_blobs([key for key in inputs if key not in blobs], root))
    missing: set[str] = set()
    previous: set[str] = set()
    for path in paths:
        if path not in data:
            data[path] = json.loads(blobs[stage + path])
        missing.update(
            path + "::" + key
            for key, row in _description_rows(data[path]).items()
            if not _has_description(row)
        )
        if bootstrap:
            previous.update(
                path + "::" + key
                for key, row in _description_rows(
                    json.loads(blobs.get("HEAD:" + path, "{}"))
                ).items()
                if not _has_description(row)
            )
    allowed = set(base.get(dimension, previous))
    return [
        f"registry-description: {key} needs one purpose sentence."
        for key in sorted(missing - allowed)
    ], sorted(missing)


def census_check(root: Path, *, baseline: str | None = None, update: bool = False) -> list[str]:
    """Ratchet exact staged findings; unchanged source comes only from cached facts."""
    out, catalog = _catalog_census(root)
    stage = subprocess.check_output(["git", "write-tree"], cwd=root, text=True).strip() + ":"
    refusal_key = stage + TARGETS["message-kinds"][0]
    refusal_blob = read_blobs([refusal_key], root).get(refusal_key, '{"entries":{}}')
    try:
        refusal_document = json.loads(refusal_blob)
        refusal_entries(refusal_document)
        out.extend(_refusal_word_problems(root, stage, refusal_document))
    except ValueError as error:
        return [*out, str(error)]
    if baseline is None:
        print("registry ratchet not configured: set ALPHALATTICE_REGISTRY_BASELINE")
        return out
    anchor = Path(baseline)
    baseline = (anchor.relative_to(root.resolve()) if anchor.is_absolute() else anchor).as_posix()
    command = ["git", "diff", "HEAD", stage[:-1], "--name-only", "--no-renames", "-z"]
    paths = set(subprocess.check_output(command, cwd=root).decode().split("\0")) - {""}
    legacy_path = "config/import-hygiene.json"
    fixed = {baseline, legacy_path, "config/package-architecture.json"}
    fixed.update(p for p, _ in TARGETS.values())
    blobs = read_blobs(
        [
            "HEAD:" + baseline,
            "HEAD:" + legacy_path,
            *(stage + p for p in (paths | fixed) - {TARGETS["message-kinds"][0]}),
        ],
        root,
    )
    blobs[refusal_key] = refusal_blob
    data = {p: json.loads(blobs[stage + p]) for p in fixed}
    base = json.loads(blobs.get("HEAD:" + baseline, blobs[stage + baseline]))
    try:
        legacy = ImportHygiene.model_validate(data[legacy_path]).legacy_lazy_imports
    except ValueError as error:
        return [*out, f"import-hygiene: {error}"]
    prior = json.loads(blobs.get("HEAD:" + legacy_path, "{}")).get("legacy_lazy_imports", legacy)
    out.extend(
        f"import-hygiene: retained import allowance grew {key}"
        for key, count in legacy.items()
        if count > prior.get(key, 0)
    )
    if (
        "HEAD:" + baseline not in blobs
        and base["source_sha"]
        != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    ):
        return [f"registry-census: {baseline} must name the reviewed bootstrap source SHA."]
    facts = base.get("by_file", {}).copy()
    description_errors, undescribed = _description_census(root, stage, blobs, data, base)
    out.extend(description_errors)
    for path in sorted(p for p in paths if p.startswith("src/") and p.endswith(".py")):
        if stage + path in blobs:
            facts[path] = _census_file_facts(path, blobs[stage + path])
        else:
            facts.pop(path, None)
    gaps = _census_gaps(facts, data, blobs[stage + "config/package-architecture.json"])

    def check_imports(prefix: str = stage) -> list[str]:
        checked = {p for p in facts if p.startswith("src/") and p.endswith(".py")}
        if prefix != stage:
            checked = {k.split("::", 1)[0] for k in legacy}
        elif not paths & {"config/package-architecture.json", legacy_path}:
            checked.intersection_update(paths)
        blobs.update(read_blobs([prefix + p for p in checked if prefix + p not in blobs], root))
        sources = {
            p[4:-3].replace("/", "."): blobs[prefix + p] for p in checked if prefix + p in blobs
        }
        known = {p[4:-3].replace("/", ".").removesuffix(".__init__") for p in facts}
        policy = PackageArchitectureRegistry.model_validate(
            data["config/package-architecture.json"] | {"legacy_lazy_imports": legacy}
        )
        return [
            *(f"stale-legacy-import {p}" for p in checked if prefix + p not in blobs),
            *import_contract_violations(sources, registry=policy, modules=known),
        ]

    out.extend(check_imports())
    if "HEAD:" + legacy_path not in blobs:
        out.extend(p for p in check_imports("HEAD:") if p.startswith("stale-legacy-import"))
    nouns = sorted(data[TARGETS["nouns"][0]]["nouns"])
    expected = {
        "by_file": facts,
        "automatic_gaps": gaps,
        "registered_nouns": nouns,
        "legacy_lazy_imports": legacy,
        "undescribed_registry_entries": undescribed,
    } | catalog

    def sync_baseline() -> None:
        for key in sorted(gaps.keys() - base.get("automatic_gaps", {}).keys()):
            kind, item = key.split("|", 1)
            registry, member = TARGETS[kind]
            form = f"{member}[{item!r}]"
            if kind == "nouns/no_owner":
                form = f"packages[] with package_id and physical_namespace_target={item!r}"
            out.append(f"Unregistered {kind} {item} {gaps[key]}; add {registry} {form}.")
        if not out and any(data[baseline].get(key) != value for key, value in expected.items()):
            if update:
                (root / baseline).write_text(
                    json.dumps(data[baseline] | expected, indent=2) + "\n", encoding="utf-8"
                )
            else:
                removed = ", ".join(
                    sorted(
                        (base.get("automatic_gaps", {}).keys() - gaps.keys())
                        | (set(base.get("registered_nouns", [])) - set(nouns))
                    )
                )
                out.append(
                    f"Sync {baseline} (removed: {removed or 'none'}) "
                    "with --staged --update; stage it."
                )

    sync_baseline()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--shrink", action="store_true", help="drop entries the tree no longer has")
    mode.add_argument("--seed", action="store_true", help="write the first baselines")
    mode.add_argument("--staged", action="store_true", help="check the staged census delta")
    parser.add_argument("--baseline", default=os.environ.get("ALPHALATTICE_REGISTRY_BASELINE"))
    parser.add_argument("--update", action="store_true", help="refresh the staged census baseline")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)
    if args.update and not args.staged:
        parser.error("--update requires --staged")
    if args.shrink or args.seed:
        # A baseline is the whole tree's: shrinking from a few files would drop the rest.
        print(json.dumps(shrink(args.root, tracked(args.root), seed=args.seed), sort_keys=True))
        return 0
    if args.staged:
        found = census_check(args.root, baseline=args.baseline, update=args.update)
    else:
        found = problems(args.root, args.files or tracked(args.root))
    for line in found:
        print(line)
    return 1 if found else 0


CHANGED = pytest.StashKey[frozenset[str] | None]()  # a scoped run's changed paths, read once


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Materialize each test's registered size and declaring owner during collection."""

    lane = os.environ.get("ALPHALATTICE_TEST_LANE", "")
    changed = changed_paths() if lane in {"merge", "fast"} else None
    empty = pytest.mark.skip(reason="the change holds no Python file for this per_file scan")
    python = any(path.endswith(".py") for path in changed or ())
    config.stash[CHANGED] = changed
    kept: list[Any] = []
    omitted: list[Any] = []
    seen: set[str] = set()
    try:
        policy = load_policy(Path(__file__).resolve().parents[3])
        if lane and lane not in policy["lanes"]:
            raise ValueError("test_size.lane_unknown")
        for item in items:
            if lane and item.nodeid in seen:
                omitted.append(item)
                continue
            seen.add(item.nodeid)
            declared = [m.name for m in item.iter_markers() if m.name in policy["sizes"]]
            size, owner, declaration, seconds = classify(item.nodeid, declared, policy)
            if not declared:
                item.add_marker(size)
            previous = item.get_closest_marker("timeout")
            if previous:
                limit = previous.args[0] if previous.args else previous.kwargs.get("timeout")
                if isinstance(limit, (int, float)) and limit > 0:
                    seconds = min(seconds, limit)
            item.add_marker(pytest.mark.timeout(seconds, func_only=False), append=False)
            item.user_properties.extend(
                [
                    ("test_size", size),
                    ("test_size_owner", owner),
                    ("test_size_declaration", declaration),
                    ("test_timeout_seconds", seconds),
                ]
            )
            if scoped(item.nodeid, policy, changed):
                item.add_marker("per_file")  # it reads only the changed files, through scanned()
                if not python:
                    item.add_marker(empty)
            (kept if included(item.nodeid, size, lane, policy, changed) else omitted).append(item)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    if lane:
        config.hook.pytest_deselected(items=omitted)
        items[:] = kept


if __name__ == "__main__":
    sys.exit(main())
