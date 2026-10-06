"""A mapped noun's decisions made outside its owner (G2; G1's owner map; LAWS.md OW1, OW6, OP12).

A noun's owner defines it: its types, its state and its decisions. Another package may import the
noun's types and read them; it does not make the noun's decisions. A decision is recognised by what
it makes, not by the file it sits in: a class the owner's defining modules export whose name says
it is decided -- a receipt, a marker, a pointer, an admission, a decision, a publication, a
recommendation, a dossier -- constructed, sealed or subclassed outside the owner. The splits the
tree holds are a baseline that only shrinks (`config/registries/noun-definitions.json`); a new one
is refused, naming the owner to write it in.
"""

from __future__ import annotations

import ast
import json
from functools import cache
from pathlib import Path

DECISION_SUFFIXES = (
    "Admission",
    "Decision",
    "Dossier",
    "Marker",
    "Pointer",
    "Publication",
    "Receipt",
    "Recommendation",
)
"""The names that say a type is decided: a record of what an owner judged, sealed or pointed at."""

_BUILDERS = frozenset({"create", "model_construct"})


@cache
def _decision_types(root: Path) -> dict[str, dict[str, tuple[str, tuple[str, ...]]]]:
    """By defining module: each decision type's noun and the owner packages it belongs to."""

    path = root / "config/registries/owners-map.json"
    if not path.is_file():  # a tree without an owner map maps no noun
        return {}
    document = json.loads(path.read_text("utf-8"))
    found: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {}
    for noun, row in document["nouns"].items():
        owners = (row["owner"], *row.get("with", ()))
        for module in row["defined_by"]:
            base = root.joinpath("src", *module.split("."))
            source = base.with_suffix(".py")
            if not source.is_file():
                source = base / "__init__.py"
            if not source.is_file():
                continue
            for node in ast.parse(source.read_text(encoding="utf-8")).body:
                if (
                    isinstance(node, ast.ClassDef)
                    and not node.name.startswith("_")
                    and node.name.endswith(DECISION_SUFFIXES)
                ):
                    found.setdefault(module, {})[node.name] = (noun, owners)
    return found


def _module_of(relative: str) -> str:
    parts = relative.removesuffix(".py").split("/")[1:]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _inside(module: str, owners: tuple[str, ...]) -> bool:
    return any(module == owner or module.startswith(owner + ".") for owner in owners)


def definitions(root: Path, relative: str, tree: ast.AST) -> set[str]:
    """`path::noun::Type` for each decision type a module outside its owner makes.

    Args:
        root: The checkout, whose owner map names the nouns.
        relative: The module's path, POSIX, from the checkout.
        tree: The module, parsed.

    Returns:
        One key per noun and type the module constructs, seals or subclasses.
    """
    if not relative.startswith("src/alphalattice/"):
        return set()
    module = _module_of(relative)
    types = _decision_types(root)
    by_package = {
        (defining, name): value
        for defining, names in types.items()
        for name, value in names.items()
    }
    named: dict[str, tuple[str, str, tuple[str, ...]]] = {}  # local name -> (Type, noun, owners)
    modules: dict[str, str] = {}  # local name -> a defining module imported whole
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                for (defining, name), (noun, owners) in by_package.items():
                    if alias.name == name and (
                        node.module == defining or _inside(node.module, owners)
                    ):
                        named[alias.asname or alias.name] = (name, noun, owners)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in types:
                    modules[alias.asname or alias.name] = alias.name
    made: set[str] = set()

    def resolve(node: ast.expr) -> tuple[str, str, tuple[str, ...]] | None:
        if isinstance(node, ast.Name):
            return named.get(node.id)
        if isinstance(node, ast.Attribute):
            owner_module = _dotted(node.value)
            defining = modules.get(owner_module or "") or owner_module
            value = types.get(defining or "", {}).get(node.attr)
            return None if value is None else (node.attr, *value)
        return None

    for node in ast.walk(tree):
        found: tuple[str, str, tuple[str, ...]] | None = None
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                found = resolve(base) or found
        elif isinstance(node, ast.Call):
            func = node.func
            found = resolve(func)
            if found is None and isinstance(func, ast.Attribute) and func.attr in _BUILDERS:
                found = resolve(func.value)
            if found is None and node.args and _called_name(func).startswith("seal"):
                found = resolve(node.args[0])
        if found is not None and not _inside(module, found[2]):
            made.add(f"{relative}::{found[1]}::{found[0]}")
    return made


def owner_of(root: Path, key: str) -> str:
    """The owner a `path::noun::Type` key's noun names in the owner map."""

    noun = key.split("::")[1]
    document = json.loads((root / "config/registries/owners-map.json").read_text("utf-8"))
    return str(document["nouns"][noun]["owner"])


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return None if inner is None else f"{inner}.{node.attr}"
    return None


def _called_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


__all__ = ["DECISION_SUFFIXES", "definitions", "owner_of"]
