"""Every identity built from source: where it is built, what it tracks, what moves it.

The product binds numerical implementations to source through the primitives in
``kernel/shared_kernel/source_identity.py``. This reads their call sites from the syntax
tree, importing nothing but the rule closure's own primitive, and answers two questions
(binding plan, B5, B14 and R1):

- for a closure, which files it tracks, each classed by what kind of code it is; a
  platform file (composition, storage, telemetry, task control, the build files) is a
  candidate to leave under B5's rule, and is reported, never refused;
- for a file, which identities an edit to it rotates.

``source_rule_closure_hash`` call sites whose tracked paths are literal (a tuple of strings, a
prefix joined to each member of one, a named constant of either) are resolved exactly; a rule
closure's tracked paths are its entries and every module the rule adds to them. The rest, whose
modules are resolved at runtime, are listed by site with their owner and role, so no closure
escapes the inventory.

    python -m devtools.architecture.identity_closures [--file PATH ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    control_arithmetic_rule_identity,
    installed_method_family_owners,
    method_family_rule_identity,
)
from alphalattice.kernel.shared_kernel import source_identity
from alphalattice.kernel.shared_kernel.source_identity import (
    IDENTITY_ROLES_PATH,
    number_deciding_closure,
    number_deciding_rule,
)
from devtools.architecture import structural

PRIMITIVES = frozenset({"source_rule_closure_hash"})
_LISTED = frozenset({"source_rule_closure_hash"})
_DEFINITIONS = "src/alphalattice/kernel/shared_kernel/source_identity.py"
_PLATFORM = (
    ("BUILD", ("pyproject.toml", "uv.lock")),
    ("COMPOSITION", ("/composition/",)),
    ("STORAGE", ("/storage/",)),
    ("TELEMETRY", ("/telemetry/",)),
    ("TASK_CONTROL", ("/task_control/",)),
    ("WORKSPACE_RUNTIME", ("/workspace_runtime/",)),
    ("INTERFACE", ("src/alphalattice/interface/",)),
)


@dataclass(frozen=True, slots=True)
class ClosureSite:
    """One call of a source-identity primitive."""

    path: str
    line: int
    function: str
    primitive: str
    semantic_owner: str | None
    numerical_role: str | None
    tracked_paths: tuple[str, ...] | None
    """The literal tracked paths, or None when they are computed at runtime."""

    @property
    def name(self) -> str:
        return f"{self.numerical_role or '?'} ({self.path}:{self.line})"


def path_class(tracked: str) -> str:
    """What kind of code a tracked file is: a platform class, or DOMAIN."""

    for label, needles in _PLATFORM:
        if any(tracked == n or tracked.endswith("/" + n) or n in tracked for n in needles):
            return label
    return "DOMAIN"


def closure_sites(root: Path) -> tuple[ClosureSite, ...]:
    """Every call of the three primitives under ``src`` and ``scripts``.

    The answer is kept whole in the line's scan store (``structural.FileFacts``) by the bytes of
    every file it reads, the Python under ``src`` and ``scripts`` and the identity-roles table,
    and by its code as loaded (this module and ``source_identity``): any of them changed is a
    miss, and an unreadable input raises every time, never kept. A kept answer is derived from a
    written copy of exactly the bytes its key hashed, so no read of the derivation, the rule
    closure's own included, can see another tree; with no store named nothing is kept, and it is
    derived from the tree itself."""

    snapshot = _snapshot(root)

    def derived(_: bytes) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="closure-sites-") as folder:
            frozen = Path(folder)
            for relative, data in snapshot.items():
                if data is not None:
                    (frozen / relative).parent.mkdir(parents=True, exist_ok=True)
                    (frozen / relative).write_bytes(data)
            return [asdict(site) for site in _sites(frozen, snapshot)]

    with structural.kept_facts("closure-sites", sys.modules[__name__], source_identity) as facts:
        if facts.state.startswith("off"):  # nothing is kept: the tree itself, as before
            rows = [asdict(site) for site in _sites(root, snapshot)]
        else:
            rows = facts.of("closure-sites", _manifest(snapshot), derived, _sites_shape)
    return tuple(
        ClosureSite(
            **{
                **row,
                "tracked_paths": None
                if row["tracked_paths"] is None
                else tuple(row["tracked_paths"]),
            }
        )
        for row in rows
    )


def _snapshot(root: Path) -> dict[str, bytes | None]:
    """Every input's bytes: the Python under ``src`` and ``scripts``, then the roles table (None
    when absent), by checkout-relative path."""
    files = sorted(f for base in ("src", "scripts") for f in (root / base).rglob("*.py"))
    snapshot: dict[str, bytes | None] = {
        f.relative_to(root).as_posix(): f.read_bytes() for f in files
    }
    roles = root / IDENTITY_ROLES_PATH
    snapshot[IDENTITY_ROLES_PATH.as_posix()] = roles.read_bytes() if roles.is_file() else None
    return snapshot


def _manifest(snapshot: dict[str, bytes | None]) -> bytes:
    return json.dumps(
        [
            (path, "absent" if data is None else hashlib.sha256(data).hexdigest())
            for path, data in snapshot.items()
        ]
    ).encode()


def _sites_shape(value: object) -> bool:
    fields = {field.name for field in dataclass_fields(ClosureSite)}
    return isinstance(value, list) and all(
        isinstance(row, dict) and set(row) == fields for row in value
    )


def _sites(root: Path, snapshot: dict[str, bytes | None]) -> list[ClosureSite]:
    sites: list[ClosureSite] = []
    for relative, data in snapshot.items():
        if relative == _DEFINITIONS or data is None or not relative.endswith(".py"):
            continue
        text = structural.file_text(data)  # the bytes the key hashed, as read_text decodes them
        if not any(name in text for name in PRIMITIVES):
            continue
        sites.extend(_sites_in(relative, ast.parse(text)))
    return [_by_rule(root, site) for site in sites]


def _by_rule(root: Path, site: ClosureSite) -> ClosureSite:
    """A rule closure with its tracked paths: its entries and every module the rule adds."""

    if site.primitive != "source_rule_closure_hash" or site.tracked_paths is None:
        return site
    entries = tuple(
        ".".join(Path(path).with_suffix("").parts[1:])
        for path in site.tracked_paths
        if path.startswith("src/") and path.endswith(".py")
    )
    added = []
    for module in number_deciding_closure(entries, root=root, rule=number_deciding_rule(root)):
        base = "src/" + module.replace(".", "/")
        added.append(base + ".py" if (root / (base + ".py")).is_file() else base + "/__init__.py")
    return replace(site, tracked_paths=tuple(sorted({*site.tracked_paths, *added})))


def closures_tracking(sites: Iterable[ClosureSite], path: str) -> tuple[ClosureSite, ...]:
    """The resolved closures an edit to ``path`` rotates."""

    wanted = Path(path).as_posix()
    return tuple(site for site in sites if site.tracked_paths and wanted in site.tracked_paths)


def _sites_in(relative: str, tree: ast.Module) -> list[ClosureSite]:
    constants: dict[str, ast.expr] = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    constants.update(
        {
            node.target.id: node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        }
    )
    found: list[ClosureSite] = []

    def visit(node: ast.AST, scope: list[str], local: dict[str, ast.expr]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                inner = dict(local)
                if not isinstance(child, ast.ClassDef):
                    inner.update(
                        {
                            target.id: statement.value
                            for statement in ast.walk(child)
                            if isinstance(statement, ast.Assign)
                            for target in statement.targets
                            if isinstance(target, ast.Name)
                        }
                    )
                visit(child, [*scope, child.name], inner)
                continue
            if isinstance(child, ast.Call):
                primitive = _called(child.func)
                if primitive in PRIMITIVES:
                    keywords = {k.arg: k.value for k in child.keywords if k.arg}
                    tracked = keywords.get("tracked_paths")
                    names = {**constants, **local}
                    found.append(
                        ClosureSite(
                            path=relative,
                            line=child.lineno,
                            function=".".join(scope) or "<module>",
                            primitive=primitive,
                            semantic_owner=_text(keywords.get("semantic_owner"), names),
                            numerical_role=_text(keywords.get("numerical_role"), names),
                            tracked_paths=(
                                None
                                if primitive not in _LISTED or tracked is None
                                else _strings(tracked, names)
                            ),
                        )
                    )
            visit(child, scope, local)

    visit(tree, [], {})
    return found


def _called(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _text(node: ast.expr | None, names: dict[str, ast.expr], depth: int = 0) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in names and depth < 4:
        return _text(names[node.id], names, depth + 1)
    if isinstance(node, ast.IfExp):
        body, other = _text(node.body, names, depth), _text(node.orelse, names, depth)
        return None if body is None or other is None else f"{body} | {other}"
    return None


def _strings(node: ast.expr, names: dict[str, ast.expr], depth: int = 0) -> tuple[str, ...] | None:
    if depth > 6:
        return None
    if isinstance(node, ast.Tuple | ast.List):
        values: list[str] = []
        for element in node.elts:
            if isinstance(element, ast.Starred):
                spliced = _strings(element.value, names, depth + 1)
                if spliced is None:
                    return None
                values.extend(spliced)
            elif isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
            else:
                text = _text(element, names, depth + 1)
                if text is None:
                    return None
                values.append(text)
        return tuple(values)
    if isinstance(node, ast.Name) and node.id in names:
        return _strings(names[node.id], names, depth + 1)
    if isinstance(node, ast.Call) and _called(node.func) in {"tuple", "sorted"} and node.args:
        return _strings(node.args[0], names, depth + 1)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _strings(node.left, names, depth + 1), _strings(node.right, names, depth + 1)
        return None if left is None or right is None else (*left, *right)
    if isinstance(node, ast.IfExp):
        # A file in either branch rotates the closure under that branch: the map
        # answers for both.
        body, other = _strings(node.body, names, depth + 1), _strings(node.orelse, names, depth + 1)
        return None if body is None or other is None else tuple(dict.fromkeys((*body, *other)))
    if isinstance(node, ast.GeneratorExp | ast.ListComp) and len(node.generators) == 1:
        loop = node.generators[0]
        members = _strings(loop.iter, names, depth + 1)
        if members is None or loop.ifs or not isinstance(loop.target, ast.Name):
            return None
        return _mapped(node.elt, loop.target.id, members, names)
    return None


def _mapped(
    element: ast.expr, variable: str, members: tuple[str, ...], names: dict[str, ast.expr]
) -> tuple[str, ...] | None:
    if isinstance(element, ast.Name) and element.id == variable:
        return members
    if isinstance(element, ast.BinOp) and isinstance(element.op, ast.Add):
        prefix = _text(element.left, names)
        if (
            prefix is not None
            and isinstance(element.right, ast.Name)
            and element.right.id == variable
        ):
            return tuple(prefix + member for member in members)
    if isinstance(element, ast.JoinedStr) and len(element.values) == 2:
        head, tail = element.values
        if (
            isinstance(head, ast.Constant)
            and isinstance(head.value, str)
            and isinstance(tail, ast.FormattedValue)
            and isinstance(tail.value, ast.Name)
            and tail.value.id == variable
        ):
            return tuple(head.value + member for member in members)
    return None


def installed_family_value_identity(method_family: str) -> str:
    """One installed Factor method family's value identity, as the readout reads it (V360).

    The owner's own functions, composed: the family's installed owners and the measured closure
    of its values, before recorded moves hold the value a Panel binds (V345). A role per family
    names each family an edit to a shared owner moves, where the Host's roles alone showed none
    (V359's first cut, which U0 caught by four refused reads). It lives here, not beside those
    functions, because an edit to ``arithmetic_identity.py`` rotates thirteen Host roles and an
    edit to this file rotates none.
    """

    owners = installed_method_family_owners(method_family)
    return str(method_family_rule_identity(method_family, owners))


def installed_control_value_identity() -> str:
    """The controls' value identity, as the readout reads it: before recorded moves hold it (V345).

    Every installed core Factor binds it through the catalog's implementation identity, so an
    edit to a module the shared materializer imports (the availability catalog's among them)
    names it here.
    """

    return str(control_arithmetic_rule_identity())


def inventory(root: Path) -> dict[str, object]:
    """The whole inventory, as data: sites, their classed files, and the file map."""

    sites = closure_sites(root)
    by_file: dict[str, list[str]] = {}
    for site in sites:
        for tracked in site.tracked_paths or ():
            by_file.setdefault(tracked, []).append(site.name)
    return {
        "sites": [
            {
                **asdict(site),
                "classes": {tracked: path_class(tracked) for tracked in site.tracked_paths or ()},
            }
            for site in sites
        ],
        "files": {path: sorted(names) for path, names in sorted(by_file.items())},
    }


def report(root: Path, files: Sequence[str] = ()) -> list[str]:
    """Readable lines: for the named files, what they rotate; else the inventory's summary."""

    sites = closure_sites(root)
    if files:
        lines = []
        for file in files:
            tracking = closures_tracking(sites, file)
            if tracking:
                lines.append(
                    f"{file} rotates {len(tracking)}: " + ", ".join(s.name for s in tracking)
                )
        return lines
    resolved = [s for s in sites if s.tracked_paths is not None]
    runtime = [s for s in sites if s.tracked_paths is None]
    lines = [
        f"{len(sites)} source-identity sites: {len(resolved)} with literal tracked paths, "
        f"{len(runtime)} resolved at runtime (listed, not expanded)"
    ]
    for site in resolved:
        platform = sorted(
            {path_class(t) for t in site.tracked_paths or () if path_class(t) != "DOMAIN"}
        )
        tracked_count = len(site.tracked_paths or ())
        lines.append(
            f"  {site.name}: {tracked_count} files"
            + (f"; platform files: {', '.join(platform)}" if platform else "")
        )
    for site in runtime:
        lines.append(f"  {site.name}: {site.primitive}, modules resolved at runtime")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--file", action="append", default=[], help="A repository path.")
    parser.add_argument("--json", type=Path, help="Write the whole inventory as JSON.")
    args = parser.parse_args(argv)
    if args.json is not None:
        args.json.write_text(json.dumps(inventory(args.root), indent=1), encoding="utf-8")
    for line in report(args.root, args.file):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
