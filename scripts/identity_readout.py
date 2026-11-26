"""Every Host implementation identity as this tree installs it, and what a change moved.

The readout of the identity-roles table (`config/identity-roles.json`, binding plan R1): each
role's installed value, computed by the product's own function, never by a copy of its closure.
A change that moves an identity is read before and after; the moved roles are the ones it must
record (LAWS.md ID1, ID2), and `--record` writes their successors when the role follows recorded
moves. Pure hashing: nothing opens a workspace.

Which roles an edit to each area may move is the reach table
(`config/registries/identity-reach.json`, LAWS.md ID8, GB): measured by these same functions, with
the rule closure's walk watched, and only shrinking. `--reach-check` refuses an area that reaches a
role outside its row, naming the import that did it; `--reach-shrink` records a smaller reach.

    python scripts/identity_readout.py --json OUT
    python scripts/identity_readout.py --compare BEFORE.json AFTER.json
    python scripts/identity_readout.py --compare BEFORE.json AFTER.json --record CHANGE REASON
    python scripts/identity_readout.py --reach-check | --reach-shrink | --reach-record
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

TABLE = ROOT / "config" / "identity-roles.json"
SUCCESSORS = ROOT / "config" / "identity-successors.json"
REACH = ROOT / "config" / "registries" / "identity-reach.json"


def roles(table: Path = TABLE) -> list[dict[str, Any]]:
    """The roles of the table, in its order."""

    return list(json.loads(table.read_text(encoding="utf-8"))["roles"])


def installed(entry: dict[str, Any]) -> str:
    """The value this tree installs for one role, by the owner's own function."""

    module_name, _, qualname = str(entry["installed"]).partition(":")
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    arguments = {
        key: (ROOT if value == "<root>" else value) for key, value in entry["arguments"].items()
    }
    if "self" in arguments:
        value = target(arguments.pop("self"), **arguments)
    else:
        value = target(**arguments)
    for part in filter(None, str(entry.get("result", "")).split(".")):
        value = getattr(value, part)  # a builder's binding: its hash
    return str(value)


def readout(table: Path = TABLE) -> dict[str, str]:
    return {str(entry["role"]): installed(entry) for entry in roles(table)}


def moved(before: dict[str, str], after: dict[str, str]) -> dict[str, tuple[str, str]]:
    """The roles whose installed value differs, with the two values."""

    return {
        role: (before[role], after[role])
        for role in sorted(set(before) & set(after))
        if before[role] != after[role]
    }


def record(changes: dict[str, tuple[str, str]], change: str, reason: str) -> list[str]:
    """Append a successor for each moved role that follows recorded moves; refuse the rest."""

    comparison = {str(entry["role"]): str(entry["comparison"]) for entry in roles()}
    exact = sorted(role for role in changes if comparison.get(role) != "IS_CURRENT")
    if exact:
        raise SystemExit(
            "identity_readout: these roles compare exactly, so a move makes their current "
            f"plans historical; give them a role that follows recorded moves first: {exact}"
        )
    document = json.loads(SUCCESSORS.read_text(encoding="utf-8"))
    for role, (predecessor, successor) in changes.items():
        document["moves"].append(
            {
                "role": role,
                "predecessor": predecessor,
                "successor": successor,
                "change": change,
                "reason": reason,
            }
        )
    SUCCESSORS.write_bytes((json.dumps(document, indent=1, ensure_ascii=False) + "\n").encode())
    return sorted(changes)


def reach(table: Path = TABLE) -> tuple[dict[str, set[str]], dict[tuple[str, str], str]]:
    """Which roles an edit to each area may move, and how each area is reached (GB, ID8).

    Each role's installed value is computed with the rule closure's walk watched: every module
    the walk reaches is one whose syntax the role hashes. A module's area is its package.
    """

    from alphalattice.kernel.shared_kernel import source_identity

    walk = source_identity.number_deciding_closure
    reached: list[dict[str, str]] = []

    def watched(
        entries: tuple[str, ...], *, root: Path, rule: Any, excluded: frozenset[str] = frozenset()
    ) -> dict[str, str]:
        found = walk(entries, root=root, rule=rule, excluded=excluded)
        reached.append(found)
        return found

    areas: dict[str, set[str]] = {}
    edges: dict[tuple[str, str], str] = {}
    source_identity.number_deciding_closure = watched  # looked up at call time
    try:
        for entry in roles(table):
            reached.clear()
            installed(entry)
            role = str(entry["role"])
            modules = {module for found in reached for module in found}
            for module in sorted(modules):
                area = _area(module)
                areas.setdefault(area, set()).add(role)
                edges.setdefault((area, role), _how_reached(module, modules))
    finally:
        source_identity.number_deciding_closure = walk
    return areas, edges


def _file_of(module: str) -> Path | None:
    base = ROOT.joinpath("src", *module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _area(module: str) -> str:
    """A module's package: the package itself for its `__init__`."""

    file = _file_of(module)
    return module if file is not None and file.name == "__init__.py" else module.rpartition(".")[0]


def _how_reached(module: str, modules: set[str]) -> str:
    """The first module of the closure that imports this one, or that it is an entry."""

    from alphalattice.kernel.shared_kernel.source_identity import source_module_imports

    for importer in sorted(modules - {module}):
        file = _file_of(importer)
        if file is None:
            continue
        package = importer if file.name == "__init__.py" else importer.rpartition(".")[0]
        if any(
            name == module or name.startswith(module + ".")
            for name in source_module_imports(file, package=package)
        ):
            return f"{importer} imports {module}"
    return f"{module} is an entry of the role"


def recorded_reach(path: Path = REACH) -> dict[str, list[str]]:
    """The reach table as recorded."""

    return dict(json.loads(path.read_text(encoding="utf-8"))["areas"])


def write_reach(areas: dict[str, set[str]], path: Path = REACH) -> None:
    document = {
        "schema": "identity-reach",
        "version": 1,
        "note": (
            "Per area (a package), the identity roles an edit to it may move, measured by "
            "scripts/identity_readout.py --reach-record. It only shrinks (LAWS.md ID8): a "
            "change reaching further is refused, naming the import that did it."
        ),
        "areas": {area: sorted(roles_) for area, roles_ in sorted(areas.items())},
    }
    path.write_bytes((json.dumps(document, indent=1) + "\n").encode("utf-8"))


def reach_growth(
    areas: dict[str, set[str]], edges: dict[tuple[str, str], str], recorded: dict[str, list[str]]
) -> list[str]:
    """Each area that reaches a role outside its recorded row, with how."""

    return [
        f"{area} now reaches {role}: {edges[(area, role)]}"
        for area, reached_roles in sorted(areas.items())
        for role in sorted(reached_roles - set(recorded.get(area, ())))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write this tree's readout")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--record", nargs=2, metavar=("CHANGE", "REASON"))
    reach_mode = parser.add_mutually_exclusive_group()
    reach_mode.add_argument("--reach-check", action="store_true")
    reach_mode.add_argument("--reach-shrink", action="store_true")
    reach_mode.add_argument("--reach-record", action="store_true")
    args = parser.parse_args(argv)
    if args.reach_check or args.reach_shrink or args.reach_record:
        areas, edges = reach()
        if args.reach_record:
            write_reach(areas)
            print(f"identity reach recorded: {len(areas)} areas")
            return 0
        grown = reach_growth(areas, edges, recorded_reach())
        if grown:
            print("identity reach grew (LAWS.md ID8):\n  " + "\n  ".join(grown))
            return 1
        if args.reach_shrink:
            write_reach(areas)
        print(f"identity reach holds: {len(areas)} areas")
        return 0
    if args.compare:
        before, after = (json.loads(path.read_text(encoding="utf-8")) for path in args.compare)
        changes = moved(before, after)
        for role, (old, new) in changes.items():
            print(f"{role}: {old[:8]} -> {new[:8]}")
        if not changes:
            print("no identity moved")
        if args.record and changes:
            print("recorded:", ", ".join(record(changes, *args.record)))
        return 0
    values = readout()
    for role, value in values.items():
        print(f"{value[:8]}  {role}")
    if args.json:
        args.json.write_text(json.dumps(values, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
