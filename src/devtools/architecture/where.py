"""``devtools where <noun>``: every registry's answer for one noun, before anyone writes code.

LAWS.md "look it up before writing it": where a noun is owned (the package registry and G1's owner
map, with what other packages still define of it and the card that moves it, and the map's note on
what stays outside its owner and why), which operations
name it, which identity roles, and what the parameter, refusal and format registries hold for it.
It reads the registries only; nothing is imported from the product but the operation registry.

    python -m devtools.architecture.where workspace
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from devtools.architecture.registries import baseline

_SHOWN = 8


def where(root: Path, noun: str) -> list[str]:
    word = noun.lower().replace("-", "_")
    lines: list[str] = []
    packages = json.loads((root / "config" / "package-architecture.json").read_text("utf-8"))
    for package in packages["packages"]:
        text = f"{package['package_id']} {package['capability_owner']}".lower().replace(" ", "_")
        if word in text:
            lines.append(
                f"package {package['package_id']}: {package['capability_owner']} "
                f"({package['role']}, {package['physical_namespace_target']})"
            )
    # G1's owner map: the decided owner, and what other packages still define until a card moves it.
    owners = json.loads((root / "config" / "registries" / "owners-map.json").read_text("utf-8"))
    for name, row in owners["nouns"].items():
        if word in name.lower().replace(" ", "_").replace("'", ""):
            lines.append(f"owner of {name}: {row['owner']} ({row['covers']})")
            lines.extend(
                f"  still defined in {item['from']}: {item['what']} ({item['card']})"
                for item in row["folds_in"]
            )
            # What the map says of the noun beyond its owner: what stays elsewhere, and why.
            if row.get("note"):
                lines.append(f"  note: {row['note']}")
    sys.path.insert(0, str(root / "src"))
    try:
        from alphalattice.interface.local_application.operations import COMMANDS
    finally:
        sys.path.pop(0)
    for (command_noun, verb), operations in sorted(COMMANDS.items()):
        for operation in operations:
            if word in command_noun.replace("-", "_") or word in operation.lower():
                lines.append(f"operation {operation}: alphalattice {command_noun} {verb}")
    roles = json.loads((root / "config" / "identity-roles.json").read_text("utf-8"))
    for role in roles["roles"]:
        if word in role["role"].lower():
            lines.append(f"identity role {role['role']}: {role['installed']}")
    for name in ("parameters", "refusals", "formats"):
        hits = sorted(key for key in baseline(root, name) if word in key.lower())
        for key in hits[:_SHOWN]:
            lines.append(f"{name} {key}")
        if len(hits) > _SHOWN:
            lines.append(f"{name}: {len(hits) - _SHOWN} more")
    return lines or [f"nothing registered names {noun}"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m devtools.architecture.where", description=__doc__.splitlines()[0]
    )
    parser.add_argument("noun")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args(argv)
    for line in where(args.root, args.noun):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
