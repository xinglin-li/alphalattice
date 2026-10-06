"""Read all declared identities from the interpreter's runtime, without adding a checkout path."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    from alphalattice.kernel.shared_kernel import project_layout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = project_layout.resolve_playpen_root(Path(project_layout.__file__))
    table = json.loads((root / "config/identity-roles.json").read_text(encoding="utf-8"))
    result = {}
    for entry in table["roles"]:
        module, _, member = entry["installed"].partition(":")
        if module == "devtools.architecture.identity_closures":
            from alphalattice.foundation.feature_engine.producers import arithmetic_identity

            if member == "installed_family_value_identity":
                family = entry["arguments"]["method_family"]
                value = arithmetic_identity.method_family_rule_identity(
                    family, arithmetic_identity.installed_method_family_owners(family)
                )
            elif member == "installed_control_value_identity":
                value = arithmetic_identity.control_arithmetic_rule_identity()
            else:
                raise ValueError(f"wheel.identity_reader_unresolved:{member}")
            result[entry["role"]] = str(value)
            continue
        target: Any = importlib.import_module(module)
        for part in member.split("."):
            target = getattr(target, part)
        arguments = {
            key: root if value == "<root>" else value for key, value in entry["arguments"].items()
        }
        value = (
            target(arguments.pop("self"), **arguments)
            if "self" in arguments
            else target(**arguments)
        )
        for part in filter(None, entry.get("result", "").split(".")):
            value = getattr(value, part)
        result[entry["role"]] = str(value)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(result)} runtime identities read")


if __name__ == "__main__":
    main()
