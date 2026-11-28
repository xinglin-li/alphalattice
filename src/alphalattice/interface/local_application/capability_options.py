"""A capability's command options, read from its input schema.

Each string, integer, number or boolean property, or array of them, is an option of its own
(`--close-raw 1 2`, `--flag` or `--no-flag`); any other property goes in the command's `--params`
JSON file, which the help names. Standard library only: the CLI builds this parser after the
Host's listing names the schema.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from typing import Any, Final

_TYPES: Final[dict[str, Callable[[str], object]]] = {"string": str, "integer": int, "number": float}


def _boolean(text: str) -> bool:
    if text.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("true or false")
    return text.lower() == "true"


def capability_parser(
    name: str,
    schema: Mapping[str, Any],
    kind: type[argparse.ArgumentParser] = argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """The parser of `plugin run <name>`'s schema options, of the caller's parser `kind` (the
    CLI's refuses in its own envelope); a required property is marked."""
    required, others = set(schema.get("required", ())), []
    parser = kind(
        prog=f"alphalattice plugin run {name}",
        description=f"Runs {name}; its options come from its input schema.",
    )
    for field, spec in schema.get("properties", {}).items():
        shape = spec.get("type")
        item = spec.get("items", {}).get("type") if shape == "array" else None
        words = " ".join(
            part
            for part in (spec.get("description", ""), "(required)" if field in required else "")
            if part
        )
        flag = "--" + field.replace("_", "-")
        if shape == "boolean":
            parser.add_argument(flag, dest=field, action=argparse.BooleanOptionalAction, help=words)
        elif shape in _TYPES or item in _TYPES or item == "boolean":
            parser.add_argument(
                flag,
                dest=field,
                type=_boolean if item == "boolean" else _TYPES[item or shape],
                nargs="+" if item else None,
                choices=spec.get("enum"),
                metavar=(item or shape).upper(),
                help=words,
            )
        else:
            others.append(field)
    if others:
        parser.epilog = "Through --params: " + ", ".join(others) + "."
    return parser
