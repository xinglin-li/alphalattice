"""The operation registry agrees with every place an operation is named (C1 rule 3), and the CLI
answers every one of them in its contract without a Host (the CLI walk).

An operation is named in the request's literal, its field contract and its two field lists,
the activity ledger's sets, the Host's routes, the research-case sets, the Task-work sets, and
in every next request an owner offers. Registering one by hand in eight places was how an
operation reached the Web and not the CLI, or was offered as a next request the CLI could not
send. This check reads them all and refuses a tree where they disagree; the walk runs every
``alphalattice <noun> <verb>`` against a workspace no Host serves and with a required field
missing, and checks the envelope, the exit code, and that no answer carries a traceback. Every
command a product text names in backticks, and each flag it names, must be the CLI's (V124).

    python -m devtools.architecture.operation_registry [--write]

``--write`` first writes the CLI's command table, ``operations.json``, from the registry.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Final

from devtools.architecture.neutral_changes import source_syntax_tree

_OFFERED = re.compile(r'"operation":\s*"([A-Z][A-Z0-9_]*[A-Z0-9])"')
_ENVELOPE = {
    "schema_version",
    "operation",
    "outcome",
    "status",
    "data",
    "failure_code",
    "detail",
    "next_requests",
    "timing",
}


def _route_operations(root: Path) -> set[str]:
    """Operations the Host routes by name: every ``(method, path, OPERATION)`` row."""

    source = root / "src/alphalattice/control/product_host/composition/local_web_session.py"
    tree = source_syntax_tree(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 3:
            values = [e.value if isinstance(e, ast.Constant) else None for e in node.elts]
            if values[0] in {"GET", "POST"} and isinstance(values[2], str):
                found.add(values[2])
    return found


def _goal_set_operations(root: Path) -> dict[str, set[str]]:
    """The goal owner's sets (``*_OPERATIONS``), read from source: the Host is not imported."""

    source = root / "src/alphalattice/control/product_host/composition/goals.py"
    tree = source_syntax_tree(source)
    found: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            for name in names:
                if name.endswith("_OPERATIONS") and node.value is not None:
                    # A mapping is keyed by operation; its values are something else.
                    scope = (
                        [k for k in node.value.keys if k is not None]
                        if isinstance(node.value, ast.Dict)
                        else [node.value]
                    )
                    found[name] = {
                        c.value
                        for part in scope
                        for c in ast.walk(part)
                        if isinstance(c, ast.Constant) and isinstance(c.value, str)
                    }
    return found


def _offered(root: Path) -> dict[str, set[str]]:
    """Every operation an owner offers as a next request, by the file that offers it."""

    offered: dict[str, set[str]] = {}
    for path in (root / "src/alphalattice").rglob("*.py"):
        for name in _OFFERED.findall(path.read_text(encoding="utf-8")):
            offered.setdefault(name, set()).add(path.relative_to(root).as_posix())
    return offered


_NAMING_TEXTS = (
    "src/alphalattice",
    ".agents/skills",
    ".claude/skills",
    ".claude/agents",
    ".codex/agents",
    "case-study",
    "docs",
    "README.md",
)
"""The texts a person or an agent reads for how to run the product; the plans are history."""

_NAMED = re.compile(
    r"`(alphalattice |python scripts/run_alphalattice\.py )?(?:--[a-z][a-z-]* \S+ )*"
    r"([a-z][a-z-]*)(?: ([a-z][a-z-]*))?((?: [^`\n]*)?)`"
)


def _cli_options() -> dict[tuple[str, str], set[str]]:
    """Each ``(noun, verb)`` the CLI parses, ``verb`` empty for a noun alone, with its options."""

    import argparse

    from alphalattice.interface.local_application import cli

    def children(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
        found = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        return dict(found[0].choices) if found else {}

    def own(parser: argparse.ArgumentParser) -> set[str]:
        return {option for action in parser._actions for option in action.option_strings}

    top = cli._parser()
    shared = own(top)
    options: dict[tuple[str, str], set[str]] = {}
    for noun, noun_parser in children(top).items():
        options[(noun, "")] = shared | own(noun_parser)
        for verb, verb_parser in children(noun_parser).items():
            options[(noun, verb)] = shared | own(verb_parser)
    return options


def named_commands(root: Path) -> list[str]:
    """Each command or flag a product text names in backticks that the CLI does not have (V124).

    A phrase is a command when it starts with a noun the CLI has, with the root command's
    prefix, or with a word and then a flag: refusal words named `capabilities --operation`
    after the command had gone, and a noun the CLI no longer has is exactly that case.

    Args:
        root: The checkout.

    Returns:
        One line per unknown command or flag, naming the text that names it.
    """

    options = _cli_options()
    nouns = {noun for noun, _verb in options}
    out: list[str] = []
    for base in _NAMING_TEXTS:
        start = root / base
        paths = [start] if start.is_file() else sorted(start.rglob("*"))
        for path in paths:
            if path.suffix not in {".py", ".md", ".toml"} or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "`" not in text:
                continue
            relative = path.relative_to(root).as_posix()
            for match in _NAMED.finditer(text):
                prefix, noun, verb, rest = match.groups()
                flags = [token.split("=", 1)[0] for token in rest.split() if token.startswith("--")]
                if noun not in nouns:
                    if prefix or (verb is None and rest.lstrip().startswith("--")):
                        out.append(f"{relative} names `{noun}`, a command the CLI does not have")
                    continue
                command = (noun, verb or "")
                if command not in options:
                    out.append(f"{relative} names `{noun} {verb}`, a command the CLI does not have")
                    continue
                for flag in flags:
                    if flag not in options[command]:
                        named = f"{noun} {verb} {flag}" if verb else f"{noun} {flag}"
                        out.append(f"{relative} names `{named}`, a flag the CLI does not have")
    return out


def incomplete_offers(root: Path) -> list[str]:
    """Each next request an owner offers without a field its operation requires (V136).

    A field the reader must choose is written as None, so the CLI shows the request as a
    template naming it; an absent one is a mistake. A request built with ``**`` takes fields
    this check cannot see, and is left to the owner.

    Args:
        root: The checkout.

    Returns:
        One line per offered request that neither fills nor names a required field.
    """

    from alphalattice.interface.local_application.operations import OPERATIONS, fields

    out: list[str] = []
    for path in sorted((root / "src/alphalattice").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if '"operation"' not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Dict) or None in node.keys:
                continue
            written = {
                key.value: value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            value = written.get("operation")
            operation = value.value if isinstance(value, ast.Constant) else None
            if not isinstance(operation, str) or operation not in OPERATIONS:
                continue
            for name in sorted(fields(operation)[0] - set(written)):
                relative = path.relative_to(root).as_posix()
                out.append(
                    f"{relative}:{node.lineno} offers {operation} without its required "
                    f"{name}; write it, or None for the reader's choice"
                )
    return out


_TEXT_AS_CODE_ROOTS = (
    "src/alphalattice/control/product_host",
    "src/alphalattice/interface/local_application",
)


def text_served_as_code(root: Path) -> list[str]:
    """Each place an exception's text (``str(error)`` of an ``except ... as error``) is served
    as a failure code: a ``failure_code`` key or keyword, or the code of ``refused(...)``.
    The owner's code is ``public_failure(error, fallback)`` (V31); a comparison of the text
    with known codes is not serving it."""

    found: list[str] = []
    for base in _TEXT_AS_CODE_ROOTS:
        for path in sorted((root / base).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            caught = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ExceptHandler) and node.name
            }

            def text_of(value: ast.expr | None, caught: set[str] = caught) -> bool:
                if isinstance(value, ast.Subscript):
                    value = value.value
                return (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "str"
                    and len(value.args) == 1
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id in caught
                )

            for node in ast.walk(tree):
                served: list[ast.expr | None] = []
                if isinstance(node, ast.Dict):
                    served = [
                        value
                        for key, value in zip(node.keys, node.values, strict=True)
                        if isinstance(key, ast.Constant) and key.value == "failure_code"
                    ]
                elif isinstance(node, ast.Call):
                    served = [k.value for k in node.keywords if k.arg == "failure_code"]
                    if isinstance(node.func, ast.Name) and node.func.id == "refused" and node.args:
                        served.append(node.args[0])
                for value in served:
                    if value is not None and text_of(value):
                        relative = path.relative_to(root).as_posix()
                        found.append(
                            f"{relative}:{value.lineno} serves an exception's text as a failure "
                            "code; answer public_failure(error, fallback)"
                        )
    return found


def problems(root: Path) -> list[str]:
    """What disagrees with the registry at ``root``; empty when every name agrees."""

    from alphalattice.interface.local_application import activity
    from alphalattice.interface.local_application.cli import CLIENT_COMMANDS
    from alphalattice.interface.local_application.operations import (
        ALTERNATIVES,
        COMMANDS,
        COMMON_PATH,
        HELP_GROUPS,
        OPERATIONS,
        flag_name,
        table_text,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
        PortfolioResearchRequestDocument,
    )

    known = set(OPERATIONS)
    out: list[str] = []
    if sorted(op for ops in COMMANDS.values() for op in ops) != sorted(OPERATIONS):
        out.append("an operation has no CLI command, or two")
    for (noun, verb), ops in sorted(COMMANDS.items()):
        selectors = [PortfolioResearchOperationRequest.field_contract(op)[0] for op in ops]  # type: ignore[arg-type]
        if len(ops) > 1 and any(
            a <= b for i, a in enumerate(selectors) for b in selectors[i + 1 :]
        ):
            out.append(f"{noun} {verb}: its operations' required selectors do not tell them apart")
        for op in ops:
            named = [flag_name(f) for f in PortfolioResearchOperationRequest.field_contract(op)[1]]  # type: ignore[arg-type]
            flags = [n for n in named if n != "file"]
            if len(flags) != len(set(flags)):
                out.append(f"{noun} {verb} ({op}): two fields share one flag")
    # An exactly-one rule names only the request's own fields (V409).
    for operation, groups in sorted(ALTERNATIVES.items()):
        allowed = PortfolioResearchOperationRequest.field_contract(operation)[1]  # type: ignore[arg-type]
        for name in sorted({name for group in groups for name in group} - set(allowed)):
            out.append(f"{operation}'s alternatives name {name}, which it does not take")
    # The help lists each object the CLI has once, and its common path only commands (V401).
    grouped = sorted(noun for _group, nouns in HELP_GROUPS for noun, _purpose in nouns)
    objects = {noun for noun, _verb in (*COMMANDS, *CLIENT_COMMANDS)} | {"serve", "request"}
    if grouped != sorted(objects):
        out.append(f"the help's groups list objects other than each once: {grouped}")
    for command in COMMON_PATH:
        noun, _space, verb = command.partition(" ")
        if (noun, verb) not in COMMANDS and (noun, verb) not in CLIENT_COMMANDS:
            out.append(f"the help's common path names {command}, which the CLI does not have")
    for operation in OPERATIONS:
        try:
            PortfolioResearchOperationRequest.field_contract(operation)  # type: ignore[arg-type]
        except KeyError:
            out.append(f"{operation} has no field contract")
    dataclass_names = {f.name for f in dataclass_fields(PortfolioResearchOperationRequest)}
    document_names = set(PortfolioResearchRequestDocument.model_fields)
    for name in sorted(dataclass_names ^ document_names):
        out.append(f"request field {name} is in one field list and not the other")

    def unknown(label: str, names: Iterable[object]) -> None:
        for name in sorted({str(n) for n in names} - known):
            out.append(f"{label} names {name}, which is not an operation")

    unknown("the Host's routes", _route_operations(root))
    for label in ("READ_OPERATIONS", "OBSERVED_OPERATIONS"):
        unknown(f"activity.{label}", getattr(activity, label))
    for label, value in _goal_set_operations(root).items():
        unknown(f"goals.{label}", value)
    for name, files in sorted(_offered(root).items()):
        if name not in known:
            out.append(f"{', '.join(sorted(files))} offers {name}, which is not an operation")
    out.extend(text_served_as_code(root))
    out.extend(named_commands(root))
    out.extend(incomplete_offers(root))
    table = root / "src/alphalattice/interface/local_application/operations.json"
    if not table.is_file() or table.read_text(encoding="utf-8") != table_text():
        out.append(
            "operations.json is not the registry's table (the CLI reads it): "
            "python -m devtools.architecture.operation_registry --write"
        )
    return out


def _client_required(noun: str, verb: str) -> list[str]:
    """The required options of one of the client's own commands, as its parser declares them."""

    import argparse

    from alphalattice.interface.local_application import cli

    def child(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
        found = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        chosen: argparse.ArgumentParser = found.choices[name]
        return chosen

    command = child(child(cli._parser(), noun), verb)
    required = [a.option_strings[0] for a in command._actions if a.required and a.option_strings]
    # A required choice of one option among several: its first option stands for it.
    required += [
        group._group_actions[0].option_strings[0]
        for group in command._mutually_exclusive_groups
        if group.required
    ]
    return sorted(required)


_OFFLINE_CASES: Final[dict[tuple[str, str], list[tuple[str, list[str], int, str]]]] = {
    ("answer", "show"): [
        ("no saved file", ["--file", "absent.answer.json"], 1, "INVALID_INPUT"),
        ("a missing field", [], 1, "INVALID_INPUT"),
    ],
    # It answers from the registry alone: a real operation, and no Host.
    ("schema", "show"): [
        ("no Host", ["STATUS"], 0, "OK"),
        ("a missing field", [], 1, "INVALID_INPUT"),
    ],
    # It restores from the backup root alone (V328), a restore the entry composes as it does
    # `serve`; nothing under src imports the Host, so the walk composes neither and the client
    # refuses by name (the backup tests run the real restore).
    ("backup", "restore"): [
        ("no Host", ["--dir", "x"], 2, "REFUSED"),
        ("a missing field", [], 1, "INVALID_INPUT"),
    ],
    # A model's files are the checkout's (EX): the walk names none, and each refuses by name.
    ("model", "scaffold"): [
        ("no declaration", ["--file", "absent.model.yaml"], 1, "INVALID_INPUT"),
        ("a missing field", [], 1, "INVALID_INPUT"),
    ],
    ("model", "check"): [
        ("no such model", ["absent_model"], 1, "INVALID_INPUT"),
        ("a missing field", [], 1, "INVALID_INPUT"),
    ],
    # The sandbox is the entry's, as a restore is: the walk composes none and the client refuses.
    ("model", "sandbox"): [
        ("no sandbox composed", ["absent_model"], 2, "REFUSED"),
        ("a missing field", [], 1, "INVALID_INPUT"),
    ],
    # The walk runs as a shell outside any agent session (`walk`): there is no session to bind,
    # so the client refuses by name and writes nothing (V568).
    ("session", "bind"): [
        ("no agent session", [], 1, "INVALID_INPUT"),
    ],
    # The person's own shell, in the walk's empty folder: no project up from it holds the
    # product's declarations, so the client refuses by name and removes nothing (V586).
    ("session", "unbind"): [
        ("no project", [], 2, "REFUSED"),
    ],
}
"""How each of the client's offline commands answers the walk's empty workspace."""


def walk(root: Path) -> list[str]:
    """Every command against a workspace no Host serves, and with a required field missing.

    It runs as a shell outside any agent session, whichever session runs the gate: a session
    would bind (`session bind`) and print its bound workspace's commands clean (V568). It runs
    in its empty folder, never the checkout, whose binding the person's `session unbind` would
    remove (V586).
    """

    from alphalattice.interface.local_application import cli
    from alphalattice.interface.local_application.cli_contract import AGENT_SESSION_VARIABLES

    with contextlib.ExitStack() as outside:
        for _vendor, variable in AGENT_SESSION_VARIABLES:
            if variable in os.environ:
                value = os.environ.pop(variable)
                outside.callback(os.environ.__setitem__, variable, value)
        return _walk(root, cli)


def _walk(root: Path, cli: Any) -> list[str]:
    from alphalattice.interface.local_application.operations import (
        COMMANDS,
        GRAMMAR,
        fields,
        flag_name,
    )

    def required_arguments(op: str) -> list[str]:
        """The operation's required fields as its command takes them; the positional last."""
        named = sorted(fields(op)[0])
        shown = GRAMMAR[op].positional
        flags = dict.fromkeys(f"--{flag_name(n)}" for n in named if n != shown)
        return [*(a for f in flags for a in (f, "x")), *(["x"] if shown in named else [])]

    out: list[str] = []
    commands: list[tuple[str, str, list[str]]] = [
        ("request", "", ["--file", "x"]),
        *(
            (noun, verb, [a for option in _client_required(noun, verb) for a in (option, "x")])
            for noun, verb in sorted(cli.CLIENT_COMMANDS)
        ),
        *(
            (noun, verb, required_arguments(op))
            for (noun, verb), ops in sorted(COMMANDS.items())
            for op in ops
        ),
    ]
    with tempfile.TemporaryDirectory() as empty, contextlib.chdir(empty):
        for noun, verb, given in commands:
            cases = [("no Host", given, 4, "NO_HOST")]
            if given:
                cases.append(("a missing field", given[2:], 1, "INVALID_INPUT"))
            if noun == "request":
                cases = cases[:1]  # its --file is optional beside --from
            if (noun, verb) in cli.OFFLINE_COMMANDS:
                cases = _OFFLINE_CASES[(noun, verb)]
            for label, arguments, code, outcome in cases:
                printed = io.StringIO()
                with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
                    try:
                        exit_code = cli.main(
                            ["--workspace", empty, noun, *([verb] if verb else []), *arguments],
                            serve=lambda _: 99,
                        )
                    except SystemExit as stop:
                        exit_code = int(stop.code or 0)
                text = printed.getvalue()
                lines = text.strip().splitlines()
                answer = json.loads(lines[-1]) if lines else {}
                if exit_code != code or answer.get("outcome") != outcome:
                    out.append(
                        f"{noun} {verb} ({label}): exit {exit_code}, {answer.get('outcome')}; "
                        f"expected {code}, {outcome}"
                    )
                elif not answer.keys() >= _ENVELOPE:
                    out.append(
                        f"{noun} {verb} ({label}): the envelope lacks {_ENVELOPE - answer.keys()}"
                    )
                elif "Traceback" in text or empty in text:
                    out.append(f"{noun} {verb} ({label}): the answer carries a traceback or a path")
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    if sys.argv[1:] == ["--write"]:
        from alphalattice.interface.local_application.operations import TABLE, table_text

        TABLE.write_bytes(table_text().encode("utf-8"))
    found = problems(root) + walk(root)
    for line in found:
        print(line)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
