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


UI_ONLY_ACTIONS: Final[dict[str, str]] = {}
"""Named action/request exceptions, with reasons; an addition needs an agent route instead."""


def _decorated_routes(node: ast.FunctionDef) -> list[tuple[str, str, bool]]:
    """Literal Host routes with their external-client admission."""
    found: list[tuple[str, str, bool]] = []
    for decorator in node.decorator_list:
        match decorator:
            case ast.Call(
                func=ast.Attribute(attr="route"),
                args=[
                    ast.Constant(value=("GET" | "POST") as method),
                    ast.Constant(value=str(path)),
                    *_,
                ],
                keywords=keywords,
            ):
                external = any(
                    key.arg == "external_client"
                    and isinstance(key.value, ast.Constant)
                    and key.value.value is True
                    for key in keywords
                )
                found.append((method, path, external))
    return found


def _literal_requests(node: ast.AST) -> set[str]:
    """Typed operation literals reached by a Host handler."""
    found = set()
    bindings = {
        ast.unparse(target): part.value
        for part in ast.walk(node)
        if isinstance(part, ast.Assign)
        for target in part.targets
    }
    for part in ast.walk(node):
        match part:
            case ast.Call(
                func=ast.Attribute(value=ast.Name(id="operations"), attr="execute"),
                args=[request, *_],
            ):
                if isinstance(request, ast.Name):
                    request = bindings.get(request.id, request)
                match request:
                    case ast.Call(
                        func=ast.Name(id="PortfolioResearchOperationRequest"), keywords=keywords
                    ):
                        for key in keywords:
                            match key:
                                case ast.keyword(
                                    arg="operation", value=ast.Constant(value=str(operation))
                                ):
                                    found.add(operation)
    return found


def _ui_host_routes(root: Path) -> tuple[dict[str, set[str]], set[str]]:
    tree = source_syntax_tree(
        root / "src/alphalattice/control/product_host/composition/local_web_session.py"
    )
    found: dict[str, set[str]] = {}
    external: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case ast.Tuple(
                elts=[
                    ast.Constant(value="GET" | "POST"),
                    ast.Constant(value=str(path)),
                    ast.Constant(value=str(operation)),
                ]
            ):
                found.setdefault(path, set()).add(operation)
            case ast.FunctionDef():
                operations = _literal_requests(node)
                decorated = _decorated_routes(node)
                external.update(path for _method, path, admitted in decorated if admitted)
                if operations:
                    for _method, path, _external in decorated:
                        found.setdefault(path, set()).update(operations)
    return found, external


def _ui_client_paths(root: Path) -> set[str]:
    tree = source_syntax_tree(root / "src/alphalattice/interface/local_application/client.py")
    found = set()
    client = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LocalResearchClient"
    )
    for node in client.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        for call in ast.walk(node):
            match call:
                case ast.Call(
                    func=ast.Attribute(value=ast.Name(id="self"), attr="_exchange"),
                    args=[ast.Constant(value=str(path)), _, *_],
                ):
                    found.add(path)
    return found


def _ui_requests(
    row: dict[str, Any], routes: dict[str, set[str]], client_paths: set[str]
) -> set[str]:
    operations, paths = set(row.get("operations", ())), set(row.get("paths", ()))
    offered = "@offered" in operations | paths
    requests = operations - {"@offered"}
    paths.discard("@offered")
    requests.update(op for path in paths for op in routes.get(path, ()))
    if offered:
        requests.update(op for values in routes.values() for op in values)
    requests.update(paths - routes.keys() - client_paths)
    requests.update(row.get("unbounded", ()))
    return requests


def _relay_forms(forms: dict[str, set[str]]) -> list[str]:
    from alphalattice.interface.local_application.operations import fields
    from alphalattice.interface.local_application.portfolio_research import PERSON_DECISIONS

    return [
        f"person relay => {operation}; add the CLI form or the relay"
        for operation in PERSON_DECISIONS
        if "person_confirmation" not in fields(operation)[1]
        or not {"--person-said", "--asked"} <= forms.get(operation, set())
    ]


def ui_agent_parity(
    root: Path,
    inventory: dict[str, Any],
    *,
    gaps: dict[str, str] | None = None,
    previous_gaps: dict[str, str] | None = None,
) -> list[str]:
    """Refuse a Workbench mutation without a parsed CLI form, client verb or person relay."""
    from alphalattice.interface.local_application.operations import COMMANDS

    options = _cli_options()
    forms = {op: options[form] for form, ops in COMMANDS.items() if form in options for op in ops}
    routes, external = _ui_host_routes(root)
    client_paths = _ui_client_paths(root) & external
    allowed = UI_ONLY_ACTIONS if gaps is None else gaps
    previous = {} if previous_gaps is None else previous_gaps
    out = [
        f"{key}: a new UI-only census row; add the CLI form or the relay"
        for key in allowed.keys() - previous.keys()
    ]
    out.extend(
        f"{action}: no Workbench handler; add the CLI form or the relay"
        for action in inventory.get("unhandled", ())
    )
    out.extend(_relay_forms(forms))
    unresolved: set[str] = set()
    for row in (*inventory.get("actions", ()), *inventory.get("posts", ())):
        for request in _ui_requests(row, routes, client_paths) - forms.keys():
            key = f"{row['action']} => {request}"
            unresolved.add(key)
            if not allowed.get(key):
                out.append(f"{row.get('source', '')}: {key}; add the CLI form or the relay")
    out.extend(
        f"{key}: its agent route exists; lower the UI-only census"
        for key in allowed.keys() - unresolved
    )
    return sorted(set(out))


def _relayed_authority(expression: ast.expr | None) -> bool:
    match expression:
        case ast.IfExp(
            body=ast.Constant(value="HUMAN"),
            orelse=ast.Name(id="caller"),
            test=ast.BoolOp(
                op=ast.And(),
                values=[
                    ast.Compare(
                        left=ast.Name(id=subject),
                        ops=[ast.IsNot()],
                        comparators=[ast.Constant(value=None)],
                    ),
                    ast.BoolOp(
                        op=ast.Or(),
                        values=[
                            ast.Attribute(value=ast.Name(id=left), attr=first),
                            ast.Attribute(value=ast.Name(id=right), attr=second),
                        ],
                    ),
                ],
            ),
        ):
            return subject == left == right and {first, second} == {"delegation", "relayed"}
    return False


def _relay_dispatch(observed: ast.FunctionDef | None) -> bool:
    if observed is None:
        return False
    assignments = {
        ast.unparse(target): node.value
        for node in ast.walk(observed)
        if isinstance(node, ast.Assign)
        for target in node.targets
    }
    dispatch = False
    for node in ast.walk(observed):
        match node:
            case ast.Call(func=ast.Attribute(attr="_execute_within_memory"), keywords=keywords):
                dispatch = any(
                    key.arg == "caller" and ast.unparse(key.value) == "authority"
                    for key in keywords
                )
                if not dispatch:
                    return False
    return dispatch and _relayed_authority(assignments.get("authority"))


def _relay_boundary_checks(nodes: list[ast.FunctionDef]) -> list[str]:
    out = []
    for node in nodes:
        for check in (
            part
            for part in ast.walk(node)
            if isinstance(part, ast.If | ast.IfExp | ast.Assert | ast.Call)
        ):
            text = ast.unparse(check.func if isinstance(check, ast.Call) else check.test)
            if ("HUMAN" in text and re.search(r"\bcaller\b", text)) or any(
                word in text
                for word in ("'UI'", "'BROWSER'", "browser_session", "ui_origin", "'WORKBENCH'")
            ):
                out.append(f"{node.name}:{check.lineno}: raw caller refuses the person relay")
    return out


def _provenance_aliases(tree: ast.Module) -> set[str]:
    """Direct raw-context aliases, distinct from an owner's computed actor or authority."""
    assignments = [
        (target, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
    ]
    raw: set[str] = set()
    while True:
        expanded = raw | {
            ast.unparse(target)
            for target, value in assignments
            if ast.unparse(value).startswith("REQUEST_PROVENANCE.get()")
            or (
                isinstance(value, ast.Name | ast.Attribute | ast.Subscript)
                and any(ast.unparse(part) in raw for part in ast.walk(value))
            )
        }
        if expanded == raw:
            return raw
        raw = expanded


def _raw_provenance_checks(name: str, tree: ast.Module) -> list[str]:
    raw = _provenance_aliases(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If | ast.IfExp | ast.Assert | ast.Compare):
            continue
        check = node if isinstance(node, ast.Compare) else node.test
        text = ast.unparse(check)
        bypass = (
            any(
                ast.unparse(node) in raw
                or (
                    isinstance(node, ast.Attribute)
                    and node.attr in {"caller", "origin", "browser_session"}
                    and ast.unparse(node.value) in raw
                )
                for node in ast.walk(check)
            )
            or "REQUEST_PROVENANCE.get()" in text
        )
        if bypass and any(
            word in text for word in ("HUMAN", "'UI'", "'BROWSER'", "browser_session", "ui_origin")
        ):
            out.append(f"{name}:{check.lineno}: raw provenance bypasses relay authority")
    return out


def _relay_owner_forwards(methods: dict[str, ast.FunctionDef]) -> list[str]:
    positions = {
        name: index - 1
        for name, node in methods.items()
        for index, argument in enumerate([*node.args.posonlyargs, *node.args.args])
        if argument.arg == "caller"
    }
    out = []
    for name, node in methods.items():
        parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if name in {"execute", "_execute_observed"} or not any(
            arg.arg == "caller" for arg in parameters
        ):
            continue
        for call in (part for part in ast.walk(node) if isinstance(part, ast.Call)):
            values = [key.value for key in call.keywords if key.arg in {"caller", "chosen_by"}]
            match call.func:
                case ast.Attribute(value=ast.Name(id="self"), attr=target):
                    position = positions.get(target)
                    if position is not None and position < len(call.args):
                        values.append(call.args[position])
            for value in values:
                if ast.unparse(value) not in {"caller", "chosen_by"}:
                    out.append(
                        f"{name}:{call.lineno}: preserve normalized caller for the person relay"
                    )
    return out


def relay_authority_problems(root: Path, *, sources: dict[str, str] | None = None) -> list[str]:
    """Relay admission uses normalized authority, never a browser-only or raw-caller gate."""
    base = "src/alphalattice/control/product_host/composition/"

    def read(name: str) -> ast.Module:
        return (
            ast.parse(sources[name])
            if sources is not None
            else source_syntax_tree(root / base / name)
        )

    host, service = read("local_web_session.py"), read("portfolio_research_operations.py")
    external = [
        node
        for node in ast.walk(host)
        if isinstance(node, ast.FunctionDef)
        and any(
            path == "/api/client/operations" and admitted
            for _method, path, admitted in _decorated_routes(node)
        )
    ]
    methods = {node.name: node for node in ast.walk(service) if isinstance(node, ast.FunctionDef)}
    out = [] if external else ["person relay: /api/client/operations needs external_client=True"]
    if not _relay_dispatch(methods.get("_execute_observed")):
        out.append("person relay: dispatch needs normalized delegation/relayed authority")
    out.extend(
        _relay_boundary_checks(
            [
                *external,
                *(methods[name] for name in ("execute", "_execute_observed") if name in methods),
            ]
        )
    )
    trees = {"local_web_session.py": host, "portfolio_research_operations.py": service}
    if sources is not None:
        trees.update({name: ast.parse(text) for name, text in sources.items() if name not in trees})
    else:
        trees.update(
            {
                path.relative_to(root).as_posix(): source_syntax_tree(path)
                for path in (root / "src/alphalattice").rglob("*.py")
                if path.name not in trees and b"REQUEST_PROVENANCE" in path.read_bytes()
            }
        )
    return (
        out
        + _relay_owner_forwards(methods)
        + [issue for name, tree in trees.items() for issue in _raw_provenance_checks(name, tree)]
    )


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
    "AGENTS.md",
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
_OUTSIDE_TOOLS = frozenset({"git", "uv", "pip", "python", "codex", "claude"})
"""Commands outside the product that a guide names beside its own."""
_OUTSIDE_NAMES = ("ANTHROPIC_",)
"""Name prefixes that belong to a host, not the product, such as its environment variables."""
_NAME = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
"""A product name in capitals: an operation, a status, an action or a code."""

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
    verbs = {verb for _noun, verb in options if verb}
    names = _code_names(root)
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
            guide = path.suffix != ".py"
            out.extend(_retired_names(relative, text, names) if guide else ())
            for match in _NAMED.finditer(text):
                _prefix, noun, verb, rest = match.groups()
                flags = [token.split("=", 1)[0] for token in rest.split() if token.startswith("--")]
                if noun not in nouns:
                    if _a_command(match.groups(), guide=guide, verbs=verbs):
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


def _code_names(root: Path) -> set[str]:
    """Every capitalised name the code has; a guide naming another names a retired one."""
    return {
        name
        for code in (root / "src", root / "scripts")
        for path in (code.rglob("*") if code.is_dir() else ())
        if path.suffix in {".py", ".json", ".js", ".cjs"} and "__pycache__" not in path.parts
        for name in _NAME.findall(path.read_text(encoding="utf-8", errors="replace"))
    }


def _retired_names(relative: str, text: str, names: set[str]) -> list[str]:
    """Each capitalised name a guide gives in backticks that the code no longer has."""
    return [
        f"{relative} names `{name}`, a name the product does not have"
        for name in sorted(set(re.findall(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`", text)))
        if name not in names and not name.startswith(_OUTSIDE_NAMES)
    ]


def _a_command(phrase: tuple[str, ...], *, guide: bool, verbs: set[str]) -> bool:
    """Whether a phrase whose first word is no CLI noun still reads as a command.

    With the root command's prefix, or a word and then a flag, it does; in a guide, a noun and
    a CLI verb with no prefix does too, unless the noun is a tool outside the product.
    """
    prefix, noun, verb, rest = phrase
    retired = guide and verb in verbs and noun not in _OUTSIDE_TOOLS
    return bool(prefix) or retired or (verb is None and rest.lstrip().startswith("--"))


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
