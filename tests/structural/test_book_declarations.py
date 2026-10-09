"""The editable-declaration seam's operation and exception sets (TE12)."""

import json
import re
from pathlib import Path

from alphalattice.interface.local_application import cli
from alphalattice.interface.local_application.cli_contract import (
    DECLARATION_OPERATION_EXCEPTIONS,
    command_table,
    declaration_contracts,
    declaration_operations,
)

ROOT = Path(__file__).resolve().parents[2]

DECLARATION_OPERATIONS = frozenset(
    {
        "CONTROLS",
        "GOAL_SCHEMA",
        "GOAL_SHOW",
        "GOAL_EXPORT",
        "GOAL_CONTINUE",
        "FEATURE_CATALOG_CONTROLS",
        "FEATURE_CATALOG_PLAN",
        "FEATURE_CATALOG_READBACK",
        "EXPERIMENT_DRAFT",
        "EXPERIMENT_CONTROLS",
        "EXPERIMENT_PREVIEW_READBACK",
        "EXPERIMENT_EXPORT",
        "EXPERIMENT_HANDOFF_PREVIEW",
        "EXPERIMENT_FOUNDATION_DRAFT",
        "EXPERIMENT_PORTFOLIO_DRAFT",
    }
)


def test_every_save_declaration_command_belongs_to_the_declared_answer_set(tmp_path, capsys):
    """contract (TE12): registry-generated declaration operations and actual CLI help
    hold the same set, with the two client writers named explicitly, never a count pin."""
    assert declaration_operations() == DECLARATION_OPERATIONS
    commands = command_table()["commands"]
    expected = {
        name for name, operations in commands.items() if set(operations) & DECLARATION_OPERATIONS
    }
    exceptions = cli.DECLARATION_CLIENT_COMMANDS
    assert set(exceptions) == {"request", "model scaffold"}
    assert all(exceptions.values())
    names = {*commands, *(" ".join(command) for command in cli.CLIENT_COMMANDS), "request"}
    accepts = set()
    for name in sorted(names):
        try:
            cli.main(["--workspace", str(tmp_path), *name.split(), "--help"], serve=lambda _: 99)
        except SystemExit as exit:
            assert exit.code == 0, name
        help = capsys.readouterr().out
        if re.search(r"^\s+--save-declaration(?:\s|$)", help, re.MULTILINE):
            accepts.add(name)
    assert accepts == expected | set(exceptions)
    selected_operations = {
        operation for name in accepts if name in commands for operation in commands[name]
    }
    assert set(DECLARATION_OPERATION_EXCEPTIONS) == {
        "EXPERIMENT_READBACK",
        "FEATURE_CATALOG_BUILD_READBACK",
    }
    assert all(DECLARATION_OPERATION_EXCEPTIONS.values())
    assert selected_operations == DECLARATION_OPERATIONS | set(DECLARATION_OPERATION_EXCEPTIONS)
    for operation, reason in DECLARATION_OPERATION_EXCEPTIONS.items():
        assert "--plan" in reason, operation


def test_unavailable_declarations_are_named_states_with_reasons():
    """Contract: a successful declaration command cannot silently omit its document;
    only these explicit states have no declaration to edit, and each carries its reason."""
    contracts = declaration_contracts()
    actual = {
        (operation, item["field"], value)
        for operation, row in contracts.items()
        for item in row.get("declaration_exceptions", ())
        for value in item["values"]
    }
    assert actual == {
        ("GOAL_SHOW", "state", "COMPLETE"),
        ("GOAL_SHOW", "state", "ABANDONED"),
        ("EXPERIMENT_CONTROLS", "status", "MODEL_TRAINING_SOURCE_SELECTION_REQUIRED"),
        ("EXPERIMENT_CONTROLS", "status", "INPUT_SELECTION_REQUIRED"),
        ("EXPERIMENT_CONTROLS", "status", "RESEARCH_INPUT_NOT_ADMITTED"),
        ("EXPERIMENT_PREVIEW_READBACK", "status", "MISSING"),
        ("EXPERIMENT_PREVIEW_READBACK", "status", "ADMITTED_AS_TASK"),
    }
    for operation, row in contracts.items():
        for item in row.get("declaration_exceptions", ()):
            assert item["reason"].strip(), operation
            assert item["field"] in {field["name"] for field in row["fields"]}, operation


def test_every_book_selector_refusal_has_both_door_words():
    """Contract: the selector class names unselected, uninstalled and empty workspace
    separately, each worded at the door even when no owner context is available."""
    words = json.loads(
        (ROOT / "src/alphalattice/interface/local_application/refusal_words.json").read_text(
            "utf-8"
        )
    )
    for code in {
        "strategy_book.strategy_package_required",
        "local_application.strategy_package_not_installed",
        "research_workspace.strategy_not_installed",
    }:
        assert set(words[code]) == {"detail", "next_action"}
        assert all(words[code].values())


def test_no_owner_exports_a_declaration_as_a_whole_request() -> None:
    """regression (the user's review at de555b07): the Feature readback exported
    `{operation, feature_document}`, which `feature plan --file` wrapped again and refused.
    Every owner's exported `yaml` is the document its plan command's `--file` reads; none dumps a
    request with its `operation` around it."""

    import ast

    wrapped = []
    for path in sorted((ROOT / "src" / "alphalattice").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "safe_dump"
                and node.args
                and isinstance(node.args[0], ast.Dict)
                and any(
                    isinstance(key, ast.Constant) and key.value == "operation"
                    for key in node.args[0].keys
                )
            ):
                wrapped.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert wrapped == [], wrapped
