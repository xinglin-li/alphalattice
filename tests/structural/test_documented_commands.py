"""Documented commands and offered choices follow the registry."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from devtools.architecture.operation_registry import (
    agent_surface,
    incomplete_offers,
    named_commands,
    surface_problems,
    userspace_problems,
)
from tests.portfolio_strategy_lab.cli_support import _assert_commands_parse

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


@pytest.mark.parametrize("change", ["remove", "type", "required", "enum", "baseline"])
def test_the_userspace_ratchet_refuses_silent_interface_changes(change: str) -> None:
    """Frozen names and typed contracts survive edits to code or the baseline."""
    entry = {"required": False, "schema": {"type": "string", "enum": ["A", "B"]}}
    frozen = {"release": "0.2.0", "entries": {"answer:STATUS.state": entry}, "retirements": {}}
    live = {
        "answer:STATUS.state": {
            "required": False,
            "schema": {"type": "string", "enum": ["A", "B", "C"]},
        }
    }
    assert surface_problems(frozen, live) == []
    previous = json.loads(json.dumps(frozen))
    if change == "remove":
        live.clear()
    elif change == "baseline":
        frozen["entries"].clear()
    elif change == "required":
        live["answer:STATUS.state"]["required"] = True
    else:
        live["answer:STATUS.state"]["schema"][change] = "integer" if change == "type" else ["A"]
    assert any(
        "answer:STATUS.state" in p for p in surface_problems(frozen, live, previous=previous)
    )


def test_retirement_is_announced_before_removal_and_cannot_be_erased() -> None:
    """A deprecated interface works until its declared removal release."""
    row = {
        "status": "Deprecated",
        "reason": "Use the typed read",
        "replacement": "operation:STATUS",
        "announced": "0.2.0",
        "earliest_removal": "0.3.0",
    }
    frozen = {
        "release": "0.2.0",
        "entries": {"command:old": {}},
        "retirements": {"command:old": row},
    }
    live = {"command:old": {}, "operation:STATUS": {}}
    assert surface_problems(frozen, live) == []
    del live["command:old"]
    assert surface_problems(frozen, live)
    frozen["release"] = "0.3.0"
    assert surface_problems(frozen, live) == []
    previous = json.loads(json.dumps(frozen))
    frozen["retirements"].clear()
    assert any("retirement erased" in p for p in surface_problems(frozen, live, previous=previous))


def test_the_frozen_surface_and_release_retirements_match_the_owners() -> None:
    """The live typed surface obeys the shipped manifest and release list."""
    root = SCRIPT.parents[1]
    assert userspace_problems(root) == []
    entries = agent_surface(root)
    assert "command:committee show" in entries and "option:committee show/--update" in entries
    assert "law:product/PR1" in entries and "law:workbench/PR1" in entries
    assert "message:RULES/STANCE" in entries and "exchange:CORRECTION" in entries


def test_a_command_a_text_names_is_one_the_cli_has(tmp_path: Path) -> None:
    """A product text names only commands and flags the CLI admits."""
    text = tmp_path / "case-study" / "README.md"
    text.parent.mkdir()
    text.write_text(
        "Run `schema show STATUS`, `study controls` and `request --from x`; not "
        "`capabilities --operation STATUS`, `study frobnicate` or `study show --sectoin run`. "
        "Prose such as `uv sync --locked` is no command.\n",
        encoding="utf-8",
    )
    guide = tmp_path / "AGENTS.md"  # a guide: a retired command or name, beside outside ones
    guide.write_text(
        "`capabilities show`, `WORKSPACE_PREPARE_PLANZ`; `git show HEAD`, `ANTHROPIC_MODEL`, "
        "`WORKSPACE_PREPARE_PLAN`.\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "names.py").write_text('PLAN = "WORKSPACE_PREPARE_PLAN"\n', "utf-8")
    found = named_commands(tmp_path)
    assert [line.split(" names ")[1] for line in found if line.startswith("case-study")] == [
        "`capabilities`, a command the CLI does not have",
        "`study frobnicate`, a command the CLI does not have",
        "`study show --sectoin`, a flag the CLI does not have",
    ]
    named = [line.split("`")[1] for line in found if line.startswith("AGENTS.md")]
    assert named == ["WORKSPACE_PREPARE_PLANZ", "capabilities"]


def test_every_documented_command_line_parses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every documented invocation parses through the CLI command grammar."""
    repository = SCRIPT.parents[1]
    roots = [repository / name for name in ("AGENTS.md", "README.md", "README.zh-CN.md")]
    roots.extend(
        repository / name
        for name in (
            ".agents/skills/alphalattice-research",
            ".claude/skills/alphalattice-research",
            ".codex/agents",
            ".claude/agents",
            "docs/public-source",
        )
    )
    paths = []
    for root in roots:
        assert root.exists(), root
        files = (
            sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else [root]
        )
        assert files, root
        paths.extend(files)
    grammar = {
        "alphalattice <object> <action>",
        "alphalattice <noun> <verb>",
        "alphalattice <command> [flags]",
        "alphalattice ...",
        "uv run alphalattice ...",
    }
    fences = re.compile("^\\s*```[^\\n]*\\n(.*?)^\\s*```[ \\t]*$", re.M | re.S)

    def examples(text: str) -> Iterator[tuple[int, str]]:
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            number, line = (index + 1, lines[index].strip())
            if line.startswith(("alphalattice ", "& $al $cli ")):
                while line.endswith(("\\", "`")) and index + 1 < len(lines):
                    index += 1
                    line = line[:-1] + " " + lines[index].strip()
                yield (number, line)
            index += 1
        prose = fences.sub(lambda block: re.sub("[^\\n]", " ", block.group()), text)
        for match in re.finditer("`([^`]+)`", prose):
            command = " ".join(match.group(1).split())
            prefix = prose[prose.rfind("\n", 0, match.start()) + 1 : match.start()]
            if command.startswith(("alphalattice ", "uv run alphalattice ")) or (
                re.fullmatch("\\s*-\\s*", prefix)
                and len(command.split()) > 1
                and (not command.startswith("--"))
            ):
                yield (prose.count("\n", 0, match.start()) + 1, command)

    dummy_id = "00000000-0000-4000-8000-000000000001"
    values = dict.fromkeys(
        [
            "alpha-kind",
            "alpha-task-id",
            "alpha_task",
            "analyst-bundle-name",
            "binding",
            "book",
            "book-task-id",
            "candidate-id",
            "candidate_id",
            "declared-model-id",
            "dir",
            "evidence-task-id",
            "existing-workspace-path",
            "explicit-workspace",
            "factor-task-id",
            "factor_task",
            "feature_factor_id",
            "feature_plan_hash",
            "goal-id",
            "goal",
            "id",
            "component-id",
            "input",
            "input-id",
            "model",
            "out",
            "package",
            "plan-hash",
            "prepared-task",
            "preview_request",
            "returned-action-name",
            "risk-kind",
            "risk-task-id",
            "risk_task",
            "session-id",
            "task",
            "task-id",
            "trial",
            "unit",
        ],
        dummy_id,
    )
    values.update(
        {
            name: "documentation-path"
            for name in ("dir", "out", "existing-workspace-path", "explicit-workspace")
        }
    )
    values.update(
        {
            "binding": "1" * 64,
            "reason": "Documentation example",
            "sentence": "Prepare the research workspace",
            "date": "2026-10-12",
            "hash": "1" * 64,
            "ROLE": "ALPHA",
            "alpha-kind": "alpha.model-development",
            "feature_plan_hash": "1" * 64,
            "plan-hash": "1" * 64,
            "key": "1" * 32,
            "risk-kind": "risk-covariance-development",
            "tickers": "AAPL",
            "unit": "u01",
        }
    )
    variables = {
        "ws": "documentation-workspace",
        "baselineTask": dummy_id,
        "factor": "formula_factor",
        "featurePlan": "1" * 64,
        "inputBinding": "1" * 64,
    }
    commands, covered = ([], set())
    for path in paths:
        text = path.read_text(encoding="utf-8")
        # Every OS's command: `uv run python` for the checkout, `bin/python` beside `Scripts`.
        assert ".venv/Scripts" not in text, path
        assert "Scripts/python.exe" not in text or "bin/python" in text, path
        for number, command in examples(text):
            if command in grammar:
                continue
            covered.add(path)
            location = f"{path.relative_to(repository)}:{number}: {command}"
            normalized = command.removeprefix("& $al $cli ")
            names = set(re.findall("<([^<>]+)>", normalized))
            assert names <= values.keys(), (location, names - values.keys())
            normalized = re.sub("<([^<>]+)>", lambda match: values[match[1]], normalized)
            names = set(re.findall("\\$([A-Za-z_]\\w*)", normalized))
            assert names <= variables.keys(), (location, names - variables.keys())
            normalized = re.sub("\\$([A-Za-z_]\\w*)", lambda match: variables[match[1]], normalized)
            normalized = normalized.removeprefix("uv run ").removeprefix("alphalattice ")
            commands.append((location, normalized))
    assert all(any(root == path or root in path.parents for path in covered) for root in roots)
    _assert_commands_parse(commands, monkeypatch, capsys)


def test_an_incomplete_offer_names_its_missing_required_choice(tmp_path: Path) -> None:
    """Incomplete source offers name every required choice they omit."""
    owner = tmp_path / "src" / "alphalattice" / "owner.py"
    owner.parent.mkdir(parents=True)
    owner.write_text(
        'OFFER = {"operation": "EXPERIMENT_PORTFOLIO_DRAFT", "task_id": "t-1"}\n', encoding="utf-8"
    )
    (line,) = incomplete_offers(tmp_path)
    assert "without its required candidate_id" in line


def test_goal_record_operations_are_validated_without_request_selectors(tmp_path: Path) -> None:
    """A typed ledger record carries an operation; actual offers still carry its selectors."""
    owner = tmp_path / "src/alphalattice/owner.py"
    owner.parent.mkdir(parents=True)
    owner.write_text(
        """class Owner:
    def __init__(self, store: GoalStore):
        self.store, self.clock = store, clock
    def record(self):
        self.store.attribute(goal_id, {"operation": "DATA_ISSUE_CONFIRM", "case_token": "case"})
    def offer(self):
        return {"operation": "EXPERIMENT_PORTFOLIO_DRAFT", "task_id": "t-1"}
""",
        encoding="utf-8",
    )
    (line,) = incomplete_offers(tmp_path)
    assert "EXPERIMENT_PORTFOLIO_DRAFT" in line and "candidate_id" in line
    from devtools.architecture.operation_registry import _offered

    assert set(_offered(tmp_path)) == {"DATA_ISSUE_CONFIRM", "EXPERIMENT_PORTFOLIO_DRAFT"}
    owner.write_text(
        owner.read_text(encoding="utf-8").replace("GoalStore", "OtherStore"), encoding="utf-8"
    )
    assert len(incomplete_offers(tmp_path)) == 5
