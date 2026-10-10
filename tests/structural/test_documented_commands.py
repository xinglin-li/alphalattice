"""Documented commands and offered choices follow the registry."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from devtools.architecture.operation_registry import incomplete_offers, named_commands
from tests.portfolio_strategy_lab.cli_support import _assert_commands_parse

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"


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
    roots = [repository / "AGENTS.md", repository / "README.md"]
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
        for number, command in examples(path.read_text(encoding="utf-8")):
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
