"""The Claude Code host files are exact derivatives of the Codex-native owners."""

import itertools
import json
import re
import shlex
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def materialize():
    from scripts import materialize_claude_host

    return materialize_claude_host


def test_committed_derivatives_match_their_owners(materialize):
    # The TOML cards, the Skill and the Codex hook command are the owners; `.claude/` is derived
    # and must not drift (edit an owner, rerun the script).
    assert materialize.drift(materialize.expected_files()) == []


def test_each_role_card_becomes_one_subagent_with_its_sandbox_tools(materialize):
    cards = {card["name"]: card for card in materialize.role_cards()}
    assert set(cards) == {
        "alphalattice_alpha",
        "alphalattice_cro",
        "alphalattice_data",
        "alphalattice_evidence_analyst",
        "alphalattice_factor",
        "alphalattice_portfolio",
        "alphalattice_risk",
    }
    agent_directory = ROOT / ".claude" / "agents"
    assert {path.stem for path in agent_directory.glob("alphalattice_*.md")} == set(cards)
    assert {
        path.stem for path in materialize.expected_files() if path.parent == agent_directory
    } == set(cards)
    for name, card in cards.items():
        text = (ROOT / ".claude" / "agents" / f"{name}.md").read_text(encoding="utf-8")
        head, body = text.split("\n---\n", 1)
        fields = dict(line.split(": ", 1) for line in head.splitlines()[1:])
        assert fields["name"] == name
        assert json.loads(fields["description"]) == card["description"]
        tools = {tool.strip() for tool in fields["tools"].split(",")}
        # No Agent for any card. The two evidence specialists read their bundle whole and
        # write their own answer file, and run nothing: the lead submits it. The stage roles
        # run their own commands and write their declarations (V384).
        evidence = name in {"alphalattice_evidence_analyst", "alphalattice_cro"}
        assert card["sandbox_mode"] == "workspace-write"
        stage = {"Read", "Grep", "Glob", "Edit", "Write", "Bash"}
        assert tools == ({"Read", "Write"} if evidence else stage)
        assert (fields["model"], fields["effort"]) == ("haiku", "high")
        assert body.strip().endswith(card["developer_instructions"].strip())
        assert "\r" not in text


def _sections(card: Path) -> dict[str, str]:
    """A card's sections by their heading, in order."""

    text = tomllib.loads(card.read_text(encoding="utf-8"))["developer_instructions"].strip()
    return {part.split("\n", 1)[0]: part for part in text.split("\n\n")}


def test_each_card_says_its_place_on_the_skills_paths(materialize):
    """Each card says its place on the skills paths."""

    skill = (ROOT / ".agents/skills/alphalattice-research/SKILL.md").read_text(encoding="utf-8")
    paths = set(re.findall(r"^- \*\*(.+?)\*\*", skill, flags=re.MULTILINE))
    for card in sorted((ROOT / ".codex" / "agents").glob("alphalattice_*.toml")):
        sections = _sections(card)
        place = " ".join(sections["# Place"].split())
        names = re.findall(r'"([^"]+)"', place.split(". ", 1)[0])
        assert names and set(names) <= paths, (card.name, set(names) - paths)
        if card.stem in materialize.BUNDLE_ROLES:
            assert "Do not run product commands or network requests" in place, card.name
        else:
            assert (
                "- EXECUTE: run only the operations your assignment authorizes"
                in (sections["# Role"])
            ), card.name


def test_each_card_carries_its_capability_and_graph(materialize):
    """Each card carries its capability and graph."""

    from alphalattice.interface.local_application.cli_contract import EXIT_CODES

    contract = ROOT / ".agents/skills/alphalattice-research/references/operating.md"
    contract_text = contract.read_text(encoding="utf-8")
    assert contract_text == materialize.operating_text()
    assert all(f"| {code} | {outcome} |" in contract_text for outcome, code in EXIT_CODES.items())
    contract_link = (
        "[Command contract](../../.agents/skills/alphalattice-research/references/operating.md)"
    )
    for card in sorted((ROOT / ".codex" / "agents").glob("alphalattice_*.toml")):
        sections = _sections(card)
        bundle = card.stem in materialize.BUNDLE_ROLES
        middle = "# Bundle" if bundle else "# CLI"
        assert list(sections) == ["# Role", "# Place", middle, "# Method", "# Boundaries"]
        block = sections[middle]
        assert block == materialize.command_block(card.stem), card.name
        assert "\nGraph (" in block, card.name
        if bundle:
            assert "ExecCommand only for read-only reads of exact listed paths" in block
            assert "ApplyPatch only on the nominated answer file" in block
            assert "Read for listed files and Write only for that answer file" in block
            continue
        assert contract_link in block, card.name
        reads, rest = block.split("Reads (ANALYZE, REVIEW):")[1].split("EXECUTE (")
        execute = rest.split("Graph (")[0]
        capability = materialize.ROLE_COMMANDS[card.stem]
        assert re.findall(r"^- `([^`]+)`", reads, flags=re.MULTILINE) == [
            command.form for command in capability.reads
        ]
        assert re.findall(r"^- `([^`]+)`", execute, flags=re.MULTILINE) == [
            command.form for command in capability.execute
        ]
        assert "task show <task>" in reads, card.name
        written = re.findall(r'--(?:output|declaration) "([^"]+)"', execute)
        assert written and all(value.startswith("<out>/") for value in written), card.name
    unknown = materialize.RoleCommand("trial show --trial-idd <trial>", "")
    materialize.ROLE_COMMANDS["probe"] = materialize.RoleCapability((unknown,), (), ())
    try:
        with pytest.raises(materialize.MaterializationError, match="flag_unknown"):
            materialize.command_block("probe")
        known = materialize.RoleCommand("task show <task>", "")
        materialize.ROLE_COMMANDS["probe"] = materialize.RoleCapability(
            (known,), (), ("`task show` → `study run`",)
        )
        with pytest.raises(materialize.MaterializationError, match=r"role_graph\.unknown"):
            materialize.command_block("probe")
    finally:
        del materialize.ROLE_COMMANDS["probe"]


def test_the_skill_carries_every_command_its_paths_use(materialize):
    """requirement (V384): the lead is an agent like the stage roles, so its Skill carries the
    same command list, current with the table, naming every command its shortest paths run."""

    skill = (ROOT / ".agents/skills/alphalattice-research/SKILL.md").read_text(encoding="utf-8")
    assert skill == materialize.skill_text()
    paths = skill.split("## Shortest paths", 1)[1].split("\n## ", 1)[0]
    used = {
        tuple(span.split()[:2]) for span in re.findall(r"`([a-z][a-z-]* [a-z][a-z-]*)[^`]*`", paths)
    }
    listed = {tuple(command.form.split()[:2]) for command in materialize.SKILL_COMMANDS}
    # A path names a person's command only to ask for it (`strategy activate`, `automation set`):
    # the person's section holds those, never the lead's commands.
    table = json.loads(
        (ROOT / "src/alphalattice/interface/local_application/operations.json").read_text(
            encoding="utf-8"
        )
    )
    persons = {
        tuple(name.split()[:2])
        for name, operations in table["commands"].items()
        if set(operations) <= set(table["person_only"])
    }
    assert used and used <= listed | persons, used - listed - persons
    assert not listed & persons, listed & persons


def test_both_default_hosts_register_no_product_hooks(materialize):
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
    with (ROOT / ".codex" / "config.toml").open("rb") as stream:
        codex = tomllib.load(stream).get("hooks", {})
    for event in ("SubagentStart", "SubagentStop"):
        for hooks in (settings.get("hooks", {}), codex):
            assert not any(g.get("matcher") == "^alphalattice_.*$" for g in hooks.get(event, []))


def test_skill_copy_is_byte_exact(materialize):
    source = materialize.SKILL_SOURCE
    copy = ROOT / ".claude" / "skills" / materialize.SKILL_NAME
    files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    assert (copy / "SKILL.md").is_file() and len(files) >= 2
    for relative in files:
        assert (copy / relative).read_bytes() == (source / relative).read_bytes()


def test_settings_merge_keeps_unrelated_keys_and_removes_only_product_matcher(materialize):
    existing = json.dumps(
        {
            "permissions": {"allow": ["Read"]},
            "hooks": {
                "SubagentStart": [
                    {
                        "matcher": "^alphalattice_.*$",
                        "hooks": [{"type": "command", "command": "old"}],
                    },
                    {"matcher": "Explore", "hooks": [{"type": "command", "command": "keep"}]},
                ]
            },
        }
    ).encode()
    merged = json.loads(materialize.settings_document(existing))
    assert merged["permissions"] == {"allow": ["Read"]}
    starts = merged["hooks"]["SubagentStart"]
    assert [g["matcher"] for g in starts] == ["Explore"]
    assert starts[0]["hooks"][0]["command"] == "keep"
    assert "SubagentStop" not in merged["hooks"]
    with pytest.raises(materialize.MaterializationError, match="hooks_invalid"):
        materialize.settings_document(json.dumps({"hooks": []}).encode())


def test_every_host_declaration_says_what_its_card_says(materialize):
    """Every host declaration says what its card says."""

    with (ROOT / ".codex" / "config.toml").open("rb") as stream:
        declared = tomllib.load(stream)["agents"]
    cards = {card["name"]: card for card in materialize.role_cards()}
    assert set(declared) == set(cards)
    for name, agent in declared.items():
        assert agent["description"] == cards[name]["description"], name
        claude = (ROOT / ".claude" / "agents" / f"{name}.md").read_text(encoding="utf-8")
        head = claude.split("\n---\n", 1)[0]
        fields = dict(line.split(": ", 1) for line in head.splitlines()[1:])
        assert json.loads(fields["description"]) == cards[name]["description"], name


def _forms(materialize) -> list[str]:
    """Every command form the generator writes into the Skill and the cards."""
    found: dict[str, None] = {}
    pending: list[object] = [getattr(materialize, name) for name in dir(materialize)]
    while pending:
        value = pending.pop()
        if isinstance(value, materialize.RoleCommand):
            found[value.form] = None
        elif isinstance(value, materialize.RoleCapability):
            pending.extend((*value.reads, *value.execute))
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, tuple | list):
            pending.extend(value)
    return sorted(found)


def test_every_command_form_the_skill_generator_writes_parses(materialize):
    """Every command form the skill generator writes parses."""

    from alphalattice.interface.local_application import cli

    task = "6f9619ff-8b86-4d01-b42d-00cf4fc964ff"
    filled = re.compile(r"<([a-z_]*)>")
    kinds = {"out": "out", "file": "file.yaml", "answer": "answer.json", "effort": "medium"}
    forms = _forms(materialize)
    assert len(forms) > 40
    refused = []
    for form in forms:
        line = filled.sub(
            lambda match: kinds.get(
                match.group(1), task if match.group(1).endswith(("task", "id")) else "x"
            ),
            form,
        )
        arguments = ["--workspace", "w", *shlex.split(line)]
        try:
            cli._parser(cli._named(arguments)).parse_args(arguments)
        except SystemExit:
            refused.append(form)
    assert refused == []


def test_every_command_example_writes_and_reads_under_out_or_the_workspace(materialize):
    """Every command example writes and reads under out or the workspace."""

    nouns = {form.split()[0] for form in _forms(materialize)} | {"request", "bundle", "study"}
    path_flags = {"--output", "--from", "--file", "--choices", "--save-declaration", "--dir"}
    documents = [
        ROOT / "AGENTS.md",
        *sorted((ROOT / ".agents/skills/alphalattice-research").rglob("*.md")),
        *sorted((ROOT / ".codex/agents").glob("*.toml")),
        *sorted((ROOT / ".claude/agents").glob("*.md")),
    ]

    def examples(text: str) -> list[str]:
        found = re.findall(r"`([^`\n]+)`", text)
        fenced = False
        for line in text.splitlines():
            if line.startswith("```"):
                fenced = not fenced
            elif fenced:
                found.append(line.strip())
        return found

    stray = []
    for document in documents:
        for example in examples(document.read_text(encoding="utf-8")):
            try:
                words = shlex.split(example)
            except ValueError:
                continue
            if words and words[0] == "alphalattice":
                words = words[1:]
                while words[:1] in (["--workspace"], ["--view"], ["--lang"], ["--goal"]):
                    words = words[2:]
            if not words or words[0] not in nouns:
                continue
            for flag, value in itertools.pairwise(words):
                kept = value == "-" or value.startswith(("<out>/", "workspaces/my-research/"))
                if flag in path_flags and not kept:
                    stray.append((document.relative_to(ROOT).as_posix(), example))
    assert stray == []


def test_both_agent_guides_review_the_book_before_activation_and_positions_after():
    """P1: activation precedes forward positions; both hosts keep cutoff and causal replay.
    The agent reviews first whether it activates under a first use's delegation (STOPS-1) or
    asks the person."""
    sentence = (
        "Before activating, or asking a person to activate, review the completed historical "
        "book and its "
        "review standing, and read `strategy_dates.information_cutoff` and the conditional "
        "`strategy_dates.first_actionable_session`."
    )
    forward = (
        "After activation, run the offered update and review its first published forward "
        "positions at the first actionable session."
    )
    hold = (
        "Hold positions only from the first actionable session; sessions before it are a "
        "causal replay, inside the research window where marked."
    )
    for path in (
        ROOT / "AGENTS.md",
        ROOT / ".agents/skills/alphalattice-research/SKILL.md",
        ROOT / ".claude/skills/alphalattice-research/SKILL.md",
    ):
        assert path.read_text(encoding="utf-8").count(sentence) == 1
        assert path.read_text(encoding="utf-8").count(forward) == 1
        assert path.read_text(encoding="utf-8").count(hold) == 1
