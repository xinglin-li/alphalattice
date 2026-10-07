"""Derive the Claude Code host files from the Codex-native owners.

Owners: the seven role cards `.codex/agents/*.toml` (their `developer_instructions` are the
professional role text) and the PM Skill `.agents/skills/alphalattice-research/`.
Derivatives: `.claude/agents/<name>.md` (one
subagent per card: sonnet -- the evidence specialists a pinned Sonnet -- high effort, the
tools its card's sandbox allows, no delegation),
a medium-effort variant `<name>_medium.md` of each evidence specialist, a byte copy of the
Skill under `.claude/skills/` (Claude Code reads only that directory). Default
`.claude/settings.json` has no product lifecycle hooks; unrelated settings are preserved.

Each stage card carries its reads, EXECUTE commands and graph (V384, V386), generated
from the operation table, request fields and CLI help. It links the Skill's shared
command contract for syntax, answers and waits; the lead's catalog is in `## Commands`.

Run it after editing an owner. `--check` reports drift without writing; the test uses it.
Nothing here grants product authority: a subagent file is professional guidance,
and every submission still goes through the maintained CLI.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from alphalattice.interface.local_application.failure_codes import setup_failure  # noqa: E402

CARDS = ROOT / ".codex" / "agents"
OPERATIONS = ROOT / "src" / "alphalattice" / "interface" / "local_application" / "operations.json"
SKILL_NAME = "alphalattice-research"
SKILL_SOURCE = ROOT / ".agents" / "skills" / SKILL_NAME
CLAUDE = ROOT / ".claude"
CLAUDE_MODEL = "sonnet"
EVIDENCE_MODEL = "claude-sonnet-5-5"
"""The evidence specialists' model, pinned: an alias moves with each release, and their
recall is measured against the model the card names. A `model` given with the call
overrides the card, so the lead passes none (the Skill's handoff references). Moving it is
a card change (the frozen Evidence/CRO strategy's change rule)."""
CLAUDE_EFFORT = "high"
MEDIUM_EFFORT_ROLES = ("alphalattice_cro", "alphalattice_evidence_analyst")
"""The evidence specialists also get a medium-effort card, `<name>_medium`: the same role text
for small material. Adopted by plan X9 (its recall held on the re-planted yardstick): the lead
gives it a bundle no larger than the one it was measured on (the Skill's handoff references),
and a binding made for the Claude host names it (`scripts/native_research.py`). Claude-side
only: the Codex cards and their model lines are unchanged."""
CLAUDE_TOOLS = {
    # No Agent tool for any card, so a specialist cannot delegate.
    "read-only": "Read, Grep, Glob",
    # The two evidence specialists read their bundle whole and write their own answer file:
    # no Grep or Glob (none of 27 measured runs used either), and no Bash, since the lead runs
    # the submit command (the Codex sandbox is unchanged).
    "workspace-write": "Read, Write",
}
STAGE_TOOLS = "Read, Grep, Glob, Edit, Write, Bash"
"""A stage card's Claude tools: it runs its commands and writes its declarations and saved
answers in the workspace, as its Codex card's workspace-write sandbox allows (V384)."""
HOOK_MATCHER = "^alphalattice_.*$"
HOOK_EVENTS = ("SubagentStart", "SubagentStop")


@dataclass(frozen=True)
class RoleCommand:
    """One command an agent runs: its exact form, what it does, and flags explained beside it."""

    form: str
    does: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoleCapability:
    """A stage role's CLI capability: the reads ANALYZE and REVIEW use, its EXECUTE commands and
    the graph of its work, each line one walk from a command to the next."""

    reads: tuple[RoleCommand, ...]
    execute: tuple[RoleCommand, ...]
    graph: tuple[str, ...]


_WORKSPACE = RoleCommand(
    "workspace show",
    "Input ids, recent studies and Tasks, and `intents`: each flow's needs, holdings and next "
    "requests.",
)
_TASK = RoleCommand("task show <task>", "A Task's actual state and permitted next step.")
_WAIT = RoleCommand(
    "activity wait --task <task>",
    "Waits, without polling, until the Task ends, needs a decision, is deferred, reports an "
    "incident or reaches --max-wait; --each-stage also returns as it verifies each stage.",
)
_FEATURE_REVIEW = RoleCommand(
    "feature review <feature_factor_id> --plan <feature_plan_hash>",
    "The review packet; its `standing` says whether the contract passed.",
)
_CURATE = RoleCommand(
    'curation submit --from "<out>/curation.json" --choices "<out>/choices.yaml" '
    '--output "<out>/decision.json"',
    "Sends `experiment_curation.choices` and `limitations_acknowledged`; the answer offers the "
    "Alpha `handoff`.",
)
_HANDOFF = RoleCommand(
    'handoff preview --from "<out>/decision.json" --save-declaration "<out>/alpha.yaml" '
    '--output "<out>/handoff.json"',
    "The Alpha declaration on the curated factors, from the decision's `handoff`: fill its "
    "target and model.",
)
_FEATURE_CONTROLS = RoleCommand(
    'feature controls --binding <binding> --save-declaration "<out>/feature.yaml"',
    "The formula language, admitted recipes and a feature declaration to edit.",
)
_FEATURE_PLAN = RoleCommand(
    'feature plan --file "<out>/feature.yaml" --output "<out>/feature-plan.json"',
    "Plans one `CREATE` (its `formula`, an explicit `preprocessing_recipe`, a `reason`); offers "
    "the `trial`, the Alpha study to choose.",
)
_TRIAL_START = RoleCommand(
    'trial run --from "<out>/feature-plan.json" --task <alpha_task> --output "<out>/trial.json"',
    "Builds and screens the feature; reruns Alpha if screening admits it. The "
    "`feature_trial.study_not_from_factor_evidence` refusal lists eligible studies.",
)
_TRIAL_SHOW = RoleCommand(
    'trial show --from "<out>/trial.json" --wait --output "<out>/trial-show.json"',
    "Follows the trial until it ends, stops or needs a decision; completed, it offers each "
    "factor's `review`.",
)
_REVIEW = RoleCommand(
    'feature review --from "<out>/trial-show.json" --output "<out>/review.json"',
    "The review packet, from the trial's `review`.",
)
_BOOK_DRAFT = RoleCommand(
    'book draft --from "<out>/alpha.json" --candidate <candidate_id> '
    '--save-declaration "<out>/portfolio.yaml" --output "<out>/draft.json"',
    "Drafts a book from the Alpha study saved in alpha.json; `portfolio.risk_task_id` with "
    "`iv1`/`iv2` or a catalog `portfolio.policy` sizes it by a Risk study on that input.",
)
_LINK_RISK = RoleCommand(
    'risk-link add --from "<out>/book-run.json" --risk-study <risk_task>',
    "Attaches a Risk report to the book as evidence; weights unchanged.",
)
_PREVIEW = RoleCommand(
    'request --from "<out>/issues.json" --action <preview_request> --output "<out>/preview.json"',
    "Previews one offered option; applies nothing.",
)
_DATA_PLAN = RoleCommand(
    'data-update plan --output "<out>/update-plan.json"',
    "Plans a data update without fetching. An update queued, running, deferred or awaiting "
    "recovery returns its own plan; run follows or resumes it.",
)
_DATA_RUN = RoleCommand(
    'data-update run --from "<out>/update-plan.json" --wait',
    "Runs the plan saved, as the workspace's network access allows; a provider's limit defers "
    "it with a retry time, and one stopped on the network is resumed by this same run once a "
    "person allows it.",
)
_ANSWERS_EDGE = "any answer: exit 2 → its `next_requests`; exit 3 → `activity wait` or `task show`"
ROLE_COMMANDS: dict[str, RoleCapability] = {
    "alphalattice_factor": RoleCapability(
        reads=(
            _WORKSPACE,
            RoleCommand(
                "study show <factor_task>",
                "A Factor study, verified: `standing` first, then `result.evidence_report`.",
            ),
            RoleCommand(
                "curation show <factor_task>",
                "Its curation choices; `next_templates` names what a decision chooses.",
            ),
            RoleCommand(
                "trial show <trial>",
                "A feature trial; its `standing` says whether it compared and what changed.",
            ),
            _FEATURE_REVIEW,
            _TASK,
        ),
        execute=(
            RoleCommand(
                'study controls --input <input> --save-declaration "<out>/factor.yaml"',
                "Writes a Factor declaration to edit.",
            ),
            RoleCommand(
                'study plan --input <input> --file "<out>/factor.yaml" --output "<out>/plan.json"',
                "Plans it; nothing runs.",
            ),
            RoleCommand(
                'study run --from "<out>/plan.json" --wait --output "<out>/run.json"',
                "Runs it.",
            ),
            RoleCommand(
                'curation show --from "<out>/run.json" --output "<out>/curation.json"',
                "Saves the study's curation choices for a decision.",
            ),
            _CURATE,
            _HANDOFF,
            _FEATURE_CONTROLS,
            _FEATURE_PLAN,
            _TRIAL_START,
            _TRIAL_SHOW,
            _REVIEW,
            _WAIT,
        ),
        graph=(
            "`workspace show` → `study controls` (input id) | `feature controls` (binding) "
            "| `study show` (Task id)",
            "`study controls` → edit factor.yaml → `study plan` → `study run` "
            "(plan.json) → `curation show` (run.json) → `curation submit` (curation.json, "
            "choices.yaml) → `handoff preview` (decision.json) → the Alpha role",
            "`feature controls` → edit feature.yaml → `feature plan` → `trial run` "
            "(feature-plan.json, an Alpha Task) → `trial show` (trial.json) → `feature review` "
            "(trial-show.json) → a person activates",
            _ANSWERS_EDGE,
        ),
    ),
    "alphalattice_alpha": RoleCapability(
        reads=(
            _WORKSPACE,
            RoleCommand(
                "study summary <alpha_task>",
                "A completed Alpha study's recorded model, training facts and metrics; "
                "metadata only (keep `METADATA_ONLY_BULK_EVIDENCE_NOT_CHECKED`).",
            ),
            RoleCommand(
                "study show <alpha_task>",
                "The study, verified: `standing` first, then candidates and folds.",
            ),
            _TASK,
        ),
        execute=(
            _HANDOFF,
            RoleCommand(
                'study plan --from "<out>/handoff.json" --file "<out>/alpha.yaml" '
                '--output "<out>/alpha-plan.json"',
                "Plans it with the handoff's references; nothing runs.",
            ),
            RoleCommand(
                'study run --from "<out>/alpha-plan.json" --wait --output "<out>/alpha-run.json"',
                "Runs it.",
            ),
            RoleCommand(
                'study draft --from "<out>/alpha-run.json" --save-declaration '
                '"<out>/alpha-next.yaml" --output "<out>/draft.json"',
                'Continues the study; plan it `--from "<out>/draft.json"`.',
            ),
            _WAIT,
        ),
        graph=(
            "`workspace show` → `study summary` | `study show` (Task id)",
            "`handoff preview` (decision.json) → fill alpha.yaml → "
            "`study plan` (handoff.json) → `study run` (alpha-plan.json) → "
            "`study show` (Task id) → the Portfolio role",
            "`study run` (alpha-run.json) → `study draft` → edit alpha-next.yaml → "
            "`study plan` (draft.json) → `study run`",
            _ANSWERS_EDGE,
        ),
    ),
    "alphalattice_risk": RoleCapability(
        reads=(
            _WORKSPACE,
            RoleCommand(
                "study show <risk_task>",
                "A Risk study, verified: `standing` first, then diagnostics, support, coverage.",
            ),
            _TASK,
        ),
        execute=(
            RoleCommand(
                "study controls --input <input> "
                '--kind risk.covariance-development --save-declaration "<out>/risk.yaml"',
                "Writes a Risk declaration to edit.",
            ),
            RoleCommand(
                'study plan --input <input> --file "<out>/risk.yaml" '
                '--output "<out>/risk-plan.json"',
                "Plans it; nothing runs.",
            ),
            RoleCommand(
                'study run --from "<out>/risk-plan.json" --wait --output "<out>/risk-run.json"',
                "Runs it.",
            ),
            _WAIT,
        ),
        graph=(
            "`workspace show` → `study controls` (input id) → edit risk.yaml → "
            "`study plan` → `study run` (risk-plan.json) → `study show` (Task id) "
            "→ the Portfolio role sizes by it",
            _ANSWERS_EDGE,
        ),
    ),
    "alphalattice_portfolio": RoleCapability(
        reads=(
            _WORKSPACE,
            RoleCommand(
                "study show <task>",
                "A book or its Alpha study, verified: `standing` first; a book's holdings, "
                "turnover, cost and performance, an Alpha study's `result.candidates`.",
            ),
            RoleCommand(
                "study compare --left <task> --right <task>",
                "Two completed books on one input and support, compared by their owner: whether "
                "they compare, and how; it names no winner. --session reads them on one date.",
                ("--session",),
            ),
            _TASK,
        ),
        execute=(
            RoleCommand(
                'study show <alpha_task> --output "<out>/alpha.json"',
                "Saves the Alpha study a book draws from; `result.candidates` names each one.",
            ),
            _BOOK_DRAFT,
            RoleCommand(
                'study plan --from "<out>/draft.json" --file "<out>/portfolio.yaml" '
                '--output "<out>/book-plan.json"',
                "Plans it; nothing runs.",
            ),
            RoleCommand(
                'study run --from "<out>/book-plan.json" --wait --output "<out>/book-run.json"',
                "Runs it.",
            ),
            _LINK_RISK,
            _WAIT,
        ),
        graph=(
            "`workspace show` → `study show` (Alpha Task id) → `book draft` "
            "(alpha.json, a candidate) → edit portfolio.yaml → `study plan` (draft.json) → "
            "`study run` (book-plan.json) → `study show` (book Task id)",
            "`study run` (book-run.json) → `risk-link add` (a Risk Task) | the lead's "
            "Evidence and CRO review (read the book's current Evidence before its CRO offers)",
            "`study show` (two book Task ids) → `study compare` (both ids)",
            _ANSWERS_EDGE,
        ),
    ),
    "alphalattice_data": RoleCapability(
        reads=(
            _WORKSPACE,
            RoleCommand(
                "data-update show", "The latest data update: fetched, deferred or refused, why."
            ),
            RoleCommand(
                "issue list",
                "Open data cases: failure, ranges, impact, offered options, preview requests.",
            ),
            _TASK,
        ),
        execute=(
            RoleCommand(
                'issue list --output "<out>/issues.json"',
                "Saves the cases for a preview.",
            ),
            _PREVIEW,
            _DATA_PLAN,
            _DATA_RUN,
            _WAIT,
        ),
        graph=(
            "`workspace show` → `data-update show` | `issue list`",
            "`issue list` (issues.json) → `request` preview (the case's request name) → the "
            "lead, and a person confirms",
            "`data-update plan` (update-plan.json) → `data-update run` (update-plan.json) → "
            "`data-update show`",
            _ANSWERS_EDGE,
        ),
    ),
}
"""Each stage role's CLI capability: its reads, its EXECUTE commands and the graph of its work."""
BUNDLE_ROLES = {
    "alphalattice_evidence_analyst": "every other file",
    "alphalattice_cro": "every file listed under Files, including holdings, findings and coverage",
}
"""The evidence specialists read a bundle and run no command; what each reads after README.md."""
SKILL_COMMANDS: tuple[RoleCommand, ...] = (
    _WORKSPACE,
    RoleCommand(
        "strategy-book controls --package <package>",
        "Reads the exact installed package's activation, book and recorded review standing "
        "before planning Forward work; an inactive book's activation is the person's action "
        "on Portfolio, separate from installation and automatic scheduling.",
        ("--package",),
    ),
    RoleCommand(
        'goal schema --save-declaration "<out>/goal.yaml"',
        "Writes the shortest goal declaration, valid as it stands, to edit.",
    ),
    RoleCommand(
        'goal open --file "<out>/goal.yaml"',
        "Opens a goal before multi-step work and binds this session to it.",
    ),
    RoleCommand(
        'goal show --save-declaration "<out>/submission.yaml"',
        "Writes the bound goal's completion to fill, bound to its revision: its criteria and "
        "deliverable slots, its references listed as the evidence to cite.",
    ),
    RoleCommand(
        'request --file "<out>/submission.yaml"',
        "Submits the filled completion to its goal's revision, last "
        "([goals](references/goals.md)).",
    ),
    RoleCommand(
        'study controls --input <input> --save-declaration "<out>/study.yaml"',
        "Writes a new study's declaration to edit: Factor, or the kind --kind names "
        "(`risk.covariance-development`).",
        ("--kind",),
    ),
    RoleCommand(
        'study plan --input <input> --file "<out>/study.yaml" --output "<out>/plan.json"',
        "Plans a declaration; nothing runs. A handoff, draft or book draft plans "
        "`--from <answer> --file <edited>.yaml`.",
    ),
    RoleCommand(
        'study run --from "<out>/plan.json" --wait --output "<out>/run.json"', "Runs a plan."
    ),
    RoleCommand(
        'study show <task> --output "<out>/study.json"',
        "A study, verified: `standing` first, then its parts (--section).",
    ),
    RoleCommand(
        "study summary <task>",
        "A completed Alpha study's recorded model, training facts and metrics; metadata only.",
    ),
    _TASK,
    RoleCommand(
        'curation show --from "<out>/run.json" --output "<out>/curation.json"',
        "A Factor study's curation choices; `next_templates` names what you choose.",
    ),
    _CURATE,
    _HANDOFF,
    RoleCommand(
        'study draft --from "<out>/run.json" --save-declaration "<out>/next.yaml" --output '
        '"<out>/draft.json"',
        'Continues a saved study; plan it `--from "<out>/draft.json"`.',
    ),
    _BOOK_DRAFT,
    _LINK_RISK,
    _FEATURE_CONTROLS,
    _FEATURE_PLAN,
    _TRIAL_START,
    _TRIAL_SHOW,
    _REVIEW,
    RoleCommand(
        'model scaffold --save-declaration "<out>/model.yaml"',
        "Writes a model's declaration to edit: the contract's fields, an installed model's "
        "values as the example.",
    ),
    RoleCommand(
        'model scaffold --file "<out>/model.yaml"',
        "Writes the model's adapter, declaration and contract test from the edited declaration.",
    ),
    RoleCommand("model check <model>", "Runs the model's contract."),
    RoleCommand(
        "model sandbox <model>",
        "Tries the model on a copy at rest, the Host stopped, for a person's activation. "
        "Uses the latest completed Alpha model-development study that names a model, or "
        "an Alpha study declaration naming one, supplied with --file <study.yaml>.",
    ),
    RoleCommand(
        'issue list --output "<out>/issues.json"', "Open data cases and their preview requests."
    ),
    _PREVIEW,
    _DATA_PLAN,
    _DATA_RUN,
    RoleCommand(
        'research-update plan --package <package> --output "<out>/research-update-plan.json"',
        "Plans a strategy's next sessions and offers `run`. An update queued, running, deferred "
        "or awaiting recovery returns its own plan to follow or resume. If workspace data is "
        "not ready, refuses with `next_requests` to settle it.",
    ),
    RoleCommand(
        'research-update run --from "<out>/research-update-plan.json" --wait',
        "Runs the plan saved, reusing an identical update already made or in flight; one stopped "
        "on the network resumes from where it stopped when this run is sent again after a person "
        "allows it, and one the provider deferred once its `retry_after_at` has passed, which its "
        "read offers as `resume`.",
    ),
    RoleCommand(
        "research-update show --task <task>",
        "The update's published positions, their dates and claim, and its review's requests; "
        "--package, in its place, reads that strategy's own latest update, never another's.",
        ("--package",),
    ),
    RoleCommand(
        "activity wait --task <task>",
        "Waits, without polling, until the Task ends, needs a decision, is deferred, reports an "
        "incident or reaches --max-wait; --goal, in its place, also wakes on the goal's messages "
        "and its closing.",
        ("--goal",),
    ),
)
"""The lead's commands in the Skill's `## Commands`: every command its shortest paths use."""
_SHARED_FLAGS = frozenset(
    {
        "--output",
        "--from",
        "--save-declaration",
        "--wait",
        "--section",
        "--file",
        "--choices",
        "--action",
    }
)
"""Flags the capability's opening explains once, where their help is the common one."""
_OWN_MEANING: frozenset[tuple[str, str, str]] = frozenset()
"""A shared flag whose meaning is the command's own (none since the grammar's `--file`)."""
_PROCEED = {
    "OK": "read `status` and `data`, a result's `standing` first; an OK read is not approval "
    "or a finished Task",
    "INVALID_INPUT": "correct the named command, document or field within scope",
    "REFUSED": "request refused or work blocked, cancelled or stopped: read "
    "`failure_code`, `fields`, `detail` and `next_requests`; take only an authorized "
    "continuation, else report the stop",
    "PENDING": "follow the admitted Task or answer its decision; never resubmit queued or "
    "running work; use the answer's offered deferral resume after its retry time",
    "NO_HOST": "report it: the lead starts or reconnects the Host",
}
"""What each outcome asks of an agent (docs/public-source/cli.md); its exit code is the CLI's."""


class MaterializationError(ValueError):
    """An owner is missing or malformed; nothing is written."""


@functools.cache
def _cli_reference() -> tuple[
    dict[tuple[str, str], dict[str, str]], dict[str, str], dict[str, int]
]:
    """Each command's own flags with their help, each request field's description, and the
    CLI's exit code for each outcome."""

    import argparse as _argparse

    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from alphalattice.interface.local_application import cli
    from alphalattice.interface.local_application.cli_contract import EXIT_CODES

    def children(parser: _argparse.ArgumentParser) -> dict[str, _argparse.ArgumentParser]:
        found = [a for a in parser._actions if isinstance(a, _argparse._SubParsersAction)]
        return dict(found[0].choices) if found else {}

    flags: dict[tuple[str, str], dict[str, str]] = {}
    for noun, noun_parser in children(cli._parser()).items():
        for verb, parser in (children(noun_parser) or {"": noun_parser}).items():
            flags[(noun, verb)] = {
                option: " ".join((action.help or "").split())
                for action in parser._actions
                for option in action.option_strings
            }
    # The fields of each operation a card or the Skill names, as `schema show` writes them:
    # the CLI's own reader (V384).
    table = cli.command_table()["commands"]
    named = {
        " ".join(word for word in command.form.split()[:2] if not word.startswith("-"))
        for capability in ROLE_COMMANDS.values()
        for command in (*capability.reads, *capability.execute)
    } | {
        " ".join(word for word in command.form.split()[:2] if not word.startswith("-"))
        for command in SKILL_COMMANDS
    }
    meanings: dict[str, str] = {}
    for operation in sorted({op for name in named if name in table for op in table[name]}):
        for name, value in cli.schema(operation)["properties"].items():
            meanings.setdefault(name, " ".join(str(value.get("description", "")).split()))
    return flags, meanings, dict(EXIT_CODES)


def _exit_table(exit_codes: dict[str, int]) -> list[str]:
    """Registered outcomes and their continuations, in the shared command contract."""
    if set(exit_codes) != set(_PROCEED):
        raise MaterializationError("cli.exit_codes_changed")
    return [
        "| Exit | Outcome | Do |",
        "| --- | --- | --- |",
        *(
            f"| {code} | {outcome} | {_PROCEED[outcome]} |"
            for outcome, code in sorted(exit_codes.items(), key=lambda item: item[1])
        ),
    ]


def _render(
    owner: str,
    commands: tuple[RoleCommand, ...],
    explained: set[tuple[str, str]],
    reference: tuple[dict[tuple[str, str], dict[str, str]], dict[str, str]],
) -> list[str]:
    """Each command, what it does, and each flag's meaning the first time the block meets it."""

    table = json.loads(OPERATIONS.read_text(encoding="utf-8"))
    flags, meanings = reference
    lines = []
    for command in commands:
        words = command.form.split()
        noun = words[0]
        verb = words[1] if len(words) > 1 and not words[1].startswith("-") else ""
        if (noun, verb) not in flags:
            raise MaterializationError(f"role_command.unknown:{owner}:{noun} {verb}")
        operations = table["commands"].get(f"{noun} {verb}".strip(), [])
        if set(operations) & set(table["person_only"]):
            raise MaterializationError(f"role_command.person_only:{owner}:{noun} {verb}")
        allowed = sorted({f for op in operations for f in table["fields"][op]["allowed"]})
        required = sorted(
            set.intersection(*(set(table["fields"][op]["required"]) for op in operations))
            if operations
            else set()
        )
        by_flag = {"--" + table["flags"][f]: f for f in allowed if f in table["flags"]}
        lines.append(f"- `{command.form}`: {command.does}")
        named = [word for word in words if word.startswith("--")] + list(command.options)
        for flag in named:
            if flag not in flags[(noun, verb)]:
                raise MaterializationError(f"role_command.flag_unknown:{owner}:{flag}")
            if flag in _SHARED_FLAGS and (noun, verb, flag) not in _OWN_MEANING:
                continue
            field = by_flag.get(flag)
            if field in table["references"]:
                continue  # an id or a hash: its placeholder says which (the rules)
            if field is not None and field in allowed and field not in table["primary"]:
                need = " (required)" if field in required else ""
                text = f"{flag}{need}: {meanings[field]}"
            elif flags[(noun, verb)][flag]:
                text = f"{flag}: {flags[(noun, verb)][flag]}"
            else:
                continue
            if (flag, text) not in explained:
                explained.add((flag, text))
                lines.append(f"  {text}")
    return lines


def _graph(owner: str, capability: RoleCapability, nouns: set[str]) -> list[str]:
    """The role's graph; every command a walk names is one of the role's commands."""

    def name(form: str) -> str:
        words = form.split()
        return words[0] if len(words) < 2 or words[1].startswith("-") else " ".join(words[:2])

    named = {name(c.form) for c in (*capability.reads, *capability.execute)}
    for line in capability.graph:
        for command in re.findall(r"`([a-z][a-z-]*(?: [a-z][a-z-]*)?)`", line):
            if command.split()[0] in nouns and command not in named:
                raise MaterializationError(f"role_graph.unknown:{owner}:{command}")
    return [
        "Graph (→ the next step; what carries over):",
        *(f"- {line}" for line in capability.graph),
    ]


def _paragraph(owner: str, lines: list[str]) -> str:
    block = "\n".join(lines)
    if "\\" in block or '"""' in block or "\n\n" in block:
        raise MaterializationError(f"role_command.unwritable:{owner}")
    return block


def command_block(role: str) -> str:
    """The card's generated section: for a stage role its CLI (how to run a command and read
    its answer, its reads, its EXECUTE commands and its graph); for an evidence specialist its
    bundle and its graph."""

    if role in BUNDLE_ROLES:
        return _paragraph(
            role,
            [
                "# Bundle",
                "Read README.md for the task and answer format, then every listed file whole, "
                "one read per file (reads may run together). In Codex, use ExecCommand only for "
                "read-only reads of exact listed paths and ApplyPatch only on the nominated "
                "answer file; in Claude, use Read for listed files and Write only for that answer "
                "file. The bundle is complete: fetch nothing.",
                "Graph (→ the next step):",
                f"- README.md → {BUNDLE_ROLES[role]}, whole, one tool call a file (calls may run "
                "together) → decide → the answer file, written once → one line: written",
                "- a shortened read → reread that file by line range until complete",
                "- the Host's correction, sent by the lead → fix or remove only named items "
                "in the same answer file → one line: written",
            ],
        )
    flags, meanings, _exit_codes = _cli_reference()
    capability = ROLE_COMMANDS[role]
    explained: set[tuple[str, str]] = set()
    return _paragraph(
        role,
        [
            "# CLI",
            "Follow [Command contract]"
            "(../../.agents/skills/alphalattice-research/references/operating.md) for syntax, "
            "answers, continuations, files and waits. The assignment and lists below set "
            "your permissions.",
            "Reads (ANALYZE, REVIEW):",
            *_render(role, capability.reads, explained, (flags, meanings)),
            "EXECUTE (only as authorized):",
            *_render(role, capability.execute, explained, (flags, meanings)),
            *_graph(role, capability, {noun for noun, _verb in flags}),
        ],
    )


def skill_text() -> str:
    """The Skill with its `## Commands` section current, right after its shortest paths."""

    flags, meanings, _exit_codes = _cli_reference()
    body = _paragraph(
        "skill",
        [
            "Follow the [Command contract](references/operating.md) for shared syntax, "
            "answers, continuations, files and waits. The catalog below gives each "
            "operation's exact form and its own flags; run it only within your scope and budget.",
            *_render("skill", SKILL_COMMANDS, set(), (flags, meanings)),
        ],
    )
    text = (SKILL_SOURCE / "SKILL.md").read_text(encoding="utf-8")
    section = "## Commands\n\n" + body + "\n"
    if "\n## Commands\n" in text:
        start = text.index("## Commands\n")
        end = text.index("\n## ", start + 1) + 1
        return text[:start] + section + "\n" + text[end:]
    anchor = "## What only a person does\n"
    if text.count(anchor) != 1:
        raise MaterializationError("skill.commands_anchor_missing")
    return text.replace(anchor, section + "\n" + anchor)


def operating_text() -> str:
    """The shared procedure's exit table, generated from the CLI's registered outcomes."""
    text = (SKILL_SOURCE / "references" / "operating.md").read_text(encoding="utf-8")
    marker = "<!-- Generated from the CLI's registered outcomes and _PROCEED meanings. -->\n"
    if text.count(marker) != 1:
        raise MaterializationError("skill.command_contract_anchor_missing")
    start = text.index(marker) + len(marker)
    end = text.index("\n\n", start)
    return text[:start] + "\n".join(_exit_table(_cli_reference()[2])) + text[end:]


_GENERATED = ("# CLI", "# Bundle", "CLI capability.", "Bundle capability.", "Your commands:")
"""How the generated paragraph of a card begins (the last, its V384 form, is replaced)."""


def card_text(path: Path) -> str:
    """The card's TOML with its capability current, right after its place paragraph."""

    text = path.read_text(encoding="utf-8")
    if path.stem not in ROLE_COMMANDS and path.stem not in BUNDLE_ROLES:
        return text
    opening = 'developer_instructions = """'
    start = text.index(opening) + len(opening)
    end = text.index('"""', start)
    paragraphs = text[start:end].split("\n\n")
    block = command_block(path.stem)
    current = [i for i, p in enumerate(paragraphs) if p.lstrip("\n").startswith(_GENERATED)]
    if current:
        paragraphs[current[0]] = block
    else:
        place = [
            i
            for i, p in enumerate(paragraphs)
            if p.lstrip("\n").startswith("Your place in the product:")
        ]
        if len(place) != 1:
            raise MaterializationError(f"role_card.place_missing:{path.name}")
        paragraphs.insert(place[0] + 1, block)
    return text[:start] + "\n\n".join(paragraphs) + text[end:]


def role_cards() -> list[dict[str, str]]:
    cards = []
    for path in sorted(CARDS.glob("alphalattice_*.toml")):
        card = tomllib.loads(card_text(path))
        for key in ("name", "description", "developer_instructions"):
            if not isinstance(card.get(key), str) or not card[key].strip():
                raise MaterializationError(f"role_card.{key}_missing:{path.name}")
        if card["name"] != path.stem:
            raise MaterializationError(f"role_card.name_mismatch:{path.name}")
        if card.get("sandbox_mode") not in CLAUDE_TOOLS:
            raise MaterializationError(f"role_card.sandbox_mode_unknown:{path.name}")
        cards.append(card)
    if not cards:
        raise MaterializationError("role_cards.missing")
    return cards


def agent_markdown(card: dict[str, str], *, effort: str = CLAUDE_EFFORT) -> str:
    """Claude Code subagent frontmatter, then the role text verbatim; a variant at another
    effort is named `<name>_<effort>` and says so in its description."""
    name, description = card["name"], card["description"]
    if effort != CLAUDE_EFFORT:
        name = f"{name}_{effort}"
        description = f"{description} Reasoning effort {effort}: for small material only."
    lines = [
        "---",
        f"name: {name}",
        f"description: {json.dumps(description, ensure_ascii=False)}",
        f"model: {EVIDENCE_MODEL if card['name'] in MEDIUM_EFFORT_ROLES else CLAUDE_MODEL}",
        f"effort: {effort}",
        "tools: "
        + (STAGE_TOOLS if card["name"] in ROLE_COMMANDS else CLAUDE_TOOLS[card["sandbox_mode"]]),
        "---",
        f"<!-- Derived from .codex/agents/{card['name']}.toml"
        " by scripts/materialize_claude_host.py; edit the TOML, then rerun the script. -->",
        "",
        card["developer_instructions"].strip(),
        "",
    ]
    return "\n".join(lines)


def settings_document(existing: bytes | None) -> str:
    """Remove default product lifecycle groups while keeping every unrelated host setting."""
    document: dict[str, object] = {}
    if existing:
        try:
            loaded = json.loads(existing.decode("utf-8"))
        except (ValueError, UnicodeError):
            raise MaterializationError("claude_settings.unreadable") from None
        if not isinstance(loaded, dict):
            raise MaterializationError("claude_settings.object_required")
        document = loaded
    hooks = document.get("hooks")
    if hooks is None:
        hooks = {}
    if not isinstance(hooks, dict):
        raise MaterializationError("claude_settings.hooks_invalid")
    for event in HOOK_EVENTS:
        kept = [
            group
            for group in (hooks.get(event) or [])
            if not (isinstance(group, dict) and group.get("matcher") == HOOK_MATCHER)
        ]
        if kept:
            hooks[event] = kept
        else:
            hooks.pop(event, None)
    if hooks:
        document["hooks"] = hooks
    else:
        document.pop("hooks", None)
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def expected_files() -> dict[Path, bytes]:
    """Every derivative, keyed by its path under `.claude/`."""
    files: dict[Path, bytes] = {}
    for path in sorted(CARDS.glob("alphalattice_*.toml")):
        if path.stem in ROLE_COMMANDS or path.stem in BUNDLE_ROLES:
            files[path] = card_text(path).encode("utf-8")
    for card in role_cards():
        files[CLAUDE / "agents" / f"{card['name']}.md"] = agent_markdown(card).encode("utf-8")
        if card["name"] in MEDIUM_EFFORT_ROLES:
            files[CLAUDE / "agents" / f"{card['name']}_medium.md"] = agent_markdown(
                card, effort="medium"
            ).encode("utf-8")
    if not (SKILL_SOURCE / "SKILL.md").is_file():
        raise MaterializationError("skill.missing")
    skill = skill_text().encode("utf-8")
    files[SKILL_SOURCE / "SKILL.md"] = skill
    files[SKILL_SOURCE / "references" / "operating.md"] = operating_text().encode("utf-8")
    for source in sorted(p for p in SKILL_SOURCE.rglob("*") if p.is_file()):
        files[CLAUDE / "skills" / SKILL_NAME / source.relative_to(SKILL_SOURCE)] = files.get(
            source, source.read_bytes()
        )
    settings = CLAUDE / "settings.json"
    files[settings] = settings_document(
        settings.read_bytes() if settings.is_file() else None
    ).encode("utf-8")
    return files


def drift(files: dict[Path, bytes]) -> list[str]:
    return sorted(
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path, data in files.items()
        if not path.is_file() or path.read_bytes() != data
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="Report drift; write nothing.")
    args = parser.parse_args(argv)
    try:
        files = expected_files()
        stale = drift(files)
        if args.check:
            print(json.dumps({"status": "CURRENT" if not stale else "DRIFT", "paths": stale}))
            return 0 if not stale else 1
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    except OSError as error:
        # The derived files belong in the checkout; one it cannot write is said, not raised (V539).
        refusal = {
            **setup_failure(error),
            "status": "REFUSED",
            "failure_code": f"claude_host.files_unwritable:{type(error).__name__}",
            "detail": (
                "The Claude Code host files are derived into the checkout's .claude directory, and "
                "this process could not write there, so they may stand part written; run the same "
                "command where the checkout can be written, and `--check` reports any drift left."
            ),
            "next_action": "MATERIALIZE_WHERE_THE_CHECKOUT_CAN_BE_WRITTEN",
        }
        print(json.dumps(refusal))
        return 2
    except Exception as error:
        print(
            json.dumps(
                {
                    **setup_failure(error),
                    "next_commands": {
                        "check": [sys.executable, "scripts/materialize_claude_host.py", "--check"]
                    },
                }
            )
        )
        return 2
    print(json.dumps({"status": "WRITTEN", "paths": stale, "files": len(files)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
