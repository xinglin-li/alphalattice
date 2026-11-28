"""Compose the CLI from the generated operations and local capability registrations."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

from alphalattice.interface.local_application.cli_contract import command_table
from alphalattice.kernel.identity import identity_blake3, tagged_blake3


class LocalCommand(NamedTuple):
    """A client capability, with the words and options owned by its local entry."""

    words: tuple[str, str]
    purpose: str
    offline: bool = False
    agent: str = ""
    fields: tuple[tuple[str, str], ...] = ()


LOCAL_COMMANDS = (
    LocalCommand(
        ("problem", "report"),
        "Write a redacted problem report for review; sending needs the person's yes.",
        offline=True,
    ),
    LocalCommand(
        ("answer", "show"),
        "Read a saved full answer locally, without a Host or fresh verification.",
        offline=True,
    ),
    LocalCommand(
        ("schema", "show"),
        "A command's request schema (each branch of a two-operation command), "
        "its answer's and a YAML template.",
        offline=True,
    ),
    LocalCommand(
        ("activity", "wait"),
        "Wait, with no timer, for a Task or a goal's work to end or need you.",
    ),
    LocalCommand(
        ("storage", "reclaim"),
        "Show the exact storage cleanup plan and ask before applying it.",
    ),
    LocalCommand(
        ("committee", "wait"),
        "Wait, as one committee member, for what its floor addresses to it: "
        "the revealed stances, a challenge, a reply or ruling on its message, or the close.",
        agent="committee-wait",
        fields=(
            ("--update", "update_task_id"),
            ("--role", "committee_role"),
            ("--key", "committee_key"),
            ("--seen", "committee_seen"),
        ),
    ),
    LocalCommand(
        ("first-use", "prepare"),
        "Open the first use from the person's sentence and named date "
        "and prepare its data under its delegation, following each Task to its end; the answer "
        "echoes the date's sessions and lays out the whole first use. "
        "The first stop is the answer.",
        agent="first-use",
        fields=(("--sentence", "objective"), ("--date", "target_date")),
    ),
    LocalCommand(
        ("strategy", "build"),
        "Run the research strategy's required Alpha and Risk studies on the "
        "workspace's one input with their defaults, one at a time, then prepare and install the "
        "strategy, following each Task to its end; the first stop is the answer.",
        agent="strategy-build",
    ),
    LocalCommand(
        ("strategy-book", "review"),
        "Run or reuse an installed strategy's whole-support book, or "
        "with --update read a date's published positions, prepare their Evidence and write every "
        "Analyst bundle, following each Task to its end; the first stop is the answer.",
        agent="book-review",
        fields=(
            ("--package", "strategy_package_id"),
            ("--dir", "bundle_root"),
            ("--update", "update_task_id"),
            ("--as-of", "evidence_as_of"),
            ("--preparation", "preparation_binding_hash"),
        ),
    ),
    LocalCommand(
        ("review", "continue"),
        "Submit every specialist answer in a folder and follow each "
        "publication; after the Analysts, write the CRO's bundle; "
        "after the CRO, read the review and "
        "the strategy's activation offer. The first stop is the answer.",
        agent="review-continue",
        fields=(
            ("--dir", "bundle_root"),
            ("--cro-dir", "cro_root"),
            ("--package", "strategy_package_id"),
        ),
    ),
    LocalCommand(
        ("backup", "restore"),
        "Restore a backup generation of the held state onto a new "
        "directory, from the backup root alone.",
        offline=True,
    ),
    LocalCommand(
        ("model", "scaffold"),
        "Write a new Alpha model's adapter, declaration and contract test "
        "from its YAML declaration.",
        offline=True,
    ),
    LocalCommand(
        ("model", "check"),
        "Run a model's contract: each check passes or names what it found.",
        offline=True,
    ),
    LocalCommand(
        ("plugin", "new"),
        "Write a factor plugin's manifest, entry module, sample slice and test "
        "into the workspace's plugins folder; the author then writes only its logic.",
        offline=True,
    ),
    LocalCommand(
        ("plugin", "verify"),
        "Run a workspace plugin in its guarded child on its sample slice: each "
        "check passes, or the first fault names what is wrong and what to change.",
        offline=True,
    ),
    LocalCommand(
        ("model", "sandbox"),
        "Try a model on a copy of the workspace at rest: one Alpha study, "
        "every saved object of the copy read before and after it, "
        "recorded for a person's activation.",
        offline=True,
    ),
    LocalCommand(
        ("session", "bind"),
        "Ask the workspace Host to bind this agent session in its exact project: "
        "its commands and its own specialists' may then leave --workspace out. Hook trust and "
        "prospective child observation are checked separately.",
    ),
    LocalCommand(
        ("session", "unbind"),
        "Remove this project's binding: the bound session's own, or the "
        "person's from their own shell, outside any agent session. The research is unchanged.",
        offline=True,
    ),
    LocalCommand(
        ("session", "exporters"),
        "Print native agent exporter settings for the served Host; "
        "only on the person's yes, write Claude Code's project settings. "
        "Codex's user config is printed.",
    ),
)

ALIASES = {("committee", "verdict"): ("committee", "submit")}
ROOT_COMMANDS = {
    "serve": "Start the Local Web Host on this workspace.",
    "request": "Send one whole request document, or one next request of a saved answer.",
}


class CliProfile(NamedTuple):
    """Each command is claimed once. Operation branches within one command share its claim."""

    operations: dict[tuple[str, str], tuple[str, ...]]
    local: dict[tuple[str, str], LocalCommand]


def compose(
    operations: Mapping[str, Sequence[str]],
    local: Iterable[LocalCommand],
    aliases: Mapping[tuple[str, str], tuple[str, str]],
) -> CliProfile:
    """Compose the operation projection, client capabilities and their compatibility aliases."""
    names: set[tuple[str, ...]] = {(name,) for name in ROOT_COMMANDS}
    remote: dict[tuple[str, str], tuple[str, ...]] = {}
    capabilities: dict[tuple[str, str], LocalCommand] = {}

    def claim(words: tuple[str, ...]) -> None:
        if words in names or words[0] in ROOT_COMMANDS:
            raise ValueError(f"cli.command_duplicate: {' '.join(words)}")
        names.add(words)

    for name, branches in operations.items():
        noun, verb = name.split()
        claim((noun, verb))
        remote[noun, verb] = tuple(branches)
    for row in local:
        claim(row.words)
        capabilities[row.words] = row
    for words, target in aliases.items():
        claim(words)
        remote[words] = remote[target]
    return CliProfile(remote, capabilities)


@lru_cache(maxsize=1)
def registered_commands() -> CliProfile:
    """Read the shipped operation projection without importing its request models."""
    return compose(command_table()["commands"], LOCAL_COMMANDS, ALIASES)


AGENT_COMMANDS = {row.agent: row for row in LOCAL_COMMANDS if row.agent}


@dataclass(slots=True)
class _Composition:
    command: str
    modules: set[str]


_ACTIVE: ContextVar[_Composition | None] = ContextVar("cli_composition", default=None)
_LOG = logging.getLogger(__name__)


@contextmanager
def composing(parser_module: str) -> Iterator[None]:
    """Keep a command's bindings local to its invocation, including on a resident Host."""
    run = _Composition("", {__name__, parser_module})
    token = _ACTIVE.set(run)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        modules = sorted(run.modules)
        # Registered words and module identifiers are NUL-free. The first field is
        # the command and every following field is a module, sorted once for identity.
        body = "\0".join((run.command, *modules)).encode("utf-8")
        digest = tagged_blake3(identity_blake3(b"alphalattice.cli-composition\0\x01" + body))
        _LOG.debug(
            "CLI composition",
            extra={
                "composition_receipt": {
                    "profile": "cli",
                    "command": run.command,
                    "modules": modules,
                    "identity": digest,
                }
            },
        )


def note_loaded(*modules: str) -> None:
    """Name the modules the profile bound at its own import boundary."""
    if (run := _ACTIVE.get()) is not None:
        run.modules.update(modules)


def selected_command(name: str) -> None:
    """Name the parsed command in its composition receipt."""
    if (run := _ACTIVE.get()) is not None:
        run.command = name
