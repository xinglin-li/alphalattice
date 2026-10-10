"""``alphalattice <object> <action>``: every command by name (LAWS OP2; the CLI's grammar).

One grammar for every operation the Host serves, declared in the operation registry
(``GRAMMAR``): each command's object, action, purpose and positional instance, and each field's
flag (a selector names its object: ``--task``, ``--plan``). A command reads at most one document,
by ``--file <path>``; the other fields are read as the request document's schema types them (a
number, a flag, a list as JSON or ``@path``). The command becomes one request document and runs
through the research client, so the answer, its envelope, its exit code and ``--output`` are
the same as every other command's. ``--wait`` follows the Task or trial the answer names;
``--list-next`` shows the next requests as the commands that send them, a page at a time
(``--next-from``); ``--from`` starts the
command from a saved answer (a draft's plan, the answer's next request for this operation, or
its Task), the fields given and ``--choices`` filling it.

The registry is read from the table it writes, ``operations.json``, with the standard library:
a command costs a process start, not the product's models (K1). Only ``schema show``, which
answers without a Host, imports the request document and the typed answers.

Beside the operations, the client's own commands (``CLIENT_COMMANDS``), which it answers
itself: one operation's schema (``schema show``: its fields, their JSON schema, its answer's
schema, each field's meaning with it (V410), and a commented YAML template to fill) and the
waiter (``activity wait``); ``serve``; and ``alphalattice request --file <file>``, a whole
request document in YAML or JSON (``-`` reads stdin), or ``--from`` a saved answer and
``--action`` one of its next requests, ``--choices`` filling what it leaves open. The
workspace, the operations, the activity feed, a declared event and the CPU budget are
operations like any other (V266).

The entry point is ``scripts/run_alphalattice.py`` (it composes the Host, which nothing under
``src`` may import); an installed ``alphalattice`` command points there at packaging.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import textwrap
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Final, Literal, cast

from alphalattice.interface.local_application import client
from alphalattice.interface.local_application.cli_contract import (
    ANSWER_LANGUAGE,
    BOUND_WORKSPACE,
    EXIT_CODES,
    SESSION_LAUNCHES,
    WORKSPACE_FROM,
    Outcome,
    agent_session,
    client_refusal,
    command_table,
    count_launch,
    declaration_operations,
    envelope,
    flag,
    offered_requests,
)
from alphalattice.interface.local_application.failure_codes import (
    owner_failure_code,
    public_failure,
)

CLIENT_COMMANDS: Final[dict[tuple[str, str], str]] = {
    (
        "problem",
        "report",
    ): "Write a redacted problem report for review; sending needs the person's yes.",
    ("answer", "show"): "Read a saved full answer locally, without a Host or fresh verification.",
    ("schema", "show"): "A command's request schema (each branch of a two-operation command), "
    "its answer's and a YAML template.",
    ("activity", "wait"): "Wait, with no timer, for a Task or a goal's work to end or need you.",
    ("committee", "wait"): "Wait, as one committee member, for what its floor addresses to it: "
    "the revealed stances, a challenge, a reply or ruling on its message, or the close.",
    ("first-use", "prepare"): "Open the first use from the person's sentence and named date "
    "and prepare its data under its delegation, following each Task to its end; the answer "
    "echoes the date's sessions and lays out the whole first use. The first stop is the answer.",
    ("strategy", "build"): "Run the research strategy's required Alpha and Risk studies on the "
    "workspace's one input with their defaults, one at a time, then prepare and install the "
    "strategy, following each Task to its end; the first stop is the answer.",
    ("strategy-book", "review"): "Run or reuse an installed strategy's whole-support book, or "
    "with --update read a date's published positions, prepare their Evidence and write every "
    "Analyst bundle, following each Task to its end; the first stop is the answer.",
    ("review", "continue"): "Submit every specialist answer in a folder and follow each "
    "publication; after the Analysts, write the CRO's bundle; after the CRO, read the review and "
    "the strategy's activation offer. The first stop is the answer.",
    ("backup", "restore"): "Restore a backup generation of the held state onto a new "
    "directory, from the backup root alone.",
    ("model", "scaffold"): "Write a new Alpha model's adapter, declaration and contract test "
    "from its YAML declaration.",
    ("model", "check"): "Run a model's contract: each check passes or names what it found.",
    ("model", "sandbox"): "Try a model on a copy of the workspace at rest: one Alpha study, "
    "every saved object of the copy read before and after it, recorded for a person's activation.",
    ("session", "bind"): "Ask the workspace Host to bind this agent session in its exact project: "
    "its commands and its own specialists' may then leave --workspace out. Hook trust and "
    "prospective child observation are checked separately.",
    ("session", "unbind"): "Remove this project's binding: the bound session's own, or the "
    "person's from their own shell, outside any agent session. The research is unchanged.",
}
"""The client's own commands, answered by the client itself, as `serve` and `request` are; every
other command is an operation of the registry (V266, OP1)."""

AGENT_VERBS: Final = {verb.words: name for name, verb in client.AGENT_VERBS.items()}
"""The client's agent verbs by their words, each with the name its `run` knows it by."""

OFFLINE_COMMANDS: Final = frozenset(
    {
        ("answer", "show"),
        ("problem", "report"),
        ("schema", "show"),
        ("backup", "restore"),
        ("model", "scaffold"),
        ("model", "check"),
        ("model", "sandbox"),
        ("session", "unbind"),
    }
)
"""The client's commands that answer without a Host: a restore's workspace may be lost, a
model's files are the checkout's, and a session can detach its project's binding."""


def _commands() -> dict[tuple[str, str], tuple[str, ...]]:
    table: dict[str, list[str]] = command_table()["commands"]
    return {
        (name.split(" ")[0], name.split(" ")[1]): tuple(operations)
        for name, operations in table.items()
    }


def _fields(operation: str) -> tuple[frozenset[str], frozenset[str]]:
    contract = command_table()["fields"][operation]
    return frozenset(contract["required"]), frozenset(contract["allowed"])


def _types(field: str) -> set[str]:
    return set(command_table()["types"].get(field, ()))


def _value(field: str, text: str) -> object:
    """Parse a flag value according to the request schema.

    ``@path`` reads a file: as text for a text field, as a YAML or JSON list for a list field
    (V132), as a YAML or JSON mapping otherwise. ``@@`` stands for a literal ``@`` (V149). A
    boolean is ``true`` or ``false`` and nothing else (V127).
    """
    kinds = _types(field)
    if text.startswith("@@"):
        text = text[1:]
    elif text.startswith("@"):
        path = Path(text[1:])
        if kinds == {"string"}:
            return client._text(path)
        if "array" in kinds and "object" not in kinds:
            return client._listed(path)
        return client._document(path)
    if "object" in kinds or "array" in kinds:
        try:
            return json.loads(text)
        except ValueError as error:
            raise client.LocalResearchClientError("local_client.document_unreadable") from error
    if "integer" in kinds:
        try:
            return int(text)
        except ValueError as error:
            raise client.LocalResearchClientError("local_client.request_invalid") from error
    if "number" in kinds:
        try:
            return float(text)
        except ValueError as error:
            raise client.LocalResearchClientError("local_client.request_invalid") from error
    if "boolean" in kinds:
        if text.lower() not in {"true", "false"}:
            raise client.LocalResearchClientError("local_client.boolean_invalid")
        return text.lower() == "true"
    return text


class _Parser(argparse.ArgumentParser):
    """Report parsing failures as INVALID_INPUT with exit status 1.

    Argparse's own exit status 2 would look like an owner's refusal.
    """

    def error(self, message: str) -> Any:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        code = "local_client.usage_invalid"
        refusal = client_refusal(code)
        # What the parser found, its usage and the help that answers it, in the envelope where
        # an agent reads (the review's F5); stderr keeps argparse's own lines.
        print(
            json.dumps(
                envelope(
                    operation=None,
                    outcome=refusal.outcome,
                    body=None,
                    elapsed_seconds=0.0,
                    status="REFUSED",
                    failure_code=code,
                    detail=refusal.detail,
                    next_action=refusal.next_action,
                    usage_error={
                        "observed": message,
                        "usage": " ".join(self.format_usage().split()),
                        "help": f"{self.prog} --help",
                        **_repair(message),
                    },
                ),
                sort_keys=True,
            )
        )
        raise SystemExit(EXIT_CODES[refusal.outcome])


def _repair(message: str) -> dict[str, str]:
    """The repair a parser's error names, where one is certain (the review's F5)."""
    if "--workspace" in message:
        return {
            "repair": "Name the workspace's folder once on the line: alphalattice --workspace "
            "<dir> <object> <action> ..."
        }
    if "unrecognized arguments" in message:
        return {"repair": "Read the action's flags in its help; a flag is not the field's name."}
    return {}


def _answers(child: argparse.ArgumentParser, *, declaration: bool = False) -> None:
    """How a command's answer is kept and followed, the same for every request."""
    child.add_argument(
        "--output", type=Path, help="Save the owner's full answer, in --format; --from reads it."
    )
    child.add_argument(
        "--format",
        choices=("json", "yaml", "html"),
        default="json",
        help="How --output writes the answer: json or yaml whole, html as its report.",
    )
    if declaration:
        child.add_argument(
            "--save-declaration",
            dest="declaration",
            type=Path,
            help="Write the answer's editable declaration to this file, to edit and plan.",
        )
    child.add_argument("--wait", action="store_true", help="Follow a Task or trial to its end.")
    child.add_argument(
        "--max-wait",
        type=float,
        help="The waiter's only timer, for a command run under a cap; omit it to wait until "
        "the work ends or needs a decision.",
    )


def _saved_read_arguments(child: argparse.ArgumentParser) -> None:
    """A saved-answer read selects a file and exactly one reading view."""
    child.add_argument(
        "--file",
        dest="saved_answer",
        type=Path,
        required=True,
        help="A full owner answer saved by --output, as JSON or YAML (at most 4 MiB).",
    )
    reading = child.add_mutually_exclusive_group()
    reading.add_argument(
        "--section",
        metavar="PATH",
        help="Read one saved part by its dotted path, list index or slice, with its unit.",
    )
    reading.add_argument(
        "--list-sections",
        action="store_true",
        help="List the saved answer's root paths.",
    )
    child.add_argument("--format", choices=("json", "yaml"), default="json")
    child.description = (child.description or "") + (
        " Both views print the selected reading whole. --output saves the full snapshot "
        "reading, including the original answer; saved next requests are never sent."
    )


def _client_command(child: argparse.ArgumentParser, noun: str, verb: str) -> None:
    child.add_argument(
        "--output",
        type=Path,
        required=(noun, verb) == ("problem", "report"),
        help="Save the full answer.",
    )
    if (noun, verb) == ("answer", "show"):
        _saved_read_arguments(child)
    elif noun == "problem":
        child.add_argument(
            "--sentence", help="Safe template: Report the <OPERATION> problem to the developers."
        )
        child.add_argument(
            "--expected-route",
            nargs="+",
            default=[],
            metavar="OPERATION",
            help="Expected registered operation names; omitted means untriaged.",
        )
    elif noun == "schema":
        child.add_argument(
            "operation_name",
            nargs="+",
            metavar="COMMAND",
            help="A command as you run it (`study plan`) or an operation's name "
            "(`EXPERIMENT_PLAN`, as `operation list` gives it).",
        )
    elif (noun, verb) == ("activity", "wait"):
        child.add_argument("--task", dest="task_id", help="The Task to wait for.")
        child.add_argument(
            "--goal", dest="wait_goal_id", help="The goal whose next Task end to wait for."
        )
        child.add_argument(
            "--max-wait",
            type=float,
            help="The only timer, for a command run under a cap; omit it to wait for the event.",
        )
        child.add_argument(
            "--notify",
            choices=("codex-queue",),
            help="With --task: register the Task's wake with the Host for this Codex thread "
            "(CODEX_THREAD_ID) and return at once; the Host queues one line when the Task ends, "
            "needs a decision or is deferred.",
        )
        child.add_argument(
            "--each-stage",
            action="store_true",
            help="With --task: also return as the Task verifies each stage (STAGE_VERIFIED), a "
            "coverage run's unit among them.",
        )
    elif (noun, verb) == ("committee", "wait"):
        child.add_argument(
            "--update", dest="update_task_id", required=True, help="The date's update Task."
        )
        child.add_argument(
            "--role",
            dest="committee_role",
            required=True,
            help="The member this wait is for: PM, ALPHA, RISK or CRO.",
        )
        child.add_argument(
            "--key",
            dest="committee_key",
            required=True,
            help="That member's key: a specialist's is in its bundle, the PM's in the open's "
            "answer.",
        )
        child.add_argument(
            "--seen",
            dest="committee_seen",
            type=int,
            default=0,
            help="The last floor message this member has seen, by number (its answer's `seen`).",
        )
    elif (noun, verb) == ("first-use", "prepare"):
        child.add_argument(
            "--sentence",
            dest="objective",
            required=True,
            help="The person's exact sentence, the first use's objective; the same sentence "
            "and date again reuse its goal.",
        )
        child.add_argument(
            "--date",
            dest="target_date",
            required=True,
            help="The date the sentence names for the positions (YYYY-MM-DD), as you read it; "
            "they are entered on the first session on or after it, decided at the close before.",
        )
    elif (noun, verb) == ("strategy-book", "review"):
        child.add_argument(
            "--package",
            dest="strategy_package_id",
            required=True,
            help="The installed strategy package whose book is reviewed.",
        )
        child.add_argument(
            "--dir",
            dest="bundle_root",
            type=Path,
            required=True,
            help="A folder for the Analyst bundles; each unit's is a new folder inside it.",
        )
        child.add_argument(
            "--update",
            dest="update_task_id",
            help="Review this update Task's published positions, bound by their publication, "
            "in place of the strategy's whole-support book.",
        )
        for field in ("evidence_as_of", "preparation_binding_hash"):
            child.add_argument(flag(field), dest=field, help=command_table()["descriptions"][field])
    elif (noun, verb) == ("review", "continue"):
        child.add_argument(
            "--dir",
            dest="bundle_root",
            type=Path,
            required=True,
            help="A bundle folder with its answer.json, or a folder of such bundles.",
        )
        child.add_argument(
            "--cro-dir",
            dest="cro_root",
            type=Path,
            help="A new folder for the CRO's bundle, needed after the Analysts' answers.",
        )
        child.add_argument(
            "--package",
            dest="strategy_package_id",
            help="After the CRO's answer: the installed strategy whose activation offer to read.",
        )
    elif (noun, verb) == ("model", "scaffold"):
        given = child.add_mutually_exclusive_group(required=True)
        given.add_argument("--file", dest="declaration", help="The model's YAML declaration.")
        given.add_argument(
            "--save-declaration",
            dest="model_declaration_out",
            type=Path,
            help="Write a declaration to edit, from the contract and an installed model's; "
            "scaffold nothing.",
        )
    elif (noun, verb) == ("model", "check"):
        child.add_argument("model_id", metavar="ID", help="The model, as its declaration names it.")
    elif (noun, verb) == ("model", "sandbox"):
        child.add_argument("model_id", metavar="ID", help="The model, as its declaration names it.")
        child.add_argument(
            "--file",
            dest="study",
            help="An Alpha study declaration to run, with this model in place of the one it names; "
            "when omitted, the workspace's latest completed Alpha study that names a model.",
        )
        child.add_argument("--keep", action="store_true", help="Keep the copy for inspection.")
    elif (noun, verb) == ("session", "bind"):
        child.add_argument(
            "--usage",
            choices=("read", "off"),
            default="read",
            help="off: do not discover or read native Session files for model and token usage.",
        )
    elif (noun, verb) == ("backup", "restore"):
        child.add_argument(
            "--dir",
            dest="restore_directory",
            required=True,
            help="A new or empty directory the held state is restored onto.",
        )
        child.add_argument(
            "--generation",
            dest="backup_generation",
            help="The generation to restore, by its hash; the newest when omitted.",
        )
        child.add_argument(
            "--workspace-id",
            help="The workspace's identity, when its manifest cannot be read.",
        )
        child.add_argument(
            "--root",
            dest="backup_root",
            type=Path,
            help="The backup root, when not ALPHALATTICE_BACKUP_ROOT or the default.",
        )
    if (noun, verb) in AGENT_VERBS:
        child.add_argument(
            "--max-wait",
            type=float,
            help="The only timer, for a command run under a cap; omit it to follow each Task.",
        )
        child.add_argument(
            "--notify",
            choices=("codex-queue",),
            help="At the first running Task, register its wake with the Host for this Codex "
            "thread (CODEX_THREAD_ID) and return; the wake names this command to run again, "
            "which reuses each finished step.",
        )


def _named(arguments: Sequence[str]) -> frozenset[str]:
    """The operation nouns a command line names, for which alone the parser builds flags: one
    parser serves every line naming them (the seam check reads two thousand lines, V449)."""
    return frozenset(arguments) & {noun for noun, _verb in _commands()}


@functools.cache
def _parser(named: frozenset[str] | None = None) -> _Parser:
    """Build the command-line parser for the selected nouns.

    With ``named``, only the operations named by the command line receive flags;
    building all operations dominates the cost of a call.
    """
    parser = _Parser(
        prog="alphalattice",
        description="Every AlphaLattice operation by name:\n"
        "  alphalattice [global options] <object> <action> [<id>] [--flag ...]",
        epilog=_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _global_options(parser, anywhere=False)
    nouns = parser.add_subparsers(
        dest="noun", required=True, metavar="<object>", help="One of the objects below."
    )
    start = nouns.add_parser("serve", description="Start the Local Web Host on this workspace.")
    start.add_argument("--stop-on-stdin", action="store_true")
    start.add_argument("--no-browser", action="store_true")
    start.add_argument("--port", type=int, default=0)
    whole = nouns.add_parser(
        "request",
        description="Send one whole request document, or one next request of a saved answer.",
    )
    whole.add_argument(
        "--file",
        type=Path,
        help="The whole request, or with --from the selected operation's declaration (- is stdin).",
    )
    whole.add_argument("--from", dest="from_response", type=Path, help="A saved answer, or -.")
    whole.add_argument("--action", help="Which of that answer's next requests; never implicit.")
    whole.add_argument(
        "--choices",
        type=Path,
        help="With --from: a YAML or JSON file of the open choices the next request leaves.",
    )
    _answers(whole, declaration=True)
    by_noun: dict[str, argparse._SubParsersAction[Any]] = {}
    for (noun, verb), purpose in CLIENT_COMMANDS.items():
        if noun not in by_noun:
            by_noun[noun] = nouns.add_parser(noun, description=_NOUN_WORDS).add_subparsers(
                dest="verb", required=True
            )
        child = by_noun[noun].add_parser(verb, help=purpose, description=purpose)
        child.set_defaults(client_command=(noun, verb))
        _client_command(child, noun, verb)
    table = command_table()
    for (noun, verb), operations in sorted(_commands().items()):
        if noun not in by_noun:
            by_noun[noun] = nouns.add_parser(noun, description=_NOUN_WORDS).add_subparsers(
                dest="verb", required=True
            )
        if named is not None and noun not in named:
            continue
        purpose = " Or: ".join(dict.fromkeys(table["purposes"][op] for op in operations))
        person = any(op in table["person_only"] for op in operations)
        allowed = sorted(set().union(*(_fields(op)[1] for op in operations)))
        required = set.intersection(*(set(_fields(op)[0]) for op in operations))
        positional = next(
            (table["positional"][op] for op in operations if op in table["positional"]), None
        )
        documents = [name for name in table["primary"] if name in allowed]
        others = [name for name in allowed if name != positional and name not in documents]
        line = " ".join(
            [
                *(["<id>"] if positional else []),
                *(["--file <path>"] if set(documents) & required else []),
                *(flag(name) for name in sorted(required & set(others))),
                *(["[--file <path>]"] if documents and not set(documents) & required else []),
                *(
                    [f"[{' '.join(flag(name) for name in sorted(set(others) - required))}]"]
                    if set(others) - required
                    else []
                ),
            ]
        )
        # One line per action: what it does, then its contract, so one object's help shows
        # every action's use (V377, the review's F3).
        child = by_noun[noun].add_parser(
            verb,
            help=f"{purpose}{' (the person decides it)' if person else ''}  {line}".strip(),
            description=f"{purpose}{_person_words(operations, table) if person else ''}",
        )
        child.set_defaults(operations=operations)
        if positional:
            child.add_argument(
                positional, nargs="?", metavar="ID", help=_field_help(positional, required)
            )
        if documents:
            child.add_argument(
                "--file",
                dest="primary_file",
                metavar="PATH",
                help=" ".join(
                    part
                    for part in (
                        "required unless --from supplies it;" if set(documents) & required else "",
                        _field_help(documents[0], set()) or "The command's document",
                        "(YAML or JSON; - reads stdin).",
                    )
                    if part
                ),
            )
        for name in others:
            child.add_argument(flag(name), dest=name, help=_field_help(name, required))
        if "person_confirmation" in others:
            child.add_argument(
                "--asked",
                dest="person_asked",
                metavar="QUESTION",
                help="With --person-said: the one-line question you asked the person, as asked.",
            )
            child.add_argument(
                "--repeat",
                dest="person_repeat",
                metavar="NONCE",
                help="With --person-said: the person's next yes to a decision already made, by "
                "the nonce its refusal named.",
            )
        child.add_argument(
            "--from",
            dest="from_answer",
            type=Path,
            help="Start from a saved answer (- is stdin): a draft's plan, the answer's next "
            "request for this operation, or its Task; the fields given fill it.",
        )
        child.add_argument(
            "--choices",
            type=Path,
            help="With --from: a YAML or JSON file of the open choices the next request leaves.",
        )
        _answers(child, declaration=bool(set(operations) & declaration_operations()))
        child.add_argument(
            "--list-next",
            action="store_true",
            help="Show the next requests as commands, each listed item's too, as the "
            "answer's data in its one envelope; compact, a page at a time (next_left names the "
            "next page's --next-from).",
        )
        child.add_argument(
            "--next-from",
            type=int,
            default=0,
            metavar="N",
            help="With --list-next: list from the Nth entry, commands then templates.",
        )
        child.add_argument(
            "--section",
            metavar="PATH",
            help="Print one part of the answer by its dotted path (result.evidence_report), a "
            "list's items from the fourth as items.3: (omitted_sections names each part the "
            "compact view left out); --output still saves it whole.",
        )
    # Every command reads the global options after its action too, as GNU reads options
    # anywhere on the line: AX14's agent put --view full after the action (V412).
    for command in _commands_of(parser):
        _global_options(command, anywhere=True)
    return parser


def _global_options(parser: argparse.ArgumentParser, *, anywhere: bool) -> None:
    """The options that set the context a command works in (V401), declared once.

    The root reads them before the object; ``anywhere`` adds them to one command, read after
    its action and left out of its help, so a value given there wins and an absent one keeps
    the root's (V412). A command's own option of the same name keeps its own meaning
    (`activity wait --goal`).
    """
    own = {name for action in parser._actions for name in action.option_strings}
    later: dict[str, Any] = (
        {"default": argparse.SUPPRESS, "help": argparse.SUPPRESS} if anywhere else {}
    )

    def add(name: str, **spec: Any) -> None:
        if name not in own:
            parser.add_argument(name, **{**spec, **later})

    add(
        "--workspace",
        type=Path,
        help="The workspace's folder; left out, the one this agent session is bound to "
        "(`session bind`).",
    )
    add(
        "--view",
        choices=("compact", "full"),
        help="compact, the default, shows an answer within one read; full prints it whole, "
        "as a script reads it (--output saves it whole either way).",
    )
    add("--request-timeout", type=float, default=client.DEFAULT_HTTP_WAIT_SECONDS)
    add(
        "--goal",
        help="The goal this request works for, where no agent session has taken one.",
    )
    add(
        "--lang",
        choices=("en", "zh"),
        default="en",
        help="The language of an answer's detail; codes and next actions stay English.",
    )


def _commands_of(parser: argparse.ArgumentParser) -> Iterator[argparse.ArgumentParser]:
    """Each command's own parser: an object's actions, or an object that has none."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                if any(isinstance(item, argparse._SubParsersAction) for item in child._actions):
                    yield from _commands_of(child)
                else:
                    yield child


def _epilog() -> str:
    """The top-level help after its options: the common path, then every object grouped in the
    research's order with what it is, as the registry declares them (V401; Git's help)."""
    table = command_table()
    shown, fields = table["positional"], table["fields"]
    path = ", ".join(
        command
        + (
            " <id>"
            if any(shown.get(op) in fields[op]["required"] for op in table["commands"][command])
            else ""
        )
        for command in table["common_path"]
    )
    groups = "\n".join(
        f"  {group}:\n" + "\n".join(f"    {noun:<18}{purpose}" for noun, purpose in nouns)
        for group, nouns in table["help_groups"]
    )
    closing = (
        "Every answer is one JSON envelope naming its context; exit by outcome: 0 OK, "
        "1 INVALID_INPUT, 2 REFUSED, 3 PENDING, 4 NO_HOST. `alphalattice <object> --help` lists "
        "its actions and what each does; `<object> <action> --help` its flags. --wait follows a "
        "run's Task; a command reads its one document by --file <path>, - for stdin."
    )
    return (
        textwrap.fill(f"The common path: {path}.", 79)
        + f"\n\nObjects, in the research's order:\n{groups}\n\n"
        + textwrap.fill(closing, 79)
    )


_NOUN_WORDS: Final = (
    "Each action says what it does, then its positional <id>, its required flags and its "
    "optional ones in brackets. Every action also takes --from (a saved answer), --choices, "
    "--output, --section, --list-next and --wait. An action whose answer supplies an editable "
    "declaration also takes --save-declaration, as its help shows; `schema show <command>` "
    "prints a request's whole schema."
)


def _field_help(name: str, required: set[str] | frozenset[str]) -> str | None:
    """A field's meaning and limits from its contract (SC3), for its flag's help (V401)."""
    table = command_table()
    kinds = _types(name)
    parts = [
        "required unless --from supplies it;" if name in required else "",
        table["descriptions"].get(name, ""),
        f"({table['limits'][name]})" if name in table["limits"] else "",
        "A list: JSON, or @<file>." if "array" in kinds and "object" not in kinds else "",
    ]
    return " ".join(part for part in parts if part) or None


def _client_fields(args: argparse.Namespace) -> dict[str, Any]:
    """The client's own waiter or agent verb, in the fields the research client's `run`
    reads."""
    if args.client_command in AGENT_VERBS:
        name = AGENT_VERBS[args.client_command]
        fields = (*(field for _flag, field in client.AGENT_VERBS[name].fields), "max_wait")
        return {"command": name, **{field: getattr(args, field) for field in (*fields, "notify")}}
    return {
        "command": "activity-wait",
        "task_id": args.task_id,
        "goal_id": args.wait_goal_id,
        "each_stage": args.each_stage,
        "max_wait": args.max_wait,
        "notify": args.notify,
    }


def _operation_schema(
    schema: dict[str, Any], fields: list[str], *, operation: str | None = None
) -> dict[str, Any]:
    """Keep the selected fields and their transitive definitions, not unrelated operations."""
    properties = {name: schema["properties"][name] for name in fields}
    if operation is not None:
        properties["operation"] = {
            "type": "string",
            "const": operation,
            "description": "This operation, by its name.",
        }
    definitions: dict[str, Any] = {}
    pending: list[Any] = [properties]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                if name not in definitions:
                    definitions[name] = schema["$defs"][name]
                    pending.append(definitions[name])
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return {"properties": properties, "$defs": definitions}


def schema(operation: str) -> dict[str, Any]:
    """One operation as a caller writes it, answered without a Host.

    It is read from the registry and the request document's own schema, and nothing here is a
    second copy of either.
    """
    from alphalattice.interface.local_application.answers import ANSWERS
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument,
    )

    required, allowed = _fields(operation)
    noun, verb = next(name for name, ops in _commands().items() if operation in ops)
    request = _operation_schema(
        PortfolioResearchRequestDocument.model_json_schema(),
        ["operation", *sorted(allowed)],
        operation=operation,
    )
    # The operation's request exactly as its owner reads it: its required fields take no null
    # and no other field is read (the review's F4).
    for name in required:
        spec = request["properties"][name]
        if "anyOf" in spec:
            kept = [option for option in spec["anyOf"] if option.get("type") != "null"]
            spec = {k: v for k, v in spec.items() if k not in {"anyOf", "default"}}
            request["properties"][name] = {
                **spec,
                **(kept[0] if len(kept) == 1 else {"anyOf": kept}),
            }
    groups = command_table()["alternatives"].get(operation, [])
    request = {
        "type": "object",
        "additionalProperties": False,
        **request,
        # Exactly one of the field groups its owner takes (V409).
        **({"oneOf": [{"required": group} for group in groups]} if groups else {}),
    }
    shown = command_table()["positional"].get(operation)
    answer = ANSWERS.get(operation)
    written = _WRITTEN_BY.get(operation)
    lines = [
        f"# {operation}: alphalattice {noun} {verb}{' <id>' if shown else ''}, or this file "
        "with alphalattice request --file <file>",
        *([f"# Its declaration is written elsewhere: {written}."] if written else []),
        f"operation: {operation}",
    ]
    for name in sorted(allowed):
        kinds = " or ".join(sorted(_types(name))) or "a value"
        if name in required:
            lines.append(f"{name}:  # required; {kinds}")
        else:
            lines.append(f"# {name}:  # optional; {kinds}")
    return {
        "operation": operation,
        "command": f"alphalattice {noun} {verb}{' <id>' if shown else ''}",
        "required": sorted(required),
        "allowed": sorted(allowed),
        **request,
        "answer_schema": None if answer is None else answer.model_json_schema(),
        **({"written_by": written} if written else {}),
        **(
            {"declaration_sections": _declaration_sections()}
            if operation == "EXPERIMENT_PLAN"
            else {}
        ),
        **({"answer_parts": parts} if (parts := _answer_parts(operation)) else {}),
        "template": "\n".join(lines) + "\n",
    }


def _answer_parts(operation: str) -> dict[str, Any] | None:
    """The nested parts an answer carries for each study kind, outlined from their owners' models.

    A Factor study's readback holds its receipt and its deterministic result, each an owner's
    model; an agent asks for one of their parts with ``--section`` instead of reading the whole
    answer again (V112: an AX0 run read it whole seven times).

    Args:
        operation: The operation whose answer is described.

    Returns:
        Each study kind's parts, or None for an operation without them.
    """
    if operation != "EXPERIMENT_READBACK":
        return None
    from alphalattice.foundation.factor_research.experiments.authoring import (
        FACTOR_EXPERIMENT_KIND,
    )
    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        FactorDevelopmentReceipt,
    )
    from alphalattice.foundation.factor_research.programs.program import (
        FactorResearchDeterministicEvidence,
    )
    from alphalattice.interface.local_application.answers import outline

    return {
        FACTOR_EXPERIMENT_KIND: {
            "receipt": outline(FactorDevelopmentReceipt),
            "result": outline(FactorResearchDeterministicEvidence),
        }
    }


def _declaration_sections() -> dict[str, Any]:
    """Each Desk's section of an experiment declaration, by its contract (V249, SC3)."""
    from alphalattice.foundation.factor_research.experiments.authoring import FactorSection
    from alphalattice.investment.alpha_research.experiments.authoring import (
        AlphaDevelopmentSection,
    )
    from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
        AlphaLifecycleSection,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.availability import (
        NOT_AVAILABLE,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
        PortfolioExperimentSpec,
    )
    from alphalattice.investment.risk_research.experiments.compiler import RiskSection

    return {
        "alpha": AlphaDevelopmentSection.model_json_schema(),
        "alpha (MODEL_LIFECYCLE_REPLAY)": AlphaLifecycleSection.model_json_schema(),
        "factor": FactorSection.model_json_schema(),
        # The declaration's schema, and what the catalog holds that a study may not declare
        # yet, with why (V313).
        "portfolio": {
            **PortfolioExperimentSpec.model_json_schema(),
            "not_available": list(NOT_AVAILABLE),
        },
        "risk": RiskSection.model_json_schema(),
    }


_PERSON_ONLY_WORDS = (
    ": the person decides it: ask them in one line and, on a clear yes, send it with "
    "--person-said and --asked; without their yes, a client's or an Agent's request is refused."
)


def _person_words(operations: tuple[str, ...], table: dict[str, Any]) -> str:
    """Who completes a person's step: the person, or for a first-use step also the agent the
    person's first-use goal delegates it to (V452)."""
    if set(operations) & set(table["first_use"]):
        return (
            ": the person decides it, or the agent running this workspace's first-use goal, "
            "opened from the person's sentence, for that goal's hours; otherwise ask the person "
            "in one line and, on a clear yes, send it with --person-said and --asked."
        )
    return _PERSON_ONLY_WORDS


_WRITTEN_BY = {
    "EXPERIMENT_PLAN": (
        "`study controls` writes a new study's declaration and `study draft` a "
        "continuation's (`foundation draft` and `book draft` a study built on another's), and "
        "`study plan --from <that answer>` plans it"
    ),
    "RESEARCH_STRATEGY_PLAN": (
        "`strategy controls` names the completed parents and the declaration's schema"
    ),
    "FEATURE_CATALOG_PLAN": "`feature controls` gives a feature declaration's controls",
    "GOAL_OPEN": "`goal schema` gives the goal declaration's schema",
    "GOAL_REVISE": "`goal schema` gives the goal declaration's schema",
    "GOAL_SUBMIT": "`goal schema` gives the completion document's schema",
}
"""The operations whose declaration a command writes, where the template alone holds none
(V125: the `EXPERIMENT_PLAN` template held only its operation, which the Host refuses)."""


def _restore(
    args: argparse.Namespace,
    started: float,
    restore: Callable[..., dict[str, Any]] | None,
) -> int:
    """Restore a backup generation with no Host: the workspace it names may be lost (V328)."""
    answer: dict[str, Any] | None = None
    code = "local_client.restore_unavailable"
    if restore is not None:
        try:
            answer = restore(
                args.workspace,
                Path(args.restore_directory),
                workspace_id=args.workspace_id,
                generation_hash=args.backup_generation,
                root=args.backup_root,
            )
        except (ValueError, OSError) as error:
            code = public_failure(error, "workspace_backup.restore_failed")
    elapsed = time.perf_counter() - started
    if answer is not None:
        print(
            json.dumps(
                envelope(
                    operation="WORKSPACE_RESTORE",
                    outcome="OK",
                    body=answer,
                    elapsed_seconds=elapsed,
                    **_saved(args, answer),
                ),
                sort_keys=True,
            )
        )
        return 0
    refusal = client_refusal(code)
    refused_body = {
        "status": "REFUSED",
        "failure_code": code,
        "detail": refusal.detail,
        "next_action": refusal.next_action,
    }
    print(
        json.dumps(
            envelope(
                operation="WORKSPACE_RESTORE",
                outcome=refusal.outcome,
                body=None,
                elapsed_seconds=elapsed,
                status="REFUSED",
                failure_code=code,
                detail=refusal.detail,
                next_action=refusal.next_action,
                **_saved(args, refused_body),
            ),
            sort_keys=True,
        )
    )
    return EXIT_CODES[refusal.outcome]


def _saved(args: argparse.Namespace, body: dict[str, Any]) -> dict[str, Any]:
    """Save a command's own answer where `--output` names, as every command does (V349).

    The owner's exact answer, a refusal included, through the one no-overwrite writer, so an
    agent that asked for a file finds one; a write that fails is named in the envelope.
    """
    output = getattr(args, "output", None)
    if output is None:
        return {}
    try:
        raw = json.dumps(body, sort_keys=True).encode("utf-8")
        client._save_output(output, body, raw, getattr(args, "format", "json"))
    except client.LocalResearchClientError as error:
        code = public_failure(error, "local_client.output_write_refused")
        refusal = client_refusal(code)
        return {
            "local_failure": {
                "failure_code": code,
                "detail": refusal.detail,
                "next_action": refusal.next_action,
            }
        }
    return {"output_file": str(output.resolve())}


def _sandbox(
    args: argparse.Namespace,
    started: float,
    sandbox: Callable[..., dict[str, Any]] | None,
) -> int:
    """`model sandbox`, with no Host: the workspace is copied at rest (EX)."""
    answer: dict[str, Any] | None = None
    code = "local_client.sandbox_unavailable"
    if sandbox is not None:
        try:
            answer = sandbox(
                args.workspace,
                args.model_id,
                study=Path(str(args.study).removeprefix("@")) if args.study else None,
                keep=bool(args.keep),
            )
        except (ValueError, OSError) as error:
            code = public_failure(error, "model_sandbox.failed")
    elapsed = time.perf_counter() - started
    if answer is not None:
        print(
            json.dumps(
                envelope(
                    operation="MODEL_SANDBOX",
                    outcome="OK",
                    body=answer,
                    elapsed_seconds=elapsed,
                    **_saved(args, answer),
                ),
                sort_keys=True,
            )
        )
        return 0
    refusal = client_refusal(code)
    refused_body = {
        "status": "REFUSED",
        "failure_code": code,
        "detail": refusal.detail,
        "next_action": refusal.next_action,
    }
    print(
        json.dumps(
            envelope(
                operation="MODEL_SANDBOX",
                outcome=refusal.outcome,
                body=None,
                elapsed_seconds=elapsed,
                status="REFUSED",
                failure_code=code,
                detail=refusal.detail,
                next_action=refusal.next_action,
                **_saved(args, refused_body),
            ),
            sort_keys=True,
        )
    )
    return EXIT_CODES[refusal.outcome]


def _model(args: argparse.Namespace, started: float) -> int:
    """`model scaffold` and `model check`, with no Host: a model's files are the checkout's (EX).

    A contract that fails is refused by its first failing check's code, with every finding in the
    answer; a refusal says the way on (OP12).
    """
    from alphalattice.capabilities.alpha_modeling.model_contract import check_model
    from alphalattice.capabilities.alpha_modeling.model_scaffold import scaffold_model

    scaffold = args.client_command == ("model", "scaffold")
    operation = "MODEL_SCAFFOLD" if scaffold else "MODEL_CHECK"
    written = getattr(args, "model_declaration_out", None)
    if scaffold and written is not None:
        # The first step of a model of one's own: a declaration to edit (V413).
        from alphalattice.capabilities.alpha_modeling.model_scaffold import declaration_template

        try:
            client._write_new(written, declaration_template().encode("utf-8"))
        except client.LocalResearchClientError as error:
            code = public_failure(error, "local_client.output_write_refused")
            refusal = client_refusal(code)
            refused_body = {
                "status": "REFUSED",
                "failure_code": code,
                "detail": refusal.detail,
                "next_action": refusal.next_action,
            }
            print(
                json.dumps(
                    envelope(
                        operation=operation,
                        outcome=refusal.outcome,
                        body=None,
                        elapsed_seconds=time.perf_counter() - started,
                        status="REFUSED",
                        failure_code=code,
                        detail=refusal.detail,
                        next_action=refusal.next_action,
                        **_saved(args, refused_body),
                    ),
                    sort_keys=True,
                )
            )
            return EXIT_CODES[refusal.outcome]
        body = {
            "status": "DECLARATION_WRITTEN",
            "declaration_file": str(written.resolve()),
            "next": f'model scaffold --file "{written}"',
        }
        print(
            json.dumps(
                envelope(
                    operation=operation,
                    outcome="OK",
                    body=body,
                    elapsed_seconds=time.perf_counter() - started,
                    **_saved(args, body),
                ),
                sort_keys=True,
            )
        )
        return EXIT_CODES["OK"]
    try:
        answer = (
            scaffold_model(Path(str(args.declaration).removeprefix("@")))
            if scaffold
            else check_model(args.model_id)
        )
    except (ValueError, OSError) as error:
        code = public_failure(error, "model_extension.refused")
        if scaffold and code == "model_declaration.invalid":
            code += _declaration_fields(Path(str(args.declaration)))
        refusal = client_refusal(code)
        refused_body = {
            "status": "REFUSED",
            "failure_code": code,
            "detail": refusal.detail,
            "next_action": refusal.next_action,
        }
        print(
            json.dumps(
                envelope(
                    operation=operation,
                    outcome=refusal.outcome,
                    body=None,
                    elapsed_seconds=time.perf_counter() - started,
                    status="REFUSED",
                    failure_code=code,
                    detail=refusal.detail,
                    next_action=refusal.next_action,
                    **_saved(args, refused_body),
                ),
                sort_keys=True,
            )
        )
        return EXIT_CODES[refusal.outcome]
    failed = next((row["code"] for row in answer.get("findings", ()) if row["code"]), None)
    found = client_refusal(failed) if failed else None
    outcome: Outcome = found.outcome if found else "OK"
    extra: dict[str, Any] = {"next_action": found.next_action} if found else {}
    print(
        json.dumps(
            envelope(
                operation=operation,
                outcome=outcome,
                body=answer,
                elapsed_seconds=time.perf_counter() - started,
                failure_code=failed,
                detail=found.detail if found else None,
                **extra,
                **_saved(args, answer),
            ),
            sort_keys=True,
        )
    )
    return EXIT_CODES[outcome]


def _declaration_fields(path: Path) -> str:
    """The fields a refused model declaration misses or gets wrong, after the code's colon (V413).

    Read here, in the CLI, so the declaration's module, which number-deciding closures hold,
    keeps its bytes.
    """
    import yaml  # type: ignore[import-untyped]
    from pydantic import ValidationError

    from alphalattice.capabilities.alpha_modeling.declaration import AlphaModelDeclaration

    try:
        AlphaModelDeclaration.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as error:
        fields = sorted({str(item["loc"][0]) for item in error.errors() if item.get("loc")})
        return ":" + ",".join(fields) if fields else ""
    except (OSError, yaml.YAMLError):
        return ""
    return ""


def _offline(args: argparse.Namespace, started: float) -> int:
    known = sorted(command_table()["fields"])
    # A command's words name its operations, each a branch of its schema (V401).
    named = " ".join(args.operation_name)
    commands = command_table()["commands"]
    args.operation_name = commands[named][0] if named in commands else named
    if args.operation_name not in known:
        import difflib  # only an unknown name needs it; a call's imports are its cost (W12, V29)

        # The registry's nearest names, where argparse answered only READ_HELP (V111).
        refusal = client_refusal("local_client.operation_unknown")
        print(
            json.dumps(
                envelope(
                    operation="SCHEMA",
                    outcome=refusal.outcome,
                    body=None,
                    elapsed_seconds=time.perf_counter() - started,
                    status="REFUSED",
                    failure_code="local_client.operation_unknown",
                    detail=refusal.detail,
                    next_action=refusal.next_action,
                    nearest_operations=difflib.get_close_matches(
                        args.operation_name, known, n=3, cutoff=0.5
                    ),
                ),
                sort_keys=True,
            )
        )
        return EXIT_CODES[refusal.outcome]
    body = command_schema(named) if named in commands else schema(args.operation_name)
    saved: dict[str, Any] = {}
    if args.output is not None:
        # The file is written as any answer's is, and never over one (V141).
        try:
            client._write_new(args.output, json.dumps(body, sort_keys=True).encode("utf-8"))
            saved["output_file"] = str(args.output.resolve())
        except client.LocalResearchClientError as error:
            code = public_failure(error, "local_client.output_write_refused")
            refusal = client_refusal(code)
            saved["local_failure"] = {
                "failure_code": code,
                "detail": refusal.detail,
                "next_action": refusal.next_action,
            }
    print(
        json.dumps(
            envelope(
                operation="SCHEMA",
                outcome="OK",
                body=body,
                elapsed_seconds=time.perf_counter() - started,
                **saved,
            ),
            sort_keys=True,
        )
    )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    serve: Callable[[list[str]], int],
    restore: Callable[..., dict[str, Any]] | None = None,
    sandbox: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """Run one command line and print its envelope.

    Args:
        argv: The command line without the program name; the process's own when omitted.
        serve: Starts the Local Web Host for ``serve``; the entry point composes it.
        restore: Restores a backup generation for ``backup restore``, with no Host;
            the entry point composes it.
        sandbox: Tries a model on a copy for ``model sandbox``, with no Host; the entry
            point composes it.

    Returns:
        The exit code of the answer's outcome.
    """
    started = time.perf_counter()
    arguments = sys.argv[1:] if argv is None else argv
    # Counted before the arguments are read, so a help launch counts too (V364).
    launches = SESSION_LAUNCHES.set(count_launch())
    # Read before the line parses, so the parser's own refusal reads in it too (V582).
    language = ANSWER_LANGUAGE.set(_line_language(arguments))
    try:
        parser = _parser(_named(arguments))
        args = parser.parse_args(arguments)
        if getattr(args, "client_command", None) == ("answer", "show"):
            return _saved_answer(args, started)
        if getattr(args, "client_command", None) in WORKSPACE_FREE:
            return _unbind(args, started)
        # Read anywhere on the line (V412); left out, it is the session's binding's (V568).
        try:
            args.workspace, bound, chosen_by = _workspace(getattr(args, "workspace", None))
        except client.LocalResearchClientError as error:
            return _refused(str(error), started)
        bound_token = BOUND_WORKSPACE.set(bound)
        chosen_token = WORKSPACE_FROM.set(chosen_by)
        try:
            return _command(parser, args, started, serve=serve, restore=restore, sandbox=sandbox)
        finally:
            WORKSPACE_FROM.reset(chosen_token)
            BOUND_WORKSPACE.reset(bound_token)
    finally:
        ANSWER_LANGUAGE.reset(language)
        SESSION_LAUNCHES.reset(launches)


def _workspace(named: Path | None) -> tuple[Path, Path | None, Literal["OPTION", "BINDING"]]:
    """The workspace a command works in, the one its agent session is bound to, and which chose
    it (V568).

    Named on the line, it is used as named. Left out, it is the session's binding's: the nearest
    binding up from the working directory, when it names this session or this session is one of
    its own specialists (`NativeResearchBinding.serves`). Any other session, or none, is refused
    in words, never given another workspace.

    Args:
        named: The workspace the line names, if it names one.

    Returns:
        The workspace, the one the session is bound to (or None), and ``OPTION`` or ``BINDING``.

    Raises:
        LocalResearchClientError: ``local_client.workspace_unbound`` or
            ``local_client.workspace_bound_to_another_session`` for a workspace left out.
    """
    session = agent_session(os.environ)
    found = None
    bound: Path | None = None
    if session is not None:
        # Only an agent session reads a binding; a call's imports are its cost (W12).
        from alphalattice.interface.local_application.native_bridge import (
            NativeBridgeError,
            NativeResearchBinding,
        )

        with suppress(NativeBridgeError, OSError):
            found = NativeResearchBinding.find(Path.cwd(), session=session)
        if found is not None and found[1].serves(session):
            bound = found[1].workspace.resolve()
    if named is not None:
        return named, bound, "OPTION"
    if bound is not None:
        return bound, bound, "BINDING"
    if found is not None:
        raise client.LocalResearchClientError("local_client.workspace_bound_to_another_session")
    raise client.LocalResearchClientError("local_client.workspace_unbound")


def _saved_answer(args: argparse.Namespace, started: float) -> int:
    """Read a historical file before resolving any workspace, binding or Host.

    The existing answer-part owner selects the display; the full snapshot is saved
    through the same format and no-overwrite owner as a live answer.
    """
    try:
        if args.output is not None and args.output.exists():
            raise client.LocalResearchClientError("local_client.output_exists_choose_another_path")
        body = client.saved_answer(
            args.saved_answer, section=args.section, list_sections=args.list_sections
        )
        full = {"snapshot": body["snapshot"], "answer": body["answer"]}
        saved: dict[str, Any] = {}
        if args.output is not None:
            client._save_output(
                args.output,
                full,
                json.dumps(full, default=client._json_value).encode("utf-8"),
                args.format,
            )
            saved["output_file"] = str(args.output.resolve())
        display = (
            {key: value for key, value in body.items() if key != "answer"}
            if args.section is not None or args.list_sections
            else body
        )
        result = envelope(
            operation="SAVED_ANSWER_SHOW",
            outcome="OK",
            body=display,
            status="READ_SAVED_SNAPSHOT",
            elapsed_seconds=time.perf_counter() - started,
            **saved,
        )
        exit_code = 0
    except client.LocalResearchClientError as error:
        code = public_failure(error, "local_client.saved_answer_invalid")
        # Word the owner's family even when its unsafe path suffix is withheld publicly.
        refusal = client_refusal(str(error))
        diagnostics = {
            name: value
            for name in ("document_size", "document_location", "sections")
            if (value := getattr(error, name, None)) is not None
        }
        result = envelope(
            operation="SAVED_ANSWER_SHOW",
            outcome=refusal.outcome,
            body=None,
            status="REFUSED",
            elapsed_seconds=time.perf_counter() - started,
            failure_code=code,
            detail=refusal.detail,
            next_action=refusal.next_action,
            **diagnostics,
        )
        exit_code = EXIT_CODES[refusal.outcome]
    print(json.dumps(result, default=client._json_value, sort_keys=True))
    return exit_code


def _refused(code: str, started: float) -> int:
    """A refusal the client raised before sending anything, in the one envelope (V568)."""
    refusal = client_refusal(code)
    print(
        json.dumps(
            envelope(
                operation=None,
                outcome=refusal.outcome,
                body=None,
                elapsed_seconds=time.perf_counter() - started,
                status="REFUSED",
                failure_code=code,
                detail=refusal.detail,
                next_action=refusal.next_action,
            ),
            sort_keys=True,
        )
    )
    return EXIT_CODES[refusal.outcome]


WORKSPACE_FREE: Final = frozenset({("session", "unbind")})
"""The client's commands that work in no workspace: a binding is removed from its project, and the
person's own shell, which names none, may remove it (V586)."""


def _unbind(args: argparse.Namespace, started: float) -> int:
    """Remove this project's binding (`session unbind`, V586): the bound session's own, or the
    person's from their own shell; never another project's."""
    from alphalattice.interface.local_application.native_bridge import (
        NativeBridgeError,
        session_project,
    )
    from alphalattice.interface.local_application.native_setup import unbind_session

    session = agent_session(os.environ)
    try:
        answer = unbind_session(
            session_project(Path.cwd(), None if session is None else session[0]),
            session_id=None if session is None else session[1],
            host=None if session is None else session[0],
        )
    except NativeBridgeError as error:
        return _refused(f"local_client.session_unbind_refused:{error}", started)
    except OSError as error:
        code = f"native_bridge.files_unavailable:{type(error).__name__}"
        return _refused(f"local_client.session_unbind_refused:{code}", started)
    print(
        json.dumps(
            envelope(
                operation="SESSION_UNBIND",
                outcome="OK",
                body=answer,
                elapsed_seconds=time.perf_counter() - started,
                **_saved(args, answer),
            ),
            sort_keys=True,
        )
    )
    return 0


def _problem(args: argparse.Namespace, started: float) -> int:
    """Write a local report from existing observations; never send it."""
    from alphalattice.interface.local_application.native_bridge import NativeBridgeError
    from alphalattice.interface.local_application.native_setup import problem_report

    try:
        body = problem_report(
            args.workspace,
            project=Path.cwd(),
            sentence=args.sentence,
            expected_route=args.expected_route,
        )
    except NativeBridgeError as error:
        return _refused(str(error), started)
    print(
        json.dumps(
            envelope(
                operation="PROBLEM_REPORT",
                outcome="OK",
                body=body,
                elapsed_seconds=time.perf_counter() - started,
                **_saved(args, body),
            )
        )
    )
    return 0


def _bind(args: argparse.Namespace, started: float) -> int:
    """Bind this agent session to the workspace named (`session bind`, V568): the binding the
    bridge reads and every later command of the session, or of its own specialists,
    finds from any folder of its project."""
    from alphalattice.interface.local_application.native_bridge import (
        NativeBridgeError,
        session_project,
    )
    from alphalattice.interface.local_application.native_setup import files_unavailable

    # This door now belongs to the served Host. Check its connection before
    # inspecting session/project declarations, like every other Host command.
    try:
        transport = client.LocalResearchClient(Path(args.workspace), timeout=args.request_timeout)
    except client.LocalResearchClientError as error:
        return _refused(str(error), started)
    session = agent_session(os.environ)
    if session is None:
        return _refused("local_client.session_unnamed", started)
    host, _session_id = session
    try:
        project = session_project(Path.cwd(), host)
        bound = transport.bind_native_session(project, usage=args.usage)
    except NativeBridgeError as error:
        return _refused(f"local_client.session_bind_refused:{error}", started)
    except client.LocalResearchClientError as error:
        return _refused(str(error), started)
    except OSError as error:
        bound = files_unavailable(error, project=Path.cwd(), workspace=Path(args.workspace))
    outcome: Outcome = "REFUSED" if bound.get("status") == "REFUSED" else "OK"
    if outcome == "REFUSED":
        code = str(bound.get("failure_code", "native_bridge.binding_invalid"))
        bound = {
            **bound,
            "failure_code": "local_client.session_bind_refused:" + code,
            "detail": bound.get("detail")
            or client_refusal("local_client.session_bind_refused:" + code).detail,
        }
    answer = bound
    print(
        json.dumps(
            envelope(
                operation="SESSION_BIND",
                outcome=outcome,
                body=answer,
                elapsed_seconds=time.perf_counter() - started,
                **_saved(args, answer),
            ),
            sort_keys=True,
        )
    )
    return EXIT_CODES[outcome]


_LINE_LANGUAGE = argparse.ArgumentParser(add_help=False, exit_on_error=False)
_LINE_LANGUAGE.add_argument("--lang", choices=("en", "zh"), default="en")


def _line_language(arguments: Sequence[str]) -> Literal["en", "zh"]:
    """The language a command line names, read as its parser reads ``--lang``, anywhere on the
    line (V412), before the rest of the line is read (V582).

    Args:
        arguments: The command line without the program name.

    Returns:
        The language named; English for a line naming none, or a value the parser refuses.
    """
    try:
        known, _rest = _LINE_LANGUAGE.parse_known_args(list(arguments))
    except argparse.ArgumentError:
        return "en"
    return cast(Literal["en", "zh"], known.lang)


def _command(
    parser: _Parser,
    args: argparse.Namespace,
    started: float,
    *,
    serve: Callable[[list[str]], int],
    restore: Callable[..., dict[str, Any]] | None,
    sandbox: Callable[..., dict[str, Any]] | None,
) -> int:
    """Run one parsed command line and print its envelope (`main`'s body, in its language)."""
    if args.noun == "serve":
        return serve(
            [
                "--workspace",
                str(args.workspace),
                "--port",
                str(args.port),
                *(["--no-browser"] if args.no_browser else []),
                *(["--stop-on-stdin"] if args.stop_on_stdin else []),
            ]
        )
    common = {
        "workspace": args.workspace,
        "view": args.view,
        "request_timeout": args.request_timeout,
        "goal": args.goal,
        "output": args.output,
    }
    if getattr(args, "client_command", None) == ("backup", "restore"):
        return _restore(args, started, restore)
    if getattr(args, "client_command", None) == ("model", "sandbox"):
        return _sandbox(args, started, sandbox)
    if getattr(args, "client_command", None) == ("session", "bind"):
        return _bind(args, started)
    if getattr(args, "client_command", None) == ("problem", "report"):
        return _problem(args, started)
    if getattr(args, "client_command", None) in {("model", "scaffold"), ("model", "check")}:
        return _model(args, started)
    if getattr(args, "client_command", None) in OFFLINE_COMMANDS:
        return _offline(args, started)
    if hasattr(args, "client_command"):
        return client.run(argparse.Namespace(**common, format="json", **_client_fields(args)))
    whole = args.noun == "request"
    request = argparse.Namespace(
        **common,
        command="request",
        file=(args.choices if args.from_response is not None else args.file) if whole else None,
        primary_file=args.file if whole and args.from_response is not None else None,
        from_response=args.from_response if whole else None,
        action=args.action if whole else None,
        format=args.format,
        wait=args.wait,
        max_wait=args.max_wait,
        list_next=getattr(args, "list_next", False),
        next_from=getattr(args, "next_from", 0),
        section=getattr(args, "section", None),
        declaration=getattr(args, "declaration", None),
    )

    def before(document: dict[str, Any]) -> None:
        _read_lead(document, workspace=args.workspace, goal=args.goal)

    def after(document: dict[str, Any], answer: dict[str, Any]) -> None:
        _read_lead(document, workspace=args.workspace, goal=args.goal, after=True, answer=answer)

    if whole:
        return client.run(request, before_send=before, after_send=after)
    return client.run(
        request,
        document_override=_line_document(parser, args),
        before_send=before,
        after_send=after,
    )


def request_of(argv: Sequence[str]) -> dict[str, Any]:
    """The request an operation's command line sends, read as the CLI reads it, never sent.

    The printed command's inverse: a next command run as printed sends the request it came
    from, which the seam checks hold for every offered request (V449).

    Args:
        argv: The command line without its program, an operation's command.

    Returns:
        The request document.

    Raises:
        ValueError: `cli.not_an_operation_command` for a client command or `request`.
    """
    arguments = list(argv)
    parser = _parser(_named(arguments))
    args = parser.parse_args(arguments)
    if args.noun in {"serve", "request"} or hasattr(args, "client_command"):
        raise ValueError("cli.not_an_operation_command")
    return _line_document(parser, args)()


def _line_document(parser: _Parser, args: argparse.Namespace) -> Callable[[], dict[str, Any]]:
    """The request an operation's parsed command line sends, read when the Host is found."""
    operation = _operation(parser, args)
    required, allowed = _fields(operation)
    documents = [name for name in command_table()["primary"] if name in allowed]
    given = {
        name: getattr(args, name)
        for name in sorted(allowed)
        if getattr(args, name, None) is not None
    }
    shown = command_table()["positional"].get(operation)
    missing = sorted(
        "<id>" if name == shown else "--file" if name in documents else flag(name)
        for name in required
        if name not in given and not (name in documents and args.primary_file is not None)
    )
    if args.choices is not None and args.from_answer is None:
        parser.error("--choices fills the next request a saved answer offers: give --from")
    if missing and args.from_answer is None:
        parser.error("the following arguments are required: " + ", ".join(missing))
    # The person's yes, asked in one line, relayed whole and bound to the request sent.
    said, asked = given.pop("person_confirmation", None), getattr(args, "person_asked", None)
    repeat = getattr(args, "person_repeat", None)
    if (said is None) != (asked is None) or (repeat and said is None):
        parser.error("--person-said and --asked go together: the person's words, your question")

    def document() -> dict[str, Any]:
        if (
            getattr(args, "declaration", None) is not None
            and operation not in declaration_operations()
        ):
            # A shared command's execution readback is not its authored preview/catalog.
            raise client.LocalResearchClientError(
                "local_client.operation_has_no_editable_declaration:" + operation
            )
        stdin = [args.from_answer, args.choices, getattr(args, "primary_file", None)]
        if sum(value == "-" or value == Path("-") for value in [*stdin, *given.values()]) > 1:
            raise client.LocalResearchClientError("local_client.document_multiple_stdin_sources")
        fields = {name: _value(name, text) for name, text in given.items()}
        if documents and args.primary_file is not None:
            field, value = client.document_field(documents, args.primary_file)
            fields[field] = value
        if args.choices is not None:
            # The choices fill the offered request's open root fields; a field given twice, by
            # a flag, --file and --choices, is refused by name, never one silently kept.
            chosen = client._document(args.choices)
            twice = sorted(set(chosen) & set(fields))
            if twice:
                raise client.LocalResearchClientError(
                    "local_client.field_given_twice:" + ",".join(twice)
                )
            fields = {**chosen, **fields}
        line = (
            {"operation": operation, **fields}
            if args.from_answer is None
            else client.continuation(operation, args.from_answer, fields, allowed)
        )
        if said is None:
            return line
        from alphalattice.interface.local_application.portfolio_research import decision_hash

        return {
            **line,
            "person_confirmation": {
                "question": asked,
                "words": said,
                "decision_hash": decision_hash(line),
                **({"nonce": repeat} if repeat else {}),
            },
        }

    return document


_LOCATORS: Final = {
    "task_id": ("task_id", "publication_task_id"),
    "experiment_plan_hash": ("plan_hash",),
    "feature_plan_hash": ("plan_hash",),
}
"""Where a saved answer names the instance a selector selects (its Task, its plan)."""


def _operation(parser: _Parser, args: argparse.Namespace) -> str:
    """The command's operation, by the grammar's dispatch rule.

    A command with two operations (`study show`, `feature show`) tells them apart by their
    selectors, which exclude each other: the one whose selector is given; with none and a saved
    answer (`--from`), the one the answer offers as a next request, when it offers exactly one,
    or else the one whose instance the answer names (its Task, its plan); otherwise a usage
    error naming both ways.
    """
    operations: tuple[str, ...] = args.operations
    if len(operations) == 1:
        return operations[0]
    ways = _ways(operations)
    selected = [
        op
        for op in operations
        if all(getattr(args, name, None) is not None for name in _fields(op)[0])
    ]
    if len(selected) > 1:
        parser.error(f"give {ways}, not both")
    if len(selected) == 1:
        return selected[0]
    if args.from_answer is not None:
        answer = client._response_document(args.from_answer)
        offered = {str(request.get("operation")) for request in offered_requests(answer).values()}
        found = [op for op in operations if op in offered]
        if len(found) == 1:
            return found[0]
        # Else the answer's own locators, as a continuation reads them: its Task or its plan.
        located = [
            op
            for op in operations
            if any(answer.get(key) for name in _fields(op)[0] for key in _LOCATORS.get(name, ()))
        ]
        if not found and len(located) == 1:
            return located[0]
    parser.error(f"name what to show: {ways}")
    raise AssertionError("unreachable")


def _ways(operations: tuple[str, ...] | list[str]) -> str:
    """How a two-operation command's branches are told apart: each one's selectors."""
    shown = command_table()["positional"]
    return " or ".join(
        "<id>"
        if shown.get(op) in _fields(op)[0]
        else " ".join(flag(name) for name in sorted(_fields(op)[0]))
        for op in operations
    )


def command_schema(command: str) -> dict[str, Any]:
    """A command's request as its owners read it (V401).

    Its operation's schema; for a command with two operations, each one a branch of ``oneOf``,
    told apart by its selectors. The branches' definitions are the request document's own, so
    they are kept once, at the top, where each branch's ``$ref`` finds them.

    Args:
        command: The command's words (``study show``).

    Returns:
        What `schema show` prints for it.
    """
    operations = command_table()["commands"][command]
    if len(operations) == 1:
        return schema(operations[0])
    branches = [schema(op) for op in operations]
    definitions: dict[str, Any] = {}
    for branch in branches:
        definitions.update(branch.pop("$defs"))
    return {
        "command": f"alphalattice {command}",
        "operations": list(operations),
        "dispatch": (
            f"Give {_ways(operations)}, never both; with neither, a saved answer (--from) "
            "decides: the one it offers as a next request, or else the one whose instance it "
            "names (its Task, its plan)."
        ),
        "$defs": definitions,
        "oneOf": branches,
    }


DECLARATION_CLIENT_COMMANDS: Final = {
    "request": "The selected operation is known only after --from or the request file is read; "
    "an answer without a declaration refuses local_client.declaration_unavailable.",
    "model scaffold": "This client command writes its model starter declaration directly; "
    "it is not a Host operation.",
}
"""The two client commands outside the operation registry that accept --save-declaration."""


LEAD_READING_OPERATIONS: Final = frozenset({"GOAL_TAKE", "GOAL_SUBMIT", "AGENT_ANSWER_SUBMIT"})
"""Read the lead after its Goal is taken, before that Goal or an answer is submitted (V301).
The Host files the reading under the admitted parent Session and its exact active Goal."""


def _read_lead(
    document: dict[str, Any],
    *,
    workspace: Path | None = None,
    goal: str | None = None,
    after: bool = False,
    answer: dict[str, Any] | None = None,
) -> None:
    """Word optional lead-delivery failures without changing the research outcome.

    Args:
        document: The exact request the client sends.
        workspace: The request's resolved workspace, independent of a native binding.
        goal: The request's explicit Goal selector, when named.
        after: The post-exchange callback; take establishes its Goal before usage is filed.
        answer: The post-exchange owner answer, when available.
    """
    operation = document.get("operation")
    if operation not in LEAD_READING_OPERATIONS or (operation == "GOAL_TAKE") != after:
        return
    if after and (answer is None or answer.get("status") != "GOAL_TAKEN"):
        return
    try:
        named_goal = answer.get("goal_id") if after and answer is not None else goal
        if named_goal is None and operation == "GOAL_SUBMIT":
            named_goal = document.get("goal_id")
        receipts = client.lead_readings(
            Path.cwd(), os.environ, workspace=workspace, goal=named_goal
        )
    except Exception as error:
        receipts = [
            {
                "status": "UNAVAILABLE",
                "reason": owner_failure_code(error) or "native_bridge.lead_usage_read_failed",
            }
        ]
    for receipt in receipts:
        if receipt.get("status") in {"DELIVERED", "SKIPPED"}:
            continue
        from alphalattice.interface.local_application.cli_contract import refusal_words
        from alphalattice.interface.local_application.failure_codes import safe_failure_code

        reason = safe_failure_code(receipt.get("reason")) or "native_bridge.lead_usage_read_failed"
        words = refusal_words(reason) or refusal_words("native_bridge.lead_usage_read_failed")
        print(
            json.dumps(
                {
                    "native_observation": {
                        "status": receipt.get("status", "UNAVAILABLE"),
                        "phase": "LEAD_USAGE",
                        "operation": operation,
                        "reason": reason,
                        **words,
                    }
                }
            ),
            file=sys.stderr,
        )


__all__ = ["CLIENT_COMMANDS", "OFFLINE_COMMANDS", "main", "schema"]
