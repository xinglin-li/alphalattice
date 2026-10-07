"""The CLI's answer, the same shape for every command (binding plan, C1 rule 1).

Every command prints one envelope: the operation, an `outcome` of five, the owner's status,
the owner's body under `data`, its failure code, its words and its next requests, and how
long the command took. The exit code is `outcome`'s, one for one, so a script or an Agent
branches on it without reading the body: 0 OK, 1 INVALID_INPUT (the request never reached an
owner), 2 REFUSED, 3 PENDING (work continues), 4 NO_HOST (no Host answered). The owner's own
status stays in `data`, unchanged; the CLI renders the owners' words and computes nothing of
its own. A refusal the client raises itself is read from one table: its outcome, the next
action and the words.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
from collections import Counter
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple
from uuid import UUID

from alphalattice.kernel.shared_kernel.project_layout import command_prefix

SCHEMA_VERSION: Final = 1

CLIENT_HEADER: Final = "X-Alphalattice-Client"
WORKSPACE_HEADER: Final = "X-Alphalattice-Workspace"
INSTANCE_HEADER: Final = "X-Alphalattice-Instance"
"""The client's wire to the Host: the connection's token, the workspace's key and the Host
instance, one header each, checked by the Host's boundary (`web.py`)."""

AGENT_VENDOR_HEADER: Final = "X-Alphalattice-Agent-Vendor"
AGENT_SESSION_HEADER: Final = "X-Alphalattice-Agent-Session"
GOAL_HEADER: Final = "X-Alphalattice-Goal"
"""The agent session a request comes from and the goal it names: provenance the client
declares, never authority and never identity (LAWS ID6, OP13); a session bound to a workspace
(`session bind`) also takes it as its default, which grants nothing a named `--workspace` would
not (V568). The Host records them beside what the request did, so a result outlives its session
and a goal holds what was done for it."""

AGENT_SESSION_VARIABLES: Final = (
    ("claude-code", "CLAUDE_CODE_SESSION_ID"),
    ("codex", "CODEX_THREAD_ID"),
)
"""Where each vendor names its session in the command environment (read on 2026-09-28; not a
documented promise, so a missing one only leaves a request without an agent session)."""

_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def agent_session(environ: Mapping[str, str]) -> tuple[str, str] | None:
    """The one agent session a command runs in, as its host names it in the environment.

    Every reader of the session reads it here, so the session a request names and the one its
    launches count to are the same (V583). Both vendors' variables at once is an agent started
    inside another, whose session cannot be told apart, so it names none.

    Args:
        environ: The command's environment.

    Returns:
        The vendor and the session's id; None for no session, two, or an id not a plain token.
    """
    found = [(vendor, environ.get(name, "").strip()) for vendor, name in AGENT_SESSION_VARIABLES]
    sessions = [(vendor, value) for vendor, value in found if value]
    if len(sessions) == 1 and _SESSION_ID.match(sessions[0][1]):
        return sessions[0]
    return None


def agent_provenance_headers(environ: Mapping[str, str], goal: str | None = None) -> dict[str, str]:
    """The headers naming the agent session a command runs in, and the goal it names.

    The session is `agent_session`'s; with none named, `--goal` still attributes the request.
    """
    session = agent_session(environ)
    headers: dict[str, str] = {}
    if session is not None:
        headers = {AGENT_VENDOR_HEADER: session[0], AGENT_SESSION_HEADER: session[1]}
    if goal is not None:
        headers[GOAL_HEADER] = goal
    return headers


@dataclass(frozen=True, slots=True)
class RequestProvenance:
    """Who sent one request, as its client declared it: an agent session and a named goal.

    `delegation` is the Host's, never a client's: the person's delegation the request acts under
    (a first-use goal's, V452), which an owner recording who decided names.
    """

    vendor: str | None = None
    session: str | None = None
    goal_id: str | None = None
    delegation: str | None = None

    def subject(self) -> dict[str, str]:
        """The references an activity row records for this request."""
        return {
            key: value
            for key, value in (
                ("agent_vendor", self.vendor),
                ("agent_session", self.session),
                ("goal_id", self.goal_id),
                ("delegation", self.delegation),
            )
            if value is not None
        }


def request_provenance(headers: Mapping[str, str]) -> RequestProvenance | None:
    """A request's provenance headers, read at the Host's boundary; a malformed one refused."""
    vendor = headers.get(AGENT_VENDOR_HEADER)
    session = headers.get(AGENT_SESSION_HEADER)
    goal = headers.get(GOAL_HEADER)
    if vendor is None and session is None and goal is None:
        return None
    if (vendor is None) != (session is None) or (
        vendor is not None
        and (
            vendor not in {name for name, _ in AGENT_SESSION_VARIABLES}
            or not _SESSION_ID.match(session or "")
        )
    ):
        raise ValueError("local_web.agent_session_invalid")
    if goal is not None:
        try:
            goal = str(UUID(goal))
        except ValueError as error:
            raise ValueError("local_web.goal_header_invalid") from error
    return RequestProvenance(vendor=vendor, session=session, goal_id=goal)


REQUEST_PROVENANCE: ContextVar[RequestProvenance | None] = ContextVar(
    "alphalattice_request_provenance", default=None
)
"""The provenance of the request the Host's boundary is running: set by `web.py` around one
external request's handler, read where the Host records what that request did."""

ANSWER_LANGUAGE: ContextVar[Literal["en", "zh"]] = ContextVar(
    "alphalattice_answer_language", default="en"
)
"""The language one command words its answer's `detail` in (`--lang`, U19): set by `cli.main`
around the command; codes, statuses and next actions stay English, for Agents."""

BOUND_WORKSPACE: ContextVar[Path | None] = ContextVar("alphalattice_bound_workspace", default=None)
"""The workspace this command's agent session is bound to (`session bind`), resolved; None when
it is bound to none (V568). Set by `cli.main`; a printed command for this workspace leaves
`--workspace` out (`entry`), since the session's next command finds it the same way."""

WORKSPACE_FROM: ContextVar[Literal["OPTION", "BINDING"]] = ContextVar(
    "alphalattice_workspace_from", default="OPTION"
)
"""How this command's workspace was chosen, which every answer's `context` names (V568): named
by `--workspace`, or the session's binding."""

SESSION_LAUNCHES: ContextVar[int | None] = ContextVar("alphalattice_session_launches", default=None)
"""How many CLI launches this agent session has made, this one included (V364): set by
`cli.main` from `count_launch`, carried by every envelope as `session_launches`."""

_LAUNCH_LINE: Final = b"1\n"


def count_launch(environment: Mapping[str, str] | None = None) -> int | None:
    """Count this launch for the agent session the environment names (V364).

    Help and schema launches count as any other, so an agent reads one total against its
    budget. The count lives in the system's temporary directory, one file per session, never
    in a workspace, which the Host alone writes. The session is `agent_session`'s, so an agent
    started inside another, whose session cannot be told apart, counts to neither (V583).

    Args:
        environment: The process environment; this process's when None.

    Returns:
        The session's launches so far, this one included; None without an agent session.
    """
    named = agent_session(os.environ if environment is None else environment)
    if named is None:
        return None
    import tempfile

    folder = Path(tempfile.gettempdir()) / "alphalattice-cli-launches"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{named[0]}--{named[1].replace(':', '_')}.count"
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags, 0o644)
        try:
            os.write(descriptor, _LAUNCH_LINE)
        finally:
            os.close(descriptor)
        return path.stat().st_size // len(_LAUNCH_LINE)
    except OSError:
        return None  # an unwritable count never stops the command


_ZH_TABLE: Final = Path(__file__).parent / "assets" / "workbench-source" / "js" / "data" / "zh.js"
"""The UI owner's Chinese table, English source text to Chinese; the CLI reads it, never a copy."""

_ZH_ENTRY: Final = re.compile(r'^\s*("(?:[^"\\]|\\.)*")\s*:\s*("(?:[^"\\]|\\.)*"),?\s*$', re.M)

MAXIMUM_REQUEST_BODY_BYTES: Final[int] = 128 * 1024
"""The largest request body this service will read, named rather than implied.

Most writes this product takes are one small JSON document -- a spec, a result
hash, a task id. The largest is a review's structured answer (an Analyst answer
or a CRO assessment), which travels whole in one request; the bound was raised
from 64 KiB to 128 KiB for it on 2026-09-25, at the researcher's request. Each
answer's owner still bounds the answer itself. A bound has to exist somewhere,
and a bound that is written down and checked *before* the read is the
difference between refusing an oversized request and allocating for it.
"""

Outcome = Literal["OK", "INVALID_INPUT", "REFUSED", "PENDING", "NO_HOST"]
EXIT_CODES: Final[dict[Outcome, int]] = {
    "OK": 0,
    "INVALID_INPUT": 1,
    "REFUSED": 2,
    "PENDING": 3,
    "NO_HOST": 4,
}

_PENDING_DISPOSITIONS: Final = frozenset(
    {"ADMITTED", "REUSED_IN_FLIGHT", "RECOVERY_RESUBMITTED", "UPSTREAM_PROMOTION_ADMITTED"}
)
_PENDING_STATES: Final = frozenset(
    {"QUEUED", "RUNNING", "DEFERRED", "REVIEW_PENDING", "CANCEL_REQUESTED", "RECOVERY_REQUIRED"}
)
_REFUSED_STATES: Final = frozenset({"BLOCKED", "CANCELLED"})
ACTION_STATES: Final = frozenset({"REVIEW_PENDING", "RECOVERY_REQUIRED"})
"""Pending states that wait on a request, not on time: a review to decide, a Task to recover."""
WAIT_EXITS: Final[Mapping[str, str]] = {
    **dict.fromkeys(ACTION_STATES, "NEEDS_DECISION"),
    "DEFERRED": "DEFERRED",
}
"""The pending states every waiter ends on, each with the event it names: a decision or a
recovery waits on a request, and a deferred Task on its retry time and then on its plan sent
again, which its read names (V507: `--wait` sat on a deferral the Task never leaves by itself)."""


TASK_STATE_FIELDS: Final = ("lifecycle", "task_lifecycle", "status")
"""Where an answer names its Task's state: ``lifecycle``, ``task_lifecycle`` where its own
``status`` names something else (a data update, a review), or, as readbacks write it,
``status`` (OP18, V455)."""


def task_state(body: dict[str, Any]) -> str | None:
    """The Task state an answer carries, wherever it names it (`TASK_STATE_FIELDS`).

    Args:
        body: The full owner answer.

    Returns:
        The first Task state named, or nothing when the answer names none.
    """
    for field in TASK_STATE_FIELDS:
        value = body.get(field)
        if isinstance(value, str) and value in _PENDING_STATES | _REFUSED_STATES:
            return value
    return None


def refused(body: dict[str, Any]) -> bool:
    """Check whether an owner's answer carries a refusal.

    Args:
        body: The full owner answer.

    Returns:
        Whether a failure code, refusal, or refused disposition is present. A `refused`
        field is a refusal only as a code: the controls answer lists the controls a book
        refuses under it, and that catalog is an answer (V333).
    """
    code = body.get("refused")
    return bool(
        body.get("failure_code")
        or (isinstance(code, str) and code)
        or any(
            str(body.get(field, "")).startswith("REFUSED") for field in ("status", "disposition")
        )
    )


_REFUSAL_WORDS: Final = Path(__file__).with_name("refusal_words.json")
"""Each refusal code's words and way on, where its owner gives none in place (OP4, V449)."""

NETWORK_ACCESS_REFUSALS: Final = frozenset(
    {
        "research_update.input_source_access_not_admitted",
        "workspace_data_update.source_access_not_admitted",
        "workspace_preparation.source_access_not_admitted",
        "evidence_review.workspace_network_not_allowed",
    }
)
"""Permission refusals whose way on reads the effective network decision (OP5, TE12)."""


@functools.cache
def _refusal_table() -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = json.loads(_REFUSAL_WORDS.read_text(encoding="utf-8"))
    return table


def refusal_words(code: str, *, workspace: Path | None = None) -> dict[str, Any]:
    """A refusal code's words and next action, where the table words it (OP4, V449).

    Args:
        code: The refusal's code; what follows its first colon fills the words' `{subject}`.
        workspace: The actual control for network advice; absent context offers a read only.

    Returns:
        Its ``detail`` and ``next_action``; empty for a code the table does not word.
    """
    base, _sep, subject = code.partition(":")
    found = _refusal_table().get(base)
    if found is None:
        return {}
    words: dict[str, Any] = {
        "detail": found["detail"].replace("{subject}", subject),
        "next_action": found["next_action"],
    }
    if base in NETWORK_ACCESS_REFUSALS:
        from alphalattice.control.workspace_runtime.network_access import network_access

        access = network_access(workspace)
        decision = access.body()
        prefix = (
            f"The research update needs {subject}. "
            if base.startswith("research_update.")
            else "The command gave SEC consent. "
            if base == "evidence_review.workspace_network_not_allowed"
            else ""
        )
        if access.decided_by in {"OPERATOR_OFFLINE_SWITCH", "RUN_HELD_OFFLINE"}:
            words.update(
                detail=prefix + str(decision["detail"]), next_action=decision["next_action"]
            )
        elif workspace is None:
            words.update(
                detail=prefix + "Read `network show` for this workspace's effective network "
                "decision and take its named way on before retrying the refused step.",
                next_action="READ_NETWORK_ACCESS",
            )
        elif access.allowed:
            words.update(
                detail=prefix + "Network access is allowed now. Retry the refused step; "
                "its owner checks any remaining source requirements.",
                next_action=decision["next_action"],
            )
        if workspace is not None or access.decided_by != "DEFAULT":
            words["network_access"] = decision
    return words


def refusal_code(body: Mapping[str, Any]) -> str | None:
    """The code a refusal answers under.

    Its failure code or `refused` code, else a refused disposition (`REFUSED_NO_BOOK_TO_REVIEW`).

    Args:
        body: The owner's answer.

    Returns:
        The code, or None for an answer that names none.
    """
    for key in ("failure_code", "refused"):
        value = body.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("disposition", "status"):
        value = body.get(key)
        if isinstance(value, str) and value.startswith("REFUSED_"):
            return value
    return None


def worded_refusal(body: dict[str, Any], *, workspace: Path | None = None) -> dict[str, Any]:
    """A refusal as it leaves the Host, with words and a way on (OP4, V449).

    Its owner's words and way on, and, where the owner gave none, its code's own.

    Args:
        body: The owner's answer.
        workspace: The actual control whose effective network decision words a refusal.

    Returns:
        The refusal with what it lacked filled from the table; any other answer as it is.
    """
    code = refusal_code(body) if refused(body) else None
    if code is None:
        return body
    words = refusal_words(code, workspace=workspace)
    if code.partition(":")[0] in NETWORK_ACCESS_REFUSALS:
        source_access = body.get("source_network_access")
        source_ways = body.get("source_ways")
        official = source_ways.get("official") if isinstance(source_ways, Mapping) else None
        recovery = official.get("before") if isinstance(official, Mapping) else None
        current = words.get("network_access")
        if (
            code.partition(":")[0] == "evidence_review.workspace_network_not_allowed"
            and isinstance(source_access, Mapping)
            and source_access.get("network_allowed") is False
            and isinstance(current, Mapping)
            and current.get("network_allowed") is True
            and isinstance(recovery, str)
        ):
            # An opened control does not replace this Host's denied source transport.
            words.update(detail=recovery, next_action="READ_NETWORK_ACCESS")
        # Keep an owner's decoded provider needs while refreshing the actual permission.
        previous = body.get("detail")
        if isinstance(previous, str) and previous.startswith("The research update needs "):
            words["detail"] = (
                previous.partition(". ")[0] + ". " + str(words["detail"]).partition(". ")[2]
            )
        body = {
            **{key: value for key, value in body.items() if key != "network_access"},
            **words,
            "next_requests": {
                **{
                    name: request
                    for name, request in (body.get("next_requests") or {}).items()
                    if request.get("operation") != "NETWORK_ACCESS_SET"
                },
                "network": {"operation": "NETWORK_ACCESS"},
            },
        }
    # An owner that words a refusal under `message` (a bundle's, a contract's located fields)
    # has worded it: the envelope shows `detail`, so its words are that.
    said = body.get("message") if isinstance(body.get("message"), str) else None
    detail = body.get("detail") or said or words.get("detail")
    missing = {
        **({} if body.get("detail") or not detail else {"detail": detail}),
        **(
            {}
            if body.get("next_action") or body.get("next_requests") or not words
            else {"next_action": words["next_action"]}
        ),
    }
    return {**body, **missing} if missing else body


def _refusal_body_problem(body: dict[str, Any], *, label: str) -> str | None:
    """Check a refusal row for words and an actual route or declared manual instruction."""
    code = refusal_code(body) or body.get("status")
    lacks = [] if body.get("detail") else ["words"]
    action = body.get("next_action")
    catalog_action = refusal_words(str(code)).get("next_action")
    manual_route = isinstance(action, str) and bool(action) and action == catalog_action
    request_routes = [
        request
        for request in offered_requests(body).values()
        if _request_route_problem(request) is None
    ]
    if not manual_route and not request_routes:
        lacks.append(
            "a way on" if not action and not body.get("next_requests") else "a usable way on"
        )
    if not lacks:
        return None
    return f"{label} {code} lacks " + " and ".join(lacks)


def _request_route_problem(request: Mapping[str, Any]) -> str | None:
    """Whether a request is accepted and names the object its command acts on."""
    if problem := request_problem(request):
        return problem
    operation = str(request.get("operation"))
    positional = command_table()["positional"].get(operation)
    if isinstance(positional, str) and not request.get(positional):
        return f"{positional} is not named"
    return None


def _collection_item_refused(row: Mapping[str, Any]) -> bool:
    """An item is refused only when its own disposition says so.

    Execution failures and unavailable metadata may carry a `failure_code` while the record
    itself was read successfully. A collection item therefore needs an explicit refused status
    or disposition; top-level refusals keep the broader `refused()` semantics.
    """
    return any(
        isinstance(value, str) and (value == "REFUSED" or value.startswith("REFUSED_"))
        for value in (row.get("status"), row.get("disposition"))
    )


def refusal_problem(
    body: dict[str, Any],
    *,
    operation: str | None = None,
    collection_fields: tuple[str, ...] = (),
) -> str | None:
    """What a refusal lacks of OP4, including refused rows in collection answers.

    Args:
        body: The answer as it left the Host.
        operation: The operation whose answer is checked, when available.
        collection_fields: The answer's registered top-level record arrays or maps, supplied by its
            answer owner. This checker does not discover operations or answer-model fields.

    Returns:
        None for an answer without an unheld refusal; else the missing words or usable next step.
    """
    if refused(body):
        if any(
            isinstance(body.get(field), (list, dict)) and body[field] for field in collection_fields
        ):
            return f"{operation} refused the collection while returning item rows"
        return _refusal_body_problem(body, label="refusal")
    if operation is None:
        return None
    for field in collection_fields:
        rows = body.get(field)
        if not isinstance(rows, (list, dict)):
            continue
        items = rows.items() if isinstance(rows, dict) else enumerate(rows)
        for index, row in items:
            if not isinstance(row, dict) or not _collection_item_refused(row):
                continue
            problem = _refusal_body_problem(row, label=f"{operation}.{field}[{index}] refusal")
            if problem:
                return problem
    return None


def outcome_of(body: dict[str, Any]) -> Outcome:
    """The one table from an owner's answer to the CLI's outcome.

    A Task's state counts wherever the answer writes it, ``lifecycle`` or ``status``, over the
    whole lifecycle: work that waits, runs or awaits a decision is pending, and a blocked or
    cancelled Task, or a feature trial stopped on a failed step, is refused (V144, V138).

    Args:
        body: The full owner answer.

    Returns:
        The CLI outcome.
    """
    state = task_state(body)
    trial = body.get("state") if body.get("status") == "FEATURE_TRIAL" else None
    if refused(body) or state in _REFUSED_STATES or trial == "STOPPED":
        return "REFUSED"
    disposition = body.get("status", body.get("disposition"))
    if (
        body.get("wait_status")
        or (isinstance(disposition, str) and disposition in _PENDING_DISPOSITIONS)
        or state in _PENDING_STATES
        or trial == "RUNNING"
    ):
        return "PENDING"
    return "OK"


class ClientRefusal(NamedTuple):
    """CLI outcome and recovery guidance for a local client refusal."""

    outcome: Outcome
    next_action: str
    detail: str


_REOPEN = "START_OR_REOPEN_LOCAL_WEB"
_INSPECT = "RECONNECT_AND_INSPECT_SAME_TASK"
_FULL = "USE_FULL_RESPONSE_OR_EXACT_SAVED_EXPORT_FOR_THIS_ACTION"
_DRAFT = "EXPORT_FRESH_EXPERIMENT_DRAFT"
_BUNDLE = "CHOOSE_A_NEW_BUNDLE_DIRECTORY"
_DAMAGED = ClientRefusal(
    "REFUSED",
    "RESTORE_ANOTHER_GENERATION",
    "A stored object of this generation no longer matches its digest, so nothing of it "
    "is trusted; restore an earlier generation with --generation.",
)
_CLIENT_REFUSALS: Final[tuple[tuple[str, ClientRefusal], ...]] = (
    # First match wins: an exact code before the prefix that also covers it.
    (
        "local_client.saved_answer_invalid",
        ClientRefusal(
            "INVALID_INPUT",
            "READ_A_FULL_SAVED_OWNER_ANSWER",
            "This file is not a full saved owner answer in JSON or safe YAML. Read the file "
            "the original command saved with --output; its owner fields are at the root, "
            "not inside a printed CLI envelope. No Host request was sent.",
        ),
    ),
    (
        "local_client.saved_answer_incomplete",
        ClientRefusal(
            "INVALID_INPUT",
            "READ_A_FULL_SAVED_OWNER_ANSWER",
            "This file holds a compact or incomplete display. Read the full owner answer "
            "the original command saved with --output; a compact display cannot supply "
            "missing values or whole references. No Host request was sent.",
        ),
    ),
    (
        "local_client.saved_answer_format_unavailable:",
        ClientRefusal(
            "INVALID_INPUT",
            "READ_A_FULL_SAVED_OWNER_ANSWER",
            "This saved display format cannot be read as a full owner answer. Use the JSON "
            "or YAML file the original command saved with --output. An HTML report is a "
            "display, not that answer. No Host request was sent.",
        ),
    ),
    (
        "local_client.saved_answer_section_unknown:",
        ClientRefusal(
            "INVALID_INPUT",
            "READ_THE_LISTED_SAVED_SECTIONS",
            "The saved answer has no part at the path named after the colon. The sections "
            "field lists paths where reading stopped; --list-sections lists the root paths. "
            "Read one of those paths from the same file. No Host request was sent.",
        ),
    ),
    (
        "activity.cursor_invalid",
        ClientRefusal("INVALID_INPUT", **refusal_words("activity.cursor_invalid")),
    ),
    (
        "activity.observation_id_invalid",
        ClientRefusal("INVALID_INPUT", **refusal_words("activity.observation_id_invalid")),
    ),
    (
        "activity.query_selection_conflict",
        ClientRefusal("INVALID_INPUT", **refusal_words("activity.query_selection_conflict")),
    ),
    # `backup restore` is the client's own, with no Host (V328): its refusals too.
    (
        "local_client.restore_unavailable",
        ClientRefusal(
            "REFUSED",
            "RUN_THE_INSTALLED_ENTRY",
            "This client was started without the restore its entry composes; run "
            "`alphalattice backup restore --dir <new directory> --generation "
            "<verified generation hash> --workspace-id <workspace id> --root <backup root>` "
            "from the installed entry.",
        ),
    ),
    (
        "workspace_backup.workspace_id_",
        ClientRefusal(
            "INVALID_INPUT",
            "NAME_THE_WORKSPACE_ID",
            "A workspace's backups are found by its identity: its manifest cannot be read, "
            "so name it with --workspace-id, as a directory name.",
        ),
    ),
    # A model's scaffold and contract (EX): each refusal says the way on (OP12).
    (
        "local_client.sandbox_unavailable",
        ClientRefusal(
            "REFUSED",
            "RUN_THE_INSTALLED_ENTRY",
            "The sandbox is the entry's own: run `model sandbox <model-id>` through "
            "the installed command.",
        ),
    ),
    (
        "model_sandbox.host_serving",
        ClientRefusal(
            "REFUSED",
            "STOP_THE_HOST_THEN_SANDBOX",
            "A sandbox copies the workspace at rest: stop the Host that serves it, run the "
            "sandbox, and start the Host again.",
        ),
    ),
    (
        "model_sandbox.no_alpha_study",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_AN_ALPHA_STUDY_WITH_--file",
            "The workspace holds no published Alpha study to declare anew with the model; "
            "give an Alpha study's declaration with --file <study.yaml>.",
        ),
    ),
    (
        "model_sandbox.study_names_no_model",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_AN_ALPHA_MODEL_STUDY_WITH_--file",
            "The study names no model to put this one in place of (a lifecycle study declares "
            "a component's recipe); give an Alpha model-development study with --file, or none "
            "for the latest published one.",
        ),
    ),
    (
        "model_sandbox.",
        ClientRefusal(
            "REFUSED",
            "READ_THE_STUDY_OR_THE_COPY_AND_SANDBOX_AGAIN",
            "The sandbox's study or its copy refused, as the code names; `--keep` keeps the "
            "copy to read.",
        ),
    ),
    (
        "model_extension.editable_checkout_required",
        ClientRefusal(
            "REFUSED",
            "AUTHOR_THE_MODEL_IN_AN_EDITABLE_CHECKOUT",
            "Model scaffolding writes source and its contract test, so it needs an editable "
            "checkout. Run uv sync --locked there, then model scaffold --file <declaration>; "
            "the installed package can check and run the activated model but cannot edit itself.",
        ),
    ),
    (
        "model_extension.library_outside_lock:",
        ClientRefusal(
            "REFUSED",
            "ASK_A_PERSON_TO_APPROVE_THE_DEPENDENCY",
            "A library outside the locked environment is a new dependency: a person approves "
            "it first, and nothing is written until the lock holds it.",
        ),
    ),
    (
        "model_extension.entry_exists:",
        ClientRefusal(
            "INVALID_INPUT",
            "NAME_ANOTHER_MODEL_OR_EDIT_ITS_FILES",
            "A model of this id is installed or already scaffolded: give the declaration "
            "another model_id, or edit the files already written.",
        ),
    ),
    (
        "model_extension.not_found:",
        ClientRefusal(
            "INVALID_INPUT",
            "SCAFFOLD_THE_MODEL_FIRST",
            "No extension names this model; `model scaffold --file <file>` writes its files.",
        ),
    ),
    (
        "model_declaration.",
        ClientRefusal(
            "INVALID_INPUT",
            "CORRECT_THE_DECLARATION",
            "The declaration is not a model's: the code names the fields to correct after "
            "its colon; `model scaffold --save-declaration <file>` writes one to edit.",
        ),
    ),
    (
        "search_axes.",
        ClientRefusal(
            "INVALID_INPUT",
            "CORRECT_THE_DECLARATION",
            "The declaration's axes are malformed; the code names the axis to correct.",
        ),
    ),
    (
        "model_extension.",
        ClientRefusal(
            "REFUSED",
            "CORRECT_THE_MODEL_AND_CHECK_AGAIN",
            "The model's code or declaration refused; correct what the code names and run "
            "`model check <model-id>` again.",
        ),
    ),
    (
        "model_contract.",
        ClientRefusal(
            "REFUSED",
            "CORRECT_THE_MODEL_AND_CHECK_AGAIN",
            "The model fails its contract at the check the code names; correct its fit, its "
            "prediction or its declaration and run `model check <model-id>` again.",
        ),
    ),
    (
        "workspace_backup.restore_target_",
        ClientRefusal(
            "INVALID_INPUT",
            "CHOOSE_A_NEW_RESTORE_DIRECTORY",
            "A restore writes onto a new or empty directory, outside the backup root and "
            "the workspace; name another --dir.",
        ),
    ),
    (
        "workspace_backup.generation_not_found",
        ClientRefusal(
            "INVALID_INPUT",
            "NAME_A_KEPT_GENERATION",
            "No generation kept for this workspace directory has a hash beginning so, or none "
            "is kept there; `backup list` lists the kept generations, and a restore takes a "
            "hash as it shows it.",
        ),
    ),
    (
        "workspace_backup.generation_ambiguous",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_MORE_OF_THE_GENERATION_HASH",
            "Two or more kept generations begin with that value; give more of the hash, as "
            "`backup list --view full` shows it whole.",
        ),
    ),
    (
        "workspace_backup.root_inside_workspace",
        ClientRefusal(
            "INVALID_INPUT",
            "NAME_A_BACKUP_ROOT_OUTSIDE_THE_WORKSPACE",
            "A backup root inside the workspace would be lost with it; name one outside "
            "with --root.",
        ),
    ),
    ("workspace_backup.object_invalid", _DAMAGED),
    ("workspace_backup.table_not_held", _DAMAGED),
    (
        "local_client.service_not_running",
        ClientRefusal(
            "NO_HOST",
            _REOPEN,
            "No Host serves this workspace, so nothing was sent. Start one with `serve`, or "
            "open the Local Web.",
        ),
    ),
    (
        "local_client.connection_lost_task_may_still_run",
        ClientRefusal(
            "NO_HOST",
            _INSPECT,
            "The connection to the Host was lost while the request was in flight, so whether "
            "it ran is unknown. Inspect the same Task before sending work again.",
        ),
    ),
    (
        "local_client.connection_unreadable",
        ClientRefusal(
            "NO_HOST",
            _REOPEN,
            "This workspace's connection record cannot be read; starting its Host writes it.",
        ),
    ),
    (
        "local_client.workspace_mismatch",
        ClientRefusal(
            "NO_HOST",
            _REOPEN,
            "The connection record names another workspace, so no Host serves this one.",
        ),
    ),
    (
        "local_client.connection_path_escapes_workspace",
        ClientRefusal(
            "REFUSED",
            _REOPEN,
            "The connection record is not a plain file inside this workspace, so it is not used.",
        ),
    ),
    (
        "local_client.connection_owner_mismatch",
        ClientRefusal(
            "NO_HOST",
            "STOP_THE_HOST_THAT_WROTE_IT",
            "This workspace's connection record names another workspace's Host, so this Host "
            "does not replace it: stop that Host, or, when none runs, remove "
            "`runtime/local-research-connection.json` and start again.",
        ),
    ),
    (
        "local_client.http_refused:",
        ClientRefusal(
            "REFUSED",
            _INSPECT,
            "The Host answered with the HTTP status after the colon and no owner's answer, so "
            "whether the request ran is unknown. Inspect the same Task before sending work again.",
        ),
    ),
    (
        "local_client.compact_reference_requires_full_response:",
        ClientRefusal(
            "INVALID_INPUT",
            _FULL,
            "The compact view left out the part named after the colon, which this continues "
            "from: continue from a full response (--view full) or the exact --output file.",
        ),
    ),
    (
        "local_client.operation_document_has_no_next_request",
        ClientRefusal(
            "INVALID_INPUT",
            "USE_REQUEST_FILE_FOR_OPERATIONS_OR_RESPONSE_FROM_WITH_ACTION",
            "This file is a request, not an answer with next requests: send it with "
            "`request --file <file>`, or continue from an answer.",
        ),
    ),
    (
        "local_client.operation_has_no_editable_declaration:",
        ClientRefusal(
            "INVALID_INPUT",
            "SAVE_THE_DECLARATION_FROM_ITS_AUTHORING_READ",
            "The selected operation named after the colon reads execution results, not an "
            "editable declaration; use study export or the retained study preview, or feature "
            "controls or the Feature catalog readback, to save its declaration.",
        ),
    ),
    (
        "local_client.operation_has_no_declaration_file:",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_OPEN_REQUEST_FIELDS_WITH_CHOICES",
            "The selected operation named after the colon takes no declaration file; "
            "keep --from and --action, and give its open request fields with --choices.",
        ),
    ),
    (
        "local_client.request_file_or_selected_response_required",
        ClientRefusal(
            "INVALID_INPUT",
            "USE_REQUEST_FILE_FOR_OPERATIONS_OR_RESPONSE_FROM_WITH_ACTION",
            "`request` sends a request file (`--file`) or continues a saved answer (`--from`, "
            "with `--action` naming one of its next requests); `--action` needs `--from`.",
        ),
    ),
    (
        # Before the broad prefix below, which matched it first (V442).
        "local_client.next_request_needs_choice:",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_THE_CHOICE_WITH_THE_SAME_ACTION",
            "The next request you chose leaves a choice to you, named after the colon and in "
            "`next_templates`: keep the same action and give it with the command's flag, or in "
            "a file with `--choices <file>` beside `--from`.",
        ),
    ),
    (
        "local_client.next_request_",
        ClientRefusal(
            "INVALID_INPUT",
            "CHOOSE_ONE_OFFERED_NEXT_REQUEST",
            "The answer offers next requests, and the command names none of them it can "
            "send; the ones offered follow the code.",
        ),
    ),
    (
        "local_client.response_reference_",
        ClientRefusal(
            "INVALID_INPUT",
            _FULL,
            "The --from file does not hold the reference this command needs; continue from "
            "the full answer that named it.",
        ),
    ),
    (
        "local_client.bound_reference_override:",
        ClientRefusal(
            "INVALID_INPUT",
            "OMIT_THE_FLAG_THE_RESPONSE_ALREADY_BINDS",
            "The --from answer already binds this field, and a flag does not change it.",
        ),
    ),
    (
        "local_client.answer_names_two_books",
        ClientRefusal(
            "INVALID_INPUT",
            "CONTINUE_FROM_THE_BOOKS_OWN_ANSWER",
            "The answer's requests name more than one book, so this command cannot tell which "
            "one it continues: continue from that book's own answer (`study show <task-id> "
            "--output <output>`), or send a request file that names it.",
        ),
    ),
    (
        "local_client.alpha_comparison_reopen_unavailable",
        ClientRefusal(
            "REFUSED",
            "OPEN_THE_ALPHA_PAGE",
            "The comparison's answer does not name both studies and both candidates, so no "
            "link reopens it; the answer stands as returned, and the Alpha page chooses the pair.",
        ),
    ),
    (
        "local_client.plan_file_or_draft_document_required",
        ClientRefusal(
            "INVALID_INPUT",
            _DRAFT,
            "A plan is made from a draft (--from) or a plan file.",
        ),
    ),
    (
        "local_client.document_empty",
        ClientRefusal(
            "INVALID_INPUT",
            "WRITE_THE_DOCUMENT_INTO_THE_FILE",
            "The file is empty: nothing was written into it. Write the answer's JSON or the "
            "request's YAML into it, then send it again.",
        ),
    ),
    (
        "local_client.document_missing",
        ClientRefusal(
            "INVALID_INPUT",
            "CHECK_THE_FILE_PATH",
            "No file is at the path `document_location.file` names: send the path an answer "
            "named or your own --output wrote, as written (a relative path resolves from the "
            "shell's directory), then send it again.",
        ),
    ),
    (
        "local_client.field_given_twice:",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_EACH_FIELD_ONCE",
            "A field was given twice, by a flag, --file or --choices (the code names it): give "
            "each field once; --choices only fills what the saved answer's request leaves open.",
        ),
    ),
    (
        "local_client.short_reference_ambiguous",
        ClientRefusal(
            "INVALID_INPUT",
            "SEND_THE_WHOLE_VALUE",
            "The short reference in `short_reference.field` begins several values the Host has "
            "answered with; `short_reference.candidates` names them: send the whole value.",
        ),
    ),
    (
        "local_client.short_reference_unknown",
        ClientRefusal(
            "INVALID_INPUT",
            "SEND_THE_WHOLE_VALUE",
            "The short reference in `short_reference.field` begins no hash the Host has "
            "answered with: send the whole value from a saved answer (`--from`, or the file "
            "`--output` wrote).",
        ),
    ),
    (
        "local_client.document_too_large",
        ClientRefusal(
            "INVALID_INPUT",
            "SEND_A_SMALLER_DOCUMENT",
            "The document is larger than one request carries: `document_size` gives its bytes "
            "and the bound. Cite exact reads by their requests (a goal submission's "
            "`references`) instead of pasting whole answers into it, then send it again.",
        ),
    ),
    (
        # A document's own syntax, named at the line and column where reading stopped (AX15).
        "local_client.document_unreadable",
        ClientRefusal(
            "INVALID_INPUT",
            "CORRECT_THE_DOCUMENT_WHERE_READING_STOPPED",
            "The document does not read as YAML or JSON; reading stopped at the line and "
            "column named. Correct its syntax there (in YAML, quote a value holding a colon "
            "followed by a space) and send it again.",
        ),
    ),
    (
        "local_client.document_",
        ClientRefusal(
            "INVALID_INPUT",
            "CHECK_INPUT_DOCUMENT_AND_SCHEMA",
            "The request document, YAML or JSON, does not fit this operation; "
            "`schema show <command>` shows its fields.",
        ),
    ),
    (
        "local_client.operation_unknown",
        ClientRefusal(
            "INVALID_INPUT",
            "READ_CAPABILITIES",
            "No operation has that name; `nearest_operations` names the closest, and "
            "`operation list` lists them all.",
        ),
    ),
    (
        "local_client.request_invalid",
        ClientRefusal(
            "INVALID_INPUT",
            "READ_CAPABILITIES_SCHEMA",
            "The request does not match the operation's schema; `schema show <command>` "
            "shows it. The document is not echoed.",
        ),
    ),
    (
        "local_client.boolean_invalid",
        ClientRefusal(
            "INVALID_INPUT",
            "WRITE_TRUE_OR_FALSE",
            "A yes-or-no field takes true or false, and nothing else.",
        ),
    ),
    (
        "local_client.declaration_unavailable",
        ClientRefusal(
            "REFUSED",
            "SAVE_THE_DECLARATION_FROM_CONTROLS_OR_A_DRAFT",
            "This answer carries no editable declaration; `study controls`, `study "
            "draft` and `feature controls` answer with one, and --output saves this answer whole.",
        ),
    ),
    (
        "local_client.section_unknown:",
        ClientRefusal(
            "INVALID_INPUT",
            "ASK_FOR_A_PART_THE_ANSWER_HAS",
            "The answer has no part at the path named after the colon; `data` holds the whole "
            "answer, whose keys are its parts, and `schema show <command>` outlines a "
            "readback's.",
        ),
    ),
    (
        "local_client.usage_invalid",
        ClientRefusal(
            "INVALID_INPUT",
            "READ_HELP",
            "The command line does not parse; `--help` after the verb shows its arguments.",
        ),
    ),
    # A workspace left out is the session's binding's, or refused in words (V568).
    (
        "local_client.workspace_unbound",
        ClientRefusal(
            "INVALID_INPUT",
            "NAME_THE_WORKSPACE_OR_BIND_THIS_SESSION",
            "No workspace is named on the line, and no binding up from this folder gives this "
            "agent session one, so nothing was sent. Name it, as in alphalattice --workspace "
            "<dir> <object> <action> ...; or, inside an agent session, bind it once with "
            "`alphalattice --workspace <dir> session bind`, and its commands and its own "
            "specialists' may leave it out.",
        ),
    ),
    (
        "local_client.workspace_bound_to_another_session",
        ClientRefusal(
            "INVALID_INPUT",
            "NAME_THE_WORKSPACE",
            "No workspace is named on the line, and the nearest binding up from this folder "
            "names another agent session, whose workspace is not this session's to assume, so "
            "nothing was sent. Name it on the line, as in alphalattice --workspace <dir> "
            "<object> <action>.",
        ),
    ),
    (
        "local_client.session_unnamed",
        ClientRefusal(
            "INVALID_INPUT",
            "BIND_FROM_INSIDE_THE_AGENT_SESSION",
            "This shell names no single agent session, neither CLAUDE_CODE_SESSION_ID nor "
            "CODEX_THREAD_ID or both at once, so there is no session to bind and nothing was "
            "written. Bind from inside the agent's own session, or name the workspace on each "
            "line, as in alphalattice --workspace <dir> <object> <action>.",
        ),
    ),
    (
        "local_client.session_bind_refused:",
        ClientRefusal(
            "REFUSED",
            "RESOLVE_THE_NAMED_CAUSE_THEN_BIND",
            "The binding was refused for the reason its code names after the colon, and nothing "
            "was written: `native_bridge.hook_declaration_missing`, no folder up from here holds "
            "this host's agent declarations (configure the project, as the guide opens, then "
            "bind); `native_bridge.existing_configuration_differs`, the project is bound to "
            "another session or workspace: the session it names removes it with `alphalattice "
            "session unbind`, as the person can from their own shell, and then bind again; "
            "`native_bridge.workspace_missing`, the workspace named is not a folder; "
            "`native_bridge.files_unavailable`, the project's `.codex` folder cannot be written.",
        ),
    ),
    (
        "local_client.session_unbind_refused:",
        ClientRefusal(
            "REFUSED",
            "RESOLVE_THE_NAMED_CAUSE_THEN_UNBIND",
            "The binding was not removed, for the reason its code names after the colon: "
            "`native_bridge.session_mismatch`, the project's binding names another agent session, "
            "and a session removes only its own: the session it names unbinds, or the person "
            "does from their own shell, outside any agent session; "
            "`native_bridge.hook_declaration_missing`, no folder up from here holds the "
            "product's agent declarations, so there is no project here to unbind; "
            "`native_bridge.files_unavailable`, the binding file could not be removed.",
        ),
    ),
    (
        "local_client.output_write_refused",
        ClientRefusal(
            "REFUSED",
            "USE_STDOUT_OR_A_WRITABLE_UNUSED_OUTPUT_PATH",
            "The answer could not be written to --output; it is printed here instead.",
        ),
    ),
    (
        "local_client.output_format_unavailable:",
        ClientRefusal(
            "REFUSED",
            "USE_JSON_OR_A_FORMAT_PROVIDED_BY_THIS_OPERATION",
            "This answer has no such format; json and yaml always exist, html for a report.",
        ),
    ),
    (
        "local_client.dated_position_requires_json",
        ClientRefusal(
            "INVALID_INPUT",
            "USE_REPORT_SESSION_WITH_JSON_OR_EXPORT_THE_UNCHANGED_WHOLE_HTML_REPORT",
            "A dated position reads only as JSON; the HTML report is exported whole.",
        ),
    ),
    (
        "local_client.output_",
        ClientRefusal(
            "INVALID_INPUT",
            "CHOOSE_UNUSED_OUTPUT_PATH",
            "The --output path already exists, and an output never overwrites; choose an "
            "unused path.",
        ),
    ),
    (
        "local_client.bundle_directory_exists",
        ClientRefusal(
            "INVALID_INPUT",
            _BUNDLE,
            "The bundle directory already exists; a bundle is written to a new one.",
        ),
    ),
    (
        "local_client.bundle_",
        ClientRefusal(
            "REFUSED",
            _BUNDLE,
            "The bundle could not be written from the Host's answer.",
        ),
    ),
    (
        "local_client.continuation_bundle_",
        ClientRefusal(
            "INVALID_INPUT",
            _DRAFT,
            "The continuation bundle is not a current draft; export a fresh one.",
        ),
    ),
    (
        "local_client.plan_source_unsupported:",
        ClientRefusal(
            "INVALID_INPUT",
            "PLAN_FROM_CONTROLS_A_DRAFT_OR_A_PREVIEW",
            "`study plan --from <that answer>` takes a `study controls` answer, a `study "
            "draft` answer, a preview's answer or one offering a plan request; the answer "
            "given is none of these (its status is after the colon). Or plan directly with "
            "--binding and the declaration in --file.",
        ),
    ),
    (
        "local_client.request_timeout_outside_0_600",
        ClientRefusal(
            "INVALID_INPUT",
            "SET_REQUEST_TIMEOUT_BETWEEN_0_AND_600_SECONDS",
            "--request-timeout is seconds of HTTP waiting, above 0 and at most 600.",
        ),
    ),
    (
        "local_client.next_from_invalid",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_--next-from_WITH_--list-next",
            "--next-from N pages the listing --list-next prints, from its Nth entry (0 or "
            "more); give both, or neither.",
        ),
    ),
    (
        "local_client.max_wait_invalid",
        ClientRefusal(
            "INVALID_INPUT",
            "SET_MAX_WAIT_ABOVE_0_SECONDS_OR_OMIT_IT",
            "--max-wait is the waiter's only timer, in seconds above 0, for a command run under "
            "a cap; omit it to wait until the work ends or needs a decision.",
        ),
    ),
    (
        "local_client.each_stage_needs_task",
        ClientRefusal(
            "INVALID_INPUT",
            "GIVE_--each-stage_WITH_--task",
            "--each-stage returns as one Task verifies each stage; name the Task with --task.",
        ),
    ),
    (
        "local_client.wait_subject_required",
        ClientRefusal(
            "INVALID_INPUT",
            "NAME_ONE_TASK_OR_ONE_GOAL",
            "`activity wait` follows one --task or one --goal.",
        ),
    ),
    (
        "local_client.codex_queue_unavailable",
        ClientRefusal(
            "INVALID_INPUT",
            "WAIT_IN_THE_TURN_OR_SET_CODEX_THREAD_ID",
            "--notify codex-queue needs the codex command and CODEX_THREAD_ID; without them, "
            "wait inside the turn.",
        ),
    ),
    (
        "local_client.response_not_",
        ClientRefusal(
            "REFUSED",
            _INSPECT,
            "The Host's answer is not one this client can read, so the outcome is unknown. "
            "Inspect the same Task before sending work again.",
        ),
    ),
)
_OTHER = ClientRefusal(
    "REFUSED",
    _INSPECT,
    "The request was refused before an owner answered; inspect the same Task before sending "
    "work again.",
)


def client_refusal(code: str) -> ClientRefusal:
    """The outcome, next action and words of a refusal the client raised itself."""
    for prefix, refusal in _CLIENT_REFUSALS:
        # A prefix ends in `_`, `:` or `.`: a code's family, a code's subjects, an owner's codes.
        if code == prefix or (prefix.endswith(("_", ":", ".")) and code.startswith(prefix)):
            return refusal
    return _OTHER


def envelope(
    *,
    operation: str | None,
    outcome: Outcome,
    body: dict[str, Any] | None,
    elapsed_seconds: float,
    display: dict[str, Any] | None = None,
    status: str | None = None,
    failure_code: str | None = None,
    detail: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Wrap an owner answer in the CLI's stable envelope.

    Args:
        operation: The operation sent, if one was admitted.
        outcome: The CLI outcome category.
        body: The owner's full answer, when available.
        elapsed_seconds: The command's elapsed time.
        display: Optional compact projection to show under ``data``.
        status: Optional status override for a client-side answer.
        failure_code: Optional client-side failure code.
        detail: Optional client-side explanation.
        extra: Additional stable envelope fields.

    Returns:
        The answer with status, failure, next requests, and timing beside data.
    """
    owner = body or {}
    refusal = owner.get("refused")
    titles: dict[str, str] = {}
    if body is not None:
        from alphalattice.interface.local_application.labels import titles_in

        # What it installs, named by the label table beside the ids the answer keeps.
        titles = titles_in(body, chinese=ANSWER_LANGUAGE.get() == "zh")
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "outcome": outcome,
        "status": status or owner.get("status", owner.get("disposition")),
        "data": body if display is None else display,
        **({"titles": titles} if titles else {}),
        "failure_code": failure_code
        or owner.get("failure_code")
        or (refusal if isinstance(refusal, str) else None),
        "detail": worded(detail or owner.get("detail")),
        "next_requests": owner.get("next_requests"),
        "timing": {"elapsed_seconds": round(elapsed_seconds, 3)},
        # This agent session's CLI launches, help and schema included (V364).
        **(
            {"session_launches": launches}
            if (launches := SESSION_LAUNCHES.get()) is not None
            else {}
        ),
        **extra,
    }


@functools.cache
def _zh() -> dict[str, str]:
    """The Chinese table's entries; a context-prefixed key (`tp|Queued`) is a page's own."""
    text = _ZH_TABLE.read_text(encoding="utf-8")
    return {json.loads(key): json.loads(value) for key, value in _ZH_ENTRY.findall(text)}


def worded(detail: str | None) -> str | None:
    """An answer's detail in the command's language (`ANSWER_LANGUAGE`, U19).

    Args:
        detail: The owner's or the client's words, English.

    Returns:
        The table's Chinese under `--lang zh` where it words the sentence; the English
        otherwise, so a sentence the table lacks reads as the owner wrote it.
    """
    if detail is None or ANSWER_LANGUAGE.get() != "zh":
        return detail
    return _chinese(detail)


def _chinese(detail: str) -> str:
    """Read a sentence or subject part through the same Chinese keys (V608)."""
    table = _zh()
    if detail in table:
        return table[detail]
    # A filled sentence reads through its template's key, `{subject}` and all, as the page
    # reads it (U76): the door's table words ten sentences around a code's subject.
    for template, words in _zh_templates():
        found = template.fullmatch(detail)
        if found is not None:
            # Each slot in its turn, as the page fills them (V566: a second slot read English).
            for part in found.groups():
                subject = "\uff1b".join(_chinese(need) for need in part.split("; "))
                words = words.replace("{subject}", subject, 1)
            return words
    return detail


@functools.cache
def _zh_templates() -> tuple[tuple[re.Pattern[str], str], ...]:
    """The Chinese table's `{subject}` sentences, each as the pattern of its filled English,
    every slot a group of its own."""
    return tuple(
        (re.compile(re.escape(key).replace(re.escape("{subject}"), "(.+?)")), value)
        for key, value in _zh().items()
        if "{subject}" in key
    )


@functools.cache
def command_table() -> dict[str, Any]:
    """Read the generated CLI operation contract.

    Returns:
        Each command's operation, each operation's fields, and each field's
        types from ``operations.json``, held by the operation registry gate.
    """
    table: dict[str, Any] = json.loads(
        Path(__file__).with_name("operations.json").read_text(encoding="utf-8")
    )
    return table


DECLARATION_OPERATION_EXCEPTIONS: Final = {
    "EXPERIMENT_READBACK": "This reads a study Task's results, not its editable declaration; "
    "study export saves the study declaration, and study show --plan saves a retained preview.",
    "FEATURE_CATALOG_BUILD_READBACK": "This reads a Feature build Task's results, not an "
    "editable catalog; feature controls or feature show --plan reads the catalog declaration.",
}
"""Non-declaration routes of shared commands: refuse --save-declaration before sending."""


@functools.cache
def declaration_contracts() -> dict[str, Any]:
    """Read the registered declaration answers and their deliberate unavailable states.

    Returns:
        Each operation whose answer declares YAML or a template, with its answer contract.
    """
    rows: dict[str, Any] = json.loads(
        Path(__file__).with_name("answers.json").read_text(encoding="utf-8")
    )
    return {
        operation: row
        for operation, row in rows.items()
        if any(field["name"] in {"yaml", "template"} for field in row["fields"])
    }


def declaration_operations() -> frozenset[str]:
    """Name the operations whose commands can save an editable declaration.

    Returns:
        The operation set generated from the answer registry.
    """
    return frozenset(declaration_contracts())


def flag(field: str) -> str:
    """A request field's command-line flag, as the grammar declares it (`--task`, `--file`).

    Args:
        field: The request field name.

    Returns:
        The field's flag.
    """
    return "--" + str(command_table()["flags"].get(field, field.replace("_", "-")))


def choices(request: dict[str, object]) -> list[str]:
    """The required fields a next request leaves to its reader (V136).

    An owner offering a request that still needs a choice writes the field as None (an
    Alpha result's ``portfolio-draft`` needs its ``candidate_id``); an absent one is left too,
    and so is a field an object's model requires that the owner's part of it leaves out (a
    curation's ``experiment_curation.choices``, V373).

    Args:
        request: The next request the owner offered.

    Returns:
        Each required field the request does not fill, in the registry's order.
    """
    table = command_table()
    operation = str(request.get("operation"))
    contract = table["fields"].get(operation, {})
    left = [name for name in contract.get("required", []) if request.get(name) is None]
    # One of an either-or pair is required (a plan's declaration, as YAML or as a document):
    # a request giving none leaves it, named as the owner wrote it, else the first (V433).
    groups = table.get("alternatives", {}).get(operation, [])
    if groups and not any(all(request.get(name) is not None for name in g) for g in groups):
        offered = [name for group in groups for name in group if name in request]
        left += offered[:1] or list(groups[0][:1])
    # An object the owner fills in part leaves the rest of its model's required fields
    # (V373: a curation's `experiment_curation` carries its receipt, not the author's choices).
    for name, inner in table.get("inner_required", {}).items():
        value = request.get(name)
        if isinstance(value, dict):
            left += [f"{name}.{key}" for key in inner if value.get(key) is None]
    return left


_ITEM_IDENTIFIERS = (
    "entry_id",
    "task_id",
    "case_token",
    "review_publication_hash",
    "feature_trial_id",
)
"""What names a listed item's requests apart when two items offer one name: a history row by
its entry, since one Task can have two reviews (V528)."""
_OFFER_DEPTH = 5
"""How deep the collector looks for offered requests: `workspace show`'s are three parts down
(`intents.0.flows.factor`), and a deeper part offers none (V425)."""


def offered_requests(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every next request an answer offers, wherever it sits (V135, V425).

    The top level's by their names; a nested part's (each of the pending decisions, each flow
    of `workspace show`'s intents) by its name, or, where two offer one name, by the name and
    the part's identifier (``recovery:<task_id>``), or its place (``decisions.2``,
    ``intents.0.flows.factor``) when it has none. A request offered twice is listed once;
    two items that name theirs alike keep both, each by its place (V528).

    Args:
        body: The owner's answer.

    Returns:
        Each offered request by the name ``--action`` takes.
    """
    found: list[tuple[str, str | None, str | None, dict[str, Any]]] = []
    seen: set[str] = set()

    def walk(value: Any, place: str | None, depth: int) -> None:
        if isinstance(value, dict):
            offered = value.get("next_requests")
            if isinstance(offered, dict):
                named = next((str(value[f]) for f in _ITEM_IDENTIFIERS if value.get(f)), None)
                for name, request in offered.items():
                    key = json.dumps(request, sort_keys=True, default=str)
                    if isinstance(request, dict) and key not in seen:
                        seen.add(key)
                        found.append((name, named, place, request))
            parts = [(key, inner) for key, inner in value.items() if key != "next_requests"]
        elif isinstance(value, list):
            parts = [(str(index), inner) for index, inner in enumerate(value)]
        else:
            return
        if depth < _OFFER_DEPTH:
            for key, inner in parts:
                walk(inner, key if place is None else f"{place}.{key}", depth + 1)

    walk(body, None, 0)
    counts = Counter(name for name, _named, _place, _request in found)
    labels = [
        name if place is None or counts[name] == 1 else f"{name}:{named or place}"
        for name, named, place, _request in found
    ]
    # Two different requests under one label keep both, each by its place, never the first
    # alone: one Task's two reviews each offered their own export (V528).
    clashes = Counter(labels)
    requests: dict[str, dict[str, Any]] = {}
    for label, (name, _named, place, request) in zip(labels, found, strict=True):
        requests[f"{name}:{place}" if clashes[label] > 1 and place is not None else label] = request
    return requests


_STAND_IN: Final = re.compile(r"<[^<>]+>")
"""A value written in a choice's place (`<candidate_id>`), which no request takes."""
_KINDS: Final = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (dict, "object"),
    (list, "array"),
)
"""Each JSON kind by the Python type that reads as it, a boolean before the integer it is."""


def named_read(request: Mapping[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """The CLI's copy of a read's answer, naming what it read (OP15, V449, V454).

    The request it sent, every field the operation takes that it gave as a value (a target, a
    reference, a revision, a session, a page), or its operation alone when it gave none (V571),
    is kept under ``read_request``, so `--from` reads the same read again from the answer
    alone, its whole selection included, before any next request the answer offers (V460).
    The Host's answer stays its owner's, byte for byte: an export's record, a measured delivery
    and a self-hash are sealed by their owners, and a read answers alike at every entry.

    Args:
        request: The request's fields.
        body: The owner's answer.

    Returns:
        The answer with its read's request; a refusal or another operation's answer as it is.
    """
    table = command_table()
    operation = str(request.get("operation"))
    if operation not in table["reads"] or refused(body) or "read_request" in body:
        return body
    allowed = set(table["fields"][operation]["allowed"])
    given = {
        name: shown
        for name, value in sorted(request.items())
        if name in allowed and (shown := _scalar(value)) is not None
    }
    # A read that gave no field is kept too, as its operation alone: its defaults are its
    # selection, so `--from` reads its first page again, never the next one it offers (V571).
    return {**body, "read_request": {"operation": operation, **given}}


def _scalar(value: object) -> object:
    """A request field's value as JSON writes it, when it is a value or a list of values: a
    text, an id, a day, a number or a truth, or a list of them, the ids an activity list
    watches among them (V556); None for an absent one, an empty list or a document."""
    if isinstance(value, list | tuple):
        items = [_scalar(item) for item in value]
        return items if items and all(item is not None for item in items) else None
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, UUID | date):
        return value.isoformat() if isinstance(value, date) else str(value)
    return value if isinstance(value, str) and value else None


def exit_problem(body: dict[str, Any]) -> str | None:
    """What an answer that completes or reuses work lacks of its exit (OP18, V449).

    A reuse, an admission or a resubmission names what holds its work: its Task, or the result
    or record it is, by an id or a hash at its top (`publication_task_id`, `observation_id`,
    `binding_hash`), or a request it offers that names one; the goal a request counted toward
    holds none of it.

    Args:
        body: The owner's answer.

    Returns:
        None when it names one; else the state that names none.
    """
    state = str(body.get("disposition") or body.get("status") or "")
    if not (state.startswith("REUSED") or state in {"ADMITTED", "RECOVERY_RESUBMITTED"}):
        return None

    def names(value: Mapping[str, Any]) -> bool:
        return any(
            isinstance(found, str)
            and found
            and key != "attributed_goal_id"
            and key.endswith(("_id", "_hash"))
            for key, found in value.items()
        )

    if names(body) or any(names(request) for request in offered_requests(body).values()):
        return None
    return f"{state} names no Task, result or record that holds its work"


def request_problem(request: Mapping[str, object]) -> str | None:
    """What keeps an offered request from being one the Host accepts as it stands (V449).

    An offered request names an operation the Host answers and only fields that operation takes,
    each of a kind the grammar declares; a field left to its reader is None or absent
    (`choices`), never a stand-in written in its place (`<candidate_id>`).

    Args:
        request: The next request an answer offers.

    Returns:
        None when the Host accepts it as offered; else what is wrong, by field.
    """
    table = command_table()
    operation = str(request.get("operation"))
    contract = table["fields"].get(operation)
    if contract is None:
        return f"operation {request.get('operation')!r} is none the Host answers"
    problems: list[str] = []
    for name, value in request.items():
        if name == "operation" or value is None:
            continue
        if name not in contract["allowed"]:
            problems.append(f"{name} is no field of {operation}")
            continue
        kind = next(word for kind_type, word in _KINDS if isinstance(value, kind_type))
        declared = set(table["types"].get(name, ()))
        if declared and kind not in declared and not (kind == "integer" and "number" in declared):
            problems.append(f"{name} is {kind}, not {' or '.join(sorted(declared))}")
        pending: list[object] = [value]
        while pending:
            inner = pending.pop()
            if isinstance(inner, str) and _STAND_IN.fullmatch(inner):
                problems.append(f"{name} holds the stand-in {inner}")
                break
            if isinstance(inner, dict):
                pending.extend(inner.values())
            elif isinstance(inner, list):
                pending.extend(inner)
    return "; ".join(problems) or None


def entry(workspace: Path, options: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The client's own root command for one workspace, as a printed command starts (V123).

    The command is ``alphalattice`` on either leg. It names the workspace, except the one the
    agent session is bound to (`BOUND_WORKSPACE`), which the session's next command finds from
    its binding (V568); ``options`` are the caller's other global options, which its next
    command keeps (V406).

    Args:
        workspace: The workspace the command acts on.
        options: The global options the caller gave besides the workspace.

    Returns:
        The command, the workspace flag with its value unless bound, and the options.
    """
    resolved = Path(workspace).resolve()
    if BOUND_WORKSPACE.get() == resolved:
        return (*command_prefix(), *options)
    return (*command_prefix(), "--workspace", str(resolved), *options)


def shell() -> str:
    """The shell printed commands are quoted for (V131).

    ``ALPHALATTICE_SHELL`` (``powershell`` or ``posix``) when set; PowerShell on Windows
    outside a POSIX shell (Git Bash names itself in ``MSYSTEM``); POSIX otherwise.

    Returns:
        ``powershell`` or ``posix``.
    """
    chosen = os.environ.get("ALPHALATTICE_SHELL", "").strip().lower()
    if chosen in {"powershell", "posix"}:
        return chosen
    if os.name == "nt" and "MSYSTEM" not in os.environ:
        return "powershell"
    return "posix"


def command(
    request: dict[str, object],
    *,
    prefix: tuple[str, ...] = ("alphalattice",),
    quoting: str | None = None,
) -> str | None:
    """A next request as the command that sends it; None for an operation the CLI lacks.

    A request that leaves a choice (`choices`) is a template: each such field is written
    ``--candidate <candidate_id>``, unquoted, so no shell runs it as it stands (V136); a
    choice inside an object writes the object's flag so, with what the owner filled and the
    fields to give (V373).

    Args:
        request: The next request the owner offered.
        prefix: What the command starts with, the checkout's entry (`entry`) when known.
        quoting: ``powershell`` or ``posix``; the current `shell` when None.

    Returns:
        The command line, every part quoted for the shell, or None.
    """
    operation = str(request.get("operation"))
    table = command_table()
    names = {op: name for name, ops in table["commands"].items() for op in ops}
    if operation not in names:
        return None
    left = choices(request)
    # A choice inside an object is given inside that object, so its flag takes the object
    # whole, the part the owner filled shown and the fields to give named (V373).
    inside = {name.split(".", 1)[0] for name in left if "." in name}
    shown = table["positional"].get(operation)
    parts = [*prefix, *names[operation].split(" ")]
    last: list[str] = []
    if shown and isinstance(request.get(shown), str) and shown not in left:
        # A positional the parser would read as an option comes last, after `--` (V449).
        given = _given(str(request[shown]))
        if given.startswith("-") and not left:
            last = ["--", given]
        else:
            parts.append(given)
    for key, value in request.items():
        if key in {"operation", shown} or value is None or key in inside:
            continue
        if key in table["primary"] and isinstance(value, str):
            # A declaration's text is continued from its answer (`--from`), not printed (V401).
            return None
        given = _given(value) if isinstance(value, str) else _json(value)
        # A value the parser would read as an option is joined to its flag (V449).
        parts += [f"{flag(key)}={given}"] if given.startswith("-") else [flag(key), given]
    parts += last
    tail = "".join(
        f" <{name}>" if name == shown else f" {flag(name)} <{name}>"
        for name in left
        if "." not in name
    ) + "".join(
        f" {flag(key)} <{key}: {_json(request[key])} with "
        + ", ".join(name.split(".", 1)[1] for name in left if name.startswith(key + "."))
        + ">"
        for key in sorted(inside)
    )
    chosen = quoting or shell()
    if chosen == "powershell" and not left and any('"' in str(p) for p in parts[len(prefix) :]):
        # Windows PowerShell 5.1 hands a native program an argument's double quotes bare
        # (`--file '{"a":1}'` arrives as `{a:1}`): such a request goes whole through stdin, as
        # ASCII JSON, which every PowerShell pipes as written (V449).
        whole = {key: value for key, value in request.items() if value is not None}
        piped = json.dumps(whole, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        return f"{_powershell(piped)} | {join([*prefix, 'request', '--file', '-'], chosen)}"
    return join(parts, chosen) + tail


def join(parts: list[str], quoting: str) -> str:
    """One command line from its parts, each quoted as the shell reads it back (V131).

    PowerShell takes single quotes (a quote doubled inside), quotes a leading ``@``, which it
    would read as splatting, and runs a quoted program through ``&``; POSIX takes `shlex`.

    Args:
        parts: The program and its arguments.
        quoting: ``powershell`` or ``posix``.

    Returns:
        The command line.
    """
    if quoting == "powershell":
        quoted = [_powershell(str(part)) for part in parts]
        return ("& " if quoted and quoted[0].startswith("'") else "") + " ".join(quoted)
    return " ".join(shlex.quote(str(part)) for part in parts)


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _given(value: str) -> str:
    """A text value as the CLI reads it back: `@path` reads a file, so a leading `@` is
    doubled, the CLI's own spelling of a literal one (V149, V449)."""
    return "@" + value if value.startswith("@") else value


def _powershell(part: str) -> str:
    safe = bool(part) and part[0] != "@" and all(c.isalnum() or c in "-_.:/\\=" for c in part)
    return part if safe else "'" + part.replace("'", "''") + "'"


__all__ = [
    "ANSWER_LANGUAGE",
    "CLIENT_HEADER",
    "EXIT_CODES",
    "INSTANCE_HEADER",
    "MAXIMUM_REQUEST_BODY_BYTES",
    "SCHEMA_VERSION",
    "SESSION_LAUNCHES",
    "WORKSPACE_HEADER",
    "ClientRefusal",
    "Outcome",
    "choices",
    "client_refusal",
    "command",
    "command_table",
    "count_launch",
    "entry",
    "envelope",
    "exit_problem",
    "flag",
    "join",
    "named_read",
    "offered_requests",
    "outcome_of",
    "refusal_code",
    "refusal_problem",
    "refusal_words",
    "refused",
    "request_problem",
    "shell",
    "worded",
    "worded_refusal",
]
