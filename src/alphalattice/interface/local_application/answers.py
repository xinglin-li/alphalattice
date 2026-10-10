"""Every operation's answer, typed (binding plan C2, CLI-16; V410).

Each model names the fields an owner's answer always carries, with their types, so an Agent or a
script can read an answer without guessing and `alphalattice schema show` can print it. An answer
may carry more (an owner adds a field before its model does); a model is never a second owner of
the answer, only its published reading, and every test that runs an operation holds its owner to
it (`answer_problem`, at the operations' one door). The most-used reads are written out here with
their parts; every other operation's answer is a row of `answers.json`: each field, its kind,
whether every answer carries it, and what it means.
"""

from __future__ import annotations

import functools
import json
import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


class _Answer(BaseModel):  # type: ignore[misc]
    # Each field's meaning is its docstring, which `alphalattice schema show` prints (SC3).
    model_config = ConfigDict(extra="allow", frozen=True, use_attribute_docstrings=True)


class TaskStatusAnswer(_Answer):
    """One Task as Task Control projects it: its kind, lifecycle and verified stages."""

    task_id: str
    """The Task, by its id."""
    task_kind: str
    """What the Task does, by its registered kind."""
    lifecycle: str
    """Where the Task stands: `QUEUED`, `RUNNING`, `DEFERRED`, `REVIEW_PENDING`,
    `CANCEL_REQUESTED`, `CANCELLED`, `SUCCEEDED`, `BLOCKED` or `RECOVERY_REQUIRED`."""
    goal_summary: str | None
    """What the Task is for, in its goal's words."""
    current_stage: str | None
    """The stage running or next, by its id; none when no stage waits."""
    verified_stage_count: int
    """How many of its stages are done and verified."""
    total_stage_count: int
    """How many stages its plan holds."""
    running_since: str | None
    """When it started, in ISO 8601; none before it starts."""
    last_activity_at: str
    """When it last moved, in ISO 8601."""
    cancel_available: bool
    """true while it can still be cancelled (it has not ended)."""
    cancel_pending: bool
    """true when a cancel was asked for and has not yet stopped it."""
    queued_next_task_id: str | None
    """The Task the queue starts next, when that is another Task."""
    latest_failure_code: str | None
    """Why it last failed or stopped, as its owner's code."""
    failure_code: str | None = None
    """A BLOCKED Task's reason, its latest failure code, as a refusal names one: a read of a
    blocked Task answers REFUSED."""
    detail: str | None = None
    """A BLOCKED Task's reason in words; a DEFERRED one's, its provider's, with what stands."""
    retry_after_at: str | None = None
    """A DEFERRED Task's retry time, in ISO 8601: its plan sent again before it is refused;
    none when its owner names no time."""
    failure_cause: dict[str, Any] | None = None
    """What the stopped stage's owner saw beside its code, where it said: the
    `exception_type`, its message (`detail`, at most 400 characters), the `step`, the `unit` in
    hand and its `first_session` and `last_session`; a `MemoryError` there is the machine, not
    the work."""
    next_requests: dict[str, Any] | None = None
    """Where a BLOCKED Task, or one waiting on its recovery, goes on: `recovery`, its recovery
    view (`recovery show <task>`), and its owner's own requests where the Host has them. A
    DEFERRED one's: `resume`, its plan sent again once `retry_after_at` has passed, and `read`,
    its owner's readback."""
    timing: dict[str, Any] | None = None
    """Task Control's recorded queued/running spans and stage timings at this read."""


class TaskArtifact(_Answer):
    """The artifact a succeeded Task's owner opened when the listing answered."""

    artifact_kind: str
    """What it is: `PortfolioResearchResult` or `ResearchExecutionEvidence`."""
    artifact_hash: str | None
    """Its content hash; none when it is unavailable."""
    availability: str
    """`AVAILABLE` when its owner opened it; `UNAVAILABLE` when it could not."""
    failure_code: str | None
    """Why it is unavailable, as a code."""


class TaskRow(TaskStatusAnswer):
    """A Task in the listing, with why it cannot resume and its running span."""

    stop_next: str | None = None
    """The stop owner's explanation of what the offered continuation keeps and redoes."""

    resume_refusal: str | None
    """Why RECOVER would be refused now, as a code; none when it would be taken."""
    running_seconds: float | None
    """How long it has run, in seconds."""
    submitted_by: dict[str, Any] | None = None
    """The agent session that submitted it (`vendor`, `session`, `goal_id`); in a listing that
    names an agent session."""
    final_state: str | None = None
    """`SUCCEEDED`, `BLOCKED` or `CANCELLED` once it ended; none while it runs, waits or may
    resume. In a listing that names an agent session."""
    artifact: TaskArtifact | None = None
    """The artifact a succeeded Task's owner opened when this answered; none when its kind
    publishes none. In a listing that names an agent session."""


class TasksAnswer(_Answer):
    """The workspace's Tasks, active first; or one agent session's, newest first."""

    tasks: list[TaskRow]
    """Every Task of the workspace, active first; or a page of 50 of one agent session's."""
    next_cursor: str | None = None
    """Where a session's listing reads on, as `history_cursor`; none on its last page."""
    refusals: list[dict[str, Any]] | None = None
    """Named Tasks whose canonical records could not be projected, with words and workspace or
    backup reads. Their lifecycle, progress and artifact availability are unknown."""


class RecoveryAction(_Answer):
    """One action an owner permits on a stopped Task, and what it would do."""

    action: str
    """The action, as the recovery view names it."""
    operation: str
    """The operation that performs it."""
    available: bool
    """true when the owner would take it now."""
    requires_confirmation: bool
    """true when a person confirms it first."""
    expected_effect: str
    """What it would do, in words."""
    reason: str
    """Why it is offered, or why not now."""


class TaskRecoveryAnswer(_Answer):
    """What stopped a Task, what stays verified and what its owners permit, at one moment."""

    task_id: str
    """The Task, by its id."""
    task_kind: str
    """What the Task does, by its registered kind."""
    lifecycle: str
    """Where the Task stands: `QUEUED`, `RUNNING`, `DEFERRED`, `REVIEW_PENDING`,
    `CANCEL_REQUESTED`, `CANCELLED`, `SUCCEEDED`, `BLOCKED` or `RECOVERY_REQUIRED`."""
    task_record_hash: str
    """The Task's version: send it back as `expected_task_hash`."""
    view_hash: str
    """This view's own hash, over everything it shows."""
    observed_at: str
    """When the view was read, in ISO 8601."""
    actions: list[RecoveryAction]
    """Each action the owner permits on the Task, and what it would do."""
    stages: list[dict[str, Any]]
    """Each stage with its lifecycle, as Task Control holds it."""
    resume_refusal: str | None
    """Why RECOVER would be refused now, as a code; none when it would be taken."""
    status: TaskStatusAnswer
    """The Task as its status read shows it."""
    subjects: list[dict[str, str]] | None = None
    """Every Evidence admission's exact book selector, admission time and selection provenance.
    A Task can serve multiple books; without an address selector the reader must choose."""
    subject_context: dict[str, str] | None = None
    """The owner's recorded subject name, market date or Evidence cutoff, where retained."""
    current_scope: dict[str, Any] | None = None
    """The coverage run's admitted groups/issuers, nothing-filed and carried counts,
    packing rule and failed_unit_ids from its unit receipts, independent of its book."""
    subject_refusal: dict[str, Any] | None = None
    """The owner's reason and way on when an admitted subject cannot be read."""
    next_requests: dict[str, Any] | None = None
    """Each permitted action that takes a version (`recover`, `cancel`), bound to the Task and
    the version this view read, and a blocked Task's owner's own requests, as its status offers
    them; `subject` names the exact book only when the Task has one unique admitted subject.
    None when there are neither."""


class ExperimentRow(_Answer):
    """One saved study: its kind, input and Program."""

    task_id: str
    """The study, by its Task."""
    lifecycle: str
    """Where the study's Task stands."""
    input_id: str
    """The research input it reads, by its id."""
    input_binding_hash: str
    """The research input's version it reads."""
    kind: str
    """The study kind, as `factor.screening-development` or `risk.covariance-development`."""
    program_hash: str
    """The sealed Program it ran, by its hash."""
    research_lane: Any
    """`EXPLORATION` when it ran on a sample of its input's names, else `PROMOTION`."""
    sessions: dict[str, Any]
    """The sessions it covers: its first and last."""
    factor_ids: list[str]
    """The factors it names."""


class ExperimentsAnswer(_Answer):
    """Every saved study of the workspace."""

    status: str
    """The answer's status."""
    experiments: list[ExperimentRow]
    """Every study of the workspace."""
    refusals: list[dict[str, Any]] | None = None
    """Studies whose saved plans could not be read, each with its Task, code and way on."""


class ExperimentReadbackAnswer(_Answer):
    """A study's verified answer; `verification_basis` says how this read proved it."""

    status: str
    """The answer's status."""
    task_id: str
    """The study, by its Task."""
    verification_basis: str | None = None
    """How this read was proved: `FULL`, or `FILES_UNCHANGED ...` naming what it checked."""
    document: dict[str, Any] | None = None
    """The study's normalized declaration."""
    next_requests: dict[str, Any] | None = None
    """The requests this study offers next, each ready to send."""
    standing: dict[str, Any] | None = None
    """What the study can claim, one standing from its owners' marks: `comparison`,
    `execution`, `contract`, `evidence` and `activation`, the owner's code behind each held
    value in `reasons`, and one statement for each in `statements`."""


class Decision(_Answer):
    """One decision that waits on a person, with the requests that take it."""

    kind: str
    """What waits for a decision."""
    detail: str
    """What the decision is about, in words."""
    next_requests: dict[str, Any]
    """The requests that decide it, each ready to send."""
    waits_on: str
    """`PERSON` for a step only a person may take; `AGENT` for one the agent takes and tells
    the person, including advice it reports."""


class PendingDecisionsAnswer(_Answer):
    """What waits on a person, and what the agent carries, in one read."""

    status: str
    """The answer's status."""
    decisions: list[Decision]
    """Each decision that waits, with the requests that take it."""
    counts: dict[str, int]
    """How many decisions wait, by kind."""
    detail: str
    """How many decisions wait on a person, in one line; the agent's are not counted."""


class RefusalCount(_Answer):
    """How often one operation was refused with one code, by one caller kind, in one day."""

    day: str
    """The day, as YYYY-MM-DD."""
    operation: str
    """The operation refused."""
    failure_code: str
    """The refusal's code."""
    caller: str
    """Who asked: a person, an Agent or external automation."""
    vendor: str | None
    """The agent's vendor, where the request named one."""
    count: int
    """How many times that day."""
    first_at: str
    """The first, in ISO 8601."""
    last_at: str
    """The last, in ISO 8601."""


class ActivityRefusalsAnswer(_Answer):
    """Where requests failed: the refusals counted over the last days, the most frequent first."""

    status: str
    """The answer's status."""
    since_day: str
    """The first day counted, as YYYY-MM-DD."""
    days: int
    """How many days are counted."""
    refusals: list[RefusalCount]
    """The refusals by day, operation, code and caller."""
    total: int
    """How many refusals in all."""


class ResultRow(_Answer):
    """One published Portfolio result."""

    result_hash: str
    """The numeric result, by its hash."""
    report_hash: str
    """Its report, by its hash."""
    program_hash: str
    """The Program it ran, by its hash."""
    task_id: str
    """The first producer in a collection; with a Task selector, that exact producer or reuser."""
    completed_at: str
    """When that Task completed, in ISO 8601."""


class ResultsAnswer(_Answer):
    """The workspace's published Portfolio results."""

    results: list[ResultRow]
    """Every Portfolio result, newest first."""
    refusals: list[dict[str, Any]] | None = None
    """Unreadable publication records, each named with its code, words and recovery route."""


class UpgradeOverviewAnswer(_Answer):
    """What an upgrade touched: each saved study, review and waiting Task, with its standing."""

    status: str
    """The answer's status."""
    installed: dict[str, Any]
    """The installed identities, by role."""
    acknowledged: dict[str, Any] | None
    """The identity set last acknowledged, if any."""
    moved: list[str]
    """The roles whose identity moved since it."""
    show: bool
    """true when something moved and that set is not yet acknowledged."""
    counts: dict[str, dict[str, int]]
    """How many studies, reviews and Tasks stand where, by kind."""
    studies: list[dict[str, Any]]
    """Each saved study with its standing under the installed code."""
    reviews: list[dict[str, Any]]
    """Each published review with its standing."""
    tasks: list[dict[str, Any]]
    """Each waiting Task with its standing."""
    next_requests: dict[str, Any]
    """The requests the overview offers, acknowledging it among them."""


class ResearchInputRow(_Answer):
    """One research input and its captured versions."""

    input_id: str
    """The research input, by its id."""
    versions: list[dict[str, Any]]
    """Its captured versions, each by its binding."""


class ResearchInputsAnswer(_Answer):
    """The workspace's research inputs, each with its versions."""

    status: str
    """The answer's status."""
    inputs: list[ResearchInputRow]
    """Every research input, each with its versions."""
    refusals: list[dict[str, Any]] | None = None
    """Unreadable publication or lineage records, each with its recorded identity, REFUSED
    status, code, words and recovery requests; no input identity is guessed from unreadable JSON."""


class EvidenceBookSummary(_Answer):
    """An exact Book's discovery metadata; current evidence is checked only when opened."""

    state: Literal["NOT_READ"]
    """Current Evidence and CRO standing has not been read."""
    verification: str
    """Metadata only; the selected read verifies the book and its descendants."""
    next_requests: dict[str, dict[str, Any]]
    """The full review read, with every field of the exact book selector."""


class HistoryEntry(_Answer):
    """One entry of the research history: what was recorded, and for which input."""

    entry_id: str
    """The entry, by its id."""
    kind: str
    """What was recorded."""
    status: str
    """Its standing."""
    recorded_at: str
    """When it was recorded, in ISO 8601."""
    task_id: str | None
    """The Task it records, if any."""
    input_id: str | None
    """The research input it is for, if any."""
    input_binding_hash: str | None
    """That input's version."""
    failure_code: str | None
    """Why it failed, as its owner's code, if it did."""
    book_summary: EvidenceBookSummary | None = None
    """Cheap Evidence-owner discovery for an entry naming a Book, without current standing."""
    next_requests: dict[str, dict[str, Any]] = {}
    """What a reader of this entry sends next, bound to it: `task`; for an entry naming a book,
    its `review`, `evidence_preview`, `dossier` and `cro_bundle`; for a CRO review, `export`."""


class ResearchHistoryAnswer(_Answer):
    """A page of the research history, newest first, and what could not be read."""

    status: str
    """The answer's status."""
    entries: list[HistoryEntry]
    """A page of entries, newest first."""
    blocked_entries: list[dict[str, Any]]
    """What could not be read, each with its recorded identity, REFUSED status, code, words and
    usable next requests; no missing publication is inferred from an unreadable record."""
    next_cursor: str | None
    """Where the next page reads on; none on the last page."""
    next_requests: dict[str, dict[str, Any]] = {}
    """The named entry's requests when one is selected, and `next_page` while more remain."""
    verification: str
    """What this read checked: metadata only; reading an entry back verifies it and what it
    names."""


class GuardianEntry(_Answer):
    """One unfinished Task as Guanyin sees it: its liveness, progress, work kept and next steps."""

    task_id: str
    """The Task, by its id."""
    task_kind: str
    """What the Task does, by its registered kind."""
    lifecycle: str
    """Where the Task stands: `QUEUED`, `RUNNING`, `DEFERRED`, `REVIEW_PENDING`,
    `CANCEL_REQUESTED`, `CANCELLED`, `SUCCEEDED`, `BLOCKED` or `RECOVERY_REQUIRED`."""
    health: dict[str, Any]
    """Its health as Guanyin judges it: a status and why."""
    liveness: dict[str, Any]
    """What its heartbeats say, their sources named."""
    progress: dict[str, Any]
    """Its stages verified, of how many, and the current one."""
    preserved_stages: list[str]
    """The stages done and verified, kept for a resume."""
    incidents: list[dict[str, Any]]
    """Its incidents, as its recovery view lists them."""
    actions: list[dict[str, Any]]
    """The actions its owners permit now."""
    next_requests: dict[str, Any]
    """The request that reads its recovery view."""


class TaskGuardianAnswer(_Answer):
    """Every Task not finished, read once for the PM; each names its recovery view."""

    status: str
    """The answer's status."""
    guardian_mode: str
    """How Guanyin acts: `G0_READ_ONLY`, reading and never repairing."""
    tasks: list[GuardianEntry]
    """Every Task not finished."""
    counts: dict[str, int]
    """How many Tasks, by health status."""
    detail: str
    """How many Tasks are not finished, in words."""


_DECLARED: Final = Path(__file__).with_name("answers.json")
"""Every operation's answer the models above leave out, a row each (V410)."""
_KINDS: Final[dict[str, Any]] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
    "null": None,
}


def _declared(operation: str, row: dict[str, Any], purpose: str) -> type[BaseModel]:
    """One operation's answer model from its row (V410): each field's kind (`string|null`),
    whether every answer carries it, and its meaning, which `schema show` prints."""
    fields: dict[str, Any] = {}
    for field in row["fields"]:
        name = str(field["name"])
        annotation = functools.reduce(
            operator.or_, (_KINDS[kind] for kind in str(field["kind"]).split("|"))
        )
        reserved = name in dir(BaseModel) or name.startswith("_")
        fields[name + ("_" if reserved else "")] = (
            annotation,
            Field(
                ... if field["required"] else None,
                description=str(field["meaning"]),
                **({"alias": name} if reserved else {}),
            ),
        )
    model: type[BaseModel] = create_model(
        "".join(part.capitalize() for part in operation.split("_")) + "Answer",
        __base__=_Answer,
        __doc__=purpose,
        **fields,
    )
    return model


def _declared_answers() -> dict[str, type[BaseModel]]:
    """Each row of `answers.json` as its operation's model, its purpose the registry's; a read's
    model names, optional, the request the CLI's copy of its answer keeps (`named_read`, OP15,
    V454)."""
    from .operations import table

    contract = table()
    rows: dict[str, dict[str, Any]] = json.loads(_DECLARED.read_text(encoding="utf-8"))
    for operation, row in rows.items():
        if operation not in contract["reads"]:
            continue
        if "read_request" in {str(field["name"]) for field in row["fields"]}:
            continue
        row["fields"] = [
            *row["fields"],
            {
                "name": "read_request",
                "kind": "object|null",
                "required": False,
                "meaning": "The request this read sent, its whole selection, which the CLI's "
                "copy of the answer keeps so `--from` reads the same read again.",
            },
        ]
    return {
        operation: _declared(operation, row, contract["purposes"][operation])
        for operation, row in rows.items()
    }


ANSWERS: Final[dict[str, type[BaseModel]]] = {
    "STATUS": TaskStatusAnswer,
    "TASKS": TasksAnswer,
    "TASK_RECOVERY": TaskRecoveryAnswer,
    "TASK_GUARDIAN": TaskGuardianAnswer,
    "EXPERIMENTS": ExperimentsAnswer,
    "EXPERIMENT_READBACK": ExperimentReadbackAnswer,
    "PENDING_DECISIONS": PendingDecisionsAnswer,
    "ACTIVITY_REFUSALS": ActivityRefusalsAnswer,
    "RESULTS": ResultsAnswer,
    "RESEARCH_INPUTS": ResearchInputsAnswer,
    "RESEARCH_HISTORY": ResearchHistoryAnswer,
    "UPGRADE_OVERVIEW": UpgradeOverviewAnswer,
    **_declared_answers(),
}
"""Each operation and the model of its answer: every operation the Host answers (V410)."""


def collection_answer_fields(operation: str) -> tuple[str, ...]:
    """The declared record arrays or keyed Task map of a collection read (OP4, TE12).

    The command table owns which operations list records. Their published answer models own
    the row fields, including optional refused-item arrays. Activity's `tasks` is the one keyed
    record map. Detail/readback arrays and ordinary object fields are excluded.
    """
    from .cli_contract import command_table

    table = command_table()
    model = ANSWERS.get(operation)
    if operation not in table["reads"] or operation not in table["collections"] or model is None:
        return ()
    arrays = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if get_origin(annotation) is not list:
            candidates = [value for value in get_args(annotation) if value is not type(None)]
            if len(candidates) != 1:
                continue
            annotation = candidates[0]
        if get_origin(annotation) is list or (
            operation == "ACTIVITY_LIST" and name == "tasks" and get_origin(annotation) is dict
        ):
            arrays.append(name)
    return tuple(arrays)


def answer_problem(
    operation: str, body: object, request: Mapping[str, Any] | None = None
) -> str | None:
    """Whether one answer holds its published model and offers only acceptable requests.

    The model is the one `schema show` publishes for its operation (V410); each request the
    answer offers is one the Host accepts as it stands (`request_problem`, V449). A refusal
    answers in the refusal's shape (`status` REFUSED, `refused`, a failure code), which no model
    describes: it holds when it carries words and a way on (OP4, `refusal_problem`). An
    operation with no model still holds its offered requests.

    Args:
        operation: The operation that answered.
        body: Its answer.
        request: The request's fields, which the CLI's copy of a read's answer names (OP15).

    Returns:
        None when it holds; else the operation and each field that fails, by path.
    """
    from .cli_contract import (
        command_table,
        exit_problem,
        named_read,
        offered_requests,
        refusal_problem,
        refused,
        request_problem,
    )

    model = ANSWERS.get(operation)
    if not isinstance(body, dict):
        return None
    is_refusal = refused(body)
    offered = [
        f"`{name}` ({problem})"
        for name, offered_request in offered_requests(body).items()
        if (problem := request_problem(offered_request)) is not None
    ]
    if offered:
        return f"{operation} offers " + ", ".join(offered)
    problem = refusal_problem(
        body, operation=operation, collection_fields=collection_answer_fields(operation)
    )
    if problem is not None:
        return f"{operation}: {problem}"
    if not is_refusal and model is not None:
        try:
            model.model_validate(body)
        except ValidationError as error:
            failed = sorted(
                {".".join(str(part) for part in item["loc"]) for item in error.errors()}
            )
            return f"{operation}: " + ", ".join(failed)
    if (declaration := declaration_problem(operation, body)) is not None:
        return f"{operation}: {declaration}"
    if (continuation := continuation_problem(body)) is not None:
        return f"{operation}: {continuation}"
    if (stopped := stopped_problem(body)) is not None:
        return f"{operation}: {stopped}"
    if is_refusal or model is None:
        return None
    if (exited := exit_problem(body)) is not None:
        return f"{operation}: {exited}"
    table = command_table()
    if operation in table["reads"]:
        # As the CLI prints and saves it, which `--from` reads (V449).
        named = named_read({**(request or {}), "operation": operation}, body)
        return _rereads(operation, named, table)
    return None


def stopped_problem(
    body: Mapping[str, Any], lifecycle: str | None = None, *, way_on: bool = True
) -> str | None:
    """Whether an answer about a stopped Task offers a wait on it, or no way on.

    Args:
        body: The answer; its Task is `task_id`, or the `blocking_task_id` a refusal names.
        lifecycle: The Task's state as its record holds it; else the answer's own.
        way_on: Whether a missing way on counts: it does for the answer's own state, not for a
            command's answer about a Task its act is still moving.

    Returns:
        None when the Task is not stopped or its answer offers its way on; else what is wrong.
    """
    from .cli_contract import STOPPED_STATES, offered_requests, refused, task_state

    task = body.get("task_id") or body.get("blocking_task_id")
    state = lifecycle or task_state(dict(body))
    # A resubmitted recovery moves its Task out of its stop by this very act.
    if task is None or state not in STOPPED_STATES or body.get("status") == "RECOVERY_RESUBMITTED":
        return None
    offers = offered_requests(dict(body))
    waits = sorted(
        name
        for name, offer in offers.items()
        if str(offer.get("task_id")) == str(task)
        and (offer.get("operation") == "WAKE_REGISTER" or offer.keys() & {"wait_seconds", "watch"})
    )
    if waits:
        return f"a {state} Task's answer offers a wait on it: {', '.join(waits)}"
    # A refusal's way on is its catalogued action, as its words carry it (OP4).
    # A cancel was the request itself; a Task that waits on a request must be offered one.
    held = offers or refused(dict(body)) or not way_on or state == "CANCELLED"
    return None if held else f"a {state} Task's answer offers no way on"


def declaration_problem(operation: str, body: Mapping[str, Any]) -> str | None:
    """Check the editable declaration at the observed answer boundary.

    Args:
        operation: The operation that answered.
        body: Its full answer, before display compaction.

    Returns:
        None for a declaration, refusal or named exception; otherwise the missing part.
    """
    from .cli_contract import declaration_contracts, refused

    contract = declaration_contracts().get(operation)
    if contract is None or refused(dict(body)):
        return None
    if isinstance(body.get("yaml"), str) and str(body["yaml"]).strip():
        return None
    if isinstance(body.get("template"), dict) and body["template"]:
        return None
    for exception in contract.get("declaration_exceptions", ()):
        if body.get(exception["field"]) in exception["values"]:
            return None
    return "editable declaration (yaml or template) is absent"


def continuation_problem(body: dict[str, Any]) -> str | None:
    """Whether every offered edge keeps its bound references or names its remaining choice.

    The operation registry supplies each edge's contract. The actual client continues it,
    including declarations and aliases; no second continuation algorithm is maintained.
    """
    from .cli_contract import client_refusal, command_table, offered_requests
    from .client import LocalResearchClientError, continued, reference_field

    table = command_table()
    for name, request in offered_requests(body).items():
        operation = str(request.get("operation"))
        contract = table["fields"].get(operation)
        if contract is None:
            return f"{name} offers no registered operation"
        # Select the one edge as request --action does; other offers for the same operation
        # are a choice, not evidence that this bound edge cannot continue.
        selected = {"next_requests": {name: request}}
        try:
            again = continued(operation, selected, {}, frozenset(contract["allowed"]))
        except LocalResearchClientError as error:
            if not client_refusal(str(error)).detail:
                return f"{name} refuses without words ({error})"
            continue
        changed = [
            key
            for key, value in request.items()
            if reference_field(key) and value is not None and again.get(key) != value
        ]
        if changed:
            return f"{name} loses its bound references: {', '.join(sorted(changed))}"
    return None


def _rereads(operation: str, body: dict[str, Any], table: dict[str, Any]) -> str | None:
    """Whether a read's answer reads again from itself through `--from` (OP15, V449)."""
    from .cli_contract import choices, request_problem
    from .client import LocalResearchClientError, continued

    allowed = frozenset(table["fields"][operation]["allowed"])
    try:
        again = continued(operation, body, {}, allowed)
    except LocalResearchClientError as error:
        code, _sep, keys = str(error).partition(":")
        if code == "local_client.next_request_required":
            # It offers several reads of its operation, which `--next` names; each is held
            # as an offered request.
            return None
        if code == "local_client.response_reference_missing" and (
            keys.split(",")[0] not in table["fields"][operation]["required"]
        ):
            # A read whose target is optional and whose file names none, a goal read with no
            # goal, is refused by `--from` rather than read the session's default (OP17); a
            # readback whose Task is optional and that read none reads again as itself (V575).
            return None
        return f"{operation}'s answer does not read again from itself ({error})"
    if left := choices(again) or request_problem(again):
        return f"{operation}'s answer does not read again from itself ({left})"
    return None


def outline(model: type[BaseModel], depth: int = 3) -> Any:
    """A model's parts by name, ``depth`` levels deep: the paths ``--section`` can ask for.

    Each part is its type, or its own parts; a list holds its item's outline. The owner's JSON
    schema is read, never restated, so the outline moves with the model (V112).

    Args:
        model: The owner's model of the part.
        depth: How many levels of parts to name.

    Returns:
        The outline, a mapping of part names at the top.
    """
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        while "$ref" in node:
            node = defs[str(node["$ref"]).rsplit("/", 1)[-1]]
        return node

    def part(node: dict[str, Any], left: int) -> Any:
        node = resolve(node)
        options = [resolve(o) for o in node.get("anyOf", []) if resolve(o).get("type") != "null"]
        if len(options) == 1:
            return part(options[0], left)
        if options:
            return " or ".join(
                sorted({str(o.get("type", o.get("title", "value"))) for o in options})
            )
        if node.get("type") == "array":
            return [part(node.get("items", {}), left)]
        properties = node.get("properties")
        if isinstance(properties, dict) and properties:
            if left == 0:
                return "object"
            return {name: part(value, left - 1) for name, value in properties.items()}
        return str(node.get("type", node.get("title", "value")))

    return part(schema, depth)
