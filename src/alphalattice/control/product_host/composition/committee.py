"""The investment committee on a date's published positions.

The lead agent is the PM; the Alpha, Risk and CRO specialists argue the Host's tension points
on one floor per update. The floor is the goal ledger's own entries: an open entry and one entry
per message, read back to derive its state, and each message the floor reveals is filed as a row
of the lead's goal conversation. Each member speaks and reads as itself by a key the Host mints:
a specialist's is in its bundle, the PM's goes to the session that opened the floor. Round one
is blind until all four stances are in or its time box ends; then challenges, replies and the
PM's rulings move by event, each specialist capped, until the PM's verdict or the floor's time
box closes it. The committee changes no number.
"""

from __future__ import annotations

import hmac
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Final
from uuid import UUID

from alphalattice.control.product_host.composition.committee_materials import (
    AGENDA_LIMIT as AGENDA_LIMIT,
)
from alphalattice.control.product_host.composition.committee_materials import (
    QUESTIONS,
    agenda,
    concentration,
    evidence_gap,
    facts,
    issue_lines,
    review_now,
    risk_view,
    status_word,
    tension_points,
)
from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
    schema_request,
)
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.interface.local_application.labels import label as installed_label
from alphalattice.interface.local_application.portfolio_research import (
    CommitteeMessage,
    PortfolioResearchOperationRequest,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import command_prefix
from alphalattice.protocols.actor_execution.bundles import bundle_slot

ROLES: Final = ("PM", "ALPHA", "RISK", "CRO")
CARDS: Final = {r: f"alphalattice_{r.lower()}" for r in ROLES[1:]} | {"PM": "research_lead"}
"""Each member's card, the Team page's participant name."""
REVIEWERS: Final = {"ANALYST": "alphalattice_evidence_analyst", "CRO": "alphalattice_cro"}
"""The review's members: each answers its own bundle on that bundle's thread."""
CHANNEL: Final = "PRODUCT_COMMITTEE"
"""The Host's own channel for floor rows; a client's event never takes it."""
REVIEW_CHANNEL: Final = "PRODUCT_REVIEW"
"""The Host's own channel for a review thread's rows, which Conversation shows."""
CHANNELS: Final = frozenset({CHANNEL, REVIEW_CHANNEL})
RELAY_WORD: Final = "Relayed by the agent"
PASSED: Final = "{member} passed its discussion turn"
"""A yielded turn, as the report reads it."""
STANCE_MINUTES, FLOOR_MINUTES, MESSAGE_CAP = 10, 40, 3
TURN: Final = frozenset({"CHALLENGE", "REPLY", "PASS"})
"""A specialist's discussion turn: spoken, or passed."""
RULES: Final[dict[str, tuple[tuple[str, ...], frozenset[str], str | None]]] = {
    "STANCE": (ROLES, frozenset({"STANCES"}), None),
    "CHALLENGE": (ROLES[1:], frozenset({"DEBATE"}), None),
    "REPLY": (ROLES, frozenset({"DEBATE"}), "ANY"),
    "PASS": (ROLES[1:], frozenset({"DEBATE"}), None),
    "RULING": (ROLES[:1], frozenset({"DEBATE"}), "CHALLENGE"),
    "VERDICT": (ROLES[:1], frozenset({"DEBATE"}), None),
    "PERSON_ANSWER": (ROLES[:1], frozenset({"DEBATE", "CLOSED"}), "FOR_THE_PERSON"),
}
"""The round state machine: each kind's senders, the stages it is sent in, and what it answers
(a challenge, any message, or an item the PM handed to the person)."""
OUTCOMES: Final = {
    "RULING": frozenset({"ADOPT", "REJECT", "FOR_THE_PERSON"}),
    "VERDICT": frozenset({"PROCEED", "PROCEED_WITH_NOTES", "DECLINE", "FOR_THE_PERSON"}),
}
"""Each kind's outcomes; a kind not named takes none (SHAPE, with what it answers)."""
COMMAND: Final = " ".join(command_prefix())
RULE_WORDS: Final = {
    "ROLE": "Only its senders send this kind, each with the key in its own bundle: a STANCE or "
    "REPLY any member, a CHALLENGE or PASS a specialist, a RULING, VERDICT or PERSON_ANSWER the "
    "PM.",
    "STANCES": "The floor is in its blind first round, which takes only one STANCE from each "
    f"member; follow `{COMMAND} committee wait` until the stances are revealed.",
    "DEBATE": "The floor is past its stances; it takes challenges, replies, passes, rulings and "
    "the verdict, not a STANCE.",
    "CLOSED": "The floor is closed; it takes only the PM's answer relayed from the person to "
    "an item handed to them. Export the report from it.",
    "STANCE_IN": "This member's stance is on the floor already, one each; speak again by a "
    "CHALLENGE or REPLY once the stances are revealed.",
    "CAP": "This member has sent its three challenges or replies after its stance; the floor "
    "takes no more from it.",
    "TURN": "A PASS yields this member's one discussion turn: it comes from a member whose stance "
    "is on the floor, before it challenged or replied, and names no target or position; after it "
    "the member sends nothing more.",
    "TARGET": "A target, position or alias names nothing on the floor; name only the tension "
    f"points, holdings and messages `{COMMAND} committee show` lists.",
    "REPLY_TO": "The message named in reply_to cannot be answered by this kind: a REPLY answers "
    "any message, a RULING a challenge not yet ruled, a PERSON_ANSWER an item handed to the "
    "person. Send the corrected message the answer offers.",
    "SHAPE": "This kind's shape does not fit: a REPLY, RULING or PERSON_ANSWER names in "
    "reply_to the message it answers and no other kind does; a RULING's or the VERDICT's "
    "outcome is required and no other kind takes one. Send the corrected message the answer "
    "offers.",
    "UNRULED": "A verdict closes the floor, so the PM rules on every challenge first and one "
    "stands unruled; read it, send its ruling in your own words, then the verdict.",
    "MEMBER": "This bundle's member submits from its own session with its key and takes the Host's "
    "corrections itself; start it with its bundle, then run this again for its receipt.",
    "KEY": "This key is not this bundle's: a member's key admits only its own bundle's answer, "
    "with the command its route names.",
    "ENDED": "This bundle's answer is accepted and its key has ended; only that answer reads its "
    "receipt again. Return the receipt it gave to your lead.",
    "FACTS": "A RESERVE rests on the floor's facts: cite by S alias the facts, or the named "
    f"absences, that `{COMMAND} committee show` lists under this member's questions; resubmit.",
    "NUMBER": "A digit is typed outside an alias ({subject}), and the Host renders every number: "
    "resubmit the same message with each typed figure or reference replaced by its alias on the "
    "floor.",
    "AGENDA": "This message's point selection does not fit the agenda of {subject} points. The PM "
    "may raise one below-line point through REPLY positions; other messages name only agenda or "
    "raised tension points. Read the floor and send the offered correction, or revise the named "
    "text field.",
}
"""Each floor rule's own words, as its refusal says them."""
_VIEWS: Final = frozenset({"STANCE", "CHALLENGE", "REPLY"})
ALIAS: Final = re.compile(r"\b[CHMST][0-9]{1,4}\b")
"""A floor alias: a component (C), a holding (H3), a message (M4), a fact (S) or a point (T2)."""
"""The kinds that carry a member's own view; a ruling, the verdict and a relay do not."""


@dataclass(frozen=True)
class Floor:
    """One update's floor as its goal's ledger holds it."""

    goal_id: UUID
    seed: str = field(repr=False)
    opened: dict[str, Any]
    messages: list[dict[str, Any]]
    filed: frozenset[str]
    author_session: str | None = field(default=None, compare=False)
    rule_cases: frozenset[str] = field(default=frozenset(), compare=False)
    rule_audits: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    rule_held: frozenset[str] = field(default=frozenset(), compare=False)

    def key(self, role: str) -> str:
        """Derive the member's key on this floor from the seed the Host keeps outside records."""
        return member_key(self.seed, self.opened["update_task_id"], role)

    def admits(self, role: str, key: str | None) -> bool:
        """Whether the key is this member's own."""
        return key is not None and hmac.compare_digest(key.encode(), self.key(role).encode())

    def stage(self, now: datetime) -> str:
        """STANCES, DEBATE once the stances are revealed, CLOSED after the verdict or time box."""
        kinds = [m["kind"] for m in self.messages]
        if "VERDICT" in kinds or now >= datetime.fromisoformat(self.opened["closes_at"]):
            return "CLOSED"
        stances = {m["role"] for m in self.messages if m["kind"] == "STANCE"}
        late = now >= datetime.fromisoformat(self.opened["stances_close_at"])
        return "DEBATE" if stances >= set(ROLES) or late else "STANCES"


OPENER_WORDS: Final = (
    "The PM's key and route belong to the session that opened this floor; its wait goes on "
    "there. Read the floor here, or continue in that session."
)
"""Why another session's open names no key, and its way on."""


def settled(floor: Floor, now: datetime) -> bool:
    """Whether the debate has settled, so the PM gives the verdict on an event.

    It has once every specialist used its turn (a CHALLENGE or REPLY) or passed it, and no
    challenge stands unruled.
    """
    spoke = {m["role"] for m in floor.messages if m["kind"] in TURN}
    ruled = {m["reply_to"] for m in floor.messages if m["kind"] == "RULING"}
    open_ = [m for m in floor.messages if m["kind"] == "CHALLENGE" and m["id"] not in ruled]
    return floor.stage(now) == "DEBATE" and spoke >= set(ROLES[1:]) and not open_


def member_key(seed: str, thread: str, role: str) -> str:
    """A member's key on one thread: it admits that member's messages there and nowhere else."""
    return hmac.new(bytes.fromhex(seed), f"{thread}:{role}".encode(), sha256).hexdigest()[:32]


def floor_of(store: GoalStore, task: str) -> Floor | None:
    """The update's floor, from the goal whose ledger opened it; None when none was opened."""
    floor = thread_of(store, task)
    return floor if floor is not None and "selector" in floor.opened else None


def published_floor(store: GoalStore, task: str, publication_hash: str) -> Floor | None:
    """The update's floor when it was opened on exactly this publication of it."""
    floor = floor_of(store, task)
    opened = floor.opened["selector"] if floor is not None else {}
    exact = (opened.get("update_task_id"), opened.get("update_publication_hash"))
    return floor if exact == (task, publication_hash) else None


def delivery(floor: Floor) -> dict[str, Any]:
    """The floor's report: the date's delivery export under its review."""
    return {
        "operation": "EXPERIMENT_DELIVERY_EXPORT",
        **floor.opened["selector"],
        **floor.opened.get("review", {}),
    }


def research_update(registry: Any, task: str | None) -> bool:
    """Whether the Task is a research update, the one a committee opens on."""
    try:
        return bool(registry.task(UUID(task)).task_kind == ADVANCEMENT_TASK_KIND)
    except (LookupError, TypeError):  # an unknown Task is the specialist bundle's own refusal
        return False


def contextualize(body: dict[str, Any], floor: Floor | None, now: datetime) -> None:
    """Put a dated read's floor standing on it, in place of any it carried."""
    body.pop("committee_context", None)
    if floor is not None:
        body["committee_context"] = context(floor, now)


def offer(host: Any, review: dict[str, Any], book: Any, caller: str) -> dict[str, Any]:
    """Offer an agent's read of a research update's published review its committee.

    ``host`` is the Host's operations: its goals' store and clock and its Task registry. The
    floor is the one opened on the publication this review read, as it stands: opened (again,
    which gives its opener the PM's key and route) until it closes, then its report; the answer
    names that next action. The page's and the Host's own reads take none. A goal record the
    floor cannot be read from is named, and nothing is offered.
    """
    read = book.request_fields() if book is not None else {}
    task = read.get("update_task_id")
    registry = host.workspace_session.task_control_registry
    if caller == "HUMAN" or review.get("state") != "REVIEW_PUBLISHED":
        return review
    if not research_update(registry, task):
        return review
    now = host.goals.clock()
    try:
        floor = published_floor(host.goals.store, str(task), read["update_publication_hash"])
    except (ValueError, OSError) as error:
        review["committee_unreadable"] = public_failure(error, "committee.refused")
        return review
    if floor is not None and floor.stage(now) == "CLOSED":
        review["next_action"] = "REPORT_THE_COMMITTEE"
        review["next_requests"]["committee_report"] = delivery(floor)
    else:
        review["next_action"] = "CONVENE_THE_COMMITTEE"
        review["next_requests"]["committee"] = {
            "operation": "COMMITTEE_OPEN",
            "update_task_id": task,
        }
    return review


LEAD_PUBLISHED: Final = (
    "Nothing was submitted for this bundle; the date's review is already published, so its "
    "committee is offered."
)
"""A refusal to the lead where the review stands published: the committee, not a new bundle."""

SUPERSEDED: Final = frozenset(
    {"agent_bundle.preparation_stale", "REFUSED_EVIDENCE_SELECTION_AMBIGUOUS"}
)
"""Refusals meaning the lead prepared again after this bundle; a transient one keeps its words."""


def published(host: Any, standing: Any, book: Any, caller: str) -> dict[str, Any]:
    """The date's committee on its publication's review standing, as `offer` gives it.

    Only a standing that is REVIEWED is a published review; any other offers nothing.
    """
    if (standing or {}).get("status") != "REVIEWED":
        return {"next_requests": {}}
    return offer(host, {"state": "REVIEW_PUBLISHED", "next_requests": {}}, book, caller)


def _update_book(fields: Mapping[str, Any]) -> Any:
    """A research update's book by its selector's fields; another book's raise KeyError."""
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector

    return BookSelector(
        update_task_id=UUID(str(fields["update_task_id"])),
        update_publication_hash=str(fields["update_publication_hash"]),
        position_basis=str(fields["position_basis"]),
    )


def lead_way(
    host: Any, book: Mapping[str, Any], lead: bool, caller: str, answered: dict[str, Any]
) -> dict[str, Any]:
    """A bundle's answer, and for the lead a superseded refusal's way on to the committee.

    The lead resends an earlier bundle after preparing again; refused as SUPERSEDED on a book
    whose date's review stands published, from that Task's own publication, it is offered that
    review's committee (`published`) in place of a member's words, its failure code kept. A
    member's answer, a transient refusal and an unreadable store's are returned as they are.
    """
    from alphalattice.control.product_host.composition.evidence_review_delivery import (
        EvidenceReviewDelivery,
    )

    if not lead or answered.get("failure_code") not in SUPERSEDED:
        return answered
    try:
        selector = _update_book(book)
        published_by = host.research_updates.publication_task(selector.update_publication_hash)
        if published_by != selector.update_task_id:
            return answered
        standing = EvidenceReviewDelivery(host.review).review_standing(selector)
    except Exception:  # another book or an unreadable store: the typed refusal stands
        return answered
    review = published(host, standing, selector, caller)
    if not review["next_requests"]:
        return answered
    words = {key: LEAD_PUBLISHED for key in ("message", "detail") if key in answered}
    return {
        **answered,
        **words,
        "next_action": review["next_action"],
        "next_requests": review["next_requests"],
    }


def update_offer(host: Any, body: dict[str, Any], caller: str) -> dict[str, Any]:
    """A research update's read, offering the date's committee once its review is published.

    A read of a previous publication, or one with its own way on (a stop), offers none.
    """
    selected = body.get("review_selector") or {}
    if not selected or body.get("publication_is_previous") or body.get("next_action"):
        return body
    book = _update_book(selected)
    review = published(host, body.get("review_standing"), book, caller)
    if review["next_requests"]:
        body["next_requests"] = {**(body.get("next_requests") or {}), **review["next_requests"]}
    return {
        **body,
        **{key: review[key] for key in ("next_action", "committee_unreadable") if key in review},
    }


def _member_thread(request: Any) -> tuple[str, tuple[str, ...]]:
    """The thread a member's own request speaks on and the roles its key may hold.

    A bundle prepare carries no key (it hands one over); any other request has no thread ("").
    """
    if request.operation == "AGENT_BUNDLE_PREPARE" and request.agent_role in ROLES[1:]:
        return str(request.task_id or ""), ()
    if request.operation == "COMMITTEE_SUBMIT" and request.committee_role in ROLES[1:]:
        return str(request.update_task_id or ""), (request.committee_role,)
    if request.operation == "AGENT_ANSWER_SUBMIT" and request.bundle_directory:
        return thread_id(bundle_slot(request.bundle_directory)), tuple(REVIEWERS)
    return "", ()


def member_goal(
    store: GoalStore, request: Any, held: Any, provenance: Any, now: datetime
) -> tuple[bool, Any]:
    """Whether a request is a member's own on an open thread, and the goal it counts toward.

    A member's message or review answer whose key its thread admits, and its keyless bundle
    prepare on a floor not yet closed while its Session holds no goal, carry the thread's context
    as a span carries its parent's trace (OpenTelemetry context propagation): they count toward
    the thread's goal. A request is a member's only while that goal is open and the caller named
    none; otherwise it keeps `held`, the goal it named or its Session holds. A key is checked
    against the thread's seed alone, so no ledger is read twice for it.
    """
    thread, roles = _member_thread(request)
    kept = store.committee(thread) if thread else None
    goal = None if kept is None else store.head(UUID(kept["goal_id"]))
    if kept is None or goal is None or goal.state != "OPEN" or getattr(provenance, "goal_id", None):
        return False, held
    if not roles:  # the bundle prepare that hands a member its key
        floor = floor_of(store, thread) if held is None else None
        opened = floor is not None and floor.stage(now) != "CLOSED"
        return opened, goal if opened else held
    keys = [member_key(kept["seed"], thread, role).encode() for role in roles]
    key = request.committee_key
    admitted = key is not None and any(hmac.compare_digest(key.encode(), k) for k in keys)
    return True, goal if admitted else held


def thread_of(store: GoalStore, thread: str) -> Floor | None:
    """One thread as its goal's ledger holds it: a committee's floor or a review bundle's."""
    held = store.committee(thread)
    if held is None:
        return None
    entries = store.attributed(UUID(held["goal_id"]))
    found = [
        dict(value)
        for e in entries
        for k in ("committee_open", "committee_message")
        if isinstance(value := e.get(k), Mapping) and value.get("update_task_id") == thread
    ]
    if not found or not {"selector", "bundle_reference"} & found[0].keys():
        return None
    channel = REVIEW_CHANNEL if "bundle_reference" in found[0] else CHANNEL
    rows = {
        str(e.get("message_id"))
        for e in entries
        if e.get("input_channel") == channel and e.get("reference") == thread
    }
    return Floor(
        UUID(held["goal_id"]),
        held["seed"],
        found[0],
        found[1:],
        frozenset(rows),
        rule_cases=store.rule_cases(thread, entries),
        rule_audits=store.rule_audits(thread, entries),
        rule_held=store.rule_held(thread, entries),
    )


def open_floor(
    store: GoalStore,
    goal_id: UUID,
    readback: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    session: tuple[str, str],
    now: datetime,
    agenda_limit: int = AGENDA_LIMIT,
) -> Floor:
    """Open the update's floor in the lead's goal, or read the one already open.

    The floor is built before the store keeps its seed, so a floor that cannot be built leaves
    nothing behind.
    """
    task, held = str(readback["task_id"]), holdings(readback)
    label = installed_label(str(readback.get("strategy_package_id") or ""))
    proposal = readback["publication"].get("pending_proposal") or {}
    opened = {
        "update_task_id": task,
        "selector": dict(readback["review_selector"]),
        "member_sessions": True,
        # The published review the floor argues on, which its report carries.
        "review": {"review_publication_hash": standing.get("review_publication_hash")}
        if (standing := readback.get("review_standing") or {}).get("status") == "REVIEWED"
        else {},
        "strategy_package_id": readback.get("strategy_package_id"),
        "labels": readback.get("labels") or (label.model_dump() if label else {}),
        "schedule": proposal.get("schedule"),
        "stances_close_at": (now + timedelta(minutes=STANCE_MINUTES)).isoformat(),
        "closes_at": (now + timedelta(minutes=FLOOR_MINUTES)).isoformat(),
        "session": list(session),
        "holdings": held,
        "date_risk": readback.get("date_risk"),
        "tension_points": tension_points(held, evidence, readback.get("date_risk"), readback),
        "facts": facts(readback, evidence, held),
    }
    opened.update(agenda(opened["tension_points"], held, agenda_limit))
    with store.lock:
        kept = store.committee(task, goal_id)
        assert kept is not None
        if floor_of(store, task) is None:
            store.attribute(UUID(kept["goal_id"]), {"committee_open": opened})
        floor = floor_of(store, task)
    assert floor is not None
    return floor


def operate(
    store: GoalStore,
    request: PortfolioResearchOperationRequest,
    now: datetime,
    *,
    provenance: RequestProvenance | None,
    dated: Callable[[], tuple[dict[str, Any], dict[str, Any]] | None],
    file_floor: Callable[[], None],
    agenda_limit: int = AGENDA_LIMIT,
) -> dict[str, object]:
    """Open, speak on or read an update's floor (`committee`), as the CLI answers it.

    A member speaks and reads with its own key. The PM's key goes to the session that opened
    the floor, at the open and again on its later open, so a lead whose context was cut keeps
    the floor. A row `file_floor` cannot file now is filed at the next read.
    """
    task, role = str(request.update_task_id), request.committee_role
    try:
        if request.operation == "COMMITTEE_SUBMIT":
            assert role is not None and request.committee_message is not None
            number = submit(
                store,
                task,
                (role, request.committee_key),
                request.committee_message,
                now,
                provenance=provenance,
            )
            with suppress(ValueError, OSError):
                file_floor()
            return {"status": "COMMITTEE_MESSAGE_ACCEPTED", "message_id": number}
        if request.operation == "COMMITTEE_OPEN":
            publication = dated()
            if publication is None:
                raise ValueError("committee.publication_required")
            if provenance is None or provenance.goal_id is None or provenance.session is None:
                raise ValueError("committee.session_required")
            session = (str(provenance.vendor), provenance.session)
            floor = open_floor(
                store,
                UUID(provenance.goal_id),
                *publication,
                session=session,
                now=now,
                agenda_limit=agenda_limit,
            )
            return _opened(floor, session, task, now)
        found = floor_of(store, task)
        if found is None:
            raise ValueError("committee.not_open")
        if role is not None and not found.admits(role, request.committee_key):
            raise ValueError("committee.message_refused:KEY")
    except ValueError as error:
        code = public_failure(error, "committee.refused")
        refused = {"status": "REFUSED", "failure_code": code, "refused": code, **explain(code)}
        if code == "committee.not_open":
            return refused | _elsewhere(store, request, now)
        return refused | _corrected(store, request, code.rpartition(":")[2], now)
    with suppress(ValueError, OSError):  # a time box's reveal reaches the Team page on read
        file_floor()
    read = state(found, role, request.committee_seen or 0, now)
    offered = {"report": read["delivery"]} if read["stage"] == "CLOSED" else {}
    if read["stage"] == "DEBATE" and _turn_open(found, role):
        offered["pass"] = {
            "operation": "COMMITTEE_SUBMIT",
            **{k: getattr(request, k) for k in ("committee_role", "committee_key")},
            "update_task_id": task,
            "committee_message": {"kind": "PASS", "text": "Nothing to add."},
        }
    return {"status": "COMMITTEE_FLOOR", **read, "next_requests": offered}


def _opened(floor: Floor, session: tuple[str, str], task: str, now: datetime) -> dict[str, Any]:
    """The open's answer, with the PM's key and route for the session that opened the floor.

    Another session is told whose they are and offered the floor's read.
    """
    mine = floor.opened["session"] == list(session)
    # Each specialist reads its bundle by `alphalattice bundle prepare`, its key in it.
    bundles = {
        f"bundle:{member}": {
            "operation": "AGENT_BUNDLE_PREPARE",
            "agent_role": member,
            "task_id": task,
            "bundle_directory": None,
        }
        for member in ("ALPHA", "RISK", "CRO")
    }
    read = {"read": {"operation": "COMMITTEE_READ", "update_task_id": task}}
    return {
        "status": "COMMITTEE_OPEN",
        "stage": floor.stage(now),
        **({"pm_key": floor.key("PM"), "pm_route": pm_route(floor)} if mine else {}),
        **({} if mine else {"detail": OPENER_WORDS}),
        "next_requests": {**({} if mine else read), **bundles},
    }


def _turn_open(floor: Floor, role: str | None) -> bool:
    """Whether this specialist may still pass: its stance is in, its turn not yet taken."""
    kinds = {m["kind"] for m in floor.messages if m["role"] == role}
    return role in ROLES[1:] and "STANCE" in kinds and not kinds & TURN


def submit(
    store: GoalStore,
    task: str,
    member: tuple[str, str | None],
    message: CommitteeMessage,
    now: datetime,
    *,
    provenance: RequestProvenance | None = None,
) -> str:
    """Take a keyed message by the round rules; an identical keyed replay keeps its author."""
    with store.lock:
        floor = floor_of(store, task)
        if floor is None:
            raise ValueError("committee.not_open")
        (role, key), (senders, stages, answers) = member, RULES[message.kind]
        # A member's message resent as it stands answers with its first admission, spending nothing.
        same = _same(floor, role, message)
        floor = _admit_member(floor, role, key, provenance, same=same)
        if same is not None:
            return str(same["id"])
        stage, mine = floor.stage(now), [m for m in floor.messages if m["role"] == role]
        known = _known(floor)
        target = next((m for m in floor.messages if m["id"] == message.reply_to), None)
        fits = target is not None and answers in {"ANY", target["kind"], target.get("outcome")}
        ruled = _ruled(floor, message)
        spoken = sum(m["kind"] not in {"STANCE", "PASS"} for m in mine)
        named = (*message.targets, *message.positions, *ALIAS.findall(message.text))
        agenda_points = _agenda_points(floor, role, message)
        typed = any(c.isnumeric() for c in ALIAS.sub("", message.text))
        for rule, broken in (
            ("ROLE", role not in senders),
            (stage, stage not in stages),
            ("STANCE_IN", message.kind == "STANCE" and any(m["kind"] == "STANCE" for m in mine)),
            ("TURN", _turn_refused(mine, message)),
            (
                "CAP",
                message.kind not in {"STANCE", "PASS"} and role != "PM" and spoken >= MESSAGE_CAP,
            ),
            ("TARGET", any(t not in known for t in named)),
            ("SHAPE", misshapen(message)),
            ("REPLY_TO", answers is not None and (not fits or ruled is not None)),
            ("UNRULED", bool(_unruled(floor, message))),
            # The Host renders every number; the person's own relayed words keep theirs.
            ("NUMBER", message.kind != "PERSON_ANSWER" and typed),
            ("FACTS", _unsourced(floor, role, message)),
            ("AGENDA", _outside_agenda(message, named, agenda_points)),
        ):
            if broken:
                raise ValueError(f"committee.message_refused:{rule}")
        number = f"M{len(floor.messages) + 1}"
        entry = {
            "update_task_id": task,
            "id": number,
            "role": role,
            "at": now.isoformat(),
            "author_session": floor.author_session,
        }
        store.attribute(
            floor.goal_id, {"committee_message": {**entry, **message.model_dump(mode="json")}}
        )
        return number


def _turn_refused(mine: Sequence[dict[str, Any]], message: CommitteeMessage) -> bool:
    """A member that passed sends nothing more; a PASS needs its stance, no spoken turn and no
    target or position."""
    kinds = {m["kind"] for m in mine}
    return "PASS" in kinds or (
        message.kind == "PASS"
        and (
            "STANCE" not in kinds
            or bool(kinds & TURN)
            or bool(message.targets or message.positions)
        )
    )


def _outside_agenda(
    message: CommitteeMessage, named: Sequence[str], points: set[str] | None
) -> bool:
    """No message names an unraised point, and a stance judges only agenda or raised points."""
    return points is not None and (
        any(a.startswith("T") and a not in points for a in named)
        or (message.kind == "STANCE" and not set(message.positions) <= points)
    )


def _agenda_points(floor: Floor, role: str | None, message: CommitteeMessage) -> set[str] | None:
    """The agenda and raised points, plus one point explicitly raised by this PM reply."""
    bounded = agenda_view(floor).get("agenda")
    if bounded is None:
        return None
    allowed = set(bounded["aliases"]) | set(bounded["raised"])
    if role == "PM" and message.kind == "REPLY":
        allowed.update(next(((a,) for a in bounded["below"] if a in message.positions), ()))
    return allowed


def _unsourced(floor: Floor, role: str, message: CommitteeMessage) -> bool:
    """A RESERVE that cites none of the floor's facts while the member's questions hold some."""
    facts = floor.opened.get("facts") or ()
    cited = {f["alias"] for f in facts} & set(ALIAS.findall(message.text))
    asked = any(f["question"] in QUESTIONS.get(role, {}) for f in facts)
    return "RESERVE" in message.positions.values() and asked and not cited


def _unruled(floor: Floor, message: CommitteeMessage) -> list[str]:
    """The challenges no ruling answers yet, which a VERDICT waits for; none for another kind."""
    if message.kind != "VERDICT":
        return []
    ruled = {m["reply_to"] for m in floor.messages if m["kind"] == "RULING"}
    return [m["id"] for m in floor.messages if m["kind"] == "CHALLENGE" and m["id"] not in ruled]


def _owed(
    floor: Floor | None,
    request: PortfolioResearchOperationRequest,
    member: dict[str, Any],
    rule: str,
    now: datetime,
) -> dict[str, Any]:
    """For a verdict refused UNRULED, each ruling it owes as a template: once per unruled
    challenge and outcome, its words left to the PM. For a NUMBER refusal whose every typed
    figure is a name the floor holds under an alias, the same message with those aliases."""
    message = request.committee_message
    if rule == "CLOSED" and floor is not None:
        return {"report": delivery(floor)}
    if rule == "AGENDA":
        assert floor is not None
        assert message is not None
        aliases = _agenda_points(floor, request.committee_role, message)
        assert aliases is not None
        blocked = set(floor.opened["agenda"]["below"]) - aliases
        offered = {"schema": schema_request("COMMITTEE_SUBMIT")}
        if blocked & set(ALIAS.findall(message.text)):
            return offered
        sent = message.model_dump(mode="json") | {
            "positions": {
                a: p
                for a, p in message.positions.items()
                if a in aliases or (message.kind != "STANCE" and a not in blocked)
            },
            "targets": [a for a in message.targets if a not in blocked],
        }
        return offered | {
            "corrected": {"operation": "COMMITTEE_SUBMIT", **member, "committee_message": sent},
        }
    renamed = _typed(floor, message)[1] if rule == "NUMBER" else None
    if message is not None and renamed and not any(c.isnumeric() for c in ALIAS.sub("", renamed)):
        sent = message.model_dump(mode="json") | {"text": renamed}
        return {"corrected": {"operation": "COMMITTEE_SUBMIT", **member, "committee_message": sent}}
    owed = [] if floor is None or message is None or rule != "UNRULED" else _unruled(floor, message)
    return {
        f"rule:{challenge}:{outcome}": {
            "operation": "COMMITTEE_SUBMIT",
            **member,
            "committee_message": {"kind": "RULING", "text": None, "reply_to": challenge}
            | {"outcome": outcome, "targets": [], "positions": {}},
        }
        for challenge in owed
        for outcome in sorted(OUTCOMES["RULING"])
    }


def _known(floor: Floor) -> set[str]:
    """Every alias the floor shows its members: points, holdings, components and messages."""
    known = {p["alias"] for p in floor.opened["tension_points"]} | {m["id"] for m in floor.messages}
    known |= {f["alias"] for f in floor.opened.get("facts", ())}
    return known | {a for h in floor.opened["holdings"] for a in (h["alias"], *h.get("scores", {}))}


def _typed(floor: Floor | None, message: CommitteeMessage | None) -> tuple[list[str], str]:
    """The figures a message types outside an alias, and its text with each that is a
    holding's name on the floor replaced by the holding's alias."""
    text = "" if message is None else message.text
    names = {} if floor is None else {h["name"]: h["alias"] for h in floor.opened["holdings"]}
    typed = [w.strip(".,;:!?()") for w in ALIAS.sub("", text).split() if any(map(str.isnumeric, w))]
    for word in typed:
        text = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", names.get(word, word), text)
    return typed, text


def _same(floor: Floor, role: str, message: CommitteeMessage) -> Mapping[str, Any] | None:
    """This member's message already on the floor exactly as sent, if any."""
    sent = message.model_dump(mode="json")
    mine = (m for m in floor.messages if m["role"] == role)
    return next((m for m in mine if all(m.get(k) == v for k, v in sent.items())), None)


def _ruled(floor: Floor | None, message: CommitteeMessage | None) -> dict[str, Any] | None:
    """The ruling already on the floor for the challenge a RULING answers; None otherwise."""
    if floor is None or message is None or message.kind != "RULING":
        return None
    rulings = (m for m in floor.messages if m["kind"] == "RULING")
    return next((m for m in rulings if m["reply_to"] == message.reply_to), None)


def _elsewhere(
    store: GoalStore, request: PortfolioResearchOperationRequest, now: datetime
) -> dict[str, Any]:
    """Where a request naming no open committee goes: the one open floor its member's key
    admits, the same request filled with that update; else each open floor's read."""
    floors = [floor_of(store, task) for task in store.committee_tasks()]
    open_ = [f for f in floors if f is not None and f.stage(now) != "CLOSED"]
    role, key, message = request.committee_role, request.committee_key, request.committee_message
    admitting = [f for f in open_ if role is not None and f.admits(role, key)]
    tasks = [str(f.opened["update_task_id"]) for f in open_]
    if len(admitting) != 1:
        reads = {
            f"floor:{task}": {"operation": "COMMITTEE_READ", "update_task_id": task}
            for task in tasks
        }
        return {"open_committees": tasks, "next_requests": reads}
    given = {
        "committee_role": role,
        "committee_key": key,
        "committee_message": message.model_dump(mode="json") if message else None,
        "committee_seen": request.committee_seen,
    }
    same = {"operation": request.operation, "update_task_id": admitting[0].opened["update_task_id"]}
    same |= {name: value for name, value in given.items() if value is not None}
    return {
        "open_committees": tasks,
        "next_action": "SEND_THE_CORRECTED_MESSAGE",
        "next_requests": {"corrected": same},
    }


def _read_fields(request: PortfolioResearchOperationRequest, rule: str) -> dict[str, Any]:
    """The member's own fields for its floor read; a key not the member's reads as anyone."""
    given = {"committee_role": request.committee_role, "committee_key": request.committee_key}
    member = {"update_task_id": str(request.update_task_id)}
    return (
        member
        if rule in {"ROLE", "KEY", "MEMBER"}
        else member | {k: v for k, v in given.items() if v}
    )


def misshapen(message: CommitteeMessage) -> bool:
    """Whether a message takes an outcome or reply its kind does not, or lacks one it must."""
    answers = RULES[message.kind][2] is not None
    return (
        message.outcome not in OUTCOMES.get(message.kind, {None})
        or (message.reply_to is None) == answers
    )


def _rule_words(
    floor: Floor | None, message: CommitteeMessage | None, rule: str, corrected: bool
) -> dict[str, Any]:
    """The rule's words name its figures, agenda size and any text the author must revise."""
    size = len(getattr(floor, "opened", {}).get("agenda", {}).get("aliases", ()))
    subject = str(size) if rule == "AGENDA" else ", ".join(_typed(floor, message)[0])
    words: dict[str, Any] = {"detail": RULE_WORDS[rule].replace("{subject}", subject), "rule": rule}
    if rule == "AGENDA":
        words["agenda_size"] = size
        if not corrected:
            words["author_revision_fields"] = ["committee_message.text"]
    return words


def _corrected(
    store: GoalStore, request: PortfolioResearchOperationRequest, rule: str, now: datetime
) -> dict[str, Any]:
    """A refused message's own rule in its words and, where only its shape is wrong, the same
    message corrected: once for each message its kind may answer and each outcome it takes."""
    if rule not in RULE_WORDS:
        return {}
    # A rule no field fixes: the member's own read of the floor, where its stage and aliases are.
    member = _read_fields(request, rule)
    floor_read = {"floor": {"operation": "COMMITTEE_READ", **member}}
    message, floor = request.committee_message, floor_of(store, str(request.update_task_id))
    owed = _owed(floor, request, member, rule, now)
    words = _rule_words(floor, message, rule, "corrected" in owed)
    lead = {"MEMBER": "START_THE_MEMBER", "CLOSED": "EXPORT_THE_REPORT"}.get(
        rule, "SEND_THE_CORRECTED_MESSAGE" if "corrected" in owed else "READ_THE_FLOOR"
    )
    words |= {"next_action": lead, "next_requests": floor_read | owed}
    if (ruling := _ruled(floor, message)) is not None:
        # The challenge is ruled already: the ruling stands, and no correction changes it.
        return words | {"ruled_by": ruling["id"]}
    if rule not in {"SHAPE", "REPLY_TO"} or message is None or floor is None:
        return words
    answers, sent = RULES[message.kind][2], message.model_dump(mode="json")
    ruled = {m["reply_to"] for m in floor.messages if m["kind"] == "RULING"}
    open_to = [
        m["id"]
        for m in floor.messages
        if answers in {"ANY", m["kind"], m.get("outcome")}
        and not (message.kind == "RULING" and m["id"] in ruled)
    ]
    outcomes: list[str | None] = [*sorted(OUTCOMES.get(message.kind, ()))] or [None]
    taken = [message.outcome] if message.outcome in outcomes else outcomes
    answered: list[str | None] = [None] if answers is None else list(open_to)
    request_of = {"operation": "COMMITTEE_SUBMIT", **member}
    words["next_action"] = "SEND_THE_CORRECTED_MESSAGE"
    words["next_requests"] = {
        ":".join(("corrected", *(str(part) for part in (to, outcome) if part))): {
            **request_of,
            "committee_message": {**sent, "reply_to": to, "outcome": outcome},
        }
        for to in answered
        for outcome in taken
    }
    return words


def context(floor: Floor, now: datetime) -> dict[str, Any]:
    """The exact publication's committee session and standing, without member credentials."""
    return {
        "goal_id": str(floor.goal_id),
        "update_task_id": floor.opened["update_task_id"],
        "review_selector": dict(floor.opened["selector"]),
        "native_host": floor.opened["session"][0],
        "native_session_id": floor.opened["session"][1],
        "stage": floor.stage(now),
        "verdict": verdict(floor, now),
        "specialists": [CARDS[r] for r in ROLES[1:]],
    }


def state(floor: Floor, role: str | None, seen: int, now: datetime) -> dict[str, Any]:
    """The floor as a member, or the person's page, reads it after `seen` messages.

    A stance is withheld until the reveal, but a member reads its own. `for_you` names what is
    addressed to the reader: the revealed stances, a challenge for the PM, a reply or ruling on
    its message, or the close.
    """
    summary = context(floor, now) | agenda_view(floor)
    stage, held = summary["stage"], floor.opened["holdings"]
    authors = {m["id"]: m["role"] for m in floor.messages}
    visible = [m for m in floor.messages if stage != "STANCES" or m["role"] == role]

    def addressed(m: Mapping[str, Any]) -> bool:
        aimed = {authors.get(str(m.get("reply_to"))), *map(authors.get, m["targets"])} - {None}
        pm = role == "PM" and m["kind"] == "CHALLENGE"
        return stage == "CLOSED" or m["kind"] == "STANCE" or pm or role in aimed

    answers = {m["reply_to"]: m for m in floor.messages if m["kind"] == "PERSON_ANSWER"}
    items = [m for m in floor.messages if m.get("outcome") == "FOR_THE_PERSON"]
    return {
        **summary,
        **{k: floor.opened[k] for k in ("strategy_package_id", "schedule")},
        **{k: floor.opened[k] for k in ("stances_close_at", "closes_at", "holdings")},
        "session": floor.opened["session"],
        "labels": floor.opened.get("labels") or {},
        "holdings_count": sum(1 for h in held if h["weight"] > 0),
        "stances_in": sum(1 for m in floor.messages if m["kind"] == "STANCE"),
        "settled": settled(floor, now),
        "member_count": len(ROLES),
        "members": {r: _member(floor, r, stage) for r in ROLES},
        "member_cards": dict(CARDS),
        "member_agents": {r: member_agent(floor, r) for r in ROLES},
        "tension_points": _points(floor, stage, summary.get("agenda", {})),
        "questions": QUESTIONS,
        "facts": floor.opened.get("facts", []),
        "messages": [
            {
                **m,
                "text": render(m["text"], held, floor.opened.get("facts", ())),
                "target_text": render(", ".join(m["targets"]), held),
                "row": floor_row(floor, m),
            }
            for m in visible
        ],
        "message_count": len(visible),
        "first_message_at": visible[0]["at"] if visible else None,
        "last_message_at": visible[-1]["at"] if visible else None,
        "standing_dissents": [m["id"] for m in standing_dissents(floor)],
        "for_you": [
            m["id"]
            for m in visible
            if role is not None and int(m["id"][1:]) > seen and m["role"] != role and addressed(m)
        ],
        "person_items": [
            {
                "id": m["id"],
                "question": render(m["text"], held),
                "asked_at": m["at"],
                "answer": answers.get(m["id"], {}).get("text"),
                "answer_at": answers.get(m["id"], {}).get("at"),
                "relay_word": RELAY_WORD,
            }
            for m in items
        ],
        # Only others' messages advance it: a member's own may outnumber a stance still withheld.
        "seen": max([seen, *(int(m["id"][1:]) for m in visible if m["role"] != role)]),
        "delivery": delivery(floor),
    }


def verdict(floor: Floor, now: datetime) -> dict[str, Any] | None:
    """The PM's verdict; the person decides when the floor closed without one."""
    given = next((m for m in floor.messages if m["kind"] == "VERDICT"), None)
    if given is not None:
        return {"outcome": given["outcome"], "text": given["text"], "at": given["at"]}
    if floor.stage(now) == "CLOSED":
        return {
            "outcome": "FOR_THE_PERSON",
            "text": "The PM gave no verdict.",
            "at": None,
            "text_word": "The PM gave no verdict.",
        }
    return None


def member_agent(floor: Floor, role: str) -> str:
    """The member's native identity under the session that opened its thread, with its unit."""
    for message in reversed(floor.messages):
        if message["role"] == role and message.get("author_session"):
            return str(message["author_session"])
    session, unit = floor.opened["session"][1], floor.opened.get("unit")
    if role in {"PM", "HOST"}:
        return str(session) if role == "PM" else "alphalattice-host"
    if floor.opened.get("member_sessions"):
        return ""  # Not observed yet; never invent a child identity for a new floor.
    return f"{session}:{_card(floor, role)}" + (f":{unit}" if unit else "")


def _card(floor: Floor, role: str) -> str:
    """The card a thread's member speaks as: a seat's on the floor, a reviewer's on its bundle."""
    review = REVIEWERS if "bundle_reference" in floor.opened else {}
    return {**CARDS, **review, "HOST": "alphalattice_host"}[role]


def floor_row(floor: Floor, message: Mapping[str, Any]) -> dict[str, Any]:
    """A revealed Team row, with the recorded speaker and recipient under the lead's session."""
    vendor, session = floor.opened["session"]
    answered = next((m for m in floor.messages if m["id"] == message.get("reply_to")), None)
    reply = (
        {"reply_to": _qualified(floor, answered["id"]), "recipient_id": _speaker(floor, answered)}
        if answered
        else {}
    )
    if message["kind"] == "ISSUE":  # from the lead to the member the bundle is for
        reply = {"recipient_id": member_agent(floor, floor.opened["role"])}
    task = floor.opened["update_task_id"]
    return {
        "event_kind": "NATIVE_COORDINATION_MESSAGE",
        "producer_id": "alphalattice-committee",
        "producer_session": task.replace("-", ""),
        "producer_sequence": int(message["id"][1:]),
        "occurred_at": message["at"],
        "summary": render(message["text"], floor.opened.get("holdings", ())),
        "subject": {
            "native_host": vendor,
            "native_session_id": session,
            "native_agent_id": _speaker(floor, message),
            "role": _card(floor, message["role"]),
            "message_kind": str(message["kind"]).lower(),
            "message_id": _qualified(floor, message["id"]),
            "reference": task,
            "input_channel": REVIEW_CHANNEL if "bundle_reference" in floor.opened else CHANNEL,
            "authorship_basis": "HOST_ACCEPTED",
            **({"relay_word": RELAY_WORD} if message["kind"] == "PERSON_ANSWER" else {}),
            **reply,
            **_references(floor, message),
        },
    }


def _speaker(floor: Floor, message: Mapping[str, Any]) -> str:
    """Own session joins usage; the card fallback reads legacy records only, until they retire."""
    if sender := message.get("author_session"):
        return str(sender)
    return str(member_agent(replace(floor, messages=[]), message["role"]))


def _qualified(floor: Floor, number: str) -> str:
    """A message's id as the Session reads it: a review thread's prefixed by its thread, so the
    M1 of each bundle's thread names its own message."""
    thread = floor.opened["update_task_id"]
    return f"{thread[:8]}-{number}" if "bundle_reference" in floor.opened else number


def _references(floor: Floor, message: Mapping[str, Any]) -> dict[str, str]:
    """A review row's bundle and, on its acceptance, the answer it accepted: what Outputs opens."""
    if "bundle_reference" not in floor.opened:
        return {}
    found = {"bundle_reference": floor.opened["bundle_reference"]}
    found["answer_reference"] = message.get("answer_reference")
    return {k: str(v) for k, v in found.items() if v}


def unfiled(floor: Floor, now: datetime) -> list[dict[str, Any]]:
    """The revealed messages not yet filed, as their goal conversation rows; none while blind."""
    if "selector" in floor.opened and floor.stage(now) == "STANCES":
        return []
    return [
        floor_row(floor, m)
        for m in floor.messages
        if m["id"] not in floor.filed
        and (m["kind"] != "ISSUE" or member_agent(floor, floor.opened["role"]))
    ]


def standing_messages(floor: Floor) -> dict[str, dict[str, str]]:
    """What each member still holds, keyed by the message that holds it."""
    # A challenge stands while a ruling rejects it or none answers it; a position stands until
    # its member changes it on that point or a ruling adopts or hands the point to the person.
    ruled = {m["reply_to"]: m["outcome"] for m in floor.messages if m["kind"] == "RULING"}
    open_ = {m["id"] for m in floor.messages if ruled.get(m["id"], "REJECT") == "REJECT"}
    settled = {
        t
        for m in floor.messages
        if m["kind"] == "CHALLENGE" and m["id"] not in open_
        for t in m["targets"]
    }
    since: dict[tuple[str, str], tuple[str, str]] = {}
    for m in floor.messages:
        for point, position in m["positions"].items():
            if since.get((m["role"], point), ("", ""))[1] != position:
                since[(m["role"], point)] = (m["id"], position)
    held = {
        m["id"]: dict[str, str]()
        for m in floor.messages
        if m["kind"] == "CHALLENGE" and m["id"] in open_
    }
    for (_role, point), (message, position) in since.items():
        if point not in settled:
            held.setdefault(message, {})[point] = position
    return {m["id"]: held[m["id"]] for m in floor.messages if m["id"] in held}


def standing_dissents(floor: Floor) -> list[dict[str, Any]]:
    """The CRO's standing challenges, and its messages that hold an objection to a point."""
    held = standing_messages(floor)
    return [
        m
        for m in floor.messages
        if m["role"] == "CRO"
        and m["id"] in held
        and (m["kind"] == "CHALLENGE" or "OBJECT" in held[m["id"]].values())
    ]


def commentary(floor: Floor, now: datetime) -> list[dict[str, Any]] | None:
    """The closed floor as the delivery's commentary; None while it is open.

    The verdict first, then each standing CRO dissent in the CRO's own words, each member's
    final view and every earlier message of it that still holds a position
    (`standing_messages`), what was handed to the person with their relayed answer, and each
    ruling.
    """
    closed = verdict(floor, now)
    if closed is None:
        return None
    answers = {m["reply_to"]: m["text"] for m in floor.messages if m["kind"] == "PERSON_ANSWER"}
    refs = {p["alias"]: [p["alias"], *p["targets"]] for p in floor.opened["tension_points"]}
    refs |= {
        m["id"]: [*m["targets"], *([m["reply_to"]] if m["reply_to"] else [])]
        for m in floor.messages
    }

    def targets(*aliases: str) -> str:
        pending, named = list(aliases), dict[str, None]()
        for alias in pending:
            if alias not in named:
                named[alias] = None
                pending.extend(refs.get(alias, ()))
        return render(
            ", ".join(a for a in named if not a.startswith("M")), floor.opened["holdings"]
        )

    named = {m["id"]: targets(m["id"]) for m in floor.messages}
    items: list[tuple[str, dict[str, str], str, dict[str, Any]]] = [
        (
            "Committee verdict: {outcome}",
            {"outcome": closed["outcome"]},
            closed["text"],
            {"text_word": closed.get("text_word")},
        ),
        *(
            (
                "CRO dissent stands on {targets}" if named[m["id"]] else "CRO dissent stands",
                {"message": m["id"], "targets": named[m["id"]]},
                m["text"],
                {},
            )
            for m in standing_dissents(floor)
        ),
        *_views(floor, targets),
        *(
            (
                "Handed to the person: {targets}" if named[m["id"]] else "Handed to the person",
                {"message": m["id"], "targets": named[m["id"]]},
                f"{m['text']} {RELAY_WORD}: " + answers.get(m["id"], "not yet given."),
                {
                    "relay_word": RELAY_WORD,
                    "question": render(m["text"], floor.opened["holdings"]),
                    "answer": answers.get(m["id"]),
                    "answer_present": m["id"] in answers,
                },
            )
            for m in floor.messages
            if m.get("outcome") == "FOR_THE_PERSON"
        ),
        *(
            (
                "PM ruling on {targets}: {outcome}" if named[m["id"]] else "PM ruling: {outcome}",
                {"message": m["reply_to"], "targets": named[m["id"]], "outcome": m["outcome"]},
                m["text"],
                {},
            )
            for m in floor.messages
            if m["kind"] == "RULING"
        ),
    ]
    if disclosed := agenda_view(floor).get("agenda"):
        items.append(
            (
                "Committee agenda",
                {},
                f"Agenda: {len(disclosed['aliases'])}; below the agenda line: "
                f"{disclosed['below_count']}. {disclosed['ranking']}",
                {"agenda": disclosed},
            )
        )
        items.extend(
            (
                "Raised by PM"
                if p["alias"] in disclosed["raised"]
                else "Agenda point"
                if p["alias"] in disclosed["aliases"]
                else "Below the agenda line, no stance required",
                {},
                f"{p['alias']}: {p['text']}",
                {"point": p["alias"]},
            )
            for p in floor.opened["tension_points"]
        )
    held = floor.opened["holdings"]
    return [
        {
            "attribution": word.format(**words),
            "text": render(text, held, floor.opened.get("facts", ())),
            "attribution_word": word,
            "attribution_words": words,
            **details,
        }
        for word, words, text, details in items
    ]


def _passed(floor: Floor, role: str) -> list[tuple[Any, ...]]:
    """A member's yielded turns, each read in its words."""
    return [
        (PASSED, {"member": role, "message": m["id"]}, m["text"], {})
        for m in floor.messages
        if m["role"] == role and m["kind"] == "PASS"
    ]


def _views(floor: Floor, targets: Callable[..., str]) -> list[tuple[Any, ...]]:
    """Each member's final view first, then each earlier message of it that still holds a
    position or a challenge (`standing_messages`), each verbatim with its id, the points it
    holds and its positions; a CRO dissent is shown as one already."""
    last = {m["role"]: m for m in floor.messages if m["kind"] in _VIEWS}
    rulings = [m for m in floor.messages if m["role"] == "PM" and m["kind"] == "RULING"]
    if rulings:
        last["PM"] = rulings[-1]
    held, shown = standing_messages(floor), {m["id"] for m in standing_dissents(floor)}
    items: list[tuple[Any, ...]] = []
    for r in ROLES:
        items += _passed(floor, r)
        final = last.get(r)
        if final is None:
            silent = {"text_word": "Not addressed."}
            items.append(("{member}, final view", {"member": r}, "Not addressed.", silent))
        earlier = [
            m
            for m in floor.messages
            if m["role"] == r and m["id"] in held and m is not final and m["id"] not in shown
        ]
        for m in ([final] if final else []) + earlier:
            on = targets(*(held.get(m["id"]) or [m["id"]]))
            word = "{member} still holds on {targets}" if on else "{member} still holds"
            words = {"member": r, "message": m["id"], "targets": on}
            items.append(
                (
                    "{member}, final view" if m is final else word,
                    words,
                    m["text"],
                    {"positions": held.get(m["id"], {})},
                )
            )
    return items


def holdings(readback: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The date's held and exited names by weight (H1 the largest).

    Each carries its component scores at formation from the publication's sealed input.
    """
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioUpdatePublication as Publication,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        portfolio_update_positions,
    )

    value = Publication.model_validate(readback["publication"])
    history = tuple(Publication.model_validate(v) for v in readback.get("history", []))
    positions = portfolio_update_positions(value, history)
    previous = positions.preceding or positions.weights
    book = None if value.pending_proposal is None else value.pending_proposal.input
    components = getattr(book, "components", None) or (() if book is None else (book,))
    names = [f"C{n}" for n in range(1, len(components) + 1)]  # a component's floor alias
    scores: dict[str, dict[str, float]] = {}
    for name, component in zip(names, components, strict=False):
        for listing, score in zip(component.ordered_listing_ids, component.scores, strict=False):
            if score >= 0:  # a name the component did not score reads -1
                scores.setdefault(listing, {})[str(name)] = score
    rows = [
        {
            "listing_id": listing,
            "name": label,
            "weight": weight,
            "change": weight - previous[i],
            "scores": scores.get(listing, {}),
        }
        for i, ((listing, label), weight) in enumerate(
            zip(readback["listing_labels"].items(), positions.weights, strict=True)
        )
        if weight > 0 or previous[i] > 0
    ]
    rows.sort(key=lambda r: (-r["weight"], -abs(r["change"]), r["name"]))
    return [{"alias": f"H{n}", **row} for n, row in enumerate(rows, 1)]


def role_lines(
    role: str,
    readback: Mapping[str, Any],
    evidence: Mapping[str, Any],
    floor: Floor | None,
    now: datetime,
) -> list[str] | None:
    """One role's view of a date's positions for its bundle (`alphalattice bundle prepare --role`).

    While a committee floor is open on them, it adds the floor's points and how the role
    speaks there. None for a role with no view of a date's positions; the CRO's is its
    committee bundle, so it has one, and a key, only while a floor is open.
    """
    if floor is not None and floor.stage(now) == "CLOSED":
        floor = None
    if role not in {"ALPHA", "RISK", "DATA"} and (role != "CRO" or floor is None):
        return None
    held = floor.opened["holdings"] if floor else holdings(readback)
    publication = readback["publication"]
    table = [
        f"{h['alias']} {h['name']}: weight {h['weight']:.2%}, change {h['change']:+.2%}"
        for h in held
    ]
    if role == "ALPHA":
        view = [
            f"{h['alias']} {h['name']}: weight {h['weight']:.2%}; component scores at formation "
            "(0 to 1, higher ranks better): "
            + (", ".join(f"{k} {v:.2f}" for k, v in sorted(h["scores"].items())) or "none")
            for h in held
        ]
    elif role == "RISK":
        risk = floor.opened.get("date_risk") if floor else readback.get("date_risk")
        linked = status_word(publication.get("risk_status"))
        linked = f"Historical linked Risk study for this proposal: {linked}. "
        view = [risk_view(risk, held) + linked + concentration(held), *table]
    elif role == "CRO":
        gap = evidence_gap(evidence)
        view = [
            review_now(readback),
            *(
                [f"No Evidence stands ({gap}): {evidence.get('explanation') or ''}"]
                if gap
                else issue_lines(evidence, held)
            ),
        ]
    else:
        view = [
            f"Observed through {publication.get('observed_through')}; the update's target "
            f"session {(readback.get('update') or {}).get('target_session')}.",
            *table,
        ]
    schedule = (publication.get("pending_proposal") or {}).get("schedule") or {}
    head = (
        f"Strategy {readback.get('strategy_package_id')}: positions entered "
        f"{schedule.get('entry_session')}, decided at the {schedule.get('formation_session')} "
        "close; research positions, never orders or advice."
    )
    if floor is None or role == "DATA":
        return [head, *view]
    return [head, *_asked(role, floor), *view]


def _asked(role: str, floor: Floor) -> list[str]:
    """A role's questions, each with the floor's facts that answer it."""
    said = floor.opened.get("facts", ())
    below = (agenda_view(floor).get("agenda") or {}).get("below", ())
    bounded = floor.opened.get("agenda")
    points = [
        *(
            [
                f"Agenda: {', '.join(bounded['aliases'])}. {bounded['ranking']} "
                f"Below the agenda line: {len(below)} points, no stance required."
            ]
            if bounded
            else []
        ),
        *(
            f"{p['alias']}: {p['text']}"
            + (" Below the agenda line, no stance required." if p["alias"] in below else "")
            for p in floor.opened["tension_points"]
        ),
    ]
    return points + [
        f"{q} {words} " + " ".join(f"{f['alias']}: {f['text']}" for f in said if f["question"] == q)
        for q, words in QUESTIONS.get(role, {}).items()
    ]


def bundle_parts(
    role: str,
    dated: tuple[dict[str, Any], dict[str, Any]] | None,
    store: GoalStore,
    now: datetime,
    directory: str | None = None,
) -> tuple[list[str] | None, tuple[str, ...] | None]:
    """A role's view of a date's positions and its route: the open floor's, or its review's."""
    if dated is None:
        return None, review_route(store, role, directory)
    floor = floor_of(store, str(dated[0]["task_id"]))
    return role_lines(role, *dated, floor, now), route(role, floor, now, directory)


def route(
    role: str, floor: Floor | None, now: datetime, directory: str | None = None
) -> tuple[str, ...] | None:
    """The member's keyed route on an open floor, with the bundle path as supplied."""
    if floor is None or role not in ROLES[1:] or floor.stage(now) == "CLOSED":
        return None
    task, key = floor.opened["update_task_id"], floor.key(role)
    message = (
        f'"{(Path(directory) / "message.yaml").as_posix()}"' if directory else "<message.yaml>"
    )
    point = "agenda" if "agenda" in floor.opened else "tension"
    return (
        f"On the committee's floor you are {role}. Send one STANCE first (the round is blind); "
        f"its position on each {point} point (T1, never a question) is about the date's "
        "positions: SUPPORT, they may stand; OBJECT, they should not stand as they are; RESERVE, "
        "this bundle cannot tell, citing by S alias the facts it rests on.",
        "Once the stances are revealed, take your one discussion turn: a CHALLENGE or REPLY, or, "
        "before any challenge or reply, a PASS (kind PASS, a line saying you have nothing to add, "
        "no targets or positions) that yields it and ends your turn; the floor settles when every "
        "member has spoken or passed. "
        f"You may send up to {MESSAGE_CAP} CHALLENGE or REPLY in all, each naming its targets by "
        "alias (T1, H3, M4, a fact by its S alias); type no other digit, since the Host renders "
        "every number. "
        "The PM may approve (PROCEED), approve with conditions (PROCEED_WITH_NOTES), decline "
        "endorsement (DECLINE), or hand the decision to the person (FOR_THE_PERSON). "
        "Write each message as a YAML file in this bundle directory "
        "(kind, text, positions, targets, reply_to), send it "
        f"with `{COMMAND} committee submit "
        f"--update {task} --role {role} --key {key} --file {message}` and follow what is "
        f"addressed to you with `{COMMAND} committee wait --update {task} "
        f"--role {role} --key {key} "
        "--seen <seen>`. Run these from your own session; the key is yours alone. "
        "A refused message names its rule; where only "
        "its shape is wrong, send the corrected message the refusal offers.",
        "This route is the bundle's only final action: there is no answer file and nothing for "
        "the lead to submit. It runs these two commands, so the lead starts you as a general "
        "subagent.",
    )


def pm_route(floor: Floor) -> tuple[str, ...]:
    """The PM's own keyed route, returned with its key: nothing is read from source."""
    task, key = floor.opened["update_task_id"], floor.key("PM")
    send = f"`{COMMAND} committee submit --update {task} --role PM --key {key} --file "
    send += "<message.yaml>`"
    return (
        "Prepare the three members' bundles, then start the three together, each in its own "
        "session from its bundle, so their stances arrive at once.",
        "As PM send your own STANCE (the round is blind), then a RULING on each CHALLENGE "
        "addressed to you (ADOPT, REJECT or FOR_THE_PERSON). Your wait returns `settled` once "
        "every member has spoken or passed its turn and no challenge stands unruled: give the "
        "VERDICT then: PROCEED, PROCEED_WITH_NOTES, DECLINE or FOR_THE_PERSON.",
        f"Write each message as a YAML file (kind, text, positions, targets, reply_to; "
        f"`{COMMAND} schema show COMMITTEE_SUBMIT` shows every field) and send it with {send}. "
        f"Follow what is addressed to you with `{COMMAND} committee wait --update {task} "
        f"--role PM --key {key} --seen <seen>`; read the whole floor with "
        f"`{COMMAND} committee show --update {task}`.",
    )


def thread_id(reference: str) -> str:
    """The thread a review bundle opens, named by the bundle's own record."""
    return str(UUID(hex=sha256(reference.encode()).hexdigest()[:32]))


def review_route(store: GoalStore, role: str, directory: str | None) -> tuple[str, ...] | None:
    """A review bundle's final action, its member's own submit with its key; none outside a goal."""
    goal = getattr(REQUEST_PROVENANCE.get(), "goal_id", None)
    if role not in REVIEWERS or goal is None or directory is None:
        return None
    thread = thread_id(bundle_slot(directory))
    with store.lock:
        held = store.committee(thread, UUID(goal)) or {}
    answer = (Path(directory) / "answer.json").as_posix()
    submit = f'{COMMAND} bundle submit --dir "{Path(directory).as_posix()}" --file "{answer}"'
    return (
        "## Your route",
        f"You submit your own answer: `{submit} --key {member_key(held['seed'], thread, role)}`. "
        "The key is yours alone and admits only this bundle's answer; run no other command.",
        "Its answer is the Host's verdict. CORRECT names the items to fix: fix or remove only "
        "those in the same answer file and run the same command again. ACCEPTED or DONE ends "
        "your work whatever the command's exit code, which follows the review's Task: return its "
        "receipt to your lead in one line.",
    )


def issued(store: GoalStore, body: dict[str, Any], now: datetime, file: Any) -> dict[str, Any]:
    """A prepared review bundle's thread opened with its ISSUE, from the lead to its member."""
    role, reference = body.get("agent_role"), body.get("bundle_reference")
    provenance = REQUEST_PROVENANCE.get()
    if role not in REVIEWERS or not isinstance(reference, str) or provenance is None:
        return body
    thread = thread_id(reference)
    with store.lock:
        held = store.committee(thread)
        if held is None or thread_of(store, thread) is not None:
            return body
        unit = body.get("task_id")
        opened = {
            "update_task_id": thread,
            "bundle_reference": reference,
            "unit": unit,
            "member_sessions": True,
        }
        opened |= {"role": role, "session": [provenance.vendor, provenance.session]}
        store.attribute(UUID(held["goal_id"]), {"committee_open": opened})
        floor = Floor(UUID(held["goal_id"]), held["seed"], opened, [], frozenset())
        text = f"{'Analyst' if role == 'ANALYST' else role} bundle issued."  # its Task: the thread
        _append(store, floor, now, role="PM", kind="ISSUE", text=text, references=[reference])
    with suppress(ValueError, OSError):
        file(thread, now)  # filed as it is written, never with the first answer
    return body


def _append(store: GoalStore, floor: Floor, now: datetime, **message: Any) -> str:
    """One message appended to a thread already read, numbered after those it holds."""
    number, thread = f"M{len(floor.messages) + 1}", floor.opened["update_task_id"]
    entry: dict[str, Any] = {"update_task_id": thread, "id": number, "at": now.isoformat()}
    entry |= {"targets": [], "positions": {}, "reply_to": None, **message}
    store.attribute(floor.goal_id, {"committee_message": entry})
    floor.messages.append(entry)
    return number


def _admit_member(
    floor: Floor,
    role: str,
    key: str | None,
    provenance: RequestProvenance | None,
    *,
    same: Mapping[str, Any] | None = None,
    ended: bool = False,
    review_receipt: bool = False,
) -> Floor:
    """One capability and sender rule; receipt reads keep their author.

    Sessions are client-declared and may be shared; the member key remains the capability.
    """
    sender = getattr(provenance, "session", None)
    rule = "KEY" if key is not None and not floor.admits(role, key) else None
    if rule is None and same is not None and (key is not None or review_receipt):
        if same.get("author_session") and sender not in {
            same["author_session"],
            floor.opened["session"][1],
        }:
            raise ValueError("committee.message_refused:MEMBER")
        return replace(floor, author_session=same.get("author_session"))
    if rule is None and ended:
        rule = "ENDED"
    parent = floor.opened["session"][1]
    if rule is None and (key is None or not sender or (role == "PM" and sender != parent)):
        rule = "MEMBER"
    if rule is not None:
        raise ValueError(f"committee.message_refused:{rule}")
    return replace(floor, author_session=sender)


def admit(
    store: GoalStore, member: tuple[str, str, Any], key: str | None
) -> Floor | dict[str, Any] | None:
    """Admit a review answer by key; only its accepted answer permits a keyless receipt read."""
    role, reference, answer = member
    floor = thread_of(store, thread_id(reference)) if role in REVIEWERS else None
    if floor is None:
        return None
    accepted, digest = _accepted(floor), str(canonical_hash(answer))
    same = next((m for m in floor.messages if accepted == digest == m.get("answer_digest")), None)
    try:
        return _admit_member(
            floor,
            role,
            key,
            REQUEST_PROVENANCE.get(),
            same=same,
            ended=accepted is not None,
            review_receipt=True,
        )
    except ValueError as error:
        rule = str(error).rpartition(":")[2]
    lead = "RETURN_THE_RECEIPT_TO_YOUR_LEAD" if rule == "ENDED" else "START_THE_MEMBER"
    code = f"committee.message_refused:{rule}"
    words = {"detail": RULE_WORDS[rule], "rule": rule, "next_action": lead}
    return {
        "status": "REFUSED",
        "failure_code": code,
        **words,
        "next_requests": {"schema": schema_request("AGENT_ANSWER_SUBMIT")},
    }


def threaded(store: GoalStore, bundle: Any) -> bool:
    """Whether a bundle's answers are a review member's own, filed on its thread, never relayed."""
    reference = getattr(bundle, "record_hash", None)
    if getattr(bundle, "role", None) not in REVIEWERS or not isinstance(reference, str):
        return False
    return store.committee(thread_id(reference)) is not None


def accepted_task(store: GoalStore, thread: str) -> str | None:
    """The Task a review thread's acceptance admitted its answer into; None while it is open."""
    floor = thread_of(store, thread)
    taken = [m for m in (floor.messages if floor else []) if m["kind"] == "ACCEPTANCE"]
    references = taken[0].get("references") or [] if taken else []
    return str(references[-1]) if len(references) > 1 else None


def _accepted(floor: Floor) -> str | None:
    """The digest of the answer the thread's acceptance took; None while it is open."""
    said = {m["id"]: m for m in floor.messages}
    taken = next((m for m in floor.messages if m["kind"] == "ACCEPTANCE"), None)
    return None if taken is None else said[taken["reply_to"]].get("answer_digest")


def answered(
    store: GoalStore, floor: Any, member: Any, body: Any, now: datetime, file: Any
) -> None:
    """The member's answer and the Host's verdict on it, on the thread `admit` read."""
    (role, _reference, answer), status = member, body.get("status")
    if not isinstance(floor, Floor) or status not in {"CORRECT", "ACCEPTED", "DONE"}:
        return
    # Filed on the bundle's thread in the lead's session, the member its author, whoever sent it.
    session, thread = floor.opened["session"], floor.opened["update_task_id"]
    own = {"author_session": floor.author_session} if floor.author_session else {}
    body["conversation"] = {"status": "RECORDED_ON_THREAD", "thread": thread, "host": session[0]}
    body["conversation"] |= {
        "session_id": session[1],
        "author": _speaker(floor, own | {"role": role}),
    }
    with store.lock:
        if _accepted(floor) is not None:
            return  # the accepted answer read again: its receipt, said once
        last = floor.messages[-1]["id"] if floor.messages else None  # its issue, or correction
        said = {"role": role, "kind": "ANSWER", "text": "Answer submitted.", "reply_to": last}
        said["answer_digest"] = str(canonical_hash(answer))
        said |= own
        verdict: dict[str, Any] = {"role": "HOST", "reply_to": _append(store, floor, now, **said)}
        if status == "CORRECT":
            problems, left = len(body.get("problems") or ()), body.get("rounds_left")
            text = f"Correct the {problems} named items; {left} rounds left."
            _append(store, floor, now, kind="CORRECTION", text=text, **verdict)
        else:
            refs = [str(body[k]) for k in ("answer_reference", "task_id") if body.get(k)]
            text = "Accepted." if status == "ACCEPTED" else "Rounds spent; the valid part stands."
            verdict |= {"outcome": status, "references": refs}
            answer_reference = body.get("answer_reference")
            _append(
                store,
                floor,
                now,
                kind="ACCEPTANCE",
                text=text,
                answer_reference=answer_reference,
                **verdict,
            )
    with suppress(ValueError, OSError):
        file(floor.opened["update_task_id"], now)


def render(
    text: str, held: Sequence[Mapping[str, Any]], said: Sequence[Mapping[str, Any]] = ()
) -> str:
    """Each holding alias with its name and weight, and each fact alias with its words."""
    named = {h["alias"]: f"{h['alias']} ({h['name']}, {h['weight']:.2%})" for h in held}
    named |= {f["alias"]: f"{f['alias']} ({f['text']})" for f in said}
    return ALIAS.sub(lambda m: named.get(m.group(0), m.group(0)), text)


def _points(floor: Floor, stage: str, bounded: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each tension point with its last ruling's state.

    A CRO challenge on it that is rejected or never ruled leaves its dissent standing
    (`dissents`, the challenges' ids).
    """
    rulings = {m["reply_to"]: m for m in floor.messages if m["kind"] == "RULING"}
    standing = {m["id"] for m in standing_dissents(floor)}
    states = {"ADOPT": "ADOPTED", "REJECT": "REJECTED", "FOR_THE_PERSON": "FOR_THE_PERSON"}
    out = []
    order = {
        a: n
        for n, a in enumerate(
            (*bounded.get("aliases", ()), *bounded.get("raised", ()), *bounded.get("below", ()))
        )
    }
    points = floor.opened["tension_points"]
    for point in sorted(points, key=lambda p: order.get(p["alias"], 0)):
        aimed = [
            m for m in floor.messages if m["kind"] == "CHALLENGE" and point["alias"] in m["targets"]
        ]
        ruled = [rulings[c["id"]] for c in aimed if c["id"] in rulings]
        dissents = [c["id"] for c in aimed if c["id"] in standing]
        last = ruled[-1] if ruled else {}
        out.append(
            {
                **point,
                "target_text": render(", ".join(point["targets"]), floor.opened["holdings"]),
                "words": {
                    k: render(v, floor.opened["holdings"]) if isinstance(v, str) else v
                    for k, v in point.get("words", {}).items()
                },
                "state": states.get(
                    last.get("outcome", ""), "NOT_ADDRESSED" if stage == "CLOSED" else "OPEN"
                )
                if point["alias"] not in bounded.get("below", ())
                else "BELOW_AGENDA",
                "ruling_id": last.get("id"),
                "dissent_stands": bool(dissents),
                "dissents": dissents,
            }
        )
    return out


def agenda_view(floor: Floor) -> dict[str, Any]:
    """Disclose the seeded agenda and points the PM brought into debate, from the same ledger."""
    bounded = floor.opened.get("agenda")
    if bounded is None:
        return {}
    raised = {
        a
        for m in floor.messages
        if m["role"] == "PM" and m["kind"] == "REPLY"
        for a in m["positions"]
        if a in bounded["below"]
    }
    below = [a for a in bounded["below"] if a not in raised]
    return {
        "agenda": {
            **bounded,
            "raised": [a for a in bounded["below"] if a in raised],
            "below": below,
            "below_count": len(below),
        }
    }


def _member(floor: Floor, role: str, stage: str) -> str:
    """A member's state in the current round: IN, NOT_YET, or NOT_ADDRESSED once it passed.

    At the close, a member that spoke at all is IN.
    """
    said = {m["kind"] for m in floor.messages if m["role"] == role}
    if stage == "CLOSED":
        return "IN" if said else "NOT_ADDRESSED"
    if stage == "STANCES":
        return "IN" if "STANCE" in said else "NOT_YET"
    if "STANCE" not in said:
        return "NOT_ADDRESSED"
    return "IN" if said - {"STANCE"} else "NOT_YET"
