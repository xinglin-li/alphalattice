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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, Final
from uuid import UUID

from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.cli_contract import RequestProvenance
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.interface.local_application.labels import label as installed_label
from alphalattice.interface.local_application.portfolio_research import (
    CommitteeMessage,
    PortfolioResearchOperationRequest,
)

ROLES: Final = ("PM", "ALPHA", "RISK", "CRO")
CARDS: Final = {r: f"alphalattice_{r.lower()}" for r in ROLES[1:]} | {"PM": "research_lead"}
"""Each member's card, the Team page's participant name."""
CHANNEL: Final = "PRODUCT_COMMITTEE"
"""The Host's own channel for floor rows; a client's event never takes it."""
RELAY_WORD: Final = "Relayed by the agent"
STANCE_MINUTES, FLOOR_MINUTES, MESSAGE_CAP = 10, 40, 3
RULES: Final[dict[str, tuple[tuple[str, ...], frozenset[str], str | None]]] = {
    "STANCE": (ROLES, frozenset({"STANCES"}), None),
    "CHALLENGE": (ROLES[1:], frozenset({"DEBATE"}), None),
    "REPLY": (ROLES, frozenset({"DEBATE"}), "ANY"),
    "RULING": (ROLES[:1], frozenset({"DEBATE"}), "CHALLENGE"),
    "VERDICT": (ROLES[:1], frozenset({"DEBATE"}), None),
    "PERSON_ANSWER": (ROLES[:1], frozenset({"DEBATE", "CLOSED"}), "FOR_THE_PERSON"),
}
"""The round state machine: each kind's senders, the stages it is sent in, and what it answers
(a challenge, any message, or an item the PM handed to the person)."""
_VIEWS: Final = frozenset({"STANCE", "CHALLENGE", "REPLY"})
ALIAS: Final = re.compile(r"\b[HTM][0-9]{1,4}\b")
"""A floor alias: a holding (H3), a tension point (T2) or a message (M4)."""
"""The kinds that carry a member's own view; a ruling, the verdict and a relay do not."""


@dataclass(frozen=True)
class Floor:
    """One update's floor as its goal's ledger holds it."""

    goal_id: UUID
    seed: str = field(repr=False)
    opened: dict[str, Any]
    messages: list[dict[str, Any]]
    filed: frozenset[str]

    def key(self, role: str) -> str:
        """Derive the member's key on this floor from the seed the Host keeps outside records."""
        named = f"{self.opened['update_task_id']}:{role}".encode()
        return hmac.new(bytes.fromhex(self.seed), named, sha256).hexdigest()[:32]

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


def floor_of(store: GoalStore, task: str) -> Floor | None:
    """The update's floor, from the goal whose ledger opened it; None when none was opened."""
    held = store.committee(task)
    if held is None:
        return None
    entries = store.attributed(UUID(held["goal_id"]))
    found = [
        dict(value)
        for e in entries
        for k in ("committee_open", "committee_message")
        if isinstance(value := e.get(k), Mapping) and value.get("update_task_id") == task
    ]
    if not found or "selector" not in found[0]:
        return None
    rows = {
        str(e.get("message_id"))
        for e in entries
        if e.get("input_channel") == CHANNEL and e.get("reference") == task
    }
    return Floor(UUID(held["goal_id"]), held["seed"], found[0], found[1:], frozenset(rows))


def open_floor(
    store: GoalStore,
    goal_id: UUID,
    readback: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    session: tuple[str, str],
    now: datetime,
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
        "strategy_package_id": readback.get("strategy_package_id"),
        "labels": readback.get("labels") or (label.model_dump() if label else {}),
        "schedule": proposal.get("schedule"),
        "stances_close_at": (now + timedelta(minutes=STANCE_MINUTES)).isoformat(),
        "closes_at": (now + timedelta(minutes=FLOOR_MINUTES)).isoformat(),
        "session": list(session),
        "holdings": held,
        "date_risk": readback.get("date_risk"),
        "tension_points": tension_points(held, evidence, readback.get("date_risk")),
    }
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
                store, task, (role, request.committee_key), request.committee_message, now
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
                store, UUID(provenance.goal_id), *publication, session=session, now=now
            )
            # Each specialist reads its bundle by `bundle prepare`, its key in it.
            return {
                "status": "COMMITTEE_OPEN",
                "stage": floor.stage(now),
                **({"pm_key": floor.key("PM")} if floor.opened["session"] == list(session) else {}),
                "next_requests": {
                    f"bundle:{member}": {
                        "operation": "AGENT_BUNDLE_PREPARE",
                        "agent_role": member,
                        "task_id": task,
                        "bundle_directory": None,
                    }
                    for member in ("ALPHA", "RISK", "CRO")
                },
            }
        found = floor_of(store, task)
        if found is None:
            raise ValueError("committee.not_open")
        if role is not None and not found.admits(role, request.committee_key):
            raise ValueError("committee.message_refused:ROLE")
    except ValueError as error:
        code = public_failure(error, "committee.refused")
        return {"status": "REFUSED", "failure_code": code, "refused": code, **explain(code)}
    with suppress(ValueError, OSError):  # a time box's reveal reaches the Team page on read
        file_floor()
    read = state(found, role, request.committee_seen or 0, now)
    offered = {"report": read["delivery"]} if read["stage"] == "CLOSED" else {}
    return {"status": "COMMITTEE_FLOOR", **read, "next_requests": offered}


def submit(
    store: GoalStore,
    task: str,
    member: tuple[str, str | None],
    message: CommitteeMessage,
    now: datetime,
) -> str:
    """Take one member's message, sent with its key, by the round rules.

    A message that breaks a rule is refused by that rule.

    Raises:
        ValueError: `committee.not_open`, or `committee.message_refused:<RULE>`.

    """
    with store.lock:
        floor = floor_of(store, task)
        if floor is None:
            raise ValueError("committee.not_open")
        (role, key), (senders, stages, answers) = member, RULES[message.kind]
        stage, mine = floor.stage(now), [m for m in floor.messages if m["role"] == role]
        known = {p["alias"] for p in floor.opened["tension_points"]}
        known |= {h["alias"] for h in floor.opened["holdings"]} | {m["id"] for m in floor.messages}
        target = next((m for m in floor.messages if m["id"] == message.reply_to), None)
        fits = target is not None and answers in {"ANY", target["kind"], target.get("outcome")}
        ruled = message.kind == "RULING" and any(
            m["kind"] == "RULING" and m["reply_to"] == message.reply_to for m in floor.messages
        )
        spoken = sum(m["kind"] != "STANCE" for m in mine)
        named = (*message.targets, *message.positions, *ALIAS.findall(message.text))
        typed = any(c.isnumeric() for c in ALIAS.sub("", message.text))
        for rule, broken in (
            ("ROLE", role not in senders or not floor.admits(role, key)),
            (stage, stage not in stages),
            ("STANCE_IN", message.kind == "STANCE" and any(m["kind"] == "STANCE" for m in mine)),
            ("CAP", message.kind != "STANCE" and role != "PM" and spoken >= MESSAGE_CAP),
            ("TARGET", any(t not in known for t in named)),
            ("REPLY_TO", answers is not None and (not fits or ruled)),
            # The Host renders every number; the person's own relayed words keep theirs.
            ("NUMBER", message.kind != "PERSON_ANSWER" and typed),
        ):
            if broken:
                raise ValueError(f"committee.message_refused:{rule}")
        number = f"M{len(floor.messages) + 1}"
        entry = {"update_task_id": task, "id": number, "role": role, "at": now.isoformat()}
        store.attribute(
            floor.goal_id, {"committee_message": {**entry, **message.model_dump(mode="json")}}
        )
        return number


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
    summary = context(floor, now)
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
        "member_count": len(ROLES),
        "members": {r: _member(floor, r, stage) for r in ROLES},
        "member_cards": dict(CARDS),
        "member_agents": {r: member_agent(floor, r) for r in ROLES},
        "tension_points": _points(floor, stage),
        "messages": [
            {
                **m,
                "text": render(m["text"], held),
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
        "delivery": {"operation": "EXPERIMENT_DELIVERY_EXPORT", **floor.opened["selector"]},
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
    """The recorded member's native identity under the session that opened its floor."""
    session = floor.opened["session"][1]
    return str(session) if role == "PM" else f"{session}:{CARDS[role]}"


def floor_row(floor: Floor, message: Mapping[str, Any]) -> dict[str, Any]:
    """One revealed message as the goal conversation row the Team page reads.

    It names its member's card under the lead's session, and the author it answers as recipient.
    """
    vendor, session = floor.opened["session"]
    answered = {m["id"]: m["role"] for m in floor.messages}.get(str(message.get("reply_to")))

    reply = (
        {"reply_to": message["reply_to"], "recipient_id": member_agent(floor, answered)}
        if answered
        else {}
    )
    task = floor.opened["update_task_id"]
    return {
        "event_kind": "NATIVE_COORDINATION_MESSAGE",
        "producer_id": "alphalattice-committee",
        "producer_session": task.replace("-", ""),
        "producer_sequence": int(message["id"][1:]),
        "occurred_at": message["at"],
        "summary": render(message["text"], floor.opened["holdings"]),
        "subject": {
            "native_host": vendor,
            "native_session_id": session,
            "native_agent_id": member_agent(floor, message["role"]),
            "role": CARDS[message["role"]],
            "message_kind": str(message["kind"]).lower(),
            "message_id": message["id"],
            "reference": task,
            "input_channel": CHANNEL,
            "authorship_basis": "HOST_ACCEPTED",
            **({"relay_word": RELAY_WORD} if message["kind"] == "PERSON_ANSWER" else {}),
            **reply,
        },
    }


def unfiled(floor: Floor, now: datetime) -> list[dict[str, Any]]:
    """The revealed messages not yet filed, as their goal conversation rows; none while blind."""
    if floor.stage(now) == "STANCES":
        return []
    return [floor_row(floor, m) for m in floor.messages if m["id"] not in floor.filed]


def standing_dissents(floor: Floor) -> list[dict[str, Any]]:
    """The CRO challenges no ruling adopted, including a challenge on a holding directly."""
    rulings = {m["reply_to"]: m["outcome"] for m in floor.messages if m["kind"] == "RULING"}
    return [
        m
        for m in floor.messages
        if m["kind"] == "CHALLENGE"
        and m["role"] == "CRO"
        and rulings.get(m["id"], "REJECT") == "REJECT"
    ]


def commentary(floor: Floor, now: datetime) -> list[dict[str, Any]] | None:
    """The closed floor as the delivery's commentary; None while it is open.

    The verdict first, then each standing CRO dissent in the CRO's own words, each member's
    final view, what was handed to the person with their relayed answer, and each ruling.
    """
    closed = verdict(floor, now)
    if closed is None:
        return None
    said = {m["role"]: m["text"] for m in floor.messages if m["kind"] in _VIEWS}
    answers = {m["reply_to"]: m["text"] for m in floor.messages if m["kind"] == "PERSON_ANSWER"}
    refs = {p["alias"]: [p["alias"], *p["targets"]] for p in floor.opened["tension_points"]}
    refs |= {
        m["id"]: [*m["targets"], *([m["reply_to"]] if m["reply_to"] else [])]
        for m in floor.messages
    }

    def targets(message: Mapping[str, Any]) -> str:
        pending, named = [message["id"]], dict[str, None]()
        for alias in pending:
            if alias not in named:
                named[alias] = None
                pending.extend(refs.get(alias, ()))
        return render(
            ", ".join(a for a in named if not a.startswith("M")), floor.opened["holdings"]
        )

    named = {m["id"]: targets(m) for m in floor.messages}
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
        *(
            (
                "{member}, final view",
                {"member": r},
                said.get(r, "Not addressed."),
                {} if r in said else {"text_word": "Not addressed."},
            )
            for r in ROLES
        ),
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
    held = floor.opened["holdings"]
    return [
        {
            "attribution": word.format(**words),
            "text": render(text, held),
            "attribution_word": word,
            "attribution_words": words,
            **details,
        }
        for word, words, text, details in items
    ]


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
    names = getattr(book, "component_ids", None) or [f"C{n}" for n in range(1, 9)]
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


def tension_points(
    held: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
    risk: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """What the committee argues.

    Held names its Evidence flags, the largest changes, the largest position, the largest
    contributors to the date's predicted risk (or why none stands), and a gap in the Evidence.
    """
    adverse = {
        h["alias"]: int(row["adverse_issue_count"])
        for row in evidence.get("issuer_rows") or ()
        for h in held
        if h["name"] in (row.get("tickers") or ()) and int(row.get("adverse_issue_count") or 0)
    }
    moved = [h["alias"] for h in sorted(held, key=lambda h: -abs(h["change"]))[:5]]
    gap = evidence_gap(evidence)
    points = [
        *(
            (
                "HELD_WITH_ADVERSE_EVIDENCE",
                "{holding} is held while its Evidence names {n} major negative(s).",
                {"holding": a, "n": n},
                [a],
            )
            for a, n in adverse.items()
        ),
        (
            "LARGEST_CHANGES",
            "The largest changes: {holdings}.",
            {"holdings": ", ".join(moved)},
            moved,
        ),
        (
            "LARGEST_POSITION",
            "The largest position, {holding}, and the five largest together.",
            {"holding": "H1"},
            [h["alias"] for h in held[:5]],
        ),
        *_risk_point(held, risk),
        *(
            [
                (
                    "EVIDENCE_GAP",
                    "No Evidence stands for these positions: {reason}.",
                    {"reason": gap},
                    [],
                )
            ]
            if gap
            else []
        ),
    ]
    return [
        {
            "alias": f"T{n}",
            "kind": kind,
            "text": word.format(**words),
            "word": word,
            "words": words,
            "targets": targets,
        }
        for n, (kind, word, words, targets) in enumerate(points, 1)
    ]


def evidence_gap(evidence: Mapping[str, Any]) -> str | None:
    """Why no Evidence stands for the positions (a refusal, or a state with no analysis)."""
    if evidence.get("status") == "REFUSED" or evidence.get("failure_code"):
        return str(evidence.get("failure_code") or "REFUSED")
    if evidence.get("evidence_as_of") is None:
        return str(evidence.get("state") or "NOT_PREPARED")
    return None


def role_lines(
    role: str,
    readback: Mapping[str, Any],
    evidence: Mapping[str, Any],
    floor: Floor | None,
    now: datetime,
) -> list[str] | None:
    """One role's view of a date's positions for its bundle (`bundle prepare --role`).

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
        sealed = f"Risk status sealed at publication: {publication.get('risk_status')}. "
        view = [sealed + _risk_view(risk, held) + _concentration(held), *table]
    elif role == "CRO":
        gap = evidence_gap(evidence)
        view = [
            f"Review of this proposal at publication: {publication.get('cro_status')}.",
            *(
                [f"No Evidence stands ({gap}): {evidence.get('explanation') or ''}"]
                if gap
                else [
                    f"{c.get('issue_handle')}: {c.get('cro_inference')} Severity if true: "
                    f"{c.get('severity_if_true')}."
                    for c in evidence.get("issue_cards") or ()
                ]
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
    return [head, *(f"{p['alias']}: {p['text']}" for p in floor.opened["tension_points"]), *view]


def bundle_parts(
    role: str,
    dated: tuple[dict[str, Any], dict[str, Any]] | None,
    store: GoalStore,
    now: datetime,
) -> tuple[list[str] | None, tuple[str, ...] | None]:
    """A role's view of a date's positions and, on an open floor, its route (`bundle prepare`)."""
    if dated is None:
        return None, None
    floor = floor_of(store, str(dated[0]["task_id"]))
    return role_lines(role, *dated, floor, now), route(role, floor, now)


def route(role: str, floor: Floor | None, now: datetime) -> tuple[str, ...] | None:
    """How a member speaks on an open floor: its bundle's one final action, with its own key.

    None with no open floor or for a role with no seat; the bundle then ends in its answer file.
    """
    if floor is None or role not in ROLES[1:] or floor.stage(now) == "CLOSED":
        return None
    task, key = floor.opened["update_task_id"], floor.key(role)
    return (
        f"On the committee's floor you are {role}. Send one STANCE first (the round is blind); "
        "its position on each point is about the date's positions: SUPPORT, they may stand; "
        "OBJECT, they should not stand as they are; RESERVE, this bundle cannot tell.",
        f"Then up to {MESSAGE_CAP} CHALLENGE or REPLY, each naming its targets by alias (T1, H3, "
        "M4); type no other digit, since the Host renders every number. Write each message as a "
        "YAML file (kind, text, positions, targets, reply_to), send it with `committee submit "
        f"--update {task} --role {role} --key {key} --file <message.yaml>` and follow what is "
        f"addressed to you with `committee wait --update {task} --role {role} --key {key} "
        "--seen <seen>`. The key is yours alone.",
        "This route is the bundle's only final action: there is no answer file and nothing for "
        "the lead to submit. It runs these two commands, so the lead starts you as a general "
        "subagent.",
    )


def render(text: str, held: Sequence[Mapping[str, Any]]) -> str:
    """Each holding alias with its name and weight, as the Host has them; agents type none."""
    named = {h["alias"]: f"{h['alias']} ({h['name']}, {h['weight']:.2%})" for h in held}
    return ALIAS.sub(lambda m: named.get(m.group(0), m.group(0)), text)


def _risk_point(
    held: Sequence[Mapping[str, Any]], risk: Mapping[str, Any] | None
) -> list[tuple[str, str, dict[str, Any], list[str]]]:
    """The date's largest risk contributors as a point, or the reason no predicted risk stands."""
    if risk is None:
        return []
    alias = {h["listing_id"]: h["alias"] for h in held}
    top = [
        alias[c["listing_id"]]
        for c in risk.get("top_contributors") or ()
        if c["listing_id"] in alias
    ]
    if risk.get("risk_status") == "EVALUATED" and top:
        return [
            (
                "TOP_RISK_CONTRIBUTORS",
                "The largest risk contributors: {holdings}.",
                {"holdings": ", ".join(top)},
                top,
            )
        ]
    reason = risk.get("reason") or "none of the held names contributes"
    return [
        (
            "RISK_GAP",
            "No predicted risk stands for these positions: {reason}.",
            {"reason": reason},
            [],
        )
    ]


def _risk_view(risk: Mapping[str, Any] | None, held: Sequence[Mapping[str, Any]]) -> str:
    """The date's predicted risk as the Host renders it, report only; members type no digit."""
    if not risk or risk.get("risk_status") != "EVALUATED":
        reason = (risk or {}).get("reason") or "risk_research.date_risk_unavailable"
        return f"Predicted risk of these positions: not evaluated ({reason}); "
    alias = {h["listing_id"]: h["alias"] for h in held}
    top = ", ".join(
        f"{alias.get(c['listing_id'], c['listing_id'])} {c['share']:.0%}"
        for c in risk["top_contributors"]
    )
    later = risk["sessions_before_the_positions"]
    return (
        f"Predicted risk of these positions at {risk['risk_as_of']}"
        + (f", {later} sessions before them" if later else "")
        + f" (the installed recipe, report only): volatility {risk['volatility_per_session']:.2%} "
        f"a session, {risk['volatility_annualized']:.1%} a year; systematic share "
        f"{risk['systematic_share']:.0%}; covered weight {risk['covered_weight']:.0%}; largest "
        f"contributors {top}; "
    )


def _concentration(held: Sequence[Mapping[str, Any]]) -> str:
    """Concentration from the date's weights alone."""
    weights = sorted((h["weight"] for h in held if h["weight"] > 0), reverse=True)
    if not weights:
        return "the date's positions hold no name."
    hhi = sum(w * w for w in weights)
    return (
        f"concentration from the date's weights only: {len(weights)} holdings, largest "
        f"{weights[0]:.2%}, five largest {sum(weights[:5]):.2%}, Herfindahl {hhi:.4f}, effective "
        f"names {1 / hhi:.1f}."
    )


def _points(floor: Floor, stage: str) -> list[dict[str, Any]]:
    """Each tension point with its last ruling's state.

    A CRO challenge on it that is rejected or never ruled leaves its dissent standing
    (`dissents`, the challenges' ids).
    """
    rulings = {m["reply_to"]: m for m in floor.messages if m["kind"] == "RULING"}
    standing = {m["id"] for m in standing_dissents(floor)}
    states = {"ADOPT": "ADOPTED", "REJECT": "REJECTED", "FOR_THE_PERSON": "FOR_THE_PERSON"}
    out = []
    for point in floor.opened["tension_points"]:
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
                ),
                "ruling_id": last.get("id"),
                "dissent_stands": bool(dissents),
                "dissents": dissents,
            }
        )
    return out


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
