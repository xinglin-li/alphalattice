"""The investment committee on a date's published positions: one floor per publication, members
by their own keys, a blind first round, capped debate by event, the PM's rulings and verdict, and
the delivery it feeds."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition import committee
from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.research_delivery import (
    export_update_delivery,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.cli_contract import RequestProvenance
from alphalattice.interface.local_application.portfolio_research import CommitteeMessage
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    advance_decision_state,
)
from tests.portfolio_strategy_lab.synthetic_numerical import (
    HASH,
    build_numerical,
    prepared_for,
    snapshot_for,
)

TASK = str(UUID(int=11))
NOW = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def readback() -> dict[str, Any]:
    """A date's sealed publication as its update readback carries it."""
    listings = tuple(f"c-{i:03d}" for i in range(60))
    n = build_numerical(listing_ids=listings)
    value = advance_decision_state(
        checkpoint=n.checkpoint,
        previous=None,
        prepared=prepared_for(n, n.first),
        observed=snapshot_for(n, n.first),
        plan_hash=HASH,
        published_at=NOW,
    )
    return {
        "task_id": TASK,
        "strategy_package_id": "BALANCED",
        "publication": value.model_dump(mode="json"),
        "history": [],
        "listing_labels": {listing: f"N{i}" for i, listing in enumerate(listings)},
        "review_selector": {
            "update_task_id": TASK,
            "update_publication_hash": value.content_hash,
            "position_basis": "CONDITIONAL_ESTIMATE",
        },
        "html": "<html><body><h1>Positions</h1></body></html>",
    }


EVIDENCE = {"state": "EVIDENCE_AUTHORITY_NOT_ADMITTED", "evidence_as_of": None}


def _floor(tmp_path: Path, readback: dict[str, Any]) -> tuple[GoalStore, list[datetime]]:
    """A floor opened in a research goal's ledger, the clock the test moves."""
    now = [NOW]
    app = GoalApplication(
        GoalStore(tmp_path, "workspace"),
        lambda: now[0],
        lambda *_: {},
        lambda _: now[0],
        {}.get,
        workspace=tmp_path,
    )
    declaration = {
        "title": "Positions",
        "objective": "Positions for a date, argued.",
        "kind": "RESEARCH",
        "criteria": [{"criterion_id": "positions", "text": "The positions stand, argued."}],
    }
    app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_id=UUID(int=7),
            goal_declaration=declaration,
            change_reason="The person's sentence",
        ),
        "EXTERNAL_AUTOMATION",
    )
    opened = [
        committee.open_floor(
            app.store, UUID(int=7), readback, EVIDENCE, session=("claude-code", "s-1"), now=NOW
        )
        for _ in range(2)  # one floor per update
    ]
    assert opened[0] == opened[1]
    assert opened[0].opened["tension_points"][-1]["kind"] == "EVIDENCE_GAP"
    return app.store, now


def _say(
    store: GoalStore, now: list[datetime], role: str, kind: str, key: str = "", **fields: Any
) -> str:
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    text = fields.pop("text", f"{role} on H1.")
    message = CommitteeMessage(kind=kind, text=text, **fields)
    return committee.submit(store, TASK, (role, key or floor.key(role)), message, now[0])


def _read(store: GoalStore, now: list[datetime], role: str | None, seen: int) -> dict[str, Any]:
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    return committee.state(floor, role, seen, now[0])


def _stances(store: GoalStore, now: list[datetime]) -> None:
    for role in committee.ROLES:
        _say(store, now, role, "STANCE")


def test_the_stances_stay_blind_until_all_four_are_in(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the committee): no member reads another's stance, and no row is filed, before
    every stance is in; then each member's wait wakes on all the others' stances."""
    store, now = _floor(tmp_path, readback)
    _say(store, now, "PM", "STANCE")
    _say(store, now, "ALPHA", "STANCE", positions={"T1": "OBJECT"})
    alpha = _read(store, now, "ALPHA", 0)
    assert alpha["stage"] == "STANCES" and [m["role"] for m in alpha["messages"]] == ["ALPHA"]
    assert (alpha["seen"], alpha["for_you"]) == (0, [])  # its own M2 never passes the PM's M1
    floor = committee.floor_of(store, TASK)
    assert floor is not None and committee.unfiled(floor, now[0]) == []
    for role in ("RISK", "CRO"):
        _say(store, now, role, "STANCE")
    revealed = _read(store, now, "ALPHA", alpha["seen"])
    assert revealed["stage"] == "DEBATE" and revealed["for_you"] == ["M1", "M3", "M4"]
    assert _read(store, now, None, 0)["for_you"] == []
    assert "(N" in revealed["messages"][0]["text"]  # the Host renders H1 with its name


def test_a_member_speaks_only_with_its_own_key(tmp_path: Path, readback: dict[str, Any]) -> None:
    """requirement (the committee): the Host mints each member's key from a seed no goal record
    holds, so neither a specialist nor injected text speaks as the PM."""
    store, now = _floor(tmp_path, readback)
    floor = committee.floor_of(store, TASK)
    assert floor is not None and floor.seed not in json.dumps(store.attributed(UUID(int=7)))
    with pytest.raises(ValueError, match=r"message_refused:ROLE"):
        _say(store, now, "PM", "STANCE", key=floor.key("CRO"))
    assert len({floor.key(role) for role in committee.ROLES}) == 4
    assert not floor.admits("PM", None) and floor.admits("PM", floor.key("PM"))


def test_the_floor_answers_a_member_only_with_its_own_key(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the committee): the PM's key goes only to the session that opened the floor,
    and a submit or a member's read with another's key, none, or a typed number is refused."""
    store, now = _floor(tmp_path, readback)
    floor = committee.floor_of(store, TASK)
    assert floor is not None

    def ask(operation: str, session: str = "s-1", **fields: Any) -> dict[str, Any]:
        sent = RequestProvenance(vendor="claude-code", session=session, goal_id=str(UUID(int=7)))
        return committee.operate(
            store,
            Request(operation=operation, update_task_id=UUID(TASK), **fields),
            now[0],
            provenance=sent,
            dated=lambda: (readback, EVIDENCE),
            file_floor=lambda: None,
        )

    assert ask("COMMITTEE_OPEN")["pm_key"] == floor.key("PM")  # again, after a lost context
    assert "pm_key" not in ask("COMMITTEE_OPEN", session="s-2")
    plain, typed = (CommitteeMessage(kind="STANCE", text=t) for t in ("Hold H1.", "Hold 5%."))
    sent = [(floor.key("CRO"), plain), (floor.key("PM"), typed)]  # a submit needs its key field
    codes = [
        ask("COMMITTEE_SUBMIT", committee_role="PM", committee_key=k, committee_message=m)
        for k, m in sent
    ] + [ask("COMMITTEE_READ", committee_role="ALPHA", committee_key=k) for k in (None, "0" * 32)]
    assert [c["failure_code"].rsplit(":")[-1] for c in codes] == ["ROLE", "NUMBER", "ROLE", "ROLE"]
    stance = ask(
        "COMMITTEE_SUBMIT",
        committee_role="PM",
        committee_key=floor.key("PM"),
        committee_message=plain,
    )
    assert stance["status"] == "COMMITTEE_MESSAGE_ACCEPTED"


def test_concurrent_messages_take_distinct_numbers(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """regression (the committee): four members sending at once each get their own number."""
    store, now = _floor(tmp_path, readback)
    with ThreadPoolExecutor(max_workers=4) as pool:
        numbers = list(pool.map(lambda role: _say(store, now, role, "STANCE"), committee.ROLES))
    assert sorted(numbers) == ["M1", "M2", "M3", "M4"]


def test_a_specialist_speaks_three_times_after_its_stance_and_only_the_pm_rules(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the committee): a specialist sends three challenges or replies after its
    stance, names numbers only by alias, and only the PM rules."""
    store, now = _floor(tmp_path, readback)
    _stances(store, now)
    with pytest.raises(ValueError, match=r"message_refused:ROLE"):
        _say(store, now, "PM", "CHALLENGE", targets=("T1",))
    with pytest.raises(ValueError, match=r"message_refused:TARGET"):
        _say(store, now, "CRO", "CHALLENGE", targets=("T1",), text="H999 is weak.")
    challenge = _say(store, now, "CRO", "CHALLENGE", targets=("T1",))
    assert challenge in _read(store, now, "PM", 4)["for_you"]
    for _ in range(2):
        _say(store, now, "CRO", "REPLY", reply_to=challenge)
    with pytest.raises(ValueError, match=r"message_refused:CAP"):
        _say(store, now, "CRO", "REPLY", reply_to=challenge)
    with pytest.raises(ValueError, match=r"message_refused:SHAPE"):
        _say(store, now, "PM", "RULING", reply_to=challenge)


def test_the_verdict_closes_the_floor_and_the_cros_dissent_stands_in_its_words(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the report): a CRO challenge the PM rejects stands in the CRO's own words
    after the verdict, and the person's relayed answer stays theirs, not the PM's view."""
    store, now = _floor(tmp_path, readback)
    _stances(store, now)
    dissent = "H1 lacks Evidence."  # a challenge on a holding, not a tension point
    challenge = _say(store, now, "CRO", "CHALLENGE", targets=("H1",), text=dissent)
    _say(store, now, "PM", "RULING", reply_to=challenge, outcome="REJECT")
    asked = _say(store, now, "RISK", "CHALLENGE", targets=("T2",))
    handed = _say(store, now, "PM", "RULING", reply_to=asked, outcome="FOR_THE_PERSON")
    _say(store, now, "PM", "PERSON_ANSWER", reply_to=handed, text="Cap at 5%.")  # their digits
    _say(store, now, "PM", "VERDICT", outcome="PROCEED_WITH_NOTES")
    with pytest.raises(ValueError, match=r"message_refused:CLOSED"):
        _say(store, now, "RISK", "CHALLENGE", targets=("T1",))
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    items = {i["attribution"]: i["text"] for i in committee.commentary(floor, now[0]) or []}
    first, second = list(items)[:2]
    assert first.endswith("PROCEED_WITH_NOTES") and items[second].startswith("H1 (N")
    assert items["PM, final view"].startswith("PM on H1 (N")
    assert any(t.endswith("Cap at 5%.") for t in items.values())


def test_a_silent_member_reads_not_addressed_when_the_time_boxes_end(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the committee): one slow member never holds the floor; its time boxes reveal
    the stances in and then close it, with the person deciding when the PM gave no verdict."""
    store, now = _floor(tmp_path, readback)
    _say(store, now, "PM", "STANCE")
    now[0] += timedelta(minutes=11)
    state = _read(store, now, None, 0)
    assert state["stage"] == "DEBATE" and state["members"]["ALPHA"] == "NOT_ADDRESSED"
    now[0] += timedelta(minutes=30)
    closed = _read(store, now, None, 0)
    assert closed["stage"] == "CLOSED" and closed["verdict"]["outcome"] == "FOR_THE_PERSON"
    assert (closed["members"]["PM"], closed["members"]["ALPHA"]) == ("IN", "NOT_ADDRESSED")


def test_a_floor_message_is_a_goal_conversation_row_for_its_member(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the Team page shows the floor): each row names its member's card under the
    lead's session, so the Team page lists every member, and its reply's author as recipient."""
    store, now = _floor(tmp_path, readback)
    _stances(store, now)
    challenge = _say(store, now, "RISK", "CHALLENGE", targets=("H1",))
    _say(store, now, "PM", "RULING", reply_to=challenge, outcome="ADOPT")
    other = {"input_channel": committee.CHANNEL, "message_id": "M6", "reference": str(UUID(int=12))}
    store.attribute(UUID(int=7), other)  # another update's floor filed its own M6
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    ruling = committee.unfiled(floor, now[0])[-1]["subject"]
    assert (ruling["message_id"], ruling["role"], ruling["native_agent_id"]) == (
        "M6",
        "research_lead",
        "s-1",
    )
    assert ruling["recipient_id"] == "s-1:alphalattice_risk"
    assert ruling["input_channel"] == "PRODUCT_COMMITTEE"


def test_a_roles_bundle_carries_its_view_of_a_dates_positions(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the bundles): a specialist bundle on a date's update holds the role's view of
    the positions, and while a floor is open, how that role speaks on it with its key."""
    alpha = committee.role_lines("ALPHA", readback, EVIDENCE, None, NOW) or []
    assert any("scores at formation" in line for line in alpha)
    assert any(
        "Observed through" in line
        for line in committee.role_lines("DATA", readback, EVIDENCE, None, NOW) or []
    )
    assert committee.role_lines("CRO", readback, EVIDENCE, None, NOW) is None  # review bundle
    store, _now = _floor(tmp_path, readback)
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    cro = committee.role_lines("CRO", readback, EVIDENCE, floor, NOW) or []
    assert any("No Evidence stands" in line for line in cro)
    assert f"--role CRO --key {floor.key('CRO')}" in cro[-1]
    exited = [{**h, "weight": 0.0} for h in floor.opened["holdings"]]
    emptied = replace(floor, opened={**floor.opened, "holdings": exited})
    assert any(
        "no name" in line for line in committee.role_lines("RISK", readback, {}, emptied, NOW) or []
    )


def test_a_dates_delivery_carries_the_closed_floor_as_its_commentary(
    readback: dict[str, Any],
) -> None:
    """requirement (the report): the delivery takes a date's publication, and the committee's
    record is its commentary, attributed to the committee's floor."""
    request = Request(
        operation="EXPERIMENT_DELIVERY_EXPORT",
        update_task_id=UUID(TASK),
        update_publication_hash=readback["review_selector"]["update_publication_hash"],
        position_basis="CONDITIONAL_ESTIMATE",
    )
    said = [{"attribution": "Committee verdict: PROCEED", "text": "Proceed."}]
    delivered = export_update_delivery(
        request=request, readback=readback, review=None, committee=said
    )
    assert delivered["commentary_provenance"]["attribution"] == "COMMITTEE_FLOOR"
    assert "Committee verdict: PROCEED" in str(delivered["html"])
