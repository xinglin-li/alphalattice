"""The investment committee on a date's published positions: one floor per publication, members
by their own keys, a blind first round, capped debate by event, the PM's rulings and verdict, and
the delivery it feeds."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from alphalattice.control.product_host.composition import committee
from alphalattice.control.product_host.composition import portfolio_result_context as context
from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.portfolio_result_context import date_risk
from alphalattice.control.product_host.composition.research_delivery import (
    export_update_delivery,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.cli_contract import RequestProvenance
from alphalattice.interface.local_application.portfolio_research import CommitteeMessage
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioUpdatePublication,
    advance_decision_state,
    portfolio_update_positions,
)
from alphalattice.investment.risk_research.surfaces.returns import RiskReturnSurfaceError
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


@pytest.fixture
def risk_source(readback: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The book's installed Risk surface over real sessions and the market data's later returns."""
    positions = portfolio_update_positions(
        PortfolioUpdatePublication.model_validate(readback["publication"])
    )
    session, listings = positions.schedule.formation_session, tuple(readback["listing_labels"])
    days = [
        p.formation_session
        for p in context.planned_local_qa_schedule(
            session - timedelta(days=950), session + timedelta(days=9)
        )
    ]
    table = np.random.default_rng(5).normal(0.0, 0.01, (len(days), len(listings)))
    rows = {d: i for i, d in enumerate(days)}
    state = SimpleNamespace(positions=positions, k=days.index(session), days=days)
    state.sessions, state.bars, state.listings = days, set(listings), listings
    authority = SimpleNamespace(sector_map_hash="b" * 64)
    classification = SimpleNamespace(
        map_hash="b" * 64,
        sector_revision="d" * 64,
        manifest_revision="e" * 64,
        entries=[
            SimpleNamespace(listing_id=v, sector_name=v[-1], sector_key=None) for v in listings
        ],
    )
    surface = SimpleNamespace(
        surface_hash="c" * 64,
        epoch=SimpleNamespace(ordered_listing_ids=listings, universe_manifest_revision="u"),
    )

    def carried(_market: Any, _universe: Any, listing: Any, sessions: Any, _connection: Any) -> Any:
        if listing.listing_id not in state.bars:
            raise RiskReturnSurfaceError("risk_research.return_bars_missing")
        column = listings.index(listing.listing_id)
        return tuple(
            {"open_total_return_log": table[rows[d], column], "row_hash": d.isoformat()}
            for d in sessions[1:]
        )

    def load(*_args: Any, **_kwargs: Any) -> Any:
        authority.risk_return_surface_hash = uuid4().hex * 2  # no memo answers another read
        return authority

    market = SimpleNamespace(
        load_universe_manifest_revision=lambda _revision: "universe",
        listing_scope=lambda _u, listing_ids: [SimpleNamespace(listing_id=v) for v in listing_ids],
        _connect=lambda read_only: SimpleNamespace(close=lambda: None),
    )
    reader = SimpleNamespace(
        available_sessions=lambda _surface: tuple(state.sessions),
        read_sessions=lambda _surface, kept: table[[rows[d] for d in kept]],
    )
    history = SimpleNamespace(reclassifications=(), current={v: v[-1] for v in listings})
    stores = SimpleNamespace(load_manifest=lambda _hash: surface)
    relative_path = f"artifacts/portfolio-strategy-lab/{context.CATEGORY}/{'a' * 64}.json"
    monkeypatch.setattr(context.PortfolioResearchArtifactStore, "load", load)
    monkeypatch.setattr(
        context.PanelClosureArtifactStore, "load_model", lambda *_a, **_k: classification
    )
    monkeypatch.setattr(
        context, "sector_history_as_of", lambda *_a: SimpleNamespace(subset=lambda _v: history)
    )
    monkeypatch.setattr(context, "RiskReturnArtifactStore", lambda _root: stores)
    monkeypatch.setattr(context, "CausalRiskReturnReader", lambda _root: reader)
    monkeypatch.setattr(context, "MarketDataRepository", lambda _workspace: market)
    monkeypatch.setattr(context, "listing_returns", carried)
    manifest = SimpleNamespace(
        strategy_artifacts=(
            SimpleNamespace(artifact_key=context.ARTIFACT_KEY, relative_path=relative_path),
        )
    )
    state.risk = lambda: date_risk(tmp_path, manifest, readback, positions)
    return state


def test_a_dates_risk_stands_at_its_formation_or_dated_at_the_surfaces_next_session(
    risk_source: Any,
) -> None:
    """requirement: a Risk carried by later returns is the one read inside the surface; without
    them it stands, dated, at the session after the surface ends."""
    s = risk_source
    inside = s.risk()
    assert inside["risk_as_of"] == s.days[s.k].isoformat()
    assert inside["sessions_before_the_positions"] == 0 and inside["covered_weight"] == 1.0
    s.sessions = s.days[: s.k - 2]
    carried = s.risk()
    assert {**carried, "source_hash": None} == {**inside, "source_hash": None}
    s.bars = set()
    stale = s.risk()
    assert stale["risk_as_of"] == s.days[s.k - 2].isoformat()
    assert stale["sessions_before_the_positions"] == 2


def test_a_name_without_later_returns_is_uncovered_and_left_out(risk_source: Any) -> None:
    """requirement: a held name the market data cannot carry forward lowers the covered weight
    and leaves the contributors; with no covered weight the Risk is not evaluated."""
    s = risk_source
    weights = dict(zip(s.listings, s.positions.weights, strict=True))
    largest = max(weights, key=lambda v: abs(weights[v]))
    s.sessions, s.bars = s.days[: s.k - 2], s.bars - {largest}
    risk = s.risk()
    total = sum(abs(w) for w in weights.values())
    assert risk["covered_weight"] == pytest.approx(1 - abs(weights[largest]) / total)
    assert largest not in [row["listing_id"] for row in risk["top_contributors"]]
    s.bars = {v for v in weights if not weights[v]}
    assert s.risk() == {
        "risk_status": "NOT_EVALUATED",
        "reason": "risk_research.positions_outside_the_risk_axis",
        "covered_weight": 0.0,
    }


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
    the positions, and while a floor is open, its route with its key as the one final action."""
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
    way = committee.route("CRO", floor, NOW) or ()
    assert f"--role CRO --key {floor.key('CRO')}" in way[1] and "only final action" in way[-1]
    assert committee.route("CRO", None, NOW) is None  # no floor: the answer file stands
    exited = [{**h, "weight": 0.0} for h in floor.opened["holdings"]]
    emptied = replace(floor, opened={**floor.opened, "holdings": exited})
    assert any(
        "no name" in line for line in committee.role_lines("RISK", readback, {}, emptied, NOW) or []
    )


def test_the_risk_member_reads_the_dates_predicted_risk(tmp_path: Path, readback: dict) -> None:
    """requirement: the RISK view and a tension point carry the date's predicted risk, dated, and
    its largest contributors, or why none stands."""
    held = committee.holdings(readback)
    top = {"listing_id": held[0]["listing_id"], "weight": held[0]["weight"], "share": 0.21}
    risk = {
        "risk_status": "EVALUATED",
        "risk_as_of": "2026-09-09",
        "sessions_before_the_positions": 0,
        "volatility_per_session": 0.011,
        "volatility_annualized": 0.1746,
        "systematic_share": 0.6,
        "top_contributors": [top],
        "covered_weight": 1.0,
    }
    view = committee.role_lines("RISK", {**readback, "date_risk": risk}, EVIDENCE, None, NOW)
    assert view is not None and all(f in view[1] for f in ("1.10%", "17.5%", "H1 21%"))
    kinds = [(p["kind"], p["targets"]) for p in committee.tension_points(held, EVIDENCE, risk)]
    assert ("TOP_RISK_CONTRIBUTORS", ["H1"]) in kinds
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="w",
        default_strategy_package_id="BALANCED",
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
    )
    positions = portfolio_update_positions(
        PortfolioUpdatePublication.model_validate(readback["publication"])
    )
    absent = date_risk(tmp_path, manifest, readback, positions)
    assert absent["reason"] == "risk_research.installed_risk_surface_absent"
    gap = committee.tension_points(held, EVIDENCE, absent)
    assert any(p["kind"] == "RISK_GAP" and absent["reason"] in p["text"] for p in gap)


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
