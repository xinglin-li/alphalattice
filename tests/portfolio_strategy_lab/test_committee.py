"""Publication-bound committee: blind stances, bounded debate, rulings and attributed delivery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from alphalattice.control.product_host.composition import committee, committee_materials
from alphalattice.control.product_host.composition import portfolio_result_context as context
from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.portfolio_result_context import date_risk
from alphalattice.control.product_host.composition.research_delivery import (
    export_update_delivery,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceError,
    ResearchWorkspaceManifest,
)
from alphalattice.control.product_host.composition.workspace_activity import WorkspaceActivity
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application.activity import (
    ExternalActivityEventDocument,
    ExternalActivityReadQuery,
)
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
    worded,
    worded_refusal,
)
from alphalattice.interface.local_application.labels import Label
from alphalattice.interface.local_application.portfolio_research import CommitteeMessage
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement_task import (
    ADVANCEMENT_TASK_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioUpdatePublication,
    advance_decision_state,
    portfolio_update_positions,
)
from alphalattice.investment.risk_research.surfaces.returns import RiskReturnSurfaceError
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError
from devtools.evaluation.committee import thread_findings
from tests.portfolio_strategy_lab.local_web_support import run_node
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
        "strategy_package_id": n.checkpoint.package.strategy_id,
        "labels": Label(
            kind="strategy",
            id=n.checkpoint.package.strategy_id,
            title="Synthetic test book",
            summary="Synthetic committee presentation fixture.",
            title_zh="合成测试组合",
            summary_zh="委员会展示使用的合成测试样例。",
        ).model_dump(),
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
    readback: dict[str, Any],
) -> None:
    """requirement: a Risk carried by later returns is the one read inside the surface; without
    them it stands, dated, at the session after the surface ends."""
    s = risk_source
    inside = s.risk()
    assert inside["risk_as_of"] == s.days[s.k].isoformat()
    assert inside["sessions_before_the_positions"] == 1 and inside["covered_weight"] == 1.0
    assert inside["systematic_share"] + inside["specific_share"] == pytest.approx(1.0)
    held = {row["listing_id"]: row for row in committee.holdings(readback)}
    for row in inside["top_contributors"]:
        assert (row["alias"], row["name"]) == (
            held[row["listing_id"]]["alias"],
            held[row["listing_id"]]["name"],
        )
    s.sessions = s.days[: s.k - 2]
    carried = s.risk()
    assert {**carried, "source_hash": None} == {**inside, "source_hash": None}
    s.bars = set()
    stale = s.risk()
    assert stale["risk_as_of"] == s.days[s.k - 2].isoformat()
    assert stale["sessions_before_the_positions"] == 3


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
    return committee.submit(
        store,
        TASK,
        (role, key or floor.key(role)),
        message,
        now[0],
        provenance=RequestProvenance(session="s-1" if role == "PM" else f"child-{role.lower()}"),
    )


def test_a_populated_floor_bounds_stances_and_discloses_every_other_point(tmp_path, readback):
    """A ranked agenda bounds judgments without hiding any exception or preventing PM discussion."""
    store, now = _floor(tmp_path, readback)
    task = str(UUID(int=12))
    evidence = {
        "evidence_as_of": "2026-09-09",
        "issuer_rows": [
            {"tickers": [name], "adverse_issue_count": 1}
            for name in readback["listing_labels"].values()
        ],
    }
    floor = committee.open_floor(
        store,
        UUID(int=7),
        readback | {"task_id": task},
        evidence,
        session=("claude-code", "s-1"),
        now=now[0],
    )
    assert len(floor.opened["tension_points"]) > 12
    view = committee.state(floor, "ALPHA", 0, now[0])
    agenda = view["agenda"]
    assert len(agenda["aliases"]) == 12
    assert len(view["tension_points"]) == len(agenda["aliases"]) + agenda["below_count"]
    first, second = next(
        (a, b) for i, a in enumerate(agenda["below"]) for b in agenda["below"][i + 1 :] if a > b
    )
    message = {
        "kind": "STANCE",
        "text": "H1 has a recorded exception.",
        "positions": {a: "OBJECT" for a in [*agenda["aliases"], first]},
        "targets": [agenda["aliases"][0], first],
    }
    request = Request(
        operation="COMMITTEE_SUBMIT",
        update_task_id=UUID(task),
        committee_role="ALPHA",
        committee_key=floor.key("ALPHA"),
        committee_message=message,
    )

    def send(given):
        return committee.operate(
            store,
            given,
            now[0],
            provenance=RequestProvenance(
                session="s-1" if given.committee_role == "PM" else "child-alpha"
            ),
            dated=lambda: None,
            file_floor=lambda: None,
        )

    for fields in (
        {"targets": [first]},
        {"text": f"Discuss {first}."},
        {"positions": {"H1": "OBJECT"}},
        {"positions": {floor.opened["facts"][0]["alias"]: "OBJECT"}},
    ):
        given = replace(
            request,
            committee_message=CommitteeMessage(
                kind="STANCE",
                text=fields.get("text", "Discuss this exception."),
                **{k: v for k, v in fields.items() if k != "text"},
            ),
        )
        refused = send(given)
        assert refused["failure_code"] == "committee.message_refused:AGENDA"
        if "text" in fields:
            assert "corrected" not in refused["next_requests"]
            assert refused["author_revision_fields"] == ["committee_message.text"]
            assert set(refused["next_requests"]) == {"floor", "schema"}
    refused = send(request)
    assert refused["failure_code"] == "committee.message_refused:AGENDA"
    assert refused["agenda_size"] == 12
    corrected = Request(**refused["next_requests"]["corrected"])
    assert set(corrected.committee_message.positions) == set(agenda["aliases"])
    assert corrected.committee_message.targets == (agenda["aliases"][0],)
    assert corrected.committee_message.text == message["text"]
    assert send(corrected)["status"] == "COMMITTEE_MESSAGE_ACCEPTED"
    now[0] += timedelta(minutes=11)
    assert send(corrected)["status"] == "COMMITTEE_MESSAGE_ACCEPTED"

    def message_request(role, **fields):
        return replace(
            request,
            committee_role=role,
            committee_key=floor.key(role),
            committee_message=CommitteeMessage(text="Discuss this exception.", **fields),
        )

    for kind, fields in (
        ("CHALLENGE", {}),
        ("REPLY", {"reply_to": "M1"}),
        ("VERDICT", {"outcome": "PROCEED"}),
    ):
        role = "PM" if kind == "VERDICT" else "ALPHA"
        for names in ({"targets": [first]}, {"positions": {first: "OBJECT"}}):
            assert send(message_request(role, kind=kind, **fields, **names))["failure_code"] == (
                "committee.message_refused:AGENDA"
            )
    challenge = message_request("ALPHA", kind="CHALLENGE", targets=[agenda["aliases"][0]])
    assert send(challenge)["status"] == "COMMITTEE_MESSAGE_ACCEPTED"
    for kind, fields in (
        ("RULING", {"reply_to": "M2", "outcome": "FOR_THE_PERSON"}),
        ("PERSON_ANSWER", {"reply_to": "M3"}),
    ):
        refused = send(message_request("PM", kind=kind, targets=[first], **fields))
        assert refused["failure_code"] == "committee.message_refused:AGENDA"
        if kind == "RULING":
            assert send(message_request("PM", kind=kind, **fields))["status"] == (
                "COMMITTEE_MESSAGE_ACCEPTED"
            )
    targets_only = message_request("PM", kind="REPLY", reply_to="M1", targets=[first])
    assert send(targets_only)["failure_code"] == "committee.message_refused:AGENDA"
    multiple = message_request(
        "PM", kind="REPLY", reply_to="M1", positions={second: "OBJECT", first: "SUPPORT"}
    )
    corrected_raise = Request(**send(multiple)["next_requests"]["corrected"])
    assert corrected_raise.committee_message.positions == {first: "SUPPORT"}
    assert send(corrected_raise)["status"] == "COMMITTEE_MESSAGE_ACCEPTED"
    assert send(message_request("PM", kind="REPLY", reply_to="M1", positions={second: "OBJECT"}))[
        "status"
    ] == ("COMMITTEE_MESSAGE_ACCEPTED")
    floor = committee.floor_of(store, task)
    raised = committee.state(floor, None, 0, now[0])
    assert raised["agenda"]["raised"] == [first, second]
    assert next(p for p in raised["tension_points"] if p["alias"] == first)["state"] == "OPEN"
    now[0] += timedelta(minutes=40)
    rows = committee.commentary(floor, now[0])
    disclosure = next(r for r in rows if "agenda" in r)
    assert disclosure["agenda"] == committee.state(floor, None, 0, now[0])["agenda"]
    assert {r["point"] for r in rows if "point" in r} == {
        p["alias"] for p in view["tension_points"]
    }


def test_a_small_floor_and_its_report_stay_exact_under_the_agenda_rule(
    tmp_path, readback, monkeypatch
):
    """A small floor stays exact while declared agenda changes bind bounded floors."""
    store, now = _floor(tmp_path, readback)
    unchanged = committee.floor_of(store, TASK)
    assert "agenda" not in unchanged.opened
    assert "each tension point" in committee.route("ALPHA", unchanged, now[0])[0]
    other, _ = _floor(tmp_path / "other", readback)
    task = str(UUID(int=13))
    arguments = {"session": ("claude-code", "s-1"), "now": now[0]}
    first = committee.open_floor(
        store, UUID(int=7), readback | {"task_id": task}, EVIDENCE, agenda_limit=12, **arguments
    )
    second = committee.open_floor(
        other, UUID(int=7), readback | {"task_id": task}, EVIDENCE, agenda_limit=24, **arguments
    )
    assert first.opened == second.opened
    at = now[0] + timedelta(minutes=41)
    assert committee.state(first, None, 0, at) == committee.state(second, None, 0, at)
    assert committee.commentary(first, at) == committee.commentary(second, at)
    manifest = ResearchWorkspaceManifest.research_only("synthetic")
    configured = manifest.with_bindings(committee_agenda_limit=12)
    assert configured.manifest_hash == manifest.manifest_hash
    assert (
        configured.with_bindings(committee_agenda_limit=24).manifest_hash
        != configured.manifest_hash
    )
    for invalid in (0, 25, True, 1.5):
        with pytest.raises(ResearchWorkspaceError) as failure:
            manifest.with_bindings(committee_agenda_limit=invalid)
        assert str(failure.value) == "research_workspace.committee_agenda_limit_invalid"
        words = worded_refusal({"failure_code": str(failure.value), "status": "REFUSED"})
        assert words["next_action"] == "CORRECT_THE_WORKSPACE_DECLARATION"
        assert all(term in words["detail"] for term in ("committee_agenda_limit", "1", "24"))
    policy = {
        "rule_version": 1,
        "limit": 1,
        "rank_keys": (
            ("affected_holding_weight", "desc"),
            ("adverse_count", "desc"),
            ("original_alias_number", "asc"),
        ),
    }
    points, held = first.opened["tension_points"][:2], first.opened["holdings"]
    before = committee_materials.agenda(points, held, 1)["agenda"]
    monkeypatch.setattr(committee_materials, "AGENDA_RANKING", "Same ordering.")
    after = committee_materials.agenda(points, held, 1)["agenda"]
    assert all(before[key] == value for key, value in policy.items())
    assert before["ranking"] != after["ranking"]
    assert before["rule_hash"] == after["rule_hash"] == canonical_hash(policy)
    assert canonical_hash(policy | {"rule_version": 2}) != before["rule_hash"]
    assert (
        canonical_hash(policy | {"rank_keys": tuple(reversed(policy["rank_keys"]))})
        != before["rule_hash"]
    )


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
    person = _read(store, now, None, 0)
    assert (person["message_count"], person["member_count"], person["stances_in"]) == (0, 4, 2)
    assert person["messages"] == [] and person["first_message_at"] is None
    assert person["session"] == ["claude-code", "s-1"]
    assert person["review_selector"] == readback["review_selector"]
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
    with pytest.raises(ValueError, match=r"message_refused:KEY"):
        _say(store, now, "PM", "STANCE", key=floor.key("CRO"))
    assert len({floor.key(role) for role in committee.ROLES}) == 4
    assert not floor.admits("PM", None) and floor.admits("PM", floor.key("PM"))


@pytest.mark.parametrize("sender", ["s-1", "child-alpha"])
def test_the_floor_answers_a_member_only_with_its_own_key(
    tmp_path: Path, readback: dict[str, Any], sender: str
) -> None:
    """Keys bind each role; shared or distinct declared sessions keep honest receipt attribution."""
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

    opened = ask("COMMITTEE_OPEN")
    assert opened["pm_key"] == floor.key("PM")  # again, after a lost context
    # The PM's own route rides with its key: its submit, the members started together.
    route = " ".join(opened["pm_route"])
    assert f"--role PM --key {floor.key('PM')}" in route and "VERDICT" in route
    assert "schema show COMMITTEE_SUBMIT" in route and "together" in route
    other = ask("COMMITTEE_OPEN", session="s-2")  # whose route it is, and the floor's read
    assert not {"pm_key", "pm_route"} & set(other) and other["detail"] == committee.OPENER_WORDS
    assert other["next_requests"]["read"] == {"operation": "COMMITTEE_READ", "update_task_id": TASK}
    plain, typed = (CommitteeMessage(kind="STANCE", text=t) for t in ("Hold H1.", "Hold 5%."))
    sent = [(floor.key("CRO"), plain), (floor.key("PM"), typed)]  # a submit needs its key field
    codes = [
        ask("COMMITTEE_SUBMIT", committee_role="PM", committee_key=k, committee_message=m)
        for k, m in sent
    ] + [ask("COMMITTEE_READ", committee_role="ALPHA", committee_key=k) for k in (None, "0" * 32)]
    assert [c["failure_code"].rsplit(":")[-1] for c in codes] == ["KEY", "NUMBER", "KEY", "KEY"]
    assert all(c["rule"] in c["failure_code"] and c["next_requests"]["floor"] for c in codes)
    assert "committee_key" not in codes[0]["next_requests"]["floor"]
    codes[1] = worded_refusal(codes[1])
    assert {"resubmit", "same", "message", "figure", "reference", "alias"} <= set(
        codes[1]["detail"].split()
    )
    assert codes[1]["next_action"] == "READ_THE_FLOOR"
    stance = ask(
        "COMMITTEE_SUBMIT",
        committee_role="PM",
        committee_key=floor.key("PM"),
        committee_message=plain,
    )
    assert stance["status"] == "COMMITTEE_MESSAGE_ACCEPTED"
    fields = dict(committee_role="ALPHA", committee_key=floor.key("ALPHA"), committee_message=plain)
    refused = ask("COMMITTEE_SUBMIT", session=None, **fields)
    assert refused["rule"] == "MEMBER" and refused["next_requests"]["floor"]
    taken = ask("COMMITTEE_SUBMIT", session=sender, **fields)
    assert taken["status"] == "COMMITTEE_MESSAGE_ACCEPTED"
    assert ask("COMMITTEE_SUBMIT", **fields)["message_id"] == taken["message_id"]
    caller = RequestProvenance(session=sender)
    with pytest.raises(ValueError, match=r"message_refused:MEMBER"):
        committee.submit(store, TASK, ("ALPHA", None), plain, now[0], provenance=caller)
    kept = committee.floor_of(store, TASK)
    assert kept.messages[-1]["author_session"] == sender
    assert committee.floor_row(kept, kept.messages[-1])["subject"]["native_agent_id"] == sender


def test_a_review_member_answers_its_own_bundle_on_its_thread(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """Keyed review corrections and acceptance preserve the author and keyless receipt reread."""
    store, now = _floor(tmp_path, readback)
    directory = str(tmp_path / "analyst-u01")
    lead = RequestProvenance(vendor="codex", session="s-1", goal_id=str(UUID(int=7)))
    token = REQUEST_PROVENANCE.set(lead)
    filed: list[str] = []

    def file(thread: str, _now: datetime) -> None:
        filed.append(thread)

    try:
        route = committee.bundle_parts("ANALYST", None, store, now[0], directory)[1] or ()
        reference = committee.bundle_slot(directory)
        ready = {"status": "AGENT_BUNDLE_READY", "agent_role": "ANALYST", "task_id": "T-1"}
        committee.issued(store, {**ready, "bundle_reference": reference}, now[0], file)
        pending = committee.thread_of(store, committee.thread_id(reference))
        assert (
            committee.unfiled(pending, now[0]) == []
        )  # no guessed ISSUE recipient before admission
    finally:
        REQUEST_PROVENANCE.reset(token)
    key = route[1].split("--key ")[1][:32]
    member, other = (("ANALYST", reference, {"findings": f}) for f in ([], [{}]))
    refused = [committee.admit(store, member, k) for k in (None, "0" * 32)]
    assert [isinstance(r, dict) and r["rule"] for r in refused] == ["MEMBER", "KEY"]
    correct = {"status": "CORRECT", "problems": [{}], "rounds_left": 2}
    for sent, body, k in (
        (other, correct, key),
        (member, ACCEPTED_BODY, key),
        (member, ACCEPTED_BODY, None),
    ):
        caller = lead if k is None else replace(lead, session="child-analyst")
        token = REQUEST_PROVENANCE.set(caller)
        try:
            thread = committee.admit(store, sent, k)
            assert isinstance(thread, committee.Floor)  # accepted receipt retains its first author
            committee.answered(store, thread, sent, body, now[0], file)
        finally:
            REQUEST_PROVENANCE.reset(token)
        assert body["conversation"]["author"] == "child-analyst"
    closed = [committee.admit(store, other, k) for k in (None, key)]  # a different answer
    assert [isinstance(r, dict) and r["rule"] for r in closed] == ["ENDED", "ENDED"]
    thread = committee.thread_of(store, committee.thread_id(reference))
    assert thread is not None and committee.floor_of(store, committee.thread_id(reference)) is None
    said = [(m["role"], m["kind"], m["reply_to"]) for m in thread.messages]
    assert said == [
        ("PM", "ISSUE", None),
        ("ANALYST", "ANSWER", "M1"),
        ("HOST", "CORRECTION", "M2"),
        ("ANALYST", "ANSWER", "M3"),
        ("HOST", "ACCEPTANCE", "M4"),
    ]
    assert thread.messages[0]["text"] == "Analyst bundle issued."  # words, no code or Task id
    assert thread_findings(thread.messages, "ANALYST") == []
    forged = [{**m, "role": "PM"} if m["kind"] == "ANSWER" else m for m in thread.messages[:3]]
    assert thread_findings(forged, "ANALYST") == ["thread.author:M2", "thread.open"]
    rows = committee.unfiled(thread, now[0])
    assert {r["subject"]["input_channel"] for r in rows} == {"PRODUCT_REVIEW"}
    assert rows[1]["subject"]["native_agent_id"] == "child-analyst"
    assert rows[-1]["subject"]["recipient_id"] == rows[1]["subject"]["native_agent_id"]
    assert rows[0]["subject"]["recipient_id"] == rows[1]["subject"]["native_agent_id"]  # issue
    assert rows[-1]["subject"]["answer_reference"] == "a" * 64 and len(filed) == 3


ACCEPTED_BODY = {"status": "ACCEPTED", "answer_reference": "a" * 64, "task_id": "T-2"}


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
    # Resent as it stands, it answers with its first admission and spends none of the three.
    assert _say(store, now, "CRO", "CHALLENGE", targets=("T1",)) == challenge
    for said in ("Agreed.", "Still so."):  # distinct replies; a resend spends nothing
        _say(store, now, "CRO", "REPLY", reply_to=challenge, text=said)
    with pytest.raises(ValueError, match=r"message_refused:CAP"):
        _say(store, now, "CRO", "REPLY", reply_to=challenge, text="Once more.")
    with pytest.raises(ValueError, match=r"message_refused:SHAPE"):
        _say(store, now, "PM", "RULING", reply_to=challenge)
    with pytest.raises(ValueError, match=r"message_refused:SHAPE"):
        _say(store, now, "PM", "RULING", reply_to=challenge, outcome="DECLINE")


def test_the_floor_settles_once_every_member_has_spoken_or_passed_its_turn(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (PASS): after the reveal each specialist takes its turn or passes it; the floor
    settles once all have and no challenge stands unruled, and the report reads a pass in words."""
    store, now = _floor(tmp_path, readback)
    _say(store, now, "PM", "STANCE")
    with pytest.raises(ValueError, match=r"message_refused:STANCES"):
        _say(store, now, "ALPHA", "PASS", text="Nothing to add.")
    _stances(store, now)
    with pytest.raises(ValueError, match=r"message_refused:TURN"):  # a pass names nothing
        _say(store, now, "CRO", "PASS", text="Nothing to add.", targets=("T1",))
    passed = [_say(store, now, role, "PASS", text="Nothing to add.") for role in ("ALPHA", "CRO")]
    assert _say(store, now, "ALPHA", "PASS", text="Nothing to add.") == passed[0]  # a replay
    for kind, fields in (("PASS", {}), ("CHALLENGE", {"targets": ("T1",)})):
        with pytest.raises(ValueError, match=r"message_refused:TURN"):  # its turn has ended
            _say(store, now, "ALPHA", kind, text="Still nothing.", **fields)
    assert not _read(store, now, "PM", 4)["settled"]  # RISK has not taken its turn
    late = _say(store, now, "RISK", "CHALLENGE", targets=("T1",))
    with pytest.raises(ValueError, match=r"message_refused:TURN"):  # it spoke its turn
        _say(store, now, "RISK", "PASS", text="Nothing to add.")
    assert not _read(store, now, "PM", 4)["settled"]  # its challenge stands unruled
    with pytest.raises(ValueError, match=r"message_refused:ROLE"):
        _say(store, now, "PM", "PASS", text="Nothing to add.")
    _say(store, now, "PM", "RULING", reply_to=late, outcome="REJECT")
    assert _read(store, now, "PM", 4)["settled"]
    _say(store, now, "PM", "VERDICT", outcome="PROCEED")
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    rows = committee.commentary(floor, now[0]) or []
    passed = [r for r in rows if r["attribution_word"] == "{member} passed its discussion turn"]
    assert {r["attribution_words"]["member"] for r in passed} == {"ALPHA", "CRO"}


@pytest.mark.parametrize("outcome", ["PROCEED_WITH_NOTES", "DECLINE"])
def test_the_verdict_closes_the_floor_and_the_cros_dissent_stands_in_its_words(
    tmp_path: Path, readback: dict[str, Any], outcome: str
) -> None:
    """The Report keeps rejected CRO dissent and relayed answers in their authors' words."""
    store, now = _floor(tmp_path, readback)
    _stances(store, now)
    dissent = "H1 lacks Evidence."  # a challenge on a holding, not a tension point
    challenge = _say(store, now, "CRO", "CHALLENGE", targets=("H1",), text=dissent)
    untargeted = _say(store, now, "CRO", "CHALLENGE", targets=())
    _say(store, now, "PM", "RULING", reply_to=challenge, outcome="REJECT")
    _say(store, now, "PM", "RULING", reply_to=untargeted, outcome="REJECT")  # a verdict waits
    asked = _say(store, now, "RISK", "CHALLENGE", targets=("T2",))
    handed = _say(store, now, "PM", "RULING", reply_to=asked, outcome="FOR_THE_PERSON")
    _say(store, now, "PM", "PERSON_ANSWER", reply_to=handed, text="Cap at 5%.")  # their digits
    _say(store, now, "PM", "VERDICT", outcome=outcome)
    with pytest.raises(ValueError, match=r"message_refused:CLOSED"):
        _say(store, now, "RISK", "CHALLENGE", targets=("T1",))
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    rows = committee.commentary(floor, now[0]) or []
    held = floor.opened["holdings"]
    holding = committee.render("H1", held)
    assert rows[0]["attribution_word"] == "Committee verdict: {outcome}"
    assert rows[0]["attribution_words"] == {"outcome": outcome}
    assert rows[1]["attribution_word"] == "CRO dissent stands on {targets}"
    assert rows[1]["attribution_words"] == {"message": challenge, "targets": holding}
    assert rows[1]["text"] == committee.render(dissent, held)
    assert rows[2]["attribution_word"] == "CRO dissent stands"
    assert rows[2]["attribution_words"] == {"message": untargeted, "targets": ""}
    final = [r for r in rows if r["attribution_word"] == "{member}, final view"]
    pm = next(r for r in final if r["attribution_words"]["member"] == "PM")
    assert pm["attribution_words"]["message"] == next(
        m["id"] for m in reversed(floor.messages) if m["kind"] == "RULING"
    )
    views = {r["attribution_words"]["member"]: r["text"] for r in final}
    spoken = [m for m in floor.messages if m["kind"] in {"STANCE", "CHALLENGE", "REPLY", "RULING"}]
    said = {m["role"]: committee.render(m["text"], held) for m in spoken}
    assert views == said
    assert not any(row.get("text_word") for row in rows if row["text"] in said.values())
    person = next(row for row in rows if "answer_present" in row)
    question = next(m["text"] for m in floor.messages if m["id"] == handed)
    expected = (committee.render(question, held), "Cap at 5%.", True)
    assert (person["question"], person["answer"], person["answer_present"]) == expected
    assert person["relay_word"] == committee.RELAY_WORD
    aimed = next(p for p in floor.opened["tension_points"] if p["alias"] == "T2")
    point = committee.render(", ".join(["T2", *aimed["targets"]]), held)
    assert person["attribution_word"] == "Handed to the person: {targets}"
    assert person["attribution_words"] == {"message": handed, "targets": point}
    ruled = [r for r in rows if r["attribution_word"] == "PM ruling on {targets}: {outcome}"]
    rulings = [r["attribution_words"] for r in ruled]
    expected = {challenge: (holding, "REJECT"), asked: (point, "FOR_THE_PERSON")}
    assert {r["message"]: (r["targets"], r["outcome"]) for r in rulings} == expected
    assert not any(m["id"] in row["attribution"] for m in floor.messages for row in rows)
    page = committee.state(floor, None, 0, now[0])
    assert page["standing_dissents"] == [challenge, untargeted]
    assert page["person_items"][0]["answer_at"] == now[0].isoformat()
    assert page["person_items"][0]["answer"] == "Cap at 5%."
    assert page["person_items"][0]["relay_word"] == committee.RELAY_WORD


def test_a_silent_member_reads_not_addressed_when_the_time_boxes_end(
    tmp_path: Path, readback: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time boxes reveal stances, then leave unruled points for the person."""
    store, now = _floor(tmp_path, readback)
    _say(store, now, "PM", "STANCE")
    now[0] += timedelta(minutes=11)
    state = _read(store, now, None, 0)
    assert state["stage"] == "DEBATE" and state["members"]["ALPHA"] == "NOT_ADDRESSED"
    # A research update's published review offers the floor opened on its publication, as it
    # stands: open, then its report; another kind of update has none to open.
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    selector = dict(floor.opened["selector"])

    def offered(read: dict[str, Any], kind: str = ADVANCEMENT_TASK_KIND) -> dict[str, Any]:
        review: dict[str, Any] = {"state": "REVIEW_PUBLISHED", "next_requests": {}}
        tasks = SimpleNamespace(task=lambda _: SimpleNamespace(task_kind=kind))
        host = SimpleNamespace(
            goals=SimpleNamespace(store=store, clock=lambda: now[0]),
            workspace_session=SimpleNamespace(task_control_registry=tasks),
        )
        book = SimpleNamespace(request_fields=lambda: read)
        return committee.offer(host, review, book, "EXTERNAL_AUTOMATION")

    opened = offered(selector)
    assert opened["next_requests"]["committee"]["operation"] == "COMMITTEE_OPEN"
    assert offered(selector, kind="portfolio_update")["next_requests"] == {}
    with pytest.raises(ValueError, match=r"message_refused:TURN"):  # no stance to pass from
        _say(store, now, "ALPHA", "PASS", text="Nothing to add.")
    assert {p["state"] for p in state["tension_points"]} == {"OPEN"}
    now[0] += timedelta(minutes=30)
    closed = _read(store, now, None, 0)
    assert closed["stage"] == "CLOSED" and closed["verdict"]["outcome"] == "FOR_THE_PERSON"
    assert (closed["members"]["PM"], closed["members"]["ALPHA"]) == ("IN", "NOT_ADDRESSED")
    ended, earlier = offered(selector), offered({**selector, "update_publication_hash": "e" * 64})
    assert ended["next_requests"] == {"committee_report": closed["delivery"]}
    assert [opened["next_action"], ended["next_action"], earlier["next_action"]] == [
        "CONVENE_THE_COMMITTEE",
        "REPORT_THE_COMMITTEE",
        "CONVENE_THE_COMMITTEE",  # a floor on another publication is not this review's
    ]

    def unreadable(*_: Any) -> None:
        raise ValueError("goal.attribution_invalid")

    with monkeypatch.context() as patched:  # a corrupt goal record
        patched.setattr(committee, "thread_of", unreadable)
        named = offered(selector)
    assert named["next_requests"] == {} and named["committee_unreadable"]
    assert {p["state"] for p in closed["tension_points"]} == {"NOT_ADDRESSED"}
    with pytest.raises(ValueError, match=r"message_refused:CLOSED"):
        _say(store, now, "ALPHA", "STANCE")
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    rows = committee.commentary(floor, now[0])
    assert rows and [r["text_word"] for r in rows if "text_word" in r] == [
        closed["verdict"]["text"],
        *["Not addressed."] * 3,
    ]


def test_only_the_leads_superseded_refusal_is_offered_its_published_committee(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lead's stale or ambiguous refusal on its Task's reviewed publication offers the
    committee, its code kept; a member's, a transient, another publication's do not."""
    from alphalattice.control.product_host.composition.evidence_review_delivery import (
        EvidenceReviewDelivery,
    )

    book = {"update_task_id": TASK, "update_publication_hash": "p" * 64}
    book["position_basis"] = "CONDITIONAL_ESTIMATE"
    standing: list[Any] = [{"status": "REVIEWED"}]

    def read(_delivery: Any, _selector: Any) -> Any:
        if isinstance(standing[0], Exception):
            raise standing[0]
        return standing[0]

    monkeypatch.setattr(EvidenceReviewDelivery, "review_standing", read)
    tasks = SimpleNamespace(task=lambda _: SimpleNamespace(task_kind=ADVANCEMENT_TASK_KIND))
    host = SimpleNamespace(
        review=None,
        research_updates=SimpleNamespace(publication_task={"p" * 64: UUID(TASK)}.get),
        goals=SimpleNamespace(store=GoalStore(tmp_path, "workspace"), clock=lambda: NOW),
        workspace_session=SimpleNamespace(task_control_registry=tasks),
    )

    def way(code: str, lead: bool = True, **fields: Any) -> dict[str, Any]:
        refused = {"status": "REFUSED", "failure_code": code, "message": "Tell your lead."}
        refused["next_action"] = "TELL_YOUR_LEAD"
        return committee.lead_way(host, {**book, **fields}, lead, "EXTERNAL_AUTOMATION", refused)

    offered = way("agent_bundle.preparation_stale")
    assert (offered["failure_code"], offered["message"], offered["next_action"]) == (
        "agent_bundle.preparation_stale",
        committee.LEAD_PUBLISHED,
        "CONVENE_THE_COMMITTEE",
    )
    assert offered["next_requests"]["committee"]["update_task_id"] == TASK
    kept = [
        way("EVIDENCE_REFRESH_IN_PROGRESS"),  # transient: its own words and way
        way("REFUSED_EVIDENCE_SELECTION_AMBIGUOUS", lead=False),  # a member, or no thread
        way("agent_bundle.preparation_stale", update_publication_hash="q" * 64),
    ]
    standing[0] = {"status": "NOT_REVIEWED"}
    kept.append(way("REFUSED_EVIDENCE_SELECTION_AMBIGUOUS"))
    standing[0] = OSError("store unreadable")
    kept.append(way("agent_bundle.preparation_stale"))
    assert [k["next_action"] for k in kept] == ["TELL_YOUR_LEAD"] * 5
    previous = {"review_selector": book, "review_standing": {"status": "REVIEWED"}}
    previous |= {"publication_is_previous": True, "next_requests": {}}
    assert committee.update_offer(host, dict(previous), "EXTERNAL_AUTOMATION") == previous


def test_a_members_request_counts_toward_its_floors_open_goal_and_never_overrides_a_named_one(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """requirement (the second member review, 2026-10-15): only while the floor's goal is open."""
    store, now = _floor(tmp_path, readback)
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    fields = ("committee_key", "committee_role", "agent_role", "task_id", "update_task_id")
    risk = {"operation": "COMMITTEE_SUBMIT", "update_task_id": TASK, "committee_role": "RISK"}
    prepare = {"operation": "AGENT_BUNDLE_PREPARE", "agent_role": "RISK", "task_id": TASK}

    def join(request: dict[str, Any], held: Any = None, named: str | None = None) -> tuple:
        asked = SimpleNamespace(**{**dict.fromkeys(fields), "bundle_directory": None, **request})
        member, goal = committee.member_goal(
            store, asked, held, SimpleNamespace(goal_id=named), now[0]
        )
        return member, getattr(goal, "goal_id", goal)

    seven = UUID(int=7)
    assert join(risk | {"committee_key": floor.key("RISK")}, "mine") == (True, seven)
    assert join(risk | {"committee_key": floor.key("PM")}, "mine") == (True, "mine")
    assert join(risk | {"committee_role": "PM", "committee_key": floor.key("PM")}, "mine") == (
        False,
        "mine",
    )
    assert join(risk | {"committee_key": floor.key("RISK")}, "named", "x") == (False, "named")
    assert join(prepare) == (True, seven) and join(prepare, "mine") == (False, "mine")
    now[0] += timedelta(days=30)  # the floor's time box has closed it
    assert join(prepare) == (False, None)
    head = store.head(seven)
    assert head is not None
    store.head = lambda _goal: head.model_copy(update={"state": "COMPLETE"})  # type: ignore[method-assign]
    assert join(risk | {"committee_key": floor.key("RISK")}, "mine") == (False, "mine")


def test_a_floor_message_is_a_goal_conversation_row_for_its_member(
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """Floor rows retain member, session and reply-recipient identities for Team."""
    store, now = _floor(tmp_path, readback)
    opened = committee.floor_of(store, TASK)
    assert opened is not None
    blind = committee.context(opened, now[0])
    assert (blind["stage"], blind["verdict"]) == ("STANCES", None)
    _stances(store, now)
    challenge = _say(store, now, "RISK", "CHALLENGE", targets=("H1",))
    _say(store, now, "PM", "RULING", reply_to=challenge, outcome="ADOPT")
    other = {"input_channel": committee.CHANNEL, "message_id": "M6", "reference": str(UUID(int=12))}
    store.attribute(UUID(int=7), other)  # another update's floor filed its own M6
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    ruling = committee.unfiled(floor, now[0])[-1]["subject"]
    expected = ("M6", "research_lead", "s-1")
    assert (ruling["message_id"], ruling["role"], ruling["native_agent_id"]) == expected
    assert ruling["recipient_id"] == "child-risk"
    legacy = replace(
        floor, opened={k: v for k, v in floor.opened.items() if k != "member_sessions"}
    )
    old = {k: v for k, v in floor.messages[2].items() if k != "author_session"}
    assert committee.floor_row(legacy, old)["subject"]["native_agent_id"] == "s-1:alphalattice_risk"
    assert ruling["input_channel"] == "PRODUCT_COMMITTEE"
    dissent = _say(store, now, "CRO", "CHALLENGE", targets=("H1",), text="H1 lacks Evidence.")
    _say(store, now, "PM", "RULING", reply_to=dissent, outcome="REJECT")
    asked = _say(store, now, "RISK", "CHALLENGE", targets=("T2",))
    handed = _say(store, now, "PM", "RULING", reply_to=asked, outcome="FOR_THE_PERSON")
    _say(store, now, "PM", "PERSON_ANSWER", reply_to=handed, text="Cap at 5%.")
    _say(store, now, "PM", "VERDICT", outcome="PROCEED_WITH_NOTES")
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    observer = WorkspaceActivity(
        workspace=tmp_path,
        workspace_id="committee-fixture",
        gate=WorkspaceMutationGate(),
        instance="committee-fixture",
        clock=lambda: now[0],
    )
    for row in committee.unfiled(floor, now[0]):
        row["subject"]["goal_id"] = str(floor.goal_id)
        observer.admit_external_event(ExternalActivityEventDocument.model_validate(row))
    page = observer.read_external(ExternalActivityReadQuery())
    observer.close()
    body = committee.state(floor, None, 0, now[0])
    assert [m["row"] for m in body["messages"]] == committee.unfiled(floor, now[0])
    assert committee.context(floor, now[0]) == {
        "goal_id": str(floor.goal_id),
        "update_task_id": TASK,
        "review_selector": readback["review_selector"],
        "native_host": "claude-code",
        "native_session_id": "s-1",
        "stage": "CLOSED",
        "verdict": body["verdict"],
        "specialists": ["alphalattice_alpha", "alphalattice_risk", "alphalattice_cro"],
    }
    assert set(blind) == set(committee.context(floor, now[0]))
    fixture = tmp_path / "committee-reader.json"
    payload = {"pages": [page], "reads": {"/api/committee": body}, "committee": body}
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    modules = (
        Path(__file__).resolve().parents[2]
        / "src/alphalattice/interface/local_application/assets/workbench-source/js/app"
    )
    script = Path(__file__).with_name("workbench_team_readback.cjs")
    args = [str(script), str(modules), str(fixture), "s-1"]
    result = run_node(args, capture_output=True, text=True, timeout=30)
    assert result is not None and result.returncode == 0, result.stderr if result else ""


def test_one_sealed_source_gives_one_fact_text_under_any_hash_seed(tmp_path: Path, readback):
    """requirement (the materials): facts order by value, then listing, never by set order."""
    (tmp_path / "r.json").write_text(json.dumps(readback), encoding="utf-8")
    code = "import json,sys;from alphalattice.control.product_host.composition import committee"
    code += ";r=json.load(open(sys.argv[1]));print(committee.facts(r,{},committee.holdings(r)))"
    run = [sys.executable, "-c", code, tmp_path / "r.json"]
    texts = [subprocess.check_output(run, env=os.environ | {"PYTHONHASHSEED": s}) for s in "12"]
    assert texts[0] == texts[1] and b"question" in texts[0]


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
    absent = [f for f in floor.opened["facts"] if f["source"] == "absence"]  # words; code apart
    assert absent and all(f["state"] in {"PENDING", "UNAVAILABLE"} for f in absent)
    assert not any(f["state"] in f["text"] for f in absent)
    at, route = "2026-09-10T23:56:21.531000+00:00", "ACCEPT_WITH_LIMITS"
    standing = {"status": "REVIEWED", "published_at": at, "cro": {"route": route}}
    reviewed = committee.role_lines(
        "CRO", {**readback, "review_standing": standing}, {}, floor, NOW
    )
    said = "now: reviewed, published 2026-09-10 23:56 UTC; CRO route accepted with limits."
    assert any(said in line for line in reviewed or [])  # never the sealed status
    way = committee.route("CRO", floor, NOW) or ()
    assert f"--role CRO --key {floor.key('CRO')}" in way[1] and "only final action" in way[-1]
    assert committee.route("CRO", None, NOW) is None  # no floor: the answer file stands
    directory = Path("bundle with spaces 文")
    for role in committee.ROLES[1:]:
        offered = committee.bundle_parts(role, (readback, EVIDENCE), store, NOW, str(directory))[1]
        assert f'--file "{(directory / "message.yaml").as_posix()}"' in offered[1]
        assert "<message.yaml>" not in offered[1]
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
    record = {
        **readback,
        "publication": {
            **readback["publication"],
            "risk_status": "NOT_EVALUATED_FOR_THIS_PROPOSAL",
        },
        "date_risk": risk,
    }
    view = committee.role_lines("RISK", record, EVIDENCE, None, NOW)
    assert view is not None and all(f in view[1] for f in ("1.10%", "17.5%", "H1 21%"))
    assert "Dated" in view[1] and "evaluated." in view[1].split("Historical")[0]
    linked = committee.status_word("NOT_EVALUATED_FOR_THIS_PROPOSAL")
    assert linked in view[1].split("Historical")[1] and linked.islower()  # words, not its code
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
    tmp_path: Path, readback: dict[str, Any]
) -> None:
    """The date's delivery retains its positions and attributed floor commentary."""
    request = Request(
        operation="EXPERIMENT_DELIVERY_EXPORT",
        update_task_id=UUID(TASK),
        update_publication_hash=readback["review_selector"]["update_publication_hash"],
        position_basis="CONDITIONAL_ESTIMATE",
    )
    positions = portfolio_update_positions(
        PortfolioUpdatePublication.model_validate(readback["publication"])
    )
    facts = {
        "schedule": positions.schedule.model_dump(mode="json"),
        "position_rows": [],
        "date_risk": {"state": "NOT_EVALUATED", "reason": "risk.not_recorded"},
    }
    store, now = _floor(tmp_path, readback)
    _stances(store, now)
    _say(store, now, "PM", "VERDICT", outcome="PROCEED")
    floor = committee.floor_of(store, TASK)
    assert floor is not None
    context = committee.context(floor, now[0])
    said = committee.commentary(floor, now[0])
    assert said is not None
    delivered = export_update_delivery(
        request=request,
        readback={**readback, **facts, "committee_context": context},
        review=None,
        committee=said,
    )
    assert delivered["committee_context"] == context
    with pytest.raises(AuthoringError, match=r"^research_delivery\.update_subject_mismatch$"):
        export_update_delivery(
            request=replace(request, position_basis="OBSERVED_RESEARCH_ENTRY"),
            readback={**readback, **facts, "committee_context": context},
            review=None,
            committee=said,
        )
    assert delivered["commentary_provenance"]["attribution"] == "COMMITTEE_FLOOR"
    # A floor opened on a published review offers the report that carries it.
    reviewed = {**readback, "review_standing": {"status": "REVIEWED"}}
    reviewed["review_standing"]["review_publication_hash"] = "c" * 64
    store, now = _floor(tmp_path / "reviewed", reviewed)
    assert _read(store, now, None, 0)["delivery"]["review_publication_hash"] == "c" * 64
    assert delivered["commentary"] == said
    assert delivered["sections"]["positions"]["value"] == {
        "html": readback["html"],
        "basis": request.position_basis,
        **facts,
    }
    assert delivered["selection"] == {
        "operation": "EXPERIMENT_DELIVERY_EXPORT",
        **readback["review_selector"],
    }
    assert (
        worded(said[0]["attribution_word"]).format(outcome=worded("Proceed")) in delivered["html"]
    )
