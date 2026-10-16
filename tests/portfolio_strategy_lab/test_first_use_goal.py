"""First-use authority is recorded, bounded and never silently widened."""

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
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

import alphalattice.interface.local_application.portfolio_research as research_requests
from alphalattice.control.product_host.composition import (
    portfolio_research_operations as operations,
)
from alphalattice.control.product_host.composition.evidence_source_ways import (
    short_source_action,
    source_ways,
)
from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioResearchOperations,
)
from alphalattice.control.product_host.composition.research_update_automation import (
    first_update,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.data_preparation.application import (
    WorkspacePreparationApplication,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.workspace_runtime.network_access import NetworkAccess, network_access
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
)
from alphalattice.interface.local_application.goals import (
    DAILY_NETWORK_ASK,
    FIRST_USE_HOURS,
    target_sessions,
)
from tests.portfolio_strategy_lab.local_web_support import _json, run_node
from tests.researcher_methodology_surface.real_workspace import _source_loader_for
from tests.workspace_maintenance.local_data_provider import recording_provider

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"
Request = research_requests.PortfolioResearchOperationRequest
PortfolioResearchRequestDocument = research_requests.PortfolioResearchRequestDocument
decision_hash = research_requests.decision_hash
SESSION = "00000000-0000-4000-8000-0000000000f1"
OTHER_SESSION = "00000000-0000-4000-8000-0000000000f2"
RELAY = ("--person-said", "Yes, go ahead.", "--asked", "May I widen the SEC budget?")
INSTALL = "--setup=--acquire-sec --entities AAPL --network-consent --install"
FIRST_USE = {
    "title": "First use",
    "objective": "Build me a reviewed book from public data.",
    "kind": "FIRST_USE",
    "criteria": [{"criterion_id": "book", "text": "A reviewed book stands for the person."}],
}


def _cli(live: Any, *arguments: str, session: str = SESSION) -> tuple[int, dict[str, Any]]:
    """One agent session's command, as its CLI sends it."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(live.workspace),
            "--view",
            "full",
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CLAUDE_CODE_SESSION_ID": session},
    )
    return result.returncode, json.loads(result.stdout)["data"]


def _code(answer: dict[str, Any]) -> str | None:
    """A refusal's code, returned by its owner or raised and answered by the Host."""
    return answer.get("failure_code") or answer.get("refused")


def _network(live: Any) -> bool:
    """The workspace's own network control, read past the tests' offline switch."""
    return network_access(live.workspace, environment={}, now=live.clock()).allowed


def test_a_first_use_goal_lets_its_agent_take_the_first_steps_and_ends_with_them(
    live: Any, tmp_path: Path
) -> None:
    """A first use goal lets its agent take the first steps and ends with them."""

    plan = "a" * 64
    code, refused = _cli(live, "network", "set", "--enabled", "true")
    assert code == 2 and refused["failure_code"] == "local_application.network_access_human_only"
    code, refused = _cli(live, "preparation", "confirm", "--plan", plan)
    assert code == 2 and _code(refused) == "workspace_preparation.human_confirmation_required"
    declaration = tmp_path / "first-use.json"
    declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    code, set_ = _cli(live, "network", "set", "--enabled", "true")
    assert code == 0 and set_["status"] == "NETWORK_ACCESS", set_
    assert _network(live)
    code, confirmed = _cli(live, "preparation", "confirm", "--plan", plan)
    assert _code(confirmed) != "workspace_preparation.human_confirmation_required", confirmed
    # A step the goal does not delegate stays the person's.
    code, storage = _cli(live, "storage", "confirm", "--plan", plan)
    assert code == 2 and _code(storage) == "storage.human_confirmation_required"
    code, shown = _cli(live, "goal", "show", opened["goal_id"])
    # The goal offers no revision of the person's sentence (the review at 118f6378), and its
    # record says its delegation and its end (U70).
    assert "revise" not in shown["next_requests"]
    assert shown["record"]["delegation"]["active"] is True
    assert set(shown["record"]["delegation"]["steps"]) >= {"NETWORK_ACCESS_SET"}
    code, network = _cli(live, "network", "show")
    assert network["set_by"]["delegation"] == f"first-use-goal:{opened['goal_id']}"
    code, decisions = _cli(live, "decision", "list")
    first = next(d for d in decisions["decisions"] if d["kind"] == "FIRST_USE")
    assert first["next_requests"]["stop"]["operation"] == "GOAL_ABANDON"
    steps = shown["record"]["delegated_steps"]
    assert [s["operation"] for s in steps] == ["NETWORK_ACCESS_SET", "WORKSPACE_PREPARE_CONFIRM"]
    assert steps[0]["network_enabled"] is True
    again = tmp_path / "again.json"
    again.write_text(json.dumps({**FIRST_USE, "objective": "Another sentence."}), encoding="utf-8")
    code, second = _cli(live, "goal", "open", "--file", str(again))
    assert code == 2 and second["failure_code"] == "goal.first_use_already_opened"
    code, ended = _cli(live, "goal", "abandon", opened["goal_id"], "--reason", "Stopped")
    assert code == 0 and ended["state"] == "ABANDONED"
    assert not _network(live)
    code, refused = _cli(live, "network", "set", "--enabled", "true")
    assert code == 2 and refused["failure_code"] == "local_application.network_access_human_only"


def test_sec_source_consent_is_the_persons_outside_a_first_use(live: Any) -> None:
    """requirement: outside a first use only the person, or their relayed yes, gives SEC source
    consent or installs, within the source policy's bounds; a malformed setup is refused."""
    code, refused = _cli(live, "evidence-consent", "set", "--per-issuer", "3")
    assert code == 2 and refused["failure_code"] == "evidence_review.consent_human_only"
    code, refused = _cli(live, "evidence", "install", INSTALL)
    assert code == 2 and refused["failure_code"] == "evidence_review.consent_human_only"
    # Each names its reason in the parser's own words (an unquoted path with a space hid it).
    for malformed, why in (
        ('--setup=--entities "AAPL', "closing quotation"),
        (INSTALL.removesuffix(" --install"), "--install"),
        ("--setup=--source-artifact-root D:/a b --install", "unrecognized arguments: b"),
    ):
        code, invalid = _cli(live, "evidence", "install", malformed)
        assert code == 2 and invalid["failure_code"] == "local_client.request_invalid", invalid
        assert why in invalid["reasons"]["evidence_setup"], invalid
    two = INSTALL.replace(" --network", " --source-artifact-root x --network")
    code, both = _cli(live, "evidence", "install", two)
    assert code == 2 and "--source-set-hash" in both["detail"], both
    code, admitted = _cli(live, "evidence", "install", INSTALL, *RELAY)
    assert code == 3 and admitted["status"] == "ADMITTED", admitted
    invalid = live.operations.execute(
        Request(operation="EVIDENCE_CONSENT_SET", evidence_documents_per_issuer=2), caller="HUMAN"
    )
    assert invalid["failure_code"] == "evidence_review.consent_budget_invalid"
    given = live.operations.execute(
        Request(operation="EVIDENCE_CONSENT_SET", evidence_documents_per_issuer=5), caller="HUMAN"
    )
    assert given["consent"]["actor"] == "HUMAN" and given["consent"]["documents_per_issuer"] == 5


def test_a_first_use_gives_the_default_sec_budget_until_its_end(live: Any, tmp_path: Path) -> None:
    """requirement: under the goal the agent gives the default budget, recorded as the goal's
    delegation, more is refused by name, and the goal's end takes the consent back."""
    declaration = tmp_path / "first-use.json"
    declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    code, given = _cli(live, "evidence-consent", "set", "--per-issuer", "3")
    assert code == 0 and given["consent"]["actor"] == f"first-use-goal:{opened['goal_id']}"
    code, wider = _cli(live, "evidence-consent", "set", "--per-issuer", "5")
    assert code == 2 and wider["failure_code"] == "evidence_review.consent_beyond_delegation"
    for beyond in ("--maximum-documents-per-issuer 5", "--analyst-timeout-seconds 30"):
        code, wider = _cli(
            live, "evidence", "install", INSTALL.replace(" --network", f" {beyond} --network")
        )
        assert code == 2 and wider["failure_code"] == "evidence_review.consent_beyond_delegation"
    code, _ended = _cli(live, "goal", "abandon", opened["goal_id"], "--reason", "Stopped")
    assert code == 0
    code, shown = _cli(live, "evidence-consent", "show")
    assert code == 0 and shown["consent"]["actor"] == "DEFAULT"


def test_a_first_use_never_raises_the_persons_sec_consent(live: Any, tmp_path: Path) -> None:
    """regression: a delegated agent could replace the person's narrower consent with the
    default budget; the delegation gives nothing past what the person gave, unless relayed."""
    narrower = live.operations.execute(
        Request(
            operation="EVIDENCE_CONSENT_SET",
            evidence_documents_per_issuer=3,
            evidence_total_bytes=1_000_000_000,
        ),
        caller="HUMAN",
    )
    assert narrower["consent"]["total_bytes"] == 1_000_000_000
    declaration = tmp_path / "first-use.json"
    declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    code, raised = _cli(
        live, "evidence-consent", "set", "--per-issuer", "3", "--bytes", "3000000000"
    )
    assert code == 2 and raised["failure_code"] == "evidence_review.consent_beyond_delegation"
    code, relayed = _cli(
        live, "evidence-consent", "set", "--per-issuer", "3", "--bytes", "3000000000", *RELAY
    )
    assert code == 0 and relayed["consent"]["total_bytes"] == 3_000_000_000, relayed
    code, shown = _cli(live, "evidence-consent", "show")
    assert code == 0 and shown["consent"]["actor"] == "HUMAN"


@pytest.mark.parametrize("relayed", [False, True])
def test_a_first_use_ends_without_undoing_what_the_person_set_since(
    live: Any, tmp_path: Path, relayed: bool
) -> None:
    """regression (V460 at b39e55f2; a daily update re-asked, 2026-10-15): the person opened the
    network in Settings, or by a yes relayed during the first use, and the goal's end closed it.
    Its end undoes only a setting its delegation still holds."""

    declaration = tmp_path / "first-use.json"
    declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    code, _set = _cli(live, "network", "set", "--enabled", "true")
    assert code == 0 and _network(live)
    if relayed:
        yes = ("--person-said", "Yes.", "--asked", "Daily network?")
        code, person = _cli(live, "network", "set", "--enabled", "true", *yes)
        assert code == 0 and person["set_by"] == {"relayed_by": f"claude-code:{SESSION}"}
    else:
        person = live.operations.execute(
            Request(operation="NETWORK_ACCESS_SET", network_enabled=True), caller="HUMAN"
        )
        assert "set_by" not in person
    code, ended = _cli(live, "goal", "abandon", opened["goal_id"], "--reason", "Stopped")
    assert code == 0 and ended["state"] == "ABANDONED"
    assert _network(live)


@pytest.mark.parametrize("relayed", [False, True])
def test_the_persons_own_record_wins_over_a_first_uses_delegation(
    live: Any, tmp_path: Path, relayed: bool
) -> None:
    """regression (the review of a36e210c4): a delegated `network set` or consent overwrote the
    person's own record, a reopen was offered and taken after their stop, and their stop left
    SEC consent standing for the network's next opening."""

    def person(line: tuple[str, ...], **request: Any) -> None:
        if relayed:
            code, answer = _cli(live, *line, "--person-said", "Yes.", "--asked", "May I?")
            assert code == 0, answer
        else:
            live.operations.execute(Request(**request), caller="HUMAN")

    declaration = tmp_path / "first-use.json"
    declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    consent, network_on = ("evidence-consent", "set", "--per-issuer", "3"), ("network", "set")
    person(consent, operation="EVIDENCE_CONSENT_SET", evidence_documents_per_issuer=3)
    person((*network_on, "--enabled", "true"), operation="NETWORK_ACCESS_SET", network_enabled=True)
    _cli(live, *consent), _cli(live, *network_on, "--enabled", "true")
    assert _cli(live, "evidence-consent", "show")[1]["consent"]["actor"] == "HUMAN"
    assert "delegation" not in _cli(live, "network", "show")[1].get("set_by", {})
    goal, delegation = (
        RequestProvenance(goal_id=opened["goal_id"]),
        f"first-use-goal:{opened['goal_id']}",
    )
    assert live.operations.goals.network_delegation(goal, "EXTERNAL_AUTOMATION") == delegation
    stop = (*network_on, "--enabled", "false")
    person(stop, operation="NETWORK_ACCESS_SET", network_enabled=False)
    _cli(live, *network_on, "--enabled", "true")
    shown = network_access(live.workspace, environment={}, now=live.clock()).body(
        delegation=delegation
    )
    assert not _network(live) and shown["next_action"] == "ASK_A_PERSON_TO_ALLOW_NETWORK_ACCESS"
    # The same day's Evidence review offers the person's one-line ask, never a delegated set
    # (the review of a28b30c73: it still offered the default budget as the agent's to give).
    ways = {
        "source_ways": source_ways(
            live.workspace, entities=None, book=None, delegation=delegation, now=live.clock()
        )
    }
    assert "next_requests" not in ways["source_ways"], ways
    assert short_source_action(ways) == "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE"
    code, again = _cli(live, *consent)
    assert code == 2 and again["failure_code"] == "evidence_review.consent_beyond_delegation"
    assert _cli(live, "goal", "abandon", opened["goal_id"], "--reason", "Stopped")[0] == 0
    assert not _network(live)
    assert _cli(live, "evidence-consent", "show")[1]["consent"]["actor"] == "REVOKED"


def test_the_daily_network_yes_answers_only_its_two_requests(live: Any) -> None:
    """requirement (the review of a36e210c4): the daily question's yes is a typed Host decision,
    the network on and SEC filings at the default budget, and answers nothing else."""

    yes = ("--person-said", "Yes.", "--asked", DAILY_NETWORK_ASK)
    for beyond in (
        ("evidence-consent", "set", "--per-issuer", "10"),
        ("network", "set", "--enabled", "false"),
    ):
        code, refused = _cli(live, *beyond, *yes)
        assert code == 2 and _code(refused) == "person_confirmation.beyond_its_question", refused
    assert _cli(live, "network", "set", "--enabled", "true", *yes)[0] == 0 and _network(live)
    code, given = _cli(live, "evidence-consent", "set", "--per-issuer", "3", *yes)
    assert code == 0 and given["consent"]["actor"] == "HUMAN", given


def test_a_persons_yes_relayed_whole_completes_their_decision_once(
    live: Any, tmp_path: Path
) -> None:
    """requirement (OP23): their yes, relayed whole, completes their decision, recorded with the
    session; the same command run again, or the yes carried to another decision, is refused,
    and their next yes goes with the nonce the refusal named."""

    line = ("network", "set", "--enabled", "true", "--person-said", "Yes.", "--asked", "Open it?")
    code, set_ = _cli(live, *line)
    assert code == 0 and _network(live), set_
    assert _cli(live, "network", "show")[1]["set_by"] == {"relayed_by": f"claude-code:{SESSION}"}
    code, again = _cli(live, *line)
    used = str(_code(again))
    assert code == 2 and used.startswith("person_confirmation.already_used:"), again
    assert _cli(live, *line, "--repeat", used.rpartition(":")[2])[0] == 0
    store = live.operations.goals.store
    relayed = [
        e["relayed"] for g in store.goal_ids() for e in store.attributed(g) if "relayed" in e
    ]
    assert [(r["words"], r["agent_session"]) for r in relayed] == [("Yes.", SESSION)] * 2
    yes = {key: relayed[0][key] for key in ("question", "words", "decision_hash")}
    path = tmp_path / "relayed.json"
    request = {"operation": "NETWORK_ACCESS_SET", "network_enabled": False}
    path.write_text(json.dumps({**request, "person_confirmation": yes}))
    code, refused = _cli(live, "request", "--file", str(path))
    assert code == 2 and _code(refused) == "person_confirmation.decision_mismatch", refused


def test_a_relayed_yes_is_held_once_and_only_from_an_agents_own_session(
    live: Any, tmp_path: Path
) -> None:
    """regression (the review at 8c142c9c3): the yes was checked before its decision ran and
    recorded after, so two copies at once could both pass; one sent from no agent session
    was not refused."""

    request = {"operation": "NETWORK_ACCESS_SET", "network_enabled": True}
    yes = {"question": "Open it?", "words": "Yes.", "decision_hash": decision_hash(request)}
    path = tmp_path / "relayed.json"
    path.write_text(json.dumps({**request, "person_confirmation": yes}))
    code, alone = _cli(live, "request", "--file", str(path), session="")
    assert code == 2 and _code(alone) == "person_confirmation.session_required", alone
    with ThreadPoolExecutor(2) as pool:
        codes = sorted(pool.map(lambda _: _cli(live, "request", "--file", str(path))[0], (1, 2)))
    assert codes == [0, 2]


def test_a_first_use_delegates_only_its_steps_for_its_hours_and_is_never_revised(
    live: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-use authority expires, while every caller keeps its open immutable date cap."""
    caller = "EXTERNAL_AUTOMATION"
    now = [datetime(2026, 10, 2, tzinfo=UTC)]
    app = live.operations.goals
    monkeypatch.setattr(app, "clock", lambda: now[0])
    monkeypatch.setattr(app, "task_facts", {}.get)
    opening = Request(
        operation="GOAL_OPEN",
        goal_id=UUID(int=7),
        goal_declaration={**FIRST_USE, "target_date": "2026-10-02"},
        change_reason="The person's sentence",
    )
    for named, code, nearest in (
        ("2026-09-07", "first_use.date_not_session", "2026-09-08"),
        ("2026-10-06", "first_use.date_after_clock", "2026-10-02"),
    ):
        declaration = {**FIRST_USE, "target_date": named}
        refused = app.operate(replace(opening, goal_declaration=declaration), caller)
        assert refused["failure_code"] == code and refused["target_date"] == named
        assert refused["nearest_date"] == nearest and app.first_use() is None
    app.operate(opening, caller)
    goal = app.store.head(UUID(int=7))
    assert goal is not None and app.first_use() == goal
    delegated = app.delegation(goal, "NETWORK_ACCESS_SET", caller)
    assert delegated == f"first-use-goal:{goal.goal_id}"
    assert app.delegation(goal, "MODEL_ACTIVATE", caller) is None
    for step in ("DATA_CHANGE_CONFIRM", "STRATEGY_ACTIVATE"):
        assert app.delegation(goal, step, caller) == delegated
    assert app.delegation(goal, "STRATEGY_DEACTIVATE", caller) is None
    assert app.delegation(goal, "NETWORK_ACCESS_SET", "HUMAN") is None
    with pytest.raises(ValueError, match=r"goal\.first_use_is_not_revised"):
        revised = replace(
            opening,
            operation="GOAL_REVISE",
            goal_declaration={**FIRST_USE, "objective": "A longer window."},
            change_reason="Widened",
        )
        app.operate(revised, caller)
    publish_research_workspace_manifest(live.workspace, live.workspace_manifest)
    owner = WorkspacePreparationApplication(
        live.session,
        clock=lambda: now[0] + timedelta(days=1),
        provider=recording_provider(),
        source_loader=_source_loader_for(("AAA",)),
    )
    live.operations.preparation = owner
    uncapped = owner.plan()
    for retained in (False, True):
        if retained:
            admitted = owner.confirm(uncapped["plan_hash"], caller="HUMAN")
            snapshot = live.session.task_control_registry.task(admitted.task_id)
        refused_plan = _json(
            live,
            "/api/workspace/preparation/confirm",
            method="POST",
            payload={"preparation_plan_hash": uncapped["plan_hash"]},
        )
        assert refused_plan["failure_code"] == "workspace_preparation.preview_stale", refused_plan
        assert refused_plan["detail"] and refused_plan["next_requests"]["replan"] == {
            "operation": "WORKSPACE_PREPARE_PLAN"
        }
        if retained:
            assert live.session.task_control_registry.task(admitted.task_id) == snapshot
    task = UUID(int=8)
    app.store.attribute(goal.goal_id, {"task_id": str(task)})
    decide = app.data_rule(task)
    assert decide is not None
    before = app.store.attributed(goal.goal_id)
    body = {
        "status": "CONFIRMED_PENDING_REVALIDATION",
        "decisions_complete": True,
        "next_requests": {
            f"continue:{task}": {
                "operation": "WORKSPACE_PREPARE_CONFIRM",
                "preparation_plan_hash": "c" * 64,
            }
        },
    }

    def refused(_request):
        raise ValueError("workspace_preparation.binding_changed")

    with monkeypatch.context() as stopped:
        stopped.setattr(app, "task_facts", lambda _: ("workspace_preparation", "BLOCKED", None))
        answer = app.resume_decided(
            goal,
            Request(
                operation="DATA_ISSUE_CONFIRM",
                data_issue_case_token="a" * 64,
                data_issue_evidence_hash="b" * 64,
                data_issue_option_id="retain",
                data_issue_option_hash="c" * 64,
            ),
            body,
            refused,
        )
    assert answer["status"] == body["status"]
    assert answer["continuation_problem"]["failure_code"] == "workspace_preparation.binding_changed"
    assert answer["continuation_problem"]["detail"]
    assert answer["next_requests"] == body["next_requests"]
    case = SimpleNamespace(run_id=str(UUID(int=9)), case_token="a" * 64, evidence_hash="b" * 64)
    with monkeypatch.context() as ended:
        ended.setattr(app.store, "head", lambda _: goal.model_copy(update={"state": "ABANDONED"}))
        assert (
            decide(case, (), None, None).submission.proposal.option_id
            == "escalate_data_truth_conflict"
        )
    now[0] += timedelta(hours=FIRST_USE_HOURS)
    assert (
        decide(case, (), None, None).submission.proposal.option_id == "escalate_data_truth_conflict"
    )
    assert app.store.attributed(goal.goal_id) == before
    assert app.delegation(goal, "NETWORK_ACCESS_SET", caller) is None
    assert app.first_use_delegation(goal)["active"] is False

    def history_refused(**scope: Any) -> dict[str, Any]:
        assert scope["network_delegation"] is None
        assert scope["requested_formation_session"].isoformat() == "2026-10-01"
        raise ValueError("data.current_universe_no_feature_ready_listings")

    preparation = Request(operation="WORKSPACE_PREPARE_PLAN")
    for provenance in (
        None,
        RequestProvenance(vendor="claude-code", session=OTHER_SESSION),
        RequestProvenance(goal_id=str(goal.goal_id)),
    ):
        token = REQUEST_PROVENANCE.set(provenance)
        try:
            recovery = app.prepare(preparation, "HUMAN", history_refused)
        finally:
            REQUEST_PROVENANCE.reset(token)
    with monkeypatch.context() as web:
        web.setattr(live.operations.preparation, "plan", history_refused)
        assert _json(live, "/api/workspace/preparation/plan", method="POST", payload={}) == recovery
    assert recovery["nearest_date_support"] == "UNAVAILABLE"
    assert recovery["new_workspace"]["choose"] == ["workspace", "supported_date"]
    assert set(recovery["next_requests"]) == {"readback", "goal", "abandon"}
    show = PortfolioResearchRequestDocument.model_validate(recovery["next_requests"]["goal"])
    shown = app.operate(show.to_operation_request(), "HUMAN")
    assert shown["goal_hash"] == goal.goal_hash
    abandon = PortfolioResearchRequestDocument.model_validate(recovery["next_requests"]["abandon"])
    app.operate(abandon.to_operation_request(), "HUMAN")
    ended_goal = app.store.head(goal.goal_id)
    assert ended_goal.state == "ABANDONED" and ended_goal.declaration == goal.declaration
    past = ended_goal.declaration.model_copy(update={"target_date": datetime(2099, 1, 2).date()})
    historical_goal = ended_goal.model_copy(update={"declaration": past})
    with monkeypatch.context() as historical:
        historical.setattr(app.store, "head", lambda _: historical_goal)
        ordinary = app.prepare(preparation, "HUMAN", lambda **scope: scope)
        assert ordinary["requested_formation_session"] is None
        assert ordinary["requested_information_cutoff_at"] is None


def test_a_delegated_activation_runs_forward_without_the_history_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegated activation also admits the date's update without requiring historical review."""

    owner = SimpleNamespace(
        activations=SimpleNamespace(activate=lambda _task: {"status": "ACTIVATED"}),
    )
    update = {"update": {"run": {"task_id": "u"}}}
    monkeypatch.setattr(operations, "first_update", lambda _host, _activated, _given: update)
    request = Request(operation="STRATEGY_ACTIVATE", task_id=UUID(int=3))
    scope = REQUEST_PROVENANCE.set(
        RequestProvenance(goal_id=str(UUID(int=7)), delegation=f"first-use-goal:{UUID(int=7)}")
    )
    try:
        answer = PortfolioResearchOperations._strategy_activation(owner, request, "HUMAN")  # type: ignore[arg-type]
    finally:
        REQUEST_PROVENANCE.reset(scope)
    assert answer["status"] == "ACTIVATED" and answer["update"] == {"run": {"task_id": "u"}}


def test_an_activation_admits_the_first_uses_dated_update_in_the_same_act(tmp_path: Path) -> None:
    """Activation admits the first-use date's update and wakes the lead under its Goal."""

    sent: list[Request] = []
    wakes: list[tuple[object, ...]] = []
    goal = SimpleNamespace(
        goal_id=UUID(int=7), state="OPEN", declaration=SimpleNamespace(target_date=None)
    )
    goal.declaration.target_date = datetime(2026, 10, 10).date()

    def execute(request: Request) -> dict[str, object]:
        sent.append(request)
        if request.operation == "RESEARCH_UPDATE_PLAN":
            return {"status": "PLANNED", "update_plan_hash": "h" * 64, "target_session": "x"}
        return {"status": "ADMITTED", "task_id": str(UUID(int=9))}

    codex = {"agent_vendor": "codex", "agent_session": "thread-1"}
    owner = SimpleNamespace(
        goals=SimpleNamespace(
            first_use=lambda: goal,
            first_use_delegation=lambda _goal: {"active": True},
            store=SimpleNamespace(attributed=lambda _goal_id: (codex,)),
        ),
        execute=execute,
        workspace_session=SimpleNamespace(
            workspace=tmp_path,
            task_control_registry=SimpleNamespace(
                register_wake=lambda *args, **_: wakes.append(args)
            ),
        ),
        dispatcher=SimpleNamespace(clock=lambda: datetime(2026, 10, 10, tzinfo=UTC)),
    )
    activated = {
        "strategy_package_id": "BAL",
        "next_requests": {"update": {"operation": "RESEARCH_UPDATE_PLAN"}},
    }
    answer = first_update(owner, activated, None)  # type: ignore[arg-type]

    # A Saturday names Monday's positions, decided at Friday's close.
    assert sent[0].observed_through == "2026-10-09"
    assert sent[1].update_plan_hash == "h" * 64
    assert answer["next_requests"] == {
        "update_status": {"operation": "STATUS", "task_id": str(UUID(int=9))}
    }
    ((task, thread, read),) = wakes
    assert (task, thread) == (UUID(int=9), "thread-1") and read.endswith(f"task show {UUID(int=9)}")
    # A refused plan's offered update keeps the named date, never the latest session's (F10).
    refusing = SimpleNamespace(**{**vars(owner), "execute": lambda _request: {"status": "REFUSED"}})
    kept = first_update(refusing, activated, None)  # type: ignore[arg-type]
    assert kept["next_requests"]["update"]["observed_through"] == "2026-10-09"
    # A date the calendars do not plan leaves the committed activation standing (the review),
    # and offers no update at all.
    goal.declaration.target_date = datetime(2099, 1, 2).date()
    outside = first_update(owner, activated, None)  # type: ignore[arg-type]
    assert outside["update"]["plan"]["failure_code"] == "first_use.date_outside_calendar"
    assert "update" not in outside["next_requests"]
    # Asked at 11:00 New York for Monday, before Friday's data settles: held until 18:00 ET,
    # run once then, and kept in the goal's ledger for a Host that starts again.
    goal.declaration.target_date = datetime(2026, 10, 12).date()
    holds: list[datetime] = []
    kept_entries: list[dict[str, object]] = []
    early = SimpleNamespace(
        **{
            **vars(owner),
            "execute": lambda _request: {
                "status": "REFUSED",
                "failure_code": "research_update.target_not_completed",
            },
            "dispatcher": SimpleNamespace(clock=lambda: datetime(2026, 10, 9, 15, tzinfo=UTC)),
            "automation": SimpleNamespace(hold=lambda at, _run: holds.append(at)),
        }
    )
    early.goals.store.attribute = lambda _goal, entry: kept_entries.append(entry)
    plan = first_update(early, activated, None)["update"]["plan"]  # type: ignore[arg-type]
    assert holds == [datetime(2026, 10, 9, 22, tzinfo=UTC)] and "18:00 ET" in plan["reading"]
    assert plan["held_until"] == holds[0].isoformat() and kept_entries[0]["held_update"]
    # A reader outside New York also gets their own time, with its date where the day differs.
    asked = datetime(2026, 10, 9, 15, tzinfo=UTC)
    west = target_sessions(goal.declaration.target_date, asked, ZoneInfo("America/Los_Angeles"))
    east = target_sessions(goal.declaration.target_date, asked, ZoneInfo("Asia/Shanghai"))
    assert "(15:00 your time)" in str(west["reading"]) and "18:00 ET" in str(west["reading"])
    assert "06:00 on 2026-10-10" in str(east["reading"])


def test_a_first_use_is_the_one_before_the_first_preparation(
    live: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (V452): a workspace already prepared has had its first use, so a first-use
    goal is refused there and the person decides each step."""

    succeeded = SimpleNamespace(lifecycle=SimpleNamespace(value="SUCCEEDED"))
    monkeypatch.setattr(live.operations.preparation, "tasks", lambda: [succeeded])
    answer = live.operations.execute(
        Request(operation="GOAL_OPEN", goal_declaration=FIRST_USE, change_reason="First"),
        caller="EXTERNAL_AUTOMATION",
    )
    assert answer["failure_code"] == "goal.first_use_after_preparation", answer


def test_a_first_use_decides_its_own_preparations_data_issue_and_its_preparation_goes_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the Host rule's cap the first use's agent decides each case, recorded and shown; its
    last decision resumes the same stopped Task in place (OP4, OP19)."""
    from alphalattice.control.data_platform import remediation_case
    from alphalattice.control.data_platform.contracts import (
        DataRemediationExecutionReceipt,
        DataRemediationExecutionSubmission,
    )
    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from alphalattice.foundation.feature_engine.inputs.gateway import FeatureInputExecutionStatus
    from alphalattice.foundation.market_data_ops.runtime.remediation import canonical_hash
    from alphalattice.protocols.actor_execution import ActorKind
    from tests.researcher_methodology_surface.real_workspace import OBSERVED_AT, _source_loader_for
    from tests.workspace_readiness.unexplained_move import unexplained_move

    provider, symbols = unexplained_move(listings=3)
    live = LocalPortfolioWebSession.from_workspace(
        tmp_path / "first-use", clock=lambda: OBSERVED_AT
    )
    live.data_provider = provider
    live.data_source_loader = _source_loader_for(symbols)

    def send(document: dict[str, Any], session: str = SESSION) -> tuple[int, dict[str, Any]]:
        path = tmp_path / f"request-{len(list(tmp_path.glob('request-*')))}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return _cli(live, "request", "--file", str(path), session=session)

    with live:
        code, budget = send(
            {"operation": "CPU_BUDGET_SET", "cpu_budget": str(min(4, os.cpu_count() or 1))}
        )
        assert code == 0 and _code(budget) is None, budget
        declaration = tmp_path / "first-use.json"
        declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
        code, opened = _cli(live, "goal", "open", "--file", str(declaration))
        assert code == 0, opened
        delegation = f"first-use-goal:{opened['goal_id']}"

        def closed(workspace: Path, *, now: datetime) -> NetworkAccess:
            assert now == OBSERVED_AT  # the Host's clock, never the machine's
            return NetworkAccess(False, "DEFAULT")

        with monkeypatch.context() as patch:
            patch.setattr(live.operations.preparation, "provider", None)
            patch.setattr(
                "alphalattice.control.product_host.data_preparation.application.network_access",
                closed,
            )
            patch.setattr(
                "alphalattice.control.product_host.composition.portfolio_research_operations.network_access",
                closed,
            )
            code, preview = _cli(live, "preparation", "plan")
            code, hint = send({"operation": "NETWORK_ACCESS"})
            assert all(hint[k] == v for k, v in preview["source_access"]["network_access"].items())
            token = REQUEST_PROVENANCE.set(RequestProvenance(goal_id=opened["goal_id"]))
            try:
                human = live.operations.execute(Request(operation="NETWORK_ACCESS"), caller="HUMAN")
            finally:
                REQUEST_PROVENANCE.reset(token)
            assert "delegation" not in human
            assert human["next_action"] == "ASK_A_PERSON_TO_ALLOW_NETWORK_ACCESS"
        code, offline = send({"operation": "NETWORK_ACCESS"})
        assert offline["decided_by"] == "OPERATOR_OFFLINE_SWITCH"
        assert "delegation" not in offline and offline["next_requests"] == {}
        access = preview["source_access"]["network_access"]
        assert access["delegation"] == delegation
        assert access["next_action"] == "SET_NETWORK_UNDER_FIRST_USE_DELEGATION"
        assert access["next_requests"]["set"] == {
            "operation": "NETWORK_ACCESS_SET",
            "network_enabled": True,
        }
        code, plan = _cli(live, "preparation", "plan")
        code, started = send(plan["next_requests"]["confirm"])
        assert started["status"] == "ADMITTED", started  # exit 3: the queued Task is pending
        live.dispatcher.drain_for_tests(timeout=900)
        registry = live.session.task_control_registry
        stopped = registry.task(UUID(started["task_id"]))
        assert stopped.failure_code == "data.truth_review_required", stopped

        def page_projection() -> dict[str, Any]:
            requests = {
                "preparation": Request(operation="WORKSPACE_PREPARE_READBACK"),
                "issues": Request(operation="DATA_ISSUES"),
                "decisions": Request(operation="PENDING_DECISIONS"),
                "recovery": Request(operation="TASK_RECOVERY", task_id=stopped.task_id),
                "dataUpdate": Request(operation="DATA_UPDATE_READBACK"),
                "activity": Request(operation="ACTIVITY_LIST", limit=200, watch=(stopped.task_id,)),
            }
            return {
                key: live.operations.execute(request, caller="HUMAN")
                for key, request in requests.items()
            }

        blocked_page = page_projection()

        code, issues = _cli(live, "issue", "list")
        assert issues["delegated_by"] == delegation
        code, prepared = _cli(
            live,
            "bundle",
            "prepare",
            "--role",
            "DATA",
            "--task",
            str(stopped.task_id),
            "--dir",
            str(tmp_path / "data-bundle"),
        )
        assert code == 0 and prepared["status"] == "AGENT_BUNDLE_READY", prepared
        material = "\n".join(
            (tmp_path / "data-bundle" / file["name"]).read_text("utf-8")
            for file in prepared["files"][1:]
        )
        view = json.loads(material[material.index("{") :])
        assert view["delegated_by"] == issues["delegated_by"] == delegation
        assert str(stopped.task_id) in material
        for issue in issues["issues"]:
            case = issue["case"]
            assert case["case_token"] in material and case["evidence_hash"] in material
            assert all(option["option_id"] in material for option in case["options"])
        # Three names of 120 move at once, past the rule's 2%: the rule leaves each to the agent.
        assert issues["escalated_by_rule"]["listing_count"] == 3, issues
        escalated = [
            e
            for e in live.operations.goals.store.attributed(UUID(opened["goal_id"]))
            if e.get("operation") == "DATA_ISSUE_CONFIRM" and e.get("decided_by") == "RULE"
        ]
        assert {(e["status"], e["option_id"]) for e in escalated} == {
            ("ESCALATED", "escalate_data_truth_conflict")
        } and len(escalated) == 3, escalated
        assert not any(name.startswith("delegate:") for name in issues["next_requests"])
        previews = sorted(name for name in issues["next_requests"] if name.startswith("preview:"))
        choice = min(previews, key=lambda name: ("retain" not in name, name))
        code, preview = send(issues["next_requests"][choice])
        assert preview["confirmation"] == "FIRST_USE_DELEGATION", preview
        assert preview["executors"]["delegated_confirmation"] == delegation
        # The list already offers that confirm: the agent needs no preview to decide.
        confirm = issues["next_requests"][choice.replace("preview:", "confirm:", 1)]
        assert confirm == preview["next_requests"]["confirm"]

        # Another agent, outside the goal, may not decide it.
        code, refused = send(preview["next_requests"]["confirm"], session=OTHER_SESSION)
        assert code == 2 and _code(refused) == "feature_input.human_confirmation_required"
        # A stale case, or a choice the case never offered, is refused with the way back.
        offered = preview["next_requests"]["confirm"]
        for wrong, failure in (
            (
                {**offered, "data_issue_case_token": "0" * 64},
                "feature_input.case_no_longer_pending",
            ),
            (
                {**offered, "data_issue_option_id": "use_unique_verified_alias_for_this_run"},
                "feature_input.option_changed",
            ),
        ):
            code, refused = send(wrong)
            assert code == 2 and _code(refused) == failure, refused
            assert refused["detail"] and refused["next_action"] == "DATA_ISSUES"
            [read] = list(refused["next_requests"].values())
            assert read == {"operation": "DATA_ISSUES"}
            code, recovered = send(read)
            assert code == 0 and recovered["case_count"] == len(issues["issues"]), recovered
            assert registry.task(stopped.task_id) == stopped  # A recovery read applies no choice.

        code, decided = send(preview["next_requests"]["confirm"])
        assert code == 0 and _code(decided) is None, decided
        page = page_projection()
        resolution = next(
            i for i in page["issues"]["issues"] if i["case"]["case_token"] == decided["case_token"]
        )["resolution"]
        assert resolution["receipt"]["actor_submission"]["actor_kind"] == "EXTERNAL_AUTOMATION"
        assert resolution["receipt"]["actor_submission"]["actor_id"] == f"claude-code:{SESSION}"
        assert resolution["receipt"]["submission"]["proposal"]["rationale"] == delegation
        operations = [
            i["payload"]
            for i in page["activity"]["items"]
            if i["payload"].get("operation") == "DATA_ISSUE_CONFIRM"
            and i["payload"].get("subject", {}).get("delegation") == delegation
        ]
        assert operations and all(p["caller"] == "EXTERNAL_AUTOMATION" for p in operations)
        receipt = DataRemediationExecutionReceipt.model_validate(resolution["receipt"])
        payload = receipt.submission.model_dump(mode="json", exclude={"submission_hash"})
        payload["proposal"]["rationale"] = "legacy decision"
        submission = DataRemediationExecutionSubmission(
            **payload, submission_hash=canonical_hash(payload)
        )
        common = receipt.model_dump(exclude={"kind", "actor_submission", "receipt_hash"})
        common["submission"] = submission
        legacy = remediation_case.seal_validated_data_remediation_execution(
            **common, actor_kind=ActorKind.HUMAN, actor_id=delegation
        )
        assert _cli(live, "goal", "take", opened["goal_id"], session=OTHER_SESSION)[0] == 0
        goals = live.operations.goals
        assert goals.recorded_data_confirmation(receipt)
        assert not goals.recorded_data_confirmation(legacy)
        foreign = remediation_case.seal_validated_data_remediation_execution(
            **receipt.model_dump(exclude={"kind", "actor_submission", "receipt_hash"}),
            actor_kind=ActorKind.EXTERNAL_AUTOMATION,
            actor_id=f"claude-code:{OTHER_SESSION}",
        )
        assert not goals.recorded_data_confirmation(foreign)
        accepted = next(
            e
            for e in goals.store.attributed(UUID(opened["goal_id"]))
            if e.get("status") == "CONFIRMED_PENDING_REVALIDATION"
        )
        for wrong in (
            {},
            *(
                {**accepted, key: value}
                for key, value in (
                    ("operation", "DATA_ISSUE_PREVIEW"),
                    ("status", "REFUSED"),
                    ("case_token", "0" * 64),
                    ("option_id", "other"),
                    ("agent_session", OTHER_SESSION),
                    ("delegation", "other"),
                    ("recorded_at", (OBSERVED_AT - timedelta(seconds=1)).isoformat()),
                    ("recorded_at", (OBSERVED_AT + timedelta(hours=FIRST_USE_HOURS)).isoformat()),
                )
            ),
        ):
            with monkeypatch.context() as proof:
                proof.setattr(goals.store, "attributed", lambda _, entry=wrong: (entry,))
                assert not goals.recorded_data_confirmation(receipt), wrong
        head = goals.store.head(UUID(opened["goal_id"]))
        unrelated = head.model_copy(
            update={"declaration": head.declaration.model_copy(update={"kind": "RESEARCH"})}
        )
        with monkeypatch.context() as proof:
            proof.setattr(goals.store, "head", lambda _: unrelated)
            assert not goals.recorded_data_confirmation(receipt)
        with monkeypatch.context() as proof:
            proof.setattr(goals, "clock", lambda: OBSERVED_AT + timedelta(days=2))
            proof.setattr(
                goals.store, "head", lambda _: head.model_copy(update={"state": "ABANDONED"})
            )
            assert goals.recorded_data_confirmation(receipt)
        panel = live.operations.data_issues.panel
        for retained in (receipt, legacy):
            with monkeypatch.context() as readers:
                prior = {
                    "receipt": retained.model_dump(mode="json"),
                    "effect": {"failure_reasons": []},
                }
                readers.setattr(panel, "feature_input_resolution", lambda _, record=prior: record)
                code, repeated = send(preview["next_requests"]["confirm"], session=OTHER_SESSION)
                assert (code, repeated["status"]) == (0, "ALREADY_APPLIED")
                assert repeated["receipt_hash"] == retained.receipt_hash
        app_dir = SCRIPT.parent.parent / "src/alphalattice/interface/local_application/assets"
        (tmp_path / "page-feed.json").write_text(
            json.dumps({**blocked_page, "decided": page}), encoding="utf-8"
        )
        run_node(
            [
                str(Path(__file__).with_name("workbench_workspace.cjs")),
                str(app_dir / "workbench-source/js/app"),
                "--blocked-preparation",
            ],
            missing="The Workbench holder requires the installed Node runtime.",
            required=True,
            input=json.dumps({**blocked_page, "decided": page}),
            text=True,
            check=True,
            timeout=15,
        )
        # The other two decided, the last resumes the same Task in the same act.
        assert decided["decisions_complete"] is False and "continued" not in decided
        code, issues = _cli(live, "issue", "list")
        confirmations = [
            value
            for value in issues["next_requests"].values()
            if value["operation"] == "DATA_ISSUE_CONFIRM"
            and value["data_issue_option_id"] == "retain_isolated_raw_move_with_caveat"
        ]
        assert len(confirmations) == 2
        for index, request in enumerate(confirmations):
            code, last = send(request)
            assert code == (0 if index == 0 else 3) and _code(last) is None, last
        assert last["decisions_complete"] is True, last
        assert last["continued"]["status"] == "ADMITTED"
        assert last["continued"]["lifecycle"] == "QUEUED"
        assert last["continued"]["task_id"] == str(stopped.task_id), last
        assert last["next_requests"]["status"]["task_id"] == str(stopped.task_id)
        live.dispatcher.drain_for_tests(timeout=900)
        continued = registry.task(stopped.task_id)
        assert continued.lifecycle.value == "SUCCEEDED", continued.failure_code
        assert [t.task_kind for t in registry.tasks()].count("workspace_preparation") == 1
        applied = panel.feature_input_resolution(decided["case_token"])
        assert applied["receipt"] == resolution["receipt"]
        assert applied["effect"]["status"] == FeatureInputExecutionStatus.RAW_VALUE_RETAINED.value
        pending = live.operations.data_issues.readback()
        assert not any(
            r.get("data_issue_case_token") == decided["case_token"]
            for r in pending["next_requests"].values()
        )

        code, shown = _cli(live, "goal", "show", opened["goal_id"])
        step = next(
            s
            for s in shown["record"]["delegated_steps"]
            if s["operation"] == "DATA_ISSUE_CONFIRM"
            and s.get("agent_session") == SESSION
            and s["status"] == "CONFIRMED_PENDING_REVALIDATION"
            and s["case_token"] == decided["case_token"]
        )
        assert (step["delegation"], step["agent_session"]) == (delegation, SESSION)
        assert (step["case_token"], step["option_id"]) == (
            preview["next_requests"]["confirm"]["data_issue_case_token"],
            preview["next_requests"]["confirm"]["data_issue_option_id"],
        )


def test_workspace_rule_requires_its_open_owned_dated_task_and_enabled_policy(tmp_path):
    """A workspace rule authorizes its dated Task, never a closed or unrelated Goal; its audits
    count against one bound with every approval for the Task and are recorded once admitted."""
    from alphalattice.control.data_platform.data_decision_rule import (
        FULL_HISTORY,
        DataDecisionRule,
    )

    now, task, goal_id = datetime(2026, 10, 2, tzinfo=UTC), UUID(int=8), UUID(int=7)
    update, first_use = UUID(int=10), UUID(int=11)
    app = GoalApplication(
        GoalStore(tmp_path, "workspace"),
        lambda: now,
        lambda *_: {},
        lambda _: now,
        {
            task: ("portfolio_public_watermark_advancement", "RUNNING", None),
            update: ("workspace_data_update", "RUNNING", None),
        }.get,
        workspace=tmp_path,
    )
    app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_id=goal_id,
            goal_declaration={**FIRST_USE, "kind": "RESEARCH"},
            change_reason="Daily use",
        ),
        "EXTERNAL_AUTOMATION",
    )
    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("daily"))
    assert app.data_rule(task) is None
    app.store.attribute(goal_id, {"task_id": str(task), "operation": "STATUS"})
    assert app.data_rule(task) is None
    app.store.attribute(goal_id, {"task_id": str(task), "operation": "RESEARCH_UPDATE_RUN"})
    decide = app.data_rule(task)
    assert decide is not None
    assert app.data_rule(UUID(int=9)) is None
    admitted: list[str] = []

    def admit(rule_hash: str) -> str:
        admitted.append(rule_hash)
        return rule_hash

    def refuse(_rule_hash: str) -> str:
        raise ValueError("data.audit_requirement_unverified")

    def audits(goal: UUID = goal_id) -> list[tuple[str, list[str]]]:
        rows = app.store.attributed(goal)
        return [(e["status"], e["listing_ids"]) for e in rows if e.get("operation") == FULL_HISTORY]

    assert decide.audit({"L1": "AAA", "L2": "BBB"}, 100, admit)  # two of a hundred, within 2%
    with pytest.raises(ValueError, match="audit_requirement_unverified"):
        decide.audit({"L4": "DDD"}, 1000, refuse)  # written ahead, then marked not admitted
    assert decide.audit({"L4": "DDD"}, 1000, admit)  # retried, it is approved and counts
    assert decide.audit({"L1": "AAA"}, 100, admit) is None  # approved already: no second record
    assert decide.audit({"L3": "CCC"}, 100, admit) is None  # past the bound: refused
    assert decide.audit({"L9": "ZZZ"}, 10**6, admit) is None  # and the Task stays escalated
    assert audits() == [
        ("DECIDED_BY_RULE", ["L1", "L2"]),
        ("DECIDED_BY_RULE", ["L4"]),
        ("NOT_ADMITTED", ["L4"]),
        ("DECIDED_BY_RULE", ["L4"]),
        ("ESCALATED", ["L3"]),
    ]
    assert len(admitted) == 2
    assert [
        pair[0] for pair in app.store.rule_audits(str(task), app.store.attributed(goal_id))
    ] == [
        "L1",
        "L2",
        "L4",
    ]
    # Inside an open first use its delegation comes first, even for a data update's run, and its
    # step names the audit's listings; a record that fails admits nothing.
    app.operate(
        Request(operation="GOAL_OPEN", goal_id=first_use, goal_declaration=FIRST_USE),
        "EXTERNAL_AUTOMATION",
    )
    app.store.attribute(first_use, {"task_id": str(update), "operation": "DATA_UPDATE_RUN"})
    first = app.data_rule(update)
    attribute = app.store.attribute
    app.store.attribute = lambda *_: (_ for _ in ()).throw(OSError("locked"))  # type: ignore[method-assign]
    with pytest.raises(OSError):
        first.audit({"L6": "FFF"}, 100, admit)
    app.store.attribute = attribute  # type: ignore[method-assign]
    assert len(admitted) == 2 and first.audit({"L5": "EEE"}, 470, admit)
    # An ordinary ex-dividend day on 470 names: six more payers stay within the audits' own
    # 2% bound; three after them make ten, past it, and are left to the person.
    assert first.audit({f"P{n}": f"P{n}" for n in range(6)}, 470, admit)
    assert first.audit({f"Q{n}": f"Q{n}" for n in range(3)}, 470, admit) is None
    head = app.store.head(first_use)
    steps = app.record(head)["delegated_steps"]
    assert [(s["status"], s["symbols"]) for s in steps if s["operation"] == FULL_HISTORY] == [
        ("DECIDED_BY_RULE", ["EEE"]),
        ("DECIDED_BY_RULE", [f"P{n}" for n in range(6)]),
        ("ESCALATED", ["Q0", "Q1", "Q2"]),
    ]
    owned = [e for e in app.store.attributed(first_use) if e.get("operation") == FULL_HISTORY]
    assert {e["authority"] for e in owned} == {"first-use-goal"}
    publish_research_workspace_manifest(
        tmp_path,
        ResearchWorkspaceManifest.create(
            workspace_id="daily",
            default_strategy_package_id=None,
            default_score_source_mode=None,
            strategy_artifacts=(),
            data_decision_rule=DataDecisionRule(max_cases=0),
            strategy_installation="NOT_INSTALLED",
        ),
    )
    assert app.data_rule(task) is None
    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("daily"))
    app.operate(
        Request(operation="GOAL_ABANDON", goal_id=goal_id, change_reason="Finished elsewhere"),
        "EXTERNAL_AUTOMATION",
    )
    assert app.data_rule(task) is None
    assert decide.audit(("L6",), 1000, admit) is None  # its Goal's authority has ended
