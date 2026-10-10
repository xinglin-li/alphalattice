"""The first-use goal (V452, LAWS OP19): from the person's one sentence the agent runs the
first use, the first use's person-only steps delegated for the goal's life and recorded in its
ledger, and nothing granted after it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

from alphalattice.control.product_host.composition import (
    portfolio_research_operations as operations,
)
from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioResearchOperations,
)
from alphalattice.control.product_host.composition.research_update_automation import (
    first_update,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.workspace_runtime.network_access import NetworkAccess, network_access
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
)
from alphalattice.interface.local_application.goals import FIRST_USE_HOURS, target_sessions
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from alphalattice.interface.local_application.portfolio_research import decision_hash
from tests.portfolio_strategy_lab.local_web_support import run_node

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"
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
    return network_access(live.workspace, environment={}).allowed


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
    # The preparation is the person's to confirm, carried by the agent: only the plan is asked.
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
    for malformed in ('--setup=--entities "AAPL', INSTALL.removesuffix(" --install")):
        code, invalid = _cli(live, "evidence", "install", malformed)
        assert code == 2 and invalid["failure_code"] == "local_client.request_invalid", invalid
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


def test_a_first_use_ends_without_undoing_what_the_person_set_since(
    live: Any, tmp_path: Path
) -> None:
    """regression (V460, an outside review at b39e55f2): the agent opened the network under the
    first use, the person then opened it in Settings, and the goal's end closed it again. Its
    end undoes only a setting its delegation still holds."""

    declaration = tmp_path / "first-use.json"
    declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
    code, opened = _cli(live, "goal", "open", "--file", str(declaration))
    assert code == 0, opened
    code, _set = _cli(live, "network", "set", "--enabled", "true")
    assert code == 0 and _network(live)
    person = live.operations.execute(
        Request(operation="NETWORK_ACCESS_SET", network_enabled=True), caller="HUMAN"
    )
    assert "set_by" not in person
    code, ended = _cli(live, "goal", "abandon", opened["goal_id"], "--reason", "Stopped")
    assert code == 0 and ended["state"] == "ABANDONED"
    assert _network(live)


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
    tmp_path: Path,
) -> None:
    """requirement (V452, OP19): the delegation holds for the first-use steps alone, for an
    agent's request, within the goal's hours; the person's sentence is never revised."""

    now = [datetime(2026, 10, 2, tzinfo=UTC)]
    app = GoalApplication(
        GoalStore(tmp_path, "workspace"),
        lambda: now[0],
        lambda *_: {},
        lambda _: now[0],
        {}.get,
        workspace=tmp_path,
    )
    app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_id=UUID(int=7),
            goal_declaration=FIRST_USE,
            change_reason="The person's sentence",
        ),
        "EXTERNAL_AUTOMATION",
    )
    goal = app.store.head(UUID(int=7))
    assert goal is not None and app.first_use() == goal
    delegated = app.delegation(goal, "NETWORK_ACCESS_SET", "EXTERNAL_AUTOMATION")
    assert delegated == f"first-use-goal:{goal.goal_id}"
    assert app.delegation(goal, "MODEL_ACTIVATE", "EXTERNAL_AUTOMATION") is None
    # Its membership changes and its reviewed book's activation are delegated; the person
    # deactivates (STOPS-1).
    for step in ("DATA_CHANGE_CONFIRM", "STRATEGY_ACTIVATE"):
        assert app.delegation(goal, step, "EXTERNAL_AUTOMATION") == delegated
    assert app.delegation(goal, "STRATEGY_DEACTIVATE", "EXTERNAL_AUTOMATION") is None
    assert app.delegation(goal, "NETWORK_ACCESS_SET", "HUMAN") is None
    with pytest.raises(ValueError, match=r"goal\.first_use_is_not_revised"):
        app.operate(
            Request(
                operation="GOAL_REVISE",
                goal_id=goal.goal_id,
                goal_declaration={**FIRST_USE, "objective": "A longer window."},
                change_reason="Widened",
            ),
            "EXTERNAL_AUTOMATION",
        )
    now[0] += timedelta(hours=FIRST_USE_HOURS)
    assert app.delegation(goal, "NETWORK_ACCESS_SET", "EXTERNAL_AUTOMATION") is None
    assert app.first_use_delegation(goal)["active"] is False


def test_a_delegated_activation_runs_forward_without_the_history_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirement (the review on the date's positions): under the first-use delegation the
    agent activates a book whose history is not reviewed, and the date's update comes with it."""

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
    """requirement (activate and update in one act): activation plans the update for the first
    use's named date and admits its run under the goal; a Codex lead's session is woken."""

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
    """A first-use preparation's delegated data decision is recorded, shown and continued."""
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

    provider, symbols = unexplained_move()
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
        declaration = tmp_path / "first-use.json"
        declaration.write_text(json.dumps(FIRST_USE), encoding="utf-8")
        code, opened = _cli(live, "goal", "open", "--file", str(declaration))
        assert code == 0, opened
        delegation = f"first-use-goal:{opened['goal_id']}"
        with monkeypatch.context() as patch:
            patch.setattr(live.operations.preparation, "provider", None)
            patch.setattr(
                "alphalattice.control.product_host.data_preparation.application.network_access",
                lambda _root: NetworkAccess(False, "DEFAULT"),
            )
            patch.setattr(
                "alphalattice.control.product_host.composition.portfolio_research_operations.network_access",
                lambda _root: NetworkAccess(False, "DEFAULT"),
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
        code, successor_plan = send(decided["next_requests"][f"continue:{stopped.task_id}"])
        assert successor_plan["predecessor_task_id"] == str(stopped.task_id), successor_plan
        code, successor = send(successor_plan["next_requests"]["confirm"])
        assert successor["status"] == "ADMITTED", successor
        assert UUID(str(successor["task_id"])) != stopped.task_id
        live.dispatcher.drain_for_tests(timeout=900)
        continued = registry.task(UUID(str(successor["task_id"])))
        assert continued.lifecycle.value == "SUCCEEDED", continued.failure_code
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
            if s["operation"] == "DATA_ISSUE_CONFIRM" and s["agent_session"] == SESSION
        )
        assert (step["delegation"], step["agent_session"]) == (delegation, SESSION)
        assert (step["case_token"], step["option_id"]) == (
            preview["next_requests"]["confirm"]["data_issue_case_token"],
            preview["next_requests"]["confirm"]["data_issue_option_id"],
        )
