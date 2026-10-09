"""The first-use goal (V452, LAWS OP19): from the person's one sentence the agent runs the
first use, the first use's person-only steps delegated for the goal's life and recorded in its
ledger, and nothing granted after it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioResearchOperations,
)
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
)
from alphalattice.interface.local_application.goals import FIRST_USE_HOURS
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from tests.portfolio_strategy_lab.local_web_support import run_node

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_alphalattice.py"
SESSION = "00000000-0000-4000-8000-0000000000f1"
OTHER_SESSION = "00000000-0000-4000-8000-0000000000f2"
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


def test_a_delegated_activation_takes_only_a_book_with_a_published_review() -> None:
    """requirement (STOPS-1): under the first-use delegation the agent activates a book only once
    its review standing is REVIEWED, and is refused by name before; the person's own activation
    asks for no review."""

    activated: list[UUID] = []
    standing = {"status": "NOT_REVIEWED"}

    def activate(task_id: UUID) -> dict[str, object]:
        activated.append(task_id)
        return {"status": "ACTIVATED"}

    owner = SimpleNamespace(
        activations=SimpleNamespace(activate=activate),
        review=object(),
        _book_review_standing=lambda _task_id: standing,
    )
    request = Request(operation="STRATEGY_ACTIVATE", task_id=UUID(int=3))

    def run() -> dict[str, object]:
        return PortfolioResearchOperations._strategy_activation(owner, request, "HUMAN")  # type: ignore[arg-type]

    scope = REQUEST_PROVENANCE.set(
        RequestProvenance(goal_id=str(UUID(int=7)), delegation=f"first-use-goal:{UUID(int=7)}")
    )
    try:
        refused = run()
        assert refused["failure_code"] == "strategy_activation.review_required", refused
        assert not activated
        standing["status"] = "REVIEWED"
        assert run()["status"] == "ACTIVATED"
    finally:
        REQUEST_PROVENANCE.reset(scope)
    standing["status"] = "NOT_REVIEWED"
    assert run()["status"] == "ACTIVATED"
    assert activated == [UUID(int=3), UUID(int=3)]


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
