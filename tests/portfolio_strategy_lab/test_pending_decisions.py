"""What waits on a person, in one read: the rows the Home assembled itself (V45, UI-P D6)."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from alphalattice.control.product_host.composition.pending_decisions import pending_decisions
from alphalattice.control.product_host.composition.research_experiment_plan import ExperimentPlan
from alphalattice.control.product_host.composition.task_recovery import task_attention
from alphalattice.control.product_host.storage.plan_previews import PreviewRegistry
from alphalattice.control.task_control.contracts import (
    ResearchPlan,
    TaskLifecycle,
    TaskRecord,
    TaskRecoveryLink,
)
from tests.workspace_task_runner.task_control_support import digest, task_contract

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)
INPUTS = {"inputs": [{"input_id": "us-core", "versions": [{"end": "2026-08-01"}]}]}


def test_a_plan_outlives_the_host_that_answered_it_until_it_expires(tmp_path) -> None:
    """A plan outlives the host that answered it until it expires."""

    from pydantic import BaseModel, model_validator

    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    class _Plan(BaseModel):  # type: ignore[misc]
        value: int
        plan_hash: str

        @model_validator(mode="after")  # type: ignore[untyped-decorator]
        def _sealed(self) -> _Plan:
            if self.plan_hash != canonical_hash({"value": self.value}):
                raise ValueError("plan_tampered")
            return self

    def plan(value: int) -> _Plan:
        return _Plan(value=value, plan_hash=canonical_hash({"value": value}))

    clock = [NOW]
    root = tmp_path / "plans"

    def store(single: bool = False) -> PreviewRegistry[_Plan]:
        return PreviewRegistry(model=_Plan, clock=lambda: clock[0], root=root, single=single)

    first, second = plan(1), plan(2)
    store().remember(first)
    store().remember(second)
    restarted = store()
    assert restarted.runnable(first.plan_hash) == first
    assert restarted.runnable(second.plan_hash) == second
    clock[0] = NOW + timedelta(minutes=61)
    assert store().runnable(first.plan_hash) is None
    assert store().get(first.plan_hash) is not None  # expired: readable, never runnable
    sealed = root / f"{first.plan_hash}.json"
    sealed.write_text(sealed.read_text(encoding="utf-8").replace('"value": 1', '"value": 3'))
    assert store().get(first.plan_hash) is None  # a record that does not verify
    clock[0] = NOW
    single = store(single=True)
    single.remember(first)
    single.remember(second)
    assert sorted(path.stem for path in root.glob("*.json")) == [second.plan_hash]
    assert store().runnable(first.plan_hash) is None


def _kinds(**owners: object) -> dict[str, dict[str, object]]:
    answer = pending_decisions(tasks=(), awaiting={}, data_issues={}, overview={}, **owners)
    return {str(item["kind"]): item for item in answer["decisions"]}  # type: ignore[attr-defined]


def test_the_workspace_state_waits_on_a_person_from_its_owners_answers() -> None:
    """requirement (V45): a workspace with no verified input and no preparation under way waits
    for its preparation; data past the newest input's end waits for a new input version; each
    names the request that takes it, and a preparation under way or refused is no decision."""

    unprepared = _kinds(preparation={"status": "NOT_PREPARED"}, inputs={"inputs": []})
    assert unprepared["WORKSPACE_PREPARATION"]["next_requests"] == {
        "plan": {"operation": "WORKSPACE_PREPARE_PLAN"}
    }
    for preparation in ({"status": "RUNNING", "task_id": "t"}, {"status": "REFUSED"}):
        assert not _kinds(preparation=preparation, inputs={"inputs": []})
    behind = _kinds(inputs=INPUTS, data_update={"inputs": {"market_through": "2026-08-03"}})
    assert behind["INPUT_VERSION"]["input_through"] == "2026-08-01"
    assert behind["INPUT_VERSION"]["next_requests"] == {
        "plan": {"operation": "RESEARCH_INPUT_PLAN", "research_input_id": "us-core"}
    }
    assert not _kinds(inputs=INPUTS, data_update={"inputs": {"market_through": "2026-08-01"}})
    assert not _kinds(inputs=INPUTS)  # a data update not read is no decision, never an empty one


def test_each_item_says_whom_it_waits_on_and_only_the_persons_are_counted() -> None:
    """requirement (STOPS-1): a step only a person may take waits on the PERSON; a step the agent
    takes and discloses, advice included, waits on the AGENT, and the headline counts only the
    person's. A preparation waits on the agent while a first-use goal delegates it."""

    owners = {
        "awaiting": {"curation": ["t1"], "promotion": ["t2"]},
        "previews": [{"plan_hash": "a" * 64, "study_kind": "k", "previewed_by": "AGENT"}],
        "inputs": INPUTS,
        "data_update": {"inputs": {"market_through": "2026-08-03"}},
    }
    answer = pending_decisions(tasks=(), data_issues={}, overview={}, **owners)  # type: ignore[arg-type]
    waits = {str(item["kind"]): item["waits_on"] for item in answer["decisions"]}  # type: ignore[attr-defined]
    assert waits == {
        "INPUT_VERSION": "PERSON",
        "PLAN_PREVIEW": "AGENT",
        "CURATION": "AGENT",
        "PROMOTION": "AGENT",
    }
    assert answer["detail"] == "1 decision waits on a person."
    unprepared = {"preparation": {"status": "NOT_PREPARED"}, "inputs": {"inputs": []}}
    assert _kinds(**unprepared)["WORKSPACE_PREPARATION"]["waits_on"] == "PERSON"
    first_use = {"goal_id": "g", "steps": [], "ends_at": NOW.isoformat(), "active": True}
    delegated = pending_decisions(
        tasks=(), awaiting={}, data_issues={}, overview={}, first_use=first_use, **unprepared
    )
    assert {item["kind"]: item["waits_on"] for item in delegated["decisions"]} == {  # type: ignore[attr-defined]
        "FIRST_USE": "AGENT",
        "WORKSPACE_PREPARATION": "AGENT",
    }
    assert delegated["detail"] == "Nothing waits on a person."


def test_a_plan_previewed_and_not_run_waits_until_it_expires() -> None:
    """requirement (V45): a PLAN previewed and not run is one decision, whoever previewed it,
    with its preview and its RUN; the registry holds it runnable only until it expires."""

    preview = {"plan_hash": "a" * 64, "study_kind": "factor.screening-development"}
    item = _kinds(previews=[{**preview, "previewed_by": "AGENT"}])["PLAN_PREVIEW"]
    assert item["previewed_by"] == "AGENT"
    assert item["goal_ids"] == [], "a historical preview supplies no inferred Goal"
    assert item["next_requests"] == {
        "preview": {"operation": "EXPERIMENT_PREVIEW_READBACK", "experiment_plan_hash": "a" * 64},
        "run": {"operation": "EXPERIMENT_RUN", "experiment_plan_hash": "a" * 64},
    }
    clock = [NOW]
    registry = PreviewRegistry(
        model=ExperimentPlan, clock=lambda: clock[0], ttl=timedelta(minutes=60)
    )
    for index, name in enumerate(("b", "c")):
        clock[0] = NOW + timedelta(minutes=index)
        plan = SimpleNamespace(plan_hash=name * 64, program=SimpleNamespace(kind="k"))
        registry.remember(plan, caller="HUMAN")  # type: ignore[arg-type]
    assert [entry.plan.plan_hash[0] for entry in registry.waiting()] == ["c", "b"]
    clock[0] = NOW + timedelta(minutes=60)
    assert [entry.plan.plan_hash[0] for entry in registry.waiting()] == ["c"]


def test_retained_goal_requests_reach_only_their_exact_pending_decisions(
    tmp_path, monkeypatch, live
) -> None:
    """Retained goal requests reach only their exact pending decisions."""

    from alphalattice.control.product_host.composition.goals import GoalApplication
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.product_host.publication.goals import GoalStore
    from alphalattice.interface.local_application.cli_contract import (
        RequestProvenance,
        envelope,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest as Request,
    )
    from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
        portfolio_review_task_contract,
    )

    goals = GoalApplication(
        GoalStore(tmp_path, "workspace"),
        lambda: NOW,
        lambda *_: {},
        lambda _: NOW,
        lambda _: None,
        workspace=tmp_path,
    )
    declaration = {
        "title": "Same question",
        "objective": "Inspect the exact work",
        "kind": "DATA",
        "criteria": [{"criterion_id": "read", "text": "Read the retained work"}],
    }

    def open_goal(kind="DATA"):
        opened = goals.operate(
            Request(
                operation="GOAL_OPEN",
                goal_id=uuid4(),
                goal_declaration={**declaration, "kind": kind},
            ),
            "HUMAN",
        )
        return goals.store.load(opened["goal_hash"])

    first, second = open_goal(), open_goal()
    provenance = RequestProvenance(goal_id=str(first.goal_id))

    def record(contract, lifecycle, failure_code=None):
        envelope, goal, plan = contract
        return TaskRecord.from_identity(
            task_id=uuid4(),
            task_kind=envelope.task_kind,
            input=envelope,
            goal=goal,
            plan=plan,
            lifecycle=lifecycle,
            active_work_item_id=None,
            latest_execution_id=None,
            admitted_at=NOW,
            started_at=None,
            updated_at=NOW,
            failure_code=failure_code,
            version=1,
        )

    stopped = record(task_contract(salt="follow-stop"), TaskLifecycle.BLOCKED, "fixture.blocked")
    study = record(task_contract(salt="follow-study"), TaskLifecycle.SUCCEEDED)
    review = record(
        portfolio_review_task_contract(review_key_payload={"decision_policy_hash": "d" * 64}),
        TaskLifecycle.SUCCEEDED,
    )
    for task in (stopped, study):
        goals.attribute(
            first,
            Request(operation="EXPERIMENT_RUN", experiment_plan_hash="a" * 64),
            {"status": "ADMITTED", "task_id": str(task.task_id)},
            provenance,
        )
    goals.attribute(
        first, Request(operation="EXPERIMENT_PLAN"), {"plan_hash": "a" * 64}, provenance
    )
    goals.attribute(
        first, Request(operation="CRO_REVIEW"), {"review_publication_hash": "b" * 64}, provenance
    )
    goals.attribute(second, Request(operation="CRO_REVIEW"), {"task_id": str(review.task_id)}, None)
    case_token, unbound_token = digest("follow-case"), digest("unattributed-case")
    offers = {
        f"preview:{case_token}:option": {
            "operation": "DATA_ISSUE_PREVIEW",
            "data_issue_case_token": case_token,
        },
        f"delegate:{stopped.task_id}:{case_token}:option": {
            "operation": "DATA_ISSUE_DELEGATE",
            "data_issue_case_token": case_token,
            "task_id": str(stopped.task_id),
        },
        f"preview:{unbound_token}:option": {
            "operation": "DATA_ISSUE_PREVIEW",
            "data_issue_case_token": unbound_token,
        },
    }
    canonical = (stopped, study, review)
    reads = []

    def collection():
        reads.append("tasks")
        return SimpleNamespace(records=canonical, refused_task_ids=())

    operations = live.operations
    with monkeypatch.context() as readers:
        readers.setattr(operations, "goals", goals)
        readers.setattr(
            operations,
            "workspace_session",
            SimpleNamespace(
                workspace=tmp_path,
                task_control_registry=SimpleNamespace(record_collection=collection),
                reads=lambda **_: nullcontext(True),
            ),
        )
        readers.setattr(
            operations.supervisor,
            "attention",
            lambda tasks: {t.task_id: task_attention(t) for t in tasks},
        )
        readers.setattr(
            operations,
            "experiments",
            SimpleNamespace(
                awaiting=lambda _: {
                    "curation": [str(study.task_id)],
                    "promotion": [str(study.task_id)],
                    "admitted": [],
                },
                waiting_previews=lambda _: [
                    {"plan_hash": "a" * 64, "study_kind": "factor.screening-development"}
                ],
            ),
        )
        readers.setattr(
            operations,
            "data_issues",
            SimpleNamespace(readback=lambda **_: {"next_requests": offers}),
        )
        readers.setattr(
            PortfolioResearchOperations,
            "execute",
            lambda _self, request, **_: (
                {"status": "NOT_PREPARED"}
                if request.operation == "WORKSPACE_PREPARE_READBACK"
                else {}
            ),
        )
        overview = {
            "reviews": [
                {
                    "state": "CHANGED",
                    "review_publication_hash": "e" * 64,
                    "review_key": review.input.input_hash,
                },
                {
                    "state": "CURRENT",
                    "review_publication_hash": "b" * 64,
                    "book_key": "book",
                    "person_action": {"action": "WEIGH"},
                },
                {
                    "state": "CHANGED",
                    "review_publication_hash": "f" * 64,
                    "review_key": study.input.input_hash,
                },
            ]
        }
        # The public overview reader supplies metadata here; its composition wrapper and
        # canonical Task-kind check remain real, including the negative non-CRO binding.
        readers.setattr(
            "alphalattice.control.product_host.composition.portfolio_research_operations.upgrade_overview",
            lambda **_: overview,
        )
        answer = operations.pending_decisions()
        assert reads == ["tasks"], "the pending operation reads canonical Tasks once"
        items = {
            item["kind"]: item
            for item in answer["decisions"]
            if item["kind"] != "DATA_ISSUE" and item.get("review_publication_hash") != "f" * 64
        }
        for kind in ("STOPPED_TASK", "CURATION", "PROMOTION", "PLAN_PREVIEW", "CRO_RECOMMENDATION"):
            assert items[kind]["goal_ids"] == [str(first.goal_id)]
            assert items[kind]["waits_on"] == "AGENT"
        assert items["REVIEW_CHANGED"]["goal_ids"] == [str(second.goal_id)]
        unrelated = next(
            item for item in answer["decisions"] if item.get("review_publication_hash") == "f" * 64
        )
        assert unrelated["goal_ids"] == [], "a non-CRO Task input never declares a review binding"
        assert items["WORKSPACE_PREPARATION"]["goal_ids"] == []
        shown = envelope(
            operation="PENDING_DECISIONS", outcome="OK", body=answer, elapsed_seconds=0
        )
        cases = {
            item["case_token"]: item
            for item in shown["data"]["decisions"]
            if item["kind"] == "DATA_ISSUE"
        }
        assert cases[case_token]["goal_ids"] == [str(first.goal_id)]
        assert cases[unbound_token]["goal_ids"] == []
        assert cases[case_token]["waits_on"] == "PERSON"
        assert sum(item["waits_on"] == "PERSON" for item in answer["decisions"]) == 3
        assert shown["detail"] == answer["detail"]
        assert all(case["waits_on"] == "PERSON" for case in cases.values())
        first_use = open_goal("FIRST_USE")
        delegated = operations.pending_decisions()
        item = next(item for item in delegated["decisions"] if item["kind"] == "FIRST_USE")
        assert item["goal_ids"] == [str(first_use.goal_id)]
        assert item["waits_on"] == "AGENT"
        assert not any(item["waits_on"] == "PERSON" for item in delegated["decisions"])
        shown = envelope(
            operation="PENDING_DECISIONS", outcome="OK", body=delegated, elapsed_seconds=0
        )
        assert shown["detail"] == delegated["detail"]
        delegated_cases = {
            item["case_token"]: item
            for item in shown["data"]["decisions"]
            if item["kind"] == "DATA_ISSUE"
        }
        assert delegated_cases.keys() == cases.keys()
        for token, item in delegated_cases.items():
            assert item["waits_on"] == "AGENT"
            assert item["detail"] == cases[token]["detail"]
            assert item["next_requests"] == cases[token]["next_requests"]


@pytest.mark.parametrize("stopped", [TaskLifecycle.BLOCKED, TaskLifecycle.RECOVERY_REQUIRED])
@pytest.mark.parametrize(
    "successor_state",
    [
        TaskLifecycle.QUEUED,
        TaskLifecycle.BLOCKED,
        TaskLifecycle.RECOVERY_REQUIRED,
        TaskLifecycle.SUCCEEDED,
    ],
)
def test_current_attention_requires_a_successful_explicit_successor_of_this_version(
    stopped: TaskLifecycle, successor_state: TaskLifecycle
) -> None:
    """Current attention requires a successful explicit successor of this version."""
    envelope, goal, plan = task_contract(salt="attention")

    def record(task_id, lifecycle, version=1) -> TaskRecord:
        return TaskRecord.from_identity(
            task_id=task_id,
            task_kind=envelope.task_kind,
            input=envelope,
            goal=goal,
            plan=plan,
            lifecycle=lifecycle,
            active_work_item_id=None,
            latest_execution_id=None,
            admitted_at=NOW,
            started_at=None,
            updated_at=NOW,
            failure_code=None,
            version=version,
        )

    source = record(uuid4(), stopped)
    successor = record(uuid4(), successor_state)
    successful_peer = record(uuid4(), TaskLifecycle.SUCCEEDED)
    request = {"operation": "EXPERIMENT_RUN", "experiment_plan_hash": "a" * 64}
    preview = TaskRecoveryLink.create(
        source_task_id=source.task_id,
        source_record_hash=source.record_hash,
        admission_request=request,
        successor_task_id=None,
        recorded_at=NOW,
    )
    confirmed = TaskRecoveryLink.create(
        source_task_id=source.task_id,
        source_record_hash=source.record_hash,
        admission_request=request,
        successor_task_id=successor.task_id,
        recorded_at=NOW,
    )
    peers = {successor.task_id: successor, successful_peer.task_id: successful_peer}
    assert task_attention(source, successors=peers).unresolved
    assert task_attention(source, successors=peers, links=(preview,)).unresolved
    fact = task_attention(source, successors=peers, links=(preview, confirmed))
    assert fact.unresolved is (successor_state is not TaskLifecycle.SUCCEEDED)
    assert fact.resolution == (
        "SUCCESSOR_SUCCEEDED" if successor_state is TaskLifecycle.SUCCEEDED else "STOPPED"
    )

    def refetch():
        return pending_decisions(
            tasks=(source,),
            attention={source.task_id: fact},
            awaiting={},
            data_issues={},
            overview={},
        )

    first = refetch()
    assert refetch() == first  # viewing changes neither resolution nor the retained history
    assert bool(first["decisions"]) is fact.unresolved
    assert source.lifecycle is stopped
    newer_stop = record(source.task_id, stopped, version=2)
    assert task_attention(newer_stop, successors=peers, links=(confirmed,)).unresolved
    assert task_attention(source, links=(confirmed,)).unresolved  # unread successor is no success


@pytest.mark.parametrize(
    "boundary",
    [
        "later_exact_plan_success",
        "changed_plan",
        "earlier_admission",
        "same_admission",
        "canonical_earlier_with_later_clock",
        "canonical_later_with_same_clock",
        "blocked_successor",
        "recovery_required_source",
    ],
)
def test_only_a_later_successful_exact_plan_supersedes_a_blocked_task(boundary: str) -> None:
    """BADGE regression: a later successful canonical admission of the same sealed plan
    clears a block; changed plans and other admission or lifecycle boundaries keep it."""
    envelope, goal, plan = task_contract(salt="exact-plan-attention")

    def record(
        lifecycle: TaskLifecycle, admitted_at: datetime, task_plan: ResearchPlan
    ) -> TaskRecord:
        return TaskRecord.from_identity(
            task_id=uuid4(),
            task_kind=envelope.task_kind,
            input=envelope,
            goal=goal,
            plan=task_plan,
            lifecycle=lifecycle,
            active_work_item_id=None,
            latest_execution_id=None,
            admitted_at=admitted_at,
            started_at=None,
            updated_at=NOW + timedelta(minutes=10),
            failure_code="fixture.stopped" if lifecycle is not TaskLifecycle.SUCCEEDED else None,
            version=1,
        )

    source_state = (
        TaskLifecycle.RECOVERY_REQUIRED
        if boundary == "recovery_required_source"
        else TaskLifecycle.BLOCKED
    )
    source = record(source_state, NOW, plan)
    successor_plan = plan
    if boundary == "changed_plan":
        successor_plan = ResearchPlan.create(
            goal_hash=goal.goal_hash,
            workflow_definition_hash=digest("changed-attention-workflow"),
            verifier_catalog_hash=plan.verifier_catalog_hash,
            work_items=plan.work_items,
        )
        assert successor_plan.plan_hash != plan.plan_hash
    successor_state = (
        TaskLifecycle.BLOCKED if boundary == "blocked_successor" else TaskLifecycle.SUCCEEDED
    )
    admission_minutes = {
        "earlier_admission": -1,
        "same_admission": 0,
        "canonical_later_with_same_clock": 0,
    }.get(boundary, 1)
    successor = record(successor_state, NOW + timedelta(minutes=admission_minutes), successor_plan)
    retained_source = source.model_dump(mode="json")
    canonical_order = {
        "canonical_earlier_with_later_clock": False,
        "canonical_later_with_same_clock": True,
    }.get(boundary)
    fact = task_attention(source, superseded_by=successor, superseded_is_later=canonical_order)
    clears = boundary in {"later_exact_plan_success", "canonical_later_with_same_clock"}
    assert fact.unresolved is (not clears)
    assert fact.resolution == ("SUCCESSOR_SUCCEEDED" if clears else "STOPPED")
    assert fact.task_record_hash == source.record_hash
    if clears:
        assert fact.successor_task_id == str(successor.task_id)
        assert fact.successor_task_hash == successor.record_hash
        assert fact.successor_lifecycle == TaskLifecycle.SUCCEEDED.value
    else:
        assert fact.successor_task_id is None
        assert fact.successor_task_hash is None
        assert fact.successor_lifecycle is None
    answer = pending_decisions(
        tasks=(source,),
        attention={source.task_id: fact},
        awaiting={},
        data_issues={},
        overview={},
    )
    assert answer["counts"] == ({} if clears else {"STOPPED_TASK": 1})
    assert bool(answer["decisions"]) is (not clears)
    assert source.model_dump(mode="json") == retained_source


def test_a_rebuilt_ledger_task_holds_no_decision_and_keeps_its_stopped_record() -> None:
    """BADGE regression: a ledger-rebuilt block leaves attention while its stopped state
    and reason stay readable, and an ordinary blocked peer still needs its decision."""
    envelope, goal, plan = task_contract(salt="rebuilt-attention")

    def record(failure_code: str) -> TaskRecord:
        return TaskRecord.from_identity(
            task_id=uuid4(),
            task_kind=envelope.task_kind,
            input=envelope,
            goal=goal,
            plan=plan,
            lifecycle=TaskLifecycle.BLOCKED,
            active_work_item_id=None,
            latest_execution_id=None,
            admitted_at=NOW,
            started_at=None,
            updated_at=NOW,
            failure_code=failure_code,
            version=1,
        )

    rebuilt = record("task_control.ledger_rebuilt")
    peer = record("fixture.blocked")
    retained = rebuilt.model_dump(mode="json")
    fact = task_attention(rebuilt)
    assert fact.unresolved is False
    assert fact.resolution == "UNRECOVERABLE"
    assert fact.task_record_hash == rebuilt.record_hash
    answer = pending_decisions(
        tasks=(rebuilt, peer),
        awaiting={},
        data_issues={},
        overview={},
    )
    assert answer["counts"] == {"STOPPED_TASK": 1}
    decisions = answer["decisions"]
    assert isinstance(decisions, list)
    assert [row["task_id"] for row in decisions] == [str(peer.task_id)]
    assert rebuilt.lifecycle is TaskLifecycle.BLOCKED
    assert rebuilt.failure_code == "task_control.ledger_rebuilt"
    assert rebuilt.model_dump(mode="json") == retained


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "plan_hash",
        "data_revision_hash",
        "panel_hash",
        "data_through",
        "task_lifecycle",
        "task_id",
        "missing_plan",
    ],
)
def test_input_version_follows_only_the_task_that_published_current_data(changed) -> None:
    """Input version follows only the task that published current data."""

    task, goal = str(uuid4()), str(uuid4())
    current = {"data_through": "2026-08-03", "data_revision_hash": "d" * 64, "panel_hash": "f" * 64}
    update = {
        "inputs": current,
        "task_id": task,
        "task_lifecycle": "SUCCEEDED",
        "plan_hash": "a" * 64,
        "receipt": {"plan_hash": "a" * 64, "after": deepcopy(current)},
    }
    if changed == "missing_plan":
        update["plan_hash"] = update["receipt"]["plan_hash"] = None
    elif changed in {"data_revision_hash", "panel_hash", "data_through"}:
        update["receipt"]["after"][changed] = "earlier"
    elif changed:
        update[changed] = "RUNNING" if changed == "task_lifecycle" else None
    answer = pending_decisions(
        tasks=(),
        awaiting={},
        data_issues={},
        overview={},
        inputs=INPUTS,
        data_update=update,
        goal_attribution={("task_id", task): {goal}},
    )
    decision = next(row for row in answer["decisions"] if row["kind"] == "INPUT_VERSION")
    assert decision["goal_ids"] == ([goal] if changed is None else [])
    assert decision.get("task_id") == (task if changed is None else None)
    assert decision["waits_on"] == "PERSON"
