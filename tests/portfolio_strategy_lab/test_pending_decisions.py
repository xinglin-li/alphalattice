"""What waits on a person, in one read: the rows the Home assembled itself (V45, UI-P D6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from alphalattice.control.product_host.composition.pending_decisions import pending_decisions
from alphalattice.control.product_host.composition.research_experiment_plan import ExperimentPlan
from alphalattice.control.product_host.storage.plan_previews import PreviewRegistry

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)
INPUTS = {"inputs": [{"input_id": "us-core", "versions": [{"end": "2026-08-01"}]}]}


def test_a_plan_outlives_the_host_that_answered_it_until_it_expires(tmp_path) -> None:
    """requirement (V525, S3): the plan store a restart reads again. A plan sealed under its
    hash is runnable from a new registry over the same folder until it expires, then readable
    but not runnable; a record that does not verify is never read; a single-plan owner's newer
    plan removes the older one."""

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


def test_a_plan_previewed_and_not_run_waits_until_it_expires() -> None:
    """requirement (V45): a PLAN previewed and not run is one decision, whoever previewed it,
    with its preview and its RUN; the registry holds it runnable only until it expires."""

    preview = {"plan_hash": "a" * 64, "study_kind": "factor.screening-development"}
    item = _kinds(previews=[{**preview, "previewed_by": "AGENT"}])["PLAN_PREVIEW"]
    assert item["previewed_by"] == "AGENT"
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
