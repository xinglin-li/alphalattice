"""A Task's numerical work in the Host's worker process (W10)."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from alphalattice.capabilities.domain_contracts import spawn
from alphalattice.capabilities.domain_contracts.bounded_memo import stage_digest
from alphalattice.capabilities.domain_contracts.source_identity import (
    open_source_role_acquisition,
    source_role_acquisition,
)
from alphalattice.control.task_control import child
from alphalattice.control.task_control.child import (
    ChildInterrupted,
    ChildRefused,
    ChildStartFailed,
    child_calls,
    run_in_child,
)
from alphalattice.foundation.data_layer import create_data_root, reading
from alphalattice.foundation.data_layer.log import HostLog
from alphalattice.kernel.execution import (
    CANCELLED,
    cancel_execution,
    execution_cancellation,
    interruptible,
    stop_reason,
)
from alphalattice.kernel.resources import (
    NATIVE_THREAD_VARIABLES,
    ResourceAmounts,
    ResourceLease,
    use_resource_lease,
)

HERE = __spec__.name if __spec__ else __name__


class _AlphaPartExecutor:
    kind = "alpha.model-development"

    def __init__(self, plan, arrays, program, batch, *, stop_after_fold=None):
        self.plan, self.arrays, self.program, self.batch = plan, arrays, program, batch
        self.stop_after_fold = stop_after_fold

    def set_cancellation_check(self, cancelled):
        self.cancelled = cancelled

    def execute(self, *, output_workspace, **_arguments):
        from alphalattice.investment.alpha_research.experiments.development_artifacts import (
            AlphaDevelopmentArtifactStore,
        )
        from alphalattice.investment.alpha_research.experiments.development_execution import (
            AlphaDevelopmentCancelled,
        )
        from alphalattice.investment.alpha_research.experiments.execution import (
            execute_alpha_model_batch,
        )
        from alphalattice.investment.alpha_research.experiments.mandate import (
            build_current_alpha_research_model_mandate,
        )

        def stop_at_fold_boundary(**_facts):
            if self.stop_after_fold == "recover_fold":
                os._exit(73)
            if self.stop_after_fold == "cancel_fold" or self.cancelled():
                raise AlphaDevelopmentCancelled("alpha_research.cancelled_at_fold_boundary")

        store = AlphaDevelopmentArtifactStore(output_workspace / "alpha-development")
        with store.admit_development_chunks():
            result = execute_alpha_model_batch(
                program=self.program,
                batch=self.batch,
                fold_plan=self.plan,
                store=store,
                array_workspace=self.arrays,
                model_mandate=build_current_alpha_research_model_mandate(),
                operational_progress=SimpleNamespace(record_candidate_fold=stop_at_fold_boundary),
            )
        return {"pid": os.getpid(), "batch": result}


class _AlphaPartRecipe:
    def __init__(self, executor):
        self.executor = executor

    def prepare(self):
        return SimpleNamespace(executor=self.executor)


def _check_alpha_child_parts(tmp_path: Path, *, interruption: str | None) -> None:
    """Fit the installed ridge model in a separate worker, then reopen its scores in the Host."""
    from alphalattice.control.product_host.research_authoring.factor_handoff import (
        ChildAlphaExecutor,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.control.task_control.registry import (
        DuckDbTaskControlRegistry,
        resolve_task_control_database,
    )
    from alphalattice.control.task_control.runner import TaskControlRunner
    from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
    from alphalattice.foundation.data_layer import history
    from alphalattice.investment.alpha_research.candidates.contracts import (
        AlphaExperimentBatch,
        seal_contract,
    )
    from alphalattice.investment.alpha_research.experiments.development_artifacts import (
        AlphaDevelopmentArtifactStore,
    )
    from alphalattice.investment.alpha_research.experiments.development_execution import (
        AlphaDevelopmentCancelled,
    )
    from tests.alpha_research.fixtures import _plan_and_workspace
    from tests.integration.alpha_research.test_model_adapter_boundary import _batch, _program
    from tests.workspace_task_runner.task_control_support import FixtureTaskAdapter, task_contract

    plan, arrays = _plan_and_workspace()
    program = _program(plan.foundation, plan)
    all_candidates = _batch(program)
    batch = seal_contract(
        AlphaExperimentBatch,
        all_candidates.model_dump(exclude={"batch_hash"}) | {"specs": all_candidates.specs[:1]},
        "batch_hash",
    )
    executor = _AlphaPartExecutor(plan, arrays, program, batch)
    history.open_history("child-parts", "child parts", tmp_path)
    try:
        store = AlphaDevelopmentArtifactStore(tmp_path / "alpha-development")
        if interruption is not None:
            stopped = _AlphaPartExecutor(plan, arrays, program, batch, stop_after_fold=interruption)
            child_executor = ChildAlphaExecutor(recipe=_AlphaPartRecipe(stopped), kind=stopped.kind)

            def stopped_call():
                return child_executor.execute(
                    program=program, document={}, authority=None, output_workspace=tmp_path
                )

            if interruption == "recover_fold":
                # A second worker also ends before finishing. The Host has already recovered
                # the first fold, so its pointer must be readable before any successful reply.
                for _ in range(2):
                    with pytest.raises(
                        ChildInterrupted, match=r"task_control\.child_process_ended"
                    ):
                        stopped_call()
            else:
                with pytest.raises(AlphaDevelopmentCancelled):
                    stopped_call()
            pointers = list(
                (store.root / "current/candidate-numerical-fold-by-binding").glob("*.json")
            )
            assert len(pointers) == 1
            try:
                recovered = store.find_candidate_numerical_fold_result(pointers[0].stem)
            except FileNotFoundError as error:
                pytest.fail(f"the visible fold has no catalog score: {error}", pytrace=False)
            assert recovered is not None and recovered.score_chunk is not None
            assert history.read_alpha_development_chunks((recovered.score_chunk.content_hash,))
        result = None

        class FitStage(FixtureTaskAdapter):
            def execute_stage(self, *, task, execution, work_item):
                nonlocal result
                if work_item.stage_id == "publish_result":
                    result = ChildAlphaExecutor(
                        recipe=_AlphaPartRecipe(executor), kind=executor.kind
                    ).execute(
                        program=program, document={}, authority=None, output_workspace=tmp_path
                    )
                return super().execute_stage(task=task, execution=execution, work_item=work_item)

        registry = DuckDbTaskControlRegistry(
            resolve_task_control_database(tmp_path),
            gate=WorkspaceMutationGate(),
            workspace_id="child-parts",
        )
        envelope, goal, task_plan = task_contract(salt="alpha-child")
        now = datetime.now(UTC)
        admitted = registry.admit(
            input_envelope=envelope, goal=goal, plan=task_plan, observed_at=now
        ).record
        adapter = FitStage()
        runner = TaskControlRunner(
            registry=registry,
            adapters={adapter.task_kind: adapter},
            runtime_path=str(tmp_path / "task-runtime.sqlite"),
            clock=lambda: now,
        )
        stopped = None
        try:
            runner.run_next()
        except (ChildInterrupted, ChildRefused) as error:
            stopped = str(error)
        finally:
            runner.close()
        terminal = registry.task(admitted.task_id)
        assert terminal.lifecycle is TaskLifecycle.SUCCEEDED, (terminal.lifecycle, stopped)
        assert result is not None and result["pid"] != os.getpid()
        fitted = result["batch"]
        assert fitted.fit_call_count == plan.fold_count - int(interruption is not None)
        for result_hash in fitted.candidates[0].numerical_result_hashes:
            numerical, scores = store.read_candidate_numerical_fold_result(result_hash)
            assert scores is not None and scores.num_rows > 0
            chunk = numerical.score_chunk
            assert chunk is not None
            catalog = history.read_alpha_development_chunks((chunk.content_hash,))[
                chunk.content_hash
            ]
            assert catalog.replace_schema_metadata(None).equals(scores)
            assert (
                store.find_candidate_numerical_fold_result(numerical.execution_binding_hash)
                == numerical
            )
    finally:
        history.close_history()


@pytest.fixture(autouse=True)
def _no_kept_workers():
    """No test inherits a kept worker another left, whose idle wait its own never reaches."""

    def retire() -> None:
        for worker, pool in list(child._WORKERS.items()):
            child._retire_worker(worker, pool)

    retire()
    yield
    retire()
    # A retired worker leaves no pending idle retirement holding its executor's descriptors.
    assert not child._IDLE, child._IDLE


@pytest.mark.parametrize("spread", [False, True])
@pytest.mark.parametrize("at", ["constructor", "start"])
@pytest.mark.parametrize("error_type", [OSError, MemoryError], ids=lambda error: error.__name__)
def test_child_creation_refuses_by_name_and_the_next_call_replaces_it(
    monkeypatch, spread, at, error_type
):
    """both child APIs normalize constructor/start resource refusals and can recover."""
    # Retire even a warm worker through the public boundary before injecting at its next start.
    with pytest.raises(ChildInterrupted):
        run_in_child(f"{HERE}:ended", {})

    def fail(*args, **kwargs):
        raise error_type("fixture child start refused")

    with monkeypatch.context() as patch:
        if at == "constructor":
            patch.setattr(ProcessPoolExecutor, "__init__", fail)
        else:
            patch.setattr(multiprocessing.get_context("spawn").Process, "start", fail)
        with pytest.raises(
            ChildStartFailed, match=r"^task_control\.child_start_failed$"
        ) as stopped:
            if spread:
                with child_calls(f"{HERE}:doubled", workers=1) as calls:
                    calls.make({"value": 21})
            else:
                run_in_child(f"{HERE}:doubled", {"value": 21})
        # The refusal and its cause hold tracebacks back to this frame; held here, either keeps its
        # call's pool alive until a collection, so only plain values stay.
        cause = stopped.value.__cause__
        refusal = (type(cause), cause.args, stopped.value.type_name)
        del stopped, cause
        assert refusal == (error_type, ("fixture child start refused",), error_type.__name__)
    assert run_in_child(f"{HERE}:doubled", {"value": 21})["value"] == 42


class _Declared(RuntimeError):
    failure_class = "OPERATIONAL_FAILURE"


def doubled(*, value: int, cancelled) -> dict[str, object]:  # type: ignore[no-untyped-def]
    from alphalattice.foundation.data_layer.root import data_root

    return {"value": value * 2, "pid": os.getpid(), "root": str(data_root())}


_DIGESTED: list[str] = []
_ACQUIRED: list[object] = []


def digested(*, key: str, cancelled: Callable[[], bool]) -> tuple[int, bool]:
    """Digest `key` in the worker's phase; answer how many times the worker hashed it, and
    whether the call observes its sources in an acquisition no earlier call held."""
    stage_digest(("phase-probe", key), key, lambda value: _DIGESTED.append(value) or value)
    acquired = open_source_role_acquisition()
    fresh = acquired is not None and all(acquired is not held for held in _ACQUIRED)
    _ACQUIRED.append(acquired)
    return _DIGESTED.count(key), fresh


def thread_variables(*, cancelled) -> dict[str, str | None]:  # type: ignore[no-untyped-def]
    return {name: os.environ.get(name) for name in NATIVE_THREAD_VARIABLES}


def numerical_product(*, cancelled) -> dict[str, object]:  # type: ignore[no-untyped-def]
    import numpy as np

    from alphalattice.capabilities.domain_contracts.environment import numerical_thread_counts
    from alphalattice.kernel.resources import current_resource_lease

    grant = current_resource_lease()
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    before = numerical_thread_counts()
    product = values @ values.T
    return {
        "counts": before,
        "after": numerical_thread_counts(),
        "product": product.tolist(),
        "granted_cpu": grant.amounts.cpu if grant is not None else None,
    }


def limited_once(*, cancelled) -> None:  # type: ignore[no-untyped-def]
    from alphalattice.capabilities.domain_contracts.environment import numerical_thread_limit

    with numerical_thread_limit(1):  # loads the libraries a limit holds, as a fit's first does
        pass


def refused(*, cancelled) -> None:  # type: ignore[no-untyped-def]
    raise ValueError("fixture.refused_by_name")


def interrupted(*, cancelled) -> None:  # type: ignore[no-untyped-def]
    raise _Declared("fixture.interrupted")


def until_cancelled(*, seconds: float, cancelled) -> str:  # type: ignore[no-untyped-def]
    ends = time.monotonic() + seconds
    while time.monotonic() < ends:
        if cancelled():
            return "cancelled"
        time.sleep(0.05)
    return "ran out"


def until_interrupted(*, seconds: float, cancelled: Callable[[], bool]) -> str:
    """Block in one interruptible operation until its interrupt, as a process the call runs
    does: a stop accepted before it registered is read at its start."""
    interrupted = threading.Event()
    with interruptible(interrupted.set):
        stopped = stop_reason() is not None or interrupted.wait(seconds)
    return "interrupted" if stopped else "ran out"


def interrupted_at(*, seconds: float, holds: bool, cancelled: Callable[[], bool]) -> object:
    """With ``holds``, an operation that registers no interrupt and answers "held" at its end;
    else `until_interrupted`, answering the wall time its interrupt came, or None."""
    if holds:
        threading.Event().wait(seconds)
        return "held"
    interrupted = threading.Event()
    with interruptible(interrupted.set):
        stopped = stop_reason() is not None or interrupted.wait(seconds)
    return time.time() if stopped else None


def ended(*, cancelled) -> None:  # type: ignore[no-untyped-def]
    os._exit(7)


def sized(*, value: int, size: int, cancelled) -> dict[str, object]:  # type: ignore[no-untyped-def]
    if value < 0:
        raise ValueError("fixture.refused_by_name")
    return {"value": value, "pid": os.getpid(), "payload": b"x" * size}


def test_calls_spread_over_kept_workers_answer_in_order_and_bring_back_large_answers() -> None:
    """requirement (PF #1, W10): independent calls run on up to N kept workers and answer in
    the order made; an answer of megabytes comes back whole, through the calls' folder, where a
    worker's pipe refused one under load (WinError 1450); a refusal answers in its turn."""

    with child_calls(f"{HERE}:sized", workers=3) as calls:
        for value in range(6):
            calls.make({"value": value, "size": 12_000_000 if value == 4 else 8})
        calls.make({"value": -1, "size": 8})
        answers = [calls.answer() for _ in range(6)]
        with pytest.raises(ChildRefused, match=r"^fixture.refused_by_name$"):
            calls.answer()
    assert [answer["value"] for answer in answers] == list(range(6))
    assert len({answer["pid"] for answer in answers}) == 3
    assert answers[4]["payload"] == b"x" * 12_000_000


@pytest.mark.proves("task_control.alpha_child_returns_parts_to_host")
@pytest.mark.parametrize("case", ["plain", "complete", "recover_fold", "cancel_fold"])
def test_a_call_runs_in_one_kept_worker_and_answers_its_failures_by_kind(
    tmp_path: Path, case: str, monkeypatch
) -> None:
    """The child answers by kind and carries real fitted parts through the Host's held log; its
    warm second call reads the data root its caller names, not the one it was started under."""

    if case != "plain":
        _check_alpha_child_parts(tmp_path, interruption=None if case == "complete" else case)
        return

    from alphalattice.foundation.data_layer.root import DATA_ROOT_ENV

    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "first"))
    first = run_in_child(f"{HERE}:doubled", {"value": 21})
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "second"))
    second = run_in_child(f"{HERE}:doubled", {"value": 1})
    assert (first["value"], second["value"]) == (42, 2)
    assert first["pid"] == second["pid"] != os.getpid()
    # The calls of one caller phase share its digests and each observes its sources afresh; a
    # call outside a phase holds its own digests.
    key = uuid4().hex
    with source_role_acquisition():
        calls = [run_in_child(f"{HERE}:digested", {"key": key}) for _ in "ab"]
    assert calls == [(1, True), (1, True)]
    assert run_in_child(f"{HERE}:digested", {"key": key}) == (2, True)
    assert (first["root"], second["root"]) == (str(tmp_path / "first"), str(tmp_path / "second"))
    with pytest.raises(ChildRefused, match=r"^fixture.refused_by_name$") as refusal:
        run_in_child(f"{HERE}:refused", {})
    assert refusal.value.type_name == "ValueError"
    with pytest.raises(ChildInterrupted, match=r"^fixture.interrupted$") as failure:
        run_in_child(f"{HERE}:interrupted", {})
    assert failure.value.type_name == "_Declared"
    assert failure.value.failure_class == "OPERATIONAL_FAILURE"


@pytest.mark.proves("task_control.a_stop_reaches_a_kept_workers_operation_when_accepted")
@pytest.mark.parametrize("call", ["run", "calls", "spread"])
def test_a_tasks_stop_interrupts_its_workers_operation_when_accepted(monkeypatch, call) -> None:
    """requirement: a Task's accepted stop reaches each kept worker's running operation of the
    call set through its flag and the worker's bell, with no period on either side, while the
    parent waits on another call (its answer wait set longer than the call)."""
    assert run_in_child(f"{HERE}:doubled", {"value": 1})["value"] == 2  # the worker is up
    monkeypatch.setattr(child, "POLL_SECONDS", 60.0)
    execution = UUID(int=41)
    stop = threading.Timer(0.5, cancel_execution, (execution,))
    with execution_cancellation(execution):
        if call == "spread":
            with child_calls(f"{HERE}:interrupted_at", workers=2) as warm:  # both workers are up
                warm.make({"seconds": 0.0, "holds": True})
                warm.make({"seconds": 0.0, "holds": True})
                assert [warm.answer(), warm.answer()] == ["held", "held"]
        stop.start()
        began, wall = time.monotonic(), time.time()
        if call == "run":
            answer = run_in_child(f"{HERE}:until_interrupted", {"seconds": 20.0})
        elif call == "calls":
            with child_calls(f"{HERE}:until_interrupted", workers=1) as calls:
                calls.make({"seconds": 20.0})
                answer = calls.answer()
        else:
            with child_calls(f"{HERE}:interrupted_at", workers=2) as calls:
                calls.make({"seconds": 6.0, "holds": True})
                calls.make({"seconds": 20.0, "holds": False})
                assert calls.answer() == "held"
                at = calls.answer()
            answer = "interrupted" if at is not None and at - wall < 4.0 else f"at {at}"
        stop.join(5)
    assert answer == "interrupted" and time.monotonic() - began < 10


def test_a_cancellation_reaches_the_worker_and_a_worker_that_ends_is_replaced() -> None:
    """requirement (W10): the parent relays the Task's cancellation while the call runs; a
    worker that ends without answering is an interruption, and the next call starts a new one."""

    asked = []

    def cancelled() -> bool:
        asked.append(True)
        return len(asked) >= 2

    answer = run_in_child(f"{HERE}:until_cancelled", {"seconds": 30.0}, cancelled=cancelled)
    assert answer == "cancelled"
    from alphalattice.control.task_control import child

    processes = tuple(child._WORKERS[0]._processes.values())
    assert len(processes) == 1
    with pytest.raises(ChildInterrupted, match="child_process_ended"):
        run_in_child(f"{HERE}:ended", {})
    assert processes[0].exitcode == 7
    assert run_in_child(f"{HERE}:doubled", {"value": 5})["value"] == 10


def test_stage_plans_keep_both_consumers_canonical_work_items() -> None:
    """The shared stage constructor preserves both consumers' complete plans and work items."""
    from alphalattice.capabilities.domain_contracts.identity import canonical_hash
    from alphalattice.control.product_host.composition import (
        research_experiments,
        strategy_calibration,
    )
    from alphalattice.control.task_control.contracts import ResearchPlan, WorkItemDefinition

    for owner, namespace in (
        (research_experiments, "research_experiment"),
        (strategy_calibration, "portfolio_calibration"),
    ):
        stages = owner.STAGES
        expected = ResearchPlan.create(
            goal_hash="0" * 64,
            workflow_definition_hash=canonical_hash(stages),
            verifier_catalog_hash=canonical_hash(stages),
            work_items=tuple(
                WorkItemDefinition.create(
                    stage_id=stage, dependency_ids=stages[:i], verifier_id=f"{namespace}.{stage}"
                )
                for i, stage in enumerate(stages)
            ),
        )
        assert (
            ResearchPlan.from_stages(
                goal_hash="0" * 64, stages=stages, verifier_namespace=namespace
            )
            == expected
        )


def test_a_cached_read_sees_a_write_another_registry_made(tmp_path: Path) -> None:
    """regression (W10): Task reads are answered again while the database files show no write;
    a write through another registry on the same file is seen by the next read."""

    from alphalattice.control.task_control.registry import (
        TASK_CONTROL_DATABASE_FILENAME,
        DuckDbTaskControlRegistry,
    )
    from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
    from tests.workspace_task_runner.task_control_support import task_contract

    now = datetime(2026, 9, 29, tzinfo=UTC)
    database = tmp_path / "runtime" / TASK_CONTROL_DATABASE_FILENAME
    first = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    envelope, goal, plan = task_contract()
    admitted = first.admit(input_envelope=envelope, goal=goal, plan=plan, observed_at=now).record
    assert first.tasks() == (admitted,)
    assert first.task(admitted.task_id) is first.task(admitted.task_id)  # answered again
    second = DuckDbTaskControlRegistry(database, gate=WorkspaceMutationGate())
    second.request_cancel(
        task_id=admitted.task_id, expected_task_hash=admitted.record_hash, observed_at=now
    )
    assert first.task(admitted.task_id).lifecycle != admitted.lifecycle
    record, items = first.task_with_work_items(admitted.task_id)
    assert record == first.task(admitted.task_id) and len(items) == len(plan.work_items)


def test_a_fit_log_records_the_numerical_thread_counts(tmp_path: Path) -> None:
    """requirement (V68, LAWS PA3): the BLAS thread count can move an Alpha metric's last bits,
    so the fit log keeps the numerical libraries' counts beside the run; a record written before
    it was kept still reads, with none."""

    import numpy  # noqa: F401 -- the Host's numerical libraries are loaded when it fits

    from alphalattice.capabilities.domain_contracts.environment import (
        numerical_thread_counts,
        numerical_thread_limit,
    )
    from alphalattice.evidence.alternative_evidence.runtime.execution import ModelFitExecution

    # Exercise a fresh process: no prior cached pool view may hide the Windows race.
    probe = """import importlib
import numpy
import threadpoolctl
from alphalattice.capabilities.domain_contracts import environment
# Fit loads its numerical libraries before reading their thread pools.
with environment.numerical_thread_limit(1):
    pass
original = threadpoolctl.ThreadpoolController
for kind, expected in (("transient", 2), ("persistent", 16), ("other", 1)):
    importlib.reload(environment)
    environment._loaded_module_handles = lambda: (101,)
    attempts = []
    def discover():
        attempts.append(1)
        if kind == "other": raise OSError("other failure")
        if kind == "persistent" or len(attempts) == 1:
            raise OSError("GetModuleFileNameEx failed")
        return original()
    threadpoolctl.ThreadpoolController = discover
    try:
        pools = environment.numerical_thread_pools()
        assert kind == "transient" and pools
    except OSError as error:
        assert kind != "transient", error
    assert len(attempts) == expected, (kind, attempts)
threadpoolctl.ThreadpoolController = original

# A changing module set must be rebuilt for this call, never applied or recorded.
importlib.reload(environment)
environment._load_limited_libraries = lambda *, preload_models: None
built, limited = [], []
class Controller:
    def __init__(self):
        self.number = len(built) + 1
        built.append(self)
    def info(self): return [{"internal_api": "fixture", "num_threads": self.number}]
    def limit(self, limits):
        limited.append((self.number, limits))
        return self
    def restore_original_limits(self): pass
threadpoolctl.ThreadpoolController = Controller
snapshots = iter([(101,), (101, 202), (101, 202), (101, 202), (101, 202)])
environment._loaded_module_handles = lambda: next(snapshots)
with environment.numerical_thread_limit(1):
    assert environment.numerical_thread_counts() == (("fixture", 2),)
assert limited == [(2, 1)] and len(built) == 2

# Sustained churn refuses both callers before applying or recording a partial view.
for caller in ("limit", "counts"):
    importlib.reload(environment)
    environment._load_limited_libraries = lambda *, preload_models: None
    built.clear(); limited.clear()
    snapshots = iter([(101,), (101, 202)] * 16)
    environment._loaded_module_handles = lambda: next(snapshots)
    try:
        if caller == "counts": environment.numerical_thread_counts()
        else:
            with environment.numerical_thread_limit(1):
                raise AssertionError("unproved controller was applied")
    except RuntimeError as error:
        assert "module" in str(error).lower()
    else: raise AssertionError("unstable view was accepted")
    assert len(built) == 16 and not limited and not environment._LIMITS
if environment.os.name == "nt":
    importlib.reload(environment)
    built.clear()
    environment._loaded_module_handles = lambda: None
    try: environment.numerical_thread_counts()
    except RuntimeError as error: assert "unavailable" in str(error)
    else: raise AssertionError("unavailable snapshot was accepted")
    assert not built
threadpoolctl.ThreadpoolController = original
"""
    result = spawn.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with numerical_thread_limit(4):
        counts = numerical_thread_counts()
    assert counts and all(isinstance(api, str) and threads >= 1 for api, threads in counts)
    record = {
        "program_hash": "a" * 64,
        "cpu_budget": "auto",
        "cores": 4,
        "lightgbm_threads": 4,
        "reason": "fixture",
        "machine": {
            "processors": 8,
            "busy_processors": 1.0,
            "total_memory_bytes": 1,
            "available_memory_bytes": 1,
            "preparations_running": 0,
        },
        "planned_at": "2026-09-29T00:00:00+00:00",
    }
    assert ModelFitExecution.model_validate(record).numerical_threads == ()
    kept = ModelFitExecution.model_validate({**record, "numerical_threads": counts})
    assert kept.numerical_threads == counts


def test_a_spread_worker_loads_its_numerical_libraries_on_one_thread(monkeypatch) -> None:
    """A spread worker loads its numerical libraries on one thread."""

    for name in NATIVE_THREAD_VARIABLES:
        monkeypatch.setenv(name, "2")
    parent = {name: os.environ.get(name) for name in NATIVE_THREAD_VARIABLES}
    assert run_in_child(f"{HERE}:thread_variables", {}) == parent
    grant = ResourceLease("feature-spread", ResourceAmounts(cpu=2, workers=2))
    with use_resource_lease(grant), child.warm_child_workers(2, modules=("numpy",)) as warm:
        warm()
        with child_calls(f"{HERE}:thread_variables", workers=2) as calls:
            for _ in range(2):
                calls.make({})
            assert all(set(calls.answer().values()) == {"1"} for _ in range(2))


def test_spread_numerical_calls_hold_one_thread_and_leave_normal_calls_unchanged() -> None:
    """Spread numerical calls hold one thread and leave normal calls unchanged."""
    import numpy as np

    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    expected = (values @ values.T).tolist()
    run_in_child(f"{HERE}:limited_once", {})  # the kept worker's first limited call came first
    ordinary = run_in_child(f"{HERE}:numerical_product", {})
    with child_calls(f"{HERE}:numerical_product", workers=2) as calls:
        for _ in range(2):
            calls.make({})
        spread = [calls.answer() for _ in range(2)]
    for answer in spread:
        assert answer["product"] == expected
        assert answer["counts"] and all(threads == 1 for _api, threads in answer["counts"])
        assert answer["after"] == answer["counts"]
    from alphalattice.kernel.resources import ResourceAmounts, ResourceLeases, use_resource_lease

    amounts = ResourceAmounts(cpu=1, workers=1)
    with use_resource_lease(ResourceLeases(amounts).grant("child-test", amounts)):
        scoped = run_in_child(f"{HERE}:numerical_product", {})
    assert scoped["granted_cpu"] == 1, "the kept child carries the Task's CPU grant"
    assert scoped["counts"] and all(threads == 1 for _api, threads in scoped["counts"])
    assert scoped["counts"] == scoped["after"] and scoped["product"] == expected
    again = run_in_child(f"{HERE}:numerical_product", {})
    assert again["granted_cpu"] is None
    assert ordinary["counts"] and ordinary["counts"] == ordinary["after"]
    assert again["counts"] == again["after"] == ordinary["counts"]
    assert ordinary["product"] == again["product"] == expected


def observe_cancellation(*, started: str, stopped: str, cancelled):
    Path(started).write_text("running", encoding="utf-8")
    deadline = time.monotonic() + 10
    while not cancelled() and time.monotonic() < deadline:
        time.sleep(0.02)
    Path(stopped).write_text("cancelled" if cancelled() else "expired", encoding="utf-8")


def test_a_later_start_refusal_cancels_an_already_running_child(monkeypatch, tmp_path):
    """a second worker start refusal drains a real first call through its cancel flag."""
    start = multiprocessing.get_context("spawn").Process.start
    processes = []

    def fail_second(process):
        if processes:
            raise OSError(1455, "fixture second worker start refused")
        start(process)
        processes.append(process)

    running = tmp_path / "running"
    stopped = tmp_path / "stopped"
    with monkeypatch.context() as patch:
        patch.setattr(multiprocessing.get_context("spawn").Process, "start", fail_second)
        # The worker reports its own PID: on Linux the started process is its pool guardian.
        with child_calls(f"{HERE}:doubled", workers=2) as preparation:
            preparation.make({"value": 0})
            worker = preparation.answer()["pid"]
        with (
            pytest.raises(ChildStartFailed, match=r"^task_control\.child_start_failed$"),
            child_calls(f"{HERE}:observe_cancellation", workers=2) as calls,
        ):
            calls.make({"started": str(running), "stopped": str(stopped)})
            deadline = time.monotonic() + 10
            while not running.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert running.exists()
            refused_at = time.monotonic()
            calls.make({"started": str(running), "stopped": str(stopped)})
    assert time.monotonic() - refused_at < 3
    assert stopped.read_text(encoding="utf-8") == "cancelled"
    # The first worker is intentionally kept idle; its next answer proves no owned call remains.
    with child_calls(f"{HERE}:doubled", workers=2) as follow_up:
        follow_up.make({"value": 21})
        assert follow_up.answer()["pid"] == worker
    with child_calls(f"{HERE}:doubled", workers=2) as calls:
        calls.make({"value": 21})
        calls.make({"value": 22})
        assert [calls.answer()["value"] for _ in range(2)] == [42, 44]


class _Counted:
    """A payload that counts, in the parent, each time it is pickled for a worker."""

    pickled = 0

    def __init__(self, data: bytes) -> None:
        self.data = data

    def __reduce__(self) -> tuple[type[_Counted], tuple[bytes]]:
        type(self).pickled += 1
        return (_Counted, (self.data,))


def shared_length(value: int, payload: _Counted, cancelled) -> dict[str, int]:
    return {"value": value, "length": len(payload.data)}


def _check_idle_worker_retirement(monkeypatch) -> None:
    """Kept workers are capped however many cores, and one idle past its wait ends; the next
    call starts it again with the same answer."""
    from alphalattice.control.observation_runtime.telemetry.process_metrics import (
        RuntimeCapacityRequest,
        worker_processes,
    )

    assert worker_processes(64) <= RuntimeCapacityRequest().maximum_kept_workers
    with child_calls(f"{HERE}:doubled", workers=1) as calls:
        calls.make({"value": 1})
        assert calls.answer()["value"] == 2
    pool = child._WORKERS[1]  # a call set's first place; 0 is the Task's own
    deadline = time.monotonic() + 30
    while pool not in child._IDLE and time.monotonic() < deadline:
        time.sleep(0.01)
    waiting = child._IDLE[pool]
    # A replaced wait's callback, already past its cancel, leaves the pool and the wait in place.
    child._retire_idle(1, pool)
    assert child._WORKERS[1] is pool and child._IDLE[pool] is waiting
    # A call that takes the pool takes it out of its wait before it is counted busy.
    assert child._worker(1, threads=1) is pool and pool not in child._IDLE
    child._retire_idle(1, pool)
    assert child._WORKERS[1] is pool
    monkeypatch.setattr(child, "KEPT_WORKER_IDLE_SECONDS", 0.5)
    with child_calls(f"{HERE}:doubled", workers=2) as calls:
        calls.make({"value": 2})
        calls.make({"value": 3})
        assert [calls.answer()["value"], calls.answer()["value"]] == [4, 6]
    started = [process for pool in child._WORKERS.values() for process in pool._processes.values()]
    deadline = time.monotonic() + 30
    while (child._WORKERS or any(p.is_alive() for p in started)) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not child._WORKERS and started and not any(p.is_alive() for p in started)
    with child_calls(f"{HERE}:doubled", workers=1) as calls:
        calls.make({"value": 5})
        assert calls.answer()["value"] == 10


@pytest.mark.proves("task_control.ahead_children_follow_their_owner")
@pytest.mark.parametrize(
    "case", ["idle", "used", "unused", "cancelled-during-start", "worker-dies", "reader-dies"]
)
def test_an_idle_kept_worker_retires_and_the_next_call_starts_it_again(tmp_path, monkeypatch, case):
    """Start ahead, use the same leased worker, and join or replace at the owner's boundary."""
    if case == "idle":
        _check_idle_worker_retirement(monkeypatch)
        return
    if case == "reader-dies":
        create_data_root(tmp_path)
        with HostLog.open(tmp_path):
            reader = reading.LogReader(tmp_path)
            try:
                reader.start_ahead()
                first = reader._child
                assert first is not None
                first.kill()
                first.wait(timeout=10)
                reader.start_ahead()
                assert reader._child is first
                assert reader.read("goal_notes") == []
                replacement = reader._child
                assert replacement is not None and replacement is not first
            finally:
                reader.close()
        assert replacement.poll() is not None
        return

    pools = []

    class Pool:
        def __init__(self, **_):
            self._processes = {1: self}
            self.broken = False
            self.started = False
            self.stopped = False
            self.joined = False
            pools.append(self)

        def submit(self, call, *args):
            future = Future()
            if call is child.os.getpid:
                self.started = True
                return future  # the start can still be in flight when the Task ends
            if self.broken:
                raise child.BrokenProcessPool
            # Spread replies travel through the call's answer file.
            Path(args[4]).write_bytes(pickle.dumps(pools.index(self)))
            future.set_result({"status": "OK"})
            return future

        def terminate(self):
            self.stopped = True

        def shutdown(self, *, wait, cancel_futures):
            self.joined = wait
            self._processes.clear()

    monkeypatch.setattr(child, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(child.spawn, "stop_pool", lambda pool: pool.terminate())
    execution = UUID(int=3)
    lease = ResourceLease(execution, ResourceAmounts(cpu=4, workers=3))
    with use_resource_lease(lease), execution_cancellation(execution):
        with child.warm_child_workers(4, modules=("fixture",)) as start:
            start()
            assert len(pools) == 3 and all(pool.started for pool in pools)
            if case == "cancelled-during-start":
                cancel_execution(execution)
                assert stop_reason() == CANCELLED
            elif case in {"worker-dies", "used"}:
                with child_calls("fixture:work", workers=4) as calls:
                    if case == "worker-dies":
                        pools[0].broken = True
                        with pytest.raises(
                            child.ChildInterrupted, match=r"^task_control\.child_process_ended$"
                        ):
                            calls.make({})
                    calls.make({})
                    replacement = case == "worker-dies"
                    assert calls.answer() == (3 if replacement else 0)
                    assert len(pools) == (4 if replacement else 3)
        assert all(pool.joined and not pool._processes for pool in pools)
        assert all(pool.stopped for pool in pools[1:])
    # The fake's class closes over this list, a cycle that would keep each fake and its real
    # doorbell (five semaphores) until a collection.
    pools.clear()


def test_a_shared_argument_ships_once_to_each_worker_of_a_call_set() -> None:
    """requirement: a shared argument goes to each worker once per call set, by its
    content key after that; another set ships its own, even interleaved on one worker, so
    nothing one set shipped answers another."""
    payload = _Counted(b"x" * 1_000_000)
    _Counted.pickled = 0
    with child_calls(f"{HERE}:shared_length", workers=2) as calls:
        for value in range(6):
            calls.make({"value": value}, shared={"payload": ("payload-key", payload)})
        answers = [calls.answer() for _ in range(6)]
    assert [answer["value"] for answer in answers] == list(range(6))
    assert {answer["length"] for answer in answers} == {1_000_000}
    assert _Counted.pickled == 2
    with child_calls(f"{HERE}:shared_length", workers=2) as calls:
        calls.make({"value": 6}, shared={"payload": ("payload-key", payload)})
        assert calls.answer() == {"value": 6, "length": 1_000_000}
    assert _Counted.pickled == 3
    small = _Counted(b"y" * 10)
    with (
        child_calls(f"{HERE}:shared_length", workers=1) as first,
        child_calls(f"{HERE}:shared_length", workers=1) as second,
    ):
        for value in range(2):
            first.make({"value": value}, shared={"payload": ("payload-key", payload)})
            assert first.answer()["length"] == 1_000_000
            second.make({"value": value}, shared={"payload": ("payload-key", small)})
            assert second.answer()["length"] == 10
