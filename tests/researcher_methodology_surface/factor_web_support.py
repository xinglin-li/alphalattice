"""Shared real Factor and Alpha producers behind the Local Web contract tests."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alphalattice.control.product_host.composition.feature_trials import FeatureTrials
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    read_research_workspace_manifest,
)
from tests.portfolio_strategy_lab.local_web_support import _json, _resolved, _Resolver


def _facts(body: dict) -> dict:  # type: ignore[type-arg]
    """A study answer's facts, without how this read proved them (`verification_basis`, L1): a
    read that reused an earlier full verification and one that verified afresh answer the same."""

    return {key: value for key, value in body.items() if key != "verification_basis"}


def refused_at_the_edges(
    live: LocalPortfolioWebSession, task_id: str, read: dict[str, object]
) -> str:
    """A study with a sealed file changed in place, read as EV2 holds it: a read answers from the
    last full check, saying so, with the facts `read` held; the export proves the bytes and
    refuses by name; once the scrub marks the study, its reads prove in full and refuse by name.
    Returns the scrub's Task."""

    readback = f"/api/experiments/readback?task_id={task_id}"
    held = _json(live, readback)
    assert held["verification_basis"].startswith("SEALED ") and _facts(held) == _facts(read)
    refused = _json(live, f"/api/experiments/export?task_id={task_id}")
    assert refused["status"] == "REFUSED" and refused["failure_code"], refused
    scrub = live.operations.verify_all()
    assert scrub["status"] == "ADMITTED", scrub
    live.dispatcher.drain_for_tests()
    assert str(task_id) in live.operations.sweep.failed_studies()
    refused = _json(live, readback)
    assert refused["status"] == "REFUSED" and refused["failure_code"], refused
    return str(scrub["task_id"])


@contextmanager
def _session(root: Path) -> Iterator[LocalPortfolioWebSession]:
    session = LocalPortfolioWebSession(
        workspace=root,
        workspace_manifest=read_research_workspace_manifest(root),
        resolver=_Resolver(_resolved()),
    )
    with session as live:
        # Four pytest workers share this machine. The operator's real budget
        # limits execution width, while every listing, session, fold and model
        # parameter stays unchanged; model fits still prove their thread canary.
        budget = live.operations.set_cpu_budget("2", chosen_by="EXTERNAL_AUTOMATION")
        assert budget["status"] == "CPU_BUDGET" and budget["cpu_budget"] == 2
        assert live.dispatcher is not None
        event = threading.Event()
        callback = live.dispatcher.on_idle

        def idle() -> None:
            try:
                if callback is not None:
                    callback()
            finally:
                event.set()

        _IDLE_EVENTS[id(live)] = event
        live.dispatcher.on_idle = idle
        try:
            yield live
        finally:
            live.dispatcher.on_idle = callback
            del _IDLE_EVENTS[id(live)]


_IDLE_EVENTS: dict[int, threading.Event] = {}


def _drain_trial(live: LocalPortfolioWebSession) -> threading.Event:
    event = _IDLE_EVENTS[id(live)]
    event.clear()
    live.dispatcher.drain_for_tests(timeout=600)
    return event


def assert_trial_reads_during_saves(trials: FeatureTrials, trial_id: str) -> None:
    """Every one of 20,000 reads sees the whole record while another thread saves it."""
    trial, together = trials._read(trial_id), threading.Barrier(2, timeout=30)
    completed, failures = {"reads": 0, "writes": 0}, []

    def save_repeatedly():
        together.wait()
        for _ in range(20_000):
            try:
                trials._save(trial)
            except OSError as error:
                failures.append(("write", type(error).__name__))
            completed["writes"] += 1

    writer = threading.Thread(target=save_repeatedly, daemon=True)
    writer.start()
    try:
        together.wait()
        for _ in range(20_000):
            try:
                assert trials._read(trial_id) == trial
            except OSError as error:
                failures.append(("read", type(error).__name__))
            completed["reads"] += 1
    finally:
        writer.join(timeout=120)
    assert not writer.is_alive()
    assert completed == {"reads": 20_000, "writes": 20_000}
    assert failures == [], (len(failures), failures[:8])


def _alpha_payload(case, parameters=None, handle="capability-1"):
    from copy import deepcopy

    _root, binding, factor_task, decision, original = case
    document = deepcopy(original)
    document["alpha"].update(
        target_recipe_id="SECTOR_RESIDUAL_ROBUST_Z",
        model_capability_handle=handle,
        model_parameters=parameters or {"family": "ridge", "alpha": 1.0},
    )
    return {
        "research_input_id": binding.input_id,
        "input_binding_hash": binding.binding_hash,
        "factor_task_id": factor_task,
        "curation_receipt_hash": decision,
        "experiment_document": document,
    }


def _publish_factor(root: Path):
    with _session(root) as session:
        controls = _json(session, "/api/experiments/controls")
        assert controls["status"] == "READY", controls
        document = controls["template"]
        # The universe is offered whole or sampled.
        (universe,) = (
            v for v in controls["controls"] if v["path"] == ["experiment", "universe_handle"]
        )
        assert (universe["type"], universe["value"]) == (
            "universe",
            document["experiment"]["universe_handle"],
        )
        document["factor"]["factor_ids"] = controls["factor_options"][:4]
        plan = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": document},
        )
        assert plan["status"] == "PLANNED", plan
        assert plan["numerical_call_count"] == 0
        run = _json(
            session,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        assert run["task_id"], run
        assert session.dispatcher is not None
        session.dispatcher.drain_for_tests()
        task_id = run["task_id"]
        report = _json(session, f"/api/experiments/readback?task_id={task_id}")
        assert report["status"] == "EXPERIMENT_PUBLISHED", (
            report,
            _json(session, f"/api/status?task_id={task_id}"),
        )
        assert report["program"] == plan["program"]
        repeat = _json(
            session,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        assert repeat["status"] == "REUSED_EXACT" and repeat["task_id"] is None
        exported = _json(session, f"/api/experiments/export?task_id={task_id}")
    return task_id, report, exported
