"""The Portfolio product-update exercise over an eighty-name real Data/Feature
workspace, and its process-death phase table.

The daily update arm, the single-action catch-up arm, the research-update
recovery children and the `_crash_child` table they report their boundary
through. The strategy scoring suite drives this on the expensive workspace it
already built rather than constructing another ten-year workspace for the
same boundary; it used to import the exercise from the decision-update suite,
which kept only its numerical cases as its own. The phase table lives here
because `forced_process` imports `_crash_child` from the module it is handed,
and that module is this one.
Test support beside its owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.synthetic_numerical import HASH, build_numerical


def _update_run(service, prepared=None, through=None):
    from tests.portfolio_strategy_lab.local_web_support import _json
    from tests.portfolio_strategy_lab.scoring_runs import PACKAGE

    payload = {"strategy_package_id": PACKAGE}
    if prepared is not None:
        payload["prepared_input_hash"] = prepared
    if through is not None:
        payload["observed_through"] = through.isoformat()
    plan = _json(service, "/api/portfolio-update/plan", method="POST", payload=payload)
    assert plan["status"] == "PLANNED", plan
    request = dict(plan["next_requests"]["run"])
    assert request == {
        "operation": "PORTFOLIO_UPDATE_RUN",
        "update_plan_hash": plan["update_plan_hash"],
    }
    route = _json(service, "/api/session")["routes"][request.pop("operation")]
    sent = _json(
        service,
        route["path"],
        method=route["method"],
        payload=request,
    )
    assert sent["status"] in {"ADMITTED", "REUSED_EXACT", "REUSED_IN_FLIGHT"}, sent
    service.dispatcher.drain_for_tests()
    result = _json(
        service,
        "/api/portfolio-update" + (f"?task_id={sent['task_id']}" if sent["task_id"] else ""),
    )
    assert result["status"] in {"PROPOSAL_PUBLISHED", "ENTRY_SETTLED", "OUTCOME_SETTLED"}, result
    return plan, sent, result


def exercise_product_updates(workspace, calibration, monkeypatch, *, symbols):
    """Called by the existing 80-name real-Data/Feature/12-model product fixture."""
    import shutil
    from pathlib import Path

    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        admit_research_workspace,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
    )
    from bind_portfolio_decision_checkpoint import bind_checkpoint
    from tests.portfolio_strategy_lab.local_web_support import InstalledAgent, _json
    from tests.portfolio_strategy_lab.process_death import forced_process
    from tests.portfolio_strategy_lab.scoring_runs import PACKAGE, _calibration_run, _plan_run
    from tests.workspace_maintenance.local_data_provider import (
        recording_provider,
        unchanged_membership_source,
    )

    admitted = admit_research_workspace(workspace)
    binding = admitted.manifest.score_inputs[0]
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

    universe = MarketDataRepository(workspace).current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    labels = tuple(v.symbol for v in sorted(universe.listings, key=lambda v: v.listing_id))
    n = build_numerical(
        installed_package=admitted.catalog.packages_by_strategy_id()[PACKAGE],
        listing_ids=tuple(calibration["input"]["ordered_listing_ids"]),
        first_day=date(2026, 8, 3),
        model_hash=binding.authority_hash,
        model_recipe_hash=calibration["input"]["component_recipe_hash"],
        listing_labels=labels,
    )
    original_packages = admitted.catalog.catalog_hash
    bind_checkpoint(workspace, n.checkpoint, expected_hash=n.checkpoint.content_hash)
    assert admit_research_workspace(workspace).catalog.catalog_hash == original_packages
    # Installation is idempotent and cannot admit a different starting book.
    before = (workspace / "research-workspace.json").read_bytes()
    bind_checkpoint(workspace, n.checkpoint, expected_hash=n.checkpoint.content_hash)
    assert (workspace / "research-workspace.json").read_bytes() == before
    now = datetime(2026, 8, 3, 23, tzinfo=UTC)
    service = LocalPortfolioWebSession.from_workspace(workspace, clock=lambda: now)
    service.start()
    frozen_proposal = None
    try:
        initial_tasks = len(service.session.task_control_registry.tasks())
        plan, sent, result = _update_run(service, calibration["input"]["content_hash"])
        assert len(service.session.task_control_registry.tasks()) == initial_tasks + 1
        frozen_proposal = result["publication"]["pending_proposal"]
        assert tuple(result["listing_labels"].values()) == labels
        with pytest.raises(ValueError, match="published_input_source_mismatch"):
            service.operations.calibration.published_input(
                calibration["input"]["content_hash"], expected_source_hash=HASH
            )
        assert result["publication"]["events"] == []
        bridge = InstalledAgent(service.operations)
        from uuid import UUID

        agent = json.loads(
            bridge.invoke(
                PortfolioResearchAgentRequest(
                    operation="PORTFOLIO_UPDATE_READBACK", task_id=UUID(sent["task_id"])
                )
            )
        )
        assert agent == result
        reused = _json(
            service,
            "/api/portfolio-update/run",
            method="POST",
            payload={"update_plan_hash": plan["update_plan_hash"]},
        )
        assert reused["status"] == "REUSED_EXACT" and reused["task_id"] is None
        assert len(service.session.task_control_registry.tasks()) == initial_tasks + 1
        assert "CLOSE_MARKED_ESTIMATE" in result["html"]
        # Preserve this exact issued prefix for the unified catch-up comparison.
        # Close every writer before copying; the daily arm below remains intact.
        service.stop()
        catchup_workspace = workspace.parent / "single-action-catchup"
        shutil.copytree(workspace, catchup_workspace)
        service = LocalPortfolioWebSession.from_workspace(workspace, clock=lambda: now)
        service.start()
        # Materialize the two admitted QA bars once, then reveal the prefix one
        # date at a time. This also proves an on-disk future bar is not consumed.
        now = datetime(2026, 8, 5, 23, tzinfo=UTC)
        service.operations.data_update.clock = lambda: now
        service.operations.data_update.provider = recording_provider(now=now, symbols=symbols)
        service.operations.data_update.source_loader = unchanged_membership_source(symbols)
        update = _json(service, "/api/data-update/plan", method="POST", payload={})
        assert update["status"] == "PLANNED", update
        _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": update["plan_hash"]},
        )
        service.dispatcher.drain_for_tests()
        for day in (date(2026, 8, 4), date(2026, 8, 5)):
            _, _, score = _plan_run(service, day.isoformat())
            _, _, prepared = _calibration_run(service, score)
            _, _, result = _update_run(service, prepared["input"]["content_hash"], day)
            assert result["history"][0]["pending_proposal"] == frozen_proposal
            phases = [e["phase"] for e in result["publication"]["events"]]
            assert "ENTRY_SETTLED" in phases
            assert ("OUTCOME_SETTLED" in phases) == (day.day == 5)
        final_task = result["task_id"]
        final_body = result
    finally:
        service.stop()
    with_reopen = LocalPortfolioWebSession.from_workspace(workspace, clock=lambda: now)
    with_reopen.start()
    try:
        assert _json(with_reopen, f"/api/portfolio-update?task_id={final_task}") == final_body
    finally:
        with_reopen.stop()
    exercise_single_action_catchup(catchup_workspace, final_body, monkeypatch, symbols=symbols)
    # Each recovery child opens a disposable copy with the same real source
    # and actor-free product composition. Settle the Aug-5 entry on Aug-6.
    recovery_source = workspace.parent / "recovery-observations"
    shutil.copytree(workspace, recovery_source)
    child_service = LocalPortfolioWebSession.from_workspace(
        recovery_source, clock=lambda: datetime(2026, 8, 6, 23, tzinfo=UTC)
    )
    child_service.data_provider = recording_provider(
        now=datetime(2026, 8, 6, 23, tzinfo=UTC), symbols=symbols
    )
    child_service.data_source_loader = unchanged_membership_source(symbols)
    child_service.start()
    try:
        update = _json(child_service, "/api/data-update/plan", method="POST", payload={})
        _json(
            child_service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": update["plan_hash"]},
        )
        child_service.dispatcher.drain_for_tests()
    finally:
        child_service.stop()
    for stage in ("seal_market_observations", "publish_decision_settlement"):
        copied = workspace.parent / ("recovery-" + stage)
        shutil.copytree(recovery_source, copied)
        with forced_process(
            copied, __name__, stage, module_dir=Path(__file__).parent
        ) as interrupted:
            task_id = interrupted["task_id"]
        resumed = LocalPortfolioWebSession.from_workspace(copied)
        resumed.start()
        try:
            resumed.dispatcher.drain_for_tests()
            answer = _json(resumed, f"/api/portfolio-update?task_id={task_id}")
            assert answer["status"] == "ENTRY_SETTLED", answer
            history = resumed.application.ledger.decision_history(n.checkpoint.content_hash)
            assert len(history) == 4
            assert history[0].pending_proposal.model_dump(mode="json") == frozen_proposal
        finally:
            resumed.stop()


def exercise_single_action_catchup(workspace, daily, monkeypatch, *, symbols):
    """One new request versus the existing daily real-owner arm, on the same source."""
    import shutil
    from pathlib import Path
    from time import perf_counter
    from uuid import UUID

    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
    )
    from tests.portfolio_strategy_lab.local_web_support import InstalledAgent, _json
    from tests.portfolio_strategy_lab.process_death import forced_process
    from tests.portfolio_strategy_lab.scoring_runs import PACKAGE
    from tests.workspace_maintenance.local_data_provider import (
        recording_provider,
        unchanged_membership_source,
    )

    now = datetime(2026, 8, 5, 23, tzinfo=UTC)
    recovery_base = workspace.parent / "catchup-recovery-base"
    shutil.copytree(workspace, recovery_base)
    service = LocalPortfolioWebSession.from_workspace(workspace, clock=lambda: now)
    service.data_provider = recording_provider(now=now, symbols=symbols)
    service.data_source_loader = unchanged_membership_source(symbols)
    service.start()
    try:
        before = len(service.session.task_control_registry.tasks())
        plan = _json(
            service,
            "/api/research-update/plan",
            method="POST",
            payload={"strategy_package_id": PACKAGE, "observed_through": "2026-08-05"},
        )
        assert plan["status"] == "PLANNED", plan
        assert plan["decision_sessions"] == ["2026-08-04", "2026-08-05"]
        started = perf_counter()
        sent = _json(
            service,
            "/api/research-update/run",
            method="POST",
            payload={"update_plan_hash": plan["update_plan_hash"]},
        )
        assert sent["status"] == "ADMITTED", sent
        repeated = _json(
            service,
            "/api/research-update/run",
            method="POST",
            payload={"update_plan_hash": plan["update_plan_hash"]},
        )
        assert repeated["status"] == "REUSED_IN_FLIGHT" and repeated["task_id"] == sent["task_id"]
        service.dispatcher.drain_for_tests()
        task = service.session.task_control_registry.task(UUID(sent["task_id"]))
        assert task.lifecycle.value == "SUCCEEDED", service.dispatcher.status(task.task_id)
        assert len(service.session.task_control_registry.tasks()) == before + 1
        print("CU_C2_CATCHUP_SECONDS", perf_counter() - started, flush=True)
        body = _json(service, "/api/research-update?task_id=" + sent["task_id"])
        # A published update remains a verified closure on every cache hit.
        report = (
            service.application.ledger.root / "html" / (body["publication"]["html_hash"] + ".html")
        )
        original_report = report.read_bytes()
        try:
            report.write_bytes(original_report + b"tampered")
            refused = _json(
                service,
                "/api/research-update/run",
                method="POST",
                payload={"update_plan_hash": plan["update_plan_hash"]},
            )
            assert (
                refused["status"] == "REFUSED" and "artifact_tampered" in refused["failure_code"]
            ), refused
            assert len(service.session.task_control_registry.tasks()) == before + 1
        finally:
            report.write_bytes(original_report)
        from alphalattice.control.product_host.composition.decision_advancement import (
            STAGES,
            DecisionAdvancementStep,
        )

        owner = service.operations.research_updates
        captured = owner.prepare(plan["update_plan_hash"])
        index = owner.index.path(
            "decision-stages", canonical_hash([captured.content_hash, STAGES[3]])
        )
        original_index = index.read_bytes()
        entry = json.loads(original_index)
        old_step = DecisionAdvancementStep.model_validate(entry["payload"])
        changed_step = DecisionAdvancementStep.create(
            **{
                **{
                    k: getattr(old_step, k)
                    for k in type(old_step).model_fields
                    if k != "content_hash"
                },
                "source_hash": canonical_hash("validly rehashed substitution"),
            }
        )
        entry.update(
            identity=changed_step.content_hash, payload=changed_step.model_dump(mode="json")
        )
        try:
            index.write_text(json.dumps(entry), encoding="utf-8")
            refused = _json(service, "/api/research-update?task_id=" + sent["task_id"])
            assert refused == {
                "status": "REFUSED",
                "failure_code": "research_update.task_evidence_invalid",
            }
        finally:
            index.write_bytes(original_index)
        with monkeypatch.context() as future:
            future.setattr(
                service.operations.research_updates,
                "clock",
                lambda: datetime(2026, 10, 1, 23, tzinfo=UTC),
            )
            refused = _json(
                service,
                "/api/research-update/plan",
                method="POST",
                payload={"strategy_package_id": PACKAGE, "observed_through": "2026-10-01"},
            )
            assert (
                refused["status"] == "REFUSED"
                and "model_epoch_unavailable" in refused["failure_code"]
            )
            assert len(service.session.task_control_registry.tasks()) == before + 1
        assert body["publication"]["claim"] == "POST_OBSERVED_QA_NOT_TIMELY_ADVICE"
        prepared = body["publication"]["pending_proposal"]["input"]
        assert (
            service.operations.calibration.published_input(prepared["content_hash"]).model_dump(
                mode="json"
            )
            == prepared
        )
        assert body["publication"]["book"] == daily["publication"]["book"]
        for actual, expected in zip(body["history"], daily["history"], strict=True):
            a, b = actual["pending_proposal"], expected["pending_proposal"]
            if a is not None:
                for field in (
                    "estimated_weights",
                    "close_weights",
                    "reference_weights",
                    "schedule",
                    "book",
                ):
                    assert a[field] == b[field], field
                for field in ("bucket_means", "sizing_rule", "scores", "decision_eligible"):
                    if field in a["input"]:
                        assert a["input"][field] == b["input"][field], field
            for event, reference in zip(actual["events"], expected["events"], strict=True):
                for field in (
                    "target_weights",
                    "turnover",
                    "gross_return",
                    "net_return_5bps",
                    "net_return_10bps",
                    "entry",
                ):
                    assert event[field] == reference[field], field
        bridge = InstalledAgent(service.operations)
        assert (
            json.loads(
                bridge.invoke(
                    PortfolioResearchAgentRequest(
                        operation="RESEARCH_UPDATE_READBACK", task_id=UUID(sent["task_id"])
                    )
                )
            )
            == body
        )
        reused_at = perf_counter()
        again = _json(
            service,
            "/api/research-update/run",
            method="POST",
            payload={"update_plan_hash": plan["update_plan_hash"]},
        )
        assert again == {
            "status": "REUSED_EXACT",
            "task_id": None,
            "publication_task_id": sent["task_id"],
        }
        print("CU_C2_REUSE_SECONDS", perf_counter() - reused_at, flush=True)
        assert len(service.session.task_control_registry.tasks()) == before + 1
    finally:
        service.stop()
    resumed = LocalPortfolioWebSession.from_workspace(workspace, clock=lambda: now)
    resumed.start()
    try:
        assert _json(resumed, "/api/research-update?task_id=" + sent["task_id"]) == body
    finally:
        resumed.stop()
    # Recovery tests reuse the already-materialized input data, retaining the
    # original issued Task/book prefix. No second history builder or data fetch.
    shutil.copy2(workspace / "market-data.duckdb", recovery_base / "market-data.duckdb")
    shutil.copytree(workspace / "artifacts", recovery_base / "artifacts", dirs_exist_ok=True)
    for stage in ("seal_inputs", "prepare_scores", "publish_readback"):
        copied = workspace.parent / ("catchup-recovery-" + stage)
        shutil.copytree(recovery_base, copied)
        with forced_process(
            copied,
            __name__,
            "research_" + stage,
            module_dir=Path(__file__).parent,
            boundary_timeout_seconds=180,
        ) as interrupted:
            task_id = interrupted["task_id"]
        from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
        from alphalattice.investment.alpha_research.scores.frozen_inference import (
            AdmittedFrozenInference,
        )

        with monkeypatch.context() as guard:

            def forbidden(*_args, **_kwargs):
                raise AssertionError("Recovery re-read live bars or repeated verified predictions")

            guard.setattr(MarketDataRepository, "raw_bars", forbidden)
            if stage != "seal_inputs":
                guard.setattr(AdmittedFrozenInference, "score", forbidden)
            recovered = LocalPortfolioWebSession.from_workspace(copied, clock=lambda: now)
            recovered.start()
            try:
                recovered.dispatcher.drain_for_tests()
                answer = _json(recovered, "/api/research-update?task_id=" + task_id)
                assert answer["status"] == "PROPOSAL_PUBLISHED", answer
                assert answer["publication"]["book"] == body["publication"]["book"]
                assert len(answer["history"]) == len(body["history"])
                assert task_id in tuple(str(v) for v in recovered.resumed_task_ids)
            finally:
                recovered.stop()
    from alphalattice.control.product_host.composition.decision_advancement import (
        DecisionAdvancementApplication,
    )

    cancelled_workspace = workspace.parent / "catchup-cancelled"
    shutil.copytree(recovery_base, cancelled_workspace)
    execute = DecisionAdvancementApplication.execute_stage

    def cancel_after_seal(owner, **kwargs):
        result = execute(owner, **kwargs)
        if kwargs["work_item"].stage_id == "seal_inputs":
            registry = owner.session.task_control_registry
            record = registry.task(kwargs["task"].task_id)
            registry.request_cancel(
                task_id=record.task_id, expected_task_hash=record.record_hash, observed_at=now
            )
        return result

    with monkeypatch.context() as cancellation:
        cancellation.setattr(DecisionAdvancementApplication, "execute_stage", cancel_after_seal)
        cancellation.setattr(AdmittedFrozenInference, "score", forbidden)
        cancelled = LocalPortfolioWebSession.from_workspace(cancelled_workspace, clock=lambda: now)
        cancelled.data_provider = recording_provider(now=now, symbols=symbols)
        cancelled.data_source_loader = unchanged_membership_source(symbols)
        cancelled.start()
        try:
            selected = _json(
                cancelled,
                "/api/research-update/plan",
                method="POST",
                payload={"strategy_package_id": PACKAGE, "observed_through": "2026-08-05"},
            )
            admitted = _json(
                cancelled,
                "/api/research-update/run",
                method="POST",
                payload={"update_plan_hash": selected["update_plan_hash"]},
            )
            cancelled.dispatcher.drain_for_tests()
            result = _json(cancelled, "/api/research-update?task_id=" + admitted["task_id"])
            assert result["status"] == "CANCELLED" and result["publication_is_previous"] is True
            assert len(result["history"]) == 1
        finally:
            cancelled.stop()


def _research_crash_child(workspace, stage):
    import os
    from pathlib import Path
    from threading import Event

    from alphalattice.control.product_host.composition.decision_advancement import (
        DecisionAdvancementApplication,
    )
    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from tests.portfolio_strategy_lab.local_web_support import _json
    from tests.portfolio_strategy_lab.scoring_runs import PACKAGE
    from tests.workspace_maintenance.local_data_provider import (
        PRODUCT_QA_SYMBOLS,
        recording_provider,
        unchanged_membership_source,
    )

    now = datetime(2026, 8, 5, 23, tzinfo=UTC)
    original = DecisionAdvancementApplication.execute_stage

    def interrupted(owner, **kwargs):
        result = original(owner, **kwargs)
        if kwargs["work_item"].stage_id == stage:
            assert owner.readback(kwargs["task"].task_id)["publication_is_previous"] is True
            print(
                "CRASH_BOUNDARY:"
                + json.dumps({"pid": os.getpid(), "task_id": str(kwargs["task"].task_id)}),
                flush=True,
            )
            Event().wait()
        return result

    DecisionAdvancementApplication.execute_stage = interrupted
    service = LocalPortfolioWebSession.from_workspace(Path(workspace), clock=lambda: now)
    service.data_provider = recording_provider(now=now, symbols=PRODUCT_QA_SYMBOLS)
    service.data_source_loader = unchanged_membership_source(PRODUCT_QA_SYMBOLS)
    service.start()
    plan = _json(
        service,
        "/api/research-update/plan",
        method="POST",
        payload={
            "strategy_package_id": PACKAGE,
            "observed_through": "2026-08-05",
        },
    )
    sent = _json(
        service,
        "/api/research-update/run",
        method="POST",
        payload={"update_plan_hash": plan["update_plan_hash"]},
    )
    assert sent["status"] == "ADMITTED", sent
    service.dispatcher.drain_for_tests()


def _crash_child(workspace: str, phase: str) -> None:
    if phase.startswith("research_"):
        return _research_crash_child(workspace, phase.removeprefix("research_"))
    import os
    from pathlib import Path
    from threading import Event

    from alphalattice.control.product_host.composition.local_web_session import (
        LocalPortfolioWebSession,
    )
    from alphalattice.control.product_host.composition.portfolio_updates import (
        PortfolioUpdateApplication,
    )

    original = PortfolioUpdateApplication.execute_stage

    def interrupted(owner, **kwargs):
        result = original(owner, **kwargs)
        if kwargs["work_item"].stage_id == phase:
            visible = owner.readback(kwargs["task"].task_id)
            assert visible["publication_is_previous"] is True
            assert len(visible["history"]) == 3
            print(
                "CRASH_BOUNDARY:"
                + json.dumps({"pid": os.getpid(), "task_id": str(kwargs["task"].task_id)}),
                flush=True,
            )
            Event().wait()
        return result

    PortfolioUpdateApplication.execute_stage = interrupted
    service = LocalPortfolioWebSession.from_workspace(Path(workspace))
    service.start()
    _update_run(service)


__all__ = ["exercise_product_updates", "exercise_single_action_catchup"]
