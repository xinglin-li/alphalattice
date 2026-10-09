"""The real Factor owner behind Local Web, independent of Portfolio numerics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from threadpoolctl import threadpool_limits

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    bind_factor_inputs,
    read_factor_bundle,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import AgentExecutionBinding
from alphalattice.protocols.research_authoring.contracts import AuthoringError
from tests.portfolio_strategy_lab.local_web_support import (
    InstalledAgent,
    _json,
    _manifest,
    _request,
    _resolved,
    _Resolver,
)
from tests.researcher_methodology_surface.factor_web_support import (
    _alpha_payload,
    _drain_trial,
    _session,
)
from tests.researcher_methodology_surface.real_workspace import (
    publish_causal_outcomes,
)
from tests.researcher_methodology_surface.session_workspace import (
    copy_workspace,
    session_workspace,
)


def _facts(body: dict) -> dict:  # type: ignore[type-arg]
    """A study answer's facts, without how this read proved them (`verification_basis`, L1): a
    read that reused an earlier full verification and one that verified afresh answer the same."""

    return {key: value for key, value in body.items() if key != "verification_basis"}


def _wait_for_boundary(process: subprocess.Popen[bytes], marker: Path, timeout: float) -> bool:
    """The child signals only after its committed marker was written; EOF wakes a failure."""
    ready = threading.Event()

    def read_signal() -> None:
        for line in process.stdout:
            if line.strip() == b"TEST_BOUNDARY":
                break
        ready.set()

    reader = threading.Thread(target=read_signal, daemon=True)
    reader.start()
    ready.wait(timeout)
    return marker.exists()


@pytest.fixture(scope="module")
def completed_alpha_seed(alpha_seed, tmp_path_factory):
    """Readback, handoff and Feature trials need this model as input, not another fit."""

    def build(root: Path) -> dict:
        copy_workspace(alpha_seed[0], root)
        case = (root, *alpha_seed[1:])
        with _session(root) as live:
            plan = _json(live, "/api/experiments/plan", method="POST", payload=_alpha_payload(case))
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            report = _json(live, f"/api/experiments/readback?task_id={sent['task_id']}")
            assert report["status"] == "EXPERIMENT_PUBLISHED", report
        return {}

    source, _metadata = session_workspace(tmp_path_factory, "completed_alpha", build)
    return source, *alpha_seed[1:]


@pytest.fixture
def completed_alpha_case(completed_alpha_seed, tmp_path_factory):
    root = copy_workspace(
        completed_alpha_seed[0], tmp_path_factory.mktemp("completed-alpha") / "workspace"
    )
    return root, *completed_alpha_seed[1:4], deepcopy(completed_alpha_seed[4])


def _lightgbm_parameters(seed=1729):
    """The installed G6 estimator point, as an author would write it in YAML."""

    from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
        build_dynamic_panel_lightgbm_recipe,
    )
    from alphalattice.investment.alpha_research.scores.product_recipe import PRODUCT_ESTIMATOR_POINT

    return dict(
        build_dynamic_panel_lightgbm_recipe(PRODUCT_ESTIMATOR_POINT.resolve(seed=seed)).parameters
    )


def test_completed_alpha_hands_off_to_portfolio_without_refitting(
    completed_alpha_case, monkeypatch, tmp_path
):
    alpha_case = completed_alpha_case
    from uuid import UUID

    from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
        HistoricalTradabilityBuilder,
        TradabilityBuildCancelled,
    )
    from alphalattice.investment.alpha_research.experiments.development_execution import (
        AlphaExperimentExecutor,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
        LANES,
        SEGMENTS,
        PortfolioExperimentCancelled,
        PortfolioReplaySegment,
    )
    from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
        PortfolioResearchArtifactStore,
    )

    root = alpha_case[0]
    with _session(root) as live:
        manifest = (root / "research-workspace.json").read_bytes()
        parents = [
            r
            for r in _json(live, "/api/experiments")["experiments"]
            if r["kind"] == "alpha.model-development" and r["lifecycle"] == "SUCCEEDED"
        ]
        if parents:
            parent_id = parents[0]["task_id"]
        else:
            # A standalone test creates its source once. The handoff itself may never fit.
            plan = _json(
                live, "/api/experiments/plan", method="POST", payload=_alpha_payload(alpha_case)
            )
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            parent_id = sent["task_id"]
        alpha = _json(live, f"/api/experiments/readback?task_id={parent_id}")
        assert alpha["status"] == "EXPERIMENT_PUBLISHED", alpha
        monkeypatch.setattr(
            AlphaExperimentExecutor,
            "execute",
            lambda *a, **k: pytest.fail("Portfolio refitted Alpha"),
        )
        candidate = alpha["result"]["candidates"][0]["candidate_id"]
        before = len(live.session.task_control_registry.tasks())
        draft = _json(
            live,
            "/api/experiments/portfolio-draft",
            method="POST",
            payload={"task_id": parent_id, "candidate_id": candidate},
        )
        assert draft["status"] == "PORTFOLIO_DRAFT_READY", draft
        assert draft["numerical_call_count"] == 0
        request = {"experiment_document": draft["document"]}
        plan = _json(live, "/api/experiments/plan", method="POST", payload=request)
        assert plan["status"] == "PLANNED", plan
        yaml_plan = _json(
            live, "/api/experiments/plan", method="POST", payload={"experiment_yaml": draft["yaml"]}
        )
        assert yaml_plan == plan
        bridge = InstalledAgent(live.operations)
        assert (
            json.loads(
                bridge.invoke(PortfolioResearchAgentRequest(operation="EXPERIMENT_PLAN", **request))
            )
            == plan
        )
        assert len(live.session.task_control_registry.tasks()) == before
        sent = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
            # Admission validates this large retained input before queuing.
            # This is an HTTP functional wait, not a latency or science budget.
            timeout=180,
        )
        live.dispatcher.drain_for_tests()
        task_id = sent.get("task_id") or sent["publication_task_id"]
        body = _json(live, f"/api/experiments/readback?task_id={task_id}")
        assert body["status"] == "EXPERIMENT_PUBLISHED", body
        assert body["portfolio_source"]["alpha_task_id"] == parent_id
        holds = [r for r in body["series"] if r["decision_mode"] == "HOLD"]
        assert len(holds) == body["execution_preview"]["hold_session_count"] > 0
        assert all(r["one_way_turnover"] == 0 for r in holds)
        selected = _json(
            live,
            f"/api/experiments/readback?task_id={task_id}&portfolio_session={holds[0]['session']}",
        )
        assert selected["position"]["session"] == holds[0]["session"]
        assert selected["metric_units"]["position.one_way_turnover"] == "portfolio fraction"
        exported = _json(
            live,
            f"/api/experiments/export?task_id={task_id}&portfolio_session={holds[0]['session']}",
        )
        assert _facts(json.loads(exported["json"])) == _facts(selected)
        assert "Portfolio development replay" in exported["html"]
        agent_read = json.loads(
            bridge.invoke(
                PortfolioResearchAgentRequest(
                    operation="EXPERIMENT_READBACK",
                    task_id=task_id,
                    portfolio_session=holds[0]["session"],
                )
            )
        )
        assert _facts(agent_read) == _facts(selected)
        assert (
            _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )["status"]
            == "REUSED_EXACT"
        )
        assert len(live.session.task_control_registry.tasks()) == before + (
            1 if sent.get("task_id") else 0
        )
        bad = json.loads(json.dumps(draft["document"]))
        bad["portfolio"]["weight_rule"] = "mu.iv1"
        assert (
            _json(
                live, "/api/experiments/plan", method="POST", payload={"experiment_document": bad}
            )["status"]
            == "REFUSED"
        )
        prepared, _ = live.operations.experiments._prepare_portfolio(UUID(parent_id), candidate)

        def cancel_input_build(*args, **kwargs):
            raise TradabilityBuildCancelled()

        with monkeypatch.context() as patch:
            patch.setattr(HistoricalTradabilityBuilder, "build", cancel_input_build)
            with pytest.raises(PortfolioExperimentCancelled, match="market_preparation_cancelled"):
                prepared.executor.inputs(tmp_path / "unused-output")
        output = root / body["document"]["experiment"]["output_workspace"]
        store = PortfolioResearchArtifactStore(output)
        segment = store.load(
            category=SEGMENTS,
            content_hash=body["receipt"]["segments"][0],
            model=PortfolioReplaySegment,
            identity_field="segment_hash",
        )
        path = next((store.root / LANES).glob(f"{segment.lanes['executed']}.*"))
        content = path.read_bytes()
        try:
            path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
            assert (
                _json(live, f"/api/experiments/readback?task_id={task_id}")["status"] == "REFUSED"
            )
        finally:
            path.write_bytes(content)
        assert (root / "research-workspace.json").read_bytes() == manifest
    with _session(root) as live:
        again = _json(live, f"/api/experiments/readback?task_id={task_id}")
        assert _facts(again) == _facts(body)
        assert (
            _json(
                live,
                f"/api/experiments/export?task_id={task_id}&portfolio_session={holds[0]['session']}",
            )
            == exported
        )

    _exercise_authored_review(root, selected, tmp_path)


def _exercise_authored_review(root, selected, tmp_path):
    """Copy the completed book for isolated review QA; no additional numerical build."""
    from dataclasses import replace
    from types import SimpleNamespace
    from urllib.parse import urlencode
    from uuid import UUID

    from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
        LANES,
        SEGMENTS,
        PortfolioReplaySegment,
    )
    from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
        PortfolioResearchArtifactStore,
    )
    from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector
    from tests.alternative_evidence_desk.planted_corpus import _NOW
    from tests.alternative_evidence_desk.review_http_support import _Service, build_authority

    workspace = tmp_path / "authored-review"
    shutil.copytree(root, workspace)
    original = read_research_workspace_manifest(workspace)
    manifest = ResearchWorkspaceManifest.research_only(original.workspace_id).with_bindings(
        experiment_inputs=original.experiment_inputs,
        data_update=original.data_update,
    )
    publish_research_workspace_manifest(workspace, manifest)
    authority = build_authority(
        tmp_path=tmp_path,
        report=SimpleNamespace(
            window_end_book=SimpleNamespace(
                positions=tuple(
                    SimpleNamespace(listing_id=v)
                    for v in selected["receipt"]["source"]["ordered_listing_ids"]
                )
            )
        ),
    )
    selector = selected["review_selector"]
    query = urlencode(selector)

    def boot(admitted):
        return LocalPortfolioWebSession.from_workspace(
            workspace,
            review_authority=admitted,
            clock=lambda: _NOW,
        )

    try:
        with boot(authority) as live:
            service = _Service(live, tmp_path)
            assert not live.operations.installed()
            section = service.get("/api/evidence-cro?" + query)
            subject = section["book"]["experiment_subject"]
            assert subject["experiment_receipt_hash"] == selected["receipt"]["receipt_hash"]
            assert subject["portfolio_session"] == selected["position"]["session"]
            assert subject["preceding_session"] == selected["preceding_position"]["session"]
            assert section["book"]["result_hash"] is None
            scope = service.review.resolve_book(
                BookSelector(
                    **{**selector, "experiment_task_id": UUID(selector["experiment_task_id"])}
                )
            )
            positions = {p.listing_id: p for p in scope.projection.positions}
            for i, listing in enumerate(selected["receipt"]["source"]["ordered_listing_ids"]):
                if listing in positions:
                    assert positions[listing].ending_weight == selected["position"]["weights"][i]
                    assert (
                        positions[listing].preceding_weight
                        == selected["preceding_position"]["weights"][i]
                    )
            before = len(service.registry.tasks())
            for wrong, refused in (
                (
                    {"experiment_receipt_hash": "0" * 64},
                    "product_host.evidence_review_experiment_receipt_mismatch",
                ),
                ({"portfolio_session": "1900-01-01"}, "portfolio_research.session_outside_report"),
            ):
                status, body = service.request(
                    "/api/evidence-cro?" + urlencode({**selector, **wrong})
                )
                # Refused by name and worded with the way on (V546).
                assert (status, body["failure_code"]) == (400, refused), body
                assert body["detail"] and body["next_action"], body
            # A result beside the study's book is refused in words, with the history to choose
            # one from (V546).
            status, body = service.request(
                "/api/evidence-cro?" + urlencode({**selector, "result_hash": "0" * 64})
            )
            assert (status, body["status"], body["failure_code"]) == (
                200,
                "REFUSED",
                "product_host.evidence_review_selector_ambiguous",
            ), body
            assert len(service.registry.tasks()) == before
            assert service.post("/api/evidence-refresh", selector)["disposition"] == "ADMITTED"
            service.drain()
            assert service.post("/api/cro-review", selector)["disposition"] == "ADMITTED"
            service.drain()
            published = service.get("/api/evidence-cro?" + query)
            assert published["state"] == "REVIEW_PUBLISHED", published
            export_selector = {
                **selector,
                "review_publication_hash": published["review_publication_hash"],
            }
            exported = service.get("/api/evidence-cro/export?" + urlencode(export_selector))
            assert exported["review"]["dossier"]["experiment_subject"] == subject
            assert exported["portfolio"]["position"] == selected["position"]
            assert exported["evidence"]["verified_spans"]
            assert "Evidence &amp; CRO" in exported["html"]
            assert (
                service.agent(
                    PortfolioResearchAgentRequest(
                        operation="EVIDENCE_CRO_EXPORT", **export_selector
                    )
                )
                == exported
            )
            tasks = len(service.registry.tasks())
            calls = len(authority.review_actor.observed_deadlines)
            assert service.post("/api/cro-review", selector)["disposition"] == "REUSED_EXACT"
            assert (
                len(service.registry.tasks()),
                len(authority.review_actor.observed_deadlines),
            ) == (tasks, calls)
            output = workspace / selected["document"]["experiment"]["output_workspace"]
            store = PortfolioResearchArtifactStore(output)
            segment = store.load(
                category=SEGMENTS,
                content_hash=selected["receipt"]["segments"][0],
                model=PortfolioReplaySegment,
                identity_field="segment_hash",
            )
            path = next((store.root / LANES).glob(f"{segment.lanes['executed']}.*"))
            saved = path.read_bytes()
            try:
                path.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
                status, _ = service.request("/api/cro-review", method="POST", payload=selector)
                assert status == 400
                assert len(service.registry.tasks()) == tasks
            finally:
                path.write_bytes(saved)
        from tests.researcher_methodology_surface.risk_web_support import (
            exercise_risk_report,
            exercise_risk_retry,
        )

        exercise_risk_report(workspace, selected["task_id"])
        exercise_risk_retry(workspace, selected["task_id"])
        from tests.researcher_methodology_surface.history_web_support import exercise_history

        exercise_history(workspace, selected["task_id"])
        with boot(replace(authority, model_authority_admitted=False, review_actor=None)) as live:
            service = _Service(live, tmp_path)
            tasks = len(service.registry.tasks())
            assert service.get("/api/evidence-cro?" + query)["state"] == "REVIEW_PUBLISHED"
            assert service.get("/api/evidence-cro/export?" + urlencode(export_selector)) == exported
            assert (
                service.post("/api/cro-review", selector)["disposition"]
                == "REFUSED_MODEL_AUTHORITY_NOT_ADMITTED"
            )
            assert len(service.registry.tasks()) == tasks
    finally:
        authority.evidence_runtime.close()


def _seal_foundation(case, live):
    _root, binding, task, decision, _document = case
    preview = _json(
        live,
        "/api/experiments/foundations/preview",
        method="POST",
        payload={
            "task_id": task,
            "curation_receipt_hash": decision,
            "research_input_id": binding.input_id,
            "input_binding_hash": binding.binding_hash,
        },
    )
    assert preview["status"] == "FOUNDATION_PREVIEWED", preview
    sealed = _json(
        live,
        "/api/experiments/foundations/seal",
        method="POST",
        payload={k: v for k, v in preview["next_requests"]["seal"].items() if k != "operation"},
    )
    assert sealed["status"] in {"FOUNDATION_SEALED", "REUSED_EXACT"}, sealed
    assert sealed["admission"] == preview["admission"]
    return sealed


def test_a_foundation_preview_is_sealed_after_a_restart_within_its_hour(alpha_case):
    """regression (V543, the sweep of V534): the Foundation kept its preview in one memory slot,
    so a Host restart, or a second preview, between `foundation preview` and `foundation seal`
    refused the seal `research_foundation.preview_required`, with no words. The preview is kept
    in the plan store: a restarted Host seals it within its hour, and past it the refusal offers
    the preview again, bound to the same study, decision and input."""

    from datetime import UTC, datetime, timedelta

    root, binding, task, decision, _document = alpha_case
    clock = [datetime.now(UTC)]

    def host() -> LocalPortfolioWebSession:
        return LocalPortfolioWebSession(
            workspace=root,
            workspace_manifest=read_research_workspace_manifest(root),
            resolver=_Resolver(_resolved()),
            clock=lambda: clock[0],
        )

    chosen = {
        "task_id": task,
        "curation_receipt_hash": decision,
        "research_input_id": binding.input_id,
        "input_binding_hash": binding.binding_hash,
    }
    with host() as live:
        preview = _json(live, "/api/experiments/foundations/preview", method="POST", payload=chosen)
    assert preview["status"] == "FOUNDATION_PREVIEWED", preview
    seal = {k: v for k, v in preview["next_requests"]["seal"].items() if k != "operation"}
    clock[0] += timedelta(minutes=61)
    with host() as live:
        refused = _json(live, "/api/experiments/foundations/seal", method="POST", payload=seal)
        assert refused["failure_code"] == "research_foundation.preview_required", refused
        assert refused["detail"].startswith("A Foundation is sealed from the exact preview")
        assert refused["next_requests"] == {
            "replan": {"operation": "EXPERIMENT_FOUNDATION_PREVIEW", **chosen}
        }
    clock[0] -= timedelta(minutes=2)
    with host() as live:
        sealed = _json(live, "/api/experiments/foundations/seal", method="POST", payload=seal)
        assert sealed["status"] in {"FOUNDATION_SEALED", "REUSED_EXACT"}, sealed
        assert sealed["admission"] == preview["admission"]


def test_foundation_axis_cannot_be_narrowed_or_reordered_by_a_direct_compiler():
    from unittest.mock import Mock

    from alphalattice.investment.alpha_research.experiments.authoring import AlphaExperimentCompiler

    # Catalogs are irrelevant to this pure axis check; no Program or authority is fabricated.
    compiler = AlphaExperimentCompiler(
        target_recipes=Mock(),
        model_mandate=Mock(),
        model_catalog=Mock(),
        panel_factor_ids=("a", "b"),
        factor_evidence_factor_ids=("a", "b"),
        factor_evidence_checkpoint_hash="a" * 64,
        research_foundation_hash="b" * 64,
    )
    assert compiler._resolve_feature_axis({"ordered_feature_ids": ["a", "b"]}) == ("a", "b")
    for axis in (["a"], ["b", "a"], ["c"]):
        with pytest.raises(ValueError, match=r"research_foundation\.feature_axis_frozen"):
            compiler._resolve_feature_axis({"ordered_feature_ids": axis})


def test_historical_sector_coverage_is_bound_without_moving_legacy_foundations():
    from alphalattice.investment.alpha_research.inputs.development_foundation import (
        AlphaDevelopmentExecutionOutcomeRef,
        AlphaDevelopmentFoundationBinding,
    )
    from alphalattice.kernel.validation.enums import RebalanceFrequency

    digest = "1" * 64
    legacy = AlphaDevelopmentFoundationBinding.create(
        research_cadence=RebalanceFrequency.DAILY,
        feature_panel_snapshot_hash=digest,
        logical_panel_hash=digest,
        logical_semantic_index_hash=digest,
        ordered_factor_ids=("factor",),
        selected_factor_ids=("factor",),
        execution_outcome=AlphaDevelopmentExecutionOutcomeRef(
            research_cadence=RebalanceFrequency.DAILY,
            snapshot_hash=digest,
            manifest_ref="fixture",
            listing_set_hash=digest,
        ),
        factor_development_receipt_hash=digest,
        factor_development_checkpoint_hash=digest,
        factor_development_program_hash=digest,
        factor_development_binding_hash=digest,
        sector_revision=digest,
    )
    payload = legacy.model_dump(mode="json")
    assert "sector_coverage_hash" not in payload
    assert AlphaDevelopmentFoundationBinding.model_validate(payload) == legacy
    expanded = AlphaDevelopmentFoundationBinding.create(
        **{
            k: getattr(legacy, k)
            for k in type(legacy).model_fields
            if k not in {"foundation_hash", "sector_coverage_hash"}
        },
        sector_coverage_hash="2" * 64,
    )
    assert expanded.foundation_hash != legacy.foundation_hash
    assert expanded.sector_revision == legacy.sector_revision
    tampered = {**expanded.model_dump(mode="json"), "sector_coverage_hash": "3" * 64}
    with pytest.raises(ValueError, match="development_foundation_identity_invalid"):
        AlphaDevelopmentFoundationBinding.model_validate(tampered)


def test_alpha_sector_coverage_follows_the_selected_panel_axis(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from alphalattice.control.product_host.research_authoring import authority, execution

    revision, sector_revision = "1" * 64, "2" * 64
    manifest = SimpleNamespace(profile=SimpleNamespace(market_profile_id="fixture"))
    seen = []

    def load(selected):
        seen.append(selected)
        return manifest

    rows = {"earlier-member": "A", "retained-member": "A"}
    state = SimpleNamespace(
        current_sector_state=lambda selected: SimpleNamespace(
            sector_revision=sector_revision, sector_by_listing_id={"retained-member": "A"}
        ),
        sector_classifications=lambda ids: {key: rows[key] for key in ids if key in rows},
    )
    monkeypatch.setattr(
        execution,
        "MarketDataRepository",
        lambda _: SimpleNamespace(
            database=None,
            load_universe_manifest_revision=load,
        ),
    )
    # The reader's body is `authority.panel_sector_labels` since V59; execution delegates to it.
    monkeypatch.setattr(authority, "FeatureStateRepository", lambda *a, **k: state)
    arguments = {
        "workspace": tmp_path,
        "market_profile_id": "fixture",
        "panel_manifest": {
            "safe_summary": {
                "lineage": {
                    "manifest_revision": revision,
                    "sector_revision": sector_revision,
                }
            }
        },
    }
    axis = ("earlier-member", "retained-member")
    labels, first = execution._sector_by_listing_id(**arguments, listing_ids=axis)
    assert labels == rows and first is not None and seen == [revision]
    assert execution._sector_by_listing_id(**arguments, listing_ids=("retained-member",)) == (
        {"retained-member": "A"},
        None,
    )
    rows["earlier-member"] = "B"
    assert execution._sector_by_listing_id(**arguments, listing_ids=axis)[1] != first
    del rows["earlier-member"]
    with pytest.raises(ValueError, match="sector_coverage_incomplete"):
        execution._sector_by_listing_id(**arguments, listing_ids=axis)


def test_foundation_seal_consumption_readback_and_refusals(alpha_case, monkeypatch):
    from copy import deepcopy

    from alphalattice.investment.alpha_research.experiments.development_artifacts import (
        AlphaDevelopmentArtifactStore,
    )
    from alphalattice.investment.alpha_research.experiments.development_contracts import (
        AlphaNumericalDevelopmentScoreChunkRef,
    )
    from alphalattice.investment.alpha_research.experiments.development_execution import (
        AlphaExperimentExecutor,
    )

    root, binding, _task, _decision, _draft = alpha_case
    manifest = (root / "research-workspace.json").read_bytes()
    pointer_paths = tuple(
        p
        for p in root.rglob("*.json")
        if p.name in {"current.json", "pre-research-projection.json"}
    )
    pointers = {p: p.read_bytes() for p in pointer_paths}
    with _session(root) as live:
        count = len(live.session.task_control_registry.tasks())
        with monkeypatch.context() as m:
            m.setattr(
                AlphaExperimentExecutor,
                "execute",
                lambda *a, **kw: pytest.fail("seal must not train"),
            )
            sealed = _seal_foundation(alpha_case, live)
            again = _seal_foundation(alpha_case, live)
            assert again["status"] == "REUSED_EXACT"
        admission = sealed["admission"]
        identity = admission["admission_hash"]
        assert len(live.session.task_control_registry.tasks()) == count
        actor = AgentExecutionBinding(
            profile_id="scripted-curator",
            mode="fixture",
            profile_hash="a" * 64,
            document_hash="b" * 64,
            response_protocol="research-experiment",
            response_protocol_hash="c" * 64,
            concrete_schema_hash="d" * 64,
        )
        bridge = InstalledAgent(live.operations, agent_execution=actor)
        agent = json.loads(
            bridge.invoke(
                PortfolioResearchAgentRequest(
                    operation="EXPERIMENT_FOUNDATION_SEAL", foundation_admission_hash=identity
                )
            )
        )
        assert agent["admission"] == admission and agent["actor"]["actor_kind"] == "INSTALLED_AGENT"
        assert (
            len(
                tuple(
                    (
                        root
                        / "artifacts/factor-research/research-desk/foundation-confirmations"
                        / identity
                    ).glob("*.json")
                )
            )
            == 2
        )
        exported = _json(
            live, f"/api/experiments/foundations/export?foundation_admission_hash={identity}"
        )
        readback = _json(
            live, f"/api/experiments/foundations/readback?foundation_admission_hash={identity}"
        )
        assert _facts(json.loads(exported["json"])) == _facts(readback)
        assert (
            json.loads(
                bridge.invoke(
                    PortfolioResearchAgentRequest(
                        operation="EXPERIMENT_FOUNDATION_READBACK",
                        foundation_admission_hash=identity,
                    )
                )
            )
            == readback
        )
        draft = _json(
            live,
            "/api/experiments/foundations/draft",
            method="POST",
            payload={"foundation_admission_hash": identity},
        )
        assert draft["document"]["alpha"]["foundation_admission_hash"] == identity, draft
        assert draft["plan_request"] == {
            "operation": "EXPERIMENT_PLAN",
            "research_input_id": admission["input_id"],
            "input_binding_hash": admission["input_binding_hash"],
            "factor_task_id": admission["factor_task_id"],
            "curation_receipt_hash": admission["curation_receipt_hash"],
        }
        storage = _json(live, "/api/workspace/storage")
        # The retention owner, not a UI pin, protects both named inputs.
        from alphalattice.control.product_host.storage.input_references import ResearchInputStorage

        _bundles, roots, _evidence = ResearchInputStorage(live.session)._references()
        assert "RESEARCH_FOUNDATION" in roots[binding.binding_hash]
        assert "RESEARCH_FOUNDATION" in roots[admission["factor_input_binding_hash"]]
        assert storage["status"] != "REFUSED", storage
        payload = _alpha_payload(alpha_case)
        reports = []
        programs = []
        for admitted in (False, True):
            request = deepcopy(payload)
            if admitted:
                request["experiment_document"]["alpha"]["foundation_admission_hash"] = identity
                # Foundation selection supplies provenance, not a copied parent Task id.
                request.pop("factor_task_id")
                request.pop("curation_receipt_hash")
            plan = _json(live, "/api/experiments/plan", method="POST", payload=request)
            assert plan["status"] == "PLANNED", plan
            programs.append(plan["program"]["program_hash"])
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            task_id = sent.get("task_id") or sent["publication_task_id"]
            report = _json(live, f"/api/experiments/readback?task_id={task_id}")
            assert report["status"] == "EXPERIMENT_PUBLISHED", report
            store = AlphaDevelopmentArtifactStore(
                root / report["document"]["experiment"]["output_workspace"] / "alpha-development"
            )
            reports.append(
                [
                    store.resolve_numerical_score_chunk(
                        AlphaNumericalDevelopmentScoreChunkRef.model_validate(v["score_chunk"])
                    ).to_pylist()
                    for v in report["fold_results"]
                ]
            )
        assert programs[0] != programs[1]
        assert reports[0] == reports[1]
        assert report["alpha_source"]["foundation_admission_hash"] == identity
        assert (
            _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )["status"]
            == "REUSED_EXACT"
        )
        path = (
            root
            / "artifacts/factor-research/research-desk/foundation-admissions"
            / f"{identity}.json"
        )
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b'"input_id":', b'"corrupted_input_id":', 1))
            assert (
                _json(
                    live,
                    f"/api/experiments/foundations/readback?foundation_admission_hash={identity}",
                )["status"]
                == "REFUSED"
            )
            assert (
                _json(live, "/api/experiments/plan", method="POST", payload=request)["status"]
                == "REFUSED"
            )
        finally:
            path.write_bytes(original)
        # V279 (AS): the listing names each admission's standing. Under the installed curation
        # policy it is CURRENT and offers a draft; a policy no longer installed makes it
        # HISTORICAL by name, offering only its export, and a plan on it is refused.
        from alphalattice.foundation.factor_research.research_loop import decisions

        def listed() -> dict:
            return next(
                v
                for v in _json(live, "/api/experiments/foundations")["foundations"]
                if v["admission"]["admission_hash"] == identity
            )

        current = listed()
        assert current["standing"] == "CURRENT" and "verification" not in current, current
        assert set(current["next_requests"]) == {"alpha-draft", "export"}
        # V633: one bad source stays local to its item; no sealed graph is inferred from a gap.
        from uuid import UUID

        from alphalattice.control.task_control.registry import TaskNotFoundError

        bad_id = "f" * 64
        bad_path = (
            root
            / "artifacts/factor-research/research-desk/foundation-admissions"
            / (bad_id + ".json")
        )
        bad_path.write_text("{", encoding="utf-8")
        try:
            collection = _json(live, "/api/experiments/foundations")
            assert collection["status"] == "AVAILABLE"
            assert len(collection["foundations"]) == 2
            assert listed() == current
            damaged = next(
                item
                for item in collection["foundations"]
                if item.get("foundation_admission_hash") == bad_id
            )
            assert damaged["status"] == "REFUSED"
            assert "admission" not in damaged and "verification" not in damaged
            assert damaged["next_requests"] == {"inputs": {"operation": "RESEARCH_INPUTS"}}
            original_read = Path.read_text

            def read_with_gap(path, *args, **kwargs):
                if path == bad_path:
                    raise FileNotFoundError(path)
                return original_read(path, *args, **kwargs)

            with monkeypatch.context() as m:
                m.setattr(Path, "read_text", read_with_gap)
                unavailable = _json(live, "/api/experiments/foundations")
                assert unavailable["status"] == "AVAILABLE" and listed() == current
                stopped = next(
                    row
                    for row in unavailable["foundations"]
                    if row.get("foundation_admission_hash") == bad_id
                )
                assert (
                    stopped["failure_code"] == "research_foundation.admission_artifact_unavailable"
                )
                assert stopped["next_requests"] == damaged["next_requests"]
            registry = live.session.task_control_registry
            original_task = registry.task
            missing_id = UUID(admission["factor_task_id"])

            def task_with_gap(task_id):
                if task_id == missing_id:
                    raise TaskNotFoundError(task_id)
                return original_task(task_id)

            with monkeypatch.context() as m:
                m.setattr(registry, "task", task_with_gap)
                collection = _json(live, "/api/experiments/foundations")
                assert collection["status"] == "AVAILABLE"
                missing = next(
                    item
                    for item in collection["foundations"]
                    if item.get("foundation_admission_hash") == identity
                )
                assert missing["status"] == "REFUSED"
                assert missing["failure_code"] == "task_control.task_not_found"
                assert missing["missing_factor_task_id"] == str(missing_id)
                assert str(missing_id) in missing["detail"]
                assert "verification" not in missing and "standing" not in missing
                assert missing["next_requests"]["factor"] == {
                    "operation": "EXPERIMENT_CONTROLS",
                    "experiment_kind": "factor.screening-development",
                    "research_input_id": admission["input_id"],
                    "input_binding_hash": admission["input_binding_hash"],
                }
                for door in ("readback", "export"):
                    assert (
                        _json(
                            live,
                            f"/api/experiments/foundations/{door}?foundation_admission_hash={identity}",
                        )
                        == missing
                    )
                before_controls = len(registry.tasks())
                controls = live.operations.execute(Request(**missing["next_requests"]["factor"]))
                assert controls["status"] != "REFUSED"
                assert len(registry.tasks()) == before_controls
        finally:
            bad_path.unlink()
        not_installed = (
            "factor_research.evidence_curation_not_host_admitted:"
            "factor_research.review_decision_policy_not_installed"
        )
        with monkeypatch.context() as m:
            m.setattr(decisions, "factor_research_decision_policy_hash", lambda *_a, **_k: "f" * 64)
            historical = listed()
            assert historical["standing"] == "HISTORICAL", historical
            assert historical["standing_code"] == not_installed
            assert historical["verification"] == "RECORDED_GRAPH_NOT_CURRENT_POLICY_ADMISSION"
            assert set(historical["next_requests"]) == {"export"}
            refused = _json(live, "/api/experiments/plan", method="POST", payload=request)
            assert refused["status"] == "REFUSED", refused
            assert refused["failure_code"] == not_installed
        wrong = deepcopy(request)
        wrong["experiment_document"]["alpha"]["ordered_feature_ids"] = ["unknown-factor"]
        assert (
            _json(live, "/api/experiments/plan", method="POST", payload=wrong)["status"]
            == "REFUSED"
        )
    with _session(root) as restarted:
        assert (
            _json(
                restarted,
                f"/api/experiments/foundations/export?foundation_admission_hash={identity}",
            )
            == exported
        )
        again = _json(restarted, f"/api/experiments/readback?task_id={task_id}")
        assert _facts(again) == _facts(report)
        assert any(
            v["admission"]["admission_hash"] == identity
            for v in _json(restarted, "/api/experiments/foundations")["foundations"]
        )
    assert (root / "research-workspace.json").read_bytes() == manifest
    assert all(p.read_bytes() == content for p, content in pointers.items())


def test_experiment_collection_keeps_readable_tasks_beside_an_invalid_saved_plan(alpha_case):
    from uuid import UUID

    from alphalattice.control.task_control.contracts import (
        ResearchGoal,
        ResearchPlan,
        TaskInputEnvelope,
    )

    root, _binding, factor_task, _decision, _draft = alpha_case
    with _session(root) as live:
        registry = live.session.task_control_registry
        original = registry.task(UUID(factor_task))
        envelope = TaskInputEnvelope.create(
            task_kind=original.task_kind,
            input_schema_id=original.input.input_schema_id,
            payload={"actor": original.input.payload["actor"]},
        )
        goal = ResearchGoal.create(
            goal_kind=original.goal.goal_kind,
            input_hash=envelope.input_hash,
            deliverable_kind=original.goal.deliverable_kind,
            summary=original.goal.summary,
            attributes=original.goal.attributes,
        )
        plan = ResearchPlan.create(
            goal_hash=goal.goal_hash,
            workflow_definition_hash=original.plan.workflow_definition_hash,
            verifier_catalog_hash=original.plan.verifier_catalog_hash,
            work_items=original.plan.work_items,
        )
        admitted = registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=datetime.now(UTC)
        )
        answer = live.operations.execute(Request(operation="EXPERIMENTS"))
        assert answer["status"] == "AVAILABLE"
        assert any(row["task_id"] == factor_task for row in answer["experiments"])
        assert len(answer["refusals"]) == 1
        refused = answer["refusals"][0]
        assert refused["task_id"] == str(admitted.record.task_id)
        assert refused["failure_code"].startswith("research_experiment.refused:")
        assert refused["next_requests"]["task"] == {
            "operation": "TASK_RECOVERY",
            "task_id": str(admitted.record.task_id),
        }


def test_pending_task_damage_keeps_curation_for_a_readable_factor_study(inputs, published):
    """V661: unreadable peers do not hide curation owned by a readable Factor study."""
    from uuid import UUID

    import duckdb

    factor_task, _report, _export = published
    with _session(inputs) as live:
        before = _json(live, "/api/decisions")
        (curation,) = [
            row
            for row in before["decisions"]
            if row["kind"] == "CURATION" and row["task_id"] == factor_task
        ]
        registry = live.session.task_control_registry
        factor = registry.task(UUID(factor_task))
        peer = registry.admit(
            input_envelope=factor.input,
            goal=factor.goal,
            plan=factor.plan,
            observed_at=datetime.now(UTC),
        ).record
        peer, _command = registry.request_cancel(
            task_id=peer.task_id,
            expected_task_hash=peer.record_hash,
            observed_at=datetime.now(UTC),
        )
        with duckdb.connect(str(registry.database_path)) as connection:
            connection.execute(
                "UPDATE workspace_task SET record_json = '{' WHERE task_id = ?",
                [str(peer.task_id)],
            )
        answer = _json(live, "/api/decisions")
        assert [row for row in answer["decisions"] if row["kind"] == "CURATION"] == [curation]
        (refusal,) = answer["refusals"]
        assert refusal["task_id"] == str(peer.task_id)
        assert refusal["failure_code"] == "task_control.database_authority_unreadable"
        offered = curation["next_requests"]["curation"]
        assert offered == {"operation": "EXPERIMENT_CURATION", "task_id": factor_task}
        choices = _json(live, f"/api/experiments/curation?task_id={offered['task_id']}")
        assert choices["status"] == "AVAILABLE"
        assert choices["receipt_hash"] == _report["receipt"]["receipt_hash"]
        assert choices["choices"]
        assert choices["next_requests"]["curate"] == {
            "operation": "EXPERIMENT_CURATE",
            "task_id": factor_task,
            "experiment_curation": {"expected_receipt_hash": choices["receipt_hash"]},
        }


def test_pending_task_damage_withholds_absence_based_preview_and_promotion_offers(
    sampled_alpha_case,
):
    """V661: a real completed exploration and runnable preview need complete Task authority."""
    from uuid import UUID

    import duckdb

    alpha_case, sampled = sampled_alpha_case
    root, _binding, factor_task, _decision, _draft = alpha_case
    sent = {"task_id": sampled["task_id"]}
    with _session(root) as live:
        preview = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload=_alpha_payload(alpha_case, parameters={"family": "ridge", "alpha": 2.0}),
        )
        assert preview["status"] == "PLANNED", preview
        before = _json(live, "/api/decisions")
        assert any(
            row["kind"] == "PROMOTION" and row["task_id"] == sent["task_id"]
            for row in before["decisions"]
        ), before
        assert any(
            row["kind"] == "PLAN_PREVIEW" and row["plan_hash"] == preview["plan_hash"]
            for row in before["decisions"]
        ), before

        # Task Control admits and cancels a real contract; only its saved canonical bytes
        # are damaged. No queued-head corruption or substituted owner answer is involved.
        registry = live.session.task_control_registry
        factor = registry.task(UUID(factor_task))
        peer = registry.admit(
            input_envelope=factor.input,
            goal=factor.goal,
            plan=factor.plan,
            observed_at=datetime.now(UTC),
        ).record
        peer, _command = registry.request_cancel(
            task_id=peer.task_id,
            expected_task_hash=peer.record_hash,
            observed_at=datetime.now(UTC),
        )
        with duckdb.connect(str(registry.database_path)) as connection:
            connection.execute(
                "UPDATE workspace_task SET record_json = '{' WHERE task_id = ?",
                [str(peer.task_id)],
            )
            stored_before = connection.execute(
                "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
            ).fetchall()

        answer = _json(live, "/api/decisions")
        assert not {"PROMOTION", "PLAN_PREVIEW"}.intersection(
            row["kind"] for row in answer["decisions"]
        ), answer
        (refusal,) = answer["refusals"]
        assert refusal["task_id"] == str(peer.task_id)
        assert refusal["failure_code"] == "task_control.database_authority_unreadable"
        assert refusal["next_requests"] == {
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
        }
        assert any(
            row["kind"] == "TASK_RECORD_UNREADABLE" and row["task_id"] == str(peer.task_id)
            for row in answer["decisions"]
        )
        studies = _json(live, "/api/experiments")
        assert {factor_task, sent["task_id"]}.issubset(
            row["task_id"] for row in studies["experiments"]
        )
        with duckdb.connect(str(registry.database_path), read_only=True) as connection:
            assert (
                connection.execute(
                    "SELECT task_id, record_json FROM workspace_task ORDER BY admission_sequence"
                ).fetchall()
                == stored_before
            )


@pytest.mark.parametrize(
    ("handle", "parameters"),
    [
        ("capability-1", {"family": "ridge", "alpha": 1.0}),
        ("capability-1", {"family": "lasso", "alpha_max_multiplier": 0.1}),
        ("capability-1", {"family": "elastic_net", "alpha_max_multiplier": 0.1, "l1_ratio": 0.5}),
        # The main product's model through the same declaration, operation and
        # readback: the second installed capability, at the installed G6 point.
        ("capability-2", _lightgbm_parameters()),
    ],
)
def test_alpha_declared_model_runs_reopens_and_reuses_existing_owners(
    alpha_case, handle, parameters
):
    root, binding, factor_task, decision, _original = alpha_case
    manifest = (root / "research-workspace.json").read_bytes()
    payload = _alpha_payload(alpha_case, parameters, handle)
    expected_adapter = {
        "capability-1": "regularized_linear",
        "capability-2": "dynamic_panel_lightgbm",
    }[handle]
    with _session(root) as live:
        count = len(live.session.task_control_registry.tasks())
        # The controls a page reads (the handoff preview's authoring options)
        # name both installed models and show this capability's parameters;
        # the PLAN preview names the one this declaration resolved to.
        handoff = _json(
            live,
            "/api/experiments/handoff",
            method="POST",
            payload={
                "task_id": factor_task,
                "curation_receipt_hash": decision,
                "research_input_id": binding.input_id,
                "input_binding_hash": binding.binding_hash,
                "experiment_document": payload["experiment_document"],
            },
        )
        assert handoff["status"] == "INPUT_COMPILER_PREFLIGHT_PASSED", handoff
        controls = handoff["authoring_options"]["controls"]
        capability_control = next(
            c for c in controls if c["path"] == ["alpha", "model_capability_handle"]
        )
        assert [o["label"] for o in capability_control["options"]] == [
            "regularized_linear",
            "dynamic_panel_lightgbm",
        ]
        document = payload["experiment_document"]

        def _stated(path):
            value = document
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            return value

        shown = {
            c["path"][-1]
            for c in controls
            if c["path"][:2] == ["alpha", "model_parameters"]
            and all(_stated(cond["path"]) in cond["values"] for cond in c.get("when", []))
        }
        assert shown == set(parameters), (shown, set(parameters))
        plan = _json(live, "/api/experiments/plan", method="POST", payload=payload)
        assert plan["status"] == "PLANNED", plan
        assert plan["numerical_call_count"] == 0
        assert plan["execution_preview"]["model_adapter_id"] == expected_adapter
        assert plan["execution_preview"]["fit_protocol"] == "DIRECT_FIT"
        assert plan["execution_preview"]["model_parameters"] == parameters
        assert len(live.session.task_control_registry.tasks()) == count
        bridge = InstalledAgent(live.operations)
        assert (
            json.loads(
                bridge.invoke(
                    PortfolioResearchAgentRequest(
                        operation="EXPERIMENT_PLAN",
                        **payload,
                    )
                )
            )
            == plan
        )
        sent = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        task_id = sent["task_id"] or sent.get("publication_task_id")
        assert task_id, sent
        if sent["task_id"] is None:
            # Another consumer of the shared fixture may already have executed
            # this exact model. Reuse must not invent a duplicate Task or a fit.
            assert sent["status"] == "REUSED_EXACT" and sent["numerical_call_count"] == 0
        live.dispatcher.drain_for_tests()
        body = _json(live, f"/api/experiments/readback?task_id={task_id}")
        assert body["status"] == "EXPERIMENT_PUBLISHED", body
        assert body["document"]["alpha"]["model_parameters"] == parameters
        assert body["alpha_source"]["factor_task_id"] == factor_task
        assert body["alpha_source"]["curation_receipt_hash"] == decision
        assert body["result"]["fit_call_count"] == plan["execution_preview"]["fold_count"]
        assert len(body["fold_results"]) == plan["execution_preview"]["fold_count"]
        assert body["evidence"]["numerical_call_count"] == 0  # This is verified readback.
        assert (
            sum(
                body["result"][key]
                for key in (
                    "fit_call_count",
                    "predict_call_count",
                    "metric_call_count",
                )
            )
            == plan["execution_preview"]["expected_numerical_calls"]
        )
        exported = _json(live, f"/api/experiments/export?task_id={task_id}")
        assert "Alpha development evidence" in exported["html"]
        assert _facts(json.loads(exported["json"])) == _facts(body)
        if parameters.get("family", "lightgbm") in {"ridge", "lightgbm"}:
            child = body["receipt"]["child_lineage"][0]["numerical_result_hash"]
            folder = root / body["document"]["experiment"]["output_workspace"]
            paths = list(folder.rglob(f"{child}.json"))
            assert len(paths) == 1
            original = paths[0].read_bytes()
            try:
                paths[0].write_bytes(b"{}")
                refused = _json(live, f"/api/experiments/readback?task_id={task_id}")
                assert refused["status"] == "REFUSED", refused
            finally:
                paths[0].write_bytes(original)
            again = _json(live, f"/api/experiments/readback?task_id={task_id}")
            assert _facts(again) == _facts(body)
        reused = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        assert reused["status"] == "REUSED_EXACT" and reused["task_id"] is None
        assert len(live.session.task_control_registry.tasks()) == count + bool(sent["task_id"])
        assert (root / "research-workspace.json").read_bytes() == manifest
        draft = _json(
            live,
            "/api/experiments/draft",
            method="POST",
            payload={
                "task_id": task_id,
                "input_binding_hash": binding.binding_hash,
            },
        )
        assert draft["status"] == "DRAFT_READY", draft
        assert draft["document"] == body["document"]
        copied = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={
                **payload,
                "experiment_document": draft["document"],
                "origin_task_id": task_id,
            },
        )
        assert copied["status"] == "PLANNED", copied
        assert copied["program"] == plan["program"]
        assert (
            _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={
                    "experiment_plan_hash": copied["plan_hash"],
                },
            )["status"]
            == "REUSED_EXACT"
        )
    with _session(root) as restarted:
        again = _json(restarted, f"/api/experiments/readback?task_id={task_id}")
        assert _facts(again) == _facts(body)
        assert _json(restarted, f"/api/experiments/export?task_id={task_id}") == exported
        assert (
            _json(
                restarted, "/api/experiments/replay", method="POST", payload={"task_id": task_id}
            )["numerical_call_count"]
            == 0
        )


def test_alpha_refuses_invalid_authority_and_work_before_admitting_a_task(alpha_case, monkeypatch):
    from copy import deepcopy

    from alphalattice.investment.alpha_research.experiments.development_execution import (
        AlphaExperimentExecutor,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("preview/refusal must not enter execution")

    monkeypatch.setattr(AlphaExperimentExecutor, "execute", forbidden)
    original = _alpha_payload(alpha_case)
    invalid = []
    for field in ("factor_task_id", "curation_receipt_hash"):
        payload = deepcopy(original)
        del payload[field]
        invalid.append(payload)
    payload = deepcopy(original)
    payload["curation_receipt_hash"] = "a" * 64
    invalid.append(payload)
    for field, value in (
        ("ordered_feature_ids", ["NOT_IN_THE_FACTOR_DECISION"]),
        ("model_parameters", {"family": "ridge", "alpha": -1}),
        ("model_parameters", {"family": "not_installed"}),
    ):
        payload = deepcopy(original)
        payload["experiment_document"]["alpha"][field] = value
        invalid.append(payload)
    # The LightGBM capability refuses at PLAN too: linear parameters routed to
    # it, a quoted count, a value outside the installed bounds, a policy
    # whose tuning partition this development split does not state, and a
    # policy written as a YAML list or mapping (a typed refusal, not a 500).
    lightgbm = _lightgbm_parameters()
    for value in (
        {"family": "ridge", "alpha": 1.0},
        {**lightgbm, "num_leaves": "31"},
        {**lightgbm, "learning_rate": 0.9},
        {**lightgbm, "training_policy": "L2_EARLY_STOPPING", "fixed_iterations": None},
        {**lightgbm, "training_policy": ["FIXED_ITERATION"]},
        {**lightgbm, "training_policy": {"policy": "FIXED_ITERATION"}},
    ):
        payload = deepcopy(original)
        payload["experiment_document"]["alpha"]["model_capability_handle"] = "capability-2"
        payload["experiment_document"]["alpha"]["model_parameters"] = value
        invalid.append(payload)
    payload = deepcopy(original)
    payload["experiment_document"]["experiment"]["budget"]["maximum_numerical_calls"] = 1
    invalid.append(payload)
    with _session(alpha_case[0]) as live:
        count = len(live.session.task_control_registry.tasks())
        results = []
        for payload in invalid:
            result = _json(live, "/api/experiments/plan", method="POST", payload=payload)
            assert result["status"] == "REFUSED", result
            assert len(live.session.task_control_registry.tasks()) == count
            results.append(result)
        assert "work_budget_exceeded" in result["failure_code"]
        # The two policy-kind declarations are refused by the model admission,
        # as a named code the page can show, not as a server error.
        for result in results[-3:-1]:
            assert "authoring_model_recipe_not_admissible" in result["failure_code"], result


def _inline(target, arguments, *, cancelled=None):  # type: ignore[no-untyped-def]
    """The worker's call run in this process, so a fold-boundary hook patched here runs (W10).

    The Host's behaviour these tests check (reuse in flight, cancellation, recovery) does not
    depend on where the fit runs; the worker's own call, cancellation relay and death are
    tested in `tests/workspace_task_runner/test_task_child.py`.
    """
    import importlib

    module_name, _, name = target.partition(":")
    function = getattr(importlib.import_module(module_name), name)
    return function(**arguments, cancelled=cancelled or (lambda: False))


def test_alpha_cancel_keeps_complete_children_and_explicit_retry_reuses_them(
    alpha_case, monkeypatch
):
    from alphalattice.control.product_host.research_authoring import factor_handoff
    from alphalattice.investment.alpha_research.experiments.development_execution import (
        _FoldCancellation,
    )

    monkeypatch.setattr(factor_handoff, "run_in_child", _inline)
    observed = []
    original = _FoldCancellation.record_candidate_fold
    with _session(alpha_case[0]) as live:

        def cancel(self, **facts):
            observed.append(facts)
            if len(observed) == 1:
                current = next(
                    t
                    for t in live.session.task_control_registry.tasks()
                    if t.lifecycle.value == "RUNNING"
                )
                reused = _json(
                    live,
                    "/api/experiments/run",
                    method="POST",
                    payload={
                        "experiment_plan_hash": plan["plan_hash"],
                    },
                )
                assert reused["status"] == "REUSED_IN_FLIGHT", reused
                assert reused["task_id"] == str(current.task_id)
                storage = _json(live, "/api/workspace/storage")
                refs = current.input.payload["plan"]
                bound_inputs = {
                    refs["binding"]["binding_hash"],
                    refs["alpha_source"]["factor_binding"]["binding_hash"],
                }
                rows = {r["binding_hash"]: r for r in storage["inputs"]}
                assert len(bound_inputs) == 2
                assert all("IN_FLIGHT_RECOVERY" in rows[h]["roots"] for h in bound_inputs)
                assert _json(
                    live,
                    "/api/cancel",
                    method="POST",
                    payload={
                        "task_id": str(current.task_id),
                    },
                )
            original(self, **facts)

        monkeypatch.setattr(_FoldCancellation, "record_candidate_fold", cancel)
        plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload=_alpha_payload(alpha_case, {"family": "ridge", "alpha": 2.0}),
        )
        assert plan["status"] == "PLANNED", plan
        sent = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={
                "experiment_plan_hash": plan["plan_hash"],
            },
        )
        live.dispatcher.drain_for_tests()
        stopped = _json(live, f"/api/experiments/readback?task_id={sent['task_id']}")
        assert stopped["status"] == "CANCELLED", stopped
        assert len(observed) == 1 and observed[0]["fit_calls"] == 1
        monkeypatch.setattr(_FoldCancellation, "record_candidate_fold", original)
        retry = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={
                "experiment_plan_hash": plan["plan_hash"],
            },
        )
        assert retry["task_id"] and retry["task_id"] != sent["task_id"], retry
        live.dispatcher.drain_for_tests()
        body = _json(live, f"/api/experiments/readback?task_id={retry['task_id']}")
        assert body["status"] == "EXPERIMENT_PUBLISHED", body
        assert body["result"]["fit_call_count"] == plan["execution_preview"]["fold_count"] - 1


_STOPPED_ALPHA = r"""
import json, sys, threading
from pathlib import Path
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.research_experiments import (
    ResearchExperimentApplication, STAGES,
)
from alphalattice.investment.alpha_research.experiments.development_execution import (
    _FoldCancellation,
)
from tests.portfolio_strategy_lab.local_web_support import _Resolver, _resolved, _json
import alphalattice.control.product_host.research_authoring.factor_handoff as handoff
from tests.researcher_methodology_surface.test_local_web_factor_experiments import _inline
# The fit runs in this process, so the fold boundary below pauses it where the test kills it.
handoff.run_in_child = _inline
root, payload, marker = map(Path, sys.argv[1:4])
boundary = sys.argv[4]
if sys.argv[5] == 'prose':
    # The rule before SH: the Task's contract hash bound its schema's prose too.
    import alphalattice.control.product_host.composition.research_experiments as composed
    composed.schema_structure = lambda model: model.model_json_schema()
def pause(task_id):
    marker.write_text(json.dumps({'task_id': str(task_id)}), encoding='utf-8')
    print('TEST_BOUNDARY', flush=True)
    threading.Event().wait()
original = ResearchExperimentApplication.execute_stage
def stage(self, **kwargs):
    result = original(self, **kwargs)
    if (boundary == 'complete_evidence' and kwargs['work_item'].stage_id == STAGES[0]
            and result.failure_code is None):
        pause(kwargs['task'].task_id)
    return result
ResearchExperimentApplication.execute_stage = stage
original_fold = _FoldCancellation.record_candidate_fold
def fold(self, **facts):
    original_fold(self, **facts)
    if boundary == 'completed_fold' and facts['fold_index'] == 1:
        task = next(t for t in live.session.task_control_registry.tasks()
                    if t.lifecycle.value == 'RUNNING')
        pause(task.task_id)
_FoldCancellation.record_candidate_fold = fold
with LocalPortfolioWebSession(workspace=root,
        workspace_manifest=read_research_workspace_manifest(root),
        resolver=_Resolver(_resolved())) as live:
    plan = _json(live, '/api/experiments/plan', method='POST',
                 payload=json.loads(payload.read_text(encoding='utf-8')))
    assert plan['status'] == 'PLANNED', plan
    sent = _json(live, '/api/experiments/run', method='POST',
                 payload={'experiment_plan_hash': plan['plan_hash']})
    assert sent['task_id'], sent
    threading.Event().wait()
"""


def _stop_alpha(root: Path, tmp_path: Path, request: dict, boundary: str, rule: str) -> str:
    """Run one Alpha study in a process killed at ``boundary``; the stopped Task's id.

    ``rule`` is the schema hash rule its Task's compatibility is computed under: the
    installed one (``structure``) or the one before SH (``prose``).
    """
    marker = tmp_path / "boundary.json"
    payload = tmp_path / "request.json"
    payload.write_text(json.dumps(request), encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _STOPPED_ALPHA,
            str(root),
            str(payload),
            str(marker),
            boundary,
            rule,
        ],
        cwd=Path(__file__).resolve().parents[2],
        # The interrupted run computes on one BLAS thread and the recovery on two: a sealed
        # fold must be reused as sealed, whatever thread count or host resumes it (OpenBLAS
        # splits a long dot product by thread count, so a recomputed metric differs).
        env={
            **os.environ,
            "PYTHONPATH": "src",
            "ALPHALATTICE_NETWORK_DISABLED": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        if not _wait_for_boundary(process, marker, 180):
            process.terminate()
            pytest.fail(f"Alpha did not reach {boundary}: {process.communicate(timeout=15)!r}")
        task_id = json.loads(marker.read_text(encoding="utf-8"))["task_id"]
        process.kill()
        process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=15)
    return str(task_id)


@pytest.mark.parametrize("boundary", ["completed_fold", "complete_evidence"])
def test_alpha_process_recovery_keeps_task_and_reuses_sealed_work(
    alpha_case, tmp_path, monkeypatch, boundary
):
    root = alpha_case[0]
    admission_hash = None
    if boundary == "complete_evidence":
        with _session(root) as live:
            admission_hash = _seal_foundation(alpha_case, live)["admission"]["admission_hash"]
    request = _alpha_payload(
        alpha_case, {"family": "ridge", "alpha": 3.0 if boundary == "completed_fold" else 4.0}
    )
    if admission_hash is not None:
        request["experiment_document"]["alpha"]["foundation_admission_hash"] = admission_hash
    task_id = _stop_alpha(root, tmp_path, request, boundary, "structure")
    from alphalattice.investment.alpha_research.experiments.development_execution import (
        AlphaExperimentExecutor,
    )

    if boundary == "complete_evidence":

        def forbidden(*args, **kwargs):
            raise AssertionError("complete evidence must not train again")

        monkeypatch.setattr(AlphaExperimentExecutor, "execute", forbidden)
    with threadpool_limits(limits=2, user_api="blas"), _session(root) as live:
        assert task_id in {str(t) for t in live.resumed_task_ids}
        live.dispatcher.drain_for_tests()
        body = _json(live, f"/api/experiments/readback?task_id={task_id}")
        assert body["status"] == "EXPERIMENT_PUBLISHED", body
        folds = body["execution_preview"]["fold_count"]
        assert body["result"]["fit_call_count"] == folds - (
            2 if boundary == "completed_fold" else 0
        )
        assert (
            sum(r["task_id"] == task_id for r in _json(live, "/api/experiments")["experiments"])
            == 1
        )


def test_a_study_stopped_under_the_old_schema_rule_is_refused_by_name_and_planned_again(
    alpha_case, tmp_path
):
    """requirement (SH, a waiting Task's recovery): a study stopped while its Task's contract
    hash still bound the schema's prose is refused at resume by name, never resumed under a
    contract it did not start with and never left stuck: its recovery offers the cancel, and
    the same request planned again runs as a new Task that keeps the stopped one's sealed folds."""

    root = alpha_case[0]
    request = _alpha_payload(alpha_case, {"family": "ridge", "alpha": 6.0})
    task_id = _stop_alpha(root, tmp_path, request, "completed_fold", "prose")
    with _session(root) as live:
        assert task_id not in {str(t) for t in live.resumed_task_ids}
        refused = _json(live, "/api/recover", method="POST", payload={"task_id": task_id})
        assert refused["status"] == "REFUSED", refused
        assert refused["disposition"] == "REFUSED_CHANGED_SINCE_ADMISSION"
        assert refused["failure_code"] == "task_control.execution_compatibility_changed"
        assert refused["lifecycle"] == "RECOVERY_REQUIRED"
        cancel = refused["next_requests"]["cancel"]
        assert cancel["operation"] == "CANCEL" and cancel["task_id"] == task_id
        _json(
            live,
            "/api/cancel",
            method="POST",
            payload={key: value for key, value in cancel.items() if key != "operation"},
        )
        live.dispatcher.drain_for_tests()
        plan = _json(live, "/api/experiments/plan", method="POST", payload=request)
        assert plan["status"] == "PLANNED", plan
        again = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        assert again["task_id"] and again["task_id"] != task_id, again
        live.dispatcher.drain_for_tests()
        body = _json(live, f"/api/experiments/readback?task_id={again['task_id']}")
        assert body["status"] == "EXPERIMENT_PUBLISHED", body
        folds = body["execution_preview"]["fold_count"]
        assert body["result"]["fit_call_count"] == folds - 2


def test_binding_refuses_a_corrupt_copy_a_changed_source_and_a_tampered_pool(
    inputs: Path, tmp_path: Path, monkeypatch
):
    """regression: each object is read once per boundary, and every boundary still refuses.

    The staging loop no longer re-reads a linked pool object or the source
    per file; what it still reads is a copied file (a corrupt copy is
    refused), the whole source at the end (a source that changed during the
    binding is refused, and nothing is published), and a pool object whose
    bytes no longer match its name (refused before it is linked).
    """

    from alphalattice.control.product_host.research_authoring import factor_inputs as owner

    original = read_research_workspace_manifest(inputs).experiment_inputs[0]
    bundle = read_factor_bundle(inputs, original.binding_hash)
    source = inputs / "research-inputs" / original.binding_hash / "source"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    publish_research_workspace_manifest(workspace, _manifest("binding-boundaries"))
    bind = dict(
        workspace=workspace,
        source=source,
        panel_snapshot_hash=bundle.panel_snapshot_hash,
        outcome_snapshot_hash=bundle.outcome_snapshot_hash,
        include_alpha_handoff=True,
    )
    copy2 = owner.shutil.copy2

    def corrupt_copy(src, dst, *args, **kwargs):
        copy2(src, dst, *args, **kwargs)
        if Path(dst).suffix == ".json" and "feature-panel/manifests" in Path(dst).as_posix():
            Path(dst).write_bytes(b"{}")

    monkeypatch.setattr(owner.shutil, "copy2", corrupt_copy)
    with pytest.raises(AuthoringError, match="source_changed_during_binding"):
        bind_factor_inputs(**bind)
    monkeypatch.setattr(owner.shutil, "copy2", copy2)
    assert not any(
        p.is_dir() and len(p.name) == 64 for p in (workspace / "research-inputs").iterdir()
    )
    # The source moves after the copies were taken: the final pass refuses.
    lifecycle = next(p for p in source.rglob("*.json") if p.parent.name == "feature-panel")
    copy_input = owner._copy_input
    original_bytes = lifecycle.read_bytes()

    def move_source_after_copy(workspace_, src, target, expected, **kwargs):
        verified = copy_input(workspace_, src, target, expected, **kwargs)
        if src == lifecycle:
            lifecycle.write_bytes(original_bytes + b"\n")
        return verified

    monkeypatch.setattr(owner, "_copy_input", move_source_after_copy)
    try:
        with pytest.raises(AuthoringError, match="source_changed_during_binding"):
            bind_factor_inputs(**bind)
    finally:
        lifecycle.write_bytes(original_bytes)
        monkeypatch.setattr(owner, "_copy_input", copy_input)
    assert not any(
        p.is_dir() and len(p.name) == 64 for p in (workspace / "research-inputs").iterdir()
    )
    # A clean binding publishes; a pool object whose bytes no longer match
    # its content-addressed name is refused by a later binding that would
    # link it, before anything is linked.
    binding = bind_factor_inputs(**bind)
    enhanced = read_factor_bundle(workspace, binding.binding_hash)
    pooled = next(
        p for p in (workspace / "research-inputs" / "parquet").iterdir() if p.suffix == ".parquet"
    )
    poisoned = pooled.read_bytes()
    pooled.write_bytes(poisoned[:-1] + bytes([poisoned[-1] ^ 1]))
    try:
        with pytest.raises(AuthoringError, match="input_file_tampered"):
            read_factor_bundle(workspace, binding.binding_hash)
        other = tmp_path / "other"
        other.mkdir()
        publish_research_workspace_manifest(other, _manifest("binding-boundaries-other"))
        shutil.copytree(
            workspace / "research-inputs" / "parquet", other / "research-inputs" / "parquet"
        )
        with pytest.raises(AuthoringError, match="input_file_tampered"):
            bind_factor_inputs(**{**bind, "workspace": other})
    finally:
        pooled.write_bytes(poisoned)
    assert read_factor_bundle(workspace, binding.binding_hash) == enhanced


def test_handoff_binding_adds_logical_closure_without_mutating_factor_inputs(
    inputs: Path, tmp_path: Path
):
    from alphalattice.control.product_host.research_authoring.factor_inputs import (
        compatible_factor_inputs,
        logical_panel_owner,
    )

    original = read_research_workspace_manifest(inputs).experiment_inputs[0]
    bundle = read_factor_bundle(inputs, original.binding_hash)
    source = inputs / "research-inputs" / original.binding_hash / "source"
    publish_research_workspace_manifest(tmp_path, _manifest("handoff-inputs"))
    binding = bind_factor_inputs(
        workspace=tmp_path,
        source=source,
        panel_snapshot_hash=bundle.panel_snapshot_hash,
        outcome_snapshot_hash=bundle.outcome_snapshot_hash,
        include_alpha_handoff=True,
    )
    enhanced = read_factor_bundle(tmp_path, binding.binding_hash)
    assert compatible_factor_inputs(bundle, enhanced)
    assert binding.binding_hash != original.binding_hash
    target = tmp_path / "research-inputs" / binding.binding_hash / "source"
    logical_panel_owner(target).verify_snapshot(bundle.panel_snapshot_hash)
    assert not tuple(target.rglob("current.json"))
    assert (
        bind_factor_inputs(
            workspace=tmp_path,
            source=source,
            panel_snapshot_hash=bundle.panel_snapshot_hash,
            outcome_snapshot_hash=bundle.outcome_snapshot_hash,
            include_alpha_handoff=True,
        )
        == binding
    )
    assert read_factor_bundle(inputs, original.binding_hash) == bundle


def test_bound_input_runs_reopens_and_replays_the_real_factor_owner(
    inputs: Path, published
) -> None:
    task_id, report, exported = published
    # The declaration's selection leads; the context it was screened in is its denominator
    # (V252).
    assert report["declared_selection"] == {
        "selected_factor_ids": report["receipt"]["input_binding"]["selected_factor_ids"],
        "screened_factor_count": len(report["result"]["program"]["factor_ids"]),
    }
    assert list(report).index("declared_selection") < list(report).index("result")
    with _session(inputs) as reopened:
        again = _json(reopened, f"/api/experiments/readback?task_id={task_id}")
        assert _facts(again) == _facts(report)
        assert _json(reopened, f"/api/experiments/export?task_id={task_id}") == exported
        replay = _json(
            reopened, "/api/experiments/replay", method="POST", payload={"task_id": task_id}
        )
        assert replay["status"] == "REUSED_EXACT" and replay["numerical_call_count"] == 0
        assert (
            sum(
                row["task_id"] == task_id
                for row in _json(reopened, "/api/experiments")["experiments"]
            )
            == 1
        )


def test_controls_and_a_missing_prerequisite_name_the_flows_way_on(inputs: Path, published) -> None:
    """requirement (V367): each kind's controls, and a refusal that means a prerequisite result
    is missing, name the flow's prerequisites on the input: the completed studies it holds,
    what is missing and the requests allowed next."""

    from uuid import UUID

    task_id, _report, _exported = published
    root = inputs
    with _session(root) as live:
        controls = _json(live, "/api/experiments/controls")
        factor, factor_binding = controls["prerequisites"], controls["input_binding_hash"]
        assert (factor["flow"], factor["missing"]) == ("FACTOR_STUDY", [])
        assert task_id in [row["task_id"] for row in factor["present"]["FACTOR_STUDY"]]
        assert factor["next_requests"]["plan"]["operation"] == "EXPERIMENT_PLAN"
        kind = "experiment_kind=alpha.model-development"
        alpha = _json(live, f"/api/experiments/controls?{kind}")["prerequisites"]
        assert alpha["flow"] == "ALPHA_STUDY"
        assert task_id in [row["task_id"] for row in alpha["present"]["FACTOR_STUDY"]]
        curated = [v for v in alpha["present"]["FACTOR_STUDY"] if v["curation_receipt_hashes"]]
        assert alpha["missing"] == ([] if curated else ["CURATED_FACTOR_STUDY"])
        assert set(alpha["next_requests"]) == ({"handoff"} if curated else {"curation"})
        drafted = live.operations.execute(
            Request(
                operation="EXPERIMENT_PORTFOLIO_DRAFT",
                task_id=UUID(task_id),
                candidate_id="any",
            )
        )
        assert drafted["failure_code"] == "portfolio_research.alpha_parent_required", drafted
        book = drafted["prerequisites"]
        assert book["flow"] == "BOOK"
        assert book["missing"] == ([] if book["present"]["ALPHA_STUDY"] else ["ALPHA_STUDY"])
        assert book["next_requests"], book
        # V376: the first read is the path, every standard flow on each input, as the controls
        # and the refusal above state them.
        shown = live.operations.execute(Request(operation="WORKSPACE_SHOW"))
        (intent,) = [v for v in shown["intents"] if v.get("input_binding_hash") == factor_binding]
        assert list(intent["flows"]) == [
            "FACTOR_STUDY",
            "ALPHA_STUDY",
            "RISK_STUDY",
            "BOOK",
            "BOOK_REVIEW",
            "FEATURE_TRIAL",
        ]
        assert intent["flows"]["ALPHA_STUDY"] == alpha
        assert intent["flows"]["FACTOR_STUDY"] == factor
        # V380: a binding the workspace does not hold is refused by its field, with the held
        # bindings and each one's feature controls (AX13: a typed binding met a bare code).
        unknown = live.operations.execute(
            Request(operation="FEATURE_CATALOG_CONTROLS", input_binding_hash="f" * 64)
        )
        assert unknown["failure_code"] == "feature_research.input_binding_unresolved", unknown
        assert factor_binding in unknown["expected"]["input_binding_hash"]
        assert unknown["next_requests"][f"controls:{factor_binding[:12]}"] == {
            "operation": "FEATURE_CATALOG_CONTROLS",
            "input_binding_hash": factor_binding,
        }
        assert unknown["fields"] == [["input_binding_hash"]] and unknown["detail"]


def test_factor_curation_and_alpha_handoff_are_durable_actor_neutral_and_never_train(
    inputs: Path,
    published,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import UUID

    from alphalattice.foundation.factor_research.experiments.execution import (
        FactorExperimentExecutor,
    )
    from alphalattice.investment.alpha_research.experiments.development_execution import (
        AlphaExperimentExecutor,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("curation or preview attempted numerical execution")

    monkeypatch.setattr(AlphaExperimentExecutor, "execute", forbidden)
    monkeypatch.setattr(FactorExperimentExecutor, "execute", forbidden)
    task_id, report, _export = published
    original = read_research_workspace_manifest(inputs).experiment_inputs[0]
    source = inputs / "research-inputs" / original.binding_hash / "source"
    bundle = read_factor_bundle(inputs, original.binding_hash)
    enhanced = bind_factor_inputs(
        workspace=inputs,
        source=source,
        include_alpha_handoff=True,
        panel_snapshot_hash=bundle.panel_snapshot_hash,
        outcome_snapshot_hash=bundle.outcome_snapshot_hash,
    )
    assert enhanced != original
    with _session(inputs) as session:
        controls = _json(session, f"/api/experiments/curation?task_id={task_id}")
        assert controls["status"] == "AVAILABLE", controls
        allowed = [v for v in controls["choices"] if v["roles"]]
        assert allowed, controls
        chosen = allowed[0]
        request = {
            "task_id": task_id,
            "experiment_curation": {
                "expected_receipt_hash": controls["receipt_hash"],
                "choices": [
                    {
                        "factor_id": chosen["factor_id"],
                        "role": chosen["roles"][0],
                        "rationale": "Explicit synthetic research decision.",
                    }
                ],
                "limitations_acknowledged": controls["limitations"],
            },
        }
        assert session.session is not None and session.operations is not None
        count = len(session.session.task_control_registry.tasks())
        agent = InstalledAgent(session.operations)
        assert (
            json.loads(
                agent.invoke(
                    PortfolioResearchAgentRequest(
                        operation="EXPERIMENT_CURATION", task_id=UUID(task_id)
                    )
                )
            )
            == controls
        )
        denied = json.loads(
            agent.invoke(PortfolioResearchAgentRequest(operation="EXPERIMENT_CURATE", **request))
        )
        assert denied["status"] == "REFUSED", denied
        for replacement in (
            {"expected_receipt_hash": "0" * 64},
            {"limitations_acknowledged": []},
            {
                "choices": [
                    {"factor_id": "not-reported", "role": "CORE", "rationale": "No authority."}
                ]
            },
        ):
            invalid = {
                "task_id": task_id,
                "experiment_curation": {**request["experiment_curation"], **replacement},
            }
            assert (
                _json(session, "/api/experiments/curation", method="POST", payload=invalid)[
                    "status"
                ]
                == "REFUSED"
            )
        human = _json(session, "/api/experiments/curation", method="POST", payload=request)
        assert human["status"] == "CURATION_PUBLISHED", human
        assert human["task"] is None
        # The decision offers its Alpha handoff filled, the receipt never copied (V391).
        assert human["next_requests"]["handoff"] == {
            "operation": "EXPERIMENT_HANDOFF_PREVIEW",
            "task_id": task_id,
            "curation_receipt_hash": human["decision"]["receipt_hash"],
            **{
                key: human["next_requests"]["foundation-preview"][key]
                for key in ("research_input_id", "input_binding_hash")
            },
        }
        actor = AgentExecutionBinding(
            profile_id="scripted-curator",
            mode="fixture",
            profile_hash="a" * 64,
            document_hash="b" * 64,
            response_protocol="research-experiment",
            response_protocol_hash="c" * 64,
            concrete_schema_hash="d" * 64,
        )
        admitted_agent = InstalledAgent(session.operations, agent_execution=actor)
        agent_result = json.loads(
            admitted_agent.invoke(
                PortfolioResearchAgentRequest(operation="EXPERIMENT_CURATE", **request)
            )
        )
        assert agent_result["decision"]["submission"] == human["decision"]["submission"]
        assert agent_result["decision"]["actor_submission"]["actor_kind"] == "INSTALLED_AGENT"
        assert (
            _json(session, "/api/experiments/curation", method="POST", payload=request)["status"]
            == "REUSED_EXACT"
        )
        selector = {"task_id": task_id, "curation_receipt_hash": human["decision"]["receipt_hash"]}
        incomplete = _json(session, "/api/experiments/handoff", method="POST", payload=selector)
        assert incomplete["status"] == "DRAFT_INCOMPLETE", incomplete
        # Who fills the open target and model, in the answer itself: the agent, disclosed,
        # never a stop for the person (FLOW-3; the Tech Lead's STOPS rulings, rows 32 and 44).
        assert incomplete["choice_owner"] == "AGENT_DISCLOSES"
        assert "does not stop to ask" in incomplete["detail"]
        assert (
            incomplete["foundation"]["selected_factor_ids"]
            == report["document"]["factor"]["factor_ids"]
        )
        assert incomplete["ordered_feature_ids"] == [chosen["factor_id"]]
        document = incomplete["document"]
        document["alpha"].update(
            {
                "target_recipe_id": "SECTOR_RESIDUAL_ROBUST_Z",
                "model_capability_handle": "capability-1",
                "model_parameters": {"family": "ridge", "alpha": 1.0},
            }
        )
        payload = {**selector, "experiment_document": document}
        preview = _json(session, "/api/experiments/handoff", method="POST", payload=payload)
        assert preview["status"] == "INPUT_COMPILER_PREFLIGHT_PASSED", preview
        assert preview["task"] is None and preview["numerical_call_count"] == 0
        # V361: the handoff offers the plan it starts, its Factor source kept, so
        # `experiment plan --from` plans it as it plans a draft.
        # V438: and the input it chose beside its binding, so the plan never falls back to
        # the Factor study's own input under another input's binding.
        assert preview["plan_request"] == {
            "operation": "EXPERIMENT_PLAN",
            "research_input_id": human["next_requests"]["foundation-preview"]["research_input_id"],
            "input_binding_hash": preview["input_binding_hash"],
            "factor_task_id": task_id,
            "curation_receipt_hash": human["decision"]["receipt_hash"],
        }
        assert (
            json.loads(
                agent.invoke(
                    PortfolioResearchAgentRequest(operation="EXPERIMENT_HANDOFF_PREVIEW", **payload)
                )
            )
            == preview
        )
        refused = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": document},
        )
        assert refused["status"] == "REFUSED", refused
        assert len(session.session.task_control_registry.tasks()) == count
        from copy import deepcopy

        for section, key, bad_value in (
            ("alpha", "ordered_feature_ids", ["unknown"]),
            ("experiment", "output_workspace", "../escape"),
        ):
            invalid_document = deepcopy(document)
            invalid_document[section][key] = bad_value
            refused = _json(
                session,
                "/api/experiments/handoff",
                method="POST",
                payload={**selector, "experiment_document": invalid_document},
            )
            assert refused["status"] == "REFUSED", refused
            if (section, key) == ("experiment", "output_workspace"):
                assert refused["failure_code"] == (
                    "factor_research.handoff_authority_field_mismatch:experiment.output_workspace"
                )
                assert refused["expected"] == {
                    "experiment.output_workspace": document["experiment"]["output_workspace"]
                }
                assert refused["next_action"] == "EDIT_DECLARATION_AND_PLAN"
                assert refused["detail"] == (
                    "A Factor handoff fixes `experiment.output_workspace` of the Alpha study it "
                    "opens: its kind, data, workspaces and publication intent come from the Factor "
                    "study, and its universe is that study's or a declared sample of it. Leave "
                    "the field out, or write the value `expected` gives, and plan again."
                )
    with _session(inputs) as restarted:
        choices = _json(restarted, f"/api/experiments/curation?task_id={task_id}")
        assert human["decision"] in choices["decisions"]
        replay = _json(
            restarted,
            "/api/experiments/handoff",
            method="POST",
            payload={**selector, "experiment_yaml": preview["yaml"]},
        )
        assert replay == preview
        # The decision remains historical evidence if an enhanced input breaks;
        # fresh downstream preparation must not silently reuse earlier success.
        target = inputs / "research-inputs" / enhanced.binding_hash / "source"
        mapping = (
            target
            / "artifacts/feature-panel/closure/logical/operational/snapshot-mappings"
            / f"{bundle.panel_snapshot_hash}.json"
        )
        saved = mapping.read_bytes()
        try:
            mapping.write_bytes(saved + b" ")
            refused = _json(
                restarted,
                "/api/experiments/handoff",
                method="POST",
                payload={**selector, "experiment_yaml": preview["yaml"]},
            )
            assert refused["status"] == "REFUSED" and "tampered" in refused["failure_code"]
            assert (
                human["decision"]
                in _json(restarted, f"/api/experiments/curation?task_id={task_id}")["decisions"]
            )
        finally:
            mapping.write_bytes(saved)


def test_agent_and_yaml_use_the_same_program_but_cannot_claim_provenance(inputs: Path) -> None:
    with _session(inputs) as session:
        controls = _json(session, "/api/experiments/controls")
        plan = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_yaml": controls["yaml"]},
        )
        assert plan["status"] == "PLANNED", plan
        assert session.operations is not None and session.session is not None
        agent = InstalledAgent(session.operations)
        other = json.loads(
            agent.invoke(
                PortfolioResearchAgentRequest(
                    operation="EXPERIMENT_PLAN", experiment_document=controls["template"]
                )
            )
        )
        assert other["program"] == plan["program"]
        before = len(session.session.task_control_registry.tasks())
        refused = json.loads(
            agent.invoke(
                PortfolioResearchAgentRequest(
                    operation="EXPERIMENT_RUN", experiment_plan_hash=plan["plan_hash"]
                )
            )
        )
        assert refused["status"] == "REFUSED"
        assert len(session.session.task_control_registry.tasks()) == before
        with pytest.raises(ValueError):
            PortfolioResearchAgentRequest(operation="EXPERIMENTS", actor_kind="HUMAN")
        actor = AgentExecutionBinding(
            profile_id="scripted-research-author",
            mode="fixture",
            profile_hash="a" * 64,
            document_hash="b" * 64,
            response_protocol="research-experiment",
            response_protocol_hash="c" * 64,
            concrete_schema_hash="d" * 64,
        )
        agent = InstalledAgent(session.operations, agent_execution=actor)
        run = json.loads(
            agent.invoke(
                PortfolioResearchAgentRequest(
                    operation="EXPERIMENT_RUN", experiment_plan_hash=plan["plan_hash"]
                )
            )
        )
        assert run["task_id"], run
        assert session.dispatcher is not None
        session.dispatcher.drain_for_tests()
        body = _json(session, f"/api/experiments/readback?task_id={run['task_id']}")
        assert body["status"] == "EXPERIMENT_PUBLISHED", body
        assert body["actor"]["actor_kind"] == "INSTALLED_AGENT"
        assert body["actor"]["agent_execution"] == actor.model_dump(mode="json")


def _study_on_the_bound_input(session: LocalPortfolioWebSession) -> str:
    """The published fixture's study over the input the workspace binds now, run or reused.

    A replay reads the bound input, and the module's tests share the workspace: the curation
    test rebinds it, so the last study listed is over the bound input only when a later test ran
    one first, which a worker running a subset of the module need not do.
    """

    controls = _json(session, "/api/experiments/controls")
    document = controls["template"]
    document["factor"]["factor_ids"] = controls["factor_options"][:4]
    plan = _json(
        session, "/api/experiments/plan", method="POST", payload={"experiment_document": document}
    )
    assert plan["status"] == "PLANNED", plan
    run = _json(
        session,
        "/api/experiments/run",
        method="POST",
        payload={"experiment_plan_hash": plan["plan_hash"]},
    )
    if run["status"] == "REUSED_EXACT":
        return str(run["publication_task_id"])
    assert run["task_id"], run
    assert session.dispatcher is not None
    session.dispatcher.drain_for_tests()
    return str(run["task_id"])


def test_input_and_result_tamper_refuse_without_losing_historical_readback(
    inputs: Path, published
) -> None:
    with _session(inputs) as session:
        task_id = _study_on_the_bound_input(session)
        old = _json(session, f"/api/experiments/readback?task_id={task_id}")
        assert old["status"] == "EXPERIMENT_PUBLISHED", old
        binding = session.workspace_manifest.experiment_inputs[0]
        bundle = read_factor_bundle(inputs, binding.binding_hash)
        chunk_name = next(name for name, _ in bundle.files if name.endswith(".parquet"))
        chunk = inputs / "research-inputs" / binding.binding_hash / "source" / chunk_name
        original = chunk.read_bytes()
        try:
            chunk.write_bytes(original + b"tampered")
            refused = _json(
                session, "/api/experiments/replay", method="POST", payload={"task_id": task_id}
            )
            assert refused["status"] == "REFUSED" and "tampered" in refused["failure_code"]
            again = _json(session, f"/api/experiments/readback?task_id={task_id}")
            assert _facts(again) == _facts(old)
        finally:
            chunk.write_bytes(original)
        output = inputs / old["document"]["experiment"]["output_workspace"] / "research-evidence"
        evidence_file = next(output.glob("*.json"))
        original = evidence_file.read_bytes()
        try:
            evidence_file.write_bytes(b"{}")
            refused = _json(session, f"/api/experiments/readback?task_id={task_id}")
            assert refused["status"] == "REFUSED"
        finally:
            evidence_file.write_bytes(original)
        again = _json(session, f"/api/experiments/readback?task_id={task_id}")
        assert _facts(again) == _facts(old)


def test_a_read_reuses_the_study_verification_while_its_files_are_unchanged(
    inputs: Path, published, monkeypatch
) -> None:
    """requirement (binding plan, L1; V265, V89): a readback answers from the study's last full
    verification while its files keep their path, size, time and identity, and says it checked
    exactly that; export verifies in full; a change that keeps them is caught by export and by
    the sweep, which names the study and drops what reads keep of it; a change of size or time
    is verified at the next read."""

    root = inputs
    task_id = published[0]
    with _session(root) as live:
        readback = f"/api/experiments/readback?task_id={task_id}"
        export = f"/api/experiments/export?task_id={task_id}"
        first, second = _json(live, readback), _json(live, readback)
        assert first["verification_basis"] == "FULL"
        assert second["verification_basis"].startswith("FILES_UNCHANGED ")
        assert _facts(second) == _facts(first)
        output = root / first["document"]["experiment"]["output_workspace"] / "research-evidence"
        evidence_file = next(output.glob("*.json"))
        original, stat = evidence_file.read_bytes(), evidence_file.stat()
        # Same size, same modification time: the fingerprint cannot see it.
        evidence_file.write_bytes(original.replace(b'"', b"'", 1))
        os.utime(evidence_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert _json(live, readback)["verification_basis"].startswith("FILES_UNCHANGED ")
        assert _json(live, export)["status"] == "REFUSED"
        sent = live.operations.verify_all()
        assert sent["status"] == "ADMITTED" and sent["saved_studies"] >= 1, sent
        live.dispatcher.drain_for_tests()
        failed = live.operations.sweep.failed_studies()
        assert str(task_id) in failed
        assert _json(live, readback)["status"] == "REFUSED"
        overview = _json(live, "/api/upgrade")
        (row,) = [study for study in overview["studies"] if study["task_id"] == str(task_id)]
        assert row["state"] == "INTEGRITY_FAILED" and row["failure_code"] == failed[str(task_id)]
        evidence_file.write_bytes(original)
        assert _json(live, readback)["verification_basis"] == "FULL"
        evidence_file.write_bytes(b"{}")
        assert _json(live, readback)["status"] == "REFUSED"


def test_an_alpha_studys_full_verification_is_kept_on_disk_across_a_restart(
    alpha_case, monkeypatch
) -> None:
    """requirement (V89, V265, decision 5): an Alpha study's full verification is kept on disk
    as its read shows it, so a Host that starts again reads it without re-deriving every chunk;
    the basis names what it checked; a file whose time moved, or another installed
    implementation, is verified in full."""

    from alphalattice.control.product_host.composition import research_experiments as owner

    root = alpha_case[0]
    with _session(root) as live:
        # A penalty no other case in this module runs: the shared workspace may already hold
        # another case's study, which a run would answer as its exact reuse, with no Task.
        plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload=_alpha_payload(alpha_case, {"family": "ridge", "alpha": 7.0}),
        )
        sent = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        live.dispatcher.drain_for_tests()
        readback = f"/api/experiments/readback?task_id={sent['task_id']}"
        first = _json(live, readback)
        assert first["verification_basis"] == "FULL", first
    with _session(root) as restarted:
        again = _json(restarted, readback)
        assert again["verification_basis"].startswith("FILES_UNCHANGED "), again
        assert "path, size, time and identity" in again["verification_basis"]
        assert _facts(again) == _facts(first)
        output = root / first["document"]["experiment"]["output_workspace"]
        moved = next(path for path in sorted(output.rglob("*")) if path.is_file())
        stat = moved.stat()
        os.utime(moved, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert _json(restarted, readback)["verification_basis"] == "FULL"
    monkeypatch.setattr(owner, "plan_implementation_hash", lambda _plan: "0" * 64)
    with _session(root) as upgraded:
        assert _json(upgraded, readback)["verification_basis"] == "FULL"


def test_the_sweep_is_due_after_a_week_or_an_upgrade_and_takes_up_what_it_left(
    inputs: Path, published, monkeypatch
) -> None:
    """requirement (V89, decision 5): an idle Host admits the sweep when none ran since an
    upgrade or a week; it stops as soon as another Task waits, and the next sweep takes up the
    rest; each sweep seals a report under its content hash."""

    from datetime import timedelta

    root = inputs
    with _session(root) as live:
        sweep = live.operations.sweep
        assert sweep.saved_studies() and not sweep.due()  # before the first, a week of work
        week_on = sweep.clock() + timedelta(days=8)
        ticks = iter(range(10_000))  # a clock a week on that moves, a second each read
        monkeypatch.setattr(sweep, "clock", lambda: week_on + timedelta(seconds=next(ticks)))
        assert sweep.due()
        waits = iter([True])  # a Task waits once, as the first sweep starts
        monkeypatch.setattr(sweep, "work_waits", lambda: next(waits, False))
        live.operations.sweep_if_due()
        live.dispatcher.drain_for_tests()
        # The idle Host takes up what the first one left; asked again here, only one admits it.
        live.operations.sweep_if_due()
        live.dispatcher.drain_for_tests()
        reports = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(sweep.root.glob("*.json"))
        ]
        assert len(reports) == 2
        stopped, whole = sorted(reports, key=lambda report: str(report["finished_at"]))
        assert stopped["verified"] == [] and stopped["remaining"] == len(sweep.saved_studies())
        assert whole["remaining"] == 0 and whole["failed"] == [] and whole == sweep.last_report()
        assert sorted(whole["verified"]) == sorted(str(t.task_id) for t in sweep.saved_studies())
        assert not sweep.due()
        later = week_on + timedelta(days=16)
        monkeypatch.setattr(sweep, "clock", lambda: later)
        assert sweep.due()  # a week passed
        monkeypatch.setattr(sweep, "clock", lambda: week_on)
        monkeypatch.setattr(sweep, "installed", lambda: "0" * 64)
        assert sweep.due()  # the installed study code moved
        assert all(
            path.stem == canonical_hash(json.loads(path.read_text(encoding="utf-8")))
            for path in sweep.root.glob("*.json")
        )


def test_hard_process_death_recovers_the_same_task_without_reexecuting_complete_evidence(
    inputs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "stage-written.json"
    script = r"""
import json, sys, threading
from pathlib import Path
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.research_experiments import (
    ResearchExperimentApplication, STAGES,
)
from tests.portfolio_strategy_lab.local_web_support import _Resolver, _resolved, _json
root, marker = Path(sys.argv[1]), Path(sys.argv[2])
original = ResearchExperimentApplication.execute_stage
def pause(self, **kwargs):
    result = original(self, **kwargs)
    if kwargs['work_item'].stage_id == STAGES[0] and result.failure_code is None:
        marker.write_text(json.dumps({'task_id': str(kwargs['task'].task_id)}), encoding='utf-8')
        print('TEST_BOUNDARY', flush=True)
        threading.Event().wait()
    return result
ResearchExperimentApplication.execute_stage = pause
with LocalPortfolioWebSession(workspace=root,
    workspace_manifest=read_research_workspace_manifest(root),
    resolver=_Resolver(_resolved())) as session:
    controls = _json(session, '/api/experiments/controls')
    document = controls['template']
    document['factor']['factor_ids'] = controls['factor_options'][-3:]
    plan = _json(session, '/api/experiments/plan', method='POST',
        payload={'experiment_document': document})
    assert plan['status'] == 'PLANNED', plan
    run = _json(session, '/api/experiments/run', method='POST',
        payload={'experiment_plan_hash': plan['plan_hash']})
    assert run['task_id'], run
    threading.Event().wait()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(inputs), str(marker)],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": "src", "ALPHALATTICE_NETWORK_DISABLED": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        if not _wait_for_boundary(process, marker, 120):
            process.terminate()
            out, error = process.communicate(timeout=15)
            pytest.fail(f"worker did not reach the committed evidence boundary: {out!r} {error!r}")
        task_id = json.loads(marker.read_text(encoding="utf-8"))["task_id"]
        process.kill()
        process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=15)
    from alphalattice.foundation.factor_research.experiments.execution import (
        FactorExperimentExecutor,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("a complete evidence graph must not execute again")

    monkeypatch.setattr(FactorExperimentExecutor, "execute", forbidden)
    with _session(inputs) as recovered:
        assert task_id in {str(value) for value in recovered.resumed_task_ids}
        recovered.dispatcher.drain_for_tests()
        result = _json(recovered, f"/api/experiments/readback?task_id={task_id}")
        assert result["status"] == "EXPERIMENT_PUBLISHED", result
        assert (
            sum(
                row["task_id"] == task_id
                for row in _json(recovered, "/api/experiments")["experiments"]
            )
            == 1
        )


def test_a_capacity_block_reopens_the_same_study_only_after_capacity_is_restored(
    inputs: Path,
) -> None:
    """An operator cap stop retains the sealed plan and requires a current recovery confirmation."""
    with _session(inputs) as live:
        controls = _json(
            live, "/api/experiments/controls?experiment_kind=risk.covariance-development"
        )
        assert controls["status"] == "READY", controls
        plan = _json(
            live,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": controls["template"]},
        )
        assert plan["status"] == "PLANNED", plan
        manifest = (inputs / "research-workspace.json").read_bytes()
        _json(live, "/api/workspace/storage/cap", method="POST", payload={"storage_cap_bytes": "1"})
        run = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        task = run["task_id"]
        live.dispatcher.drain_for_tests()
        view = _json(live, "/api/tasks/recovery?task_id=" + task)
        assert view["lifecycle"] == "BLOCKED", view
        assert view["stop"]["code"] == "storage.managed_capacity_exceeded"
        assert next(a for a in view["actions"] if a["action"] == "RECOVER")["available"]
        recover = {k: v for k, v in view["next_requests"]["recover"].items() if k != "operation"}
        refused = _json(live, "/api/recover", method="POST", payload=recover)
        assert refused["failure_code"] == "storage.managed_capacity_exceeded", refused
        unchanged = _json(live, "/api/tasks/recovery?task_id=" + task)
        assert unchanged["task_record_hash"] == view["task_record_hash"]
        assert unchanged["stages"] == view["stages"]
        _json(
            live,
            "/api/workspace/storage/cap",
            method="POST",
            payload={"storage_cap_bytes": str(20 * 1024**3)},
        )
        # Restored capacity cannot authorize a changed input. The same confirmed
        # Task remains blocked until the exact input bytes are restored too.
        source = inputs / "research-inputs" / controls["input_binding_hash"] / "source"
        held = next(
            p
            for p in source.rglob("*.json")
            if p.parent.name == "manifests" and p.parent.parent.name == "feature-panel"
        )
        saved = held.read_bytes()
        try:
            held.write_bytes(b"{}")
            changed = _json(live, "/api/recover", method="POST", payload=recover)
            assert changed["disposition"] == "REFUSED", changed
            assert "input_file_tampered" in changed["failure_code"], changed
            still = _json(live, "/api/tasks/recovery?task_id=" + task)
            assert still["task_record_hash"] == view["task_record_hash"]
            assert still["stages"] == view["stages"]
        finally:
            held.write_bytes(saved)
        resumed = _json(live, "/api/recover", method="POST", payload=recover)
        assert resumed["resumed_task_ids"] == [task], resumed
        live.dispatcher.drain_for_tests()
        finished = _json(live, "/api/tasks/recovery?task_id=" + task)
        assert finished["lifecycle"] == "SUCCEEDED", finished
        assert finished["verified_stage_count"] == finished["total_stage_count"] == 2
        assert _json(live, "/api/recover", method="POST", payload=recover)["failure_code"] == (
            "local_application.confirmation_stale"
        )
        result = _json(live, "/api/experiments/readback?task_id=" + task)
        assert result["status"] == "EXPERIMENT_PUBLISHED", result
        assert result["document"] == plan["document"]
        assert (inputs / "research-workspace.json").read_bytes() == manifest
        reused = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        assert reused["status"] == "REUSED_EXACT" and reused["publication_task_id"] == task


def test_bad_requests_admit_no_task_and_do_not_affect_portfolio(inputs: Path) -> None:
    from tests.portfolio_strategy_lab.local_web_support import _request

    with _session(inputs) as session:
        controls = _json(session, "/api/experiments/controls")
        before = len(session.session.task_control_registry.tasks())
        for payload in (
            {"experiment_yaml": 5},
            {"experiment_document": []},
            {"experiment_document": {"experiment": []}},
            {"experiment_yaml": "!!python/object:builtins.object {}"},
            {"experiment_yaml": controls["yaml"], "chosen_by": "INSTALLED_AGENT"},
        ):
            code, _headers, raw = _request(
                session, "/api/experiments/plan", method="POST", payload=payload
            )
            body = json.loads(raw)
            assert code == 400 or body.get("status") == "REFUSED", body
            assert code != 500
        document = controls["template"]
        document["experiment"]["output_workspace"] = "../outside"
        body = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": document},
        )
        assert body["status"] == "REFUSED"
        assert len(session.session.task_control_registry.tasks()) == before
        assert _json(session, "/api/controls")["controls"]
        first = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_yaml": controls["yaml"]},
        )
        alternate = json.loads(json.dumps(controls["template"]))
        alternate["experiment"]["output_workspace"] = "managed"
        alternate["factor"]["factor_ids"] = alternate["factor"]["factor_ids"][:2]
        _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": alternate},
        )
        # A second PLAN no longer displaces the first: both stay addressable by
        # their own hash, and only a hash this service never previewed is refused.
        retained = _json(
            session, f"/api/experiments/preview?experiment_plan_hash={first['plan_hash']}"
        )
        assert retained["status"] == "AVAILABLE" and retained["retained_previews"] == 2
        stale = _json(
            session,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": "0" * 64},
        )
        assert stale["status"] == "REFUSED" and "preview_required" in stale["failure_code"]
        assert len(session.session.task_control_registry.tasks()) == before
        document = controls["template"]
        document["experiment"]["output_workspace"] = "managed"
        document["experiment"]["sessions"] = {
            "start": "2024-02-01",
            "end": "2024-03-01",
            "as_of": {"session": "2024-03-15", "phase": "OFFICIAL_CLOSE"},
        }
        cutoff = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_document": document},
        )
        assert (
            cutoff["status"] == "REFUSED"
            and "outcomes_after_declared_cutoff" in cutoff["failure_code"]
        )
        assert len(session.session.task_control_registry.tasks()) == before


def test_a_feature_listing_page_is_read_as_its_number(tmp_path: Path) -> None:
    """regression (the UI pass's closing stills): over the Local Web a page number is a query
    string; the route reads `extensions_page` as the int it names, so a page past the last is
    refused by that name, not as an invalid field."""

    root = tmp_path / "workspace"
    # Pagination reads the extension records, not market rows or a bound study.
    publish_research_workspace_manifest(root, _manifest("feature-page"))
    with _session(root) as live:
        first = _json(live, "/api/feature-research?extensions_page=1")
        assert (first["status"], first["page"]) == ("FEATURE_EXTENSIONS", 1)
        past = first["page_count"] + 1
        _status, _headers, body = _request(live, f"/api/feature-research?extensions_page={past}")
        refused = json.loads(body)
        assert (refused["status"], refused["failure_code"]) == (
            "REFUSED",
            f"feature_extension.extensions_page_out_of_range:{past} of {past - 1}",
        ), refused


def test_cli_and_web_compile_the_same_bound_document(inputs: Path, tmp_path: Path) -> None:
    with _session(inputs) as session:
        controls = _json(session, "/api/experiments/controls")
        preview = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_yaml": controls["yaml"]},
        )
        assert preview["status"] == "PLANNED", preview
    document = tmp_path / "experiment.yaml"
    document.write_text(controls["yaml"], encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    run = subprocess.run(
        [
            sys.executable,
            "scripts/run_research_experiment.py",
            "--preflight",
            "--workspace",
            str(inputs),
            "--research-input",
            controls["input_id"],
            str(document),
        ],
        cwd=root,
        env={**os.environ, "ALPHALATTICE_NETWORK_DISABLED": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert run.returncode == 0, (run.stdout, run.stderr)
    body = json.loads(run.stdout)
    assert body["program"]["program_hash"] == preview["program"]["program_hash"]
    assert body["document"] == preview["document"]


def test_small_input_reports_the_production_requirement_before_any_task(
    real_risk_workspace,
    tmp_path: Path,
) -> None:
    publish_research_workspace_manifest(tmp_path, _manifest("small-factor"))
    outcome, _ref = publish_causal_outcomes(
        real_risk_workspace, at=datetime(2026, 8, 2, tzinfo=UTC)
    )
    bind_factor_inputs(
        workspace=tmp_path, source=real_risk_workspace.workspace, outcome_snapshot_hash=outcome
    )
    with _session(tmp_path) as session:
        controls = _json(session, "/api/experiments/controls")
        result = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={"experiment_yaml": controls["yaml"]},
        )
        assert result["status"] == "REFUSED"
        assert "insufficient_listing_support" in result["failure_code"]
        assert session.session.task_control_registry.tasks() == ()


# ---- trying a feature is one request (binding plan, B18)

_TRIAL_FIELDS = {
    "provider_adjusted_close",
    "close_split_adjusted",
    "open_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
    "volume_raw",
    "close_raw",
}


def test_a_feature_is_tried_in_one_request_and_reopened_when_asked_again(completed_alpha_case):
    alpha_case = completed_alpha_case
    root = alpha_case[0]
    with _session(root) as live:
        ops = live.operations
        rows = _json(live, "/api/experiments")["experiments"]
        alphas = [
            r
            for r in rows
            if r["kind"] == "alpha.model-development" and r["lifecycle"] == "SUCCEEDED"
        ]
        if alphas:
            alpha = alphas[0]
        else:
            plan = _json(
                live, "/api/experiments/plan", method="POST", payload=_alpha_payload(alpha_case)
            )
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            alpha = next(
                r
                for r in _json(live, "/api/experiments")["experiments"]
                if r["task_id"] == sent["task_id"]
            )
        controls = ops.execute(
            Request(
                operation="FEATURE_CATALOG_CONTROLS", input_binding_hash=alpha["input_binding_hash"]
            )
        )
        present = {d.get("factor_id") for d in controls.get("definitions") or []}
        formula = next(
            f
            for f in controls["registered_formulas"]
            if f["factor_id"] not in present
            and ".experimental." in str(f.get("formula_ref"))
            and set(f["required_fields"]) <= _TRIAL_FIELDS
        )
        document = json.loads(json.dumps(controls["template"]))
        document["edits"] = [
            {
                "operation": "CREATE",
                "factor_id": f"trial_{formula['factor_id']}",
                "specification": {**formula, "factor_id": f"trial_{formula['factor_id']}"},
            }
        ]
        document["reason"] = "Regression: try one registered formula."
        planned = ops.execute(Request(operation="FEATURE_CATALOG_PLAN", feature_document=document))
        plan_hash = planned.get("plan_hash") or planned["plan"]["plan_hash"]
        request = {"feature_plan_hash": plan_hash, "task_id": alpha["task_id"]}

        # A queued study holds this workspace. The trial must remain stopped until
        # a request can admit work, and reopening then lets the idle hook advance it.
        from uuid import UUID

        registry = live.session.task_control_registry
        baseline = registry.task(UUID(alpha["task_id"]))
        busy = registry.admit(
            input_envelope=baseline.input,
            goal=baseline.goal,
            plan=baseline.plan,
            observed_at=datetime.now(UTC),
        ).record
        stopped = _json(live, "/api/feature-trials", method="POST", payload=request)
        assert stopped["state"] == "STOPPED", stopped
        assert stopped["stopped"] == {
            "step": "FEATURE_BUILD",
            "failure_code": "feature_research.finish_or_recover_existing_task",
        }
        refused_again = _json(live, "/api/feature-trials", method="POST", payload=request)
        assert refused_again["state"] == "STOPPED"
        assert refused_again["stopped"] == stopped["stopped"]
        registry.request_cancel(
            task_id=busy.task_id,
            expected_task_hash=busy.record_hash,
            observed_at=datetime.now(UTC),
        )

        started = _json(live, "/api/feature-trials", method="POST", payload=request)
        assert (started["status"], started["state"]) == ("FEATURE_TRIAL", "RUNNING"), started
        assert started["stopped"] is None
        trial_id = started["feature_trial_id"]
        # The real CLI waits while the Host's idle hook advances the reopened chain.
        answer_path = root / "trial-wait-answer.json"
        waited = subprocess.run(
            [
                sys.executable,
                "scripts/run_alphalattice.py",
                "--workspace",
                str(root),
                "trial",
                "show",
                trial_id,
                "--wait",
                "--max-wait",
                "600",
                "--output",
                str(answer_path),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "ALPHALATTICE_NETWORK_DISABLED": "1"},
            capture_output=True,
            text=True,
            timeout=630,
            check=False,
        )
        assert waited.returncode == 0, (waited.stdout, waited.stderr)
        waited_answer = json.loads(answer_path.read_text(encoding="utf-8"))
        assert waited_answer["state"] == "COMPLETED", waited_answer
        assert waited_answer["stopped"] is None
        body = _json(live, f"/api/feature-trials/readback?feature_trial_id={trial_id}")
        assert body["state"] == "COMPLETED", body
        assert body["outcome"] in {"COMPARED", "FEATURE_NOT_ADMITTED_BY_SCREENING"}
        steps = {step["step"]: step["state"] for step in body["steps"]}
        assert steps["FEATURE_BUILD"] == steps["FACTOR_STUDY"] == "SUCCEEDED"
        if body["outcome"] == "COMPARED":
            assert steps["ALPHA_STUDY"] == "SUCCEEDED" and body["comparison"]["feature"]
            owner = body["comparison"]["alpha_without_and_with"]["owner_comparison"]
            assert (owner["status"], owner["relation"]) == ("COMPARABLE", "FEATURE_ADDITION")

        tasks = len(live.session.task_control_registry.tasks())
        again = _json(live, "/api/feature-trials", method="POST", payload=request)
        live.dispatcher.drain_for_tests(timeout=600)
        assert again["feature_trial_id"] == trial_id and again["state"] == "COMPLETED"
        assert _json(live, f"/api/feature-trials/readback?feature_trial_id={trial_id}") == body
        assert len(live.session.task_control_registry.tasks()) == tasks
        listed = _json(live, "/api/feature-trials")["trials"]
        assert [t["feature_trial_id"] for t in listed] == [trial_id]

        # A record damaged in place is named with its way on, never skipped or read (V272):
        # starting the trial again keeps it aside and runs the chain anew, by reuse alone.
        record = root / "runtime" / "feature-trials" / f"{trial_id}.json"
        record.write_bytes(record.read_bytes()[:40])
        damaged = _json(live, f"/api/feature-trials/readback?feature_trial_id={trial_id}")
        assert damaged["failure_code"] == "feature_trial.record_damaged", damaged
        assert "Start the trial again" in damaged["detail"]
        listing = _json(live, "/api/feature-trials")
        assert listing["trials"] == [] and listing["damaged"] == [
            {
                "status": "REFUSED",
                "feature_trial_id": trial_id,
                "failure_code": "feature_trial.record_damaged",
                "detail": (
                    "This trial's record in the workspace no longer reads: it was changed or cut "
                    "short. Start the trial again with its feature plan and study; the damaged "
                    "record is kept aside and every step that did not change is reused."
                ),
                "next_requests": {
                    "trials": {"operation": "FEATURE_TRIALS"},
                    "storage": {"operation": "STORAGE_READBACK"},
                },
            }
        ]
        restarted = _json(live, "/api/feature-trials", method="POST", payload=request)
        assert restarted["feature_trial_id"] == trial_id
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            progress = _drain_trial(live)
            restarted = _json(live, f"/api/feature-trials/readback?feature_trial_id={trial_id}")
            if restarted["state"] != "RUNNING":
                break
            progress.wait(max(0.0, deadline - time.monotonic()))
        assert restarted["state"] == "COMPLETED" and restarted["outcome"] == body["outcome"]
        assert len(live.session.task_control_registry.tasks()) == tasks
        kept = sorted(path.name for path in record.parent.iterdir())
        assert len(kept) == 2 and kept[0] == record.name
        assert kept[1].startswith(f"{record.name}.damaged-")

        cancelled_document = deepcopy(document)
        cancelled_document["edits"][0]["factor_id"] = "cancelled_trial_feature"
        cancelled_document["edits"][0]["specification"]["factor_id"] = "cancelled_trial_feature"
        cancelled_plan = ops.execute(
            Request(operation="FEATURE_CATALOG_PLAN", feature_document=cancelled_document)
        )
        cancelled_hash = cancelled_plan.get("plan_hash") or cancelled_plan["plan"]["plan_hash"]
        # Admission runs on this thread. Holding the writer boundary keeps its worker
        # from starting until the public cancellation has ended this queued Task.
        with live.session.mutation_gate.hold():
            pending = ops.execute(
                Request(
                    operation="FEATURE_TRIAL",
                    feature_plan_hash=cancelled_hash,
                    task_id=UUID(alpha["task_id"]),
                )
            )
            build = next(step for step in pending["steps"] if step["step"] == "FEATURE_BUILD")
            task = registry.task(UUID(build["task_id"]))
            registry.request_cancel(
                task_id=task.task_id,
                expected_task_hash=task.record_hash,
                observed_at=datetime.now(UTC),
            )
        live.dispatcher.drain_for_tests()
        cancelled = _json(
            live,
            "/api/feature-trials",
            method="POST",
            payload={"feature_plan_hash": cancelled_hash, "task_id": alpha["task_id"]},
        )
        assert cancelled["state"] == "STOPPED" and cancelled["stopped"]["step"] == "FEATURE_BUILD"
        cancelled_again = _json(
            live,
            "/api/feature-trials",
            method="POST",
            payload={"feature_plan_hash": cancelled_hash, "task_id": alpha["task_id"]},
        )
        assert cancelled_again["state"] == "STOPPED"
        assert cancelled_again["stopped"] == cancelled["stopped"]
        assert registry.task(task.task_id).lifecycle.value == "CANCELLED"


def test_a_formula_reading_the_sector_leaf_is_built_and_tried_but_not_activated(
    completed_alpha_case,
):
    """requirement (V359): a formula reading `sector_return` is built from the day's Sector
    returns across the input's cross-section and tried like any feature; its review refuses the
    activation by name until the daily build carries the Sector child."""
    alpha_case = completed_alpha_case

    root = alpha_case[0]
    with _session(root) as live:
        ops = live.operations
        alpha = next(
            (
                r
                for r in _json(live, "/api/experiments")["experiments"]
                if r["kind"] == "alpha.model-development" and r["lifecycle"] == "SUCCEEDED"
            ),
            None,
        )
        if alpha is None:
            plan = _json(
                live, "/api/experiments/plan", method="POST", payload=_alpha_payload(alpha_case)
            )
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            alpha = next(
                r
                for r in _json(live, "/api/experiments")["experiments"]
                if r["task_id"] == sent["task_id"]
            )
        controls = ops.execute(
            Request(
                operation="FEATURE_CATALOG_CONTROLS", input_binding_hash=alpha["input_binding_hash"]
            )
        )
        assert "sector_return" in controls["formula_language"]["research_leaves"]
        base = next(
            f for f in controls["registered_formulas"] if ".experimental." in str(f["formula_ref"])
        )
        formula = "ts_sum(log(adjusted_close / lag(adjusted_close, 1)) - sector_return, 5)"
        document = json.loads(json.dumps(controls["template"]))
        document["edits"] = [
            {
                "operation": "CREATE",
                "factor_id": "formula_sector_gap_5",
                "specification": {
                    **base,
                    "factor_id": "formula_sector_gap_5",
                    "formula_ref": "factor.formula",
                    "formula": formula,
                    "lag_sessions": 0,
                },
                "preprocessing_recipe": "ROBUST_UNIVERSE_Z",
            }
        ]
        document["reason"] = "A formula reading the day's Sector return."
        planned = ops.execute(Request(operation="FEATURE_CATALOG_PLAN", feature_document=document))
        plan_hash = planned.get("plan_hash") or planned["plan"]["plan_hash"]
        request = {"feature_plan_hash": plan_hash, "task_id": alpha["task_id"]}
        started = _json(live, "/api/feature-trials", method="POST", payload=request)
        trial_id = started["feature_trial_id"]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            progress = _drain_trial(live)
            body = _json(live, f"/api/feature-trials/readback?feature_trial_id={trial_id}")
            if body["state"] != "RUNNING":
                break
            progress.wait(max(0.0, deadline - time.monotonic()))
        assert body["state"] == "COMPLETED", (body["stopped"], body["steps"])

        query = f"feature_plan_hash={plan_hash}&feature_factor_id=formula_sector_gap_5"
        packet = _json(live, f"/api/feature-research/review?{query}")
        assert packet["contract"]["status"] == "PASSED", packet["contract"]
        kept = packet["declaration"]["specification"]
        assert kept["formula_ref"] == "factor.formula.sector"
        assert "sector_return_log" in kept["required_fields"]
        assert 0 < packet["build"]["coverage"] <= 1
        assert packet["active_panel"] == {
            "admitted": False,
            "reason": "feature_extension.sector_leaf_research_only:formula_sector_gap_5",
        }
        assert "activate" not in packet["next_requests"]


def test_a_trial_refused_its_study_names_the_studies_it_can_run_against(completed_alpha_case):
    """requirement (V354): a feature's plan states what its trial runs against before any build,
    and offers the trial with the study left to choose; a trial asked of a study not handed off
    from Factor evidence is refused with the completed studies on the feature's input it can run
    against, newest first, and creates nothing."""
    alpha_case = completed_alpha_case

    from uuid import UUID

    root, _binding, factor_task, _decision, _original = alpha_case
    with _session(root) as live:
        ops = live.operations
        eligible = [
            r
            for r in _json(live, "/api/experiments")["experiments"]
            if r["kind"] == "alpha.model-development"
            and r["lifecycle"] == "SUCCEEDED"
            and "target_recipe_id" in r
        ]
        if not eligible:
            plan = _json(
                live, "/api/experiments/plan", method="POST", payload=_alpha_payload(alpha_case)
            )
            _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            eligible = [
                r
                for r in _json(live, "/api/experiments")["experiments"]
                if r["kind"] == "alpha.model-development" and r["lifecycle"] == "SUCCEEDED"
            ]
        binding = eligible[0]["input_binding_hash"]
        controls = ops.execute(
            Request(operation="FEATURE_CATALOG_CONTROLS", input_binding_hash=binding)
        )
        base = next(
            f for f in controls["registered_formulas"] if ".experimental." in str(f["formula_ref"])
        )
        document = json.loads(json.dumps(controls["template"]))
        document["edits"] = [
            {
                "operation": "CREATE",
                "factor_id": "formula_way_on_20",
                "specification": {
                    **base,
                    "factor_id": "formula_way_on_20",
                    "formula_ref": "factor.formula",
                    "formula": "close / lag(close, 20) - 1",
                    "lag_sessions": 0,
                },
                "preprocessing_recipe": "ROBUST_UNIVERSE_Z",
            }
        ]
        document["reason"] = "A trial's way on."
        planned = ops.execute(Request(operation="FEATURE_CATALOG_PLAN", feature_document=document))
        plan_hash = planned.get("plan_hash") or planned["plan"]["plan_hash"]
        readback = ops.execute(
            Request(operation="FEATURE_CATALOG_READBACK", feature_plan_hash=plan_hash)
        )
        # V559 (the user's review at de555b07): the exported declaration is the document
        # `feature plan --file` reads, never a whole request around it, and plans as it is.
        from alphalattice.interface.local_application.client import document_field

        exported = root.parent / "feature-edit.yaml"
        # What `--save-declaration` writes for an answer that carries its `yaml`.
        exported.write_text(readback["yaml"], encoding="utf-8")
        field, declared = document_field(["feature_document"], str(exported))
        assert (field, declared) == ("feature_document", readback["document"])
        replanned = ops.execute(
            Request(operation="FEATURE_CATALOG_PLAN", feature_document=declared)
        )
        assert (replanned.get("plan_hash") or replanned["plan"]["plan_hash"]) == plan_hash
        assert "Factor study" in readback["trial_baseline"]
        assert readback["next_requests"]["trial"] == {
            "operation": "FEATURE_TRIAL",
            "feature_plan_hash": plan_hash,
            "task_id": None,
        }
        before = ops.execute(Request(operation="FEATURE_TRIALS"))["trials"]
        refused = ops.execute(
            Request(
                operation="FEATURE_TRIAL", feature_plan_hash=plan_hash, task_id=UUID(factor_task)
            )
        )
        assert refused["failure_code"] == "feature_trial.study_not_from_factor_evidence"
        offered = refused["next_requests"]
        assert offered and len(offered) <= 5
        studies = {r["task_id"]: r for r in _json(live, "/api/experiments")["experiments"]}
        for name, request in offered.items():
            task_id = name.removeprefix("trial:")
            assert request == {
                "operation": "FEATURE_TRIAL",
                "feature_plan_hash": plan_hash,
                "task_id": task_id,
            }
            study = studies[task_id]
            assert study["lifecycle"] == "SUCCEEDED"
            assert study["kind"].startswith("portfolio.") or (
                "target_recipe_id" in study and study["input_binding_hash"] == binding
            )
        assert ops.execute(Request(operation="FEATURE_TRIALS"))["trials"] == before


def test_a_formula_factor_is_reviewed_and_a_person_activates_it(completed_alpha_case):
    """requirement (EX, the formula point): a formula factor declared with its recipe is built
    and tried like any feature; its review packet states the contract, the build's coverage, the
    trial's evidence and the formulas tried; an agent is refused the activation, a person
    activates it into the workspace's registry and deactivates it."""
    alpha_case = completed_alpha_case

    from alphalattice.control.product_host.composition.research_workspace import (
        read_research_workspace_manifest,
    )
    from alphalattice.control.product_host.maintenance.data_update import (
        installed_data_update_binding,
    )
    from alphalattice.control.product_host.research_authoring.feature_activations import (
        activated_feature_specs,
        feature_catalog_for,
        workspace_feature_catalog,
    )

    root = alpha_case[0]
    with _session(root) as live:
        ops = live.operations
        alphas = [
            r
            for r in _json(live, "/api/experiments")["experiments"]
            if r["kind"] == "alpha.model-development" and r["lifecycle"] == "SUCCEEDED"
        ]
        if alphas:
            alpha = alphas[0]
        else:
            plan = _json(
                live, "/api/experiments/plan", method="POST", payload=_alpha_payload(alpha_case)
            )
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            alpha = next(
                r
                for r in _json(live, "/api/experiments")["experiments"]
                if r["task_id"] == sent["task_id"]
            )
        controls = ops.execute(
            Request(
                operation="FEATURE_CATALOG_CONTROLS", input_binding_hash=alpha["input_binding_hash"]
            )
        )
        assert controls["formula_language"]["preprocessing_recipes"]
        base = next(
            f for f in controls["registered_formulas"] if ".experimental." in str(f["formula_ref"])
        )
        specification = {
            **base,
            "factor_id": "formula_reversal_5",
            "formula_ref": "factor.formula",
            "formula": "-(close / lag(close, 5) - 1)",
            "lag_sessions": 0,
        }
        document = json.loads(json.dumps(controls["template"]))
        document["edits"] = [
            {
                "operation": "CREATE",
                "factor_id": "formula_reversal_5",
                "specification": specification,
                "preprocessing_recipe": "ROBUST_SECTOR_NEUTRAL_Z",
            }
        ]
        document["reason"] = "A formula factor, reviewed and activated."
        planned = ops.execute(Request(operation="FEATURE_CATALOG_PLAN", feature_document=document))
        plan_hash = planned.get("plan_hash") or planned["plan"]["plan_hash"]
        request = {"feature_plan_hash": plan_hash, "task_id": alpha["task_id"]}
        started = _json(live, "/api/feature-trials", method="POST", payload=request)
        trial_id = started["feature_trial_id"]
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            progress = _drain_trial(live)
            body = _json(live, f"/api/feature-trials/readback?feature_trial_id={trial_id}")
            if body["state"] != "RUNNING":
                break
            progress.wait(max(0.0, deadline - time.monotonic()))
        assert body["state"] == "COMPLETED", (body["stopped"], body["steps"])
        assert body["temporal_scope"]["status"] == "RECORDED"
        # V368: the trial states what it can claim; a person activates from the packet.
        alpha_side = body["comparison"].get("alpha_without_and_with")
        assert body["standing"]["comparison"] == (
            alpha_side["standing"] if alpha_side else "NOT_APPLICABLE"
        )
        assert (body["standing"]["execution"], body["standing"]["activation"]) == (
            "SUCCEEDED",
            "NOT_ACTIVATABLE",
        )

        query = f"feature_plan_hash={plan_hash}&feature_factor_id=formula_reversal_5"
        packet = _json(live, f"/api/feature-research/review?{query}")
        assert (packet["status"], packet["state"]) == ("FEATURE_REVIEW", "NOT_ACTIVE")
        assert (packet["standing"]["contract"], packet["standing"]["activation"]) == (
            "PASSED",
            "A_PERSON_MAY_ACTIVATE",
        )
        assert packet["standing"]["comparison"] == body["standing"]["comparison"]
        assert packet["declaration"]["preprocessing_recipe"] == "ROBUST_SECTOR_NEUTRAL_Z"
        assert packet["contract"]["status"] == "PASSED", packet["contract"]
        assert packet["identity"]["moves"] == []
        assert 0 < packet["build"]["coverage"] <= 1
        assert packet["build"]["missing_share"] == pytest.approx(1 - packet["build"]["coverage"])
        assert [row["feature_trial_id"] for row in packet["trials"]] == [trial_id]
        assert packet["tried"]["workspace_formulas"] >= 1
        assert packet["active_panel"]["admitted"] is True
        activate = packet["next_requests"]["activate"]
        assert activate["operation"] == "FEATURE_ACTIVATE"
        # U56: the listing names the factor with the requests its packet offers.
        listed = _json(live, "/api/feature-research")
        assert listed["status"] == "FEATURE_EXTENSIONS" and listed["page"] == 1
        row = next(
            v
            for v in listed["factors"]
            if (v["feature_plan_hash"], v["factor_id"]) == (plan_hash, "formula_reversal_5")
        )
        assert (row["state"], row["preprocessing_recipe"]) == (
            "NOT_ACTIVE",
            "ROBUST_SECTOR_NEUTRAL_Z",
        )
        assert row["trial"]["feature_trial_id"] == trial_id and row["trial"]["state"] == "COMPLETED"
        assert row["next_requests"]["activate"] == activate
        assert row["next_requests"]["review"] == {
            "operation": "FEATURE_REVIEW",
            "feature_plan_hash": plan_hash,
            "feature_factor_id": "formula_reversal_5",
        }
        assert "refused_plans" not in listed
        healthy_shape = set(listed)

        # A broken unrelated definition must be named beside this readable plan. Exercise the
        # actual on-disk JSON and the stored-name/address identity check through the owner route.
        plan_store = root / "artifacts/feature-panel/closure/research-feature-plans"
        corrupt_hash = "0" * 64 if plan_hash != "0" * 64 else "1" * 64
        mismatch_hash = "f" * 64 if plan_hash != "f" * 64 else "e" * 64
        mismatch_target = "d" * 64 if mismatch_hash != "d" * 64 else "c" * 64
        (plan_store / f"{corrupt_hash}.json").write_bytes(b"{")
        (plan_store / f"{mismatch_target}.json").write_text(
            json.dumps({"plan_hash": mismatch_hash}), encoding="utf-8", newline="\n"
        )
        mixed = _json(live, "/api/feature-research")
        assert set(mixed) == healthy_shape | {"refused_plans"}
        assert (
            next(
                value
                for value in mixed["factors"]
                if (value["feature_plan_hash"], value["factor_id"])
                == (plan_hash, "formula_reversal_5")
            )
            == row
        )
        refusals = {value["feature_plan_hash"]: value for value in mixed["refused_plans"]}
        assert set(refusals) == {corrupt_hash, mismatch_target}
        assert refusals[corrupt_hash]["status"] == "REFUSED"
        assert refusals[corrupt_hash]["failure_code"].startswith(
            "feature_extension.plan_unreadable:"
        )
        assert refusals[mismatch_target]["failure_code"] == (
            "feature_research.plan_reference_mismatch"
        )
        assert refusals[mismatch_target]["next_requests"]["inputs"]["operation"] == (
            "RESEARCH_INPUTS"
        )
        assert (
            "feature controls --binding <selected-binding>" in refusals[mismatch_target]["detail"]
        )

        # V509: the real completed trial under A cannot supply B's review or activation.
        for recipe, same_work in (("ROBUST_UNIVERSE_Z", False), ("ROBUST_SECTOR_NEUTRAL_Z", True)):
            changed = deepcopy(document)
            changed["reason"] = "Review the same formula under an explicit declaration."
            changed["edits"][0]["preprocessing_recipe"] = recipe
            other = ops.execute(Request(operation="FEATURE_CATALOG_PLAN", feature_document=changed))
            other_hash = other.get("plan_hash") or other["plan"]["plan_hash"]
            assert other_hash != plan_hash
            reviewed = ops.execute(
                Request(
                    operation="FEATURE_REVIEW",
                    feature_factor_id="formula_reversal_5",
                    feature_plan_hash=other_hash,
                )
            )
            assert reviewed["declaration"]["preprocessing_recipe"] == recipe
            if same_work:
                assert reviewed["trials"] == packet["trials"]
                assert reviewed["build"] == packet["build"]
                assert reviewed["next_requests"]["activate"]["feature_plan_hash"] == other_hash
            else:
                assert reviewed["trials"] == [] and reviewed["build"] is None
                assert reviewed["standing"]["activation"] == "HELD"
                assert "activate" not in reviewed["next_requests"]
        agent = ops.execute(
            Request(**{**activate, "operation": "FEATURE_ACTIVATE"}), caller="EXTERNAL_AUTOMATION"
        )
        assert agent["failure_code"] == "feature_extension.human_confirmation_required", agent
        assert activated_feature_specs(root) == ()
        payload = {k: v for k, v in activate.items() if k != "operation"}
        activated = _json(live, "/api/feature-research/activate", method="POST", payload=payload)
        assert activated["status"] == "ACTIVATED", activated
        assert activated["activation"]["packet_hash"] == packet["packet_hash"]
        assert [v.factor_id for v in activated_feature_specs(root)] == ["formula_reversal_5"]
        # The daily catalog takes it: the data update binds the catalog the activation made, the
        # manifest is rebound to it, and a Panel under either catalog resolves (part 3c-4).
        made = workspace_feature_catalog(root).binding.catalog_hash
        assert "formula_reversal_5" in workspace_feature_catalog(root).factor_ids
        assert installed_data_update_binding(root).feature_catalog_hash == made
        manifest = read_research_workspace_manifest(root)
        if manifest.data_update is not None:
            assert manifest.data_update.feature_catalog_hash == made
        assert feature_catalog_for(root, made) is not None
        again = _json(live, f"/api/feature-research/review?{query}")
        assert again["state"] == "ACTIVE" and "deactivate" in again["next_requests"]
        assert again["standing"]["activation"] == "ACTIVE"
        listed = _json(live, "/api/feature-research")["factors"]
        row = next(v for v in listed if v["feature_plan_hash"] == plan_hash)
        assert row["state"] == "ACTIVE"
        assert row["next_requests"]["deactivate"] == again["next_requests"]["deactivate"]
        _json(
            live,
            "/api/feature-research/deactivate",
            method="POST",
            payload={"feature_factor_id": "formula_reversal_5"},
        )
        assert activated_feature_specs(root) == ()
        shipped = installed_data_update_binding()
        assert installed_data_update_binding(root) == shipped
        manifest = read_research_workspace_manifest(root)
        if manifest.data_update is not None:
            assert manifest.data_update == shipped


def test_a_compact_prefixed_history_entry_reads_back_through_the_cli(inputs, published, capsys):
    """V512: the printed experiment:<UUID> is accepted by history list --entry unchanged."""
    from run_alphalattice import main

    with _session(inputs):
        assert main(["--workspace", str(inputs), "history", "list"]) == 0
        shown = json.loads(capsys.readouterr().out)["data"]
        entry = next(
            row["entry_id"] for row in shown["entries"] if row["entry_id"].startswith("experiment:")
        )
        assert entry in (f"experiment:{published[0]}", f"experiment:{published[0][:12]}")
        assert (
            main(
                ["--workspace", str(inputs), "--view", "full", "history", "list", "--entry", entry]
            )
            == 0
        )
        selected = json.loads(capsys.readouterr().out)["data"]
        assert selected["entries"][0]["entry_id"] == f"experiment:{published[0]}"


def test_the_study_reads_answer_their_published_models(inputs: Path, published) -> None:
    """contract (binding plan C2, CLI-16): beside a published study, each typed study read's
    answer validates against the model `alphalattice schema show` publishes for it."""

    from alphalattice.interface.local_application.answers import ANSWERS

    root = inputs
    with _session(root) as live:
        bridge = InstalledAgent(live.operations)
        for operation, fields in (
            ("EXPERIMENTS", {}),
            ("EXPERIMENT_READBACK", {"task_id": published[0]}),
            ("RESEARCH_INPUTS", {}),
            ("RESEARCH_HISTORY", {}),
        ):
            body = json.loads(
                bridge.invoke(PortfolioResearchAgentRequest(operation=operation, **fields))
            )
            assert body.get("status") != "REFUSED", (operation, body)
            ANSWERS[operation].model_validate(body)


def test_a_declaration_carries_no_identity_the_host_computes() -> None:
    """requirement (V128, LAWS OP8): an authored ``envelope_hash`` or a foreign ``schema_id``
    is refused at PLAN, before it can shape a managed output path or move an identity."""

    from alphalattice.control.product_host.composition.research_experiments import (
        ENVELOPE_SCHEMA_ID,
        refuse_authored_identity,
    )

    section = {"kind": "factor.screening-development", "schema_id": ENVELOPE_SCHEMA_ID}
    refuse_authored_identity(section)
    with pytest.raises(AuthoringError, match="resolved_identity_authored:envelope_hash"):
        refuse_authored_identity({**section, "envelope_hash": "0" * 64})
    with pytest.raises(AuthoringError, match="schema_id_not_installed"):
        refuse_authored_identity({**section, "schema_id": "research-experiment-envelope-v2"})


def test_an_object_document_is_checked_as_a_yaml_one_is() -> None:
    """requirement (V134): the object entry of PLAN runs the YAML entry's guard, so a document
    that names a module is refused whichever way it arrives."""

    from alphalattice.control.research_program.authoring.document import (
        load_authoring_document,
        require_authoring_document,
    )

    named = {"experiment": {"kind": "risk.covariance-development"}, "risk": {"module": "typo"}}
    with pytest.raises(AuthoringError, match="document_declares_python_path"):
        load_authoring_document(json.dumps(named))
    with pytest.raises(AuthoringError, match="document_declares_python_path"):
        require_authoring_document(named)
    with pytest.raises(AuthoringError, match="document_not_a_mapping"):
        require_authoring_document(["experiment"])
    plain = {"experiment": {"kind": "risk.covariance-development"}, "risk": {"estimator": {}}}
    assert require_authoring_document(plain) is plain


def test_a_continuation_without_its_policy_is_told_the_field_and_the_draft(
    inputs: Path, published
) -> None:
    """regression (V110): a continuation whose declaration lost `screening_policy` was refused
    as not installed with no way on, and one agent stopped there after 18 calls; the refusal
    names the field and what is installed, and offers the study's own draft."""

    with _session(inputs) as session:
        # A study over the input the workspace binds now: the curation test rebinds the shared
        # input, so the published fixture's study need not be draftable any more.
        task_id = _study_on_the_bound_input(session)
        draft = _json(
            session, "/api/experiments/draft", method="POST", payload={"task_id": task_id}
        )
        assert draft["status"] == "DRAFT_READY", draft
        document = draft["document"]
        del document["factor"]["screening_policy"]
        request = {key: value for key, value in draft["plan_request"].items() if key != "operation"}
        refused = _json(
            session,
            "/api/experiments/plan",
            method="POST",
            payload={**request, "experiment_document": document},
        )
    assert refused["failure_code"] == "factor_research.authoring_screening_policy_not_installed"
    assert "`factor.screening_policy`" in refused["detail"]
    assert "`BENJAMINI_YEKUTIELI_FDR`" in refused["detail"]
    assert refused["next_requests"] == {
        "draft": {"operation": "EXPERIMENT_DRAFT", "task_id": task_id}
    }


def test_a_qualification_concludes_every_study_on_its_question_since_the_goal_opened(alpha_case):
    """requirement (GR3, V77): the Alpha owner's qualification is a Task over every development
    study on the question admitted from a goal's opening on, whether a goal's session ran it or
    not: the Host finds the family in Task Control, the Task refits what it nominates, applies
    the benchmark and the stability rule, and seals a candidate set or an evidence-complete stop
    that its readback reads at the Alpha owner's store. No stable model is a result."""

    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )

    root, _binding, _factor_task, _decision, _original = alpha_case
    with _session(root) as live:
        opened = live.operations.goals.operate(
            PortfolioResearchOperationRequest(
                operation="GOAL_OPEN",
                goal_declaration={
                    "title": "Ridge penalties on the fixture",
                    "objective": "Does a ridge Alpha survive its whole family?",
                    "kind": "RESEARCH",
                    "scope": "The fixture's bound input and curated Foundation",
                    "criteria": [
                        {"criterion_id": "qualified", "text": "A qualification ends the family."}
                    ],
                    "deliverables": [
                        {
                            "deliverable_id": "qualification",
                            "kind": "RESULT",
                            "description": "The qualification's sealed end",
                        }
                    ],
                    "research": {
                        "purpose": "NEW_RESEARCH",
                        "comparison_design": "Two ridge penalties, one input and target",
                        "required_stages": ["ALPHA"],
                    },
                },
                change_reason="Register before the studies",
            ),
            "HUMAN",
        )
        studies: list[str] = []
        candidates: list[str] = []
        for penalty in (5.0, 50.0):
            payload = _alpha_payload(alpha_case, {"family": "ridge", "alpha": penalty})
            plan = _json(live, "/api/experiments/plan", method="POST", payload=payload)
            assert plan["status"] == "PLANNED", plan
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            assert sent["task_id"], sent  # a study admitted after the goal opened
            live.dispatcher.drain_for_tests()
            body = _json(live, f"/api/experiments/readback?task_id={sent['task_id']}")
            assert body["status"] == "EXPERIMENT_PUBLISHED", body
            studies.append(sent["task_id"])
            candidates.append(body["result"]["candidates"][0]["candidate_id"])
        document = {
            "experiment": {
                "kind": "alpha.model-development",
                "schema_id": "research-experiment-envelope",
            },
            "alpha": {
                "methodology_id": "ALPHA_FAMILY_QUALIFICATION",
                "goal_id": opened["goal_id"],
                "question_task_id": studies[0],
                "nominated_candidate_ids": candidates,
            },
        }
        plan = _json(
            live, "/api/experiments/plan", method="POST", payload={"experiment_document": document}
        )
        assert plan["status"] == "PLANNED", plan
        preview = plan["execution_preview"]
        assert preview["study_count"] == 2 and preview["attempted_candidate_ids"] == candidates
        assert preview["sealed_holdout"] == "UNREAD"
        sent = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": plan["plan_hash"]},
        )
        assert sent["task_id"], sent
        live.dispatcher.drain_for_tests()
        from uuid import UUID

        task = live.session.task_control_registry.task(UUID(sent["task_id"]))
        assert task.lifecycle.value == "SUCCEEDED", (
            task.lifecycle,
            task.failure_code,
            live.dispatcher.failure(task.task_id),
        )
        body = _json(live, f"/api/experiments/readback?task_id={sent['task_id']}")
        assert body["status"] == "EXPERIMENT_PUBLISHED", body
        sealed = body["alpha_qualification"]
        assert sealed["attempted_candidate_ids"] == candidates
        assert [m["task_id"] for m in body["qualification_family"]["members"]] == studies
        if sealed["disposition"] == "CURRENT_ALPHA_CANDIDATE_SET_READY":
            assert sealed["candidate_set_hash"] and not sealed["scientific_stop_hash"]
            assert set(sealed["selected_candidate_ids"]) <= set(sealed["current_qualified_ids"])
        else:
            assert sealed["scientific_stop_hash"] and not sealed["candidate_set_hash"]
            assert sealed["selected_candidate_ids"] == []
        # A study on the question after the qualification planned changes its family: the plan
        # it sealed no longer runs.
        payload = _alpha_payload(alpha_case, {"family": "ridge", "alpha": 12.0})
        later = _json(live, "/api/experiments/plan", method="POST", payload=payload)
        again = _json(
            live, "/api/experiments/plan", method="POST", payload={"experiment_document": document}
        )
        assert again["plan_hash"] == plan["plan_hash"]
        ran = _json(
            live,
            "/api/experiments/run",
            method="POST",
            payload={"experiment_plan_hash": later["plan_hash"]},
        )
        live.dispatcher.drain_for_tests()
        assert ran["task_id"]
        grown = _json(
            live, "/api/experiments/plan", method="POST", payload={"experiment_document": document}
        )
        assert grown["plan_hash"] != plan["plan_hash"]
        assert grown["execution_preview"]["study_count"] == 3


@pytest.fixture(scope="module")
def risk_seed(completed_alpha_seed, tmp_path_factory):
    """One real Risk prerequisite over the already published Alpha study."""
    from urllib.parse import urlencode

    def build(root: Path) -> dict:
        alpha_case = completed_alpha_seed
        copy_workspace(alpha_case[0], root)
        with _session(root) as live:
            parent_id = next(
                row["task_id"]
                for row in _json(live, "/api/experiments")["experiments"]
                if row["kind"] == "alpha.model-development" and row["lifecycle"] == "SUCCEEDED"
            )
            alpha = _json(live, f"/api/experiments/readback?task_id={parent_id}")
            assert alpha["status"] == "EXPERIMENT_PUBLISHED", alpha
            draft = _json(
                live,
                "/api/experiments/portfolio-draft",
                method="POST",
                payload={
                    "task_id": parent_id,
                    "candidate_id": alpha["result"]["candidates"][0]["candidate_id"],
                },
            )
            assert draft["status"] == "PORTFOLIO_DRAFT_READY", draft
            document = draft["document"]
            window = document["experiment"]["sessions"]

            binding = alpha_case[1]
            selection = {
                "research_input_id": binding.input_id,
                "input_binding_hash": binding.binding_hash,
            }
            controls = _json(
                live,
                "/api/experiments/controls?"
                + urlencode({**selection, "experiment_kind": "risk.covariance-development"}),
            )
            assert controls["status"] == "READY", controls
            risk_document = controls["template"]
            risk_document["experiment"]["sessions"].update(start=window["start"], end=window["end"])
            risk_document["experiment"]["sessions"]["as_of"] = deepcopy(window["as_of"])
            risk_plan = _json(
                live,
                "/api/experiments/plan",
                method="POST",
                payload={**selection, "experiment_document": risk_document},
            )
            assert risk_plan["status"] == "PLANNED", risk_plan
            risk_sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": risk_plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests(timeout=300)
            risk_task = risk_sent.get("task_id") or risk_sent["publication_task_id"]
            assert (
                _json(live, f"/api/experiments/readback?task_id={risk_task}")["status"]
                == "EXPERIMENT_PUBLISHED"
            )
            return {"document": document, "risk_task": risk_task}

    return session_workspace(tmp_path_factory, "risk_link", build)


@pytest.fixture
def risk_link(risk_seed, tmp_path_factory):
    """Each consumer opens its own writable copy and Host."""
    from types import SimpleNamespace

    source, metadata = risk_seed
    root = copy_workspace(
        source, tmp_path_factory.mktemp("risk-link") / "workspace", link_parquet=True
    )
    with _session(root) as live:
        yield SimpleNamespace(
            live=live, document=deepcopy(metadata["document"]), risk_task=metadata["risk_task"]
        )


def _planned(link, **portfolio):
    """The draft's Portfolio declaration with ``portfolio`` changed, planned."""

    from copy import deepcopy

    request = deepcopy(link.document)
    request["portfolio"].update(portfolio)
    return _json(
        link.live, "/api/experiments/plan", method="POST", payload={"experiment_document": request}
    )


def _published(link, plan):
    """A planned Portfolio study, run and read back."""

    run = _json(
        link.live,
        "/api/experiments/run",
        method="POST",
        payload={"experiment_plan_hash": plan["plan_hash"]},
        timeout=180,
    )
    link.live.dispatcher.drain_for_tests()
    task_id = run.get("task_id") or run["publication_task_id"]
    body = _json(link.live, f"/api/experiments/readback?task_id={task_id}")
    assert body["status"] == "EXPERIMENT_PUBLISHED", body
    return body


def test_a_portfolio_study_weighs_by_a_linked_risk_studys_volatility(risk_link):
    """requirement (V310, card RP): the Portfolio experiment took no Risk input, so its
    inverse-volatility rules could not run; it names a completed Risk study on the same research
    input, binds it as it binds the Alpha source, and weighs each name by the study's per-name
    volatility, while equal weight reads none and an `iv` rule without a study is refused, its
    way back the Alpha study's Portfolio draft (V322)."""

    link = risk_link
    refused = _planned(link, weight_rule="iv1")
    assert refused["failure_code"] == "portfolio_research.risk_study_required", refused
    source = link.document["portfolio"]
    assert refused["next_requests"] == {
        "portfolio_draft": {
            "operation": "EXPERIMENT_PORTFOLIO_DRAFT",
            "task_id": source["alpha_task_id"],
            "candidate_id": source["candidate_id"],
        }
    }, refused
    unread = _planned(link, risk_task_id=link.risk_task)
    assert unread["failure_code"] == "portfolio_research.risk_study_unread", unread
    weighed = _planned(link, weight_rule="iv1", risk_task_id=link.risk_task)
    assert weighed["status"] == "PLANNED", weighed
    assert (
        weighed["execution_preview"]["risk_disposition"]
        == "INVERSE_VOLATILITY_FROM_THE_LINKED_RISK_STUDY"
    )
    body = _published(link, weighed)
    source = body["portfolio_source"]
    assert source["risk_task_id"] == link.risk_task and source["risk_surface_hash"]
    equal = _published(link, _planned(link))
    assert "risk_surface_hash" not in equal["portfolio_source"]
    # The same names on the same sessions, weighed by inverse volatility rather than equally:
    # the held books differ, and no equal book is more concentrated than its iv1 twin.
    assert body["position"]["weights"] != equal["position"]["weights"]
    assert body["position"]["hhi"] > equal["position"]["hhi"]


def test_a_portfolio_study_runs_a_catalog_policy_on_a_linked_risk_studys_covariance(risk_link):
    """requirement (V310, card RP): the catalog's solver-backed policies ran on nothing once the
    Stage 6 campaign retired; a Portfolio study declares one (the kept `PortfolioPolicySpec`) and
    decides each scored formation on the linked Risk study's covariance, projected one segment at
    a time, while the tranche book's own fields stay unset and a declaration that cannot run is
    refused before any work, pointing back to the Alpha study's Portfolio draft with the bound
    it broke (V322); the policy's document leaves out the tranche fields it replaces (V323)."""

    link, task = risk_link, risk_link.risk_task
    minimum = {"family": "TOP_K_MINIMUM_VARIANCE", "top_k": 20, "maximum_weight": 0.1}
    conflict = _planned(link, policy=minimum, weight_rule="iv1", risk_task_id=task)
    assert conflict["failure_code"] == "portfolio_research.policy_book_conflict", conflict
    assert conflict["fields"] == [["portfolio", "policy"]]
    draft = conflict["next_requests"]["portfolio_draft"]
    assert draft["operation"] == "EXPERIMENT_PORTFOLIO_DRAFT"
    assert draft["task_id"] == link.document["portfolio"]["alpha_task_id"]
    unlinked = _planned(link, policy=minimum)
    assert unlinked["failure_code"] == "portfolio_research.risk_study_required", unlinked
    wide = _planned(link, policy={**minimum, "top_k": 100_000}, risk_task_id=task)
    assert wide["failure_code"] == "portfolio_research.top_k_exceeds_universe", wide
    assert 0 < wide["expected"]["portfolio.policy.top_k"]["maximum"] < 100_000, wide
    thin = _planned(link, policy={**minimum, "maximum_weight": 0.01}, risk_task_id=task)
    assert thin["failure_code"] == "portfolio_research.policy_cap_infeasible", thin

    plan = _planned(link, policy=minimum, risk_task_id=task)
    assert plan["status"] == "PLANNED", plan
    preview = plan["execution_preview"]
    assert preview["risk_disposition"] == "COVARIANCE_FROM_THE_LINKED_RISK_STUDY"
    assert preview["solver_calls"] == preview["score_session_count"]
    variance = _published(link, plan)
    assert variance["receipt"]["spec"]["policy"]["family"] == "TOP_K_MINIMUM_VARIANCE"
    tranche = {"top_k", "tranches", "exit_rank", "weight_rule"}
    assert not tranche & set(variance["receipt"]["spec"]), variance["receipt"]["spec"]
    held = [w for w in variance["position"]["weights"] if w > 1e-9]
    assert 0 < len(held) <= 20 and max(held) <= 0.1 + 1e-6
    costed = _published(
        link,
        _planned(
            link,
            policy={
                "family": "TOP_K_SCORE_RISK_COST",
                "top_k": 20,
                "maximum_weight": 0.1,
                "risk_aversion": 10.0,
                "turnover_regularization": 0.001,
            },
            risk_task_id=task,
        ),
    )
    # Two objectives on one covariance and one score lane hold two books.
    assert costed["position"]["weights"] != variance["position"]["weights"]


def test_the_portfolio_draft_offers_the_weight_rules_risk_studies_and_policies(risk_link):
    """requirement (V338, U41, U42): the draft offers the weight rule, the completed Risk
    studies on its research input and the catalog's families, each family's fields shown by
    `when` with the catalog's own bounds, so the Lab authors what PLAN admits without YAML."""

    link = risk_link
    source = link.document["portfolio"]
    draft = _json(
        link.live,
        "/api/experiments/portfolio-draft",
        method="POST",
        payload={"task_id": source["alpha_task_id"], "candidate_id": source["candidate_id"]},
    )
    assert draft["status"] == "PORTFOLIO_DRAFT_READY", draft
    controls = {tuple(value["path"]): value for value in draft["controls"]}
    assert controls[("portfolio", "weight_rule")]["options"] == ["ew", "iv1", "iv2"]
    assert controls[("portfolio", "risk_task_id")]["options"] == [link.risk_task]
    families = controls[("portfolio", "policy", "family")]["options"]
    assert families == [
        "TOP_K_EQUAL_WEIGHT",
        "TOP_K_MINIMUM_VARIANCE",
        "TOP_K_SCORE_RISK_COST",
        "SECTOR_DEVIATION_PENALTY",
    ]
    top_k = controls[("portfolio", "policy", "top_k")]
    assert top_k["when"] == [{"path": ["portfolio", "policy", "family"], "values": families}]
    assert (top_k["min"], top_k["step"]) == (1, 1) and top_k["max"] >= 1
    weight = controls[("portfolio", "policy", "maximum_weight")]
    assert (weight["min"], weight["min_exclusive"], weight["max"], weight["step"]) == (
        0.0,
        True,
        1.0,
        "any",
    )
    penalty = controls[("portfolio", "policy", "sector_deviation_penalty")]
    assert penalty["when"][0]["values"] == ["SECTOR_DEVIATION_PENALTY"]


def test_a_portfolio_draft_offers_only_risk_choices_its_reader_admits(
    risk_seed, tmp_path, monkeypatch
):
    """regression (V616's class): a scoped or incomplete Risk read is not offered for
    sizing; an integrity failure remains a named refusal, rather than disappearing."""
    import gc

    source, metadata = risk_seed
    root = copy_workspace(source, tmp_path / "workspace", link_parquet=True)
    try:
        with _session(root) as live:
            experiments = live.operations.experiments
            original_read = experiments.readback
            risk_task = metadata["risk_task"]
            source_portfolio = metadata["document"]["portfolio"]
            payload = {
                "task_id": source_portfolio["alpha_task_id"],
                "candidate_id": source_portfolio["candidate_id"],
            }
            task_count = len(live.session.task_control_registry.tasks())

            for cause in ("accepted", "scoped", "missing_input", "incomplete", "integrity"):

                def read(task_id, *args, _cause=cause, **kwargs):
                    if str(task_id) != risk_task:
                        return original_read(task_id, *args, **kwargs)
                    if _cause == "integrity":
                        raise AuthoringError("risk_research.covariance_chunk_identity_invalid")
                    body = deepcopy(original_read(task_id, *args, **kwargs))
                    if _cause == "scoped":
                        body["risk_surface"]["scope_surfaces"] = [body["risk_surface"].copy()]
                    elif _cause == "missing_input":
                        body.pop("risk_input")
                    elif _cause == "incomplete":
                        body["status"] = "BLOCKED"
                    return body

                # A public verified-reader port supplies these metadata cases; no
                # numerical artifact, scientific bound or stored result changes.
                monkeypatch.setattr(experiments, "readback", read)
                draft = _json(
                    live, "/api/experiments/portfolio-draft", method="POST", payload=payload
                )
                if cause == "integrity":
                    assert draft["status"] == "REFUSED", draft
                    assert (
                        draft["failure_code"] == "risk_research.covariance_chunk_identity_invalid"
                    )
                else:
                    assert draft["status"] == "PORTFOLIO_DRAFT_READY", draft
                    control = next(
                        row
                        for row in draft["controls"]
                        if row["path"] == ["portfolio", "risk_task_id"]
                    )
                    assert control["options"] == ([risk_task] if cause == "accepted" else [])
                assert len(live.session.task_control_registry.tasks()) == task_count
    finally:
        gc.collect()
        if root.exists():
            assert root.resolve().is_relative_to(tmp_path.resolve())
            shutil.rmtree(root)


def test_the_kept_backups_are_read_without_making_one(tmp_path: Path):
    """requirement (V337, U48): the Storage page reads the backup root and the kept generations
    at GET /api/workspace/backup (`workspace backups`), and reading makes none."""

    root = tmp_path / "workspace"
    # This read uses the manifest and backup directory, not any study or market rows.
    publish_research_workspace_manifest(root, _manifest("backup-read"))
    with _session(root) as live:
        first = _json(live, "/api/workspace/backup")
        assert (first["status"], first["generation_hash"]) == ("READ", None), first
        assert first["backup_root"] and isinstance(first["generations"], list)
        again = _json(live, "/api/workspace/backup")
        assert again["generations"] == first["generations"]


def stop_child_for_cli_start_check(*, cancelled):
    os._exit(7)


def test_cli_wait_reads_a_real_alpha_task_whose_child_cannot_start(
    alpha_case, monkeypatch, tmp_path, capsys
):
    """V610/OP4/OP18: inject at OS child start, run the real CLI, read stop and legal recovery."""
    import multiprocessing

    from alphalattice.control.task_control.child import ChildInterrupted, run_in_child
    from run_alphalattice import main

    root = alpha_case[0]
    with _session(root) as live:
        payload = _alpha_payload(alpha_case, parameters={"family": "ridge", "alpha": 1.61})
        plan = _json(live, "/api/experiments/plan", method="POST", payload=payload)
        assert plan["status"] == "PLANNED", plan
        with pytest.raises(ChildInterrupted):
            run_in_child(f"{__name__}:stop_child_for_cli_start_check", {})
        request = tmp_path / "run.json"
        request.write_text(
            json.dumps({"operation": "EXPERIMENT_RUN", "experiment_plan_hash": plan["plan_hash"]}),
            encoding="utf-8",
        )
        attempts = []

        def cannot_start(process):
            attempts.append(process.name)
            raise OSError(1455, "fixture paging file too small")

        started = time.monotonic()
        with monkeypatch.context() as patch:
            patch.setattr(multiprocessing.get_context("spawn").Process, "start", cannot_start)
            code = main(
                [
                    "--workspace",
                    str(root),
                    "--view",
                    "full",
                    "request",
                    "--file",
                    str(request),
                    "--wait",
                    "--max-wait",
                    "30",
                ]
            )
        output = capsys.readouterr().out
        answer = json.loads(output)
        assert attempts and time.monotonic() - started < 30
        assert (code, answer["outcome"]) == (2, "REFUSED"), answer
        assert answer["failure_code"] == "task_control.child_start_failed", answer
        assert "operating system" in answer["detail"] and answer["next_requests"]
        assert "fixture paging file" not in output and "MAX_WAIT" not in output
        task_id = answer["data"]["task_id"]
        assert main(["--workspace", str(root), "--view", "full", "task", "show", task_id]) == 2
        shown = json.loads(capsys.readouterr().out)
        assert shown["data"]["lifecycle"] == "RECOVERY_REQUIRED"
        assert shown["data"]["latest_failure_code"] == "task_control.child_start_failed"
        request.write_text(json.dumps(answer["next_requests"]["recovery"]), encoding="utf-8")
        assert (
            main(["--workspace", str(root), "--view", "full", "request", "--file", str(request)])
            == 3
        )
        recovery = json.loads(capsys.readouterr().out)["data"]
        assert recovery["stop"]["code"] == "task_control.child_start_failed"
        assert "operating system" in recovery["stop"]["detail"]
        assert any(a["action"] == "RECOVER" and a["available"] for a in recovery["actions"])
