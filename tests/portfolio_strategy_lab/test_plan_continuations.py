"""Verified expired plans offer their exact next declaration, never a default."""

import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from alphalattice.control.product_host.composition import (
    decision_advancement,
    portfolio_updates,
    strategy_calibration,
    strategy_scoring,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceCalibrationInput,
    ResearchWorkspaceManifest,
    ResearchWorkspaceScoreInput,
)
from alphalattice.control.product_host.data_preparation import (
    feature_research,
    input_capture,
    model_training,
    research_strategy,
)
from alphalattice.control.product_host.research_authoring.frozen_portfolio import (
    FrozenPortfolioPreparationRequest,
)
from alphalattice.control.product_host.research_authoring.input_revisions import ResearchInputSource
from alphalattice.control.product_host.storage.plan_previews import PreviewRegistry
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.interface.local_application.answers import continuation_problem
from alphalattice.interface.local_application.cli_contract import command_table
from alphalattice.interface.local_application.client import continued
from alphalattice.interface.local_application.dispatcher import CommandSubmission
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)


def test_an_expired_capture_plan_replans_its_input_or_names_an_unavailable_source(
    tmp_path, monkeypatch
):
    """regression (S3): expiry forbids the old run, while a verified plan's input
    still binds its re-plan. A missing or tampered record never becomes an anchor input."""
    plan = input_capture.ResearchInputCapturePlan.create(
        workspace_id="s1-workspace",
        workspace_manifest_hash="a" * 64,
        input_id="s1-input",
        anchor_binding_hash="b" * 64,
        prior_binding_hash="c" * 64,
        previous_publication_hash=None,
        source=ResearchInputSource(
            binding_hash="d" * 64,
            manifest_revision="source-revision",
            data_revision_hash="e" * 64,
            panel_snapshot_hash="f" * 64,
            panel_through=date(2026, 9, 30),
            outcome_watermark_hash="1" * 64,
            outcome_recipe_hash="2" * 64,
            outcome_policy_hash="3" * 64,
            materializer_hash="4" * 64,
        ),
        implementation_hash="5" * 64,
    )
    _exercise_replan(
        tmp_path,
        monkeypatch,
        input_capture,
        plan,
        lambda session, clock: input_capture.ResearchInputCaptureApplication(session, clock=clock),
        {"replan": {"operation": "RESEARCH_INPUT_PLAN", "research_input_id": plan.input_id}},
        "s1-input",
        input_capture._task_contract,
    )


def _exercise_replan(tmp_path, monkeypatch, owner, plan, make_app, expected, tamper, task_contract):
    """Use the real sealed store and public owner/client boundary; run no study."""
    now = datetime(2026, 10, 2, tzinfo=UTC)
    folder = tmp_path / "plans"
    PreviewRegistry(model=type(plan), clock=lambda: now, root=folder).remember(plan)

    def clock():
        return now + timedelta(hours=2)

    def registry(**_kwargs):
        return PreviewRegistry(model=type(plan), clock=clock, root=folder)

    monkeypatch.setattr(owner, "PreviewRegistry", registry)
    session = SimpleNamespace(workspace=tmp_path)
    app = make_app(session, clock)
    assert registry().runnable(plan.plan_hash) is None
    offers = app.replan_requests(plan.plan_hash)
    assert offers == expected
    assert continuation_problem({"next_requests": offers}) is None
    for offer in offers.values():
        operation = offer["operation"]
        request = continued(
            operation,
            {"next_requests": {"selected": offer}},
            {},
            frozenset(command_table()["fields"][operation]["allowed"]),
        )
        PortfolioResearchRequestDocument.model_validate(request)
        assert request == offer
    assert app.replan_requests("0" * 64) == {}
    path = folder / f"{plan.plan_hash}.json"
    assert tamper in path.read_text(encoding="utf-8")
    path.write_text(
        path.read_text(encoding="utf-8").replace(tamper, "another-source"),
        encoding="utf-8",
        newline="\n",
    )
    restarted = make_app(session, clock)
    assert restarted.replan_requests(plan.plan_hash) == {}
    task = _stopped_task(task_contract(plan))
    expected_request = next(iter(expected.values()))
    _assert_request(restarted.replan_request(task), expected_request)
    path.unlink()
    assert restarted.replan_requests(plan.plan_hash) == {}
    _assert_request(restarted.replan_request(task), expected_request)


def _stopped_task(contract):
    """A validated sealed Task retained after its ledger was rebuilt; no numerical run."""
    envelope, goal, workflow = contract
    now = datetime(2026, 10, 2, tzinfo=UTC)
    return TaskRecord.from_identity(
        task_id=uuid4(),
        task_kind=envelope.task_kind,
        input=envelope,
        goal=goal,
        plan=workflow,
        lifecycle=TaskLifecycle.BLOCKED,
        active_work_item_id=None,
        latest_execution_id=None,
        admitted_at=now,
        started_at=None,
        updated_at=now,
        failure_code="task_control.ledger_rebuilt",
        version=1,
    )


def _assert_request(request, expected):
    """The durable declaration crosses the same saved-answer client boundary unchanged."""
    assert request == expected
    offered = {"next_requests": {"replan": request}}
    assert continuation_problem(offered) is None
    operation = request["operation"]
    continued_request = continued(
        operation,
        offered,
        {},
        frozenset(command_table()["fields"][operation]["allowed"]),
    )
    PortfolioResearchRequestDocument.model_validate(continued_request)
    assert continued_request == expected


def test_an_expired_training_plan_keeps_its_component_input_and_revision(tmp_path, monkeypatch):
    """regression (S3): re-planning names the same training component and exact input;
    expiry still forbids running, and unavailable plans offer no guessed source."""
    plan = model_training.ModelTrainingInputPlan.create(
        workspace_id="s1-workspace",
        workspace_manifest_hash="a" * 64,
        input_id="s1-input",
        input_binding_hash="b" * 64,
        component_ids=("G2_R0_TREND",),
        listing_count=50,
        source_session_count=252,
        implementation_hash="c" * 64,
    )
    _exercise_replan(
        tmp_path,
        monkeypatch,
        model_training,
        plan,
        lambda session, clock: model_training.ModelTrainingInputApplication(session, clock=clock),
        {
            "replan:G2_R0_TREND": {
                "operation": "MODEL_TRAINING_INPUT_PLAN",
                "component_id": "G2_R0_TREND",
                "research_input_id": plan.input_id,
                "input_binding_hash": plan.input_binding_hash,
                # A plan made before the light default names its full lifecycle.
                "model_lifecycle": "FULL",
            }
        },
        "s1-input",
        lambda value: model_training._task_contract(value, "INSTALLED_AGENT"),
    )


def test_an_expired_strategy_plan_keeps_its_parents_and_policy(tmp_path, monkeypatch):
    """regression (S3): a re-plan retains the authored Alpha/Risk parents and policy;
    unavailable or tampered declarations do not silently select other studies."""
    declaration = FrozenPortfolioPreparationRequest(
        input_binding_hash="b" * 64,
        alpha_task_ids=(UUID("00000000-0000-0000-0000-000000000001"),),
        risk_task_id=UUID("00000000-0000-0000-0000-000000000002"),
        unavailable_return_policy="quarantine_listings",
    )
    plan = research_strategy.ResearchStrategyPlan.create(
        workspace_manifest_hash="a" * 64,
        request=declaration,
        implementation_hash="c" * 64,
    )
    _exercise_replan(
        tmp_path,
        monkeypatch,
        research_strategy,
        plan,
        lambda session, clock: research_strategy.ResearchStrategyPreparation(
            session,
            clock=clock,
            read_experiment=lambda _task: {},
            list_experiments=lambda: {"experiments": []},
        ),
        {
            "replan": {
                "operation": "RESEARCH_STRATEGY_PLAN",
                "experiment_document": declaration.model_dump(mode="json"),
            }
        },
        declaration.input_binding_hash,
        lambda value: research_strategy._contract(value, "INSTALLED_AGENT"),
    )


@pytest.mark.parametrize("component", [None, "G2_R0_TREND"])
def test_an_expired_score_plan_keeps_its_package_component_and_session(
    tmp_path, monkeypatch, component
):
    """regression (S3): a re-plan keeps the exact package, declared component and
    formation; it does not substitute a default package or the latest session."""
    binding = ResearchWorkspaceScoreInput(
        strategy_package_id="s1-package",
        strategy_package_hash="b" * 64,
        component_id=component,
        authority_relative_path="artifacts/s1-authority",
        authority_hash="c" * 64,
        source_kind="WORKSPACE_DATA_FEATURE",
    )
    plan = strategy_scoring.StrategyScorePlan.create(
        workspace_id="s1-workspace",
        workspace_manifest_hash="a" * 64,
        binding=binding,
        formation_session=date(2026, 9, 30),
        source_identity_hash="d" * 64,
        implementation_hash="e" * 64,
    )
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="s1-workspace",
        default_strategy_package_id=None,
        default_score_source_mode=None,
        strategy_artifacts=(),
        strategy_installation="NOT_INSTALLED",
    )
    _exercise_replan(
        tmp_path,
        monkeypatch,
        strategy_scoring,
        plan,
        lambda session, clock: strategy_scoring.StrategyScoringApplication(
            session=session, manifest=manifest, packages=dict, clock=clock
        ),
        {
            "replan": {
                "operation": "STRATEGY_SCORE_PLAN",
                "strategy_package_id": binding.strategy_package_id,
                "formation_session": plan.formation_session.isoformat(),
                **({"component_id": component} if component is not None else {}),
            }
        },
        binding.strategy_package_id,
        strategy_scoring._task_contract,
    )


def test_an_expired_calibration_plan_keeps_its_package_and_score(tmp_path, monkeypatch):
    """regression (S3): a re-plan keeps the same package and exact published score;
    an unavailable retained source cannot choose a newer score or a default package."""
    binding = ResearchWorkspaceCalibrationInput(
        strategy_package_id="s1-package",
        strategy_package_hash="b" * 64,
        seed_hash="c" * 64,
        source_kind="WORKSPACE_DATA_FEATURE",
    )
    plan = strategy_calibration.CalibrationPlan.create(
        workspace_manifest_hash="a" * 64,
        binding=binding,
        score_snapshot_hash="d" * 64,
        history_score_hashes=("e" * 64,),
        source_hash="f" * 64,
        implementation_hash="1" * 64,
    )
    _exercise_replan(
        tmp_path,
        monkeypatch,
        strategy_calibration,
        plan,
        lambda session, clock: strategy_calibration.StrategyCalibrationApplication(
            scoring=SimpleNamespace(session=session), clock=clock
        ),
        {
            "replan": {
                "operation": "STRATEGY_CALIBRATION_PLAN",
                "strategy_package_id": binding.strategy_package_id,
                "score_snapshot_hash": plan.score_snapshot_hash,
            }
        },
        binding.strategy_package_id,
        strategy_calibration._task_contract,
    )


@pytest.mark.parametrize("requested_input", [None, "b" * 64])
def test_a_stopped_portfolio_update_retains_the_requested_input_not_its_resolved_input(
    tmp_path, requested_input
):
    """regression (TE12): a stop re-plans the person's input choice and observation end;
    a resolved input or newer workspace default cannot silently replace it."""
    session = SimpleNamespace(workspace=tmp_path)
    calibration = SimpleNamespace(scoring=SimpleNamespace(session=session))
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="fixture",
        default_strategy_package_id=None,
        default_score_source_mode=None,
        strategy_artifacts=(),
        strategy_installation="NOT_INSTALLED",
    )
    app = portfolio_updates.PortfolioUpdateApplication(
        application=SimpleNamespace(session=session, ledger=None),
        calibration=calibration,
        manifest=manifest,
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
    )
    plan = portfolio_updates.PortfolioUpdatePlan.create(
        strategy_package_id="chosen-package",
        workspace_manifest_hash="a" * 64,
        catalog_hash="a" * 64,
        checkpoint_hash="a" * 64,
        parent_hash="a" * 64,
        prepared_input_hash="c" * 64,
        requested_input_hash=requested_input,
        observed_through=date(2026, 9, 30),
        source_hash="a" * 64,
        market_hash="a" * 64,
        implementation_hash="a" * 64,
    )
    task = _stopped_task(portfolio_updates._task_contract(plan))
    _assert_request(
        app.replan_request(task),
        {
            "operation": "PORTFOLIO_UPDATE_PLAN",
            "strategy_package_id": "chosen-package",
            "prepared_input_hash": requested_input,
            "observed_through": "2026-09-30",
        },
    )


def test_a_stopped_research_update_retains_the_selected_package_and_target(tmp_path):
    """regression (TE12): the durable update target survives a stop without any current
    scoring, calibration, data or checkpoint admission being required for its next plan."""
    from alphalattice.control.data_platform.maintenance.contracts import (
        MaintenanceTrigger,
        WorkspaceDataUpdateBinding,
        WorkspaceDataUpdatePlan,
        WorkspaceInputStatus,
        WorkspaceMaintenanceRequest,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    target = date(2026, 9, 30)
    fixture = "a" * 64
    binding = WorkspaceDataUpdateBinding.seal(
        profile_file_hash=fixture,
        data_policy_hash=fixture,
        feature_catalog_hash=fixture,
    )
    before = WorkspaceInputStatus.seal(
        manifest_revision="r1",
        data_revision_hash=fixture,
        data_through=target,
        adjusted_through=target,
        panel_hash=fixture,
        panel_through=target,
        readiness_status="READY",
        sources_checked_at=None,
        foundation_hash=None,
        foundation_panel_hash=None,
        foundation_disposition="MISSING",
    )
    request = WorkspaceMaintenanceRequest.create(
        market_profile_id=binding.market_profile_id,
        target_market_session=target,
        knowledge_cutoff_at=None,
        trigger=MaintenanceTrigger.USER_REQUEST,
        membership_revision="r1",
        data_policy_hash=fixture,
        feature_policy_hash=canonical_hash({"catalog": fixture, "invalidation": "domain-topology"}),
        requested_at=datetime(2026, 10, 2, tzinfo=UTC),
    )
    plan = decision_advancement.DecisionAdvancementPlan.create(
        workspace_manifest_hash=fixture,
        catalog_hash=fixture,
        package_id="chosen-package",
        score_binding=ResearchWorkspaceScoreInput(
            strategy_package_id="chosen-package",
            strategy_package_hash=fixture,
            authority_relative_path="authority/fixture.json",
            authority_hash=fixture,
            source_kind="WORKSPACE_DATA_FEATURE",
        ),
        checkpoint_hash=fixture,
        parent_hash=None,
        target=target,
        decision_sessions=(target,),
        score_sessions=(target,),
        history_score_hashes=(),
        data_plan=WorkspaceDataUpdatePlan.seal(
            workspace_id="fixture",
            workspace_manifest_hash=fixture,
            binding=binding,
            before=before,
            request=request,
        ),
        implementation_hash=fixture,
        score_implementation_hash=fixture,
        calibration_implementation_hash=fixture,
        decision_implementation_hash=fixture,
    )
    session = SimpleNamespace(workspace=tmp_path)
    app = decision_advancement.DecisionAdvancementApplication(
        SimpleNamespace(
            session=session,
            clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
            calibration=SimpleNamespace(scoring=SimpleNamespace()),
        ),
        SimpleNamespace(),
    )
    task = _stopped_task(decision_advancement.task_contract(plan))
    _assert_request(
        app.replan_request(task),
        {
            "operation": "RESEARCH_UPDATE_PLAN",
            "strategy_package_id": "chosen-package",
            "observed_through": "2026-09-30",
        },
    )


@pytest.mark.parametrize("preprocess", [False, True])
def test_a_stopped_feature_build_retains_its_authored_definition(tmp_path, preprocess):
    """regression (TE12): raw and prepared builds re-plan their exact authored edit,
    reason and input revision, without rechecking the current numerical implementation."""
    from alphalattice.control.product_host.research_authoring.feature_research import (
        CATEGORY,
        ResearchFeatureDefinitions,
    )
    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
    from alphalattice.foundation.feature_engine.catalog.research import (
        ResearchFeatureChange,
        plan_research_feature_change,
    )
    from alphalattice.foundation.feature_engine.catalog.service import FeatureCatalogCrudPlanner
    from alphalattice.foundation.feature_engine.producers.factors.catalog import (
        default_extension_kernel_registry,
    )
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        formula_specification,
    )

    catalog = FeatureCatalog.load()
    base = FeatureCatalogCrudPlanner(default_extension_kernel_registry()).revision(catalog)
    specification = formula_specification(
        catalog.factors[0].model_copy(
            update={
                "factor_id": "v615_formula",
                "formula_ref": "factor.formula",
                "formula": "-(close / lag(close, 5) - 1)",
                "lag_sessions": 0,
                "core_anchor": False,
            }
        )
    )
    document = {
        "input_binding_hash": "a" * 64,
        "base_revision_hash": base.revision_hash,
        "edits": [
            {
                "operation": "CREATE",
                "factor_id": "v615_formula",
                "specification": specification.model_dump(mode="json"),
                "preprocessing_recipe": "ROBUST_SECTOR_NEUTRAL_Z",
            }
        ],
        "reason": "Keep this authored declaration after a rebuilt ledger",
    }
    plan = plan_research_feature_change(
        catalog=catalog,
        base=base,
        source_panel_snapshot_hash="b" * 64,
        request=ResearchFeatureChange.model_validate_json(json.dumps(document)),
    )
    definitions = ResearchFeatureDefinitions(tmp_path)
    definitions.store.publish_json(
        category=CATEGORY,
        content_hash=plan.plan_hash,
        payload=plan.model_dump(mode="json"),
    )
    payload = {
        "sessions_hash": "c" * 64,
        "listing_ids_hash": "d" * 64,
        "definition_plan_hash": plan.plan_hash,
        "input_binding_hash": "a" * 64,
        "implementation_hash": "e" * 64,
        "source_projection_hash": "f" * 64,
        "caller": "INSTALLED_AGENT",
        **({"purpose": "PREPROCESS_VALUES"} if preprocess else {}),
    }
    task = _stopped_task(feature_research._contract(payload))
    app = feature_research.ResearchFeatureBuildApplication(
        SimpleNamespace(workspace=tmp_path),
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
    )
    _assert_request(
        app.replan_request(task),
        {
            "operation": "FEATURE_CATALOG_PLAN",
            "feature_document": plan.request.model_dump(mode="json"),
        },
    )


@pytest.mark.parametrize("default_selection", [False, True])
def test_a_stopped_book_retains_its_controls_window_and_admitted_package(
    tmp_path, default_selection
):
    """regression (TE12): a durable book's complete public declaration round-trips,
    freezing the admitted package when the original request used a workspace default."""
    from alphalattice.interface.local_application.portfolio_research import spec_from_document
    from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
        WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID,
        PortfolioResearchSpec,
    )
    from tests.portfolio_strategy_lab.local_web_support import TEST_PACKAGE, _harness

    spec = PortfolioResearchSpec.create(
        strategy_package_id=WORKSPACE_DEFAULT_STRATEGY_PACKAGE_ID
        if default_selection
        else TEST_PACKAGE.strategy_id,
        top_k=35,
        tranches=3,
        exit_rank=70,
        cost_bps_per_side="7.5",
        study_start=date(2024, 1, 2),
        study_end=date(2024, 2, 1),
    )
    with _harness(tmp_path) as harness:
        planned = harness.application.plan(spec)
        admitted = harness.application.admit(spec=spec, planned=planned)
        original = harness.session.task_control_registry.task(admitted.task_id)
        task = _stopped_task((original.input, original.goal, original.plan))
        request = harness.application.replan_request(task)
        _assert_request(
            request,
            {
                "operation": "PLAN",
                "spec": {
                    "strategy_package_id": TEST_PACKAGE.strategy_id,
                    "score_source_mode": "HISTORICAL_ARRAY_REPLAY",
                    "top_k": "35",
                    "tranches": "3",
                    "exit_rank": "70",
                    "weight_rule": spec.weight_rule,
                    "cost_bps_per_side": "7.5",
                    "secondary_benchmark_view": spec.secondary_benchmark_view,
                    "report_unit": spec.report_unit,
                    "study_start": "2024-01-02",
                    "study_end": "2024-02-01",
                },
            },
        )
        reopened = spec_from_document(request["spec"])
        assert reopened.strategy_package_id == TEST_PACKAGE.strategy_id
        assert reopened.model_dump(mode="json", exclude={"spec_hash", "strategy_package_id"}) == (
            spec.model_dump(mode="json", exclude={"spec_hash", "strategy_package_id"})
        )
        assert harness.resolver.numerical_calls == 0


def test_a_training_replan_refuses_an_ambiguous_component_selection(tmp_path):
    """A retained training Task cannot choose one component from an ambiguous selection."""
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="s1-workspace",
        default_strategy_package_id=None,
        default_score_source_mode=None,
        strategy_artifacts=(),
        strategy_installation="NOT_INSTALLED",
    )
    plan = model_training.ModelTrainingInputPlan.create(
        workspace_id="s1-workspace",
        workspace_manifest_hash=model_training.manifest_fields_hash(
            manifest, model_training.PLAN_FIELDS
        ),
        input_id="s1-input",
        input_binding_hash="b" * 64,
        component_ids=("G2_R0_TREND", "G6_R0_FAST_REBOUND"),
        listing_count=50,
        source_session_count=252,
        implementation_hash="c" * 64,
    )
    task = _stopped_task(model_training._task_contract(plan, "INSTALLED_AGENT"))
    app = model_training.ModelTrainingInputApplication(
        SimpleNamespace(workspace=tmp_path),
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match=r"model_training\.component_selection_invalid"):
        app.replan_request(task)


@pytest.mark.parametrize(
    "owner",
    [
        "workspace_preparation",
        "feature_research",
        "research_input",
        "research_strategy",
        "workspace_data_update",
        "model_training",
    ],
)
@pytest.mark.parametrize("suffix", [None, "", "invalid-task"])
def test_an_admission_refusal_names_its_blocking_task_and_offers_read_and_wait(owner, suffix):
    """Only a valid blocking Task supplies read/wait requests across the saved-answer seam."""
    suffix = str(uuid4()) if suffix is None else suffix
    code = f"{owner}.finish_or_recover_existing_task" + (f":{suffix}" if suffix else "")
    refused = CommandSubmission(
        command_kind=owner,
        disposition="REFUSED_INVALID_COMMAND",
        submitted_at=datetime(2026, 10, 2, tzinfo=UTC),
        refusal_detail=code,
    ).answer()
    expected = dict(
        status="REFUSED_INVALID_COMMAND", task_id=None, lifecycle=None, failure_code=code
    )
    if suffix not in ("", "invalid-task"):
        expected["blocking_task_id"] = suffix
        expected["next_requests"] = {
            "read": {"operation": "STATUS", "task_id": suffix},
            "wait": {"operation": "STATUS", "task_id": suffix, "wait_seconds": 20},
        }
    assert refused == expected
    for request in refused.get("next_requests", {}).values():
        _assert_request(request, request)


def test_a_stopped_tasks_answer_offers_its_way_on_never_a_wait_on_it() -> None:
    """requirement (answer truth): an answer about a BLOCKED, CANCELLED or RECOVERY_REQUIRED Task
    offers its way on; a wait or wake on that Task, or no offer at all, is refused."""
    from alphalattice.interface.local_application.answers import stopped_problem

    task = str(uuid4())
    wait = {"operation": "STATUS", "task_id": task, "wait_seconds": 20}
    wake = {"operation": "WAKE_REGISTER", "task_id": task}
    way = {"operation": "WORKSPACE_PREPARE_PLAN", "recovery_task_id": task}

    def asked(lifecycle: str, **offers: object) -> str | None:
        return stopped_problem({"task_id": task, "lifecycle": lifecycle, "next_requests": offers})

    assert asked("BLOCKED", wait=wait) and asked("CANCELLED", w=wake) and asked("RECOVERY_REQUIRED")
    assert asked("BLOCKED", replan=way) is None and asked("RUNNING", wait=wait) is None
    assert asked("CANCELLED") is None  # a cancel was the request: it needs no way on
    # The Task's record outranks an answer that names no state, as a refusal's blocker does.
    assert stopped_problem({"blocking_task_id": task, "next_requests": {"wait": wait}}, "BLOCKED")


def test_an_offer_sent_back_as_it_stands_is_admitted_and_keeps_the_first_uses_date(
    truth_check,
) -> None:
    """requirement (answer truth): a recovery-bound offer sent back verbatim is never refused as
    not offered, and an update offered under a dated first use keeps that date."""
    task = str(uuid4())
    record = SimpleNamespace(lifecycle=SimpleNamespace(value="BLOCKED"))
    goal = SimpleNamespace(declaration=SimpleNamespace(target_date=date(2026, 10, 10)))
    host = SimpleNamespace(
        replans=lambda: {
            "prep": SimpleNamespace(
                preview="WORKSPACE_PREPARE_PLAN", admitting="WORKSPACE_PREPARE_CONFIRM"
            )
        },
        workspace_session=SimpleNamespace(
            task_control_registry=SimpleNamespace(task=lambda _: record)
        ),
        goals=SimpleNamespace(
            first_use=lambda: goal, first_use_delegation=lambda _: {"active": True}
        ),
    )

    def answer(digit: str, **update: object) -> dict[str, object]:
        replan = {"operation": "WORKSPACE_PREPARE_PLAN", "recovery_task_id": task}
        update = {"operation": "RESEARCH_UPDATE_PLAN", "strategy_package_id": "BAL", **update}
        offers = {"replan": {**replan, "recovery_task_hash": digit * 64}, "update": update}
        return {"task_id": task, "lifecycle": "BLOCKED", "next_requests": offers}

    code = "portfolio_research.recovery_request_not_offered"
    planted = list(truth_check(host, "STATUS", answer("a"), lambda _: {"failure_code": code}))
    assert [line.rsplit(" ", 1)[-1] for line in planted] == [code, "date"]
    dated = answer("b", observed_through="2026-10-09")
    assert list(truth_check(host, "STATUS", dated, lambda _: {"status": "PLANNED"})) == []


def test_a_full_training_plan_keeps_its_hash_and_a_light_one_names_its_lifecycle() -> None:
    """A FULL training plan keeps its existing hash, while a LIGHT plan names its lifecycle."""
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    fields = dict(
        workspace_id="s1-workspace",
        workspace_manifest_hash="a" * 64,
        input_id="s1-input",
        input_binding_hash="b" * 64,
        component_ids=("G6_R0_FAST_REBOUND",),
        listing_count=50,
        source_session_count=252,
        implementation_hash="c" * 64,
    )
    before = model_training.ModelTrainingInputPlan.create(**fields)
    named = model_training.ModelTrainingInputPlan.create(**fields, model_lifecycle="FULL")
    light = model_training.ModelTrainingInputPlan.create(**fields, model_lifecycle="LIGHT")
    assert named == before and before.plan_hash == canonical_hash(fields)
    assert "model_lifecycle" not in before.model_dump(mode="json")
    assert light.plan_hash != before.plan_hash
    assert light.model_dump(mode="json")["model_lifecycle"] == "LIGHT"
    assert (
        model_training.ModelTrainingInputPlan.model_validate(light.model_dump(mode="json")) == light
    )


def test_recovery_provenance_is_paired_typed_and_worded() -> None:
    """A recovery request retains its source pair or refuses it by its own code."""

    from alphalattice.interface.local_application.cli_contract import refusal_words
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
        PortfolioResearchRequestDocument,
    )

    source = UUID(int=83)
    context = {"recovery_task_id": str(source), "recovery_task_hash": "a" * 64}
    document = PortfolioResearchRequestDocument.model_validate(
        {"operation": "WORKSPACE_PREPARE_PLAN", **context}
    )
    request = document.to_operation_request()
    assert request.recovery_task_id == source and request.recovery_task_hash == "a" * 64
    assert (
        PortfolioResearchRequestDocument(operation="WORKSPACE_PREPARE_PLAN").recovery_task_id
        is None
    )
    for field in context:
        with pytest.raises(ValueError, match=r"portfolio_research\.recovery_context_pair_required"):
            PortfolioResearchRequestDocument.model_validate(
                {"operation": "WORKSPACE_PREPARE_PLAN", field: context[field]}
            )
    with pytest.raises(ValueError, match=r"portfolio_research\.recovery_task_id_invalid"):
        PortfolioResearchOperationRequest(
            operation="WORKSPACE_PREPARE_PLAN",
            recovery_task_id="not-a-task",
            recovery_task_hash="a" * 64,
        )  # type: ignore[arg-type]
    for code in (
        "portfolio_research.recovery_context_pair_required",
        "portfolio_research.recovery_task_id_invalid",
        "portfolio_research.recovery_request_not_offered",
    ):
        words = refusal_words(code)
        assert "recovery" in words["detail"]
        assert words["next_action"] == "READ_THE_TASK_AND_CONFIRM_AGAIN"
