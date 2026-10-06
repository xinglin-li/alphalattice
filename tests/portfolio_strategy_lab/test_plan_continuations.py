"""Verified expired plans offer their exact next declaration, never a default (V523)."""

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
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)


def test_an_expired_capture_plan_replans_its_input_or_names_an_unavailable_source(
    tmp_path, monkeypatch
):
    """regression (V523/S3): expiry forbids the old run, while a verified plan's input
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
    """regression (V523/S3): re-planning names the same training component and exact input;
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
            }
        },
        "s1-input",
        lambda value: model_training._task_contract(value, "INSTALLED_AGENT"),
    )


def test_an_expired_strategy_plan_keeps_its_parents_and_policy(tmp_path, monkeypatch):
    """regression (V523/S3): a re-plan retains the authored Alpha/Risk parents and policy;
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
    """regression (V523/S3): a re-plan keeps the exact package, declared component and
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
            session=session, manifest=manifest, packages={}, clock=clock
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
    """regression (V523/S3): a re-plan keeps the same package and exact published score;
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
    """regression (V615/TE12): a stop re-plans the person's input choice and observation end;
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
    """regression (V615/TE12): the durable update target survives a stop without any current
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
    """regression (V615/TE12): raw and prepared builds re-plan their exact authored edit,
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
    """regression (V615/TE12): a durable book's complete public declaration round-trips,
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


def test_a_training_task_never_silently_drops_a_component_from_its_replan(tmp_path):
    """regression (V615/TE12): the singleton planning entry refuses a multi-component retained
    Task by name, instead of inventing a new request by dropping one of its selections."""
    plan = model_training.ModelTrainingInputPlan.create(
        workspace_id="s1-workspace",
        workspace_manifest_hash="a" * 64,
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
