"""Sealed synthetic metadata for the strategy-date CLI tests; no model fit or real data."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from alphalattice.capabilities.portfolio_backtesting.clocks import EveryFormationClock
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioStateTransitionBinding,
)
from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceArtifact,
    ResearchWorkspaceDecisionUpdate,
    ResearchWorkspaceManifest,
    ResearchWorkspaceScoreInput,
)
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenScoreObservationSnapshot,
    seal_current_contract,
)
from alphalattice.investment.alpha_research.scores.frozen_inference import FrozenMarketScale
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AlphaImportedChild,
    AlphaModelLifecycleAdmission,
    AlphaPreparedRefit,
    AlphaTrainingObservations,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaModelLifecycleRecipe,
    resolve_alpha_refit_plan,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    FROZEN_RESEARCH_BOOK_RECIPES,
    frozen_book_package,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    ARTIFACT_KEY,
    CATEGORY,
    LifecyclePortfolioAuthority,
    LifecycleScoreEvidence,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.local_web_support import _resolved, _Resolver
from tests.portfolio_strategy_lab.synthetic_numerical import build_numerical

HASH = canonical_hash("synthetic strategy date metadata; not strategy evidence")


def _write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(), encoding="utf-8", newline="\n")
    return relative


class _InstalledDates(_Resolver):
    def __init__(self, packages):
        super().__init__(_resolved())
        self.packages = packages

    def installed_packages(self):
        return {p.package_hash: p for p in self.packages}


@contextmanager
def date_host(
    workspace,
    *,
    active=False,
    missing=False,
    undated=False,
    score_gap=False,
    book_before_data_end=False,
    observed_at=None,
    activated_at=None,
    clock=None,
):
    """The real Host and CLI on valid sealed metadata, with both installed frozen packages.

    Numerical payloads are deliberately absent: reading dates must need neither arrays nor fits.
    The active checkpoint reuses the decision tests' synthetic numerical authority.
    """
    workspace.mkdir()
    axis = (date(2026, 6, 29), date(2026, 6, 30))
    if score_gap:
        axis = (date(2026, 6, 26), *axis)
    training = []
    admissions = {}
    for cid, end in (("G2_R0_TREND", date(2026, 7, 5)), ("G6_R0_FAST_REBOUND", date(2026, 7, 3))):
        component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(cid)
        lifecycle = AlphaModelLifecycleRecipe.from_component(component)
        days = tuple(
            date(2010, 1, 1) + timedelta(days=i) for i in range((end - date(2010, 1, 1)).days)
        )
        snapshot = seal_current_contract(
            FrozenScoreObservationSnapshot,
            dict(
                disposition="SYNTHETIC_INPUT_QA",
                formation_sessions=days,
                ordered_listing_ids=("a", "b"),
                sector_by_listing_id={"a": "synthetic", "b": "synthetic"},
                array_content_hash=HASH,
                source_binding_hash=HASH,
            ),
            "snapshot_hash",
        )
        _write(
            workspace,
            f"artifacts/alpha-research/current/frozen-observation-snapshots/{snapshot.snapshot_hash}.json",
            snapshot,
        )
        observations = AlphaTrainingObservations.create(
            observation_hash=snapshot.snapshot_hash,
            target_method_id=component.target_recipe,
            target_authority_hash=HASH,
            label_available_sessions=tuple(
                None if undated and cid == "G6_R0_FAST_REBOUND" else d + timedelta(days=1)
                for d in days
            ),
            array_file_hash=HASH,
        )
        _write(
            workspace,
            f"artifacts/alpha-research/current/lifecycle-training-observations/{observations.content_hash}.json",
            observations,
        )
        plan = resolve_alpha_refit_plan(
            lifecycle=lifecycle,
            vintage="2024-07",
            sessions=days,
            component_recipe_hash=component.recipe_hash,
            source_binding_hash=observations.content_hash,
            ordered_listing_ids=("a", "b"),
            ordered_feature_ids=component.ordered_feature_ids,
        )
        scale = FrozenMarketScale(
            vintage=plan.vintage,
            training_sessions=plan.training_sessions,
            source_binding_hash=observations.content_hash,
            center=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
        )
        prepared = AlphaPreparedRefit.create(
            plan=plan,
            observations_hash=observations.content_hash,
            array_file_hash=HASH,
            training_binding_hash=HASH,
            row_axis_hash=HASH,
            row_count=2,
            market_scale=scale,
        )
        child = AlphaImportedChild.create(
            prepared_hash=prepared.content_hash,
            lifecycle_hash=lifecycle.content_hash,
            vintage=plan.vintage,
            seed=lifecycle.seeds[0],
            estimator_hash=HASH,
            payload_hash=HASH,
            source_index_hash=HASH,
            environment_hash=HASH,
        )
        admission = AlphaModelLifecycleAdmission.create(
            component=component,
            lifecycle=lifecycle,
            observations_hash=observations.content_hash,
            prepared=(prepared,),
            initial_children=(child,),
            formation_start=date(2024, 7, 1),
            formation_end=date(2026, 7, 29),
            maximum_fit_attempts=3 * len(lifecycle.seeds) if active else 0,
            environment_hash=HASH,
            # The grant can retain an unused vintage from its original study.
            fit_vintages=("2024-07", "2024-10", "2026-01", "2026-04", "2026-07") if active else (),
            renewal_through=date(2026, 9, 30) if active else None,
        )
        path = (
            f"artifacts/alpha-research/current/lifecycle-admissions/{admission.content_hash}.json"
        )
        if not missing or cid != "G6_R0_FAST_REBOUND":
            _write(workspace, path, admission)
        admissions[cid] = (path, admission)
        training.append(
            LifecycleScoreEvidence(
                component_id=cid,
                task_id=uuid4(),
                program_hash=HASH,
                receipt_hash=HASH,
                component_recipe_hash=component.recipe_hash,
                lifecycle_hash=lifecycle.content_hash,
                formation_sessions=(axis[0], axis[-1])
                if score_gap and cid == "G2_R0_TREND"
                else axis,
                scores_hash=HASH,
            )
        )
    transition = PortfolioStateTransitionBinding.create(
        transition_implementation_hash=HASH,
        rebalance_clock=EveryFormationClock().binding,
        reference_mark_method="MARKED_TO_MARKET_AT_CLOSE_T",
        mark_manifest_ref="synthetic",
        mark_surface_hash=HASH,
        mark_epoch_hash=HASH,
        mark_price_basis="synthetic",
        mark_source_watermark_hash=HASH,
        mark_availability_policy_hash=HASH,
        mark_availability_policy_id="synthetic",
    )
    authority = LifecyclePortfolioAuthority.create(
        input_binding_hash=HASH,
        preparation_request_hash=HASH,
        preparation_implementation_hash=HASH,
        components=tuple(training),
        formation_sessions=axis,
        ordered_listing_ids=("a", "b"),
        decision_hash=HASH,
        execution_hash=HASH,
        returns_hash=HASH,
        adv_hash=HASH,
        marks_hash=HASH,
        entry_sessions=(*axis[1:], date(2026, 7, 1)),
        holding_end_sessions=tuple(d + timedelta(days=1) for d in (*axis[1:], date(2026, 7, 1))),
        transition=transition,
        tradability_decision_hash=HASH,
        outcome_snapshot_hash=HASH,
        risk_return_surface_hash=HASH,
        sector_map_hash=HASH,
        curve_hash=HASH,
        latest_calibration_hash=HASH,
        unavailable_return_policy="require_complete",
    )
    PortfolioResearchArtifactStore(
        workspace / "artifacts/research-strategy-inputs/synthetic"
    ).publish(category=CATEGORY, value=authority, identity_field="authority_hash")
    packages = tuple(
        frozen_book_package(recipe=recipe, evidence_identity_hash=authority.authority_hash)
        for recipe in FROZEN_RESEARCH_BOOK_RECIPES
    )
    score_inputs = tuple(
        ResearchWorkspaceScoreInput(
            strategy_package_id=p.strategy_id,
            strategy_package_hash=p.package_hash,
            component_id=cid,
            authority_relative_path=admissions[cid][0],
            authority_hash=admissions[cid][1].content_hash,
            source_kind="WORKSPACE_DATA_FEATURE",
        )
        for p in packages
        for cid in p.component_ids
    )
    decisions = []
    if active:
        for package, recipe in zip(packages, FROZEN_RESEARCH_BOOK_RECIPES, strict=True):
            numerical = build_numerical(
                installed_package=package,
                book_recipe=recipe,
                first_day=date(2026, 7, 1) if book_before_data_end else date(2026, 7, 23),
                epoch_start=date(2026, 7, 1) if book_before_data_end else date(2026, 7, 23),
            )
            values = {
                name: getattr(numerical.checkpoint, name)
                for name in type(numerical.checkpoint).model_fields
                if name != "content_hash"
            }
            values.update(
                purpose="PERSON_ACTIVATED_FORWARD_RESEARCH",
                activated_at=activated_at or datetime(2026, 7, 22, 23, tzinfo=UTC),
                book_task_id=uuid4(),
            )
            checkpoint = PortfolioDecisionCheckpoint.create(**values)
            PortfolioLedgerStore.for_workspace(workspace).publish_decision_checkpoint(checkpoint)
            decisions.append(
                ResearchWorkspaceDecisionUpdate(
                    strategy_package_id=package.strategy_id,
                    strategy_package_hash=package.package_hash,
                    checkpoint_hash=checkpoint.content_hash,
                )
            )
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="synthetic-strategy-dates",
        strategy_installation="NON_DEFAULT_RESEARCH",
        default_strategy_package_id=None,
        default_score_source_mode=None,
        strategy_artifacts=(
            ResearchWorkspaceArtifact(
                artifact_key=ARTIFACT_KEY,
                relative_path=f"artifacts/research-strategy-inputs/synthetic/portfolio-strategy-lab/{CATEGORY}/{authority.authority_hash}.json",
            ),
        ),
        score_inputs=score_inputs,
        decision_updates=tuple(decisions) or None,
    )
    live = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_InstalledDates(packages),
        clock=clock or (lambda: observed_at or datetime(2026, 10, 3, 12, tzinfo=UTC)),
    )
    live.start()
    try:
        yield live
    finally:
        live.stop()


def study_date_owner(live):
    """Public reader ports over sealed study model sets; full evidence reads must not occur."""
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceModelTrainingInput,
    )
    from alphalattice.control.product_host.composition.strategy_activation import StrategyActivation
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        AlphaModelSetPublication,
        read_lifecycle_admission,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
        LifecyclePortfolioAuthority,
    )

    root = live.workspace
    current = live.workspace_manifest
    authority = LifecyclePortfolioAuthority.model_validate_json(
        (root / current.strategy_artifacts[0].relative_path).read_bytes()
    )
    bindings = []
    summaries = {}
    store = AlphaCurrentArtifactStore(root / "artifacts")
    for evidence in authority.components:
        bound = next(b for b in current.score_inputs if b.component_id == evidence.component_id)
        admission = read_lifecycle_admission(
            root / bound.authority_relative_path, expected_hash=bound.authority_hash
        )
        prototype = admission.prepared[0]
        observations = AlphaTrainingObservations.model_validate_json(
            (
                root
                / "artifacts/alpha-research/current/lifecycle-training-observations"
                / f"{admission.observations_hash}.json"
            ).read_bytes()
        )
        original = store.load_frozen_observation_snapshot(observations.observation_hash)
        plans = tuple(
            resolve_alpha_refit_plan(
                lifecycle=admission.lifecycle,
                vintage=v,
                sessions=original.formation_sessions,
                component_recipe_hash=admission.component.recipe_hash,
                source_binding_hash=admission.observations_hash,
                ordered_listing_ids=prototype.plan.ordered_listing_ids,
                ordered_feature_ids=prototype.plan.ordered_feature_ids,
            )
            for v in admission.lifecycle.vintages(date(2024, 7, 1))
        )
        preparations = tuple(
            AlphaPreparedRefit.create(
                plan=p,
                observations_hash=admission.observations_hash,
                array_file_hash=HASH,
                training_binding_hash=HASH,
                row_axis_hash=HASH,
                row_count=2,
                market_scale=FrozenMarketScale(
                    vintage=p.vintage,
                    training_sessions=p.training_sessions,
                    source_binding_hash=admission.observations_hash,
                    center=(0.0, 0.0, 0.0),
                    scale=(1.0, 1.0, 1.0),
                ),
            )
            for p in plans
        )
        values = {
            name: getattr(admission, name)
            for name in type(admission).model_fields
            if name != "content_hash"
        }
        values.update(prepared=preparations, initial_children=())
        if current.decision_updates:
            values["formation_start"] = date(2026, 7, 23)
        training = AlphaModelLifecycleAdmission.create(**values)
        path = _write(
            root,
            f"artifacts/alpha-research/current/lifecycle-admissions/{training.content_hash}.json",
            training,
        )
        binding = ResearchWorkspaceModelTrainingInput(
            component_id=evidence.component_id,
            input_binding_hash=HASH,
            source_identity_hash=HASH,
            authority_relative_path=path,
            authority_hash=training.content_hash,
        )
        bindings.append(binding)
        children = tuple(
            AlphaImportedChild.create(
                prepared_hash=p.content_hash,
                lifecycle_hash=admission.lifecycle.content_hash,
                vintage=p.plan.vintage,
                seed=seed,
                estimator_hash=HASH,
                payload_hash=HASH,
                source_index_hash=HASH,
                environment_hash=HASH,
            )
            for p in preparations
            for seed in admission.lifecycle.seeds
        )
        publication = AlphaModelSetPublication.create(
            component=admission.component,
            lifecycle=admission.lifecycle,
            formation=date(2024, 7, 1),
            children=children,
        )
        output = f"studies/{evidence.component_id}"
        _write(
            root,
            f"{output}/alpha-lifecycle/alpha-research/current/lifecycle-model-sets/{publication.content_hash}.json",
            publication,
        )
        summaries[evidence.task_id] = {
            "document": {
                "experiment": {
                    "data_snapshot_handle": binding.source_handle,
                    "output_workspace": output,
                }
            }
        }
    manifest = current.with_bindings(
        score_inputs=None,
        model_training_inputs=tuple(
            sorted(bindings, key=lambda b: (b.component_id, b.authority_hash))
        ),
    )

    def no_bulk(_):
        raise AssertionError("date read called the full evidence verifier")

    return StrategyActivation(
        session=live.session,
        manifest=lambda: manifest,
        packages=lambda: {
            p.strategy_id: p for p in live.application.resolver.installed_packages().values()
        },
        ledger=live.application.ledger,
        read_experiment=no_bulk,
        read_information=lambda task: summaries[task],
        clock=live.clock,
        hold=lambda _: None,
    )
