"""Acceptance tests for an Autonomous-style external Alpha consumer."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alphalattice.capabilities.alpha_modeling.adapters.regularized_linear import (
    build_regularized_linear_search_domain,
)
from alphalattice.capabilities.alpha_modeling.catalog import build_installed_alpha_model_catalog
from alphalattice.foundation.research_foundation.contracts import (
    ResearchDeskExecutionOutcomeRef,
    ResearchFoundationBinding,
)
from alphalattice.investment.alpha_research.experiments.bindings import (
    AlphaDevelopmentProgramAuthorityError,
)
from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaDevelopmentProgram,
    AlphaDevelopmentSplitPolicy,
    AlphaExperimentBatch,
    seal_contract,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.execution import execute_alpha_model_batch
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaModelCapabilityMandate,
    build_current_alpha_research_model_mandate,
)
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
    build_alpha_fold_commitment,
)
from alphalattice.investment.alpha_research.inputs.training import (
    build_alpha_training_input_binding,
)
from alphalattice.investment.alpha_research.targets.execution_outcome import (
    AlphaTargetLane,
    build_alpha_target_policy,
)
from alphalattice.kernel.shared_kernel.domain.enums import DataValidityClass
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.contracts import (
    RebalanceEvidencePoint,
    ResearchSplitSpec,
    ValidationTimeline,
)
from alphalattice.kernel.validation.enums import RebalanceFrequency, SplitMode
from alphalattice.kernel.validation.splitting import build_research_split
from tests.autonomous_alpha_consumer.external_consumer import (
    ADAPTER_ID,
    ConsumerOwnedMeanAdapter,
    build_external_development_authority,
    run_external_development,
)

FACTOR_IDS = ("consumer-signal", "consumer-control")
LISTING_IDS = tuple(f"listing-{index:02d}" for index in range(10))


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _foundation() -> ResearchFoundationBinding:
    execution = ResearchDeskExecutionOutcomeRef.model_construct(
        research_cadence=RebalanceFrequency.DAILY,
        snapshot_hash="d" * 64,
        schedule_hash="e" * 64,
        development_content_hash="f" * 64,
        sealed_holdout_content_hash="0" * 64,
        marker_hash="1" * 64,
        market_as_of="2026-07-31",
        data_validity_class=DataValidityClass.CURRENT_UNIVERSE_RESEARCH_ONLY,
    )
    return ResearchFoundationBinding.model_construct(
        research_cadence=RebalanceFrequency.DAILY,
        feature_panel_snapshot_hash="2" * 64,
        logical_panel_hash="a" * 64,
        logical_semantic_index_hash="c" * 64,
        factor_training_outcome_snapshot_hash="3" * 64,
        factor_screening_result_hash="4" * 64,
        factor_candidate_slate_hash="5" * 64,
        research_desk_factor_input_hash="b" * 64,
        execution_outcome=execution,
        ordered_factor_ids=FACTOR_IDS,
        downstream_factor_research_forbidden=True,
        secondary_feature_preprocessing_forbidden=True,
        foundation_hash="9" * 64,
    )


class _ArrayWorkspace:
    def __init__(self, fold: AlphaFoldArrays) -> None:
        self.fold = fold

    @contextmanager
    def fold_lease(self, fold_index: int):  # type: ignore[no-untyped-def]
        assert fold_index == 0
        yield self.fold

    def load_fold(self, fold_index: int) -> AlphaFoldArrays:
        assert fold_index == 0
        return self.fold

    def prepare_current_refit(self):  # type: ignore[no-untyped-def]
        raise AssertionError("current refit is outside this development consumer")


def _plan_and_workspace() -> tuple[AlphaFoldArrayPlan, _ArrayWorkspace]:
    first_session = date(2026, 1, 1)
    sessions = tuple(first_session + timedelta(days=index) for index in range(11))
    frozen_at = datetime(2026, 2, 1, tzinfo=UTC)
    timeline = ValidationTimeline(
        points=tuple(
            RebalanceEvidencePoint(
                session_date=session,
                evidence_available_at=frozen_at,
                benchmark_available_at=frozen_at,
            )
            for session in sessions
        )
    )
    split_plan = build_research_split(
        timeline,
        ResearchSplitSpec(
            mode=SplitMode.ROLLING,
            as_of_timestamp=frozen_at,
            train_sessions=6,
            validation_sessions=2,
            step_sessions=2,
            purge_sessions=1,
            embargo_sessions=0,
            holdout_sessions=1,
            minimum_folds=1,
        ),
    )
    assert len(split_plan.windows) == 1
    window = split_plan.windows[0]
    train_keys = tuple(
        (session, listing) for session in window.train_sessions for listing in LISTING_IDS
    )
    validation_keys = tuple(
        (session, listing) for session in window.validation_sessions for listing in LISTING_IDS
    )

    def materialize(
        keys: tuple[tuple[date, str], ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.empty((len(keys), len(FACTOR_IDS)), dtype=np.float64)
        targets = np.empty(len(keys), dtype=np.float64)
        economic_returns = np.empty(len(keys), dtype=np.float64)
        for row, (session, listing) in enumerate(keys):
            session_index = (session - first_session).days
            listing_index = int(listing.rsplit("-", 1)[1])
            cross_section = (listing_index - 4.5) / 4.5
            signal = cross_section + 0.03 * session_index
            control = np.sin((listing_index + session_index) / 4.0)
            features[row] = (signal, control)
            targets[row] = 0.7 * signal - 0.15 * control
            economic_returns[row] = 0.012 * signal + 0.002 * control
        return _readonly(features), _readonly(targets), _readonly(economic_returns)

    training_features, training_targets, training_economic = materialize(train_keys)
    validation_features, validation_targets, validation_economic = materialize(validation_keys)
    training_mask = _readonly(np.ones(len(train_keys), dtype=np.bool_))
    validation_mask = _readonly(np.ones(len(validation_keys), dtype=np.bool_))
    training_sessions = tuple(value[0] for value in train_keys)
    training_listings = tuple(value[1] for value in train_keys)
    validation_sessions = tuple(value[0] for value in validation_keys)
    validation_listings = tuple(value[1] for value in validation_keys)
    training_feature_hashes = tuple(canonical_hash(("feature", *value)) for value in train_keys)
    training_outcome_hashes = tuple(canonical_hash(("outcome", *value)) for value in train_keys)
    validation_feature_hashes = tuple(
        canonical_hash(("feature", *value)) for value in validation_keys
    )
    validation_outcome_hashes = tuple(
        canonical_hash(("outcome", *value)) for value in validation_keys
    )
    foundation = _foundation()
    target_policy = build_alpha_target_policy(
        lane=AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS,
        sector_revision="6" * 64,
    )
    commitment = build_alpha_fold_commitment(window)
    binding = build_alpha_training_input_binding(
        scope="DEVELOPMENT_FOLD",
        foundation_hash=foundation.foundation_hash,
        feature_panel_snapshot_hash=foundation.feature_panel_snapshot_hash,
        causal_outcome_snapshot_hash=foundation.execution_outcome.snapshot_hash,
        target_policy=target_policy,
        ordered_feature_ids=FACTOR_IDS,
        feature_context_hash=None,
        training_row_sessions=training_sessions,
        training_row_listing_ids=training_listings,
        prediction_row_sessions=validation_sessions,
        prediction_row_listing_ids=validation_listings,
        training_features=training_features,
        training_targets=training_targets,
        training_mask=training_mask,
        prediction_features=validation_features,
        prediction_targets=validation_targets,
        prediction_mask=validation_mask,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        prediction_feature_row_hashes=validation_feature_hashes,
        prediction_outcome_row_hashes=validation_outcome_hashes,
        training_economic_returns=training_economic,
        prediction_economic_returns=validation_economic,
        training_cutoff=window.train_sessions[-1],
        outcome_maturity_session=window.train_sessions[-1],
        prediction_anchor=window.validation_sessions[0],
        fold_commitment_hash=commitment.commitment_hash,
    )
    fold = AlphaFoldArrays(
        commitment=commitment,
        ordered_factor_ids=FACTOR_IDS,
        training_sessions=tuple(window.train_sessions),
        training_listing_ids=training_listings,
        training_features=training_features,
        training_targets=training_targets,
        training_feature_complete=training_mask,
        training_outcome_complete=training_mask,
        validation_sessions=tuple(window.validation_sessions),
        validation_row_sessions=validation_sessions,
        validation_listing_ids=validation_listings,
        validation_features=validation_features,
        validation_targets=validation_targets,
        validation_feature_complete=validation_mask,
        validation_outcome_complete=validation_mask,
        validation_feature_row_hashes=validation_feature_hashes,
        validation_outcome_row_hashes=validation_outcome_hashes,
        training_economic_returns=training_economic,
        validation_economic_returns=validation_economic,
        training_row_sessions=training_sessions,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        training_input_binding=binding,
    )
    plan = AlphaFoldArrayPlan(
        foundation=foundation,
        ordered_listing_ids=LISTING_IDS,
        feature_reader=object(),  # type: ignore[arg-type]
        outcome_reader=object(),  # type: ignore[arg-type]
        feature_panel_manifest_ref="playpen://fixture/external-consumer-panel",
        causal_outcome_manifest_ref="playpen://fixture/external-consumer-outcome",
        split_plan=split_plan,
        target_policy=target_policy,
    )
    return plan, _ArrayWorkspace(fold)


def test_external_consumer_runs_without_goal_governance_and_replays_exactly(
    tmp_path: Path,
) -> None:
    plan, workspace = _plan_and_workspace()
    ConsumerOwnedMeanAdapter.observed_fit_calls = 0

    program, first = run_external_development(
        fold_plan=plan,
        array_workspace=workspace,
        artifact_root=tmp_path,
    )
    replay_program, replay = run_external_development(
        fold_plan=plan,
        array_workspace=workspace,
        artifact_root=tmp_path,
    )

    assert replay_program == program
    assert replay.candidates == first.candidates
    assert replay.development_surface_hash == first.development_surface_hash
    assert replay.fold_surface_hashes == first.fold_surface_hashes
    assert (first.fit_call_count, first.predict_call_count, first.metric_call_count) == (1, 2, 1)
    assert (replay.fit_call_count, replay.predict_call_count, replay.metric_call_count) == (0, 0, 0)
    assert ConsumerOwnedMeanAdapter.observed_fit_calls == 1
    assert program.target_policy is not None
    assert program.target_policy.lane is AlphaTargetLane.SECTOR_RESIDUAL_RANK_GAUSS
    assert (
        program.split_policy.train_sessions,
        program.split_policy.validation_sessions,
        program.split_policy.expected_complete_folds,
    ) == (6, 2, 1)
    assert not {
        "pm_plan_hash",
        "stability_policy_hash",
        "goal_criteria_hash",
        "user_authorization_hash",
        "research_goal_hash",
    } & set(type(program).model_fields)

    default_catalog = build_installed_alpha_model_catalog()
    current_mandate = build_current_alpha_research_model_mandate(catalog=default_catalog)
    assert ADAPTER_ID not in default_catalog.adapter_ids
    assert all(value.adapter_id != ADAPTER_ID for value in current_mandate.ordered_search_domains)


def test_standalone_contracts_round_trip_and_reject_tamper() -> None:
    plan, _workspace = _plan_and_workspace()
    _catalog, mandate, program, _batch = build_external_development_authority(plan)

    assert AlphaModelCapabilityMandate.model_validate(mandate.model_dump(mode="json")) == mandate
    assert AlphaDevelopmentProgram.model_validate(program.model_dump(mode="json")) == program

    tampered_mandate = mandate.model_dump(mode="json")
    tampered_mandate["ordered_search_domains"][0]["constraints"]["scale_max"] = 2.0
    with pytest.raises(ValidationError, match="ALPHA_MODEL_SEARCH_DOMAIN_IDENTITY_INVALID"):
        AlphaModelCapabilityMandate.model_validate(tampered_mandate)

    tampered_program = program.model_dump(mode="json")
    tampered_program["ordered_feature_ids_hash"] = "7" * 64
    with pytest.raises(ValidationError, match="ALPHA_DEVELOPMENT_PROGRAM_IDENTITY_INVALID"):
        AlphaDevelopmentProgram.model_validate(tampered_program)


def test_wrong_program_or_economic_surface_fails_before_fit(tmp_path: Path) -> None:
    plan, workspace = _plan_and_workspace()
    catalog, mandate, program, batch = build_external_development_authority(plan)
    ConsumerOwnedMeanAdapter.observed_fit_calls = 0

    program_values = {
        name: getattr(program, name)
        for name in type(program).model_fields
        if name != "program_hash"
    }
    program_values["ordered_feature_ids_hash"] = "7" * 64
    wrong_program = seal_contract(
        AlphaDevelopmentProgram,
        program_values,
        "program_hash",
    )
    wrong_batch = seal_contract(
        AlphaExperimentBatch,
        {
            **{
                name: getattr(batch, name)
                for name in type(batch).model_fields
                if name != "batch_hash"
            },
            "program_hash": wrong_program.program_hash,
        },
        "batch_hash",
    )
    with pytest.raises(AlphaDevelopmentProgramAuthorityError) as wrong_axis:
        execute_alpha_model_batch(
            program=wrong_program,
            batch=wrong_batch,
            fold_plan=plan,
            store=AlphaDevelopmentArtifactStore(tmp_path / "wrong-axis"),
            array_workspace=workspace,
            model_catalog=catalog,
            model_mandate=mandate,
        )
    assert wrong_axis.value.failure_class == "AUTHORITY_FAILURE"
    assert ConsumerOwnedMeanAdapter.observed_fit_calls == 0

    assert workspace.fold.validation_economic_returns is not None
    wrong_returns = _readonly(workspace.fold.validation_economic_returns + 0.01)
    wrong_workspace = _ArrayWorkspace(
        replace(workspace.fold, validation_economic_returns=wrong_returns)
    )
    with pytest.raises(AlphaDevelopmentProgramAuthorityError) as wrong_economic:
        execute_alpha_model_batch(
            program=program,
            batch=batch,
            fold_plan=plan,
            store=AlphaDevelopmentArtifactStore(tmp_path / "wrong-economic"),
            array_workspace=wrong_workspace,
            model_catalog=catalog,
            model_mandate=mandate,
        )
    assert wrong_economic.value.failure_class == "AUTHORITY_FAILURE"
    assert ConsumerOwnedMeanAdapter.observed_fit_calls == 0


def test_target_fold_metric_package_and_catalog_mismatch_are_authority_failures(
    tmp_path: Path,
) -> None:
    plan, workspace = _plan_and_workspace()
    catalog, mandate, program, batch = build_external_development_authority(plan)
    ConsumerOwnedMeanAdapter.observed_fit_calls = 0

    alternate_target = build_alpha_target_policy(
        lane=AlphaTargetLane.LOG_RETURN_RANK_GAUSS,
        sector_revision="6" * 64,
    )
    production_shaped_split_policy = seal_contract(
        AlphaDevelopmentSplitPolicy,
        {
            "mode": "ROLLING",
            "train_sessions": 756,
            "purge_sessions": 1,
            "validation_sessions": 252,
            "step_sessions": 252,
            "embargo_sessions": 0,
            "sealed_holdout_sessions": 252,
            "minimum_folds": 3,
            "expected_complete_folds": 5,
        },
        "policy_hash",
    )
    changes = (
        {"target_policy": alternate_target},
        {"split_hash": "8" * 64},
        {"split_policy": production_shaped_split_policy},
    )
    for index, change in enumerate(changes):
        values = {
            name: getattr(program, name)
            for name in type(program).model_fields
            if name != "program_hash"
        }
        values.update(change)
        wrong_program = seal_contract(AlphaDevelopmentProgram, values, "program_hash")
        wrong_batch = seal_contract(
            AlphaExperimentBatch,
            {
                **{
                    name: getattr(batch, name)
                    for name in type(batch).model_fields
                    if name not in {"batch_hash", "program_hash"}
                },
                "program_hash": wrong_program.program_hash,
            },
            "batch_hash",
        )
        with pytest.raises(AlphaDevelopmentProgramAuthorityError) as raised:
            execute_alpha_model_batch(
                program=wrong_program,
                batch=wrong_batch,
                fold_plan=plan,
                store=AlphaDevelopmentArtifactStore(tmp_path / f"wrong-program-{index}"),
                array_workspace=workspace,
                model_catalog=catalog,
                model_mandate=mandate,
            )
        assert raised.value.failure_class == "AUTHORITY_FAILURE"

    for nested_field, identity_field in (
        ("metric_policy", "policy_hash"),
        ("package_identity", "package_hash"),
    ):
        payload = program.model_dump(mode="json")
        payload[nested_field][identity_field] = "7" * 64
        with pytest.raises(ValidationError):
            AlphaDevelopmentProgram.model_validate(payload)

    default_catalog = build_installed_alpha_model_catalog()
    wrong_mandate = AlphaModelCapabilityMandate.create(
        catalog_binding=default_catalog.binding,
        ordered_search_domains=(build_regularized_linear_search_domain(),),
    )
    with pytest.raises(AlphaDevelopmentProgramAuthorityError) as wrong_model:
        execute_alpha_model_batch(
            program=program,
            batch=batch,
            fold_plan=plan,
            store=AlphaDevelopmentArtifactStore(tmp_path / "wrong-model-authority"),
            array_workspace=workspace,
            model_catalog=default_catalog,
            model_mandate=wrong_mandate,
        )
    assert wrong_model.value.failure_class == "AUTHORITY_FAILURE"
    assert ConsumerOwnedMeanAdapter.observed_fit_calls == 0
