"""Focused requirements for the remediated Factor-to-Alpha Campaign contracts."""

from __future__ import annotations

import dataclasses
import inspect
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from alphalattice.capabilities.alpha_modeling.adapters.rank_composite import (
    RankCompositeAdapter,
    RankCompositeParameters,
    build_rank_composite_recipe,
    build_rank_composite_search_domain,
    fit_rank_composite,
)
from alphalattice.capabilities.alpha_modeling.catalog import (
    build_alpha_campaign_development_catalog,
)
from alphalattice.capabilities.alpha_modeling.contracts import (
    alpha_model_array_content_hash,
)
from alphalattice.foundation.factor_research.experiments.campaign import (
    CuratedFactorCheckpoint,
)
from alphalattice.investment.alpha_research.calibration.stock_returns import (
    StockCalibrationRecipe,
    calibrate_stock_returns_cross_fitted,
)
from alphalattice.investment.alpha_research.experiments.campaign import (
    APPROVED_RANK_COMPOSITE_SELECTION_COUNTS,
    AlphaCampaignBoundaryError,
    AlphaDevelopmentProgram,
    AlphaDevelopmentRequest,
    AlphaHorizonAuthority,
    compile_alpha_development_program,
)
from alphalattice.investment.alpha_research.experiments.campaign_evidence import (
    AlphaCampaignDecisionReceipt,
    AlphaCampaignDecisionSubmission,
    AlphaDevelopmentTrialEvidence,
    AlphaInnerTrialScore,
    alpha_campaign_decision_policy_hash,
    derive_alpha_inner_selection,
)
from alphalattice.investment.alpha_research.experiments.campaign_execution import (
    AlphaCampaignPredictionSurface,
    alpha_campaign_prediction_artifact_hash,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.scaling.catalog import (
    build_alpha_campaign_cross_sectional_scale_catalog,
)
from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    AGGREGATION_CONTROL_SPAN,
    AlphaScoreAggregationError,
    AlphaScoreAggregationEvidence,
    AlphaScoreAggregationFoldSelection,
    AlphaScoreAggregationProgram,
    AlphaScoreAggregationSurface,
    aggregate_trailing_mean,
    row_axis_hash,
    score_value_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution import (
    ActorKind,
    AgentExecutionBinding,
    seal_actor_submission,
)


def _request() -> AlphaDevelopmentRequest:
    return AlphaDevelopmentRequest(
        model_method_ids=(
            "RIDGE",
            "LASSO",
            "ELASTIC_NET",
            "STATE_INTERACTION_LINEAR",
            "LIGHTGBM",
            "HIST_GRADIENT_BOOSTING",
            "HUBER_LINEAR",
        ),
        model_search_domain_ids=(
            "REGULARIZED_LINEAR_BOUNDED_GRID",
            "CHRONOLOGICAL_LIGHTGBM_BOUNDED_GRID",
            "HIST_GRADIENT_BOOSTING_CONDITIONAL_GRID",
            "HUBER_CONDITIONAL_GRID",
        ),
        scale_method_ids=(
            "LAGGED_XS_DISPERSION",
            "EWMA_XS_DISPERSION",
            "ASYMMETRIC_EWMA_XS_DISPERSION",
            "HAR_XS_DISPERSION",
        ),
        scale_search_domain_ids=(
            "LAGGED_CONTROL",
            "EWMA_HALF_LIVES_10_21_42",
            "ASYMMETRIC_EWMA_RISE_10_DECAY_42",
            "HAR_MINIMUM_63_RIDGE_RATIO_1E_6",
        ),
    )


def _program(
    ordered_factor_ids: tuple[str, ...] = ("control", "market_drawdown_x_momentum"),
    *,
    request: AlphaDevelopmentRequest | None = None,
    ordered_context_ids: tuple[str, ...] = (),
):
    catalog = build_alpha_campaign_development_catalog()
    program = compile_alpha_development_program(
        request=request or _request(),
        factor_checkpoint_hash="1" * 64,
        feature_source_hash="2" * 64,
        ordered_factor_ids=ordered_factor_ids,
        horizon_authority={
            1: AlphaHorizonAuthority(
                horizon_sessions=1,
                target_evidence_hash="3" * 64,
                target_recipe_binding_hash="4" * 64,
                outcome_method_binding_hash="5" * 64,
                maturity_lag_sessions=2,
                embargo_sessions=1,
            ),
            5: AlphaHorizonAuthority(
                horizon_sessions=5,
                target_evidence_hash="6" * 64,
                target_recipe_binding_hash="7" * 64,
                outcome_method_binding_hash="8" * 64,
                maturity_lag_sessions=6,
                embargo_sessions=5,
            ),
        },
        model_catalog=catalog,
        scale_catalog=build_alpha_campaign_cross_sectional_scale_catalog(),
        ordered_context_ids=ordered_context_ids,
        context_source_hash="9" * 64 if ordered_context_ids else None,
    )
    return program, catalog


def test_program_seals_the_complete_approved_search_surface() -> None:
    """requirement: the approved Ridge/Lasso/ElasticNet/LightGBM surface is sealed."""

    program, _ = _program()
    counts: dict[str, int] = {}
    for trial in program.ordered_trials:
        counts[trial.method_id] = counts.get(trial.method_id, 0) + 1
    assert counts == {
        "RIDGE": 4,
        "LASSO": 4,
        "ELASTIC_NET": 12,
        "STATE_INTERACTION_LINEAR": 1,
        "LIGHTGBM": 60,
    }
    ridge_alphas = sorted(
        trial.recipe.parameters["alpha"]
        for trial in program.ordered_trials
        if trial.method_id == "RIDGE"
    )
    assert ridge_alphas == [0.1, 1.0, 10.0, 100.0]
    lightgbm_configurations = {
        (
            trial.recipe.parameters["num_leaves"],
            trial.recipe.parameters["learning_rate"],
            trial.recipe.parameters["max_depth"],
            trial.recipe.parameters["min_child_samples"],
        )
        for trial in program.ordered_trials
        if trial.method_id == "LIGHTGBM"
    }
    assert len(lightgbm_configurations) == 20
    assert {value.method_id for value in program.conditional_method_plans} == {
        "HIST_GRADIENT_BOOSTING",
        "HUBER_LINEAR",
    }


def test_yaml_selects_installed_methods_but_never_authority() -> None:
    """requirement: YAML carries selections only; authority-bearing keys refuse."""

    selections = """
model_method_ids: [RIDGE, LASSO, ELASTIC_NET, STATE_INTERACTION_LINEAR, LIGHTGBM,
                   HIST_GRADIENT_BOOSTING, HUBER_LINEAR]
model_search_domain_ids: [REGULARIZED_LINEAR_BOUNDED_GRID, CHRONOLOGICAL_LIGHTGBM_BOUNDED_GRID,
                          HIST_GRADIENT_BOOSTING_CONDITIONAL_GRID, HUBER_CONDITIONAL_GRID]
scale_method_ids: [LAGGED_XS_DISPERSION, EWMA_XS_DISPERSION, ASYMMETRIC_EWMA_XS_DISPERSION,
                   HAR_XS_DISPERSION]
scale_search_domain_ids: [LAGGED_CONTROL, EWMA_HALF_LIVES_10_21_42,
                          ASYMMETRIC_EWMA_RISE_10_DECAY_42, HAR_MINIMUM_63_RIDGE_RATIO_1E_6]
"""
    request = AlphaDevelopmentRequest.from_yaml(selections)
    assert "LIGHTGBM" in request.model_method_ids
    for forbidden_key in (
        "program_hash: aa",
        "artifact_uri: playpen://x",
        "executor: os.system",
        "import_path: alphalattice.evil",
        "publication_target: current",
    ):
        with pytest.raises(AlphaCampaignBoundaryError, match="AUTHORITY_FORBIDDEN"):
            AlphaDevelopmentRequest.from_yaml(selections + forbidden_key + "\n")
    with pytest.raises(ValidationError):
        AlphaDevelopmentRequest.from_yaml("model_method_ids: [NOT_INSTALLED_METHOD]")

    joint_request = AlphaDevelopmentRequest.from_yaml(
        (Path(__file__).parents[2] / "config" / "joint-primary-alpha-campaign.yaml").read_text(
            encoding="utf-8"
        )
    )
    joint_program, _ = _program(request=joint_request)
    joint_trials = tuple(
        value for value in joint_program.ordered_trials if value.method_id == "LIGHTGBM_REGULARIZED"
    )
    assert joint_program.feature_preprocessing_method_id == ("JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z")
    assert joint_program.target_method_id == "UNIVERSE_BOUND_SECTOR_RESIDUAL_STD_Z"
    assert len(joint_trials) == 72
    assert {
        (value.recipe.parameters["lambda_l1"], value.recipe.parameters["lambda_l2"])
        for value in joint_trials
    } == {(0.0, 0.0), (0.0, 1000.0), (0.0, 10000.0)}
    assert {
        (value.recipe.parameters["feature_fraction"], value.recipe.parameters["bagging_fraction"])
        for value in joint_trials
    } == {(0.5, 1.0), (0.5, 0.5), (1.0, 1.0), (1.0, 0.5)}


def test_actor_profile_changes_do_not_change_host_policy_identity() -> None:
    """requirement: actor provenance never reaches the Host decision policy."""

    policy = alpha_campaign_decision_policy_hash()
    submission = AlphaCampaignDecisionSubmission.create(
        dossier_hash="1" * 64,
        selected_method_ids=(),
        selected_scale_method_ids=(),
        selected_blend_recipe_hash=None,
        rationale="development-only regression",
    )
    receipts = []
    agent_execution = AgentExecutionBinding(
        profile_id="alpha-campaign-analyst",
        mode="STRUCTURED",
        profile_hash="a" * 64,
        document_hash="b" * 64,
        response_protocol="alpha-decision.v1",
        response_protocol_hash="c" * 64,
        concrete_schema_hash="d" * 64,
    )
    for actor_kind, actor_id, execution in (
        (ActorKind.HUMAN, "researcher-1", None),
        (ActorKind.EXTERNAL_AUTOMATION, "codex-factor-alpha-campaign", None),
        (ActorKind.INSTALLED_AGENT, "installed-agent-7", agent_execution),
    ):
        actor_submission = seal_actor_submission(
            actor_kind=actor_kind,
            actor_id=actor_id,
            submission_hash=submission.submission_hash,
            agent_execution=execution,
        )
        values = {
            "kind": "AlphaCampaignDecisionReceipt",
            "submission": submission.model_dump(mode="json"),
            "program_hash": "2" * 64,
            "actor_submission": actor_submission.model_dump(mode="json"),
            "decision_policy_hash": policy,
            "disposition": "DEVELOPMENT_NOT_SELECTED",
        }
        receipts.append(
            AlphaCampaignDecisionReceipt(
                submission=submission,
                program_hash="2" * 64,
                actor_submission=actor_submission,
                decision_policy_hash=policy,
                disposition="DEVELOPMENT_NOT_SELECTED",
                receipt_hash=canonical_hash(values),
            )
        )
    assert {value.decision_policy_hash for value in receipts} == {policy}
    assert len({value.receipt_hash for value in receipts}) == 3
    with pytest.raises(ValueError, match="installed Agent"):
        seal_actor_submission(
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher-1",
            submission_hash=submission.submission_hash,
            agent_execution=agent_execution,
        )


def test_yaml_subset_selection_controls_the_compiled_program() -> None:
    """requirement: a valid subset compiles only its own model and scale surface."""

    catalog = build_alpha_campaign_development_catalog()
    request = AlphaDevelopmentRequest.from_yaml(
        "model_method_ids: [RIDGE]\n"
        "model_search_domain_ids: [REGULARIZED_LINEAR_BOUNDED_GRID]\n"
        "scale_method_ids: [LAGGED_XS_DISPERSION, EWMA_XS_DISPERSION,"
        " ASYMMETRIC_EWMA_XS_DISPERSION]\n"
        "scale_search_domain_ids: [LAGGED_CONTROL, EWMA_HALF_LIVES_10_21_42,"
        " ASYMMETRIC_EWMA_RISE_10_DECAY_42]\n"
    )
    program = compile_alpha_development_program(
        request=request,
        factor_checkpoint_hash="1" * 64,
        feature_source_hash="2" * 64,
        ordered_factor_ids=("control", "market_drawdown_x_momentum"),
        horizon_authority={
            1: AlphaHorizonAuthority(
                horizon_sessions=1,
                target_evidence_hash="3" * 64,
                target_recipe_binding_hash="4" * 64,
                outcome_method_binding_hash="5" * 64,
                maturity_lag_sessions=2,
                embargo_sessions=1,
            ),
            5: AlphaHorizonAuthority(
                horizon_sessions=5,
                target_evidence_hash="6" * 64,
                target_recipe_binding_hash="7" * 64,
                outcome_method_binding_hash="8" * 64,
                maturity_lag_sessions=6,
                embargo_sessions=5,
            ),
        },
        model_catalog=catalog,
        scale_catalog=build_alpha_campaign_cross_sectional_scale_catalog(),
    )
    assert program.model_method_ids == ("RIDGE",)
    assert {value.method_id for value in program.ordered_trials} == {"RIDGE"}
    assert len(program.ordered_trials) == 4
    assert program.conditional_method_plans == ()
    assert "HAR_XS_DISPERSION" not in program.scale_method_ids
    assert "HAR_XS_DISPERSION" not in program.scale_parameter_domain
    # A selection whose declared domains do not match its methods is refused,
    # whether a required domain is missing or an unrelated one is added.
    for domains in (
        "[]",
        "[CHRONOLOGICAL_LIGHTGBM_BOUNDED_GRID]",
        "[REGULARIZED_LINEAR_BOUNDED_GRID, HUBER_CONDITIONAL_GRID]",
    ):
        with pytest.raises((AlphaCampaignBoundaryError, ValidationError)):
            AlphaDevelopmentRequest.from_yaml(
                "model_method_ids: [RIDGE]\n"
                f"model_search_domain_ids: {domains}\n"
                "scale_method_ids: [LAGGED_XS_DISPERSION, EWMA_XS_DISPERSION,"
                " ASYMMETRIC_EWMA_XS_DISPERSION]\n"
                "scale_search_domain_ids: [LAGGED_CONTROL, EWMA_HALF_LIVES_10_21_42,"
                " ASYMMETRIC_EWMA_RISE_10_DECAY_42]\n"
            )
    # The installed scale comparison needs its three primaries, so dropping one
    # is refused; and an authority-bearing key never reaches validation at all.
    with pytest.raises((AlphaCampaignBoundaryError, ValidationError)):
        AlphaDevelopmentRequest.from_yaml(
            "model_method_ids: [RIDGE]\n"
            "model_search_domain_ids: [REGULARIZED_LINEAR_BOUNDED_GRID]\n"
            "scale_method_ids: [LAGGED_XS_DISPERSION, EWMA_XS_DISPERSION, HAR_XS_DISPERSION]\n"
            "scale_search_domain_ids: [LAGGED_CONTROL, EWMA_HALF_LIVES_10_21_42,"
            " HAR_MINIMUM_63_RIDGE_RATIO_1E_6]\n"
        )
    with pytest.raises(AlphaCampaignBoundaryError, match="AUTHORITY_FORBIDDEN"):
        AlphaDevelopmentRequest.from_yaml(
            "model_method_ids: [RIDGE]\n"
            "model_search_domain_ids: [REGULARIZED_LINEAR_BOUNDED_GRID]\n"
            "scale_method_ids: [LAGGED_XS_DISPERSION, EWMA_XS_DISPERSION,"
            " ASYMMETRIC_EWMA_XS_DISPERSION]\n"
            "scale_search_domain_ids: [LAGGED_CONTROL, EWMA_HALF_LIVES_10_21_42,"
            " ASYMMETRIC_EWMA_RISE_10_DECAY_42]\n"
            "factor_checkpoint_hash: aa\n"
        )


def _canonical_materialization(
    *, sessions: tuple[date, ...], listings: tuple[str, ...], seed: int = 7
) -> object:
    """A small compiled canonical surface, reconciled the way readback reconciles one."""

    import pyarrow as pa

    from alphalattice.investment.alpha_research.targets.canonical import (
        CanonicalAlphaTargetRecipeBinding,
        build_canonical_alpha_target_recipe,
        compile_canonical_alpha_target_surface,
        seal_canonical_alpha_target_evidence,
    )
    from alphalattice.investment.alpha_research.targets.catalog import (
        build_installed_alpha_target_catalog,
    )
    from alphalattice.investment.alpha_research.targets.materialization import (
        CanonicalTargetMaterialization,
    )

    rng = np.random.default_rng(seed)
    rows = len(sessions) * len(listings)
    table = pa.table(
        {
            "formation_session": pa.array(
                [value for value in sessions for _ in listings], type=pa.date32()
            ),
            "listing_id": pa.array(list(listings) * len(sessions), type=pa.string()),
            "fit_target": pa.array(rng.normal(scale=0.02, size=rows), type=pa.float64()),
            "simple_economic_return": pa.array(
                rng.normal(scale=0.02, size=rows), type=pa.float64()
            ),
        }
    )
    sector_by_listing_id = {value: f"S{index % 2}" for index, value in enumerate(sorted(listings))}
    recipe = build_canonical_alpha_target_recipe(
        execution_outcome_recipe_id="one-session-open-to-open",
        sector_revision="a" * 64,
        minimum_coverage=0.5,
        minimum_sector_sample=2,
    )
    surface = compile_canonical_alpha_target_surface(
        source_table=table, recipe=recipe, sector_by_listing_id=sector_by_listing_id
    )
    binding = CanonicalAlphaTargetRecipeBinding.create(
        recipe=recipe,
        target_catalog_hash=build_installed_alpha_target_catalog().binding.catalog_hash,
        causal_outcome_snapshot_hash="b" * 64,
        outcome_method_binding_hash="c" * 64,
        maturity_lag_sessions=2,
    )
    return CanonicalTargetMaterialization(
        evidence=seal_canonical_alpha_target_evidence(surface=surface, recipe_binding=binding),
        recipe_binding=binding,
        surface=surface,
        outcome_method_binding_hash="c" * 64,
        maturity_lag_sessions=2,
        simple_economic_return_identity=surface.lane_identity.simple_economic_return_identity,
    )


def test_calibration_shrinkage_is_dimensionless() -> None:
    """Calibration shrinkage is dimensionless."""

    sessions = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(8))
    scores = np.asarray([0.1, 0.2, -0.1, 0.3, 0.2, -0.2, 0.4, 0.1], dtype=np.float64)
    returns = np.asarray([0.02, 0.03, -0.01, 0.04, 0.01, -0.03, 0.05, 0.02])
    folds = ((np.arange(4), np.arange(4, 8)),)
    for values in (scores, returns):
        values.setflags(write=False)
    _, base = calibrate_stock_returns_cross_fitted(
        recipe=StockCalibrationRecipe.create(),
        horizon_sessions=1,
        target_identity="1" * 64,
        score_identity="2" * 64,
        ordered_formation_sessions=sessions,
        scores=scores,
        raw_economic_returns=returns,
        folds=folds,
    )
    scaled_scores = scores * 100.0
    scaled_returns = returns * 100.0
    scaled_scores.setflags(write=False)
    scaled_returns.setflags(write=False)
    _, scaled = calibrate_stock_returns_cross_fitted(
        recipe=StockCalibrationRecipe.create(),
        horizon_sessions=1,
        target_identity="1" * 64,
        score_identity="3" * 64,
        ordered_formation_sessions=sessions,
        scores=scaled_scores,
        raw_economic_returns=scaled_returns,
        folds=folds,
    )
    assert base.fold_states[0].shrinkage_ratio == scaled.fold_states[0].shrinkage_ratio
    assert base.fold_states[0].normalized_cross_moment == pytest.approx(
        scaled.fold_states[0].normalized_cross_moment
    )
    assert base.fold_states[0].slope == pytest.approx(scaled.fold_states[0].slope)

    # --- the return-unit successor ------------------------------------------

    import pyarrow as pa

    from alphalattice.investment.alpha_research.calibration.authority import (
        installed_alpha_return_unit_capability,
        resolve_alpha_return_unit_calibration,
        seal_installed_return_unit_calibration,
    )
    from alphalattice.investment.alpha_research.calibration.return_unit import (
        AlphaReturnUnitCalibrationError,
        AlphaReturnUnitCalibrationEvidence,
        applied_expected_return_matrix,
        calibrate_alpha_return_unit_signal,
    )
    from alphalattice.investment.alpha_research.experiments.selected_scores import (
        AlphaSelectedFold,
        AlphaSelectedRowAxis,
        AlphaSelectedScoreProjection,
    )

    target_sessions = tuple(date(2026, 3, 1) + timedelta(days=index) for index in range(9))
    listings = tuple(f"L{index}" for index in range(6))
    materialization = _canonical_materialization(sessions=target_sessions, listings=listings)
    surface = materialization.surface
    row_sessions = tuple(value for value in target_sessions for _ in listings)
    row_listings = tuple(listings) * len(target_sessions)
    rng = np.random.default_rng(11)
    predicted = np.ascontiguousarray(rng.normal(size=len(row_sessions)))
    predicted.setflags(write=False)
    fold_counts = (len(listings) * 3,) * 3
    row_axis = AlphaSelectedRowAxis(
        ordered_folds=tuple(
            AlphaSelectedFold(
                fold_index=index,
                horizon_sessions=1,
                selected_method_id="RIDGE",
                selected_trial_hash=f"{index:0>64x}",
                inner_selection_record_hash=f"{index + 10:0>64x}",
                prediction_artifact_hash=f"{index + 20:0>64x}",
                prediction_value_hash=f"{index + 30:0>64x}",
                formation_session_count=3,
            )
            for index in range(3)
        ),
        row_sessions=row_sessions,
        row_listing_ids=row_listings,
        predicted_z=predicted,
        ordered_fold_row_counts=fold_counts,
        row_axis_hash="d" * 64,
    )
    economic = np.asarray(
        surface.targets["simple_economic_return"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    superseded_folds = (
        (np.arange(0, fold_counts[0]), np.arange(fold_counts[0], fold_counts[0] + fold_counts[1])),
        (
            np.arange(0, fold_counts[0] + fold_counts[1]),
            np.arange(fold_counts[0] + fold_counts[1], len(row_sessions)),
        ),
    )
    _, superseded = calibrate_stock_returns_cross_fitted(
        recipe=StockCalibrationRecipe.create(),
        horizon_sessions=1,
        target_identity=materialization.evidence.evidence_hash,
        score_identity="4" * 64,
        ordered_formation_sessions=row_sessions,
        scores=predicted,
        raw_economic_returns=np.ascontiguousarray(economic),
        folds=superseded_folds,
    )
    projection = AlphaSelectedScoreProjection.create(
        dossier_hash="5" * 64,
        decision_receipt_hash="6" * 64,
        program_hash="7" * 64,
        target_evidence_hash=materialization.evidence.evidence_hash,
        target_recipe_binding_hash=materialization.recipe_binding.binding_hash,
        outcome_method_binding_hash=materialization.outcome_method_binding_hash,
        horizon_sessions=1,
        admitted_method_ids=("RIDGE",),
        ordered_folds=row_axis.ordered_folds,
        formation_sessions=target_sessions,
        ordered_listing_ids=listings,
        resolved_cell_count=len(row_sessions),
        unresolved_cell_count=0,
        predicted_z_values_hash="e" * 64,
    )

    capability = installed_alpha_return_unit_capability()
    # Neither the writer nor the verifier accepts an installed capability. A
    # parameter for it would let a caller state what "installed" means and then be
    # told its artifact matches, which is the check inverted -- and a test able to
    # pass one would be defining the very state it claims to verify.
    for owner in (seal_installed_return_unit_calibration, resolve_alpha_return_unit_calibration):
        assert "capability" not in inspect.signature(owner).parameters

    # Through the installed sealer, which is the only writer: it resolves the
    # capability, hands the numerical function the environment that capability
    # names, and writes the capability's identity into the evidence.
    result = seal_installed_return_unit_calibration(
        projection=projection,
        row_axis=row_axis,
        materialization=materialization,
        superseded=superseded,
    )
    assert result.evidence.capability_binding_hash == capability.capability_hash
    assert result.evidence.implementation_closure_hash == (capability.implementation_closure_hash)
    dispersion = np.asarray(
        surface.dispersion["cross_sectional_dispersion"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    # x is exactly the inverse of the standardization, formation by formation.
    expected_x = np.repeat(dispersion, len(listings)) * predicted
    assert np.array_equal(result.scaled_scores, expected_x)
    assert result.evidence.slope_units == "DIMENSIONLESS_RETURN_PER_RETURN"
    assert result.evidence.superseded_calibration_evidence_hash == superseded.evidence_hash
    assert result.evidence.superseded_disposition == "SUPERSEDED_FOR_RETURN_UNIT_COMPOSITION"

    # Training rows are never given a calibrated value: the first outer fold has
    # nothing fitted before it, so its rows stay unresolved rather than borrowing
    # a slope fitted on themselves.
    applied = result.applied_expected_returns
    assert not np.isfinite(applied[: fold_counts[0]]).any()
    for index, (train, output) in enumerate(superseded_folds):
        slope = result.evidence.calibration.fold_states[index].slope
        assert np.allclose(applied[output], slope * result.scaled_scores[output])
        assert set(train) & set(output) == set()
    matrix = applied_expected_return_matrix(
        calibration_values=applied,
        row_axis=row_axis,
        formation_sessions=target_sessions,
        ordered_listing_ids=listings,
    )
    assert matrix.shape == (len(target_sessions), len(listings))
    assert not np.isfinite(matrix[:3]).any()

    # --- the installed capability, and a re-sealed forgery -------------------
    # The artifact proves only that it agrees with itself. Anyone who can write
    # the file can move a slope, move the applied lane beside it, recompute
    # ``evidence_hash`` over the changed payload, and publish something that
    # validates perfectly. The authority is therefore in the owner: the verifier
    # re-runs the installed method on the real inputs and admits the document only
    # if the re-derivation reproduces it.
    resolved = resolve_alpha_return_unit_calibration(
        evidence=result.evidence,
        projection=projection,
        row_axis=row_axis,
        materialization=materialization,
        superseded=superseded,
    )
    assert resolved.disposition == "REDERIVED_UNDER_INSTALLED_CAPABILITY"
    # The handles a consumer must re-project on travel with the resolution rather
    # than being rebuilt beside it.
    assert resolved.row_axis is row_axis
    assert resolved.target_recipe_id == materialization.recipe_binding.target_recipe_id
    assert resolved.rederived is True
    assert resolved.evidence.evidence_hash == result.evidence.evidence_hash
    assert resolved.applied_expected_returns is not None
    # Recomputed, never decoded: the lane a consumer gets is the one the verifier
    # produced, and it agrees with the artifact's own only because the artifact is
    # honest.
    assert np.array_equal(
        np.nan_to_num(resolved.applied_expected_returns, nan=0.0),
        np.nan_to_num(applied, nan=0.0),
    )

    # A calibration fitted on a target that is half the score: a perfect
    # correlation, a much larger slope, a different applied lane, and every hash
    # inside it consistent with every other. Nothing in the document is wrong;
    # it simply is not what the installed method produces from the real rows.
    _forged_applied, forged_calibration = calibrate_stock_returns_cross_fitted(
        recipe=StockCalibrationRecipe.create(),
        horizon_sessions=1,
        target_identity=materialization.evidence.evidence_hash,
        score_identity=result.evidence.scaled_score_identity,
        ordered_formation_sessions=row_sessions,
        scores=result.scaled_scores,
        raw_economic_returns=np.ascontiguousarray(result.scaled_scores * 0.5),
        folds=superseded_folds,
    )
    assert [value.slope for value in forged_calibration.fold_states] != [
        value.slope for value in result.evidence.calibration.fold_states
    ]
    forged = AlphaReturnUnitCalibrationEvidence.create(
        **{
            **result.evidence.model_dump(exclude={"evidence_hash", "calibration"}),
            "calibration": forged_calibration,
        }
    )
    assert forged.evidence_hash != result.evidence.evidence_hash
    with pytest.raises(AlphaReturnUnitCalibrationError, match="slope_not_rederivable"):
        resolve_alpha_return_unit_calibration(
            evidence=forged,
            projection=projection,
            row_axis=row_axis,
            materialization=materialization,
            superseded=superseded,
        )

    def _resolve(evidence):  # type: ignore[no-untyped-def]
        return resolve_alpha_return_unit_calibration(
            evidence=evidence,
            projection=projection,
            row_axis=row_axis,
            materialization=materialization,
            superseded=superseded,
        )

    def _reseal(**overrides):  # type: ignore[no-untyped-def]
        values = {
            **result.evidence.model_dump(exclude={"evidence_hash", "calibration"}),
            "calibration": result.evidence.calibration,
        }
        values.update(overrides)
        return AlphaReturnUnitCalibrationEvidence.create(**values)

    # An artifact published before the capability was durable. It never recorded
    # what produced it, so there is nothing to compare and nothing was ever
    # claimed: readable, with no applied lane for a consumer to use by accident,
    # and never promoted by re-fitting it under today's method.
    legacy = _reseal(capability_binding_hash=None, implementation_closure_hash=None)
    assert legacy.evidence_hash != result.evidence.evidence_hash
    readback = _resolve(legacy)
    assert readback.disposition == "READBACK_ONLY_NO_INSTALLED_AUTHORITY"
    assert readback.rederived is False
    assert readback.applied_expected_returns is None

    # An artifact that *did* record a capability and recorded one this build is
    # not. Two unequal hashes cannot support calling that an honest other build:
    # a capability binding is not resolvable from its hash, so nothing here can
    # tell a real other build from a value that was simply typed in. The name says
    # what is known -- mismatched, and unverified -- and the outcome is the same
    # either way, which is that nothing is admitted.
    mismatched = _reseal(capability_binding_hash="8" * 64, implementation_closure_hash="9" * 64)
    unverified = _resolve(mismatched)
    assert unverified.disposition == "READBACK_ONLY_UNVERIFIED_CAPABILITY_MISMATCH"
    assert unverified.rederived is False
    assert unverified.applied_expected_returns is None
    assert "DRIFT" not in unverified.disposition

    # Half an authority is refused outright: a document naming a capability
    # without the code it ran cannot be placed on either side of that boundary.
    # Raised inside a model validator, so what escapes is pydantic's wrapper
    # around the Desk's own ValueError subclass.
    assert issubclass(AlphaReturnUnitCalibrationError, ValueError)
    with pytest.raises(ValueError, match="authority_partial"):
        _reseal(capability_binding_hash="8" * 64, implementation_closure_hash=None)
    # The capability identity is Host-derived and has three parts, none of which a
    # caller can supply.
    assert capability.method_id == "DISPERSION_SCALED_NORMALIZED_MOMENT_NONNEGATIVE_SLOPE"
    assert (
        len(
            {
                capability.recipe_hash,
                capability.implementation_closure_hash,
                capability.numerical_environment_hash,
            }
        )
        == 3
    )

    # A same-shaped economic lane that is not the one the reconciliation admitted.
    substituted = surface.targets.set_column(
        surface.targets.schema.get_field_index("simple_economic_return"),
        "simple_economic_return",
        pa.array(np.roll(economic, 1), type=pa.float64()),
    )
    with pytest.raises(AlphaReturnUnitCalibrationError, match="economic_lane_substituted"):
        calibrate_alpha_return_unit_signal(
            projection=projection,
            row_axis=row_axis,
            materialization=dataclasses.replace(
                materialization,
                surface=dataclasses.replace(surface, targets=substituted),
            ),
            superseded=superseded,
            numerical_environment_hash=capability.numerical_environment_hash,
        )

    # A row axis of the right length whose rows are not Stage 3's rows.
    with pytest.raises(AlphaReturnUnitCalibrationError, match="row_axis_not_stage_three"):
        calibrate_alpha_return_unit_signal(
            projection=projection,
            row_axis=dataclasses.replace(row_axis, row_sessions=tuple(reversed(row_sessions))),
            materialization=materialization,
            superseded=superseded,
            numerical_environment_hash=capability.numerical_environment_hash,
        )


def test_prediction_surface_is_durable_and_identity_bound(tmp_path: Path) -> None:
    values = np.asarray([0.1, -0.2], dtype=np.float64)
    values.setflags(write=False)
    rows = ("2026-01-01|A", "2026-01-01|B")
    value_hash = alpha_model_array_content_hash(values)
    axis_hash = str(canonical_hash(rows))
    artifact_hash = alpha_campaign_prediction_artifact_hash(
        program_hash="1" * 64,
        horizon_sessions=1,
        trial_hash="2" * 64,
        fold_index=0,
        prediction_value_hash=value_hash,
        validation_row_axis_hash=axis_hash,
    )
    evidence = AlphaDevelopmentTrialEvidence.create(
        program_hash="1" * 64,
        horizon_sessions=1,
        trial_hash="2" * 64,
        fold_index=0,
        inner_selection_record_hash="6" * 64,
        training_binding_hash="3" * 64,
        estimator_content_hash="4" * 64,
        numerical_environment_hash="5" * 64,
        prediction_value_hash=value_hash,
        prediction_artifact_hash=artifact_hash,
        validation_row_axis_hash=axis_hash,
        validation_mse=0.1,
        validation_rank_ic=1.0,
        raw_economic_return_correlation=1.0,
        fit_call_count=1,
        predict_call_count=1,
        metric_call_count=3,
    )
    surface = AlphaCampaignPredictionSurface(
        trial_evidence_hash=evidence.evidence_hash,
        artifact_hash=artifact_hash,
        validation_row_ids=rows,
        predictions=values,
    )
    store = AlphaDevelopmentArtifactStore(tmp_path)
    store.publish_campaign_prediction_surface(evidence=evidence, surface=surface)
    loaded_rows, loaded_values = store.load_campaign_prediction_surface(evidence=evidence)
    assert loaded_rows == rows
    assert loaded_values == pytest.approx(values)


"""The superseded comparison used a privately declared artifact root."""


def _single_method_program(method_id: str = "RIDGE"):
    """A Program carrying exactly one model family and the successor rule."""

    domain = (
        "CHRONOLOGICAL_LIGHTGBM_REGULARIZED_GRID"
        if method_id == "LIGHTGBM_REGULARIZED"
        else "REGULARIZED_LINEAR_BOUNDED_GRID"
    )
    program, _catalog = _program(
        request=AlphaDevelopmentRequest.from_yaml(
            f"model_method_ids: [{method_id}]\n"
            f"model_search_domain_ids: [{domain}]\n"
            "scale_method_ids: [LAGGED_XS_DISPERSION, EWMA_XS_DISPERSION,"
            " ASYMMETRIC_EWMA_XS_DISPERSION]\n"
            "scale_search_domain_ids: [LAGGED_CONTROL, EWMA_HALF_LIVES_10_21_42,"
            " ASYMMETRIC_EWMA_RISE_10_DECAY_42]\n"
            "inner_selection_rule_id:"
            " MAXIMUM_INNER_NET_DAILY_RANK_IC_THEN_METHOD_ID\n"
        )
    )
    return program


def test_inner_selection_rules_disagree_and_the_loss_rule_is_unchanged() -> None:
    """Inner selection rules disagree and the loss rule is unchanged."""

    program = _single_method_program()
    trials = tuple(value.trial_hash for value in program.ordered_trials)
    scores = (
        AlphaInnerTrialScore(
            trial_hash=trials[0],
            method_id="RIDGE",
            inner_validation_mse=0.990,
            inner_net_daily_rank_ic=0.001,
            inner_discriminating_session_count=100,
            inner_constant_session_count=0,
        ),
        AlphaInnerTrialScore(
            trial_hash=trials[1],
            method_id="RIDGE",
            inner_validation_mse=0.995,
            inner_net_daily_rank_ic=0.020,
            inner_discriminating_session_count=100,
            inner_constant_session_count=0,
        ),
    )
    assert derive_alpha_inner_selection(
        program=program,
        trial_scores=scores,
        selection_rule_id="MINIMUM_INNER_MSE_THEN_METHOD_ID",
    ) == ("RIDGE", trials[0])
    assert derive_alpha_inner_selection(
        program=program,
        trial_scores=scores,
        selection_rule_id="MAXIMUM_INNER_NET_DAILY_RANK_IC_THEN_METHOD_ID",
    ) == ("RIDGE", trials[1])


def test_a_shrunk_constant_configuration_cannot_win_on_cheap_turnover() -> None:
    """A shrunk constant configuration cannot win on cheap turnover."""

    program = _single_method_program()
    trials = tuple(value.trial_hash for value in program.ordered_trials)
    degenerate = AlphaInnerTrialScore(
        trial_hash=trials[0],
        method_id="RIDGE",
        inner_validation_mse=0.999,
        inner_net_daily_rank_ic=0.050,
        inner_discriminating_session_count=10,
        inner_constant_session_count=90,
    )
    honest = AlphaInnerTrialScore(
        trial_hash=trials[1],
        method_id="RIDGE",
        inner_validation_mse=0.995,
        inner_net_daily_rank_ic=0.004,
        inner_discriminating_session_count=100,
        inner_constant_session_count=0,
    )
    assert derive_alpha_inner_selection(
        program=program,
        trial_scores=(degenerate, honest),
        selection_rule_id="MAXIMUM_INNER_NET_DAILY_RANK_IC_THEN_METHOD_ID",
    ) == ("RIDGE", trials[1])


def test_sealed_inner_selection_record_survives_the_new_members() -> None:
    """Sealed inner selection record survives the new members."""

    legacy: dict[str, Any] = {
        "trial_hash": "1" * 64,
        "method_id": "RIDGE",
        "inner_validation_mse": 0.9957086753244833,
        "inner_validation_rank_ic": 0.018586,
        "inner_residual_excess_kurtosis": 1.2170212563127816,
    }
    sealed = canonical_hash(legacy)
    score = AlphaInnerTrialScore(**legacy)
    assert score.inner_net_daily_rank_ic is None
    assert score.inner_mean_one_way_turnover is None
    identity = score.model_dump(mode="json")
    assert canonical_hash(identity) == sealed
    assert set(identity) == set(legacy)


def test_regularized_lightgbm_grid_has_no_duplicate_configuration() -> None:
    """Regularized LightGBM grid has no duplicate configuration."""

    program = _single_method_program(method_id="LIGHTGBM_REGULARIZED")
    trials = program.ordered_trials
    assert len(trials) == 72
    configurations = {
        (
            value.recipe.parameters["max_depth"],
            value.recipe.parameters["num_leaves"],
            value.recipe.parameters["feature_fraction"],
            value.recipe.parameters["bagging_fraction"],
            value.recipe.parameters["bagging_freq"],
            value.recipe.parameters["lambda_l1"],
            value.recipe.parameters["lambda_l2"],
        )
        for value in trials
    }
    assert len(configurations) == 24
    assert {value.recipe.parameters["seed"] for value in trials} == {1729, 2718, 31415}


def _curated_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "CuratedFactorCheckpoint",
        "deterministic_checkpoint_hash": "1" * 64,
        "campaign_dossier_hash": "2" * 64,
        "curation_decision_hash": "3" * 64,
        "control_factor_ids": ("alpha", "beta"),
        "curated_candidate_ids": ("gamma",),
        "ordered_factor_ids": ("alpha", "beta", "gamma"),
        "feature_source_hash": "9" * 64,
        "development_only": True,
    }
    payload.update(overrides)
    return payload


def test_curated_checkpoint_sealed_before_pruning_keeps_its_identity() -> None:
    """The members added for pruning must not move an axis sealed without them."""

    payload = _curated_payload()
    sealed = canonical_hash(payload)

    checkpoint = CuratedFactorCheckpoint(**payload, checkpoint_hash=sealed)

    assert checkpoint.redundancy_pruned_factor_ids is None
    assert checkpoint.redundancy_structure_hash is None
    dumped = checkpoint.model_dump(mode="json")
    assert "redundancy_pruned_factor_ids" not in dumped
    assert canonical_hash({k: v for k, v in dumped.items() if k != "checkpoint_hash"}) == sealed


def test_curated_checkpoint_narrows_its_axis_only_by_naming_what_it_dropped() -> None:
    payload = _curated_payload(
        ordered_factor_ids=("alpha", "gamma"),
        redundancy_structure_hash="4" * 64,
        redundancy_pruned_factor_ids=("beta",),
    )

    checkpoint = CuratedFactorCheckpoint(**payload, checkpoint_hash=canonical_hash(payload))

    # The controls record what the Desk was handed and never narrow; only the
    # axis handed on does.
    assert checkpoint.control_factor_ids == ("alpha", "beta")
    assert checkpoint.ordered_factor_ids == ("alpha", "gamma")


def test_curated_checkpoint_refuses_a_narrowed_axis_without_provenance() -> None:
    payload = _curated_payload(
        ordered_factor_ids=("alpha", "gamma"), redundancy_pruned_factor_ids=("beta",)
    )

    with pytest.raises(ValidationError, match="FACTOR_CURATED_CHECKPOINT_INVALID"):
        CuratedFactorCheckpoint(**payload, checkpoint_hash=canonical_hash(payload))


def test_curated_checkpoint_refuses_an_axis_its_pruned_set_does_not_explain() -> None:
    payload = _curated_payload(
        ordered_factor_ids=("alpha",),
        redundancy_structure_hash="4" * 64,
        redundancy_pruned_factor_ids=("beta",),
    )

    with pytest.raises(ValidationError, match="FACTOR_CURATED_CHECKPOINT_INVALID"):
        CuratedFactorCheckpoint(**payload, checkpoint_hash=canonical_hash(payload))


def _regime_request() -> AlphaDevelopmentRequest:
    return _request().model_copy(
        update={
            "model_method_ids": ("RIDGE", "LIGHTGBM"),
            "model_search_domain_ids": (
                "REGULARIZED_LINEAR_BOUNDED_GRID",
                "CHRONOLOGICAL_LIGHTGBM_BOUNDED_GRID",
            ),
            "context_axis_method_id": "MARKET_REGIME_STATE_CONTEXT",
        }
    )


def test_only_the_tree_methods_are_offered_the_regime_context() -> None:
    """Only the tree methods are offered the regime context."""

    program, _ = _program(
        ("alpha_factor", "beta_factor"),
        request=_regime_request(),
        ordered_context_ids=("market_regime__state",),
    )

    assert program.ordered_context_ids == ("market_regime__state",)
    assert program.context_axis_method_id == "MARKET_REGIME_STATE_CONTEXT"
    by_method = {
        trial.method_id: (trial.ordered_feature_ids, trial.feature_view)
        for trial in program.ordered_trials
    }
    assert by_method["LIGHTGBM"] == (
        ("alpha_factor", "beta_factor", "market_regime__state"),
        "FACTOR_AND_REGIME_CONTEXT",
    )
    assert by_method["RIDGE"] == (("alpha_factor", "beta_factor"), "FULL_AXIS")


def test_a_program_without_a_context_selection_keeps_the_factor_axis() -> None:
    program, _ = _program(
        ("alpha_factor", "beta_factor"),
        request=_regime_request().model_copy(update={"context_axis_method_id": None}),
    )

    assert program.ordered_context_ids is None
    assert all(
        trial.ordered_feature_ids == ("alpha_factor", "beta_factor")
        for trial in program.ordered_trials
    )


def test_a_context_selection_without_a_resolved_axis_is_refused() -> None:
    """A named lane that resolved to nothing must fail loudly, not silently."""

    with pytest.raises(AlphaCampaignBoundaryError, match="CONTEXT_AXIS_UNRESOLVED"):
        _program(("alpha_factor", "beta_factor"), request=_regime_request())


def test_a_context_axis_may_not_restate_a_factor() -> None:
    program, _ = _program(
        ("alpha_factor", "beta_factor"),
        request=_regime_request(),
        ordered_context_ids=("market_regime__state",),
    )

    # Re-sealed so only the disjointness rule can refuse it: a Factor restated as
    # a market state would be read twice under two different meanings.
    doctored = {**program.model_dump(mode="json"), "ordered_context_ids": ["alpha_factor"]}
    doctored.pop("program_hash")
    doctored["program_hash"] = canonical_hash(doctored)

    with pytest.raises(ValidationError, match="ALPHA_DEVELOPMENT_PROGRAM_INVALID"):
        AlphaDevelopmentProgram.model_validate(doctored)


def test_a_context_axis_without_its_provenance_is_refused() -> None:
    program, _ = _program(
        ("alpha_factor", "beta_factor"),
        request=_regime_request(),
        ordered_context_ids=("market_regime__state",),
    )
    payload = program.model_dump(mode="json")
    payload.pop("context_source_hash")
    payload.pop("program_hash")
    payload["program_hash"] = canonical_hash(payload)

    with pytest.raises(ValidationError, match="ALPHA_DEVELOPMENT_PROGRAM_INVALID"):
        AlphaDevelopmentProgram.model_validate(payload)


def _null_request() -> AlphaDevelopmentRequest:
    return _request().model_copy(
        update={
            "model_method_ids": ("EQUAL_WEIGHT_RANK_COMPOSITE",),
            "model_search_domain_ids": ("RANK_COMPOSITE_BOUNDED_GRID",),
        }
    )


def test_the_null_control_contributes_its_whole_selection_grid() -> None:
    """The null control contributes its whole selection grid."""

    program, _ = _program(("alpha_factor", "beta_factor", "gamma_factor"), request=_null_request())

    counts = {trial.method_id for trial in program.ordered_trials}
    assert counts == {"EQUAL_WEIGHT_RANK_COMPOSITE"}
    selected = tuple(
        int(trial.recipe.parameters["selected_factor_count"]) for trial in program.ordered_trials
    )
    assert selected == APPROVED_RANK_COMPOSITE_SELECTION_COUNTS
    assert all(
        trial.ordered_feature_ids == ("alpha_factor", "beta_factor", "gamma_factor")
        for trial in program.ordered_trials
    )


def test_the_null_control_selects_by_correlation_and_weights_equally() -> None:
    """requirement: no coefficient is estimated, and a negative Factor is held short."""

    generator = np.random.default_rng(11)
    ordered = ("a_weak", "b_strong_positive", "c_strong_negative", "d_noise")
    features = generator.normal(size=(400, 4))
    targets = 2.0 * features[:, 1] - 2.0 * features[:, 2] + generator.normal(scale=4.0, size=400)

    fit = fit_rank_composite(
        parameters=RankCompositeParameters(selected_factor_count=2),
        ordered_feature_ids=ordered,
        training_features=features,
        training_targets=targets,
    )

    assert set(fit.selected_feature_ids) == {"b_strong_positive", "c_strong_negative"}
    assert fit.coefficients[1] == pytest.approx(0.5)
    # A Factor's published sign is a convention; a negative one is a short book.
    assert fit.coefficients[2] == pytest.approx(-0.5)
    assert fit.coefficients[0] == 0.0 and fit.coefficients[3] == 0.0
    assert float(np.abs(fit.coefficients).sum()) == pytest.approx(1.0)


def test_the_null_control_breaks_ties_on_the_factor_id() -> None:
    """Two identical columns must not let column order decide the composite."""

    generator = np.random.default_rng(3)
    column = generator.normal(size=400)
    features = np.column_stack([column, column, generator.normal(size=400)])
    targets = column + generator.normal(scale=0.5, size=400)

    forward = fit_rank_composite(
        parameters=RankCompositeParameters(selected_factor_count=1),
        ordered_feature_ids=("z_last", "a_first", "noise"),
        training_features=features,
        training_targets=targets,
    )

    assert forward.selected_feature_ids == ("a_first",)


def test_the_null_control_refuses_a_selection_the_domain_never_admitted() -> None:
    adapter = RankCompositeAdapter()
    domain = build_rank_composite_search_domain()

    with pytest.raises(ValueError, match="OUTSIDE_SEARCH_DOMAIN"):
        adapter.validate_recipe_for_domain(
            recipe=build_rank_composite_recipe(RankCompositeParameters(selected_factor_count=7)),
            domain=domain,
        )


def test_the_null_control_refuses_more_factors_than_the_axis_holds() -> None:
    with pytest.raises(ValueError, match="SELECTION_EXCEEDS_AXIS"):
        fit_rank_composite(
            parameters=RankCompositeParameters(selected_factor_count=5),
            ordered_feature_ids=("only", "two"),
            training_features=np.zeros((10, 2)),
            training_targets=np.zeros(10),
        )


def _aggregation_surface(
    *, sessions: int = 12, listings: int = 5, absent: tuple[tuple[int, int], ...] = ()
) -> tuple[tuple[date, ...], tuple[str, ...], npt.NDArray[np.float64]]:
    """A dense score surface with named holes, for the aggregation requirements."""

    generator = np.random.default_rng(20260821)
    row_sessions: list[date] = []
    row_listings: list[str] = []
    values: list[float] = []
    for index in range(sessions):
        for member in range(listings):
            if (index, member) in absent:
                continue
            row_sessions.append(date(2024, 1, 1) + timedelta(days=index))
            row_listings.append(f"L{member}")
            values.append(float(generator.normal()))
    scores = np.ascontiguousarray(values, dtype=np.float64)
    scores.setflags(write=False)
    return tuple(row_sessions), tuple(row_listings), scores


def test_the_control_span_leaves_the_score_bitwise_unchanged() -> None:
    sessions, listings, scores = _aggregation_surface()

    aggregated, depth = aggregate_trailing_mean(
        row_sessions=sessions,
        row_listing_ids=listings,
        scores=scores,
        trailing_span_sessions=AGGREGATION_CONTROL_SPAN,
    )

    assert aggregated.tobytes() == scores.tobytes()
    assert set(depth.tolist()) == {1}


def test_an_aggregate_cannot_read_a_later_session() -> None:
    sessions, listings, scores = _aggregation_surface()
    boundary = sessions[len(sessions) // 2]
    disturbed = np.array(scores, dtype=np.float64, copy=True)
    later = np.asarray([value > boundary for value in sessions])
    disturbed[later] += 1000.0
    disturbed.setflags(write=False)

    original, _depth = aggregate_trailing_mean(
        row_sessions=sessions, row_listing_ids=listings, scores=scores, trailing_span_sessions=21
    )
    moved, _moved_depth = aggregate_trailing_mean(
        row_sessions=sessions, row_listing_ids=listings, scores=disturbed, trailing_span_sessions=21
    )

    earlier = ~later
    assert original[earlier].tobytes() == moved[earlier].tobytes()
    assert not np.array_equal(original[later], moved[later])


def test_an_absent_session_contributes_nothing_and_is_never_filled() -> None:
    hole = (2, 1)
    sessions, listings, scores = _aggregation_surface(absent=(hole,))

    aggregated, depth = aggregate_trailing_mean(
        row_sessions=sessions, row_listing_ids=listings, scores=scores, trailing_span_sessions=21
    )

    rows = [
        index
        for index, (session, listing) in enumerate(zip(sessions, listings, strict=True))
        if listing == f"L{hole[1]}"
    ]
    # Three sessions have passed and the listing was scored on two of them, so
    # the window holds two values: the absent session is not filled, and it does
    # not let the window reach further back to make up the depth.
    assert int(depth[rows[2]]) == 3
    assert aggregated[rows[2]] == pytest.approx(float(np.mean(scores[rows[:3]])))
    assert date(2024, 1, 3) not in {sessions[index] for index in rows}


def test_a_span_outside_the_installed_domain_is_refused() -> None:
    sessions, listings, scores = _aggregation_surface()

    with pytest.raises(AlphaScoreAggregationError, match="span_not_in_domain"):
        aggregate_trailing_mean(
            row_sessions=sessions,
            row_listing_ids=listings,
            scores=scores,
            trailing_span_sessions=30,
        )


def test_aggregation_program_evidence_and_surface_have_durable_readback(
    tmp_path: Path,
) -> None:
    # historical-readback: seal a frozen schema without executing its retired selector.
    values = {
        "method_id": "TRAILING_MEAN_FORMATION_SCORE",
        "candidate_span_domain": (1, 21, 42, 63),
        "control_span_sessions": 1,
        "selection_rule_id": (
            "MAXIMUM_MATURED_PRIOR_FOLD_NET_DECILE_SPREAD_5BPS_THEN_SMALLEST_SPAN"
        ),
        "window_unit": "FORMATION_SESSIONS_ON_SURFACE_AXIS",
        "minimum_periods_policy": "ONE_ROW_AXIS_INVARIANT",
        "absence_policy": "ABSENT_SESSION_CONTRIBUTES_NOTHING_NO_FILL",
        "tie_policy": "TIES_PRESERVED_NO_TIE_BREAK_MANUFACTURED",
        "selection_lane_id": "RAW_ECONOMIC_SIMPLE_RETURN",
        "selection_cost_bps": 5.0,
        "maturity_policy": "OUTCOME_MATURITY_STRICTLY_BEFORE_CURRENT_FOLD_FIRST_FORMATION",
        "capability_hash": "a" * 64,
        "implementation_binding_hash": "b" * 64,
        "installed_catalog_hash": "c" * 64,
    }
    provisional = AlphaScoreAggregationProgram.model_construct(
        **values,
        program_hash="0" * 64,
    )
    program_values = provisional.model_dump(mode="json", exclude={"program_hash"})
    program = AlphaScoreAggregationProgram(
        **program_values,
        program_hash=canonical_hash(program_values),
    )
    sessions = (date(2025, 1, 2), date(2025, 1, 3))
    listings = ("listing-a", "listing-b")
    aggregated = np.asarray((0.25, -0.5), dtype=np.float64)
    aggregated.setflags(write=False)
    selection = AlphaScoreAggregationFoldSelection(
        fold_index=0,
        selected_span_sessions=1,
        selection_basis="DECLARED_CONTROL_NO_PRIOR_EVIDENCE",
        prior_fold_row_count=0,
        prior_fold_session_count=0,
        matured_prior_fold_row_count=0,
        matured_prior_fold_session_count=0,
        current_fold_first_formation_session=sessions[0],
        scored_row_count=2,
        candidate_criteria=((1, None), (21, None), (42, None), (63, None)),
    )
    evidence = AlphaScoreAggregationEvidence.create(
        program_hash=program.program_hash,
        methodology_id="HISTORICAL_TEST_ARM",
        horizon_sessions=1,
        upstream_score_projection_hash="1" * 64,
        upstream_dossier_hash="2" * 64,
        upstream_decision_receipt_hash="3" * 64,
        target_recipe_binding_hash="4" * 64,
        outcome_snapshot_hash="5" * 64,
        outcome_method_binding_hash="6" * 64,
        execution_clock_id="OPEN_T_PLUS_ONE_TO_FOLLOWING_OPEN",
        row_axis_hash=row_axis_hash(sessions, listings),
        fold_axis_hash=canonical_hash((0, 0)),
        maturity_axis_hash=canonical_hash(("2025-01-03", "2025-01-06")),
        input_score_value_hash=score_value_hash(aggregated),
        raw_economic_return_value_hash=score_value_hash(aggregated),
        aggregated_score_value_hash=score_value_hash(aggregated),
        ordered_fold_selections=(selection,),
        mean_realized_depth=1.0,
        minimum_realized_depth=1,
    )
    store = AlphaDevelopmentArtifactStore(tmp_path)
    surface = AlphaScoreAggregationSurface.create(
        evidence_hash=evidence.evidence_hash,
        program_hash=evidence.program_hash,
        row_axis_hash=evidence.row_axis_hash,
        aggregated_score_value_hash=evidence.aggregated_score_value_hash,
        row_count=len(sessions),
    )
    for category, value, identity in (
        ("development/score-aggregation/programs", program, program.program_hash),
        ("development/score-aggregation/evidence", evidence, evidence.evidence_hash),
        ("development/score-aggregation/surfaces", surface, surface.surface_hash),
    ):
        target = store.root / category / f"{identity}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(value.model_dump(mode="json"), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = (
        store.root / "development/score-aggregation/surfaces" / f"{surface.surface_hash}.parquet"
    )
    pq.write_table(
        pa.table(
            {
                "row_id": pa.array(
                    [
                        f"{session.isoformat()}|{listing}"
                        for session, listing in zip(sessions, listings, strict=True)
                    ],
                    type=pa.string(),
                ),
                "aggregated_score": pa.array(aggregated, type=pa.float64()),
            }
        ),
        parquet,
        compression="zstd",
    )

    assert store.load_score_aggregation_program(program.program_hash) == program
    assert store.load_score_aggregation_evidence(evidence.evidence_hash) == evidence
    loaded_surface, row_ids, values = store.load_score_aggregation_surface(surface.surface_hash)
    assert loaded_surface.evidence_hash == evidence.evidence_hash
    assert row_ids[0] == f"{sessions[0].isoformat()}|{listings[0]}"
    assert values.tobytes() == aggregated.tobytes()
