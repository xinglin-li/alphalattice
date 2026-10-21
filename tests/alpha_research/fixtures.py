"""Small frozen numerical fixtures for Alpha Research acceptance tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np

from alphalattice.foundation.research_foundation.contracts import (
    ResearchDeskExecutionOutcomeRef,
    ResearchFoundationBinding,
)
from alphalattice.investment.alpha_research.experiments.bindings import (
    build_alpha_experiment_request,
)
from alphalattice.investment.alpha_research.experiments.contracts import (
    AlphaCandidateRole,
    AlphaExperimentCard,
    AlphaExperimentRequest,
    AlphaFoldCommitment,
    seal_contract,
)
from alphalattice.investment.alpha_research.inputs.folds import (
    AlphaCurrentRefitArrays,
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
    PreparedAlphaArrays,
)
from alphalattice.investment.alpha_research.inputs.training import (
    build_alpha_training_input_binding,
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
from tests.alpha_research.candidate_inventories import load_initial_candidate_inventory
from tests.alpha_research.registered_catalog import (
    load_registered_alpha_catalog,
)

FIXTURE_FOUNDATION_HASH = "390ca69c479b0e7f9d2b02e057c0e45b96d1df60f7d8e7e0e25334a161ee7a83"
FIXTURE_FACTOR_IDS = (
    "dollar_volume_252",
    "dollar_volume_21",
    "downside_beta_252",
    "rev_5",
    "market_corr_252",
    "mom_126_21",
    "residual_mom_252_21",
    "momentum_consistency_252",
    "mom_252_21",
    "market_corr_63",
    "beta_63",
    "coskew_252",
    "skew_252",
    "trend_r2_252",
    "price_volume_corr_63",
    "volume_zscore_21",
    "cmf_21",
    "stoch_k_14",
    "directional_strength_14",
    "obv_slope_63",
)
FIXTURE_LISTING_IDS = tuple(f"listing-{index:03d}" for index in range(100))


def registered_test_cards() -> tuple[AlphaExperimentCard, ...]:
    rows = (
        {
            "kind": "AlphaExperimentCard",
            "candidate_id": "benchmark.zero-forecast",
            "source_card_hash": "1" * 64,
            "family_id": "zero_forecast",
            "role": AlphaCandidateRole.BENCHMARK,
            "feature_shape": "NONE",
            "factor_count": 0,
            "secondary_feature_preprocessing": False,
            "package_identity": "alpha-research-deterministic",
        },
        {
            "kind": "AlphaExperimentCard",
            "candidate_id": "benchmark.historical-mean",
            "source_card_hash": "2" * 64,
            "family_id": "historical_mean",
            "role": AlphaCandidateRole.BENCHMARK,
            "feature_shape": "NONE",
            "factor_count": 0,
            "secondary_feature_preprocessing": False,
            "package_identity": "alpha-research-deterministic",
        },
        *(
            {
                "kind": "AlphaExperimentCard",
                "candidate_id": candidate_id,
                "source_card_hash": source_hash * 64,
                "family_id": "ridge",
                "role": AlphaCandidateRole.REGULARIZED_ALPHA,
                "feature_shape": "FROZEN_ORDERED",
                "factor_count": len(FIXTURE_FACTOR_IDS),
                "alpha": alpha,
                "fit_intercept": True,
                "solver": "svd",
                "tolerance": 1e-8,
                "deterministic_seed": 1729,
                "secondary_feature_preprocessing": False,
                "package_identity": "scikit-learn==1.9.0",
            }
            for candidate_id, source_hash, alpha in (
                ("model.ridge.alpha-1p0", "3", 1.0),
                ("model.ridge.alpha-10p0", "4", 10.0),
            )
        ),
    )
    return tuple(seal_contract(AlphaExperimentCard, value, "card_hash") for value in rows)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


FIXTURE_SESSIONS = tuple(date(2025, 1, 1) + timedelta(days=index) for index in range(11))
_FIXTURE_SCALES = _readonly(
    np.asarray([10.0 ** ((index % 7) - 3) for index in range(len(FIXTURE_FACTOR_IDS))])
)


def _fixture_matrix(
    keys: tuple[tuple[date, str], ...], *, observed: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((len(keys), len(FIXTURE_FACTOR_IDS)), dtype=np.float64)
    targets = np.empty(len(keys), dtype=np.float64)
    for row_index, (session, listing) in enumerate(keys):
        listing_number = int(listing.rsplit("-", 1)[1])
        session_number = FIXTURE_SESSIONS.index(session)
        base = np.asarray(
            [
                ((listing_number + 3 * factor + session_number) % 37 - 18) / 18.0
                for factor in range(len(FIXTURE_FACTOR_IDS))
            ],
            dtype=np.float64,
        )
        features[row_index] = base * _FIXTURE_SCALES
        targets[row_index] = (
            0.004 * listing_number
            + 0.01 * session_number
            + float(np.dot(base, np.linspace(-0.03, 0.04, len(FIXTURE_FACTOR_IDS))))
            if observed
            else np.nan
        )
    return _readonly(features), _readonly(targets)


def synthetic_prepared_arrays() -> PreparedAlphaArrays:
    sessions = FIXTURE_SESSIONS
    frozen_at = datetime(2025, 1, 31, tzinfo=UTC)
    timeline = ValidationTimeline(
        points=tuple(
            RebalanceEvidencePoint(
                session_date=value,
                evidence_available_at=frozen_at,
                benchmark_available_at=frozen_at,
            )
            for value in sessions
        )
    )
    plan = build_research_split(
        timeline,
        ResearchSplitSpec(
            mode=SplitMode.ROLLING,
            as_of_timestamp=frozen_at,
            train_sessions=3,
            validation_sessions=2,
            step_sessions=2,
            purge_sessions=1,
            embargo_sessions=0,
            holdout_sessions=2,
            minimum_folds=2,
        ),
    )
    listings = tuple(f"listing-{index:03d}" for index in range(100))
    folds: list[AlphaFoldArrays] = []
    for window in plan.windows:
        train_keys = tuple(
            (session, listing) for session in window.train_sessions for listing in listings
        )
        validation_keys = tuple(
            (session, listing) for session in window.validation_sessions for listing in listings
        )

        train_features, train_targets = _fixture_matrix(train_keys)
        validation_features, validation_targets = _fixture_matrix(validation_keys)
        train_feature_complete = _readonly(np.ones(len(train_keys), dtype=np.bool_))
        train_outcome_complete = _readonly(np.ones(len(train_keys), dtype=np.bool_))
        validation_feature_complete = _readonly(np.ones(len(validation_keys), dtype=np.bool_))
        validation_outcome_complete = _readonly(np.ones(len(validation_keys), dtype=np.bool_))
        commitment_values = {
            "fold_index": window.fold_index,
            "train_first": window.train_sessions[0],
            "train_last": window.train_sessions[-1],
            "validation_first": window.validation_sessions[0],
            "validation_last": window.validation_sessions[-1],
            "train_session_count": len(window.train_sessions),
            "validation_session_count": len(window.validation_sessions),
            "train_sessions_hash": canonical_hash(window.train_sessions),
            "validation_sessions_hash": canonical_hash(window.validation_sessions),
        }
        folds.append(
            AlphaFoldArrays(
                commitment=seal_contract(AlphaFoldCommitment, commitment_values, "commitment_hash"),
                ordered_factor_ids=FIXTURE_FACTOR_IDS,
                training_sessions=tuple(window.train_sessions),
                training_listing_ids=tuple(key[1] for key in train_keys),
                training_features=train_features,
                training_targets=train_targets,
                training_feature_complete=train_feature_complete,
                training_outcome_complete=train_outcome_complete,
                validation_sessions=tuple(window.validation_sessions),
                validation_row_sessions=tuple(key[0] for key in validation_keys),
                validation_listing_ids=tuple(key[1] for key in validation_keys),
                validation_features=validation_features,
                validation_targets=validation_targets,
                validation_feature_complete=validation_feature_complete,
                validation_outcome_complete=validation_outcome_complete,
                validation_feature_row_hashes=tuple(
                    canonical_hash(("feature", *key)) for key in validation_keys
                ),
                validation_outcome_row_hashes=tuple(
                    canonical_hash(("outcome", *key)) for key in validation_keys
                ),
            )
        )
    return PreparedAlphaArrays(split_plan=plan, folds=tuple(folds))


def synthetic_request() -> tuple[AlphaExperimentRequest, object]:
    """The fixture's experiment request over the four-card inventory, with that inventory."""
    inventory = load_initial_candidate_inventory()
    foundation = _foundation(foundation_hash=FIXTURE_FOUNDATION_HASH)
    cards = load_registered_alpha_catalog(inventory, ordered_factor_ids=FIXTURE_FACTOR_IDS)
    request = build_alpha_experiment_request(
        foundation=foundation,
        inventory=inventory,
        cards=cards,
        listing_set_hash=canonical_hash(FIXTURE_LISTING_IDS),
        ordered_listing_ids=FIXTURE_LISTING_IDS,
    )
    return request, inventory


def _foundation(foundation_hash: str = "9" * 64) -> ResearchFoundationBinding:
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
        ordered_factor_ids=FIXTURE_FACTOR_IDS,
        downstream_factor_research_forbidden=True,
        secondary_feature_preprocessing_forbidden=True,
        foundation_hash=foundation_hash,
    )


def _bound_fold(fold: AlphaFoldArrays, foundation: ResearchFoundationBinding) -> AlphaFoldArrays:
    training_row_sessions = tuple(
        session
        for session in fold.training_sessions
        for _listing in sorted(set(fold.training_listing_ids))
    )
    training_feature_hashes = tuple(
        canonical_hash(("feature", session, listing))
        for session, listing in zip(training_row_sessions, fold.training_listing_ids, strict=True)
    )
    training_outcome_hashes = tuple(
        canonical_hash(("outcome", session, listing))
        for session, listing in zip(training_row_sessions, fold.training_listing_ids, strict=True)
    )
    training_economic = _readonly(fold.training_targets.copy())
    validation_economic = _readonly(fold.validation_targets.copy())
    binding = build_alpha_training_input_binding(
        scope="DEVELOPMENT_FOLD",
        foundation_hash=foundation.foundation_hash,
        feature_panel_snapshot_hash=foundation.feature_panel_snapshot_hash,
        causal_outcome_snapshot_hash=foundation.execution_outcome.snapshot_hash,
        target_policy=None,
        ordered_feature_ids=fold.ordered_factor_ids,
        feature_context_hash=None,
        training_row_sessions=training_row_sessions,
        training_row_listing_ids=fold.training_listing_ids,
        prediction_row_sessions=fold.validation_row_sessions,
        prediction_row_listing_ids=fold.validation_listing_ids,
        training_features=fold.training_features,
        training_targets=fold.training_targets,
        training_mask=fold.training_model_mask,
        prediction_features=fold.validation_features,
        prediction_targets=fold.validation_targets,
        prediction_mask=fold.validation_feature_complete,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        prediction_feature_row_hashes=fold.validation_feature_row_hashes,
        prediction_outcome_row_hashes=fold.validation_outcome_row_hashes,
        training_economic_returns=training_economic,
        prediction_economic_returns=validation_economic,
        training_cutoff=fold.training_sessions[-1],
        outcome_maturity_session=fold.training_sessions[-1],
        prediction_anchor=fold.validation_sessions[0],
        fold_commitment_hash=fold.commitment.commitment_hash,
    )
    return replace(
        fold,
        training_economic_returns=training_economic,
        validation_economic_returns=validation_economic,
        training_row_sessions=training_row_sessions,
        training_feature_row_hashes=training_feature_hashes,
        training_outcome_row_hashes=training_outcome_hashes,
        training_input_binding=binding,
    )


class _ArrayWorkspace:
    def __init__(
        self, folds: tuple[AlphaFoldArrays, ...], *, current: AlphaCurrentRefitArrays | None = None
    ) -> None:
        self.folds = folds
        self.current = current

    @contextmanager
    def fold_lease(self, fold_index: int):
        yield self.folds[fold_index]

    def load_fold(self, fold_index: int) -> AlphaFoldArrays:
        return self.folds[fold_index]

    def prepare_current_refit(self) -> AlphaCurrentRefitArrays:
        if self.current is None:
            raise AssertionError("current refit is outside this fixture")
        return self.current


def _plan_and_workspace() -> tuple[AlphaFoldArrayPlan, _ArrayWorkspace]:
    request, _inventory = synthetic_request()
    prepared = synthetic_prepared_arrays()
    foundation = _foundation()
    plan = AlphaFoldArrayPlan(
        foundation=foundation,
        ordered_listing_ids=request.ordered_listing_ids,
        feature_reader=object(),  # type: ignore[arg-type]
        outcome_reader=object(),  # type: ignore[arg-type]
        feature_panel_manifest_ref="playpen://fixture/panel",
        causal_outcome_manifest_ref="playpen://fixture/outcome",
        split_plan=prepared.split_plan,
    )
    folds = tuple(_bound_fold(value, foundation) for value in prepared.folds)
    return plan, _ArrayWorkspace(folds)


def _bind_current_refit(
    plan: AlphaFoldArrayPlan, arrays: AlphaCurrentRefitArrays
) -> AlphaCurrentRefitArrays:
    listing_count = len(arrays.ordered_listing_ids)
    missing = _readonly(np.full(listing_count, np.nan, dtype=np.float64))
    binding = build_alpha_training_input_binding(
        scope="CURRENT_REFIT",
        foundation_hash=plan.foundation.foundation_hash,
        feature_panel_snapshot_hash=plan.foundation.feature_panel_snapshot_hash,
        causal_outcome_snapshot_hash=plan.foundation.execution_outcome.snapshot_hash,
        target_policy=None,
        ordered_feature_ids=arrays.ordered_factor_ids,
        feature_context_hash=None,
        training_row_sessions=arrays.training_row_sessions,
        training_row_listing_ids=arrays.ordered_listing_ids * len(arrays.training_sessions),
        prediction_row_sessions=(arrays.formation_session,) * listing_count,
        prediction_row_listing_ids=arrays.ordered_listing_ids,
        training_features=arrays.training_features,
        training_targets=arrays.training_targets,
        training_mask=arrays.training_mask,
        prediction_features=arrays.current_features,
        prediction_targets=missing,
        prediction_mask=arrays.current_feature_complete,
        training_feature_row_hashes=arrays.training_feature_row_hashes,
        training_outcome_row_hashes=arrays.training_outcome_row_hashes,
        prediction_feature_row_hashes=arrays.current_feature_row_hashes,
        prediction_outcome_row_hashes=(None,) * listing_count,
        training_economic_returns=arrays.training_economic_returns,
        prediction_economic_returns=missing,
        training_cutoff=arrays.training_cutoff,
        outcome_maturity_session=arrays.formation_session,
        prediction_anchor=arrays.formation_session,
        fold_commitment_hash=None,
    )
    return replace(arrays, training_input_binding=binding)


def _current_refit_fixture() -> tuple[AlphaFoldArrays, AlphaCurrentRefitArrays]:
    """A fold of the synthetic plan and the current window its validation rows open."""

    plan, workspace = _plan_and_workspace()
    fold = workspace.folds[0]
    assert fold.training_input_binding is not None
    listing_count = len(set(fold.validation_listing_ids))
    current_features = fold.validation_features[:listing_count]
    current_complete = fold.validation_feature_complete[:listing_count]
    current_listings = fold.validation_listing_ids[:listing_count]
    current_feature_hashes = fold.validation_feature_row_hashes[:listing_count]
    assert fold.training_economic_returns is not None
    arrays = AlphaCurrentRefitArrays(
        ordered_factor_ids=fold.ordered_factor_ids,
        ordered_listing_ids=current_listings,
        training_sessions=fold.training_sessions,
        training_cutoff=fold.training_sessions[-1],
        formation_session=fold.validation_sessions[0],
        training_features=fold.training_features,
        training_targets=fold.training_targets,
        training_mask=fold.training_model_mask,
        current_features=current_features,
        current_feature_complete=current_complete,
        training_row_sessions=fold.training_row_sessions,
        training_feature_row_hashes=fold.training_feature_row_hashes,
        training_outcome_row_hashes=fold.training_outcome_row_hashes,
        current_feature_row_hashes=current_feature_hashes,
        training_economic_returns=fold.training_economic_returns,
    )
    return fold, _bind_current_refit(plan, arrays)


def qualification_inputs() -> tuple[AlphaFoldArrayPlan, _ArrayWorkspace]:
    """Bound array double: latest features and declared available non-holdout labels."""
    plan, workspace = _plan_and_workspace()
    plan = replace(plan, panel_sessions=FIXTURE_SESSIONS)
    formation = plan.panel_sessions[-1]
    # Synthetic development labels are declared available by formation;
    # the source maturity reader itself is outside this array double.
    development = tuple(
        day for day in plan.panel_sessions if day < plan.split_plan.sealed_holdout.first_session
    )
    training = development[-plan.split_plan.spec.train_sessions :]
    keys = tuple((day, listing) for day in training for listing in plan.ordered_listing_ids)
    current_keys = tuple((formation, listing) for listing in plan.ordered_listing_ids)
    features, targets = _fixture_matrix(keys)
    current_features, _unread = _fixture_matrix(current_keys, observed=False)
    arrays = AlphaCurrentRefitArrays(
        ordered_factor_ids=plan.base_feature_ids,
        ordered_listing_ids=plan.ordered_listing_ids,
        training_sessions=training,
        training_cutoff=training[-1],
        formation_session=formation,
        training_features=features,
        training_targets=targets,
        training_mask=_readonly(np.ones(len(keys), dtype=np.bool_)),
        current_features=current_features,
        current_feature_complete=_readonly(np.ones(len(current_keys), dtype=np.bool_)),
        training_row_sessions=tuple(day for day, _listing in keys),
        training_feature_row_hashes=tuple(canonical_hash(("feature", *key)) for key in keys),
        training_outcome_row_hashes=tuple(canonical_hash(("outcome", *key)) for key in keys),
        current_feature_row_hashes=tuple(canonical_hash(("feature", *key)) for key in current_keys),
        training_economic_returns=targets,
    )
    return plan, _ArrayWorkspace(workspace.folds, current=_bind_current_refit(plan, arrays))


__all__ = [
    "qualification_inputs",
    "registered_test_cards",
    "synthetic_prepared_arrays",
    "synthetic_request",
]


def research_foundation_binding(factor_ids: tuple[str, ...]) -> ResearchFoundationBinding:
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
        ordered_factor_ids=factor_ids,
        downstream_factor_research_forbidden=True,
        secondary_feature_preprocessing_forbidden=True,
        foundation_hash="9" * 64,
    )
