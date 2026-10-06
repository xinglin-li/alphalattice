"""Small frozen numerical fixtures for Alpha Research acceptance tests."""

from __future__ import annotations

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
from alphalattice.investment.alpha_research.inputs.folds import AlphaFoldArrays, PreparedAlphaArrays
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
    value.setflags(write=False)
    return value


def synthetic_prepared_arrays() -> PreparedAlphaArrays:
    sessions = tuple(date(2025, 1, 1) + timedelta(days=index) for index in range(11))
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
    scales = np.asarray([10.0 ** ((index % 7) - 3) for index in range(len(FIXTURE_FACTOR_IDS))])
    folds: list[AlphaFoldArrays] = []
    for window in plan.windows:
        train_keys = tuple(
            (session, listing) for session in window.train_sessions for listing in listings
        )
        validation_keys = tuple(
            (session, listing) for session in window.validation_sessions for listing in listings
        )

        def matrix(keys: tuple[tuple[date, str], ...]) -> tuple[np.ndarray, np.ndarray]:
            features = np.empty((len(keys), len(FIXTURE_FACTOR_IDS)), dtype=np.float64)
            targets = np.empty(len(keys), dtype=np.float64)
            for row_index, (session, listing) in enumerate(keys):
                listing_number = int(listing.rsplit("-", 1)[1])
                session_number = sessions.index(session)
                base = np.asarray(
                    [
                        ((listing_number + 3 * factor + session_number) % 37 - 18) / 18.0
                        for factor in range(len(FIXTURE_FACTOR_IDS))
                    ],
                    dtype=np.float64,
                )
                features[row_index] = base * scales
                targets[row_index] = (
                    0.004 * listing_number
                    + 0.01 * session_number
                    + float(np.dot(base, np.linspace(-0.03, 0.04, len(FIXTURE_FACTOR_IDS))))
                )
            return _readonly(features), _readonly(targets)

        train_features, train_targets = matrix(train_keys)
        validation_features, validation_targets = matrix(validation_keys)
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
    foundation = ResearchFoundationBinding.model_construct(
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
        foundation_hash=FIXTURE_FOUNDATION_HASH,
    )
    cards = load_registered_alpha_catalog(inventory, ordered_factor_ids=FIXTURE_FACTOR_IDS)
    request = build_alpha_experiment_request(
        foundation=foundation,
        inventory=inventory,
        cards=cards,
        listing_set_hash=canonical_hash(FIXTURE_LISTING_IDS),
        ordered_listing_ids=FIXTURE_LISTING_IDS,
    )
    return request, inventory


__all__ = [
    "registered_test_cards",
    "synthetic_prepared_arrays",
    "synthetic_request",
]
