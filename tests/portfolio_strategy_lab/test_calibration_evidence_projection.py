"""Calibration binds complete causal evidence while consuming only its numerical columns."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np

from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceCalibrationInput,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    STAGES,
    CalibrationPlan,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import planned_local_qa_schedule
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenComponentScoreSnapshot,
    seal_current_contract,
)
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    FrozenRankCalibrationRule,
    load_observations,
    publish_observations,
)
from tests.portfolio_strategy_lab.local_web_support import TEST_PACKAGE


def test_calibration_keeps_full_evidence_when_projected_values_agree(live):
    owner = live.operations.calibration
    assert owner is not None
    schedule = planned_local_qa_schedule(date(2026, 6, 1), date(2026, 8, 4))
    axis = tuple(point.formation_session for point in schedule)
    days = axis[-4:]
    ids = tuple(f"synthetic-listing-{index:03d}" for index in range(100))
    values = np.tile(np.arange(len(ids), dtype=np.float64), (3, 1))
    seed = publish_observations(
        owner.store,
        scores=values,
        returns=np.full_like(values, np.nan),
        eligible=np.ones(values.shape, dtype=np.bool_),
        strategy_package_hash=TEST_PACKAGE.package_hash,
        component_recipe_hash="b" * 64,
        rule=FrozenRankCalibrationRule.create(activation_session=date(2021, 9, 7)),
        formation_sessions=days[:3],
        holding_end_sessions=(None,) * 3,
        ordered_listing_ids=ids,
        source_hashes=("c" * 64,),
        origin="SYNTHETIC_QA_INPUTS",
    )
    score = seal_current_contract(
        FrozenComponentScoreSnapshot,
        {
            "request_hash": "d" * 64,
            "strategy_package_hash": TEST_PACKAGE.package_hash,
            "component_recipe_hash": "b" * 64,
            "inference_authority_hash": "c" * 64,
            "observation_snapshot_hash": "e" * 64,
            "formation_session": days[-1],
            "ordered_listing_ids": ids,
            "feature_values_hashes": (),
            "model_identity_hashes": (),
            "projection_hash": "f" * 64,
            "scores": tuple(map(float, range(len(ids)))),
            "live": (True,) * len(ids),
            "prediction_calls": 0,
        },
        "snapshot_hash",
    )
    owner.scoring.store.publish_frozen_component_score(score)
    plan = CalibrationPlan.create(
        workspace_manifest_hash=owner.manifest.manifest_hash,
        binding=ResearchWorkspaceCalibrationInput(
            strategy_package_id=TEST_PACKAGE.strategy_id,
            strategy_package_hash=TEST_PACKAGE.package_hash,
            seed_hash=seed.content_hash,
            source_kind="WORKSPACE_DATA_FEATURE",
        ),
        score_snapshot_hash=score.snapshot_hash,
        history_score_hashes=(score.snapshot_hash,),
        source_hash="c" * 64,
        implementation_hash="f" * 64,
    )
    captured = LocalQAMarketSnapshot.create(
        source_hash=plan.source_hash,
        through=days[-1],
        ordered_listing_ids=ids,
        schedule=schedule,
        bars=tuple(
            RawDailyBar(
                listing_id=listing,
                provider="SYNTHETIC_QA",
                session_date=day,
                open=100.0 + index,
                high=102.0 + index,
                low=99.0 + index,
                close=101.0 + index,
                volume=100_000,
            )
            for index, day in enumerate(axis)
            for listing in ids
        ),
        actions=(),
    )
    corrected = LocalQAMarketSnapshot.create(
        source_hash=captured.source_hash,
        through=captured.through,
        ordered_listing_ids=captured.ordered_listing_ids,
        schedule=captured.schedule,
        bars=tuple(
            replace(bar, volume=100_001)
            if bar.listing_id == ids[0] and bar.session_date == days[1]
            else bar
            for bar in captured.bars
        ),
        actions=captured.actions,
    )
    results = []
    for snapshot in (captured, corrected):
        stage = owner.execute_step(
            plan, STAGES[1], lambda _stage: seed.content_hash, captured=snapshot
        )
        results.append(load_observations(owner.store, stage.evidence[0].content_hash))
    original, changed = results[0][0], results[1][0]
    assert original.formation_sessions == changed.formation_sessions
    assert original.holding_end_sessions == changed.holding_end_sessions
    assert original.ordered_listing_ids == changed.ordered_listing_ids
    assert original.array_hash == changed.array_hash
    for original_values, changed_values in zip(results[0][1:], results[1][1:], strict=True):
        np.testing.assert_array_equal(original_values, changed_values)
    assert np.isfinite(results[0][2]).any()
    assert original.source_hashes[:2] == changed.source_hashes[:2]
    assert original.source_hashes[-1] != changed.source_hashes[-1]
    assert original.content_hash != changed.content_hash
