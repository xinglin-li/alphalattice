"""Calibration binds complete causal evidence while consuming only its numerical columns."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np

from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    decision_eligible_at_close,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceCalibrationInput,
)
from alphalattice.control.product_host.composition.strategy_calibration import (
    STAGES,
    CalibrationPlan,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import (
    local_qa_snapshot_rows,
    planned_local_qa_schedule,
)
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenComponentScoreSnapshot,
    seal_current_contract,
)
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    FrozenRankCalibrationRule,
    load_observations,
    publish_consumed_observations,
    publish_observations,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
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


def test_calibration_keeps_dated_membership_and_score_refusals_with_scalar_eligibility(live):
    """requirement: batched market qualification cannot revive a missing member or false score."""

    owner = live.operations.calibration
    assert owner is not None
    schedule = planned_local_qa_schedule(date(2026, 6, 1), date(2026, 8, 4))
    axis = tuple(point.formation_session for point in schedule)
    days = axis[-4:]
    ids = tuple(f"synthetic-listing-{index:03d}" for index in range(100))
    values = np.tile(np.arange(len(ids), dtype=np.float64), (3, 1))
    seed_eligible = np.ones(values.shape, dtype=np.bool_)
    seed_eligible[0, 0] = False
    rule = FrozenRankCalibrationRule.create(activation_session=date(2021, 9, 7))
    seed = publish_observations(
        owner.store,
        scores=values,
        returns=np.full_like(values, np.nan),
        eligible=seed_eligible,
        strategy_package_hash=TEST_PACKAGE.package_hash,
        component_recipe_hash="b" * 64,
        rule=rule,
        formation_sessions=days[:3],
        holding_end_sessions=(None,) * 3,
        ordered_listing_ids=ids,
        source_hashes=("c" * 64,),
        origin="SYNTHETIC_QA_INPUTS",
    )
    history = []
    for day, omitted, inactive in ((days[1], ids[1], ids[2]), (days[3], ids[4], ids[3])):
        selected = tuple(listing for listing in ids if listing != omitted)
        score = seal_current_contract(
            FrozenComponentScoreSnapshot,
            {
                "request_hash": "d" * 64,
                "strategy_package_hash": TEST_PACKAGE.package_hash,
                "component_recipe_hash": "b" * 64,
                "inference_authority_hash": "c" * 64,
                "observation_snapshot_hash": "e" * 64,
                "formation_session": day,
                "ordered_listing_ids": selected,
                "feature_values_hashes": (),
                "model_identity_hashes": (),
                "projection_hash": "f" * 64,
                "scores": tuple(
                    None if listing == inactive else float(ids.index(listing))
                    for listing in selected
                ),
                "live": tuple(listing != inactive for listing in selected),
                "prediction_calls": 0,
            },
            "snapshot_hash",
        )
        owner.scoring.store.publish_frozen_component_score(score)
        history.append(score)
    plan = CalibrationPlan.create(
        workspace_manifest_hash=owner.manifest.manifest_hash,
        binding=ResearchWorkspaceCalibrationInput(
            strategy_package_id=TEST_PACKAGE.strategy_id,
            strategy_package_hash=TEST_PACKAGE.package_hash,
            seed_hash=seed.content_hash,
            source_kind="WORKSPACE_DATA_FEATURE",
        ),
        score_snapshot_hash=history[-1].snapshot_hash,
        history_score_hashes=tuple(score.snapshot_hash for score in history),
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
                volume=0 if listing == ids[5] and day == days[1] else 100_000,
            )
            for index, day in enumerate(axis)
            for listing in ids
        ),
        actions=(),
    )
    table, bars, source_axis = local_qa_snapshot_rows(captured, sessions=days, through=days[-1])
    expected_scores = np.full((4, len(ids)), np.nan, dtype=np.float64)
    expected_scores[:3] = values
    expected_eligible = np.zeros(expected_scores.shape, dtype=np.bool_)
    expected_eligible[:3] = seed_eligible
    membership = {day: set(ids) for day in days[:3]}
    for score in history:
        row = days.index(score.formation_session)
        expected_scores[row] = np.nan
        expected_eligible[row] = False
        membership[score.formation_session] = set(score.ordered_listing_ids)
        for listing, value, eligible in zip(
            score.ordered_listing_ids, score.scores, score.live, strict=True
        ):
            column = ids.index(listing)
            expected_scores[row, column] = value
            expected_eligible[row, column] = eligible
    realized = np.full(expected_scores.shape, np.nan, dtype=np.float64)
    ends = {}
    for item in table.select(
        ["listing_id", "formation_session", "simple_return", "holding_end_session"]
    ).to_pylist():
        row, column = days.index(item["formation_session"]), ids.index(item["listing_id"])
        realized[row, column] = np.nan if item["simple_return"] is None else item["simple_return"]
        ends[item["formation_session"]] = item["holding_end_session"]
    for row, day in enumerate(days):
        position = source_axis.index(day)
        for column, listing in enumerate(ids):
            expected_eligible[row, column] &= listing in membership.get(
                day, ()
            ) and decision_eligible_at_close(
                listing_id=listing,
                formation_session=day,
                history=source_axis[max(0, position - 19) : position + 1],
                bars=bars,
            )
    expected = publish_consumed_observations(
        owner.store,
        decision_session=days[-1],
        scores=expected_scores,
        returns=realized,
        eligible=expected_eligible,
        origin="LOCAL_QA_OBSERVATIONS",
        strategy_package_hash=seed.strategy_package_hash,
        component_recipe_hash=seed.component_recipe_hash,
        rule=rule,
        formation_sessions=days,
        holding_end_sessions=tuple(ends.get(day) for day in days),
        ordered_listing_ids=ids,
        source_hashes=(
            seed.content_hash,
            plan.source_hash,
            canonical_hash(table["row_hash"].to_pylist()),
        ),
        score_snapshot_hashes=plan.history_score_hashes,
    )
    stage = owner.execute_step(plan, STAGES[1], lambda _stage: seed.content_hash, captured=captured)
    actual = load_observations(owner.store, stage.evidence[0].content_hash)
    reference = load_observations(owner.store, expected.content_hash)
    assert actual[0] == reference[0]
    for observed, scalar in zip(actual[1:], reference[1:], strict=True):
        assert observed.dtype == scalar.dtype
        assert observed.shape == scalar.shape
        assert observed.tobytes(order="C") == scalar.tobytes(order="C")
    assert days[0] in actual[0].formation_sessions
    assert days[1] in actual[0].formation_sessions
    for day, listing in (
        (days[0], ids[0]),
        (days[1], ids[1]),
        (days[1], ids[2]),
        (days[3], ids[4]),
        (days[3], ids[3]),
    ):
        row = actual[0].formation_sessions.index(day)
        assert not actual[3][row, ids.index(listing)]
