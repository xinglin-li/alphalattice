"""The closed-form tranche book's rules.

The tests that matter here are the ones covering semantics that would have been
got wrong by reading the control names: the `mu.ivP` lift and floor, the sleeve
ceiling being a multiple of the *sleeve's* equal weight rather than the 6%
aggregate cap, and the first formation staging every sleeve.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    AGGREGATE_NAME_CAP,
    INSTALLED_TRANCHE_BOOK_RECIPE,
    MU_FLOOR_SPREAD_SHARE,
    SLEEVE_CAP_EQUAL_WEIGHT_MULTIPLE,
    TrancheBookError,
    TrancheBookRecipe,
    aggregate_sleeve_book,
    cap_and_renormalise,
    due_sleeves,
    rank_bucket_mu,
    sleeve_target_weights,
)


def test_authored_data_quarantine_preserves_source_and_requires_a_usable_remainder():
    from datetime import date, timedelta

    from alphalattice.investment.portfolio_strategy_lab.application import (
        research_experiment as owner,
    )

    sessions = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(4))
    listings = tuple(f"L{i:03}" for i in range(80))
    decision = np.ones((4, 80), dtype=bool)
    realized = np.zeros((4, 80))
    realized[1:3, 5] = np.nan
    decision[2, 5] = False  # Still needs valuation if held from a previous formation.
    spec = owner.PortfolioExperimentSpec(alpha_task_id="fixture", candidate_id="fixture")
    assert "unavailable_return_policy" not in spec.model_dump(mode="json")
    args = dict(sessions=sessions, listings=listings, decision=decision, realized=realized)
    with pytest.raises(ValueError, match="benchmark_support_absent"):
        owner.resolve_research_universe(spec=spec, **args)
    fallback = spec.model_copy(update={"unavailable_return_policy": "quarantine_listings"})
    effective, excluded = owner.resolve_research_universe(spec=fallback, **args)
    assert excluded[0].listing_id == "L005" and excluded[0].affected_sessions == sessions[1:3]
    assert len(excluded) == 1 and not effective[:, 5].any()
    assert np.array_equal(effective[:, 6:], decision[:, 6:])
    assert decision[0, 5] and np.isnan(realized[1, 5])  # No source mutation/filling.
    np.testing.assert_array_equal(realized[effective], 0)
    realized[:, 10:] = np.nan
    with pytest.raises(ValueError, match="universe_too_small_after_quarantine"):
        owner.resolve_research_universe(spec=fallback, **args)


def test_a_support_refusal_names_its_cause_and_where_it_holds():
    """Regression: a first use's book stopped on `benchmark_support_absent`, and
    neither the Task, its recovery view nor the words said which condition held or where, so the
    agent searched the source. The code's subject names the cause, the names and sessions it
    holds on and their span; a session with no eligible name comes first, since no
    unavailable-return policy repairs it, a session the quarantine emptied among them. The code
    fits a Task's record and reads as a typed code at any size."""

    from datetime import date, timedelta

    from alphalattice.control.task_control.contracts import FAILURE_CODE_MAX_LENGTH
    from alphalattice.interface.local_application.failure_codes import safe_failure_code
    from alphalattice.investment.portfolio_strategy_lab.application import (
        research_experiment as owner,
    )

    code = "portfolio_research.benchmark_support_absent"
    sessions = tuple(date(2020, 3, 2) + timedelta(days=i) for i in range(5))
    listings = tuple(f"L{i:03}" for i in range(40))
    decision = np.ones((5, 40), dtype=bool)
    realized = np.zeros((5, 40))
    realized[1:4, 3] = np.nan
    realized[2, 7] = np.nan
    strict = owner.PortfolioExperimentSpec(alpha_task_id="fixture", candidate_id="fixture")

    def refused(spec: owner.PortfolioExperimentSpec, eligible: np.ndarray) -> str:
        with pytest.raises(ValueError) as raised:
            owner.resolve_research_universe(
                spec=spec,
                sessions=sessions,
                listings=listings,
                decision=eligible,
                realized=realized,
            )
        return str(raised.value)

    unavailable = refused(strict, decision)
    assert unavailable == f"{code}:returns_unavailable:2 names,3 sessions,2020-03-03..2020-03-05"
    empty = decision.copy()
    empty[4] = False
    assert refused(strict, empty) == f"{code}:no_eligible_name:1 sessions,2020-03-06..2020-03-06"
    # The quarantine excludes the names without returns; a session only they filled is empty.
    quarantine = strict.model_copy(update={"unavailable_return_policy": "quarantine_listings"})
    alone = decision.copy()
    alone[2] = False
    alone[2, 3] = True
    assert refused(quarantine, alone) == (
        f"{code}:no_eligible_name:1 sessions,2020-03-04..2020-03-04"
    )
    # Five-digit counts of names and sessions still fit (a view, nothing allocated).
    every = tuple(date(1990, 1, 1) + timedelta(days=i) for i in range(12_000))
    widest = owner._support_absence(
        every, np.zeros(len(every), dtype=bool), np.broadcast_to(np.True_, (12_000, 12_000))
    )
    assert widest.startswith(f"{code}:returns_unavailable:12000 names,12000 sessions,")
    assert len(widest) <= FAILURE_CODE_MAX_LENGTH and safe_failure_code(widest) == widest


def test_authored_replay_segments_preserve_holds_state_and_refuse_changed_scores(tmp_path):
    from dataclasses import replace
    from datetime import date, timedelta

    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        MARKED_TO_MARKET_AT_CLOSE_T,
    )
    from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
    from alphalattice.capabilities.portfolio_backtesting.segments import (
        run_portfolio_walk_forward_segment,
    )
    from alphalattice.investment.alpha_research.experiments.development_contracts import (
        canonical_score_value_identity,
    )
    from alphalattice.investment.portfolio_strategy_lab.application import (
        research_experiment as owner,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
        TrancheBookDecisionProvider,
        TrancheFormationInputs,
    )
    from alphalattice.protocols.research_authoring.contracts import (
        ResearchExperimentEnvelope,
        ResolvedResearchAuthority,
        SealedResearchProgram,
    )

    sessions = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(45))
    listings = tuple(f"L{i:03}" for i in range(80))
    gaps = {20, 21, 33}
    scored = tuple(s for i, s in enumerate(sessions) if i not in gaps)
    values = np.sin(np.arange(45)[:, None] + np.arange(80)[None, :] / 13)
    values[list(gaps)] = np.nan
    eligible = np.isfinite(values)
    returns = np.cos(np.arange(45)[:, None] / 3 + np.arange(80)[None, :] / 9) / 100
    source = owner.PortfolioExperimentSource.create(
        alpha_task_id="fixture",
        alpha_program_hash="a" * 64,
        alpha_receipt_hash="b" * 64,
        candidate_id="fixture",
        target_recipe_id="fixture-rank",
        target_binding_hash="c" * 64,
        input_binding_hash="d" * 64,
        panel_snapshot_hash="e" * 64,
        outcome_snapshot_hash="f" * 64,
        universe_revision="a" * 64,
        ordered_listing_ids=listings,
        formation_sessions=sessions,
        score_sessions=scored,
        score_refs=(("a" * 64, "b" * 64),),
        score_value_hash=canonical_score_value_identity(
            values[[i for i in range(45) if i not in gaps]]
        ),
    )
    lane = ReferenceMarkLane(
        method=MARKED_TO_MARKET_AT_CLOSE_T,
        marks_by_session={s: np.zeros(80) for s in sessions},
        entry_session_by_formation={s: s + timedelta(days=1) for s in sessions},
        state_transition_binding_hash="a" * 64,
    )
    inputs = owner.PortfolioExperimentInputs(
        sessions,
        listings,
        values,
        eligible,
        np.ones((45, 80), dtype=bool),
        returns,
        np.full((45, 80), 10000.0),
        np.ones((1, 80)),
        np.ones(1),
        {s: returns[i] for i, s in enumerate(sessions)},
        lane,
        {"source": source.input_binding_hash},
    )
    spec = owner.PortfolioExperimentSpec(alpha_task_id="fixture", candidate_id="fixture")
    assert (
        owner.PortfolioExperimentSpec(
            alpha_task_id="fixture", candidate_id="fixture", cost_bps_per_side="5.00"
        )
        == spec
    )
    with pytest.raises(ValueError, match="cost_invalid"):
        owner.PortfolioExperimentSpec(
            alpha_task_id="fixture", candidate_id="fixture", cost_bps_per_side="not-a-cost"
        )
    envelope = ResearchExperimentEnvelope.create(
        kind=owner.KIND,
        schema_id="research-experiment-envelope",
        data_snapshot_handle=source.panel_snapshot_hash,
        universe_handle="fixture",
        sessions={
            "start": sessions[0],
            "end": sessions[-1],
            "as_of": {"session": sessions[-1], "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 80, "maximum_numerical_calls": 100},
        determinism={"seed": 0, "thread_limit": 1, "network_disabled": True},
        output_workspace="out",
        baseline_workspace="source",
    )
    authority = ResolvedResearchAuthority.create(
        data_snapshot_handle=source.panel_snapshot_hash,
        universe_handle="fixture",
        panel_snapshot_hash=source.panel_snapshot_hash,
        panel_manifest_ref="fixture",
        universe_revision_sha256=source.universe_revision,
        ordered_listing_ids=listings,
        sessions=sessions,
        source_watermark_hash="a" * 64,
    )
    document = {
        "experiment": envelope.model_dump(mode="json"),
        "portfolio": spec.model_dump(mode="json"),
    }
    executor = owner.PortfolioExperimentExecutor(source, lambda output: inputs)
    compiled = executor.compiler.compile_desk_program(
        envelope=envelope, document=document, authority=authority
    )
    program = SealedResearchProgram.create(
        kind=owner.KIND,
        envelope_hash=envelope.envelope_hash,
        resolved_sessions=sessions,
        authority_hash=authority.authority_hash,
        **compiled.model_dump(mode="json"),
    )
    full = executor.execute(
        program=program, document=document, authority=authority, output_workspace=tmp_path / "full"
    )
    receipt, rows = owner.read_portfolio_experiment(tmp_path / "full", full.artifact_uris[0])
    # The verifier keeps the read it proved for the request that asked for it,
    # and keeps nothing from a verification it refused.
    from alphalattice.protocols.research_authoring.contracts import ResearchExecutionEvidence

    evidence = ResearchExecutionEvidence.create(
        kind=owner.KIND,
        program_hash=program.program_hash,
        desk_program_hash=program.desk_program_hash,
        method_binding_hash=program.method_binding_hash,
        authority_hash=authority.authority_hash,
        disposition=full.disposition,
        numerical_call_count=full.numerical_call_count,
        artifact_uris=full.artifact_uris,
        formation_sessions=full.formation_sessions,
        desk_input_binding_hash=full.desk_input_binding_hash,
    )
    verifier = owner.PortfolioExperimentVerifier()
    verifier.verify(
        program=program, evidence=evidence, authority=authority, output_workspace=tmp_path / "full"
    )
    assert verifier.verified_readback == (receipt, rows)
    with pytest.raises(ValueError, match="receipt_program_mismatch"):
        verifier.verify(
            program=program,
            evidence=evidence.model_copy(update={"desk_input_binding_hash": "0" * 64}),
            authority=authority,
            output_workspace=tmp_path / "full",
        )
    assert verifier.verified_readback is None
    provider = TrancheBookDecisionProvider(
        recipe=spec.recipe,
        formations=tuple(
            TrancheFormationInputs(s, values[i], eligible[i], None, None)
            for i, s in enumerate(sessions)
        ),
        ordered_listing_ids=listings,
        sector_exposure_matrix=inputs.sector_exposure_matrix,
        equal_weight_sector_exposure=inputs.equal_weight_sector_exposure,
    )
    direct = run_portfolio_walk_forward_segment(
        workspace=inputs,
        decision_provider=provider,
        start_index=0,
        stop_index=45,
        rebalance_clock=source.clock,
        reference_mark=lane,
    )
    assert [r["weights"] for r in rows] == [list(r) for r in direct.executed_weights]
    assert [r["gross_simple_return"] for r in rows] == list(direct.gross_simple_returns)
    assert all(
        rows[i]["decision_mode"] == "HOLD" and rows[i]["one_way_turnover"] == 0 for i in gaps
    )
    assert "data_exclusions" not in receipt.model_dump(mode="json")
    # A different declared policy has a different Program, retains the source
    # axis, and excludes the unavailable name from both holdings and benchmark.
    damaged = returns.copy()
    damaged[2, 3] = np.nan
    fallback = spec.model_copy(update={"unavailable_return_policy": "quarantine_listings"})
    admitted, exclusions = owner.resolve_research_universe(
        spec=fallback,
        sessions=sessions,
        listings=listings,
        decision=np.ones_like(eligible),
        realized=damaged,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    fallback_inputs = replace(
        inputs,
        decision_eligible=eligible & admitted,
        realized_simple_returns=damaged,
        passive_returns_by_session={s: damaged[i][admitted[i]] for i, s in enumerate(sessions)},
        data_exclusions=exclusions,
        market_binding={
            **inputs.market_binding,
            "data_quarantine": canonical_hash([v.model_dump(mode="json") for v in exclusions]),
        },
    )
    fallback_document = {**document, "portfolio": fallback.model_dump(mode="json")}
    fallback_program = SealedResearchProgram.create(
        kind=owner.KIND,
        envelope_hash=envelope.envelope_hash,
        resolved_sessions=sessions,
        authority_hash=authority.authority_hash,
        **executor.compiler.compile_desk_program(
            envelope=envelope, document=fallback_document, authority=authority
        ).model_dump(mode="json"),
    )
    repaired = owner.PortfolioExperimentExecutor(source, lambda output: fallback_inputs).execute(
        program=fallback_program,
        document=fallback_document,
        authority=authority,
        output_workspace=tmp_path / "quarantined",
    )
    repaired_receipt, repaired_rows = owner.read_portfolio_experiment(
        tmp_path / "quarantined", repaired.artifact_uris[0]
    )
    assert all(row["weights"][3] == row["targets"][3] == 0 for row in repaired_rows)
    assert repaired_receipt.source == receipt.source
    assert repaired_receipt.data_quality["effective_listing_count"] == 79
    assert repaired_receipt.data_exclusions == exclusions
    with pytest.raises(ValueError, match="receipt_identity_invalid"):
        owner.PortfolioReplayReceipt.model_validate(
            {**repaired_receipt.model_dump(mode="json"), "data_exclusions": []}
        )
    resumed_root = tmp_path / "resumed"
    cancelled = owner.PortfolioExperimentExecutor(
        source,
        lambda output: inputs,
        cancellation=lambda: bool(
            tuple((resumed_root / "portfolio-strategy-lab" / owner.SEGMENTS).glob("*.json"))
        ),
    )
    with pytest.raises(owner.PortfolioExperimentCancelled):
        cancelled.execute(
            program=program, document=document, authority=authority, output_workspace=resumed_root
        )
    resumed = executor.execute(
        program=program, document=document, authority=authority, output_workspace=resumed_root
    )
    assert resumed.numerical_call_count == full.numerical_call_count - 21
    assert owner.read_portfolio_experiment(resumed_root, resumed.artifact_uris[0]) == (
        receipt,
        rows,
    )
    changed = owner.PortfolioExperimentExecutor(
        source, lambda output: replace(inputs, scores=np.zeros_like(values))
    )
    with pytest.raises(ValueError, match="input_values_changed"):
        changed.execute(
            program=program,
            document=document,
            authority=authority,
            output_workspace=tmp_path / "refused",
        )
    assert not (tmp_path / "refused").exists()


def test_a_rebalance_never_selects_from_a_pool_smaller_than_the_book(tmp_path):
    """requirement (the guide's own path): a book never stops mid-walk for a short pool.
    A formation its candidate under-scored -- fewer scored names than a rebalance selects, as a
    feature unavailable across the universe leaves it -- is held from the plan on, as an embargo
    session is, and the book completes with no turnover there; a scored formation tradability
    leaves short is refused before any segment is written; an under-scored first or last
    formation is refused at plan time; each refusal names its session, worded with a way on."""

    from dataclasses import replace
    from datetime import date, timedelta

    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        MARKED_TO_MARKET_AT_CLOSE_T,
    )
    from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
    from alphalattice.control.product_host.research_authoring.portfolio_handoff import (
        hold_under_scored_formations,
    )
    from alphalattice.interface.local_application.cli_contract import refusal_words
    from alphalattice.investment.alpha_research.experiments.development_contracts import (
        canonical_score_value_identity,
    )
    from alphalattice.investment.alpha_research.experiments.score_rows import (
        DevelopmentScoreMatrix,
    )
    from alphalattice.investment.portfolio_strategy_lab.application import (
        research_experiment as owner,
    )
    from alphalattice.protocols.research_authoring.contracts import (
        ResearchExperimentEnvelope,
        ResolvedResearchAuthority,
        SealedResearchProgram,
    )

    sessions = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(45))
    listings = tuple(f"L{i:03}" for i in range(80))
    spec = owner.PortfolioExperimentSpec(alpha_task_id="fixture", candidate_id="fixture")
    values = np.sin(np.arange(45)[:, None] + np.arange(80)[None, :] / 13)
    values[25, 10:] = np.nan  # the candidate scored ten names, fewer than the book selects
    published = DevelopmentScoreMatrix(
        sessions,
        listings,
        values,
        np.isfinite(values),
        (("a" * 64, "b" * 64),),
        canonical_score_value_identity(values),
    )
    scores, held = hold_under_scored_formations(published, spec.top_k)
    assert held == (sessions[25],)
    assert scores.sessions == sessions[:25] + sessions[26:]
    assert hold_under_scored_formations(scores, spec.top_k) == (scores, ())
    for end in (0, 44):
        unscored = values.copy()
        unscored[end, 10:] = np.nan
        with pytest.raises(ValueError) as refused:
            hold_under_scored_formations(
                replace(published, scores=unscored, available=np.isfinite(unscored)), spec.top_k
            )
        code = str(refused.value)
        assert code == f"portfolio_research.book_end_under_scored:{sessions[end]}"
        assert str(sessions[end]) in refusal_words(code)["detail"]
        assert refusal_words(code)["next_action"]

    source = owner.PortfolioExperimentSource.create(
        alpha_task_id="fixture",
        alpha_program_hash="a" * 64,
        alpha_receipt_hash="b" * 64,
        candidate_id="fixture",
        target_recipe_id="fixture-rank",
        target_binding_hash="c" * 64,
        input_binding_hash="d" * 64,
        panel_snapshot_hash="e" * 64,
        outcome_snapshot_hash="f" * 64,
        universe_revision="a" * 64,
        ordered_listing_ids=listings,
        formation_sessions=sessions,
        score_sessions=scores.sessions,
        score_refs=scores.refs,
        score_value_hash=scores.value_hash,
    )
    # The market inputs fill the scored formations alone, so a held one reads no score.
    filled = values.copy()
    filled[25] = np.nan
    returns = np.cos(np.arange(45)[:, None] / 3 + np.arange(80)[None, :] / 9) / 100
    lane = ReferenceMarkLane(
        method=MARKED_TO_MARKET_AT_CLOSE_T,
        marks_by_session={s: np.zeros(80) for s in sessions},
        entry_session_by_formation={s: s + timedelta(days=1) for s in sessions},
        state_transition_binding_hash="a" * 64,
    )
    inputs = owner.PortfolioExperimentInputs(
        sessions,
        listings,
        filled,
        np.isfinite(filled),
        np.ones((45, 80), dtype=bool),
        returns,
        np.full((45, 80), 10000.0),
        np.ones((1, 80)),
        np.ones(1),
        {s: returns[i] for i, s in enumerate(sessions)},
        lane,
        {"source": source.input_binding_hash},
    )
    envelope = ResearchExperimentEnvelope.create(
        kind=owner.KIND,
        schema_id="research-experiment-envelope",
        data_snapshot_handle=source.panel_snapshot_hash,
        universe_handle="fixture",
        sessions={
            "start": sessions[0],
            "end": sessions[-1],
            "as_of": {"session": sessions[-1], "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 80, "maximum_numerical_calls": 100},
        determinism={"seed": 0, "thread_limit": 1, "network_disabled": True},
        output_workspace="out",
        baseline_workspace="source",
    )
    authority = ResolvedResearchAuthority.create(
        data_snapshot_handle=source.panel_snapshot_hash,
        universe_handle="fixture",
        panel_snapshot_hash=source.panel_snapshot_hash,
        panel_manifest_ref="fixture",
        universe_revision_sha256=source.universe_revision,
        ordered_listing_ids=listings,
        sessions=sessions,
        source_watermark_hash="a" * 64,
    )
    document = {
        "experiment": envelope.model_dump(mode="json"),
        "portfolio": spec.model_dump(mode="json"),
    }
    executor = owner.PortfolioExperimentExecutor(source, lambda output: inputs)
    program = SealedResearchProgram.create(
        kind=owner.KIND,
        envelope_hash=envelope.envelope_hash,
        resolved_sessions=sessions,
        authority_hash=authority.authority_hash,
        **executor.compiler.compile_desk_program(
            envelope=envelope, document=document, authority=authority
        ).model_dump(mode="json"),
    )
    book = executor.execute(
        program=program, document=document, authority=authority, output_workspace=tmp_path / "book"
    )
    _receipt, rows = owner.read_portfolio_experiment(tmp_path / "book", book.artifact_uris[0])
    assert rows[25]["decision_mode"] == "HOLD" and rows[25]["one_way_turnover"] == 0
    assert rows[26]["decision_mode"] == "REBALANCE"

    # Tradability leaves a scored formation short: refused before the walk writes a segment.
    tradable = np.isfinite(filled)
    tradable[30, 20:] = False
    short = owner.PortfolioExperimentExecutor(
        source, lambda output: replace(inputs, decision_eligible=tradable)
    )
    with pytest.raises(ValueError) as refused:
        short.execute(
            program=program,
            document=document,
            authority=authority,
            output_workspace=tmp_path / "short",
        )
    code = str(refused.value)
    assert code == f"portfolio_research.eligible_pool_short:{sessions[30]}"
    assert str(sessions[30]) in refusal_words(code)["detail"]
    assert refusal_words(code)["next_action"]
    assert not (tmp_path / "short").exists()


def test_the_frozen_default_is_three_sleeves_of_thirty_five_at_exit_seventy() -> None:
    recipe = INSTALLED_TRANCHE_BOOK_RECIPE
    assert (recipe.top_k, recipe.tranches, recipe.exit_rank) == (35, 3, 70)
    assert recipe.weight_rule == "mu.iv1"
    assert recipe.exit_rank_multiple == 2.0
    assert recipe.consumes_mu and recipe.consumes_risk


def test_the_sleeve_ceiling_is_a_multiple_of_the_sleeve_equal_weight() -> None:
    """2.1 / top_k, and it holds that meaning at every admitted width.

    This is the semantic most easily confused with the 6% aggregate cap. At the
    default width the two are numerically identical, which is exactly why a test
    that only checked 35 would not notice the confusion.
    """

    assert INSTALLED_TRANCHE_BOOK_RECIPE.sleeve_ceiling == pytest.approx(2.1 / 35)
    for top_k in (17, 35, 75):
        recipe = TrancheBookRecipe.create(top_k=top_k)
        assert recipe.sleeve_ceiling == pytest.approx(SLEEVE_CAP_EQUAL_WEIGHT_MULTIPLE / top_k)
        assert recipe.sleeve_ceiling == pytest.approx(2.1 * (1.0 / top_k))


def test_every_sleeve_is_staged_on_the_first_formation_then_one_at_a_time() -> None:
    """A one-sleeve-per-session ramp from empty would measure the ramp."""

    assert due_sleeves(formation_index=0, tranches=3) == (0, 1, 2)
    assert [due_sleeves(formation_index=i, tranches=3)[0] for i in range(1, 10)] == [
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        2,
        0,
    ]


def test_each_sleeve_is_reviewed_once_per_cycle() -> None:
    for tranches in (3, 7, 10):
        reviewed = [
            due_sleeves(formation_index=i, tranches=tranches)[0] for i in range(1, 1 + tranches)
        ]
        assert sorted(reviewed) == list(range(tranches))


def test_mu_rules_lift_negative_bucket_means_instead_of_dropping_them() -> None:
    """Bucket means are simple returns; a long-only book cannot hold a negative.

    The weakest bucket must keep a small positive weight, because membership
    already voted it into the sleeve. Zeroing it here would be an exclusion no
    rule asked for.
    """

    mu = np.array([-0.02, 0.0, 0.05], dtype=np.float64)
    variance = np.full(3, 0.04, dtype=np.float64)
    result = sleeve_target_weights(mu=mu, variance=variance, rule="mu.iv1", ceiling=1.0)

    assert (result.weights > 0.0).all()
    assert result.weights.sum() == pytest.approx(1.0)
    # Ordering follows mu at equal variance, and the floor is a tenth of the
    # lifted spread rather than zero.
    assert result.weights[0] < result.weights[1] < result.weights[2]
    lifted = mu - mu.min()
    expected = lifted + MU_FLOOR_SPREAD_SHARE * lifted.max()
    assert result.weights == pytest.approx(expected / expected.sum())


def test_a_flat_mu_lane_falls_back_to_the_inverse_volatility_shape() -> None:
    """Zero spread means the tilt has nothing to say, not that weights collapse."""

    mu = np.zeros(3, dtype=np.float64)
    variance = np.array([0.01, 0.04, 0.09], dtype=np.float64)
    tilted = sleeve_target_weights(mu=mu, variance=variance, rule="mu.iv1", ceiling=1.0)
    plain = sleeve_target_weights(mu=None, variance=variance, rule="iv1", ceiling=1.0)
    assert tilted.weights == pytest.approx(plain.weights)


def test_inverse_volatility_exponents_differ_and_equal_weight_ignores_risk() -> None:
    variance = np.array([0.01, 0.04, 0.09], dtype=np.float64)
    iv1 = sleeve_target_weights(mu=None, variance=variance, rule="iv1", ceiling=1.0).weights
    iv2 = sleeve_target_weights(mu=None, variance=variance, rule="iv2", ceiling=1.0).weights
    ew = sleeve_target_weights(mu=None, variance=variance, rule="ew", ceiling=1.0).weights

    assert iv1 == pytest.approx((1 / np.sqrt(variance)) / (1 / np.sqrt(variance)).sum())
    assert iv2 == pytest.approx((1 / variance) / (1 / variance).sum())
    assert ew == pytest.approx(np.full(3, 1 / 3))
    # A higher exponent concentrates harder into the calmest name.
    assert iv2[0] > iv1[0] > ew[0]


def test_a_mu_rule_refuses_a_missing_mu_lane_rather_than_trading_iv1() -> None:
    """Falling back would report mu.iv1 while trading iv1."""

    with pytest.raises(TrancheBookError, match="tranche_mu_lane_absent"):
        sleeve_target_weights(mu=None, variance=np.full(3, 0.04), rule="mu.iv1", ceiling=1.0)


def test_a_non_mu_rule_refuses_a_supplied_mu_lane() -> None:
    with pytest.raises(TrancheBookError, match="tranche_mu_lane_not_consumed"):
        sleeve_target_weights(mu=np.zeros(3), variance=np.full(3, 0.04), rule="iv1", ceiling=1.0)


def test_the_ceiling_is_respected_and_the_excess_is_redistributed() -> None:
    raw = np.array([10.0, 1.0, 1.0, 1.0], dtype=np.float64)
    result = cap_and_renormalise(raw, ceiling=0.4)
    assert result.disposition == "CAPPED"
    assert result.weights.max() <= 0.4 + 1e-12
    assert result.weights.sum() == pytest.approx(1.0)
    # Redistribution is proportional, so the three equal names stay equal.
    assert result.weights[1] == pytest.approx(result.weights[2])
    assert result.weights[2] == pytest.approx(result.weights[3])


def test_an_infeasible_ceiling_returns_a_visible_equal_weight_fallback() -> None:
    """A cap smaller than 1/n cannot be met; the fallback must not be silent."""

    result = cap_and_renormalise(np.array([1.0, 2.0, 3.0]), ceiling=0.2)
    assert result.disposition == "EQUAL_WEIGHT_CEILING_INFEASIBLE"
    assert result.weights == pytest.approx(np.full(3, 1 / 3))


def test_no_positive_mass_returns_a_visible_equal_weight_fallback() -> None:
    result = cap_and_renormalise(np.zeros(4), ceiling=0.5)
    assert result.disposition == "EQUAL_WEIGHT_NO_POSITIVE_MASS"
    assert result.weights == pytest.approx(np.full(4, 0.25))


def test_the_aggregate_cap_binds_on_a_name_no_single_sleeve_overweights() -> None:
    """This is why the 6% cap is applied to the assembled book, not per sleeve.

    Each sleeve here holds the shared name at a modest weight, well inside any
    sleeve ceiling. Only the assembled book concentrates it.
    """

    # Ten names, three five-name sleeves, and name 0 is in all three. Inside any
    # sleeve it holds 0.2, which is that sleeve's equal weight and far under a
    # 2.1x sleeve ceiling of 0.42. Assembled, it holds 0.2 of the book.
    def sleeve(members: tuple[int, ...]) -> np.ndarray:
        weights = np.zeros(10, dtype=np.float64)
        weights[list(members)] = 1.0 / len(members)
        return weights

    books = (sleeve((0, 1, 2, 3, 4)), sleeve((0, 5, 6, 7, 8)), sleeve((0, 9, 1, 5, 2)))
    for one in books:
        assert one.max() <= 2.1 / 5

    book = aggregate_sleeve_book(
        sleeve_books=books,
        sleeve_shares=(1 / 3, 1 / 3, 1 / 3),
        aggregate_name_cap=0.15,
    )
    assert book.disposition == "CAPPED"
    assert book.weights[0] <= 0.15 + 1e-12
    assert book.weights.sum() == pytest.approx(1.0)


def test_sleeves_overlap_so_the_book_holds_fewer_names_than_top_k_times_tranches() -> None:
    """The readout the plan forbids presenting as `top_k * tranches`."""

    overlapping = np.array([0.5, 0.5, 0.0], dtype=np.float64)
    book = aggregate_sleeve_book(
        sleeve_books=(overlapping, overlapping, np.array([0.0, 0.5, 0.5])),
        sleeve_shares=(1 / 3, 1 / 3, 1 / 3),
    )
    assert int((book.weights > 0).sum()) == 3
    assert book.weights.sum() == pytest.approx(1.0)


def test_rank_bucket_mu_maps_positions_to_buckets_and_neutralises_absent_ones() -> None:
    means = np.array([0.05, np.nan, -0.01], dtype=np.float64)
    selected = np.array([0, 1, 2], dtype=np.int64)
    positions = np.array([0, 4, 8], dtype=np.int64)
    mu = rank_bucket_mu(
        selected=selected,
        ranked_positions=positions,
        eligible_count=9,
        bucket_means=means,
    )
    assert mu == pytest.approx(np.array([0.05, 0.0, -0.01]))


def test_the_recipe_refuses_an_exit_rank_outside_its_multiple_band() -> None:
    with pytest.raises((TrancheBookError, ValidationError)):
        TrancheBookRecipe.create(top_k=35, exit_rank=34)
    with pytest.raises((TrancheBookError, ValidationError)):
        TrancheBookRecipe.create(top_k=35, exit_rank=6 * 35 + 1)
    assert TrancheBookRecipe.create(top_k=35, exit_rank=210).exit_rank_multiple == 6.0


def test_the_recipe_refuses_controls_outside_the_admitted_ranges() -> None:
    for bad in ({"top_k": 14}, {"top_k": 76}, {"tranches": 2}, {"tranches": 11}):
        with pytest.raises(ValidationError):
            TrancheBookRecipe.create(**bad)  # type: ignore[arg-type]


def test_the_caps_are_frozen_and_not_a_control() -> None:
    payload = INSTALLED_TRANCHE_BOOK_RECIPE.model_dump(mode="json")
    payload["aggregate_name_cap"] = 0.10
    with pytest.raises(ValidationError, match="tranche_caps_not_frozen"):
        TrancheBookRecipe.model_validate(payload)


def test_recipe_identity_is_tamper_evident_and_stable() -> None:
    payload = INSTALLED_TRANCHE_BOOK_RECIPE.model_dump(mode="json")
    payload["top_k"] = 50
    with pytest.raises(ValidationError, match="tranche_recipe_identity_invalid"):
        TrancheBookRecipe.model_validate(payload)
    assert TrancheBookRecipe.create().recipe_hash == INSTALLED_TRANCHE_BOOK_RECIPE.recipe_hash


def test_a_changed_control_rotates_the_recipe_identity() -> None:
    assert (
        TrancheBookRecipe.create(top_k=50).recipe_hash != INSTALLED_TRANCHE_BOOK_RECIPE.recipe_hash
    )
    assert AGGREGATE_NAME_CAP == 0.06


def test_a_sparse_book_judges_feasibility_on_held_names_not_axis_length() -> None:
    """Regression: the zeros can never absorb redistributed weight.

    Sixteen held names out of a twenty-four name axis. Judging the ceiling
    against the axis reads `0.06 * 24 = 1.44` and calls it feasible; the names
    that can actually take weight give `0.06 * 16 = 0.96`, which is not. The
    earlier version returned a vector whose maximum was above the ceiling while
    reporting that it had been capped.
    """

    weights = np.zeros(24, dtype=np.float64)
    weights[:16] = 1 / 16
    result = cap_and_renormalise(weights, ceiling=0.06)
    assert result.disposition == "EQUAL_WEIGHT_CEILING_INFEASIBLE"
    assert result.weights.sum() == pytest.approx(1.0)


def test_redistribution_never_opens_a_position_the_selection_did_not_make() -> None:
    """Regression, and the more serious of the two.

    A long-only book must hold what the membership rule selected and nothing
    else. The earlier version spread excess weight into zero-weight names, so a
    sixteen-name book came back holding twenty-four.
    """

    weights = np.zeros(24, dtype=np.float64)
    weights[:16] = 1 / 16
    assert int((cap_and_renormalise(weights, ceiling=0.06).weights > 0).sum()) == 16

    tilted = np.zeros(24, dtype=np.float64)
    tilted[:20] = np.linspace(1.0, 5.0, 20)
    capped = cap_and_renormalise(tilted, ceiling=0.08)
    assert capped.disposition == "CAPPED"
    assert int((capped.weights > 0).sum()) == 20
    assert capped.weights.max() <= 0.08 + 1e-12


def test_a_capped_disposition_is_a_checked_claim_not_a_label() -> None:
    """The post-condition exists so a non-converging loop cannot report success."""

    for size, held, ceiling in ((24, 16, 0.06), (40, 40, 0.03), (12, 5, 0.15)):
        raw = np.zeros(size, dtype=np.float64)
        raw[:held] = np.linspace(1.0, 9.0, held)
        result = cap_and_renormalise(raw, ceiling=ceiling)
        if result.disposition == "CAPPED":
            assert result.weights.max() <= ceiling + 1e-12
        assert result.weights.sum() == pytest.approx(1.0)
        assert int((result.weights > 0).sum()) <= held


def test_capped_names_are_pinned_and_do_not_reabsorb_weight() -> None:
    """Review finding: the loop recomputed the over-set each pass.

    A name sitting at exactly the ceiling is not ``> ceiling``, so it fell back
    into the redistribution pool, took more weight and went over again. On this
    vector that oscillates until the loop gives up, even though the answer below
    is valid and obvious.
    """

    result = cap_and_renormalise(np.array([0.5, 0.3, 0.2]), ceiling=0.34)
    assert result.disposition == "CAPPED"
    assert result.weights == pytest.approx(np.array([0.34, 0.34, 0.32]))
    assert result.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("held", [3, 5, 17, 20, 40])
def test_water_filling_succeeds_whenever_the_ceiling_is_arithmetically_reachable(
    held: int,
) -> None:
    """``ceiling * held >= 1`` is the whole feasibility condition.

    Nothing about the tilt's shape may decide it. The earlier implementation
    made steep tilts fail, which is what produced the false claim that the
    boundary was data dependent.
    """

    ceiling = 1.0 / held * 1.2
    for steepness in (1.0, 5.0, 50.0, 500.0):
        raw = np.zeros(held + 10, dtype=np.float64)
        raw[:held] = np.linspace(steepness, 1.0, held)
        result = cap_and_renormalise(raw, ceiling=ceiling)
        assert result.disposition == "CAPPED", (held, steepness)
        assert result.weights.max() <= ceiling + 1e-12
        assert result.weights.sum() == pytest.approx(1.0)
        assert int((result.weights > 0).sum()) == held


def test_only_a_ceiling_below_one_over_held_is_infeasible() -> None:
    raw = np.zeros(30, dtype=np.float64)
    raw[:20] = np.linspace(5.0, 1.0, 20)
    assert cap_and_renormalise(raw, ceiling=1.0 / 20 - 1e-6).disposition == (
        "EQUAL_WEIGHT_CEILING_INFEASIBLE"
    )
    assert cap_and_renormalise(raw, ceiling=1.0 / 20 + 1e-6).disposition == "CAPPED"


def test_the_aggregate_cap_sets_an_exact_arithmetic_floor_on_top_k() -> None:
    """The real control-surface finding, once the algorithm is correct.

    At the staging formation every sleeve selects the same names, so the book
    holds exactly ``top_k``. A 6% aggregate cap therefore needs
    ``top_k >= ceil(1 / 0.06) = 17``, which is a constant and not a property of
    the data. The declared control minimum of 15 is below it.
    """

    for top_k in (15, 16):
        assert AGGREGATE_NAME_CAP * top_k < 1.0
    for top_k in (17, 35, 75):
        assert AGGREGATE_NAME_CAP * top_k >= 1.0
