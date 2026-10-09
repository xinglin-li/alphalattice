"""Acceptance for the Risk/Portfolio study's platform seams.

Each test here answers one question of the form "can the next researcher do this
without editing a central owner". A seam that only its first user can use is not
a seam, so every one of these installs or configures a *second* thing through the
route the first thing used.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from alphalattice.capabilities.portfolio_backtesting.active_metrics import (
    ActiveMetricsError,
    active_path_metrics,
)
from alphalattice.capabilities.portfolio_backtesting.clocks import (
    EveryFormationClock,
    EveryNFormationsClock,
    build_installed_rebalance_clock_catalog,
    resolve_rebalance_clock,
)
from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioCostPolicy,
    PortfolioWalkForwardError,
)
from alphalattice.protocols.research_authoring.contracts import AuthoringError
from alphalattice.protocols.research_authoring.selection import (
    require_selection_only_document,
)

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- selection only
def test_committed_request_documents_carry_no_resolved_authority() -> None:
    """Every committed request document passes the rule its own loader applies."""

    documents = sorted((ROOT / "config").glob("*.yaml"))
    assert documents
    for document in documents:
        require_selection_only_document(yaml.safe_load(document.read_text(encoding="utf-8")))


def test_a_nested_identity_is_refused_where_a_key_scan_would_miss_it() -> None:
    """The defect this rule exists for: an identity one level down, under a list.

    A top-level key scan cannot see this shape, which is exactly how six of them
    survived in a file whose header said there were none.
    """

    payload = {"request": {"upstream": [{"stage": "ALPHA", "handle": "current"}]}}
    require_selection_only_document(payload)

    payload["request"]["upstream"][0]["dossier_or_root_hash"] = "b" * 64  # type: ignore[index]
    with pytest.raises(AuthoringError) as declared:
        require_selection_only_document(payload)
    assert "authority_key" in str(declared.value)

    # And again with an innocent key name, so the value rule is what holds.
    payload["request"]["upstream"][0] = {  # type: ignore[index]
        "stage": "ALPHA",
        "handle": "current",
        "revision": "c" * 64,
    }
    with pytest.raises(AuthoringError) as valued:
        require_selection_only_document(payload)
    assert "resolved_identity" in str(valued.value)


# ------------------------------------------------------------------- risk arms
def test_a_second_risk_method_installs_without_a_central_edit() -> None:
    """R2 is reachable through the catalog and its own declared domain."""

    from alphalattice.investment.risk_research.estimators.capability import (
        CANONICAL_NO_RANDOMNESS_SEED,
    )
    from alphalattice.investment.risk_research.estimators.catalog import (
        build_installed_risk_estimator_catalog,
    )

    catalog = build_installed_risk_estimator_catalog()
    handles = {value.recipe_schema_id for value in catalog.binding.ordered_capabilities}
    assert {
        "EWMA_STANDARDIZED_LEDOIT_WOLF_CORRELATION",
        "FAST_SLOW_LEDOIT_WOLF_CORRELATION",
        "DIAGONAL_SHRUNK_CORRELATION",
    } <= handles

    # Through the compiler's own admission route, which is what a Program uses.
    admission = catalog.admit_recipe(
        capability_handle="DIAGONAL_SHRUNK_CORRELATION",
        parameters={"correlation_shrinkage": 1.0},
        seed=CANONICAL_NO_RANDOMNESS_SEED,
    )
    assert admission.adapter_id == "risk-covariance-diagonal-shrunk-ledoit-wolf"
    assert len(admission.recipe_hash) == 64


def test_the_shrunk_diagonal_reproduces_the_control_at_zero_and_deletes_it_at_one() -> None:
    """The equivalence that makes R2 a probe rather than a fourth guess.

    At ``phi = 0`` the values equal the control's exactly. They are *not*
    byte-identical: ``(1 - 0) * C`` preserves a negative zero and ``+ 0 * I``
    destroys it, and ``matrix_content_hash`` is signed-zero sensitive. The values
    are what every weight and every log score downstream depend on, so the value
    equality is the one that carries the claim.
    """

    from alphalattice.investment.risk_research.estimators.covariance import (
        CovarianceCapability,
        estimate_dynamic_covariance,
    )
    from alphalattice.investment.risk_research.estimators.diagonal import (
        DiagonalShrunkCovarianceCapability,
        estimate_diagonal_shrunk_covariance,
    )

    rng = np.random.default_rng(20260822)
    listings = tuple(f"L{index:03d}" for index in range(12))
    returns = np.ascontiguousarray(rng.normal(0.0, 0.01, size=(315, 12)), dtype=np.float64)
    session = date(2024, 7, 1)

    control_capability = CovarianceCapability()
    control_recipe, _ = control_capability.seal(control_capability.parameter_domain.admit({}))
    control = estimate_dynamic_covariance(
        returns=returns,
        ordered_listing_ids=listings,
        formation_session=session,
        recipe=control_recipe,
    )

    diagonal = DiagonalShrunkCovarianceCapability()
    zero_recipe, _ = diagonal.seal(diagonal.parameter_domain.admit({"correlation_shrinkage": 0.0}))
    zero = estimate_diagonal_shrunk_covariance(
        returns=returns,
        ordered_listing_ids=listings,
        formation_session=session,
        recipe=zero_recipe,
    )
    assert np.array_equal(zero.matrix, control.matrix)

    one_recipe, _ = diagonal.seal(diagonal.parameter_domain.admit({"correlation_shrinkage": 1.0}))
    one = estimate_diagonal_shrunk_covariance(
        returns=returns,
        ordered_listing_ids=listings,
        formation_session=session,
        recipe=one_recipe,
    )
    off_diagonal = one.matrix - np.diag(np.diag(one.matrix))
    assert float(np.max(np.abs(off_diagonal))) == 0.0
    assert one.diagnostics.shrinkage == 1.0


# ----------------------------------------------------------- rebalance cadence
def test_a_second_rebalance_clock_installs_through_the_catalog() -> None:
    catalog = build_installed_rebalance_clock_catalog()
    # The whole-book clock joined the catalog with the baseline package at
    # `ea28c8a8` (2026-08-27); the seam this proves is that installation is the
    # catalog's, so the expected set names every installed clock. The later
    # score-axis contract also installs SCORED_FORMATIONS; it is not a test plugin.
    assert set(catalog) == {
        "EVERY_FORMATION",
        "EVERY_N_FORMATIONS",
        "WHOLE_BOOK_EVERY_N_FORMATIONS",
        "SCORED_FORMATIONS",
    }
    weekly = resolve_rebalance_clock("EVERY_N_FORMATIONS", interval=5)
    assert weekly.binding.clock_id == "EVERY_N_FORMATIONS"
    assert weekly.binding.parameters == (("interval", 5),)
    with pytest.raises(PortfolioWalkForwardError):
        resolve_rebalance_clock("NOT_INSTALLED")


def test_cadence_is_measured_from_the_segment_start_not_the_global_axis() -> None:
    """Otherwise each fold would get a different first decision session.

    A validation region begins wherever the split policy put it, so phase against
    the global index would make a cadence comparison partly a comparison of which
    folds happened to start on a decision.
    """

    clock = EveryNFormationsClock(interval=5)
    decisions = [
        index
        for index in range(326, 336)
        if clock.rebalances(formation_index=index, segment_start_index=326)
    ]
    assert decisions == [326, 331]
    assert EveryFormationClock().rebalances(formation_index=7, segment_start_index=3)
    with pytest.raises(PortfolioWalkForwardError):
        EveryNFormationsClock(interval=1)


# ------------------------------------------------------------------ cost lanes
def test_a_second_cost_lane_is_a_declaration_not_a_module_literal() -> None:
    policy = PortfolioCostPolicy(reporting_bps=(2, 5, 10, 20, 40), selection_bps=5)
    assert policy.selection_bps == 5
    # A complete A-to-B asset rotation has one-way turnover one. At 10 bps the
    # installed convention therefore charges 0.001 once, not 0.002 as a
    # per-side label would imply; an out-and-back rotation charges it twice.
    assert np.array_equal(
        policy.cost_fraction(one_way_turnover=np.asarray([1.0, 1.0, 0.5]), cost_bps=10.0),
        np.asarray([0.001, 0.001, 0.0005]),
    )
    assert policy.researcher_payload()["rate_basis"] == ("BASIS_POINTS_PER_ONE_WAY_ASSET_TURNOVER")
    legacy = PortfolioCostPolicy(transaction_cost_convention=None)
    assert legacy.cost_fraction(one_way_turnover=np.asarray([1.0]), cost_bps=10.0)[0] == 0.001
    assert legacy.convention_disposition == "LEGACY_AMBIGUOUS_COST_CONVENTION"
    with pytest.raises(PortfolioWalkForwardError):
        PortfolioCostPolicy(reporting_bps=(5, 10), selection_bps=40)
    with pytest.raises(PortfolioWalkForwardError):
        PortfolioCostPolicy(reporting_bps=(), selection_bps=0)


# --------------------------------------------------------------- active metrics
def test_active_metrics_have_one_owner_and_a_typed_degenerate_refusal() -> None:
    rng = np.random.default_rng(11)
    benchmark = rng.normal(0.0004, 0.01, size=200)
    portfolio = 1.1 * benchmark + rng.normal(0.0, 0.002, size=200)
    metrics = active_path_metrics(portfolio_simple=portfolio, benchmark_simple=benchmark)
    assert 0.9 < metrics.beta < 1.3
    assert metrics.tracking_error > 0.0
    assert np.isfinite(metrics.sharpe)

    flat = np.full(200, 0.0005)
    with pytest.raises(ActiveMetricsError):
        active_path_metrics(portfolio_simple=portfolio, benchmark_simple=flat)


def test_a_portfolio_that_reproduces_its_benchmark_has_infinite_information_ratio() -> None:
    """The convention all three previous copies used, kept by the one that replaced them."""

    rng = np.random.default_rng(3)
    benchmark = rng.normal(0.0004, 0.01, size=120)
    metrics = active_path_metrics(portfolio_simple=benchmark, benchmark_simple=benchmark)
    assert metrics.tracking_error == 0.0
    assert metrics.information_ratio == float("inf")
    assert metrics.beta == pytest.approx(1.0)


# ----------------------------------------------------------- execution budget
def test_the_budget_owner_is_shared_and_keeps_each_desks_error_strings() -> None:
    # The owner reads no environment: the workflow holds a run offline, so no switch is
    # set here.
    from alphalattice.protocols.research_authoring.contracts import (
        ResearchExperimentEnvelope,
    )
    from alphalattice.protocols.research_authoring.execution_policy import (
        ExecutionPlan,
        RuntimeCapabilityRequirement,
        enforce_declared_execution_policy,
    )

    envelope = ResearchExperimentEnvelope.create(
        kind="risk.covariance-development",
        schema_id="research-experiment-envelope",
        data_snapshot_handle="current",
        universe_handle="us-current-index-research",
        sessions={"start": date(2024, 1, 2), "end": date(2024, 6, 28), "as_of": date(2026, 8, 22)},
        budget={"maximum_candidates": 1, "maximum_numerical_calls": 100},
        determinism={"seed": 0, "thread_limit": 1, "network_disabled": True},
        output_workspace="workspaces/probe/evidence",
        baseline_workspace="workspaces/probe/baseline",
        publication_intent="DEVELOPMENT_EVIDENCE_ONLY",
    )
    enforce_declared_execution_policy(
        envelope=envelope,
        plan=ExecutionPlan(planned_numerical_calls=100, planned_trial_count=6),
        requirements=(RuntimeCapabilityRequirement(requires_single_thread=True),),
    )
    with pytest.raises(AuthoringError) as over:
        enforce_declared_execution_policy(
            envelope=envelope, plan=ExecutionPlan(planned_numerical_calls=101)
        )
    assert str(over.value) == "research_authoring.numerical_budget_exceeded"


# ------------------------------------------------------- covariance-blind control
def test_the_installed_equal_weight_control_is_blind_to_the_covariance_it_reports() -> None:
    """The null control for Q1, and it is the one that was already installed.

    ``TOP_K_EQUAL_WEIGHT`` selects through ``stable_top_k`` on the full signed
    score and splits free capital evenly. It never calls the optimizer and never
    reads the covariance to decide, so its weights are identical across Risk
    arms; it does report ``w' Sigma w``, which is not, and that is what makes it
    a calibration read with no optimizer feedback in it.

    A ``TOP_K_SCORE_ONLY_PROPORTIONAL`` policy was drafted for this study and
    withdrawn. It filtered on ``score > 0``, which makes every negatively-scored
    name indistinguishable -- and it was a second copy of a control that already
    existed.
    """

    from alphalattice.investment.portfolio_strategy_lab.contracts import TopKEqualWeightPolicy
    from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
        BoundPolicyDecisionInput,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.top_k_equal_weight import (
        TopKEqualWeightAdapter,
    )

    rng = np.random.default_rng(7)
    size = 40
    # Deliberately centred on zero, so more than half the scores are negative.
    scores = rng.normal(0.0, 0.01, size=size)
    assert int((scores < 0.0).sum()) > 10
    eligible = np.ones(size, dtype=bool)
    sectors = np.zeros((2, size), dtype=np.float64)
    sectors[0, : size // 2] = 1.0
    sectors[1, size // 2 :] = 1.0

    def decide(covariance: np.ndarray) -> tuple[np.ndarray, float]:
        decision = TopKEqualWeightAdapter().decide(
            policy=TopKEqualWeightPolicy(top_k=10, maximum_weight=0.2),
            inputs=BoundPolicyDecisionInput(
                scores=scores,
                covariance=covariance,
                decision_eligible=eligible,
                reference_weights=np.zeros(size),
                sector_exposure_matrix=sectors,
                equal_weight_sector_exposure=np.full(2, 0.5),
            ),
            optimizer=None,  # type: ignore[arg-type]
        )
        return decision.target_weights, decision.predicted_variance

    dense_weights, dense_variance = decide(np.eye(size) * 1e-4 + 2e-5)
    diagonal_weights, diagonal_variance = decide(np.eye(size) * 1e-4)

    # Identical to the byte across two very different covariances. A difference
    # here would be a defect in the study, not a finding about risk.
    assert np.array_equal(dense_weights, diagonal_weights)
    assert dense_weights.sum() == pytest.approx(1.0)
    # The full signed ordering decides the pool: the ten largest scores are held,
    # negative or not, and nothing was truncated at zero.
    expected = set(np.argsort(-scores, kind="stable")[:10].tolist())
    assert set(np.flatnonzero(dense_weights > 0.0).tolist()) == expected
    # And the reported variance still separates the two covariances.
    assert dense_variance > diagonal_variance


# ------------------------------------------------------- simple signed score
def test_the_simple_score_is_signed_standardized_and_loses_nothing() -> None:
    """Every property the deleted frozen-score package failed to have.

    Signed with no clipping, standardized per formation, unresolved cells left
    unresolved rather than zero-filled, and the raw ordering preserved exactly --
    so the score a policy ranks on is the feature it was built from.
    """

    from alphalattice.investment.alpha_research.simple_signal.standardize import (
        finite_listing_counts,
        standardize_cross_section,
    )

    rng = np.random.default_rng(20260822)
    raw = rng.normal(0.0, 1.0, size=(6, 8))
    raw[0, 0] = np.nan
    standardized = standardize_cross_section(raw)

    resolved = standardized[np.isfinite(standardized)]
    assert int((resolved < 0.0).sum()) > 0
    assert np.allclose(np.nanmean(standardized, axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.nanstd(standardized, axis=1, ddof=1), 1.0, atol=1e-12)
    # An unresolved feature stays unresolved: a zero here would read as a real
    # neutral view on a name nobody measured.
    assert np.isnan(standardized[0, 0])
    assert np.array_equal(np.argsort(raw[1]), np.argsort(standardized[1]))
    assert finite_listing_counts(standardized).tolist() == [7, 8, 8, 8, 8, 8]


def test_a_row_with_no_dispersion_is_unresolved_rather_than_zero() -> None:
    from alphalattice.investment.alpha_research.simple_signal.standardize import (
        standardize_cross_section,
    )

    constant = np.full((2, 5), 0.25)
    assert bool(np.isnan(standardize_cross_section(constant)).all())


def test_the_score_axis_is_the_cross_section_it_was_standardized_over() -> None:
    """A session or listing the score does not carry is a refusal, not a gap.

    Widening the axis silently would change what every z means: the moments were
    taken over one cross-section, and a projection onto a different one is a
    different statistic wearing the same name.
    """

    from alphalattice.investment.alpha_research.simple_signal.standardize import (
        SimpleSignalError,
        simple_score_matrix,
    )

    values = np.random.default_rng(3).normal(size=(4, 5))
    sessions = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(4))
    listings = tuple(f"L{index}" for index in range(5))
    projected = simple_score_matrix(
        standardized_values=values,
        score_sessions=sessions,
        score_listing_ids=listings,
        formation_sessions=sessions[1:3],
        ordered_listing_ids=listings[:3],
    )
    assert np.array_equal(projected, values[1:3, :3])
    with pytest.raises(SimpleSignalError, match="session_not_covered"):
        simple_score_matrix(
            standardized_values=values,
            score_sessions=sessions,
            score_listing_ids=listings,
            formation_sessions=(date(2030, 1, 1),),
            ordered_listing_ids=listings[:3],
        )


def test_the_score_coefficient_is_declared_and_zero_is_admitted() -> None:
    """kappa is a policy parameter, and zero is the most informative point.

    ``stable_top_k`` picks the pool from the *unscaled* score before the
    coefficient is applied, so zero is not degenerate -- it separates what the
    score is worth for choosing names from what it is worth for sizing them.
    """

    from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
        build_installed_portfolio_policy_catalog,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.score_risk_cost import (
        ScoreRiskCostTrialRecipe,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.search_domain import (
        stage_six_trial_domains,
    )

    domain = stage_six_trial_domains()["TOP_K_SCORE_RISK_COST"]
    axes = {axis.name for axis in domain.ordered_axes}
    assert "score_utility_coefficient" in axes
    recipe = ScoreRiskCostTrialRecipe.create(
        top_k=50,
        maximum_weight=0.04,
        risk_aversion=1000.0,
        turnover_regularization=0.01,
        score_utility_coefficient=0.0,
    )
    assert recipe.policy_id == "TOP_K_SCORE_RISK_COST"
    assert recipe.score_utility_coefficient == 0.0
    # It routes to the installed adapter unchanged, so the frozen policy union
    # and every published regularization trial stay untouched.
    assert build_installed_portfolio_policy_catalog().resolve(recipe).solver_backed


def test_a_fully_invested_book_survives_the_drift_it_produced() -> None:
    """The two holdings invariants are one invariant, at one tolerance.

    ``execute_orders`` admits cash down to ``-TOLERANCE`` and clamps it to zero,
    so the state it hands on can sum to ``1 + TOLERANCE``. ``drift_holdings``
    used to check that state ten times tighter, which is not a stricter check --
    it is a check of something the producer never promised. It cost 32 of 156
    walk-forward segments in a real study, every one of them the score-weighted
    policy, because a nearly fully invested book is the one whose residual cash
    lands in that band.
    """

    from alphalattice.capabilities.portfolio_backtesting.execution import (
        TOLERANCE,
        drift_holdings,
        execute_orders,
    )

    size = 40
    pretrade = np.full(size, 0.02, dtype=np.float64)
    # Nearly all capital deployed, which is where the residual lands in the band.
    target = np.full(size, 0.025, dtype=np.float64)
    executed, cash, _turnover, _missed = execute_orders(
        pretrade_weights=pretrade,
        pretrade_cash=1.0 - float(pretrade.sum()),
        target_weights=target,
        execution_available=np.ones(size, dtype=bool),
    )
    assert cash >= 0.0

    # The worst state the producer's own contract permits, handed straight to the
    # consumer. Before the fix this raised.
    strained = np.array(executed, dtype=np.float64)
    strained[0] += TOLERANCE
    drifted, drifted_cash, _gross = drift_holdings(
        weights=strained,
        cash=cash,
        returns=np.linspace(-0.02, 0.03, size),
    )
    assert abs(float(drifted.sum()) + drifted_cash - 1.0) <= TOLERANCE * 1.01

    # Still a check: a state that is genuinely not normalized is still refused.
    with pytest.raises(PortfolioWalkForwardError, match="holdings_carry_invalid"):
        drift_holdings(
            weights=np.full(size, 0.05, dtype=np.float64),
            cash=0.5,
            returns=np.zeros(size, dtype=np.float64),
        )


# ------------------------------------------------- score clock and boundaries
def test_the_score_clock_comes_from_the_feature_owner_not_from_this_package() -> None:
    """requirement: observation and availability are the Feature owner's, derived.

    ``observation_clock_for`` derives the clock from the recipe that already
    carries it -- deliberately, because a clock stored beside a recipe is a
    number that can disagree with the window it describes. This package asks for
    it; it does not compute one, and it may not hold an execution term.
    """

    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
    from alphalattice.foundation.feature_engine.catalog.observation_clock import (
        installed_feature_availability_policy,
        observation_clock_for,
    )
    from alphalattice.investment.alpha_research.simple_signal.provenance import (
        resolve_feature_clock,
    )
    from alphalattice.investment.alpha_research.simple_signal.standardize import SimpleSignalError

    resolved = resolve_feature_clock("mom_252_21")
    spec = next(v for v in FeatureCatalog.load().factors if v.factor_id == "mom_252_21")
    owner = observation_clock_for(spec)
    availability = installed_feature_availability_policy()

    assert resolved.observation_session_offset_sessions == owner.latest_consumed_session_offset
    assert resolved.availability_delay_sessions == availability.publication_delay_sessions
    assert resolved.observation_clock_hash == owner.clock_hash
    assert resolved.source_interval == owner.source_interval_rendered == "[t-252,t-21]"

    # The 21 is the Formula's own economic skip, not a portfolio's tradability
    # margin. The Feature remediation exists to keep those separable.
    assert owner.formula_skip_sessions == 21

    # A feature the catalog does not carry has no owner-derived clock, and a
    # default of zero would be indistinguishable from a verified one.
    with pytest.raises(SimpleSignalError, match="feature_clock_not_declared"):
        resolve_feature_clock("not_a_real_feature")


def test_a_score_producer_cannot_state_an_execution_term() -> None:
    """regression: the producer used to hard-code the execution constants.

    ``FORMATION_SESSION_CLOSE``, ``entry_offset_sessions = 1`` and
    ``NEXT_COMMON_SESSION_OFFICIAL_OPEN`` lived in a score binding, copied out of
    ``causal_outcomes.execution``. A producer that states an entry offset has
    taken a position on when its own values are tradable, and a later change to
    the execution method would have left the copy silently stale.
    """

    from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
        ScoreObservationAuthority,
    )
    from alphalattice.investment.alpha_research.simple_signal import authority as producer

    fields = set(ScoreObservationAuthority.model_fields)
    forbidden = {
        "decision_cutoff_convention",
        "entry_offset_sessions",
        "entry_timing",
        "exit_offset_sessions",
        "exit_timing",
        "rebalance_clock_id",
    }
    assert not fields & forbidden
    # ``extra="forbid"``, so one cannot be added by a producer that would like to.
    # Pydantic wraps the refusal, so the exception type is its own rather than a
    # Desk error -- what is being pinned is that the field cannot enter at all.
    with pytest.raises(ValidationError, match="entry_offset_sessions"):
        ScoreObservationAuthority.create(
            observation_session_offset_sessions=21,
            availability_delay_sessions=0,
            availability_policy_id="p",
            availability_policy_hash="a" * 64,
            observation_clock_hash="b" * 64,
            formula_observation_semantics="x",
            source_authority_id="y",
            methodology_identity="c" * 64,
            strategy_scope="CONDITIONAL_FIXED_SCORE_NOT_A_STRATEGY_SIGNAL",
            entry_offset_sessions=1,
        )

    source = Path(producer.__file__).read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    body = executable.split('"""')
    code = "".join(body[::2])
    assert "NEXT_COMMON_SESSION_OFFICIAL_OPEN" not in code
    assert "entry_offset_sessions" not in code


def test_the_score_producer_clock_reaches_the_shared_owner_unchanged() -> None:
    """requirement: a producer's two offsets become two anchors, and nothing else.

    This module used to implement the ordering itself -- ``available <= decision
    < entry < exit``, with no way to express an order deadline, so a score not
    finished at the close could only be admitted by moving its own observation
    back a session. That chain is retired. What is left is the resolver, and the
    thing to check about a resolver is that it neither adds information nor
    loses it.
    """

    from alphalattice.capabilities.causal_inputs.contracts import CausalInputAuthority
    from alphalattice.capabilities.portfolio_inputs.signed_score import contracts as signed_score

    assert not hasattr(signed_score, "require_causal_score_events")
    assert not hasattr(signed_score, "score_available_at")

    observation = signed_score.ScoreObservationAuthority.create(
        observation_session_offset_sessions=21,
        availability_delay_sessions=0,
        availability_policy_id="feature-availability.daily-provider-session-close",
        availability_policy_hash="a" * 64,
        observation_clock_hash="b" * 64,
        formula_observation_semantics="total-return close at t-21 over [t-252,t-21]",
        source_authority_id="alpha_research.simple_signal:mom_252_21:factor.mom_252_21.v1",
        methodology_identity="c" * 64,
        strategy_scope="CONDITIONAL_FIXED_SCORE_NOT_A_STRATEGY_SIGNAL",
    )
    authority = signed_score.score_input_authority(
        observation, surface_hash="d" * 64, input_id="alpha_score"
    )
    assert isinstance(authority, CausalInputAuthority)
    # The economic skip reaches backwards; the publication delay reaches
    # forwards. Two terms, not one net offset, because only one is the Formula's.
    assert authority.observed_through.anchor.offset_sessions == -21
    assert authority.source_available.anchor.offset_sessions == -21
    # Availability is what the repository asserts about its sources, not what a
    # venue published, and the basis says which.
    assert authority.source_available.basis == "INSTALLED_SOURCE_AVAILABILITY_POLICY"
    assert authority.observed_through.basis == "EXCHANGE_PUBLISHED_FACT"
    # A producer states observation and availability and nothing about trading.
    assert "entry" not in set(CausalInputAuthority.model_fields)
    assert authority.temporal_usage == "DECISION_INPUT"
    # The owner handle is the producer's own sealed declaration, so a verifier
    # re-deriving it at the producer compares against a whole document rather
    # than against a field that was easy to re-seal.
    assert authority.owner_identity_hash == observation.authority_hash


def test_a_score_that_lost_its_negative_half_is_refused_at_the_boundary() -> None:
    """regression: every score defect this study had removed the negative half.

    ``score > 0`` selection, non-negative slope calibration and dispersion
    reconstruction all shipped at some point, and each looked like a signal until
    someone checked the signs. The consumer checks rather than trusts.
    """

    from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
        SignedScoreClockError,
        require_signed_finite,
    )

    signed = np.random.default_rng(4).normal(size=(30, 12))
    require_signed_finite(signed)

    with pytest.raises(SignedScoreClockError, match="not_signed"):
        require_signed_finite(np.abs(signed))
    with pytest.raises(SignedScoreClockError, match="identically_zero"):
        require_signed_finite(np.zeros((30, 12)))
    thin = np.full((30, 12), np.nan)
    thin[:, 0] = 1.0
    with pytest.raises(SignedScoreClockError, match="formation_coverage_insufficient"):
        require_signed_finite(thin)
