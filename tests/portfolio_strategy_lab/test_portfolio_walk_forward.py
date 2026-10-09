"""Causal walk-forward regression coverage for Portfolio development."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.investment.portfolio_strategy_lab.contracts import (
    PortfolioScoreMode,
    PortfolioStrategyLabMandate,
    TopKEqualWeightPolicy,
    seal_contract,
)

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"

_HASHES = tuple(f"{value:064x}" for value in range(1, 22))


def _mandate(sessions: tuple[date, ...]) -> PortfolioStrategyLabMandate:
    return seal_contract(
        PortfolioStrategyLabMandate,
        "mandate_hash",
        portfolio_input_bundle_hash=_HASHES[0],
        portfolio_development_mandate_hash=_HASHES[1],
        alpha_score_surface_hash=_HASHES[2],
        risk_surface_hash=_HASHES[3],
        tradability_bundle_hash=_HASHES[4],
        universe_epoch_hash=_HASHES[5],
        ordered_listing_ids_hash=_HASHES[6],
        ordered_candidate_ids=("candidate",),
        score_modes=(
            PortfolioScoreMode.STOCK_ONLY,
            PortfolioScoreMode.STOCK_PLUS_SECTOR_COMPONENT,
        ),
        development_formation_sessions=sessions,
        bootstrap_resamples=2000,
        limitations=("Development only.",),
    )


@dataclass(frozen=True, slots=True)
class _DevelopmentWorkspace:
    """The fixture supplies the fields of ``PortfolioWalkForwardWorkspace``."""

    mandate: PortfolioStrategyLabMandate
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    raw_prediction_summaries: dict[str, dict[str, float]]
    scores: dict[tuple[str, PortfolioScoreMode], np.ndarray]
    covariances: np.ndarray
    decision_eligible: np.ndarray
    execution_available: np.ndarray
    realized_simple_returns: np.ndarray
    causal_adv20: np.ndarray
    sector_ids: tuple[str, ...]
    sector_exposure_matrix: np.ndarray
    equal_weight_sector_exposure: np.ndarray
    workspace_hash: str
    passive_returns_by_session: dict[date, np.ndarray] = dataclass_field(default_factory=dict)
    causal_rank_return_curve: object | None = None


def _workspace(
    *,
    missing_held_return: bool = False,
    missing_nonholding_adv20: bool = False,
) -> _DevelopmentWorkspace:
    sessions = tuple(date(2022, 1, 1) + timedelta(days=value) for value in range(494))
    asset_count = 8
    score = np.tile(np.linspace(1.0, -1.0, asset_count), (494, 1))
    covariance = np.tile(np.eye(asset_count)[None, :, :] * 0.01, (494, 1, 1))
    decision = np.ones((494, asset_count), dtype=np.bool_)
    execution = np.ones((494, asset_count), dtype=np.bool_)
    execution[0, 0] = False
    returns = np.tile(np.linspace(0.002, -0.001, asset_count), (494, 1))
    if missing_held_return:
        returns[1, 0] = np.nan
    adv20 = np.full((494, asset_count), 1_000_000.0)
    if missing_nonholding_adv20:
        adv20[:, -1] = np.nan
    sector_matrix = np.zeros((2, asset_count))
    sector_matrix[0, :4] = 1.0
    sector_matrix[1, 4:] = 1.0
    for value in (score, covariance, decision, execution, returns, adv20, sector_matrix):
        value.setflags(write=False)
    equal_sector = np.asarray([0.5, 0.5])
    equal_sector.setflags(write=False)
    return _DevelopmentWorkspace(
        mandate=_mandate(sessions),
        formation_sessions=sessions,
        ordered_listing_ids=tuple(f"L{value}" for value in range(asset_count)),
        candidate_ids=("candidate",),
        raw_prediction_summaries={
            "candidate": {"finite_fraction": 1.0, "mean": 0.0, "standard_deviation": 1.0}
        },
        scores={
            ("candidate", PortfolioScoreMode.STOCK_ONLY): score,
            ("candidate", PortfolioScoreMode.STOCK_PLUS_SECTOR_COMPONENT): score,
        },
        covariances=covariance,
        decision_eligible=decision,
        execution_available=execution,
        realized_simple_returns=returns,
        causal_adv20=adv20,
        sector_ids=("S1", "S2"),
        sector_exposure_matrix=sector_matrix,
        equal_weight_sector_exposure=equal_sector,
        workspace_hash=_HASHES[7],
        passive_returns_by_session={
            session: returns[index] for index, session in enumerate(sessions)
        },
    )


class _ClockOwnerView:
    """An honest causal-outcome owner that publishes the full row schema.

    Distinct from the multi-graph forgery owner further down this file, which
    exists to answer a re-sealed handle with contradicting rows and reads only
    the four columns the events resolution asks for. This one answers the
    six the *clock* derivation needs -- both session labels as well as the three
    instants -- with timezone-aware timestamps, because that is the schema the
    real snapshot is written under and a naive timestamp is refused by design.
    """

    def __init__(self, *, axis: tuple[date, ...], manifest_ref: str) -> None:
        self.axis = axis
        self.manifest_ref = manifest_ref
        self.read_count = 0
        self.seal_count = 0

    def read_development_sessions(self, manifest_ref: str, sessions: tuple[date, ...]):  # type: ignore[no-untyped-def]
        assert manifest_ref == self.manifest_ref
        self.read_count += 1
        position = {value: index for index, value in enumerate(self.axis)}
        rows = [
            {
                "formation_session": value,
                "formation_close_at": datetime(
                    value.year, value.month, value.day, 20, 0, tzinfo=UTC
                ),
                "entry_session": self.axis[position[value] + 1],
                "entry_open_at": _open_at(self.axis[position[value] + 1]),
                "holding_end_session": self.axis[position[value] + 2],
                "holding_end_open_at": _open_at(self.axis[position[value] + 2]),
            }
            for value in sessions
        ]
        return pa.Table.from_pylist(rows)

    def available_development_sessions(self, manifest_ref: str, *, holding_end_through: date):  # type: ignore[no-untyped-def]
        del manifest_ref, holding_end_through
        return self.axis[:-2]

    def resolve_method_seal(self, snapshot_hash: str):  # type: ignore[no-untyped-def]
        del snapshot_hash
        self.seal_count += 1
        from alphalattice.foundation.causal_outcomes.execution.methods import (
            build_one_session_recipe,
        )

        recipe = build_one_session_recipe()
        return SimpleNamespace(
            disposition="METHOD_BOUND",
            binding=SimpleNamespace(
                binding_hash=_HASHES[14],
                recipe_id=str(recipe.recipe_id),
                recipe_hash=str(recipe.recipe_hash),
            ),
        )


def _open_at(session: date) -> datetime:
    return datetime(session.year, session.month, session.day, 13, 30, tzinfo=UTC)


def test_execution_clock_history_serves_risk_without_expanding_event_axis() -> None:
    """regression: Risk window opens come from owner history, not invented clocks."""

    from alphalattice.capabilities.causal_inputs.admission import admit_decision_input_set
    from alphalattice.capabilities.causal_inputs.schedules import (
        resolve_installed_schedule_handle,
        resolve_strategy_decision_schedule,
    )
    from alphalattice.foundation.causal_outcomes.execution.methods import (
        build_installed_execution_outcome_method_catalog,
    )
    from alphalattice.investment.portfolio_strategy_lab.inputs.development import (
        resolve_execution_clock_context,
    )
    from alphalattice.investment.risk_research.experiments.temporal import (
        RISK_READINESS_BUDGET,
        risk_covariance_authority,
    )

    axis = tuple(date(2022, 1, 1) + timedelta(days=index) for index in range(324))
    sessions = axis[316:-2]
    manifest_ref = "manifests/" + _HASHES[3] + ".json"
    owner = _ClockOwnerView(axis=axis, manifest_ref=manifest_ref)
    execution = resolve_execution_clock_context(
        reader=owner,
        manifest_ref=manifest_ref,
        sessions=sessions,
        history_margin_sessions=316,
    )
    recipe = build_installed_execution_outcome_method_catalog().resolve(
        execution.events.execution_recipe_id
    )
    schedule = resolve_strategy_decision_schedule(
        schedule_handle=resolve_installed_schedule_handle(recipe.recipe_id),
        recipe=recipe,
        method_binding_hash=execution.events.method_binding_hash,
    )
    admitted = admit_decision_input_set(
        authorities=(
            risk_covariance_authority(
                surface_hash=_HASHES[18],
                owner_identity_hash=_HASHES[18],
                input_id="risk_covariance",
                readiness_handle=RISK_READINESS_BUDGET,
            ),
        ),
        schedule=schedule,
        formation_sessions=sessions,
        ordered_axis=tuple(sorted(execution.session_clocks)),
        session_clocks=execution.session_clocks,
    )
    assert admitted.disposition == "ADMITTED"
    assert execution.events.ordered_formation_sessions == sessions
    assert owner.read_count == 1


def _route_probe(monkeypatch, *, owner: _ClockOwnerView, clocks) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Count every numerical entry point a refusal must not reach.

    Load-bearing by construction: each counter is incremented by the real symbol
    the route calls, so restoring the old ordering moves a count from 0 to 1 and
    the assertion fails on the number rather than on an exception type.
    """

    from alphalattice.capabilities.portfolio_inputs.benchmark import (
        PortfolioBenchmarkSourceAuthority,
        PortfolioBenchmarkSourceBinding,
    )
    from alphalattice.investment.portfolio_strategy_lab.campaign import authority as route
    from alphalattice.investment.portfolio_strategy_lab.campaign.authority import (
        ResolvedStageSixCovariance,
    )

    counts = {
        "covariance_project": 0,
        "market_numerical_load": 0,
        "market_exposure_surface": 0,
        "benchmark_surface": 0,
    }

    def _count(name: str):  # type: ignore[no-untyped-def]
        # Counts and returns rather than raising. A spy that raised would make
        # the old ordering fail with *an* error, and the point of these tests is
        # that it fails on the number: the route reached a numerical entry point
        # it had no business reaching, and the count says which one.
        def _spy(*args: object, **kwargs: object) -> object:
            del args, kwargs
            counts[name] += 1
            return SimpleNamespace()

        return _spy

    monkeypatch.setattr(
        route, "resolve_portfolio_market_clocks", lambda **kwargs: clocks, raising=True
    )
    monkeypatch.setattr(
        route, "load_portfolio_market_inputs", _count("market_numerical_load"), raising=True
    )
    monkeypatch.setattr(
        route, "build_market_exposure_surface", _count("market_exposure_surface"), raising=True
    )
    monkeypatch.setattr(
        route, "build_portfolio_benchmark_surface", _count("benchmark_surface"), raising=True
    )
    monkeypatch.setattr(
        route,
        "resolve_portfolio_benchmark_source_binding",
        lambda **kwargs: PortfolioBenchmarkSourceBinding.create(
            campaign=PortfolioBenchmarkSourceAuthority.create(
                formation_sessions=tuple(cast(tuple[date, ...], kwargs["campaign_sessions"])),
                universe_manifest_revision=_HASHES[16],
                raw_input_hash=_HASHES[17],
                action_set_hash=_HASHES[18],
            ),
            exposure=PortfolioBenchmarkSourceAuthority.create(
                formation_sessions=tuple(cast(tuple[date, ...], kwargs["exposure_sessions"])),
                universe_manifest_revision=_HASHES[16],
                raw_input_hash=_HASHES[17],
                action_set_hash=_HASHES[18],
            ),
        ),
        raising=True,
    )
    monkeypatch.setattr(
        ResolvedStageSixCovariance, "project", _count("covariance_project"), raising=True
    )
    # The resolution fixture is a stand-in for the metadata half of a real
    # resolution, so the same spy is bound to it: patching only the real class
    # would let the old ordering fail on a missing attribute instead of a count.
    counts["_covariance_spy"] = _count("covariance_project")  # type: ignore[assignment]
    del owner
    return counts


def _campaign_resolution(  # type: ignore[no-untyped-def]
    *, axis: tuple[date, ...], score, counts, alpha=None
):
    """The metadata half of a resolution: axes and identities, no matrices."""

    listings = ("AAA", "BBB", "CCC")
    return SimpleNamespace(
        request=SimpleNamespace(
            signal_source="PORTFOLIO_SIMPLE_SIGNED_SCORE",
            reporting_cost_bps=(0,),
            selection_cost_bps=0,
            rebalance_clock_id="EVERY_FORMATION",
            rebalance_interval=None,
        ),
        alpha=alpha,
        simple_score=score,
        frozen_panel_score=None,
        risk=SimpleNamespace(
            binding=SimpleNamespace(dossier_or_root_hash=_HASHES[14], decision_hash=_HASHES[15]),
            comparison=SimpleNamespace(comparison_hash=_HASHES[16], selected_candidate_id="R0"),
        ),
        sector=None,
        source_score_horizon=None,
        excluded_formation_sessions=(),
        coverage_note="fixture",
        score_projection=lambda sessions, names: np.zeros((len(sessions), len(names))),
        formation_sessions=axis,
        decision_universe=SimpleNamespace(
            ordered_decision_listing_ids=listings,
            risk_position_by_decision_index=(0, 1, 2),
        ),
        stage_six_covariance=SimpleNamespace(
            formation_sessions=axis,
            # Identity only. The matrices behind it are what admission must not
            # reach, and `project` is spied on to prove it does not.
            surface=SimpleNamespace(surface_hash=_HASHES[18]),
            project=counts["_covariance_spy"],
        ),
    )


def _signed_score(sessions: tuple[date, ...]):  # type: ignore[no-untyped-def]
    from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
        ResolvedSignedScore,
        ScoreObservationAuthority,
    )

    return ResolvedSignedScore(
        binding_hash=_HASHES[7],
        values_identity=_HASHES[8],
        ordered_formation_sessions=sessions,
        ordered_listing_ids=("AAA", "BBB", "CCC"),
        observation=ScoreObservationAuthority.create(
            observation_session_offset_sessions=0,
            availability_delay_sessions=0,
            availability_policy_id="feature-availability.daily-provider-session-close",
            availability_policy_hash=_HASHES[9],
            observation_clock_hash=_HASHES[10],
            formula_observation_semantics="close of t",
            source_authority_id="alpha_research.simple_signal:fixture",
            methodology_identity=_HASHES[11],
            strategy_scope="CONDITIONAL_FIXED_SCORE_NOT_A_STRATEGY_SIGNAL",
        ),
        rederived=True,
        disposition="REDERIVED",
    )


def _publish_route_marks(artifact_root: Path, *, sessions) -> str:  # type: ignore[no-untyped-def]
    """Publish one mark surface for the route fixture and return its identity."""

    _store, surface = _publish_marks(
        artifact_root,
        sessions=tuple(sessions),
        prices={
            listing: tuple((100.0, 101.0) for _ in sessions) for listing in ("AAA", "BBB", "CCC")
        },
    )
    return str(surface.surface_hash)


def _operational_route_mark(artifact_root: Path, surface_hash: str):  # type: ignore[no-untyped-def]
    """The same exact store/manifest pair the CLI passes to Campaign composition."""

    from alphalattice.foundation.market_data_ops.publication.session_marks import (
        SessionMarkArtifactStore,
    )

    store = SessionMarkArtifactStore(artifact_root)
    return store, store.load_manifest(surface_hash)


def _publish_operational_tradability(  # type: ignore[no-untyped-def]
    artifact_root: Path,
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    watermark_hash: str,
):
    """Publish one tiny exact bundle through the real Tradability owner."""

    from alphalattice.capabilities.portfolio_inputs.tradability.contracts import (
        DecisionTradabilityStatus,
        ExecutionObservationMethod,
        HistoricalDecisionTradabilitySurface,
        HistoricalExecutionAvailabilitySurface,
        HistoricalTradabilityBundle,
        seal_tradability_contract,
    )
    from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
        TradabilityArtifactStore,
    )
    from alphalattice.kernel.data.enums import ExecutionSessionStatus

    store = TradabilityArtifactStore(artifact_root)
    intended = tuple(value + timedelta(days=1) for value in sessions)
    decision_rows = [
        {
            "formation_session": session,
            "intended_execution_session": intended[index],
            "listing_id": listing,
            "decision_status": DecisionTradabilityStatus.PLANNED_ORDER_ELIGIBLE.value,
            "reason_code": "CAUSAL_MARKET_DATA_COMPLETE",
            "causal_adv20": 1_000_000.0,
            "observed_through": session,
            "source_row_hash": _HASHES[0],
            "row_hash": _HASHES[1],
        }
        for index, session in enumerate(sessions)
        for listing in listings
    ]
    execution_rows = [
        {
            "formation_session": session,
            "intended_execution_session": intended[index],
            "listing_id": listing,
            "execution_status": ExecutionSessionStatus.VERIFIED_ELIGIBLE.value,
            "observation_method": ExecutionObservationMethod.POST_SESSION_DAILY_BAR.value,
            "observed_through": intended[index],
            "source_row_hash": _HASHES[2],
            "row_hash": _HASHES[3],
        }
        for index, session in enumerate(sessions)
        for listing in listings
    ]
    decision_chunk = store.publish_chunk(
        surface_kind="DECISION", table=pa.Table.from_pylist(decision_rows)
    )
    execution_chunk = store.publish_chunk(
        surface_kind="EXECUTION", table=pa.Table.from_pylist(execution_rows)
    )
    common = {
        "universe_epoch_hash": _HASHES[4],
        "ordered_listing_ids": listings,
        "source_watermark_hash": watermark_hash,
        "schedule_hash": _HASHES[5],
        "formation_sessions": sessions,
        "intended_execution_sessions": intended,
        "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        "limitations": ("fixture",),
    }
    decision = seal_tradability_contract(
        HistoricalDecisionTradabilitySurface,
        "surface_hash",
        **common,
        chunks=(decision_chunk,),
        eligible_row_count=len(decision_rows),
        unavailable_row_count=0,
    )
    execution = seal_tradability_contract(
        HistoricalExecutionAvailabilitySurface,
        "surface_hash",
        **common,
        chunks=(execution_chunk,),
        execution_status_counts={
            ExecutionSessionStatus.VERIFIED_ELIGIBLE.value: len(execution_rows)
        },
    )
    bundle = seal_tradability_contract(
        HistoricalTradabilityBundle,
        "bundle_hash",
        universe_epoch_hash=_HASHES[4],
        schedule_hash=_HASHES[5],
        decision_surface_hash=decision.surface_hash,
        execution_surface_hash=execution.surface_hash,
        formation_count=len(sessions),
        asset_count=len(listings),
        coverage_start=sessions[0],
        coverage_end=sessions[-1],
    )
    for category, artifact, identity in (
        ("decision/manifests", decision, "surface_hash"),
        ("execution/manifests", execution, "surface_hash"),
        ("bundles", bundle, "bundle_hash"),
    ):
        store.publish_json(
            category=category,
            payload=artifact.model_dump(mode="json"),
            identity_field=identity,
        )
    return store, bundle, decision, execution


def test_a_future_entry_open_cannot_move_the_real_rebalance_decision() -> None:
    """A future entry open cannot move the real rebalance decision."""

    from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
        PortfolioPolicyDecisionProvider,
    )

    provider = PortfolioPolicyDecisionProvider(
        workspace=_workspace(),
        candidate_id="candidate",
        score_mode=PortfolioScoreMode.STOCK_ONLY,
        policy=TopKEqualWeightPolicy(top_k=4, maximum_weight=0.25),
    )
    reference = np.full(8, 1.0 / 8.0)
    quiet = np.full(8, 1.0 / 8.0)
    violent = np.array([0.9, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01])

    left = provider(
        formation_index=3,
        reference_weights=reference,
        pretrade_weights=quiet,
        decision_mode="REBALANCE",
    )
    right = provider(
        formation_index=3,
        reference_weights=reference,
        pretrade_weights=violent,
        decision_mode="REBALANCE",
    )
    assert np.array_equal(left.target_weights, right.target_weights)
    assert left.predicted_variance == right.predicted_variance
    assert left.decision_mode == right.decision_mode == "REBALANCE"
    # The two pretrade books really were different, so the invariance is a
    # finding rather than an accident of the fixture.
    assert not np.array_equal(quiet, violent)


def test_a_hold_makes_no_decision_and_no_forecast() -> None:
    """A hold makes no decision and no forecast."""

    from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
        PortfolioPolicyDecisionProvider,
    )

    provider = PortfolioPolicyDecisionProvider(
        workspace=_workspace(),
        candidate_id="candidate",
        score_mode=PortfolioScoreMode.STOCK_ONLY,
        policy=TopKEqualWeightPolicy(top_k=4, maximum_weight=0.25),
    )
    drifted = np.array([0.3, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0])
    before = provider.decided_count
    decision = provider(
        formation_index=3,
        reference_weights=np.full(8, 1.0 / 8.0),
        pretrade_weights=drifted,
        decision_mode="HOLD",
    )
    assert decision.decision_mode == "HOLD"
    # No adapter call, so no decision was counted and no solver ran.
    assert provider.decided_count == before
    assert decision.solver_call_count == 0
    # The realised path is continuous -- the book is carried, not rebuilt -- and
    # the forecast is absent rather than computed from an ex-post state.
    assert np.array_equal(decision.target_weights, drifted)
    # Absent, not zero. A zero survives a durable round trip and reads as a
    # forecast to anything that does not also check the mode.
    assert decision.predicted_variance is None


def test_close_of_t_information_does_reach_the_decision_state() -> None:
    """Information available at the session close reaches the decision state."""

    from alphalattice.capabilities.portfolio_backtesting.execution import drift_holdings

    executed_prev = np.array([0.4, 0.35, 0.25])
    quiet, _cash, _gross = drift_holdings(
        weights=executed_prev, cash=0.0, returns=np.array([0.0, 0.0, 0.0])
    )
    moved, _cash2, _gross2 = drift_holdings(
        weights=executed_prev, cash=0.0, returns=np.array([0.20, -0.10, 0.05])
    )
    assert not np.array_equal(quiet, moved)
    assert np.array_equal(quiet, executed_prev)


def _transition_binding(**overrides):  # type: ignore[no-untyped-def]
    """A complete close-marked state transition, with one term at a time moved."""

    from alphalattice.capabilities.portfolio_backtesting.clocks import EveryFormationClock
    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        MARKED_TO_MARKET_AT_CLOSE_T,
        PortfolioStateTransitionBinding,
    )

    values: dict = {
        "transition_implementation_hash": _HASHES[0],
        "execution_method_binding_hash": _HASHES[1],
        "rebalance_clock": EveryFormationClock().binding,
        "reference_mark_method": MARKED_TO_MARKET_AT_CLOSE_T,
        "mark_manifest_ref": f"manifests/{_HASHES[3]}.json",
        "mark_surface_hash": _HASHES[3],
        "mark_epoch_hash": _HASHES[4],
        "mark_price_basis": "split_adjusted",
        "mark_source_watermark_hash": _HASHES[5],
        "mark_availability_policy_hash": _HASHES[6],
        "mark_availability_policy_id": "feature-availability.daily-provider-session-close",
    }
    values.update(overrides)
    return PortfolioStateTransitionBinding.create(**values)


def test_the_holdings_transition_follows_the_installed_clock_not_a_hardcoded_step() -> None:
    """The holdings transition follows the installed clock not a hardcoded step."""

    from alphalattice.capabilities.portfolio_backtesting.clocks import (
        EveryFormationClock,
        EveryNFormationsClock,
    )

    daily = EveryFormationClock()
    weekly = EveryNFormationsClock(interval=5)
    assert daily.rebalances(formation_index=3, segment_start_index=0)
    assert daily.rebalances(formation_index=4, segment_start_index=0)
    assert weekly.rebalances(formation_index=0, segment_start_index=0)
    assert not weekly.rebalances(formation_index=1, segment_start_index=0)
    assert weekly.rebalances(formation_index=5, segment_start_index=0)
    assert daily.binding.clock_id != weekly.binding.clock_id


def test_buffered_equal_weight_keeps_a_full_book_without_covariance_or_nonrebalance_trade() -> None:
    """50-name book is direct selection/allocation, never a QP candidate axis."""

    from alphalattice.capabilities.portfolio_backtesting.clocks import (
        WholeBookEveryNFormationsClock,
    )
    from alphalattice.capabilities.portfolio_backtesting.segments import (
        run_portfolio_walk_forward_segment as run_segment,
    )
    from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
        PortfolioPolicyDecisionProvider,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.buffered_equal_weight import (
        WholeBookHysteresisEqualWeightRecipe,
        whole_book_equal_weight_target,
        whole_book_hysteresis_selection,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
        build_installed_portfolio_policy_catalog,
    )

    asset_count = 200
    listing_ids = ("Z000", "A000", *(f"L{index:03d}" for index in range(2, asset_count)))
    scores = np.linspace(float(asset_count), 1.0, asset_count)
    scores[:2] = asset_count + 1.0
    eligible = np.ones(asset_count, dtype=np.bool_)
    cold = whole_book_hysteresis_selection(
        scores=scores,
        decision_eligible=eligible,
        ordered_listing_ids=listing_ids,
        previous_target_weights=None,
        top_k=50,
        exit_rank=150,
    )
    assert tuple(cold[:2]) == (1, 0)
    held = np.zeros(asset_count)
    ranked = np.asarray(
        sorted(range(asset_count), key=lambda index: (-scores[index], listing_ids[index]))
    )
    held[ranked[149]] = 0.02
    held[ranked[150]] = 0.02
    selected = whole_book_hysteresis_selection(
        scores=scores,
        decision_eligible=eligible,
        ordered_listing_ids=listing_ids,
        previous_target_weights=held,
        top_k=50,
        exit_rank=150,
    )
    assert selected.size == len(set(selected.tolist())) == 50
    assert ranked[149] in selected
    assert ranked[150] not in selected

    frozen_eligible = np.array(eligible, copy=True)
    frozen_eligible[-1] = False
    reference = np.zeros(asset_count)
    reference[-1] = 0.05
    frozen_previous = np.zeros(asset_count)
    frozen_previous[-1] = 0.02
    assert -1 not in whole_book_hysteresis_selection(
        scores=scores,
        decision_eligible=frozen_eligible,
        ordered_listing_ids=listing_ids,
        previous_target_weights=frozen_previous,
        top_k=50,
        exit_rank=150,
    )
    target = whole_book_equal_weight_target(
        selected=selected,
        decision_eligible=frozen_eligible,
        reference_weights=reference,
        top_k=50,
    )
    assert target[-1] == pytest.approx(0.05)
    assert np.allclose(target[selected], 0.95 / 50.0)
    assert target.sum() == pytest.approx(1.0)

    # The shared workspace owner requires its durable 494-session axis.  The
    # segment below still exercises only six formations, so it cannot turn
    # this policy proof into a Campaign or a numerical study.
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(494))
    engine_asset_count = 50
    engine_listing_ids = tuple(f"E{index:03d}" for index in range(engine_asset_count))
    score_matrix = np.tile(
        np.linspace(float(engine_asset_count), 1.0, engine_asset_count),
        (len(sessions), 1),
    )
    nan_covariances = np.full((len(sessions), engine_asset_count, engine_asset_count), np.nan)
    decision = np.ones_like(score_matrix, dtype=np.bool_)
    execution = np.ones_like(score_matrix, dtype=np.bool_)
    returns = np.zeros_like(score_matrix)
    adv20 = np.ones_like(score_matrix)
    sector = np.ones((1, engine_asset_count))
    workspace = _DevelopmentWorkspace(
        mandate=_mandate(sessions),
        formation_sessions=sessions,
        ordered_listing_ids=engine_listing_ids,
        candidate_ids=("candidate",),
        raw_prediction_summaries={
            "candidate": {"finite_fraction": 1.0, "mean": 0.0, "standard_deviation": 1.0}
        },
        scores={("candidate", PortfolioScoreMode.STOCK_ONLY): score_matrix},
        covariances=nan_covariances,
        decision_eligible=decision,
        execution_available=execution,
        realized_simple_returns=returns,
        causal_adv20=adv20,
        sector_ids=("S1",),
        sector_exposure_matrix=sector,
        equal_weight_sector_exposure=np.array([1.0]),
        workspace_hash=_HASHES[20],
        passive_returns_by_session={
            session: returns[index] for index, session in enumerate(sessions)
        },
    )
    provider = PortfolioPolicyDecisionProvider(
        workspace=workspace,
        candidate_id="candidate",
        score_mode=PortfolioScoreMode.STOCK_ONLY,
        policy=WholeBookHysteresisEqualWeightRecipe.create(),
        optimizer=None,
        policies=build_installed_portfolio_policy_catalog(),
    )
    segment = run_segment(
        workspace=workspace,
        decision_provider=provider,
        start_index=0,
        stop_index=6,
        rebalance_clock=WholeBookEveryNFormationsClock(interval=3, offset=1),
    )
    assert segment.decision_modes == ("HOLD", "REBALANCE", "HOLD", "HOLD", "REBALANCE", "HOLD")
    assert all(segment.one_way_turnovers[index] == 0.0 for index in (0, 2, 3, 5))
    assert segment.predicted_variances == (None,) * 6
    assert segment.risk_forecast_required == (False,) * 6
    assert provider.decided_count == 2
    assert provider.solver_call_count == 0
    assert provider.recorded_solver_call_counts == [0] * 6
    assert provider.optimizer_construction_count == 0
    assert provider.recorded_optimizer_construction_counts == [0] * 6


def test_buffered_inverse_volatility_weighs_the_full_book_or_falls_back_unsupported() -> None:
    """consumes only selected diagonal vols; one bad vol restores whole-book equal weight."""

    from alphalattice.investment.portfolio_strategy_lab.policies import (
        buffered_inverse_volatility as inverse_volatility,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.buffered_equal_weight import (
        whole_book_equal_weight_target,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
        build_installed_portfolio_policy_catalog,
    )

    selected = np.asarray((0, 1, 2), dtype=np.int64)
    eligible = np.asarray((True, True, True, False), dtype=np.bool_)
    reference = np.asarray((0.0, 0.0, 0.0, 0.1), dtype=np.float64)
    covariance = np.diag(np.asarray((1.0, 4.0, 16.0, 9.0), dtype=np.float64))
    allocation = inverse_volatility.whole_book_inverse_volatility_target(
        selected=selected,
        covariance=covariance,
        decision_eligible=eligible,
        reference_weights=reference,
        top_k=3,
    )
    assert allocation.produces_risk_forecast
    target = allocation.target_weights
    assert target[selected] == pytest.approx(np.asarray((0.9 * 4 / 7, 0.9 * 2 / 7, 0.9 / 7)))
    assert target[3] == pytest.approx(0.1)
    invalid_covariance = np.array(covariance, copy=True)
    invalid_covariance[1, 1] = 0.0
    fallback = inverse_volatility.whole_book_inverse_volatility_target(
        selected=selected,
        covariance=invalid_covariance,
        decision_eligible=eligible,
        reference_weights=reference,
        top_k=3,
    )
    assert not fallback.produces_risk_forecast
    fallback_weights = fallback.target_weights
    assert np.flatnonzero(fallback_weights > 0.0).tolist() == [0, 1, 2, 3]
    assert np.allclose(
        fallback_weights,
        whole_book_equal_weight_target(
            selected=selected,
            decision_eligible=eligible,
            reference_weights=reference,
            top_k=3,
        ),
    )
    nan_covariance = np.array(covariance, copy=True)
    nan_covariance[2, 2] = np.nan
    nan_fallback = inverse_volatility.whole_book_inverse_volatility_target(
        selected=selected,
        covariance=nan_covariance,
        decision_eligible=eligible,
        reference_weights=reference,
        top_k=3,
    )
    assert not nan_fallback.produces_risk_forecast
    assert np.allclose(nan_fallback.target_weights, fallback_weights)

    from alphalattice.capabilities.portfolio_backtesting.clocks import (
        WholeBookEveryNFormationsClock,
    )
    from alphalattice.capabilities.portfolio_backtesting.segments import (
        run_portfolio_walk_forward_segment as run_segment,
    )
    from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
        PortfolioPolicyDecisionProvider,
    )

    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(494))
    asset_count = 50
    listing_ids = tuple(f"C{index:03d}" for index in range(asset_count))
    score_matrix = np.tile(
        np.linspace(float(asset_count), 1.0, asset_count),
        (len(sessions), 1),
    )
    segment_covariances = np.tile(np.eye(asset_count), (len(sessions), 1, 1))
    segment_covariances[0, 0, 0] = np.nan
    segment_covariances[3, 1, 1] = 0.0
    decision = np.ones_like(score_matrix, dtype=np.bool_)
    returns = np.zeros_like(score_matrix)
    workspace = _DevelopmentWorkspace(
        mandate=_mandate(sessions),
        formation_sessions=sessions,
        ordered_listing_ids=listing_ids,
        candidate_ids=("candidate",),
        raw_prediction_summaries={
            "candidate": {"finite_fraction": 1.0, "mean": 0.0, "standard_deviation": 1.0}
        },
        scores={("candidate", PortfolioScoreMode.STOCK_ONLY): score_matrix},
        covariances=segment_covariances,
        decision_eligible=decision,
        execution_available=decision,
        realized_simple_returns=returns,
        causal_adv20=np.ones_like(score_matrix),
        sector_ids=("S1",),
        sector_exposure_matrix=np.ones((1, asset_count)),
        equal_weight_sector_exposure=np.array([1.0]),
        workspace_hash=_HASHES[20],
        passive_returns_by_session={
            session: returns[index] for index, session in enumerate(sessions)
        },
    )
    provider = PortfolioPolicyDecisionProvider(
        workspace=workspace,
        candidate_id="candidate",
        score_mode=PortfolioScoreMode.STOCK_ONLY,
        policy=inverse_volatility.WholeBookHysteresisInverseVolatilityRecipe.create(),
        optimizer=None,
        policies=build_installed_portfolio_policy_catalog(),
        record_targets=True,
    )
    segment = run_segment(
        workspace=workspace,
        decision_provider=provider,
        start_index=0,
        stop_index=7,
        rebalance_clock=WholeBookEveryNFormationsClock(interval=3, offset=0),
    )
    assert segment.decision_modes == (
        "REBALANCE",
        "HOLD",
        "HOLD",
        "REBALANCE",
        "HOLD",
        "HOLD",
        "REBALANCE",
    )
    assert segment.predicted_variances[:6] == (None,) * 6
    assert segment.predicted_variances[6] is not None
    assert math.isfinite(segment.predicted_variances[6])
    assert segment.risk_forecast_required == (False, True, True, False, True, True, True)
    assert np.allclose(provider.recorded_targets[0], provider.recorded_targets[3])
    assert np.allclose(provider.recorded_targets[0], np.full(asset_count, 1.0 / asset_count))
    assert np.count_nonzero(provider.recorded_targets[0]) == asset_count
    assert all(value == 0 for value in provider.recorded_solver_call_counts)
    assert provider.solver_call_count == provider.optimizer_construction_count == 0


def test_buffered_rank_return_uses_only_matured_outcomes_and_never_an_optimizer() -> None:
    """curve is exact, causal, per-session averaged, and closed form."""

    from alphalattice.capabilities.portfolio_backtesting.clocks import (
        WholeBookEveryNFormationsClock,
    )
    from alphalattice.capabilities.portfolio_backtesting.segments import (
        run_portfolio_walk_forward_segment as run_segment,
    )
    from alphalattice.capabilities.portfolio_inputs.signed_score.contracts import (
        PortfolioExecutionEvents,
    )
    from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
        PortfolioPolicyDecisionProvider,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import (
        CausalRankReturnCurveSlice,
        WholeBookHysteresisCausalRankMuRecipe,
        build_causal_rank_return_curve,
        causal_rank_mu_maturity_support,
        whole_book_diagonal_rank_mu_target,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
        build_installed_portfolio_policy_catalog,
    )

    def _events(axis: tuple[date, ...], *, delayed_first: bool = False):  # type: ignore[no-untyped-def]
        decision_at = tuple(
            datetime(value.year, value.month, value.day, 16, tzinfo=UTC) for value in axis
        )
        exit_at = tuple(
            datetime(value.year, value.month, value.day, 15, 30, tzinfo=UTC) for value in axis
        )
        if delayed_first:
            exit_at = (
                decision_at[252] + timedelta(microseconds=1),
                *exit_at[1:],
            )
        return PortfolioExecutionEvents.create(
            outcome_snapshot_hash=_HASHES[1],
            outcome_manifest_ref=f"outcomes/{_HASHES[1]}.json",
            method_seal_disposition="METHOD_BOUND",
            method_binding_hash=_HASHES[2],
            execution_recipe_id="NEXT_COMMON_SESSION_OPEN_TO_FOLLOWING_OPEN",
            execution_recipe_hash=_HASHES[3],
            ordered_formation_sessions=axis,
            decision_at=decision_at,
            entry_at=decision_at,
            exit_at=exit_at,
            session_close_at={
                value: instant for value, instant in zip(axis, decision_at, strict=True)
            },
        )

    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(494))
    listing_ids = tuple(f"C{index:03d}" for index in range(100))
    scores = np.tile(np.linspace(100.0, 1.0, 100), (len(sessions), 1))
    returns = np.tile(np.linspace(0.02, -0.02, 100), (len(sessions), 1))
    eligible = np.ones_like(scores, dtype=np.bool_)
    events = _events(sessions)
    curve = build_causal_rank_return_curve(
        formation_sessions=sessions,
        ordered_listing_ids=listing_ids,
        scores=scores,
        score_authority_identity=_HASHES[4],
        decision_eligible=eligible,
        realized_simple_returns=returns,
        execution_events=events,
    )
    maturity = causal_rank_mu_maturity_support(events)
    assert maturity.formation_sessions == tuple(
        session
        for index, session in enumerate(sessions)
        if curve.at(index).disposition == "AVAILABLE"
    )
    assert maturity.formation_sessions[0] == sessions[252]
    available = curve.at(252)
    assert available.disposition == "AVAILABLE"
    assert available.admitted_formation_count == 252
    assert available.bucket_support_counts == (252,) * 20

    partial_returns = np.array(returns, copy=True)
    partial_returns[0, :21] = np.nan
    partial_curve = build_causal_rank_return_curve(
        formation_sessions=sessions,
        ordered_listing_ids=listing_ids,
        scores=scores,
        score_authority_identity=_HASHES[4],
        decision_eligible=eligible,
        realized_simple_returns=partial_returns,
        execution_events=events,
    )
    partial = partial_curve.at(252)
    assert partial.disposition == "PARTIAL_BUCKET_SUPPORT"
    assert partial.bucket_support_counts == (251,) * 20
    assert partial.bucket_means == pytest.approx(available.bucket_means)
    assert partial_curve.curve_hash != curve.curve_hash

    changed_future = np.array(returns, copy=True)
    changed_future[252] = 99.0
    future_curve = build_causal_rank_return_curve(
        formation_sessions=sessions,
        ordered_listing_ids=listing_ids,
        scores=scores,
        score_authority_identity=_HASHES[4],
        decision_eligible=eligible,
        realized_simple_returns=changed_future,
        execution_events=events,
    )
    assert np.array_equal(
        available.bucket_means,
        future_curve.at(252).bucket_means,
        equal_nan=True,
    )
    delayed = build_causal_rank_return_curve(
        formation_sessions=sessions,
        ordered_listing_ids=listing_ids,
        scores=scores,
        score_authority_identity=_HASHES[4],
        decision_eligible=eligible,
        realized_simple_returns=returns,
        execution_events=_events(sessions, delayed_first=True),
    )
    assert delayed.at(252).disposition == "INSUFFICIENT_MATURED_FORMATION_HISTORY"

    short_axis = tuple(date(2025, 1, 2) + timedelta(days=index) for index in range(3))
    short_scores = np.tile(np.asarray((10.0, 9.0, 8.0, 7.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0)), (3, 1))
    short_returns = np.asarray(
        (
            (1.0, 1.0, 1.0, 1.0, 100.0, 5.0, 5.0, 5.0, 5.0, 5.0),
            (3.0, 3.0, 3.0, 3.0, 100.0, 7.0, 9.0, 9.0, 9.0, 9.0),
            (0.0,) * 10,
        ),
        dtype=np.float64,
    )
    short_curve = build_causal_rank_return_curve(
        formation_sessions=short_axis,
        ordered_listing_ids=("L0", "L1", "L2", "L3", "Z", "A", "L6", "L7", "L8", "L9"),
        scores=short_scores,
        score_authority_identity=_HASHES[4],
        decision_eligible=np.ones_like(short_scores, dtype=np.bool_),
        realized_simple_returns=short_returns,
        execution_events=_events(short_axis),
        bucket_count=2,
        lookback=2,
    )
    assert short_curve.at(2).bucket_means == pytest.approx((2.8, 25.6))

    allocation = whole_book_diagonal_rank_mu_target(
        selected=np.asarray((0, 1, 2), dtype=np.int64),
        scores=np.asarray((3.0, 2.0, 1.0, np.nan)),
        covariance=np.diag(np.asarray((1.0, 1.0, 1.0, 1.0))),
        decision_eligible=np.asarray((True, True, True, False)),
        reference_weights=np.asarray((0.0, 0.0, 0.0, 0.1)),
        ordered_listing_ids=("A", "B", "C", "D"),
        curve=CausalRankReturnCurveSlice(
            formation_index=2,
            formation_session=short_axis[2],
            bucket_means=np.asarray((-1.0, 0.0, 1.0)),
            bucket_support_counts=(2, 2, 2),
            admitted_formation_count=2,
            disposition="AVAILABLE",
            curve_hash=_HASHES[5],
        ),
        top_k=3,
        bucket_count=3,
        kappa=0.2,
        maximum_weight=0.45,
    )
    assert allocation.target_weights == pytest.approx((0.1125, 0.3375, 0.45, 0.1))
    assert allocation.target_weights.max() <= 0.45

    workspace = _DevelopmentWorkspace(
        mandate=_mandate(sessions),
        formation_sessions=sessions,
        ordered_listing_ids=listing_ids,
        candidate_ids=("candidate",),
        raw_prediction_summaries={
            "candidate": {"finite_fraction": 1.0, "mean": 0.0, "standard_deviation": 1.0}
        },
        scores={("candidate", PortfolioScoreMode.STOCK_ONLY): scores},
        covariances=np.tile(np.eye(100), (len(sessions), 1, 1)),
        decision_eligible=eligible,
        execution_available=eligible,
        realized_simple_returns=np.zeros_like(scores),
        causal_adv20=np.ones_like(scores),
        sector_ids=("S1",),
        sector_exposure_matrix=np.ones((1, 100)),
        equal_weight_sector_exposure=np.array([1.0]),
        workspace_hash=_HASHES[6],
        passive_returns_by_session={session: np.zeros(100) for session in sessions},
        causal_rank_return_curve=curve,
    )
    provider = PortfolioPolicyDecisionProvider(
        workspace=workspace,
        candidate_id="candidate",
        score_mode=PortfolioScoreMode.STOCK_ONLY,
        policy=WholeBookHysteresisCausalRankMuRecipe.create(),
        optimizer=None,
        policies=build_installed_portfolio_policy_catalog(),
    )
    segment = run_segment(
        workspace=workspace,
        decision_provider=provider,
        start_index=0,
        stop_index=255,
        rebalance_clock=WholeBookEveryNFormationsClock(interval=3, offset=0),
    )
    assert segment.decision_modes[252:] == ("REBALANCE", "HOLD", "HOLD")
    assert segment.predicted_variances[252] is not None
    assert math.isfinite(segment.predicted_variances[252])
    assert segment.risk_forecast_required == (True,) * 255
    assert provider.optimizer_construction_count == provider.solver_call_count == 0


def _trial_metrics_payload(**overrides: object) -> dict[str, object]:
    """A real ``PortfolioTrialMetrics`` payload, produced by ``model_dump``."""

    from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioTrialMetrics

    values: dict[str, object] = {
        "cumulative_gross_log_wealth": 0.1,
        "cumulative_net_log_wealth_5bps": 0.09,
        "cumulative_net_log_wealth_10bps": 0.08,
        "cumulative_net_log_wealth_20bps": 0.07,
        "realized_annualized_volatility": 0.2,
        "maximum_drawdown": 0.1,
        "mean_one_way_turnover": 0.05,
        "mean_hhi": 0.2,
        "mean_holding_count": 4.0,
        "mean_weighted_adv20": 1000.0,
        "minimum_weighted_adv20": 900.0,
        "solver_failure_count": 0,
        "missing_execution_count": 0,
    }
    values.update(overrides)
    return dict(PortfolioTrialMetrics.model_validate(values).model_dump(mode="json"))


def _projected_bar(session: date, *, open_price: float, close_price: float):  # type: ignore[no-untyped-def]
    from alphalattice.foundation.market_data_ops.publication.projection import ProjectedBar

    return ProjectedBar(
        listing_id="AAA",
        session_date=session,
        open_raw=open_price,
        high_raw=max(open_price, close_price),
        low_raw=min(open_price, close_price),
        close_raw=close_price,
        volume_raw=1000,
        cash_dividend=0.0,
        new_shares_per_old_share=1.0,
        split_ratio=1.0,
        open_split_adjusted=open_price,
        high_split_adjusted=max(open_price, close_price),
        low_split_adjusted=min(open_price, close_price),
        close_split_adjusted=close_price,
        volume_split_adjusted=1000.0,
        close_total_return_adjusted=close_price,
        action_set_hash=_HASHES[0],
        anchor_session=session,
    )


def test_the_session_mark_is_a_listing_safe_intraday_fact() -> None:
    """The session mark is a listing safe intraday fact."""

    from alphalattice.foundation.market_data_ops.publication.session_marks import (
        SESSION_MARK_OBSERVED_THROUGH_EVENT,
        SessionMarkError,
        session_mark_matrix,
    )

    sessions = (date(2024, 1, 2), date(2024, 1, 3))
    listings = ("AAA", "BBB")

    def bar(session: date, listing: str, *, open_price: float, close_price: float):  # type: ignore[no-untyped-def]
        value = _projected_bar(session, open_price=open_price, close_price=close_price)
        return replace(value, listing_id=listing)

    bars = [
        bar(sessions[0], "AAA", open_price=100.0, close_price=110.0),
        bar(sessions[0], "BBB", open_price=50.0, close_price=47.5),
        bar(sessions[1], "AAA", open_price=110.0, close_price=104.5),
        bar(sessions[1], "BBB", open_price=47.5, close_price=47.5),
    ]
    marks = session_mark_matrix(bars, sessions=sessions, listing_ids=listings)
    assert marks[0][0] == pytest.approx(0.10)
    assert marks[0][1] == pytest.approx(-0.05)
    assert marks[1][1] == pytest.approx(0.0)
    assert SESSION_MARK_OBSERVED_THROUGH_EVENT == "OFFICIAL_CLOSE"

    # A duplicated row is refused rather than overwritten.
    with pytest.raises(SessionMarkError, match="session_mark_row_duplicated"):
        session_mark_matrix([*bars, bars[0]], sessions=sessions, listing_ids=listings)
    # A missing cell is refused rather than filled: a substituted zero asserts a
    # listing did not move on a day nobody observed, and a quiet day is identical.
    with pytest.raises(SessionMarkError, match="session_mark_absent"):
        session_mark_matrix(bars[:3], sessions=sessions, listing_ids=listings)
    with pytest.raises(SessionMarkError, match="session_mark_price_invalid"):
        session_mark_matrix(
            [bar(sessions[0], "AAA", open_price=0.0, close_price=1.0)],
            sessions=(sessions[0],),
            listing_ids=("AAA",),
        )
    with pytest.raises(SessionMarkError, match="session_mark_axis_duplicated"):
        session_mark_matrix(bars, sessions=sessions, listing_ids=("AAA", "AAA"))


@dataclass(frozen=True, slots=True)
class _MarkLaneWorkspace:
    """The smallest workspace the segment loop accepts, on a named session axis.

    Purpose-built rather than a slice of ``_workspace``: what these tests need is
    a *short* axis whose sessions can carry deliberately different marks, and the
    shared fixture is 494 formations of uniform returns -- exactly the shape in
    which a one-session misalignment is invisible.
    """

    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    execution_available: np.ndarray
    realized_simple_returns: np.ndarray
    causal_adv20: np.ndarray
    sector_exposure_matrix: np.ndarray
    equal_weight_sector_exposure: np.ndarray
    passive_returns_by_session: dict
    mandate: object = None


def _mark_lane_workspace(
    *, sessions: tuple[date, ...], assets: int = 2, realized: np.ndarray | None = None
) -> _MarkLaneWorkspace:
    formations = len(sessions)
    outcomes = np.zeros((formations, assets)) if realized is None else np.asarray(realized)
    return _MarkLaneWorkspace(
        formation_sessions=sessions,
        ordered_listing_ids=tuple(f"L{value}" for value in range(assets)),
        execution_available=np.ones((formations, assets), dtype=np.bool_),
        realized_simple_returns=outcomes,
        causal_adv20=np.full((formations, assets), 1e6),
        sector_exposure_matrix=np.ones((1, assets)),
        equal_weight_sector_exposure=np.asarray([1.0]),
        passive_returns_by_session=dict(zip(sessions, outcomes, strict=True)),
    )


def test_the_close_mark_is_the_route_the_decision_loop_actually_runs() -> None:
    """The close mark is the route the decision loop actually runs."""

    from alphalattice.capabilities.portfolio_backtesting import segments
    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        EXECUTED_BOOK_AT_OPEN_T_PROXY,
        MARKED_TO_MARKET_AT_CLOSE_T,
        PortfolioWalkForwardError,
    )
    from alphalattice.capabilities.portfolio_backtesting.reference_marks import (
        OPEN_PROXY_REFERENCE_MARK_LANE,
        ReferenceMarkLane,
    )

    # 1. A close-mark lane with no marks behind it cannot be constructed at all,
    #    so "sealed close, ran proxy" is not expressible as a lane.
    with pytest.raises(PortfolioWalkForwardError, match="reference_mark_lane_incomplete"):
        ReferenceMarkLane(method=MARKED_TO_MARKET_AT_CLOSE_T)
    with pytest.raises(PortfolioWalkForwardError, match="proxy_carries_marks"):
        ReferenceMarkLane(
            method=EXECUTED_BOOK_AT_OPEN_T_PROXY, marks_by_session={date(2024, 1, 2): (0.0,)}
        )
    assert OPEN_PROXY_REFERENCE_MARK_LANE.method == EXECUTED_BOOK_AT_OPEN_T_PROXY
    assert not OPEN_PROXY_REFERENCE_MARK_LANE.marks_to_close

    # 2. The loop values the reference through the lane and through nothing else.
    #    The old route assigned ``optimizer_reference = executed`` directly; that
    #    line no longer exists, and the source says which call replaced it.
    # And the sequence owner carries the lane, which is the argument the first
    # version of this seam was missing entirely.
    assert (
        "reference_mark"
        in inspect.signature(
            segments.run_portfolio_walk_forward_segment_sequence_with_passive_holds
        ).parameters
    )


def test_the_reference_is_the_book_at_the_decisions_own_open() -> None:
    """The reference is the book at the decision's own open."""

    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        MARKED_TO_MARKET_AT_CLOSE_T,
        PortfolioTargetDecision,
        PortfolioWalkForwardError,
    )
    from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
    from alphalattice.capabilities.portfolio_backtesting.segments import (
        run_portfolio_walk_forward_segment,
    )

    axis = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(3))
    workspace = _mark_lane_workspace(sessions=axis)
    # Session 0 moves the first listing, session 1 moves the second. Nothing
    # else in the fixture distinguishes them.
    marks = {axis[0]: (0.25, 0.0), axis[1]: (0.0, 0.25), axis[2]: (0.0, 0.0)}
    # Every formation fills somewhere, including the last one on the axis, so the
    # projection covers all of them the way the execution snapshot's own column
    # does.
    projection = {value: value + timedelta(days=1) for value in axis}

    def run(lane):  # type: ignore[no-untyped-def]
        seen: list[np.ndarray] = []

        def provider(*, formation_index, reference_weights, pretrade_weights, decision_mode):  # type: ignore[no-untyped-def]
            del pretrade_weights, decision_mode
            seen.append(np.array(reference_weights, copy=True))
            # A two-name book on purpose. A book concentrated in one listing is
            # invariant under *any* mark once renormalized, so it could not tell
            # a correct lane from a one-day-early one.
            return PortfolioTargetDecision(
                target_weights=np.array([0.5, 0.5]), predicted_variance=1e-4
            )

        run_portfolio_walk_forward_segment(
            workspace=workspace,
            decision_provider=provider,
            start_index=0,
            stop_index=3,
            reference_mark=lane,
        )
        return seen

    correct = run(
        ReferenceMarkLane(
            method=MARKED_TO_MARKET_AT_CLOSE_T,
            marks_by_session=marks,
            entry_session_by_formation=projection,
            state_transition_binding_hash=_HASHES[0],
        )
    )
    # Formation 0 opens flat. Formation 1 sees the book filled at open(axis[1]),
    # valued by *axis[1]*'s move -- which lifts the second listing, so the
    # reference tilts towards it.
    assert np.allclose(correct[0], np.zeros(2))
    assert correct[1] == pytest.approx(np.array([0.5, 0.625]) / 1.125)

    # The one-day-early lane: the same marks, shifted by one session. It produces
    # a *different* reference at formation 1, which is the whole point -- the
    # misalignment is observable rather than a rounding change.
    shifted = run(
        ReferenceMarkLane(
            method=MARKED_TO_MARKET_AT_CLOSE_T,
            marks_by_session={axis[0]: (0.0, 0.0), axis[1]: (0.25, 0.0), axis[2]: (0.0, 0.25)},
            entry_session_by_formation=projection,
            state_transition_binding_hash=_HASHES[0],
        )
    )
    assert not np.allclose(correct[1], shifted[1])
    # And it tilts the *other* way, which is what a one-session offset does to
    # a book held across two sessions that moved different names.
    assert shifted[1][0] > shifted[1][1] and correct[1][0] < correct[1][1]

    # A projection whose fills do not land on the next decision's open is refused
    # rather than valued against a book that had not been filled yet.
    with pytest.raises(PortfolioWalkForwardError, match="entry_after_next_decision"):
        run(
            ReferenceMarkLane(
                method=MARKED_TO_MARKET_AT_CLOSE_T,
                marks_by_session=marks,
                entry_session_by_formation={
                    axis[0]: axis[2],
                    axis[1]: axis[2],
                    axis[2]: axis[2],
                },
                state_transition_binding_hash=_HASHES[0],
            )
        )


def test_every_state_carry_edge_refusal_is_reachable_and_named() -> None:
    """Every state carry edge refusal is reachable and named."""

    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        MARKED_TO_MARKET_AT_CLOSE_T,
        PortfolioWalkForwardError,
    )
    from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane

    axis = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(6))

    def lane(projection) -> ReferenceMarkLane:  # type: ignore[no-untyped-def]
        return ReferenceMarkLane(
            method=MARKED_TO_MARKET_AT_CLOSE_T,
            marks_by_session={value: (0.0,) for value in axis},
            entry_session_by_formation=projection,
            state_transition_binding_hash=_HASHES[0],
        )

    contiguous = {value: value + timedelta(days=1) for value in axis}

    # A formation the execution snapshot published no fill for. Refused rather
    # than skipped: a formation with no fill has no book to value.
    partial = {value: contiguous[value] for value in axis[:3]}
    with pytest.raises(PortfolioWalkForwardError, match="entry_session_absent"):
        lane(partial).require_entry_projection(axis)
    with pytest.raises(PortfolioWalkForwardError, match="entry_session_absent"):
        lane(partial).require_carry_segment(axis)
    with pytest.raises(PortfolioWalkForwardError, match="entry_session_absent"):
        lane(partial).require_carry_transition(
            last_decided=axis[4], embargo_session=axis[5], next_decision=axis[5]
        )

    # A method filling after the next decision: no filled book exists at that
    # close, so there is nothing to value at all.
    late = {value: value + timedelta(days=3) for value in axis}
    with pytest.raises(PortfolioWalkForwardError, match="entry_after_next_decision"):
        lane(late).require_carry_segment(axis)

    # A gap: the fill lands before the next decision, which is a lawful axis and
    # a state chain this engine cannot carry.
    gapped = (*axis[:2], *axis[3:])
    with pytest.raises(PortfolioWalkForwardError, match="state_carry_gap_unsupported"):
        lane(contiguous).require_carry_segment(gapped)

    # The embargo edge: both fills, checked from the same column. The contiguous
    # projection satisfies it; either half moved by a session does not.
    lane(contiguous).require_carry_transition(
        last_decided=axis[1], embargo_session=axis[2], next_decision=axis[3]
    )
    with pytest.raises(PortfolioWalkForwardError, match="state_carry_gap_unsupported"):
        lane(contiguous).require_carry_transition(
            last_decided=axis[1], embargo_session=axis[3], next_decision=axis[4]
        )
    with pytest.raises(PortfolioWalkForwardError, match="entry_after_next_decision"):
        lane(contiguous).require_carry_transition(
            last_decided=axis[1], embargo_session=axis[2], next_decision=axis[2]
        )

    # And the session offered as an embargo may not itself be a decision. A split
    # policy that excludes its embargo index makes the two regions adjacent, so
    # the session handed over is the first *validation* formation -- and drifting
    # the book through a formation's own outcome window before that formation is
    # decided on is a hindsight leak, not an embargo. It refuses under its own
    # name rather than under a message about a gap in the axis.
    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        PortfolioTargetDecision,
    )
    from alphalattice.capabilities.portfolio_backtesting.segments import (
        run_portfolio_walk_forward_segment_sequence_with_passive_holds,
    )

    workspace = _mark_lane_workspace(sessions=axis[:4])

    def provider(*, formation_index, reference_weights, pretrade_weights, decision_mode):  # type: ignore[no-untyped-def]
        del formation_index, reference_weights, pretrade_weights, decision_mode
        return PortfolioTargetDecision(target_weights=np.array([0.5, 0.5]), predicted_variance=1e-4)

    with pytest.raises(PortfolioWalkForwardError, match="embargo_session_is_a_decision"):
        run_portfolio_walk_forward_segment_sequence_with_passive_holds(
            workspace=workspace,
            decision_provider=provider,
            ranges=((0, 2), (2, 4)),
            passive_sessions=(axis[2],),
        )
    # The same sequence with a real embargo -- a session between the two regions
    # that nobody decides on -- carries through.
    run_portfolio_walk_forward_segment_sequence_with_passive_holds(
        workspace=_mark_lane_workspace(sessions=axis[:4]),
        decision_provider=provider,
        ranges=((0, 2), (3, 4)),
        passive_sessions=(axis[2],),
    )


def test_the_reference_carries_its_own_cash_through_an_embargo() -> None:
    """The reference carries its own cash through an embargo."""

    from alphalattice.capabilities.portfolio_backtesting.contracts import (
        PortfolioWalkForwardError,
        PortfolioWalkForwardState,
    )
    from alphalattice.capabilities.portfolio_backtesting.state import (
        advance_portfolio_state_without_decision,
        validate_portfolio_state,
    )

    axis = (date(2024, 1, 2),)
    workspace = _mark_lane_workspace(sessions=axis, realized=np.array([[0.10, -0.05]]))
    # A book that is *not* fully invested, whose reference sits at a different
    # instant with its own cash. Both pairs sum to one; neither pair is the other.
    weights = np.array([0.30, 0.20])
    reference = np.array([0.40, 0.10])
    state = PortfolioWalkForwardState(
        pretrade_weights=weights,
        pretrade_cash=0.50,
        optimizer_reference=reference,
        optimizer_reference_cash=0.50,
    )
    passive_hold = advance_portfolio_state_without_decision(
        workspace=workspace, state=state, formation_session=axis[0]
    )
    advanced = passive_hold.final_state
    assert passive_hold.state_mode == "NO_SCORE_PASSIVE_HOLD"
    assert passive_hold.optimizer_call_count == passive_hold.target_decision_count == 0
    validate_portfolio_state(advanced, 2)
    # The book drifted through the embargo formation's own outcome; the reference
    # is the book at the *next* decision's open, which across an untraded session
    # is exactly the book that entered it.
    assert not np.allclose(advanced.pretrade_weights, weights)
    assert np.allclose(advanced.optimizer_reference, weights)
    assert advanced.optimizer_reference_cash == pytest.approx(0.50)

    # And the validator checks the reference against its own cash, so a
    # mismatched pair is refused where it is built rather than several calls on.
    with pytest.raises(PortfolioWalkForwardError, match="segment_state_invalid"):
        validate_portfolio_state(
            PortfolioWalkForwardState(
                pretrade_weights=weights,
                pretrade_cash=0.50,
                optimizer_reference=reference,
                optimizer_reference_cash=0.20,
            ),
            2,
        )


def _publish_marks(root: Path, *, sessions: tuple[date, ...], prices: dict, **overrides):  # type: ignore[no-untyped-def]
    """Publish one mark surface over a named axis, through the real publisher.

    ``prices`` maps a listing to a per-session ``(open, close)`` pair, so a test
    can make two sessions move different listings and see a misalignment rather
    than a rounding change.
    """

    from alphalattice.capabilities.portfolio_inputs.session_marks import (
        resolve_session_mark_availability,
    )
    from alphalattice.foundation.market_data_ops.publication.session_marks import (
        SessionMarkArtifactStore,
        publish_session_mark_surface,
    )

    store = SessionMarkArtifactStore(root)
    fields = {
        "universe_manifest_revision": "revision-1",
        "market_profile_id": "us-current-index-research",
        "corporate_action_identity": "provider-split-adjusted-open-and-period-dividend",
        "source_watermark_hash": _HASHES[2],
        # Resolved at the Feature owner and handed across as one object. Reaching
        # into that catalog from Market Data would close a dependency cycle,
        # since the Feature engine already depends on that package.
        "availability": resolve_session_mark_availability(),
    }
    fields.update(overrides)
    surface = publish_session_mark_surface(
        store=store,
        bars_by_listing={
            listing: [
                replace(
                    _projected_bar(session, open_price=pair[0], close_price=pair[1]),
                    listing_id=listing,
                )
                for session, pair in zip(sessions, pairs, strict=True)
            ]
            for listing, pairs in prices.items()
        },
        sessions=sessions,
        **fields,
    )
    return store, surface


def test_the_mark_surface_publishes_reads_and_reverifies_every_block(tmp_path: Path) -> None:
    """The mark surface publishes reads and reverifies every block."""

    import pyarrow.parquet as pq

    from alphalattice.capabilities.portfolio_backtesting.execution import (
        mark_book_to_session_close,
    )
    from alphalattice.capabilities.portfolio_inputs.session_marks import (
        resolve_session_mark_availability,
    )
    from alphalattice.foundation.market_data_ops.publication.session_marks import (
        SessionMarkError,
    )
    from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
        PortfolioWalkForwardError,
    )

    policy = resolve_session_mark_availability()
    sessions = (date(2024, 1, 2), date(2024, 1, 3))
    store, surface = _publish_marks(
        tmp_path,
        sessions=sessions,
        prices={
            "AAA": ((100.0, 110.0), (110.0, 110.0)),
            "BBB": ((50.0, 50.0), (50.0, 55.0)),
        },
    )

    assert surface.availability_policy_id == policy.policy_id
    assert surface.availability_policy_hash == policy.policy_hash
    assert surface.observed_through_event == "OFFICIAL_CLOSE"
    assert surface.observed_through_offset_sessions == 0
    assert surface.manifest_uri == store.manifest_uri(surface.surface_hash)
    # Strategy-neutral: nothing about a schedule may appear in its identity.
    assert not {"schedule", "entry", "exit", "rebalance"} & {
        name.split("_")[0] for name in type(surface).model_fields
    }
    # The chunk describes its own axes, so an absent cell is distinguishable from
    # a shorter block.
    chunk = surface.chunks[0]
    assert (chunk.session_count, chunk.listing_count, chunk.row_count) == (2, 2, 4)

    # It reads back through its own manifest, on the axis the caller names.
    reloaded = store.load_manifest(surface.surface_hash)
    assert reloaded.surface_hash == surface.surface_hash
    marks = store.read_marks(reloaded, sessions=sessions, listing_ids=("AAA", "BBB"))
    assert marks[0] == pytest.approx((0.10, 0.0))
    assert marks[1] == pytest.approx((0.0, 0.10))

    # And the consumer marks by *label*. The two sessions move different
    # listings, so using the wrong one is visible rather than a rounding change.
    by_session = dict(zip(sessions, marks, strict=True))
    weights = np.array([0.5, 0.5])
    first, _cash = mark_book_to_session_close(
        weights=weights, cash=0.0, marks_by_session=by_session, session=sessions[0]
    )
    second, _cash2 = mark_book_to_session_close(
        weights=weights, cash=0.0, marks_by_session=by_session, session=sessions[1]
    )
    assert first[0] > first[1]
    assert second[1] > second[0]

    # A session the surface never published is refused, never filled.
    with pytest.raises(PortfolioWalkForwardError, match="session_mark_absent_for_session"):
        mark_book_to_session_close(
            weights=weights, cash=0.0, marks_by_session=by_session, session=date(2024, 1, 4)
        )

    # --- what a read refuses -------------------------------------------------
    with pytest.raises(SessionMarkError, match="session_mark_read_outside_surface"):
        store.read_marks(reloaded, sessions=(date(2024, 1, 4),), listing_ids=("AAA",))
    with pytest.raises(SessionMarkError, match="session_mark_listing_not_in_epoch"):
        store.read_marks(reloaded, sessions=sessions, listing_ids=("ZZZ",))
    with pytest.raises(SessionMarkError, match="session_mark_read_axis_invalid"):
        store.read_marks(reloaded, sessions=(sessions[1], sessions[0]), listing_ids=("AAA",))
    with pytest.raises(SessionMarkError, match="session_mark_axis_duplicated"):
        store.read_marks(reloaded, sessions=sessions, listing_ids=("AAA", "AAA"))
    with pytest.raises(SessionMarkError, match="session_mark_surface_absent"):
        store.load_manifest(_HASHES[1])
    # The file answering to one name, sealed under another. Its own validator
    # only proves internal consistency, which a whole re-seal satisfies for free.
    (store.root / "manifests" / f"{_HASHES[1]}.json").write_bytes(
        surface.model_dump_json().encode("utf-8")
    )
    with pytest.raises(SessionMarkError, match="session_mark_surface_not_this_handle"):
        store.load_manifest(_HASHES[1])

    # --- what a *tampered block* refuses, on the way out ---------------------
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    def _row_identity(row: dict[str, object]) -> str:
        # The formula a forger follows: the canonical hash of the row's six columns.
        columns = (
            "session_date",
            "listing_id",
            "intraday_simple_return",
            "open_split_adjusted",
            "close_split_adjusted",
            "action_set_hash",
        )
        return str(canonical_hash({name: row[name] for name in columns}))

    path = store.resolve_chunk(chunk)
    original = pq.read_table(path)
    schema = original.schema.remove_metadata()
    metadata = original.schema.metadata

    def rewrite(replacement, *, stamp: dict | None = None) -> None:  # type: ignore[no-untyped-def]
        # Cast back to the declared schema, because a forger rewriting this file
        # would produce a *valid* file. Leaving Arrow's nullable columns in would
        # be caught by the schema check and never reach the identity checks --
        # which is the half worth testing.
        sealed = replacement.cast(schema)
        pq.write_table(sealed.replace_schema_metadata(stamp or metadata), path, compression="zstd")

    def read_back() -> None:
        store.read_marks(reloaded, sessions=sessions, listing_ids=("AAA", "BBB"))

    # 1. A mark value edited in place, with the original metadata left alone.
    edited = original.column("intraday_simple_return").to_pylist()
    edited[0] = edited[0] + 0.05
    rewrite(original.set_column(2, "intraday_simple_return", pa.array(edited, pa.float64())))
    with pytest.raises(SessionMarkError, match="session_mark_chunk_tampered"):
        read_back()

    # 2. The same edit, made *consistent* with its own row hash so the row
    #    re-derives -- and still not the ratio of the two prices beside it. This
    #    is why the price columns are inside the identity instead of sitting in
    #    the file unprotected: there is no single-field edit left that a reader
    #    accepts, and no self-consistent one either.
    hashes = original.column("row_hash").to_pylist()
    hashes[0] = _row_identity(
        {
            "session_date": original.column("session_date").to_pylist()[0],
            "listing_id": original.column("listing_id").to_pylist()[0],
            "intraday_simple_return": edited[0],
            "open_split_adjusted": original.column("open_split_adjusted").to_pylist()[0],
            "close_split_adjusted": original.column("close_split_adjusted").to_pylist()[0],
            "action_set_hash": original.column("action_set_hash").to_pylist()[0],
        }
    )
    rewrite(
        original.set_column(2, "intraday_simple_return", pa.array(edited, pa.float64())).set_column(
            6, "row_hash", pa.array(hashes, pa.string())
        )
    )
    with pytest.raises(SessionMarkError, match="session_mark_row_not_the_formula"):
        read_back()

    # 3. A permuted listing axis, which a session-keyed store would have served
    #    happily and multiplied the wrong weight by the wrong move.
    listings = original.column("listing_id").to_pylist()
    rewrite(original.set_column(1, "listing_id", pa.array(listings[::-1], pa.string())))
    with pytest.raises(SessionMarkError, match="session_mark_chunk_tampered"):
        read_back()

    # 4. A deleted row. The chunk's declared counts are what makes it visible.
    rewrite(original.slice(0, 3))
    with pytest.raises(SessionMarkError, match="session_mark_chunk_row_count_invalid"):
        read_back()

    # 4b. A *duplicated* key rather than a missing one, re-sealed row by row so
    #     every identity re-derives: the count still matches, both prices still
    #     divide to their own mark, and one listing has silently replaced
    #     another. A store keyed on the session alone would have served this with
    #     no error at all, and one that checked only row identities would too.
    columns = {name: original.column(name).to_pylist() for name in original.schema.names}
    columns["listing_id"] = ["AAA"] * 4
    columns["row_hash"] = [
        _row_identity({name: columns[name][index] for name in original.schema.names[:-1]})
        for index in range(4)
    ]
    rewrite(pa.table(columns, schema=schema))
    with pytest.raises(SessionMarkError, match="session_mark_row_duplicated"):
        read_back()

    # 5. Re-stamped metadata around unchanged rows.
    rewrite(
        original,
        stamp={**{k.decode(): v.decode() for k, v in metadata.items()}, "extra": "stamped"},
    )
    with pytest.raises(SessionMarkError, match="session_mark_chunk_metadata_tampered"):
        read_back()

    # Restored, and the same read that refused five forgeries succeeds.
    rewrite(original)
    assert store.read_marks(reloaded, sessions=sessions, listing_ids=("AAA", "BBB")) == marks


def test_the_mark_is_admitted_as_a_decision_input_at_its_owners(tmp_path: Path) -> None:
    """The mark is admitted as a decision input at its owners."""

    from alphalattice.capabilities.causal_inputs.contracts import TemporalAdmissionError
    from alphalattice.capabilities.portfolio_inputs import session_marks as adapter

    sessions = (date(2024, 1, 2), date(2024, 1, 3))
    _store, surface = _publish_marks(
        tmp_path,
        sessions=sessions,
        prices={"AAA": ((100.0, 101.0), (101.0, 101.0))},
    )
    authority = adapter.session_mark_input_authority(surface)
    assert authority.input_id == "session_mark"
    assert authority.temporal_usage == "DECISION_INPUT"
    assert authority.surface_hash == surface.surface_hash
    # observed_through is the exchange fact the surface sealed: this session's
    # own close, at offset zero.
    assert authority.observed_through.basis == "EXCHANGE_PUBLISHED_FACT"
    assert authority.observed_through.anchor.offset_sessions == 0
    assert authority.observed_through.anchor.event == "OFFICIAL_CLOSE"
    # source_available is installed policy and says so, at the delay the owner
    # declares rather than at a number restated here.
    installed = adapter.resolve_session_mark_availability()
    assert authority.source_available.basis == "INSTALLED_SOURCE_AVAILABILITY_POLICY"
    assert authority.source_available.policy_id == installed.policy_id
    assert authority.source_available.anchor.offset_sessions == installed.publication_delay_sessions
    # A mark is two published prices divided, not a fit, so there is no
    # computation interval to declare -- and inventing one would put a fictional
    # number inside the deadline comparison.
    assert authority.derived_ready is None

    # A surface naming an installed policy under a hash its owner does not agree
    # with is refused at the adapter, before admission compares anything.
    forged = type(surface).create(
        **{
            **surface.model_dump(exclude={"surface_hash"}),
            "availability_policy_hash": _HASHES[4],
        }
    )
    with pytest.raises(TemporalAdmissionError, match="availability_policy_not_this_build"):
        adapter.session_mark_input_authority(forged)
    unknown = type(surface).create(
        **{
            **surface.model_dump(exclude={"surface_hash"}),
            "availability_policy_id": "feature-availability.not-installed",
        }
    )
    with pytest.raises(TemporalAdmissionError, match="availability_owner_not_this_build"):
        adapter.session_mark_input_authority(unknown)


def test_the_entry_session_projection_comes_from_the_snapshot(tmp_path: Path) -> None:
    """requirement: formation -> entry session is read, never inferred."""

    from alphalattice.capabilities.causal_inputs.contracts import TemporalAdmissionError
    from alphalattice.capabilities.causal_inputs.schedules import entry_session_by_formation

    axis = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(6))
    table = pa.table(
        {
            "formation_session": list(axis[:3]),
            "entry_session": [axis[index + 1] for index in range(3)],
        }
    )
    projection = entry_session_by_formation(table)
    # The installed next-open method fills the session after the formation, and
    # the projection says so from the rows rather than from an offset.
    assert projection == {value: axis[axis.index(value) + 1] for value in axis[:3]}

    del tmp_path
    contradictory = pa.table(
        {
            "formation_session": [axis[0], axis[0]],
            "entry_session": [axis[1], axis[2]],
        }
    )
    with pytest.raises(TemporalAdmissionError, match="entry_session_disagrees"):
        entry_session_by_formation(contradictory)
    backwards = pa.table({"formation_session": [axis[2]], "entry_session": [axis[1]]})
    with pytest.raises(TemporalAdmissionError, match="entry_not_after_formation"):
        entry_session_by_formation(backwards)


def _delayed_availability_catalog(sessions_late: int):  # type: ignore[no-untyped-def]
    """The installed catalog with the price feed publishing ``n`` sessions late.

    A change made at the *owner*, which is the only place it can lawfully be
    made. The surface does not move: the fact is still the intraday move of one
    session, complete at that session's close. What moves is when a strategy may
    use it -- which is the whole point of separating the two, and the reason this
    forgery has to be built by editing the catalog rather than by editing a
    number on the artifact.
    """

    from alphalattice.foundation.feature_engine.catalog.observation_clock import (
        FeatureAvailabilityPolicy,
        SourceAvailabilityCatalog,
        installed_source_availability_catalog,
    )

    installed = installed_source_availability_catalog()
    owners = tuple(
        FeatureAvailabilityPolicy.create(
            **{
                **owner.model_dump(exclude={"policy_hash"}),
                "publication_delay_sessions": sessions_late,
            }
        )
        if owner.policy_id == installed.field_authorities["open_split_adjusted"]
        else owner
        for owner in installed.owners
    )
    return SourceAvailabilityCatalog.create(
        catalog_id=installed.catalog_id,
        owners=owners,
        field_authorities=dict(installed.field_authorities),
    )
