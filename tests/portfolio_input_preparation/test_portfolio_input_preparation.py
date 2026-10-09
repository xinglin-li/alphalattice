from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.capabilities.portfolio_inputs.tradability.contracts import (
    HistoricalDecisionTradabilitySurface,
    HistoricalExecutionAvailabilitySurface,
    HistoricalTradabilityBundle,
    TradabilitySurfaceChunk,
    seal_tradability_contract,
    tradability_chunk_identity,
)
from alphalattice.capabilities.portfolio_inputs.tradability.publication import (
    CurrentTradabilityDataService,
)
from alphalattice.capabilities.portfolio_inputs.tradability.readback import (
    read_tradability_matrices,
)
from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    HistoricalTradabilityBuilder,
    PortfolioTradabilityRiskInputs,
    PublishedTradabilitySurfaces,
    TradabilityArtifactStore,
    TradabilitySurfaceError,
    _decision_row,
    _decision_schema,
    _execution_row,
    _execution_schema,
    _TradabilityBuildInputs,
    decision_eligible_at_close,
)
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_H = "a" * 64


def _bar(session: date, *, listing_id: str = "A", close: float = 10.0) -> RawDailyBar:
    return RawDailyBar(
        listing_id=listing_id,
        provider="fixture",
        session_date=session,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100,
    )


def _admission_inputs(
    *, complete_alpha: bool, evaluated_at: datetime | None = None
) -> dict[str, Any]:
    formations = (date(2026, 8, 7), date(2026, 8, 8))
    listings = ("A", "B")
    risk_surface = SimpleNamespace(
        chunks=(SimpleNamespace(formation_sessions=formations),),
        formation_count=2,
        surface_hash="b" * 64,
        epoch_hash="c" * 64,
        recipe=SimpleNamespace(output_unit="one-session-open-to-open-log-return-covariance"),
    )
    epoch = SimpleNamespace(epoch_hash="c" * 64, ordered_listing_ids=listings)
    alpha_scores = SimpleNamespace(
        surface_hash="d" * 64,
        program_hash="e" * 64,
        foundation_hash="f" * 64,
        candidate_set_snapshot_hash="1" * 64,
        risk_surface_hash=risk_surface.surface_hash,
        universe_epoch_hash=epoch.epoch_hash,
        ordered_listing_ids_hash=canonical_hash(listings),
        risk_covered_formation_count=2 if complete_alpha else 1,
        risk_missing_formation_count=0 if complete_alpha else 1,
        complete_for_risk_axis=complete_alpha,
        limitations=("Current-universe research only.",),
    )
    decision = SimpleNamespace(
        universe_epoch_hash=epoch.epoch_hash,
        ordered_listing_ids=listings,
        formation_sessions=formations,
        limitations=("Daily-bar evidence.",),
    )
    execution = SimpleNamespace(
        universe_epoch_hash=epoch.epoch_hash,
        ordered_listing_ids=listings,
        formation_sessions=formations,
    )
    return {
        "alpha_program": SimpleNamespace(
            foundation_hash="f" * 64,
            pm_plan_hash="2" * 64,
            program_hash="e" * 64,
        ),
        "candidate_set": SimpleNamespace(
            program_hash="e" * 64,
            foundation_hash="f" * 64,
            snapshot_hash="1" * 64,
            ordered_candidate_ids=("ridge", "lasso", "elastic-net"),
        ),
        "alpha_scores": alpha_scores,
        "risk": SimpleNamespace(
            projection=SimpleNamespace(
                status="RISK_RUNTIME_READY_PORTFOLIO_INPUT_PENDING",
                all_hard_gates_passed=True,
                limitations=("Risk is current-universe research only.",),
            ),
            marker=SimpleNamespace(marker_hash="3" * 64),
        ),
        "risk_lineage": SimpleNamespace(
            covariance_surface=risk_surface,
            return_surface=SimpleNamespace(epoch=epoch),
        ),
        "tradability": SimpleNamespace(
            projection=SimpleNamespace(status="DATA_TRADABILITY_READY"),
            marker=SimpleNamespace(marker_hash="4" * 64),
        ),
        "tradability_lineage": SimpleNamespace(
            decision=decision,
            execution=execution,
            bundle=SimpleNamespace(bundle_hash="5" * 64),
        ),
        "guardian": None,
        "evaluated_at": evaluated_at or datetime(2026, 8, 10, tzinfo=UTC),
    }


def test_decision_and_execution_surfaces_are_strictly_separate() -> None:
    assert "causal_adv20" in _decision_schema().names
    assert "execution_status" not in _decision_schema().names
    assert "execution_status" in _execution_schema().names
    assert "causal_adv20" not in _execution_schema().names
    assert _decision_schema().field("formation_session").type == pa.date32()
    assert _execution_schema().field("intended_execution_session").type == pa.date32()


def test_decision_row_uses_only_formation_and_prior_sessions() -> None:
    formation = date(2026, 8, 10)
    history = tuple(formation - timedelta(days=value) for value in reversed(range(20)))
    bars = {
        (session, "A"): _bar(session, close=10.0 + index) for index, session in enumerate(history)
    }
    row = _decision_row(
        listing_id="A",
        formation_session=formation,
        intended_execution_session=formation + timedelta(days=1),
        history=history,
        bars=bars,
    )
    assert row["decision_status"] == "PLANNED_ORDER_ELIGIBLE"
    assert row["observed_through"] == formation
    assert row["causal_adv20"] == sum((10.0 + index) * 100 for index in range(20)) / 20


def test_decision_eligibility_is_the_decision_row_status() -> None:
    """Decision eligibility is the decision row status."""

    formation = date(2026, 8, 10)
    history = tuple(formation - timedelta(days=value) for value in reversed(range(20)))
    complete = {(session, "A"): _bar(session) for session in history}
    incomplete = {key: bar for key, bar in complete.items() if key[0] != history[3]}
    unusable = {**complete, (formation, "A"): _bar(formation, close=0.0)}
    for bars in (complete, incomplete, unusable):
        row = _decision_row(
            listing_id="A",
            formation_session=formation,
            intended_execution_session=None,
            history=history,
            bars=bars,
        )
        assert decision_eligible_at_close(
            listing_id="A", formation_session=formation, history=history, bars=bars
        ) is (row["decision_status"] == "PLANNED_ORDER_ELIGIBLE")
    assert not decision_eligible_at_close(
        listing_id="A", formation_session=formation, history=history[1:], bars=complete
    )


def test_execution_row_is_post_session_evidence() -> None:
    formation = date(2026, 8, 10)
    execution = formation + timedelta(days=1)
    present = _execution_row(
        listing_id="A",
        formation_session=formation,
        intended_execution_session=execution,
        bar=_bar(execution),
    )
    absent = _execution_row(
        listing_id="A",
        formation_session=formation,
        intended_execution_session=execution,
        bar=None,
    )
    assert present["observed_through"] == execution
    assert present["observation_method"] == "POST_SESSION_DAILY_BAR"
    assert absent["observation_method"] == "POST_SESSION_DAILY_BAR_ABSENCE"


def test_current_tradability_reuse_binds_schedule_and_source_watermark() -> None:
    formation = date(2026, 8, 10)
    execution = date(2026, 8, 11)
    builder = HistoricalTradabilityBuilder.__new__(HistoricalTradabilityBuilder)
    cached = dict(
        epoch_hash="c" * 64,
        formations=(formation,),
        available_sessions=(formation, execution),
        session_positions={formation: 0, execution: 1},
        intended_execution_sessions=(execution,),
        schedule_hash="d" * 64,
        ordered_listing_ids=("A",),
        source_watermark_hash="e" * 64,
    )
    decision = SimpleNamespace(
        universe_epoch_hash="c" * 64,
        ordered_listing_ids=("A",),
        source_watermark_hash="e" * 64,
        schedule_hash="d" * 64,
        formation_sessions=(formation,),
        intended_execution_sessions=(execution,),
    )
    execution_surface = SimpleNamespace(**vars(decision))
    bundle = SimpleNamespace(universe_epoch_hash="c" * 64, schedule_hash="d" * 64)
    arguments: dict[str, Any] = {
        "risk_inputs": PortfolioTradabilityRiskInputs(
            return_surface_hash="a" * 64,
            covariance_surface_hash="b" * 64,
            epoch_hash="c" * 64,
            market_profile_id="us-current-index-research",
            ordered_listing_ids=("A",),
            universe_manifest_revision="f" * 64,
            formation_sessions=(formation,),
            formation_count=1,
            asset_count=1,
            available_return_sessions=(formation, execution),
        ),
        "decision": decision,
        "execution": execution_surface,
        "bundle": bundle,
    }
    builder._inputs = _TradabilityBuildInputs(
        authority_hash=canonical_hash(asdict(arguments["risk_inputs"])), **cached
    )
    assert builder.current_surfaces_match(**arguments)
    stale_schedule = SimpleNamespace(**{**vars(decision), "schedule_hash": "f" * 64})
    assert not builder.current_surfaces_match(**{**arguments, "decision": stale_schedule})
    stale_source = SimpleNamespace(**{**vars(decision), "source_watermark_hash": "f" * 64})
    assert not builder.current_surfaces_match(**{**arguments, "execution": stale_source})


@pytest.mark.parametrize("historical_coverage", [False, True])
def test_market_only_tradability_has_identical_outputs_to_the_risk_projection(
    tmp_path, monkeypatch, historical_coverage
):
    from alphalattice.capabilities.portfolio_inputs.tradability import surface

    sessions = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(25))
    manifest = SimpleNamespace(
        revision_sha256="a" * 64,
        profile=SimpleNamespace(market_profile_id="fixture"),
        listings=(SimpleNamespace(listing_id="B" if historical_coverage else "A"),),
    )
    requested = []

    def load(revision):
        assert revision == manifest.revision_sha256
        requested.append(revision)
        return manifest

    def scope(selected, *, listing_ids):
        assert selected is manifest and listing_ids == ("A",)
        return (SimpleNamespace(listing_id="A"),)

    market = SimpleNamespace(
        load_universe_manifest_revision=load,
        listing_scope=scope,
        execution_source_watermark=lambda *a, **k: {"watermark_hash": "b" * 64},
    )
    monkeypatch.setattr(
        surface,
        "_read_raw_bar_table",
        lambda *a, **k: pa.Table.from_pylist([asdict(_bar(s)) for s in sessions]),
    )
    risk = PortfolioTradabilityRiskInputs(
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "fixture",
        ("A",),
        manifest.revision_sha256,
        sessions[19:23],
        4,
        1,
        sessions,
    )
    neutral = surface.PortfolioTradabilityMarketInputs(
        "f" * 64,
        risk.epoch_hash,
        risk.market_profile_id,
        risk.ordered_listing_ids,
        risk.universe_manifest_revision,
        risk.formation_sessions,
        risk.available_return_sessions,
    )
    old = HistoricalTradabilityBuilder(market_store=market, artifact_root=tmp_path / "risk").build(
        risk_inputs=risk
    )
    current = HistoricalTradabilityBuilder(
        market_store=market, artifact_root=tmp_path / "market"
    ).build(market_inputs=neutral)
    assert old.decision == current.decision
    assert old.execution == current.execution
    assert old.bundle == current.bundle
    assert len(requested) == 2


def test_a_built_surface_states_every_row_as_the_cell_rule_states_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built surface states every row as the cell rule states it."""

    import pyarrow.parquet as pq

    from alphalattice.capabilities.portfolio_inputs.tradability import surface

    sessions = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(32))
    formations = sessions[19:29]
    listings = ("A", "B", "C")
    bars: list[RawDailyBar] = []
    for index, session in enumerate(sessions):
        bars.append(_bar(session, listing_id="A", close=10.0 + index))
        if session != sessions[5]:  # one bar absent inside B's early windows
            bars.append(_bar(session, listing_id="B", close=20.0 + index / 7))
        if session != sessions[27]:  # C has no bar on one execution session
            volume = 0 if session == sessions[23] else 100  # and one bar without volume
            bars.append(
                RawDailyBar(
                    listing_id="C",
                    provider="fixture",
                    session_date=session,
                    open=30.0 + index / 3,
                    high=31.0,
                    low=29.0,
                    close=30.5 + index / 3,
                    volume=volume,
                )
            )
    table = pa.Table.from_pylist([asdict(bar) for bar in bars])
    monkeypatch.setattr(surface, "_read_raw_bar_table", lambda *a, **k: table)
    manifest = SimpleNamespace(
        revision_sha256="a" * 64,
        profile=SimpleNamespace(market_profile_id="fixture"),
        listings=tuple(SimpleNamespace(listing_id=v) for v in listings),
    )
    market = SimpleNamespace(
        load_universe_manifest_revision=lambda revision: manifest,
        listing_scope=lambda selected, *, listing_ids: manifest.listings,
        execution_source_watermark=lambda *a, **k: {"watermark_hash": "b" * 64},
    )
    inputs = surface.PortfolioTradabilityMarketInputs(
        "f" * 64, "d" * 64, "fixture", listings, manifest.revision_sha256, formations, sessions
    )
    builder = HistoricalTradabilityBuilder(market_store=market, artifact_root=tmp_path)
    published = builder.build(market_inputs=inputs)
    positions = {session: index for index, session in enumerate(sessions)}
    bar_map = surface._bar_map(table, listings)
    compared = 0
    for kind, chunks in (
        ("DECISION", published.decision.chunks),
        ("EXECUTION", published.execution.chunks),
    ):
        for chunk in chunks:
            for row in pq.read_table(builder.artifacts.resolve_chunk(chunk)).to_pylist():
                formation = row["formation_session"]
                execution = sessions[positions[formation] + 1]
                if kind == "DECISION":
                    expected = _decision_row(
                        listing_id=row["listing_id"],
                        formation_session=formation,
                        intended_execution_session=execution,
                        history=sessions[positions[formation] - 19 : positions[formation] + 1],
                        bars=bar_map,
                    )
                else:
                    expected = _execution_row(
                        listing_id=row["listing_id"],
                        formation_session=formation,
                        intended_execution_session=execution,
                        bar=bar_map.get((execution, row["listing_id"])),
                    )
                assert row == expected, (kind, row["listing_id"], formation)
                compared += 1
    assert compared == 2 * len(formations) * len(listings)
    statuses = {
        (row["listing_id"], row["formation_session"]): row["reason_code"]
        for chunk in published.decision.chunks
        for row in pq.read_table(builder.artifacts.resolve_chunk(chunk)).to_pylist()
    }
    assert statuses[("A", formations[0])] == "CAUSAL_MARKET_DATA_COMPLETE"
    assert statuses[("B", formations[0])] == "ADV20_HISTORY_INCOMPLETE"
    assert statuses[("C", sessions[23])] == "FORMATION_MARKET_DATA_UNAVAILABLE"
    assert published.execution.execution_status_counts.get("NO_OFFICIAL_OPEN") == 1


def test_chunk_identity_has_one_owner_and_payload_tamper_fails(tmp_path: Path) -> None:
    table = pa.table(
        {
            "formation_session": [date(2026, 8, 10)],
            "intended_execution_session": [date(2026, 8, 11)],
            "listing_id": ["A"],
            "decision_status": ["PLANNED_ORDER_ELIGIBLE"],
            "reason_code": ["CAUSAL_MARKET_DATA_COMPLETE"],
            "causal_adv20": [1000.0],
            "observed_through": [date(2026, 8, 10)],
            "source_row_hash": [_H],
            "row_hash": [_H],
        },
        schema=_decision_schema(),
    )
    store = TradabilityArtifactStore(tmp_path)
    chunk = store.publish_chunk(surface_kind="DECISION", table=table)
    identity = tradability_chunk_identity(
        surface_kind=chunk.surface_kind,
        formation_sessions=chunk.formation_sessions,
        row_count=chunk.row_count,
        payload_sha256=chunk.payload_sha256,
        metadata_hash=chunk.metadata_hash,
    )
    assert chunk == TradabilitySurfaceChunk.model_validate(chunk.model_dump())
    assert canonical_hash(identity) == chunk.content_hash
    path = store.resolve_chunk(chunk)
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 1
    path.write_bytes(content)
    with pytest.raises(TradabilitySurfaceError, match="chunk_tampered"):
        store.resolve_chunk(chunk)


def test_public_readback_filters_owner_chunks_into_exact_matrices(tmp_path: Path) -> None:
    sessions = (date(2026, 8, 10), date(2026, 8, 11))
    execution_sessions = (date(2026, 8, 11), date(2026, 8, 12))
    listings = ("A", "B")
    store = TradabilityArtifactStore(tmp_path)
    decision_chunk = store.publish_chunk(
        surface_kind="DECISION",
        table=pa.table(
            {
                "formation_session": [sessions[0], sessions[0], sessions[1], sessions[1]],
                "intended_execution_session": [
                    execution_sessions[0],
                    execution_sessions[0],
                    execution_sessions[1],
                    execution_sessions[1],
                ],
                "listing_id": ["A", "B", "A", "B"],
                "decision_status": [
                    "PLANNED_ORDER_ELIGIBLE",
                    "PLANNED_ORDER_UNAVAILABLE",
                    "PLANNED_ORDER_ELIGIBLE",
                    "PLANNED_ORDER_ELIGIBLE",
                ],
                "reason_code": ["FIXTURE"] * 4,
                "causal_adv20": [100.0, 200.0, 300.0, 400.0],
                "observed_through": [sessions[0]] * 4,
                "source_row_hash": [_H] * 4,
                "row_hash": [_H] * 4,
            },
            schema=_decision_schema(),
        ),
    )
    execution_chunk = store.publish_chunk(
        surface_kind="EXECUTION",
        table=pa.table(
            {
                "formation_session": [sessions[0], sessions[0], sessions[1], sessions[1]],
                "intended_execution_session": [
                    execution_sessions[0],
                    execution_sessions[0],
                    execution_sessions[1],
                    execution_sessions[1],
                ],
                "listing_id": ["A", "B", "A", "B"],
                "execution_status": [
                    "VERIFIED_ELIGIBLE",
                    "UNAVAILABLE",
                    "ASSUMED_ELIGIBLE_FROM_DAILY_BAR",
                    "VERIFIED_ELIGIBLE",
                ],
                "observation_method": ["FIXTURE"] * 4,
                "observed_through": [sessions[1]] * 4,
                "source_row_hash": [_H] * 4,
                "row_hash": [_H] * 4,
            },
            schema=_execution_schema(),
        ),
    )
    common = {
        "universe_epoch_hash": _H,
        "ordered_listing_ids": listings,
        "source_watermark_hash": _H,
        "schedule_hash": _H,
        "formation_sessions": sessions,
        "intended_execution_sessions": execution_sessions,
        "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        "limitations": ("fixture",),
    }
    decision = seal_tradability_contract(
        HistoricalDecisionTradabilitySurface,
        "surface_hash",
        **common,
        chunks=(decision_chunk,),
        eligible_row_count=3,
        unavailable_row_count=1,
    )
    execution = seal_tradability_contract(
        HistoricalExecutionAvailabilitySurface,
        "surface_hash",
        **common,
        chunks=(execution_chunk,),
        execution_status_counts={
            "VERIFIED_ELIGIBLE": 2,
            "ASSUMED_ELIGIBLE_FROM_DAILY_BAR": 1,
            "UNAVAILABLE": 1,
        },
    )

    decision_values, execution_values, adv20 = read_tradability_matrices(
        store=store,
        decision=decision,
        execution=execution,
        sessions=sessions,
        listings=listings,
    )
    assert decision_values.tolist() == [[True, False], [True, True]]
    assert execution_values.tolist() == [[True, False], [True, True]]
    assert adv20 == pytest.approx(np.array([[100.0, 200.0], [300.0, 400.0]]))


def test_current_tradability_reuses_lineage_across_publish_times(
    tmp_path: Path,
) -> None:
    formation = date(2026, 8, 10)
    execution = date(2026, 8, 11)
    store = TradabilityArtifactStore(tmp_path)
    decision_chunk = store.publish_chunk(
        surface_kind="DECISION",
        table=pa.table(
            {
                "formation_session": [formation],
                "intended_execution_session": [execution],
                "listing_id": ["A"],
                "decision_status": ["PLANNED_ORDER_ELIGIBLE"],
                "reason_code": ["CAUSAL_MARKET_DATA_COMPLETE"],
                "causal_adv20": [1000.0],
                "observed_through": [formation],
                "source_row_hash": [_H],
                "row_hash": [_H],
            },
            schema=_decision_schema(),
        ),
    )
    execution_chunk = store.publish_chunk(
        surface_kind="EXECUTION",
        table=pa.table(
            {
                "formation_session": [formation],
                "intended_execution_session": [execution],
                "listing_id": ["A"],
                "execution_status": ["AVAILABLE"],
                "observation_method": ["POST_SESSION_DAILY_BAR"],
                "observed_through": [execution],
                "source_row_hash": [_H],
                "row_hash": [_H],
            },
            schema=_execution_schema(),
        ),
    )
    common = {
        "universe_epoch_hash": _H,
        "ordered_listing_ids": ("A",),
        "source_watermark_hash": _H,
        "schedule_hash": _H,
        "formation_sessions": (formation,),
        "intended_execution_sessions": (execution,),
        "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        "limitations": ("fixture",),
    }
    decision = seal_tradability_contract(
        HistoricalDecisionTradabilitySurface,
        "surface_hash",
        **common,
        chunks=(decision_chunk,),
        eligible_row_count=1,
        unavailable_row_count=0,
    )
    execution_surface = seal_tradability_contract(
        HistoricalExecutionAvailabilitySurface,
        "surface_hash",
        **common,
        chunks=(execution_chunk,),
        execution_status_counts={"AVAILABLE": 1},
    )
    bundle = seal_tradability_contract(
        HistoricalTradabilityBundle,
        "bundle_hash",
        universe_epoch_hash=_H,
        schedule_hash=_H,
        decision_surface_hash=decision.surface_hash,
        execution_surface_hash=execution_surface.surface_hash,
        formation_count=1,
        asset_count=1,
        coverage_start=formation,
        coverage_end=formation,
    )
    decision_artifact = store.publish_json(
        category="decision/manifests",
        payload=decision.model_dump(mode="json"),
        identity_field="surface_hash",
    )
    execution_artifact = store.publish_json(
        category="execution/manifests",
        payload=execution_surface.model_dump(mode="json"),
        identity_field="surface_hash",
    )
    bundle_artifact = store.publish_json(
        category="bundles",
        payload=bundle.model_dump(mode="json"),
        identity_field="bundle_hash",
    )
    surfaces = PublishedTradabilitySurfaces(
        decision=decision,
        decision_artifact=decision_artifact,
        execution=execution_surface,
        execution_artifact=execution_artifact,
        bundle=bundle,
        bundle_artifact=bundle_artifact,
    )
    service = CurrentTradabilityDataService(tmp_path)
    first = service.publish(
        surfaces=surfaces,
        published_at=datetime(2026, 8, 10, tzinfo=UTC),
    )
    exact = service.publish(
        surfaces=surfaces,
        published_at=datetime(2026, 8, 11, tzinfo=UTC),
    )
    assert exact.action == "REUSED_EXACT"
    assert exact.pointer == first.pointer
