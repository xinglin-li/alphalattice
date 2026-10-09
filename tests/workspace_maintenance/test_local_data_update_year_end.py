"""Three consecutive days of local data update through the page's routes,
across the year boundary, offline."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.maintenance.data_update import bind_existing_data_workspace
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import (
    InstalledAgent,
    _json,
    _manifest,
    _resolved,
    _Resolver,
)
from tests.researcher_methodology_surface.real_workspace import (
    SYMBOLS,
    build_real_risk_workspace,
)
from tests.workspace_maintenance.data_update_support import (
    _active_panel_manifest,
    _seed_foundation,
    set_parallel_test_budget,
)
from tests.workspace_maintenance.local_data_provider import (
    recording_provider,
    unchanged_membership_source,
)

# --------------------------------------------------------------------------
# Three consecutive days through the page's routes: a clean append that
# reuses every closed partition, a restart, and a historical correction that
# crosses the year boundary under the product's own approval.

YEAR_END_AS_OF = date(2026, 1, 5)


YEAR_END_OBSERVED_AT = datetime(2026, 1, 6, tzinfo=UTC)


YEAR_END_HISTORY_START = date(2016, 1, 5)
"""The ten-calendar-year anniversary of the backdated as-of, a trading day.

The seeded walk must answer the whole ten-year window, and the onboarding
fetch must start at the walk's first session: the recording provider the
updates use stamps volumes by session ordinal, which agrees with the
onboarding's window-relative stamp only when the two starts coincide.
"""


DAY_ONE = datetime(2026, 1, 6, 23, tzinfo=UTC)


DAY_TWO = datetime(2026, 1, 7, 23, tzinfo=UTC)


CORRECTED_SESSION = date(2025, 12, 30)
"""Inside the 45-day refresh overlap of the day-two fetch and in the previous year."""


@pytest.fixture(scope="module")
def year_end(tmp_path_factory):
    """The ten-name workspace with its Panel closing on 2026-01-05.

    Built fresh through the product's own onboarding gate and maintenance
    coordinator: the golden cache keys on the source tree and the universe,
    not on the as-of, so a backdated baseline must bypass it. A deterministic
    seeded-walk Provider, explicitly not live data.
    """

    from tests.researcher_methodology_surface import real_workspace as owner

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "AS_OF", YEAR_END_AS_OF)
        patch.setattr(owner, "OBSERVED_AT", YEAR_END_OBSERVED_AT)
        patch.setattr(owner, "HISTORY_START", YEAR_END_HISTORY_START)
        built = build_real_risk_workspace(
            tmp_path_factory.mktemp("data-update-year-end"), fresh=True
        )
        publish_research_workspace_manifest(built.workspace, _manifest("data-update-qa"))
        _seed_foundation(built.workspace, built.panel_snapshot_hash)
        set_parallel_test_budget(built.workspace)
    return built.workspace


def _chunk_file(workspace: Path, chunk: dict) -> Path:
    return workspace / "artifacts" / "feature-panel" / "chunks" / f"{chunk['chunk_hash']}.parquet"


def _row_hashes(workspace: Path, chunk: dict) -> dict[tuple[date, str], str]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        _chunk_file(workspace, chunk), columns=["session_date", "listing_id", "row_hash"]
    )
    return {
        (session, listing): row_hash
        for session, listing, row_hash in zip(
            table.column("session_date").to_pylist(),
            table.column("listing_id").to_pylist(),
            table.column("row_hash").to_pylist(),
            strict=True,
        )
    }


def _lifecycles(workspace: Path) -> dict[str, tuple[str, str | None]]:
    """Every snapshot's lifecycle as the database says it and as the projection says it."""

    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository

    market = MarketDataRepository(workspace)
    resolver = ArtifactResolver(workspace / "artifacts")
    panel_state = PanelStateRepository(market.database, market_data=market)
    result = {}
    for item in panel_state.feature_panel_snapshot_lifecycles():
        projected = resolver.feature_panel_snapshot_lifecycle(
            resolver.feature_panel_manifest_uri(str(item["snapshot_hash"]))
        )
        assert (projected["lifecycle"], projected["reason"]) == (item["lifecycle"], item["reason"])
        result[str(item["snapshot_hash"])] = (str(item["lifecycle"]), item["reason"])
    return result


def _capture_panel_closure(workspace: Path):
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.panels.closure import PanelClosurePublisher
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.closure_source import (
        PanelClosureSourceRepository,
    )

    resolver = ArtifactResolver(workspace / "artifacts")
    store = PanelClosureArtifactStore(resolver)
    source = PanelClosureSourceRepository(
        database_path=workspace / "market-data.duckdb", resolver=resolver
    )
    publisher = PanelClosurePublisher(resolver=resolver, source=source, store=store)
    return publisher.publish(), store, resolver


def test_three_day_journey_reuses_partitions_and_corrects_across_the_year_boundary(
    year_end, tmp_path, monkeypatch
):
    """Three day journey reuses partitions and corrects across the year boundary."""

    from alphalattice.foundation.feature_engine.panels.closure_contracts import (
        PanelDerivationRecipe,
    )
    from alphalattice.foundation.feature_engine.panels.rematerialization import (
        ArtifactOnlyPanelRematerializer,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelClippingEvidence,
    )
    from alphalattice.foundation.feature_engine.publication.snapshots import (
        resolve_panel_preprocessing_lineage,
    )
    from tests.workspace_maintenance import local_data_provider

    # The seeded walk's closes are a cumulative product from its first
    # session: the update Providers must start where the baseline's did.
    monkeypatch.setattr(local_data_provider, "HISTORY_START", YEAR_END_HISTORY_START)
    workspace = tmp_path / "workspace"
    shutil.copytree(year_end, workspace)
    manifest = bind_existing_data_workspace(workspace)
    market = MarketDataRepository(workspace)
    corrected_listing = next(
        item.listing_id
        for item in market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        ).listings
        if item.symbol == SYMBOLS[0]
    )
    baseline = _active_panel_manifest(workspace)
    assert baseline["as_of_session"] == YEAR_END_AS_OF.isoformat()
    years = [int(chunk["year"]) for chunk in baseline["chunks"]]
    assert years[0] < 2025 and years[-1] == 2026
    baseline_binding = str(baseline["panel_binding_hash"])
    files_before = {
        chunk["chunk_hash"]: _chunk_file(workspace, chunk).stat().st_mtime_ns
        for chunk in baseline["chunks"]
    }

    def boot(now, provider):
        return LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=manifest,
            resolver=_Resolver(_resolved()),
            data_provider=provider,
            data_source_loader=unchanged_membership_source(SYMBOLS),
            clock=lambda: now,
        )

    def snapshot_count() -> int:
        return len(_lifecycles(workspace))

    # Day one: an ordinary new session, same membership, sector, catalog and policy.
    provider_one = recording_provider(now=DAY_ONE)
    with boot(DAY_ONE, provider_one) as live:
        plan_one = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert plan_one["status"] == "PLANNED", plan_one
        request_one = {"update_plan_hash": plan_one["plan_hash"]}
        admitted = _json(live, "/api/data-update/run", method="POST", payload=request_one)
        assert admitted["status"] == "ADMITTED", admitted
        live.dispatcher.drain_for_tests(timeout=600)
        status = _json(live, f"/api/status?task_id={admitted['task_id']}")
        assert status["lifecycle"] == "SUCCEEDED", (status, _json(live, "/api/data-update"))
        readback = _json(live, "/api/data-update")
        assert readback["status"] == "PUBLISHED"
        assert readback["inputs"]["panel_through"] == DAY_ONE.date().isoformat()
        calls_one = list(provider_one.calls)
        snapshots_after_day_one = snapshot_count()
        repeated = _json(live, "/api/data-update/run", method="POST", payload=request_one)
        assert repeated["status"] == "REUSED_EXACT" and repeated["task_id"] is None
        assert provider_one.calls == calls_one
        assert len(live.session.task_control_registry.tasks()) == 1
        assert snapshot_count() == snapshots_after_day_one
    day_one = _active_panel_manifest(workspace)
    day_one_binding = str(day_one["panel_binding_hash"])
    assert day_one_binding != baseline_binding
    assert day_one["as_of_session"] == DAY_ONE.date().isoformat()
    assert day_one["safe_summary"]["partition_reuse"] == {
        "reused_partition_count": len(years) - 1,
        "composed_partition_count": 1,
    }
    baseline_by_year = {int(chunk["year"]): chunk for chunk in baseline["chunks"]}
    for chunk in day_one["chunks"][:-1]:
        assert chunk == baseline_by_year[int(chunk["year"])]
        assert chunk["origin_binding_hash"] == baseline_binding
        assert _chunk_file(workspace, chunk).stat().st_mtime_ns == files_before[chunk["chunk_hash"]]
    current = day_one["chunks"][-1]
    assert int(current["year"]) == 2026
    assert current["origin_binding_hash"] == day_one_binding
    assert current["chunk_hash"] != baseline_by_year[2026]["chunk_hash"]
    assert current["last_session"] == DAY_ONE.date().isoformat()
    assert (
        _row_hashes(workspace, baseline_by_year[2026]).items()
        <= _row_hashes(workspace, current).items()
    )
    origins_one = day_one["safe_summary"]["lineage"]["partition_origins"]
    assert set(origins_one) == {baseline_binding, day_one_binding}
    lifecycles = _lifecycles(workspace)
    assert lifecycles[str(baseline["snapshot_hash"])] == ("SUPERSEDED", "not_current_active_panel")
    assert lifecycles[str(day_one["snapshot_hash"])] == ("ACTIVE", None)

    # Restart: the published result is what the page reads back; nothing is
    # fetched or rebuilt, and the same plan is an exact reuse.
    provider_restart = recording_provider(now=DAY_ONE)
    with boot(DAY_ONE, provider_restart) as live:
        assert not live.resumed_task_ids
        reopened = _json(live, "/api/data-update")
        assert reopened["status"] == "PUBLISHED"
        assert reopened["inputs"]["panel_through"] == DAY_ONE.date().isoformat()
        reused = _json(live, "/api/data-update/run", method="POST", payload=request_one)
        assert reused["status"] == "REUSED_EXACT" and reused["task_id"] is None
        again = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert again["status"] == "PLANNED" and again["target_session"] == (
            DAY_ONE.date().isoformat()
        )
        assert again["inputs"]["panel_through"] == DAY_ONE.date().isoformat()
        assert provider_restart.calls == []
        assert len(live.session.task_control_registry.tasks()) == 1
    assert _active_panel_manifest(workspace)["snapshot_hash"] == day_one["snapshot_hash"]
    assert snapshot_count() == snapshots_after_day_one

    # The closure captured now freezes day one's derivation before any cell moves.
    captured_one, store, resolver = _capture_panel_closure(workspace)
    assert {str(baseline["snapshot_hash"]), str(day_one["snapshot_hash"])} <= set(
        captured_one.recipes
    )
    recipe_one = captured_one.recipes[str(day_one["snapshot_hash"])]

    # Day two: the Provider restates one close in the previous year, inside
    # the refresh overlap, for one listing. Small enough to be no raw-move
    # case: what it triggers is the full-history audit approval.
    provider_two = recording_provider(now=DAY_TWO)
    closes = provider_two._closes
    original_close = closes(SYMBOLS[0], DAY_TWO.date())[CORRECTED_SESSION]

    def with_restated_close(symbol, end):
        values = closes(symbol, end)
        if symbol == SYMBOLS[0] and CORRECTED_SESSION in values:
            values[CORRECTED_SESSION] *= 1.003
        return values

    monkeypatch.setattr(provider_two, "_closes", with_restated_close)
    revisions_before = market.revision_count(corrected_listing)
    with boot(DAY_TWO, provider_two) as live:
        plan_two = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert plan_two["status"] == "PLANNED", plan_two
        blocked = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan_two["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=600)
        status = _json(live, f"/api/status?task_id={blocked['task_id']}")
        assert status["lifecycle"] == "BLOCKED", status
        assert status["latest_failure_code"] == "data.full_history_audit_approval_required", status
        # Refused whole: the restated row is not applied, nothing is published.
        held = {bar.session_date: bar.close for bar in market.raw_bars(corrected_listing)}
        assert held[CORRECTED_SESSION] == original_close
        assert market.revision_count(corrected_listing) == revisions_before
        assert _json(live, "/api/data-update")["inputs"]["panel_through"] == (
            DAY_ONE.date().isoformat()
        )
        assert snapshot_count() == snapshots_after_day_one
        # The approval is a plan naming the audit. The audit is a default the agent
        # takes and discloses (person-stops row 49), so its confirmation admits the Task.
        proposal = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert proposal["status"] == "CONFIRMATION_REQUIRED", proposal
        assert proposal["change"]["action"] == "FULL_HISTORY_AUDIT"
        assert proposal["change"]["full_history_listing_ids"] == [corrected_listing]
        assert proposal["audit_labels"] == [SYMBOLS[0]]
        agent = InstalledAgent(live.operations)
        approved = json.loads(
            agent.invoke(
                PortfolioResearchAgentRequest(
                    operation="DATA_CHANGE_CONFIRM", update_plan_hash=proposal["plan_hash"]
                )
            )
        )
        assert (approved["status"], approved["lifecycle"]) == ("APPROVED", "QUEUED"), approved
        # The confirmation admits the Task; the page's run dispatches that
        # same Task, never a second one.
        dispatched = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": proposal["plan_hash"]},
        )
        assert dispatched["task_id"] == approved["task_id"], dispatched
        live.dispatcher.drain_for_tests(timeout=900)
        status = _json(live, f"/api/status?task_id={approved['task_id']}")
        assert status["lifecycle"] == "SUCCEEDED", status
        published = _json(live, "/api/data-update")
        assert published["status"] == "PUBLISHED"
        assert published["inputs"]["panel_through"] == DAY_TWO.date().isoformat()
        corrected = {bar.session_date: bar.close for bar in market.raw_bars(corrected_listing)}
        assert corrected[CORRECTED_SESSION] == pytest.approx(original_close * 1.003)
        assert market.revision_count(corrected_listing) == revisions_before + 1
        assert snapshot_count() == snapshots_after_day_one + 1

    day_two = _active_panel_manifest(workspace)
    day_two_binding = str(day_two["panel_binding_hash"])
    assert day_two["as_of_session"] == DAY_TWO.date().isoformat()
    assert day_two["safe_summary"]["partition_reuse"] == {
        "reused_partition_count": len(years) - 2,
        "composed_partition_count": 2,
    }
    day_one_by_year = {int(chunk["year"]): chunk for chunk in day_one["chunks"]}
    day_two_by_year = {int(chunk["year"]): chunk for chunk in day_two["chunks"]}
    for year in years[:-2]:
        assert day_two_by_year[year] == day_one_by_year[year]
    for year in (2025, 2026):
        assert day_two_by_year[year]["chunk_hash"] != day_one_by_year[year]["chunk_hash"]
        assert day_two_by_year[year]["origin_binding_hash"] == day_two_binding
    origins_two = day_two["safe_summary"]["lineage"]["partition_origins"]
    assert (
        {baseline_binding, day_two_binding}
        <= set(origins_two)
        <= {
            baseline_binding,
            day_one_binding,
            day_two_binding,
        }
    )
    # The output window is the corrected session onward: every row before it
    # keeps its identity; from it on the corrected listing's rows change, and
    # so do its neighbours' -- the transform is cross-sectional per session.
    before = _row_hashes(workspace, day_one_by_year[2025]) | _row_hashes(
        workspace, day_one_by_year[2026]
    )
    after = _row_hashes(workspace, day_two_by_year[2025]) | _row_hashes(
        workspace, day_two_by_year[2026]
    )
    assert {key for key in before if key[0] < CORRECTED_SESSION} == {
        key for key in after if key[0] < CORRECTED_SESSION
    }
    assert all(after[key] == value for key, value in before.items() if key[0] < CORRECTED_SESSION)
    changed_sessions = sorted({key[0] for key in before if after.get(key) != before[key]})
    assert changed_sessions and changed_sessions[0] == CORRECTED_SESSION
    assert all(
        after[(session, corrected_listing)] != before[(session, corrected_listing)]
        for session in changed_sessions
    )
    assert any(
        after[(CORRECTED_SESSION, listing)] != before[(CORRECTED_SESSION, listing)]
        for (session, listing) in before
        if session == CORRECTED_SESSION and listing != corrected_listing
    )
    lifecycles = _lifecycles(workspace)
    assert lifecycles[str(day_two["snapshot_hash"])] == ("ACTIVE", None)
    assert lifecycles[str(day_one["snapshot_hash"])] == ("SUPERSEDED", "not_current_active_panel")
    assert lifecycles[str(baseline["snapshot_hash"])] == ("SUPERSEDED", "not_current_active_panel")
    lineage = resolve_panel_preprocessing_lineage(
        resolver, panel_content_hash=str(day_two["panel_content_hash"])
    )
    evidence = PanelClippingEvidence.model_validate(
        resolver.load_panel_clipping_evidence(
            resolver.panel_clipping_evidence_uri(str(lineage["clipping_evidence_hash"]))
        )
    )
    calendar_length = sum(int(chunk["row_count"]) for chunk in day_two["chunks"]) // len(SYMBOLS)
    assert all(
        len(item.per_session_clipped_counts) == calendar_length for item in evidence.factor_records
    )

    # Old and new snapshots stay verifiable: day one's recipe (captured before
    # the correction) and day two's (mixed origins, after it) both reproduce
    # their bytes from the closure alone.
    captured_two, store, resolver = _capture_panel_closure(workspace)
    recipe_two = captured_two.recipes[str(day_two["snapshot_hash"])]
    # Day one's cells from the corrected session on were restamped by day
    # two's build, so the live availability no longer describes day one (nor
    # the baseline). The capture does not seal a recipe no replay could
    # reproduce: it selects the recipe the earlier capture sealed over the
    # values of its day, which stays exactly as recoverable as then.
    assert str(day_two["snapshot_hash"]) not in captured_two.blocked_snapshots
    # The fixture's pre-admission snapshot (superseded by the admitted
    # republish on the same day, never captured while its rows stood) is the
    # one that has no earlier recipe to fall back on; it is reported, not sealed.
    assert set(captured_two.blocked_snapshots.values()) <= {"feature_panel.availability_superseded"}
    assert not {str(baseline["snapshot_hash"]), str(day_one["snapshot_hash"])} & set(
        captured_two.blocked_snapshots
    )
    assert captured_two.recipes[str(day_one["snapshot_hash"])] == recipe_one
    assert (
        captured_two.recipes[str(baseline["snapshot_hash"])]
        == (captured_one.recipes[str(baseline["snapshot_hash"])])
    )
    assert (
        store.load_model(
            category="recipes", content_hash=recipe_one.recipe_hash, model=PanelDerivationRecipe
        )
        == recipe_one
    )
    assert {origin.panel_binding_hash for origin in recipe_two.partition_origins or ()} == (
        set(origins_two) - {day_two_binding}
    )
    rematerializer = ArtifactOnlyPanelRematerializer(resolver=resolver, store=store)
    for recipe in (recipe_one, recipe_two):
        result = rematerializer.rematerialize(recipe.recipe_hash)
        assert result.logical_parity is True and result.physical_parity is True
