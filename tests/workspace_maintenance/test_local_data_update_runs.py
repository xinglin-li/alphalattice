"""Local data update runs through real Data/Feature owners, offline: plans,
reviews, quarantine, refusals, cancellation and recovery."""

from __future__ import annotations

import json
import shutil
from contextlib import closing
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pyarrow as pa
import pytest

from alphalattice.control.product_host.composition.local_web_session import LocalPortfolioWebSession
from alphalattice.control.product_host.composition.research_workspace import (
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.maintenance.data_update import (
    WorkspaceDataUpdateApplication,
    bind_existing_data_workspace,
)
from alphalattice.foundation.feature_engine.contracts import (
    FeatureBuildOutcome,
    FeatureBuildStatus,
)
from alphalattice.foundation.feature_engine.runtime.service import FeatureFoundationService
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import (
    InstalledAgent,
    _json,
    _manifest,
    _request,
    _resolved,
    _Resolver,
)
from tests.researcher_methodology_surface.real_workspace import (
    HISTORY_START,
    SYMBOLS,
    build_real_risk_workspace,
)
from tests.researcher_methodology_surface.session_workspace import copy_workspace, session_workspace
from tests.workspace_maintenance.data_update_support import (
    _active_panel_manifest,
    _seed_foundation,
    set_parallel_test_budget,
)
from tests.workspace_maintenance.local_data_provider import (
    NOW,
    recording_provider,
    unchanged_membership_source,
)

TEN_SECTOR_SYMBOLS = tuple(f"QD{index:03d}" for index in range(60))
"""Sixty names in ten sectors of six: one exclusion keeps every sector at or above
the five-member floor and coverage at 59/60, so a membership-changing quality
decision can pass the product's own admission policy. A reduced-scope fixture
universe, not a claim about any real universe."""


@pytest.fixture(scope="module")
def ten_sector(tmp_path_factory):
    """A prepared sixty-name workspace whose sectors hold six members each."""

    def build(root):
        built = build_real_risk_workspace(
            tmp_path_factory.mktemp("data-update-ten-sector"),
            symbols=TEN_SECTOR_SYMBOLS,
            sector_size=6,
        )
        publish_research_workspace_manifest(built.workspace, _manifest("data-update-qa"))
        _seed_foundation(built.workspace, built.panel_snapshot_hash)
        set_parallel_test_budget(built.workspace)
        copy_workspace(built.workspace, root)
        return {}

    seed, _metadata = session_workspace(tmp_path_factory, "maintenance_ten_sector", build)
    return copy_workspace(seed, tmp_path_factory.mktemp("ten-sector") / "workspace")


def _bytes(root: Path):
    """Every workspace file's bytes, except the writer lease, the activity observer's own
    ledger under `runtime/` (`observations.sqlite` and its WAL/SHM) and the sealed plan
    previews (`runtime/plan-previews`, V525, V537): the observer records each observed
    operation and a plan seals its preview, planning included, so those files move with the
    request itself and say nothing about what the update wrote."""
    previews = root / "runtime" / "plan-previews"
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != ".alphalattice-writer.lock"
        and not path.name.startswith("observations.sqlite")
        and previews not in path.parents
    }


@pytest.mark.parametrize(
    "failure,cancelled,expected",
    [
        (None, False, None),
        ("data.empty_payload", False, None),
        (
            "data.full_history_audit_approval_required",
            False,
            "data.full_history_audit_approval_required",
        ),
        ("data.action_audit_invalid", False, "data.action_audit_invalid"),
        (None, True, "workspace_data_update.formation_cancelled"),
    ],
)
def test_formation_catchup_distinguishes_absence_from_authority_failure(
    monkeypatch, failure, cancelled, expected
):

    from alphalattice.control.data_platform.maintenance.data_changes import WorkspaceDataChanges
    from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
        CurrentUniverseMaintenanceOutcome,
        CurrentUniverseMaintenanceStatus,
    )

    calls = []
    rows = (
        ()
        if failure is None
        else (
            SimpleNamespace(
                state="FAILED",
                failure_code=failure,
                listing_id="listing-tap",
                change_document=None,
            ),
        )
    )
    runner = SimpleNamespace(
        maintenance_id="bounded-formation-catchup",
        store=SimpleNamespace(current_universe_maintenance_listings=lambda _: rows),
    )

    def run(**kwargs):
        calls.append(kwargs)
        return CurrentUniverseMaintenanceOutcome(
            runner.maintenance_id,
            CurrentUniverseMaintenanceStatus.COMPLETED,
            1,
            int(failure is None),
            int(failure is not None),
        )

    runner.run = run
    owner = object.__new__(WorkspaceDataChanges)
    scope = SimpleNamespace(listings=(SimpleNamespace(listing_id="listing-tap", symbol="TAP"),))
    monkeypatch.setattr(owner, "formation_scope", lambda *_: scope)
    monkeypatch.setattr(owner, "_quote_runner", lambda *_: runner)
    result = owner.maintain_formation(
        plan=SimpleNamespace(before=None, change=None),
        provider=None,
        observed_at=NOW,
        cancelled=lambda: cancelled,
    )
    assert (None if result is None else result[0]) == expected
    if result is not None:
        assert result[1] == (
            None
            if cancelled
            else {
                "exception_type": "UNKNOWN",
                "detail": "The source failure cause was not recorded.",
                "step": "Provider price history",
                "unit": "TAP",
                "row_count": "UNKNOWN",
                "sanitizer_code": "UNKNOWN",
            }
        )
    assert len(calls) == int(not cancelled)
    assert not calls or calls[0]["work_budget"] == 1
    assert not rows or rows[0].change_document is None


@pytest.mark.parametrize("empty,subcode", [(True, "CORRUPTED_PAYLOAD"), (False, "INVALID_VOLUME")])
def test_failed_price_history_keeps_source_facts_without_admitting_rows(tmp_path, empty, subcode):
    from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
        CurrentUniverseMaintenance,
        CurrentUniverseMaintenanceStatus,
    )
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_quality_filtered_research_manifest,
    )
    from alphalattice.foundation.market_data_ops.sources.providers import HydrationEvidence
    from alphalattice.foundation.market_data_ops.sources.sanitization import (
        CorruptedPayload,
        sanitize_payload,
    )
    from tests.workspace_maintenance.acquisition_manifest import acquisition_manifest

    manifest = build_quality_filtered_research_manifest(
        acquisition_manifest(), eligible_listing_ids=("listing-aapl",)
    )
    market = MarketDataRepository(tmp_path / "workspace")
    market.bootstrap(manifest)
    valid = {
        "session_date": "2026-07-30",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
    }
    market.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, "fixture", {"AAPL": (valid,)}, ("AAPL",)),
        ingestion_id="qualified-history",
        observed_at=NOW,
    )
    before = market.raw_bars("listing-aapl")
    rejected = () if empty else ({**valid, "session_date": "2026-07-31", "volume": -1},)
    with pytest.raises(CorruptedPayload) as caught:
        sanitize_payload(manifest, "fixture", {"AAPL": rejected}, ("AAPL",))
    assert caught.value.code == subcode

    class RejectedHistoryProvider:
        name = "fixture"

        def fetch_hydration(self, *, listing_id, provider_symbol, start, end):
            assert listing_id == "listing-aapl" and provider_symbol == "AAPL"
            return HydrationEvidence(daily_rows=rejected, actions=(), adjusted_closes=())

    runner = CurrentUniverseMaintenance(
        store=market,
        manifest=manifest,
        provider=RejectedHistoryProvider(),
        as_of_session=date(2026, 7, 31),
    )
    outcome = runner.run(observed_at=NOW)
    assert outcome.status is CurrentUniverseMaintenanceStatus.COMPLETED
    assert outcome.updated == 0 and outcome.failed == 1
    assert outcome.listing_changes == ()
    listing = market.current_universe_maintenance_listings(runner.maintenance_id)[0]
    assert listing.state == "FAILED"
    assert listing.failure_code == "data.sanitizer.corrupted_payload"
    assert listing.change_document == {
        "failure_cause": {
            "exception_type": "CorruptedPayload",
            "detail": "The provider price history failed validation.",
            "step": "Provider price history",
            "unit": "AAPL",
            "row_count": len(rejected),
            "sanitizer_code": subcode,
        },
        # The rejected rows are kept as source evidence beside the cause, never admitted.
        "rejected_history": {
            "provider": "fixture",
            "requested_through": "2026-07-31",
            "rows": [dict(row) for row in rejected],
            "provider_policy_hash": None,
            "repaired_sessions": [],
        },
    }
    assert market.raw_bars("listing-aapl") == before
    assert market.manifest_raw_through(manifest) == date(2026, 7, 30)


@pytest.mark.parametrize("source_unavailable", (False, True))
def test_real_update_plan_run_reuse_and_foundation_readback(
    qualified, tmp_path, monkeypatch, source_unavailable
):
    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    assert bind_existing_data_workspace(workspace) == manifest
    provider = recording_provider()
    market = MarketDataRepository(workspace)
    observations_before = market.universe_source_observations("us-current-index-research")
    membership_before = market.membership_events("us-current-index-research")
    source_calls = []
    verified_source = unchanged_membership_source(SYMBOLS)

    def source_loader(**kwargs):
        source_calls.append(kwargs["observed_at"])
        if source_unavailable:
            raise RuntimeError("fixture candidate source unavailable")
        return verified_source(**kwargs)

    before_foundation = _bytes(workspace / "artifacts/factor-research")

    def forbidden(*args, **kwargs):
        raise AssertionError("input update must not run Factor research")

    from alphalattice.foundation.research_foundation.mandate.foundation import (
        ResearchFoundationService,
    )

    monkeypatch.setattr(ResearchFoundationService, "publish", forbidden)
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=source_loader,
        clock=lambda: NOW,
    )
    service.start()
    try:
        before = _bytes(workspace)
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        assert planned["status"] == "PLANNED", planned
        assert planned["next_requests"] == {  # V418: `data-update run --from` reads it
            "run": {"operation": "DATA_UPDATE_RUN", "update_plan_hash": planned["plan_hash"]}
        }
        assert _bytes(workspace) == before and provider.calls == []
        # Before any update Task: no work facts to read, and the Panel's own account of
        # its first composition (every partition composed, none reused).
        untouched = _json(service, "/api/data-update")
        assert untouched["task_id"] is None and "cycle" not in untouched, untouched
        assert untouched["partition_reuse"] == {
            "reused_partition_count": 0,
            "composed_partition_count": len(_active_panel_manifest(workspace)["chunks"]),
            "source": "CURRENT_PANEL",
        }, untouched
        agent = InstalledAgent(service.operations)
        via_agent = json.loads(
            agent.invoke(PortfolioResearchAgentRequest(operation="DATA_UPDATE_PLAN"))
        )
        assert via_agent == planned
        response = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": planned["plan_hash"]},
        )
        assert response["status"] == "ADMITTED", response
        assert response["next_requests"] == {
            "show": {"operation": "DATA_UPDATE_READBACK", "task_id": response["task_id"]}
        }
        service.dispatcher.drain_for_tests()
        status = _json(service, f"/api/status?task_id={response['task_id']}")
        assert status["lifecycle"] == "SUCCEEDED", status
        readback = _json(service, "/api/data-update")
        assert readback["inputs"]["data_through"] == NOW.date().isoformat()
        assert readback["inputs"]["panel_through"] == NOW.date().isoformat()
        assert readback["inputs"]["foundation_disposition"] == "PRIOR_INPUTS"
        # The update's own work, read back for its Task from the existing records: the
        # cycle it admitted (completed, one new session for every name), the maintenance
        # runner's units (every name updated, none pending, each with the new session),
        # the Task-bound progress of the maintenance stage (retained once the Task ended,
        # bound to that stage), and the Panel's own account of what it reused.
        assert readback["task_id"] == response["task_id"]
        cycle = readback["cycle"]
        assert (cycle["phase"], cycle["status"], cycle["failure_code"]) == (
            "completed",
            "completed",
            None,
        ), cycle
        assert cycle["change_set"]["listings_with_new_sessions"] == len(SYMBOLS)
        assert cycle["change_set"]["new_sessions"] == len(SYMBOLS)
        assert cycle["change_set"]["membership_additions"] == 0
        assert cycle["change_set"]["listings_with_corrections"] == 0
        assert cycle["change_set"]["restated_sessions"] == 0
        maintenance = readback["maintenance"]
        assert maintenance["availability"] == "AVAILABLE"
        assert maintenance["cycle_id"] == cycle["cycle_id"]
        assert maintenance["as_of_session"] == NOW.date().isoformat()
        # The units are keyed by the cycle's own request (its membership revision), which
        # for this plain update is the plan's manifest before and after alike.
        assert maintenance["manifest_revision"] == readback["inputs"]["manifest_revision"]
        assert maintenance["counts"] == {
            "listings": len(SYMBOLS),
            "processed": len(SYMBOLS),
            "updated": len(SYMBOLS),
            "failed": 0,
            "pending": 0,
            "with_new_sessions": len(SYMBOLS),
            "corrected": 0,
            "reverified": 0,
        }, maintenance["counts"]
        assert maintenance["moved"] == maintenance["retained"] == len(SYMBOLS)
        assert {row["symbol"] for row in maintenance["rows"]} == set(SYMBOLS)
        assert all(
            row["state"] == "UPDATED"
            and row["new_sessions"] == [NOW.date().isoformat()]
            and row["raw_through"] == NOW.date().isoformat()
            and row["audit_scope"] == "ROLLING"
            and row["restated_sessions"] == 0  # the new session's own return is not a correction
            for row in maintenance["rows"]
        ), maintenance["rows"][0]
        progress = readback["work_progress"]
        assert progress["availability"] == "NOT_CURRENT", progress
        assert progress["stage"] == "maintain_data_feature"
        assert progress["delivery"]["delivered"] >= 1 and progress["delivery"]["failures"] == 0
        sidecar = workspace / "runtime/data-update" / response["task_id"] / "work-progress.json"
        assert sidecar.is_file()
        # The receipt's resulting Panel, named, not whichever Panel is active today.
        assert readback["partition_reuse"] == {
            "reused_partition_count": len(_active_panel_manifest(workspace)["chunks"]) - 1,
            "composed_partition_count": 1,
            "source": "RECEIPT_PANEL",
            "panel_hash": readback["receipt"]["after"]["panel_hash"],
        }
        assert _json(service, "/api/data-update?task_id=" + response["task_id"])["cycle"] == cycle
        if source_unavailable:
            assert source_calls == [NOW]
            assert (
                readback["inputs"]["source_disclosure"]
                == "LAST_KNOWN_MEMBERSHIP_AFTER_SOURCE_FAILURE"
            )
            assert (
                readback["inputs"]["sources_checked_at"] == planned["inputs"]["sources_checked_at"]
            )
            assert readback["inputs"]["source_check_failed_at"] is not None
            assert (
                market.universe_source_observations("us-current-index-research")
                == observations_before
            )
            assert market.membership_events("us-current-index-research") == membership_before
        else:
            assert "source_disclosure" not in readback["inputs"]
        assert _bytes(workspace / "artifacts/factor-research") == before_foundation
        calls = list(provider.calls)
        repeated = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": planned["plan_hash"]},
        )
        assert repeated["status"] == "REUSED_EXACT" and repeated["task_id"] is None
        assert provider.calls == calls and len(service.session.task_control_registry.tasks()) == 1
        assert all(start > HISTORY_START for _, start, _ in calls)
        assert (
            json.loads(
                agent.invoke(PortfolioResearchAgentRequest(operation="DATA_UPDATE_READBACK"))
            )
            == readback
        )

        def cancelled_valuation(**_):
            raise ValueError("workspace_data_update.valuation_cancelled")

        owner = service.operations.data_update
        monkeypatch.setattr(owner.changes, "maintain_valuation", cancelled_valuation)
        cancelled = owner.execute_step(
            owner.prepare(planned["plan_hash"]),
            "maintain_data_feature",
            cancelled=lambda: False,
        )
        assert cancelled.disposition.value == "CANCELLED"
    finally:
        service.stop()


def test_a_restarted_host_runs_a_maintenance_plan_by_its_hash_within_its_hour(qualified, tmp_path):
    """A restarted host runs a maintenance plan by its hash within its hour."""

    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    clock = [NOW]

    def boot() -> LocalPortfolioWebSession:
        service = LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=manifest,
            resolver=_Resolver(_resolved()),
            data_provider=recording_provider(),
            data_source_loader=unchanged_membership_source(SYMBOLS),
            clock=lambda: clock[0],
        )
        service.start()
        return service

    service = boot()
    try:
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        assert planned["status"] == "PLANNED", planned
    finally:
        service.stop()
    plan_hash = planned["plan_hash"]
    assert (workspace / f"runtime/plan-previews/data-update/{plan_hash}.json").is_file()
    assert not (workspace / f"runtime/artifacts/data-update-plans/{plan_hash}.json").exists()
    clock[0] = NOW + timedelta(minutes=59)
    service = boot()
    try:
        owner = service.operations.data_update
        assert owner.last_plan is None
        assert owner.prepare(plan_hash).content_hash == plan_hash
        clock[0] = NOW + timedelta(minutes=61)
        for hash_sent in (plan_hash, "e" * 64):
            refused = _json(
                service,
                "/api/data-update/run",
                method="POST",
                payload={"update_plan_hash": hash_sent},
            )
            assert refused["status"] == "REFUSED", refused
            assert refused["failure_code"] == "workspace_data_update.plan_required", refused
            assert refused["detail"].startswith("A data update's plan stays runnable"), refused
            # Plan the update again (V543).
            assert refused["next_requests"] == {"replan": {"operation": "DATA_UPDATE_PLAN"}}
        assert service.session.task_control_registry.tasks() == ()
    finally:
        service.stop()


def test_selected_update_readback_is_scoped_by_its_own_cycle_and_counts_distinct_dates(
    tmp_path: Path, monkeypatch
) -> None:
    """Selected update readback is scoped by its own cycle and counts distinct dates."""

    from alphalattice.control.data_platform.maintenance.contracts import (
        ListingMarketDataChange,
        MaintenancePhase,
        MarketDataChangeSet,
    )
    from alphalattice.control.product_host.maintenance import data_update as owner

    session_date = date(2026, 8, 3)
    corrected = date(2026, 7, 30)
    revision_a, revision_b = "a" * 64, "b" * 64
    request_a = SimpleNamespace(
        membership_revision=revision_a,
        target_market_session=session_date,
        full_history_listing_ids=(),
    )
    cycle_a = SimpleNamespace(
        cycle_id="cycle-a",
        request=request_a,
        phase=MaintenancePhase.COMPLETED,
        status=owner.MaintenanceStatus.COMPLETED,
        updated_at=datetime(2026, 8, 3, 23, 1, tzinfo=UTC),
        retry_after_at=None,
        failure_code=None,
        transport_workers=4,
        market_data_change_set=MarketDataChangeSet(
            listing_changes=(
                ListingMarketDataChange(
                    listing_id="l1",
                    new_sessions=(session_date,),
                    raw_correction_sessions=(corrected,),
                    adjusted_return_change_sessions=(corrected, session_date),
                ),
                ListingMarketDataChange(listing_id="l2", new_sessions=(session_date,)),
            ),
        ),
    )
    manifests = {revision_a: SimpleNamespace(revision_sha256=revision_a)}
    runs: dict[str, object] = {}
    listings: dict[str, tuple[object, ...]] = {}

    def unit(tag: str, listing: str, state: str, change: dict | None, seconds: int) -> object:
        return SimpleNamespace(
            listing_id=listing,
            symbol=f"{tag}_{listing}",
            state=state,
            failure_code="data.fetch_failed" if state == "FAILED" else None,
            raw_through=session_date if state == "UPDATED" else None,
            change_document=change,
            attempt_count=1,
            updated_at=datetime(2026, 8, 3, 23, 0, seconds),
        )

    for revision, tag in ((revision_a, "A"), (revision_b, "B")):
        manifest = SimpleNamespace(revision_sha256=revision)
        manifests[revision] = manifest
        maintenance_id = owner.current_universe_maintenance_id(manifest, as_of_session=session_date)
        runs[maintenance_id] = SimpleNamespace(
            lifecycle="COMPLETED",
            as_of_session=session_date,
            research_manifest_revision=revision,
            updated_at=datetime(2026, 8, 3, 23, 0, 30),
        )
        listings[maintenance_id] = (
            unit(
                tag,
                "l1",
                "UPDATED",
                {
                    "new_sessions": [session_date.isoformat()],
                    "raw_correction_sessions": [corrected.isoformat()],
                    "adjusted_return_change_sessions": [
                        corrected.isoformat(),
                        session_date.isoformat(),
                    ],
                    "audit_scope": "ROLLING",
                },
                1,
            ),
            unit(
                tag,
                "l2",
                "UPDATED",
                {
                    "raw_correction_sessions": [corrected.isoformat()],
                    "adjusted_return_change_sessions": [corrected.isoformat()],
                    "audit_scope": "ROLLING",
                },
                2,
            ),
            unit(tag, "l3", "UPDATED", {"audit_scope": "ROLLING"}, 3),
            unit(tag, "l4", "FAILED", None, 4),
            unit(tag, "l5", "PENDING", None, 0),
        )
    active = SimpleNamespace(active_manifest_id="manifest-b", active_manifest_revision=revision_b)

    class Market:
        def __init__(self, workspace: Path) -> None:
            self.readiness = SimpleNamespace(load=lambda profile: active)

        def load_universe_manifest(self, manifest_id: str) -> object:
            assert manifest_id == "manifest-b"
            return manifests[revision_b]

        def load_universe_manifest_revision(self, revision: str) -> object:
            if revision not in manifests or revision == "missing":
                raise ValueError("workspace manifest revision is missing or ambiguous")
            return manifests[revision]

        def current_universe_maintenance_run(self, maintenance_id: str) -> object:
            if maintenance_id not in runs:
                raise ValueError("current-universe maintenance does not exist")
            return runs[maintenance_id]

        def current_universe_maintenance_listings(self, maintenance_id: str) -> tuple:
            return listings[maintenance_id]

    monkeypatch.setattr(owner, "MarketDataRepository", Market)
    application = owner.WorkspaceDataUpdateApplication.__new__(owner.WorkspaceDataUpdateApplication)
    application.session = SimpleNamespace(workspace=tmp_path)
    application.telemetry = SimpleNamespace(
        instant=lambda value: value if value.tzinfo else value.replace(tzinfo=UTC),
        age=lambda value: 0.0,
    )
    plan = SimpleNamespace(request=request_a, binding=SimpleNamespace(market_profile_id="p"))
    monkeypatch.setattr(application, "_cycle", lambda plan: cycle_a)
    scopes: dict[str, dict[str, str] | None] = {"cycle-a": None}
    monkeypatch.setattr(
        application,
        "_registry",
        lambda: SimpleNamespace(market_data_scope=lambda cycle_id: scopes.get(cycle_id)),
    )

    # A cycle recorded before the scope record existed: its own request, verified against
    # the run found (revision A), never the active manifest (revision B).
    maintenance = application._maintenance_readback(plan)
    assert maintenance["availability"] == "AVAILABLE"
    assert maintenance["scope_source"] == "CYCLE_REQUEST"
    assert maintenance["cycle_id"] == "cycle-a"
    assert maintenance["manifest_revision"] == revision_a, "the cycle's own revision, not B"
    # A cycle that recorded the run it admitted (a transition ran under revision B while its
    # request still names A): the recorded run, by name.
    run_b = owner.current_universe_maintenance_id(manifests[revision_b], as_of_session=session_date)
    scopes["cycle-a"] = {
        "maintenance_id": run_b,
        "manifest_revision": revision_b,
        "as_of_session": session_date.isoformat(),
    }
    recorded = application._maintenance_readback(plan)
    assert recorded["availability"] == "AVAILABLE" and recorded["scope_source"] == "CYCLE_RECORD"
    assert recorded["manifest_revision"] == revision_b
    assert {row["symbol"] for row in recorded["rows"]} == {"B_l1", "B_l2", "B_l3", "B_l4"}
    # A recorded run the store no longer holds is unavailable by name, never today's units.
    scopes["cycle-a"] = {
        "maintenance_id": "gone",
        "manifest_revision": revision_b,
        "as_of_session": "x",
    }
    missing = application._maintenance_readback(plan)
    assert missing["availability"] == "UNAVAILABLE"
    assert missing["failure_code"] == "workspace_data_update.maintenance_run_missing"
    scopes["cycle-a"] = None
    assert {row["symbol"] for row in maintenance["rows"]} == {"A_l1", "A_l2", "A_l3", "A_l4"}
    by_listing = {row["listing_id"]: row for row in maintenance["rows"]}
    assert by_listing["l1"]["restated_sessions"] == 1, "one date in both corrections is one"
    assert by_listing["l1"]["new_sessions"] == [session_date.isoformat()]
    assert by_listing["l2"]["restated_sessions"] == 1 and by_listing["l2"]["new_sessions"] == []
    assert by_listing["l3"]["restated_sessions"] == 0 and by_listing["l3"]["new_sessions"] == []
    assert maintenance["counts"] == {
        "listings": 5,
        "processed": 4,
        "updated": 3,
        "failed": 1,
        "pending": 1,
        "with_new_sessions": 1,
        "corrected": 1,
        "reverified": 1,
    }
    cycle = application._cycle_readback(plan)
    assert cycle["change_set"]["listings_with_corrections"] == 1
    assert cycle["change_set"]["restated_sessions"] == 1
    assert cycle["change_set"]["new_sessions"] == 2
    assert cycle["change_set"]["listings_with_new_sessions"] == 2
    # A membership revision the store no longer resolves is unavailable by name.
    request_a.membership_revision = "missing"
    unavailable = application._maintenance_readback(plan)
    assert unavailable["availability"] == "UNAVAILABLE"
    assert unavailable["failure_code"] == "workspace_data_update.maintenance_manifest_unavailable"
    request_a.membership_revision = revision_a
    # Partition reuse: the receipt's resulting Panel, or unavailable by name -- never the
    # Panel active today.
    receipt = SimpleNamespace(after=SimpleNamespace(panel_hash="panel-of-the-receipt"))
    summaries = {
        "panel-of-the-receipt": {
            "partition_reuse": {"reused_partition_count": 7, "composed_partition_count": 2}
        },
        "panel-active-today": {
            "partition_reuse": {"reused_partition_count": 0, "composed_partition_count": 9}
        },
    }

    def panel_summary(resolver: object, panel_hash: str) -> dict:
        if panel_hash not in summaries:
            raise FileNotFoundError(panel_hash)
        return summaries[panel_hash]

    monkeypatch.setattr(
        owner.WorkspaceDataUpdateApplication, "_panel_summary", staticmethod(panel_summary)
    )
    assert application._receipt_partition_reuse(None, receipt) == {
        "reused_partition_count": 7,
        "composed_partition_count": 2,
        "source": "RECEIPT_PANEL",
        "panel_hash": "panel-of-the-receipt",
    }
    gone = SimpleNamespace(after=SimpleNamespace(panel_hash="panel-gone"))
    assert application._receipt_partition_reuse(None, gone) == {
        "source": "RECEIPT_PANEL",
        "availability": "UNAVAILABLE",
        "failure_code": "workspace_data_update.receipt_panel_missing",
        "panel_hash": "panel-gone",
    }


def test_raw_move_review_reopens_and_resumes_the_same_data_task(qualified, tmp_path, monkeypatch):
    from alphalattice.interface.local_application.client import LocalResearchClient

    workspace = tmp_path / "review-workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    provider = recording_provider()
    closes = provider._closes

    def with_unexplained_move(symbol, end):
        values = closes(symbol, end)
        if symbol == SYMBOLS[0] and NOW.date() in values:
            values[NOW.date()] *= 2.0
        return values

    monkeypatch.setattr(provider, "_closes", with_unexplained_move)

    def boot():
        return LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=manifest,
            resolver=_Resolver(_resolved()),
            data_provider=provider,
            data_source_loader=unchanged_membership_source(SYMBOLS),
            clock=lambda: NOW,
        )

    with boot() as live:
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        request = {"update_plan_hash": plan["plan_hash"]}
        admitted = _json(live, "/api/data-update/run", method="POST", payload=request)
        live.dispatcher.drain_for_tests(timeout=180)
        status = _json(live, "/api/status?task_id=" + admitted["task_id"])
        assert status["lifecycle"] == "BLOCKED", status
        assert status["latest_failure_code"] == "data.truth_review_required", status
        body = _json(live, "/api/workspace/data-issues")
        client = LocalResearchClient(workspace)
        assert client.request({"operation": "DATA_ISSUES"}) == body
        issue = body["issues"][0]
        case = issue["case"]
        option = next(
            v for v in case["options"] if v["option_id"] == "retain_isolated_raw_move_with_caveat"
        )
        choice = {
            "data_issue_case_token": case["case_token"],
            "data_issue_evidence_hash": case["evidence_hash"],
            "data_issue_option_id": option["option_id"],
            "data_issue_option_hash": option["option_hash"],
        }
        key = f"preview:{case['case_token']}:{option['option_id']}"
        assert body["next_requests"][key] == {"operation": "DATA_ISSUE_PREVIEW", **choice}
        assert {
            value["data_issue_option_id"]
            for name, value in body["next_requests"].items()
            if name.startswith("preview:") and value["data_issue_case_token"] == case["case_token"]
        } == set(issue["confirmable_option_ids"])
        refused = client.request({"operation": "DATA_ISSUE_CONFIRM", **choice})
        assert "human_confirmation_required" in str(refused)
        from alphalattice.control.product_host.data_preparation.remediation import (
            WorkspaceDataIssueApplication,
        )
        from alphalattice.interface.local_application.cli import main as client_main

        exported = tmp_path / "data-issues.json"
        exported.write_text(json.dumps(body), encoding="utf-8")
        preview = tmp_path / "preview.json"
        assert (
            client_main(
                [
                    "--workspace",
                    str(workspace),
                    "request",
                    "--from",
                    str(exported),
                    "--action",
                    key,
                    "--output",
                    str(preview),
                ],
                serve=lambda _: pytest.fail("preview must use the running Host"),
            )
            == 0
        )
        proposed = json.loads(preview.read_text(encoding="utf-8"))
        assert proposed["status"] == "CONFIRMATION_REQUIRED" and not proposed["effect_applied"]
        expired = WorkspaceDataIssueApplication(
            live.session, lambda: NOW + timedelta(days=2)
        ).readback()
        # The elapsed case offers no choice; the stopped Task still offers to continue, and
        # the pending read is always offered (V237).
        assert [
            name
            for name in expired["next_requests"]
            if not name.startswith("continue:") and name != "pending"
        ] == []
        before = MarketDataRepository(workspace).raw_bars(case["listing_ids"][0])
        decision = _json(live, "/api/workspace/data-issues/confirm", method="POST", payload=choice)
        assert decision["status"] == "CONFIRMED_PENDING_REVALIDATION"
        assert decision["continuations"][0]["task_id"] == admitted["task_id"]
        calls = list(provider.calls)

    with boot() as live:
        pending = _json(live, "/api/workspace/data-issues")
        assert pending["issues"][0]["case"] == case
        assert pending["issues"][0]["status"] == "CONFIRMED_PENDING_REVALIDATION"
        assert not any(
            value.get("data_issue_case_token") == case["case_token"]
            for value in pending["next_requests"].values()
        )
        # The preview offered before the decision, sent now, refuses as the list does.
        held = _request(live, "/api/workspace/data-issues/preview", method="POST", payload=choice)
        assert json.loads(held[2])["failure_code"] == "feature_input.case_already_decided", held
        resumed = _json(live, "/api/data-update/run", method="POST", payload=request)
        assert resumed["task_id"] == admitted["task_id"]
        live.dispatcher.drain_for_tests(timeout=180)
        status = _json(live, "/api/status?task_id=" + admitted["task_id"])
        assert status["lifecycle"] == "SUCCEEDED", status
        # Listing acquisition was already complete. SPY was still owed because
        # the first command stopped at quality before its benchmark phase.
        assert all(symbols == ("SPY",) for symbols, _start, _end in provider.calls[len(calls) :])
        assert MarketDataRepository(workspace).raw_bars(case["listing_ids"][0]) == before
        assert len(live.session.task_control_registry.tasks()) == 1
        repeated = _json(live, "/api/data-update/run", method="POST", payload=request)
        assert repeated["status"] == "REUSED_EXACT" and repeated["task_id"] is None
        effect = _json(live, "/api/workspace/data-issues/confirm", method="POST", payload=choice)
        assert effect["status"] == "ALREADY_APPLIED"
        assert effect["receipt_hash"] == decision["receipt_hash"]


def _quarantine_review_workspace(
    ten_sector, tmp_path, monkeypatch, *, through=NOW, anomaly="single_session"
):
    """A sixty-name workspace one update behind; its first listing shows an unexplained move.

    ``anomaly="single_session"`` doubles the first listing's close on the
    update's session alone; ``"level_shift"`` doubles it from that session on
    (a historical jump that persists, the shape of a standing quarantine's
    anomaly). The provider serves sessions through ``through``; ``boot``
    takes the session's clock.
    """

    workspace = tmp_path / "quarantine-workspace"
    shutil.copytree(ten_sector, workspace)
    manifest = bind_existing_data_workspace(workspace)
    provider = recording_provider(symbols=TEN_SECTOR_SYMBOLS, sector_size=6, now=through)
    closes = provider._closes
    state = {"anomaly": anomaly}

    def with_unexplained_move(symbol, end):
        values = closes(symbol, end)
        if symbol != TEN_SECTOR_SYMBOLS[0]:
            return values
        shape = state["anomaly"]
        for session in values:
            if session == NOW.date() or (session > NOW.date() and shape == "level_shift"):
                values[session] *= 2.0
        return values

    monkeypatch.setattr(provider, "_closes", with_unexplained_move)

    def boot(clock=NOW):
        return LocalPortfolioWebSession(
            workspace=workspace,
            workspace_manifest=manifest,
            resolver=_Resolver(_resolved()),
            data_provider=provider,
            data_source_loader=unchanged_membership_source(TEN_SECTOR_SYMBOLS),
            clock=lambda: clock,
        )

    boot.provider = provider  # type: ignore[attr-defined]
    boot.anomaly = state  # type: ignore[attr-defined]
    return workspace, boot


def _decide_quarantine(live, plan):
    """Plan, run to the review stop, confirm ``recoverable_quarantine`` on the page's route."""

    request = {"update_plan_hash": plan["plan_hash"]}
    admitted = _json(live, "/api/data-update/run", method="POST", payload=request)
    live.dispatcher.drain_for_tests(timeout=180)
    status = _json(live, "/api/status?task_id=" + admitted["task_id"])
    assert status["lifecycle"] == "BLOCKED", status
    assert status["latest_failure_code"] == "data.truth_review_required", status
    issue = _json(live, "/api/workspace/data-issues")["issues"][0]
    case = issue["case"]
    option = next(v for v in case["options"] if v["option_id"] == "recoverable_quarantine")
    choice = {
        "data_issue_case_token": case["case_token"],
        "data_issue_evidence_hash": case["evidence_hash"],
        "data_issue_option_id": option["option_id"],
        "data_issue_option_hash": option["option_hash"],
    }
    decision = _json(live, "/api/workspace/data-issues/confirm", method="POST", payload=choice)
    assert decision["status"] == "CONFIRMED_PENDING_REVALIDATION"
    continuation = decision["continuations"][0]
    assert continuation["task_id"] == admitted["task_id"]
    # The stop is explained on the page's own route, by its stable code.
    assert continuation["failure_reason"]["code"] == "data.truth_review_required"
    assert "data decision" in continuation["failure_reason"]["explanation"]
    return admitted["task_id"], request, case


@pytest.fixture(scope="module")
def standing_quarantine_seed(ten_sector, tmp_path_factory):
    """One original first-day decision/publication; every recheck owns its copy."""

    def build(root):
        with pytest.MonkeyPatch.context() as monkeypatch:
            day_one, day_two_clock = NOW, NOW + timedelta(days=1)
            workspace, boot = _quarantine_review_workspace(
                ten_sector,
                tmp_path_factory.mktemp("standing-quarantine"),
                monkeypatch,
                through=day_two_clock,
                anomaly="level_shift",
            )
            with boot(day_one) as live:
                plan = _json(live, "/api/data-update/plan", method="POST", payload={})
                task_id, request, case = _decide_quarantine(live, plan)
            # Reopened between the decision and its continuation: the same Task
            # continues from the recorded decision, nothing is asked again.
            with boot(day_one) as live:
                assert [str(item) for item in live.resumed_task_ids] == []
                resumed = _json(live, "/api/data-update/run", method="POST", payload=request)
                assert resumed["task_id"] == task_id
                live.dispatcher.drain_for_tests(timeout=300)
                status = _json(live, "/api/status?task_id=" + task_id)
                assert status["lifecycle"] == "SUCCEEDED", status
                assert (
                    _json(live, "/api/data-update")["inputs"]["panel_through"]
                    == day_one.date().isoformat()
                )
        copy_workspace(workspace, root)
        return {"case": case}

    return session_workspace(tmp_path_factory, "standing_quarantine", build)


@pytest.mark.parametrize("day_two", ["unchanged", "new_anomaly", "explained"])
def test_a_standing_quarantine_is_continued_across_an_unchanged_anomaly_and_not_otherwise(
    standing_quarantine_seed, tmp_path, monkeypatch, day_two
):
    """A standing quarantine is continued across an unchanged anomaly and not otherwise."""

    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.sources.contracts import CorporateActionEvent

    day_one, day_two_clock = NOW, NOW + timedelta(days=1)
    source, metadata = standing_quarantine_seed
    workspace, boot = _quarantine_review_workspace(
        source, tmp_path, monkeypatch, through=day_two_clock, anomaly="level_shift"
    )
    market = MarketDataRepository(workspace)
    panel_state = PanelStateRepository(market.database, market_data=market)
    parent = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    provider = boot.provider  # type: ignore[attr-defined]
    case = deepcopy(metadata["case"])
    listing_id = case["listing_ids"][0]
    standing = panel_state.active_listing_quarantines(
        parent.revision_sha256, include_profile_history=True, qualification_domain="MARKET_DATA"
    )
    assert [item.listing_id for item in standing] == [listing_id]
    original = standing[0]
    assert original.recheck_after_at <= day_two_clock

    if day_two == "new_anomaly":
        # A second jump on the new session: a different anomaly, not the old one.
        closes = provider._closes

        def with_second_jump(symbol, end):
            values = closes(symbol, end)
            if symbol == TEN_SECTOR_SYMBOLS[0] and day_two_clock.date() in values:
                values[day_two_clock.date()] *= 2.0
            return values

        monkeypatch.setattr(provider, "_closes", with_second_jump)
    elif day_two == "explained":
        # The provider now reports the split behind the jump.
        def split_history(*, listing_id, provider_symbol, start, end):
            if provider_symbol == TEN_SECTOR_SYMBOLS[0] and start <= day_one.date() <= end:
                return (
                    CorporateActionEvent(
                        listing_id=listing_id,
                        provider="yfinance",
                        effective_date=day_one.date(),
                        action_kind="SPLIT",
                        new_shares_per_old_share=0.5,
                        cash_amount=None,
                        provisional=False,
                        provenance="fixture",
                    ),
                )
            return ()

        monkeypatch.setattr(provider, "fetch_action_history", split_history)

    with boot(day_two_clock) as live:
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        assert plan["status"] == "PLANNED", plan
        run = _json(
            live,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + run["task_id"])
        issues = _json(live, "/api/workspace/data-issues")
        current = panel_state.active_listing_quarantines(
            parent.revision_sha256, include_profile_history=True, qualification_domain="MARKET_DATA"
        )
        if day_two == "new_anomaly":
            assert status["lifecycle"] == "BLOCKED", status
            assert status["latest_failure_code"] == "data.truth_review_required"
            pending = [item for item in issues["issues"] if item["status"] == "AWAITING_CHOICE"]
            assert len(pending) == 1 and pending[0]["case"]["listing_ids"] == [listing_id]
            assert pending[0]["case"]["case_token"] != case["case_token"]
            sessions = pending[0]["case"]["evidence"][0]["unexplained_sessions"]
            assert sessions == [day_one.date().isoformat(), day_two_clock.date().isoformat()]
            assert [item.quarantine_hash for item in current] == [original.quarantine_hash]
            # The page says why it asks again over a listing that carries a decision.
            (note,) = pending[0]["standing_quarantine"]["standing"]
            assert note["quarantine_hash"] == original.quarantine_hash
            assert note["execution_receipt_hash"] == original.execution_receipt_hash
            assert note["continuation_refused"] == ["ANOMALY_CHANGED"]
            assert issues["continued_dispositions"] == []
            return
        assert status["lifecycle"] == "SUCCEEDED", status.get("worker_failure")
        assert not [item for item in issues["issues"] if item["status"] == "AWAITING_CHOICE"]
        readback = _json(live, "/api/data-update")
        assert readback["status"] == "PUBLISHED"
        assert readback["inputs"]["panel_through"] == day_two_clock.date().isoformat()
        assert readback["inputs"]["manifest_revision"] == parent.revision_sha256
        assert market.membership_events("us-current-index-research") == ()
        snapshot = panel_state.feature_panel_snapshot_for_active("us-current-index-research")
        from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver

        payload = ArtifactResolver(workspace / "artifacts").load_feature_panel_manifest(
            str(snapshot["manifest_uri"])
        )
        if day_two == "explained":
            assert current == ()
            assert payload["safe_summary"]["quality_governance"]["admitted_listing_count"] == 60
            return
        # Continued: a new quarantine over the new evidence, chained to the
        # original decision, due again a day later; no new decision recorded.
        assert len(current) == 1 and current[0].listing_id == listing_id
        continued = current[0]
        assert continued.quarantine_hash != original.quarantine_hash
        assert continued.continued_from_quarantine_hash == original.quarantine_hash
        assert continued.recheck_after_at > day_two_clock
        assert payload["safe_summary"]["quality_governance"]["admitted_listing_count"] == 59
        dispositions = issues["continued_dispositions"]
        assert len(dispositions) == 1
        chain = dispositions[0]
        assert chain["listing_id"] == listing_id
        assert chain["original"]["case_token"] == case["case_token"]
        assert chain["original"]["option_id"] == "recoverable_quarantine"
        assert chain["original"]["execution_receipt_hash"] == original.execution_receipt_hash
        assert [item["quarantine_hash"] for item in chain["continuations"]] == [
            continued.quarantine_hash
        ]
        latest = chain["continuations"][-1]
        assert latest["continued_from_quarantine_hash"] == original.quarantine_hash
        assert latest["rule"] == "recoverable_quarantine.unchanged_unexplained_move.v1"
        assert latest["current_evidence_hash"] != original.evidence_hash
        assert latest["unexplained_sessions"] == [day_one.date().isoformat()]
        assert len(live.session.task_control_registry.tasks()) == 2
        # The CLI operation and the page's route read the same disposition object.
        from alphalattice.interface.local_application.client import LocalResearchClient

        assert LocalResearchClient(workspace).request({"operation": "DATA_ISSUES"}) == issues
    # Reopened after the continuation: the disposition reads back, nothing resumes.
    with boot(day_two_clock) as live:
        assert not live.resumed_task_ids
        again = _json(live, "/api/workspace/data-issues")["continued_dispositions"]
        assert [item["continuations"][-1]["quarantine_hash"] for item in again] == [
            continued.quarantine_hash
        ]


def test_quarantine_decision_keeps_nominal_membership_and_continues_the_same_task(
    ten_sector, tmp_path, monkeypatch
):
    """Quarantine decision keeps nominal membership and continues the same task."""
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.panels.reader import (
        FeaturePanelReader,
        FeaturePanelReadRequest,
    )
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository

    workspace, boot = _quarantine_review_workspace(ten_sector, tmp_path, monkeypatch)
    market = MarketDataRepository(workspace)
    parent = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    assert market.universe_bootstrap("us-current-index-research") is not None
    with boot() as live:
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        task_id, request, case = _decide_quarantine(live, plan)
        resumed = _json(live, "/api/data-update/run", method="POST", payload=request)
        assert resumed["task_id"] == task_id
        live.dispatcher.drain_for_tests(timeout=300)
        status = _json(live, "/api/status?task_id=" + task_id)
        assert status["lifecycle"] == "SUCCEEDED", status
        readback = _json(live, "/api/data-update")
        assert readback["status"] == "PUBLISHED" and readback["current_input_failure"] is None
        assert readback["inputs"]["manifest_revision"] == parent.revision_sha256
        assert readback["inputs"]["panel_through"] == NOW.date().isoformat()
        current = market.current_quality_filtered_research_manifest(
            market_profile_id="us-current-index-research"
        )
        assert current.listings == parent.listings
        assert market.membership_events("us-current-index-research") == ()
        snapshot = PanelStateRepository(
            market.database, market_data=market
        ).feature_panel_snapshot_for_active("us-current-index-research")
        resolver = ArtifactResolver(workspace / "artifacts")
        payload = resolver.load_feature_panel_manifest(str(snapshot["manifest_uri"]))
        assert payload["safe_summary"]["membership"]["as_of_member_count"] == 60
        assert payload["safe_summary"]["quality_governance"]["admitted_listing_count"] == 59
        assert payload["safe_summary"]["membership"]["source_exclusions"]
        rows = pa.Table.from_batches(
            list(
                FeaturePanelReader(resolver).batches(
                    FeaturePanelReadRequest(
                        str(snapshot["manifest_uri"]), NOW.date(), NOW.date(), ("mom_126_21",)
                    )
                )
            )
        ).to_pandas()
        assert len(rows) == 60
        refused = rows["listing_id"].isin(case["listing_ids"])
        assert rows.loc[refused, "mom_126_21"].isna().all()
        assert rows.loc[~refused, "mom_126_21"].notna().sum() == 59
        assert len(live.session.task_control_registry.tasks()) == 1
        repeated = _json(live, "/api/data-update/run", method="POST", payload=request)
        assert repeated["status"] == "REUSED_EXACT" and repeated["task_id"] is None
    with boot() as live:
        reopened = _json(live, "/api/data-update")
        assert reopened["status"] == "PUBLISHED"
        assert reopened["inputs"]["manifest_revision"] == parent.revision_sha256
        assert not live.resumed_task_ids


def test_missing_quality_scope_refuses_publication_and_child_evidence_still_refuses(
    ten_sector, tmp_path, monkeypatch
):
    """Missing quality scope refuses publication and child evidence still refuses."""

    from alphalattice.control.product_host.composition.workspace import WorkspaceRuntime
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_quality_filtered_research_manifest,
    )

    workspace, boot = _quarantine_review_workspace(ten_sector, tmp_path, monkeypatch)
    market = MarketDataRepository(workspace)
    parent = market.current_quality_filtered_research_manifest(
        market_profile_id="us-current-index-research"
    )
    with boot() as live:
        plan = _json(live, "/api/data-update/plan", method="POST", payload={})
        before = _json(live, "/api/data-update")["inputs"]
        task_id, request, _case = _decide_quarantine(live, plan)

        def unavailable(self, **kwargs):
            raise ValueError("fixture quality scope unavailable")

        monkeypatch.setattr(PanelStateRepository, "panel_source_exclusions", unavailable)
        for _attempt in range(2):
            assert (
                _json(live, "/api/data-update/run", method="POST", payload=request)["task_id"]
                == task_id
            )
            live.dispatcher.drain_for_tests(timeout=180)
            status = _json(live, "/api/status?task_id=" + task_id)
            assert status["lifecycle"] == "BLOCKED", status
            assert status["latest_failure_code"] == "feature.source_eligibility_unavailable", status
        after = _json(live, "/api/data-update")
        assert after["status"] == "NO_UPDATE_PUBLICATION"
        assert after["inputs"]["manifest_revision"] == parent.revision_sha256
        assert after["inputs"]["panel_through"] == before["panel_through"]

        def refuse(self, source, target, *, requested_as_of, now):
            raise ValueError("source manifest has no reusable action-audit receipt: fixture")

        monkeypatch.setattr(MarketDataRepository, "bind_action_audit_receipts_to_manifest", refuse)
        child = build_quality_filtered_research_manifest(
            parent,
            eligible_listing_ids=tuple(item.listing_id for item in parent.listings[1:]),
        )
        with closing(
            WorkspaceRuntime.create(
                workspace=workspace,
                manifest=parent,
                provider=live.data_provider,
                artifact_root=workspace / "artifacts",
                application_controls=live.session,
            )
        ) as runtime:
            coordinator = runtime.maintenance_coordinator(
                readiness_gate=live.operations.data_update._readiness(market),
                clock=lambda: NOW,
            )
            assert not coordinator._bind_derived_manifest_evidence(
                parent,
                child,
                requested_as_of=NOW.date(),
                observed_at=NOW,
            )
        assert (
            market.current_quality_filtered_research_manifest(
                market_profile_id="us-current-index-research"
            )
            == parent
        )


def test_plan_and_binding_refusals_preserve_portfolio(qualified, tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="existing_data_required"):
        bind_existing_data_workspace(missing)
    assert not missing.exists()
    manifest = bind_existing_data_workspace(workspace)
    provider = recording_provider()
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    )
    service.start()
    try:
        with pytest.raises(RuntimeError, match="writer_already_owned"):
            bind_existing_data_workspace(workspace)
        plan = _json(service, "/api/data-update/plan", method="POST", payload={})
        market = MarketDataRepository(workspace)
        record = market.readiness.load("us-current-index-research")
        payload = record.__dict__.copy()
        payload["observed_at"] = payload.pop("updated_at")
        payload["last_checked_at"] = NOW
        market.readiness.save(**payload)
        refused = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": plan["plan_hash"]},
        )
        assert (
            refused["status"] == "REFUSED_INVALID_COMMAND"
            and "stale_plan" in refused["failure_code"]
        )
        assert provider.calls == [] and service.session.task_control_registry.tasks() == ()
        assert _json(service, "/api/plan", method="POST", payload={})["spec_hash"]
        # A frozen earlier manifest keeps the same strategy/default fields.
        assert (
            manifest.default_strategy_package_id
            == _manifest("data-update-qa").default_strategy_package_id
        )
    finally:
        service.stop()


def _formula(factor_id: str, formula: str):
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        formula_specification,
    )
    from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

    return formula_specification(
        FactorSpec(
            factor_id=factor_id,
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref="factor.formula",
            formula=formula,
            window_sessions=1,
            lag_sessions=0,
            return_convention="as declared",
            required_fields=("close_split_adjusted",),
            literature_sources=("research-local note",),
            minimum_observations=1,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
        )
    )


def _activate(workspace: Path, *specs) -> None:
    """A person's activations, recorded as the Host records them."""
    from alphalattice.control.product_host.research_authoring.feature_activations import (
        FeatureActivation,
        FeatureExtensionRegistry,
    )

    FeatureExtensionRegistry(
        active=tuple(
            FeatureActivation(
                factor_id=spec.factor_id,
                specification=spec,
                preprocessing_recipe="ROBUST_UNIVERSE_Z",
                feature_plan_hash="a" * 64,
                trial_id="b" * 64,
                methodology_hash="c" * 64,
                packet_hash="d" * 64,
                activated_at=NOW,
            )
            for spec in specs
        )
    ).write(workspace)


def _run_data_update(workspace: Path) -> None:
    manifest = bind_existing_data_workspace(workspace)
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=recording_provider(),
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    )
    service.start()
    try:
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        assert planned["status"] == "PLANNED", planned
        response = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": planned["plan_hash"]},
        )
        assert response["status"] == "ADMITTED", response
        service.dispatcher.drain_for_tests()
        status = _json(service, f"/api/status?task_id={response['task_id']}")
        # Whole, as text: a stop keeps its cause where saferepr would cut the dict (V602).
        assert status["lifecycle"] == "SUCCEEDED", json.dumps(status, indent=1, default=str)
    finally:
        service.stop()


def _feature_rows(workspace: Path, catalog) -> dict[tuple[str, date], dict[str, object]]:
    from alphalattice.foundation.feature_engine.storage.repositories import (
        FeatureStateRepository,
    )

    market = MarketDataRepository(workspace)
    readiness = market.readiness.load("us-current-index-research")
    listings = tuple(
        item.listing_id
        for item in market.load_universe_manifest(readiness.active_manifest_id).listings
    )
    rows = FeatureStateRepository(workspace, installed_catalog=catalog).feature_rows(
        listing_ids=listings,
        catalog_hash=catalog.binding.catalog_hash,
        start=HISTORY_START,
        end=NOW.date(),
    )
    assert isinstance(rows, list)
    return {(str(row["listing_id"]), row["session_date"]): row for row in rows}


def _stored_catalogs(workspace: Path) -> dict[str, int]:
    connection = MarketDataRepository(workspace).database.connect(read_only=True)
    try:
        return dict(
            connection.execute(
                "SELECT catalog_hash, count(*) FROM feature_daily_current GROUP BY 1"
            ).fetchall()
        )
    finally:
        connection.close()


def _same_value(old: object, new: object) -> bool:
    return (old is None and new is None) or old == new or (old != old and new != new)


def test_an_activation_computes_its_column_and_the_update_carries_every_other_value(
    qualified, tmp_path, monkeypatch
):
    """An activation computes its column and the update carries every other value."""

    from alphalattice.control.product_host.research_authoring.feature_activations import (
        workspace_feature_catalog,
    )
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
    from alphalattice.foundation.feature_engine.catalog.layer import FeatureCatalogLayer
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
        FeatureClosureLedger,
    )

    layered, whole = tmp_path / "layered", tmp_path / "whole"
    shutil.copytree(qualified, layered)
    shutil.copytree(qualified, whole)
    shipped = FeatureCatalog.load()
    before = _feature_rows(layered, shipped)
    assert before
    spec = _formula("formula_reversal_5", "close / lag(close, 5) - 1")
    _activate(layered, spec)
    _activate(whole, spec)
    extended = workspace_feature_catalog(layered)
    layer = FeatureCatalogLayer.over(extended)
    assert layer.layered and layer.base.binding.catalog_hash == shipped.binding.catalog_hash
    column = layer.columns[0].binding.catalog_hash
    _run_data_update(layered)
    with monkeypatch.context() as unlayered:
        # The whole build an unlayered store makes, for the equivalence below.
        unlayered.setattr(
            FeatureCatalogLayer,
            "over",
            classmethod(lambda cls, catalog, base=None: cls(catalog=catalog, base=catalog)),
        )
        _run_data_update(whole)

    held = _feature_rows(layered, shipped)
    assert all(held[key]["row_hash"] == row["row_hash"] for key, row in before.items())
    stored = _stored_catalogs(layered)
    assert stored == {shipped.binding.catalog_hash: len(held), column: len(held)}, stored
    assert _stored_catalogs(whole) == {extended.binding.catalog_hash: len(held)}

    composed = _feature_rows(layered, extended)
    built = _feature_rows(whole, extended)
    assert composed.keys() == built.keys() == held.keys()
    for key, row in built.items():
        assert json.loads(str(composed[key]["input_cutoffs_json"])) == json.loads(
            str(row["input_cutoffs_json"])
        ), key
        for factor_id in extended.factor_ids:
            assert _same_value(row[factor_id], composed[key][factor_id]), (factor_id, key)
    for key, row in before.items():
        for factor_id in shipped.factor_ids:
            assert _same_value(row[factor_id], composed[key][factor_id]), (factor_id, key)
    computed = [row["formula_reversal_5"] for row in composed.values()]
    assert sum(value is not None and value == value for value in computed) > len(computed) // 2

    panel = _active_panel_manifest(layered)
    assert panel["safe_summary"]["lineage"]["catalog_hash"] == extended.binding.catalog_hash
    assert panel["panel_content_hash"] == _active_panel_manifest(whole)["panel_content_hash"]
    ledger = FeatureClosureLedger(
        PanelClosureArtifactStore(ArtifactResolver(layered / "artifacts"))
    )
    binding = ledger.panel_binding(str(panel["snapshot_hash"]))
    assert binding is not None
    assert binding.closure_head_hash == ledger.require_head(shipped.binding.catalog_hash).head_hash
    assert [(item.catalog_hash, item.closure_head_hash) for item in binding.column_closures] == [
        (column, ledger.require_head(column).head_hash)
    ]


def test_a_layer_grows_by_an_activation_and_shrinks_by_a_deactivation(qualified, tmp_path):
    """requirement (V92): a second activation computes its own column and holds the first; a
    deactivation drops its column catalog's rows and computes nothing, the shipped catalog's
    rows held throughout, and each update publishes the Panel of the catalog it installed."""

    from alphalattice.control.product_host.research_authoring.feature_activations import (
        workspace_feature_catalog,
    )
    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
    from alphalattice.foundation.feature_engine.catalog.layer import FeatureCatalogLayer

    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    shipped = FeatureCatalog.load()
    reversal = _formula("formula_reversal_5", "close / lag(close, 5) - 1")
    drift = _formula("formula_drift_10", "close / lag(close, 10) - 1")
    _activate(workspace, reversal)
    _run_data_update(workspace)
    first = FeatureCatalogLayer.over(workspace_feature_catalog(workspace))
    base = _feature_rows(workspace, shipped)
    reversal_rows = {
        key: row["row_hash"] for key, row in _feature_rows(workspace, first.columns[0]).items()
    }

    _activate(workspace, reversal, drift)
    _run_data_update(workspace)
    grown = FeatureCatalogLayer.over(workspace_feature_catalog(workspace))
    assert first.columns[0].binding.catalog_hash in grown.part_hashes
    assert {
        key: row["row_hash"] for key, row in _feature_rows(workspace, first.columns[0]).items()
    } == reversal_rows
    assert {key: row["row_hash"] for key, row in _feature_rows(workspace, shipped).items()} == {
        key: row["row_hash"] for key, row in base.items()
    }
    assert _stored_catalogs(workspace) == {part: len(base) for part in grown.part_hashes}
    assert (
        _active_panel_manifest(workspace)["safe_summary"]["lineage"]["catalog_hash"]
        == grown.catalog.binding.catalog_hash
    )

    _activate(workspace)
    _run_data_update(workspace)
    assert _stored_catalogs(workspace) == {shipped.binding.catalog_hash: len(base)}
    assert {key: row["row_hash"] for key, row in _feature_rows(workspace, shipped).items()} == {
        key: row["row_hash"] for key, row in base.items()
    }
    assert (
        _active_panel_manifest(workspace)["safe_summary"]["lineage"]["catalog_hash"]
        == shipped.binding.catalog_hash
    )


def test_transient_source_failure_resumes_captured_update(qualified, tmp_path):
    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    provider = recording_provider()
    provider.unavailable = True
    # An explicit recorded source-readiness signal, not a reclassification of
    # malformed Yahoo bars or proof that Yahoo exposes this finality signal.
    provider.unavailable_code = "data.daily_bar_not_final"
    clock = [NOW]
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: clock[0],
    )
    service.start()
    try:
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        sent = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": planned["plan_hash"]},
        )
        service.dispatcher.drain_for_tests()
        status = _json(service, f"/api/status?task_id={sent['task_id']}")
        assert status["lifecycle"] == "DEFERRED", status.get("worker_failure") or status
        issues = _json(service, "/api/workspace/data-issues")
        assert issues["status"] == "DATA_TASK_WAITING_OR_BLOCKED" and not issues["issues"]
        calls = list(provider.calls)
        early = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": planned["plan_hash"]},
        )
        assert "retry_not_due" in early["failure_code"] and provider.calls == calls
        assert _json(service, "/api/workspace/data-issues") == issues
        assert _json(service, "/api/data-update")["receipt"] is None
        provider.unavailable = False
        clock[0] = NOW + timedelta(minutes=5)
        resumed = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": planned["plan_hash"]},
        )
        assert resumed["task_id"] == sent["task_id"]
        service.dispatcher.drain_for_tests()
        final = _json(service, f"/api/status?task_id={sent['task_id']}")
        assert final["lifecycle"] == "SUCCEEDED", final
        assert len(service.session.task_control_registry.tasks()) == 1
    finally:
        service.stop()


def test_a_failed_backup_stops_the_next_update_by_name_and_its_rerun_resumes(
    qualified, tmp_path, monkeypatch
):
    """requirement: the next update's first stage retries the backup the last one deferred; a
    failure stops it by name as recoverable, and its plan run again resumes the same Task."""
    from alphalattice.control.product_host.storage.backup import defer_automatic_backup

    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    wall = tmp_path / "wall"
    wall.write_text("a file, not a directory", encoding="utf-8")
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(wall / "store"))
    defer_automatic_backup(workspace, at=NOW)
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=recording_provider(),
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    )
    service.start()
    try:
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        run = {"update_plan_hash": planned["plan_hash"]}
        sent = _json(service, "/api/data-update/run", method="POST", payload=run)
        service.dispatcher.drain_for_tests()
        stopped = _json(service, f"/api/status?task_id={sent['task_id']}")
        assert stopped["lifecycle"] == "BLOCKED", stopped
        assert stopped["failure_code"] == "workspace_data_update.archive_failed"
        monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(tmp_path / "backups"))
        resumed = _json(service, "/api/data-update/run", method="POST", payload=run)
        assert resumed["task_id"] == sent["task_id"]
        service.dispatcher.drain_for_tests()
        final = _json(service, f"/api/status?task_id={sent['task_id']}")
        assert final["lifecycle"] == "SUCCEEDED", final
        assert list((tmp_path / "backups").rglob("generations/*.json"))
    finally:
        service.stop()


def test_a_deferred_feature_build_is_built_again_once_its_retry_is_due(
    qualified, tmp_path, monkeypatch
):
    """The update's cycle runs the Feature build as its step (W10, V102): a deferred build records
    its retry time on the cycle, the Task waits for it, and the next run builds again, where the
    retired Feature registry answered its deferred Task's projection and nothing resumed it."""
    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    clock = [NOW]
    retry_at = NOW + timedelta(minutes=2)
    builds: list[str] = []
    build = FeatureFoundationService.build

    def deferred_once(self, request, **kwargs):
        builds.append(request.request_hash)
        if len(builds) > 1:
            return build(self, request, **kwargs)
        # The cycle holds nothing across the build: another writer's short transaction runs.
        free = Event()
        probe = Thread(target=lambda: self.mutation_gate.run(free.set), daemon=True)
        probe.start()
        assert free.wait(timeout=5), "the cycle held the mutation gate across the Feature build"
        probe.join(timeout=5)
        return FeatureBuildOutcome(
            FeatureBuildStatus.DEFERRED,
            request.request_hash,
            None,
            {},
            retry_after_at=retry_at,
            failure_code="feature.sector_rate_limited",
        )

    monkeypatch.setattr(FeatureFoundationService, "build", deferred_once)
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=recording_provider(),
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: clock[0],
    )
    service.start()
    try:
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        run = {"update_plan_hash": planned["plan_hash"]}
        sent = _json(service, "/api/data-update/run", method="POST", payload=run)
        service.dispatcher.drain_for_tests()
        status = _json(service, f"/api/status?task_id={sent['task_id']}")
        assert status["lifecycle"] == "DEFERRED", status
        assert len(builds) == 1
        early = _json(service, "/api/data-update/run", method="POST", payload=run)
        assert "retry_not_due" in early["failure_code"] and len(builds) == 1
        clock[0] = retry_at + timedelta(minutes=1)
        resumed = _json(service, "/api/data-update/run", method="POST", payload=run)
        assert resumed["task_id"] == sent["task_id"]
        service.dispatcher.drain_for_tests()
        final = _json(service, f"/api/status?task_id={sent['task_id']}")
        assert final["lifecycle"] == "SUCCEEDED", final
        assert len(builds) >= 2 and builds[1] == builds[0]
    finally:
        service.stop()


def test_a_feature_build_that_failed_names_its_cause_beside_its_code(
    qualified, tmp_path, monkeypatch
):
    """A feature build that failed names its cause beside its code."""
    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    failure = {
        "stage": "base_feature_materialization",
        "exception_type": "MemoryError",
        "listing_id": "listing-a",
        "first_target_session": "2026-09-01",
        "last_target_session": NOW.date().isoformat(),
        "detail": "",
    }

    def exhausted(self, request, **kwargs):
        return FeatureBuildOutcome(
            FeatureBuildStatus.BLOCKED,
            request.request_hash,
            None,
            {"materialized_listings": 0, "materialization_failure": failure},
            failure_code="feature.materialization_failed",
        )

    monkeypatch.setattr(FeatureFoundationService, "build", exhausted)
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=recording_provider(),
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    )
    service.start()
    try:
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        run = {"update_plan_hash": planned["plan_hash"]}
        sent = _json(service, "/api/data-update/run", method="POST", payload=run)
        service.dispatcher.drain_for_tests()
        status = _json(service, f"/api/status?task_id={sent['task_id']}")
        assert status["lifecycle"] == "BLOCKED", status
        assert status["failure_code"] == "feature.materialization_failed"
        cause = {
            "exception_type": "MemoryError",
            "detail": "",
            "step": "base_feature_materialization",
            "unit": "listing-a",
            "first_session": "2026-09-01",
            "last_session": NOW.date().isoformat(),
        }
        assert status["failure_cause"] == cause
        assert _json(service, "/api/data-update")["failure_cause"] == cause
    finally:
        service.stop()


def test_honoured_cancel_marks_the_cycle_and_a_new_plan_reuses_its_listing_progress(
    qualified, tmp_path, monkeypatch
):
    """Honoured cancel marks the cycle and a new plan reuses its listing progress."""

    from alphalattice.control.data_platform.maintenance.contracts import (
        MaintenanceStatus,
    )
    from alphalattice.control.product_host.maintenance import (
        data_update,
    )

    monkeypatch.setattr(data_update, "_MAINTENANCE_CYCLE_LISTINGS", 1)

    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    provider = recording_provider()
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    )
    service.start()
    try:
        owner = service.operations.data_update
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        plan = owner.prepare(planned["plan_hash"])
        checks = {"count": 0}

        def cancel_after_first_run() -> bool:
            checks["count"] += 1
            return checks["count"] > 1

        result = owner.execute_step(plan, "maintain_data_feature", cancelled=cancel_after_first_run)
        assert result.disposition.value == "CANCELLED"
        cycle = owner._cycle(plan)
        assert cycle.status is MaintenanceStatus.CANCELLED
        assert cycle.failure_code == "workspace_maintenance.cancelled_at_safe_checkpoint"
        assert cycle.phase.value == "market_data" and cycle.retry_after_at is None
        fetched_before_cancel = {symbol for symbols, _s, _e in provider.calls for symbol in symbols}
        assert fetched_before_cancel
        # Recording the cancellation twice adds nothing; a completed cycle cannot be cancelled.
        assert owner._registry().cancel(cycle.cycle_id, observed_at=NOW) == cycle
        # A legitimate new plan continues from the listing work the cancelled cycle made.
        calls = len(provider.calls)
        again = _json(service, "/api/data-update/plan", method="POST", payload={})
        run = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": again["plan_hash"]},
        )
        service.dispatcher.drain_for_tests()
        status = _json(service, f"/api/status?task_id={run['task_id']}")
        assert status["lifecycle"] == "SUCCEEDED", status
        fetched_after = {symbol for symbols, _s, _e in provider.calls[calls:] for symbol in symbols}
        assert fetched_after.isdisjoint(fetched_before_cancel - {"SPY"})
        # Under this frozen clock the new plan binds the same request, so its
        # cycle is the cancelled one resumed; under a moving clock it is a new
        # cycle over the same maintenance operation. Either way the listing
        # progress was reused, and a completed cycle cannot be cancelled.
        resumed = owner._cycle(owner.prepare(again["plan_hash"]))
        assert resumed.status is MaintenanceStatus.COMPLETED
        with pytest.raises(ValueError, match="already complete"):
            owner._registry().cancel(resumed.cycle_id, observed_at=NOW)
    finally:
        service.stop()


def test_update_authority_and_missing_input_refuse_before_work(qualified, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "1")
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        clock=lambda: NOW,
        data_source_loader=unchanged_membership_source(),
    )
    service.start()
    manifest_path = workspace / "research-workspace.json"
    original_manifest = manifest_path.read_bytes()
    try:
        planned = _json(service, "/api/data-update/plan", method="POST", payload={})
        refused = _json(
            service,
            "/api/data-update/run",
            method="POST",
            payload={"update_plan_hash": planned["plan_hash"]},
        )
        assert "source_access_not_admitted" in refused["failure_code"]
        app = service.operations.data_update
        from alphalattice.control.data_platform.maintenance.contracts import WorkspaceDataUpdatePlan

        altered = app.last_plan.model_dump(mode="json")
        altered["request"]["target_market_session"] = "2026-08-04"
        with pytest.raises(ValueError, match="identity_invalid"):
            WorkspaceDataUpdatePlan.model_validate(altered)

        payload = json.loads(original_manifest)
        payload["data_update"]["profile_file_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            assert (
                _json(service, "/api/data-update/plan", method="POST", payload={})["status"]
                == "REFUSED"
            )
        finally:
            manifest_path.write_bytes(original_manifest)

        panel = json.loads(
            (
                workspace
                / "artifacts/feature-panel/manifests"
                / f"{planned['inputs']['panel_hash']}.json"
            ).read_text()
        )
        chunk = (
            workspace
            / "artifacts/feature-panel/chunks"
            / f"{panel['chunks'][0]['chunk_hash']}.parquet"
        )
        held = chunk.with_suffix(".held")
        chunk.rename(held)
        try:
            missing = _json(service, "/api/data-update/plan", method="POST", payload={})
            assert (
                missing.items()
                >= {
                    "status": "REFUSED",
                    "failure_code": "workspace_data_update.artifact_missing",
                }.items()
            )
            assert (
                _json(service, "/api/data-update")["current_input_failure"]
                == missing["failure_code"]
            )
        finally:
            held.rename(chunk)
        market = MarketDataRepository(workspace)
        record = market.readiness.load("us-current-index-research")
        state = record.__dict__.copy()
        state["observed_at"] = state.pop("updated_at")
        market.readiness.save(**{**state, "status": "MANIFEST_UPDATE_PENDING"})
        blocked = _json(service, "/api/data-update/plan", method="POST", payload={})
        assert blocked["next_action"] == "RESOLVE_EXISTING_WORKSPACE_TRANSITION"
        assert service.session.task_control_registry.tasks() == ()
        market.readiness.save(**state)

        # A setup rebind must not replace authority below an admitted, unrun Task.
        app.provider = recording_provider()
        app.plan()
        app.admit(app.last_plan)
        assert app.provider.calls == []
    finally:
        service.stop()
    with pytest.raises(ValueError, match="unsettled_tasks"):
        bind_existing_data_workspace(workspace)
    assert manifest_path.read_bytes() == original_manifest


def test_published_update_survives_process_death(qualified, tmp_path):
    from tests.portfolio_strategy_lab.process_death import forced_process

    workspace = tmp_path / "workspace"
    shutil.copytree(qualified, workspace)
    manifest = bind_existing_data_workspace(workspace)
    with forced_process(
        workspace, __name__, "publish_update_receipt", module_dir=Path(__file__).parent
    ) as stopped:
        pass
    provider = recording_provider()
    service = LocalPortfolioWebSession(
        workspace=workspace,
        workspace_manifest=manifest,
        resolver=_Resolver(_resolved()),
        data_provider=provider,
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    )
    service.start()
    try:
        service.dispatcher.drain_for_tests()
        status = _json(service, f"/api/status?task_id={stopped['task_id']}")
        assert status["lifecycle"] == "SUCCEEDED", status
        assert provider.calls == []
        assert len(service.session.task_control_registry.tasks()) == 1
        assert _json(service, "/api/data-update")["receipt"] is not None
    finally:
        service.stop()


def _crash_child(workspace: str, phase: str) -> None:
    import os
    from threading import Event

    from alphalattice.control.product_host.composition.research_workspace import (
        read_research_workspace_manifest,
    )

    root = Path(workspace)
    service = LocalPortfolioWebSession(
        workspace=root,
        workspace_manifest=read_research_workspace_manifest(root),
        resolver=_Resolver(_resolved()),
        data_provider=recording_provider(),
        data_source_loader=unchanged_membership_source(SYMBOLS),
        clock=lambda: NOW,
    )
    original = WorkspaceDataUpdateApplication.execute_stage

    def interrupted(self, **kwargs):
        result = original(self, **kwargs)
        if kwargs["work_item"].stage_id == phase:
            print(
                "CRASH_BOUNDARY:"
                + json.dumps({"pid": os.getpid(), "task_id": str(kwargs["task"].task_id)}),
                flush=True,
            )
            Event().wait()
        return result

    WorkspaceDataUpdateApplication.execute_stage = interrupted
    service.start()
    plan = _json(service, "/api/data-update/plan", method="POST", payload={})
    _json(
        service,
        "/api/data-update/run",
        method="POST",
        payload={"update_plan_hash": plan["plan_hash"]},
    )
    service.dispatcher.drain_for_tests()
    service.stop()
