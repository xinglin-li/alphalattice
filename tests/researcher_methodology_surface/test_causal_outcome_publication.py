"""One outcome snapshot carries one seal, however often the Host re-derives it.

The source watermark the publisher keys exact reuse on is a counter over source
revisions, not over the rows a snapshot consumed. It moves on a correction that
changes nothing the outcome reads -- on the real 2026-09-11 workspace, one
dividend observed twice by two daily-update attempts and settling on its first
payload -- and the next publication then re-derives the identical snapshot.
Before this case the publisher minted a second binding and seal marker for it,
and every reader thereafter refused the snapshot as ``does not resolve one
method seal``: the successor research input could not be published, by the same
Task that had just written the duplicate.

The private golden copy is mutated on purpose; the shared session workspace is
never touched.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.causal_outcomes.execution.publication import (
    CausalExecutionOutcomePublisher,
    PublishedCausalExecutionOutcome,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.market_data_ops.sources.contracts import CorporateActionEvent
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from tests.researcher_methodology_surface.real_workspace import (
    HISTORY_START,
    OBSERVED_AT,
    RealRiskWorkspace,
    build_real_risk_workspace,
)


def test_risk_publication_reads_only_its_declared_contiguous_window(tmp_path, monkeypatch):
    from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
    from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
    from alphalattice.investment.risk_research.surfaces.returns import (
        CausalRiskReturnReader,
        CausalRiskReturnSurfacePublisher,
        RiskReturnSurfaceError,
    )

    axis = tuple(date(2020, 1, 1) + timedelta(days=i) for i in range(330))
    scope = axis[:317]
    digest = "1" * 64
    manifest = SimpleNamespace(
        revision_sha256=digest,
        profile=SimpleNamespace(market_profile_id="fixture", daily_price_basis="split_adjusted"),
    )
    reads = []

    def bars(listing, *, start, through, **_kwargs):
        assert listing == "A"  # B's unrelated unavailable history must never be read.
        assert start == scope[0] and through == scope[-1]
        reads.append((start, through))
        return tuple(
            RawDailyBar(
                listing_id=listing,
                provider="fixture",
                session_date=day,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.0 + i,
                volume=1000,
            )
            for i, day in enumerate(scope)
        )

    market = SimpleNamespace(
        load_universe_manifest_revision=lambda _: manifest,
        listing_scope=lambda *a, **k: tuple(
            SimpleNamespace(listing_id=listing, symbol=listing) for listing in k["listing_ids"]
        ),
        execution_source_watermark=lambda *a, **k: {"watermark_hash": digest},
        _connect=lambda **k: SimpleNamespace(close=lambda: None),
        raw_bars=bars,
        actions=lambda *a, **k: (),
    )
    resolver = ArtifactResolver(tmp_path / "artifacts")
    monkeypatch.setattr(
        resolver,
        "load_feature_panel_manifest",
        lambda _: {
            "snapshot_hash": digest,
            "safe_summary": {"lineage": {"manifest_revision": digest}},
        },
    )
    monkeypatch.setattr(FeaturePanelReader, "listing_ids", lambda *a: ("A",))
    publisher = CausalRiskReturnSurfacePublisher(
        store=market,
        resolver=resolver,
        artifact_root=resolver.root,
        mutation_gate=WorkspaceMutationGate(),
    )
    monkeypatch.setattr(
        publisher.index_service,
        "obtain",
        lambda _: (
            SimpleNamespace(sessions=tuple(SimpleNamespace(session_date=s) for s in axis)),
            None,
            False,
        ),
    )
    surface, _ = publisher.publish(
        panel_manifest_ref="fixture",
        market_profile_id="fixture",
        required_sessions=scope,
    )
    assert reads == [(scope[0], scope[-1])]
    assert surface.formation_count == 316 and surface.last_formation_session == scope[-1]
    assert CausalRiskReturnReader(resolver.root).read_sessions(surface, scope[-2:]).shape == (2, 1)
    with pytest.raises(RiskReturnSurfaceError, match="return_session_axis_invalid"):
        publisher.publish(
            panel_manifest_ref="fixture",
            market_profile_id="fixture",
            required_sessions=scope[:100] + scope[101:],
        )
    assert len(reads) == 1  # Invalid scope is refused before source reads/calculation.
    monkeypatch.setattr(FeaturePanelReader, "listing_ids", lambda *a: ("A", "B"))
    selected, _ = publisher.publish(
        panel_manifest_ref="fixture",
        market_profile_id="fixture",
        required_sessions=scope,
        required_listing_ids=("A",),
    )
    assert selected == surface  # Same actual scope and source preserve its identity.
    for invalid in ((), ("B", "A"), ("UNKNOWN",)):
        with pytest.raises(RiskReturnSurfaceError, match="return_listing_scope_invalid"):
            publisher.publish(
                panel_manifest_ref="fixture",
                market_profile_id="fixture",
                required_sessions=scope,
                required_listing_ids=invalid,
            )
    assert len(reads) == 2


def _publish(workspace: RealRiskWorkspace, *, hours: int) -> PublishedCausalExecutionOutcome:
    with patch.object(
        MarketDataRepository,
        "current_quality_filtered_research_manifest",
        side_effect=AssertionError("a frozen Panel cannot read a later current manifest"),
    ):
        return CausalExecutionOutcomePublisher(
            store=MarketDataRepository(workspace.workspace),  # type: ignore[arg-type]
            resolver=ArtifactResolver(workspace.artifact_root),
            artifact_root=workspace.artifact_root,
            mutation_gate=WorkspaceMutationGate(),
        ).publish_daily(
            panel_manifest_ref=workspace.panel_manifest_ref,
            completed_at=OBSERVED_AT + timedelta(hours=hours),
        )


def test_a_re_derived_snapshot_keeps_its_one_seal_and_reuses_exactly_afterwards(
    tmp_path: Path,
) -> None:
    """requirement: a moved watermark over unchanged rows re-derives, never re-seals.

    The dividend recorded here is dated before the first formation session, so no
    schedule interval reaches it and no published row or action lineage moves;
    only the store's action counters do. The second publication must read the
    sources again, land on the same snapshot, keep the first binding and seal
    marker as the snapshot's sole authority, and leave a receipt for the new
    watermark so the third publication is an exact reuse without a source read.
    """

    workspace = build_real_risk_workspace(tmp_path / "workspace")
    first = _publish(workspace, hours=0)
    assert first.receipt.action == "PUBLISHED"
    snapshot_hash = first.manifest.snapshot_hash
    assert first.manifest.development_chunks[0].first_formation_session > HISTORY_START

    store = MarketDataRepository(workspace.workspace)
    listing = workspace.manifest.listings[0]
    connection = store._connect()
    try:
        outcome = store._upsert_action(
            connection,
            CorporateActionEvent(
                listing.listing_id,
                workspace.manifest.profile.provider,
                HISTORY_START,
                "CASH_DIVIDEND",
                cash_amount=0.25,
            ),
            observed_at=(OBSERVED_AT + timedelta(minutes=30)).astimezone(UTC).replace(tzinfo=None),
        )
    finally:
        connection.close()
    assert outcome == "INSERTED"
    moved = workspace.freshness_probe(through=first.manifest.market_as_of)
    assert moved != first.method_binding.source_watermark_hash

    second = _publish(workspace, hours=1)
    assert second.receipt.action == "PUBLISHED"
    assert second.receipt.raw_bar_payload_rows_read == first.receipt.raw_bar_payload_rows_read > 0
    assert second.receipt.source_watermark_hash == moved
    assert second.manifest.snapshot_hash == snapshot_hash
    assert second.method_binding == first.method_binding
    assert second.method_seal_marker == first.method_seal_marker

    reader = CausalExecutionOutcomeDevelopmentReader(workspace.artifact_root)
    bindings = [b for b in reader.artifacts.method_bindings() if b.snapshot_hash == snapshot_hash]
    seals = [s for s in reader.artifacts.method_seal_markers() if s.snapshot_hash == snapshot_hash]
    assert [b.binding_hash for b in bindings] == [first.method_binding.binding_hash]
    assert [s.seal_marker_hash for s in seals] == [first.method_seal_marker.seal_marker_hash]
    resolved = reader.resolve_method_seal(snapshot_hash)
    assert resolved.disposition == "METHOD_BOUND"
    assert resolved.binding is not None
    assert resolved.binding.binding_hash == first.method_binding.binding_hash
    receipts = sorted(
        (r.source_watermark_hash, r.action)
        for r in reader.artifacts.receipts()
        if r.snapshot_hash == snapshot_hash
    )
    assert receipts == sorted(
        [(first.receipt.source_watermark_hash, "PUBLISHED"), (moved, "PUBLISHED")]
    )

    third = _publish(workspace, hours=2)
    assert third.receipt.action == "REUSED_EXACT"
    assert third.receipt.raw_bar_payload_rows_read == 0
    assert third.manifest.snapshot_hash == snapshot_hash
    assert third.method_binding == first.method_binding


def test_a_superseded_panel_with_an_index_on_disk_is_refused_before_any_read(
    tmp_path: Path,
) -> None:
    """regression: the coverage axis admits the snapshot; an index on disk admits nothing.

    The first publication built the Panel's semantic index. The Panel is
    then superseded. The next publication must be refused by the axis
    reader's admission -- before the index is consulted, before a source
    row is read, and without a receipt or any other artifact written --
    exactly as the row-validating axis refused it before. The snapshot
    already published over the Panel stays readable through the outcome
    readers: reading history is not admitting new research. A snapshot
    that lost its Gateway admission, and a chunk whose row count the
    manifest misstates, are refused by the same owner.
    """

    from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
    from alphalattice.foundation.feature_engine.storage.repositories import (
        PanelStateRepository,
    )

    workspace = build_real_risk_workspace(tmp_path / "workspace")
    resolver = ArtifactResolver(workspace.artifact_root)
    first = _publish(workspace, hours=0)
    assert first.receipt.action == "PUBLISHED"
    assert (
        resolver.find_feature_panel_semantic_index(
            panel_snapshot_hash=workspace.panel_snapshot_hash
        )
        is not None
    )
    store = MarketDataRepository(workspace.workspace)
    panel_state = PanelStateRepository(store.database, market_data=store)
    panel_state.set_feature_panel_snapshot_lifecycle(
        snapshot_hash=workspace.panel_snapshot_hash,
        lifecycle="SUPERSEDED",
        reason="not_current_active_panel",
        observed_at=OBSERVED_AT + timedelta(hours=1),
    )
    resolver.publish_feature_panel_lifecycle_projection(
        snapshots=panel_state.feature_panel_snapshot_lifecycles()
    )
    files = sorted(path for path in Path(workspace.artifact_root).rglob("*") if path.is_file())
    with (
        patch.object(
            MarketDataRepository, "raw_bars", side_effect=AssertionError("a source row was read")
        ),
        pytest.raises(ValueError, match="requires an ACTIVE Feature Panel snapshot"),
    ):
        _publish(workspace, hours=2)
    assert sorted(path for path in Path(workspace.artifact_root).rglob("*") if path.is_file()) == (
        files
    )
    reader = CausalExecutionOutcomeDevelopmentReader(workspace.artifact_root)
    assert reader.resolve_method_seal(first.manifest.snapshot_hash).disposition == "METHOD_BOUND"

    panel_state.set_feature_panel_snapshot_lifecycle(
        snapshot_hash=workspace.panel_snapshot_hash,
        lifecycle="QUARANTINED",
        reason="fixture",
        observed_at=OBSERVED_AT + timedelta(hours=3),
    )
    resolver.publish_feature_panel_lifecycle_projection(
        snapshots=panel_state.feature_panel_snapshot_lifecycles()
    )
    with pytest.raises(ValueError, match="requires an ACTIVE Feature Panel snapshot"):
        FeaturePanelReader(resolver).listing_axis(workspace.panel_manifest_ref)

    active = build_real_risk_workspace(tmp_path / "active")
    active_resolver = ArtifactResolver(active.artifact_root)
    panel = active_resolver.load_feature_panel_manifest(active.panel_manifest_ref)
    summary = dict(panel["safe_summary"])
    unadmitted = {
        **panel,
        "safe_summary": {
            **summary,
            "quality_governance": {**summary["quality_governance"], "gateway_qualified": False},
        },
    }
    with (
        patch.object(ArtifactResolver, "load_feature_panel_manifest", return_value=unadmitted),
        pytest.raises(ValueError, match="Feature Input Gateway admission"),
    ):
        FeaturePanelReader(active_resolver).listing_axis(active.panel_manifest_ref)
    chunks = [dict(item) for item in panel["chunks"]]
    chunks[0]["row_count"] = int(chunks[0]["row_count"]) + 1
    misstated = {**panel, "chunks": chunks}
    with (
        patch.object(ArtifactResolver, "load_feature_panel_manifest", return_value=misstated),
        pytest.raises(ValueError, match="chunk row count mismatch"),
    ):
        FeaturePanelReader(active_resolver).listing_axis(active.panel_manifest_ref)
    assert FeaturePanelReader(active_resolver).listing_axis(active.panel_manifest_ref) == (
        FeaturePanelReader(active_resolver).listing_ids(active.panel_manifest_ref)
    )


def test_parallel_derivation_publishes_the_identical_snapshot_and_bytes(tmp_path: Path) -> None:
    """requirement: worker-process derivation changes how long a publication takes, nothing it says.

    The same product-built workspace is published twice from two fresh
    copies: once with every listing derived in this process, once with the
    parallel path forced (two workers, the size threshold lowered). The
    snapshot hash, every chunk's logical identity and the bytes of every
    chunk file must be the same, and one listing's unit of derivation must
    answer the same rows, hashes and source bindings from a worker as from
    this process.
    """

    from concurrent.futures import ProcessPoolExecutor

    from alphalattice.control.product_host.research_authoring.factor_inputs import file_digest
    from alphalattice.foundation.causal_outcomes.execution import publication as owner
    from alphalattice.foundation.causal_outcomes.execution.compile import (
        ListingRowDerivation,
        _derive_in_worker,
        _initialize_derivation_worker,
    )
    from alphalattice.foundation.causal_outcomes.execution.methods import (
        build_one_session_recipe,
        resolve_schedule_points,
    )

    serial_workspace = build_real_risk_workspace(tmp_path / "serial")
    parallel_workspace = build_real_risk_workspace(tmp_path / "parallel")

    def publish(workspace: RealRiskWorkspace, *, workers: int) -> PublishedCausalExecutionOutcome:
        return CausalExecutionOutcomePublisher(
            store=MarketDataRepository(workspace.workspace),  # type: ignore[arg-type]
            resolver=ArtifactResolver(workspace.artifact_root),
            artifact_root=workspace.artifact_root,
            mutation_gate=WorkspaceMutationGate(),
            derivation_workers=workers,
        ).publish_daily(panel_manifest_ref=workspace.panel_manifest_ref, completed_at=OBSERVED_AT)

    serial = publish(serial_workspace, workers=1)
    with patch.object(owner, "PARALLEL_DERIVATION_ROW_THRESHOLD", 1):
        parallel = publish(parallel_workspace, workers=2)
    assert serial.receipt.action == parallel.receipt.action == "PUBLISHED"
    assert parallel.manifest.snapshot_hash == serial.manifest.snapshot_hash
    assert parallel.manifest == serial.manifest
    assert parallel.method_binding == serial.method_binding
    serial_reader = CausalExecutionOutcomeDevelopmentReader(serial_workspace.artifact_root)
    parallel_reader = CausalExecutionOutcomeDevelopmentReader(parallel_workspace.artifact_root)
    chunks = serial.manifest.development_chunks
    assert chunks and parallel.manifest.development_chunks == chunks
    for chunk in chunks:
        assert file_digest(parallel_reader.artifacts.resolve_chunk(chunk)) == file_digest(
            serial_reader.artifacts.resolve_chunk(chunk)
        )
    # One listing, one unit: derived here and in a worker process alike.
    store = MarketDataRepository(serial_workspace.workspace)
    resolver = ArtifactResolver(serial_workspace.artifact_root)
    panel = resolver.load_feature_panel_manifest(serial_workspace.panel_manifest_ref)
    index, _ref, _backfilled = owner.FeaturePanelSemanticIndexService(resolver).obtain(
        serial_workspace.panel_manifest_ref
    )
    sessions = tuple(value.session_date for value in index.sessions)
    calendar = owner.materialize_calendar_schedule(
        ("XNYS", "XNAS"), start=sessions[0], end=sessions[-1], as_of_timestamp=OBSERVED_AT
    )
    clocks: dict[date, dict[str, object]] = {}
    for row in calendar.to_pylist():
        if row["calendar_id"] == "XNYS":
            clocks[row["session_date"]] = row
    recipe = build_one_session_recipe()
    points = resolve_schedule_points(
        recipe=recipe, ordered_sessions=sessions, session_clocks=clocks
    )
    derivation = ListingRowDerivation(
        points=points, recipe=recipe, ordered_sessions=sessions, sealed_formations=frozenset()
    )
    listing = serial_workspace.manifest.listings[0]
    market_as_of = date.fromisoformat(str(panel["as_of_session"]))
    bars = store.raw_bars(listing.listing_id, through=market_as_of)
    actions = tuple(
        value for value in store.actions(listing.listing_id) if value.effective_date <= market_as_of
    )
    here = derivation.derive(
        listing_id=listing.listing_id, symbol=listing.symbol, bars=bars, actions=actions
    )
    with ProcessPoolExecutor(
        max_workers=1, initializer=_initialize_derivation_worker, initargs=(derivation,)
    ) as pool:
        there = pool.submit(
            _derive_in_worker, (listing.listing_id, listing.symbol, tuple(bars), actions)
        ).result()
    assert there.listing_id == here.listing_id
    assert there.source_row_bindings == here.source_row_bindings
    assert set(there.tables) == set(here.tables)
    for key, table in here.tables.items():
        assert there.tables[key].equals(table)


def test_a_worker_failure_publishes_nothing_and_leaves_no_worker_behind(tmp_path: Path) -> None:
    """recovery: one failing unit fails the publication whole, cleanly.

    The parallel path is forced (two workers). One listing's actions carry
    an event the derivation refuses, so its unit raises inside a worker.
    The publication must surface that refusal, write no artifact -- no
    chunk, manifest, binding, seal or receipt -- close the store handle it
    opened, and leave no worker process or pool thread behind; the same
    workspace then publishes normally once the source is sound.
    """

    import threading

    import psutil

    from alphalattice.control.workspace_runtime.database import live_workspace_connections
    from alphalattice.foundation.causal_outcomes.execution import publication as owner

    workspace = build_real_risk_workspace(tmp_path / "workspace")
    store = MarketDataRepository(workspace.workspace)
    resolver = ArtifactResolver(workspace.artifact_root)
    poisoned = workspace.manifest.listings[3].listing_id
    original_actions = MarketDataRepository.actions

    def actions_with_a_spin_off(self, listing_id, *args, **kwargs):
        values = original_actions(self, listing_id, *args, **kwargs)
        if listing_id != poisoned:
            return values
        return (
            *values,
            CorporateActionEvent(
                listing_id, workspace.manifest.profile.provider, HISTORY_START, "SPIN_OFF"
            ),
        )

    def publish() -> PublishedCausalExecutionOutcome:
        return CausalExecutionOutcomePublisher(
            store=store,  # type: ignore[arg-type]
            resolver=resolver,
            artifact_root=workspace.artifact_root,
            mutation_gate=WorkspaceMutationGate(),
            derivation_workers=2,
        ).publish_daily(panel_manifest_ref=workspace.panel_manifest_ref, completed_at=OBSERVED_AT)

    files = sorted(path for path in Path(workspace.artifact_root).rglob("*") if path.is_file())
    me = psutil.Process()
    # The Host's kept workers (W10) are this process's for its life, so one an earlier test
    # started is not the publication's: only the children it starts must be gone (V358).
    children_before = {child.pid for child in me.children(recursive=True)}
    threads_before = {thread.name for thread in threading.enumerate()}
    with (
        patch.object(owner, "PARALLEL_DERIVATION_ROW_THRESHOLD", 1),
        patch.object(MarketDataRepository, "actions", actions_with_a_spin_off),
        pytest.raises(ValueError, match="unsupported corporate action"),
    ):
        publish()
    assert sorted(path for path in Path(workspace.artifact_root).rglob("*") if path.is_file()) == (
        files
    )
    assert [c for c in me.children(recursive=True) if c.pid not in children_before] == []
    assert {thread.name for thread in threading.enumerate()} <= threads_before
    live = live_workspace_connections(store.database.path)
    assert live is None or live.handles == 0
    with patch.object(owner, "PARALLEL_DERIVATION_ROW_THRESHOLD", 1):
        published = publish()
    assert published.receipt.action == "PUBLISHED"
    assert [c for c in me.children(recursive=True) if c.pid not in children_before] == []


def test_executor_creation_and_process_start_refusals_publish_nothing(tmp_path: Path) -> None:
    """failure: public worker-start refusals keep their cause and publish no artifact.

    An eighty-listing product-written workspace crosses the publisher's parallel
    derivation threshold without changing private policy. Refuse executor creation
    through multiprocessing's public context factory, then refuse a real spawn
    Process.start call. Finally admit one child and refuse the next start, proving
    pool shutdown removes a worker already admitted. Every refusal must surface
    the same stable code, preserve the OS error as its cause, and publish nothing.
    """

    import multiprocessing

    import psutil

    from alphalattice.control.task_control.child import ChildStartFailed

    workspace = build_real_risk_workspace(
        tmp_path / "workspace", symbols=tuple(f"QA{index:03d}" for index in range(80))
    )
    store = MarketDataRepository(workspace.workspace)
    resolver = ArtifactResolver(workspace.artifact_root)

    def publish() -> PublishedCausalExecutionOutcome:
        return CausalExecutionOutcomePublisher(
            store=store,  # type: ignore[arg-type]
            resolver=resolver,
            artifact_root=workspace.artifact_root,
            mutation_gate=WorkspaceMutationGate(),
            derivation_workers=2,
        ).publish_daily(panel_manifest_ref=workspace.panel_manifest_ref, completed_at=OBSERVED_AT)

    def artifact_files() -> tuple[Path, ...]:
        return tuple(
            sorted(
                path.relative_to(workspace.artifact_root)
                for path in Path(workspace.artifact_root).rglob("*")
                if path.is_file()
            )
        )

    before = artifact_files()
    constructor_error = OSError("process pool creation refused")
    with (
        patch("multiprocessing.get_context", side_effect=constructor_error) as get_context,
        pytest.raises(ChildStartFailed, match=r"^task_control\.child_start_failed$") as raised,
    ):
        publish()
    assert get_context.called
    assert raised.value.__cause__ is constructor_error
    assert artifact_files() == before

    spawn = multiprocessing.get_context("spawn")
    process_start_error = OSError("operating system refused child start")
    with (
        patch("multiprocessing.get_context", return_value=spawn) as get_context,
        patch.object(spawn.Process, "start", side_effect=process_start_error) as process_start,
        pytest.raises(ChildStartFailed, match=r"^task_control\.child_start_failed$") as raised,
    ):
        publish()
    assert get_context.called
    assert process_start.called
    assert raised.value.__cause__ is process_start_error
    assert artifact_files() == before

    original_start = spawn.Process.start
    start_attempts = 0
    started_pids: list[int] = []
    partial_start_error = OSError(1455, "operating system refused another child start")

    def start_then_refuse_second(process) -> None:
        nonlocal start_attempts
        start_attempts += 1
        if start_attempts == 2:
            raise partial_start_error
        original_start(process)
        assert process.pid is not None
        started_pids.append(process.pid)

    with (
        patch("multiprocessing.get_context", return_value=spawn) as get_context,
        patch.object(spawn.Process, "start", new=start_then_refuse_second),
        pytest.raises(ChildStartFailed, match=r"^task_control\.child_start_failed$") as raised,
    ):
        publish()
    assert get_context.called
    assert raised.value.__cause__ is partial_start_error
    assert start_attempts == 2
    assert len(started_pids) == 1
    assert all(not psutil.pid_exists(pid) for pid in started_pids)
    assert artifact_files() == before


def test_existing_seal_reuse_ignores_only_the_source_revision_counter() -> None:
    from alphalattice.foundation.causal_outcomes.execution.methods import (
        build_execution_outcome_method_binding,
        build_one_session_recipe,
    )
    from alphalattice.foundation.causal_outcomes.execution.publication import _method_authority

    values = dict(
        recipe=build_one_session_recipe(),
        catalog_hash="a" * 64,
        publication_policy_hash="b" * 64,
        snapshot_hash="c" * 64,
        schedule_hash="d" * 64,
        ordered_session_triples_hash="e" * 64,
        source_watermark_hash="revision-before",
        action_lineage="f" * 64,
    )
    first = build_execution_outcome_method_binding(**values)
    counter_only = build_execution_outcome_method_binding(
        **{**values, "source_watermark_hash": "revision-after"}
    )
    changed_actions = build_execution_outcome_method_binding(
        **{**values, "source_watermark_hash": "revision-after", "action_lineage": "0" * 64}
    )
    assert first.binding_hash != counter_only.binding_hash
    assert _method_authority(first) == _method_authority(counter_only)
    assert _method_authority(first) != _method_authority(changed_actions)


def test_outcomes_and_risk_keep_the_historical_members_of_a_ragged_panel(tmp_path: Path) -> None:
    """Real publishers/readers after an exit; no fabricated Panel or Outcome artifact.

    The synthetic membership decision is submitted to the existing coordinator's
    binding and governance owners. This is a producer/consumer integration case,
    not a browser or official-index-source acceptance.
    """
    from alphalattice.control.data_platform.maintenance.contracts import (
        MaintenanceTrigger,
        WorkspaceMaintenanceRequest,
    )
    from alphalattice.control.data_platform.readiness import WorkspaceReadinessGate
    from alphalattice.control.product_host.composition.workspace import WorkspaceRuntime
    from alphalattice.control.product_host.research_authoring.authority import (
        WorkspaceResearchAuthorityResolver,
    )
    from alphalattice.control.research_program.authoring.document import load_authoring_document
    from alphalattice.foundation.causal_outcomes.selection.listing_set import (
        resolve_published_execution_outcome,
    )
    from alphalattice.foundation.feature_engine.contracts import (
        FeatureBuildRequest,
        FeatureBuildStatus,
        FeatureInvalidation,
    )
    from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
    from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
        CurrentUniverseMaintenance,
        CurrentUniverseMaintenanceStatus,
    )
    from alphalattice.foundation.market_data_ops.sources.manifest import (
        build_quality_filtered_research_manifest,
    )
    from alphalattice.investment.risk_research.surfaces.returns import (
        CausalRiskReturnReader,
        CausalRiskReturnSurfacePublisher,
    )
    from alphalattice.kernel.data.calendar import materialize_calendar_schedule
    from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope
    from tests.researcher_methodology_surface.real_workspace import (
        MARKET_PROFILE_ID,
        PROFILE_PATH,
        SYMBOLS,
        SeededWalkProvider,
    )

    source = build_real_risk_workspace(tmp_path / "workspace")
    exit_day = date(2026, 8, 3)
    day = date(2026, 8, 11)
    now = datetime(2026, 8, 11, 22, tzinfo=UTC)
    sessions = tuple(
        row["session_date"]
        for row in materialize_calendar_schedule(
            ("XNAS",),
            start=HISTORY_START,
            end=day,
            as_of_timestamp=now + timedelta(hours=3),
        ).to_pylist()
    )
    provider = SeededWalkProvider(SYMBOLS, sessions)
    market = MarketDataRepository(source.workspace)
    updated = CurrentUniverseMaintenance(
        store=market,
        manifest=source.manifest,
        provider=provider,
        as_of_session=day,
        max_workers=1,
        # The fixture's initial audit is older than its admitted TTL. Authorize
        # its synthetic audit explicitly, rather than weakening production TTL.
        full_audit_listing_ids=frozenset(item.listing_id for item in source.manifest.listings),
    ).run(observed_at=now)
    assert updated.status is CurrentUniverseMaintenanceStatus.COMPLETED and updated.failed == 0
    # Keep one complete five-member sector, so no sector/population floor is weakened.
    retained = tuple(
        item.listing_id for item in source.manifest.listings if item.symbol in SYMBOLS[:5]
    )
    child = build_quality_filtered_research_manifest(source.manifest, eligible_listing_ids=retained)
    with closing(
        WorkspaceRuntime.create(
            workspace=source.workspace,
            manifest=source.manifest,
            provider=provider,
            artifact_root=source.artifact_root,
        )
    ) as runtime:
        coordinator = runtime.maintenance_coordinator(
            readiness_gate=WorkspaceReadinessGate(
                market_data=runtime.market_data,
                feature_state=runtime.feature_state,
                panel_state=runtime.panel_state,
                profile_path=PROFILE_PATH,
                provider=provider,
            ),
            clock=lambda: now,
        )
        assert coordinator._bind_derived_manifest_evidence(
            source.manifest,
            child,
            requested_as_of=day,
            observed_at=now,
        )
        # A declared engineering event observed on 08-03, not a later discovery
        # backdated to make these labels pass. The provider above supplies the
        # later quote coverage explicitly; no post-exit membership is added.
        coordinator._bind_manifest(
            child, observed_at=datetime(2026, 8, 3, 18, tzinfo=UTC), effective_session=exit_day
        )
        request = WorkspaceMaintenanceRequest.create(
            market_profile_id=MARKET_PROFILE_ID,
            target_market_session=day,
            knowledge_cutoff_at=now,
            trigger=MaintenanceTrigger.STARTUP,
            membership_revision=child.revision_sha256,
            data_policy_hash="1" * 64,
            feature_policy_hash="2" * 64,
        )
        governance = coordinator._govern_quality(
            None,
            request=request,
            observed_at=now,
            macro_retry_exhausted=False,
            observed_workers=1,
        )
        assert (
            coordinator._handle_governance_result(
                governance,
                maintenance_id=None,
                request=request,
                observed_at=now,
                observed_workers=1,
            )
            is None
        )
        service = runtime.feature_foundation
        assert (
            service.refresh_spy(as_of_session=day, observed_at=now).status
            is FeatureBuildStatus.COMPLETED
        )
        spy = runtime.feature_state.market_reference("SPY")
        assert spy is not None
        built = service.build(
            FeatureBuildRequest.create(
                manifest_revision=service.manifest.revision_sha256,
                catalog=service.catalog.binding,
                spy_revision=str(spy["revision_hash"]),
                history_start=sessions[0],
                as_of_session=day,
                invalidations=(
                    FeatureInvalidation("manifest_removal", earliest_session=exit_day),
                    *(
                        FeatureInvalidation(
                            "normal_new_session",
                            listing_id=listing_id,
                            earliest_session=exit_day,
                            affected_sessions=tuple(
                                value for value in sessions if value >= exit_day
                            ),
                        )
                        for listing_id in retained
                    ),
                ),
            ),
            observed_at=now,
        )
        assert built.status is FeatureBuildStatus.COMPLETED, built
        published = runtime.panel_snapshot_publisher().publish(
            manifest=service.manifest,
            history_start=sessions[0],
            as_of_session=day,
            observed_at=now,
        )
        ref = published.artifact.uri
    resolver = ArtifactResolver(source.artifact_root)
    panel = resolver.load_feature_panel_manifest(ref)
    assert panel["safe_summary"]["membership"]["as_of_member_count"] == 5
    assert len(FeaturePanelReader(resolver).listing_ids(ref)) == 10
    # The coverage axis without row-content validation is the same verified
    # axis, and it refuses a manifest whose listing_set_hash names another.
    assert FeaturePanelReader(resolver).listing_axis(ref) == FeaturePanelReader(
        resolver
    ).listing_ids(ref)
    with (
        patch.object(
            ArtifactResolver,
            "load_feature_panel_manifest",
            return_value={**panel, "listing_set_hash": "0" * 64},
        ),
        pytest.raises(ValueError, match="listing_axis_mismatch"),
    ):
        FeaturePanelReader(resolver).listing_axis(ref)
    outcome = CausalExecutionOutcomePublisher(
        store=market,
        resolver=resolver,
        artifact_root=source.artifact_root,
        mutation_gate=WorkspaceMutationGate(),
    ).publish_daily(panel_manifest_ref=ref, completed_at=now)
    assert len(outcome.manifest.listing_ids) == 10
    from alphalattice.foundation.causal_outcomes.execution.methods import FIVE_SESSION_RECIPE_ID
    from alphalattice.foundation.causal_outcomes.execution.readers import (
        DevelopmentOnlyExecutionOutcomeReader,
    )
    from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReadRequest

    diagnostic = CausalExecutionOutcomePublisher(
        store=market,
        resolver=resolver,
        artifact_root=source.artifact_root,
        mutation_gate=WorkspaceMutationGate(),
    ).publish_daily(panel_manifest_ref=ref, completed_at=now, recipe_id=FIVE_SESSION_RECIPE_ID)
    diagnostic_reader = DevelopmentOnlyExecutionOutcomeReader(source.artifact_root)
    formation = date(2026, 7, 31)
    rows = diagnostic_reader.read_development_sessions(
        diagnostic.manifest_artifact.uri, (formation,)
    )
    assert rows.num_rows == 10
    assert set(rows["holding_end_session"].to_pylist()) == {date(2026, 8, 10)}
    assert rows["simple_return"].null_count == 0
    assert formation not in diagnostic_reader.available_development_sessions(
        diagnostic.manifest_artifact.uri, holding_end_through=date(2026, 8, 7)
    )
    assert formation in diagnostic_reader.available_development_sessions(
        diagnostic.manifest_artifact.uri, holding_end_through=day
    )
    for session, expected in ((formation, 10), (exit_day, 5)):
        batches = tuple(
            FeaturePanelReader(resolver).batches(
                FeaturePanelReadRequest(ref, session, session, ("rev_21",))
            )
        )
        assert sum(batch.num_rows for batch in batches) == expected
    assert (
        resolve_published_execution_outcome(
            artifact_root=source.artifact_root,
            listing_set_hash=str(panel["listing_set_hash"]),
            snapshot_handle=outcome.manifest.snapshot_hash,
        )
        == outcome.manifest
    )
    risk, _ = CausalRiskReturnSurfacePublisher(
        store=market,
        resolver=resolver,
        artifact_root=source.artifact_root,
        mutation_gate=WorkspaceMutationGate(),
    ).publish(panel_manifest_ref=ref, market_profile_id=MARKET_PROFILE_ID)
    assert risk.epoch.ordered_listing_ids == outcome.manifest.listing_ids
    assert CausalRiskReturnReader(source.artifact_root).read_sessions(
        risk, (sessions[-2],)
    ).shape == (1, 10)
    document = load_authoring_document(
        (Path(__file__).parent / "fixtures/risk_covariance_development.yaml").read_text()
    )
    envelope = ResearchExperimentEnvelope.create(
        **{**document["experiment"], "data_snapshot_handle": str(panel["snapshot_hash"])}
    )
    authority = WorkspaceResearchAuthorityResolver(workspace=source.workspace).resolve(envelope)
    assert authority.ordered_listing_ids == risk.epoch.ordered_listing_ids
    assert authority.universe_revision_sha256 == risk.epoch.universe_manifest_revision
