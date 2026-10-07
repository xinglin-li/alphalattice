"""Public storage cleanup preserves preparation roots and scientific publications."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    ResearchWorkspaceScoreInput,
    publish_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_scoring import (
    workspace_observation_history_scope,
)
from alphalattice.control.product_host.storage.input_references import ResearchInputStorage
from alphalattice.control.product_host.storage.retention import StorageRetentionError
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskInputEnvelope,
    WorkItemDefinition,
)
from alphalattice.foundation.causal_outcomes.execution.artifacts import (
    local_qa_preparation_retention_inventory,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import (
    PreparedLocalQASnapshotRows,
    planned_local_qa_schedule,
)
from alphalattice.foundation.market_data_ops.sources.contracts import RawDailyBar
from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _publish_alpha(store, storage, scope, generation):
    sessions = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(4))
    values = np.arange(8, dtype=np.float64).reshape(4, 2) + 100.0 * generation
    columns = {"close": values, "formula::value": values + 1.0}
    return store.publish_workspace_observation_history(
        scope_hash=scope,
        selection_hash=canonical_hash({"model": generation}),
        dependency_prefix_hash=canonical_hash(
            {name: sha256(value.tobytes()).hexdigest() for name, value in columns.items()}
        ),
        formation_sessions=sessions,
        ordered_listing_ids=("A", "B"),
        stable_session_count=3,
        columns=columns,
        capacity=storage.admit_evidence_bytes,
    )


def _publish_causal(workspace, storage, quote):
    schedule = planned_local_qa_schedule(date(2026, 9, 1), date(2026, 9, 11))
    days = tuple(item.formation_session for item in schedule)
    bars = tuple(
        RawDailyBar(
            listing,
            "synthetic",
            day,
            100.0 + index,
            102.0 + index,
            99.0 + index,
            101.0 + index,
            1000,
        )
        for index, day in enumerate(days)
        for listing in ("A", "B")
    )
    source = LocalQAMarketSnapshot.create(
        source_hash=canonical_hash("synthetic-bars"),
        through=days[-1],
        ordered_listing_ids=("A", "B"),
        schedule=schedule,
        bars=tuple(
            replace(bar, open=quote)
            if (bar.session_date, bar.listing_id) == (days[2], "A")
            else bar
            for bar in bars
        ),
        actions=(),
    )
    PreparedLocalQASnapshotRows(
        source, artifact_root=workspace / "artifacts", capacity=storage.admit_evidence_bytes
    ).rows(sessions=days[:6], through=days[5])
    return local_qa_preparation_retention_inventory(workspace / "artifacts").current_head_hash


@pytest.fixture
def preparation_storage(tmp_path):
    bindings = tuple(
        ResearchWorkspaceScoreInput(
            strategy_package_id="synthetic-retention",
            strategy_package_hash=canonical_hash("synthetic-retention"),
            component_id=component,
            authority_relative_path=f"artifacts/alpha-research/authorities/{component}.json",
            authority_hash=canonical_hash(component),
            source_kind="WORKSPACE_DATA_FEATURE",
        )
        for component in ("FIRST", "SECOND", "RETIRED")
    )
    manifest = ResearchWorkspaceManifest.create(
        workspace_id="preparation-retention",
        default_strategy_package_id="synthetic-retention",
        default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
        strategy_artifacts=(),
        score_inputs=bindings[:2],
    )
    publish_research_workspace_manifest(tmp_path, manifest)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        storage = ResearchInputStorage(session)
        assert storage.set_cap(str(1024**3), caller="HUMAN")["status"] == "CONFIGURED"
        store = AlphaCurrentArtifactStore(tmp_path / "artifacts")
        values = np.arange(8, dtype=np.float64).reshape(4, 2) + 100.0
        snapshot = store.publish_frozen_observations(
            FrozenPriceVolumeInputs(
                formation_sessions=tuple(date(2026, 1, 1) + timedelta(days=i) for i in range(4)),
                ordered_listing_ids=("A", "B"),
                sector_by_listing_id={"A": "ONE", "B": "ONE"},
                open=values,
                high=values + 2.0,
                low=values - 1.0,
                close=values + 1.0,
                volume=values + 1000.0,
                market_context_values=np.zeros((4, 3), dtype=np.float64),
                sector_trend_values=np.zeros((4, 1), dtype=np.float64),
                source_binding_hash=canonical_hash("scientific-observations"),
            ),
            disposition="SYNTHETIC_INPUT_QA",
        )
        scientific = {path: _digest(path) for path in store.root.rglob("*") if path.is_file()}
        scopes = tuple(workspace_observation_history_scope(binding) for binding in bindings)
        heads = tuple(
            tuple(_publish_alpha(store, storage, scope, generation) for generation in generations)
            for scope, generations in zip(scopes, ((1, 2, 3), (4, 5, 6), (7,)), strict=True)
        )
        causal_heads = tuple(
            _publish_causal(tmp_path, storage, quote) for quote in (111.0, 122.0, 133.0)
        )
        yield session, storage, store, bindings, scopes, heads, causal_heads, scientific, snapshot


def _inventory(workspace, store, scopes):
    alpha = store.workspace_observation_history_retention(active_scope_hashes=scopes[:2])
    causal = local_qa_preparation_retention_inventory(workspace / "artifacts")
    roots = {item.path: item.sha256 for item in alpha.roots} | {
        workspace / "artifacts" / item.relative_path: item.file_hash
        for item in causal.protected_files
    }
    targets = {
        item.path.relative_to(workspace).as_posix(): item.sha256 for item in alpha.targets
    } | {f"artifacts/{item.relative_path}": item.file_hash for item in causal.candidate_files}
    return alpha, causal, roots, targets


def test_confirmed_cleanup_deletes_only_exact_unrooted_preparation_files(preparation_storage):
    session, storage, store, _, scopes, heads, causal_heads, scientific, snapshot = (
        preparation_storage
    )
    alpha, causal, roots, targets = _inventory(session.workspace, store, scopes)
    assert targets and alpha.targets and causal.candidate_files
    assert causal.current_head_hash == causal_heads[2]
    assert causal.previous_head_hash == causal_heads[1]
    rooted_heads = {path.stem for path in roots}
    for slot_heads in heads[:2]:
        assert {slot_heads[1].head_hash, slot_heads[2].head_hash} <= rooted_heads
        assert slot_heads[0].head_hash not in rooted_heads
    assert scopes[2] not in rooted_heads and heads[2][0].head_hash not in rooted_heads
    plan = storage.plan()
    assert plan["targets"] == targets
    assert (
        not {path.relative_to(session.workspace).as_posix() for path in scientific} & targets.keys()
    )
    for relative, digest in targets.items():
        assert _digest(session.workspace / relative) == digest
    with pytest.raises(ValueError, match=r"storage\.human_confirmation_required"):
        storage.confirm(plan["plan_hash"], caller="SERVICE_AUTOMATION")
    result = storage.confirm(plan["plan_hash"], caller="HUMAN")
    assert result["status"] == "COMPLETED" and result["deleted_paths"] == len(targets)
    assert all(not (session.workspace / relative).exists() for relative in targets)
    assert all(_digest(path) == digest for path, digest in roots.items())
    assert all(_digest(path) == digest for path, digest in scientific.items())
    assert store.load_frozen_observation_snapshot(snapshot.snapshot_hash) == snapshot
    for scope, slot_heads in zip(scopes[:2], heads[:2], strict=True):
        assert store.load_workspace_observation_history(scope)[0] == slot_heads[2]
    assert (
        local_qa_preparation_retention_inventory(session.workspace / "artifacts").candidate_files
        == ()
    )


def test_explicit_owner_head_references_protect_parts_without_admitting_new_pin_surface(
    preparation_storage,
):
    session, storage, store, _, scopes, heads, causal_heads, _, _ = preparation_storage
    alpha = store.workspace_observation_history_retention(
        active_scope_hashes=scopes[:2], referenced_heads=(heads[0][0].head_hash,)
    )
    assert heads[0][0].head_hash in {item.path.stem for item in alpha.roots}
    assert heads[0][0].head_hash not in {item.path.stem for item in alpha.targets}
    causal = local_qa_preparation_retention_inventory(
        session.workspace / "artifacts", referenced_heads=(causal_heads[0],)
    )
    assert causal.referenced_heads == (causal_heads[0],) and causal.candidate_files == ()
    before = storage.plan()["targets"]
    with pytest.raises(ValueError, match=r"storage\.input_pin_unavailable"):
        storage.pin(heads[0][0].head_hash, pinned=True, caller="HUMAN")
    assert storage.plan()["targets"] == before


def test_in_flight_task_protects_all_candidates_and_stales_prior_cleanup(preparation_storage):
    session, storage, _store, _, _, _, _, scientific, _ = preparation_storage
    plan = storage.plan()
    assert plan["targets"]
    envelope = TaskInputEnvelope.create(
        task_kind="storage_preparation_test", input_schema_id="storage.preparation.v1", payload={}
    )
    goal = ResearchGoal.create(
        goal_kind="STORAGE_PREPARATION",
        input_hash=envelope.input_hash,
        deliverable_kind="preparation.proof",
        summary="Retain preparation during admitted work",
    )
    workflow = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=canonical_hash("retention-workflow"),
        verifier_catalog_hash=canonical_hash("retention-verifiers"),
        work_items=(
            WorkItemDefinition.create(
                stage_id="prepare_history", dependency_ids=(), verifier_id="test.preparation"
            ),
        ),
    )
    session.task_control_registry.admit(
        input_envelope=envelope, goal=goal, plan=workflow, observed_at=datetime.now(UTC)
    )
    assert storage.plan()["targets"] == {}
    with pytest.raises(StorageRetentionError) as refused:
        storage.confirm(plan["plan_hash"], caller="HUMAN")
    assert refused.value.failure_code == "storage.cleanup_plan_stale"
    assert all(
        _digest(session.workspace / relative) == digest
        for relative, digest in plan["targets"].items()
    )
    assert all(_digest(path) == digest for path, digest in scientific.items())


@pytest.mark.parametrize("change", ["target_bytes", "active_slots", "publication"])
def test_changed_cleanup_proof_refuses_before_deleting_any_file(preparation_storage, change):
    session, storage, store, bindings, scopes, heads, _, scientific, _ = preparation_storage
    plan = storage.plan()
    if change == "target_bytes":
        path = next(
            session.workspace / relative
            for relative in plan["targets"]
            if relative.endswith(f"/heads/{heads[0][0].head_hash}.json")
        )
        # A valid sealed head with different JSON bytes retains the same scientific values.
        path.write_text(
            json.dumps(json.loads(path.read_bytes()), indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        assert _digest(path) != plan["targets"][path.relative_to(session.workspace).as_posix()]
    elif change == "active_slots":
        update_research_workspace_manifest(
            session.workspace,
            lambda manifest: manifest.with_bindings(score_inputs=bindings),
            gate=session.mutation_gate,
        )
    else:
        _publish_alpha(store, storage, scopes[0], 8)
    before = {
        session.workspace / relative: _digest(session.workspace / relative)
        for relative in plan["targets"]
    }
    with pytest.raises(StorageRetentionError) as refused:
        storage.confirm(plan["plan_hash"], caller="HUMAN")
    assert refused.value.failure_code == "storage.cleanup_plan_stale"
    assert all(_digest(path) == digest for path, digest in before.items())
    assert all(_digest(path) == digest for path, digest in scientific.items())
