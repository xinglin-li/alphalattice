"""Materialize the frozen Gate Q/V strategies into one independent workspace.

This command is the only consumer of the research worktree.  The ordinary
product launcher remains workspace-only, and the resulting authority manifest
contains relative paths plus content identities rather than source paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
if str(PLAYPEN_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PLAYPEN_ROOT / "src"))

from alphalattice.control.product_host.composition.research_workspace import (  # noqa: E402
    ResearchWorkspaceArtifact,
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
    read_research_workspace_manifest,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (  # noqa: E402
    BROAD_FEATURE_STRATEGY_ID,
    POST_OBSERVED_STRATEGY_AUTHORITY_ROOT_KEY,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (  # noqa: E402
    POST_OBSERVED_ASSET_INDEX_SHA256,
    POST_OBSERVED_AUTHORITY_MANIFEST_NAME,
    POST_OBSERVED_DESCRIPTOR_SHA256,
    POST_OBSERVED_FINAL_RECEIPT_SHA256,
    POST_OBSERVED_MODEL_INDEX_SHA256,
    POST_OBSERVED_QV_MAP_SHA256,
    POST_OBSERVED_RESEARCH_SHA,
    FrozenAuthorityArtifact,
    PostObservedStrategyAuthorityManifest,
    admit_post_observed_strategy_authority,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash  # noqa: E402

SOURCE_FILES = {
    "descriptor": (
        "post-governance-onboarding-descriptor.json",
        POST_OBSERVED_DESCRIPTOR_SHA256,
    ),
    "qv_map": ("q-v-authority-map.json", POST_OBSERVED_QV_MAP_SHA256),
    "model_index": ("model-payload-index.json", POST_OBSERVED_MODEL_INDEX_SHA256),
    "asset_index": (
        "research-authority-asset-index.json",
        POST_OBSERVED_ASSET_INDEX_SHA256,
    ),
    "final_receipt": (
        "research-authority-final-receipt.json",
        POST_OBSERVED_FINAL_RECEIPT_SHA256,
    ),
}

SHARED_RUNTIME_ARTIFACT_ROOTS = (
    "data-operations/execution-outcomes",
    "data-operations/risk-returns",
    "data-operations/tradability",
    "feature-panel/closure",
)
"""Immutable shared inputs needed by installed Portfolio packages.

Task state, result/report publications and prior Evidence/CRO publications are
workspace-local runtime state.  Materialising a new workspace must not clone
them merely because the shared Market/Risk artifacts live below the same root.
"""


class MaterializationError(ValueError):
    """Stable refusal before a partial target workspace becomes visible."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise MaterializationError(f"source_unreadable:{path}") from error
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"source_json_unreadable:{path}") from error
    if not isinstance(payload, dict):
        raise MaterializationError(f"source_json_invalid:{path}")
    return cast(dict[str, Any], payload)


def _verified_file(path: Path, expected_hash: str) -> Path:
    resolved = path.resolve()
    if _sha256(resolved) != expected_hash:
        raise MaterializationError(f"source_hash_mismatch:{resolved}")
    return resolved


def _repository_root(closure: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(closure), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MaterializationError("research_repository_unreadable")
    root = Path(result.stdout.strip()).resolve()
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0 or head.stdout.strip() != POST_OBSERVED_RESEARCH_SHA:
        raise MaterializationError("research_repository_sha_mismatch")
    return root


def _source_asset(binding: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(binding, dict):
        raise MaterializationError(f"source_binding_invalid:{label}")
    path = binding.get("absolute_path")
    digest = binding.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise MaterializationError(f"source_binding_invalid:{label}")
    return _verified_file(Path(path), digest), digest


def _copy_asset(
    *,
    source: Path,
    digest: str,
    artifact_id: str,
    relative_path: str,
    authority_root: Path,
    is_array: bool,
) -> FrozenAuthorityArtifact:
    destination = authority_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(destination) != digest:
        raise MaterializationError(f"copied_asset_hash_mismatch:{artifact_id}")
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    if is_array:
        try:
            array = np.load(destination, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise MaterializationError(f"source_array_unreadable:{artifact_id}") from error
        dtype = array.dtype.str
        shape = tuple(int(value) for value in array.shape)
    return FrozenAuthorityArtifact(
        artifact_id=artifact_id,
        relative_path=relative_path,
        sha256=digest,
        bytes=destination.stat().st_size,
        dtype=dtype,
        shape=shape,
    )


def _write_array_asset(
    *,
    value: np.ndarray[Any, Any],
    artifact_id: str,
    relative_path: str,
    authority_root: Path,
) -> FrozenAuthorityArtifact:
    destination = authority_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, value, allow_pickle=False)
    digest = _sha256(destination)
    return FrozenAuthorityArtifact(
        artifact_id=artifact_id,
        relative_path=relative_path,
        sha256=digest,
        bytes=destination.stat().st_size,
        dtype=value.dtype.str,
        shape=tuple(int(axis) for axis in value.shape),
    )


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise MaterializationError(f"shared_workspace_input_absent:{source}")
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def _research_inputs(
    closure: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _repository_root(closure)
    held: dict[str, dict[str, Any]] = {}
    for key, (name, digest) in SOURCE_FILES.items():
        path = _verified_file(closure / name, digest)
        held[key] = _json(path)
    asset_index = held["asset_index"]
    transitive = asset_index.get("transitive_gate_asset_indexes")
    if not isinstance(transitive, dict):
        raise MaterializationError("source_asset_index_invalid")
    gate_indexes: dict[str, dict[str, Any]] = {}
    for gate in ("Q", "V"):
        path, _digest = _source_asset(
            transitive.get(gate), label=f"gate_{gate.lower()}_asset_index"
        )
        gate_indexes[gate] = _json(path)
    gates = held["qv_map"].get("gates")
    if not isinstance(gates, list) or len(gates) != 6:
        raise MaterializationError("source_qv_map_invalid")
    for index, gate in ((0, "Q"), (5, "V")):
        entry = gates[index]
        if not isinstance(entry, dict) or entry.get("gate") != gate:
            raise MaterializationError(f"source_qv_gate_invalid:{gate}")
        _source_asset(entry.get("canonical_receipt"), label=f"gate_{gate.lower()}_receipt")
    final = held["final_receipt"]
    if (
        final.get("readiness") != "RESEARCH_CLOSURE_READY_FOR_POST_GOVERNANCE_ONBOARDING"
        or final.get("fit_count") != 0
        or final.get("book_replay_count") != 0
        or final.get("current_pointer_mutated") is not False
    ):
        raise MaterializationError("source_final_receipt_invalid")
    return held["descriptor"], held["final_receipt"], gate_indexes["V"]


def _snapshot_asset(snapshot: dict[str, Any], *, suffix: str, label: str) -> tuple[Path, str]:
    normalized = suffix.replace("/", "\\").lower()
    matches = [
        value
        for key, value in snapshot.items()
        if str(key).replace("/", "\\").lower().endswith(normalized)
    ]
    if len(matches) != 1:
        raise MaterializationError(f"source_snapshot_binding_invalid:{label}")
    return _source_asset(matches[0], label=label)


def _materialize_authority(
    *, closure: Path, authority_root: Path
) -> PostObservedStrategyAuthorityManifest:
    descriptor, _, gate_v_assets = _research_inputs(closure)
    roles = descriptor.get("roles")
    sizing = descriptor.get("sizing")
    strategies = descriptor.get("strategies")
    geometry = descriptor.get("research_geometry")
    clock = descriptor.get("clock")
    cost = descriptor.get("cost")
    descriptor_sections = (roles, sizing, strategies, geometry, clock, cost)
    if not all(isinstance(value, dict) for value in descriptor_sections):
        raise MaterializationError("source_descriptor_invalid")
    roles = cast(dict[str, Any], roles)
    sizing = cast(dict[str, Any], sizing)
    strategies = cast(dict[str, Any], strategies)
    geometry = cast(dict[str, Any], geometry)
    clock = cast(dict[str, Any], clock)
    cost = cast(dict[str, Any], cost)
    if (
        descriptor.get("status") != "POST_OBSERVED_RESEARCH_FREEZE_NOT_INSTALLED"
        or geometry.get("supported_sessions") != 1_747
        or geometry.get("first_session") != "2019-08-09"
        or geometry.get("last_session") != "2026-07-29"
        or clock.get("decision_cut") != "CLOSE_T"
        or clock.get("execution") != "OPEN_T_PLUS_1"
        or cost.get("primary_bps_per_side") != 5
        or cost.get("sensitivity_bps_per_side") != 10
    ):
        raise MaterializationError("source_descriptor_contract_invalid")

    g2_surface = roles["G2_R0_TREND"]["historical_score_surface"]
    g6_surface = roles["G6_R0_FAST_REBOUND"]["historical_score_surface"]
    g2_score, g2_score_hash = _source_asset(g2_surface["artifact"], label="g2_scores")
    g6_score, g6_score_hash = _source_asset(g6_surface["artifact"], label="g6_scores")
    g2_manifest, g2_manifest_hash = _source_asset(g2_surface["manifest"], label="g2_score_manifest")
    g6_manifest, g6_manifest_hash = _source_asset(g6_surface["manifest"], label="g6_score_manifest")
    g2_manifest_json = _json(g2_manifest)
    g6_manifest_json = _json(g6_manifest)
    g2_identity = g2_manifest_json.get("identity")
    g6_identity = g6_manifest_json.get("identity")
    if not isinstance(g2_identity, dict) or not isinstance(g6_identity, dict):
        raise MaterializationError("source_score_manifest_invalid")
    if (
        g2_identity.get("sessions") != g6_identity.get("sessions")
        or g2_identity.get("listings") != g6_identity.get("listings")
        or g2_surface.get("session_axis_hash") != g6_surface.get("session_axis_hash")
        or g2_surface.get("listing_axis_hash") != g6_surface.get("listing_axis_hash")
    ):
        raise MaterializationError("source_score_axis_mismatch")
    source_sessions = g2_identity.get("sessions")
    listings = g2_identity.get("listings")
    if not isinstance(source_sessions, list) or not isinstance(listings, list):
        raise MaterializationError("source_score_axis_invalid")
    sessions = tuple(date.fromisoformat(str(value)) for value in source_sessions[4:])
    ordered_listings = tuple(str(value) for value in listings)

    gate_v_artifacts = gate_v_assets.get("artifacts")
    if not isinstance(gate_v_artifacts, dict):
        raise MaterializationError("source_gate_v_asset_index_invalid")
    source_snapshot_path, source_snapshot_hash = _source_asset(
        gate_v_artifacts.get("source-snapshot-before.json"),
        label="gate_v_source_snapshot",
    )
    snapshot = _json(source_snapshot_path)
    panel_ready_path, _ = _snapshot_asset(
        snapshot,
        suffix="_panel_shared/ready.json",
        label="shared_development_manifest",
    )
    panel_returns_path, _ = _snapshot_asset(
        snapshot,
        suffix="_panel_shared/returns.npy",
        label="shared_development_returns",
    )
    panel_eligible_path, _ = _snapshot_asset(
        snapshot,
        suffix="_panel_shared/eligible.npy",
        label="shared_development_eligible",
    )
    deployment_manifest_path, _ = _snapshot_asset(
        snapshot,
        suffix="current-successor-tail/composite/manifest.json",
        label="shared_observed_manifest",
    )
    deployment_returns_path, _ = _snapshot_asset(
        snapshot,
        suffix="current-successor-tail/composite/returns.npy",
        label="shared_observed_returns",
    )
    deployment_eligible_path, _ = _snapshot_asset(
        snapshot,
        suffix="current-successor-tail/composite/eligible.npy",
        label="shared_observed_eligible",
    )
    panel_ready = _json(panel_ready_path)
    deployment_manifest = _json(deployment_manifest_path)
    panel_sessions = tuple(str(value)[:10] for value in panel_ready.get("sessions", ()))
    panel_listings = tuple(str(value) for value in panel_ready.get("listing_ids", ()))
    deployment_sessions = tuple(
        str(value) for value in deployment_manifest.get("formation_sessions", ())
    )
    deployment_listings = tuple(
        str(value) for value in deployment_manifest.get("ordered_listing_ids", ())
    )
    panel_session_lookup = {value: index for index, value in enumerate(panel_sessions)}
    panel_listing_lookup = {value: index for index, value in enumerate(panel_listings)}
    deployment_session_lookup = {value: index for index, value in enumerate(deployment_sessions)}
    deployment_listing_lookup = {value: index for index, value in enumerate(deployment_listings)}
    full_sessions = tuple(str(value) for value in source_sessions)
    output_listing_lookup = {value: index for index, value in enumerate(ordered_listings)}
    try:
        panel_columns = [output_listing_lookup[value] for value in panel_listings]
        development_rows = [
            index for index, value in enumerate(full_sessions) if value <= "2024-07-29"
        ]
        observed_rows = [
            index for index, value in enumerate(full_sessions) if value >= "2024-07-30"
        ]
        panel_rows = [panel_session_lookup[full_sessions[index]] for index in development_rows]
        observed_source_rows = [
            deployment_session_lookup[full_sessions[index]] for index in observed_rows
        ]
        observed_columns = [deployment_listing_lookup[value] for value in ordered_listings]
    except KeyError as error:
        raise MaterializationError("source_shared_axis_unresolved") from error
    panel_returns = np.load(panel_returns_path, allow_pickle=False, mmap_mode="r")
    panel_eligible = np.load(panel_eligible_path, allow_pickle=False, mmap_mode="r")
    observed_returns = np.load(deployment_returns_path, allow_pickle=False, mmap_mode="r")
    observed_eligible = np.load(deployment_eligible_path, allow_pickle=False, mmap_mode="r")
    realized: np.ndarray[Any, Any] = np.full(
        (len(full_sessions), len(ordered_listings)), np.nan, dtype=np.float64
    )
    eligible: np.ndarray[Any, Any] = np.zeros(realized.shape, dtype=np.bool_)
    realized[np.ix_(development_rows, panel_columns)] = panel_returns[
        np.ix_(panel_rows, [panel_listing_lookup[value] for value in panel_listings])
    ]
    eligible[np.ix_(development_rows, panel_columns)] = panel_eligible[
        np.ix_(panel_rows, [panel_listing_lookup[value] for value in panel_listings])
    ]
    realized[np.ix_(observed_rows, range(len(ordered_listings)))] = observed_returns[
        np.ix_(observed_source_rows, observed_columns)
    ]
    eligible[np.ix_(observed_rows, range(len(ordered_listings)))] = observed_eligible[
        np.ix_(observed_source_rows, observed_columns)
    ]
    realized = np.ascontiguousarray(np.nan_to_num(realized[4:], nan=0.0), dtype="<f8")
    eligible = np.ascontiguousarray(eligible[4:], dtype=np.bool_)
    execution_available = np.ones(eligible.shape, dtype=np.bool_)

    mu = sizing["mu.iv0"]
    curve, curve_hash = _source_asset(mu["curve"], label="g6_causal_curve")
    latest, latest_hash = _source_asset(
        mu["latest_calibration_row"], label="g6_latest_calibration_row"
    )
    return_assets = strategies["RETURN"]["historical_contract"]["assets"]
    balanced_assets = strategies["BALANCED"]["historical_contract"]["assets"]

    source_bindings = (
        ("g2_scores", g2_score, g2_score_hash, "artifacts/g2-scores.npy", True),
        (
            "g2_score_manifest",
            g2_manifest,
            g2_manifest_hash,
            "artifacts/g2-score-manifest.json",
            False,
        ),
        ("g6_scores", g6_score, g6_score_hash, "artifacts/g6-scores.npy", True),
        (
            "g6_score_manifest",
            g6_manifest,
            g6_manifest_hash,
            "artifacts/g6-score-manifest.json",
            False,
        ),
        (
            "g6_causal_curve",
            curve,
            curve_hash,
            "artifacts/g6-causal-curve.npy",
            True,
        ),
        (
            "g6_latest_calibration_row",
            latest,
            latest_hash,
            "artifacts/g6-latest-calibration-row.npy",
            True,
        ),
    )
    artifacts = [
        _copy_asset(
            source=source,
            digest=digest,
            artifact_id=artifact_id,
            relative_path=relative,
            authority_root=authority_root,
            is_array=is_array,
        )
        for artifact_id, source, digest, relative, is_array in source_bindings
    ]
    artifacts.extend(
        (
            _write_array_asset(
                value=eligible,
                artifact_id="shared_decision_eligible",
                relative_path="artifacts/shared-decision-eligible.npy",
                authority_root=authority_root,
            ),
            _write_array_asset(
                value=execution_available,
                artifact_id="shared_execution_available",
                relative_path="artifacts/shared-execution-available.npy",
                authority_root=authority_root,
            ),
            _write_array_asset(
                value=realized,
                artifact_id="shared_realized_returns",
                relative_path="artifacts/shared-realized-returns.npy",
                authority_root=authority_root,
            ),
        )
    )
    for prefix, values in (("return", return_assets), ("balanced", balanced_assets)):
        for source_key, artifact_id, filename, is_array in (
            ("weights", f"{prefix}_weights_oracle", f"{prefix}-weights-oracle.npy", True),
            ("daily_5", f"{prefix}_daily_5bps", f"{prefix}-daily-5bps.parquet", False),
            ("daily_10", f"{prefix}_daily_10bps", f"{prefix}-daily-10bps.parquet", False),
        ):
            source, digest = _source_asset(values[source_key], label=artifact_id)
            artifacts.append(
                _copy_asset(
                    source=source,
                    digest=digest,
                    artifact_id=artifact_id,
                    relative_path=f"artifacts/{filename}",
                    authority_root=authority_root,
                    is_array=is_array,
                )
            )
    manifest = PostObservedStrategyAuthorityManifest.create(
        research_repository_sha=POST_OBSERVED_RESEARCH_SHA,
        source_descriptor_sha256=POST_OBSERVED_DESCRIPTOR_SHA256,
        source_qv_map_sha256=POST_OBSERVED_QV_MAP_SHA256,
        source_model_index_sha256=POST_OBSERVED_MODEL_INDEX_SHA256,
        source_asset_index_sha256=POST_OBSERVED_ASSET_INDEX_SHA256,
        source_final_receipt_sha256=POST_OBSERVED_FINAL_RECEIPT_SHA256,
        source_shared_input_snapshot_sha256=source_snapshot_hash,
        formation_sessions=sessions,
        ordered_listing_ids=ordered_listings,
        source_score_prefix_rows=4,
        maturity_start_formation=521,
        review_phase=1,
        decision_cut="CLOSE_T",
        execution="OPEN_T_PLUS_1",
        primary_cost_bps_per_side=5,
        sensitivity_cost_bps_per_side=10,
        execution_availability_disposition=("ASSUMED_FULL_FILL_FIXED_COST_HISTORICAL_REPLAY"),
        capacity_disposition="NOT_ADMITTED",
        artifacts=tuple(sorted(artifacts, key=lambda value: value.artifact_id)),
    )
    (authority_root / POST_OBSERVED_AUTHORITY_MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    admit_post_observed_strategy_authority(authority_root)
    return manifest


def materialize(
    *, research_closure: Path, base_workspace: Path, destination_workspace: Path
) -> dict[str, object]:
    """Build an independent workspace and publish it only after full admission."""

    source_workspace = base_workspace.resolve()
    destination = destination_workspace.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise MaterializationError("destination_workspace_not_empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    ).resolve()
    try:
        source_manifest = read_research_workspace_manifest(source_workspace)
        source_runtime = source_workspace / "runtime" / "artifacts"
        target_runtime = temporary / "runtime" / "artifacts"
        for relative in SHARED_RUNTIME_ARTIFACT_ROOTS:
            _copy_tree(source_runtime / relative, target_runtime / relative)
        bindings: list[ResearchWorkspaceArtifact] = []
        for binding in source_manifest.strategy_artifacts:
            source = source_workspace / Path(binding.relative_path)
            target = temporary / Path(binding.relative_path)
            _copy_tree(source, target)
            bindings.append(binding)
        if source_manifest.evidence_review is not None:
            _copy_tree(source_workspace / "authority", temporary / "authority")
            semantic = source_workspace / "evidence-cro-authority"
            if semantic.is_dir():
                _copy_tree(semantic, temporary / "evidence-cro-authority")
        existing_authority = next(
            (b for b in bindings if b.artifact_key == POST_OBSERVED_STRATEGY_AUTHORITY_ROOT_KEY),
            None,
        )
        authority_relative = (
            existing_authority.relative_path
            if existing_authority is not None
            else "post-observed-strategy-authority"
        )
        authority_root = temporary / authority_relative
        if existing_authority is not None:
            authority_hash = admit_post_observed_strategy_authority(authority_root).authority_hash
        else:
            authority_root.mkdir(parents=True, exist_ok=False)
            authority_hash = _materialize_authority(
                closure=research_closure.resolve(), authority_root=authority_root
            ).authority_hash
            bindings.append(
                ResearchWorkspaceArtifact(
                    artifact_key=POST_OBSERVED_STRATEGY_AUTHORITY_ROOT_KEY,
                    relative_path=authority_relative,
                )
            )
        workspace_manifest = ResearchWorkspaceManifest.create(
            workspace_id=destination.name,
            default_strategy_package_id=BROAD_FEATURE_STRATEGY_ID,
            default_score_source_mode="HISTORICAL_ARRAY_REPLAY",
            strategy_artifacts=tuple(bindings),
            evidence_review=source_manifest.evidence_review,
        )
        publish_research_workspace_manifest(temporary, workspace_manifest)
        receipt_body: dict[str, object] = {
            "kind": "PostGovernanceStrategyMaterializationReceipt",
            "research_closure": str(research_closure.resolve()),
            "research_repository_sha": POST_OBSERVED_RESEARCH_SHA,
            "base_workspace": str(source_workspace),
            "destination_workspace": str(destination),
            "authority_hash": authority_hash,
            "workspace_manifest_hash": workspace_manifest.manifest_hash,
            "default_strategy_package_id": BROAD_FEATURE_STRATEGY_ID,
            "installed_strategy_package_ids": [
                "RETURN_G6_MU_ONLY",
                "BALANCED_G2_G6_EQUAL_CAPITAL",
            ],
        }
        receipt = {**receipt_body, "receipt_hash": canonical_hash(receipt_body)}
        (temporary / "post-governance-materialization-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            destination.rmdir()
        os.replace(temporary, destination)
        return receipt
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def prepare_model_lifecycle(
    *,
    research_closure: Path,
    workspace: Path,
    component_id: str = "G6_R0_FAST_REBOUND",
    renewal_through: date | None = None,
) -> dict[str, object]:
    """One setup lease; never replace a running task's source authority."""
    if component_id not in {"G2_R0_TREND", "G6_R0_FAST_REBOUND"}:
        raise MaterializationError("lifecycle_component_not_admitted")
    from alphalattice.control.task_control.registry import (
        DuckDbTaskControlRegistry,
        resolve_task_control_database,
    )
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease

    with closing(WorkspaceWriterLease.acquire(workspace.resolve())):
        if any(
            t.lifecycle.value not in {"SUCCEEDED", "BLOCKED", "CANCELLED"}
            for t in DuckDbTaskControlRegistry.read_existing_tasks(
                resolve_task_control_database(workspace)
            )
        ):
            raise MaterializationError("lifecycle_unsettled_tasks")
        return _prepare_model_lifecycle(
            research_closure=research_closure,
            workspace=workspace,
            component_id=component_id,
            renewal_through=renewal_through,
        )


def _prepare_model_lifecycle(
    *, research_closure: Path, workspace: Path, component_id: str, renewal_through: date | None
) -> dict[str, object]:
    """Admit the July quarter QA transition, retaining prior models but fitting none."""
    from alphalattice.capabilities.alpha_modeling.contracts import AlphaEstimatorContent
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceScoreInput,
    )
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )
    from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
        INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    )
    from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (
        AlphaRuntimeHeterogeneousPredictionOwner,
    )
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        AlphaImportedChild,
        AlphaModelLifecycleAdmission,
        AlphaTrainingObservations,
        prepare_alpha_refit,
    )
    from alphalattice.investment.alpha_research.scores.product_lifecycle import (
        AlphaModelLifecycleRecipe,
        resolve_alpha_refit_plan,
    )

    previous = read_research_workspace_manifest(workspace)
    if previous.decision_updates:
        raise MaterializationError("lifecycle_checkpoint_already_admitted")
    observations_hash = prepare_model_training_inputs(
        research_closure=research_closure,
        workspace=workspace,
        component_id=component_id,
        include_training_support=renewal_through is not None,
    )
    store = AlphaCurrentArtifactStore(workspace.resolve() / "artifacts")
    observations = store._load(
        "lifecycle-training-observations",
        observations_hash,
        "content_hash",
        AlphaTrainingObservations,
    )
    source = store.load_frozen_observations(observations.observation_hash)
    component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id)
    lifecycle = AlphaModelLifecycleRecipe.from_component(component)
    index = _json(
        _verified_file(
            research_closure / SOURCE_FILES["model_index"][0], SOURCE_FILES["model_index"][1]
        )
    )
    role = index[component.component_id]
    family = "deployment_boosters" if "deployment_boosters" in role else "deployment"
    files = role[family]
    entries = {v["relative_path"]: v for v in files["files"]}
    environment = AlphaRuntimeHeterogeneousPredictionOwner(
        component_ids=(component.component_id,)
    ).environment_hash
    prepared, children = [], []
    start = date(2026, 6, 29)
    end = renewal_through or date(2026, 7, 29)
    if end < start or end > source.formation_sessions[-1]:
        raise MaterializationError("lifecycle_qa_source_support_invalid")
    initial_vintages = set(lifecycle.vintages(start))
    required_vintages = {
        v
        for day in source.formation_sessions
        if start <= day <= end
        for v in lifecycle.vintages(day)
    }
    selected_vintages = initial_vintages if renewal_through is not None else required_vintages
    for vintage in sorted(selected_vintages):
        plan = resolve_alpha_refit_plan(
            lifecycle=lifecycle,
            vintage=vintage,
            sessions=source.formation_sessions,
            component_recipe_hash=component.recipe_hash,
            source_binding_hash=observations_hash,
            ordered_listing_ids=source.ordered_listing_ids,
            ordered_feature_ids=component.ordered_feature_ids,
        )
        current = prepare_alpha_refit(
            store, plan=plan, observations=observations, component=component
        )
        prepared.append(current)
        if vintage not in initial_vintages:
            continue  # the new quarter must be produced, not installed from the oracle
        for seed in lifecycle.seeds:
            key = (
                f"boosters/R0_CORE/deployment-{vintage}-seed-{seed}.txt"
                if family == "deployment_boosters"
                else f"boosters/deployment/{vintage}-seed-{seed}.json"
            )
            entry = entries[key]
            path = _verified_file(Path(files["root"]) / entry["relative_path"], entry["sha256"])
            payload = path.read_bytes()
            estimator = AlphaEstimatorContent.model_validate_json(payload)
            if estimator.ordered_feature_ids != component.ordered_feature_ids:
                raise MaterializationError("lifecycle_model_feature_axis_invalid")
            if (
                store._publish_packed_bytes(category="current/imported-models", payload=payload)
                != entry["sha256"]
            ):
                raise MaterializationError("lifecycle_imported_payload_identity_invalid")
            child = AlphaImportedChild.create(
                prepared_hash=current.content_hash,
                lifecycle_hash=lifecycle.content_hash,
                vintage=vintage,
                seed=seed,
                estimator_hash=estimator.content_hash,
                payload_hash=entry["sha256"],
                source_index_hash=SOURCE_FILES["model_index"][1],
                environment_hash=environment,
            )
            store._publish("lifecycle-imported-children", child, "content_hash")
            children.append(child)
    admission = AlphaModelLifecycleAdmission.create(
        component=component,
        lifecycle=lifecycle,
        observations_hash=observations_hash,
        prepared=tuple(prepared),
        initial_children=tuple(children),
        formation_start=start,
        formation_end=max(
            day
            for day in source.formation_sessions
            if start <= day <= end
            and (renewal_through is None or set(lifecycle.vintages(day)) <= selected_vintages)
        ),
        maximum_fit_attempts=6,
        environment_hash=environment,
        fit_vintages=tuple(sorted(required_vintages - initial_vintages)),
        training_factor_ids=tuple(sorted(source.formula_values))
        if renewal_through is not None
        else (),
        renewal_through=renewal_through,
    )
    relative = f"authority/model-lifecycle/{admission.content_hash}"
    path = workspace.resolve() / relative / "model-lifecycle-admission.json"
    payload = admission.model_dump_json(indent=2).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise MaterializationError("lifecycle_existing_admission_changed")
    else:
        store._atomic_write(path, payload)
    package_id = (
        "RETURN_G6_MU_ONLY"
        if component_id == "G6_R0_FAST_REBOUND"
        else "BALANCED_G2_G6_EQUAL_CAPITAL"
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        admit_research_workspace,
    )

    catalog = admit_research_workspace(workspace).require_catalog()
    package = catalog.select(
        strategy_id=package_id, score_source_mode="HISTORICAL_ARRAY_REPLAY"
    ).package
    binding = ResearchWorkspaceScoreInput(
        strategy_package_id=package_id,
        strategy_package_hash=package.package_hash,
        component_id=component_id
        if renewal_through is not None or component_id != "G6_R0_FAST_REBOUND"
        else None,
        authority_relative_path=relative,
        authority_hash=admission.content_hash,
        source_kind="WORKSPACE_DATA_FEATURE",
        observation_snapshot_hash=None,
    )
    replacements = [binding]
    if component_id == "G6_R0_FAST_REBOUND" and renewal_through is not None:
        balanced = catalog.select(
            strategy_id="BALANCED_G2_G6_EQUAL_CAPITAL", score_source_mode="HISTORICAL_ARRAY_REPLAY"
        ).package
        replacements.append(
            binding.model_copy(
                update={
                    "strategy_package_id": balanced.strategy_id,
                    "strategy_package_hash": balanced.package_hash,
                }
            )
        )
    retained = tuple(
        v
        for v in previous.score_inputs or ()
        if not any(
            v.strategy_package_id == new.strategy_package_id
            and v.component_id in {None, component_id}
            for new in replacements
        )
    )
    if any(
        old.component_id is None
        and any(old.strategy_package_id == new.strategy_package_id for new in replacements)
        and len(catalog.packages_by_strategy_id()[old.strategy_package_id].component_ids) != 1
        for old in previous.score_inputs or ()
    ):
        raise MaterializationError("lifecycle_existing_component_binding_ambiguous")
    updated = ResearchWorkspaceManifest.create(
        **{
            **{
                k: getattr(previous, k)
                for k in type(previous).model_fields
                if k not in {"kind", "manifest_schema", "manifest_hash"}
            },
            "score_inputs": (*retained, *replacements),
        }
    )
    publish_research_workspace_manifest(workspace, updated)
    return {
        "admission_hash": admission.content_hash,
        "observations_hash": observations_hash,
        "fits": 0,
    }


def prepare_model_training_inputs(
    *,
    research_closure: Path,
    workspace: Path,
    component_id: str = "G6_R0_FAST_REBOUND",
    include_training_support: bool = False,
) -> str:
    """Materialize the admitted G6 observation/label authority, without fitting.

    Only this one-off boundary reads research roots. Runtime receives local
    content-addressed arrays, not Python modules or external path references.
    """
    from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
        FrozenPriceVolumeInputs,
    )
    from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
        preflight_panel_feature_plan,
    )
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        publish_training_observations,
    )

    # The checker names scripts by repository namespace; the runtime bootstrap
    # intentionally loads only this script directory, even outside the repo cwd.
    if TYPE_CHECKING:
        from scripts.broad_ensemble_research_closure import load_cached_panel_source
    else:
        from broad_ensemble_research_closure import load_cached_panel_source

    descriptor, _, _ = _research_inputs(research_closure.resolve())
    index = _json(
        _verified_file(
            research_closure / SOURCE_FILES["model_index"][0], SOURCE_FILES["model_index"][1]
        )
    )
    research_root = Path(index["G6_R0_FAST_REBOUND"]["deployment_boosters"]["root"])
    cache = (
        Path(descriptor["sizing"]["mu.iv0"]["curve_owner"]["absolute_path"]).parent
        / "cache/current-source"
    )
    manifest = _json(cache / "manifest.json")
    if manifest["cache_identity_hash"] != (
        "290a8d656177abc28e4c7d00b5c737a4a73b025c509d78029e89215c0ff792d0"
    ) or manifest["cache_identity_hash"] != canonical_hash(
        {key: value for key, value in manifest.items() if key != "cache_identity_hash"}
    ):
        raise MaterializationError("training_source_identity_invalid")
    raw_root = research_root / "cache/forward-qualified-source"
    raw = _json(raw_root / "qualified-source-manifest.json")
    if (
        raw["identity"]["formation_sessions"] != manifest["formation_sessions"]
        or raw["identity"]["ordered_listing_ids"] != manifest["ordered_listing_ids"]
        or canonical_hash(
            {
                **raw["identity"],
                "array_hashes": {name: value["sha256"] for name, value in raw["arrays"].items()},
            }
        )
        != raw["source_identity_hash"]
    ):
        raise MaterializationError("training_observation_axis_invalid")

    def array(root: Path, relative: str, digest: str) -> np.ndarray[Any, Any]:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise MaterializationError("training_source_path_outside_root")
        return cast(
            np.ndarray[Any, Any],
            np.load(_verified_file(path, digest), mmap_mode="r", allow_pickle=False),
        )

    fields = manifest["arrays"]
    market = array(
        cache, fields["market_context_values"]["path"], fields["market_context_values"]["sha256"]
    )
    sector = array(
        cache, fields["sector_context_values"]["path"], fields["sector_context_values"]["sha256"]
    )
    targets = array(
        cache, fields["total_return_target_z"]["path"], fields["total_return_target_z"]["sha256"]
    )
    prices = {
        name: array(raw_root, f"qualified-{name}.npy", raw["arrays"][name]["sha256"])
        for name in ("open", "high", "low", "close", "volume")
    }
    source = FrozenPriceVolumeInputs(
        formation_sessions=tuple(date.fromisoformat(v) for v in manifest["formation_sessions"]),
        ordered_listing_ids=tuple(manifest["ordered_listing_ids"]),
        sector_by_listing_id=manifest["sector_by_listing_id"],
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        volume=prices["volume"],
        market_context_values=market[:, [3, 8, 15]],
        sector_trend_values=sector[:, :, 1],
        source_binding_hash=raw["source_identity_hash"],
    )
    # Eligibility is an independent source lane; do not infer it from labels.
    panel_source = load_cached_panel_source(cache)
    eligible = preflight_panel_feature_plan(
        source=panel_source,
        selected_method_ids=("SPARSE_SESSION_AMPLITUDE",),
        maximum_aggregation_span=1,
    ).common_row_mask
    store = AlphaCurrentArtifactStore(workspace.resolve() / "artifacts")
    observation = store.publish_frozen_observations(source, disposition="RECORDED_INPUT_QA")
    if include_training_support or component_id != "G6_R0_FAST_REBOUND":
        from dataclasses import replace

        from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
            INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
        )
        from alphalattice.investment.alpha_research.scores.model_renewal import (
            publish_component_training_observations,
        )

        source = replace(
            store.load_frozen_observations(observation.snapshot_hash),
            formula_values={
                name: panel_source.raw_formula_values[:, :, i]
                for i, name in enumerate(panel_source.ordered_factor_ids)
            },
        )
        return publish_component_training_observations(
            store,
            component=INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id),
            source=source,
            training=panel_source,
        ).content_hash
    return str(
        publish_training_observations(
            store,
            observation_hash=observation.snapshot_hash,
            target_method_id="EXACT_FROZEN_G0_H1_WHOLE_UNIVERSE_TARGET",
            target_authority_hash=manifest["cache_identity_hash"],
            label_available_sessions=tuple(
                date.fromisoformat(v) for v in manifest["holding_end_sessions"]
            ),
            targets=targets,
            eligible=eligible,
        ).content_hash
    )


def prepare_score_inputs(
    *,
    research_closure: Path,
    workspace: Path,
    model_closure: Path,
    strategy_package_id: str = "RETURN_G6_MU_ONLY",
    recorded_inputs: bool = False,
) -> dict[str, Any]:
    """Install a bounded QA inference authority into an existing workspace.

    All external paths are used here only. Models and optional recorded inputs
    become product-owned immutable files; the regular launcher stays unchanged.
    """
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceScoreInput,
        admit_research_workspace,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.control.task_control.registry import (
        DuckDbTaskControlRegistry,
        resolve_task_control_database,
    )
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
    from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
        FrozenPriceVolumeInputs,
    )
    from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
        _fit_temporal_scale,
    )
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )
    from alphalattice.investment.alpha_research.scores.frozen_inference import (
        FrozenComponentInferenceAuthority,
        FrozenMarketScale,
        admit_frozen_inference,
    )
    from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (
        HeterogeneousClosureFile,
        admit_heterogeneous_current_closure,
    )

    descriptor, _, _ = _research_inputs(research_closure.resolve())
    index = _json(
        _verified_file(
            research_closure / SOURCE_FILES["model_index"][0], SOURCE_FILES["model_index"][1]
        )
    )
    origin = admit_heterogeneous_current_closure(model_closure.resolve())
    component_id = "G6_R0_FAST_REBOUND"
    component = origin.authority.component(component_id)
    closure_component = next(
        value for value in origin.manifest.components if value.component_id == component_id
    )
    research_root = Path(index[component_id]["deployment_boosters"]["root"]).resolve()
    cache = (
        Path(descriptor["sizing"]["mu.iv0"]["curve_owner"]["absolute_path"]).parent
        / "cache/current-source"
    )
    source_manifest = _json(cache / "manifest.json")
    if source_manifest[
        "cache_identity_hash"
    ] != "290a8d656177abc28e4c7d00b5c737a4a73b025c509d78029e89215c0ff792d0" or source_manifest[
        "cache_identity_hash"
    ] != canonical_hash({k: v for k, v in source_manifest.items() if k != "cache_identity_hash"}):
        raise MaterializationError("scoring_training_source_identity_invalid")
    market_info = source_manifest["arrays"]["market_context_values"]
    market = np.load(
        _verified_file(cache / market_info["path"], market_info["sha256"]),
        mmap_mode="r",
        allow_pickle=False,
    )[:, [3, 8, 15]]
    market = np.array(market, copy=True)
    market[1:, 0] = market[:-1, 0]
    market[0, 0] = np.nan
    sessions = tuple(date.fromisoformat(value) for value in source_manifest["formation_sessions"])
    session_lookup = {value: index for index, value in enumerate(sessions)}
    scales = []
    for surface in closure_component.surfaces:
        matrix = _json(
            research_root / "cache/deployment/R0_CORE" / surface.vintage / "manifest.json"
        )
        if (
            matrix["identity_hash"] != surface.source_binding_hash
            or canonical_hash(matrix["identity"]) != matrix["identity_hash"]
        ):
            raise MaterializationError("scoring_training_matrix_identity_invalid")
        training = tuple(
            date.fromisoformat(value) for value in matrix["identity"]["training_sessions"]
        )
        _, parameters, _ = _fit_temporal_scale(
            market, np.asarray([session_lookup[value] for value in training], dtype=np.int64)
        )
        scales.append(
            FrozenMarketScale(
                vintage=surface.vintage,
                training_sessions=training,
                source_binding_hash=source_manifest["cache_identity_hash"],
                center=parameters["center"],
                scale=parameters["scale"],
            )
        )

    destination = workspace.resolve()
    if not (destination / "research-workspace.json").is_file():
        raise MaterializationError("scoring_existing_workspace_required")
    with closing(WorkspaceWriterLease.acquire(destination)):
        admitted = admit_research_workspace(destination)
        selected = (
            admitted.require_catalog()
            .select(strategy_id=strategy_package_id, score_source_mode="HISTORICAL_ARRAY_REPLAY")
            .package
        )
        if component_id not in selected.component_ids:
            raise MaterializationError("scoring_package_component_invalid")
        registry_paths = (resolve_task_control_database(destination),)
        if any(
            task.lifecycle
            not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
            for path in registry_paths
            for task in DuckDbTaskControlRegistry.read_existing_tasks(path)
        ):
            raise MaterializationError("scoring_unsettled_tasks")
        with tempfile.TemporaryDirectory(
            prefix=".component-inference-", dir=destination
        ) as staging:
            target = Path(staging)
            children = []
            known_files = {
                value["relative_path"]: value
                for value in index[component_id]["deployment_boosters"]["files"]
            }
            for child in closure_component.children:
                source = child.payload.resolve_under(model_closure)
                declared = known_files[
                    f"boosters/R0_CORE/deployment-{child.vintage}-seed-{child.seed}.txt"
                ]
                if child.payload.sha256 != declared["sha256"]:
                    raise MaterializationError("scoring_model_index_mismatch")
                _verified_file(source, declared["sha256"])
                relative = f"models/{child.vintage}-{child.seed}.json"
                copied = target / relative
                copied.parent.mkdir(exist_ok=True)
                shutil.copy2(source, copied)
                children.append(
                    child.model_copy(
                        update={
                            "payload": HeterogeneousClosureFile(
                                relative_path=relative,
                                sha256=_sha256(copied),
                                byte_count=copied.stat().st_size,
                            )
                        }
                    )
                )
            authority = FrozenComponentInferenceAuthority.create(
                model_set=component,
                children=tuple(children),
                market_scales=tuple(scales),
                gate_m_receipt_hash=origin.manifest.gate_m_receipt_hash,
                numerical_environment_hash=origin.manifest.numerical_environment_hash,
            )
            (target / "inference-authority.json").write_text(
                authority.model_dump_json(indent=2), encoding="utf-8"
            )
            admit_frozen_inference(target, expected_hash=authority.authority_hash)
            relative_root = f"authority/component-inference/{authority.authority_hash}"
            final_root = destination / relative_root
            if final_root.exists():
                admit_frozen_inference(final_root, expected_hash=authority.authority_hash)
            else:
                final_root.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, final_root)

        snapshot_hash = None
        if recorded_inputs:
            raw_root = research_root / "cache/forward-qualified-source"
            raw_manifest = _json(raw_root / "qualified-source-manifest.json")
            if (
                raw_manifest["identity"]["formation_sessions"]
                != source_manifest["formation_sessions"]
                or raw_manifest["identity"]["ordered_listing_ids"]
                != source_manifest["ordered_listing_ids"]
            ):
                raise MaterializationError("scoring_recorded_axis_mismatch")
            prices = {
                name: np.load(
                    _verified_file(raw_root / f"qualified-{name}.npy", value["sha256"]),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name, value in raw_manifest["arrays"].items()
            }
            sector_info = source_manifest["arrays"]["sector_context_values"]
            sectors = np.load(
                _verified_file(cache / sector_info["path"], sector_info["sha256"]),
                mmap_mode="r",
                allow_pickle=False,
            )
            raw_market = np.load(cache / market_info["path"], mmap_mode="r", allow_pickle=False)[
                :, [3, 8, 15]
            ]
            inputs = FrozenPriceVolumeInputs(
                formation_sessions=sessions,
                ordered_listing_ids=tuple(source_manifest["ordered_listing_ids"]),
                sector_by_listing_id=source_manifest["sector_by_listing_id"],
                **prices,
                market_context_values=raw_market,
                sector_trend_values=sectors[:, :, 1],
                source_binding_hash=raw_manifest["source_identity_hash"],
            )
            snapshot_hash = (
                AlphaCurrentArtifactStore(destination / "artifacts")
                .publish_frozen_observations(inputs, disposition="RECORDED_INPUT_QA")
                .snapshot_hash
            )
        binding = ResearchWorkspaceScoreInput(
            strategy_package_id=strategy_package_id,
            strategy_package_hash=selected.package_hash,
            authority_relative_path=relative_root,
            authority_hash=authority.authority_hash,
            source_kind="RECORDED_INPUT_SNAPSHOT" if recorded_inputs else "WORKSPACE_DATA_FEATURE",
            observation_snapshot_hash=snapshot_hash,
        )
        previous = admitted.manifest
        bindings = [
            value
            for value in previous.score_inputs or ()
            if value.strategy_package_id != strategy_package_id
        ]
        bindings.append(binding)
        updated = ResearchWorkspaceManifest.create(
            workspace_id=previous.workspace_id,
            default_strategy_package_id=previous.default_strategy_package_id,
            default_score_source_mode=previous.default_score_source_mode,
            strategy_artifacts=previous.strategy_artifacts,
            evidence_review=previous.evidence_review,
            data_update=previous.data_update,
            score_inputs=tuple(sorted(bindings, key=lambda value: value.strategy_package_id)),
            calibration_inputs=previous.calibration_inputs,
            decision_updates=previous.decision_updates,
        )
        publish_research_workspace_manifest(destination, updated)
        return {
            "kind": "FrozenScoreInputMaterialization",
            "authority_hash": authority.authority_hash,
            "workspace_manifest_hash": updated.manifest_hash,
            "source_kind": binding.source_kind,
            "observation_snapshot_hash": snapshot_hash,
            "model_count": len(children),
            "fits": 0,
        }


def prepare_calibration_inputs(
    *,
    research_closure: Path,
    workspace: Path,
    strategy_package_id: str = "RETURN_G6_MU_ONLY",
) -> dict[str, Any]:
    """Copy raw causal evidence, never the zero-filled replay outcome lane.

    Source files are verified through the frozen Gate-V snapshot. The resulting
    packed seed is owned by Portfolio; no absolute path enters its runtime data.
    The retained curve is checked here as an oracle, not installed as an answer.
    """
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceCalibrationInput,
        admit_research_workspace,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle
    from alphalattice.control.task_control.registry import (
        DuckDbTaskControlRegistry,
        resolve_task_control_database,
    )
    from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )
    from alphalattice.investment.alpha_research.scores.model_renewal import (
        admit_component_inference,
    )
    from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
        FrozenRankCalibrationRule,
        publish_observations,
    )
    from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import (
        complete_matured_rank_curve,
        rank_bucket_observations,
    )
    from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
        PortfolioResearchArtifactStore,
    )

    descriptor, _, gate = _research_inputs(research_closure.resolve())
    snapshot_path, snapshot_hash = _source_asset(
        gate["artifacts"]["source-snapshot-before.json"], label="calibration_source_snapshot"
    )
    snapshot = _json(snapshot_path)
    surface = descriptor["roles"]["G6_R0_FAST_REBOUND"]["historical_score_surface"]
    score_path, score_hash = _source_asset(surface["artifact"], label="calibration_scores")
    manifest_path, manifest_hash = _source_asset(
        surface["manifest"], label="calibration_score_axis"
    )
    identity = _json(manifest_path)["identity"]
    sessions = tuple(date.fromisoformat(v) for v in identity["sessions"][4:])
    listings = tuple(identity["listings"])
    scores = np.ascontiguousarray(np.load(score_path, allow_pickle=False)[4:], dtype=np.float64)
    returns = np.full(scores.shape, np.nan, dtype=np.float64)
    eligible = np.zeros(scores.shape, dtype=np.bool_)
    columns = {v: i for i, v in enumerate(listings)}
    hashes = [POST_OBSERVED_DESCRIPTOR_SHA256, snapshot_hash, score_hash, manifest_hash]
    for prefix, manifest_name, session_key, listing_key, development in (
        ("_panel_shared", "ready.json", "sessions", "listing_ids", True),
        (
            "current-successor-tail/composite",
            "manifest.json",
            "formation_sessions",
            "ordered_listing_ids",
            False,
        ),
    ):
        files = {}
        for name in (manifest_name, "returns.npy", "eligible.npy"):
            path, digest = _snapshot_asset(
                snapshot, suffix=f"{prefix}/{name}", label=f"calibration_{prefix}_{name}"
            )
            files[name] = path
            hashes.append(digest)
        axes = _json(files[manifest_name])
        source_rows = {str(v)[:10]: i for i, v in enumerate(axes[session_key])}
        source_columns = list(axes[listing_key])
        rows = [i for i, day in enumerate(sessions) if (day <= date(2024, 7, 29)) == development]
        source_indices = [source_rows[sessions[i].isoformat()] for i in rows]
        target_columns = [columns[v] for v in source_columns]
        for target, name in ((returns, "returns.npy"), (eligible, "eligible.npy")):
            values = np.load(files[name], allow_pickle=False)
            target[np.ix_(rows, target_columns)] = values[source_indices]
    full_path, full_hash = _snapshot_asset(
        snapshot,
        suffix="frozen-iw184-holdout-smoke/cache/current-source/manifest.json",
        label="calibration_full_clock",
    )
    full = tuple(date.fromisoformat(v) for v in _json(full_path)["formation_sessions"])
    lookup = {v: i for i, v in enumerate(full)}
    ends = tuple(full[lookup[v] + 2] if lookup[v] + 2 < len(full) else None for v in sessions)
    hashes.append(full_hash)
    mu = descriptor["sizing"]["mu.iv0"]
    rule = FrozenRankCalibrationRule.create(
        activation_session=date.fromisoformat(mu["maturity_start"]),
        activation_position=mu["warmup_sessions"],
    )
    if sessions.index(rule.activation_session) != rule.activation_position:
        raise MaterializationError("calibration_activation_axis_mismatch")
    observed = rank_bucket_observations(
        scores=scores,
        realized_simple_returns=returns,
        decision_eligible=eligible,
        rank_keys=np.arange(len(listings)),
    )
    curve, newest = complete_matured_rank_curve(
        observations=observed,
        observation_sessions=sessions,
        holding_end_sessions=ends,
        decision_sessions=sessions,
    )
    for name, actual in (("curve", curve), ("latest_calibration_row", newest)):
        oracle_path, digest = _source_asset(mu[name], label=f"calibration_{name}_oracle")
        oracle = np.load(oracle_path, allow_pickle=False)
        if not np.array_equal(actual, oracle, equal_nan=True):
            raise MaterializationError(f"calibration_frozen_parity_failed:{name}")
        hashes.append(digest)
    destination = workspace.resolve()
    with closing(WorkspaceWriterLease.acquire(destination)):
        admitted = admit_research_workspace(destination)
        scoring = next(
            (
                v
                for v in admitted.manifest.score_inputs or ()
                if v.strategy_package_id == strategy_package_id
            ),
            None,
        )
        if scoring is None:
            raise MaterializationError("calibration_score_input_not_installed")
        package = admitted.require_catalog().packages_by_strategy_id()[strategy_package_id]
        control = package.controls.frozen_control("weight_rule")
        if control is None or control.frozen_display != "mu.iv0":
            raise MaterializationError("calibration_package_does_not_consume_mu")
        authority = admit_component_inference(
            destination / scoring.authority_relative_path,
            expected_hash=scoring.authority_hash,
            store=AlphaCurrentArtifactStore(destination / "artifacts"),
        )
        if any(
            t.lifecycle
            not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.CANCELLED, TaskLifecycle.BLOCKED}
            for t in DuckDbTaskControlRegistry.read_existing_tasks(
                resolve_task_control_database(destination)
            )
        ):
            raise MaterializationError("calibration_unsettled_tasks")
        store = PortfolioResearchArtifactStore(destination / "artifacts")
        if not store.root.resolve().is_relative_to(destination):
            raise MaterializationError("calibration_artifact_root_outside_workspace")
        seed = publish_observations(
            store,
            scores=scores,
            returns=returns,
            eligible=eligible,
            origin="FROZEN_RESEARCH_INPUTS",
            strategy_package_hash=package.package_hash,
            component_recipe_hash=authority.authority.model_set.recipe_hash,
            rule=rule,
            formation_sessions=sessions,
            holding_end_sessions=ends,
            ordered_listing_ids=listings,
            source_hashes=tuple(hashes),
        )
        previous = admitted.manifest
        bindings = [
            v
            for v in previous.calibration_inputs or ()
            if v.strategy_package_id != strategy_package_id
        ]
        bindings.append(
            ResearchWorkspaceCalibrationInput(
                strategy_package_id=strategy_package_id,
                strategy_package_hash=package.package_hash,
                seed_hash=seed.content_hash,
                source_kind=scoring.source_kind,
            )
        )
        updated = ResearchWorkspaceManifest.create(
            workspace_id=previous.workspace_id,
            default_strategy_package_id=previous.default_strategy_package_id,
            default_score_source_mode=previous.default_score_source_mode,
            strategy_artifacts=previous.strategy_artifacts,
            evidence_review=previous.evidence_review,
            data_update=previous.data_update,
            score_inputs=previous.score_inputs,
            calibration_inputs=tuple(sorted(bindings, key=lambda v: v.strategy_package_id)),
            decision_updates=previous.decision_updates,
        )
        publish_research_workspace_manifest(destination, updated)
    return {
        "kind": "CalibrationInputMaterialization",
        "seed_hash": seed.content_hash,
        "workspace_manifest_hash": updated.manifest_hash,
        "curve_bitwise_equal": True,
        "newest_rows_exact": True,
        "fits": 0,
        "source_snapshot_hash": snapshot_hash,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-closure", type=Path, required=True)
    parser.add_argument("--base-workspace", type=Path, required=True)
    parser.add_argument("--destination-workspace", type=Path, required=True)
    parser.add_argument("--prepare-score-inputs", action="store_true")
    parser.add_argument("--recorded-score-inputs", action="store_true")
    parser.add_argument("--prepare-calibration-inputs", action="store_true")
    parser.add_argument("--prepare-model-lifecycle", action="store_true")
    parser.add_argument(
        "--model-component",
        choices=("G2_R0_TREND", "G6_R0_FAST_REBOUND"),
        default="G6_R0_FAST_REBOUND",
    )
    parser.add_argument("--renewal-through", type=date.fromisoformat)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.prepare_model_lifecycle and (
        args.renewal_through is not None or args.model_component != "G6_R0_FAST_REBOUND"
    ):
        raise MaterializationError("model_options_require_lifecycle_preparation")
    if args.prepare_model_lifecycle:
        if (
            args.prepare_score_inputs
            or args.recorded_score_inputs
            or args.prepare_calibration_inputs
        ):
            raise MaterializationError("choose_one_preparation")
        print(
            json.dumps(
                prepare_model_lifecycle(
                    research_closure=args.research_closure,
                    workspace=args.destination_workspace,
                    component_id=args.model_component,
                    renewal_through=args.renewal_through,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.prepare_calibration_inputs:
        if args.prepare_score_inputs or args.recorded_score_inputs:
            raise MaterializationError("choose_one_preparation")
        receipt = prepare_calibration_inputs(
            research_closure=args.research_closure, workspace=args.destination_workspace
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.prepare_score_inputs:
        base = read_research_workspace_manifest(args.base_workspace)
        model = next(
            (
                value
                for value in base.strategy_artifacts
                if value.artifact_key == "HETEROGENEOUS_CURRENT_CLOSURE_ROOT"
            ),
            None,
        )
        if model is None:
            raise MaterializationError("scoring_model_closure_not_installed")
        receipt = prepare_score_inputs(
            research_closure=args.research_closure,
            workspace=args.destination_workspace,
            model_closure=args.base_workspace / model.relative_path,
            recorded_inputs=args.recorded_score_inputs,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.recorded_score_inputs:
        raise MaterializationError("recorded_score_inputs_requires_preparation")
    receipt = materialize(
        research_closure=args.research_closure,
        base_workspace=args.base_workspace,
        destination_workspace=args.destination_workspace,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
