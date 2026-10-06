"""Generate a development-only monthly-refit Alpha score lane.

The installed profile chooses an Alpha-owned Feature view; materialization,
fold scaling and LightGBM execution stay on the existing owners.  The monthly
clock is one rolling fit per calendar month with a one-session purge before the
month's first formation.  Output is private research evidence, not publication.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, cast

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"
if str(PLAYPEN_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYPEN_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (  # noqa: E402
    DynamicPanelLightGBMParameters,
    build_dynamic_panel_lightgbm_recipe,
    build_dynamic_panel_lightgbm_search_domain,
)
from alphalattice.capabilities.alpha_modeling.contracts import (  # noqa: E402
    AlphaEstimatorContent,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
    alpha_model_array_content_hash,
)
from alphalattice.capabilities.alpha_modeling.runtime.service import (  # noqa: E402
    AlphaModelRuntimeService,
)
from alphalattice.control.product_host.research_authoring.authority import (  # noqa: E402
    InstalledResearchSnapshot,
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.panel_methodology_sources import (  # noqa: E402
    PanelMethodologySourceRoots,
    resolve_corrected_feature_t_panel_source,
    resolve_corrected_panel_snapshot_hash,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (  # noqa: E402
    build_panel_model_catalog,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (  # noqa: E402
    PanelFeatureProjection,
    materialize_panel_feature_projection,
    preflight_panel_feature_plan,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (  # noqa: E402
    InstalledPanelViewId,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash  # noqa: E402
from alphalattice.protocols.research_authoring.contracts import (  # noqa: E402
    ResearchExperimentEnvelope,
)

ANNUAL_ROOT_HASH = "3ec1b4175eec87901f69752f3a2a1e53998550a4b9ead277ca52eae116329cbb"
ANNUAL_AUTHORITY_HASH = "ab809e997e834cf6f439044ceea85c9fe406611661b574fa93bb5c51e8b07eea"
ANNUAL_SOURCE_RESOLUTION_HASH = "bb06155d2282cd5992988d134c2d108005f9b7f23e9410f1a1af9ac537a89392"
ANNUAL_PANEL_SNAPSHOT_HASH = "5c902b329c34dfd09934906ead549afab95c1df4d8248e3647fdebcd19787f65"

TRAINING_SESSION_COUNT = 756
PURGE_SESSION_COUNT = 1


@dataclass(frozen=True, slots=True)
class MonthlyAlphaRefitProfile:
    profile_id: str
    view_id: InstalledPanelViewId
    model_recipe_id: str
    expected_feature_count: int
    default_output_root: Path
    scientific_disposition: str
    historical_recipe_sha256: str | None = None


PROFILES = {
    "sparse-session-amplitude-195": MonthlyAlphaRefitProfile(
        profile_id="sparse-session-amplitude-195",
        view_id="SPARSE_SESSION_AMPLITUDE",
        model_recipe_id="SPARSE_SESSION_AMPLITUDE_LIGHTGBM",
        expected_feature_count=195,
        default_output_root=Path("workspaces/monthly-alpha-refit-research"),
        scientific_disposition="INSTALLED_FIXED_VIEW",
    ),
    "historical-semantic-intersection": MonthlyAlphaRefitProfile(
        profile_id="historical-semantic-intersection",
        view_id="SPARSE_HISTORICAL_SEMANTIC_INTERSECTION",
        model_recipe_id="SPARSE_HISTORICAL_SEMANTIC_INTERSECTION_LIGHTGBM",
        expected_feature_count=134,
        default_output_root=Path("workspaces/historical-semantic-intersection-alpha-refit"),
        scientific_disposition="CURRENT_195_SEMANTIC_INTERSECTION_NOT_HISTORICAL_REPRODUCTION",
        historical_recipe_sha256=(
            "8baaaa8864ba821b680ac4bc2412a595768b42cf66c6433658fca8766a0bc65a"
        ),
    ),
}


def _profile_identity_fields(profile: MonthlyAlphaRefitProfile) -> dict[str, Any]:
    if profile.profile_id == "sparse-session-amplitude-195":
        return {}
    return {
        "feature_profile_id": profile.profile_id,
        "feature_profile_disposition": profile.scientific_disposition,
        "historical_recipe_sha256": profile.historical_recipe_sha256,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research-root",
        type=Path,
        help="Root of the exact historical research asset layout; required unless consolidating. "
        "This operator driver is not the first-use product research path.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="sparse-session-amplitude-195",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Private output root; defaults to the selected profile's isolated workspace.",
    )
    parser.add_argument(
        "--month",
        help="Run only one YYYY-MM vintage; omit to run/resume the complete surface.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Resolve and validate source/geometry without fitting.",
    )
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help=(
            "Verify existing exact shards and build the combined surface without reopening sources."
        ),
    )
    arguments = parser.parse_args()
    if not arguments.consolidate_only and arguments.research_root is None:
        parser.error("--research-root is required; no developer filesystem is assumed")
    return arguments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _historical_sources(root: Path) -> tuple[Path, Path, PanelMethodologySourceRoots]:
    """Keep the frozen legacy layout, but require its root from the operator."""
    annual = root / (
        "playpen-dynamic-panel-total-return/workspaces/"
        "dynamic-panel-platform-remediation-feature-t/sparse-session-amplitude"
    )
    market = root / "playpen/workspaces/us-research/runtime"
    roots = PanelMethodologySourceRoots(
        repository_root=root / "playpen-dynamic-panel-total-return-workspace-handoff-20260823",
        panel_artifact_root=root
        / (
            "playpen-feature-observation-clock-workspace/"
            "successor-panel-2c200228/workspace/artifacts"
        ),
        legacy_panel_artifact_root=root
        / "playpen-production-refactor/workspaces/panel-remediation/workspace/artifacts",
        feature_artifact_root=root
        / "playpen-production-refactor/workspaces/joint-primary-alpha-campaign/artifacts",
        failed_baseline_workspace=root
        / "playpen-dynamic-panel-total-return-workspace-handoff-20260823",
        sector_context_artifact_root=market / "artifacts",
        execution_outcome_artifact_root=root
        / "playpen-production-refactor/workspaces/stage-one-lawful-panel/evidence/artifacts",
        development_overlay_artifact_root=annual / "feature-development",
    )
    return (
        annual
        / "portfolio-development/portfolio-strategy-lab/development/paired-panel-research/roots"
        / f"{ANNUAL_ROOT_HASH}.json",
        market,
        roots,
    )


def _load_annual_root(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("root_hash") != ANNUAL_ROOT_HASH
        or payload.get("authority_hash") != ANNUAL_AUTHORITY_HASH
        or payload.get("source_resolution_hash") != ANNUAL_SOURCE_RESOLUTION_HASH
        or len(payload.get("common_formation_sessions", ())) != 1260
    ):
        raise RuntimeError("monthly_alpha_refit.annual_root_identity_invalid")
    return cast(dict[str, Any], payload)


def _resolve_source(*, market: Path, roots: PanelMethodologySourceRoots) -> tuple[Any, Any, float]:
    started = perf_counter()
    envelope = ResearchExperimentEnvelope.create(
        kind="alpha.model-development",
        schema_id="alpha-panel-methodology-development",
        data_snapshot_handle="panel-feature-t-successor",
        universe_handle="us-current-index-research",
        sessions={
            "start": "2016-08-01",
            "end": "2025-07-31",
            "as_of": {"session": "2025-07-31", "phase": "OFFICIAL_CLOSE"},
        },
        budget={"maximum_candidates": 1, "maximum_numerical_calls": 128},
        determinism={"seed": 1729, "thread_limit": 1, "network_disabled": True},
        output_workspace="workspaces/monthly-alpha-refit-research",
        publication_intent="DEVELOPMENT_EVIDENCE_ONLY",
    )
    snapshot_hash = resolve_corrected_panel_snapshot_hash(roots=roots)
    if snapshot_hash != ANNUAL_PANEL_SNAPSHOT_HASH:
        raise RuntimeError("monthly_alpha_refit.panel_snapshot_identity_mismatch")
    authority = WorkspaceResearchAuthorityResolver(
        workspace=market,
        artifact_root=roots.panel_artifact_root,
        installed_snapshots=(
            InstalledResearchSnapshot(
                handle="panel-feature-t-successor",
                panel_snapshot_hash=snapshot_hash,
            ),
        ),
    ).resolve(envelope)
    if authority.authority_hash != ANNUAL_AUTHORITY_HASH:
        raise RuntimeError(
            f"monthly_alpha_refit.authority_identity_mismatch:{authority.authority_hash}"
        )
    resolved = resolve_corrected_feature_t_panel_source(
        roots=roots,
        authority=authority,
        input_method_id="FEATURE_T_PANEL_TOTAL_RETURN_INPUTS",
        development_overlay_method_id="SESSION_OBSERVATION_FORMULA_OVERLAY",
    )
    if resolved.resolution.resolution_hash != ANNUAL_SOURCE_RESOLUTION_HASH:
        raise RuntimeError(
            "monthly_alpha_refit.source_resolution_identity_mismatch:"
            f"{resolved.resolution.resolution_hash}"
        )
    return authority, resolved, perf_counter() - started


def _monthly_geometry(
    *, source_sessions: tuple[date, ...], validation_sessions: tuple[date, ...]
) -> tuple[tuple[str, tuple[date, ...], tuple[date, ...], date], ...]:
    positions = {value: index for index, value in enumerate(source_sessions)}
    grouped: dict[str, list[date]] = defaultdict(list)
    for session in validation_sessions:
        if session not in positions:
            raise RuntimeError("monthly_alpha_refit.validation_session_not_in_source")
        grouped[f"{session.year:04d}-{session.month:02d}"].append(session)
    output = []
    for month, month_sessions in sorted(grouped.items()):
        first = positions[month_sessions[0]]
        training_end = first - PURGE_SESSION_COUNT
        training_start = training_end - TRAINING_SESSION_COUNT
        if training_start < 0:
            raise RuntimeError("monthly_alpha_refit.insufficient_training_history")
        training = source_sessions[training_start:training_end]
        purge = source_sessions[training_end:first]
        if (
            len(training) != TRAINING_SESSION_COUNT
            or len(purge) != PURGE_SESSION_COUNT
            or max(training) >= min(purge)
            or max(purge) >= min(month_sessions)
        ):
            raise RuntimeError("monthly_alpha_refit.causal_geometry_invalid")
        output.append((month, training, tuple(month_sessions), purge[0]))
    return tuple(output)


def _recipe() -> Any:
    return build_dynamic_panel_lightgbm_recipe(
        DynamicPanelLightGBMParameters(
            seed=1729,
            max_depth=5,
            num_leaves=31,
            min_child_samples=50,
            learning_rate=0.05,
            lambda_l1=10.0,
            lambda_l2=100.0,
            min_gain_to_split=0.01,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=1,
            training_policy="FIXED_ITERATION",
            fixed_iterations=300,
        )
    )


def _training_binding_hash(
    *,
    profile: MonthlyAlphaRefitProfile,
    policy_hash: str,
    vintage_index: int,
    projection: PanelFeatureProjection,
) -> str:
    return str(
        canonical_hash(
            {
                "monthly_refit_policy_hash": policy_hash,
                "vintage_index": vintage_index,
                "phase": "MONTHLY_OUTER_DIRECT_FIT",
                "view_id": profile.view_id,
                "training_row_axis_hash": projection.common_training_row_axis_hash,
                "ordered_feature_ids": list(projection.ordered_feature_ids),
                "training_feature_hash": alpha_model_array_content_hash(
                    projection.training_features
                ),
                "training_target_hash": alpha_model_array_content_hash(projection.training_targets),
            }
        )
    )


def _fit_month_with_estimator(
    *,
    runtime: AlphaModelRuntimeService,
    recipe: Any,
    domain: Any,
    profile: MonthlyAlphaRefitProfile,
    policy_hash: str,
    vintage_index: int,
    month: str,
    projection: PanelFeatureProjection,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    dict[str, Any],
    AlphaEstimatorContent,
]:
    binding_hash = _training_binding_hash(
        profile=profile,
        policy_hash=policy_hash,
        vintage_index=vintage_index,
        projection=projection,
    )
    training_feature_hash = alpha_model_array_content_hash(projection.training_features)
    training_target_hash = alpha_model_array_content_hash(projection.training_targets)
    # The installed LightGBM capability has one nested-fit protocol even for
    # FIXED_ITERATION.  The adapter intentionally ignores the tuning values on
    # that branch and fits all parent rows for exactly 300 iterations.  Bind the
    # already-admitted, strictly pre-validation training lane into both required
    # slots so no outer-month value enters the fit plan.
    fit_identity = {
        "protocol_id": "NESTED_EARLY_STOPPING_REFIT",
        "parent_training_binding_hash": binding_hash,
        "ordered_feature_ids": list(projection.ordered_feature_ids),
        "tuning_training_row_axis_hash": projection.common_training_row_axis_hash,
        "purge_row_axis_hash": str(canonical_hash([])),
        "tuning_validation_row_axis_hash": projection.common_training_row_axis_hash,
        "tuning_training_feature_values_hash": training_feature_hash,
        "tuning_training_target_values_hash": training_target_hash,
        "tuning_validation_feature_values_hash": training_feature_hash,
        "tuning_validation_target_values_hash": training_target_hash,
        "selection_metric_id": "fixed_iteration",
        "maximum_iterations": 500,
        "early_stopping_rounds": 50,
    }
    fit_plan = BoundAlphaModelFitInput(
        fit_plan_hash=str(canonical_hash(fit_identity)),
        protocol_id="NESTED_EARLY_STOPPING_REFIT",
        parent_training_binding_hash=binding_hash,
        ordered_feature_ids=projection.ordered_feature_ids,
        tuning_training_row_axis_hash=projection.common_training_row_axis_hash,
        purge_row_axis_hash=str(canonical_hash([])),
        tuning_validation_row_axis_hash=projection.common_training_row_axis_hash,
        tuning_training_feature_values_hash=training_feature_hash,
        tuning_training_target_values_hash=training_target_hash,
        tuning_validation_feature_values_hash=training_feature_hash,
        tuning_validation_target_values_hash=training_target_hash,
        tuning_training_features=projection.training_features,
        tuning_training_targets=projection.training_targets,
        tuning_validation_features=projection.training_features,
        tuning_validation_targets=projection.training_targets,
        selection_metric_id="fixed_iteration",
        maximum_iterations=500,
        early_stopping_rounds=50,
    )
    result = runtime.execute(
        recipe=recipe,
        domain=domain,
        fit_plan=fit_plan,
        training_input=BoundAlphaTrainingInput(
            training_binding_hash=binding_hash,
            ordered_feature_ids=projection.ordered_feature_ids,
            features=projection.training_features,
            targets=projection.training_targets,
        ),
        prediction_input=BoundAlphaPredictionInput(
            training_binding_hash=binding_hash,
            ordered_feature_ids=projection.ordered_feature_ids,
            features=projection.transformed_features,
        ),
        package_identity_hash=str(
            canonical_hash(
                {
                    "kind": "MONTHLY_ALPHA_REFIT_RESEARCH",
                    "annual_root_hash": ANNUAL_ROOT_HASH,
                    "monthly_refit_policy_hash": policy_hash,
                    "model_recipe_id": profile.model_recipe_id,
                }
            )
        ),
    )
    scores = np.ascontiguousarray(result.prediction.predictions, dtype=np.float64)
    scores.setflags(write=False)
    receipt = {
        "kind": "MonthlyAlphaRefitVintageReceipt",
        "month": month,
        "vintage_index": vintage_index,
        "training_binding_hash": binding_hash,
        "fit_plan_hash": fit_plan.fit_plan_hash,
        "recipe_hash": recipe.recipe_hash,
        "estimator_content_hash": result.fit.estimator_content.content_hash,
        "provenance_hash": result.provenance.provenance_hash,
        "numerical_binding_hash": result.numerical_binding.numerical_binding_hash,
        "numerical_environment_hash": result.numerical_environment.environment_hash,
        "fit_call_count": result.provenance.fit_call_count,
        "predict_call_count": result.provenance.predict_call_count,
        "training_row_count": len(projection.training_targets),
        "validation_row_count": len(scores),
        "score_value_hash": alpha_model_array_content_hash(scores),
        "feature_axis_hash": str(canonical_hash(list(projection.ordered_feature_ids))),
        "scale_receipt_hashes": [value.receipt_hash for value in projection.scale_receipts],
    }
    receipt["receipt_hash"] = str(canonical_hash(receipt))
    return scores, receipt, result.fit.estimator_content


def _fit_month(
    *,
    runtime: AlphaModelRuntimeService,
    recipe: Any,
    domain: Any,
    profile: MonthlyAlphaRefitProfile,
    policy_hash: str,
    vintage_index: int,
    month: str,
    projection: PanelFeatureProjection,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], dict[str, Any]]:
    """Compatibility projection for research lanes that retain predictions only."""

    scores, receipt, _estimator = _fit_month_with_estimator(
        runtime=runtime,
        recipe=recipe,
        domain=domain,
        profile=profile,
        policy_hash=policy_hash,
        vintage_index=vintage_index,
        month=month,
        projection=projection,
    )
    return scores, receipt


def _write_month(
    *,
    output_root: Path,
    month: str,
    projection: PanelFeatureProjection,
    scores: np.ndarray[Any, np.dtype[np.float64]],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    shard = output_root / "score-shards" / f"{month}.parquet"
    shard.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_suffix(".parquet.tmp")
    table = pa.table(
        {
            "formation_session": pa.array(projection.row_sessions, type=pa.date32()),
            "listing_id": pa.array(projection.row_listing_ids, type=pa.string()),
            "score": pa.array(scores, type=pa.float64()),
        }
    )
    pq.write_table(table, temporary, compression="zstd", version="2.6")
    temporary.replace(shard)
    shard_hash = _sha256(shard)
    receipt = {
        **receipt,
        "artifact_relative_path": shard.relative_to(output_root).as_posix(),
        "artifact_sha256": shard_hash,
        "row_axis_hash": str(
            canonical_hash(
                [
                    [session.isoformat(), listing]
                    for session, listing in zip(
                        projection.row_sessions,
                        projection.row_listing_ids,
                        strict=True,
                    )
                ]
            )
        ),
        "first_formation_session": min(projection.row_sessions).isoformat(),
        "last_formation_session": max(projection.row_sessions).isoformat(),
    }
    receipt["receipt_hash"] = str(
        canonical_hash({key: value for key, value in receipt.items() if key != "receipt_hash"})
    )
    receipt_path = output_root / "vintage-receipts" / f"{month}.json"
    _write_json(receipt_path, receipt)
    return receipt


def _load_completed(output_root: Path, month: str) -> dict[str, Any] | None:
    receipt_path = output_root / "vintage-receipts" / f"{month}.json"
    if not receipt_path.is_file():
        return None
    receipt = cast(dict[str, Any], json.loads(receipt_path.read_text(encoding="utf-8")))
    artifact = output_root / receipt["artifact_relative_path"]
    expected = receipt.get("receipt_hash")
    actual = str(
        canonical_hash({key: value for key, value in receipt.items() if key != "receipt_hash"})
    )
    if (
        expected != actual
        or not artifact.is_file()
        or _sha256(artifact) != receipt.get("artifact_sha256")
    ):
        raise RuntimeError(f"monthly_alpha_refit.resume_artifact_invalid:{month}")
    return receipt


def _consolidate(output_root: Path, receipts: list[dict[str, Any]]) -> dict[str, Any]:
    tables = [
        pq.read_table(output_root / value["artifact_relative_path"])
        for value in sorted(receipts, key=lambda item: item["month"])
    ]
    combined = pa.concat_tables(tables)
    surface = output_root / "monthly-alpha-scores.parquet"
    temporary = surface.with_suffix(".parquet.tmp")
    pq.write_table(combined, temporary, compression="zstd", version="2.6")
    temporary.replace(surface)
    sessions = combined.column("formation_session").to_pylist()
    listings = combined.column("listing_id").to_pylist()
    scores = combined.column("score").to_numpy(zero_copy_only=False)
    identity = {
        "artifact_sha256": _sha256(surface),
        "row_count": combined.num_rows,
        "formation_session_count": len(set(sessions)),
        "first_formation_session": min(sessions).isoformat(),
        "last_formation_session": max(sessions).isoformat(),
        "row_axis_hash": str(
            canonical_hash(
                [
                    [session.isoformat(), listing]
                    for session, listing in zip(sessions, listings, strict=True)
                ]
            )
        ),
        "score_value_hash": alpha_model_array_content_hash(np.asarray(scores, dtype=np.float64)),
    }
    return identity


def _consolidate_existing(output_root: Path, profile: MonthlyAlphaRefitProfile) -> dict[str, Any]:
    source_path = output_root / "source-receipt.json"
    if not source_path.is_file():
        source_path = output_root / "source-receipt-part-0.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_profile_id = source.get("feature_profile_id")
    if source_profile_id is None and profile.profile_id == "sparse-session-amplitude-195":
        source_profile_id = profile.profile_id
    if (
        source_profile_id != profile.profile_id
        or source.get("feature_view_id") != profile.view_id
        or int(source.get("feature_count", -1)) != profile.expected_feature_count
    ):
        raise RuntimeError("monthly_alpha_refit.profile_identity_mismatch")
    receipt_paths = sorted((output_root / "vintage-receipts").glob("*.json"))
    receipts = []
    for path in receipt_paths:
        receipt = _load_completed(output_root, path.stem)
        assert receipt is not None
        receipts.append(receipt)
    if len(receipts) != int(source["monthly_vintage_count"]) or sum(
        int(value["validation_session_count"]) for value in receipts
    ) != int(source["validation_session_count"]):
        raise RuntimeError("monthly_alpha_refit.surface_vintage_incomplete")
    surface = _consolidate(output_root, receipts)
    profile_identity = {
        key: source[key]
        for key in (
            "feature_profile_id",
            "feature_profile_disposition",
            "historical_recipe_sha256",
        )
        if key in source
    }
    manifest = {
        "kind": "MonthlyAlphaRefitResearchSurface",
        "identity_class": "DEVELOPMENT_RESEARCH_ONLY",
        "membership_disposition": "CURRENT_MEMBERSHIP_BACKFILLED",
        "policy_hash": source["policy_hash"],
        "annual_frozen_alpha_root_hash": ANNUAL_ROOT_HASH,
        "model_recipe_id": source["model_recipe_id"],
        "model_recipe_hash": source["model_recipe_hash"],
        "feature_view_id": source["feature_view_id"],
        **profile_identity,
        "feature_count": source["feature_count"],
        "feature_axis_hash": source["feature_axis_hash"],
        "vintage_count": len(receipts),
        "vintage_receipt_hashes": [value["receipt_hash"] for value in receipts],
        "fit_call_count": sum(int(value["fit_call_count"]) for value in receipts),
        "predict_call_count": sum(int(value["predict_call_count"]) for value in receipts),
        "metric_call_count": 0,
        "solver_call_count": 0,
        "search_call_count": 0,
        "artifact_relative_path": "monthly-alpha-scores.parquet",
        **surface,
    }
    manifest["surface_hash"] = str(canonical_hash(manifest))
    _write_json(output_root / "monthly-alpha-score-surface.json", manifest)
    return manifest


def main() -> int:
    arguments = _arguments()
    profile = PROFILES[arguments.profile]
    if (
        arguments.partition_count < 1
        or arguments.partition_index < 0
        or arguments.partition_index >= arguments.partition_count
        or (arguments.month is not None and arguments.partition_count != 1)
    ):
        raise RuntimeError("monthly_alpha_refit.partition_invalid")
    os.environ.setdefault("ALPHALATTICE_NETWORK_DISABLED", "1")
    output_root = (arguments.output_root or profile.default_output_root).resolve()
    if arguments.consolidate_only:
        manifest = _consolidate_existing(output_root, profile)
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return 0
    annual_path, market, roots = _historical_sources(arguments.research_root.resolve())
    annual = _load_annual_root(annual_path)
    print("resolving exact frozen Panel/Alpha source", flush=True)
    authority, resolved, source_wall = _resolve_source(market=market, roots=roots)
    print(
        f"source resolved in {source_wall:.3f}s: "
        f"{len(resolved.arrays.formation_sessions)} sessions x "
        f"{len(resolved.arrays.ordered_listing_ids)} listings",
        flush=True,
    )
    plan_started = perf_counter()
    plan = preflight_panel_feature_plan(
        source=resolved.arrays,
        selected_method_ids=(profile.view_id,),
        maximum_aggregation_span=1,
    )
    method = plan.catalog.resolve(profile.view_id)
    feature_ids = tuple(
        feature_id
        for role in method.roles
        for feature_id in plan.role_descriptors_by_selection_hash[
            role.selection_hash
        ].ordered_feature_ids
    )
    if len(feature_ids) != profile.expected_feature_count:
        raise RuntimeError(f"monthly_alpha_refit.feature_count_invalid:{len(feature_ids)}")
    validation_sessions = tuple(
        date.fromisoformat(value) for value in annual["common_formation_sessions"]
    )
    geometry = _monthly_geometry(
        source_sessions=resolved.arrays.formation_sessions,
        validation_sessions=validation_sessions,
    )
    if arguments.month is not None:
        geometry = tuple(value for value in geometry if value[0] == arguments.month)
        if len(geometry) != 1:
            raise RuntimeError("monthly_alpha_refit.requested_month_not_admitted")
    elif arguments.partition_count > 1:
        geometry = tuple(
            value
            for index, value in enumerate(geometry)
            if index % arguments.partition_count == arguments.partition_index
        )
    recipe = _recipe()
    policy_identity: dict[str, Any] = {
        "kind": "MonthlyAlphaRefitResearchPolicy",
        "disposition": "DEVELOPMENT_RESEARCH_ONLY",
        "membership_disposition": "CURRENT_MEMBERSHIP_BACKFILLED",
        "annual_root_hash": ANNUAL_ROOT_HASH,
        "annual_program_hash": annual["program_hash"],
        "authority_hash": authority.authority_hash,
        "source_resolution_hash": resolved.resolution.resolution_hash,
        "panel_snapshot_hash": authority.panel_snapshot_hash,
        "feature_view_id": profile.view_id,
        "feature_count": len(feature_ids),
        "feature_axis_hash": str(canonical_hash(list(feature_ids))),
        "model_recipe_id": profile.model_recipe_id,
        "model_recipe_hash": recipe.recipe_hash,
        "training_session_count": TRAINING_SESSION_COUNT,
        "purge_session_count": PURGE_SESSION_COUNT,
        "refit_clock": "FIRST_ADMITTED_FORMATION_OF_EACH_CALENDAR_MONTH",
        "validation_clock": "ALL_ANNUAL_FROZEN_OOS_FORMATIONS_IN_CALENDAR_MONTH",
        "formation_to_outcome_clock": "CLOSE_T_TO_NEXT_OPEN_T_PLUS_1_TO_FOLLOWING_OPEN",
        "search_call_count": 0,
    }
    policy_identity.update(_profile_identity_fields(profile))
    policy_hash = str(canonical_hash(policy_identity))
    source_receipt = {
        **policy_identity,
        "policy_hash": policy_hash,
        "source_session_count": len(resolved.arrays.formation_sessions),
        "source_listing_count": len(resolved.arrays.ordered_listing_ids),
        "validation_session_count": len(validation_sessions),
        "monthly_vintage_count": len(
            _monthly_geometry(
                source_sessions=resolved.arrays.formation_sessions,
                validation_sessions=validation_sessions,
            )
        ),
        "source_resolution_wall_seconds": source_wall,
        "feature_preflight_wall_seconds": perf_counter() - plan_started,
    }
    source_receipt_name = (
        "source-receipt.json"
        if arguments.partition_count == 1
        else f"source-receipt-part-{arguments.partition_index}.json"
    )
    _write_json(output_root / source_receipt_name, source_receipt)
    print(
        f"geometry admitted: {source_receipt['monthly_vintage_count']} vintages, "
        f"{source_receipt['validation_session_count']} validation sessions, "
        f"{len(feature_ids)} features",
        flush=True,
    )
    if arguments.prepare_only:
        return 0

    runtime = AlphaModelRuntimeService(build_panel_model_catalog())
    domain = build_dynamic_panel_lightgbm_search_domain()
    receipts: list[dict[str, Any]] = []
    all_geometry = _monthly_geometry(
        source_sessions=resolved.arrays.formation_sessions,
        validation_sessions=validation_sessions,
    )
    vintage_index_by_month = {value[0]: index for index, value in enumerate(all_geometry)}
    for month, training, validation, purge_session in geometry:
        completed = _load_completed(output_root, month)
        if completed is not None:
            receipts.append(completed)
            print(f"{month}: reused exact shard", flush=True)
            continue
        started = perf_counter()
        projection = materialize_panel_feature_projection(
            plan=plan,
            method_id=profile.view_id,
            program_hash=policy_hash,
            fold_index=vintage_index_by_month[month],
            boundary_id="OUTER",
            training_sessions=training,
            transform_sessions=validation,
        )
        scores, receipt = _fit_month(
            runtime=runtime,
            recipe=recipe,
            domain=domain,
            profile=profile,
            policy_hash=policy_hash,
            vintage_index=vintage_index_by_month[month],
            month=month,
            projection=projection,
        )
        receipt.update(
            {
                "training_session_count": len(training),
                "training_first_session": training[0].isoformat(),
                "training_last_session": training[-1].isoformat(),
                "purge_session": purge_session.isoformat(),
                "validation_session_count": len(validation),
                "validation_first_session": validation[0].isoformat(),
                "validation_last_session": validation[-1].isoformat(),
                "wall_seconds": perf_counter() - started,
            }
        )
        written = _write_month(
            output_root=output_root,
            month=month,
            projection=projection,
            scores=scores,
            receipt=receipt,
        )
        receipts.append(written)
        print(
            f"{month}: fit+predict {written['validation_row_count']} rows in "
            f"{written['wall_seconds']:.3f}s",
            flush=True,
        )
        del projection, scores
        gc.collect()

    if arguments.month is None and arguments.partition_count == 1:
        if len(receipts) != len(all_geometry):
            raise RuntimeError("monthly_alpha_refit.surface_vintage_incomplete")
        surface = _consolidate(output_root, receipts)
        manifest = {
            "kind": "MonthlyAlphaRefitResearchSurface",
            "identity_class": "DEVELOPMENT_RESEARCH_ONLY",
            "membership_disposition": "CURRENT_MEMBERSHIP_BACKFILLED",
            "policy_hash": policy_hash,
            "annual_frozen_alpha_root_hash": ANNUAL_ROOT_HASH,
            "model_recipe_id": profile.model_recipe_id,
            "model_recipe_hash": recipe.recipe_hash,
            "feature_view_id": profile.view_id,
            **_profile_identity_fields(profile),
            "feature_count": len(feature_ids),
            "feature_axis_hash": str(canonical_hash(list(feature_ids))),
            "vintage_count": len(receipts),
            "vintage_receipt_hashes": [
                value["receipt_hash"] for value in sorted(receipts, key=lambda item: item["month"])
            ],
            "fit_call_count": sum(int(value["fit_call_count"]) for value in receipts),
            "predict_call_count": sum(int(value["predict_call_count"]) for value in receipts),
            "metric_call_count": 0,
            "solver_call_count": 0,
            "search_call_count": 0,
            "artifact_relative_path": "monthly-alpha-scores.parquet",
            **surface,
        }
        manifest["surface_hash"] = str(canonical_hash(manifest))
        _write_json(output_root / "monthly-alpha-score-surface.json", manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    elif arguments.partition_count > 1:
        print(
            f"partition {arguments.partition_index}/{arguments.partition_count} complete: "
            f"{len(receipts)} exact shards",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
