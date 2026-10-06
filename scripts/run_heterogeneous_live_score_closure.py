"""Reopen all 48 successor models: score one current formation, or seal a local closure.

Two modes over one set of loaders. The Gate M mode scores the current
formation and writes the live-score closure receipt. The materialisation mode,
authorised for development only, derives the minimum product-readable closure
for the last few formations -- the forty-eight payloads, each component's
per-vintage Feature surfaces and the raw 12-1 momentum lane -- and seals it
under one content-addressed manifest that the product's closure reader verifies
before anything is installed.

Every source either mode reads must be a file the Gate M source readback names,
with the digest it recorded. Nothing in either source worktree is written.

The research inputs are identity-bound data. The monthly refit driver, the three
workspace modules of the monthly research worktree and the Gate D2 runner are
named and hashed exactly as before, and verified byte for byte against the
digests the local port in ``broad_ensemble_research_closure`` was taken from; none of
them is executed. The code that runs is this tree's: the ported closure, this
tree's copy of the monthly refit driver, and the installed product owners.

``--preflight`` names and hashes every source, verifies the port provenance and
the executable owners, prints that record, and stops before any array is read.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

# Only when run as a script, before the NumPy-bearing imports below: a test importing this
# module keeps its own threads and network (W11).
if __name__ == "__main__":
    for _name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_name] = "1"
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"
    sys.dont_write_bytecode = True

SCRIPTS_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = SCRIPTS_ROOT.parent
for _entry in (str(PLAYPEN_ROOT / "src"), str(SCRIPTS_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import numpy as np  # noqa: E402

import broad_ensemble_research_closure as closure  # noqa: E402
from alphalattice.capabilities.alpha_modeling.contracts import AlphaEstimatorContent  # noqa: E402
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (  # noqa: E402
    materialize_panel_feature_projection,
    preflight_panel_feature_plan,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (  # noqa: E402
    COMPONENT_IDS,
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    LIVE_MODEL_CLOSURE_MANIFEST_HASH,
    LIVE_MODEL_VINTAGES,
    LIVE_SCORE_CLOSURE_RECEIPT_HASH,
    LIVE_SOURCE_READBACK_SHA256,
    ComponentId,
    HeterogeneousLiveScoreClosureReceipt,
    array_value_hash,
    live_model_set_hash,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (  # noqa: E402
    CLOSURE_MANIFEST_FILE_NAME,
    AlphaRuntimeHeterogeneousPredictionOwner,
    HeterogeneousClosureChild,
    HeterogeneousClosureComponent,
    HeterogeneousClosureFile,
    HeterogeneousClosureMomentum,
    HeterogeneousClosureSurface,
    HeterogeneousCurrentClosureManifest,
    admit_heterogeneous_current_closure,
)
from alphalattice.investment.alpha_research.scores.product_replay import (  # noqa: E402
    HeterogeneousFormationScoreInput,
    HeterogeneousLiveModel,
    HeterogeneousVintageFeatureSurface,
    score_heterogeneous_component,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (  # noqa: E402
    CappedSleeveComponent,
    TrancheFormationInputs,
    open_component_book,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (  # noqa: E402
    INSTALLED_HETEROGENEOUS_BOOK_RECIPE,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash  # noqa: E402

VINTAGES = ("2025-10", "2026-01", "2026-04", "2026-07")
SEEDS = (1729, 2718, 31415)
FORMATION = date(2026, 7, 29)
MOMENTUM_FACTOR_ID = "mom_252_21"

type FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("alpha_research.gate_m_json_object_required")
    return cast(dict[str, Any], value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _snapshot(paths: set[Path]) -> dict[str, dict[str, Any]]:
    missing = tuple(str(path) for path in sorted(paths) if not path.is_file())
    if missing:
        raise ValueError("alpha_research.gate_m_source_absent:" + "|".join(missing))
    return {
        str(path.resolve()): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
    }


def _immutable(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)


def _specialist_roots(args: argparse.Namespace, component: ComponentId) -> tuple[Path, Path, Path]:
    root = args.heterogeneous_research_worktree / "workspaces/heterogeneous-alpha-strategy-research"
    if component == "G2_R0_TREND":
        gate = root / "gate-b-g2"
        return (
            gate / "cache/deployment",
            gate / "predictions/deployment",
            gate / "boosters/deployment",
        )
    if component == "G6_R0_FAST_REBOUND":
        gate = root / "gate-c3-g4-g6/g6-fast-rebound"
        return (
            gate / "cache/deployment/R0_CORE",
            gate / "predictions/deployment/R0_CORE",
            gate / "boosters/R0_CORE",
        )
    if component == "G7_R1_CONTEXTUAL_MOMENTUM":
        gate = root / "gate-c1-g7-contextual-momentum"
        return (
            gate / "cache/deployment/R1_CONTEXT_CONFIRMED",
            gate / "predictions/deployment/R1_CONTEXT_CONFIRMED",
            gate / "boosters/deployment/R1_CONTEXT_CONFIRMED",
        )
    raise ValueError("alpha_research.gate_m_specialist_component_invalid")


def _specialist_paths(
    args: argparse.Namespace, component: ComponentId, vintage: str, seed: int
) -> tuple[Path, Path, Path, Path]:
    matrix_root, prediction_root, booster_root = _specialist_roots(args, component)
    matrix = matrix_root / vintage
    if component == "G2_R0_TREND":
        prediction = prediction_root / vintage / f"seed-{seed}"
        booster = booster_root / f"{vintage}-seed-{seed}.json"
    elif component == "G6_R0_FAST_REBOUND":
        prediction = prediction_root / f"{vintage}-seed-{seed}"
        booster = booster_root / f"deployment-{vintage}-seed-{seed}.txt"
    else:
        prediction = prediction_root / vintage / f"seed-{seed}"
        booster = booster_root / f"{vintage}-seed-{seed}.json"
    return matrix, prediction.with_suffix(".json"), prediction.with_suffix(".npz"), booster


def _source_paths(args: argparse.Namespace) -> set[Path]:
    monthly = args.monthly_research_worktree / "workspaces/monthly-alpha-refit-research"
    runner = args.monthly_research_worktree / "scripts/run_monthly_alpha_refit_research.py"
    paths = {
        runner,
        monthly / "run_iw184_quarterly_5y_ensemble.py",
        monthly / "feature-recipe-exploration/run_feature_recipe_exploration.py",
        monthly / "frozen-iw184-holdout-smoke/run_frozen_iw184_holdout_smoke.py",
        args.gate_d2_runner,
        args.g0_model_root / "live-model-closure-manifest.json",
    }
    source_root = monthly / "frozen-iw184-holdout-smoke/cache/current-source"
    source_manifest = _read_json(source_root / "manifest.json")
    paths.add(source_root / "manifest.json")
    paths.update(source_root / value["path"] for value in source_manifest["arrays"].values())
    for component in COMPONENT_IDS:
        paths.update(
            {
                args.gate_d_score_root / f"{component}.json",
                args.gate_d_score_root / f"{component}.npy",
            }
        )
    for vintage in reversed(VINTAGES):
        paths.update({args.g0_axis_root / f"{vintage}.json", args.g0_axis_root / f"{vintage}.npz"})
        for seed in SEEDS:
            paths.update(
                {
                    args.g0_model_root / vintage / f"seed-{seed}.json",
                    args.g0_prediction_root / vintage / f"seed-{seed}.json",
                    args.g0_prediction_root / vintage / f"seed-{seed}.npz",
                }
            )
            for component in COMPONENT_IDS[1:]:
                matrix, receipt, prediction, booster = _specialist_paths(
                    args, component, vintage, seed
                )
                manifest = _read_json(matrix / "manifest.json")
                paths.add(matrix / "manifest.json")
                paths.update(
                    matrix / manifest["artifacts"][key]["relative_path"]
                    for key in (
                        "validation-features",
                        "validation-row-listings",
                        "validation-row-sessions",
                    )
                )
                paths.update({receipt, prediction, booster})
    return paths


def _map_rows(
    *,
    row_sessions: np.ndarray[Any, Any],
    row_listings: np.ndarray[Any, Any],
    row_values: np.ndarray[Any, Any],
    listings: tuple[str, ...],
    formation: date = FORMATION,
) -> np.ndarray[Any, Any]:
    mask = np.asarray(row_sessions, dtype="datetime64[D]") == np.datetime64(formation)
    if not mask.any():
        raise ValueError("alpha_research.gate_m_formation_absent")
    selected_listings = np.asarray(row_listings)[mask].astype(str)
    if len(set(selected_listings)) != len(selected_listings):
        raise ValueError("alpha_research.gate_m_formation_listing_duplicate")
    lookup = {value: index for index, value in enumerate(listings)}
    try:
        positions: np.ndarray[Any, Any] = np.asarray(
            [lookup[value] for value in selected_listings], dtype=np.intp
        )
    except KeyError as error:
        raise ValueError("alpha_research.gate_m_listing_axis_mismatch") from error
    shape = (len(listings), *row_values.shape[1:])
    output: np.ndarray[Any, Any] = np.full(shape, np.nan, dtype=np.float64)
    output[positions] = np.asarray(row_values[mask], dtype=np.float64)
    return output


def _stack_rows(
    *,
    row_sessions: np.ndarray[Any, Any],
    row_listings: np.ndarray[Any, Any],
    row_values: np.ndarray[Any, Any],
    listings: tuple[str, ...],
    formations: tuple[date, ...],
) -> np.ndarray[Any, Any]:
    """One `_map_rows` per formation, stacked as `(formations, listings, ...)`."""

    stacked: np.ndarray[Any, Any] = np.stack(
        [
            _map_rows(
                row_sessions=row_sessions,
                row_listings=row_listings,
                row_values=row_values,
                listings=listings,
                formation=formation,
            )
            for formation in formations
        ],
        axis=0,
    )
    return stacked


@dataclass(frozen=True, slots=True)
class _Bundle:
    """One component's children and lanes over the requested formations."""

    component_id: ComponentId
    models: tuple[HeterogeneousLiveModel, ...]
    surfaces: dict[str, tuple[np.ndarray[Any, Any], str]]
    """Per vintage, `(features (formations, listings, feature_count), source_binding_hash)`."""

    expected: dict[tuple[str, int], np.ndarray[Any, Any]]
    """Per `(vintage, seed)`, the research predictions on `(formations, listings)`."""

    model_set_manifest_hash: str
    payload_paths: dict[tuple[str, int], Path]


def _specialist_bundle(
    args: argparse.Namespace,
    *,
    component_id: ComponentId,
    listings: tuple[str, ...],
    formations: tuple[date, ...],
) -> _Bundle:
    component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id)
    surfaces: dict[str, tuple[np.ndarray[Any, Any], str]] = {}
    models = []
    expected: dict[tuple[str, int], np.ndarray[Any, Any]] = {}
    payloads: dict[tuple[str, int], Path] = {}
    for vintage in reversed(VINTAGES):
        matrix_root, _, _, _ = _specialist_paths(args, component_id, vintage, SEEDS[0])
        manifest = _read_json(matrix_root / "manifest.json")
        identity_hash = str(manifest["identity_hash"])
        if identity_hash != canonical_hash(manifest["identity"]):
            raise ValueError("alpha_research.gate_m_matrix_identity_invalid")
        sessions = np.load(
            matrix_root / "validation-row-sessions.npy", mmap_mode="r", allow_pickle=False
        )
        row_listings = np.load(
            matrix_root / "validation-row-listings.npy", mmap_mode="r", allow_pickle=False
        )
        features = np.load(
            matrix_root / "validation-features.npy", mmap_mode="r", allow_pickle=False
        )
        surfaces[vintage] = (
            _stack_rows(
                row_sessions=sessions,
                row_listings=row_listings,
                row_values=features,
                listings=listings,
                formations=formations,
            ),
            identity_hash,
        )
        for seed in SEEDS:
            _, receipt_path, prediction_path, booster_path = _specialist_paths(
                args, component_id, vintage, seed
            )
            receipt = _read_json(receipt_path)
            estimator = AlphaEstimatorContent.model_validate_json(
                booster_path.read_text(encoding="utf-8")
            )
            fit = receipt["fit_receipt"]
            if estimator.content_hash != fit["estimator_content_hash"]:
                raise ValueError("alpha_research.gate_m_model_content_mismatch")
            models.append(
                HeterogeneousLiveModel(
                    vintage=vintage,
                    seed=seed,
                    recipe_hash=str(fit["recipe_hash"]),
                    training_binding_hash=str(fit["training_binding_hash"]),
                    lineage_hash=str(receipt["identity_hash"]),
                    estimator=estimator,
                )
            )
            payloads[(vintage, seed)] = booster_path
            with np.load(prediction_path, allow_pickle=False) as payload:
                raw = np.asarray(payload["scores"], dtype=np.float64)
            expected[(vintage, seed)] = _stack_rows(
                row_sessions=sessions,
                row_listings=row_listings,
                row_values=raw,
                listings=listings,
                formations=formations,
            )
    return _Bundle(
        component_id=component_id,
        models=tuple(models),
        surfaces=surfaces,
        expected=expected,
        model_set_manifest_hash=live_model_set_hash(
            recipe_hash=component.recipe_hash,
            content_hashes=tuple(value.estimator.content_hash for value in models),
        ),
        payload_paths=payloads,
    )


def _g0_bundle(
    args: argparse.Namespace,
    *,
    listings: tuple[str, ...],
    formations: tuple[date, ...],
) -> tuple[_Bundle, np.ndarray[Any, Any], str]:
    """The G0 children, their Feature surfaces, and the momentum lane, plus the source identity."""

    runner = closure.monthly_runner
    source_root = args.monthly_research_worktree / closure.SOURCE_CACHE_RELATIVE_PATH
    source_manifest = _read_json(source_root / "manifest.json")
    source = closure.load_cached_panel_source(source_root)
    if tuple(source.ordered_listing_ids) != listings:
        raise ValueError("alpha_research.gate_m_g0_listing_axis_mismatch")
    resolution_hash = str(source_manifest["cache_identity_hash"])
    resolved = SimpleNamespace(
        arrays=source,
        resolution=SimpleNamespace(resolution_hash=resolution_hash),
    )
    profile = runner.PROFILES["sparse-session-amplitude-195"]
    candidate_axis = closure.candidate_axis(args.monthly_research_worktree)
    plan = preflight_panel_feature_plan(
        source=source,
        selected_method_ids=(profile.view_id,),
        maximum_aggregation_span=1,
    )
    validation_sessions = tuple(
        session for session in source.formation_sessions if session.isoformat() >= "2019-08-05"
    )
    specs = closure.quarter_specs(
        source_sessions=source.formation_sessions,
        validation_sessions=validation_sessions,
        first_vintage=VINTAGES[0],
        last_vintage=VINTAGES[-1],
    )
    if tuple(value["month"] for value in specs) != VINTAGES:
        raise ValueError("alpha_research.gate_m_g0_vintage_axis_invalid")
    manifest = _read_json(args.g0_model_root / "live-model-closure-manifest.json")
    manifest_rows = {
        (str(value["vintage"]), int(value["seed"])): value for value in manifest["models"]
    }
    surfaces: dict[str, tuple[np.ndarray[Any, Any], str]] = {}
    models = []
    expected: dict[tuple[str, int], np.ndarray[Any, Any]] = {}
    payloads: dict[tuple[str, int], Path] = {}
    for spec in reversed(specs):
        projection = materialize_panel_feature_projection(
            plan=plan,
            method_id=profile.view_id,
            program_hash=canonical_hash(
                {
                    "kind": "IW184_FIXED1260_QUARTERLY_SEED3_VINTAGE4_LINEAR",
                    "quarter_start": spec["month"],
                    "source_resolution_hash": resolved.resolution.resolution_hash,
                    "candidate_axis_hash": canonical_hash(list(candidate_axis)),
                }
            ),
            fold_index=int(spec["ordinal"]),
            boundary_id="OUTER",
            training_sessions=spec["training"],
            transform_sessions=spec["transform"],
        )
        lookup = {value: index for index, value in enumerate(projection.ordered_feature_ids)}
        columns: np.ndarray[Any, Any] = np.asarray(
            [lookup[value] for value in candidate_axis], dtype=np.intp
        )
        transformed = np.asarray(projection.transformed_features[:, columns], dtype=np.float64)
        mapped = _stack_rows(
            row_sessions=np.asarray(projection.row_sessions, dtype="datetime64[D]"),
            row_listings=np.asarray(projection.row_listing_ids, dtype=str),
            row_values=transformed,
            listings=listings,
            formations=formations,
        )
        source_binding = canonical_hash(
            {
                "source_resolution_hash": resolved.resolution.resolution_hash,
                "training_row_axis_hash": projection.common_training_row_axis_hash,
                "transform_row_axis_hash": projection.common_transform_row_axis_hash,
                "scale_receipt_hashes": [value.receipt_hash for value in projection.scale_receipts],
            }
        )
        surfaces[str(spec["month"])] = (mapped, str(source_binding))
        training = np.ascontiguousarray(projection.training_features[:, columns], dtype=np.float64)
        targets = np.ascontiguousarray(projection.training_targets, dtype=np.float64)
        training.setflags(write=False)
        targets.setflags(write=False)
        for seed in SEEDS:
            row = manifest_rows[(str(spec["month"]), seed)]
            payload_path = args.g0_model_root / row["model_payload_relative_path"]
            estimator = AlphaEstimatorContent.model_validate_json(
                payload_path.read_text(encoding="utf-8")
            )
            identity = closure.seed_model_identity(
                axis_hash=canonical_hash(list(candidate_axis)),
                recipe_hash=str(row["recipe_hash"]),
                seed=seed,
                spec=spec,
                source_resolution_hash=resolved.resolution.resolution_hash,
            )
            candidate_projection = SimpleNamespace(
                ordered_feature_ids=candidate_axis,
                training_features=training,
                training_targets=targets,
                common_training_row_axis_hash=projection.common_training_row_axis_hash,
            )
            training_binding = runner._training_binding_hash(
                profile=profile,
                policy_hash=identity["identity_hash"],
                vintage_index=int(spec["ordinal"]),
                projection=candidate_projection,
            )
            models.append(
                HeterogeneousLiveModel(
                    vintage=str(spec["month"]),
                    seed=seed,
                    recipe_hash=str(row["recipe_hash"]),
                    training_binding_hash=training_binding,
                    lineage_hash=str(row["identity_hash"]),
                    estimator=estimator,
                )
            )
            payloads[(str(spec["month"]), seed)] = payload_path
            with np.load(
                args.g0_axis_root / f"{spec['month']}.npz", allow_pickle=False
            ) as axis_payload:
                row_sessions = np.asarray(axis_payload["row_sessions"])
                row_listings = np.asarray(axis_payload["row_listing_ids"])
            with np.load(
                args.g0_prediction_root / spec["month"] / f"seed-{seed}.npz",
                allow_pickle=False,
            ) as prediction_payload:
                raw = np.asarray(prediction_payload["scores"], dtype=np.float64)
            expected[(str(spec["month"]), seed)] = _stack_rows(
                row_sessions=row_sessions,
                row_listings=row_listings,
                row_values=raw,
                listings=listings,
                formations=formations,
            )
        del projection, transformed, mapped, training, targets
        gc.collect()
    session_lookup = {value: index for index, value in enumerate(source.formation_sessions)}
    factor_index = tuple(source.ordered_factor_ids).index(MOMENTUM_FACTOR_ID)
    momentum = np.stack(
        [
            np.asarray(
                source.raw_formula_values[session_lookup[formation], :, factor_index],
                dtype=np.float64,
            )
            for formation in formations
        ],
        axis=0,
    )
    bundle = _Bundle(
        component_id="G0_IW184",
        models=tuple(models),
        surfaces=surfaces,
        expected=expected,
        model_set_manifest_hash=str(manifest["manifest_hash"]),
        payload_paths=payloads,
    )
    return bundle, momentum, resolution_hash


def _inputs_at(
    bundle: _Bundle,
    *,
    index: int,
    formation: date,
    listings: tuple[str, ...],
    eligible: np.ndarray[Any, Any],
    momentum: np.ndarray[Any, Any] | None,
) -> HeterogeneousFormationScoreInput:
    component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(bundle.component_id)
    return HeterogeneousFormationScoreInput.create(
        formation_session=formation,
        ordered_listing_ids=listings,
        decision_eligible=eligible,
        raw_12_1_momentum=momentum,
        feature_surfaces=tuple(
            HeterogeneousVintageFeatureSurface.create(
                vintage=vintage,
                ordered_listing_ids=listings,
                ordered_feature_ids=(
                    component.ordered_feature_ids or bundle.models[0].estimator.ordered_feature_ids
                ),
                features=bundle.surfaces[vintage][0][index],
                source_binding_hash=bundle.surfaces[vintage][1],
            )
            for vintage in reversed(VINTAGES)
        ),
        models=bundle.models,
        model_set_manifest_hash=bundle.model_set_manifest_hash,
    )


def _raw_prediction_gap(
    bundle: _Bundle,
    *,
    formation_count: int,
    owner: AlphaRuntimeHeterogeneousPredictionOwner,
) -> float:
    """Reopened predictions against the research predictions, on every formation."""

    component = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(bundle.component_id)
    maximum = 0.0
    for model in bundle.models:
        surface = bundle.surfaces[model.vintage][0]
        expected = bundle.expected[(model.vintage, model.seed)]
        for index in range(formation_count):
            feature = surface[index]
            positions = np.flatnonzero(np.isfinite(feature).all(axis=1))
            actual = owner(
                component=component,
                model=model,
                features=np.asarray(_immutable(feature[positions]), dtype=np.float64),
            )
            expected_positions = np.flatnonzero(np.isfinite(expected[index]))
            if not np.array_equal(positions, expected_positions):
                raise ValueError("alpha_research.gate_m_raw_prediction_axis_mismatch")
            maximum = max(maximum, float(np.max(np.abs(actual - expected[index][positions]))))
    return maximum


def _admitted_axes(
    args: argparse.Namespace, *, formation_count: int
) -> tuple[closure.CandidateAxes, tuple[date, ...]]:
    """The Gate D2 book axes, admitted by the ported loader over the declared roots."""

    roots = closure.candidate_axis_roots(
        heterogeneous_worktree=args.heterogeneous_research_worktree,
        monthly_worktree=args.monthly_research_worktree,
        risk_lab_root=args.risk_lab_root,
    )
    closure.validate_candidate_axis_batch_sources(roots)
    axes, _ = closure.load_candidate_axes(roots)
    if axes.sessions[-1] != FORMATION.isoformat():
        raise ValueError("alpha_research.gate_m_current_formation_invalid")
    formations = tuple(date.fromisoformat(value) for value in axes.sessions[-formation_count:])
    return axes, formations


def _require_named_sources(
    args: argparse.Namespace, sources: set[Path]
) -> dict[str, dict[str, Any]]:
    """Every source read must be one the Gate M readback named, with the digest it recorded."""

    readback_path = args.gate_m_root / "source-readback.json"
    if _sha256(readback_path) != LIVE_SOURCE_READBACK_SHA256:
        raise ValueError("alpha_research.gate_m_source_readback_identity_invalid")
    named = _read_json(readback_path)["sources"]
    before = _snapshot(sources)
    unnamed = sorted(
        key for key, value in before.items() if named.get(key, {}).get("sha256") != value["sha256"]
    )
    if unnamed:
        raise ValueError("alpha_research.gate_m_source_not_named_by_readback:" + "|".join(unnamed))
    receipt = _read_json(args.gate_m_root / "live-score-closure-receipt.json")
    if receipt.get("receipt_hash") != LIVE_SCORE_CLOSURE_RECEIPT_HASH:
        raise ValueError("alpha_research.gate_m_receipt_identity_invalid")
    gate_l = _read_json(args.g0_model_root / "live-model-closure-manifest.json")
    if gate_l.get("manifest_hash") != LIVE_MODEL_CLOSURE_MANIFEST_HASH:
        raise ValueError("alpha_research.gate_l_manifest_identity_invalid")
    return before


def _provenance(args: argparse.Namespace) -> dict[str, str]:
    return closure.verify_ported_research_sources(
        monthly_worktree=args.monthly_research_worktree,
        gate_d2_runner=args.gate_d2_runner,
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    sources = _source_paths(args)
    readback = _snapshot(sources)
    provenance = _provenance(args)
    return {
        "kind": "HeterogeneousLiveScoreClosurePreflight",
        "disposition": "PREFLIGHT_ONLY",
        "formation_session": FORMATION.isoformat(),
        "source_readback": readback,
        "ported_source_provenance": provenance,
        "executable_owner_readback": closure.executable_owner_readback(),
        "fit_count": 0,
        "prediction_call_count": 0,
        "pointer_mutation_count": 0,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    sources = _source_paths(args)
    before = _snapshot(sources)
    provenance = _provenance(args)
    axes, formations = _admitted_axes(args, formation_count=1)
    listings = tuple(str(value) for value in axes.listings)
    eligible = np.asarray(axes.eligible[-1], dtype=np.bool_)
    g0, momentum, _ = _g0_bundle(args, listings=listings, formations=formations)
    bundles = {"G0_IW184": g0}
    for component in COMPONENT_IDS[1:]:
        bundles[component] = _specialist_bundle(
            args, component_id=component, listings=listings, formations=formations
        )

    projections = {}
    raw_gaps = {}
    aggregate_gaps = {}
    owner = AlphaRuntimeHeterogeneousPredictionOwner()
    for component in COMPONENT_IDS:
        raw_gaps[component] = _raw_prediction_gap(
            bundles[component], formation_count=1, owner=owner
        )
        projection = score_heterogeneous_component(
            component=INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component),
            inputs=_inputs_at(
                bundles[component],
                index=0,
                formation=FORMATION,
                listings=listings,
                eligible=eligible,
                momentum=momentum[0],
            ),
            prediction_owner=owner,
        )
        expected_surface = np.load(
            args.gate_d_score_root / f"{component}.npy", mmap_mode="r", allow_pickle=False
        )[-1]
        if not np.array_equal(np.isfinite(projection.scores), np.isfinite(expected_surface)):
            raise ValueError("alpha_research.gate_m_aggregate_score_axis_mismatch")
        live = np.flatnonzero(np.isfinite(expected_surface))
        aggregate_gaps[component] = float(
            np.max(np.abs(projection.scores[live] - expected_surface[live]))
        )
        projections[component] = projection

    formation_inputs = {
        component: (
            TrancheFormationInputs(
                formation_session=FORMATION,
                scores=projection.scores,
                decision_eligible=projection.live,
                risk_allocation=None,
                risk_attribution=None,
                causal_rank_return_curve=None,
                score_projection=projection,
            ),
        )
        for component, projection in projections.items()
    }
    receipt_hash = canonical_hash(
        [projections[component].projection_hash for component in COMPONENT_IDS]
    )
    book = INSTALLED_HETEROGENEOUS_BOOK_RECIPE.component_book_recipe
    provider = open_component_book(
        tuple(
            CappedSleeveComponent(
                component_id=component,
                allocation_basis_points=2_500,
                top_k=book.top_k,
                exit_rank=book.exit_rank,
                tranches=book.tranches,
                aggregate_name_cap=book.aggregate_name_cap,
                aggregate_cap_start_formation=book.aggregate_cap_start_formation,
                formations=formation_inputs[component],
                ordered_listing_ids=listings,
            )
            for component in COMPONENT_IDS
        ),
        initial_sleeve_weights=None,
        schedule_offset=0,
    )
    decision = provider(
        formation_index=0,
        reference_weights=np.zeros(len(listings), dtype=np.float64),
        pretrade_weights=np.zeros(len(listings), dtype=np.float64),
        decision_mode="REBALANCE",
    )
    if (
        not np.isfinite(decision.target_weights).all()
        or abs(float(decision.target_weights.sum()) - 1.0) > 1e-12
        or decision.requires_risk_forecast
    ):
        raise ValueError("portfolio_strategy_lab.gate_m_live_consumer_invalid")
    after = _snapshot(sources)
    if before != after:
        raise ValueError("alpha_research.gate_m_source_changed_during_run")
    result = HeterogeneousLiveScoreClosureReceipt.seal(
        {
            "kind": "HeterogeneousLiveScoreClosureReceipt",
            "formation_session": FORMATION,
            "strategy_hash": INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash,
            "book_recipe_hash": INSTALLED_HETEROGENEOUS_BOOK_RECIPE.recipe_hash,
            "component_model_set_hashes": {
                component: bundles[component].model_set_manifest_hash for component in COMPONENT_IDS
            },
            "component_projection_hashes": {
                component: projections[component].projection_hash for component in COMPONENT_IDS
            },
            "raw_prediction_maximum_gaps": raw_gaps,
            "aggregate_score_maximum_gaps": aggregate_gaps,
            "maximum_raw_prediction_gap": max(raw_gaps.values()),
            "maximum_aggregate_score_gap": max(aggregate_gaps.values()),
            "decision_eligible_count": int(eligible.sum()),
            "target_name_count": int(np.count_nonzero(decision.target_weights)),
            "target_weight_sum": float(decision.target_weights.sum()),
            "target_weights_hash": hashlib.sha256(
                np.ascontiguousarray(decision.target_weights, dtype="<f8").tobytes()
            ).hexdigest(),
            "consumed_score_receipt_hash": receipt_hash,
            "fit_count": 0,
            "prediction_call_count": 96,
            "portfolio_decision_count": 1,
            "provider_call_count": 0,
            "protected_evaluation_count": 0,
            "pointer_mutation_count": 0,
            "source_readback_unchanged": True,
            "disposition": "READY",
        }
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    serialized = cast(dict[str, Any], result.model_dump(mode="json"))
    _write_json(args.output_root / "live-score-closure-receipt.json", serialized)
    _write_json(
        args.output_root / "source-readback.json",
        {
            "sources": after,
            "ported_source_provenance": provenance,
            "executable_owner_readback": closure.executable_owner_readback(),
        },
    )
    return serialized


def _closure_file(root: Path, relative: str) -> HeterogeneousClosureFile:
    path = root / relative
    return HeterogeneousClosureFile(
        relative_path=relative, sha256=_sha256(path), byte_count=path.stat().st_size
    )


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    """Seal the minimum product-readable closure for the last formations."""

    target = args.closure_output_root.resolve()
    if target.exists():
        raise ValueError("alpha_research.closure_root_exists_and_is_immutable")
    sources = _source_paths(args)
    provenance = _provenance(args)
    before = _require_named_sources(args, sources)
    axes, formations = _admitted_axes(args, formation_count=args.closure_formation_count)
    listings = tuple(str(value) for value in axes.listings)
    count = len(formations)
    g0, momentum, resolution_hash = _g0_bundle(args, listings=listings, formations=formations)
    bundles: dict[ComponentId, _Bundle] = {"G0_IW184": g0}
    for component in COMPONENT_IDS[1:]:
        bundles[component] = _specialist_bundle(
            args, component_id=component, listings=listings, formations=formations
        )
    owner = AlphaRuntimeHeterogeneousPredictionOwner()
    gaps = {
        component: _raw_prediction_gap(bundles[component], formation_count=count, owner=owner)
        for component in COMPONENT_IDS
    }
    if max(gaps.values()) != 0.0 or owner.environment_hash is None:
        raise ValueError("alpha_research.closure_raw_prediction_gap_nonzero")

    building = target.with_name(target.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    written: list[Path] = []

    def _save(relative: str, writer: Any) -> str:
        path = building / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        writer(path)
        written.append(path)
        return relative

    components = []
    for component_id in COMPONENT_IDS:
        bundle = bundles[component_id]
        recipe = INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(component_id)
        children = []
        for model in bundle.models:
            source = bundle.payload_paths[(model.vintage, model.seed)]
            relative = _save(
                f"payloads/{component_id}/{model.vintage}-seed-{model.seed}.json",
                lambda path, source=source: path.write_bytes(source.read_bytes()),
            )
            children.append(
                HeterogeneousClosureChild(
                    vintage=model.vintage,
                    seed=model.seed,
                    recipe_hash=model.recipe_hash,
                    content_hash=model.estimator.content_hash,
                    lineage_hash=model.lineage_hash,
                    training_binding_hash=model.training_binding_hash,
                    payload=_closure_file(building, relative),
                )
            )
        surfaces = []
        axis = recipe.ordered_feature_ids or bundle.models[0].estimator.ordered_feature_ids
        for vintage in reversed(VINTAGES):
            features, source_binding = bundle.surfaces[vintage]
            stacked = np.ascontiguousarray(features, dtype=np.float64)
            relative = _save(
                f"surfaces/{component_id}/{vintage}.npz",
                lambda path, stacked=stacked: np.savez(path, features=stacked),
            )
            surfaces.append(
                HeterogeneousClosureSurface(
                    vintage=vintage,
                    source_binding_hash=source_binding,
                    feature_values_hashes=tuple(
                        HeterogeneousVintageFeatureSurface.create(
                            vintage=vintage,
                            ordered_listing_ids=listings,
                            ordered_feature_ids=axis,
                            features=stacked[index],
                            source_binding_hash=source_binding,
                        ).feature_values_hash
                        for index in range(count)
                    ),
                    file=_closure_file(building, relative),
                )
            )
        components.append(
            HeterogeneousClosureComponent(
                component_id=component_id,
                recipe_hash=recipe.recipe_hash,
                feature_axis_hash=recipe.feature_axis_hash,
                feature_count=recipe.feature_count,
                model_set_hash=live_model_set_hash(
                    recipe_hash=recipe.recipe_hash,
                    content_hashes=tuple(value.estimator.content_hash for value in bundle.models),
                ),
                children=tuple(children),
                surfaces=tuple(surfaces),
            )
        )
    momentum_lane = np.ascontiguousarray(momentum, dtype=np.float64)
    momentum_relative = _save("momentum.npy", lambda path: np.save(path, momentum_lane))
    manifest = HeterogeneousCurrentClosureManifest.seal(
        strategy_hash=INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.strategy_hash,
        gate_m_receipt_hash=LIVE_SCORE_CLOSURE_RECEIPT_HASH,
        gate_m_source_readback_sha256=LIVE_SOURCE_READBACK_SHA256,
        gate_l_manifest_hash=LIVE_MODEL_CLOSURE_MANIFEST_HASH,
        live_vintages=LIVE_MODEL_VINTAGES,
        formation_sessions=formations,
        ordered_listing_ids=listings,
        listing_axis_hash=canonical_hash(list(listings)),
        numerical_environment_hash=owner.environment_hash,
        maximum_raw_prediction_gap=max(gaps.values()),
        source_provenance={
            "gate_m_source_readback_sha256": LIVE_SOURCE_READBACK_SHA256,
            "gate_d2_runner_sha256": before[str(args.gate_d2_runner.resolve())]["sha256"],
            "monthly_current_source_cache_identity_hash": resolution_hash,
            "prediction_call_count": str(owner.predictions),
            "executable_owner_readback_hash": str(
                canonical_hash(closure.executable_owner_readback())
            ),
        },
        components=tuple(components),
        momentum=HeterogeneousClosureMomentum(
            factor_id=MOMENTUM_FACTOR_ID,
            source_identity_hash=canonical_hash(
                {
                    "kind": "HeterogeneousMomentumLane",
                    "source_resolution_hash": resolution_hash,
                    "factor_id": MOMENTUM_FACTOR_ID,
                }
            ),
            value_hashes=tuple(array_value_hash(momentum_lane[index]) for index in range(count)),
            file=_closure_file(building, momentum_relative),
        ),
    )
    _write_json(
        building / CLOSURE_MANIFEST_FILE_NAME,
        cast(dict[str, Any], manifest.model_dump(mode="json")),
    )
    written.append(building / CLOSURE_MANIFEST_FILE_NAME)
    after = _snapshot(sources)
    if before != after:
        shutil.rmtree(building)
        raise ValueError("alpha_research.gate_m_source_changed_during_run")
    for path in written:
        os.chmod(path, 0o444)
    os.replace(building, target)
    admitted = admit_heterogeneous_current_closure(target)
    return {
        "closure_root": str(target),
        "manifest_hash": admitted.manifest.manifest_hash,
        "authority_hash": admitted.authority.authority_hash,
        "formation_sessions": [value.isoformat() for value in formations],
        "listing_count": len(listings),
        "numerical_environment_hash": owner.environment_hash,
        "maximum_raw_prediction_gap": max(gaps.values()),
        "prediction_call_count": owner.predictions,
        "files": len(written),
        "ported_source_provenance": provenance,
        "executable_owner_readback": closure.executable_owner_readback(),
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="alphalattice-heterogeneous-live-score-closure")
    parser.add_argument("--monthly-research-worktree", type=Path, required=True)
    parser.add_argument("--heterogeneous-research-worktree", type=Path, required=True)
    parser.add_argument(
        "--gate-d2-runner",
        type=Path,
        required=True,
        help=(
            "the Gate D2 runner of the heterogeneous research worktree; named and hashed, never run"
        ),
    )
    parser.add_argument(
        "--risk-lab-root",
        type=Path,
        required=True,
        help=(
            "the Risk lab experiment root holding `_panel_shared` and the pinned batch owners "
            "that Gate D2 admits its axes against"
        ),
    )
    parser.add_argument("--gate-d-score-root", type=Path, required=True)
    parser.add_argument("--g0-model-root", type=Path, required=True)
    parser.add_argument("--g0-prediction-root", type=Path, required=True)
    parser.add_argument("--g0-axis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gate-m-root",
        type=Path,
        default=None,
        help="the sealed Gate M output root; required to materialise a closure",
    )
    parser.add_argument(
        "--closure-output-root",
        type=Path,
        default=None,
        help="seal a product-readable closure here instead of scoring one formation",
    )
    parser.add_argument("--closure-formation-count", type=int, default=5)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "name and hash every source, verify the port provenance and the executable "
            "owners, then stop before any array is read"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.preflight:
        result = _preflight(args)
    elif args.closure_output_root is not None:
        if args.gate_m_root is None:
            raise SystemExit("--gate-m-root is required to materialise a closure")
        result = _materialize(args)
    else:
        result = _run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
