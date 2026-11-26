"""Reconstruct the twelve frozen IW184 live estimators and prove score parity.

Gate L. The research inputs are identity-bound data: the monthly refit driver
and the three workspace modules of the monthly research worktree, the qualified
current-source cache, the candidate axis, and the cached prediction receipts.
Every one of them is named in the manifest with its digest, and the Python
files among them are verified byte for byte against the digests the local port
in ``broad_ensemble_research_closure`` was taken from. None of them is executed. The
code that runs is this tree's: the ported closure, this tree's copy of the
monthly refit driver, and the installed Alpha runtime.

    .venv/Scripts/python.exe scripts/run_broad_ensemble_live_model_closure.py
        --research-worktree <monthly research worktree>
        --prediction-root <cached predictions> --axis-root <prediction axes>
        --output-root <where the twelve payloads and the manifest go>

``--preflight`` names and hashes every source, verifies the port provenance and
the executable owners, prints that record, and stops before any array is read
or any model is fitted.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
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
from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_chronological import (  # noqa: E402
    load_lightgbm_runtime,
)
from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (  # noqa: E402
    build_dynamic_panel_lightgbm_search_domain,
)
from alphalattice.capabilities.alpha_modeling.contracts import AlphaEstimatorContent  # noqa: E402
from alphalattice.capabilities.alpha_modeling.runtime.service import (  # noqa: E402
    AlphaModelRuntimeService,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (  # noqa: E402
    build_panel_model_catalog,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (  # noqa: E402
    materialize_panel_feature_projection,
    preflight_panel_feature_plan,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash  # noqa: E402

VINTAGES = ("2025-10", "2026-01", "2026-04", "2026-07")
SEEDS = (1729, 2718, 31415)
PREFLIGHT_DISPOSITION = "PREFLIGHT_ONLY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _snapshot(paths: set[Path]) -> dict[str, dict[str, Any]]:
    missing = tuple(str(path) for path in sorted(paths) if not path.is_file())
    if missing:
        raise ValueError("alpha_research.gate_l_source_absent:" + "|".join(missing))
    return {
        str(path.resolve()): {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
    }


def _research_paths(args: argparse.Namespace) -> dict[str, Path]:
    """The identity-bound research inputs, all under the declared monthly worktree."""

    research = args.research_worktree / "workspaces/monthly-alpha-refit-research"
    return {
        "runner": args.research_worktree / "scripts/run_monthly_alpha_refit_research.py",
        "ensemble": research / "run_iw184_quarterly_5y_ensemble.py",
        "explorer": research / "feature-recipe-exploration/run_feature_recipe_exploration.py",
        "smoke": research / "frozen-iw184-holdout-smoke/run_frozen_iw184_holdout_smoke.py",
        "candidate_axes": args.research_worktree / closure.CANDIDATE_AXES_RELATIVE_PATH,
        "source_root": args.research_worktree / closure.SOURCE_CACHE_RELATIVE_PATH,
    }


def _named_sources(args: argparse.Namespace) -> tuple[set[Path], dict[str, Any]]:
    paths = _research_paths(args)
    source_root = paths["source_root"]
    source_manifest_path = source_root / "manifest.json"
    source_manifest = cast(
        dict[str, Any], json.loads(source_manifest_path.read_text(encoding="utf-8"))
    )
    sources = {
        paths["runner"],
        paths["ensemble"],
        paths["explorer"],
        paths["smoke"],
        paths["candidate_axes"],
        source_manifest_path,
    }
    sources.update(source_root / value["path"] for value in source_manifest["arrays"].values())
    for vintage in VINTAGES:
        sources.update(
            {
                args.axis_root / f"{vintage}.json",
                args.axis_root / f"{vintage}.npz",
            }
        )
        for seed in SEEDS:
            sources.update(
                {
                    args.prediction_root / vintage / f"seed-{seed}.json",
                    args.prediction_root / vintage / f"seed-{seed}.npz",
                }
            )
    return sources, source_manifest


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="alphalattice-iw184-live-model-closure")
    parser.add_argument("--research-worktree", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--axis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "name and hash every source, verify the port provenance and the executable "
            "owners, then stop before any array is read or any model is fitted"
        ),
    )
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    sources, source_manifest = _named_sources(args)
    readback = _snapshot(sources)
    provenance = closure.verify_ported_research_sources(monthly_worktree=args.research_worktree)
    return {
        "kind": "IW184LiveModelClosurePreflight",
        "disposition": PREFLIGHT_DISPOSITION,
        "source_cache_identity_hash": source_manifest.get("cache_identity_hash"),
        "source_readback": readback,
        "ported_source_provenance": provenance,
        "executable_owner_readback": closure.executable_owner_readback(),
        "fit_count": 0,
        "pointer_mutation_count": 0,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    sources, source_manifest = _named_sources(args)
    paths = _research_paths(args)
    before = _snapshot(sources)
    provenance = closure.verify_ported_research_sources(monthly_worktree=args.research_worktree)
    profile = closure.monthly_runner.PROFILES["sparse-session-amplitude-195"]
    candidate_axis = closure.candidate_axis(args.research_worktree)
    axis_hash = str(canonical_hash(list(candidate_axis)))
    source = closure.load_cached_panel_source(paths["source_root"])
    resolved = SimpleNamespace(
        arrays=source,
        resolution=SimpleNamespace(resolution_hash=source_manifest["cache_identity_hash"]),
    )
    plan = preflight_panel_feature_plan(
        source=resolved.arrays,
        selected_method_ids=(profile.view_id,),
        maximum_aggregation_span=1,
    )
    validation_sessions = tuple(
        session
        for session in resolved.arrays.formation_sessions
        if session.isoformat() >= "2019-08-05"
    )
    try:
        specs = closure.quarter_specs(
            source_sessions=resolved.arrays.formation_sessions,
            validation_sessions=validation_sessions,
            first_vintage=VINTAGES[0],
            last_vintage=VINTAGES[-1],
        )
    except RuntimeError as error:
        geometry = closure.monthly_runner._monthly_geometry(
            source_sessions=resolved.arrays.formation_sessions,
            validation_sessions=validation_sessions,
        )
        available = tuple(
            month
            for month, _training, _validation, _purge in geometry
            if int(month[5:7]) in (1, 4, 7, 10) and VINTAGES[0] <= month <= VINTAGES[-1]
        )
        raise ValueError(
            "alpha_research.gate_l_vintage_axis_unavailable:"
            + ",".join(available)
            + f":source_last={resolved.arrays.formation_sessions[-1].isoformat()}"
        ) from error
    if tuple(value["month"] for value in specs) != VINTAGES:
        raise ValueError("alpha_research.gate_l_vintage_axis_invalid")
    recipes = {seed: closure.broad_ensemble_recipe(seed) for seed in SEEDS}
    runtime = AlphaModelRuntimeService(build_panel_model_catalog())
    domain = build_dynamic_panel_lightgbm_search_domain()
    lightgbm = load_lightgbm_runtime()
    models: list[dict[str, Any]] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        projection = materialize_panel_feature_projection(
            plan=plan,
            method_id=profile.view_id,
            program_hash=str(
                canonical_hash(
                    {
                        "kind": "IW184_FIXED1260_QUARTERLY_SEED3_VINTAGE4_LINEAR",
                        "quarter_start": spec["month"],
                        "source_resolution_hash": resolved.resolution.resolution_hash,
                        "candidate_axis_hash": axis_hash,
                    }
                )
            ),
            fold_index=int(spec["ordinal"]),
            boundary_id="OUTER",
            training_sessions=spec["training"],
            transform_sessions=spec["transform"],
        )
        canonical = tuple(str(value) for value in projection.ordered_feature_ids)
        lookup = {feature_id: index for index, feature_id in enumerate(canonical)}
        positions = np.asarray([lookup[value] for value in candidate_axis], dtype=np.intp)
        training = np.ascontiguousarray(
            projection.training_features[:, positions], dtype=np.float64
        )
        transformed = np.ascontiguousarray(
            projection.transformed_features[:, positions], dtype=np.float64
        )
        targets = np.ascontiguousarray(projection.training_targets, dtype=np.float64)
        training.setflags(write=False)
        transformed.setflags(write=False)
        targets.setflags(write=False)
        candidate_projection = SimpleNamespace(
            ordered_feature_ids=candidate_axis,
            training_features=training,
            training_targets=targets,
            transformed_features=transformed,
            common_training_row_axis_hash=projection.common_training_row_axis_hash,
            scale_receipts=projection.scale_receipts,
        )
        for seed in SEEDS:
            expected_receipt_path = args.prediction_root / spec["month"] / f"seed-{seed}.json"
            expected_artifact_path = args.prediction_root / spec["month"] / f"seed-{seed}.npz"
            expected_receipt = cast(
                dict[str, Any], json.loads(expected_receipt_path.read_text(encoding="utf-8"))
            )
            identity = closure.seed_model_identity(
                axis_hash=axis_hash,
                recipe_hash=recipes[seed].recipe_hash,
                seed=seed,
                spec=spec,
                source_resolution_hash=resolved.resolution.resolution_hash,
            )
            scores, fit_receipt, estimator = closure.monthly_runner._fit_month_with_estimator(
                runtime=runtime,
                recipe=recipes[seed],
                domain=domain,
                profile=profile,
                policy_hash=identity["identity_hash"],
                vintage_index=int(spec["ordinal"]),
                month=f"{spec['month']}-iw184-1260-seed-{seed}",
                projection=candidate_projection,
            )
            with np.load(expected_artifact_path, allow_pickle=False) as saved:
                expected_scores = np.asarray(saved["scores"], dtype=np.float64)
            max_gap = float(np.max(np.abs(scores - expected_scores)))
            model_text = estimator.payload.get("model_text")
            if not isinstance(model_text, str):
                raise ValueError("alpha_research.gate_l_estimator_payload_invalid")
            reopened = np.asarray(
                lightgbm.Booster(model_str=model_text).predict(transformed), dtype=np.float64
            )
            reopen_gap = float(np.max(np.abs(scores - reopened)))
            if (
                identity["identity_hash"] != expected_receipt.get("identity_hash")
                or recipes[seed].recipe_hash != expected_receipt.get("recipe_hash")
                or estimator.content_hash != expected_receipt.get("estimator_content_hash")
                or fit_receipt["training_row_count"] != expected_receipt.get("training_row_count")
                or len(scores) != expected_receipt.get("prediction_row_count")
                or max_gap > 0.0
                or reopen_gap > 0.0
            ):
                raise ValueError(
                    "alpha_research.gate_l_model_parity_invalid:"
                    f"{spec['month']}:{seed}:{max_gap:.17g}:{reopen_gap:.17g}"
                )
            model_path = args.output_root / spec["month"] / f"seed-{seed}.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_suffix(".json.tmp")
            temporary.write_text(estimator.model_dump_json(indent=2) + "\n", encoding="utf-8")
            temporary.replace(model_path)
            reopened_content = AlphaEstimatorContent.model_validate_json(
                model_path.read_text(encoding="utf-8")
            )
            if reopened_content.content_hash != estimator.content_hash:
                raise ValueError("alpha_research.gate_l_model_readback_invalid")
            models.append(
                {
                    "vintage": spec["month"],
                    "seed": seed,
                    "identity_hash": identity["identity_hash"],
                    "recipe_hash": recipes[seed].recipe_hash,
                    "estimator_content_hash": estimator.content_hash,
                    "prediction_receipt_sha256": _sha256(expected_receipt_path),
                    "prediction_artifact_sha256": _sha256(expected_artifact_path),
                    "model_payload_relative_path": model_path.relative_to(
                        args.output_root
                    ).as_posix(),
                    "model_payload_sha256": _sha256(model_path),
                    "training_row_count": len(training),
                    "prediction_row_count": len(scores),
                    "maximum_cached_prediction_gap": max_gap,
                    "maximum_reopen_prediction_gap": reopen_gap,
                }
            )
        del projection, candidate_projection, training, transformed, targets
        gc.collect()
    after = _snapshot(sources)
    if before != after:
        raise ValueError("alpha_research.gate_l_source_changed_during_run")
    manifest: dict[str, Any] = {
        "kind": "IW184LiveModelClosureManifest",
        "recipe_id": "IW184_IMPLIED_PLUS_WITHIN_FIXED1260_QUARTERLY_SEED3_VINTAGE4",
        "candidate_axis_hash": axis_hash,
        "vintages": list(VINTAGES),
        "seeds": list(SEEDS),
        "model_count": len(models),
        "models": models,
        "source_readback": after,
        "source_readback_unchanged": True,
        "ported_source_provenance": provenance,
        "executable_owner_readback": closure.executable_owner_readback(),
        "fit_count": len(models),
        "pointer_mutation_count": 0,
        "disposition": "READY",
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    _write_json(args.output_root / "live-model-closure-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    result = _preflight(args) if args.preflight else _run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
