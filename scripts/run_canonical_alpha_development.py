"""Publish the canonical Alpha target and its cross-sectional scale for one snapshot.

The installed entry point for `CanonicalAlphaDevelopmentService`. Until this
existed the service had a real Host implementation and no non-test caller, which
is the same gap in a different place: a capability module rather than an
installed capability.

Deliberately three flags and no schedule. Which snapshot to process, which Sector
revision it is residualized against, and how much maturity to require are the
caller's to state; every identity that gives the result authority is derived
inside the service from the evidence those handles resolve to.

``--calibrate-return-unit`` is the third mode and the reason the roots below are
separable. It rebuilds one published canonical surface from the read-only causal
outcome authority, projects the raw economic return onto the exact row axis a
Stage 3 decision calibrated on, and re-fits the slope against
``dispersion * predicted_z`` so the constant is dimensionless. It reads three
roots that need not be the same workspace -- the outcome snapshot, the Feature
Panel that carries the Sector map, and the Alpha campaign evidence -- because in
practice they are not.

No model is fitted here and no current pointer is written. This publishes
development evidence -- a canonical target surface, its recipe binding, the
lagged dispersion forecast, and the return-unit calibration successor -- and then
re-reads and verifies it.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"
if str(PLAYPEN_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYPEN_SRC))

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.investment.alpha_research.calibration.authority import (
    seal_installed_return_unit_calibration,
)
from alphalattice.investment.alpha_research.calibration.return_unit import (
    AlphaReturnUnitCalibrationError,
)
from alphalattice.investment.alpha_research.experiments.campaign_evidence import (
    verify_alpha_campaign_decision,
)
from alphalattice.investment.alpha_research.experiments.canonical_development import (
    CanonicalAlphaDevelopmentService,
    CanonicalDevelopmentError,
    CanonicalDevelopmentRequest,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.selected_scores import (
    project_alpha_selected_rows,
    project_alpha_selected_scores,
)


def _service(
    workspace: Path,
    *,
    outcome_artifacts: Path | None = None,
    panel_artifacts: Path | None = None,
) -> CanonicalAlphaDevelopmentService:
    """Compose the service. The three roots are separable because they differ.

    A single workspace is the common case and stays the default. The canonical
    target's causal outcome authority, the Feature Panel that carries its Sector
    map, and the Alpha campaign evidence that consumed it are three published
    locations, and requiring them to be one directory would force a copy -- which
    is how a "read-only source" quietly becomes a second source of truth.
    """

    artifact_root = workspace / "artifacts"
    return CanonicalAlphaDevelopmentService(
        artifact_root=artifact_root,
        outcome_reader=CausalExecutionOutcomeDevelopmentReader(outcome_artifacts or artifact_root),
        resolver=ArtifactResolver(panel_artifacts or artifact_root),
    )


def _calibrate_return_unit(
    *,
    workspace: Path,
    outcome_artifacts: Path,
    panel_artifacts: Path,
    upstream_evidence: Path,
    dossier_hash: str,
    decision_hash: str,
    horizon_sessions: int,
    sector_revision: str | None,
) -> dict[str, object]:
    """Rebuild the canonical y lane and re-fit the Alpha slope in return units."""

    upstream = AlphaDevelopmentArtifactStore(upstream_evidence)
    dossier = upstream.load_campaign_dossier(dossier_hash)
    decision = upstream.load_campaign_decision(decision_hash)
    # The Alpha Desk's own rule, not a field read: the decision is re-derived
    # from the dossier and the installed policy before anything it selected is
    # consumed.
    verify_alpha_campaign_decision(decision=decision, dossier=dossier)

    surfaces = upstream.root / "development" / "campaign" / "prediction-surfaces"
    projection, _matrix = project_alpha_selected_scores(
        dossier=dossier,
        decision=decision,
        prediction_surface_root=surfaces,
        horizon_sessions=horizon_sessions,
    )
    row_axis = project_alpha_selected_rows(
        dossier=dossier,
        decision=decision,
        prediction_surface_root=surfaces,
        horizon_sessions=horizon_sessions,
    )
    superseded = next(
        (
            value
            for value in dossier.calibration_evidence
            if value.horizon_sessions == horizon_sessions
        ),
        None,
    )
    if superseded is None:
        raise CanonicalDevelopmentError("alpha_research.canonical_superseded_calibration_absent")

    reader = CanonicalAlphaDevelopmentService(
        artifact_root=upstream_evidence,
        outcome_reader=CausalExecutionOutcomeDevelopmentReader(outcome_artifacts),
        resolver=ArtifactResolver(panel_artifacts),
    )
    revision = reader.resolved_sector_revision(sector_revision)
    materialization = reader.materialize(
        evidence_hash=projection.target_evidence_hash, sector_revision=revision
    )
    # Through the installed sealer, so the published artifact carries the
    # capability that was allowed to produce it. Calling the numerical function
    # directly would publish the same numbers with nothing to check them against.
    calibration = seal_installed_return_unit_calibration(
        projection=projection,
        row_axis=row_axis,
        materialization=materialization,
        superseded=superseded,
    )
    # Written to the development output workspace, never back to a source root.
    store = AlphaDevelopmentArtifactStore((workspace / "artifacts").resolve())
    uri = store.publish_return_unit_calibration(calibration.evidence)
    readback = store.load_return_unit_calibration(calibration.evidence.evidence_hash)
    if readback.evidence_hash != calibration.evidence.evidence_hash:
        raise CanonicalDevelopmentError("alpha_research.canonical_successor_readback_mismatch")
    return {
        "action": "CALIBRATED",
        "return_unit_calibration_evidence_hash": readback.evidence_hash,
        "return_unit_calibration_uri": uri,
        "composition": readback.composition,
        "slope_units": readback.slope_units,
        "capability_binding_hash": readback.capability_binding_hash,
        "implementation_closure_hash": readback.implementation_closure_hash,
        "slope_by_fold": list(readback.slope_by_fold),
        "superseded_calibration_evidence_hash": readback.superseded_calibration_evidence_hash,
        "superseded_disposition": readback.superseded_disposition,
        "superseded_slope_units": readback.superseded_slope_units,
        "superseded_slope_by_fold": list(readback.superseded_slope_by_fold),
        "row_count": readback.row_count,
        "row_axis_hash": readback.row_axis_hash,
        "ordered_fold_row_counts": list(readback.ordered_fold_row_counts),
        "canonical_target_evidence_hash": readback.target_evidence_hash,
        "canonical_lane_identity_hash": readback.canonical_lane_identity_hash,
        "simple_economic_return_lane_identity": readback.simple_economic_return_lane_identity,
        "cross_sectional_dispersion_lane_identity": (
            readback.cross_sectional_dispersion_lane_identity
        ),
        "scaled_score_identity": readback.scaled_score_identity,
        "economic_return_identity": readback.economic_return_identity,
        "sector_revision": revision,
        "fold_states": [
            {
                "fold_index": value.fold_index,
                "training_observation_count": value.training_observation_count,
                "score_rms": value.score_rms,
                "target_rms": value.target_rms,
                "normalized_cross_moment": value.normalized_cross_moment,
                "shrinkage_ratio": value.shrinkage_ratio,
                "slope": value.slope,
            }
            for value in readback.calibration.fold_states
        ],
        "current_pointer_writes": 0,
        "model_fits": 0,
        "provider_calls": 0,
    }


def _publish(
    workspace: Path, *, snapshot_hash: str, sector_revision: str, holding_end_through: date
) -> dict[str, object]:
    service = _service(workspace)
    published = service.publish(
        CanonicalDevelopmentRequest(
            causal_outcome_snapshot_hash=snapshot_hash,
            sector_revision=sector_revision,
            holding_end_through=holding_end_through,
        )
    )
    # Verified from the store, not from the objects still in memory.
    lineage = service.verify(
        evidence_hash=published.evidence.evidence_hash,
        dispersion_hash=published.dispersion.forecast_hash,
    )
    return {
        "action": "PUBLISHED",
        "causal_outcome_snapshot_hash": lineage.outcome_snapshot_hash,
        "outcome_method_binding_hash": lineage.outcome_method_binding_hash,
        "maturity_lag_sessions": lineage.maturity_lag_sessions,
        "target_recipe_binding_hash": published.recipe_binding.binding_hash,
        "canonical_target_evidence_hash": published.evidence.evidence_hash,
        "dispersion_forecast_hash": published.dispersion.forecast_hash,
        "formation_count": len(published.formation_sessions),
        "listing_count": len(published.ordered_listing_ids),
        "current_pointer_writes": 0,
        "model_fits": 0,
        "provider_calls": 0,
    }


def _inspect(workspace: Path, *, snapshot_hash: str) -> dict[str, object]:
    """Report what is already published for one snapshot, verifying each entry."""

    service = _service(workspace)
    evidence_hashes = service.find_published_evidence(snapshot_hash=snapshot_hash)
    return {
        "action": "INSPECTED",
        "causal_outcome_snapshot_hash": snapshot_hash,
        "canonical_target_evidence_hashes": list(evidence_hashes),
        "matched_score_and_dispersion_pairs": [
            {"canonical_score_binding_hash": score, "dispersion_forecast_hash": forecast}
            for score, forecast in service.find_matched_canonical_inputs()
        ],
        "current_pointer_writes": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=PLAYPEN_ROOT / "workspaces" / "us-research" / "runtime",
    )
    parser.add_argument("--snapshot-hash", default=None)
    parser.add_argument("--sector-revision", default=None)
    parser.add_argument("--holding-end-through", type=date.fromisoformat, default=None)
    parser.add_argument("--outcome-artifacts", type=Path, default=None)
    parser.add_argument("--panel-artifacts", type=Path, default=None)
    parser.add_argument("--upstream-evidence", type=Path, default=None)
    parser.add_argument("--dossier-hash", default=None)
    parser.add_argument("--decision-hash", default=None)
    parser.add_argument("--horizon-sessions", type=int, default=1)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--publish", action="store_true")
    action.add_argument("--inspect", action="store_true")
    action.add_argument("--calibrate-return-unit", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()

    if args.calibrate_return_unit:
        missing = [
            name
            for name, value in (
                ("--outcome-artifacts", args.outcome_artifacts),
                ("--panel-artifacts", args.panel_artifacts),
                ("--upstream-evidence", args.upstream_evidence),
                ("--dossier-hash", args.dossier_hash),
                ("--decision-hash", args.decision_hash),
            )
            if value is None
        ]
        if missing:
            parser.error("--calibrate-return-unit requires " + ", ".join(missing))
        try:
            result = _calibrate_return_unit(
                workspace=workspace,
                outcome_artifacts=args.outcome_artifacts.resolve(),
                panel_artifacts=args.panel_artifacts.resolve(),
                upstream_evidence=args.upstream_evidence.resolve(),
                dossier_hash=str(args.dossier_hash),
                decision_hash=str(args.decision_hash),
                horizon_sessions=int(args.horizon_sessions),
                sector_revision=args.sector_revision,
            )
        except (CanonicalDevelopmentError, AlphaReturnUnitCalibrationError) as error:
            result = {
                "action": "REFUSED",
                "failure_code": str(error),
                "current_pointer_writes": 0,
            }
        print(json.dumps(result, sort_keys=True, default=str))
        return 0

    if args.snapshot_hash is None:
        parser.error("--publish and --inspect require --snapshot-hash")

    if args.inspect:
        result = _inspect(workspace, snapshot_hash=args.snapshot_hash)
    else:
        if args.sector_revision is None or args.holding_end_through is None:
            parser.error("--publish requires --sector-revision and --holding-end-through")
        try:
            result = _publish(
                workspace,
                snapshot_hash=args.snapshot_hash,
                sector_revision=args.sector_revision,
                holding_end_through=args.holding_end_through,
            )
        except FileNotFoundError:
            # A handle naming evidence that was never published. Distinct from a
            # refusal, because nothing is wrong with the request except that the
            # snapshot it names does not exist here.
            result = {
                "action": "UNRESOLVED",
                "failure_code": "alpha_research.canonical_outcome_snapshot_not_published",
                "current_pointer_writes": 0,
            }
        except CanonicalDevelopmentError as error:
            # A typed refusal is the informative outcome, not a stack trace: an
            # unsealed snapshot or an axis too short for the maturity lag is a
            # statement about the evidence, not a defect in this entry point.
            result = {
                "action": "REFUSED",
                "failure_code": str(error),
                "current_pointer_writes": 0,
            }
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
