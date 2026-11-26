"""Publish or inspect deterministic Sector Research development experiments.

The installed entry point for ``SectorResearchDevelopmentService``. Without it
the Sector capability would repeat the pattern this program has now corrected
twice: a Host service with real behaviour and no non-test caller, which is a
proposal wearing the name of a capability.

The caller chooses a snapshot handle, a sector revision, an installed method,
optionally a restatement of the admitted singleton parameters, and the training
and forecast ranges. The caller cannot submit a catalog hash, method binding,
target hash, axes hash or evidence hash: every identity that gives the result
authority is derived inside the service from the evidence those handles resolve
to.

No Alpha model is fitted and no current pointer is written. This publishes
development evidence -- a clean-target evidence record, a forecast surface, an
evaluation and the experiment root -- and then re-reads and verifies it.
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
from alphalattice.investment.sector_research.contracts import SectorResearchError
from alphalattice.investment.sector_research.experiments.campaign import SectorCampaignRequest
from alphalattice.investment.sector_research.experiments.campaign_evidence import (
    SectorForecastSelectionSubmission,
)
from alphalattice.investment.sector_research.experiments.service import (
    SectorExperimentRequest,
    SectorResearchDevelopmentService,
)
from alphalattice.protocols.actor_execution import ActorKind


def _service(
    workspace: Path,
    *,
    outcome_root: Path | None = None,
    membership_root: Path | None = None,
    output_root: Path | None = None,
) -> SectorResearchDevelopmentService:
    """Compose the Host, optionally reading and writing in different places.

    A single workspace is the ordinary shape and stays the default. A Campaign
    against evidence somebody else produced needs the other one: outcomes and
    Sector membership resolve out of read-only source workspaces while every new
    artifact is written to a separate development root, so a research run cannot
    add to the workspace whose authority it is consuming.
    """

    artifact_root = workspace / "artifacts"
    return SectorResearchDevelopmentService(
        artifact_root=output_root or artifact_root,
        outcome_reader=CausalExecutionOutcomeDevelopmentReader(outcome_root or artifact_root),
        resolver=ArtifactResolver(membership_root or artifact_root),
    )


def _publish(workspace: Path, args: argparse.Namespace) -> dict[str, object]:
    service = _service(workspace)
    parameters = (
        {str(key): int(value) for key, value in json.loads(args.parameters).items()}
        if args.parameters is not None
        else None
    )
    published = service.publish(
        SectorExperimentRequest(
            causal_outcome_snapshot_hash=args.snapshot_hash,
            sector_revision=args.sector_revision,
            method_id=args.method,
            training_start=args.training_start,
            training_end=args.training_end,
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
            parameters=parameters,
        )
    )
    # Verified from the store, not from the objects still in memory.
    lineage = service.verify(experiment_hash=published.experiment.experiment_hash)
    return {
        "action": "PUBLISHED",
        "experiment_hash": published.experiment.experiment_hash,
        "sector_target_evidence_hash": published.target_evidence.evidence_hash,
        "forecast_surface_hash": published.surface.surface_hash,
        "evaluation_hash": published.evaluation.evaluation_hash,
        "causal_outcome_snapshot_hash": lineage.outcome_snapshot_hash,
        "outcome_method_binding_hash": lineage.outcome_method_binding_hash,
        "maturity_lag_sessions": lineage.maturity_lag_sessions,
        "method_id": published.surface.recipe.method_id,
        "formation_count": len(published.formation_sessions),
        "forecast_formation_count": len(published.forecast_formation_sessions),
        "sector_count": len(published.ordered_sectors),
        "current_pointer_writes": 0,
        "alpha_model_fits": 0,
    }


def _inspect(workspace: Path, *, experiment_hash: str | None) -> dict[str, object]:
    """Verify what is already published, walking each experiment back to its seal."""

    service = _service(workspace)
    hashes = (
        (experiment_hash,) if experiment_hash is not None else service.find_published_experiments()
    )
    experiments: list[dict[str, object]] = []
    for value in hashes:
        lineage = service.verify(experiment_hash=value)
        experiments.append(
            {
                "experiment_hash": value,
                "method_id": lineage.surface.recipe.method_id,
                "causal_outcome_snapshot_hash": lineage.outcome_snapshot_hash,
                "outcome_method_binding_hash": lineage.outcome_method_binding_hash,
                "sector_target_evidence_hash": lineage.target_evidence.evidence_hash,
                "forecast_formation_count": len(lineage.surface.forecast_formation_sessions),
                "verification": "VERIFIED",
            }
        )
    return {
        "action": "INSPECTED",
        "experiments": experiments,
        "current_pointer_writes": 0,
    }


def _campaign(workspace: Path, args: argparse.Namespace) -> dict[str, object]:
    """Run the whole comparison, submit a recommendation, seal it, replay it.

    External automation submits the recommendation the published comparison
    already decided; it cannot seal one. The Host re-derives the decision from
    the comparison's own rows before sealing, so a submission that disagrees
    with the evidence is refused rather than recorded.
    """

    service = _service(
        workspace,
        outcome_root=args.outcome_artifact_root,
        membership_root=args.membership_artifact_root,
        output_root=args.output_artifact_root,
    )
    request = SectorCampaignRequest.from_yaml(
        args.config.resolve().read_text(encoding="utf-8"),
        causal_outcome_snapshot_hash=str(args.snapshot_hash),
        sector_revision=str(args.sector_revision),
    )
    published = service.run_campaign(request)
    comparison = published.comparison

    submission = SectorForecastSelectionSubmission.create(
        dossier_hash=published.dossier.dossier_hash,
        selected_method_id=comparison.recommended_method_id,
        rationale=(
            "Development selection from the published cross-fitted Sector "
            "comparison under the frozen selection rule."
        ),
    )
    receipt = service.seal_campaign_decision(
        dossier=published.dossier,
        comparison=comparison,
        submission=submission,
        actor_kind=ActorKind.EXTERNAL_AUTOMATION,
        actor_id="codex-sector-forecast-campaign",
    )
    replay = service.verify_campaign(
        dossier_hash=published.dossier.dossier_hash,
        decision_receipt_hash=receipt.receipt_hash,
    )
    return {
        "action": "CAMPAIGN_PUBLISHED",
        "dossier_hash": published.dossier.dossier_hash,
        "comparison_hash": comparison.comparison_hash,
        "decision_receipt_hash": receipt.receipt_hash,
        "replay_hash": replay.replay_hash,
        "replay_disposition": replay.disposition,
        "verified_child_count": replay.verified_child_count,
        "target_evidence_hash": published.target_evidence.evidence_hash,
        "selected_method_id": receipt.selection.method_id,
        "selected_experiment_hash": receipt.selection.sector_experiment_hash,
        "joint_dynamics_triggered": comparison.joint_dynamics_triggered,
        "conditional_dispositions": dict(published.dossier.conditional_dispositions),
        "forecast_call_count": published.dossier.forecast_call_count,
        "metric_call_count": published.dossier.metric_call_count,
        "replay_forecast_call_count": replay.forecast_call_count,
        "replay_metric_call_count": replay.metric_call_count,
        "provider_call_count": published.dossier.provider_call_count,
        "holdout_access_count": published.dossier.holdout_access_count,
        "pointer_mutation_count": published.dossier.pointer_mutation_count,
        "methods": [
            {
                "method_id": row.method_id,
                "disposition": row.disposition,
                "evaluated_pair_count": row.evaluated_pair_count,
                "available_cell_count": row.available_cell_count,
                "mean_cross_fitted_slope": row.mean_cross_fitted_slope,
                "mean_calibrated_economic_squared_error": (
                    row.mean_calibrated_economic_squared_error
                ),
                "mean_identity_economic_squared_error": (row.mean_identity_economic_squared_error),
                "experiment_hash": row.experiment_hash,
            }
            for row in comparison.rows
        ],
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
    parser.add_argument("--method", default=None)
    parser.add_argument("--parameters", default=None, help="JSON restatement of the singleton")
    parser.add_argument("--training-start", type=date.fromisoformat, default=None)
    parser.add_argument("--training-end", type=date.fromisoformat, default=None)
    parser.add_argument("--forecast-start", type=date.fromisoformat, default=None)
    parser.add_argument("--forecast-end", type=date.fromisoformat, default=None)
    parser.add_argument("--experiment-hash", default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=PLAYPEN_ROOT / "config" / "sector-forecast-campaign.yaml",
        help="YAML Campaign selections: installed method ids, restated singleton "
        "parameters, ranges, handles and development-only intent. Never authority.",
    )
    parser.add_argument(
        "--outcome-artifact-root",
        type=Path,
        default=None,
        help="Read-only source of causal execution outcomes; defaults to the workspace.",
    )
    parser.add_argument(
        "--membership-artifact-root",
        type=Path,
        default=None,
        help="Read-only source of published Sector revisions; defaults to the workspace.",
    )
    parser.add_argument(
        "--output-artifact-root",
        type=Path,
        default=None,
        help="Where new development evidence is written; defaults to the workspace.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--publish", action="store_true")
    action.add_argument("--inspect", action="store_true")
    action.add_argument("--campaign", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    for field in ("outcome_artifact_root", "membership_artifact_root", "output_artifact_root"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())

    try:
        if args.campaign:
            if args.snapshot_hash is None or args.sector_revision is None:
                # The document used to carry both. They are the operator's inputs
                # now, so a Campaign run states which evidence it consumed at the
                # moment it consumes it rather than in a file that outlives it.
                parser.error("--campaign requires --snapshot-hash and --sector-revision")
            result = _campaign(workspace, args)
        elif args.inspect:
            result = _inspect(workspace, experiment_hash=args.experiment_hash)
        else:
            required = (
                args.snapshot_hash,
                args.sector_revision,
                args.method,
                args.training_start,
                args.training_end,
                args.forecast_start,
                args.forecast_end,
            )
            if any(value is None for value in required):
                parser.error(
                    "--publish requires --snapshot-hash, --sector-revision, --method and the"
                    " four range dates"
                )
            result = _publish(workspace, args)
    except FileNotFoundError:
        # A handle naming evidence that was never published. Distinct from a
        # refusal: nothing is wrong with the request except that what it names
        # does not exist here.
        result = {
            "action": "UNRESOLVED",
            "failure_code": "sector_research.evidence_not_published",
            "current_pointer_writes": 0,
        }
    except SectorResearchError as error:
        # A typed refusal is the informative outcome, not a stack trace.
        result = {
            "action": "REFUSED",
            "failure_code": str(error),
            "current_pointer_writes": 0,
        }
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
