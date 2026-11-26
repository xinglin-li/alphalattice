"""Read the frozen Alpha current terminal and emit an isolated parity receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
SRC = PLAYPEN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alphalattice.investment.alpha_research.candidates.artifacts import (  # noqa: E402
    AlphaGoalResearchArtifactStore,
)
from alphalattice.investment.alpha_research.candidates.committer import (  # noqa: E402
    AlphaGoalResultCommitter,
)
from alphalattice.investment.alpha_research.publication.artifacts import (  # noqa: E402
    AlphaCurrentArtifactStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash  # noqa: E402


def _resolved_workspace_pair(baseline_workspace: Path, output_workspace: Path) -> tuple[Path, Path]:
    baseline = baseline_workspace.resolve()
    output = output_workspace.resolve()
    if baseline == output or baseline in output.parents or output in baseline.parents:
        raise ValueError("alpha_research.parity_workspace_overlap")
    return baseline, output


def _readback(baseline_workspace: Path) -> dict[str, object]:
    artifact_root = baseline_workspace / "artifacts"
    goal_store = AlphaGoalResearchArtifactStore(artifact_root)
    numerical_store = AlphaCurrentArtifactStore(artifact_root)
    marker = goal_store.find_active_marker()
    if marker is None:
        raise ValueError("alpha_research.parity_active_marker_missing")
    program = goal_store.load_program(marker.program_hash)
    committed = AlphaGoalResultCommitter(
        goal_store,
        numerical_store=numerical_store,
    ).load_exact_replay(program.program_hash)
    if committed is None or committed.marker.marker_hash != marker.marker_hash:
        raise ValueError("alpha_research.parity_terminal_lineage_incomplete")
    registry = goal_store.load_registry(marker.registry_hash)
    qualification = goal_store.load_qualification(marker.qualification_hash)
    progress = goal_store.load_goal_progress(marker.goal_progress_hash)
    candidate_state_hashes: list[str] = []
    score_child_hashes: list[str] = []
    for candidate in registry.candidates:
        if candidate.current_state_hash is not None:
            state = numerical_store.load_estimator_state(candidate.current_state_hash)
            if state.candidate_id != candidate.candidate_id:
                raise ValueError("alpha_research.parity_current_state_binding_mismatch")
            candidate_state_hashes.append(state.state_hash)
        if candidate.current_score_child_hash is not None:
            child = numerical_store.load_current_candidate_score(candidate.current_score_child_hash)
            if child.candidate_id != candidate.candidate_id:
                raise ValueError("alpha_research.parity_current_score_binding_mismatch")
            score_child_hashes.append(child.child_hash)
    identity = {
        "program_hash": program.program_hash,
        "marker_hash": marker.marker_hash,
        "disposition": marker.disposition,
        "registry_hash": registry.registry_hash,
        "qualification_hash": qualification.qualification_hash,
        "goal_progress_hash": progress.progress_hash,
        "candidate_set_snapshot_hash": marker.candidate_set_snapshot_hash,
        "scientific_stop_hash": marker.scientific_stop_hash,
        "current_state_hashes": tuple(candidate_state_hashes),
        "current_score_child_hashes": tuple(score_child_hashes),
    }
    return {
        "kind": "AlphaModelAdapterParityReceipt",
        "mode": "VERIFIED_ARTIFACT_READBACK_ONLY",
        "numerical_fit_call_count": 0,
        "numerical_predict_call_count": 0,
        "identity": identity,
        "receipt_hash": canonical_hash(identity),
    }


def _publish(output_workspace: Path, receipt: dict[str, object]) -> Path:
    destination = output_workspace / "artifacts" / "alpha-model-adapter-parity"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{receipt['receipt_hash']}.json"
    encoded = json.dumps(receipt, sort_keys=True, indent=2, default=str).encode("utf-8")
    if path.is_file():
        if path.read_bytes() != encoded:
            raise ValueError("alpha_research.parity_receipt_identity_reused")
        return path
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-workspace", type=Path, required=True)
    parser.add_argument("--output-workspace", type=Path, required=True)
    args = parser.parse_args()
    baseline, output = _resolved_workspace_pair(
        args.baseline_workspace,
        args.output_workspace,
    )
    receipt = _readback(baseline)
    receipt_path = _publish(output, receipt)
    print(
        json.dumps(
            {**receipt, "receipt_path": str(receipt_path)},
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
