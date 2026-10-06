"""Admit one exact, externally prepared QA checkpoint; ordinary launch stays workspace-only."""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alphalattice.control.product_host.composition.research_workspace import (  # noqa: E402
    ResearchWorkspaceDecisionUpdate,
    ResearchWorkspaceManifest,
    admit_research_workspace,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.strategy_activation import (  # noqa: E402
    admit_decision_checkpoint,
)
from alphalattice.control.task_control.contracts import TaskLifecycle  # noqa: E402
from alphalattice.control.task_control.registry import (  # noqa: E402
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease  # noqa: E402
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (  # noqa: E402
    PortfolioDecisionCheckpoint,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (  # noqa: E402
    PortfolioLedgerStore,
)


def bind_checkpoint(
    workspace: Path, checkpoint: PortfolioDecisionCheckpoint, *, expected_hash: str
) -> str:
    """The hash is an explicit QA admission, not a claim of a live-account bootstrap.

    Preparation must retain its executed-entry provenance. This command admits
    only the supplied complete state, never constructs a book from final weights.
    """
    if checkpoint.content_hash != expected_hash:
        raise ValueError("portfolio_update.checkpoint_admission_hash_mismatch")
    root = workspace.resolve()
    with closing(WorkspaceWriterLease.acquire(root)):
        admitted = admit_research_workspace(root)
        previous = admitted.manifest
        package_id = checkpoint.package.strategy_id
        if (
            admitted.require_catalog().packages_by_strategy_id().get(package_id)
            != checkpoint.package
        ):
            raise ValueError("portfolio_update.checkpoint_package_mismatch")
        # The checks every binder runs, a person's activation among them (OW10).
        admit_decision_checkpoint(root, checkpoint, previous.score_inputs or ())
        if any(
            task.lifecycle
            not in {TaskLifecycle.SUCCEEDED, TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}
            for task in DuckDbTaskControlRegistry.read_existing_tasks(
                resolve_task_control_database(root)
            )
        ):
            raise ValueError("portfolio_update.unsettled_tasks")
        bindings = tuple(
            v for v in previous.decision_updates or () if v.strategy_package_id != package_id
        )
        old = next(
            (v for v in previous.decision_updates or () if v.strategy_package_id == package_id),
            None,
        )
        wanted = ResearchWorkspaceDecisionUpdate(
            strategy_package_id=package_id,
            strategy_package_hash=checkpoint.package.package_hash,
            checkpoint_hash=checkpoint.content_hash,
        )
        if old is not None and old != wanted:
            raise ValueError("portfolio_update.checkpoint_already_admitted")
        store = PortfolioLedgerStore.for_workspace(root)
        store.publish_decision_checkpoint(checkpoint)
        values = {
            name: getattr(previous, name)
            for name in type(previous).model_fields
            if name not in {"kind", "manifest_schema", "manifest_hash"}
        }
        values["decision_updates"] = tuple(
            sorted((*bindings, wanted), key=lambda v: v.strategy_package_id)
        )
        updated = ResearchWorkspaceManifest.create(**values)
        publish_research_workspace_manifest(root, updated)
    return str(updated.manifest_hash)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-hash", required=True)
    args = parser.parse_args()
    value = PortfolioDecisionCheckpoint.model_validate_json(args.checkpoint.read_bytes())
    print(bind_checkpoint(args.workspace, value, expected_hash=args.expected_hash))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
