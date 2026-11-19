"""The ordinary Host's first 0.2.8 start: one upgrade plan, acknowledged, then run.

The plan names every family a merged card declares (`FAMILIES`, in order) and every workspace that
holds a declared family's old data: the started workspace and its project's `workspaces/*`. For the
data root and each such workspace it names the verified backup the upgrade keeps: an existing one
by its generation, or the default repository a new one will be written to, with its estimated bytes
and the free space left after. Nothing moves until the person acknowledges the plan's hash; then
the Host writes and verifies each named backup and runs `UpgradeRunner` under each workspace's
writer lease, and serves its full composition only once every family is done. A started run
resumes on its recorded plan without a new acknowledgement. A workspace that holds old data and
first appears after the machine's run completed is upgraded alone at its open, under its own
subject, by the same families.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Literal

from alphalattice.control.product_host.composition.application_session import (
    verify_upgrade_backups,
)
from alphalattice.control.product_host.composition.research_workspace import (
    RESEARCH_WORKSPACE_MANIFEST_NAME,
    named_workspace_id,
)
from alphalattice.control.product_host.storage.backup import (
    WorkspaceBackupError,
    WorkspaceBackups,
    default_backup_root,
)
from alphalattice.control.workspace_runtime.storage.capacity import FREE_FLOOR_BYTES
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.foundation.causal_outcomes.execution.readers import (
    execution_outcome_upgrade_family,
)
from alphalattice.foundation.market_data_ops.storage.world import market_upgrade_family
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    alpha_array_upgrade_family,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    lane_upgrade_family,
)
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    risk_covariance_upgrade_family,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.backup import backup_data_root, verify_data_root
from alphalattice.kernel.data_layer.log import HostLog
from alphalattice.kernel.data_layer.upgrade import PROGRESS, TARGET, Family, UpgradeRunner
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.model_validation.protected_gate import validation_upgrade_family

UNHELD: Final = ("data_layer.outcome_missing",)
"""The refusal that says only that the catalog holds no body for a chunk: no profile for its
digest, or one without rows. An unavailable reader or any other catalog fault is not evidence
of absence, so it stops the start by its own name."""


def execution_outcomes() -> Family:
    """The execution-outcome family, holding a workspace whose committed legacy chunks the catalog
    lacks or disagrees with: its own ALL EQUAL check, asked of that workspace alone. Its module
    stays as merged, so no study's implementation identity moves for this question."""
    family = execution_outcome_upgrade_family()

    def holds(workspace: Path) -> bool:
        try:
            return family.verify(SimpleNamespace(workspaces=(workspace,))) is not True
        except DataError as error:  # a chunk the catalog does not hold yet; any other fault stops
            if error.failure.code not in UNHELD:
                raise
            return True

    return replace(family, holds=holds)


def strategy_lab_lanes() -> Family:
    """The Strategy Lab's lanes into the array door, each held by the workspace it lies in."""
    return lane_upgrade_family(named_workspace_id)


def alpha_arrays() -> Family:
    """Alpha's old array bodies and score chunks into the array door."""
    return alpha_array_upgrade_family(named_workspace_id)


def protected_validation() -> Family:
    """The protected gate's receipts and permit claims, held by their original workspace."""
    return validation_upgrade_family(named_workspace_id)


def risk_covariance() -> Family:
    """Move Risk's sealed covariance history to the catalog under the workspace that owns it."""
    return risk_covariance_upgrade_family(named_workspace_id)


FAMILIES: Final[tuple[Callable[[], Family], ...]] = (
    partial(market_upgrade_family, named_workspace_id),
    execution_outcomes,
    strategy_lab_lanes,
    alpha_arrays,
    protected_validation,
    risk_covariance,
)
"""Every upgrade family a merged card declares, in its migration order: a family card adds its
factory here. The Task store, its telemetry and heartbeats move at first use by their own seams."""

State = Literal["PENDING", "RUNNING", "FAILED", "COMPLETED"]


def discover(workspace: Path, families: tuple[Family, ...]) -> tuple[Path, ...]:
    """The started workspace and its project's `workspaces/*`, those holding a family's old data."""
    started = workspace.resolve()
    found = {started}
    if started.parent.name == "workspaces":
        found |= {
            path.resolve()
            for path in started.parent.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and (path / RESEARCH_WORKSPACE_MANIFEST_NAME).is_file()
        }
    return tuple(sorted(path for path in found if any(f.holds(path) for f in families)))


@dataclass(frozen=True)
class Plan:
    """One upgrade run to acknowledge: its subject, workspaces, families and backups."""

    subject: str
    workspaces: tuple[Path, ...]
    families: tuple[str, ...]
    backups: tuple[dict[str, Any], ...]
    refusal: str | None = None
    recorded: bool = False

    @property
    def plan_hash(self) -> str:
        """What the person acknowledges, and what a resumed start shows again: the run, its
        workspaces and families, and the repository of each backup it keeps. The generation a
        backup gets once written, and the estimates, are left out of it."""
        kept = [{k: b.get(k) for k in ("kind", "workspace", "repository")} for b in self.backups]
        return str(
            canonical_hash(
                {
                    "subject": self.subject,
                    "workspaces": [str(path) for path in self.workspaces],
                    "families": list(self.families),
                    "backups": kept,
                }
            )
        )


def _refuse(error: OSError) -> None:
    raise error  # an unreadable directory refuses by name; it never reads as empty


def _bytes(folder: Path) -> int:
    return sum(
        os.stat(os.path.join(top, name)).st_size
        for top, _folders, files in os.walk(folder, onerror=_refuse)
        for name in files
    )


def _free(path: Path) -> int:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def _machine_backup(root: Path, written: int) -> dict[str, Any]:
    """The data root's latest generation if it holds the current head, else a new one's place."""
    repository = default_backup_root("machine")
    generations = repository / "generations"
    named = os.listdir(generations) if generations.is_dir() else []  # a partial is .<name>
    latest = max(
        (generations / n for n in named if not n.startswith(".") and (generations / n).is_dir()),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if latest is not None:
        try:
            verify_data_root(latest, source=root)
            return {"kind": "machine", "repository": str(repository), "generation": str(latest)}
        except DataError:
            pass
    estimate = _bytes(root)
    return {
        "kind": "machine",
        "repository": str(repository),
        "generation": None,
        "bytes": estimate,
        "free_after": _free(repository) - written - estimate,
    }


def _workspace_backup(workspace: Path, written: int) -> dict[str, Any]:
    """The workspace's latest generation if it holds the workspace now, else a new one's place."""
    backups = WorkspaceBackups(workspace, workspace_id=str(named_workspace_id(workspace)))
    row = {"kind": "workspace", "workspace": str(workspace), "repository": str(backups.root)}
    try:
        return row | {"generation": backups.verify(current=True).generation_hash}
    except WorkspaceBackupError:
        estimate = _bytes(workspace)
        return row | {
            "generation": None,
            "bytes": estimate,
            "free_after": _free(backups.root) - written - estimate,
        }


def plan(
    subject: str, workspaces: tuple[Path, ...], families: tuple[Family, ...], root: Path
) -> Plan:
    """A new run's plan; it refuses, before any write, a backup that would cross the free floor.

    Each estimate counts the bytes the plan's earlier backups write, as if on one volume."""
    backups: list[dict[str, Any]] = [_machine_backup(root, 0)]
    for workspace in workspaces:
        backups.append(_workspace_backup(workspace, sum(b.get("bytes", 0) for b in backups)))
    short = [
        b["repository"] for b in backups if b.get("free_after", FREE_FLOOR_BYTES) < FREE_FLOOR_BYTES
    ]
    refusal = (
        "Writing this upgrade's backups would leave less than "
        f"{FREE_FLOOR_BYTES >> 20} MiB free at {', '.join(short)}; free space there and "
        "start the Host again."
        if short
        else None
    )
    names = tuple(family.name for family in families)
    return Plan(subject, workspaces, names, tuple(backups), refusal)


def recorded(subject: str, started: dict[str, Any]) -> Plan:
    """A started run's plan, as its `started` progress recorded it."""
    rows, machine = started["backups"]["workspaces"], started["backups"]["machine"]["generation"]
    backups = (
        {
            "kind": "machine",
            "repository": str(Path(machine).parent.parent),  # <repository>/generations/<id>
            "generation": machine,
        },
        *(
            {
                "kind": "workspace",
                "workspace": r["workspace"],
                "repository": r["root"],
                "generation": r["generation"],
            }
            for r in rows
        ),
    )
    workspaces = tuple(Path(r["workspace"]) for r in rows)
    return Plan(subject, workspaces, tuple(started["families"]), backups, recorded=True)


class FirstStart:
    """The upgrade this Host start owes before it composes, if any: its plan, run and progress."""

    def __init__(self, root: Path, log: HostLog, workspace: Path) -> None:
        """Read what the started workspace's Host owes: a recorded run, a new plan, or none."""
        self.root, self.log, self.workspace = root.resolve(), log, workspace.resolve()
        self.families = tuple(make() for make in FAMILIES)
        self.failure: str | None = None
        self.plan = self._due()
        self.state: State = "PENDING"
        if self.plan is not None and self.plan.recorded:
            self.state = "RUNNING"  # acknowledged when it started: it resumes as recorded

    def _runner(
        self,
        subject: str,
        workspaces: tuple[Path, ...],
        check: Any = None,
        *,
        families: tuple[Family, ...] | None = None,
    ) -> UpgradeRunner:
        return UpgradeRunner(
            self.root,
            self.log,
            workspaces,
            self.families if families is None else families,
            check or (lambda *_: {}),
            subject=subject,
        )

    def _new_plan(self, subject: str, workspaces: tuple[Path, ...]) -> Plan:
        families = tuple(
            family
            for family in self.families
            if family.deferred is None or not any(map(family.deferred, workspaces))
        )
        return plan(subject, workspaces, families, self.root)

    def _deferred_plan(self, subject: str, earlier: dict[str, Any]) -> Plan | None:
        """An omitted family remains eligible after the workspace's other migrations finish."""
        for family in self.families:
            if family.deferred is None or family.name in earlier["families"]:
                continue
            own = f"{subject}/{family.name}"
            workspaces = (self.workspace,)
            follow = self._runner(own, workspaces, families=(family,))
            if follow.complete:
                continue
            if follow.started is not None:
                return recorded(own, follow.started)
            if family.holds(self.workspace):
                return plan(own, workspaces, (family,), self.root)
        return None

    def _due(self) -> Plan | None:
        machine = self._runner(TARGET, ())
        started = machine.started
        if not machine.complete:
            if started is not None:
                return recorded(TARGET, started)
            found = discover(self.workspace, self.families)
            return self._new_plan(TARGET, found) if found else None
        subject = f"{TARGET}/{named_workspace_id(self.workspace)}"
        if started is not None and any(
            self.workspace == Path(row["workspace"]) for row in started["backups"]["workspaces"]
        ):
            return self._deferred_plan(subject, started)
        alone = self._runner(subject, (self.workspace,))
        started = alone.started
        if started is not None:
            if alone.complete:
                return self._deferred_plan(subject, started)
            return recorded(subject, started)
        if not any(f.holds(self.workspace) for f in self.families):
            return None
        return self._new_plan(subject, (self.workspace,))

    def admit(self, plan_hash: object) -> None:
        """Take the acknowledgement of the plan shown; running it is the caller's next step."""
        if self.state != "PENDING" or self.plan is None:
            raise DataError(
                "The upgrade is already acknowledged; read its progress on /api/upgrade.",
                code="upgrade.acknowledged",
            )
        if plan_hash != self.plan.plan_hash:
            raise DataError(
                "The upgrade plan named is not the one shown now; read /api/upgrade again and "
                "acknowledge the plan it shows.",
                code="upgrade.plan_changed",
            )
        if self.plan.refusal is not None:
            raise DataError(self.plan.refusal, code="upgrade.backup_space")
        self.state = "RUNNING"

    def run(self) -> None:
        """Write and verify the plan's missing backups, then run its families, each workspace held
        by its writer lease. A refusal is kept for the progress and the Host stays restricted."""
        assert self.plan is not None and self.state == "RUNNING"
        try:
            with ExitStack() as leases:
                for workspace in self.plan.workspaces:
                    leases.callback(WorkspaceWriterLease.acquire(workspace).close)
                families = tuple(
                    family for family in self.families if family.name in self.plan.families
                )
                self._runner(
                    self.plan.subject, self.plan.workspaces, self._backups(), families=families
                ).run()
        except Exception as error:
            self.state, self.failure = "FAILED", refusal_reason(error)
            raise
        self.state = "COMPLETED"

    def _backups(self) -> Callable[..., dict[str, Any]]:
        """Each named backup, written when the plan has none yet; the machine's last, so the
        generation holds the head the run starts from."""
        assert self.plan is not None
        kept: list[WorkspaceBackups] = []
        machine: dict[str, Any] = {}
        for row in self.plan.backups:
            if row["kind"] == "machine":
                machine = row
                continue
            workspace = Path(row["workspace"])
            backups = WorkspaceBackups(
                workspace,
                workspace_id=str(named_workspace_id(workspace)),
                root=Path(row["repository"]),
            )
            if row["generation"] is None:
                backups.create(reason="REQUEST")
            kept.append(backups)
        self.log.flush()  # the generation holds every event the run starts from
        generation = (
            Path(machine["generation"])
            if machine["generation"] is not None
            else backup_data_root(self.root, Path(machine["repository"]))
        )
        return partial(
            verify_upgrade_backups, machine_generation=generation, workspace_backups=tuple(kept)
        )

    def view(self) -> dict[str, object]:
        """The plan and progress /api/upgrade answers, in the upgrade overview's form: the
        acknowledgement it offers carries the plan hash as its `upgrade_set_hash`."""
        assert self.plan is not None
        done, failed = [], None
        for (attrs,) in self.log.read(
            "SELECT attrs FROM event WHERE type = ? AND subject_id = ? ORDER BY seq",
            (PROGRESS, self.plan.subject),
        ):
            event = json.loads(attrs)
            if event["kind"] == "family_done":
                done.append(event["family"])
            elif event["kind"] == "family_failed":
                failed = event["family"]
        pending = self.state == "PENDING"
        return {
            "status": "UPGRADE_OVERVIEW",
            "installed": {"set_hash": self.plan.plan_hash, "identities": {}},
            "acknowledged": None,
            "moved": [],
            "show": pending,
            "counts": {},
            "studies": [],
            "reviews": [],
            "tasks": [],
            "first_start": {
                "state": self.state,
                "subject": self.plan.subject,
                "workspaces": [str(path) for path in self.plan.workspaces],
                "families": list(self.plan.families),
                "families_done": done,
                "family_failed": failed,
                "backups": list(self.plan.backups),
                "refusal": self.failure or self.plan.refusal,
            },
            "next_requests": (
                {
                    "acknowledge": {
                        "operation": "UPGRADE_ACKNOWLEDGE",
                        "upgrade_set_hash": self.plan.plan_hash,
                    }
                }
                if pending and self.plan.refusal is None
                else {}
            ),
        }


def refusal_reason(error: Exception) -> str:
    """A stopped upgrade's reason as /api/upgrade shows it: its owner's message, else its kind."""
    failure = getattr(error, "failure", None)
    return str(getattr(failure, "message", None) or type(error).__name__)
