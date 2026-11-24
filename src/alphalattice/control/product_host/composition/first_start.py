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
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from functools import cached_property, partial
from importlib import import_module, resources
from importlib.resources.abc import Traversable
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Final, Literal

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
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
from alphalattice.control.workspace_runtime.content_store import ContentAddressedStoreError
from alphalattice.control.workspace_runtime.storage.capacity import FREE_FLOOR_BYTES
from alphalattice.control.workspace_runtime.writer_lease import WorkspaceWriterLease
from alphalattice.foundation.data_layer.backup import backup_data_root, verify_data_root
from alphalattice.foundation.data_layer.log import HostLog, uuid7
from alphalattice.foundation.data_layer.upgrade import (
    CLEAN,
    PROGRESS,
    TARGET,
    Family,
    InputStamp,
    UpgradeRunner,
    start_records,
    workspace_files,
    workspace_plan_prefix,
)
from alphalattice.kernel.identity import Stamp, file_stamp, handle_stamp, integrity_sha256_stream

UNHELD: Final = ("data_layer.outcome_missing",)
"""The refusal that says only that the catalog holds no body for a chunk: no profile for its
digest, or one without rows. An unavailable reader or any other catalog fault is not evidence
of absence, so it stops the start by its own name."""


def execution_outcomes() -> Family:
    """The execution-outcome family, holding a workspace whose committed legacy chunks the catalog
    lacks or disagrees with: its own ALL EQUAL check, asked of that workspace alone. Its module
    stays as merged, so no study's implementation identity moves for this question."""
    from alphalattice.foundation.causal_outcomes.execution.readers import (  # lazy: on use
        execution_outcome_upgrade_family,
    )

    family = execution_outcome_upgrade_family()

    def holds(workspace: Path) -> bool:
        try:
            return family.verify(SimpleNamespace(workspaces=(workspace,))) is not True
        except DataError as error:  # a chunk the catalog does not hold yet; any other fault stops
            if error.failure.code not in UNHELD:
                raise
            return True

    return replace(family, holds=holds)


def _portfolio_holds(family: Family, workspace: Path) -> bool:
    """Ask the family's catalog comparison about one workspace."""
    try:
        return family.verify(SimpleNamespace(workspaces=(workspace,))) is not True
    except (DataError, ContentAddressedStoreError) as error:
        missing = error.__cause__ if isinstance(error, ContentAddressedStoreError) else error
        if (
            not isinstance(missing, DataError)
            or missing.failure.code != "data_layer.benchmark_missing"
        ):
            raise
        return True


def _host_portfolio_family(family: Family) -> Family:
    """A Portfolio family whose workspaces are judged by its own catalog comparison."""
    return replace(family, holds=partial(_portfolio_holds, family))


def _declared_family(declaration: dict[str, str]) -> Callable[[], Family]:
    """Bind one registered owner to the Host's workspace and verification adapters."""
    target = declaration["factory"]
    if target == ":execution_outcomes":
        return execution_outcomes
    module, _, name = target.partition(":")
    if not module.startswith("alphalattice.") or not name.isidentifier():
        raise ValueError(f"invalid first-start factory {target}")
    factory = getattr(import_module(module), name)
    mode = declaration["mode"]
    if mode == "direct":
        return factory
    if mode == "machine-backup":
        return lambda: factory(default_backup_root("machine"))
    scoped = partial(factory, named_workspace_id)
    if mode == "workspace":
        return scoped
    if mode == "portfolio":
        return lambda: _host_portfolio_family(scoped())
    raise ValueError(f"unknown first-start adapter {mode} for {name}")


@dataclass(frozen=True)
class _Registration:
    """A family's input declaration can be read without loading its migration."""

    name: str
    trust_paths: tuple[str, ...]
    shared_inputs: tuple[str, ...]
    declaration: dict[str, str]

    def __call__(self) -> Family:
        try:
            family = _declared_family(self.declaration)()
            if family.name != self.name:
                raise ValueError("the declared name differs from the family owner")
        except (AttributeError, ImportError, KeyError, TypeError, ValueError) as error:
            raise DataError(
                f"cannot resolve the first-start family {self.name}",
                code="data_layer.schema_mismatch",
            ) from error
        return replace(family, trust_paths=self.trust_paths, shared_inputs=self.shared_inputs)


def _registration(path: Traversable) -> _Registration:
    """Read one installed family declaration."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        paths, shared = row["trust_paths"], row.get("shared_inputs", [])
        for patterns in (paths, shared):
            if not isinstance(patterns, list) or not all(
                isinstance(pattern, str) and pattern.startswith(("/", "*/")) for pattern in patterns
            ):
                raise ValueError("invalid family trust paths")
        return _Registration(
            row["name"],
            tuple(paths),
            tuple(shared),
            {"factory": row["factory"], "mode": row["mode"]},
        )
    except (AttributeError, ImportError, KeyError, OSError, TypeError, ValueError) as error:
        raise DataError(
            f"cannot read first-start family registration {path.name}",
            code="data_layer.schema_mismatch",
        ) from error


def _declarations() -> list[Traversable]:
    """The registered family declarations in their numbered migration order."""
    directory = resources.files(__package__).joinpath("first_start_families")
    return [
        path
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.name.endswith(".json")
    ]


_DECLARATION_FILES: Final = _declarations()
FAMILIES: Final = tuple(_registration(path) for path in _DECLARATION_FILES)
"""Every upgrade family in migration order. Research-input copies go last, after the families
that read them. The Task store and its telemetry move at first use by their own seams."""

DECLARATIONS: Final = " ".join(path.name for path in _DECLARATION_FILES)
"""The installed declarations in order. Changed declarations need new trust decisions."""


def _examined(workspace: Path) -> list[object]:
    """Use the workspace ID and the folder identity for restores and manifest-less opens."""
    folder = workspace.stat()
    return [named_workspace_id(workspace), folder.st_dev, folder.st_ino]


def _shared_content(path: Path, stamp: Stamp) -> str:
    """A hard link can change the file's change clock without touching its bytes."""
    with path.open("rb") as stream:
        digest = integrity_sha256_stream(stream).hexdigest()
        if handle_stamp(stream) != stamp:
            raise OSError(f"Upgrade input changed while being read: {path}")
    return digest


def _input_stamps(
    workspace: Path,
    owners: Mapping[str, _Registration | Family],
    previous: Mapping[str, InputStamp],
) -> dict[str, InputStamp]:
    """Inventory the declared files once, using the last digest for unchanged shared files."""
    examined = _examined(workspace)
    found: dict[str, InputStamp] = {
        name: {
            "basis": [
                DECLARATIONS,
                owner.declaration if isinstance(owner, _Registration) else None,
                list(owner.trust_paths),
                list(owner.shared_inputs),
                examined,
            ],
            "files": {},
        }
        for name, owner in owners.items()
    }
    if not any(owner.trust_paths for owner in owners.values()):
        return found
    for path in workspace_files(workspace):
        key = "/" + path.relative_to(workspace).as_posix()
        matching = [
            name
            for name, owner in owners.items()
            if any(fnmatchcase(key, pattern) for pattern in owner.trust_paths)
        ]
        if not matching:
            continue
        stamp, linked = file_stamp(path), path.is_symlink()
        metadata = list(stamp)
        for name in matching:
            content = None
            if any(fnmatchcase(key, pattern) for pattern in owners[name].shared_inputs):
                entry = previous.get(name)
                prior = entry["files"].get(key) if entry is not None else None
                if (
                    prior is not None
                    and prior["linked"] == linked
                    and prior["metadata"] == metadata
                ):
                    content = prior["content"]
                if content is None:
                    content = _shared_content(path, stamp)
            found[name]["files"][key] = {"linked": linked, "metadata": metadata, "content": content}
    return found


def _same_inputs(previous: InputStamp | None, current: InputStamp) -> bool:
    if previous is None or previous["basis"] != current["basis"]:
        return False
    older, files = previous["files"], current["files"]
    if older.keys() != files.keys():
        return False
    for path, stamp in files.items():
        prior = older[path]
        if prior == stamp:
            continue
        if (
            stamp["content"] is None
            or prior["content"] != stamp["content"]
            or prior["linked"] != stamp["linked"]
            or prior["metadata"][:-1] != stamp["metadata"][:-1]
        ):
            return False
    return True


State = Literal["PENDING", "RUNNING", "FAILED", "COMPLETED"]


def candidates(workspace: Path) -> tuple[Path, ...]:
    """The started workspace and its project's `workspaces/*`, whose old data a plan may name."""
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
    return tuple(sorted(found))


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
        self.failure: str | None = None
        self._candidates = candidates(self.workspace)
        self._records, self._trusted = start_records(
            log,
            self.workspace,
            self._candidates,
            tuple(self._owners),
            named_workspace_id(self.workspace),
        )
        self._loaded: dict[str, Family] = {}
        self.plan = self._due()
        self.state: State = "PENDING"
        if self.plan is not None and self.plan.recorded:
            self.state = "RUNNING"  # acknowledged when it started: it resumes as recorded

    @cached_property
    def _owners(self) -> dict[str, _Registration | Family]:
        """Installed signatures stay available even when their owners need not be imported."""
        owners = [make if isinstance(make, _Registration) else make() for make in FAMILIES]
        return {owner.name: owner for owner in owners}

    def _family(self, name: str) -> Family:
        if name not in self._loaded:
            owner = self._owners[name]
            self._loaded[name] = owner() if isinstance(owner, _Registration) else owner
        return self._loaded[name]

    @property
    def families(self) -> tuple[Family, ...]:
        """Open the upgrade families when the recorded progress leaves work to inspect or run."""
        return tuple(self._family(name) for name in self._owners)

    def _runner(
        self,
        subject: str,
        workspaces: tuple[Path, ...],
        check: Any = None,
        *,
        families: tuple[Family, ...] = (),
        startup: bool = False,
    ) -> UpgradeRunner:
        return UpgradeRunner(
            self.root,
            self.log,
            workspaces,
            families,
            check or (lambda *_: {}),
            subject=subject,
            records=self._records.get(subject, ()) if startup else None,
            completion_inputs=self._completion_inputs if not startup else None,
            settled_inputs=self._settled_inputs if not startup else None,
        )

    def _new_plan(self, subject: str, workspaces: tuple[Path, ...]) -> Plan:
        families = tuple(
            family
            for family in self.families
            if family.deferred is None or not any(map(family.deferred, workspaces))
        )
        return plan(subject, workspaces, families, self.root)

    def _completion_inputs(
        self, family: Family, workspaces: tuple[Path, ...]
    ) -> dict[str, InputStamp]:
        """Seal what the family just verified under the workspace writer leases."""
        declarations = {family.name: self._owners[family.name]}
        return {
            str(workspace): _input_stamps(workspace, declarations, {})[family.name]
            for workspace in workspaces
        }

    def _settled_inputs(
        self,
        families: tuple[Family, ...],
        workspaces: tuple[Path, ...],
        verified: Mapping[str, dict[str, InputStamp]],
    ) -> dict[str, dict[str, InputStamp]]:
        """Recheck any verified input another family changed during the same upgrade."""
        declarations = {family.name: self._owners[family.name] for family in families}
        settled: dict[str, dict[str, InputStamp]] = {}
        for workspace in workspaces:
            folder = str(workspace)
            previous = {
                name: self._trusted[folder, name]
                for name in declarations
                if (folder, name) in self._trusted
            }
            previous.update(
                {name: stamps[folder] for name, stamps in verified.items() if folder in stamps}
            )
            stamps = _input_stamps(workspace, declarations, previous)
            for family in families:
                stamp = stamps[family.name]
                prior = previous.get(family.name)
                if prior == stamp:
                    continue
                if family.deferred is not None and family.deferred(workspace):
                    continue
                if not _same_inputs(prior, stamp) and family.holds(workspace):
                    raise DataError(
                        f"Upgrade family {family.name} needs work after the recorded verification; "
                        "restore the verified inputs before resuming.",
                        code="upgrade.refused",
                    )
                settled.setdefault(folder, {})[family.name] = stamp
        return settled

    def _due(self) -> Plan | None:
        machine = self._runner(TARGET, (), startup=True)
        started = machine.started
        if started is not None and not machine.complete:
            return recorded(TARGET, started)
        if machine.complete:
            for subject in self._records:
                if subject == TARGET:
                    continue
                other = self._runner(subject, (self.workspace,), startup=True)
                pending = other.started
                if (
                    pending is not None
                    and not other.complete
                    and any(
                        self.workspace == Path(row["workspace"])
                        for row in pending["backups"]["workspaces"]
                    )
                ):
                    return recorded(subject, pending)

        holding: dict[Path, list[Family]] = {}
        initial: set[Path] = set()
        checked: dict[Path, dict[str, InputStamp]] = {}
        workspaces = (self.workspace,) if machine.complete else self._candidates
        for workspace in workspaces:
            previous = {
                name: self._trusted[str(workspace), name]
                for name in self._owners
                if (str(workspace), name) in self._trusted
            }
            if not machine.complete or not previous:
                initial.add(workspace)
            stamps = _input_stamps(workspace, self._owners, previous)
            for name, stamp in stamps.items():
                prior = previous.get(name)
                if prior == stamp:
                    continue
                if _same_inputs(prior, stamp):
                    checked.setdefault(workspace, {})[name] = stamp
                    continue
                family = self._family(name)
                if family.deferred is not None and family.deferred(workspace):
                    continue
                if family.holds(workspace):
                    holding.setdefault(workspace, []).append(family)
                    if workspace in initial:
                        break  # the first upgrade runs all eligible families in their order
                else:
                    checked.setdefault(workspace, {})[name] = stamp
        if holding:
            if not machine.complete:
                return self._new_plan(TARGET, tuple(holding))
            families = tuple(holding[self.workspace])
            subject = f"{workspace_plan_prefix(self.workspace)}{uuid7()}"
            if self.workspace in initial:
                return self._new_plan(subject, (self.workspace,))
            return plan(subject, (self.workspace,), families, self.root)
        if checked:
            recorder = self._runner(CLEAN, (), startup=True)
            for workspace, stamps in checked.items():
                recorder.record_none_due(workspace, DECLARATIONS, inputs=stamps)
        return None

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
        began = perf_counter()
        try:
            with ExitStack() as leases:
                for workspace in self.plan.workspaces:
                    leases.callback(WorkspaceWriterLease.acquire(workspace).close)
                families = tuple(
                    family for family in self.families if family.name in self.plan.families
                )
                runner = self._runner(
                    self.plan.subject, self.plan.workspaces, self._backups(), families=families
                )
                runner.run(began=began)
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
        done, failed, upgrade_seconds = [], None, None
        for (attrs,) in self.log.read(
            "SELECT attrs FROM event WHERE type = ? AND subject_id = ? ORDER BY seq",
            (PROGRESS, self.plan.subject),
        ):
            event = json.loads(attrs)
            if event["kind"] == "family_done":
                done.append(event["family"])
            elif event["kind"] == "family_failed":
                failed = event["family"]
            elif event["kind"] == "completed":
                upgrade_seconds = event.get("upgrade_seconds")
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
                "upgrade_seconds": upgrade_seconds,
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
