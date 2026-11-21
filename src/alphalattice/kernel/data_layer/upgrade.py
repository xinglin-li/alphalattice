"""The data layer's ordered, verified family migration at the first 0.2.8 Host start.

The Host holds its ordinary writer fences throughout. A family owns its restart-safe step,
which reads the old data itself, and its ALL EQUAL comparison; each family has one reader, the
new store's. This module owns only their order and durable progress through HostLog. A crash
before family_done resumes that family. Nothing serves until every registered family is done.
Old data stays, and the pre-upgrade backup is the way back.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import canonical

if TYPE_CHECKING:
    from alphalattice.kernel.data_layer.log import HostLog

TARGET: Final = "0.2.8"
PROGRESS: Final = "upgrade.progress"
PROGRESS_SPEC: Final = {
    "work": False,
    "subject": "upgrade",
    "required": ["kind"],
    "text": "none",
    "durable": True,
}
"""Durable facts, registered through the existing definition writer before the first move."""


def workspace_files(workspace: Path) -> Iterator[Path]:
    """Enumerate a workspace without suppressing unreadable directories or following aliases."""

    def unreadable(error: OSError) -> None:
        raise error

    for parent, directories, files in workspace.walk(on_error=unreadable):
        directories.sort()
        for name in directories:
            path = parent / name
            if path.is_symlink():
                raise ValueError(f"workspace directory is an alias: {path}")
        for name in sorted(files):
            yield parent / name


@dataclass(frozen=True)
class Family:
    """One declared family: its step is restart-safe and reads the old data itself.

    A verify returns True only after comparing all of this family's old and new reads. The
    tuple's order is the declared migration order. A step writes IDs with runner.map_id; its
    repeated call must retain the same mapping and finish its own incomplete writes.
    """

    name: str
    step: Callable[[UpgradeRunner], None]
    verify: Callable[[UpgradeRunner], bool]
    holds: Callable[[Path], bool] = lambda _workspace: False
    """Whether a workspace holds old data this family's new store lacks: only such a workspace is
    upgraded, so a copy of an upgraded one is not."""
    deferred: Callable[[Path], bool] | None = None
    """Whether this family's old data must stay readable while its admission authority is absent."""


class UpgradeRunner:
    """Resume the first unfinished family under the caller's open, exclusive HostLog.

    backup_check returns the machine/workspace owners' verified backup authority. The first
    start records it; a resume checks those same generations, not an arbitrary historical one.
    The backup owners supply the production adapter; fixture families supply it in tests.
    """

    def __init__(
        self,
        root: Path,
        log: HostLog,
        workspaces: tuple[Path, ...],
        families: tuple[Family, ...],
        backup_check: Callable[[Path, tuple[Path, ...], dict[str, Any] | None], dict[str, Any]],
        *,
        subject: str = TARGET,
    ) -> None:
        """Read this upgrade's indexed progress once; the caller retains the writer's lifetime.

        `subject` names one run's progress: the machine's first start is TARGET, and a workspace
        that first appears after it completed runs the same families alone under
        `TARGET/<workspace_id>`, so the machine run's recorded plan never changes."""
        self.root, self.log, self.workspaces = root, log, workspaces
        self.subject = subject
        self.families, self.backup_check = families, backup_check
        self._ids: dict[tuple[str, str], str] = {}
        self._done: set[str] = set()
        self._plan: list[str] | None = None
        self._backups: dict[str, Any] | None = None
        self._complete = False
        self._none_due: set[tuple[str, str]] = set()
        for row in log.read(
            "SELECT attrs FROM event WHERE type = ? AND subject_id = ? ORDER BY seq",
            (PROGRESS, subject),
        ):
            event = json.loads(row[0])
            if event["kind"] == "started":
                self._plan = event["families"]
                self._backups = event.get("backups")
            elif event["kind"] == "family_done":
                self._done.add(event["family"])
            elif event["kind"] == "id_mapped":
                self._ids[event["family"], event["old_id"]] = event["new_id"]
            elif event["kind"] == "completed":
                self._complete = True
            elif event["kind"] == "none_due":
                self._none_due.add((event["folder"], event["examined"]))

    def run(self) -> bool:
        """Verify backups, migrate and verify in order, then permit the Host to serve.

        False means the recorded run was already complete: its steps and comparisons are
        not rerun. A failed verification ends this start; a later explicit start resumes its
        first not-done family.
        """
        names = [family.name for family in self.families]
        if len(set(names)) != len(names) or (self._plan is not None and self._plan != names):
            raise DataError(
                "The upgrade's declared family order differs from its recorded start; "
                "restore that registration before resuming.",
                code="upgrade.refused",
            )
        if self._complete:
            return False
        if self._plan is not None and self._backups is None:
            raise DataError(
                "The upgrade lacks its recorded pre-upgrade backup authority; "
                "recover its original progress record before resuming.",
                code="upgrade.refused",
            )
        verified: dict[str, Any] | None
        try:
            self.log.flush()
            verified = self.backup_check(self.root, self.workspaces, self._backups)
        except Exception:
            verified = None
        if (
            not isinstance(verified, dict)
            or not verified
            or (self._backups is not None and verified != self._backups)
        ):
            raise DataError(
                "Verified pre-upgrade backups of the data root and every workspace are "
                "required; before the first start create fresh backups, and on resume "
                "use the recorded pre-upgrade generations.",
                code="upgrade.refused",
            )
        self._register()
        if self._plan is None:
            self._record("started", families=names, backups=verified)
            self._plan = names
            self._backups = verified
        for family in self.families:
            if family.name in self._done:
                continue
            point = "start record"
            try:
                self._record("family_started", family=family.name)
                point = "step"
                family.step(self)
                point = "ALL EQUAL verification"
                if family.verify(self) is not True:
                    raise DataError(
                        "Family verification did not return ALL EQUAL.", code="upgrade.refused"
                    )
                point = "completion record"
                self._record("family_done", family=family.name, verdict="ALL EQUAL")
                self._done.add(family.name)
            except Exception:
                try:
                    self._record("family_failed", family=family.name, point=point)
                finally:
                    # Only the declared family and stage cross this diagnostic boundary;
                    # a step's provider data, exception body and environment never do.
                    raise DataError(
                        f"Upgrade family {family.name} stopped at {point}; the Host will not "
                        "serve. Repair the named family before starting again.",
                        code="upgrade.refused",
                    ) from None
        self._record("completed")
        self._complete = True
        return True

    @property
    def complete(self) -> bool:
        """Whether this run's `completed` is recorded."""
        return self._complete

    @property
    def started(self) -> dict[str, Any] | None:
        """This run's recorded plan (families and backups), None before its first start."""
        if self._plan is None:
            return None
        return {"families": list(self._plan), "backups": self._backups}

    def map_id(self, family: str, old_id: str, new_id: str) -> str:
        """Keep one immutable old-to-new mapping; a repeated step reuses its existing entry."""
        key = (family, old_id)
        if key in self._ids:
            if self._ids[key] != new_id:
                raise DataError(
                    f"Upgrade family {family} tried to replace a recorded ID mapping; "
                    "restore its original mapping before resuming.",
                    code="upgrade.refused",
                )
            return self._ids[key]
        self._record("id_mapped", family=family, old_id=old_id, new_id=new_id)
        self._ids[key] = new_id
        return new_id

    @property
    def id_map(self) -> Mapping[tuple[str, str], str]:
        """The one event-backed mapping, read-only; family steps write through map_id."""
        return MappingProxyType(self._ids)

    def none_due(self, workspace: Path, examined: str) -> bool:
        """Whether this subject recorded that the workspace folder held no family's old data
        in the state `examined` names (the caller's: what the families asked and what they
        asked of); another state of the folder is asked again."""
        return (str(workspace.resolve()), examined) in self._none_due

    def record_none_due(self, workspace: Path, examined: str) -> None:
        """Record that the workspace folder held no family's old data in the state `examined`:
        only an older version writes old data, and its way back is a restored backup."""
        if self.none_due(workspace, examined):
            return
        self._register()
        folder = str(workspace.resolve())
        self._record("none_due", folder=folder, examined=examined)
        self._none_due.add((folder, examined))

    def _record(self, kind: str, **attrs: object) -> None:
        """Append progress through the one log writer; its durable type commits on return."""
        self.log.record(
            PROGRESS,
            ("upgrade", self.subject),
            {"kind": kind, **attrs},
            actor=("host", "upgrade"),
            node="upgrade",
        )

    def _register(self) -> None:
        """Admit the progress type through definition.registered, including an older data root."""
        if self.log.read(
            "SELECT 1 FROM definition WHERE family = 'event_type' AND name = ? AND version = 1",
            (PROGRESS,),
        ):
            return
        definition = {"family": "event_type", "name": PROGRESS, "version": 1}
        self.log.record(
            "definition.registered",
            ("definition", f"event_type/{PROGRESS}"),
            definition,
            text=canonical(
                {
                    **definition,
                    "status": "standard",
                    "spec": PROGRESS_SPEC,
                    "text": "First 0.2.8 upgrade progress and its immutable ID mapping.",
                }
            ).decode(),
            actor=("host", "upgrade"),
            node="upgrade",
        )
