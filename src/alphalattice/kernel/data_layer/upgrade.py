"""The data layer's ordered, verified family migration at the first 0.2.8 Host start.

The Host holds its ordinary writer fences throughout. A family owns its restart-safe step,
which reads the old data itself, and its ALL EQUAL comparison; each family has one reader, the
new store's. This module owns only their order and durable progress through HostLog. A crash
before family_done resumes that family. Nothing serves until every registered family is done.
Old data stays, and the pre-upgrade backup is the way back.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypedDict

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import canonical

if TYPE_CHECKING:
    from alphalattice.kernel.data_layer.log import HostLog

TARGET: Final = "0.2.8"
CLEAN: Final = f"{TARGET}/clean"
_WORKSPACE_PLANS: Final = f"{TARGET}/workspace/"
PROGRESS: Final = "upgrade.progress"
INPUT_FACTS: Final = "upgrade.input_facts"
PROGRESS_SPEC: Final = {
    "work": False,
    "subject": "upgrade",
    "required": ["kind"],
    "text": "none",
    "durable": True,
}
"""Durable facts, registered through the existing definition writer before the first move."""
INPUT_FACTS_SPEC: Final = {**PROGRESS_SPEC, "required": ["part"], "durable": False}
_FACT_PART_CHARACTERS: Final = 1800
"""An ASCII fragment can at most double when put in an attr string (log attrs allow 4096)."""


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


def workspace_plan_prefix(workspace: Path) -> str:
    """The scope of this physical folder's plans, including after a manifest replacement."""
    return f"{_WORKSPACE_PLANS}{canonical(str(workspace)).decode()}/"


def _fact_subject(folder: str, family: str) -> str:
    return canonical([TARGET, folder, family]).decode()


def start_records(
    log: HostLog,
    workspace: Path,
    candidates: tuple[Path, ...],
    families: Sequence[str],
    legacy_id: str | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], InputStamp]]:
    """Read current decisions and pending plans with the log's type/subject and sequence indexes.

    Each family has its own subject. Its newest row points at the beginning of the last
    atomically written set of facts, so even the SQL range read excludes superseded bodies.
    Only the machine's unfinished plan can name workspaces beyond the one being opened.
    A folder's exact plan subject points to its unfinished start row or records completion.
    Older versions used one fixed subject per workspace and deferred family, also exact
    lookups. Keep fact heads as the outer join loop so the log uses their row IDs.
    """
    prefix = workspace_plan_prefix(workspace)
    legacy = f"{TARGET}/{legacy_id}"
    rows = log.read(
        """
        WITH machine AS (
            SELECT seq, attrs FROM event
            WHERE type=?1 AND subject_id=?3
                AND json_extract(attrs,'$.kind') IN ('started','completed')
            ORDER BY seq DESC LIMIT 1
        ), own AS (
            SELECT json_extract(attrs,'$.started_seq') AS started_seq FROM event
            WHERE type=?1 AND subject_id=?4 ORDER BY event.seq DESC LIMIT 1
        ), legacy_subjects AS (
            SELECT ?5 AS subject_id
            UNION ALL SELECT ?5 || '/' || value FROM json_each(?8)
        ), legacy AS (
            SELECT status.seq FROM legacy_subjects
            CROSS JOIN event status ON status.seq=(
                SELECT seq FROM event WHERE type=?1 AND subject_id=legacy_subjects.subject_id
                    AND json_extract(attrs,'$.kind') IN ('started','completed')
                ORDER BY seq DESC LIMIT 1
            )
            WHERE json_extract(status.attrs,'$.kind')='started'
                AND EXISTS (SELECT 1 FROM json_each(status.attrs,'$.backups.workspaces')
                    WHERE json_extract(value,'$.workspace')=?6)
        ), folders AS (
            SELECT value AS path FROM json_each(?7)
                WHERE value=?6 OR NOT EXISTS (
                    SELECT 1 FROM machine WHERE json_extract(attrs,'$.kind')='completed'
                )
            UNION
            SELECT json_extract(value,'$.workspace')
                FROM machine, json_each(machine.attrs,'$.backups.workspaces')
                WHERE json_extract(machine.attrs,'$.kind')='started'
        ), fact_heads AS MATERIALIZED (
            SELECT (SELECT seq FROM event
                WHERE type=?2 AND subject_id=json_array(?3,folders.path,names.value)
                ORDER BY seq DESC LIMIT 1) AS seq
            FROM folders, json_each(?8) names
        )
        SELECT type,subject_id,attrs,seq FROM event WHERE seq IN (
            SELECT seq FROM machine
            UNION ALL SELECT started_seq FROM own
            UNION ALL SELECT seq FROM legacy
        )
        UNION ALL
        SELECT part.type,part.subject_id,part.attrs,part.seq FROM fact_heads
            CROSS JOIN event head ON head.seq=fact_heads.seq
            CROSS JOIN event part ON part.seq
                BETWEEN coalesce(json_extract(head.attrs,'$.first'),head.seq) AND head.seq
                AND part.type=?2 AND part.subject_id=head.subject_id
        ORDER BY seq
        """,
        (
            PROGRESS,
            INPUT_FACTS,
            TARGET,
            prefix[:-1],
            legacy,
            str(workspace),
            canonical([str(path) for path in candidates]).decode(),
            canonical(list(families)).decode(),
        ),
    )
    records: dict[str, list[dict[str, Any]]] = {}
    parts: dict[str, list[str]] = {}
    for type_, subject, attrs, _seq in rows:
        event = json.loads(attrs)
        if type_ == PROGRESS:
            records.setdefault(subject, []).append(event)
        else:
            parts.setdefault(subject, []).append(event["part"])
    facts = {}
    for subject, fragments in parts.items():
        _, folder, family = json.loads(subject)
        facts[folder, family] = json.loads("".join(fragments))
    return records, facts


class InputFileStamp(TypedDict):
    """Identity and metadata of one declared file, with a digest for shared archival bytes."""

    linked: bool
    metadata: list[int]
    content: str | None


class InputStamp(TypedDict):
    """The exact declaration and workspace identity, then the declared files by path."""

    basis: list[object]
    files: dict[str, InputFileStamp]


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
    trust_paths: tuple[str, ...] = ()
    """Legacy inputs a first-start decision covers, relative to its workspace."""
    shared_inputs: tuple[str, ...] = ()
    """Declared files whose bytes can be vouched for after a hard link changes their stamp."""


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
        records: Sequence[Mapping[str, Any]] | None = None,
        completion_inputs: Callable[[Family, tuple[Path, ...]], dict[str, InputStamp]]
        | None = None,
        settled_inputs: Callable[
            [tuple[Family, ...], tuple[Path, ...], Mapping[str, dict[str, InputStamp]]],
            dict[str, dict[str, InputStamp]],
        ]
        | None = None,
    ) -> None:
        """Read this upgrade's indexed progress once; the caller retains the writer's lifetime.

        `subject` names one run's progress: the machine's first start is TARGET, and a workspace
        that first appears after it completed runs the same families alone under
        `TARGET/<workspace_id>`, so the machine run's recorded plan never changes."""
        self.root, self.log, self.workspaces = root, log, workspaces
        self.subject = subject
        self.families, self.backup_check = families, backup_check
        self._completion_inputs = completion_inputs
        self._settled_inputs = settled_inputs
        self._ids: dict[tuple[str, str], str] = {}
        self._done: set[str] = set()
        self._verified_inputs: dict[str, dict[str, InputStamp]] = {}
        self._plan: list[str] | None = None
        self._backups: dict[str, Any] | None = None
        self._complete = False
        self._none_due: set[tuple[str, str]] = set()
        if records is None:
            records = [
                json.loads(row[0])
                for row in log.read(
                    "SELECT attrs FROM event WHERE type = ? AND subject_id = ? ORDER BY seq",
                    (PROGRESS, subject),
                )
            ]
        for event in records:
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

    def run(self, *, began: float | None = None) -> bool:
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
                facts = {}
                if self._completion_inputs is not None:
                    inputs = self._verified_inputs[family.name] = self._completion_inputs(
                        family, self.workspaces
                    )
                    facts = {(folder, family.name): stamp for folder, stamp in inputs.items()}
                self._record_facts("family_done", facts, family=family.name, verdict="ALL EQUAL")
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
        final = (
            self._settled_inputs(self.families, self.workspaces, self._verified_inputs)
            if self._settled_inputs is not None
            else {}
        )
        settled = {
            (folder, name): stamp
            for folder, inputs in final.items()
            for name, stamp in inputs.items()
        }
        timing = {"upgrade_seconds": perf_counter() - began} if began is not None else {}
        self._record_facts("completed", settled, **timing)
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

    def record_none_due(
        self, workspace: Path, examined: str, *, inputs: Mapping[str, InputStamp] | None = None
    ) -> None:
        """Record that the workspace folder held no family's old data in the state `examined`:
        only an older version writes old data, and its way back is a restored backup."""
        if inputs is None and self.none_due(workspace, examined):
            return
        self._register()
        folder = str(workspace.resolve())
        facts = {(folder, name): stamp for name, stamp in (inputs or {}).items()}
        self._record_facts("none_due", facts, folder=folder, examined=examined)
        self._none_due.add((folder, examined))

    def _record_facts(
        self, kind: str, facts: Mapping[tuple[str, str], InputStamp], **attrs: object
    ) -> None:
        if not facts:
            self._record(kind, **attrs)
            return
        self._register(INPUT_FACTS)
        with self.log.unit():
            for (folder, family), stamp in facts.items():
                subject = ("upgrade", _fact_subject(folder, family))
                encoded = json.dumps(
                    stamp, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                )
                first = None
                for offset in range(0, len(encoded), _FACT_PART_CHARACTERS):
                    seq = self.log.record(
                        INPUT_FACTS,
                        subject,
                        {"part": encoded[offset : offset + _FACT_PART_CHARACTERS], "first": first},
                        actor=("host", "upgrade"),
                        node="upgrade",
                    )
                    if first is None:
                        first = seq
            self._record(kind, **attrs)

    def _record(self, kind: str, **attrs: object) -> None:
        # A start and its folder's pointer commit together. Completion clears the pointer
        # in the same transaction, including any final input decisions that commit with it.
        scope = (
            self.subject.rpartition("/")[0]
            if kind in {"started", "completed"} and self.subject.startswith(_WORKSPACE_PLANS)
            else None
        )
        with self.log.unit() if scope is not None else nullcontext():
            seq = self.log.record(
                PROGRESS,
                ("upgrade", self.subject),
                {"kind": kind, **attrs},
                actor=("host", "upgrade"),
                node="upgrade",
            )
            if scope is not None:
                self.log.record(
                    PROGRESS,
                    ("upgrade", scope),
                    {"kind": "plan_state", "started_seq": seq if kind == "started" else None},
                    actor=("host", "upgrade"),
                    node="upgrade",
                )

    def _register(self, name: str = PROGRESS) -> None:
        """Admit the upgrade's events through definition.registered, even on an older root."""
        if self.log.read(
            "SELECT 1 FROM definition WHERE family = 'event_type' AND name = ? AND version = 1",
            (name,),
        ):
            return
        spec, meaning = (
            (PROGRESS_SPEC, "First 0.2.8 upgrade progress and its immutable ID mapping.")
            if name == PROGRESS
            else (INPUT_FACTS_SPEC, "Bounded input facts of a verified upgrade decision.")
        )
        definition = {"family": "event_type", "name": name, "version": 1}
        self.log.record(
            "definition.registered",
            ("definition", f"event_type/{name}"),
            definition,
            text=canonical(
                {**definition, "status": "standard", "spec": spec, "text": meaning}
            ).decode(),
            actor=("host", "upgrade"),
            node="upgrade",
        )
