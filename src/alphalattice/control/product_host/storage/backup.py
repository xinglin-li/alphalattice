"""A workspace's held state, backed up outside it and restored onto a new directory (ST).

The held set is what the storage map says nothing else rebuilds. The market store's revision
journals, universe history, provider attempts, audit receipts and admission table, and the
workspace's Tasks, are exported table by table as Parquet: the market tables through the owner's
shared database instance, the Tasks from the machine's log in the shape a 0.2.7 Task Control store
held them, which a restore writes back as such a store for the workspace's first open to move onto
the log it holds; the Evidence and CRO records, the research programs and evidence, the Host's cases
and records and the manifest are copied as they are; the research inputs and the installed
authority packages are listed by digest only, the first being the size of the store and the
second rebuilt by its installation. Everything else (the Feature values, the Panels, the
outcomes, the index, the model copy, the projections) is rebuilt from these.

A generation is a manifest of digests over a content-addressed store outside the workspace,
`%LOCALAPPDATA%/AlphaLattice/backups/<workspace_id>` unless `ALPHALATTICE_BACKUP_ROOT` names
another root; the newest seven are kept unless the request keeps another count (an execution
parameter), and an object no kept generation names is removed. A restore writes one generation
onto a new directory: the copied files where they were, the Task Control store created by its
owner and filled from its tables, and the market store's tables as Parquet under `held-state/`
for the market store's rebuild to take; the listed entries are named, not written.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing, nullcontext
from datetime import datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal, Self

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceError,
    read_research_workspace_manifest,
)
from alphalattice.control.task_control.registry import TASK_CONTROL_DATABASE_FILENAME
from alphalattice.control.task_control.upgrade import create_0_2_7_store, task_rows
from alphalattice.control.workspace_runtime.content_store import staging_directory
from alphalattice.control.workspace_runtime.database import (
    WorkspaceDatabase,
    hold_workspace_database,
    open_workspace_database,
)
from alphalattice.control.workspace_runtime.lock import LOCK_SUFFIX
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.paths import resolve_confined
from alphalattice.evidence.alternative_evidence.runtime.execution import below_normal_thread
from alphalattice.evidence.alternative_evidence.storage.world import carry_originals
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    create_market_profile,
    market_format,
)
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    carry_risk_covariance,
)
from alphalattice.kernel.clock import utc_now
from alphalattice.kernel.config import current as process_config
from alphalattice.kernel.data_layer import arrays, history
from alphalattice.kernel.door import DECISIONS_DATABASE
from alphalattice.kernel.identity import integrity_sha256
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceSecurityError
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.spans import span

MARKET_DATABASE: Final = "market-data.duckdb"
TASK_CONTROL_DATABASE: Final = Path("runtime") / TASK_CONTROL_DATABASE_FILENAME
MARKET_TABLES: Final[tuple[str, ...]] = (
    # The revision journals.
    "bar_revision",
    "corporate_action_revision",
    "provider_adjusted_series_revision",
    "raw_bar_rebase",
    # The universe history.
    "universe_manifest",
    "universe_manifest_listing",
    "universe_bootstrap",
    "universe_membership_event",
    "universe_source_observation",
    # The provider attempts.
    "provider_attempt",
    "current_universe_hydration_deferred",
    # The audit receipts, the chain and the maintenance records.
    "action_audit_receipt",
    "remediation_execution",
    "action_audit_chain_receipt",
    "data_remediation_failure_receipt",
    "workspace_data_update_receipt",
    "workspace_maintenance_cycle",
    "workspace_maintenance_event",
    # The admission table.
    "current_universe_quality_admission",
)
"""The market store's held tables, as the storage registry names them."""
TASK_CONTROL_TABLES: Final[tuple[str, ...]] = (
    "workspace_task",
    "workspace_task_command",
    "workspace_task_event",
    "workspace_task_execution",
    "workspace_task_projection",
    "workspace_task_stage_receipt",
    "workspace_task_work_item",
)
"""The Task Control store's tables, as the storage registry names them."""
COPIED: Final[tuple[str, ...]] = (
    "research-workspace.json",
    "runtime/artifacts/alternative-evidence",
    "research-experiments",
    "artifacts/product-host",
    "runtime/artifacts/product-host",
    "runtime/feature-trials",
    "runtime/artifacts/data-update-plans",
    "runtime/artifacts/data-change-executions",
    "runtime/artifacts/data-valuation-receipts",
    "runtime/artifacts/index",
    "runtime/artifacts/portfolio-strategy-lab",
)
"""The records copied as they are: the manifest, the Evidence and CRO records, the research
programs and evidence, the Host's records (goals, Portfolio research runs), the feature trials,
the data changes' plans, executions, valuation receipts and their committed index, and the
Portfolio ledger's records, whose descriptors rebuild the Portfolio results the backup carries."""
LISTED: Final[tuple[str, ...]] = (
    "research-inputs",
    "authority",
    "evidence-cro-authority",
    "post-observed-strategy-authority",
    "heterogeneous-current-closure",
    "iw184-quarterly-5y-ensemble",
)
"""Listed by digest only: the research inputs and the installed authority packages, the
retained retrieval model's copy among them (the storage inventory's AUTHORITY and
RETRIEVAL_MODEL roots)."""
BYTE_ADDRESSED: Final[frozenset[tuple[str, ...]]] = frozenset(
    {
        ("alpha-research", "current", "frozen-observation-arrays"),
        ("alpha-research", "current", "lifecycle-arrays"),
    }
)
"""The Alpha development store's packed arrays among the copied records: it names each by the
SHA-256 of its bytes (`AlphaDevelopmentArtifactStore._publish_packed_bytes`) and its loaders
re-hash it on every read. A backup that already holds that object records the file by its name
without reading it again; one it does not hold yet is read and stored as any other file."""
GENERATIONS_KEPT: Final = 7
"""How many generations a backup keeps unless a request keeps another count."""
REMOVED_GRACE: Final = timedelta(days=30)
"""How long a removed workspace's backups stay: a missing workspace's folder may come back (an
unplugged drive, a moved folder opened again), and a restore needs them meanwhile."""
PRUNED: Final = "workspace.backups_pruned"
"""The receipt of a removed workspace's pruned backups: how many generations and bytes went, and
the backup root they went from; a retry resumes only that root."""
_PRUNED_SPEC: Final = {
    "work": False,
    "subject": "workspace",
    "required": ["generations", "bytes", "root"],
    "text": "none",
    "durable": True,
}
_HEX: Final = frozenset("0123456789abcdef")


class WorkspaceBackupError(ValueError):
    """A backup or a restore refused, its code naming why."""


class BackupEntry(BaseModel):  # type: ignore[misc]
    """One held item of a generation: a copied file, an exported table, or a listed entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["FILE", "TABLE", "LISTED"]
    path: str = Field(description="The workspace-relative path; for a table, `<store>/<table>`.")
    digest: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the stored bytes.")
    size: int = Field(ge=0, description="The stored bytes; a listed entry's bytes on disk.")
    rows: int | None = Field(default=None, ge=0, description="A table's rows.")


class BackupGeneration(BaseModel):  # type: ignore[misc]
    """One backup: the held set's digests at one time, sealed by its own hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["WorkspaceBackupGeneration"] = "WorkspaceBackupGeneration"
    workspace_id: str = Field(min_length=1)
    workspace_path: str = Field(
        min_length=1,
        description="The directory backed up: a copy of a workspace keeps its identity, and its "
        "generations are kept and restored apart from the original's.",
    )
    created_at: datetime
    reason: Literal["DATA_UPDATE", "REQUEST"]
    entries: tuple[BackupEntry, ...]
    absent: tuple[str, ...] = Field(
        description="Held items this workspace does not hold: a table or a path."
    )
    market_format: str | None = Field(
        default=None,
        description="The market store's format mark when backed up; none in a generation "
        "written before the mark was kept, and absent from its sealed body.",
    )
    generation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def seal(cls, **values: Any) -> Self:
        """Seal a generation by the hash of everything else it holds.

        Args:
            **values: Every field but `generation_hash`.

        Returns:
            The sealed generation.
        """
        draft = cls.model_construct(**{"market_format": None, **values}, generation_hash="0" * 64)
        return cls(**values, generation_hash=canonical_hash(draft._body()))

    def _body(self) -> dict[str, Any]:
        """What the hash seals: every field but the hash, and no market format a generation
        written before it was kept does not hold, so such a generation verifies as written."""
        unsealed = {"generation_hash"} | (
            {"market_format"} if self.market_format is None else set()
        )
        return dict(self.model_dump(mode="json", exclude=unsealed))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _sealed(self) -> Self:
        if canonical_hash(self._body()) != self.generation_hash:
            raise WorkspaceBackupError("workspace_backup.generation_invalid")
        return self


def default_backup_root(workspace_id: str) -> Path:
    """The workspace's backup root: `ALPHALATTICE_BACKUP_ROOT`, else the application data.

    Args:
        workspace_id: The workspace's identity, a directory of the root.

    Returns:
        `<root>/<workspace_id>`.
    """
    if workspace_id in ("", ".") or any(part in workspace_id for part in ("/", "\\", "..")):
        raise WorkspaceBackupError("workspace_backup.workspace_id_invalid")
    return _backup_base() / workspace_id


def _backup_base() -> Path:
    """The folder every workspace's backup root is a child of."""
    override = process_config().get("ALPHALATTICE_BACKUP_ROOT", "").strip()
    if override:
        return Path(override)
    if os.name == "nt":
        local = process_config().get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(local) / "AlphaLattice" / "backups"
    data = process_config().get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(data) / "alphalattice" / "backups"


type _ReadKind = Literal["source", "object"]


def _unreadable(kind: _ReadKind) -> WorkspaceBackupError:
    """Name the file class whose read failed."""
    return WorkspaceBackupError(
        "workspace_backup.source_unreadable"
        if kind == "source"
        else "workspace_backup.object_unreadable"
    )


def _read_block(stream: BinaryIO, kind: _ReadKind) -> bytes:
    """Read one block without exposing source paths in the refusal."""
    try:
        return stream.read(1 << 20)
    except OSError as error:
        raise _unreadable(kind) from error


def _copy_file(source: Path, destination: Path, *, kind: _ReadKind = "source") -> None:
    """Classify source reads; destination open/write failures keep their own cause."""
    with ExitStack() as stack:
        try:
            reader = stack.enter_context(open(source, "rb"))
        except OSError as error:
            raise _unreadable(kind) from error
        with destination.open("wb") as writer:
            for block in iter(lambda: _read_block(reader, kind), b""):
                writer.write(block)


def _digest_file(path: Path, *, kind: _ReadKind = "source") -> str:
    sha = integrity_sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: _read_block(handle, kind), b""):
                sha.update(block)
    except OSError as error:
        raise _unreadable(kind) from error
    return sha.hexdigest()


def _files(root: Path) -> Iterator[Path]:
    if root.is_file():
        if root.suffix != LOCK_SUFFIX:
            yield root
        return
    for path in sorted(root.rglob("*")):
        if path.suffix != LOCK_SUFFIX and path.is_file() and "__pycache__" not in path.parts:
            yield path


class _Objects:
    """The content-addressed store under one backup root."""

    def __init__(self, root: Path) -> None:
        self.root = root / "objects"

    def path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def held(self, source: Path) -> tuple[str, int] | None:
        """A byte-addressed file's object this store already holds, found by its name alone."""
        name = source.stem
        if (
            source.suffix != ".bin"
            or source.parent.parts[-3:] not in BYTE_ADDRESSED
            or len(name) != 64
            or not set(name) <= _HEX
        ):
            return None
        target = self.path(name)
        size = source.stat().st_size
        if not target.is_file() or target.stat().st_size != size:
            return None
        return name, size

    def put_file(self, source: Path) -> tuple[str, int]:
        """Store a file's bytes under their digest, once."""
        digest = _digest_file(source)
        target = self.path(digest)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_suffix(".partial")
            _copy_file(source, staged)
            if _digest_file(staged, kind="object") != digest:
                staged.unlink()
                raise WorkspaceBackupError("workspace_backup.object_changed_while_copied")
            os.replace(staged, target)
        return digest, target.stat().st_size


def _market_format(database: Path) -> str | None:
    """The market store's format mark, read under its hold; None with no store."""
    if not database.is_file():
        return None
    with hold_workspace_database(database):
        connection = open_workspace_database(database, read_only=True)
        try:
            found: str | None = market_format(connection)
        finally:
            connection.close()
    return found


def _restore_market_store(target: Path) -> None:
    """Write the restored workspace's market store as the current writer creates it, with its
    format mark. A restore never writes the backed-up market tables into the store (they stay
    Parquet under `held-state/market`), so the store it writes holds only what this product
    writes, whatever format the generation records, a generation written before marks were kept
    among them. An older product's own store is not this one; its upgrade stays as
    strict as ever."""
    connection = open_workspace_database(target / MARKET_DATABASE, read_only=False)
    try:
        create_market_profile(connection)
    finally:
        connection.close()
    # The store's first composition: its maintenance tables, which an open under a read
    # boundary cannot create in a store that exists.
    DuckDbWorkspaceMaintenanceRegistry(target / MARKET_DATABASE, gate=WorkspaceMutationGate())


def _existing_tables(database: Path) -> set[str]:
    connection = open_workspace_database(database, read_only=True)
    try:
        rows = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'"
        ).fetchall()
    finally:
        connection.close()
    return {str(row[0]) for row in rows}


def _export_table(database: Path, table: str, staging: Path) -> tuple[Path, int]:
    """One table as Parquet, its rows in a stable order so an unchanged table dedups."""
    connection = open_workspace_database(database, read_only=True)
    try:
        # The name comes from the held set's own list, never from a request.
        try:
            arrow: pa.Table = connection.execute(
                f'SELECT * FROM "{table}" ORDER BY ALL'
            ).to_arrow_table()
        except duckdb.Error:
            arrow = connection.execute(f'SELECT * FROM "{table}"').to_arrow_table()
    finally:
        connection.close()
    target = staging / f"{table}.parquet"
    pq.write_table(arrow, target, compression="zstd", write_statistics=False)
    return target, arrow.num_rows


def _export_rows(rows: list[dict[str, object]], table: str, staging: Path) -> tuple[Path, int]:
    """Rows already read in a stable order, as Parquet."""
    target = staging / f"{table}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), target, compression="zstd", write_statistics=False)
    return target, len(rows)


class WorkspaceBackups:
    """Back up one workspace's held state, and restore a generation onto a new directory."""

    def __init__(
        self,
        workspace: Path,
        *,
        workspace_id: str,
        root: Path | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        """Bind a workspace to its backup root.

        Args:
            workspace: The workspace directory.
            workspace_id: Its identity, the root's directory.
            root: The backup root, else `default_backup_root(workspace_id)`.
            clock: The time a generation is stamped with.

        Raises:
            WorkspaceBackupError: The root lies inside the workspace.
        """
        self.workspace = workspace.resolve()
        self.workspace_id = workspace_id
        self.root = (root or default_backup_root(workspace_id)).resolve()
        if self.root.is_relative_to(self.workspace):
            raise WorkspaceBackupError("workspace_backup.root_inside_workspace")
        self.clock = clock
        self.objects = _Objects(self.root)

    def create(
        self,
        *,
        reason: Literal["DATA_UPDATE", "REQUEST"],
        generations_kept: int = GENERATIONS_KEPT,
        stop: Callable[[], bool] = lambda: False,
    ) -> BackupGeneration:
        """Back up the held set as one generation, then keep the newest `generations_kept`.

        Args:
            reason: What asked for it: a data update, or a request.
            generations_kept: How many generations to keep, at least one.
            stop: Asked between tables and folders; True leaves without a generation.

        Returns:
            The new generation.

        Raises:
            WorkspaceBackupError: The count is below one, the root cannot be written, or a file
                changed while copied.
        """
        if generations_kept < 1:
            raise WorkspaceBackupError("workspace_backup.generations_kept_invalid")
        self._writable_root()
        entries, absent = self._capture(
            lambda path: self.objects.held(path) or self.objects.put_file(path), stop
        )
        generation = BackupGeneration.seal(
            workspace_id=self.workspace_id,
            workspace_path=str(self.workspace),
            created_at=self.clock(),
            reason=reason,
            entries=tuple(entries),
            absent=tuple(absent),
            market_format=_market_format(self.workspace / MARKET_DATABASE),
        )
        generations = self.root / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        stamp = generation.created_at.strftime("%Y%m%dT%H%M%S%fZ")
        path = generations / f"{stamp}-{generation.generation_hash[:16]}.json"
        staged = path.with_suffix(".partial")
        staged.write_text(generation.model_dump_json(indent=1), encoding="utf-8")
        os.replace(staged, path)
        self._keep(generations_kept)
        return generation

    def _capture(
        self,
        read_file: Callable[[Path], tuple[str, int]],
        stop: Callable[[], bool],
    ) -> tuple[list[BackupEntry], list[str]]:
        """Enumerate the same held set for publication and first-upgrade verification."""
        entries: list[BackupEntry] = []
        absent: list[str] = []
        with staging_directory(self.root, ".export-") as staging_name:
            staging = Path(staging_name)
            # A 0.2.7 store not yet moved onto the log is still its Tasks' authority: it is
            # exported as it is; otherwise the log holds them.
            legacy = self.workspace / TASK_CONTROL_DATABASE
            unmoved = legacy if legacy.is_file() else None
            tasks = {} if unmoved is not None else task_rows(legacy)
            for store, database, tables in (
                ("market", self.workspace / MARKET_DATABASE, MARKET_TABLES),
                ("task-control", unmoved, TASK_CONTROL_TABLES),
            ):
                with hold_workspace_database(database) if database is not None else nullcontext():
                    if database is None:
                        present = set(tables)
                    else:
                        present = _existing_tables(database) if database.is_file() else set()
                    for table in tables:
                        if stop():
                            raise _Stopped
                        if table not in present:
                            absent.append(f"{store}/{table}")
                            continue
                        exported, rows = (
                            _export_rows(tasks[table], table, staging)
                            if database is None
                            else _export_table(database, table, staging)
                        )
                        digest, size = read_file(exported)
                        entries.append(
                            BackupEntry(
                                kind="TABLE",
                                path=f"{store}/{table}",
                                digest=digest,
                                size=size,
                                rows=rows,
                            )
                        )
            # The lanes the machine holds for the workspace go as a copy carries them: a restore
            # writes them back there, and the Host that opens it moves them onto its log.
            history.carry_arrays(self.workspace, staging)
            carried = staging / arrays.CARRIED
            for path in sorted(carried.iterdir()) if carried.is_dir() else ():
                if stop():
                    raise _Stopped
                digest, size = read_file(path)
                relative = (arrays.CARRIED / path.name).as_posix()
                entries.append(BackupEntry(kind="FILE", path=relative, digest=digest, size=size))
            # Carry the results and Evidence the machine alone keeps.
            history.carry_results(self.workspace, staging)
            for path in chain(
                history.carried_result_files(staging),
                history.carry_knowledge_snapshots(self.workspace, staging),
                carry_originals(self.workspace, staging),
            ):
                if stop():
                    raise _Stopped
                digest, size = read_file(path)
                relative = path.relative_to(staging).as_posix()
                entries.append(BackupEntry(kind="FILE", path=relative, digest=digest, size=size))
            for path in carry_risk_covariance(
                self.workspace / "research-experiments", staging / "research-experiments"
            ):
                if stop():
                    raise _Stopped
                digest, size = read_file(path)
                relative = path.relative_to(staging).as_posix()
                entries.append(BackupEntry(kind="FILE", path=relative, digest=digest, size=size))
        decisions = self.workspace / DECISIONS_DATABASE
        if decisions.exists():
            if stop():
                raise _Stopped
            # In rollback journal mode, the SQLite read transaction keeps the main file
            # stable through hashing and copying. A pending writer cannot commit over it.
            try:
                with closing(
                    sqlite3.connect(decisions.as_uri() + "?mode=ro", uri=True, timeout=1)
                ) as reader:
                    reader.execute("BEGIN")
                    reader.execute("SELECT id FROM decision LIMIT 1").fetchone()
                    if reader.execute("PRAGMA journal_mode").fetchone() != ("delete",):
                        raise _unreadable("source")
                    digest, size = read_file(decisions)
            except sqlite3.Error as error:
                raise _unreadable("source") from error
            entries.append(
                BackupEntry(
                    kind="FILE", path=DECISIONS_DATABASE.as_posix(), digest=digest, size=size
                )
            )
        for relative in COPIED:
            if stop():
                raise _Stopped
            source = self.workspace / relative
            if not source.exists():
                absent.append(relative)
                continue
            for path in _files(source):
                digest, size = read_file(path)
                entries.append(
                    BackupEntry(
                        kind="FILE",
                        path=path.relative_to(self.workspace).as_posix(),
                        digest=digest,
                        size=size,
                    )
                )
        for relative in LISTED:
            source = self.workspace / relative
            if not source.exists():
                absent.append(relative)
                continue
            entries.extend(self._listed(source))
        return entries, absent

    def _writable_root(self) -> None:
        """Make the root and show it takes a file before anything is exported into it.

        A root this Host cannot create or write -- an application-data directory a sandbox
        denies -- is refused by name, its way on a root named by `ALPHALATTICE_BACKUP_ROOT`.

        Raises:
            WorkspaceBackupError: `workspace_backup.root_unwritable`.
        """
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=self.root, prefix=".write-probe-"):
                pass
        except OSError as error:
            raise WorkspaceBackupError("workspace_backup.root_unwritable") from error

    def _listed(self, source: Path) -> Iterator[BackupEntry]:
        """A listed directory's entries by digest: a content-addressed child by its name."""
        for child in sorted(source.iterdir()):
            size = sum(path.stat().st_size for path in _files(child))
            relative = child.relative_to(self.workspace).as_posix()
            if child.is_dir() and len(child.name) == 64 and set(child.name) <= _HEX:
                yield BackupEntry(kind="LISTED", path=relative, digest=child.name, size=size)
                continue
            sha = integrity_sha256()
            for path in _files(child):
                sha.update(path.relative_to(child).as_posix().encode("utf-8") + b"\0")
                sha.update(_digest_file(path).encode("ascii"))
            yield BackupEntry(kind="LISTED", path=relative, digest=sha.hexdigest(), size=size)

    def generations(self) -> tuple[BackupGeneration, ...]:
        """This workspace directory's kept generations, the newest first.

        Returns:
            The generations the root keeps for this directory.
        """
        return tuple(
            value for _, value in self._all() if value.workspace_path == str(self.workspace)
        )

    def generation_collection(
        self,
    ) -> tuple[tuple[BackupGeneration, ...], tuple[dict[str, str], ...]]:
        """Read the backup-list collection without hiding one unreadable record.

        This partial scan is only for the person-facing backup listing. Restore without an exact
        selector and pruning continue to use the strict `_all` path, so an unreadable generation
        cannot be treated as absent when choosing the newest generation or deciding what to keep.

        Returns:
            This workspace directory's readable generations newest first, and one refusal per
            unreadable file. For conventionally named files the hash field is only the 16-hex
            filename prefix, not a verified generation hash.
        """
        folder = self.root / "generations"
        paths: list[Path] = []
        try:
            for path in folder.iterdir():
                if path.name.endswith(".json"):
                    paths.append(path)
        except FileNotFoundError:
            if not paths:
                return (), ()
            # The directory changed during enumeration; keep observed rows and name the unknown
            # remainder rather than presenting a complete listing.
            paths.append(Path("generations"))
        except OSError:
            # The directory itself may have more unreadable children than have been enumerated.
            # Keep any rows already read and name the collection boundary's unknown remainder.
            paths.append(Path("generations"))
        readable: list[BackupGeneration] = []
        refusals: list[dict[str, str]] = []
        for path in sorted(paths):
            if path == Path("generations"):
                refusals.append(self._generation_file_refusal(path.name))
                continue
            try:
                value = BackupGeneration.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                refusals.append(self._generation_file_refusal(path.name))
                continue
            if value.workspace_path == str(self.workspace):
                readable.append(value)
        return (
            tuple(sorted(readable, key=lambda value: value.created_at, reverse=True)),
            tuple(refusals),
        )

    @staticmethod
    def _generation_file_refusal(name: str) -> dict[str, str]:
        """Name one unverified backup-root file without claiming its sealed contents."""
        row = {
            "status": "REFUSED",
            "failure_code": "workspace_backup.generation_unreadable",
            "generation_file": name,
        }
        stem = Path(name).stem
        suffix = stem.rsplit("-", 1)[-1]
        if len(suffix) == 16 and all(character in "0123456789abcdef" for character in suffix):
            row["generation_hash"] = suffix
        return row

    def _all(self) -> list[tuple[Path, BackupGeneration]]:
        folder = self.root / "generations"
        if not folder.is_dir():
            return []
        found = [
            (path, BackupGeneration.model_validate_json(path.read_text(encoding="utf-8")))
            for path in folder.glob("*.json")
        ]
        return sorted(found, key=lambda pair: pair[1].created_at, reverse=True)

    def _keep(self, generations_kept: int) -> None:
        # Only this directory's generations are pruned: a copy never evicts the original's.
        mine = [pair for pair in self._all() if pair[1].workspace_path == str(self.workspace)]
        for path, _ in mine[generations_kept:]:
            path.unlink()
        named = {
            entry.digest
            for _, generation in self._all()
            for entry in generation.entries
            if entry.kind != "LISTED"
        }
        for path in list(self.objects.root.rglob("*")) if self.objects.root.is_dir() else []:
            if path.is_file() and path.name not in named:
                path.unlink()

    def _generation(self, generation_hash: str | None) -> BackupGeneration:
        """The kept generation a hash names, whole or by its beginning.

        `backup list` shows each hash's beginning, so a restore takes it as shown: the one
        kept generation it begins, an exact hash first; two that share it are refused by name.
        """
        if generation_hash is None:
            kept = self.generations()
            if not kept:
                raise WorkspaceBackupError("workspace_backup.generation_not_found")
            return kept[0]
        folder = self.root / "generations"
        try:
            paths = sorted(folder.glob("*.json"))
        except OSError as error:
            raise WorkspaceBackupError("workspace_backup.generation_not_found") from error
        requested_prefix = generation_hash[:16]
        candidates = []
        for path in paths:
            suffix = path.stem.rsplit("-", 1)[-1]
            if (
                len(suffix) == 16
                and all(character in "0123456789abcdef" for character in suffix)
                and suffix.startswith(requested_prefix)
            ):
                candidates.append(path)
        found: list[BackupGeneration] = []
        unreadable = False
        for path in candidates:
            try:
                value = BackupGeneration.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                unreadable = True
                continue
            # A full hash names its body wherever it was taken; a prefix stays this folder's,
            # since copies of one workspace share its id.
            if value.generation_hash == generation_hash or (
                value.workspace_path == str(self.workspace)
                and value.generation_hash.startswith(generation_hash)
            ):
                found.append(value)
        exact = [value for value in found if value.generation_hash == generation_hash]
        if exact:
            # A full validated content hash selects this exact body even if an unrelated damaged
            # file shares the truncated filename prefix.
            return exact[0]
        if unreadable:
            # A short selector cannot be proven unique while a matching filename is unreadable.
            raise WorkspaceBackupError("workspace_backup.generation_unreadable")
        if len(found) > 1:
            raise WorkspaceBackupError("workspace_backup.generation_ambiguous")
        if not found:
            raise WorkspaceBackupError("workspace_backup.generation_not_found")
        return found[0]

    def verify(
        self, *, generation_hash: str | None = None, current: bool = False
    ) -> BackupGeneration:
        """Verify sealed bytes; first-upgrade admission also compares the current held set.

        Current verification exports only temporary table snapshots. It publishes no generation,
        stores no object and prunes nothing; historical restore verification stays unchanged.
        """
        chosen = self._generation(generation_hash)
        if chosen.workspace_id != self.workspace_id:
            raise WorkspaceBackupError("workspace_backup.generation_invalid")
        for entry in chosen.entries:
            if entry.kind != "LISTED":
                self._verified_object(entry)
        if current:
            entries, absent = self._capture(
                lambda path: (_digest_file(path), path.stat().st_size), lambda: False
            )
            if chosen.entries != tuple(entries) or chosen.absent != tuple(absent):
                raise WorkspaceBackupError("workspace_backup.generation_invalid")
        return chosen

    def _verified_object(self, entry: BackupEntry) -> Path:
        """The one object proof used by both verification and restore."""
        source = self.objects.path(entry.digest)
        if not source.is_file() or _digest_file(source, kind="object") != entry.digest:
            raise WorkspaceBackupError("workspace_backup.object_invalid")
        return source

    def restore(self, target: Path, *, generation_hash: str | None = None) -> dict[str, object]:
        """Write one generation onto a new directory.

        Args:
            target: A directory that does not exist yet, or is empty.
            generation_hash: The generation to restore, whole or by the beginning `backup
                list` shows; the newest when absent.

        Returns:
            What was written and what is only listed.

        Raises:
            WorkspaceBackupError: No such generation, a beginning two generations share, the
                target holds files or lies in the backup root, or an object's bytes changed.
        """
        target = target.resolve()
        if target.exists() and any(target.iterdir()):
            raise WorkspaceBackupError("workspace_backup.restore_target_not_empty")
        if target.is_relative_to(self.root) or target == self.workspace:
            raise WorkspaceBackupError("workspace_backup.restore_target_invalid")
        chosen = self._generation(generation_hash)
        # A whole hash selects a generation from any folder, so it must still be this workspace's.
        if chosen.workspace_id != self.workspace_id:
            raise WorkspaceBackupError("workspace_backup.generation_invalid")
        target.mkdir(parents=True, exist_ok=True)
        files = tables = 0
        task_control: list[BackupEntry] = []
        held_tables: set[str] = set()
        for entry in chosen.entries:
            if entry.kind == "LISTED":
                continue
            source = self._verified_object(entry)
            if entry.kind == "FILE":
                written = target / entry.path
                written.parent.mkdir(parents=True, exist_ok=True)
                _copy_file(source, written, kind="object")
                files += 1
                continue
            held_tables.add(entry.path)
            store, table = entry.path.split("/", 1)
            if store == "task-control":
                task_control.append(entry)
            else:
                written = target / "held-state" / store / f"{table}.parquet"
                written.parent.mkdir(parents=True, exist_ok=True)
                _copy_file(source, written, kind="object")
            tables += 1
        if task_control:
            self._restore_task_control(target, task_control)
        _restore_market_store(target)
        (target / "held-state").mkdir(exist_ok=True)
        (target / "held-state" / "generation.json").write_text(
            chosen.model_dump_json(indent=1), encoding="utf-8"
        )
        return {
            "status": "RESTORED",
            "generation_hash": chosen.generation_hash,
            "created_at": chosen.created_at.isoformat(),
            "target": str(target),
            "files": files,
            "tables": tables,
            "absent": sorted(
                set(chosen.absent)
                | {
                    f"{store}/{table}"
                    for store, names in (
                        ("market", MARKET_TABLES),
                        ("task-control", TASK_CONTROL_TABLES),
                    )
                    for table in names
                    if f"{store}/{table}" not in held_tables
                }
            ),
            "listed": [
                entry.model_dump(mode="json") for entry in chosen.entries if entry.kind == "LISTED"
            ],
            "limitations": [
                "HELD_STATE_ONLY_THE_REST_IS_REBUILT",
                "MARKET_TABLES_AS_PARQUET_FOR_THE_MARKET_STORE_REBUILD",
                "LISTED_ENTRIES_ARE_RECAPTURED_OR_REINSTALLED",
            ],
        }

    def _restore_task_control(self, target: Path, entries: list[BackupEntry]) -> None:
        """Write the Tasks back as a 0.2.7 Task Control store, each table filled by name; the
        restored workspace's first open moves them onto the log its process holds."""
        database = target / TASK_CONTROL_DATABASE
        create_0_2_7_store(database)
        connection = open_workspace_database(database, read_only=False)
        try:
            for entry in entries:
                table = entry.path.split("/", 1)[1]
                if table not in TASK_CONTROL_TABLES:
                    raise WorkspaceBackupError("workspace_backup.table_not_held")
                if not entry.rows:
                    continue  # an empty table's file names no columns to insert by
                source = self.objects.path(entry.digest).as_posix().replace("'", "''")
                connection.execute(
                    f"INSERT INTO \"{table}\" BY NAME SELECT * FROM read_parquet('{source}')"
                )
        finally:
            connection.close()


ATTEMPT: Final = Path("runtime") / "backup" / "last-automatic-attempt.json"
"""The last automatic backup's outcome, kept in the workspace, where a backup root it could
not write would not hold it."""


def restore_from_root(
    workspace: Path,
    target: Path,
    *,
    workspace_id: str | None = None,
    generation_hash: str | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    """Restore a generation from the backup root alone, with no Host.

    The workspace may be lost or unservable, as a restored one usually is: its identity is
    ``workspace_id`` when given, else its manifest's, and the generations restored from are the
    ones recorded for its directory.

    Args:
        workspace: The directory the backup was made of; it need not exist any more.
        target: A directory that does not exist yet, or is empty.
        workspace_id: The workspace's identity, when its manifest cannot be read.
        generation_hash: The generation to restore; the newest when absent.
        root: The backup root, else `default_backup_root(workspace_id)`.

    Returns:
        What was written and what is only listed.

    Raises:
        WorkspaceBackupError: No identity was given and the manifest cannot be read, or as
            `WorkspaceBackups.restore` raises.
    """
    if workspace_id is None:
        try:
            workspace_id = read_research_workspace_manifest(workspace).workspace_id
        except (ResearchWorkspaceError, OSError, ValueError) as error:
            raise WorkspaceBackupError("workspace_backup.workspace_id_required") from error
    return WorkspaceBackups(workspace, workspace_id=workspace_id, root=root).restore(
        target, generation_hash=generation_hash
    )


def root_bytes(root: Path) -> int:
    """The bytes a backup root holds: its generations and the objects they name."""
    return sum(path.stat().st_size for path in _files(root)) if root.is_dir() else 0


def collect_removed_backups(*, now_us: int | None = None) -> list[str]:
    """The backup owner's collection, in the Host's idle time: each workspace whose last folder
    is gone is recorded missing, and one whose folder is back active (`history.mark_missing`);
    then the backups of every workspace missing or deleted for longer than REMOVED_GRACE go.
    Each try is admitted in the log unit that re-reads the workspace's status and folder, before
    any file goes; a first try records its `workspace.backups_pruned` receipt there, durably, with
    the generations and bytes it removes and the backup root they are in. A removal that fails
    stops the collection; a later one resumes each root under that root's own receipt from the
    same removal, and a root without one is a removal of its own. Only workspaces this data
    root's log knows are read, each as one confined folder of the backup root: a backup folder of
    another data root's workspace, or a file at a backup path, is never touched.

    Args:
        now_us: The log time to measure the grace at; the log's clock by default.

    Returns:
        The workspaces whose backups went.
    """
    history.mark_missing()
    pruned = []
    for workspace_id, receipts in history.removed_since(REMOVED_GRACE, PRUNED, now_us=now_us):
        root = _removal_root(workspace_id)
        if root is None or not root.is_dir():
            continue  # an id that names no folder of the backup root, or nothing left to remove
        receipt = next((each for each in receipts if each.get("root") == str(root)), None)
        resumed = receipt is not None  # this removal's receipt for this root keeps its counts
        if receipt is None:  # a first try at this root
            files = list(_files(root))
            generations = sum(1 for path in files if path.parent.name == "generations")
            size = sum(f.stat().st_size for f in files)
            receipt = {"generations": generations, "bytes": size, "root": str(root)}
            history.define(
                "event_type", PRUNED, "A removed workspace's backups went.", _PRUNED_SPEC
            )
        if not history.admit_removal(workspace_id, PRUNED, receipt, resumed=resumed):
            continue  # opened again, or its folder is back: at this try, as at the first
        try:
            shutil.rmtree(root)
        except OSError:
            break  # the collection stops; the next one finishes this admitted removal
        pruned.append(workspace_id)
    return pruned


def _removal_root(workspace_id: str) -> Path | None:
    """A removed workspace's backup root as one confined child of the backup root, through the
    workspace path authority (no `.`, `..`, separator or link); None for an id that names none."""
    if len(Path(workspace_id).parts) != 1:
        return None
    try:
        root: Path = resolve_confined(_backup_base(), workspace_id, reject_existing_hardlink=False)
    except (WorkspaceSecurityError, ValueError, OSError):
        return None
    return root


def record_backup_failure(workspace: Path, error: BaseException | None, *, at: datetime) -> None:
    """Record the last automatic backup's outcome in the workspace.

    Args:
        workspace: The workspace directory.
        error: What refused the backup; None when it succeeded.
        at: When it was attempted.
    """
    code = None if error is None else str(getattr(error, "failure_code", "") or error)
    _record_attempt(
        workspace,
        "BACKED_UP" if error is None else "FAILED",
        None if code is None else code.split(":")[0][:200],
        at,
    )


def defer_automatic_backup(workspace: Path, *, at: datetime) -> None:
    """Record that a data update's backup waits for the Host's idle time.

    The update publishes without it; `take_automatic_backup` takes it once no Task runs, and
    the next update's first stage settles one still waiting or failed
    (`settle_automatic_backup`).

    Args:
        workspace: The workspace directory.
        at: When the update deferred it.
    """
    _record_attempt(workspace, "PENDING", None, at)


def take_automatic_backup(
    workspace: Path,
    *,
    clock: Callable[[], datetime],
    gate: WorkspaceMutationGate,
    stop: Callable[[], bool],
) -> None:
    """Take a waiting automatic backup in the Host's idle time, at below-normal priority, after
    the collection of removed workspaces' backups (`collect_removed_backups`).

    It holds the workspace's write gate, and leaves the backup waiting whenever ``stop`` says a
    Task waits or the Host closes (checked between tables and folders) or the gate stays busy.

    Args:
        workspace: The workspace directory.
        clock: The time the generation and its outcome are stamped with.
        gate: The workspace's mutation gate.
        stop: Whether to leave the backup waiting now.
    """
    if stop():
        return
    with below_normal_thread():
        collect_removed_backups()
        if (last_automatic_attempt(workspace) or {}).get("status") == "PENDING":
            _take(workspace, clock=clock, gate=gate, stop=stop, wait_seconds=1.0)


def settle_automatic_backup(
    workspace: Path, *, clock: Callable[[], datetime], gate: WorkspaceMutationGate
) -> str | None:
    """Take the last update's backup if it still waits or failed, with one retry, in an update.

    A waiting backup gets its one retry here; a failed one has had its first attempt in the
    Host's idle time.

    Args:
        workspace: The workspace directory.
        clock: The time the generation and its outcome are stamped with.
        gate: The workspace's mutation gate.

    Returns:
        The failure code the backup still ends with; None when it succeeded or none was due.
    """
    status = (last_automatic_attempt(workspace) or {}).get("status")
    if status not in {"PENDING", "FAILED"}:
        return None
    for _attempt in range(2 if status == "PENDING" else 1):
        failure = _take(workspace, clock=clock, gate=gate, stop=lambda: False, wait_seconds=30.0)
        if failure is None:
            return None
    return failure


class _Stopped(Exception):
    """The backup was left waiting: a Task waits, the Host closes, or the gate stays busy."""


def _take(
    workspace: Path,
    *,
    clock: Callable[[], datetime],
    gate: WorkspaceMutationGate,
    stop: Callable[[], bool],
    wait_seconds: float,
) -> str | None:
    try:
        with (
            gate.try_hold(timeout_seconds=wait_seconds),
            WorkspaceDatabase(workspace).retain(read_only=False),
            span("copy", "held_state_backup"),
        ):
            WorkspaceBackups(
                workspace,
                workspace_id=read_research_workspace_manifest(workspace).workspace_id,
                clock=clock,
            ).create(reason="DATA_UPDATE", stop=stop)
    except (_Stopped, TimeoutError):
        return None
    except Exception as error:
        record_backup_failure(workspace, error, at=clock())
        return str((last_automatic_attempt(workspace) or {}).get("failure_code"))
    record_backup_failure(workspace, None, at=clock())
    return None


def _record_attempt(workspace: Path, status: str, code: str | None, at: datetime) -> None:
    path = workspace / ATTEMPT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "failure_code": code, "at": at.isoformat()}
    staged = path.with_suffix(".partial")
    staged.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(staged, path)


def last_automatic_attempt(workspace: Path) -> dict[str, object] | None:
    """The last automatic backup's recorded outcome, if one was attempted.

    Args:
        workspace: The workspace directory.

    Returns:
        Its status, failure code and time; None when none was attempted.
    """
    path = workspace / ATTEMPT
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def backup_answer(backups: WorkspaceBackups, generation: BackupGeneration | None) -> dict[str, Any]:
    """The answer a backup request gives: the root and every kept generation, briefly.

    Args:
        backups: The workspace's backups.
        generation: The generation this request made, if it made one.

    Returns:
        The answer.
    """
    automatic = last_automatic_attempt(backups.workspace)
    if automatic is not None and automatic.get("failure_code"):
        # The automatic backup's refusal reads in words, with its way on, as a request's does.
        automatic = {**automatic, **refusal_words(str(automatic["failure_code"]))}
    readable_generations, refusals = backups.generation_collection()
    body = {
        "status": "BACKED_UP" if generation is not None else "READ",
        "backup_root": str(backups.root),
        "bytes": root_bytes(backups.root),
        "last_automatic_attempt": automatic,
        "generation_hash": None if generation is None else generation.generation_hash,
        "generations": [
            {
                "generation_hash": value.generation_hash,
                "created_at": value.created_at.isoformat(),
                "reason": value.reason,
                "files": sum(entry.kind == "FILE" for entry in value.entries),
                "tables": sum(entry.kind == "TABLE" for entry in value.entries),
                "listed": sum(entry.kind == "LISTED" for entry in value.entries),
                "absent": list(value.absent),
            }
            for value in readable_generations
        ],
    }
    if refusals:
        body["refusals"] = [
            {
                **row,
                "detail": (
                    f"Backup-root record {row['generation_file']} could not be verified; its "
                    "sealed identity, workspace association and suitability for restore are "
                    "unknown. Read `workspace show` and `backup list`. If a listed verified "
                    "generation is appropriate, restore that full hash with `alphalattice backup "
                    "restore --dir <new directory> --generation <verified generation hash> "
                    "--workspace-id <workspace id> --root <backup root>`. Do not select this "
                    "unverified file by its filename prefix."
                ),
                "next_requests": {
                    "workspace": {"operation": "WORKSPACE_SHOW"},
                    "backups": {"operation": "WORKSPACE_BACKUPS"},
                },
            }
            for row in refusals
        ]
    return body


__all__ = [
    "COPIED",
    "GENERATIONS_KEPT",
    "LISTED",
    "MARKET_TABLES",
    "PRUNED",
    "REMOVED_GRACE",
    "TASK_CONTROL_TABLES",
    "BackupEntry",
    "BackupGeneration",
    "WorkspaceBackupError",
    "WorkspaceBackups",
    "backup_answer",
    "collect_removed_backups",
    "default_backup_root",
    "defer_automatic_backup",
    "last_automatic_attempt",
    "record_backup_failure",
    "root_bytes",
    "settle_automatic_backup",
    "take_automatic_backup",
]
