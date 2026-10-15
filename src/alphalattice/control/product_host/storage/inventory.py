"""Execution-day inventory and verified-root resolution for desktop storage."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from alphalattice.control.workspace_runtime.storage.capacity import StorageCapacity, StorageCapStore

RECOVERY_HEADROOM_BYTES = 1_610_612_736
"""Physical publication/recovery reserve, a safety limit independent of the managed cap."""

MANAGED_ROOTS = (
    ("market-data.duckdb", "WORKING_DATABASE"),
    ("research-inputs", "IMMUTABLE_INPUTS"),
    ("artifacts", "ARTIFACTS"),
    ("research-experiments", "RESEARCH_EXPERIMENTS"),
    ("runtime/feature-trials", "RESEARCH_EXPERIMENTS"),
    ("yfinance-cache", "PROVIDER_CACHE"),
    ("source-probe-cache", "PROVIDER_CACHE"),
    ("staging", "STAGING"),
    ("runtime/evidence-knowledge", "EVIDENCE_KNOWLEDGE"),
    ("runtime/artifacts/alternative-evidence", "ALTERNATIVE_EVIDENCE"),
    ("runtime/artifacts/portfolio-strategy-lab", "PORTFOLIO_LEDGER"),
    # Every other store that holds a workspace's content, so the budget is the disk it takes
    # (V208); none is released by a clean-up, whose targets are the input owner's alone. The live
    # operational stores (Task Control, the Feature build's registry, the checkpoints and
    # observations, a Task's execution folder) are left out: a running Task's own bookkeeping
    # grows and shrinks them under its admitted writes (DuckDB truncates its file at a
    # checkpoint), so counting them would let a write through at the cap.
    ("runtime/preparation", "PREPARATION"),
    ("preparation-evidence", "PREPARATION"),
    ("runtime/artifacts/data-update-plans", "HOST_RECORDS"),
    ("runtime/artifacts/data-change-executions", "HOST_RECORDS"),
    ("runtime/artifacts/data-valuation-receipts", "HOST_RECORDS"),
    ("runtime/artifacts/index", "HOST_RECORDS"),
    ("runtime/artifacts/product-host", "HOST_RECORDS"),
    ("authority", "AUTHORITY"),
    ("post-observed-strategy-authority", "AUTHORITY"),
    ("heterogeneous-current-closure", "AUTHORITY"),
    ("iw184-quarterly-5y-ensemble", "AUTHORITY"),
    # The retained retrieval recipe's own copy; a later recipe links the machine's store
    # (`semantic-model-<recipe>`), whose bytes are not the workspace's.
    ("evidence-cro-authority/semantic-model", "RETRIEVAL_MODEL"),
)
"""One set of owned roots for physical enforcement and its user-facing breakdown; a role may
name several roots."""

_measured_data: dict[Path, int] = {}
"""Last complete physical sample; observation retention reuses it between managed admissions."""


@dataclass
class _StageMeasure:
    """One stage's walked managed trees per workspace, plus the bytes it admitted since."""

    lock: Lock = field(default_factory=Lock)
    trees: dict[Path, int] = field(default_factory=dict)
    files: dict[Path, tuple[Path, ...]] = field(default_factory=dict)


_STAGE_MEASURE: ContextVar[_StageMeasure | None] = ContextVar("storage_stage", default=None)


@contextmanager
def storage_capacity_scope() -> Iterator[None]:
    """One managed-inventory walk per workspace for a Task stage.

    The first admission in the scope walks the managed trees; each later one adds the bytes
    the stage already admitted, so the write that crosses the cap still refuses at that write.
    A root that is one file (the market database, which grows under the stage's own
    unadmitted writes) and free disk are read at every admission. Deletions, and another
    process's writes, count from the next stage's walk on. Outside a scope nothing changes.
    """
    token = _STAGE_MEASURE.set(_StageMeasure())
    try:
        yield
    finally:
        _STAGE_MEASURE.reset(token)


def managed_file_inventory(workspace: Path) -> tuple[tuple[str, int, str], ...]:
    """Count owned current/input/cache bytes; hard-linked objects count once.

    The file identity is physical accounting, never a scientific content hash.
    Unknown filesystem identities conservatively count as separate files.

    The walk uses directory entries for the link, junction and directory
    tests and one ``stat`` per file for its size and physical identity. The
    previous ``Path`` walk made four to five system calls per entry, which made
    this check -- run before every one-listing maintenance step and every
    evidence write -- cost about 0.12 s on a 1,600-file workspace; the junction
    test then still stat'ed each entry again (``os.path.isjunction``), about a
    sixth of the walk on a first day's 7,700-file evidence workspace. Scope,
    ordering, identity keys and the link refusal are unchanged.
    """
    root = workspace.resolve()
    files = []
    for relative, role in MANAGED_ROOTS:
        path = root / relative
        # The retained retrieval layout links the machine's model store: those
        # bytes are not the workspace's, so a link there is passed over; elsewhere it is refused.
        store_links = role == "RETRIEVAL_MODEL"
        if not path.exists():
            continue
        if path.is_symlink() or path.is_junction():
            if store_links:
                continue
            raise StorageInventoryError(
                "storage.retention_root_mismatch", "Managed storage contains a link."
            )
        if not path.is_dir():
            stat = path.stat()
            name = path.relative_to(root).as_posix()
            key = f"{stat.st_dev}:{stat.st_ino}" if stat.st_ino else name
            files.append((name, stat.st_size, key))
            continue
        pending = [path]
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_symlink() or entry.is_junction():
                        if store_links:
                            continue
                        raise StorageInventoryError(
                            "storage.retention_root_mismatch", "Managed storage contains a link."
                        )
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        continue
                    stat = os.stat(entry.path)
                    name = Path(entry.path).relative_to(root).as_posix()
                    key = f"{stat.st_dev}:{stat.st_ino}" if stat.st_ino else name
                    files.append((name, stat.st_size, key))
    return tuple(sorted(files))


def unique_managed_bytes(files: tuple[tuple[str, int, str], ...]) -> int:
    return sum(dict((identity, size) for _name, size, identity in files).values())


def workspace_storage_capacity(workspace: Path, *, last_sample: bool = False) -> StorageCapacity:
    """Resolve the operator setting over all deduplicated managed data, with its limits."""
    root = workspace.resolve()
    setting = StorageCapStore(root).read()
    if last_sample and setting.cap_bytes != "auto":
        return StorageCapStore(root).capacity(measured_data_bytes=_measured_data.get(root, 0))
    if not last_sample or root not in _measured_data:
        _measured_data[root] = unique_managed_bytes(managed_file_inventory(root))
    return StorageCapStore(root).capacity(measured_data_bytes=_measured_data[root])


def require_storage_capacity(workspace: Path, *, additional_bytes: int) -> None:
    """Admit bytes against the current operator cap and physical recovery reserve."""
    stage = _STAGE_MEASURE.get()
    if stage is None:
        _admit(workspace_storage_capacity(workspace), additional_bytes)
        return
    root = workspace.resolve()
    with stage.lock:
        if root not in stage.trees:
            rows = managed_file_inventory(root)
            _measured_data[root] = unique_managed_bytes(rows)
            roots = {relative for relative, _role in MANAGED_ROOTS}
            stage.files[root] = tuple(root / row[0] for row in rows if row[0] in roots)
            stage.trees[root] = unique_managed_bytes(
                tuple(row for row in rows if row[0] not in roots)
            )
        single: dict[str, int] = {}
        for path in stage.files[root]:
            with suppress(FileNotFoundError):
                stat = path.stat()
                single[f"{stat.st_dev}:{stat.st_ino}" if stat.st_ino else str(path)] = stat.st_size
        measured = stage.trees[root] + sum(single.values())
        _admit(StorageCapStore(root).capacity(measured_data_bytes=measured), additional_bytes)
        stage.trees[root] += additional_bytes


def _admit(capacity: StorageCapacity, additional_bytes: int) -> None:
    if capacity.measured_data_bytes + additional_bytes > capacity.cap_bytes:
        raise StorageInventoryError(
            "storage.managed_capacity_exceeded",
            "Managed storage exceeds the workspace cap. "
            "Raise the cap (`storage set --cap-bytes <bytes>`) or plan a cleanup (`storage plan`).",
        )
    if capacity.free_disk_bytes < additional_bytes + RECOVERY_HEADROOM_BYTES:
        raise StorageInventoryError(
            "storage.disk_space_insufficient",
            "Insufficient space for publication and recovery headroom.",
        )


class StorageInventoryError(RuntimeError):
    """Carry a named public storage inventory refusal with explicit detail."""

    def __init__(self, failure_code: str, detail: str) -> None:
        """Retain the explicit storage inventory failure code and message.

        Args:
            failure_code: Named owner-supplied public failure code.
            detail: Explicit refusal explanation.
        """
        super().__init__(detail)
        self.failure_code = failure_code


__all__ = ["StorageInventoryError"]
