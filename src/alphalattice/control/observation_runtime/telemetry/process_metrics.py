"""Cross-platform process resources shared by deterministic workspace runs.

Measurement and one bound. The bound is here rather than in a runner because a
capacity ceiling is a property of *this process*, exactly like the resident-set
readings beside it, and because the only mechanism that actually covers every
thread a run creates -- BLAS, OpenMP, and a DuckDB thread pool the runner never
sees -- is a process-level one.

Nothing here is scientific evidence. A scheduling limit changes how long a run
takes and never what it computes, so none of it may enter a Formula, a catalog or
a Panel identity.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import psutil  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _windows_memory_bytes() -> tuple[int, int]:
    """Read current and peak Windows working sets without platform-only exports."""
    loader: Any = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return 0, 0

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = loader("kernel32", use_last_error=True)
    psapi = loader("psapi", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_memory_info = psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Counters),
        ctypes.c_ulong,
    ]
    get_memory_info.restype = ctypes.c_bool
    if not get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb):
        return 0, 0
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def current_rss_bytes() -> int:
    """Return current resident memory in bytes, or zero when unavailable."""
    if os.name == "nt":
        return _windows_memory_bytes()[0]
    try:
        sysconf = getattr(os, "sysconf", None)
        if not callable(sysconf):
            return 0
        page_count = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return page_count * int(sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        return 0


def peak_rss_bytes() -> int:
    """Return process peak resident memory in bytes, or zero when unavailable."""
    if os.name == "nt":
        return _windows_memory_bytes()[1]

    resource: Any = importlib.import_module("resource")
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


@dataclass(frozen=True, slots=True)
class ProcessResourceUsage:
    """Capacity evidence normalized to a percentage of the whole machine."""

    wall_seconds: float
    average_machine_cpu_percent: float
    peak_machine_cpu_percent: float
    peak_rss_bytes: int
    process_tree_peak_rss_bytes: int = 0
    process_tree_peak_live_descendant_count: int = 0
    sample_interval_seconds: float = 0.25
    measurement_scope: Literal["PARENT_PROCESS_ONLY", "PARENT_AND_LIVE_DESCENDANTS"] = (
        "PARENT_PROCESS_ONLY"
    )
    instrumentation_limitation: str | None = None


@dataclass(frozen=True, slots=True)
class _ProcessTreeSample:
    """One simultaneous observation of the parent and presently live children."""

    cpu_seconds_by_pid: dict[int, float]
    rss_bytes_by_pid: dict[int, int]
    instrumentation_limitation: str | None = None


def _windows_descendant_process_ids(root_pid: int) -> tuple[tuple[int, ...], str | None]:
    """Return the live descendant graph with ToolHelp, without a new dependency."""
    loader: Any = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return (root_pid,), "PROCESS_TREE_ENUMERATION_UNAVAILABLE"

    class _ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = loader("kernel32", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        return (root_pid,), "PROCESS_TREE_ENUMERATION_UNAVAILABLE"
    try:
        first = kernel32.Process32FirstW
        next_entry = kernel32.Process32NextW
        first.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry)]
        next_entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessEntry)]
        first.restype = next_entry.restype = ctypes.c_bool
        children: dict[int, list[int]] = {}
        entry = _ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        if not first(snapshot, ctypes.byref(entry)):
            return (root_pid,), "PROCESS_TREE_ENUMERATION_UNAVAILABLE"
        while True:
            children.setdefault(int(entry.th32ParentProcessID), []).append(int(entry.th32ProcessID))
            entry = _ProcessEntry()
            entry.dwSize = ctypes.sizeof(entry)
            if not next_entry(snapshot, ctypes.byref(entry)):
                break
        pending = [root_pid]
        result: list[int] = []
        while pending:
            current = pending.pop()
            result.append(current)
            pending.extend(children.get(current, ()))
        return tuple(result), None
    finally:
        kernel32.CloseHandle(snapshot)


def _linux_descendant_process_ids(root_pid: int) -> tuple[tuple[int, ...], str | None]:
    """Read the Linux process parent graph from procfs when it is available."""
    parents: dict[int, list[int]] = {}
    try:
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            payload = (path / "stat").read_text(encoding="ascii")
            tail = payload.rsplit(")", 1)[1].split()
            parents.setdefault(int(tail[1]), []).append(int(path.name))
    except (OSError, ValueError, IndexError):
        return (root_pid,), "PROCESS_TREE_ENUMERATION_UNAVAILABLE"
    pending = [root_pid]
    result: list[int] = []
    while pending:
        current = pending.pop()
        result.append(current)
        pending.extend(parents.get(current, ()))
    return tuple(result), None


class _ProcessExited(Exception):
    """A descendant that exited between the tree's enumeration and its read: gone, not
    unreadable, so its absence is no limitation of the measurement (V472)."""


_ERROR_INVALID_PARAMETER = 87
"""Windows' answer to opening a process id that no longer names a process."""


def _process_tree_ids(root_pid: int) -> tuple[tuple[int, ...], str | None]:
    if os.name == "nt":
        return _windows_descendant_process_ids(root_pid)
    if Path("/proc").is_dir():
        return _linux_descendant_process_ids(root_pid)
    return (root_pid,), "PROCESS_TREE_ENUMERATION_UNAVAILABLE"


def _windows_process_usage(process_id: int) -> tuple[float, int] | None:
    """Read one process's CPU and working set, skipping an exited child."""
    loader: Any = getattr(ctypes, "WinDLL", None)
    if loader is None:
        return None

    class _FileTime(ctypes.Structure):
        _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32 = loader("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, process_id)
    if not handle:
        if ctypes.get_last_error() == _ERROR_INVALID_PARAMETER:
            raise _ProcessExited
        return None
    try:
        creation = _FileTime()
        exited = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        get_times = kernel32.GetProcessTimes
        get_times.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
            ctypes.POINTER(_FileTime),
        ]
        get_times.restype = ctypes.c_bool
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        psapi = loader("psapi", use_last_error=True)
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
        get_memory.restype = ctypes.c_bool
        if not get_times(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ) or not get_memory(handle, ctypes.byref(counters), counters.cb):
            return None
        kernel_ticks = (int(kernel.dwHighDateTime) << 32) | int(kernel.dwLowDateTime)
        user_ticks = (int(user.dwHighDateTime) << 32) | int(user.dwLowDateTime)
        return (kernel_ticks + user_ticks) / 10_000_000.0, int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def _linux_process_usage(process_id: int) -> tuple[float, int] | None:
    try:
        stat = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii").rsplit(")", 1)[1]
        values = stat.split()
        ticks = int(values[11]) + int(values[12])
        pages = int(Path(f"/proc/{process_id}/statm").read_text(encoding="ascii").split()[1])
        sysconf = getattr(os, "sysconf", None)
        if not callable(sysconf):
            return None
        seconds = ticks / float(sysconf("SC_CLK_TCK"))
        return seconds, pages * int(sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError, ProcessLookupError) as gone:
        raise _ProcessExited from gone
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _process_tree_sample(root_pid: int, *, include_descendants: bool) -> _ProcessTreeSample:
    process_ids, limitation = (
        _process_tree_ids(root_pid) if include_descendants else ((root_pid,), None)
    )
    cpu_seconds: dict[int, float] = {}
    rss_bytes: dict[int, int] = {}
    unreadable = False
    for process_id in process_ids:
        usage: tuple[float, int] | None
        try:
            if process_id == root_pid:
                usage = (process_cpu_seconds(), current_rss_bytes())
            elif os.name == "nt":
                usage = _windows_process_usage(process_id)
            elif Path("/proc").is_dir():
                usage = _linux_process_usage(process_id)
            else:
                usage = None
        except _ProcessExited:
            continue
        if usage is None:
            unreadable = True
            continue
        cpu_seconds[process_id], rss_bytes[process_id] = usage
    if unreadable and limitation is None:
        limitation = "PROCESS_TREE_DESCENDANT_ACCESS_PARTIAL"
    return _ProcessTreeSample(cpu_seconds, rss_bytes, limitation)


class ProcessResourceMonitor:
    """Sample a parent or its live process tree at shared sample instants."""

    def __init__(
        self,
        *,
        sample_interval_seconds: float = 0.25,
        include_live_descendants: bool = False,
    ) -> None:
        """Configure sampled parent or live-descendant operational resource measurements.

        Args:
            sample_interval_seconds: Positive interval between shared process-tree samples.
            include_live_descendants: Whether to sample presently live descendants with the parent.

        Raises:
            ValueError: The sample interval is not positive.
        """
        if sample_interval_seconds <= 0.0:
            raise ValueError("process_metrics.sample_interval_invalid")
        self._sample_interval_seconds = sample_interval_seconds
        self._include_live_descendants = include_live_descendants
        self._root_pid = os.getpid()
        self._logical_cpu_count = max(1, os.cpu_count() or 1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_wall = 0.0
        self._last_wall = 0.0
        self._cpu_seconds_by_pid: dict[int, float] = {}
        self._total_cpu_seconds = 0.0
        self._peak_machine_cpu_percent = 0.0
        self._peak_rss_bytes = 0
        self._process_tree_peak_rss_bytes = 0
        self._process_tree_peak_live_descendant_count = 0
        self._instrumentation_limitation: str | None = None
        self._usage: ProcessResourceUsage | None = None

    def start(self) -> None:
        """Take an initial sample and start the background operational sampler.

        Raises:
            RuntimeError: This monitor has already been started.
        """
        if self._thread is not None:
            raise RuntimeError("process_metrics.monitor_already_started")
        self._start_wall = self._last_wall = time.perf_counter()
        self._peak_rss_bytes = current_rss_bytes()
        self._sample(initial=True)
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="alphalattice-process-resource-monitor",
            daemon=True,
        )
        self._thread.start()

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self._sample_interval_seconds):
            self._sample()

    def _sample(self, *, initial: bool = False) -> None:
        wall = time.perf_counter()
        sample = _process_tree_sample(
            self._root_pid,
            include_descendants=self._include_live_descendants,
        )
        elapsed = wall - self._last_wall
        if sample.instrumentation_limitation is not None:
            self._instrumentation_limitation = sample.instrumentation_limitation
        cpu_delta = 0.0
        for process_id, cpu_seconds in sample.cpu_seconds_by_pid.items():
            previous = self._cpu_seconds_by_pid.get(process_id)
            if previous is not None:
                cpu_delta += max(0.0, cpu_seconds - previous)
            self._cpu_seconds_by_pid[process_id] = cpu_seconds
        if not initial:
            self._total_cpu_seconds += cpu_delta
        if elapsed > 0.0 and not initial:
            usage = cpu_delta / elapsed / self._logical_cpu_count * 100.0
            self._peak_machine_cpu_percent = max(
                self._peak_machine_cpu_percent,
                max(0.0, usage),
            )
        self._peak_rss_bytes = max(self._peak_rss_bytes, current_rss_bytes())
        self._process_tree_peak_rss_bytes = max(
            self._process_tree_peak_rss_bytes,
            sum(sample.rss_bytes_by_pid.values()),
        )
        self._process_tree_peak_live_descendant_count = max(
            self._process_tree_peak_live_descendant_count,
            max(0, len(sample.rss_bytes_by_pid) - 1),
        )
        self._last_wall = wall

    def finish(self) -> ProcessResourceUsage:
        """Stop sampling and return the cached operational usage receipt.

        CPU percentages are normalized by the detected logical processor count. Live-tree
        RSS is a sum at shared sample instants, not a sum of per-child historical peaks.
        Short-lived descendants between samples may not be observed. Parent-only mode
        leaves tree-specific fields at zero and identifies its measurement scope.

        Returns:
            The sampled usage receipt; subsequent calls return the same receipt.

        Raises:
            RuntimeError: Sampling has not been started.
        """
        if self._usage is not None:
            return self._usage
        if self._thread is None:
            raise RuntimeError("process_metrics.monitor_not_started")
        self._stop.set()
        self._thread.join()
        self._sample()
        wall_seconds = max(0.0, self._last_wall - self._start_wall)
        average = (
            self._total_cpu_seconds / wall_seconds / self._logical_cpu_count * 100.0
            if wall_seconds > 0.0
            else 0.0
        )
        self._usage = ProcessResourceUsage(
            wall_seconds=wall_seconds,
            average_machine_cpu_percent=max(0.0, average),
            peak_machine_cpu_percent=self._peak_machine_cpu_percent,
            peak_rss_bytes=self._peak_rss_bytes,
            process_tree_peak_rss_bytes=(
                self._process_tree_peak_rss_bytes if self._include_live_descendants else 0
            ),
            process_tree_peak_live_descendant_count=(
                self._process_tree_peak_live_descendant_count
                if self._include_live_descendants
                else 0
            ),
            sample_interval_seconds=self._sample_interval_seconds,
            measurement_scope=(
                "PARENT_AND_LIVE_DESCENDANTS"
                if self._include_live_descendants
                else "PARENT_PROCESS_ONLY"
            ),
            instrumentation_limitation=(
                self._instrumentation_limitation if self._include_live_descendants else None
            ),
        )
        return self._usage


LOGICAL_PROCESSOR_BUDGET_DIVISOR = 2
"""Half the machine, which is the ceiling these runs are held to."""


class ProcessCapacityError(RuntimeError):
    """Operational refusal: a declared capacity bound could not be enforced."""

    failure_class = "OPERATIONAL_FAILURE"

    def __init__(self, code: str) -> None:
        """Retain the operational refusal code for callers and exception text.

        Args:
            code: Machine-readable process capacity or enforcement failure code.
        """
        self.code = code
        super().__init__(code)


class _CapacityContract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class RuntimeCapacityRequest(_CapacityContract):
    """Declare memory reservation and processor headroom for operational admission.

    Attributes:
        maximum_memory_gib: Optional positive absolute memory reservation; live admission may refuse
            it.
        maximum_memory_fraction: Fractional memory ceiling used when no absolute reservation is
            supplied.
        reserved_logical_processors: Positive processor headroom kept outside the workload.
    """

    maximum_memory_gib: float | None = Field(default=None, gt=0.0)
    maximum_memory_fraction: float = Field(default=0.60, gt=0.0, le=0.90)
    reserved_logical_processors: int = Field(default=1, ge=1)


class RuntimeParallelismRequest(_CapacityContract):
    """Declare supported fold, database and numerical-library concurrency.

    AUTO lets the owner resolve fold and DuckDB counts under machine and profile
    bounds. Installed LightGBM and BLAS policies remain single-threaded; the request
    does not authorize an unverified model-thread override.

    Attributes:
        fold_workers: AUTO or a positive requested fold worker limit.
        lightgbm_threads_per_fit: Installed fixed single-thread model policy.
        duckdb_threads: AUTO or a positive requested DuckDB thread limit.
        blas_threads: Fixed single-thread BLAS policy.
    """

    fold_workers: Literal["AUTO"] | int = "AUTO"
    # The only deterministic model-thread evidence installed for this adapter
    # is its fixed single-thread recipe.  Do not present an override as active
    # until a bounded parity receipt installs another count.
    lightgbm_threads_per_fit: Literal[1] = 1
    duckdb_threads: Literal["AUTO"] | int = "AUTO"
    blas_threads: Literal[1] = 1

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_positive_overrides(self) -> Self:
        """Reject nonpositive integer concurrency overrides.

        Returns:
            This validated request.

        Raises:
            ProcessCapacityError: A declared integer override is below one.
        """
        if any(
            isinstance(value, int) and value < 1
            for value in (
                self.fold_workers,
                self.lightgbm_threads_per_fit,
                self.duckdb_threads,
            )
        ):
            raise ProcessCapacityError("process_capacity.parallelism_override_invalid")
        return self


class ResearchRuntimeRequest(_CapacityContract):
    """Compose a runtime profile, capacity reservation and concurrency request.

    Attributes:
        profile: Explicit operating profile or AUTO resolution policy.
        capacity: Memory reservation and processor headroom.
        parallelism: Supported fold/database/numerical-library concurrency.
    """

    profile: Literal["AUTO", "LOW_MEMORY", "BALANCED", "THROUGHPUT"] = "AUTO"
    capacity: RuntimeCapacityRequest = Field(default_factory=RuntimeCapacityRequest)
    parallelism: RuntimeParallelismRequest = Field(default_factory=RuntimeParallelismRequest)


RuntimeWorkload = Literal[
    "METADATA_PREFLIGHT",
    "ALPHA_FOLDS",
    "SCORE_FILTER_NUMERICAL",
    "PORTFOLIO_NUMERICAL",
]


class RuntimeMachineCapacity(_CapacityContract):
    """Carry a live machine snapshot used to admit a sealed execution policy.

    Attributes:
        detected_logical_processors: Positive logical processor count detected on the machine.
        allowed_logical_processor_ids: Nonempty distinct IDs permitted by current process affinity.
        total_memory_bytes: Positive observed physical RAM capacity.
        available_memory_bytes: Positive observed available RAM, no larger than total RAM.
    """

    detected_logical_processors: int = Field(ge=1)
    allowed_logical_processor_ids: tuple[int, ...] = Field(min_length=1)
    total_memory_bytes: int = Field(ge=1)
    available_memory_bytes: int = Field(ge=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_capacity(self) -> Self:
        """Require distinct allowed processors and bounded available memory.

        Returns:
            This validated machine snapshot.

        Raises:
            ProcessCapacityError: Processor IDs repeat or available memory exceeds total RAM.
        """
        if (
            len(set(self.allowed_logical_processor_ids)) != len(self.allowed_logical_processor_ids)
            or self.available_memory_bytes > self.total_memory_bytes
        ):
            raise ProcessCapacityError("process_capacity.machine_snapshot_invalid")
        return self


class ResolvedRuntimeCapacityPlan(_CapacityContract):
    """Seal workload admission and execution limits separately from live RAM telemetry.

    The plan hash excludes total and available RAM snapshots while retaining the
    declared memory budget, resolved concurrency, processor policy and admission.
    Estimated tree RSS and NOT_MEASURED markers are not measured process-tree peaks.
    Applied affinity IDs are empty until a verified enforcement receipt is bound.

    Attributes:
        profile: Requested operating profile.
        profile_resolution: Resolved low-memory, balanced or throughput profile.
        workload: Metadata, Alpha fold or numerical workload being admitted.
        detected_logical_processors: Detected machine processor count.
        allowed_logical_processors: Count allowed by the current affinity snapshot.
        total_memory_bytes: Live RAM snapshot excluded from plan identity.
        available_memory_bytes: Live available RAM snapshot excluded from plan identity.
        admitted_memory_budget_bytes: Sealed positive memory reservation.
        reserved_logical_processors: Processor headroom outside the workload.
        process_logical_processor_limit: Positive process affinity limit.
        fold_workers: Admitted Alpha fold count, zero for other or refused work.
        lightgbm_threads_per_fit: One for Alpha models, zero for workloads with no model owner.
        duckdb_threads: Positive declared database thread bound.
        blas_threads: Fixed single-thread numerical-library policy.
        applied_logical_processor_ids: Distinct enforced affinity IDs, or empty before enforcement.
        estimated_peak_process_tree_rss_bytes: Reservation estimate, not a sampled peak.
        execution_waves: Positive estimated fold scheduling wave count.
        admission: ADMITTED or REFUSED under live capacity and sealed policy.
        refusal_code: Absent exactly when admission is ADMITTED.
        plan_hash: Canonical policy identity excluding its hash and live RAM snapshots.
    """

    kind: Literal["ResolvedRuntimeCapacityPlan"] = "ResolvedRuntimeCapacityPlan"
    profile: Literal["AUTO", "LOW_MEMORY", "BALANCED", "THROUGHPUT"]
    profile_resolution: Literal["LOW_MEMORY", "BALANCED", "THROUGHPUT"]
    workload: RuntimeWorkload
    detected_logical_processors: int = Field(ge=1)
    allowed_logical_processors: int = Field(ge=1)
    total_memory_bytes: int = Field(ge=1)
    available_memory_bytes: int = Field(ge=1)
    admitted_memory_budget_bytes: int = Field(ge=1)
    reserved_logical_processors: int = Field(ge=1)
    process_logical_processor_limit: int = Field(ge=1)
    fold_workers: int = Field(ge=0)
    lightgbm_threads_per_fit: int = Field(ge=0, le=1)
    duckdb_threads: int = Field(ge=1)
    blas_threads: Literal[1] = 1
    applied_logical_processor_ids: tuple[int, ...] = ()
    estimated_peak_process_tree_rss_bytes: int = Field(ge=1)
    execution_waves: int = Field(ge=1)
    admission: Literal["ADMITTED", "REFUSED"]
    refusal_code: str | None = None
    process_tree_peak_cpu_measurement: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    process_tree_peak_rss_measurement: Literal["NOT_MEASURED"] = "NOT_MEASURED"
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def _identity_payload(values: dict[str, object]) -> dict[str, object]:
        """Exclude live machine telemetry from the sealed execution policy."""
        payload = dict(values)
        payload.pop("plan_hash", None)
        payload.pop("total_memory_bytes", None)
        payload.pop("available_memory_bytes", None)
        return payload

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal and validate resolved execution-policy fields.

        Args:
            values: Typed plan field values; the plan hash is derived here.

        Returns:
            The validated plan with its canonical policy identity.

        Raises:
            ProcessCapacityError: Admission, applied processors or the resulting identity is
                inconsistent.
        """
        provisional = cls.model_construct(**values, plan_hash="0" * 64)
        return cls(
            **values,
            plan_hash=str(
                canonical_hash(cls._identity_payload(provisional.model_dump(mode="json")))
            ),
        )

    def with_applied_logical_processors(
        self, processor_ids: tuple[int, ...]
    ) -> ResolvedRuntimeCapacityPlan:
        """Bind enforced processor IDs and reseal this execution plan.

        Args:
            processor_ids: Distinct IDs from the affinity enforcement receipt.

        Returns:
            A new validated plan retaining the other field values.

        Raises:
            ProcessCapacityError: Applied IDs repeat or their count differs from the declared limit.
        """
        values = self.model_dump(mode="python", exclude={"plan_hash"})
        values["applied_logical_processor_ids"] = processor_ids
        return type(self).create(**values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_plan(self) -> Self:
        """Validate admission, applied affinity shape and canonical policy identity.

        Returns:
            This validated plan.

        Raises:
            ProcessCapacityError: Admission/refusal, applied processors or the hash is inconsistent.
        """
        if (
            (self.admission == "ADMITTED") != (self.refusal_code is None)
            or len(set(self.applied_logical_processor_ids))
            != len(self.applied_logical_processor_ids)
            or (
                self.applied_logical_processor_ids
                and len(self.applied_logical_processor_ids) != self.process_logical_processor_limit
            )
            or self.plan_hash
            != canonical_hash(self._identity_payload(self.model_dump(mode="json")))
        ):
            raise ProcessCapacityError("process_capacity.resolved_plan_invalid")
        return self


def _memory_capacity() -> tuple[int, int]:
    if os.name == "nt":
        loader: Any = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise ProcessCapacityError("process_capacity.memory_snapshot_unavailable")

        class _Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.length = ctypes.sizeof(status)
        kernel32 = loader("kernel32", use_last_error=True)
        call = kernel32.GlobalMemoryStatusEx
        call.argtypes = [ctypes.POINTER(_Status)]
        call.restype = ctypes.c_bool
        if not call(ctypes.byref(status)):
            raise ProcessCapacityError("process_capacity.memory_snapshot_unavailable")
        return int(status.total_physical), int(status.available_physical)
    try:
        sysconf = cast(Any, getattr(os, "".join(("sys", "conf")), None))
        if sysconf is None:
            raise ProcessCapacityError("process_capacity.memory_snapshot_unavailable")
        page = int(sysconf("SC_PAGE_SIZE"))
        total = page * int(sysconf("SC_PHYS_PAGES"))
        available = page * int(sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise ProcessCapacityError("process_capacity.memory_snapshot_unavailable") from error
    return total, min(total, available)


def runtime_machine_capacity() -> RuntimeMachineCapacity:
    """Read live RAM capacity and the process's permitted logical processor IDs.

    Returns:
        The validated machine snapshot for operational capacity admission.

    Raises:
        ProcessCapacityError: RAM or affinity cannot be read or the snapshot is invalid.
    """
    total, available = _memory_capacity()
    return RuntimeMachineCapacity(
        detected_logical_processors=max(1, os.cpu_count() or 1),
        allowed_logical_processor_ids=current_allowed_logical_processors(),
        total_memory_bytes=total,
        available_memory_bytes=available,
    )


def worker_processes(cores: int) -> int:
    """How many kept workers work given `cores` may use: one core stays with the caller.

    An execution parameter, never an identity: work spread over them answers what one worker
    answers, which the owner that spreads it proves. The cores are the CPU budget's for the Task
    (the operator's; binding plan, B7); one stays with the process that hands the work out and
    writes what comes back, never fewer than one worker. Memory admits as many workers of this
    process's current size as fit in the share of available memory a runtime plan admits by
    default: a worker imports what its parent imported, so the parent's size is a cautious
    measure of one.
    """
    _total, available = _memory_capacity()
    admitted = int(available * RuntimeCapacityRequest().maximum_memory_fraction)
    return max(1, min(cores - 1, admitted // max(1, current_rss_bytes())))


def resolve_runtime_capacity_plan(
    *,
    request: ResearchRuntimeRequest,
    machine: RuntimeMachineCapacity,
    fold_count: int,
    parent_reservation_bytes: int,
    worker_reservation_bytes: int,
    workload: RuntimeWorkload = "ALPHA_FOLDS",
) -> ResolvedRuntimeCapacityPlan:
    """Resolve one workload-specific plan from OS capacity and policy."""
    if fold_count < 1 or parent_reservation_bytes < 1 or worker_reservation_bytes < 1:
        raise ProcessCapacityError("process_capacity.resolution_input_invalid")
    gib = 1024**3
    declared_memory = (
        int(request.capacity.maximum_memory_gib * gib)
        if request.capacity.maximum_memory_gib is not None
        else machine.total_memory_bytes
    )
    fraction_budget = int(machine.total_memory_bytes * request.capacity.maximum_memory_fraction)
    os_reserve = max(4 * gib, machine.total_memory_bytes // 10)
    available_budget = max(0, machine.available_memory_bytes - os_reserve)
    # An explicit absolute reservation is sealed policy. Machine total and
    # available RAM are live admission observations: they may refuse this plan,
    # never silently mint a smaller Program identity between preflight and run.
    memory_budget = (
        declared_memory if request.capacity.maximum_memory_gib is not None else fraction_budget
    )
    allowed_count = len(machine.allowed_logical_processor_ids)
    half_machine_budget = max(1, machine.detected_logical_processors // 2)
    usable_processors = min(
        half_machine_budget,
        max(0, allowed_count - request.capacity.reserved_logical_processors),
    )
    profile_memory_bytes = (
        declared_memory
        if request.capacity.maximum_memory_gib is not None
        else machine.total_memory_bytes
    )
    profile_resolution = (
        request.profile
        if request.profile != "AUTO"
        else (
            "LOW_MEMORY"
            if profile_memory_bytes <= 16 * gib
            else "THROUGHPUT"
            if profile_memory_bytes >= 128 * gib
            else "BALANCED"
        )
    )
    alpha_folds = workload == "ALPHA_FOLDS"
    # A fixed model count of one is a real applied adapter setting, while zero
    # means this workload has no model owner at all.  In particular, Portfolio
    # metadata and numerical stages must not reserve dormant Alpha threads.
    model_threads = int(request.parallelism.lightgbm_threads_per_fit) if alpha_folds else 0
    requested_duckdb_threads = request.parallelism.duckdb_threads
    requested_duckdb_limit = (
        max(1, min(4, usable_processors))
        if requested_duckdb_threads == "AUTO"
        else max(1, int(requested_duckdb_threads))
    )
    memory_worker_limit = max(
        0, (memory_budget - parent_reservation_bytes) // worker_reservation_bytes
    )
    processor_worker_limit = usable_processors // model_threads if alpha_folds else 0
    profile_limit = {
        "LOW_MEMORY": 1,
        "BALANCED": 3,
        "THROUGHPUT": 4,
    }[profile_resolution]
    requested_workers = request.parallelism.fold_workers
    worker_limit = profile_limit if requested_workers == "AUTO" else max(1, int(requested_workers))
    requested_workers_for_workload = (
        min(fold_count, worker_limit, memory_worker_limit, processor_worker_limit)
        if alpha_folds
        else 0
    )
    # Reserve one active DuckDB slot before admitting model workers.  This is
    # conservative by design: the process affinity receipt then bounds every
    # live native thread rather than relying on non-overlapping phases.
    workers = (
        min(requested_workers_for_workload, max(0, usable_processors - 1)) if alpha_folds else 0
    )
    duckdb_threads = min(
        requested_duckdb_limit,
        max(0, usable_processors - workers * model_threads),
    )
    estimated_peak = parent_reservation_bytes + workers * worker_reservation_bytes
    refusal = None
    if (
        (memory_budget < parent_reservation_bytes + worker_reservation_bytes and alpha_folds)
        or fraction_budget < memory_budget
        or available_budget < estimated_peak
    ):
        refusal = "process_capacity.memory_not_admitted"
    elif (
        workload in {"PORTFOLIO_NUMERICAL", "SCORE_FILTER_NUMERICAL"}
        and machine.total_memory_bytes <= 16 * gib
    ):
        # Metadata is admitted cold on a desktop.  A numerical claim needs a
        # contemporaneous tree receipt, not a parent-only reservation estimate.
        refusal = "process_capacity.desktop_numerical_tree_measurement_required"
    elif usable_processors < 1 or (alpha_folds and workers < 1):
        refusal = "process_capacity.processor_plan_not_admitted"
    elif duckdb_threads < 1:
        refusal = "process_capacity.duckdb_threads_not_admitted"
    elif workers * model_threads + duckdb_threads > usable_processors:
        refusal = "process_capacity.worker_model_threads_oversubscribed"
    if refusal is not None:
        workers = 0
    return ResolvedRuntimeCapacityPlan.create(
        profile=request.profile,
        profile_resolution=profile_resolution,
        workload=workload,
        detected_logical_processors=machine.detected_logical_processors,
        allowed_logical_processors=allowed_count,
        total_memory_bytes=machine.total_memory_bytes,
        available_memory_bytes=machine.available_memory_bytes,
        admitted_memory_budget_bytes=max(1, memory_budget),
        reserved_logical_processors=request.capacity.reserved_logical_processors,
        process_logical_processor_limit=max(
            1, min(usable_processors, duckdb_threads + workers * model_threads)
        ),
        fold_workers=workers,
        lightgbm_threads_per_fit=model_threads,
        duckdb_threads=max(1, duckdb_threads),
        blas_threads=1,
        estimated_peak_process_tree_rss_bytes=estimated_peak,
        execution_waves=(
            max(1, (fold_count + workers - 1) // workers) if alpha_folds and workers else 1
        ),
        admission="REFUSED" if refusal is not None else "ADMITTED",
        refusal_code=refusal,
    )


def default_logical_processor_budget() -> int:
    """Half of this machine's logical processors, and never fewer than one."""
    detected = os.cpu_count() or 1
    return max(1, detected // LOGICAL_PROCESSOR_BUDGET_DIVISOR)


def _windows_kernel32() -> Any:
    loader: Any = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise ProcessCapacityError("process_capacity.logical_processor_bound_unenforceable")
    return loader("kernel32", use_last_error=True)


def _windows_process_mask() -> int:
    """Read the affinity mask this process is currently allowed, without changing it."""
    kernel32 = _windows_kernel32()
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    process_mask = ctypes.c_size_t(0)
    system_mask = ctypes.c_size_t(0)
    get_mask = kernel32.GetProcessAffinityMask
    get_mask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    get_mask.restype = ctypes.c_bool
    if not get_mask(get_current_process(), ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise ProcessCapacityError("process_capacity.logical_processor_bound_unenforceable")
    return int(process_mask.value)


def _windows_affinity(mask: int) -> int:
    """Set and read back this process's affinity mask, returning what took effect."""
    kernel32 = _windows_kernel32()
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    set_mask = kernel32.SetProcessAffinityMask
    set_mask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    set_mask.restype = ctypes.c_bool
    if not set_mask(get_current_process(), ctypes.c_size_t(mask)):
        raise ProcessCapacityError("process_capacity.logical_processor_bound_unenforceable")
    return _windows_process_mask()


def current_allowed_logical_processors() -> tuple[int, ...]:
    """The processors this process may already run on, read from the OS.

    A container, a job object or an operator may already have narrowed this
    process before it started. That narrowing is authority from outside and must
    never be widened by a bound that only knows about ``os.cpu_count()``.
    """
    if os.name == "nt":
        mask = _windows_process_mask()
        detected = os.cpu_count() or 1
        return tuple(index for index in range(detected) if mask & (1 << index))
    getter = getattr(os, "sched_getaffinity", None)
    if not callable(getter):
        raise ProcessCapacityError("process_capacity.logical_processor_bound_unenforceable")
    return tuple(sorted(getter(0)))


def bind_process_logical_processors(
    limit: int | None = None, *, library_thread_limit: int | None = None
) -> dict[str, object]:
    """Hold this process to at most ``limit`` logical processors, or refuse.

    Returns the receipt a run should record: what the machine has, what this
    process was already permitted, the half-machine budget, the ceiling those two
    produce, what was configured, and what the operating system confirmed on
    read-back. The read-back is the point -- a bound that was requested and
    silently ignored is worse than no bound, because the run would report a
    ceiling it is not under.

    ``limit`` lowers and never raises. The effective ceiling is the smaller of
    half the machine and whatever this process was already allowed, so a
    container or job object that narrowed it further keeps that narrowing.

    Process affinity is the mechanism because it is the only one that reaches
    everything. ``OMP_NUM_THREADS`` and its siblings bind the linear-algebra
    libraries and do not touch DuckDB, which sizes its own pool from the core
    count; a database opened deep inside a workspace runtime is not reachable
    from a runner in any case. Threads created by this process inherit its
    affinity mask on both supported platforms, so a pool of any size is still
    scheduled onto the admitted processors.

    The environment variables are set as well, so the libraries that do read them
    stop oversubscribing inside the allowance rather than merely being throttled
    by it.
    """
    detected = os.cpu_count() or 1
    allowed = current_allowed_logical_processors()
    if not allowed:
        raise ProcessCapacityError("process_capacity.logical_processor_bound_unenforceable")

    # The ceiling, and it is a ceiling rather than a default. Half the machine is
    # the budget this work is held to; an existing narrower allowance is outside
    # authority and wins over it. ``limit`` may only lower what comes out of
    # this, never raise it -- the predecessor validated ``limit`` against the
    # machine's total instead, so on a 32-processor host ``--logical-processors
    # 32`` was accepted and delivered the whole machine while the help text said
    # otherwise.
    ceiling = min(default_logical_processor_budget(), len(allowed))
    configured = ceiling if limit is None else int(limit)
    if configured < 1 or configured > ceiling:
        raise ProcessCapacityError("process_capacity.logical_processor_budget_invalid")

    # A subset of what is already permitted, chosen from that set rather than
    # from ``range(configured)``: on a process pinned to processors 8..15,
    # ``range(4)`` would name processors it may not use at all.
    requested = tuple(allowed[:configured])
    if os.name == "nt":
        applied_mask = _windows_affinity(sum(1 << index for index in requested))
        applied = tuple(index for index in range(detected) if applied_mask & (1 << index))
    else:
        setter = getattr(os, "sched_setaffinity", None)
        if not callable(setter):
            raise ProcessCapacityError("process_capacity.logical_processor_bound_unenforceable")
        setter(0, set(requested))
        applied = current_allowed_logical_processors()
    if len(applied) != configured or not set(applied).issubset(allowed):
        # Requested and ignored, silently clamped, or somehow widened. Refuse
        # rather than continue under a ceiling that is not in force.
        raise ProcessCapacityError("process_capacity.logical_processor_bound_unenforceable")

    library_threads = configured if library_thread_limit is None else int(library_thread_limit)
    if library_threads < 1 or library_threads > configured:
        raise ProcessCapacityError("process_capacity.library_thread_budget_invalid")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(library_threads)

    return {
        "kind": "ProcessCapacityBound",
        "logical_processors_detected": detected,
        "logical_processors_already_allowed": len(allowed),
        "half_machine_budget": default_logical_processor_budget(),
        "effective_ceiling": ceiling,
        "configured_logical_processors": configured,
        "configured_share_of_machine": round(configured / detected, 4),
        "applied_logical_processors": list(applied),
        "enforcement": "PROCESS_AFFINITY",
        "library_thread_env": library_threads,
    }


def _system_times() -> tuple[float, float] | None:
    """Read cumulative idle and total machine processor time.

    Values use the operating system's own unit. Return None where system
    processor times cannot be read.
    """
    if os.name == "nt":
        loader: Any = getattr(ctypes, "WinDLL", None)
        if loader is None:
            return None

        class _FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        call = loader("kernel32", use_last_error=True).GetSystemTimes
        call.argtypes = [ctypes.POINTER(_FileTime)] * 3
        call.restype = ctypes.c_bool
        if not call(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None

        def value(time: _FileTime) -> float:
            return float((int(time.high) << 32) | int(time.low))

        # Kernel time includes the idle time.
        return value(idle), value(kernel) + value(user)
    try:
        fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()[1:]
        ticks = [float(field) for field in fields]
    except (OSError, IndexError, ValueError):
        return None
    return ticks[3] + (ticks[4] if len(ticks) > 4 else 0.0), sum(ticks)


def busy_logical_processors(
    interval_seconds: float = 0.25, *, processor_ids: Sequence[int] | None = None
) -> float | None:
    """Estimate the number of busy logical processors: the whole machine's, or these alone.

    The count is bounded by the processors counted and includes work outside this process.
    A process confined to some processors shares those, not the machine: with
    `processor_ids` each is read on its own and only theirs is counted (V357), so a busy
    machine outside them leaves them free. When processor-time counters are unavailable,
    the reader uses a bounded load-average fallback if the operating system exposes one.

    Args:
        interval_seconds: Sampling delay between machine processor-time reads.
        processor_ids: The logical processors to count, or None for the whole machine.

    Returns:
        Estimated busy processor count, or None when both readers are unavailable.
    """
    if processor_ids is not None:
        try:
            shares = psutil.cpu_percent(interval=interval_seconds, percpu=True)
        except (OSError, RuntimeError):
            shares = []
        if shares:
            busy = sum(float(shares[i]) for i in processor_ids if 0 <= i < len(shares)) / 100.0
            return round(min(float(len(processor_ids)), max(0.0, busy)), 1)
    detected = os.cpu_count() or 1
    first = _system_times()
    if first is None:
        getter = getattr(os, "getloadavg", None)
        return None if getter is None else min(float(detected), float(getter()[0]))
    time.sleep(interval_seconds)
    second = _system_times()
    if second is None or second[1] <= first[1]:
        return None
    idle = (second[0] - first[0]) / (second[1] - first[1])
    return round(detected * min(1.0, max(0.0, 1.0 - idle)), 1)


def process_cpu_seconds() -> float:
    """Total CPU time this process has consumed, user plus system.

    Reported beside the configured bound rather than folded into it: what a run
    was *allowed* and what it actually used are different facts, and only the
    second one can show that a ceiling was doing anything.
    """
    times = os.times()
    return float(times.user + times.system)


__all__ = [
    "LOGICAL_PROCESSOR_BUDGET_DIVISOR",
    "ProcessCapacityError",
    "ProcessResourceMonitor",
    "ProcessResourceUsage",
    "ResearchRuntimeRequest",
    "ResolvedRuntimeCapacityPlan",
    "RuntimeCapacityRequest",
    "RuntimeMachineCapacity",
    "RuntimeParallelismRequest",
    "RuntimeWorkload",
    "bind_process_logical_processors",
    "busy_logical_processors",
    "current_allowed_logical_processors",
    "current_rss_bytes",
    "default_logical_processor_budget",
    "peak_rss_bytes",
    "process_cpu_seconds",
    "resolve_runtime_capacity_plan",
    "runtime_machine_capacity",
    "worker_processes",
]
