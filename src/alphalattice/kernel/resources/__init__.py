"""Admission and lifetime of the resources an execution holds, and the register of the caches
that hold memory."""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from uuid import UUID

from alphalattice.kernel.config import Change, declare_native_startup
from alphalattice.kernel.config import current as process_config

from .caches import CacheBound, CacheRow, cache_bound, cache_register
from .port import ResourceAmounts, ResourceLease, ResourcesPort

__all__ = [
    "CacheBound",
    "CacheRow",
    "ResourceAmounts",
    "ResourceLease",
    "ResourceLeases",
    "ResourceRefused",
    "ResourcesPort",
    "bound_native_threads",
    "cache_bound",
    "cache_register",
    "carry_resource_lease",
    "cpu_width",
    "current_resource_lease",
    "logical_cpu_count",
    "native_thread_settings",
    "use_resource_lease",
    "worker_resource_lease",
    "worker_width",
]


def logical_cpu_count() -> int | None:
    """Read a machine fact for the policy resolver and diagnostic surfaces."""
    return os.cpu_count()


_CURRENT: ContextVar[ResourceLease | None] = ContextVar("task_resource_lease", default=None)


def current_resource_lease() -> ResourceLease | None:
    """Read the grant held by this Task, if any."""
    return _CURRENT.get()


def worker_width(requested: int | None = None) -> int:
    """Size the current operation's worker pool within the holder's grant.

    An operation's requested width also limits parallelism. A standalone caller
    keeps its explicit width or uses the machine's capacity when it supplies none.
    """
    lease = current_resource_lease()
    if lease is None:
        return max(1, requested or logical_cpu_count() or 1)
    available = lease.amounts.workers
    if available < 1:
        raise ResourceRefused("resources.capacity_exceeded:workers")
    return max(1, min(requested or available, available))


def cpu_width(requested: int | None = None) -> int:
    """Give a numerical or database engine the current CPU grant within configured limits."""
    lease = current_resource_lease()
    share = process_config().settings.cpu_cores
    available = lease.amounts.cpu if lease is not None else share or logical_cpu_count()
    if lease is not None and not available:
        raise ResourceRefused("resources.capacity_exceeded:cpu")
    ceiling = available or 1
    return max(1, min(requested or ceiling, share or ceiling, ceiling))


NATIVE_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def native_thread_settings() -> dict[str, str | None]:
    """Read the operator's limits apart from the process's native defaults."""
    inherited = process_config().child_environment()
    return {name: inherited.get(name) for name in NATIVE_THREAD_VARIABLES}


def bound_native_threads(requested: int) -> None:
    """Publish the CPU ceiling before native libraries read their startup environment."""
    ceiling = cpu_width(requested)
    widths = {}
    for name, value in native_thread_settings().items():
        text = (value or "").strip()
        width = int(text) if text.isdecimal() else 0
        widths[name] = str(min(ceiling, width or ceiling))
    declare_native_startup(Change("native-thread-budget", widths))


def worker_resource_lease(threads: int | None) -> ResourceLease | None:
    """Carry the Task grant to a child, restricting each spread worker to its slice."""
    grant = current_resource_lease()
    if grant is None or threads is None:
        return grant
    return ResourceLease(grant.holder, grant.amounts._replace(cpu=cpu_width(threads), workers=1))


@contextmanager
def use_resource_lease(lease: ResourceLease | None) -> Iterator[None]:
    """Make the holder's already granted resources available in this execution context."""
    token = _CURRENT.set(lease)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def carry_resource_lease[T](call: Callable[[], T]) -> Callable[[], T]:
    """Carry the same grant to the runner's worker thread."""
    lease = current_resource_lease()

    def run() -> T:
        with use_resource_lease(lease):
            return call()

    return run


class ResourceRefused(RuntimeError):
    """A resource request that cannot be admitted from the available budget."""


class ResourceLeases:
    """Track available resources and active grants in one in-memory owner."""

    def __init__(self, budget: ResourceAmounts) -> None:
        """Start with a budget resolved by policy at the Host boundary."""
        _validate(budget)
        self._budget = self._available = budget
        self._leases: dict[str | UUID, ResourceLease] = {}
        self._lock = Lock()

    @property
    def capacity(self) -> ResourceAmounts:
        """The policy ceiling currently in force, including resources already held."""
        with self._lock:
            return self._budget

    @property
    def available(self) -> ResourceAmounts:
        """Report the quantities currently available to the next grant."""
        with self._lock:
            return self._available

    def configure(self, budget: ResourceAmounts) -> None:
        """Apply a policy update while preserving the quantities existing holders own."""
        _validate(budget)
        with self._lock:
            self._available = ResourceAmounts(
                *(
                    available + limit - previous
                    for available, limit, previous in zip(
                        self._available, budget, self._budget, strict=True
                    )
                )
            )
            self._budget = budget

    def __len__(self) -> int:
        """Count the active grants without visiting them."""
        with self._lock:
            return len(self._leases)

    def holders(self) -> tuple[str | UUID, ...]:
        """Report the holders in admission order for the Host's status read."""
        with self._lock:
            return tuple(self._leases)

    def widest_cpu(self) -> int | None:
        """Report the CPU width the process's shared reader engines can use."""
        with self._lock:
            return max((lease.amounts.cpu for lease in self._leases.values()), default=None)

    def grant(self, holder: str | UUID, request: ResourceAmounts) -> ResourceLease:
        """Debit all seven quantities atomically without consulting existing grants."""
        _validate(request)
        with self._lock:
            if holder in self._leases:
                raise ResourceRefused("resources.holder_already_leased")
            remaining = ResourceAmounts(
                *(left - taken for left, taken in zip(self._available, request, strict=True))
            )
            for name, left in zip(ResourceAmounts._fields, remaining, strict=True):
                if left < 0:
                    raise ResourceRefused(f"resources.capacity_exceeded:{name}")
            lease = ResourceLease(holder, request)
            self._leases[holder] = lease
            self._available = remaining
            return lease

    def release(self, lease: ResourceLease) -> None:
        """Credit only the current grant; repeated or stale releases are harmless."""
        with self._lock:
            if self._leases.get(lease.holder) is lease:
                self._available = ResourceAmounts(
                    *(
                        left + taken
                        for left, taken in zip(self._available, lease.amounts, strict=True)
                    )
                )
                del self._leases[lease.holder]

    def process_exited(self, lease: ResourceLease) -> None:
        """Use the Host's observed exit as the release event for a process grant."""
        self.release(lease)


def _validate(amounts: ResourceAmounts) -> None:
    for name, value in zip(ResourceAmounts._fields, amounts, strict=True):
        if type(value) is not int or value < 0:
            raise ValueError(f"resources.invalid_quantity:{name}")
