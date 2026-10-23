"""Bounded derived values; owners recompute exactly on a miss, never evict admission state."""

from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, MutableMapping
from os import cpu_count, getpid
from threading import RLock
from typing import Final
from uuid import uuid4

from alphalattice.kernel.shared_kernel.spans import SpanLedger, active_ledger


class BoundedMemo[K: Hashable, V](MutableMapping[K, V]):
    """A locked LRU with an owner capacity; the default admits 1024 entries per logical CPU.

    cache_id enables phase telemetry; its instance ref is operational, never scientific identity.
    """

    def __init__(
        self,
        *,
        capacity: int | None = None,
        cache_id: str | None = None,
        retained_size_estimate: Callable[[K, V], int] | None = None,
    ) -> None:
        """Retained bytes (estimate) measure owner representation, never Host commit.

        Replace resized mutable values to refresh insertion/replacement estimates.
        """
        self.capacity: Final = max(1, capacity if capacity is not None else self.default_capacity())
        self._values: OrderedDict[K, V] = OrderedDict()
        self._lock = RLock()
        self.cache_id, self._retained_size_estimate = cache_id, retained_size_estimate
        self._instance_ref = uuid4().hex if cache_id is not None else ""
        self._sizes: dict[K, int | None] = {}
        self._retained = self._unknown = self._sequence = 0

    def _forget_size(self, key: K) -> None:
        size = self._sizes.pop(key, 0)
        if size is None:
            self._unknown -= 1
        else:
            self._retained -= size

    def _observe(self, ledger: SpanLedger | None, **counts: int) -> None:
        if ledger is not None and self.cache_id is not None:
            observed = self._retained_size_estimate is not None and self._unknown == 0
            try:
                ledger.observe_cache(
                    dict(
                        cache_id=self.cache_id,
                        instance_ref=self._instance_ref,
                        source_pid=getpid(),
                        outcome_scope="LOOKUP",
                        sequence=self._sequence,
                        retained_bytes_estimate=self._retained if observed else None,
                        retained_size_status="OBSERVED" if observed else "UNOBSERVED",
                        missing=[] if observed else ["CACHE_RETAINED_BYTES_UNOBSERVED"],
                        hits=counts.get("hits", 0),
                        misses=counts.get("misses", 0),
                        capacity_evictions=counts.get("capacity_evictions", 0),
                    )
                )
            except Exception:
                ledger.cache_observation_failed()

    @staticmethod
    def default_capacity() -> int:
        """Return the metadata-entry allowance derived from this machine's logical CPUs."""
        return max(1, cpu_count() or 1) * 1024

    def __getitem__(self, key: K) -> V:
        """Return a held value and make its key most recently used."""
        ledger = active_ledger() if self.cache_id is not None else None
        with self._lock:
            try:
                value = self._values[key]
            except KeyError:
                self._observe(ledger, misses=1)
                raise
            self._values.move_to_end(key)
            self._observe(ledger, hits=1)
            return value

    def __setitem__(self, key: K, value: V) -> None:
        """Keep a derived value and forget the least recently used excess."""
        ledger = active_ledger() if self.cache_id is not None else None
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            if self.cache_id is not None:
                self._forget_size(key)
                try:
                    size = (
                        self._retained_size_estimate(key, value)
                        if self._retained_size_estimate
                        else None
                    )
                    size = size if type(size) is int and size >= 0 else None
                except Exception:
                    size = None
                self._sizes[key] = size
                self._unknown += int(size is None)
                self._retained += size or 0
            evicted = 0
            while len(self._values) > self.capacity:
                old, _ = self._values.popitem(last=False)
                self._forget_size(old)
                evicted += 1
            self._sequence += 1
            self._observe(ledger, capacity_evictions=evicted)

    def __delitem__(self, key: K) -> None:
        """Forget one derived value without changing its owner's inputs."""
        ledger = active_ledger() if self.cache_id is not None else None
        with self._lock:
            del self._values[key]
            self._forget_size(key)
            self._sequence += 1
            self._observe(ledger)

    def __iter__(self) -> Iterator[K]:
        """Iterate a stable key snapshot."""
        with self._lock:
            return iter(tuple(self._values))

    def __len__(self) -> int:
        """Count held entries."""
        with self._lock:
            return len(self._values)

    def clear(self) -> None:
        """Forget retained values in one locked pass."""
        ledger = active_ledger() if self.cache_id is not None else None
        with self._lock:
            self._values.clear()
            self._sizes.clear()
            self._retained = self._unknown = 0
            self._sequence += 1
            self._observe(ledger)
