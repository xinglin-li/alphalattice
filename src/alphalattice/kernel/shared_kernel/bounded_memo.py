"""Bounded derived values; owners recompute exactly on a miss, never evict admission state."""

from collections import OrderedDict
from collections.abc import Hashable, Iterator, MutableMapping
from os import cpu_count
from threading import RLock
from typing import Final


class BoundedMemo[K: Hashable, V](MutableMapping[K, V]):
    """A locked LRU with an owner capacity; the default admits 1024 entries per logical CPU."""

    def __init__(self, *, capacity: int | None = None) -> None:
        """Use the owner's positive entry capacity, or the machine-derived default."""
        self.capacity: Final = max(1, capacity if capacity is not None else self.default_capacity())
        self._values: OrderedDict[K, V] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def default_capacity() -> int:
        """Return the metadata-entry allowance derived from this machine's logical CPUs."""
        return max(1, cpu_count() or 1) * 1024

    def __getitem__(self, key: K) -> V:
        """Return a held value and make its key most recently used."""
        with self._lock:
            value = self._values[key]
            self._values.move_to_end(key)
            return value

    def __setitem__(self, key: K, value: V) -> None:
        """Keep a derived value and forget the least recently used excess."""
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)

    def __delitem__(self, key: K) -> None:
        """Forget one derived value without changing its owner's inputs."""
        with self._lock:
            del self._values[key]

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
        with self._lock:
            self._values.clear()
