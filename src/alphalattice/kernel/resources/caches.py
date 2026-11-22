"""The cache register: each hand-written cache's name, owner, bound and invalidation key.

A cache holds memory the process does not account for elsewhere, so each one is a resource with
a bound. Its rows are config/registries/caches.json's entries, which this module reads as
given: the register's composer passes them in, as the kernel reads no configuration itself. A
row's bound is a count with its unit ("128 entries", "512 MiB", "1 projection"), an owner's
capacity ("BoundedMemo capacity", "capacity 8"), or a budget's admission; "unbounded", an empty
bound and an open-ended one ("one entry per key in flight") name no bound.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

_COUNT = re.compile(r"(\d+)\s+(\S.*)")
_CAPACITY = re.compile(r"(?:boundedmemo\b.*\bcapacity\b|\bcapacity\s+\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CacheBound:
    """What holds a cache's size: a count of a unit, or an owner's capacity or admission."""

    unit: str
    count: int | None = None


@dataclass(frozen=True)
class CacheRow:
    """One registered cache: its name, owner, bound (None when it names none) and the
    invalidation key that ends an entry."""

    name: str
    owner: str
    bound: CacheBound | None
    key: str


def cache_bound(text: object) -> CacheBound | None:
    """The bound a row's ``size_bound`` names, or None when it names none."""
    if not isinstance(text, str) or not text.strip() or text.strip().lower() == "unbounded":
        return None
    stripped = text.strip()
    if counted := _COUNT.fullmatch(stripped):
        count = int(counted.group(1))
        return CacheBound(counted.group(2), count) if count > 0 else None
    if _CAPACITY.search(stripped):
        return CacheBound("owner capacity")
    if stripped.lower().endswith("admission"):
        return CacheBound("budget admission")
    return None


def cache_register(entries: Mapping[str, Mapping[str, object]]) -> dict[str, CacheRow]:
    """Each registered cache by its row key, read from the registry's entries."""
    return {
        key: CacheRow(
            name=key,
            owner=str(row.get("owner", "")),
            bound=cache_bound(row.get("size_bound")),
            key=str(row.get("invalidation_rule", "")),
        )
        for key, row in entries.items()
    }


__all__ = ["CacheBound", "CacheRow", "cache_bound", "cache_register"]
