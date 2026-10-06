"""Plans kept by their exact hash until they expire, sealed on disk (V105, V525).

An answer that names a plan is saved by its reader and sent again later, possibly after the Host
restarted or past the memory's bound. Each owner keeps every plan it answered under its hash,
sealed in the workspace's `runtime/`, so a run or a confirm finds it. The plan model checks its
own hash when the sealed record is read back. Expiry keeps a stale plan readable but never
runnable; an expired file goes when the next plan is sealed. An owner that keeps one plan
(`single`) keeps its newest: sealing it removes every other.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from pydantic import BaseModel

PREVIEW_CAPACITY = 8
"""Unadmitted plans one owner keeps in memory at a time, per workspace session."""

PREVIEW_TTL = timedelta(minutes=60)
"""How long a plan stays runnable. Afterwards its run or confirm is refused toward a new plan."""

PLAN_PREVIEWS_DIRECTORY = "plan-previews"
"""Under the workspace's `runtime/`: one folder of sealed plans per owner (V525)."""


@dataclass(frozen=True, slots=True)
class RetainedPreview[P: BaseModel]:
    """An exact plan kept with its caller and its bounded expiry."""

    plan: P
    previewed_at: datetime
    expires_at: datetime
    caller: str


class PreviewRegistry[P: BaseModel]:
    """Plans addressable by their exact hash, sealed on disk until they expire.

    The durable Task input remains the only record of a plan that was admitted. Memory holds
    the newest plans, so competing plans from two actors stay addressable side by side.

    The Local Web service answers requests on several threads, so every read and write of the
    table holds one lock: a plan arriving while another evicts can neither corrupt the table nor
    observe a half-applied eviction. Recency is the table's insertion order (a refreshed hash
    moves to the end), so eviction takes the first entry and never scans timestamps that can tie.
    """

    def __init__(
        self,
        *,
        model: type[P],
        clock: Callable[[], datetime],
        capacity: int = PREVIEW_CAPACITY,
        ttl: timedelta = PREVIEW_TTL,
        root: Path | None = None,
        single: bool = False,
        hash_field: str = "plan_hash",
    ) -> None:
        """Create a bounded, expiring registry, its plans sealed on disk when it has a root.

        Args:
            model: The sealed plan model, which checks its own hash when it is read back.
            clock: Explicit observed-time source.
            capacity: Positive maximum count of plans held in memory.
            ttl: Positive plan lifetime.
            root: Optional durable plan directory.
            single: Keep only the newest plan: sealing one removes every other.
            hash_field: The plan model's own identity field (`content_hash` for a data update).

        Raises:
            ValueError: Capacity or lifetime is not positive.
        """
        if capacity < 1 or ttl <= timedelta():
            raise ValueError("research_experiment.preview_registry_bounds_invalid")
        self._model, self._clock, self._capacity, self._ttl = model, clock, capacity, ttl
        self._entries: dict[str, RetainedPreview[P]] = {}
        self._lock = Lock()
        self._root, self._single, self._field = root, single, hash_field

    def _hash(self, plan: P) -> str:
        return str(getattr(plan, self._field))

    @property
    def retention(self) -> str:
        """Where a plan lives until it expires."""
        return "IN_MEMORY_UNTIL_EXPIRY_OR_RESTART" if self._root is None else "ON_DISK_UNTIL_EXPIRY"

    def remember(self, plan: P, *, caller: str = "HOST") -> RetainedPreview[P]:
        """Keep one plan; the same plan again refreshes its expiry.

        Args:
            plan: The exact sealed plan.
            caller: Who planned it.

        Returns:
            The kept plan and its expiry.
        """
        now = self._clock()
        entry = RetainedPreview(
            plan=plan, previewed_at=now, expires_at=now + self._ttl, caller=caller
        )
        with self._lock:
            if self._single:
                self._entries.clear()
            self._hold(entry)
            if self._root is not None:
                self._seal(entry, now)
        return entry

    def get(self, plan_hash: str) -> RetainedPreview[P] | None:
        """Read an exact plan, reopening its sealed record when it is not in memory.

        Args:
            plan_hash: Exact plan identity.

        Returns:
            The kept plan, or None when absent; expiry is checked separately.
        """
        with self._lock:
            entry = self._entries.get(plan_hash)
            if entry is None and self._root is not None:
                entry = self._sealed(plan_hash)
                if entry is not None:
                    self._hold(entry)
            return entry

    def runnable(self, plan_hash: str) -> P | None:
        """The exact plan when it is kept and has not expired, else None.

        Args:
            plan_hash: Exact plan identity.

        Returns:
            The plan its run or confirm may take.
        """
        entry = self.get(plan_hash)
        return None if entry is None or self.expired(entry) else entry.plan

    def _hold(self, entry: RetainedPreview[P]) -> None:
        key = self._hash(entry.plan)
        self._entries.pop(key, None)
        self._entries[key] = entry
        while len(self._entries) > self._capacity:
            del self._entries[next(iter(self._entries))]

    def _seal(self, entry: RetainedPreview[P], now: datetime) -> None:
        assert self._root is not None
        self._root.mkdir(parents=True, exist_ok=True)
        key = self._hash(entry.plan)
        for path in self._root.glob("*.json"):
            if self._single and path.stem != key:
                path.unlink(missing_ok=True)
                continue
            try:
                expires_at = datetime.fromisoformat(
                    json.loads(path.read_text(encoding="utf-8"))["expires_at"]
                )
            except (OSError, ValueError, KeyError, TypeError):
                expires_at = now
            if now >= expires_at:
                path.unlink(missing_ok=True)
        path = self._root / f"{key}.json"
        staged = path.with_name(f"{path.name}.partial")
        staged.write_text(
            json.dumps(
                {
                    "plan": entry.plan.model_dump(mode="json"),
                    "previewed_at": entry.previewed_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat(),
                    "caller": entry.caller,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(staged, path)

    def _sealed(self, plan_hash: str) -> RetainedPreview[P] | None:
        """A sealed plan, or none where it is absent or does not read as sealed."""
        assert self._root is not None
        try:
            document = json.loads((self._root / f"{plan_hash}.json").read_text(encoding="utf-8"))
            plan = self._model.model_validate(document["plan"])
            if self._hash(plan) != plan_hash:
                return None
            return RetainedPreview(
                plan=plan,
                previewed_at=datetime.fromisoformat(document["previewed_at"]),
                expires_at=datetime.fromisoformat(document["expires_at"]),
                caller=str(document["caller"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def expired(self, entry: RetainedPreview[P]) -> bool:
        """Compare the current observed time with one kept plan's expiry.

        Args:
            entry: Explicit kept plan.

        Returns:
            Whether its expiry has been reached.
        """
        return self._clock() >= entry.expires_at

    def waiting(self) -> tuple[RetainedPreview[P], ...]:
        """Every plan still runnable: the sealed ones, else those in memory.

        Returns:
            The plans whose expiry is still ahead, newest first.
        """
        with self._lock:
            held = dict(self._entries)
        if self._root is not None:
            for path in self._root.glob("*.json"):
                sealed = self._sealed(path.stem)
                if sealed is not None:
                    held.setdefault(self._hash(sealed.plan), sealed)
        now = self._clock()
        return tuple(
            sorted(
                (entry for entry in held.values() if now < entry.expires_at),
                key=lambda entry: (entry.previewed_at, self._hash(entry.plan)),
                reverse=True,
            )
        )

    @property
    def capacity(self) -> int:
        """Read the configured upper bound of the plans held in memory.

        Returns:
            Positive configured capacity.
        """
        return self._capacity

    def __len__(self) -> int:
        """Count the plans held in memory under the registry lock.

        Returns:
            Current held entry count.
        """
        with self._lock:
            return len(self._entries)


__all__ = [
    "PLAN_PREVIEWS_DIRECTORY",
    "PREVIEW_CAPACITY",
    "PREVIEW_TTL",
    "PreviewRegistry",
    "RetainedPreview",
]
