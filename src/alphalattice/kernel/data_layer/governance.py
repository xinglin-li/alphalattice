"""Storage governance on the log: the four authorities over what the data root keeps, in one place.

- **Live references** (`live_ref`): what holds an artifact's body, written only by the projection
  (project.py) from its root's own events. Today: the workspace that keeps it (`artifact.held`,
  or an artifact produced `held`: arrays and execution outcomes), released when the workspace is
  deleted (`workspace.status_changed`); a person's pin. The book, kept-trial and publication
  roots are declared and projected; their emitters land with the rows that first put their
  objects on the log (a book's holdings with row 8a, a trial's keep with row 9a, a publication
  when its outputs go on the log); a Task holds nothing its workspace does not.
- **Pins**: a `pin` root, held by `artifact.pinned` and let go by `artifact.unpinned`, both
  durable and naming the pin (`pin_id`).
- **Retention windows**: UNHELD here, the age after which a body no live reference holds is
  collected (log.sql's rule); the catalog's KEEP_SNAPSHOTS and SETTLE_FILES (store.py); a derived
  file body's UNREAD (workspace_runtime/panel_reads.py).
- **Deletion and forgetting**: one collector, `StorePort.collect`, over every family in the data
  root: derived partitions no workspace roots, and artifact bodies no live reference holds past
  UNHELD (`unheld`). It records `artifact.collected` and queues the body in one log unit, and
  removes the body after, so a crash over-retains, never under-retains; the Host's start removes
  what the queue still names (store.remove_bodies). Forgetting is `record.forgotten`, which
  removes one event's text and never a text the log rebuilds from (project.MACHINE_TEXT).
- **Backup**: data_layer.backup copies the data root holding `COPYING`, which every collection
  holds too, so no body its snapshot names goes during the copy; a generation then holds its
  own copy of every body it names.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import timedelta
from typing import Any, Final

UNHELD: Final = timedelta(days=7)
"""How long after its production a body no live reference holds is kept: log.sql's rule, so a
result a caller has not yet rooted survives its producer's next steps."""
COPYING: Final = threading.RLock()
"""Held by a backup while it copies the data root and by every collection: a collection waits
for the copy, and a copy for the collection. The Host is the one process that writes its data
root (the log's lock), so a lock in it is enough."""
BODY_HANDLES: Final = {"blob": ("blobs", "blob"), "table": ("results", "rows")}
"""Where a collected body outside the log waits in the deletion queue until it is gone, by the
artifact's storage: its handle's folder and the queue's kind. An inline body goes with its
`artifact.collected`."""

type Read = Callable[[str, tuple[Any, ...]], list[Any]]


def unheld(read: Read, before_us: int) -> list[tuple[str, str]]:
    """The artifacts with a stored body, produced before `before_us`, that no live reference
    holds: (artifact_id, storage), those the collector removes."""
    rows = read(
        "SELECT a.artifact_id, a.storage FROM artifact a JOIN event e ON e.seq = a.produced_seq"
        " WHERE a.body_state = 'present' AND a.storage <> 'none' AND e.recorded_us < ?"
        " AND NOT EXISTS (SELECT 1 FROM live_ref r WHERE r.artifact_id = a.artifact_id)"
        " ORDER BY a.produced_seq",
        (before_us,),
    )
    return [(str(row[0]), str(row[1])) for row in rows]


__all__ = ["BODY_HANDLES", "COPYING", "UNHELD", "unheld"]
