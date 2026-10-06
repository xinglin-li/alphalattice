"""Port used by Feature orchestration to persist one bounded batch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal, Protocol

import duckdb

from alphalattice.foundation.feature_engine.storage.contracts import FeatureMaterializationWrite

FeaturePersistenceTimingSink = Callable[[str, float], None]
"""Where an implementation reports one persistence stage and its duration.

Declared on the port because the storage layer has carried the seam for some
time with no caller supplying one: the stage timings were measured and
discarded, so attributing a slow build meant re-deriving from outside numbers
the process already had.
"""


FeatureClosureDisposition = Literal[
    "CLOSURE_ABSENT",
    "MEMBERSHIP_INCOMPLETE",
    "TRANSITION_RECOVERY_PENDING",
    "CLOSURE_AUTHORITY_UNAVAILABLE",
    "MEMBERSHIP_COMPLETE",
]
"""What the durable base closure is, from the owner that maintains it.

Named on the port rather than inferred by callers. ``assert_ready`` answers
whether work may proceed and nothing else, so a caller that classified its
exception was choosing one name for four situations that recover differently --
and the one it chose, "a transition is pending", is the only one of the four that
is safe to reconcile automatically.
"""


_ADMITTED_WRITES = 8
_ADMITTED_COLUMN_WRITES = 64


def admitted_writes(factor_count: int) -> int:
    """How many listings' writes one closure transition admits for an axis this wide.

    Every transition re-seals the closure digest over every row persisted so far (rows of a
    new listing sit at every session, so the sealed prefix cannot be extended), and commits
    once; over a 466-name first build eight per transition halves both against four, for one
    more listing block of rows held while it is sealed. A column catalog's row holds one value
    (V92), so its transitions take eight times as many listings: an activation's column seals
    its closure eight times rather than fifty-nine.

    Args:
        factor_count: The width of the catalog's factor axis.

    Returns:
        The admitted number of writes per transition.
    """
    return _ADMITTED_COLUMN_WRITES if factor_count == 1 else _ADMITTED_WRITES


class FeatureMaterializationPersistence(Protocol):
    """Persist Feature batches and report the durable closure disposition."""

    def reconcile_pending(
        self,
        catalog_hash: str,
        *,
        connection: duckdb.DuckDBPyConnection,
    ) -> None:
        """Reconcile a pending closure transition on the shared connection."""
        ...

    def persist_batch(
        self,
        writes: Sequence[FeatureMaterializationWrite],
        *,
        connection: duckdb.DuckDBPyConnection,
        timing_sink: FeaturePersistenceTimingSink | None = None,
    ) -> tuple[str, ...]:
        """Persist a bounded batch and return its materialization receipts."""
        ...

    def begin_first_loads(
        self,
        catalog_hashes: Sequence[str],
        *,
        idempotency_key: str,
        observed_at: datetime,
        connection: duckdb.DuckDBPyConnection,
    ) -> tuple[str, ...]:
        """Begin the first load of each part that holds no row under its genesis head (V92)."""
        ...

    def complete_first_loads(
        self, catalog_hashes: Sequence[str], *, connection: duckdb.DuckDBPyConnection
    ) -> None:
        """Advance each first-loaded part's head to the digest of the rows it wrote."""
        ...

    def assert_ready(self, catalog_hash: str) -> None:
        """Refuse work when the catalog's durable base closure is not ready."""
        ...

    def closure_disposition(
        self, catalog_hash: str, *, expected_listing_ids: Sequence[str]
    ) -> FeatureClosureDisposition:
        """Classify closure readiness for the expected listing membership."""
        ...


__all__ = ["FeatureClosureDisposition", "FeatureMaterializationPersistence", "admitted_writes"]
