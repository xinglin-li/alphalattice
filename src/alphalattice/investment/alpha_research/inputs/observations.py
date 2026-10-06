"""Safe, invocation-local observations for Alpha numerical array reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

type AlphaArrayReadOperation = Literal[
    "available_development_sessions",
    "read_development_sessions",
    "read_development",
    "feature_batches",
]


@dataclass(frozen=True, slots=True)
class AlphaArrayReadObservation:
    """Record one array-read operation with nonnegative row, chunk and byte counts."""

    operation: AlphaArrayReadOperation
    row_count: int
    chunk_count: int
    byte_count: int

    def __post_init__(self) -> None:
        """Require nonnegative measured row, chunk and byte counts.

        Raises:
            ValueError: A declared read count is negative.
        """
        if min(self.row_count, self.chunk_count, self.byte_count) < 0:
            raise ValueError("alpha_research.array_read_observation_invalid")


class AlphaArrayReadObserver(Protocol):
    """Receive array-read observations without owning numerical calculation or publication."""

    def observe(self, observation: AlphaArrayReadObservation, /) -> None:
        """Receive one typed read observation from the deterministic array owner.

        Args:
            observation: Operation and nonnegative resource counts to observe.
        """
        ...


class AlphaArrayReadLedger:
    """Collect safe counters without retaining rows or array contents."""

    def __init__(self) -> None:
        """Initialize an empty in-memory ledger of array-read observations."""
        self._observations: list[AlphaArrayReadObservation] = []

    def observe(self, observation: AlphaArrayReadObservation, /) -> None:
        """Append one typed observation in encounter order.

        Args:
            observation: Resource-count observation retained by the ledger.
        """
        self._observations.append(observation)

    @property
    def observations(self) -> tuple[AlphaArrayReadObservation, ...]:
        """Read an immutable tuple of the observations retained so far.

        Returns:
            Observation records in append order; the tuple does not expose the mutable list.
        """
        return tuple(self._observations)

    def count(self, operation: AlphaArrayReadOperation) -> int:
        """Count observation records for one operation kind.

        Args:
            operation: Typed read operation whose events are counted.

        Returns:
            Number of matching records, rather than the sum of their row/chunk/byte counts.
        """
        return sum(value.operation == operation for value in self._observations)

    def total(self, field: Literal["row_count", "chunk_count", "byte_count"]) -> int:
        """Sum one declared resource-count field across retained read observations.

        Args:
            field: row_count, chunk_count or byte_count to aggregate.

        Returns:
            Integer total for the selected count field.
        """
        return sum(int(getattr(value, field)) for value in self._observations)


__all__ = [
    "AlphaArrayReadLedger",
    "AlphaArrayReadObservation",
    "AlphaArrayReadObserver",
    "AlphaArrayReadOperation",
]
