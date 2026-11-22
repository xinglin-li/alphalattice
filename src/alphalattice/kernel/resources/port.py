"""Resource budgets and leases."""

from dataclasses import dataclass
from typing import NamedTuple, Protocol
from uuid import UUID


class ResourceAmounts(NamedTuple):
    """Seven resource quantities in the units declared by the Host's policy."""

    cpu: int = 0
    memory_bytes: int = 0
    workers: int = 0
    disk_bytes: int = 0
    connections: int = 0
    network: int = 0
    spend: int = 0


@dataclass(frozen=True, slots=True, eq=False)
class ResourceLease:
    """The exact grant to a holder, whose identity stays valid across a late exit notice."""

    holder: str | UUID
    amounts: ResourceAmounts


class ResourcesPort(Protocol):
    """The operations the Host uses to admit and retire resource holders."""

    def grant(self, holder: str | UUID, request: ResourceAmounts) -> ResourceLease:
        """Take the whole request from the available budget, or raise a named refusal."""
        ...

    def release(self, lease: ResourceLease) -> None:
        """Return the current grant when its holder finishes or is cancelled."""
        ...

    def process_exited(self, lease: ResourceLease) -> None:
        """Reclaim the exact grant after the Host observes its holder's exit."""
        ...
