"""One live campaign's budget, durable across the processes that spend it.

A campaign is admitted once, with its declared totals, and spent by
however many processes resume it; what each attempt consumed is written
before the next one starts, in an append-only ledger whose lines are
hash-chained, so a process that resumes the campaign receives only what
remains and a line that was edited or dropped is a refusal, never a fresh
allowance. The ledger is the campaign's own record beside the artifact
store; it is not a counter a caller may set.

The crash boundary is the reservation: an attempt is reserved -- one
attempt, the most bytes the request may read -- before it is made and
settled with what it actually read afterwards. A reservation the process
never settled (it died mid-transfer) counts at its reserved size when the
campaign is resumed, never at zero.

What the record proves, and what it does not. The chain proves every line
present is the line that was written, in order; the head file beside the
ledger (`<ledger>.head`, the last line's hash) proves the ledger was not
cut short -- a ledger one line ahead of its head is the one crash window
(a line appended, the head not yet moved) and is accepted, one behind is
refused. Deleting both files removes the campaign's record altogether;
the next admission under that id is then visibly a first admission
(`resumed` is false), not a resumption -- the record cannot refuse its
own absence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from alphalattice.control.workspace_runtime.lock import WorkspaceLock
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError

CAMPAIGN_LEDGER_ROOT = "sec-campaigns"
"""The directory under the artifact root that holds one ledger per campaign."""

_GENESIS = "0" * 64
_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
"""A campaign id is one path component: letters, digits, dot, underscore and
hyphen, opening with a letter or digit. Nothing that a filesystem could read
as a separator, a drive, a parent or a hidden file."""


def require_campaign_id(value: str) -> str:
    """The id as given, or the refusal `sec_campaign_id_invalid`."""
    if _CAMPAIGN_ID.match(value) is None:
        raise ValueError("alternative_evidence.sec_campaign_id_invalid")
    return value


@dataclass(frozen=True, slots=True)
class SecCampaignDeclaration:
    """The totals one campaign is admitted with; they never move."""

    campaign_id: str
    maximum_total_attempts: int
    maximum_total_response_bytes: int
    maximum_body_resources: int
    maximum_document_bytes: int

    def __post_init__(self) -> None:
        """Validate the campaign identifier and positive budget totals."""
        require_campaign_id(self.campaign_id)
        for value in (
            self.maximum_total_attempts,
            self.maximum_total_response_bytes,
            self.maximum_body_resources,
            self.maximum_document_bytes,
        ):
            if value < 1:
                raise ValueError("alternative_evidence.sec_campaign_budget_invalid")


class SecCampaignLedger:
    """Own one append-only campaign ledger and its lock.

    The append-only, hash-chained ledger of one campaign, owned by one
    holder at a time.

    Opening takes the ledger's OS lock (`<ledger>.lock`, the workspace's
    own non-blocking one-byte lock) and holds it until `close`: a second
    holder of the same ledger is refused before any request could leave
    (`sec_campaign_ledger_held`), so two processes can never spend one
    balance. The lock is advisory ownership, not a database transaction,
    and nothing waits on the network under it. Every append also proves
    the file still ends where this holder last left it
    (`sec_campaign_ledger_stale` otherwise), so a writer that got past the
    lock cannot make this one append on a stale balance.
    """

    def __init__(self, path: Path, declaration: SecCampaignDeclaration) -> None:
        """Bind a campaign declaration to its durable ledger path."""
        self.path = path
        self.declaration = declaration
        self._lines: list[dict[str, object]] = []
        self._previous = _GENESIS
        self._reserved: dict[int, dict[str, object]] = {}
        self._settled_bytes = 0
        self._attempts = 0
        self._bodies: set[str] = set()
        self._lock: WorkspaceLock | None = None
        self.resumed = False

    @property
    def held(self) -> bool:
        """Report whether this process owns the ledger lock."""
        return self._lock is not None

    def close(self) -> None:
        """Release the campaign's ownership; the ledger stays as written."""
        lock = self._lock
        self._lock = None
        if lock is not None:
            lock.release()

    def __enter__(self) -> SecCampaignLedger:
        """Return the locked ledger for a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the ledger lock when the context ends."""
        self.close()

    # ------------------------------------------------------------- opening
    @classmethod
    def open(
        cls,
        path: Path,
        *,
        declaration: SecCampaignDeclaration,
        admitted_at: datetime | None = None,
    ) -> SecCampaignLedger:
        """Admit a new campaign at `path`, or resume the one already there.

        A ledger that exists is read whole and its chain verified; its
        admission must state exactly this declaration -- another set of
        totals is another campaign, which needs its own admission and its
        own ledger. A ledger that does not exist is created with the
        admission line.
        """
        require_campaign_id(declaration.campaign_id)
        ledger = cls(path, declaration)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ledger._lock = WorkspaceLock(path.with_name(path.name + ".lock")).acquire()
        except WorkspaceConflictError as error:
            raise ValueError("alternative_evidence.sec_campaign_ledger_held") from error
        try:
            if path.exists():
                ledger._read()
                ledger.resumed = True
                return ledger
            ledger._append(
                {
                    "kind": "ADMISSION",
                    "campaign_id": declaration.campaign_id,
                    "maximum_total_attempts": declaration.maximum_total_attempts,
                    "maximum_total_response_bytes": declaration.maximum_total_response_bytes,
                    "maximum_body_resources": declaration.maximum_body_resources,
                    "maximum_document_bytes": declaration.maximum_document_bytes,
                    "admitted_at": (admitted_at or datetime.now(UTC)).isoformat(),
                }
            )
            return ledger
        except BaseException:
            ledger.close()
            raise

    def _read(self) -> None:
        raw = self.path.read_bytes().decode("utf-8")
        previous = _GENESIS
        hashes: list[str] = []
        for number, text in enumerate(raw.split("\n"), start=1):
            if not text:
                continue
            try:
                line = json.loads(text)
            except ValueError as error:
                raise ValueError("alternative_evidence.sec_campaign_ledger_invalid") from error
            if not isinstance(line, dict) or line.get("previous") != previous:
                raise ValueError("alternative_evidence.sec_campaign_ledger_invalid")
            recorded = line.get("line_hash")
            if recorded != _line_hash(previous, line):
                raise ValueError("alternative_evidence.sec_campaign_ledger_invalid")
            if number == 1:
                if line.get("kind") != "ADMISSION" or not self._matches(line):
                    raise ValueError("alternative_evidence.sec_campaign_declaration_mismatch")
            elif line.get("kind") == "ADMISSION":
                raise ValueError("alternative_evidence.sec_campaign_ledger_invalid")
            self._apply(line)
            self._lines.append(line)
            previous = str(recorded)
            hashes.append(previous)
        if not self._lines:
            raise ValueError("alternative_evidence.sec_campaign_ledger_invalid")
        head = (
            self._head_path.read_text(encoding="utf-8").strip() if self._head_path.exists() else ""
        )
        # The head names the last line, or the one before it when the
        # process died between appending a line and moving the head.
        if head not in {hashes[-1], hashes[-2] if len(hashes) > 1 else ""}:
            # Shorter than its head says, or a head from another record:
            # what was consumed cannot be told, so nothing is admitted.
            raise ValueError("alternative_evidence.sec_campaign_ledger_invalid")
        self._previous = previous
        if head != previous:
            self._write_head()

    @property
    def _head_path(self) -> Path:
        return self.path.with_name(self.path.name + ".head")

    def _write_head(self) -> None:
        staged = self._head_path.with_name(self._head_path.name + ".tmp")
        staged.write_text(self._previous, encoding="utf-8")
        os.replace(staged, self._head_path)

    def _matches(self, line: dict[str, object]) -> bool:
        declared = self.declaration
        return (
            line.get("campaign_id") == declared.campaign_id
            and line.get("maximum_total_attempts") == declared.maximum_total_attempts
            and line.get("maximum_total_response_bytes") == declared.maximum_total_response_bytes
            and line.get("maximum_body_resources") == declared.maximum_body_resources
            and line.get("maximum_document_bytes") == declared.maximum_document_bytes
        )

    def _apply(self, line: dict[str, object]) -> None:
        kind = line.get("kind")
        if kind == "RESERVE":
            sequence = int(str(line["sequence"]))
            self._reserved[sequence] = line
            self._attempts += 1
            if line.get("resource") == "body":
                self._bodies.add(str(line["url"]))
        elif kind == "SETTLE":
            sequence = int(str(line["sequence"]))
            self._reserved.pop(sequence, None)
            self._settled_bytes += int(str(line["actual_bytes"]))

    def _require_current_tail(self) -> None:
        """The file must still end at the line this holder last saw."""
        if self._lock is None:
            raise ValueError("alternative_evidence.sec_campaign_ledger_closed")
        if not self.path.exists():
            if self._previous != _GENESIS:
                raise ValueError("alternative_evidence.sec_campaign_ledger_stale")
            return
        tail = self.path.read_bytes().rstrip(b"\n").rsplit(b"\n", 1)[-1]
        try:
            last = json.loads(tail.decode("utf-8")) if tail else None
        except ValueError as error:
            raise ValueError("alternative_evidence.sec_campaign_ledger_stale") from error
        if last != (self._lines[-1] if self._lines else None):
            raise ValueError("alternative_evidence.sec_campaign_ledger_stale")

    def _append(self, content: dict[str, object]) -> None:
        self._require_current_tail()
        line = {**content, "previous": self._previous}
        line["line_hash"] = _line_hash(self._previous, line)
        payload = (json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(line)
        self._lines.append(line)
        self._previous = str(line["line_hash"])
        self._write_head()

    # ----------------------------------------------------------- spending
    @property
    def attempts_consumed(self) -> int:
        """Count attempts reserved by this campaign."""
        return self._attempts

    @property
    def bytes_consumed(self) -> int:
        """Count settled bytes and unsettled reservations.

        Settled actual bytes plus every unsettled reservation at its
        reserved size -- a transfer no process settled is never zero.
        """
        return self._settled_bytes + sum(
            int(str(value["reserved_bytes"])) for value in self._reserved.values()
        )

    @property
    def unsettled(self) -> int:
        """Count attempts whose reservations have not settled."""
        return len(self._reserved)

    @property
    def remaining_attempts(self) -> int:
        """Count attempts still allowed by the declaration."""
        return max(0, self.declaration.maximum_total_attempts - self._attempts)

    @property
    def remaining_bytes(self) -> int:
        """Count response bytes left after settled and reserved use."""
        return max(0, self.declaration.maximum_total_response_bytes - self.bytes_consumed)

    @property
    def body_resources(self) -> frozenset[str]:
        """Return distinct filing bodies already attempted."""
        return frozenset(self._bodies)

    def reserve(self, *, url: str, resource: str, maximum_bytes: int) -> int:
        """One attempt and the most it may read, written before it is made."""
        sequence = self._attempts + 1
        self._append(
            {
                "kind": "RESERVE",
                "sequence": sequence,
                "url": url,
                "resource": resource,
                "reserved_bytes": maximum_bytes,
                "at": datetime.now(UTC).isoformat(),
            }
        )
        return sequence

    def settle(self, sequence: int, *, actual_bytes: int, outcome: str) -> None:
        """What the reserved attempt actually read, partial or complete."""
        if sequence not in self._reserved:
            raise ValueError("alternative_evidence.sec_campaign_settlement_unknown")
        self._append(
            {
                "kind": "SETTLE",
                "sequence": sequence,
                "actual_bytes": actual_bytes,
                "outcome": outcome,
                "at": datetime.now(UTC).isoformat(),
            }
        )

    def summary(self) -> dict[str, object]:
        """Report the campaign's spent and remaining budgets."""
        return {
            "campaign_id": self.declaration.campaign_id,
            "resumed": self.resumed,
            "attempts_consumed": self.attempts_consumed,
            "attempts_remaining": self.remaining_attempts,
            "bytes_consumed": self.bytes_consumed,
            "bytes_remaining": self.remaining_bytes,
            "unsettled_reservations": self.unsettled,
            "overrun_bytes": max(
                0, self.bytes_consumed - self.declaration.maximum_total_response_bytes
            ),
            "body_resources": len(self._bodies),
            "body_resources_remaining": max(
                0, self.declaration.maximum_body_resources - len(self._bodies)
            ),
            "lines": len(self._lines),
        }


def _line_hash(previous: str, content: dict[str, object]) -> str:
    body = json.dumps({k: v for k, v in content.items() if k != "line_hash"}, sort_keys=True)
    return hashlib.sha256((previous + body).encode("utf-8")).hexdigest()


__all__ = [
    "CAMPAIGN_LEDGER_ROOT",
    "SecCampaignDeclaration",
    "SecCampaignLedger",
    "require_campaign_id",
]
