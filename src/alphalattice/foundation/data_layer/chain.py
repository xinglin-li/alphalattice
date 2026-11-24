"""The event log's hash chain: head(n) = digest(head(n-1) || the canonical envelope of event n).

The chain covers `event` alone: every other row of log.sqlite derives from events or is bound by a
digest in one (an event's text by its text_digest). Its canonical encoding is versioned; each
checkpoint records the version the events up to it were folded with, so a later encoding never
reinterprets an earlier segment.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any, Final, Protocol

from alphalattice.kernel.identity import integrity_blake3

CHAIN_VERSION: Final = 1
"""The canonical encoding new events are folded with."""
ENVELOPE: Final = (
    "seq",
    "event_id",
    "type",
    "type_version",
    "phase",
    "occurred_us",
    "started_us",
    "recorded_us",
    "actor_kind",
    "actor_id",
    "node",
    "workspace_id",
    "subject_kind",
    "subject_id",
    "trace_id",
    "span_id",
    "parent_span_id",
    "cause_seq",
    "attrs",
    "text_digest",
    "legacy",
)
"""An event's columns, in the order version 1 of the canonical encoding folds them."""


class Event(Protocol):
    """An event row read by column name: a dict, or a sqlite3.Row."""

    def __getitem__(self, column: str, /) -> Any:
        """The value of one of the event's columns."""


def digest(data: bytes) -> str:
    """Content identity of `data`, tagged with its algorithm."""
    return f"blake3:{integrity_blake3(data).hexdigest()}"


def canonical(value: Any) -> bytes:
    """One byte form of a JSON value: sorted keys, no spaces, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _link_v1(head: str, event: Event) -> str:
    return digest(head.encode("ascii") + canonical([event[column] for column in ENVELOPE]))


LINKS: Final[dict[int, Callable[[str, Event], str]]] = {1: _link_v1}
"""Each canonical encoding's fold, by chain_version."""


def fold(head: str, events: Iterable[Event], version: int = CHAIN_VERSION) -> str:
    """The head after folding `events`, in seq order, onto `head` under one encoding."""
    link = LINKS[version]
    for event in events:
        head = link(head, event)
    return head


def fold_segments(
    events: Iterable[Event],
    checkpoints: list[tuple[int, int]],
    head_version: int,
) -> str:
    """The head of the whole chain, each segment folded under its own encoding.

    An event is folded under the encoding of the first checkpoint at or after it (`checkpoints`
    holds (seq, chain_version) in seq order); events after the last one under the head's.
    """
    head, at = "", 0
    for event in events:
        while at < len(checkpoints) and checkpoints[at][0] < event["seq"]:
            at += 1
        head = LINKS[checkpoints[at][1] if at < len(checkpoints) else head_version](head, event)
    return head
