"""The event log's hash chain: head(n) = digest(head(n-1) || the canonical envelope of event n).

The chain covers `event` alone: every other row of log.sqlite derives from events or is bound by a
digest in one (an event's text by its text_digest). Its canonical encoding is versioned; each
checkpoint records the version the events up to it were folded with, so a later encoding never
reinterprets an earlier segment.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from typing import Any, Final, Protocol

from blake3 import blake3

from alphalattice.kernel.data.errors import DataError

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
    return f"blake3:{blake3(data).hexdigest()}"


def canonical(value: Any) -> bytes:
    """One byte form of a JSON value: sorted keys, no spaces, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def row_digests(batch: Any) -> list[str]:
    """Each row's `digest(canonical(row))`, built over an Arrow record batch's columns.

    The bytes are canonical()'s, column by column: Arrow writes each value, and json.dumps writes
    the few Arrow writes otherwise (an escaped string, a float in exponent form or below 1e-4).
    A column of another type than text, number, boolean or null is refused by name.
    """
    import pyarrow.compute as pc  # lazy: the optional data extra

    names = sorted(batch.schema.names)
    if not names:
        return [digest(b"{}")] * batch.num_rows
    with ThreadPoolExecutor() as pool:  # Arrow's kernels leave the GIL: one column a thread
        texts = list(pool.map(lambda name: _json_texts(batch.column(name), name), names))
    keyed = [
        pc.binary_join_element_wise(json.dumps(name, ensure_ascii=False) + ":", text, "")
        for name, text in zip(names, texts, strict=True)
    ]
    rows = pc.binary_join_element_wise("{", pc.binary_join_element_wise(*keyed, ","), "}", "")
    data, offsets = rows.buffers()[2], rows.buffers()[1]
    body = memoryview(data if data is not None else b"").cast("B")
    ends = memoryview(offsets).cast("i")[rows.offset : rows.offset + len(rows) + 1].tolist()
    return [f"blake3:{blake3(body[start:end]).hexdigest()}" for start, end in pairwise(ends)]


def _json_texts(column: Any, name: str) -> Any:
    """Each value of one column as json.dumps writes it, `null` for a missing one."""
    import pyarrow as pa  # lazy: as above
    import pyarrow.compute as pc  # lazy: as above

    kind = column.type
    if pa.types.is_dictionary(kind):
        column, kind = column.dictionary_decode(), kind.value_type
    if pa.types.is_null(kind):
        return pa.array(["null"] * len(column), pa.string())
    if pa.types.is_integer(kind) or pa.types.is_boolean(kind):
        return pc.fill_null(pc.cast(column, pa.string()), "null")
    if pa.types.is_string(kind) or pa.types.is_large_string(kind):
        column = column.cast(pa.string())
        texts = pc.binary_join_element_wise('"', column, '"', "")
        odd = pc.match_substring_regex(column, r'[\x00-\x1f"\\]')  # json.dumps escapes these
    elif pa.types.is_floating(kind):
        column = column.cast(pa.float64())
        texts = pc.cast(column, pa.string())
        # Arrow and repr agree on a fixed-point float's digits; repr keeps a ".0"; below 1e-4
        # and in exponent form (or nan, inf) the two differ, so json.dumps writes those.
        tiny = pc.and_(pc.less(pc.abs(column), 1e-4), pc.not_equal(column, 0.0))
        odd = pc.or_(pc.match_substring_regex(texts, "[eni]"), tiny)
        whole = pc.binary_join_element_wise(texts, ".0", "")
        texts = pc.if_else(pc.match_substring(texts, "."), texts, whole)
    else:
        raise DataError(
            f"column {name} is {kind}; StorePort digests text, numbers, booleans and nulls,"
            " so send dates as YYYY-MM-DD text and times as microseconds",
            code="data_layer.admission_column_unsupported",
        )
    odd = pc.fill_null(odd, False)
    if pc.any(odd).as_py():
        written = [
            json.dumps(value, ensure_ascii=False) for value in pc.filter(column, odd).to_pylist()
        ]
        texts = pc.replace_with_mask(texts, odd, pa.array(written, pa.string()))
    return pc.fill_null(texts, "null")


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
