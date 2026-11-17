"""The workspace's observations, read from the Host's log.

The Host writes each kept observation into its log as one event (`trace_record`): its payload as
the event's attributes and the record's own fields (its clocks, source and sequence, authority,
retention, Task, run, stage and correlation ids) as the `observation` attribute. The event's type
is its kind, the event's id its identity and its place in the log its ordinal; the store epoch is
the log's first event, which changes only when the data root is made again. An agent's event
carries its subject's session, agent, model and token counts under their OpenTelemetry GenAI names
(`agent_attributes`) and its summary as the event's text; `payload_of` inverts both.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal, Protocol

from .contracts import (
    ObservationReadCursor,
    ObservationReadItem,
    ObservationReadPage,
    ObservationRecord,
)

MAXIMUM_READ_PAGE: Final = 500
"""The most observations one read returns."""
KINDS: Final = {
    "product.operation": "ProductOperationObserved",
    "agent.activity": "ExternalActivityObserved",
    "task.transition": "TaskControlTransition",
    "artifact.verified": "ArtifactVerificationObserved",
    "clock.correlated": "ClockCorrelationRecorded",
}
"""The event types an observation is written as, and the kind each is."""
TYPES: Final = tuple(KINDS)
TYPE_OF: Final = {kind: type_ for type_, kind in KINDS.items()}
OBSERVATION_ID: Final = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
"""An observation's id: the id of the log event that keeps it, issued by the Host and opaque; only
its shape is checked before it is looked up."""
GENAI_IDS: Final = ("native_session_id", "native_agent_id", "model")
GENAI: Final = {
    "native_session_id": "gen_ai.conversation.id",
    "native_agent_id": "gen_ai.agent.id",
    "model": "gen_ai.response.model",
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cache_read_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_write_tokens": "gen_ai.usage.cache_creation.input_tokens",
}
"""An agent event's subject fields with an OpenTelemetry GenAI name: the ids, then the counts."""
_CACHED: Final = ("cache_read_tokens", "cache_write_tokens")
INDEXED: Final = (
    "event_kind",
    "operation",
    "caller",
    "phase",
    *(GENAI[key] for key in GENAI_IDS),
)
SUBJECT_INDEXED: Final = ("native_session_id", "agent_session", "agent_vendor", "role")
OBSERVATION_INDEXED: Final = (
    "source_kind",
    "authority",
    "run_id",
    "correlation_ids",
)
"""What the reads select an observation's event on: its attributes, its subject's and its
own fields', kept as attributes when the whole is the event's text (`observation.in_text`)."""
_OBSERVATIONS: Final = f"type IN ({', '.join(repr(name) for name in TYPES)}) AND workspace_id = ?"
_NATIVE_SESSION: Final = (
    "COALESCE(json_extract(attrs, '$.\"gen_ai.conversation.id\"'), "
    "json_extract(attrs, '$.subject.native_session_id'))"
)
"""The subject's native session: an agent's event names it under its GenAI name."""
_SESSION: Final = f"COALESCE({_NATIVE_SESSION}, json_extract(attrs, '$.subject.agent_session'))"
"""The Session an exchange or an agent's operation belongs to."""
_EVENT_KIND: Final = "json_extract(attrs, '$.event_kind')"
_SNAPSHOT_GROUP: Final = f"""{_EVENT_KIND}, json_extract(attrs, '$."gen_ai.agent.id"'),
    CASE {_EVENT_KIND}
        WHEN 'NATIVE_AGENT_USAGE' THEN json_extract(attrs, '$."gen_ai.response.model"')
        ELSE json_extract(attrs, '$.subject.role') END"""
_SNAPSHOT_KINDS: Final = f"""({_EVENT_KIND} IN (
    'NATIVE_AGENT_USAGE', 'NATIVE_SUBAGENT_START_HOOK', 'NATIVE_SUBAGENT_STOP_HOOK')
    OR ({_EVENT_KIND} = 'NATIVE_COORDINATION_MESSAGE'
        AND json_extract(attrs, '$.subject.message_kind') = 'session_bound'))"""
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


class ObservationStorageError(RuntimeError):
    """Bounded, user-safe failure to read or keep the workspace's records."""

    def __init__(self, failure_code: str, detail: str) -> None:
        """Keep a stable failure code and bounded user-safe detail."""
        safe_detail = " ".join(str(detail).split())[:1000] or type(self).__name__
        super().__init__(safe_detail)
        self.failure_code = failure_code
        self.safe_detail = safe_detail


class LogReader(Protocol):
    """What the reads need of the Host's log: rows as the Host sees them, its open batch too."""

    def read(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Rows of `sql` with `args`."""
        ...


def agent_attributes(payload: Mapping[str, Any]) -> dict[str, Any]:
    """An agent event's attributes, its summary aside: its subject's session, agent, model and
    token counts under their OpenTelemetry GenAI names, the counts as numbers and the input with
    its cached part (as GenAI counts it); the rest of its subject as it came."""
    attributes = {key: value for key, value in payload.items() if key not in ("summary", "subject")}
    subject = dict(payload.get("subject") or {})
    attributes |= {GENAI[key]: subject.pop(key) for key in GENAI_IDS if key in subject}
    counts = {key: int(subject.pop(key)) for key in GENAI if str(subject.get(key, "")).isdigit()}
    attributes |= {GENAI[key]: count for key, count in counts.items()}
    if "input_tokens" in counts:  # the uncached input, as the vendors' session files count it
        attributes[GENAI["input_tokens"]] += sum(counts.get(key, 0) for key in _CACHED)
    return attributes | ({"subject": subject} if subject else {})


def payload_of(event_type: str, attributes: Mapping[str, Any], text: str | None) -> dict[str, Any]:
    """The payload an observation's event was written from: `agent_attributes` inverted for an
    agent's event (its counts strings again, as the session files give them), whose summary is the
    text; any other event's attributes as they are. The `observation` attribute is set aside."""
    payload = {key: value for key, value in attributes.items() if key != "observation"}
    if event_type != "agent.activity":
        return payload
    subject = dict(payload.pop("subject", None) or {})
    counts = {key: payload.pop(GENAI[key], None) for key in GENAI if key not in GENAI_IDS}
    subject |= {key: payload.pop(GENAI[key]) for key in GENAI_IDS if GENAI[key] in payload}
    counts = {key: value for key, value in counts.items() if value is not None}
    if "input_tokens" in counts:
        counts["input_tokens"] -= sum(counts.get(key, 0) for key in _CACHED)
    subject |= {key: str(value) for key, value in counts.items()}
    return payload | {"summary": text} | ({"subject": subject} if subject else {})


def agent_event(source_id: str, source_sequence: int) -> str:
    """The subject id of an agent's event: its producer's source and sequence, which a replay
    repeats."""
    return f"{source_id}#{source_sequence}"


class LogObservations:
    """The observation ledger's reads, answered from one workspace's events in the Host's log."""

    def __init__(self, log: LogReader, workspace_id: str) -> None:
        """Read `workspace_id`'s observations from `log`."""
        self._log, self._workspace = log, workspace_id
        self._epoch: str | None = None

    @property
    def store_epoch(self) -> str:
        """The log's identity, its first event's id: cursors are scoped to it."""
        if self._epoch is None:
            rows = self._rows("SELECT event_id FROM event ORDER BY seq LIMIT 1", ())
            self._epoch = str(rows[0][0]).replace("-", "") if rows else "0" * 32
        return self._epoch

    def head_ordinal(self) -> int:
        """The newest observation's place in the log, 0 when there is none."""
        rows = self._rows(f"SELECT COALESCE(MAX(seq), 0) FROM event WHERE {_OBSERVATIONS}")
        return int(rows[0][0])

    def observations_after(
        self, cursor: ObservationReadCursor | None, *, limit: int
    ) -> ObservationReadPage:
        """At most `limit` observations in log order after the cursor; no cursor reads the newest
        `limit`, and a cursor from another epoch or past the head is RESET with a fresh tail."""
        if not 1 <= limit <= MAXIMUM_READ_PAGE:
            raise ValueError("observation.read_limit_invalid")
        head = self.head_ordinal()
        if cursor is None or cursor.store_epoch != self.store_epoch or cursor.ordinal > head:
            disposition: Literal["TAIL", "RESET"] = "TAIL" if cursor is None else "RESET"
            rows = self._events(f"{_OBSERVATIONS} ORDER BY seq DESC LIMIT ?", (limit + 1,))
            return self._page(disposition, head, list(reversed(rows[:limit])), len(rows) > limit)
        rows = self._events(
            f"{_OBSERVATIONS} AND seq > ? ORDER BY seq LIMIT ?", (cursor.ordinal, limit + 1)
        )
        return self._page("CONTINUED", head, rows[:limit], len(rows) > limit)

    def observations_of_kind(
        self,
        schema_kind: str,
        *,
        limit: int,
        before: int | None = None,
        native_session_id: str | None = None,
        snapshots: bool = False,
        head_ordinal: int | None = None,
    ) -> ObservationReadPage:
        """A kind's, or a selected Session's, observations in log order, as the ledger read them:
        a Session's exchanges with its agent operations, or its latest usage, hooks and binding
        (`snapshots`), independently of `before`."""
        if not 1 <= limit <= MAXIMUM_READ_PAGE:
            raise ValueError("observation.read_limit_invalid")
        if (
            not schema_kind
            or (before is not None and before < 1)
            or (head_ordinal is not None and head_ordinal < 0)
        ):
            raise ValueError("observation.read_cursor_invalid")
        head = self.head_ordinal() if head_ordinal is None else head_ordinal
        where = f"{_OBSERVATIONS} AND type = ?"
        args: list[str | int] = [self._workspace, TYPE_OF.get(schema_kind, "")]
        session = schema_kind == "ExternalActivityObserved" and not snapshots
        if native_session_id is not None and session:
            # Operations belong to the Session span; its last message is not its endpoint.
            where = f"""{_OBSERVATIONS} AND {_SESSION} = ? AND (type = 'agent.activity'
                OR (type = 'product.operation'
                    AND json_extract(attrs, '$.observation.source_kind') = 'PRODUCT_OPERATION'
                    AND json_extract(attrs, '$.observation.authority') = 'OPERATIONAL_ASSERTION'
                    AND json_extract(attrs, '$.subject.agent_vendor')
                        IN ('codex', 'claude-code')))"""
            args = [self._workspace, native_session_id]
        elif native_session_id is not None:
            where += f" AND {_NATIVE_SESSION} = ?"
            args.append(native_session_id)
        if snapshots:
            where = f"""seq IN (SELECT MAX(seq) FROM event WHERE {where} AND seq <= ?
                AND {_SNAPSHOT_KINDS} GROUP BY {_SNAPSHOT_GROUP})"""
            args.append(head)
        else:
            where += " AND seq < ?"
            args.append(head + 1 if before is None else min(before, head + 1))
            if native_session_id is not None:
                where += f" AND {_EVENT_KIND} IS NOT 'NATIVE_AGENT_USAGE'"
        rows = self._events(f"{where} ORDER BY seq DESC LIMIT ?", (*args, limit + 1), scoped=False)
        return self._page("TAIL", head, list(reversed(rows[:limit])), len(rows) > limit)

    def read(self, observation_id: str) -> ObservationRecord:
        """One observation by its id, the id of the event that keeps it.

        Raises:
            ValueError: No observation has that id (`observation.tampered`, as the ledger said).
        """
        rows = self._events(f"{_OBSERVATIONS} AND event_id = ? LIMIT 1", (observation_id,))
        if not rows:
            raise ValueError("observation.tampered")
        return rows[0].record

    def replayed(self, source_id: str, source_sequence: int) -> ObservationRecord | None:
        """The agent's event already kept for this producer source and sequence, if any."""
        rows = self._events(
            "workspace_id = ? AND type = 'agent.activity' AND subject_kind = 'agent_event'"
            " AND subject_id = ? LIMIT 1",
            (agent_event(source_id, source_sequence),),
        )
        return rows[0].record if rows else None

    def unique_correlated_observation(
        self, run_id: str, schema_kind: str, correlation_id: str
    ) -> ObservationRecord | None:
        """The one observation of a run and kind whose correlation ids hold `correlation_id`.

        Raises:
            ValueError: A selector is empty, or more than one observation matches.
        """
        if any(
            not isinstance(value, str) or not value
            for value in (run_id, schema_kind, correlation_id)
        ):
            raise ValueError("observation.correlation_selector_invalid")
        rows = self._events(
            f"""{_OBSERVATIONS} AND json_extract(attrs, '$.observation.run_id') = ? AND type = ?
            AND EXISTS (SELECT 1 FROM json_each(attrs, '$.observation.correlation_ids')
                WHERE type = 'text' AND value = ?) LIMIT 2""",
            (run_id, TYPE_OF.get(schema_kind, ""), correlation_id),
        )
        if len(rows) > 1:
            raise ValueError("observation.correlation_ambiguous")
        return rows[0].record if rows else None

    def calls(self, traces: set[str]) -> dict[str, dict[str, object]]:
        """The CLI call at the root of each of `traces`, by trace: its command, conversation, start
        and, once it ended, its end and exit code; a call still running has no end."""
        named = sorted(traces)
        calls: dict[str, dict[str, object]] = {}
        for at in range(0, len(named), MAXIMUM_READ_PAGE):
            chunk = named[at : at + MAXIMUM_READ_PAGE]
            rows = self._rows(
                f"""SELECT trace_id, span_id, phase, started_us, occurred_us, attrs FROM event
                WHERE type = 'cli.call' AND workspace_id = ?
                AND trace_id IN ({", ".join("?" * len(chunk))}) ORDER BY seq""",
                (self._workspace, *chunk),
            )
            for trace, span, phase, started, occurred, attrs in rows:
                values = json.loads(attrs)
                call = calls.setdefault(
                    trace,
                    {
                        "span_id": span,
                        "command": values.get("command"),
                        "conversation": values.get("gen_ai.conversation.id"),
                        "unattributed": bool(values.get("unattributed")),
                        "started_at": _moment(occurred if phase == "started" else started),
                        "ended_at": None,
                    },
                )
                if phase != "started":
                    call |= {"ended_at": _moment(occurred), "exit_code": values.get("exit_code")}
        return calls

    def refusal_counts(self, *, since_day: str) -> tuple[dict[str, object], ...]:
        """The refusals on or after a day, one row a day, operation, code, caller kind and vendor,
        with its count and first and last moments, the most frequent first."""
        since = int(
            (datetime.fromisoformat(since_day).replace(tzinfo=UTC) - _EPOCH)
            / timedelta(microseconds=1)
        )
        fields = ("operation", "failure_code", "caller", "vendor")
        columns = ", ".join(f"json_extract(attrs, '$.{name}') AS {name}" for name in fields)
        rows = self._rows(
            f"""SELECT date(occurred_us / 1000000, 'unixepoch') AS day, {columns},
                COUNT(*) AS count, MIN(occurred_us), MAX(occurred_us) FROM event
            WHERE type = 'request.refused' AND workspace_id = ? AND occurred_us >= ?
            GROUP BY day, {", ".join(fields)}
            ORDER BY count DESC, day DESC, {", ".join(fields)} LIMIT ?""",
            (self._workspace, since, MAXIMUM_READ_PAGE),
        )
        return tuple(
            dict(zip(("day", *fields), row[:5], strict=True))
            | {"count": int(row[5]), "first_at": _moment(row[6]), "last_at": _moment(row[7])}
            for row in rows
        )

    def _events(
        self, where: str, args: tuple[Any, ...], *, scoped: bool = True
    ) -> list[ObservationReadItem]:
        text = "(SELECT text FROM event_text WHERE event_text.seq = event.seq)"
        columns = "seq, type, attrs, " + text + ", event_id, trace_id, span_id, parent_span_id"
        sql = f"SELECT {columns} FROM event WHERE {where}"
        rows = self._rows(sql, (self._workspace, *args) if scoped else args)
        return [_item(row) for row in rows]

    def _page(
        self,
        disposition: Literal["TAIL", "CONTINUED", "RESET"],
        head: int,
        items: list[ObservationReadItem],
        more: bool,
    ) -> ObservationReadPage:
        return ObservationReadPage(
            disposition=disposition,
            store_epoch=self.store_epoch,
            head_ordinal=head,
            items=tuple(items),
            more=more,
        )

    def _rows(self, sql: str, args: tuple[Any, ...] | None = None) -> list[sqlite3.Row]:
        return _guarded(lambda: self._log.read(sql, (self._workspace,) if args is None else args))


_OWN: Final = frozenset(ObservationRecord.model_fields) - {
    "kind",
    "schema_kind",
    "observation_id",
    "inline_safe_payload",
}
"""The record's own fields an event's `observation` attribute keeps. Any other key there is set
aside: the in-text marker, and the envelope's fields (its kind, version, sensitivity, hashes and
minted id) that rows written before the envelope retired still hold."""


def _item(row: sqlite3.Row) -> ObservationReadItem:
    """The observation one event row keeps: its kind the event's type, its id the event's, its own
    fields and its payload from the event's attributes and text."""
    attributes, text = json.loads(row[2]), row[3]
    if attributes["observation"].get("in_text"):
        whole = json.loads(text)
        attributes, text = whole["attributes"], whole["text"]
    own = {key: value for key, value in attributes["observation"].items() if key in _OWN}
    record = ObservationRecord.model_validate(
        own
        | {
            "schema_kind": KINDS[row[1]],
            "observation_id": row[4],
            "inline_safe_payload": payload_of(row[1], attributes, text),
        }
    )
    return ObservationReadItem(
        ordinal=int(row[0]), record=record, trace_id=row[5], span_id=row[6], parent_span_id=row[7]
    )


def _moment(us: int) -> str:
    return (_EPOCH + timedelta(microseconds=int(us))).isoformat()


def _guarded[Result](read: Callable[[], Result]) -> Result:
    try:
        return read()
    except sqlite3.Error as error:
        raise ObservationStorageError(
            "activity.storage_unavailable", type(error).__name__
        ) from error
