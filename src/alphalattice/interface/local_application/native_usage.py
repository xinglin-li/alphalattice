"""What a native agent ran and spent, read from its host's own session file (AU, V300).

This is the only code that opens an agent's transcript, and it keeps a whitelist: each
response's id, model, the effort it ran at, its token counts and its time, and the file's first
and last time; and a child's exact parent and role from Codex's first record or Claude Code's
bounded sidecar and first record. Everything else in the file, the conversation above all,
is read past and dropped (LAWS OP14: an agent's reading stays in its own context). A file is
read only when it lies under the host's own session root, so a hook cannot point at another
file.

Claude Code writes one record per content block, repeating a response's usage under its message
id, so a response counts once, at its last record; its `input_tokens` are the uncached input.
Codex names the model and effort in each turn's context and records each response's usage once;
its `input_tokens` include the cached reads and writes, so the uncached input is what remains.
Only lines that can hold these facts are decoded (a 400 MB file reads in under a second).
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from uuid import UUID

HOST_CLAUDE_CODE = "claude-code"
HOST_CODEX = "codex"
_MARKERS = {
    HOST_CLAUDE_CODE: (b'"usage":',),
    HOST_CODEX: (b"token_usage_record", b"turn_context"),
}


class NativeUsageReadLimitError(ValueError):
    """A bounded scan stopped before complete cumulative totals could be read."""

    code = "native_usage.read_limit_exceeded"

    def __init__(self, limit: str) -> None:
        """Retain a bound name, without a native path or transcript text.

        Args:
            limit: The code-owned name of the byte bound exceeded by the scan.
        """
        self.limit = limit
        super().__init__("native_usage_limit_exceeded")


@dataclass(frozen=True, slots=True)
class ResponseUsage:
    """One model response's counts; ``input_tokens`` is the uncached input."""

    response_id: str
    model: str
    effort: str | None
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    at: str | None


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """One model's responses in a session file, summed."""

    model: str
    efforts: tuple[str, ...]
    responses: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    first_at: str | None
    last_at: str | None


@dataclass(frozen=True, slots=True)
class SessionUsage:
    """A session file's responses, in the order each first appeared, and its recorded span.

    ``incomplete`` counts the usage records whose counts were missing or impossible; they are
    left out of ``responses``, so a reader that needs every response can refuse the file.
    """

    responses: tuple[ResponseUsage, ...]
    incomplete: int
    first_at: str | None
    last_at: str | None

    def by_model(self) -> tuple[ModelUsage, ...]:
        """One row per model, in model order.

        Returns:
            The summed counts of each model's responses.
        """
        rows: dict[str, list[ResponseUsage]] = {}
        for response in self.responses:
            rows.setdefault(response.model, []).append(response)
        return tuple(_model_usage(model, rows[model]) for model in sorted(rows))


def session_root(host: str) -> Path:
    """Where the host keeps its session files: the only place the reader opens one.

    Args:
        host: ``claude-code`` or ``codex``.

    Returns:
        The root directory; Claude Code honours ``CLAUDE_CONFIG_DIR`` and Codex ``CODEX_HOME``.
    """
    if host == HOST_CLAUDE_CODE:
        return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"


def admitted_session_file(path: object, *, host: str, stem: str) -> Path | None:
    """The session file a hook named, when it is that host's file of that session or agent.

    Args:
        path: The path the hook's payload carried.
        host: The host whose session root admits it.
        stem: The file name the session or agent id gives it, without ``.jsonl``.

    Returns:
        The resolved file, or ``None`` for another name, a link, a path outside the root or
        anything that is not a file.
    """
    if not isinstance(path, str) or not path:
        return None
    candidate = Path(path)
    if candidate.name != f"{stem}.jsonl":
        return None
    try:
        if not candidate.is_absolute() or candidate.is_symlink():
            return None
        resolved = candidate.resolve()
        root = session_root(host).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_relative_to(root) and resolved.is_file() else None


def codex_session_file(thread_id: str) -> Path | None:
    """The newest Codex rollout file of a thread, found by the id its name ends with.

    Args:
        thread_id: The Codex thread (a session or a native subagent).

    Returns:
        The file, or ``None`` when the thread has none or the id is not a plain token.
    """
    if not 0 < len(thread_id) <= 128 or not all(c.isalnum() or c == "-" for c in thread_id):
        return None
    files = sorted(session_root(HOST_CODEX).glob(f"*/*/*/rollout-*{thread_id}.jsonl"))
    if not files:
        return None
    return admitted_session_file(str(files[-1]), host=HOST_CODEX, stem=files[-1].stem)


FIRST_RECORD_BYTES = 1_048_576
"""The most of a Codex rollout's first record that is read for its spawn: the session's own
metadata, never a turn after it."""


@dataclass(frozen=True, slots=True)
class ThreadSpawn:
    """A child's recorded direct parent, role and optional Codex canonical path (V568)."""

    parent_thread_id: str
    agent_role: str | None
    agent_path: str | None = None


def canonical_agent_path(value: object) -> str | None:
    """Admit the bounded native task path syntax, without asserting an agent's identity.

    Args:
        value: A native path selected from session metadata or a tool return.

    Returns:
        The unchanged canonical ``/root/...`` path, or None for any other shape.
    """
    if (
        not isinstance(value, str)
        or len(value) > 200
        or re.fullmatch(r"/root(?:/[a-z0-9_]{1,64})+", value) is None
    ):
        return None
    return value


def codex_thread_spawn(thread_id: str) -> ThreadSpawn | None:
    """Which thread spawned a Codex thread, as the thread's own rollout first records it.

    Codex opens each rollout with its ``session_meta``; a native subagent's names
    ``source.subagent.thread_spawn``. Only that first record is read, up to
    `FIRST_RECORD_BYTES`, and only ``parent_thread_id``, ``agent_role`` and an optional
    canonical ``agent_path`` leave it. The path must be named in the spawn; when the
    top-level metadata also names it, both must agree. The rest of the record is discarded
    and every turn after it is never read (LAWS OP14).

    Args:
        thread_id: The thread, by the id its rollout's name ends with.

    Returns:
        Its spawn; None for a thread with no rollout, one no thread spawned, or a first record
        past the bound, of another thread or of another shape.
    """
    path = codex_session_file(thread_id)
    if path is None:
        return None
    try:
        with path.open("rb") as stream:
            first = stream.readline(FIRST_RECORD_BYTES + 1)
    except OSError:
        return None
    if len(first) > FIRST_RECORD_BYTES:
        return None
    return _codex_spawn_record(first, thread_id)


def _codex_spawn_record(first: bytes, thread_id: str) -> ThreadSpawn | None:
    """Discard every field except the exact first-record spawn identity."""
    record = _record(first)
    payload = record.get("payload")
    meta = payload if isinstance(payload, Mapping) else {}
    source = meta.get("source")
    subagent = source.get("subagent") if isinstance(source, Mapping) else None
    spawn = subagent.get("thread_spawn") if isinstance(subagent, Mapping) else None
    if record.get("type") != "session_meta" or meta.get("id") != thread_id:
        return None
    parent = _text(spawn.get("parent_thread_id")) if isinstance(spawn, Mapping) else None
    if parent is None or not isinstance(spawn, Mapping):
        return None
    agent_path = canonical_agent_path(spawn.get("agent_path"))
    if "agent_path" in meta and meta.get("agent_path") != agent_path:
        agent_path = None
    return ThreadSpawn(
        parent_thread_id=parent,
        agent_role=_text(spawn.get("agent_role")),
        agent_path=agent_path,
    )


CODEX_ASSIGNMENT_FILES = 4096
CODEX_ASSIGNMENT_BYTES = 512 * 1024 * 1024


def _codex_assignment_file(session_id: str, agent_path: str) -> Path | None:
    """Resolve a canonical tool task name by exact parent/path metadata, never by role or age.

    Only bounded first records are read. Two matching files, unreadable metadata or an
    exceeded census refuse the association rather than returning a partial match.
    """
    if canonical_agent_path(agent_path) != agent_path:
        return None
    paths = list(
        islice(session_root(HOST_CODEX).glob("*/*/*/rollout-*.jsonl"), CODEX_ASSIGNMENT_FILES + 1)
    )
    if len(paths) > CODEX_ASSIGNMENT_FILES:
        raise NativeUsageReadLimitError("child_metadata_files")
    found = None
    consumed = 0
    parent_bytes, path_bytes = session_id.encode("utf-8"), agent_path.encode("utf-8")
    for candidate in paths:
        ident = candidate.stem[-36:]
        try:
            if str(UUID(ident)) != ident:
                continue
        except ValueError:
            continue
        path = admitted_session_file(str(candidate), host=HOST_CODEX, stem=candidate.stem)
        if path is None or path != candidate:
            return None
        try:
            with path.open("rb") as stream:
                first = stream.readline(FIRST_RECORD_BYTES + 1)
        except OSError:
            return None
        consumed += len(first)
        if consumed > CODEX_ASSIGNMENT_BYTES:
            raise NativeUsageReadLimitError("child_metadata_bytes")
        if len(first) > FIRST_RECORD_BYTES:
            return None
        # Reject irrelevant native headers cheaply. This is not an identity match:
        # every possible match is still parsed and checked below. Escaped identities
        # take the complete parser path too; no JSON spelling decides an association.
        if (parent_bytes not in first or path_bytes not in first) and not (
            b"\\u" in first or b"\\/" in first
        ):
            continue
        spawn = _codex_spawn_record(first, ident)
        if spawn is None or (spawn.parent_thread_id, spawn.agent_path) != (session_id, agent_path):
            continue
        if found is not None:
            return None
        found = path
    return found


def codex_assigned_spawn(session_id: str, agent_id: str) -> ThreadSpawn | None:
    """Read a UUID or unique canonical assignment's exact native parent and role.

    This associates count files only; it supplies no authorship or lifecycle credit.
    """
    if canonical_agent_path(agent_id) is None:
        return codex_thread_spawn(agent_id)
    path = _codex_assignment_file(session_id, agent_id)
    if path is None:
        return None
    try:
        with path.open("rb") as stream:
            first = stream.readline(FIRST_RECORD_BYTES + 1)
    except OSError:
        return None
    if len(first) > FIRST_RECORD_BYTES:
        return None
    spawn = _codex_spawn_record(first, path.stem[-36:])
    return (
        spawn
        if spawn is not None
        and (spawn.parent_thread_id, spawn.agent_path) == (session_id, agent_id)
        else None
    )


def session_file(host: str, session_id: str, agent_id: str | None = None) -> Path | None:
    """A session's file, or one of its subagents', found by their ids where no hook names it.

    A command run inside a session knows the session and nothing else (V301), so the file is
    looked up under the host's session root by the name the ids give it and admitted as a
    hook's would be: Claude Code keeps ``<project>/<session>.jsonl`` and
    ``<project>/<session>/subagents/agent-<id>.jsonl``, Codex one rollout per thread.

    Args:
        host: ``claude-code`` or ``codex``.
        session_id: The lead's session.
        agent_id: One of its subagents; the lead's own file when None.

    Returns:
        The selected Codex rollout or unique Claude file, or ``None`` when the ids or source
        are unavailable or ambiguous. Claude files are never selected by modification time.
    """
    if host == HOST_CODEX:
        if agent_id is not None and canonical_agent_path(agent_id) is not None:
            return _codex_assignment_file(session_id, agent_id)
        return codex_session_file(agent_id or session_id)
    ids = (session_id,) if agent_id is None else (session_id, agent_id)
    if not all(0 < len(v) <= 128 and all(c.isalnum() or c in "-_" for c in v) for v in ids):
        return None
    if agent_id is not None:
        parent = session_file(host, session_id)
        if parent is None:
            return None
        candidate = parent.parent / session_id / "subagents" / f"agent-{agent_id}.jsonl"
        admitted = admitted_session_file(str(candidate), host=host, stem=f"agent-{agent_id}")
        return admitted if admitted == candidate else None
    files = {
        admitted
        for candidate in session_root(host).glob(f"*/{session_id}.jsonl")
        if (admitted := admitted_session_file(str(candidate), host=host, stem=session_id))
        is not None
        and admitted == candidate
    }
    return next(iter(files)) if len(files) == 1 else None


CLAUDE_METADATA_BYTES = 16_384
"""The bound for one Claude agent sidecar; no description or prompt leaves its read."""


def claude_thread_spawn(session_id: str, agent_id: str) -> ThreadSpawn | None:
    """Read an exactly named Claude child's direct-parent and specialist-role metadata.

    Claude Code writes ``agentType``, ``spawnDepth`` and an optional ``parentAgentId`` beside
    the child's file. Its first transcript record carries ``sessionId``, ``agentId`` and
    ``isSidechain``. All must agree with the bound lead and exact assignment recipient; a
    nested child, duplicate lead file, redirected path or missing metadata admits nothing.
    This associates count files only and supplies no native hook or authorship credit.

    Args:
        session_id: The bound lead Session, never a path or role alias.
        agent_id: The exact native child id named by the assignment.

    Returns:
        The direct-parent and explicit role tuple, or None for unavailable, invalid or
        over-bound metadata. Neither the conversation nor sidecar description is retained.
    """
    if agent_id == session_id:
        return None
    try:
        path = session_file(HOST_CLAUDE_CODE, session_id, agent_id)
        if path is None:
            return None
        metadata = path.with_suffix(".meta.json")
        if metadata.is_symlink() or metadata.resolve() != metadata or not metadata.is_file():
            return None
        with metadata.open("rb") as stream:
            raw = stream.read(CLAUDE_METADATA_BYTES + 1)
        with path.open("rb") as stream:
            first = stream.readline(FIRST_RECORD_BYTES + 1)
    except (OSError, RuntimeError, ValueError):
        return None
    if len(raw) > CLAUDE_METADATA_BYTES or len(first) > FIRST_RECORD_BYTES:
        return None
    meta, record = _record(raw), _record(first)
    role = _text(meta.get("agentType"))
    if (
        role is None
        or type(meta.get("spawnDepth")) is not int
        or meta.get("spawnDepth") != 1
        or meta.get("parentAgentId") is not None
        or record.get("sessionId") != session_id
        or record.get("agentId") != agent_id
        or record.get("isSidechain") is not True
    ):
        return None
    return ThreadSpawn(parent_thread_id=session_id, agent_role=role)


def hook_session_files(
    payload: Mapping[str, object], *, host: str, session_id: str, agent_id: str
) -> dict[str, Path]:
    """The lead's and the agent's session files at a subagent hook, as admitted.

    Claude Code's payload names both files (``transcript_path`` for the session,
    ``agent_transcript_path`` for the subagent), each admitted only under its own name. Codex
    names each thread's rollout by the thread's id, so its payload's paths are not read.

    Args:
        payload: The hook's decoded input.
        host: The host the hook came from.
        session_id: The lead's session.
        agent_id: The subagent.

    Returns:
        ``lead`` and ``agent`` mapped to the files found; a file not admitted is left out.
    """
    if host == HOST_CODEX:
        found = {"lead": codex_session_file(session_id), "agent": codex_session_file(agent_id)}
    else:
        found = {
            "lead": admitted_session_file(
                payload.get("transcript_path"), host=host, stem=session_id
            ),
            "agent": admitted_session_file(
                payload.get("agent_transcript_path"), host=host, stem=f"agent-{agent_id}"
            ),
        }
    return {name: path for name, path in found.items() if path is not None}


def read_session(
    path: Path,
    *,
    host: str,
    max_bytes: int | None = None,
    max_line_bytes: int | None = None,
) -> SessionUsage:
    """The responses a session file records and the span of its recorded times.

    Args:
        path: An admitted session file (``admitted_session_file``, ``codex_session_file``).
        host: ``claude-code`` or ``codex``, which decides the file's format.
        max_bytes: Optional positive byte bound over the whole scan, including discarded lines.
        max_line_bytes: Optional positive byte bound over each complete native line.

    Returns:
        The file's responses, each once, and how many usage records could not be counted.

    Raises:
        OSError: If the admitted file cannot be read.
        ValueError: If a supplied byte bound is invalid.
        NativeUsageReadLimitError: If a bound prevents a complete scan; no totals are returned.
    """
    markers = _MARKERS[host]
    responses: dict[str, ResponseUsage] = {}
    incomplete = 0
    first_at: str | None = None
    tail: deque[bytes] = deque(maxlen=8)
    turn: tuple[str | None, str | None] = (None, None)
    for line in _lines(path, max_bytes=max_bytes, max_line_bytes=max_line_bytes):
        tail.append(line)
        if first_at is None:
            first_at = _text(_record(line).get("timestamp"))
        if not any(marker in line for marker in markers):
            continue
        record = _record(line)
        if not record:
            incomplete += 1
            continue
        if host == HOST_CLAUDE_CODE:
            response = _claude_response(record)
        else:
            turn = _codex_turn(record, turn)
            response = _codex_response(record, turn)
        if response is None:
            continue
        if isinstance(response, int):
            incomplete += response
            continue
        known = responses.get(response.response_id)
        responses[response.response_id] = (
            response if known is None else _replace_counts(known, response)
        )
    return SessionUsage(
        responses=tuple(responses.values()),
        incomplete=incomplete,
        first_at=first_at,
        last_at=next(
            (stamp for line in reversed(tail) if (stamp := _text(_record(line).get("timestamp")))),
            None,
        ),
    )


def _model_usage(model: str, responses: list[ResponseUsage]) -> ModelUsage:
    stamps = sorted(response.at for response in responses if response.at)
    return ModelUsage(
        model=model,
        efforts=tuple(sorted({response.effort for response in responses if response.effort})),
        responses=len(responses),
        input_tokens=sum(response.input_tokens for response in responses),
        cache_read_tokens=sum(response.cache_read_tokens for response in responses),
        cache_write_tokens=sum(response.cache_write_tokens for response in responses),
        output_tokens=sum(response.output_tokens for response in responses),
        first_at=stamps[0] if stamps else None,
        last_at=stamps[-1] if stamps else None,
    )


def _replace_counts(known: ResponseUsage, later: ResponseUsage) -> ResponseUsage:
    """A response's later record carries its final counts; it keeps its first place."""
    return ResponseUsage(
        response_id=known.response_id,
        model=later.model,
        effort=later.effort or known.effort,
        input_tokens=later.input_tokens,
        cache_read_tokens=later.cache_read_tokens,
        cache_write_tokens=later.cache_write_tokens,
        output_tokens=later.output_tokens,
        at=later.at or known.at,
    )


def _lines(
    path: Path, *, max_bytes: int | None = None, max_line_bytes: int | None = None
) -> Iterator[bytes]:
    for bound in (max_bytes, max_line_bytes):
        if bound is not None and (type(bound) is not int or bound <= 0):
            raise ValueError("native_usage_bound_invalid")
    with path.open("rb") as stream:
        scanned = 0
        while True:
            remaining = max_bytes - scanned if max_bytes is not None else None
            bounds = [bound for bound in (remaining, max_line_bytes) if bound is not None]
            line = stream.readline(min(bounds) + 1 if bounds else -1)
            if not line:
                break
            scanned += len(line)
            if max_bytes is not None and scanned > max_bytes:
                raise NativeUsageReadLimitError("max_bytes")
            if max_line_bytes is not None and len(line) > max_line_bytes:
                raise NativeUsageReadLimitError("max_line_bytes")
            if line.strip():
                yield line


def _record(line: bytes) -> Mapping[str, object]:
    """One JSON object line; a line that does not parse reads as an empty record."""
    try:
        record = json.loads(line)
    except (ValueError, RecursionError):
        return {}
    return record if isinstance(record, dict) else {}


def _count(usage: Mapping[str, object], key: str) -> int | None:
    value = usage.get(key)
    return value if type(value) is int and value >= 0 else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 128 else None


def _claude_response(record: Mapping[str, object]) -> ResponseUsage | int | None:
    message = record.get("message")
    if record.get("type") != "assistant" or not isinstance(message, dict):
        return None
    model, ident, usage = _text(message.get("model")), message.get("id"), message.get("usage")
    # A synthetic record (an error or an interruption) names no model that ran.
    if model is None or model.startswith("<") or not isinstance(ident, str):
        return None
    if not isinstance(usage, dict):
        return 1
    uncached, read, written, output = (
        _count(usage, key)
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
        )
    )
    if uncached is None or read is None or written is None or output is None:
        return 1
    return ResponseUsage(
        response_id=ident,
        model=model,
        effort=_text(record.get("effort")),
        input_tokens=uncached,
        cache_read_tokens=read,
        cache_write_tokens=written,
        output_tokens=output,
        at=_text(record.get("timestamp")),
    )


def _codex_turn(
    record: Mapping[str, object], turn: tuple[str | None, str | None]
) -> tuple[str | None, str | None]:
    payload = record.get("payload")
    if record.get("type") != "turn_context" or not isinstance(payload, dict):
        return turn
    return _text(payload.get("model")) or turn[0], _text(payload.get("effort")) or turn[1]


def _codex_response(
    record: Mapping[str, object], turn: tuple[str | None, str | None]
) -> ResponseUsage | int | None:
    payload = record.get("payload")
    if record.get("type") != "token_usage_record" or not isinstance(payload, dict):
        return None
    ident, usage = payload.get("response_id"), payload.get("usage")
    if turn[0] is None or not isinstance(ident, str) or not isinstance(usage, dict):
        return 1
    total, cached, written, output = (
        _count(usage, key)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
        )
    )
    if total is None or cached is None or written is None or output is None:
        return 1
    if total < cached + written:
        return 1
    return ResponseUsage(
        response_id=ident,
        model=turn[0],
        effort=turn[1],
        input_tokens=total - cached - written,
        cache_read_tokens=cached,
        cache_write_tokens=written,
        output_tokens=output,
        at=_text(record.get("timestamp")),
    )


__all__ = [
    "HOST_CLAUDE_CODE",
    "HOST_CODEX",
    "ModelUsage",
    "NativeUsageReadLimitError",
    "ResponseUsage",
    "SessionUsage",
    "admitted_session_file",
    "codex_assigned_spawn",
    "codex_session_file",
    "hook_session_files",
    "read_session",
    "session_file",
    "session_root",
]
