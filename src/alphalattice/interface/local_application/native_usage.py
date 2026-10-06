"""What a native agent ran and spent, read from its host's own session file (AU, V300).

This is the only code that opens an agent's transcript, and it keeps a whitelist: each
response's id, model, the effort it ran at, its token counts and its time, and the file's first
and last time; and a Codex thread's spawn, its parent, role and optional canonical native path,
from its first record alone (V568). Everything else in the file, the conversation above all,
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
from pathlib import Path

HOST_CLAUDE_CODE = "claude-code"
HOST_CODEX = "codex"
_MARKERS = {
    HOST_CLAUDE_CODE: (b'"usage":',),
    HOST_CODEX: (b"token_usage_record", b"turn_context"),
}


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
    """A Codex thread's recorded parent, role and optional canonical native path (V568)."""

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
        The newest such file, or ``None`` when there is none or an id is not a plain token.
    """
    if host == HOST_CODEX:
        return codex_session_file(agent_id or session_id)
    ids = (session_id,) if agent_id is None else (session_id, agent_id)
    if not all(0 < len(v) <= 128 and all(c.isalnum() or c in "-_" for c in v) for v in ids):
        return None
    stem = session_id if agent_id is None else f"agent-{agent_id}"
    pattern = f"*/{stem}.jsonl" if agent_id is None else f"*/{session_id}/subagents/{stem}.jsonl"

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    files = sorted(session_root(host).glob(pattern), key=modified)
    return admitted_session_file(str(files[-1]), host=host, stem=stem) if files else None


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


def read_session(path: Path, *, host: str) -> SessionUsage:
    """The responses a session file records and the span of its recorded times.

    Args:
        path: An admitted session file (``admitted_session_file``, ``codex_session_file``).
        host: ``claude-code`` or ``codex``, which decides the file's format.

    Returns:
        The file's responses, each once, and how many usage records could not be counted.
    """
    markers = _MARKERS[host]
    responses: dict[str, ResponseUsage] = {}
    incomplete = 0
    first_at: str | None = None
    tail: deque[bytes] = deque(maxlen=8)
    turn: tuple[str | None, str | None] = (None, None)
    for line in _lines(path):
        tail.append(line)
        if first_at is None:
            first_at = _text(_record(line).get("timestamp"))
        if not any(marker in line for marker in markers):
            continue
        record = _record(line)
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


def _lines(path: Path) -> Iterator[bytes]:
    with path.open("rb") as stream:
        for line in stream:
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
    "ResponseUsage",
    "SessionUsage",
    "admitted_session_file",
    "codex_session_file",
    "hook_session_files",
    "read_session",
    "session_file",
    "session_root",
]
