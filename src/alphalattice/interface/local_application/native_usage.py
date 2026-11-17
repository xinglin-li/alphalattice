"""Bounded Codex session-header authorization, and the usage publication value type.

Only the first session_meta record is read for native binding authorization. Numerical
usage, model time and observed agent lineage now come from the Host's telemetry catalog;
no conversation body, token record or member-key command is read here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

HOST_CLAUDE_CODE = "claude-code"
HOST_CODEX = "codex"
FIRST_RECORD_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """One model's numerical response observations, summed."""

    model: str
    efforts: tuple[str, ...]
    responses: int
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    first_at: str | None
    last_at: str | None


def session_root(
    host: str, environ: Mapping[str, str] | None = None, *, home: str | Path | None = None
) -> Path:
    """Where the host keeps its session files: the only place the reader opens one.

    Args:
        host: ``claude-code`` or ``codex``.
        environ: The environment whose home applies; this process's own when None.
        home: An explicitly owned home, taking precedence over the environment.

    Returns:
        The root directory; Claude Code honours ``CLAUDE_CONFIG_DIR`` and Codex ``CODEX_HOME``.
    """
    environ = os.environ if environ is None else environ
    if host == HOST_CLAUDE_CODE:
        return (
            Path(home or environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "projects"
        )
    return Path(home or environ.get("CODEX_HOME") or Path.home() / ".codex") / "sessions"


def admitted_session_file(
    path: object,
    *,
    host: str,
    stem: str,
    environ: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> Path | None:
    """The session file a hook named, when it is that host's file of that session or agent.

    Args:
        path: The path the hook's payload carried.
        host: The host whose session root admits it.
        stem: The file name the session or agent id gives it, without ``.jsonl``.
        environ: The environment whose home applies; this process's own when None.
        home: The same explicitly owned home used to discover this file.

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
        root = session_root(host, environ, home=home).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_relative_to(root) and resolved.is_file() else None


def codex_session_file(thread_id: str, *, home: str | Path | None = None) -> Path | None:
    """The newest Codex rollout file of a thread, found by the id its name ends with.

    Args:
        thread_id: The Codex thread (a session or a native subagent).
        home: The run's Codex home, when distinct from the reader's environment.

    Returns:
        The file, or ``None`` when the thread has none or the id is not a plain token.
    """
    if not 0 < len(thread_id) <= 128 or not all(c.isalnum() or c == "-" for c in thread_id):
        return None
    files = sorted(session_root(HOST_CODEX, home=home).glob(f"*/*/*/rollout-*{thread_id}.jsonl"))
    if not files:
        return None
    return admitted_session_file(str(files[-1]), host=HOST_CODEX, stem=files[-1].stem, home=home)


@dataclass(frozen=True, slots=True)
class ThreadSpawn:
    """A child's recorded direct parent and role."""

    parent_thread_id: str
    agent_role: str | None


def codex_thread_spawn(thread_id: str, *, home: str | Path | None = None) -> ThreadSpawn | None:
    """Which thread spawned a Codex thread, as the thread's own rollout first records it.

    Codex opens each rollout with its ``session_meta``; a native subagent's names
    ``source.subagent.thread_spawn``. Only that first record is read, up to
    `FIRST_RECORD_BYTES`, and only ``parent_thread_id`` and ``agent_role`` leave it. The
    rest of the record is discarded and every turn after it is never read (LAWS OP14).

    Args:
        thread_id: The thread, by the id its rollout's name ends with.
        home: The same owned home used for session discovery and admission.

    Returns:
        Its spawn; None for a thread with no rollout, one no thread spawned, or a first record
        past the bound, of another thread or of another shape.
    """
    path = codex_session_file(thread_id, home=home)
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
    return ThreadSpawn(parent_thread_id=parent, agent_role=_text(spawn.get("agent_role")))


def _record(line: bytes) -> Mapping[str, object]:
    """One JSON object line; a line that does not parse reads as an empty record."""
    try:
        record = json.loads(line)
    except (ValueError, RecursionError):
        return {}
    return record if isinstance(record, dict) else {}


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and 0 < len(value) <= 128 else None
