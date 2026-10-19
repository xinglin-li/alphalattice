"""Bound native usage, resumed through one checked checkpoint per exact file.

Only response-id hashes, model/effort, token counts, timestamps and exact-parent child
locators survive, and of a Claude member's own commands, the hash of the key it ran with and
the thread it named (`member_claims`). Conversation bytes are discarded. Complete records
advance the cursor; source replacement, shrink, same-size modification or a bad checksum
rebuild the checkpoint.
Only older checkpoints with child locators need discovery migration; it is not a source change.
A cumulative mismatch remains named without rescanning an unchanged file. Claude repeats
response counts by block; Codex includes cached reads/writes in its input counts.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID

from alphalattice.protocols.actor_execution.bundles import bundle_slot

HOST_CLAUDE_CODE = "claude-code"
HOST_CODEX = "codex"
_CHILD_DISCOVERY_VERSION = 2
_CLAIMS_VERSION = 1
"""A child's checkpoint without this many claims passes is read once again from its start."""
_MARKERS = {
    HOST_CLAUDE_CODE: (b'"usage":', b"ACCEPTED"),
    HOST_CODEX: (b"token_usage_record", b"turn_context", b"SubAgentActivity"),
}


class NativeUsageReadLimitError(ValueError):
    """One native record exceeds the admitted per-read or per-line byte bound."""

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
    children: tuple[str, ...] = ()
    """A Codex lead's children, by thread, in the order its file first names them."""
    read_through: int = 0
    total_bytes: int = 0
    bytes_read: int = 0
    rescan: bool = False
    cumulative_mismatch: bool = False
    members: tuple[dict[str, Any], ...] = ()
    """A Claude file's key evidence (`member_claims`), for the key's owner to verify."""

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
    """A child's recorded direct parent and role (V568)."""

    parent_thread_id: str
    agent_role: str | None


def codex_thread_spawn(thread_id: str) -> ThreadSpawn | None:
    """Which thread spawned a Codex thread, as the thread's own rollout first records it.

    Codex opens each rollout with its ``session_meta``; a native subagent's names
    ``source.subagent.thread_spawn``. Only that first record is read, up to
    `FIRST_RECORD_BYTES`, and only ``parent_thread_id`` and ``agent_role`` leave it. The
    rest of the record is discarded and every turn after it is never read (LAWS OP14).

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
    return ThreadSpawn(parent_thread_id=parent, agent_role=_text(spawn.get("agent_role")))


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
        agent_id: The exact native child id, as the lead's own Session directory names it.

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
        type(meta.get("spawnDepth")) is not int
        or meta.get("spawnDepth") != 1
        or meta.get("parentAgentId") is not None
        or record.get("sessionId") != session_id
        or record.get("agentId") != agent_id
        or record.get("isSidechain") is not True
    ):
        return None
    return ThreadSpawn(parent_thread_id=session_id, agent_role=role)


def claude_children(session_id: str) -> tuple[str, ...]:
    """The children a Claude Code lead keeps under its own Session directory.

    Only ``<project>/<session>/subagents/agent-<id>.jsonl`` beside the lead's own file is
    listed; each id is then admitted by ``claude_thread_spawn``.

    Args:
        session_id: The bound lead Session.

    Returns:
        The child ids in name order; none when the lead's file or its directory is missing.
    """
    lead = session_file(HOST_CLAUDE_CODE, session_id)
    if lead is None:
        return ()
    directory = lead.parent / session_id / "subagents"
    if directory.is_symlink() or not directory.is_dir():
        return ()
    names = sorted(
        path.name[len("agent-") : -len(".jsonl")] for path in directory.glob("agent-*.jsonl")
    )
    return tuple(name for name in names if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name))


def read_session(
    path: Path,
    *,
    host: str,
    max_bytes: int | None = None,
    max_line_bytes: int | None = None,
    checkpoint: dict[str, Any] | None = None,
    binding_key: str | None = None,
    members: bool = False,
) -> SessionUsage:
    """Read whitelisted counts, resuming complete records of this exact native file.

    Byte bounds apply per read and per line; a record that cannot fit raises a read-limit
    refusal. The optional binding hash associates the checkpoint with its cleanup owner.
    ``members`` also keeps a child's key claims (`member_claims`); a lead's are never read.
    """
    with path.open("rb") as stream:
        state = {} if checkpoint is None else checkpoint
        stat = os.fstat(stream.fileno())
        rescan = _resume(stream, state, stat, members)
        state.update(
            **({"claims": _CLAIMS_VERSION} if members else {}),
            child_discovery=_CHILD_DISCOVERY_VERSION,
            offset=state.get("offset", 0),
            size=stat.st_size,
            identity=[stat.st_dev, stat.st_ino],
            mtime=stat.st_mtime_ns,
            binding_key=binding_key,
        )
        if not state.get("hashed_ids"):
            for row in state.get("responses", ()):
                row["response_id"] = sha256(row["response_id"].encode()).hexdigest()
            state["hashed_ids"] = True
        claims = state.setdefault("members", []) if members else None
        collect = partial(member_claims, claims=claims) if claims is not None else _kept_none
        reading = _read_records(stream, host, state, (max_bytes, max_line_bytes), collect)
        through = state.get("offset", stream.tell())
        cumulative = state.get("cumulative")
        mismatch = (
            through == stat.st_size
            and cumulative is not None
            and any(
                sum(getattr(r, key) for r in reading.responses) != value
                for key, value in cumulative.items()
            )
        )
        state["restart"] = mismatch
        state["boundary"] = _boundary(stream, through)
        state["digest"] = _checksum(state)
        return replace(
            reading,
            incomplete=reading.incomplete + int(mismatch),
            read_through=through,
            total_bytes=stat.st_size,
            bytes_read=state["bytes_read"],
            rescan=rescan,
            cumulative_mismatch=mismatch,
        )


def _checksum(state: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            {k: v for k, v in state.items() if k != "digest"}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def checkpoint_children(state: dict[str, Any]) -> tuple[str, ...]:
    """Checked child locators; each still needs its own exact-parent check."""
    return (
        tuple(state.get("children", ()))
        if state.get("child_discovery") == _CHILD_DISCOVERY_VERSION
        and state.get("digest") == _checksum(state)
        else ()
    )


def _boundary(stream: BinaryIO, offset: int) -> list[str]:
    stream.seek(0)
    prefix = stream.read(min(offset, 64))
    stream.seek(max(0, offset - 64))
    tail = stream.read(min(offset, 64))
    return [
        sha256(prefix).hexdigest(),
        sha256(tail).hexdigest(),
        str(not offset or tail.endswith(b"\n")),
    ]


def _resume(stream: BinaryIO, state: dict[str, Any], stat: os.stat_result, members: bool) -> bool:
    offset = state.get("offset")
    valid = (
        state.get("digest") == _checksum(state)
        and not (
            state.get("restart")
            and (state.get("size"), state.get("mtime")) != (stat.st_size, stat.st_mtime_ns)
        )
        and state.get("identity") == [stat.st_dev, stat.st_ino]
        and type(offset) is int
        and 0 <= offset <= stat.st_size
        and not (state.get("size") == stat.st_size and state.get("mtime") != stat.st_mtime_ns)
    )
    valid = (
        valid
        and isinstance(offset, int)
        and state.get("boundary") == _boundary(stream, offset)
        and state["boundary"][-1] == "True"
    )
    rescan = bool(state) and not valid
    migrate = (
        bool(state.get("children")) and state.get("child_discovery") != _CHILD_DISCOVERY_VERSION
    ) or (members and bool(state) and state.get("claims") != _CLAIMS_VERSION)
    if not valid or migrate:
        state.clear()
    stream.seek(state.get("offset", 0))
    return rescan


def _read_records(
    stream: BinaryIO,
    host: str,
    state: dict[str, Any],
    bounds: tuple[int | None, int | None],
    collect: Callable[[Mapping[str, object]], None],
) -> SessionUsage:
    max_bytes, max_line_bytes = bounds
    markers = _MARKERS[host]
    responses = {row["response_id"]: ResponseUsage(**row) for row in state.get("responses", ())}
    incomplete = state.get("incomplete", 0)
    first_at = state.get("first_at")
    tail = deque(state.get("tail", ()), maxlen=8)
    turn = tuple(state.get("turn", (None, None)))
    children = dict.fromkeys(state.get("children", ()))
    state.update(bytes_read=0, pending=False)
    lines = (
        _lines(
            stream,
            max_bytes=max_bytes,
            max_line_bytes=max_line_bytes,
            checkpoint=state,
        )
        if stream.tell() < state["size"]
        else ()
    )
    for line in lines:
        tail.append(line)
        if first_at is None:
            first_at = _text(_record(line).get("timestamp"))
        if not any(marker in line for marker in markers):
            continue
        record = _record(line)
        if not record:
            incomplete += int(host == HOST_CODEX or b'"usage":' in line)
            continue
        if host == HOST_CODEX and b"SubAgentActivity" in line:
            child = _codex_child(record)
            if child is not None:
                children.setdefault(child)
            continue
        if host == HOST_CLAUDE_CODE:
            collect(record)
            response = _claude_response(record)
        else:
            turn = _codex_turn(record, turn)
            response = _codex_response(record, turn)
            incomplete += _codex_cumulative(record, state)
        if response is None:
            continue
        if isinstance(response, int):
            incomplete += response
            continue
        known = responses.get(response.response_id)
        responses[response.response_id] = (
            response if known is None else _replace_counts(known, response)
        )
    stamps = [
        _text(_record(line).get("timestamp")) if isinstance(line, bytes) else line for line in tail
    ]
    state.update(
        responses=[asdict(r) for r in responses.values()],
        incomplete=incomplete,
        first_at=first_at,
        tail=stamps,
        turn=turn,
        children=tuple(children),
    )
    return SessionUsage(
        responses=tuple(responses.values()),
        incomplete=incomplete + int(state.get("pending", False)),
        first_at=first_at,
        last_at=next((stamp for stamp in reversed(stamps) if stamp), None),
        children=tuple(children),
        members=tuple(state.get("members", ())),
    )


def _codex_cumulative(record: Mapping[str, object], state: dict[str, Any]) -> int:
    payload = record.get("payload")
    if (
        record.get("type") != "token_usage_record"
        or not isinstance(payload, dict)
        or "thread_token_usage" not in payload
    ):
        return 0
    counts = _codex_counts(payload["thread_token_usage"])
    if counts is None:
        return 1
    state["cumulative"] = counts
    return 0


def _codex_child(record: Mapping[str, object]) -> str | None:
    """A child thread a Codex lead's own rollout names; only its id leaves the record."""
    payload = record.get("payload")
    item = payload.get("item") if isinstance(payload, dict) else None
    if (
        record.get("type") != "event_msg"
        or not isinstance(payload, dict)
        or payload.get("type") != "item_completed"
        or not isinstance(item, dict)
        or item.get("type") != "SubAgentActivity"
        or item.get("kind") == "interacted"
    ):
        return None
    thread = item.get("agent_thread_id")
    try:
        return thread if isinstance(thread, str) and str(UUID(thread)) == thread else None
    except ValueError:
        return None


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
    return replace(
        later,
        response_id=known.response_id,
        effort=later.effort or known.effort,
        at=later.at or known.at,
    )


def _lines(
    stream: BinaryIO,
    *,
    max_bytes: int | None = None,
    max_line_bytes: int | None = None,
    checkpoint: dict[str, Any],
) -> Iterator[bytes]:
    for bound in (max_bytes, max_line_bytes):
        if bound is not None and (type(bound) is not int or bound <= 0):
            raise ValueError("native_usage_bound_invalid")
    scanned = 0
    stream.seek(checkpoint["offset"])
    while True:
        remaining = max_bytes - scanned if max_bytes is not None else None
        remaining = min(
            remaining if remaining is not None else checkpoint["size"],
            checkpoint["size"] - stream.tell(),
        )
        if remaining <= 0:
            break
        line_bound = max_line_bytes + 1 if max_line_bytes is not None else None
        bounds = [bound for bound in (remaining, line_bound) if bound is not None]
        line = stream.readline(min(bounds))
        if not line:
            break
        scanned += len(line)
        if max_line_bytes is not None and len(line) > max_line_bytes:
            raise NativeUsageReadLimitError("max_line_bytes")
        checkpoint["bytes_read"] = scanned
        if not line.endswith(b"\n"):
            if scanned == len(line) and stream.tell() < checkpoint["size"]:
                raise NativeUsageReadLimitError("max_bytes")
            checkpoint["pending"] = stream.tell() == checkpoint["size"]
            break
        checkpoint["offset"] = stream.tell()
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
        response_id=sha256(ident.encode()).hexdigest(),
        model=model,
        effort=_text(record.get("effort")),
        input_tokens=uncached,
        cache_read_tokens=read,
        cache_write_tokens=written,
        output_tokens=output,
        at=_text(record.get("timestamp")),
    )


MEMBER_CLAIMS = 64
"""The most distinct key claims one file keeps; a member runs a handful."""
_KEY = re.compile(r"--key[=\s]+['\"]?([0-9a-f]{32})\b")
_COMMITTEE = re.compile(r"\bcommittee\s+(submit|wait)\b")
_UPDATE = re.compile(r"--update[=\s]+['\"]?([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\b")
_ROLE = re.compile(r"--role[=\s]+['\"]?([A-Z]+)\b")
_BUNDLE = re.compile(r"\bbundle\s+submit\b")
_DIR = re.compile(r"--dir[=\s]+(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
_USE_FIELD = {"tool_use": "id", "tool_result": "tool_use_id"}
_SPOKE = re.compile(r'"(?:message_id|answer_reference)":\s*"(M\d+|[0-9a-f]{64})"')
_ACCEPTED = re.compile(r'"(?:answer_)?status":\s*"(?:COMMITTEE_MESSAGE_)?ACCEPTED"')


def _kept_none(record: Mapping[str, object]) -> None:
    """A lead's file: its claims are never verified, so none are kept."""


def member_claims(record: Mapping[str, object], claims: list[dict[str, Any]]) -> None:
    """Add the key evidence one Claude record holds: a command it ran, or that command's result.

    A command that names one member key and one committee thread and role, or one review bundle
    directory, is a claim: the key's SHA-256 (never the key), the thread or the bundle's slot,
    the role and whether it submits. A later result that reports the submit accepted names what
    it made: the floor message's id or the review answer's reference (``spoke``). The key's
    owner verifies a claim; the reader trusts none.
    """
    message = record.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    for block in content if isinstance(content, list) else ():
        kind = block.get("type") if isinstance(block, dict) else None
        use = block.get(_USE_FIELD.get(kind, "")) if kind else None
        if not isinstance(use, str):
            continue
        use = sha256(use.encode()).hexdigest()[:16]
        if kind == "tool_result":
            _accepted(block.get("content"), use, claims)
            continue
        found = _claim(block.get("input"), record.get("cwd"))
        if found is None:
            continue
        same = next((c for c in claims if c["id"] == found["id"]), None)
        if same is None and len(claims) < MEMBER_CLAIMS:
            claims.append(same := found)
        if same is not None and same["submit"]:
            same["use"] = use  # the latest attempt is the one a result answers


def _accepted(text: object, use: str, claims: list[dict[str, Any]]) -> None:
    """Name what the submit a result answers made, when the result reports it accepted."""
    if isinstance(text, list):
        text = " ".join(str(p.get("text", "")) for p in text if isinstance(p, dict))
    if isinstance(text, str) and _ACCEPTED.search(text):
        for claim in (c for c in claims if c.get("use") == use):
            claim["spoke"] = sorted({*claim["spoke"], *_SPOKE.findall(text)})[:8]


def _claim(tool_input: object, cwd: object) -> dict[str, Any] | None:
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    keys = set(_KEY.findall(command)) if isinstance(command, str) else set()
    if not isinstance(command, str) or len(keys) != 1:
        return None
    key = sha256(keys.pop().encode()).hexdigest()
    threads, roles = set(_UPDATE.findall(command)), set(_ROLE.findall(command))
    verb, folders = _COMMITTEE.search(command), _DIR.findall(command)
    if verb is not None and len(threads) == len(roles) == 1:
        found = {"thread": threads.pop(), "role": roles.pop(), "submit": verb[1] == "submit"}
    elif _BUNDLE.search(command) is not None and len(folders) == 1:
        named = next(part for part in folders[0] if part)
        folder = Path(re.sub(r"^/([a-zA-Z])/", r"\1:/", named) if os.name == "nt" else named)
        if not folder.is_absolute() and not isinstance(cwd, str):
            return None
        folder = folder if folder.is_absolute() else Path(str(cwd)) / folder
        found = {"reference": bundle_slot(str(folder)), "submit": True}
    else:
        return None
    ident = sha256(json.dumps([key, found], sort_keys=True).encode()).hexdigest()[:16]
    return {"id": ident, "key": key, **found, "spoke": []}


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
    counts = _codex_counts(usage)
    if counts is None:
        return 1
    return ResponseUsage(
        response_id=sha256(ident.encode()).hexdigest(),
        model=turn[0],
        effort=turn[1],
        **counts,
        at=_text(record.get("timestamp")),
    )


def _codex_counts(usage: object) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    total, cached, written, output = (
        _count(usage, k)
        for k in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "output_tokens",
        )
    )
    if total is None or cached is None or written is None or output is None:
        return None
    if total < cached + written:
        return None
    return {
        "input_tokens": total - cached - written,
        "cache_read_tokens": cached,
        "cache_write_tokens": written,
        "output_tokens": output,
    }


__all__ = [
    "HOST_CLAUDE_CODE",
    "HOST_CODEX",
    "ModelUsage",
    "NativeUsageReadLimitError",
    "ResponseUsage",
    "SessionUsage",
    "admitted_session_file",
    "checkpoint_children",
    "claude_children",
    "claude_thread_spawn",
    "codex_session_file",
    "codex_thread_spawn",
    "read_session",
    "session_file",
    "session_root",
]
