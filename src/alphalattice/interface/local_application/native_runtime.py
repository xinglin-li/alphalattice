"""Read native runtime definitions and project prospective observation readiness (ID7)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from alphalattice.interface.local_application.native_bridge import (
    JUDGMENT_ROLES,
    LEAD_ROLE,
    NativeResearchBinding,
    judgment_agent_evidence,
)

PRODUCT_MATCHER = "^alphalattice_.*$"
HOST_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
MAX_RPC_BYTES = 2 * 1024 * 1024
MAX_CHAIN_CANDIDATES = 32
MAX_EVIDENCE_IDS = 64
RuntimeRequester = Callable[[str, dict[str, Any]], object]
AcceptedAnswerReader = Callable[..., Mapping[str, Any]]


def definition_digest(project: Path, host: str) -> str:
    """Return the local definition fingerprint, never a host hash or trust assertion."""
    from alphalattice.interface.local_application.native_hook_input import (
        definition_digest as local_definition_digest,
    )

    return local_definition_digest(project, host)


class NativeRuntimeError(ValueError):
    """A safe named runtime refusal; protocol bodies and stderr are withheld."""


class ManagedRuntimeRPC:
    """Bounded read-only WebSocket RPC through the running Codex control proxy.

    The proxy relays raw bytes to the managed control socket. It launches no app
    server and writes no configuration; an absent or incompatible managed endpoint
    establishes no connection to the person's active app.
    """

    def __init__(self, project: Path, *, timeout: float = 5):
        """Bind the managed proxy's read-only connection and its bounded wait."""
        if not 0 < timeout <= 15:
            raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")
        executable = shutil.which("codex")
        codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        socket = codex_home / "app-server-control" / "app-server-control.sock"
        if executable is None or not socket.exists() or socket.is_symlink():
            raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")
        self.deadline = time.monotonic() + timeout
        self.serial = 0
        self.buffer = bytearray()
        self.chunks: queue.Queue[bytes | None] = queue.Queue()
        self.process = subprocess.Popen(
            [executable, "app-server", "proxy", "--sock", str(socket)],
            cwd=project,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=self._read, daemon=True).start()
        try:
            nonce = base64.b64encode(os.urandom(16)).decode("ascii")
            self._write(
                (
                    "GET / HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n"
                    "Connection: Upgrade\r\nSec-WebSocket-Version: 13\r\n"
                    f"Sec-WebSocket-Key: {nonce}\r\n\r\n"
                ).encode("ascii")
            )
            header = bytearray()
            while not header.endswith(b"\r\n\r\n"):
                header.extend(self._take(1))
                if len(header) > 16 * 1024:
                    raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")
            lines = header.decode("ascii").split("\r\n")
            fields = {
                key.strip().lower(): value.strip()
                for line in lines[1:]
                if ":" in line
                for key, value in [line.split(":", 1)]
            }
            accept = base64.b64encode(
                hashlib.sha1((nonce + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
            ).decode("ascii")
            if (
                lines[0].split(" ")[1:2] != ["101"]
                or fields.get("sec-websocket-accept") != accept
                or fields.get("upgrade", "").lower() != "websocket"
            ):
                raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")
            self.request(
                "initialize",
                {
                    "clientInfo": {"name": "alphalattice_native_readiness", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            self._send({"method": "initialized", "params": {}})
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        try:
            assert self.process.stdout is not None
            total = 0
            while chunk := os.read(self.process.stdout.fileno(), 64 * 1024):
                total += len(chunk)
                if total > MAX_RPC_BYTES:
                    break
                self.chunks.put(chunk)
        except OSError:
            pass
        finally:
            self.chunks.put(None)

    def _take(self, count: int) -> bytes:
        while len(self.buffer) < count:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")
            try:
                chunk = self.chunks.get(timeout=remaining)
            except queue.Empty:
                raise NativeRuntimeError("native_bridge.runtime_connection_unavailable") from None
            if chunk is None or len(self.buffer) + len(chunk) > MAX_RPC_BYTES:
                raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def _write(self, value: bytes) -> None:
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(value)
            self.process.stdin.flush()
        except (OSError, ValueError):
            raise NativeRuntimeError("native_bridge.runtime_connection_unavailable") from None

    def _frame(self, opcode: int, payload: bytes) -> None:
        if len(payload) > MAX_RPC_BYTES:
            raise NativeRuntimeError("native_bridge.runtime_response_invalid")
        mask = os.urandom(4)
        length = len(payload)
        header = bytes([0x80 | opcode])
        header += (
            bytes([0x80 | length])
            if length < 126
            else b"\xfe" + struct.pack("!H", length)
            if length < 65536
            else b"\xff" + struct.pack("!Q", length)
        )
        self._write(
            header + mask + bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        )

    def _send(self, document: dict[str, Any]) -> None:
        self._frame(1, json.dumps(document, separators=(",", ":")).encode())

    def _message(self) -> object:
        assembled = bytearray()
        started = False
        while True:
            first, second = self._take(2)
            opcode = first & 15
            length = second & 127
            if first & 0x70 or second & 0x80:
                raise NativeRuntimeError("native_bridge.runtime_response_invalid")
            if length == 126:
                length = struct.unpack("!H", self._take(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._take(8))[0]
            if length > MAX_RPC_BYTES or len(assembled) + length > MAX_RPC_BYTES:
                raise NativeRuntimeError("native_bridge.runtime_response_invalid")
            payload = self._take(length)
            if opcode == 9 and first & 0x80 and length <= 125:
                self._frame(10, payload)
                continue
            if opcode == 10 and first & 0x80 and length <= 125:
                continue
            if opcode == 8:
                raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")
            if opcode == 1 and not started:
                started = True
            elif opcode != 0 or not started:
                raise NativeRuntimeError("native_bridge.runtime_response_invalid")
            assembled.extend(payload)
            if first & 0x80:
                try:
                    return json.loads(assembled)
                except (ValueError, UnicodeError, RecursionError):
                    raise NativeRuntimeError("native_bridge.runtime_response_invalid") from None

    def request(self, method: str, params: dict[str, Any]) -> object:
        """Permit initialization, definition and attachment reads; no mutation has a path."""
        if method not in {"initialize", "hooks/list", "thread/read", "thread/loaded/list"}:
            raise NativeRuntimeError("native_bridge.runtime_read_only")
        self.serial += 1
        self._send({"id": self.serial, "method": method, "params": params})
        while time.monotonic() < self.deadline:
            message = self._message()
            if not isinstance(message, dict):
                raise NativeRuntimeError("native_bridge.runtime_response_invalid")
            if message.get("id") == self.serial:
                if "error" in message:
                    raise NativeRuntimeError("native_bridge.runtime_request_refused")
                if "result" not in message:
                    raise NativeRuntimeError("native_bridge.runtime_response_invalid")
                return message["result"]
        raise NativeRuntimeError("native_bridge.runtime_connection_unavailable")

    def close(self) -> None:
        """Reap only this proxy process; the person's managed app remains running."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()


def validate_runtime_definitions(
    raw: object,
    *,
    project: Path,
    workspace: Path,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select exact host definition facts from hooks/list; no command enters the receipt."""
    expected_cwds = {str(path.resolve()) for path in (project, workspace)}
    config = project / ".codex/config.toml"
    groups: list[dict[str, Any]] = []
    proved: set[str] = set()
    hashes: dict[str, set[str]] = {"subagentStart": set(), "subagentStop": set()}
    expected: dict[str, dict[str, Any]] = {}
    try:
        if any(
            path.is_symlink() or (os.name == "nt" and path.is_junction())
            for path in (config, config.parent)
        ):
            raise ValueError
        config = config.resolve()
        with config.open("rb") as stream:
            declaration = stream.read(MAX_RPC_BYTES + 1)
        if len(declaration) > MAX_RPC_BYTES:
            raise ValueError
        declared = tomllib.loads(declaration.decode())["hooks"]
        for event, native in (("SubagentStart", "subagentStart"), ("SubagentStop", "subagentStop")):
            declaration_groups = declared[event]
            matched = [
                (index, group)
                for index, group in enumerate(declaration_groups)
                if isinstance(group, dict) and group.get("matcher") == PRODUCT_MATCHER
            ]
            if len(matched) != 1:
                raise ValueError
            index, group = matched[0]
            handlers = group["hooks"]
            if (
                not isinstance(handlers, list)
                or len(handlers) != 1
                or not isinstance(handlers[0], dict)
            ):
                raise ValueError
            handler = handlers[0]
            if (
                handler.get("type") != "command"
                or not isinstance(handler.get("command"), str)
                or type(handler.get("timeout")) is not int
                or handler["timeout"] <= 0
                or type(handler.get("async", False)) is not bool
            ):
                raise ValueError
            expected[native] = {
                "handlerType": "command",
                "command": handler["command"],
                "timeoutSec": handler["timeout"],
                "async": handler.get("async", False),
                "isManaged": False,
                "index": index,
            }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        expected = {}
    data = raw.get("data") if isinstance(raw, dict) else None
    shape_valid = (
        isinstance(data, list) and len(data) == len(expected_cwds) and set(expected) == set(hashes)
    )
    for group in data[: len(expected_cwds) + 1] if isinstance(data, list) else []:
        if not isinstance(group, dict):
            shape_valid = False
            continue
        cwd = group.get("cwd")
        safe_cwd = str(Path(cwd).resolve()) if isinstance(cwd, str) and len(cwd) <= 512 else None
        hooks = group.get("hooks")
        hooks = (
            [
                hook
                for hook in hooks
                if isinstance(hook, dict)
                and hook.get("source") == "project"
                and hook.get("matcher") == PRODUCT_MATCHER
            ]
            if isinstance(hooks, list)
            else []
        )
        if len(hooks) > MAX_EVIDENCE_IDS:
            shape_valid = False
            hooks = hooks[:MAX_EVIDENCE_IDS]
        definitions = []
        for hook in hooks:
            definitions.append(
                {
                    key: hook.get(key)
                    for key in (
                        "eventName",
                        "source",
                        "sourcePath",
                        "matcher",
                        "enabled",
                        "currentHash",
                        "trustStatus",
                        "isManaged",
                        "handlerType",
                        "timeoutSec",
                        "async",
                    )
                    if isinstance(hook.get(key), (str, bool, int)) and len(str(hook[key])) <= 512
                }
            )
        valid = (
            safe_cwd in expected_cwds
            and safe_cwd not in proved
            and group.get("errors") == []
            and group.get("warnings") == []
            and len(hooks) == 2
            and {hook.get("eventName") for hook in hooks} == set(hashes)
        )
        for hook in hooks:
            current = hook.get("currentHash")
            source = hook.get("sourcePath")
            event = hook.get("eventName")
            event_key = "subagent_start" if event == "subagentStart" else "subagent_stop"
            definition_index = expected.get(str(event), {}).get("index")
            exact = (
                event in hashes
                and hook.get("enabled") is True
                and hook.get("trustStatus") == "trusted"
                and isinstance(current, str)
                and HOST_HASH.fullmatch(current) is not None
                and isinstance(source, str)
                and len(source) <= 512
                and Path(source).resolve() == config
                and Path(source).is_absolute()
                and not Path(source).is_symlink()
                and str(event) in expected
                and hook.get("isManaged") is False
                and type(hook.get("async")) is bool
                and type(hook.get("timeoutSec")) is int
                and hook.get("key") == f"{source}:{event_key}:{definition_index}:0"
                and all(
                    hook.get(key) == value
                    for key, value in expected.get(str(event), {}).items()
                    if key != "index"
                )
                and (expected_hashes is None or expected_hashes.get(str(event)) == current)
            )
            valid = valid and exact
            if exact:
                hashes[str(event)].add(current)
        if valid:
            proved.add(str(safe_cwd))
        groups.append(
            {
                "cwd": safe_cwd,
                "errors_count": len(group["errors"])
                if isinstance(group.get("errors"), list)
                else None,
                "warnings_count": len(group["warnings"])
                if isinstance(group.get("warnings"), list)
                else None,
                "definitions": definitions,
            }
        )
    trusted = bool(
        shape_valid
        and proved == expected_cwds
        and all(len(values) == 1 for values in hashes.values())
    )
    return {
        "status": "RUNTIME_PROJECT_HOOKS_TRUSTED" if trusted else "RUNTIME_HOOK_TRUST_UNPROVED",
        "host_trust": "TRUSTED" if trusted else "UNPROVED",
        "hooks": groups,
        "definition_hashes": {
            name: next(iter(values)) for name, values in hashes.items() if len(values) == 1
        },
        "trust_changed": False,
        **({"reason": "native_bridge.runtime_definition_unproved"} if not trusted else {}),
    }


def runtime_definitions(
    project: Path,
    workspace: Path,
    host: str,
    *,
    session_id: str | None = None,
    requester: RuntimeRequester | None = None,
) -> dict[str, Any]:
    """Read the managed runtime once, or report precisely what could not be checked."""
    if host != "codex":
        return {
            "status": "NOT_CHECKED",
            "host_trust": "NOT_CHECKED",
            "reason": "native_bridge.runtime_not_checked",
            "trust_changed": False,
        }
    rpc: ManagedRuntimeRPC | None = None
    exact_runtime = requester is not None
    try:
        if requester is None:
            rpc = ManagedRuntimeRPC(project)
            requester = rpc.request
        raw = requester("hooks/list", {"cwds": [str(project), str(workspace)]})
        result = validate_runtime_definitions(raw, project=project, workspace=workspace)
        attachment = "NOT_PROVED"
        if session_id is not None:
            thread_result = requester(
                "thread/read", {"threadId": session_id, "includeTurns": False}
            )
            thread = thread_result.get("thread") if isinstance(thread_result, dict) else None
            status = thread.get("status") if isinstance(thread, dict) else None
            cursor: str | None = None
            seen: set[str] = set()
            loaded: set[str] = set()
            for _ in range(100):
                page = requester(
                    "thread/loaded/list",
                    {"limit": 100, **({"cursor": cursor} if cursor is not None else {})},
                )
                if not isinstance(page, dict):
                    raise NativeRuntimeError("native_bridge.runtime_response_invalid")
                identifiers = page.get("data")
                if not isinstance(identifiers, list) or any(
                    not isinstance(value, str) for value in identifiers
                ):
                    raise NativeRuntimeError("native_bridge.runtime_response_invalid")
                loaded.update(identifiers)
                following = page.get("nextCursor")
                if following is None:
                    break
                if not isinstance(following, str) or not following or following in seen:
                    raise NativeRuntimeError("native_bridge.runtime_response_invalid")
                seen.add(following)
                cursor = following
            else:
                raise NativeRuntimeError("native_bridge.runtime_response_invalid")
            if (
                exact_runtime
                and isinstance(thread, dict)
                and thread.get("id") == session_id
                and isinstance(thread.get("cwd"), str)
                and Path(thread["cwd"]).resolve() in {project.resolve(), workspace.resolve()}
                and isinstance(status, dict)
                and status.get("type") in {"idle", "active"}
                and session_id in loaded
            ):
                attachment = "PROVED"
        if attachment != "PROVED":
            return {
                "status": "NOT_CHECKED",
                "host_trust": "NOT_CHECKED",
                "reason": "native_bridge.runtime_attachment_unproved",
                "attachment_diagnostic": "ACTIVE_APP_RPC_NOT_EXPOSED"
                if not exact_runtime
                else "EXACT_SESSION_NOT_LOADED_IN_PROJECT",
                "definition_check": result,
                "connection": "MANAGED_CONTROL_RPC",
                "active_session_attachment": attachment,
                "proof_surface": "CODEX_APP_RUNTIME",
                "trust_changed": False,
            }
        return {
            **result,
            "connection": "EXACT_RUNTIME_RPC",
            "active_session_attachment": attachment,
            "proof_surface": "CODEX_APP_RUNTIME",
        }
    except (NativeRuntimeError, OSError, ValueError, TypeError) as error:
        reason = (
            str(error)
            if isinstance(error, NativeRuntimeError)
            else "native_bridge.runtime_connection_unavailable"
        )
        return {
            "status": "NOT_CHECKED",
            "host_trust": "NOT_CHECKED",
            "reason": reason,
            "trust_changed": False,
            "connection": "UNPROVED",
            "active_session_attachment": "NOT_PROVED",
            "proof_surface": "CODEX_APP_RUNTIME",
        }
    finally:
        if rpc is not None:
            rpc.close()


def _aware(value: object) -> datetime | None:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return (
            parsed.astimezone(UTC)
            if parsed.tzinfo is not None and parsed.utcoffset() is not None
            else None
        )
    except (ValueError, TypeError):
        return None


def _fresh_usage(row: Mapping[str, Any], checkpoint: datetime | None) -> bool:
    subject = (row.get("payload") or {}).get("subject") or {}
    last = _aware(subject.get("last_at"))
    responses = subject.get("responses")
    return bool(
        checkpoint is not None
        and last is not None
        and last >= checkpoint
        and isinstance(responses, str)
        and responses.isdecimal()
        and int(responses) > 0
        and subject.get("input_channel") in {"CODEX_SESSION_FILE", "CLAUDE_CODE_SESSION_FILE"}
    )


def retained_history(
    read_page: Callable[..., Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]] | None, dict[str, Any]]:
    """Read complete admitted external pages; failure never becomes an empty ledger."""
    rows: list[Mapping[str, Any]] = []
    before: int | None = None
    epoch: object = None
    try:
        for index in range(100):
            page = read_page(before=before)
            items = page.get("items")
            if page.get("disposition") == "UNAVAILABLE" or not isinstance(items, list):
                raise ValueError
            if index and page.get("epoch") != epoch:
                raise ValueError
            epoch = page.get("epoch")
            if any(not isinstance(row, dict) for row in items):
                raise ValueError
            rows.extend(items)
            if not page.get("more"):
                return rows, {"status": "AVAILABLE", "pages": index + 1, "epoch": epoch}
            oldest = page.get("oldest")
            if (
                not isinstance(oldest, int)
                or isinstance(oldest, bool)
                or oldest <= 0
                or (before is not None and oldest >= before)
            ):
                raise ValueError
            before = oldest
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None, {"status": "UNAVAILABLE", "reason": "native_bridge.history_unavailable"}


def accepted_answer_metadata(
    read_external: AcceptedAnswerReader | None,
    *,
    observation_id: object,
    expected: Mapping[str, object],
) -> dict[str, Any]:
    """Read one selected sealed answer's exact bindings without retaining its contribution.

    An AVAILABLE owner readback verifies the accepted record, not the claimed
    delivery of native lifecycle events alongside it.
    """
    unproved = {"status": "NOT_PROVED", "reason": "credited_accepted_answer"}
    if (
        read_external is None
        or not isinstance(observation_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", observation_id) is None
    ):
        return unproved
    try:
        selected = read_external(observation_id=observation_id)
        accepted = selected.get("accepted_answer")
        fields = (
            "native_host",
            "native_session_id",
            "native_agent_id",
            "role",
            "submitted_by",
            "task_id",
            "bundle_reference",
            "answer_reference",
        )
        if (
            selected.get("observation_id") != observation_id
            or not isinstance(accepted, dict)
            or accepted.get("status") != "AVAILABLE"
            or accepted.get("observation_id") != observation_id
            or any(
                not isinstance(expected.get(field), str)
                or not expected[field]
                or accepted.get(field) != expected[field]
                for field in fields
            )
            or not isinstance(accepted.get("answer_digest"), str)
            or re.fullmatch(r"[0-9a-f]{64}", accepted["answer_digest"]) is None
        ):
            return unproved
        return {"status": "AVAILABLE", "answer_digest": accepted["answer_digest"]}
    except (OSError, ValueError, TypeError, AttributeError):
        return unproved


def readiness(
    project: Path,
    binding: NativeResearchBinding | None,
    events: Iterable[Mapping[str, Any]] | None,
    *,
    goal_id: str | None = None,
    runtime: Mapping[str, Any] | None = None,
    requester: RuntimeRequester | None = None,
    read_external: AcceptedAnswerReader | None = None,
    history_available: bool = True,
) -> dict[str, Any]:
    """Project retained native claims and exact accepted records, with every missing link.

    Runtime facts come from the host RPC and accepted answers from their selected
    sealed-record owner. Retained lifecycle labels and producer reservations do
    not verify native delivery; no current owner supplies that authority. These
    chains remain diagnostic claims. Research operations are never gated here.
    """
    missing: list[str] = ["native_event_delivery_unverified"]
    host = "codex" if binding is None else binding.host
    try:
        fingerprint: str | None = definition_digest(project, host)
    except (ValueError, OSError):
        fingerprint = None
        missing.append("local_lifecycle_definitions")
    snapshot = (
        dict(runtime)
        if runtime is not None
        else (
            runtime_definitions(
                project, binding.workspace, host, session_id=binding.session_id, requester=requester
            )
            if binding is not None
            else {"status": "NOT_CHECKED", "host_trust": "NOT_CHECKED", "trust_changed": False}
        )
    )
    if snapshot.get("status") != "RUNTIME_PROJECT_HOOKS_TRUSTED":
        missing.append("actual_runtime_definitions_and_trust")
    if snapshot.get("active_session_attachment") != "PROVED":
        missing.append("active_native_session_attachment")
    checkpoint = _aware(getattr(binding, "observation_started_at", None))
    if binding is None:
        missing.append("native_session_binding")
    if checkpoint is None:
        missing.append("prospective_observation_checkpoint")
    rows = [] if events is None else list(events)
    available = events is not None and history_available
    if not available:
        missing.append("native_history")
    fresh: list[Mapping[str, Any]] = []
    session = None if binding is None else binding.session_id
    for row in rows:
        subject = (row.get("payload") or {}).get("subject") or {}
        if row.get("payload") is None and session in (row.get("correlation_ids") or ()):
            available = False
            if "retained_history" not in missing:
                missing.append("retained_history")
        if subject.get("native_session_id") != session or subject.get("native_host") != host:
            continue
        observed, occurred = _aware(row.get("observed_at")), _aware(row.get("occurred_at"))
        if (
            checkpoint is None
            or observed is None
            or occurred is None
            or min(observed, occurred) < checkpoint
        ):
            continue
        kind = (row.get("payload") or {}).get("event_kind")
        if kind in {"NATIVE_SUBAGENT_START_HOOK", "NATIVE_SUBAGENT_STOP_HOOK"} and (
            fingerprint is None or subject.get("native_definition_digest") != fingerprint
        ):
            continue
        fresh.append(row)
    roles: dict[str, dict[str, Any]] = {}
    chains: list[dict[str, Any]] = []
    for role in JUDGMENT_ROLES:
        candidates = [
            row
            for row in fresh
            if (row.get("payload") or {}).get("event_kind") == "NATIVE_COORDINATION_MESSAGE"
            and (subject := (row.get("payload") or {}).get("subject") or {}).get("input_channel")
            == "PRODUCT_ACCEPTED_ANSWER"
            and subject.get("message_kind") == "answer"
            and subject.get("authorship_basis") == "HOOK"
            and isinstance(subject.get("bundle_reference"), str)
            and re.fullmatch(r"[0-9a-f]{64}", subject["bundle_reference"])
            and any(
                subject.get("role") == card or str(subject.get("role", "")).startswith(card + "_")
                for card in JUDGMENT_ROLES[role]
            )
        ]
        if len(candidates) > MAX_CHAIN_CANDIDATES:
            roles[role] = {
                "status": "NOT_PROVED",
                "missing": ["native_evidence_projection_budget"],
                "candidate_count": len(candidates),
                "chains": [],
            }
            if "native_evidence_projection_budget" not in missing:
                missing.append("native_evidence_projection_budget")
            continue
        role_missing = {
            "fresh_same_definition_start",
            "exact_assignment",
            "credited_accepted_answer",
            "native_subagent_stop",
            "child_usage",
            "lead_usage",
            "exact_goal_binding",
            "native_event_delivery_unverified",
        }
        role_chains = []
        for accepted in candidates:
            answer_subject = accepted["payload"]["subject"]
            decision = judgment_agent_evidence(
                fresh,
                host=host,
                session_id=session or "",
                bundle_role=role,
                bundle_reference=answer_subject["bundle_reference"],
                history_available=available,
            )
            run = decision["agent_run"]
            chain_missing = {
                "native_event_delivery_unverified",
                *(decision["diagnostic"].get("missing") or []),
            }
            proof = dict(decision["evidence"])
            for key, value in proof.items():
                if isinstance(value, list) and len(value) > MAX_EVIDENCE_IDS:
                    proof[key] = value[:MAX_EVIDENCE_IDS]
                    chain_missing.add("native_evidence_projection_budget")
            child = run.get("agent_id")
            credited = bool(
                child
                and child != session
                and run.get("basis") == "HOOK"
                and child == answer_subject.get("native_agent_id")
                and answer_subject.get("submitted_by") == session
                and run.get("role") == answer_subject.get("role")
                and re.fullmatch(r"[0-9a-f]{64}", str(answer_subject.get("answer_reference", "")))
            )
            try:
                source_task = answer_subject.get("reference")
                credited = (
                    credited
                    and isinstance(source_task, str)
                    and str(UUID(source_task)) == source_task
                )
            except ValueError:
                credited = False
            owner_answer = accepted_answer_metadata(
                read_external,
                observation_id=accepted.get("observation_id"),
                expected={
                    "native_host": host,
                    "native_session_id": session,
                    "native_agent_id": child,
                    "role": run.get("role"),
                    "submitted_by": session,
                    "task_id": answer_subject.get("reference"),
                    "bundle_reference": answer_subject["bundle_reference"],
                    "answer_reference": answer_subject.get("answer_reference"),
                },
            )
            credited = credited and owner_answer["status"] == "AVAILABLE"
            if not credited:
                chain_missing.add("credited_accepted_answer")
            subject_goal = answer_subject.get("goal_id")
            try:
                exact_goal = (
                    isinstance(subject_goal, str)
                    and str(UUID(subject_goal)) == subject_goal
                    and (goal_id is None or subject_goal == goal_id)
                )
            except ValueError:
                exact_goal = False
            if not exact_goal:
                chain_missing.add("exact_goal_binding")
            starts = [
                row
                for row in fresh
                if row.get("observation_id") == decision["evidence"].get("start_observation_id")
            ]
            assignments = [
                row
                for row in fresh
                if row.get("observation_id")
                in decision["evidence"].get("assignment_observation_ids", [])
            ]
            if not starts:
                chain_missing.add("fresh_same_definition_start")
            if not assignments or any(
                row["payload"]["subject"].get("goal_id") != subject_goal for row in assignments
            ):
                chain_missing.add("exact_assignment")
            if any(row["payload"]["subject"].get("goal_id") != subject_goal for row in starts):
                chain_missing.add("exact_goal_binding")
            if not any(
                int(start.get("ordinal") or 0)
                < int(assignment.get("ordinal") or 0)
                < int(accepted.get("ordinal") or 0)
                for start in starts
                for assignment in assignments
            ):
                chain_missing.add("ordered_start_assignment_acceptance")
            stops = [
                row
                for row in fresh
                if (row.get("payload") or {}).get("event_kind") == "NATIVE_SUBAGENT_STOP_HOOK"
                and row["payload"]["subject"].get("native_agent_id") == child
                and row["payload"]["subject"].get("role") == run.get("role")
                and any(
                    int(row.get("ordinal") or 0) > int(assignment.get("ordinal") or 0)
                    for assignment in assignments
                )
            ]
            if not stops:
                chain_missing.add("native_subagent_stop")
            if any(row["payload"]["subject"].get("goal_id") != subject_goal for row in stops):
                chain_missing.add("exact_goal_binding")
            usages = [
                row
                for row in fresh
                if (row.get("payload") or {}).get("event_kind") == "NATIVE_AGENT_USAGE"
                and _fresh_usage(row, checkpoint)
            ]
            child_usage = [
                row
                for row in usages
                if row["payload"]["subject"].get("native_agent_id") == child
                and row["payload"]["subject"].get("role") == run.get("role")
                and row["payload"]["subject"].get("model") == run.get("model")
                and any(
                    int(row.get("ordinal") or 0) > int(start.get("ordinal") or 0)
                    for start in starts
                )
            ]
            lead_usage = [
                row
                for row in usages
                if row["payload"]["subject"].get("native_agent_id") == session
                and row["payload"]["subject"].get("role") == LEAD_ROLE
            ]
            if not child_usage:
                chain_missing.add("child_usage")
            if not lead_usage:
                chain_missing.add("lead_usage")
            if any(
                row["payload"]["subject"].get("goal_id") != subject_goal
                for row in child_usage + lead_usage
            ):
                chain_missing.add("exact_goal_binding")
            if any(len(items) > MAX_EVIDENCE_IDS for items in (stops, child_usage, lead_usage)):
                chain_missing.add("native_evidence_projection_budget")
            chain = {
                "status": "retained_not_proved",
                "bundle_role": role,
                "bundle_reference": answer_subject["bundle_reference"],
                "source_task_id": answer_subject.get("reference"),
                "answer_reference": answer_subject.get("answer_reference"),
                "goal_id": subject_goal if exact_goal else None,
                "agent_run": run,
                "accepted_answer": owner_answer,
                "missing": sorted(chain_missing),
                "evidence": {
                    **proof,
                    "accepted_observation_id": accepted.get("observation_id"),
                    "stop_observation_ids": [
                        row.get("observation_id") for row in stops[:MAX_EVIDENCE_IDS]
                    ],
                    "child_usage_observation_ids": [
                        row.get("observation_id") for row in child_usage[:MAX_EVIDENCE_IDS]
                    ],
                    "lead_usage_observation_ids": [
                        row.get("observation_id") for row in lead_usage[:MAX_EVIDENCE_IDS]
                    ],
                },
            }
            role_chains.append(chain)
            role_missing = set().union(*(set(chain["missing"]) for chain in role_chains))
        chains.extend(role_chains)
        roles[role] = {
            "status": "NOT_PROVED",
            "missing": sorted(role_missing),
            "chains": role_chains,
        }
    missing.append("complete_fresh_native_chain")
    ready = not missing
    return {
        "status": "READY" if ready else "REFUSED",
        "failure_code": None if ready else "native_bridge.readiness_incomplete",
        "host": host,
        "session_id": session,
        "host_trust": snapshot.get("host_trust", "NOT_CHECKED"),
        "foreground_attachment": "NOT_PROVED",
        "observation_started_at": None if checkpoint is None else checkpoint.isoformat(),
        "local_definition_digest": fingerprint,
        "runtime": snapshot,
        "missing": missing,
        "roles": roles,
        "evidence": chains,
        "research_nonblocking": True,
        "trust_changed": False,
        "projection_limits": {
            "chain_candidates_per_role": MAX_CHAIN_CANDIDATES,
            "evidence_ids_per_link": MAX_EVIDENCE_IDS,
        },
    }
