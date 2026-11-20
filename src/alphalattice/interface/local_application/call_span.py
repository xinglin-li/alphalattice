"""One CLI invocation as one span: begun by the CLI, carried to the Host as W3C context, ended.

`cli.main` begins the invocation's span (`begin`): in the trace the caller's TRACEPARENT names, or
a new one, so each invocation is its own trace's root unless the caller joins it to theirs. While
the command runs, the process's own TRACEPARENT names the span and its BAGGAGE adds the span's
start, parent and command under `alphalattice.call_`; every request sends both, so each Host step is
the span's child. The Host reads the span from a request's headers (`request_call`), records it as
started the first time it sees it, and ends it when the command ends (`end`, the `CALL_PATH`
route), with its exit code. The SDK and the Host's history are imported by the Host alone.
"""

from __future__ import annotations

import re
import secrets
import time
from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote, unquote, urlsplit

from alphalattice.interface.local_application.cli_contract import (
    CLIENT_HEADER,
    INSTANCE_HEADER,
    REQUEST_PROVENANCE,
    WORKSPACE_HEADER,
    RequestProvenance,
    agent_provenance_headers,
)
from alphalattice.kernel.shared_kernel import telemetry_names as telemetry

CALL_PATH: Final = "/api/client/call"
"""Where a command ends its span at the Host its requests reached."""
VENDOR_PROVIDERS: Final = {"claude-code": "anthropic", "codex": "openai"}
"""Each agent vendor's provider, by its OpenTelemetry GenAI name (`gen_ai.provider.name`)."""
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")
_KEYS: Final = (
    telemetry.ALPHALATTICE_CALL_STARTED_US,
    telemetry.ALPHALATTICE_CALL_PARENT,
    telemetry.ALPHALATTICE_COMMAND,
)


@dataclass(frozen=True, slots=True)
class CallSpan:
    """One invocation's span; `workspace` is where the CLI ends it, unknown to the Host."""

    trace: str
    span: str
    parent: str | None
    started_us: int
    command: str
    workspace: Path | None = None


CALL: ContextVar[CallSpan | None] = ContextVar("alphalattice_call", default=None)
"""The invocation this process runs: begun by `cli.main`, given its workspace once resolved."""
_REPLACED: ContextVar[dict[str, str | None]] = ContextVar("alphalattice_call_replaced")
"""The caller's own TRACEPARENT and BAGGAGE, which `begin` replaced and `end` restores."""
REQUEST_CALL: ContextVar[CallSpan | None] = ContextVar("alphalattice_request_call", default=None)
"""The invocation the request the Host's boundary is running comes from: set by `web.py`."""


def begin(environ: MutableMapping[str, str], command: str) -> None:
    """Begin this invocation's span and name it in `environ` for every request it sends, keeping
    the TRACEPARENT and BAGGAGE it replaces for `end` to restore."""
    named = _TRACEPARENT.match(environ.get("TRACEPARENT", "").strip())
    trace, parent = (named.group(1), named.group(2)) if named else (secrets.token_hex(16), None)
    form = "".join(c for c in command if " " <= c <= "~")[:200] or "-"
    call = CallSpan(trace, secrets.token_hex(8), parent, time.time_ns() // 1000, form)
    CALL.set(call)
    saved = {name: environ.get(name) for name in ("TRACEPARENT", "BAGGAGE")}
    own = zip(_KEYS, (str(call.started_us), parent or "-", quote(form, safe="")), strict=True)
    entries = [f"{key}={value}" for key, value in own]
    environ["TRACEPARENT"] = f"00-{trace}-{call.span}-01"
    environ["BAGGAGE"] = ",".join(filter(None, (saved["BAGGAGE"], *entries)))
    _REPLACED.set(saved)


def at(workspace: Path) -> None:
    """Name the workspace whose Host this invocation's span ends at."""
    if (call := CALL.get()) is not None:
        CALL.set(replace(call, workspace=workspace))


def end(environ: MutableMapping[str, str], code: int, *, reached: bool = True) -> None:
    """End this invocation's span at its workspace's Host with its exit code, then restore what
    `begin` replaced. A command with no workspace, one that found no Host (`reached` False), or
    whose Host is gone, ends nothing: the span stays started, and the answer stands."""
    call = CALL.get()
    if call is None:
        return
    try:
        if call.workspace is not None and reached:
            _send_end(call, call.workspace, code, environ)
    finally:
        for name, value in _REPLACED.get().items():
            if value is None:
                environ.pop(name, None)
            else:
                environ[name] = value


def _send_end(call: CallSpan, workspace: Path, code: int, environ: Mapping[str, str]) -> None:
    """One request to the Host, authenticated as every request of this command is."""
    import http.client  # lazy: only a command with a workspace ends its span at a Host

    from alphalattice.interface.local_application import client  # lazy: the CLI's, not the Host's

    try:
        connection = client.LocalResearchConnection.read(workspace)
    except (OSError, ValueError, client.LocalResearchClientError):
        return
    port = urlsplit(connection.url).port
    headers = {
        "Content-Type": "application/json",
        CLIENT_HEADER: connection.token,
        WORKSPACE_HEADER: client.workspace_connection_key(workspace),
        INSTANCE_HEADER: connection.instance,
        **agent_provenance_headers(environ),
    }
    ended = {"ended_us": max(time.time_ns() // 1000, call.started_us), "exit_code": code}
    exchange = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        exchange.request("POST", CALL_PATH, body=client.request_body(ended), headers=headers)
        exchange.getresponse().read()
    except (OSError, http.client.HTTPException):
        return
    finally:
        exchange.close()


def request_call(headers: Mapping[str, str]) -> CallSpan | None:
    """The invocation a request comes from, read from its W3C headers at the Host's boundary;
    None for a request no CLI invocation sent. A malformed one is refused by name."""
    entries = dict(
        part.strip().split("=", 1)
        for part in headers.get("baggage", "").split(",")
        if part.strip().startswith(_KEYS)
    )
    if not entries:
        return None
    named = _TRACEPARENT.match(headers.get("traceparent", ""))
    started, parent = entries.get(_KEYS[0], ""), entries.get(_KEYS[1], "")
    command = unquote(entries.get(_KEYS[2], ""))
    form_ok = 0 < len(command) <= 200 and all(" " <= c <= "~" for c in command)
    parent_ok = parent == "-" or re.fullmatch(r"[0-9a-f]{16}", parent) is not None
    if named is None or not (started.isdigit() and len(started) <= 19 and parent_ok and form_ok):
        raise ValueError("local_web.call_context_invalid")
    span_parent = None if parent == "-" else parent
    return CallSpan(named.group(1), named.group(2), span_parent, int(started), command)


def record(
    call: CallSpan, provenance: RequestProvenance | None, ended: tuple[int, int] | None = None
) -> None:
    """Record an invocation's span in the Host's history: live at its first request, ended with
    its end and exit code. Its caller's session and vendor carry their GenAI names."""
    from alphalattice.kernel.data_layer.history import call as recorded  # lazy: the Host's

    who = provenance or RequestProvenance()
    ids: dict[str, Any] = {telemetry.COMMAND: call.command}
    if who.vendor is not None and who.session is not None:
        ids[telemetry.GEN_AI_CONVERSATION_ID] = who.session
        ids[telemetry.GEN_AI_PROVIDER_NAME] = VENDOR_PROVIDERS.get(who.vendor, who.vendor)
        if who.vendor == "codex":  # its thread is one agent's; Claude Code names no subagent
            ids[telemetry.GEN_AI_AGENT_ID] = who.session
    if who.goal_id is not None:
        ids[telemetry.GOAL_ID] = who.goal_id
    recorded(
        (call.trace, call.span, call.parent),
        call.started_us,
        ids,
        caller=None if who.session is None else ("agent", who.session),
        conversation=who.session,
        ended=ended,
    )


def ended(_query: Mapping[str, list[str]], payload: dict[str, Any]) -> object:
    """The `CALL_PATH` route: end the invocation the request comes from, with its exit code."""
    call, at_us, code = REQUEST_CALL.get(), payload.get("ended_us"), payload.get("exit_code")
    if call is None:
        raise ValueError("local_web.call_context_missing")
    if type(at_us) is not int or type(code) is not int or at_us < call.started_us:
        raise ValueError("local_web.call_end_invalid")
    record(call, REQUEST_PROVENANCE.get(), ended=(at_us, code))
    return {"status": "RECORDED"}


__all__ = [
    "CALL",
    "CALL_PATH",
    "REQUEST_CALL",
    "CallSpan",
    "at",
    "begin",
    "end",
    "ended",
    "record",
    "request_call",
]
