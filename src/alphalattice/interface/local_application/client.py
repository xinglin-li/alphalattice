"""Local client transport over the running Host, never another workspace writer."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Literal, Self
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4

from alphalattice.interface.local_application.cli_contract import (
    AGENT_SESSION_HEADER,
    AGENT_VENDOR_HEADER,
    ANSWER_LANGUAGE,
    CLIENT_HEADER,
    EXIT_CODES,
    INSTANCE_HEADER,
    MAXIMUM_REQUEST_BODY_BYTES,
    WAIT_EXITS,
    WORKSPACE_FROM,
    WORKSPACE_HEADER,
    Outcome,
    agent_provenance_headers,
    choices,
    client_refusal,
    command,
    command_table,
    entry,
    envelope,
    join,
    named_read,
    offered_requests,
    outcome_of,
    shell,
    task_state,
)

CONNECTION_NAME = "local-research-connection.json"
REFERENCE_LEDGER_NAME = "reference-ledger.txt"
"""Beside the connection: every hash and id the Host has answered a client with (V393)."""
WHOLE_REFERENCE = re.compile(
    r"\b(?:[0-9a-f]{64}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b"
)
"""A whole hash (a SHA-256 hex digest) or id (a UUID)."""
DEFAULT_HTTP_WAIT_SECONDS = 120.0
"""Transport waiting, not a Task budget; verified historical PLAN/readback may exceed 30 s."""


def request_body(document: dict[str, Any]) -> bytes:
    """A request's JSON body, its text in UTF-8 rather than escaped.

    Escaped, a Chinese character takes six bytes instead of three, so a request whose fields
    are each within their limits could exceed the Host's body limit (V147).

    Args:
        document: The request document.

    Returns:
        The body the Host reads.
    """
    return json.dumps(document, default=_json_value, ensure_ascii=False).encode("utf-8")


def workspace_connection_key(workspace: Path) -> str:
    """Bind local transport to the requested location, not a supplied descriptor id."""
    return sha256(os.fsencode(os.path.normcase(str(workspace.resolve())))).hexdigest()


class LocalResearchClientError(ValueError):
    """A bounded client failure, never an exception containing connection secrets."""

    document_location: dict[str, int | str] | None = None
    document_size: dict[str, int | None] | None = None
    """A document over the request bound: its `bytes` (None when read from standard input)
    and the `limit_bytes` it exceeds (V372)."""
    short_reference: dict[str, object] | None = None
    """A short reference read back as no value or several: its `field`, its `value` and the
    `candidates` it begins (V393)."""
    sections: list[str] | None = None
    """The available sibling paths when a saved answer lacks the requested section."""


@dataclass(frozen=True, slots=True)
class LocalResearchConnection:
    """Describe a workspace's local Host connection.

    The Host writes this descriptor at start, and each client call reads it
    locally to locate and authenticate to that Host.
    """

    workspace: str
    workspace_id: str
    url: str
    instance: str
    token: str = field(repr=False)
    schema_id: Literal["local-research-connection"] = "local-research-connection"

    def __post_init__(self) -> None:
        """Reject malformed or non-loopback connection descriptors.

        Raises:
            ValueError: If the descriptor is malformed or points outside loopback.
        """
        values = (self.schema_id, self.workspace, self.workspace_id, self.url, self.instance)
        if (
            not all(isinstance(value, str) for value in (*values, self.token))
            or self.schema_id != "local-research-connection"
            or not self.workspace_id
            or not 16 <= len(self.instance) <= 128
            or not 32 <= len(self.token) <= 128
        ):
            raise ValueError("local_client.connection_unreadable")
        target = urlsplit(self.url)
        if (
            not Path(self.workspace).is_absolute()
            or target.scheme != "http"
            or target.hostname != "127.0.0.1"
            or not target.port
            or target.username is not None
            or target.password is not None
            or target.path not in ("", "/")
            or target.query
            or target.fragment
            or self.url.rstrip("/") != f"http://127.0.0.1:{target.port}"
        ):
            raise ValueError("local_client.connection_not_local")

    def to_json(self) -> str:
        """Serialize the connection descriptor for its controlled local file.

        Returns:
            Compact JSON including the local authentication token.
        """
        return json.dumps(
            {
                "schema_id": self.schema_id,
                "workspace": self.workspace,
                "workspace_id": self.workspace_id,
                "url": self.url,
                "instance": self.instance,
                "token": self.token,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def path(workspace: Path) -> Path:
        """Locate the connection descriptor within the workspace runtime.

        Args:
            workspace: The served workspace root.

        Returns:
            The contained connection-file path.

        Raises:
            LocalResearchClientError: If the path is a symlink or escapes the root.
        """
        root = workspace.resolve()
        path = root / "runtime" / CONNECTION_NAME
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise LocalResearchClientError("local_client.connection_path_escapes_workspace")
        return path

    @classmethod
    def read(cls, workspace: Path) -> Self:
        """Read a bounded descriptor for this exact workspace.

        Args:
            workspace: The workspace expected by the client.

        Returns:
            A validated connection to its running Host.

        Raises:
            LocalResearchClientError: If the Host or descriptor is unavailable or unsafe.
        """
        path = cls.path(workspace)
        try:
            if path.stat().st_size > 4096:
                raise ValueError("oversized connection")
            record = json.loads(path.read_bytes())
            if not isinstance(record, dict):
                raise ValueError("not a connection record")
            value = cls(**record)
        except FileNotFoundError as error:
            raise LocalResearchClientError("local_client.service_not_running") from error
        except (ValueError, TypeError, OSError) as error:
            raise LocalResearchClientError("local_client.connection_unreadable") from error
        if Path(value.workspace).resolve() != workspace.resolve():
            raise LocalResearchClientError("local_client.workspace_mismatch")
        return value


def _json_value(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise LocalResearchClientError("local_client.document_not_json_compatible")


class LocalResearchClient:
    """Read the connection anew per invocation; never keep a last spec or result."""

    def __init__(
        self,
        workspace: Path,
        *,
        timeout: float = DEFAULT_HTTP_WAIT_SECONDS,
        goal: str | None = None,
    ):
        """Bind a client to one served workspace, its HTTP wait limit and a named goal."""
        self.workspace = workspace.resolve()
        self.connection = LocalResearchConnection.read(workspace)
        self.timeout = timeout
        named = {"goal_id": goal}
        whole_references(named, self.workspace)  # `--goal` as the compact view shows it (V393)
        self.goal = named["goal_id"]

    def request(
        self,
        document: dict[str, Any] | None = None,
        *,
        include_context: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send an operation, or read the session.

        The session is the operations (`OPERATION_LIST`), or the workspace with its context
        (`WORKSPACE_SHOW`).

        Args:
            document: The operation document, or ``None`` for the session.
            include_context: Whether a session read is the workspace with its context.
            timeout: A shorter HTTP wait than the client's, for a waiter's remaining time.

        Returns:
            The Host owner's unaltered answer body.
        """
        return self.exchange(document, include_context=include_context, timeout=timeout)[0]

    def exchange(
        self,
        document: dict[str, Any] | None = None,
        *,
        include_context: bool = False,
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        """One request returns both display data and its unaltered transport bytes."""
        if document is None:
            document = {"operation": "WORKSPACE_SHOW" if include_context else "OPERATION_LIST"}
        return self._exchange("/api/client/operations", document, timeout=timeout)

    def activity(
        self,
        *,
        after: str | None = None,
        limit: int | None = None,
        watch: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """One bounded page of workspace activity; reading records nothing."""
        return self.request(
            {
                "operation": "ACTIVITY_LIST",
                **({"after": after} if after else {}),
                **({"limit": limit} if limit is not None else {}),
                **({"watch": list(watch)} if watch else {}),
            }
        )

    def activity_recent(self, *, limit: int | None = None) -> dict[str, Any]:
        """The newest requests, grouped by agent session and goal; reading records nothing."""
        return _with_read_commands(
            self.request(
                {"operation": "ACTIVITY_RECENT", **({"limit": limit} if limit is not None else {})}
            ),
            _entry_of(self),
        )

    def publish_event(self, document: dict[str, Any]) -> dict[str, Any]:
        """Declare one event about this client's own work at the external-client level.

        The Host answers `APPENDED`, `REUSED_EXACT` for a replayed producer
        sequence, or a typed refusal. Nothing here claims product authority.
        """
        return self.request({"operation": "EVENT_DECLARE", "event": document})

    def bind_native_session(self, project: Path, *, usage: str = "read") -> dict[str, Any]:
        """Ask this workspace's Host to write its exact configured project's binding."""
        return self._exchange(
            "/api/client/session/bind", {"project": str(project), "usage": usage}
        )[0]

    def publish_native_event(self, project: Path, event: dict[str, Any]) -> dict[str, Any]:
        """Deliver a selected native event with sequence metadata written by the Host."""
        return self._exchange(
            "/api/client/session/event", {"project": str(project), "event": event}
        )[0]

    def read_external(
        self, *, before: int | None = None, observation_id: str | None = None
    ) -> dict[str, Any]:
        """Read a native activity page or exact accepted-event readback from this Host.

        Args:
            before: The owner's exclusive older cursor, or the newest page.
            observation_id: One exact retained observation, exclusive of a page cursor.

        Returns:
            The unchanged bounded activity page or selected owner readback.

        Raises:
            LocalResearchClientError: A selector is invalid or the admitted transport fails.
        """
        if before is not None and (type(before) is not int or before <= 0):
            raise LocalResearchClientError("activity.cursor_invalid")
        if observation_id is not None:
            if (
                not isinstance(observation_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", observation_id) is None
            ):
                raise LocalResearchClientError("activity.observation_id_invalid")
            if before is not None:
                raise LocalResearchClientError("activity.query_selection_conflict")
            return self._exchange("/api/activity/external?observation_id=" + observation_id)[0]
        suffix = "" if before is None else f"?before={before}"
        return self._exchange("/api/activity/external" + suffix)[0]

    def cpu_budget(
        self, value: str | None = None, *, tasks_waiting: str | None = None
    ) -> dict[str, Any]:
        """Read or set the workspace's CPU budget, or how many Tasks may wait.

        A value of ``auto`` or a core count changes preparation speed, never
        research meaning; so does ``tasks_waiting``, ``auto`` or a number of
        Tasks. The answer includes machine load, the active budget and the queue.
        """
        if tasks_waiting is not None:
            return self.request({"operation": "CPU_BUDGET_SET", "tasks_waiting": tasks_waiting})
        if value is not None:
            return self.request({"operation": "CPU_BUDGET_SET", "cpu_budget": value})
        return self.request({"operation": "CPU_BUDGET_SHOW"})

    def _exchange(
        self,
        path: str,
        document: dict[str, Any] | None = None,
        *,
        expect_html: bool = False,
        timeout: float | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        import http.client  # with its email headers and ssl: only once a request is sent

        if document is not None:
            # In place: a page link and a wait built from the request read the whole values.
            whole_references(document, self.workspace)
        connection = self.connection
        data = None if document is None else request_body(document)
        headers = {
            "Content-Type": "application/json",
            CLIENT_HEADER: connection.token,
            WORKSPACE_HEADER: workspace_connection_key(self.workspace),
            INSTANCE_HEADER: connection.instance,
            # The agent session this command runs in and the goal it names (OP13).
            **agent_provenance_headers(os.environ, self.goal),
        }
        # The recorded loopback port itself: no configured proxy is consulted, and a redirect
        # is an answer, never followed.
        port = urlsplit(connection.url).port
        wait = self.timeout if timeout is None else max(0.1, min(self.timeout, timeout))
        exchange = http.client.HTTPConnection("127.0.0.1", port, timeout=wait)
        try:
            exchange.request("GET" if data is None else "POST", path, body=data, headers=headers)
            response = exchange.getresponse()
            status_code = response.status
            raw = response.read()
        except ConnectionRefusedError as error:
            # Nothing listens on the recorded port: the request never reached a Host.
            raise LocalResearchClientError("local_client.service_not_running") from error
        except (TimeoutError, OSError, http.client.HTTPException) as error:
            raise LocalResearchClientError(
                "local_client.connection_lost_task_may_still_run"
            ) from error
        finally:
            exchange.close()
        if expect_html and status_code < 300:
            if response.headers.get_content_type() != "text/html":
                raise LocalResearchClientError("local_client.response_not_html")
            try:
                return {"html": raw.decode("utf-8")}, raw
            except UnicodeDecodeError as error:
                raise LocalResearchClientError("local_client.response_not_html") from error
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise LocalResearchClientError("local_client.response_not_json") from error
        if not isinstance(body, dict):
            raise LocalResearchClientError("local_client.response_not_object")
        if status_code >= 300 and not body.get("refused") and body.get("status") != "REFUSED":
            raise LocalResearchClientError(f"local_client.http_refused:{status_code}")
        return body, raw

    def report_html(self, result_hash: str) -> tuple[dict[str, Any], bytes]:
        """Read the owner's sealed page, using the same confined local transport."""
        named = {"result_hash": result_hash}
        whole_references(named, self.workspace)
        result_hash = named["result_hash"]
        body, raw = self._exchange(
            "/api/client/report?" + urlencode({"result_hash": result_hash}), expect_html=True
        )
        return {**body, "result_hash": result_hash}, raw

    def selected_url(self, document: dict[str, Any], body: dict[str, Any]) -> str:
        """Only existing public selectors, not an artifact-path or hash guessing rule."""
        query: dict[str, object] = {}
        operation = str(document.get("operation", ""))
        if operation in {"FEATURE_CATALOG_PLAN", "FEATURE_CATALOG_READBACK"}:
            feature_plan = body.get("plan_hash") or document.get("feature_plan_hash")
            if feature_plan:
                return self.connection.url + "?" + urlencode({"feature_plan": feature_plan})
        if operation == "EXPERIMENT_ALPHA_COMPARE":
            next_requests = body.get("next_requests")
            reopen = next_requests.get("reopen", {}) if isinstance(next_requests, dict) else {}
            if not isinstance(reopen, dict):
                reopen = {}
            selection = {
                field: reopen.get(field, document.get(field))
                for field in (
                    "left_task_id",
                    "left_candidate_id",
                    "right_task_id",
                    "right_candidate_id",
                )
            }
            if not all(isinstance(value, str) and value for value in selection.values()):
                raise LocalResearchClientError("local_client.alpha_comparison_reopen_unavailable")
            return (
                self.connection.url.rstrip("/")
                + "/#"
                + urlencode(
                    {
                        "page": "alpha",
                        "study": selection["left_task_id"],
                        "alpha_left_task": selection["left_task_id"],
                        "alpha_left_candidate": selection["left_candidate_id"],
                        "alpha_right_task": selection["right_task_id"],
                        "alpha_right_candidate": selection["right_candidate_id"],
                    }
                )
            )
        if operation.startswith("GOAL_"):
            goal_hash = body.get("goal_hash") or document.get("goal_hash")
            return (
                self.connection.url.rstrip("/")
                + "/#"
                + urlencode({"page": "goal", "goal": goal_hash} if goal_hash else {"page": "goals"})
            )
        activation = body.get("activation")
        if isinstance(activation, dict):
            offered = activation.get("next_requests")
            activate = offered.get("activate") if isinstance(offered, dict) else None
            held = activation.get("held")
            book_task = (
                activation.get("book_task_id")
                or (activate.get("task_id") if isinstance(activate, dict) else None)
                or (held.get("task_id") if isinstance(held, dict) else None)
            )
            if book_task:
                return (
                    self.connection.url.rstrip("/")
                    + "/#"
                    + urlencode({"page": "portfolio", "book": book_task})
                )
        review = body.get("review_publication_hash") or document.get("review_publication_hash")
        if review:
            query = {"history": f"review:{review}"}
        elif operation.startswith(("CRO_", "EVIDENCE_")) and document.get("experiment_task_id"):
            query = {"review_experiment": document["experiment_task_id"]}
            query.update(
                {
                    key: document[field]
                    for key, field in (("r", "experiment_receipt_hash"), ("d", "portfolio_session"))
                    if document.get(field)
                }
            )
        elif operation.startswith(("CRO_", "EVIDENCE_")) and document.get("update_task_id"):
            query = {"review_update": document["update_task_id"]}
            query.update(
                {
                    key: document[field]
                    for key, field in (("p", "update_publication_hash"), ("b", "position_basis"))
                    if document.get(field)
                }
            )
        elif operation.startswith(("CRO_", "EVIDENCE_")) and document.get("result_hash"):
            query = {"review_result": document["result_hash"]}
        elif operation.startswith(("CRO_", "EVIDENCE_")) and body.get("task_id"):
            query = {"task": body["task_id"]}
        elif document.get("experiment_task_id"):
            query = {"history": f"experiment:{document['experiment_task_id']}"}
        elif operation in {"EXPERIMENT_PLAN", "EXPERIMENT_PREVIEW_READBACK"} and (
            body.get("plan_hash") or document.get("experiment_plan_hash")
        ):
            query = {"plan": body.get("plan_hash") or document["experiment_plan_hash"]}
        elif operation.startswith("EXPERIMENT"):
            task = body.get("publication_task_id") or body.get("task_id") or document.get("task_id")
            if task:
                query = (
                    {"task": task}
                    if body.get("lifecycle") not in {None, "SUCCEEDED"}
                    and not body.get("publication_task_id")
                    else {"history": f"experiment:{task}"}
                )
        elif body.get("result_hash") or document.get("result_hash"):
            query = {"history": f"result:{body.get('result_hash') or document['result_hash']}"}
        elif document.get("history_entry_id"):
            query = {"history": document["history_entry_id"]}
        elif body.get("publication_task_id") or body.get("task_id") or document.get("task_id"):
            query = {
                "task": body.get("publication_task_id")
                or body.get("task_id")
                or document["task_id"]
            }
        elif operation.startswith(("DATA_", "WORKSPACE_", "RESEARCH_INPUT", "STORAGE_")):
            query = {"panel": "workspace"}
        if str(query.get("history", "")).startswith(("experiment:", "result:")) and document.get(
            "portfolio_session"
        ):
            query["portfolio_session"] = document["portfolio_session"]
        # The workbench is the one entry (2026-09-19); every selector here is read at its load.
        return self.connection.url + ("?" + urlencode(query) if query else "")

    def navigation(self, document: dict[str, Any], body: dict[str, Any]) -> dict[str, str]:
        """Describe where a person can open this answer in Local Web.

        Args:
            document: The operation sent to the Host.
            body: Its owner answer.

        Returns:
            A labelled URL and navigation kind, never product authority.
        """
        url = self.selected_url(document, body)
        query = parse_qs(urlsplit(url).query)
        page = parse_qs(urlsplit(url).fragment).get("page", [""])[0]
        entry = query.get("history", [""])[0]
        if document.get("operation") == "EXPERIMENT_ALPHA_COMPARE":
            kind, label = "alpha_comparison", "Open this exact saved Alpha comparison"
        elif document.get("operation", "").startswith("GOAL_"):
            kind, label = "goal", "Open Goals" if page == "goals" else "Open this goal revision"
        elif page == "portfolio":
            kind, label = "portfolio_book", "Open this book on Portfolio"
        elif entry.startswith("review:"):
            kind, label = "published_cro_review", "Open this published CRO review"
        elif any(key.startswith("review_") for key in query):
            kind, label = "book_review_entry", "Open Evidence & CRO for this book"
        elif entry.startswith("result:"):
            kind, label = "portfolio_result", "Open this Portfolio result"
        elif entry.startswith("experiment:"):
            kind, label = "experiment_result", "Open this experiment result"
        elif "feature_plan" in query:
            kind, label = "feature_definition", "Open this exact local formula definition"
        elif "task" in query:
            kind, label = "task", "Follow this Task"
        elif "plan" in query:
            kind, label = "shared_plan", "Inspect this exact PLAN in Local Web"
        else:
            kind, label = "workspace", "Open the research workspace"
        return {"kind": kind, "label": label, "url": url, "claim": "NAVIGATION_NOT_AUTHORITY"}


_STDIN_BYTES = 4 * 1024 * 1024 + 1
"""The most of standard input a command reads: one byte past the largest document it takes."""
_STDIN_READ: list[tuple[object, bytes]] = []
"""Standard input as read, by its stream: a command that reads its `-` document twice (to
choose its operation, then to continue from it) reads the same bytes (V401)."""


def _stdin() -> bytes:
    stream = sys.stdin
    for seen, payload in _STDIN_READ:
        if seen is stream:
            return payload
    payload = stream.buffer.read(_STDIN_BYTES)
    _STDIN_READ.append((stream, payload))
    return payload


def _text(path: Path, *, limit: int = MAXIMUM_REQUEST_BODY_BYTES) -> str:
    try:
        if path == Path("-"):
            payload = _stdin()[: limit + 1]
        else:
            with path.open("rb") as source:
                payload = source.read(limit + 1)
        if len(payload) > limit:
            failure = LocalResearchClientError("local_client.document_too_large")
            # Its size and the bound, so the next document can fit (V372).
            failure.document_size = {
                "bytes": None if path == Path("-") else path.stat().st_size,
                "limit_bytes": limit,
            }
            raise failure
        # Preserve the former read_text() universal-newline behavior on Windows.
        return payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except LocalResearchClientError:
        raise
    except FileNotFoundError as error:
        # A path with no file is not a document of the wrong shape: name the path (V390).
        failure = LocalResearchClientError("local_client.document_missing")
        failure.document_location = {"file": str(path), "expected": "an existing file"}
        raise failure from error
    except (ValueError, OSError) as error:
        failure = LocalResearchClientError("local_client.document_unreadable")
        failure.document_location = {"file": str(path), "expected": "a readable UTF-8 file"}
        raise failure from error


def _yaml_value(path: Path, expected: str, *, text: str | None = None) -> Any:
    """A file read by the one declaration loader, its syntax errors located."""

    import yaml  # type: ignore[import-untyped]

    from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document

    try:
        return load_safe_yaml_document(_text(path) if text is None else text)
    except LocalResearchClientError:
        raise
    except yaml.YAMLError as error:
        failure = LocalResearchClientError("local_client.document_unreadable")
        mark = getattr(error, "problem_mark", None)
        if mark is not None:
            failure.document_location = {
                "line": mark.line + 1,
                "column": mark.column + 1,
                "expected": expected,
            }
        raise failure from error
    except (ValueError, OSError) as error:
        raise LocalResearchClientError("local_client.document_unreadable") from error


def _document(path: Path) -> dict[str, Any]:
    # Read once: standard input cannot be read a second time.
    text = _text(path)
    if not text.strip():
        # Nothing was written into it (V222): an empty file is not a document of another shape.
        failure = LocalResearchClientError("local_client.document_empty")
        failure.document_location = {"file": path.name, "expected": "a safe YAML mapping"}
        raise failure
    document = _yaml_value(path, "a safe YAML mapping", text=text)
    if not isinstance(document, dict):
        raise LocalResearchClientError("local_client.document_not_mapping")
    return document


def _listed(path: Path) -> list[Any]:
    """A list field's file, a YAML or JSON list (V132)."""

    document = _yaml_value(path, "a safe YAML list")
    if not isinstance(document, list):
        raise LocalResearchClientError("local_client.document_not_list")
    return document


def _response_document(path: Path) -> dict[str, Any]:
    """A saved answer, in JSON or in the YAML `--output --format yaml` writes (V130).

    A compact display is read too; what continues from it checks that the parts it reads are
    whole (`_whole`), since a display cut elsewhere still holds them (V133).
    """

    import yaml

    from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document

    text = _text(path, limit=4 * 1024 * 1024)
    try:
        try:
            result = json.loads(text)
        except ValueError:
            result = load_safe_yaml_document(text)
        value = result.get("data", result)
        if not isinstance(value, dict):
            raise ValueError("not an operation response")
    except (ValueError, AttributeError, yaml.YAMLError) as error:
        raise LocalResearchClientError("local_client.response_reference_invalid") from error
    return value


def _cut(value: Any) -> bool:
    """Whether a compact display left out part of ``value`` (its ``_omitted`` markers)."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "_omitted" in item or "_omitted_keys" in item:
                return True
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def _whole(value: Any, part: str) -> Any:
    """A part a continuation reads, refused when a compact display cut it (V133)."""
    if _cut(value):
        raise LocalResearchClientError(
            "local_client.compact_reference_requires_full_response:" + part
        )
    return value


BOOK_SELECTORS: Final = frozenset(
    {
        "result_hash",
        "handoff_hash",
        "update_task_id",
        "update_publication_hash",
        "experiment_task_id",
        "experiment_receipt_hash",
    }
)
"""The fields that name a book for its Evidence and review (V473)."""

_BOOK_CONTEXT: Final = frozenset({"position_basis", "portfolio_session"})
"""What travels with a book's selector: its position basis and its session."""


def _offered_book(offered: dict[str, dict[str, Any]], allowed: frozenset[str]) -> dict[str, Any]:
    """The one book the answer's own requests select, as the operation takes it (V473).

    A book's answer offers its Evidence and review bound to the book; an operation it does not
    offer continues that book, never the workspace's default. Two books are refused by name.
    """
    books = {
        tuple(sorted((k, v) for k, v in request.items() if k in BOOK_SELECTORS | _BOOK_CONTEXT))
        for request in offered.values()
        if isinstance(request, dict) and BOOK_SELECTORS & set(request)
    }
    if len(books) > 1:
        raise LocalResearchClientError("local_client.answer_names_two_books")
    return {k: v for k, v in next(iter(books), ()) if k in allowed}


LOCATORS: Final[dict[str, tuple[str, ...]]] = {
    "task_id": ("task_id", "publication_task_id", "prepared_task_id"),
    "goal_id": ("goal_id",),
    "experiment_plan_hash": ("experiment_plan_hash", "plan_hash"),
    "feature_plan_hash": ("feature_plan_hash", "plan_hash"),
    "feature_factor_id": ("feature_factor_id", "factor_id"),
    "evidence_unit_id": ("evidence_unit_id", "prepared_unit_id"),
}
"""Where an answer names the instance a request field selects when not under the field's own
name: a published or prepared Task, a plan, a factor, a prepared unit (V449)."""


def _located(answer: dict[str, Any], *keys: str) -> str:
    """Read returned locators, not declarations of authority; the Host revalidates.

    The answer's own first, then the read's request its CLI copy keeps (`named_read`, V454).
    """
    kept = answer.get("read_request")
    for source in (answer, kept if isinstance(kept, dict) else {}):
        for key in keys:
            if isinstance(source.get(key), str) and source[key]:
                return str(source[key])
    raise LocalResearchClientError("local_client.response_reference_missing:" + ",".join(keys))


def _write_bundle(
    directory: Path, body: dict[str, Any], *, prefix: tuple[str, ...]
) -> dict[str, Any]:
    """Write the Host's bundle into a new directory.

    Show files by name and size, never their text, beside the answer file for
    the specialist and the one submit command for the lead.
    """
    files = body.get("files")
    if not isinstance(files, list) or not files:
        raise LocalResearchClientError("local_client.bundle_response_invalid")
    listing: list[dict[str, object]] = []
    try:
        directory.mkdir(parents=True, exist_ok=False)
        for item in files:
            name, text = str(item.get("name", "")), item.get("text")
            if not name or Path(name).name != name or not isinstance(text, str):
                raise LocalResearchClientError("local_client.bundle_response_invalid")
            with (directory / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
            listing.append(
                {"name": name, "lines": text.count("\n"), "bytes": len(text.encode("utf-8"))}
            )
    except OSError as error:
        raise LocalResearchClientError("local_client.bundle_write_refused") from error
    answer = directory / "answer.json"
    # The one command as the specialist runs it, from any directory: the checkout's entry with
    # the context this call kept, its goal included (`_entry_of`, V406, V449), quoted for the
    # shell in use, as every printed command is (V439).
    width = prefix.index("--workspace") if "--workspace" in prefix else len(prefix)
    arguments = [
        *prefix[width:],
        "bundle",
        "submit",
        "--dir",
        str(directory),
        "--file",
        str(answer),
    ]
    command = join([*prefix[:width], *arguments], shell())
    return {
        **{key: value for key, value in body.items() if key != "files"},
        "files": listing,
        "answer_file": str(answer),
        "submit_arguments": arguments,
        **({} if command is None else {"submit_command": command}),
    }


def _bundle_directory(document: dict[str, Any]) -> Path | None:
    """Resolve the bundle directory named by a supported operation.

    A bundle is prepared only into a directory that does not yet exist.
    """
    operation, value = document.get("operation"), document.get("bundle_directory")
    if operation not in {"AGENT_BUNDLE_PREPARE", "AGENT_ANSWER_SUBMIT"} or not isinstance(
        value, str
    ):
        return None
    directory = Path(value).resolve()
    if operation == "AGENT_BUNDLE_PREPARE" and directory.exists():
        raise LocalResearchClientError("local_client.bundle_directory_exists")
    document["bundle_directory"] = str(directory)
    return directory


def _request_fields(
    template: dict[str, Any], supplied: dict[str, Any], path: str = ""
) -> dict[str, Any]:
    """Fill declared choices without silently replacing an owner's bound reference.

    A field the owner wrote as None is a choice left to the reader (V136), filled like an
    absent one.
    """
    result = dict(template)
    for key, value in supplied.items():
        field = f"{path}.{key}" if path else str(key)
        if key in result and result[key] is not None:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _request_fields(result[key], value, field)
            elif _one_reference(str(key), result[key], value):
                result[key] = max(result[key], value, key=len)  # the whole spelling
            elif result[key] != value:
                raise LocalResearchClientError("local_client.bound_reference_override:" + field)
        else:
            result[key] = value
    return result


def _one_reference(key: str, bound: object, given: object) -> bool:
    """Whether two spellings name one reference: a whole value and the beginning the compact
    view shows of it, in a field the contract types as a hash or an id (V399)."""
    if key not in command_table()["references"] or not (
        isinstance(bound, str) and isinstance(given, str)
    ):
        return False
    return _shown_as(bound, given)


def _shown_as(left: str, right: str) -> bool:
    """Whether one value is the other's whole and the other its shown beginning."""
    short, whole = sorted((left, right), key=len)
    return (
        short != whole
        and _SHORT_REFERENCE.fullmatch(short) is not None
        and WHOLE_REFERENCE.fullmatch(whole) is not None
        and whole.startswith(short)
    )


def _document_value(field: str, given: str) -> object:
    """The document `--file` names: its text for a text field (a declaration kept as written),
    the YAML or JSON it holds otherwise; `-` reads stdin, and a value that is JSON itself, as a
    next command prints it, is the document."""
    if given.lstrip().startswith(("{", "[")):
        try:
            return json.loads(given)
        except ValueError as error:
            raise LocalResearchClientError("local_client.document_unreadable") from error
    path = Path(given)
    if set(command_table()["types"].get(field, ())) == {"string"}:
        return _text(path)
    return _document(path)


def document_field(documents: list[str], given: str) -> tuple[str, object]:
    """Read the declaration into the selected operation's registered document field.

    A path fills the first field; an inline JSON value fills the first whose kind fits it,
    so a printed `--file '{...}'` sends the request it came from (V432).

    Args:
        documents: The selected operation's document fields, in registry order.
        given: A file path, stdin marker or inline JSON value.

    Returns:
        The request field and the document it reads.
    """
    value = _document_value(documents[0], given)
    if isinstance(value, dict | list):
        kind = "object" if isinstance(value, dict) else "array"
        return next(
            (name for name in documents if kind in set(command_table()["types"].get(name, ()))),
            documents[0],
        ), value
    return documents[0], value


def _next_request(args: argparse.Namespace) -> dict[str, Any]:
    primary_file = getattr(args, "primary_file", None)
    if (
        sum(
            path is not None and str(path) == "-"
            for path in (args.from_response, args.file, primary_file)
        )
        > 1
    ):
        raise LocalResearchClientError("local_client.document_multiple_stdin_sources")
    if args.from_response is None:
        if args.action is not None or args.file is None:
            raise LocalResearchClientError(
                "local_client.request_file_or_selected_response_required"
            )
        return _document(args.file)
    bundle = _response_document(args.from_response)
    offered = offered_requests(bundle)
    action = args.action
    if action not in offered and action is not None:
        matches = [name for name in offered if name.startswith(action + ":")]
        if not matches and ":" in action:
            # `recovery:<task>` names one item whether its id is whole or shown short (V399).
            head, _, item = action.partition(":")
            matches = [
                name
                for name in offered
                if name.partition(":")[0] == head and _shown_as(name.partition(":")[2], item)
            ]
        if len(matches) == 1:
            action = matches[0]
    if action not in offered:
        if not offered and isinstance(bundle.get("operation"), str):
            raise LocalResearchClientError("local_client.operation_document_has_no_next_request")
        if _cut(bundle):  # the request may be in a part the compact display left out
            _whole(bundle, "next_requests")
        raise LocalResearchClientError(
            "local_client.next_request_required:" + ",".join(sorted(offered))
        )
    template = _whole(offered[action], str(action))
    if not isinstance(template.get("operation"), str):
        raise LocalResearchClientError("local_client.next_request_invalid")
    fields = _document(args.file) if args.file else {}
    if primary_file is not None:
        table = command_table()
        allowed = table["fields"][template["operation"]]["allowed"]
        documents = [name for name in table["primary"] if name in allowed]
        if not documents:
            raise LocalResearchClientError(
                "local_client.operation_has_no_declaration_file:" + template["operation"]
            )
        field, value = document_field(documents, str(primary_file))
        if field in fields:
            raise LocalResearchClientError("local_client.field_given_twice:" + field)
        fields[field] = value
    return _chosen(_request_fields(template, fields))


def _chosen(request: dict[str, Any]) -> dict[str, Any]:
    """The request, once every choice its owner left is given (V136)."""
    left = choices(request)
    if left:
        raise LocalResearchClientError("local_client.next_request_needs_choice:" + ",".join(left))
    return request


_DRAFT_STATUSES = frozenset(
    {"DRAFT_READY", "DRAFT_INCOMPLETE", "INPUT_COMPILER_PREFLIGHT_PASSED", "PORTFOLIO_DRAFT_READY"}
)
_DECLARATIONS = frozenset({"experiment_yaml", "experiment_document"})


def _controls(bundle: dict[str, Any]) -> bool:
    """Whether an answer is `study controls`': its declaration template and its input."""
    return (
        bundle.get("status") == "READY"
        and isinstance(bundle.get("template"), dict)
        and isinstance(bundle.get("input_binding_hash"), str)
    )


def continuation(
    operation: str, answer: Path, given: dict[str, Any], allowed: frozenset[str]
) -> dict[str, Any]:
    """Build an operation request from a saved answer and explicit fields (`continued`).

    Args:
        operation: The operation the command sends.
        answer: The saved answer `--from` names.
        given: The fields the command's flags give.
        allowed: The fields the operation takes.

    Returns:
        The request.
    """
    return continued(operation, _response_document(answer), given, allowed)


def continued(
    operation: str, bundle: dict[str, Any], given: dict[str, Any], allowed: frozenset[str]
) -> dict[str, Any]:
    """Build an operation request from an answer and explicit fields.

    A draft's ``plan_request`` starts a plan with the references it binds kept (a field that
    would replace one is refused); a declaration given replaces the draft's, and without one
    the draft's own document is planned. A controls
    answer starts one on its input, its template planned when no declaration is given (V355).
    A next request for the operation starts it with its bound references kept (a field that
    would replace one is refused, as ``request --from`` refuses it); a preview's ``replan`` is
    one, and a declaration given replaces the one it offers, as it replaces a draft's (V148).
    Otherwise the answer's locators fill the fields the operation takes. Its Task and its goal
    are the request's target: one the answer does not name is refused, and a flag naming another
    is refused, never preferred. Every other reference the operation takes (a hash or an id) is
    filled from the answer under its own name or its `LOCATORS` alias, so a read reads again
    from its own answer; a flag giving one wins over the answer's (V449).

    Args:
        operation: The operation the command sends.
        bundle: The answer it continues.
        given: The fields the command's flags give.
        allowed: The fields the operation takes.

    Returns:
        The request.

    Raises:
        LocalResearchClientError: A target the answer does not name, a flag naming another,
            a choice left open, or a part a compact view cut.
    """
    offered = offered_requests(bundle)
    kept = bundle.get("read_request")
    # A saved read reads again as itself, its whole selection, before any next request the
    # answer offers for the same operation (V460: a revision read again as the latest);
    # `--action` chooses a next request by name.
    again = isinstance(kept, dict) and kept.get("operation") == operation
    matches = [
        request
        for request in offered.values()
        if request.get("operation") == operation and not again
    ]
    if operation == "EXPERIMENT_PLAN":
        template = bundle.get("plan_request")
        status = bundle.get("status")
        draft_status = isinstance(status, str) and status in _DRAFT_STATUSES
        drafted = (
            draft_status and isinstance(template, dict) and template.get("operation") == operation
        )
        if not drafted and not matches and _controls(bundle):
            # A controls answer plans its declaration on its own input (V355).
            # Its input and binding both: the input the controls named is not asked again.
            controlled = _request_fields(
                {
                    "operation": operation,
                    "input_binding_hash": bundle["input_binding_hash"],
                    **(
                        {"research_input_id": bundle["research_input_id"]}
                        if isinstance(bundle.get("research_input_id"), str)
                        else {}
                    ),
                },
                given,
            )
            if not _DECLARATIONS & set(controlled):
                controlled["experiment_document"] = _whole(bundle["template"], "template")
            return controlled
        if not drafted and not matches:
            if _cut(bundle):  # a draft's plan may be in a part the display left out
                _whole(bundle, "plan_request")
            if not draft_status:
                raise LocalResearchClientError(
                    "local_client.plan_source_unsupported:" + str(bundle.get("status", ""))
                )
            raise LocalResearchClientError(
                "local_client.continuation_bundle_invalid_export_fresh_draft"
            )
        if drafted:
            assert isinstance(template, dict)
            bound = _whole(template, "plan_request")
            if _DECLARATIONS & set(given):
                bound = {k: v for k, v in bound.items() if k not in _DECLARATIONS}
            plan = _request_fields(bound, given)
            if not _DECLARATIONS & set(plan):
                declaration = bundle.get("document", bundle.get("template"))
                if not isinstance(declaration, dict):
                    raise LocalResearchClientError(
                        "local_client.plan_file_or_draft_document_required"
                    )
                plan["experiment_document"] = _whole(declaration, "document")
            return plan
        if _DECLARATIONS & set(given):
            matches = [{k: v for k, v in m.items() if k not in _DECLARATIONS} for m in matches]
    if len(matches) > 1:
        raise LocalResearchClientError(
            "local_client.next_request_required:"
            + ",".join(sorted(n for n, r in offered.items() if r.get("operation") == operation))
        )
    if matches:
        return _chosen(_request_fields(_whole(matches[0], operation), given))
    document: dict[str, Any] = {"operation": operation}
    if "task_id" in allowed:
        # A saved answer's Task read follows the Task the answer started, as `--wait` does
        # (V137, V446); any other operation reads the answer's own Task.
        started = ("follow_task_id",) if operation == "STATUS" else ()
        try:
            document["task_id"] = _located(bundle, *started, *LOCATORS["task_id"])
        except LocalResearchClientError:
            # A receipt that started no Task names its subject's in the requests it offers:
            # the one Task they all name (V479); none or two leave the refusal as it was.
            named = {
                request["task_id"]
                for request in offered.values()
                if isinstance(request, dict) and isinstance(request.get("task_id"), str)
            }
            # A read again as itself names no Task when its saved request named none and its
            # operation needs none: a readback saved before its first Task reads again (V575).
            taskless = again and "task_id" not in command_table()["fields"][operation]["required"]
            if len(named) == 1:
                document["task_id"] = named.pop()
            elif named or not taskless:
                raise
    if "goal_id" in allowed and operation != "GOAL_OPEN":
        # The goal the file names, never the session's own by default: a file naming none is
        # refused (V449; an opened goal's id is its own, never a file's).
        document["goal_id"] = _located(bundle, *LOCATORS["goal_id"])
    # The target never yields to a flag naming another (V449, as `_request_fields` refuses it
    # for an offered request); every other reference the answer names fills what the flags
    # leave, so a read reads again from its own answer (V449).
    request = _request_fields(document, given)
    if again:
        assert isinstance(kept, dict)
        # The same read again: its whole selection, as the CLI's copy kept it, fills what the
        # flags leave, a day or a page included (V454); a write's text is the caller's to give
        # again, never a saved answer's (OP15).
        for name, value in kept.items():
            if name in allowed and name not in {"operation", "task_id", "goal_id"}:
                request.setdefault(name, value)
    if allowed & BOOK_SELECTORS and not BOOK_SELECTORS & set(request):
        # The book this answer's own requests select, bound as they bind it (V473).
        for name, value in _offered_book(offered, allowed).items():
            request.setdefault(name, value)
    references = {name for name in allowed if reference_field(name)} | set(LOCATORS)
    for name in sorted((allowed & references) - {"task_id", "goal_id"}):
        keys = LOCATORS.get(name, (name,))
        found = [
            source[key]
            for source in (bundle, kept if isinstance(kept, dict) else {})
            for key in keys
            if isinstance(source.get(key), str) and source[key]
        ]
        if found:
            request.setdefault(name, found[0])
    required = set(command_table()["fields"][operation]["required"]) & references
    missing = sorted(name for name in required if not request.get(name))
    if missing and not again:
        raise LocalResearchClientError(
            "local_client.response_reference_missing:" + ",".join(missing)
        )
    # A new read from a saved answer must select that answer's subject. An unrelated
    # answer cannot silently reopen the workspace's default input/package (V515).
    # A saved read of an unfiltered list keeps that explicit selection in read_request.
    selectors = (allowed & references) - ({"goal_id"} if operation == "GOAL_OPEN" else set())
    if selectors and not again and not any(request.get(name) for name in selectors):
        raise LocalResearchClientError(
            "local_client.response_reference_missing:" + ",".join(sorted(selectors))
        )
    return request


def _save_output(
    path: Path, body: dict[str, Any], raw: bytes, format: str, *, replace: bool = False
) -> None:
    """The whole answer in the format asked, which `--from` reads back; html is its report.

    YAML is written in the declaration loader's dialect, so it reads back the same values
    (V146); the editable declaration alone is `--save-declaration`'s (V130). ``replace`` is for the
    admission this command saved before its wait, which the final answer takes over (V273).
    """
    if format == "json":
        payload = body["json"].encode("utf-8") if isinstance(body.get("json"), str) else raw
    elif format == "yaml":
        from alphalattice.protocols.research_authoring.selection import dump_declaration

        # The answer JSON saves: an export's inner answer where it carries one as text, so
        # `--from` reads the same Task and next requests from either file (V447).
        whole = (
            json.loads(body["json"])
            if isinstance(body.get("json"), str)
            else json.loads(json.dumps(body, default=_json_value))
        )
        payload = dump_declaration(whole).encode("utf-8")
    else:
        value = body.get(format)
        if not isinstance(value, str):
            raise LocalResearchClientError("local_client.output_format_unavailable:" + format)
        payload = value.encode("utf-8")
    _write_new(path, payload, replace=replace)


def _save_declaration(path: Path, body: dict[str, Any]) -> None:
    """The answer's editable declaration, for an author to edit (V130): its ``yaml`` part, or
    its ``template`` in the declaration loader's dialect, as `feature controls` answers (V362)."""
    value = body.get("yaml")
    if not isinstance(value, str) and isinstance(body.get("template"), dict):
        from alphalattice.protocols.research_authoring.selection import dump_declaration

        value = dump_declaration(json.loads(json.dumps(body["template"], default=_json_value)))
    if not isinstance(value, str):
        raise LocalResearchClientError("local_client.declaration_unavailable")
    _write_new(path, value.encode("utf-8"))


def _write_new(path: Path, payload: bytes, *, replace: bool = False) -> None:
    """Write a file that does not exist yet; an existing one is never replaced, but for
    ``replace``: the file this command wrote itself, swapped whole for its successor."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if replace:
            staged = path.with_name(path.name + ".tmp")
            staged.write_bytes(payload)
            os.replace(staged, path)
            return
        with path.open("xb") as stream:
            stream.write(payload)
    except OSError as error:
        raise LocalResearchClientError("local_client.output_write_refused") from error


_SHOWN_REFERENCE = re.compile(r"(?<![/\\])" + WHOLE_REFERENCE.pattern + r"(?![/\\]|\.[A-Za-z])")
"""A whole hash or id outside a path: a path keeps its values whole, so it still opens."""
_SHORT_REFERENCE = re.compile(r"[0-9a-f]{12}|[0-9a-f]{8}-[0-9a-f]{3}")
"""The first twelve characters of a hash, or of a UUID, as the compact display shows them."""
_REFERENCE_KEYS = ("_id", "_hash", "_ids", "_hashes")
"""The fields that hold a typed reference, by their names' endings: a task, a plan, a hash."""
_SHOWN_SHORT_REFERENCE = re.compile(
    r"(?<![/\\0-9a-f-])(?:" + _SHORT_REFERENCE.pattern + r")(?![/\\0-9a-f-]|\.[A-Za-z])"
)


def reference_field(name: str | None) -> bool:
    """Whether a field names a subject a continuation carries from a saved answer (V523, V515).

    Schema-typed references and named locators, prefixed history entries and string ids, an
    authored selector such as a research input among them. Which of them the compact display
    shortens and the client reads back is `_issued_reference`'s (V561).
    """
    return name is not None and (
        name.endswith(_REFERENCE_KEYS) or name in command_table()["references"]
    )


def _issued_reference(name: str) -> bool:
    """Whether a request field holds only values the Host issued: a hash, or a reference the
    request contract types (a UUID, a hash, an issued entry id). Only there is a short value
    read back as the whole one it begins, and refused when it begins none or several. An
    authored id, a dated input or a criterion named by the person, is sent as written whatever
    the ledger holds (V393, V544, V561)."""
    return name.endswith(("_hash", "_hashes")) or name in command_table()["references"]


def _shortened(name: str | None) -> bool:
    """Whether the compact display shortens a field's whole hashes and ids: an issued reference,
    or an answer's own id the request contract does not take, which a reader sends back through
    the typed request field it selects. A request field the contract leaves untyped holds an
    authored value, shown and sent as written (V561)."""
    if name is None or not reference_field(name):
        return False
    return _issued_reference(name) or name not in command_table()["types"]


def short_references(value: Any, key: str | None = None) -> Any:
    """The compact display with each typed reference shown by its first twelve characters (V393).

    The client reads such a beginning back as the whole value (`whole_references`), so what is
    shown may be sent. Only a field that holds a reference is shortened, its list's items
    included: a path, a link, a command the Host wrote and every other text stay whole, since a
    directory named after a Task (`cro-<id>-r1`) must still open and its command still run
    (V478). `--view full` and `--output` keep every value.

    Args:
        value: The display, or a part of it.
        key: The part's key, when it is a mapping's value or a list's under one.

    Returns:
        The display with the references' whole hashes and UUIDs shortened.
    """
    if isinstance(value, str):
        if not _shortened(key):
            return value
        return _SHOWN_REFERENCE.sub(lambda match: match.group(0)[:12], value)
    if isinstance(value, dict):
        return {k: short_references(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [short_references(item, key) for item in value]
    return value


def whole_references(document: dict[str, Any], workspace: Path) -> None:
    """Read each short reference in a request as its whole value, in place, before it is sent.

    A string of exactly the shape the compact display gives a hash or an id, standing in a
    field the request contract types as one (the table's `references`, at any depth, a list's
    items included), is read as the one value the Host has answered with that begins so
    (V393). Such a field holds only issued values, so one that begins several values or none
    is refused. A field the contract does not type as a reference, an authored id among them,
    is sent byte for byte whatever the ledger holds (V544, V561), as is every other string,
    authored text included (V399).

    Args:
        document: The request; its short references are replaced.
        workspace: The served workspace, whose runtime holds the Host's ledger.

    Raises:
        LocalResearchClientError: `local_client.short_reference_ambiguous` or
            `local_client.short_reference_unknown`.
    """
    ledger: list[str] = []  # read once, at the first short reference

    def wholes(short: str) -> list[str]:
        if not ledger:
            try:
                ledger.append(
                    (workspace / "runtime" / REFERENCE_LEDGER_NAME).read_text(encoding="utf-8")
                )
            except OSError:
                ledger.append("")
        return sorted(set(re.findall(rf"^{short}[0-9a-f-]+$", ledger[0], flags=re.MULTILINE)))

    def whole(value: Any, path: str, typed: bool) -> Any:
        if isinstance(value, str):
            if not typed:
                return value

            def restore(match: re.Match[str]) -> str:
                short = match.group(0)
                candidates = wholes(short)
                if len(candidates) == 1:
                    return candidates[0]
                failure = LocalResearchClientError(
                    "local_client.short_reference_" + ("ambiguous" if candidates else "unknown")
                )
                failure.short_reference = {
                    "field": path,
                    "value": short,
                    "candidates": candidates[:5],
                }
                raise failure

            return _SHOWN_SHORT_REFERENCE.sub(restore, value)
        if isinstance(value, dict):
            return {
                name: whole(
                    item, f"{path}.{name}" if path else str(name), _issued_reference(str(name))
                )
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [whole(item, f"{path}.{index}", typed) for index, item in enumerate(value)]
        return value

    document.update(whole(document, "", False))


def _same(value: Any) -> Any:
    return value


def _context(
    workspace: Path, goal: str | None, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The context a command worked in, which every answer names, as the agent CLIs' JSON
    output names its session (V401): the workspace and how it was chosen (`WORKSPACE_FROM`:
    named, or the session's binding, V568), the goal the Host counted the request toward (the
    one named, or the one the session took) and the agent session it came from."""
    headers = agent_provenance_headers(os.environ)
    session = headers.get(AGENT_SESSION_HEADER)
    counted = (body or {}).get("attributed_goal_id")
    return {
        "workspace": str(workspace.resolve()),
        "workspace_from": WORKSPACE_FROM.get(),
        "goal": counted if isinstance(counted, str) else goal,
        "agent_session": f"{headers[AGENT_VENDOR_HEADER]}:{session}" if session else None,
    }


COMPACT_ANSWER_BYTES = 32 * 1024
"""A compact answer is one whole read on every agent host, as a bundle file is
(`BUNDLE_FILE_BYTES`): Codex hands the model at most 10,000 tokens of one call's output, about
40,000 bytes, and cuts the middle of the rest (AX1). Its data keeps what the envelope, the next
requests and commands leave of it, never less than `COMPACT_DATA_FLOOR_BYTES` (V402)."""
COMPACT_DATA_FLOOR_BYTES = 8 * 1024
_MARK_BYTES = 192
"""What one omission prints at most, its mark and its name in `omitted_sections`: each part
waiting to be shown keeps this much of the budget, so its mark always fits."""
_REPRESENTATION = "COMPACT_DISPLAY_NOT_AN_AUTHORITY_DOCUMENT"
_FULL_RESPONSE = (
    "Each omitted part is a --section path: repeat a read with --section <part>; after a run "
    "or submit, read its Task or its --output file. --view full prints all; --output saves the "
    "answer exactly, this envelope's `data` at the file's root (read `position` there, not "
    "`data.position`)."
)
"""The envelope's note on the full answer: a saved file is the answer itself, never this
envelope (V531: RR5d's agent read its saved answers as envelopes, `data.*`, turn after turn)."""


def _size(value: Any) -> int:
    """The bytes ``value`` prints as in the CLI's JSON (ASCII escapes, default separators)."""
    return len(json.dumps(value, default=_json_value))


COMPACT_NEXT_BYTES = COMPACT_ANSWER_BYTES // 4
"""What a compact answer's next requests print at most, their requests, commands and templates
together: the owner's first, in its order, while they fit; `next_left` counts the rest, which
`--list-next` lists a page at a time (V411)."""
_NEXT_KEYS = ("next_requests", "next_commands", "next_templates")


def _hold_next(answer: dict[str, Any], budget: int) -> None:
    """Hold an envelope's next requests to ``budget`` bytes, counting the rest (V411).

    The three views of one offered request (the request, its command and its template) stay
    together, and the owner's first requests are kept, in its order, while they fit.
    """
    views = [value if isinstance(value, dict) else {} for value in map(answer.get, _NEXT_KEYS)]
    if _size(views) <= budget:
        return
    names = list(dict.fromkeys(name for view in views for name in view))
    kept: set[str] = set()
    size = 0
    for name in names:
        size += _size(name) + _size([view.get(name) for view in views])
        if size > budget and kept:
            break
        kept.add(name)
    for key, view in zip(_NEXT_KEYS, views, strict=True):
        if view:
            answer[key] = {name: value for name, value in view.items() if name in kept} or None
    answer["next_left"] = {"count": len(names) - len(kept), "list": "--list-next"}


def _listing_page(
    listing: dict[str, Any], start: int, budget: int | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """`--list-next`'s listing from its ``start``th entry, commands then templates (V411).

    Within ``budget`` bytes, at least one entry, or whole when ``budget`` is None; beside it the
    count left and the option that lists them, when some are.
    """
    entries = [
        (key, name, value)
        for key in ("next_commands", "next_templates")
        for name, value in (listing.get(key) or {}).items()
    ]
    page: dict[str, Any] = {"next_commands": None, "next_templates": None}
    # The count left and the option that lists them print beside the page, within the same
    # bytes: held back at their longest (TS1, after V478 kept commands whole).
    longest = {"count": len(entries), "list": f"--list-next --next-from {len(entries)}"}
    size = _size(page) + _size(longest)
    shown = 0
    for key, name, value in entries[start:]:
        size += _size({name: value})
        if budget is not None and size > budget and shown:
            break
        page[key] = {**(page[key] or {}), name: value}
        shown += 1
    left = max(0, len(entries) - start - shown)
    return page, (
        {"count": left, "list": f"--list-next --next-from {start + shown}"} if left else None
    )


def compact_display(
    body: Any, *, budget: int = COMPACT_ANSWER_BYTES, section: str = "", whole: bool = False
) -> dict[str, Any]:
    """Bound display work and size without inventing values or replacing a raw export.

    The display, its list of omissions included, prints within ``budget`` bytes (V402). Every
    omitted subtree is marked and named in ``omitted_sections`` as ``--section`` reads it from
    the answer's top (``section`` is the part this display shows): a key's value, or a list's
    rest as a page (``items.3:``). ``whole`` shows the answer whole when it fits: a refusal,
    whose fields say what to do next. A section's list shows as many leading items whole as
    fit, the rest its next page. Counts describe representation size, not research coverage;
    this view cannot be hashed as the original artifact.
    """
    trimmed: list[str] = []
    if isinstance(body, dict) and "timing" in body and not section:
        # A Task's stage timings: no agent's decision reads them, and they were half of a
        # `task show` answer (V280). The full view and --output keep them.
        body = {key: value for key, value in body.items() if key != "timing"}
        trimmed.append("timing")
    if whole and _size(body) + _size(trimmed) + len(_FULL_RESPONSE) + 160 <= budget:
        return {
            "representation": _REPRESENTATION,
            "data": body,
            "omitted_sections": trimmed,
            "full_response": _FULL_RESPONSE,
        }
    limit = budget
    for _ in range(4):
        shown = _compact(body, limit, section, list(trimmed))
        over = _size(shown) - budget
        if over <= 0:
            break
        # The tally is close, not exact (long keys, escapes): draw again with less.
        limit -= over + _MARK_BYTES
    return shown


def _compact(body: Any, budget: int, section: str, omissions: list[str]) -> dict[str, Any]:
    """One drawing of ``compact_display`` within ``budget`` bytes by its own tally."""
    remaining = 300
    spent = len(_REPRESENTATION) + len(_FULL_RESPONSE) + 120 + _size(omissions)
    pending: deque[tuple[Any, str, int, Any, str | int]] = deque()
    important = {
        "status",
        "lifecycle",
        "disposition",
        "evidence_verification",
        "failure_code",
        "refused",
        "next_action",
        "next_requests",
        "plan_request",
        "submission_template",
        "limitations",
        "limits",
        "claim_limit",
        "coverage",
        "required_actions",
        "review",
        "review_status",
        "recommendation",
        "summary",
        "readout",
        "result",
        "position",
        "data_quality",
        "execution_preview",
        "progress",
        "execution_intent",
        "execution_numerical_call_count",
        "declaration_changes",
    }

    def room() -> int:
        return budget - spent - len(pending) * _MARK_BYTES

    def part(path: str, key: str | int) -> str:
        return f"{path}.{key}" if path else str(key)

    def mark(path: str, omitted: dict[str, Any]) -> dict[str, Any]:
        nonlocal spent
        # A part of a text's JSON preview is read as the text it came from; the part this
        # display shows is never its own continuation (--output keeps it whole).
        name = path.split(".preview", 1)[0]
        if name and name != section and name not in omissions:
            omissions.append(name)
            spent += _size(name) + 2
        spent += _size(omitted)
        return omitted

    def render(value: Any, path: str, depth: int) -> Any:
        nonlocal remaining, spent
        remaining -= 1
        # Reserve queued siblings before expanding a deeper receipt. A left
        # comparison must not consume the right subject's display budget.
        available = max(0, remaining - len(pending))
        if isinstance(value, str) and (len(value) > 900 or _size(value) > room()):
            encoded = value.encode("utf-8")
            omitted = {
                "_omitted": "text",
                "utf8_bytes": len(encoded),
                "text_sha256": sha256(encoded).hexdigest(),
            }
            if (path == "json" or path.endswith(".json")) and depth < 5 and available:
                with suppress(ValueError):
                    pending.append(
                        (json.loads(value), path + ".preview", depth + 1, omitted, "preview")
                    )
            return mark(path, omitted)
        if not isinstance(value, dict | list):
            spent += _size(value)
            return value
        if isinstance(value, list) and depth == 0 and section:
            # A section's list: its leading items whole, as many as fit, the rest a page.
            leading: list[Any] = []
            size = 2
            for item in value:
                if spent + size + _size(item) + 2 + _MARK_BYTES > budget:
                    break
                leading.append(item)
                size += _size(item) + 2
            if leading:
                spent += size
                if len(leading) == len(value):
                    return leading
                # A page of a page is named from the list it pages: `items.122:`.
                stem, _dot, last = path.rpartition(".")
                paged = _PAGE.fullmatch(last)
                rest = (
                    f"{stem}.{int(paged.group(1) or 0) + len(leading)}:{paged.group(2)}"
                    if paged and stem
                    else part(path, f"{len(leading)}:")
                )
                tail = mark(rest, {"_omitted": "list_tail"})
                return {**tail, "item_count": len(value), "preview": leading}
        # Each part shown keeps a mark's room until it is drawn; one more for this one's own.
        fits = max(0, room() // _MARK_BYTES - 1)
        if (not (available and fits) and value) or depth > 6:
            return mark(path, {"_omitted": "collection", "item_count": len(value)})
        if isinstance(value, list):
            count = min(3 if len(value) > 8 else len(value), available, fits)
            items: list[Any] = [None] * count
            spent += 2 + 2 * count
            pending.extend(
                (v, part(path, i), depth + 1, items, i) for i, v in enumerate(value[:count])
            )
            if count < len(value):
                tail = mark(part(path, f"{count}:"), {"_omitted": "list_tail"})
                return {**tail, "item_count": len(value), "preview": items}
            return items
        keys = sorted(
            value,
            key=lambda key: (
                key not in important and not key.endswith(("_hash", "_id", "_at", "_through")),
                key,
            ),
        )
        count = min(40, available, len(keys), fits)
        shown: dict[str, Any] = {}
        spent += 2 + sum(_size(str(key)) + 4 for key in keys[:count])
        pending.extend((value[key], part(path, key), depth + 1, shown, key) for key in keys[:count])
        if len(keys) > count:
            if path == section:
                # The display's own top: each key left out is a part `--section` reads, named
                # while the names fit; a deeper mapping is read whole by its own path.
                for key in keys[count:]:
                    if room() < _MARK_BYTES:
                        break
                    mark(part(path, key), {})
            else:
                mark(path, {})
            shown["_omitted_keys"] = len(keys) - count
        return shown

    # What a result can claim and what its flow needs next, which an agent reads first, are
    # bounded by their owners (five studies of each result at most), so they show whole and
    # spend only their own size (V368, V367).
    kept = (
        {key: body[key] for key in ("standing", "review_standing", "prerequisites") if key in body}
        if isinstance(body, dict) and not section
        else {}
    )
    spent += _size(kept)
    rest = {k: v for k, v in body.items() if k not in kept} if kept else body
    displayed = render(rest, section, 0)
    while pending:
        value, path, depth, parent, key = pending.popleft()
        parent[key] = render(value, path, depth)
    return {
        "representation": _REPRESENTATION,
        "data": {**kept, **displayed} if kept else displayed,
        "omitted_sections": omissions,
        "full_response": _FULL_RESPONSE,
    }


_PAGE = re.compile(r"(\d*):(\d*)")


def _step(value: Any, parts: list[str], at: int) -> tuple[Any, int] | None:
    """What part ``at`` of a section path names in ``value``, and how many parts it took: a
    mapping's key (one holding dots, `manifest.json`, takes several), a list's index, or a page
    of a list (`3:`, `3:40`, V402); None where ``value`` holds no such part."""
    name = parts[at]
    if isinstance(value, dict):
        for end in range(at + 1, len(parts) + 1):
            if (key := ".".join(parts[at:end])) in value:
                return value[key], end - at
        return None
    if isinstance(value, list) and (page := _PAGE.fullmatch(name)):
        try:
            start, stop = (int(bound) if bound else None for bound in page.groups())
        except ValueError:
            return None
        return value[start:stop], 1
    if isinstance(value, list) and name.isdigit():
        try:
            index = int(name)
        except ValueError:
            return None
        if index < len(value):
            return value[index], 1
    return None


def answer_part(body: dict[str, Any], path: str) -> dict[str, Any]:
    """One part of an answer by its dotted path, a list's item by its index (V112).

    ``result.evidence_report.items.0`` is the first Factor's evidence and ``items.3:`` its
    items from the fourth on, a page the compact view names (V402); `schema show` outlines a
    readback's parts (``schema show``'s `answer_parts`). A part whose unit the answer names in its
    ``metric_units`` carries it (V343).
    """
    value: Any = body
    parts, at = path.split("."), 0
    while at < len(parts):
        step = _step(value, parts, at)
        if step is None:
            reached = ".".join(parts[: at + 1])
            raise LocalResearchClientError("local_client.section_unknown:" + reached)
        value, taken = step
        at += taken
    units = body.get("metric_units")
    unit = units.get(path) if isinstance(units, dict) else None
    return {"section": path, "value": value, **({"unit": unit} if unit else {})}


def saved_answer(
    path: Path, *, section: str | None = None, list_sections: bool = False
) -> dict[str, Any]:
    """Read a saved full owner answer locally, without reopening or reverifying its object.

    The CLI's saved JSON/YAML keeps owner fields at the root, unlike its printed envelope.
    The original answer is always retained for the output writer; a selected part is only a
    presentation of that snapshot. Saved requests remain data and no reference is shortened.

    Args:
        path: The full answer file, or standard input as ``-``.
        section: An existing ``answer_part`` path, index or list slice.
        list_sections: List the answer's root paths instead of selecting a part.

    Returns:
        The historical snapshot, original answer and optional selected part or root paths.

    Raises:
        LocalResearchClientError: A file failure, unsupported representation or absent part.
    """
    import yaml

    from alphalattice.protocols.research_authoring.selection import load_safe_yaml_document

    text = _text(path, limit=4 * 1024 * 1024)
    if text.lstrip().startswith("<"):
        raise LocalResearchClientError("local_client.saved_answer_format_unavailable:html")
    try:
        try:
            body = json.loads(text)
        except ValueError:
            # Full exported answers are trees. Refuse scalar aliases too, before
            # a short YAML file can expand them repeatedly during JSON serialization.
            if any(isinstance(token, yaml.tokens.AliasToken) for token in yaml.scan(text)):
                raise ValueError("not a JSON tree") from None
            body = load_safe_yaml_document(text)
        if not isinstance(body, dict):
            raise ValueError("not an owner answer")
        # The full exporter writes a JSON tree, including quoted date strings. Reject
        # aliases before serialization so a small YAML file cannot amplify its containers.
        stack: list[Any] = [body]
        seen: set[int] = set()
        while stack:
            value = stack.pop()
            if isinstance(value, (dict, list)):
                if id(value) in seen:
                    raise ValueError("not a JSON tree")
                seen.add(id(value))
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    raise ValueError("not JSON-compatible keys")
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        json.dumps(body, allow_nan=False)
    except (ValueError, TypeError, RecursionError, yaml.YAMLError) as error:
        failure = LocalResearchClientError("local_client.saved_answer_invalid")
        failure.document_location = {
            "file": str(path),
            "expected": "a full saved owner answer as a JSON-compatible mapping",
        }
        mark = getattr(error, "problem_mark", None)
        if mark is not None:
            failure.document_location.update({"line": mark.line + 1, "column": mark.column + 1})
        raise failure from error
    if _cut(body) or body.get("representation") == _REPRESENTATION or "omitted_sections" in body:
        raise LocalResearchClientError("local_client.saved_answer_incomplete")
    if {"schema_version", "operation", "outcome", "data", "timing"} <= body.keys():
        raise LocalResearchClientError("local_client.saved_answer_invalid")
    if section is not None and list_sections:
        raise LocalResearchClientError("local_client.saved_answer_invalid")
    reading: dict[str, Any] = {
        "snapshot": {
            "file": "-" if path == Path("-") else str(path.resolve()),
            "claim": "HISTORICAL_SAVED_ANSWER_NOT_REVERIFIED",
        },
        "answer": body,
    }
    if section is not None:
        try:
            reading.update(answer_part(body, section))
        except LocalResearchClientError as error:
            reached = str(error).partition(":")[2]
            failure = LocalResearchClientError(
                "local_client.saved_answer_section_unknown:" + reached
            )
            failure.sections = _section_parts(body, section)
            raise failure from error
    elif list_sections:
        reading["sections"] = list(body)
    return reading


def _section_parts(body: dict[str, Any], path: str) -> list[str]:
    """The parts an answer holds where a ``--section`` path stopped, each a path the next call
    can ask for (V319): the owner's outcome stands beside them (V129)."""
    value: Any = body
    parts, at = path.split("."), 0
    while at < len(parts):
        step = _step(value, parts, at)
        if step is None:
            stem = ".".join(parts[:at])
            names = (
                [str(name) for name in value]
                if isinstance(value, dict)
                else [str(index) for index in range(len(value))]
                if isinstance(value, list)
                else []
            )
            return [f"{stem}.{name}" if stem else name for name in names]
        value, taken = step
        at += taken
    return []


def _next_commands(
    body: dict[str, Any],
    workspace: Path,
    *,
    listed: bool = False,
    left: bool = False,
    options: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Each next request the owner offers, as the checkout's command that sends it (V123).

    Every answer carries its top level's; ``--list-next`` (``listed``) lists the listed items'
    as well, where a pending decision carries its own (V135), so the answer stays its size.
    A request that leaves a choice is never a runnable command: ``left`` gives those instead,
    each a template with the fields to choose (V136).
    """
    offered = offered_requests(body) if listed else body.get("next_requests")
    if not isinstance(offered, dict) or not offered:
        return None
    prefix = entry(workspace, options)
    found: dict[str, Any] = {}
    for name, request in offered.items():
        if not isinstance(request, dict) or bool(choices(request)) != left:
            continue
        line = command(request, prefix=prefix)
        if line is not None:
            found[name] = {"command": line, "choose": choices(request)} if left else line
    return found or None


RECONNECT: frozenset[str] = frozenset(
    {
        "local_client.service_not_running",
        "local_client.connection_lost_task_may_still_run",
        "local_client.connection_unreadable",
        "local_web.external_workspace_or_instance_mismatch",
        "local_web.external_token_invalid",
    }
)
"""What a Host that stopped or restarted answers: the waiter re-reads the connection file and
waits on, since the Task it waits for runs on under Task Control (WK)."""


def _sleep_before_read(delay: float, deadline: float | None) -> bool:
    """Sleep only to the cap; a completed sleep always has a read after it."""
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        return False
    time.sleep(delay if remaining is None else min(delay, remaining))
    return True


def _read_through_restarts(
    client: LocalResearchClient, document: dict[str, Any], deadline: float | None
) -> tuple[LocalResearchClient, dict[str, Any] | None]:
    """One read, waiting out a Host that stopped or restarted, including a last read at the cap.

    Each request waits no longer than the time left, beyond a long poll's own wait (V400).
    """
    delay = 1.0
    while True:
        remaining = None if deadline is None else deadline - time.monotonic()
        request = document
        if remaining is not None and "wait_seconds" in document:
            request = {**document, "wait_seconds": min(document["wait_seconds"], max(0, remaining))}
        try:
            body = client.request(
                request, timeout=None if remaining is None else max(0, remaining) + _POLL_MARGIN
            )
        except LocalResearchClientError as error:
            if str(error) not in RECONNECT:
                raise
        else:
            if str(body.get("refused") or body.get("failure_code") or "") not in RECONNECT:
                return client, body
        if not _sleep_before_read(delay, deadline):
            return client, None
        delay = min(delay * 2, 30.0)
        with suppress(LocalResearchClientError):
            client = LocalResearchClient(
                client.workspace,
                timeout=getattr(client, "timeout", DEFAULT_HTTP_WAIT_SECONDS),
                goal=getattr(client, "goal", None),
            )


_POLL_MARGIN = 5.0
"""Seconds a request may run past a waiter's deadline: a long poll asks the Host for no more
than the time left, and its answer needs a moment to arrive (V400)."""


def wait_event(
    kind: str,
    read: dict[str, Any] | None,
    *,
    prefix: tuple[str, ...] = ("alphalattice",),
    **facts: Any,
) -> dict[str, Any]:
    """The one line a waiter ends with: what happened and the command that reads it (WK).

    Args:
        kind: What happened.
        read: The request that reads it, or None.
        prefix: The entry its command starts with, its workspace named (V428).
        facts: What the event names; a None is left out.

    Returns:
        The event, its facts and its read command.
    """
    return {
        "event": kind,
        **{key: value for key, value in facts.items() if value is not None},
        "read": None if read is None else command(read, prefix=prefix),
    }


def _kept_options(client: LocalResearchClient, view: str = "compact") -> tuple[str, ...]:
    """The context every command this call prints keeps: its view, its language and the goal it
    named, so the agent copies a command and never retypes them (V406, V428, V449)."""
    language = ANSWER_LANGUAGE.get()
    goal = getattr(client, "goal", None)
    return (
        *(("--view", view) if view != "compact" else ()),
        *(("--lang", language) if language != "en" else ()),
        *(("--goal", goal) if goal else ()),
    )


def _entry_of(client: LocalResearchClient) -> tuple[str, ...]:
    """The entry a command this client prints starts with: the checkout's own, its workspace,
    and the context this call kept (`_kept_options`)."""
    return entry(client.workspace, _kept_options(client))


def _follow(
    client: LocalResearchClient,
    body: dict[str, Any],
    max_wait: float | None,
    *,
    each_stage: bool = False,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Follow admitted work until it ends or needs a decision; never on a timer (WK).

    A Task is followed through STATUS long polls, a feature trial through its readback. A
    Host that stops or restarts is waited for: the connection is re-read and the follow goes
    on, since the Task runs on under Task Control. Only ``max_wait``, the caller's own cap,
    ends it early, answered as pending. ``each_stage`` also ends it when the Task verifies a
    stage beyond those it held at the start (a coverage run's unit is one, V421). A state the
    Task never leaves by itself ends it too (`WAIT_EXITS`): a decision, a recovery, or a deferral,
    whose read names its retry time and the request that resumes it (V507). State changes go to
    stderr; the final answer keeps the admission that started the work and names what happened
    in ``wait_event``.
    """
    if max_wait is not None and not max_wait > 0:
        raise LocalResearchClientError("local_client.max_wait_invalid")
    trial = body.get("feature_trial_id") if body.get("status") == "FEATURE_TRIAL" else None
    # The Task this answer started, when it names one apart from its subject: a Portfolio
    # promotion runs a new Alpha Task first (V137).
    task = body.get("follow_task_id") or body.get("publication_task_id") or body.get("task_id")
    if outcome_of(body) != "PENDING" or task_state(body) in WAIT_EXITS or not (trial or task):
        return body
    document = (
        {"operation": "FEATURE_TRIAL_READBACK", "feature_trial_id": trial}
        if trial
        else {"operation": "STATUS", "task_id": task}
    )

    # The read that started the follow, when the follow stayed on the Task or trial it
    # selected: `--from` reads that selection again. A promotion or child Task followed instead
    # is another subject, which the parent's selection never names (V554).
    selected = body.get("read_request")
    kept_read = (
        {"read_request": selected}
        if isinstance(selected, dict)
        and str(selected.get("feature_trial_id" if trial else "task_id")) == str(trial or task)
        else {}
    )

    def final(current: dict[str, Any], kind: str, **extra: Any) -> dict[str, Any]:
        """The last read, with the admission, its next requests and the event it ended on."""
        return {
            **current,
            **kept_read,
            "admission": body,
            **(
                {"next_requests": body["next_requests"]}
                if not current.get("next_requests") and body.get("next_requests")
                else {}
            ),
            "wait_event": wait_event(
                kind,
                document,
                prefix=_entry_of(client),
                task_id=None if trial else str(task),
                feature_trial_id=trial,
                lifecycle=task_state(current),
                verified_stage_count=current.get("verified_stage_count")
                if kind == "STAGE_VERIFIED"
                else None,
            ),
            **extra,
        }

    if deadline is None and max_wait is not None:
        deadline = time.monotonic() + max_wait
    held = int(body.get("verified_stage_count") or 0)
    delay, last, current = 0.25, None, body
    # An incident open when the follow began is not news; a new one wakes it (GY2, WK).
    known = (body.get("incident") or {}).get("key")
    while True:
        # A Task's status waits on the Host for the Task to move on (N8); a trial is polled.
        remaining = None if deadline is None else deadline - time.monotonic()
        request = (
            document
            if trial or (remaining is not None and remaining < 0.1)
            else {**document, "wait_seconds": round(min(20.0, remaining or 20.0), 3)}
        )
        client, read = _read_through_restarts(client, request, deadline)
        if read is None:
            return final(current, "MAX_WAIT_REACHED", wait_status="MAX_WAIT_TASK_CONTINUES")
        current = read
        state = (
            (
                current.get("state"),
                [(s.get("step"), s.get("state")) for s in current.get("steps", [])],
            )
            if trial
            else (
                current.get("lifecycle"),
                current.get("verified_stage_count"),
                current.get("total_stage_count"),
            )
        )
        if state != last:
            print(json.dumps({"follow": document, "state": state}), file=sys.stderr, flush=True)
            last = state
        # A review to decide, a Task to recover or a deferral waits on a request, not on this
        # waiter (V507).
        if (exit_event := WAIT_EXITS.get(task_state(current) or "")) is not None:
            return final(current, exit_event)
        incident = current.get("incident") or {}
        if incident.get("key") and incident["key"] != known:
            return final(current, "INCIDENT")
        if outcome_of(current) != "PENDING":
            return final(current, "ENDED" if outcome_of(current) == "OK" else "STOPPED")
        if each_stage and int(current.get("verified_stage_count") or 0) > held:
            return final(current, "STAGE_VERIFIED")
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return final(current, "MAX_WAIT_REACHED", wait_status="MAX_WAIT_TASK_CONTINUES")
        if trial:
            _sleep_before_read(delay, deadline)
            delay = min(delay * 2, 1.0)


def _wakes(message: dict[str, Any], me: str | None, mine: set[str]) -> bool:
    """Whether a goal's message wakes this waiter (V503): one addressed to it, a reply to one it
    sent, or one addressed to no one; never its own, nor one between two other agents. A waiter
    whose session is unknown wakes on every new message, as before."""
    if me is None:
        return True
    if message.get("agent_id") == me:
        return False
    recipient = message.get("recipient_id")
    if recipient is not None:
        return bool(recipient == me)
    reply = message.get("reply_to")
    return reply is None or reply in mine


def _wait_for_goal(
    client: LocalResearchClient, goal_id: str, max_wait: float | None
) -> dict[str, Any]:
    """Wait for the next Task of a goal to end or need a decision, the next message under it
    for this waiter (an assignment wakes its assignee, a reply its sender), or the goal to close.

    The waiter is the agent session the command runs in, the bound session being the lead's
    (`native_research.py message`'s default sender); a message between two other agents never
    wakes it (V503)."""
    me = agent_provenance_headers(os.environ).get(AGENT_SESSION_HEADER)
    document = {"operation": "GOAL_NARRATIVE", "goal_id": goal_id}
    read = {"operation": "GOAL_SHOW", "goal_id": goal_id}
    deadline = None if max_wait is None else time.monotonic() + max_wait
    client, current = _read_through_restarts(client, document, deadline)
    prefix = _entry_of(client)
    if current is None or current.get("status") == "REFUSED":
        # A cap reached before the goal's first read leaves it as it was: pending, as every
        # other cap (V563).
        return current or {
            "wait_status": "MAX_WAIT_TASK_CONTINUES",
            "wait_event": wait_event("MAX_WAIT_REACHED", read, prefix=prefix, goal_id=goal_id),
        }
    # Each Task's state as last seen: a Task that recovered and stopped again is news, as
    # its first stop was (V535: a set of first states slept through the second).
    seen = {t["task_id"]: t["state"] for t in current["record"]["tasks"]}
    heard = {m["observation_id"] for m in current["record"]["conversation"]}
    delay = 2.0
    while True:
        if current.get("state") != "OPEN":
            return {
                **current,
                "wait_event": wait_event("GOAL_CLOSED", read, prefix=prefix, goal_id=goal_id),
            }
        conversation = current["record"]["conversation"]
        mine = {str(m.get("message_id")) for m in conversation if me and m.get("agent_id") == me}
        for message in conversation:
            if message["observation_id"] not in heard and _wakes(message, me, mine):
                return {
                    **current,
                    "wait_event": wait_event(
                        "MESSAGE",
                        read,
                        prefix=prefix,
                        goal_id=goal_id,
                        message_kind=message.get("message_kind"),
                        message_id=message.get("message_id"),
                        sender=message.get("agent_id"),
                        recipient=message.get("recipient_id"),
                    ),
                }
        for task in current["record"]["tasks"]:
            # Each Task's state as the Task waiter reads it: a decision or a deferral waits on a
            # request (V507), and one that ended or stopped (BLOCKED among them, V440) ends the
            # wait.
            state = task["state"]
            outcome = outcome_of({"lifecycle": state})
            settled = state in WAIT_EXITS or outcome != "PENDING"
            if settled and seen.get(task["task_id"]) != state:
                return {
                    **current,
                    "wait_event": wait_event(
                        WAIT_EXITS.get(state) or ("ENDED" if outcome == "OK" else "STOPPED"),
                        {"operation": "STATUS", "task_id": task["task_id"]},
                        prefix=prefix,
                        goal_id=goal_id,
                        task_id=task["task_id"],
                        lifecycle=task["state"],
                    ),
                }
        seen.update({t["task_id"]: t["state"] for t in current["record"]["tasks"]})
        if not _sleep_before_read(delay, deadline):
            return {
                **current,
                "wait_status": "MAX_WAIT_TASK_CONTINUES",
                "wait_event": wait_event("MAX_WAIT_REACHED", read, prefix=prefix, goal_id=goal_id),
            }
        delay = min(delay * 1.5, 15.0)
        client, again = _read_through_restarts(client, document, deadline)
        if again is not None:
            current = again


def _with_read_commands(body: dict[str, Any], prefix: tuple[str, ...]) -> dict[str, Any]:
    """A recent read's items, each with the command that opens it beside its request, starting
    with the checkout's entry and the workspace (V428)."""
    for group in body.get("groups", []):
        for item in group.get("items", []):
            if isinstance(item.get("read"), dict):
                item["read_command"] = command(item["read"], prefix=prefix)
    return body


def _codex_queue_ready() -> str:
    """The Codex thread a wake may be queued to, checked before the agent ends its turn."""
    import shutil  # only a Codex wake needs it; a call's imports are its cost (W12, V29)

    thread = os.environ.get("CODEX_THREAD_ID", "").strip()
    if not thread or shutil.which("codex") is None:
        raise LocalResearchClientError("local_client.codex_queue_unavailable")
    return thread


def _queue_wake(client: LocalResearchClient, thread: str, event: dict[str, Any]) -> dict[str, Any]:
    """Queue the event's one line to a Codex thread: best-effort, the Host's record the truth.

    A queued message arrives as a user message, so it carries no event body and no
    instruction: only what happened and the command that reads it. A failed queue is retried,
    then left as an undelivered wake beside the event, never a fallback in the user's voice.
    """
    import subprocess  # only a Codex wake needs it; a call's imports are its cost (W12, V29)

    message = f"Host event {event['event']}: read and verify it with {event['read']}"
    failure = None
    for attempt, pause in enumerate((0.0, 2.0, 4.0), start=1):
        time.sleep(pause)
        try:
            subprocess.run(
                ["codex", "queue", "--thread", thread, "--message", message],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except FileNotFoundError:
            failure = "CODEX_COMMAND_MISSING"
        except subprocess.CalledProcessError:
            failure = "CODEX_QUEUE_FAILED"
        except subprocess.TimeoutExpired:
            failure = "CODEX_QUEUE_TIMED_OUT"
        else:
            return {"channel": "codex-queue", "delivered": True, "attempts": attempt}
    subject = {key: str(event[key]) for key in ("task_id", "goal_id") if event.get(key) is not None}
    with suppress(LocalResearchClientError):
        client.publish_event(
            {
                "event_kind": "WAKE_UNDELIVERED",
                "producer_id": "alphalattice-waiter",
                "producer_session": uuid4().hex,
                "producer_sequence": 0,
                "occurred_at": datetime.now(UTC).isoformat(),
                "summary": f"A Codex wake for {event['event']} was not delivered ({failure}).",
                "subject": subject,
            }
        )
    return {"channel": "codex-queue", "delivered": False, "failure": failure, "attempts": 3}


def _wait(client: LocalResearchClient, args: argparse.Namespace) -> dict[str, Any]:
    """`activity wait`: one waiter, one subscription, one line when its event comes (WK)."""
    if (args.task_id is None) == (args.goal_id is None):
        raise LocalResearchClientError("local_client.wait_subject_required")
    each_stage = bool(getattr(args, "each_stage", False))
    if each_stage and args.task_id is None:
        raise LocalResearchClientError("local_client.each_stage_needs_task")
    if args.max_wait is not None and not args.max_wait > 0:
        raise LocalResearchClientError("local_client.max_wait_invalid")
    thread = _codex_queue_ready() if args.notify == "codex-queue" else None
    if args.goal_id is not None:
        body = _wait_for_goal(client, str(args.goal_id), args.max_wait)
    else:
        document = {"operation": "STATUS", "task_id": str(args.task_id)}
        # The cap counts from the waiter's start, its first read included (V400).
        started = time.monotonic()
        deadline = None if args.max_wait is None else started + args.max_wait
        client, status = _read_through_restarts(client, document, deadline)
        if status is None:
            # The cap came before a first read: an end like any other, its wake queued below
            # (V496).
            body = {
                "wait_status": "MAX_WAIT_TASK_CONTINUES",
                "wait_event": wait_event(
                    "MAX_WAIT_REACHED",
                    document,
                    prefix=_entry_of(client),
                    task_id=str(args.task_id),
                    lifecycle=None,
                ),
            }
        else:
            body = _follow(client, status, args.max_wait, each_stage=each_stage, deadline=deadline)
            if "wait_event" not in body:
                state = task_state(body)
                body = {
                    **body,
                    "wait_event": wait_event(
                        WAIT_EXITS.get(state or "", "ENDED"),
                        document,
                        prefix=_entry_of(client),
                        task_id=str(args.task_id),
                        lifecycle=state,
                    ),
                }
    # Every end queues its wake: the cap, a decision, an incident or the Task's end.
    if thread is not None and "wait_event" in body:
        body = {**body, "wake": _queue_wake(client, thread, body["wait_event"])}
    return body


def run(
    args: argparse.Namespace,
    *,
    document_override: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    before_send: Callable[[dict[str, Any]], None] | None = None,
    after_send: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> int:
    """One command over the running Host, printed as one envelope; returns its exit code.

    ``args.command`` is ``request`` (one operation document) or one of the client's own:
    ``workspace``, ``operations``, ``activity``, ``event``, ``cpu-budget``; the other fields are
    that command's. ``document_override`` is the request document, or the function that reads
    it, so a document that cannot be read is refused in the same envelope. ``before_send`` sees
    the request document before it goes: the CLI reads a session's lead there (V301), which the
    native bridge delivers through this client, so the client never imports the bridge.
    """
    started = time.perf_counter()
    body: dict[str, Any] | None = None
    raw_body: bytes | None = None
    saved_output: Path | None = None
    document: dict[str, Any] = {} if callable(document_override) else dict(document_override or {})
    display_view = args.view or "compact"
    listed = bool(getattr(args, "list_next", False))
    next_from = int(getattr(args, "next_from", 0) or 0)
    selected: dict[str, Any] = {}  # the --section part, once read from the answer (V112)
    saved_declaration: list[Path] = []  # the --save-declaration file, once written (V130)

    def answered(client: LocalResearchClient, outcome: Outcome, **extra: Any) -> int:
        """Print the owner's answer in its envelope, ``extra`` beside it; return its exit code."""

        assert body is not None
        options = _kept_options(client, display_view)
        commands = _next_commands(body, client.workspace, listed=listed, options=options)
        templates = _next_commands(
            body, client.workspace, listed=listed, left=True, options=options
        )
        # `--list-next` shows every listed request as the answer's view, in the one envelope
        # (V261); a `--section` asked for beside it is the view instead.
        listing = {"next_commands": commands, "next_templates": templates} if listed else None
        viewed = selected or listing
        compact = display_view == "compact"
        navigation: dict[str, Any] = (
            {"navigation": client.navigation(document, body)} if compact and not viewed else {}
        )
        answer = envelope(
            operation=document.get("operation") or args.command,
            outcome=outcome,
            body=body,
            elapsed_seconds=time.perf_counter() - started,
            display=viewed or ({} if compact else None),
            local_web_url=client.selected_url(document, body),
            next_commands=commands,
            next_templates=templates,
            context=_context(client.workspace, client.goal, body),
            **navigation,
            **({"output_file": str(saved_output.resolve())} if saved_output is not None else {}),
            **(
                {"declaration_file": str(saved_declaration[0].resolve())}
                if saved_declaration
                else {}
            ),
            **extra,
        )
        if listing is not None and not selected:
            if compact:
                # The listing is the data; the envelope repeats none of it (V411).
                answer.update(dict.fromkeys(_NEXT_KEYS))
            elif next_from:
                answer["data"] = _listing_page(listing, next_from, None)[0]
        if compact:
            # Its next requests too are held to the read, the rest counted (V411).
            _hold_next(answer, COMPACT_NEXT_BYTES)
        if compact and listing is not None and not selected:
            room = max(
                COMPACT_DATA_FLOOR_BYTES,
                COMPACT_ANSWER_BYTES - _size(short_references({**answer, "data": None})),
            )
            answer["data"], left = _listing_page(listing, next_from, room)
            answer["next_left"] = left
        if compact and (selected or not viewed):
            # A compact answer is one read: its data keeps what the rest leaves (V402).
            room = max(
                COMPACT_DATA_FLOOR_BYTES,
                COMPACT_ANSWER_BYTES - _size(short_references({**answer, "data": None})),
            )
            if not viewed:
                answer.update(
                    compact_display(
                        body, budget=room, whole=outcome in {"REFUSED", "INVALID_INPUT"}
                    )
                )
            elif _size(selected) > room:
                cut = compact_display(selected["value"], budget=room, section=selected["section"])
                answer.update({**cut, "data": {**selected, "value": cut["data"]}})
        print(
            json.dumps(
                (short_references if compact else _same)(answer),
                default=_json_value,
                sort_keys=True,
            )
        )
        return EXIT_CODES[outcome]

    try:
        if not 0 < args.request_timeout <= 600:
            raise LocalResearchClientError("local_client.request_timeout_outside_0_600")
        if next_from < 0 or (next_from and not listed):
            raise LocalResearchClientError("local_client.next_from_invalid")
        if sum(getattr(args, field, None) == Path("-") for field in ("file", "from_response")) > 1:
            raise LocalResearchClientError("local_client.document_multiple_stdin_sources")
        output = getattr(args, "output", None)
        # Compact unless asked: one read, every caller alike; `--view full` is the
        # exact form a script reads, and `--output` keeps it whatever is shown (V408).
        display_view = args.view or "compact"
        if output is not None and output.exists():
            raise LocalResearchClientError("local_client.output_exists_choose_another_path")
        client = LocalResearchClient(
            args.workspace, timeout=args.request_timeout, goal=getattr(args, "goal", None)
        )
        if callable(document_override):
            # Read after the Host is found, as a request file is: no Host answers NO_HOST.
            document = document_override()
        if args.command == "activity-wait":
            body = _wait(client, args)
        else:
            if document_override is None:
                document = _next_request(args)
            if (
                document.get("operation") == "REPORT"
                and document.get("portfolio_session")
                and args.format != "json"
            ):
                raise LocalResearchClientError("local_client.dated_position_requires_json")
            if before_send is not None:
                before_send(document)
            directory = _bundle_directory(document)
            body, raw_body = (
                client.report_html(str(document.get("result_hash")))
                if document.get("operation") in {"REPORT", "EXPORT"} and args.format == "html"
                else client.exchange(document)
            )
            if after_send is not None:
                after_send(document, body)
            if (named := named_read(document, body)) is not body:
                # The CLI's copy names what the read read, printed and saved, so `--from` reads
                # it again; the Host's answer stays its owner's (OP15, V449).
                body, raw_body = named, None
            if document.get("operation") == "ACTIVITY_RECENT":
                body = _with_read_commands(body, _entry_of(client))
            if args.wait:
                if output is not None and args.format in {"json", "yaml"}:
                    # The admission first, so an interrupt while waiting never loses the Task
                    # it started; the final answer replaces it (V273, OP3).
                    _save_output(
                        output,
                        body,
                        raw_body or json.dumps(body, default=_json_value).encode("utf-8"),
                        args.format,
                    )
                    saved_output = output
                body = _follow(client, body, getattr(args, "max_wait", None))
                raw_body = None  # The answer is the final read, not the transport bytes.
            if (
                document.get("operation") == "AGENT_BUNDLE_PREPARE"
                and directory is not None
                and body.get("status") == "AGENT_BUNDLE_READY"
            ):
                body = _write_bundle(directory, body, prefix=_entry_of(client))
        if output is not None:
            # The owner's exact answer, a refusal included, so a script reading --output
            # never loses one and --from can continue a blocked owner's next requests.
            # Every command shares the same no-overwrite writer. Preserve transport
            # bytes when available; schema/activity/wait projections are full JSON,
            # never the compact presentation that is printed below.
            raw_body = raw_body or json.dumps(body, default=_json_value).encode("utf-8")
            _save_output(
                output,
                body,
                raw_body,
                getattr(args, "format", "json"),
                replace=saved_output is not None,
            )
            saved_output = output
        if getattr(args, "declaration", None) is not None:
            _save_declaration(args.declaration, body)
            saved_declaration.append(args.declaration)
        if getattr(args, "section", None):
            # Read before anything prints, so a path the answer lacks is reported beside the
            # owner's outcome, the whole answer shown, as any local failure is (V129).
            selected.update(answer_part(body, str(args.section)))
        return answered(client, outcome_of(body))
    except LocalResearchClientError as error:
        code = str(error)
        transport: dict[str, Any] = {}
        if code == "local_client.connection_lost_task_may_still_run":
            transport = {
                "transport_reason": (
                    "HTTP_WAIT_EXPIRED"
                    if isinstance(error.__cause__, TimeoutError)
                    else "CONNECTION_INTERRUPTED"
                ),
                "request_timeout_seconds": args.request_timeout,
                "operation_outcome": "UNKNOWN",
                "guidance": (
                    "Inspect the same Task before resubmitting work. For a read-only "
                    "request, --request-timeout may increase HTTP waiting, not the Task budget."
                ),
            }
        refusal = client_refusal(code)
        if body is not None:
            # The owner answered before a local step failed (writing --output or a bundle,
            # following the Task): the operation's own outcome stands, and the local failure
            # is reported beside it, so an admitted operation never reads as refused and is
            # never sent again (V129). A wait that lost its Host keeps the admission it has
            # in --output, as the receipt of the work it started (V139).
            wanted = getattr(args, "output", None)
            if wanted is not None and saved_output is None and not wanted.exists():
                with suppress(LocalResearchClientError):
                    _save_output(
                        wanted,
                        body,
                        raw_body or json.dumps(body, default=_json_value).encode("utf-8"),
                        getattr(args, "format", "json"),
                    )
                    saved_output = wanted
            return answered(
                client,
                outcome_of(body),
                local_failure={
                    "failure_code": code,
                    "detail": refusal.detail,
                    "next_action": refusal.next_action,
                    **transport,
                    **(
                        {"sections": _section_parts(body, str(args.section))}
                        if code.startswith("local_client.section_unknown:")
                        else {}
                    ),
                },
            )
        print(
            json.dumps(
                envelope(
                    operation=document.get("operation") or args.command,
                    outcome=refusal.outcome,
                    body=None,
                    elapsed_seconds=time.perf_counter() - started,
                    status="REFUSED",
                    failure_code=code,
                    detail=refusal.detail,
                    next_action=refusal.next_action,
                    context=_context(args.workspace, getattr(args, "goal", None)),
                    **(
                        {"document_location": error.document_location}
                        if error.document_location is not None
                        else {}
                    ),
                    **(
                        {"document_size": error.document_size}
                        if error.document_size is not None
                        else {}
                    ),
                    **(
                        {"short_reference": error.short_reference}
                        if error.short_reference is not None
                        else {}
                    ),
                    **transport,
                ),
                default=_json_value,
                sort_keys=True,
            )
        )
        return EXIT_CODES[refusal.outcome]
    except ValueError:
        # Pydantic errors can contain the entire document. Never echo it.
        code = "local_client.request_invalid"
        refusal = client_refusal(code)
        print(
            json.dumps(
                envelope(
                    operation=None,
                    outcome=refusal.outcome,
                    body=None,
                    elapsed_seconds=time.perf_counter() - started,
                    status="REFUSED",
                    failure_code=code,
                    detail=refusal.detail,
                    next_action=refusal.next_action,
                ),
                sort_keys=True,
            )
        )
        return EXIT_CODES[refusal.outcome]
