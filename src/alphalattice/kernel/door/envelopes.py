"""The Door's stable request, answer and refusal envelopes."""

import json
from collections.abc import Callable, Mapping
from typing import Any, Final, Literal

OPERATIONS_PATH: Final = "/api/client/operations"
CLI_OPERATION: Final = "CLI_INVOKE"
SCHEMA_VERSION: Final = 1
Outcome = Literal["OK", "INVALID_INPUT", "REFUSED", "PENDING", "NO_HOST"]
_NO_HOST_ENVELOPE = (None, "NO_HOST", "local_client.service_not_running")


def request_bytes(
    fields: dict[str, Any],
    *,
    default: Callable[[object], str] | None = None,
    ensure_ascii: bool = True,
) -> bytes:
    """Encode the client document in its route's existing escaping dialect."""
    return json.dumps(fields, default=default, ensure_ascii=ensure_ascii).encode("utf-8")


def answer_bytes(answer: object) -> bytes:
    """Encode a normal Host answer in its stable key order."""
    return json.dumps(answer, default=str, sort_keys=True).encode("utf-8")


def refusal_fields(reason: str) -> dict[str, object]:
    """The transport refusal, with optional words added by the interface."""
    return {"refused": reason}


def cli_answer(
    *,
    operation: str | None,
    outcome: Outcome,
    body: dict[str, Any] | None,
    elapsed_seconds: float,
    display: dict[str, Any] | None,
    status: str | None,
    failure_code: str | None,
    detail: str | None,
    request: Mapping[str, Any] | None,
    titles: dict[str, str],
    launches: int | None,
    owner_refusal: str | None,
    present: Callable[[Any], str | None],
    project_error: Callable[[dict[str, Any]], dict[str, Any]],
    notices: Callable[[str | None, Mapping[str, Any], Mapping[str, Any]], dict[str, Any]],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep the CLI envelope's fields and order while the interface supplies its words."""
    owner = body or {}
    refusal = owner.get("refused")
    return {
        "status": status or owner.get("status", owner.get("disposition")),
        "next_action": owner.get("next_action"),
        "task_id": owner.get("task_id") or owner.get("publication_task_id"),
        "schema_version": SCHEMA_VERSION,
        "operation": (
            "request" if (body, outcome, failure_code) == _NO_HOST_ENVELOPE else operation
        ),
        "outcome": outcome,
        "data": body if display is None else display,
        **({"titles": titles} if titles else {}),
        "failure_code": failure_code
        or owner.get("failure_code")
        or (refusal if isinstance(refusal, str) else None),
        "detail": present(detail or owner.get("detail")),
        **project_error(owner | {"failure_code": failure_code or owner_refusal}),
        **(
            {"closure_detail": present(owner["closure_message_id"])}
            if owner.get("closure_message_id")
            else {}
        ),
        "next_requests": owner.get("next_requests"),
        "timing": {"elapsed_seconds": round(elapsed_seconds, 3)},
        # This agent session's CLI launches, help and schema included.
        **({"session_launches": launches} if launches is not None else {}),
        **extra,
        **notices(
            operation,
            owner | {"failure_code": failure_code or owner.get("failure_code")},
            request or {},
        ),
    }
