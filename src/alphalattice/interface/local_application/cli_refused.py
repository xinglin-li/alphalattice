"""Refusal detection for client answers."""

from typing import Any


def refused(body: dict[str, Any]) -> bool:
    """Check whether an owner's answer carries a refusal.

    Args:
        body: The full owner answer.

    Returns:
        Whether a failure code, explicit refusal, or refused disposition is present.
        A deferred readback can name the provider reason while its Task waits. A `refused`
        field is a refusal only as a code: the controls answer lists the controls a book
        refuses under it, and that catalog is an answer.
    """
    code = body.get("refused")
    return bool(
        (body.get("failure_code") and body.get("status") != "DEFERRED")
        or (isinstance(code, str) and code)
        or any(
            str(body.get(field, "")).startswith("REFUSED") for field in ("status", "disposition")
        )
    )
