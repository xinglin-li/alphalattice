"""JSON encoding for the local Host transport."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from alphalattice.interface.local_application.failure_codes import LocalResearchClientError
from alphalattice.kernel.door import request_bytes


def _json_value(value: object) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise LocalResearchClientError("local_client.document_not_json_compatible")


def request_body(document: dict[str, Any]) -> bytes:
    """Use UTF-8 JSON: escapes could make admitted Unicode fields exceed the body limit."""
    return request_bytes(document, default=_json_value, ensure_ascii=False)
