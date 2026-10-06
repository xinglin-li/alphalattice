"""Repository-defined canonical JSON and hashing profile."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from alphalattice.kernel.shared_kernel.domain.errors import DomainValidationError


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise DomainValidationError("naive datetime is not canonical", code="json.naive_time")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        normalized_text = unicodedata.normalize("NFC", value)
        try:
            normalized_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise DomainValidationError(
                "string cannot be represented as strict UTF-8",
                code="json.utf8",
            ) from exc
        return normalized_text
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DomainValidationError("non-finite float is not canonical", code="json.non_finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise DomainValidationError(
                    "JSON object keys must be strings", code="json.key_type"
                )
            normalized_key = unicodedata.normalize("NFC", key)
            try:
                normalized_key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise DomainValidationError(
                    "JSON object key cannot be represented as strict UTF-8",
                    code="json.utf8",
                ) from exc
            if normalized_key in normalized_mapping:
                raise DomainValidationError(
                    "JSON object keys collide after NFC normalization",
                    code="json.key_collision",
                )
            normalized_mapping[normalized_key] = _normalize(item)
        return normalized_mapping
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | memoryview):
        return [_normalize(item) for item in value]
    raise DomainValidationError(
        f"unsupported canonical JSON value: {type(value).__name__}",
        code="json.unsupported_type",
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize one value to the exact AlphaLattice canonical JSON byte profile."""
    normalized = _normalize(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def sha256_hex(value: bytes) -> str:
    """Hash exact bytes with SHA-256.

    Args:
        value: Exact byte sequence to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(value).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        normalized_key = unicodedata.normalize("NFC", key)
        if normalized_key in result:
            raise DomainValidationError(
                "duplicate or NFC-colliding JSON key", code="json.key_collision"
            )
        result[normalized_key] = value
    return result


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise DomainValidationError(
            f"invalid JSON number: {value}",
            code="json.non_finite",
        )
    return parsed


def strict_json_loads(data: bytes | str) -> Any:
    """Parse UTF-8 JSON while rejecting duplicate keys and non-finite constants."""
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_float=_strict_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                DomainValidationError(
                    f"invalid JSON constant: {value}",
                    code="json.non_finite",
                )
            ),
        )
    except UnicodeDecodeError as exc:
        raise DomainValidationError("JSON is not strict UTF-8", code="json.utf8") from exc
    except json.JSONDecodeError as exc:
        raise DomainValidationError("invalid JSON", code="json.invalid") from exc


def parse_model[ModelT: BaseModel](data: bytes, model_type: type[ModelT]) -> ModelT:
    """Parse strict JSON and validate its canonical representation as a contract.

    Args:
        data: UTF-8 JSON bytes; duplicate/NFC-colliding keys and non-finite numbers are refused.
        model_type: Pydantic contract used to validate the normalized JSON.

    Returns:
        Validated instance of the supplied contract.

    Raises:
        DomainValidationError: JSON is invalid or violates the canonical profile.
        pydantic.ValidationError: The parsed value violates the supplied contract.
    """
    parsed = strict_json_loads(data)
    model: ModelT = model_type.model_validate_json(canonical_json_bytes(parsed))
    return model
