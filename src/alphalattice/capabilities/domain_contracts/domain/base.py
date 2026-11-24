"""Shared validation primitives for persisted domain records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, StringConstraints

type SchemaVersion = Literal["1"]


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _require_uuid4(value: UUID) -> UUID:
    if value.version != 4:
        raise ValueError("identifier must be UUIDv4")
    return value


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_as_utc)]
Uuid4 = Annotated[UUID, AfterValidator(_require_uuid4)]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ShortString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class DomainModel(BaseModel):  # type: ignore[misc]
    """Strict, immutable base for records that can cross a persistence boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: SchemaVersion = "1"
