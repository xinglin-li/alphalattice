"""The winsorized market-excess Sector EMA's surfaces, read back.

The builder retired with the goal loop that wrote these surfaces (GR3); what
stays is the shape an existing surface is read back and verified in: its sealed
manifest, the table beside it and the table's content hash.

It was never a scientific control for the clean Sector target. It modelled
market-*excess* returns and published an ordinal companion; the clean target is
the absolute mean constituent log return, so it is absent from the clean-target
catalog.

Every class name, artifact kind and hash is unchanged. The `kind` literals are
inside the hashed payloads, so renaming any of them to match the module name
would break every frozen manifest this method has ever written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self, cast

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectorEmaAlphaManifest(_Contract):
    """Seal a retained EMA-alpha surface to policy, source, sector axis and table identity.

    The manifest records sessions, available values and canonical sector identifiers. manifest_hash
    binds these commitments for historical readback.
    """

    kind: Literal["SectorEmaAlphaManifest"] = "SectorEmaAlphaManifest"
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_count: int = Field(ge=1)
    sector_ids: tuple[str, ...] = Field(min_length=1)
    available_value_count: int = Field(ge=0)
    table_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_manifest(self) -> Self:
        """Require a canonical sector axis and exact retained EMA manifest identity.

        Returns:
            This manifest after sector-axis and canonical-hash validation.

        Raises:
            ValueError: Sector identifiers are unordered/duplicated or manifest_hash is invalid.
        """
        if self.sector_ids != tuple(sorted(set(self.sector_ids))):
            raise ValueError("Sector EMA sector axis is not canonical")
        if self.manifest_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"manifest_hash"})
        ):
            raise ValueError("Sector EMA manifest hash is invalid")
        return self


@dataclass(frozen=True, slots=True)
class SectorEmaAlphaSurface:
    """Hold a retained EMA-alpha Arrow table with its sealed manifest.

    The producer supplies logical table identity and availability counts through the manifest; this
    value holder performs no new numerical fit.
    """

    table: pa.Table
    manifest: SectorEmaAlphaManifest


def sector_ema_table_content_hash(table: pa.Table) -> str:
    """Hash the logical EMA Arrow schema and rows after removing metadata/chunk layout.

    Args:
        table: Retained EMA-alpha table whose logical content is being identified.

    Returns:
        Canonical schema/row digest independent of schema metadata and physical chunking.
    """
    logical = table.replace_schema_metadata(None).combine_chunks()
    return cast(
        str,
        canonical_hash({"schema": str(logical.schema), "rows": logical.to_pylist()}),
    )


__all__ = ["SectorEmaAlphaManifest", "SectorEmaAlphaSurface", "sector_ema_table_content_hash"]
