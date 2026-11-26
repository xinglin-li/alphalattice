"""Immutable local Formula columns in the existing Feature artifact store."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Self

import numpy as np
import numpy.typing as npt
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
    arrow_table_logical_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import ClosureArtifactRef
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"
VALUE_CATEGORY = "research-formula-values"
COLUMN_CATEGORY = "research-formula-columns"
INDEX_CATEGORY = "research-formula-value-index"
RECEIPT_CATEGORY = "research-feature-materializations"
PREPARATION_CATEGORY = "research-feature-preparations"


def column_binding_hash(
    *,
    input_binding_hash: str,
    numerical_spec_hash: str,
    implementation_hash: str,
    source_projection_hash: str,
) -> str:
    """Hash the inputs, numerical recipe, implementation, and source projection."""
    return str(
        canonical_hash(
            {
                "kind": "ResearchFormulaBinding",
                "input_binding_hash": input_binding_hash,
                "numerical_spec_hash": numerical_spec_hash,
                "implementation_hash": implementation_hash,
                "source_projection_hash": source_projection_hash,
            }
        )
    )


class _Sealed(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        draft = cls.model_construct(**values, content_hash="0" * 64)
        return cls(
            **values,
            content_hash=canonical_hash(draft.model_dump(mode="json", exclude={"content_hash"})),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify_hash(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("feature_research.artifact_identity_invalid")
        return self


class ResearchFormulaColumn(_Sealed):
    """Describe one immutable raw Formula value column and its axes."""

    kind: Literal["ResearchFormulaColumn"] = "ResearchFormulaColumn"
    input_binding_hash: str = Field(pattern=_HASH)
    numerical_spec_hash: str = Field(pattern=_HASH)
    implementation_hash: str = Field(pattern=_HASH)
    source_projection_hash: str = Field(pattern=_HASH)
    binding_hash: str = Field(pattern=_HASH)
    sessions: tuple[date, ...] = Field(min_length=1)
    listing_ids: tuple[str, ...] = Field(min_length=1)
    values: ClosureArtifactRef
    available_count: int = Field(ge=0)
    interpretation: Literal["RAW_FORMULA_NOT_PANEL_OR_FACTOR_ADMISSION"] = (
        "RAW_FORMULA_NOT_PANEL_OR_FACTOR_ADMISSION"
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify_axis(self) -> Self:
        """Verify the column binding, axes, and value count."""
        if (
            self.binding_hash
            != column_binding_hash(
                input_binding_hash=self.input_binding_hash,
                numerical_spec_hash=self.numerical_spec_hash,
                implementation_hash=self.implementation_hash,
                source_projection_hash=self.source_projection_hash,
            )
            or self.sessions != tuple(sorted(set(self.sessions)))
            or self.listing_ids != tuple(sorted(set(self.listing_ids)))
            or self.values.row_count != len(self.sessions) * len(self.listing_ids)
            or self.available_count > self.values.row_count
            or self.values.kind != "ResearchFormulaValues"
        ):
            raise ValueError("feature_research.column_binding_invalid")
        return self


class ResearchFeatureMaterialization(_Sealed):
    """Record computed and reused raw research Formula columns."""

    kind: Literal["ResearchFeatureMaterialization"] = "ResearchFeatureMaterialization"
    definition_plan_hash: str = Field(pattern=_HASH)
    input_binding_hash: str = Field(pattern=_HASH)
    implementation_hash: str = Field(pattern=_HASH)
    columns: tuple[tuple[str, str], ...]
    inherited_input_factor_ids: tuple[str, ...]
    computed_factor_ids: tuple[str, ...]
    reused_factor_ids: tuple[str, ...]
    kernel_calls_in_this_attempt: int = Field(ge=0)
    claim: Literal["RAW_COLUMNS_ONLY_NO_GLOBAL_OR_DOWNSTREAM_ADMISSION"] = (
        "RAW_COLUMNS_ONLY_NO_GLOBAL_OR_DOWNSTREAM_ADMISSION"
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify_scope(self) -> Self:
        """Verify the computed and reused column scopes."""
        ids = tuple(k for k, _ in self.columns)
        if (
            ids != tuple(sorted(set(ids)))
            or set(self.computed_factor_ids) & set(self.reused_factor_ids)
            or set(ids) != set(self.computed_factor_ids) | set(self.reused_factor_ids)
            or set(ids) & set(self.inherited_input_factor_ids)
            or any(
                len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for _, h in self.columns
            )
        ):
            raise ValueError("feature_research.materialization_scope_invalid")
        return self


class ResearchFeaturePreparation(_Sealed):
    """Record a development overlay prepared from raw Formula values."""

    kind: Literal["ResearchFeaturePreparation"] = "ResearchFeaturePreparation"
    definition_plan_hash: str = Field(pattern=_HASH)
    input_binding_hash: str = Field(pattern=_HASH)
    raw_materialization_hash: str = Field(pattern=_HASH)
    overlay_hash: str | None = Field(default=None, pattern=_HASH, exclude_if=lambda v: v is None)
    implementation_hash: str = Field(pattern=_HASH)
    preprocessing_calls_in_this_attempt: int = Field(ge=0)
    claim: Literal["PREPROCESSED_DEVELOPMENT_OVERLAY_NOT_REGISTERED_FACTOR_INPUT"] = (
        "PREPROCESSED_DEVELOPMENT_OVERLAY_NOT_REGISTERED_FACTOR_INPUT"
    )


class ResearchFormulaValues:
    """Publish/verify value children; definitions and Task authority stay outside."""

    def __init__(self, store: PanelClosureArtifactStore):
        """Use the closure artifact store for Formula value children."""
        self.store = store

    def read(self, content_hash: str) -> ResearchFormulaColumn:
        """Read and verify a Formula column descriptor."""
        return self.open(content_hash)[0]

    def open(self, content_hash: str) -> tuple[ResearchFormulaColumn, npt.NDArray[np.float64]]:
        """One verified read for a caller that consumes the values and their descriptor."""
        column = self.store.load_model(
            category=COLUMN_CATEGORY, content_hash=content_hash, model=ResearchFormulaColumn
        )
        if column.content_hash != content_hash:
            raise ValueError("feature_research.column_reference_mismatch")
        return column, self.values(column)

    def values(self, column: ResearchFormulaColumn) -> npt.NDArray[np.float64]:
        """Read and verify the value matrix for a Formula column."""
        table = self.store.load_parquet(category=VALUE_CATEGORY, reference=column.values)
        if (
            table.column_names != ["value_bits", "available"]
            or arrow_table_logical_hash(table) != column.values.content_hash
        ):
            raise ValueError("feature_research.value_content_mismatch")
        bits: npt.NDArray[np.uint64] = np.asarray(table["value_bits"].to_numpy(), dtype="<u8")
        values = bits.view("<f8")
        available = np.asarray(table["available"].to_numpy(), dtype=bool)
        if (
            not np.array_equal(np.isfinite(values), available)
            or int(available.sum()) != column.available_count
        ):
            raise ValueError("feature_research.value_availability_mismatch")
        return values.reshape((len(column.sessions), len(column.listing_ids)))

    def lookup(self, binding_hash: str) -> ResearchFormulaColumn | None:
        """Find a verified Formula column by binding hash, if present."""
        path = self.store.root / INDEX_CATEGORY / f"{binding_hash}.json"
        if not path.exists():
            return None
        index = self.store.load_json(category=INDEX_CATEGORY, content_hash=binding_hash)
        column = self.read(str(index["column_hash"]))
        if index.get("binding_hash") != binding_hash or column.binding_hash != binding_hash:
            raise ValueError("feature_research.value_index_mismatch")
        return column

    def publish(self, *, values: npt.NDArray[np.float64], **binding: Any) -> ResearchFormulaColumn:
        """Publish a verified raw Formula column and binding index."""
        array = np.ascontiguousarray(values, dtype="<f8")
        if (
            array.shape != (len(binding["sessions"]), len(binding["listing_ids"]))
            or np.isinf(array).any()
        ):
            raise ValueError("feature_research.value_axis_invalid")
        available = np.isfinite(array)
        array = np.ascontiguousarray(np.where(available, array, np.nan), dtype="<f8")
        table = pa.table(
            {"value_bits": array.reshape(-1).view("<u8"), "available": available.reshape(-1)}
        )
        reference = self.store.publish_parquet(
            category=VALUE_CATEGORY,
            content_hash=arrow_table_logical_hash(table),
            kind="ResearchFormulaValues",
            table=table,
        )
        column = ResearchFormulaColumn.create(
            **binding, values=reference, available_count=int(available.sum())
        )
        self.store.publish_json(
            category=COLUMN_CATEGORY,
            content_hash=column.content_hash,
            payload=column.model_dump(mode="json"),
        )
        self.read(column.content_hash)
        self.store.publish_json(
            category=INDEX_CATEGORY,
            content_hash=column.binding_hash,
            payload={"binding_hash": column.binding_hash, "column_hash": column.content_hash},
        )
        return column
