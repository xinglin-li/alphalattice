"""Feature and trial request fields and their public descriptions."""

from typing import Any, Literal

from pydantic import ConfigDict, Field

from alphalattice.kernel.identity import BLAKE3_PATTERN

from .contracts import RequestFieldModel


class RequestFields(RequestFieldModel):
    """Feature preparation and trial fields for their operations."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    feature_document: dict[str, Any] | None = None
    """A Feature definition as a document, for the Feature catalog to plan."""
    feature_plan_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """A Feature catalog plan, by the hash its PLAN answered."""
    feature_output: Literal["RAW_VALUES", "PREPROCESSED_VALUES"] | None = None
    """What a Feature build writes: `RAW_VALUES`, or `PREPROCESSED_VALUES` after the catalog's
    preprocessing."""
    feature_trial_id: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """A feature trial, by the ID its request answered with."""
    feature_factor_id: str | None = Field(default=None, min_length=1, max_length=160)
    """A formula factor a research plan declares, which its review and activation name."""
    extensions_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the declared formula factors to read, from 1."""
