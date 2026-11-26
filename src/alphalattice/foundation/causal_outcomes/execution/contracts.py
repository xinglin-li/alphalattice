"""Typed causal execution-outcome authorities and release contracts."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.enums import RebalanceFrequency

_FORMULA_ID = "formation-close-next-common-open-following-common-open-simple-return"
_ACTION_ID = "provider-split-adjusted-open-and-period-dividend"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class CausalExecutionSchedulePoint(_Contract):
    """Pair a formation close with its actual entry and holding end opens."""

    sequence: int = Field(ge=1)
    formation_session: date
    formation_close_at: datetime
    entry_session: date
    entry_open_at: datetime
    holding_end_session: date
    holding_end_open_at: datetime
    actual_session_span: int = Field(ge=2)


class LocalQAMarketSnapshot(_Contract):
    """Source-exact daily observations; scheduled future dates are not observations."""

    kind: Literal["LocalQAMarketSnapshot"] = "LocalQAMarketSnapshot"
    purpose: Literal["QA_NOT_FORWARD_AVAILABILITY_AUTHORITY"] = (
        "QA_NOT_FORWARD_AVAILABILITY_AUTHORITY"
    )
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    through: date
    ordered_listing_ids: tuple[str, ...]
    schedule: tuple[CausalExecutionSchedulePoint, ...]
    bars: tuple[RawDailyBar, ...]
    actions: tuple[CorporateActionEvent, ...]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = cls.model_construct(**values).model_dump(mode="json", exclude={"content_hash"})
        return cls(**payload, content_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_snapshot(self) -> Self:
        axis = self.ordered_listing_ids
        listing_scope = set(axis)
        keys = tuple((v.session_date, v.listing_id) for v in self.bars)
        sessions = tuple(v.formation_session for v in self.schedule)
        if (
            not axis
            or axis != tuple(sorted(listing_scope))
            or keys != tuple(sorted(set(keys)))
            or not sessions
            or sessions != tuple(sorted(set(sessions)))
            or self.through not in sessions
            or any(
                v.session_date > self.through or v.listing_id not in listing_scope
                for v in self.bars
            )
            or any(
                v.effective_date > self.through or v.listing_id not in listing_scope
                for v in self.actions
            )
            or any(
                not (v.formation_session < v.entry_session < v.holding_end_session)
                for v in self.schedule
            )
            or self.content_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"content_hash"}))
        ):
            raise ValueError("causal_outcomes.qa_market_snapshot_invalid")
        return self


class LocalQAOutcomePreparationSourcePrefix(_Contract):
    """Actual bounded values committed by the QA preparation owner in memory.

    ``through`` bounds consumed observations; it does not claim they were
    acquired then. A manifest revision, acquisition clock or source-admission
    claim is absent because none proves an unchanged semantic prefix.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, ser_json_inf_nan="constants"
    )
    through: date
    listing_ids: tuple[str, ...] = Field(min_length=1)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    schedule: tuple[CausalExecutionSchedulePoint, ...]
    bars: tuple[RawDailyBar, ...]
    actions: tuple[CorporateActionEvent, ...]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_prefix_scope(self) -> Self:
        """Refuse unbounded observations or inconsistent ordered source axes."""
        keys = tuple((value.session_date, value.listing_id) for value in self.bars)
        sessions = tuple(value.formation_session for value in self.schedule)
        listing_scope = set(self.listing_ids)
        if (
            self.listing_ids != tuple(sorted(listing_scope))
            or keys != tuple(sorted(set(keys)))
            or sessions != tuple(sorted(set(sessions)))
            or any(value.formation_session > self.through for value in self.schedule)
            or any(
                value.session_date > self.through or value.listing_id not in listing_scope
                for value in self.bars
            )
            or any(
                value.effective_date > self.through or value.listing_id not in listing_scope
                for value in self.actions
            )
        ):
            raise ValueError("causal_outcomes.qa_preparation_source_prefix_invalid")
        return self


class _LocalQAPreparationRecord(_Contract):
    """Seal only the local QA preparation owner's immutable records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = cls.model_construct(**values).model_dump(mode="json", exclude={"content_hash"})
        return cast(
            Self,
            cls.model_validate_json(
                json.dumps({**payload, "content_hash": canonical_hash(payload)})
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_preparation_identity(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("causal_outcomes.qa_preparation_identity_invalid")
        return self


class LocalQAOutcomePreparationPart(_LocalQAPreparationRecord):
    """Commit one Parquet base or increment without granting outcome authority."""

    kind: Literal["LocalQAOutcomePreparationPart"] = "LocalQAOutcomePreparationPart"
    role: Literal["BASE", "INCREMENT"]
    listing_count: int = Field(ge=1)
    point_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    ipc_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_part_shape(self) -> Self:
        """Require one row per listing and matured point in the sealed part."""
        if self.row_count != self.listing_count * self.point_count:
            raise ValueError("causal_outcomes.qa_preparation_part_invalid")
        return self


class LocalQAOutcomePreparationHead(_LocalQAPreparationRecord):
    """Link immutable QA rows to a freshly verified source prefix and request."""

    kind: Literal["LocalQAOutcomePreparationHead"] = "LocalQAOutcomePreparationHead"
    purpose: Literal["QA_NOT_FORWARD_AVAILABILITY_AUTHORITY"] = (
        "QA_NOT_FORWARD_AVAILABILITY_AUTHORITY"
    )
    schema_id: Literal["desktop-causal-execution-outcome"] = "desktop-causal-execution-outcome"
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_count: int = Field(ge=1)
    requested_session_count: int = Field(ge=1)
    requested_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    matured_point_count: int = Field(ge=0)
    matured_points_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_prefix_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    through: date
    previous_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parts: tuple[LocalQAOutcomePreparationPart, ...] = Field(min_length=1)
    part: LocalQAOutcomePreparationPart | None = None
    ipc_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_head_shape(self) -> Self:
        """Require the exact ordered base and increment rows of this request."""
        if (
            self.matured_point_count > self.requested_session_count
            or self.parts[0].role != "BASE"
            or any(value.role != "INCREMENT" for value in self.parts[1:])
            or any(value.listing_count != self.listing_count for value in self.parts)
            or sum(value.point_count for value in self.parts) != self.matured_point_count
            or (self.previous_head_hash is None and (self.part is None or self.part.role != "BASE"))
            or (self.previous_head_hash is None and self.parts != (self.part,))
            or (
                self.previous_head_hash is not None
                and self.part is not None
                and self.part.role != "INCREMENT"
            )
            or (
                self.part is not None
                and (
                    self.part.listing_count != self.listing_count
                    or self.part != self.parts[-1]
                    or self.part.point_count > self.matured_point_count
                    or (
                        self.previous_head_hash is None
                        and self.part.point_count != self.matured_point_count
                    )
                )
            )
        ):
            raise ValueError("causal_outcomes.qa_preparation_head_invalid")
        return self


class LocalQAOutcomePreparationMarker(_LocalQAPreparationRecord):
    """Publish the current QA head and its rollback root atomically, last."""

    kind: Literal["LocalQAOutcomePreparationMarker"] = "LocalQAOutcomePreparationMarker"
    purpose: Literal["QA_NOT_FORWARD_AVAILABILITY_AUTHORITY"] = (
        "QA_NOT_FORWARD_AVAILABILITY_AUTHORITY"
    )
    current_head_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class LocalQAOutcomePreparationFile(_Contract):
    """Name exact physical bytes for the existing governed retention owner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    relative_path: str
    byte_count: int = Field(ge=0)
    file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class LocalQAOutcomePreparationRetentionInventory(_LocalQAPreparationRecord):
    """List rooted preparation files and exact unrooted candidates; never evict."""

    kind: Literal["LocalQAOutcomePreparationRetentionInventory"] = (
        "LocalQAOutcomePreparationRetentionInventory"
    )
    current_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    previous_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    referenced_heads: tuple[str, ...] = ()
    protected_files: tuple[LocalQAOutcomePreparationFile, ...]
    candidate_files: tuple[LocalQAOutcomePreparationFile, ...]


class CausalExecutionOutcomeChunk(_Contract):
    """Identify one immutable outcome table by split, year, and content."""

    split: Literal["DEVELOPMENT", "SEALED_HOLDOUT"]
    year: int = Field(ge=1900, le=2200)
    row_count: int = Field(ge=1)
    first_formation_session: date
    last_formation_session: date
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    uri: str


class CausalExecutionOutcomeManifest(_Contract):
    """Seal the development and holdout chunks of one daily outcome axis."""

    kind: Literal["CausalExecutionOutcomeSnapshot"] = "CausalExecutionOutcomeSnapshot"
    research_cadence: Literal[RebalanceFrequency.DAILY]  # type: ignore[valid-type]
    market_as_of: date
    listing_ids: tuple[str, ...] = Field(min_length=1)
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    schedule_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_session_triples_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_rows_semantic_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    price_basis: Literal["open_split_adjusted"] = "open_split_adjusted"
    return_formula_identity: Literal[
        "formation-close-next-common-open-following-common-open-simple-return"
    ] = _FORMULA_ID  # type: ignore[assignment]
    corporate_action_identity: Literal[
        "split-adjusted-open-and-period-dividend",
        "provider-split-adjusted-open-and-period-dividend",
    ] = _ACTION_ID  # type: ignore[assignment]
    data_validity_class: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"]
    development_chunks: tuple[CausalExecutionOutcomeChunk, ...]
    sealed_holdout_chunks: tuple[CausalExecutionOutcomeChunk, ...]
    development_formation_count: int = Field(ge=1)
    sealed_holdout_formation_count: int = Field(ge=1)
    limitations: tuple[str, ...]
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> CausalExecutionOutcomeManifest:
        """Require canonical listings, physical split, and manifest hash."""
        if self.listing_ids != tuple(sorted(set(self.listing_ids))):
            raise ValueError("causal execution listing order is not canonical")
        if not self.development_chunks or not self.sealed_holdout_chunks:
            raise ValueError("causal execution outcome requires physical dev/holdout separation")
        if any(value.split != "DEVELOPMENT" for value in self.development_chunks) or any(
            value.split != "SEALED_HOLDOUT" for value in self.sealed_holdout_chunks
        ):
            raise ValueError("causal execution chunk split is invalid")
        if self.snapshot_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"snapshot_hash"})
        ):
            raise ValueError("causal execution outcome snapshot hash is invalid")
        return self


class CausalExecutionOutcomeMarker(_Contract):
    """Point to one published outcome manifest under a sealed marker hash."""

    kind: Literal["CausalExecutionOutcomeMarker"] = "CausalExecutionOutcomeMarker"
    snapshot_hash: str
    manifest_ref: str
    schedule_hash: str
    listing_set_hash: str
    marker_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> CausalExecutionOutcomeMarker:
        """Verify the marker hash against its published fields."""
        if self.marker_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"marker_hash"})
        ):
            raise ValueError("causal execution marker hash is invalid")
        return self


class DevelopmentOnlyExecutionOutcomeManifest(_Contract):
    """A snapshot whose label span is too long to be split at a sealed boundary.

    The frozen ``CausalExecutionOutcomeManifest`` cannot describe this method and
    should not be widened to. It pins ``return_formula_identity`` to the
    one-session formula, and it requires a sealed-holdout child -- which for a
    six-session span would mean writing rows whose labels overlap the boundary
    into a Holdout nobody authorized for them. Both constraints are correct for
    the method they were written for.

    So this is a successor rather than a relaxation. It carries development
    chunks and nothing else, states its scope in a field rather than leaving it
    to a directory name, and names its return formula openly because the method
    seal is what fixes which formula was installed. The frozen contract, its
    marker and its reader keep their exact bytes and behaviour.
    """

    kind: Literal["DevelopmentOnlyExecutionOutcomeSnapshot"] = (
        "DevelopmentOnlyExecutionOutcomeSnapshot"
    )
    research_cadence: Literal[RebalanceFrequency.DAILY]  # type: ignore[valid-type]
    market_as_of: date
    listing_ids: tuple[str, ...] = Field(min_length=1)
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    schedule_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_session_triples_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_rows_semantic_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    price_basis: Literal["open_split_adjusted"] = "open_split_adjusted"
    return_formula_identity: str = Field(min_length=1, max_length=128)
    """Stated, not enumerated. Which formula was actually installed is fixed by
    the method seal, and a second closed enumeration here would have to be
    widened for every future method that this contract exists to keep out of the
    frozen one."""

    corporate_action_identity: Literal[
        "split-adjusted-open-and-period-dividend",
        "provider-split-adjusted-open-and-period-dividend",
    ] = _ACTION_ID  # type: ignore[assignment]
    data_validity_class: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"]
    allowed_scope: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    """The whole point of the successor, and a field rather than a convention so
    a consumer can refuse it without inspecting where the file happened to sit."""

    development_chunks: tuple[CausalExecutionOutcomeChunk, ...] = Field(min_length=1)
    development_formation_count: int = Field(ge=1)
    maturity_lag_sessions: int = Field(ge=2)
    limitations: tuple[str, ...]
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require canonical development chunks and the snapshot hash."""
        if self.listing_ids != tuple(sorted(set(self.listing_ids))):
            raise ValueError("development-only execution listing order is not canonical")
        if any(value.split != "DEVELOPMENT" for value in self.development_chunks):
            # A sealed chunk reaching this contract would be a Holdout row
            # published under a scope that says it holds none.
            raise ValueError("development-only execution outcome carries a sealed chunk")
        if self.snapshot_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"snapshot_hash"})
        ):
            raise ValueError("development-only execution outcome snapshot hash is invalid")
        return self


class DevelopmentOnlyExecutionOutcomeMarker(_Contract):
    """Terminal publication marker for one development-only snapshot.

    A separate ``kind`` from the frozen marker, so the two can never be resolved
    for each other even if a file were moved between categories.
    """

    kind: Literal["DevelopmentOnlyExecutionOutcomeMarker"] = "DevelopmentOnlyExecutionOutcomeMarker"
    snapshot_hash: str
    manifest_ref: str
    schedule_hash: str
    listing_set_hash: str
    allowed_scope: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    marker_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Verify the development-only marker hash."""
        if self.marker_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"marker_hash"})
        ):
            raise ValueError("development-only execution marker hash is invalid")
        return self


class CausalExecutionOutcomeReceipt(_Contract):
    """Record one publication or exact reuse with clock and work counters."""

    kind: Literal["CausalExecutionOutcomeReceipt"] = "CausalExecutionOutcomeReceipt"
    snapshot_hash: str
    marker_ref: str
    action: Literal["PUBLISHED", "REUSED_EXACT"]
    source_watermark_hash: str
    completed_at: datetime
    raw_bar_payload_rows_read: int = Field(ge=0)
    action_payload_rows_read: int = Field(ge=0)
    durations_seconds: dict[str, float]
    receipt_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> CausalExecutionOutcomeReceipt:
        """Require a timezone-aware clock, valid durations, and receipt hash."""
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("causal execution receipt clock is invalid")
        if any(value < 0 for value in self.durations_seconds.values()):
            raise ValueError("causal execution receipt duration is invalid")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("causal execution receipt hash is invalid")
        return self


class PolicyHoldoutExecutionOutcomeRelease(_Contract):
    """Claim-bound release of one exact Portfolio Policy Holdout outcome axis."""

    kind: Literal["PolicyHoldoutExecutionOutcomeRelease"] = "PolicyHoldoutExecutionOutcomeRelease"
    claim_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    universe_manifest_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_watermark_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    embargo_session: date
    formation_sessions: tuple[date, ...] = Field(min_length=252, max_length=252)
    extension_chunk: CausalExecutionOutcomeChunk
    typed_user_authority: Literal["USER_AUTHORIZED_ONE_TIME_POLICY_HOLDOUT"] = (
        "USER_AUTHORIZED_ONE_TIME_POLICY_HOLDOUT"
    )
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    release_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_release(self) -> Self:
        """Require the authorized axis, extension chunk, and release hash."""
        if (
            self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or self.extension_chunk.split != "SEALED_HOLDOUT"
            or self.extension_chunk.first_formation_session != self.formation_sessions[-1]
            or self.extension_chunk.last_formation_session != self.formation_sessions[-1]
            or self.release_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"release_hash"}))
        ):
            raise ValueError("policy holdout execution outcome release is invalid")
        return self


class PolicyHoldoutExecutionOutcomeAuthority(_Contract):
    """Model Validation capability checked against its active durable claim."""

    kind: Literal["PolicyHoldoutExecutionOutcomeAuthority"] = (
        "PolicyHoldoutExecutionOutcomeAuthority"
    )
    claim_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mandate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    slate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1, max_length=120)
    source_manifest_ref: str = Field(min_length=1)
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    alpha_program_hashes: tuple[str, ...] = Field(min_length=1, max_length=2)
    validation_numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    embargo_session: date
    formation_sessions: tuple[date, ...] = Field(min_length=252, max_length=252)
    claimed_at: datetime
    typed_user_authority: Literal["USER_AUTHORIZED_ONE_TIME_POLICY_HOLDOUT"] = (
        "USER_AUTHORIZED_ONE_TIME_POLICY_HOLDOUT"
    )
    system_holdout_state: Literal["UNREAD"] = "UNREAD"
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_authority(self) -> Self:
        """Require canonical formation authority and its sealed hash."""
        if (
            self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or self.embargo_session >= self.formation_sessions[0]
            or self.claimed_at.tzinfo is None
            or self.claimed_at.utcoffset() is None
            or self.authority_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"authority_hash"}))
        ):
            raise ValueError("policy holdout execution outcome authority is invalid")
        return self


__all__ = [
    "CausalExecutionOutcomeChunk",
    "CausalExecutionOutcomeManifest",
    "CausalExecutionOutcomeMarker",
    "CausalExecutionOutcomeReceipt",
    "CausalExecutionSchedulePoint",
    "DevelopmentOnlyExecutionOutcomeManifest",
    "DevelopmentOnlyExecutionOutcomeMarker",
    "LocalQAOutcomePreparationFile",
    "LocalQAOutcomePreparationHead",
    "LocalQAOutcomePreparationMarker",
    "LocalQAOutcomePreparationPart",
    "LocalQAOutcomePreparationRetentionInventory",
    "LocalQAOutcomePreparationSourcePrefix",
    "PolicyHoldoutExecutionOutcomeAuthority",
    "PolicyHoldoutExecutionOutcomeRelease",
]
