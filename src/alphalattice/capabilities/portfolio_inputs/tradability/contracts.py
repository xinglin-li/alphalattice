"""Typed authorities for decision-time and realized execution tradability."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    CausalInputAuthority,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_from_dump

Hash = str
_HASH = r"^[0-9a-f]{64}$"


class TradabilityContract(BaseModel):  # type: ignore[misc]
    """Forbid undeclared fields and freeze Data-owned tradability contract values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DecisionTradabilityStatus(StrEnum):
    """State whether a planned order is eligible on decision-time evidence.

    Eligibility and unavailability describe causal planned-order admission. They do
    not claim that a future execution session produced a successful fill.
    """

    PLANNED_ORDER_ELIGIBLE = "PLANNED_ORDER_ELIGIBLE"
    PLANNED_ORDER_UNAVAILABLE = "PLANNED_ORDER_UNAVAILABLE"


class DecisionTradabilityReason(StrEnum):
    """Explain planned-order eligibility using formation-time market data and ADV history.

    Complete causal data admits the planned order. Missing formation data or
    incomplete ADV20 history retains a distinct unavailable reason.
    """

    CAUSAL_MARKET_DATA_COMPLETE = "CAUSAL_MARKET_DATA_COMPLETE"
    FORMATION_MARKET_DATA_UNAVAILABLE = "FORMATION_MARKET_DATA_UNAVAILABLE"
    ADV20_HISTORY_INCOMPLETE = "ADV20_HISTORY_INCOMPLETE"


class ExecutionObservationMethod(StrEnum):
    """Describe the post-session daily-bar evidence used to observe execution availability.

    A present bar and an observed bar absence are separate methods. Neither is
    decision-time knowledge of a later session's realized execution outcome.
    """

    POST_SESSION_DAILY_BAR = "POST_SESSION_DAILY_BAR"
    POST_SESSION_DAILY_BAR_ABSENCE = "POST_SESSION_DAILY_BAR_ABSENCE"


class TradabilitySurfaceChunk(TradabilityContract):
    """Bind a bounded Parquet chunk to ordered sessions, payload bytes and metadata.

    Attributes:
        surface_kind: DECISION for planned eligibility or EXECUTION for observed availability.
        formation_sessions: Sorted unique nonempty formation labels, at most 21 per chunk.
        row_count: Positive number of records committed by the chunk.
        payload_sha256: SHA-256 commitment to the exact Parquet payload.
        metadata_hash: Commitment to canonical serialized Arrow metadata.
        content_hash: Canonical identity of kind, sessions, count and byte/metadata commitments.
        uri: Exact content-addressed playpen URI derived from kind and identity.
    """

    surface_kind: Literal["DECISION", "EXECUTION"]
    formation_sessions: tuple[date, ...] = Field(min_length=1, max_length=21)
    row_count: int = Field(ge=1)
    payload_sha256: Hash = Field(pattern=_HASH)
    metadata_hash: Hash = Field(pattern=_HASH)
    content_hash: Hash = Field(pattern=_HASH)
    uri: str = Field(min_length=1)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_chunk(self) -> Self:
        """Require canonical session order, the chunk preimage hash and its exact URI.

        Returns:
            This validated chunk reference.

        Raises:
            ValueError: Session order, content identity or canonical URI differs.
        """
        if self.formation_sessions != tuple(sorted(set(self.formation_sessions))):
            raise ValueError("data_tradability.chunk_session_axis_invalid")
        identity = tradability_chunk_identity(
            surface_kind=self.surface_kind,
            formation_sessions=self.formation_sessions,
            row_count=self.row_count,
            payload_sha256=self.payload_sha256,
            metadata_hash=self.metadata_hash,
        )
        if self.content_hash != canonical_hash(identity):
            raise ValueError("data_tradability.chunk_identity_invalid")
        expected_uri = (
            "playpen://data-operations/tradability/"
            f"{self.surface_kind.lower()}/chunks/{self.content_hash}"
        )
        if self.uri != expected_uri:
            raise ValueError("data_tradability.chunk_uri_invalid")
        return self


class HistoricalDecisionTradabilitySurface(TradabilityContract):
    """Seal formation-time planned-order eligibility on a current-universe axis.

    Eligible and unavailable counts exhaust the formation-by-listing grid. Chunks
    retain decision-time data and history commitments; later observed execution
    availability is owned by the separate execution surface.

    Attributes:
        universe_epoch_hash: Universe epoch shared by the decision and execution surfaces.
        ordered_listing_ids: Nonempty sorted unique listing axis.
        source_watermark_hash: Source-data watermark commitment.
        schedule_hash: Intended decision/execution schedule commitment.
        formation_sessions: Nonempty sorted unique decision-session axis.
        intended_execution_sessions: Aligned later sessions intended for fills.
        chunks: Same-kind immutable chunks covering every formation/listing row.
        data_validity_class: Current-universe research qualification, without historical PIT
            membership.
        limitations: Explicit research, execution-observation and liquidity limits.
        surface_hash: Canonical identity of all other surface fields.
        eligible_row_count: Nonnegative number of planned-order-eligible cells.
        unavailable_row_count: Nonnegative number of unavailable planned-order cells.
    """

    kind: Literal["HistoricalDecisionTradabilitySurface"] = "HistoricalDecisionTradabilitySurface"
    universe_epoch_hash: Hash = Field(pattern=_HASH)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    source_watermark_hash: Hash = Field(pattern=_HASH)
    schedule_hash: Hash = Field(pattern=_HASH)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    intended_execution_sessions: tuple[date, ...] = Field(min_length=1)
    chunks: tuple[TradabilitySurfaceChunk, ...] = Field(min_length=1)
    eligible_row_count: int = Field(ge=0)
    unavailable_row_count: int = Field(ge=0)
    data_validity_class: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"] = (
        "CURRENT_UNIVERSE_RESEARCH_ONLY"
    )
    limitations: tuple[str, ...]
    surface_hash: Hash = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> Self:
        """Validate complete decision chunks, causal schedule, row totals and surface identity.

        Returns:
            This validated decision surface.

        Raises:
            ValueError: Axes, schedule, chunk coverage/kind, counts or identity are invalid.
        """
        _validate_common_surface(self, "DECISION")
        if self.eligible_row_count + self.unavailable_row_count != (
            len(self.formation_sessions) * len(self.ordered_listing_ids)
        ):
            raise ValueError("data_tradability.decision_counts_invalid")
        _validate_hash(self, "surface_hash")
        return self


class HistoricalExecutionAvailabilitySurface(TradabilityContract):
    """Seal observed execution availability separately from planned order eligibility.

    Status counts exhaust the formation-by-listing grid. The schedule retains later
    intended execution sessions while EXECUTION chunks identify the observed daily-bar
    evidence. This surface cannot grant knowledge of future fills to a decision.

    Attributes:
        universe_epoch_hash: Universe epoch shared by the decision and execution surfaces.
        ordered_listing_ids: Nonempty sorted unique listing axis.
        source_watermark_hash: Source-data watermark commitment.
        schedule_hash: Intended decision/execution schedule commitment.
        formation_sessions: Nonempty sorted unique decision-session axis.
        intended_execution_sessions: Aligned later sessions intended for fills.
        chunks: Same-kind immutable chunks covering every formation/listing row.
        data_validity_class: Current-universe research qualification, without historical PIT
            membership.
        limitations: Explicit research, execution-observation and liquidity limits.
        surface_hash: Canonical identity of all other surface fields.
        execution_status_counts: Execution-status counts whose sum equals the complete row grid.
    """

    kind: Literal["HistoricalExecutionAvailabilitySurface"] = (
        "HistoricalExecutionAvailabilitySurface"
    )
    universe_epoch_hash: Hash = Field(pattern=_HASH)
    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    source_watermark_hash: Hash = Field(pattern=_HASH)
    schedule_hash: Hash = Field(pattern=_HASH)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    intended_execution_sessions: tuple[date, ...] = Field(min_length=1)
    chunks: tuple[TradabilitySurfaceChunk, ...] = Field(min_length=1)
    execution_status_counts: dict[str, int]
    data_validity_class: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"] = (
        "CURRENT_UNIVERSE_RESEARCH_ONLY"
    )
    limitations: tuple[str, ...]
    surface_hash: Hash = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_surface(self) -> Self:
        """Validate complete execution chunks, causal schedule, status totals and identity.

        Returns:
            This validated execution surface.

        Raises:
            ValueError: Axes, schedule, chunk coverage/kind, count total or identity is invalid.
        """
        _validate_common_surface(self, "EXECUTION")
        if sum(self.execution_status_counts.values()) != (
            len(self.formation_sessions) * len(self.ordered_listing_ids)
        ):
            raise ValueError("data_tradability.execution_counts_invalid")
        _validate_hash(self, "surface_hash")
        return self


class HistoricalTradabilityBundle(TradabilityContract):
    """Bind decision and execution surface identities to one declared coverage scope.

    Attributes:
        universe_epoch_hash: Shared universe epoch commitment.
        schedule_hash: Shared execution schedule identity.
        decision_surface_hash: Planned-order eligibility surface identity.
        execution_surface_hash: Observed execution availability surface identity.
        formation_count: Positive formation-axis length.
        asset_count: Positive listing-axis length.
        coverage_start: First covered formation date.
        coverage_end: Last covered formation date.
        bundle_hash: Canonical bundle identity excluding this hash.
    """

    kind: Literal["HistoricalTradabilityBundle"] = "HistoricalTradabilityBundle"
    universe_epoch_hash: Hash = Field(pattern=_HASH)
    schedule_hash: Hash = Field(pattern=_HASH)
    decision_surface_hash: Hash = Field(pattern=_HASH)
    execution_surface_hash: Hash = Field(pattern=_HASH)
    formation_count: int = Field(ge=1)
    asset_count: int = Field(ge=1)
    coverage_start: date
    coverage_end: date
    bundle_hash: Hash = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_bundle(self) -> Self:
        """Require an ordered coverage interval and canonical bundle identity.

        Returns:
            This validated bundle.

        Raises:
            ValueError: Coverage is reversed or the bundle hash differs.
        """
        if self.coverage_start > self.coverage_end:
            raise ValueError("data_tradability.bundle_range_invalid")
        _validate_hash(self, "bundle_hash")
        return self


class CurrentTradabilityDataProjection(TradabilityContract):
    """Expose bounded current tradability coverage and readiness without source rows.

    Attributes:
        status: DATA_TRADABILITY_READY or DATA_TRADABILITY_BLOCKED.
        universe_epoch_label: Explicit current-universe research-only qualification.
        coverage_start: First declared coverage date.
        coverage_end: Last declared coverage date.
        formation_count: Positive number of formations.
        asset_count: Positive number of listings.
        latest_decision_formation: Latest formation represented in decision evidence.
        latest_intended_execution: Latest intended fill session, no earlier than that formation.
        latest_observed_execution: Latest execution session represented by observation evidence.
        eligible_decision_count: Nonnegative count of planned-order-eligible cells.
        limitations: Declared scope and source limitations.
        projection_hash: Canonical identity of all other projection fields.
    """

    kind: Literal["CurrentTradabilityDataProjection"] = "CurrentTradabilityDataProjection"
    status: Literal["DATA_TRADABILITY_READY", "DATA_TRADABILITY_BLOCKED"]
    universe_epoch_label: Literal["CURRENT_UNIVERSE_RESEARCH_ONLY"]
    coverage_start: date
    coverage_end: date
    formation_count: int = Field(ge=1)
    asset_count: int = Field(ge=1)
    latest_decision_formation: date
    latest_intended_execution: date
    latest_observed_execution: date
    eligible_decision_count: int = Field(ge=0)
    limitations: tuple[str, ...]
    projection_hash: Hash = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_projection(self) -> Self:
        """Validate formation/intended-execution clock order and projection identity.

        Returns:
            This validated projection.

        Raises:
            ValueError: The latest formation follows intended execution or the hash differs.
        """
        if self.latest_decision_formation > self.latest_intended_execution:
            raise ValueError("data_tradability.current_clock_invalid")
        _validate_hash(self, "projection_hash")
        return self


class TradabilityArtifactReference(TradabilityContract):
    """Carry a typed content-addressed reference used by a publication receipt.

    Attributes:
        artifact_kind: Decision surface, execution surface, bundle or current projection.
        content_hash: Typed artifact identity.
        metadata_hash: Published artifact metadata commitment.
        uri: Owner-resolved playpen artifact handle.
    """

    artifact_kind: Literal[
        "decision-surface", "execution-surface", "tradability-bundle", "current-projection"
    ]
    content_hash: Hash = Field(pattern=_HASH)
    metadata_hash: Hash = Field(pattern=_HASH)
    uri: str = Field(min_length=1)


class CurrentTradabilityDataReceipt(TradabilityContract):
    """Seal the four ordered artifact references and an aware publication clock.

    Attributes:
        artifacts: Exactly decision, execution, bundle and projection references in that order.
        published_at: Timezone-aware publication instant.
        receipt_hash: Canonical identity of the publication fields.
    """

    kind: Literal["CurrentTradabilityDataReceipt"] = "CurrentTradabilityDataReceipt"
    artifacts: tuple[TradabilityArtifactReference, ...] = Field(min_length=4, max_length=4)
    published_at: datetime
    receipt_hash: Hash = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> Self:
        """Require receipt artifact order, an aware clock and a matching hash.

        Returns:
            This validated receipt.

        Raises:
            ValueError: Reference order, publication clock or receipt identity is invalid.
        """
        if tuple(value.artifact_kind for value in self.artifacts) != (
            "decision-surface",
            "execution-surface",
            "tradability-bundle",
            "current-projection",
        ):
            raise ValueError("data_tradability.current_receipt_order_invalid")
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("data_tradability.current_receipt_clock_invalid")
        _validate_hash(self, "receipt_hash")
        return self


class CurrentTradabilityDataMarker(TradabilityContract):
    """Seal terminal publication lineage before the active pointer is replaced.

    Attributes:
        receipt_hash: Exact receipt identity.
        receipt_ref: Content-addressed receipt handle.
        projection_hash: Exact projection identity.
        projection_ref: Content-addressed projection handle.
        bundle_hash: Exact tradability bundle identity.
        bundle_ref: Content-addressed bundle handle.
        terminal_state: Ready or blocked terminal publication state.
        completed_at: Timezone-aware publication completion clock.
        marker_hash: Canonical terminal marker identity.
    """

    kind: Literal["CurrentTradabilityDataMarker"] = "CurrentTradabilityDataMarker"
    receipt_hash: Hash = Field(pattern=_HASH)
    receipt_ref: str = Field(min_length=1)
    projection_hash: Hash = Field(pattern=_HASH)
    projection_ref: str = Field(min_length=1)
    bundle_hash: Hash = Field(pattern=_HASH)
    bundle_ref: str = Field(min_length=1)
    terminal_state: Literal["DATA_TRADABILITY_READY", "DATA_TRADABILITY_BLOCKED"]
    completed_at: datetime
    marker_hash: Hash = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_marker(self) -> Self:
        """Require an aware completion clock and canonical terminal marker identity.

        Returns:
            This validated marker.

        Raises:
            ValueError: The completion clock or marker identity is invalid.
        """
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("data_tradability.current_marker_clock_invalid")
        _validate_hash(self, "marker_hash")
        return self


class CurrentTradabilityDataPointer(TradabilityContract):
    """Name the active terminal marker and its receipt/projection lineage.

    Attributes:
        marker_hash: Active terminal marker identity.
        marker_ref: Exact marker handle.
        receipt_hash: Receipt identity linked by the marker.
        receipt_ref: Exact receipt handle.
        projection_hash: Projection identity linked by the marker.
        projection_ref: Exact projection handle.
        pointer_hash: Canonical identity of all pointer fields except this hash.
    """

    kind: Literal["CurrentTradabilityDataPointer"] = "CurrentTradabilityDataPointer"
    marker_hash: Hash = Field(pattern=_HASH)
    marker_ref: str = Field(min_length=1)
    receipt_hash: Hash = Field(pattern=_HASH)
    receipt_ref: str = Field(min_length=1)
    projection_hash: Hash = Field(pattern=_HASH)
    projection_ref: str = Field(min_length=1)
    pointer_hash: Hash = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_pointer(self) -> Self:
        """Require the canonical identity of the active-pointer declaration.

        Returns:
            This validated pointer; owner readback verifies its referenced lineage.

        Raises:
            ValueError: The pointer hash differs from its fields.
        """
        _validate_hash(self, "pointer_hash")
        return self


def seal_tradability_contract[ContractT: TradabilityContract](
    model: type[ContractT], identity_field: str, /, **values: Any
) -> ContractT:
    """Derive a named identity field from the typed dump and validate the result.

    Args:
        model: Tradability contract type to construct.
        identity_field: Hash field excluded from its own preimage.
        values: Contract values whose normalized typed dump is sealed.

    Returns:
        The validated contract with its owner-derived canonical identity.
    """
    return seal_model_from_dump(model, values, field=identity_field)


def tradability_chunk_identity(
    *,
    surface_kind: Literal["DECISION", "EXECUTION"],
    formation_sessions: tuple[date, ...],
    row_count: int,
    payload_sha256: str,
    metadata_hash: str,
) -> dict[str, object]:
    """Return the sole canonical identity preimage for a Parquet chunk."""
    return {
        "surface_kind": surface_kind,
        "formation_sessions": formation_sessions,
        "row_count": row_count,
        "payload_sha256": payload_sha256,
        "metadata_hash": metadata_hash,
    }


def _validate_common_surface(
    value: HistoricalDecisionTradabilitySurface | HistoricalExecutionAvailabilitySurface,
    kind: Literal["DECISION", "EXECUTION"],
) -> None:
    if value.ordered_listing_ids != tuple(sorted(set(value.ordered_listing_ids))):
        raise ValueError("data_tradability.listing_axis_invalid")
    if value.formation_sessions != tuple(sorted(set(value.formation_sessions))):
        raise ValueError("data_tradability.formation_axis_invalid")
    if len(value.formation_sessions) != len(value.intended_execution_sessions):
        raise ValueError("data_tradability.schedule_axis_invalid")
    schedule = zip(
        value.formation_sessions,
        value.intended_execution_sessions,
        strict=True,
    )
    if any(left >= right for left, right in schedule):
        raise ValueError("data_tradability.schedule_clock_invalid")
    if any(chunk.surface_kind != kind for chunk in value.chunks):
        raise ValueError("data_tradability.chunk_kind_invalid")
    chunk_sessions = tuple(
        session for chunk in value.chunks for session in chunk.formation_sessions
    )
    if chunk_sessions != value.formation_sessions:
        raise ValueError("data_tradability.chunk_axis_invalid")
    if sum(chunk.row_count for chunk in value.chunks) != (
        len(value.formation_sessions) * len(value.ordered_listing_ids)
    ):
        raise ValueError("data_tradability.row_count_invalid")


def _validate_hash(model: BaseModel, field: str) -> None:
    if getattr(model, field) != canonical_hash(model.model_dump(mode="json", exclude={field})):
        raise ValueError(f"data_tradability.{model.__class__.__name__}_identity_invalid")


__all__ = [
    "CurrentTradabilityDataMarker",
    "CurrentTradabilityDataPointer",
    "CurrentTradabilityDataProjection",
    "CurrentTradabilityDataReceipt",
    "DecisionTradabilityReason",
    "DecisionTradabilityStatus",
    "ExecutionObservationMethod",
    "HistoricalDecisionTradabilitySurface",
    "HistoricalExecutionAvailabilitySurface",
    "HistoricalTradabilityBundle",
    "TradabilityArtifactReference",
    "TradabilitySurfaceChunk",
    "seal_tradability_contract",
    "tradability_chunk_identity",
    "tradability_input_authority",
]


def tradability_input_authority(
    *, bundle_hash: str, universe_epoch_hash: str, input_id: str = "tradability_universe"
) -> CausalInputAuthority:
    """The decision universe's own clock, and the one thing it cannot claim.

    The surface records ``observed_through`` as a *date* -- the formation session
    -- so the finest instant it can honestly name is that session's official
    close, which is what an ADV20 and an eligibility decision are complete at.
    That is stated here rather than sharpened: a surface whose column is a date
    cannot support a claim about an intra-session instant, and pretending
    otherwise is how a universe acquires information it never had.

    ``CURRENT_MEMBERSHIP_BACKFILLED`` is not a timing verdict. The rows are
    causal on every session and the membership is still today's, so the two
    findings travel separately and the second one blocks nothing here -- it is
    the science owner's call, not the clock's.
    """
    return CausalInputAuthority.create(
        input_id=input_id,
        input_kind="TRADABILITY_UNIVERSE",
        temporal_usage="DECISION_INPUT",
        surface_hash=bundle_hash,
        owner_authority_id="portfolio_inputs.tradability.surface",
        owner_identity_hash=universe_epoch_hash,
        observed_through=AnchoredInstantPolicy.exchange_event(
            offset_sessions=0, event="OFFICIAL_CLOSE", policy_id="TRADABILITY_OBSERVED_THROUGH"
        ),
        source_available=AnchoredInstantPolicy.exchange_event(
            offset_sessions=0, event="OFFICIAL_CLOSE", policy_id="TRADABILITY_AVAILABLE"
        ),
        observation_semantics=(
            "Decision eligibility, execution availability and causal ADV20 for "
            "formation T, recorded with a session-granular observed_through and "
            "therefore resolved no finer than the official close of T."
        ),
        point_in_time_disposition="CURRENT_MEMBERSHIP_BACKFILLED",
    )
