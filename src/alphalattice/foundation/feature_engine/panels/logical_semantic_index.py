"""Native semantic index over exact logical Feature Panel rows."""

from __future__ import annotations

from datetime import date
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.feature_engine.panels.logical_contracts import (
    PanelLogicalMembershipEpoch,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

LOGICAL_SLICE_ALGORITHM_IDENTITY = "ordered-logical-row-digest"


class FeaturePanelLogicalSessionSemantic(BaseModel):  # type: ignore[misc]
    """Bind one logical Panel session to its row count and ordered digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_date: date
    row_count: int = Field(gt=0)
    ordered_logical_row_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeaturePanelLogicalSemanticIndex(BaseModel):  # type: ignore[misc]
    """Bind logical Panel sessions, factors, and membership to slice identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = "FeaturePanelLogicalSemanticIndex"
    logical_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_ids: tuple[str, ...]
    active_listing_count: int = Field(gt=0)
    calendar_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    slice_algorithm_identity: str = LOGICAL_SLICE_ALGORITHM_IDENTITY
    sessions: tuple[FeaturePanelLogicalSessionSemantic, ...] = Field(min_length=1)
    # Present for a Panel whose sessions hold their own members: each
    # session's row count is its epoch's member count rather than the axis
    # length. Absent for a dense Panel and excluded from its identity.
    membership_epochs: tuple[PanelLogicalMembershipEpoch, ...] | None = None
    index_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_index(self) -> FeaturePanelLogicalSemanticIndex:
        """Reject inconsistent factor, session, membership, or index identities."""
        if not self.factor_ids or len(self.factor_ids) != len(set(self.factor_ids)):
            raise ValueError("logical semantic index factor axis is invalid")
        dates = tuple(item.session_date for item in self.sessions)
        if dates != tuple(sorted(set(dates))):
            raise ValueError("logical semantic index sessions must be sorted and unique")
        for item in self.sessions:
            if item.row_count != self.expected_row_count(item.session_date):
                raise ValueError("logical semantic index has an incomplete session")
        if self.calendar_hash != canonical_hash(dates):
            raise ValueError("logical semantic index calendar hash is invalid")
        if self.index_hash != canonical_hash(_index_identity(self)):
            raise ValueError("logical semantic index hash is invalid")
        return self

    def expected_row_count(self, session: date) -> int:
        """The rows a session holds: its members under per-session membership, else the axis."""
        if self.membership_epochs is None:
            return self.active_listing_count
        for epoch in self.membership_epochs:
            if epoch.first_session <= session <= epoch.last_session:
                return epoch.member_count
        raise ValueError("logical semantic index session has no membership epoch")

    def slice_hash(self, sessions: tuple[date, ...]) -> str:
        """Hash a sorted unique session slice under this logical index."""
        if not sessions or sessions != tuple(sorted(set(sessions))):
            raise ValueError("logical Panel slice sessions must be sorted and unique")
        by_date = {item.session_date: item for item in self.sessions}
        try:
            selected = tuple(by_date[value] for value in sessions)
        except KeyError as exc:
            raise ValueError("Factor Research slice is outside the logical semantic index") from exc
        return cast(
            str,
            canonical_hash(
                {
                    "slice_algorithm_identity": self.slice_algorithm_identity,
                    "logical_panel_hash": self.logical_panel_hash,
                    "catalog_semantics_hash": self.catalog_semantics_hash,
                    "listing_set_hash": self.listing_set_hash,
                    "factor_ids": self.factor_ids,
                    "sessions": [item.model_dump(mode="json") for item in selected],
                },
            ),
        )


def _index_identity(index: FeaturePanelLogicalSemanticIndex) -> dict[str, object]:
    excluded = {"index_hash"}
    if index.membership_epochs is None:
        excluded.add("membership_epochs")
    return cast(dict[str, object], index.model_dump(mode="json", exclude=excluded))


def build_logical_semantic_index(
    *,
    logical_panel_hash: str,
    catalog_semantics_hash: str,
    listing_set_hash: str,
    factor_ids: tuple[str, ...],
    active_listing_count: int,
    sessions: tuple[FeaturePanelLogicalSessionSemantic, ...],
    membership_epochs: tuple[PanelLogicalMembershipEpoch, ...] | None = None,
) -> FeaturePanelLogicalSemanticIndex:
    """Seal a logical semantic index from its explicit session commitments."""
    values = {
        "logical_panel_hash": logical_panel_hash,
        "catalog_semantics_hash": catalog_semantics_hash,
        "listing_set_hash": listing_set_hash,
        "factor_ids": factor_ids,
        "active_listing_count": active_listing_count,
        "calendar_hash": canonical_hash(tuple(item.session_date for item in sessions)),
        "slice_algorithm_identity": LOGICAL_SLICE_ALGORITHM_IDENTITY,
        "sessions": sessions,
        "membership_epochs": membership_epochs,
    }
    provisional = FeaturePanelLogicalSemanticIndex.model_construct(**values, index_hash="0" * 64)
    return FeaturePanelLogicalSemanticIndex(
        **values, index_hash=canonical_hash(_index_identity(provisional))
    )


__all__ = [
    "LOGICAL_SLICE_ALGORITHM_IDENTITY",
    "FeaturePanelLogicalSemanticIndex",
    "FeaturePanelLogicalSessionSemantic",
    "build_logical_semantic_index",
]
