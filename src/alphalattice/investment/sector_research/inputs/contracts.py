"""Typed authority for the causal Sector state surface.

Sector Research owns what a Sector *was* at a formation. It does not own what a
Desk did with that state afterwards, which is why the Alpha experiment report
that used to live here did not come along: its fields are program and split
hashes, candidate ids, fit/predict/metric counts and policy-OOS state -- an
account of an Alpha experiment, stored here only because the surface it cited
happened to be stored here too. It had no consumer beyond its own store methods
and no artifact of it exists, so it was removed rather than relocated.

``SectorContextPolicy`` keeps its ``scaler`` field. That field describes a
training-only transform and on ownership grounds does not belong to this Desk
either -- but it sits inside the hashed ``policy_hash`` of every frozen manifest,
so removing it would break readback to buy tidiness. It is a compatibility field
and must not be read as preprocessing authority: Alpha's fold-fitted transformer
owns its own scaling, and a new Sector method may not reach for this one.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model
from alphalattice.kernel.shared_kernel.sector_treatment import SectorHistoryTreatment

_HASH = r"^[0-9a-f]{64}$"


class SectorContextContract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectorContextPolicy(SectorContextContract):
    """Seal the fixed four-feature Sector context policy and classification revision.

    The policy fixes trend/surprise windows, equal weighting, minimum history/membership and the
    training-only median/MAD scaler. Classification is current and backfilled;
    sector_point_in_time_qualified is explicitly False.
    """

    kind: Literal["SectorContextPolicy"] = "SectorContextPolicy"
    feature_ids: tuple[str, str, str, str] = (
        "sector_trend_20",
        "sector_surprise_0",
        "sector_surprise_1",
        "sector_surprise_5",
    )
    trend_half_life_sessions: Literal[20] = 20
    surprise_window_sessions: Literal[5] = 5
    surprise_half_life_sessions: Literal[3] = 3
    minimum_history_sessions: Literal[40] = 40
    minimum_sector_members: Literal[5] = 5
    source_semantics: Literal["LOG_EXECUTION_RETURN"] = "LOG_EXECUTION_RETURN"
    sector_weighting: Literal["EQUAL_WEIGHT"] = "EQUAL_WEIGHT"
    scaler: Literal["TRAINING_UNIQUE_SECTOR_MEDIAN_MAD"] = "TRAINING_UNIQUE_SECTOR_MEDIAN_MAD"
    sector_revision: str = Field(pattern=_HASH)
    sector_history_treatment: SectorHistoryTreatment = "CURRENT_CLASSIFICATION_BACKFILLED"
    sector_point_in_time_qualified: Literal[False] = False
    policy_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> Self:
        """Require the exact canonical Sector context policy identity.

        Returns:
            This policy after verifying policy_hash.

        Raises:
            ValueError: policy_hash differs from the complete policy content excluding that hash.
        """
        _validate_hash(self, "policy_hash")
        return self


class SectorContextManifest(SectorContextContract):
    """Bind a Sector context table to policy, source, ordered axes and payload bytes.

    Formation sessions and sector identifiers are sorted and unique. The manifest records available
    rows, logical table identity, serialized payload SHA-256 and its own canonical identity.
    """

    kind: Literal["SectorContextManifest"] = "SectorContextManifest"
    policy_hash: str = Field(pattern=_HASH)
    source_surface_hash: str = Field(pattern=_HASH)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    sector_ids: tuple[str, ...] = Field(min_length=1)
    available_row_count: int = Field(ge=0)
    table_content_hash: str = Field(pattern=_HASH)
    payload_sha256: str = Field(pattern=_HASH)
    manifest_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_manifest(self) -> Self:
        """Require canonical session/sector axes and the exact manifest identity.

        Returns:
            This manifest after axis and manifest_hash validation.

        Raises:
            ValueError: Either axis is unordered/duplicated or manifest_hash is inconsistent.
        """
        if self.formation_sessions != tuple(sorted(set(self.formation_sessions))):
            raise ValueError("alpha_research.sector_context_session_axis_invalid")
        if self.sector_ids != tuple(sorted(set(self.sector_ids))):
            raise ValueError("alpha_research.sector_context_sector_axis_invalid")
        _validate_hash(self, "manifest_hash")
        return self


def seal_sector_context_contract[ContractT: SectorContextContract](
    model: type[ContractT], identity_field: str, /, **values: object
) -> ContractT:
    """Seal one Sector context contract from explicit values.

    Args:
        model: Concrete Sector context model to construct.
        identity_field: Canonical self-identity field to compute.
        values: Declared contract values; identity is derived by the shared sealing owner.

    Returns:
        Validated context contract with its canonical self identity.

    Raises:
        pydantic.ValidationError: Values or contract consistency violate the concrete model.
    """
    return seal_model(model, values, field=identity_field)


def _validate_hash(value: SectorContextContract, field: str) -> None:
    if getattr(value, field) != canonical_hash(value.model_dump(mode="json", exclude={field})):
        raise ValueError("alpha_research.sector_context_identity_invalid")


__all__ = [
    "SectorContextManifest",
    "SectorContextPolicy",
    "seal_sector_context_contract",
]
