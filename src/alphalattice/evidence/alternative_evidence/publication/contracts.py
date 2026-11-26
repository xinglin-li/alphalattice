"""The pointer-free Alternative Evidence analysis publication."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceContract,
    validate_contract_identity,
)
from alphalattice.kernel.shared_kernel.identity_successors import is_current

ACQUISITION_BINDING_ROLE = "alternative_evidence.acquisition_binding"
CANONICALIZATION_BINDING_ROLE = "alternative_evidence.canonicalization_binding"
RETRIEVAL_BINDING_ROLE = "alternative_evidence.retrieval_binding"
ANALYSIS_POLICY_ROLE = "alternative_evidence.analysis_policy"
DECISION_POLICY_ROLE = "alternative_evidence.decision_policy"
PUBLICATION_BINDING_ROLE = "alternative_evidence.publication_binding"
BINDING_ROLES = (
    ACQUISITION_BINDING_ROLE,
    CANONICALIZATION_BINDING_ROLE,
    RETRIEVAL_BINDING_ROLE,
    ANALYSIS_POLICY_ROLE,
    DECISION_POLICY_ROLE,
    PUBLICATION_BINDING_ROLE,
)
"""The identity role of each binding a publication records, in its fields' order
(`config/identity-roles.json`). A recorded binding names the installed one when it is that
value or recorded moves lead from it to that value (LAWS.md ID1)."""


def bindings_current(recorded: Sequence[str], installed: Sequence[str]) -> bool:
    """Whether each recorded binding names the installed one, by value or recorded moves."""
    return all(
        is_current(role, value, current)
        for role, value, current in zip(BINDING_ROLES, recorded, installed, strict=True)
    )


class AlternativeEvidenceAnalysisPublication(AlternativeEvidenceContract):
    """One immutable analysis lineage. Reuse is by identity, never by a pointer.

    `obligation_hash` is what a review compares against the obligation it
    re-derives from its own issuer scope. Entity-axis equality alone would not
    be enough: two runs can cover the same issuers under different cutoffs,
    source families and required checks, and only one of them answered the
    question this Portfolio asked.
    """

    kind: Literal["AlternativeEvidenceAnalysisPublication"] = (
        "AlternativeEvidenceAnalysisPublication"
    )
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_set_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_generation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    access_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    analyst_brief_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    analyst_brief_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cro_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    obligation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    acquisition_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonicalization_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    published_at: datetime
    expires_at: datetime
    publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_publication(self) -> Self:
        """Require aware publication, later expiry and canonical analysis-publication identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Publication is naive, expiry is not later, or canonical publication identity
                differs.
        """
        if (
            self.published_at.tzinfo is None
            or self.published_at.utcoffset() is None
            or self.expires_at <= self.published_at
        ):
            raise ValueError("alternative_evidence.analysis_publication_clock_invalid")
        validate_contract_identity(self, "publication_hash")
        return self


__all__ = [
    "ACQUISITION_BINDING_ROLE",
    "ANALYSIS_POLICY_ROLE",
    "BINDING_ROLES",
    "CANONICALIZATION_BINDING_ROLE",
    "DECISION_POLICY_ROLE",
    "PUBLICATION_BINDING_ROLE",
    "RETRIEVAL_BINDING_ROLE",
    "AlternativeEvidenceAnalysisPublication",
    "bindings_current",
]
