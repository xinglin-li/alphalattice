"""The evidence selections recorded for a Portfolio issuer scope.

Which analysis a book's review reads is an answer a person or an agent records, sealed as an
`AdmittedEvidenceSelection` in the Evidence artifact store; the answer in force for a scope is the
newest record naming it.
"""

from __future__ import annotations

from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
)

from ..contracts import AdmittedEvidenceSelection

SELECTIONS_CATEGORY = "cro-evidence-selections"
"""The Evidence artifact category the selections are recorded in."""


def recorded_selections(
    artifacts: AlternativeEvidenceArtifactStore, scope_hash: str
) -> tuple[AdmittedEvidenceSelection, ...]:
    """Read every selection recorded for an issuer scope, the newest first.

    Args:
        artifacts: The Evidence artifact store.
        scope_hash: The issuer scope's hash.

    Returns:
        The selections by when each was chosen, the newest first.
    """
    records = [
        value
        for value in artifacts.values(SELECTIONS_CATEGORY, AdmittedEvidenceSelection)
        if value.issuer_scope_hash == scope_hash
    ]
    return tuple(
        sorted(records, key=lambda value: (value.chosen_at, value.selection_hash), reverse=True)
    )


def recorded_selection(
    artifacts: AlternativeEvidenceArtifactStore, scope_hash: str
) -> AdmittedEvidenceSelection | None:
    """Return the selection in force for an issuer scope: the newest record naming it.

    Args:
        artifacts: The Evidence artifact store.
        scope_hash: The issuer scope's hash.

    Returns:
        The newest selection, or None when none is recorded.
    """
    records = recorded_selections(artifacts, scope_hash)
    return records[0] if records else None


__all__ = ["SELECTIONS_CATEGORY", "recorded_selection", "recorded_selections"]
