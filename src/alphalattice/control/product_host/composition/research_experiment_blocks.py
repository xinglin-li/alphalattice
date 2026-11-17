"""Repaired artifact and capacity blocks."""

RETRYABLE_ARTIFACT_BLOCKS: frozenset[str] = frozenset(
    {
        "risk_research.covariance_chunk_identity_invalid",
        "risk_research.covariance_chunk_tampered",
        "risk_research.development_checkpoint_matrix_mismatch",
        "risk_research.development_checkpoint_scope_mismatch",
        "risk_research.development_checkpoint_axis_mismatch",
        "research_authoring.evidence_artifact_unverifiable",
    }
)
"""Stage refusals that name a required artifact that was missing or did not read.

Once it is back, the same execution contract can continue from its verified prefix
and checkpoint, so a person may ask for that. A scientific or permission refusal,
an exhausted budget or an unrepaired source is not in this set and stays a stop.
"""


CAPACITY_BLOCKS = {
    "storage.managed_capacity_exceeded": "Raise the cap (`storage set --cap-bytes <bytes>`) or "
    "confirm a cleanup on the person's yes",
    "storage.disk_space_insufficient": "Free disk space or move the workspace to a larger disk",
}
"""Capacity stops and their way on; the same declaration goes on once restored."""


def retryable_block(failure_code: str | None) -> str | None:
    """The repaired-artifact or operator-capacity stop this owner permits retrying."""
    if not failure_code:
        return None
    code = failure_code.split(":", 1)[0]
    return code if code in RETRYABLE_ARTIFACT_BLOCKS or code in CAPACITY_BLOCKS else None
