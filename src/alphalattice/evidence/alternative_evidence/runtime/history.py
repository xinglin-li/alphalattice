"""What the evidence runtime keeps for records sealed under a retired format or selection.

The public release upgrades workspaces in place, so these readers and refusals ship. Each
names the sealed shape it serves and the migration that ends it; retiring one needs that
migration -- an upgrade-time conversion of the old records to the current format, with
provenance -- not a deletion (the retirement list, rows H6, M1 and M3). What stays where it
is, and why, is on that list too: the single-request preparation helpers (H7) are keyed by
the adapter's own Task identity and would import it back.
"""

from __future__ import annotations

from ..analysis.contracts import AlternativeEvidenceRetrievalAccessReceipt
from ..contracts import (
    INTEGRATED_FAMILY_SPELLING,
    MATTER_SELECTION_CANDIDATE,
    MATTER_SELECTION_INTEGRATED,
    AlternativeEvidenceRequest,
    MatterSelectionPolicy,
    matter_selection_identity,
    matter_selection_retired,
)


def receipt_matter_selection_id(receipt: AlternativeEvidenceRetrievalAccessReceipt) -> str:
    """Return the matter selection identity sealed in a receipt.

    The record carries the production plan (without an allocation, or none),
    or the candidate allocation with its named families -- the same identity
    `AlternativeEvidenceRequest.matter_selection_id` gives the request.

    H6: the production and candidate branches read receipts sealed under the
    selections the first-release integration retired, so the sealed-selection
    index keys them apart from the integrated method's; they end with the
    migration that re-keys those receipts (D2 and M1).
    """
    matters = receipt.litigation_matters
    if matters is None or matters.allocation_rules_id is None:
        return matter_selection_identity(None)
    if receipt.routing is not None:
        # The integrated method is one spelling -- the three families the
        # workspace manifests bind -- whatever inventories the method ran
        # (`matters.families` names them; the operations inventory joined
        # them under the completeness assignment without moving the identity
        # every sealed request and manifest carries).
        return matter_selection_identity(
            MatterSelectionPolicy(
                method=MATTER_SELECTION_INTEGRATED, families=INTEGRATED_FAMILY_SPELLING
            )
        )
    return matter_selection_identity(
        MatterSelectionPolicy(method=MATTER_SELECTION_CANDIDATE, families=matters.families)
    )


def require_current_selection(request: AlternativeEvidenceRequest) -> None:
    """Refuse a request under a retired matter selection.

    The ``matter_selection_retired`` refusal applies to the production plan (the
    omitted field) and the candidate needs allocation, retired by the
    first-release integration, and section V's residual opt-ins,
    retired in section X. Their sealed receipts read back; a new
    selection or continuation under one is neither dealt under another
    method nor run.

    M1: a workspace installed under a retired selection reaches this (67 of
    the 120 retained workspaces on 2026-09-25); it ends when every workspace
    has been installed again under the integrated selection.
    """
    if matter_selection_retired(request.matter_selection):
        raise ValueError("alternative_evidence.matter_selection_policy_retired")


def require_answer_format[AnswerT](answer: AnswerT | None) -> AnswerT:
    """Refuse an analysis submitted in the retired answer format.

    `submission_format_retired` carries a whole brief as the actor
    wrote it (`SubmittedEvidenceAnalysis.submission`), which still parses so
    such Tasks read back for their lineage, but is no longer executed.

    M3: 244 such Tasks were still queued in the retained workspaces on
    2026-09-25; it ends when none is queued in any workspace.
    """
    if answer is None:
        raise ValueError("alternative_evidence.submission_format_retired")
    return answer


__all__ = [
    "receipt_matter_selection_id",
    "require_answer_format",
    "require_current_selection",
]
