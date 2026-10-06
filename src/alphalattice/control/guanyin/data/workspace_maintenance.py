"""Guanyin G0 projection for ordinary pre-PM workspace maintenance."""

from __future__ import annotations


def maintenance_failure_detail(code: str) -> str:
    """Explain a maintenance stop by its stable code or governed fallback.

    Args:
        code: Stable maintenance failure code to explain.

    Returns:
        The curated explanation for a known code, or the governed-boundary
        fallback when the code has no installed explanation.
    """
    details = {
        "workspace_maintenance.data_engineer_diagnosis_deferred": (
            "Listings need a data decision before the update can continue. Open the "
            "data issues, choose a permitted option for each, then continue the Task."
        ),
        "data.truth_review_required": (
            "Listings need a data decision before the update can continue. Open the "
            "data issues, choose a permitted option for each, then continue the Task."
        ),
        "DATA_REMEDIATION_BUSINESS_VALIDATION_FAILED": (
            "A recorded data decision could not be applied and no stable reason was "
            "declared. The decision is kept; continue the Task to apply it again, or "
            "choose another option."
        ),
        "feature_input.resolution_policy_changed": (
            "The data-issue policy changed after this decision was recorded. Reopen "
            "the data issues and choose again under the current catalog."
        ),
        "feature_input.resolution_catalog_changed": (
            "The option catalog changed after this decision was recorded. Reopen the "
            "data issues and choose again under the current catalog."
        ),
        "feature_input.sector_evidence_missing": (
            "Sector evidence for the current membership is missing, so an exclusion "
            "cannot be projected. The next Feature build establishes it."
        ),
        "workspace_maintenance.derived_manifest_evidence_incomplete": (
            "The membership the data decision derived could not inherit its parent's "
            "audit evidence: a listing's evidence is missing, expired or changed since "
            "the audit. Run the update again to re-audit before the membership changes."
        ),
        "workspace_data_update.membership_changed_review_required": (
            "The data decision changed the research membership, which this update plan "
            "did not declare. The update itself is applied; plan again against the new "
            "membership to publish it."
        ),
        "workspace_maintenance.worker_lost": (
            "A prior maintenance worker disappeared. Its verified artifacts were preserved "
            "and the orphaned cycle was terminalized."
        ),
        "workspace_maintenance.cancelled_at_safe_checkpoint": (
            "The update Task was cancelled at a safe checkpoint. Listing work already "
            "completed is kept for the next plan; plan again to continue."
        ),
        "data.full_history_audit_approval_required": (
            "A bounded rolling audit found an ambiguity that would require a separately "
            "authorized full-history audit. No historical expansion was executed."
        ),
        "data.listing_updates_incomplete": (
            "One or more listing refresh units exhausted their bounded attempts."
        ),
        "workspace_maintenance.candidate_recheck_source_changed": (
            "The candidate source or its admission changed after this update was "
            "planned, so the planned candidate recheck no longer describes it. Plan "
            "the update again against the current source."
        ),
        "workspace_maintenance.candidate_data_recheck_incomplete": (
            "The bounded retry of failed candidates did not complete. Run the update "
            "again to continue it."
        ),
    }
    return details.get(
        code,
        "Workspace maintenance stopped at a governed boundary whose reason has no installed "
        "explanation. Read the Task's recovery view for the actions its owner permits.",
    )


__all__ = []
