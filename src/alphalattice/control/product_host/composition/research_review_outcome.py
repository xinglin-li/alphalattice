"""The retained evidence review outcome shape."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.evidence_review_application import (
        ReviewOutcome,
    )


def _review_outcome_body(outcome: ReviewOutcome | dict[str, object]) -> dict[str, object]:
    """One JSON shape for both evidence actions, including their refusals.

    `task_id` is present exactly when a Task was admitted, so a reader can tell
    an admitted refresh from a reuse or a refusal without parsing the detail.
    """

    if isinstance(outcome, dict):
        return outcome
    body: dict[str, object] = {"disposition": outcome.disposition, "detail": outcome.detail}
    if outcome.task_id is not None:
        body["task_id"] = str(outcome.task_id)
    if outcome.lifecycle is not None:
        body["lifecycle"] = outcome.lifecycle
    if outcome.review is not None:
        body["review_key"] = outcome.review.publication.review_key
        body["review_publication_hash"] = outcome.review.publication.publication_hash
    if outcome.evidence_as_of is not None:
        body["evidence_as_of"] = outcome.evidence_as_of.isoformat()
    if outcome.failure_code is not None:
        body["failure_code"] = outcome.failure_code
    if outcome.next_requests:
        body["next_requests"] = dict(outcome.next_requests)
    if outcome.evidence_unit_id is not None:
        body["evidence_unit_id"] = outcome.evidence_unit_id
    if outcome.answer is not None:
        body["answer"] = dict(outcome.answer)
    if outcome.network_access is not None:
        body["network_access"] = dict(outcome.network_access)
    if outcome.source_network_access is not None:
        body["source_network_access"] = dict(outcome.source_network_access)
    if outcome.source_ways is not None:
        body["source_ways"] = dict(outcome.source_ways)
    return body
