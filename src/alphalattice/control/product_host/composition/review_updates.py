"""The exact update a review reads, verified once while it is sealed."""

from __future__ import annotations

from functools import partial
from typing import Any, cast
from uuid import UUID

from alphalattice.control.product_host.composition import committee
from alphalattice.control.product_host.composition.decision_advancement import (
    SCHEMA as DECISION_ADVANCEMENT_SCHEMA,
)
from alphalattice.control.product_host.composition.decision_advancement import (
    DecisionAdvancementApplication,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    TASK_KIND as PORTFOLIO_UPDATE_TASK_KIND,
)
from alphalattice.control.product_host.composition.portfolio_updates import (
    PortfolioUpdateApplication,
)
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioUpdatePublication,
)


def read_review_update(
    operations: Any,
    task_id: UUID,
    publication_hash: str,
    *,
    readback: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read the exact review origin, reusing a matching result verified in this request.

    A sealed (SUCCEEDED) update is verified once for every reader: each Analyst submit and
    bundle reads it, and verifying it each time cost about 165 s apiece with 15 at once (11-04).

    Args:
        operations: The Host's operations owner: its updates, Task registry, goals and clock.
        task_id: The update's Task.
        publication_hash: The exact publication the review reads.
        readback: A readback this request already verified, reused when it matches.

    Returns:
        The update's readback with the committee context its published floor carries.

    Raises:
        ValueError: The Task is not an update, its reader is not configured, or a displayed
            parent's origin is not exactly one producer.
    """
    assert operations.updates is not None
    task = operations.workspace_session.task_control_registry.task(task_id)
    reader: DecisionAdvancementApplication | PortfolioUpdateApplication
    if task.input.input_schema_id == DECISION_ADVANCEMENT_SCHEMA:
        if operations.research_updates is None:
            raise ValueError("product_host.evidence_review_update_not_configured")
        reader = operations.research_updates
    elif task.task_kind == PORTFOLIO_UPDATE_TASK_KIND:
        reader = operations.updates
    else:
        raise ValueError("product_host.evidence_review_update_task_mismatch")
    publication = None if readback is None else readback.get("publication")
    if (
        readback is not None
        and readback.get("task_id") == str(task_id)
        and isinstance(publication, dict)
        and publication.get("content_hash") == publication_hash
    ):
        body = dict(readback)
    elif task.lifecycle is TaskLifecycle.SUCCEEDED:
        key = (type(reader).__name__, str(task_id), task.record_hash, publication_hash)
        verify = partial(reader.readback, task_id, publication_hash=publication_hash)
        body = operations.readbacks.read(key, verify)
    else:
        body = reader.readback(task_id, publication_hash=publication_hash)
    plan = cast(dict[str, object], task.input.payload["plan"])
    if publication_hash == plan.get("parent_hash"):
        value = PortfolioUpdatePublication.model_validate(body["publication"])
        producers = [operations.updates.publication_task(value)]
        if operations.research_updates is not None:
            producers.append(operations.research_updates.publication_task(publication_hash))
        found = [v for v in producers if v is not None]
        if len(found) != 1:
            raise ValueError("product_host.evidence_review_publication_origin_unavailable")
        # A displayed parent keeps its original review identity and exact reuse.
        return read_review_update(operations, found[0], publication_hash)
    floor = committee.published_floor(operations.goals.store, str(task_id), publication_hash)
    committee.contextualize(body, floor, operations.dispatcher.clock())
    return body
