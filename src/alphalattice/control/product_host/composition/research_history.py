"""Bounded discovery from durable metadata; selected owners still verify their evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.decision_advancement import (
    SCHEMA as UPDATE_SCHEMA,
)
from alphalattice.control.product_host.composition.decision_advancement import (
    STAGES as UPDATE_STAGES,
)
from alphalattice.control.product_host.composition.decision_advancement import (
    DecisionAdvancementApplication,
)
from alphalattice.control.product_host.composition.evidence_review_application import (
    EvidenceReviewApplication,
)
from alphalattice.control.product_host.composition.evidence_review_projection import (
    EvidenceCroProjector,
)
from alphalattice.control.product_host.composition.plain_refusals import task_record_refusal
from alphalattice.control.product_host.composition.portfolio_updates import (
    PortfolioUpdateApplication,
)
from alphalattice.control.product_host.composition.research_experiments import (
    TASK_KIND as EXPERIMENT_TASK_KIND,
)
from alphalattice.control.product_host.composition.research_experiments import (
    ResearchExperimentApplication,
    review_requests,
)
from alphalattice.control.product_host.publication.portfolio_research import (
    PortfolioResearchPipelineStore,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskRecord
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioUpdatePublication,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
    portfolio_research_task_input,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import BookSelector
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewPublication,
)


@dataclass(frozen=True)
class HistoryEntry:
    """Retain one metadata-only history row and its exact selected readback association."""

    entry_id: str
    kind: str
    recorded_at: datetime
    task_id: UUID | None
    status: str
    strategy_package_id: str | None = None
    input_id: str | None = None
    input_binding_hash: str | None = None
    book: BookSelector | None = None
    review_publication_hash: str | None = None
    failure_code: str | None = None

    def body(self) -> dict[str, Any]:
        """Serialize history metadata with an explicit timestamp and non-null book selector.

        Returns:
            JSON-compatible history row; no descendant verification is performed.
        """
        value: dict[str, Any] = json.loads(json.dumps(asdict(self), default=str))
        value["recorded_at"] = self.recorded_at.isoformat()
        if value["book"] is not None:
            value["book"] = {k: v for k, v in value["book"].items() if v is not None}
            assert self.book is not None
            value["book_summary"] = EvidenceCroProjector.book_summary(self.book)
        value["next_requests"] = self.requests()
        return value

    def requests(self) -> dict[str, dict[str, Any]]:
        """What a reader of this row sends next, bound to it, so no locator is copied (V499).

        Its Task's read; for a row naming a book, that book's review (V483's requests); for a
        row naming a CRO review, its export.

        Returns:
            Each request ready to send, by name.
        """
        offered: dict[str, dict[str, Any]] = {}
        if self.task_id is not None:
            offered["task"] = {"operation": "STATUS", "task_id": str(self.task_id)}
        if self.book is not None:
            fields = self.book.request_fields()
            offered.update(review_requests(fields))
            if self.review_publication_hash is not None:
                offered["export"] = {
                    "operation": "EVIDENCE_CRO_EXPORT",
                    **fields,
                    "review_publication_hash": self.review_publication_hash,
                }
        return offered


class ResearchHistory:
    """A read-only join, not a new registry, cache or publication authority."""

    @staticmethod
    def latest_update(owner: DecisionAdvancementApplication, task: TaskRecord) -> dict[str, object]:
        """Read a completed update's final publication metadata without branch verification.

        Args:
            owner: The update owner holding the Task's stage receipts.
            task: The selected completed update Task.

        Returns:
            Its recorded date/claim and exact Evidence request, or a named metadata refusal.
            Opening the update or Evidence still verifies the whole branch.
        """
        try:
            plan = owner._plan_of(task, current=False)
            identities = owner._need(plan, UPDATE_STAGES[5]).products
            value = owner._load("decision-candidates", identities[-1], PortfolioUpdatePublication)
            entry = ResearchHistory._update_entry(task.task_id, plan.package_id, value)
            assert entry.book is not None
            return {
                "verification": "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS",
                "target_session": plan.target.isoformat(),
                "claim": value.claim,
                "review_selector": entry.book.request_fields(),
            }
        except (ValueError, OSError, IndexError, KeyError, TypeError) as error:
            return {
                "status": "REFUSED",
                "failure_code": public_failure(error, "research_history.entry_unreadable"),
                "detail": (
                    "Research History could not verify the recorded metadata "
                    f"for Task {task.task_id}. "
                    "Read that Task's status before treating "
                    "its result or publication as absent."
                ),
                "next_requests": {"task": {"operation": "STATUS", "task_id": str(task.task_id)}},
            }

    def __init__(
        self,
        session: WorkspaceApplicationSession,
        experiments: ResearchExperimentApplication,
        pipeline: PortfolioResearchPipelineStore | None,
        updates: PortfolioUpdateApplication | None,
        research_updates: DecisionAdvancementApplication | None,
        review: EvidenceReviewApplication | None,
    ) -> None:
        """Compose retained task, experiment, pipeline, update and review history readers.

        Args:
            session: Retained task/workspace session.
            experiments: Exact experiment reader.
            pipeline: Optional portfolio pipeline store.
            updates: Optional direct decision update owner.
            research_updates: Optional continuous research update owner.
            review: Optional historical CRO review reader.
        """
        self.session, self.experiments, self.pipeline = session, experiments, pipeline
        self.updates, self.research_updates, self.review = updates, research_updates, review

    def listing(
        self,
        *,
        strategy: str | None = None,
        input_id: str | None = None,
        input_hash: str | None = None,
        kind: str | None = None,
        limit: int = 25,
        cursor: str | None = None,
        entry_id: str | None = None,
        installed_packages: tuple[str, ...] = (),
    ) -> dict[str, object]:
        """Read bounded metadata history with exact filter-bound pagination.

        Args:
            strategy: Optional strategy package filter.
            input_id: Optional research input filter.
            input_hash: Optional exact input revision filter.
            kind: Optional history kind filter.
            limit: Integer page size from 1 through 50.
            cursor: Optional continuation bound to the same filters.
            entry_id: Optional exact entry, exclusive with pagination cursor.
            installed_packages: Explicit known package identities.

        Returns:
            Ordered bounded entries, next cursor and named unreadable rows; selected descendant
            readback verifies separately.

        Raises:
            ValueError: Limit, strategy, exact entry or filter-bound cursor is invalid.
        """
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("research_history.limit_outside_1_50")
        rows: dict[str, HistoryEntry] = {}
        task_batch = self.session.task_control_registry.record_collection()
        tasks = task_batch.records
        failures: list[dict[str, Any]] = [
            {
                **task_record_refusal(task_id),
                "entry_id": f"task:{task_id}",
                "record_id": f"task:{task_id}",
            }
            for task_id in task_batch.refused_task_ids
        ]
        task_scopes: dict[UUID, tuple[str | None, str | None, str | None]] = {}
        if self.pipeline is None:
            manifests = {}
        else:
            readable_manifests, unreadable_manifests = self.pipeline.manifest_collection()
            manifests = {item.task_id: item for item in readable_manifests}
            for issue in unreadable_manifests:
                identifier = issue.get("result_hash") or issue["index_file"]
                requests: dict[str, dict[str, Any]] = {
                    "workspace": {"operation": "WORKSPACE_SHOW"},
                    "backups": {"operation": "WORKSPACE_BACKUPS"},
                }
                if result_hash := issue.get("result_hash"):
                    requests["report"] = {"operation": "REPORT", "result_hash": result_hash}
                failed_entry_id = issue.get("entry_id", f"result-index:{identifier}")
                failures.append(
                    {
                        **issue,
                        "entry_id": failed_entry_id,
                        "record_id": identifier,
                        "kind": "INSTALLED_RESULT",
                        "detail": (
                            f"The saved result record {identifier} could not be verified. Its "
                            "originating Task and whether the result can be read remain unknown. "
                            "Read `workspace show` and `backup list`; if a listed verified "
                            "generation holds it, restore that generation into a new directory "
                            "with `alphalattice backup restore --dir <new directory> "
                            "--generation <verified generation hash> --workspace-id "
                            "<workspace id> --root <backup root>`, then read it there."
                        ),
                        "next_requests": requests,
                    }
                )
        for task in tasks:
            try:
                produced: list[HistoryEntry] = []
                if task.task_kind == EXPERIMENT_TASK_KIND:
                    plan, _actor = self.experiments._of(task)
                    book = None
                    if task.lifecycle is TaskLifecycle.SUCCEEDED and plan.portfolio_source:
                        evidence = self.experiments._stored_evidence(plan)
                        book = BookSelector(
                            experiment_task_id=task.task_id,
                            experiment_receipt_hash=evidence.artifact_uris[0]
                            .rsplit("/", 1)[-1]
                            .removesuffix(".json"),
                            portfolio_session=evidence.formation_sessions[-1].isoformat(),
                        )
                    produced.append(
                        HistoryEntry(
                            f"experiment:{task.task_id}",
                            plan.program.kind,
                            task.admitted_at,
                            task.task_id,
                            task.lifecycle.value,
                            input_id=plan.binding.input_id,
                            input_binding_hash=plan.binding.binding_hash,
                            book=book,
                            failure_code=task.failure_code,
                        )
                    )
                elif task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND:
                    request = portfolio_research_task_input(task)
                    # A Task that reused an earlier result names it too (V189).
                    manifest = manifests.get(task.task_id) or (
                        None if self.pipeline is None else self.pipeline.find_for_task(task.task_id)
                    )
                    produced.append(
                        HistoryEntry(
                            f"result:{manifest.result_hash}"
                            if manifest
                            else f"task:{task.task_id}",
                            "INSTALLED_RESULT",
                            task.admitted_at,
                            task.task_id,
                            task.lifecycle.value,
                            strategy_package_id=request.selected_strategy_package_id,
                            book=None
                            if manifest is None
                            else BookSelector(result_hash=manifest.result_hash),
                            failure_code=task.failure_code,
                        )
                    )
                elif task.input.input_schema_id == UPDATE_SCHEMA and self.research_updates:
                    owner = self.research_updates
                    advance_plan = owner._plan_of(task, current=False)
                    if task.lifecycle is TaskLifecycle.SUCCEEDED:
                        for identity in owner._need(advance_plan, UPDATE_STAGES[5]).products:
                            publication = owner._load(
                                "decision-candidates", identity, PortfolioUpdatePublication
                            )
                            produced.append(
                                self._update_entry(
                                    task.task_id,
                                    advance_plan.package_id,
                                    publication,
                                )
                            )
                    else:
                        produced.append(
                            HistoryEntry(
                                f"task:{task.task_id}",
                                "CONTINUOUS_UPDATE",
                                task.admitted_at,
                                task.task_id,
                                task.lifecycle.value,
                                strategy_package_id=advance_plan.package_id,
                                failure_code=task.failure_code,
                            )
                        )
                elif task.input.input_schema_id == "portfolio-decision-update" and self.updates:
                    update_plan = self.updates._plan_of(task, current=False)
                    single_publication = (
                        self.updates._publication(update_plan)
                        if task.lifecycle is TaskLifecycle.SUCCEEDED
                        else None
                    )
                    if single_publication:
                        produced.append(
                            self._update_entry(
                                task.task_id,
                                update_plan.strategy_package_id,
                                single_publication,
                            )
                        )
                    else:
                        produced.append(
                            HistoryEntry(
                                f"task:{task.task_id}",
                                "CONTINUOUS_UPDATE",
                                task.admitted_at,
                                task.task_id,
                                task.lifecycle.value,
                                strategy_package_id=update_plan.strategy_package_id,
                                failure_code=task.failure_code,
                            )
                        )
                for entry in produced:
                    prior = rows.get(entry.entry_id)
                    if prior is not None and prior != entry:
                        raise ValueError("research_history.publication_producer_ambiguous")
                    rows[entry.entry_id] = entry
                    task_scopes[task.task_id] = (
                        entry.strategy_package_id,
                        entry.input_id,
                        entry.input_binding_hash,
                    )
            except (ValueError, KeyError, OSError) as error:
                task_id_text = str(task.task_id)
                failures.append(
                    {
                        "entry_id": f"task:{task_id_text}",
                        "record_id": f"task:{task_id_text}",
                        "kind": "INSTALLED_RESULT"
                        if task.task_kind == PORTFOLIO_PUBLIC_TASK_KIND
                        else task.task_kind,
                        "status": "REFUSED",
                        "task_id": task_id_text,
                        "failure_code": public_failure(error, "research_history.entry_unreadable"),
                        "detail": (
                            f"Research History could not verify the recorded metadata for Task "
                            f"{task_id_text}. Read that Task's status before treating its result "
                            "or publication as absent."
                        ),
                        "next_requests": {"task": {"operation": "STATUS", "task_id": task_id_text}},
                    }
                )
        if self.review is not None:
            publications, unreadable_reviews = EvidenceCroProjector.review_metadata(self.review)
            failures.extend(
                {
                    "entry_id": f"review:{identity}",
                    "record_id": identity,
                    "kind": "CRO_REVIEW",
                    "status": "REFUSED",
                    "failure_code": code,
                    "detail": (
                        f"The saved review metadata {identity} could not be read. "
                        "Read `alphalattice workspace show` and `alphalattice backup list`; "
                        "restore a listed verified generation into a new directory before "
                        "reading that review there."
                    ),
                    "next_requests": {
                        "workspace": {"operation": "WORKSPACE_SHOW"},
                        "backups": {"operation": "WORKSPACE_BACKUPS"},
                    },
                }
                for identity, code in unreadable_reviews
            )
            for publication in publications:
                book = self._review_book(publication)
                subject_task = (
                    None if book is None else book.experiment_task_id or book.update_task_id
                )
                if publication.result_hash:
                    matching = next(
                        (
                            v
                            for v in rows.values()
                            if v.book and v.book.result_hash == publication.result_hash
                        ),
                        None,
                    )
                    subject_task = None if matching is None else matching.task_id
                scope = (
                    task_scopes.get(subject_task, (None, None, None))
                    if subject_task
                    else (None, None, None)
                )
                entry = HistoryEntry(
                    f"review:{publication.publication_hash}",
                    "CRO_REVIEW",
                    publication.published_at,
                    subject_task,
                    "HISTORICAL_REVIEW" if book else "READBACK_SELECTOR_UNAVAILABLE",
                    *scope,
                    book=book,
                    review_publication_hash=publication.publication_hash,
                )
                rows[entry.entry_id] = entry
        if (
            strategy is not None
            and strategy not in installed_packages
            and not any(v.strategy_package_id == strategy for v in rows.values())
        ):
            raise ValueError("research_history.strategy_unknown")
        selected = sorted(
            (
                v
                for v in rows.values()
                if (strategy is None or v.strategy_package_id == strategy)
                and (input_id is None or v.input_id == input_id)
                and (input_hash is None or v.input_binding_hash == input_hash)
                and (kind is None or v.kind == kind)
            ),
            key=lambda v: (v.recorded_at, v.entry_id),
            reverse=True,
        )
        query_hash = str(canonical_hash((strategy, input_id, input_hash, kind)))
        if entry_id is not None:
            selected = [v for v in selected if v.entry_id == entry_id]
            if not selected:
                unreadable = next((v for v in failures if v["entry_id"] == entry_id), None)
                if unreadable is not None:
                    return unreadable
                raise ValueError("research_history.entry_unavailable")
            if cursor is not None:
                raise ValueError("research_history.exact_selection_has_no_cursor")
        if cursor:
            prefix, separator, after = cursor.partition("/")
            if not separator or prefix != query_hash:
                raise ValueError("research_history.cursor_filter_mismatch")
            index = next((i for i, v in enumerate(selected) if v.entry_id == after), None)
            if index is None:
                raise ValueError("research_history.cursor_entry_unavailable")
            selected = selected[index + 1 :]
        page = selected[:limit]
        following = f"{query_hash}/{page[-1].entry_id}" if len(selected) > limit else None
        # One entry selected: its own requests are the answer's; a page offers the next (V499).
        offered: dict[str, dict[str, Any]] = {}
        if entry_id is not None and page:
            offered.update(page[0].requests())
        if following is not None:
            offered["next_page"] = {
                "operation": "RESEARCH_HISTORY",
                "history_cursor": following,
                "history_limit": limit,
                **{
                    name: value
                    for name, value in (
                        ("strategy_package_id", strategy),
                        ("research_input_id", input_id),
                        ("input_binding_hash", input_hash),
                        ("history_kind", kind),
                    )
                    if value is not None
                },
            }
        return {
            "status": "AVAILABLE" if not failures else "PARTIAL_METADATA_READBACK",
            "entries": [v.body() for v in page],
            "next_cursor": following,
            "next_requests": offered,
            "blocked_entries": failures,
            "metadata_task_count": len(tasks),
            "verification": "METADATA_ONLY_SELECTED_READBACK_VERIFIES_DESCENDANTS",
        }

    @staticmethod
    def _update_entry(
        task_id: UUID, package: str, value: PortfolioUpdatePublication
    ) -> HistoryEntry:
        # Each publication is dated by its own publication, never by its Task's
        # admission: a continuous Task publishes many, newest last (V190).
        return HistoryEntry(
            f"update:{value.content_hash}",
            "CONTINUOUS_UPDATE",
            value.published_at,
            task_id,
            "SUCCEEDED",
            strategy_package_id=package,
            book=BookSelector(
                update_task_id=task_id,
                update_publication_hash=value.content_hash,
                position_basis="CONDITIONAL_ESTIMATE"
                if value.pending_proposal
                else "OBSERVED_RESEARCH_ENTRY",
            ),
        )

    @staticmethod
    def _review_book(value: PortfolioReviewPublication) -> BookSelector | None:
        if value.experiment_subject:
            subject = value.experiment_subject
            return BookSelector(
                experiment_task_id=subject.experiment_task_id,
                experiment_receipt_hash=subject.experiment_receipt_hash,
                portfolio_session=subject.portfolio_session.isoformat(),
            )
        if value.update_subject:
            update = value.update_subject
            return BookSelector(
                update_task_id=update.update_task_id,
                update_publication_hash=update.update_publication_hash,
                position_basis=update.position_basis,
            )
        return BookSelector(result_hash=value.result_hash) if value.result_hash else None
