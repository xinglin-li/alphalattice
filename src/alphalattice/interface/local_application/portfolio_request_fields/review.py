"""Evidence and review request fields and their public descriptions."""

from typing import Any
from uuid import UUID

from pydantic import ConfigDict, Field

from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    PortfolioReviewAnswer,
)

from .contracts import RequestFieldModel


class RequestFields(RequestFieldModel):
    """Evidence and review fields for the operations that admit them."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    portfolio_session: str | None = None
    """The session a dated view reads, as YYYY-MM-DD; for a book, names the book with the other
    book fields, as a book's offered requests fill them."""
    result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A Portfolio result, by its hash; for a book, names the book with the other book fields, as
    a book's offered requests fill them."""
    handoff_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The research handoff a book comes from. Names the book with the other book fields, as a
    book's offered requests fill them."""
    update_task_id: UUID | None = None
    """The Portfolio update Task a book comes from. Names the book with the other book fields, as
    a book's offered requests fill them."""
    experiment_task_id: UUID | None = None
    """The Portfolio study Task a book comes from. Names the book with the other book fields, as
    a book's offered requests fill them."""
    experiment_receipt_hash: str | None = None
    """The Portfolio study's result, by its receipt hash. Names the book with the other book
    fields, as a book's offered requests fill them."""
    update_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Portfolio update's publication a book comes from. Names the book with the other book
    fields, as a book's offered requests fill them."""
    position_basis: str | None = None
    """The positions a book's update published them on. Names the book with the other book
    fields, as a book's offered requests fill them."""
    review_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A published CRO review, by its hash."""
    analysis_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A published Evidence analysis, by its hash."""
    review_dossier_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The CRO dossier a review answers, by its hash."""
    review_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The review policy the CRO answers under, as the dossier named it."""
    review_schema_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The review answer's schema, as the dossier named it."""
    review_read_at: str | None = Field(default=None, min_length=1, max_length=64)
    """When the review's dossier was read, as its bundle or first part sealed it."""
    review_answer: PortfolioReviewAnswer | dict[str, Any] | None = None
    """The schema is the answer's, so capabilities describe it; an answer the
    schema would refuse is still carried, as written, for the Host to name
    each problem by item."""
    analysis_context_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Evidence packet's context an analysis answers, by its hash."""
    analysis_answer: AlternativeEvidenceAnalystAnswer | dict[str, Any] | None = None
    """The Analyst's answer to an Evidence packet."""
    evidence_as_of: str | None = Field(default=None, min_length=1, max_length=64)
    """The time Evidence is prepared as of, in ISO 8601."""
    preparation_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Evidence preparation's binding, by its hash."""
    evidence_unit_id: str | None = Field(default=None, pattern=r"^u[0-9]{2,3}$")
    """One Evidence unit, by its id (u01, u02 ...)."""
    delivery_part: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which part of an answer delivered in parts to read, from 1."""
    delivery_budget_bytes: int | None = Field(default=None, ge=1, strict=True)
    """The most bytes one delivered part may hold."""
    evidence_detail: str | None = Field(
        default=None,
        pattern=r"^(litigation_inventory|session_windows|topic_coverage|time_view)$",
    )
    """Which detail of the packet to read: `litigation_inventory`, `session_windows`,
    `topic_coverage` or `time_view`; `EVIDENCE_CRO` also takes `time_view` for the
    published reading."""
    view_entity_id: str | None = Field(default=None, min_length=1, max_length=32)
    """Only this entity's Evidence."""
    view_topic: str | None = Field(default=None, pattern=r"^[A-Z_]{1,40}$")
    """Only this topic's Evidence."""
    view_last_days: int | None = Field(default=None, ge=1, le=3660, strict=True)
    """Only the last this many days."""
    view_from: str | None = Field(default=None, min_length=1, max_length=64)
    """The first day read, as YYYY-MM-DD."""
    view_to: str | None = Field(default=None, min_length=1, max_length=64)
    """The last day read, as YYYY-MM-DD."""
    ledger_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the Evidence ledger to read, from 1."""
    documents_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the workspace's retained documents to read, from 1."""
    citation_entity_id: str | None = Field(default=None, min_length=1, max_length=32)
    """The entity whose citation to open."""
    citation_unit_id: str | None = Field(default=None, pattern=r"^u[0-9]{2,3}$")
    """The Evidence unit the citation belongs to."""
    citation_page: int | None = Field(default=None, ge=1, le=100_000, strict=True)
    """Which page of the cited source to read, from 1."""
    continuation_of: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The prepared packet a continuation reads on from, by the receipt it sealed."""
    continuation_spans: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The packet's sealed span set, by its hash."""
    session_limit: int | None = Field(default=None, ge=1, le=99, strict=True)
    """The most reading sessions the continued chain may hold."""
    window_limit: int | None = Field(default=None, ge=1, le=4096, strict=True)
    """The most windows the continued chain may read."""
    finding_handle: str | None = Field(default=None, pattern=r"^FIND-[A-Z0-9-]{1,80}$")
    """A CRO finding, by its handle (FIND-...)."""
    prior_review_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """An earlier CRO review the export compares with."""
    evidence_documents_per_issuer: int | None = Field(default=None, strict=True)
    """The SEC filings official acquisition may take per issuer; the default budget is three."""
    evidence_total_documents: int | None = Field(default=None, strict=True)
    """The SEC filings official acquisition may take in all; the default budget's when absent."""
    evidence_total_bytes: int | None = Field(default=None, strict=True)
    """The bytes official acquisition may read in all; the default budget's when absent."""
    evidence_setup: str | None = Field(default=None, min_length=1, max_length=4096)
    """The Evidence setup's options, as its script takes them, for the running Host to install
    (`--acquire-sec --entities AAPL --network-consent --install`)."""
