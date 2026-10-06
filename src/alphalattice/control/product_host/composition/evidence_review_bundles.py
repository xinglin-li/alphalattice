"""The agent bundles: each role's bundle, what it binds, and the refusals of a stale one.

The Analyst's bundle is one prepared packet's (its Task and unit), the CRO's the selected
book's current dossier; the Host keeps the directory beside the exact submission an
answer completes, so no file carries a hash. Split from the review application by
surface (card C, 2026-09-25): it reads the application; the application never reads it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, get_args
from uuid import UUID

from pydantic import ValidationError

from alphalattice.evidence.alternative_evidence.analysis.read_model import earlier_findings
from alphalattice.evidence.alternative_evidence.analysis.views import render_analyst_bundle
from alphalattice.evidence.alternative_evidence.contracts import AlternativeEvidenceRequest
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidencePublicationError,
)
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
    PortfolioEvidenceReviewError,
)
from alphalattice.oversight.chief_risk_officer.decision.views import (
    EarlierReview,
    render_review_bundle,
)
from alphalattice.protocols.actor_execution.answers import (
    SPECIALIST_REFERENCE_CHARACTERS,
    SPECIALIST_REFERENCE_COUNT,
)
from alphalattice.protocols.actor_execution.bundles import (
    BUNDLE_INDEX,
    PACKING_RULE,
    READING_RULE,
    AgentBundle,
    AgentBundleRecord,
    AgentRole,
    BundleBlock,
    BundleSection,
    bundle_directory_key,
    bundle_slot,
    pack_sections,
)

from .evidence_review_application import BUNDLE_CATEGORY, ReviewOutcome

if TYPE_CHECKING:
    from .evidence_review_application import EvidenceReviewApplication

TASK_PROCEDURES: dict[str, str] = {
    "ANALYST": "skills/alternative-evidence/governed-evidence-analysis/SKILL.md",
    "CRO": "skills/chief-risk-officer/review-portfolio-evidence/SKILL.md",
}
"""What each role's bundle opens with: the role's task procedure, the one text every
agent that answers the role reads."""

ANSWER_FIELDS: dict[str, str] = {"ANALYST": "analysis_answer", "CRO": "review_answer"}
"""Which submission field an answer to each role's bundle completes."""

STALE_BUNDLE_CODES = frozenset(
    {
        "alternative_evidence.external_analysis_binding_changed",
        "alternative_evidence.prepared_book_scope_mismatch",
        "chief_risk_officer.external_review_binding_changed",
    }
)
"""A bundle whose binding no longer holds: the evidence or the book moved on."""

_BUNDLE_REFUSALS = {
    "agent_bundle.not_prepared": (
        "This directory is not a bundle this workspace prepared. Nothing was submitted; "
        "tell your lead."
    ),
    "agent_bundle.preparation_stale": (
        "The evidence or the book changed after this bundle was prepared, so it no longer "
        "matches what the Host holds. Nothing was submitted; tell your lead, who prepares "
        "a new bundle."
    ),
    "agent_bundle.analyst_task_required": (
        "An Analyst's bundle is one prepared packet's, so it takes the preparation's `task_id` "
        "beside the book, and a coverage run's `evidence_unit_id`, as the preparation's "
        "`packet_<unit>` request writes them. Nothing was prepared."
    ),
    "agent_bundle.analyst_unit_required": (
        "This preparation is a coverage run, whose packets are one unit's each: give the "
        "`evidence_unit_id` of a unit named after the colon, as its `packet_<unit>` request "
        "writes it. Nothing was prepared."
    ),
    "agent_bundle.answer_settled": (
        "This bundle's corrections ran out: the Host kept the acceptable part of the last "
        "answer they allowed and reads no other. Nothing was submitted; return your receipt to "
        "your lead (the same answer sent again reads it)."
    ),
    "agent_bundle.seal_bound_exceeded": (
        "The answer is within its format, but the review it would seal exceeds a bound of the "
        "record at the places named under 'bounds', so nothing was submitted. This is the "
        "Host's limit, not the answer's: tell your lead, who reports it with these bounds."
    ),
    "agent_bundle.specialist_task_required": (
        "This specialist's bundle takes the exact assigned Task's `task_id`. Nothing was "
        "prepared; give that Task id to your lead."
    ),
    "agent_bundle.reference_bound_exceeded": (
        "This Task holds more references than one specialist bundle can carry. Nothing was "
        "prepared; tell your lead to select a smaller assigned Task."
    ),
    "agent_bundle.specialist_task_changed": (
        "The assigned Task changed after this bundle was prepared. Nothing was submitted; "
        "tell your lead to prepare a new bundle for the Task's current record."
    ),
}

_BOUND_ERRORS = frozenset({"too_long", "string_too_long"})


def agent_bundle_refusal(
    code: str,
    *,
    role: AgentRole | None = None,
    bounds: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """A submission to a bundle refused before any answer was read, its role when known."""
    way: dict[str, object]
    if code == "agent_bundle.seal_bound_exceeded":
        way = {"bounds": [dict(value) for value in bounds], "next_action": "TELL_YOUR_LEAD"}
    elif code == "agent_bundle.answer_settled":
        way = {"next_action": "RETURN_THE_RECEIPT_TO_YOUR_LEAD"}
    elif code.partition(":")[0] in PACKET_SELECTOR_CODES:
        way = {"next_action": "GIVE_THE_PACKETS_TASK_AND_UNIT"}  # V242
    else:
        way = {"next_action": "TELL_YOUR_LEAD"}
    return {
        "status": "REFUSED",
        **({} if role is None else {"agent_role": role}),
        "failure_code": code,
        "message": _BUNDLE_REFUSALS[code.partition(":")[0]],
        **way,
    }


PACKET_SELECTOR_CODES = frozenset(
    {"agent_bundle.analyst_task_required", "agent_bundle.analyst_unit_required"}
)
"""An Analyst bundle's request that lacks its packet's Task or unit, answered with its words."""


def sealed_bounds_exceeded(error: ValidationError) -> tuple[dict[str, object], ...]:
    """Name each bound a record to be sealed exceeds.

    Each place gives its path, how many it holds and how many the record allows; an error
    of any other kind stays the owner's refusal (V212).

    Args:
        error: The validation error raised while the record was sealed.

    Returns:
        One mapping per exceeded bound, or an empty tuple for any other error.
    """
    found: list[dict[str, object]] = []
    for item in error.errors():
        location = tuple(item.get("loc", ()))
        if not location or item.get("type") not in _BOUND_ERRORS:
            return ()
        context = item.get("ctx") or {}
        value = item.get("input")
        count = context.get("actual_length", len(value) if isinstance(value, str) else None)
        found.append(
            {
                "field": ".".join(str(part) for part in location),
                "count": count,
                "at_most": context.get("max_length"),
            }
        )
    return tuple(found)


class EvidenceReviewBundles:
    """Every agent bundle the section hands out, over the application it serves."""

    def __init__(self, app: EvidenceReviewApplication) -> None:
        """Bind bounded evidence deliveries to the retained review application.

        Args:
            app: Deterministic evidence review application owner.
        """
        self.app = app

    def prepare_agent_bundle(
        self,
        *,
        role: AgentRole,
        selector: BookSelector | None,
        directory: str,
        task_id: UUID | None = None,
        unit_id: str | None = None,
        dispatcher: LocalBackgroundDispatcher | None = None,
    ) -> dict[str, object] | ReviewOutcome:
        """Prepare one exact role-specific agent bundle and retained submission association.

        One bundle for one agent role, rendered by the view the installed
        agent reads, and what it binds kept here.

        The Analyst's is one prepared packet's (its Task and unit); the CRO's
        the selected book's current dossier. The caller writes the files to
        `directory`; the Host keeps that directory beside the exact submission
        an answer completes, so no file carries a hash and the answer needs
        none. One directory holds one bundle.
        """
        if role not in get_args(AgentRole.__value__):
            raise PortfolioEvidenceReviewError("agent_bundle.role_unknown")
        if not Path(directory).is_absolute():
            raise PortfolioEvidenceReviewError("agent_bundle.directory_not_absolute")
        if role not in TASK_PROCEDURES:
            return self.prepare_specialist_bundle(role=role, directory=directory, task_id=task_id)
        chosen = self.app._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        procedure = (self.app.playpen_root / TASK_PROCEDURES[role]).read_text(encoding="utf-8")
        if role == "ANALYST":
            if task_id is None:
                raise PortfolioEvidenceReviewError("agent_bundle.analyst_task_required")
            adapter = self.app.evidence_task_adapter
            run = (
                None
                if adapter is None or unit_id is not None
                else adapter.run_of(self.app.session.task_control_registry.task(task_id))
            )
            if run is not None:
                # Every book is a run: its packet is one unit's, named by the
                # unit the preparation's own packet requests carry.
                raise PortfolioEvidenceReviewError(
                    "agent_bundle.analyst_unit_required:"
                    + ",".join(unit.unit_id for unit in run.units)
                )
            # The bundle the lead waits on runs above the book's other units: their
            # model work waits at its next call while it is written (F2).
            with adapter.runtime.foreground() if adapter is not None else nullcontext():
                packet, context, *_ = self.app._book_packet(
                    chosen, task_id=task_id, unit_id=unit_id
                )
                bundle = render_analyst_bundle(
                    packet,
                    task_procedure=procedure,
                    earlier=self._earlier_findings(packet.request),
                )
            submission = self.app._analysis_submission(
                chosen,
                task_id=task_id,
                unit_id=unit_id,
                context_hash=str(context["analysis_context_hash"]),
            )
        else:
            # The dossier as read now, and that time sealed with the bundle: its answer
            # meets this dossier, never one read on a later clock (V255).
            read_at = self.app.clock()
            dossier = self.app._resolve_review_dossier(chosen, read_at=read_at)
            if isinstance(dossier, ReviewOutcome):
                return dossier
            carried = (
                None
                if dispatcher is None
                else self.app._carry_forward(dossier, dispatcher=dispatcher)
            )
            if carried is not None:
                # Nothing the CRO read has changed: its review carries forward, and there is
                # nothing to answer; the book it read and its Task go with it (V427), as a
                # submitted answer's do.
                return replace(
                    carried,
                    disposition="REVIEW_CARRIED_FORWARD",
                    next_requests={
                        **dict(carried.next_requests or {}),
                        **(
                            {"task": {"operation": "STATUS", "task_id": str(carried.task_id)}}
                            if carried.task_id is not None
                            else {}
                        ),
                        "book": {"operation": "EVIDENCE_CRO", **chosen.request_fields()},
                    },
                )
            bundle = render_review_bundle(
                dossier, task_procedure=procedure, last_review=self._last_review(chosen)
            )
            submission = self.app._review_submission(dossier, read_at=read_at)
        return self.store_bundle(
            role=role, directory=directory, bundle=bundle, submission=submission
        )

    def prepare_specialist_bundle(
        self, *, role: AgentRole, directory: str, task_id: UUID | None
    ) -> dict[str, object]:
        """Prepare one generic interpretation from exact retained Task metadata and references."""
        if task_id is None:
            raise PortfolioEvidenceReviewError("agent_bundle.specialist_task_required")
        registry = self.app.session.task_control_registry
        task = registry.task(task_id)
        projection = registry.safe_projection(task_id)
        references = [str(task_id), task.record_hash, task.input.input_hash]
        for reference in projection.artifact_refs:
            references.append(reference)
            digest = reference.rsplit(":", 1)[-1]
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                references.append(digest)
        allowed = tuple(dict.fromkeys(references))
        if len(allowed) > SPECIALIST_REFERENCE_COUNT or any(
            len(reference) > SPECIALIST_REFERENCE_CHARACTERS for reference in allowed
        ):
            raise PortfolioEvidenceReviewError("agent_bundle.reference_bound_exceeded")
        materials = pack_sections(
            (
                BundleSection(
                    key=str(task_id),
                    heading="The exact assigned Task record",
                    blocks=(
                        BundleBlock(
                            lines=(
                                f"Task id: {task_id}",
                                f"Task kind: {task.task_kind}",
                                f"Goal: {projection.goal_summary}",
                                f"Lifecycle: {projection.lifecycle}",
                                f"Verified stages: {projection.verified_stage_count}",
                                f"Total stages: {projection.total_stage_count}",
                                "Exact references allowed in the answer:",
                                *(f"- {reference}" for reference in allowed),
                            ),
                        ),
                    ),
                ),
            ),
            stem="task",
        )
        index = "\n".join(
            (
                f"# {role} specialist interpretation",
                "",
                "Read the listed product Task metadata and exact references. This bundle does "
                "not contain numerical diagnostics unless explicitly shown in its material; "
                "name absent evidence in your answer.",
                *(f"- {file.name}" for file in materials),
                "",
                READING_RULE,
                "",
                'Write one JSON answer: {"text": "your interpretation", "references": []}.',
                "Text is nonempty and at most 4000 characters; references are at most 64 exact "
                "strings listed in the material. The product checks format, size and binding, "
                "never scientific quality. You may add `read` with the listed files read whole.",
                "Write the final structured answer only to the nominated answer file. Return "
                "`written` to your lead; the lead submits it. Never submit the answer yourself.",
                "Keep the loaded role card's stage and network permissions: product CLI reads "
                "in ANALYZE/REVIEW, and only authorized EXECUTE outputs in its assigned out "
                "directory. This final answer file does not authorize other writes.",
                "",
                PACKING_RULE,
                "",
            )
        )
        bundle = AgentBundle(
            ((BUNDLE_INDEX, index), *((file.name, file.text) for file in materials))
        )
        return self.store_bundle(
            role=role,
            directory=directory,
            bundle=bundle,
            submission={
                "task_id": str(task_id),
                "task_record_hash": task.record_hash,
            },
            allowed_references=allowed,
        )

    def store_bundle(
        self,
        *,
        role: AgentRole,
        directory: str,
        bundle: AgentBundle,
        submission: dict[str, str],
        allowed_references: tuple[str, ...] = (),
    ) -> dict[str, object]:
        """Keep every role's existing write-once bundle association and public read contract."""
        key = bundle_directory_key(directory)
        record = AgentBundleRecord(
            role=role,
            bundle_directory=key,
            submission=submission,
            files=tuple(name for name, _text in bundle.files),
            allowed_references=allowed_references,
            record_hash=bundle_slot(key),
        )
        try:
            self.app.artifacts.publish(BUNDLE_CATEGORY, record.record_hash, record)
        except AlternativeEvidencePublicationError as error:
            if str(error) != "alternative_evidence.artifact_identity_reused":
                raise
            raise PortfolioEvidenceReviewError(
                "agent_bundle.directory_holds_another_bundle"
            ) from error
        return {
            "status": "AGENT_BUNDLE_READY",
            "agent_role": role,
            "bundle_directory": directory,
            # What a lead's assignment names, so the answer is credited to whom it assigned:
            # a key any message carries, where the directory may be too long for one (V574).
            "bundle_reference": record.record_hash,
            **({"task_id": submission["task_id"]} if "task_id" in submission else {}),
            "index": BUNDLE_INDEX,
            "files": [{"name": name, "text": text} for name, text in bundle.files],
            "packing_rule": PACKING_RULE,
            # Its submit, the directory bound and the answer the one thing left (V406).
            "next_requests": {
                "submit": {
                    "operation": "AGENT_ANSWER_SUBMIT",
                    "bundle_directory": directory,
                    "agent_answer": None,
                }
            },
        }

    def _last_review(self, chosen: BookSelector) -> EarlierReview | None:
        """The book's last review, as it reads back and under which CRO policy (V256, OP10):
        the reviewer weighs an earlier policy's review as that policy's. Only the newest day
        holding a review of the book is read, so a day's reads do not grow with the days the
        workspace holds (Z2); older reviews stay readable by their exports."""

        view = self.app.published_review_for_book(self.app.resolve_book(chosen).book)
        if view is None:
            return None
        return EarlierReview(
            published_on=view.publication.published_at.date(),
            evidence_as_of=view.dossier.evidence_as_of.date(),
            under_installed_policy=view.under_installed_policy,
        )

    def _earlier_findings(self, request: AlternativeEvidenceRequest) -> tuple[tuple[str, str], ...]:
        """One line per finding an earlier reading made in a filing this request
        passes over, by issuer: what the Analyst is told was read before."""

        service = self.app.evidence_publications
        issues = self.app.review_publications.open_issues(
            frozenset(request.ordered_entity_ids), as_of=request.evidence_as_of
        )
        opened = tuple(
            (
                entity,
                f"{entity} (open issue from {state.raised_on.isoformat()}, last assessed "
                f"{state.assessed_on.isoformat()}): {state.issue.causal_channel} Say whether a "
                "new filing here changes or resolves it.",
            )
            for state in issues
            for entity in state.issue.affected_entities
            if entity in request.ordered_entity_ids
        )
        if service is None or not request.read_filings:
            return opened
        now = self.app.clock()
        grouped: dict[str, set[tuple[str, str]]] = {}
        for value in request.read_filings:
            grouped.setdefault(value.publication_hash, set()).add(
                (value.entity_id, value.accession)
            )
        readings = []
        for publication_hash in sorted(grouped):
            record = self.app.artifacts.load(
                "analysis-publications", publication_hash, AlternativeEvidenceAnalysisPublication
            )
            if service.standing(record, now=now) is not None:
                view = service.replay(publication_hash, now=now)
                readings.append((view, frozenset(grouped[publication_hash])))
        return (*opened, *earlier_findings(tuple(readings)))

    def agent_bundle(self, directory: str) -> AgentBundleRecord | None:
        """The bundle the Host prepared in this directory, if it prepared one."""
        slot = bundle_slot(directory)
        if not self.app.artifacts.exists(BUNDLE_CATEGORY, slot):
            return None
        return self.app.artifacts.load(BUNDLE_CATEGORY, slot, AgentBundleRecord)
