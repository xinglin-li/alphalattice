"""The production route: one sealed book, refreshed evidence, one review.

This is the owner a non-test caller reaches. The operation owner asks it three
questions -- what is the state, refresh the evidence, review with CRO -- for one
selected book, and every answer is either a typed state or a real Task admitted
through the one dispatcher.

The order is the design:

1. open the selected sealed book (a development result, a frozen candidate or
   a validated handoff) and its report's own window-end book;
2. project the book over the admitted listing/ticker authority, keeping every
   row, and map it onto admitted issuers;
3. derive the evidence-only obligation from the resulting scope;
4. admit or reuse the separate Alternative Evidence Task;
5. reopen an exact **non-expired** publication with the current reader;
6. compile the dossier;
7. resolve exact CRO reuse from the review key **before** any actor exists;
8. otherwise admit the three-stage CRO Task;
9. reopen the published recommendation for the interface.

Nothing here adds a runner, a store, a report, a pointer or an Agent board.
"""

from __future__ import annotations

import math
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from pydantic import ValidationError

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskReplan,
    TaskSafeProjection,
    failure_code_from,
)
from alphalattice.control.task_control.registry import TaskNotFoundError
from alphalattice.control.task_control.runner import TaskControlRunner
from alphalattice.control.workspace_runtime.network_access import NetworkAccess, network_access
from alphalattice.evidence.alternative_evidence.analysis.contracts import (
    AlternativeEvidenceResearchObligation,
    AlternativeEvidenceRetrievalAccessReceipt,
)
from alphalattice.evidence.alternative_evidence.analysis.disclosures import (
    compare_typed_disclosures,
)
from alphalattice.evidence.alternative_evidence.analysis.packet import (
    AlternativeEvidencePacket,
    reading_remainders,
    routed_pending_work,
    span_aliases,
)
from alphalattice.evidence.alternative_evidence.analysis.submissions import (
    accepted_analyst_answer,
    screen_analyst_answer,
)
from alphalattice.evidence.alternative_evidence.contracts import (
    ADMITTED_DOCUMENT_CAPACITY,
    AlternativeEvidenceAdmission,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    SecIssuerRegistrySnapshot,
    matter_selection_retired,
)
from alphalattice.evidence.alternative_evidence.publication.analysis import (
    AlternativeEvidenceAnalysisPublicationService,
    AlternativeEvidenceAnalysisPublicationView,
)
from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
    verified_evidence_records,
)
from alphalattice.evidence.alternative_evidence.publication.contracts import (
    AlternativeEvidenceAnalysisPublication,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import (
    COVERAGE_RUN_PURPOSE,
    AlternativeEvidenceCoverageRun,
    LogicalSourceCounts,
    coverage_intent_hash,
    coverage_unit_ids,
    coverage_units,
    sources_short,
)
from alphalattice.evidence.alternative_evidence.runtime.policy import (
    RETIRED_SELECTION_EXPLANATION,
    AdmittedEvidencePolicy,
)
from alphalattice.evidence.alternative_evidence.runtime.reuse import (
    ANALYSIS_PUBLICATION_CATEGORY,
    RecordDays,
    analysis_records,
    analysis_records_newest_first,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    INPUT_SCHEMA_ID as EVIDENCE_INPUT_SCHEMA_ID,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    AlternativeEvidenceDocumentTaskAdapter,
    EvidenceContinuation,
    SubmittedEvidenceAnalysis,
    alternative_evidence_document_task_contract,
    coverage_run_task_contract,
)
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.dispatcher import (
    CommandAdmission,
    CommandSubmission,
    LocalBackgroundDispatcher,
)
from alphalattice.interface.local_application.evidence_cro import (
    AMBIGUOUS_EXPLANATION,
    REFRESH_EXPLANATION,
    EvidenceCroBook,
    EvidenceSelectionLabel,
    book_projection,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    ValidatedPortfolioHandoff,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    ENCODER_RUNTIME_TORCH_CUDA,
    RECIPE_MINILM_CPU,
    RERANKER_RUNTIME_TORCH_CUDA,
    HybridIndexSpec,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import command_prefix, resolve_playpen_root
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    BookSelector,
    CarriedReading,
    PortfolioEvidenceReviewError,
    PortfolioEvidenceReviewInputs,
    SealedBook,
    assessment_schema_hash,
    carried_reading_admitted,
    compile_portfolio_coverage_dossier,
    compile_portfolio_review_dossier,
    compile_portfolio_scope,
    finding_dispositions,
    open_sealed_book,
    portfolio_review_key_payload,
    project_coverage_run,
    project_unit_obligation,
    review_book_key,
    typed_disclosure_records,
)
from alphalattice.oversight.chief_risk_officer.decision.portfolio_review import (
    OpenIssueState,
    PortfolioReviewAnswer,
    PortfolioReviewDossier,
    PortfolioReviewPublication,
    PortfolioReviewReceipt,
    review_basis,
)
from alphalattice.oversight.chief_risk_officer.decision.submissions import (
    CROHostPolicyBinding,
    accepted_review_answer,
    build_portfolio_review_policy_binding,
    screen_review_answer,
    seal_portfolio_review,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    AdmittedEvidenceSelection,
    AdmittedListingTickerAuthority,
    PortfolioExperimentReviewSubject,
    PortfolioExposureProjection,
    PortfolioIssuerScope,
    PortfolioUpdateReviewSubject,
    book_key,
    seal_portfolio_evidence_contract,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.coverage.selections import (
    SELECTIONS_CATEGORY,
    recorded_selections,
)
from alphalattice.oversight.chief_risk_officer.publication.portfolio_review import (
    PortfolioReviewPublicationService,
    PortfolioReviewView,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    INPUT_SCHEMA_ID as CRO_INPUT_SCHEMA_ID,
)
from alphalattice.oversight.chief_risk_officer.runtime.portfolio_review_task import (
    CarriedPortfolioReviewActor,
    PortfolioReviewActor,
    PortfolioReviewTaskAdapter,
    PortfolioReviewTaskResources,
    SubmittedPortfolioReviewActor,
    portfolio_review_task_contract,
)
from alphalattice.protocols.actor_execution import ActorKind, seal_actor_submission
from alphalattice.protocols.actor_execution.answers import (
    ANSWER_CORRECTION_BOUND,
    AgentAnswerRecord,
    AgentRun,
    AnswerProblem,
    AnswerVerdict,
    ScreenedAnswer,
    answer_digest,
    answer_slot,
    answer_verdict,
    correction_text,
    screen_specialist_answer,
)
from alphalattice.protocols.actor_execution.bundles import AgentBundleRecord

EVIDENCE_REFRESH_COMMAND = "alternative_evidence.scoped_refresh"
CRO_REVIEW_COMMAND = "chief_risk_officer.portfolio_review"


def _authored(book: BookSelector | None) -> bool:
    """Whether a book is one a study authored, whose research input the setup names."""
    return book is not None and book.experiment_task_id is not None


SETUP_OPTIONS: dict[str, str] = {
    "accessions": (
        "up to four exact accession numbers of the one issuer --entities names, in the official "
        "##########-##-###### form its own submissions index holds"
    ),
    "entities": (
        "the issuers to acquire or import, by ticker, at most eight, each in the workspace's "
        "universe and, for an import, in the source's request"
    ),
    "evidence_as_of": (
        "a timezone-aware ISO 8601 cutoff no later than now, the filings selected being those the "
        "official source accepted by then; the run's clock unless given"
    ),
    "maximum_documents_per_issuer": (
        "from 3 to 20 filings each issuer may contribute, at most 24 across the issuers named; 3 "
        "unless given"
    ),
    "minimum_entity_coverage": (
        "the package's floor: the share, from 0 to 1, of its counted issuers that must hold a "
        "document; 0.60 unless given, the installed one on a rebind"
    ),
    "model_name": (
        "an optional managed analyst and reviewer model, named with its --deepseek-base-url, or "
        "neither for native analysis"
    ),
    "research_input_hash": (
        "that input's exact sealed binding, also named by `study show <id>`, never inferred from "
        "the latest data"
    ),
    "research_input_id": (
        "the admitted research input the book's study was built on, as `study show <id>` names it"
    ),
    "semantic_model": (
        "the retained recipe's pack directory inside the workspace, the installed one on a "
        "rebind; without it the recipe is bound from the application model store"
    ),
    "source_artifact_root": (
        "the alternative-evidence artifact store of the workspace that acquired the source, "
        "runtime/artifacts/alternative-evidence under that workspace, which holds "
        "source-document-sets/, snapshots/, registries/ and requests/"
    ),
    "source_knowledge_root": (
        "the evidence knowledge store of that workspace, runtime/evidence-knowledge under it, "
        "which holds the source objects a reference-form source set names; by default the one "
        "beside the artifact store's runtime/artifacts"
    ),
    "source_set_hash": (
        "the 64-character lowercase hexadecimal identity of a source set under that store, as "
        "the acquisition's answer names it"
    ),
}
"""What each option of the Evidence setup must hold, by its name in the answer: the one owner the
setup's offer states and its refusals name (V590, V591, V592)."""


def _holds(*options: str) -> str:
    """What the offered options must hold, each from `SETUP_OPTIONS` (V591)."""
    return (
        "; ".join(
            f"--{option.replace('_', '-')} names {SETUP_OPTIONS[option]}" for option in options
        )
        + "."
    )


def evidence_setup(
    workspace: Path, *, authored: bool = False, rebind: bool = False
) -> dict[str, object]:
    """The commands that complete the Evidence setup (V405, the UI line's R2-02).

    Each is written as a person types it at the product folder, its interpreter named. The
    retrieval pack first (the authority binds the recipe from the application model
    store), then the authority's check and installation in the declared retrieval environment,
    as the Evidence handoff's first-package steps run them. What the Host knows is filled in:
    the workspace, the recipe, the interpreters and whether the retrieval environment exists.
    What only a person decides stays open in ``choose``: the issuers, a recorded package in place
    of the SEC acquisition, and for a book a study authored the research input it was built on.
    A workspace installed under a retired matter selection is rebound (``rebind``).

    Args:
        workspace: The workspace's folder.
        authored: Whether the book is one a study authored.
        rebind: Whether the installed authority is rebound under the integrated selection.

    Returns:
        The setup's ``pack`` and ``authority`` steps.
    """
    windows = os.name == "nt"
    python = ".venv/Scripts/python.exe" if windows else ".venv/bin/python"
    retrieval = ".venv-retrieval/Scripts/python.exe" if windows else ".venv-retrieval/bin/python"
    script = f'{retrieval} scripts/materialize_evidence_cro_authority.py --workspace "{workspace}"'
    installed = not (resolve_playpen_root(Path(__file__)) / "pyproject.toml").is_file()
    if installed:
        from alphalattice.interface.local_application.cli_contract import join, shell

        python = join([sys.executable], shell())
        retrieval = python
        script = (
            f"{python} -m alphalattice.control.product_host.composition.evidence_authority_setup"
            f' --workspace "{workspace}"'
        )
    if rebind:
        return {
            "authority": {
                "install": f"{script} --rebind-installed --matter-selection "
                "INTEGRATED_TOPIC_ROUTING --install",
                "before": "Stop the Host first: the installation takes the workspace's lease.",
            }
        }
    pack: dict[str, object] = {
        "recipe": RECIPE_MINILM_CPU,
        "check": f"{retrieval} scripts/install_retrieval_pack.py --status",
        "install": f"{retrieval} scripts/install_retrieval_pack.py --install "
        f"{RECIPE_MINILM_CPU} --network",
    }
    if installed:
        entry = f"{python} -m alphalattice.control.product_host.composition.retrieval_pack_setup"
        pack["check"] = f"{entry} --status"
        pack["install"] = f"{entry} --install {RECIPE_MINILM_CPU} --network"
        pack["environment"] = (
            f"{python} -m alphalattice.interface.local_application.retrieval_environment --offline"
        )
    elif not (resolve_playpen_root(Path(__file__)) / retrieval).is_file():
        pack["environment"] = f"{python} scripts/create_retrieval_environment.py --offline"
    scope = " ".join(
        [
            *(["--research-input-id <id> --research-input-hash <hash>"] if authored else []),
            "--acquire-sec --entities <TICKER> ...",
        ]
    )
    # Each choice states what its options must hold, from the table its refusals name (V591).
    choose = [
        {"arg": "--entities", "why": _holds("entities")},
        {
            "arg": "--source-artifact-root <dir> --source-set-hash <hash>",
            "why": "In place of --acquire-sec and --network-consent, SEC artifacts already "
            "captured, read offline: " + _holds("source_artifact_root", "source_set_hash"),
        },
    ]
    if authored:
        choose.append(
            {
                "arg": "--research-input-id <id> --research-input-hash <hash>",
                "why": _holds("research_input_id", "research_input_hash"),
            }
        )
    return {
        "pack": pack,
        "authority": {
            "check": f"{script} {scope} --preflight",
            "install": f"{script} {scope} --network-consent --install",
            "choose": choose,
            "before": "Stop the Host first: the installation takes the workspace's lease. The "
            "acquisition reads your contact from SEC_USER_AGENT in its environment.",
        },
    }


PACKAGE_RULE = (
    "A recorded package covers at most eight issuers, one unit, and an install replaces the "
    "package before it: a packet prepared under that one goes stale, and units prepared under "
    "different packages cannot be reviewed together, so a book's whole review takes one package "
    "covering every unit's issuers, or official acquisition."
)
"""What a recorded package can review (V546, V547): said wherever a package is offered or met."""


def source_ways(
    workspace: Path, *, entities: tuple[str, ...] | None, book: BookSelector | None
) -> dict[str, object]:
    """The lawful ways on when the recorded package holds too few sources (V541, V546).

    Official SEC acquisition first, the person's decision: the Host served with their consent
    reads every holding's filing index at one cutoff, names those that filed nothing in the
    window and fetches the rest, so every unit stands at that cutoff and the book is reviewed
    whole. A recorded package is offered only where one covers every unit's issuers, a book of
    one unit (`PACKAGE_RULE`).

    Args:
        workspace: The workspace's folder.
        entities: Every issuer of the book, when one package covers them all; else None.
        book: The book, whose research input the package names when a study authored it.

    Returns:
        The ``official`` serve command with what it needs first, the package rule, and the
        ``package`` steps with the issuers filled where one package covers the book.
    """
    # The person's own restart, in their shell: it names the workspace on either leg (V568, V429).
    ways: dict[str, object] = {
        "official": {
            "serve": f'{" ".join(command_prefix())} --workspace "{workspace}" serve '
            "--sec-network-consent",
            "before": str(network_access(workspace).body()["detail"]) + " "
            "The person's decision: their consent to acquire SEC filings from the "
            "official endpoints, their contact in SEC_USER_AGENT and the workspace's network "
            "open. Restart only an idle Host you started; then preview and prepare again, which "
            "reads every unit at one cutoff.",
        },
        "package_rule": PACKAGE_RULE,
    }
    if entities is None:
        return ways
    setup = evidence_setup(workspace, authored=_authored(book))
    authority = cast(dict[str, Any], setup["authority"])
    named = "--entities " + " ".join(entities)
    ways["package"] = {
        "entities": list(entities),
        "check": str(authority["check"]).replace("--entities <TICKER> ...", named),
        "install": str(authority["install"]).replace("--entities <TICKER> ...", named),
        "choose": [value for value in authority["choose"] if value["arg"] != "--entities"],
        "before": authority["before"],
        "covers": "Every issuer of this book, in place of the installed package; an issuer that "
        "filed nothing in the window holds no document in a package.",
    }
    return ways


DOSSIER_STANDINGS = frozenset({"REFUSED_EVIDENCE_CUTOFFS_DIFFER", "REFUSED_DOSSIER_INVALID"})
"""A dossier the book's current analyses cannot make: a refusal at every read on the review
route, and at the book's readback its standing, never a failed read (V546)."""


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cutoffs_differ(
    chosen: BookSelector,
    children: list[
        tuple[
            str, AlternativeEvidenceResearchObligation, AlternativeEvidenceAnalysisPublicationView
        ]
    ],
    carried: tuple[CarriedReading, ...],
) -> ReviewOutcome | None:
    """The refusal of units read at different cutoffs, said before the dossier refuses (V546).

    A dossier reads as of its children's earliest cutoff, and every citation must be available
    by then: late evidence is never admitted as early. Units prepared at different cutoffs -- two
    recorded packages installed one after the other, each for its own unit -- can cite sources
    that became available after the earliest, and then cannot be reviewed together. Said with
    the cutoff, the units at it, how many citations of which units post-date it and the earliest
    of their dates, and the way on: one preparation at one cutoff (official acquisition).

    Args:
        chosen: The book.
        children: Each unit's id, obligation and current analysis.
        carried: The earlier readings the review carries, each at its own cutoff.

    Returns:
        The refusal; None when every unit's citations stand at or before the earliest cutoff.
    """
    cutoffs = {unit: evidence.lineage.request.evidence_as_of for unit, _q, evidence in children}
    cutoffs |= {f"c{index:02d}": value.as_of for index, value in enumerate(carried, start=1)}
    if not cutoffs:
        return None
    cutoff = min(cutoffs.values())
    late = {
        unit: [
            span.available_at
            for span in evidence.lineage.cro_package.verified_spans
            if span.available_at > cutoff
        ]
        for unit, _obligation, evidence in children
    }
    late = {unit: dates for unit, dates in late.items() if dates}
    if not late:
        return None
    count = sum(len(dates) for dates in late.values())
    earliest = min(date for dates in late.values() for date in dates)
    at_cutoff = sorted(unit for unit, value in cutoffs.items() if value == cutoff)
    citations = "citation" if count == 1 else "citations"
    units = "unit" if len(late) == 1 else "units"
    return ReviewOutcome(
        disposition="REFUSED_EVIDENCE_CUTOFFS_DIFFER",
        failure_code=(
            f"chief_risk_officer.dossier_cutoff_invalid:{count} {citations} of {len(late)} "
            f"{units} after the cutoff {_stamp(cutoff)}, the earliest {_stamp(earliest)}"
        ),
        detail=(
            "This book's current analyses were prepared at different cutoffs, and a review reads "
            f"the book as of one, the earliest, {_stamp(cutoff)} ({', '.join(at_cutoff)}): "
            f"{count} {citations} of {', '.join(sorted(late))} became available after it, the "
            f"earliest at {_stamp(earliest)}. Late evidence is never admitted as early, so these "
            "analyses cannot be reviewed together. Official SEC acquisition, with the person's "
            "consent, prepares every unit at one cutoff (the preview's `source_ways`). "
            + PACKAGE_RULE
        ),
        next_requests={"preview": {"operation": "EVIDENCE_PREVIEW", **chosen.request_fields()}},
    )


def _dossier_refused(chosen: BookSelector, error: ValidationError) -> ReviewOutcome:
    """Any other refusal of the dossier's own model, said with its code and a way on (V546).

    The checks of the dossier's consistency -- its children, entities, citations, issues,
    authority -- are the CRO record's; a refusal of one is not the evidence's fault, and a read
    that meets it answers in words rather than failing.
    """
    code = failure_code_from(error)
    return ReviewOutcome(
        disposition="REFUSED_DOSSIER_INVALID",
        failure_code=f"product_host.evidence_review_dossier_refused:{code}",
        detail=(
            f"The review's dossier did not hold together (`{code}`), a check of its own record "
            "rather than of the evidence, so nothing was read for review. Read the book's Evidence "
            "and prepare it again if a unit changed; if the same code returns, report it."
        ),
        next_requests={"book": {"operation": "EVIDENCE_CRO", **chosen.request_fields()}},
    )


class EvidenceSelectionAmbiguous(PortfolioEvidenceReviewError):
    """Two different current publications answer one obligation. Refused, not resolved."""


@dataclass(frozen=True, slots=True)
class ResolvedBookScope:
    """The deterministic half: no Task, no clock, no model."""

    book: SealedBook
    projection: PortfolioExposureProjection
    scope: PortfolioIssuerScope


@dataclass(frozen=True, slots=True)
class ResolvedEvidenceQuestion:
    """One unit's question: the obligation names the unit's issuers and cutoff."""

    resolved: ResolvedBookScope
    obligation: AlternativeEvidenceResearchObligation


@dataclass(frozen=True, slots=True)
class ResolvedCoverage:
    """The book's whole obligation at one cutoff: its units, sealed as one run.

    Every book is a run (C2, 2026-09-23). A book that fits one unit is a run
    of one unit whose request, obligation and intent are exactly those a
    single preparation of the same book carried, so a single preparation's
    completion is that unit's completion.
    """

    resolved: ResolvedBookScope
    run: AlternativeEvidenceCoverageRun
    prepare_only: bool


_QuestionRecord = tuple[ResolvedEvidenceQuestion, AlternativeEvidenceAnalysisPublication]


@dataclass(frozen=True, slots=True)
class _CurrentEvidenceSelection:
    disposition: str
    question: ResolvedEvidenceQuestion | None = None
    evidence: AlternativeEvidenceAnalysisPublicationView | None = None
    """The current analysis, or on EXPIRED / SUPERSEDED the newest such one so the
    state can say which analysis it is and why it is not current."""


_NO_BOOK_NAMED = (
    "No book is named, and this workspace has no default one (a handoff or a latest Portfolio "
    "result). A book is named by "
    "`experiment_task_id` with `experiment_receipt_hash` and `portfolio_session`, by "
    "`update_task_id`, by `handoff_hash` or by `result_hash`; a preparation's `packet_<unit>` "
    "request writes the whole selector, its `task_id` and `evidence_unit_id` beside the book, "
    "and its `analyst_bundle_<unit>` the Analyst's bundle request; an experiment's readback "
    "offers its `cro_bundle`. The `history` request lists the workspace's books."
)
_DOCUMENTS_PAGE = 50
"""Retained documents a page of `EVIDENCE_DOCUMENTS` lists (A6): the Local Web's table page."""

_BUNDLE_FIELDS = frozenset(
    {
        "result_hash",
        "handoff_hash",
        "update_task_id",
        "update_publication_hash",
        "position_basis",
        "experiment_task_id",
        "experiment_receipt_hash",
        "portfolio_session",
        "task_id",
        "evidence_unit_id",
    }
)


def with_analyst_bundles(requests: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Each packet request with the Analyst's bundle request beside it (V295).

    `packet` and `packet_<unit>` gain `analyst_bundle` and `analyst_bundle_<unit>`: the same
    book, Task and unit as `AGENT_BUNDLE_PREPARE` for the Analyst, the bundle's directory left
    to choose, so a lead never assembles the selector itself.

    Args:
        requests: The next requests an answer offers.

    Returns:
        The same requests with a bundle request beside each packet request.
    """
    found = dict(requests)
    for name, request in requests.items():
        if request.get("operation") == "EVIDENCE_PACKET" and (
            name == "packet" or name.startswith("packet_")
        ):
            found["analyst_bundle" + name.removeprefix("packet")] = {
                "operation": "AGENT_BUNDLE_PREPARE",
                "agent_role": "ANALYST",
                **{key: value for key, value in request.items() if key in _BUNDLE_FIELDS},
            }
    return found


def units_short_of_sources(
    inventory: Mapping[str, object], mode: AlternativeEvidenceMode, *, floor: float
) -> list[dict[str, object]]:
    """The units RECORDED mode cannot prepare, each with the code it would fail (V445, V541).

    A packet reads only the documents the workspace holds. A unit whose issuers holding one fall
    short of the installed floor fails `alternative_evidence.minimum_entity_coverage_not_met`
    at its acquisition, the same rule its refusal applies (`sources_short`); one whose issuers
    hold none under a floor that admits that fails `alternative_evidence.document_set_empty`.
    Under official acquisition the filing index is read at the run, so none is named.

    Args:
        inventory: The preview's source inventory: each issuer, its unit and its documents.
        mode: The Host's evidence mode.
        floor: The installed package's minimum issuer coverage.

    Returns:
        Each such unit in the run's order, heaviest first: its id, its code and its issuers
        without a source; empty outside RECORDED mode.
    """
    if mode is not AlternativeEvidenceMode.RECORDED:
        return []
    units: dict[str, list[tuple[str, int]]] = {}
    for issuer in cast(list[dict[str, object]], inventory.get("issuers") or []):
        units.setdefault(str(issuer.get("unit_id")), []).append(
            (str(issuer.get("entity_id")), int(cast(int, issuer.get("documents") or 0)))
        )
    named: list[dict[str, object]] = []
    for unit_id, issuers in units.items():
        entities = tuple(entity for entity, _documents in issuers)
        sourced = {entity for entity, documents in issuers if documents}
        short = sources_short(entities, sourced, floor=floor)
        if short is not None or not sourced:
            named.append(
                {
                    "unit_id": unit_id,
                    "failure_code": "alternative_evidence.document_set_empty"
                    if short is None
                    else str(short),
                    "issuers_without_source": list(entities if short is None else short.uncovered),
                }
            )
    return named


@dataclass(frozen=True, slots=True)
class ReviewOutcome:
    """What asking for a refresh or a review produced, without waiting for any of it."""

    disposition: str
    detail: str
    task_id: UUID | None = None
    lifecycle: str | None = None
    review: PortfolioReviewView | None = None
    evidence_as_of: datetime | None = None
    """The cutoff the preparation answers as of -- the original one on a reuse."""
    failure_code: str | None = None
    next_requests: dict[str, dict[str, str]] | None = None
    evidence_unit_id: str | None = None
    """The coverage unit a submitted answer was for, when it was for one."""
    answer: dict[str, object] | None = None
    """How the Host received the answer this admitted: its verdict
    (`ACCEPTED` or `DONE`), its number, the items accepted and dropped."""
    network_access: dict[str, object] | None = None
    """The effective network owner's reading when source acquisition was refused."""
    source_network_access: dict[str, object] | None = None
    """The launch reading that composed this Host's source transport."""
    source_ways: dict[str, object] | None = None
    """The existing source recovery, including replacement of an idle denied Host."""

    @property
    def reused(self) -> bool:
        """Read whether review disposition reused exact retained evidence.

        Returns:
            True exactly for REUSED_EXACT.
        """
        return self.disposition == "REUSED_EXACT"


def retrieval_recipe_view(spec: HybridIndexSpec) -> dict[str, object]:
    """Project the bound retrieval recipe and its resolved device requirements.

    The retrieval recipe a workspace is bound to, as a reader sees it: the
    encoder and reranker by name and revision, the device the recipe needs,
    and -- for a GPU recipe -- what this host resolves for it. Never a
    quality claim; a pack's presence is proven by the capability hash the
    manifest binds, not restated here.
    """
    gpu = (
        spec.encoder_runtime == ENCODER_RUNTIME_TORCH_CUDA
        or spec.reranker_runtime == RERANKER_RUNTIME_TORCH_CUDA
    )
    view: dict[str, object] = {
        "recipe": spec.policy_id,
        "encoder": {
            "model_id": spec.model_id,
            "revision": spec.model_revision,
            "runtime": spec.encoder_runtime,
            "dimension": spec.embedding_dimension,
        },
        "reranker": {
            "model_id": spec.reranker_model_id,
            "revision": spec.reranker_model_revision,
            "runtime": spec.reranker_runtime,
        },
        "device": "gpu" if gpu else "cpu",
    }
    if gpu:
        from alphalattice.kernel.knowledge.model_store import runtime_availability

        view["host_runtime"] = runtime_availability()
    return view


ANSWER_CATEGORY = "agent-answers"
BUNDLE_CATEGORY = "agent-bundles"


def _answer_view(
    record: AgentAnswerRecord,
    delivery: dict[str, object] | None = None,
    *,
    record_filed: bool = True,
) -> dict[str, object]:
    """What the agent that answered reads back: the verdict, the answer's
    number, the corrections left and the items -- in plain words."""

    problems = [value.model_dump(mode="json") for value in record.problems]
    # After a correction the Host reads this many more answers to the bundle;
    # after the last it keeps the acceptable part.
    rounds_left = (
        ANSWER_CORRECTION_BOUND - record.corrections_used + 1
        if record.verdict is AnswerVerdict.CORRECT
        else 0
    )
    view: dict[str, object] = {
        "verdict": str(record.verdict),
        "number": record.number,
        "rounds_left": rounds_left,
        "accepted_items": list(record.accepted_items),
    }
    stored = delivery.get("answer_record") if delivery is not None else None
    if isinstance(stored, dict):
        view["answer_reference"] = stored["record_hash"]
        if stored.get("agent_run") is not None:
            view["recorded_agent"] = stored["agent_run"]
    elif record_filed:
        view["answer_reference"] = record.record_hash
        if record.agent_run is not None:
            view["recorded_agent"] = record.agent_run.model_dump(mode="json")
    if record.verdict is AnswerVerdict.CORRECT:
        view["problems"] = problems
        view["message"] = correction_text(
            record.problems, accepted=record.accepted_items, rounds_left=rounds_left
        )
    elif record.verdict is AnswerVerdict.DONE:
        view["dropped"] = problems
    if delivery is not None:
        view["accepted_delivery"] = delivery
    return view


def agent_answer_result(
    role: str, body: Mapping[str, object], book: Mapping[str, object] | None = None
) -> dict[str, object]:
    """Project the specialist answer verdict, corrections and accepted receipt.

    What the specialist that submitted reads: the verdict in plain words,
    the problems to correct or the items dropped, and -- once the Host has
    accepted or finished with the answer -- the receipt it returns to its lead,
    which is all the lead receives, with the exact accepted record reference.
    """
    answer = body.get("answer")
    if body.get("status") == "CORRECT" and isinstance(answer, dict):
        return {
            "status": "CORRECT",
            "agent_role": role,
            **(
                {"recorded_agent": answer["recorded_agent"]}
                if isinstance(answer.get("recorded_agent"), dict)
                else {}
            ),
            "message": answer["message"],
            "problems": answer["problems"],
            "accepted_items": answer["accepted_items"],
            "answer_number": answer["number"],
            "rounds_left": answer["rounds_left"],
            "next_action": "CORRECT_THE_NAMED_ITEMS_AND_SUBMIT_AGAIN",
        }
    disposition = str(body.get("disposition", ""))
    if isinstance(answer, dict) and disposition in {"ADMITTED", "REUSED_EXACT"}:
        verdict = str(answer["verdict"])
        dropped = list(cast(list[object], answer.get("dropped", [])))
        accepted = len(cast(list[object], answer["accepted_items"]))
        task = body.get("task_id") or body.get("publication_task_id")
        message = (
            "The Host accepted your answer."
            if verdict == "ACCEPTED"
            else f"The Host kept the {accepted} acceptable item(s) of your answer and dropped "
            f"{len(dropped)}, listed under 'dropped'; no more answers are read for this bundle."
        )
        return {
            "status": verdict,
            "agent_role": role,
            **(
                {"recorded_agent": answer["recorded_agent"]}
                if isinstance(answer.get("recorded_agent"), dict)
                else {}
            ),
            **(
                {"answer_reference": answer["answer_reference"]}
                if isinstance(answer.get("answer_reference"), str)
                else {}
            ),
            **({"task_id": body["task_id"]} if isinstance(body.get("task_id"), str) else {}),
            # The Task the answer started, its state beside the verdict, so a wait follows it
            # to its end rather than ending on the verdict (OP18, V580).
            **(
                {"task_lifecycle": body["lifecycle"]}
                if isinstance(body.get("lifecycle"), str)
                else {}
            ),
            "message": message + " Stop here and return the receipt to your lead.",
            **({"dropped": dropped} if dropped else {}),
            "receipt": {
                "agent_role": role,
                "verdict": verdict,
                **(
                    {"answer_reference": answer["answer_reference"]}
                    if isinstance(answer.get("answer_reference"), str)
                    else {}
                ),
                "task_id": task,
                "accepted_items": accepted,
                "dropped_items": len(dropped),
            },
            "next_action": "RETURN_THE_RECEIPT_TO_YOUR_LEAD",
            # For the lead: the Task the answer started and the book it answered for, so the
            # work goes on from this answer and no earlier one is looked for (V406).
            **(
                {
                    "next_requests": {
                        **({"task": {"operation": "STATUS", "task_id": task}} if task else {}),
                        **({"book": {"operation": "EVIDENCE_CRO", **book}} if book else {}),
                    }
                }
                if task or book
                else {}
            ),
        }
    return {
        "status": "REFUSED",
        "agent_role": role,
        "failure_code": body.get("failure_code") or disposition or "agent_bundle.not_admitted",
        "message": str(
            body.get("detail") or "The Host admitted nothing for this answer; tell your lead."
        ),
        "next_action": "TELL_YOUR_LEAD",
    }


def _answer_correction(code: str, record: AgentAnswerRecord) -> dict[str, object]:
    """An answer returned for correction: nothing admitted, every problem by
    item in plain words, the items already acceptable, the corrections left."""

    return {
        "status": "CORRECT",
        "failure_code": code,
        "task_id": None,
        "answer": _answer_view(record),
        "next_action": "CORRECT_THE_NAMED_ITEMS_AND_SUBMIT_AGAIN",
    }


def _submitted(submission: CommandSubmission) -> ReviewOutcome:
    return ReviewOutcome(
        disposition=submission.disposition,
        detail=submission.refusal_detail or "One Task was admitted.",
        task_id=submission.task_id,
        lifecycle=submission.lifecycle,
    )


@dataclass
class AlternativeEvidenceRefreshCommand:
    """One scoped Alternative Evidence Task, as the generic dispatcher sees it.

    A book's coverage run, which carries every unit's request itself; or one
    unit's own Task -- its submitted analysis or its continuation -- by its
    request, obligation and admission. The Host admits no single preparation
    of a whole book since C2; one admitted before recovers as it was admitted.
    """

    application: EvidenceReviewApplication
    obligation: AlternativeEvidenceResearchObligation | None = None
    request: AlternativeEvidenceRequest | None = None
    admission: AlternativeEvidenceAdmission | None = None
    prepare_only: bool = False
    submitted_analysis: SubmittedEvidenceAnalysis | None = None
    continuation: EvidenceContinuation | None = None
    run: AlternativeEvidenceCoverageRun | None = None
    _task_id: UUID | None = field(default=None, init=False, repr=False)

    @property
    def command_kind(self) -> str:
        """Read the installed evidence refresh command identity.

        Returns:
            Exact command-kind declaration.
        """
        return EVIDENCE_REFRESH_COMMAND

    def admit(self) -> CommandAdmission:
        """Admit the prepared evidence refresh contract through durable task ownership.

        Returns:
            Task identity/lifecycle and retained admitted task ID.

        Raises:
            PortfolioEvidenceReviewError: Required adapter, contract or prepared review is absent.
        """
        envelope, goal, plan = self.contract()
        record = self.application.session.task_control_registry.admit(
            input_envelope=envelope,
            goal=goal,
            plan=plan,
            observed_at=self.application.clock(),
        ).record
        self._task_id = record.task_id
        return CommandAdmission(task_id=record.task_id, lifecycle=record.lifecycle.value)

    def contract(self) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
        """Compile exact coverage-run or document-refresh task input and resource authority.

        Returns:
            Task envelope, goal and workflow selected by the prepared request/continuation.

        Raises:
            PortfolioEvidenceReviewError: Adapter or required request/obligation/admission fields
                are absent.
        """
        adapter = self.application.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        if self.run is not None:
            return coverage_run_task_contract(run=self.run, prepare_only=self.prepare_only)
        if self.request is None or self.obligation is None or self.admission is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_refresh_command_incomplete")
        return alternative_evidence_document_task_contract(
            request=self.request,
            admission=self.admission,
            obligation=self.obligation,
            resource_binding_hash=(
                adapter.resources.preparation_binding_hash
                if self.prepare_only
                or self.submitted_analysis is not None
                or self.continuation is not None
                else adapter.resources.binding_hash
            ),
            prepare_only=self.prepare_only,
            submitted_analysis=self.submitted_analysis,
            continuation=self.continuation,
        )

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Drive the admitted evidence refresh task under its retained application owner.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        self.application.drive_evidence_task(task_id=task_id, expected_task_hash=expected_task_hash)

    @classmethod
    def recover(
        cls, *, application: EvidenceReviewApplication, task: TaskRecord
    ) -> AlternativeEvidenceRefreshCommand:
        """Rebuild this command from the Task input that was actually admitted."""
        payload = task.input.payload
        if payload.get("purpose") == COVERAGE_RUN_PURPOSE:
            adapter = application.evidence_task_adapter
            if adapter is None:
                raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
            # The run by the hash the Task names; whether this workspace's
            # package still binds it is the adapter's to refuse when it runs.
            run = adapter.run_of(task)
            if run is None:
                raise PortfolioEvidenceReviewError("product_host.evidence_coverage_run_unreadable")
            return cls(
                application=application,
                run=run,
                prepare_only=bool(payload.get("prepare_only", True)),
            )
        return cls(
            application=application,
            obligation=AlternativeEvidenceResearchObligation.model_validate(payload["obligation"]),
            request=AlternativeEvidenceRequest.model_validate(payload["request"]),
            admission=AlternativeEvidenceAdmission.model_validate(payload["admission"]),
            prepare_only=payload.get("purpose") == "PREPARE_PACKET",
            submitted_analysis=(
                SubmittedEvidenceAnalysis.model_validate(payload["submitted_analysis"])
                if payload.get("purpose") == "SUBMITTED_ANALYSIS"
                else None
            ),
            continuation=(
                EvidenceContinuation.model_validate(payload["continuation"])
                if payload.get("purpose") == "CONTINUE_READING"
                else None
            ),
        )


@dataclass
class PortfolioReviewCommand:
    """One three-stage CRO review, admitted against an already-published dossier."""

    application: EvidenceReviewApplication
    dossier: PortfolioReviewDossier | None = None
    actor: PortfolioReviewActor | None = None
    contract: tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan] | None = None
    _task_id: UUID | None = field(default=None, init=False, repr=False)

    @property
    def command_kind(self) -> str:
        """Read the installed CRO review command identity.

        Returns:
            Exact command-kind declaration.
        """
        return CRO_REVIEW_COMMAND

    def admit(self) -> CommandAdmission:
        """Admit the prepared CRO review contract through durable task ownership.

        Returns:
            Task identity/lifecycle and retained admitted task ID.

        Raises:
            PortfolioEvidenceReviewError: Required adapter, contract or prepared review is absent.
        """
        if self.dossier is None or self.actor is None or self.contract is None:
            raise PortfolioEvidenceReviewError("chief_risk_officer.review_not_prepared")
        envelope, goal, plan = self.contract
        record = self.application.session.task_control_registry.admit(
            input_envelope=envelope,
            goal=goal,
            plan=plan,
            observed_at=self.application.clock(),
        ).record
        self._task_id = record.task_id
        return CommandAdmission(task_id=record.task_id, lifecycle=record.lifecycle.value)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Drive the admitted CRO review task under its retained application owner.

        Args:
            task_id: Exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        actor = self.actor or self.application.recover_review_actor(
            self.application.session.task_control_registry.task(task_id)
        )
        self.application.drive_review_task(
            actor=actor, task_id=task_id, expected_task_hash=expected_task_hash
        )


_CHECKS_ITEM = "checks:"
"""The delivery item of one issuer's reported checks in a dossier read."""


CURRENT_WITHIN = timedelta(seconds=604_800)
"""The longest analysis TTL a request admits (`AlternativeEvidenceRequest.ttl_seconds`):
an analysis is published at or after its cutoff, so one published before now less this
has expired, and a reader of current analyses reads no older record (Z2)."""


@dataclass
class EvidenceReviewApplication:
    """The Product Host owner for the whole cross-Desk route."""

    def replan_request(self, task: TaskRecord) -> dict[str, object]:
        """Ask for the book selector the evidence-only Task does not retain."""
        from alphalattice.control.product_host.composition.plain_refusals import refused

        return refused("evidence_review.replan_book_selection_required")

    replans = (
        TaskReplan(
            task_kind=AlternativeEvidenceDocumentTaskAdapter.task_kind,
            preview="EVIDENCE_PREVIEW",
            admitting="EVIDENCE_PREPARE",
        ),
        TaskReplan(task_kind=CRO_REVIEW_COMMAND, admitting="CRO_REVIEW"),
    )
    """The re-plans of the Task kinds this route admits, which the recovery view offers
    (V188): the CRO review has no confirmation-free preview."""

    workspace_id: str
    workspace: Path
    session: WorkspaceApplicationSession
    ledger: PortfolioLedgerStore | None
    artifacts: AlternativeEvidenceArtifactStore
    evidence_publications: AlternativeEvidenceAnalysisPublicationService | None
    review_publications: PortfolioReviewPublicationService
    playpen_root: Path
    latest_result_hash: Callable[[], str | None]
    """The workspace's most recent completed result, the default book to review."""

    finalization: PortfolioFinalizationStore | None = None
    registry: SecIssuerRegistrySnapshot | None = None
    listing_authority: AdmittedListingTickerAuthority | None = None
    handoff: ValidatedPortfolioHandoff | None = None
    evidence_task_adapter: AlternativeEvidenceDocumentTaskAdapter | None = None
    evidence_policy: AdmittedEvidencePolicy = field(default_factory=AdmittedEvidencePolicy)
    model_authority_admitted: bool = True
    """False when no Provider credential is admitted to this process.

    Book/evidence readback still works. Automatic analyst/reviewer work refuses
    before admission; a separately validated external assessment uses its own
    explicit submission entry, not an installed-model identity.
    """
    review_actor: PortfolioReviewActor | None = None
    selected_analysis_publication_hash: str | None = None
    response_schema_hash: str = "0" * 64
    typed_user_authority: str = "local product reviewer"
    portfolio_report_link: str = "/report"
    clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC)
    runtime_path: Path | None = None
    read_update: Callable[[UUID, str], dict[str, object]] | None = None
    installed_temporal_statements: Callable[[date, date], tuple[str, ...]] | None = None
    read_experiment: Callable[[UUID, str], dict[str, object]] | None = None
    campaign_summary: Callable[[], dict[str, object]] | None = None
    network_access: NetworkAccess | None = None
    """The official source's admission reading; runtime data, never a run binding."""
    _unit_obligation_cache: dict[
        tuple[str, datetime],
        dict[tuple[str, ...], tuple[str, AlternativeEvidenceResearchObligation]],
    ] = field(default_factory=dict, init=False, repr=False)
    """Sealed unit obligations by (scope, cutoff): the same words every time."""
    _sealed_export: tuple[str, datetime, dict[str, object]] | None = field(
        default=None, init=False, repr=False
    )
    """The export citation pages slice (V94): `EvidenceReviewDelivery`'s, the last it sealed."""
    _sealed_export_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _answer_turns: dict[str, threading.Lock] = field(default_factory=dict, init=False, repr=False)
    """One lock a binding: its answers are numbered and filed one at a time (V557)."""
    _answer_turns_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Each review keeps its day as it is published, so a book's review is
        # chosen from the records of their days (Z2).
        """Register dated CRO publication notes with the installed evidence runtime."""
        if self.evidence_task_adapter is not None:
            self.review_publications.noted = self.evidence_task_adapter.runtime.days_of(
                "cro-review-publications", "published_at"
            ).note

    # ------------------------------------------------------ deterministic half

    @property
    def has_evidence_authority(self) -> bool:
        """A registry and a listing authority. Without them no issuer can be named."""
        return self.registry is not None and self.listing_authority is not None

    def inputs(self) -> PortfolioEvidenceReviewInputs:
        """Project retained deterministic evidence/review readback owners.

        Returns:
            PortfolioEvidenceReviewInputs over exact ledger, registry, listing, report/handoff and
            temporal statement owners.
        """
        return PortfolioEvidenceReviewInputs(
            ledger=self.ledger,
            finalization=self.finalization,
            registry=self.registry,
            listing_authority=self.listing_authority,
            portfolio_report_link=self.portfolio_report_link,
            handoff=self.handoff,
            read_update=self.read_update,
            read_experiment=self.read_experiment,
            installed_temporal_statements=self.installed_temporal_statements,
        )

    def default_selector(self, selector: BookSelector | None = None) -> BookSelector | None:
        """The book to review when none was named: a given handoff, else the latest result."""
        if selector is not None and (
            selector.result_hash is not None
            or selector.handoff_hash is not None
            or selector.update_task_id is not None
            or selector.experiment_task_id is not None
        ):
            return selector
        if self.handoff is not None:
            return BookSelector(handoff_hash=self.handoff.handoff_hash)
        latest = self.latest_result_hash()
        if latest is None:
            return None
        return BookSelector(result_hash=latest)

    def resolve_book(self, selector: BookSelector) -> ResolvedBookScope:
        """Open the book and derive the issuer scope without inventing a cutoff."""
        inputs = self.inputs()
        book = self._open_book(inputs, selector)
        projection, scope = compile_portfolio_scope(inputs, book)
        return ResolvedBookScope(book=book, projection=projection, scope=scope)

    def _open_book(
        self, inputs: PortfolioEvidenceReviewInputs, selector: BookSelector
    ) -> SealedBook:
        """The book a request's selector names, its Task refused by the selector's code (V546).

        A study's selector that names a Task of another kind -- an installed strategy's public
        development replay, say, whose book is its result -- is refused by that kind, and a
        study's or an update's Task the workspace does not hold by the selector's own code.
        """
        try:
            return open_sealed_book(inputs, selector)
        except TaskNotFoundError as error:
            if selector.experiment_task_id is not None:
                code = "product_host.evidence_review_experiment_not_published"
            elif selector.update_task_id is not None:
                code = "product_host.evidence_review_update_task_mismatch"
            else:
                raise
            raise PortfolioEvidenceReviewError(code) from error
        except ValueError as error:
            if selector.experiment_task_id is None or not str(error).endswith(
                "research_experiment.task_kind_mismatch"
            ):
                raise
            kind = self.session.task_control_registry.task(selector.experiment_task_id).task_kind
            raise PortfolioEvidenceReviewError(
                f"product_host.evidence_review_study_selector_not_a_study:{kind}"
            ) from error

    def resolve_coverage(
        self,
        selector: BookSelector,
        *,
        evidence_as_of: datetime,
        prepare_only: bool = True,
        read_inventory: bool = False,
    ) -> ResolvedCoverage:
        """Open the book and seal every unit it is owed at this cutoff.

        `read_inventory` is the preparation's: each holding's filing index is
        read once at the cutoff before packing (a preview, read on every page,
        packs from what is already known and requests nothing).
        """
        adapter = self.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        resolved = self.resolve_book(selector)
        policy = (
            replace(self.evidence_policy, admit_model_review=False)
            if prepare_only
            else self.evidence_policy
        )
        binding = adapter.run_binding_hash(prepare_only=prepare_only)
        # The run this scope sealed at this cutoff under the same package and
        # permissions is the coverage at this cutoff: a captured intent
        # resolves to it after its own acquisition, or a later delta, has
        # moved the counts a fresh packing would read.
        for task in self._run_tasks(resolved.scope, evidence_as_of=evidence_as_of):
            sealed = adapter.run_of(task)
            if (
                sealed is not None
                and sealed.scope_hash == resolved.scope.scope_hash
                and sealed.evidence_as_of == evidence_as_of
                and sealed.resource_binding_hash == binding
                and (
                    sealed.network_consent,
                    sealed.admit_live_official,
                    sealed.admit_model_review,
                )
                == (policy.network_consent, policy.admit_live_official, policy.admit_model_review)
                and all(self.evidence_policy.matches(unit.request) for unit in sealed.units)
            ):
                return ResolvedCoverage(resolved=resolved, run=sealed, prepare_only=prepare_only)
        if (
            read_inventory
            and policy.mode is not AlternativeEvidenceMode.RECORDED
            and policy.network_consent
        ):
            refusal = self._source_acquisition_refusal()
            if refusal is not None:
                assert refusal.failure_code is not None
                raise PortfolioEvidenceReviewError(refusal.failure_code)
            adapter.read_inventory(
                entities=resolved.scope.ordered_entity_ids,
                evidence_as_of=evidence_as_of,
                request_for=lambda entities: self.evidence_policy.request(
                    ordered_entity_ids=entities, evidence_as_of=evidence_as_of
                ),
            )
        counted = self._source_counts(resolved.scope, evidence_as_of=evidence_as_of)
        run = project_coverage_run(
            scope=resolved.scope,
            evidence_as_of=evidence_as_of,
            approved_source_families=self.evidence_policy.approved_source_families,
            request_for=lambda entities, as_of, read: self.evidence_policy.request(
                ordered_entity_ids=entities, evidence_as_of=as_of, read_filings=read
            ),
            resource_binding_hash=binding,
            network_consent=policy.network_consent,
            admit_live_official=policy.admit_live_official,
            admit_model_review=policy.admit_model_review,
            source_counts=counted.counts,
            nothing_filed=counted.nothing_filed,
            carried=counted.carried,
            read_filings=counted.read_filings,
        )
        return ResolvedCoverage(resolved=resolved, run=run, prepare_only=prepare_only)

    def _source_acquisition_refusal(
        self, *, evidence_as_of: datetime | None = None
    ) -> ReviewOutcome | None:
        """Project the admitted source's network hold using its owner's words."""
        if self.network_access is None:
            return None
        current = network_access(self.workspace)
        if self.network_access.allowed and current.allowed:
            return None
        from alphalattice.control.product_host.composition.plain_refusals import refused

        words = refused("evidence_review.workspace_network_not_allowed")
        ways = source_ways(self.workspace, entities=None, book=None)
        official = cast(dict[str, object], ways["official"])
        return ReviewOutcome(
            disposition="REFUSED_NETWORK_ACCESS",
            detail=str(official["before"]),
            failure_code=words["failure_code"],
            evidence_as_of=evidence_as_of,
            network_access=current.body(),
            source_network_access=self.network_access.body(),
            source_ways=ways,
            next_requests={"network": {"operation": "NETWORK_ACCESS"}},
        )

    def _source_counts(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime
    ) -> LogicalSourceCounts:
        """Each issuer's logical source count at the cutoff, with its basis, the
        issuers that filed nothing in the window and those whose every filing in
        it was read earlier, from the adapter's source owners; the packing reads
        the counts, the preview shows the bases. No inventory request, no
        acquisition."""

        adapter = self.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        policy = self.evidence_policy
        return adapter.logical_source_counts(
            entities=scope.ordered_entity_ids,
            evidence_as_of=evidence_as_of,
            mode=policy.mode,
            evidence_classes=policy.evidence_classes,
            policy_budget=policy.source_policy.maximum_documents_per_issuer,
            window_days=policy.source_policy.sec_recent_8k_days,
        )

    def _units(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime | None = None
    ) -> tuple[tuple[str, ...], ...]:
        """The scope's units in execution order; none for a scope naming no
        issuer. At a cutoff, the units of the run this scope sealed at that
        cutoff when one exists -- the packing a preparation seals is the
        packing at its cutoff, whatever the sealed plans say afterwards --
        else the packing from each issuer's logical source count at that
        cutoff; without a cutoff -- a view of what the book holds -- the units
        of the newest coverage Task that covers the scope, else the packing
        at the present clock."""

        if not scope.selected_issuers:
            return ()
        adapter = self.evidence_task_adapter
        if adapter is None:
            # No source owner to count from: the issuer limit alone, as a
            # view before the package is admitted.
            return coverage_units(scope.ordered_entity_ids, priority_rank=scope.priority_rank)
        sealed = self._sealed_run(scope, evidence_as_of=evidence_as_of)
        if sealed is not None:
            # A run's own acquisition seals plans at its cutoff, and the
            # count read "at or before the cutoff" would then move under
            # the run: its units are re-derived from the run it sealed,
            # never from the plans it left (seen on the book journey: the
            # packet of a nine-unit run refused as another book's unit
            # once its own plans packed the same cutoff into eight).
            return tuple(unit.ordered_entity_ids for unit in sealed.units)
        if evidence_as_of is None:
            evidence_as_of = self.clock()
        counted = self._source_counts(scope, evidence_as_of=evidence_as_of)
        return coverage_units(
            scope.ordered_entity_ids,
            priority_rank=scope.priority_rank,
            source_counts=counted.counts,
            weight_rank={issuer.entity_id: issuer.weight_rank for issuer in scope.selected_issuers},
            nothing_filed=counted.nothing_filed,
            carried=counted.carried,
        )

    def _unit_obligations(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime
    ) -> dict[tuple[str, ...], tuple[str, AlternativeEvidenceResearchObligation]]:
        """One (unit id, obligation) per unit of this scope at this cutoff, keyed by
        the unit's issuers, in execution order. A book that fits one unit has one."""

        if not scope.selected_issuers:
            return {}
        key = (scope.scope_hash, evidence_as_of)
        cached = self._unit_obligation_cache.get(key)
        if cached is None:
            units = self._units(scope, evidence_as_of=evidence_as_of)
            ids = coverage_unit_ids(units)
            cached = {
                entities: (
                    ids[entities],
                    project_unit_obligation(
                        ordered_entity_ids=entities,
                        evidence_as_of=evidence_as_of,
                        approved_source_families=self.evidence_policy.approved_source_families,
                    ),
                )
                for entities in units
            }
            self._unit_obligation_cache[key] = cached
        return cached

    # ------------------------------------------------------------- evidence

    def _publication_question(
        self, *, resolved: ResolvedBookScope, publication: AlternativeEvidenceAnalysisPublication
    ) -> ResolvedEvidenceQuestion | None:
        """Rebuild a publication's question against the scope held now."""

        request = self.artifacts.load(
            "requests", publication.request_hash, AlternativeEvidenceRequest
        )
        if not self.evidence_policy.matches(request):
            return None
        unit = self._unit_obligations(resolved.scope, evidence_as_of=request.evidence_as_of).get(
            tuple(request.ordered_entity_ids)
        )
        if unit is None or publication.obligation_hash != unit[1].obligation_hash:
            return None
        return ResolvedEvidenceQuestion(resolved=resolved, obligation=unit[1])

    def eligible_evidence(
        self, resolved: ResolvedBookScope
    ) -> tuple[tuple[ResolvedEvidenceQuestion, AlternativeEvidenceAnalysisPublicationView], ...]:
        """Every current analysis that answers this book's scope, newest first.

        What a person is choosing between. Expired analyses are left out because
        selecting one could not produce a review anyway.
        """
        service = self.evidence_publications
        if service is None:
            return ()
        now = self.clock()
        found: list[_QuestionRecord] = []
        for publication in analysis_records(
            self.artifacts, self._publication_days(), now - CURRENT_WITHIN
        ):
            if service.standing(publication, now=now) != "CURRENT":
                continue
            question = self._publication_question(resolved=resolved, publication=publication)
            if question is not None:
                found.append((question, publication))
        found.sort(key=lambda value: value[1].published_at, reverse=True)
        return tuple(
            (question, service.replay(publication.publication_hash, now=now))
            for question, publication in found
        )

    def select_evidence(
        self,
        *,
        selector: BookSelector,
        analysis_publication_hash: str,
        chosen_by: str = "HUMAN",
    ) -> AdmittedEvidenceSelection:
        """Record which analysis this book's review is to be read against.

        Refuses anything that could not produce a truthful review: a publication
        this scope's question does not match, or one that is no longer current.
        The record is appended, never overwritten, so the previous answer and
        both publications remain readable.
        """
        resolved = self.resolve_book(selector)
        eligible = {
            view.publication.publication_hash: (question, view)
            for question, view in self.eligible_evidence(resolved)
        }
        chosen = eligible.get(analysis_publication_hash)
        if chosen is None:
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_selection_not_eligible:" + analysis_publication_hash[:12]
            )
        if chosen_by not in {"HUMAN", "INSTALLED_AGENT", "EXTERNAL_AUTOMATION"}:
            raise PortfolioEvidenceReviewError("product_host.evidence_selection_actor_invalid")
        question, _view = chosen
        selection = seal_portfolio_evidence_contract(
            AdmittedEvidenceSelection,
            "selection_hash",
            issuer_scope_hash=resolved.scope.scope_hash,
            analysis_publication_hash=analysis_publication_hash,
            evidence_as_of=question.obligation.evidence_as_of,
            chosen_at=self.clock(),
            chosen_by=chosen_by,
        )
        self.artifacts.publish(SELECTIONS_CATEGORY, selection.selection_hash, selection)
        return selection

    def _select_unit_evidence(
        self, resolved: ResolvedBookScope
    ) -> dict[tuple[str, ...], _CurrentEvidenceSelection]:
        """Select explicit/unique current evidence for every unit, without
        re-deriving any cutoff.

        Keyed by the unit's issuers, in execution order. Each unit is answered
        by the rule one request always had: the explicitly selected
        publication where it answers this unit; else the one current
        publication; two current publications are refused, not resolved,
        unless the newest recorded selection among this unit's publications
        names one of them; else the newest publication that is not current,
        so the state can say which analysis it is and why; else nothing.
        """

        units = self._units(resolved.scope)
        service = self.evidence_publications
        if service is None:
            return {entities: _CurrentEvidenceSelection(disposition="ABSENT") for entities in units}
        now = self.clock()
        current: dict[tuple[str, ...], list[_QuestionRecord]] = {entities: [] for entities in units}
        stale: dict[tuple[str, ...], _QuestionRecord] = {}
        answers: dict[tuple[str, ...], _CurrentEvidenceSelection] = {}
        if self.selected_analysis_publication_hash is not None:
            view = service.replay(self.selected_analysis_publication_hash, now=now)
            question = self._publication_question(resolved=resolved, publication=view.publication)
            if question is None:
                raise PortfolioEvidenceReviewError(
                    "product_host.evidence_review_selected_publication_off_scope"
                )
            answers[question.obligation.ordered_entity_ids] = _CurrentEvidenceSelection(
                disposition=str(view.current_eligibility), question=question, evidence=view
            )
        # Chosen from the records, newest first: every current analysis is
        # asked its question, one that is not current only while a unit has no
        # answer yet (X2) -- a question reads its request and its day's run.
        # Only a record of the last TTL can be current; an older one is read
        # only while a unit still waits, day by day (Z2).
        records = sorted(
            analysis_records(self.artifacts, self._publication_days(), now - CURRENT_WITHIN),
            key=lambda value: value.published_at,
            reverse=True,
        )
        by_hash = {value.publication_hash: value for value in records}
        eligibility = {
            value.publication_hash: service.standing(value, now=now) for value in records
        }
        asked: dict[str, tuple[str, ...] | None] = {}
        questions: dict[str, ResolvedEvidenceQuestion] = {}

        def unit_of(publication_hash: str) -> tuple[str, ...] | None:
            if publication_hash not in asked:
                question = self._publication_question(
                    resolved=resolved, publication=by_hash[publication_hash]
                )
                asked[publication_hash] = None
                if question is not None and question.obligation.ordered_entity_ids in current:
                    questions[publication_hash] = question
                    asked[publication_hash] = question.obligation.ordered_entity_ids
            return asked[publication_hash]

        for publication in records:
            key = (
                unit_of(publication.publication_hash)
                if eligibility[publication.publication_hash] == "CURRENT"
                else None
            )
            if key is not None and key not in answers:
                current[key].append((questions[publication.publication_hash], publication))
        waiting = {key for key in units if key not in answers and not current[key]}
        for publication in analysis_records_newest_first(
            self.artifacts, self._publication_days(), by_hash, waiting=waiting
        ):
            eligibility.setdefault(
                publication.publication_hash, service.standing(publication, now=now)
            )
            if not waiting:
                break
            # Published under an authority this workspace has rotated past, or
            # current and already counted.
            if eligibility[publication.publication_hash] in {None, "CURRENT"}:
                continue
            key = unit_of(publication.publication_hash)
            if key in waiting:
                # The newest analysis that is not current, whatever the
                # reason: its eligibility says whether it expired or was
                # sealed under a superseded contract.
                stale[key] = (questions[publication.publication_hash], publication)
                waiting.discard(key)
        recorded = recorded_selections(self.artifacts, resolved.scope.scope_hash)
        for key in units:
            if key in answers:
                continue
            candidates = current[key]
            if len(candidates) > 1:
                # A person may already have answered this for the unit. That
                # answer is read here rather than at composition, so it
                # survives a restart: the newest record among this unit's
                # publications, and it must name one that is current.
                named = next(
                    (
                        value.analysis_publication_hash
                        for value in recorded
                        if value.analysis_publication_hash in by_hash
                        and unit_of(value.analysis_publication_hash) == key
                    ),
                    None,
                )
                chosen = [
                    value
                    for value in candidates
                    if named is not None and value[1].publication_hash == named
                ]
                if len(chosen) != 1:
                    raise EvidenceSelectionAmbiguous(
                        "product_host.evidence_review_evidence_ambiguous:"
                        + ",".join(sorted(value.publication_hash[:12] for _, value in candidates))
                    )
                candidates = chosen
            answer = candidates[0] if candidates else stale.get(key)
            if answer is None:
                answers[key] = _CurrentEvidenceSelection(disposition="ABSENT")
                continue
            view = service.replay(answer[1].publication_hash, now=now)
            answers[key] = _CurrentEvidenceSelection(
                disposition=str(view.current_eligibility), question=answer[0], evidence=view
            )
        return {key: answers[key] for key in units}

    def current_evidence(
        self, *, question: ResolvedEvidenceQuestion
    ) -> AlternativeEvidenceAnalysisPublicationView | None:
        """The one current publication answering this exact obligation, or nothing."""
        service = self.evidence_publications
        if service is None:
            return None
        now = self.clock()
        matches = []
        for publication in analysis_records(
            self.artifacts, self._publication_days(), now - CURRENT_WITHIN
        ):
            if (
                publication.obligation_hash != question.obligation.obligation_hash
                or service.standing(publication, now=now) != "CURRENT"
            ):
                continue
            request = self.artifacts.load(
                "requests", publication.request_hash, AlternativeEvidenceRequest
            )
            if self.evidence_policy.matches(request):
                matches.append(publication)
        if len(matches) > 1:
            raise EvidenceSelectionAmbiguous(
                "product_host.evidence_review_evidence_ambiguous:"
                + ",".join(sorted(value.publication_hash[:12] for value in matches))
            )
        return service.replay(matches[0].publication_hash, now=now) if matches else None

    def _obligation_matches_scope(
        self,
        *,
        obligation: AlternativeEvidenceResearchObligation,
        request: AlternativeEvidenceRequest,
        scope: PortfolioIssuerScope,
    ) -> bool:
        if request.evidence_as_of != obligation.evidence_as_of or not (
            self.evidence_policy.matches(request)
        ):
            return False
        unit = self._unit_obligations(scope, evidence_as_of=obligation.evidence_as_of).get(
            tuple(request.ordered_entity_ids)
        )
        return unit is not None and unit[1].obligation_hash == obligation.obligation_hash

    def _run_covers_scope(self, task: TaskRecord, scope: PortfolioIssuerScope) -> bool:
        """Whether a coverage Task's run is the work this scope is owed: the same
        issuers under the admitted policy, whatever the book's weights were --
        except that a run of one unit asks its question in the book's priority
        order, so another order is another question."""

        adapter = self.evidence_task_adapter
        if adapter is None:
            return False
        run = adapter.run_of(task)
        if run is None:
            return False
        # The holdings that filed nothing in the window, and those whose every
        # filing an earlier analysis read, are the run's too, named beside its
        # units rather than packed into them.
        packed = tuple(
            entity
            for entity in scope.ordered_entity_ids
            if entity not in run.nothing_filed and entity not in run.carried
        )
        return (
            run.covered_entity_ids == set(scope.ordered_entity_ids)
            and bool(run.units)
            and (len(run.units) > 1 or tuple(run.units[0].ordered_entity_ids) == packed)
            and self.evidence_policy.matches(run.units[0].request)
        )

    def _run_tasks(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime | None = None
    ) -> Iterator[TaskRecord]:
        """Every coverage Task of this scope's issuers -- at this cutoff, when
        one is named -- newest first, whatever became of it. Each run is read
        as the caller reaches it, and a Task at another cutoff is passed over
        by the cutoff its input names, so a caller reads the runs it wants,
        not one a day (X2)."""

        adapter = self.evidence_task_adapter
        if adapter is None:
            return
        tasks = sorted(
            (
                task
                for task in self.session.task_control_registry.tasks()
                if task.task_kind == adapter.task_kind
                and task.input.payload.get("purpose") == COVERAGE_RUN_PURPOSE
                and (
                    evidence_as_of is None
                    or datetime.fromisoformat(str(task.input.payload["evidence_as_of"]))
                    == evidence_as_of
                )
            ),
            key=lambda task: (task.updated_at, str(task.task_id)),
            reverse=True,
        )
        yield from (task for task in tasks if self._run_covers_scope(task, scope))

    def active_evidence_refresh(self, *, scope: PortfolioIssuerScope) -> TaskSafeProjection | None:
        """The Alternative Evidence Task this scope is currently owed, if any.

        Read from Task Control's own safe projection -- the same one the
        dispatcher reports -- rather than from a flag this application keeps.
        `RECOVERY_REQUIRED` counts as in progress because the work is still owed.
        """
        latest = self._active_refresh_task(scope=scope)
        if latest is None:
            return None
        return self.session.task_control_registry.safe_projection(latest.task_id)

    def _active_refresh_task(self, *, scope: PortfolioIssuerScope) -> TaskRecord | None:
        """The newest Alternative Evidence Task still owed for this scope's
        issuers under the admitted policy, from Task Control's records."""

        active = {TaskLifecycle.QUEUED, TaskLifecycle.RUNNING, TaskLifecycle.RECOVERY_REQUIRED}
        registry = self.session.task_control_registry
        owed = []
        for task in registry.tasks():
            if (
                task.task_kind != AlternativeEvidenceDocumentTaskAdapter.task_kind
                or task.lifecycle not in active
            ):
                continue
            if task.input.payload.get("purpose") == COVERAGE_RUN_PURPOSE:
                if self._run_covers_scope(task, scope):
                    owed.append(task)
                continue
            raw_obligation = task.input.payload.get("obligation")
            raw_request = task.input.payload.get("request")
            if raw_obligation is None or raw_request is None:
                continue
            if self._obligation_matches_scope(
                obligation=AlternativeEvidenceResearchObligation.model_validate(raw_obligation),
                request=AlternativeEvidenceRequest.model_validate(raw_request),
                scope=scope,
            ):
                owed.append(task)
        if not owed:
            return None
        return max(owed, key=lambda task: (task.updated_at, str(task.task_id)))

    def evidence_selection_label(
        self, *, obligation_hash: str, publication_hash: str
    ) -> EvidenceSelectionLabel:
        """How the analysis behind this review came to be the one. Derived from the store.

        A sibling answer to the same obligation sealed under an authority this
        workspace has rotated past is not a candidate: it is excluded, as the
        section's other readers exclude it, so one orphaned publication never
        closes the section over a current one. The reviewed publication itself
        still refuses by name when it is the orphan.
        """
        service = self._require_evidence_publications()
        now = self.clock()
        current = []
        records = {
            value.publication_hash: value
            for value in analysis_records(
                self.artifacts, self._publication_days(), now - CURRENT_WITHIN
            )
        }
        if publication_hash not in records:
            # The reviewed analysis is read whatever its day: it refuses by
            # name when it is the orphan.
            with suppress(FileNotFoundError):
                records[publication_hash] = self.artifacts.load(
                    "analysis-publications",
                    publication_hash,
                    AlternativeEvidenceAnalysisPublication,
                )
        for publication in records.values():
            if publication.obligation_hash != obligation_hash:
                continue
            request = self.artifacts.load(
                "requests", publication.request_hash, AlternativeEvidenceRequest
            )
            if not self.evidence_policy.matches(request):
                continue
            if publication.publication_hash == publication_hash:
                # The reviewed analysis is replayed, and refuses by name when
                # it is the orphan; a sibling counts by its record.
                if service.replay(publication_hash, now=now).is_current:
                    current.append(publication)
            elif service.standing(publication, now=now) == "CURRENT":
                current.append(publication)
        chosen = next(
            (value for value in current if value.publication_hash == publication_hash), None
        )
        if chosen is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_selection_not_current")
        if len(current) == 1:
            return "UNIQUE_CURRENT"
        if any(value.published_at > chosen.published_at for value in current):
            return "EXPLICIT_OLDER_CURRENT_SELECTION"
        return "EXPLICIT_CURRENT_SELECTION"

    def preview_evidence(self, selector: BookSelector | None = None) -> dict[str, object]:
        """Explain source preparation independently of optional managed inference."""
        chosen = self._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            missing = chosen.disposition == "REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY"
            book = self.default_selector(selector)
            return {
                "status": "EVIDENCE_PREREQUISITES_MISSING",
                "failure_code": chosen.disposition,
                "explanation": chosen.detail,
                "next_action": "SELECT_A_BOOK_AND_ADMIT_ITS_LOCAL_EVIDENCE_PACKAGE",
                "next_requests": {
                    **(chosen.next_requests or {}),
                    "workspace": {"operation": "WORKSPACE_SHOW"},
                },
                # The commands that complete the setup; the script's help where none applies
                # (the page reads `setup` since U68, its fallback `setup_help`).
                **(
                    {"setup": evidence_setup(self.workspace, authored=_authored(book))}
                    if missing
                    else {
                        "setup_help": (
                            "scripts/materialize_evidence_cro_authority.py --help"
                            if (resolve_playpen_root(Path(__file__)) / "pyproject.toml").is_file()
                            else "python -m alphalattice.control.product_host.composition."
                            "evidence_authority_setup --help"
                        )
                    }
                ),
                "claim": "No Task, acquisition or model work performed.",
            }
        adapter = self.evidence_task_adapter
        if adapter is None or self.evidence_publications is None:
            return {
                "status": "EVIDENCE_PREREQUISITES_MISSING",
                "failure_code": "product_host.evidence_task_adapter_absent",
                "next_action": "ADMIT_SOURCE_PACKAGE_AND_PINNED_RETRIEVAL_RUNTIME",
                "setup": evidence_setup(self.workspace, authored=_authored(chosen)),
                "claim": "Source and retrieval authority are the workspace's admitted packages.",
            }
        if matter_selection_retired(self.evidence_policy.matter_selection):
            return {
                "status": "EVIDENCE_PREREQUISITES_MISSING",
                "failure_code": "alternative_evidence.matter_selection_policy_retired",
                "explanation": RETIRED_SELECTION_EXPLANATION,
                "next_action": "REINSTALL_THE_AUTHORITY_UNDER_THE_INTEGRATED_SELECTION",
                "next_requests": {"workspace": {"operation": "WORKSPACE_SHOW"}},
                "setup": evidence_setup(self.workspace, rebind=True),
                "matter_selection": self.evidence_policy.matter_selection_view(),
                "claim": "No Task, acquisition or model work performed.",
            }
        now = self.clock()
        coverage = self.resolve_coverage(chosen, evidence_as_of=now, prepare_only=True)
        resolved, run = coverage.resolved, coverage.run
        scope = resolved.scope
        resources = adapter.resources
        recorded = tuple(
            value
            for value in resources.recorded_documents
            if value.entity_id in scope.ordered_entity_ids
        )
        # What this preview describes, captured so that submitting it later --
        # after the clock has moved -- is the same preparation and not a new one
        # as of the moment of submission. The packing is not part of it: the
        # preparation reads each holding's filing index and packs from that.
        request = self.evidence_policy.run_request(run)
        selector_fields = chosen.request_fields()
        next_requests: dict[str, dict[str, str]] = {}
        prepared_task_id: str | None = None
        intent = coverage_intent_hash(run, policy=request)
        units_body: list[dict[str, object]] = []
        units_prepared = 0
        for unit in run.units:
            found = adapter.completed_unit(
                unit.preparation_intent_hash, now=now, evidence_as_of=unit.request.evidence_as_of
            )
            if found is not None:
                units_prepared += 1
                next_requests[f"packet_{unit.unit_id}"] = {
                    "operation": "EVIDENCE_PACKET",
                    **selector_fields,
                    "task_id": str(found[0]),
                    **({} if found[1] is None else {"evidence_unit_id": found[1]}),
                }
            units_body.append(
                {
                    "unit_id": unit.unit_id,
                    "ordered_entity_ids": list(unit.ordered_entity_ids),
                    "request_hash": unit.request.request_hash,
                    "obligation_hash": unit.obligation.obligation_hash,
                    "preparation_intent_hash": unit.preparation_intent_hash,
                    "prepared_task_id": None if found is None else str(found[0]),
                    "prepared_unit_id": None if found is None else found[1],
                    "source_check": (
                        None
                        if found is None
                        else adapter.prepared_source_check(str(found[0]), found[1])
                    ),
                }
            )
        completed = adapter.completed_run(run, now=now)
        if completed is not None:
            prepared_task_id = str(completed.task_id)
        binding: dict[str, object] = {
            "run_hash": run.run_hash,
            "issuer_scope_hash": scope.scope_hash,
            "resource_binding_hash": run.resource_binding_hash,
            "unit_count": len(run.units),
            "unit_limit": run.unit_limit,
            "packing_rules_id": run.packing_rules_id,
        }
        inventory = adapter.source_inventory(
            run, recorded, mode=self.evidence_policy.mode, ciks=self._issuer_ciks()
        )
        floor = resources.minimum_entity_coverage
        short = units_short_of_sources(inventory, self.evidence_policy.mode, floor=floor)
        # A run no unit of which can prepare is not offered: each unit would fail as named, and
        # the ways on are the person's (V541). Under official acquisition, the run reads the index.
        source_ready = (
            len(short) < len(run.units)
            if self.evidence_policy.mode is AlternativeEvidenceMode.RECORDED
            else resources.live_source is not None
        )
        coverage_body: dict[str, object] = {
            "unit_limit": run.unit_limit,
            "unit_count": len(run.units),
            "units_prepared": units_prepared,
            "units": units_body,
            "units_short_of_sources": short,
            **self._packing_view(run, scope),
        }
        ways: dict[str, object] = {}
        if short:
            ways = {
                # Said before any run, never learnt from a coverage that failed (V445, V541).
                "source_limit": (
                    "This Host reads only the documents the workspace holds (RECORDED), and a "
                    "unit is prepared only when enough of its issuers hold one for the installed "
                    f"floor of {floor:.0%}: {len(short)} of {len(run.units)} units fall short and "
                    "would fail with the codes `coverage.units_short_of_sources` names, beside "
                    "the issuers without a source. Official SEC acquisition, the person's "
                    "decision, reads every holding's filing index at one cutoff; a package "
                    "covers at most one unit (`source_ways`)."
                ),
                "source_ways": source_ways(
                    self.workspace,
                    entities=run.units[0].ordered_entity_ids if len(run.units) == 1 else None,
                    book=chosen,
                ),
            }
            if not source_ready:
                # The book cannot be reviewed under these sources: said, with its ways on.
                ways |= {
                    "failure_code": "alternative_evidence.book_sources_short",
                    "next_action": "ASK_FOR_OFFICIAL_ACQUISITION_OR_A_COVERING_PACKAGE",
                }
        network_refusal = self._source_acquisition_refusal(evidence_as_of=run.evidence_as_of)
        if network_refusal is not None:
            envelope, _goal, _plan = coverage_run_task_contract(run=run, prepare_only=True)
            reusable = (
                completed is not None
                or self._refresh_in_flight(coverage, identity=envelope.input_hash) is not None
            )
            if not reusable:
                source_ready = False
                ways |= {
                    "failure_code": network_refusal.failure_code,
                    "detail": network_refusal.detail,
                    "network_access": network_refusal.network_access,
                    "source_network_access": network_refusal.source_network_access,
                    "source_ways": network_refusal.source_ways,
                }
                next_requests.update(network_refusal.next_requests or {})
        if source_ready:
            next_requests["prepare"] = {
                "operation": "EVIDENCE_PREPARE",
                **selector_fields,
                "evidence_as_of": run.evidence_as_of.isoformat(),
                "preparation_binding_hash": intent,
            }
        units_word = "unit" if len(run.units) == 1 else "units"
        return {
            "status": "EVIDENCE_PREPARATION_READY"
            if source_ready
            else "EVIDENCE_PREREQUISITES_MISSING",
            "book": asdict(self._book_projection(resolved)),
            "scope": scope.model_dump(mode="json"),
            "coverage": coverage_body,
            "source_mode": self.evidence_policy.mode.value,
            "approved_source_families": list(self.evidence_policy.approved_source_families),
            "recorded_candidate_count": len(recorded),
            "source_inventory": inventory,
            **ways,
            "source_candidates": [
                {
                    "entity_id": value.entity_id,
                    "revision": value.revision,
                    "available_at": value.available_at.isoformat(),
                    "document_type": value.document_type,
                }
                for value in recorded
            ],
            "source_work": "RECORDED_LOCAL_READ"
            if self.evidence_policy.mode is AlternativeEvidenceMode.RECORDED
            else "EXPLICITLY_ADMITTED_OFFICIAL_ACQUISITION",
            "matter_selection": self.evidence_policy.matter_selection_view(),
            "managed_model_required": False,
            "retrieval": retrieval_recipe_view(adapter.runtime.retrieval.index_spec),
            "work_estimate": (
                f"One Task of {len(run.units)} bounded source/canonicalization/retrieval "
                f"{units_word} of at most {run.unit_limit} issuers, in priority order; "
                "no analyst or CRO inference. Candidate count is not accepted document "
                "count; time and size estimates are unavailable. The matter selection "
                "reads under one per-session allowance and leaves its pending scope "
                "explicit; see matter_selection."
            ),
            "evidence_as_of": run.evidence_as_of.isoformat(),
            "acquisition_deadline": request.acquisition_deadline.isoformat(),
            "preparation_binding_hash": intent,
            "preparation_binding": binding,
            "prepared_task_id": prepared_task_id,
            # A one-unit book's check is the book's check; a wider book's are
            # per unit, in `coverage.units`.
            "source_check": units_body[0]["source_check"] if len(units_body) == 1 else None,
            "admission": self._admission_view(),
            "campaign": self._campaign_view(),
            "reuse": adapter.reuse_view(
                [
                    (str(unit["prepared_task_id"]), cast("str | None", unit["prepared_unit_id"]))
                    for unit in units_body
                    if unit["prepared_task_id"] is not None
                ]
            ),
            "next_requests": with_analyst_bundles(next_requests),
            "claim": (
                "Preview performs no acquisition and grants no source/network permission. "
                "Preparation revalidates current authority and dates at submission; the "
                "captured intent is reused while its packet reads, and a request without "
                "it prepares as of the moment it is submitted."
            ),
        }

    def _packing_view(
        self, run: AlternativeEvidenceCoverageRun, scope: PortfolioIssuerScope
    ) -> dict[str, object]:
        """How the run's units were packed: each issuer's logical source count
        and its basis, the rule, and the capacity every unit's selections were
        packed under -- so a reader sees that an issuer's filings were decided
        by its own policy, never by how many issuers share its unit."""

        basis = self._source_counts(scope, evidence_as_of=run.evidence_as_of).basis
        return {
            "packing": {
                "rules_id": run.packing_rules_id,
                "document_capacity_per_unit": ADMITTED_DOCUMENT_CAPACITY,
                "issuer_limit_per_unit": run.unit_limit,
                "source_counts": dict(run.source_counts),
                "source_count_basis": basis,
                "nothing_filed": list(run.nothing_filed),
                "carried": list(run.carried),
                "read_earlier": len(run.read_filings),
                "rule": (
                    "An issuer's recent filings are selected under its own policy budget "
                    "before the units are packed; the units are packed heaviest holding first "
                    "under the issuer limit and the admitted document set, and a holding whose "
                    "filing index holds nothing in the window is named, not packed. A filing an "
                    "earlier analysis read is not read again while it stays in the window: its "
                    "findings carry, and a holding with nothing new is carried, not packed. A "
                    "unit whose selections still exceed the set defers filings by name at "
                    "acquisition."
                ),
            }
        }

    def refresh_evidence(
        self,
        *,
        dispatcher: LocalBackgroundDispatcher,
        selector: BookSelector | None = None,
        evidence_as_of: datetime | None = None,
        prepare_only: bool = False,
        preparation_binding_hash: str | None = None,
    ) -> ReviewOutcome:
        """`Refresh evidence`, all the way through: book, scope, request, Task.

        A preparation names its cutoff. Given none, the cutoff is the moment of
        submission and the request is new. Given the cutoff and binding a
        preview handed out, the request is that captured intent: a completed
        preparation for it is reused while its packet still reads, whatever
        the clock says now; a binding that no longer describes this workspace
        (scope, source package, policy or permission moved) is refused by the
        part that moved.
        """
        chosen = self.default_selector(selector)
        if chosen is None:
            return ReviewOutcome(
                disposition="REFUSED_NO_BOOK_TO_REVIEW",
                detail="This workspace holds no sealed book to review yet.",
            )
        if not self.has_evidence_authority:
            return ReviewOutcome(
                disposition="REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY",
                detail="No issuer registry and listing authority are admitted for this workspace.",
            )
        if not prepare_only and not self.model_authority_admitted:
            return ReviewOutcome(
                disposition="REFUSED_MODEL_AUTHORITY_NOT_ADMITTED",
                failure_code="evidence_review.model_authority_not_admitted",
                detail=(
                    "The product runs no model of its own: an agent answers the Analyst's "
                    "packet through its bundle (agent bundle-prepare). Saved evidence and "
                    "reviews remain readable."
                ),
                # The request the words name, the book filled (V305).
                next_requests={
                    "analyst_bundle": {
                        "operation": "AGENT_BUNDLE_PREPARE",
                        "agent_role": "ANALYST",
                        **chosen.request_fields(),
                    }
                },
            )
        if self.evidence_task_adapter is None or self.evidence_publications is None:
            return ReviewOutcome(
                disposition="REFUSED_NO_ADMITTED_EVIDENCE_RUNTIME",
                detail=(
                    "This workspace has no admitted evidence runtime, so a refresh has "
                    "nothing to acquire with."
                ),
            )
        if matter_selection_retired(self.evidence_policy.matter_selection):
            return ReviewOutcome(
                disposition="REFUSED_MATTER_SELECTION_RETIRED",
                detail=RETIRED_SELECTION_EXPLANATION,
                failure_code="alternative_evidence.matter_selection_policy_retired",
            )
        now = self.clock()
        try:
            coverage = self.resolve_coverage(
                chosen,
                evidence_as_of=evidence_as_of or now,
                prepare_only=prepare_only,
                read_inventory=True,
            )
        except PortfolioEvidenceReviewError as error:
            refusal = self._source_acquisition_refusal(evidence_as_of=evidence_as_of or now)
            if refusal is None or str(error) != refusal.failure_code:
                raise
            return refusal
        preview = {
            "operation": "EVIDENCE_PREVIEW",
            **chosen.request_fields(),
        }
        return self._refresh_coverage(
            dispatcher=dispatcher,
            coverage=coverage,
            preparation_binding_hash=preparation_binding_hash,
            preview=preview,
            now=now,
            selector=chosen,
        )

    def _refresh_in_flight(
        self, coverage: ResolvedCoverage, *, identity: str
    ) -> ReviewOutcome | None:
        """The refresh already in flight for this exact run, when one is.

        Two callers asking the same question -- a second click, a second
        session submitting the same captured intent -- get the one Task that
        is queued or running for it, not a second acquisition of the same
        bodies behind it. The same question means the same Task input (run,
        cutoff, resource binding, purpose). A request for another cutoff,
        another package or another purpose -- or a single preparation queued
        before every book was a run -- is another question: it is not
        answered by the active Task, and the existing queue admits or refuses
        it as busy. Decided last, after every reuse and refusal of the request
        itself, so a completed preparation for the exact run is still
        `REUSED_EXACT`.
        """

        active = self._active_refresh_task(scope=coverage.resolved.scope)
        if active is None:
            return None
        adapter = self.evidence_task_adapter
        if adapter is None or adapter.run_of(active) is None:
            return None
        if active.input.input_hash != identity:
            return None
        projection = self.session.task_control_registry.safe_projection(active.task_id)
        return ReviewOutcome(
            disposition="REUSED_IN_FLIGHT",
            detail=(
                "This exact preparation is already "
                f"{projection.lifecycle.value.lower().replace('_', ' ')}; this request "
                "joins it instead of acquiring the same sources twice."
            ),
            task_id=active.task_id,
            lifecycle=projection.lifecycle.value,
            evidence_as_of=coverage.run.evidence_as_of,
        )

    def _refresh_coverage(
        self,
        *,
        dispatcher: LocalBackgroundDispatcher,
        coverage: ResolvedCoverage,
        preparation_binding_hash: str | None,
        preview: dict[str, str],
        now: datetime,
        selector: BookSelector,
    ) -> ReviewOutcome:
        """Admit a book's run, or reuse its completion.

        The captured intent is what a person approved (`coverage_intent_hash`:
        the book, the cutoff, the policy, the package and the permissions) --
        what a preview hands out and a later submission names; the packing is
        the Host's, from the filing index read at this cutoff. A run whose every unit completed
        and still reads is reused whole; a run some of whose units failed or
        expired is admitted again, and the adapter carries every unit that
        did complete -- in any run, or in a single preparation admitted
        before C2 -- into it instead of preparing it twice.
        """

        adapter = self.evidence_task_adapter
        assert adapter is not None
        run = coverage.run
        if (
            (coverage.prepare_only or not run.units)
            and preparation_binding_hash is not None
            and preparation_binding_hash
            != (coverage_intent_hash(run, policy=self.evidence_policy.run_request(run)))
        ):
            return self._run_binding_refusal(preparation_binding_hash, run=run, preview=preview)
        if not run.units:
            # Nothing is left to read: every holding's filings in the window were
            # read earlier, or it filed nothing. The run is sealed where every run
            # is -- the review reads it -- and nothing executes; the last review
            # carries forward when nothing it read has changed (W3).
            adapter.admit_run(run)
            return self._nothing_new(run, selector=selector, dispatcher=dispatcher)
        deadline = run.units[0].request.acquisition_deadline
        if coverage.prepare_only:
            completed = adapter.completed_run(run, now=now)
            if completed is not None:
                return ReviewOutcome(
                    disposition="REUSED_EXACT",
                    detail=f"This preparation of {len(run.units)} units is already complete; "
                    f"every packet reads as of {run.evidence_as_of.isoformat()}.",
                    task_id=completed.task_id,
                    lifecycle=completed.lifecycle.value,
                    evidence_as_of=run.evidence_as_of,
                )
            if adapter.run_expired(run, now=now):
                return ReviewOutcome(
                    disposition="REFUSED_PREPARATION_EXPIRED",
                    detail="This preparation completed but its packets have expired; preview "
                    "again to prepare as of a new cutoff.",
                    evidence_as_of=run.evidence_as_of,
                    failure_code="alternative_evidence.brief_source_stale",
                    next_requests={"preview": preview},
                )
            if now > deadline:
                return ReviewOutcome(
                    disposition="REFUSED_ACQUISITION_WINDOW_CLOSED",
                    detail="This intent was never prepared and its acquisition window has "
                    "closed; preview again to prepare as of a new cutoff.",
                    evidence_as_of=run.evidence_as_of,
                    failure_code="alternative_evidence.acquisition_deadline_exceeded",
                    next_requests={"preview": preview},
                )
        envelope, _goal, _plan = coverage_run_task_contract(
            run=run, prepare_only=coverage.prepare_only
        )
        in_flight = self._refresh_in_flight(coverage, identity=envelope.input_hash)
        if in_flight is not None:
            return in_flight
        network_refusal = self._source_acquisition_refusal(evidence_as_of=run.evidence_as_of)
        if network_refusal is not None:
            return network_refusal
        adapter.admit_run(run)
        command = AlternativeEvidenceRefreshCommand(
            application=self, run=run, prepare_only=coverage.prepare_only
        )
        submitted = _submitted(dispatcher.submit(command))
        return replace(submitted, evidence_as_of=run.evidence_as_of)

    def continue_evidence(
        self,
        *,
        dispatcher: LocalBackgroundDispatcher,
        selector: BookSelector | None,
        task_id: UUID,
        unit_id: str | None,
        continuation_of: str,
        continuation_spans: str,
        session_limit: int,
        window_limit: int,
    ) -> ReviewOutcome:
        """Admit one exact bounded source-reading continuation for a prepared packet.

        `Continue reading`: one more bounded matter-reading session over a
        prepared packet of this book, admitted as its own Task under the
        packet's request, obligation and admission -- source reading only,
        no search, no typed extraction, no model.

        The packet is named by the receipt and span set it sealed: a stale or
        unrelated receipt is refused, as is a packet with nothing pending, a
        chain whose declared cumulative limits are exhausted or differ from
        the ones it was continued under, and an expired packet. The same
        continuation asked twice is the completed Task, not a second session:
        the input is the identity. A delivery part of a packet is never this
        request.
        """
        chosen = self._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        adapter = self.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        now = self.clock()
        task = self.session.task_control_registry.task(task_id)
        authority = adapter.unit_authority(task, unit_id)
        request = authority.request
        resolved = self.resolve_book(chosen)
        units = self._unit_obligations(resolved.scope, evidence_as_of=request.evidence_as_of)
        expected = units.get(tuple(request.ordered_entity_ids))
        if expected is None or authority.obligation != expected[1]:
            raise PortfolioEvidenceReviewError("alternative_evidence.prepared_book_scope_mismatch")
        try:
            _cutoff, expires_at = adapter.preparation_window(task, unit_id)
        except ValueError as error:
            if str(error) != "alternative_evidence.preparation_superseded":
                raise
            # Prepared under an acquisition, canonicalization, retrieval or
            # allocation contract this workspace has since superseded: what it
            # read stays readable as history, but no current plan continues it.
            return ReviewOutcome(
                disposition="REFUSED_PREPARATION_SUPERSEDED",
                detail="This packet was prepared under a contract this workspace has since "
                "superseded; what it read stays readable by its handles, but its reading "
                "plan is not continued. Preview and prepare again under the current contract.",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.preparation_superseded",
            )
        if matter_selection_retired(request.matter_selection):
            return ReviewOutcome(
                disposition="REFUSED_MATTER_SELECTION_RETIRED",
                detail=RETIRED_SELECTION_EXPLANATION,
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.matter_selection_policy_retired",
            )
        if now > expires_at:
            return ReviewOutcome(
                disposition="REFUSED_PREPARATION_EXPIRED",
                detail="This packet has expired; more of its source cannot be read under "
                "its cutoff. Preview again to prepare as of a new cutoff.",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.brief_source_stale",
            )
        receipt_hash, span_set_hash = adapter.prepared_receipt_identity(task_id, unit_id=unit_id)
        if (receipt_hash, span_set_hash) != (continuation_of, continuation_spans):
            return ReviewOutcome(
                disposition="REFUSED_CONTINUATION_PRIOR_MISMATCH",
                detail="The named receipt and span set are not the ones this packet sealed; "
                "a continuation names the packet's own continuation_request.",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.litigation_continuation_prior_mismatch",
            )
        receipt = adapter.runtime.artifacts.load(
            "retrieval-access-receipts", receipt_hash, AlternativeEvidenceRetrievalAccessReceipt
        )
        matters = receipt.litigation_matters
        if matters is None or (
            matters.pending_windows == 0 and not any(routed_pending_work(receipt).values())
        ):
            return ReviewOutcome(
                disposition="REFUSED_CONTINUATION_NOTHING_PENDING",
                detail="This packet's reading plan has no pending window, no sealed "
                "candidate and no table page left; there is nothing to continue.",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.litigation_continuation_nothing_pending",
            )
        if matters.historical_reference_needs:
            return ReviewOutcome(
                disposition="REFUSED_CONTINUATION_PRIOR_INCOMPATIBLE",
                detail="This packet's matter record was sealed before references carried "
                "a target state; what it delivered reads under its own contract, but no "
                "current plan continues it. Preview and prepare again.",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.litigation_continuation_prior_incompatible",
            )
        if matters.plan_hash is None:
            return ReviewOutcome(
                disposition="REFUSED_CONTINUATION_PLAN_UNIDENTIFIED",
                detail="This packet was sealed before reading plans carried an identity, so "
                "what it read cannot be proved against today's plan; preview and prepare "
                "again to read its scope under an identified plan.",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.litigation_plan_changed",
            )
        cumulative = (
            matters.plan_offset + matters.read_windows
            if matters.cumulative_read_windows is None
            else matters.cumulative_read_windows
        )
        if matters.session_index > 1 and (matters.session_limit, matters.window_limit) != (
            session_limit,
            window_limit,
        ):
            return ReviewOutcome(
                disposition="REFUSED_CONTINUATION_LIMITS_CHANGED",
                detail=f"This chain was continued under {matters.session_limit} sessions and "
                f"{matters.window_limit} windows; a continuation under other limits is another "
                "request, not this chain's.",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.litigation_continuation_limits_changed",
            )
        if matters.session_index + 1 > session_limit or cumulative >= window_limit:
            # An exhausted allowance is not a read plan: every channel the
            # plan still owes is named, and what no session could read too.
            channels = routed_pending_work(receipt)
            remainders = reading_remainders(receipt)
            return ReviewOutcome(
                disposition="REFUSED_CONTINUATION_BUDGET_EXHAUSTED",
                detail=f"The declared allowance ({session_limit} sessions, {window_limit} "
                f"windows) is used up after {matters.session_index} session(s) and "
                f"{cumulative} windows; {matters.pending_windows} "
                f"window(s) / {matters.pending_source_bytes} source bytes, "
                f"{channels['candidates']} sealed candidate(s), "
                f"{channels['partial_table_pages']} table(s) delivered in part and "
                f"{channels['tables_not_dealt']} table(s) not dealt remain pending and unread"
                + (
                    "; beyond the plan: "
                    + ", ".join(f"{value} {name}" for name, value in remainders.items())
                    if remainders
                    else ""
                )
                + ".",
                evidence_as_of=request.evidence_as_of,
                failure_code="alternative_evidence.litigation_continuation_budget_exhausted",
            )
        command = AlternativeEvidenceRefreshCommand(
            application=self,
            obligation=authority.obligation,
            request=request,
            admission=authority.admission,
            continuation=EvidenceContinuation(
                prepared_task_id=task_id,
                prepared_unit_id=unit_id,
                continuation_of=continuation_of,
                continuation_spans=continuation_spans,
                session_limit=session_limit,
                window_limit=window_limit,
            ),
        )
        envelope, _goal, plan = command.contract()
        completed = self._completed_task(envelope.input_hash, plan.plan_hash)
        if completed is not None:
            return ReviewOutcome(
                disposition="REUSED_EXACT",
                detail="This continuation already completed; its packet is the successor of "
                "the named receipt.",
                task_id=completed.task_id,
                lifecycle=completed.lifecycle.value,
                evidence_as_of=request.evidence_as_of,
            )
        submitted = _submitted(dispatcher.submit(command))
        return replace(submitted, evidence_as_of=request.evidence_as_of)

    def packet_requests(
        self, selector: BookSelector | None, *, task_id: UUID
    ) -> dict[str, dict[str, str]]:
        """Read the exact packet requests answered by a preparation task or its coverage units.

        The packet request(s) a preparation Task answers: one for a single
        preparation, one per unit for a coverage run.
        """
        chosen = self.default_selector(selector)
        if chosen is None:
            return {}
        fields = chosen.request_fields()
        adapter = self.evidence_task_adapter
        task = self.session.task_control_registry.task(task_id)
        if adapter is not None and task.task_kind != adapter.task_kind:
            # A review carried forward is no preparation: it has no packet (V434).
            return {}
        run = None if adapter is None else adapter.run_of(task)
        if run is None:
            packet = {"operation": "EVIDENCE_PACKET", **fields, "task_id": str(task_id)}
            return with_analyst_bundles({"packet": packet})
        return with_analyst_bundles(
            {
                f"packet_{unit.unit_id}": {
                    "operation": "EVIDENCE_PACKET",
                    **fields,
                    "task_id": str(task_id),
                    "evidence_unit_id": unit.unit_id,
                }
                for unit in run.units
            }
        )

    def _run_binding_refusal(
        self, claimed: str, *, run: AlternativeEvidenceCoverageRun, preview: dict[str, str]
    ) -> ReviewOutcome:
        """Name what moved between the captured run and this workspace now."""

        adapter = self.evidence_task_adapter
        assert adapter is not None
        moved = adapter.binding_changes(claimed, run, policy=self.evidence_policy)
        detail = (
            "The captured preparation binding no longer describes this workspace"
            + (": " + ", ".join(moved) + " moved" if moved else "")
            + "; preview again."
        )
        return ReviewOutcome(
            disposition="REFUSED_PREPARATION_BINDING_CHANGED",
            detail=detail,
            evidence_as_of=run.evidence_as_of,
            failure_code="alternative_evidence.preparation_binding_changed",
            next_requests={"preview": preview},
        )

    def _completed_task(self, input_hash: str, plan_hash: str) -> TaskRecord | None:
        """The SUCCEEDED Task admitted for exactly this input and plan, if any."""

        for task in self.session.task_control_registry.tasks():
            if (
                task.lifecycle is TaskLifecycle.SUCCEEDED
                and task.input.input_hash == input_hash
                and task.plan.plan_hash == plan_hash
            ):
                return task
        return None

    def submit_analysis(
        self,
        *,
        dispatcher: LocalBackgroundDispatcher,
        selector: BookSelector | None,
        task_id: UUID,
        context_hash: str,
        answer: object,
        caller: str,
        unit_id: str | None = None,
        read_files: tuple[str, ...] | None = None,
        agent_run: AgentRun | None = None,
    ) -> ReviewOutcome | dict[str, object]:
        """Screen and bind one explicit Analyst answer to its exact prepared packet.

        Admit one answer to one prepared packet: screened item by item,
        refused with every problem in plain words, or its findings bound.
        """
        if caller not in {"HUMAN", "EXTERNAL_AUTOMATION"}:
            raise PortfolioEvidenceReviewError(
                "alternative_evidence.external_submission_entry_required"
            )
        chosen = self._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        if self.evidence_task_adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        # The packet this request verified serves the screening and the admission's check;
        # the Task verifies it again at execution, since the materials may change between.
        packet, context, *_ = self._book_packet(chosen, task_id=task_id, unit_id=unit_id)
        if context["analysis_context_hash"] != context_hash:
            raise PortfolioEvidenceReviewError(
                "alternative_evidence.external_analysis_binding_changed"
            )
        adapter = self.evidence_task_adapter
        assert adapter is not None
        entity_ids = packet.request.ordered_entity_ids
        screened = screen_analyst_answer(
            answer, entity_ids=entity_ids, aliases=span_aliases(packet.spans, entity_ids)
        )
        binding: dict[str, object] = {
            "role": "alternative_evidence.analyst",
            "task_id": str(task_id),
            "unit_id": unit_id,
            "analysis_context_hash": context_hash,
        }
        with self._answer_turn(binding):
            record = self._next_answer(
                binding,
                answer,
                screened,
                read_files=read_files,
                agent_run=agent_run,
            )
            if record.verdict is AnswerVerdict.CORRECT:
                self._file_answer(record)
                return _answer_correction("alternative_evidence.answer_problems", record)
            accepted = accepted_analyst_answer(screened, entity_ids=entity_ids)
            dropped = screened.problems if record.verdict is AnswerVerdict.DONE else ()
            submitted = SubmittedEvidenceAnalysis(
                prepared_task_id=task_id,
                prepared_unit_id=unit_id,
                packet_hash=cast(str, context["packet_hash"]),
                analysis_policy_hash=cast(str, context["analysis_policy_hash"]),
                decision_policy_hash=cast(str, context["decision_policy_hash"]),
                answer=accepted,
                dropped=dropped,
                actor_submission=seal_actor_submission(
                    actor_kind=ActorKind(caller),
                    actor_id="local-research-external"
                    if caller == "EXTERNAL_AUTOMATION"
                    else "local-web-human",
                    submission_hash=canonical_hash(accepted.model_dump(mode="json")),
                ),
            )
            adapter.validate_submission(submitted, now=self.clock(), packet=packet)
            prepared = self.session.task_control_registry.task(task_id)
            if unit_id is None:
                # The packet may be a continuation's: the answer is admitted under
                # its request, obligation and admission, never as another session.
                command = AlternativeEvidenceRefreshCommand.recover(application=self, task=prepared)
                command.prepare_only = False
                command.continuation = None
                command.submitted_analysis = submitted
            else:
                # One unit's answer is admitted as the request, question and
                # admission that unit was prepared under, so the same answer to
                # the same packet is the same Task input wherever it is submitted.
                authority = adapter.unit_authority(prepared, unit_id)
                command = AlternativeEvidenceRefreshCommand(
                    application=self,
                    obligation=authority.obligation,
                    request=authority.request,
                    admission=authority.admission,
                    prepare_only=False,
                    submitted_analysis=submitted,
                )
            envelope, _, plan = command.contract()
            published = self._completed_task(envelope.input_hash, plan.plan_hash)
            if published is not None:
                publication = adapter.published_analysis(published.task_id, now=self.clock())
                return {
                    "disposition": "REUSED_EXACT",
                    "task_id": None,
                    "publication_task_id": str(published.task_id),
                    "analysis_publication_hash": publication.publication.publication_hash,
                    **({} if unit_id is None else {"evidence_unit_id": unit_id}),
                    "external_host_usage": "UNAVAILABLE",
                    "answer": _answer_view(
                        record,
                        self.accepted_answer_delivery(
                            record, accepted.model_dump(mode="json"), task_id=published.task_id
                        ),
                        record_filed=False,
                    ),
                }
            submission = dispatcher.submit(command)
            newly_filed = False
            if submission.admitted:
                newly_filed = self._file_answer(record)
            delivery = (
                self.accepted_answer_delivery(
                    record,
                    accepted.model_dump(mode="json"),
                    task_id=submission.task_id,
                    newly_filed=newly_filed,
                )
                if submission.admitted and submission.task_id is not None
                else None
            )
            outcome = replace(
                _submitted(submission),
                answer=_answer_view(record, delivery, record_filed=submission.admitted),
            )
            # The unit is the answer's own field, as an answer already published says it; a next
            # request is one the Host accepts (V449).
            return replace(outcome, evidence_unit_id=unit_id)

    def drive_evidence_task(
        self, *, task_id: UUID | None = None, expected_task_hash: str | None = None
    ) -> None:
        """Require an installed evidence adapter and drive its exact admitted task.

        Args:
            task_id: Optional exact task selected by the retained runner.
            expected_task_hash: Optional optimistic task identity.

        Raises:
            PortfolioEvidenceReviewError: Evidence task adapter is absent.
        """
        adapter = self.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_adapter_absent")
        self._run(adapter, task_id=task_id, expected_task_hash=expected_task_hash)

    # ----------------------------------------------------------------- review

    def review_task_contract(
        self, *, dossier: PortfolioReviewDossier, actor: PortfolioReviewActor
    ) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
        """Compile exact dossier/actor review keys into durable task contracts.

        Args:
            dossier: Exact prepared review dossier.
            actor: Explicit review actor with bound process/response authority.

        Returns:
            Task envelope, goal and plan from the deterministic review contract owner.
        """
        return portfolio_review_task_contract(
            review_key_payload=self.review_key_payload(dossier=dossier, actor=actor)
        )

    def review_key_payload(
        self, *, dossier: PortfolioReviewDossier, actor: PortfolioReviewActor
    ) -> dict[str, object]:
        """Bind dossier, decision policy, actor authority and explicit prepared answer semantics.

        Args:
            dossier: Exact prepared review dossier.
            actor: Review actor with retained process identity.

        Returns:
            Exact review key payload including carried/durable submission fields when declared.

        Raises:
            PortfolioEvidenceReviewError: A durable submission lacks a concrete prepared
                PortfolioReviewAnswer.
        """
        prepared = getattr(actor, "answer", None)
        submission_hash = None
        kind = getattr(actor, "actor_kind", None)
        if prepared is not None and kind is not None and kind is not ActorKind.INSTALLED_AGENT:
            submission_hash = str(canonical_hash(prepared.model_dump(mode="json")))
        payload = portfolio_review_key_payload(
            dossier=dossier,
            decision_policy_hash=self._decision_policy().binding_hash,
            typed_user_authority=self.typed_user_authority,
            actor_kind=str(getattr(actor, "actor_kind", "INSTALLED_AGENT")),
            actor_id=str(getattr(actor, "actor_id", "installed-agent")),
            process_binding_hash=actor.process_binding_hash,
            response_schema_hash=self.response_schema_hash,
            submission_hash=submission_hash,
        )
        if isinstance(actor, CarriedPortfolioReviewActor):
            payload["carried_from"] = actor.receipt.receipt_hash
        if getattr(actor, "durable_submission", False):
            if not isinstance(prepared, PortfolioReviewAnswer):
                raise PortfolioEvidenceReviewError("chief_risk_officer.prepared_submission_invalid")
            payload["prepared_answer"] = prepared.model_dump(mode="json")
            dropped = tuple(getattr(actor, "dropped", ()))
            if dropped:
                payload["prepared_dropped"] = [value.model_dump(mode="json") for value in dropped]
        return payload

    def recover_review_actor(self, task: TaskRecord) -> PortfolioReviewActor:
        """Resolve each queued review's own typed answer, never the last queued actor."""
        payload = task.input.payload
        if "carried_from" in payload:
            receipt = self.artifacts.load(
                "cro-review-receipts", str(payload["carried_from"]), PortfolioReviewReceipt
            )
            return CarriedPortfolioReviewActor(
                receipt,
                self.artifacts.load(
                    "cro-review-dossiers", receipt.dossier_hash, PortfolioReviewDossier
                ),
            )
        if "prepared_submission" in payload:
            # Queued before the answer format and never executed: its whole
            # assessment is no longer admitted; the reviewer answers again.
            raise PortfolioEvidenceReviewError("chief_risk_officer.submission_format_retired")
        if "prepared_answer" in payload:
            kind = ActorKind(str(payload["actor_kind"]))
            if kind not in {ActorKind.HUMAN, ActorKind.EXTERNAL_AUTOMATION}:
                raise PortfolioEvidenceReviewError("chief_risk_officer.submitted_actor_invalid")
            answer = PortfolioReviewAnswer.model_validate(payload["prepared_answer"])
            if canonical_hash(answer.model_dump(mode="json")) != payload.get("submission_hash"):
                raise PortfolioEvidenceReviewError(
                    "chief_risk_officer.submission_identity_mismatch"
                )
            return SubmittedPortfolioReviewActor(
                answer=answer,
                dropped=tuple(
                    AnswerProblem.model_validate(value)
                    for value in cast(list[object], payload.get("prepared_dropped", []))
                ),
                actor_kind=kind,
                actor_id=str(payload["actor_id"]),
                durable_submission=True,
            )
        if self.review_actor is None:
            raise PortfolioEvidenceReviewError("chief_risk_officer.review_actor_unavailable")
        return self.review_actor

    def _review_selector(self, selector: BookSelector | None) -> BookSelector | ReviewOutcome:
        chosen = self.default_selector(selector)
        if chosen is None:
            return ReviewOutcome(
                disposition="REFUSED_NO_BOOK_TO_REVIEW",
                # What names a book, where a request that named a Task or a unit alone read
                # as an empty workspace; an experiment's book is no default (V242), and the
                # history read is the way to one (V295).
                detail=_NO_BOOK_NAMED,
                next_requests={"history": {"operation": "RESEARCH_HISTORY"}},
            )
        if not self.has_evidence_authority:
            return ReviewOutcome(
                disposition="REFUSED_NO_ADMITTED_EVIDENCE_AUTHORITY",
                detail="No issuer registry and listing authority are admitted for this workspace.",
            )
        return chosen

    def review(
        self,
        *,
        dispatcher: LocalBackgroundDispatcher,
        selector: BookSelector | None = None,
        actor: PortfolioReviewActor | None = None,
        evidence_as_of: datetime | None = None,
    ) -> ReviewOutcome:
        """Resolve reuse first, then admit one Task. Never the other way round."""
        chosen = self._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        if not self.model_authority_admitted and actor is None:
            return ReviewOutcome(
                disposition="REFUSED_MODEL_AUTHORITY_NOT_ADMITTED",
                failure_code="evidence_review.model_authority_not_admitted",
                detail=(
                    "The product runs no model of its own: an agent answers the CRO's "
                    "dossier through its bundle (agent bundle-prepare). Saved reviews "
                    "remain readable."
                ),
                # The request the words name, the book filled (V305).
                next_requests={
                    "cro_bundle": {
                        "operation": "AGENT_BUNDLE_PREPARE",
                        "agent_role": "CRO",
                        **chosen.request_fields(),
                    }
                },
            )
        reviewer = actor or self.review_actor
        if reviewer is None:
            return ReviewOutcome(
                disposition="REFUSED_NO_ADMITTED_REVIEW_ACTOR",
                detail=(
                    "No typed actor authority is admitted for this workspace, so there is "
                    "nobody this review could be attributed to."
                ),
            )
        dossier = self._resolve_review_dossier(chosen, evidence_as_of=evidence_as_of)
        if isinstance(dossier, ReviewOutcome):
            return dossier
        carried = self._carry_forward(dossier, dispatcher=dispatcher)
        if carried is not None:
            return carried
        return self._submit_review(dossier, reviewer, dispatcher)

    def _carry_forward(
        self, dossier: PortfolioReviewDossier, *, dispatcher: LocalBackgroundDispatcher
    ) -> ReviewOutcome | None:
        """The last fresh review of this dossier's basis, carried forward as of
        its cutoff with no model call (W3), or nothing: the holdings, their
        bands, the findings and the open issues are what the CRO last assessed,
        so its assessment stands and is routed again on this dossier."""

        basis = review_basis(dossier, decision_policy_hash=self._decision_policy().binding_hash)
        prior = self.review_publications.find_for_basis(basis, open_issues=dossier.open_issues)
        if (
            prior is None
            or prior.receipt.answer is None
            or prior.dossier.dossier_hash == dossier.dossier_hash
        ):
            return None
        outcome = self._submit_review(
            dossier, CarriedPortfolioReviewActor(prior.receipt, prior.dossier), dispatcher
        )
        return replace(
            outcome,
            detail=(
                "Nothing the CRO read has changed since its review as of "
                f"{prior.dossier.evidence_as_of.date().isoformat()}: that review carries "
                f"forward as of {dossier.evidence_as_of.date().isoformat()}, with no model "
                "call. " + outcome.detail
            ),
        )

    def _nothing_new(
        self,
        run: AlternativeEvidenceCoverageRun,
        *,
        selector: BookSelector,
        dispatcher: LocalBackgroundDispatcher,
    ) -> ReviewOutcome:
        """What a day with nothing to read yields: the words, and the last
        review carried forward when nothing it read has changed."""

        window = self.evidence_policy.run_request(run).source_policy.sec_recent_8k_days
        if run.carried:
            outcome = ReviewOutcome(
                disposition="NOTHING_NEW",
                detail=(
                    f"No holding filed anything new with the SEC in the last {window} days: "
                    f"{len(run.carried)} keep what was read in their filings earlier and "
                    f"{len(run.nothing_filed)} filed nothing. Nothing is prepared, and the "
                    f"review reads as of {run.evidence_as_of.date().isoformat()}."
                ),
                evidence_as_of=run.evidence_as_of,
            )
        else:
            outcome = ReviewOutcome(
                disposition="NOTHING_FILED",
                detail=(
                    f"No holding filed anything with the SEC in the last {window} days, so "
                    "there is nothing to prepare. Having nothing to read is not a finding of "
                    "no risk."
                ),
                evidence_as_of=run.evidence_as_of,
            )
        dossier = self._resolve_review_dossier(selector, evidence_as_of=run.evidence_as_of)
        if isinstance(dossier, ReviewOutcome):
            return outcome
        carried = self._carry_forward(dossier, dispatcher=dispatcher)
        if carried is None:
            return replace(
                outcome,
                detail=outcome.detail
                + " What the CRO last read has changed, so the CRO reads the book again.",
            )
        return replace(
            outcome,
            detail=outcome.detail + " " + carried.detail,
            task_id=carried.task_id,
            lifecycle=carried.lifecycle,
            review=carried.review,
        )

    def _resolve_review_dossier(
        self,
        chosen: BookSelector,
        *,
        evidence_as_of: datetime | None = None,
        read_at: datetime | None = None,
    ) -> PortfolioReviewDossier | ReviewOutcome:
        """Read the dossier at the same cutoff its delivery and submission bind."""
        dossier, _cutoff = self._review_dossier_read(
            chosen, evidence_as_of=evidence_as_of, read_at=read_at
        )
        return dossier

    def _review_dossier_read(
        self,
        chosen: BookSelector,
        *,
        evidence_as_of: datetime | None = None,
        read_at: datetime | None = None,
    ) -> tuple[PortfolioReviewDossier | ReviewOutcome, datetime | None]:
        """The book's review dossier. With no run and no current analysis its cutoff is the
        time it is read: ``read_at``, the time a bundle or a first part sealed, so every later
        read of that review resolves the same dossier; the clock only for a first read (V255).
        """
        resolved = self.resolve_book(chosen)
        active = self.active_evidence_refresh(scope=resolved.scope)
        if active is not None:
            return (
                ReviewOutcome(
                    disposition="EVIDENCE_REFRESH_IN_PROGRESS",
                    detail=REFRESH_EXPLANATION,
                    task_id=active.task_id,
                    lifecycle=str(active.lifecycle.value),
                ),
                None,
            )
        try:
            selections = self._select_evidence_for_review(resolved, evidence_as_of=evidence_as_of)
        except EvidenceSelectionAmbiguous:
            return (
                ReviewOutcome(
                    disposition="REFUSED_EVIDENCE_SELECTION_AMBIGUOUS", detail=AMBIGUOUS_EXPLANATION
                ),
                None,
            )
        current = [
            (unit_id, value)
            for unit_id, value in selections.items()
            if value.disposition == "CURRENT" and value.evidence is not None
        ]
        run = self._sealed_run(resolved.scope, evidence_as_of=evidence_as_of)
        # The day the review reads as of: the run's cutoff, else the one asked
        # for, else the newest analysis read, else the time the review is read.
        cutoff = (
            run.evidence_as_of
            if run is not None
            else evidence_as_of
            or max(
                (
                    value.evidence.lineage.request.evidence_as_of
                    for _unit, value in current
                    if value.evidence is not None
                ),
                default=read_at or self.clock(),
            )
        )
        issues = self.review_publications.open_issues(
            frozenset(resolved.scope.ordered_entity_ids), as_of=cutoff
        )
        carried, lost = self._carried_readings(
            resolved.scope,
            run,
            issues=issues,
            exclude=frozenset(
                value.evidence.publication.publication_hash
                for _unit, value in current
                if value.evidence is not None
            ),
            as_of=cutoff,
        )
        if not current and not carried:
            # Nothing to read anywhere in the book: the newest analysis that is
            # not current says why, as one request always did.
            dispositions = {value.disposition for value in selections.values()}
            if "EXPIRED" in dispositions:
                return (
                    ReviewOutcome(
                        disposition="REFUSED_ALTERNATIVE_EVIDENCE_EXPIRED",
                        detail=(
                            "The matching Alternative Evidence analysis expired. "
                            "Refresh evidence before asking for a current review."
                        ),
                    ),
                    cutoff,
                )
            if "SUPERSEDED" in dispositions:
                return (
                    ReviewOutcome(
                        disposition="REFUSED_ALTERNATIVE_EVIDENCE_SUPERSEDED",
                        detail=(
                            "The matching Alternative Evidence analysis was sealed under a "
                            "superseded retrieval contract. It stays readable by its handle; "
                            "prepare evidence again before asking for a current review."
                        ),
                    ),
                    cutoff,
                )
            return (
                ReviewOutcome(
                    disposition="REFUSED_AWAITING_ALTERNATIVE_EVIDENCE",
                    detail="No current Alternative Evidence analysis answers this issuer scope.",
                ),
                cutoff,
            )
        nothing_filed = () if run is None else run.nothing_filed
        if len(selections) == 1 and not nothing_filed and not carried and not lost and not issues:
            _unit_id, selected = current[0]
            assert selected.question is not None and selected.evidence is not None
            try:
                return (
                    compile_portfolio_review_dossier(
                        inputs=self.inputs(),
                        book=resolved.book,
                        projection=resolved.projection,
                        scope=resolved.scope,
                        obligation=selected.question.obligation,
                        evidence=selected.evidence,
                    ),
                    cutoff,
                )
            except ValidationError as error:
                return (_dossier_refused(chosen, error), cutoff)
        # A book wider than one unit: every unit with a current analysis is
        # read exactly; every unit without one is named, with its reason, and
        # stays in the coverage denominator. Complete and partial are told
        # apart by the coverage the dossier reports, never by rounding.
        children = []
        unreviewed = []
        for unit_id, value in selections.items():
            if value.disposition == "CURRENT" and value.evidence is not None:
                assert value.question is not None
                children.append((unit_id, value.question.obligation, value.evidence))
                continue
            entities = self._unit_entities(resolved.scope, unit_id)
            if value.disposition == "EXPIRED" and value.evidence is not None:
                reason = (
                    f"its analysis expired at {value.evidence.publication.expires_at.isoformat()}"
                )
            elif value.disposition == "SUPERSEDED":
                reason = "its analysis was sealed under a superseded retrieval contract"
            else:
                reason = "no current analysis answers it"
            unreviewed.append((unit_id, entities, reason))
        mixed = _cutoffs_differ(chosen, children, carried)
        if mixed is not None:
            return (mixed, cutoff)
        try:
            return (
                compile_portfolio_coverage_dossier(
                    inputs=self.inputs(),
                    book=resolved.book,
                    projection=resolved.projection,
                    scope=resolved.scope,
                    children=tuple(children),
                    unreviewed=(*unreviewed, *lost),
                    nothing_filed=nothing_filed,
                    carried=carried,
                    open_issues=issues,
                ),
                cutoff,
            )
        except ValidationError as error:
            return (_dossier_refused(chosen, error), cutoff)

    def _sealed_run(
        self, scope: PortfolioIssuerScope, *, evidence_as_of: datetime | None
    ) -> AlternativeEvidenceCoverageRun | None:
        """The run this scope sealed at this cutoff -- without one, its newest --
        which `_units` and the review both read: a coverage Task's run, or a run
        with nothing left to read, sealed and never a Task. The latter is the
        newest only by a later cutoff."""

        adapter = self.evidence_task_adapter
        if adapter is None:
            return None
        found = None
        for task in self._run_tasks(scope, evidence_as_of=evidence_as_of):
            run = adapter.run_of(task)
            if (
                run is not None
                and run.scope_hash == scope.scope_hash
                and (evidence_as_of is None or run.evidence_as_of == evidence_as_of)
            ):
                found = run
                break
        for run in adapter.carried_runs(scope.scope_hash):
            if evidence_as_of is not None and run.evidence_as_of != evidence_as_of:
                continue
            if found is None or run.evidence_as_of > found.evidence_as_of:
                found = run
            break
        return found

    def _carried_readings(
        self,
        scope: PortfolioIssuerScope,
        run: AlternativeEvidenceCoverageRun | None,
        *,
        issues: tuple[OpenIssueState, ...] = (),
        exclude: frozenset[str] = frozenset(),
        as_of: datetime | None = None,
    ) -> tuple[tuple[CarriedReading, ...], tuple[tuple[str, tuple[str, ...], str], ...]]:
        """The earlier analyses a review at the run's cutoff reads: those whose
        filings in the window the run names as read -- each with the filings it
        carries and the carried holdings it answers for (each held by its latest
        reading) -- and those an open issue of the register rests on, with the
        findings it cites, whatever their filing's age or the contract they were
        sealed under (W3; `carried_reading_admitted`); never an analysis
        the review reads as one of its units (`exclude`). With them, the carried
        holdings whose reading no longer reads under the current contract."""

        service = self.evidence_publications
        if service is None or (not issues and (run is None or not run.read_filings)):
            return (), ()
        now = self.clock()
        filings: dict[str, set[tuple[str, str]]] = {}
        for value in () if run is None else run.read_filings:
            filings.setdefault(value.publication_hash, set()).add(
                (value.entity_id, value.accession)
            )
        rests_on: dict[str, set[str]] = {}
        for state in issues:
            for publication_hash, finding_handle, _found in state.findings:
                rests_on.setdefault(publication_hash, set()).add(finding_handle)
        views = {}
        for publication_hash in sorted({*filings, *rests_on} - exclude):
            record = self.artifacts.load(
                "analysis-publications", publication_hash, AlternativeEvidenceAnalysisPublication
            )
            if carried_reading_admitted(
                service.standing(record, now=now),
                open_issue_only=publication_hash not in filings,
            ):
                views[publication_hash] = service.replay(publication_hash, now=now)
        owner: dict[str, str] = {}
        carried = () if run is None else run.carried
        for entity in carried:
            held = [
                (
                    views[value.publication_hash].lineage.request.evidence_as_of,
                    value.publication_hash,
                )
                for value in (() if run is None else run.read_filings)
                if value.entity_id == entity
                and value.publication_hash in views
                and value.publication_hash in filings
            ]
            if held:
                owner[entity] = max(held)[1]
        lost = tuple(entity for entity in carried if entity not in owner)
        cutoff = run.evidence_as_of if run is not None else as_of or now
        policy = self.evidence_policy.run_request(run) if run is not None else None
        window = timedelta(
            days=(
                self.evidence_policy.source_policy if policy is None else policy.source_policy
            ).sec_recent_8k_days
        )

        def leaves_window(publication_hash: str) -> datetime:
            """The day the earliest filing a reading carries leaves the window
            (Z1); the window from the cutoff for one carried only for an open
            issue, which does not age out with its filing."""

            held = filings.get(publication_hash, set())
            return min(
                (
                    (value.accepted_at or value.available_at) + window
                    for value in views[publication_hash].lineage.document_set.documents
                    if (value.entity_id, value.revision_label) in held
                ),
                default=cutoff + window,
            )

        readings = tuple(
            CarriedReading(
                evidence=views[publication_hash],
                filings=frozenset(filings.get(publication_hash, ())),
                entities=tuple(
                    entity
                    for entity in scope.ordered_entity_ids
                    if owner.get(entity) == publication_hash
                ),
                as_of=cutoff,
                expires_at=leaves_window(publication_hash),
                issue_findings=frozenset(rests_on.get(publication_hash, ())),
            )
            for publication_hash in sorted(
                views, key=lambda value: (views[value].lineage.request.evidence_as_of, value)
            )
        )
        unreviewed = (
            ()
            if not lost
            else (
                (
                    "carried",
                    lost,
                    "its earlier reading is sealed under a superseded contract and reads no "
                    "longer; prepare evidence again",
                ),
            )
        )
        return readings, unreviewed

    def _unit_entities(self, scope: PortfolioIssuerScope, unit_id: str) -> tuple[str, ...]:
        units = self._units(scope)
        ids = coverage_unit_ids(units)
        for entities in units:
            if ids[entities] == unit_id:
                return entities
        raise PortfolioEvidenceReviewError("product_host.evidence_review_unit_unknown:" + unit_id)

    def _select_evidence_for_review(
        self, resolved: ResolvedBookScope, *, evidence_as_of: datetime | None
    ) -> dict[str, _CurrentEvidenceSelection]:
        """Each unit's evidence for the review, by unit id in execution order.

        Given no cutoff, the explicit/unique current analysis each unit has;
        given one, the one current analysis answering each unit's exact
        obligation at that cutoff, or nothing.
        """

        if evidence_as_of is None:
            by_entities = self._select_unit_evidence(resolved)
            ids = coverage_unit_ids(tuple(by_entities))
            return {ids[entities]: value for entities, value in by_entities.items()}
        selections: dict[str, _CurrentEvidenceSelection] = {}
        for entities, (unit_id, obligation) in self._unit_obligations(
            resolved.scope, evidence_as_of=evidence_as_of
        ).items():
            del entities
            question = ResolvedEvidenceQuestion(resolved=resolved, obligation=obligation)
            evidence = self.current_evidence(question=question)
            selections[unit_id] = _CurrentEvidenceSelection(
                disposition="CURRENT" if evidence is not None else "ABSENT",
                question=question if evidence is not None else None,
                evidence=evidence,
            )
        return selections

    def submit_assessment(
        self,
        *,
        dispatcher: LocalBackgroundDispatcher,
        selector: BookSelector | None,
        dossier_hash: str,
        policy_hash: str,
        schema_hash: str,
        answer: object,
        caller: str,
        read_files: tuple[str, ...] | None = None,
        agent_run: AgentRun | None = None,
        read_at: datetime | None = None,
    ) -> ReviewOutcome | dict[str, object]:
        """Screen and bind one explicit CRO answer to its exact dossier read time.

        Admit one review answer: screened risk by risk, refused with every
        problem in plain words, or its risks bound and routed by the Task. ``read_at`` is
        when its dossier was read, so the answer meets that dossier (V255).
        """
        if caller not in {"HUMAN", "EXTERNAL_AUTOMATION"}:
            raise PortfolioEvidenceReviewError(
                "chief_risk_officer.external_submission_entry_required"
            )
        chosen = self._review_selector(selector)
        if isinstance(chosen, ReviewOutcome):
            return chosen
        dossier = self._resolve_review_dossier(chosen, read_at=read_at)
        if isinstance(dossier, ReviewOutcome):
            return dossier
        policy = self._decision_policy()
        if (
            dossier.dossier_hash != dossier_hash
            or policy.binding_hash != policy_hash
            or assessment_schema_hash(dossier) != schema_hash
        ):
            raise PortfolioEvidenceReviewError("chief_risk_officer.external_review_binding_changed")
        screened = screen_review_answer(answer, dossier=dossier)
        binding: dict[str, object] = {
            "role": "chief_risk_officer.reviewer",
            "dossier_hash": dossier_hash,
            "policy_hash": policy_hash,
            "schema_hash": schema_hash,
        }
        with self._answer_turn(binding):
            record = self._next_answer(
                binding,
                answer,
                screened,
                read_files=read_files,
                agent_run=agent_run,
            )
            if record.verdict is AnswerVerdict.CORRECT:
                self._file_answer(record)
                return _answer_correction("chief_risk_officer.answer_problems", record)
            accepted = accepted_review_answer(screened)
            actor = SubmittedPortfolioReviewActor(
                answer=accepted,
                dropped=screened.problems if record.verdict is AnswerVerdict.DONE else (),
                actor_kind=ActorKind(caller),
                actor_id="local-research-external"
                if caller == "EXTERNAL_AUTOMATION"
                else "local-web-human",
                durable_submission=True,
            )
            # Reuse the actual sealer for the before-admission checks. Its value is
            # not filed here; the existing Task seals/publishes after its own checks.
            seal_portfolio_review(
                dossier=dossier,
                answer=accepted,
                evidence_is_current=True,
                decision_policy=policy,
                playpen_root=self.playpen_root,
                actor_kind=actor.actor_kind,
                actor_id=actor.actor_id,
                dropped=actor.dropped,
            )
            outcome = self._submit_review(
                dossier, actor, dispatcher, expected_policy_hash=policy_hash
            )
            newly_filed = False
            if outcome.disposition == "ADMITTED":
                newly_filed = self._file_answer(record)
            delivery = (
                self.accepted_answer_delivery(
                    record,
                    accepted.model_dump(mode="json"),
                    task_id=outcome.task_id,
                    newly_filed=newly_filed,
                    input_hash=(
                        outcome.review.publication.review_key
                        if outcome.review is not None
                        else None
                    ),
                )
                if outcome.disposition in {"ADMITTED", "REUSED_EXACT"}
                else None
            )
            return replace(
                outcome,
                answer=_answer_view(
                    record, delivery, record_filed=outcome.disposition == "ADMITTED"
                ),
            )

    def submit_specialist_answer(
        self,
        *,
        bundle: AgentBundleRecord,
        answer: object,
        read_files: tuple[str, ...] | None = None,
        agent_run: AgentRun | None = None,
    ) -> dict[str, object]:
        """Seal one interpretation's shape and exact references without judging its science."""
        task_id = UUID(bundle.submission["task_id"])
        task = self.session.task_control_registry.task(task_id)
        binding: dict[str, object] = {
            "bundle_reference": bundle.record_hash,
            "role": bundle.role,
            "task_id": str(task_id),
            "task_record_hash": bundle.submission["task_record_hash"],
        }
        screened = screen_specialist_answer(answer, allowed_references=bundle.allowed_references)
        with self._answer_turn(binding):
            record = self._next_answer(
                binding, answer, screened, read_files=read_files, agent_run=agent_run
            )
            filed = self.filed_answer(record)
            if filed is None and task.record_hash != bundle.submission["task_record_hash"]:
                raise PortfolioEvidenceReviewError("agent_bundle.specialist_task_changed")
            if record.verdict is AnswerVerdict.CORRECT:
                self._file_answer(record)
                return _answer_correction("actor_execution.specialist_answer_problems", record)
            newly_filed = False
            if filed is not None:
                record = filed
            else:
                accepted = screened.items[0][1] if screened.items else None
                record = AgentAnswerRecord.model_validate(
                    {
                        **record.model_dump(mode="json"),
                        "accepted_text": accepted.text if accepted is not None else None,
                        "accepted_references": accepted.references if accepted is not None else (),
                        "accepted_at": self.clock(),
                    }
                )
                newly_filed = self._file_answer(record)
            contribution: dict[str, object] = {
                "text": record.accepted_text,
                "references": list(record.accepted_references),
            }
            return {
                "disposition": "ADMITTED" if newly_filed else "REUSED_EXACT",
                "task_id": str(task_id),
                "lifecycle": str(task.lifecycle),
                "answer": _answer_view(
                    record,
                    self.accepted_answer_delivery(
                        record, contribution, task_id=task_id, newly_filed=newly_filed
                    ),
                ),
            }

    def filed_answer(self, record: AgentAnswerRecord) -> AgentAnswerRecord | None:
        """Read the first filed terminal record for this exact answer digest and binding."""
        for number in range(1, record.number + 1):
            slot = answer_slot(record.bundle_key, number)
            if not self.artifacts.exists(ANSWER_CATEGORY, slot):
                break
            candidate = self.artifacts.load(ANSWER_CATEGORY, slot, AgentAnswerRecord)
            if candidate.answer_digest == record.answer_digest and candidate.verdict in {
                AnswerVerdict.ACCEPTED,
                AnswerVerdict.DONE,
            }:
                return candidate
        return None

    @verified_evidence_records()
    def read_accepted_answer(self, subject: Mapping[str, object]) -> dict[str, object]:
        """Read a sealed accepted contribution bound to one observed native answer.

        This opens the retained answer, bundle and exact Task only for a selected
        accepted event. Typed submissions recover their screened Task input; generic
        contributions must still match the original legal answer's digest.
        """
        metadata = {
            key: subject.get(key)
            for key in (
                "native_host",
                "native_session_id",
                "native_agent_id",
                "role",
                "submitted_by",
                "answer_reference",
                "bundle_reference",
            )
        }
        metadata["task_id"] = subject.get("reference")
        unavailable = {
            **metadata,
            "status": "UNAVAILABLE",
            "reason": "activity.accepted_answer_unavailable",
        }
        mismatch = {**unavailable, "reason": "activity.accepted_answer_binding_mismatch"}
        try:
            answer = self.artifacts.load(
                ANSWER_CATEGORY, str(subject.get("answer_reference")), AgentAnswerRecord
            )
            bundle = self.artifacts.load(
                BUNDLE_CATEGORY, str(subject.get("bundle_reference")), AgentBundleRecord
            )
            run = answer.agent_run
            if (
                answer.verdict not in {AnswerVerdict.ACCEPTED, AnswerVerdict.DONE}
                or run is None
                or run.basis != "HOOK"
                or run.agent_id is None
                or run.role is None
            ):
                return unavailable
            if (
                subject.get("authorship_basis") != "HOOK"
                or subject.get("native_host") != run.host
                or subject.get("native_session_id") != run.session_id
                or subject.get("native_agent_id") != run.agent_id
                or subject.get("role") != run.role
                or subject.get("submitted_by") != run.session_id
            ):
                return mismatch
            return self._read_accepted_contribution(
                answer, bundle, subject.get("reference"), metadata
            )
        except (OSError, ValueError, TaskNotFoundError):
            return unavailable

    @verified_evidence_records()
    def read_product_accepted_answer(
        self, subject: Mapping[str, object], *, verdict: str
    ) -> dict[str, object]:
        """Read a sealed answer selected by an owner operation-return observation.

        Workspace activity verifies that selected row's source and authority before
        asking here. The submitting request's Session and Goal remain separate from
        the answer's stored attribution; absent attribution is never reconstructed.
        """
        metadata = {
            key: subject.get(key)
            for key in (
                "agent_role",
                "agent_vendor",
                "agent_session",
                "goal_id",
                "answer_reference",
                "bundle_reference",
                "task_id",
            )
        }
        metadata["record_kind"] = "PRODUCT_OPERATION"
        unavailable = {
            **metadata,
            "status": "UNAVAILABLE",
            "reason": "activity.accepted_answer_unavailable",
        }
        mismatch = {**unavailable, "reason": "activity.accepted_answer_binding_mismatch"}
        try:
            answer = self.artifacts.load(
                ANSWER_CATEGORY, str(subject.get("answer_reference")), AgentAnswerRecord
            )
            bundle = self.artifacts.load(
                BUNDLE_CATEGORY, str(subject.get("bundle_reference")), AgentBundleRecord
            )
            if answer.verdict not in {AnswerVerdict.ACCEPTED, AnswerVerdict.DONE}:
                return unavailable
            if answer.verdict.value != verdict or subject.get("agent_role") != bundle.role:
                return mismatch
            goal = subject.get("goal_id")
            if goal is not None and str(UUID(str(goal))) != goal:
                return mismatch
            vendor, session = subject.get("agent_vendor"), subject.get("agent_session")
            if (vendor is None) != (session is None) or (
                vendor is not None
                and (
                    vendor not in {"claude-code", "codex"}
                    or not isinstance(session, str)
                    or not 1 <= len(session) <= 128
                )
            ):
                return mismatch
            read = self._read_accepted_contribution(
                answer, bundle, subject.get("task_id"), metadata
            )
            if read.get("status") == "AVAILABLE":
                read["recorded_agent"] = (
                    None if answer.agent_run is None else answer.agent_run.model_dump(mode="json")
                )
            return read
        except (OSError, ValueError, TaskNotFoundError):
            return unavailable

    def _read_accepted_contribution(
        self,
        answer: AgentAnswerRecord,
        bundle: AgentBundleRecord,
        task_reference: object,
        metadata: Mapping[str, object],
    ) -> dict[str, object]:
        """Reopen the exact Task and sealed contribution for either selected receipt."""
        unavailable = {
            **metadata,
            "status": "UNAVAILABLE",
            "reason": "activity.accepted_answer_unavailable",
        }
        mismatch = {**unavailable, "reason": "activity.accepted_answer_binding_mismatch"}
        try:
            task_id = UUID(str(task_reference))
            task = self.session.task_control_registry.task(task_id)
            if str(task.task_id) != task_reference:
                return mismatch
            submission = bundle.submission
            contribution: dict[str, object]
            if "operation" not in submission:
                if submission.get("task_id") != str(task_id) or not submission.get(
                    "task_record_hash"
                ):
                    return mismatch
                binding: dict[str, object] = {
                    "bundle_reference": bundle.record_hash,
                    "role": bundle.role,
                    "task_id": str(task_id),
                    "task_record_hash": submission["task_record_hash"],
                }
                if answer.accepted_text is None:
                    return unavailable
                contribution = {
                    "text": answer.accepted_text,
                    "references": list(answer.accepted_references),
                }
                digests = {answer_digest(contribution)}
                if not answer.accepted_references:
                    digests.add(answer_digest({"text": answer.accepted_text}))
                if (
                    answer.answer_digest not in digests
                    or answer.accepted_items != (1,)
                    or any(
                        value not in bundle.allowed_references
                        for value in answer.accepted_references
                    )
                ):
                    return mismatch
            elif (
                task.input.task_kind == AlternativeEvidenceDocumentTaskAdapter.task_kind
                and task.input.input_schema_id == EVIDENCE_INPUT_SCHEMA_ID
                and task.input.payload.get("purpose") == "SUBMITTED_ANALYSIS"
            ):
                if (
                    not {"request", "admission", "obligation", "submitted_analysis"}
                    <= task.input.payload.keys()
                ):
                    return unavailable
                command = AlternativeEvidenceRefreshCommand.recover(application=self, task=task)
                submitted = command.submitted_analysis
                adapter = self.evidence_task_adapter
                if submitted is None or submitted.answer is None or adapter is None:
                    return unavailable
                _, context = adapter.analysis_context(
                    submitted.prepared_task_id,
                    now=task.admitted_at,
                    unit_id=submitted.prepared_unit_id,
                )
                context_hash = context.get("analysis_context_hash")
                if not isinstance(context_hash, str):
                    return unavailable
                if (
                    submission.get("operation") != "EVIDENCE_ANALYSIS_SUBMIT"
                    or submission.get("task_id") != str(submitted.prepared_task_id)
                    or submission.get("evidence_unit_id") != submitted.prepared_unit_id
                    or submission.get("analysis_context_hash") != context_hash
                    or submitted.packet_hash != context.get("packet_hash")
                    or submitted.analysis_policy_hash != context.get("analysis_policy_hash")
                    or submitted.decision_policy_hash != context.get("decision_policy_hash")
                ):
                    return mismatch
                binding = {
                    "role": "alternative_evidence.analyst",
                    "task_id": str(submitted.prepared_task_id),
                    "unit_id": submitted.prepared_unit_id,
                    "analysis_context_hash": context_hash,
                }
                contribution = submitted.answer.model_dump(mode="json")
            elif (
                task.input.task_kind == CRO_REVIEW_COMMAND
                and task.input.input_schema_id == CRO_INPUT_SCHEMA_ID
                and "prepared_answer" in task.input.payload
            ):
                if (
                    not {
                        "actor_kind",
                        "actor_id",
                        "dossier_hash",
                        "decision_policy_hash",
                        "submission_hash",
                    }
                    <= task.input.payload.keys()
                ):
                    return unavailable
                actor = self.recover_review_actor(task)
                if not isinstance(actor, SubmittedPortfolioReviewActor):
                    return unavailable
                dossier = self.artifacts.load(
                    "cro-review-dossiers",
                    str(task.input.payload["dossier_hash"]),
                    PortfolioReviewDossier,
                )
                schema_hash = assessment_schema_hash(dossier)
                if (
                    submission.get("operation") != "CRO_REVIEW_SUBMIT"
                    or submission.get("review_dossier_hash") != dossier.dossier_hash
                    or submission.get("review_policy_hash")
                    != task.input.payload["decision_policy_hash"]
                    or submission.get("review_schema_hash") != schema_hash
                ):
                    return mismatch
                binding = {
                    "role": "chief_risk_officer.reviewer",
                    "dossier_hash": dossier.dossier_hash,
                    "policy_hash": task.input.payload["decision_policy_hash"],
                    "schema_hash": schema_hash,
                }
                contribution = actor.answer.model_dump(mode="json")
            else:
                return unavailable
            if str(canonical_hash(binding)) != answer.bundle_key:
                return mismatch
            return {
                **metadata,
                "status": "AVAILABLE",
                "verdict": answer.verdict.value,
                "answer_digest": answer.answer_digest,
                "contribution": contribution,
            }
        except (OSError, ValueError, TaskNotFoundError):
            return unavailable

    def accepted_answer_delivery(
        self,
        record: AgentAnswerRecord,
        contribution: Mapping[str, object],
        *,
        task_id: UUID | None = None,
        input_hash: str | None = None,
        newly_filed: bool = False,
    ) -> dict[str, object] | None:
        """Project optional Team metadata without undoing scientific acceptance.

        The scientific admission and answer filing have already succeeded. A failure
        reading their conversation metadata is returned as observation unavailability.
        """
        try:
            return self._accepted_answer_delivery(
                record,
                contribution,
                task_id=task_id,
                input_hash=input_hash,
                newly_filed=newly_filed,
            )
        except Exception:
            return {"status": "UNAVAILABLE", "reason": "native_bridge.accepted_answer_not_recorded"}

    def _accepted_answer_delivery(
        self,
        record: AgentAnswerRecord,
        contribution: Mapping[str, object],
        *,
        task_id: UUID | None,
        input_hash: str | None,
        newly_filed: bool,
    ) -> dict[str, object] | None:
        """Resolve the generic filed answer and exact admitted Task for observation (V691).

        Exact retries use the first filed acceptable answer with this digest, including
        its original author. An unfiled answer and a correction confer no credit. The
        contribution comes only from the screened accepted part, so dropped items never appear.
        A publication reused without a filed accepted answer supplies no author. A
        retry may repair a missed observation, retaining the original filed authorship.
        """
        filed = self.filed_answer(record)
        if filed is None:
            return None
        if task_id is None:
            tasks = [
                task
                for task in self.session.task_control_registry.tasks()
                if input_hash is not None and task.input.input_hash == input_hash
            ]
            if len(tasks) != 1:
                return {
                    "status": "UNAVAILABLE",
                    "reason": "native_bridge.accepted_answer_not_recorded",
                }
            task_id = tasks[0].task_id
        return {
            "answer_record": filed.model_dump(mode="json"),
            "contribution": dict(contribution),
            "task_id": str(task_id),
            "first_submission": newly_filed and filed.record_hash == record.record_hash,
        }

    def _book_packet(
        self, chosen: BookSelector, *, task_id: UUID, unit_id: str | None
    ) -> tuple[
        AlternativeEvidencePacket,
        dict[str, object],
        ResolvedBookScope,
        dict[tuple[str, ...], tuple[str, AlternativeEvidenceResearchObligation]],
        tuple[str, AlternativeEvidenceResearchObligation],
    ]:
        """One prepared packet that is one of this book's units at its own
        cutoff -- the same issuers, the same question -- or a named refusal."""

        adapter = self.evidence_task_adapter
        if adapter is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
        packet, context = adapter.analysis_context(task_id, now=self.clock(), unit_id=unit_id)
        resolved = self.resolve_book(chosen)
        units = self._unit_obligations(resolved.scope, evidence_as_of=packet.request.evidence_as_of)
        expected = units.get(tuple(packet.request.ordered_entity_ids))
        if expected is None or packet.obligation != expected[1]:
            raise PortfolioEvidenceReviewError("alternative_evidence.prepared_book_scope_mismatch")
        return packet, context, resolved, units, expected

    def _analysis_submission(
        self, chosen: BookSelector, *, task_id: UUID, unit_id: str | None, context_hash: str
    ) -> dict[str, str]:
        """The operation document an answer to this packet completes."""

        selector_fields = chosen.request_fields()
        return {
            "operation": "EVIDENCE_ANALYSIS_SUBMIT",
            **selector_fields,
            "task_id": str(task_id),
            **({} if unit_id is None else {"evidence_unit_id": unit_id}),
            "analysis_context_hash": context_hash,
        }

    def _review_submission(
        self, dossier: PortfolioReviewDossier, *, read_at: datetime
    ) -> dict[str, str]:
        """The operation document an answer to this dossier, read at ``read_at``, completes."""

        return {
            "operation": "CRO_REVIEW_SUBMIT",
            **self._selector_for_dossier(dossier).request_fields(),
            "review_dossier_hash": dossier.dossier_hash,
            "review_policy_hash": self._decision_policy().binding_hash,
            "review_schema_hash": assessment_schema_hash(dossier),
            "review_read_at": read_at.isoformat(),
        }

    def _next_answer(
        self,
        binding: dict[str, object],
        raw: object,
        screened: ScreenedAnswer[Any],
        *,
        read_files: tuple[str, ...] | None = None,
        agent_run: AgentRun | None = None,
    ) -> AgentAnswerRecord:
        """Count one answer against its bundle: its verdict follows the
        corrections already filed for the same binding, so the count survives
        a restart. Nothing is filed here -- a correction is filed at once, an
        acceptable answer only when a Task admits it, so a refused request or
        an exact reuse never spends one of the bundle's answers. A bundle whose
        corrections ran out (DONE) reads no other answer: the same one again is
        its own record, whose receipt the exact reuse returns (V417). An accepted
        answer closes nothing: another is a review of its own."""

        bundle_key = str(canonical_hash(binding))
        digest = answer_digest(raw)
        number, corrections = 1, 0
        while self.artifacts.exists(ANSWER_CATEGORY, answer_slot(bundle_key, number)):
            filed = self.artifacts.load(
                ANSWER_CATEGORY, answer_slot(bundle_key, number), AgentAnswerRecord
            )
            if filed.verdict is AnswerVerdict.DONE:
                if filed.answer_digest != digest:
                    raise PortfolioEvidenceReviewError("agent_bundle.answer_settled")
                return filed
            corrections = filed.corrections_used
            number += 1
        verdict = answer_verdict(clean=screened.clean, corrections_used=corrections)
        record = AgentAnswerRecord(
            bundle_key=bundle_key,
            number=number,
            verdict=verdict,
            corrections_used=corrections + (1 if verdict is AnswerVerdict.CORRECT else 0),
            problems=screened.problems[:256],
            accepted_items=screened.accepted_numbers[:256],
            answer_digest=digest,
            read_files=read_files,
            agent_run=agent_run,
            record_hash=answer_slot(bundle_key, number),
        )
        return record

    @contextmanager
    def _answer_turn(self, binding: dict[str, object]) -> Iterator[None]:
        """Hold one binding's turn: its next answer is counted, admitted and filed before another
        is counted, so two answers sent at once are numbered one and two, each kept (V557)."""
        key = str(canonical_hash(binding))
        with self._answer_turns_lock:
            turn = self._answer_turns.setdefault(key, threading.Lock())
        with turn:
            yield

    def _file_answer(self, record: AgentAnswerRecord) -> bool:
        """File one answer in its slot. The same record again is the one filed -- a DONE answer
        sent again (V417), however old its record -- and another in a slot already taken is
        refused, never skipped (V557)."""
        if not self.artifacts.exists(ANSWER_CATEGORY, record.record_hash):
            self.artifacts.publish(ANSWER_CATEGORY, record.record_hash, record)
            return True
        filed = self.artifacts.load(ANSWER_CATEGORY, record.record_hash, AgentAnswerRecord)
        if filed != record:
            raise PortfolioEvidenceReviewError("agent_bundle.answer_slot_taken")
        return False

    def _submit_review(
        self,
        dossier: PortfolioReviewDossier,
        reviewer: PortfolioReviewActor,
        dispatcher: LocalBackgroundDispatcher,
        *,
        expected_policy_hash: str | None = None,
    ) -> ReviewOutcome:
        contract = self.review_task_contract(dossier=dossier, actor=reviewer)
        envelope, _goal, _plan = contract
        if (
            expected_policy_hash is not None
            and envelope.payload["decision_policy_hash"] != expected_policy_hash
        ):
            raise PortfolioEvidenceReviewError("chief_risk_officer.external_review_binding_changed")
        self.artifacts.publish("cro-review-dossiers", dossier.dossier_hash, dossier)
        existing = self.review_publications.find_for_review_key(envelope.input_hash)
        if existing is not None:
            return ReviewOutcome(
                disposition="REUSED_EXACT",
                detail="This exact review was already published.",
                review=existing,
            )
        command = PortfolioReviewCommand(
            application=self, dossier=dossier, actor=reviewer, contract=contract
        )
        return _submitted(dispatcher.submit(command))

    def drive_review_task(
        self,
        *,
        actor: PortfolioReviewActor,
        task_id: UUID | None = None,
        expected_task_hash: str | None = None,
    ) -> None:
        """Drive the admitted review using the explicitly bound actor adapter.

        Args:
            actor: Explicit review actor.
            task_id: Optional exact admitted task.
            expected_task_hash: Optional optimistic task identity.
        """
        self._run(
            self._review_adapter(actor), task_id=task_id, expected_task_hash=expected_task_hash
        )

    def published_review_for_book(
        self,
        book: SealedBook,
        *,
        analysis_publication_hash: str | None = None,
        result_hash: str | None = None,
        use_selected_analysis: bool = True,
    ) -> PortfolioReviewView | None:
        """Read the newest published recommendation for this exact book.

        The published recommendation this workspace holds for this exact book,
        if any: the newest, chosen from the publication records and read once
        -- a book reviewed every day holds a review a day (W3).

        `use_selected_analysis=False` reads the book's latest published review independently
        of a currently selected Evidence analysis, as an activation decision requires (V614).
        `result_hash` binds that read to the exact installed result the book Task published.
        """
        wanted = (
            self.selected_analysis_publication_hash if use_selected_analysis else None,
            analysis_publication_hash,
        )
        # The newest day holding a review of the book holds its newest review:
        # older records are not read (Z2).
        days = self._review_days(None)
        for group in (None,) if days is None else days:
            publications = [
                value
                for value in self._book_publications(book, group)
                if all(
                    named is None or value.analysis_publication_hash == named for named in wanted
                )
                and (result_hash is None or value.result_hash == result_hash)
            ]
            if publications:
                newest = max(
                    publications, key=lambda value: (value.published_at, value.publication_hash)
                )
                return self.review_publications.read(newest.publication_hash)
        return None

    def _review_days(self, since: datetime | None) -> tuple[tuple[str, ...], ...] | None:
        """The review publication records published since a moment, one group
        a day, newest first; None without the runtime that keeps the days."""

        adapter = self.evidence_task_adapter
        if adapter is None:
            return None
        return adapter.runtime.days_of("cro-review-publications", "published_at").by_day(since)

    def _book_publications(
        self, book: SealedBook, identities: tuple[str, ...] | None = None
    ) -> tuple[PortfolioReviewPublication, ...]:
        """The publication record of every review of this exact book -- among
        the named records, when named."""

        store = self.review_publications.store
        records = (
            store.values("cro-review-publications", PortfolioReviewPublication)
            if identities is None
            else tuple(
                store.load("cro-review-publications", identity, PortfolioReviewPublication)
                for identity in identities
            )
        )
        wanted = book_key(
            book.report_hash, book.update_subject, book.experiment_subject, book.authority
        )
        return tuple(
            publication for publication in records if review_book_key(publication) == wanted
        )

    def _published_reviews_for_book(
        self, book: SealedBook, *, since: datetime | None = None
    ) -> tuple[PortfolioReviewView, ...]:
        """Every published recommendation this workspace holds for this exact
        book -- since a moment, when the caller asks about analyses no earlier
        review could have read."""

        days = None if since is None else self._review_days(since)
        return tuple(
            self.review_publications.read(publication.publication_hash)
            for publication in self._book_publications(
                book, None if days is None else tuple(i for group in days for i in group)
            )
            if since is None or publication.published_at >= since
        )

    def review_changes(
        self, publication_hash: str, prior_publication_hash: str
    ) -> dict[str, object]:
        """Compare exact published review facts and report incomparable subjects explicitly.

        What changed between two published reviews, as facts a reader can
        trace: findings new, changed, no longer found or unchanged (matched on
        issuer set, topic and the analyst's own words, never by handle, which
        is per publication), dispositions that moved, coverage and issuer
        position changes, and the sources each side stood on. Unavailable,
        with the reason, when the two reviews are not comparable (another
        book, another subject). Never a judgment: a finding that is no longer
        found is reported as that, not as resolved.
        """
        current = self.review_publications.read(publication_hash)
        prior = self.review_publications.read(prior_publication_hash)
        head, base = current.dossier, prior.dossier
        subject = (
            head.book_authority,
            head.update_subject,
            head.experiment_subject,
            head.registry_hash,
        )
        prior_subject = (
            base.book_authority,
            base.update_subject,
            base.experiment_subject,
            base.registry_hash,
        )
        if subject != prior_subject:
            return {
                "status": "REVIEWS_NOT_COMPARABLE",
                "reason": "the two reviews are of different books or subjects",
                "current_publication_hash": publication_hash,
                "prior_publication_hash": prior_publication_hash,
            }
        # The prior must come first: by publication, then -- for two reviews
        # published in the same instant -- by the evidence cutoff each read.
        if (prior.publication.published_at, base.evidence_as_of) > (
            current.publication.published_at,
            head.evidence_as_of,
        ):
            return {
                "status": "REVIEWS_NOT_COMPARABLE",
                "reason": (
                    "the prior review was published after the current one"
                    if prior.publication.published_at > current.publication.published_at
                    else "the prior review reads evidence as of a later cutoff than the current one"
                ),
                "current_publication_hash": publication_hash,
                "prior_publication_hash": prior_publication_hash,
            }

        def key(finding: Any) -> tuple[str, str, str]:
            return (
                ",".join(sorted(finding.affected_entities)),
                str(finding.topic),
                " ".join(finding.summary.split()),
            )

        def loose_key(finding: Any) -> tuple[str, str]:
            return (",".join(sorted(finding.affected_entities)), str(finding.topic))

        base_by_key = {key(f): f for f in base.findings}
        head_dispositions, base_dispositions = (
            finding_dispositions(current),
            finding_dispositions(prior),
        )
        unchanged: list[dict[str, object]] = []
        changed: list[dict[str, object]] = []
        new: list[dict[str, object]] = []
        gone: list[dict[str, object]] = []
        base_loose: dict[tuple[str, str], list[Any]] = {}
        for finding in base.findings:
            base_loose.setdefault(loose_key(finding), []).append(finding)
        matched_base: set[str] = set()
        for finding in head.findings:
            exact = base_by_key.get(key(finding))
            if exact is not None:
                matched_base.add(exact.finding_handle)
                now_disposition = head_dispositions.get(finding.finding_handle)
                then_disposition = base_dispositions.get(exact.finding_handle)
                unchanged.append(
                    {
                        "finding_handle": finding.finding_handle,
                        "prior_finding_handle": exact.finding_handle,
                        "affected_entities": list(finding.affected_entities),
                        "topic": str(finding.topic),
                        "summary": finding.summary,
                        "disposition": now_disposition,
                        "prior_disposition": then_disposition,
                        "disposition_moved": now_disposition != then_disposition,
                    }
                )
                continue
            candidates = [
                f
                for f in base_loose.get(loose_key(finding), ())
                if f.finding_handle not in matched_base
            ]
            if candidates:
                previous = candidates[0]
                matched_base.add(previous.finding_handle)
                changed.append(
                    {
                        "finding_handle": finding.finding_handle,
                        "prior_finding_handle": previous.finding_handle,
                        "affected_entities": list(finding.affected_entities),
                        "topic": str(finding.topic),
                        "summary": finding.summary,
                        "prior_summary": previous.summary,
                        "lifecycle": str(finding.lifecycle),
                        "prior_lifecycle": str(previous.lifecycle),
                        "disposition": head_dispositions.get(finding.finding_handle),
                        "prior_disposition": base_dispositions.get(previous.finding_handle),
                    }
                )
                continue
            new.append(
                {
                    "finding_handle": finding.finding_handle,
                    "affected_entities": list(finding.affected_entities),
                    "topic": str(finding.topic),
                    "summary": finding.summary,
                    "disposition": head_dispositions.get(finding.finding_handle),
                }
            )
        for finding in base.findings:
            if finding.finding_handle not in matched_base:
                gone.append(
                    {
                        "prior_finding_handle": finding.finding_handle,
                        "affected_entities": list(finding.affected_entities),
                        "topic": str(finding.topic),
                        "summary": finding.summary,
                        "prior_disposition": base_dispositions.get(finding.finding_handle),
                        "meaning": "no longer among the current findings; not thereby resolved",
                    }
                )
        head_issuers = {i.entity_id: i for i in head.issuers}
        base_issuers = {i.entity_id: i for i in base.issuers}
        positions = []
        for entity in sorted(set(head_issuers) | set(base_issuers)):
            now, before = head_issuers.get(entity), base_issuers.get(entity)
            if now is None or before is None:
                positions.append(
                    {
                        "entity_id": entity,
                        "change": "ENTERED_SCOPE" if before is None else "LEFT_SCOPE",
                        "ending_weight": None if now is None else now.ending_weight,
                        "prior_ending_weight": None if before is None else before.ending_weight,
                    }
                )
            elif (
                now.ending_weight != before.ending_weight
                or now.transition != before.transition
                or now.review_state != before.review_state
            ):
                positions.append(
                    {
                        "entity_id": entity,
                        "change": "CHANGED",
                        "ending_weight": now.ending_weight,
                        "prior_ending_weight": before.ending_weight,
                        "transition": now.transition,
                        "prior_transition": before.transition,
                        "review_state": now.review_state,
                        "prior_review_state": before.review_state,
                    }
                )
        coverage = {
            name: {
                "current": getattr(head.coverage, name),
                "prior": getattr(base.coverage, name),
            }
            for name in (
                "reviewed_ending_weight_coverage",
                "reviewed_absolute_change_coverage",
                "mapping_coverage",
                "selected_issuer_coverage",
            )
            if getattr(head.coverage, name) != getattr(base.coverage, name)
        }
        typed_changes = self._typed_disclosure_changes(head, base)
        return {
            "status": "REVIEW_CHANGES",
            "current_publication_hash": publication_hash,
            "prior_publication_hash": prior_publication_hash,
            "current_published_at": current.publication.published_at.isoformat(),
            "prior_published_at": prior.publication.published_at.isoformat(),
            "current_evidence_as_of": head.evidence_as_of.isoformat(),
            "prior_evidence_as_of": base.evidence_as_of.isoformat(),
            "route": {
                "current": str(current.recommendation.route),
                "prior": str(prior.recommendation.route),
            },
            "review_state": {
                "current": str(current.recommendation.review_state),
                "prior": str(prior.recommendation.review_state),
            },
            "findings": {
                "new": new,
                "changed": changed,
                "no_longer_found": gone,
                "unchanged": unchanged,
            },
            "dispositions_moved": [e for e in unchanged if e["disposition_moved"]],
            "coverage_changes": coverage,
            "position_changes": positions,
            "typed_disclosure_changes": typed_changes,
            "sources": {
                "current_publications": list(head.evidence_publication_hashes),
                "prior_publications": list(base.evidence_publication_hashes),
            },
            "claim": (
                "Matched on issuers, topic and the analyst's words; a finding no longer found "
                "is reported as absent, never as resolved; nothing here re-judges either review."
            ),
        }

    def _typed_disclosure_changes(
        self, head: PortfolioReviewDossier, base: PortfolioReviewDossier
    ) -> dict[str, object]:
        """The typed disclosure families of two reviews compared as facts:
        each (issuer, family, form, item) scope's movement between the prior
        review's evidence and the current one's, under the rules each ran.
        Unavailable, with the reason, when either side's evidence did not run
        the families or cannot be replayed."""

        service = self._require_evidence_publications()
        now = self.clock()

        replay = partial(service.replay, now=now)
        current, current_rules = typed_disclosure_records(head, replay)
        prior, prior_rules = typed_disclosure_records(base, replay)
        if current_rules is None or prior_rules is None:
            return {
                "status": "TYPED_DISCLOSURES_NOT_COMPARABLE",
                "reason": "the typed disclosure families did not run for "
                + (
                    "either review"
                    if current_rules is None and prior_rules is None
                    else "one review"
                ),
            }
        changes = compare_typed_disclosures(
            current=tuple(current),
            prior=tuple(prior),
            current_rules=current_rules,
            prior_rules=prior_rules,
        )
        return {
            "status": "TYPED_DISCLOSURE_CHANGES",
            "current_rules": {"rules_id": current_rules[0], "definitions_hash": current_rules[1]},
            "prior_rules": {"rules_id": prior_rules[0], "definitions_hash": prior_rules[1]},
            "changes": [
                {
                    "scope": value.key,
                    "kind": value.kind,
                    "detail": value.detail,
                    "current_observation_id": value.current_observation_id,
                    "prior_observation_id": value.prior_observation_id,
                    "current_state": value.current_state,
                    "prior_state": value.prior_state,
                    "changed_fields": list(value.changed_fields),
                }
                for value in changes
            ],
            "claim": (
                "Compared per issuer, family, form and item under the rules each side ran, "
                "assertion by assertion on the parsed fields (quantity, terms, persons, "
                "qualifiers), not on event labels alone; a scope no longer observed is "
                "reported as that, never as resolved; a change under different rules or from "
                "the same source bytes is named as such; a re-reported period changes what it "
                "states, it does not create a new event."
            ),
        }

    # ------------------------------------------------------------ projection

    def _admission_view(self) -> dict[str, object]:
        """The bounds every preparation of this host runs under, as admitted:
        the per-document cap and the acquisition window a request carries
        from its cutoff. Stated before any request; never raised after a
        refusal."""

        return {
            "maximum_document_bytes": self.evidence_policy.source_policy.maximum_document_bytes,
            "acquisition_window_seconds": self.evidence_policy.acquisition_window_seconds,
        }

    def _campaign_view(self) -> dict[str, object] | None:
        """The durable campaign the live source spends, read from its ledger
        at this moment -- what is consumed, what remains and what is still
        reserved -- never the admission's snapshot. None when no campaign
        was named or the host prepares from a recorded package."""

        return None if self.campaign_summary is None else self.campaign_summary()

    def _book_projection(self, resolved: ResolvedBookScope) -> EvidenceCroBook:
        return book_projection(
            authority=resolved.book.authority,
            result_hash=resolved.book.result_hash,
            formation_session=resolved.projection.formation_session,
            held_count=resolved.projection.held_count,
            update_subject=None
            if resolved.book.update_subject is None
            else resolved.book.update_subject.model_dump(mode="json"),
            experiment_subject=None
            if resolved.book.experiment_subject is None
            else resolved.book.experiment_subject.model_dump(mode="json"),
        )

    # --------------------------------------------------------------- internals

    def _publication_days(self) -> RecordDays | None:
        """The analysis publications' dated index, where the Evidence runtime keeps one."""
        adapter = self.evidence_task_adapter
        return (
            None if adapter is None else adapter.runtime.record_days[ANALYSIS_PUBLICATION_CATEGORY]
        )

    def documents(self, *, page: int | None = None) -> dict[str, object]:
        """The workspace's retained SEC documents, newest accepted first, a page at a time (A6).

        Every registered issuer's committed references through the local sources' own reader,
        one per accession, as the references state them (their bodies verified at use, as
        every reader of them verifies). Nothing is fetched, and no book is needed.

        Args:
            page: The page, from 1; the first when None.

        Returns:
            The page's documents, the page and the page count, the total, and the next page's
            request while one remains.

        Raises:
            PortfolioEvidenceReviewError: `alternative_evidence.documents_page_out_of_range`
                for a page past the last.
        """
        adapter = self.evidence_task_adapter
        if adapter is None:
            return {
                "status": "EVIDENCE_PREREQUISITES_MISSING",
                "failure_code": "product_host.evidence_task_adapter_absent",
                "next_action": "ADMIT_SOURCE_PACKAGE_AND_PINNED_RETRIEVAL_RUNTIME",
            }
        sources = adapter.runtime.local_sources
        rows = [
            {
                "entity_id": reference.entity_id,
                "cik": reference.source_cik,
                "document_type": reference.document_type,
                "revision": reference.revision,
                "title": reference.title,
                "accepted_at": None
                if reference.accepted_at is None
                else reference.accepted_at.isoformat(),
                "available_at": reference.available_at.isoformat(),
                "report_period_end": None
                if reference.report_period_end is None
                else reference.report_period_end.isoformat(),
                "content_bytes": reference.content_bytes,
                "retrieved_at": None
                if reference.retrieved_at is None
                else reference.retrieved_at.isoformat(),
            }
            for entity, cik in sorted(self._issuer_ciks().items())
            for reference in sources.holdings(entity_id=entity, cik=cik)
        ]
        rows.sort(
            key=lambda row: (str(row["accepted_at"] or ""), str(row["revision"])), reverse=True
        )
        page_count = max(1, math.ceil(len(rows) / _DOCUMENTS_PAGE))
        number = 1 if page is None else page
        if not 1 <= number <= page_count:
            raise PortfolioEvidenceReviewError(
                f"alternative_evidence.documents_page_out_of_range:{number} of {page_count}"
            )
        refused_documents: list[dict[str, object]] = []
        for category, identity, failure_code in sources.damaged_artifacts:
            words = refusal_words(f"{failure_code}:{identity}")
            refused_documents.append(
                {
                    "status": "REFUSED",
                    "artifact_kind": category,
                    "artifact_hash": identity,
                    "failure_code": failure_code,
                    "detail": words.get("detail")
                    or (
                        f"The retained Evidence commitment {identity} ({category}) could not be "
                        f"read ({failure_code}). Its record is refused; other readable "
                        "commitments remain listed. Inspect stored files and workspace backups "
                        "before relying on this commitment."
                    ),
                    "next_action": words.get("next_action"),
                    "next_requests": {
                        "documents": {
                            "operation": "EVIDENCE_DOCUMENTS",
                            "documents_page": number,
                        },
                        "storage": {"operation": "STORAGE_READBACK"},
                        "workspace": {"operation": "WORKSPACE_SHOW"},
                        "backups": {"operation": "WORKSPACE_BACKUPS"},
                    },
                }
            )
        answer: dict[str, object] = {
            "status": "EVIDENCE_DOCUMENTS",
            "documents": rows[(number - 1) * _DOCUMENTS_PAGE : number * _DOCUMENTS_PAGE],
            "page": number,
            "page_count": page_count,
            "total": len(rows),
            "retained_bytes": sum(int(str(row["content_bytes"])) for row in rows),
            "next_requests": {}
            if number == page_count
            else {"next": {"operation": "EVIDENCE_DOCUMENTS", "documents_page": number + 1}},
            "claim": (
                "What this workspace holds, from its sealed references; nothing was fetched, "
                "and no source's freshness is claimed."
            ),
        }
        if refused_documents:
            answer["refused_documents"] = refused_documents
        return answer

    def _issuer_ciks(self) -> dict[str, str]:
        """Each registered issuer's CIK by its entity id; none without a registry."""
        registry = self.registry
        return (
            {} if registry is None else {value.entity_id: value.cik for value in registry.entries}
        )

    def _require_evidence_publications(self) -> AlternativeEvidenceAnalysisPublicationService:
        if self.evidence_publications is None:
            raise PortfolioEvidenceReviewError("product_host.evidence_review_service_absent")
        return self.evidence_publications

    def _decision_policy(self) -> CROHostPolicyBinding:
        return build_portfolio_review_policy_binding(self.playpen_root)

    def _review_adapter(self, actor: PortfolioReviewActor) -> PortfolioReviewTaskAdapter:
        return PortfolioReviewTaskAdapter(
            registry=self.session.task_control_registry,
            artifacts=self.artifacts,
            resources=PortfolioReviewTaskResources(
                actor=actor,
                decision_policy=self._decision_policy(),
                playpen_root=self.playpen_root,
                publications=self.review_publications,
                evidence_publications=self._require_evidence_publications(),
                response_schema_hash=self.response_schema_hash,
                typed_user_authority=self.typed_user_authority,
                actor_kind=str(getattr(actor, "actor_kind", "INSTALLED_AGENT")),
                actor_id=str(getattr(actor, "actor_id", "installed-agent")),
                clock=self.clock,
                verify_update_subject=self._verify_update_subject,
                verify_experiment_subject=self._verify_experiment_subject,
                verify_external_review_context=self._verify_external_review_context,
            ),
        )

    def _verify_experiment_subject(self, subject: PortfolioExperimentReviewSubject) -> None:
        book = open_sealed_book(
            self.inputs(),
            BookSelector(
                experiment_task_id=subject.experiment_task_id,
                experiment_receipt_hash=subject.experiment_receipt_hash,
                portfolio_session=subject.portfolio_session.isoformat(),
            ),
        )
        if book.experiment_subject != subject:
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_experiment_subject_mismatch"
            )

    @staticmethod
    def _selector_for_dossier(dossier: PortfolioReviewDossier) -> BookSelector:
        if dossier.experiment_subject is not None:
            subject = dossier.experiment_subject
            selector = BookSelector(
                experiment_task_id=subject.experiment_task_id,
                experiment_receipt_hash=subject.experiment_receipt_hash,
                portfolio_session=subject.portfolio_session.isoformat(),
            )
        elif dossier.update_subject is not None:
            update = dossier.update_subject
            selector = BookSelector(
                update_task_id=update.update_task_id,
                update_publication_hash=update.update_publication_hash,
                position_basis=update.position_basis,
            )
        else:
            selector = BookSelector(
                handoff_hash=dossier.handoff_hash,
                result_hash=None if dossier.handoff_hash else dossier.result_hash,
            )
        return selector

    def _verify_external_review_context(self, dossier: PortfolioReviewDossier) -> None:
        current = self._resolve_review_dossier(
            self._selector_for_dossier(dossier), read_at=dossier.evidence_as_of
        )
        if (
            not isinstance(current, PortfolioReviewDossier)
            or current.dossier_hash != dossier.dossier_hash
        ):
            raise PortfolioEvidenceReviewError("chief_risk_officer.external_review_context_changed")

    def _verify_update_subject(self, subject: PortfolioUpdateReviewSubject) -> None:
        book = open_sealed_book(
            self.inputs(),
            BookSelector(
                update_task_id=subject.update_task_id,
                update_publication_hash=subject.update_publication_hash,
                position_basis=subject.position_basis,
            ),
        )
        if book.update_subject != subject:
            raise PortfolioEvidenceReviewError(
                "product_host.evidence_review_update_subject_mismatch"
            )

    def _run(
        self,
        adapter: object,
        *,
        task_id: UUID | None = None,
        expected_task_hash: str | None = None,
    ) -> None:
        """Run the named Task; only legacy callers without an id ask for the queue."""

        runner = TaskControlRunner(
            registry=self.session.task_control_registry,
            adapters={adapter.task_kind: adapter},  # type: ignore[attr-defined]
            runtime_path=str(self.runner_runtime_path()),
            clock=self.clock,
        )
        try:
            if task_id is None:
                runner.run_next()
            else:
                task = self.session.task_control_registry.task(task_id)
                if task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED:
                    runner.recover(task_id, expected_task_hash=expected_task_hash)
                elif task.lifecycle is TaskLifecycle.QUEUED:
                    runner.run_next(expected_task_id=task_id, expected_task_hash=expected_task_hash)
        finally:
            runner.close()

    def recovery_commands(self, *, owed_only: bool = True) -> dict[str, object]:
        """Build recorded recovery commands; admission needs complete Task authority.

        Read-only compatibility views inspect readable peers without admitting a command.
        Actual recovery keeps the default complete-or-refused scan.
        """
        commands: dict[str, object] = {}
        registry = self.session.task_control_registry
        tasks = registry.tasks() if owed_only else registry.record_collection().records
        for task in tasks:
            if task.lifecycle not in {TaskLifecycle.RECOVERY_REQUIRED, TaskLifecycle.QUEUED}:
                continue
            kind = task.task_kind
            if (
                kind == AlternativeEvidenceDocumentTaskAdapter.task_kind
                and self.evidence_task_adapter is not None
            ):
                commands[kind] = AlternativeEvidenceRefreshCommand.recover(
                    application=self, task=task
                )
            elif kind == PortfolioReviewTaskAdapter.task_kind and (
                self.review_actor is not None or "prepared_submission" in task.input.payload
            ):
                commands[kind] = PortfolioReviewCommand(application=self)
        return commands

    def runner_runtime_path(self) -> Path:
        """Name this application's operational runner heartbeat sidecar.

        The runtime file whose name names this application's runners' operational
        heartbeat sidecar, for read-only liveness readers.
        """
        if self.runtime_path is not None:
            return self.runtime_path
        return self.workspace / "runtime" / "evidence-review-checkpoints.sqlite"


__all__ = [
    "CRO_REVIEW_COMMAND",
    "EVIDENCE_REFRESH_COMMAND",
    "AlternativeEvidenceRefreshCommand",
    "EvidenceReviewApplication",
    "EvidenceSelectionAmbiguous",
    "PortfolioReviewCommand",
    "ResolvedBookScope",
    "ResolvedCoverage",
    "ResolvedEvidenceQuestion",
    "ReviewOutcome",
]
